//! The evaluator and the operation count — docs/dialect.md §4:
//!
//!   1. every node evaluated costs 1 — applied, not written
//!   2. every tensor operator costs `1 + max(elements read, elements produced)`
//!   3. literals cost 1 however large
//!   4. the two directions are counted separately
//!   5. over budget aborts immediately, mid-operator, and is never retried
//!
//! Rule 5 is why the budget lives on the context and is checked inside the tensor operators' own
//! loops: a bound enforced after the work bounds the report, not the work.

use std::sync::Arc;

use super::ops;
use super::value::{fmt_num, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum Fault {
    /// The program spent its budget. Never retried.
    OverBudget { budget: u64 },
    /// Not in the dialect — an unknown operator, a bad arity, a wrong type.
    Invalid(String),
}

impl Fault {
    pub fn invalid(msg: impl Into<String>) -> Fault {
        Fault::Invalid(msg.into())
    }
    pub fn code(&self) -> &'static str {
        match self {
            Fault::OverBudget { .. } => "ADAPTER_FAILED",
            Fault::Invalid(_) => "ADAPTER_INVALID",
        }
    }
    pub fn over_budget(&self) -> bool {
        matches!(self, Fault::OverBudget { .. })
    }
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fault::OverBudget { budget } => write!(f, "over the operation budget of {budget}"),
            Fault::Invalid(m) => write!(f, "{m}"),
        }
    }
}

pub type Res<T> = Result<T, Fault>;

pub struct Ctx {
    pub ops: u64,
    pub budget: u64,
    depth: u32,
}

/// Far above any adapter anyone would write and far below what exhausts the stack. A program that
/// reaches it is malformed rather than expensive, so it is `Invalid` and not `OverBudget`.
const MAX_DEPTH: u32 = 64;

impl Ctx {
    pub fn new(budget: u64) -> Ctx {
        Ctx { ops: 0, budget, depth: 0 }
    }

    /// The only place the budget is enforced.
    #[inline]
    pub fn charge(&mut self, n: u64) -> Res<()> {
        self.ops = self.ops.saturating_add(n);
        if self.ops > self.budget {
            Err(Fault::OverBudget { budget: self.budget })
        } else {
            Ok(())
        }
    }
}

/// Evaluate a program against a document.
pub fn run(program: &serde_json::Value, data: &Value, budget: u64) -> (Res<Value>, u64) {
    let mut ctx = Ctx::new(budget);
    let out = eval(program, data, &mut ctx);
    (out, ctx.ops)
}

/// One node. Costs 1 before anything else happens, so nested literals pay for their own size.
pub fn eval(node: &serde_json::Value, data: &Value, ctx: &mut Ctx) -> Res<Value> {
    ctx.charge(1)?;
    ctx.depth += 1;
    if ctx.depth > MAX_DEPTH {
        ctx.depth -= 1;
        return Err(Fault::invalid(format!("expression nested deeper than {MAX_DEPTH}")));
    }
    let out = eval_inner(node, data, ctx);
    ctx.depth -= 1;
    out
}

fn eval_inner(node: &serde_json::Value, data: &Value, ctx: &mut Ctx) -> Res<Value> {
    let obj = match node {
        // An array in expression position is an array of expressions.
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(eval(it, data, ctx)?);
            }
            return Ok(Value::arr(out));
        }
        serde_json::Value::Object(o) => o,
        // Scalars are themselves.
        other => return Ok(Value::from_json(other)),
    };

    // The object rule — docs/dialect.md §1: an object whose single key is a known operator is an
    // operation; any other object is a literal whose values are evaluated and whose keys are not.
    if obj.len() == 1 {
        let (op, arg) = obj.iter().next().unwrap();
        if is_operator(op) {
            return apply(op, arg, data, ctx);
        }
        // The `tb.` namespace is reserved, so `{"tb.scattr": ...}` is a refusal here rather than
        // an object handed to the graph as an input that is not a tensor.
        if op.starts_with("tb.") {
            return Err(Fault::invalid(format!(
                "no operator '{op}' in this dialect; the 'tb.' prefix is reserved for it"
            )));
        }
    }
    let mut out = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        out.push((Arc::from(&**k), eval(v, data, ctx)?));
    }
    Ok(Value::obj(out))
}

/// An operator's arguments, which JSONLogic allows to be written bare when there is one of them.
fn args_of(arg: &serde_json::Value) -> Vec<&serde_json::Value> {
    match arg {
        serde_json::Value::Array(a) => a.iter().collect(),
        other => vec![other],
    }
}

fn eval_args(arg: &serde_json::Value, data: &Value, ctx: &mut Ctx) -> Res<Vec<Value>> {
    args_of(arg).into_iter().map(|a| eval(a, data, ctx)).collect()
}

pub fn is_operator(name: &str) -> bool {
    CORE.contains(&name) || ops::is_tensor_op(name)
}

/// The core operator set: a list, not "whatever JSONLogic has", because this surface is what
/// `dialect_version` versions and what the re-validation sweep is defined against.
#[rustfmt::skip]
pub const CORE: &[&str] = &[
    "var", "val", "missing", "missing_some", "if", "?:", "==", "===", "!=", "!==", "!", "!!",
    "and", "or", ">", ">=", "<", "<=", "+", "-", "*", "/", "%", "max", "min", "cat", "substr",
    "in", "merge", "map", "filter", "reduce", "all", "some", "none", "length", "slice", "sort",
    "distinct", "keys", "values", "entries", "??", "type", "abs", "ceil", "floor",
];

fn apply(op: &str, arg: &serde_json::Value, data: &Value, ctx: &mut Ctx) -> Res<Value> {
    match op {
        // ---------------------------------------------------------------- access
        "var" | "val" => {
            let a = args_of(arg);
            let path = if a.is_empty() { Value::Null } else { eval(a[0], data, ctx)? };
            let found = lookup(data, &path);
            match found {
                Some(v) => Ok(v),
                None if a.len() > 1 => eval(a[1], data, ctx),
                None => Ok(Value::Null),
            }
        }
        "missing" => {
            let a = eval_args(arg, data, ctx)?;
            let names: Vec<&Value> = if a.len() == 1 {
                a[0].as_arr().map(|s| s.iter().collect()).unwrap_or_else(|| a.iter().collect())
            } else {
                a.iter().collect()
            };
            let mut out = Vec::new();
            for n in names {
                if lookup(data, n).map(|v| matches!(v, Value::Null)).unwrap_or(true) {
                    out.push(n.clone());
                }
            }
            Ok(Value::arr(out))
        }
        "missing_some" => {
            let a = eval_args(arg, data, ctx)?;
            let need = a.first().and_then(Value::to_num).unwrap_or(0.0) as usize;
            let names = a.get(1).and_then(Value::as_arr).unwrap_or(&[]).to_vec();
            let mut missing = Vec::new();
            let mut have = 0usize;
            for n in &names {
                if lookup(data, n).map(|v| matches!(v, Value::Null)).unwrap_or(true) {
                    missing.push(n.clone());
                } else {
                    have += 1;
                }
            }
            Ok(if have >= need { Value::arr(vec![]) } else { Value::arr(missing) })
        }

        // ---------------------------------------------------------------- control
        "if" | "?:" => {
            let a = args_of(arg);
            if a.is_empty() {
                return Ok(Value::Null);
            }
            let mut i = 0;
            while i + 1 < a.len() {
                if eval(a[i], data, ctx)?.truthy() {
                    return eval(a[i + 1], data, ctx);
                }
                i += 2;
            }
            if i < a.len() {
                eval(a[i], data, ctx)
            } else {
                Ok(Value::Null)
            }
        }
        "and" => {
            let a = args_of(arg);
            let mut last = Value::Bool(true);
            for e in a {
                last = eval(e, data, ctx)?;
                if !last.truthy() {
                    return Ok(last);
                }
            }
            Ok(last)
        }
        "or" => {
            let a = args_of(arg);
            let mut last = Value::Bool(false);
            for e in a {
                last = eval(e, data, ctx)?;
                if last.truthy() {
                    return Ok(last);
                }
            }
            Ok(last)
        }
        "??" => {
            let a = args_of(arg);
            for e in a {
                let v = eval(e, data, ctx)?;
                if !matches!(v, Value::Null) {
                    return Ok(v);
                }
            }
            Ok(Value::Null)
        }
        "!" => Ok(Value::Bool(!first(arg, data, ctx)?.truthy())),
        "!!" => Ok(Value::Bool(first(arg, data, ctx)?.truthy())),

        // ---------------------------------------------------------------- comparison
        "==" => bin(arg, data, ctx, |x, y| Value::Bool(x.loose_eq(y))),
        "!=" => bin(arg, data, ctx, |x, y| Value::Bool(!x.loose_eq(y))),
        "===" => bin(arg, data, ctx, |x, y| Value::Bool(x.strict_eq(y))),
        "!==" => bin(arg, data, ctx, |x, y| Value::Bool(!x.strict_eq(y))),
        ">" | ">=" | "<" | "<=" => {
            let a = eval_args(arg, data, ctx)?;
            if a.len() < 2 {
                return Err(Fault::invalid(format!("'{op}' needs two arguments")));
            }
            let cmp = |x: &Value, y: &Value| -> bool {
                match (x, y) {
                    (Value::Str(p), Value::Str(q)) => match op {
                        ">" => p > q,
                        ">=" => p >= q,
                        "<" => p < q,
                        _ => p <= q,
                    },
                    _ => match (x.to_num(), y.to_num()) {
                        (Some(p), Some(q)) => match op {
                            ">" => p > q,
                            ">=" => p >= q,
                            "<" => p < q,
                            _ => p <= q,
                        },
                        _ => false,
                    },
                }
            };
            // Three arguments is the between form: a < b < c.
            let ok = if a.len() >= 3 {
                cmp(&a[0], &a[1]) && cmp(&a[1], &a[2])
            } else {
                cmp(&a[0], &a[1])
            };
            Ok(Value::Bool(ok))
        }

        // ---------------------------------------------------------------- arithmetic
        "+" | "*" => {
            let a = eval_args(arg, data, ctx)?;
            let init = if op == "+" { 0.0 } else { 1.0 };
            let mut acc = init;
            for v in &a {
                let n = v.to_num().ok_or_else(|| {
                    Fault::invalid(format!(
                        "'{op}' got a {} where a number was needed",
                        v.type_name()
                    ))
                })?;
                acc = if op == "+" { acc + n } else { acc * n };
            }
            Ok(Value::Num(acc))
        }
        "-" => {
            let a = eval_args(arg, data, ctx)?;
            let nums = to_nums(&a, op)?;
            Ok(Value::Num(match nums.len() {
                0 => 0.0,
                1 => -nums[0],
                _ => nums[1..].iter().fold(nums[0], |x, y| x - y),
            }))
        }
        "/" | "%" => {
            let a = eval_args(arg, data, ctx)?;
            let nums = to_nums(&a, op)?;
            if nums.len() < 2 {
                return Err(Fault::invalid(format!("'{op}' needs two arguments")));
            }
            // Not a fault: dividing by a count that happens to be zero on one observation should
            // not refuse the turn. `null` propagates and `??` says what to do about it.
            if nums[1] == 0.0 {
                return Ok(Value::Null);
            }
            Ok(Value::Num(if op == "/" { nums[0] / nums[1] } else { nums[0] % nums[1] }))
        }
        "max" | "min" => {
            let a = eval_args(arg, data, ctx)?;
            let nums = to_nums(&a, op)?;
            if nums.is_empty() {
                return Ok(Value::Null);
            }
            Ok(Value::Num(nums.iter().copied().fold(nums[0], |x, y| {
                if (op == "max") == (y > x) {
                    y
                } else {
                    x
                }
            })))
        }
        "abs" => Ok(Value::Num(num1(arg, data, ctx, op)?.abs())),
        "ceil" => Ok(Value::Num(num1(arg, data, ctx, op)?.ceil())),
        "floor" => Ok(Value::Num(num1(arg, data, ctx, op)?.floor())),

        // ---------------------------------------------------------------- strings
        "cat" => {
            let a = eval_args(arg, data, ctx)?;
            let mut s = String::new();
            for v in &a {
                s.push_str(&v.to_string_val());
            }
            Ok(Value::str(s))
        }
        "substr" => {
            let a = eval_args(arg, data, ctx)?;
            let s = a.first().map(Value::to_string_val).unwrap_or_default();
            let chars: Vec<char> = s.chars().collect();
            let n = chars.len() as i64;
            let start = a.get(1).and_then(Value::to_num).unwrap_or(0.0) as i64;
            let start = if start < 0 { (n + start).max(0) } else { start.min(n) } as usize;
            let end = match a.get(2).and_then(Value::to_num) {
                Some(l) if l < 0.0 => ((n + l as i64).max(0) as usize).max(start),
                Some(l) => (start + l as usize).min(chars.len()),
                None => chars.len(),
            };
            Ok(Value::str(chars[start..end.min(chars.len())].iter().collect::<String>()))
        }
        "type" => Ok(Value::str(first(arg, data, ctx)?.type_name())),

        // ---------------------------------------------------------------- arrays
        "in" => {
            let a = eval_args(arg, data, ctx)?;
            if a.len() < 2 {
                return Ok(Value::Bool(false));
            }
            Ok(Value::Bool(match &a[1] {
                Value::Arr(items) => items.iter().any(|v| v.loose_eq(&a[0])),
                Value::Str(s) => s.contains(&a[0].to_string_val()),
                _ => false,
            }))
        }
        "merge" => {
            let a = eval_args(arg, data, ctx)?;
            let mut out = Vec::new();
            for v in a {
                match v {
                    Value::Arr(items) => out.extend(items.iter().cloned()),
                    other => out.push(other),
                }
            }
            Ok(Value::arr(out))
        }
        "length" => {
            let v = first(arg, data, ctx)?;
            Ok(Value::Num(match &v {
                Value::Arr(a) => a.len() as f64,
                Value::Str(s) => s.chars().count() as f64,
                Value::Obj(o) => o.len() as f64,
                _ => 0.0,
            }))
        }
        "slice" => {
            let a = eval_args(arg, data, ctx)?;
            let items = a.first().and_then(Value::as_arr).unwrap_or(&[]);
            let n = items.len() as i64;
            let idx = |v: Option<&Value>, dflt: i64| -> usize {
                let x = v.and_then(Value::to_num).map(|f| f as i64).unwrap_or(dflt);
                (if x < 0 { n + x } else { x }).clamp(0, n) as usize
            };
            let start = idx(a.get(1), 0);
            let end = idx(a.get(2), n).max(start);
            Ok(Value::arr(items[start..end].to_vec()))
        }
        "sort" => {
            let a = eval_args(arg, data, ctx)?;
            let mut items = a.first().and_then(Value::as_arr).unwrap_or(&[]).to_vec();
            // Total, and stable, so two machines agree on ties.
            items.sort_by(|x, y| match (x.to_num(), y.to_num()) {
                (Some(p), Some(q)) => p.partial_cmp(&q).unwrap_or(std::cmp::Ordering::Equal),
                _ => x.to_string_val().cmp(&y.to_string_val()),
            });
            Ok(Value::arr(items))
        }
        "distinct" => {
            let a = args_of(arg);
            let Some(src) = a.first() else { return Ok(Value::arr(vec![])) };
            let items = eval(src, data, ctx)?;
            let items = items.as_arr().unwrap_or(&[]).to_vec();
            let key_expr = a.get(1);
            let mut seen: Vec<Value> = Vec::new();
            let mut out = Vec::new();
            for it in items {
                let k = match key_expr {
                    Some(e) => eval(e, &it, ctx)?,
                    None => it.clone(),
                };
                if !seen.iter().any(|s| s.strict_eq(&k)) {
                    seen.push(k);
                    out.push(it);
                }
            }
            Ok(Value::arr(out))
        }
        "keys" | "values" | "entries" => {
            let v = first(arg, data, ctx)?;
            let pairs = match &v {
                Value::Obj(o) => o.as_slice(),
                _ => &[],
            };
            Ok(Value::arr(match op {
                "keys" => pairs.iter().map(|(k, _)| Value::Str(k.clone())).collect(),
                "values" => pairs.iter().map(|(_, v)| v.clone()).collect(),
                _ => pairs
                    .iter()
                    .map(|(k, v)| {
                        Value::obj(vec![
                            (Arc::from("key"), Value::Str(k.clone())),
                            (Arc::from("value"), v.clone()),
                        ])
                    })
                    .collect(),
            }))
        }

        // ---------------------------------------------------------------- iteration
        //
        // A body is evaluated with the ELEMENT as its data: there is no path from inside one to
        // the outer document. `reduce`'s seed, evaluated in the outer scope, is the only channel
        // in, and the platform's own workflows depend on that asymmetry.
        "map" | "filter" | "all" | "some" | "none" => {
            let a = args_of(arg);
            if a.len() < 2 {
                return Err(Fault::invalid(format!("'{op}' needs an array and a body")));
            }
            let src = eval(a[0], data, ctx)?;
            let items = src.as_arr().unwrap_or(&[]).to_vec();
            match op {
                "map" => {
                    let mut out = Vec::with_capacity(items.len());
                    for it in &items {
                        out.push(eval(a[1], it, ctx)?);
                    }
                    Ok(Value::arr(out))
                }
                "filter" => {
                    let mut out = Vec::new();
                    for it in &items {
                        if eval(a[1], it, ctx)?.truthy() {
                            out.push(it.clone());
                        }
                    }
                    Ok(Value::arr(out))
                }
                _ => {
                    // `all` over an empty array is false, matching JSONLogic rather than logic.
                    if items.is_empty() {
                        return Ok(Value::Bool(op == "none"));
                    }
                    let mut any = false;
                    let mut every = true;
                    for it in &items {
                        if eval(a[1], it, ctx)?.truthy() {
                            any = true;
                            if op == "some" {
                                break;
                            }
                        } else {
                            every = false;
                            if op == "all" {
                                break;
                            }
                        }
                    }
                    Ok(Value::Bool(match op {
                        "all" => every,
                        "some" => any,
                        _ => !any,
                    }))
                }
            }
        }
        "reduce" => {
            let a = args_of(arg);
            if a.len() < 2 {
                return Err(Fault::invalid("'reduce' needs an array and a body"));
            }
            let src = eval(a[0], data, ctx)?;
            let items = src.as_arr().unwrap_or(&[]).to_vec();
            let mut acc = match a.get(2) {
                Some(e) => eval(e, data, ctx)?,
                None => Value::Null,
            };
            for it in &items {
                let scope = Value::obj(vec![
                    (Arc::from("current"), it.clone()),
                    (Arc::from("accumulator"), acc.clone()),
                ]);
                acc = eval(a[1], &scope, ctx)?;
            }
            Ok(acc)
        }

        // ---------------------------------------------------------------- tensors
        other => {
            let a = eval_args(arg, data, ctx)?;
            ops::apply(other, &a, ctx)
        }
    }
}

// ------------------------------------------------------------------ helpers

fn first(arg: &serde_json::Value, data: &Value, ctx: &mut Ctx) -> Res<Value> {
    let a = args_of(arg);
    match a.first() {
        Some(e) => eval(e, data, ctx),
        None => Ok(Value::Null),
    }
}

fn num1(arg: &serde_json::Value, data: &Value, ctx: &mut Ctx, op: &str) -> Res<f64> {
    first(arg, data, ctx)?.to_num().ok_or_else(|| Fault::invalid(format!("'{op}' needs a number")))
}

fn to_nums(a: &[Value], op: &str) -> Res<Vec<f64>> {
    a.iter()
        .map(|v| {
            v.to_num().ok_or_else(|| {
                Fault::invalid(format!("'{op}' got a {} where a number was needed", v.type_name()))
            })
        })
        .collect()
}

fn bin(
    arg: &serde_json::Value,
    data: &Value,
    ctx: &mut Ctx,
    f: impl Fn(&Value, &Value) -> Value,
) -> Res<Value> {
    let a = eval_args(arg, data, ctx)?;
    let x = a.first().cloned().unwrap_or(Value::Null);
    let y = a.get(1).cloned().unwrap_or(Value::Null);
    Ok(f(&x, &y))
}

/// `var`/`val` path resolution: a string splits on `.`, a number indexes, an array is a chain of
/// already-evaluated segments, and an empty path is the document itself.
pub fn lookup(data: &Value, path: &Value) -> Option<Value> {
    match path {
        Value::Null => Some(data.clone()),
        Value::Str(s) if s.is_empty() => Some(data.clone()),
        Value::Str(s) => {
            let mut cur = data.clone();
            for seg in s.split('.') {
                cur = step(&cur, seg)?;
            }
            Some(cur)
        }
        Value::Num(n) => step(data, &fmt_num(*n)),
        Value::Arr(segs) => {
            let mut cur = data.clone();
            for seg in segs.iter() {
                cur = step(&cur, &seg.to_string_val())?;
            }
            Some(cur)
        }
        _ => None,
    }
}

fn step(cur: &Value, seg: &str) -> Option<Value> {
    match cur {
        Value::Obj(_) => cur.get(seg).cloned(),
        Value::Arr(a) => seg.parse::<usize>().ok().and_then(|i| a.get(i).cloned()),
        // The only two things a program may ask a tensor.
        Value::Tensor(t) => match seg {
            "shape" => Some(Value::arr(t.shape.iter().map(|&n| Value::Num(n as f64)).collect())),
            "dtype" => Some(Value::str(t.dtype.name())),
            _ => None,
        },
        _ => None,
    }
}
