//! The dialect's value model: JSON, plus one thing JSON does not have.
//!
//! A program in this dialect works over ordinary JSON — the observation going in, the action
//! coming out — with exactly one addition: a **tensor**, which is opaque. A program can produce
//! one, pass it to another operator, and ask its shape or its dtype. It cannot see an element.
//!
//! That is layer 04 §4.2, and it is why this file exists rather than `serde_json::Value` being
//! used directly: a tensor has to be able to live inside an array (`tb.stack` takes a list of
//! them) and inside an object (an `in` program's result is `{input_name: tensor}`), and
//! `serde_json::Value` has no room for one.
//!
//! Objects preserve insertion order and are backed by a `Vec`. Adapters build objects with a
//! handful of keys — the graph's input names — so linear lookup is faster than a map and the
//! ordering is one less thing that can differ between two machines.

use std::fmt;
use std::sync::Arc;

pub use crate::dialect::tensor::{DType, Tensor};

#[derive(Clone)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(Arc<str>),
    Arr(Arc<Vec<Value>>),
    Obj(Arc<Vec<(Arc<str>, Value)>>),
    Tensor(Arc<Tensor>),
}

impl Value {
    pub fn str(s: impl AsRef<str>) -> Value {
        Value::Str(Arc::from(s.as_ref()))
    }
    pub fn arr(v: Vec<Value>) -> Value {
        Value::Arr(Arc::new(v))
    }
    pub fn obj(v: Vec<(Arc<str>, Value)>) -> Value {
        Value::Obj(Arc::new(v))
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Num(_) => "number",
            Value::Str(_) => "string",
            Value::Arr(_) => "array",
            Value::Obj(_) => "object",
            Value::Tensor(_) => "tensor",
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(m) => m.iter().find(|(k, _)| &**k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn index(&self, i: usize) -> Option<&Value> {
        match self {
            Value::Arr(a) => a.get(i),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_tensor(&self) -> Option<&Arc<Tensor>> {
        match self {
            Value::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// JSONLogic truthiness, and it is not Rust's. Empty string, empty array, `0` and `null` are
    /// all falsy; an empty **object** is truthy, which is the one people get wrong. A tensor is
    /// always truthy — it exists.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Num(n) => *n != 0.0 && !n.is_nan(),
            Value::Str(s) => !s.is_empty(),
            Value::Arr(a) => !a.is_empty(),
            Value::Obj(_) => true,
            Value::Tensor(_) => true,
        }
    }

    /// Numeric coercion for arithmetic and comparison. `null` is 0, `true` is 1, a numeric string
    /// is its number; anything else has no number and the caller decides what that means.
    pub fn to_num(&self) -> Option<f64> {
        match self {
            Value::Null => Some(0.0),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Num(n) => Some(*n),
            Value::Str(s) => {
                let t = s.trim();
                if t.is_empty() {
                    Some(0.0)
                } else {
                    t.parse::<f64>().ok()
                }
            }
            _ => None,
        }
    }

    pub fn to_string_val(&self) -> String {
        match self {
            Value::Null => String::new(),
            Value::Bool(b) => b.to_string(),
            Value::Num(n) => fmt_num(*n),
            Value::Str(s) => s.to_string(),
            Value::Arr(a) => a.iter().map(|v| v.to_string_val()).collect::<Vec<_>>().join(","),
            Value::Obj(_) => "[object Object]".into(),
            Value::Tensor(t) => format!("[tensor {:?} {}]", t.shape, t.dtype.name()),
        }
    }

    /// Strict equality — `===`. No coercion, and types must match.
    pub fn strict_eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Num(a), Value::Num(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Arr(a), Value::Arr(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.strict_eq(y))
            }
            (Value::Obj(a), Value::Obj(b)) => {
                a.len() == b.len()
                    && a.iter().all(|(k, v)| {
                        b.iter().find(|(k2, _)| k2 == k).is_some_and(|(_, v2)| v.strict_eq(v2))
                    })
            }
            (Value::Tensor(a), Value::Tensor(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Loose equality — `==`. JSONLogic's, which is JavaScript's, which is a minefield.
    ///
    /// The one that matters here is that **`null` equals only `null` and `undefined`** — it does
    /// NOT equal `0` or `""`. The spike found an engine where `{"==": [0, null]}` was true, and a
    /// join written against an unresolvable path therefore selected the falsy elements and looked
    /// correct for as long as the value it was compared against was zero
    /// (`03-spike/FINDINGS.md` §2.6). A competitor's adapter is exactly the place that trap would
    /// be discovered by accident and never diagnosed, so this dialect follows JavaScript instead.
    pub fn loose_eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Null, _) | (_, Value::Null) => false,
            (Value::Arr(_), _) | (_, Value::Arr(_)) | (Value::Obj(_), _) | (_, Value::Obj(_)) => {
                self.strict_eq(other)
            }
            (Value::Tensor(_), _) | (_, Value::Tensor(_)) => self.strict_eq(other),
            (Value::Str(a), Value::Str(b)) => a == b,
            _ => match (self.to_num(), other.to_num()) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            },
        }
    }
}

/// Numbers render the way JSON expects: an integral f64 is `3`, not `3.0`.
pub fn fmt_num(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Tensor(t) => write!(f, "<tensor {:?} {}>", t.shape, t.dtype.name()),
            other => write!(f, "{}", other.to_json_string()),
        }
    }
}

// ------------------------------------------------------------------ JSON bridge

impl Value {
    pub fn from_json(j: &serde_json::Value) -> Value {
        match j {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => Value::Num(n.as_f64().unwrap_or(f64::NAN)),
            serde_json::Value::String(s) => Value::str(s),
            serde_json::Value::Array(a) => Value::arr(a.iter().map(Value::from_json).collect()),
            serde_json::Value::Object(o) => {
                Value::obj(o.iter().map(|(k, v)| (Arc::from(&**k), Value::from_json(v))).collect())
            }
        }
    }

    /// Back to JSON. A tensor has no JSON form and becomes `null` — which is unreachable for a
    /// program's *result*, because both directions are checked before they are handed on: an `in`
    /// program must produce tensors and an `out` program must not.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            // An integral value renders as a JSON integer, not `3.0`. This is not cosmetic: the
            // action is canonically serialized for the determinism audit (PROTOCOL.md §3 rule 2)
            // and RFC 8785 renders an integral number without a fraction, so emitting `20.0` where
            // a game expects `20` is a different document with a different hash. Caught by the
            // differential test, which is exactly the kind of thing it is for.
            Value::Num(n) => {
                if n.is_finite() && n.fract() == 0.0 && n.abs() < 9.007_199_254_740_992e15 {
                    serde_json::Value::Number(serde_json::Number::from(*n as i64))
                } else {
                    serde_json::Number::from_f64(*n)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                }
            }
            Value::Str(s) => serde_json::Value::String(s.to_string()),
            Value::Arr(a) => serde_json::Value::Array(a.iter().map(Value::to_json).collect()),
            Value::Obj(o) => serde_json::Value::Object(
                o.iter().map(|(k, v)| (k.to_string(), v.to_json())).collect(),
            ),
            Value::Tensor(_) => serde_json::Value::Null,
        }
    }

    pub fn to_json_string(&self) -> String {
        self.to_json().to_string()
    }
}
