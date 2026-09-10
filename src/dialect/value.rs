//! JSON plus the one thing JSON does not have: an opaque tensor, which can live inside an array
//! (`tb.stack` takes a list of them) or an object (an `in` program answers `{input_name: tensor}`).
//! That is why this exists rather than `serde_json::Value` being used directly.
//!
//! Objects keep insertion order and are backed by a `Vec`: adapters build objects with a handful of
//! keys, so linear lookup beats a map and the ordering cannot differ between two machines.

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

    /// JSONLogic truthiness: empty string, empty array, `0` and `null` are falsy; an empty object
    /// is truthy, which is the one people get wrong. A tensor exists, so it is truthy.
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

    /// `null` is 0, `true` is 1, a numeric string is its number; anything else has no number and
    /// the caller decides what that means.
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

    /// Loose equality — `==`, JavaScript's. The rule that matters: `null` equals only `null`, not
    /// `0` or `""`. `datalogic-rs` answers true for `{"==": [0, null]}`, which makes a comparison
    /// against an unresolvable path select the falsy elements and look right for as long as the
    /// other side is zero. `tests/differential.rs` pins the divergence.
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

    /// Back to JSON. A tensor has no JSON form and becomes `null`, unreachable for a program's
    /// result because both directions are checked before they are handed on.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            // An integral value renders as a JSON integer, not `3.0`: the action is canonically
            // serialized for the determinism audit, and RFC 8785 writes an integral number without
            // a fraction, so `20.0` where a game expects `20` hashes differently.
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
