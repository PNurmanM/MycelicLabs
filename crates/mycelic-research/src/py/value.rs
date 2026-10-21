//! JSON values with Python semantics: the result rows are read back as `json.loads` gave them to the Python
//! analysis, so ordering, `str()` and truthiness follow Python's rules for int/float/bool/str/None.

use super::fmt;
use serde_json::{Map, Value};
use std::cmp::Ordering;

pub type Row = Map<String, Value>;

/// The numeric value of a bool/int/float (Python treats `True` as 1).
pub fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// `row[key]` as a float (panics like a KeyError when absent).
pub fn f(row: &Row, key: &str) -> f64 {
    row.get(key).and_then(num).unwrap_or_else(|| panic!("row has no numeric {key:?}: {row:?}"))
}

/// Python's `<` / `==` ordering for the value kinds the rows contain.  Numbers (incl. bools) compare by value,
/// strings lexicographically by code point, lists element by element; mixing incomparable kinds panics as
/// Python raises TypeError.
pub fn py_cmp(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::String(x), Value::String(y)) => x.as_str().cmp(y.as_str()),
        (Value::Array(x), Value::Array(y)) => {
            for (p, q) in x.iter().zip(y) {
                let o = py_cmp(p, q);
                if o != Ordering::Equal {
                    return o;
                }
            }
            x.len().cmp(&y.len())
        }
        (Value::Null, Value::Null) => Ordering::Equal,
        _ => match (num(a), num(b)) {
            (Some(x), Some(y)) => {
                match (a.as_i64().or(a.as_bool().map(i64::from)), b.as_i64().or(b.as_bool().map(i64::from))) {
                    (Some(i), Some(j)) => i.cmp(&j),
                    _ => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
                }
            }
            _ => panic!("cannot order {a} and {b}"),
        },
    }
}

/// Python equality between two row values (1 == 1.0 == True).
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (num(a), num(b)) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

/// `str(v)` for a value loaded from JSON.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                fmt::repr(n.as_f64().unwrap())
            }
        }
        Value::String(s) => s.clone(),
        Value::Array(_) | Value::Object(_) => py_repr(v),
    }
}

/// `repr(v)` for a value loaded from JSON (lists and dicts print their items' reprs).
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => super::json::repr_str(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, x)| format!("{}: {}", super::json::repr_str(k), py_repr(x)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => py_str(other),
    }
}

/// A value is a Python float (as opposed to an int or bool) - what `isinstance(v, float)` asks.
pub fn is_float(v: &Value) -> bool {
    matches!(v, Value::Number(n) if n.is_f64())
}

/// Python truthiness.
pub fn truthy(v: &Value) -> bool {
    super::json::truthy(v)
}

/// A float for a row (a Python float stays a float in the JSON it is written to).
pub fn fl(x: f64) -> Value {
    serde_json::Number::from_f64(x).map(Value::Number).unwrap_or(Value::Null)
}
