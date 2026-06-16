//! Typed scalar values and multi-valued fields carried by a payload.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// A single typed scalar held by a payload field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// UTF-8 text.
    Text(String),
    /// Signed 64-bit integer.
    Integer(i64),
    /// 64-bit floating point.
    Float(f64),
    /// Boolean.
    Bool(bool),
}

impl Value {
    /// Numeric view used for ordered comparisons, `None` for non-numeric values.
    #[allow(clippy::cast_precision_loss)]
    fn as_f64(&self) -> Option<f64> {
        match self {
            // Integer -> f64 loses precision past 2^53; acceptable for filter ordering.
            Value::Integer(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns whether this is a numeric value (`Integer` or `Float`).
    fn is_numeric(&self) -> bool {
        matches!(self, Value::Integer(_) | Value::Float(_))
    }

    /// Total-ish ordering for filter comparisons.
    ///
    /// Integers compare exactly, mixed numeric compares through `f64`, text
    /// compares lexicographically. Any other pairing (type mismatch, bool,
    /// non-finite float) yields `None`, which callers treat as "does not match".
    #[must_use]
    pub fn ordered(&self, other: &Value) -> Option<Ordering> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Some(a.cmp(b)),
            (Value::Text(a), Value::Text(b)) => Some(a.cmp(b)),
            (a, b) if a.is_numeric() && b.is_numeric() => a.as_f64()?.partial_cmp(&b.as_f64()?),
            _ => None,
        }
    }
}

/// A payload field: a single scalar, or an array of scalars (multi-value / tags).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    /// One scalar value.
    Scalar(Value),
    /// Several scalar values (e.g. tags).
    Array(Vec<Value>),
}

#[cfg(test)]
mod tests {
    use super::{FieldValue, Value};
    use std::cmp::Ordering;

    #[test]
    fn numeric_values_order_across_integer_and_float() {
        assert_eq!(
            Value::Integer(2).ordered(&Value::Float(2.5)),
            Some(Ordering::Less)
        );
        assert_eq!(
            Value::Float(3.0).ordered(&Value::Integer(3)),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn text_values_order_lexicographically() {
        assert_eq!(
            Value::Text("a".into()).ordered(&Value::Text("b".into())),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn mismatched_types_have_no_order() {
        assert_eq!(Value::Text("a".into()).ordered(&Value::Integer(1)), None);
        assert_eq!(Value::Bool(true).ordered(&Value::Bool(false)), None);
    }

    #[test]
    fn field_value_equality_distinguishes_variants() {
        let scalar = FieldValue::Scalar(Value::Integer(7));
        let array = FieldValue::Array(vec![Value::Text("x".into()), Value::Bool(true)]);
        assert_eq!(scalar, FieldValue::Scalar(Value::Integer(7)));
        assert_ne!(scalar, array);
    }
}
