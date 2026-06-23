//! A validated, schemaless-but-typed payload attached to a vector.

use crate::{FieldValue, PayloadError, Value};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A map of field name to typed value, attached to a `VectorId`.
///
/// `BTreeMap` gives a deterministic field order for reproducible serialization
/// and tests. Floats are validated finite at construction, including the
/// deserialization path (serde routes through `TryFrom`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "BTreeMap<String, FieldValue>",
    into = "BTreeMap<String, FieldValue>"
)]
pub struct Payload(BTreeMap<String, FieldValue>);

impl Payload {
    /// Builds a payload, rejecting any non-finite float value.
    pub fn new(fields: BTreeMap<String, FieldValue>) -> Result<Self, PayloadError> {
        for field in fields.values() {
            match field {
                FieldValue::Scalar(value) => check_finite(value)?,
                FieldValue::Array(values) => {
                    for value in values {
                        check_finite(value)?;
                    }
                }
            }
        }
        Ok(Self(fields))
    }

    /// Returns the value of `name`, or `None` if absent.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&FieldValue> {
        self.0.get(name)
    }

    /// Number of fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the payload has no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterates over all field entries as `(name, value)` pairs.
    ///
    /// Order is deterministic (ascending by field name) because the backing store
    /// is a [`BTreeMap`].
    pub fn iter(&self) -> impl Iterator<Item = (&String, &FieldValue)> {
        self.0.iter()
    }
}

fn check_finite(value: &Value) -> Result<(), PayloadError> {
    if let Value::Float(f) = value {
        if !f.is_finite() {
            return Err(PayloadError::NonFiniteValue);
        }
    }
    Ok(())
}

impl TryFrom<BTreeMap<String, FieldValue>> for Payload {
    type Error = PayloadError;

    fn try_from(fields: BTreeMap<String, FieldValue>) -> Result<Self, Self::Error> {
        Payload::new(fields)
    }
}

impl From<Payload> for BTreeMap<String, FieldValue> {
    fn from(payload: Payload) -> Self {
        payload.0
    }
}

#[cfg(test)]
mod tests {
    use super::Payload;
    use crate::{FieldValue, PayloadError, Value};
    use std::collections::BTreeMap;

    fn map(pairs: Vec<(&str, FieldValue)>) -> BTreeMap<String, FieldValue> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    #[test]
    fn builds_and_reads_back_fields() {
        let payload = Payload::new(map(vec![
            ("source", FieldValue::Scalar(Value::Text("wiki".into()))),
            ("score", FieldValue::Scalar(Value::Float(0.5))),
        ]))
        .expect("valid payload");
        assert_eq!(
            payload.get("source"),
            Some(&FieldValue::Scalar(Value::Text("wiki".into())))
        );
        assert_eq!(payload.get("missing"), None);
        assert_eq!(payload.len(), 2);
    }

    #[test]
    fn rejects_non_finite_scalar_float() {
        let err = Payload::new(map(vec![("x", FieldValue::Scalar(Value::Float(f64::NAN)))]));
        assert!(matches!(err, Err(PayloadError::NonFiniteValue)));
    }

    #[test]
    fn rejects_non_finite_float_inside_array() {
        let err = Payload::new(map(vec![(
            "xs",
            FieldValue::Array(vec![Value::Float(1.0), Value::Float(f64::INFINITY)]),
        )]));
        assert!(matches!(err, Err(PayloadError::NonFiniteValue)));
    }

    #[test]
    fn deserialization_rejects_non_finite_via_try_from() {
        let raw: BTreeMap<String, FieldValue> =
            map(vec![("x", FieldValue::Scalar(Value::Float(f64::NAN)))]);
        let result: Result<Payload, _> = Payload::try_from(raw);
        assert!(matches!(result, Err(PayloadError::NonFiniteValue)));
    }
}
