//! Conversions between protobuf wire types and domain payload types.
//!
//! Covers [`pb::Value`], [`pb::FieldValue`], and [`pb::Payload`], mapping them
//! to and from the domain types [`Value`], [`FieldValue`], and [`Payload`].

use crate::error::ConversionError;
use crate::pb;
use eidosdb_query::{FieldValue, Payload, Value};
use std::collections::BTreeMap;

/// Converts a domain [`Value`] to its protobuf wire representation.
#[must_use]
pub fn value_to_pb(value: &Value) -> pb::Value {
    let kind = match value {
        Value::Text(s) => pb::value::Kind::Text(s.clone()),
        Value::Integer(i) => pb::value::Kind::Integer(*i),
        Value::Float(f) => pb::value::Kind::FloatValue(*f),
        Value::Bool(b) => pb::value::Kind::BoolValue(*b),
    };
    pb::Value { kind: Some(kind) }
}

/// Converts a protobuf [`pb::Value`] to the domain [`Value`].
///
/// Returns [`ConversionError::MissingField`] when `kind` is `None`.
pub fn value_from_pb(value: pb::Value) -> Result<Value, ConversionError> {
    match value.kind {
        None => Err(ConversionError::MissingField("value.kind")),
        Some(pb::value::Kind::Text(s)) => Ok(Value::Text(s)),
        Some(pb::value::Kind::Integer(i)) => Ok(Value::Integer(i)),
        Some(pb::value::Kind::FloatValue(f)) => Ok(Value::Float(f)),
        Some(pb::value::Kind::BoolValue(b)) => Ok(Value::Bool(b)),
    }
}

/// Converts a domain [`FieldValue`] to its protobuf wire representation.
#[must_use]
pub fn field_value_to_pb(field_value: &FieldValue) -> pb::FieldValue {
    let kind = match field_value {
        FieldValue::Scalar(v) => pb::field_value::Kind::Scalar(value_to_pb(v)),
        FieldValue::Array(values) => pb::field_value::Kind::Array(pb::ValueArray {
            values: values.iter().map(value_to_pb).collect(),
        }),
    };
    pb::FieldValue { kind: Some(kind) }
}

/// Converts a protobuf [`pb::FieldValue`] to the domain [`FieldValue`].
///
/// Returns [`ConversionError::MissingField`] when `kind` is `None`.
pub fn field_value_from_pb(field_value: pb::FieldValue) -> Result<FieldValue, ConversionError> {
    match field_value.kind {
        None => Err(ConversionError::MissingField("field_value.kind")),
        Some(pb::field_value::Kind::Scalar(v)) => value_from_pb(v).map(FieldValue::Scalar),
        Some(pb::field_value::Kind::Array(arr)) => {
            let values: Result<Vec<Value>, ConversionError> =
                arr.values.into_iter().map(value_from_pb).collect();
            values.map(FieldValue::Array)
        }
    }
}

/// Converts a domain [`Payload`] to its protobuf wire representation.
#[must_use]
pub fn payload_to_pb(payload: &Payload) -> pb::Payload {
    let fields = payload
        .iter()
        .map(|(k, v)| (k.clone(), field_value_to_pb(v)))
        .collect();
    pb::Payload { fields }
}

/// Converts a protobuf [`pb::Payload`] to the domain [`Payload`].
///
/// Returns [`ConversionError::MissingField`] when a field value has no `kind`,
/// or [`ConversionError::Domain`] when the domain layer rejects the payload
/// (e.g. a non-finite float).
pub fn payload_from_pb(payload: pb::Payload) -> Result<Payload, ConversionError> {
    let fields: Result<BTreeMap<String, FieldValue>, ConversionError> = payload
        .fields
        .into_iter()
        .map(|(k, v)| field_value_from_pb(v).map(|fv| (k, fv)))
        .collect();
    Payload::new(fields?).map_err(|e| ConversionError::Domain(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidosdb_query::{FieldValue, Payload, Value};
    use std::collections::BTreeMap;

    #[test]
    fn payload_round_trips() {
        let mut fields = BTreeMap::new();
        fields.insert(
            "title".to_string(),
            FieldValue::Scalar(Value::Text("hello".into())),
        );
        fields.insert("count".to_string(), FieldValue::Scalar(Value::Integer(7)));
        fields.insert("ratio".to_string(), FieldValue::Scalar(Value::Float(0.5)));
        fields.insert("flag".to_string(), FieldValue::Scalar(Value::Bool(true)));
        fields.insert(
            "tags".to_string(),
            FieldValue::Array(vec![Value::Text("a".into()), Value::Integer(1)]),
        );
        let p = Payload::new(fields).expect("valid payload");
        let back = payload_from_pb(payload_to_pb(&p)).expect("round trip");
        assert_eq!(back, p);
    }

    #[test]
    fn value_with_no_kind_is_rejected() {
        let bad = pb::Value { kind: None };
        assert!(value_from_pb(bad).is_err());
    }

    #[test]
    fn field_value_with_no_kind_is_rejected() {
        let bad = pb::FieldValue { kind: None };
        assert!(field_value_from_pb(bad).is_err());
    }

    #[test]
    fn value_round_trips_all_variants() {
        let cases = [
            Value::Text("world".into()),
            Value::Integer(-42),
            Value::Float(1.5),
            Value::Bool(false),
        ];
        for v in cases {
            let back = value_from_pb(value_to_pb(&v)).expect("round trip");
            assert_eq!(back, v);
        }
    }

    #[test]
    fn field_value_round_trips_scalar_and_array() {
        let scalar = FieldValue::Scalar(Value::Integer(99));
        let array = FieldValue::Array(vec![Value::Text("x".into()), Value::Bool(true)]);
        for fv in [scalar, array] {
            let back = field_value_from_pb(field_value_to_pb(&fv)).expect("round trip");
            assert_eq!(back, fv);
        }
    }
}
