//! Conversions between protobuf wire types and the domain [`Filter`] AST.
//!
//! Covers recursive translation of [`pb::Filter`] to and from
//! [`eidosdb_query::filter::Filter`], including the `And`/`Or` list nodes,
//! the `Not` box, and all comparison leaves.

use crate::convert::payload::{value_from_pb, value_to_pb};
use crate::error::ConversionError;
use crate::pb;
use eidosdb_query::{Filter, Value};

/// Converts a domain [`Filter`] to its protobuf wire representation.
#[must_use]
pub fn filter_to_pb(filter: &Filter) -> pb::Filter {
    let kind = match filter {
        Filter::Eq(field, value) => pb::filter::Kind::Eq(pb::Comparison {
            field: field.clone(),
            value: Some(value_to_pb(value)),
        }),
        Filter::Ne(field, value) => pb::filter::Kind::Ne(pb::Comparison {
            field: field.clone(),
            value: Some(value_to_pb(value)),
        }),
        Filter::Lt(field, value) => pb::filter::Kind::Lt(pb::Comparison {
            field: field.clone(),
            value: Some(value_to_pb(value)),
        }),
        Filter::Lte(field, value) => pb::filter::Kind::Lte(pb::Comparison {
            field: field.clone(),
            value: Some(value_to_pb(value)),
        }),
        Filter::Gt(field, value) => pb::filter::Kind::Gt(pb::Comparison {
            field: field.clone(),
            value: Some(value_to_pb(value)),
        }),
        Filter::Gte(field, value) => pb::filter::Kind::Gte(pb::Comparison {
            field: field.clone(),
            value: Some(value_to_pb(value)),
        }),
        Filter::In(field, values) => pb::filter::Kind::In(pb::InFilter {
            field: field.clone(),
            values: values.iter().map(value_to_pb).collect(),
        }),
        Filter::Contains(field, value) => pb::filter::Kind::Contains(pb::ContainsFilter {
            field: field.clone(),
            value: Some(value_to_pb(value)),
        }),
        Filter::Exists(field) => pb::filter::Kind::Exists(field.clone()),
        Filter::And(filters) => pb::filter::Kind::And(pb::FilterList {
            filters: filters.iter().map(filter_to_pb).collect(),
        }),
        Filter::Or(filters) => pb::filter::Kind::Or(pb::FilterList {
            filters: filters.iter().map(filter_to_pb).collect(),
        }),
        Filter::Not(inner) => pb::filter::Kind::Not(Box::new(filter_to_pb(inner))),
    };
    pb::Filter { kind: Some(kind) }
}

/// Converts a protobuf [`pb::Filter`] to the domain [`Filter`].
///
/// Returns [`ConversionError::MissingField`] when `kind` is `None` or when a
/// required nested field (e.g. `comparison.value`) is absent.
pub fn filter_from_pb(filter: pb::Filter) -> Result<Filter, ConversionError> {
    match filter.kind {
        None => Err(ConversionError::MissingField("filter.kind")),
        Some(pb::filter::Kind::Eq(c)) => {
            let (field, value) = comparison_from_pb(c)?;
            Ok(Filter::Eq(field, value))
        }
        Some(pb::filter::Kind::Ne(c)) => {
            let (field, value) = comparison_from_pb(c)?;
            Ok(Filter::Ne(field, value))
        }
        Some(pb::filter::Kind::Lt(c)) => {
            let (field, value) = comparison_from_pb(c)?;
            Ok(Filter::Lt(field, value))
        }
        Some(pb::filter::Kind::Lte(c)) => {
            let (field, value) = comparison_from_pb(c)?;
            Ok(Filter::Lte(field, value))
        }
        Some(pb::filter::Kind::Gt(c)) => {
            let (field, value) = comparison_from_pb(c)?;
            Ok(Filter::Gt(field, value))
        }
        Some(pb::filter::Kind::Gte(c)) => {
            let (field, value) = comparison_from_pb(c)?;
            Ok(Filter::Gte(field, value))
        }
        Some(pb::filter::Kind::In(inf)) => {
            let field = inf.field;
            let values: Result<Vec<Value>, ConversionError> =
                inf.values.into_iter().map(value_from_pb).collect();
            Ok(Filter::In(field, values?))
        }
        Some(pb::filter::Kind::Contains(cf)) => {
            let value = cf
                .value
                .ok_or(ConversionError::MissingField("contains.value"))
                .and_then(value_from_pb)?;
            Ok(Filter::Contains(cf.field, value))
        }
        Some(pb::filter::Kind::Exists(field)) => Ok(Filter::Exists(field)),
        Some(pb::filter::Kind::And(list)) => {
            let filters: Result<Vec<Filter>, ConversionError> =
                list.filters.into_iter().map(filter_from_pb).collect();
            Ok(Filter::And(filters?))
        }
        Some(pb::filter::Kind::Or(list)) => {
            let filters: Result<Vec<Filter>, ConversionError> =
                list.filters.into_iter().map(filter_from_pb).collect();
            Ok(Filter::Or(filters?))
        }
        Some(pb::filter::Kind::Not(inner)) => {
            let inner = filter_from_pb(*inner)?;
            Ok(Filter::Not(Box::new(inner)))
        }
    }
}

/// Extracts the `(field, value)` pair from a [`pb::Comparison`].
///
/// Returns [`ConversionError::MissingField`] when `value` is `None`.
fn comparison_from_pb(c: pb::Comparison) -> Result<(String, Value), ConversionError> {
    let value = c
        .value
        .ok_or(ConversionError::MissingField("comparison.value"))
        .and_then(value_from_pb)?;
    Ok((c.field, value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidosdb_query::{Filter, Value};

    #[test]
    fn nested_filter_round_trips() {
        let f = Filter::And(vec![
            Filter::Eq("kind".into(), Value::Text("note".into())),
            Filter::Or(vec![
                Filter::Gt("score".into(), Value::Integer(3)),
                Filter::Not(Box::new(Filter::Exists("archived".into()))),
            ]),
            Filter::In(
                "tag".into(),
                vec![Value::Text("a".into()), Value::Text("b".into())],
            ),
            Filter::Contains("list".into(), Value::Integer(7)),
        ]);
        let back = filter_from_pb(filter_to_pb(&f)).expect("round trip");
        assert_eq!(back, f);
    }

    #[test]
    fn filter_with_no_kind_is_rejected() {
        assert!(filter_from_pb(pb::Filter { kind: None }).is_err());
    }

    #[test]
    fn comparison_with_no_value_is_rejected() {
        let bad = pb::Filter {
            kind: Some(pb::filter::Kind::Eq(pb::Comparison {
                field: "k".into(),
                value: None,
            })),
        };
        assert!(filter_from_pb(bad).is_err());
    }

    #[test]
    fn all_comparison_leaves_round_trip() {
        let cases = [
            Filter::Eq("f".into(), Value::Integer(1)),
            Filter::Ne("f".into(), Value::Integer(2)),
            Filter::Lt("f".into(), Value::Float(1.5)),
            Filter::Lte("f".into(), Value::Float(2.5)),
            Filter::Gt("f".into(), Value::Bool(true)),
            Filter::Gte("f".into(), Value::Text("x".into())),
        ];
        for f in cases {
            let back = filter_from_pb(filter_to_pb(&f)).expect("round trip");
            assert_eq!(back, f);
        }
    }

    #[test]
    fn exists_round_trips() {
        let f = Filter::Exists("presence".into());
        let back = filter_from_pb(filter_to_pb(&f)).expect("round trip");
        assert_eq!(back, f);
    }

    #[test]
    fn not_round_trips() {
        let f = Filter::Not(Box::new(Filter::Exists("gone".into())));
        let back = filter_from_pb(filter_to_pb(&f)).expect("round trip");
        assert_eq!(back, f);
    }

    #[test]
    fn in_filter_round_trips() {
        let f = Filter::In(
            "color".into(),
            vec![Value::Text("red".into()), Value::Text("blue".into())],
        );
        let back = filter_from_pb(filter_to_pb(&f)).expect("round trip");
        assert_eq!(back, f);
    }

    #[test]
    fn contains_with_no_value_is_rejected() {
        let bad = pb::Filter {
            kind: Some(pb::filter::Kind::Contains(pb::ContainsFilter {
                field: "items".into(),
                value: None,
            })),
        };
        assert!(filter_from_pb(bad).is_err());
    }

    #[test]
    fn empty_and_and_or_round_trip() {
        let and = Filter::And(vec![]);
        let or = Filter::Or(vec![]);
        assert_eq!(filter_from_pb(filter_to_pb(&and)).unwrap(), and);
        assert_eq!(filter_from_pb(filter_to_pb(&or)).unwrap(), or);
    }
}
