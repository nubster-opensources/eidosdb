//! The payload filter AST and its compilation into an evaluable predicate.

use crate::{FieldValue, Payload, Value};
use std::cmp::Ordering;

/// A payload filter expression.
///
/// Evaluation is total: any comparison against an absent field, a type mismatch,
/// or an unorderable pair yields `false` rather than an error. Field absence is
/// tested only through [`Filter::Exists`] (negate it for "is null").
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    /// Field is a scalar structurally equal to the value.
    ///
    /// Equality is by [`Value`] variant: `Eq(field, Value::Float(3.0))` does not
    /// match a field stored as `Value::Integer(3)`. This differs from the ordered
    /// comparisons ([`Filter::Gte`] and friends), which project `Integer` and
    /// `Float` onto a common scale before comparing. The strictness is intentional
    /// so a typed schema can keep integers and floats distinct under equality.
    Eq(String, Value),
    /// Field is a present scalar not equal to the value.
    Ne(String, Value),
    /// Field is a scalar ordered strictly less than the value.
    Lt(String, Value),
    /// Field is a scalar ordered less than or equal to the value.
    Lte(String, Value),
    /// Field is a scalar ordered strictly greater than the value.
    Gt(String, Value),
    /// Field is a scalar ordered greater than or equal to the value.
    Gte(String, Value),
    /// Field is a scalar present in the list.
    ///
    /// Membership uses the same by-variant equality as [`Filter::Eq`]: an
    /// `Integer` field is not matched by a `Float` entry of equal magnitude.
    In(String, Vec<Value>),
    /// Field is an array containing the value.
    Contains(String, Value),
    /// Field is present (scalar or array).
    Exists(String),
    /// Conjunction; an empty `And` is vacuously true.
    And(Vec<Filter>),
    /// Disjunction; an empty `Or` is false.
    Or(Vec<Filter>),
    /// Negation.
    Not(Box<Filter>),
}

impl Filter {
    /// Compiles this filter into an evaluable form.
    #[must_use]
    pub fn compile(&self) -> CompiledFilter {
        CompiledFilter { root: self.clone() }
    }
}

/// A compiled filter, ready to test against payloads.
///
/// The compilation seam lets later bricks build richer evaluation plans (for
/// example indexable predicates) without changing the `PayloadStore` port.
pub struct CompiledFilter {
    root: Filter,
}

impl CompiledFilter {
    /// Returns whether `payload` satisfies the filter.
    #[must_use]
    pub fn matches(&self, payload: &Payload) -> bool {
        evaluate(&self.root, payload)
    }
}

fn evaluate(filter: &Filter, payload: &Payload) -> bool {
    match filter {
        Filter::Eq(field, value) => scalar(payload, field).is_some_and(|s| s == value),
        Filter::Ne(field, value) => scalar(payload, field).is_some_and(|s| s != value),
        Filter::Lt(field, value) => ordered_is(payload, field, value, Ordering::Less),
        Filter::Lte(field, value) => {
            ordered_is(payload, field, value, Ordering::Less)
                || ordered_is(payload, field, value, Ordering::Equal)
        }
        Filter::Gt(field, value) => ordered_is(payload, field, value, Ordering::Greater),
        Filter::Gte(field, value) => {
            ordered_is(payload, field, value, Ordering::Greater)
                || ordered_is(payload, field, value, Ordering::Equal)
        }
        Filter::In(field, values) => {
            scalar(payload, field).is_some_and(|s| values.iter().any(|v| v == s))
        }
        Filter::Contains(field, value) => match payload.get(field) {
            Some(FieldValue::Array(values)) => values.iter().any(|v| v == value),
            _ => false,
        },
        Filter::Exists(field) => payload.get(field).is_some(),
        Filter::And(filters) => filters.iter().all(|f| evaluate(f, payload)),
        Filter::Or(filters) => filters.iter().any(|f| evaluate(f, payload)),
        Filter::Not(inner) => !evaluate(inner, payload),
    }
}

/// Returns the scalar at `field`, or `None` if absent or an array.
fn scalar<'a>(payload: &'a Payload, field: &str) -> Option<&'a Value> {
    match payload.get(field) {
        Some(FieldValue::Scalar(value)) => Some(value),
        _ => None,
    }
}

fn ordered_is(payload: &Payload, field: &str, value: &Value, expected: Ordering) -> bool {
    scalar(payload, field)
        .and_then(|s| s.ordered(value))
        .is_some_and(|ordering| ordering == expected)
}

#[cfg(test)]
mod tests {
    use super::Filter;
    use crate::{FieldValue, Payload, Value};
    use std::collections::BTreeMap;

    fn payload(pairs: Vec<(&str, FieldValue)>) -> Payload {
        let map: BTreeMap<String, FieldValue> =
            pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Payload::new(map).expect("valid")
    }

    fn scalar(v: Value) -> FieldValue {
        FieldValue::Scalar(v)
    }

    fn matches(filter: &Filter, p: &Payload) -> bool {
        filter.compile().matches(p)
    }

    #[test]
    fn eq_matches_present_scalar() {
        let p = payload(vec![("k", scalar(Value::Text("a".into())))]);
        assert!(matches(
            &Filter::Eq("k".into(), Value::Text("a".into())),
            &p
        ));
        assert!(!matches(
            &Filter::Eq("k".into(), Value::Text("b".into())),
            &p
        ));
    }

    #[test]
    fn eq_on_absent_field_is_false() {
        let p = payload(vec![("k", scalar(Value::Integer(1)))]);
        assert!(!matches(
            &Filter::Eq("missing".into(), Value::Integer(1)),
            &p
        ));
    }

    #[test]
    fn ne_on_absent_field_is_false() {
        let p = payload(vec![("k", scalar(Value::Integer(1)))]);
        assert!(!matches(
            &Filter::Ne("missing".into(), Value::Integer(1)),
            &p
        ));
    }

    #[test]
    fn ne_is_true_only_for_present_differing_scalar() {
        let p = payload(vec![("k", scalar(Value::Integer(1)))]);
        assert!(matches(&Filter::Ne("k".into(), Value::Integer(2)), &p));
        assert!(!matches(&Filter::Ne("k".into(), Value::Integer(1)), &p));
    }

    #[test]
    fn numeric_ranges_match() {
        let p = payload(vec![("price", scalar(Value::Integer(100)))]);
        assert!(matches(
            &Filter::Lt("price".into(), Value::Integer(150)),
            &p
        ));
        assert!(matches(
            &Filter::Gte("price".into(), Value::Float(100.0)),
            &p
        ));
        assert!(!matches(
            &Filter::Gt("price".into(), Value::Integer(100)),
            &p
        ));
    }

    #[test]
    fn type_mismatch_comparison_is_false() {
        let p = payload(vec![("k", scalar(Value::Text("a".into())))]);
        assert!(!matches(&Filter::Lt("k".into(), Value::Integer(1)), &p));
    }

    #[test]
    fn in_matches_membership() {
        let p = payload(vec![("color", scalar(Value::Text("red".into())))]);
        let filter = Filter::In(
            "color".into(),
            vec![Value::Text("red".into()), Value::Text("blue".into())],
        );
        assert!(matches(&filter, &p));
    }

    #[test]
    fn contains_matches_array_membership() {
        let p = payload(vec![(
            "tags",
            FieldValue::Array(vec![Value::Text("rust".into()), Value::Text("db".into())]),
        )]);
        assert!(matches(
            &Filter::Contains("tags".into(), Value::Text("rust".into())),
            &p
        ));
        assert!(!matches(
            &Filter::Contains("tags".into(), Value::Text("go".into())),
            &p
        ));
        let q = payload(vec![("tags", scalar(Value::Text("rust".into())))]);
        assert!(!matches(
            &Filter::Contains("tags".into(), Value::Text("rust".into())),
            &q
        ));
    }

    #[test]
    fn exists_and_not_exists() {
        let p = payload(vec![("k", scalar(Value::Bool(true)))]);
        assert!(matches(&Filter::Exists("k".into()), &p));
        assert!(!matches(&Filter::Exists("missing".into()), &p));
        let is_null = Filter::Not(Box::new(Filter::Exists("missing".into())));
        assert!(matches(&is_null, &p));
    }

    #[test]
    fn boolean_combinators_compose() {
        let p = payload(vec![
            ("a", scalar(Value::Integer(1))),
            ("b", scalar(Value::Text("x".into()))),
        ]);
        let both = Filter::And(vec![
            Filter::Eq("a".into(), Value::Integer(1)),
            Filter::Eq("b".into(), Value::Text("x".into())),
        ]);
        assert!(matches(&both, &p));
        let either = Filter::Or(vec![
            Filter::Eq("a".into(), Value::Integer(999)),
            Filter::Eq("b".into(), Value::Text("x".into())),
        ]);
        assert!(matches(&either, &p));
        assert!(matches(
            &Filter::Not(Box::new(Filter::Eq("a".into(), Value::Integer(2)))),
            &p
        ));
    }

    #[test]
    fn empty_and_is_true_empty_or_is_false() {
        let p = payload(vec![("k", scalar(Value::Integer(1)))]);
        assert!(matches(&Filter::And(vec![]), &p));
        assert!(!matches(&Filter::Or(vec![]), &p));
    }
}
