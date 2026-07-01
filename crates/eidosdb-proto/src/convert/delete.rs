//! Conversions for the delete-by-filter request.

use crate::convert::{filter_from_pb, filter_to_pb};
use crate::error::ConversionError;
use crate::pb;
use eidosdb_query::Filter;

/// Builds the wire request for a delete-by-filter call.
#[must_use]
pub fn delete_by_filter_to_pb(collection: &str, filter: &Filter) -> pb::DeleteByFilterRequest {
    pb::DeleteByFilterRequest {
        collection: collection.to_string(),
        filter: Some(filter_to_pb(filter)),
    }
}

/// Parses a wire delete-by-filter request into the collection name and domain filter.
///
/// # Errors
///
/// Returns [`ConversionError::MissingField`] if the filter is absent.
pub fn delete_by_filter_from_pb(
    request: pb::DeleteByFilterRequest,
) -> Result<(String, Filter), ConversionError> {
    let filter = request
        .filter
        .ok_or(ConversionError::MissingField("delete_by_filter.filter"))?;
    Ok((request.collection, filter_from_pb(filter)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidosdb_query::Value;

    #[test]
    fn round_trips_collection_and_filter() {
        let filter = Filter::Eq("theme".into(), Value::Text("press".into()));
        let wire = delete_by_filter_to_pb("press", &filter);
        let (collection, parsed) = delete_by_filter_from_pb(wire).expect("round trip");
        assert_eq!(collection, "press");
        assert_eq!(parsed, filter);
    }

    #[test]
    fn rejects_missing_filter() {
        let wire = pb::DeleteByFilterRequest {
            collection: "press".into(),
            filter: None,
        };
        assert!(delete_by_filter_from_pb(wire).is_err());
    }
}
