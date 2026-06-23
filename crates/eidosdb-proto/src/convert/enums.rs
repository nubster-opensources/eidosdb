//! Conversions between protobuf enum wire types and domain or local enum types.

use crate::error::ConversionError;
use crate::pb;
use eidosdb_core::Metric;

/// Local enum representing the index algorithm to use when creating a collection.
///
/// There is no domain-level `IndexType`; this choice is local to the gRPC layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexTypeChoice {
    /// Flat brute-force index.
    Flat,
    /// Hierarchical navigable small-world graph index.
    Hnsw,
}

/// Converts a domain [`Metric`] to its protobuf wire representation.
#[must_use]
pub fn metric_to_pb(metric: Metric) -> pb::Metric {
    match metric {
        Metric::Cosine => pb::Metric::Cosine,
        Metric::DotProduct => pb::Metric::DotProduct,
        Metric::Euclidean => pb::Metric::Euclidean,
    }
}

/// Converts a protobuf [`pb::Metric`] to the domain [`Metric`].
///
/// Returns [`ConversionError::MissingField`] when the value is the unspecified sentinel.
pub fn metric_from_pb(metric: pb::Metric) -> Result<Metric, ConversionError> {
    match metric {
        pb::Metric::Cosine => Ok(Metric::Cosine),
        pb::Metric::DotProduct => Ok(Metric::DotProduct),
        pb::Metric::Euclidean => Ok(Metric::Euclidean),
        pb::Metric::Unspecified => Err(ConversionError::MissingField("metric")),
    }
}

/// Converts a local [`IndexTypeChoice`] to its protobuf wire representation.
#[must_use]
pub fn index_type_to_pb(choice: IndexTypeChoice) -> pb::IndexType {
    match choice {
        IndexTypeChoice::Flat => pb::IndexType::Flat,
        IndexTypeChoice::Hnsw => pb::IndexType::Hnsw,
    }
}

/// Converts a protobuf [`pb::IndexType`] to the local [`IndexTypeChoice`].
///
/// Returns [`ConversionError::MissingField`] when the value is the unspecified sentinel.
pub fn index_type_from_pb(index_type: pb::IndexType) -> Result<IndexTypeChoice, ConversionError> {
    match index_type {
        pb::IndexType::Flat => Ok(IndexTypeChoice::Flat),
        pb::IndexType::Hnsw => Ok(IndexTypeChoice::Hnsw),
        pb::IndexType::Unspecified => Err(ConversionError::MissingField("index_type")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidosdb_core::Metric;

    #[test]
    fn metric_round_trips_through_pb() {
        for m in [Metric::Cosine, Metric::DotProduct, Metric::Euclidean] {
            let back = metric_from_pb(metric_to_pb(m)).expect("known metric");
            assert_eq!(back, m);
        }
    }

    #[test]
    fn unspecified_metric_is_rejected() {
        assert!(matches!(
            metric_from_pb(pb::Metric::Unspecified),
            Err(ConversionError::MissingField(_))
        ));
    }

    #[test]
    fn index_type_round_trips_through_pb() {
        for c in [IndexTypeChoice::Flat, IndexTypeChoice::Hnsw] {
            let back = index_type_from_pb(index_type_to_pb(c)).expect("known index type");
            assert_eq!(back, c);
        }
    }

    #[test]
    fn unspecified_index_type_is_rejected() {
        assert!(matches!(
            index_type_from_pb(pb::IndexType::Unspecified),
            Err(ConversionError::MissingField(_))
        ));
    }
}
