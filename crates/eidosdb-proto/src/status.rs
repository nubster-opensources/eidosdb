//! Mapping from domain error types to [`tonic::Status`].
//!
//! Each public function converts a strongly-typed domain error into the gRPC
//! status code that best communicates the failure to a caller.

use eidosdb_core::IndexError;
use eidosdb_query::QueryError;
use tonic::Status;

use crate::error::ConversionError;

/// Converts an [`IndexError`] into the appropriate [`tonic::Status`].
///
/// Mapping:
/// - `DuplicateId`                              -> `ALREADY_EXISTS`
/// - `DimensionMismatch | EmptyEmbedding`
///   `| NonFiniteComponent | UnsupportedMetric` -> `INVALID_ARGUMENT`
/// - `Backend`                                  -> `INTERNAL`
#[must_use]
pub fn index_error_to_status(error: &IndexError) -> Status {
    match error {
        IndexError::DuplicateId(_) => Status::already_exists(error.to_string()),
        IndexError::DimensionMismatch { .. }
        | IndexError::EmptyEmbedding
        | IndexError::NonFiniteComponent
        | IndexError::UnsupportedMetric(_) => Status::invalid_argument(error.to_string()),
        IndexError::Backend(_) => Status::internal(error.to_string()),
    }
}

/// Converts a [`QueryError`] into the appropriate [`tonic::Status`].
///
/// Mapping:
/// - `EmptyQuery | UnsupportedMetric | Payload(NonFiniteValue)` -> `INVALID_ARGUMENT`
/// - `Index(e)`                                                 -> delegates to `index_error_to_status`
/// - `Payload(Serialization | Backend) | Lexical`               -> `INTERNAL`
#[must_use]
pub fn query_error_to_status(error: &QueryError) -> Status {
    use eidosdb_query::PayloadError;

    match error {
        QueryError::EmptyQuery
        | QueryError::UnsupportedMetric(_)
        | QueryError::Payload(PayloadError::NonFiniteValue) => {
            Status::invalid_argument(error.to_string())
        }
        QueryError::Index(inner) => index_error_to_status(inner),
        QueryError::Payload(PayloadError::Serialization(_) | PayloadError::Backend(_)) => {
            Status::internal(error.to_string())
        }
        QueryError::Lexical(_) => Status::internal(error.to_string()),
    }
}

/// Converts a [`ConversionError`] into a [`tonic::Status`].
///
/// All conversion errors map to `INVALID_ARGUMENT` — they represent malformed
/// input from the caller.
#[must_use]
pub fn conversion_error_to_status(error: &ConversionError) -> Status {
    match error {
        ConversionError::InvalidUuid(_)
        | ConversionError::MissingField(_)
        | ConversionError::Domain(_) => Status::invalid_argument(error.to_string()),
    }
}

/// Builds a `NOT_FOUND` status for a named resource.
#[must_use]
pub fn not_found(name: &str) -> Status {
    Status::not_found(format!("resource not found: {name}"))
}

/// Builds an `ALREADY_EXISTS` status for a named resource.
#[must_use]
pub fn already_exists(name: &str) -> Status {
    Status::already_exists(format!("resource already exists: {name}"))
}

#[cfg(test)]
mod tests {
    use eidosdb_core::{IndexError, Metric, VectorId};
    use eidosdb_lexical::LexicalError;
    use eidosdb_query::{PayloadError, QueryError};
    use tonic::Code;

    use super::*;
    use crate::error::ConversionError;

    // --- index_error_to_status ---

    #[test]
    fn index_duplicate_id_maps_to_already_exists() {
        let id = VectorId::new();
        let err = IndexError::DuplicateId(id);
        assert_eq!(index_error_to_status(&err).code(), Code::AlreadyExists);
    }

    #[test]
    fn index_dimension_mismatch_maps_to_invalid_argument() {
        let err = IndexError::DimensionMismatch {
            expected: 768,
            got: 384,
        };
        assert_eq!(index_error_to_status(&err).code(), Code::InvalidArgument);
    }

    #[test]
    fn index_empty_embedding_maps_to_invalid_argument() {
        let err = IndexError::EmptyEmbedding;
        assert_eq!(index_error_to_status(&err).code(), Code::InvalidArgument);
    }

    #[test]
    fn index_non_finite_component_maps_to_invalid_argument() {
        let err = IndexError::NonFiniteComponent;
        assert_eq!(index_error_to_status(&err).code(), Code::InvalidArgument);
    }

    #[test]
    fn index_unsupported_metric_maps_to_invalid_argument() {
        let err = IndexError::UnsupportedMetric(Metric::Cosine);
        assert_eq!(index_error_to_status(&err).code(), Code::InvalidArgument);
    }

    #[test]
    fn index_backend_maps_to_internal() {
        let err = IndexError::Backend("disk full".to_string());
        assert_eq!(index_error_to_status(&err).code(), Code::Internal);
    }

    #[test]
    fn index_error_status_message_contains_error_text() {
        let err = IndexError::Backend("redb crashed".to_string());
        let status = index_error_to_status(&err);
        assert!(
            status.message().contains("redb crashed"),
            "message should contain the error text, got: {:?}",
            status.message()
        );
    }

    // --- query_error_to_status ---

    #[test]
    fn query_empty_query_maps_to_invalid_argument() {
        let err = QueryError::EmptyQuery;
        assert_eq!(query_error_to_status(&err).code(), Code::InvalidArgument);
    }

    #[test]
    fn query_unsupported_metric_maps_to_invalid_argument() {
        let err = QueryError::UnsupportedMetric(Metric::DotProduct);
        assert_eq!(query_error_to_status(&err).code(), Code::InvalidArgument);
    }

    #[test]
    fn query_index_variant_delegates_to_index_mapping() {
        let inner = IndexError::DuplicateId(VectorId::new());
        let err = QueryError::Index(inner);
        assert_eq!(query_error_to_status(&err).code(), Code::AlreadyExists);
    }

    #[test]
    fn query_index_backend_delegates_correctly() {
        let inner = IndexError::Backend("io error".to_string());
        let err = QueryError::Index(inner);
        assert_eq!(query_error_to_status(&err).code(), Code::Internal);
    }

    #[test]
    fn query_payload_non_finite_maps_to_invalid_argument() {
        let err = QueryError::Payload(PayloadError::NonFiniteValue);
        assert_eq!(query_error_to_status(&err).code(), Code::InvalidArgument);
    }

    #[test]
    fn query_payload_serialization_maps_to_internal() {
        let err = QueryError::Payload(PayloadError::Serialization("bad json".to_string()));
        assert_eq!(query_error_to_status(&err).code(), Code::Internal);
    }

    #[test]
    fn query_payload_backend_maps_to_internal() {
        let err = QueryError::Payload(PayloadError::Backend("redb".to_string()));
        assert_eq!(query_error_to_status(&err).code(), Code::Internal);
    }

    #[test]
    fn query_lexical_maps_to_internal() {
        let err = QueryError::Lexical(LexicalError::Backend("tantivy".to_string()));
        assert_eq!(query_error_to_status(&err).code(), Code::Internal);
    }

    #[test]
    fn query_error_status_message_contains_error_text() {
        let err = QueryError::EmptyQuery;
        let status = query_error_to_status(&err);
        assert!(
            !status.message().is_empty(),
            "status message must not be empty"
        );
    }

    // --- conversion_error_to_status ---

    #[test]
    fn conversion_invalid_uuid_maps_to_invalid_argument() {
        let err = ConversionError::InvalidUuid("not-a-uuid".to_string());
        assert_eq!(
            conversion_error_to_status(&err).code(),
            Code::InvalidArgument
        );
    }

    #[test]
    fn conversion_missing_field_maps_to_invalid_argument() {
        let err = ConversionError::MissingField("vector");
        assert_eq!(
            conversion_error_to_status(&err).code(),
            Code::InvalidArgument
        );
    }

    #[test]
    fn conversion_domain_maps_to_invalid_argument() {
        let err = ConversionError::Domain("bad metric value".to_string());
        assert_eq!(
            conversion_error_to_status(&err).code(),
            Code::InvalidArgument
        );
    }

    // --- not_found / already_exists ---

    #[test]
    fn not_found_returns_not_found_code() {
        let status = not_found("collection:my_coll");
        assert_eq!(status.code(), Code::NotFound);
    }

    #[test]
    fn not_found_message_contains_name() {
        let status = not_found("collection:my_coll");
        assert!(
            status.message().contains("my_coll"),
            "message should contain the resource name, got: {:?}",
            status.message()
        );
    }

    #[test]
    fn already_exists_returns_already_exists_code() {
        let status = already_exists("collection:my_coll");
        assert_eq!(status.code(), Code::AlreadyExists);
    }

    #[test]
    fn already_exists_message_contains_name() {
        let status = already_exists("collection:my_coll");
        assert!(
            status.message().contains("my_coll"),
            "message should contain the resource name, got: {:?}",
            status.message()
        );
    }
}
