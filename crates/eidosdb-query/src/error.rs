//! Errors raised by the query layer.

use eidosdb_core::{IndexError, Metric};
use eidosdb_lexical::LexicalError;

/// Failure modes of a `PayloadStore` operation.
#[derive(Debug, thiserror::Error)]
pub enum PayloadError {
    /// A payload carried a non-finite float (NaN or infinity).
    #[error("payload contains a non-finite float value")]
    NonFiniteValue,
    /// (De)serialization of a stored payload failed.
    #[error("payload serialization error: {0}")]
    Serialization(String),
    /// The persistence backend reported an error.
    #[error("payload backend error: {0}")]
    Backend(String),
}

/// Failure modes of a `Collection` query.
#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    /// The requested metric is not supported by the index.
    #[error("metric not supported by this index: {0:?}")]
    UnsupportedMetric(Metric),
    /// An underlying index operation failed.
    #[error(transparent)]
    Index(#[from] IndexError),
    /// An underlying payload store operation failed.
    #[error(transparent)]
    Payload(#[from] PayloadError),
    /// A hybrid query supplied neither a dense vector nor query text.
    #[error("query has neither a vector nor text")]
    EmptyQuery,
    /// A lexical index operation failed.
    #[error(transparent)]
    Lexical(#[from] LexicalError),
}

#[cfg(test)]
mod tests {
    use super::{PayloadError, QueryError};
    use eidosdb_core::Metric;

    #[test]
    fn unsupported_metric_message_is_explicit() {
        let err = QueryError::UnsupportedMetric(Metric::DotProduct);
        assert_eq!(
            err.to_string(),
            "metric not supported by this index: DotProduct"
        );
    }

    #[test]
    fn non_finite_payload_message_is_explicit() {
        assert_eq!(
            PayloadError::NonFiniteValue.to_string(),
            "payload contains a non-finite float value"
        );
    }

    #[test]
    fn empty_query_message_is_explicit() {
        assert_eq!(
            super::QueryError::EmptyQuery.to_string(),
            "query has neither a vector nor text"
        );
    }
}
