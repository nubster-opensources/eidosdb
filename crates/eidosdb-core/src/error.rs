//! Errors returned by vector index operations.

use crate::VectorId;

/// Failure modes of a `VectorIndex` operation.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IndexError {
    /// An embedding did not match the index dimension.
    #[error("dimension mismatch: index expects {expected}, got {got}")]
    DimensionMismatch {
        /// Dimension the index was created with.
        expected: usize,
        /// Dimension of the offending embedding.
        got: usize,
    },
    /// An id already present in the index was inserted again.
    #[error("duplicate id: {0:?}")]
    DuplicateId(VectorId),
    /// An embedding was constructed from an empty vector.
    #[error("embedding is empty")]
    EmptyEmbedding,
    /// An embedding contained a non-finite component (NaN or infinity).
    #[error("embedding contains a non-finite component")]
    NonFiniteComponent,
    /// A persistence backend operation failed.
    #[error("storage backend error: {0}")]
    Backend(String),
}

#[cfg(test)]
mod tests {
    use super::IndexError;

    #[test]
    fn dimension_mismatch_message_is_explicit() {
        let err = IndexError::DimensionMismatch {
            expected: 768,
            got: 384,
        };
        assert_eq!(
            err.to_string(),
            "dimension mismatch: index expects 768, got 384"
        );
    }

    #[test]
    fn backend_message_is_explicit() {
        let err = IndexError::Backend("disk full".to_string());
        assert_eq!(err.to_string(), "storage backend error: disk full");
    }
}
