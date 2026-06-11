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
    /// A query embedding was empty.
    #[error("query embedding is empty")]
    EmptyQuery,
    /// An embedding was constructed from an empty vector.
    #[error("embedding is empty")]
    EmptyEmbedding,
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
}
