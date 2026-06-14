//! Errors raised by the storage backend.

use eidosdb_core::IndexError;

/// Failure modes of the persistent storage backend.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// An underlying I/O operation failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The redb catalog reported an error.
    #[error("catalog error: {0}")]
    Catalog(String),
    /// An on-disk file did not match the expected `EidosDB` format.
    #[error("corrupt store: {0}")]
    Corruption(String),
    /// The store on disk was built with parameters incompatible with this open.
    #[error("format mismatch: {0}")]
    FormatMismatch(String),
    /// A domain-level index error occurred.
    #[error(transparent)]
    Index(#[from] IndexError),
}

impl From<StorageError> for IndexError {
    fn from(error: StorageError) -> Self {
        match error {
            StorageError::Index(inner) => inner,
            other => IndexError::Backend(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StorageError;
    use eidosdb_core::IndexError;

    #[test]
    fn corruption_maps_to_backend() {
        let err: IndexError = StorageError::Corruption("bad magic".to_string()).into();
        assert_eq!(err, IndexError::Backend("corrupt store: bad magic".to_string()));
    }

    #[test]
    fn index_variant_round_trips() {
        let err: IndexError = StorageError::Index(IndexError::EmptyEmbedding).into();
        assert_eq!(err, IndexError::EmptyEmbedding);
    }
}
