//! Errors returned by lexical index operations.

/// Failure modes of a `LexicalIndex` operation.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LexicalError {
    /// A document was constructed from blank text.
    #[error("document text is empty")]
    EmptyDocument,
    /// Encoding or decoding a stored value failed.
    #[error("serialization error: {0}")]
    Serialization(String),
    /// A persistence backend operation failed.
    #[error("storage backend error: {0}")]
    Backend(String),
}

#[cfg(test)]
mod tests {
    use super::LexicalError;

    #[test]
    fn empty_document_message_is_explicit() {
        assert_eq!(
            LexicalError::EmptyDocument.to_string(),
            "document text is empty"
        );
    }

    #[test]
    fn backend_message_is_explicit() {
        let err = LexicalError::Backend("disk full".to_string());
        assert_eq!(err.to_string(), "storage backend error: disk full");
    }
}
