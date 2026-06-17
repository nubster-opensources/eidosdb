//! The searchable text body associated with a vector.

use crate::LexicalError;

/// The text indexed for lexical retrieval. Non-blank by construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Document(String);

impl Document {
    /// Builds a document, rejecting text that is empty or only whitespace.
    pub fn new(text: impl Into<String>) -> Result<Self, LexicalError> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(LexicalError::EmptyDocument);
        }
        Ok(Self(text))
    }

    /// The raw document text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::Document;
    use crate::LexicalError;

    #[test]
    fn keeps_original_text() {
        let doc = Document::new("Quick brown fox").expect("valid");
        assert_eq!(doc.as_str(), "Quick brown fox");
    }

    #[test]
    fn rejects_blank_text() {
        assert_eq!(Document::new("   "), Err(LexicalError::EmptyDocument));
        assert_eq!(Document::new(""), Err(LexicalError::EmptyDocument));
    }
}
