//! The `LexicalIndex` port: a BM25 retrieval contract over text documents.

use crate::{Document, LexicalError};
use eidosdb_core::VectorId;

/// A lexical index: stores per-id documents and answers BM25 text queries.
///
/// The port knows ids and text, never vectors or payloads. Filtering enters as
/// an admissibility predicate, exactly as in `VectorIndex::search_filtered`.
pub trait LexicalIndex {
    /// Indexes `document` under `id`, replacing any prior document for that id.
    fn insert(&mut self, id: VectorId, document: &Document) -> Result<(), LexicalError>;

    /// Removes the document for `id`. Idempotent: absent ids are a no-op success.
    fn remove(&mut self, id: &VectorId) -> Result<(), LexicalError>;

    /// Number of indexed documents.
    fn len(&self) -> usize;

    /// Whether the index holds no documents.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Top-`k` ids by descending BM25 score for `query`, considering every
    /// document. Ties break by ascending id.
    fn search_text(&self, query: &str, k: usize) -> Vec<(VectorId, f64)> {
        self.search_text_filtered(query, k, &|_| true)
    }

    /// Top-`k` ids by descending BM25 score, restricted to ids for which
    /// `is_admissible` returns `true`. The predicate is applied during the scan.
    fn search_text_filtered(
        &self,
        query: &str,
        k: usize,
        is_admissible: &dyn Fn(&VectorId) -> bool,
    ) -> Vec<(VectorId, f64)>;
}
