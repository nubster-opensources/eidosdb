//! `InMemoryLexicalIndex`: the BM25 oracle the persistent index is checked
//! against.

use crate::{Document, LexicalError, LexicalIndex, bm25, tokenize};
use eidosdb_core::VectorId;
use std::collections::HashMap;

#[derive(Default)]
struct DocStats {
    term_frequencies: HashMap<String, u32>,
    length: u32,
}

/// In-memory BM25 index. Maintains per-document term frequencies, document
/// frequencies, and the running token total for `avgdl`.
#[derive(Default)]
pub struct InMemoryLexicalIndex {
    docs: HashMap<VectorId, DocStats>,
    doc_frequencies: HashMap<String, usize>,
    total_tokens: u64,
}

impl InMemoryLexicalIndex {
    /// Creates an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn forget(&mut self, id: &VectorId) {
        if let Some(stats) = self.docs.remove(id) {
            for term in stats.term_frequencies.keys() {
                if let Some(count) = self.doc_frequencies.get_mut(term) {
                    *count -= 1;
                    if *count == 0 {
                        self.doc_frequencies.remove(term);
                    }
                }
            }
            self.total_tokens -= u64::from(stats.length);
        }
    }
}

impl LexicalIndex for InMemoryLexicalIndex {
    fn insert(&mut self, id: VectorId, document: &Document) -> Result<(), LexicalError> {
        self.forget(&id);
        let mut term_frequencies: HashMap<String, u32> = HashMap::new();
        let mut length: u32 = 0;
        for token in tokenize(document.as_str()) {
            *term_frequencies.entry(token).or_default() += 1;
            length += 1;
        }
        for term in term_frequencies.keys() {
            *self.doc_frequencies.entry(term.clone()).or_default() += 1;
        }
        self.total_tokens += u64::from(length);
        self.docs.insert(
            id,
            DocStats {
                term_frequencies,
                length,
            },
        );
        Ok(())
    }

    fn remove(&mut self, id: &VectorId) -> Result<(), LexicalError> {
        self.forget(id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn search_text_filtered(
        &self,
        query: &str,
        k: usize,
        is_admissible: &dyn Fn(&VectorId) -> bool,
    ) -> Vec<(VectorId, f64)> {
        let corpus_size = self.docs.len();
        if corpus_size == 0 {
            return Vec::new();
        }
        #[allow(clippy::cast_precision_loss)]
        let average_doc_length = self.total_tokens as f64 / corpus_size as f64;
        let query_terms = tokenize(query);
        let mut scored: Vec<(VectorId, f64)> = self
            .docs
            .iter()
            .filter(|(id, _)| is_admissible(id))
            .filter_map(|(id, stats)| {
                let mut score = 0.0;
                for term in &query_terms {
                    let Some(&frequency) = stats.term_frequencies.get(term) else {
                        continue;
                    };
                    let doc_freq = self.doc_frequencies.get(term).copied().unwrap_or(0);
                    let idf = bm25::idf(corpus_size, doc_freq);
                    score += bm25::term_score(frequency, stats.length, average_doc_length, idf);
                }
                if score > 0.0 {
                    Some((*id, score))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::InMemoryLexicalIndex;
    use crate::{Document, LexicalIndex, bm25};
    use eidosdb_core::VectorId;

    fn doc(text: &str) -> Document {
        Document::new(text).expect("valid")
    }

    #[test]
    fn ranks_documents_containing_the_query_term() {
        let mut index = InMemoryLexicalIndex::new();
        let a = VectorId::new();
        let b = VectorId::new();
        index.insert(a, &doc("the quick brown fox")).expect("a");
        index.insert(b, &doc("the lazy dog")).expect("b");
        let hits = index.search_text("fox", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, a);
        // Hand-computed: N = 2, avgdl = 3.5, df(fox) = 1, tf = 1, len = 4.
        let expected = bm25::term_score(1, 4, 3.5, bm25::idf(2, 1));
        assert!((hits[0].1 - expected).abs() < 1e-12);
    }

    #[test]
    fn insert_replaces_prior_document() {
        let mut index = InMemoryLexicalIndex::new();
        let id = VectorId::new();
        index.insert(id, &doc("alpha")).expect("first");
        index.insert(id, &doc("beta")).expect("second");
        assert_eq!(index.len(), 1);
        assert!(index.search_text("alpha", 10).is_empty());
        assert_eq!(index.search_text("beta", 10).len(), 1);
    }

    #[test]
    fn remove_is_idempotent() {
        let mut index = InMemoryLexicalIndex::new();
        let id = VectorId::new();
        index.insert(id, &doc("alpha")).expect("insert");
        index.remove(&id).expect("remove");
        index.remove(&id).expect("remove again");
        assert!(index.is_empty());
        assert!(index.search_text("alpha", 10).is_empty());
    }

    #[test]
    fn admissibility_predicate_excludes_ids() {
        let mut index = InMemoryLexicalIndex::new();
        let keep = VectorId::new();
        let drop = VectorId::new();
        index.insert(keep, &doc("shared term")).expect("keep");
        index.insert(drop, &doc("shared term")).expect("drop");
        let hits = index.search_text_filtered("shared", 10, &|id| *id == keep);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, keep);
    }

    #[test]
    fn empty_corpus_returns_nothing() {
        let index = InMemoryLexicalIndex::new();
        assert!(index.search_text("anything", 5).is_empty());
    }
}
