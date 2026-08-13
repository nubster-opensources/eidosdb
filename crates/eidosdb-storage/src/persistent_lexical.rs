//! `PersistentLexicalIndex`: a durable `LexicalIndex` backed by redb, with
//! postings, document lengths, and per-document term lists serialized via
//! postcard.

use crate::redb_compat;
use eidosdb_core::VectorId;
use eidosdb_lexical::{Document, LexicalError, LexicalIndex, bm25, tokenize};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use std::collections::HashMap;
use std::fmt::Display;
use std::path::Path;
use uuid::Uuid;

const POSTINGS: TableDefinition<&str, &[u8]> = TableDefinition::new("postings");
const DOC_LENGTHS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("doc_lengths");
const DOC_TERMS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("doc_terms");

/// A redb-backed BM25 index. Parity-checked against `InMemoryLexicalIndex`.
pub struct PersistentLexicalIndex {
    db: Database,
    corpus_size: usize,
    total_tokens: u64,
}

fn backend<E: Display>(error: E) -> LexicalError {
    LexicalError::Backend(error.to_string())
}

fn encode<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, LexicalError> {
    postcard::to_allocvec(value).map_err(|e| LexicalError::Serialization(e.to_string()))
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, LexicalError> {
    postcard::from_bytes(bytes).map_err(|e| LexicalError::Serialization(e.to_string()))
}

impl PersistentLexicalIndex {
    /// Opens a lexical index at `path`, creating it if absent.
    pub fn open(path: &Path) -> Result<Self, LexicalError> {
        let db = redb_compat::create(path).map_err(backend)?;
        let txn = db.begin_write().map_err(backend)?;
        {
            let _ = txn.open_table(POSTINGS).map_err(backend)?;
            let _ = txn.open_table(DOC_LENGTHS).map_err(backend)?;
            let _ = txn.open_table(DOC_TERMS).map_err(backend)?;
        }
        txn.commit().map_err(backend)?;
        let (corpus_size, total_tokens) = {
            let txn = db.begin_read().map_err(backend)?;
            let table = txn.open_table(DOC_LENGTHS).map_err(backend)?;
            let mut count = 0usize;
            let mut tokens = 0u64;
            for entry in table.iter().map_err(backend)? {
                let (_, value) = entry.map_err(backend)?;
                let length: u32 = decode(value.value())?;
                count += 1;
                tokens += u64::from(length);
            }
            (count, tokens)
        };
        Ok(Self {
            db,
            corpus_size,
            total_tokens,
        })
    }
}

impl LexicalIndex for PersistentLexicalIndex {
    fn insert(&mut self, id: VectorId, document: &Document) -> Result<(), LexicalError> {
        let key = id.as_uuid().into_bytes();
        let mut term_frequencies: HashMap<String, u32> = HashMap::new();
        let mut length: u32 = 0;
        for token in tokenize(document.as_str()) {
            *term_frequencies.entry(token).or_default() += 1;
            length += 1;
        }
        let id_value = id.as_uuid().as_u128();

        let txn = self.db.begin_write().map_err(backend)?;
        let mut size_delta: i64 = 0;
        let mut token_delta: i64 = 0;
        {
            let mut postings = txn.open_table(POSTINGS).map_err(backend)?;
            let mut lengths = txn.open_table(DOC_LENGTHS).map_err(backend)?;
            let mut terms = txn.open_table(DOC_TERMS).map_err(backend)?;

            // Remove a prior document for this id, if any.
            let prior_terms: Option<Vec<String>> = terms
                .get(key.as_slice())
                .map_err(backend)?
                .map(|v| decode(v.value()))
                .transpose()?;
            if let Some(prior_terms) = prior_terms {
                for term in &prior_terms {
                    let mut list: Vec<(u128, u32)> = postings
                        .get(term.as_str())
                        .map_err(backend)?
                        .map(|v| decode(v.value()))
                        .transpose()?
                        .unwrap_or_default();
                    list.retain(|(other, _)| *other != id_value);
                    if list.is_empty() {
                        postings.remove(term.as_str()).map_err(backend)?;
                    } else {
                        postings
                            .insert(term.as_str(), encode(&list)?.as_slice())
                            .map_err(backend)?;
                    }
                }
                if let Some(old) = lengths.get(key.as_slice()).map_err(backend)? {
                    let old_length: u32 = decode(old.value())?;
                    token_delta -= i64::from(old_length);
                }
                size_delta -= 1;
            }

            // Insert the new document.
            for (term, frequency) in &term_frequencies {
                let mut list: Vec<(u128, u32)> = postings
                    .get(term.as_str())
                    .map_err(backend)?
                    .map(|v| decode(v.value()))
                    .transpose()?
                    .unwrap_or_default();
                list.push((id_value, *frequency));
                postings
                    .insert(term.as_str(), encode(&list)?.as_slice())
                    .map_err(backend)?;
            }
            let distinct: Vec<String> = term_frequencies.keys().cloned().collect();
            terms
                .insert(key.as_slice(), encode(&distinct)?.as_slice())
                .map_err(backend)?;
            lengths
                .insert(key.as_slice(), encode(&length)?.as_slice())
                .map_err(backend)?;
            token_delta += i64::from(length);
            size_delta += 1;
        }
        txn.commit().map_err(backend)?;

        apply_delta(&mut self.corpus_size, size_delta);
        apply_token_delta(&mut self.total_tokens, token_delta);
        Ok(())
    }

    fn remove(&mut self, id: &VectorId) -> Result<(), LexicalError> {
        let key = id.as_uuid().into_bytes();
        let id_value = id.as_uuid().as_u128();
        let txn = self.db.begin_write().map_err(backend)?;
        let mut size_delta: i64 = 0;
        let mut token_delta: i64 = 0;
        {
            let mut postings = txn.open_table(POSTINGS).map_err(backend)?;
            let mut lengths = txn.open_table(DOC_LENGTHS).map_err(backend)?;
            let mut terms = txn.open_table(DOC_TERMS).map_err(backend)?;
            let prior_terms: Option<Vec<String>> = terms
                .get(key.as_slice())
                .map_err(backend)?
                .map(|v| decode(v.value()))
                .transpose()?;
            if let Some(prior_terms) = prior_terms {
                for term in &prior_terms {
                    let mut list: Vec<(u128, u32)> = postings
                        .get(term.as_str())
                        .map_err(backend)?
                        .map(|v| decode(v.value()))
                        .transpose()?
                        .unwrap_or_default();
                    list.retain(|(other, _)| *other != id_value);
                    if list.is_empty() {
                        postings.remove(term.as_str()).map_err(backend)?;
                    } else {
                        postings
                            .insert(term.as_str(), encode(&list)?.as_slice())
                            .map_err(backend)?;
                    }
                }
                if let Some(old) = lengths.get(key.as_slice()).map_err(backend)? {
                    let old_length: u32 = decode(old.value())?;
                    token_delta -= i64::from(old_length);
                }
                lengths.remove(key.as_slice()).map_err(backend)?;
                terms.remove(key.as_slice()).map_err(backend)?;
                size_delta -= 1;
            }
        }
        txn.commit().map_err(backend)?;
        apply_delta(&mut self.corpus_size, size_delta);
        apply_token_delta(&mut self.total_tokens, token_delta);
        Ok(())
    }

    fn len(&self) -> usize {
        self.corpus_size
    }

    fn search_text_filtered(
        &self,
        query: &str,
        k: usize,
        is_admissible: &dyn Fn(&VectorId) -> bool,
    ) -> Vec<(VectorId, f64)> {
        match self.scored(query, is_admissible) {
            Ok(mut scored) => {
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                scored.truncate(k);
                scored
            }
            Err(_) => Vec::new(),
        }
    }
}

impl PersistentLexicalIndex {
    fn scored(
        &self,
        query: &str,
        is_admissible: &dyn Fn(&VectorId) -> bool,
    ) -> Result<Vec<(VectorId, f64)>, LexicalError> {
        if self.corpus_size == 0 {
            return Ok(Vec::new());
        }
        #[allow(clippy::cast_precision_loss)]
        let average_doc_length = self.total_tokens as f64 / self.corpus_size as f64;
        let txn = self.db.begin_read().map_err(backend)?;
        let postings = txn.open_table(POSTINGS).map_err(backend)?;
        let lengths = txn.open_table(DOC_LENGTHS).map_err(backend)?;

        let mut scores: HashMap<u128, f64> = HashMap::new();
        for term in tokenize(query) {
            let Some(value) = postings.get(term.as_str()).map_err(backend)? else {
                continue;
            };
            let list: Vec<(u128, u32)> = decode(value.value())?;
            let idf = bm25::idf(self.corpus_size, list.len());
            for (id_value, frequency) in list {
                let id = VectorId::from_uuid(Uuid::from_u128(id_value));
                if !is_admissible(&id) {
                    continue;
                }
                let key = id.as_uuid().into_bytes();
                let Some(length_value) = lengths.get(key.as_slice()).map_err(backend)? else {
                    continue;
                };
                let length: u32 = decode(length_value.value())?;
                let contribution = bm25::term_score(frequency, length, average_doc_length, idf);
                *scores.entry(id_value).or_default() += contribution;
            }
        }
        Ok(scores
            .into_iter()
            .filter(|(_, score)| *score > 0.0)
            .map(|(id_value, score)| (VectorId::from_uuid(Uuid::from_u128(id_value)), score))
            .collect())
    }
}

fn apply_delta(value: &mut usize, delta: i64) {
    if delta >= 0 {
        *value += usize::try_from(delta).unwrap_or(0);
    } else {
        *value = value.saturating_sub(usize::try_from(-delta).unwrap_or(0));
    }
}

fn apply_token_delta(value: &mut u64, delta: i64) {
    if delta >= 0 {
        *value += u64::try_from(delta).unwrap_or(0);
    } else {
        *value = value.saturating_sub(u64::try_from(-delta).unwrap_or(0));
    }
}

#[cfg(test)]
mod tests {
    use super::PersistentLexicalIndex;
    use eidosdb_core::VectorId;
    use eidosdb_lexical::{Document, InMemoryLexicalIndex, LexicalIndex};
    use proptest::prelude::*;
    use tempfile::TempDir;

    fn ranked_ids(hits: &[(VectorId, f64)]) -> Vec<VectorId> {
        hits.iter().map(|(id, _)| *id).collect()
    }

    proptest! {
        #[test]
        fn parity_with_in_memory_oracle(
            words in proptest::collection::vec(
                proptest::collection::vec(
                    proptest::sample::select(vec!["fox", "dog", "quick", "lazy", "the", "river"]),
                    1..6,
                ),
                1..12,
            ),
            query_terms in proptest::collection::vec(
                proptest::sample::select(vec!["fox", "dog", "quick", "lazy", "river", "missing"]),
                1..3,
            ),
        ) {
            let dir = TempDir::new().expect("tempdir");
            let mut persistent =
                PersistentLexicalIndex::open(&dir.path().join("lex.redb")).expect("open");
            let mut oracle = InMemoryLexicalIndex::new();

            for tokens in &words {
                let id = VectorId::new();
                let text = tokens.join(" ");
                let document = Document::new(text).expect("non-empty");
                persistent.insert(id, &document).expect("persistent insert");
                oracle.insert(id, &document).expect("oracle insert");
            }

            prop_assert_eq!(persistent.len(), oracle.len());

            let query = query_terms.join(" ");
            let got = persistent.search_text(&query, 100);
            let want = oracle.search_text(&query, 100);
            prop_assert_eq!(ranked_ids(&got), ranked_ids(&want));
            for ((_, g), (_, w)) in got.iter().zip(want.iter()) {
                prop_assert!((g - w).abs() < 1e-9);
            }
        }
    }
}
