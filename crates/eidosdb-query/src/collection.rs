//! `Collection`: binds a `VectorIndex`, a `LexicalIndex`, and a `PayloadStore`
//! into one queryable unit that runs filtered, metric-aware search by
//! pre-filtering ids before the geometric scan.

use crate::{Filter, Payload, PayloadStore, QueryError};
use eidosdb_core::{Embedding, Metric, Score, VectorId, VectorIndex};
use eidosdb_lexical::{Document, LexicalIndex};
use std::collections::HashSet;

/// A search request against a `Collection`.
pub struct SearchQuery {
    /// The query embedding.
    pub embedding: Embedding,
    /// Number of neighbors to return.
    pub k: usize,
    /// Metric to score with; `None` uses the index default metric.
    pub metric: Option<Metric>,
    /// Optional payload filter; `None` considers every vector.
    pub filter: Option<Filter>,
}

/// A search result: a scored id with its payload hydrated.
pub struct SearchHit {
    /// Identifier of the matched vector.
    pub id: VectorId,
    /// Similarity score (higher is closer).
    pub score: Score,
    /// The vector's payload, if any.
    pub payload: Option<Payload>,
}

/// Binds a geometric index, a lexical index, and a payload store into one unit.
pub struct Collection<I, L, P> {
    pub(crate) index: I,
    pub(crate) lexical: L,
    pub(crate) payloads: P,
}

impl<I: VectorIndex, L: LexicalIndex, P: PayloadStore> Collection<I, L, P> {
    /// Creates a collection over `index`, `lexical`, and `payloads`.
    pub fn new(index: I, lexical: L, payloads: P) -> Self {
        Self {
            index,
            lexical,
            payloads,
        }
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the collection holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Inserts or overwrites a vector, its optional document, and its optional
    /// payload. A `None` document removes any prior lexical entry for `id`.
    pub fn upsert(
        &mut self,
        id: VectorId,
        embedding: Embedding,
        document: Option<&Document>,
        payload: Option<Payload>,
    ) -> Result<(), QueryError> {
        let _ = self.index.remove(id)?;
        self.index.insert(id, embedding)?;
        match document {
            Some(document) => self.lexical.insert(id, document)?,
            None => self.lexical.remove(&id)?,
        }
        match payload {
            Some(payload) => self.payloads.set(id, payload)?,
            None => {
                self.payloads.remove(&id)?;
            }
        }
        Ok(())
    }

    /// Returns a mutable reference to the underlying vector index.
    ///
    /// Used by maintenance operations such as compaction that operate directly
    /// on the index layer. Lexical and payload stores are not exposed here
    /// because compaction of those components is out of scope for V0.1.
    pub fn index_mut(&mut self) -> &mut I {
        &mut self.index
    }

    /// Deletes a vector, its document, and its payload, returning whether the
    /// vector was present.
    pub fn delete(&mut self, id: &VectorId) -> Result<bool, QueryError> {
        let removed = self.index.remove(*id)?;
        self.lexical.remove(id)?;
        self.payloads.remove(id)?;
        Ok(removed)
    }

    /// Runs a filtered, metric-aware search and hydrates payloads onto the hits.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchHit>, QueryError> {
        let metric = query.metric.unwrap_or_else(|| self.index.metric());
        if !self.index.supported_metrics().contains(&metric) {
            return Err(QueryError::UnsupportedMetric(metric));
        }
        let neighbors = match &query.filter {
            Some(filter) => {
                let compiled = filter.compile();
                let allowed: HashSet<VectorId> = self.payloads.matching_ids(&compiled)?;
                self.index
                    .search_filtered(&query.embedding, query.k, metric, &|id| {
                        allowed.contains(id)
                    })?
            }
            None => self
                .index
                .search_filtered(&query.embedding, query.k, metric, &|_| true)?,
        };
        let mut hits = Vec::with_capacity(neighbors.len());
        for neighbor in neighbors {
            let payload = self.payloads.get(&neighbor.id)?;
            hits.push(SearchHit {
                id: neighbor.id,
                score: neighbor.score,
                payload,
            });
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::{Collection, SearchQuery};
    use crate::{FieldValue, Filter, InMemoryPayloadStore, Payload, Value};
    use eidosdb_core::{Dimension, Embedding, FlatIndex, Metric, VectorId};
    use eidosdb_lexical::InMemoryLexicalIndex;
    use std::collections::BTreeMap;

    fn embedding(values: &[f32]) -> Embedding {
        Embedding::new(values.to_vec()).expect("non-empty")
    }

    fn payload(source: &str) -> Payload {
        let mut map = BTreeMap::new();
        map.insert(
            "source".to_string(),
            FieldValue::Scalar(Value::Text(source.into())),
        );
        Payload::new(map).expect("valid")
    }

    fn collection() -> Collection<FlatIndex, InMemoryLexicalIndex, InMemoryPayloadStore> {
        Collection::new(
            FlatIndex::new(Metric::Cosine, Dimension(2)),
            InMemoryLexicalIndex::new(),
            InMemoryPayloadStore::new(),
        )
    }

    #[test]
    fn search_without_filter_returns_all_ranked() {
        let mut c = collection();
        let near = VectorId::new();
        let far = VectorId::new();
        c.upsert(near, embedding(&[1.0, 0.0]), None, None)
            .expect("near");
        c.upsert(far, embedding(&[-1.0, 0.0]), None, None)
            .expect("far");
        let hits = c
            .search(&SearchQuery {
                embedding: embedding(&[1.0, 0.0]),
                k: 2,
                metric: None,
                filter: None,
            })
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, near);
    }

    #[test]
    fn filter_excludes_non_matching_payloads() {
        let mut c = collection();
        let wiki = VectorId::new();
        let blog = VectorId::new();
        c.upsert(wiki, embedding(&[1.0, 0.0]), None, Some(payload("wiki")))
            .expect("wiki");
        c.upsert(blog, embedding(&[1.0, 0.0]), None, Some(payload("blog")))
            .expect("blog");
        let hits = c
            .search(&SearchQuery {
                embedding: embedding(&[1.0, 0.0]),
                k: 10,
                metric: None,
                filter: Some(Filter::Eq("source".into(), Value::Text("wiki".into()))),
            })
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, wiki);
        assert_eq!(hits[0].payload, Some(payload("wiki")));
    }

    #[test]
    fn upsert_overwrites_vector_and_payload() {
        let mut c = collection();
        let id = VectorId::new();
        c.upsert(id, embedding(&[1.0, 0.0]), None, Some(payload("old")))
            .expect("first");
        c.upsert(id, embedding(&[0.0, 1.0]), None, Some(payload("new")))
            .expect("second");
        assert_eq!(c.len(), 1);
        let hits = c
            .search(&SearchQuery {
                embedding: embedding(&[0.0, 1.0]),
                k: 1,
                metric: None,
                filter: None,
            })
            .expect("search");
        assert_eq!(hits[0].payload, Some(payload("new")));
    }

    #[test]
    fn delete_removes_from_both_stores() {
        let mut c = collection();
        let id = VectorId::new();
        c.upsert(id, embedding(&[1.0, 0.0]), None, Some(payload("x")))
            .expect("insert");
        assert!(c.delete(&id).expect("delete"));
        assert!(!c.delete(&id).expect("delete again"));
        assert_eq!(c.len(), 0);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn filtered_search_equals_brute_force_oracle(
            rows in proptest::collection::vec(
                (proptest::collection::vec(-3.0_f32..3.0, 2), 0u8..3),
                1..25,
            ),
            query in proptest::collection::vec(-3.0_f32..3.0, 2),
        ) {
            let mut c = collection();
            let mut expected: Vec<(VectorId, [f32; 2], u8)> = Vec::new();
            for (v, bucket) in &rows {
                let id = VectorId::new();
                let mut map = BTreeMap::new();
                map.insert("bucket".to_string(),
                    FieldValue::Scalar(Value::Integer(i64::from(*bucket))));
                c.upsert(id, embedding(v), None, Some(Payload::new(map).expect("valid")))
                    .expect("upsert");
                expected.push((id, [v[0], v[1]], *bucket));
            }
            let filter = Filter::Eq("bucket".into(), Value::Integer(1));
            let hits = c.search(&SearchQuery {
                embedding: embedding(&query),
                k: rows.len(),
                metric: None,
                filter: Some(filter),
            }).expect("search");

            let q = embedding(&query);
            let mut oracle: Vec<(VectorId, f32)> = expected
                .iter()
                .filter(|(_, _, bucket)| *bucket == 1)
                .map(|(id, v, _)| (*id, Metric::Cosine.score(q.as_slice(), v).0))
                .collect();
            oracle.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let got_ids: Vec<VectorId> = hits.iter().map(|h| h.id).collect();
            let want_ids: Vec<VectorId> = oracle.iter().map(|(id, _)| *id).collect();
            prop_assert_eq!(got_ids, want_ids);
        }
    }
}
