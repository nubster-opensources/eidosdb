//! `HybridQuery` and `Collection::search_hybrid`: a thin orchestration of the
//! dense and lexical channels fused by RRF.

use crate::{Collection, PayloadStore, QueryError, SearchHit};
use eidosdb_core::{Embedding, Metric, Score, VectorId, VectorIndex};
use eidosdb_lexical::{LexicalIndex, fuse_rrf};

/// Default RRF dampening constant.
pub const DEFAULT_RRF_K: f64 = 60.0;
/// Default per-channel over-fetch multiplier.
pub const DEFAULT_OVERFETCH_FACTOR: usize = 4;

/// A hybrid search request. At least one of `vector` or `text` must be set.
pub struct HybridQuery {
    /// Dense channel query; `None` runs lexical-only.
    pub vector: Option<Embedding>,
    /// Lexical channel query; `None` runs dense-only.
    pub text: Option<String>,
    /// Number of hits to return.
    pub k: usize,
    /// Optional payload filter, pushed into both channels.
    pub filter: Option<crate::Filter>,
    /// Dense metric; `None` uses the index default.
    pub metric: Option<Metric>,
    /// RRF dampening constant.
    pub rrf_k: f64,
    /// Per-channel over-fetch multiplier.
    pub overfetch_factor: usize,
}

impl<I: VectorIndex, L: LexicalIndex, P: PayloadStore> Collection<I, L, P> {
    /// Runs a hybrid query. With both channels set, fuses them by RRF; with one
    /// channel, returns it directly; with neither, errors. Hybrid hits carry the
    /// fused RRF score; single-channel hits carry that channel's native score.
    pub fn search_hybrid(&self, query: &HybridQuery) -> Result<Vec<SearchHit>, QueryError> {
        if query.vector.is_none() && query.text.is_none() {
            return Err(QueryError::EmptyQuery);
        }
        let allowed = match &query.filter {
            Some(filter) => Some(self.payloads.matching_ids(&filter.compile())?),
            None => None,
        };
        let is_admissible = |id: &VectorId| allowed.as_ref().is_none_or(|set| set.contains(id));
        let depth = query.k.saturating_mul(query.overfetch_factor);

        let dense: Option<Vec<(VectorId, f64)>> = match &query.vector {
            Some(embedding) => {
                let metric = query.metric.unwrap_or_else(|| self.index.metric());
                if !self.index.supported_metrics().contains(&metric) {
                    return Err(QueryError::UnsupportedMetric(metric));
                }
                let neighbors =
                    self.index
                        .search_filtered(embedding, depth, metric, &is_admissible)?;
                Some(
                    neighbors
                        .into_iter()
                        .map(|n| (n.id, f64::from(n.score.0)))
                        .collect(),
                )
            }
            None => None,
        };

        let lexical: Option<Vec<(VectorId, f64)>> = query.text.as_ref().map(|text| {
            self.lexical
                .search_text_filtered(text, depth, &is_admissible)
        });

        let ranked: Vec<(VectorId, f64)> = match (dense, lexical) {
            (Some(dense), Some(lexical)) => {
                let dense_ids: Vec<VectorId> = dense.iter().map(|(id, _)| *id).collect();
                let lexical_ids: Vec<VectorId> = lexical.iter().map(|(id, _)| *id).collect();
                fuse_rrf(&[dense_ids, lexical_ids], query.rrf_k)
            }
            (Some(single), None) | (None, Some(single)) => single,
            (None, None) => unreachable!("empty query rejected above"),
        };

        let mut hits = Vec::new();
        for (id, score) in ranked.into_iter().take(query.k) {
            let payload = self.payloads.get(&id)?;
            // Scores feed ranking and display only; the f64 to f32 narrowing
            // loses precision well below any tie that matters here.
            #[allow(clippy::cast_possible_truncation)]
            hits.push(SearchHit {
                id,
                score: Score(score as f32),
                payload,
            });
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_OVERFETCH_FACTOR, DEFAULT_RRF_K, HybridQuery};
    use crate::{Collection, InMemoryPayloadStore};
    use eidosdb_core::{Dimension, Embedding, FlatIndex, Metric, VectorId, VectorIndex};
    use eidosdb_lexical::{Document, InMemoryLexicalIndex, LexicalIndex, fuse_rrf};

    fn embedding(values: &[f32]) -> Embedding {
        Embedding::new(values.to_vec()).expect("non-empty")
    }

    fn collection() -> Collection<FlatIndex, InMemoryLexicalIndex, InMemoryPayloadStore> {
        Collection::new(
            FlatIndex::new(Metric::Cosine, Dimension(2)),
            InMemoryLexicalIndex::new(),
            InMemoryPayloadStore::new(),
        )
    }

    fn query(vector: Option<Embedding>, text: Option<&str>, k: usize) -> HybridQuery {
        HybridQuery {
            vector,
            text: text.map(ToString::to_string),
            k,
            filter: None,
            metric: None,
            rrf_k: DEFAULT_RRF_K,
            overfetch_factor: DEFAULT_OVERFETCH_FACTOR,
        }
    }

    #[test]
    fn empty_query_is_rejected() {
        let c = collection();
        assert!(c.search_hybrid(&query(None, None, 5)).is_err());
    }

    #[test]
    fn dense_only_degenerates_to_vector_search() {
        let mut c = collection();
        let near = VectorId::new();
        let far = VectorId::new();
        c.upsert(near, embedding(&[1.0, 0.0]), None, None)
            .expect("near");
        c.upsert(far, embedding(&[-1.0, 0.0]), None, None)
            .expect("far");
        let hits = c
            .search_hybrid(&query(Some(embedding(&[1.0, 0.0])), None, 2))
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, near);
        // Single channel must thread the native cosine score, not an RRF rank
        // score. The aligned vector scores 1.0; the opposite one scores below 0.
        assert!(
            (hits[0].score.0 - 1.0).abs() < 1e-6,
            "native cosine score is threaded through, not RRF"
        );
        assert!(
            hits[1].score.0 < 0.0,
            "opposite vector keeps its negative cosine"
        );
    }

    #[test]
    fn lexical_only_degenerates_to_text_search() {
        let mut c = collection();
        let fox = VectorId::new();
        let dog = VectorId::new();
        c.upsert(
            fox,
            embedding(&[1.0, 0.0]),
            Some(&Document::new("quick fox").expect("d")),
            None,
        )
        .expect("fox");
        c.upsert(
            dog,
            embedding(&[0.0, 1.0]),
            Some(&Document::new("lazy dog").expect("d")),
            None,
        )
        .expect("dog");
        let hits = c
            .search_hybrid(&query(None, Some("fox"), 10))
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, fox);
    }

    #[test]
    fn hybrid_matches_independent_rrf_oracle() {
        let mut c = collection();
        let a = VectorId::new();
        let b = VectorId::new();
        let d = VectorId::new();
        c.upsert(
            a,
            embedding(&[1.0, 0.0]),
            Some(&Document::new("fox").expect("d")),
            None,
        )
        .expect("a");
        c.upsert(
            b,
            embedding(&[0.9, 0.1]),
            Some(&Document::new("dog").expect("d")),
            None,
        )
        .expect("b");
        c.upsert(
            d,
            embedding(&[-1.0, 0.0]),
            Some(&Document::new("fox tale").expect("d")),
            None,
        )
        .expect("d");

        let hits = c
            .search_hybrid(&query(Some(embedding(&[1.0, 0.0])), Some("fox"), 3))
            .expect("search");

        // Oracle: recompute the two channels and fuse them independently.
        let dense: Vec<VectorId> = {
            let neighbors = {
                let index = &c.index;
                index
                    .search_filtered(&embedding(&[1.0, 0.0]), 12, Metric::Cosine, &|_| true)
                    .expect("dense")
            };
            neighbors.into_iter().map(|n| n.id).collect()
        };
        let lexical: Vec<VectorId> = {
            let lexical = &c.lexical;
            lexical
                .search_text("fox", 12)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        };
        let want: Vec<VectorId> = fuse_rrf(&[dense, lexical], 60.0)
            .into_iter()
            .map(|(id, _)| id)
            .take(3)
            .collect();
        let got: Vec<VectorId> = hits.iter().map(|h| h.id).collect();
        assert_eq!(got, want);
    }
}
