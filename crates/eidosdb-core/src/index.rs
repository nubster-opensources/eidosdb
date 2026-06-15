//! The `VectorIndex` port: a purely geometric nearest-neighbor contract.

use crate::{Dimension, Embedding, IndexError, Metric, Neighbor, VectorId};

/// A vector index: stores embeddings and answers nearest-neighbor queries.
///
/// The port knows only geometry. Payloads, filtering, persistence and transport
/// live in layers above and never leak into this contract.
pub trait VectorIndex {
    /// The default metric this index scores with.
    fn metric(&self) -> Metric;

    /// The metrics this index can score a query with.
    ///
    /// Flat-style indexes that keep raw vectors support every metric; a
    /// graph index built for one metric supports only that one.
    fn supported_metrics(&self) -> &[Metric];

    /// The dimensionality every embedding must match.
    fn dimension(&self) -> Dimension;

    /// Number of stored vectors.
    fn len(&self) -> usize;

    /// Whether the index holds no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Inserts a vector, failing on dimension mismatch or duplicate id.
    fn insert(&mut self, id: VectorId, embedding: Embedding) -> Result<(), IndexError>;

    /// Removes a vector by id, returning whether it was present.
    fn remove(&mut self, id: VectorId) -> Result<bool, IndexError>;

    /// Returns the `k` closest vectors to `query` under the requested `metric`,
    /// restricted to ids for which `is_admissible` returns `true`, sorted by
    /// descending score.
    ///
    /// The predicate keeps the index purely geometric: it sees ids, never
    /// payloads. Implementations that cannot honor `metric` return
    /// [`IndexError::UnsupportedMetric`].
    fn search_filtered(
        &self,
        query: &Embedding,
        k: usize,
        metric: Metric,
        is_admissible: &dyn Fn(&VectorId) -> bool,
    ) -> Result<Vec<Neighbor>, IndexError>;

    /// Returns the `k` closest vectors to `query` under the default metric,
    /// considering every stored vector.
    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<Neighbor>, IndexError> {
        self.search_filtered(query, k, self.metric(), &|_| true)
    }
}
