//! The `VectorIndex` port: a purely geometric nearest-neighbor contract.

use crate::{Dimension, Embedding, IndexError, Metric, Neighbor, VectorId};

/// A vector index: stores embeddings and answers nearest-neighbor queries.
///
/// The port knows only geometry. Payloads, filtering, persistence and transport
/// live in layers above and never leak into this contract.
pub trait VectorIndex {
    /// The metric this index scores with.
    fn metric(&self) -> Metric;

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

    /// Returns the `k` closest vectors to `query`, sorted by descending score.
    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<Neighbor>, IndexError>;
}
