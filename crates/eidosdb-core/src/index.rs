//! The `VectorIndex` port: a purely geometric nearest-neighbor contract.

use crate::{Dimension, Embedding, IndexError, Metric, Neighbor, VectorId};

/// A vector index: stores embeddings and answers nearest-neighbor queries.
///
/// The port knows only geometry. Payloads, filtering, persistence and transport
/// live in layers above and never leak into this contract.
///
/// # Stability contract
///
/// This trait is frozen as of 0.1: any method added afterward carries a
/// default implementation, so an external implementation compiled against
/// 0.1 keeps compiling against a later 0.x without changes. `search` above
/// is the existing example of this shape.
///
/// The trait also stays object-safe: `search_filtered` takes a predicate as
/// `&dyn Fn(&VectorId) -> bool` rather than a generic parameter, precisely so
/// callers can hold a `&dyn VectorIndex`. The `_vector_index_is_object_safe`
/// compile-time test below guards this; a method that broke object safety
/// (for example, one taking `self` by value or introducing a generic
/// parameter) would fail that test to compile.
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

#[cfg(test)]
mod tests {
    use super::VectorIndex;

    /// Compile-time guard: `VectorIndex` must stay object-safe (any method
    /// added after 0.1 needs a default implementation) so callers can hold a
    /// `&dyn VectorIndex`.
    fn _vector_index_is_object_safe(_: &dyn VectorIndex) {}
}
