//! `FlatIndex`: exact brute-force nearest-neighbor search, the recall oracle.

use crate::{Dimension, Embedding, IndexError, Metric, Neighbor, VectorId, VectorIndex};

/// Exact index: compares a query against every stored vector. O(n) per query.
///
/// Correct by construction, so it serves as the ground-truth oracle that
/// approximate indexes are measured against.
pub struct FlatIndex {
    metric: Metric,
    dimension: Dimension,
    points: Vec<(VectorId, Embedding)>,
}

impl FlatIndex {
    /// Creates an empty index for `dimension`-sized embeddings scored by `metric`.
    #[must_use]
    pub fn new(metric: Metric, dimension: Dimension) -> Self {
        Self {
            metric,
            dimension,
            points: Vec::new(),
        }
    }
}

impl VectorIndex for FlatIndex {
    fn metric(&self) -> Metric {
        self.metric
    }

    fn dimension(&self) -> Dimension {
        self.dimension
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn insert(&mut self, id: VectorId, embedding: Embedding) -> Result<(), IndexError> {
        if embedding.dimension() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension.get(),
                got: embedding.dimension().get(),
            });
        }
        if self.points.iter().any(|(existing, _)| *existing == id) {
            return Err(IndexError::DuplicateId(id));
        }
        self.points.push((id, embedding));
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<bool, IndexError> {
        if let Some(position) = self.points.iter().position(|(existing, _)| *existing == id) {
            self.points.remove(position);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn supported_metrics(&self) -> &[Metric] {
        &[Metric::Cosine, Metric::DotProduct, Metric::Euclidean]
    }

    fn search_filtered(
        &self,
        query: &Embedding,
        k: usize,
        metric: Metric,
        is_admissible: &dyn Fn(&VectorId) -> bool,
    ) -> Result<Vec<Neighbor>, IndexError> {
        if query.dimension() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension.get(),
                got: query.dimension().get(),
            });
        }
        let mut scored: Vec<Neighbor> = self
            .points
            .iter()
            .filter(|(id, _)| is_admissible(id))
            .map(|(id, embedding)| Neighbor {
                id: *id,
                score: metric.score(query.as_slice(), embedding.as_slice()),
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .0
                .total_cmp(&a.score.0)
                .then_with(|| a.id.cmp(&b.id))
        });
        scored.truncate(k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::FlatIndex;
    use crate::{Dimension, Embedding, IndexError, Metric, VectorId, VectorIndex};

    fn embedding(values: &[f32]) -> Embedding {
        Embedding::new(values.to_vec()).expect("non-empty")
    }

    #[test]
    fn new_index_is_empty() {
        let index = FlatIndex::new(Metric::Cosine, Dimension(2));
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn insert_rejects_dimension_mismatch() {
        let mut index = FlatIndex::new(Metric::Cosine, Dimension(3));
        let err = index.insert(VectorId::new(), embedding(&[1.0, 0.0]));
        assert_eq!(
            err,
            Err(IndexError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn insert_rejects_duplicate_id() {
        let mut index = FlatIndex::new(Metric::Cosine, Dimension(2));
        let id = VectorId::new();
        index
            .insert(id, embedding(&[1.0, 0.0]))
            .expect("first insert");
        assert_eq!(
            index.insert(id, embedding(&[0.0, 1.0])),
            Err(IndexError::DuplicateId(id))
        );
    }

    #[test]
    fn remove_reports_presence() {
        let mut index = FlatIndex::new(Metric::Cosine, Dimension(2));
        let id = VectorId::new();
        index.insert(id, embedding(&[1.0, 0.0])).expect("insert");
        assert_eq!(index.remove(id), Ok(true));
        assert_eq!(index.remove(id), Ok(false));
        assert!(index.is_empty());
    }

    #[test]
    fn search_rejects_dimension_mismatch() {
        let index = FlatIndex::new(Metric::Cosine, Dimension(3));
        assert_eq!(
            index.search(&embedding(&[1.0, 0.0]), 1),
            Err(IndexError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn search_returns_closest_first() {
        let mut index = FlatIndex::new(Metric::Cosine, Dimension(2));
        let near = VectorId::new();
        let far = VectorId::new();
        index
            .insert(near, embedding(&[1.0, 0.0]))
            .expect("insert near");
        index
            .insert(far, embedding(&[-1.0, 0.0]))
            .expect("insert far");
        let results = index.search(&embedding(&[1.0, 0.0]), 2).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, near);
        assert_eq!(results[1].id, far);
    }

    #[test]
    fn search_truncates_to_k() {
        let mut index = FlatIndex::new(Metric::Euclidean, Dimension(1));
        for value in [0.0_f32, 1.0, 2.0, 3.0] {
            index
                .insert(VectorId::new(), embedding(&[value]))
                .expect("insert");
        }
        let results = index.search(&embedding(&[0.0]), 2).expect("search");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn a_vector_is_its_own_nearest_neighbor() {
        let mut index = FlatIndex::new(Metric::Euclidean, Dimension(3));
        let target = VectorId::new();
        index
            .insert(target, embedding(&[0.5, 0.5, 0.5]))
            .expect("insert target");
        for _ in 0..10 {
            index
                .insert(VectorId::new(), embedding(&[1.0, 1.0, 1.0]))
                .expect("insert noise");
        }
        let results = index
            .search(&embedding(&[0.5, 0.5, 0.5]), 1)
            .expect("search");
        assert_eq!(results[0].id, target);
    }

    #[test]
    fn ties_are_broken_by_ascending_id() {
        let mut index = FlatIndex::new(Metric::DotProduct, Dimension(2));
        let first = VectorId::new();
        let second = VectorId::new();
        // Identical embeddings -> identical score for any query -> a tie.
        index
            .insert(first, embedding(&[1.0, 0.0]))
            .expect("insert first");
        index
            .insert(second, embedding(&[1.0, 0.0]))
            .expect("insert second");
        let results = index.search(&embedding(&[1.0, 0.0]), 2).expect("search");
        let lower = first.min(second);
        let higher = first.max(second);
        assert_eq!(results[0].id, lower);
        assert_eq!(results[1].id, higher);
    }

    #[test]
    fn supported_metrics_lists_all_three() {
        let index = FlatIndex::new(Metric::Cosine, Dimension(2));
        let metrics = index.supported_metrics();
        assert!(metrics.contains(&Metric::Cosine));
        assert!(metrics.contains(&Metric::DotProduct));
        assert!(metrics.contains(&Metric::Euclidean));
        assert_eq!(metrics.len(), 3);
    }

    #[test]
    fn search_filtered_excludes_non_admissible_ids() {
        let mut index = FlatIndex::new(Metric::Cosine, Dimension(2));
        let kept = VectorId::new();
        let blocked = VectorId::new();
        index.insert(kept, embedding(&[1.0, 0.0])).expect("kept");
        index
            .insert(blocked, embedding(&[1.0, 0.0]))
            .expect("blocked");
        let results = index
            .search_filtered(&embedding(&[1.0, 0.0]), 10, Metric::Cosine, &|id| {
                *id == kept
            })
            .expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, kept);
    }

    #[test]
    fn search_filtered_honors_requested_metric() {
        let mut cosine_index = FlatIndex::new(Metric::Cosine, Dimension(2));
        let mut euclidean_oracle = FlatIndex::new(Metric::Euclidean, Dimension(2));
        let points = [[0.1_f32, 0.9], [0.8, 0.2], [0.5, 0.5]];
        for p in points {
            let id = VectorId::new();
            cosine_index.insert(id, embedding(&p)).expect("c");
            euclidean_oracle.insert(id, embedding(&p)).expect("e");
        }
        let query = embedding(&[0.4, 0.6]);
        let filtered = cosine_index
            .search_filtered(&query, 3, Metric::Euclidean, &|_| true)
            .expect("filtered");
        let oracle = euclidean_oracle.search(&query, 3).expect("oracle");
        assert_eq!(filtered, oracle);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn top_k_is_sorted_descending(
            values in proptest::collection::vec(
                proptest::collection::vec(-10.0_f32..10.0, 4),
                1..30,
            )
        ) {
            let mut index = FlatIndex::new(Metric::DotProduct, Dimension(4));
            for v in &values {
                index.insert(VectorId::new(), embedding(v)).expect("insert");
            }
            let results = index.search(&embedding(&[1.0, 1.0, 1.0, 1.0]), values.len())
                .expect("search");
            for pair in results.windows(2) {
                prop_assert!(pair[0].score.0 >= pair[1].score.0);
            }
        }

        #[test]
        fn top_k_sorted_descending_for_every_metric(
            metric_index in 0usize..3,
            values in proptest::collection::vec(
                proptest::collection::vec(-1.0_f32..1.0, 4),
                1..20,
            )
        ) {
            let metric = [Metric::DotProduct, Metric::Cosine, Metric::Euclidean][metric_index];
            let mut index = FlatIndex::new(metric, Dimension(4));
            for v in &values {
                index.insert(VectorId::new(), embedding(v)).expect("insert");
            }
            let results = index.search(&embedding(&[1.0, 1.0, 1.0, 1.0]), values.len())
                .expect("search");
            for pair in results.windows(2) {
                prop_assert!(pair[0].score.0 >= pair[1].score.0);
            }
        }

        #[test]
        #[allow(clippy::cast_precision_loss)]
        fn search_never_returns_more_than_k(
            count in 1usize..30,
            k in 0usize..40,
        ) {
            let mut index = FlatIndex::new(Metric::Cosine, Dimension(2));
            for i in 0..count {
                let angle = i as f32;
                index.insert(VectorId::new(), embedding(&[angle.cos(), angle.sin()]))
                    .expect("insert");
            }
            let results = index.search(&embedding(&[1.0, 0.0]), k).expect("search");
            prop_assert!(results.len() <= k);
            prop_assert!(results.len() <= count);
        }
    }
}
