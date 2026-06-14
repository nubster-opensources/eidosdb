//! Recall@k of a candidate result set against a ground-truth oracle.

use eidosdb_core::Neighbor;

/// Fraction of the oracle's top-k ids that also appear in the candidate's top-k.
///
/// Returns `1.0` when the oracle returned nothing (nothing to miss).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn recall_at_k(oracle: &[Neighbor], candidate: &[Neighbor]) -> f32 {
    if oracle.is_empty() {
        return 1.0;
    }
    let found = oracle
        .iter()
        .filter(|truth| candidate.iter().any(|c| c.id == truth.id))
        .count();
    found as f32 / oracle.len() as f32
}

#[cfg(test)]
mod tests {
    use super::recall_at_k;
    use eidosdb_core::{Neighbor, Score, VectorId};

    fn neighbor(id: VectorId) -> Neighbor {
        Neighbor {
            id,
            score: Score(1.0),
        }
    }

    #[test]
    fn identical_sets_score_one() {
        let ids: Vec<VectorId> = (0..3).map(|_| VectorId::new()).collect();
        let truth: Vec<Neighbor> = ids.iter().map(|id| neighbor(*id)).collect();
        assert!((recall_at_k(&truth, &truth) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn half_overlap_scores_half() {
        let shared = VectorId::new();
        let only_truth = VectorId::new();
        let only_candidate = VectorId::new();
        let truth = vec![neighbor(shared), neighbor(only_truth)];
        let candidate = vec![neighbor(shared), neighbor(only_candidate)];
        assert!((recall_at_k(&truth, &candidate) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn empty_oracle_scores_one() {
        assert!((recall_at_k(&[], &[]) - 1.0).abs() < 1e-6);
    }
}
