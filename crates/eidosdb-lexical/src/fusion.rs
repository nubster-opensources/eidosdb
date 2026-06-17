//! Reciprocal Rank Fusion over ranked id lists.

use eidosdb_core::VectorId;
use std::collections::HashMap;

/// Fuses several rankings into one. For each id, sums `1 / (k + rank)` over the
/// rankings where it appears, with `rank` 1-based (the first element is rank 1).
/// Results are sorted by descending fused score, ties broken by ascending id.
#[must_use]
pub fn fuse_rrf(rankings: &[Vec<VectorId>], k: f64) -> Vec<(VectorId, f64)> {
    let mut scores: HashMap<VectorId, f64> = HashMap::new();
    for ranking in rankings {
        for (position, id) in ranking.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let rank = (position + 1) as f64;
            *scores.entry(*id).or_default() += 1.0 / (k + rank);
        }
    }
    let mut fused: Vec<(VectorId, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    fused
}

#[cfg(test)]
mod tests {
    use super::fuse_rrf;
    use eidosdb_core::VectorId;
    use uuid::Uuid;

    fn id(n: u128) -> VectorId {
        VectorId::from_uuid(Uuid::from_u128(n))
    }

    #[test]
    fn fuses_two_rankings_by_reciprocal_rank() {
        let (a, b, c) = (id(1), id(2), id(3));
        // dense: a, b, c   lexical: a, c
        let fused = fuse_rrf(&[vec![a, b, c], vec![a, c]], 60.0);
        let order: Vec<VectorId> = fused.iter().map(|(id, _)| *id).collect();
        assert_eq!(order, vec![a, c, b]);
        // a appears first in both: 1/61 + 1/61.
        assert!((fused[0].1 - (1.0 / 61.0 + 1.0 / 61.0)).abs() < 1e-12);
    }

    #[test]
    fn equal_scores_break_ties_by_ascending_id() {
        let (a, b) = (id(1), id(2));
        // Symmetric rankings: a and b earn identical fused scores.
        let fused = fuse_rrf(&[vec![a, b], vec![b, a]], 60.0);
        let order: Vec<VectorId> = fused.iter().map(|(id, _)| *id).collect();
        assert_eq!(order, vec![a, b]);
    }

    #[test]
    fn empty_rankings_fuse_to_nothing() {
        assert!(fuse_rrf(&[], 60.0).is_empty());
    }
}
