//! A search result: an identified vector and its score.

use crate::{Score, VectorId};

/// One nearest-neighbor result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Identifier of the matched vector.
    pub id: VectorId,
    /// Similarity score (higher is closer).
    pub score: Score,
}

#[cfg(test)]
mod tests {
    use super::Neighbor;
    use crate::{Score, VectorId};

    #[test]
    fn carries_id_and_score() {
        let id = VectorId::new();
        let neighbor = Neighbor {
            id,
            score: Score(0.9),
        };
        assert_eq!(neighbor.id, id);
        assert!((neighbor.score.0 - 0.9).abs() < 1e-6);
    }
}
