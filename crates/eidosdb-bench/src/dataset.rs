//! Synthetic dataset generation for benchmarks.

use eidosdb_core::{Embedding, VectorId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A reproducible synthetic dataset: stored points plus query vectors.
pub struct Dataset {
    /// Points to load into an index.
    pub points: Vec<(VectorId, Embedding)>,
    /// Query embeddings to search with.
    pub queries: Vec<Embedding>,
}

/// Generates `point_count` points and `query_count` queries of `dimension`
/// components, drawn uniformly from `[-1, 1]`. The embedding vectors are
/// reproducible for a given `seed`; the `VectorId`s are freshly generated each
/// run and are not seeded.
#[must_use]
pub fn generate(seed: u64, dimension: usize, point_count: usize, query_count: usize) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let sample = |rng: &mut StdRng| {
        let values: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Embedding::new(values).expect("dimension is non-zero")
    };
    let points = (0..point_count)
        .map(|_| (VectorId::new(), sample(&mut rng)))
        .collect();
    let queries = (0..query_count).map(|_| sample(&mut rng)).collect();
    Dataset { points, queries }
}

#[cfg(test)]
mod tests {
    use super::generate;

    #[test]
    fn produces_requested_shapes() {
        let dataset = generate(42, 8, 100, 5);
        assert_eq!(dataset.points.len(), 100);
        assert_eq!(dataset.queries.len(), 5);
        assert_eq!(dataset.points[0].1.as_slice().len(), 8);
    }

    #[test]
    fn same_seed_is_reproducible() {
        let a = generate(7, 4, 10, 2);
        let b = generate(7, 4, 10, 2);
        assert_eq!(a.points[0].1.as_slice(), b.points[0].1.as_slice());
    }
}
