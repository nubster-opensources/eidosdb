//! Distance metrics and the normalized similarity score.

/// Similarity metric used to compare two embeddings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Cosine similarity in `[-1, 1]`, higher is closer.
    Cosine,
    /// Raw dot product, higher is closer.
    DotProduct,
    /// Euclidean (L2) distance, returned negated so higher is closer.
    Euclidean,
}

/// A normalized similarity score where a greater value always means a closer match.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score(pub f32);

impl Metric {
    /// Scores two equal-length slices under this metric.
    ///
    /// Higher is always closer, regardless of the metric.
    #[must_use]
    pub fn score(self, a: &[f32], b: &[f32]) -> Score {
        match self {
            Metric::Cosine => Score(cosine(a, b)),
            Metric::DotProduct => Score(dot(a, b)),
            Metric::Euclidean => Score(-euclidean(a, b)),
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot(a, b) / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::Metric;

    #[test]
    fn cosine_of_identical_unit_vectors_is_one() {
        let v = [1.0, 0.0, 0.0];
        assert!((Metric::Cosine.score(&v, &v).0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!(Metric::Cosine.score(&a, &b).0.abs() < 1e-6);
    }

    #[test]
    fn dot_product_matches_manual_computation() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((Metric::DotProduct.score(&a, &b).0 - 32.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_is_negated_so_closer_is_greater() {
        let a = [0.0, 0.0];
        let near = [1.0, 0.0];
        let far = [5.0, 0.0];
        assert!(Metric::Euclidean.score(&a, &near).0 > Metric::Euclidean.score(&a, &far).0);
    }

    #[test]
    fn euclidean_of_identical_vectors_is_zero() {
        let v = [3.0, 4.0];
        assert!(Metric::Euclidean.score(&v, &v).0.abs() < 1e-6);
    }
}
