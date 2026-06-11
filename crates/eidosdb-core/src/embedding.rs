//! A validated, owned embedding vector.

use crate::{Dimension, IndexError};

/// An embedding: a non-empty vector of `f32` components.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Builds an embedding, rejecting an empty vector.
    pub fn new(values: Vec<f32>) -> Result<Self, IndexError> {
        if values.is_empty() {
            return Err(IndexError::EmptyEmbedding);
        }
        Ok(Self(values))
    }

    /// Returns the dimensionality of this embedding.
    #[must_use]
    pub fn dimension(&self) -> Dimension {
        Dimension(self.0.len())
    }

    /// Returns the components as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::Embedding;
    use crate::{Dimension, IndexError};

    #[test]
    fn rejects_empty_vector() {
        assert_eq!(Embedding::new(vec![]), Err(IndexError::EmptyEmbedding));
    }

    #[test]
    fn reports_its_dimension() {
        let embedding = Embedding::new(vec![1.0, 2.0, 3.0]).expect("non-empty");
        assert_eq!(embedding.dimension(), Dimension(3));
    }

    #[test]
    fn exposes_components_as_slice() {
        let embedding = Embedding::new(vec![1.0, -1.0]).expect("non-empty");
        assert_eq!(embedding.as_slice(), &[1.0, -1.0]);
    }
}
