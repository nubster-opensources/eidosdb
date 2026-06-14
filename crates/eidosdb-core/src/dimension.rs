//! The dimensionality of an embedding space.

/// Number of components in an embedding vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dimension(pub usize);

impl Dimension {
    /// Returns the dimension as a `usize`.
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::Dimension;

    #[test]
    fn exposes_inner_value() {
        assert_eq!(Dimension(768).get(), 768);
    }
}
