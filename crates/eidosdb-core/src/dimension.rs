//! The dimensionality of an embedding space.

use serde::{Deserialize, Serialize};

/// Number of components in an embedding vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dimension(pub usize);

/// Errors returned when validating a [`Dimension`].
///
/// Not implemented yet: this is the target shape for the upcoming
/// `Dimension::new` constructor; call sites still build `Dimension` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DimensionError {
    /// The requested dimension was zero.
    #[error("dimension must be greater than zero")]
    Zero,
    /// The requested dimension does not fit in a `u32`.
    #[error("dimension {0} exceeds the supported maximum of {max}", max = u32::MAX)]
    TooLarge(usize),
}

impl Dimension {
    /// Returns the dimension as a `usize`.
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }

    /// Validates and builds a dimension from a component count.
    ///
    /// Not implemented yet: the constructor and the underlying private
    /// representation land in the Building phase of this lot.
    ///
    /// # Errors
    ///
    /// Returns [`DimensionError::Zero`] when `components` is zero, or
    /// [`DimensionError::TooLarge`] when it exceeds `u32::MAX`.
    pub fn new(_components: usize) -> Result<Self, DimensionError> {
        todo!()
    }
}

impl TryFrom<u32> for Dimension {
    type Error = DimensionError;

    /// Not implemented yet: lands alongside the private representation.
    fn try_from(_value: u32) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl From<Dimension> for u32 {
    /// Not implemented yet: lands alongside the private representation.
    fn from(_dimension: Dimension) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::{Dimension, DimensionError};

    #[test]
    fn exposes_inner_value() {
        assert_eq!(Dimension(768).get(), 768);
    }

    #[test]
    fn new_rejects_zero() {
        assert_eq!(Dimension::new(0), Err(DimensionError::Zero));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn new_rejects_too_large() {
        let overflow = u32::MAX as usize + 1;
        assert_eq!(
            Dimension::new(overflow),
            Err(DimensionError::TooLarge(overflow))
        );
    }

    #[test]
    fn new_accepts_a_realistic_dimension() {
        assert_eq!(Dimension::new(768).expect("768 is valid").get(), 768);
    }

    #[test]
    fn json_zero_is_rejected() {
        let result: Result<Dimension, _> = serde_json::from_str("0");
        assert!(result.is_err());
    }

    #[test]
    fn json_realistic_value_is_accepted() {
        let dimension: Dimension = serde_json::from_str("768").expect("768 deserializes");
        assert_eq!(dimension.get(), 768);
    }
}
