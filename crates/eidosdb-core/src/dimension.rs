//! The dimensionality of an embedding space.

use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};

/// Number of components in an embedding vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct Dimension(NonZeroU32);

/// Errors returned when validating a [`Dimension`].
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
    ///
    /// The underlying value is always a valid `u32`, and every target this
    /// crate supports has a `usize` at least as wide as `u32`, so the
    /// fallback branch below is unreachable in practice.
    #[must_use]
    pub fn get(self) -> usize {
        usize::try_from(self.0.get()).unwrap_or(usize::MAX)
    }

    /// Validates and builds a dimension from a component count.
    ///
    /// # Errors
    ///
    /// Returns [`DimensionError::Zero`] when `components` is zero, or
    /// [`DimensionError::TooLarge`] when it exceeds `u32::MAX`.
    pub fn new(components: usize) -> Result<Self, DimensionError> {
        let raw = u32::try_from(components).map_err(|_| DimensionError::TooLarge(components))?;
        NonZeroU32::new(raw).map(Self).ok_or(DimensionError::Zero)
    }
}

impl TryFrom<u32> for Dimension {
    type Error = DimensionError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        NonZeroU32::new(value).map(Self).ok_or(DimensionError::Zero)
    }
}

impl From<Dimension> for u32 {
    fn from(dimension: Dimension) -> Self {
        dimension.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::{Dimension, DimensionError};

    #[test]
    fn exposes_inner_value() {
        assert_eq!(Dimension::new(42).expect("42 is valid").get(), 42);
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
