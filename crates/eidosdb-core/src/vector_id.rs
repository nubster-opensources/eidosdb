//! Stable identifier for a stored vector.

use uuid::Uuid;

/// Unique identifier of a vector, backed by a time-ordered UUID v7.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorId(Uuid);

impl VectorId {
    /// Generates a fresh, time-ordered identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Wraps an existing UUID.
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Returns the underlying UUID.
    #[must_use]
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for VectorId {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::VectorId;
    use uuid::Uuid;

    #[test]
    fn round_trips_through_uuid() {
        let uuid = Uuid::now_v7();
        assert_eq!(VectorId::from_uuid(uuid).as_uuid(), uuid);
    }

    #[test]
    fn fresh_ids_are_distinct() {
        assert_ne!(VectorId::new(), VectorId::new());
    }
}
