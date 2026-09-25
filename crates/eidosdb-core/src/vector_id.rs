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

impl From<Uuid> for VectorId {
    /// Not implemented yet: equivalent to [`VectorId::from_uuid`], lands in
    /// the Building phase alongside the other conversions.
    fn from(_uuid: Uuid) -> Self {
        todo!()
    }
}

impl From<VectorId> for Uuid {
    /// Not implemented yet: equivalent to [`VectorId::as_uuid`], lands in
    /// the Building phase alongside the other conversions.
    fn from(_id: VectorId) -> Self {
        todo!()
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

    #[test]
    fn from_uuid_round_trips_via_from() {
        let uuid = Uuid::now_v7();
        let id: VectorId = VectorId::from(uuid);
        let back: Uuid = Uuid::from(id);
        assert_eq!(back, uuid);
    }

    #[test]
    fn accepts_and_restores_a_uuid_v4() {
        // A v4 UUID built by hand (version nibble = 4): no need for the `v4`
        // feature, only the bit pattern matters to VectorId.
        let v4 = Uuid::from_u128(0x1234_5678_1234_4234_9234_5678_9abc_def0);
        let id: VectorId = VectorId::from(v4);
        let back: Uuid = Uuid::from(id);
        assert_eq!(back, v4);
    }
}
