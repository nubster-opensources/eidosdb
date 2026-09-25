//! Stable identifier for a stored vector.
//!
//! # Identifier policy
//!
//! [`VectorId`] wraps an opaque 128-bit UUID. Any UUID version is accepted:
//! [`VectorId::new`] always generates a time-ordered version 7 UUID, but a
//! `VectorId` built from a UUID of another version (v4, v5, ...) via
//! [`VectorId::from_uuid`] or [`From<Uuid>`] works identically everywhere
//! else; nothing in `EidosDB` inspects the version nibble.
//!
//! There is no native support for a non-UUID external identifier (a `u64`
//! primary key, a natural string key, ...). To key a point on such an
//! identifier while still storing a `VectorId`, derive a stable UUID from it
//! client-side with [`Uuid::new_v5`], under a namespace UUID your application
//! owns:
//!
//! ```rust,ignore
//! use eidosdb_core::VectorId;
//! use uuid::Uuid;
//!
//! // A namespace UUID your application picks once and keeps fixed.
//! const MY_NAMESPACE: Uuid = Uuid::from_u128(0x1234_5678_9abc_def0_1234_5678_9abc_def0);
//!
//! fn id_for_external_key(key: u64) -> VectorId {
//!     let uuid = Uuid::new_v5(&MY_NAMESPACE, &key.to_be_bytes());
//!     VectorId::from(uuid)
//! }
//! ```
//!
//! The same recipe applies to a text key by hashing its UTF-8 bytes instead
//! of `key.to_be_bytes()`. `Uuid::new_v5` is deterministic: the same
//! `(namespace, bytes)` pair always derives the same `VectorId`, so an
//! external identifier can be recovered without a side table.
//!
//! This example is `ignore`d rather than run as a doc-test: it needs the
//! `uuid` crate's `v5` feature, which this workspace does not enable (only
//! `v7`, for [`VectorId::new`]).
//!
//! The on-disk format and the wire protocol both store the raw 128 bits of
//! the UUID; neither changes with this policy.

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
    /// Equivalent to [`VectorId::from_uuid`]: any UUID version is accepted.
    fn from(uuid: Uuid) -> Self {
        Self::from_uuid(uuid)
    }
}

impl From<VectorId> for Uuid {
    /// Equivalent to [`VectorId::as_uuid`].
    fn from(id: VectorId) -> Self {
        id.as_uuid()
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
