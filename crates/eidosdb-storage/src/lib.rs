//! Durable storage for `EidosDB`.
//!
//! This crate makes the Flat index persistent: a redb catalog holds the manifest
//! and the `VectorId -> slot` table, while an append-only segment file stores the
//! raw `f32` vectors. The segment is memory-mapped read-only for zero-copy scans.
//!
//! # Unsafe policy
//!
//! Unlike the rest of the workspace (`unsafe_code = "deny"`), this crate allows
//! `unsafe`. The single use is the read-only `memmap2::Mmap::map` of the segment
//! file in `segment`; its safety argument is documented there. No other `unsafe`
//! is permitted.

mod error;
pub use error::StorageError;

mod catalog;
mod manifest;
mod segment;

mod persistent_flat;
pub use persistent_flat::PersistentFlatIndex;

mod persistent_payload;
pub use persistent_payload::PersistentPayloadStore;

mod persistent_lexical;
pub use persistent_lexical::PersistentLexicalIndex;
