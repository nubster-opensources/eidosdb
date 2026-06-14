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

#[allow(dead_code)]
mod manifest;
