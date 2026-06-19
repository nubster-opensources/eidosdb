//! Sovereign in-house HNSW (Hierarchical Navigable Small World) index for
//! `EidosDB`. Implements the `VectorIndex` port from `eidosdb-core`.
//!
//! This crate is a peer of the geometric core. It introduces no `unsafe` code:
//! the only `unsafe` in the workspace remains the read-only `Mmap::map` in
//! `eidosdb-storage`.

mod config;
pub use config::{DEFAULT_SEED, HnswConfig};

pub(crate) mod rng;
