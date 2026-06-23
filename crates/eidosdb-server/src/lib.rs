//! `eidosdb-server` - gRPC service implementation and collection management for `EidosDB`.
//!
//! This crate hosts the tonic service handlers and the on-disk persistence layer
//! for collection metadata and indexes.

pub mod collection_kind;
pub mod error;
pub mod meta;
