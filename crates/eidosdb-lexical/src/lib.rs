//! Lexical layer for `EidosDB`: a BM25 inverted index behind the `LexicalIndex`
//! port, a deterministic zero-dependency analyzer, and reciprocal rank fusion.
//!
//! This crate is a peer of the geometric core: it knows ids and text, never
//! vectors or payloads. `fuse_rrf` knows only ranked ids.

mod error;
pub use error::LexicalError;

mod document;
pub use document::Document;

mod analyzer;
pub use analyzer::tokenize;

pub mod bm25;

mod index;
pub use index::LexicalIndex;

mod in_memory;
pub use in_memory::InMemoryLexicalIndex;
