//! Query layer for `EidosDB`: typed payloads, a filter AST, and a `Collection`
//! that orchestrates payload-filtered, metric-aware search over a `VectorIndex`.
//!
//! The geometric core stays pure: this crate produces an admissibility predicate
//! from a filter and hands it to `VectorIndex::search_filtered`. Payload semantics
//! never leak into the index.

mod value;
pub use value::{FieldValue, Value};

mod error;
pub use error::{PayloadError, QueryError};

mod payload;
pub use payload::Payload;

mod filter;
pub use filter::{CompiledFilter, Filter};

mod payload_store;
pub use payload_store::{InMemoryPayloadStore, PayloadStore};

mod collection;
pub use collection::{Collection, SearchHit, SearchQuery};

mod hybrid;
pub use hybrid::{DEFAULT_OVERFETCH_FACTOR, DEFAULT_RRF_K, HybridQuery};
