//! Core domain of `EidosDB`: the `VectorIndex` port and the Flat exact adapter.

mod dimension;
pub use dimension::Dimension;

mod vector_id;
pub use vector_id::VectorId;

mod error;
pub use error::IndexError;

mod embedding;
pub use embedding::Embedding;

mod metric;
pub use metric::{Metric, Score};
