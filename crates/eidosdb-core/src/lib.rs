//! Core domain of `EidosDB`: the `VectorIndex` port and the Flat exact adapter.
//!
//! Types that carry an invariant never derive a bare `Deserialize`; see
//! `CONTRIBUTING.md` and [`Dimension`] for the validating-constructor pattern.

mod dimension;
pub use dimension::{Dimension, DimensionError};

mod vector_id;
pub use vector_id::VectorId;

mod error;
pub use error::IndexError;

mod embedding;
pub use embedding::Embedding;

mod metric;
pub use metric::{Metric, Score};

mod neighbor;
pub use neighbor::Neighbor;

mod index;
pub use index::VectorIndex;

mod flat;
pub use flat::FlatIndex;
