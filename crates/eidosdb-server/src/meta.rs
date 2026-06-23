//! Persistence of collection metadata on disk.
//!
//! Each collection directory contains a `collection.meta` file that stores
//! the [`CollectionMeta`] struct serialised as pretty-printed JSON.

use std::path::Path;

use eidosdb_core::{Dimension, Metric};
use eidosdb_hnsw::HnswConfig;
use eidosdb_proto::convert::IndexTypeChoice;
use serde::{Deserialize, Serialize};

use crate::error::ServerError;

/// Metadata describing a collection: the metric space, dimensionality,
/// index algorithm and (optionally) HNSW tuning parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionMeta {
    /// Name of the collection (used as the directory name on disk).
    pub name: String,
    /// Distance metric used by this collection.
    pub metric: Metric,
    /// Embedding dimensionality.
    pub dimension: Dimension,
    /// Index algorithm to use.
    pub index_type: IndexTypeChoice,
    /// HNSW tuning parameters; `None` when `index_type` is [`IndexTypeChoice::Flat`].
    pub hnsw: Option<HnswConfig>,
}

/// File name used to persist collection metadata inside a collection directory.
const META_FILE: &str = "collection.meta";

/// Writes `meta` to `dir/collection.meta` as pretty-printed JSON.
///
/// # Errors
///
/// Returns [`ServerError::Serde`] if serialisation fails, or [`ServerError::Io`]
/// if the file cannot be written.
pub fn write_meta(dir: &Path, meta: &CollectionMeta) -> Result<(), ServerError> {
    let bytes = serde_json::to_vec_pretty(meta).map_err(|e| ServerError::Serde(e.to_string()))?;
    std::fs::write(dir.join(META_FILE), bytes).map_err(|e| ServerError::Io(e.to_string()))
}

/// Reads and deserialises the `collection.meta` file from `dir`.
///
/// # Errors
///
/// Returns [`ServerError::Io`] if the file cannot be read, or [`ServerError::Serde`]
/// if deserialisation fails.
pub fn read_meta(dir: &Path) -> Result<CollectionMeta, ServerError> {
    let bytes = std::fs::read(dir.join(META_FILE)).map_err(|e| ServerError::Io(e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| ServerError::Serde(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidosdb_core::{Dimension, Metric};
    use eidosdb_hnsw::HnswConfig;
    use eidosdb_proto::convert::IndexTypeChoice;

    #[test]
    fn meta_round_trips_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let meta = CollectionMeta {
            name: "notes".to_string(),
            metric: Metric::Cosine,
            dimension: Dimension(3),
            index_type: IndexTypeChoice::Hnsw,
            hnsw: Some(HnswConfig::default()),
        };
        write_meta(dir.path(), &meta).expect("write");
        assert_eq!(read_meta(dir.path()).expect("read"), meta);
    }

    #[test]
    fn read_missing_meta_is_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(read_meta(dir.path()).is_err());
    }
}
