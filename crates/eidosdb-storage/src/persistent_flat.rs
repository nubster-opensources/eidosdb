//! `PersistentFlatIndex`: a durable, disk-backed exact index implementing the
//! `VectorIndex` port from `eidosdb-core`.

use crate::catalog::Catalog;
use crate::error::StorageError;
use crate::manifest::{Manifest, FORMAT_VERSION};
use crate::segment::Segment;
use eidosdb_core::{Dimension, Embedding, IndexError, Metric, Neighbor, VectorId, VectorIndex};
use std::path::{Path, PathBuf};

const SEGMENT_FILE: &str = "vectors.seg";
const CATALOG_FILE: &str = "meta.redb";

/// A durable exact index: redb metadata plus a memory-mapped vector segment.
pub struct PersistentFlatIndex {
    // read by compact and snapshot
    #[allow(dead_code)]
    dir: PathBuf,
    catalog: Catalog,
    segment: Segment,
    metric: Metric,
    dimension: Dimension,
    record_count: u64,
    live_count: usize,
    tail: Vec<(VectorId, Embedding)>,
}

impl PersistentFlatIndex {
    /// Opens an index at `dir`, creating it if absent.
    ///
    /// On first call the directory is initialised with a fresh `redb` catalog and
    /// an empty segment file. On subsequent calls the stored manifest is validated
    /// against `metric` and `dimension`; a mismatch returns
    /// [`StorageError::FormatMismatch`].
    pub fn open(dir: &Path, metric: Metric, dimension: Dimension) -> Result<Self, StorageError> {
        std::fs::create_dir_all(dir)?;
        let catalog_path = dir.join(CATALOG_FILE);
        let segment_path = dir.join(SEGMENT_FILE);
        let usize_dim = dimension.get();

        if catalog_path.exists() {
            let (catalog, manifest) = Catalog::open(&catalog_path)?;
            if manifest.format_version != FORMAT_VERSION {
                return Err(StorageError::FormatMismatch(format!(
                    "catalog version {} != {FORMAT_VERSION}",
                    manifest.format_version
                )));
            }
            let expected_dim = u32::try_from(usize_dim)
                .map_err(|_| StorageError::FormatMismatch("dimension exceeds u32".to_string()))?;
            if manifest.dimension != expected_dim || manifest.metric != metric {
                return Err(StorageError::FormatMismatch(
                    "open parameters differ from stored manifest".to_string(),
                ));
            }
            let segment = Segment::open(&segment_path, metric, usize_dim, manifest.record_count)?;
            let live_count = catalog.live_count()?;
            Ok(Self {
                dir: dir.to_path_buf(),
                catalog,
                segment,
                metric,
                dimension,
                record_count: manifest.record_count,
                live_count,
                tail: Vec::new(),
            })
        } else {
            let expected_dim = u32::try_from(usize_dim)
                .map_err(|_| StorageError::FormatMismatch("dimension exceeds u32".to_string()))?;
            let manifest = Manifest {
                format_version: FORMAT_VERSION,
                dimension: expected_dim,
                metric,
                record_count: 0,
            };
            let catalog = Catalog::create(&catalog_path, &manifest)?;
            let segment = Segment::create(&segment_path, metric, usize_dim)?;
            Ok(Self {
                dir: dir.to_path_buf(),
                catalog,
                segment,
                metric,
                dimension,
                record_count: 0,
                live_count: 0,
                tail: Vec::new(),
            })
        }
    }
}

impl VectorIndex for PersistentFlatIndex {
    fn metric(&self) -> Metric {
        self.metric
    }

    fn dimension(&self) -> Dimension {
        self.dimension
    }

    fn len(&self) -> usize {
        self.live_count
    }

    fn insert(&mut self, id: VectorId, embedding: Embedding) -> Result<(), IndexError> {
        if embedding.dimension() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension.get(),
                got: embedding.dimension().get(),
            });
        }
        if self.catalog.contains(id)? {
            return Err(IndexError::DuplicateId(id));
        }
        let slot = self.record_count;
        // Data before pointer: durable bytes first, then commit the slot.
        self.segment.append(embedding.as_slice())?;
        self.catalog.insert_slot(id, slot, slot + 1)?;
        self.record_count = slot + 1;
        self.live_count += 1;
        self.tail.push((id, embedding));
        Ok(())
    }

    fn remove(&mut self, _id: VectorId) -> Result<bool, IndexError> {
        Err(IndexError::Backend("remove not implemented yet".to_string()))
    }

    fn search(&self, _query: &Embedding, _k: usize) -> Result<Vec<Neighbor>, IndexError> {
        Err(IndexError::Backend("search not implemented yet".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::PersistentFlatIndex;
    use eidosdb_core::{Dimension, Embedding, IndexError, Metric, VectorId, VectorIndex};
    use tempfile::tempdir;

    fn embedding(values: &[f32]) -> Embedding {
        Embedding::new(values.to_vec()).expect("non-empty")
    }

    #[test]
    fn new_index_is_empty() {
        let dir = tempdir().expect("tempdir");
        let index = PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(3))
            .expect("open");
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.metric(), Metric::Cosine);
        assert_eq!(index.dimension(), Dimension(3));
    }

    #[test]
    fn reopen_rejects_dimension_change() {
        let dir = tempdir().expect("tempdir");
        PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(3)).expect("create");
        assert!(PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(4)).is_err());
    }

    #[test]
    fn insert_increases_len() {
        let dir = tempdir().expect("tempdir");
        let mut index = PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2))
            .expect("open");
        index.insert(VectorId::new(), embedding(&[1.0, 0.0])).expect("insert");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn insert_rejects_dimension_mismatch() {
        let dir = tempdir().expect("tempdir");
        let mut index = PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(3))
            .expect("open");
        assert_eq!(
            index.insert(VectorId::new(), embedding(&[1.0, 0.0])),
            Err(IndexError::DimensionMismatch { expected: 3, got: 2 })
        );
    }

    #[test]
    fn insert_rejects_duplicate_id() {
        let dir = tempdir().expect("tempdir");
        let mut index = PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2))
            .expect("open");
        let id = VectorId::new();
        index.insert(id, embedding(&[1.0, 0.0])).expect("first");
        assert_eq!(
            index.insert(id, embedding(&[0.0, 1.0])),
            Err(IndexError::DuplicateId(id))
        );
    }
}
