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

impl PersistentFlatIndex {
    fn values_at(&self, slot: u64, mapped: u64) -> Result<&[f32], IndexError> {
        if slot < mapped {
            self.segment
                .record(slot)
                .ok_or_else(|| IndexError::Backend(format!("missing mapped slot {slot}")))
        } else {
            let index = usize::try_from(slot - mapped)
                .map_err(|_| IndexError::Backend("tail index overflow".to_string()))?;
            self.tail
                .get(index)
                .map(|(_, emb)| emb.as_slice())
                .ok_or_else(|| IndexError::Backend(format!("missing tail slot {slot}")))
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

    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<Neighbor>, IndexError> {
        if query.dimension() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension.get(),
                got: query.dimension().get(),
            });
        }
        let live = self.catalog.live_slots().map_err(IndexError::from)?;
        let mapped = self.segment.mapped_records();
        let mut scored: Vec<Neighbor> = Vec::with_capacity(live.len());
        for (id, slot) in live {
            let values = self.values_at(slot, mapped)?;
            scored.push(Neighbor {
                id,
                score: self.metric.score(query.as_slice(), values),
            });
        }
        scored.sort_by(|a, b| {
            b.score
                .0
                .total_cmp(&a.score.0)
                .then_with(|| a.id.cmp(&b.id))
        });
        scored.truncate(k);
        Ok(scored)
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

    #[test]
    fn search_returns_closest_first() {
        let dir = tempdir().expect("tempdir");
        let mut index = PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2))
            .expect("open");
        let near = VectorId::new();
        let far = VectorId::new();
        index.insert(near, embedding(&[1.0, 0.0])).expect("near");
        index.insert(far, embedding(&[-1.0, 0.0])).expect("far");
        let results = index.search(&embedding(&[1.0, 0.0]), 2).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, near);
        assert_eq!(results[1].id, far);
    }

    #[test]
    fn search_rejects_dimension_mismatch() {
        let dir = tempdir().expect("tempdir");
        let index = PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(3))
            .expect("open");
        assert_eq!(
            index.search(&embedding(&[1.0, 0.0]), 1),
            Err(IndexError::DimensionMismatch { expected: 3, got: 2 })
        );
    }

    #[test]
    fn matches_flat_oracle() {
        use eidosdb_core::FlatIndex;
        let dir = tempdir().expect("tempdir");
        let mut persistent =
            PersistentFlatIndex::open(dir.path(), Metric::Euclidean, Dimension(3)).expect("open");
        let mut oracle = FlatIndex::new(Metric::Euclidean, Dimension(3));
        let vectors = [
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],
            [0.4, 0.4, 0.4],
            [0.0, 1.0, 0.0],
        ];
        for v in vectors {
            let id = VectorId::new();
            persistent.insert(id, embedding(&v)).expect("persistent insert");
            oracle.insert(id, embedding(&v)).expect("oracle insert");
        }
        let query = embedding(&[0.3, 0.3, 0.3]);
        assert_eq!(
            persistent.search(&query, 4).expect("p"),
            oracle.search(&query, 4).expect("o")
        );
    }
}
