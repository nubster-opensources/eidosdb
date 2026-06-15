//! `PersistentFlatIndex`: a durable, disk-backed exact index implementing the
//! `VectorIndex` port from `eidosdb-core`.

use crate::catalog::Catalog;
use crate::error::StorageError;
use crate::manifest::{FORMAT_VERSION, Manifest};
use crate::segment::Segment;
use eidosdb_core::{Dimension, Embedding, IndexError, Metric, Neighbor, VectorId, VectorIndex};
use std::path::{Path, PathBuf};

const SEGMENT_FILE: &str = "vectors.seg";
const CATALOG_FILE: &str = "meta.redb";

/// A durable exact index: redb metadata plus a memory-mapped vector segment.
pub struct PersistentFlatIndex {
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
    /// Inserts many vectors with a single fsync and a single catalog commit.
    ///
    /// Validates every item first (dimension and duplicate id) so the batch is
    /// all-or-nothing: nothing is written if any item is invalid.
    pub fn insert_batch(&mut self, items: Vec<(VectorId, Embedding)>) -> Result<(), IndexError> {
        if items.is_empty() {
            return Ok(());
        }
        // Validate up front.
        let mut seen = std::collections::HashSet::new();
        for (id, embedding) in &items {
            if embedding.dimension() != self.dimension {
                return Err(IndexError::DimensionMismatch {
                    expected: self.dimension.get(),
                    got: embedding.dimension().get(),
                });
            }
            if !seen.insert(*id) || self.catalog.contains(*id)? {
                return Err(IndexError::DuplicateId(*id));
            }
        }
        // Append all records in one flat write, then commit slots.
        let mut flat: Vec<f32> = Vec::with_capacity(items.len() * self.dimension.get());
        let mut slots: Vec<(VectorId, u64)> = Vec::with_capacity(items.len());
        let mut next = self.record_count;
        for (id, embedding) in &items {
            flat.extend_from_slice(embedding.as_slice());
            slots.push((*id, next));
            next += 1;
        }
        self.segment.append(&flat)?;
        self.catalog.insert_slots(&slots, next)?;
        self.record_count = next;
        self.live_count += items.len();
        self.tail.extend(items);
        Ok(())
    }

    /// Remaps the segment so all durable records are visible through the map and
    /// clears the RAM tail. Durability is unaffected (inserts already fsync).
    pub fn checkpoint(&mut self) -> Result<(), IndexError> {
        self.segment
            .remap(self.record_count)
            .map_err(IndexError::from)?;
        self.tail.clear();
        Ok(())
    }

    fn values_at_storage(&self, slot: u64, mapped: u64) -> Result<&[f32], StorageError> {
        if slot < mapped {
            self.segment
                .record(slot)
                .ok_or_else(|| StorageError::Corruption(format!("missing mapped slot {slot}")))
        } else {
            let index = usize::try_from(slot - mapped)
                .map_err(|_| StorageError::Corruption("tail index overflow".to_string()))?;
            self.tail
                .get(index)
                .map(|(_, embedding)| embedding.as_slice())
                .ok_or_else(|| StorageError::Corruption(format!("missing tail slot {slot}")))
        }
    }

    /// Writes a consistent, self-contained copy of the store into `dest`.
    ///
    /// `dest` is a directory reopenable by [`PersistentFlatIndex::open`]. Because
    /// every insert is already durable (fsync + commit), rebuilding the catalog
    /// and copying the segment up to the watermark yields a consistent image.
    pub fn snapshot(&self, dest: &Path) -> Result<(), IndexError> {
        self.snapshot_inner(dest).map_err(IndexError::from)
    }

    fn snapshot_inner(&self, dest: &Path) -> Result<(), StorageError> {
        std::fs::create_dir_all(dest)?;
        // The live catalog file is locked by redb while open, so it cannot be byte
        // copied. Rebuild an equivalent catalog at the destination from the live
        // set instead, using redb's own API.
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            dimension: u32::try_from(self.dimension.get())
                .map_err(|_| StorageError::FormatMismatch("dimension exceeds u32".to_string()))?,
            metric: self.metric,
            record_count: self.record_count,
        };
        let dest_catalog = Catalog::create(&dest.join(CATALOG_FILE), &manifest)?;
        let live = self.catalog.live_slots()?;
        dest_catalog.replace_slots(&live, self.record_count)?;
        drop(dest_catalog);

        let stride = u64::try_from(self.dimension.get())
            .map_err(|_| StorageError::Corruption("dimension overflow".to_string()))?
            * 4;
        let valid_len = crate::segment::HEADER_LEN as u64 + self.record_count * stride;
        let bytes = std::fs::read(self.dir.join(SEGMENT_FILE))?;
        let end = usize::try_from(valid_len)
            .map_err(|_| StorageError::Corruption("segment length overflow".to_string()))?;
        let slice = bytes.get(..end).ok_or_else(|| {
            StorageError::Corruption("segment shorter than watermark".to_string())
        })?;
        std::fs::write(dest.join(SEGMENT_FILE), slice)?;
        Ok(())
    }

    /// Rewrites the store keeping only live records, reclaiming dead space.
    ///
    /// Rewrites the segment file in place (truncate to header, re-append live
    /// records) and replaces the `redb` slot table in one transaction. No file
    /// rename, so it is safe regardless of held file handles.
    pub fn compact(&mut self) -> Result<(), IndexError> {
        self.compact_inner().map_err(IndexError::from)
    }

    fn compact_inner(&mut self) -> Result<(), StorageError> {
        let mut live = self.catalog.live_slots()?;
        live.sort_by_key(|(_, slot)| *slot);
        let mapped = self.segment.mapped_records();
        let dim = self.dimension.get();
        let mut flat: Vec<f32> = Vec::with_capacity(live.len() * dim);
        let mut new_slots: Vec<(VectorId, u64)> = Vec::with_capacity(live.len());
        for (new_index, (id, slot)) in live.iter().enumerate() {
            let values = self.values_at_storage(*slot, mapped)?;
            flat.extend_from_slice(values);
            let new_slot = u64::try_from(new_index)
                .map_err(|_| StorageError::Corruption("slot index overflow".to_string()))?;
            new_slots.push((*id, new_slot));
        }
        let count = u64::try_from(new_slots.len())
            .map_err(|_| StorageError::Corruption("record count overflow".to_string()))?;
        self.segment.truncate_to_empty()?;
        self.segment.append(&flat)?;
        self.segment.remap(count)?;
        self.catalog.replace_slots(&new_slots, count)?;
        self.record_count = count;
        self.tail.clear();
        self.live_count = new_slots.len();
        Ok(())
    }

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

    fn remove(&mut self, id: VectorId) -> Result<bool, IndexError> {
        let existed = self.catalog.remove_slot(id).map_err(IndexError::from)?;
        if existed {
            self.live_count -= 1;
        }
        Ok(existed)
    }

    fn supported_metrics(&self) -> &[Metric] {
        &[Metric::Cosine, Metric::DotProduct, Metric::Euclidean]
    }

    fn search_filtered(
        &self,
        query: &Embedding,
        k: usize,
        metric: Metric,
        is_admissible: &dyn Fn(&VectorId) -> bool,
    ) -> Result<Vec<Neighbor>, IndexError> {
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
            if !is_admissible(&id) {
                continue;
            }
            let values = self.values_at(slot, mapped)?;
            scored.push(Neighbor {
                id,
                score: metric.score(query.as_slice(), values),
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
        let index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(3)).expect("open");
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
        let mut index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
        index
            .insert(VectorId::new(), embedding(&[1.0, 0.0]))
            .expect("insert");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn insert_rejects_dimension_mismatch() {
        let dir = tempdir().expect("tempdir");
        let mut index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(3)).expect("open");
        assert_eq!(
            index.insert(VectorId::new(), embedding(&[1.0, 0.0])),
            Err(IndexError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn insert_rejects_duplicate_id() {
        let dir = tempdir().expect("tempdir");
        let mut index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
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
        let mut index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
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
        let index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(3)).expect("open");
        assert_eq!(
            index.search(&embedding(&[1.0, 0.0]), 1),
            Err(IndexError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn remove_tombstones_and_excludes_from_search() {
        let dir = tempdir().expect("tempdir");
        let mut index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
        let keep = VectorId::new();
        let drop = VectorId::new();
        index.insert(keep, embedding(&[1.0, 0.0])).expect("keep");
        index.insert(drop, embedding(&[1.0, 0.0])).expect("drop");
        assert_eq!(index.remove(drop), Ok(true));
        assert_eq!(index.remove(drop), Ok(false));
        assert_eq!(index.len(), 1);
        let results = index.search(&embedding(&[1.0, 0.0]), 10).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, keep);
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
            persistent
                .insert(id, embedding(&v))
                .expect("persistent insert");
            oracle.insert(id, embedding(&v)).expect("oracle insert");
        }
        let query = embedding(&[0.3, 0.3, 0.3]);
        assert_eq!(
            persistent.search(&query, 4).expect("p"),
            oracle.search(&query, 4).expect("o")
        );
    }

    #[test]
    fn checkpoint_keeps_results_and_clears_tail() {
        let dir = tempdir().expect("tempdir");
        let mut index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
        let a = VectorId::new();
        index.insert(a, embedding(&[1.0, 0.0])).expect("insert");
        index.checkpoint().expect("checkpoint");
        // After checkpoint, the same vector is served from the mmap, not the tail.
        let results = index.search(&embedding(&[1.0, 0.0]), 1).expect("search");
        assert_eq!(results[0].id, a);
        // A further insert still works (slot beyond the remapped region).
        let b = VectorId::new();
        index.insert(b, embedding(&[0.9, 0.1])).expect("insert b");
        assert_eq!(
            index
                .search(&embedding(&[1.0, 0.0]), 2)
                .expect("search")
                .len(),
            2
        );
    }

    #[test]
    fn insert_batch_matches_individual_inserts() {
        let dir_a = tempdir().expect("a");
        let dir_b = tempdir().expect("b");
        let items: Vec<(VectorId, Embedding)> = (0_u8..5)
            .map(|i| (VectorId::new(), embedding(&[f32::from(i), 1.0])))
            .collect();

        let mut batched =
            PersistentFlatIndex::open(dir_a.path(), Metric::Cosine, Dimension(2)).expect("a");
        batched.insert_batch(items.clone()).expect("batch");

        let mut single =
            PersistentFlatIndex::open(dir_b.path(), Metric::Cosine, Dimension(2)).expect("b");
        for (id, e) in items {
            single.insert(id, e).expect("single");
        }

        let query = embedding(&[2.0, 1.0]);
        assert_eq!(batched.len(), single.len());
        assert_eq!(
            batched.search(&query, 5).expect("ba"),
            single.search(&query, 5).expect("si")
        );
    }

    #[test]
    fn insert_batch_rejects_dimension_mismatch() {
        let dir = tempdir().expect("tempdir");
        let mut index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
        let items = vec![(VectorId::new(), embedding(&[1.0, 2.0, 3.0]))];
        assert_eq!(
            index.insert_batch(items),
            Err(IndexError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        );
    }

    #[test]
    fn reopen_preserves_data() {
        let dir = tempdir().expect("tempdir");
        let id = VectorId::new();
        {
            let mut index =
                PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
            index.insert(id, embedding(&[1.0, 0.0])).expect("insert");
        }
        let index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("reopen");
        assert_eq!(index.len(), 1);
        let results = index.search(&embedding(&[1.0, 0.0]), 1).expect("search");
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn compact_reclaims_space_and_preserves_live() {
        let dir = tempdir().expect("tempdir");
        let keep = VectorId::new();
        let noise1 = VectorId::new();
        let noise2 = VectorId::new();

        let mut index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
        index
            .insert(noise1, embedding(&[0.5, 0.5]))
            .expect("noise1");
        index.insert(keep, embedding(&[1.0, 0.0])).expect("keep");
        index
            .insert(noise2, embedding(&[0.5, 0.5]))
            .expect("noise2");
        // Checkpoint so all records are mapped (and the test exercises the mmap path).
        index.checkpoint().expect("checkpoint");

        let seg_path = dir.path().join("vectors.seg");
        let size_before = std::fs::metadata(&seg_path).expect("meta").len();

        index.remove(noise1).expect("remove noise1");
        index.remove(noise2).expect("remove noise2");
        index.compact().expect("compact");

        let size_after = std::fs::metadata(&seg_path).expect("meta after").len();
        assert!(
            size_after < size_before,
            "segment should shrink after compaction"
        );

        assert_eq!(index.len(), 1);
        let results = index.search(&embedding(&[1.0, 0.0]), 10).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, keep);

        // Reopen: persisted state must be consistent.
        drop(index);
        let reopened =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("reopen");
        assert_eq!(reopened.len(), 1);
        let results2 = reopened
            .search(&embedding(&[1.0, 0.0]), 10)
            .expect("search after reopen");
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].id, keep);
    }

    #[test]
    fn snapshot_is_reopenable_and_identical() {
        let dir = tempdir().expect("tempdir");
        let snap = tempdir().expect("snap");
        let mut index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
        let ids: Vec<VectorId> = (0..6).map(|_| VectorId::new()).collect();
        for (i, id) in ids.iter().enumerate() {
            let v = f32::from(u8::try_from(i).expect("small"));
            index.insert(*id, embedding(&[v, 1.0])).expect("insert");
        }
        index.checkpoint().expect("checkpoint");
        index.snapshot(snap.path()).expect("snapshot");

        let restored = PersistentFlatIndex::open(snap.path(), Metric::Cosine, Dimension(2))
            .expect("reopen snap");
        let query = embedding(&[3.0, 1.0]);
        assert_eq!(restored.len(), index.len());
        assert_eq!(
            restored.search(&query, 6).expect("r"),
            index.search(&query, 6).expect("i")
        );
    }

    #[test]
    fn reopen_after_orphan_tail_is_consistent() {
        use std::fs::OpenOptions;
        use std::io::Write;

        let dir = tempdir().expect("tempdir");
        {
            let mut index =
                PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
            index
                .insert(VectorId::new(), embedding(&[1.0, 0.0]))
                .expect("insert");
        }
        // Simulate a crash mid-append: extra bytes past the committed watermark.
        {
            let mut file = OpenOptions::new()
                .append(true)
                .open(dir.path().join("vectors.seg"))
                .expect("open seg");
            file.write_all(&[0xAB; 8]).expect("write orphan bytes");
        }
        let index =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("reopen");
        assert_eq!(index.len(), 1, "watermark ignores orphan bytes");
        assert_eq!(
            index
                .search(&embedding(&[1.0, 0.0]), 10)
                .expect("search")
                .len(),
            1
        );
    }

    #[test]
    fn search_filtered_matches_flat_oracle_with_predicate() {
        use eidosdb_core::FlatIndex;
        let dir = tempdir().expect("tempdir");
        let mut persistent =
            PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(2)).expect("open");
        let mut oracle = FlatIndex::new(Metric::Cosine, Dimension(2));
        let mut kept = Vec::new();
        for i in 0..6 {
            let id = VectorId::new();
            let v = f32::from(u8::try_from(i).expect("small"));
            persistent.insert(id, embedding(&[v, 1.0])).expect("p");
            oracle.insert(id, embedding(&[v, 1.0])).expect("o");
            if i % 2 == 0 {
                kept.push(id);
            }
        }
        let allow = move |id: &VectorId| kept.contains(id);
        let query = embedding(&[2.0, 1.0]);
        let got = persistent
            .search_filtered(&query, 6, Metric::Cosine, &allow)
            .expect("p search");
        let want = oracle
            .search_filtered(&query, 6, Metric::Cosine, &allow)
            .expect("o search");
        assert_eq!(got, want);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn persistent_equals_oracle_under_inserts_and_removes(
            vectors in proptest::collection::vec(
                proptest::collection::vec(-5.0_f32..5.0, 3),
                1..25,
            ),
            remove_every in 2usize..5,
        ) {
            use eidosdb_core::FlatIndex;
            let dir = tempdir().expect("tempdir");
            let mut persistent =
                PersistentFlatIndex::open(dir.path(), Metric::Cosine, Dimension(3)).expect("open");
            let mut oracle = FlatIndex::new(Metric::Cosine, Dimension(3));

            let mut ids = Vec::new();
            for (i, v) in vectors.iter().enumerate() {
                let id = VectorId::new();
                let e = embedding(v);
                persistent.insert(id, e.clone()).expect("p insert");
                oracle.insert(id, e).expect("o insert");
                ids.push(id);
                if i % remove_every == 0 {
                    persistent.remove(id).expect("p remove");
                    oracle.remove(id).expect("o remove");
                }
            }
            // Exercise the remap path mid-stream.
            persistent.checkpoint().expect("checkpoint");

            let query = embedding(&[1.0, 1.0, 1.0]);
            prop_assert_eq!(
                persistent.search(&query, ids.len()).expect("p search"),
                oracle.search(&query, ids.len()).expect("o search")
            );
        }
    }
}
