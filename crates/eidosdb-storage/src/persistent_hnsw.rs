//! `PersistentHnswIndex`: durable HNSW backed by redb graph tables and the
//! mmap vector segment from B1.
//!
//! Layout:
//! - `vectors.seg`: raw f32 vectors (one record per node-index, B1 pattern).
//! - `hnsw.redb`: three redb tables:
//!   - `hnsw_adjacency`: `(node_idx: u64, layer: u32)` -> `Vec<u64>` (neighbor indices).
//!   - `hnsw_nodes`: `node_idx: u64` -> `NodeRow { id_bytes, level, tombstone }`.
//!   - `hnsw_meta`: `&str` -> postcard bytes. Keys `"config"` and `"state"`.
//!
//! Reads use the RAM-resident `HnswIndex` graph (memory speed).
//! Writes are write-through: each `insert`/`remove` commits ONE redb transaction
//! covering all touched adjacency rows, node rows, and the `"state"` meta row.
//! `open` reloads the full graph EXACTLY from redb via `HnswIndex::from_snapshot`:
//! no re-insertion, no RNG draw on restore.

use crate::error::StorageError;
use crate::segment::Segment;
use eidosdb_core::{Dimension, Embedding, IndexError, Metric, Neighbor, VectorId, VectorIndex};
use eidosdb_hnsw::{GraphSnapshot, HnswConfig, HnswIndex, SnapshotNode};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---- redb table definitions ----
const ADJ: TableDefinition<&[u8], &[u8]> = TableDefinition::new("hnsw_adjacency");
const NODES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("hnsw_nodes");
const META: TableDefinition<&str, &[u8]> = TableDefinition::new("hnsw_meta");

const META_CONFIG_KEY: &str = "config";
const META_STATE_KEY: &str = "state";
const SEGMENT_FILE: &str = "vectors.seg";
const CATALOG_FILE: &str = "hnsw.redb";

// ---- on-disk row types ----

#[derive(Serialize, Deserialize)]
struct NodeRow {
    id_bytes: [u8; 16],
    level: u32,
    tombstone: bool,
}

#[derive(Serialize, Deserialize)]
struct AdjKey {
    node_idx: u64,
    layer: u32,
}

#[derive(Serialize, Deserialize)]
struct MetaConfig {
    metric_byte: u8,
    dimension: u32,
    m: u64,
    ef_construction: u64,
    ef_search: u64,
    seed: u64,
}

#[derive(Serialize, Deserialize)]
struct MetaState {
    entry_point: Option<u64>,
    entry_level: u32,
    rng_state: u64,
    node_count: u64,
}

fn metric_to_u8(metric: Metric) -> u8 {
    match metric {
        Metric::Cosine => 0,
        Metric::DotProduct => 1,
        Metric::Euclidean => 2,
    }
}

fn metric_from_u8(v: u8) -> Result<Metric, StorageError> {
    match v {
        0 => Ok(Metric::Cosine),
        1 => Ok(Metric::DotProduct),
        2 => Ok(Metric::Euclidean),
        other => Err(StorageError::Corruption(format!(
            "unknown metric byte {other}"
        ))),
    }
}

fn catalog_err<E: std::fmt::Display>(e: E) -> StorageError {
    StorageError::Catalog(e.to_string())
}

fn encode<T: Serialize>(v: &T) -> Result<Vec<u8>, StorageError> {
    postcard::to_allocvec(v).map_err(|e| StorageError::Corruption(e.to_string()))
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, StorageError> {
    postcard::from_bytes(bytes).map_err(|e| StorageError::Corruption(e.to_string()))
}

/// Durable HNSW index. The graph lives in RAM for query-speed reads; writes
/// are committed atomically to redb before returning.
pub struct PersistentHnswIndex {
    graph: HnswIndex,
    db: Database,
    segment: Segment,
    /// Directory path retained for future use (e.g. compaction snapshots,
    /// backup). Not currently read after construction; silenced with leading `_`.
    _dir: PathBuf,
}

impl PersistentHnswIndex {
    /// Creates a new, empty persistent HNSW index at `path`.
    pub fn create(
        path: &Path,
        config: HnswConfig,
        dimension: Dimension,
    ) -> Result<Self, StorageError> {
        std::fs::create_dir_all(path)?;
        let db = Database::create(path.join(CATALOG_FILE)).map_err(catalog_err)?;
        let txn = db.begin_write().map_err(catalog_err)?;
        {
            let _ = txn.open_table(ADJ).map_err(catalog_err)?;
            let _ = txn.open_table(NODES).map_err(catalog_err)?;
            let mut meta_table = txn.open_table(META).map_err(catalog_err)?;
            let cfg = MetaConfig {
                metric_byte: metric_to_u8(config.metric),
                dimension: u32::try_from(dimension.get()).map_err(|_| {
                    StorageError::FormatMismatch("dimension exceeds u32".to_string())
                })?,
                m: u64::try_from(config.m).unwrap_or(u64::MAX),
                ef_construction: u64::try_from(config.ef_construction).unwrap_or(u64::MAX),
                ef_search: u64::try_from(config.ef_search).unwrap_or(u64::MAX),
                seed: config.seed,
            };
            let state = MetaState {
                entry_point: None,
                entry_level: 0,
                rng_state: config.seed,
                node_count: 0,
            };
            meta_table
                .insert(META_CONFIG_KEY, encode(&cfg)?.as_slice())
                .map_err(catalog_err)?;
            meta_table
                .insert(META_STATE_KEY, encode(&state)?.as_slice())
                .map_err(catalog_err)?;
        }
        txn.commit().map_err(catalog_err)?;

        let segment = Segment::create(&path.join(SEGMENT_FILE), config.metric, dimension.get())?;
        Ok(Self {
            graph: HnswIndex::new(config, dimension),
            db,
            segment,
            _dir: path.to_path_buf(),
        })
    }

    /// Opens an existing persistent HNSW index, reloading the graph EXACTLY
    /// from redb via `HnswIndex::from_snapshot`. No re-insertion, no RNG draw.
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        let db = Database::open(path.join(CATALOG_FILE)).map_err(catalog_err)?;

        let (cfg_row, state_row): (MetaConfig, MetaState) = {
            let txn = db.begin_read().map_err(catalog_err)?;
            let table = txn.open_table(META).map_err(catalog_err)?;
            let cfg_bytes = table
                .get(META_CONFIG_KEY)
                .map_err(catalog_err)?
                .ok_or_else(|| StorageError::Corruption("missing hnsw config meta".to_string()))?;
            let state_bytes = table
                .get(META_STATE_KEY)
                .map_err(catalog_err)?
                .ok_or_else(|| StorageError::Corruption("missing hnsw state meta".to_string()))?;
            (decode(cfg_bytes.value())?, decode(state_bytes.value())?)
        };

        let metric = metric_from_u8(cfg_row.metric_byte)?;
        let dim = usize::try_from(cfg_row.dimension)
            .map_err(|_| StorageError::Corruption("dimension exceeds usize".to_string()))?;
        let dimension = Dimension(dim);
        let config = HnswConfig {
            metric,
            m: usize::try_from(cfg_row.m).unwrap_or(16),
            ef_construction: usize::try_from(cfg_row.ef_construction).unwrap_or(200),
            ef_search: usize::try_from(cfg_row.ef_search).unwrap_or(64),
            seed: cfg_row.seed,
        };
        let node_count = usize::try_from(state_row.node_count)
            .map_err(|_| StorageError::Corruption("node_count exceeds usize".to_string()))?;

        let segment = Segment::open(&path.join(SEGMENT_FILE), metric, dim, state_row.node_count)?;

        let snapshot = {
            let txn = db.begin_read().map_err(catalog_err)?;
            let nodes_table = txn.open_table(NODES).map_err(catalog_err)?;
            let adj_table = txn.open_table(ADJ).map_err(catalog_err)?;

            let mut node_rows: Vec<(u64, NodeRow)> = Vec::with_capacity(node_count);
            for entry in nodes_table.iter().map_err(catalog_err)? {
                let (key, value) = entry.map_err(catalog_err)?;
                let node_idx: u64 = decode(key.value())?;
                let row: NodeRow = decode(value.value())?;
                node_rows.push((node_idx, row));
            }
            node_rows.sort_by_key(|(idx, _)| *idx);

            let mut snap_nodes: Vec<SnapshotNode> = Vec::with_capacity(node_rows.len());
            for (node_idx, row) in &node_rows {
                let level = usize::try_from(row.level)
                    .map_err(|_| StorageError::Corruption("node level overflow".to_string()))?;
                let mut neighbors_per_layer: Vec<Vec<u64>> = Vec::with_capacity(level + 1);
                for layer in 0..=level {
                    let adj_key = AdjKey {
                        node_idx: *node_idx,
                        layer: u32::try_from(layer).unwrap_or(u32::MAX),
                    };
                    let encoded_key = encode(&adj_key)?;
                    let neighbors: Vec<u64> = adj_table
                        .get(encoded_key.as_slice())
                        .map_err(catalog_err)?
                        .map(|v| decode::<Vec<u64>>(v.value()))
                        .transpose()?
                        .unwrap_or_default();
                    neighbors_per_layer.push(neighbors);
                }
                snap_nodes.push(SnapshotNode {
                    id_bytes: row.id_bytes,
                    level: row.level,
                    tombstone: row.tombstone,
                    neighbors_per_layer,
                });
            }

            let live_count = snap_nodes.iter().filter(|n| !n.tombstone).count();
            GraphSnapshot {
                nodes: snap_nodes,
                entry_point: state_row.entry_point,
                entry_level: state_row.entry_level,
                rng_state: state_row.rng_state,
                live_count: u64::try_from(live_count).unwrap_or(u64::MAX),
            }
        };

        // The closure borrows `segment` immutably. Once `from_snapshot` returns
        // (synchronously), the borrow ends and `segment` can be moved into `Self`.
        let graph = HnswIndex::from_snapshot(config, dimension, &snapshot, &|idx| {
            segment
                .record(u64::try_from(idx).ok()?)
                .map(<[f32]>::to_vec)
        })
        .map_err(StorageError::Index)?;

        Ok(Self {
            graph,
            db,
            segment,
            _dir: path.to_path_buf(),
        })
    }

    /// Builds the index in RAM via `HnswIndex::bulk_insert` then flushes the
    /// full snapshot to redb in ONE transaction.
    pub fn bulk_load(
        path: &Path,
        config: HnswConfig,
        dimension: Dimension,
        points: impl IntoIterator<Item = (VectorId, Embedding)>,
    ) -> Result<Self, StorageError> {
        let mut index = Self::create(path, config, dimension)?;
        let points: Vec<(VectorId, Embedding)> = points.into_iter().collect();
        for (_, embedding) in &points {
            index.segment.append(embedding.as_slice())?;
        }
        index
            .graph
            .bulk_insert(points)
            .map_err(StorageError::Index)?;
        index.flush_snapshot_to_redb()?;
        Ok(index)
    }

    /// Compacts: rebuilds live-only in RAM via `HnswIndex::compact`, then
    /// rewrites the full snapshot to redb in one transaction.
    pub fn compact(&mut self) -> Result<(), StorageError> {
        self.graph.compact().map_err(StorageError::Index)?;
        self.flush_snapshot_to_redb()
    }

    /// Write-through insert (incremental): appends the vector to the segment,
    /// calls `graph.insert_tracked`, commits ONE redb transaction covering only
    /// the touched node rows, their adjacency rows, and the `"state"` meta row.
    fn write_through_insert(
        &mut self,
        id: VectorId,
        embedding: Embedding,
    ) -> Result<(), StorageError> {
        self.segment.append(embedding.as_slice())?;

        let delta = self
            .graph
            .insert_tracked(id, embedding)
            .map_err(StorageError::Index)?;

        let txn = self.db.begin_write().map_err(catalog_err)?;
        {
            let mut nodes_table = txn.open_table(NODES).map_err(catalog_err)?;
            let mut adj_table = txn.open_table(ADJ).map_err(catalog_err)?;
            let mut meta_table = txn.open_table(META).map_err(catalog_err)?;

            for &node_idx in &delta.touched_nodes {
                if let Some(snap_node) = self.graph.node_snapshot(node_idx) {
                    let row = NodeRow {
                        id_bytes: snap_node.id_bytes,
                        level: snap_node.level,
                        tombstone: snap_node.tombstone,
                    };
                    nodes_table
                        .insert(encode(&node_idx)?.as_slice(), encode(&row)?.as_slice())
                        .map_err(catalog_err)?;
                    for (layer, neighbors) in snap_node.neighbors_per_layer.iter().enumerate() {
                        let adj_key = AdjKey {
                            node_idx,
                            layer: u32::try_from(layer).unwrap_or(u32::MAX),
                        };
                        adj_table
                            .insert(encode(&adj_key)?.as_slice(), encode(neighbors)?.as_slice())
                            .map_err(catalog_err)?;
                    }
                }
            }

            let (entry_point, entry_level, rng_state, node_count) = self.graph.state_meta();
            let state = MetaState {
                entry_point,
                entry_level,
                rng_state,
                node_count,
            };
            meta_table
                .insert(META_STATE_KEY, encode(&state)?.as_slice())
                .map_err(catalog_err)?;
        }
        txn.commit().map_err(catalog_err)?;
        Ok(())
    }

    /// Write-through remove (incremental): tombstones via `graph.remove_tracked`,
    /// commits ONE redb transaction: only the node row + state row.
    fn write_through_remove(&mut self, id: VectorId) -> Result<bool, StorageError> {
        let Some(delta) = self.graph.remove_tracked(id).map_err(StorageError::Index)? else {
            return Ok(false);
        };

        let txn = self.db.begin_write().map_err(catalog_err)?;
        {
            let mut nodes_table = txn.open_table(NODES).map_err(catalog_err)?;
            let mut meta_table = txn.open_table(META).map_err(catalog_err)?;

            for &node_idx in &delta.touched_nodes {
                if let Some(snap_node) = self.graph.node_snapshot(node_idx) {
                    let row = NodeRow {
                        id_bytes: snap_node.id_bytes,
                        level: snap_node.level,
                        tombstone: snap_node.tombstone,
                    };
                    nodes_table
                        .insert(encode(&node_idx)?.as_slice(), encode(&row)?.as_slice())
                        .map_err(catalog_err)?;
                }
            }

            let (entry_point, entry_level, rng_state, node_count) = self.graph.state_meta();
            let state = MetaState {
                entry_point,
                entry_level,
                rng_state,
                node_count,
            };
            meta_table
                .insert(META_STATE_KEY, encode(&state)?.as_slice())
                .map_err(catalog_err)?;
        }
        txn.commit().map_err(catalog_err)?;
        Ok(true)
    }

    /// Rewrites ALL redb tables from the current in-memory snapshot in ONE
    /// transaction. Used ONLY by `bulk_load` and `compact`.
    fn flush_snapshot_to_redb(&mut self) -> Result<(), StorageError> {
        let snap = self.graph.snapshot();
        let txn = self.db.begin_write().map_err(catalog_err)?;
        {
            // Drop and recreate to clear stale rows (same pattern as catalog.rs).
            txn.delete_table(NODES).map_err(catalog_err)?;
            txn.delete_table(ADJ).map_err(catalog_err)?;
            let mut nodes_table = txn.open_table(NODES).map_err(catalog_err)?;
            let mut adj_table = txn.open_table(ADJ).map_err(catalog_err)?;
            let mut meta_table = txn.open_table(META).map_err(catalog_err)?;

            for (node_idx, snap_node) in snap.nodes.iter().enumerate() {
                let nidx = u64::try_from(node_idx).unwrap_or(u64::MAX);
                let row = NodeRow {
                    id_bytes: snap_node.id_bytes,
                    level: snap_node.level,
                    tombstone: snap_node.tombstone,
                };
                nodes_table
                    .insert(encode(&nidx)?.as_slice(), encode(&row)?.as_slice())
                    .map_err(catalog_err)?;
                for (layer, neighbors) in snap_node.neighbors_per_layer.iter().enumerate() {
                    let adj_key = AdjKey {
                        node_idx: nidx,
                        layer: u32::try_from(layer).unwrap_or(u32::MAX),
                    };
                    adj_table
                        .insert(encode(&adj_key)?.as_slice(), encode(neighbors)?.as_slice())
                        .map_err(catalog_err)?;
                }
            }
            let state = MetaState {
                entry_point: snap.entry_point,
                entry_level: snap.entry_level,
                rng_state: snap.rng_state,
                node_count: u64::try_from(snap.nodes.len()).unwrap_or(u64::MAX),
            };
            meta_table
                .insert(META_STATE_KEY, encode(&state)?.as_slice())
                .map_err(catalog_err)?;
        }
        txn.commit().map_err(catalog_err)?;
        Ok(())
    }
}

impl VectorIndex for PersistentHnswIndex {
    fn metric(&self) -> Metric {
        self.graph.metric()
    }

    fn supported_metrics(&self) -> &[Metric] {
        self.graph.supported_metrics()
    }

    fn dimension(&self) -> Dimension {
        self.graph.dimension()
    }

    fn len(&self) -> usize {
        self.graph.len()
    }

    fn insert(&mut self, id: VectorId, embedding: Embedding) -> Result<(), IndexError> {
        self.write_through_insert(id, embedding)
            .map_err(IndexError::from)
    }

    fn remove(&mut self, id: VectorId) -> Result<bool, IndexError> {
        self.write_through_remove(id).map_err(IndexError::from)
    }

    fn search_filtered(
        &self,
        query: &Embedding,
        k: usize,
        metric: Metric,
        is_admissible: &dyn Fn(&VectorId) -> bool,
    ) -> Result<Vec<Neighbor>, IndexError> {
        self.graph.search_filtered(query, k, metric, is_admissible)
    }
}

#[cfg(test)]
mod tests {
    use super::PersistentHnswIndex;
    use eidosdb_core::{Dimension, Embedding, Metric, VectorId, VectorIndex};
    use eidosdb_hnsw::{HnswConfig, HnswIndex};
    use proptest::prelude::*;
    use tempfile::TempDir;

    fn emb(values: &[f32]) -> Embedding {
        Embedding::new(values.to_vec()).expect("non-empty")
    }

    fn cfg() -> HnswConfig {
        HnswConfig {
            metric: Metric::Cosine,
            m: 4,
            ef_construction: 20,
            ef_search: 20,
            seed: 0,
        }
    }

    #[test]
    fn create_open_insert_search() {
        let dir = TempDir::new().expect("tempdir");
        let mut index =
            PersistentHnswIndex::create(dir.path(), cfg(), Dimension(2)).expect("create");
        let near = VectorId::new();
        let far = VectorId::new();
        index.insert(near, emb(&[1.0, 0.0])).expect("near");
        index.insert(far, emb(&[-1.0, 0.0])).expect("far");
        let results = index.search(&emb(&[1.0, 0.0]), 2).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, near);
    }

    #[test]
    fn remove_tombstones_and_excludes_from_search() {
        let dir = TempDir::new().expect("tempdir");
        let mut index =
            PersistentHnswIndex::create(dir.path(), cfg(), Dimension(2)).expect("create");
        let keep = VectorId::new();
        let drop_id = VectorId::new();
        index.insert(keep, emb(&[1.0, 0.0])).expect("keep");
        index.insert(drop_id, emb(&[1.0, 0.0])).expect("drop");
        assert_eq!(index.remove(drop_id), Ok(true));
        assert_eq!(index.remove(drop_id), Ok(false));
        assert_eq!(index.len(), 1);
        let results = index.search(&emb(&[1.0, 0.0]), 10).expect("search");
        assert!(results.iter().all(|r| r.id != drop_id));
    }

    #[test]
    fn close_open_search_cycle() {
        let dir = TempDir::new().expect("tempdir");
        let id = VectorId::new();
        {
            let mut index =
                PersistentHnswIndex::create(dir.path(), cfg(), Dimension(2)).expect("create");
            index.insert(id, emb(&[1.0, 0.0])).expect("insert");
        }
        let index = PersistentHnswIndex::open(dir.path()).expect("reopen");
        assert_eq!(index.len(), 1);
        let results = index.search(&emb(&[1.0, 0.0]), 1).expect("search");
        assert_eq!(results[0].id, id);
    }

    proptest! {
        /// Reload-fidelity: persist -> close -> open -> snapshot must equal pre-close snapshot.
        /// (a) Build persistent, record pre-close `GraphSnapshot` and search results.
        /// (b) Reload, assert search results AND full snapshot equality.
        /// (c) Pure in-memory build must match persistent.
        #[test]
        fn reload_fidelity_and_parity_with_in_memory(
            vectors in proptest::collection::vec(
                proptest::collection::vec(-1.0_f32..1.0, 4),
                1..12,
            ),
        ) {
            let ids: Vec<VectorId> = (0..vectors.len())
                .map(|i| VectorId::from_uuid(uuid::Uuid::from_u128(
                    u128::try_from(i).expect("index fits u128")
                )))
                .collect();
            let base_cfg = HnswConfig {
                metric: Metric::Cosine,
                m: 4,
                ef_construction: 20,
                ef_search: 20,
                seed: 0,
            };
            let dir = TempDir::new().expect("tempdir");
            let query = emb(&[1.0, 1.0, 1.0, 1.0]);

            // (a) Build persistent, record search results and pre-close snapshot.
            let (r_persist, snapshot_before) = {
                let mut persistent =
                    PersistentHnswIndex::create(dir.path(), base_cfg, Dimension(4))
                        .expect("create");
                for (id, v) in ids.iter().zip(&vectors) {
                    persistent.insert(*id, emb(v)).expect("p insert");
                }
                let r = persistent.search(&query, 5).expect("p search");
                let snap = persistent.graph.snapshot();
                (r, snap)
            };

            // (b) Reload from disk, assert search results and FULL snapshot equality.
            let snapshot_after = {
                let reloaded = PersistentHnswIndex::open(dir.path()).expect("reopen");
                let r_reload = reloaded.search(&query, 5).expect("reload search");
                prop_assert_eq!(
                    r_reload.iter().map(|n| n.id).collect::<Vec<_>>(),
                    r_persist.iter().map(|n| n.id).collect::<Vec<_>>(),
                    "reload search results differ from pre-close results"
                );
                reloaded.graph.snapshot()
            };
            // Full structural equality: covers entry_point, rng_state, live_count,
            // AND every node's adjacency lists per layer.
            prop_assert_eq!(
                snapshot_after,
                snapshot_before,
                "snapshot changed across reload (check adjacency, entry_point, rng_state)"
            );

            // (c) Pure in-memory build with same seed and order must match persistent.
            let mut mem = HnswIndex::new(base_cfg, Dimension(4));
            for (id, v) in ids.iter().zip(&vectors) {
                mem.insert(*id, emb(v)).expect("m insert");
            }
            let r_inmem = mem.search(&query, 5).expect("m search");
            prop_assert_eq!(
                r_inmem.iter().map(|n| n.id).collect::<Vec<_>>(),
                r_persist.iter().map(|n| n.id).collect::<Vec<_>>(),
                "in-memory results differ from persistent results"
            );
        }
    }

    /// Verifies that `bulk_load` produces the same search results as individual
    /// incremental inserts, given identical ids, vectors, config, and seed.
    #[test]
    fn bulk_load_produces_same_results_as_individual_inserts() {
        let dir_bulk = TempDir::new().expect("bulk dir");
        let dir_indv = TempDir::new().expect("indv dir");
        // Fixed ids so both builds have identical insertion order and the same seed.
        let ids: Vec<VectorId> = (0..10_u128)
            .map(|i| VectorId::from_uuid(uuid::Uuid::from_u128(i + 500)))
            .collect();
        let bulk_cfg = HnswConfig {
            metric: Metric::Cosine,
            m: 4,
            ef_construction: 20,
            ef_search: 20,
            seed: 0,
        };
        let items: Vec<(VectorId, Embedding)> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let v = f32::from(u8::try_from(i).expect("index fits u8")) / 10.0;
                (*id, emb(&[v, 1.0 - v]))
            })
            .collect();

        let bulk =
            PersistentHnswIndex::bulk_load(dir_bulk.path(), bulk_cfg, Dimension(2), items.clone())
                .expect("bulk_load");

        let mut indv =
            PersistentHnswIndex::create(dir_indv.path(), bulk_cfg, Dimension(2)).expect("create");
        for (id, e) in &items {
            indv.insert(*id, e.clone()).expect("indv insert");
        }

        let query = emb(&[0.5, 0.5]);
        let b_ids: Vec<VectorId> = bulk
            .search(&query, 5)
            .expect("bulk search")
            .into_iter()
            .map(|n| n.id)
            .collect();
        let i_ids: Vec<VectorId> = indv
            .search(&query, 5)
            .expect("indv search")
            .into_iter()
            .map(|n| n.id)
            .collect();
        assert_eq!(b_ids, i_ids);
    }

    /// Full lifecycle: create / insert / remove (tombstone) / compact / close /
    /// open / search. Asserts tombstoned ids never return and live results
    /// survive the round-trip.
    #[test]
    fn full_lifecycle_insert_remove_compact_close_open_search() {
        let dir = TempDir::new().expect("tempdir");
        let life_cfg = HnswConfig {
            metric: Metric::Cosine,
            m: 4,
            ef_construction: 20,
            ef_search: 20,
            seed: 0,
        };
        let keep = VectorId::new();
        let noise = VectorId::new();
        {
            let mut index =
                PersistentHnswIndex::create(dir.path(), life_cfg, Dimension(2)).expect("create");
            index.insert(keep, emb(&[1.0, 0.0])).expect("keep");
            index.insert(noise, emb(&[0.0, 1.0])).expect("noise");
            // Tombstone `noise` then compact to reclaim space.
            index.remove(noise).expect("remove");
            index.compact().expect("compact");
            // After compact, the tombstoned id must not appear.
            let results = index
                .search(&emb(&[1.0, 0.0]), 10)
                .expect("pre-close search");
            assert!(
                results.iter().all(|r| r.id != noise),
                "tombstoned id in pre-close search results"
            );
            assert!(
                results.iter().any(|r| r.id == keep),
                "live id missing from pre-close search results"
            );
        }
        // Reopen: durability check.
        let index = PersistentHnswIndex::open(dir.path()).expect("reopen");
        let results = index
            .search(&emb(&[1.0, 0.0]), 10)
            .expect("post-open search");
        assert!(
            results.iter().all(|r| r.id != noise),
            "tombstoned id reappeared after reopen"
        );
        assert!(
            results.iter().any(|r| r.id == keep),
            "live id missing after reopen"
        );
    }

    /// Verifies that `compact` removes all tombstone ghosts from the persisted
    /// graph while preserving every live result.
    #[test]
    fn compact_persistent_preserves_results() {
        let dir = TempDir::new().expect("tempdir");
        let compact_cfg = HnswConfig {
            metric: Metric::Cosine,
            m: 4,
            ef_construction: 20,
            ef_search: 20,
            seed: 0,
        };
        let mut index =
            PersistentHnswIndex::create(dir.path(), compact_cfg, Dimension(2)).expect("create");
        let keep = VectorId::new();
        for _ in 0..5 {
            let noise = VectorId::new();
            index.insert(noise, emb(&[0.0, 1.0])).expect("noise");
            index.remove(noise).expect("remove noise");
        }
        index.insert(keep, emb(&[1.0, 0.0])).expect("keep");
        index.compact().expect("compact");
        assert_eq!(index.len(), 1);
        let results = index.search(&emb(&[1.0, 0.0]), 10).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, keep);
    }
}
