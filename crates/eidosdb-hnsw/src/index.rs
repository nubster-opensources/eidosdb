//! `HnswIndex`: the in-memory HNSW graph implementing the `VectorIndex` port.

use crate::config::HnswConfig;
use crate::graph::{HnswGraph, NodeIdx};
use crate::rng::SplitMix64;
use eidosdb_core::{
    Dimension, Embedding, IndexError, Metric, Neighbor, Score, VectorId, VectorIndex,
};
use std::collections::BinaryHeap;

/// In-memory HNSW approximate nearest-neighbor index.
///
/// Implements `VectorIndex`. The graph is built for a single metric
/// (`config.metric`); querying with any other metric returns
/// `IndexError::UnsupportedMetric`.
pub struct HnswIndex {
    config: HnswConfig,
    dimension: Dimension,
    graph: HnswGraph,
    rng: SplitMix64,
    /// The single `Metric` slice returned by `supported_metrics`.
    supported: [Metric; 1],
}

impl HnswIndex {
    /// Creates an empty index.
    #[must_use]
    pub fn new(config: HnswConfig, dimension: Dimension) -> Self {
        let supported = [config.metric];
        let rng = SplitMix64::new(config.seed);
        Self {
            config,
            dimension,
            graph: HnswGraph::new(),
            rng,
            supported,
        }
    }

    /// Creates an empty index with a pre-allocated node capacity.
    #[must_use]
    pub fn with_capacity(config: HnswConfig, dimension: Dimension, capacity: usize) -> Self {
        let supported = [config.metric];
        let rng = SplitMix64::new(config.seed);
        Self {
            config,
            dimension,
            graph: HnswGraph::with_capacity(capacity),
            rng,
            supported,
        }
    }

    /// Inserts all points from an iterator in one pass (mass construction).
    ///
    /// Equivalent to calling `insert` for each item but signals bulk intent
    /// to the caller. Returns the first error encountered (nothing is rolled
    /// back on error).
    pub fn bulk_insert(
        &mut self,
        points: impl IntoIterator<Item = (VectorId, Embedding)>,
    ) -> Result<(), IndexError> {
        for (id, embedding) in points {
            self.insert(id, embedding)?;
        }
        Ok(())
    }

    /// Rebuilds the graph keeping only live nodes.
    ///
    /// Collects all live `(VectorId, Embedding)` pairs, resets the graph, and
    /// re-inserts them via `bulk_insert`. The RNG is re-seeded from
    /// `config.seed` so the resulting graph is deterministic for the same live
    /// set in the same order.
    ///
    /// Returns `Err` if a stored embedding is somehow malformed or a re-insert
    /// fails (neither should happen in normal operation).
    pub fn compact(&mut self) -> Result<(), IndexError> {
        let live: Vec<(VectorId, Embedding)> = self
            .graph
            .live_nodes()
            .map(|(_, id, slice, _)| {
                Embedding::new(slice.to_vec()).map(|embedding| (id, embedding))
            })
            .collect::<Result<_, _>>()?;
        self.rng = SplitMix64::new(self.config.seed);
        self.graph = HnswGraph::with_capacity(live.len());
        for (id, embedding) in live {
            self.insert(id, embedding)?;
        }
        Ok(())
    }

    /// Captures a complete, serializable snapshot of the current graph state.
    ///
    /// Built by iterating all node indices, calling `node_snapshot` for each
    /// (DRY: single construction site for `SnapshotNode`), then filling scalar
    /// fields from `state_meta`.
    #[must_use]
    pub fn snapshot(&self) -> crate::graph::GraphSnapshot {
        use crate::graph::GraphSnapshot;
        let snap_nodes: Vec<crate::graph::SnapshotNode> = (0..self.graph.node_count())
            .filter_map(|idx| self.node_snapshot(u64::try_from(idx).unwrap_or(u64::MAX)))
            .collect();
        let (entry_point, entry_level, rng_state, _node_count) = self.state_meta();
        let live_count = snap_nodes.iter().filter(|n| !n.tombstone).count();
        GraphSnapshot {
            nodes: snap_nodes,
            entry_point,
            entry_level,
            rng_state,
            live_count: u64::try_from(live_count).unwrap_or(u64::MAX),
        }
    }

    /// Reconstructs an `HnswIndex` from a `GraphSnapshot` EXACTLY: no
    /// re-insertion, no RNG draw. The restored index is structurally identical
    /// to the one that produced the snapshot (same adjacency, same entry-point,
    /// same RNG state).
    ///
    /// `embeddings_by_node_idx`: closure mapping node-index to its `f32` values
    /// (backed by the mmap segment in `PersistentHnswIndex`).
    pub fn from_snapshot(
        config: HnswConfig,
        dimension: Dimension,
        snapshot: &crate::graph::GraphSnapshot,
        embeddings_by_node_idx: &dyn Fn(usize) -> Option<Vec<f32>>,
    ) -> Result<Self, IndexError> {
        let mut graph = HnswGraph::with_capacity(snapshot.nodes.len());
        for (idx, node) in snapshot.nodes.iter().enumerate() {
            let values = embeddings_by_node_idx(idx)
                .ok_or_else(|| IndexError::Backend(format!("missing embedding for node {idx}")))?;
            let embedding = Embedding::new(values)?;
            let id = VectorId::from_uuid(uuid::Uuid::from_bytes(node.id_bytes));
            let level = usize::try_from(node.level)
                .map_err(|_| IndexError::Backend("node level overflow".to_string()))?;
            graph.add_node(id, embedding, level)?;
            if node.tombstone {
                graph.tombstone(idx);
            }
            for (layer, neighbors) in node.neighbors_per_layer.iter().enumerate() {
                let neighbor_idxs: Vec<NodeIdx> = neighbors
                    .iter()
                    .map(|&n| {
                        usize::try_from(n)
                            .map_err(|_| IndexError::Backend("neighbor index overflow".to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                graph.set_neighbors(idx, layer, neighbor_idxs);
            }
        }
        let entry_point = snapshot
            .entry_point
            .map(|ep| {
                usize::try_from(ep)
                    .map_err(|_| IndexError::Backend("entry_point overflow".to_string()))
            })
            .transpose()?;
        graph.set_entry_point(entry_point);

        let supported = [config.metric];
        Ok(Self {
            config,
            dimension,
            graph,
            rng: crate::rng::SplitMix64::from_state(snapshot.rng_state),
            supported,
        })
    }
}

// ---- beam search primitives ----

/// A candidate entry in the beam heap: (`score_bits_for_ordering`, `node_idx`).
/// We store the raw bits of the f32 score so we can use `Ord` in the heap.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Candidate {
    /// Negated bits so that a max-heap gives us nearest-first ordering.
    score_bits: u32,
    node: NodeIdx,
    id: VectorId,
}

impl Candidate {
    fn new(score: Score, node: NodeIdx, id: VectorId) -> Self {
        Self {
            score_bits: score.0.to_bits(),
            node,
            id,
        }
    }

    fn score(&self) -> Score {
        Score(f32::from_bits(self.score_bits))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher score = closer = comes first in max-heap.
        self.score()
            .0
            .total_cmp(&other.score().0)
            .then_with(|| other.id.cmp(&self.id)) // tie-break: ascending id first
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Nil `VectorId` used as a deterministic fallback when a node-index has no id.
///
/// `VectorId::new()` would mint a fresh random `UUIDv7` (non-deterministic).
/// Using a fixed nil UUID keeps all fallbacks deterministic. In practice,
/// for a valid node-index obtained from the graph, `node_id` is always `Some`;
/// this value is dead but must not introduce randomness.
fn nil_id() -> VectorId {
    VectorId::from_uuid(uuid::Uuid::nil())
}

/// Greedy descent from `entry` down to `target_layer + 1`, returning the
/// single closest node at `target_layer + 1` as the new entry for the next
/// phase. Navigates all nodes (including tombstones) for connectivity.
fn greedy_descent(
    graph: &HnswGraph,
    query: &[f32],
    metric: Metric,
    entry: NodeIdx,
    from_layer: usize,
    to_layer: usize,
) -> NodeIdx {
    let mut current = entry;
    let mut current_score = metric.score(query, graph.node_embedding(current).unwrap_or(&[]));
    for layer in (to_layer..=from_layer).rev() {
        loop {
            let mut improved = false;
            for &neighbor in graph.neighbors(current, layer) {
                let s = metric.score(query, graph.node_embedding(neighbor).unwrap_or(&[]));
                let is_tie = s.0.total_cmp(&current_score.0).is_eq();
                if s.0 > current_score.0
                    || (is_tie
                        && graph.node_id(neighbor).unwrap_or_else(nil_id)
                            < graph.node_id(current).unwrap_or_else(nil_id))
                {
                    current = neighbor;
                    current_score = s;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
    current
}

/// Beam search at `layer` with candidate list of size `ef`.
/// Returns up to `ef` candidates sorted descending by score (closest first).
/// Navigates ALL nodes (tombstoned and non-admissible) for connectivity, but
/// only admits nodes where `is_admissible` returns true AND the node is live.
fn beam_search(
    graph: &HnswGraph,
    query: &[f32],
    metric: Metric,
    entry: NodeIdx,
    ef: usize,
    layer: usize,
    is_admissible: &dyn Fn(&VectorId) -> bool,
) -> Vec<Candidate> {
    // visited: tracks all nodes entered to avoid re-visiting.
    let mut visited = vec![false; graph.node_count()];
    visited[entry] = true;

    let entry_score = metric.score(query, graph.node_embedding(entry).unwrap_or(&[]));
    let entry_id = graph.node_id(entry).unwrap_or_else(nil_id);

    // frontier: max-heap by score - always explore the closest unexplored node first.
    let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();
    frontier.push(Candidate::new(entry_score, entry, entry_id));

    // result: min-heap by score (via Reverse) limited to ef elements.
    // peek()/pop() on this heap always touches the WORST (lowest-score) element,
    // which is what we need for both the early-exit test and eviction.
    let mut result: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();
    // Only add entry to result if it is live and admissible.
    if !graph.is_tombstone(entry) && is_admissible(&entry_id) {
        result.push(std::cmp::Reverse(Candidate::new(
            entry_score,
            entry,
            entry_id,
        )));
    }

    // Score of the worst (lowest-scoring) candidate currently in result.
    let worst_in_result = |r: &BinaryHeap<std::cmp::Reverse<Candidate>>| -> f32 {
        r.peek().map_or(f32::NEG_INFINITY, |c| c.0.score().0)
    };

    while let Some(candidate) = frontier.peek().copied() {
        // If the best unexplored candidate is worse than the worst in result
        // and result is full, we cannot improve result further: stop.
        if result.len() >= ef && candidate.score().0 < worst_in_result(&result) {
            break;
        }
        frontier.pop();
        for &neighbor in graph.neighbors(candidate.node, layer) {
            if neighbor >= visited.len() || visited[neighbor] {
                continue;
            }
            visited[neighbor] = true;
            let s = metric.score(query, graph.node_embedding(neighbor).unwrap_or(&[]));
            let nid = graph.node_id(neighbor).unwrap_or_else(nil_id);
            // Always explore for connectivity.
            frontier.push(Candidate::new(s, neighbor, nid));
            // Only admit live + admissible to result heap.
            if !graph.is_tombstone(neighbor)
                && is_admissible(&nid)
                && (result.len() < ef || s.0 > worst_in_result(&result))
            {
                result.push(std::cmp::Reverse(Candidate::new(s, neighbor, nid)));
                // Evict the worst (lowest-score) candidate when over capacity.
                while result.len() > ef {
                    result.pop();
                }
            }
        }
    }

    // Drain result into a Vec sorted descending by score (closest first).
    let mut out: Vec<Candidate> = result.into_iter().map(|r| r.0).collect();
    out.sort_by(|a, b| {
        b.score()
            .0
            .total_cmp(&a.score().0)
            .then_with(|| a.id.cmp(&b.id))
    });
    out
}

/// SELECT-NEIGHBORS-HEURISTIC from Malkov & Yashunin (2018).
///
/// Given a list of candidates sorted by ascending distance to `base` (i.e.
/// descending score), keeps at most `m_max` of them, preferring those that
/// are closer to `base` than to any already-selected neighbor (directional
/// diversity).
fn select_neighbors_heuristic(
    candidates: &[Candidate],
    // The distance to base is taken from the precomputed `Candidate` score, so
    // the base embedding itself is not needed here.
    _base_embedding: &[f32],
    metric: Metric,
    m_max: usize,
    graph: &HnswGraph,
) -> Vec<NodeIdx> {
    // candidates is sorted descending by score (closest first).
    let mut selected: Vec<NodeIdx> = Vec::with_capacity(m_max);
    'outer: for candidate in candidates {
        if selected.len() >= m_max {
            break;
        }
        let cand_emb = graph.node_embedding(candidate.node).unwrap_or(&[]);
        let score_to_base = candidate.score().0; // score of candidate vs base query
        // Accept candidate if it is closer to base than to any already-selected neighbor.
        for &sel in &selected {
            let sel_emb = graph.node_embedding(sel).unwrap_or(&[]);
            let score_cand_to_sel = metric.score(cand_emb, sel_emb).0;
            if score_cand_to_sel > score_to_base {
                // Candidate is closer to selected neighbor than to base: skip it.
                continue 'outer;
            }
        }
        selected.push(candidate.node);
    }
    selected
}

impl VectorIndex for HnswIndex {
    fn metric(&self) -> Metric {
        self.config.metric
    }

    fn supported_metrics(&self) -> &[Metric] {
        &self.supported
    }

    fn dimension(&self) -> Dimension {
        self.dimension
    }

    fn len(&self) -> usize {
        self.graph.live_count()
    }

    fn insert(&mut self, id: VectorId, embedding: Embedding) -> Result<(), IndexError> {
        // Delegate to insert_tracked and discard the delta (trait callers do
        // not need it; PersistentHnswIndex calls insert_tracked directly).
        self.insert_tracked(id, embedding).map(|_| ())
    }

    fn remove(&mut self, id: VectorId) -> Result<bool, IndexError> {
        // Delegate to remove_tracked and discard the delta.
        self.remove_tracked(id).map(|opt| opt.is_some())
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
        if metric != self.config.metric {
            return Err(IndexError::UnsupportedMetric(metric));
        }
        let Some(entry) = self.graph.entry_point() else {
            return Ok(Vec::new());
        };

        let entry_level = self.graph.node_level(entry);
        let q = query.as_slice();
        let ef = self.config.ef_search.max(k);

        // Greedy descent to layer 1.
        let current_entry = if entry_level > 0 {
            greedy_descent(&self.graph, q, metric, entry, entry_level, 1)
        } else {
            entry
        };

        // Beam search at layer 0.
        let candidates = beam_search(&self.graph, q, metric, current_entry, ef, 0, is_admissible);

        let mut results: Vec<Neighbor> = candidates
            .iter()
            .take(k)
            .map(|c| Neighbor {
                id: c.id,
                score: c.score(),
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .0
                .total_cmp(&a.score.0)
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(k);
        Ok(results)
    }
}

// ---- incremental-write-through helpers (used by PersistentHnswIndex) ----

/// The set of rows that changed during a single `insert_tracked` call.
/// `touched_nodes` lists every node-index whose `hnsw_nodes` row or at
/// least one `hnsw_adjacency` row was written: the new node plus every
/// existing node whose adjacency list was modified (the M neighbors it
/// linked to per layer, and any neighbor whose list was pruned by the
/// selection heuristic). `entry_changed` is `true` when the entry-point
/// node-index was updated.
pub struct InsertDelta {
    /// Node-index of the newly inserted node.
    pub new_node_idx: u64,
    /// All node-indices whose persisted rows changed (includes `new_node_idx`).
    pub touched_nodes: Vec<u64>,
    /// Whether the entry-point was updated by this insert.
    pub entry_changed: bool,
}

/// The set of rows that changed during a single `remove_tracked` call.
pub struct RemoveDelta {
    /// Node-indices whose `hnsw_nodes` row changed (the tombstoned node).
    pub touched_nodes: Vec<u64>,
    /// Whether the entry-point was updated (i.e. the tombstoned node was
    /// the entry point and a new one was elected).
    pub entry_changed: bool,
}

impl HnswIndex {
    /// Inserts `id`/`embedding` and reports exactly which persisted rows
    /// changed, for incremental write-through in `PersistentHnswIndex`.
    ///
    /// `touched_nodes` collects: the new node-index, every existing node
    /// whose adjacency list was modified (the M neighbors linked per layer,
    /// plus any neighbor whose list was pruned by the selection heuristic).
    /// The trait `VectorIndex::insert` delegates here and discards the delta.
    // `embedding` must be owned: it is forwarded by value to `add_node` (which
    // stores it) and the trait boundary passes it by value too.
    #[allow(clippy::needless_pass_by_value)]
    pub fn insert_tracked(
        &mut self,
        id: VectorId,
        embedding: Embedding,
    ) -> Result<InsertDelta, IndexError> {
        if embedding.dimension() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension.get(),
                got: embedding.dimension().get(),
            });
        }
        if self.graph.id_to_node(id).is_some() {
            return Err(IndexError::DuplicateId(id));
        }

        let entry_before = self.graph.entry_point();
        let level = self.rng.next_level(self.config.m_l());
        let new_node = self.graph.add_node(id, embedding.clone(), level)?;
        let new_node_u64 = u64::try_from(new_node).unwrap_or(u64::MAX);

        // Track which adjacency lists are mutated (new node + affected neighbors).
        let mut touched: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        touched.insert(new_node_u64);

        let Some(entry) = self.graph.entry_point() else {
            self.graph.set_entry_point(Some(new_node));
            return Ok(InsertDelta {
                new_node_idx: new_node_u64,
                touched_nodes: vec![new_node_u64],
                entry_changed: true,
            });
        };

        let entry_level = self.graph.node_level(entry);
        let metric = self.config.metric;
        let q = embedding.as_slice();

        let mut current_entry = entry;
        if entry_level > level {
            current_entry = greedy_descent(&self.graph, q, metric, entry, entry_level, level + 1);
        }

        let ef_c = self.config.ef_construction;
        let m_max0 = self.config.m_max0();
        let m_max = self.config.m_max();

        for layer in (0..=level.min(entry_level)).rev() {
            let m_limit = if layer == 0 { m_max0 } else { m_max };
            let candidates =
                beam_search(&self.graph, q, metric, current_entry, ef_c, layer, &|_| {
                    true
                });
            let selected = select_neighbors_heuristic(&candidates, q, metric, m_limit, &self.graph);
            self.graph.set_neighbors(new_node, layer, selected.clone());
            for &sel in &selected {
                touched.insert(u64::try_from(sel).unwrap_or(u64::MAX));
                let mut sel_neighbors: Vec<NodeIdx> = self.graph.neighbors(sel, layer).to_vec();
                sel_neighbors.push(new_node);
                if sel_neighbors.len() > m_limit {
                    let sel_emb = self.graph.node_embedding(sel).unwrap_or(&[]);
                    let mut pruning_candidates: Vec<Candidate> = sel_neighbors
                        .iter()
                        .map(|&n| {
                            let s =
                                metric.score(sel_emb, self.graph.node_embedding(n).unwrap_or(&[]));
                            let nid = self.graph.node_id(n).unwrap_or_else(nil_id);
                            Candidate::new(s, n, nid)
                        })
                        .collect();
                    pruning_candidates.sort_by(|a, b| {
                        b.score()
                            .0
                            .total_cmp(&a.score().0)
                            .then_with(|| a.id.cmp(&b.id))
                    });
                    sel_neighbors = select_neighbors_heuristic(
                        &pruning_candidates,
                        sel_emb,
                        metric,
                        m_limit,
                        &self.graph,
                    );
                }
                self.graph.set_neighbors(sel, layer, sel_neighbors);
            }
            if let Some(best) = candidates.first() {
                current_entry = best.node;
            }
        }

        let entry_changed = if level > entry_level {
            self.graph.set_entry_point(Some(new_node));
            true
        } else {
            self.graph.entry_point() != entry_before
        };

        Ok(InsertDelta {
            new_node_idx: new_node_u64,
            touched_nodes: touched.into_iter().collect(),
            entry_changed,
        })
    }

    /// Tombstones `id` and reports which rows changed, for incremental
    /// write-through in `PersistentHnswIndex`. Returns `None` if the id has
    /// no live node (mirrors `Ok(false)` at the trait boundary).
    /// The trait `VectorIndex::remove` delegates here and discards the delta.
    pub fn remove_tracked(&mut self, id: VectorId) -> Result<Option<RemoveDelta>, IndexError> {
        let Some(node) = self.graph.id_to_node(id) else {
            return Ok(None);
        };
        let was_entry = self.graph.entry_point() == Some(node);
        let became_tombstone = self.graph.tombstone(node);
        if !became_tombstone {
            return Ok(None);
        }
        let mut entry_changed = false;
        if was_entry {
            let new_entry = self
                .graph
                .live_nodes()
                .max_by_key(|(_, _, _, level)| *level)
                .map(|(idx, _, _, _)| idx);
            self.graph.set_entry_point(new_entry);
            entry_changed = true;
        }
        let node_u64 = u64::try_from(node).unwrap_or(u64::MAX);
        Ok(Some(RemoveDelta {
            touched_nodes: vec![node_u64],
            entry_changed,
        }))
    }

    /// Returns the full persisted row for one node-index: id, level,
    /// tombstone flag, and adjacency lists per layer. Used by
    /// `PersistentHnswIndex` to write exactly the rows that changed during
    /// an incremental `insert_tracked` or `remove_tracked`.
    #[must_use]
    pub fn node_snapshot(&self, node_idx: u64) -> Option<crate::graph::SnapshotNode> {
        let idx = usize::try_from(node_idx).ok()?;
        if idx >= self.graph.node_count() {
            return None;
        }
        let level = self.graph.node_level(idx);
        let neighbors_per_layer: Vec<Vec<u64>> = (0..=level)
            .map(|layer| {
                self.graph
                    .neighbors(idx, layer)
                    .iter()
                    .map(|&n| {
                        #[allow(clippy::cast_possible_truncation)]
                        let n64 = n as u64;
                        n64
                    })
                    .collect()
            })
            .collect();
        Some(crate::graph::SnapshotNode {
            id_bytes: self
                .graph
                .node_id(idx)
                .map_or([0u8; 16], |id| id.as_uuid().into_bytes()),
            level: u32::try_from(level).unwrap_or(u32::MAX),
            tombstone: self.graph.is_tombstone(idx),
            neighbors_per_layer,
        })
    }

    /// Returns the current mutable meta state for the `"state"` redb row:
    /// `(entry_point, entry_level, rng_state, node_count)`.
    ///
    /// Used by `PersistentHnswIndex` to write the `"state"` row after each
    /// incremental insert or remove, without taking a full snapshot.
    #[must_use]
    pub fn state_meta(&self) -> (Option<u64>, u32, u64, u64) {
        #[allow(clippy::cast_possible_truncation)]
        let entry_point = self.graph.entry_point().map(|ep| ep as u64);
        let entry_level = self.graph.entry_point().map_or(0, |ep| {
            u32::try_from(self.graph.node_level(ep)).unwrap_or(0)
        });
        let rng_state = self.rng.state();
        let node_count = u64::try_from(self.graph.node_count()).unwrap_or(u64::MAX);
        (entry_point, entry_level, rng_state, node_count)
    }
}

#[cfg(test)]
mod tests {
    use super::HnswIndex;
    use crate::config::HnswConfig;
    use eidosdb_core::{
        Dimension, Embedding, FlatIndex, IndexError, Metric, VectorId, VectorIndex,
    };

    fn emb(values: &[f32]) -> Embedding {
        Embedding::new(values.to_vec()).expect("non-empty")
    }

    fn config() -> HnswConfig {
        HnswConfig {
            metric: Metric::Cosine,
            m: 4,
            ef_construction: 20,
            ef_search: 20,
            ..HnswConfig::default()
        }
    }

    #[test]
    fn new_index_is_empty() {
        let index = HnswIndex::new(config(), Dimension(2));
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.metric(), Metric::Cosine);
        assert_eq!(index.dimension(), Dimension(2));
    }

    #[test]
    fn insert_rejects_dimension_mismatch() {
        let mut index = HnswIndex::new(config(), Dimension(3));
        assert_eq!(
            index.insert(VectorId::new(), emb(&[1.0, 0.0])),
            Err(IndexError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn insert_rejects_duplicate_live_id() {
        let mut index = HnswIndex::new(config(), Dimension(2));
        let id = VectorId::new();
        index.insert(id, emb(&[1.0, 0.0])).expect("first");
        assert_eq!(
            index.insert(id, emb(&[0.0, 1.0])),
            Err(IndexError::DuplicateId(id))
        );
    }

    #[test]
    fn search_unsupported_metric_is_rejected() {
        let mut index = HnswIndex::new(config(), Dimension(2));
        index
            .insert(VectorId::new(), emb(&[1.0, 0.0]))
            .expect("insert");
        assert_eq!(
            index.search_filtered(&emb(&[1.0, 0.0]), 1, Metric::Euclidean, &|_| true),
            Err(IndexError::UnsupportedMetric(Metric::Euclidean))
        );
    }

    #[test]
    fn supported_metrics_contains_only_config_metric() {
        let index = HnswIndex::new(config(), Dimension(2));
        let supported = index.supported_metrics();
        assert_eq!(supported, &[Metric::Cosine]);
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let index = HnswIndex::new(config(), Dimension(2));
        let results = index.search(&emb(&[1.0, 0.0]), 10).expect("search");
        assert!(results.is_empty());
    }

    #[test]
    fn single_insert_is_its_own_nearest_neighbor() {
        let mut index = HnswIndex::new(config(), Dimension(2));
        let id = VectorId::new();
        index.insert(id, emb(&[1.0, 0.0])).expect("insert");
        let results = index.search(&emb(&[1.0, 0.0]), 1).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn nearest_neighbor_matches_flat_oracle_small_corpus() {
        // 20-point 2-D corpus; compare top-3 with FlatIndex.
        let cfg = HnswConfig {
            metric: Metric::Cosine,
            m: 4,
            ef_construction: 40,
            ef_search: 40,
            seed: 0,
        };
        let mut hnsw = HnswIndex::new(cfg, Dimension(2));
        let mut flat = FlatIndex::new(Metric::Cosine, Dimension(2));
        let points: Vec<(VectorId, Embedding)> = (0..20)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let angle = (i as f32) * std::f32::consts::PI / 10.0;
                (VectorId::new(), emb(&[angle.cos(), angle.sin()]))
            })
            .collect();
        for (id, e) in &points {
            hnsw.insert(*id, e.clone()).expect("hnsw insert");
            flat.insert(*id, e.clone()).expect("flat insert");
        }
        let query = emb(&[1.0, 0.0]);
        let hnsw_results = hnsw.search(&query, 3).expect("hnsw search");
        let flat_results = flat.search(&query, 3).expect("flat search");
        // The top-1 must agree (strong constraint on a tiny corpus).
        assert_eq!(hnsw_results[0].id, flat_results[0].id);
    }

    #[test]
    fn remove_returns_true_for_live_id_false_otherwise() {
        let mut index = HnswIndex::new(config(), Dimension(2));
        let id = VectorId::new();
        index.insert(id, emb(&[1.0, 0.0])).expect("insert");
        assert_eq!(index.remove(id), Ok(true));
        assert_eq!(index.len(), 0);
        assert_eq!(index.remove(id), Ok(false));
    }

    #[test]
    fn tombstone_not_returned_in_search() {
        let mut index = HnswIndex::new(config(), Dimension(2));
        let keep = VectorId::new();
        let drop = VectorId::new();
        index.insert(keep, emb(&[1.0, 0.0])).expect("keep");
        index.insert(drop, emb(&[1.0, 0.0])).expect("drop");
        index.remove(drop).expect("remove");
        let results = index.search(&emb(&[1.0, 0.0]), 10).expect("search");
        assert!(results.iter().all(|r| r.id != drop));
    }

    #[test]
    fn len_excludes_tombstones() {
        let mut index = HnswIndex::new(config(), Dimension(2));
        let id = VectorId::new();
        index.insert(id, emb(&[1.0, 0.0])).expect("insert");
        assert_eq!(index.len(), 1);
        index.remove(id).expect("remove");
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn reinsert_tombstoned_id_succeeds() {
        let mut index = HnswIndex::new(config(), Dimension(2));
        let id = VectorId::new();
        index.insert(id, emb(&[1.0, 0.0])).expect("first");
        index.remove(id).expect("remove");
        // Reinsertion must succeed (not DuplicateId).
        index.insert(id, emb(&[0.5, 0.5])).expect("reinsert");
        assert_eq!(index.len(), 1);
        let results = index.search(&emb(&[0.5, 0.5]), 1).expect("search");
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn compact_preserves_live_results() {
        let cfg = HnswConfig {
            m: 4,
            ef_construction: 40,
            ef_search: 40,
            ..HnswConfig::default()
        };
        let mut index = HnswIndex::new(cfg, Dimension(2));
        let keep = VectorId::new();
        let drop = VectorId::new();
        index.insert(keep, emb(&[1.0, 0.0])).expect("keep");
        index.insert(drop, emb(&[0.0, 1.0])).expect("drop");
        index.remove(drop).expect("remove");
        index.compact().expect("compact");
        assert_eq!(index.len(), 1);
        let results = index.search(&emb(&[1.0, 0.0]), 10).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, keep);
    }

    #[test]
    fn ties_broken_by_ascending_id() {
        // Insert two identical embeddings; the one with lower VectorId must come first.
        let cfg = HnswConfig {
            m: 4,
            ef_construction: 40,
            ef_search: 40,
            ..HnswConfig::default()
        };
        let mut index = HnswIndex::new(cfg, Dimension(2));
        let mut flat = FlatIndex::new(Metric::Cosine, Dimension(2));
        let first = VectorId::new();
        let second = VectorId::new();
        for id in [first, second] {
            index.insert(id, emb(&[1.0, 0.0])).expect("insert");
            flat.insert(id, emb(&[1.0, 0.0])).expect("flat insert");
        }
        let h = index.search(&emb(&[1.0, 0.0]), 2).expect("hnsw");
        let f = flat.search(&emb(&[1.0, 0.0]), 2).expect("flat");
        assert_eq!(h[0].id, f[0].id);
        assert_eq!(h[1].id, f[1].id);
    }

    #[test]
    fn select_neighbors_heuristic_prefers_directional_diversity() {
        // 3-D geometry that makes the heuristic non-trivial.
        //
        // base = [1, 0, 0]
        // A = ~[0.900, 0.436, 0] (unit, ~26 deg from base; score ~0.90)
        // B = ~[0.850, 0.527, 0] (unit, ~32 deg from base; score ~0.85)
        //      cosine(B,A) ~0.995 >> cosine(B,base) ~0.85 -> B is closer to A
        //      than to base, so the heuristic REJECTS B.
        // C = [0, 0, 1] (orthogonal to the XY plane; cosine with A and base = 0)
        //      cosine(C,A) = 0 == cosine(C,base) -> NOT rejected; C is selected.
        //
        // Top-2 by raw score: A then B. Heuristic picks A and C (diversity over B).
        use super::{Candidate, HnswGraph, select_neighbors_heuristic};
        use eidosdb_core::{Embedding, Metric, VectorId};

        let mut g = HnswGraph::new();
        let base_emb = [1.0_f32, 0.0, 0.0];
        let id_a = VectorId::new();
        let id_b = VectorId::new();
        let id_c = VectorId::new();
        // A and B are normalised manually: A ~ (0.9, 0.436, 0)/1.0, B ~ (0.85, 0.527, 0)/1.0.
        let ea = Embedding::new(vec![0.9_f32, 0.436, 0.0]).expect("a");
        let eb = Embedding::new(vec![0.85_f32, 0.527, 0.0]).expect("b");
        let ec = Embedding::new(vec![0.0_f32, 0.0, 1.0]).expect("c");
        let na = g.add_node(id_a, ea, 0).expect("a");
        let nb = g.add_node(id_b, eb, 0).expect("b");
        let nc = g.add_node(id_c, ec, 0).expect("c");

        let metric = Metric::Cosine;
        // Pre-compute scores relative to base.
        let score_a = metric.score(&base_emb, &[0.9, 0.436, 0.0]);
        let score_b = metric.score(&base_emb, &[0.85, 0.527, 0.0]);
        let score_c = metric.score(&base_emb, &[0.0, 0.0, 1.0]);

        // Build candidates list sorted descending by score (closest first).
        let mut candidates = vec![
            Candidate::new(score_a, na, id_a),
            Candidate::new(score_b, nb, id_b),
            Candidate::new(score_c, nc, id_c),
        ];
        candidates.sort_by(|a, b| {
            b.score()
                .0
                .total_cmp(&a.score().0)
                .then_with(|| a.id.cmp(&b.id))
        });

        let selected = select_neighbors_heuristic(&candidates, &base_emb, metric, 2, &g);
        // A must be selected (closest to base).
        assert!(selected.contains(&na), "A (closest) must be selected");
        // C must be selected over B (diversity: B too close to A).
        assert!(
            selected.contains(&nc),
            "C (diverse direction) must be selected"
        );
        assert!(
            !selected.contains(&nb),
            "B (redundant with A) must be rejected"
        );
    }

    // ---- proptest port properties ----

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn determinism_same_seed_same_results(
            vectors in proptest::collection::vec(
                proptest::collection::vec(-1.0_f32..1.0_f32, 4_usize..=4),
                2..20_usize,
            ),
        ) {
            let cfg = HnswConfig {
                m: 4,
                ef_construction: 20,
                ef_search: 20,
                seed: 42,
                ..HnswConfig::default()
            };
            let mut a = HnswIndex::new(cfg, Dimension(4));
            let mut b = HnswIndex::new(cfg, Dimension(4));
            let ids: Vec<VectorId> = (0..vectors.len())
                .map(|i| VectorId::from_uuid(uuid::Uuid::from_u128(u128::try_from(i).expect("index fits u128"))))
                .collect();
            for (id, v) in ids.iter().zip(&vectors) {
                let e = emb(v);
                a.insert(*id, e.clone()).expect("a");
                b.insert(*id, e).expect("b");
            }
            let query = emb(&[1.0, 1.0, 1.0, 1.0]);
            let ra = a.search(&query, 5).expect("a search");
            let rb = b.search(&query, 5).expect("b search");
            prop_assert_eq!(ra, rb);
        }

        #[test]
        fn filtering_parity_with_flat_oracle(
            vectors in proptest::collection::vec(
                proptest::collection::vec(-1.0_f32..1.0_f32, 4_usize..=4),
                4..15_usize,
            ),
        ) {
            let cfg = HnswConfig {
                m: 4,
                ef_construction: 40,
                ef_search: 40,
                seed: 7,
                ..HnswConfig::default()
            };
            let mut hnsw = HnswIndex::new(cfg, Dimension(4));
            let mut flat = FlatIndex::new(Metric::Cosine, Dimension(4));
            let ids: Vec<VectorId> = (0..vectors.len())
                .map(|i| VectorId::from_uuid(uuid::Uuid::from_u128(u128::try_from(i).expect("index fits u128") + 100)))
                .collect();
            for (id, v) in ids.iter().zip(&vectors) {
                let e = emb(v);
                hnsw.insert(*id, e.clone()).expect("hnsw");
                flat.insert(*id, e).expect("flat");
            }
            // Admit only even-indexed ids. With MIN=4 vectors, step_by(2)
            // guarantees at least 2 admissible ids (indices 0 and 2).
            let allowed: Vec<VectorId> = ids.iter().step_by(2).copied().collect();
            let pred = |id: &VectorId| allowed.contains(id);
            let query = emb(&[1.0, 1.0, 1.0, 1.0]);
            let hnsw_ids: Vec<VectorId> = hnsw
                .search_filtered(&query, 5, Metric::Cosine, &pred)
                .expect("hnsw search")
                .into_iter()
                .map(|n| n.id)
                .collect();
            let flat_ids: Vec<VectorId> = flat
                .search_filtered(&query, 5, Metric::Cosine, &pred)
                .expect("flat search")
                .into_iter()
                .map(|n| n.id)
                .collect();
            // Precision: HNSW must only return admissible ids.
            for id in &hnsw_ids {
                prop_assert!(
                    allowed.contains(id),
                    "search_filtered returned a non-admissible id {id:?}; allowed={allowed:?}"
                );
            }
            // Recall: every flat-oracle result must also appear in hnsw results.
            // HNSW is exact on tiny corpora (ef=40 >> corpus size), so this holds.
            for id in &flat_ids {
                prop_assert!(
                    hnsw_ids.contains(id),
                    "flat result {id:?} missing from hnsw; hnsw={hnsw_ids:?} flat={flat_ids:?}"
                );
            }
        }

        #[test]
        fn tombstone_never_in_results_prop(
            vectors in proptest::collection::vec(
                proptest::collection::vec(-1.0_f32..1.0_f32, 4_usize..=4),
                2..12_usize,
            ),
        ) {
            let cfg = HnswConfig {
                m: 4,
                ef_construction: 20,
                ef_search: 20,
                seed: 99,
                ..HnswConfig::default()
            };
            let mut index = HnswIndex::new(cfg, Dimension(4));
            let ids: Vec<VectorId> = (0..vectors.len())
                .map(|i| VectorId::from_uuid(uuid::Uuid::from_u128(u128::try_from(i).expect("index fits u128") + 200)))
                .collect();
            for (id, v) in ids.iter().zip(&vectors) {
                index.insert(*id, emb(v)).expect("insert");
            }
            // Tombstone the first id.
            index.remove(ids[0]).expect("remove");
            let results = index
                .search(&emb(&[1.0, 1.0, 1.0, 1.0]), ids.len())
                .expect("search");
            prop_assert!(results.iter().all(|r| r.id != ids[0]));
        }
    }
}
