//! Private graph structure for `HnswIndex`.
//!
//! Nodes are identified by a dense internal `NodeIdx` (a `usize`) distinct from
//! the public `VectorId`. The mapping `VectorId -> NodeIdx` always points at the
//! LIVE node for an id. A tombstoned node retains its `NodeIdx` and adjacency
//! lists so it remains navigable (preserving graph connectivity) until a
//! `compact` rebuilds the graph from scratch.

use eidosdb_core::{Dimension, Embedding, IndexError, VectorId};
use std::collections::HashMap;

/// Internal node index. Dense, monotonically increasing.
// Allow: consumed by Task 5 (HnswIndex).
#[allow(dead_code)]
pub(crate) type NodeIdx = usize;

/// One node in the HNSW graph.
// Allow: consumed by Task 5 (HnswIndex).
#[allow(dead_code)]
struct Node {
    id: VectorId,
    embedding: Embedding,
    /// Adjacency list per layer. `neighbors[0]` is the layer-0 list, etc.
    neighbors: Vec<Vec<NodeIdx>>,
    /// Maximum layer this node participates in.
    level: usize,
    /// Soft-delete flag. Tombstoned nodes remain navigable but are excluded
    /// from result heaps.
    tombstone: bool,
}

/// The private HNSW graph: owns all nodes and the live id map.
// Allow: consumed by Task 5 (HnswIndex).
#[allow(dead_code)]
pub(crate) struct HnswGraph {
    nodes: Vec<Node>,
    /// Maps a `VectorId` to the `NodeIdx` of its LIVE node. A tombstoned node is
    /// removed from this map so that a subsequent insert of the same id creates
    /// a new live entry pointing at a new node.
    live: HashMap<VectorId, NodeIdx>,
    /// Current entry point (the node with the highest level).
    entry_point: Option<NodeIdx>,
    /// Number of live (non-tombstoned) nodes.
    live_count: usize,
}

// Allow: all methods consumed by Task 5 (HnswIndex).
#[allow(dead_code)]
impl HnswGraph {
    /// Creates an empty graph.
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            live: HashMap::new(),
            entry_point: None,
            live_count: 0,
        }
    }

    /// Creates an empty graph with pre-allocated capacity.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            live: HashMap::with_capacity(capacity),
            entry_point: None,
            live_count: 0,
        }
    }

    /// Adds a new LIVE node. Returns `DuplicateId` if the id already has a live
    /// node. Reinserting a tombstoned id succeeds: a new live node is created
    /// and the map is updated.
    pub(crate) fn add_node(
        &mut self,
        id: VectorId,
        embedding: Embedding,
        level: usize,
    ) -> Result<NodeIdx, IndexError> {
        if embedding.dimension() != self.dimension_hint() && !self.nodes.is_empty() {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension_hint().0,
                got: embedding.dimension().0,
            });
        }
        if self.live.contains_key(&id) {
            return Err(IndexError::DuplicateId(id));
        }
        let idx = self.nodes.len();
        // Adjacency lists: one empty Vec per layer.
        let neighbors = vec![Vec::new(); level + 1];
        self.nodes.push(Node {
            id,
            embedding,
            neighbors,
            level,
            tombstone: false,
        });
        self.live.insert(id, idx);
        self.live_count += 1;
        Ok(idx)
    }

    /// Returns the `Dimension` of the first node, or `Dimension(0)` when empty.
    fn dimension_hint(&self) -> eidosdb_core::Dimension {
        self.nodes
            .first()
            .map_or(Dimension(0), |n| n.embedding.dimension())
    }

    /// Replaces the neighbor list for `node` at `layer`.
    pub(crate) fn set_neighbors(&mut self, node: NodeIdx, layer: usize, neighbors: Vec<NodeIdx>) {
        if let Some(n) = self.nodes.get_mut(node) {
            if layer < n.neighbors.len() {
                n.neighbors[layer] = neighbors;
            }
        }
    }

    /// Returns the neighbor list for `node` at `layer` (empty slice if out of range).
    pub(crate) fn neighbors(&self, node: NodeIdx, layer: usize) -> &[NodeIdx] {
        self.nodes
            .get(node)
            .and_then(|n| n.neighbors.get(layer))
            .map_or(&[], Vec::as_slice)
    }

    /// Whether `node` is tombstoned.
    pub(crate) fn is_tombstone(&self, node: NodeIdx) -> bool {
        self.nodes.get(node).is_some_and(|n| n.tombstone)
    }

    /// Marks `node` as tombstoned and removes it from the live map.
    /// Returns `true` if the node was live, `false` if already tombstoned or absent.
    pub(crate) fn tombstone(&mut self, node: NodeIdx) -> bool {
        let Some(n) = self.nodes.get_mut(node) else {
            return false;
        };
        if n.tombstone {
            return false;
        }
        n.tombstone = true;
        self.live.remove(&n.id);
        self.live_count = self.live_count.saturating_sub(1);
        true
    }

    /// Maximum layer of `node`.
    pub(crate) fn node_level(&self, node: NodeIdx) -> usize {
        self.nodes.get(node).map_or(0, |n| n.level)
    }

    /// `VectorId` of `node`.
    pub(crate) fn node_id(&self, node: NodeIdx) -> Option<VectorId> {
        self.nodes.get(node).map(|n| n.id)
    }

    /// Embedding slice of `node` (zero-copy).
    pub(crate) fn node_embedding(&self, node: NodeIdx) -> Option<&[f32]> {
        self.nodes.get(node).map(|n| n.embedding.as_slice())
    }

    /// Current entry point (highest-level node).
    pub(crate) fn entry_point(&self) -> Option<NodeIdx> {
        self.entry_point
    }

    /// Sets the entry point.
    pub(crate) fn set_entry_point(&mut self, node: Option<NodeIdx>) {
        self.entry_point = node;
    }

    /// Total number of nodes (live + tombstoned).
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of live (non-tombstoned) nodes.
    pub(crate) fn live_count(&self) -> usize {
        self.live_count
    }

    /// Returns the `NodeIdx` of the live node for `id`, if any.
    pub(crate) fn id_to_node(&self, id: VectorId) -> Option<NodeIdx> {
        self.live.get(&id).copied()
    }

    /// Iterates over all node indices (including tombstoned ones).
    pub(crate) fn all_nodes(&self) -> impl Iterator<Item = NodeIdx> {
        0..self.nodes.len()
    }

    /// Iterates over all live (non-tombstoned) nodes as `(NodeIdx, VectorId, &[f32], level)`.
    pub(crate) fn live_nodes(&self) -> impl Iterator<Item = (NodeIdx, VectorId, &[f32], usize)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.tombstone)
            .map(|(idx, n)| (idx, n.id, n.embedding.as_slice(), n.level))
    }
}

// ---- Graph snapshot (serializable, used by PersistentHnswIndex) ----

/// Per-node record inside a `GraphSnapshot`. Does not carry the embedding (which
/// lives in the mmap segment keyed by node-index, following the B1 pattern).
///
/// `PartialEq`/`Eq` are derived (all fields are integers, bool, or `Vec<Vec<u64>>`)
/// to enable full structural equality checks in the reload-fidelity proptest.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SnapshotNode {
    /// Public vector identifier (16 bytes, UUID).
    pub id_bytes: [u8; 16],
    /// Maximum layer this node participates in.
    pub level: u32,
    /// Soft-delete flag.
    pub tombstone: bool,
    /// Adjacency lists per layer: `neighbors[l]` is the list of neighbor
    /// node-indices at layer `l`. Length == `level + 1`.
    pub neighbors_per_layer: Vec<Vec<u64>>,
}

/// A complete, serializable snapshot of an `HnswGraph`/`HnswIndex` state.
///
/// `GraphSnapshot` is the unit of persistence: `PersistentHnswIndex` uses it
/// on `open` via `HnswIndex::from_snapshot` to reload the graph EXACTLY.
/// Embeddings are NOT included (they live in the mmap segment keyed by
/// node-index). The `rng_state` field allows exact resumption of the RNG
/// without any new draws on restore.
///
/// `PartialEq`/`Eq` are derived (all fields are integers, bool, `Option<u64>`,
/// and `Vec<SnapshotNode>`) to enable a single `assert_eq!` in the
/// reload-fidelity proptest that covers adjacency lists, not just scalar fields.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GraphSnapshot {
    /// Dense node records in node-index order (index 0 = first node inserted).
    pub nodes: Vec<SnapshotNode>,
    /// Entry-point node-index (`None` when the index is empty).
    pub entry_point: Option<u64>,
    /// Level of the entry-point node (0 when `entry_point` is `None`).
    pub entry_level: u32,
    /// Current `SplitMix64` state, captured at snapshot time so restore is exact.
    pub rng_state: u64,
    /// Number of live (non-tombstoned) nodes. Derived on restore, stored for
    /// fast `len()` without scanning the full node list.
    pub live_count: u64,
}

#[cfg(test)]
mod tests {
    use super::HnswGraph;
    use eidosdb_core::{Embedding, IndexError, VectorId};

    fn emb(values: &[f32]) -> Embedding {
        Embedding::new(values.to_vec()).expect("non-empty")
    }

    #[test]
    fn new_graph_is_empty() {
        let g = HnswGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.live_count(), 0);
        assert!(g.entry_point().is_none());
    }

    #[test]
    fn add_node_returns_sequential_indices() {
        let mut g = HnswGraph::new();
        let a = VectorId::new();
        let b = VectorId::new();
        let i0 = g.add_node(a, emb(&[1.0, 0.0]), 0).expect("a");
        let i1 = g.add_node(b, emb(&[0.0, 1.0]), 1).expect("b");
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.live_count(), 2);
    }

    #[test]
    fn add_node_rejects_duplicate_live_id() {
        let mut g = HnswGraph::new();
        let id = VectorId::new();
        g.add_node(id, emb(&[1.0, 0.0]), 0).expect("first");
        assert_eq!(
            g.add_node(id, emb(&[0.0, 1.0]), 0),
            Err(IndexError::DuplicateId(id))
        );
    }

    #[test]
    fn tombstone_removes_from_live_map() {
        let mut g = HnswGraph::new();
        let id = VectorId::new();
        let idx = g.add_node(id, emb(&[1.0, 0.0]), 0).expect("add");
        assert!(g.tombstone(idx));
        assert!(g.is_tombstone(idx));
        assert_eq!(g.live_count(), 0);
        assert!(g.id_to_node(id).is_none());
    }

    #[test]
    fn reinsert_after_tombstone_creates_new_live_node() {
        let mut g = HnswGraph::new();
        let id = VectorId::new();
        let idx0 = g.add_node(id, emb(&[1.0, 0.0]), 0).expect("first");
        g.tombstone(idx0);
        // Second insert of the same id succeeds now.
        let idx1 = g.add_node(id, emb(&[0.5, 0.5]), 1).expect("reinsert");
        assert_ne!(idx0, idx1);
        assert_eq!(g.live_count(), 1);
        assert_eq!(g.id_to_node(id), Some(idx1));
        // Old tombstone node is still navigable.
        assert!(g.is_tombstone(idx0));
    }

    #[test]
    fn neighbors_set_and_get() {
        let mut g = HnswGraph::new();
        let a = g.add_node(VectorId::new(), emb(&[1.0, 0.0]), 1).expect("a");
        let b = g.add_node(VectorId::new(), emb(&[0.0, 1.0]), 0).expect("b");
        g.set_neighbors(a, 0, vec![b]);
        assert_eq!(g.neighbors(a, 0), &[b]);
        assert_eq!(g.neighbors(a, 1), &[] as &[usize]);
    }

    #[test]
    fn node_level_reflects_construction() {
        let mut g = HnswGraph::new();
        let idx = g
            .add_node(VectorId::new(), emb(&[1.0, 0.0]), 3)
            .expect("add");
        assert_eq!(g.node_level(idx), 3);
    }
}
