//! `CollectionKind`: runtime dispatch enum wrapping a persistent `Collection`.
//!
//! Two variants exist: [`CollectionKind::Flat`] (exact nearest-neighbour via
//! `PersistentFlatIndex`) and [`CollectionKind::Hnsw`] (approximate
//! nearest-neighbour via `PersistentHnswIndex`). Both expose an identical
//! surface for insert, delete, and search so callers need not know which
//! algorithm backs the collection.
//!
//! Layout on disk (relative to the collection root `dir`):
//!
//! ```text
//! dir/
//!   vector/          <- PersistentFlatIndex or PersistentHnswIndex directory
//!   lexical.redb     <- PersistentLexicalIndex file
//!   payload.redb     <- PersistentPayloadStore file
//! ```

use std::path::Path;

use eidosdb_core::{Dimension, Embedding, Metric, VectorId};
use eidosdb_hnsw::HnswConfig;
use eidosdb_lexical::Document;
use eidosdb_query::{Collection, HybridQuery, Payload, QueryError, SearchHit, SearchQuery};
use eidosdb_storage::{
    PersistentFlatIndex, PersistentHnswIndex, PersistentLexicalIndex, PersistentPayloadStore,
};

use crate::error::ServerError;

// ---------------------------------------------------------------------------
// Sub-path constants
// ---------------------------------------------------------------------------

const VECTOR_DIR: &str = "vector";
const LEXICAL_FILE: &str = "lexical.redb";
const PAYLOAD_FILE: &str = "payload.redb";

// ---------------------------------------------------------------------------
// Type aliases for the two fully-typed collections
// ---------------------------------------------------------------------------

type FlatCollection =
    Collection<PersistentFlatIndex, PersistentLexicalIndex, PersistentPayloadStore>;
type HnswCollection =
    Collection<PersistentHnswIndex, PersistentLexicalIndex, PersistentPayloadStore>;

// ---------------------------------------------------------------------------
// CollectionKind
// ---------------------------------------------------------------------------

/// Runtime dispatch over the two supported index algorithms.
pub enum CollectionKind {
    /// Exact k-NN backed by a `PersistentFlatIndex`.
    Flat(FlatCollection),
    /// Approximate k-NN backed by a `PersistentHnswIndex`.
    Hnsw(HnswCollection),
}

impl CollectionKind {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Opens (or creates) a flat exact-search collection at `dir`.
    ///
    /// The `vector/` sub-directory is created on first call;
    /// `lexical.redb` and `payload.redb` are created if absent.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError`] if any sub-store fails to open.
    pub fn open_flat(
        dir: &Path,
        metric: Metric,
        dimension: Dimension,
    ) -> Result<Self, ServerError> {
        let vector_dir = dir.join(VECTOR_DIR);
        let lexical_path = dir.join(LEXICAL_FILE);
        let payload_path = dir.join(PAYLOAD_FILE);

        let index = PersistentFlatIndex::open(&vector_dir, metric, dimension)
            .map_err(|e| ServerError::Storage(e.to_string()))?;
        let lexical = PersistentLexicalIndex::open(&lexical_path)
            .map_err(|e| ServerError::Storage(e.to_string()))?;
        let payloads = PersistentPayloadStore::open(&payload_path)
            .map_err(|e| ServerError::Storage(e.to_string()))?;

        Ok(Self::Flat(Collection::new(index, lexical, payloads)))
    }

    /// Creates a new, empty HNSW collection at `dir`.
    ///
    /// Fails if the `vector/` sub-directory already contains an existing index.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError`] if any sub-store fails to create.
    pub fn create_hnsw(
        dir: &Path,
        config: HnswConfig,
        dimension: Dimension,
    ) -> Result<Self, ServerError> {
        let vector_dir = dir.join(VECTOR_DIR);
        let lexical_path = dir.join(LEXICAL_FILE);
        let payload_path = dir.join(PAYLOAD_FILE);

        let index = PersistentHnswIndex::create(&vector_dir, config, dimension)
            .map_err(|e| ServerError::Storage(e.to_string()))?;
        let lexical = PersistentLexicalIndex::open(&lexical_path)
            .map_err(|e| ServerError::Storage(e.to_string()))?;
        let payloads = PersistentPayloadStore::open(&payload_path)
            .map_err(|e| ServerError::Storage(e.to_string()))?;

        Ok(Self::Hnsw(Collection::new(index, lexical, payloads)))
    }

    /// Reopens an existing HNSW collection from `dir`.
    ///
    /// Metric and dimension are reloaded from the on-disk manifest;
    /// `lexical.redb` and `payload.redb` are reopened in place.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError`] if any sub-store fails to open.
    pub fn open_hnsw(dir: &Path) -> Result<Self, ServerError> {
        let vector_dir = dir.join(VECTOR_DIR);
        let lexical_path = dir.join(LEXICAL_FILE);
        let payload_path = dir.join(PAYLOAD_FILE);

        let index = PersistentHnswIndex::open(&vector_dir)
            .map_err(|e| ServerError::Storage(e.to_string()))?;
        let lexical = PersistentLexicalIndex::open(&lexical_path)
            .map_err(|e| ServerError::Storage(e.to_string()))?;
        let payloads = PersistentPayloadStore::open(&payload_path)
            .map_err(|e| ServerError::Storage(e.to_string()))?;

        Ok(Self::Hnsw(Collection::new(index, lexical, payloads)))
    }

    /// Creates an HNSW collection and bulk-inserts `points` in one pass.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError`] if any sub-store fails to create or if
    /// bulk insertion encounters an index error.
    pub fn bulk_load_hnsw(
        dir: &Path,
        config: HnswConfig,
        dimension: Dimension,
        points: impl IntoIterator<Item = (VectorId, Embedding)>,
    ) -> Result<Self, ServerError> {
        let vector_dir = dir.join(VECTOR_DIR);
        let lexical_path = dir.join(LEXICAL_FILE);
        let payload_path = dir.join(PAYLOAD_FILE);

        let index = PersistentHnswIndex::bulk_load(&vector_dir, config, dimension, points)
            .map_err(|e| ServerError::Index(e.to_string()))?;
        let lexical = PersistentLexicalIndex::open(&lexical_path)
            .map_err(|e| ServerError::Storage(e.to_string()))?;
        let payloads = PersistentPayloadStore::open(&payload_path)
            .map_err(|e| ServerError::Storage(e.to_string()))?;

        Ok(Self::Hnsw(Collection::new(index, lexical, payloads)))
    }

    // -----------------------------------------------------------------------
    // Uniform surface (dispatch by match, no _ => arm)
    // -----------------------------------------------------------------------

    /// Number of vectors currently stored in this collection.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Flat(c) => c.len(),
            Self::Hnsw(c) => c.len(),
        }
    }

    /// Returns `true` if the collection contains no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Flat(c) => c.is_empty(),
            Self::Hnsw(c) => c.is_empty(),
        }
    }

    /// Inserts or replaces the vector, its optional document, and its optional
    /// payload identified by `id`.
    ///
    /// Signature note: `document` is `Option<&Document>` (matches
    /// `Collection::upsert`); `payload` is `Option<Payload>` (owned, matching
    /// `Collection::upsert`).
    ///
    /// # Errors
    ///
    /// Propagates [`QueryError`] from the underlying collection.
    pub fn upsert(
        &mut self,
        id: VectorId,
        embedding: Embedding,
        document: Option<&Document>,
        payload: Option<Payload>,
    ) -> Result<(), QueryError> {
        match self {
            Self::Flat(c) => c.upsert(id, embedding, document, payload),
            Self::Hnsw(c) => c.upsert(id, embedding, document, payload),
        }
    }

    /// Removes the vector identified by `id`, returning `true` if it was
    /// present.
    ///
    /// # Errors
    ///
    /// Propagates [`QueryError`] from the underlying collection.
    pub fn delete(&mut self, id: &VectorId) -> Result<bool, QueryError> {
        match self {
            Self::Flat(c) => c.delete(id),
            Self::Hnsw(c) => c.delete(id),
        }
    }

    /// Runs a filtered, metric-aware dense search.
    ///
    /// # Errors
    ///
    /// Propagates [`QueryError`] from the underlying collection.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchHit>, QueryError> {
        match self {
            Self::Flat(c) => c.search(query),
            Self::Hnsw(c) => c.search(query),
        }
    }

    /// Runs a hybrid (dense + lexical) search fused by RRF.
    ///
    /// # Errors
    ///
    /// Propagates [`QueryError`] from the underlying collection.
    pub fn search_hybrid(&self, query: &HybridQuery) -> Result<Vec<SearchHit>, QueryError> {
        match self {
            Self::Flat(c) => c.search_hybrid(query),
            Self::Hnsw(c) => c.search_hybrid(query),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::CollectionKind;
    use eidosdb_core::{Dimension, Embedding, Metric, VectorId};
    use eidosdb_hnsw::HnswConfig;
    use eidosdb_query::SearchQuery;
    use tempfile::tempdir;

    fn dim() -> Dimension {
        Dimension(4)
    }

    fn emb(values: [f32; 4]) -> Embedding {
        Embedding::new(values.to_vec()).expect("valid embedding")
    }

    fn hnsw_config() -> HnswConfig {
        HnswConfig {
            metric: Metric::Cosine,
            ..HnswConfig::default()
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Flat upsert + dense search
    // -----------------------------------------------------------------------

    #[test]
    fn collection_kind_flat_upsert_and_search() {
        let dir = tempdir().expect("tempdir");
        let mut kind =
            CollectionKind::open_flat(dir.path(), Metric::Cosine, dim()).expect("open_flat");

        assert!(kind.is_empty());

        let id = VectorId::new();
        kind.upsert(id, emb([1.0, 0.0, 0.0, 0.0]), None, None)
            .expect("upsert");

        assert_eq!(kind.len(), 1);

        let hits = kind
            .search(&SearchQuery {
                embedding: emb([1.0, 0.0, 0.0, 0.0]),
                k: 1,
                metric: None,
                filter: None,
            })
            .expect("search");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id);
    }

    // -----------------------------------------------------------------------
    // Test 2: HNSW upsert + dense search
    // -----------------------------------------------------------------------

    #[test]
    fn collection_kind_hnsw_upsert_and_search() {
        let dir = tempdir().expect("tempdir");
        let mut kind =
            CollectionKind::create_hnsw(dir.path(), hnsw_config(), dim()).expect("create_hnsw");

        assert!(kind.is_empty());

        let id = VectorId::new();
        kind.upsert(id, emb([0.0, 1.0, 0.0, 0.0]), None, None)
            .expect("upsert");

        assert_eq!(kind.len(), 1);

        let hits = kind
            .search(&SearchQuery {
                embedding: emb([0.0, 1.0, 0.0, 0.0]),
                k: 1,
                metric: None,
                filter: None,
            })
            .expect("search");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id);
    }

    // -----------------------------------------------------------------------
    // Test 3: HNSW bulk_load
    // -----------------------------------------------------------------------

    #[test]
    fn collection_kind_hnsw_bulk_load() {
        let dir = tempdir().expect("tempdir");
        let id_a = VectorId::new();
        let id_b = VectorId::new();
        let points = vec![
            (id_a, emb([1.0, 0.0, 0.0, 0.0])),
            (id_b, emb([0.0, 1.0, 0.0, 0.0])),
        ];

        let kind = CollectionKind::bulk_load_hnsw(dir.path(), hnsw_config(), dim(), points)
            .expect("bulk_load_hnsw");

        assert_eq!(kind.len(), 2);
        assert!(!kind.is_empty());

        let hits = kind
            .search(&SearchQuery {
                embedding: emb([1.0, 0.0, 0.0, 0.0]),
                k: 1,
                metric: None,
                filter: None,
            })
            .expect("search after bulk_load");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id_a);
    }
}
