//! Collection registry: owns all named collections and manages their lifecycle.
//!
//! The [`Registry`] holds a map of collection names to [`CollectionHandle`]s.
//! It does not scan the disk on startup — that is deferred to B4 (reload).

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use crate::{
    collection_kind::CollectionKind,
    error::ServerError,
    meta::{CollectionMeta, write_meta},
};
use eidosdb_proto::convert::IndexTypeChoice;

// ---------------------------------------------------------------------------
// CollectionHandle
// ---------------------------------------------------------------------------

/// A live, thread-safe handle to an open collection.
///
/// `inner` holds the [`CollectionKind`] behind a per-collection [`RwLock`] so
/// that B5 can acquire read or write locks on individual collections without
/// blocking unrelated ones.
pub struct CollectionHandle {
    /// The underlying index, protected by a per-collection lock.
    pub inner: RwLock<CollectionKind>,
    /// Snapshot of the metadata that was used to create or reload this handle.
    pub meta: CollectionMeta,
    /// Absolute path to the collection directory on disk.
    pub dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry of named collections.
///
/// Thread-safe: the internal map is protected by an [`RwLock`].  Individual
/// collections are themselves wrapped in a per-collection [`RwLock`] (see
/// [`CollectionHandle`]).
pub struct Registry {
    root: PathBuf,
    map: RwLock<HashMap<String, Arc<CollectionHandle>>>,
}

impl Registry {
    /// Creates a new, empty registry rooted at `root`.
    ///
    /// The directory is not scanned; existing collections are not loaded.
    /// Call the B4 reload routine to re-populate from disk.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            map: RwLock::new(HashMap::new()),
        }
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Creates a new collection described by `meta`.
    ///
    /// Steps performed in order:
    /// 1. Validate `meta.name` — returns [`ServerError::BadName`] on failure.
    /// 2. Reject duplicates — returns [`ServerError::AlreadyExists`] if the
    ///    name is already registered.
    /// 3. Create the collection directory on disk.
    /// 4. Write `collection.meta` to disk.
    /// 5. Instantiate the [`CollectionKind`].
    /// 6. Insert the handle into the map.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError`] if name validation fails, the name is already
    /// taken, any I/O operation fails, or the index cannot be created.
    pub fn create(&self, meta: CollectionMeta) -> Result<(), ServerError> {
        // Step 1 — validate name before touching the disk.
        if !is_valid_name(&meta.name) {
            return Err(ServerError::BadName(meta.name.clone()));
        }

        // Step 2 — reject duplicates (read lock only).
        {
            let guard = self
                .map
                .read()
                .map_err(|_| ServerError::Storage("lock poisoned".into()))?;
            if guard.contains_key(&meta.name) {
                return Err(ServerError::AlreadyExists(meta.name.clone()));
            }
        }

        // Step 3 — create directory.
        let dir = self.root.join(&meta.name);
        std::fs::create_dir_all(&dir).map_err(|e| ServerError::Io(e.to_string()))?;

        // Step 4 — persist metadata.
        write_meta(&dir, &meta)?;

        // Step 5 — instantiate the index (outside the write lock).
        // Clone the fields needed to build the index so that `meta` stays intact
        // and can be moved into CollectionHandle below.
        let kind = match meta.index_type {
            IndexTypeChoice::Flat => CollectionKind::open_flat(&dir, meta.metric, meta.dimension)?,
            IndexTypeChoice::Hnsw => {
                let config = meta.hnsw.unwrap_or_default();
                CollectionKind::create_hnsw(&dir, config, meta.dimension)?
            }
        };

        // Step 6 — insert into map (brief write lock).
        let handle = Arc::new(CollectionHandle {
            inner: RwLock::new(kind),
            meta,
            dir,
        });
        let mut guard = self
            .map
            .write()
            .map_err(|_| ServerError::Storage("lock poisoned".into()))?;
        guard.insert(handle.meta.name.clone(), handle);

        Ok(())
    }

    /// Removes a collection from the registry and deletes its directory.
    ///
    /// Returns `Ok(true)` if the collection existed and was removed,
    /// `Ok(false)` if the name was not registered (idempotent).
    ///
    /// # Errors
    ///
    /// Returns [`ServerError`] if the map lock is poisoned or if the
    /// directory cannot be deleted.
    pub fn drop_collection(&self, name: &str) -> Result<bool, ServerError> {
        let handle = {
            let mut guard = self
                .map
                .write()
                .map_err(|_| ServerError::Storage("lock poisoned".into()))?;
            guard.remove(name)
        };

        match handle {
            None => Ok(false),
            Some(h) => {
                std::fs::remove_dir_all(&h.dir).map_err(|e| ServerError::Io(e.to_string()))?;
                Ok(true)
            }
        }
    }

    /// Returns a snapshot of all collection metadata, in unspecified order.
    ///
    /// # Panics
    ///
    /// Does not panic; returns an empty `Vec` if the map lock is poisoned.
    #[must_use]
    pub fn list(&self) -> Vec<CollectionMeta> {
        self.map
            .read()
            .map(|g| g.values().map(|h| h.meta.clone()).collect())
            .unwrap_or_default()
    }

    /// Returns a clone of the [`Arc`]-wrapped handle for `name`, or `None`
    /// if no such collection is registered.
    ///
    /// Returns `None` if the collection is absent, or if the map lock is poisoned
    /// (lock-poison implies the server is in an unrecoverable state).
    ///
    /// Cloning an [`Arc`] is `O(1)` and does not hold the map lock after return.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<CollectionHandle>> {
        self.map.read().ok()?.get(name).map(Arc::clone)
    }
}

// ---------------------------------------------------------------------------
// Name validation
// ---------------------------------------------------------------------------

/// Returns `true` if `name` is a valid collection name.
///
/// A valid name is non-empty, at most 64 characters, and contains only
/// ASCII letters, digits, underscores (`_`), or hyphens (`-`).
///
/// This rule intentionally excludes `/`, `\`, `.`, `..`, and spaces, which
/// prevents path-traversal attacks when the name is used as a directory name.
#[must_use]
pub fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use eidosdb_core::{Dimension, Metric};
    use eidosdb_hnsw::HnswConfig;
    use eidosdb_proto::convert::IndexTypeChoice;

    fn hnsw_meta(name: &str) -> CollectionMeta {
        CollectionMeta {
            name: name.to_string(),
            metric: Metric::Cosine,
            dimension: Dimension(3),
            index_type: IndexTypeChoice::Hnsw,
            hnsw: Some(HnswConfig::default()),
        }
    }

    #[test]
    fn create_then_get_and_list() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reg = Registry::new(dir.path().to_path_buf());
        reg.create(hnsw_meta("notes")).expect("create");
        assert!(reg.get("notes").is_some());
        assert_eq!(reg.list().len(), 1);
    }

    #[test]
    fn duplicate_create_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reg = Registry::new(dir.path().to_path_buf());
        reg.create(hnsw_meta("notes")).expect("create");
        assert!(reg.create(hnsw_meta("notes")).is_err());
    }

    #[test]
    fn bad_name_with_separator_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reg = Registry::new(dir.path().to_path_buf());
        assert!(reg.create(hnsw_meta("../escape")).is_err());
        assert!(reg.create(hnsw_meta("a/b")).is_err());
    }

    #[test]
    fn drop_removes_collection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reg = Registry::new(dir.path().to_path_buf());
        reg.create(hnsw_meta("notes")).expect("create");
        assert!(reg.drop_collection("notes").expect("drop"));
        assert!(reg.get("notes").is_none());
        assert!(!reg.drop_collection("notes").expect("drop again"));
    }

    #[test]
    fn valid_name_rules() {
        assert!(is_valid_name("notes"));
        assert!(is_valid_name("my-collection_2"));
        assert!(!is_valid_name(""));
        assert!(!is_valid_name("../x"));
        assert!(!is_valid_name("a b"));
    }
}
