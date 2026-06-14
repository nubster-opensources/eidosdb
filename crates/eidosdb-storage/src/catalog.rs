//! redb-backed catalog: the manifest row and the `VectorId -> slot` table.

use crate::error::StorageError;
use crate::manifest::Manifest;
use eidosdb_core::VectorId;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::path::Path;
use uuid::Uuid;

const MANIFEST: TableDefinition<&str, &[u8]> = TableDefinition::new("manifest");
const SLOTS: TableDefinition<&[u8], u64> = TableDefinition::new("slots");
const MANIFEST_KEY: &str = "manifest";

/// Owns the redb database and exposes manifest and slot operations.
pub struct Catalog {
    db: Database,
}

impl Catalog {
    /// Creates a new catalog with the given manifest and an empty slot table.
    pub fn create(path: &Path, manifest: &Manifest) -> Result<Self, StorageError> {
        let db = Database::create(path).map_err(catalog_err)?;
        let catalog = Self { db };
        let txn = catalog.db.begin_write().map_err(catalog_err)?;
        {
            let mut table = txn.open_table(MANIFEST).map_err(catalog_err)?;
            table
                .insert(MANIFEST_KEY, manifest.to_bytes().as_slice())
                .map_err(catalog_err)?;
            // Touch the slots table so it exists on disk.
            let _ = txn.open_table(SLOTS).map_err(catalog_err)?;
        }
        txn.commit().map_err(catalog_err)?;
        Ok(catalog)
    }

    /// Opens an existing catalog and returns its manifest.
    pub fn open(path: &Path) -> Result<(Self, Manifest), StorageError> {
        let db = Database::open(path).map_err(catalog_err)?;
        let catalog = Self { db };
        let manifest = catalog.read_manifest()?;
        Ok((catalog, manifest))
    }

    /// Reads the current manifest.
    pub fn read_manifest(&self) -> Result<Manifest, StorageError> {
        let txn = self.db.begin_read().map_err(catalog_err)?;
        let table = txn.open_table(MANIFEST).map_err(catalog_err)?;
        let row = table
            .get(MANIFEST_KEY)
            .map_err(catalog_err)?
            .ok_or_else(|| StorageError::Corruption("missing manifest row".to_string()))?;
        Manifest::from_bytes(row.value())
    }

    /// Number of live (non-tombstoned) slots.
    pub fn live_count(&self) -> Result<usize, StorageError> {
        let txn = self.db.begin_read().map_err(catalog_err)?;
        let table = txn.open_table(SLOTS).map_err(catalog_err)?;
        let len = table.len().map_err(catalog_err)?;
        usize::try_from(len)
            .map_err(|_| StorageError::Corruption("slot count exceeds usize".to_string()))
    }

    /// Returns `true` if `id` is present (live) in the slot table.
    pub fn contains(&self, id: VectorId) -> Result<bool, StorageError> {
        let key = id.as_uuid().into_bytes();
        let txn = self.db.begin_read().map_err(catalog_err)?;
        let table = txn.open_table(SLOTS).map_err(catalog_err)?;
        Ok(table.get(key.as_slice()).map_err(catalog_err)?.is_some())
    }

    /// Inserts one slot and updates the manifest watermark, atomically.
    pub fn insert_slot(
        &self,
        id: VectorId,
        slot: u64,
        new_record_count: u64,
    ) -> Result<(), StorageError> {
        self.insert_slots(&[(id, slot)], new_record_count)
    }

    /// Inserts many slots and updates the watermark in a single transaction.
    pub fn insert_slots(
        &self,
        items: &[(VectorId, u64)],
        new_record_count: u64,
    ) -> Result<(), StorageError> {
        let txn = self.db.begin_write().map_err(catalog_err)?;
        {
            let mut slots = txn.open_table(SLOTS).map_err(catalog_err)?;
            for (id, slot) in items {
                let key = id.as_uuid().into_bytes();
                slots.insert(key.as_slice(), *slot).map_err(catalog_err)?;
            }
            let mut manifest_table = txn.open_table(MANIFEST).map_err(catalog_err)?;
            // Read the current manifest, bump the watermark, write it back.
            let row = manifest_table
                .get(MANIFEST_KEY)
                .map_err(catalog_err)?
                .ok_or_else(|| StorageError::Corruption("missing manifest row".to_string()))?;
            let mut manifest = Manifest::from_bytes(row.value())?;
            // Drop the guard before the mutable borrow of `manifest_table`.
            drop(row);
            manifest.record_count = new_record_count;
            manifest_table
                .insert(MANIFEST_KEY, manifest.to_bytes().as_slice())
                .map_err(catalog_err)?;
        }
        txn.commit().map_err(catalog_err)?;
        Ok(())
    }

    /// Removes a slot (tombstone). Returns `true` if the id was present.
    pub fn remove_slot(&self, id: VectorId) -> Result<bool, StorageError> {
        let key = id.as_uuid().into_bytes();
        let txn = self.db.begin_write().map_err(catalog_err)?;
        let existed;
        {
            let mut slots = txn.open_table(SLOTS).map_err(catalog_err)?;
            existed = slots.remove(key.as_slice()).map_err(catalog_err)?.is_some();
        }
        txn.commit().map_err(catalog_err)?;
        Ok(existed)
    }

    /// Collects all live `(id, slot)` pairs into an owned vector.
    pub fn live_slots(&self) -> Result<Vec<(VectorId, u64)>, StorageError> {
        let txn = self.db.begin_read().map_err(catalog_err)?;
        let table = txn.open_table(SLOTS).map_err(catalog_err)?;
        let mut out = Vec::new();
        for entry in table.iter().map_err(catalog_err)? {
            let (key, value) = entry.map_err(catalog_err)?;
            let bytes: [u8; 16] = key
                .value()
                .try_into()
                .map_err(|_| StorageError::Corruption("slot key is not 16 bytes".to_string()))?;
            out.push((VectorId::from_uuid(Uuid::from_bytes(bytes)), value.value()));
        }
        Ok(out)
    }
}

fn catalog_err<E: std::fmt::Display>(error: E) -> StorageError {
    StorageError::Catalog(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::Catalog;
    use crate::manifest::{Manifest, FORMAT_VERSION};
    use eidosdb_core::{Metric, VectorId};
    use tempfile::tempdir;

    fn manifest() -> Manifest {
        Manifest {
            format_version: FORMAT_VERSION,
            dimension: 2,
            metric: Metric::Cosine,
            record_count: 0,
        }
    }

    #[test]
    fn create_then_open_reads_manifest() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("meta.redb");
        Catalog::create(&path, &manifest()).expect("create");
        let (_catalog, parsed) = Catalog::open(&path).expect("open");
        assert_eq!(parsed, manifest());
    }

    #[test]
    fn insert_and_remove_slot_tracks_presence() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("meta.redb");
        let catalog = Catalog::create(&path, &manifest()).expect("create");
        let id = VectorId::new();
        catalog.insert_slot(id, 0, 1).expect("insert");
        assert!(catalog.contains(id).expect("contains"));
        assert_eq!(catalog.live_count().expect("count"), 1);
        assert_eq!(catalog.read_manifest().expect("manifest").record_count, 1);
        assert!(catalog.remove_slot(id).expect("remove"));
        assert!(!catalog.contains(id).expect("contains"));
        assert!(!catalog.remove_slot(id).expect("remove again"));
    }

    #[test]
    fn live_slots_lists_all_pairs() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("meta.redb");
        let catalog = Catalog::create(&path, &manifest()).expect("create");
        let a = VectorId::new();
        let b = VectorId::new();
        catalog.insert_slots(&[(a, 0), (b, 1)], 2).expect("batch insert");
        let mut pairs = catalog.live_slots().expect("live");
        pairs.sort_by_key(|(_, slot)| *slot);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], (a, 0));
        assert_eq!(pairs[1], (b, 1));
    }
}
