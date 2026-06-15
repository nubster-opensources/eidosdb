//! `PersistentPayloadStore`: a durable `PayloadStore` backed by redb, with
//! payloads serialized via postcard.

use eidosdb_core::VectorId;
use eidosdb_query::{CompiledFilter, Payload, PayloadError, PayloadStore};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::collections::HashSet;
use std::fmt::Display;
use std::path::Path;
use uuid::Uuid;

const PAYLOADS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("payloads");

/// A redb-backed payload store. Keys are 16-byte `VectorId`s; values are
/// postcard-encoded payloads.
pub struct PersistentPayloadStore {
    db: Database,
    count: usize,
}

impl PersistentPayloadStore {
    /// Opens a payload store at `path`, creating it if absent.
    pub fn open(path: &Path) -> Result<Self, PayloadError> {
        let db = Database::create(path).map_err(backend)?;
        let txn = db.begin_write().map_err(backend)?;
        {
            let _ = txn.open_table(PAYLOADS).map_err(backend)?;
        }
        txn.commit().map_err(backend)?;
        let count = {
            let txn = db.begin_read().map_err(backend)?;
            let table = txn.open_table(PAYLOADS).map_err(backend)?;
            let len = table.len().map_err(backend)?;
            usize::try_from(len).map_err(|_| PayloadError::Backend("count exceeds usize".into()))?
        };
        Ok(Self { db, count })
    }
}

fn backend<E: Display>(error: E) -> PayloadError {
    PayloadError::Backend(error.to_string())
}

fn encode(payload: &Payload) -> Result<Vec<u8>, PayloadError> {
    postcard::to_allocvec(payload).map_err(|e| PayloadError::Serialization(e.to_string()))
}

fn decode(bytes: &[u8]) -> Result<Payload, PayloadError> {
    postcard::from_bytes(bytes).map_err(|e| PayloadError::Serialization(e.to_string()))
}

impl PayloadStore for PersistentPayloadStore {
    fn set(&mut self, id: VectorId, payload: Payload) -> Result<(), PayloadError> {
        let bytes = encode(&payload)?;
        let key = id.as_uuid().into_bytes();
        let txn = self.db.begin_write().map_err(backend)?;
        let was_new;
        {
            let mut table = txn.open_table(PAYLOADS).map_err(backend)?;
            let prior = table
                .insert(key.as_slice(), bytes.as_slice())
                .map_err(backend)?;
            was_new = prior.is_none();
        }
        txn.commit().map_err(backend)?;
        if was_new {
            self.count += 1;
        }
        Ok(())
    }

    fn get(&self, id: &VectorId) -> Result<Option<Payload>, PayloadError> {
        let key = id.as_uuid().into_bytes();
        let txn = self.db.begin_read().map_err(backend)?;
        let table = txn.open_table(PAYLOADS).map_err(backend)?;
        match table.get(key.as_slice()).map_err(backend)? {
            Some(row) => Ok(Some(decode(row.value())?)),
            None => Ok(None),
        }
    }

    fn remove(&mut self, id: &VectorId) -> Result<bool, PayloadError> {
        let key = id.as_uuid().into_bytes();
        let txn = self.db.begin_write().map_err(backend)?;
        let existed;
        {
            let mut table = txn.open_table(PAYLOADS).map_err(backend)?;
            existed = table.remove(key.as_slice()).map_err(backend)?.is_some();
        }
        txn.commit().map_err(backend)?;
        if existed {
            self.count -= 1;
        }
        Ok(existed)
    }

    fn len(&self) -> usize {
        self.count
    }

    fn matching_ids(&self, filter: &CompiledFilter) -> Result<HashSet<VectorId>, PayloadError> {
        let txn = self.db.begin_read().map_err(backend)?;
        let table = txn.open_table(PAYLOADS).map_err(backend)?;
        let mut matched = HashSet::new();
        for entry in table.iter().map_err(backend)? {
            let (key, value) = entry.map_err(backend)?;
            let bytes: [u8; 16] = key
                .value()
                .try_into()
                .map_err(|_| PayloadError::Backend("payload key is not 16 bytes".into()))?;
            let payload = decode(value.value())?;
            if filter.matches(&payload) {
                matched.insert(VectorId::from_uuid(Uuid::from_bytes(bytes)));
            }
        }
        Ok(matched)
    }
}

#[cfg(test)]
mod tests {
    use super::PersistentPayloadStore;
    use eidosdb_core::VectorId;
    use eidosdb_query::{FieldValue, Filter, InMemoryPayloadStore, Payload, PayloadStore, Value};
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    fn payload(source: &str, score: i64) -> Payload {
        let mut map = BTreeMap::new();
        map.insert(
            "source".to_string(),
            FieldValue::Scalar(Value::Text(source.into())),
        );
        map.insert(
            "score".to_string(),
            FieldValue::Scalar(Value::Integer(score)),
        );
        Payload::new(map).expect("valid")
    }

    #[test]
    fn set_get_remove_round_trip() {
        let dir = tempdir().expect("tempdir");
        let mut store = PersistentPayloadStore::open(&dir.path().join("p.redb")).expect("open");
        let id = VectorId::new();
        store.set(id, payload("wiki", 3)).expect("set");
        assert_eq!(store.get(&id).expect("get"), Some(payload("wiki", 3)));
        assert_eq!(store.len(), 1);
        assert!(store.remove(&id).expect("remove"));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn set_overwrites_without_double_counting() {
        let dir = tempdir().expect("tempdir");
        let mut store = PersistentPayloadStore::open(&dir.path().join("p.redb")).expect("open");
        let id = VectorId::new();
        store.set(id, payload("a", 1)).expect("set");
        store.set(id, payload("b", 2)).expect("overwrite");
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(&id).expect("get"), Some(payload("b", 2)));
    }

    #[test]
    fn reopen_preserves_payloads() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("p.redb");
        let id = VectorId::new();
        {
            let mut store = PersistentPayloadStore::open(&path).expect("open");
            store.set(id, payload("wiki", 5)).expect("set");
        }
        let store = PersistentPayloadStore::open(&path).expect("reopen");
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(&id).expect("get"), Some(payload("wiki", 5)));
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn persistent_matches_in_memory_oracle(
            rows in proptest::collection::vec((0u8..4, 0i64..50), 1..20),
            threshold in 0i64..50,
        ) {
            let dir = tempdir().expect("tempdir");
            let mut persistent =
                PersistentPayloadStore::open(&dir.path().join("p.redb")).expect("open");
            let mut oracle = InMemoryPayloadStore::new();
            for (bucket, score) in &rows {
                let id = VectorId::new();
                let p = payload(&format!("b{bucket}"), *score);
                persistent.set(id, p.clone()).expect("p set");
                oracle.set(id, p).expect("o set");
            }
            let filter = Filter::Gte("score".into(), Value::Integer(threshold)).compile();
            let mut got: Vec<VectorId> =
                persistent.matching_ids(&filter).expect("p match").into_iter().collect();
            let mut want: Vec<VectorId> =
                oracle.matching_ids(&filter).expect("o match").into_iter().collect();
            got.sort();
            want.sort();
            prop_assert_eq!(got, want);
        }
    }
}
