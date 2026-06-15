//! The `PayloadStore` port and an in-memory oracle implementation.

use crate::{CompiledFilter, Payload, PayloadError};
use eidosdb_core::VectorId;
use std::collections::{HashMap, HashSet};

/// Storage for per-vector payloads, with a pre-filter scan.
pub trait PayloadStore {
    /// Stores (or overwrites) the payload for `id`.
    fn set(&mut self, id: VectorId, payload: Payload) -> Result<(), PayloadError>;

    /// Returns the payload for `id`, or `None` if absent.
    fn get(&self, id: &VectorId) -> Result<Option<Payload>, PayloadError>;

    /// Removes the payload for `id`, returning whether it was present.
    fn remove(&mut self, id: &VectorId) -> Result<bool, PayloadError>;

    /// Number of stored payloads.
    fn len(&self) -> usize;

    /// Whether the store holds no payloads.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns every id whose payload satisfies `filter`.
    fn matching_ids(&self, filter: &CompiledFilter) -> Result<HashSet<VectorId>, PayloadError>;
}

/// In-memory `PayloadStore`, the oracle the persistent store is checked against.
#[derive(Default)]
pub struct InMemoryPayloadStore {
    payloads: HashMap<VectorId, Payload>,
}

impl InMemoryPayloadStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl PayloadStore for InMemoryPayloadStore {
    fn set(&mut self, id: VectorId, payload: Payload) -> Result<(), PayloadError> {
        self.payloads.insert(id, payload);
        Ok(())
    }

    fn get(&self, id: &VectorId) -> Result<Option<Payload>, PayloadError> {
        Ok(self.payloads.get(id).cloned())
    }

    fn remove(&mut self, id: &VectorId) -> Result<bool, PayloadError> {
        Ok(self.payloads.remove(id).is_some())
    }

    fn len(&self) -> usize {
        self.payloads.len()
    }

    fn matching_ids(&self, filter: &CompiledFilter) -> Result<HashSet<VectorId>, PayloadError> {
        Ok(self
            .payloads
            .iter()
            .filter(|(_, payload)| filter.matches(payload))
            .map(|(id, _)| *id)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::{InMemoryPayloadStore, PayloadStore};
    use crate::{FieldValue, Filter, Payload, Value};
    use eidosdb_core::VectorId;
    use std::collections::BTreeMap;

    fn payload(source: &str) -> Payload {
        let mut map = BTreeMap::new();
        map.insert(
            "source".to_string(),
            FieldValue::Scalar(Value::Text(source.into())),
        );
        Payload::new(map).expect("valid")
    }

    #[test]
    fn set_get_remove_round_trip() {
        let mut store = InMemoryPayloadStore::new();
        let id = VectorId::new();
        assert_eq!(store.get(&id).expect("get"), None);
        store.set(id, payload("wiki")).expect("set");
        assert_eq!(store.get(&id).expect("get"), Some(payload("wiki")));
        assert_eq!(store.len(), 1);
        assert!(store.remove(&id).expect("remove"));
        assert!(!store.remove(&id).expect("remove again"));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn matching_ids_returns_only_matches() {
        let mut store = InMemoryPayloadStore::new();
        let wiki = VectorId::new();
        let blog = VectorId::new();
        store.set(wiki, payload("wiki")).expect("set wiki");
        store.set(blog, payload("blog")).expect("set blog");
        let filter = Filter::Eq("source".into(), Value::Text("wiki".into())).compile();
        let matched = store.matching_ids(&filter).expect("match");
        assert_eq!(matched.len(), 1);
        assert!(matched.contains(&wiki));
        assert!(!matched.contains(&blog));
    }
}
