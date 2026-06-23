//! Conversions for search request and response types.
//!
//! Covers [`pb::SearchRequest`] and [`pb::SearchHybridRequest`] decoded into
//! domain query types, and [`eidosdb_query::SearchHit`] encoded into [`pb::Hit`].

use crate::convert::{
    embedding_from_pb, filter_from_pb, metric_from_pb, payload_to_pb, vector_id_to_pb,
};
use crate::error::ConversionError;
use crate::pb;
use eidosdb_core::Metric;
use eidosdb_query::{DEFAULT_OVERFETCH_FACTOR, DEFAULT_RRF_K, HybridQuery, SearchHit, SearchQuery};

/// Converts an optional `i32` metric field from the wire into an optional domain [`Metric`].
///
/// Returns `Ok(None)` when the value is absent or the unspecified sentinel (0).
/// Returns `Ok(Some(m))` when the value maps to a known metric variant.
/// Returns `Err(Domain(...))` when the value is out of range.
pub(crate) fn optional_metric_from_pb(
    metric: Option<i32>,
) -> Result<Option<Metric>, ConversionError> {
    match metric {
        None => Ok(None),
        Some(raw) => {
            let pb_metric = pb::Metric::try_from(raw).map_err(|_| {
                ConversionError::Domain(format!("unknown metric discriminant: {raw}"))
            })?;
            if pb_metric == pb::Metric::Unspecified {
                Ok(None)
            } else {
                metric_from_pb(pb_metric).map(Some)
            }
        }
    }
}

/// Decodes a [`pb::SearchRequest`] into a collection name and a domain [`SearchQuery`].
///
/// Returns [`ConversionError::Domain`] when the vector is empty or contains non-finite
/// components, or when a filter field is malformed.
pub fn search_query_from_pb(
    request: pb::SearchRequest,
) -> Result<(String, SearchQuery), ConversionError> {
    let embedding = embedding_from_pb(request.vector)?;
    let k =
        usize::try_from(request.k).map_err(|_| ConversionError::Domain("k out of range".into()))?;
    let metric = optional_metric_from_pb(request.metric)?;
    let filter = request.filter.map(filter_from_pb).transpose()?;
    Ok((
        request.collection,
        SearchQuery {
            embedding,
            k,
            metric,
            filter,
        },
    ))
}

/// Decodes a [`pb::SearchHybridRequest`] into a collection name and a domain [`HybridQuery`].
///
/// An empty `vector` becomes `None`; an absent or empty `text` becomes `None`.
/// When `rrf_k <= 0.0` the default [`DEFAULT_RRF_K`] is used.
/// When `overfetch_factor == 0` the default [`DEFAULT_OVERFETCH_FACTOR`] is used.
pub fn hybrid_query_from_pb(
    request: pb::SearchHybridRequest,
) -> Result<(String, HybridQuery), ConversionError> {
    let vector = if request.vector.is_empty() {
        None
    } else {
        Some(embedding_from_pb(request.vector)?)
    };
    let text = match request.text {
        Some(t) if !t.is_empty() => Some(t),
        _ => None,
    };
    let k =
        usize::try_from(request.k).map_err(|_| ConversionError::Domain("k out of range".into()))?;
    let filter = request.filter.map(filter_from_pb).transpose()?;
    let metric = optional_metric_from_pb(request.metric)?;
    let rrf_k = if request.rrf_k <= 0.0 {
        DEFAULT_RRF_K
    } else {
        request.rrf_k
    };
    let overfetch_factor = if request.overfetch_factor == 0 {
        DEFAULT_OVERFETCH_FACTOR
    } else {
        usize::try_from(request.overfetch_factor)
            .map_err(|_| ConversionError::Domain("overfetch_factor out of range".into()))?
    };
    Ok((
        request.collection,
        HybridQuery {
            vector,
            text,
            k,
            filter,
            metric,
            rrf_k,
            overfetch_factor,
        },
    ))
}

/// Encodes a domain [`SearchHit`] into a [`pb::Hit`] for wire transmission.
#[must_use]
pub fn hit_to_pb(hit: &SearchHit) -> pb::Hit {
    pb::Hit {
        id: vector_id_to_pb(hit.id),
        score: hit.score.0,
        payload: hit.payload.as_ref().map(payload_to_pb),
    }
}

/// Encodes a list of domain [`SearchHit`]s into a [`pb::SearchResponse`].
#[must_use]
pub fn hits_to_pb(hits: &[SearchHit]) -> pb::SearchResponse {
    pb::SearchResponse {
        hits: hits.iter().map(hit_to_pb).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidosdb_core::{Score, VectorId};
    use eidosdb_query::{DEFAULT_OVERFETCH_FACTOR, DEFAULT_RRF_K};

    #[test]
    fn search_request_maps_fields() {
        let req = pb::SearchRequest {
            collection: "notes".into(),
            vector: vec![0.1, 0.2, 0.3],
            k: 5,
            metric: Some(pb::Metric::Cosine as i32),
            filter: None,
        };
        let (name, q) = search_query_from_pb(req).expect("valid");
        assert_eq!(name, "notes");
        assert_eq!(q.k, 5);
        assert_eq!(q.embedding.as_slice(), &[0.1_f32, 0.2, 0.3]);
        assert!(q.metric.is_some());
    }

    #[test]
    fn search_request_empty_vector_is_rejected() {
        let req = pb::SearchRequest {
            collection: "n".into(),
            vector: vec![],
            k: 1,
            metric: None,
            filter: None,
        };
        assert!(search_query_from_pb(req).is_err());
    }

    #[test]
    fn search_request_no_metric_is_none() {
        let req = pb::SearchRequest {
            collection: "n".into(),
            vector: vec![1.0],
            k: 1,
            metric: None,
            filter: None,
        };
        let (_, q) = search_query_from_pb(req).expect("valid");
        assert!(q.metric.is_none());
    }

    #[test]
    fn search_request_unspecified_metric_is_none() {
        let req = pb::SearchRequest {
            collection: "n".into(),
            vector: vec![1.0],
            k: 1,
            metric: Some(pb::Metric::Unspecified as i32),
            filter: None,
        };
        let (_, q) = search_query_from_pb(req).expect("valid");
        assert!(q.metric.is_none());
    }

    #[test]
    fn search_request_unknown_metric_is_rejected() {
        let req = pb::SearchRequest {
            collection: "n".into(),
            vector: vec![1.0],
            k: 1,
            metric: Some(999),
            filter: None,
        };
        assert!(search_query_from_pb(req).is_err());
    }

    #[test]
    fn hybrid_defaults_when_zero() {
        let req = pb::SearchHybridRequest {
            collection: "n".into(),
            vector: vec![1.0],
            text: Some("x".into()),
            k: 3,
            filter: None,
            metric: None,
            rrf_k: 0.0,
            overfetch_factor: 0,
        };
        let (_, q) = hybrid_query_from_pb(req).expect("valid");
        assert!(q.rrf_k.to_bits() == DEFAULT_RRF_K.to_bits());
        assert_eq!(q.overfetch_factor, DEFAULT_OVERFETCH_FACTOR);
    }

    #[test]
    fn hybrid_empty_vector_and_text_become_none() {
        let req = pb::SearchHybridRequest {
            collection: "n".into(),
            vector: vec![],
            text: None,
            k: 3,
            filter: None,
            metric: None,
            rrf_k: 60.0,
            overfetch_factor: 4,
        };
        let (_, q) = hybrid_query_from_pb(req).expect("valid");
        assert!(q.vector.is_none());
        assert!(q.text.is_none());
    }

    #[test]
    fn hybrid_empty_text_string_becomes_none() {
        let req = pb::SearchHybridRequest {
            collection: "n".into(),
            vector: vec![1.0],
            text: Some(String::new()),
            k: 3,
            filter: None,
            metric: None,
            rrf_k: 60.0,
            overfetch_factor: 4,
        };
        let (_, q) = hybrid_query_from_pb(req).expect("valid");
        assert!(q.text.is_none());
    }

    #[test]
    fn hit_encodes_id_and_score() {
        let id = VectorId::new();
        let hit = SearchHit {
            id,
            score: Score(0.75),
            payload: None,
        };
        let pb_hit = hit_to_pb(&hit);
        assert_eq!(pb_hit.id, id.as_uuid().to_string());
        assert!((pb_hit.score - 0.75_f32).abs() < f32::EPSILON);
        assert!(pb_hit.payload.is_none());
    }

    #[test]
    fn hits_to_pb_preserves_order() {
        let id1 = VectorId::new();
        let id2 = VectorId::new();
        let hits = vec![
            SearchHit {
                id: id1,
                score: Score(0.9),
                payload: None,
            },
            SearchHit {
                id: id2,
                score: Score(0.5),
                payload: None,
            },
        ];
        let resp = hits_to_pb(&hits);
        assert_eq!(resp.hits.len(), 2);
        assert_eq!(resp.hits[0].id, id1.as_uuid().to_string());
        assert_eq!(resp.hits[1].id, id2.as_uuid().to_string());
    }
}
