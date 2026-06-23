//! Conversions between protobuf wire types and domain point types.
//!
//! Covers [`pb::Point`] mapped to and from the domain types
//! [`VectorId`], [`Embedding`], [`Document`], and [`Payload`].

use crate::convert::{payload_from_pb, payload_to_pb};
use crate::error::ConversionError;
use crate::pb;
use eidosdb_core::{Embedding, VectorId};
use eidosdb_lexical::Document;
use eidosdb_query::Payload;
use uuid::Uuid;

/// A point decoded from the wire representation, with validated domain types.
pub struct DecodedPoint {
    /// The vector identifier.
    pub id: VectorId,
    /// The validated embedding.
    pub embedding: Embedding,
    /// Optional searchable document text.
    pub document: Option<Document>,
    /// Optional schemaless payload.
    pub payload: Option<Payload>,
}

/// Converts a [`VectorId`] to its protobuf wire representation (hyphenated UUID string).
#[must_use]
pub fn vector_id_to_pb(id: VectorId) -> String {
    id.as_uuid().to_string()
}

/// Parses a UUID string from the wire and wraps it as a [`VectorId`].
///
/// Returns [`ConversionError::InvalidUuid`] when the string is not a valid UUID.
pub fn vector_id_from_pb(id: &str) -> Result<VectorId, ConversionError> {
    Uuid::parse_str(id)
        .map(VectorId::from_uuid)
        .map_err(|_| ConversionError::InvalidUuid(id.to_string()))
}

/// Copies an [`Embedding`]'s components into a `Vec<f32>` for wire transmission.
#[must_use]
pub fn embedding_to_pb(embedding: &Embedding) -> Vec<f32> {
    embedding.as_slice().to_vec()
}

/// Builds a validated [`Embedding`] from a wire vector.
///
/// Returns [`ConversionError::Domain`] when the domain layer rejects the vector
/// (empty or containing non-finite components).
pub fn embedding_from_pb(vector: Vec<f32>) -> Result<Embedding, ConversionError> {
    Embedding::new(vector).map_err(|e| ConversionError::Domain(e.to_string()))
}

/// Decodes a [`pb::Point`] into a validated [`DecodedPoint`].
///
/// Returns [`ConversionError::InvalidUuid`] when the `id` field is not a valid UUID,
/// [`ConversionError::Domain`] when the embedding or document is rejected by the domain,
/// or [`ConversionError::MissingField`] when a payload field value has no `kind`.
pub fn point_from_pb(point: pb::Point) -> Result<DecodedPoint, ConversionError> {
    let id = vector_id_from_pb(&point.id)?;
    let embedding = embedding_from_pb(point.vector)?;
    let document = point
        .document
        .map(Document::new)
        .transpose()
        .map_err(|e| ConversionError::Domain(e.to_string()))?;
    let payload = point.payload.map(payload_from_pb).transpose()?;
    Ok(DecodedPoint {
        id,
        embedding,
        document,
        payload,
    })
}

/// Encodes domain types into a [`pb::Point`] for wire transmission.
#[must_use]
pub fn point_to_pb(
    id: VectorId,
    embedding: &Embedding,
    document: Option<&Document>,
    payload: Option<&Payload>,
) -> pb::Point {
    pb::Point {
        id: vector_id_to_pb(id),
        vector: embedding_to_pb(embedding),
        document: document.map(|d| d.as_str().to_string()),
        payload: payload.map(payload_to_pb),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidosdb_core::{Embedding, VectorId};
    use eidosdb_lexical::Document;

    #[test]
    fn vector_id_round_trips() {
        let id = VectorId::new();
        assert_eq!(vector_id_from_pb(&vector_id_to_pb(id)).expect("uuid"), id);
    }

    #[test]
    fn bad_uuid_is_rejected() {
        assert!(matches!(
            vector_id_from_pb("not-a-uuid"),
            Err(ConversionError::InvalidUuid(_))
        ));
    }

    #[test]
    fn empty_vector_is_rejected() {
        assert!(embedding_from_pb(vec![]).is_err());
    }

    #[test]
    fn non_finite_vector_is_rejected() {
        assert!(embedding_from_pb(vec![1.0, f32::NAN, 0.0]).is_err());
    }

    #[test]
    fn point_round_trips_with_document_and_payload() {
        let id = VectorId::new();
        let emb = Embedding::new(vec![1.0, 0.0, 0.0]).expect("embedding");
        let doc = Document::new("hello world").expect("document");
        let pb_point = point_to_pb(id, &emb, Some(&doc), None);
        let decoded = point_from_pb(pb_point).expect("decode");
        assert_eq!(decoded.id, id);
        assert_eq!(decoded.embedding.as_slice(), emb.as_slice());
        assert_eq!(decoded.document.expect("doc").as_str(), "hello world");
        assert!(decoded.payload.is_none());
    }
}
