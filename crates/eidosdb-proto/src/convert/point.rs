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

/// Length in bytes of a vector identifier on the wire.
const ID_LENGTH: usize = 16;

/// Converts a [`VectorId`] to its protobuf wire representation (16 bytes, network order).
#[must_use]
pub fn vector_id_to_pb(id: VectorId) -> Vec<u8> {
    id.as_uuid().as_bytes().to_vec()
}

/// Decodes a 16-byte wire identifier into a [`VectorId`].
///
/// # Errors
///
/// Returns [`ConversionError::Domain`] when `id` is not exactly 16 bytes long.
pub fn vector_id_from_pb(id: &[u8]) -> Result<VectorId, ConversionError> {
    if id.len() != ID_LENGTH {
        return Err(ConversionError::Domain("id must be 16 bytes".into()));
    }
    Uuid::from_slice(id)
        .map(VectorId::from_uuid)
        .map_err(|_| ConversionError::Domain("id must be 16 bytes".into()))
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
/// Returns [`ConversionError::Domain`] when the `id` field is not exactly 16 bytes,
/// or when the embedding or document is rejected by the domain, or
/// [`ConversionError::MissingField`] when a payload field value has no `kind`.
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
    fn vector_id_round_trips_as_sixteen_bytes() {
        let id = VectorId::new();
        let wire = vector_id_to_pb(id);
        assert_eq!(wire.len(), 16);
        assert_eq!(wire.as_slice(), id.as_uuid().as_bytes());
        assert_eq!(vector_id_from_pb(&wire).expect("16 bytes"), id);
    }

    #[test]
    fn id_of_wrong_length_is_rejected() {
        for length in [15_usize, 17, 36] {
            let wire = vec![7_u8; length];
            assert!(
                matches!(vector_id_from_pb(&wire), Err(ConversionError::Domain(_))),
                "length {length} must be rejected"
            );
        }
    }

    #[test]
    fn empty_id_is_rejected() {
        assert!(matches!(
            vector_id_from_pb(&[]),
            Err(ConversionError::Domain(_))
        ));
    }

    #[test]
    fn textual_uuid_bytes_are_rejected() {
        let text = VectorId::new().as_uuid().to_string();
        assert!(vector_id_from_pb(text.as_bytes()).is_err());
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
