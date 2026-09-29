//! Conversions for the point batch shared by the batch and bulk upsert requests.

use crate::convert::{DecodedPoint, point_from_pb};
use crate::error::ConversionError;
use crate::pb;

/// Decodes a [`pb::PointBatch`] into its collection name and validated points.
///
/// # Errors
///
/// Propagates the first [`ConversionError`] raised while decoding a point.
pub fn point_batch_from_pb(
    batch: pb::PointBatch,
) -> Result<(String, Vec<DecodedPoint>), ConversionError> {
    let points = batch
        .points
        .into_iter()
        .map(point_from_pb)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((batch.collection, points))
}

/// Wraps a collection name and wire points into a [`pb::PointBatch`].
#[must_use]
pub fn point_batch_to_pb(collection: &str, points: Vec<pb::Point>) -> pb::PointBatch {
    pb::PointBatch {
        collection: collection.to_string(),
        points,
    }
}

/// Decodes a [`pb::BatchUpsertRequest`] through [`point_batch_from_pb`].
///
/// # Errors
///
/// Returns [`ConversionError::MissingField`] when `batch` is absent, or propagates
/// the error of [`point_batch_from_pb`].
pub fn batch_upsert_from_pb(
    request: pb::BatchUpsertRequest,
) -> Result<(String, Vec<DecodedPoint>), ConversionError> {
    request
        .batch
        .ok_or(ConversionError::MissingField("batch_upsert.batch"))
        .and_then(point_batch_from_pb)
}

/// Decodes one [`pb::BulkUpsertRequest`] message through [`point_batch_from_pb`].
///
/// # Errors
///
/// Returns [`ConversionError::MissingField`] when `batch` is absent, or propagates
/// the error of [`point_batch_from_pb`].
pub fn bulk_upsert_from_pb(
    request: pb::BulkUpsertRequest,
) -> Result<(String, Vec<DecodedPoint>), ConversionError> {
    request
        .batch
        .ok_or(ConversionError::MissingField("bulk_upsert.batch"))
        .and_then(point_batch_from_pb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::point_to_pb;
    use crate::pb;
    use eidosdb_core::{Embedding, VectorId};

    fn sample_batch() -> pb::PointBatch {
        let embedding = Embedding::new(vec![1.0, 0.0, 0.0]).expect("embedding");
        let points = vec![
            point_to_pb(VectorId::new(), &embedding, None, None),
            point_to_pb(VectorId::new(), &embedding, None, None),
        ];
        point_batch_to_pb("notes", points)
    }

    #[test]
    fn batch_and_bulk_requests_decode_through_the_same_conversion() {
        let batch = sample_batch();
        let (batch_collection, batch_points) = batch_upsert_from_pb(pb::BatchUpsertRequest {
            batch: Some(batch.clone()),
        })
        .expect("batch");
        let (bulk_collection, bulk_points) =
            bulk_upsert_from_pb(pb::BulkUpsertRequest { batch: Some(batch) }).expect("bulk");
        assert_eq!(batch_collection, "notes");
        assert_eq!(bulk_collection, "notes");
        let batch_ids: Vec<_> = batch_points.iter().map(|point| point.id).collect();
        let bulk_ids: Vec<_> = bulk_points.iter().map(|point| point.id).collect();
        assert_eq!(batch_ids, bulk_ids);
    }

    #[test]
    fn requests_without_batch_are_rejected() {
        assert!(matches!(
            batch_upsert_from_pb(pb::BatchUpsertRequest { batch: None }),
            Err(ConversionError::MissingField(_))
        ));
        assert!(matches!(
            bulk_upsert_from_pb(pb::BulkUpsertRequest { batch: None }),
            Err(ConversionError::MissingField(_))
        ));
    }

    #[test]
    fn one_bad_id_rejects_the_whole_batch() {
        let mut batch = sample_batch();
        batch.points[1].id = vec![0_u8; 15];
        assert!(point_batch_from_pb(batch).is_err());
    }
}
