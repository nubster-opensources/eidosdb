//! The store manifest: fixed parameters plus the segment watermark.

use crate::error::StorageError;
use eidosdb_core::Metric;

/// Serialized length of a manifest record, in bytes.
pub const MANIFEST_LEN: usize = 17;

/// On-disk format version for both the manifest and the segment.
pub const FORMAT_VERSION: u32 = 1;

/// Fixed parameters and mutable watermark of a persistent index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Manifest {
    /// On-disk format version.
    pub format_version: u32,
    /// Number of components per stored vector.
    pub dimension: u32,
    /// Metric used to score queries.
    pub metric: Metric,
    /// Number of valid records in the segment (the durability watermark).
    pub record_count: u64,
}

impl Manifest {
    /// Serializes the manifest to its fixed-length representation.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::FormatMismatch`] when the metric has no
    /// on-disk encoding (see [`metric_to_u8`]).
    pub fn to_bytes(self) -> Result<[u8; MANIFEST_LEN], StorageError> {
        let mut bytes = [0u8; MANIFEST_LEN];
        bytes[0..4].copy_from_slice(&self.format_version.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.dimension.to_le_bytes());
        bytes[8] = metric_to_u8(self.metric)?;
        bytes[9..17].copy_from_slice(&self.record_count.to_le_bytes());
        Ok(bytes)
    }

    /// Parses a manifest from its on-disk representation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, StorageError> {
        if bytes.len() != MANIFEST_LEN {
            return Err(StorageError::Corruption(format!(
                "manifest length {} != {MANIFEST_LEN}",
                bytes.len()
            )));
        }
        let format_version = read_u32(&bytes[0..4])?;
        let dimension = read_u32(&bytes[4..8])?;
        let metric = metric_from_u8(bytes[8])?;
        let record_count = read_u64(&bytes[9..17])?;
        Ok(Self {
            format_version,
            dimension,
            metric,
            record_count,
        })
    }
}

fn read_u32(bytes: &[u8]) -> Result<u32, StorageError> {
    let array: [u8; 4] = bytes
        .try_into()
        .map_err(|_| StorageError::Corruption("expected 4 bytes".to_string()))?;
    Ok(u32::from_le_bytes(array))
}

fn read_u64(bytes: &[u8]) -> Result<u64, StorageError> {
    let array: [u8; 8] = bytes
        .try_into()
        .map_err(|_| StorageError::Corruption("expected 8 bytes".to_string()))?;
    Ok(u64::from_le_bytes(array))
}

/// Encodes a metric as a single byte.
///
/// # Errors
///
/// `Metric` is `#[non_exhaustive]`: a variant added after this format was
/// frozen has no assigned byte, and is rejected with
/// [`StorageError::FormatMismatch`] before anything reaches the disk, so a
/// store is never written in a shape it cannot read back.
pub fn metric_to_u8(metric: Metric) -> Result<u8, StorageError> {
    match metric {
        Metric::Cosine => Ok(0),
        Metric::DotProduct => Ok(1),
        Metric::Euclidean => Ok(2),
        other => Err(StorageError::FormatMismatch(format!(
            "metric {other:?} has no on-disk encoding"
        ))),
    }
}

/// Decodes a metric from a single byte.
pub fn metric_from_u8(value: u8) -> Result<Metric, StorageError> {
    match value {
        0 => Ok(Metric::Cosine),
        1 => Ok(Metric::DotProduct),
        2 => Ok(Metric::Euclidean),
        other => Err(StorageError::Corruption(format!(
            "unknown metric byte {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{FORMAT_VERSION, Manifest, metric_from_u8};
    use crate::error::StorageError;
    use eidosdb_core::Metric;

    #[test]
    fn round_trips_through_bytes() {
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            dimension: 768,
            metric: Metric::Cosine,
            record_count: 42,
        };
        let parsed = Manifest::from_bytes(&manifest.to_bytes().expect("encode")).expect("parse");
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn every_metric_round_trips() {
        for metric in [Metric::Cosine, Metric::DotProduct, Metric::Euclidean] {
            let manifest = Manifest {
                format_version: FORMAT_VERSION,
                dimension: 4,
                metric,
                record_count: 0,
            };
            let parsed =
                Manifest::from_bytes(&manifest.to_bytes().expect("encode")).expect("parse");
            assert_eq!(parsed.metric, metric);
        }
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(matches!(
            Manifest::from_bytes(&[0u8; 3]),
            Err(StorageError::Corruption(_))
        ));
    }

    #[test]
    fn rejects_unknown_metric_byte() {
        assert!(matches!(
            metric_from_u8(9),
            Err(StorageError::Corruption(_))
        ));
    }
}
