//! The vector segment: an append-only file of fixed-size `f32` records.
//!
//! Reads go through a read-only memory map for zero-copy scans. Writes use plain
//! buffered file I/O followed by `fsync`. Separating the two paths keeps the only
//! `unsafe` to a single read-only `Mmap::map`.

use crate::error::StorageError;
use crate::manifest::{FORMAT_VERSION, metric_from_u8, metric_to_u8};
use eidosdb_core::Metric;
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Size of the segment header in bytes.
pub const HEADER_LEN: usize = 32;

const MAGIC: &[u8; 8] = b"EIDOSSEG";

/// An append-only file of dense `f32` records, memory-mapped for reads.
pub struct Segment {
    file: File,
    dimension: usize,
    mmap: Option<Mmap>,
    mapped_records: u64,
}

impl Segment {
    /// Byte stride of one record.
    #[must_use]
    pub fn stride(&self) -> u64 {
        self.dimension as u64 * 4
    }

    /// Number of records currently visible through the memory map.
    #[must_use]
    pub fn mapped_records(&self) -> u64 {
        self.mapped_records
    }

    /// Creates a new, empty segment file with a header.
    pub fn create(path: &Path, metric: Metric, dimension: usize) -> Result<Self, StorageError> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        write_header(&mut file, metric, dimension)?;
        file.sync_all()?;
        let mut segment = Self {
            file,
            dimension,
            mmap: None,
            mapped_records: 0,
        };
        segment.remap(0)?;
        Ok(segment)
    }

    /// Opens an existing segment, validating its header and truncating it to
    /// `record_count` valid records (dropping any orphan tail bytes from a crash).
    pub fn open(
        path: &Path,
        metric: Metric,
        dimension: usize,
        record_count: u64,
    ) -> Result<Self, StorageError> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        validate_header(&mut file, metric, dimension)?;
        let mut segment = Self {
            file,
            dimension,
            mmap: None,
            mapped_records: 0,
        };
        let valid_len = HEADER_LEN as u64 + record_count * segment.stride();
        segment.file.set_len(valid_len)?;
        segment.file.sync_all()?;
        segment.remap(record_count)?;
        Ok(segment)
    }

    /// Appends `count` records (a flat slice of `count * dimension` floats) and
    /// fsyncs. Does not remap; visibility through the map updates on `remap`.
    pub fn append(&mut self, values: &[f32]) -> Result<(), StorageError> {
        if values.len() % self.dimension != 0 {
            return Err(StorageError::Corruption(format!(
                "append length {} is not a multiple of dimension {}",
                values.len(),
                self.dimension
            )));
        }
        self.file.seek(SeekFrom::End(0))?;
        self.file
            .write_all(bytemuck::cast_slice::<f32, u8>(values))?;
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Rebuilds the read-only memory map so `record_count` records are visible.
    ///
    /// # Safety
    ///
    /// `Mmap::map` is unsafe because concurrent external mutation of the file
    /// would be undefined behavior. `EidosDB` owns the file exclusively (single
    /// instance, see crate docs and the concurrency non-goal), writes only by
    /// append + fsync through `self.file`, and never shrinks the live region
    /// except in `open`/compaction when no map is held. The map is read-only.
    pub fn remap(&mut self, record_count: u64) -> Result<(), StorageError> {
        self.mmap = None;
        let needed = HEADER_LEN as u64 + record_count * self.stride();
        let actual = self.file.metadata()?.len();
        if actual < needed {
            return Err(StorageError::Corruption(format!(
                "segment file len {actual} < required {needed}"
            )));
        }
        let map = unsafe { Mmap::map(&self.file)? };
        self.mmap = Some(map);
        self.mapped_records = record_count;
        Ok(())
    }

    /// Drops the map and truncates the file back to an empty, header-only segment.
    pub fn truncate_to_empty(&mut self) -> Result<(), StorageError> {
        self.mmap = None;
        self.mapped_records = 0;
        self.file.set_len(HEADER_LEN as u64)?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Returns the record at `slot` as a zero-copy `&[f32]`, if mapped.
    #[must_use]
    pub fn record(&self, slot: u64) -> Option<&[f32]> {
        if slot >= self.mapped_records {
            return None;
        }
        let map = self.mmap.as_ref()?;
        let start = HEADER_LEN + usize::try_from(slot * self.stride()).ok()?;
        let end = start + usize::try_from(self.stride()).ok()?;
        let bytes = map.get(start..end)?;
        Some(bytemuck::cast_slice::<u8, f32>(bytes))
    }
}

fn write_header(file: &mut File, metric: Metric, dimension: usize) -> Result<(), StorageError> {
    let dimension = u32::try_from(dimension)
        .map_err(|_| StorageError::FormatMismatch("dimension exceeds u32".to_string()))?;
    let mut header = [0u8; HEADER_LEN];
    header[0..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&dimension.to_le_bytes());
    header[16] = metric_to_u8(metric);
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)?;
    file.flush()?;
    Ok(())
}

fn validate_header(file: &mut File, metric: Metric, dimension: usize) -> Result<(), StorageError> {
    let mut header = [0u8; HEADER_LEN];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut header)?;
    if &header[0..8] != MAGIC {
        return Err(StorageError::Corruption("bad segment magic".to_string()));
    }
    let version = u32::from_le_bytes(
        header[8..12]
            .try_into()
            .map_err(|_| StorageError::Corruption("header".to_string()))?,
    );
    if version != FORMAT_VERSION {
        return Err(StorageError::FormatMismatch(format!(
            "segment version {version} != {FORMAT_VERSION}"
        )));
    }
    let dim = u32::from_le_bytes(
        header[12..16]
            .try_into()
            .map_err(|_| StorageError::Corruption("header".to_string()))?,
    );
    let expected = u32::try_from(dimension)
        .map_err(|_| StorageError::FormatMismatch("dimension exceeds u32".to_string()))?;
    if dim != expected {
        return Err(StorageError::FormatMismatch(format!(
            "segment dimension {dim} != {expected}"
        )));
    }
    if metric_from_u8(header[16])? != metric {
        return Err(StorageError::FormatMismatch(
            "segment metric mismatch".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Segment;
    use eidosdb_core::Metric;
    use tempfile::tempdir;

    #[test]
    fn new_segment_has_no_records() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("vectors.seg");
        let segment = Segment::create(&path, Metric::Cosine, 3).expect("create");
        assert_eq!(segment.mapped_records(), 0);
        assert_eq!(segment.record(0), None);
    }

    #[test]
    fn append_then_remap_exposes_records() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("vectors.seg");
        let mut segment = Segment::create(&path, Metric::Cosine, 2).expect("create");
        segment
            .append(&[1.0, 2.0, 3.0, 4.0])
            .expect("append two records");
        assert_eq!(segment.record(0), None, "not visible before remap");
        segment.remap(2).expect("remap");
        assert_eq!(segment.record(0), Some(&[1.0, 2.0][..]));
        assert_eq!(segment.record(1), Some(&[3.0, 4.0][..]));
    }

    #[test]
    fn reopen_reads_persisted_records() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("vectors.seg");
        {
            let mut segment = Segment::create(&path, Metric::DotProduct, 2).expect("create");
            segment.append(&[5.0, 6.0]).expect("append");
        }
        let segment = Segment::open(&path, Metric::DotProduct, 2, 1).expect("open");
        assert_eq!(segment.record(0), Some(&[5.0, 6.0][..]));
    }

    #[test]
    fn open_truncates_orphan_bytes() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("vectors.seg");
        {
            let mut segment = Segment::create(&path, Metric::Cosine, 2).expect("create");
            // Two records on disk, but the watermark will say only one is valid.
            segment.append(&[1.0, 1.0, 9.0, 9.0]).expect("append");
        }
        let segment = Segment::open(&path, Metric::Cosine, 2, 1).expect("open at watermark 1");
        assert_eq!(segment.mapped_records(), 1);
        assert_eq!(segment.record(1), None, "orphan record dropped");
    }

    #[test]
    fn open_rejects_dimension_mismatch() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("vectors.seg");
        Segment::create(&path, Metric::Cosine, 2).expect("create");
        assert!(Segment::open(&path, Metric::Cosine, 4, 0).is_err());
    }
}
