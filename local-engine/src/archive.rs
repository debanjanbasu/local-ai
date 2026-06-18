//! Compressed model archive format (`.lma`).
//!
//! Stores model weights in per-component independently-compressed frames with
//! BF16→FP16 pre-conversion, byte shuffling, and zstd compression for optimal
//! storage density and fast on-the-fly decompression directly into Metal buffers.
//!
//! # File layout (footer-indexed)
//!
//! ```text
//! [frame 0 data][frame 1 data]...[frame N data]
//! [frame index: N × 32 bytes]
//! [footer: 20 bytes]
//! ```
//!
//! Each frame is an independent zstd stream containing byte-shuffled FP16 data.
//! The footer and index are at the end (like ZIP), so frames can be written
//! sequentially without knowing sizes in advance.

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    FP16 = 0,
    Tq2_0 = 1,
}

impl QuantType {
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::FP16),
            1 => Some(Self::Tq2_0),
            _ => None,
        }
    }

    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::FP16 => 512,
            Self::Tq2_0 => 8,
        }
    }

    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "FP16" | "fp16" => Some(Self::FP16),
            "TQ2_0" | "tq2_0" => Some(Self::Tq2_0),
            _ => None,
        }
    }
}

mod manifest;
pub mod output;
pub use manifest::{ArchiveManifest, CalibrationProvenance, QuantRecipeStamp};
pub use output::{ArchiveOutputPlan, OutputPolicy, ResumeManifest};

/// Magic bytes identifying a `.lma` file.
const MAGIC: [u8; 4] = *b"LMAR";

/// Current archive format version.
///
/// v3: `TQ2_0` quantized weights for Gemma4 QAT.
const VERSION: u32 = 3;

/// Size of each frame index entry in bytes.
const FRAME_ENTRY_BYTES: usize = 32;

/// Size of the file footer in bytes.
const FOOTER_BYTES: usize = 20;

// ── Frame Kind ──────────────────────────────────────────────────────────────

/// Pre-compression data transformation stored in frame entry padding bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShuffleMode {
    /// No transformation — raw zstd.
    None = 0,
    /// 2-byte element shuffle (high/low byte grouping). Best for FP16 data.
    ByteShuffle2 = 1,
}

impl ShuffleMode {
    /// Construct from u8 discriminant.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::ByteShuffle2),
            _ => None,
        }
    }
}

/// Identifies what a frame contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameKind {
    /// Safetensors JSON header (tensor names, shapes, dtypes — metadata only).
    Metadata = 0,
    /// Embedding table (FP16, byte-shuffled).
    Embedding = 1,
    /// Pre-packed layer weights in pool-buffer order (FP16, byte-shuffled).
    Layer = 2,
    /// Final RMS norm weight (FP16, byte-shuffled).
    FinalNorm = 4,
    /// PLE embedding table: `embed_tokens_per_layer.weight` (FP16, byte-shuffled).
    /// Shape: `[vocab_size, num_layers * ple_dim]`.
    PleEmbedding = 6,
    /// PLE global tensors concatenated (FP16, byte-shuffled):
    /// `per_layer_model_projection.weight` `[num_layers*ple_dim, hidden_size]`
    /// followed by `per_layer_projection_norm.weight` `[ple_dim]`.
    PleGlobal = 7,
    /// Gemma4 assistant transformer layer weights (packed like backbone Layer).
    /// id = `layer_idx` (`0..num_hidden_layers`).
    GemmaMtpLayer = 9,
    /// Gemma4 assistant pre/post projection weights.
    /// id = 0 (`pre_projection`) or 1 (`post_projection`).
    GemmaMtpProjection = 10,
    /// Gemma4 assistant embedding table.
    /// id = 0.
    GemmaMtpEmbedding = 11,
    /// Gemma4 assistant final `RMSNorm` weight.
    /// id = 0.
    GemmaMtpNorm = 12,
}

impl FrameKind {
    const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Metadata),
            1 => Some(Self::Embedding),
            2 => Some(Self::Layer),
            4 => Some(Self::FinalNorm),
            6 => Some(Self::PleEmbedding),
            7 => Some(Self::PleGlobal),
            8 => Some(Self::GemmaMtpLayer),
            10 => Some(Self::GemmaMtpProjection),
            11 => Some(Self::GemmaMtpEmbedding),
            12 => Some(Self::GemmaMtpNorm),
            _ => None,
        }
    }
}

// ── Frame Index Entry ───────────────────────────────────────────────────────

/// Describes one frame in the archive.
#[derive(Debug, Clone)]
pub struct FrameEntry {
    pub kind: FrameKind,
    /// Pre-compression data transformation applied to this frame.
    pub shuffle_mode: ShuffleMode,
    /// `QuantType` discriminant.
    pub quant_type_raw: u8,
    /// Semantic id: layer index for `Layer`, 0 for `Embedding`/`FinalNorm`.
    pub id: u32,
    /// Absolute byte offset in the archive file.
    pub file_offset: u64,
    /// Size of the compressed (on-disk) data.
    pub compressed_size: u64,
    /// Size after decompression (before byte-unshuffle — same size).
    pub uncompressed_size: u64,
}

impl FrameEntry {
    fn to_bytes(&self) -> [u8; FRAME_ENTRY_BYTES] {
        let mut buf = [0u8; FRAME_ENTRY_BYTES];
        buf[0] = self.kind as u8;
        buf[1] = self.shuffle_mode as u8;
        buf[2] = self.quant_type_raw;
        // byte 3 reserved (0)
        buf[4..8].copy_from_slice(&self.id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.file_offset.to_le_bytes());
        buf[16..24].copy_from_slice(&self.compressed_size.to_le_bytes());
        buf[24..32].copy_from_slice(&self.uncompressed_size.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8; FRAME_ENTRY_BYTES]) -> crate::Result<Self> {
        let kind = FrameKind::from_u8(buf[0]).ok_or_else(|| {
            crate::Error::InvalidFormat(format!("unknown frame kind: {}", buf[0]))
        })?;
        let shuffle_mode = ShuffleMode::from_u8(buf[1]).unwrap_or(ShuffleMode::None);
        let quant_type_raw = buf[2];
        let id = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let file_offset = u64::from_le_bytes(buf[8..16].try_into().unwrap_or([0; 8]));
        let compressed_size = u64::from_le_bytes(buf[16..24].try_into().unwrap_or([0; 8]));
        let uncompressed_size = u64::from_le_bytes(buf[24..32].try_into().unwrap_or([0; 8]));
        Ok(Self {
            kind,
            shuffle_mode,
            quant_type_raw,
            id,
            file_offset,
            compressed_size,
            uncompressed_size,
        })
    }
}

// ── Byte Shuffling ──────────────────────────────────────────────────────────

/// Byte-shuffle for 2-byte elements (BF16/FP16).
///
/// Groups all high bytes (sign + exponent) together, then all low bytes
/// (mantissa). This dramatically improves zstd compression because exponent
/// bytes are highly repetitive within a tensor.
#[must_use]
#[allow(unsafe_code)]
pub fn byte_shuffle_2(data: &[u8]) -> Vec<u8> {
    let n = data.len() / 2;
    let mut out = vec![0u8; data.len()];
    let hi_base = 0;
    let lo_base = n;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{vld1q_u8, vst1q_u8, vuzp1q_u8, vuzp2q_u8};

        let chunks = n / 16;
        let src_ptr = data.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for c in 0..chunks {
            unsafe {
                let s = src_ptr.add(c * 32);
                let a = vld1q_u8(s);
                let b = vld1q_u8(s.add(16));
                // Even-indexed bytes = lo bytes (indices 0,2,4,...), odd = hi bytes (1,3,5,...)
                let lo = vuzp1q_u8(a, b);
                let hi = vuzp2q_u8(a, b);
                vst1q_u8(out_ptr.add(hi_base + c * 16), hi);
                vst1q_u8(out_ptr.add(lo_base + c * 16), lo);
            }
        }

        // Scalar remainder
        for i in (chunks * 16)..n {
            out[hi_base + i] = data[i * 2 + 1];
            out[lo_base + i] = data[i * 2];
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            out[hi_base + i] = data[i * 2 + 1];
            out[lo_base + i] = data[i * 2];
        }
    }

    // Trailing odd byte
    if !data.len().is_multiple_of(2) {
        // Already allocated via vec![0u8; data.len()], last byte is at index 2*n
        out[2 * n] = data[data.len() - 1];
    }

    out
}

/// Byte-unshuffle 2-byte elements from `src` into `dest`.
///
/// Inverse of [`byte_shuffle_2`]. `dest` must be at least `src.len()` bytes.
pub fn byte_unshuffle_2_into(src: &[u8], dest: &mut [u8]) {
    let n = src.len() / 2;
    for i in 0..n {
        dest[i * 2 + 1] = src[i]; // high byte
        dest[i * 2] = src[n + i]; // low byte
    }
    if !src.len().is_multiple_of(2) {
        dest[src.len() - 1] = src[src.len() - 1];
    }
}

// ── Archive Writer ──────────────────────────────────────────────────────────

/// Build a zstd compressor with the frame-checksum flag enabled (xxhash64,
/// auto-verified by every decompression — silent-corruption detection for
/// archives that outlive their source files).
fn new_checksummed_compressor(zstd_level: i32) -> crate::Result<zstd::bulk::Compressor<'static>> {
    let mut compressor = zstd::bulk::Compressor::new(zstd_level)
        .map_err(|e| crate::Error::Io(format!("zstd compressor init: {e}")))?;
    compressor
        .set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(true))
        .map_err(|e| crate::Error::Io(format!("zstd checksum flag: {e}")))?;
    Ok(compressor)
}

/// Compress one frame payload on the calling thread (checksummed).
///
/// No internal multithreading: frames are independent, so callers parallelize
/// *across* frames — zstd's own job-splitting cannot engage on chunk-sized
/// inputs anyway.
///
/// # Errors
///
/// Returns an error if compression fails.
pub fn compress_frame_payload(data: &[u8], zstd_level: i32) -> crate::Result<Vec<u8>> {
    new_checksummed_compressor(zstd_level)?
        .compress(data)
        .map_err(|e| crate::Error::Io(format!("zstd compress: {e}")))
}

/// Writes a `.lma` compressed archive.
///
/// Frames are written sequentially. Call [`finish`](ArchiveWriter::finish) to
/// write the frame index and footer.
///
/// Generic over the underlying writer — use [`ArchiveWriter::new`] for
/// file-based archives (with multithreaded zstd) or [`ArchiveWriter::from_writer`]
/// for any `Write + Seek` destination.
pub struct ArchiveWriter<W: Write + Seek> {
    file: W,
    entries: Vec<FrameEntry>,
    compressor: zstd::bulk::Compressor<'static>,
    bytes_written: u64,
    /// Archive version to write in the footer. Always [`VERSION`] (3).
    archive_version: u32,
}

impl ArchiveWriter<BufWriter<std::fs::File>> {
    /// Create a new file-backed archive writer.
    ///
    /// Enables zstd multithreaded compression using all available CPU cores.
    ///
    /// # Errors
    /// Returns an error if the output file cannot be created or compressor init fails.
    pub fn new(path: &Path, zstd_level: i32) -> crate::Result<Self> {
        let file = std::fs::File::create(path)
            .map_err(|e| crate::Error::Io(format!("create {}: {e}", path.display())))?;
        Self::from_file(file, zstd_level, Vec::new(), 0)
    }

    /// Resume writing to an existing partial archive using previously written frame entries.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or compressor init fails.
    pub fn resume(path: &Path, zstd_level: i32, entries: Vec<FrameEntry>) -> crate::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)
            .map_err(|e| crate::Error::Io(format!("open {}: {e}", path.display())))?;
        let bytes_written = file
            .metadata()
            .map_err(|e| crate::Error::Io(format!("metadata {}: {e}", path.display())))?
            .len();
        Self::from_file(file, zstd_level, entries, bytes_written)
    }

    fn from_file(
        file: std::fs::File,
        zstd_level: i32,
        entries: Vec<FrameEntry>,
        bytes_written: u64,
    ) -> crate::Result<Self> {
        let n_workers =
            std::thread::available_parallelism().map_or(4, std::num::NonZero::get) as u32;
        let mut compressor = new_checksummed_compressor(zstd_level)?;
        compressor
            .multithread(n_workers)
            .map_err(|e| crate::Error::Io(format!("zstd multithread: {e}")))?;
        Ok(Self {
            file: BufWriter::with_capacity(4 * 1024 * 1024, file),
            entries,
            compressor,
            bytes_written,
            archive_version: VERSION,
        })
    }
}

impl<W: Write + Seek> ArchiveWriter<W> {
    /// Create an archive writer from any `Write + Seek` destination.
    ///
    /// Does **not** enable multithreaded zstd (use [`Self::new`] for file-based
    /// archives that benefit from parallel compression).
    ///
    /// # Errors
    /// Returns an error if the zstd compressor cannot be initialised.
    pub fn from_writer(writer: W, zstd_level: i32) -> crate::Result<Self> {
        Ok(Self {
            file: writer,
            entries: Vec::new(),
            compressor: new_checksummed_compressor(zstd_level)?,
            bytes_written: 0,
            archive_version: VERSION,
        })
    }

    /// Write a frame: compress raw data with zstd, write to file.
    ///
    /// # Errors
    /// Returns an error on compression or I/O failure.
    pub fn write_frame(&mut self, kind: FrameKind, id: u32, data: &[u8]) -> crate::Result<()> {
        self.write_frame_shuffled(kind, id, data, ShuffleMode::None, 0)
    }

    /// Write a frame with a pre-compression shuffle transformation.
    ///
    /// The caller must have already applied the shuffle to `data` (byte-shuffle
    /// or IQ deinterleave). This method only records the mode in the frame entry
    /// so that [`CompressedArchive::decompress_into`] can reverse it on read.
    ///
    /// # Errors
    /// Returns an error on compression or I/O failure.
    pub fn write_frame_shuffled(
        &mut self,
        kind: FrameKind,
        id: u32,
        data: &[u8],
        shuffle_mode: ShuffleMode,
        quant_type_raw: u8,
    ) -> crate::Result<()> {
        let compressed = self
            .compressor
            .compress(data)
            .map_err(|e| crate::Error::Io(format!("zstd compress: {e}")))?;

        let entry = FrameEntry {
            kind,
            shuffle_mode,
            quant_type_raw,
            id,
            file_offset: self.bytes_written,
            compressed_size: compressed.len() as u64,
            uncompressed_size: data.len() as u64,
        };

        self.file
            .write_all(&compressed)
            .map_err(|e| crate::Error::Io(format!("write frame: {e}")))?;
        self.bytes_written += compressed.len() as u64;
        self.entries.push(entry);
        Ok(())
    }

    /// Write a frame from raw pre-compressed data (no processing).
    ///
    /// Used when the caller has already done compression.
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    pub fn write_raw_frame(
        &mut self,
        kind: FrameKind,
        id: u32,
        compressed: &[u8],
        uncompressed_size: u64,
    ) -> crate::Result<()> {
        self.write_raw_frame_shuffled(
            kind,
            id,
            compressed,
            uncompressed_size,
            ShuffleMode::None,
            0,
        )
    }

    /// Write a pre-compressed frame with shuffle metadata.
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    pub fn write_raw_frame_shuffled(
        &mut self,
        kind: FrameKind,
        id: u32,
        compressed: &[u8],
        uncompressed_size: u64,
        shuffle_mode: ShuffleMode,
        quant_type_raw: u8,
    ) -> crate::Result<()> {
        let entry = FrameEntry {
            kind,
            shuffle_mode,
            quant_type_raw,
            id,
            file_offset: self.bytes_written,
            compressed_size: compressed.len() as u64,
            uncompressed_size,
        };
        self.file
            .write_all(compressed)
            .map_err(|e| crate::Error::Io(format!("write raw frame: {e}")))?;
        self.bytes_written += compressed.len() as u64;
        self.entries.push(entry);
        Ok(())
    }

    #[must_use]
    pub fn last_entry(&self) -> Option<&FrameEntry> {
        self.entries.last()
    }

    /// Write the frame index and footer, completing the archive.
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    pub fn finish(mut self) -> crate::Result<()> {
        let index_offset = self.bytes_written;

        // Write frame index
        for entry in &self.entries {
            self.file
                .write_all(&entry.to_bytes())
                .map_err(|e| crate::Error::Io(format!("write index: {e}")))?;
        }

        // Write footer
        let num_frames = self.entries.len() as u32;
        let mut footer = [0u8; FOOTER_BYTES];
        footer[0..4].copy_from_slice(&MAGIC);
        footer[4..8].copy_from_slice(&self.archive_version.to_le_bytes());
        footer[8..12].copy_from_slice(&num_frames.to_le_bytes());
        footer[12..20].copy_from_slice(&index_offset.to_le_bytes());
        self.file
            .write_all(&footer)
            .map_err(|e| crate::Error::Io(format!("write footer: {e}")))?;

        self.file
            .flush()
            .map_err(|e| crate::Error::Io(format!("flush: {e}")))?;
        Ok(())
    }
}

// ── Archive Reader ──────────────────────────────────────────────────────────

/// Reader for `.lma` compressed model archives.
///
/// Supports random access at frame granularity: seek to any frame,
/// read compressed bytes, decompress + unshuffle directly into a target buffer.
///
/// Generic over the underlying reader — use [`CompressedArchive::open`] for
/// file-based archives or [`CompressedArchive::from_bytes`] / [`CompressedArchive::open_reader`]
/// for in-memory data.
pub struct CompressedArchive<R: Read + Seek> {
    file: R,
    entries: Vec<FrameEntry>,
    /// Fast lookup: (kind, id) → index into `entries`.
    index: HashMap<(u8, u32), usize>,
    /// Archive format version (3).
    version: u32,
}

impl CompressedArchive<BufReader<std::fs::File>> {
    /// Open an archive and read its frame index from the footer.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read, the magic is invalid,
    /// or the version is unsupported.
    pub fn open(path: &Path) -> crate::Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| crate::Error::Io(format!("{}: {e}", path.display())))?;
        let file_len = file
            .metadata()
            .map_err(|e| crate::Error::Io(e.to_string()))?
            .len();
        if file_len < FOOTER_BYTES as u64 {
            return Err(crate::Error::InvalidFormat(
                "file too small for LMAR footer".into(),
            ));
        }
        let reader = BufReader::with_capacity(4 * 1024 * 1024, file);
        Self::open_reader(reader)
    }
}

impl<'a> CompressedArchive<std::io::Cursor<&'a [u8]>> {
    /// Open an archive from an in-memory byte slice.
    ///
    /// # Errors
    /// Returns an error if the data is too small, the magic is invalid,
    /// or the version is unsupported.
    pub fn from_bytes(data: &'a [u8]) -> crate::Result<Self> {
        if (data.len() as u64) < FOOTER_BYTES as u64 {
            return Err(crate::Error::InvalidFormat(
                "data too small for LMAR footer".into(),
            ));
        }
        let cursor = std::io::Cursor::new(data);
        Self::open_reader(cursor)
    }
}

impl<R: Read + Seek> CompressedArchive<R> {
    /// Open an archive from any `Read + Seek` source.
    ///
    /// Reads the footer and frame index from the end of the stream.
    ///
    /// # Errors
    /// Returns an error if the magic is invalid or the version is unsupported.
    #[allow(clippy::cast_possible_wrap)]
    pub fn open_reader(mut reader: R) -> crate::Result<Self> {
        // Read footer
        reader
            .seek(SeekFrom::End(-(FOOTER_BYTES as i64)))
            .map_err(|e| crate::Error::Io(e.to_string()))?;
        let mut footer = [0u8; FOOTER_BYTES];
        reader
            .read_exact(&mut footer)
            .map_err(|e| crate::Error::Io(e.to_string()))?;

        if footer[0..4] != MAGIC {
            return Err(crate::Error::InvalidFormat("invalid LMAR magic".into()));
        }
        let version = u32::from_le_bytes([footer[4], footer[5], footer[6], footer[7]]);
        if version != 3 {
            return Err(crate::Error::ArchiveVersionMismatch {
                expected: VERSION,
                found: version,
            });
        }
        let num_frames = u32::from_le_bytes([footer[8], footer[9], footer[10], footer[11]]);
        let index_offset = u64::from_le_bytes(footer[12..20].try_into().unwrap_or([0u8; 8]));

        // Read frame index
        reader
            .seek(SeekFrom::Start(index_offset))
            .map_err(|e| crate::Error::Io(e.to_string()))?;

        let mut entries = Vec::with_capacity(num_frames as usize);
        let mut lookup = HashMap::with_capacity(num_frames as usize);
        for i in 0..num_frames {
            let mut buf = [0u8; FRAME_ENTRY_BYTES];
            reader
                .read_exact(&mut buf)
                .map_err(|e| crate::Error::Io(e.to_string()))?;
            let entry = FrameEntry::from_bytes(&buf)?;
            lookup.insert((entry.kind as u8, entry.id), i as usize);
            entries.push(entry);
        }

        Ok(Self {
            file: reader,
            entries,
            index: lookup,
            version,
        })
    }

    /// Look up a frame by kind and id.
    #[must_use]
    pub fn find_frame(&self, kind: FrameKind, id: u32) -> Option<&FrameEntry> {
        self.index
            .get(&(kind as u8, id))
            .map(|&idx| &self.entries[idx])
    }

    /// Archive format version (3).
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Decompress a frame directly into a caller-provided buffer.
    ///
    /// `dest` must be at least `frame.uncompressed_size` bytes.
    /// The data is zstd-decompressed then byte-unshuffled in place.
    ///
    /// # Errors
    /// Returns an error on seek, read, or decompression failure.
    pub fn decompress_into(
        &mut self,
        kind: FrameKind,
        id: u32,
        dest: &mut [u8],
    ) -> crate::Result<usize> {
        let entry = self
            .find_frame(kind, id)
            .ok_or_else(|| {
                crate::Error::InvalidFormat(format!("frame not found: kind={kind:?}, id={id}"))
            })?
            .clone();

        if (dest.len() as u64) < entry.uncompressed_size {
            return Err(crate::Error::InvalidArgument(format!(
                "buffer too small: {} < {}",
                dest.len(),
                entry.uncompressed_size
            )));
        }

        // Seek and read compressed data
        self.file
            .seek(SeekFrom::Start(entry.file_offset))
            .map_err(|e| crate::Error::Io(e.to_string()))?;
        let mut compressed = vec![0u8; entry.compressed_size as usize];
        self.file
            .read_exact(&mut compressed)
            .map_err(|e| crate::Error::Io(e.to_string()))?;

        // Decompress
        let decompressed = zstd::bulk::decompress(&compressed, entry.uncompressed_size as usize)
            .map_err(|e| crate::Error::Io(format!("zstd decompress: {e}")))?;
        drop(compressed);

        // Reverse any pre-compression transformation
        match entry.shuffle_mode {
            ShuffleMode::None => {
                dest[..decompressed.len()].copy_from_slice(&decompressed);
            }
            ShuffleMode::ByteShuffle2 => {
                byte_unshuffle_2_into(&decompressed, &mut dest[..decompressed.len()]);
            }
        }
        Ok(entry.uncompressed_size as usize)
    }

    /// Decompress a frame, returning the unshuffled data as a `Vec<u8>`.
    ///
    /// # Errors
    /// Returns an error on seek, read, or decompression failure.
    pub fn decompress_vec(&mut self, kind: FrameKind, id: u32) -> crate::Result<Vec<u8>> {
        let size = self
            .find_frame(kind, id)
            .ok_or_else(|| {
                crate::Error::InvalidFormat(format!("frame not found: kind={kind:?}, id={id}"))
            })?
            .uncompressed_size as usize;
        let mut buf = vec![0u8; size];
        self.decompress_into(kind, id, &mut buf)?;
        Ok(buf)
    }

    /// All frame entries in the archive.
    #[must_use]
    pub fn entries(&self) -> &[FrameEntry] {
        &self.entries
    }

    /// Read a frame's raw **compressed** bytes (no decompression), returning
    /// them with the frame's uncompressed size. Only valid for unshuffled
    /// frames; the caller decompresses with `zstd::bulk::decompress` — this
    /// enables decompressing many frames in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is missing, shuffled, or unreadable.
    pub fn read_frame_compressed(
        &mut self,
        kind: FrameKind,
        id: u32,
    ) -> crate::Result<(usize, Vec<u8>)> {
        let entry = self
            .find_frame(kind, id)
            .ok_or_else(|| {
                crate::Error::InvalidFormat(format!("frame not found: kind={kind:?}, id={id}"))
            })?
            .clone();
        if entry.shuffle_mode != ShuffleMode::None {
            return Err(crate::Error::InvalidFormat(
                "read_frame_compressed only supports unshuffled frames".into(),
            ));
        }
        self.file
            .seek(SeekFrom::Start(entry.file_offset))
            .map_err(|e| crate::Error::Io(e.to_string()))?;
        let mut compressed = vec![0u8; entry.compressed_size as usize];
        self.file
            .read_exact(&mut compressed)
            .map_err(|e| crate::Error::Io(e.to_string()))?;
        Ok((entry.uncompressed_size as usize, compressed))
    }

    /// Number of frames.
    #[must_use]
    pub const fn num_frames(&self) -> usize {
        self.entries.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────
