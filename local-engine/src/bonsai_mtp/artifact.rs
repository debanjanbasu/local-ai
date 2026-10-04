//! Shipping the int8 head, so an install can run speculation from an artifact
//! instead of an 849 MB BF16 source.
//!
//! `local-ai bonsai --export-mtp-head <dir>` writes `mtp-head-int8-v2.bin` beside
//! nothing else: no GPU, no 5.9 GB target checkpoint, and no second copy of the
//! transform. The same file can hold its sections verbatim or behind one zstd
//! frame; which one it holds is in the file's own magic, so the loader reads
//! either without being told.

use std::path::{Path, PathBuf};

use super::cache::{Codec, HeadFile, MTP_HEAD_ARTIFACT, write_artifact, write_zstd_artifact};
use super::weights::{spec, transform_sections};

/// What one exported int8 head contains.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MtpHeadArtifact {
    /// The artifact that was written.
    pub path: PathBuf,
    /// The BF16 safetensors it was quantized from.
    pub source: PathBuf,
    /// Sections in artifact order; the header names every one of them.
    pub sections: usize,
    /// Section bytes with the inter-section padding skipped: what a load hashes
    /// and uploads, and the artifact's content identity.
    pub payload_bytes: u64,
    /// Bytes on disk, header page and padding included — less than
    /// [`Self::canonical_bytes`] for a compressed export.
    pub file_bytes: u64,
    /// Bytes the same head occupies with its sections stored verbatim, which is
    /// what [`Self::file_bytes`] means for the stored export.
    pub canonical_bytes: u64,
    /// How the file encodes its sections: `stored` or `zstd`. Recorded here for
    /// the reader of this report; the file itself says it in its magic.
    pub compression: &'static str,
    pub payload_sha256: String,
}

impl MtpHeadArtifact {
    fn exported(
        source: &Path,
        sections: usize,
        codec: Codec,
        head: HeadFile,
    ) -> crate::Result<Self> {
        Ok(Self {
            path: head.path,
            source: source.to_owned(),
            sections,
            payload_bytes: widen(head.payload_bytes, "MTP head payload exceeds 64 bits")?,
            file_bytes: widen(head.file_bytes, "MTP head file exceeds 64 bits")?,
            canonical_bytes: widen(head.canonical_bytes, "MTP head image exceeds 64 bits")?,
            compression: codec.name(),
            payload_sha256: head.payload_sha256,
        })
    }
}

fn widen(bytes: usize, message: &'static str) -> crate::Result<u64> {
    u64::try_from(bytes).map_err(|_| crate::Error::InvalidFormat(message.into()))
}

/// Quantize the BF16 head at `source` into `<directory>/mtp-head-int8-v2.bin`,
/// with its sections stored verbatim.
///
/// The transform is [`transform_sections`], the one `Weights::load` runs on a
/// cache miss, so the artifact is byte-identical to the payload the loader would
/// have built for itself. Every later load re-derives the digest from the file and
/// compares it with the header, which is what lets an install ship this file and
/// trust it.
///
/// This is the default because a load of the stored form costs no anonymous RAM:
/// the file is mapped and its sections reach Metal as demand-paged pointers.
pub fn export_head(source: &Path, directory: &Path) -> crate::Result<MtpHeadArtifact> {
    let spec = spec();
    let sections = transform_sections(source)?;
    let count = sections.len();
    let head = write_artifact(&directory.join(MTP_HEAD_ARTIFACT), &spec, &sections)?;
    MtpHeadArtifact::exported(source, count, Codec::Stored, head)
}

/// Quantize the BF16 head at `source` into `<directory>/mtp-head-int8-v2.bin`,
/// with its section region behind one zstd frame.
///
/// Same transform, same spec, same section names and offsets, and the same
/// `payload_sha256`: inflating the frame rebuilds the file [`export_head`] writes,
/// byte for byte. The frame is decoded into the image before any of it is
/// validated, so the two forms are held to one standard rather than trusted
/// separately.
///
/// **The trade, stated plainly.** The stored head is mapped, and
/// `from_bytes_no_copy` hands Metal pointers into pages the kernel demand-pages and
/// reclaims — so a load spends no anonymous RAM on it. An inflated head has to
/// exist in full before a single section can be read, so every load through one
/// spends ~425 MB of anonymous RAM that stays resident for the life of the process
/// and cannot be paged out. Measured on the shipped head, reading it that way peaks
/// at 752 MiB resident against 412 MiB for the mapped file, and the difference is
/// the inflated image plus the frame's page cache. In exchange the file is 69,425,452
/// bytes smaller — 425,263,104 -> 355,837,652 on disk, 16.3% — and the transform is
/// not needed at load time either way. Seventy megabytes of disk is not worth 425 MB
/// of resident memory on every speculative load, so this stays opt-in: it writes to
/// the same installed path, overwriting whatever was there, and the loader needs
/// nothing to be told about it.
pub fn export_head_zstd(source: &Path, directory: &Path) -> crate::Result<MtpHeadArtifact> {
    let spec = spec();
    let sections = transform_sections(source)?;
    let count = sections.len();
    let head = write_zstd_artifact(&directory.join(MTP_HEAD_ARTIFACT), &spec, &sections)?;
    MtpHeadArtifact::exported(source, count, Codec::Zstd, head)
}
