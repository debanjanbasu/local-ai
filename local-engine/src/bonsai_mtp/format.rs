//! The MTP head file: a header page naming every section, then the sections
//! themselves, each on its own page so they map straight into Metal buffers.

use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};

use super::Section;

/// Derives the spec a head file must match from its decoded section names.
pub(super) type SpecBinder<'a> = dyn Fn(&[&str]) -> Result<Vec<Section>, String> + 'a;

/// The width of a head file's magic, and so of the field it is read as.
const MAGIC_BYTES: usize = 8;
/// A head file: sections are its matrices, each packed `PTQ1_0` blocks
/// (`<name>.ptq1`) or int8 rows and their F32 scales (`<name>.int8`,
/// `<name>.row_scales`), then the folded norms. The section names record each
/// matrix's format, so an all-ternary head is byte for byte what it was before
/// int8 matrices existed.
const PTQ1_MAGIC: &[u8; MAGIC_BYTES] = b"MTPT1\0\0\0";
/// Version of the head container: header page, named page-aligned sections,
/// payload digest. Mixed heads use the same container; a build that predates
/// int8 sections rejects one by its section names rather than misreading it.
pub(super) const PTQ1_VERSION: u32 = 1;
pub(super) const HEADER_BYTES: usize = 4096;
pub(super) const PAGE: usize = 16 * 1024;
/// The payload digest is stored as ASCII hex, so this is the header field width.
const DIGEST_HEX: usize = 64;

/// The distributable head, installed in `models/bonsai2-27b-mtp/`.
///
/// Written by `local-ai bonsai --export mtp-head=DIR` from
/// `DIR/model_mtp_ternary.safetensors`. Its matrices are `PTQ1_0` blocks the
/// target's own kernels multiply, or per-row int8 for the matrices the trainer
/// kept at higher precision. The version is in the name, so a future
/// format is a different file rather than a misparse of this one.
pub const MTP_HEAD_ARTIFACT: &str = "mtp-head-ptq1-v1.bin";

/// A validated head file: its mapping, and where each section starts and how
/// long it is.
///
/// The sections stay in the file's own mapping, so loading costs no anonymous
/// RAM: `from_bytes_no_copy` hands Metal pointers into pages the kernel
/// demand-pages and reclaims.
pub(super) struct HeadMap {
    pub(super) map: Arc<Mmap>,
    pub(super) sections: Vec<(usize, usize)>,
    /// The spec the file was validated against, as its names selected it.
    pub(super) spec: Vec<Section>,
}

/// What one written head file ended up holding.
pub(super) struct HeadFile {
    pub(super) path: PathBuf,
    pub(super) file_bytes: usize,
    pub(super) payload_bytes: usize,
    pub(super) payload_sha256: String,
}

pub(super) fn align_page(value: usize) -> Option<usize> {
    value.checked_add(PAGE - 1).map(|n| n / PAGE * PAGE)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            let _ = write!(output, "{byte:02x}");
            output
        },
    )
}

fn reject(message: impl Into<String>) -> String {
    message.into()
}

fn malformed(message: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

fn widen(error: std::num::TryFromIntError) -> std::io::Error {
    malformed(format!("MTP head layout is wider than 64 bits: {error}"))
}

/// sha256 over the logical section bytes in order, skipping the inter-section
/// padding. Padding is a writer detail, so digesting whole files would make the
/// identity depend on it; the payload is what every consumer binds.
fn digest_of<'a>(regions: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut digest = Sha256::new();
    for region in regions {
        digest.update(region);
    }
    hex(digest.finalize().as_slice())
}

/// Sequential reader over the fixed header page. Every field is bounds-checked,
/// so a short or foreign file is rejected rather than misread.
struct HeaderReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> HeaderReader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let field = self
            .bytes
            .get(self.at..self.at + len)
            .ok_or_else(|| reject(format!("header is truncated at byte {}", self.at)))?;
        self.at += len;
        Ok(field)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| reject("header field is truncated"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| reject("header field is truncated"))?;
        Ok(u64::from_le_bytes(bytes))
    }
}

struct HeaderWriter {
    bytes: Vec<u8>,
}

impl HeaderWriter {
    fn raw(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }
}

/// Everything a header page claims, checked against `spec` but not yet against
/// the bytes it points at.
struct HeadHeader {
    sections: Vec<(usize, usize)>,
    spec: Vec<Section>,
    stored: [u8; DIGEST_HEX],
    file_bytes: u64,
    image_bytes: usize,
}

/// [`read_head_with`] against one fixed spec, whatever the file's names say.
#[cfg(test)]
pub(super) fn read_head(path: &Path, spec: &[Section]) -> Result<HeadMap, String> {
    read_head_with(path, &|_| Ok(spec.to_vec()))
}

/// Read and fully validate a head file: framing, the section names and sizes
/// against the spec `bind` derives from the file's own names (a mixed head
/// names each matrix's format, so its names select which sections the rest of
/// the file must hold), page alignment, section bounds, and the payload digest.
///
/// Every failure is reported rather than swallowed, so the loader can turn the
/// reason into a hard error: a broken install should not look like a healthy
/// one decoding without speculation.
pub(super) fn read_head_with(path: &Path, bind: &SpecBinder<'_>) -> Result<HeadMap, String> {
    let open = |error: std::io::Error| reject(format!("{}: {error}", path.display()));
    let file = File::open(path).map_err(open)?;
    // SAFETY: immutable mapping retained by every no-copy Metal buffer.
    #[allow(unsafe_code)]
    let map = unsafe { MmapOptions::new().map(&file) }.map_err(open)?;
    let header = map.get(..HEADER_BYTES).ok_or_else(|| {
        reject(format!(
            "{} is shorter than one {HEADER_BYTES}-byte header page",
            path.display()
        ))
    })?;
    let parsed = parse_header(header, path, bind)?;
    let map_len =
        u64::try_from(map.len()).map_err(|_| reject(format!("{} is enormous", path.display())))?;
    if parsed.file_bytes != map_len {
        return Err(reject(format!(
            "{}: header records {} bytes, file is {map_len}",
            path.display(),
            parsed.file_bytes
        )));
    }
    // The last padded section ends the file, which rules out appended bytes.
    if map.len() != parsed.image_bytes {
        return Err(reject(format!(
            "{}: sections end at byte {} of a {} byte file",
            path.display(),
            parsed.image_bytes,
            map.len()
        )));
    }
    let payload_sha256 = digest_of(
        parsed
            .sections
            .iter()
            .map(|&(offset, length)| &map[offset..offset + length]),
    );
    if payload_sha256.as_bytes() != parsed.stored.as_slice() {
        return Err(reject(format!(
            "{}: payload hashes to {payload_sha256}, header records {}",
            path.display(),
            String::from_utf8_lossy(&parsed.stored)
        )));
    }
    Ok(HeadMap {
        map: Arc::new(map),
        sections: parsed.sections,
        spec: parsed.spec,
    })
}

/// Check a header page against `spec` and return what it claims. No section
/// byte is read here, so a file whose framing is wrong is rejected before
/// anything is hashed.
#[allow(clippy::too_many_lines)]
fn parse_header(header: &[u8], path: &Path, bind: &SpecBinder<'_>) -> Result<HeadHeader, String> {
    let mut cursor = HeaderReader {
        bytes: header,
        at: 0,
    };
    if cursor.take(MAGIC_BYTES)? != PTQ1_MAGIC {
        return Err(reject(format!(
            "{} is not an MTP head artifact",
            path.display()
        )));
    }
    let version = cursor.u32()?;
    if version != PTQ1_VERSION {
        return Err(reject(format!(
            "{} is MTP head version {version}, this build reads version {PTQ1_VERSION}",
            path.display(),
        )));
    }
    let count = cursor.u32()? as usize;
    let payload_len = cursor.u64()?;
    let file_bytes = cursor.u64()?;
    let stored = cursor
        .take(DIGEST_HEX)?
        .try_into()
        .map_err(|_| reject("header digest is truncated"))?;
    let names_len = cursor.u32()? as usize;
    let names = cursor.take(names_len)?;
    // Every name takes at least one byte of the table, which bounds `count`.
    let mut decoded = Vec::with_capacity(count.min(names_len));
    let mut at = 0;
    for _ in 0..count {
        let truncated = || {
            reject(format!(
                "{}: section name table is truncated",
                path.display()
            ))
        };
        let len = usize::from(*names.get(at).ok_or_else(truncated)?);
        let name = names.get(at + 1..at + 1 + len).ok_or_else(truncated)?;
        decoded.push(
            std::str::from_utf8(name)
                .map_err(|_| reject(format!("{}: section name is not UTF-8", path.display())))?,
        );
        at += 1 + len;
    }
    if at != names_len {
        return Err(reject(format!(
            "{}: section name table has {} trailing bytes",
            path.display(),
            names_len - at
        )));
    }
    let spec = bind(&decoded).map_err(|reason| reject(format!("{}: {reason}", path.display())))?;
    if count != spec.len() {
        return Err(reject(format!(
            "{} holds {count} sections, this build binds {}",
            path.display(),
            spec.len()
        )));
    }
    // Section lengths alone cannot bind these sections: `k_proj` and `v_proj`
    // are both `[KV, WIDTH]`, so only the names say which matrix a section is.
    for (index, (found, (expected, _))) in decoded.iter().zip(&spec).enumerate() {
        if found != expected {
            return Err(reject(format!(
                "{}: section {index} is {found}, this build binds {expected}",
                path.display(),
            )));
        }
    }
    // The table is read whole before any of it is checked, because where the
    // image ends is the last entry's own padded end and every other section has
    // to be judged against that rather than against the file.
    let mut sections = Vec::with_capacity(count);
    for _ in 0..count {
        let offset = cursor.u64()? as usize;
        let length = cursor.u64()? as usize;
        sections.push((offset, length));
    }
    let image_bytes = image_end(&sections, path)?;
    let mut summed = 0_u64;
    for (index, (&(offset, length), &(_, expected))) in sections.iter().zip(&spec).enumerate() {
        let padded = align_page(length).ok_or_else(|| {
            reject(format!(
                "{}: section {index} size overflows",
                path.display()
            ))
        })?;
        if length != expected {
            return Err(reject(format!(
                "{}: section {index} is {length} bytes, this build binds {expected}",
                path.display()
            )));
        }
        if !offset.is_multiple_of(PAGE) {
            return Err(reject(format!(
                "{}: section {index} offset {offset} is not {PAGE}-byte aligned",
                path.display()
            )));
        }
        if offset < HEADER_BYTES {
            return Err(reject(format!(
                "{}: section {index} offset {offset} runs into the header page",
                path.display()
            )));
        }
        if offset
            .checked_add(padded)
            .is_none_or(|end| end > image_bytes)
        {
            return Err(reject(format!(
                "{}: section {index} runs past the end of the file",
                path.display()
            )));
        }
        summed = summed
            .checked_add(u64::try_from(length).map_err(|_| reject("size overflow"))?)
            .ok_or_else(|| reject(format!("{}: payload size overflows", path.display())))?;
    }
    if summed != payload_len {
        return Err(reject(format!(
            "{}: sections total {summed} bytes, header records {payload_len}",
            path.display()
        )));
    }
    Ok(HeadHeader {
        sections,
        spec,
        stored,
        file_bytes,
        image_bytes,
    })
}

/// Where the file a head's offsets describe ends: the last section's padded end.
fn image_end(sections: &[(usize, usize)], path: &Path) -> Result<usize, String> {
    sections.last().map_or(Ok(0), |&(offset, length)| {
        let padded = align_page(length).ok_or_else(|| {
            reject(format!(
                "{}: the last section size overflows",
                path.display()
            ))
        })?;
        offset.checked_add(padded).ok_or_else(|| {
            reject(format!(
                "{}: the last section offset overflows",
                path.display()
            ))
        })
    })
}

/// Where each section lands, and where the file then ends. The first section
/// starts one page in, so the header page and the alignment padding never
/// overlap one, and every later section starts on a page boundary: identical
/// payloads therefore produce identical files.
fn layout(sections: &[Vec<u8>]) -> Result<(Vec<(usize, usize)>, usize), String> {
    let mut offsets = Vec::with_capacity(sections.len());
    let mut end = PAGE;
    for section in sections {
        let padded =
            align_page(section.len()).ok_or_else(|| reject("MTP head section size overflows"))?;
        offsets.push((end, section.len()));
        end = end
            .checked_add(padded)
            .ok_or_else(|| reject("MTP head file size overflows"))?;
    }
    Ok((offsets, end))
}

fn encode_header(
    offsets: &[(usize, usize)],
    spec: &[Section],
    digest: &str,
    payload_bytes: usize,
    file_bytes: usize,
) -> Result<Vec<u8>, String> {
    let stored = digest.as_bytes();
    if stored.len() != DIGEST_HEX || !stored.iter().all(u8::is_ascii_hexdigit) {
        return Err(reject("payload digest is not 64 hex-encoded bytes"));
    }
    let mut out = HeaderWriter { bytes: Vec::new() };
    out.raw(PTQ1_MAGIC);
    out.u32(PTQ1_VERSION);
    out.u32(u32::try_from(spec.len()).map_err(|_| reject("too many sections"))?);
    out.u64(u64::try_from(payload_bytes).map_err(|_| reject("payload length overflows"))?);
    out.u64(u64::try_from(file_bytes).map_err(|_| reject("file length overflows"))?);
    out.raw(stored);
    let names_len = spec
        .iter()
        .try_fold(0_usize, |total, (name, _)| {
            total.checked_add(1 + name.len())
        })
        .ok_or_else(|| reject("section name table length overflows"))?;
    out.u32(u32::try_from(names_len).map_err(|_| reject("name table length overflows"))?);
    for (name, _) in spec {
        let len = u8::try_from(name.len()).map_err(|_| reject("section name is too long"))?;
        out.raw(&[len]);
        out.raw(name.as_bytes());
    }
    for &(offset, length) in offsets {
        out.u64(u64::try_from(offset).map_err(|_| reject("offset overflows"))?);
        out.u64(u64::try_from(length).map_err(|_| reject("length overflows"))?);
    }
    if out.bytes.len() > HEADER_BYTES {
        return Err(reject(format!(
            "MTP head header needs {} bytes, a page holds {HEADER_BYTES}",
            out.bytes.len()
        )));
    }
    out.bytes.resize(HEADER_BYTES, 0);
    Ok(out.bytes)
}

/// Write one head artifact at `path` through a temporary file, so a reader
/// either sees a whole head or none of it. The header records the payload
/// digest, which is what lets the file validate itself on every load.
pub(super) fn write_artifact(
    path: &Path,
    spec: &[Section],
    sections: &[Vec<u8>],
) -> std::io::Result<HeadFile> {
    if spec.len() != sections.len() {
        return Err(malformed(reject(format!(
            "spec binds {} sections, {} were built",
            spec.len(),
            sections.len()
        ))));
    }
    for (index, (section, &(_, expected))) in sections.iter().zip(spec).enumerate() {
        if section.len() != expected {
            return Err(malformed(reject(format!(
                "section {index} is {} bytes, spec binds {expected}",
                section.len()
            ))));
        }
    }
    let digest = digest_of(sections.iter().map(Vec::as_slice));
    let (offsets, image_bytes) = layout(sections).map_err(malformed)?;
    let payload_bytes = sections
        .iter()
        .try_fold(0_usize, |total, section| total.checked_add(section.len()))
        .ok_or_else(|| malformed(reject("payload length overflows")))?;
    let header =
        encode_header(&offsets, spec, &digest, payload_bytes, image_bytes).map_err(malformed)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // A crashed writer can leave a temporary behind; the process id in the name
    // makes clearing it safe.
    let temp = path.with_file_name(format!(
        ".mtp-head-{}-{}.tmp",
        std::process::id(),
        digest.get(..16).unwrap_or(&digest)
    ));
    let _ = std::fs::remove_file(&temp);
    let written = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.set_len(u64::try_from(image_bytes).map_err(widen)?)?;
        file.write_all(&header)?;
        for (&(offset, _), section) in offsets.iter().zip(sections) {
            file.seek(SeekFrom::Start(u64::try_from(offset).map_err(widen)?))?;
            file.write_all(section)?;
        }
        file.sync_all()?;
        std::fs::rename(&temp, path)?;
        Ok(())
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    written?;
    Ok(HeadFile {
        path: path.to_owned(),
        file_bytes: image_bytes,
        payload_bytes,
        payload_sha256: digest,
    })
}
