use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};

use super::Section;

/// The width of a head file's magic, and so of the field it is read as.
const CACHE_MAGIC_BYTES: usize = 8;
const CACHE_MAGIC: &[u8; CACHE_MAGIC_BYTES] = b"MTPQ8\0\0\0";
const CACHE_MAGIC_ZSTD: &[u8; CACHE_MAGIC_BYTES] = b"MTPZ8\0\0\0";
pub(super) const CACHE_VERSION: u32 = 2;
pub(super) const CACHE_HEADER_BYTES: usize = 4096;
pub(super) const CACHE_PAGE: usize = 16 * 1024;
pub(super) const CACHE_SECTIONS: usize = 25;
/// The payload digest is stored and named as ASCII hex, so this is both the
/// header field width and the length of the name suffix.
const DIGEST_HEX: usize = 64;
const CACHE_EXTENSION: &str = "bin";
/// The zstd level [`write_zstd_artifact`] compresses at.
///
/// 19 is where this payload's ratio turns over, and widening the window does not
/// help. Compressing the shipped head's 425,246,720-byte section region at 19
/// measures 355,837,652 bytes — 69,409,068 saved, 1.195x — while `--long=27` and
/// level 22 both measure worse on the same payload (355,688,546 and 358,063,503).
///
/// Streaming costs a little against compressing the whole region in one call
/// (`zstd -19` on the same bytes measures 355,444,400): the frame is built from
/// writes as the sections come rather than from one buffer, which is what keeps a
/// second 425 MB image from existing at all. The 0.1% buys that.
const ZSTD_LEVEL: i32 = 19;

/// How a head file stores the region between its header page and its end.
///
/// The discriminator is the magic rather than a field beside it, so eight bytes
/// settle both questions a reader has about a head file: is this an int8 artifact
/// at all, and which of the two encodings are its sections in. Nothing outside the
/// file takes part — not its name, not the build — so a compressed head can sit at
/// the one installed artifact path and still be read as itself.
///
/// The framing is identical either way: same version, same offsets table, same
/// payload digest. Inflating a [`Codec::Zstd`] frame rebuilds the exact bytes a
/// [`Codec::Stored`] file holds, so one validator holds both forms to one
/// standard, and a head's sections cannot drift between them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Codec {
    /// Sections stored verbatim after the header page.
    Stored,
    /// One zstd frame of that same region, padding included.
    Zstd,
}

impl Codec {
    const fn magic(self) -> &'static [u8; CACHE_MAGIC_BYTES] {
        match self {
            Self::Stored => CACHE_MAGIC,
            Self::Zstd => CACHE_MAGIC_ZSTD,
        }
    }

    pub(super) fn from_magic(magic: [u8; CACHE_MAGIC_BYTES]) -> Option<Self> {
        if magic == *CACHE_MAGIC {
            Some(Self::Stored)
        } else if magic == *CACHE_MAGIC_ZSTD {
            Some(Self::Zstd)
        } else {
            None
        }
    }

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::Zstd => "zstd",
        }
    }
}

/// The distributable int8 head, installed beside the pinned BF16 source and
/// written by `local-ai bonsai --export-mtp-head`.
///
/// Its version is [`CACHE_VERSION`], so a future format is a different file
/// rather than a misparse of this one. It deliberately does *not* start with
/// `mtp-head-v`, so the collector in [`write_cache`] can never reclaim a shipped
/// artifact. The name says nothing about how the file encodes its sections —
/// [`Codec`] does, in the file's own magic — because the installed artifact has
/// one path and this is the only way a compressed head can be found there.
pub const MTP_HEAD_ARTIFACT: &str = "mtp-head-int8-v2.bin";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadCacheStatus {
    Hit,
    Miss,
    Disabled,
}

impl HeadCacheStatus {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::Disabled => "disabled",
        }
    }
}

/// The immutable bytes a head's sections live in.
///
/// A stored head keeps them in the file's own mapping, which is why loading one
/// costs no anonymous RAM: the kernel demand-pages the sections a GPU read
/// actually touches and reclaims the rest, and `from_bytes_no_copy` hands Metal
/// pointers straight into those pages. An inflated head has no mapping to borrow,
/// so its sections are ~425 MB of anonymous memory that stays resident for the
/// life of the process and cannot be paged out. That is the whole trade behind
/// [`write_zstd_artifact`], and the reason [`write_artifact`] is the default.
pub(super) enum Mapping {
    Mapped(Mmap),
    Inflated(Vec<u8>),
}

impl Deref for Mapping {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(map) => map,
            Self::Inflated(bytes) => bytes,
        }
    }
}

/// A validated head file: its bytes, where each section starts and how long it
/// is, and the digest its payload hashed to.
pub(super) struct CacheMap {
    pub(super) map: Arc<Mapping>,
    pub(super) sections: Vec<(usize, usize)>,
    pub(super) payload_sha256: String,
}

/// What one written head file ended up holding.
pub(super) struct HeadFile {
    pub(super) path: PathBuf,
    pub(super) file_bytes: usize,
    /// Bytes the same head occupies with its sections stored verbatim: equal to
    /// [`Self::file_bytes`] unless the file compressed them.
    pub(super) canonical_bytes: usize,
    pub(super) payload_bytes: usize,
    pub(super) payload_sha256: String,
}

pub(super) fn align_cache(value: usize) -> Option<usize> {
    value
        .checked_add(CACHE_PAGE - 1)
        .map(|n| n / CACHE_PAGE * CACHE_PAGE)
}

/// Which of the four states a head file is in.
pub(super) enum HeadKind {
    /// An int8 head stored verbatim: its sections are mapped, so a load pays no
    /// anonymous RAM for them.
    Artifact,
    /// An int8 head whose section region is one zstd frame, read by inflating it
    /// into a buffer that costs ~425 MB of anonymous RAM for the process.
    ZstdArtifact,
    /// A BF16 safetensors source, to be quantized on first load.
    Safetensors,
    /// Too short to hold either format's discriminator, so it is neither. Named
    /// as its own case rather than guessed at, because guessing wrong here would
    /// hide a broken install behind a warm cache.
    Unrecognizable(u64),
}

/// Classify `path` from its first eight bytes: an int8 magic — which says both
/// that this is a head artifact and how its sections are encoded — or a
/// safetensors length prefix, which cannot collide with either. Nothing larger is
/// read, which is what lets an int8 hit skip mapping the 849 MB source at all.
pub(super) fn head_kind(path: &Path) -> std::io::Result<HeadKind> {
    let mut magic = [0_u8; CACHE_MAGIC_BYTES];
    let mut file = File::open(path)?;
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(match Codec::from_magic(magic) {
            Some(Codec::Stored) => HeadKind::Artifact,
            Some(Codec::Zstd) => HeadKind::ZstdArtifact,
            None => HeadKind::Safetensors,
        }),
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
            Ok(HeadKind::Unrecognizable(file.metadata()?.len()))
        }
        Err(error) => Err(error),
    }
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
    malformed(format!("int8 head layout is wider than 64 bits: {error}"))
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
///
/// `image_bytes` and `file_bytes` are the same number for a stored head and
/// different ones for a compressed one, which is why both are carried: the
/// offsets table pins the image, and the file's own length is checked against the
/// header separately, so a frame that is truncated or has bytes appended behind
/// it is caught by name rather than as a decode failure.
struct HeadHeader {
    codec: Codec,
    sections: Vec<(usize, usize)>,
    stored: [u8; DIGEST_HEX],
    file_bytes: u64,
    image_bytes: usize,
}

/// Read and fully validate an int8 head file: framing, the pinned section names
/// and sizes, page alignment, section bounds, and the payload digest.
///
/// Both encodings take this path. A [`Codec::Zstd`] head is inflated into the
/// bytes a [`Codec::Stored`] head holds *before* any of it is validated, so the
/// two forms are held to one standard: the same spec binds them, and the same
/// digest check proves an inflation reproduced the payload rather than merely
/// something of the right length.
///
/// Every failure is reported rather than swallowed, so a caller with no BF16
/// source to rebuild from can turn the reason into a hard error instead of a
/// silent rebuild attempt with nothing to rebuild from.
#[allow(clippy::too_many_lines)]
pub(super) fn read_head(path: &Path, spec: &[Section]) -> Result<CacheMap, String> {
    let open = |error: std::io::Error| reject(format!("{}: {error}", path.display()));
    let file = File::open(path).map_err(open)?;
    // SAFETY: immutable mapping retained by every no-copy Metal buffer.
    #[allow(unsafe_code)]
    let map = unsafe { MmapOptions::new().map(&file) }.map_err(open)?;
    let header = map.get(..CACHE_HEADER_BYTES).ok_or_else(|| {
        reject(format!(
            "{} is shorter than one {CACHE_HEADER_BYTES}-byte header page",
            path.display()
        ))
    })?;
    let parsed = parse_header(header, path, spec)?;
    let map_len =
        u64::try_from(map.len()).map_err(|_| reject(format!("{} is enormous", path.display())))?;
    if parsed.file_bytes != map_len {
        return Err(reject(format!(
            "{}: header records {} bytes, file is {map_len}",
            path.display(),
            parsed.file_bytes
        )));
    }
    let image: Arc<Mapping> = match parsed.codec {
        Codec::Stored => Arc::new(Mapping::Mapped(map)),
        Codec::Zstd => Arc::new(Mapping::Inflated(inflate(&map, &parsed, path)?)),
    };
    // The last padded section ends the image either way. For a stored head the
    // image is the file, so this is also what rules out appended bytes; for an
    // inflated one the frame's decoded length already had to match it.
    if image.len() != parsed.image_bytes {
        return Err(reject(format!(
            "{}: sections end at byte {} of a {} byte file",
            path.display(),
            parsed.image_bytes,
            image.len()
        )));
    }
    let payload_sha256 = digest_of(
        parsed
            .sections
            .iter()
            .map(|&(offset, length)| &image[offset..offset + length]),
    );
    if payload_sha256.as_bytes() != parsed.stored.as_slice() {
        return Err(reject(format!(
            "{}: payload hashes to {payload_sha256}, header records {}",
            path.display(),
            String::from_utf8_lossy(&parsed.stored)
        )));
    }
    Ok(CacheMap {
        map: image,
        sections: parsed.sections,
        payload_sha256,
    })
}

/// Check a header page against `spec` and return what it claims. No section byte
/// is read here, so a file whose framing is wrong is rejected before anything is
/// inflated.
#[allow(clippy::too_many_lines)]
fn parse_header(header: &[u8], path: &Path, spec: &[Section]) -> Result<HeadHeader, String> {
    let mut cursor = HeaderReader {
        bytes: header,
        at: 0,
    };
    let codec = cursor
        .take(CACHE_MAGIC_BYTES)?
        .first_chunk::<CACHE_MAGIC_BYTES>()
        .copied()
        .and_then(Codec::from_magic)
        .ok_or_else(|| {
            reject(format!(
                "{} is not an int8 MTP head artifact",
                path.display()
            ))
        })?;
    let version = cursor.u32()?;
    if version != CACHE_VERSION {
        return Err(reject(format!(
            "{} is int8 MTP head version {version}, this build reads version {CACHE_VERSION}",
            path.display()
        )));
    }
    let count = cursor.u32()? as usize;
    if count != spec.len() {
        return Err(reject(format!(
            "{} holds {count} sections, this build binds {}",
            path.display(),
            spec.len()
        )));
    }
    let payload_len = cursor.u64()?;
    let file_bytes = cursor.u64()?;
    let stored = cursor
        .take(DIGEST_HEX)?
        .try_into()
        .map_err(|_| reject("header digest is truncated"))?;
    let names_len = cursor.u32()? as usize;
    let names = cursor.take(names_len)?;
    let mut decoded = Vec::with_capacity(count);
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
    // Section lengths alone cannot bind these sections: `k_proj` and `v_proj`
    // are both `[KV, WIDTH]`, so only the names say which matrix a section is.
    for (index, (found, (_, _))) in decoded.iter().zip(spec).enumerate() {
        if *found != spec[index].0 {
            return Err(reject(format!(
                "{}: section {index} is {found}, this build binds {}",
                path.display(),
                spec[index].0
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
    for (index, (&(offset, length), &(_, expected))) in sections.iter().zip(spec).enumerate() {
        let padded = align_cache(length).ok_or_else(|| {
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
        if !offset.is_multiple_of(CACHE_PAGE) {
            return Err(reject(format!(
                "{}: section {index} offset {offset} is not {CACHE_PAGE}-byte aligned",
                path.display()
            )));
        }
        if offset < CACHE_HEADER_BYTES {
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
        codec,
        sections,
        stored,
        file_bytes,
        image_bytes,
    })
}

/// Where the image a head's offsets describe ends: the last section's padded end.
/// For a stored head that is the file it maps; for a compressed one it is exactly
/// what its frame has to decode to.
fn image_end(sections: &[(usize, usize)], path: &Path) -> Result<usize, String> {
    sections.last().map_or(Ok(0), |&(offset, length)| {
        let padded = align_cache(length).ok_or_else(|| {
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

/// Undo a [`Codec::Zstd`] head's encoding: inflate its one frame into the exact
/// bytes a [`Codec::Stored`] head holds, header page and the gap below the first
/// section included, so every offset in the header means what it meant before and
/// one validator covers both forms.
///
/// The frame lands straight in the image's own section region rather than in a
/// buffer that is then copied in, so the peak is one image rather than two, and
/// the mapping is borrowed rather than kept: it is dropped the moment this
/// returns. What is left is the honest cost of the compressed form — ~425 MB of
/// anonymous RAM for the life of the process, against a mapped head's demand
/// pages.
fn inflate(map: &Mmap, header: &HeadHeader, path: &Path) -> Result<Vec<u8>, String> {
    if header.image_bytes < CACHE_PAGE {
        return Err(reject(format!(
            "{}: sections describe {} bytes, less than the page the first one starts on",
            path.display(),
            header.image_bytes
        )));
    }
    let mut image = vec![0_u8; header.image_bytes];
    let (page, rest) = image.split_at_mut(CACHE_HEADER_BYTES);
    page.copy_from_slice(
        map.get(..CACHE_HEADER_BYTES)
            .ok_or_else(|| reject(format!("{} has no header page", path.display())))?,
    );
    // The frame starts at the first section, not at the end of the header page:
    // the rest of that page is the gap a stored file leaves unwritten too.
    let (_, sections) = rest.split_at_mut(CACHE_PAGE - CACHE_HEADER_BYTES);
    let frame = map.get(CACHE_PAGE..).ok_or_else(|| {
        reject(format!(
            "{} has no frame after its header page",
            path.display()
        ))
    })?;
    let decoded = zstd::bulk::Decompressor::new()
        .and_then(|mut decoder| decoder.decompress_to_buffer(frame, sections))
        .map_err(|error| {
            reject(format!(
                "{}: zstd frame is unusable: {error}",
                path.display()
            ))
        })?;
    if decoded != sections.len() {
        return Err(reject(format!(
            "{}: zstd frame decodes to {decoded} bytes, its header describes {}",
            path.display(),
            sections.len()
        )));
    }
    Ok(image)
}

/// Read a machine cache file. Beyond [`read_head`] this insists the file is named
/// after its own payload digest, so a renamed or hand-edited file cannot pass as
/// a cache entry.
pub(super) fn read_cache(path: &Path, spec: &[Section]) -> Result<CacheMap, String> {
    let named = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.strip_prefix(&cache_prefix()))
        .filter(|digest| digest.len() == DIGEST_HEX);
    let cache = read_head(path, spec)?;
    if Some(cache.payload_sha256.as_str()) != named {
        return Err(reject(format!(
            "{} is named for {named:?}, not for its payload digest {}",
            path.display(),
            cache.payload_sha256
        )));
    }
    Ok(cache)
}

fn cache_prefix() -> String {
    format!("mtp-head-v{CACHE_VERSION}-")
}

/// Where each section lands, and where the file then ends. The first section
/// starts one page in, so the header page and the alignment padding never
/// overlap one, and every later section starts on a page boundary: identical
/// payloads therefore produce identical files.
fn layout(sections: &[Vec<u8>]) -> Result<(Vec<(usize, usize)>, usize), String> {
    let mut offsets = Vec::with_capacity(sections.len());
    let mut end = CACHE_PAGE;
    for section in sections {
        let padded =
            align_cache(section.len()).ok_or_else(|| reject("int8 head section size overflows"))?;
        offsets.push((end, section.len()));
        end = end
            .checked_add(padded)
            .ok_or_else(|| reject("int8 head file size overflows"))?;
    }
    Ok((offsets, end))
}

fn encode_header(
    offsets: &[(usize, usize)],
    spec: &[Section],
    codec: Codec,
    digest: &str,
    payload_bytes: usize,
    file_bytes: usize,
) -> Result<Vec<u8>, String> {
    let stored = digest.as_bytes();
    if stored.len() != DIGEST_HEX || !stored.iter().all(u8::is_ascii_hexdigit) {
        return Err(reject("payload digest is not 64 hex-encoded bytes"));
    }
    let mut out = HeaderWriter { bytes: Vec::new() };
    // The magic is the whole discriminator: everything past this byte is framed
    // the same way whichever codec the file carries.
    out.raw(codec.magic());
    out.u32(CACHE_VERSION);
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
    if out.bytes.len() > CACHE_HEADER_BYTES {
        return Err(reject(format!(
            "int8 head header needs {} bytes, a page holds {CACHE_HEADER_BYTES}",
            out.bytes.len()
        )));
    }
    out.bytes.resize(CACHE_HEADER_BYTES, 0);
    Ok(out.bytes)
}

/// Where each section lands, how many payload bytes they hold between them, and
/// how large the image they occupy is.
type Framing = (Vec<(usize, usize)>, usize, usize);

/// The canonical placement of `sections` under `spec`, their payload size, and the
/// image they occupy: what every writer frames a head with and what every reader
/// expects back. Nothing here knows which codec the file will carry, because the
/// compressed form is this same image behind one frame.
fn framed(spec: &[Section], sections: &[Vec<u8>]) -> Result<Framing, std::io::Error> {
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
    let (offsets, image_bytes) = layout(sections).map_err(malformed)?;
    let payload_bytes = sections
        .iter()
        .try_fold(0_usize, |total, section| total.checked_add(section.len()))
        .ok_or_else(|| malformed(reject("payload length overflows")))?;
    Ok((offsets, payload_bytes, image_bytes))
}

/// A writer's scratch name beside `path`. A crashed writer can leave one behind;
/// the process id in it makes clearing it safe.
fn temp_path(path: &Path, digest: &str) -> PathBuf {
    path.with_file_name(format!(
        ".mtp-head-{}-{}.tmp",
        std::process::id(),
        digest.get(..16).unwrap_or(digest)
    ))
}
/// Write `sections` to `path` through a temporary file, so a reader either sees a
/// whole head or none of it. `digest` must be the payload digest: recording it in
/// the header is what lets the file validate itself on the next read.
fn write_head(
    path: &Path,
    spec: &[Section],
    sections: &[Vec<u8>],
    digest: &str,
) -> std::io::Result<HeadFile> {
    let (offsets, payload_bytes, image_bytes) = framed(spec, sections)?;
    let header = encode_header(
        &offsets,
        spec,
        Codec::Stored,
        digest,
        payload_bytes,
        image_bytes,
    )
    .map_err(malformed)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = temp_path(path, digest);
    // A crashed writer can leave a temporary behind; the process id in the name
    // makes clearing it safe.
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
        canonical_bytes: image_bytes,
        payload_bytes,
        payload_sha256: digest.to_owned(),
    })
}

/// Write `sections` to `path` with their region behind one zstd frame: the same
/// canonical head [`write_head`] produces, with everything after the header page
/// compressed.
///
/// The frame covers exactly the bytes a stored head holds there — every section,
/// padding included — so inflating it rebuilds that file byte for byte: same
/// offsets, same sections, same payload digest, and the same spec binding both.
/// The padding goes in for exactly that reason; being runs of zeroes it costs a
/// few dozen bytes, and compressing it away would mean scattering the payload back
/// into gaps on every read to keep the offsets true.
///
/// Sections are streamed through the encoder rather than concatenated into one
/// image first, so this never holds a second copy of the head.
///
/// What it costs the reader is the reason this is opt-in. A stored head is mapped,
/// and `from_bytes_no_copy` hands Metal pointers into pages the kernel
/// demand-pages and reclaims. An inflated head has to exist in full before a
/// single section can be read, so every load through one spends ~425 MB of
/// anonymous RAM that stays resident and cannot be paged out. Seventy megabytes of
/// disk is not worth that on every speculative load, which is why
/// [`write_artifact`] remains what an export calls by default.
pub(super) fn write_zstd_artifact(
    path: &Path,
    spec: &[Section],
    sections: &[Vec<u8>],
) -> std::io::Result<HeadFile> {
    let (offsets, payload_bytes, image_bytes) = framed(spec, sections)?;
    let digest = digest_of(sections.iter().map(Vec::as_slice));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = temp_path(path, &digest);
    let _ = std::fs::remove_file(&temp);
    let written = (|| -> std::io::Result<usize> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        // The frame starts where the first section would, one page past the
        // header. The header records how long the frame turned out to be, so it
        // is written last and the file only takes its real name once both are
        // down: a reader cannot find a half-written head under the installed
        // name.
        file.seek(SeekFrom::Start(u64::try_from(CACHE_PAGE).map_err(widen)?))?;
        let mut encoder = zstd::stream::write::Encoder::new(&mut file, ZSTD_LEVEL)?;
        // The encoder is told how many bytes are coming, so the frame records its
        // own decoded length and `finish` fails outright if the sections and their
        // padding did not add up to the image the header is about to describe.
        encoder.set_pledged_src_size(Some(
            u64::try_from(image_bytes - CACHE_PAGE).map_err(widen)?,
        ))?;
        let padding = vec![0_u8; CACHE_PAGE];
        for (&(_, length), section) in offsets.iter().zip(sections) {
            let padded = align_cache(length)
                .ok_or_else(|| malformed(reject("int8 head section size overflows")))?;
            encoder.write_all(section)?;
            encoder.write_all(
                padding
                    .get(..padded - length)
                    .ok_or_else(|| malformed(reject("int8 head section padding overflows")))?,
            )?;
        }
        let file = encoder.finish()?;
        let frame_bytes = usize::try_from(file.stream_position()?)
            .map_err(widen)?
            .checked_sub(CACHE_PAGE)
            .ok_or_else(|| malformed(reject("compressed int8 head has no frame")))?;
        let file_bytes = CACHE_PAGE + frame_bytes;
        let header = encode_header(
            &offsets,
            spec,
            Codec::Zstd,
            &digest,
            payload_bytes,
            file_bytes,
        )
        .map_err(malformed)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)?;
        Ok(file_bytes)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    let file_bytes = written?;
    Ok(HeadFile {
        path: path.to_owned(),
        file_bytes,
        canonical_bytes: image_bytes,
        payload_bytes,
        payload_sha256: digest,
    })
}

/// Write the machine cache, named after its payload so the name means the same
/// thing on every machine, and reclaim every other generation in `directory`.
///
/// The collector matches on the `mtp-head-v` prefix and the `.bin` extension, so
/// it also reclaims the mtime-keyed v1 files this format replaces.
pub(super) fn write_cache(
    directory: &Path,
    spec: &[Section],
    sections: &[Vec<u8>],
) -> std::io::Result<HeadFile> {
    std::fs::create_dir_all(directory)?;
    let digest = digest_of(sections.iter().map(Vec::as_slice));
    let head = write_head(&cache_path(directory, &digest), spec, sections, &digest)?;
    // The head is already renamed into place, so a collector failure costs disk,
    // not the load: the caller falls back to the sections it just built.
    if let Ok(entries) = std::fs::read_dir(directory) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path != head.path
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("mtp-head-v")
                            && Path::new(name).extension().is_some_and(|extension| {
                                extension.eq_ignore_ascii_case(CACHE_EXTENSION)
                            })
                    })
            {
                let _ = std::fs::remove_file(path);
            }
        }
    }
    Ok(head)
}

/// Write one shipped artifact at an explicit path, which discovery then finds by
/// name. Only the file's own directory is touched: a user's install directory is
/// never swept for old generations.
///
/// This is the stored form, and the default an export takes, because a load of it
/// costs no anonymous RAM: the sections stay in the file's own mapping and reach
/// Metal as demand-paged pointers. [`write_zstd_artifact`] writes the same head in
/// ~70 MB less, and is worth choosing only when the RAM its readers spend is worth
/// less than the disk.
pub(super) fn write_artifact(
    path: &Path,
    spec: &[Section],
    sections: &[Vec<u8>],
) -> std::io::Result<HeadFile> {
    let digest = digest_of(sections.iter().map(Vec::as_slice));
    write_head(path, spec, sections, &digest)
}

/// The single content address a machine cache directory holds. [`write_cache`]
/// keeps exactly one generation, so it is the only candidate, and its own digest
/// is what proves it. A missing or unreadable directory yields `None`; a present
/// but invalid file is reported by [`read_cache`].
pub(super) fn find_cache(directory: &Path) -> Option<PathBuf> {
    let prefix = cache_prefix();
    std::fs::read_dir(directory)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(CACHE_EXTENSION))
        })
        .min()
}

fn cache_path(directory: &Path, digest: &str) -> PathBuf {
    directory.join(format!("{}{digest}.{CACHE_EXTENSION}", cache_prefix()))
}
