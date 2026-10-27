//! Versioned, integrity-checked disk tier for Bonsai prompt snapshots.

use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::bonsai_native::{PageBytes, PromptSnapshot};

mod blocks;
mod loader;
mod writer;

pub use self::loader::Load;
pub use self::writer::{Closed, Closer, StoreJob, Writer, daily_write_budget};

const MAGIC: &[u8; 8] = b"BPCACHE1";

/// Bytes hashed or written between checks of a shutdown abort, so an aborted
/// write of a ~10 GB snapshot stops within one piece (tens of milliseconds).
const WRITE_PIECE: usize = 64 * 1024 * 1024;

/// Darwin's `O_EXLOCK` (`<sys/fcntl.h>`): `open` takes an exclusive `flock` on
/// the new file atomically with creating it.
///
/// A temporary is live exactly while the process writing it holds this lock;
/// the kernel drops it when the descriptor closes, including on SIGKILL or a
/// crash. So whether a `.tmp` is an orphan is asked of the kernel, not guessed
/// from its age, and the answer is right across processes sharing the
/// directory: a sweep can reclaim an orphan the moment it exists (at startup,
/// and after every store) and can never reclaim a write still running. Taking
/// the lock in `open` itself leaves no window in which a just-created
/// temporary is unlocked.
#[cfg(target_os = "macos")]
const O_EXLOCK: i32 = 0x0000_0020;

#[derive(Serialize, Deserialize)]
struct Header {
    model_key: String,
    layout: String,
    position: usize,
    token_count: usize,
    recurrent: usize,
    mtp_hidden: usize,
    target_kv: usize,
    mtp_kv: usize,
    /// SHA-256 of the contiguous payload, in files written before the block
    /// layout; still read, never written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    payload_sha256: Option<String>,
    /// CRC-32C of each block of each section, in the block layout (see
    /// [`blocks`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    payload_crc32c: Option<String>,
    session_id: Option<String>,
    #[serde(default)]
    reusable_boundary: bool,
}

pub struct DiskEntry {
    pub path: PathBuf,
    pub tokens: Vec<u32>,
    pub session_id: Option<String>,
    pub reusable_boundary: bool,
}

pub fn discover(root: &Path, model_key: &str) -> Vec<DiskEntry> {
    let dir = root.join(model_key);
    let mut entries = Vec::new();
    let Ok(files) = fs::read_dir(dir) else {
        return entries;
    };
    for file in files.flatten() {
        let path = file.path();
        if path.extension().and_then(|value| value.to_str()) != Some("bpc") {
            continue;
        }
        if let Ok((header, token_bytes)) = read_index(&path)
            && header.model_key == model_key
        {
            let tokens = token_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect();
            entries.push(DiskEntry {
                path,
                tokens,
                session_id: header.session_id,
                reusable_boundary: header.reusable_boundary,
            });
        }
    }
    entries.sort_by_key(|entry| fs::metadata(&entry.path).and_then(|m| m.modified()).ok());
    entries
}

fn read_index(path: &Path) -> crate::Result<(Header, Vec<u8>)> {
    let mut file = File::open(path)?;
    let header = read_header(&mut file)?;
    let token_bytes = header
        .token_count
        .checked_mul(4)
        .ok_or_else(|| crate::Error::InvalidFormat("prompt cache token size overflow".into()))?;
    let mut tokens = vec![0; token_bytes];
    file.read_exact(&mut tokens)?;
    Ok((header, tokens))
}

pub fn load(path: &Path, model_key: &str) -> crate::Result<PromptSnapshot> {
    load_abortable(path, model_key, &AtomicBool::new(false))
}

/// `load`, giving up between blocks once `abort` is set.
fn load_abortable(
    path: &Path,
    model_key: &str,
    abort: &AtomicBool,
) -> crate::Result<PromptSnapshot> {
    let mut file = File::open(path)?;
    let header = read_header(&mut file)?;
    if header.model_key != model_key {
        return Err(crate::Error::InvalidFormat(
            "stale prompt cache model key".into(),
        ));
    }
    let token_bytes = header
        .token_count
        .checked_mul(4)
        .ok_or_else(|| crate::Error::InvalidFormat("prompt cache token size overflow".into()))?;
    let lengths = [
        token_bytes,
        header.recurrent,
        header.mtp_hidden,
        header.target_kv,
        header.mtp_kv,
    ];
    let start = file.stream_position()?;
    let file_length = file.metadata()?.len();
    let [_tokens, recurrent, mtp_prev_hidden, target_kv, mtp_kv] =
        match (&header.payload_crc32c, &header.payload_sha256) {
            (Some(digits), _) => blocks::read(&file, start, file_length, lengths, digits, abort)?,
            (None, Some(digest)) => read_sha256_payload(&mut file, lengths, digest, abort)?,
            (None, None) => {
                return Err(crate::Error::InvalidFormat(
                    "prompt cache has no checksum".into(),
                ));
            }
        };
    Ok(PromptSnapshot {
        position: header.position,
        layout: header.layout,
        recurrent,
        mtp_prev_hidden,
        target_kv,
        mtp_kv,
    })
}

/// Read the contiguous payload of a snapshot written before the block layout
/// from `file`'s position, and check it against its SHA-256 `digest`.
fn read_sha256_payload(
    file: &mut File,
    lengths: [usize; 5],
    digest: &str,
    abort: &AtomicBool,
) -> crate::Result<[PageBytes; 5]> {
    let payload_bytes = lengths
        .into_iter()
        .try_fold(0_usize, usize::checked_add)
        .ok_or_else(|| crate::Error::InvalidFormat("prompt cache section size overflow".into()))?;
    let remaining = file
        .metadata()?
        .len()
        .saturating_sub(file.stream_position()?);
    if u64::try_from(payload_bytes).ok() != Some(remaining) {
        return Err(crate::Error::InvalidFormat(
            "truncated prompt cache section".into(),
        ));
    }
    let mut hasher = Sha256::new();
    let mut sections = Vec::with_capacity(lengths.len());
    for length in lengths {
        sections.push(read_hashed(file, length, &mut hasher, abort)?);
    }
    let mut trailing = [0];
    if file.read(&mut trailing)? != 0 {
        return Err(crate::Error::InvalidFormat(
            "prompt cache trailing bytes".into(),
        ));
    }
    if hex(hasher.finalize().as_slice()) != digest {
        return Err(crate::Error::InvalidFormat(
            "prompt cache checksum mismatch".into(),
        ));
    }
    sections
        .try_into()
        .map_err(|_| crate::Error::InvalidFormat("prompt cache section count mismatch".into()))
}

#[cfg(test)]
pub fn store(
    root: &Path,
    model_key: &str,
    tokens: &[u32],
    session_id: Option<&str>,
    snapshot: &PromptSnapshot,
    reusable_boundary: bool,
) -> crate::Result<PathBuf> {
    store_abortable(
        root,
        model_key,
        tokens,
        session_id,
        snapshot,
        reusable_boundary,
        &AtomicBool::new(false),
    )
}

/// The header `store` writes, with `checksums` as the block checksums.
fn header(
    model_key: &str,
    tokens: &[u32],
    session_id: Option<&str>,
    snapshot: &PromptSnapshot,
    reusable_boundary: bool,
    checksums: String,
) -> Header {
    Header {
        model_key: model_key.to_owned(),
        layout: snapshot.layout.clone(),
        position: snapshot.position,
        token_count: tokens.len(),
        recurrent: snapshot.recurrent.len(),
        mtp_hidden: snapshot.mtp_prev_hidden.len(),
        target_kv: snapshot.target_kv.len(),
        mtp_kv: snapshot.mtp_kv.len(),
        payload_sha256: None,
        payload_crc32c: Some(checksums),
        session_id: session_id.map(str::to_owned),
        reusable_boundary,
    }
}

/// The section lengths of a snapshot of `tokens`, in file order.
fn section_lengths(tokens: &[u32], snapshot: &PromptSnapshot) -> [usize; 5] {
    [
        tokens.len() * 4,
        snapshot.recurrent.len(),
        snapshot.mtp_prev_hidden.len(),
        snapshot.target_kv.len(),
        snapshot.mtp_kv.len(),
    ]
}

/// What precedes the payload: the magic, the header's length and the header,
/// padded with spaces (which JSON ignores) so the payload starts on a
/// [`blocks::ALIGN`] boundary.
fn preamble(header: &Header) -> crate::Result<Vec<u8>> {
    let encoded = serde_json::to_vec(header)?;
    let fixed = MAGIC.len() + 8;
    let padded = (fixed + encoded.len()).next_multiple_of(blocks::ALIGN) - fixed;
    let mut preamble = Vec::with_capacity(fixed + padded);
    preamble.extend_from_slice(MAGIC);
    preamble.extend_from_slice(&(padded as u64).to_le_bytes());
    preamble.extend_from_slice(&encoded);
    preamble.resize(fixed + padded, b' ');
    Ok(preamble)
}

/// The exact size of the file `store` would write, known before checksumming:
/// the checksums take a fixed number of digits per block, so a placeholder of
/// that length encodes to the same header length.
fn stored_bytes(
    model_key: &str,
    tokens: &[u32],
    session_id: Option<&str>,
    snapshot: &PromptSnapshot,
    reusable_boundary: bool,
) -> u64 {
    let lengths = section_lengths(tokens, snapshot);
    let placeholder = "0".repeat(blocks::checksum_digits(lengths));
    let header = header(
        model_key,
        tokens,
        session_id,
        snapshot,
        reusable_boundary,
        placeholder,
    );
    let preamble = preamble(&header).map_or(0, |bytes| bytes.len());
    std::iter::once(preamble)
        .chain(lengths.map(blocks::padded))
        .map(|bytes| bytes as u64)
        .sum()
}

fn aborted() -> crate::Error {
    crate::Error::Io(std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        "prompt cache write aborted at shutdown",
    ))
}

/// `store`, giving up between pieces once `abort` is set. An aborted or
/// failed store removes its temporary, so it leaves nothing behind.
fn store_abortable(
    root: &Path,
    model_key: &str,
    tokens: &[u32],
    session_id: Option<&str>,
    snapshot: &PromptSnapshot,
    reusable_boundary: bool,
    abort: &AtomicBool,
) -> crate::Result<PathBuf> {
    let dir = root.join(model_key);
    fs::create_dir_all(&dir)?;
    let stored_tokens = token_bytes(tokens);
    let sections = sections(&stored_tokens, snapshot);
    let checksums = blocks::checksums(&sections, WRITE_PIECE, abort).ok_or_else(aborted)?;
    // Named by its content, as before, so a snapshot rewritten unchanged
    // replaces its file: the checksums of every block stand for it.
    let name = hex(&Sha256::digest(checksums.as_bytes())[..8]);
    let header = header(
        model_key,
        tokens,
        session_id,
        snapshot,
        reusable_boundary,
        checksums,
    );
    let preamble = preamble(&header)?;
    let path = dir.join(format!("{}-{name}.bpc", snapshot.position));
    let temporary = path.with_extension("tmp");
    let committed = write_temporary(&temporary, &preamble, &sections, abort)
        .and_then(|()| fs::rename(&temporary, &path).map_err(crate::Error::from));
    if committed.is_err() {
        // The temporary is invisible to `discover`, so nothing else would reclaim it.
        let _ = fs::remove_file(&temporary);
    }
    committed.map(|()| path)
}

/// The payload sections in file order, after the header.
fn sections<'a>(tokens: &'a [u8], snapshot: &'a PromptSnapshot) -> [&'a [u8]; 5] {
    [
        tokens,
        &snapshot.recurrent,
        &snapshot.mtp_prev_hidden,
        &snapshot.target_kv,
        &snapshot.mtp_kv,
    ]
}

/// Create `temporary` holding its exclusive lock (see [`O_EXLOCK`]).
fn create_locked(temporary: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(O_EXLOCK);
        options.open(temporary)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let file = options.open(temporary)?;
        file.lock()?;
        Ok(file)
    }
}

fn write_temporary(
    temporary: &Path,
    preamble: &[u8],
    sections: &[&[u8]],
    abort: &AtomicBool,
) -> crate::Result<()> {
    // Locked until `file` drops, after the caller's rename or removal.
    let mut file = create_locked(temporary)?;
    file.write_all(preamble)?;
    let padding = [0; blocks::ALIGN];
    for section in sections {
        for piece in section.chunks(WRITE_PIECE) {
            if abort.load(Ordering::Acquire) {
                return Err(aborted());
            }
            file.write_all(piece)?;
        }
        file.write_all(&padding[..blocks::padded(section.len()) - section.len()])?;
    }
    file.sync_all()?;
    Ok(())
}

/// Reclaim temporaries abandoned by a `store` that never reached `fs::rename`.
///
/// Nothing else covers this window: `store` removes its own temporary on the
/// error path, but SIGKILL and power loss run no Rust code at all, and the
/// temporary is invisible to `discover`, so it is invisible to the budget too.
///
/// Only the `.tmp` extension is swept, and only inside this model's directory,
/// where every `.tmp` was written by `write_temporary` — a committed snapshot
/// can never be a victim. A temporary whose lock some process still holds
/// belongs to a running `store` and is left alone; any other is an orphan,
/// whatever its age (see [`O_EXLOCK`]). Nothing here feeds the disk budget:
/// `discover` stays the only source of snapshot sizes, so a live temporary can
/// neither evict nor reorder a real snapshot.
fn sweep_abandoned_temporaries(root: &Path, model_key: &str) {
    let dir = root.join(model_key);
    let Ok(files) = fs::read_dir(dir) else {
        return;
    };
    for file in files.flatten() {
        let path = file.path();
        if path.extension().and_then(|value| value.to_str()) != Some("tmp") {
            continue;
        }
        let Ok(handle) = File::open(&path) else {
            continue;
        };
        if handle.try_lock().is_ok() {
            let _ = fs::remove_file(&path);
        }
    }
}

/// Bring a prompt-cache directory back to a state a running engine maintains:
/// no orphaned temporaries and the committed snapshots within `budget`.
///
/// Run when an engine adopts the directory, before it serves. A previous
/// process can leave it either way: killed mid-write (a temporary), or stopped
/// between a commit and the trim that follows it, or run with a larger budget
/// (free disk is part of the automatic one). Returns the snapshots deleted.
pub fn reclaim(root: &Path, model_key: &str, budget: u64) -> Vec<PathBuf> {
    sweep_abandoned_temporaries(root, model_key);
    DiskIndex::discover(root, model_key).evict_over(budget)
}

/// Remove snapshots, by the eviction policy of [`eviction_order`], until
/// `budget` holds.
///
/// Reads the whole directory with `discover`. The engine trims through the
/// background [`Writer`], whose [`DiskIndex`] pays that only once per
/// directory; this standalone form is the reference the tests pin it to.
#[cfg(test)]
pub fn trim(root: &Path, model_key: &str, budget: u64) {
    // Reclaimed before the budget is computed, and independently of it: this
    // sweep is about files no snapshot ever referenced.
    let _ = reclaim(root, model_key, budget);
}

struct Indexed {
    path: PathBuf,
    tokens: Vec<u32>,
    bytes: u64,
    reusable_boundary: bool,
}

/// Whether shared-prefix boundary `child` only barely extends boundary
/// `parent`: `parent` is a proper prefix of it, and the tokens it adds are
/// fewer than `parent` holds.
///
/// Every request `child` serves, `parent` serves too, so when the budget
/// cannot hold both, keeping `parent` costs a request that shares `child` the
/// prefill of the extension, while keeping `child` costs a request that shares
/// only `parent` the prefill of all of `parent`. With no knowledge of which
/// requests come next, `parent` is the cheaper loss whenever the extension is
/// the shorter of the two. The case this exists for: questions that share a
/// 128K-token repository and a system turn diverge from one another a few
/// tokens into their own text, and a boundary there (the first words of one
/// kind of question) must not replace the system-turn boundary every question
/// restores.
pub fn barely_extends(child: &[u32], parent: &[u32]) -> bool {
    parent.len() < child.len()
        && child.len() - parent.len() < parent.len()
        && child.starts_with(parent)
}

/// Which of `entries` (tokens and whether each is a shared-prefix boundary,
/// oldest first) to evict next, when one must go:
///
/// 1. the oldest request tail;
/// 2. else the newest boundary that barely extends another one (see
///    [`barely_extends`]);
/// 3. else the oldest boundary.
///
/// The disk tier applies it until its byte budget holds, the GPU tier when
/// it holds one checkpoint too many, to all but its two newest checkpoints.
pub fn next_victim(entries: &[(&[u32], bool)]) -> Option<usize> {
    let near_duplicate = |&(tokens, boundary): &(&[u32], bool)| {
        boundary
            && entries
                .iter()
                .any(|&(other, shared)| shared && barely_extends(tokens, other))
    };
    entries
        .iter()
        .position(|&(_, boundary)| !boundary)
        .or_else(|| entries.iter().rposition(near_duplicate))
        .or_else(|| (!entries.is_empty()).then_some(0))
}

/// Whether the writer would keep a snapshot of `tokens` taking about `bytes`
/// beside the committed `entries` (oldest first) under `budget`: the check
/// `DiskIndex::admits` makes before a write, made before the snapshot is
/// read back from the GPU, which at long prompts costs gigabytes of copying
/// on the engine thread. Writes still in flight are not counted, so this errs
/// toward a readback the writer may then refuse.
pub fn would_admit(
    entries: &[DiskEntry],
    tokens: &[u32],
    bytes: u64,
    reusable_boundary: bool,
    budget: u64,
) -> bool {
    let mut index = DiskIndex {
        dir: PathBuf::new(),
        entries: entries
            .iter()
            .filter_map(|entry| {
                Some(Indexed {
                    path: entry.path.clone(),
                    tokens: entry.tokens.clone(),
                    bytes: fs::metadata(&entry.path).ok()?.len(),
                    reusable_boundary: entry.reusable_boundary,
                })
            })
            .collect(),
    };
    index.admits(tokens, bytes, reusable_boundary, budget)
}

/// The order in which entries (oldest first) are evicted to fit `budget`,
/// by [`next_victim`], always keeping one.
fn eviction_order(entries: &[Indexed], budget: u64) -> Vec<usize> {
    let mut live = (0..entries.len()).collect::<Vec<_>>();
    let mut bytes = entries.iter().map(|entry| entry.bytes).sum::<u64>();
    let mut order = Vec::new();
    while bytes > budget && live.len() > 1 {
        let view = live
            .iter()
            .map(|&index| {
                (
                    entries[index].tokens.as_slice(),
                    entries[index].reusable_boundary,
                )
            })
            .collect::<Vec<_>>();
        let Some(position) = next_victim(&view) else {
            break;
        };
        let victim = live.remove(position);
        bytes = bytes.saturating_sub(entries[victim].bytes);
        order.push(victim);
    }
    order
}

/// The committed snapshots of one model directory, oldest first, with the
/// sizes the budget is computed from.
///
/// Seeded by `discover` and then kept current by its owner's own stores and
/// evictions. Files another process adds are not seen until the next seed;
/// files deleted behind its back are counted until an eviction reaches them,
/// which errs toward evicting early, never toward exceeding the budget.
struct DiskIndex {
    dir: PathBuf,
    entries: Vec<Indexed>,
}

impl DiskIndex {
    fn discover(root: &Path, model_key: &str) -> Self {
        let entries = discover(root, model_key)
            .into_iter()
            .filter_map(|entry| {
                let bytes = fs::metadata(&entry.path).ok()?.len();
                Some(Indexed {
                    path: entry.path,
                    tokens: entry.tokens,
                    bytes,
                    reusable_boundary: entry.reusable_boundary,
                })
            })
            .collect();
        Self {
            dir: root.join(model_key),
            entries,
        }
    }

    fn is_for(&self, root: &Path, model_key: &str) -> bool {
        self.dir == root.join(model_key)
    }

    /// Record a snapshot just committed at `path` as the newest.
    fn insert(&mut self, path: &Path, tokens: &[u32], reusable_boundary: bool) {
        // The name is the content digest, so an equal name is the same file
        // rewritten by `rename`, not a second one.
        self.entries.retain(|entry| entry.path != path);
        if let Ok(metadata) = fs::metadata(path) {
            self.entries.push(Indexed {
                path: path.to_owned(),
                tokens: tokens.to_vec(),
                bytes: metadata.len(),
                reusable_boundary,
            });
        }
    }

    /// Whether a snapshot of `bytes` would survive the trim after its own
    /// store. One the trim would delete again is not worth writing: a request
    /// tail behind shared boundaries that fill the budget, or a boundary that
    /// only barely extends a kept one.
    fn admits(&mut self, tokens: &[u32], bytes: u64, reusable_boundary: bool, budget: u64) -> bool {
        // An entry with the same tokens is replaced by the store, not kept
        // beside it.
        let same = self.entries.iter().position(|entry| entry.tokens == tokens);
        let replaced = same.map(|index| self.entries.remove(index));
        self.entries.push(Indexed {
            path: PathBuf::new(),
            tokens: tokens.to_vec(),
            bytes,
            reusable_boundary,
        });
        let candidate = self.entries.len() - 1;
        let admitted = !eviction_order(&self.entries, budget).contains(&candidate);
        self.entries.pop();
        if let (Some(index), Some(entry)) = (same, replaced) {
            self.entries.insert(index, entry);
        }
        admitted
    }

    /// Delete snapshots until the total fits `budget`, by
    /// [`eviction_order`], always keeping one.
    fn evict_over(&mut self, budget: u64) -> Vec<PathBuf> {
        let mut order = eviction_order(&self.entries, budget);
        let evicted = order
            .iter()
            .map(|&index| {
                let path = self.entries[index].path.clone();
                let _ = fs::remove_file(&path);
                path
            })
            .collect();
        order.sort_unstable();
        for index in order.into_iter().rev() {
            self.entries.remove(index);
        }
        evicted
    }
}

/// Bytes read per `read` call when restoring a section.
///
/// Reading straight into the destination and hashing each piece while it is
/// still in cache replaces 16 KiB reads through a scratch buffer, which cost a
/// syscall and an extra copy per 16 KiB of a ~152 MiB snapshot.
const READ_CHUNK: usize = 1024 * 1024;

fn read_hashed(
    file: &mut File,
    length: usize,
    digest: &mut Sha256,
    abort: &AtomicBool,
) -> crate::Result<PageBytes> {
    let mut bytes = PageBytes::zeroed(length)?;
    for chunk in bytes.chunks_mut(READ_CHUNK) {
        if abort.load(Ordering::Acquire) {
            return Err(crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "prompt cache load abandoned",
            )));
        }
        file.read_exact(chunk)?;
        digest.update(&*chunk);
    }
    Ok(bytes)
}

/// The token section as stored: little-endian `u32`s, one contiguous buffer, so
/// it is hashed and written in one call each rather than once per token.
fn token_bytes(tokens: &[u32]) -> Vec<u8> {
    tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect()
}

fn read_header(file: &mut File) -> crate::Result<Header> {
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(crate::Error::InvalidFormat(
            "invalid prompt cache magic".into(),
        ));
    }
    let mut size = [0; 8];
    file.read_exact(&mut size)?;
    let size = usize::try_from(u64::from_le_bytes(size))
        .map_err(|_| crate::Error::InvalidFormat("prompt cache header too large".into()))?;
    if size > 1024 * 1024 {
        return Err(crate::Error::InvalidFormat(
            "prompt cache header too large".into(),
        ));
    }
    let mut encoded = vec![0; size];
    file.read_exact(&mut encoded)?;
    Ok(serde_json::from_slice(&encoded)?)
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

#[cfg(test)]
mod tests;
