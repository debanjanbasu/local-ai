//! Versioned, integrity-checked disk tier for Bonsai prompt snapshots.

use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::bonsai_native::{PageBytes, PromptSnapshot};

mod writer;

pub use self::writer::{Closed, Closer, StoreJob, Writer};

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
    payload_sha256: String,
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
    let payload_bytes = [
        token_bytes,
        header.recurrent,
        header.mtp_hidden,
        header.target_kv,
        header.mtp_kv,
    ]
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
    let mut digest = Sha256::new();
    let mut stored_tokens = vec![0; token_bytes];
    file.read_exact(&mut stored_tokens)?;
    digest.update(&stored_tokens);
    let recurrent = read_hashed(&mut file, header.recurrent, &mut digest)?;
    let mtp_prev_hidden = read_hashed(&mut file, header.mtp_hidden, &mut digest)?;
    let target_kv = read_hashed(&mut file, header.target_kv, &mut digest)?;
    let mtp_kv = read_hashed(&mut file, header.mtp_kv, &mut digest)?;
    let mut trailing = [0];
    if file.read(&mut trailing)? != 0 {
        return Err(crate::Error::InvalidFormat(
            "prompt cache trailing bytes".into(),
        ));
    }
    if hex(digest.finalize().as_slice()) != header.payload_sha256 {
        return Err(crate::Error::InvalidFormat(
            "prompt cache checksum mismatch".into(),
        ));
    }
    let snapshot = PromptSnapshot {
        position: header.position,
        layout: header.layout,
        recurrent,
        mtp_prev_hidden,
        target_kv,
        mtp_kv,
    };
    Ok(snapshot)
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

/// The header `store` writes, with `digest` as its checksum.
fn header(
    model_key: &str,
    tokens: &[u32],
    session_id: Option<&str>,
    snapshot: &PromptSnapshot,
    reusable_boundary: bool,
    digest: String,
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
        payload_sha256: digest,
        session_id: session_id.map(str::to_owned),
        reusable_boundary,
    }
}

/// The exact size of the file `store` would write, known before hashing: the
/// checksum is always 64 hex digits, so a placeholder of that length encodes
/// to the same header length.
fn stored_bytes(
    model_key: &str,
    tokens: &[u32],
    session_id: Option<&str>,
    snapshot: &PromptSnapshot,
    reusable_boundary: bool,
) -> u64 {
    let placeholder = "0".repeat(64);
    let header = header(
        model_key,
        tokens,
        session_id,
        snapshot,
        reusable_boundary,
        placeholder,
    );
    let header = serde_json::to_vec(&header).map_or(0, |bytes| bytes.len());
    [
        MAGIC.len(),
        8,
        header,
        tokens.len() * 4,
        snapshot.recurrent.len(),
        snapshot.mtp_prev_hidden.len(),
        snapshot.target_kv.len(),
        snapshot.mtp_kv.len(),
    ]
    .into_iter()
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
    let mut digest = Sha256::new();
    for section in sections(&stored_tokens, snapshot) {
        for piece in section.chunks(WRITE_PIECE) {
            if abort.load(Ordering::Acquire) {
                return Err(aborted());
            }
            digest.update(piece);
        }
    }
    let digest = hex(digest.finalize().as_slice());
    let short_digest = digest.get(..16).unwrap_or(&digest).to_owned();
    let header = header(
        model_key,
        tokens,
        session_id,
        snapshot,
        reusable_boundary,
        digest,
    );
    let header = serde_json::to_vec(&header)?;
    let path = dir.join(format!("{}-{short_digest}.bpc", snapshot.position));
    let temporary = path.with_extension("tmp");
    let committed = write_temporary(&temporary, &header, &stored_tokens, snapshot, abort)
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
    header: &[u8],
    tokens: &[u8],
    snapshot: &PromptSnapshot,
    abort: &AtomicBool,
) -> crate::Result<()> {
    // Locked until `file` drops, after the caller's rename or removal.
    let mut file = create_locked(temporary)?;
    let mut preamble = Vec::with_capacity(MAGIC.len() + 8 + header.len());
    preamble.extend_from_slice(MAGIC);
    preamble.extend_from_slice(&(header.len() as u64).to_le_bytes());
    preamble.extend_from_slice(header);
    file.write_all(&preamble)?;
    for section in sections(tokens, snapshot) {
        for piece in section.chunks(WRITE_PIECE) {
            if abort.load(Ordering::Acquire) {
                return Err(aborted());
            }
            file.write_all(piece)?;
        }
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

fn read_hashed(file: &mut File, length: usize, digest: &mut Sha256) -> crate::Result<PageBytes> {
    let mut bytes = PageBytes::zeroed(length)?;
    for chunk in bytes.chunks_mut(READ_CHUNK) {
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
