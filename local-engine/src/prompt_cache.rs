//! Versioned, integrity-checked disk tier for Bonsai prompt snapshots.

use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::bonsai_native::{PageBytes, PromptSnapshot};

mod writer;

pub use self::writer::{StoreJob, Writer};

const MAGIC: &[u8; 8] = b"BPCACHE1";

/// How old a `.tmp` must be before `trim` treats it as an orphan instead of a
/// write still in flight.
///
/// Why an age and not a lock: one engine's [`Writer`] runs its stores and trims
/// serially, but nothing excludes another process sharing the directory, and a
/// lock across processes would mean choosing a concurrency model for `store`,
/// `discover` and `trim` together. An age keeps the guarantee local to the single file
/// that needs it. Because `store` renames into place, a sweep can only ever
/// observe a temporary while a write is still running, and `write_temporary`
/// advances the mtime as it fills the file — so the only window where a live
/// temporary looks stale is the tail, the final `sync_all`, which does not touch
/// the mtime. That leaves "fully written, still syncing" as the oldest a
/// live temporary can look.
///
/// Ten minutes clears that tail with room to spare. A snapshot is ~152 MiB: at
/// the 500 MiB/s this module treats as its SSD-first threshold that is ~0.3 s
/// to write and roughly the same again to sync, and even a pathological
/// 10 MiB/s volume needs only ~15 s. Ten minutes is ~20x the slower of those
/// estimates and ~2000x the faster one, so no plausible `store` is ever a
/// candidate — while a SIGKILL or power-loss orphan, which will never run again
/// to clean up after itself, is reclaimed by the next `trim`.
const ORPHANED_TEMPORARY_AGE: Duration = Duration::from_secs(600);

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

pub fn store(
    root: &Path,
    model_key: &str,
    tokens: &[u32],
    session_id: Option<&str>,
    snapshot: &PromptSnapshot,
    reusable_boundary: bool,
) -> crate::Result<PathBuf> {
    let dir = root.join(model_key);
    fs::create_dir_all(&dir)?;
    let stored_tokens = token_bytes(tokens);
    let mut digest = Sha256::new();
    digest.update(&stored_tokens);
    digest.update(&snapshot.recurrent);
    digest.update(&snapshot.mtp_prev_hidden);
    digest.update(&snapshot.target_kv);
    digest.update(&snapshot.mtp_kv);
    let digest = hex(digest.finalize().as_slice());
    let header = Header {
        model_key: model_key.to_owned(),
        layout: snapshot.layout.clone(),
        position: snapshot.position,
        token_count: tokens.len(),
        recurrent: snapshot.recurrent.len(),
        mtp_hidden: snapshot.mtp_prev_hidden.len(),
        target_kv: snapshot.target_kv.len(),
        mtp_kv: snapshot.mtp_kv.len(),
        payload_sha256: digest.clone(),
        session_id: session_id.map(str::to_owned),
        reusable_boundary,
    };
    let header = serde_json::to_vec(&header)?;
    let short_digest = digest.get(..16).unwrap_or(&digest);
    let path = dir.join(format!("{}-{short_digest}.bpc", snapshot.position));
    let temporary = path.with_extension("tmp");
    let committed = write_temporary(&temporary, &header, &stored_tokens, snapshot)
        .and_then(|()| fs::rename(&temporary, &path).map_err(crate::Error::from));
    if committed.is_err() {
        // The temporary is invisible to `discover`, so nothing else would reclaim it.
        let _ = fs::remove_file(&temporary);
    }
    committed.map(|()| path)
}

fn write_temporary(
    temporary: &Path,
    header: &[u8],
    tokens: &[u8],
    snapshot: &PromptSnapshot,
) -> crate::Result<()> {
    let mut file = File::create(temporary)?;
    let mut preamble = Vec::with_capacity(MAGIC.len() + 8 + header.len() + tokens.len());
    preamble.extend_from_slice(MAGIC);
    preamble.extend_from_slice(&(header.len() as u64).to_le_bytes());
    preamble.extend_from_slice(header);
    preamble.extend_from_slice(tokens);
    file.write_all(&preamble)?;
    file.write_all(&snapshot.recurrent)?;
    file.write_all(&snapshot.mtp_prev_hidden)?;
    file.write_all(&snapshot.target_kv)?;
    file.write_all(&snapshot.mtp_kv)?;
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
/// can never be a victim, at any age. A temporary younger than
/// [`ORPHANED_TEMPORARY_AGE`] belongs to a running `store` and is left alone.
/// Nothing here feeds the disk budget: `discover` stays the only source of
/// snapshot sizes, so a live temporary can neither evict nor reorder a real
/// snapshot.
fn sweep_abandoned_temporaries(root: &Path, model_key: &str) {
    let dir = root.join(model_key);
    let Ok(files) = fs::read_dir(dir) else {
        return;
    };
    let Some(cutoff) = SystemTime::now().checked_sub(ORPHANED_TEMPORARY_AGE) else {
        return;
    };
    for file in files.flatten() {
        let path = file.path();
        if path.extension().and_then(|value| value.to_str()) != Some("tmp") {
            continue;
        }
        let abandoned = file
            .metadata()
            .and_then(|metadata| metadata.modified())
            .is_ok_and(|modified| modified < cutoff);
        if abandoned {
            let _ = fs::remove_file(&path);
        }
    }
}

/// Remove snapshots, oldest request tails first, until `budget` holds.
///
/// Reads the whole directory with `discover`. The engine trims through the
/// background [`Writer`], whose [`DiskIndex`] pays that only once per
/// directory; this standalone form is the reference the tests pin it to.
#[cfg(test)]
pub fn trim(root: &Path, model_key: &str, budget: u64) {
    // Reclaimed before the budget is computed, and independently of it: this
    // sweep is about files no snapshot ever referenced.
    sweep_abandoned_temporaries(root, model_key);
    DiskIndex::discover(root, model_key).evict_over(budget);
}

struct Indexed {
    path: PathBuf,
    bytes: u64,
    reusable_boundary: bool,
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
    fn insert(&mut self, path: &Path, reusable_boundary: bool) {
        // The name is the content digest, so an equal name is the same file
        // rewritten by `rename`, not a second one.
        self.entries.retain(|entry| entry.path != path);
        if let Ok(metadata) = fs::metadata(path) {
            self.entries.push(Indexed {
                path: path.to_owned(),
                bytes: metadata.len(),
                reusable_boundary,
            });
        }
    }

    /// Delete snapshots until the total fits `budget`, always keeping one.
    /// Request tails go before shared-prefix boundaries, oldest first.
    fn evict_over(&mut self, budget: u64) -> Vec<PathBuf> {
        let mut bytes = self.entries.iter().map(|entry| entry.bytes).sum::<u64>();
        let mut evicted = Vec::new();
        while bytes > budget && self.entries.len() > 1 {
            let victim = self
                .entries
                .iter()
                .position(|entry| !entry.reusable_boundary)
                .unwrap_or(0);
            let entry = self.entries.remove(victim);
            bytes = bytes.saturating_sub(entry.bytes);
            let _ = fs::remove_file(&entry.path);
            evicted.push(entry.path);
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
#[allow(clippy::expect_used)]
mod tests;
