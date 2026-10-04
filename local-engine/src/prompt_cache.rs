//! Versioned, integrity-checked disk tier for Bonsai prompt snapshots.

use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::bonsai_native::{PageBytes, PromptSnapshot};

const MAGIC: &[u8; 8] = b"BPCACHE1";

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
    hash_exact(&mut file, token_bytes, &mut digest, None)?;
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
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
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
    let mut file = File::create(&temporary)?;
    file.write_all(MAGIC)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    for token in tokens {
        file.write_all(&token.to_le_bytes())?;
    }
    file.write_all(&snapshot.recurrent)?;
    file.write_all(&snapshot.mtp_prev_hidden)?;
    file.write_all(&snapshot.target_kv)?;
    file.write_all(&snapshot.mtp_kv)?;
    file.sync_all()?;
    fs::rename(&temporary, &path)?;
    Ok(path)
}

pub fn trim(root: &Path, model_key: &str, budget: u64) {
    let mut files = discover(root, model_key);
    let mut bytes = files
        .iter()
        .filter_map(|e| fs::metadata(&e.path).ok())
        .map(|m| m.len())
        .sum::<u64>();
    while bytes > budget && files.len() > 1 {
        let victim = files
            .iter()
            .position(|entry| !entry.reusable_boundary)
            .unwrap_or(0);
        let entry = files.remove(victim);
        if let Ok(metadata) = fs::metadata(&entry.path) {
            bytes = bytes.saturating_sub(metadata.len());
        }
        let _ = fs::remove_file(entry.path);
    }
}

fn read_hashed(file: &mut File, length: usize, digest: &mut Sha256) -> crate::Result<PageBytes> {
    let mut bytes = PageBytes::zeroed(length)?;
    hash_exact(file, length, digest, Some(&mut bytes))?;
    Ok(bytes)
}

fn hash_exact(
    file: &mut File,
    length: usize,
    digest: &mut Sha256,
    mut output: Option<&mut [u8]>,
) -> crate::Result<()> {
    let mut remaining = length;
    let mut offset = 0;
    let mut scratch = [0_u8; 16 * 1024];
    while remaining != 0 {
        let count = remaining.min(scratch.len());
        file.read_exact(&mut scratch[..count])?;
        digest.update(&scratch[..count]);
        if let Some(bytes) = output.as_deref_mut() {
            bytes[offset..offset + count].copy_from_slice(&scratch[..count]);
        }
        remaining -= count;
        offset += count;
    }
    Ok(())
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
mod tests {
    use super::*;

    #[test]
    fn disk_snapshot_round_trip_and_corruption_rejection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = PromptSnapshot {
            position: 3,
            layout: "f16".into(),
            recurrent: PageBytes::concat([&[1, 2, 3][..]]).expect("bytes"),
            mtp_prev_hidden: PageBytes::concat([&[4, 5][..]]).expect("bytes"),
            target_kv: PageBytes::concat([&[6, 7, 8, 9][..]]).expect("bytes"),
            mtp_kv: PageBytes::concat([&[10, 11][..]]).expect("bytes"),
        };
        let path = store(
            dir.path(),
            "model",
            &[12, 34, 56],
            Some("a"),
            &snapshot,
            true,
        )
        .expect("store snapshot");
        let entries = discover(dir.path(), "model");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].tokens, [12, 34, 56]);
        assert!(entries[0].reusable_boundary);
        let loaded = load(&path, "model").expect("load snapshot");
        assert_eq!(loaded.position, snapshot.position);
        assert_eq!(*loaded.target_kv, *snapshot.target_kv);
        let mut bytes = fs::read(&path).expect("read snapshot");
        let last = bytes.last_mut().expect("non-empty snapshot");
        *last ^= 1;
        fs::write(&path, bytes).expect("corrupt snapshot");
        assert!(load(&path, "model").is_err());
    }
}
