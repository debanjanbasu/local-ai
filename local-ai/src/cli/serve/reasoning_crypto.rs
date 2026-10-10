//! Stateless encrypted reasoning replay.
//!
//! A [`ReasoningCipher`] seals reasoning text into an opaque, versioned
//! base64url envelope that clients may hold and send back later; nothing but
//! the key is persisted server-side. Envelopes are AES-256-GCM (via `ring`)
//! with a fresh random 96-bit nonce from the OS for every seal, and the
//! envelope version, model ID and item ID are bound as associated data, so an
//! altered envelope, one sealed for another model or item, one from another
//! key, or one claiming another version fails to open.
//!
//! The key is 32 random bytes in an owner-only file that is created on first
//! use and reused on every later start. Creating it never overwrites anything:
//! the key is written and synced to a private temporary file which is then
//! hard-linked into place, an atomic create-if-absent that also keeps
//! concurrent openers from ever seeing a partially written key. Existing key
//! files are opened without following symlinks and rejected unless they are
//! regular files owned by the current user with no group or other access.
//!
//! Envelope layout, base64url without padding:
//! `version (1 byte) || nonce (12 bytes) || ciphertext || tag (16 bytes)`.
//! Associated data: `DOMAIN || version || len(model_id) || model_id ||
//! len(item_id) || item_id`, lengths as big-endian `u64`, so distinct
//! `(model_id, item_id)` pairs never share an encoding.

use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read as _, Write as _};
use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use ring::aead::{AES_256_GCM, Aad, LessSafeKey, NONCE_LEN, Nonce, UnboundKey};
use ring::rand::{SecureRandom as _, SystemRandom};
use rustix::fs::{Mode, OFlags};
use rustix::io::Errno;

/// Envelope format version; bound into the associated data.
const VERSION: u8 = 1;
/// Domain separation for the associated data.
const DOMAIN: &[u8] = b"local-ai/reasoning-envelope";
/// Raw AES-256 key length.
const KEY_LEN: usize = 32;
/// AES-GCM authentication tag length.
const TAG_LEN: usize = 16;
/// Version byte plus nonce.
const HEADER_LEN: usize = 1 + NONCE_LEN;
/// Key file contents: this magic line, then the raw key bytes.
const KEY_MAGIC: &[u8] = b"local-ai reasoning key v1\n";
const KEY_FILE_LEN: usize = KEY_MAGIC.len() + KEY_LEN;
/// How often to retry when the key file vanishes between racing openers.
const OPEN_ATTEMPTS: usize = 4;

/// Largest reasoning text, in bytes, that may be sealed or unsealed.
pub const MAX_REASONING_BYTES: usize = 16 * 1024 * 1024;
/// Largest envelope, in characters, accepted by [`ReasoningCipher::unseal`].
const MAX_ENVELOPE_CHARS: usize = (HEADER_LEN + MAX_REASONING_BYTES + TAG_LEN).div_ceil(3) * 4;

/// Errors from loading the key or sealing and opening envelopes.
///
/// No variant carries key material or reasoning text.
#[derive(Debug)]
#[non_exhaustive]
pub enum ReasoningCryptoError {
    /// The key path has no file name component.
    InvalidKeyPath(PathBuf),
    /// The key path is a symlink, which is refused.
    SymlinkKeyPath(PathBuf),
    /// The key path exists but is not a regular file.
    NotRegularFile(PathBuf),
    /// The key file grants group or other access.
    InsecureKeyPermissions { path: PathBuf, mode: u32 },
    /// The key file is owned by another user.
    ForeignKeyOwner(PathBuf),
    /// The key file does not hold a well-formed key.
    InvalidKeyFile(PathBuf),
    /// The key file kept appearing and vanishing while opening it.
    KeyFileUnstable(PathBuf),
    /// A filesystem operation on the key file failed.
    Io {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    /// The operating system's random number generator failed.
    Randomness,
    /// The AEAD key could not be set up or sealing failed.
    Cipher,
    /// The reasoning text exceeds [`MAX_REASONING_BYTES`].
    PayloadTooLarge,
    /// The envelope is not valid base64url or is too short or too long.
    MalformedEnvelope,
    /// The envelope declares a format version this build cannot open.
    UnsupportedVersion(u8),
    /// The envelope failed authentication: altered, sealed with another key,
    /// or sealed for another model or item.
    Authentication,
}

impl fmt::Display for ReasoningCryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKeyPath(path) => {
                write!(f, "reasoning key path {} has no file name", path.display())
            }
            Self::SymlinkKeyPath(path) => {
                write!(f, "reasoning key path {} is a symlink", path.display())
            }
            Self::NotRegularFile(path) => {
                write!(f, "reasoning key path {} is not a regular file", path.display())
            }
            Self::InsecureKeyPermissions { path, mode } => write!(
                f,
                "reasoning key file {} has mode {:04o}; it must be owner-only (0600)",
                path.display(),
                mode & 0o7777
            ),
            Self::ForeignKeyOwner(path) => write!(
                f,
                "reasoning key file {} is not owned by the current user",
                path.display()
            ),
            Self::InvalidKeyFile(path) => write!(
                f,
                "reasoning key file {} does not contain a valid key",
                path.display()
            ),
            Self::KeyFileUnstable(path) => write!(
                f,
                "reasoning key file {} kept changing while being opened",
                path.display()
            ),
            Self::Io {
                action,
                path,
                source,
            } => write!(
                f,
                "failed to {action} reasoning key file {}: {source}",
                path.display()
            ),
            Self::Randomness => f.write_str("the system random number generator failed"),
            Self::Cipher => f.write_str("reasoning encryption failed"),
            Self::PayloadTooLarge => write!(
                f,
                "reasoning text exceeds the {MAX_REASONING_BYTES}-byte limit"
            ),
            Self::MalformedEnvelope => f.write_str("encrypted reasoning envelope is malformed"),
            Self::UnsupportedVersion(version) => write!(
                f,
                "encrypted reasoning envelope version {version} is not supported"
            ),
            Self::Authentication => f.write_str(
                "encrypted reasoning envelope failed authentication or belongs to another model, item or key",
            ),
        }
    }
}

impl std::error::Error for ReasoningCryptoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

type Result<T> = std::result::Result<T, ReasoningCryptoError>;

/// Seals and opens reasoning envelopes with a persistent on-disk key.
pub struct ReasoningCipher {
    key: LessSafeKey,
    rng: SystemRandom,
}

impl fmt::Debug for ReasoningCipher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReasoningCipher")
            .field("algorithm", &"AES-256-GCM")
            .field("key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl ReasoningCipher {
    /// Load the key at `path`, creating it owner-only with 32 fresh random
    /// bytes if it does not exist yet. The parent directory must exist.
    ///
    /// Fails, without modifying anything, if `path` is a symlink, is not a
    /// regular file, is not owned by the current user, grants group or other
    /// access, or does not hold a key in this module's format.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let rng = SystemRandom::new();
        for _ in 0..OPEN_ATTEMPTS {
            if let Some(key) = read_key(path)? {
                return Ok(Self { key, rng });
            }
            if let Some(key) = create_key(path, &rng)? {
                return Ok(Self { key, rng });
            }
        }
        Err(ReasoningCryptoError::KeyFileUnstable(path.to_path_buf()))
    }

    /// Encrypt `reasoning` for `item_id` produced by `model_id`, returning an
    /// opaque base64url envelope. Every call uses a fresh random nonce, so
    /// sealing the same text twice yields different envelopes.
    pub fn seal(&self, model_id: &str, item_id: &str, reasoning: &str) -> Result<String> {
        if reasoning.len() > MAX_REASONING_BYTES {
            return Err(ReasoningCryptoError::PayloadTooLarge);
        }
        let mut nonce = [0_u8; NONCE_LEN];
        self.rng
            .fill(&mut nonce)
            .map_err(|_| ReasoningCryptoError::Randomness)?;
        let aad = associated_data(VERSION, model_id, item_id);
        let mut sealed = Vec::with_capacity(HEADER_LEN + reasoning.len() + TAG_LEN);
        sealed.push(VERSION);
        sealed.extend_from_slice(&nonce);
        sealed.extend_from_slice(reasoning.as_bytes());
        let tag = self
            .key
            .seal_in_place_separate_tag(
                Nonce::assume_unique_for_key(nonce),
                Aad::from(aad.as_slice()),
                &mut sealed[HEADER_LEN..],
            )
            .map_err(|_| ReasoningCryptoError::Cipher)?;
        sealed.extend_from_slice(tag.as_ref());
        Ok(URL_SAFE_NO_PAD.encode(&sealed))
    }

    /// Decrypt an envelope from [`seal`](Self::seal), verifying that it is
    /// unaltered, was sealed with this key, and was sealed for exactly this
    /// `model_id` and `item_id`.
    pub fn unseal(&self, model_id: &str, item_id: &str, envelope: &str) -> Result<String> {
        if envelope.len() > MAX_ENVELOPE_CHARS {
            return Err(ReasoningCryptoError::MalformedEnvelope);
        }
        let mut sealed = URL_SAFE_NO_PAD
            .decode(envelope)
            .map_err(|_| ReasoningCryptoError::MalformedEnvelope)?;
        let version = *sealed
            .first()
            .ok_or(ReasoningCryptoError::MalformedEnvelope)?;
        if version != VERSION {
            return Err(ReasoningCryptoError::UnsupportedVersion(version));
        }
        if sealed.len() < HEADER_LEN + TAG_LEN {
            return Err(ReasoningCryptoError::MalformedEnvelope);
        }
        let mut nonce = [0_u8; NONCE_LEN];
        nonce.copy_from_slice(&sealed[1..HEADER_LEN]);
        let aad = associated_data(version, model_id, item_id);
        let plaintext = self
            .key
            .open_in_place(
                Nonce::assume_unique_for_key(nonce),
                Aad::from(aad.as_slice()),
                &mut sealed[HEADER_LEN..],
            )
            .map_err(|_| ReasoningCryptoError::Authentication)?;
        // Only text from `seal` authenticates, so this cannot fail in practice.
        String::from_utf8(plaintext.to_vec()).map_err(|_| ReasoningCryptoError::Authentication)
    }
}

/// The associated data binding an envelope to its version, model and item.
fn associated_data(version: u8, model_id: &str, item_id: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(DOMAIN.len() + 1 + 16 + model_id.len() + item_id.len());
    aad.extend_from_slice(DOMAIN);
    aad.push(version);
    for part in [model_id, item_id] {
        aad.extend_from_slice(&(part.len() as u64).to_be_bytes());
        aad.extend_from_slice(part.as_bytes());
    }
    aad
}

fn io_error(
    action: &'static str,
    path: &Path,
    source: impl Into<io::Error>,
) -> ReasoningCryptoError {
    ReasoningCryptoError::Io {
        action,
        path: path.to_path_buf(),
        source: source.into(),
    }
}

/// Build the AEAD key from raw bytes, then wipe them (best effort).
fn aead_key(raw: &mut [u8; KEY_LEN]) -> Result<LessSafeKey> {
    let key = UnboundKey::new(&AES_256_GCM, raw).map_err(|_| ReasoningCryptoError::Cipher);
    raw.fill(0);
    let _ = std::hint::black_box(&raw);
    key.map(LessSafeKey::new)
}

/// Read an existing key; `Ok(None)` if there is no file at `path`.
fn read_key(path: &Path) -> Result<Option<LessSafeKey>> {
    // O_NOFOLLOW refuses a symlink as the final component; O_NONBLOCK keeps
    // a FIFO planted at the path from blocking the open.
    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
    let fd = match rustix::fs::open(path, flags, Mode::empty()) {
        Ok(fd) => fd,
        Err(errno) if errno == Errno::NOENT => return Ok(None),
        Err(errno) if errno == Errno::LOOP => {
            return Err(ReasoningCryptoError::SymlinkKeyPath(path.to_path_buf()));
        }
        Err(errno) => return Err(io_error("open", path, errno)),
    };
    let file = File::from(fd);
    let metadata = file
        .metadata()
        .map_err(|error| io_error("inspect", path, error))?;
    if !metadata.file_type().is_file() {
        return Err(ReasoningCryptoError::NotRegularFile(path.to_path_buf()));
    }
    if metadata.uid() != rustix::process::geteuid().as_raw() {
        return Err(ReasoningCryptoError::ForeignKeyOwner(path.to_path_buf()));
    }
    let mode = metadata.permissions().mode();
    if mode & 0o077 != 0 {
        return Err(ReasoningCryptoError::InsecureKeyPermissions {
            path: path.to_path_buf(),
            mode,
        });
    }
    let mut contents = Vec::with_capacity(KEY_FILE_LEN + 1);
    let read = file
        .take(KEY_FILE_LEN as u64 + 1)
        .read_to_end(&mut contents)
        .map_err(|error| io_error("read", path, error));
    let mut raw = [0_u8; KEY_LEN];
    let valid = read.is_ok() && contents.len() == KEY_FILE_LEN && contents.starts_with(KEY_MAGIC);
    if valid {
        raw.copy_from_slice(&contents[KEY_MAGIC.len()..]);
    }
    contents.fill(0);
    let _ = std::hint::black_box(&contents);
    read?;
    if !valid {
        return Err(ReasoningCryptoError::InvalidKeyFile(path.to_path_buf()));
    }
    aead_key(&mut raw).map(Some)
}

/// Removes the temporary key file when dropped.
struct TempFile<'a>(&'a Path);

impl Drop for TempFile<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.0);
    }
}

/// Create a fresh key at `path` unless something already exists there, in
/// which case `Ok(None)` tells the caller to read that instead.
fn create_key(path: &Path, rng: &SystemRandom) -> Result<Option<LessSafeKey>> {
    let name = path
        .file_name()
        .ok_or_else(|| ReasoningCryptoError::InvalidKeyPath(path.to_path_buf()))?;
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let mut suffix = [0_u8; 8];
    rng.fill(&mut suffix)
        .map_err(|_| ReasoningCryptoError::Randomness)?;
    let mut temp_name = OsString::from(".");
    temp_name.push(name);
    temp_name.push(format!(".{:016x}.tmp", u64::from_le_bytes(suffix)));
    let temp_path = dir.join(temp_name);

    let mut contents = [0_u8; KEY_FILE_LEN];
    contents[..KEY_MAGIC.len()].copy_from_slice(KEY_MAGIC);
    let mut raw = [0_u8; KEY_LEN];
    rng.fill(&mut raw)
        .map_err(|_| ReasoningCryptoError::Randomness)?;
    contents[KEY_MAGIC.len()..].copy_from_slice(&raw);

    let written = write_temp_key(&temp_path, &contents);
    contents.fill(0);
    let _ = std::hint::black_box(&contents);
    let guard = match written {
        Ok(()) => TempFile(&temp_path),
        Err(error) => {
            raw.fill(0);
            return Err(error);
        }
    };
    // link(2) never replaces an existing entry, so this is an atomic
    // create-if-absent of a file that is already complete and synced.
    let linked = fs::hard_link(&temp_path, path);
    drop(guard);
    match linked {
        Ok(()) => {
            // Make the new directory entry durable; failure here is harmless
            // beyond a key that may need regenerating after a power loss.
            if let Ok(dir) = File::open(dir) {
                let _ = dir.sync_all();
            }
            aead_key(&mut raw).map(Some)
        }
        Err(error) => {
            raw.fill(0);
            if error.kind() == io::ErrorKind::AlreadyExists {
                Ok(None)
            } else {
                Err(io_error("create", path, error))
            }
        }
    }
}

/// Write `contents` to a new owner-only file at `temp_path` and sync it.
fn write_temp_key(temp_path: &Path, contents: &[u8]) -> Result<()> {
    let flags = OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let fd = rustix::fs::open(temp_path, flags, Mode::RUSR | Mode::WUSR)
        .map_err(|errno| io_error("create", temp_path, errno))?;
    let guard = TempFile(temp_path);
    let mut file = File::from(fd);
    // The umask can only clear bits; pin the mode to exactly 0600.
    file.set_permissions(fs::Permissions::from_mode(0o600))
        .and_then(|()| file.write_all(contents))
        .and_then(|()| file.sync_all())
        .map_err(|error| io_error("write", temp_path, error))?;
    std::mem::forget(guard);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::fs;
    use std::os::unix::fs::{DirBuilderExt as _, PermissionsExt as _};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{MAX_REASONING_BYTES, ReasoningCipher, ReasoningCryptoError};

    const MODEL: &str = "bonsai-2-27b";
    const ITEM: &str = "rs_0123456789abcdef";

    /// An owner-only directory under the system temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos());
            let dir = std::env::temp_dir().join(format!(
                "local-ai-reasoning-key-test-{}-{}-{nanos}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            fs::DirBuilder::new()
                .mode(0o700)
                .create(&dir)
                .expect("scratch dir is created");
            Self(dir)
        }

        fn key(&self) -> PathBuf {
            self.0.join("reasoning.key")
        }

        fn entries(&self) -> Vec<String> {
            let mut names: Vec<String> = fs::read_dir(&self.0)
                .expect("scratch dir lists")
                .map(|entry| {
                    entry
                        .expect("entry reads")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn open(path: &Path) -> ReasoningCipher {
        ReasoningCipher::open(path).expect("cipher opens")
    }

    fn write_mode(path: &Path, contents: &[u8], mode: u32) {
        fs::write(path, contents).expect("file is written");
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).expect("mode is set");
    }

    #[test]
    fn roundtrips_after_reopening_and_persists_only_an_owner_only_key() {
        let scratch = Scratch::new();
        let reasoning = "First, compare both options; the second is cheaper.";
        let envelope = open(&scratch.key())
            .seal(MODEL, ITEM, reasoning)
            .expect("seals");
        assert!(!envelope.contains(reasoning));
        assert!(
            envelope
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        );

        let metadata = fs::symlink_metadata(scratch.key()).expect("key exists");
        assert!(metadata.file_type().is_file());
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        assert_eq!(scratch.entries(), ["reasoning.key"]);
        let key_bytes = fs::read(scratch.key()).expect("key reads");
        assert!(
            !key_bytes
                .windows(reasoning.len())
                .any(|window| window == reasoning.as_bytes())
        );

        let reopened = open(&scratch.key());
        assert_eq!(fs::read(scratch.key()).expect("key reads"), key_bytes);
        assert_eq!(
            reopened.unseal(MODEL, ITEM, &envelope).expect("unseals"),
            reasoning
        );
    }

    #[test]
    fn unicode_and_empty_payloads_roundtrip() {
        let scratch = Scratch::new();
        let cipher = open(&scratch.key());
        for text in ["", "😀 推理 — naïve ∑ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢\n\t\0end", "x"] {
            let envelope = cipher.seal(MODEL, ITEM, text).expect("seals");
            assert_eq!(
                cipher.unseal(MODEL, ITEM, &envelope).expect("unseals"),
                text
            );
        }
        let envelope = cipher.seal("", "", "empty ids").expect("seals");
        assert_eq!(
            cipher.unseal("", "", &envelope).expect("unseals"),
            "empty ids"
        );
        let envelope = cipher.seal("modèle", "élément", "ids").expect("seals");
        assert_eq!(
            cipher
                .unseal("modèle", "élément", &envelope)
                .expect("unseals"),
            "ids"
        );
    }

    #[test]
    fn same_plaintext_seals_to_unique_nonces_and_ciphertexts() {
        let scratch = Scratch::new();
        let cipher = open(&scratch.key());
        let envelopes: Vec<String> = (0..256)
            .map(|_| cipher.seal(MODEL, ITEM, "identical").expect("seals"))
            .collect();
        // The first 16 characters encode the version byte and 11 nonce bytes.
        let mut prefixes: Vec<&[u8]> = envelopes
            .iter()
            .map(|envelope| &envelope.as_bytes()[..16])
            .collect();
        prefixes.sort_unstable();
        prefixes.dedup();
        assert_eq!(prefixes.len(), envelopes.len());
        let mut unique = envelopes.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), envelopes.len());
        for envelope in &envelopes {
            assert_eq!(
                cipher.unseal(MODEL, ITEM, envelope).expect("unseals"),
                "identical"
            );
        }
    }

    #[test]
    fn tampered_envelopes_fail() {
        let scratch = Scratch::new();
        let cipher = open(&scratch.key());
        let secret = "do not leak this reasoning";
        let envelope = cipher.seal(MODEL, ITEM, secret).expect("seals");
        for index in 0..envelope.len() {
            let mut bytes = envelope.clone().into_bytes();
            bytes[index] = if bytes[index] == b'A' { b'B' } else { b'A' };
            let altered = String::from_utf8(bytes).expect("ASCII stays UTF-8");
            let error = cipher
                .unseal(MODEL, ITEM, &altered)
                .expect_err("altered envelope fails");
            assert!(!error.to_string().contains(secret));
        }
        let truncated = |len: usize| envelope.chars().take(len).collect::<String>();
        for altered in [
            String::new(),
            truncated(envelope.len() - 1),
            truncated(envelope.len() - 4),
            truncated(20),
            format!("{envelope}A"),
            format!("{envelope}AAAA"),
            format!("{envelope}="),
            envelope.replace('-', "+").replace('_', "/") + "!",
        ] {
            assert!(cipher.unseal(MODEL, ITEM, &altered).is_err());
        }
    }

    #[test]
    fn foreign_version_is_rejected() {
        let scratch = Scratch::new();
        let cipher = open(&scratch.key());
        let envelope = cipher.seal(MODEL, ITEM, "text").expect("seals");
        // Version 1 encodes as a leading 'A'; 'B' turns the byte into 5.
        assert!(envelope.starts_with('A'));
        let altered: String = std::iter::once('B')
            .chain(envelope.chars().skip(1))
            .collect();
        assert!(matches!(
            cipher.unseal(MODEL, ITEM, &altered),
            Err(ReasoningCryptoError::UnsupportedVersion(5))
        ));
    }

    #[test]
    fn model_and_item_are_bound() {
        let scratch = Scratch::new();
        let cipher = open(&scratch.key());
        let envelope = cipher.seal(MODEL, ITEM, "bound").expect("seals");
        for (model, item) in [
            ("other-model", ITEM),
            (MODEL, "rs_other"),
            (ITEM, MODEL),
            ("", ""),
            (format!("{MODEL} ").as_str(), ITEM),
        ] {
            assert!(matches!(
                cipher.unseal(model, item, &envelope),
                Err(ReasoningCryptoError::Authentication)
            ));
        }
        // Length prefixes keep a shifted boundary from colliding.
        let envelope = cipher.seal("ab", "c", "split").expect("seals");
        assert!(matches!(
            cipher.unseal("a", "bc", &envelope),
            Err(ReasoningCryptoError::Authentication)
        ));
    }

    #[test]
    fn another_key_cannot_open() {
        let scratch = Scratch::new();
        let first = open(&scratch.key());
        let second = open(&scratch.0.join("other.key"));
        let envelope = first.seal(MODEL, ITEM, "keyed").expect("seals");
        assert!(matches!(
            second.unseal(MODEL, ITEM, &envelope),
            Err(ReasoningCryptoError::Authentication)
        ));
        assert_ne!(
            fs::read(scratch.key()).expect("key reads"),
            fs::read(scratch.0.join("other.key")).expect("key reads")
        );
    }

    #[test]
    fn oversized_payload_is_refused() {
        let scratch = Scratch::new();
        let cipher = open(&scratch.key());
        let huge = "a".repeat(MAX_REASONING_BYTES + 1);
        assert!(matches!(
            cipher.seal(MODEL, ITEM, &huge),
            Err(ReasoningCryptoError::PayloadTooLarge)
        ));
    }

    #[test]
    fn concurrent_creation_converges_on_one_key() {
        let scratch = Scratch::new();
        let path = Arc::new(scratch.key());
        let barrier = Arc::new(Barrier::new(16));
        let ciphers: Vec<ReasoningCipher> = (0..16)
            .map(|_| {
                let path = Arc::clone(&path);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    ReasoningCipher::open(path.as_path())
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .expect("thread finishes")
                    .expect("cipher opens")
            })
            .collect();
        assert_eq!(scratch.entries(), ["reasoning.key"]);
        let envelope = ciphers[0].seal(MODEL, ITEM, "shared").expect("seals");
        for cipher in &ciphers {
            assert_eq!(
                cipher.unseal(MODEL, ITEM, &envelope).expect("unseals"),
                "shared"
            );
        }
    }

    #[test]
    fn insecure_permissions_are_refused_without_modification() {
        let scratch = Scratch::new();
        open(&scratch.key());
        let original = fs::read(scratch.key()).expect("key reads");
        for mode in [0o640, 0o604, 0o644, 0o660, 0o666, 0o610] {
            fs::set_permissions(scratch.key(), fs::Permissions::from_mode(mode))
                .expect("mode is set");
            assert!(matches!(
                ReasoningCipher::open(scratch.key()),
                Err(ReasoningCryptoError::InsecureKeyPermissions { .. })
            ));
            assert_eq!(fs::read(scratch.key()).expect("key reads"), original);
        }
        fs::set_permissions(scratch.key(), fs::Permissions::from_mode(0o400)).expect("mode is set");
        open(&scratch.key());
    }

    #[test]
    fn malformed_key_files_are_refused_without_modification() {
        let scratch = Scratch::new();
        let mut valid = b"local-ai reasoning key v1\n".to_vec();
        valid.extend_from_slice(&[7; 32]);
        let mut wrong_magic = valid.clone();
        wrong_magic[0] = b'L';
        let mut long = valid.clone();
        long.push(0);
        for contents in [
            Vec::new(),
            vec![0; 32],
            valid[..valid.len() - 1].to_vec(),
            wrong_magic,
            long,
        ] {
            write_mode(&scratch.key(), &contents, 0o600);
            assert!(matches!(
                ReasoningCipher::open(scratch.key()),
                Err(ReasoningCryptoError::InvalidKeyFile(_))
            ));
            assert_eq!(fs::read(scratch.key()).expect("key reads"), contents);
        }
        // A well-formed operator-provided key is accepted as is.
        write_mode(&scratch.key(), &valid, 0o600);
        let cipher = open(&scratch.key());
        let envelope = cipher.seal(MODEL, ITEM, "provided").expect("seals");
        assert_eq!(
            cipher.unseal(MODEL, ITEM, &envelope).expect("unseals"),
            "provided"
        );
        assert_eq!(fs::read(scratch.key()).expect("key reads"), valid);
    }

    #[test]
    fn symlinks_and_non_files_are_refused() {
        let scratch = Scratch::new();
        let target = scratch.0.join("target.key");
        open(&target);
        let link = scratch.0.join("link.key");
        std::os::unix::fs::symlink(&target, &link).expect("symlink is created");
        assert!(matches!(
            ReasoningCipher::open(&link),
            Err(ReasoningCryptoError::SymlinkKeyPath(_))
        ));

        let dangling = scratch.0.join("dangling.key");
        let missing = scratch.0.join("missing.key");
        std::os::unix::fs::symlink(&missing, &dangling).expect("symlink is created");
        assert!(matches!(
            ReasoningCipher::open(&dangling),
            Err(ReasoningCryptoError::SymlinkKeyPath(_))
        ));
        assert!(fs::symlink_metadata(&missing).is_err());

        let dir = scratch.0.join("dir.key");
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&dir)
            .expect("dir is created");
        assert!(matches!(
            ReasoningCipher::open(&dir),
            Err(ReasoningCryptoError::NotRegularFile(_))
        ));

        assert!(matches!(
            ReasoningCipher::open(scratch.0.join("absent").join("reasoning.key")),
            Err(ReasoningCryptoError::Io { .. })
        ));
        assert_eq!(
            scratch.entries(),
            ["dangling.key", "dir.key", "link.key", "target.key"]
        );
    }

    #[test]
    fn diagnostics_do_not_expose_secrets() {
        use std::fmt::Write as _;

        let scratch = Scratch::new();
        let cipher = open(&scratch.key());
        let debug = format!("{cipher:?}");
        assert!(debug.contains("redacted"));
        let key = fs::read(scratch.key()).expect("key reads");
        let key_hex = key[key.len() - 32..]
            .iter()
            .fold(String::new(), |mut out, byte| {
                write!(out, "{byte:02x}").expect("write to string");
                out
            });
        assert!(!debug.contains(&key_hex));
        let error = cipher
            .unseal(
                "other",
                ITEM,
                &cipher.seal(MODEL, ITEM, "private chain").expect("seals"),
            )
            .expect_err("mismatch fails");
        assert!(!format!("{error} {error:?}").contains("private chain"));
    }
}
