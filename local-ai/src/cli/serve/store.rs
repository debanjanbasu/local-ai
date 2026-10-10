//! The opt-in Responses store behind `--response-store DIR`.
//!
//! One JSON file per stored Response, named by its ID. Each record holds the
//! final Response object and the *resolved* input it was generated from: the
//! prior response's input and output followed by this request's own items.
//! Replaying `previous_response_id` therefore reads exactly one file and never
//! walks a chain, and deleting an earlier response does not break a later one
//! that already carries its history.
//!
//! The records hold user prompts, tool results and the model's raw reasoning
//! in plain text. That is protected local data, not encrypted data: the
//! directory is created owner-only (`0700`) and every record is written
//! owner-only (`0600`), and nothing here ever claims more than that.
//!
//! Writes are atomic: a record is written to a temporary file in the same
//! directory, flushed to disk and renamed over its final name, so a reader
//! sees either the whole record or none of it, and a crash mid-write leaves
//! only an ignored temporary file. Opening a second server must not remove
//! the first server's in-flight writes or unrelated files in the directory.
//!
//! A background Response is written more than once (queued, then terminal),
//! so its writer holds a [`ResponseLease`]: an exclusive advisory lock on an
//! owner-only `.{id}.lock` sidecar, taken before the first save and released
//! after the terminal one. Sidecars are never unlinked, so every process
//! always locks the same inode. When a store opens, a background record still
//! `queued` or `in_progress` whose lease is free has lost its writer to a
//! crash or restart; it is marked `failed` rather than resumed. A record whose
//! lease is held belongs to a live server sharing the directory and is left
//! alone.

use std::fs;
use std::io::{ErrorKind, Write as _};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::response::new_id;

/// Version of the on-disk record layout, checked on read.
const RECORD_VERSION: u64 = 1;

/// Suffix of a record being written. A leading dot keeps it apart from IDs.
const TEMPORARY_SUFFIX: &str = ".tmp";

/// Longest accepted ID after `resp_`; this server's own are 32 hex digits.
const MAX_ID_TAIL: usize = 64;

/// Suffix of a response's lease sidecar, `.{id}.lock`.
const LOCK_SUFFIX: &str = ".lock";

/// The error a recovered background Response reports.
const INTERRUPTED_MESSAGE: &str = "the server stopped before this background response finished; \
     interrupted execution is not resumed";

#[derive(Debug)]
pub(super) struct ResponseStore {
    dir: PathBuf,
}

/// Exclusive ownership of one stored response's writes, across processes.
///
/// Held for as long as this value lives; dropping it closes the sidecar and
/// releases the lock. The sidecar itself stays, so its inode never changes.
#[derive(Debug)]
pub(super) struct ResponseLease {
    _file: fs::File,
}

/// A stored Response and the resolved input it was generated from.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StoredResponse {
    pub(super) response: Value,
    pub(super) input_items: Vec<Value>,
}

impl StoredResponse {
    /// The history a follow-up turn continues: every input item, then every
    /// output item, in order.
    pub(super) fn history(&self) -> Vec<Value> {
        let mut items = self.input_items.clone();
        if let Some(output) = self.response.get("output").and_then(Value::as_array) {
            items.extend(output.iter().filter(|item| !item.is_null()).cloned());
        }
        items
    }
}

/// Whether `id` can name a stored response.
///
/// This is the traversal guard: only `resp_` followed by ASCII letters and
/// digits is ever turned into a path, so `..`, separators, percent escapes and
/// NUL never reach the filesystem.
pub(super) fn valid_id(id: &str) -> bool {
    id.strip_prefix("resp_").is_some_and(|tail| {
        !tail.is_empty()
            && tail.len() <= MAX_ID_TAIL
            && tail.bytes().all(|byte| byte.is_ascii_alphanumeric())
    })
}

impl ResponseStore {
    /// Open the store at `dir`, creating it owner-only if needed.
    pub(super) fn open(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt as _;
            builder.mode(0o700);
        }
        builder.create(&dir)?;
        if !fs::metadata(&dir)?.is_dir() {
            return Err(std::io::Error::new(
                ErrorKind::InvalidInput,
                format!("{} is not a directory", dir.display()),
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            if fs::metadata(&dir)?.permissions().mode() & 0o077 != 0 {
                return Err(std::io::Error::new(
                    ErrorKind::PermissionDenied,
                    "response store directory must be owner-only (mode 0700)",
                ));
            }
        }
        let store = Self { dir };
        store.recover_interrupted()?;
        Ok(store)
    }

    fn path(&self, id: &str) -> Option<PathBuf> {
        valid_id(id).then(|| self.dir.join(format!("{id}.json")))
    }

    /// Take the write lease for response `id` without waiting.
    ///
    /// `Ok(None)` when another writer, in this process or another, holds it.
    /// An `id` that could not name a response is refused, as is a sidecar that
    /// is a symlink, not a regular file, or not owner-only.
    pub(super) fn lease(&self, id: &str) -> std::io::Result<Option<ResponseLease>> {
        if !valid_id(id) {
            return Err(std::io::Error::new(
                ErrorKind::InvalidInput,
                format!("invalid response ID {id:?}"),
            ));
        }
        let path = self.dir.join(format!(".{id}{LOCK_SUFFIX}"));
        let file = open_lock(&path)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(ResponseLease { _file: file })),
            Err(fs::TryLockError::WouldBlock) => Ok(None),
            Err(fs::TryLockError::Error(error)) => Err(error),
        }
    }

    /// Mark every background response left `queued` or `in_progress` by a
    /// writer that no longer holds its lease as `failed`.
    ///
    /// Records that cannot be read are skipped: they are not this pass's to
    /// judge, and `load` reports them to whoever asks for them.
    fn recover_interrupted(&self) -> std::io::Result<()> {
        for entry in fs::read_dir(&self.dir)? {
            let name = entry?.file_name();
            let Some(id) = name.to_str().and_then(|name| name.strip_suffix(".json")) else {
                continue;
            };
            if !valid_id(id) || !matches!(self.load(id), Ok(Some(stored)) if interrupted(&stored)) {
                continue;
            }
            let Some(_lease) = self.lease(id)? else {
                continue;
            };
            // Re-read under the lease: the writer may have finished between
            // the first read and taking the lock.
            let Ok(Some(mut stored)) = self.load(id) else {
                continue;
            };
            if !interrupted(&stored) {
                continue;
            }
            if let Some(response) = stored.response.as_object_mut() {
                response.insert("status".into(), json!("failed"));
                response.insert("completed_at".into(), Value::Null);
                response.insert(
                    "error".into(),
                    json!({"code":"server_error","message":INTERRUPTED_MESSAGE}),
                );
            }
            self.save(&stored.response, &stored.input_items)?;
        }
        Ok(())
    }

    /// Persist `response` and its resolved input, durably, before returning.
    pub(super) fn save(&self, response: &Value, input_items: &[Value]) -> std::io::Result<()> {
        let id = response.get("id").and_then(Value::as_str).unwrap_or("");
        let path = self.path(id).ok_or_else(|| {
            std::io::Error::new(
                ErrorKind::InvalidInput,
                format!("invalid response ID {id:?}"),
            )
        })?;
        let record = json!({
            "object":"local_ai.stored_response",
            "version":RECORD_VERSION,
            "response":response,
            "input_items":input_items,
        });
        let bytes = serde_json::to_vec(&record).map_err(std::io::Error::other)?;
        let temporary = self
            .dir
            .join(format!(".{id}.{}{TEMPORARY_SUFFIX}", new_id("")));
        let written = write_new(&temporary, &bytes).and_then(|()| fs::rename(&temporary, &path));
        if let Err(error) = written {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        sync_dir(&self.dir);
        Ok(())
    }

    /// The stored response `id`; `None` when it is not stored, was deleted, or
    /// `id` could not name one.
    pub(super) fn load(&self, id: &str) -> std::io::Result<Option<StoredResponse>> {
        let Some(path) = self.path(id) else {
            return Ok(None);
        };
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut record: Value = serde_json::from_slice(&bytes).map_err(|error| {
            std::io::Error::new(
                ErrorKind::InvalidData,
                format!("stored response {id} is corrupt: {error}"),
            )
        })?;
        let version = record.get("version").and_then(Value::as_u64);
        let response = record.get_mut("response").map(Value::take);
        let input_items = record.get_mut("input_items").map(Value::take);
        match (version, response, input_items) {
            (Some(RECORD_VERSION), Some(response), Some(Value::Array(input_items)))
                if response.get("id").and_then(Value::as_str) == Some(id) =>
            {
                Ok(Some(StoredResponse {
                    response,
                    input_items,
                }))
            }
            _ => Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("stored response {id} is not a version {RECORD_VERSION} record"),
            )),
        }
    }

    /// Delete the stored response `id`, reporting whether there was one.
    pub(super) fn delete(&self, id: &str) -> std::io::Result<bool> {
        let Some(path) = self.path(id) else {
            return Ok(false);
        };
        match fs::remove_file(&path) {
            Ok(()) => {
                sync_dir(&self.dir);
                Ok(true)
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }
}

/// Create `path` owner-only, write `bytes` and flush them to the device.
fn write_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Whether `stored` is a background response its writer never finished.
fn interrupted(stored: &StoredResponse) -> bool {
    stored.response.get("background").and_then(Value::as_bool) == Some(true)
        && matches!(
            stored.response.get("status").and_then(Value::as_str),
            Some("queued" | "in_progress")
        )
}

/// Open, creating owner-only if needed, the lease sidecar at `path`.
///
/// Never truncates or unlinks it, so concurrent openers share one inode. On
/// Unix a symlink as the final component is refused rather than followed, a
/// FIFO cannot block the open, and the result must be an owner-only regular
/// file belonging to this user.
#[cfg(unix)]
fn open_lock(path: &Path) -> std::io::Result<fs::File> {
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

    use rustix::fs::{Mode, OFlags};

    let flags =
        OFlags::RDWR | OFlags::CREATE | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
    let fd = rustix::fs::open(path, flags, Mode::RUSR | Mode::WUSR).map_err(|errno| {
        std::io::Error::new(
            std::io::Error::from(errno).kind(),
            format!("cannot open response lease {}: {errno}", path.display()),
        )
    })?;
    let file = fs::File::from(fd);
    let metadata = file.metadata()?;
    let refused = |why: &str| {
        Err(std::io::Error::new(
            ErrorKind::PermissionDenied,
            format!("response lease {} {why}", path.display()),
        ))
    };
    if !metadata.file_type().is_file() {
        return refused("is not a regular file");
    }
    if metadata.uid() != rustix::process::geteuid().as_raw() {
        return refused("belongs to another user");
    }
    if metadata.permissions().mode() & 0o077 != 0 {
        return refused("must be owner-only (mode 0600)");
    }
    Ok(file)
}

#[cfg(not(unix))]
fn open_lock(path: &Path) -> std::io::Result<fs::File> {
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

/// Make a rename or unlink in `dir` durable, where the platform allows it.
///
/// Best effort: not every platform can sync a directory handle, and the record
/// itself is already on disk, so a failure here only widens the window in which
/// a power cut could undo the last rename.
fn sync_dir(dir: &Path) {
    if let Ok(handle) = fs::File::open(dir) {
        let _ = handle.sync_all();
    }
}

/// A Responses `input_items` page, per `GET /v1/responses/{id}/input_items`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ItemPage {
    pub(super) limit: usize,
    pub(super) descending: bool,
    pub(super) after: Option<String>,
}

/// Why a page could not be produced.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum PageError {
    /// A malformed query: 400.
    Invalid(String),
    /// `after` names no item of this response: 404, as the contract says.
    AfterNotFound(String),
}

impl ItemPage {
    /// Parse the query string of an `input_items` request.
    ///
    /// `include` is refused when non-empty because none of its values has
    /// anything to add here; unknown parameters are ignored, as HTTP clients
    /// and client libraries add their own.
    pub(super) fn parse(query: Option<&str>) -> Result<Self, PageError> {
        let mut page = Self {
            limit: 20,
            descending: true,
            after: None,
        };
        for (key, value) in query_pairs(query) {
            match key.as_str() {
                "limit" => {
                    page.limit = value
                        .parse()
                        .ok()
                        .filter(|limit| (1..=100).contains(limit))
                        .ok_or_else(|| {
                            PageError::Invalid(format!(
                                "limit must be an integer from 1 to 100, not {value:?}"
                            ))
                        })?;
                }
                "order" => {
                    page.descending = match value.as_str() {
                        "asc" => false,
                        "desc" => true,
                        other => {
                            return Err(PageError::Invalid(format!(
                                "order must be \"asc\" or \"desc\", not {other:?}"
                            )));
                        }
                    };
                }
                "after" => page.after = Some(value),
                "include" | "include[]" if !value.is_empty() => {
                    return Err(PageError::Invalid(format!(
                        "include {value:?} is not supported: there are no logprobs, \
                             encrypted reasoning or tool outputs to add"
                    )));
                }
                _ => {}
            }
        }
        Ok(page)
    }

    /// The `list` object for `items`, which are in input order.
    pub(super) fn list(&self, items: &[Value]) -> Result<Value, PageError> {
        let ordered: Vec<&Value> = if self.descending {
            items.iter().rev().collect()
        } else {
            items.iter().collect()
        };
        let start = match &self.after {
            None => 0,
            Some(after) => {
                ordered
                    .iter()
                    .position(|item| item.get("id").and_then(Value::as_str) == Some(after))
                    .ok_or_else(|| PageError::AfterNotFound(after.clone()))?
                    + 1
            }
        };
        let rest = ordered.get(start..).unwrap_or_default();
        let data: Vec<&Value> = rest.iter().take(self.limit).copied().collect();
        // The published ResponseItemList schema requires string cursors even
        // on an empty page; an empty string names no item.
        let id = |item: Option<&&Value>| {
            item.and_then(|item| item.get("id"))
                .cloned()
                .unwrap_or_else(|| json!(""))
        };
        Ok(json!({
            "object":"list",
            "data":data,
            "first_id":id(data.first()),
            "last_id":id(data.last()),
            "has_more":rest.len() > data.len(),
        }))
    }
}

/// Query parameters with `+` and percent escapes decoded. A pair that does not
/// decode to UTF-8 is kept with replacement characters, so it fails later
/// validation instead of vanishing.
pub(super) fn query_pairs(query: Option<&str>) -> Vec<(String, String)> {
    query
        .unwrap_or_default()
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (percent_decode(key), percent_decode(value))
        })
        .collect()
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while let Some(&byte) = bytes.get(index) {
        let hex = |offset: usize| {
            bytes
                .get(index + offset)
                .and_then(|digit| char::from(*digit).to_digit(16))
        };
        match byte {
            b'+' => out.push(b' '),
            b'%' => {
                if let (Some(high), Some(low)) = (hex(1), hex(2)) {
                    out.push((high * 16 + low) as u8);
                    index += 3;
                    continue;
                }
                out.push(byte);
            }
            _ => out.push(byte),
        }
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
