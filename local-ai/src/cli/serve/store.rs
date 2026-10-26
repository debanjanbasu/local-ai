//! The Responses store: by default in the user's data directory, pruned of
//! responses older than 30 days at startup and every hour after, or
//! `--response-store DIR`.
//!
//! One JSON file per stored Response, named by its ID. Each record holds the
//! final Response object and the *resolved* input it was generated from: the
//! prior response's input and output followed by this request's own items.
//! Replaying `previous_response_id` therefore reads exactly one file and never
//! walks a chain, and deleting an earlier response does not break a later one
//! that already carries its history. The price is disk space: every turn of a
//! conversation repeats all of the turns before it, inline images (data URLs)
//! included, so a conversation of `n` turns stores about `n²/2` turns' worth.
//!
//! The records hold user prompts, tool results and the model's raw reasoning
//! in plain text. That is protected local data, not encrypted data: the
//! directory is created owner-only (`0700`) and every record is written
//! owner-only (`0600`), and nothing here ever claims more than that.
//!
//! Writes are atomic: a record is written to a temporary file in the same
//! directory, flushed to disk and renamed over its final name, so a reader
//! sees either the whole record or none of it, and a crash mid-write leaves
//! only a temporary file, which a sweep removes once it is an hour old.
//! Opening a second server must not remove the first server's in-flight
//! writes or unrelated files in the directory.
//!
//! A background Response is written more than once (queued, then terminal),
//! so its writer holds a [`ResponseLease`]: an exclusive advisory lock on an
//! owner-only `.{id}.lock` sidecar, taken before the first save and released
//! after the terminal one. A sidecar is unlinked only by whoever holds its
//! lease, once its record is settled or gone, and [`ResponseStore::lease`]
//! checks after locking that the name still leads to the inode it locked, so
//! two processes never both hold one response's lease. A background record
//! still `queued` or `in_progress` whose lease is free has lost its writer to
//! a crash or restart. While it is pending, a durable background record also
//! keeps the request it was created from (`request`), and a server resumes it
//! from that (see [`super::background::Background::resume`]); one without it
//! (one continuing a conversation) is marked `failed` when a store opens. A
//! record whose lease is held belongs to a live server sharing the directory
//! and is left alone.
//!
//! A background Response created with `stream: true` also has an owner-only
//! `.{id}.events` journal of the events it published (see [`super::journal`]).
//! It is created before the record and deleted after it, so every streamed
//! record has one, and recovery reconciles it with the record it belongs to.
//!
//! A resumable background Response that declares hosted tools gets an empty
//! owner-only `.{id}.tools` marker, flushed to disk before its first hosted
//! call starts executing: a record without one provably never ran a hosted
//! call, so a restarted server may regenerate it without running any twice.
//! It is deleted with the record.
//!
//! A pending background record is written by its lease holder, and saving
//! one creates its sidecar if it has none, so every record that can be
//! pending has a lease sidecar. A sweep of the directory (when a store
//! opens, and on every pruning pass) therefore reads only records that have
//! one (to settle an interrupted one that cannot be resumed, or to unlink the
//! sidecar of one that has settled) and, when pruning, records not written
//! for the whole retention period; [`ResponseStore::resumable`] reads only
//! records that have one too. Every other record is never opened.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime};

use serde::Deserialize;
use serde::de::IgnoredAny;
use serde_json::{Value, json};

use super::journal::{JournalReader, JournalWriter, ended_with, terminal_events};
use super::response::new_id;

/// Version of the on-disk record layout, checked on read.
const RECORD_VERSION: u64 = 1;

/// Suffix of a record being written. A leading dot keeps it apart from IDs.
const TEMPORARY_SUFFIX: &str = ".tmp";

/// Longest accepted ID after `resp_`; this server's own are 32 hex digits.
const MAX_ID_TAIL: usize = 64;

/// Suffix of a response's lease sidecar, `.{id}.lock`.
const LOCK_SUFFIX: &str = ".lock";

/// Suffix of a streamed response's event journal, `.{id}.events`.
const JOURNAL_SUFFIX: &str = ".events";

/// Suffix of the marker a resumable background response gets, `.{id}.tools`,
/// durably before its first hosted tool call starts executing.
const TOOLS_SUFFIX: &str = ".tools";

/// The error a background Response that could not be resumed reports,
/// before why.
const INTERRUPTED_MESSAGE: &str =
    "the server stopped before this background response finished, and it could not be resumed";

/// Why a record without its request cannot be resumed.
pub(super) const NO_REQUEST: &str = "its record keeps no request to regenerate it from";

/// How long a temporary file, or a lease sidecar with neither a record nor a
/// journal, must have gone unmodified before a sweep removes it.
///
/// A save writes, flushes and renames its temporary file within seconds, and
/// a writer locks a sidecar it creates at once, so a live server's in-flight
/// file is always far younger than this, and anything older is a crash's.
const STALE_AFTER: Duration = Duration::from_hours(1);

/// How often a pruning store prunes again after [`ResponseStore::prune`].
///
/// A timer, because no event marks a record becoming old enough to go, and a
/// deadline per record would mean reading every record to learn its age,
/// which is the work a sweep avoids. An hour past 30 days is immaterial to
/// the retention, and a pass reads only the records it deletes.
const PRUNE_INTERVAL: Duration = Duration::from_hours(1);

/// How often [`ResponseStore::lease`] locks the sidecar again when it finds
/// the one it locked was unlinked meanwhile. A sidecar is unlinked only once
/// its response has settled or gone, so a second unlink in the same moment
/// is already unlikely.
const LEASE_ATTEMPTS: usize = 4;

#[derive(Debug)]
pub(super) struct ResponseStore {
    dir: PathBuf,
    /// Stops the periodic pruning [`Self::prune`] starts when this store is
    /// dropped. `None` for the handle the pruning thread itself works with.
    pruner: Option<Arc<Pruner>>,
}

/// The periodic pruning of one store.
#[derive(Debug, Default)]
struct Pruner {
    state: Mutex<PrunerState>,
    /// Wakes the pruning thread early, to stop.
    wake: Condvar,
}

#[derive(Debug, Default)]
struct PrunerState {
    started: bool,
    closed: bool,
}

impl Pruner {
    fn lock(&self) -> MutexGuard<'_, PrunerState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for ResponseStore {
    fn drop(&mut self) {
        if let Some(pruner) = &self.pruner {
            pruner.lock().closed = true;
            pruner.wake.notify_all();
        }
    }
}

/// Exclusive ownership of one stored response's writes, across processes.
///
/// Held for as long as this value lives; dropping it closes the sidecar and
/// releases the lock. The sidecar stays: only a lease holder unlinks it, and
/// only once the response has settled or gone.
#[derive(Debug)]
pub(super) struct ResponseLease {
    _file: fs::File,
}

/// What a sweep needs to know of a record, read without building the
/// Response or its input: unknown members, the bulk of a record, are skipped.
#[derive(Deserialize)]
struct Header {
    version: Option<u64>,
    response: Option<ResponseHeader>,
    #[serde(default)]
    request: Option<IgnoredAny>,
}

#[derive(Deserialize)]
struct ResponseHeader {
    id: Option<Value>,
    background: Option<Value>,
    status: Option<Value>,
    created_at: Option<Value>,
}

impl Header {
    fn field(&self, pick: impl Fn(&ResponseHeader) -> Option<&Value>) -> Option<&Value> {
        self.response.as_ref().and_then(pick)
    }

    /// Still `queued` or `in_progress`, as [`super::background::pending`].
    fn pending(&self) -> bool {
        matches!(
            self.field(|response| response.status.as_ref())
                .and_then(Value::as_str),
            Some("queued" | "in_progress")
        )
    }

    /// A background response its writer never finished, as [`interrupted`].
    fn interrupted(&self) -> bool {
        self.field(|response| response.background.as_ref())
            .and_then(Value::as_bool)
            == Some(true)
            && self.pending()
    }

    fn created_at(&self) -> Option<u64> {
        self.field(|response| response.created_at.as_ref())
            .and_then(Value::as_u64)
    }
}

/// The store's files by ID, from one listing of its directory.
#[derive(Default)]
struct Listing {
    /// Every record, with when it was last written if that was asked for.
    records: HashMap<String, Option<SystemTime>>,
    /// Every lease sidecar, with when it was last modified.
    locks: HashMap<String, Option<SystemTime>>,
    /// The IDs with a journal or a hosted-tool marker.
    sidecars: BTreeSet<String>,
    /// Temporary files of saves, with when each was last written.
    temporaries: Vec<(PathBuf, Option<SystemTime>)>,
}

/// A stored Response and the resolved input it was generated from.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StoredResponse {
    pub(super) response: Value,
    pub(super) input_items: Vec<Value>,
    /// The request body a pending durable background Response is regenerated
    /// from after a restart; `None` once it has ended, and for every other.
    pub(super) request: Option<Value>,
    /// Who created that pending request, kept beside it so a resumed
    /// generation's usage is attributed to them: `{"project_id",
    /// "api_key_id"}` for a project API key. `None` for an unattributed
    /// caller.
    pub(super) creator: Option<Value>,
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
        let store = Self {
            dir,
            pruner: Some(Arc::default()),
        };
        store.sweep(true, None)?;
        Ok(store)
    }

    fn path(&self, id: &str) -> Option<PathBuf> {
        valid_id(id).then(|| self.dir.join(format!("{id}.json")))
    }

    fn lock_path(&self, id: &str) -> Option<PathBuf> {
        valid_id(id).then(|| self.dir.join(format!(".{id}{LOCK_SUFFIX}")))
    }

    /// Take the write lease for response `id` without waiting.
    ///
    /// `Ok(None)` when another writer, in this process or another, holds it.
    /// An `id` that could not name a response is refused, as is a sidecar that
    /// is a symlink, not a regular file, or not owner-only.
    ///
    /// A sidecar found unlinked after it was opened (its response settled or
    /// was deleted meanwhile) is opened again by name, so every holder locks
    /// the sidecar the name leads to.
    pub(super) fn lease(&self, id: &str) -> std::io::Result<Option<ResponseLease>> {
        let path = self.lock_path(id).ok_or_else(|| invalid_id(id))?;
        for _ in 0..LEASE_ATTEMPTS {
            let file = open_private(&path, Access::Lock)?;
            let locked = match file.try_lock() {
                Ok(()) => true,
                Err(fs::TryLockError::WouldBlock) => false,
                Err(fs::TryLockError::Error(error)) => return Err(error),
            };
            if still_named(&file, &path)? {
                return Ok(locked.then_some(ResponseLease { _file: file }));
            }
        }
        Ok(None)
    }

    /// One pass over the directory, reading only the records that need it.
    ///
    /// Removes temporary files and lease sidecars with no record that are
    /// [`STALE_AFTER`] old, and journals and markers whose record is gone and
    /// whose lease is free. Of the records with a lease sidecar (every one
    /// that can be pending): `at_open`, one whose writer is gone and that
    /// cannot be resumed (its record keeps no request) is settled; a settled
    /// one's sidecar is unlinked under its lease. With a `cutoff` (Unix
    /// seconds), every settled record created and last written before it is
    /// deleted with its sidecars; its file's modification time decides which
    /// records are read at all. Returns how many records were deleted.
    ///
    /// Records that cannot be read are skipped: they are not this pass's to
    /// judge, and `load` reports them to whoever asks for them. Failing to
    /// remove a leftover file is reported and skipped too.
    fn sweep(&self, at_open: bool, cutoff: Option<u64>) -> std::io::Result<usize> {
        let listing = self.list(cutoff.is_some())?;
        let now = SystemTime::now();
        let stale = |modified: Option<SystemTime>| {
            modified
                .and_then(|modified| now.duration_since(modified).ok())
                .is_some_and(|age| age >= STALE_AFTER)
        };
        for (path, modified) in &listing.temporaries {
            if stale(*modified)
                && let Err(error) = remove_if_present(path)
            {
                eprintln!("response store: temporary file not removed: {error}");
            }
        }
        let orphans = listing
            .sidecars
            .iter()
            .chain(listing.locks.keys())
            .filter(|id| !listing.records.contains_key(*id))
            .collect::<BTreeSet<_>>();
        for id in orphans {
            // A writer creates its sidecar just before it locks it, and saves
            // its first record only after that: a young lone sidecar may be
            // in that window.
            let lone_lock = !listing.sidecars.contains(id);
            if lone_lock
                && !listing
                    .locks
                    .get(id)
                    .is_some_and(|modified| stale(*modified))
            {
                continue;
            }
            if let Err(error) = self.remove_orphan(id) {
                eprintln!("response {id}: leftover sidecars not removed: {error}");
            }
        }
        let cutoff_time = cutoff.map(|cutoff| SystemTime::UNIX_EPOCH + Duration::from_secs(cutoff));
        let mut removed = 0;
        for (id, modified) in &listing.records {
            let leased = listing.locks.contains_key(id);
            // A record is written after its response was created, so one
            // written since the cutoff was created after it too.
            let old = cutoff_time
                .is_some_and(|cutoff| modified.is_some_and(|modified| modified < cutoff));
            if !leased && !old {
                continue;
            }
            let Ok(Some(header)) = self.header(id) else {
                continue;
            };
            if header.pending() {
                if at_open && header.interrupted() && header.request.is_none() {
                    self.recover(id)?;
                }
                continue;
            }
            if old
                && cutoff.is_some_and(|cutoff| {
                    header.created_at().is_some_and(|created| created < cutoff)
                })
            {
                if self.delete(id)? {
                    removed += 1;
                }
                continue;
            }
            if leased && let Err(error) = self.remove_lock(id) {
                eprintln!("response {id}: lease sidecar not removed: {error}");
            }
        }
        Ok(removed)
    }

    /// The store's files, by ID; `record_times` also reads when each record
    /// was last written.
    fn list(&self, record_times: bool) -> std::io::Result<Listing> {
        let mut listing = Listing::default();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let modified = || {
                entry
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .ok()
            };
            let id = |suffix: &str| {
                name.strip_prefix('.')
                    .and_then(|name| name.strip_suffix(suffix))
                    .filter(|id| valid_id(id))
                    .map(str::to_owned)
            };
            if let Some(id) = name.strip_suffix(".json").filter(|id| valid_id(id)) {
                let modified = if record_times { modified() } else { None };
                listing.records.insert(id.to_owned(), modified);
            } else if let Some(id) = id(LOCK_SUFFIX) {
                listing.locks.insert(id, modified());
            } else if let Some(id) = id(JOURNAL_SUFFIX).or_else(|| id(TOOLS_SUFFIX)) {
                listing.sidecars.insert(id);
            } else if temporary_name(name) {
                listing.temporaries.push((entry.path(), modified()));
            }
        }
        Ok(listing)
    }

    /// What a sweep needs of record `id`, without building it; `None` when it
    /// is not stored.
    fn header(&self, id: &str) -> std::io::Result<Option<Header>> {
        let Some(path) = self.path(id) else {
            return Ok(None);
        };
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let header: Header = serde_json::from_slice(&bytes).map_err(|error| {
            std::io::Error::new(
                ErrorKind::InvalidData,
                format!("stored response {id} is corrupt: {error}"),
            )
        })?;
        let named = header.field(|response| response.id.as_ref()) == Some(&json!(id));
        if header.version == Some(RECORD_VERSION) && named {
            Ok(Some(header))
        } else {
            Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("stored response {id} is not a version {RECORD_VERSION} record"),
            ))
        }
    }

    /// Delete every settled response created more than `retention` ago,
    /// with its journal and sidecars, and return how many went; then keep
    /// doing so every [`PRUNE_INTERVAL`] for as long as this store is open
    /// (the first call starts that). Pending background responses are kept
    /// whatever their age, and records that cannot be read are left for
    /// `load` to report.
    ///
    /// A record's file modification time decides which records are read: one
    /// written within `retention` (a background response settled late, or a
    /// file copied without its times) waits until that much time has passed
    /// since it was written.
    pub(super) fn prune(&self, retention: Duration) -> std::io::Result<usize> {
        self.keep_pruning(retention);
        self.prune_once(retention)
    }

    fn prune_once(&self, retention: Duration) -> std::io::Result<usize> {
        let cutoff = super::response::unix_now().saturating_sub(retention.as_secs());
        self.sweep(false, Some(cutoff))
    }

    /// Start pruning this store every [`PRUNE_INTERVAL`] on a thread of its
    /// own, until the store is dropped. Once only.
    fn keep_pruning(&self, retention: Duration) {
        let Some(pruner) = &self.pruner else {
            return;
        };
        {
            let mut state = pruner.lock();
            if state.started {
                return;
            }
            state.started = true;
        }
        let (store, pruner) = (
            Self {
                dir: self.dir.clone(),
                pruner: None,
            },
            Arc::clone(pruner),
        );
        let spawned = std::thread::Builder::new()
            .name("response-store-pruner".into())
            .spawn(move || {
                let mut state = pruner.lock();
                loop {
                    state = pruner
                        .wake
                        .wait_timeout_while(state, PRUNE_INTERVAL, |state| !state.closed)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                    if state.closed {
                        return;
                    }
                    drop(state);
                    match store.prune_once(retention) {
                        Ok(0) => {}
                        Ok(removed) => eprintln!(
                            "removed {removed} stored responses older than {} days",
                            retention.as_secs() / 86_400
                        ),
                        Err(error) => {
                            eprintln!("warning: old stored responses were not pruned: {error}");
                        }
                    }
                    state = pruner.lock();
                }
            });
        if let Err(error) = spawned {
            eprintln!("warning: stored responses will be pruned only at startup: {error}");
        }
    }

    /// The IDs of every background response still `queued` or `in_progress`
    /// whose record keeps the request to resume it from, oldest first. Their
    /// leases are not checked: whoever resumes one takes it. Only records
    /// with a lease sidecar are read, as only those can be pending.
    pub(super) fn resumable(&self) -> std::io::Result<Vec<String>> {
        let listing = self.list(false)?;
        let mut found = Vec::new();
        for id in listing.records.keys() {
            if listing.locks.contains_key(id)
                && let Ok(Some(header)) = self.header(id)
                && header.interrupted()
                && header.request.is_some()
            {
                found.push((header.created_at().unwrap_or_default(), id.clone()));
            }
        }
        found.sort();
        Ok(found.into_iter().map(|(_, id)| id).collect())
    }

    /// Settle response `id` as not resumed if its writer is gone: `Ok(false)`
    /// when another writer holds its lease, so it is still being generated.
    pub(super) fn recover(&self, id: &str) -> std::io::Result<bool> {
        let Some(lease) = self.lease(id)? else {
            return Ok(false);
        };
        let resumable = matches!(self.load(id), Ok(Some(stored)) if stored.request.is_some());
        let reason = if resumable {
            "it was not resumed"
        } else {
            NO_REQUEST
        };
        self.settle_interrupted(&lease, id, reason)?;
        Ok(true)
    }

    /// Settle response `id`, whose writer is gone, under `_lease`, which the
    /// caller holds: nothing when it is no longer pending.
    ///
    /// A pending record whose journal already ends the stream adopts that end,
    /// because it was published; otherwise it becomes `failed` because of
    /// `reason`, and the events saying so are appended to its journal after
    /// the record is written.
    pub(super) fn settle_interrupted(
        &self,
        _lease: &ResponseLease,
        id: &str,
        reason: &str,
    ) -> std::io::Result<()> {
        // Re-read under the lease: the writer may have finished between the
        // caller's read and taking the lock.
        let Ok(Some(mut stored)) = self.load(id) else {
            return Ok(());
        };
        if !interrupted(&stored) {
            return Ok(());
        }
        let mut journal = self.open_journal(id).unwrap_or_else(|error| {
            eprintln!("response {id}: event journal unreadable: {error}");
            None
        });
        let last = match journal.as_mut().map(JournalReader::last_line).transpose() {
            Ok(last) => last.flatten(),
            Err(error) => {
                eprintln!("response {id}: event journal unreadable: {error}");
                journal = None;
                None
            }
        };
        let adopted = last
            .as_ref()
            .filter(|line| line.terminal)
            .and_then(|line| ended_with(&stored.response, line));
        let ended = adopted.is_some();
        if let Some(response) = adopted {
            stored.response = response;
        } else if let Some(response) = stored.response.as_object_mut() {
            response.insert("status".into(), json!("failed"));
            response.insert("completed_at".into(), Value::Null);
            response.insert(
                "error".into(),
                json!({"code":"server_error","message":format!("{INTERRUPTED_MESSAGE}: {reason}")}),
            );
        }
        self.save(&stored.response, &stored.input_items)?;
        if let Some(reader) = journal {
            let events = if ended {
                Vec::new()
            } else {
                terminal_events(&stored.response)
            };
            if let Err(error) = self.extend_journal(id, &reader, events) {
                eprintln!("response {id}: event journal not reconciled: {error}");
            }
        }
        Ok(())
    }

    /// Append the events that end `response` to its journal, unless it has
    /// already ended. The caller holds the lease. Without a journal, nothing.
    pub(super) fn seal_journal(&self, id: &str, response: &Value) -> std::io::Result<()> {
        let Some(mut reader) = self.open_journal(id)? else {
            return Ok(());
        };
        if reader.last_line()?.is_some_and(|line| line.terminal) {
            return Ok(());
        }
        self.extend_journal(id, &reader, terminal_events(response))
    }

    /// Continue the journal of response `id` after the last whole event
    /// `reader` has read, cutting off anything after it. The caller holds the
    /// lease.
    pub(super) fn reopen_journal(
        &self,
        id: &str,
        reader: &JournalReader,
    ) -> std::io::Result<JournalWriter> {
        let path = self.journal_path(id).ok_or_else(|| invalid_id(id))?;
        JournalWriter::resume(&path, reader)
    }

    /// Cut the journal `reader` has read to its last whole event and append
    /// `events` after it.
    fn extend_journal(
        &self,
        id: &str,
        reader: &JournalReader,
        events: Vec<Value>,
    ) -> std::io::Result<()> {
        let mut writer = self.reopen_journal(id, reader)?;
        writer.append(events)?;
        writer.sync()
    }

    /// Remove the journal, hosted-tool marker and lease sidecar of a response
    /// with no record, unless a writer holds its lease: that is an admission
    /// between its first writes.
    fn remove_orphan(&self, id: &str) -> std::io::Result<()> {
        let (Some(record), Some(journal), Some(marker), Some(lock)) = (
            self.path(id),
            self.journal_path(id),
            self.marker_path(id),
            self.lock_path(id),
        ) else {
            return Ok(());
        };
        if record.exists() {
            return Ok(());
        }
        let Some(_lease) = self.lease(id)? else {
            return Ok(());
        };
        if !record.exists() {
            remove_if_present(&journal)?;
            remove_if_present(&marker)?;
            if UNLINKS_LOCKS {
                remove_if_present(&lock)?;
            }
        }
        Ok(())
    }

    /// Unlink the lease sidecar of response `id`, under its lease, once its
    /// record is settled or gone. Nothing when it has none, when a writer
    /// holds it (the next sweep removes it), or when its record cannot be
    /// read.
    fn remove_lock(&self, id: &str) -> std::io::Result<()> {
        let Some(lock) = self.lock_path(id) else {
            return Ok(());
        };
        // Leasing creates the sidecar, so a response without one is left so.
        if !UNLINKS_LOCKS || fs::symlink_metadata(&lock).is_err() {
            return Ok(());
        }
        let Some(_lease) = self.lease(id)? else {
            return Ok(());
        };
        match self.header(id) {
            Ok(Some(header)) if header.pending() => Ok(()),
            Ok(_) => remove_if_present(&lock).map(drop),
            Err(_) => Ok(()),
        }
    }

    fn marker_path(&self, id: &str) -> Option<PathBuf> {
        valid_id(id).then(|| self.dir.join(format!(".{id}{TOOLS_SUFFIX}")))
    }

    /// Record, durably, that a hosted tool call of response `id` is about to
    /// start executing. Idempotent.
    pub(super) fn mark_tools_started(&self, id: &str) -> std::io::Result<()> {
        let path = self.marker_path(id).ok_or_else(|| invalid_id(id))?;
        open_private(&path, Access::Lock)?.sync_all()?;
        sync_dir(&self.dir);
        Ok(())
    }

    /// Whether a hosted tool call of response `id` may have started
    /// executing: its marker exists, whatever it is.
    pub(super) fn tools_started(&self, id: &str) -> std::io::Result<bool> {
        let path = self.marker_path(id).ok_or_else(|| invalid_id(id))?;
        match fs::symlink_metadata(path) {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn journal_path(&self, id: &str) -> Option<PathBuf> {
        valid_id(id).then(|| self.dir.join(format!(".{id}{JOURNAL_SUFFIX}")))
    }

    /// Create the journal of response `id` holding `events`, fully flushed.
    /// The caller holds the lease.
    pub(super) fn create_journal(
        &self,
        id: &str,
        events: Vec<Value>,
    ) -> std::io::Result<JournalWriter> {
        let path = self.journal_path(id).ok_or_else(|| invalid_id(id))?;
        let created = JournalWriter::create(&path).and_then(|mut writer| {
            writer.append(events)?;
            writer.sync()?;
            Ok(writer)
        });
        if created.is_err() {
            let _ = fs::remove_file(&path);
        }
        sync_dir(&self.dir);
        created
    }

    /// The journal of response `id`; `None` when it has none.
    pub(super) fn open_journal(&self, id: &str) -> std::io::Result<Option<JournalReader>> {
        self.journal_path(id)
            .map_or_else(|| Ok(None), |path| JournalReader::open(&path))
    }

    /// The journal path of response `id`, for its live writer.
    #[cfg(test)]
    pub(super) fn journal_file(&self, id: &str) -> Option<PathBuf> {
        self.journal_path(id)
    }

    /// Persist `response` and its resolved input, durably, before returning.
    pub(super) fn save(&self, response: &Value, input_items: &[Value]) -> std::io::Result<()> {
        self.save_with(response, input_items, None)
    }

    /// [`Self::save`], keeping `pending` in the record as well: the body a
    /// pending background Response is regenerated from after a restart, and
    /// who created it.
    pub(super) fn save_with(
        &self,
        response: &Value,
        input_items: &[Value],
        pending: Option<(&Value, Option<&Value>)>,
    ) -> std::io::Result<()> {
        let id = response.get("id").and_then(Value::as_str).unwrap_or("");
        let path = self.path(id).ok_or_else(|| invalid_id(id))?;
        let mut record = json!({
            "object":"local_ai.stored_response",
            "version":RECORD_VERSION,
            "response":response,
            "input_items":input_items,
        });
        if let Some((request, creator)) = pending {
            record["request"] = request.clone();
            if let Some(creator) = creator {
                record["creator"] = creator.clone();
            }
        }
        // A sweep reads only records with a lease sidecar, so a pending
        // background record always has one, even one saved without its lease.
        if response.get("background").and_then(Value::as_bool) == Some(true)
            && super::background::pending(response)
        {
            let lock = self.lock_path(id).ok_or_else(|| invalid_id(id))?;
            open_private(&lock, Access::Lock)?;
        }
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
        let request = record.get_mut("request").map(Value::take);
        let creator = record
            .get_mut("creator")
            .map(Value::take)
            .filter(|creator| request.is_some() && creator.is_object());
        match (version, response, input_items) {
            (Some(RECORD_VERSION), Some(response), Some(Value::Array(input_items)))
                if response.get("id").and_then(Value::as_str) == Some(id) =>
            {
                Ok(Some(StoredResponse {
                    response,
                    input_items,
                    request,
                    creator,
                }))
            }
            _ => Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("stored response {id} is not a version {RECORD_VERSION} record"),
            )),
        }
    }

    /// Delete the stored response `id`, its event journal, its hosted-tool
    /// marker and, unless a writer holds it, its lease sidecar, reporting
    /// whether there was one.
    ///
    /// The record goes first: once it is gone the response is not found,
    /// whatever became of the rest, and a sidecar left by a failed unlink or
    /// a held lease is removed by the next sweep.
    pub(super) fn delete(&self, id: &str) -> std::io::Result<bool> {
        let (Some(path), Some(journal)) = (self.path(id), self.journal_path(id)) else {
            return Ok(false);
        };
        let found = remove_if_present(&path)?;
        if let Err(error) = remove_if_present(&journal) {
            if !found {
                return Err(error);
            }
            eprintln!("response {id}: event journal not removed: {error}");
        }
        if let Some(marker) = self.marker_path(id)
            && let Err(error) = remove_if_present(&marker)
        {
            // Removed when a store next opens, as the record is gone.
            eprintln!("response {id}: hosted tool marker not removed: {error}");
        }
        if let Err(error) = self.remove_lock(id) {
            eprintln!("response {id}: lease sidecar not removed: {error}");
        }
        sync_dir(&self.dir);
        Ok(found)
    }
}

/// Whether lease sidecars are ever unlinked. Only where [`still_named`] can
/// tell that a locked sidecar was unlinked meanwhile; elsewhere they stay.
const UNLINKS_LOCKS: bool = cfg!(unix);

/// Whether `path` still names the open `file`: `false` once it was unlinked,
/// or replaced by another file.
#[cfg(unix)]
fn still_named(file: &fs::File, path: &Path) -> std::io::Result<bool> {
    use std::os::unix::fs::MetadataExt as _;
    let held = file.metadata()?;
    match fs::symlink_metadata(path) {
        Ok(named) => Ok(named.dev() == held.dev() && named.ino() == held.ino()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn still_named(_file: &fs::File, _path: &Path) -> std::io::Result<bool> {
    Ok(true)
}

/// Whether `name` is a save's temporary file, `.{id}.{nonce}.tmp`.
fn temporary_name(name: &str) -> bool {
    name.strip_prefix('.')
        .and_then(|name| name.strip_suffix(TEMPORARY_SUFFIX))
        .and_then(|name| name.rsplit_once('.'))
        .is_some_and(|(id, nonce)| {
            valid_id(id)
                && !nonce.is_empty()
                && nonce.bytes().all(|byte| byte.is_ascii_alphanumeric())
        })
}

fn invalid_id(id: &str) -> std::io::Error {
    std::io::Error::new(
        ErrorKind::InvalidInput,
        format!("invalid response ID {id:?}"),
    )
}

/// Unlink `path`, reporting whether it existed.
fn remove_if_present(path: &Path) -> std::io::Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
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

/// How [`open_private`] opens a store sidecar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Access {
    /// A lease sidecar: read-write, created if missing, never truncated.
    Lock,
    /// A new journal: append-only, and an existing file is an error.
    CreateNew,
    /// An existing journal, to append to.
    Append,
    /// An existing journal, to read.
    Read,
}

/// Open the owner-only sidecar at `path` for `access`.
///
/// Never truncates or unlinks it, so concurrent openers share one inode. On
/// Unix a symlink as the final component is refused rather than followed, a
/// FIFO cannot block the open, and the result must be an owner-only regular
/// file belonging to this user.
///
/// An error names the file but not the directory it is in, since it may
/// reach a client; the full path is logged instead (except for a file that
/// does not exist, which callers expect).
#[cfg(unix)]
pub(super) fn open_private(path: &Path, access: Access) -> std::io::Result<fs::File> {
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

    use rustix::fs::{Mode, OFlags};

    let flags = OFlags::NOFOLLOW
        | OFlags::NONBLOCK
        | OFlags::CLOEXEC
        | match access {
            Access::Lock => OFlags::RDWR | OFlags::CREATE,
            Access::CreateNew => OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::APPEND,
            Access::Append => OFlags::WRONLY | OFlags::APPEND,
            Access::Read => OFlags::RDONLY,
        };
    let name = path
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    let fd = rustix::fs::open(path, flags, Mode::RUSR | Mode::WUSR).map_err(|errno| {
        let kind = std::io::Error::from(errno).kind();
        if kind != ErrorKind::NotFound {
            eprintln!("response store: cannot open {}: {errno}", path.display());
        }
        std::io::Error::new(kind, format!("cannot open response file {name}: {errno}"))
    })?;
    let file = fs::File::from(fd);
    let metadata = file.metadata()?;
    let refused = |why: &str| {
        eprintln!("response store: {} {why}", path.display());
        Err(std::io::Error::new(
            ErrorKind::PermissionDenied,
            format!("response file {name} {why}"),
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
pub(super) fn open_private(path: &Path, access: Access) -> std::io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    match access {
        Access::Lock => options.read(true).write(true).create(true).truncate(false),
        Access::CreateNew => options.append(true).create_new(true),
        Access::Append => options.append(true),
        Access::Read => options.read(true),
    };
    options.open(path)
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
