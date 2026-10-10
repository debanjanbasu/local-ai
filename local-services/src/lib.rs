//! Durable native services shared by the CLI and the HTTP server.
//!
//! One owner-only directory holds one `SQLite` database in WAL mode, so several
//! handles, threads and processes can share it safely. Every method is
//! blocking; async callers run them through a bounded `spawn_blocking`.
//!
//! Each component owns and versions its own tables; this
//! module only provides the hardened connection and the shared vocabulary.

pub mod batches;
pub mod conversations;
pub mod files;
pub mod workspace;

pub use conversations::{
    Append, Appended, Conversation, ConversationDeleted, History, Item, ListItems, Metadata, Order,
    Page,
};

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OpenFlags, OptionalExtension as _, TransactionBehavior, params};

/// The result of every service operation.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Why a service operation failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The caller's input is malformed or outside the supported contract.
    #[error("{0}")]
    InvalidArgument(String),
    /// The named object does not exist (or was deleted).
    #[error("{0}")]
    NotFound(String),
    /// The request is well-formed but conflicts with the current state.
    #[error("{0}")]
    Conflict(String),
    /// A filesystem failure, including refused insecure paths
    /// ([`ErrorKind::PermissionDenied`]).
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A database failure, including a lock wait that outlived the busy timeout.
    #[error(transparent)]
    Database(#[from] rusqlite::Error),
    /// A stored or supplied JSON document could not be (de)serialized.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// The database file inside the store directory.
const DATABASE: &str = "services.sqlite3";
/// Files `SQLite` creates next to the database; never followed if symlinks.
const SIDECARS: [&str; 3] = ["-wal", "-shm", "-journal"];
/// How long a writer waits for another writer before failing with `SQLITE_BUSY`.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// A handle to the durable store; cheap to clone and share across threads.
///
/// Each operation opens its own connection, so a `Store` holds no lock and no
/// open file between calls.
#[derive(Clone, Debug)]
pub struct Store {
    root: Arc<PathBuf>,
}

impl Store {
    /// Open the store in directory `path`, creating it (mode 0700) and its
    /// database (mode 0600) if needed and initializing every owned schema.
    ///
    /// Refuses, with an [`Error::Io`] of kind [`ErrorKind::PermissionDenied`],
    /// a directory or database that is a symlink, is not of the expected type,
    /// or is accessible by anyone but its owner.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        create_private_dir(&root)?;
        create_private_file(&root, &root.join(DATABASE))?;
        let store = Self {
            root: Arc::new(root.canonicalize()?),
        };
        let mut connection = store.connection()?;
        conversations::initialize(&mut connection)?;
        files::initialize(&mut connection)?;
        batches::initialize(&mut connection)?;
        Ok(store)
    }

    /// The store directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// A new hardened connection to the store's database.
    ///
    /// WAL journal, `synchronous = FULL` with `fullfsync`, foreign keys on,
    /// untrusted schema, and a bounded busy timeout. The paths are re-checked
    /// on every open and the database is opened without following symlinks.
    pub(crate) fn connection(&self) -> Result<Connection> {
        check_private_dir(&self.root)?;
        let database = self.root.join(DATABASE);
        check_private_file(&database, false)?;
        for suffix in SIDECARS {
            check_private_file(&self.root.join(format!("{DATABASE}{suffix}")), true)?;
        }
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW;
        let connection = Connection::open_with_flags(&database, flags)?;
        connection.busy_timeout(BUSY_TIMEOUT)?;
        let mode: String =
            connection.query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(Error::Io(std::io::Error::other(format!(
                "{} cannot use WAL journaling (journal mode is {mode})",
                database.display()
            ))));
        }
        connection.execute_batch(
            "PRAGMA synchronous = FULL;
             PRAGMA fullfsync = ON;
             PRAGMA checkpoint_fullfsync = ON;
             PRAGMA foreign_keys = ON;
             PRAGMA trusted_schema = OFF;",
        )?;
        Ok(connection)
    }
}

/// Bring `component`'s tables up to date, atomically.
///
/// `migrations[n]` upgrades the schema from version `n` to `n + 1`; the
/// versions applied are recorded in the shared `schema_versions` table. All of
/// it runs in one immediate transaction, so concurrent openers (threads or
/// processes) apply each migration exactly once. A database written by a newer
/// build, with more migrations than this one knows, is refused.
pub(crate) fn migrate(
    connection: &mut Connection,
    component: &str,
    migrations: &[&str],
) -> Result<()> {
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_versions (
             component TEXT PRIMARY KEY NOT NULL,
             version INTEGER NOT NULL
         ) STRICT;",
    )?;
    let current: i64 = transaction
        .query_row(
            "SELECT version FROM schema_versions WHERE component = ?1",
            [component],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0);
    let known = i64::try_from(migrations.len()).map_err(std::io::Error::other)?;
    let applied = usize::try_from(current)
        .ok()
        .filter(|_| current <= known)
        .ok_or_else(|| {
            Error::Conflict(format!(
                "{component} schema version {current} is not supported by this build \
                 (latest {known})"
            ))
        })?;
    for migration in migrations.iter().skip(applied) {
        transaction.execute_batch(migration)?;
    }
    transaction.execute(
        "INSERT INTO schema_versions (component, version) VALUES (?1, ?2)
         ON CONFLICT (component) DO UPDATE SET version = excluded.version",
        params![component, known],
    )?;
    transaction.commit()?;
    Ok(())
}

/// A fresh identifier: `prefix` and 48 hex digits (192 bits) of OS randomness.
pub(crate) fn new_id(prefix: &str) -> Result<String> {
    use ring::rand::{SecureRandom as _, SystemRandom};

    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut bytes = [0_u8; 24];
    SystemRandom::new().fill(&mut bytes).map_err(|_| {
        Error::Io(std::io::Error::other(
            "operating system randomness is unavailable",
        ))
    })?;
    let mut id = String::with_capacity(prefix.len() + bytes.len() * 2);
    id.push_str(prefix);
    for byte in bytes {
        id.push(char::from(HEX[usize::from(byte >> 4)]));
        id.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    Ok(id)
}

/// Seconds since the Unix epoch.
pub(crate) fn now() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .map_err(|error| Error::Io(std::io::Error::other(error)))
}

fn refused(path: &Path, why: &str) -> Error {
    Error::Io(std::io::Error::new(
        ErrorKind::PermissionDenied,
        format!("store path {} {why}", path.display()),
    ))
}

fn create_private_dir(root: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    builder.create(root)?;
    check_private_dir(root)
}

fn check_private_dir(root: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() {
        return Err(refused(root, "is a symlink"));
    }
    if !metadata.is_dir() {
        return Err(refused(root, "is not a directory"));
    }
    check_mode(root, &metadata)
}

/// Create the database file owner-only if it does not exist yet.
///
/// `create_new` is `O_CREAT | O_EXCL`, which never follows a symlink, so a
/// planted link fails here as "already exists" and is then refused.
fn create_private_file(root: &Path, path: &Path) -> Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(file) => {
            file.sync_all()?;
            // Make the new directory entry durable where the platform allows.
            if let Ok(directory) = fs::File::open(root) {
                directory.sync_all().ok();
            }
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    check_private_file(path, false)
}

fn check_private_file(path: &Path, optional: bool) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if optional && error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() {
        return Err(refused(path, "is a symlink"));
    }
    if !metadata.is_file() {
        return Err(refused(path, "is not a regular file"));
    }
    check_mode(path, &metadata)
}

/// Group and other permission bits are refused. Ownership by another user
/// needs no separate check: such an owner-only file or directory could not be
/// opened at all.
#[cfg(unix)]
fn check_mode(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    if metadata.permissions().mode().trailing_zeros() >= 6 {
        Ok(())
    } else {
        Err(refused(path, "must be accessible only by its owner"))
    }
}

#[cfg(not(unix))]
fn check_mode(_path: &Path, _metadata: &fs::Metadata) -> Result<()> {
    Ok(())
}
