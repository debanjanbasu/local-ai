//! Native persistence for Files and multipart Uploads.
//!
//! # Storage layout
//!
//! All bytes live under `<store root>/file-store/`, in owner-only (`0700`)
//! directories. Like the store root, an existing storage directory that is a
//! symlink, not a directory, or accessible by group/other is refused with
//! [`std::io::ErrorKind::PermissionDenied`], never repaired:
//!
//! * `blobs/<blob id>`: committed or staged content, mode `0600`.
//! * `tmp/<blob id>.tmp`: content being streamed in, mode `0600`.
//!
//! Blob ids use secure OS randomness and are validated as
//! plain `[A-Za-z0-9_-]` path components. User-supplied filenames are metadata
//! only and are never used to build paths.
//!
//! # Crash safety
//!
//! A `file_blobs` row (state `staging`) is committed *before* any bytes touch
//! the disk. Content is written to `tmp/`, `fsync`ed, renamed into `blobs/`, and
//! the directory is `fsync`ed. Only then does a single transaction mark the blob
//! `committed` and insert the owning `files` / `upload_parts` row. Removal
//! always deletes metadata first and unlinks bytes afterwards. Therefore:
//!
//! * a disk entry without a `file_blobs` row is always a crash or race leftover
//!   and is safe to delete;
//! * a `staging` row older than [`STAGING_TTL_SECONDS`] belongs to an abandoned
//!   write.
//!
//! # Expiry and physical cleanup
//!
//! There is no background task. Expiry is enforced on access: expired files are
//! reported as not found and omitted from listings, and expired uploads reject
//! every mutation. Bytes of expired files, parts of expired uploads, abandoned
//! staging blobs, and orphaned disk entries are physically removed only when
//! [`Store::purge_file_storage`] is called (for example at startup or from an
//! operator-triggered maintenance step). Upload rows themselves (pending,
//! expired, completed, cancelled) are retained as small metadata records.
//!
//! # Purposes
//!
//! Purposes are validated and stored verbatim. Storing a file with a purpose
//! such as `vision` or `fine-tune` does not imply that any inference, embedding,
//! or training capability exists for it.
//!
//! `batch_output` is internal: batch result files carry it, and they can be
//! listed, read and deleted like any other file, but [`Store::create_file`] and
//! [`Store::create_upload`] refuse it. Internal files have deterministic ids
//! and are created idempotently: a retry returns the file already committed
//! under that id, and an id whose file was deleted or has expired is never
//! created again (so its lifetime cannot be reset).

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, Store};

const MIB: u64 = 1024 * 1024;

/// Largest file accepted by [`Store::create_file`] for non-batch purposes.
pub const MAX_FILE_BYTES: u64 = 512 * MIB;
/// Largest file accepted for the `batch` purpose (direct or via uploads).
pub const MAX_BATCH_FILE_BYTES: u64 = 200 * MIB;
/// Largest total size an upload may declare.
pub const MAX_UPLOAD_BYTES: u64 = 8 * 1024 * MIB;
/// Largest single upload part.
pub const MAX_UPLOAD_PART_BYTES: u64 = 64 * MIB;
/// Lifetime of a pending upload.
pub const UPLOAD_TTL_SECONDS: u64 = 3600;
/// Smallest accepted `expires_after.seconds`.
pub const MIN_EXPIRES_AFTER_SECONDS: u64 = 3600;
/// Largest accepted `expires_after.seconds`.
pub const MAX_EXPIRES_AFTER_SECONDS: u64 = 2_592_000;
/// Expiry applied to `batch` files when none is requested.
pub const DEFAULT_BATCH_EXPIRES_AFTER_SECONDS: u64 = 2_592_000;
/// Largest internal `batch_output` file (a batch result file).
pub const MAX_BATCH_OUTPUT_BYTES: u64 = 512 * MIB;
/// Expiry of `batch_output` files, from their creation.
pub const BATCH_OUTPUT_EXPIRES_AFTER_SECONDS: u64 = 2_592_000;
/// Largest and default page size for [`Store::list_files`].
pub const MAX_LIST_LIMIT: u32 = 10_000;
/// Age after which an uncommitted staging blob is treated as abandoned.
pub const STAGING_TTL_SECONDS: u64 = 86_400;

const STORAGE_DIR: &str = "file-store";
const BLOB_DIR: &str = "blobs";
const TMP_DIR: &str = "tmp";
const TMP_SUFFIX: &str = ".tmp";
const COPY_BUFFER_BYTES: usize = 64 * 1024;
const MAX_FILENAME_BYTES: usize = 1024;
const MAX_MIME_TYPE_BYTES: usize = 255;
const MAX_ID_BYTES: usize = 128;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS file_blobs (
    id TEXT PRIMARY KEY NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('staging', 'committed')),
    created_at INTEGER NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS file_blobs_state_created ON file_blobs (state, created_at);
CREATE TABLE IF NOT EXISTS files (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    blob_id TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    purpose TEXT NOT NULL,
    bytes INTEGER NOT NULL CHECK (bytes >= 0),
    created_at INTEGER NOT NULL,
    expires_at INTEGER
) STRICT;
CREATE INDEX IF NOT EXISTS files_purpose_seq ON files (purpose, seq);
CREATE INDEX IF NOT EXISTS files_expires_at ON files (expires_at);
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY NOT NULL,
    filename TEXT NOT NULL,
    purpose TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    bytes INTEGER NOT NULL CHECK (bytes > 0),
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'cancelled')),
    file_expires_after INTEGER,
    file_id TEXT
) STRICT;
CREATE INDEX IF NOT EXISTS uploads_status_expires ON uploads (status, expires_at);
CREATE TABLE IF NOT EXISTS upload_parts (
    id TEXT PRIMARY KEY NOT NULL,
    upload_id TEXT NOT NULL,
    blob_id TEXT NOT NULL UNIQUE,
    bytes INTEGER NOT NULL CHECK (bytes > 0),
    created_at INTEGER NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS upload_parts_upload ON upload_parts (upload_id);
";

/// Version 2: every internal (deterministic) file id ever committed. Rows
/// are never removed, so a deleted or expired internal file stays gone.
const INTERNAL_FILES: &str = "
CREATE TABLE internal_files (
    id TEXT PRIMARY KEY NOT NULL,
    purpose TEXT NOT NULL,
    filename TEXT NOT NULL,
    created_at INTEGER NOT NULL
) STRICT;
";

/// Validated file purpose. Stored verbatim; implies no processing capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FilePurpose {
    #[serde(rename = "assistants")]
    Assistants,
    #[serde(rename = "batch")]
    Batch,
    #[serde(rename = "fine-tune")]
    FineTune,
    #[serde(rename = "vision")]
    Vision,
    #[serde(rename = "user_data")]
    UserData,
    #[serde(rename = "evals")]
    Evals,
    /// Internal: batch result files. Listable, never externally creatable.
    #[serde(rename = "batch_output")]
    BatchOutput,
}

impl FilePurpose {
    /// Wire representation of the purpose.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Assistants => "assistants",
            Self::Batch => "batch",
            Self::FineTune => "fine-tune",
            Self::Vision => "vision",
            Self::UserData => "user_data",
            Self::Evals => "evals",
            Self::BatchOutput => "batch_output",
        }
    }

    /// Whether only the services themselves create files with this purpose;
    /// [`Store::create_file`] and [`Store::create_upload`] refuse it.
    #[must_use]
    pub const fn is_internal(self) -> bool {
        matches!(self, Self::BatchOutput)
    }

    /// Largest file [`Store::create_file`] accepts for this purpose.
    #[must_use]
    pub const fn max_file_bytes(self) -> u64 {
        match self {
            Self::Batch => MAX_BATCH_FILE_BYTES,
            Self::BatchOutput => MAX_BATCH_OUTPUT_BYTES,
            _ => MAX_FILE_BYTES,
        }
    }

    /// Largest total an upload may declare for this purpose (0: no uploads).
    #[must_use]
    pub const fn max_upload_bytes(self) -> u64 {
        match self {
            Self::Batch => MAX_BATCH_FILE_BYTES,
            Self::BatchOutput => 0,
            _ => MAX_UPLOAD_BYTES,
        }
    }

    const fn default_expires_after(self) -> Option<u64> {
        match self {
            Self::Batch => Some(DEFAULT_BATCH_EXPIRES_AFTER_SECONDS),
            Self::BatchOutput => Some(BATCH_OUTPUT_EXPIRES_AFTER_SECONDS),
            _ => None,
        }
    }
}

impl FromStr for FilePurpose {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "assistants" => Ok(Self::Assistants),
            "batch" => Ok(Self::Batch),
            "fine-tune" => Ok(Self::FineTune),
            "vision" => Ok(Self::Vision),
            "user_data" => Ok(Self::UserData),
            "evals" => Ok(Self::Evals),
            "batch_output" => Ok(Self::BatchOutput),
            other => Err(Error::InvalidArgument(format!(
                "unsupported file purpose {other:?}"
            ))),
        }
    }
}

/// Anchor for [`ExpiresAfter`]; only `created_at` is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ExpiresAfterAnchor {
    #[default]
    #[serde(rename = "created_at")]
    CreatedAt,
}

/// Requested file expiry, `seconds` in `3600..=2592000`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpiresAfter {
    pub anchor: ExpiresAfterAnchor,
    pub seconds: u64,
}

/// Input for [`Store::create_file`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateFile {
    pub filename: String,
    pub purpose: FilePurpose,
    pub expires_after: Option<ExpiresAfter>,
}

/// Stored file metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileObject {
    pub id: String,
    pub bytes: u64,
    pub created_at: u64,
    pub expires_at: Option<u64>,
    pub filename: String,
    pub purpose: FilePurpose,
}

/// Metadata plus a safely opened, read-only handle to the file bytes.
#[derive(Debug)]
pub struct FileContent {
    pub file: FileObject,
    pub content: File,
}

/// Result of [`Store::delete_file`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletedFile {
    pub id: String,
    pub deleted: bool,
}

/// Listing order by creation sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SortOrder {
    Asc,
    #[default]
    Desc,
}

/// Query for [`Store::list_files`]. `limit` defaults to [`MAX_LIST_LIMIT`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ListFiles {
    pub after: Option<String>,
    pub limit: Option<u32>,
    pub order: SortOrder,
    pub purpose: Option<FilePurpose>,
}

/// One page of files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileList {
    pub data: Vec<FileObject>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
}

/// Input for [`Store::create_upload`]. `expires_after` applies to the file
/// produced on completion, anchored at completion time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateUpload {
    pub filename: String,
    pub purpose: FilePurpose,
    pub bytes: u64,
    pub mime_type: String,
    pub expires_after: Option<ExpiresAfter>,
}

/// What [`Store::create_internal_file`] found or made under an id.
#[derive(Debug)]
pub(crate) enum InternalFile {
    /// The live file: just created, or committed by an earlier attempt.
    Live(FileObject),
    /// A file was committed under the id but was deleted or has expired.
    Gone,
}

/// Upload lifecycle state. `Expired` is derived from `expires_at` on access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UploadStatus {
    Pending,
    Completed,
    Cancelled,
    Expired,
}

/// Stored upload metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UploadObject {
    pub id: String,
    pub bytes: u64,
    pub created_at: u64,
    pub expires_at: u64,
    pub filename: String,
    pub purpose: FilePurpose,
    pub mime_type: String,
    pub status: UploadStatus,
    /// The produced file, when completed and the file still exists.
    pub file: Option<FileObject>,
}

/// One stored upload part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UploadPart {
    pub id: String,
    pub upload_id: String,
    pub bytes: u64,
    pub created_at: u64,
}

/// Input for [`Store::complete_upload`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CompleteUpload {
    /// Ordered, unique part ids; their sizes must sum to the declared bytes.
    pub part_ids: Vec<String>,
    /// Optional hex MD5 of the assembled bytes, verified before completion.
    pub md5: Option<String>,
}

/// Counts reported by [`Store::purge_file_storage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FileStoragePurge {
    pub expired_files: u64,
    pub expired_upload_parts: u64,
    pub stale_staging_blobs: u64,
    pub orphan_entries: u64,
}

impl Store {
    /// Streams `content` into a new file, enforcing the purpose byte limit
    /// while reading.
    pub fn create_file(&self, request: &CreateFile, content: impl Read) -> Result<FileObject> {
        validate_external_purpose(request.purpose)?;
        validate_filename(&request.filename)?;
        validate_purpose_filename(request.purpose, &request.filename)?;
        let expires_after = validate_expires_after(request.purpose, request.expires_after)?;
        let limit = request.purpose.max_file_bytes();
        let file_id = new_resource_id("file")?;
        let dirs = self.file_dirs()?;
        let mut conn = self.files_connection()?;
        let blob_id = stage_blob(&conn)?;

        let written = dirs.write_blob(&blob_id, |file| {
            copy_limited(content, file, limit, &mut |_: &[u8]| {})
        });
        let bytes = match written {
            Ok(0) => {
                discard_blob(&conn, &dirs, &blob_id);
                return Err(Error::InvalidArgument("file content is empty".to_owned()));
            }
            Ok(bytes) => bytes,
            Err(error) => {
                discard_blob(&conn, &dirs, &blob_id);
                return Err(error);
            }
        };

        let created_at = crate::now()?;
        let object = FileObject {
            id: file_id,
            bytes,
            created_at,
            expires_at: expires_after.map(|seconds| created_at.saturating_add(seconds)),
            filename: request.filename.clone(),
            purpose: request.purpose,
        };
        if let Err(error) = commit_new_file(&mut conn, &blob_id, &object) {
            discard_blob(&conn, &dirs, &blob_id);
            return Err(error);
        }
        Ok(object)
    }

    /// Returns metadata for a live (not deleted, not expired) file.
    pub fn get_file(&self, file_id: &str) -> Result<FileObject> {
        let conn = self.files_connection()?;
        let now = sql_int(crate::now()?)?;
        live_file_row(&conn, file_id, now)?
            .ok_or_else(|| file_not_found(file_id))?
            .into_object()
    }

    /// Lists live files ordered by creation sequence.
    pub fn list_files(&self, query: &ListFiles) -> Result<FileList> {
        let limit = query.limit.unwrap_or(MAX_LIST_LIMIT);
        if !(1..=MAX_LIST_LIMIT).contains(&limit) {
            return Err(Error::InvalidArgument(format!(
                "limit must be between 1 and {MAX_LIST_LIMIT}"
            )));
        }
        let conn = self.files_connection()?;
        let now = sql_int(crate::now()?)?;
        let cursor = match &query.after {
            Some(after) => Some(
                conn.query_row(
                    "SELECT seq FROM files WHERE id = ?1",
                    params![after],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .ok_or_else(|| Error::NotFound(format!("cursor file {after} not found")))?,
            ),
            None => None,
        };
        let (comparison, direction, start) = match query.order {
            SortOrder::Desc => ("<", "DESC", i64::MAX),
            SortOrder::Asc => (">", "ASC", i64::MIN),
        };
        let sql = format!(
            "SELECT {FILE_COLUMNS} FROM files \
             WHERE seq {comparison} ?1 \
               AND (expires_at IS NULL OR expires_at > ?2) \
               AND (?3 IS NULL OR purpose = ?3) \
             ORDER BY seq {direction} LIMIT ?4"
        );
        let mut statement = conn.prepare(&sql)?;
        let mut rows = statement
            .query_map(
                params![
                    cursor.unwrap_or(start),
                    now,
                    query.purpose.map(FilePurpose::as_str),
                    i64::from(limit) + 1
                ],
                read_file_row,
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let page = usize::try_from(limit).unwrap_or(usize::MAX);
        let has_more = rows.len() > page;
        rows.truncate(page);
        let data = rows
            .into_iter()
            .map(FileRow::into_object)
            .collect::<Result<Vec<_>>>()?;
        Ok(FileList {
            first_id: data.first().map(|file| file.id.clone()),
            last_id: data.last().map(|file| file.id.clone()),
            data,
            has_more,
        })
    }

    /// Deletes a live file. Metadata is removed transactionally before the
    /// bytes are unlinked; an unlink failure leaves an orphan for
    /// [`Store::purge_file_storage`].
    pub fn delete_file(&self, file_id: &str) -> Result<DeletedFile> {
        let dirs = self.file_dirs()?;
        let mut conn = self.files_connection()?;
        let tx = immediate(&mut conn)?;
        // Read the clock only once the write lock is held, so a lock wait
        // cannot make an expired file look live.
        let now = sql_int(crate::now()?)?;
        let row = live_file_row(&tx, file_id, now)?.ok_or_else(|| file_not_found(file_id))?;
        tx.execute("DELETE FROM files WHERE id = ?1", params![file_id])?;
        tx.execute("DELETE FROM file_blobs WHERE id = ?1", params![row.blob_id])?;
        tx.commit()?;
        let _ = dirs.remove_blob(&row.blob_id);
        Ok(DeletedFile {
            id: file_id.to_owned(),
            deleted: true,
        })
    }

    /// Opens the bytes of a live file without following symlinks and after
    /// verifying the stored size.
    pub fn open_file_content(&self, file_id: &str) -> Result<FileContent> {
        let dirs = self.file_dirs()?;
        let conn = self.files_connection()?;
        let now = sql_int(crate::now()?)?;
        let row = live_file_row(&conn, file_id, now)?.ok_or_else(|| file_not_found(file_id))?;
        let blob_id = row.blob_id.clone();
        let file = row.into_object()?;
        let content = dirs.open_blob(&blob_id, file.bytes)?;
        Ok(FileContent { file, content })
    }

    /// Streams `content` into the internal file `file_id`, at most once.
    ///
    /// Idempotent per id: when a file was already committed under `file_id`
    /// (with this purpose and filename) nothing is written and that file is
    /// returned, or [`InternalFile::Gone`] if it has since been deleted or has
    /// expired; such an id is never created again. The bytes are staged and
    /// become visible only in the commit transaction, so an interrupted write
    /// is never mistaken for a committed file. The purpose's byte limit is
    /// enforced while streaming.
    pub(crate) fn create_internal_file(
        &self,
        file_id: &str,
        filename: &str,
        purpose: FilePurpose,
        content: impl Read,
    ) -> Result<InternalFile> {
        if !purpose.is_internal() {
            return Err(Error::InvalidArgument(format!(
                "purpose {} is not internal",
                purpose.as_str()
            )));
        }
        validate_path_component(file_id)?;
        validate_filename(filename)?;
        let dirs = self.file_dirs()?;
        let mut conn = self.files_connection()?;
        let now = sql_int(crate::now()?)?;
        if let Some(found) = committed_internal(&conn, file_id, filename, purpose, now)? {
            return Ok(found);
        }
        let blob_id = stage_blob(&conn)?;
        let written = dirs.write_blob(&blob_id, |file| {
            copy_limited(content, file, purpose.max_file_bytes(), &mut |_: &[u8]| {})
        });
        let bytes = match written {
            Ok(bytes) => bytes,
            Err(error) => {
                discard_blob(&conn, &dirs, &blob_id);
                return Err(error);
            }
        };
        match commit_internal_file(&mut conn, &blob_id, file_id, filename, purpose, bytes) {
            Ok((file, true)) => Ok(file),
            Ok((found, false)) => {
                discard_blob(&conn, &dirs, &blob_id);
                Ok(found)
            }
            Err(error) => {
                discard_blob(&conn, &dirs, &blob_id);
                Err(error)
            }
        }
    }

    /// Creates a pending upload that expires after [`UPLOAD_TTL_SECONDS`].
    pub fn create_upload(&self, request: &CreateUpload) -> Result<UploadObject> {
        validate_external_purpose(request.purpose)?;
        validate_filename(&request.filename)?;
        validate_purpose_filename(request.purpose, &request.filename)?;
        validate_mime_type(&request.mime_type)?;
        let file_expires_after = validate_expires_after(request.purpose, request.expires_after)?;
        let max = request.purpose.max_upload_bytes();
        if request.bytes == 0 || request.bytes > max {
            return Err(Error::InvalidArgument(format!(
                "upload bytes must be between 1 and {max} for purpose {}",
                request.purpose.as_str()
            )));
        }
        let id = new_resource_id("upload")?;
        let created_at = crate::now()?;
        let expires_at = created_at.saturating_add(UPLOAD_TTL_SECONDS);
        let conn = self.files_connection()?;
        conn.execute(
            "INSERT INTO uploads (id, filename, purpose, mime_type, bytes, created_at, expires_at, \
                                  status, file_expires_after, file_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending', ?8, NULL)",
            params![
                id,
                request.filename,
                request.purpose.as_str(),
                request.mime_type,
                sql_int(request.bytes)?,
                sql_int(created_at)?,
                sql_int(expires_at)?,
                file_expires_after.map(sql_int).transpose()?,
            ],
        )?;
        Ok(UploadObject {
            id,
            bytes: request.bytes,
            created_at,
            expires_at,
            filename: request.filename.clone(),
            purpose: request.purpose,
            mime_type: request.mime_type.clone(),
            status: UploadStatus::Pending,
            file: None,
        })
    }

    /// Returns an upload; pending uploads past `expires_at` report `Expired`.
    pub fn get_upload(&self, upload_id: &str) -> Result<UploadObject> {
        let conn = self.files_connection()?;
        let now = crate::now()?;
        let row = upload_row(&conn, upload_id)?;
        upload_object(&conn, row, now)
    }

    /// Streams one part (at most [`MAX_UPLOAD_PART_BYTES`], and never beyond
    /// the upload's remaining declared bytes) into a pending upload.
    pub fn add_upload_part(&self, upload_id: &str, content: impl Read) -> Result<UploadPart> {
        let dirs = self.file_dirs()?;
        let mut conn = self.files_connection()?;
        let upload = upload_row(&conn, upload_id)?;
        ensure_mutable(&upload, crate::now()?)?;
        let declared = unsigned(upload.bytes)?;
        let remaining = declared.saturating_sub(parts_total(&conn, upload_id)?);
        if remaining == 0 {
            return Err(Error::Conflict(format!(
                "upload {upload_id} already holds all {declared} declared bytes"
            )));
        }
        let limit = MAX_UPLOAD_PART_BYTES.min(remaining);
        let part_id = new_resource_id("part")?;
        let blob_id = stage_blob(&conn)?;

        let written = dirs.write_blob(&blob_id, |file| {
            copy_limited(content, file, limit, &mut |_: &[u8]| {})
        });
        let bytes = match written {
            Ok(0) => {
                discard_blob(&conn, &dirs, &blob_id);
                return Err(Error::InvalidArgument("upload part is empty".to_owned()));
            }
            Ok(bytes) => bytes,
            Err(error) => {
                discard_blob(&conn, &dirs, &blob_id);
                return Err(error);
            }
        };
        let mut part = UploadPart {
            id: part_id,
            upload_id: upload_id.to_owned(),
            bytes,
            created_at: 0,
        };
        if let Err(error) = commit_new_part(&mut conn, &blob_id, &mut part, declared) {
            discard_blob(&conn, &dirs, &blob_id);
            return Err(error);
        }
        Ok(part)
    }

    /// Assembles the listed parts, in order, into a new file by streaming.
    ///
    /// Part ids must be unique, belong to this upload, and their sizes must sum
    /// exactly to the declared byte count. The terminal transition, file
    /// creation, and part release happen in one transaction.
    pub fn complete_upload(
        &self,
        upload_id: &str,
        request: &CompleteUpload,
    ) -> Result<UploadObject> {
        if request.part_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "part_ids must not be empty".to_owned(),
            ));
        }
        let mut seen = HashSet::with_capacity(request.part_ids.len());
        if let Some(duplicate) = request.part_ids.iter().find(|id| !seen.insert(id.as_str())) {
            return Err(Error::InvalidArgument(format!(
                "duplicate part id {duplicate}"
            )));
        }
        let expected_md5 = request.md5.as_deref().map(parse_md5).transpose()?;

        let dirs = self.file_dirs()?;
        let mut conn = self.files_connection()?;
        let upload = upload_row(&conn, upload_id)?;
        ensure_mutable(&upload, crate::now()?)?;
        let declared = unsigned(upload.bytes)?;
        let parts = resolve_parts(&conn, upload_id, &request.part_ids)?;
        let total = parts
            .iter()
            .try_fold(0_u64, |sum, part| sum.checked_add(part.bytes))
            .ok_or_else(|| Error::InvalidArgument("part sizes overflow".to_owned()))?;
        if total != declared {
            return Err(Error::InvalidArgument(format!(
                "parts total {total} bytes but upload {upload_id} declared {declared}"
            )));
        }

        let file_id = new_resource_id("file")?;
        let blob_id = stage_blob(&conn)?;
        let mut hasher = expected_md5.map(|_| md5::Context::new());
        let mut observe = |chunk: &[u8]| {
            if let Some(hasher) = hasher.as_mut() {
                hasher.consume(chunk);
            }
        };

        let written = dirs.write_blob(&blob_id, |file| {
            let mut copied = 0_u64;
            for part in &parts {
                let source = dirs.open_blob(&part.blob_id, part.bytes)?;
                copied =
                    copied.saturating_add(copy_limited(source, file, part.bytes, &mut observe)?);
            }
            Ok(copied)
        });
        match written {
            Ok(bytes) if bytes == declared => {}
            Ok(bytes) => {
                discard_blob(&conn, &dirs, &blob_id);
                return Err(Error::Conflict(format!(
                    "assembled {bytes} bytes but upload {upload_id} declared {declared}"
                )));
            }
            Err(error) => {
                discard_blob(&conn, &dirs, &blob_id);
                return Err(error);
            }
        }
        if let (Some(expected), Some(hasher)) = (expected_md5, hasher)
            && hasher.finalize().0 != expected
        {
            discard_blob(&conn, &dirs, &blob_id);
            return Err(Error::InvalidArgument(
                "MD5 checksum does not match the assembled bytes".to_owned(),
            ));
        }

        let file_expires_after = upload.file_expires_after.map(unsigned).transpose()?;
        let mut file = FileObject {
            id: file_id,
            bytes: declared,
            created_at: 0,
            expires_at: None,
            filename: upload.filename.clone(),
            purpose: upload.purpose.parse()?,
        };
        let released = match commit_completion(
            &mut conn,
            upload_id,
            &request.part_ids,
            &blob_id,
            &mut file,
            file_expires_after,
        ) {
            Ok(released) => released,
            Err(error) => {
                discard_blob(&conn, &dirs, &blob_id);
                return Err(error);
            }
        };
        for part_blob in &released {
            let _ = dirs.remove_blob(part_blob);
        }
        let row = upload_row(&conn, upload_id)?;
        upload_object(&conn, row, file.created_at)
    }

    /// Cancels a pending, unexpired upload and releases its parts.
    pub fn cancel_upload(&self, upload_id: &str) -> Result<UploadObject> {
        let dirs = self.file_dirs()?;
        let mut conn = self.files_connection()?;
        let tx = immediate(&mut conn)?;
        let now = crate::now()?;
        let upload = upload_row(&tx, upload_id)?;
        ensure_mutable(&upload, now)?;
        tx.execute(
            "UPDATE uploads SET status = 'cancelled' WHERE id = ?1 AND status = 'pending'",
            params![upload_id],
        )?;
        let released = release_parts(&tx, upload_id)?;
        tx.commit()?;
        for blob_id in &released {
            let _ = dirs.remove_blob(blob_id);
        }
        let row = upload_row(&conn, upload_id)?;
        upload_object(&conn, row, now)
    }

    /// Physically removes expired files, parts of expired uploads, abandoned
    /// staging blobs, and disk entries without metadata. Never runs
    /// automatically.
    pub fn purge_file_storage(&self) -> Result<FileStoragePurge> {
        let dirs = self.file_dirs()?;
        let mut conn = self.files_connection()?;
        let now = sql_int(crate::now()?)?;
        let stale_before = now.saturating_sub(sql_int(STAGING_TTL_SECONDS)?);
        let mut report = FileStoragePurge::default();

        let tx = immediate(&mut conn)?;
        let expired_files = query_strings(
            &tx,
            "SELECT blob_id FROM files WHERE expires_at IS NOT NULL AND expires_at <= ?1",
            now,
        )?;
        tx.execute(
            "DELETE FROM file_blobs WHERE id IN \
             (SELECT blob_id FROM files WHERE expires_at IS NOT NULL AND expires_at <= ?1)",
            params![now],
        )?;
        tx.execute(
            "DELETE FROM files WHERE expires_at IS NOT NULL AND expires_at <= ?1",
            params![now],
        )?;
        let expired_uploads = query_strings(
            &tx,
            "SELECT id FROM uploads WHERE status = 'pending' AND expires_at <= ?1",
            now,
        )?;
        let mut expired_parts = Vec::new();
        for upload_id in &expired_uploads {
            expired_parts.extend(release_parts(&tx, upload_id)?);
        }
        let stale = query_strings(
            &tx,
            "SELECT id FROM file_blobs WHERE state = 'staging' AND created_at <= ?1",
            stale_before,
        )?;
        tx.execute(
            "DELETE FROM file_blobs WHERE state = 'staging' AND created_at <= ?1",
            params![stale_before],
        )?;
        tx.commit()?;

        for blob_id in &expired_files {
            dirs.remove_blob(blob_id)?;
        }
        for blob_id in &expired_parts {
            dirs.remove_blob(blob_id)?;
        }
        for blob_id in &stale {
            dirs.remove_blob(blob_id)?;
        }
        report.expired_files = count(expired_files.len());
        report.expired_upload_parts = count(expired_parts.len());
        report.stale_staging_blobs = count(stale.len());
        report.orphan_entries = dirs.remove_orphans(&conn)?;
        Ok(report)
    }

    fn files_connection(&self) -> Result<Connection> {
        self.connection()
    }

    fn file_dirs(&self) -> Result<Dirs> {
        let base = self.root().join(STORAGE_DIR);
        ensure_private_dir(&base)?;
        let blobs = base.join(BLOB_DIR);
        ensure_private_dir(&blobs)?;
        let tmp = base.join(TMP_DIR);
        ensure_private_dir(&tmp)?;
        Ok(Dirs { blobs, tmp })
    }
}

const FILE_COLUMNS: &str = "id, blob_id, filename, purpose, bytes, created_at, expires_at";
const UPLOAD_COLUMNS: &str = "id, filename, purpose, mime_type, bytes, created_at, expires_at, \
                              status, file_expires_after, file_id";

pub(crate) fn initialize(connection: &mut Connection) -> Result<()> {
    crate::migrate(connection, "files", &[SCHEMA, INTERNAL_FILES])
}

struct FileRow {
    id: String,
    blob_id: String,
    filename: String,
    purpose: String,
    bytes: i64,
    created_at: i64,
    expires_at: Option<i64>,
}

impl FileRow {
    fn into_object(self) -> Result<FileObject> {
        Ok(FileObject {
            id: self.id,
            bytes: unsigned(self.bytes)?,
            created_at: unsigned(self.created_at)?,
            expires_at: self.expires_at.map(unsigned).transpose()?,
            filename: self.filename,
            purpose: self.purpose.parse()?,
        })
    }
}

fn read_file_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileRow> {
    Ok(FileRow {
        id: row.get(0)?,
        blob_id: row.get(1)?,
        filename: row.get(2)?,
        purpose: row.get(3)?,
        bytes: row.get(4)?,
        created_at: row.get(5)?,
        expires_at: row.get(6)?,
    })
}

struct UploadRow {
    id: String,
    filename: String,
    purpose: String,
    mime_type: String,
    bytes: i64,
    created_at: i64,
    expires_at: i64,
    status: String,
    file_expires_after: Option<i64>,
    file_id: Option<String>,
}

struct PartRow {
    blob_id: String,
    bytes: u64,
}

fn immediate(conn: &mut Connection) -> Result<Transaction<'_>> {
    Ok(conn.transaction_with_behavior(TransactionBehavior::Immediate)?)
}

fn live_file_row(conn: &Connection, file_id: &str, now: i64) -> Result<Option<FileRow>> {
    Ok(conn
        .query_row(
            &format!(
                "SELECT {FILE_COLUMNS} FROM files \
                 WHERE id = ?1 AND (expires_at IS NULL OR expires_at > ?2)"
            ),
            params![file_id, now],
            read_file_row,
        )
        .optional()?)
}

fn upload_row(conn: &Connection, upload_id: &str) -> Result<UploadRow> {
    conn.query_row(
        &format!("SELECT {UPLOAD_COLUMNS} FROM uploads WHERE id = ?1"),
        params![upload_id],
        |row| {
            Ok(UploadRow {
                id: row.get(0)?,
                filename: row.get(1)?,
                purpose: row.get(2)?,
                mime_type: row.get(3)?,
                bytes: row.get(4)?,
                created_at: row.get(5)?,
                expires_at: row.get(6)?,
                status: row.get(7)?,
                file_expires_after: row.get(8)?,
                file_id: row.get(9)?,
            })
        },
    )
    .optional()?
    .ok_or_else(|| Error::NotFound(format!("upload {upload_id} not found")))
}

fn upload_status(row: &UploadRow, now: u64) -> Result<UploadStatus> {
    match row.status.as_str() {
        "pending" if now >= unsigned(row.expires_at)? => Ok(UploadStatus::Expired),
        "pending" => Ok(UploadStatus::Pending),
        "completed" => Ok(UploadStatus::Completed),
        "cancelled" => Ok(UploadStatus::Cancelled),
        other => Err(Error::Conflict(format!(
            "unknown stored upload status {other:?}"
        ))),
    }
}

fn ensure_mutable(row: &UploadRow, now: u64) -> Result<()> {
    match upload_status(row, now)? {
        UploadStatus::Pending => Ok(()),
        UploadStatus::Expired => Err(Error::Conflict(format!("upload {} has expired", row.id))),
        UploadStatus::Completed => Err(Error::Conflict(format!(
            "upload {} is already completed",
            row.id
        ))),
        UploadStatus::Cancelled => Err(Error::Conflict(format!(
            "upload {} is already cancelled",
            row.id
        ))),
    }
}

fn upload_object(conn: &Connection, row: UploadRow, now: u64) -> Result<UploadObject> {
    let status = upload_status(&row, now)?;
    let file = match (&row.file_id, status) {
        (Some(file_id), UploadStatus::Completed) => live_file_row(conn, file_id, sql_int(now)?)?
            .map(FileRow::into_object)
            .transpose()?,
        _ => None,
    };
    Ok(UploadObject {
        bytes: unsigned(row.bytes)?,
        created_at: unsigned(row.created_at)?,
        expires_at: unsigned(row.expires_at)?,
        purpose: row.purpose.parse()?,
        id: row.id,
        filename: row.filename,
        mime_type: row.mime_type,
        status,
        file,
    })
}

fn parts_total(conn: &Connection, upload_id: &str) -> Result<u64> {
    let total: i64 = conn.query_row(
        "SELECT COALESCE(SUM(bytes), 0) FROM upload_parts WHERE upload_id = ?1",
        params![upload_id],
        |row| row.get(0),
    )?;
    unsigned(total)
}

fn resolve_parts(conn: &Connection, upload_id: &str, part_ids: &[String]) -> Result<Vec<PartRow>> {
    let mut statement =
        conn.prepare("SELECT blob_id, bytes FROM upload_parts WHERE id = ?1 AND upload_id = ?2")?;
    part_ids
        .iter()
        .map(|part_id| {
            let (blob_id, bytes) = statement
                .query_row(params![part_id, upload_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })
                .optional()?
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "part {part_id} does not belong to upload {upload_id}"
                    ))
                })?;
            Ok(PartRow {
                blob_id,
                bytes: unsigned(bytes)?,
            })
        })
        .collect()
}

/// Deletes all part rows (and their blob rows) of an upload, returning the
/// blob ids whose bytes the caller must unlink after commit.
fn release_parts(tx: &Transaction<'_>, upload_id: &str) -> Result<Vec<String>> {
    let mut statement = tx.prepare("SELECT blob_id FROM upload_parts WHERE upload_id = ?1")?;
    let blobs = statement
        .query_map(params![upload_id], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    tx.execute(
        "DELETE FROM file_blobs WHERE id IN (SELECT blob_id FROM upload_parts WHERE upload_id = ?1)",
        params![upload_id],
    )?;
    tx.execute(
        "DELETE FROM upload_parts WHERE upload_id = ?1",
        params![upload_id],
    )?;
    Ok(blobs)
}

fn query_strings(conn: &Connection, sql: &str, value: i64) -> Result<Vec<String>> {
    let mut statement = conn.prepare(sql)?;
    let values = statement
        .query_map(params![value], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(values)
}

fn stage_blob(conn: &Connection) -> Result<String> {
    let id = crate::new_id("blob")?;
    validate_path_component(&id)?;
    conn.execute(
        "INSERT INTO file_blobs (id, state, created_at) VALUES (?1, 'staging', ?2)",
        params![id, sql_int(crate::now()?)?],
    )?;
    Ok(id)
}

fn commit_blob(tx: &Transaction<'_>, blob_id: &str) -> Result<()> {
    let updated = tx.execute(
        "UPDATE file_blobs SET state = 'committed' WHERE id = ?1 AND state = 'staging'",
        params![blob_id],
    )?;
    if updated == 1 {
        Ok(())
    } else {
        Err(Error::Conflict(format!(
            "staged blob {blob_id} is no longer present (purged as abandoned?)"
        )))
    }
}

fn insert_file_row(tx: &Transaction<'_>, blob_id: &str, file: &FileObject) -> Result<()> {
    tx.execute(
        "INSERT INTO files (id, blob_id, filename, purpose, bytes, created_at, expires_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            file.id,
            blob_id,
            file.filename,
            file.purpose.as_str(),
            sql_int(file.bytes)?,
            sql_int(file.created_at)?,
            file.expires_at.map(sql_int).transpose()?,
        ],
    )?;
    Ok(())
}

fn commit_new_file(conn: &mut Connection, blob_id: &str, file: &FileObject) -> Result<()> {
    let tx = immediate(conn)?;
    commit_blob(&tx, blob_id)?;
    insert_file_row(&tx, blob_id, file)?;
    tx.commit()?;
    Ok(())
}

/// The outcome of an earlier commit of internal file `file_id`, or `None`
/// when nothing was ever committed under it.
fn committed_internal(
    conn: &Connection,
    file_id: &str,
    filename: &str,
    purpose: FilePurpose,
    now: i64,
) -> Result<Option<InternalFile>> {
    let recorded: Option<(String, String)> = conn
        .query_row(
            "SELECT purpose, filename FROM internal_files WHERE id = ?1",
            params![file_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    match recorded {
        Some((stored_purpose, stored_filename))
            if stored_purpose == purpose.as_str() && stored_filename == filename =>
        {
            Ok(Some(
                live_file_row(conn, file_id, now)?
                    .map(FileRow::into_object)
                    .transpose()?
                    .map_or(InternalFile::Gone, InternalFile::Live),
            ))
        }
        Some(_) => Err(Error::Conflict(format!(
            "internal file {file_id} was created with another purpose or filename"
        ))),
        None => {
            let taken = conn
                .prepare("SELECT 1 FROM files WHERE id = ?1")?
                .exists(params![file_id])?;
            if taken {
                Err(Error::Conflict(format!(
                    "file id {file_id} is already taken"
                )))
            } else {
                Ok(None)
            }
        }
    }
}

/// Commits a staged internal blob as file `file_id`; `false` with the earlier
/// outcome when another attempt committed that id first (the caller then
/// discards its blob).
fn commit_internal_file(
    conn: &mut Connection,
    blob_id: &str,
    file_id: &str,
    filename: &str,
    purpose: FilePurpose,
    bytes: u64,
) -> Result<(InternalFile, bool)> {
    let tx = immediate(conn)?;
    // The lifetime starts at the commit, read under the write lock.
    let created_at = crate::now()?;
    if let Some(found) = committed_internal(&tx, file_id, filename, purpose, sql_int(created_at)?)?
    {
        return Ok((found, false));
    }
    let file = FileObject {
        id: file_id.to_owned(),
        bytes,
        created_at,
        expires_at: purpose
            .default_expires_after()
            .map(|seconds| created_at.saturating_add(seconds)),
        filename: filename.to_owned(),
        purpose,
    };
    commit_blob(&tx, blob_id)?;
    insert_file_row(&tx, blob_id, &file)?;
    tx.execute(
        "INSERT INTO internal_files (id, purpose, filename, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![file_id, purpose.as_str(), filename, sql_int(created_at)?],
    )?;
    tx.commit()?;
    Ok((InternalFile::Live(file), true))
}

fn commit_new_part(
    conn: &mut Connection,
    blob_id: &str,
    part: &mut UploadPart,
    declared: u64,
) -> Result<()> {
    let tx = immediate(conn)?;
    // The terminal/expiry check uses a clock read under the write lock: the
    // stream and any lock wait may have outlived the upload.
    part.created_at = crate::now()?;
    let upload = upload_row(&tx, &part.upload_id)?;
    ensure_mutable(&upload, part.created_at)?;
    let used = parts_total(&tx, &part.upload_id)?;
    if used.saturating_add(part.bytes) > declared {
        return Err(Error::Conflict(format!(
            "part would exceed the {declared} bytes declared for upload {}",
            part.upload_id
        )));
    }
    commit_blob(&tx, blob_id)?;
    tx.execute(
        "INSERT INTO upload_parts (id, upload_id, blob_id, bytes, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            part.id,
            part.upload_id,
            blob_id,
            sql_int(part.bytes)?,
            sql_int(part.created_at)?
        ],
    )?;
    tx.commit()?;
    Ok(())
}

fn commit_completion(
    conn: &mut Connection,
    upload_id: &str,
    part_ids: &[String],
    blob_id: &str,
    file: &mut FileObject,
    file_expires_after: Option<u64>,
) -> Result<Vec<String>> {
    let tx = immediate(conn)?;
    // As for parts: assembly and the lock wait may have outlived the upload.
    let created_at = crate::now()?;
    file.created_at = created_at;
    file.expires_at = file_expires_after.map(|seconds| created_at.saturating_add(seconds));
    let upload = upload_row(&tx, upload_id)?;
    ensure_mutable(&upload, file.created_at)?;
    let parts = resolve_parts(&tx, upload_id, part_ids)?;
    let total = parts
        .iter()
        .map(|part| part.bytes)
        .fold(0_u64, u64::saturating_add);
    if total != file.bytes {
        return Err(Error::Conflict(format!(
            "parts of upload {upload_id} changed during completion"
        )));
    }
    commit_blob(&tx, blob_id)?;
    insert_file_row(&tx, blob_id, file)?;
    tx.execute(
        "UPDATE uploads SET status = 'completed', file_id = ?2 WHERE id = ?1 AND status = 'pending'",
        params![upload_id, file.id],
    )?;
    let released = release_parts(&tx, upload_id)?;
    tx.commit()?;
    Ok(released)
}

/// Remove only a definitely uncommitted blob. A reported commit error can have
/// an unknown outcome; never unlink bytes if the row is already committed or
/// the database cannot establish its state.
fn discard_blob(conn: &Connection, dirs: &Dirs, blob_id: &str) {
    if matches!(
        conn.execute(
            "DELETE FROM file_blobs WHERE id = ?1 AND state = 'staging'",
            params![blob_id],
        ),
        Ok(1)
    ) {
        let _ = dirs.remove_blob(blob_id);
    }
}

struct Dirs {
    blobs: PathBuf,
    tmp: PathBuf,
}

impl Dirs {
    fn blob_path(&self, blob_id: &str) -> PathBuf {
        self.blobs.join(blob_id)
    }

    fn tmp_path(&self, blob_id: &str) -> PathBuf {
        self.tmp.join(format!("{blob_id}{TMP_SUFFIX}"))
    }

    /// Writes through a fresh owner-only temp file, syncs it, and renames it
    /// into place. Never overwrites an existing blob.
    fn write_blob(
        &self,
        blob_id: &str,
        write: impl FnOnce(&mut File) -> Result<u64>,
    ) -> Result<u64> {
        let tmp = self.tmp_path(blob_id);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        let written = write(&mut file).and_then(|bytes| {
            file.sync_all()?;
            Ok(bytes)
        });
        drop(file);
        let result = written.and_then(|bytes| {
            let target = self.blob_path(blob_id);
            match fs::symlink_metadata(&target) {
                Ok(_) => {
                    return Err(Error::Conflict(format!("blob {blob_id} already exists")));
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            fs::rename(&tmp, &target)?;
            sync_dir(&self.blobs)?;
            Ok(bytes)
        });
        if result.is_err() {
            let _ = remove_if_exists(&tmp);
        }
        result
    }

    /// Opens a blob read-only, refusing symlinks, non-regular files, files
    /// swapped between check and open, and unexpected sizes.
    fn open_blob(&self, blob_id: &str, expected_bytes: u64) -> Result<File> {
        let path = self.blob_path(blob_id);
        let before = fs::symlink_metadata(&path)?;
        if !before.file_type().is_file() {
            return Err(Error::Conflict(format!(
                "refusing to open blob {blob_id}: not a regular file"
            )));
        }
        let file = File::open(&path)?;
        let after = file.metadata()?;
        if !after.is_file() || after.dev() != before.dev() || after.ino() != before.ino() {
            return Err(Error::Conflict(format!(
                "blob {blob_id} changed while opening"
            )));
        }
        if after.len() != expected_bytes {
            return Err(Error::Conflict(format!(
                "blob {blob_id} holds {} bytes, expected {expected_bytes}",
                after.len()
            )));
        }
        Ok(file)
    }

    fn remove_blob(&self, blob_id: &str) -> Result<()> {
        remove_if_exists(&self.blob_path(blob_id))?;
        remove_if_exists(&self.tmp_path(blob_id))?;
        Ok(())
    }

    /// Removes entries that have no `file_blobs` row. Safe against concurrent
    /// writers because rows are always committed before disk entries exist.
    fn remove_orphans(&self, conn: &Connection) -> Result<u64> {
        let mut statement = conn.prepare("SELECT 1 FROM file_blobs WHERE id = ?1")?;
        let mut removed = 0;
        for (dir, suffix) in [(&self.blobs, ""), (&self.tmp, TMP_SUFFIX)] {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    continue;
                }
                let name = entry.file_name();
                let blob_id = name
                    .to_str()
                    .and_then(|name| name.strip_suffix(suffix))
                    .filter(|id| validate_path_component(id).is_ok());
                let known = match blob_id {
                    Some(id) => statement.exists(params![id])?,
                    None => false,
                };
                if !known {
                    remove_if_exists(&entry.path())?;
                    removed += 1;
                }
            }
        }
        Ok(removed)
    }
}

/// Creates `path` owner-only if missing, then applies the same check as the
/// store root: a symlink, a non-directory, or any group/other permission bit
/// is refused ([`io::ErrorKind::PermissionDenied`]) rather than repaired, so
/// existing caller data is never modified.
fn ensure_private_dir(path: &Path) -> Result<()> {
    match fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    crate::check_private_dir(path)
}

fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

/// Copies at most `limit` bytes; reading one byte more is an error.
fn copy_limited(
    mut reader: impl Read,
    writer: &mut File,
    limit: u64,
    observe: &mut dyn FnMut(&[u8]),
) -> Result<u64> {
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut total = 0_u64;
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => return Ok(total),
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        };
        total = total.saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
        if total > limit {
            return Err(Error::InvalidArgument(format!(
                "content exceeds the {limit}-byte limit"
            )));
        }
        let chunk = &buffer[..read];
        writer.write_all(chunk)?;
        observe(chunk);
    }
}

fn validate_filename(filename: &str) -> Result<()> {
    if filename.is_empty() || filename.len() > MAX_FILENAME_BYTES {
        return Err(Error::InvalidArgument(format!(
            "filename must be 1 to {MAX_FILENAME_BYTES} bytes"
        )));
    }
    if filename.chars().any(char::is_control) {
        return Err(Error::InvalidArgument(
            "filename must not contain control characters".to_owned(),
        ));
    }
    Ok(())
}

fn validate_purpose_filename(purpose: FilePurpose, filename: &str) -> Result<()> {
    let is_jsonl = filename
        .rsplit_once('.')
        .is_some_and(|(_, extension)| extension.eq_ignore_ascii_case("jsonl"));
    if purpose == FilePurpose::Batch && !is_jsonl {
        return Err(Error::InvalidArgument(
            "batch files must be .jsonl".to_owned(),
        ));
    }
    Ok(())
}

fn validate_external_purpose(purpose: FilePurpose) -> Result<()> {
    if purpose.is_internal() {
        return Err(Error::InvalidArgument(format!(
            "purpose {} is reserved for files the server creates",
            purpose.as_str()
        )));
    }
    Ok(())
}

fn validate_mime_type(mime_type: &str) -> Result<()> {
    let valid = !mime_type.is_empty()
        && mime_type.len() <= MAX_MIME_TYPE_BYTES
        && mime_type.contains('/')
        && mime_type.chars().all(|c| c.is_ascii_graphic() || c == ' ');
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "invalid mime_type {mime_type:?}"
        )))
    }
}

fn validate_expires_after(
    purpose: FilePurpose,
    expires_after: Option<ExpiresAfter>,
) -> Result<Option<u64>> {
    match expires_after {
        None => Ok(purpose.default_expires_after()),
        Some(ExpiresAfter { seconds, .. })
            if (MIN_EXPIRES_AFTER_SECONDS..=MAX_EXPIRES_AFTER_SECONDS).contains(&seconds) =>
        {
            Ok(Some(seconds))
        }
        Some(_) => Err(Error::InvalidArgument(format!(
            "expires_after.seconds must be between {MIN_EXPIRES_AFTER_SECONDS} and \
             {MAX_EXPIRES_AFTER_SECONDS}"
        ))),
    }
}

fn validate_path_component(id: &str) -> Result<()> {
    let valid = !id.is_empty()
        && id.len() <= MAX_ID_BYTES
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!("unsafe internal id {id:?}")))
    }
}

fn new_resource_id(prefix: &str) -> Result<String> {
    crate::new_id(prefix)
}

fn parse_md5(value: &str) -> Result<[u8; 16]> {
    let invalid = || Error::InvalidArgument("md5 must be 32 hexadecimal characters".to_owned());
    let bytes = value.as_bytes();
    if bytes.len() != 32 {
        return Err(invalid());
    }
    let mut digest = [0_u8; 16];
    let (pairs, _) = bytes.as_chunks::<2>();
    for (slot, [high, low]) in digest.iter_mut().zip(pairs) {
        let high = hex_value(*high).ok_or_else(invalid)?;
        let low = hex_value(*low).ok_or_else(invalid)?;
        *slot = (high << 4) | low;
    }
    Ok(digest)
}

const fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn sql_int(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("value {value} is out of range")))
}

fn unsigned(value: i64) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::Conflict(format!("stored value {value} is negative")))
}

fn count(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

fn file_not_found(file_id: &str) -> Error {
    Error::NotFound(format!("file {file_id} not found"))
}
