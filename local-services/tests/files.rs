#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Files and multipart Uploads persistence, using small disposable data.

use std::fs;
use std::io::{self, Read};
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use local_services::files::{
    CompleteUpload, CreateFile, CreateUpload, DEFAULT_BATCH_EXPIRES_AFTER_SECONDS, ExpiresAfter,
    ExpiresAfterAnchor, FilePurpose, FileStoragePurge, ListFiles, SortOrder, UPLOAD_TTL_SECONDS,
    UploadStatus,
};
use local_services::{Error, Store};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "local-services-files-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn store_path(&self) -> PathBuf {
        self.0.join("store")
    }

    fn open(&self) -> Store {
        Store::open(self.store_path()).unwrap()
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn request(filename: &str, purpose: FilePurpose) -> CreateFile {
    CreateFile {
        filename: filename.to_owned(),
        purpose,
        expires_after: None,
    }
}

fn upload_request(bytes: u64) -> CreateUpload {
    CreateUpload {
        filename: "data.bin".to_owned(),
        purpose: FilePurpose::UserData,
        bytes,
        mime_type: "application/octet-stream".to_owned(),
        expires_after: None,
    }
}

fn read_all(store: &Store, file_id: &str) -> Vec<u8> {
    let mut content = store.open_file_content(file_id).unwrap();
    let mut bytes = Vec::new();
    content.content.read_to_end(&mut bytes).unwrap();
    assert_eq!(content.file.id, file_id);
    bytes
}

fn entries(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir).map_or_else(
        |_| Vec::new(),
        |entries| entries.map(|entry| entry.unwrap().path()).collect(),
    )
}

fn assert_empty<T: std::fmt::Debug>(items: &[T]) {
    assert_eq!(items.len(), 0, "expected no items, found {items:?}");
}

fn blobs_dir(store: &Store) -> PathBuf {
    store.root().join("file-store").join("blobs")
}

fn tmp_dir(store: &Store) -> PathBuf {
    store.root().join("file-store").join("tmp")
}

/// Opens the store's SQLite database directly to simulate the passage of time.
fn database(store: &Store) -> rusqlite::Connection {
    fn find(dir: &Path, depth: u32) -> Option<PathBuf> {
        for path in entries(dir) {
            let meta = fs::symlink_metadata(&path).ok()?;
            if meta.is_dir() {
                if depth > 0
                    && !path.ends_with("file-store")
                    && let Some(found) = find(&path, depth - 1)
                {
                    return Some(found);
                }
            } else if meta.is_file() {
                let mut header = [0_u8; 16];
                let is_sqlite = fs::File::open(&path)
                    .and_then(|mut file| file.read_exact(&mut header))
                    .is_ok()
                    && &header == b"SQLite format 3\0";
                if is_sqlite {
                    return Some(path);
                }
            }
        }
        None
    }
    let path = find(store.root(), 2).expect("store database");
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.busy_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    conn
}

#[test]
fn binary_roundtrip_survives_reopen_and_never_uses_filename_as_path() {
    let temp = TempRoot::new();
    let payload: Vec<u8> = (0..4096_u32).map(|i| (i * 31 % 256) as u8).collect();
    let hostile = "../../escape/\u{e9}vil.bin";
    let file = {
        let store = temp.open();
        let file = store
            .create_file(&request(hostile, FilePurpose::UserData), payload.as_slice())
            .unwrap();
        assert_eq!(file.bytes, 4096);
        assert_eq!(file.filename, hostile);
        assert_eq!(file.expires_at, None);
        assert_eq!(read_all(&store, &file.id), payload);
        file
    };

    let store = temp.open();
    assert_eq!(store.get_file(&file.id).unwrap(), file);
    assert_eq!(read_all(&store, &file.id), payload);

    assert!(!temp.0.join("escape").exists());
    assert!(!store.root().parent().unwrap().join("escape").exists());
    let blobs = entries(&blobs_dir(&store));
    assert_eq!(blobs.len(), 1);
    let blob_name = blobs[0].file_name().unwrap().to_str().unwrap().to_owned();
    assert!(!blob_name.contains("vil"));
    assert_eq!(
        fs::metadata(&blobs[0]).unwrap().permissions().mode() & 0o777,
        0o600
    );
    for dir in ["file-store", "file-store/blobs", "file-store/tmp"] {
        let mode = fs::metadata(store.root().join(dir))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700, "{dir}");
    }
    assert_empty(&entries(&tmp_dir(&store)));
}

#[test]
fn validates_requests_and_expiry() {
    let temp = TempRoot::new();
    let store = temp.open();
    let invalid = |request: &CreateFile, data: &[u8]| {
        matches!(
            store.create_file(request, data),
            Err(Error::InvalidArgument(_))
        )
    };
    assert!(invalid(&request("", FilePurpose::UserData), b"x"));
    assert!(invalid(&request("a\nb", FilePurpose::UserData), b"x"));
    assert!(invalid(&request("batch.json", FilePurpose::Batch), b"{}"));
    assert!(invalid(&request("empty.txt", FilePurpose::UserData), b""));
    for seconds in [3599, 2_592_001] {
        let mut req = request("a.txt", FilePurpose::UserData);
        req.expires_after = Some(ExpiresAfter {
            anchor: ExpiresAfterAnchor::CreatedAt,
            seconds,
        });
        assert!(invalid(&req, b"x"));
    }
    assert!(matches!(
        "bogus".parse::<FilePurpose>(),
        Err(Error::InvalidArgument(_))
    ));
    // `batch_output` is internal: it parses (to filter listings) but no
    // caller may create a file or an upload with it.
    assert_eq!(
        "batch_output".parse::<FilePurpose>().unwrap(),
        FilePurpose::BatchOutput
    );
    assert!(invalid(
        &request("out.jsonl", FilePurpose::BatchOutput),
        b"{}\n"
    ));
    let mut internal_upload = upload_request(1);
    internal_upload.purpose = FilePurpose::BatchOutput;
    assert!(matches!(
        store.create_upload(&internal_upload),
        Err(Error::InvalidArgument(_))
    ));
    let batch_outputs = ListFiles {
        purpose: Some(FilePurpose::BatchOutput),
        ..ListFiles::default()
    };
    assert_empty(&store.list_files(&batch_outputs).unwrap().data);
    assert_empty(&store.list_files(&ListFiles::default()).unwrap().data);

    let batch = store
        .create_file(&request("in.JSONL", FilePurpose::Batch), &b"{}\n"[..])
        .unwrap();
    assert_eq!(
        batch.expires_at,
        Some(batch.created_at + DEFAULT_BATCH_EXPIRES_AFTER_SECONDS)
    );
    let mut req = request("a.txt", FilePurpose::Vision);
    req.expires_after = Some(ExpiresAfter {
        anchor: ExpiresAfterAnchor::CreatedAt,
        seconds: 3600,
    });
    let file = store.create_file(&req, &b"x"[..]).unwrap();
    assert_eq!(file.expires_at, Some(file.created_at + 3600));

    let mut big = upload_request(8 * 1024 * 1024 * 1024 + 1);
    assert!(matches!(
        store.create_upload(&big),
        Err(Error::InvalidArgument(_))
    ));
    big.bytes = 0;
    assert!(matches!(
        store.create_upload(&big),
        Err(Error::InvalidArgument(_))
    ));
    let batch_upload = CreateUpload {
        filename: "in.jsonl".to_owned(),
        purpose: FilePurpose::Batch,
        bytes: 200 * 1024 * 1024 + 1,
        mime_type: "application/jsonl".to_owned(),
        expires_after: None,
    };
    assert!(matches!(
        store.create_upload(&batch_upload),
        Err(Error::InvalidArgument(_))
    ));
    let mut bad_mime = upload_request(1);
    bad_mime.mime_type = "nonsense".to_owned();
    assert!(matches!(
        store.create_upload(&bad_mime),
        Err(Error::InvalidArgument(_))
    ));
    let large = store
        .create_upload(&upload_request(8 * 1024 * 1024 * 1024))
        .unwrap();
    assert_eq!(large.expires_at, large.created_at + UPLOAD_TTL_SECONDS);
}

#[test]
fn streaming_limits_stop_reading_and_leave_no_leftovers() {
    let temp = TempRoot::new();
    let store = temp.open();
    let upload = store.create_upload(&upload_request(10)).unwrap();
    // An endless reader must be cut off at the remaining declared bytes.
    let err = store
        .add_upload_part(&upload.id, io::repeat(7))
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    assert!(matches!(
        store.add_upload_part(&upload.id, io::empty()),
        Err(Error::InvalidArgument(_))
    ));
    assert_empty(&entries(&blobs_dir(&store)));
    assert_empty(&entries(&tmp_dir(&store)));
    assert_eq!(store.purge_file_storage().unwrap().stale_staging_blobs, 0);

    store.add_upload_part(&upload.id, &[1_u8; 10][..]).unwrap();
    assert!(matches!(
        store.add_upload_part(&upload.id, &[1_u8][..]),
        Err(Error::Conflict(_))
    ));
}

#[test]
fn refuses_symlinked_blobs_and_storage_dirs() {
    let temp = TempRoot::new();
    let store = temp.open();
    let file = store
        .create_file(&request("a.bin", FilePurpose::UserData), &b"real"[..])
        .unwrap();
    let blob = entries(&blobs_dir(&store)).remove(0);
    let decoy = temp.0.join("decoy");
    fs::write(&decoy, b"fake").unwrap();
    fs::remove_file(&blob).unwrap();
    symlink(&decoy, &blob).unwrap();
    assert!(matches!(
        store.open_file_content(&file.id),
        Err(Error::Conflict(_))
    ));

    let temp = TempRoot::new();
    let store = temp.open();
    let elsewhere = temp.0.join("elsewhere");
    fs::create_dir(&elsewhere).unwrap();
    symlink(&elsewhere, store.root().join("file-store")).unwrap();
    assert!(refused(&store.create_file(
        &request("a.bin", FilePurpose::UserData),
        &b"x"[..]
    )));
    assert_empty(&entries(&elsewhere));
}

fn refused<T: std::fmt::Debug>(result: &Result<T, Error>) -> bool {
    matches!(result, Err(Error::Io(error)) if error.kind() == io::ErrorKind::PermissionDenied)
}

#[test]
fn refuses_insecure_storage_dirs_without_modifying_them() {
    let temp = TempRoot::new();
    let store = temp.open();
    let file = store
        .create_file(&request("a.bin", FilePurpose::UserData), &b"data"[..])
        .unwrap();
    for dir in ["file-store", "file-store/blobs", "file-store/tmp"] {
        let path = store.root().join(dir);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o750)).unwrap();
        assert!(refused(&store.open_file_content(&file.id)), "{dir}");
        assert!(refused(&store.purge_file_storage()), "{dir}");
        let mode = fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o750, "{dir} must be left as found");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    assert_eq!(read_all(&store, &file.id), b"data");
    assert_eq!(entries(&blobs_dir(&store)).len(), 1);
}

#[test]
fn lists_with_cursor_order_purpose_and_delete() {
    let temp = TempRoot::new();
    let store = temp.open();
    let purposes = [
        FilePurpose::UserData,
        FilePurpose::Assistants,
        FilePurpose::UserData,
        FilePurpose::Evals,
        FilePurpose::UserData,
    ];
    let ids: Vec<String> = purposes
        .iter()
        .enumerate()
        .map(|(i, purpose)| {
            store
                .create_file(&request(&format!("f{i}.txt"), *purpose), &[i as u8][..])
                .unwrap()
                .id
        })
        .collect();
    let ids_of = |query: &ListFiles| -> (Vec<String>, bool) {
        let page = store.list_files(query).unwrap();
        assert_eq!(page.first_id, page.data.first().map(|f| f.id.clone()));
        assert_eq!(page.last_id, page.data.last().map(|f| f.id.clone()));
        (page.data.into_iter().map(|f| f.id).collect(), page.has_more)
    };

    let mut desc: Vec<String> = ids.iter().rev().cloned().collect();
    assert_eq!(ids_of(&ListFiles::default()), (desc.clone(), false));

    let mut query = ListFiles {
        limit: Some(2),
        ..ListFiles::default()
    };
    assert_eq!(ids_of(&query), (desc[0..2].to_vec(), true));
    query.after = Some(desc[1].clone());
    assert_eq!(ids_of(&query), (desc[2..4].to_vec(), true));
    query.after = Some(desc[3].clone());
    assert_eq!(ids_of(&query), (desc[4..].to_vec(), false));

    let asc = ListFiles {
        order: SortOrder::Asc,
        after: Some(ids[1].clone()),
        ..ListFiles::default()
    };
    assert_eq!(ids_of(&asc), (ids[2..].to_vec(), false));

    let user_data = ListFiles {
        purpose: Some(FilePurpose::UserData),
        ..ListFiles::default()
    };
    assert_eq!(
        ids_of(&user_data).0,
        vec![ids[4].clone(), ids[2].clone(), ids[0].clone()]
    );

    for limit in [0, 10_001] {
        let query = ListFiles {
            limit: Some(limit),
            ..ListFiles::default()
        };
        assert!(matches!(
            store.list_files(&query),
            Err(Error::InvalidArgument(_))
        ));
    }

    let deleted = store.delete_file(&ids[2]).unwrap();
    assert!(deleted.deleted);
    assert_eq!(deleted.id, ids[2]);
    assert!(matches!(store.get_file(&ids[2]), Err(Error::NotFound(_))));
    assert!(matches!(
        store.open_file_content(&ids[2]),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        store.delete_file(&ids[2]),
        Err(Error::NotFound(_))
    ));
    desc.retain(|id| id != &ids[2]);
    assert_eq!(ids_of(&ListFiles::default()).0, desc);
    assert_eq!(entries(&blobs_dir(&store)).len(), 4);
}

#[test]
fn completes_parts_added_in_reverse_order() {
    let temp = TempRoot::new();
    let store = temp.open();
    let upload = store.create_upload(&upload_request(10)).unwrap();
    assert_eq!(upload.status, UploadStatus::Pending);
    let tail = store.add_upload_part(&upload.id, &b"ij"[..]).unwrap();
    let middle = store.add_upload_part(&upload.id, &b"efgh"[..]).unwrap();
    let head = store.add_upload_part(&upload.id, &b"abcd"[..]).unwrap();
    assert_eq!((head.bytes, middle.bytes, tail.bytes), (4, 4, 2));
    assert_eq!(head.upload_id, upload.id);

    let done = store
        .complete_upload(
            &upload.id,
            &CompleteUpload {
                part_ids: vec![head.id, middle.id, tail.id.clone()],
                md5: None,
            },
        )
        .unwrap();
    assert_eq!(done.status, UploadStatus::Completed);
    let file = done.file.clone().unwrap();
    assert_eq!((file.bytes, file.filename.as_str()), (10, "data.bin"));
    assert_eq!(read_all(&store, &file.id), b"abcdefghij");
    assert_eq!(store.get_upload(&upload.id).unwrap(), done);
    assert_eq!(entries(&blobs_dir(&store)).len(), 1, "part blobs released");

    assert!(matches!(
        store.add_upload_part(&upload.id, &b"x"[..]),
        Err(Error::Conflict(_))
    ));
    assert!(matches!(
        store.complete_upload(
            &upload.id,
            &CompleteUpload {
                part_ids: vec![tail.id],
                md5: None
            }
        ),
        Err(Error::Conflict(_))
    ));
    assert!(matches!(
        store.cancel_upload(&upload.id),
        Err(Error::Conflict(_))
    ));
    assert!(matches!(
        store.get_upload("upload_missing"),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn rejects_wrong_sizes_duplicates_and_foreign_parts() {
    let temp = TempRoot::new();
    let store = temp.open();
    let upload = store.create_upload(&upload_request(6)).unwrap();
    let other = store.create_upload(&upload_request(6)).unwrap();
    let a = store.add_upload_part(&upload.id, &b"abc"[..]).unwrap();
    let b = store.add_upload_part(&upload.id, &b"def"[..]).unwrap();
    let foreign = store.add_upload_part(&other.id, &b"xyz"[..]).unwrap();
    let complete = |ids: &[&str]| {
        store.complete_upload(
            &upload.id,
            &CompleteUpload {
                part_ids: ids.iter().map(|id| (*id).to_owned()).collect(),
                md5: None,
            },
        )
    };
    for ids in [
        vec![],
        vec![a.id.as_str()],
        vec![a.id.as_str(), a.id.as_str()],
        vec![a.id.as_str(), foreign.id.as_str()],
        vec![a.id.as_str(), "part_unknown"],
    ] {
        assert!(
            matches!(complete(&ids), Err(Error::InvalidArgument(_))),
            "{ids:?}"
        );
    }
    assert_eq!(
        store.get_upload(&upload.id).unwrap().status,
        UploadStatus::Pending
    );
    assert_empty(&store.list_files(&ListFiles::default()).unwrap().data);

    let done = complete(&[b.id.as_str(), a.id.as_str()]).unwrap();
    assert_eq!(read_all(&store, &done.file.unwrap().id), b"defabc");
    // The other upload's part is untouched.
    assert_eq!(
        store.get_upload(&other.id).unwrap().status,
        UploadStatus::Pending
    );
    assert_eq!(entries(&blobs_dir(&store)).len(), 2);
}

#[test]
fn cancel_is_terminal_and_releases_parts() {
    let temp = TempRoot::new();
    let store = temp.open();
    let upload = store.create_upload(&upload_request(3)).unwrap();
    let part = store.add_upload_part(&upload.id, &b"abc"[..]).unwrap();
    let cancelled = store.cancel_upload(&upload.id).unwrap();
    assert_eq!(cancelled.status, UploadStatus::Cancelled);
    assert!(cancelled.file.is_none());
    assert_empty(&entries(&blobs_dir(&store)));
    assert!(matches!(
        store.cancel_upload(&upload.id),
        Err(Error::Conflict(_))
    ));
    assert!(matches!(
        store.add_upload_part(&upload.id, &b"a"[..]),
        Err(Error::Conflict(_))
    ));
    assert!(matches!(
        store.complete_upload(
            &upload.id,
            &CompleteUpload {
                part_ids: vec![part.id],
                md5: None
            }
        ),
        Err(Error::Conflict(_))
    ));
    assert_empty(&store.list_files(&ListFiles::default()).unwrap().data);
}

#[test]
fn concurrent_completions_produce_exactly_one_file() {
    let temp = TempRoot::new();
    let store = temp.open();
    let upload = store.create_upload(&upload_request(4)).unwrap();
    let part = store.add_upload_part(&upload.id, &b"once"[..]).unwrap();
    let request = CompleteUpload {
        part_ids: vec![part.id],
        md5: None,
    };
    let results: Vec<bool> = std::thread::scope(|scope| {
        // Collecting first is required so all threads run concurrently.
        #[allow(clippy::needless_collect)]
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let store = store.clone();
                let request = request.clone();
                let upload_id = upload.id.clone();
                scope.spawn(move || store.complete_upload(&upload_id, &request).is_ok())
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    assert_eq!(results.iter().filter(|ok| **ok).count(), 1);
    assert_eq!(
        store.list_files(&ListFiles::default()).unwrap().data.len(),
        1
    );
    assert_eq!(store.purge_file_storage().unwrap().orphan_entries, 0);
    assert_eq!(entries(&blobs_dir(&store)).len(), 1);
    assert_empty(&entries(&tmp_dir(&store)));
}

/// Yields `data` once the test allows it, after announcing the first read.
struct GatedReader {
    started: std::sync::mpsc::Sender<()>,
    go: std::sync::mpsc::Receiver<()>,
    data: &'static [u8],
}

impl Read for GatedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.data.is_empty() {
            return Ok(0);
        }
        let _ = self.started.send(());
        self.go.recv().unwrap();
        let len = self.data.len().min(buf.len());
        buf[..len].copy_from_slice(&self.data[..len]);
        self.data = &self.data[len..];
        Ok(len)
    }
}

/// A part whose stream ends while another writer holds the lock must have its
/// expiry judged when the commit actually runs, not when the stream ended.
#[test]
fn part_commit_rechecks_expiry_after_waiting_for_the_write_lock() {
    let temp = TempRoot::new();
    let store = temp.open();
    let upload = store.create_upload(&upload_request(4)).unwrap();
    let (started_tx, started) = std::sync::mpsc::channel();
    let (go, go_rx) = std::sync::mpsc::channel();
    let reader = GatedReader {
        started: started_tx,
        go: go_rx,
        data: b"late",
    };
    let result = std::thread::scope(|scope| {
        let writer = scope.spawn(|| store.add_upload_part(&upload.id, reader));
        started.recv().unwrap();
        // Hold the write lock while the stream finishes, then expire the
        // upload at a strictly later second before releasing the lock.
        let db = database(&store);
        db.execute_batch("BEGIN IMMEDIATE").unwrap();
        go.send(()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2100));
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        db.execute(
            "UPDATE uploads SET expires_at = ?1 WHERE id = ?2",
            rusqlite::params![i64::try_from(now).unwrap(), upload.id],
        )
        .unwrap();
        db.execute_batch("COMMIT").unwrap();
        writer.join().unwrap()
    });
    assert!(matches!(result, Err(Error::Conflict(_))), "{result:?}");
    assert_eq!(
        store.get_upload(&upload.id).unwrap().status,
        UploadStatus::Expired
    );
    assert_empty(&entries(&blobs_dir(&store)));
    assert_empty(&entries(&tmp_dir(&store)));
    assert_eq!(
        store.purge_file_storage().unwrap(),
        FileStoragePurge::default()
    );
}

#[test]
fn expiry_is_enforced_on_access_and_purge_reclaims_bytes() {
    let temp = TempRoot::new();
    let store = temp.open();
    let keep = store
        .create_file(&request("keep.txt", FilePurpose::UserData), &b"keep"[..])
        .unwrap();
    let old = store
        .create_file(&request("old.txt", FilePurpose::UserData), &b"old"[..])
        .unwrap();
    let upload = store.create_upload(&upload_request(4)).unwrap();
    let part = store.add_upload_part(&upload.id, &b"ab"[..]).unwrap();

    let db = database(&store);
    db.execute("UPDATE files SET expires_at = 1 WHERE id = ?1", [&old.id])
        .unwrap();
    db.execute(
        "UPDATE uploads SET expires_at = 1 WHERE id = ?1",
        [&upload.id],
    )
    .unwrap();
    drop(db);

    assert!(matches!(store.get_file(&old.id), Err(Error::NotFound(_))));
    assert!(matches!(
        store.open_file_content(&old.id),
        Err(Error::NotFound(_))
    ));
    let listed = store.list_files(&ListFiles::default()).unwrap();
    assert_eq!(listed.data, vec![keep.clone()]);

    assert_eq!(
        store.get_upload(&upload.id).unwrap().status,
        UploadStatus::Expired
    );
    assert!(matches!(
        store.add_upload_part(&upload.id, &b"cd"[..]),
        Err(Error::Conflict(_))
    ));
    assert!(matches!(
        store.complete_upload(
            &upload.id,
            &CompleteUpload {
                part_ids: vec![part.id],
                md5: None
            }
        ),
        Err(Error::Conflict(_))
    ));
    assert!(matches!(
        store.cancel_upload(&upload.id),
        Err(Error::Conflict(_))
    ));

    // Bytes remain until an explicit purge.
    assert_eq!(entries(&blobs_dir(&store)).len(), 3);
    fs::write(blobs_dir(&store).join("stray"), b"junk").unwrap();
    fs::write(tmp_dir(&store).join("crashed.tmp"), b"junk").unwrap();
    let report = store.purge_file_storage().unwrap();
    assert_eq!(report.expired_files, 1);
    assert_eq!(report.expired_upload_parts, 1);
    assert_eq!(report.orphan_entries, 2);
    assert_eq!(entries(&blobs_dir(&store)).len(), 1);
    assert_empty(&entries(&tmp_dir(&store)));
    assert_eq!(read_all(&store, &keep.id), b"keep");
    assert_eq!(
        store.purge_file_storage().unwrap(),
        FileStoragePurge::default()
    );
}

#[test]
fn verifies_optional_md5() {
    let temp = TempRoot::new();
    let store = temp.open();
    let upload = store.create_upload(&upload_request(3)).unwrap();
    let part = store.add_upload_part(&upload.id, &b"abc"[..]).unwrap();
    let mut request = CompleteUpload {
        part_ids: vec![part.id],
        md5: Some("00000000000000000000000000000000".to_owned()),
    };
    assert!(matches!(
        store.complete_upload(&upload.id, &request),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(
        store.get_upload(&upload.id).unwrap().status,
        UploadStatus::Pending
    );
    request.md5 = Some("not-hex".to_owned());
    assert!(matches!(
        store.complete_upload(&upload.id, &request),
        Err(Error::InvalidArgument(_))
    ));
    // RFC 1321 test vector for "abc".
    request.md5 = Some("900150983CD24FB0D6963F7D28E17F72".to_owned());
    let done = store.complete_upload(&upload.id, &request).unwrap();
    assert_eq!(done.status, UploadStatus::Completed);
    assert_eq!(entries(&blobs_dir(&store)).len(), 1);
}
