//! HTTP adapters for the durable native services in [`local_services`]:
//! Conversations, Files and Uploads, in their `OpenAI`-compatible shapes.
//!
//! Routes (all behind the server's API key, checked before [`handle`]):
//!
//! | method   | path                                         | answer                      |
//! |----------|----------------------------------------------|-----------------------------|
//! | `POST`   | `/v1/conversations`                          | `conversation`              |
//! | `GET`    | `/v1/conversations/{id}`                     | `conversation`              |
//! | `POST`   | `/v1/conversations/{id}`                     | `conversation` (metadata)   |
//! | `DELETE` | `/v1/conversations/{id}`                     | `conversation.deleted`      |
//! | `GET`    | `/v1/conversations/{id}/items`               | `list` of items             |
//! | `POST`   | `/v1/conversations/{id}/items`               | `list` of the added items   |
//! | `GET`    | `/v1/conversations/{id}/items/{item_id}`     | the item                    |
//! | `DELETE` | `/v1/conversations/{id}/items/{item_id}`     | `conversation`              |
//! | `POST`   | `/v1/files` (multipart)                      | `file`                      |
//! | `GET`    | `/v1/files`                                  | `list` of files             |
//! | `GET`    | `/v1/files/{id}`                             | `file`                      |
//! | `DELETE` | `/v1/files/{id}`                             | `{id, object: file, deleted}` |
//! | `GET`    | `/v1/files/{id}/content`                     | the raw bytes               |
//! | `POST`   | `/v1/uploads`                                | `upload`                    |
//! | `POST`   | `/v1/uploads/{id}/parts` (multipart)         | `upload.part`               |
//! | `POST`   | `/v1/uploads/{id}/complete`                  | `upload` with its `file`    |
//! | `POST`   | `/v1/uploads/{id}/cancel`                    | `upload`                    |
//!
//! # Requests
//!
//! JSON bodies are at most [`MAX_REQUEST_BYTES`], must be one object, and are
//! strict: an unknown field or a key repeated anywhere in the document is a
//! `400`. Query strings are strict the same way: only the documented
//! parameters, each at most once. Conversation items list newest first, 20 per
//! page (1 to 100) by default; files list newest first, up to 10,000 per page
//! by default. `include` is accepted only as `reasoning.encrypted_content`,
//! which changes nothing because items are returned as they were stored.
//!
//! Multipart bodies stream: every file or part field is spooled, chunk by
//! chunk, into an unnamed owner-only temporary file in the store's private
//! `http-spool` directory and then handed to the native store on the blocking
//! pool, so no upload is ever held in memory. A file is at most 512 MiB (200
//! MiB for `batch`), a part at most 64 MiB; larger bodies are a `413`. At most
//! [`MAX_TRANSFERS`] multipart bodies and [`MAX_DOWNLOADS`] downloads are in
//! flight at once, which also bounds the disk the spool can use; beyond that
//! the answer is a `503` with `Retry-After`.
//!
//! `expires_after` sets a file's `expires_at` to `created_at + seconds`
//! (3600 to 2592000; `batch` files default to 30 days). Expired files are
//! not found. An upload expires an hour after it is created.
//!
//! # Answers
//!
//! Every store call runs through a bounded `spawn_blocking`, never on a runtime
//! worker. Invalid input is a `400`, an unknown or deleted object a `404`, a
//! state conflict a `409`, and a part, completion or cancel on an upload that
//! has expired or was cancelled a `410`. A deleted conversation keeps its items
//! on disk, but neither it nor they are reachable any more.
//!
//! A file's `purpose` is stored and echoed only: storing a `vision`,
//! `fine-tune` or `batch` file starts no processing. A batch input file is
//! processed only after a separate request to `/v1/batches`.

use std::io::{ErrorKind, Seek as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::multipart::{Field, MultipartError};
use axum::extract::{DefaultBodyLimit, FromRequest as _, Multipart, Request};
use axum::http::request::Parts;
use axum::http::{Method, StatusCode, header};
use axum::response::Response;
use http_body_util::{BodyExt as _, LengthLimitError, Limited};
use serde::Deserialize;
use serde::de::{self, DeserializeOwned, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, json};
use tokio::io::AsyncWriteExt as _;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_stream::StreamExt as _;
use tokio_util::io::ReaderStream;

use local_services::files::{
    CompleteUpload, CreateFile, CreateUpload, ExpiresAfter, ExpiresAfterAnchor, FileList,
    FileObject, FilePurpose, ListFiles, MAX_FILE_BYTES, MAX_UPLOAD_PART_BYTES, SortOrder,
    UploadObject, UploadPart, UploadStatus,
};
use local_services::{Error, ListItems, Metadata, Order, Store};

use super::response::{add_alt_svc, error_response, json_response};
use super::store::query_pairs;
use super::{AppState, MAX_REQUEST_BYTES};

/// Store calls running on the blocking pool at once.
pub(super) const MAX_BLOCKING: usize = 8;
/// Multipart bodies being received (and spooled) at once.
pub(super) const MAX_TRANSFERS: usize = 4;
/// File downloads streaming at once.
pub(super) const MAX_DOWNLOADS: usize = 16;
/// Multipart framing allowed on top of a field's own byte limit.
const FORM_OVERHEAD_BYTES: u64 = 64 * 1024;
/// Fields one multipart body may carry.
const MAX_FORM_FIELDS: usize = 8;
/// Largest text field (`purpose`, `expires_after[...]`).
const MAX_TEXT_FIELD_BYTES: usize = 1024;
/// Read size for downloads.
const DOWNLOAD_CHUNK_BYTES: usize = 64 * 1024;
/// Served with every `503` so a client can retry instead of guessing.
const RETRY_AFTER_SECONDS: &str = "1";
/// The private spool directory inside the store directory.
const SPOOL_DIR: &str = "http-spool";
/// The one `include` value accepted; a no-op (see the module docs).
const INCLUDE_ENCRYPTED_REASONING: &str = "reasoning.encrypted_content";

/// Route prefixes served here.
const PREFIXES: [&str; 3] = ["/v1/conversations", "/v1/files", "/v1/uploads"];

/// Why a request failed: its status and message.
type Failure = (StatusCode, String);

/// Whether `path` belongs to the native services (any method).
pub(super) fn matches(path: &str) -> bool {
    PREFIXES.iter().any(|prefix| {
        path.strip_prefix(prefix)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
    })
}

/// Answer a native services request. Authentication has already passed.
pub(super) async fn handle(state: &AppState, parts: &Parts, body: Body) -> Response {
    let Some(services) = state.services.as_ref() else {
        return error_response(
            StatusCode::NOT_FOUND,
            "route not found: this server has no native services store",
            state,
            &parts.headers,
        )
        .await;
    };
    match services.dispatch(parts, body).await {
        Reply::Json(status, value) => json_response(status, value, state, &parts.headers).await,
        Reply::Error(status, message) => {
            let mut response = error_response(status, &message, state, &parts.headers).await;
            if status == StatusCode::SERVICE_UNAVAILABLE {
                response.headers_mut().insert(
                    header::RETRY_AFTER,
                    header::HeaderValue::from_static(RETRY_AFTER_SECONDS),
                );
            }
            response
        }
        Reply::Content(download) => {
            let mut response = download.into_response();
            add_alt_svc(&mut response, state);
            response
        }
    }
}

/// The native services: the store plus the bounds on work done for it.
#[derive(Debug)]
pub(super) struct Services {
    store: Store,
    spool: PathBuf,
    blocking: Arc<Semaphore>,
    transfers: Arc<Semaphore>,
    downloads: Arc<Semaphore>,
}

impl Services {
    pub(super) const fn store(&self) -> &Store {
        &self.store
    }

    /// Use the native macOS application-data location without backend configuration.
    pub(super) fn open_default() -> crate::Result<Arc<Self>> {
        let path = std::env::var_os("HOME")
            .filter(|home| !home.is_empty())
            .map(PathBuf::from)
            .ok_or_else(|| {
                crate::Error::InvalidArgument("HOME must name the user's directory".into())
            })?
            .join("Library/Application Support/local-ai/services");
        Self::open(&path).map(Arc::new).map_err(|error| {
            crate::Error::InvalidArgument(format!(
                "cannot open native services at {}: {error}",
                path.display()
            ))
        })
    }

    /// Open (creating if needed) the store in directory `path`, prepare its
    /// private spool directory and remove storage left behind by expired or
    /// abandoned files. Blocks; call before serving or off the runtime.
    pub(super) fn open(path: impl AsRef<Path>) -> local_services::Result<Self> {
        let store = Store::open(path)?;
        let spool = store.root().join(SPOOL_DIR);
        create_private_dir(&spool)?;
        store.purge_file_storage()?;
        Ok(Self {
            store,
            spool,
            blocking: Arc::new(Semaphore::new(MAX_BLOCKING)),
            transfers: Arc::new(Semaphore::new(MAX_TRANSFERS)),
            downloads: Arc::new(Semaphore::new(MAX_DOWNLOADS)),
        })
    }

    /// Answer one request, independent of the server's other state.
    pub(super) async fn dispatch(&self, parts: &Parts, body: Body) -> Reply {
        self.route(parts, body)
            .await
            .unwrap_or_else(|(status, message)| Reply::Error(status, message))
    }

    #[allow(clippy::too_many_lines)]
    async fn route(&self, parts: &Parts, body: Body) -> Result<Reply, Failure> {
        let segments: Vec<&str> = parts
            .uri
            .path()
            .strip_prefix("/v1/")
            .unwrap_or_default()
            .split('/')
            .collect();
        if segments.iter().any(|segment| segment.is_empty()) {
            return Err(route_not_found());
        }
        let query = parts.uri.query();
        let method = parts.method.clone();
        let ok = |value| Ok(Reply::Json(StatusCode::OK, value));
        match (method, segments.as_slice()) {
            (Method::POST, ["conversations"]) => {
                no_query(query)?;
                ok(self.create_conversation(body).await?)
            }
            (Method::GET, ["conversations", id]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                ok(self
                    .call(move |store| store.get_conversation(&id).map(to_json))
                    .await??)
            }
            (Method::POST, ["conversations", id]) => {
                no_query(query)?;
                ok(self.update_conversation(id, body).await?)
            }
            (Method::DELETE, ["conversations", id]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                ok(self
                    .call(move |store| store.delete_conversation(&id).map(to_json))
                    .await??)
            }
            (Method::GET, ["conversations", id, "items"]) => {
                let list = items_query(query)?;
                let id = (*id).to_owned();
                ok(self
                    .call(move |store| store.list_items(&id, &list).map(to_json))
                    .await??)
            }
            (Method::POST, ["conversations", id, "items"]) => {
                include_only(query)?;
                ok(self.add_items(id, body).await?)
            }
            (Method::GET, ["conversations", id, "items", item]) => {
                include_only(query)?;
                let (id, item) = ((*id).to_owned(), (*item).to_owned());
                ok(self
                    .call(move |store| store.get_item(&id, &item).map(to_json))
                    .await??)
            }
            (Method::DELETE, ["conversations", id, "items", item]) => {
                no_query(query)?;
                let (id, item) = ((*id).to_owned(), (*item).to_owned());
                ok(self
                    .call(move |store| store.delete_item(&id, &item).map(to_json))
                    .await??)
            }
            (Method::POST, ["files"]) => {
                no_query(query)?;
                ok(self.create_file(parts, body).await?)
            }
            (Method::GET, ["files"]) => {
                let list = files_query(query)?;
                ok(self
                    .call(move |store| store.list_files(&list).map(|list| file_list_json(&list)))
                    .await??)
            }
            (Method::GET, ["files", id]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                ok(self
                    .call(move |store| store.get_file(&id).map(|file| file_json(&file)))
                    .await??)
            }
            (Method::DELETE, ["files", id]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                ok(self
                    .call(move |store| {
                        store.delete_file(&id).map(|deleted| {
                            json!({"id":deleted.id,"object":"file","deleted":deleted.deleted})
                        })
                    })
                    .await??)
            }
            (Method::GET, ["files", id, "content"]) => {
                no_query(query)?;
                self.file_content(id).await.map(Reply::Content)
            }
            (Method::POST, ["uploads"]) => {
                no_query(query)?;
                ok(self.create_upload(body).await?)
            }
            (Method::POST, ["uploads", id, "parts"]) => {
                no_query(query)?;
                ok(self.add_part(id, parts, body).await?)
            }
            (Method::POST, ["uploads", id, "complete"]) => {
                no_query(query)?;
                ok(self.complete_upload(id, body).await?)
            }
            (Method::POST, ["uploads", id, "cancel"]) => {
                no_query(query)?;
                ok(self.cancel_upload(id, body).await?)
            }
            _ => Err(route_not_found()),
        }
    }

    /// Run `work` against the store on the blocking pool, at most
    /// [`MAX_BLOCKING`] at once.
    async fn blocking<T, F>(&self, work: F) -> Result<T, Failure>
    where
        T: Send + 'static,
        F: FnOnce(&Store) -> Result<T, Failure> + Send + 'static,
    {
        let permit = Arc::clone(&self.blocking)
            .acquire_owned()
            .await
            .map_err(|_| unavailable("the native services are shutting down"))?;
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            work(&store)
        })
        .await
        .unwrap_or_else(|error| {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("native services task failed: {error}"),
            ))
        })
    }

    /// [`Self::blocking`] for one plain store call; the outer result is the
    /// task, the inner one the call, mapped to its status.
    async fn call<T, F>(&self, work: F) -> Result<Result<T, Failure>, Failure>
    where
        T: Send + 'static,
        F: FnOnce(&Store) -> local_services::Result<T> + Send + 'static,
    {
        self.blocking(move |store| Ok(work(store).map_err(failure)))
            .await
    }

    async fn create_conversation(&self, body: Body) -> Result<Value, Failure> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Create {
            #[serde(default)]
            items: Option<Vec<Value>>,
            #[serde(default)]
            metadata: Option<Metadata>,
        }
        let create: Create = typed(read_json(body).await?.unwrap_or_default())?;
        let (metadata, items) = (
            create.metadata.unwrap_or_default(),
            create.items.unwrap_or_default(),
        );
        self.call(move |store| store.create_conversation(&metadata, items).map(to_json))
            .await?
    }

    async fn update_conversation(&self, id: &str, body: Body) -> Result<Value, Failure> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Update {
            metadata: Option<Metadata>,
        }
        let object = read_json(body).await?.ok_or_else(body_required)?;
        if !object.contains_key("metadata") {
            return Err(bad_request("metadata is required"));
        }
        let metadata = typed::<Update>(object)?.metadata.unwrap_or_default();
        let id = id.to_owned();
        self.call(move |store| store.update_conversation(&id, &metadata).map(to_json))
            .await?
    }

    async fn add_items(&self, id: &str, body: Body) -> Result<Value, Failure> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Add {
            items: Vec<Value>,
        }
        let add: Add = typed(read_json(body).await?.ok_or_else(body_required)?)?;
        let id = id.to_owned();
        self.call(move |store| store.add_items(&id, add.items).map(to_json))
            .await?
    }

    async fn create_file(&self, parts: &Parts, body: Body) -> Result<Value, Failure> {
        let _transfer = self.transfer()?;
        let mut form = multipart(parts, body, MAX_FILE_BYTES).await?;
        let mut file: Option<(String, std::fs::File)> = None;
        let mut purpose: Option<FilePurpose> = None;
        let mut anchor: Option<String> = None;
        let mut seconds: Option<String> = None;
        let mut fields = 0;
        while let Some(field) = next_field(&mut form, &mut fields).await? {
            let name = field.name().unwrap_or_default().to_owned();
            match name.as_str() {
                "file" => {
                    once(file.is_some(), &name)?;
                    let filename = field
                        .file_name()
                        .map(str::to_owned)
                        .ok_or_else(|| bad_request("the file field must carry a filename"))?;
                    let limit = purpose.map_or(MAX_FILE_BYTES, FilePurpose::max_file_bytes);
                    file = Some((filename, self.spool(field, limit, "file").await?));
                }
                "purpose" => {
                    once(purpose.is_some(), &name)?;
                    let parsed: FilePurpose =
                        read_text(field, &name).await?.parse().map_err(failure)?;
                    if parsed.is_internal() {
                        return Err(bad_request(
                            "batch_output is reserved for generated batch results",
                        ));
                    }
                    purpose = Some(parsed);
                }
                "expires_after[anchor]" => {
                    once(anchor.is_some(), &name)?;
                    anchor = Some(read_text(field, &name).await?);
                }
                "expires_after[seconds]" => {
                    once(seconds.is_some(), &name)?;
                    seconds = Some(read_text(field, &name).await?);
                }
                _ => return Err(unknown_field(&name)),
            }
        }
        let (filename, content) = file.ok_or_else(|| bad_request("the file field is required"))?;
        let purpose = purpose.ok_or_else(|| bad_request("the purpose field is required"))?;
        let expires_after = form_expiry(anchor, seconds)?;
        let request = CreateFile {
            filename,
            purpose,
            expires_after,
        };
        self.blocking(move |store| {
            let mut content = content;
            content.rewind().map_err(|error| io_failure(&error))?;
            store
                .create_file(&request, content)
                .map(|file| file_json(&file))
                .map_err(failure)
        })
        .await
    }

    async fn file_content(&self, id: &str) -> Result<Download, Failure> {
        let permit = Arc::clone(&self.downloads)
            .try_acquire_owned()
            .map_err(|_| unavailable("too many downloads are in progress"))?;
        let id = id.to_owned();
        let opened = self
            .call(move |store| store.open_file_content(&id))
            .await??;
        Ok(Download {
            bytes: opened.file.bytes,
            file: opened.content,
            permit,
        })
    }

    async fn create_upload(&self, body: Body) -> Result<Value, Failure> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Create {
            filename: String,
            purpose: FilePurpose,
            bytes: u64,
            mime_type: String,
            #[serde(default)]
            expires_after: Option<Expiry>,
        }
        let create: Create = typed(read_json(body).await?.ok_or_else(body_required)?)?;
        let request = CreateUpload {
            filename: create.filename,
            purpose: create.purpose,
            bytes: create.bytes,
            mime_type: create.mime_type,
            expires_after: create.expires_after.map(Expiry::native),
        };
        self.call(move |store| {
            store
                .create_upload(&request)
                .map(|upload| upload_json(&upload))
        })
        .await?
    }

    async fn add_part(&self, id: &str, parts: &Parts, body: Body) -> Result<Value, Failure> {
        // Refuse before reading a byte of the body when the upload cannot
        // take a part anyway.
        let upload = id.to_owned();
        let status = self
            .call(move |store| store.get_upload(&upload))
            .await??
            .status;
        if let Some(refusal) = immutable(id, status) {
            return Err(refusal);
        }
        let _transfer = self.transfer()?;
        let mut form = multipart(parts, body, MAX_UPLOAD_PART_BYTES).await?;
        let mut data = None;
        let mut fields = 0;
        while let Some(field) = next_field(&mut form, &mut fields).await? {
            let name = field.name().unwrap_or_default().to_owned();
            if name != "data" {
                return Err(unknown_field(&name));
            }
            once(data.is_some(), &name)?;
            data = Some(self.spool(field, MAX_UPLOAD_PART_BYTES, "part").await?);
        }
        let data = data.ok_or_else(|| bad_request("the data field is required"))?;
        let id = id.to_owned();
        self.blocking(move |store| {
            let mut data = data;
            data.rewind().map_err(|error| io_failure(&error))?;
            store
                .add_upload_part(&id, data)
                .map(|part| part_json(&part))
                .map_err(|error| upload_failure(store, &id, error))
        })
        .await
    }

    async fn complete_upload(&self, id: &str, body: Body) -> Result<Value, Failure> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Complete {
            part_ids: Vec<String>,
            #[serde(default)]
            md5: Option<String>,
        }
        let complete: Complete = typed(read_json(body).await?.ok_or_else(body_required)?)?;
        let request = CompleteUpload {
            part_ids: complete.part_ids,
            md5: complete.md5,
        };
        let id = id.to_owned();
        self.blocking(move |store| {
            store
                .complete_upload(&id, &request)
                .map(|upload| upload_json(&upload))
                .map_err(|error| upload_failure(store, &id, error))
        })
        .await
    }

    async fn cancel_upload(&self, id: &str, body: Body) -> Result<Value, Failure> {
        if let Some(key) = read_json(body).await?.unwrap_or_default().keys().next() {
            return Err(unknown_field(key));
        }
        let id = id.to_owned();
        self.blocking(move |store| {
            store
                .cancel_upload(&id)
                .map(|upload| upload_json(&upload))
                .map_err(|error| upload_failure(store, &id, error))
        })
        .await
    }

    /// A slot for one multipart body, or a `503`.
    fn transfer(&self) -> Result<OwnedSemaphorePermit, Failure> {
        Arc::clone(&self.transfers)
            .try_acquire_owned()
            .map_err(|_| unavailable("too many uploads are in progress"))
    }

    /// Stream `field` into a new unnamed private spool file, refusing more
    /// than `limit` bytes as it arrives.
    async fn spool(
        &self,
        mut field: Field<'_>,
        limit: u64,
        what: &str,
    ) -> Result<std::fs::File, Failure> {
        let directory = self.spool.clone();
        let spooled = tokio::task::spawn_blocking(move || tempfile::tempfile_in(directory))
            .await
            .map_err(|error| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("spool task failed: {error}"),
                )
            })?
            .map_err(|error| io_failure(&error))?;
        let mut spooled = tokio::fs::File::from_std(spooled);
        let mut total = 0_u64;
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|error| multipart_failure(&error))?
        {
            total = total.saturating_add(chunk.len() as u64);
            if total > limit {
                return Err((
                    StatusCode::PAYLOAD_TOO_LARGE,
                    format!("the {what} exceeds the {limit}-byte limit"),
                ));
            }
            spooled
                .write_all(&chunk)
                .await
                .map_err(|error| io_failure(&error))?;
        }
        spooled.flush().await.map_err(|error| io_failure(&error))?;
        Ok(spooled.into_std().await)
    }
}

/// What [`Services::dispatch`] answers, rendered by [`handle`].
pub(super) enum Reply {
    Json(StatusCode, Value),
    Error(StatusCode, String),
    Content(Download),
}

/// An opened file streaming to the client; holds a download slot until the
/// body is dropped.
pub(super) struct Download {
    bytes: u64,
    file: std::fs::File,
    permit: OwnedSemaphorePermit,
}

impl Download {
    /// The raw bytes, streamed without buffering or compression.
    pub(super) fn into_response(self) -> Response {
        let Self {
            bytes,
            file,
            permit,
        } = self;
        let stream =
            ReaderStream::with_capacity(tokio::fs::File::from_std(file), DOWNLOAD_CHUNK_BYTES).map(
                move |chunk| {
                    let _slot = &permit;
                    chunk
                },
            );
        let mut response = Response::new(Body::from_stream(stream));
        let headers = response.headers_mut();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/octet-stream"),
        );
        headers.insert(header::CONTENT_LENGTH, header::HeaderValue::from(bytes));
        headers.insert(
            header::CACHE_CONTROL,
            header::HeaderValue::from_static("no-store"),
        );
        headers.insert(
            header::X_CONTENT_TYPE_OPTIONS,
            header::HeaderValue::from_static("nosniff"),
        );
        response
    }
}

/// `expires_after` in a JSON body.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expiry {
    anchor: ExpiresAfterAnchor,
    seconds: u64,
}

impl Expiry {
    const fn native(self) -> ExpiresAfter {
        ExpiresAfter {
            anchor: self.anchor,
            seconds: self.seconds,
        }
    }
}

/// `expires_after[anchor]` and `expires_after[seconds]` in a form: both or
/// neither.
fn form_expiry(
    anchor: Option<String>,
    seconds: Option<String>,
) -> Result<Option<ExpiresAfter>, Failure> {
    match (anchor, seconds) {
        (None, None) => Ok(None),
        (Some(anchor), Some(seconds)) => {
            if anchor != "created_at" {
                return Err(bad_request("expires_after[anchor] must be created_at"));
            }
            let seconds = seconds
                .parse()
                .map_err(|_| bad_request("expires_after[seconds] must be an integer"))?;
            Ok(Some(ExpiresAfter {
                anchor: ExpiresAfterAnchor::CreatedAt,
                seconds,
            }))
        }
        _ => Err(bad_request(
            "expires_after[anchor] and expires_after[seconds] must be given together",
        )),
    }
}

/// A multipart reader over `body`, bounded at `limit` bytes of content plus
/// framing.
async fn multipart(parts: &Parts, body: Body, limit: u64) -> Result<Multipart, Failure> {
    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .cloned()
        .ok_or_else(|| bad_request("a multipart/form-data body is required"))?;
    let mut request = Request::new(body);
    request
        .headers_mut()
        .insert(header::CONTENT_TYPE, content_type);
    let bound = usize::try_from(limit.saturating_add(FORM_OVERHEAD_BYTES)).unwrap_or(usize::MAX);
    DefaultBodyLimit::max(bound).apply(&mut request);
    Multipart::from_request(request, &())
        .await
        .map_err(|rejection| (rejection.status(), rejection.body_text()))
}

async fn next_field<'a>(
    form: &'a mut Multipart,
    count: &mut usize,
) -> Result<Option<Field<'a>>, Failure> {
    let field = form
        .next_field()
        .await
        .map_err(|error| multipart_failure(&error))?;
    if field.is_some() {
        *count += 1;
        if *count > MAX_FORM_FIELDS {
            return Err(bad_request(format!(
                "a form may carry at most {MAX_FORM_FIELDS} fields"
            )));
        }
    }
    Ok(field)
}

async fn read_text(mut field: Field<'_>, name: &str) -> Result<String, Failure> {
    if field.file_name().is_some() {
        return Err(bad_request(format!("{name} must be a text field")));
    }
    let mut text = Vec::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|error| multipart_failure(&error))?
    {
        if text.len() + chunk.len() > MAX_TEXT_FIELD_BYTES {
            return Err(bad_request(format!(
                "{name} may be at most {MAX_TEXT_FIELD_BYTES} bytes"
            )));
        }
        text.extend_from_slice(&chunk);
    }
    String::from_utf8(text).map_err(|_| bad_request(format!("{name} must be UTF-8 text")))
}

fn once(seen: bool, name: &str) -> Result<(), Failure> {
    if seen {
        Err(bad_request(format!("field {name:?} is repeated")))
    } else {
        Ok(())
    }
}

fn unknown_field(name: &str) -> Failure {
    bad_request(format!("field {name:?} is not supported here"))
}

fn multipart_failure(error: &MultipartError) -> Failure {
    (
        error.status(),
        format!("invalid multipart body: {}", error.body_text()),
    )
}

/// Read a JSON body of at most [`MAX_REQUEST_BYTES`]: `None` when empty,
/// otherwise one object with no repeated key anywhere.
async fn read_json(body: Body) -> Result<Option<Map<String, Value>>, Failure> {
    let bytes = match Limited::new(body, MAX_REQUEST_BYTES).collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) if error.downcast_ref::<LengthLimitError>().is_some() => {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                "HTTP request is too large".to_owned(),
            ));
        }
        Err(error) => return Err(bad_request(format!("invalid HTTP body: {error}"))),
    };
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes);
    let Strict(value) = Strict::deserialize(&mut deserializer)
        .and_then(|value| deserializer.end().map(|()| value))
        .map_err(|error| bad_request(format!("invalid JSON body: {error}")))?;
    match value {
        Value::Object(object) => Ok(Some(object)),
        _ => Err(bad_request("the JSON body must be an object")),
    }
}

fn typed<T: DeserializeOwned>(object: Map<String, Value>) -> Result<T, Failure> {
    serde_json::from_value(Value::Object(object))
        .map_err(|error| bad_request(format!("invalid request body: {error}")))
}

/// A JSON value that refuses any object with a repeated key.
struct Strict(Value);

impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictVisitor).map(Self)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_str<E>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(Strict(value)) = seq.next_element()? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if object.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate key {key:?}")));
            }
            let Strict(value) = map.next_value()?;
            object.insert(key, value);
        }
        Ok(Value::Object(object))
    }
}

/// Known query parameters with their values, and the `include` values.
type Query<'a> = (Vec<(&'a str, String)>, Vec<String>);

/// The query parameters, each known one at most once.
fn query_map<'a>(query: Option<&str>, known: &[&'a str]) -> Result<Query<'a>, Failure> {
    let mut single: Vec<(&'a str, String)> = Vec::new();
    let mut include = Vec::new();
    for (key, value) in query_pairs(query) {
        if matches!(key.as_str(), "include" | "include[]") && known.contains(&"include") {
            include.push(value);
            continue;
        }
        let Some(name) = known
            .iter()
            .find(|name| **name == key && **name != "include")
        else {
            return Err(bad_request(format!(
                "query parameter {key:?} is not supported"
            )));
        };
        if single.iter().any(|(seen, _)| seen == name) {
            return Err(bad_request(format!("query parameter {key:?} is repeated")));
        }
        single.push((name, value));
    }
    if let Some(value) = include
        .iter()
        .find(|value| value.as_str() != INCLUDE_ENCRYPTED_REASONING)
    {
        return Err(bad_request(format!(
            "include {value:?} is not supported; only {INCLUDE_ENCRYPTED_REASONING} is"
        )));
    }
    Ok((single, include))
}

fn no_query(query: Option<&str>) -> Result<(), Failure> {
    query_map(query, &[]).map(drop)
}

fn include_only(query: Option<&str>) -> Result<(), Failure> {
    query_map(query, &["include"]).map(drop)
}

fn parse_limit(value: &str) -> Result<u32, Failure> {
    value
        .parse()
        .map_err(|_| bad_request(format!("limit {value:?} is not a positive integer")))
}

fn parse_order(value: &str) -> Result<bool, Failure> {
    match value {
        "asc" => Ok(true),
        "desc" => Ok(false),
        _ => Err(bad_request(format!("order {value:?} must be asc or desc"))),
    }
}

fn items_query(query: Option<&str>) -> Result<ListItems, Failure> {
    let (pairs, _) = query_map(query, &["limit", "order", "after", "include"])?;
    let mut list = ListItems::default();
    for (name, value) in pairs {
        match name {
            "limit" => list.limit = Some(parse_limit(&value)?),
            "order" => {
                list.order = if parse_order(&value)? {
                    Order::Asc
                } else {
                    Order::Desc
                };
            }
            _ => list.after = Some(value),
        }
    }
    Ok(list)
}

fn files_query(query: Option<&str>) -> Result<ListFiles, Failure> {
    let (pairs, _) = query_map(query, &["limit", "order", "after", "purpose"])?;
    let mut list = ListFiles::default();
    for (name, value) in pairs {
        match name {
            "limit" => list.limit = Some(parse_limit(&value)?),
            "order" => {
                list.order = if parse_order(&value)? {
                    SortOrder::Asc
                } else {
                    SortOrder::Desc
                };
            }
            "purpose" => list.purpose = Some(value.parse().map_err(failure)?),
            _ => list.after = Some(value),
        }
    }
    Ok(list)
}

fn to_json<T: serde::Serialize>(value: T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// The public `file` object.
fn file_json(file: &FileObject) -> Value {
    let mut value = json!({
        "id": file.id,
        "object": "file",
        "bytes": file.bytes,
        "created_at": file.created_at,
        "filename": file.filename,
        "purpose": file.purpose.as_str(),
        "status": "processed",
    });
    // These optional OpenAI fields are not nullable. No error detail exists
    // for a processed local file; expiry is present only when configured.
    if let Some(expires_at) = file.expires_at {
        value["expires_at"] = json!(expires_at);
    }
    value
}

fn file_list_json(list: &FileList) -> Value {
    json!({
        "object": "list",
        "data": list.data.iter().map(file_json).collect::<Vec<_>>(),
        "first_id": list.first_id,
        "last_id": list.last_id,
        "has_more": list.has_more,
    })
}

const fn status_name(status: UploadStatus) -> &'static str {
    match status {
        UploadStatus::Pending => "pending",
        UploadStatus::Completed => "completed",
        UploadStatus::Cancelled => "cancelled",
        UploadStatus::Expired => "expired",
    }
}

/// The public `upload` object.
fn upload_json(upload: &UploadObject) -> Value {
    json!({
        "id": upload.id,
        "object": "upload",
        "bytes": upload.bytes,
        "created_at": upload.created_at,
        "expires_at": upload.expires_at,
        "filename": upload.filename,
        "purpose": upload.purpose.as_str(),
        "status": status_name(upload.status),
        "file": upload.file.as_ref().map(file_json),
    })
}

/// The public `upload.part` object.
fn part_json(part: &UploadPart) -> Value {
    json!({
        "id": part.id,
        "object": "upload.part",
        "created_at": part.created_at,
        "upload_id": part.upload_id,
    })
}

/// The refusal for a mutation of upload `id` in `status`, if it is not
/// pending: `410` once expired or cancelled, `409` once completed.
fn immutable(id: &str, status: UploadStatus) -> Option<Failure> {
    match status {
        UploadStatus::Pending => None,
        UploadStatus::Expired | UploadStatus::Cancelled => Some((
            StatusCode::GONE,
            format!("upload {id} is {}", status_name(status)),
        )),
        UploadStatus::Completed => Some((
            StatusCode::CONFLICT,
            format!("upload {id} is already completed"),
        )),
    }
}

/// [`failure`] for an upload mutation, telling a gone upload (`410`) from
/// any other conflict by its state now.
fn upload_failure(store: &Store, id: &str, error: Error) -> Failure {
    if matches!(error, Error::Conflict(_))
        && let Ok(upload) = store.get_upload(id)
        && let Some(refusal) = immutable(id, upload.status)
    {
        return refusal;
    }
    failure(error)
}

/// The status a native error is answered with.
fn failure(error: Error) -> Failure {
    match error {
        Error::InvalidArgument(message) => (StatusCode::BAD_REQUEST, message),
        Error::NotFound(message) => (StatusCode::NOT_FOUND, message),
        Error::Conflict(message) => (StatusCode::CONFLICT, message),
        other => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("native services failed: {other}"),
        ),
    }
}

fn io_failure(error: &std::io::Error) -> Failure {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("native services failed: {error}"),
    )
}

fn bad_request(message: impl Into<String>) -> Failure {
    (StatusCode::BAD_REQUEST, message.into())
}

fn body_required() -> Failure {
    bad_request("a JSON request body is required")
}

fn unavailable(message: &str) -> Failure {
    (StatusCode::SERVICE_UNAVAILABLE, message.to_owned())
}

fn route_not_found() -> Failure {
    (StatusCode::NOT_FOUND, "route not found".to_owned())
}

/// Create `path` owner-only (`0700`) if needed and refuse a symlink, a
/// non-directory, or one anyone else may use.
fn create_private_dir(path: &Path) -> local_services::Result<()> {
    use std::os::unix::fs::{DirBuilderExt as _, PermissionsExt as _};
    match std::fs::DirBuilder::new().mode(0o700).create(path) {
        Err(error) if error.kind() != ErrorKind::AlreadyExists => return Err(error.into()),
        _ => {}
    }
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_dir()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(Error::Io(std::io::Error::new(
            ErrorKind::PermissionDenied,
            format!(
                "spool path {} must be a directory only its owner can use",
                path.display()
            ),
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
#[path = "services_tests.rs"]
mod tests;
