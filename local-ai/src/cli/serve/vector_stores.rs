//! `/v1/vector_stores`: the HTTP adapter for [`local_services::vector_stores`]
//! and its background indexer.
//!
//! | method   | path                                                   | answer                           |
//! |----------|--------------------------------------------------------|----------------------------------|
//! | `POST`   | `/v1/vector_stores`                                    | `vector_store`                   |
//! | `GET`    | `/v1/vector_stores`                                    | `list` of stores                 |
//! | `GET`    | `/v1/vector_stores/{id}`                               | `vector_store`                   |
//! | `POST`   | `/v1/vector_stores/{id}`                               | `vector_store` (modified)        |
//! | `DELETE` | `/v1/vector_stores/{id}`                               | `vector_store.deleted`           |
//! | `POST`   | `/v1/vector_stores/{id}/files`                         | `vector_store.file`              |
//! | `GET`    | `/v1/vector_stores/{id}/files`                         | `list` of files                  |
//! | `GET`    | `/v1/vector_stores/{id}/files/{file_id}`               | `vector_store.file`              |
//! | `POST`   | `/v1/vector_stores/{id}/files/{file_id}`               | `vector_store.file` (attributes) |
//! | `DELETE` | `/v1/vector_stores/{id}/files/{file_id}`               | `vector_store.file.deleted`      |
//! | `GET`    | `/v1/vector_stores/{id}/files/{file_id}/content`       | `vector_store.file_content.page` |
//! | `POST`   | `/v1/vector_stores/{id}/file_batches`                  | `vector_store.file_batch`        |
//! | `GET`    | `/v1/vector_stores/{id}/file_batches/{batch_id}`       | `vector_store.file_batch`        |
//! | `POST`   | `/v1/vector_stores/{id}/file_batches/{batch_id}/cancel`| `vector_store.file_batch`        |
//! | `GET`    | `/v1/vector_stores/{id}/file_batches/{batch_id}/files` | `list` of files                  |
//! | `POST`   | `/v1/vector_stores/{id}/search`                        | `vector_store.search_results.page` |
//!
//! Bodies are strict JSON (unknown fields and repeated keys are a `400`);
//! lists take `limit` (1 to 100, default 20), `order` (`desc` by default),
//! `after`, `before` and, for files, `filter` (a status). Invalid input is a
//! `400`, an unknown object a `404`, and a state conflict (cancelling a
//! finished batch, searching or attaching to an expired store) a `409`.
//!
//! Search is lexical BM25, or hybrid BM25 and embedding search when the
//! embedding model is installed; see the native module for the rankings,
//! the score mapping and what `rewrite_query` and `ranking_options` do. Attaching
//! files answers at once with `in_progress` objects and wakes the indexer;
//! clients follow the status as they would with `OpenAI`.

use std::sync::Arc;

use axum::body::Body;
use axum::http::request::Parts;
use axum::http::{Method, StatusCode};
use axum::response::Response;
use serde_json::Value;
use tokio::sync::Semaphore;

use local_services::Store;
use local_services::vector_stores::{
    Attributes, CreateFileBatch, CreateVectorStore, CreateVectorStoreFile, FileStatus, ListOrder,
    ListQuery, SearchRequest, UpdateVectorStore, VectorIndex, VectorStoreIndexer,
};

use super::AppState;
use super::response::{error_response, json_response};
use super::services::{
    bad_request, body_required, failure, read_json, route_not_found, typed, unavailable,
};
use super::store::query_pairs;

/// The route prefix served here.
const PREFIX: &str = "/v1/vector_stores";
/// Store calls running on the blocking pool at once.
const MAX_BLOCKING: usize = 4;

/// Why a request failed: its status and message.
type Failure = (StatusCode, String);

/// Whether `path` belongs to the Vector Stores API (any method).
pub(super) fn matches(path: &str) -> bool {
    path.strip_prefix(PREFIX)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// Answer a Vector Stores request. Authentication has already passed.
pub(super) async fn handle(state: &AppState, parts: &Parts, body: Body) -> Response {
    let Some(vector_stores) = state.vector_stores.as_ref() else {
        return error_response(
            StatusCode::NOT_FOUND,
            "route not found: vector stores are unavailable on this server (no services \
             store, or another local-ai process owns this data directory's index)",
            state,
            &parts.headers,
        )
        .await;
    };
    match vector_stores.dispatch(parts, body).await {
        Ok(value) => json_response(StatusCode::OK, value, state, &parts.headers).await,
        Err((status, message)) => error_response(status, &message, state, &parts.headers).await,
    }
}

/// The Vector Stores service: the index (which owns the store handle), its
/// indexer and the bound on work.
#[derive(Debug)]
pub(super) struct VectorStores {
    index: Arc<VectorIndex>,
    indexer: VectorStoreIndexer,
    blocking: Arc<Semaphore>,
}

impl VectorStores {
    /// Open the index (reconciling it with the database, which may rebuild
    /// it; blocks) and start the indexer, which resumes files a stopped
    /// server left `in_progress`.
    pub(super) fn start(store: &Store) -> local_services::Result<Arc<Self>> {
        let index = Arc::new(VectorIndex::open(store)?);
        let indexer = VectorStoreIndexer::start(Arc::clone(&index))?;
        Ok(Arc::new(Self {
            index,
            indexer,
            blocking: Arc::new(Semaphore::new(MAX_BLOCKING)),
        }))
    }

    /// Stop the indexer after its current file. Blocks.
    pub(super) fn close(&self) {
        self.indexer.close();
    }

    /// Embed chunks and queries with `embedder` from now on: new files are
    /// embedded as they are indexed, the indexer backfills files indexed
    /// before (in the background), and `auto` searches become hybrid.
    pub(super) fn set_embedder(
        &self,
        embedder: Arc<dyn local_services::vector_stores::ChunkEmbedder>,
    ) {
        self.index.set_embedder(Some(embedder));
        self.indexer.notify();
    }

    /// Wake the indexer after a change outside this adapter (deleting a
    /// File removes it from every store and queues its index documents).
    pub(super) fn notify(&self) {
        self.indexer.notify();
    }

    /// Block until the indexer has caught up with every attach so far.
    #[cfg(test)]
    pub(super) fn wait_idle(&self) {
        self.indexer.wait_idle();
    }

    /// Answer one request.
    #[allow(clippy::too_many_lines)]
    pub(super) async fn dispatch(&self, parts: &Parts, body: Body) -> Result<Value, Failure> {
        let segments: Vec<String> = parts
            .uri
            .path()
            .strip_prefix("/v1/")
            .unwrap_or_default()
            .split('/')
            .map(str::to_owned)
            .collect();
        if segments.iter().any(String::is_empty) {
            return Err(route_not_found());
        }
        let query = parts.uri.query();
        let segments: Vec<&str> = segments.iter().map(String::as_str).collect();
        match (parts.method.clone(), segments.as_slice()) {
            (Method::POST, ["vector_stores"]) => {
                no_query(query)?;
                let request: CreateVectorStore = typed(read_json(body).await?.unwrap_or_default())?;
                let attached = !request.file_ids.is_empty();
                let value = self
                    .call(move |index| index.store().create_vector_store(&request))
                    .await?;
                if attached {
                    self.indexer.notify();
                }
                Ok(value)
            }
            (Method::GET, ["vector_stores"]) => {
                let list = list_query(query, false)?;
                self.call(move |index| index.store().list_vector_stores(&list))
                    .await
            }
            (Method::GET, ["vector_stores", id]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                self.call(move |index| index.store().get_vector_store(&id))
                    .await
            }
            (Method::POST, ["vector_stores", id]) => {
                no_query(query)?;
                let request: UpdateVectorStore =
                    typed(read_json(body).await?.ok_or_else(body_required)?)?;
                let id = (*id).to_owned();
                self.call(move |index| index.store().update_vector_store(&id, &request))
                    .await
            }
            (Method::DELETE, ["vector_stores", id]) => {
                no_query(query)?;
                let id = (*id).to_owned();
                let value = self
                    .call(move |index| index.store().delete_vector_store(&id))
                    .await?;
                // The indexer drops the removed files' index documents.
                self.indexer.notify();
                Ok(value)
            }
            (Method::POST, ["vector_stores", id, "files"]) => {
                no_query(query)?;
                let request: CreateVectorStoreFile =
                    typed(read_json(body).await?.ok_or_else(body_required)?)?;
                let id = (*id).to_owned();
                let value = self
                    .call(move |index| index.store().create_vector_store_file(&id, &request))
                    .await?;
                self.indexer.notify();
                Ok(value)
            }
            (Method::GET, ["vector_stores", id, "files"]) => {
                let list = list_query(query, true)?;
                let id = (*id).to_owned();
                self.call(move |index| index.store().list_vector_store_files(&id, &list))
                    .await
            }
            (Method::GET, ["vector_stores", id, "files", file]) => {
                no_query(query)?;
                let (id, file) = ((*id).to_owned(), (*file).to_owned());
                self.call(move |index| index.store().get_vector_store_file(&id, &file))
                    .await
            }
            (Method::POST, ["vector_stores", id, "files", file]) => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Update {
                    attributes: Option<Attributes>,
                }
                no_query(query)?;
                let object = read_json(body).await?.ok_or_else(body_required)?;
                if !object.contains_key("attributes") {
                    return Err(bad_request("attributes is required"));
                }
                let attributes = typed::<Update>(object)?.attributes.unwrap_or_default();
                let (id, file) = ((*id).to_owned(), (*file).to_owned());
                self.call(move |index| {
                    index
                        .store()
                        .update_vector_store_file(&id, &file, &attributes)
                })
                .await
            }
            (Method::DELETE, ["vector_stores", id, "files", file]) => {
                no_query(query)?;
                let (id, file) = ((*id).to_owned(), (*file).to_owned());
                let value = self
                    .call(move |index| index.store().delete_vector_store_file(&id, &file))
                    .await?;
                self.indexer.notify();
                Ok(value)
            }
            (Method::GET, ["vector_stores", id, "files", file, "content"]) => {
                no_query(query)?;
                let (id, file) = ((*id).to_owned(), (*file).to_owned());
                self.call(move |index| index.store().vector_store_file_content(&id, &file))
                    .await
            }
            (Method::POST, ["vector_stores", id, "file_batches"]) => {
                no_query(query)?;
                let request: CreateFileBatch =
                    typed(read_json(body).await?.ok_or_else(body_required)?)?;
                let id = (*id).to_owned();
                let value = self
                    .call(move |index| index.store().create_vector_store_file_batch(&id, &request))
                    .await?;
                self.indexer.notify();
                Ok(value)
            }
            (Method::GET, ["vector_stores", id, "file_batches", batch]) => {
                no_query(query)?;
                let (id, batch) = ((*id).to_owned(), (*batch).to_owned());
                self.call(move |index| index.store().get_vector_store_file_batch(&id, &batch))
                    .await
            }
            (Method::POST, ["vector_stores", id, "file_batches", batch, "cancel"]) => {
                no_query(query)?;
                if let Some(key) = read_json(body).await?.unwrap_or_default().keys().next() {
                    return Err(bad_request(format!("unknown field {key:?}")));
                }
                let (id, batch) = ((*id).to_owned(), (*batch).to_owned());
                self.call(move |index| index.store().cancel_vector_store_file_batch(&id, &batch))
                    .await
            }
            (Method::GET, ["vector_stores", id, "file_batches", batch, "files"]) => {
                let list = list_query(query, true)?;
                let (id, batch) = ((*id).to_owned(), (*batch).to_owned());
                self.call(move |index| {
                    index
                        .store()
                        .list_vector_store_file_batch_files(&id, &batch, &list)
                })
                .await
            }
            (Method::POST, ["vector_stores", id, "search"]) => {
                no_query(query)?;
                let request: SearchRequest =
                    typed(read_json(body).await?.ok_or_else(body_required)?)?;
                let id = (*id).to_owned();
                self.call(move |index| index.search(&id, &request)).await
            }
            _ => Err(route_not_found()),
        }
    }

    /// Run one store call on the blocking pool, at most [`MAX_BLOCKING`] at
    /// once, and serialize its answer.
    async fn call<T, F>(&self, work: F) -> Result<Value, Failure>
    where
        T: serde::Serialize + Send + 'static,
        F: FnOnce(&VectorIndex) -> local_services::Result<T> + Send + 'static,
    {
        let permit = Arc::clone(&self.blocking)
            .acquire_owned()
            .await
            .map_err(|_| unavailable("the vector stores are shutting down"))?;
        let index = Arc::clone(&self.index);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            work(&index).map_err(failure).and_then(|value| {
                serde_json::to_value(value).map_err(|error| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("vector store answer failed: {error}"),
                    )
                })
            })
        })
        .await
        .unwrap_or_else(|error| {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("vector stores task failed: {error}"),
            ))
        })
    }
}

fn no_query(query: Option<&str>) -> Result<(), Failure> {
    match query_pairs(query).first() {
        Some((key, _)) => Err(bad_request(format!(
            "query parameter {key:?} is not supported"
        ))),
        None => Ok(()),
    }
}

/// `limit`, `order`, `after`, `before` and (for files) `filter`, each at
/// most once.
fn list_query(query: Option<&str>, files: bool) -> Result<ListQuery, Failure> {
    let mut list = ListQuery::default();
    let mut seen: Vec<String> = Vec::new();
    for (key, value) in query_pairs(query) {
        if seen.contains(&key) {
            return Err(bad_request(format!("query parameter {key:?} is repeated")));
        }
        match key.as_str() {
            "limit" => {
                list.limit = Some(value.parse().map_err(|_| {
                    bad_request(format!("limit {value:?} is not a positive integer"))
                })?);
            }
            "order" => {
                list.order = match value.as_str() {
                    "asc" => ListOrder::Asc,
                    "desc" => ListOrder::Desc,
                    _ => return Err(bad_request(format!("order {value:?} must be asc or desc"))),
                };
            }
            "after" => list.after = Some(value),
            "before" => list.before = Some(value),
            "filter" if files => {
                list.filter = Some(match value.as_str() {
                    "in_progress" => FileStatus::InProgress,
                    "completed" => FileStatus::Completed,
                    "failed" => FileStatus::Failed,
                    "cancelled" => FileStatus::Cancelled,
                    _ => {
                        return Err(bad_request(format!(
                            "filter {value:?} must be in_progress, completed, failed or cancelled"
                        )));
                    }
                });
            }
            _ => {
                return Err(bad_request(format!(
                    "query parameter {key:?} is not supported"
                )));
            }
        }
        seen.push(key);
    }
    Ok(list)
}

#[cfg(test)]
#[path = "vector_stores_tests.rs"]
mod tests;
