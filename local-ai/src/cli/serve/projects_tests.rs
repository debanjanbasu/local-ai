//! CPU-only tests of per-project isolation through the HTTP adapters and the
//! request preparation the generation routes share: a caller acting for one
//! project never sees, reads, lists, changes or references another
//! project's stored objects, nor unowned ones (created without a project or
//! before owners were kept), which a caller without a project still sees.
//! An object a project may not see answers exactly as a missing ID does.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};

use local_engine::Event;
use local_engine::bonsai_model::StopReason;
use local_services::Store;
use local_services::batches::Owner;
use local_services::files::{CreateFile, FilePurpose};

use super::admin::Meter;
use super::background::{Background, Temporary};
use super::background_tests::{
    Scratch as ResponseScratch, drained, event, manual_temporary, script, stats, stopper,
};
use super::batches::{Batches, Config, Submit};
use super::response::{Protocol, Reply};
use super::responses::{
    PreparedResponses, ResponsesState, prepare_responses_conversing, prepare_resumed,
};
use super::services::{Reply as ServicesReply, Services};
use super::store::ResponseStore;
use super::vector_stores::VectorStores;
use super::{Api, QueueDepth, locate, prepare_retaining, stored_answer};

const A: Option<&str> = Some("proj_a");
const B: Option<&str> = Some("proj_b");

/// A private temporary directory for a services store.
struct Scratch {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        // SQLite refuses a path through a symlink (macOS `/var`).
        let root = dir
            .path()
            .canonicalize()
            .expect("canonical temporary directory")
            .join("services");
        Self { _dir: dir, root }
    }
}

/// `id` with its last character changed: the same shape, naming nothing.
fn absent(id: &str) -> String {
    let mut fake = id.to_owned();
    let last = fake.pop().expect("an ID");
    fake.push(if last == '0' { '1' } else { '0' });
    fake
}

/// `hidden` (the answer about `id`) is exactly `missing` (the answer about
/// the [`absent`] ID of the same shape) but for the ID.
fn same_as_missing<T: std::fmt::Debug>(hidden: T, missing: T, id: &str) {
    let (hidden, missing) = (format!("{hidden:?}"), format!("{missing:?}"));
    assert!(
        !hidden.is_empty() && hidden.replace(id, &absent(id)) == missing,
        "{id}: {hidden} differs from a missing ID's {missing}"
    );
}

fn file(store: &Store, purpose: FilePurpose, name: &str, content: &str) -> String {
    store
        .create_file(
            &CreateFile {
                filename: name.to_owned(),
                purpose,
                expires_after: None,
            },
            content.as_bytes(),
        )
        .expect("file created")
        .id
}

fn request(method: Method, uri: &str, body: &Value) -> Request<Body> {
    let body = if body.is_null() {
        Body::empty()
    } else {
        Body::from(body.to_string())
    };
    Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(body)
        .expect("request")
}

// --- Responses -------------------------------------------------------------

/// `body` prepared for a caller acting for `project`.
fn prepare(
    body: &Value,
    (store, temporary): (Option<&Arc<ResponseStore>>, Option<&Arc<Temporary>>),
    services: Option<&Store>,
    project: Option<&str>,
) -> crate::Result<PreparedResponses> {
    let services = services.map(|store| store.scoped(project));
    prepare_responses_conversing(
        body.to_string().as_bytes(),
        true,
        store,
        temporary,
        (services.as_ref(), project),
        None,
        "m",
    )
}

fn response_state(prepared: PreparedResponses) -> ResponsesState {
    let reply = Reply::new(Protocol::Responses(Arc::new(prepared.echo)), "m".into());
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    ResponsesState::new(&reply, Arc::clone(echo), false)
}

/// Generate a foreground Response for `prepared`, storing it as the server
/// does, and return it.
fn finish(prepared: PreparedResponses) -> Value {
    let mut state = response_state(prepared);
    state.start();
    assert!(state.event(Event::Content("Hi".into())).is_none());
    state
        .event(Event::Finished(stats(StopReason::Eos)))
        .expect("finished")
}

fn located(
    store: &Arc<ResponseStore>,
    temporary: Option<&Arc<Temporary>>,
    id: &str,
    project: Option<&str>,
) -> Result<Arc<ResponseStore>, (StatusCode, String)> {
    locate(temporary, Some(store), &Background::default(), id, project).map(|found| found.store)
}

#[test]
fn stored_responses_are_private_to_the_project_that_created_them() {
    let scratch = ResponseScratch::new();
    let store = scratch.open();
    let response = finish(prepare(&json!({"input":"hi"}), (Some(&store), None), None, A).unwrap());
    let id = response["id"].as_str().unwrap();
    assert_eq!(store.load(id).unwrap().unwrap().project.as_deref(), A);

    // GET, DELETE, `input_items`, cancel and `?stream=true` all locate the
    // Response first: for another project it is not there.
    same_as_missing(
        located(&store, None, id, B).unwrap_err(),
        located(&store, None, &absent(id), B).unwrap_err(),
        id,
    );
    for project in [A, None] {
        let found = located(&store, None, id, project).expect("visible");
        assert_eq!(
            stored_answer(&found, &Method::GET, id, None).unwrap(),
            Some(response.clone())
        );
    }

    // Nor can it be continued from another project.
    let continuing = |id: &str, project| {
        let body = json!({"input":"next","previous_response_id":id});
        prepare(&body, (Some(&store), None), None, project).map(|prepared| prepared.request)
    };
    same_as_missing(
        continuing(id, B).unwrap_err(),
        continuing(&absent(id), B).unwrap_err(),
        id,
    );
    assert!(continuing(id, A).is_ok());
    assert!(continuing(id, None).is_ok());

    // One stored without a project is unowned: no project sees it.
    let unowned =
        finish(prepare(&json!({"input":"hi"}), (Some(&store), None), None, None).unwrap());
    let unowned = unowned["id"].as_str().unwrap();
    assert_eq!(store.load(unowned).unwrap().unwrap().project, None);
    same_as_missing(
        located(&store, None, unowned, A).unwrap_err(),
        located(&store, None, &absent(unowned), A).unwrap_err(),
        unowned,
    );
    assert!(continuing(unowned, A).is_err());
    assert!(located(&store, None, unowned, None).is_ok());
}

#[test]
fn records_written_before_owners_were_kept_read_as_unowned() {
    let scratch = ResponseScratch::new();
    let store = scratch.open();
    // A version 1 record exactly as an earlier build wrote it.
    let id = "resp_0123456789abcdef0123456789abcdef";
    let response = json!({"id":id,"object":"response","status":"completed","output":[]});
    let record = json!({
        "object":"local_ai.stored_response",
        "version":1,
        "response":response,
        "input_items":[{"type":"message","role":"user","content":"old"}],
    });
    std::fs::write(scratch.0.join(format!("{id}.json")), record.to_string()).unwrap();

    let stored = store.load(id).unwrap().expect("still readable");
    assert_eq!((stored.response, stored.project), (response.clone(), None));
    let found = located(&store, None, id, None).expect("visible without a project");
    assert_eq!(
        stored_answer(&found, &Method::GET, id, None).unwrap(),
        Some(response)
    );
    same_as_missing(
        located(&store, None, id, A).unwrap_err(),
        located(&store, None, &absent(id), A).unwrap_err(),
        id,
    );
}

#[test]
fn background_and_temporary_responses_keep_their_project() {
    let (durable_dir, temporary_dir) = (ResponseScratch::new(), ResponseScratch::new());
    let durable = durable_dir.open();
    let (temporary, _clock) = manual_temporary(&temporary_dir);
    let background = Arc::new(Background::default());
    let depth = QueueDepth::default();
    for body in [
        json!({"input":"hi","background":true}),
        json!({"input":"hi","background":true,"store":false}),
    ] {
        let prepared = prepare(&body, (Some(&durable), Some(&temporary)), None, A).unwrap();
        let doc = response_state(prepared);
        let id = doc.id().to_owned();
        let (_, stop) = stopper();
        let steps = vec![
            event(Event::Content("Hi".into())),
            event(Event::Finished(stats(StopReason::Eos))),
        ];
        let (_, pump) = background
            .admit(doc, script(steps), stop, Some(depth.admit()))
            .expect("admitted");
        let find = |id: &str, project| {
            locate(Some(&temporary), Some(&durable), &background, id, project)
                .map(|found| found.store)
        };
        // Queued, running and ended alike.
        same_as_missing(
            find(&id, B).unwrap_err(),
            find(&absent(&id), B).unwrap_err(),
            &id,
        );
        pump.run();
        drained(&background);
        same_as_missing(
            find(&id, B).unwrap_err(),
            find(&absent(&id), B).unwrap_err(),
            &id,
        );
        let store = find(&id, A).expect("its owner finds it");
        assert_eq!(store.load(&id).unwrap().unwrap().project.as_deref(), A);
        assert!(find(&id, None).is_ok());
    }

    // A temporary response of another project is not even named when
    // continued: it is as unknown as any other ID.
    let prepared = prepare(
        &json!({"input":"hi","background":true,"store":false}),
        (Some(&durable), Some(&temporary)),
        None,
        A,
    )
    .unwrap();
    let doc = response_state(prepared);
    let id = doc.id().to_owned();
    let (_, stop) = stopper();
    let (_, pump) = background
        .admit(doc, script(Vec::new()), stop, Some(depth.admit()))
        .expect("admitted");
    pump.run();
    drained(&background);
    let continuing = |id: &str, project| {
        let body = json!({"input":"next","previous_response_id":id});
        prepare(&body, (Some(&durable), Some(&temporary)), None, project)
            .err()
            .map(|error| error.to_string())
    };
    same_as_missing(continuing(&id, B), continuing(&absent(&id), B), &id);
    assert!(
        continuing(&id, A)
            .unwrap()
            .contains("background=true and store=false")
    );

    // A resumed background Response is stored again as its project's.
    let prepared = prepare(
        &json!({"input":"hi","background":true}),
        (Some(&durable), None),
        None,
        A,
    )
    .unwrap();
    let doc = response_state(prepared);
    let id = doc.id().to_owned();
    doc.save(&doc.snapshot("queued")).unwrap();
    let stored = durable.load(&id).unwrap().unwrap();
    let request = stored.request.clone().expect("a resumable request");
    let (_, reply) = prepare_resumed(&stored, &request, (true, None), &durable, None).unwrap();
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    let resumed = ResponsesState::new(&reply, Arc::clone(echo), false);
    resumed.save(&resumed.snapshot("in_progress")).unwrap();
    assert_eq!(durable.load(&id).unwrap().unwrap().project.as_deref(), A);
}

#[test]
fn generation_requests_reference_only_their_projects_conversations_and_files() {
    let scratch = Scratch::new();
    let services = Store::open(&scratch.root).unwrap();
    let conversation = services
        .scoped(A)
        .create_conversation(&local_services::Metadata::new(), Vec::new())
        .unwrap()
        .id;
    let image = file(&services.scoped(A), FilePurpose::Vision, "a.png", "bytes");

    // Responses `conversation`.
    let conversing = |id: &str, project| {
        prepare(
            &json!({"input":"hi","conversation":id}),
            (None, None),
            Some(&services),
            project,
        )
        .map(|prepared| prepared.request)
    };
    same_as_missing(
        conversing(&conversation, B).unwrap_err(),
        conversing(&absent(&conversation), B).unwrap_err(),
        &conversation,
    );
    assert!(conversing(&conversation, A).is_ok());
    assert!(conversing(&conversation, None).is_ok());

    // `file_id` media parts of Responses and Chat Completions.
    let responses = |id: &str, project| {
        let body = json!({"input":[{"role":"user","content":[
            {"type":"input_image","file_id":id}]}]});
        prepare(&body, (None, None), Some(&services), project).map(|prepared| prepared.request)
    };
    let chat = |id: &str, project| {
        let body = json!({"messages":[{"role":"user","content":[
            {"type":"file","file":{"file_id":id}}]}]});
        let scoped = services.scoped(project);
        prepare_retaining(
            Api::Chat,
            body.to_string().as_bytes(),
            true,
            (None, None),
            (Some(&scoped), project),
            None,
            "m",
        )
        .map(|(_, request, _)| request)
    };
    same_as_missing(
        responses(&image, B).unwrap_err(),
        responses(&absent(&image), B).unwrap_err(),
        &image,
    );
    same_as_missing(
        chat(&image, B).unwrap_err(),
        chat(&absent(&image), B).unwrap_err(),
        &image,
    );
    for project in [A, None] {
        assert!(responses(&image, project).is_ok());
        assert!(chat(&image, project).is_ok());
    }
    // An unowned file is no project's.
    let unowned = file(&services, FilePurpose::Vision, "u.png", "bytes");
    assert!(chat(&unowned, A).is_err());
    assert!(chat(&unowned, None).is_ok());
}

// --- Conversations, Files and Uploads ----------------------------------------

async fn services_call(
    services: &Services,
    project: Option<&str>,
    (method, uri, body): (Method, &str, &Value),
) -> (StatusCode, Value) {
    let (parts, body) = request(method, uri, body).into_parts();
    match services.dispatch(&parts, body, project).await {
        ServicesReply::Json(status, value) => (status, value),
        ServicesReply::Error(status, message) => (status, json!({"error":{"message":message}})),
        ServicesReply::Content(_) => (StatusCode::OK, json!("content")),
    }
}

/// Every route of `routes` naming `id` answers project B exactly as for a
/// missing ID: a `404`.
async fn services_hidden(services: &Services, routes: Vec<(Method, String, Value)>, id: &str) {
    for (method, uri, body) in routes {
        let fake = uri.replace(id, &absent(id));
        let theirs = services_call(services, B, (method.clone(), &uri, &body)).await;
        let missing = services_call(services, B, (method, &fake, &body)).await;
        assert_eq!(theirs.0, StatusCode::NOT_FOUND, "{uri}: {theirs:?}");
        same_as_missing(theirs, missing, id);
    }
}

#[tokio::test]
async fn conversations_are_private_over_http() {
    let scratch = Scratch::new();
    let services = Services::open(&scratch.root).unwrap();
    let call = |project, method, uri: &str, body: Value| {
        let (services, uri) = (&services, uri.to_owned());
        async move { services_call(services, project, (method, &uri, &body)).await }
    };
    let item = json!({"items":[{"type":"message","role":"user","content":"hi"}]});
    let (status, conversation) = call(A, Method::POST, "/v1/conversations", item.clone()).await;
    assert_eq!(status, StatusCode::OK);
    let id = conversation["id"].as_str().unwrap().to_owned();
    let base = format!("/v1/conversations/{id}");
    let (_, items) = call(A, Method::GET, &format!("{base}/items"), Value::Null).await;
    let first = items["data"][0]["id"].as_str().unwrap().to_owned();
    services_hidden(
        &services,
        vec![
            (Method::GET, base.clone(), Value::Null),
            (Method::POST, base.clone(), json!({"metadata":{}})),
            (Method::GET, format!("{base}/items"), Value::Null),
            (Method::POST, format!("{base}/items"), item),
            (Method::GET, format!("{base}/items/{first}"), Value::Null),
            (Method::DELETE, format!("{base}/items/{first}"), Value::Null),
            (Method::DELETE, base.clone(), Value::Null),
        ],
        &id,
    )
    .await;
    let (_, mine) = call(A, Method::GET, &format!("{base}/items"), Value::Null).await;
    assert_eq!(
        mine["data"].as_array().unwrap().len(),
        1,
        "B changed nothing"
    );
    assert_eq!(
        call(None, Method::GET, &base, Value::Null).await.0,
        StatusCode::OK
    );

    // One created without a project is no project's.
    let (_, unowned) = call(None, Method::POST, "/v1/conversations", json!({})).await;
    let unowned = format!("/v1/conversations/{}", unowned["id"].as_str().unwrap());
    assert_eq!(
        call(A, Method::GET, &unowned, Value::Null).await.0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn files_and_uploads_are_private_over_http() {
    let scratch = Scratch::new();
    let services = Services::open(&scratch.root).unwrap();
    let call = |project, method, uri: &str, body: Value| {
        let (services, uri) = (&services, uri.to_owned());
        async move { services_call(services, project, (method, &uri, &body)).await }
    };
    let store = services.store();
    let owned = file(&store.scoped(A), FilePurpose::Assistants, "a.txt", "alpha");
    let theirs = file(&store.scoped(B), FilePurpose::Assistants, "b.txt", "beta");
    let unowned = file(store, FilePurpose::Assistants, "u.txt", "legacy");
    services_hidden(
        &services,
        vec![
            (Method::GET, format!("/v1/files/{owned}"), Value::Null),
            (
                Method::GET,
                format!("/v1/files/{owned}/content"),
                Value::Null,
            ),
            (Method::DELETE, format!("/v1/files/{owned}"), Value::Null),
            (Method::GET, format!("/v1/files?after={owned}"), Value::Null),
        ],
        &owned,
    )
    .await;
    let listed = |project| async move {
        let (_, list) = call(project, Method::GET, "/v1/files", Value::Null).await;
        let mut ids: Vec<String> = list["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|file| file["id"].as_str().unwrap().to_owned())
            .collect();
        ids.sort();
        ids
    };
    assert_eq!(listed(A).await, std::slice::from_ref(&owned));
    assert_eq!(listed(B).await, std::slice::from_ref(&theirs));
    let mut all = vec![owned.clone(), theirs, unowned.clone()];
    all.sort();
    assert_eq!(listed(None).await, all);
    let (status, _) = call(A, Method::GET, &format!("/v1/files/{unowned}"), Value::Null).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "unowned files are no project's"
    );
    let content = format!("/v1/files/{owned}/content");
    assert_eq!(
        call(A, Method::GET, &content, Value::Null).await.0,
        StatusCode::OK
    );

    let (status, upload) = call(
        A,
        Method::POST,
        "/v1/uploads",
        json!({"filename":"u.txt","purpose":"assistants","bytes":5,"mime_type":"text/plain"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let upload = upload["id"].as_str().unwrap().to_owned();
    let base = format!("/v1/uploads/{upload}");
    services_hidden(
        &services,
        vec![
            (Method::POST, format!("{base}/parts"), Value::Null),
            (
                Method::POST,
                format!("{base}/complete"),
                json!({"part_ids":["part_x"]}),
            ),
            (Method::POST, format!("{base}/cancel"), Value::Null),
        ],
        &upload,
    )
    .await;
    let (status, cancelled) = call(A, Method::POST, &format!("{base}/cancel"), Value::Null).await;
    assert_eq!(
        (status, &cancelled["status"]),
        (StatusCode::OK, &json!("cancelled"))
    );
}

// --- Vector stores ------------------------------------------------------------

async fn vector_call(
    service: &VectorStores,
    project: Option<&str>,
    (method, uri, body): (Method, &str, &Value),
) -> (StatusCode, Value) {
    let (parts, body) = request(method, uri, body).into_parts();
    match service.dispatch(&parts, body, project).await {
        Ok(value) => (StatusCode::OK, value),
        Err((status, message)) => (status, json!({"error":{"message":message}})),
    }
}

#[tokio::test]
#[expect(clippy::too_many_lines)]
async fn vector_stores_are_private_over_http_and_to_file_search() {
    let scratch = Scratch::new();
    let store = Store::open(&scratch.root).unwrap();
    let service = VectorStores::start(&store).unwrap();
    let call = |project, method, uri: String, body: Value| {
        let service = &service;
        async move { vector_call(service, project, (method, &uri, &body)).await }
    };
    let owned = file(
        &store.scoped(A),
        FilePurpose::Assistants,
        "a.txt",
        "alpha beta",
    );
    let theirs = file(&store.scoped(B), FilePurpose::Assistants, "b.txt", "gamma");

    // Another project's file cannot be attached.
    let create = |file: &str| json!({"file_ids":[file]});
    let refused = call(B, Method::POST, "/v1/vector_stores".into(), create(&owned)).await;
    let missing = call(
        B,
        Method::POST,
        "/v1/vector_stores".into(),
        create(&absent(&owned)),
    )
    .await;
    assert_eq!(refused.0, StatusCode::NOT_FOUND);
    same_as_missing(refused, missing, &owned);

    let (status, created) = call(A, Method::POST, "/v1/vector_stores".into(), create(&owned)).await;
    assert_eq!(status, StatusCode::OK, "{created}");
    let id = created["id"].as_str().unwrap().to_owned();
    let (status, batch) = call(
        A,
        Method::POST,
        format!("/v1/vector_stores/{id}/file_batches"),
        json!({"file_ids":[owned]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{batch}");
    let batch = batch["id"].as_str().unwrap().to_owned();
    let refused = call(
        A,
        Method::POST,
        format!("/v1/vector_stores/{id}/files"),
        json!({"file_id":theirs}),
    )
    .await;
    assert_eq!(refused.0, StatusCode::NOT_FOUND, "{refused:?}");

    let base = format!("/v1/vector_stores/{id}");
    let routes = vec![
        (Method::GET, base.clone(), Value::Null),
        (Method::POST, base.clone(), json!({"name":"x"})),
        (Method::GET, format!("{base}/files"), Value::Null),
        (
            Method::POST,
            format!("{base}/files"),
            json!({"file_id":theirs}),
        ),
        (Method::GET, format!("{base}/files/{owned}"), Value::Null),
        (
            Method::POST,
            format!("{base}/files/{owned}"),
            json!({"attributes":{}}),
        ),
        (
            Method::GET,
            format!("{base}/files/{owned}/content"),
            Value::Null,
        ),
        (Method::DELETE, format!("{base}/files/{owned}"), Value::Null),
        (
            Method::POST,
            format!("{base}/file_batches"),
            json!({"file_ids":[theirs]}),
        ),
        (
            Method::GET,
            format!("{base}/file_batches/{batch}"),
            Value::Null,
        ),
        (
            Method::POST,
            format!("{base}/file_batches/{batch}/cancel"),
            Value::Null,
        ),
        (
            Method::GET,
            format!("{base}/file_batches/{batch}/files"),
            Value::Null,
        ),
        (
            Method::POST,
            format!("{base}/search"),
            json!({"query":"alpha"}),
        ),
        (
            Method::GET,
            format!("/v1/vector_stores?after={id}"),
            Value::Null,
        ),
        (Method::DELETE, base.clone(), Value::Null),
    ];
    for (method, uri, body) in routes {
        let fake = uri.replace(&id, &absent(&id));
        let hidden = call(B, method.clone(), uri.clone(), body.clone()).await;
        let missing = call(B, method, fake, body).await;
        assert_eq!(hidden.0, StatusCode::NOT_FOUND, "{uri}: {hidden:?}");
        same_as_missing(hidden, missing, &id);
    }
    let (_, list) = call(B, Method::GET, "/v1/vector_stores".into(), Value::Null).await;
    assert_eq!(list["data"], json!([]));
    let (_, list) = call(A, Method::GET, "/v1/vector_stores".into(), Value::Null).await;
    assert_eq!(list["data"][0]["id"], json!(id));
    assert_eq!(
        call(None, Method::GET, base.clone(), Value::Null).await.0,
        StatusCode::OK
    );
    assert_eq!(
        call(
            A,
            Method::POST,
            format!("{base}/search"),
            json!({"query":"alpha"})
        )
        .await
        .0,
        StatusCode::OK
    );

    // The `file_search` tool's `vector_store_ids`.
    same_as_missing(
        service
            .check_searchable(vec![id.clone()], B)
            .await
            .unwrap_err(),
        service
            .check_searchable(vec![absent(&id)], B)
            .await
            .unwrap_err(),
        &id,
    );
    assert!(service.check_searchable(vec![id.clone()], A).await.is_ok());
    assert!(
        service
            .check_searchable(vec![id.clone()], None)
            .await
            .is_ok()
    );

    // A store created without a project is no project's.
    let (_, unowned) = call(None, Method::POST, "/v1/vector_stores".into(), json!({})).await;
    let unowned = unowned["id"].as_str().unwrap().to_owned();
    let (status, _) = call(
        A,
        Method::GET,
        format!("/v1/vector_stores/{unowned}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(service.check_searchable(vec![unowned], A).await.is_err());
    service.close();
}

// --- Batches --------------------------------------------------------------------

async fn batch_call(
    batches: &Batches,
    project: Option<&str>,
    (method, uri, body): (Method, &str, &Value),
) -> (StatusCode, Value) {
    let (parts, body) = request(method, uri, body).into_parts();
    let owner = project.map(|project| Owner {
        project_id: project.to_owned(),
        api_key_id: format!("key_{project}"),
    });
    match batches.dispatch(&parts, body, owner).await {
        Ok(value) => (StatusCode::OK, value),
        Err((status, message)) => (status, json!({"error":{"message":message}})),
    }
}

#[tokio::test]
async fn batches_are_private_over_http() {
    let scratch = Scratch::new();
    let store = Store::open(&scratch.root).unwrap();
    // No engine: every line fails, which these tests do not look at.
    let submit: Submit = Arc::new(|_| Err(crate::Error::Generation("no engine".into())));
    let batches = Batches::spawn(
        store.clone(),
        submit,
        Config {
            model: "local".into(),
            thinking: false,
            cipher: None,
            depth: QueueDepth::default(),
            meter: Meter::default(),
        },
    )
    .unwrap();
    let call = |project, method, uri: String, body: Value| {
        let batches = &batches;
        async move { batch_call(batches, project, (method, &uri, &body)).await }
    };
    let line = json!({"custom_id":"a","method":"POST","url":"/v1/chat/completions",
                      "body":{"model":"local","messages":[{"role":"user","content":"hi"}]}});
    let input = |store: &Store| file(store, FilePurpose::Batch, "in.jsonl", &format!("{line}\n"));
    let create = |file: &str| json!({"input_file_id":file,"endpoint":"/v1/chat/completions","completion_window":"24h"});
    let mine = input(&store.scoped(A));

    // Another project's input file is refused as an unknown one.
    let refused = call(B, Method::POST, "/v1/batches".into(), create(&mine)).await;
    let missing = call(
        B,
        Method::POST,
        "/v1/batches".into(),
        create(&absent(&mine)),
    )
    .await;
    assert_eq!(refused.0, StatusCode::BAD_REQUEST);
    same_as_missing(refused, missing, &mine);

    let (status, batch) = call(A, Method::POST, "/v1/batches".into(), create(&mine)).await;
    assert_eq!(status, StatusCode::OK, "{batch}");
    let id = batch["id"].as_str().unwrap().to_owned();
    for (method, uri) in [
        (Method::GET, format!("/v1/batches/{id}")),
        (Method::POST, format!("/v1/batches/{id}/cancel")),
        (Method::GET, format!("/v1/batches?after={id}")),
    ] {
        let fake = uri.replace(&id, &absent(&id));
        let hidden = call(B, method.clone(), uri.clone(), Value::Null).await;
        let missing = call(B, method, fake, Value::Null).await;
        assert_eq!(hidden.0, StatusCode::NOT_FOUND, "{uri}: {hidden:?}");
        same_as_missing(hidden, missing, &id);
    }
    let (_, list) = call(B, Method::GET, "/v1/batches".into(), Value::Null).await;
    assert_eq!(list["data"], json!([]));
    let (_, list) = call(A, Method::GET, "/v1/batches".into(), Value::Null).await;
    assert_eq!(list["data"][0]["id"], json!(id));
    let (status, _) = call(None, Method::GET, format!("/v1/batches/{id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);

    // A batch created without a project is no project's.
    let (_, unowned) = call(
        None,
        Method::POST,
        "/v1/batches".into(),
        create(&input(&store)),
    )
    .await;
    let unowned = unowned["id"].as_str().unwrap().to_owned();
    let (status, _) = call(
        A,
        Method::GET,
        format!("/v1/batches/{unowned}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, list) = call(None, Method::GET, "/v1/batches".into(), Value::Null).await;
    assert_eq!(list["data"].as_array().unwrap().len(), 2);
    batches.close();
    batches.drained().await;
}
