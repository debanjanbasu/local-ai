//! HTTP-level tests for the native services adapters: real `Request`s, real
//! multipart bodies and a real store in a private temporary directory,
//! dispatched without a model.

use std::path::PathBuf;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};

use super::{MAX_REQUEST_BYTES, Reply, Services, matches};

const BOUNDARY: &str = "local-ai-test-boundary";

/// A store in a temporary directory, removed when dropped.
struct Scratch {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        // SQLite refuses a database path through a symlink, and the macOS
        // temporary directory is under one (`/var` -> `/private/var`).
        let root = dir
            .path()
            .canonicalize()
            .expect("canonical temporary directory")
            .join("services");
        Self { _dir: dir, root }
    }

    fn open(&self) -> Services {
        Services::open(&self.root).expect("services open")
    }
}

/// What a request answered: status, JSON body (the error object for
/// errors), and the raw bytes of a download.
struct Answer {
    status: StatusCode,
    json: Value,
    bytes: Option<(Vec<u8>, header::HeaderMap)>,
}

impl Answer {
    fn message(&self) -> &str {
        self.json["error"]["message"].as_str().unwrap_or_default()
    }
}

async fn send(services: &Services, request: Request<Body>) -> Answer {
    let (parts, body) = request.into_parts();
    match services.dispatch(&parts, body).await {
        Reply::Json(status, json) => Answer {
            status,
            json,
            bytes: None,
        },
        Reply::Error(status, message) => Answer {
            status,
            json: json!({"error":{"message":message}}),
            bytes: None,
        },
        Reply::Content(download) => {
            let response = download.into_response();
            let headers = response.headers().clone();
            let bytes = response
                .into_body()
                .collect()
                .await
                .expect("download body")
                .to_bytes()
                .to_vec();
            Answer {
                status: StatusCode::OK,
                json: Value::Null,
                bytes: Some((bytes, headers)),
            }
        }
    }
}

fn request(method: Method, uri: &str, body: impl Into<Body>) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body.into())
        .expect("request")
}

async fn call(services: &Services, method: Method, uri: &str, body: &Value) -> Answer {
    let body = if body.is_null() {
        String::new()
    } else {
        body.to_string()
    };
    send(services, request(method, uri, body)).await
}

async fn raw(services: &Services, method: Method, uri: &str, body: &str) -> Answer {
    send(services, request(method, uri, body.to_owned())).await
}

/// One multipart field: name, optional filename, content.
type FormField<'a> = (&'a str, Option<&'a str>, &'a [u8]);

fn form(fields: &[FormField<'_>]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, filename, content) in fields {
        body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
        let disposition = filename.map_or_else(
            || format!("Content-Disposition: form-data; name=\"{name}\"\r\n"),
            |filename| {
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n\
                     Content-Type: application/octet-stream\r\n"
                )
            },
        );
        body.extend_from_slice(disposition.as_bytes());
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(content);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    body
}

async fn post_form(services: &Services, uri: &str, fields: &[FormField<'_>]) -> Answer {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(form(fields)))
        .expect("request");
    send(services, request).await
}

async fn upload_file(services: &Services, filename: &str, content: &[u8]) -> Value {
    let answer = post_form(
        services,
        "/v1/files",
        &[
            ("purpose", None, b"assistants"),
            ("file", Some(filename), content),
        ],
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK, "{}", answer.json);
    answer.json
}

async fn download(services: &Services, id: &str) -> Vec<u8> {
    let answer = call(
        services,
        Method::GET,
        &format!("/v1/files/{id}/content"),
        &Value::Null,
    )
    .await;
    assert_eq!(answer.status, StatusCode::OK, "{}", answer.json);
    answer.bytes.expect("download").0
}

fn message(role: &str, text: &str) -> Value {
    json!({"type":"message","role":role,"content":text})
}

fn ids(list: &Value) -> Vec<&str> {
    list["data"]
        .as_array()
        .map(|data| data.iter().filter_map(|item| item["id"].as_str()).collect())
        .unwrap_or_default()
}

#[test]
fn only_the_service_prefixes_match() {
    for path in [
        "/v1/conversations",
        "/v1/conversations/conv_1/items",
        "/v1/files",
        "/v1/files/file_1/content",
        "/v1/uploads",
        "/v1/uploads/upload_1/parts",
    ] {
        assert!(matches(path), "{path}");
    }
    for path in [
        "/v1/conversationsx",
        "/v1/filesystem",
        "/v1/responses",
        "/v1/models",
        "/conversations",
    ] {
        assert!(!matches(path), "{path}");
    }
}

#[tokio::test]
async fn conversations_support_crud_and_items_and_deletion_hides_everything() {
    let scratch = Scratch::new();
    let services = scratch.open();

    let created = call(
        &services,
        Method::POST,
        "/v1/conversations",
        &json!({"metadata":{"topic":"demo"},"items":[message("user","hello")]}),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK, "{}", created.json);
    assert_eq!(created.json["object"], "conversation");
    assert_eq!(created.json["metadata"], json!({"topic":"demo"}));
    assert!(created.json["created_at"].is_u64());
    assert!(created.json.get("version").is_none());
    let id = created.json["id"].as_str().expect("id").to_owned();
    assert!(id.starts_with("conv_"));
    let base = format!("/v1/conversations/{id}");

    let empty = raw(&services, Method::POST, "/v1/conversations", "").await;
    assert_eq!(empty.status, StatusCode::OK, "{}", empty.json);
    assert_eq!(empty.json["metadata"], json!({}));

    let fetched = call(&services, Method::GET, &base, &Value::Null).await;
    assert_eq!(fetched.json, created.json);

    let updated = call(
        &services,
        Method::POST,
        &base,
        &json!({"metadata":{"topic":"changed"}}),
    )
    .await;
    assert_eq!(updated.status, StatusCode::OK, "{}", updated.json);
    assert_eq!(updated.json["metadata"], json!({"topic":"changed"}));

    let added = call(
        &services,
        Method::POST,
        &format!("{base}/items?include=reasoning.encrypted_content"),
        &json!({"items":[message("assistant","hi there"),{"type":"function_call","call_id":"c1","name":"read","arguments":"{}"}]}),
    )
    .await;
    assert_eq!(added.status, StatusCode::OK, "{}", added.json);
    assert_eq!(added.json["object"], "list");
    assert_eq!(added.json["data"].as_array().map(Vec::len), Some(2));
    assert_eq!(added.json["data"][0]["content"][0]["type"], "output_text");
    assert_eq!(added.json["data"][0]["status"], "completed");
    let call_id = added.json["data"][1]["id"].as_str().expect("id").to_owned();

    let listed = call(
        &services,
        Method::GET,
        &format!("{base}/items?order=asc"),
        &Value::Null,
    )
    .await;
    assert_eq!(listed.status, StatusCode::OK, "{}", listed.json);
    assert_eq!(listed.json["data"][0]["content"][0]["text"], "hello");
    assert_eq!(listed.json["data"][2]["id"], call_id.as_str());
    assert_eq!(listed.json["has_more"], false);
    assert_eq!(listed.json["first_id"], listed.json["data"][0]["id"]);

    let item = call(
        &services,
        Method::GET,
        &format!("{base}/items/{call_id}"),
        &Value::Null,
    )
    .await;
    assert_eq!(item.json["type"], "function_call");
    assert_eq!(item.json["name"], "read");

    let after_delete = call(
        &services,
        Method::DELETE,
        &format!("{base}/items/{call_id}"),
        &Value::Null,
    )
    .await;
    assert_eq!(after_delete.status, StatusCode::OK, "{}", after_delete.json);
    assert_eq!(after_delete.json["object"], "conversation");
    assert_eq!(after_delete.json["id"], id.as_str());
    let gone = call(
        &services,
        Method::GET,
        &format!("{base}/items/{call_id}"),
        &Value::Null,
    )
    .await;
    assert_eq!(gone.status, StatusCode::NOT_FOUND);

    deleted_conversations_are_not_found(&services, &id).await;
}

async fn deleted_conversations_are_not_found(services: &Services, id: &str) {
    let base = format!("/v1/conversations/{id}");
    let deleted = call(services, Method::DELETE, &base, &Value::Null).await;
    assert_eq!(
        deleted.json,
        json!({"id":id,"object":"conversation.deleted","deleted":true})
    );
    for (method, uri, body) in [
        (Method::GET, base.clone(), Value::Null),
        (Method::DELETE, base.clone(), Value::Null),
        (Method::POST, base.clone(), json!({"metadata":{}})),
        (Method::GET, format!("{base}/items"), Value::Null),
        (
            Method::POST,
            format!("{base}/items"),
            json!({"items":[message("user","late")]}),
        ),
    ] {
        let answer = call(services, method.clone(), &uri, &body).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND, "{method} {uri}");
        assert!(answer.json["error"]["message"].is_string());
    }
}

#[tokio::test]
async fn items_page_newest_first_twenty_at_a_time_by_default() {
    let scratch = Scratch::new();
    let services = scratch.open();
    let items: Vec<Value> = (0..20).map(|n| message("user", &format!("m{n}"))).collect();
    let created = call(
        &services,
        Method::POST,
        "/v1/conversations",
        &json!({"items":items}),
    )
    .await;
    let base = format!(
        "/v1/conversations/{}",
        created.json["id"].as_str().expect("id")
    );
    let more: Vec<Value> = (20..25)
        .map(|n| message("user", &format!("m{n}")))
        .collect();
    let added = call(
        &services,
        Method::POST,
        &format!("{base}/items"),
        &json!({"items":more}),
    )
    .await;
    assert_eq!(added.status, StatusCode::OK, "{}", added.json);

    let first = call(
        &services,
        Method::GET,
        &format!("{base}/items"),
        &Value::Null,
    )
    .await;
    assert_eq!(first.json["data"].as_array().map(Vec::len), Some(20));
    assert_eq!(first.json["data"][0]["content"][0]["text"], "m24");
    assert_eq!(first.json["data"][19]["content"][0]["text"], "m5");
    assert_eq!(first.json["has_more"], true);
    let last = first.json["last_id"].as_str().expect("last id");
    let second = call(
        &services,
        Method::GET,
        &format!("{base}/items?after={last}"),
        &Value::Null,
    )
    .await;
    assert_eq!(second.json["data"].as_array().map(Vec::len), Some(5));
    assert_eq!(second.json["data"][4]["content"][0]["text"], "m0");
    assert_eq!(second.json["has_more"], false);

    let small = call(
        &services,
        Method::GET,
        &format!("{base}/items?limit=2&order=asc"),
        &Value::Null,
    )
    .await;
    assert_eq!(small.json["data"][1]["content"][0]["text"], "m1");
    assert_eq!(small.json["has_more"], true);

    for query in [
        "limit=0",
        "limit=101",
        "limit=x",
        "order=up",
        "limit=1&limit=2",
        "page=2",
        "include=message.output_text.logprobs",
        "after=msg_unknown",
    ] {
        let answer = call(
            &services,
            Method::GET,
            &format!("{base}/items?{query}"),
            &Value::Null,
        )
        .await;
        assert_eq!(
            answer.status,
            StatusCode::BAD_REQUEST,
            "{query}: {}",
            answer.json
        );
    }
}

#[tokio::test]
async fn json_bodies_are_strict_and_bounded() {
    let scratch = Scratch::new();
    let services = scratch.open();
    let cases = [
        (r#"{"metadata":{},"extra":1}"#, "unknown field"),
        (r#"{"metadata":{"a":"1"},"metadata":{}}"#, "duplicate key"),
        (
            r#"{"items":[{"type":"message","role":"user","role":"user","content":"x"}]}"#,
            "duplicate key",
        ),
        (r#"{"metadata":{"k":"v","k":"w"}}"#, "duplicate key"),
        (r#"{"metadata":{"k":1}}"#, "invalid request body"),
        ("[]", "must be an object"),
        ("{", "invalid JSON body"),
        ("{} {}", "invalid JSON body"),
        (
            r#"{"items":[{"type":"input_image","image_url":"x"}]}"#,
            "not supported",
        ),
    ];
    for (body, expected) in cases {
        let answer = raw(&services, Method::POST, "/v1/conversations", body).await;
        assert_eq!(answer.status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            answer.message().contains(expected),
            "{body}: {}",
            answer.message()
        );
    }
    let huge = format!(
        r#"{{"metadata":{{"k":"{}"}}}}"#,
        "x".repeat(MAX_REQUEST_BYTES)
    );
    let answer = raw(&services, Method::POST, "/v1/conversations", &huge).await;
    assert_eq!(answer.status, StatusCode::PAYLOAD_TOO_LARGE);

    let created = call(&services, Method::POST, "/v1/conversations", &json!({})).await;
    let base = format!(
        "/v1/conversations/{}",
        created.json["id"].as_str().expect("id")
    );
    let answer = call(&services, Method::POST, &base, &json!({})).await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
    assert!(answer.message().contains("metadata is required"));
    let answer = call(
        &services,
        Method::POST,
        &format!("{base}/items"),
        &Value::Null,
    )
    .await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
    let answer = call(
        &services,
        Method::POST,
        &format!("{base}/items"),
        &json!({"items":[]}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);
    let answer = call(&services, Method::GET, &format!("{base}?x=1"), &Value::Null).await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST);

    for (method, uri) in [
        (Method::PUT, "/v1/conversations"),
        (Method::GET, "/v1/conversations"),
        (Method::GET, "/v1/conversations/"),
        (Method::GET, "/v1/files/a/b/c"),
        (Method::GET, "/v1/uploads/upload_1"),
    ] {
        let answer = call(&services, method.clone(), uri, &Value::Null).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND, "{method} {uri}");
    }
}

#[tokio::test]
async fn a_multipart_file_round_trips_byte_for_byte() {
    let scratch = Scratch::new();
    let services = scratch.open();
    // Every byte value, CRLFs, and a prefix of the delimiter.
    let mut content: Vec<u8> = (0..=255_u8).cycle().take(200_000).collect();
    content.extend_from_slice(b"\r\n--local-ai-test\r\n\r\n--");
    content.extend_from_slice(&[0, 0xff, b'\r', b'\n']);

    let first = upload_file(&services, "first.bin", b"one").await;
    let file = upload_file(&services, "data.bin", &content).await;
    assert_eq!(file["object"], "file");
    assert_eq!(file["bytes"], content.len());
    assert_eq!(file["filename"], "data.bin");
    assert_eq!(file["purpose"], "assistants");
    assert_eq!(file["status"], "processed");
    assert!(file.get("expires_at").is_none());
    assert!(file.get("status_details").is_none());
    let id = file["id"].as_str().expect("id");

    let answer = call(
        &services,
        Method::GET,
        &format!("/v1/files/{id}/content"),
        &Value::Null,
    )
    .await;
    let (bytes, headers) = answer.bytes.expect("download");
    assert_eq!(bytes, content);
    assert_eq!(
        headers[header::CONTENT_LENGTH],
        content.len().to_string().as_str()
    );
    assert_eq!(headers[header::CONTENT_TYPE], "application/octet-stream");

    let fetched = call(
        &services,
        Method::GET,
        &format!("/v1/files/{id}"),
        &Value::Null,
    )
    .await;
    assert_eq!(fetched.json, file);

    let listed = call(&services, Method::GET, "/v1/files", &Value::Null).await;
    assert_eq!(listed.json["object"], "list");
    assert_eq!(ids(&listed.json), [id, first["id"].as_str().expect("id")]);
    assert_eq!(listed.json["has_more"], false);
    let page = call(
        &services,
        Method::GET,
        "/v1/files?limit=1&order=asc",
        &Value::Null,
    )
    .await;
    assert_eq!(ids(&page.json), [first["id"].as_str().expect("id")]);
    assert_eq!(page.json["has_more"], true);
    let filtered = call(
        &services,
        Method::GET,
        "/v1/files?purpose=batch",
        &Value::Null,
    )
    .await;
    assert_eq!(ids(&filtered.json), Vec::<&str>::new());

    let expiring = post_form(
        &services,
        "/v1/files",
        &[
            ("file", Some("later.txt"), b"soon gone"),
            ("purpose", None, b"user_data"),
            ("expires_after[anchor]", None, b"created_at"),
            ("expires_after[seconds]", None, b"3600"),
        ],
    )
    .await;
    assert_eq!(expiring.status, StatusCode::OK, "{}", expiring.json);
    assert_eq!(
        expiring.json["expires_at"].as_u64(),
        expiring.json["created_at"].as_u64().map(|at| at + 3600)
    );

    let deleted = call(
        &services,
        Method::DELETE,
        &format!("/v1/files/{id}"),
        &Value::Null,
    )
    .await;
    assert_eq!(
        deleted.json,
        json!({"id":id,"object":"file","deleted":true})
    );
    for uri in [format!("/v1/files/{id}"), format!("/v1/files/{id}/content")] {
        let answer = call(&services, Method::GET, &uri, &Value::Null).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND, "{uri}");
    }
    let again = call(
        &services,
        Method::DELETE,
        &format!("/v1/files/{id}"),
        &Value::Null,
    )
    .await;
    assert_eq!(again.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn malformed_file_forms_are_refused() {
    let scratch = Scratch::new();
    let services = scratch.open();
    let file: FormField<'_> = ("file", Some("a.txt"), b"abc");
    let purpose: FormField<'_> = ("purpose", None, b"assistants");
    let cases: [(&[FormField<'_>], &str); 9] = [
        (&[purpose], "file field is required"),
        (&[file], "purpose field is required"),
        (&[file, purpose, purpose], "repeated"),
        (&[file, file, purpose], "repeated"),
        (&[file, purpose, ("model", None, b"x")], "not supported"),
        (&[("file", None, b"abc"), purpose], "filename"),
        (
            &[file, ("purpose", None, b"weights")],
            "unsupported file purpose",
        ),
        (
            &[file, purpose, ("expires_after[seconds]", None, b"3600")],
            "together",
        ),
        (
            &[
                file,
                purpose,
                ("expires_after[anchor]", None, b"created_at"),
                ("expires_after[seconds]", None, b"60"),
            ],
            "expires_after.seconds",
        ),
    ];
    for (fields, expected) in cases {
        let answer = post_form(&services, "/v1/files", fields).await;
        assert_eq!(answer.status, StatusCode::BAD_REQUEST, "{expected}");
        assert!(
            answer.message().contains(expected),
            "{expected}: {}",
            answer.message()
        );
    }
    let empty = post_form(
        &services,
        "/v1/files",
        &[("file", Some("a.txt"), b""), purpose],
    )
    .await;
    assert_eq!(empty.status, StatusCode::BAD_REQUEST);
    let batch = post_form(
        &services,
        "/v1/files",
        &[("file", Some("a.txt"), b"{}"), ("purpose", None, b"batch")],
    )
    .await;
    assert_eq!(batch.status, StatusCode::BAD_REQUEST);

    let not_multipart = raw(&services, Method::POST, "/v1/files", "{}").await;
    assert_eq!(not_multipart.status, StatusCode::BAD_REQUEST);
    let truncated = Request::builder()
        .method(Method::POST)
        .uri("/v1/files")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a\"\r\n\r\nabc"
        )))
        .expect("request");
    let answer = send(&services, truncated).await;
    assert_eq!(answer.status, StatusCode::BAD_REQUEST, "{}", answer.json);

    // Nothing above created a file.
    let listed = call(&services, Method::GET, "/v1/files", &Value::Null).await;
    assert_eq!(ids(&listed.json), Vec::<&str>::new());
}

async fn create_upload(services: &Services, bytes: usize) -> String {
    let created = call(
        services,
        Method::POST,
        "/v1/uploads",
        &json!({"filename":"joined.txt","purpose":"assistants","bytes":bytes,"mime_type":"text/plain"}),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK, "{}", created.json);
    assert_eq!(created.json["object"], "upload");
    assert_eq!(created.json["status"], "pending");
    assert_eq!(created.json["file"], Value::Null);
    assert_eq!(
        created.json["expires_at"].as_u64(),
        created.json["created_at"].as_u64().map(|at| at + 3600)
    );
    created.json["id"].as_str().expect("id").to_owned()
}

async fn add_part(services: &Services, upload: &str, data: &[u8]) -> Answer {
    post_form(
        services,
        &format!("/v1/uploads/{upload}/parts"),
        &[("data", Some("blob"), data)],
    )
    .await
}

#[tokio::test]
async fn uploads_assemble_parts_in_the_order_completion_names_them() {
    let scratch = Scratch::new();
    let services = scratch.open();
    let upload = create_upload(&services, 9).await;

    let first = add_part(&services, &upload, b"abc").await;
    assert_eq!(first.status, StatusCode::OK, "{}", first.json);
    assert_eq!(first.json["object"], "upload.part");
    assert_eq!(first.json["upload_id"], upload.as_str());
    let second = add_part(&services, &upload, b"de\x00\xff\r\n").await;
    assert_eq!(second.status, StatusCode::OK, "{}", second.json);
    let (first, second) = (
        first.json["id"].as_str().expect("id").to_owned(),
        second.json["id"].as_str().expect("id").to_owned(),
    );
    let over = add_part(&services, &upload, b"x").await;
    assert_eq!(over.status, StatusCode::CONFLICT, "{}", over.json);

    let complete = format!("/v1/uploads/{upload}/complete");
    for body in [
        json!({"part_ids":[first]}),
        json!({"part_ids":[first, first]}),
        json!({"part_ids":[]}),
        json!({"part_ids":[first, "part_unknown"]}),
        json!({"part_ids":[first, second],"extra":true}),
    ] {
        let answer = call(&services, Method::POST, &complete, &body).await;
        assert_eq!(
            answer.status,
            StatusCode::BAD_REQUEST,
            "{body}: {}",
            answer.json
        );
    }

    let done = call(
        &services,
        Method::POST,
        &complete,
        &json!({"part_ids":[second, first]}),
    )
    .await;
    assert_eq!(done.status, StatusCode::OK, "{}", done.json);
    assert_eq!(done.json["status"], "completed");
    assert_eq!(done.json["file"]["object"], "file");
    assert_eq!(done.json["file"]["bytes"], 9);
    assert_eq!(done.json["file"]["filename"], "joined.txt");
    let file = done.json["file"]["id"].as_str().expect("file id");
    assert_eq!(download(&services, file).await, b"de\x00\xff\r\nabc");

    // A completed upload is a conflict, not gone.
    let answer = add_part(&services, &upload, b"x").await;
    assert_eq!(answer.status, StatusCode::CONFLICT, "{}", answer.json);
    let answer = call(
        &services,
        Method::POST,
        &complete,
        &json!({"part_ids":[second, first]}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::CONFLICT, "{}", answer.json);
    let answer = call(
        &services,
        Method::POST,
        &format!("/v1/uploads/{upload}/cancel"),
        &Value::Null,
    )
    .await;
    assert_eq!(answer.status, StatusCode::CONFLICT, "{}", answer.json);
}

#[tokio::test]
async fn a_cancelled_upload_is_gone_and_bad_uploads_are_refused() {
    let scratch = Scratch::new();
    let services = scratch.open();
    let upload = create_upload(&services, 4).await;
    let part = add_part(&services, &upload, b"ab").await;
    let part = part.json["id"].as_str().expect("part id").to_owned();
    let too_big = add_part(&services, &upload, b"abc").await;
    assert_eq!(too_big.status, StatusCode::BAD_REQUEST, "{}", too_big.json);
    let unknown = post_form(
        &services,
        &format!("/v1/uploads/{upload}/parts"),
        &[("file", Some("blob"), b"ab")],
    )
    .await;
    assert_eq!(unknown.status, StatusCode::BAD_REQUEST);

    let cancel = format!("/v1/uploads/{upload}/cancel");
    let refused = call(&services, Method::POST, &cancel, &json!({"reason":"x"})).await;
    assert_eq!(refused.status, StatusCode::BAD_REQUEST);
    let cancelled = call(&services, Method::POST, &cancel, &Value::Null).await;
    assert_eq!(cancelled.status, StatusCode::OK, "{}", cancelled.json);
    assert_eq!(cancelled.json["status"], "cancelled");

    let answer = add_part(&services, &upload, b"cd").await;
    assert_eq!(answer.status, StatusCode::GONE, "{}", answer.json);
    let answer = call(
        &services,
        Method::POST,
        &format!("/v1/uploads/{upload}/complete"),
        &json!({"part_ids":[part]}),
    )
    .await;
    assert_eq!(answer.status, StatusCode::GONE, "{}", answer.json);
    let answer = call(&services, Method::POST, &cancel, &json!({})).await;
    assert_eq!(answer.status, StatusCode::GONE, "{}", answer.json);

    let missing = add_part(&services, "upload_missing", b"ab").await;
    assert_eq!(missing.status, StatusCode::NOT_FOUND);
    let missing = call(
        &services,
        Method::POST,
        "/v1/uploads/upload_missing/cancel",
        &Value::Null,
    )
    .await;
    assert_eq!(missing.status, StatusCode::NOT_FOUND);

    for body in [
        json!({"filename":"a.txt","purpose":"assistants","bytes":0,"mime_type":"text/plain"}),
        json!({"filename":"a.txt","purpose":"weights","bytes":1,"mime_type":"text/plain"}),
        json!({"filename":"a.txt","purpose":"assistants","bytes":1}),
        json!({"filename":"a.txt","purpose":"assistants","bytes":1,"mime_type":"plain"}),
        json!({"filename":"a.txt","purpose":"assistants","bytes":1,"mime_type":"text/plain","expires_after":{"anchor":"created_at","seconds":1}}),
        json!({"filename":"a.txt","purpose":"assistants","bytes":1,"mime_type":"text/plain","expires_after":{"anchor":"last_active_at","seconds":3600}}),
        json!({"filename":"a.txt","purpose":"assistants","bytes":1,"mime_type":"text/plain","md5":"x"}),
    ] {
        let answer = call(&services, Method::POST, "/v1/uploads", &body).await;
        assert_eq!(
            answer.status,
            StatusCode::BAD_REQUEST,
            "{body}: {}",
            answer.json
        );
    }
}

#[tokio::test]
async fn conversations_and_files_survive_reopening_the_store() {
    let scratch = Scratch::new();
    let (conversation, file) = {
        let services = scratch.open();
        let created = call(
            &services,
            Method::POST,
            "/v1/conversations",
            &json!({"metadata":{"k":"v"},"items":[message("user","kept")]}),
        )
        .await;
        let file = upload_file(&services, "kept.bin", b"\x00kept\xff").await;
        (created.json, file)
    };
    let services = scratch.open();
    let id = conversation["id"].as_str().expect("id");
    let fetched = call(
        &services,
        Method::GET,
        &format!("/v1/conversations/{id}"),
        &Value::Null,
    )
    .await;
    assert_eq!(fetched.json, conversation);
    let items = call(
        &services,
        Method::GET,
        &format!("/v1/conversations/{id}/items"),
        &Value::Null,
    )
    .await;
    assert_eq!(items.json["data"][0]["content"][0]["text"], "kept");
    let file_id = file["id"].as_str().expect("file id");
    let fetched = call(
        &services,
        Method::GET,
        &format!("/v1/files/{file_id}"),
        &Value::Null,
    )
    .await;
    assert_eq!(fetched.json, file);
    assert_eq!(download(&services, file_id).await, b"\x00kept\xff");
}
