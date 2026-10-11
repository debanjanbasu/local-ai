#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! HTTP-level tests for the Vector Stores adapter: real `Request`s against a
//! real store and indexer in a private temporary directory, without a model.
//! Every successful answer is checked against the pinned `OpenAI` schemas in
//! `fixtures/vector_stores_schemas.json` (a subset of the contract's
//! `components.schemas`, descriptions stripped).

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};

use local_services::Store;
use local_services::files::{CreateFile, FilePurpose};

use super::{VectorStores, matches};

const SCHEMAS: &str = include_str!("fixtures/vector_stores_schemas.json");

struct Scratch {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        // SQLite refuses a database path through a symlink (macOS `/var`).
        let root = dir
            .path()
            .canonicalize()
            .expect("canonical temporary directory")
            .join("services");
        Self { _dir: dir, root }
    }

    fn store(&self) -> Store {
        Store::open(&self.root).expect("store opens")
    }

    fn start(&self) -> Arc<VectorStores> {
        VectorStores::start(&self.store()).expect("vector stores start")
    }
}

fn upload(store: &Store, name: &str, text: &str) -> String {
    store
        .create_file(
            &CreateFile {
                filename: name.to_owned(),
                purpose: FilePurpose::Assistants,
                expires_after: None,
            },
            text.as_bytes(),
        )
        .unwrap()
        .id
}

async fn send(
    service: &VectorStores,
    method: Method,
    uri: &str,
    body: Option<&str>,
) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .body(body.map_or_else(Body::empty, |text| Body::from(text.to_owned())))
        .unwrap();
    let (parts, body) = request.into_parts();
    match service.dispatch(&parts, body).await {
        Ok(value) => (StatusCode::OK, value),
        Err((status, message)) => (status, json!({"error": {"message": message}})),
    }
}

/// Send and require a `200` that conforms to `schema`.
async fn ok(
    service: &VectorStores,
    method: Method,
    uri: &str,
    body: Option<Value>,
    schema: &str,
) -> Value {
    let text = body.map(|body| body.to_string());
    let (status, value) = send(service, method, uri, text.as_deref()).await;
    assert_eq!(status, StatusCode::OK, "{uri}: {value}");
    conforms(schema, &value);
    value
}

async fn refused(
    service: &VectorStores,
    method: Method,
    uri: &str,
    body: Option<&str>,
    expected: StatusCode,
) -> String {
    let (status, value) = send(service, method, uri, body).await;
    assert_eq!(status, expected, "{uri}: {value}");
    value["error"]["message"].as_str().unwrap().to_owned()
}

fn conforms(name: &str, value: &Value) {
    let document: Value = serde_json::from_str(SCHEMAS).unwrap();
    let schemas = &document["schemas"];
    let schema = json!({"$ref": format!("#/components/schemas/{name}")});
    if let Err(error) = check(schemas, &schema, value, "$") {
        panic!("{name} does not conform: {error}\n{value:#}");
    }
}

/// A validator for the JSON Schema subset the contract's response schemas
/// use: `$ref`, `type` (one or several), `enum`, `anyOf`/`oneOf`/`allOf`,
/// `required`, `properties`, `additionalProperties`, `maxProperties`,
/// `items`, `minimum`, `maximum`, `maxLength`.
fn check(schemas: &Value, schema: &Value, value: &Value, path: &str) -> Result<(), String> {
    if let Some(reference) = schema["$ref"].as_str() {
        let name = reference.rsplit('/').next().unwrap();
        let target = schemas
            .get(name)
            .ok_or_else(|| format!("{path}: unknown schema {name}"))?;
        return check(schemas, target, value, path);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(options) = schema[key].as_array() {
            let errors: Vec<String> = options
                .iter()
                .filter_map(|option| check(schemas, option, value, path).err())
                .collect();
            if errors.len() == options.len() {
                return Err(format!("{path}: no {key} branch matches: {errors:?}"));
            }
        }
    }
    if let Some(all) = schema["allOf"].as_array() {
        for part in all {
            check(schemas, part, value, path)?;
        }
    }
    let types: Vec<&str> = match &schema["type"] {
        Value::String(kind) => vec![kind.as_str()],
        Value::Array(kinds) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    if !types.is_empty() {
        let fits = types.iter().any(|kind| match *kind {
            "string" => value.is_string(),
            "integer" => value.is_i64() || value.is_u64(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            "null" => value.is_null(),
            _ => false,
        });
        if !fits {
            return Err(format!("{path}: {value} is not of type {types:?}"));
        }
    }
    if let Some(allowed) = schema["enum"].as_array()
        && !allowed.contains(value)
    {
        return Err(format!("{path}: {value} is not one of {allowed:?}"));
    }
    if let Some(number) = value.as_f64() {
        if schema["minimum"].as_f64().is_some_and(|min| number < min) {
            return Err(format!("{path}: {number} is below the minimum"));
        }
        if schema["maximum"].as_f64().is_some_and(|max| number > max) {
            return Err(format!("{path}: {number} is above the maximum"));
        }
    }
    if let (Some(text), Some(max)) = (value.as_str(), schema["maxLength"].as_u64())
        && text.chars().count() as u64 > max
    {
        return Err(format!("{path}: string longer than {max}"));
    }
    if let Value::Object(object) = value {
        for key in schema["required"].as_array().into_iter().flatten() {
            let key = key.as_str().unwrap();
            if !object.contains_key(key) {
                return Err(format!("{path}: missing required {key}"));
            }
        }
        if schema["maxProperties"]
            .as_u64()
            .is_some_and(|max| object.len() as u64 > max)
        {
            return Err(format!("{path}: too many properties"));
        }
        let properties = schema["properties"].as_object();
        for (key, item) in object {
            let item_path = format!("{path}.{key}");
            match properties.and_then(|properties| properties.get(key)) {
                Some(property) => check(schemas, property, item, &item_path)?,
                None => match &schema["additionalProperties"] {
                    Value::Bool(false) => return Err(format!("{path}: unexpected {key}")),
                    Value::Object(_) => {
                        check(schemas, &schema["additionalProperties"], item, &item_path)?;
                    }
                    _ => {}
                },
            }
        }
    }
    if let (Value::Array(items), Some(item_schema)) = (value, schema.get("items")) {
        for (index, item) in items.iter().enumerate() {
            check(schemas, item_schema, item, &format!("{path}[{index}]"))?;
        }
    }
    Ok(())
}

#[test]
fn routes_match_only_the_vector_store_prefix() {
    assert!(matches("/v1/vector_stores"));
    assert!(matches("/v1/vector_stores/vs_1/search"));
    assert!(!matches("/v1/vector_storesx"));
    assert!(!matches("/v1/files"));
}

#[test]
fn the_validator_rejects_what_the_contract_forbids() {
    let good = json!({"id":"vs_1","object":"vector_store.deleted","deleted":true});
    conforms("DeleteVectorStoreResponse", &good);
    let document: Value = serde_json::from_str(SCHEMAS).unwrap();
    let schema = json!({"$ref": "#/components/schemas/VectorStoreSearchResultsPage"});
    for bad in [
        json!({"object":"list","search_query":[],"data":[],"has_more":false,"next_page":null}),
        json!({"object":"vector_store.search_results.page","search_query":[],"data":[],
               "has_more":false}),
        json!({"object":"vector_store.search_results.page","search_query":[],"data":[
            {"file_id":"f","filename":"n","score":1.5,"attributes":{},"content":[]}],
               "has_more":false,"next_page":null}),
        json!({"object":"vector_store.search_results.page","search_query":[],"data":[],
               "has_more":false,"next_page":null,"extra":1}),
    ] {
        assert!(
            check(&document["schemas"], &schema, &bad, "$").is_err(),
            "{bad}"
        );
    }
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn the_api_answers_with_conforming_objects() {
    let scratch = Scratch::new();
    let store = scratch.store();
    let service = scratch.start();
    let cats = upload(
        &store,
        "cats.txt",
        "Cats sleep most of the day in warm places.",
    );
    let dogs = upload(&store, "dogs.txt", "Dogs need daily walks and play.");

    let created = ok(
        &service,
        Method::POST,
        "/v1/vector_stores",
        Some(json!({"name":"pets","file_ids":[cats],"metadata":{"k":"v"},
                    "expires_after":{"anchor":"last_active_at","days":7},
                    "chunking_strategy":{"type":"static",
                        "static":{"max_chunk_size_tokens":200,"chunk_overlap_tokens":100}}})),
        "VectorStoreObject",
    )
    .await;
    let id = created["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("vs_"));
    assert_eq!(created["name"], "pets");
    service.wait_idle();
    let store_uri = format!("/v1/vector_stores/{id}");
    let fetched = ok(&service, Method::GET, &store_uri, None, "VectorStoreObject").await;
    assert_eq!(fetched["status"], "completed");
    assert_eq!(fetched["file_counts"]["completed"], 1);
    assert_eq!(fetched["expires_after"]["days"], 7);

    let attached = ok(
        &service,
        Method::POST,
        &format!("{store_uri}/files"),
        Some(json!({"file_id": dogs, "attributes": {"kind": "dog", "legs": 4}})),
        "VectorStoreFileObject",
    )
    .await;
    assert_eq!(attached["status"], "in_progress");
    assert_eq!(
        attached["chunking_strategy"]["static"]["max_chunk_size_tokens"],
        800
    );
    service.wait_idle();
    let file_uri = format!("{store_uri}/files/{dogs}");
    let file = ok(
        &service,
        Method::GET,
        &file_uri,
        None,
        "VectorStoreFileObject",
    )
    .await;
    assert_eq!(file["status"], "completed");
    assert!(file["last_error"].is_null());

    let list = ok(
        &service,
        Method::GET,
        &format!("{store_uri}/files?limit=1&order=asc"),
        None,
        "ListVectorStoreFilesResponse",
    )
    .await;
    assert_eq!(list["data"][0]["id"], cats.as_str());
    assert_eq!(list["has_more"], true);
    let stores = ok(
        &service,
        Method::GET,
        "/v1/vector_stores?limit=5",
        None,
        "ListVectorStoresResponse",
    )
    .await;
    assert_eq!(stores["data"][0]["id"], id.as_str());

    let content = ok(
        &service,
        Method::GET,
        &format!("{file_uri}/content"),
        None,
        "VectorStoreFileContentResponse",
    )
    .await;
    assert_eq!(
        content["data"][0]["text"],
        "Dogs need daily walks and play."
    );

    let results = ok(
        &service,
        Method::POST,
        &format!("{store_uri}/search"),
        Some(json!({"query":"walk the dogs","max_num_results":5,
                    "filters":{"type":"eq","key":"kind","value":"dog"},
                    "ranking_options":{"ranker":"none","score_threshold":0}})),
        "VectorStoreSearchResultsPage",
    )
    .await;
    assert_eq!(results["data"].as_array().unwrap().len(), 1);
    assert_eq!(results["data"][0]["file_id"], dogs.as_str());
    assert_eq!(results["data"][0]["filename"], "dogs.txt");
    assert_eq!(results["search_query"], json!(["walk the dogs"]));

    let modified = ok(
        &service,
        Method::POST,
        &file_uri,
        Some(json!({"attributes": null})),
        "VectorStoreFileObject",
    )
    .await;
    assert_eq!(modified["attributes"], json!({}));
    let renamed = ok(
        &service,
        Method::POST,
        &store_uri,
        Some(json!({"name": null, "expires_after": null})),
        "VectorStoreObject",
    )
    .await;
    assert_eq!(renamed["name"], "");
    assert!(renamed["expires_at"].is_null());

    let birds = upload(&store, "birds.txt", "Birds sing at dawn.");
    let batch = ok(
        &service,
        Method::POST,
        &format!("{store_uri}/file_batches"),
        Some(json!({"files":[{"file_id": birds, "attributes": {"kind": "bird"}}]})),
        "VectorStoreFileBatchObject",
    )
    .await;
    let batch_uri = format!("{store_uri}/file_batches/{}", batch["id"].as_str().unwrap());
    service.wait_idle();
    let batch = ok(
        &service,
        Method::GET,
        &batch_uri,
        None,
        "VectorStoreFileBatchObject",
    )
    .await;
    assert_eq!(batch["status"], "completed");
    assert_eq!(batch["file_counts"]["completed"], 1);
    let batch_files = ok(
        &service,
        Method::GET,
        &format!("{batch_uri}/files?filter=completed"),
        None,
        "ListVectorStoreFilesResponse",
    )
    .await;
    assert_eq!(batch_files["data"][0]["id"], birds.as_str());
    refused(
        &service,
        Method::POST,
        &format!("{batch_uri}/cancel"),
        None,
        StatusCode::CONFLICT,
    )
    .await;

    let deleted = ok(
        &service,
        Method::DELETE,
        &file_uri,
        None,
        "DeleteVectorStoreFileResponse",
    )
    .await;
    assert_eq!(deleted["deleted"], true);
    let deleted = ok(
        &service,
        Method::DELETE,
        &store_uri,
        None,
        "DeleteVectorStoreResponse",
    )
    .await;
    assert_eq!(deleted["id"], id.as_str());
    refused(
        &service,
        Method::GET,
        &store_uri,
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    service.close();
}

#[tokio::test]
async fn bad_requests_are_refused_with_their_status() {
    let scratch = Scratch::new();
    let service = scratch.start();
    let (status, created) = send(&service, Method::POST, "/v1/vector_stores", None).await;
    assert_eq!(status, StatusCode::OK);
    let id = created["id"].as_str().unwrap();
    let search = format!("/v1/vector_stores/{id}/search");
    let bad = StatusCode::BAD_REQUEST;
    for body in [
        r#"{"query":"x","unknown":1}"#,
        r#"{"query":"x","query":"y"}"#,
        r#"{"query":"x","rewrite_query":true}"#,
        r#"{"query":"x","max_num_results":51}"#,
        r#"{"query":"x","ranking_options":{"ranker":"default-2024-11-15"}}"#,
        r#"{"query":"x","filters":{"type":"in","key":"k","value":"v"}}"#,
        r#"{"query":7}"#,
        "[]",
    ] {
        refused(&service, Method::POST, &search, Some(body), bad).await;
    }
    refused(&service, Method::POST, &search, None, bad).await;
    for uri in [
        "/v1/vector_stores?limit=0",
        "/v1/vector_stores?limit=x",
        "/v1/vector_stores?order=up",
        "/v1/vector_stores?filter=completed",
        "/v1/vector_stores?limit=1&limit=2",
        "/v1/vector_stores?purpose=x",
    ] {
        refused(&service, Method::GET, uri, None, bad).await;
    }
    let files = format!("/v1/vector_stores/{id}/files");
    refused(
        &service,
        Method::GET,
        &format!("{files}?filter=done"),
        None,
        bad,
    )
    .await;
    refused(
        &service,
        Method::POST,
        &files,
        Some(r#"{"file_id":"file-missing"}"#),
        StatusCode::NOT_FOUND,
    )
    .await;
    refused(
        &service,
        Method::POST,
        "/v1/vector_stores",
        Some(r#"{"expires_after":{"anchor":"created_at","days":1}}"#),
        bad,
    )
    .await;
    refused(
        &service,
        Method::POST,
        "/v1/vector_stores",
        Some(r#"{"chunking_strategy":{"type":"static"}}"#),
        bad,
    )
    .await;
    refused(
        &service,
        Method::GET,
        "/v1/vector_stores/vs_missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    refused(
        &service,
        Method::GET,
        "/v1/vector_stores?after=vs_missing",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    refused(
        &service,
        Method::PUT,
        "/v1/vector_stores",
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    refused(
        &service,
        Method::GET,
        &format!("/v1/vector_stores/{id}/"),
        None,
        StatusCode::NOT_FOUND,
    )
    .await;
    service.close();
}

#[tokio::test]
async fn a_restarted_server_resumes_and_keeps_its_index() {
    let scratch = Scratch::new();
    let store = scratch.store();
    let text = "persistent lexical search survives restarts";
    let file = upload(&store, "p.txt", text);
    let id = {
        let service = scratch.start();
        let (_, created) = send(
            &service,
            Method::POST,
            "/v1/vector_stores",
            Some(&json!({"file_ids": [file]}).to_string()),
        )
        .await;
        service.wait_idle();
        service.close();
        created["id"].as_str().unwrap().to_owned()
    };
    let service = scratch.start();
    let results = ok(
        &service,
        Method::POST,
        &format!("/v1/vector_stores/{id}/search"),
        Some(json!({"query": ["restart", "nothing"]})),
        "VectorStoreSearchResultsPage",
    )
    .await;
    assert_eq!(results["data"][0]["content"][0]["text"], text);
    service.close();
}
