//! The Administration API adapter: authentication on every route family,
//! routing, errors, and conformance of every answer to the pinned `OpenAI`
//! schemas in `fixtures/admin_schemas.json` (a subset of the contract's
//! `components.schemas`, descriptions and examples stripped).

use std::sync::Arc;

use axum::http::{Method, StatusCode};
use serde_json::{Map, Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{Event, EventStream, GenerationStats, Signal, Stats};
use local_services::Store;
use local_services::admin::usage::{UsageKind, UsageRecord};
use local_services::admin::{Actor, Admin};

use super::super::chunked::start_chunked;
use super::super::response::{Protocol, Reply};
use super::super::responses::prepare_responses_with;
use super::super::sse::start_stream;
use super::{Caller, Gate, Meter, dispatch};

const SCHEMAS: &str = include_str!("fixtures/admin_schemas.json");

pub(in crate::cli::serve) fn open() -> (tempfile::TempDir, Arc<Admin>) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("store")).unwrap();
    (dir, Arc::new(Admin::new(store)))
}

/// Check `value` against the pinned schema `name`.
fn conforms(name: &str, value: &Value) {
    let spec: Value = serde_json::from_str(SCHEMAS).unwrap();
    let mut root = Map::new();
    root.insert("$ref".into(), json!(format!("#/components/schemas/{name}")));
    root.insert("components".into(), json!({"schemas": spec["schemas"]}));
    let validator = jsonschema::options()
        .build(&Value::Object(root))
        .expect("the pinned schema compiles");
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|error| error.to_string())
        .collect();
    assert!(errors.is_empty(), "{name}: {errors:#?}\n{value:#}");
}

#[allow(clippy::needless_pass_by_value)]
fn call(admin: &Admin, method: Method, uri: &str, body: &Value) -> (StatusCode, Value) {
    let (path, query) = uri
        .split_once('?')
        .map_or((uri, None), |(path, query)| (path, Some(query)));
    let bytes = if body.is_null() {
        Vec::new()
    } else {
        body.to_string().into_bytes()
    };
    dispatch(
        admin,
        &method,
        &format!("/v1/organization{path}"),
        query,
        &bytes,
        &Actor::AdminKey("key_test".into()),
    )
}

fn expect(admin: &Admin, method: Method, uri: &str, body: &Value, schema: &str) -> Value {
    let (status, value) = call(admin, method, uri, body);
    assert_eq!(status, StatusCode::OK, "{uri}: {value:#}");
    conforms(schema, &value);
    value
}

fn refused(admin: &Admin, method: Method, uri: &str, body: &Value, status: StatusCode) -> Value {
    let (actual, value) = call(admin, method, uri, body);
    assert_eq!(actual, status, "{uri}: {value:#}");
    conforms("ErrorResponse", &value);
    value
}

/// Normal API route families, one representative path each.
const API_PATHS: &[&str] = &[
    "/health",
    "/v1/models",
    "/v1/responses",
    "/v1/responses/resp_1",
    "/v1/responses/input_tokens",
    "/v1/responses/compact",
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/conversations",
    "/v1/files",
    "/v1/uploads",
    "/v1/batches",
    "/v1/vector_stores",
    "/v1/decisions",
    "/v1/moderations",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/organizations",
];

/// Administration API route families.
const ADMIN_PATHS: &[&str] = &[
    "/v1/organization",
    "/v1/organization/admin_api_keys",
    "/v1/organization/projects",
    "/v1/organization/projects/proj_1/api_keys",
    "/v1/organization/projects/proj_1/service_accounts",
    "/v1/organization/projects/proj_1/rate_limits",
    "/v1/organization/users",
    "/v1/organization/invites",
    "/v1/organization/audit_logs",
    "/v1/organization/usage/completions",
    "/v1/organization/costs",
    "/v1/organization/certificates",
];

fn status(result: Result<Caller, super::Refusal>) -> StatusCode {
    result.map_or_else(|refusal| refusal.status, |_| StatusCode::OK)
}

#[test]
fn every_route_family_is_open_until_enabled_except_administration() {
    let (_dir, admin) = open();
    let gate = Gate::new(Some(Arc::clone(&admin)), None).unwrap();
    for path in API_PATHS {
        assert_eq!(gate.check(path, None), Ok(Caller::Anonymous), "{path}");
    }
    for path in ADMIN_PATHS {
        assert_eq!(
            status(gate.check(path, None)),
            StatusCode::FORBIDDEN,
            "{path}"
        );
    }
}

#[test]
fn once_enabled_every_route_family_needs_the_right_kind_of_key() {
    let (_dir, admin) = open();
    let gate = Gate::new(Some(Arc::clone(&admin)), None).unwrap();
    let admin_key = admin.create_admin_key("a", None, &Actor::Owner).unwrap();
    let project = admin.create_project("p", None, &Actor::Owner).unwrap();
    let account = admin
        .create_service_account(&project.id, "bot", false, None, &Actor::Owner)
        .unwrap();
    let project_key = account.api_key.unwrap();
    let admin_bearer = format!("Bearer {}", admin_key.value);
    let project_bearer = format!("Bearer {}", project_key.value);
    let forged = format!(
        "Bearer {}0",
        project_key
            .value
            .get(..project_key.value.len() - 1)
            .unwrap()
    );
    for path in API_PATHS {
        assert_eq!(
            status(gate.check(path, None)),
            StatusCode::UNAUTHORIZED,
            "{path}"
        );
        assert_eq!(
            status(gate.check(path, Some("Bearer sk-proj-wrong"))),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(gate.check(path, Some(&forged))),
            if forged == project_bearer {
                StatusCode::OK
            } else {
                StatusCode::UNAUTHORIZED
            }
        );
        assert_eq!(
            status(gate.check(path, Some(&admin_bearer))),
            StatusCode::FORBIDDEN,
            "{path}"
        );
        assert_eq!(
            gate.check(path, Some(&project_bearer)),
            Ok(Caller::Project {
                key_id: project_key.id.clone(),
                project_id: project.id.clone(),
            }),
            "{path}"
        );
    }
    for path in ADMIN_PATHS {
        assert_eq!(
            status(gate.check(path, None)),
            StatusCode::UNAUTHORIZED,
            "{path}"
        );
        assert_eq!(
            status(gate.check(path, Some(&project_bearer))),
            StatusCode::FORBIDDEN,
            "{path}"
        );
        assert_eq!(
            gate.check(path, Some(&admin_bearer)),
            Ok(Caller::Admin {
                key_id: admin_key.object.id.clone()
            }),
            "{path}"
        );
    }
    // Revocation takes effect on the next request.
    admin
        .delete_project_key(&project.id, &project_key.id, &Actor::Owner)
        .unwrap();
    assert_eq!(
        status(gate.check("/v1/responses", Some(&project_bearer))),
        StatusCode::UNAUTHORIZED
    );
}

#[test]
fn the_fixed_api_key_authorizes_the_api_but_not_administration() {
    let (_dir, admin) = open();
    let gate = Gate::new(Some(Arc::clone(&admin)), Some("fixed-secret")).unwrap();
    assert_eq!(
        gate.check("/v1/responses", Some("Bearer fixed-secret")),
        Ok(Caller::Static)
    );
    assert_eq!(
        status(gate.check("/v1/responses", Some("Bearer fixed-secreT"))),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        status(gate.check("/v1/responses", None)),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        status(gate.check("/v1/organization/projects", Some("Bearer fixed-secret"))),
        StatusCode::FORBIDDEN
    );
    // Its comparison is the constant-time HMAC verification.
    let source = include_str!("admin.rs");
    assert!(source.contains("ring::hmac::verify(key, token.as_bytes(), tag.as_ref())"));
    assert!(!source.contains("token == "));
}

#[test]
#[allow(clippy::too_many_lines)]
fn key_project_and_service_account_lifecycle_conforms_to_the_spec() {
    let (_dir, admin) = open();
    let created = expect(
        &admin,
        Method::POST,
        "/admin_api_keys",
        &json!({"name":"ops","expires_in_seconds":3600}),
        "AdminApiKeyCreateResponse",
    );
    let key_id = created["id"].as_str().unwrap().to_owned();
    assert!(created["value"].as_str().unwrap().starts_with("sk-admin-"));
    let listed = expect(
        &admin,
        Method::GET,
        "/admin_api_keys?limit=5&order=desc",
        &Value::Null,
        "ApiKeyList",
    );
    assert_eq!(listed["data"][0]["id"], key_id);
    assert!(listed["data"][0].get("value").is_none());
    let got = expect(
        &admin,
        Method::GET,
        &format!("/admin_api_keys/{key_id}"),
        &Value::Null,
        "AdminApiKey",
    );
    assert!(got.get("value").is_none());

    let project = expect(
        &admin,
        Method::POST,
        "/projects",
        &json!({"name":"Apps"}),
        "Project",
    );
    let project_id = project["id"].as_str().unwrap().to_owned();
    assert_eq!(project["status"], "active");
    expect(
        &admin,
        Method::GET,
        "/projects?limit=1",
        &Value::Null,
        "ProjectListResponse",
    );
    let modified = expect(
        &admin,
        Method::POST,
        &format!("/projects/{project_id}"),
        &json!({"name":"Apps 2","external_key_id":null}),
        "Project",
    );
    assert_eq!(modified["name"], "Apps 2");

    let account = expect(
        &admin,
        Method::POST,
        &format!("/projects/{project_id}/service_accounts"),
        &json!({"name":"bot"}),
        "ProjectServiceAccountCreateResponse",
    );
    conforms("ProjectServiceAccountApiKey", &account["api_key"]);
    let account_id = account["id"].as_str().unwrap().to_owned();
    let first_key = account["api_key"]["id"].as_str().unwrap().to_owned();
    expect(
        &admin,
        Method::GET,
        &format!("/projects/{project_id}/service_accounts"),
        &Value::Null,
        "ProjectServiceAccountListResponse",
    );
    let updated = expect(
        &admin,
        Method::POST,
        &format!("/projects/{project_id}/service_accounts/{account_id}"),
        &json!({"role":"owner"}),
        "ProjectServiceAccount",
    );
    assert_eq!(updated["role"], "owner");
    let second = expect(
        &admin,
        Method::POST,
        &format!("/projects/{project_id}/service_accounts/{account_id}/api_keys"),
        &json!({"name":"ci","expires_in_seconds":60}),
        "ServiceAccountApiKeyBody",
    );
    assert!(second["value"].as_str().unwrap().starts_with("sk-proj-"));
    let keys = expect(
        &admin,
        Method::GET,
        &format!("/projects/{project_id}/api_keys?limit=1"),
        &Value::Null,
        "ProjectApiKeyListResponse",
    );
    assert_eq!(keys["has_more"], true);
    let key = expect(
        &admin,
        Method::GET,
        &format!("/projects/{project_id}/api_keys/{first_key}"),
        &Value::Null,
        "ProjectApiKey",
    );
    assert_eq!(key["owner"]["type"], "service_account");
    assert!(
        key["redacted_value"]
            .as_str()
            .unwrap()
            .starts_with("sk-proj...")
    );
    expect(
        &admin,
        Method::DELETE,
        &format!("/projects/{project_id}/api_keys/{first_key}"),
        &Value::Null,
        "ProjectApiKeyDeleteResponse",
    );
    refused(
        &admin,
        Method::GET,
        &format!("/projects/{project_id}/api_keys/{first_key}"),
        &Value::Null,
        StatusCode::NOT_FOUND,
    );
    expect(
        &admin,
        Method::DELETE,
        &format!("/projects/{project_id}/service_accounts/{account_id}"),
        &Value::Null,
        "ProjectServiceAccountDeleteResponse",
    );
    let archived = expect(
        &admin,
        Method::POST,
        &format!("/projects/{project_id}/archive"),
        &Value::Null,
        "Project",
    );
    assert_eq!(archived["status"], "archived");
    refused(
        &admin,
        Method::POST,
        &format!("/projects/{project_id}/service_accounts"),
        &json!({"name":"late"}),
        StatusCode::CONFLICT,
    );
    let visible = expect(
        &admin,
        Method::GET,
        "/projects",
        &Value::Null,
        "ProjectListResponse",
    );
    assert_eq!(visible["data"].as_array().unwrap().len(), 0);
    let all = expect(
        &admin,
        Method::GET,
        "/projects?include_archived=true",
        &Value::Null,
        "ProjectListResponse",
    );
    assert_eq!(all["data"].as_array().unwrap().len(), 1);

    let logs = expect(
        &admin,
        Method::GET,
        "/audit_logs?limit=100",
        &Value::Null,
        "ListAuditLogsResponse",
    );
    let kinds: Vec<&str> = logs["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|log| log["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds.first(), Some(&"project.archived"));
    assert!(kinds.contains(&"service_account.deleted"));
    let filtered = expect(
        &admin,
        Method::GET,
        &format!("/audit_logs?event_types[]=project.created&project_ids[]={project_id}"),
        &Value::Null,
        "ListAuditLogsResponse",
    );
    assert_eq!(filtered["data"].as_array().unwrap().len(), 1);
    expect(
        &admin,
        Method::GET,
        "/audit_logs?effective_at[gte]=0&effective_at[lt]=1",
        &Value::Null,
        "ListAuditLogsResponse",
    );

    expect(
        &admin,
        Method::DELETE,
        &format!("/admin_api_keys/{key_id}"),
        &Value::Null,
        "AdminApiKeyDeleteResponse",
    );
    refused(
        &admin,
        Method::DELETE,
        &format!("/admin_api_keys/{key_id}"),
        &Value::Null,
        StatusCode::NOT_FOUND,
    );
}

#[test]
fn users_and_invites_are_the_single_local_owner() {
    let (_dir, admin) = open();
    let users = expect(
        &admin,
        Method::GET,
        "/users",
        &Value::Null,
        "UserListResponse",
    );
    let owner = users["data"][0]["id"].as_str().unwrap().to_owned();
    assert_eq!(users["data"][0]["role"], "owner");
    expect(
        &admin,
        Method::GET,
        &format!("/users/{owner}"),
        &Value::Null,
        "User",
    );
    refused(
        &admin,
        Method::GET,
        "/users/user_other",
        &Value::Null,
        StatusCode::NOT_FOUND,
    );
    let none = expect(
        &admin,
        Method::GET,
        "/users?emails=a@example.com",
        &Value::Null,
        "UserListResponse",
    );
    assert_eq!(none["data"].as_array().unwrap().len(), 0);
    refused(
        &admin,
        Method::DELETE,
        &format!("/users/{owner}"),
        &Value::Null,
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::POST,
        &format!("/users/{owner}"),
        &json!({"role":"reader"}),
        StatusCode::BAD_REQUEST,
    );
    expect(
        &admin,
        Method::GET,
        "/invites",
        &Value::Null,
        "InviteListResponse",
    );
    refused(
        &admin,
        Method::POST,
        "/invites",
        &json!({"email":"a@example.com","role":"reader"}),
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::GET,
        "/invites/invite-1",
        &Value::Null,
        StatusCode::NOT_FOUND,
    );
}

#[test]
fn unenforced_features_are_refused_with_reasons() {
    let (_dir, admin) = open();
    let project = admin.create_project("p", None, &Actor::Owner).unwrap();
    for (method, uri) in [
        (Method::GET, format!("/projects/{}/rate_limits", project.id)),
        (
            Method::POST,
            format!("/projects/{}/rate_limits/rl_1", project.id),
        ),
        (Method::GET, "/certificates".to_owned()),
        (Method::GET, format!("/projects/{}/users", project.id)),
        (Method::GET, "/roles".to_owned()),
        (Method::GET, "/groups".to_owned()),
        (Method::GET, "/spend_alerts".to_owned()),
        (Method::GET, "/data_retention".to_owned()),
        (Method::GET, "/external_storage".to_owned()),
        (
            Method::GET,
            format!("/projects/{}/model_permissions", project.id),
        ),
    ] {
        let value = refused(&admin, method, &uri, &Value::Null, StatusCode::BAD_REQUEST);
        assert_eq!(value["error"]["code"], "unsupported", "{uri}");
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not supported")
        );
    }
    let value = refused(
        &admin,
        Method::POST,
        "/projects",
        &json!({"name":"x","residency":"EU_STORAGE_PROCESSING"}),
        StatusCode::BAD_REQUEST,
    );
    assert_eq!(value["error"]["code"], "unsupported");
    let account = admin
        .create_service_account(&project.id, "bot", true, None, &Actor::Owner)
        .unwrap();
    refused(
        &admin,
        Method::POST,
        &format!(
            "/projects/{}/service_accounts/{}/api_keys",
            project.id, account.account.id
        ),
        &json!({"scopes":["api.model.read"]}),
        StatusCode::BAD_REQUEST,
    );
}

#[test]
fn requests_are_strict() {
    let (_dir, admin) = open();
    refused(
        &admin,
        Method::GET,
        "/projects?bogus=1",
        &Value::Null,
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::GET,
        "/projects?limit=1&limit=2",
        &Value::Null,
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::GET,
        "/projects?limit=1000",
        &Value::Null,
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::POST,
        "/projects",
        &json!({"name":"x","extra":1}),
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::POST,
        "/projects",
        &json!({}),
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::POST,
        "/admin_api_keys",
        &json!({"name":"x","expires_in_seconds":0}),
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::GET,
        "/projects/proj_missing",
        &Value::Null,
        StatusCode::NOT_FOUND,
    );
    refused(
        &admin,
        Method::GET,
        "/nothing",
        &Value::Null,
        StatusCode::NOT_FOUND,
    );
    refused(
        &admin,
        Method::GET,
        "/usage/everything?start_time=0",
        &Value::Null,
        StatusCode::NOT_FOUND,
    );
    refused(
        &admin,
        Method::GET,
        "/usage/completions",
        &Value::Null,
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::GET,
        "/usage/completions?start_time=0&bucket_width=1w",
        &Value::Null,
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::GET,
        "/usage/embeddings?start_time=0&batch=true",
        &Value::Null,
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::GET,
        "/usage/completions?start_time=10&end_time=5",
        &Value::Null,
        StatusCode::BAD_REQUEST,
    );
    refused(
        &admin,
        Method::GET,
        "/costs?start_time=0&bucket_width=1h",
        &Value::Null,
        StatusCode::BAD_REQUEST,
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn usage_families_aggregate_recorded_requests_and_costs_are_empty() {
    let (_dir, admin) = open();
    let day = 1_730_419_200;
    for (kind, project, input) in [
        (UsageKind::Completions, "proj_a", 10),
        (UsageKind::Completions, "proj_b", 20),
        (UsageKind::Completions, "proj_a", 5),
        (UsageKind::Embeddings, "proj_a", 3),
    ] {
        admin
            .record_usage(&UsageRecord {
                at: Some(day + 100),
                project_id: Some(project.into()),
                api_key_id: Some("key_x".into()),
                model: Some("bonsai".into()),
                input_tokens: input,
                output_tokens: 1,
                ..UsageRecord::new(kind)
            })
            .unwrap();
    }
    let end = day + 2 * 86_400;
    let page = expect(
        &admin,
        Method::GET,
        &format!(
            "/usage/completions?start_time={day}&end_time={end}&group_by[]=project_id&group_by[]=model&project_ids=proj_a&project_ids=proj_b"
        ),
        &Value::Null,
        "UsageResponse",
    );
    let results = page["data"][0]["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["project_id"], "proj_a");
    assert_eq!(results[0]["input_tokens"], 15);
    assert_eq!(results[0]["num_model_requests"], 2);
    assert_eq!(results[0]["model"], "bonsai");
    assert_eq!(results[0]["api_key_id"], Value::Null);
    assert_eq!(page["data"][1]["results"].as_array().unwrap().len(), 0);
    for family in [
        "embeddings",
        "moderations",
        "audio_speeches",
        "audio_transcriptions",
        "images",
        "code_interpreter_sessions",
        "file_search_calls",
        "web_search_calls",
        "vector_stores",
    ] {
        let page = expect(
            &admin,
            Method::GET,
            &format!("/usage/{family}?start_time={day}&end_time={end}&bucket_width=1h&limit=3"),
            &Value::Null,
            "UsageResponse",
        );
        assert_eq!(page["data"].as_array().unwrap().len(), 3, "{family}");
        assert_eq!(page["has_more"], true);
        let next = page["next_page"].as_str().unwrap().to_owned();
        expect(
            &admin,
            Method::GET,
            &format!(
                "/usage/{family}?start_time={day}&end_time={end}&bucket_width=1h&limit=3&page={next}"
            ),
            &Value::Null,
            "UsageResponse",
        );
    }
    let embeddings = expect(
        &admin,
        Method::GET,
        &format!("/usage/embeddings?start_time={day}&end_time={end}"),
        &Value::Null,
        "UsageResponse",
    );
    assert_eq!(embeddings["data"][0]["results"][0]["input_tokens"], 3);
    let costs = expect(
        &admin,
        Method::GET,
        &format!("/costs?start_time={day}&end_time={end}&group_by=line_item"),
        &Value::Null,
        "UsageResponse",
    );
    assert_eq!(costs["data"].as_array().unwrap().len(), 2);
    assert!(
        costs["data"]
            .as_array()
            .unwrap()
            .iter()
            .all(|bucket| bucket["results"].as_array().is_some_and(Vec::is_empty))
    );
    // The current vector store level lands in the bucket holding now.
    let now = i64::try_from(super::super::response::unix_now()).unwrap();
    let stores = expect(
        &admin,
        Method::GET,
        &format!("/usage/vector_stores?start_time={}", now - 60),
        &Value::Null,
        "UsageResponse",
    );
    assert_eq!(stores["data"][0]["results"][0]["usage_bytes"], 0);
}

fn finished(prompt: usize, reused: usize, generated: usize) -> Event {
    Event::Finished(Box::new(Stats {
        stop_reason: StopReason::Eos,
        cache_source: PromptCacheSource::None,
        reasoning_tokens: 0,
        generation: GenerationStats {
            prompt_tokens: prompt,
            reused_prompt_tokens: reused,
            generated_tokens: generated,
            ..GenerationStats::default()
        },
    }))
}

fn project() -> Caller {
    Caller::Project {
        key_id: "key_p".into(),
        project_id: "proj_p".into(),
    }
}

/// A `completions` meter of the test project's into `admin`.
pub(in crate::cli::serve) fn project_meter(admin: &Arc<Admin>) -> Meter {
    Meter::into_store(
        Some(admin),
        project().usage(UsageKind::Completions, "bonsai"),
    )
}

/// How many `completions` requests `admin` recorded in the last minute.
pub(in crate::cli::serve) fn completions(admin: &Admin) -> u64 {
    recent(admin, "completions")
        .iter()
        .map(|result| result["num_model_requests"].as_u64().unwrap())
        .sum()
}

/// Wait until `admin` holds `requests` `completions` requests. A dropped
/// stream records from a task of its own that tells no one, so this polls,
/// briefly and within a deadline.
pub(in crate::cli::serve) async fn recorded(admin: &Admin, requests: u64) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    while completions(admin) < requests {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{} of {requests} requests recorded",
            completions(admin)
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    // Nothing more arrives late.
    tokio::task::yield_now().await;
    tokio::task::spawn_blocking(|| ()).await.unwrap();
    assert_eq!(completions(admin), requests);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_generation_is_recorded_once_by_the_read_that_returns_its_end() {
    let (_dir, admin) = open();
    let (source, events) = EventStream::relay();
    let mut stream = project_meter(&admin).stream(events);
    let producer = tokio::task::spawn_blocking(move || {
        assert!(source.blocking_send(Signal::Event(Event::Content("hi".into()))));
        assert!(source.blocking_send(Signal::Event(finished(11, 4, 2))));
    });
    let reader = Arc::clone(&admin);
    let received = tokio::task::spawn_blocking(move || {
        let mut seen = Vec::new();
        while let Ok(Some(signal)) = stream.next_signal(None) {
            // Committed before the consumer holds `Finished`, never after.
            let finished = matches!(signal, Signal::Event(Event::Finished(_)));
            assert_eq!(completions(&reader), u64::from(finished));
            seen.push(signal);
        }
        seen
    })
    .await
    .unwrap();
    producer.await.unwrap();
    assert_eq!(received.len(), 2);
    assert!(matches!(&received[0], Signal::Event(Event::Content(text)) if text == "hi"));
    assert!(matches!(&received[1], Signal::Event(Event::Finished(_))));
    recorded(&admin, 1).await;
    let now = super::super::response::unix_now();
    let page = expect(
        &admin,
        Method::GET,
        &format!(
            "/usage/completions?start_time={}&bucket_width=1h&group_by=project_id&group_by=api_key_id",
            now - 60
        ),
        &Value::Null,
        "UsageResponse",
    );
    let results: Vec<Value> = page["data"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|bucket| bucket["results"].as_array().unwrap().clone())
        .collect();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["project_id"], "proj_p");
    assert_eq!(results[0]["api_key_id"], "key_p");
    assert_eq!(results[0]["input_tokens"], 11);
    assert_eq!(results[0]["input_cached_tokens"], 4);
    assert_eq!(results[0]["output_tokens"], 2);
    assert_eq!(results[0]["num_model_requests"], 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_abandoned_generation_is_cancelled_and_still_recorded_once() {
    let (_dir, admin) = open();
    let (source, events) = EventStream::relay();
    let mut stream = project_meter(&admin).stream(events);
    assert!(source.send(Signal::Event(Event::Content("h".into()))).await);
    // The client reads one token and goes away.
    let stream = tokio::task::spawn_blocking(move || {
        assert!(matches!(
            stream.next_signal(None),
            Ok(Some(Signal::Event(Event::Content(_))))
        ));
        stream
    })
    .await
    .unwrap();
    drop(stream);
    tokio::time::timeout(std::time::Duration::from_secs(5), source.cancelled())
        .await
        .expect("the generation is cancelled");
    assert_eq!(completions(&admin), 0);
    // The engine reports what the cancelled generation produced.
    let Event::Finished(mut stats) = finished(9, 0, 3) else {
        unreachable!()
    };
    stats.stop_reason = StopReason::Cancelled;
    assert!(source.send(Signal::Event(Event::Content("i".into()))).await);
    assert!(source.send(Signal::Event(Event::Finished(stats))).await);
    drop(source);
    recorded(&admin, 1).await;
    assert_eq!(recent(&admin, "completions")[0]["output_tokens"], 3);
}

#[test]
fn metered_generations_hold_no_blocking_thread_until_they_end() {
    const GENERATIONS: usize = 8;
    let (_dir, admin) = open();
    // One blocking thread, as if every other were taken: a queued
    // generation, read or abandoned, must not need it.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut sources = Vec::new();
        let mut held = Vec::new();
        for index in 0..GENERATIONS {
            let (source, events) = EventStream::relay();
            let stream = project_meter(&admin).stream(events);
            // Half are abandoned while queued, half wait for their reader.
            if index % 2 == 0 {
                drop(stream);
            } else {
                held.push(stream);
            }
            sources.push((index % 2 == 0, source));
        }
        let answered = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            tokio::task::spawn_blocking(|| 7),
        )
        .await;
        assert_eq!(answered.expect("the blocking pool is free").unwrap(), 7);
        for (abandoned, source) in sources {
            if abandoned {
                tokio::time::timeout(std::time::Duration::from_secs(5), source.cancelled())
                    .await
                    .expect("an abandoned generation is cancelled");
            }
            assert!(source.send(Signal::Event(finished(5, 0, 1))).await);
        }
        tokio::task::spawn_blocking(move || {
            for mut stream in held {
                while let Ok(Some(_)) = stream.next_signal(None) {}
            }
        })
        .await
        .unwrap();
        recorded(&admin, GENERATIONS as u64).await;
    });
}

fn chat_reply(include_usage: bool) -> Reply {
    Reply {
        protocol: Protocol::Chat {
            include_usage,
            logprobs: false,
        },
        id: "chatcmpl-1".into(),
        created: 5,
        model: "m".into(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn usage_is_recorded_before_the_response_can_end() {
    const STALL: std::time::Duration = std::time::Duration::from_secs(30);
    for chunked in [false, true] {
        let (_dir, admin) = open();
        let (source, events) = EventStream::relay();
        let events = project_meter(&admin).stream(events);
        assert!(
            source
                .send(Signal::Event(Event::Content("hi".into())))
                .await
        );
        let foreground = Arc::new(super::super::Foreground::default());
        let admitted = super::super::QueueDepth::default().admit();
        let mut frames = if chunked {
            let reply = chat_reply(false);
            let started = start_chunked(events, reply, admitted, &foreground, STALL, false);
            started.await.unwrap().0
        } else {
            // Usage adds a frame after the finish chunk, so the pump is still
            // running when the finish chunk is read: a record made when it
            // returns cannot be there yet.
            let reply = chat_reply(true);
            start_stream(events, reply, admitted, &foreground, STALL)
                .await
                .unwrap()
        };
        assert!(source.send(Signal::Event(finished(11, 4, 2))).await);
        // The source stays open, so the chunked pump is still waiting for
        // the end of the stream when its last byte arrives.
        let mut body = String::new();
        loop {
            let frame = tokio::time::timeout(std::time::Duration::from_secs(5), frames.recv())
                .await
                .expect("a frame in time")
                .expect("a frame")
                .unwrap();
            body.push_str(std::str::from_utf8(&frame).unwrap());
            let seen = completions(&admin);
            let ended = if chunked {
                serde_json::from_str::<Value>(&body).is_ok()
            } else {
                body.contains("\"finish_reason\":\"stop\"")
            };
            if ended {
                assert_eq!(seen, 1, "chunked={chunked}: recorded before the end");
                break;
            }
        }
        drop(source);
        while let Some(frame) =
            tokio::time::timeout(std::time::Duration::from_secs(5), frames.recv())
                .await
                .expect("the end in time")
        {
            body.push_str(std::str::from_utf8(&frame.unwrap()).unwrap());
        }
        assert!(chunked || body.ends_with("data: [DONE]\n\n"), "{body}");
        recorded(&admin, 1).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_background_job_is_recorded_once_before_it_settles() {
    use super::super::background::{self, Background};
    use super::super::background_tests::{Scratch, status};
    let (_dir, admin) = open();
    let scratch = Scratch::new();
    let store = scratch.open();
    let background = Arc::new(Background::default());
    let body = json!({"input":"hi","background":true}).to_string();
    let prepared = prepare_responses_with(body.as_bytes(), true, Some(&store), None, "m").unwrap();
    let reply = Reply::new(Protocol::Responses(Arc::new(prepared.echo)), "m".into());
    let (source, events) = EventStream::relay();
    let queued = background::start(
        Arc::clone(&background),
        project_meter(&admin).stream(events),
        reply,
        super::super::QueueDepth::default().admit(),
        false,
    )
    .await
    .unwrap();
    let id = queued["id"].as_str().unwrap().to_owned();
    let mut published = background.watch(&id).expect("running here");
    assert!(
        source
            .send(Signal::Event(Event::Content("hi".into())))
            .await
    );
    assert!(source.send(Signal::Event(finished(11, 4, 2))).await);
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        published.wait_for(|published| published.settled),
    )
    .await
    .expect("settled in time")
    .unwrap();
    // The terminal state is observable, and the pump still waits for the end
    // of the stream: the usage is already there.
    assert_eq!(completions(&admin), 1);
    assert_eq!(status(&store, &id), "completed");
    drop(source);
    background.drained().await;
    recorded(&admin, 1).await;
}

/// Every result of `family` recorded in the last minute, one bucket.
pub(in crate::cli::serve) fn recent(admin: &Admin, family: &str) -> Vec<Value> {
    let now = super::super::response::unix_now();
    let page = expect(
        admin,
        Method::GET,
        &format!(
            "/usage/{family}?start_time={}&bucket_width=1d&group_by=project_id&group_by=model",
            now - 60
        ),
        &Value::Null,
        "UsageResponse",
    );
    page["data"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|bucket| bucket["results"].as_array().unwrap().clone())
        .collect()
}

#[test]
fn each_route_records_its_own_usage_family() {
    let (_dir, admin) = open();
    let caller = Caller::Project {
        key_id: "key_p".into(),
        project_id: "proj_p".into(),
    };
    let meter = |record| super::Meter::into_store(Some(&admin), record).write();
    meter(super::token_usage(
        &caller,
        UsageKind::Embeddings,
        "embeddinggemma-2",
        7,
    ));
    meter(super::token_usage(
        &caller,
        UsageKind::Moderations,
        "embeddinggemma-2-moderation-1",
        12,
    ));
    // Characters, not bytes.
    meter(super::speech_usage(&caller, "kokoro-82m", "Héllo wörld."));
    // A transcription and a translation, a started second counting as one.
    meter(super::transcription_usage(&caller, "whisper-large-v3", 2.2));
    meter(super::transcription_usage(&caller, "whisper-large-v3", 0.4));

    let embeddings = recent(&admin, "embeddings");
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0]["input_tokens"], 7);
    assert_eq!(embeddings[0]["model"], "embeddinggemma-2");
    assert_eq!(embeddings[0]["project_id"], "proj_p");
    let moderations = recent(&admin, "moderations");
    assert_eq!(moderations[0]["input_tokens"], 12);
    assert_eq!(moderations[0]["num_model_requests"], 1);
    assert_eq!(moderations[0]["model"], "embeddinggemma-2-moderation-1");
    let speeches = recent(&admin, "audio_speeches");
    assert_eq!(speeches[0]["characters"], 12);
    assert_eq!(speeches[0]["model"], "kokoro-82m");
    let transcriptions = recent(&admin, "audio_transcriptions");
    assert_eq!(transcriptions[0]["seconds"], 4);
    assert_eq!(transcriptions[0]["num_model_requests"], 2);
    assert_eq!(recent(&admin, "completions"), Vec::<Value>::new());
}

#[test]
fn out_of_band_generations_are_recorded_as_completions() {
    let (_dir, admin) = open();
    let caller = Caller::Project {
        key_id: "key_p".into(),
        project_id: "proj_p".into(),
    };
    // A compaction or reasoning summary of a project's request.
    let mut summary = caller.usage(UsageKind::Completions, "bonsai");
    summary.model = Some("bonsai".into());
    let meter = super::Meter::into_store(Some(&admin), summary);
    let Event::Finished(stats) = finished(30, 10, 5) else {
        unreachable!()
    };
    meter.generation(&stats);
    // A batch line of an unattributed batch.
    let mut line = UsageRecord::new(UsageKind::Completions);
    line.model = Some("bonsai".into());
    line.batch = true;
    let submit: super::super::batches::Submit = Arc::new(|_request| {
        let mut signals = vec![
            Signal::Event(Event::Content("ok".into())),
            Signal::Event(finished(8, 0, 2)),
        ]
        .into_iter();
        Ok(super::super::batches::Generation {
            next: Box::new(move |_| Ok(signals.next())),
            cancel: Arc::new(|| {}),
        })
    });
    let request = local_engine::compaction::reasoning_summary_request(
        "thought",
        local_engine::compaction::SummaryDetail::Auto,
    );
    let mut generation = super::super::batches::metered(
        submit(request).unwrap(),
        super::Meter::into_store(Some(&admin), line),
    );
    while let Ok(Some(_)) = (generation.next)(None) {}
    // Without a store nothing is written and nothing fails.
    super::Meter::default().generation(&stats);

    let now = super::super::response::unix_now();
    let page = expect(
        &admin,
        Method::GET,
        &format!(
            "/usage/completions?start_time={}&bucket_width=1d&group_by=batch&group_by=project_id",
            now - 60
        ),
        &Value::Null,
        "UsageResponse",
    );
    let mut results: Vec<Value> = page["data"][0]["results"].as_array().unwrap().clone();
    results.sort_by_key(|result| result["batch"].as_bool());
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["batch"], false);
    assert_eq!(results[0]["project_id"], "proj_p");
    assert_eq!(results[0]["input_tokens"], 30);
    assert_eq!(results[0]["input_cached_tokens"], 10);
    assert_eq!(results[0]["output_tokens"], 5);
    assert_eq!(results[1]["batch"], true);
    assert_eq!(results[1]["project_id"], Value::Null);
    assert_eq!(results[1]["input_tokens"], 8);
    assert_eq!(results[1]["output_tokens"], 2);
}
