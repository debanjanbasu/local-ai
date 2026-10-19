//! Authentication and the `OpenAI` Administration API
//! (`/v1/organization/*`) over [`local_services::admin`].
//!
//! # Authentication
//!
//! Off by default, as before: on loopback every local process may use the
//! server (web pages a local browser opens may not; see `super::browser`). It turns
//! on when the owner creates the first admin key with `local-ai admin
//! create-key`, which prints the key once; from then on every request needs
//! `Authorization: Bearer <key>` (see [`Gate::check`]):
//!
//! * a project API key (`sk-proj-...`) authorizes every API except the
//!   Administration API, and attributes usage to its project and key;
//! * an admin API key (`sk-admin-...`) authorizes only `/v1/organization/*`;
//! * the `--api-key` flag's key, if given, authorizes every API except the
//!   Administration API, unattributed. It is compared in constant time.
//!
//! The Administration API is refused while authentication is off: without a
//! key anyone local could mint keys.
//!
//! # What is implemented and what is refused
//!
//! Implemented: admin API keys; projects (create, list, retrieve, modify,
//! archive); project service accounts and their API keys (the only way the
//! API creates project keys); project API keys (list, retrieve, delete); the
//! one local owner as the organization's only user; invites as an empty
//! list; audit logs of key, project and service account lifecycle; usage of
//! every family, from records this server writes; costs as buckets with no
//! line items, because nothing is billed.
//!
//! Refused with `400 unsupported` and the reason: modifying or deleting the
//! owner, sending invites, project users and groups, roles, rate limits
//! (none are enforced), certificates (mutual TLS is not supported), spend
//! alerts and limits (nothing is billed), data retention, model and hosted
//! tool permissions, external storage, and key scopes.

use std::sync::Arc;

use axum::body::Body;
use axum::http::request::Parts;
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::Response;
use http_body_util::{BodyExt as _, Limited};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use local_engine::{Event, EventStream, Signal, Stats};
use local_services::Error;
use local_services::admin::usage::{
    DAY, UsageKind, UsageQuery, UsageRecord, bucket_spans, parse_width,
};
use local_services::admin::{
    Actor, Admin, AuditQuery, Credential, Owner, Page, ProjectChanges, Verdict,
};

use super::AppState;
use super::response::json_response;
use super::store::query_pairs;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#[path = "admin_tests.rs"]
mod tests;

/// Route prefix of the Administration API.
pub(super) const PREFIX: &str = "/v1/organization";
/// Largest Administration API request body.
const MAX_BODY_BYTES: usize = 64 * 1024;
/// Default and largest `limit` of `GET /organization/costs`.
const COST_LIMITS: (usize, usize) = (7, 180);

/// Whether `path` belongs to the Administration API.
pub(super) fn matches(path: &str) -> bool {
    path.strip_prefix(PREFIX)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// Who an authorized request acts as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Caller {
    /// Authentication is off.
    Anonymous,
    /// The `--api-key` flag's key.
    Static,
    /// A project API key.
    Project { key_id: String, project_id: String },
    /// An admin API key.
    Admin { key_id: String },
}

impl Caller {
    /// A usage record of `kind` attributed to this caller.
    pub(super) fn usage(&self, kind: UsageKind, model: &str) -> UsageRecord {
        let mut record = UsageRecord::new(kind);
        record.model = Some(model.to_owned());
        if let Self::Project { key_id, project_id } = self {
            record.project_id = Some(project_id.clone());
            record.api_key_id = Some(key_id.clone());
        }
        record
    }

    /// The project and key that work stored for this caller, and run later
    /// (resumed background Responses), is attributed to: `None` unless
    /// this is a project API key.
    pub(super) fn creator(&self) -> Option<Value> {
        match self {
            Self::Project { key_id, project_id } => {
                Some(json!({"project_id":project_id,"api_key_id":key_id}))
            }
            Self::Anonymous | Self::Static | Self::Admin { .. } => None,
        }
    }

    /// The caller [`Self::creator`] stored: the project key it names, or
    /// [`Self::Anonymous`] (unattributed) for none or a malformed one.
    pub(super) fn from_creator(creator: Option<&Value>) -> Self {
        let field = |name: &str| {
            creator
                .and_then(|creator| creator.get(name))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        };
        match (field("project_id"), field("api_key_id")) {
            (Some(project_id), Some(key_id)) => Self::Project { key_id, project_id },
            _ => Self::Anonymous,
        }
    }
}

/// Why a request was not authorized.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Refusal {
    pub(super) status: StatusCode,
    pub(super) message: String,
}

impl Refusal {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

/// The authentication policy: the optional `--api-key` key and the admin
/// store's keys.
pub(super) struct Gate {
    admin: Option<Arc<Admin>>,
    /// The `--api-key` key as an HMAC tag under a random per-process key, so
    /// it is compared with the same constant-time verification as stored keys.
    fixed: Option<(ring::hmac::Key, ring::hmac::Tag)>,
}

impl std::fmt::Debug for Gate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Gate")
            .field("admin", &self.admin.is_some())
            .field("fixed", &self.fixed.is_some())
            .finish()
    }
}

impl Gate {
    pub(super) fn new(admin: Option<Arc<Admin>>, api_key: Option<&str>) -> crate::Result<Self> {
        let fixed = api_key
            .map(|key| {
                let rng = ring::rand::SystemRandom::new();
                let process_key = ring::hmac::Key::generate(ring::hmac::HMAC_SHA256, &rng)
                    .map_err(|_| {
                        crate::Error::InvalidArgument(
                            "operating system randomness is unavailable".into(),
                        )
                    })?;
                let tag = ring::hmac::sign(&process_key, key.as_bytes());
                Ok::<_, crate::Error>((process_key, tag))
            })
            .transpose()?;
        Ok(Self { admin, fixed })
    }

    /// The admin store, when the server has one.
    pub(super) const fn admin(&self) -> Option<&Arc<Admin>> {
        self.admin.as_ref()
    }

    /// Authorize a request for `path` carrying `authorization`. Blocks on
    /// the store; call off the runtime.
    pub(super) fn check(&self, path: &str, authorization: Option<&str>) -> Result<Caller, Refusal> {
        let administration = matches(path);
        let token = authorization.and_then(|value| {
            value
                .strip_prefix("Bearer ")
                .or_else(|| value.strip_prefix("bearer "))
                .map(str::trim)
        });
        if let (Some((key, tag)), Some(token)) = (&self.fixed, token)
            && ring::hmac::verify(key, token.as_bytes(), tag.as_ref()).is_ok()
        {
            return if administration {
                Err(Refusal::new(
                    StatusCode::FORBIDDEN,
                    "the --api-key key does not authorize the Administration API; use an admin \
                     API key",
                ))
            } else {
                Ok(Caller::Static)
            };
        }
        let verdict = match &self.admin {
            Some(admin) => admin.authenticate(token).map_err(|error| {
                Refusal::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("the authentication store failed: {error}"),
                )
            })?,
            None => Verdict::Open,
        };
        match verdict {
            Verdict::Open if self.fixed.is_some() => {
                Err(Refusal::new(StatusCode::UNAUTHORIZED, "invalid API key"))
            }
            Verdict::Open if administration => Err(Refusal::new(
                StatusCode::FORBIDDEN,
                "the Administration API requires authentication, which is off: create the first \
                 admin key with `local-ai admin create-key`",
            )),
            Verdict::Open => Ok(Caller::Anonymous),
            Verdict::Refused(message) => Err(Refusal::new(StatusCode::UNAUTHORIZED, message)),
            Verdict::Granted(Credential::Admin { key_id }) if administration => {
                Ok(Caller::Admin { key_id })
            }
            Verdict::Granted(Credential::Admin { .. }) => Err(Refusal::new(
                StatusCode::FORBIDDEN,
                "admin API keys authorize only the Administration API (/v1/organization); use a \
                 project API key",
            )),
            Verdict::Granted(Credential::Project { .. }) if administration => Err(Refusal::new(
                StatusCode::FORBIDDEN,
                "project API keys do not authorize the Administration API; use an admin API key",
            )),
            Verdict::Granted(Credential::Project {
                key_id, project_id, ..
            }) => Ok(Caller::Project { key_id, project_id }),
        }
    }
}

impl Gate {
    /// Whether requests need a key (a failing store counts as yes). Blocks
    /// on the store; call off the runtime.
    fn enabled(&self) -> bool {
        self.fixed.is_some()
            || self
                .admin
                .as_ref()
                .is_some_and(|admin| admin.auth_enabled().unwrap_or(true))
    }
}

/// [`Gate::enabled`] from the runtime.
pub(super) async fn enabled(state: &AppState) -> bool {
    let gate = Arc::clone(&state.auth);
    tokio::task::spawn_blocking(move || gate.enabled())
        .await
        .unwrap_or(true)
}

/// Say at startup whether requests need a key, warning when anyone who can
/// reach a non-loopback address may use the server.
pub(super) fn announce(gate: &Gate, host: std::net::IpAddr) {
    if gate.enabled() {
        eprintln!("authentication: on (Authorization: Bearer <key> required)");
    } else if host.is_loopback() {
        eprintln!(
            "authentication: off; any local process may use this server (enable it with \
             `local-ai admin create-key`)"
        );
    } else {
        eprintln!(
            "warning: authentication is off on non-loopback address {host}; anyone who can \
             reach it may use this server (enable it with `local-ai admin create-key`)"
        );
    }
}

/// Authorize the request in `parts`, or the response refusing it.
pub(super) async fn authorize(state: &AppState, parts: &Parts) -> Result<Caller, Box<Response>> {
    let authorization = parts
        .headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let (gate, path) = (Arc::clone(&state.auth), parts.uri.path().to_owned());
    let checked = tokio::task::spawn_blocking(move || gate.check(&path, authorization.as_deref()))
        .await
        .unwrap_or_else(|error| {
            Err(Refusal::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("authentication failed: {error}"),
            ))
        });
    match checked {
        Ok(caller) => Ok(caller),
        Err(refusal) => {
            let code = match refusal.status {
                StatusCode::UNAUTHORIZED => json!("invalid_api_key"),
                StatusCode::FORBIDDEN => json!("insufficient_permissions"),
                _ => Value::Null,
            };
            let mut response = reply(
                state,
                &parts.headers,
                refusal.status,
                error_body(refusal.status, &refusal.message, &code),
            )
            .await;
            if refusal.status == StatusCode::UNAUTHORIZED {
                response.headers_mut().insert(
                    header::WWW_AUTHENTICATE,
                    header::HeaderValue::from_static("Bearer"),
                );
            }
            Err(Box::new(response))
        }
    }
}

async fn reply(state: &AppState, headers: &HeaderMap, status: StatusCode, body: Value) -> Response {
    json_response(status, body, state, headers).await
}

fn error_body(status: StatusCode, message: &str, code: &Value) -> Value {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    json!({"error":{"message":message,"type":kind,"code":code,"param":null}})
}

/// Answer an Administration API request by `caller` (an admin key).
pub(super) async fn handle(
    state: &AppState,
    parts: &Parts,
    body: Body,
    caller: &Caller,
) -> Response {
    let (status, value) = match (state.auth.admin(), caller) {
        (Some(admin), Caller::Admin { key_id }) => {
            match Limited::new(body, MAX_BODY_BYTES).collect().await {
                Ok(collected) => {
                    let (admin, actor) = (Arc::clone(admin), Actor::AdminKey(key_id.clone()));
                    let (method, path) = (parts.method.clone(), parts.uri.path().to_owned());
                    let query = parts.uri.query().map(str::to_owned);
                    let bytes = collected.to_bytes();
                    tokio::task::spawn_blocking(move || {
                        dispatch(&admin, &method, &path, query.as_deref(), &bytes, &actor)
                    })
                    .await
                    .unwrap_or_else(|error| {
                        failure(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string())
                    })
                }
                Err(error) => failure(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    &format!("invalid HTTP body (at most {MAX_BODY_BYTES} bytes): {error}"),
                ),
            }
        }
        _ => failure(
            StatusCode::FORBIDDEN,
            "the Administration API requires an admin API key",
        ),
    };
    reply(state, &parts.headers, status, value).await
}

type Answer = (StatusCode, Value);

fn failure(status: StatusCode, message: &str) -> Answer {
    (status, error_body(status, message, &Value::Null))
}

fn unsupported(reason: &str) -> Answer {
    (
        StatusCode::BAD_REQUEST,
        error_body(
            StatusCode::BAD_REQUEST,
            &format!("not supported by this local server: {reason}"),
            &json!("unsupported"),
        ),
    )
}

fn store_failure(error: &Error) -> Answer {
    let status = match error {
        Error::InvalidArgument(_) => StatusCode::BAD_REQUEST,
        Error::NotFound(_) => StatusCode::NOT_FOUND,
        Error::Conflict(_) => StatusCode::CONFLICT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    failure(status, &error.to_string())
}

fn ok<T: serde::Serialize>(result: local_services::Result<T>) -> Answer {
    match result.and_then(|value| serde_json::to_value(value).map_err(Error::from)) {
        Ok(value) => (StatusCode::OK, value),
        Err(error) => store_failure(&error),
    }
}

fn deleted(result: local_services::Result<()>, id: &str, object: &str) -> Answer {
    ok(result.map(|()| json!({"id":id,"object":object,"deleted":true})))
}

/// The parsed query string: only `allowed` names, scalars at most once,
/// arrays as `name=a&name=b` or `name[]=a`.
struct Query(Vec<(String, String)>);

impl Query {
    fn parse(query: Option<&str>, allowed: &[&str]) -> Result<Self, Answer> {
        let pairs: Vec<(String, String)> = query_pairs(query)
            .into_iter()
            .map(|(name, value)| {
                let name = name
                    .strip_suffix("[]")
                    .map_or_else(|| name.clone(), str::to_owned);
                (name, value)
            })
            .collect();
        for (name, _) in &pairs {
            let base = name.split('[').next().unwrap_or(name);
            if !allowed.contains(&base) {
                return Err(failure(
                    StatusCode::BAD_REQUEST,
                    &format!("unknown query parameter {name:?}"),
                ));
            }
        }
        Ok(Self(pairs))
    }

    fn one(&self, name: &str) -> Result<Option<&str>, Answer> {
        let mut values = self.0.iter().filter(|(key, _)| key == name);
        let first = values.next().map(|(_, value)| value.as_str());
        if values.next().is_some() {
            return Err(failure(
                StatusCode::BAD_REQUEST,
                &format!("query parameter {name:?} may appear only once"),
            ));
        }
        Ok(first)
    }

    fn all(&self, name: &str) -> Vec<String> {
        self.0
            .iter()
            .filter(|(key, _)| key == name)
            .flat_map(|(_, value)| value.split(',').map(str::to_owned).collect::<Vec<_>>())
            .filter(|value| !value.is_empty())
            .collect()
    }

    fn number<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, Answer> {
        self.one(name)?
            .map(|value| {
                value.parse().map_err(|_| {
                    failure(
                        StatusCode::BAD_REQUEST,
                        &format!("{name} must be an integer, not {value:?}"),
                    )
                })
            })
            .transpose()
    }

    fn flag(&self, name: &str) -> Result<Option<bool>, Answer> {
        match self.one(name)? {
            None => Ok(None),
            Some("true") => Ok(Some(true)),
            Some("false") => Ok(Some(false)),
            Some(other) => Err(failure(
                StatusCode::BAD_REQUEST,
                &format!("{name} must be true or false, not {other:?}"),
            )),
        }
    }

    fn page(&self) -> Result<Page, Answer> {
        let descending = match self.one("order")? {
            None | Some("asc") => false,
            Some("desc") => true,
            Some(other) => {
                return Err(failure(
                    StatusCode::BAD_REQUEST,
                    &format!("order must be asc or desc, not {other:?}"),
                ));
            }
        };
        Ok(Page {
            limit: self.number("limit")?,
            after: self.one("after")?.map(str::to_owned),
            descending,
        })
    }
}

/// A strict JSON object body; an empty body is `{}`.
fn body<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, Answer> {
    let bytes = if bytes.iter().all(u8::is_ascii_whitespace) {
        b"{}".as_slice()
    } else {
        bytes
    };
    serde_json::from_slice(bytes).map_err(|error| {
        failure(
            StatusCode::BAD_REQUEST,
            &format!("invalid request body: {error}"),
        )
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateAdminKey {
    name: String,
    expires_in_seconds: Option<i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateProject {
    name: String,
    geography: Option<String>,
    residency: Option<String>,
    external_key_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::option_option)]
struct ModifyProject {
    name: Option<String>,
    #[serde(default, deserialize_with = "present")]
    external_key_id: Option<Option<String>>,
    geography: Option<String>,
}

/// Distinguishes an explicit `null` (`Some(None)`) from an absent field.
#[allow(clippy::option_option)]
fn present<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Option<String>>, D::Error> {
    Option::<String>::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_field_names)]
struct CreateServiceAccount {
    name: String,
    create_service_account_only: Option<bool>,
    expires_in_seconds: Option<i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateServiceAccount {
    name: Option<String>,
    role: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateServiceAccountKey {
    name: Option<String>,
    scopes: Option<Vec<String>>,
    expires_in_seconds: Option<i64>,
}

const RESIDENCY: &str = "data residency does not apply: every byte stays on this machine";

/// Route and answer one Administration API request. Blocks on the store.
#[allow(clippy::too_many_lines)]
pub(super) fn dispatch(
    admin: &Admin,
    method: &Method,
    path: &str,
    query: Option<&str>,
    bytes: &[u8],
    actor: &Actor,
) -> Answer {
    let rest = path.strip_prefix(PREFIX).unwrap_or_default();
    let segments: Vec<&str> = rest.split('/').filter(|part| !part.is_empty()).collect();
    let run = || -> Result<Answer, Answer> {
        let get = *method == Method::GET;
        let post = *method == Method::POST;
        let delete = *method == Method::DELETE;
        Ok(match segments.as_slice() {
            ["admin_api_keys"] if get => {
                let query = Query::parse(query, &["after", "order", "limit"])?;
                ok(admin.list_admin_keys(&query.page()?))
            }
            ["admin_api_keys"] if post => {
                Query::parse(query, &[])?;
                let request: CreateAdminKey = body(bytes)?;
                ok(admin.create_admin_key(&request.name, request.expires_in_seconds, actor))
            }
            ["admin_api_keys", id] if get => ok(admin.get_admin_key(id)),
            ["admin_api_keys", id] if delete => deleted(
                admin.delete_admin_key(id, actor),
                id,
                "organization.admin_api_key.deleted",
            ),
            ["projects"] if get => {
                let query = Query::parse(query, &["limit", "after", "include_archived"])?;
                let archived = query.flag("include_archived")?.unwrap_or(false);
                ok(admin.list_projects(&query.page()?, archived))
            }
            ["projects"] if post => {
                let request: CreateProject = body(bytes)?;
                if request.geography.is_some() || request.residency.is_some() {
                    return Err(unsupported(RESIDENCY));
                }
                ok(admin.create_project(&request.name, request.external_key_id.as_deref(), actor))
            }
            ["projects", id] if get => ok(admin.get_project(id)),
            ["projects", id] if post => {
                let request: ModifyProject = body(bytes)?;
                if request.geography.is_some() {
                    return Err(unsupported(RESIDENCY));
                }
                let changes = ProjectChanges {
                    name: request.name,
                    external_key_id: request.external_key_id,
                };
                ok(admin.modify_project(id, &changes, actor))
            }
            ["projects", id, "archive"] if post => ok(admin.archive_project(id, actor)),
            ["projects", id, "api_keys"] if get => {
                let query = Query::parse(query, &["limit", "after", "owner_project_access"])?;
                let access = query.one("owner_project_access")?;
                ok(admin.list_project_keys(id, &query.page()?, access))
            }
            ["projects", id, "api_keys", key] if get => ok(admin.get_project_key(id, key)),
            ["projects", id, "api_keys", key] if delete => deleted(
                admin.delete_project_key(id, key, actor),
                key,
                "organization.project.api_key.deleted",
            ),
            ["projects", id, "service_accounts"] if get => {
                let query = Query::parse(query, &["limit", "after"])?;
                ok(admin.list_service_accounts(id, &query.page()?))
            }
            ["projects", id, "service_accounts"] if post => {
                let request: CreateServiceAccount = body(bytes)?;
                ok(admin.create_service_account(
                    id,
                    &request.name,
                    request.create_service_account_only.unwrap_or(false),
                    request.expires_in_seconds,
                    actor,
                ))
            }
            ["projects", id, "service_accounts", account] if get => {
                ok(admin.get_service_account(id, account))
            }
            ["projects", id, "service_accounts", account] if post => {
                let request: UpdateServiceAccount = body(bytes)?;
                ok(admin.update_service_account(
                    id,
                    account,
                    request.name.as_deref(),
                    request.role.as_deref(),
                    actor,
                ))
            }
            ["projects", id, "service_accounts", account] if delete => deleted(
                admin.delete_service_account(id, account, actor),
                account,
                "organization.project.service_account.deleted",
            ),
            ["projects", id, "service_accounts", account, "api_keys"] if post => {
                let request: CreateServiceAccountKey = body(bytes)?;
                if request
                    .scopes
                    .as_ref()
                    .is_some_and(|scopes| !scopes.is_empty())
                {
                    return Err(unsupported(
                        "API key scopes are not enforced; every project key may use every \
                         non-administrative API. Omit scopes",
                    ));
                }
                ok(admin.create_service_account_key(
                    id,
                    account,
                    request.name.as_deref(),
                    request.expires_in_seconds,
                    actor,
                ))
            }
            ["projects", _, "rate_limits", ..] => {
                return Err(unsupported(
                    "rate limits are not enforced; the engine queue is the only bound \
                     (requests over it get 503 with Retry-After)",
                ));
            }
            ["projects", _, "users" | "groups", ..] => {
                return Err(unsupported(
                    "project membership is not modeled: the single local owner owns every \
                     project; use service accounts for project keys",
                ));
            }
            ["projects", _, "certificates", ..] | ["certificates", ..] => {
                return Err(unsupported(
                    "mutual TLS client certificates are not supported; TLS is configured from \
                     files at startup",
                ));
            }
            ["projects", _, "spend_alerts" | "spend_limit", ..]
            | ["spend_alerts" | "spend_limit", ..] => {
                return Err(unsupported(
                    "nothing is billed locally, so there is no spend",
                ));
            }
            ["projects", _, "data_retention", ..] | ["data_retention", ..] => {
                return Err(unsupported(
                    "retention is local: stored objects stay until deleted or expired",
                ));
            }
            [
                "projects",
                _,
                "model_permissions" | "hosted_tool_permissions",
                ..,
            ] => {
                return Err(unsupported(
                    "per-project permissions are not enforced; hosted tools are granted by \
                     server flags",
                ));
            }
            ["users"] if get => {
                let query = Query::parse(query, &["limit", "after", "emails"])?;
                query.page()?;
                let owner = admin.owner().map_err(|error| store_failure(&error))?;
                // The owner has no email address, so an email filter matches nothing.
                let data = if query.all("emails").is_empty() && query.one("after")?.is_none() {
                    vec![owner.user()]
                } else {
                    Vec::new()
                };
                list_of(&data)
            }
            ["users", id] if get => {
                let owner = admin.owner().map_err(|error| store_failure(&error))?;
                if *id == owner.id {
                    (StatusCode::OK, owner.user())
                } else {
                    return Err(failure(
                        StatusCode::NOT_FOUND,
                        &format!("no user with id {id:?}"),
                    ));
                }
            }
            ["users" | "groups" | "roles", ..] => {
                return Err(unsupported(&format!(
                    "the organization is the single local owner ({}), who cannot be modified, \
                     removed or given roles",
                    Owner::NAME
                )));
            }
            ["invites"] if get => {
                Query::parse(query, &["limit", "after"])?;
                list_of(&[])
            }
            ["invites", id] if get || delete => {
                return Err(failure(
                    StatusCode::NOT_FOUND,
                    &format!("no invite with id {id:?}"),
                ));
            }
            ["invites"] if post => {
                return Err(unsupported(
                    "the organization is the single local owner; there is no one to invite",
                ));
            }
            ["external_storage", ..] => {
                return Err(unsupported("every object is stored on this machine"));
            }
            ["audit_logs"] if get => audit_logs(admin, query)?,
            ["usage", kind] if get => {
                let kind = UsageKind::from_segment(kind).ok_or_else(|| {
                    failure(StatusCode::NOT_FOUND, &format!("no usage family {kind:?}"))
                })?;
                ok(admin.usage(&usage_query(kind, query)?))
            }
            ["costs"] if get => costs(query)?,
            _ => {
                return Err(failure(StatusCode::NOT_FOUND, "route not found"));
            }
        })
    };
    run().unwrap_or_else(|answer| answer)
}

fn list_of(data: &[Value]) -> Answer {
    let first = data
        .first()
        .and_then(|item| item["id"].as_str())
        .map(str::to_owned);
    let last = data
        .last()
        .and_then(|item| item["id"].as_str())
        .map(str::to_owned);
    (
        StatusCode::OK,
        json!({"object":"list","data":data,"first_id":first,"last_id":last,"has_more":false}),
    )
}

fn audit_logs(admin: &Admin, query: Option<&str>) -> Result<Answer, Answer> {
    let query = Query::parse(
        query,
        &[
            "effective_at",
            "project_ids",
            "event_types",
            "actor_ids",
            "actor_emails",
            "resource_ids",
            "tenant_only",
            "limit",
            "after",
            "before",
        ],
    )?;
    let mut audit = AuditQuery {
        project_ids: query.all("project_ids"),
        event_types: query.all("event_types"),
        actor_ids: query.all("actor_ids"),
        actor_emails: query.all("actor_emails"),
        resource_ids: query.all("resource_ids"),
        limit: query.number("limit")?,
        after: query.one("after")?.map(str::to_owned),
        before: query.one("before")?.map(str::to_owned),
        ..AuditQuery::default()
    };
    for (name, slot) in [
        ("effective_at[gt]", &mut audit.effective_gt),
        ("effective_at[gte]", &mut audit.effective_gte),
        ("effective_at[lt]", &mut audit.effective_lt),
        ("effective_at[lte]", &mut audit.effective_lte),
    ] {
        *slot = query.number(name)?;
    }
    if let Some((name, _)) = query
        .0
        .iter()
        .find(|(name, _)| name.starts_with("effective_at") && !name.ends_with(']'))
    {
        return Err(failure(
            StatusCode::BAD_REQUEST,
            &format!("{name} must be effective_at[gt|gte|lt|lte]"),
        ));
    }
    // Every entry is the organization's own; there are no tenant-level ones.
    let tenant_only = query.flag("tenant_only")?.unwrap_or(false);
    if tenant_only {
        return Ok(list_of(&[]));
    }
    Ok(ok(admin.list_audit_logs(&audit)))
}

fn usage_query(kind: UsageKind, query: Option<&str>) -> Result<UsageQuery, Answer> {
    let mut allowed = vec![
        "start_time",
        "end_time",
        "bucket_width",
        "project_ids",
        "group_by",
        "limit",
        "page",
    ];
    if !matches!(
        kind,
        UsageKind::VectorStores | UsageKind::CodeInterpreterSessions
    ) {
        allowed.extend(["user_ids", "api_key_ids"]);
    }
    if !matches!(
        kind,
        UsageKind::VectorStores | UsageKind::CodeInterpreterSessions | UsageKind::FileSearchCalls
    ) {
        allowed.push("models");
    }
    match kind {
        UsageKind::Completions => allowed.push("batch"),
        UsageKind::Images => allowed.extend(["sources", "sizes"]),
        UsageKind::WebSearchCalls => allowed.push("context_levels"),
        UsageKind::FileSearchCalls => allowed.push("vector_store_ids"),
        _ => {}
    }
    let query = Query::parse(query, &allowed)?;
    let start_time = query.number("start_time")?.ok_or_else(|| {
        failure(
            StatusCode::BAD_REQUEST,
            "start_time (Unix seconds) is required",
        )
    })?;
    let bucket_width = match query.one("bucket_width")? {
        None => DAY,
        Some(width) => parse_width(width).ok_or_else(|| {
            failure(
                StatusCode::BAD_REQUEST,
                &format!("bucket_width must be 1m, 1h or 1d, not {width:?}"),
            )
        })?,
    };
    // Sources, sizes, context levels and vector store IDs are never recorded:
    // this server makes no images, web searches or file search calls, so
    // those families are always empty and the filters change nothing.
    Ok(UsageQuery {
        kind,
        start_time,
        end_time: query.number("end_time")?,
        bucket_width,
        limit: query.number("limit")?,
        page: query.one("page")?.map(str::to_owned),
        project_ids: query.all("project_ids"),
        user_ids: query.all("user_ids"),
        api_key_ids: query.all("api_key_ids"),
        models: query.all("models"),
        batch: query.flag("batch")?,
        group_by: query.all("group_by"),
    })
}

/// `GET /organization/costs`: nothing is billed locally, so every daily
/// bucket in range is returned with no cost line items.
fn costs(query: Option<&str>) -> Result<Answer, Answer> {
    let query = Query::parse(
        query,
        &[
            "start_time",
            "end_time",
            "bucket_width",
            "project_ids",
            "api_key_ids",
            "line_items",
            "group_by",
            "limit",
            "page",
        ],
    )?;
    let start_time: i64 = query.number("start_time")?.ok_or_else(|| {
        failure(
            StatusCode::BAD_REQUEST,
            "start_time (Unix seconds) is required",
        )
    })?;
    if let Some(width) = query.one("bucket_width")?
        && width != "1d"
    {
        return Err(failure(
            StatusCode::BAD_REQUEST,
            &format!("bucket_width must be 1d, not {width:?}"),
        ));
    }
    for field in query.all("group_by") {
        if !matches!(
            field.as_str(),
            "project_id" | "user_id" | "line_item" | "api_key_id" | "api_source"
        ) {
            return Err(failure(
                StatusCode::BAD_REQUEST,
                &format!("group_by {field:?} is not supported for costs"),
            ));
        }
    }
    let limit = query.number("limit")?.unwrap_or(COST_LIMITS.0);
    if !(1..=COST_LIMITS.1).contains(&limit) {
        return Err(failure(
            StatusCode::BAD_REQUEST,
            &format!("limit must be between 1 and {}", COST_LIMITS.1),
        ));
    }
    // Without an end, through the end of the current day.
    let end_time = query.number("end_time")?.unwrap_or_else(|| {
        let now = i64::try_from(super::response::unix_now()).unwrap_or(i64::MAX);
        (now.div_euclid(DAY) + 1) * DAY
    });
    let (spans, next_page) = bucket_spans(start_time, end_time, DAY, limit, query.one("page")?)
        .map_err(|error| store_failure(&error))?;
    let data: Vec<Value> = spans
        .into_iter()
        .map(|(start, end)| json!({"object":"bucket","start_time":start,"end_time":end,"results":[]}))
        .collect();
    Ok((
        StatusCode::OK,
        json!({"object":"page","data":data,"has_more":next_page.is_some(),"next_page":next_page}),
    ))
}

/// Record `record` off the runtime, logging (never failing the request) on
/// a store error.
pub(super) fn record(state: &AppState, record: UsageRecord) {
    if let Some(admin) = state.auth.admin().cloned() {
        tokio::task::spawn_blocking(move || {
            if let Err(error) = admin.record_usage(&record) {
                eprintln!("warning: usage not recorded: {error}");
            }
        });
    }
}

/// Where a generation that runs outside the request's own event stream
/// (batch lines, eval runs, compaction and reasoning summaries) records its
/// usage: the usage store and a record template, or nowhere while there is
/// no store.
#[derive(Clone, Default)]
pub(super) struct Meter(Option<(Arc<Admin>, UsageRecord)>);

impl Meter {
    /// A `completions` meter of `caller`'s for the served model.
    pub(super) fn completions(state: &AppState, caller: &Caller) -> Self {
        Self::into_store(
            state.auth.admin(),
            caller.usage(UsageKind::Completions, &state.model),
        )
    }

    /// `record` (counts zero) as the template written to `admin`.
    pub(super) fn into_store(admin: Option<&Arc<Admin>>, record: UsageRecord) -> Self {
        Self(admin.map(|admin| (Arc::clone(admin), record)))
    }

    /// This meter with its records attributed to project `project_id` and
    /// its API key `api_key_id` (work stored by that caller and run later).
    pub(super) fn attributed(&self, project_id: &str, api_key_id: &str) -> Self {
        let mut meter = self.clone();
        if let Some((_, record)) = &mut meter.0 {
            record.project_id = Some(project_id.to_owned());
            record.api_key_id = Some(api_key_id.to_owned());
        }
        meter
    }

    /// Record one finished generation from its final measurements. Blocks on
    /// the store: call it from a blocking context.
    pub(super) fn generation(&self, stats: &Stats) {
        if let Some((admin, template)) = &self.0 {
            let mut record = template.clone();
            let generation = &stats.generation;
            record.input_tokens = count(generation.prompt_tokens);
            record.input_cached_tokens = count(generation.reused_prompt_tokens);
            record.output_tokens = count(generation.generated_tokens);
            if let Err(error) = admin.record_usage(&record) {
                eprintln!("warning: usage not recorded: {error}");
            }
        }
    }

    /// Record the template as it is (a request whose quantity was known when
    /// the meter was made). Blocks on the store.
    pub(super) fn write(&self) {
        if let Some((admin, record)) = &self.0
            && let Err(error) = admin.record_usage(record)
        {
            eprintln!("warning: usage not recorded: {error}");
        }
    }

    /// Record the template completed by `fill`. Blocks on the store.
    pub(super) fn write_with(&self, fill: impl FnOnce(&mut UsageRecord)) {
        if let Some((admin, template)) = &self.0 {
            let mut record = template.clone();
            fill(&mut record);
            if let Err(error) = admin.record_usage(&record) {
                eprintln!("warning: usage not recorded: {error}");
            }
        }
    }

    /// [`Self::generation`] off the runtime.
    pub(super) fn spawn_generation(&self, stats: &Stats) {
        if self.0.is_some() {
            let (meter, stats) = (self.clone(), stats.clone());
            tokio::task::spawn_blocking(move || meter.generation(&stats));
        }
    }
}

/// `events` with its usage recorded, once it finishes, as a `completions`
/// record of `caller`'s.
///
/// The events are forwarded unchanged (prefill progress included) through a
/// relay on the blocking pool; the consumer's cancellation reaches the
/// generation. A consumer that goes away still has the tokens it caused
/// recorded: the generation is cancelled and drained to its final
/// measurements. The record is written after the final event is forwarded
/// and before the stream ends.
pub(super) fn meter(state: &AppState, caller: &Caller, events: EventStream) -> EventStream {
    match state.auth.admin() {
        Some(admin) => metered(
            Arc::clone(admin),
            caller.usage(UsageKind::Completions, &state.model),
            events,
        ),
        None => events,
    }
}

/// [`meter`] into `admin`, completing `record` from the final measurements.
fn metered(admin: Arc<Admin>, record: UsageRecord, events: EventStream) -> EventStream {
    let (relay, stream) = EventStream::relay();
    let relay = Arc::new(relay);
    let cancel = events.cancel_handle();
    let canceller = {
        let relay = Arc::clone(&relay);
        tokio::spawn(async move {
            relay.cancelled().await;
            cancel.cancel();
        })
    };
    tokio::task::spawn_blocking(move || {
        let mut events = events;
        let mut record = record;
        let mut finished = false;
        let mut forwarding = true;
        while let Ok(Some(signal)) = events.next_signal(None) {
            if let Signal::Event(Event::Finished(stats)) = &signal {
                let generation = &stats.generation;
                record.input_tokens += count(generation.prompt_tokens);
                record.input_cached_tokens += count(generation.reused_prompt_tokens);
                record.output_tokens += count(generation.generated_tokens);
                finished = true;
            }
            if forwarding && !relay.blocking_send(signal) {
                forwarding = false;
                events.cancel_handle().cancel();
            }
        }
        // Recorded before the stream ends, so a client that has read the
        // whole response finds its usage.
        if finished && let Err(error) = admin.record_usage(&record) {
            eprintln!("warning: usage not recorded: {error}");
        }
        drop(relay);
        canceller.abort();
    });
    stream
}

/// The usage of one request of a family measured in input tokens
/// (`embeddings`, `moderations`).
pub(super) fn token_usage(
    caller: &Caller,
    kind: UsageKind,
    model: &str,
    tokens: usize,
) -> UsageRecord {
    let mut record = caller.usage(kind, model);
    record.input_tokens = count(tokens);
    record
}

/// The `audio_speeches` usage of one speech request: the characters of its
/// input.
pub(super) fn speech_usage(caller: &Caller, model: &str, input: &str) -> UsageRecord {
    let mut record = caller.usage(UsageKind::AudioSpeeches, model);
    record.units = count(input.chars().count());
    record
}

/// The `audio_transcriptions` usage of one transcription or translation:
/// the seconds of audio it decoded, a started second counting as one.
pub(super) fn transcription_usage(caller: &Caller, model: &str, seconds: f64) -> UsageRecord {
    let mut record = caller.usage(UsageKind::AudioTranscriptions, model);
    record.units = audio_seconds(seconds);
    record
}

/// Seconds of audio as a usage column, a started second counting as one.
pub(super) fn audio_seconds(seconds: f64) -> i64 {
    if seconds.is_finite() && seconds > 0.0 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let whole = seconds.ceil().min(i64::MAX as f64) as i64;
        whole
    } else {
        0
    }
}

/// A token count as a usage column.
pub(super) fn count(tokens: usize) -> i64 {
    i64::try_from(tokens).unwrap_or(i64::MAX)
}
