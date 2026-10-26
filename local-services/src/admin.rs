//! Opt-in authentication and the `OpenAI` Administration API objects.
//!
//! For a local, single-organization server: admin API keys, projects,
//! project service accounts and their API keys, audit logs, and per-request
//! usage.
//!
//! # Keys
//!
//! A key is `sk-admin-` or `sk-proj-`, then 16 hex digits of lookup, `_`, and
//! 64 hex digits (256 bits) of OS randomness. Only the lookup, the redacted
//! form (`sk-admin...xyz`: the prefix and the last three characters, as the
//! `OpenAI` API shows them) and an HMAC-SHA-256 of the whole key under a fresh
//! 256-bit per-key salt are stored; the plaintext is returned once, at
//! creation, and never again. Verification recomputes the HMAC and compares
//! the tags with [`ring::hmac::verify`], which runs in constant time. A key
//! of 256 random bits needs no slow password hash: the salt only prevents one
//! tag from matching across keys or databases.
//!
//! # Authentication state
//!
//! Authentication is off until the first admin key is created (the
//! `local-ai admin create-key` command) and stays on, even if every admin key
//! is later deleted, until the owner turns it off from the command line. It is
//! read from the database on every check, so the command takes effect on a
//! running server at once.

use rusqlite::types::Value as SqlValue;
use rusqlite::{Connection, OptionalExtension as _, Row, TransactionBehavior, params};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{Error, Result, Store, new_id, now};

pub mod usage;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;

/// Prefix of an admin API key.
pub const ADMIN_KEY_PREFIX: &str = "sk-admin-";
/// Prefix of a project API key.
pub const PROJECT_KEY_PREFIX: &str = "sk-proj-";
/// Hex digits of a key's lookup part.
const LOOKUP_HEX: usize = 16;
/// Hex digits of a key's secret part.
const SECRET_HEX: usize = 64;
/// Longest key lifetime the API accepts (one year).
pub const MAX_EXPIRES_IN_SECONDS: i64 = 31_536_000;
/// `last_used_at` is written at most this often per key.
const LAST_USED_GRANULARITY: i64 = 60;
/// Page sizes for lists.
pub const DEFAULT_LIST_LIMIT: usize = 20;
pub const MAX_LIST_LIMIT: usize = 100;
/// Longest name accepted for keys, projects and service accounts.
const MAX_NAME_BYTES: usize = 256;

const MIGRATIONS: &[&str] = &["
CREATE TABLE admin_settings (
    name TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
) STRICT;
INSERT INTO admin_settings (name, value) VALUES
    ('owner_id', 'user_' || lower(hex(randomblob(12)))),
    ('owner_created_at', CAST(unixepoch() AS TEXT)),
    ('auth_enabled', '0');
CREATE TABLE admin_api_keys (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    lookup TEXT NOT NULL UNIQUE,
    salt BLOB NOT NULL,
    tag BLOB NOT NULL,
    name TEXT,
    redacted TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER,
    last_used_at INTEGER
) STRICT;
CREATE TABLE admin_projects (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    external_key_id TEXT,
    created_at INTEGER NOT NULL,
    archived_at INTEGER
) STRICT;
CREATE TABLE admin_service_accounts (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    project_id TEXT NOT NULL REFERENCES admin_projects (id),
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('owner', 'member', 'none')),
    created_at INTEGER NOT NULL
) STRICT;
CREATE INDEX admin_service_accounts_project ON admin_service_accounts (project_id, seq);
CREATE TABLE admin_project_keys (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    lookup TEXT NOT NULL UNIQUE,
    salt BLOB NOT NULL,
    tag BLOB NOT NULL,
    project_id TEXT NOT NULL REFERENCES admin_projects (id),
    service_account_id TEXT NOT NULL
        REFERENCES admin_service_accounts (id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    redacted TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER,
    last_used_at INTEGER
) STRICT;
CREATE INDEX admin_project_keys_project ON admin_project_keys (project_id, seq);
CREATE TABLE admin_audit_logs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    effective_at INTEGER NOT NULL,
    project_id TEXT,
    project_name TEXT,
    resource_id TEXT,
    actor TEXT NOT NULL,
    details TEXT NOT NULL
) STRICT;
CREATE TABLE usage_records (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    at INTEGER NOT NULL,
    project_id TEXT,
    user_id TEXT,
    api_key_id TEXT,
    model TEXT,
    batch INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    input_cached_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    units INTEGER NOT NULL,
    requests INTEGER NOT NULL
) STRICT;
CREATE INDEX usage_records_kind_at ON usage_records (kind, at);
"];

pub(crate) fn initialize(connection: &mut Connection) -> Result<()> {
    crate::migrate(connection, "admin", MIGRATIONS)
}

/// Which family a key belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyKind {
    Admin,
    Project,
}

impl KeyKind {
    const fn prefix(self) -> &'static str {
        match self {
            Self::Admin => ADMIN_KEY_PREFIX,
            Self::Project => PROJECT_KEY_PREFIX,
        }
    }

    const fn table(self) -> &'static str {
        match self {
            Self::Admin => "admin_api_keys",
            Self::Project => "admin_project_keys",
        }
    }
}

/// A freshly minted key: everything stored about it plus the plaintext.
struct Minted {
    token: String,
    lookup: String,
    salt: [u8; 32],
    tag: Vec<u8>,
    redacted: String,
}

fn random_bytes<const N: usize>() -> Result<[u8; N]> {
    use ring::rand::{SecureRandom as _, SystemRandom};
    let mut bytes = [0_u8; N];
    SystemRandom::new().fill(&mut bytes).map_err(|_| {
        Error::Io(std::io::Error::other(
            "operating system randomness is unavailable",
        ))
    })?;
    Ok(bytes)
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(HEX[usize::from(byte >> 4)]));
        text.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    text
}

/// The salted HMAC-SHA-256 tag stored for `token`.
fn tag(salt: &[u8], token: &str) -> Vec<u8> {
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, salt);
    ring::hmac::sign(&key, token.as_bytes()).as_ref().to_vec()
}

/// Whether `token` is the key whose salt and tag are stored, in constant time.
fn verify(salt: &[u8], stored_tag: &[u8], token: &str) -> bool {
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, salt);
    ring::hmac::verify(&key, token.as_bytes(), stored_tag).is_ok()
}

fn mint(kind: KeyKind) -> Result<Minted> {
    let lookup = hex(&random_bytes::<{ LOOKUP_HEX / 2 }>()?);
    let secret = hex(&random_bytes::<{ SECRET_HEX / 2 }>()?);
    let token = format!("{}{lookup}_{secret}", kind.prefix());
    let salt = random_bytes::<32>()?;
    let tag = tag(&salt, &token);
    let tail: String = token
        .chars()
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .iter()
        .rev()
        .collect();
    let redacted = format!("{}...{tail}", kind.prefix().trim_end_matches('-'));
    Ok(Minted {
        token,
        lookup,
        salt,
        tag,
        redacted,
    })
}

/// The family and lookup of a well-formed key, or `None`.
fn parse_token(token: &str) -> Option<(KeyKind, &str)> {
    let (kind, rest) = if let Some(rest) = token.strip_prefix(ADMIN_KEY_PREFIX) {
        (KeyKind::Admin, rest)
    } else {
        (KeyKind::Project, token.strip_prefix(PROJECT_KEY_PREFIX)?)
    };
    let (lookup, secret) = rest.split_once('_')?;
    let hex_digits = |text: &str, length: usize| {
        text.len() == length
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    (hex_digits(lookup, LOOKUP_HEX) && hex_digits(secret, SECRET_HEX)).then_some((kind, lookup))
}

/// Who an authenticated request acts as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Credential {
    /// An admin API key: the Administration API only.
    Admin { key_id: String },
    /// A project API key: every other API, attributed to its project.
    Project {
        key_id: String,
        project_id: String,
        service_account_id: String,
    },
}

/// The outcome of checking a request's bearer token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Authentication is off: every request is allowed.
    Open,
    /// The token is a valid, live key.
    Granted(Credential),
    /// Authentication is on and the token is missing or not valid; the reason
    /// names no secret.
    Refused(&'static str),
}

/// Who performed an administrative action, for the audit log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Actor {
    /// The owner, from the `local-ai admin` command line.
    Owner,
    /// A request made with admin API key `id`.
    AdminKey(String),
}

/// The local organization's one user: whoever owns the data directory.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Owner {
    pub id: String,
    pub created_at: i64,
}

impl Owner {
    pub const NAME: &'static str = "Local owner";

    /// The owner as an admin key's `owner`.
    fn key_owner(&self) -> Value {
        json!({"type":"user","object":"organization.user","id":self.id,"name":Self::NAME,
               "created_at":self.created_at,"role":"owner"})
    }

    /// The owner as an `organization.user`.
    #[must_use]
    pub fn user(&self) -> Value {
        json!({"object":"organization.user","id":self.id,"name":Self::NAME,"email":null,
               "role":"owner","added_at":self.created_at,"is_default":true,
               "is_service_account":false})
    }
}

/// A page of objects in the `OpenAI` list shape.
#[derive(Clone, Debug, Serialize)]
pub struct List<T> {
    pub object: &'static str,
    pub data: Vec<T>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
}

/// An object with an `id`, for list cursors.
pub trait Identified {
    fn id(&self) -> &str;
}

/// Which slice of a list to return.
#[derive(Clone, Debug, Default)]
pub struct Page {
    /// 1 to [`MAX_LIST_LIMIT`]; [`DEFAULT_LIST_LIMIT`] when `None`.
    pub limit: Option<usize>,
    /// Objects after this ID in list order.
    pub after: Option<String>,
    /// Newest first instead of oldest first.
    pub descending: bool,
}

impl Page {
    fn limit(&self) -> Result<usize> {
        let limit = self.limit.unwrap_or(DEFAULT_LIST_LIMIT);
        if (1..=MAX_LIST_LIMIT).contains(&limit) {
            Ok(limit)
        } else {
            Err(Error::InvalidArgument(format!(
                "limit must be between 1 and {MAX_LIST_LIMIT}"
            )))
        }
    }
}

/// `organization.admin_api_key`.
#[derive(Clone, Debug, Serialize)]
pub struct AdminApiKey {
    pub object: &'static str,
    pub id: String,
    pub name: Option<String>,
    pub redacted_value: String,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub last_used_at: Option<i64>,
    pub owner: Value,
}

impl Identified for AdminApiKey {
    fn id(&self) -> &str {
        &self.id
    }
}

/// An object returned at creation together with its key's only plaintext.
#[derive(Clone, Debug, Serialize)]
pub struct WithValue<T> {
    #[serde(flatten)]
    pub object: T,
    pub value: String,
}

/// `organization.project`.
#[derive(Clone, Debug, Serialize)]
pub struct Project {
    pub id: String,
    pub object: &'static str,
    pub name: String,
    pub created_at: i64,
    pub archived_at: Option<i64>,
    pub status: &'static str,
    pub external_key_id: Option<String>,
}

impl Identified for Project {
    fn id(&self) -> &str {
        &self.id
    }
}

/// `organization.project.service_account`.
#[derive(Clone, Debug, Serialize)]
pub struct ServiceAccount {
    pub object: &'static str,
    pub id: String,
    pub name: String,
    pub role: String,
    pub created_at: i64,
}

impl Identified for ServiceAccount {
    fn id(&self) -> &str {
        &self.id
    }
}

/// `organization.project.service_account.api_key`: a new key, shown once.
#[derive(Clone, Debug, Serialize)]
pub struct ServiceAccountApiKey {
    pub object: &'static str,
    pub value: String,
    pub name: String,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub id: String,
}

/// A created service account and, unless it was created alone, its first key.
#[derive(Clone, Debug, Serialize)]
pub struct CreatedServiceAccount {
    #[serde(flatten)]
    pub account: ServiceAccount,
    pub api_key: Option<ServiceAccountApiKey>,
}

/// `organization.project.api_key`.
#[derive(Clone, Debug, Serialize)]
pub struct ProjectApiKey {
    pub object: &'static str,
    pub redacted_value: String,
    pub name: String,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub last_used_at: Option<i64>,
    pub id: String,
    pub owner_project_access: &'static str,
    pub owner: Value,
}

impl Identified for ProjectApiKey {
    fn id(&self) -> &str {
        &self.id
    }
}

/// An audit log entry, already in the `OpenAI` shape.
#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
pub struct AuditLog {
    pub value: Value,
    #[serde(skip)]
    id: String,
}

impl Identified for AuditLog {
    fn id(&self) -> &str {
        &self.id
    }
}

/// Filters of `GET /organization/audit_logs`.
#[derive(Clone, Debug, Default)]
pub struct AuditQuery {
    pub effective_gt: Option<i64>,
    pub effective_gte: Option<i64>,
    pub effective_lt: Option<i64>,
    pub effective_lte: Option<i64>,
    pub project_ids: Vec<String>,
    pub event_types: Vec<String>,
    pub actor_ids: Vec<String>,
    pub actor_emails: Vec<String>,
    pub resource_ids: Vec<String>,
    pub limit: Option<usize>,
    pub after: Option<String>,
    pub before: Option<String>,
}

/// Fields of `POST /organization/projects/{id}`.
#[derive(Clone, Debug, Default)]
pub struct ProjectChanges {
    pub name: Option<String>,
    pub external_key_id: Option<Option<String>>,
}

fn check_name(name: &str, what: &str) -> Result<()> {
    if name.trim().is_empty() || name.len() > MAX_NAME_BYTES {
        return Err(Error::InvalidArgument(format!(
            "{what} must be a non-empty string of at most {MAX_NAME_BYTES} bytes"
        )));
    }
    Ok(())
}

fn expiry(created_at: i64, expires_in: Option<i64>) -> Result<Option<i64>> {
    expires_in
        .map(|seconds| {
            if (1..=MAX_EXPIRES_IN_SECONDS).contains(&seconds) {
                Ok(created_at + seconds)
            } else {
                Err(Error::InvalidArgument(format!(
                    "expires_in_seconds must be between 1 and {MAX_EXPIRES_IN_SECONDS}"
                )))
            }
        })
        .transpose()
}

fn now_signed() -> Result<i64> {
    i64::try_from(now()?).map_err(|error| Error::Io(std::io::Error::other(error)))
}

/// The administrative store over a [`Store`].
#[derive(Debug)]
pub struct Admin {
    store: Store,
}

impl Admin {
    #[must_use]
    pub const fn new(store: Store) -> Self {
        Self { store }
    }

    /// The underlying store.
    #[must_use]
    pub const fn store(&self) -> &Store {
        &self.store
    }

    /// The local owner.
    pub fn owner(&self) -> Result<Owner> {
        owner(&*self.store.connection()?)
    }

    /// Whether authentication is on.
    pub fn auth_enabled(&self) -> Result<bool> {
        auth_enabled(&*self.store.connection()?)
    }

    /// Turn authentication off (the command line's `disable-auth`); keys are kept.
    pub fn disable_auth(&self) -> Result<()> {
        let connection = self.store.connection()?;
        connection.execute(
            "UPDATE admin_settings SET value = '0' WHERE name = 'auth_enabled'",
            [],
        )?;
        Ok(())
    }

    /// Turn authentication back on; refused while no live admin key exists,
    /// because the Administration API would then be unreachable.
    pub fn enable_auth(&self) -> Result<()> {
        let connection = self.store.connection()?;
        let live: i64 = connection.query_row(
            "SELECT COUNT(*) FROM admin_api_keys WHERE expires_at IS NULL OR expires_at > ?1",
            [now_signed()?],
            |row| row.get(0),
        )?;
        if live == 0 {
            return Err(Error::Conflict(
                "no live admin API key exists; create one with `local-ai admin create-key`".into(),
            ));
        }
        connection.execute(
            "UPDATE admin_settings SET value = '1' WHERE name = 'auth_enabled'",
            [],
        )?;
        Ok(())
    }

    /// Check bearer token `token` (without the `Bearer ` prefix).
    pub fn authenticate(&self, token: Option<&str>) -> Result<Verdict> {
        // A pooled connection: no database connection is opened per request.
        authenticate(&*self.store.connection()?, token)
    }

    /// Create an admin API key, turning authentication on.
    pub fn create_admin_key(
        &self,
        name: &str,
        expires_in: Option<i64>,
        actor: &Actor,
    ) -> Result<WithValue<AdminApiKey>> {
        check_name(name, "name")?;
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let created_at = now_signed()?;
        let expires_at = expiry(created_at, expires_in)?;
        let minted = mint(KeyKind::Admin)?;
        let id = new_id("key_")?;
        transaction.execute(
            "INSERT INTO admin_api_keys (id, lookup, salt, tag, name, redacted, created_at,
                 expires_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                id,
                minted.lookup,
                minted.salt.as_slice(),
                minted.tag,
                name,
                minted.redacted,
                created_at,
                expires_at
            ],
        )?;
        transaction.execute(
            "UPDATE admin_settings SET value = '1' WHERE name = 'auth_enabled'",
            [],
        )?;
        audit(
            &transaction,
            actor,
            "api_key.created",
            None,
            &id,
            &json!({"id":id,"data":{"scopes":["api.management.read","api.management.write"]}}),
        )?;
        let key = admin_key_by_id(&transaction, &id)?;
        transaction.commit()?;
        Ok(WithValue {
            object: key,
            value: minted.token,
        })
    }

    pub fn list_admin_keys(&self, page: &Page) -> Result<List<AdminApiKey>> {
        let connection = self.store.connection()?;
        let owner = owner(&connection)?;
        list(
            &connection,
            "admin_api_keys",
            ADMIN_KEY_COLUMNS,
            "1",
            Vec::new(),
            page,
            |row| admin_key(row, &owner),
        )
    }

    pub fn get_admin_key(&self, id: &str) -> Result<AdminApiKey> {
        admin_key_by_id(&*self.store.connection()?, id)
    }

    pub fn delete_admin_key(&self, id: &str, actor: &Actor) -> Result<()> {
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if transaction.execute("DELETE FROM admin_api_keys WHERE id = ?1", [id])? == 0 {
            return Err(not_found("admin API key", id));
        }
        audit(
            &transaction,
            actor,
            "api_key.deleted",
            None,
            id,
            &json!({"id":id}),
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn create_project(
        &self,
        name: &str,
        external_key_id: Option<&str>,
        actor: &Actor,
    ) -> Result<Project> {
        check_name(name, "name")?;
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let id = new_id("proj_")?;
        transaction.execute(
            "INSERT INTO admin_projects (id, name, external_key_id, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![id, name, external_key_id, now_signed()?],
        )?;
        let project = project_by_id(&transaction, &id)?;
        audit(
            &transaction,
            actor,
            "project.created",
            Some(&project),
            &id,
            &json!({"id":id,"data":{"name":name,"title":name}}),
        )?;
        transaction.commit()?;
        Ok(project)
    }

    pub fn list_projects(&self, page: &Page, include_archived: bool) -> Result<List<Project>> {
        let filter = if include_archived {
            "1"
        } else {
            "archived_at IS NULL"
        };
        list(
            &*self.store.connection()?,
            "admin_projects",
            PROJECT_COLUMNS,
            filter,
            Vec::new(),
            page,
            project,
        )
    }

    pub fn get_project(&self, id: &str) -> Result<Project> {
        project_by_id(&*self.store.connection()?, id)
    }

    pub fn modify_project(
        &self,
        id: &str,
        changes: &ProjectChanges,
        actor: &Actor,
    ) -> Result<Project> {
        if let Some(name) = &changes.name {
            check_name(name, "name")?;
        }
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = project_by_id(&transaction, id)?;
        if current.archived_at.is_some() {
            return Err(Error::Conflict(format!(
                "project {id} is archived and cannot be modified"
            )));
        }
        if let Some(name) = &changes.name {
            transaction.execute(
                "UPDATE admin_projects SET name = ?2 WHERE id = ?1",
                params![id, name],
            )?;
        }
        if let Some(external) = &changes.external_key_id {
            transaction.execute(
                "UPDATE admin_projects SET external_key_id = ?2 WHERE id = ?1",
                params![id, external],
            )?;
        }
        let project = project_by_id(&transaction, id)?;
        let mut requested = serde_json::Map::new();
        if let Some(name) = &changes.name {
            requested.insert("title".into(), json!(name));
        }
        audit(
            &transaction,
            actor,
            "project.updated",
            Some(&project),
            id,
            &json!({"id":id,"changes_requested":requested}),
        )?;
        transaction.commit()?;
        Ok(project)
    }

    /// Archive project `id`: its keys stop authenticating. Idempotent.
    pub fn archive_project(&self, id: &str, actor: &Actor) -> Result<Project> {
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = project_by_id(&transaction, id)?;
        if current.archived_at.is_some() {
            return Ok(current);
        }
        transaction.execute(
            "UPDATE admin_projects SET archived_at = ?2 WHERE id = ?1",
            params![id, now_signed()?],
        )?;
        let project = project_by_id(&transaction, id)?;
        audit(
            &transaction,
            actor,
            "project.archived",
            Some(&project),
            id,
            &json!({"id":id}),
        )?;
        transaction.commit()?;
        Ok(project)
    }

    pub fn create_service_account(
        &self,
        project_id: &str,
        name: &str,
        create_only: bool,
        expires_in: Option<i64>,
        actor: &Actor,
    ) -> Result<CreatedServiceAccount> {
        check_name(name, "name")?;
        if create_only && expires_in.is_some() {
            return Err(Error::InvalidArgument(
                "expires_in_seconds cannot be used with create_service_account_only".into(),
            ));
        }
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let project = active_project(&transaction, project_id)?;
        let created_at = now_signed()?;
        let id = new_id("svc_acct_")?;
        let role = if create_only { "none" } else { "member" };
        transaction.execute(
            "INSERT INTO admin_service_accounts (id, project_id, name, role, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, project_id, name, role, created_at],
        )?;
        audit(
            &transaction,
            actor,
            "service_account.created",
            Some(&project),
            &id,
            &json!({"id":id,"data":{"role":role}}),
        )?;
        let api_key = if create_only {
            None
        } else {
            Some(insert_project_key(
                &transaction,
                &project,
                &id,
                "Secret Key",
                expires_in,
                actor,
            )?)
        };
        let account = service_account_by_id(&transaction, project_id, &id)?;
        transaction.commit()?;
        Ok(CreatedServiceAccount { account, api_key })
    }

    pub fn list_service_accounts(
        &self,
        project_id: &str,
        page: &Page,
    ) -> Result<List<ServiceAccount>> {
        let connection = self.store.connection()?;
        project_by_id(&connection, project_id)?;
        list(
            &connection,
            "admin_service_accounts",
            SERVICE_ACCOUNT_COLUMNS,
            "project_id = ?",
            vec![SqlValue::Text(project_id.to_owned())],
            page,
            service_account,
        )
    }

    pub fn get_service_account(&self, project_id: &str, id: &str) -> Result<ServiceAccount> {
        let connection = self.store.connection()?;
        project_by_id(&connection, project_id)?;
        service_account_by_id(&connection, project_id, id)
    }

    pub fn update_service_account(
        &self,
        project_id: &str,
        id: &str,
        name: Option<&str>,
        role: Option<&str>,
        actor: &Actor,
    ) -> Result<ServiceAccount> {
        if let Some(name) = name {
            check_name(name, "name")?;
        }
        if let Some(role) = role
            && !matches!(role, "member" | "owner")
        {
            return Err(Error::InvalidArgument(
                "role must be \"member\" or \"owner\"".into(),
            ));
        }
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let project = active_project(&transaction, project_id)?;
        service_account_by_id(&transaction, project_id, id)?;
        if let Some(name) = name {
            transaction.execute(
                "UPDATE admin_service_accounts SET name = ?2 WHERE id = ?1",
                params![id, name],
            )?;
        }
        if let Some(role) = role {
            transaction.execute(
                "UPDATE admin_service_accounts SET role = ?2 WHERE id = ?1",
                params![id, role],
            )?;
        }
        let mut requested = serde_json::Map::new();
        if let Some(role) = role {
            requested.insert("role".into(), json!(role));
        }
        audit(
            &transaction,
            actor,
            "service_account.updated",
            Some(&project),
            id,
            &json!({"id":id,"changes_requested":requested}),
        )?;
        let account = service_account_by_id(&transaction, project_id, id)?;
        transaction.commit()?;
        Ok(account)
    }

    /// Delete a service account and, with it, every key it owns.
    pub fn delete_service_account(&self, project_id: &str, id: &str, actor: &Actor) -> Result<()> {
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let project = project_by_id(&transaction, project_id)?;
        service_account_by_id(&transaction, project_id, id)?;
        let keys: Vec<String> = transaction
            .prepare("SELECT id FROM admin_project_keys WHERE service_account_id = ?1")?
            .query_map([id], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        transaction.execute("DELETE FROM admin_service_accounts WHERE id = ?1", [id])?;
        for key in keys {
            audit(
                &transaction,
                actor,
                "api_key.deleted",
                Some(&project),
                &key,
                &json!({"id":key}),
            )?;
        }
        audit(
            &transaction,
            actor,
            "service_account.deleted",
            Some(&project),
            id,
            &json!({"id":id}),
        )?;
        transaction.commit()?;
        Ok(())
    }

    /// Create another key for service account `account_id`.
    pub fn create_service_account_key(
        &self,
        project_id: &str,
        account_id: &str,
        name: Option<&str>,
        expires_in: Option<i64>,
        actor: &Actor,
    ) -> Result<ServiceAccountApiKey> {
        let name = name.unwrap_or("Secret Key");
        check_name(name, "name")?;
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let project = active_project(&transaction, project_id)?;
        service_account_by_id(&transaction, project_id, account_id)?;
        let key = insert_project_key(&transaction, &project, account_id, name, expires_in, actor)?;
        transaction.commit()?;
        Ok(key)
    }

    pub fn list_project_keys(
        &self,
        project_id: &str,
        page: &Page,
        access: Option<&str>,
    ) -> Result<List<ProjectApiKey>> {
        let connection = self.store.connection()?;
        let project = project_by_id(&connection, project_id)?;
        let filter = match access {
            None => "project_id = ?",
            Some("active") if project.archived_at.is_none() => "project_id = ?",
            Some("inactive") if project.archived_at.is_some() => "project_id = ?",
            Some("active" | "inactive") => "project_id = ? AND 0",
            Some(_) => {
                return Err(Error::InvalidArgument(
                    "owner_project_access must be \"active\" or \"inactive\"".into(),
                ));
            }
        };
        list(
            &connection,
            "admin_project_keys",
            PROJECT_KEY_COLUMNS,
            filter,
            vec![SqlValue::Text(project_id.to_owned())],
            page,
            |row| project_key(&connection, row, &project),
        )
    }

    pub fn get_project_key(&self, project_id: &str, id: &str) -> Result<ProjectApiKey> {
        let connection = self.store.connection()?;
        let project = project_by_id(&connection, project_id)?;
        connection
            .query_row(
                &format!(
                    "SELECT {PROJECT_KEY_COLUMNS} FROM admin_project_keys
                     WHERE project_id = ?1 AND id = ?2"
                ),
                params![project_id, id],
                |row| project_key(&connection, row, &project),
            )
            .optional()?
            .ok_or_else(|| not_found("project API key", id))
    }

    pub fn delete_project_key(&self, project_id: &str, id: &str, actor: &Actor) -> Result<()> {
        let mut connection = self.store.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let project = project_by_id(&transaction, project_id)?;
        if transaction.execute(
            "DELETE FROM admin_project_keys WHERE project_id = ?1 AND id = ?2",
            params![project_id, id],
        )? == 0
        {
            return Err(not_found("project API key", id));
        }
        audit(
            &transaction,
            actor,
            "api_key.deleted",
            Some(&project),
            id,
            &json!({"id":id}),
        )?;
        transaction.commit()?;
        Ok(())
    }

    /// Audit log entries, newest first.
    pub fn list_audit_logs(&self, query: &AuditQuery) -> Result<List<AuditLog>> {
        let connection = self.store.connection()?;
        let limit = Page {
            limit: query.limit,
            ..Page::default()
        }
        .limit()?;
        let mut clauses = vec!["1".to_owned()];
        let mut values = Vec::new();
        for (bound, operator) in [
            (query.effective_gt, ">"),
            (query.effective_gte, ">="),
            (query.effective_lt, "<"),
            (query.effective_lte, "<="),
        ] {
            if let Some(bound) = bound {
                clauses.push(format!("effective_at {operator} ?"));
                values.push(SqlValue::Integer(bound));
            }
        }
        for (column, wanted) in [
            ("project_id", &query.project_ids),
            ("type", &query.event_types),
            ("json_extract(actor, '$.api_key.id')", &query.actor_ids),
            ("resource_id", &query.resource_ids),
        ] {
            in_clause(&mut clauses, &mut values, column, wanted);
        }
        if !query.actor_emails.is_empty() {
            // The local owner has no email address, so no entry matches.
            clauses.push("0".into());
        }
        let mut descending = true;
        if let Some(after) = &query.after {
            clauses.push("seq < ?".into());
            values.push(SqlValue::Integer(audit_seq(&connection, after)?));
        }
        if let Some(before) = &query.before {
            clauses.push("seq > ?".into());
            values.push(SqlValue::Integer(audit_seq(&connection, before)?));
            descending = query.after.is_some();
        }
        values.push(SqlValue::Integer(
            i64::try_from(limit + 1).unwrap_or(i64::MAX),
        ));
        let sql = format!(
            "SELECT id, type, effective_at, project_id, project_name, actor, details
             FROM admin_audit_logs WHERE {} ORDER BY seq {} LIMIT ?",
            clauses.join(" AND "),
            if descending { "DESC" } else { "ASC" }
        );
        let mut data = connection
            .prepare(&sql)?
            .query_map(rusqlite::params_from_iter(values), audit_log)?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let has_more = data.len() > limit;
        data.truncate(limit);
        if !descending {
            data.reverse();
        }
        Ok(listed(data, has_more))
    }
}

const ADMIN_KEY_COLUMNS: &str = "id, name, redacted, created_at, expires_at, last_used_at";
const PROJECT_COLUMNS: &str = "id, name, created_at, archived_at, external_key_id";
const SERVICE_ACCOUNT_COLUMNS: &str = "id, name, role, created_at";
const PROJECT_KEY_COLUMNS: &str =
    "id, name, redacted, created_at, expires_at, last_used_at, service_account_id";

fn not_found(what: &str, id: &str) -> Error {
    Error::NotFound(format!("no {what} with id {id:?}"))
}

fn owner(connection: &Connection) -> Result<Owner> {
    let setting = |name: &str| -> Result<String> {
        Ok(connection.query_row(
            "SELECT value FROM admin_settings WHERE name = ?1",
            [name],
            |row| row.get(0),
        )?)
    };
    Ok(Owner {
        id: setting("owner_id")?,
        created_at: setting("owner_created_at")?.parse().unwrap_or_default(),
    })
}

fn auth_enabled(connection: &Connection) -> Result<bool> {
    let value: String = connection.query_row(
        "SELECT value FROM admin_settings WHERE name = 'auth_enabled'",
        [],
        |row| row.get(0),
    )?;
    Ok(value == "1")
}

/// A key row: id, salt, tag, expiry, last use, project, account, archival.
type Found = (
    String,
    Vec<u8>,
    Vec<u8>,
    Option<i64>,
    Option<i64>,
    Option<String>,
    Option<String>,
    Option<i64>,
);

fn authenticate(connection: &Connection, token: Option<&str>) -> Result<Verdict> {
    if !auth_enabled(connection)? {
        return Ok(Verdict::Open);
    }
    let Some(token) = token else {
        return Ok(Verdict::Refused(
            "authentication is required: send an API key as `Authorization: Bearer <key>`",
        ));
    };
    let Some((kind, lookup)) = parse_token(token) else {
        return Ok(Verdict::Refused("invalid API key"));
    };
    let sql = match kind {
        KeyKind::Admin => {
            "SELECT id, salt, tag, expires_at, last_used_at, NULL, NULL, NULL
             FROM admin_api_keys WHERE lookup = ?1"
        }
        KeyKind::Project => {
            "SELECT k.id, k.salt, k.tag, k.expires_at, k.last_used_at, k.project_id,
                    k.service_account_id, p.archived_at
             FROM admin_project_keys k JOIN admin_projects p ON p.id = k.project_id
             WHERE k.lookup = ?1"
        }
    };
    let found: Option<Found> = connection
        .query_row(sql, [lookup], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
            ))
        })
        .optional()?;
    let Some((id, salt, stored_tag, expires_at, last_used, project, account, archived)) = found
    else {
        return Ok(Verdict::Refused("invalid API key"));
    };
    if !verify(&salt, &stored_tag, token) {
        return Ok(Verdict::Refused("invalid API key"));
    }
    let now = now_signed()?;
    if expires_at.is_some_and(|expires_at| expires_at <= now) {
        return Ok(Verdict::Refused("the API key has expired"));
    }
    if archived.is_some() {
        return Ok(Verdict::Refused("the API key's project is archived"));
    }
    if last_used.is_none_or(|last| now - last >= LAST_USED_GRANULARITY) {
        connection.execute(
            &format!(
                "UPDATE {} SET last_used_at = ?2 WHERE id = ?1",
                kind.table()
            ),
            params![id, now],
        )?;
    }
    Ok(Verdict::Granted(match (kind, project, account) {
        (KeyKind::Project, Some(project_id), Some(service_account_id)) => Credential::Project {
            key_id: id,
            project_id,
            service_account_id,
        },
        _ => Credential::Admin { key_id: id },
    }))
}

fn audit(
    connection: &Connection,
    actor: &Actor,
    kind: &str,
    project: Option<&Project>,
    resource_id: &str,
    details: &Value,
) -> Result<()> {
    let owner = owner(connection)?;
    let actor = match actor {
        Actor::Owner => json!({"type":"session","session":{"user":{"id":owner.id}}}),
        Actor::AdminKey(id) => {
            json!({"type":"api_key","api_key":{"id":id,"type":"user","user":{"id":owner.id}}})
        }
    };
    connection.execute(
        "INSERT INTO admin_audit_logs (id, type, effective_at, project_id, project_name,
             resource_id, actor, details) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            new_id("audit_log-")?,
            kind,
            now_signed()?,
            project.map(|project| project.id.as_str()),
            project.map(|project| project.name.as_str()),
            resource_id,
            actor.to_string(),
            details.to_string()
        ],
    )?;
    Ok(())
}

fn audit_seq(connection: &Connection, id: &str) -> Result<i64> {
    connection
        .query_row(
            "SELECT seq FROM admin_audit_logs WHERE id = ?1",
            [id],
            |row| row.get(0),
        )
        .optional()?
        .ok_or_else(|| not_found("audit log entry", id))
}

fn audit_log(row: &Row<'_>) -> rusqlite::Result<Result<AuditLog>> {
    let id: String = row.get(0)?;
    let kind: String = row.get(1)?;
    let effective_at: i64 = row.get(2)?;
    let project_id: Option<String> = row.get(3)?;
    let project_name: Option<String> = row.get(4)?;
    let actor: String = row.get(5)?;
    let details: String = row.get(6)?;
    Ok((|| {
        let mut value = json!({"id":id,"type":kind,"effective_at":effective_at,
                               "actor":serde_json::from_str::<Value>(&actor)?});
        if let Some(project_id) = project_id {
            value["project"] = json!({"id":project_id,"name":project_name});
        }
        value[kind.as_str()] = serde_json::from_str(&details)?;
        Ok(AuditLog { value, id })
    })())
}

fn in_clause(
    clauses: &mut Vec<String>,
    values: &mut Vec<SqlValue>,
    column: &str,
    wanted: &[String],
) {
    if wanted.is_empty() {
        return;
    }
    clauses.push(format!(
        "{column} IN ({})",
        vec!["?"; wanted.len()].join(", ")
    ));
    values.extend(wanted.iter().cloned().map(SqlValue::Text));
}

fn listed<T: Identified>(data: Vec<T>, has_more: bool) -> List<T> {
    List {
        object: "list",
        first_id: data.first().map(|item| item.id().to_owned()),
        last_id: data.last().map(|item| item.id().to_owned()),
        data,
        has_more,
    }
}

/// One page of `table` rows matching `filter`, in creation order.
fn list<T: Identified>(
    connection: &Connection,
    table: &str,
    columns: &str,
    filter: &str,
    mut values: Vec<SqlValue>,
    page: &Page,
    map: impl FnMut(&Row<'_>) -> rusqlite::Result<T>,
) -> Result<List<T>> {
    let limit = page.limit()?;
    let mut clauses = format!("({filter})");
    if let Some(after) = &page.after {
        let seq: i64 = connection
            .query_row(
                &format!("SELECT seq FROM {table} WHERE id = ?1"),
                [after],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("no object with id {after:?} to list after")))?;
        clauses.push_str(if page.descending {
            " AND seq < ?"
        } else {
            " AND seq > ?"
        });
        values.push(SqlValue::Integer(seq));
    }
    values.push(SqlValue::Integer(
        i64::try_from(limit + 1).unwrap_or(i64::MAX),
    ));
    let sql = format!(
        "SELECT {columns} FROM {table} WHERE {clauses} ORDER BY seq {} LIMIT ?",
        if page.descending { "DESC" } else { "ASC" }
    );
    let mut data = connection
        .prepare(&sql)?
        .query_map(rusqlite::params_from_iter(values), map)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let has_more = data.len() > limit;
    data.truncate(limit);
    Ok(listed(data, has_more))
}

fn admin_key(row: &Row<'_>, owner: &Owner) -> rusqlite::Result<AdminApiKey> {
    Ok(AdminApiKey {
        object: "organization.admin_api_key",
        id: row.get(0)?,
        name: row.get(1)?,
        redacted_value: row.get(2)?,
        created_at: row.get(3)?,
        expires_at: row.get(4)?,
        last_used_at: row.get(5)?,
        owner: owner.key_owner(),
    })
}

fn admin_key_by_id(connection: &Connection, id: &str) -> Result<AdminApiKey> {
    let owner = owner(connection)?;
    connection
        .query_row(
            &format!("SELECT {ADMIN_KEY_COLUMNS} FROM admin_api_keys WHERE id = ?1"),
            [id],
            |row| admin_key(row, &owner),
        )
        .optional()?
        .ok_or_else(|| not_found("admin API key", id))
}

fn project(row: &Row<'_>) -> rusqlite::Result<Project> {
    let archived_at: Option<i64> = row.get(3)?;
    Ok(Project {
        id: row.get(0)?,
        object: "organization.project",
        name: row.get(1)?,
        created_at: row.get(2)?,
        archived_at,
        status: if archived_at.is_some() {
            "archived"
        } else {
            "active"
        },
        external_key_id: row.get(4)?,
    })
}

fn project_by_id(connection: &Connection, id: &str) -> Result<Project> {
    connection
        .query_row(
            &format!("SELECT {PROJECT_COLUMNS} FROM admin_projects WHERE id = ?1"),
            [id],
            project,
        )
        .optional()?
        .ok_or_else(|| not_found("project", id))
}

fn active_project(connection: &Connection, id: &str) -> Result<Project> {
    let project = project_by_id(connection, id)?;
    if project.archived_at.is_some() {
        return Err(Error::Conflict(format!("project {id} is archived")));
    }
    Ok(project)
}

fn service_account(row: &Row<'_>) -> rusqlite::Result<ServiceAccount> {
    Ok(ServiceAccount {
        object: "organization.project.service_account",
        id: row.get(0)?,
        name: row.get(1)?,
        role: row.get(2)?,
        created_at: row.get(3)?,
    })
}

fn service_account_by_id(
    connection: &Connection,
    project_id: &str,
    id: &str,
) -> Result<ServiceAccount> {
    connection
        .query_row(
            &format!(
                "SELECT {SERVICE_ACCOUNT_COLUMNS} FROM admin_service_accounts
                 WHERE project_id = ?1 AND id = ?2"
            ),
            params![project_id, id],
            service_account,
        )
        .optional()?
        .ok_or_else(|| not_found("service account", id))
}

fn project_key(
    connection: &Connection,
    row: &Row<'_>,
    project: &Project,
) -> rusqlite::Result<ProjectApiKey> {
    let account_id: String = row.get(6)?;
    let account = connection
        .query_row(
            &format!("SELECT {SERVICE_ACCOUNT_COLUMNS} FROM admin_service_accounts WHERE id = ?1"),
            [&account_id],
            service_account,
        )
        .optional()?;
    let owner = account.map_or_else(
        || json!({"type":"service_account"}),
        |account| {
            json!({"type":"service_account","service_account":{"id":account.id,
                   "name":account.name,"created_at":account.created_at,"role":account.role}})
        },
    );
    Ok(ProjectApiKey {
        object: "organization.project.api_key",
        id: row.get(0)?,
        name: row.get(1)?,
        redacted_value: row.get(2)?,
        created_at: row.get(3)?,
        expires_at: row.get(4)?,
        last_used_at: row.get(5)?,
        owner_project_access: if project.archived_at.is_some() {
            "inactive"
        } else {
            "active"
        },
        owner,
    })
}

fn insert_project_key(
    connection: &Connection,
    project: &Project,
    account_id: &str,
    name: &str,
    expires_in: Option<i64>,
    actor: &Actor,
) -> Result<ServiceAccountApiKey> {
    let created_at = now_signed()?;
    let expires_at = expiry(created_at, expires_in)?;
    let minted = mint(KeyKind::Project)?;
    let id = new_id("key_")?;
    connection.execute(
        "INSERT INTO admin_project_keys (id, lookup, salt, tag, project_id, service_account_id,
             name, redacted, created_at, expires_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            id,
            minted.lookup,
            minted.salt.as_slice(),
            minted.tag,
            project.id,
            account_id,
            name,
            minted.redacted,
            created_at,
            expires_at
        ],
    )?;
    audit(
        connection,
        actor,
        "api_key.created",
        Some(project),
        &id,
        &json!({"id":id,"data":{"scopes":["api.all"]}}),
    )?;
    Ok(ServiceAccountApiKey {
        object: "organization.project.service_account.api_key",
        value: minted.token,
        name: name.to_owned(),
        created_at,
        expires_at,
        id,
    })
}
