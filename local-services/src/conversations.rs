//! Conversations: durable, `OpenAI`-compatible conversation objects and items.
//!
//! A conversation is an ordered, append-mostly list of items. Items are
//! validated and normalized on the way in, against exactly the forms local
//! generation implements (text messages, function calls and outputs, and
//! reasoning); anything else, such as images or files, is refused rather than
//! stored and silently ignored later.
//!
//! Deleting a conversation marks it deleted (its items are kept, as the public
//! contract specifies) and every later operation on it reports not found.
//! Every change to the item list bumps the conversation's
//! [`version`](Conversation::version), which [`Store::append_items`] uses for
//! optimistic concurrency, together with a per-request idempotency key.

use std::collections::{BTreeMap, HashSet};

use rusqlite::{Connection, ErrorCode, OptionalExtension as _, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Error, Result, Store, migrate, new_id, now};

/// Most metadata pairs a conversation may carry.
pub const MAX_METADATA_PAIRS: usize = 16;
/// Longest metadata key, in characters.
pub const MAX_METADATA_KEY_CHARS: usize = 64;
/// Longest metadata value, in characters.
pub const MAX_METADATA_VALUE_CHARS: usize = 512;
/// Most items one create or add call may insert.
pub const MAX_ADD_ITEMS: usize = 20;
/// Most items one [`Store::append_items`] call may insert.
pub const MAX_APPEND_ITEMS: usize = 256;
/// Largest serialized item.
pub const MAX_ITEM_BYTES: usize = 4 << 20;
/// Page size when [`ListItems::limit`] is absent.
pub const DEFAULT_LIST_LIMIT: u32 = 20;
/// Largest page size.
pub const MAX_LIST_LIMIT: u32 = 100;
/// Longest caller-supplied item or request ID.
pub const MAX_ID_CHARS: usize = 128;

/// Conversation metadata: string keys and values, ordered by key.
pub type Metadata = BTreeMap<String, String>;

/// A conversation, serialized as the public `conversation` object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "conversation")]
pub struct Conversation {
    pub id: String,
    /// Seconds since the Unix epoch.
    pub created_at: u64,
    pub metadata: Metadata,
    /// The history version: bumped by every item insertion or deletion.
    /// Internal, so it is not part of the serialized object.
    #[serde(skip)]
    pub version: u64,
}

/// The public `conversation.deleted` acknowledgement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "conversation.deleted")]
pub struct ConversationDeleted {
    pub id: String,
    pub deleted: bool,
}

/// A stored conversation item. Serializes as one flat object: `id` plus the
/// normalized `body` (`type`, `role`, `content`, `status`, ...).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    pub id: String,
    #[serde(flatten)]
    pub body: Map<String, Value>,
}

impl Item {
    /// The item's `type`: `message`, `function_call`, `function_call_output`
    /// or `reasoning`.
    #[must_use]
    pub fn kind(&self) -> &str {
        self.body
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }
}

/// A page of results, serialized as the public `list` object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "object", rename = "list")]
pub struct Page<T> {
    pub data: Vec<T>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
}

impl Page<Item> {
    fn of(data: Vec<Item>, has_more: bool) -> Self {
        Self {
            first_id: data.first().map(|item| item.id.clone()),
            last_id: data.last().map(|item| item.id.clone()),
            data,
            has_more,
        }
    }
}

/// Item list order; newest first by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Order {
    Asc,
    #[default]
    Desc,
}

/// Item list query.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ListItems {
    /// 1 to [`MAX_LIST_LIMIT`]; [`DEFAULT_LIST_LIMIT`] when absent.
    pub limit: Option<u32>,
    pub order: Order,
    /// Return only items after this item ID, in `order`.
    pub after: Option<String>,
}

/// An idempotent, version-checked append (see [`Store::append_items`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Append {
    /// Idempotency key, unique per conversation: 1 to [`MAX_ID_CHARS`] ASCII
    /// letters, digits, `_` or `-` (a response ID fits).
    pub request_id: String,
    /// The [`Conversation::version`] the caller generated against.
    pub expected_version: u64,
    /// 1 to [`MAX_APPEND_ITEMS`] items, in the same forms `add_items` takes.
    pub items: Vec<Value>,
}

/// The outcome of [`Store::append_items`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Appended {
    /// The conversation after the append (its current state on a replay).
    pub conversation: Conversation,
    /// The items this request appended, as they were stored.
    pub items: Vec<Item>,
    /// Whether this request ID had already been applied, so nothing changed.
    pub replayed: bool,
}

/// A consistent snapshot of a whole conversation, oldest item first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct History {
    pub conversation: Conversation,
    pub items: Vec<Item>,
}

/// Version 1: conversations, their items (the `AUTOINCREMENT` sequence is the
/// order and the cursor, never reused after a deletion), and the record of
/// applied append requests.
const MIGRATIONS: &[&str] = &["
CREATE TABLE conversations (
    id TEXT PRIMARY KEY NOT NULL,
    created_at INTEGER NOT NULL,
    metadata TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 0,
    deleted_at INTEGER
) STRICT;
CREATE TABLE conversation_items (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations (id),
    id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    body TEXT NOT NULL,
    UNIQUE (conversation_id, id)
) STRICT;
CREATE INDEX conversation_items_order ON conversation_items (conversation_id, seq);
CREATE TABLE conversation_appends (
    conversation_id TEXT NOT NULL REFERENCES conversations (id),
    request_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    items TEXT NOT NULL,
    PRIMARY KEY (conversation_id, request_id)
) STRICT, WITHOUT ROWID;
"];

/// Create or upgrade the conversation tables.
pub(crate) fn initialize(connection: &mut Connection) -> Result<()> {
    migrate(connection, "conversations", MIGRATIONS)
}

impl Store {
    /// Create a conversation with `metadata` and up to [`MAX_ADD_ITEMS`]
    /// initial items, atomically.
    pub fn create_conversation(
        &self,
        metadata: &Metadata,
        items: Vec<Value>,
    ) -> Result<Conversation> {
        validate_metadata(metadata)?;
        if items.len() > MAX_ADD_ITEMS {
            return Err(invalid(format!(
                "a conversation may be created with at most {MAX_ADD_ITEMS} items"
            )));
        }
        let items = normalize_items(items)?;
        let id = new_id("conv_")?;
        let created_at = now()?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT INTO conversations (id, created_at, metadata) VALUES (?1, ?2, ?3)",
            params![id, signed(created_at)?, serde_json::to_string(metadata)?],
        )?;
        insert_items(&transaction, &id, &items, created_at)?;
        let conversation = live(&transaction, &id)?;
        transaction.commit()?;
        Ok(conversation)
    }

    /// The live conversation `id`.
    pub fn get_conversation(&self, id: &str) -> Result<Conversation> {
        live(&self.connection()?, id)
    }

    /// Replace conversation `id`'s metadata.
    pub fn update_conversation(&self, id: &str, metadata: &Metadata) -> Result<Conversation> {
        validate_metadata(metadata)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        live(&transaction, id)?;
        transaction.execute(
            "UPDATE conversations SET metadata = ?2 WHERE id = ?1",
            params![id, serde_json::to_string(metadata)?],
        )?;
        let conversation = live(&transaction, id)?;
        transaction.commit()?;
        Ok(conversation)
    }

    /// Mark conversation `id` deleted. Its items are kept, but neither it nor
    /// they are reachable through this API any more.
    pub fn delete_conversation(&self, id: &str) -> Result<ConversationDeleted> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        live(&transaction, id)?;
        transaction.execute(
            "UPDATE conversations SET deleted_at = ?2 WHERE id = ?1",
            params![id, signed(now()?)?],
        )?;
        transaction.commit()?;
        Ok(ConversationDeleted {
            id: id.to_owned(),
            deleted: true,
        })
    }

    /// Append 1 to [`MAX_ADD_ITEMS`] items to conversation `id`, atomically:
    /// one invalid item and none are stored. Returns the stored items.
    pub fn add_items(&self, id: &str, items: Vec<Value>) -> Result<Page<Item>> {
        if items.is_empty() || items.len() > MAX_ADD_ITEMS {
            return Err(invalid(format!(
                "between 1 and {MAX_ADD_ITEMS} items may be added at once"
            )));
        }
        let items = normalize_items(items)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        live(&transaction, id)?;
        insert_items(&transaction, id, &items, now()?)?;
        transaction.commit()?;
        Ok(Page::of(items, false))
    }

    /// A page of conversation `id`'s items.
    pub fn list_items(&self, id: &str, query: &ListItems) -> Result<Page<Item>> {
        let limit = query.limit.unwrap_or(DEFAULT_LIST_LIMIT);
        if !(1..=MAX_LIST_LIMIT).contains(&limit) {
            return Err(invalid(format!(
                "limit must be between 1 and {MAX_LIST_LIMIT}"
            )));
        }
        let mut connection = self.connection()?;
        // A read transaction: the cursor lookup and the page see one snapshot.
        let transaction = connection.transaction()?;
        live(&transaction, id)?;
        let after = match &query.after {
            None => None,
            Some(after) => Some(
                transaction
                    .query_row(
                        "SELECT seq FROM conversation_items
                         WHERE conversation_id = ?1 AND id = ?2",
                        params![id, after],
                        |row| row.get::<_, i64>(0),
                    )
                    .optional()?
                    .ok_or_else(|| {
                        invalid(format!(
                            "after {after:?} is not an item of conversation {id}"
                        ))
                    })?,
            ),
        };
        let sql = match query.order {
            Order::Asc => {
                "SELECT id, body FROM conversation_items
                 WHERE conversation_id = ?1 AND seq > ?2 ORDER BY seq ASC LIMIT ?3"
            }
            Order::Desc => {
                "SELECT id, body FROM conversation_items
                 WHERE conversation_id = ?1 AND seq < ?2 ORDER BY seq DESC LIMIT ?3"
            }
        };
        let bound = after.unwrap_or(match query.order {
            Order::Asc => 0,
            Order::Desc => i64::MAX,
        });
        let mut items = select_items(&transaction, sql, params![id, bound, i64::from(limit) + 1])?;
        let has_more = items.len() > limit as usize;
        items.truncate(limit as usize);
        Ok(Page::of(items, has_more))
    }

    /// Item `item_id` of conversation `id`.
    pub fn get_item(&self, id: &str, item_id: &str) -> Result<Item> {
        let connection = self.connection()?;
        live(&connection, id)?;
        select_items(
            &connection,
            "SELECT id, body FROM conversation_items WHERE conversation_id = ?1 AND id = ?2",
            params![id, item_id],
        )?
        .pop()
        .ok_or_else(|| item_not_found(id, item_id))
    }

    /// Delete item `item_id` of conversation `id`, returning the conversation.
    pub fn delete_item(&self, id: &str, item_id: &str) -> Result<Conversation> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        live(&transaction, id)?;
        let deleted = transaction.execute(
            "DELETE FROM conversation_items WHERE conversation_id = ?1 AND id = ?2",
            params![id, item_id],
        )?;
        if deleted == 0 {
            return Err(item_not_found(id, item_id));
        }
        bump_version(&transaction, id)?;
        let conversation = live(&transaction, id)?;
        transaction.commit()?;
        Ok(conversation)
    }

    /// The whole of conversation `id`, oldest item first, with the version a
    /// later [`append_items`](Self::append_items) should expect.
    pub fn conversation_history(&self, id: &str) -> Result<History> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        let conversation = live(&transaction, id)?;
        let items = select_items(
            &transaction,
            "SELECT id, body FROM conversation_items WHERE conversation_id = ?1 ORDER BY seq ASC",
            params![id],
        )?;
        Ok(History {
            conversation,
            items,
        })
    }

    /// Append generated history exactly once.
    ///
    /// Applied at most once per `request_id`: a retry with the same request ID
    /// and the same items returns the originally stored items with
    /// `replayed: true`, whatever the conversation's version is now; the same
    /// request ID with different items is a [`Error::Conflict`]. A new request
    /// is applied only if the conversation is still at `expected_version`,
    /// so generation that raced another writer cannot append stale history:
    /// it gets [`Error::Conflict`] and nothing is stored.
    pub fn append_items(&self, id: &str, append: Append) -> Result<Appended> {
        let Append {
            request_id,
            expected_version,
            items,
        } = append;
        validate_token("request id", &request_id)?;
        if items.is_empty() || items.len() > MAX_APPEND_ITEMS {
            return Err(invalid(format!(
                "between 1 and {MAX_APPEND_ITEMS} items may be appended at once"
            )));
        }
        let fingerprint = fingerprint(&items)?;
        let items = normalize_items(items)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let conversation = live(&transaction, id)?;
        let applied: Option<(String, String)> = transaction
            .query_row(
                "SELECT fingerprint, items FROM conversation_appends
                 WHERE conversation_id = ?1 AND request_id = ?2",
                params![id, request_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((stored, recorded)) = applied {
            if stored != fingerprint {
                return Err(Error::Conflict(format!(
                    "request {request_id} was already applied to conversation {id} with different items"
                )));
            }
            return Ok(Appended {
                conversation,
                items: serde_json::from_str(&recorded)?,
                replayed: true,
            });
        }
        if conversation.version != expected_version {
            return Err(Error::Conflict(format!(
                "conversation {id} is at version {}, not the expected {expected_version}",
                conversation.version
            )));
        }
        insert_items(&transaction, id, &items, now()?)?;
        transaction.execute(
            "INSERT INTO conversation_appends (conversation_id, request_id, fingerprint, items)
             VALUES (?1, ?2, ?3, ?4)",
            params![id, request_id, fingerprint, serde_json::to_string(&items)?],
        )?;
        let conversation = live(&transaction, id)?;
        transaction.commit()?;
        Ok(Appended {
            conversation,
            items,
            replayed: false,
        })
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidArgument(message.into())
}

fn item_not_found(id: &str, item_id: &str) -> Error {
    Error::NotFound(format!("item {item_id} not found in conversation {id}"))
}

fn corrupt(what: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("stored {what} is out of range"),
    ))
}

fn signed(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| corrupt("integer"))
}

fn unsigned(value: i64, what: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| corrupt(what))
}

/// The conversation `id`, unless it does not exist or was deleted.
fn live(connection: &Connection, id: &str) -> Result<Conversation> {
    let row: Option<(i64, String, i64)> = connection
        .query_row(
            "SELECT created_at, metadata, version FROM conversations
             WHERE id = ?1 AND deleted_at IS NULL",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let (created_at, metadata, version) =
        row.ok_or_else(|| Error::NotFound(format!("conversation {id} not found")))?;
    Ok(Conversation {
        id: id.to_owned(),
        created_at: unsigned(created_at, "creation time")?,
        metadata: serde_json::from_str(&metadata)?,
        version: unsigned(version, "version")?,
    })
}

fn bump_version(connection: &Connection, id: &str) -> Result<()> {
    connection.execute(
        "UPDATE conversations SET version = version + 1 WHERE id = ?1",
        [id],
    )?;
    Ok(())
}

/// Insert `items`, in order, and bump the version once. Nothing for none.
fn insert_items(connection: &Connection, id: &str, items: &[Item], created_at: u64) -> Result<()> {
    if items.is_empty() {
        return Ok(());
    }
    let created_at = signed(created_at)?;
    let mut statement = connection.prepare(
        "INSERT INTO conversation_items (conversation_id, id, created_at, body)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    for item in items {
        let body = serde_json::to_string(&item.body)?;
        match statement.execute(params![id, item.id, created_at, body]) {
            Ok(_) => {}
            Err(rusqlite::Error::SqliteFailure(failure, _))
                if failure.code == ErrorCode::ConstraintViolation =>
            {
                return Err(Error::Conflict(format!(
                    "item {} already exists in conversation {id}",
                    item.id
                )));
            }
            Err(error) => return Err(error.into()),
        }
    }
    bump_version(connection, id)
}

fn select_items(
    connection: &Connection,
    sql: &str,
    parameters: impl rusqlite::Params,
) -> Result<Vec<Item>> {
    let mut statement = connection.prepare(sql)?;
    let rows = statement.query_map(parameters, |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut items = Vec::new();
    for row in rows {
        let (id, body) = row?;
        items.push(Item {
            id,
            body: serde_json::from_str(&body)?,
        });
    }
    Ok(items)
}

/// SHA-256 of the request's items as sent, before IDs are generated.
fn fingerprint(items: &[Value]) -> Result<String> {
    use std::fmt::Write as _;
    let digest = ring::digest::digest(&ring::digest::SHA256, &serde_json::to_vec(items)?);
    let mut hex = String::with_capacity(64);
    for byte in digest.as_ref() {
        write!(hex, "{byte:02x}").map_err(std::io::Error::other)?;
    }
    Ok(hex)
}

fn validate_metadata(metadata: &Metadata) -> Result<()> {
    if metadata.len() > MAX_METADATA_PAIRS {
        return Err(invalid(format!(
            "metadata may hold at most {MAX_METADATA_PAIRS} pairs"
        )));
    }
    for (key, value) in metadata {
        if key.chars().count() > MAX_METADATA_KEY_CHARS {
            return Err(invalid(format!(
                "metadata keys may be at most {MAX_METADATA_KEY_CHARS} characters"
            )));
        }
        if value.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(invalid(format!(
                "metadata values may be at most {MAX_METADATA_VALUE_CHARS} characters"
            )));
        }
    }
    Ok(())
}

/// 1 to [`MAX_ID_CHARS`] ASCII letters, digits, `_` or `-`.
fn validate_token(what: &str, token: &str) -> Result<()> {
    let valid = !token.is_empty()
        && token.len() <= MAX_ID_CHARS
        && token
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(invalid(format!(
            "{what} must be 1 to {MAX_ID_CHARS} ASCII letters, digits, '_' or '-'"
        )))
    }
}

/// Validate and normalize a batch; IDs must be unique within it.
fn normalize_items(items: Vec<Value>) -> Result<Vec<Item>> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        let item = normalize_item(item).map_err(|error| match error {
            Error::InvalidArgument(message) => invalid(format!("items[{index}]: {message}")),
            other => other,
        })?;
        if !seen.insert(item.id.clone()) {
            return Err(invalid(format!(
                "items[{index}]: duplicate item id {}",
                item.id
            )));
        }
        normalized.push(item);
    }
    Ok(normalized)
}

fn normalize_item(item: Value) -> Result<Item> {
    let Value::Object(mut body) = item else {
        return Err(invalid("an item must be an object"));
    };
    let kind = match body.remove("type") {
        None => "message".to_owned(),
        Some(Value::String(kind)) => kind,
        Some(_) => return Err(invalid("type must be a string")),
    };
    let id = match body.remove("id") {
        None | Some(Value::Null) => None,
        Some(Value::String(id)) => {
            validate_token("item id", &id)?;
            Some(id)
        }
        Some(_) => return Err(invalid("id must be a string")),
    };
    let prefix = match kind.as_str() {
        "message" => {
            normalize_message(&mut body)?;
            "msg_"
        }
        "function_call" => {
            only_keys(&body, &["call_id", "name", "arguments", "status"])?;
            required_string(&body, "call_id", false)?;
            required_string(&body, "name", false)?;
            required_string(&body, "arguments", true)?;
            default_status(&mut body)?;
            "fc_"
        }
        "function_call_output" => {
            only_keys(&body, &["call_id", "output", "status"])?;
            required_string(&body, "call_id", false)?;
            match body.get_mut("output") {
                Some(Value::String(_)) => {}
                Some(Value::Array(parts)) => {
                    text_parts(parts, &["input_text"], "input_text", false)?;
                }
                _ => {
                    return Err(invalid(
                        "output must be a string or an array of input_text parts",
                    ));
                }
            }
            default_status(&mut body)?;
            "fco_"
        }
        "reasoning" => {
            normalize_reasoning(&mut body, id.is_some())?;
            "rs_"
        }
        other => {
            return Err(invalid(format!(
                "item type {other:?} is not supported; only message, function_call, \
                 function_call_output and reasoning items are"
            )));
        }
    };
    body.insert("type".to_owned(), Value::String(kind));
    let id = id.map_or_else(|| new_id(prefix), Ok)?;
    if serde_json::to_vec(&body)?.len() > MAX_ITEM_BYTES {
        return Err(invalid(format!(
            "an item may be at most {MAX_ITEM_BYTES} bytes"
        )));
    }
    Ok(Item { id, body })
}

/// `user`, `system` and `developer` messages hold `input_text` parts;
/// `assistant` messages hold `output_text` parts with annotations.
/// String content becomes a single part. Every returned message has a status.
fn normalize_message(body: &mut Map<String, Value>) -> Result<()> {
    only_keys(body, &["role", "content", "status"])?;
    let assistant = match body.get("role").and_then(Value::as_str) {
        Some("assistant") => true,
        Some("user" | "system" | "developer") => false,
        _ => {
            return Err(invalid(
                "a message role must be user, assistant, system or developer",
            ));
        }
    };
    let part = if assistant {
        "output_text"
    } else {
        "input_text"
    };
    let content = body
        .get_mut("content")
        .ok_or_else(|| invalid("a message requires content"))?;
    match content {
        Value::String(text) => {
            let mut single = Map::new();
            single.insert("type".to_owned(), Value::String(part.to_owned()));
            single.insert("text".to_owned(), Value::String(std::mem::take(text)));
            if assistant {
                single.insert("annotations".to_owned(), Value::Array(Vec::new()));
                single.insert("logprobs".to_owned(), Value::Array(Vec::new()));
            }
            *content = Value::Array(vec![Value::Object(single)]);
        }
        Value::Array(parts) if assistant => {
            text_parts(parts, &["output_text", "input_text"], part, true)?;
        }
        Value::Array(parts) => text_parts(parts, &["input_text"], part, false)?,
        _ => return Err(invalid("message content must be a string or an array")),
    }
    default_status(body)
}

fn normalize_reasoning(body: &mut Map<String, Value>, has_id: bool) -> Result<()> {
    only_keys(body, &["summary", "content", "encrypted_content", "status"])?;
    match body
        .entry("summary")
        .or_insert_with(|| Value::Array(Vec::new()))
    {
        Value::Array(parts) => text_parts(parts, &["summary_text"], "summary_text", false)?,
        _ => return Err(invalid("reasoning summary must be an array")),
    }
    if body.get("content").is_some_and(Value::is_null) {
        body.remove("content");
    }
    match body.get_mut("content") {
        None => {}
        Some(Value::Array(parts)) => {
            text_parts(parts, &["reasoning_text"], "reasoning_text", false)?;
        }
        Some(_) => return Err(invalid("reasoning content must be an array")),
    }
    match body.get("encrypted_content") {
        Some(Value::String(_)) if !has_id => {
            return Err(invalid("encrypted reasoning requires its original item id"));
        }
        None | Some(Value::String(_)) => {}
        Some(_) => return Err(invalid("encrypted_content must be a string")),
    }
    check_status(body)
}

/// Check text parts: each an object of an `accepted` type with a string
/// `text`, rewritten to `canonical`. Output text has `annotations` and
/// `logprobs` arrays, defaulted to empty.
fn text_parts(
    parts: &mut [Value],
    accepted: &[&str],
    canonical: &str,
    annotations: bool,
) -> Result<()> {
    for part in parts {
        let Value::Object(part) = part else {
            return Err(invalid("a content part must be an object"));
        };
        let allowed: &[&str] = if annotations {
            &["type", "text", "annotations", "logprobs"]
        } else {
            &["type", "text"]
        };
        only_keys(part, allowed)?;
        match part.get("type").and_then(Value::as_str) {
            Some(kind) if accepted.contains(&kind) => {}
            kind => {
                return Err(invalid(format!(
                    "content part type {} is not supported here; expected {}",
                    kind.unwrap_or("(missing)"),
                    accepted.join(" or ")
                )));
            }
        }
        required_string(part, "text", true)?;
        part.insert("type".to_owned(), Value::String(canonical.to_owned()));
        if annotations {
            for key in ["annotations", "logprobs"] {
                if !part
                    .entry(key)
                    .or_insert_with(|| Value::Array(Vec::new()))
                    .is_array()
                {
                    return Err(invalid(format!("{key} must be an array")));
                }
            }
        }
    }
    Ok(())
}

fn only_keys(object: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    object
        .keys()
        .find(|key| !allowed.contains(&key.as_str()))
        .map_or(Ok(()), |key| {
            Err(invalid(format!("field {key:?} is not supported here")))
        })
}

fn required_string(object: &Map<String, Value>, key: &str, empty: bool) -> Result<()> {
    match object.get(key) {
        Some(Value::String(value)) if empty || !value.is_empty() => Ok(()),
        _ if empty => Err(invalid(format!("{key} must be a string"))),
        _ => Err(invalid(format!("{key} must be a non-empty string"))),
    }
}

fn check_status(object: &Map<String, Value>) -> Result<()> {
    let valid = object.get("status").is_none_or(|status| {
        matches!(
            status.as_str(),
            Some("in_progress" | "completed" | "incomplete")
        )
    });
    if valid {
        Ok(())
    } else {
        Err(invalid(
            "status must be in_progress, completed or incomplete",
        ))
    }
}

fn default_status(object: &mut Map<String, Value>) -> Result<()> {
    check_status(object)?;
    object
        .entry("status")
        .or_insert_with(|| Value::String("completed".to_owned()));
    Ok(())
}
