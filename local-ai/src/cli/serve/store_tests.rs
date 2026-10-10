//! CPU-only tests for the opt-in Responses store: persistence, replay through
//! `previous_response_id`, deletion and the retrieval endpoints' request
//! semantics, driven by synthetic engine events so no model or GPU is touched.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::http::Method;
use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{Event, GenerationStats, Stats, ToolCall};

use super::options::parse;
use super::response::{Protocol, Reply, new_id};
use super::responses::{PreparedResponses, ResponsesState, prepare_responses_with};
use super::store::{ItemPage, PageError, ResponseStore, query_pairs, valid_id};
use super::{stored_answer, stored_request};
use crate::GenerateParams;

/// A store directory under the system temp dir, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        Self(std::env::temp_dir().join(new_id("local-ai-store-test-")))
    }

    fn open(&self) -> Arc<ResponseStore> {
        Arc::new(ResponseStore::open(&self.0).expect("store opens"))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn stats(stop_reason: StopReason) -> Box<Stats> {
    Box::new(Stats {
        stop_reason,
        cache_source: PromptCacheSource::None,
        reasoning_tokens: 2,
        generation: GenerationStats {
            prompt_tokens: 11,
            generated_tokens: 5,
            ..GenerationStats::default()
        },
    })
}

fn prepare(body: &Value, store: Option<&Arc<ResponseStore>>) -> crate::Result<PreparedResponses> {
    prepare_responses_with(body.to_string().as_bytes(), true, store, None, "m")
}

fn error(body: &Value, store: Option<&Arc<ResponseStore>>) -> String {
    prepare(body, store)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default()
}

/// Run `events` through a Response for `prepared`, as the server would, and
/// return the terminal Response and every streamed event type.
fn run(prepared: PreparedResponses, events: Vec<Event>) -> (Value, Vec<String>) {
    let reply = Reply::new(Protocol::Responses(Arc::new(prepared.echo)), "m".into());
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    let mut state = ResponsesState::new(&reply, Arc::clone(echo), true);
    state.start();
    let mut terminal = Value::Null;
    for event in events {
        if let Some(done) = state.event(event) {
            terminal = done;
        }
    }
    let kinds = state
        .take_events()
        .iter()
        .filter_map(|event| event["type"].as_str().map(str::to_owned))
        .collect();
    (terminal, kinds)
}

fn read_call() -> ToolCall {
    ToolCall {
        id: "call_1".into(),
        name: "read".into(),
        arguments: json!({"path":"main.rs"}),
    }
}

fn first_turn() -> Vec<Event> {
    vec![
        Event::Reasoning("Read the file first.".into()),
        Event::ToolCall(read_call()),
        Event::Finished(stats(StopReason::Eos)),
    ]
}

fn ids(items: &[Value]) -> Vec<&str> {
    items
        .iter()
        .filter_map(|item| item["id"].as_str())
        .collect()
}

#[test]
fn store_flag_is_opt_in() {
    let to_vec = |values: &[&str]| values.iter().map(|v| (*v).to_owned()).collect::<Vec<_>>();
    assert_eq!(parse(&to_vec(&[])).expect("defaults").response_store, None);
    let parsed = parse(&to_vec(&["--response-store", "/tmp/r"])).expect("flag");
    assert_eq!(parsed.response_store, Some(PathBuf::from("/tmp/r")));
    assert!(parse(&to_vec(&["--response-store"])).is_err());
    assert!(parse(&to_vec(&["--response-store", ""])).is_err());
}

#[test]
fn without_a_store_responses_stay_stateless() {
    let (response, _) = run(
        prepare(&json!({"input":"hi"}), None).expect("valid"),
        vec![Event::Finished(stats(StopReason::Eos))],
    );
    assert_eq!(response["store"], false);
    assert_eq!(response["previous_response_id"], Value::Null);
    let message = error(&json!({"input":"hi","store":true}), None);
    assert!(message.contains("--response-store"), "{message}");
    let message = error(
        &json!({"input":"hi","previous_response_id":"resp_abc"}),
        None,
    );
    assert!(message.contains("--response-store"), "{message}");
}

#[test]
fn a_completed_response_is_stored_before_it_is_reported_and_survives_a_restart() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let body = json!({"input":"Read main.rs","instructions":"sys one",
        "tools":[{"type":"function","name":"read"}]});
    let (response, kinds) = run(prepare(&body, Some(&store)).expect("valid"), first_turn());
    assert_eq!(response["status"], "completed");
    assert_eq!(
        response["store"], true,
        "store defaults to true once configured"
    );
    assert_eq!(kinds.last().map(String::as_str), Some("response.completed"));
    let id = response["id"].as_str().expect("id");

    // A fresh store over the same directory is what a restarted server opens.
    let reopened = ResponseStore::open(&scratch.0).expect("reopens");
    let stored = reopened.load(id).expect("readable").expect("stored");
    assert_eq!(stored.response, response);
    assert_eq!(stored.input_items.len(), 1);
    assert_eq!(
        stored.input_items[0]["content"],
        json!([{"type":"input_text","text":"Read main.rs"}])
    );
    assert_eq!(stored.input_items[0]["role"], "user");
    assert!(
        stored.input_items[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_"))
    );
    // Raw reasoning is kept as written; nothing is presented as encrypted.
    assert_eq!(
        stored.response["output"][0]["content"][0]["text"],
        "Read the file first."
    );
    assert!(
        !std::fs::read_to_string(scratch.0.join(format!("{id}.json")))
            .expect("record")
            .contains("encrypted")
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = |path: &Path| std::fs::metadata(path).expect("meta").permissions().mode();
        assert_eq!(mode(&scratch.0.join(format!("{id}.json"))) & 0o777, 0o600);
        assert_eq!(mode(&scratch.0) & 0o777, 0o700);
    }
}

#[test]
fn previous_response_id_replays_items_but_not_request_settings() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let first = json!({"input":"Read main.rs","instructions":"sys one","temperature":0.2,
        "tools":[{"type":"function","name":"read"}]});
    let (response, _) = run(prepare(&first, Some(&store)).expect("valid"), first_turn());
    let id = response["id"].as_str().expect("id").to_owned();

    // After a restart, only the tool result goes back.
    let store = scratch.open();
    let second = json!({"previous_response_id":id,
        "input":[{"type":"function_call_output","call_id":"call_1","output":"fn main() {}"}]});
    let prepared = prepare(&second, Some(&store)).expect("replayed");
    let request = &prepared.request;
    let roles: Vec<&str> = request.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool"], "no inherited system");
    assert_eq!(
        request.messages[1].reasoning_content.as_deref(),
        Some("Read the file first.")
    );
    assert_eq!(request.messages[1].tool_calls, vec![read_call()]);
    assert_eq!(request.messages[2].tool_call_id.as_deref(), Some("call_1"));
    assert!(request.tools.is_empty(), "tools are not inherited");
    assert!(
        (request.sampling.0.temperature - GenerateParams::default().temperature).abs()
            < f32::EPSILON,
        "sampling is not inherited"
    );

    let (follow, _) = run(
        prepared,
        vec![
            Event::Content("It prints nothing.".into()),
            Event::Finished(stats(StopReason::Eos)),
        ],
    );
    assert_eq!(follow["previous_response_id"], id.as_str());
    assert_eq!(follow["instructions"], Value::Null);
    let stored = store
        .load(follow["id"].as_str().expect("id"))
        .expect("readable")
        .expect("stored");
    // The whole resolved history is in the new record, with unique IDs, so
    // deleting the first response does not break continuing the second.
    let kinds: Vec<&str> = stored
        .input_items
        .iter()
        .filter_map(|item| item["type"].as_str())
        .collect();
    assert_eq!(
        kinds,
        [
            "message",
            "reasoning",
            "function_call",
            "function_call_output"
        ]
    );
    let all = ids(&stored.input_items);
    assert_eq!(all.len(), 4);
    assert!(all.iter().all(|id| !id.is_empty()));
    assert_eq!(stored.input_items[3]["status"], "completed");
    assert!(store.delete(&id).expect("deletes"));
    let third = json!({"previous_response_id":follow["id"].clone(),"input":"thanks"});
    let roles: Vec<String> = prepare(&third, Some(&store))
        .expect("independent of the deleted ancestor")
        .request
        .messages
        .iter()
        .map(|m| m.role.clone())
        .collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant", "user"]);
}

#[test]
fn store_false_is_respected_and_missing_or_deleted_ids_fail() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let (kept, _) = run(
        prepare(&json!({"input":"hi"}), Some(&store)).expect("valid"),
        vec![Event::Finished(stats(StopReason::Eos))],
    );
    let kept_id = kept["id"].as_str().expect("id").to_owned();

    let unstored = json!({"input":"again","store":false,"previous_response_id":kept_id});
    let (response, _) = run(
        prepare(&unstored, Some(&store)).expect("store=false may still continue"),
        vec![Event::Finished(stats(StopReason::Eos))],
    );
    assert_eq!(response["store"], false);
    assert_eq!(response["previous_response_id"], kept_id.as_str());
    assert!(
        store
            .load(response["id"].as_str().expect("id"))
            .expect("readable")
            .is_none()
    );

    assert!(store.delete(&kept_id).expect("deletes"));
    assert!(!store.delete(&kept_id).expect("second delete"));
    for missing in [kept_id.as_str(), "resp_0000", "resp_../../etc/passwd", "x"] {
        let message = error(
            &json!({"input":"hi","previous_response_id":missing}),
            Some(&store),
        );
        assert!(message.contains("not found"), "{missing}: {message}");
    }
}

#[test]
fn incomplete_responses_are_stored_and_failures_are_not() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let (incomplete, _) = run(
        prepare(&json!({"input":"hi"}), Some(&store)).expect("valid"),
        vec![
            Event::Content("cut".into()),
            Event::Finished(stats(StopReason::TokenLimit)),
        ],
    );
    assert_eq!(incomplete["status"], "incomplete");
    assert_eq!(incomplete["store"], true);
    assert!(
        store
            .load(incomplete["id"].as_str().expect("id"))
            .expect("readable")
            .is_some()
    );
    for terminal in [
        Event::Finished(stats(StopReason::Cancelled)),
        Event::Error("metal said no".into()),
    ] {
        let (failed, _) = run(
            prepare(&json!({"input":"hi"}), Some(&store)).expect("valid"),
            vec![Event::Content("partial".into()), terminal],
        );
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["store"], false);
        assert!(
            store
                .load(failed["id"].as_str().expect("id"))
                .expect("readable")
                .is_none()
        );
    }
}

#[test]
fn a_response_that_cannot_be_stored_fails_instead_of_claiming_storage() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let prepared = prepare(&json!({"input":"hi"}), Some(&store)).expect("valid");
    std::fs::remove_dir_all(&scratch.0).expect("store directory removed");
    let (response, kinds) = run(
        prepared,
        vec![
            Event::Content("answer".into()),
            Event::Finished(stats(StopReason::Eos)),
        ],
    );
    assert_eq!(response["status"], "failed");
    assert_eq!(response["store"], false);
    assert!(
        response["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("could not be stored"))
    );
    assert!(!kinds.iter().any(|kind| kind == "response.completed"));
    assert_eq!(kinds.last().map(String::as_str), Some("response.failed"));
}

#[test]
fn ids_that_could_leave_the_store_never_reach_the_filesystem() {
    for valid in ["resp_abc123", "resp_0123456789abcdef0123456789abcdef"] {
        assert!(valid_id(valid), "{valid}");
    }
    let long = format!("resp_{}", "a".repeat(65));
    for invalid in [
        "",
        "resp_",
        "abc",
        "resp_..",
        "resp_../x",
        "resp_a/b",
        "resp_a\\b",
        "resp_a%2Fb",
        "resp_a\0",
        "resp_a.json",
        long.as_str(),
    ] {
        assert!(!valid_id(invalid), "{invalid:?}");
    }
    let scratch = Scratch::new();
    let store = scratch.open();
    // A file beside the store, named as a traversal would reach it.
    let outside = scratch.0.with_extension("outside.json");
    std::fs::write(&outside, b"{}").expect("planted");
    let name = outside
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("name")
        .to_owned();
    for traversal in [format!("../{name}"), format!("resp_/../../{name}")] {
        assert!(store.load(&traversal).expect("no error").is_none());
        assert!(!store.delete(&traversal).expect("no error"));
    }
    assert!(outside.exists());
    let _ = std::fs::remove_file(outside);
}

#[test]
fn an_interrupted_write_leaves_nothing_a_reader_sees() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let partial = scratch.0.join(".resp_abc.123.tmp");
    std::fs::write(&partial, b"{\"trunc").expect("partial");
    assert!(store.load("resp_abc").expect("readable").is_none());
    let _reopened = scratch.open();
    assert!(
        partial.exists(),
        "another process could still be writing it"
    );
    assert!(store.load("resp_abc").expect("readable").is_none());
}

#[test]
fn retrieval_and_deletion_answer_with_the_documented_shapes() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let (response, _) = run(
        prepare(&json!({"input":"hi"}), Some(&store)).expect("valid"),
        vec![Event::Finished(stats(StopReason::Eos))],
    );
    let id = response["id"].as_str().expect("id");
    let page = stored_request(&Method::GET, false, None).expect("plain get");
    assert_eq!(
        stored_answer(&store, &Method::GET, id, page.as_ref()).expect("ok"),
        Some(response.clone())
    );
    let page = stored_request(&Method::GET, true, None).expect("items");
    let list = stored_answer(&store, &Method::GET, id, page.as_ref())
        .expect("ok")
        .expect("found");
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"][0]["role"], "user");
    assert_eq!(list["has_more"], false);
    assert_eq!(
        stored_answer(&store, &Method::DELETE, id, None).expect("ok"),
        Some(json!({"id":id,"object":"response.deleted","deleted":true}))
    );
    for method in [Method::GET, Method::DELETE] {
        assert_eq!(
            stored_answer(&store, &method, id, None).expect("ok"),
            None,
            "deleted is 404"
        );
    }
    assert_eq!(
        stored_answer(&store, &Method::GET, "resp_../x", None).expect("ok"),
        None
    );
    for query in [
        "stream=true",
        "starting_after=3",
        "include=reasoning.encrypted_content",
    ] {
        assert!(
            stored_request(&Method::GET, false, Some(query)).is_err(),
            "{query}"
        );
    }
    assert!(stored_request(&Method::GET, false, Some("stream=false&x=1")).is_ok());
    assert!(stored_request(&Method::DELETE, false, Some("stream=true")).is_ok());
}

#[test]
fn input_items_paginate_newest_first_by_default() {
    let items: Vec<Value> = (0..5).map(|n| json!({"id":format!("msg_{n}")})).collect();
    let page = |query: &str| ItemPage::parse(Some(query)).expect("valid query");
    let list = |query: &str| page(query).list(&items).expect("page");

    let all = ItemPage::parse(None).expect("defaults");
    assert_eq!(
        all,
        ItemPage {
            limit: 20,
            descending: true,
            after: None
        }
    );
    let newest = list("limit=2");
    assert_eq!(newest["data"], json!([{"id":"msg_4"},{"id":"msg_3"}]));
    assert_eq!(newest["first_id"], "msg_4");
    assert_eq!(newest["last_id"], "msg_3");
    assert_eq!(newest["has_more"], true);
    let next = list("limit=2&after=msg_3");
    assert_eq!(next["data"], json!([{"id":"msg_2"},{"id":"msg_1"}]));
    let oldest = list("limit=2&after=msg_1");
    assert_eq!(oldest["data"], json!([{"id":"msg_0"}]));
    assert_eq!(oldest["has_more"], false);
    let ascending = list("order=asc&limit=3&after=msg_0");
    assert_eq!(
        ascending["data"],
        json!([{"id":"msg_1"},{"id":"msg_2"},{"id":"msg_3"}])
    );
    assert_eq!(ascending["has_more"], true);
    let empty = list("order=asc&after=msg_4");
    assert_eq!(empty["data"], json!([]));
    assert_eq!(empty["first_id"], "");
    assert_eq!(empty["last_id"], "");
    assert_eq!(empty["has_more"], false);
    // Percent escapes are decoded before the cursor is matched.
    assert_eq!(
        list("after=msg%5F4&limit=1")["data"],
        json!([{"id":"msg_3"}])
    );
    assert_eq!(
        page("after=missing").list(&items),
        Err(PageError::AfterNotFound("missing".into()))
    );
    for invalid in ["limit=0", "limit=101", "limit=x", "order=up", "include=x"] {
        assert!(
            matches!(ItemPage::parse(Some(invalid)), Err(PageError::Invalid(_))),
            "{invalid}"
        );
    }
    assert!(ItemPage::parse(Some("limit=100&include=&extra=1")).is_ok());
    assert_eq!(
        query_pairs(Some("a=b+c&d=%E2%9C%93&e")),
        [
            ("a".to_owned(), "b c".to_owned()),
            ("d".to_owned(), "✓".to_owned()),
            ("e".to_owned(), String::new()),
        ]
    );
}

#[test]
fn stored_input_keeps_unique_client_ids_and_replaces_repeats() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let body = json!({"input":[
        {"id":"msg_mine","role":"user","content":"one"},
        {"id":"msg_mine","role":"user","content":"two"},
        {"type":"message","role":"assistant","content":"ok"},
        {"role":"assistant","content":[{"type":"input_text","text":"array"},
            {"type":"output_text","text":"reply"}]},
    ]});
    let (response, _) = run(
        prepare(&body, Some(&store)).expect("valid"),
        vec![Event::Finished(stats(StopReason::Eos))],
    );
    let stored = store
        .load(response["id"].as_str().expect("id"))
        .expect("readable")
        .expect("stored");
    let all = ids(&stored.input_items);
    assert_eq!(all[0], "msg_mine");
    assert_ne!(all[1], "msg_mine");
    assert_eq!(stored.input_items[1]["type"], "message");
    assert_eq!(stored.input_items[2]["status"], "completed");
    assert_eq!(
        stored.input_items[2]["content"],
        json!([{"type":"output_text","text":"ok","annotations":[]}])
    );
    assert_eq!(
        stored.input_items[3]["content"],
        json!([
            {"type":"output_text","text":"array","annotations":[]},
            {"type":"output_text","text":"reply","annotations":[]}
        ])
    );
    let replay = prepare(
        &json!({"previous_response_id":response["id"]}),
        Some(&store),
    )
    .expect("stored history needs no additional input");
    assert_eq!(replay.request.messages[0].content, "one");
    assert_eq!(replay.request.messages[2].content, "ok");
    assert_eq!(
        replay.request.messages.last().expect("assistant").content,
        "arrayreply"
    );
}

#[test]
fn token_count_prepares_the_same_history_without_storing() {
    use super::responses::prepare_input_tokens;
    let scratch = Scratch::new();
    let store = scratch.open();
    let (response, _) = run(
        prepare(&json!({"input":"Read main.rs"}), Some(&store)).expect("valid"),
        first_turn(),
    );
    let body = json!({"previous_response_id":response["id"],"reasoning":{"effort":"none"},
        "input":[{"type":"function_call_output","call_id":"call_1","output":"hello"}]});
    let request = prepare_input_tokens(body.to_string().as_bytes(), true, Some(&store), None, "m")
        .expect("count input");
    assert!(!request.thinking);
    assert_eq!(request.messages[1].tool_calls, vec![read_call()]);
    assert_eq!(request.messages[2].content, "hello");
    assert_eq!(std::fs::read_dir(&scratch.0).expect("records").count(), 1);
    for extra in ["store", "stream", "max_output_tokens", "personality"] {
        let mut invalid = body.clone();
        invalid[extra] = json!(true);
        assert!(
            prepare_input_tokens(
                invalid.to_string().as_bytes(),
                true,
                Some(&store),
                None,
                "m"
            )
            .is_err()
        );
    }
}

#[test]
fn encrypted_reasoning_replays_after_restart_and_rejects_tampering() {
    use super::reasoning_crypto::ReasoningCipher;
    let scratch = Scratch::new();
    let store = scratch.open();
    let path = scratch.0.join("reasoning.key");
    let cipher = Arc::new(ReasoningCipher::open(&path).expect("cipher"));
    let body = json!({"input":"hi","store":false,"include":["reasoning.encrypted_content"]});
    let prepared =
        prepare_responses_with(body.to_string().as_bytes(), true, None, Some(&cipher), "m")
            .expect("request");
    let (response, _) = run(
        prepared,
        vec![
            Event::Reasoning("Check α before choosing.".into()),
            Event::Content("OK".into()),
            Event::Finished(stats(StopReason::Eos)),
        ],
    );
    let mut reasoning = response["output"][0].clone();
    assert!(
        reasoning["encrypted_content"]
            .as_str()
            .is_some_and(|text| !text.contains("Check"))
    );
    reasoning["content"] = json!([{"type":"reasoning_text","text":"untrusted replacement"}]);
    let mut follow = json!({"input":[{"role":"user","content":"hi"},reasoning,
        response["output"][1],{"role":"user","content":"continue"}]});
    let reopened = Arc::new(ReasoningCipher::open(path).expect("reopened"));
    let replay = |body: &Value| {
        prepare_responses_with(
            body.to_string().as_bytes(),
            true,
            None,
            Some(&reopened),
            "m",
        )
    };
    let request = replay(&follow).expect("authenticated history").request;
    assert_eq!(
        request.messages[1].reasoning_content.as_deref(),
        Some("Check α before choosing.")
    );
    let prepared = prepare_responses_with(
        follow.to_string().as_bytes(),
        true,
        Some(&store),
        Some(&reopened),
        "m",
    )
    .expect("stored encrypted replay");
    let (continued, _) = run(prepared, vec![Event::Finished(stats(StopReason::Eos))]);
    let saved = store
        .load(continued["id"].as_str().expect("id"))
        .expect("load")
        .expect("stored");
    assert_eq!(
        saved.input_items[1]["encrypted_content"],
        response["output"][0]["encrypted_content"]
    );
    assert_eq!(
        saved.input_items[1]["content"],
        response["output"][0]["content"]
    );
    let mut missing = follow.clone();
    missing["input"][1]
        .as_object_mut()
        .expect("item")
        .remove("id");
    assert!(
        replay(&missing)
            .err()
            .expect("missing id rejected")
            .to_string()
            .contains("item id")
    );
    let mut duplicate = follow.clone();
    duplicate["input"][0]["id"] = duplicate["input"][1]["id"].clone();
    assert!(
        replay(&duplicate)
            .err()
            .expect("duplicate id rejected")
            .to_string()
            .contains("item id")
    );
    follow["input"][1]["id"] = json!("rs_other");
    assert!(replay(&follow).is_err());
    follow["input"][1]["id"] = response["output"][0]["id"].clone();
    follow["input"][1]["encrypted_content"] = json!("tampered");
    assert!(replay(&follow).is_err());
}

/// A stored Response object with the fields recovery reads and must keep.
fn record(id: &str, background: bool, status: &str) -> Value {
    json!({
        "id":id,
        "object":"response",
        "created_at":1,
        "status":status,
        "completed_at":null,
        "background":background,
        "error":null,
        "model":"m",
        "output":[{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}],
        "instructions":"be brief",
        "metadata":{"k":"v"},
    })
}

fn input() -> Vec<Value> {
    vec![json!({"id":"msg_0","type":"message","role":"user","content":"hi"})]
}

#[test]
fn a_held_lease_excludes_other_stores_and_protects_an_active_job() {
    let scratch = Scratch::new();
    let first = scratch.open();
    let second = scratch.open();
    let id = "resp_active";
    let lease = first.lease(id).expect("lease").expect("free");
    assert!(second.lease(id).expect("lease").is_none(), "held elsewhere");
    assert!(
        first.lease(id).expect("lease").is_none(),
        "held by this store too"
    );
    assert!(
        second.lease("resp_other").expect("lease").is_some(),
        "per response"
    );
    first
        .save(&record(id, true, "queued"), &input())
        .expect("queued");
    // A server starting beside the live one must not fail its job.
    let third = scratch.open();
    let stored = third.load(id).expect("readable").expect("stored");
    assert_eq!(stored.response, record(id, true, "queued"));
    first
        .save(&record(id, true, "in_progress"), &input())
        .expect("running");
    drop(lease);
    assert!(
        second.lease(id).expect("lease").is_some(),
        "released on drop"
    );
}

#[test]
fn an_abandoned_background_job_is_failed_on_open_but_not_resumed() {
    let scratch = Scratch::new();
    let store = scratch.open();
    for (id, status) in [("resp_queued", "queued"), ("resp_running", "in_progress")] {
        let lease = store.lease(id).expect("lease").expect("free");
        store
            .save(&record(id, true, status), &input())
            .expect("saved");
        drop(lease);
        let reopened = scratch.open();
        let stored = reopened.load(id).expect("readable").expect("stored");
        let mut expected = record(id, true, status);
        expected["status"] = json!("failed");
        expected["error"] = json!({
            "code":"server_error",
            "message":"the server stopped before this background response finished; \
                interrupted execution is not resumed",
        });
        assert_eq!(stored.response, expected, "{status}");
        assert_eq!(stored.input_items, input());
        assert!(stored.response["completed_at"].is_null());
    }
}

#[test]
fn recovery_leaves_foreground_terminal_and_unrelated_files_alone() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let records = [
        record("resp_fg", false, "in_progress"),
        record("resp_done", true, "completed"),
        record("resp_failed", true, "failed"),
        record("resp_cancelled", true, "cancelled"),
        record("resp_incomplete", true, "incomplete"),
    ];
    for response in &records {
        store.save(response, &input()).expect("saved");
    }
    let unrelated = [
        scratch.0.join("notes.txt"),
        scratch.0.join(".resp_gone.lock"),
        scratch.0.join("resp_corrupt.json"),
        scratch.0.join(".resp_fg.123.tmp"),
    ];
    for path in &unrelated {
        std::fs::write(path, b"{\"trunc").expect("planted");
    }
    let reopened = scratch.open();
    for response in &records {
        let id = response["id"].as_str().expect("id");
        let stored = reopened.load(id).expect("readable").expect("stored");
        assert_eq!(&stored.response, response, "{id}");
    }
    for path in &unrelated {
        assert!(path.exists(), "{}", path.display());
    }
}

#[test]
fn leases_refuse_bad_ids_and_symlinks_and_keep_their_sidecar() {
    let scratch = Scratch::new();
    let store = scratch.open();
    let long = format!("resp_{}", "a".repeat(65));
    for invalid in [
        "",
        "resp_",
        "resp_../x",
        "resp_a/b",
        "../resp_a",
        "resp_a.json",
        long.as_str(),
    ] {
        let error = store.lease(invalid).expect_err("refused");
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::InvalidInput,
            "{invalid:?}"
        );
    }
    let id = "resp_stable";
    let sidecar = scratch.0.join(format!(".{id}.lock"));
    drop(store.lease(id).expect("lease").expect("free"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        let metadata = std::fs::metadata(&sidecar).expect("sidecar kept");
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        drop(store.lease(id).expect("lease").expect("free"));
        assert_eq!(
            std::fs::metadata(&sidecar).expect("still kept").ino(),
            metadata.ino()
        );
        let target = scratch.0.with_extension("lock-target");
        std::fs::write(&target, b"").expect("target");
        std::os::unix::fs::symlink(&target, scratch.0.join(".resp_link.lock")).expect("link");
        assert!(store.lease("resp_link").is_err(), "symlink not followed");
        let _ = std::fs::remove_file(target);
    }
    assert!(sidecar.exists());
}
