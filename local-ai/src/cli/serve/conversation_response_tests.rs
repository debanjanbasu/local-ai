//! CPU-only tests for Responses that continue a durable conversation: the
//! history replayed into the prompt, the exactly-once, version-checked append
//! of the new input and output, and the refusals and failures around it,
//! driven by synthetic engine events so no model or GPU is touched.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{Event, GenerationStats, Stats};
use local_services::{Metadata, Store};

use super::super::reasoning_crypto::ReasoningCipher;
use super::super::response::{Protocol, Reply, new_id};
use super::super::store::ResponseStore;
use super::{
    Echo, PreparedResponses, ResponsesState, prepare_input_tokens_conversing,
    prepare_responses_conversing,
};

/// A scratch directory under the system temp dir, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        Self(std::env::temp_dir().join(new_id("local-ai-conversation-test-")))
    }

    fn conversations(&self) -> Store {
        Store::open(self.0.join("services")).expect("conversation store opens")
    }

    fn responses(&self) -> Arc<ResponseStore> {
        Arc::new(ResponseStore::open(self.0.join("responses")).expect("response store opens"))
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
        reasoning_tokens: 1,
        generation: GenerationStats {
            prompt_tokens: 9,
            generated_tokens: 3,
            ..GenerationStats::default()
        },
    })
}

/// A conversation holding one earlier exchange.
fn seeded(conversations: &Store) -> String {
    conversations
        .create_conversation(
            &Metadata::new(),
            vec![
                json!({"type":"message","role":"user","content":"Hello"}),
                json!({"type":"message","role":"assistant","content":"Hi there"}),
            ],
        )
        .expect("conversation created")
        .id
}

fn prepare(
    body: &Value,
    conversations: Option<&Store>,
    responses: Option<&Arc<ResponseStore>>,
    cipher: Option<&Arc<ReasoningCipher>>,
) -> crate::Result<PreparedResponses> {
    prepare_responses_conversing(
        body.to_string().as_bytes(),
        true,
        responses,
        None,
        conversations,
        cipher,
        "m",
    )
}

fn error(
    body: &Value,
    conversations: Option<&Store>,
    responses: Option<&Arc<ResponseStore>>,
) -> String {
    prepare(body, conversations, responses, None)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default()
}

fn roles(prepared: &PreparedResponses) -> Vec<(String, String)> {
    prepared
        .request
        .messages
        .iter()
        .map(|message| (message.role.clone(), message.content.clone()))
        .collect()
}

/// Run `events` through a streaming Response for `prepared`, as the server
/// would. Returns the terminal Response, every streamed event type, and the
/// request's echo.
fn run(prepared: PreparedResponses, events: Vec<Event>) -> (Value, Vec<String>, Arc<Echo>) {
    let echo = Arc::new(prepared.echo);
    let reply = Reply::new(Protocol::Responses(Arc::clone(&echo)), "m".into());
    let mut state = ResponsesState::new(&reply, Arc::clone(&echo), true);
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
    (terminal, kinds, echo)
}

fn answer(text: &str, stop: StopReason) -> Vec<Event> {
    vec![
        Event::Reasoning("Think.".into()),
        Event::Content(text.into()),
        Event::Finished(stats(stop)),
    ]
}

/// The conversation's items, oldest first, and its version.
fn history(conversations: &Store, id: &str) -> (Vec<Value>, u64) {
    let history = conversations
        .conversation_history(id)
        .expect("conversation readable");
    let items = history
        .items
        .iter()
        .map(|item| serde_json::to_value(item).expect("item serializes"))
        .collect();
    (items, history.conversation.version)
}

#[test]
fn conversation_history_precedes_the_new_input_and_is_echoed() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let id = seeded(&conversations);
    for conversation in [json!(id), json!({"id":id})] {
        let body = json!({"input":"How are you?","instructions":"sys","conversation":conversation});
        let prepared = prepare(&body, Some(&conversations), None, None).expect("valid");
        assert_eq!(
            roles(&prepared),
            [
                ("system".to_owned(), "sys".to_owned()),
                ("user".to_owned(), "Hello".to_owned()),
                ("assistant".to_owned(), "Hi there".to_owned()),
                ("user".to_owned(), "How are you?".to_owned()),
            ]
        );
    }
    // Input may be omitted: the conversation alone is the prompt.
    let prepared = prepare(
        &json!({"conversation":id}),
        Some(&conversations),
        None,
        None,
    )
    .expect("valid");
    assert_eq!(prepared.request.messages.len(), 2);
    let (response, kinds, _) = run(prepared, answer("Fine.", StopReason::Eos));
    assert_eq!(response["status"], "completed");
    assert_eq!(response["conversation"], json!({"id":id}));
    assert_eq!(response["previous_response_id"], Value::Null);
    assert!(
        kinds.contains(&"response.completed".to_owned()),
        "{kinds:?}"
    );
    // A Response without a conversation keeps its existing shape.
    let plain = prepare(&json!({"input":"hi"}), Some(&conversations), None, None).expect("valid");
    let (response, _, _) = run(plain, answer("ok", StopReason::Eos));
    assert!(response.get("conversation").is_none(), "{response}");
}

#[test]
fn conversation_needs_its_store_and_excludes_previous_response_id() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let responses = scratch.responses();
    let id = seeded(&conversations);
    let message = error(&json!({"input":"hi","conversation":id}), None, None);
    assert!(message.contains("conversation"), "{message}");
    let both = json!({"input":"hi","conversation":id,"previous_response_id":"resp_1"});
    let message = error(&both, Some(&conversations), Some(&responses));
    assert!(message.contains("cannot be used together"), "{message}");
    for shape in [
        json!(42),
        json!(""),
        json!({"id":""}),
        json!({"id":7}),
        json!({"id":id,"extra":1}),
    ] {
        let message = error(
            &json!({"input":"hi","conversation":shape}),
            Some(&conversations),
            None,
        );
        assert!(
            message.contains("conversation must be"),
            "{shape}: {message}"
        );
    }
    let message = error(
        &json!({"input":"hi","conversation":"conv_absent"}),
        Some(&conversations),
        None,
    );
    assert!(message.contains("not found"), "{message}");
    conversations.delete_conversation(&id).expect("deleted");
    let message = error(
        &json!({"input":"hi","conversation":id}),
        Some(&conversations),
        None,
    );
    assert!(message.contains("not found"), "{message}");
}

#[test]
fn new_input_the_conversation_cannot_keep_is_refused_before_generation() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let id = seeded(&conversations);
    for (input, needle) in [
        (
            json!([{"role":"user","content":"x","name":"bob"}]),
            "field \"name\"",
        ),
        (
            json!([{"id":"bad.id","role":"user","content":"x"}]),
            "item id",
        ),
        (
            json!([{"type":"function_call","call_id":"c","name":"f","arguments":{"a":1}}]),
            "arguments must be a JSON string",
        ),
    ] {
        let message = error(
            &json!({"input":input,"conversation":id}),
            Some(&conversations),
            None,
        );
        assert!(message.contains(needle), "{input}: {message}");
    }
    assert_eq!(history(&conversations, &id).0.len(), 2, "nothing written");
}

#[test]
fn input_token_counting_reads_the_conversation_without_changing_it() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let id = seeded(&conversations);
    let before = history(&conversations, &id);
    let count = |body: &Value| {
        prepare_input_tokens_conversing(
            body.to_string().as_bytes(),
            true,
            None,
            Some(&conversations),
            None,
            "m",
        )
    };
    let request = count(&json!({"input":"next","conversation":id})).expect("countable");
    let contents: Vec<&str> = request
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(contents, ["Hello", "Hi there", "next"]);
    let request = count(&json!({"conversation":{"id":id}})).expect("countable");
    assert_eq!(request.messages.len(), 2);
    assert_eq!(
        history(&conversations, &id),
        before,
        "counting never appends"
    );
    let message = prepare_input_tokens_conversing(
        json!({"input":"x","conversation":id})
            .to_string()
            .as_bytes(),
        true,
        None,
        None,
        None,
        "m",
    )
    .err()
    .map(|error| error.to_string())
    .unwrap_or_default();
    assert!(message.contains("conversation"), "{message}");
}

#[test]
fn a_completed_response_appends_its_new_input_and_output_exactly_once() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let responses = scratch.responses();
    let id = seeded(&conversations);
    let (_, version) = history(&conversations, &id);
    let body = json!({"input":"How are you?","conversation":id});
    let prepared = prepare(&body, Some(&conversations), Some(&responses), None).expect("valid");
    let (response, kinds, echo) = run(prepared, answer("Fine.", StopReason::Eos));
    assert_eq!(response["status"], "completed", "{response}");
    assert_eq!(response["store"], true);
    assert!(
        kinds.contains(&"response.completed".to_owned()),
        "{kinds:?}"
    );
    let (items, after) = history(&conversations, &id);
    assert_eq!(after, version + 1, "one append");
    // The prepended history is not repeated: two earlier items, the new
    // input, then the output, in order.
    let kinds: Vec<&str> = items
        .iter()
        .filter_map(|item| item["type"].as_str())
        .collect();
    assert_eq!(
        kinds,
        ["message", "message", "message", "reasoning", "message"]
    );
    assert_eq!(items[2]["content"][0]["text"], "How are you?");
    assert_eq!(items[3]["id"], response["output"][0]["id"]);
    assert_eq!(items[4]["id"], response["output"][1]["id"]);
    assert_eq!(items[4]["content"][0]["text"], "Fine.");
    assert_eq!(items[4]["content"][0]["logprobs"], json!([]));
    assert_eq!(items[4], response["output"][1]);
    let stored = responses
        .load(response["id"].as_str().expect("id"))
        .expect("readable")
        .expect("stored");
    assert_eq!(stored.response["conversation"], json!({"id":id}));
    // A repeat of the same append (the same response ID) changes nothing.
    echo.conversation
        .as_ref()
        .expect("conversing")
        .append(&response)
        .expect("replayed");
    assert_eq!(history(&conversations, &id), (items, after));
    // The next turn sees the whole conversation once.
    let next = prepare(
        &json!({"input":"And then?","conversation":id}),
        Some(&conversations),
        None,
        None,
    )
    .expect("valid");
    let contents: Vec<&str> = next
        .request
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(
        contents,
        ["Hello", "Hi there", "How are you?", "Fine.", "And then?"]
    );
    assert_eq!(
        next.request.messages[3].reasoning_content.as_deref(),
        Some("Think.")
    );
}

#[test]
fn a_stale_generation_conflicts_and_appends_nothing() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let responses = scratch.responses();
    let id = seeded(&conversations);
    let body = json!({"input":"How are you?","conversation":id});
    let prepared = prepare(&body, Some(&conversations), Some(&responses), None).expect("valid");
    // Another writer moves the conversation on while this one generates.
    conversations
        .add_items(&id, vec![json!({"role":"user","content":"interleaved"})])
        .expect("added");
    let before = history(&conversations, &id);
    let (response, kinds, _) = run(prepared, answer("Fine.", StopReason::Eos));
    assert_eq!(response["status"], "failed", "{response}");
    assert_eq!(response["store"], false, "nothing was kept");
    assert_eq!(response["conversation"], json!({"id":id}));
    let message = response["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains(&id) && message.contains("version"),
        "{message}"
    );
    assert!(
        kinds.ends_with(&["error".to_owned(), "response.failed".to_owned()]),
        "{kinds:?}"
    );
    assert!(
        !kinds.contains(&"response.completed".to_owned()),
        "{kinds:?}"
    );
    assert_eq!(history(&conversations, &id), before);
    let id = response["id"].as_str().expect("id");
    assert!(
        responses.load(id).expect("readable").is_none(),
        "not stored"
    );
}

#[test]
fn failed_and_cancelled_responses_append_nothing() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let id = seeded(&conversations);
    let body = json!({"input":"How are you?","conversation":id});
    let before = history(&conversations, &id);
    for events in [
        answer("Fi", StopReason::Cancelled),
        vec![
            Event::Content("Fi".into()),
            Event::Error("engine failed".into()),
        ],
    ] {
        let prepared = prepare(&body, Some(&conversations), None, None).expect("valid");
        let (response, _, _) = run(prepared, events);
        assert_eq!(response["status"], "failed", "{response}");
        assert_eq!(history(&conversations, &id), before);
    }
    // A conversation deleted during generation is not appended to.
    let prepared = prepare(&body, Some(&conversations), None, None).expect("valid");
    conversations.delete_conversation(&id).expect("deleted");
    let (response, _, _) = run(prepared, answer("Fine.", StopReason::Eos));
    assert_eq!(response["status"], "failed", "{response}");
    let message = response["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("not found"), "{message}");
}

#[test]
fn an_incomplete_response_is_appended_as_incomplete() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let id = seeded(&conversations);
    let body = json!({"input":"How are you?","conversation":id});
    let prepared = prepare(&body, Some(&conversations), None, None).expect("valid");
    let (response, _, _) = run(prepared, answer("Fi", StopReason::TokenLimit));
    assert_eq!(response["status"], "incomplete", "{response}");
    let (items, _) = history(&conversations, &id);
    assert_eq!(items.len(), 5);
    assert_eq!(items[4]["status"], "incomplete");
}

#[test]
fn a_background_response_appends_only_its_end_and_only_once() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let responses = scratch.responses();
    let id = seeded(&conversations);
    let body = json!({"input":"How are you?","conversation":id,"background":true});
    let prepared = prepare(&body, Some(&conversations), Some(&responses), None).expect("valid");
    assert!(prepared.echo.background());
    let echo = Arc::new(prepared.echo);
    let reply = Reply::new(Protocol::Responses(Arc::clone(&echo)), "m".into());
    let mut state = ResponsesState::new(&reply, echo, false);
    let before = history(&conversations, &id);
    state.save(&state.snapshot("queued")).expect("queued saved");
    state.started();
    state
        .save(&state.snapshot("in_progress"))
        .expect("progress saved");
    let mut terminal = Value::Null;
    for event in answer("Fine.", StopReason::Eos) {
        if let Some(done) = state.event(event) {
            terminal = done;
        }
    }
    assert_eq!(terminal["status"], "completed");
    assert_eq!(
        history(&conversations, &id),
        before,
        "the pump appends, not finish"
    );
    state.save(&terminal).expect("settled");
    let (items, version) = history(&conversations, &id);
    assert_eq!(items.len(), 5);
    state.save(&terminal).expect("saved again");
    assert_eq!(
        history(&conversations, &id),
        (items, version),
        "exactly once"
    );
    // A cancelled end appends nothing.
    let prepared = prepare(&body, Some(&conversations), Some(&responses), None).expect("valid");
    let echo = Arc::new(prepared.echo);
    let reply = Reply::new(Protocol::Responses(Arc::clone(&echo)), "m".into());
    let mut state = ResponsesState::new(&reply, echo, false);
    let before = history(&conversations, &id);
    let cancelled = state.cancelled();
    state.save(&cancelled).expect("cancel saved");
    assert_eq!(history(&conversations, &id), before);
}

#[test]
fn replayed_encrypted_reasoning_needs_the_cipher_and_stays_sealed() {
    let scratch = Scratch::new();
    let conversations = scratch.conversations();
    let cipher = Arc::new(ReasoningCipher::open(scratch.0.join("reasoning.key")).expect("cipher"));
    let envelope = cipher
        .seal("m", "rs_earlier", "Earlier thought.")
        .expect("sealed");
    let id = conversations
        .create_conversation(
            &Metadata::new(),
            vec![
                json!({"role":"user","content":"Hello"}),
                json!({"id":"rs_earlier","type":"reasoning","summary":[],"encrypted_content":envelope}),
                json!({"role":"assistant","content":"Hi there"}),
            ],
        )
        .expect("created")
        .id;
    let message = error(
        &json!({"input":"next","conversation":id}),
        Some(&conversations),
        None,
    );
    assert!(message.contains("--reasoning-key"), "{message}");
    let new_envelope = cipher.seal("m", "rs_new", "New thought.").expect("sealed");
    let body = json!({"conversation":id,"input":[
        {"id":"rs_new","type":"reasoning","summary":[],"encrypted_content":new_envelope},
        {"role":"assistant","content":"Interim."},
        {"role":"user","content":"next"}]});
    let prepared =
        prepare(&body, Some(&conversations), None, Some(&cipher)).expect("authenticated");
    assert_eq!(
        prepared.request.messages[1].reasoning_content.as_deref(),
        Some("Earlier thought.")
    );
    assert_eq!(
        prepared.request.messages[2].reasoning_content.as_deref(),
        Some("New thought.")
    );
    let (response, _, _) = run(
        prepared,
        vec![
            Event::Content("ok".into()),
            Event::Finished(stats(StopReason::Eos)),
        ],
    );
    assert_eq!(response["status"], "completed", "{response}");
    let (items, _) = history(&conversations, &id);
    let stored = items
        .iter()
        .find(|item| item["id"] == "rs_new")
        .expect("new reasoning appended");
    assert_eq!(stored["encrypted_content"], json!(new_envelope));
    assert!(
        stored.get("content").is_none(),
        "kept sealed, as sent: {stored}"
    );
}
