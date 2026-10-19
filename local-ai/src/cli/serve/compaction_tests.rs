//! CPU-only tests for compaction and reasoning summaries: what is kept and
//! summarized, how compacted items replay, persistence through a durable
//! conversation, the context budget, and the documents and events checked
//! against the pinned `OpenAI` schemas. Engine work is substituted: token
//! counts by a fixed measure, summaries by fixed text, generation by
//! synthetic events, so no model or GPU is touched.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::compaction::{COMPACTION_PROMPT, SUMMARY_HEADER};
use local_engine::{ChatMessage, ChatRequest, Event, GenerationStats, Stats};
use local_services::compaction::should_compact;
use local_services::{Metadata, Store};

use super::super::reasoning_crypto::ReasoningCipher;
use super::super::response::{Protocol, Reply, new_id};
use super::{
    Generate, History, PreparedResponses, ResponsesState, Summarizer, compaction_resource,
    prepare_compact, prepare_responses_conversing,
};

/// A scratch directory under the system temp dir, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let root = std::env::temp_dir().join(new_id("local-ai-compaction-test-"));
        std::fs::create_dir_all(&root).expect("scratch directory");
        Self(root)
    }

    fn cipher(&self) -> Arc<ReasoningCipher> {
        Arc::new(ReasoningCipher::open(self.0.join("reasoning.key")).expect("cipher opens"))
    }

    fn conversations(&self) -> Store {
        Store::open(self.0.join("services")).expect("conversation store opens")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---- the pinned OpenAI schemas -------------------------------------------

fn schemas() -> Value {
    serde_json::from_str(include_str!("openai_compaction_schemas.json")).expect("fixture parses")
}

/// Violations of `schema` by `value`: the subset of JSON Schema the pinned
/// spec uses for these documents — `$ref`, `allOf`, `anyOf`/`oneOf` (one
/// branch must hold), `type`, `enum`, `const`, `required`, `properties`,
/// `items`.
fn violations(value: &Value, schema: &Value, root: &Value, path: &str, out: &mut Vec<String>) {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let name = reference.rsplit('/').next().unwrap_or_default();
        violations(value, &root["schemas"][name], root, path, out);
    }
    for part in schema["allOf"].as_array().into_iter().flatten() {
        violations(value, part, root, path, out);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = schema[key].as_array() {
            let holds = branches.iter().any(|branch| {
                let mut found = Vec::new();
                violations(value, branch, root, path, &mut found);
                found.is_empty()
            });
            if !holds {
                out.push(format!("{path}: matches no {key} branch"));
            }
        }
    }
    if let Some(kinds) = schema.get("type") {
        let kinds: Vec<&str> = match kinds {
            Value::String(kind) => vec![kind.as_str()],
            Value::Array(kinds) => kinds.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        let fits = kinds.iter().any(|kind| match *kind {
            "string" => value.is_string(),
            "integer" => value.is_i64() || value.is_u64(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "object" => value.is_object(),
            "array" => value.is_array(),
            "null" => value.is_null(),
            _ => false,
        });
        if !fits {
            out.push(format!("{path}: {value} is not {kinds:?}"));
            return;
        }
    }
    if let Some(options) = schema["enum"].as_array()
        && !options.contains(value)
    {
        out.push(format!("{path}: {value} not in {options:?}"));
    }
    if let Some(constant) = schema.get("const")
        && constant != value
    {
        out.push(format!("{path}: {value} is not {constant}"));
    }
    if let Some(object) = value.as_object() {
        for required in schema["required"].as_array().into_iter().flatten() {
            if let Some(name) = required.as_str()
                && !object.contains_key(name)
            {
                out.push(format!("{path}: missing {name}"));
            }
        }
        for (name, property) in schema["properties"].as_object().into_iter().flatten() {
            if let Some(member) = object.get(name) {
                violations(member, property, root, &format!("{path}.{name}"), out);
            }
        }
    }
    if let (Some(elements), Some(items)) = (value.as_array(), schema.get("items")) {
        for (index, element) in elements.iter().enumerate() {
            violations(element, items, root, &format!("{path}[{index}]"), out);
        }
    }
}

/// Require `value` to conform to the pinned schema `name`.
pub(in crate::cli::serve) fn conforms(value: &Value, name: &str) {
    let root = schemas();
    assert!(root["schemas"].get(name).is_some(), "no schema {name}");
    let mut out = Vec::new();
    violations(value, &json!({"$ref": name}), &root, "$", &mut out);
    assert!(out.is_empty(), "{name}: {out:?}\n{value}");
}

/// The validator itself refuses what the spec refuses.
#[test]
fn the_schema_check_rejects_nonconforming_documents() {
    let root = schemas();
    let mut out = Vec::new();
    let bad = json!({"type":"compaction","id":"cmp_1"});
    violations(
        &bad,
        &json!({"$ref":"CompactionBody"}),
        &root,
        "$",
        &mut out,
    );
    assert!(
        out.iter().any(|line| line.contains("encrypted_content")),
        "{out:?}"
    );
    let mut out = Vec::new();
    let bad = json!({"type":"response.reasoning_summary_text.delta","item_id":"rs","output_index":0,"summary_index":0,"sequence_number":1});
    let schema = json!({"$ref":"ResponseReasoningSummaryTextDeltaEvent"});
    violations(&bad, &schema, &root, "$", &mut out);
    assert!(out.iter().any(|line| line.contains("delta")), "{out:?}");
}

// ---- helpers ---------------------------------------------------------------

fn stats(stop_reason: StopReason) -> Box<Stats> {
    Box::new(Stats {
        stop_reason,
        cache_source: PromptCacheSource::None,
        reasoning_tokens: 3,
        generation: GenerationStats {
            prompt_tokens: 40,
            reused_prompt_tokens: 30,
            generated_tokens: 12,
            ..GenerationStats::default()
        },
    })
}

fn prepare(
    body: &Value,
    conversations: Option<&Store>,
    cipher: Option<&Arc<ReasoningCipher>>,
) -> crate::Result<PreparedResponses> {
    prepare_responses_conversing(
        body.to_string().as_bytes(),
        true,
        None,
        None,
        conversations,
        cipher,
        "m",
    )
}

fn error(body: &Value, cipher: Option<&Arc<ReasoningCipher>>) -> String {
    prepare(body, None, cipher)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default()
}

/// Run `events` through a Response for `prepared`, as the server would.
/// Returns the terminal Response and every streamed event.
fn run(prepared: PreparedResponses, events: Vec<Event>, streaming: bool) -> (Value, Vec<Value>) {
    let echo = Arc::new(prepared.echo);
    let reply = Reply::new(Protocol::Responses(Arc::clone(&echo)), "m".into());
    let mut state = ResponsesState::new(&reply, echo, streaming);
    state.start();
    let mut terminal = Value::Null;
    for event in events {
        if let Some(done) = state.event(event) {
            terminal = done;
        }
    }
    (terminal, state.take_events())
}

fn kinds(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|event| event["type"].as_str())
        .collect()
}

/// A summarizer answering with `pieces`, recording each request it ran.
fn summarizer(pieces: &'static [&'static str], seen: Arc<Mutex<Vec<ChatRequest>>>) -> Summarizer {
    let generate: Generate = Arc::new(move |request, piece| {
        seen.lock().expect("lock").push(request);
        for text in pieces {
            piece(text);
        }
        Ok(*stats(StopReason::Eos))
    });
    Summarizer(generate)
}

/// One token per rendered message: enough to drive the budget. What is
/// counted must render as a prompt, as the engine's count requires.
#[allow(clippy::needless_pass_by_value)]
fn by_messages(messages: Vec<ChatMessage>) -> crate::Result<usize> {
    local_engine::Engine::render_chat_with_tools(&messages, false, &[])?;
    Ok(messages.len())
}

fn user(text: &str) -> Value {
    json!({"type":"message","role":"user","content":text})
}

fn assistant(text: &str) -> Value {
    json!({"type":"message","role":"assistant","content":text})
}

fn call(id: &str, name: &str) -> Value {
    json!({"type":"function_call","call_id":id,"name":name,"arguments":"{\"path\":\"src/lib.rs\"}"})
}

fn output(id: &str, text: &str) -> Value {
    json!({"type":"function_call_output","call_id":id,"output":text})
}

/// A two-task agent transcript with three tool round trips.
fn transcript() -> Vec<Value> {
    vec![
        json!({"type":"message","role":"developer","content":"You are a careful coding agent."}),
        user("Build a CSV parser in src/lib.rs."),
        json!({"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"Read the file first."}]}),
        assistant("Reading the file."),
        call("call_1", "read_file"),
        output("call_1", "fn main() {}"),
        call("call_2", "write_file"),
        output("call_2", "written 120 bytes"),
        assistant("The parser is in src/lib.rs."),
        user("Now add quoted-field support."),
        call("call_3", "write_file"),
        output("call_3", "written 180 bytes"),
        assistant("Quoted fields are supported."),
    ]
}

/// Each assistant message's calls are answered, in order, by the tool
/// messages right after it, and no tool message answers anything else.
fn pairs(messages: &[ChatMessage]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut open: Vec<String> = Vec::new();
    for message in messages {
        if message.role == "tool" {
            let id = message.tool_call_id.clone().unwrap_or_default();
            assert_eq!(
                open.first(),
                Some(&id),
                "tool result {id} answers no open call"
            );
            open.remove(0);
            pairs.push((id, message.content.clone()));
        } else {
            assert!(open.is_empty(), "calls {open:?} left unanswered");
            open = message
                .tool_calls
                .iter()
                .map(|call| call.id.clone())
                .collect();
        }
    }
    assert!(open.is_empty(), "calls {open:?} left unanswered");
    pairs
}

fn compact(
    items: &[Value],
    keep: usize,
    summary: &str,
    cipher: &Arc<ReasoningCipher>,
) -> (Vec<Value>, Vec<ChatMessage>) {
    let body = json!({"model":"m","input":items,"instructions":"sys"});
    let prepared = prepare_compact(body.to_string().as_bytes(), None, Some(cipher), "m")
        .expect("compact request is valid");
    let pass = prepared
        .history
        .compaction(keep, Vec::new(), None, by_messages)
        .expect("plans")
        .expect("something to summarize");
    let request = pass.request.clone();
    assert!(
        !request.thinking,
        "a summary is visible content, never reasoning"
    );
    assert_eq!(
        request
            .messages
            .last()
            .map(|message| message.content.as_str()),
        Some(COMPACTION_PROMPT)
    );
    let mut output = pass.kept().to_vec();
    let (item, _) = pass
        .finish(summary, &prepared.history, cipher, "m")
        .expect("seals");
    output.push(item);
    // As the client receives them, in output form.
    let resource = compaction_resource(&output, None);
    let output = resource["output"].as_array().cloned().expect("output");
    (output, request.messages)
}

// ---- compaction ------------------------------------------------------------

#[test]
fn compaction_keeps_tool_pairs_whole_and_replays_the_kept_items_faithfully() {
    let scratch = Scratch::new();
    let cipher = scratch.cipher();
    let items = transcript();
    // Before compaction: three exact call/result pairs.
    let original = prepare(&json!({"input":items}), None, None).expect("valid");
    let expected = pairs(&original.request.messages);
    assert_eq!(expected.len(), 3);

    for keep in 0..items.len() + 2 {
        let (output, summarized) = compact(&items, keep, "Parser written; quoting asked.", &cipher);
        let item = output.last().expect("compaction item");
        assert_eq!(item["type"], "compaction");
        assert!(item["id"].as_str().is_some_and(|id| id.starts_with("cmp_")));
        let sealed = item["encrypted_content"].as_str().expect("sealed");
        assert!(!sealed.contains("Parser written"), "the summary is sealed");
        // The window replays: summary, kept items, then the next request.
        let mut next = output.clone();
        next.push(user("Run the tests."));
        let replay = prepare(
            &json!({"input":next,"instructions":"sys"}),
            None,
            Some(&cipher),
        )
        .expect("compacted window is accepted as input");
        let messages = &replay.request.messages;
        // The instructions join the kept developer message, which takes the
        // template's one system slot.
        assert_eq!(messages[0].role, "developer");
        assert!(messages[0].content.starts_with("sys"));
        assert!(
            messages[0].content.contains("careful coding agent"),
            "instructions kept"
        );
        assert!(messages[1].content.starts_with(SUMMARY_HEADER));
        assert!(
            messages[1]
                .content
                .ends_with("Parser written; quoting asked.")
        );
        assert_eq!(
            messages.last().map(|message| message.content.as_str()),
            Some("Run the tests.")
        );
        assert!(
            messages
                .iter()
                .any(|message| message.content == "Now add quoted-field support."),
            "the latest request is kept verbatim"
        );
        // Every pair that survives does so exactly; none is split across
        // the summary, and the summarized pairs were rendered whole.
        let kept = pairs(messages);
        let summarized_pairs = pairs(summarized.get(..summarized.len() - 1).unwrap_or_default());
        for pair in &kept {
            assert!(expected.contains(pair), "{pair:?} altered");
        }
        for pair in &summarized_pairs {
            assert!(expected.contains(pair), "{pair:?} altered");
        }
        assert!(kept.len() + summarized_pairs.len() >= expected.len());
        // Exactly the kept items are rendered: a summarized one never is.
        let kept_reading = output
            .iter()
            .any(|item| item["content"][0]["text"] == "Reading the file.");
        assert_eq!(
            messages
                .iter()
                .any(|message| message.content == "Reading the file."),
            kept_reading,
            "keep {keep}"
        );
    }
}

#[test]
fn the_summary_follows_every_leading_instruction_message() {
    let scratch = Scratch::new();
    let cipher = scratch.cipher();
    // Codex leads with more than one developer message.
    let mut items = transcript();
    items.insert(
        1,
        json!({"type":"message","role":"developer","content":"Use the sandbox."}),
    );
    let (output, _) = compact(&items, 3, "The summary.", &cipher);
    let replay = prepare(&json!({"input":output}), None, Some(&cipher)).expect("valid");
    let messages = &replay.request.messages;
    let roles: Vec<&str> = messages
        .iter()
        .map(|message| message.role.as_str())
        .collect();
    assert_eq!(roles[..3], ["developer", "developer", "user"], "{roles:?}");
    assert!(messages[2].content.ends_with("The summary."));
    local_engine::Engine::render_chat_with_tools(messages, false, &[])
        .expect("renders with every instruction leading");
}

#[test]
fn recompaction_folds_the_earlier_summary_in() {
    let scratch = Scratch::new();
    let cipher = scratch.cipher();
    let (first, _) = compact(&transcript(), 3, "First summary.", &cipher);
    let mut longer = first;
    longer.extend([
        user("Add a header option."),
        assistant("Header option added."),
        user("Benchmark it."),
        assistant("2.1 GB/s."),
    ]);
    let (second, summarized) = compact(&longer, 1, "Second summary.", &cipher);
    // The second pass summarizes the first summary with what followed.
    assert!(
        summarized
            .iter()
            .any(|message| message.content.ends_with("First summary."))
    );
    assert!(
        summarized
            .iter()
            .any(|message| message.content == "Add a header option.")
    );
    let replay = prepare(&json!({"input":second}), None, Some(&cipher)).expect("valid");
    let rendered: Vec<&str> = replay
        .request
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect();
    assert!(
        rendered
            .iter()
            .any(|text| text.ends_with("Second summary."))
    );
    assert!(!rendered.iter().any(|text| text.ends_with("First summary.")));
    assert!(rendered.contains(&"Benchmark it."), "{rendered:?}");
}

#[test]
fn the_compact_resource_conforms_to_the_pinned_schema() {
    let scratch = Scratch::new();
    let cipher = scratch.cipher();
    let (output, _) = compact(&transcript(), 4, "Summary.", &cipher);
    let resource = compaction_resource(&output, Some(&stats(StopReason::Eos)));
    conforms(&resource, "CompactResource");
    assert_eq!(resource["object"], "response.compaction");
    assert_eq!(resource["usage"]["input_tokens"], 40);
    assert_eq!(
        resource["usage"]["input_tokens_details"]["cached_tokens"],
        30
    );
    assert_eq!(resource["usage"]["output_tokens"], 12);
    assert_eq!(resource["usage"]["total_tokens"], 52);
    let item = output.last().expect("compaction item");
    conforms(item, "CompactionBody");
    conforms(item, "CompactionSummaryItemParam");
    // Nothing to summarize: the window comes back as it was, unbilled.
    let normalized = json!({"type":"message","id":"msg_1","role":"user",
        "content":[{"type":"input_text","text":"hi"}]});
    let unchanged = compaction_resource(&[normalized], None);
    conforms(&unchanged, "CompactResource");
    assert_eq!(unchanged["usage"]["total_tokens"], 0);
}

#[test]
fn compaction_needs_the_key_and_an_authentic_sealed_item() {
    let scratch = Scratch::new();
    let cipher = scratch.cipher();
    let body = json!({"model":"m","input":"hi"});
    let refused = prepare_compact(body.to_string().as_bytes(), None, None, "m")
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(refused.contains("--reasoning-key"), "{refused}");
    for (body, needle) in [
        (json!({"input":"hi","extra":1}), "unknown field"),
        (json!({"model":"m"}), "input is required"),
        (
            json!({"input":"hi","previous_response_id":"resp_1"}),
            "--response-store",
        ),
    ] {
        let message = prepare_compact(body.to_string().as_bytes(), None, Some(&cipher), "m")
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(message.contains(needle), "{body}: {message}");
    }
    let (output, _) = compact(&transcript(), 2, "S.", &cipher);
    let item = output.last().expect("item").clone();
    // Without the key, and under another key, a compaction item is refused.
    assert!(error(&json!({"input":[&item]}), None).contains("--reasoning-key"));
    let other = Scratch::new().cipher();
    assert!(error(&json!({"input":[&item]}), Some(&other)).contains("compaction item"));
    // Moved to another id, or tampered with, it fails authentication.
    let mut moved = item.clone();
    moved["id"] = json!("cmp_other");
    assert!(error(&json!({"input":[moved]}), Some(&cipher)).contains("compaction item"));
    let mut anonymous = item;
    anonymous.as_object_mut().expect("object").remove("id");
    assert!(
        error(&json!({"input":[anonymous]}), Some(&cipher)).contains("original unique item id")
    );
}

#[test]
fn context_management_is_parsed_strictly_and_needs_the_key() {
    let scratch = Scratch::new();
    let cipher = scratch.cipher();
    let entry = |value: Value| json!({"input":"hi","context_management":value});
    let parsed = prepare(
        &entry(json!([{"type":"compaction","compact_threshold":1000}])),
        None,
        Some(&cipher),
    )
    .expect("valid");
    assert_eq!(
        parsed.context_compaction.and_then(|entry| entry.threshold),
        Some(1000)
    );
    let unbounded =
        prepare(&entry(json!([{"type":"compaction"}])), None, Some(&cipher)).expect("valid");
    assert!(
        unbounded
            .context_compaction
            .is_some_and(|entry| entry.threshold.is_none())
    );
    let absent = prepare(&json!({"input":"hi"}), None, Some(&cipher)).expect("valid");
    assert!(absent.context_compaction.is_none());
    for (value, needle) in [
        (
            json!([{"type":"compaction","compact_threshold":999}]),
            "at least 1000",
        ),
        (
            json!([{"type":"compaction","compact_threshold":"big"}]),
            "at least 1000",
        ),
        (json!([{"type":"truncation"}]), "only compaction"),
        (json!([{"type":"compaction","keep":2}]), "unknown field"),
        (json!([]), "non-empty array"),
        (
            json!([{"type":"compaction"},{"type":"compaction"}]),
            "one compaction entry",
        ),
    ] {
        let message = error(&entry(value.clone()), Some(&cipher));
        assert!(message.contains(needle), "{value}: {message}");
    }
    let message = error(&entry(json!([{"type":"compaction"}])), None);
    assert!(message.contains("--reasoning-key"), "{message}");
    // Not refused for background (see `combos_tests`); without a store a
    // background response is refused for that alone.
    let background =
        json!({"input":"hi","background":true,"context_management":[{"type":"compaction"}]});
    let message = error(&background, Some(&cipher));
    assert!(!message.contains("context_management"), "{message}");
}

#[test]
fn the_context_budget_decides_when_and_how_much_is_kept() {
    // Due at the threshold, or when the output reservation no longer fits.
    assert!(should_compact(1000, Some(1000), 256, 65_536));
    assert!(!should_compact(999, Some(1000), 256, 65_536));
    assert!(should_compact(65_300, None, 256, 65_536));
    let scratch = Scratch::new();
    let cipher = scratch.cipher();
    let body = json!({"input":transcript(),"context_management":[{"type":"compaction","compact_threshold":1000}]});
    let mut prepared = prepare(&body, None, Some(&cipher)).expect("valid");
    let before = prepared.request.messages.len();
    let keep = 3;
    let pass = prepared
        .history
        .compaction(keep, Vec::new(), None, by_messages)
        .expect("plans")
        .expect("due");
    // The kept tail fits the budget, apart from the always-kept instructions
    // and latest request.
    let kept = prepare(&json!({"input":pass.kept()}), None, None).expect("valid");
    assert!(
        kept.request.messages.len() <= keep + 2,
        "{}",
        kept.request.messages.len()
    );
    let (item, history) = pass
        .finish("Summary.", &prepared.history, &cipher, "m")
        .expect("seals");
    prepared.compacted(item.clone(), history).expect("renders");
    assert!(prepared.request.messages.len() < before);
    assert_eq!(prepared.echo.compaction.as_ref(), Some(&item));
    // An empty summary would lose the history, so it fails instead.
    let pass = prepared
        .history
        .compaction(0, Vec::new(), None, by_messages)
        .expect("plans")
        .expect("due");
    assert!(
        pass.finish("  \n", &prepared.history, &cipher, "m")
            .is_err()
    );
}

#[test]
fn a_compacted_conversation_survives_a_restart() {
    let scratch = Scratch::new();
    let cipher = scratch.cipher();
    let id = {
        let conversations = scratch.conversations();
        let id = conversations
            .create_conversation(&Metadata::new(), transcript())
            .expect("created")
            .id;
        let body = json!({"conversation":id,"input":"Add a delimiter option.","stream":true,
            "context_management":[{"type":"compaction","compact_threshold":1000}]});
        let mut prepared = prepare(&body, Some(&conversations), Some(&cipher)).expect("valid");
        let pass = prepared
            .history
            .compaction(2, Vec::new(), None, by_messages)
            .expect("plans")
            .expect("due");
        let (item, history) = pass
            .finish(
                "CSV parser with quoting done.",
                &prepared.history,
                &cipher,
                "m",
            )
            .expect("seals");
        prepared.compacted(item, history).expect("renders");
        let (response, events) = run(
            prepared,
            vec![
                Event::Content("Delimiter option added.".into()),
                Event::Finished(stats(StopReason::Eos)),
            ],
            true,
        );
        assert_eq!(response["status"], "completed", "{response}");
        conforms(&response, "Response");
        assert_eq!(response["output"][0]["type"], "compaction");
        assert_eq!(response["output"][1]["type"], "message");
        assert_eq!(
            &kinds(&events)[..4],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.output_item.done"
            ]
        );
        assert_eq!(events[2]["item"]["type"], "compaction");
        assert_eq!(events[4]["output_index"], 1);
        for event in &events[2..4] {
            conforms(
                event,
                if event["type"] == "response.output_item.added" {
                    "ResponseOutputItemAddedEvent"
                } else {
                    "ResponseOutputItemDoneEvent"
                },
            );
        }
        id
    };
    // A new process: reopen the store and continue the conversation.
    let conversations = scratch.conversations();
    let listed = conversations
        .conversation_history(&id)
        .expect("history")
        .items;
    assert_eq!(listed.len(), transcript().len() + 3, "nothing is deleted");
    let next = prepare(
        &json!({"conversation":id,"input":"Thanks."}),
        Some(&conversations),
        Some(&cipher),
    )
    .expect("valid");
    let rendered: Vec<(&str, &str)> = next
        .request
        .messages
        .iter()
        .map(|message| (message.role.as_str(), message.content.as_str()))
        .collect();
    assert_eq!(
        rendered[0],
        ("developer", "You are a careful coding agent.")
    );
    assert!(rendered[1].1.ends_with("CSV parser with quoting done."));
    assert_eq!(
        &rendered[rendered.len() - 3..],
        [
            ("user", "Add a delimiter option."),
            ("assistant", "Delimiter option added."),
            ("user", "Thanks."),
        ]
    );
    assert!(!rendered.contains(&("assistant", "Reading the file.")));
}

#[test]
fn history_without_a_compaction_item_renders_as_before() {
    let history = History {
        summary: None,
        items: vec![user("a"), assistant("b")],
        instructions: Some("sys".into()),
        files: None,
    };
    let roles: Vec<String> = history
        .messages()
        .expect("renders")
        .into_iter()
        .map(|message| message.role)
        .collect();
    assert_eq!(roles, ["system", "user", "assistant"]);
}

// ---- reasoning summaries ---------------------------------------------------

fn summarized(effort: &str, detail: &str, stream: bool) -> PreparedResponses {
    prepare(
        &json!({"input":"What is 2+2?","stream":stream,"reasoning":{"effort":effort,"summary":detail}}),
        None,
        None,
    )
    .expect("valid")
}

fn thought(stop: StopReason) -> Vec<Event> {
    vec![
        Event::Reasoning("Raw: 2 plus 2 ".into()),
        Event::Reasoning("is 4.".into()),
        Event::Content("4".into()),
        Event::Finished(stats(stop)),
    ]
}

#[test]
fn reasoning_summaries_stream_per_spec_without_relabeling_raw_reasoning() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut prepared = summarized("xhigh", "concise", true);
    assert!(prepared.echo.summarizes());
    prepared
        .echo
        .set_summarizer(summarizer(&["I added ", "two and two."], Arc::clone(&seen)));
    let (response, events) = run(prepared, thought(StopReason::Eos), true);
    assert_eq!(response["status"], "completed", "{response}");
    conforms(&response, "Response");
    assert_eq!(response["reasoning"]["summary"], "concise");
    let reasoning = &response["output"][0];
    assert_eq!(
        reasoning["content"],
        json!([{"type":"reasoning_text","text":"Raw: 2 plus 2 is 4."}])
    );
    assert_eq!(
        reasoning["summary"],
        json!([{"type":"summary_text","text":"I added two and two."}])
    );
    assert_eq!(response["output"][1]["content"][0]["text"], "4");
    // The summary generation saw the raw reasoning, without thinking.
    let requests = seen.lock().expect("lock");
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].thinking);
    assert!(
        requests[0].messages[0]
            .content
            .contains("Raw: 2 plus 2 is 4.")
    );
    drop(requests);
    let order = kinds(&events);
    let at = |kind: &str| {
        order
            .iter()
            .position(|seen| *seen == kind)
            .unwrap_or(usize::MAX)
    };
    let rs_done = order
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == "response.output_item.done")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(rs_done.len(), 2);
    // The answer finishes first; the reasoning item finishes with its summary.
    assert_eq!(events[rs_done[0]]["output_index"], 1);
    assert_eq!(events[rs_done[1]]["output_index"], 0);
    assert!(at("response.reasoning_text.done") < at("response.output_text.delta"));
    assert!(rs_done[0] < at("response.reasoning_summary_part.added"));
    assert!(
        at("response.reasoning_summary_part.added") < at("response.reasoning_summary_text.delta")
    );
    assert!(
        at("response.reasoning_summary_text.done") < at("response.reasoning_summary_part.done")
    );
    assert!(at("response.reasoning_summary_part.done") < rs_done[1]);
    assert!(rs_done[1] < at("response.completed"));
    assert_eq!(
        order
            .iter()
            .filter(|kind| **kind == "response.reasoning_summary_text.delta")
            .count(),
        2
    );
    let schema = |kind: &str| match kind {
        "response.reasoning_summary_part.added" => Some("ResponseReasoningSummaryPartAddedEvent"),
        "response.reasoning_summary_part.done" => Some("ResponseReasoningSummaryPartDoneEvent"),
        "response.reasoning_summary_text.delta" => Some("ResponseReasoningSummaryTextDeltaEvent"),
        "response.reasoning_summary_text.done" => Some("ResponseReasoningSummaryTextDoneEvent"),
        "response.output_item.done" => Some("ResponseOutputItemDoneEvent"),
        "response.output_item.added" => Some("ResponseOutputItemAddedEvent"),
        _ => None,
    };
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event["sequence_number"], index);
        if let Some(name) = event["type"].as_str().and_then(schema) {
            conforms(event, name);
        }
    }
    let done = &events[rs_done[1]]["item"];
    assert_eq!(done["summary"], reasoning["summary"]);
    assert_eq!(done["content"], reasoning["content"]);
}

#[test]
fn a_buffered_response_carries_the_same_summary() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut prepared = summarized("xhigh", "detailed", false);
    prepared.echo.set_summarizer(summarizer(&["Summed."], seen));
    let (response, events) = run(prepared, thought(StopReason::TokenLimit), false);
    assert_eq!(events, Vec::<Value>::new());
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["output"][0]["summary"][0]["text"], "Summed.");
    assert_eq!(response["reasoning"]["summary"], "detailed");
    conforms(&response, "Response");
}

#[test]
fn summary_failure_fails_and_cancellation_skips_the_summary() {
    let failing: Generate = Arc::new(|_, _| Err(crate::Error::Generation("queue is full".into())));
    let mut prepared = summarized("xhigh", "auto", true);
    prepared.echo.set_summarizer(Summarizer(failing));
    let (response, events) = run(prepared, thought(StopReason::Eos), true);
    assert_eq!(response["status"], "failed");
    assert!(
        response["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("queue is full")),
        "{response}"
    );
    assert_eq!(response["output"][0]["summary"], json!([]));
    let finished = events
        .iter()
        .filter(|event| event["type"] == "response.output_item.done")
        .count();
    assert_eq!(finished, 2, "every added item is finished");

    let ran = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&ran);
    let watching: Generate = Arc::new(move |_, _| {
        flag.store(true, Ordering::SeqCst);
        Ok(*stats(StopReason::Eos))
    });
    let mut prepared = summarized("xhigh", "auto", true);
    prepared.echo.set_summarizer(Summarizer(watching));
    let (response, _) = run(prepared, thought(StopReason::Cancelled), true);
    assert_eq!(response["status"], "failed");
    assert!(
        !ran.load(Ordering::SeqCst),
        "no summary for a cancelled generation"
    );
}

#[test]
fn reasoning_summary_options_are_validated() {
    let body = |reasoning: Value| json!({"input":"hi","reasoning":reasoning});
    assert!(error(&body(json!({"summary":"verbose"})), None).contains("auto, concise or detailed"));
    // The deprecated `generate_summary` is validated as `summary` is.
    assert!(
        error(&body(json!({"generate_summary":"verbose"})), None)
            .contains("auto, concise or detailed")
    );
    let background = json!({"input":"hi","background":true,"reasoning":{"summary":"auto"}});
    let message = error(&background, None);
    assert!(!message.contains("reasoning.summary"), "{message}");
    let current = prepare(&body(json!({"summary":"concise"})), None, None).expect("valid");
    assert!(current.echo.summarizes());
    // With reasoning off there is nothing to summarize and nothing runs.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut prepared = summarized("none", "auto", true);
    prepared
        .echo
        .set_summarizer(summarizer(&["x"], Arc::clone(&seen)));
    let (response, _) = run(
        prepared,
        vec![
            Event::Content("hi".into()),
            Event::Finished(stats(StopReason::Eos)),
        ],
        true,
    );
    assert_eq!(response["status"], "completed");
    assert!(seen.lock().expect("lock").is_empty());
    // Without one, no summary is reported as requested.
    let plain = prepare(&json!({"input":"hi"}), None, None).expect("valid");
    let (response, _) = run(plain, thought(StopReason::Eos), true);
    assert_eq!(response["reasoning"]["summary"], Value::Null);
    assert_eq!(response["output"][0]["summary"], json!([]));
}

/// A streamed summary's events leave as each piece is generated, in sequence
/// with the rest of the stream, and a client that goes away stops it.
#[test]
fn summary_events_are_flushed_as_generated_and_a_gone_client_stops_them() {
    // A generation that, like the engine's, stops when told nobody listens.
    let generate: Generate = Arc::new(|_, piece| {
        for text in ["one ", "two ", "three"] {
            if !piece(text) {
                return Err(crate::Error::Generation("client went away".into()));
            }
        }
        Ok(*stats(StopReason::Eos))
    });
    for listening in [usize::MAX, 1] {
        let mut prepared = summarized("xhigh", "auto", true);
        prepared
            .echo
            .set_summarizer(Summarizer(Arc::clone(&generate)));
        let echo = Arc::new(prepared.echo);
        let reply = Reply::new(Protocol::Responses(Arc::clone(&echo)), "m".into());
        let mut state = ResponsesState::new(&reply, echo, true);
        let flushed = Arc::new(Mutex::new(Vec::<Vec<Value>>::new()));
        let sink = Arc::clone(&flushed);
        state.set_flush(Box::new(move |events| {
            let mut sink = sink.lock().expect("lock");
            sink.push(events);
            sink.len() < listening
        }));
        state.start();
        let mut events = state.take_events();
        let mut terminal = Value::Null;
        for event in thought(StopReason::Eos) {
            if let Some(done) = state.event(event) {
                terminal = done;
            }
            if terminal.is_null() {
                events.extend(state.take_events());
            }
        }
        let flushed = flushed.lock().expect("lock").clone();
        // One flush per piece, each ending with its delta.
        for batch in &flushed {
            assert_eq!(
                batch.last().map(|event| &event["type"]),
                Some(&json!("response.reasoning_summary_text.delta"))
            );
        }
        events.extend(flushed.into_iter().flatten());
        events.extend(state.take_events());
        for (index, event) in events.iter().enumerate() {
            assert_eq!(event["sequence_number"], index, "{event}");
        }
        let deltas = kinds(&events)
            .into_iter()
            .filter(|kind| *kind == "response.reasoning_summary_text.delta")
            .count();
        if listening == usize::MAX {
            assert_eq!(terminal["status"], "completed", "{terminal}");
            assert_eq!(deltas, 3);
            assert_eq!(terminal["output"][0]["summary"][0]["text"], "one two three");
        } else {
            assert_eq!(terminal["status"], "failed", "{terminal}");
            assert_eq!(deltas, 1, "the summary stopped with the client");
        }
    }
}
