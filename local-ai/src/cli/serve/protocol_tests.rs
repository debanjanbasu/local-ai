//! CPU-only protocol tests for tool calling and the Responses API: request
//! parsing, SSE framing and buffered documents, all driven by synthetic engine
//! events so no model or GPU is touched.

use std::sync::Arc;

use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{ChatRequest, Event, GenerationStats, PrefillProgress, Signal, Stats, ToolCall};

use super::chunked::{Body, Flow, absorb};
use super::request::{GenerationRequest, prepare_generation};
use super::response::{Protocol, Reply};
use super::responses::{ResponsesState, prepare_responses};
use super::sse::Frames;
use super::{models_json, prepare};

fn chat_request(body: &Value, thinking: bool) -> crate::Result<ChatRequest> {
    let prepared = prepare_generation(body.to_string().as_bytes(), true, thinking)?;
    match prepared.request {
        GenerationRequest::Chat(request) => Ok(request),
        GenerationRequest::Completion(_) => Err(crate::Error::InvalidArgument("not chat".into())),
    }
}

fn chat_error(body: &Value) -> String {
    chat_request(body, true)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default()
}

fn responses_error(body: &Value) -> String {
    prepare_responses(body.to_string().as_bytes(), true)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default()
}

fn stats(stop_reason: StopReason) -> Box<Stats> {
    Box::new(Stats {
        stop_reason,
        cache_source: PromptCacheSource::None,
        reasoning_tokens: 2,
        generation: GenerationStats {
            prompt_tokens: 11,
            reused_prompt_tokens: 4,
            generated_tokens: 5,
            ..GenerationStats::default()
        },
    })
}

fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
    }
}

fn user() -> Value {
    json!({"role":"user","content":"weather?"})
}

#[test]
fn chat_parses_tools_history_and_nullable_content() {
    let body = json!({
        "messages":[
            {"role":"developer","content":"be brief"},
            user(),
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_a","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Oslo\"}"}},
                {"id":"call_b","type":"function","function":{"name":"time","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"call_a","content":"12C"},
            {"role":"tool","tool_call_id":"call_b","content":null}
        ],
        "tools":[
            {"type":"function","function":{"name":"weather","description":"Look up","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},
            {"type":"function","function":{"name":"time","strict":false}}
        ],
        "tool_choice":"auto",
        "max_tokens":10,
        "max_completion_tokens":20,
        "prompt_cache_key":"cache",
        "session_id":"session",
        "user":"user"
    });
    let request = chat_request(&body, true).expect("valid chat request");
    assert_eq!(request.max_tokens, 20, "max_completion_tokens wins");
    assert_eq!(request.session.as_deref(), Some("cache"));
    assert_eq!(request.tools.len(), 2);
    assert_eq!(request.tools[0].description.as_deref(), Some("Look up"));
    assert_eq!(request.tools[1].parameters, Value::Null);
    let assistant = &request.messages[2];
    assert_eq!(assistant.content, "");
    let ids: Vec<&str> = assistant
        .tool_calls
        .iter()
        .map(|call| call.id.as_str())
        .collect();
    assert_eq!(ids, ["call_a", "call_b"], "order is preserved");
    assert_eq!(assistant.tool_calls[0].arguments, json!({"city":"Oslo"}));
    assert_eq!(request.messages[3].tool_call_id.as_deref(), Some("call_a"));
    assert_eq!(request.messages[4].content, "");
}

#[test]
fn chat_session_falls_back_through_session_id_then_user() {
    let base = |extra: Value| {
        let mut body = json!({"messages":[user()]});
        if let (Value::Object(body), Value::Object(extra)) = (&mut body, extra) {
            body.extend(extra);
        }
        chat_request(&body, true).expect("valid").session
    };
    assert_eq!(
        base(json!({"session_id":"s","user":"u"})).as_deref(),
        Some("s")
    );
    assert_eq!(base(json!({"user":"u"})).as_deref(), Some("u"));
    assert_eq!(base(json!({})), None);
    let request = chat_request(&json!({"messages":[user()],"max_tokens":7}), true).expect("valid");
    assert_eq!(request.max_tokens, 7);
}

#[test]
fn chat_reasoning_effort_only_accepts_the_modes_the_template_has() {
    let effort = |effort: &str, server: bool| {
        chat_request(
            &json!({"messages":[user()],"reasoning_effort":effort}),
            server,
        )
        .map(|request| request.thinking)
    };
    assert!(!effort("none", true).expect("none"));
    assert!(effort("xhigh", true).expect("xhigh"));
    assert!(!effort("none", false).expect("none without thinking"));
    let error = effort("xhigh", false).expect_err("operator disabled thinking");
    assert!(error.to_string().contains("--no-thinking"), "{error}");
    for invalid in ["low", "medium", "high", "minimal", "max", "bogus"] {
        let error = effort(invalid, true).expect_err(invalid);
        assert!(error.to_string().contains("reasoning effort"), "{error}");
    }
    assert!(
        chat_request(&json!({"messages":[user()]}), false).is_ok_and(|request| !request.thinking)
    );
}

#[test]
fn chat_tool_choice_none_withholds_tools_and_forced_modes_fail() {
    let tools = json!([{"type":"function","function":{"name":"weather"}}]);
    let none = chat_request(
        &json!({"messages":[user()],"tools":tools,"tool_choice":"none"}),
        true,
    )
    .expect("none is valid");
    assert_eq!(none.tools, []);
    for forced in [
        json!("required"),
        json!({"type":"function","function":{"name":"weather"}}),
    ] {
        let error = chat_error(&json!({"messages":[user()],"tools":tools,"tool_choice":forced}));
        assert!(error.contains("constrained decoding"), "{error}");
    }
    assert!(
        chat_error(&json!({"messages":[user()],"tool_choice":"sometimes"})).contains("tool_choice")
    );
}

#[test]
fn chat_rejects_options_it_cannot_honour() {
    for (extra, needle) in [
        (json!({"n":2}), "n is not supported"),
        (json!({"stop":["\n"]}), "stop"),
        (json!({"logprobs":true}), "logprobs"),
        (
            json!({"response_format":{"type":"json_schema","json_schema":{}}}),
            "response_format",
        ),
        (
            json!({"response_format":{"type":"json_object"}}),
            "response_format",
        ),
        (json!({"functions":[{"name":"f"}]}), "functions"),
        (json!({"parallel_tool_calls":false}), "parallel_tool_calls"),
        (json!({"service_tier":"flex"}), "service_tier"),
        (
            json!({"tools":[{"type":"function","function":{"name":"f","strict":true}}]}),
            "strict",
        ),
        (json!({"tools":[{"type":"web_search"}]}), "tool type"),
    ] {
        let mut body = json!({"messages":[user()]});
        if let (Value::Object(body), Value::Object(extra)) = (&mut body, extra) {
            body.extend(extra);
        }
        let error = chat_error(&body);
        assert!(error.contains(needle), "{body}: {error}");
    }
    // Defaults spelled out are not options being asked for.
    let accepted = json!({"messages":[user()],"n":1,"stop":null,"logprobs":false,
        "response_format":{"type":"text"},"parallel_tool_calls":true,"service_tier":"auto",
        "stream":null});
    assert!(chat_request(&accepted, true).is_ok());
}

#[test]
fn chat_history_errors_are_reported_before_generation() {
    let bad_arguments = json!({"messages":[user(),{"role":"assistant","tool_calls":[
        {"id":"c","type":"function","function":{"name":"f","arguments":"{not json"}}]}]});
    assert!(chat_error(&bad_arguments).contains("not valid JSON"));
    let orphan = json!({"messages":[user(),{"role":"tool","content":"x"}]});
    assert!(chat_error(&orphan).contains("tool_call_id"));
    let user_null = json!({"messages":[{"role":"user","content":null}]});
    assert!(chat_error(&user_null).contains("text"));
}

#[test]
fn raw_completions_keep_their_lenient_parsing() {
    // Options a chat request is refused for were always ignored here, and the
    // raw API keeps that contract.
    let body = json!({"prompt":"hi","stop":["x"],"n":2,"max_tokens":3,"stream":true});
    let prepared = prepare_generation(body.to_string().as_bytes(), false, true).expect("valid");
    assert!(prepared.stream);
    assert!(!prepared.include_usage);
    let GenerationRequest::Completion(request) = prepared.request else {
        unreachable!("completion")
    };
    assert_eq!(request.max_tokens, 3);
    assert_eq!(request.prompt, "hi");
}

fn chat_reply(include_usage: bool) -> Reply {
    Reply {
        protocol: Protocol::Chat { include_usage },
        id: "chatcmpl-1".into(),
        created: 5,
        model: "m".into(),
    }
}

fn parse_data(frame: &str) -> Value {
    let data = frame
        .strip_prefix("data: ")
        .and_then(|rest| rest.strip_suffix("\n\n"))
        .unwrap_or_default();
    serde_json::from_str(data).unwrap_or(Value::Null)
}

#[test]
fn chat_stream_emits_complete_tool_calls_in_order_and_finishes_with_tool_calls() {
    let mut frames = Frames::new(chat_reply(true));
    assert_eq!(frames.start(), Vec::<String>::new());
    let mut out = Vec::new();
    for event in [
        Event::Reasoning("think".into()),
        Event::Content("Checking.".into()),
        Event::ToolCall(call("call_1", "weather", json!({"city":"Oslo"}))),
        Event::TokenIds(vec![1, 2]),
        Event::ToolCall(call("call_2", "time", json!({}))),
    ] {
        let (frames, terminal) = frames.event(event);
        assert!(!terminal);
        out.extend(frames);
    }
    let (tail, terminal) = frames.event(Event::Finished(stats(StopReason::Eos)));
    assert!(terminal);
    out.extend(tail);
    assert_eq!(out.len(), 7, "{out:#?}");
    let chunks: Vec<Value> = out[..6].iter().map(|frame| parse_data(frame)).collect();
    for chunk in &chunks {
        assert_eq!(chunk["id"], "chatcmpl-1", "one ID for the whole stream");
    }
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    assert_eq!(chunks[0]["created"], 5);
    assert_eq!(
        chunks[0]["choices"][0]["delta"],
        json!({"role":"assistant","reasoning_content":"think"})
    );
    assert_eq!(
        chunks[1]["choices"][0]["delta"],
        json!({"content":"Checking."})
    );
    assert_eq!(
        chunks[2]["choices"][0]["delta"]["tool_calls"],
        json!([{"index":0,"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Oslo\"}"}}])
    );
    assert_eq!(
        chunks[3]["choices"][0]["delta"]["tool_calls"][0]["index"],
        1
    );
    assert_eq!(
        chunks[3]["choices"][0]["delta"]["tool_calls"][0]["id"],
        "call_2"
    );
    assert_eq!(chunks[4]["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(chunks[5]["choices"], json!([]));
    assert_eq!(chunks[5]["usage"]["prompt_tokens"], 11);
    assert_eq!(
        chunks[5]["usage"]["completion_tokens_details"]["reasoning_tokens"],
        2
    );
    assert_eq!(
        chunks[5]["usage"]["prompt_tokens_details"]["cached_tokens"],
        4
    );
    assert_eq!(out[6], "data: [DONE]\n\n");
}

#[test]
fn chat_stream_reports_length_even_after_a_call_and_omits_usage_unless_asked() {
    let mut frames = Frames::new(chat_reply(false));
    let _ = frames.event(Event::ToolCall(call("c", "f", json!({}))));
    let (tail, _) = frames.event(Event::Finished(stats(StopReason::TokenLimit)));
    assert_eq!(tail.len(), 2, "{tail:?}");
    assert_eq!(
        parse_data(&tail[0])["choices"][0]["finish_reason"],
        "length"
    );
}

#[test]
fn completion_stream_is_unchanged_apart_from_its_identity() {
    let reply = Reply {
        protocol: Protocol::Completion,
        id: "cmpl-1".into(),
        created: 1,
        model: "m".into(),
    };
    let mut frames = Frames::new(reply);
    let (out, _) = frames.event(Event::Content("hi".into()));
    let chunk = parse_data(&out[0]);
    assert_eq!(chunk["object"], "text_completion");
    assert_eq!(chunk["choices"][0]["text"], "hi");
    let (out, terminal) = frames.event(Event::Finished(stats(StopReason::Eos)));
    assert!(terminal);
    assert_eq!(parse_data(&out[0])["choices"][0]["finish_reason"], "stop");
    assert_eq!(out[1], "data: [DONE]\n\n");
}

/// Fold signals through a buffered body as the pump does and return the bytes.
fn body_bytes(reply: Reply, signals: Vec<Signal>) -> Vec<u8> {
    let (mut body, _) = Body::open(reply, false);
    let mut out = body
        .take(false)
        .map(|frame| frame.to_vec())
        .unwrap_or_default();
    for signal in signals {
        let flow = absorb(&mut body, signal);
        if flow == Flow::Stop {
            break;
        }
        if flow != Flow::Keep
            && let Some(frame) = body.take(flow == Flow::Last)
        {
            out.extend_from_slice(&frame);
        }
    }
    out
}

#[test]
fn chat_body_lists_calls_with_null_content_and_tool_calls_finish() {
    let signals = vec![
        Signal::Event(Event::Reasoning("why".into())),
        Signal::Event(Event::ToolCall(call(
            "call_1",
            "weather",
            json!({"city":"Oslo"}),
        ))),
        Signal::Event(Event::ToolCall(call("call_2", "time", json!({})))),
        Signal::Event(Event::Finished(stats(StopReason::Eos))),
    ];
    let body = body_bytes(chat_reply(false), signals);
    let document: Value = serde_json::from_slice(&body).expect("valid JSON");
    let choice = &document["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert_eq!(choice["message"]["content"], Value::Null);
    assert_eq!(choice["message"]["reasoning_content"], "why");
    assert_eq!(
        choice["message"]["tool_calls"],
        json!([
            {"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Oslo\"}"}},
            {"id":"call_2","type":"function","function":{"name":"time","arguments":"{}"}}
        ])
    );
    assert_eq!(document["id"], "chatcmpl-1");
    assert_eq!(document["created"], 5);
}

#[test]
fn chat_body_keeps_text_alongside_calls() {
    let signals = vec![
        Signal::Event(Event::Content("Let me check.".into())),
        Signal::Event(Event::ToolCall(call("c", "f", json!({"a":1})))),
        Signal::Event(Event::Finished(stats(StopReason::Eos))),
    ];
    let document: Value =
        serde_json::from_slice(&body_bytes(chat_reply(false), signals)).expect("valid JSON");
    assert_eq!(
        document["choices"][0]["message"]["content"],
        "Let me check."
    );
    assert_eq!(
        document["choices"][0]["message"]["tool_calls"][0]["id"],
        "c"
    );
}

#[test]
fn responses_parse_instructions_history_and_tools() {
    let body = json!({
        "model":"anything",
        "instructions":"Be terse.",
        "input":[
            {"role":"system","content":"House rules."},
            {"role":"developer","content":[{"type":"input_text","text":"dev"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"weather?"}]},
            {"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"need tool"}]},
            {"type":"message","role":"assistant","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"Checking.","annotations":[]}]},
            {"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Oslo\"}","status":"completed"},
            {"type":"function_call_output","call_id":"call_1","output":"12C"},
            {"type":"function_call_output","call_id":"call_2","output":[{"type":"input_text","text":"a"},{"type":"input_text","text":"b"}]}
        ],
        "tools":[{"type":"function","name":"weather","description":"Look up","parameters":{"type":"object"},"strict":false}],
        "tool_choice":"auto",
        "max_output_tokens":64,
        "reasoning":{"effort":"none"},
        "store":false,
        "prompt_cache_key":"k",
        "metadata":{"a":"b"},
        "temperature":0.5
    });
    let prepared = prepare_responses(body.to_string().as_bytes(), true).expect("valid");
    let request = prepared.request;
    assert!(!prepared.stream);
    assert!(!request.thinking);
    assert_eq!(request.max_tokens, 64);
    assert_eq!(request.session.as_deref(), Some("k"));
    assert_eq!(request.tools.len(), 1);
    let roles: Vec<&str> = request.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(
        roles,
        ["system", "developer", "user", "assistant", "tool", "tool"]
    );
    assert_eq!(request.messages[0].content, "Be terse.\n\nHouse rules.");
    let turn = &request.messages[3];
    assert_eq!(turn.content, "Checking.");
    assert_eq!(turn.reasoning_content.as_deref(), Some("need tool"));
    assert_eq!(
        turn.tool_calls,
        vec![call("call_1", "weather", json!({"city":"Oslo"}))]
    );
    assert_eq!(request.messages[4].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(request.messages[5].content, "ab");
}

#[test]
fn responses_string_input_is_one_user_message_and_instructions_lead() {
    let prepared = prepare_responses(
        json!({"input":"hi","instructions":"sys"})
            .to_string()
            .as_bytes(),
        true,
    )
    .expect("valid");
    let roles: Vec<(&str, &str)> = prepared
        .request
        .messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str()))
        .collect();
    assert_eq!(roles, [("system", "sys"), ("user", "hi")]);
    assert!(prepared.request.thinking, "server default applies");
}

#[test]
fn responses_reject_what_they_cannot_honour() {
    for (body, needle) in [
        (json!({"input":"hi","store":true}), "store=true"),
        (
            json!({"input":"hi","previous_response_id":"resp_1"}),
            "previous_response_id",
        ),
        (
            json!({"input":"hi","conversation":"conv_1"}),
            "conversation",
        ),
        (json!({"input":"hi","background":true}), "background"),
        (
            json!({"input":"hi","tools":[{"type":"web_search"}]}),
            "tool type",
        ),
        (
            json!({"input":"hi","tools":[{"type":"function","name":"f","strict":true}]}),
            "strict",
        ),
        (
            json!({"input":"hi","tool_choice":"required"}),
            "constrained decoding",
        ),
        (
            json!({"input":"hi","tool_choice":{"type":"function","name":"f"}}),
            "constrained decoding",
        ),
        (
            json!({"input":"hi","text":{"format":{"type":"json_schema","name":"x","schema":{}}}}),
            "text.format",
        ),
        (
            json!({"input":"hi","reasoning":{"effort":"high"}}),
            "reasoning effort",
        ),
        (
            json!({"input":"hi","reasoning":{"summary":"auto"}}),
            "reasoning.summary",
        ),
        (
            json!({"input":"hi","include":["reasoning.encrypted_content"]}),
            "include",
        ),
        (json!({"input":"hi","truncation":"auto"}), "truncation"),
        (json!({"input":"hi","frobnicate":1}), "unknown field"),
        (
            json!({"input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]}),
            "text-only",
        ),
        (
            json!({"input":[{"type":"item_reference","id":"msg_1"}]}),
            "item_reference",
        ),
        (
            json!({"input":[{"type":"reasoning","summary":[],"encrypted_content":"x"}]}),
            "encrypted",
        ),
        (
            json!({"input":[{"type":"web_search_call","id":"ws"}]}),
            "input item type",
        ),
        (json!({"instructions":"x"}), "input is required"),
    ] {
        let error = responses_error(&body);
        assert!(error.contains(needle), "{body}: {error:?}");
    }
    // Explicit nulls and defaults are accepted.
    let accepted = json!({"input":"hi","store":false,"previous_response_id":null,
        "background":false,"truncation":"disabled","include":[],"text":{"format":{"type":"text"}},
        "tool_choice":"auto","parallel_tool_calls":true,"service_tier":"default",
        "reasoning":{"effort":"xhigh","summary":null},"conversation":null});
    assert!(prepare_responses(accepted.to_string().as_bytes(), true).is_ok());
}

fn responses_reply(body: &Value) -> Reply {
    let prepared = prepare_responses(body.to_string().as_bytes(), true).expect("valid");
    Reply {
        protocol: Protocol::Responses(Arc::new(prepared.echo)),
        id: "resp_abc".into(),
        created: 9,
        model: "m".into(),
    }
}

fn sse_events(frames: &[String]) -> Vec<Value> {
    frames
        .iter()
        .map(|frame| {
            let mut lines = frame.lines();
            let kind = lines
                .next()
                .and_then(|line| line.strip_prefix("event: "))
                .unwrap_or_default()
                .to_owned();
            let data: Value = lines
                .next()
                .and_then(|line| line.strip_prefix("data: "))
                .and_then(|data| serde_json::from_str(data).ok())
                .unwrap_or(Value::Null);
            assert_eq!(
                data["type"],
                kind.as_str(),
                "event line matches the payload"
            );
            data
        })
        .collect()
}

#[test]
fn responses_stream_follows_the_documented_lifecycle() {
    let reply =
        responses_reply(&json!({"input":"hi","tools":[{"type":"function","name":"weather"}]}));
    let mut frames = Frames::new(reply);
    let mut out = frames.start();
    for event in [
        Event::Reasoning("Th".into()),
        Event::Reasoning("ink".into()),
        Event::Content("Hi".into()),
        Event::Content(" there".into()),
        Event::ToolCall(call("call_9", "weather", json!({"city":"Oslo"}))),
        Event::TokenIds(vec![3]),
        Event::Finished(stats(StopReason::Eos)),
    ] {
        out.extend(frames.event(event).0);
    }
    let events = sse_events(&out);
    let kinds: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert_eq!(
        kinds,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event["sequence_number"], index, "{event}");
    }
    assert_eq!(events[0]["response"]["status"], "in_progress");
    assert_eq!(events[0]["response"]["output"], json!([]));
    assert_eq!(events[0]["response"]["store"], false);
    assert_eq!(events[0]["response"]["id"], "resp_abc");
    assert_eq!(events[13]["text"], "Hi there");
    assert_eq!(events[17]["delta"], "{\"city\":\"Oslo\"}");
    assert_eq!(events[18]["arguments"], "{\"city\":\"Oslo\"}");
    let done = &events[20]["response"];
    assert_eq!(done["status"], "completed");
    assert_eq!(done["created_at"], 9);
    assert!(done["completed_at"].is_u64());
    assert_eq!(done["usage"]["input_tokens"], 11);
    assert_eq!(done["usage"]["input_tokens_details"]["cached_tokens"], 4);
    assert_eq!(
        done["usage"]["input_tokens_details"]["cache_write_tokens"],
        0
    );
    assert_eq!(done["usage"]["output_tokens"], 5);
    assert_eq!(
        done["usage"]["output_tokens_details"]["reasoning_tokens"],
        2
    );
    assert_eq!(done["usage"]["total_tokens"], 16);
    assert_eq!(done["tools"][0]["name"], "weather");
    assert_eq!(done["tools"][0]["strict"], false);
    let output = done["output"].as_array().expect("output");
    assert_eq!(output.len(), 3);
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["content"][0]["text"], "Think");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"][0]["text"], "Hi there");
    assert_eq!(output[2]["type"], "function_call");
    assert_eq!(
        output[2]["call_id"], "call_9",
        "engine call ID is the call_id"
    );
    assert_eq!(output[2]["status"], "completed");
    // Item IDs are stable across every event that names the item, and distinct.
    let ids: Vec<&str> = output
        .iter()
        .filter_map(|item| item["id"].as_str())
        .collect();
    assert_eq!(events[2]["item"]["id"], ids[0]);
    assert_eq!(events[11]["item_id"], ids[1]);
    assert_eq!(events[17]["item_id"], ids[2]);
    assert!(ids[0].starts_with("rs_") && ids[1].starts_with("msg_") && ids[2].starts_with("fc_"));
    assert!(ids[0] != ids[1] && ids[1] != ids[2]);
    for (event, index) in [(&events[2], 0), (&events[9], 1), (&events[16], 2)] {
        assert_eq!(event["output_index"], index);
    }
}

#[test]
fn responses_stream_marks_a_token_limit_incomplete() {
    let mut frames = Frames::new(responses_reply(
        &json!({"input":"hi","max_output_tokens":2}),
    ));
    let _ = frames.start();
    let _ = frames.event(Event::Content("Hel".into()));
    let (out, terminal) = frames.event(Event::Finished(stats(StopReason::TokenLimit)));
    assert!(terminal);
    let events = sse_events(&out);
    let last = events.last().expect("terminal event");
    assert_eq!(last["type"], "response.incomplete");
    assert_eq!(last["response"]["status"], "incomplete");
    assert_eq!(
        last["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(last["response"]["completed_at"], Value::Null);
    assert_eq!(last["response"]["max_output_tokens"], 2);
    assert_eq!(last["response"]["output"][0]["status"], "incomplete");
}

#[test]
fn responses_stream_reports_a_mid_generation_failure() {
    let mut frames = Frames::new(responses_reply(&json!({"input":"hi"})));
    let _ = frames.start();
    let _ = frames.event(Event::Content("par".into()));
    let (out, terminal) = frames.event(Event::Error("metal said no".into()));
    assert!(terminal);
    let events = sse_events(&out);
    let kinds: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert_eq!(
        kinds,
        [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "error",
            "response.failed"
        ]
    );
    assert_eq!(events[3]["message"], "metal said no");
    assert_eq!(events[4]["response"]["status"], "failed");
    assert_eq!(events[4]["response"]["error"]["code"], "server_error");
}

#[test]
fn responses_body_is_the_streamed_terminal_response_after_whitespace() {
    let request = json!({"input":"hi","instructions":"sys","metadata":{"k":"v"}});
    let reply = responses_reply(&request);
    let events = || {
        vec![
            Event::Reasoning("r".into()),
            Event::Content("answer".into()),
            Event::ToolCall(call("call_1", "f", json!({"x":[1,2]}))),
            Event::Finished(stats(StopReason::Eos)),
        ]
    };
    let mut signals = vec![Signal::Progress(PrefillProgress {
        tokens: 128,
        chunks: 1,
    })];
    signals.extend(events().into_iter().map(Signal::Event));
    let body = body_bytes(reply.clone(), signals);
    // A space for the head and one per prefill boundary, then the document.
    assert!(
        body.starts_with(b"  {"),
        "{:?}",
        String::from_utf8_lossy(&body)
    );
    let mut document: Value = serde_json::from_slice(&body).expect("valid JSON");
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    let mut state = ResponsesState::new(&reply, Arc::clone(echo), true);
    state.start();
    let mut streamed = Value::Null;
    for event in events() {
        if let Some(response) = state.event(event) {
            streamed = response;
        }
    }
    // Only the wall-clock completion time may differ between two runs.
    document["completed_at"] = Value::Null;
    streamed["completed_at"] = Value::Null;
    assert_eq!(document, streamed);
    assert_eq!(document["object"], "response");
    assert_eq!(document["instructions"], "sys");
    assert_eq!(document["metadata"], json!({"k":"v"}));
    assert_eq!(document["output"][2]["arguments"], "{\"x\":[1,2]}");
}

#[test]
fn responses_body_failure_is_not_a_json_document() {
    let reply = responses_reply(&json!({"input":"hi"}));
    let body = body_bytes(
        reply,
        vec![
            Signal::Event(Event::Content("partial".into())),
            Signal::Event(Event::Error("boom".into())),
        ],
    );
    assert!(body.iter().all(u8::is_ascii_whitespace), "{body:?}");
    assert!(serde_json::from_slice::<Value>(&body).is_err());
}

#[test]
fn route_preparation_maps_each_api_to_its_protocol() {
    let (stream, request, protocol) = prepare(
        super::Api::Responses,
        br#"{"input":"hi","stream":true}"#,
        true,
    )
    .expect("ok");
    assert!(stream);
    assert!(matches!(request, GenerationRequest::Chat(_)));
    assert!(matches!(protocol, Protocol::Responses(_)));
    let (_, _, protocol) = prepare(
        super::Api::Chat,
        br#"{"messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_usage":true}}"#,
        true,
    )
    .expect("ok");
    assert!(matches!(
        protocol,
        Protocol::Chat {
            include_usage: true
        }
    ));
    let (_, request, protocol) =
        prepare(super::Api::Completion, br#"{"prompt":"x"}"#, true).expect("ok");
    assert!(matches!(request, GenerationRequest::Completion(_)));
    assert!(matches!(protocol, Protocol::Completion));
}

#[test]
fn reply_ids_are_prefixed_and_unique() {
    let a = Reply::new(
        Protocol::Chat {
            include_usage: false,
        },
        "m".into(),
    );
    let b = Reply::new(
        Protocol::Chat {
            include_usage: false,
        },
        "m".into(),
    );
    assert!(a.id.starts_with("chatcmpl-") && a.id.len() == "chatcmpl-".len() + 32);
    assert_ne!(a.id, b.id);
    assert!(
        Reply::new(Protocol::Completion, "m".into())
            .id
            .starts_with("cmpl-")
    );
}

#[test]
fn models_report_the_admitted_context() {
    let models = models_json("bonsai", 32768);
    assert_eq!(models["data"][0]["id"], "bonsai");
    assert_eq!(models["data"][0]["context_length"], 32768);
    assert_eq!(models["data"][0]["max_model_len"], 32768);
}
