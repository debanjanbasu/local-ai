//! CPU-only protocol tests for tool calling and the Responses API: request
//! parsing, SSE framing and buffered documents, all driven by synthetic engine
//! events so no model or GPU is touched.

use std::sync::Arc;

use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{
    ChatRequest, Event, GenerationStats, PrefillProgress, ResponseFormat, Signal, Stats, ToolCall,
    ToolChoice,
};

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
fn chat_tool_choice_maps_onto_the_native_choice() {
    let tools = json!([
        {"type":"function","function":{"name":"weather"}},
        {"type":"function","function":{"name":"time"}}
    ]);
    let choose = |choice: Value| {
        chat_request(
            &json!({"messages":[user()],"tools":tools,"tool_choice":choice}),
            true,
        )
    };
    for (choice, native) in [
        (Value::Null, ToolChoice::Auto),
        (json!("auto"), ToolChoice::Auto),
        (json!("none"), ToolChoice::None),
        (json!("required"), ToolChoice::Required),
        (
            json!({"type":"function","function":{"name":"time"}}),
            ToolChoice::Function("time".into()),
        ),
    ] {
        let request = choose(choice.clone()).expect("valid choice");
        assert_eq!(request.tool_choice, native, "{choice}");
        // `none` is enforced natively; the tools are still declared, so the
        // prompt is the same for every choice.
        assert_eq!(request.tools.len(), 2, "{choice}");
        assert!(request.parallel_tool_calls, "parallel calls by default");
    }
    let request = chat_request(&json!({"messages":[user()]}), true).expect("no tools");
    assert_eq!(request.tool_choice, ToolChoice::Auto);
    assert_eq!(request.tools, []);
}

#[test]
fn chat_tool_choice_shape_is_exact_and_checked_against_tools() {
    let tools = json!([{"type":"function","function":{"name":"weather"}}]);
    for (choice, needle) in [
        (json!("sometimes"), "invalid tool_choice \"sometimes\""),
        (json!("any"), "invalid tool_choice \"any\""),
        (json!(true), "invalid tool_choice true"),
        (json!({}), "tool_choice.type is required"),
        (json!({"type":1}), "tool_choice.type must be a string"),
        (
            json!({"type":"allowed_tools","allowed_tools":{"mode":"auto","tools":[]}}),
            "allowed_tools",
        ),
        (
            json!({"type":"custom","custom":{"name":"weather"}}),
            "tool_choice type \"custom\" is not supported",
        ),
        // The Responses shape is not the Chat one.
        (
            json!({"type":"function","name":"weather"}),
            "unknown field tool_choice.name",
        ),
        (
            json!({"type":"function"}),
            "tool_choice.function is required",
        ),
        (
            json!({"type":"function","function":"weather"}),
            "tool_choice.function must be an object",
        ),
        (
            json!({"type":"function","function":{}}),
            "tool_choice.function.name is required",
        ),
        (
            json!({"type":"function","function":{"name":7}}),
            "tool_choice.function.name must be a string",
        ),
        (
            json!({"type":"function","function":{"name":""}}),
            "tool_choice.function.name must not be empty",
        ),
        (
            json!({"type":"function","function":{"name":"weather","arguments":"{}"}}),
            "unknown field tool_choice.function.arguments",
        ),
        (
            json!({"type":"function","function":{"name":"time"}}),
            "\"time\", which is not declared in tools",
        ),
    ] {
        let error = chat_error(&json!({"messages":[user()],"tools":tools,"tool_choice":choice}));
        assert!(error.contains(needle), "{choice}: {error}");
    }
    let error = chat_error(&json!({"messages":[user()],"tool_choice":"required"}));
    assert!(error.contains("needs at least one tool"), "{error}");
    let error = chat_error(&json!({"messages":[user()],
        "tool_choice":{"type":"function","function":{"name":"weather"}}}));
    assert!(error.contains("not declared in tools"), "{error}");
    // Without tools, `auto` and `none` change nothing and stay accepted.
    for choice in ["auto", "none"] {
        assert!(chat_request(&json!({"messages":[user()],"tool_choice":choice}), true).is_ok());
    }
}

#[test]
fn chat_parallel_tool_calls_and_strict_reach_the_engine() {
    let body = json!({"messages":[user()],"parallel_tool_calls":false,"tools":[
        {"type":"function","function":{"name":"weather","strict":true,"parameters":{
            "type":"object","properties":{"city":{"type":"string"}},
            "required":["city"],"additionalProperties":false}}},
        {"type":"function","function":{"name":"time","strict":false}},
        {"type":"function","function":{"name":"noop","strict":null}},
        {"type":"function","function":{"name":"plain"}}
    ]});
    let request = chat_request(&body, true).expect("valid");
    assert!(!request.parallel_tool_calls);
    let strict: Vec<bool> = request.tools.iter().map(|tool| tool.strict).collect();
    assert_eq!(strict, [true, false, false, false]);
    let mut parallel = body;
    parallel["parallel_tool_calls"] = json!(true);
    assert!(
        chat_request(&parallel, true)
            .expect("valid")
            .parallel_tool_calls
    );
    let error = chat_error(&json!({"messages":[user()],"tools":[
        {"type":"function","function":{"name":"f","strict":"yes"}}]}));
    assert!(error.contains("invalid JSON request"), "{error}");
    let error = chat_error(&json!({"messages":[user()],"parallel_tool_calls":"no"}));
    assert!(error.contains("invalid JSON request"), "{error}");
}

#[test]
fn chat_rejects_options_it_cannot_honour() {
    for (extra, needle) in [
        (json!({"n":2}), "n is not supported"),
        (json!({"stop":["\n"]}), "stop"),
        (json!({"logprobs":true}), "logprobs"),
        (json!({"functions":[{"name":"f"}]}), "functions"),
        (json!({"service_tier":"flex"}), "service_tier"),
        (json!({"store":true}), "store"),
        (json!({"verbosity":"high"}), "verbosity"),
        (json!({"tools":[{"type":"web_search"}]}), "tool type"),
        (
            json!({"tools":[{"type":"function"}]}),
            "requires a function object",
        ),
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
fn raw_completions_reject_options_they_cannot_honour() {
    for (extra, needle) in [
        (json!({"stop":["x"]}), "stop"),
        (json!({"n":2}), "n is not supported"),
        (json!({"logprobs":0}), "logprobs is not supported"),
        (json!({"logprobs":5}), "logprobs is not supported"),
        (json!({"logprobs":false}), "logprobs must be"),
        (json!({"logit_bias":{"1":2}}), "logit_bias"),
        (json!({"best_of":2}), "best_of"),
        (json!({"echo":true}), "echo"),
        (json!({"suffix":"tail"}), "suffix"),
        (
            json!({"stream_options":{"include_usage":true}}),
            "include_usage",
        ),
    ] {
        let mut body = json!({"prompt":"hi"});
        body.as_object_mut()
            .expect("object")
            .extend(extra.as_object().expect("object").clone());
        let error = prepare_generation(body.to_string().as_bytes(), false, true)
            .err()
            .expect("unsupported option")
            .to_string();
        assert!(error.contains(needle), "{body}: {error}");
    }
    let body = json!({"prompt":"hi","stop":null,"n":1,"best_of":1,"echo":false,
        "suffix":null,"logprobs":null,"max_tokens":3,"stream":true});
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
    assert_eq!(chunk["choices"][0].get("logprobs"), Some(&Value::Null));
    let (out, terminal) = frames.event(Event::Finished(stats(StopReason::Eos)));
    assert!(terminal);
    assert_eq!(parse_data(&out[0])["choices"][0]["finish_reason"], "stop");
    assert_eq!(out[1], "data: [DONE]\n\n");
    let mut failed = Frames::new(Reply::new(Protocol::Completion, "m".into()));
    let (out, terminal) = failed.event(Event::Error("failed".into()));
    assert!(terminal);
    assert_eq!(
        parse_data(&out[0])["error"],
        json!({
            "message":"failed","type":"server_error","code":null,"param":null
        })
    );
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
    assert_eq!(choice.get("logprobs"), Some(&Value::Null));
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
        "reasoning":{"effort":"none","context":"all_turns"},
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
        json!({"input":"hi","instructions":"sys","client_metadata":{
            "session_id":"telemetry-only", "instructions":"not a model instruction",
            "x-codex-turn-metadata":"{\"turn_id\":\"turn-1\"}"}})
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
    assert!(
        prepared.request.session.is_none(),
        "telemetry is not a session key"
    );
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
            json!({"input":"hi","tool_choice":"required"}),
            "needs at least one tool",
        ),
        (
            json!({"input":"hi","tool_choice":{"type":"function","name":"f"}}),
            "not declared in tools",
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
            json!({"input":"hi","client_metadata":{"turn_id":42}}),
            "expected a string",
        ),
        (json!({"input":"hi","client_metadata":[]}), "expected a map"),
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
        "reasoning":{"effort":"xhigh","summary":null},"conversation":null,"client_metadata":null});
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
        if let Some(response) = event.get("response") {
            assert_eq!(response.get("access_programs"), Some(&Value::Null));
            assert_eq!(response["reasoning"]["context"], "all_turns");
        }
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
    let request = json!({"input":"hi","instructions":"sys","metadata":{"k":"v"},
        "client_metadata":{"session_id":"telemetry-only"}});
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
    assert!(document.get("client_metadata").is_none());
    assert!(!document.to_string().contains("telemetry-only"));
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

/// A request Oh My Pi's `openai-responses` provider sends with the custom
/// provider in `local-engine/README.md`. Recorded from `omp` 18.8.6 against a
/// stand-in endpoint, with `instructions` and the tool description shortened.
/// With those flags it sends no `include`, `reasoning.summary` or `strict`;
/// `--thinking off` sends `effort: "none"` and every other level `xhigh`.
fn omp_turn(effort: &str, input: &Value) -> Value {
    json!({
        "model":"ternary-bonsai-2-27b",
        "input":input,
        "instructions":"You are omp's trusted coding agent.",
        "tools":[{"type":"function","name":"read","description":"Read a file.","parameters":{
            "type":"object",
            "properties":{
                "i":{"type":"string","description":"concise intent"},
                "path":{"type":"string","description":"Local path, internal URI, or URL; selectors inline."}
            },
            "required":["path","i"],
            "additionalProperties":false
        }}],
        "stream":true,
        "prompt_cache_key":"01a122e6-26a1-7649-85a2-6679d989b419",
        "store":false,
        "max_output_tokens":8192,
        "reasoning":{"effort":effort}
    })
}

/// Oh My Pi resends earlier output items stripped of IDs and output-only
/// statuses; reasoning keeps only its summary and content.
fn omp_replay(item: &Value) -> Value {
    let mut item = item.as_object().cloned().unwrap_or_default();
    if item.get("type").and_then(Value::as_str) == Some("reasoning") {
        item.retain(|key, _| matches!(key.as_str(), "type" | "summary" | "content"));
    } else {
        item.remove("id");
        item.remove("status");
    }
    Value::Object(item)
}

#[test]
fn responses_accept_an_oh_my_pi_read_and_tool_result_turn() {
    let prompt = json!({"role":"user","content":[
        {"type":"input_text","text":"<system-reminder>\nToday: 2026-10-10.\n</system-reminder>"},
        {"type":"input_text","text":"Read main.rs and tell me what it prints."}
    ]});
    let first = omp_turn("xhigh", &json!([prompt]));
    let prepared = prepare_responses(first.to_string().as_bytes(), true).expect("first turn");
    assert!(prepared.stream);
    assert!(prepared.request.thinking);
    assert_eq!(
        prepared.request.session.as_deref(),
        Some("01a122e6-26a1-7649-85a2-6679d989b419")
    );
    assert_eq!(prepared.request.tools[0].name, "read");

    // The server's own answer to that turn, as Oh My Pi replays it.
    let read = call(
        "call_1",
        "read",
        json!({"i":"see what it prints","path":"main.rs"}),
    );
    let reply = responses_reply(&first);
    let Protocol::Responses(echo) = &reply.protocol else {
        unreachable!("responses reply")
    };
    let mut state = ResponsesState::new(&reply, Arc::clone(echo), false);
    let mut response = Value::Null;
    for event in [
        Event::Reasoning("Read the file first.".into()),
        Event::ToolCall(read.clone()),
        Event::Finished(stats(StopReason::Eos)),
    ] {
        if let Some(done) = state.event(event) {
            response = done;
        }
    }
    let mut input = vec![prompt];
    input.extend(
        response["output"]
            .as_array()
            .expect("output")
            .iter()
            .map(omp_replay),
    );
    assert_eq!(
        input[2],
        json!({"type":"function_call","call_id":"call_1","name":"read",
            "arguments":"{\"i\":\"see what it prints\",\"path\":\"main.rs\"}"})
    );
    input.push(json!({"type":"function_call_output","call_id":"call_1",
        "output":"fn main() {\n    println!(\"hello\");\n}"}));

    // The user turned thinking off before the tool result went back.
    let second = omp_turn("none", &Value::Array(input));
    let request = prepare_responses(second.to_string().as_bytes(), true)
        .expect("tool-result turn")
        .request;
    assert!(!request.thinking);
    let roles: Vec<&str> = request.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["system", "user", "assistant", "tool"]);
    assert!(
        request.messages[1]
            .content
            .ends_with("tell me what it prints.")
    );
    let turn = &request.messages[2];
    assert_eq!(
        turn.reasoning_content.as_deref(),
        Some("Read the file first.")
    );
    assert_eq!(turn.tool_calls, vec![read]);
    let result = &request.messages[3];
    assert_eq!(result.tool_call_id.as_deref(), Some("call_1"));
    assert!(result.content.contains("println!(\"hello\")"));

    // Without those flags Oh My Pi asks for semantics this server lacks, and
    // the request fails instead of being answered as if they were honoured.
    let mut encrypted = first.clone();
    encrypted["include"] = json!(["reasoning.encrypted_content"]);
    let mut summary = first;
    summary["reasoning"]["summary"] = json!("auto");
    for (body, needle) in [(encrypted, "include"), (summary, "reasoning.summary")] {
        let error = responses_error(&body);
        assert!(error.contains(needle), "{error:?}");
    }
}

#[test]
fn route_preparation_maps_each_api_to_its_protocol() {
    let (stream, request, protocol) = prepare(
        super::Api::Responses,
        br#"{"input":"hi","stream":true}"#,
        true,
        None,
        None,
        "m",
    )
    .expect("ok");
    assert!(stream);
    assert!(matches!(request, GenerationRequest::Chat(_)));
    assert!(matches!(protocol, Protocol::Responses(_)));
    let (_, _, protocol) = prepare(
        super::Api::Chat,
        br#"{"messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_usage":true}}"#,
        true,
        None,
        None,
        "m",
    )
    .expect("ok");
    assert!(matches!(
        protocol,
        Protocol::Chat {
            include_usage: true
        }
    ));
    let (_, request, protocol) = prepare(
        super::Api::Completion,
        br#"{"prompt":"x"}"#,
        true,
        None,
        None,
        "m",
    )
    .expect("ok");
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
    let models = models_json("bonsai", 32768, 1234);
    assert_eq!(models["data"][0]["id"], "bonsai");
    assert_eq!(models["data"][0]["created"], 1234);
    assert_eq!(models["data"][0]["context_length"], 32768);
    assert_eq!(models["data"][0]["max_model_len"], 32768);
}

fn structured_stats(stop_reason: StopReason, complete: bool) -> Box<Stats> {
    let mut stats = stats(stop_reason);
    stats.generation.response_format_complete = Some(complete);
    stats
}

fn schema() -> Value {
    json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})
}

#[test]
fn chat_response_format_maps_onto_the_native_format() {
    let format = |format: Value| {
        let mut body = json!({"messages":[user()]});
        body["response_format"] = format;
        chat_request(&body, true).map(|request| request.response_format)
    };
    assert_eq!(
        chat_request(&json!({"messages":[user()]}), true)
            .expect("valid")
            .response_format,
        ResponseFormat::Text
    );
    assert_eq!(format(Value::Null).expect("null"), ResponseFormat::Text);
    assert_eq!(
        format(json!({"type":"text"})).expect("text"),
        ResponseFormat::Text
    );
    assert_eq!(
        format(json!({"type":"json_object"})).expect("json_object"),
        ResponseFormat::JsonObject
    );
    // `strict` is accepted either way and never relaxes enforcement.
    for strict in [json!(true), json!(false), Value::Null] {
        let format = format(json!({"type":"json_schema","json_schema":{
            "name":"Weather_1-x","description":"d","schema":schema(),"strict":strict}}))
        .expect("json_schema");
        assert_eq!(format, ResponseFormat::JsonSchema(schema()));
    }
}

#[test]
fn chat_response_format_envelope_is_strict() {
    let long = "n".repeat(65);
    for (format, needle) in [
        (json!("json_object"), "response_format must be an object"),
        (json!({}), "response_format.type is required"),
        (json!({"type":1}), "response_format.type must be a string"),
        (
            json!({"type":"grammar"}),
            "type \"grammar\" is not supported",
        ),
        (
            json!({"type":"text","schema":{}}),
            "unknown field response_format.schema",
        ),
        (
            json!({"type":"json_object","strict":true}),
            "unknown field response_format.strict",
        ),
        (
            json!({"type":"json_schema"}),
            "response_format.json_schema is required",
        ),
        (
            json!({"type":"json_schema","json_schema":[]}),
            "json_schema must be an object",
        ),
        // The Responses shape is not the Chat one.
        (
            json!({"type":"json_schema","name":"x","schema":schema()}),
            "unknown field response_format.name",
        ),
        (
            json!({"type":"json_schema","json_schema":{"schema":schema()}}),
            "response_format.json_schema.name is required",
        ),
        (
            json!({"type":"json_schema","json_schema":{"name":"a b","schema":schema()}}),
            "must be 1 to 64 characters",
        ),
        (
            json!({"type":"json_schema","json_schema":{"name":long,"schema":schema()}}),
            "must be 1 to 64 characters",
        ),
        (
            json!({"type":"json_schema","json_schema":{"name":"x"}}),
            "response_format.json_schema.schema is required",
        ),
        (
            json!({"type":"json_schema","json_schema":{"name":"x","schema":true}}),
            "schema must be a JSON Schema object",
        ),
        (
            json!({"type":"json_schema","json_schema":{"name":"x","schema":{},"strict":"yes"}}),
            "strict must be a boolean",
        ),
        (
            json!({"type":"json_schema","json_schema":{"name":"x","schema":{},"description":1}}),
            "description must be a string",
        ),
        (
            json!({"type":"json_schema","json_schema":{"name":"x","schema":{},"extra":1}}),
            "unknown field response_format.json_schema.extra",
        ),
    ] {
        let error = chat_error(&json!({"messages":[user()],"response_format":format}));
        assert!(error.contains(needle), "{format}: {error}");
    }
}

#[test]
fn structured_formats_combine_with_tools_and_every_choice() {
    let tools = json!([{"type":"function","function":{"name":"weather"}}]);
    for choice in [
        json!("auto"),
        json!("none"),
        json!("required"),
        json!({"type":"function","function":{"name":"weather"}}),
    ] {
        let body = json!({"messages":[user()],"tools":tools,"tool_choice":choice,
            "response_format":{"type":"json_object"}});
        let request = chat_request(&body, true).expect("tools with a format");
        assert_eq!(request.tools.len(), 1, "{choice}");
        assert_eq!(request.response_format, ResponseFormat::JsonObject);
    }

    let tools = json!([{"type":"function","name":"weather"}]);
    for choice in [json!("auto"), json!("none"), json!("required")] {
        let body = json!({"input":"hi","tools":tools,"tool_choice":choice,
            "text":{"format":{"type":"json_schema","name":"w","schema":schema()}}});
        let prepared = prepare_responses(body.to_string().as_bytes(), true).expect("combined");
        assert_eq!(prepared.request.tools.len(), 1, "{choice}");
        assert_eq!(
            prepared.request.response_format,
            ResponseFormat::JsonSchema(schema())
        );
    }
}

#[test]
fn legacy_completions_refuse_tools() {
    for extra in [
        json!({"tools":[{"type":"function","function":{"name":"f"}}]}),
        json!({"tool_choice":"none"}),
        json!({"parallel_tool_calls":false}),
    ] {
        let mut body = json!({"prompt":"hi"});
        if let (Value::Object(body), Value::Object(extra)) = (&mut body, extra) {
            body.extend(extra);
        }
        let error = prepare_generation(body.to_string().as_bytes(), false, true)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains("not legacy Completions"), "{body}: {error}");
    }
    let defaults = json!({"prompt":"hi","tools":[],"tool_choice":null,"parallel_tool_calls":true});
    assert!(prepare_generation(defaults.to_string().as_bytes(), false, true).is_ok());
}

#[test]
fn responses_tool_policy_maps_onto_the_engine_and_is_echoed_as_accepted() {
    let terminal = |body: &Value| {
        let mut frames = Frames::new(responses_reply(body));
        let _ = frames.start();
        let _ = frames.event(Event::Content("ok".into()));
        let (out, _) = frames.event(Event::Finished(stats(StopReason::Eos)));
        sse_events(&out).last().expect("terminal")["response"].clone()
    };
    let tools = json!([
        {"type":"function","name":"weather","parameters":{"type":"object",
            "properties":{"city":{"type":"string"}},"required":["city"],
            "additionalProperties":false},"strict":true},
        {"type":"function","name":"time","strict":false},
        {"type":"function","name":"plain"}
    ]);
    let defaults = json!({"input":"hi","tools":tools});
    let prepared = prepare_responses(defaults.to_string().as_bytes(), true).expect("valid");
    assert_eq!(prepared.request.tool_choice, ToolChoice::Auto);
    assert!(prepared.request.parallel_tool_calls);
    let strict: Vec<bool> = prepared
        .request
        .tools
        .iter()
        .map(|tool| tool.strict)
        .collect();
    assert_eq!(strict, [true, false, false]);
    let response = terminal(&defaults);
    assert_eq!(response["tool_choice"], "auto");
    assert_eq!(response["parallel_tool_calls"], true);
    let echoed: Vec<&Value> = response["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|tool| &tool["strict"])
        .collect();
    assert_eq!(echoed, [&json!(true), &json!(false), &json!(false)]);

    for (choice, native) in [
        (json!("none"), ToolChoice::None),
        (json!("required"), ToolChoice::Required),
        (
            json!({"type":"function","name":"time"}),
            ToolChoice::Function("time".into()),
        ),
    ] {
        let body = json!({"input":"hi","tools":tools,"tool_choice":choice,
            "parallel_tool_calls":false});
        let prepared = prepare_responses(body.to_string().as_bytes(), true).expect("valid");
        assert_eq!(prepared.request.tool_choice, native, "{choice}");
        assert!(!prepared.request.parallel_tool_calls);
        assert_eq!(prepared.request.tools.len(), 3, "tools stay declared");
        let response = terminal(&body);
        assert_eq!(response["tool_choice"], choice);
        assert_eq!(response["parallel_tool_calls"], false);
    }
}

#[test]
fn responses_tool_choice_shape_is_exact_and_checked_against_tools() {
    let tools = json!([{"type":"function","name":"weather"}]);
    for (choice, needle) in [
        (json!("sometimes"), "invalid tool_choice \"sometimes\""),
        (json!(1), "invalid tool_choice 1"),
        (json!({"name":"weather"}), "tool_choice.type is required"),
        (
            json!({"type":"allowed_tools","mode":"auto","tools":[]}),
            "allowed_tools",
        ),
        (
            json!({"type":"web_search_preview"}),
            "tool_choice type \"web_search_preview\" is not supported",
        ),
        (
            json!({"type":"mcp","server_label":"x"}),
            "tool_choice type \"mcp\" is not supported",
        ),
        // The Chat shape is not the Responses one.
        (
            json!({"type":"function","function":{"name":"weather"}}),
            "unknown field tool_choice.function",
        ),
        (json!({"type":"function"}), "tool_choice.name is required"),
        (
            json!({"type":"function","name":["weather"]}),
            "tool_choice.name must be a string",
        ),
        (
            json!({"type":"function","name":"time"}),
            "\"time\", which is not declared in tools",
        ),
    ] {
        let error = responses_error(&json!({"input":"hi","tools":tools,"tool_choice":choice}));
        assert!(error.contains(needle), "{choice}: {error}");
    }
    for (tool, needle) in [
        (
            json!({"type":"function","name":"f","strict":"yes"}),
            "strict must be a boolean",
        ),
        (json!("f"), "each tool must be an object"),
        (
            json!({"type":"function","name":"f","parameters":[]}),
            "parameters must be a JSON Schema object",
        ),
    ] {
        let error = responses_error(&json!({"input":"hi","tools":[tool]}));
        assert!(error.contains(needle), "{tool}: {error}");
    }
    let error = responses_error(&json!({"input":"hi","parallel_tool_calls":"no"}));
    assert!(error.contains("invalid JSON request"), "{error}");
}

#[test]
fn legacy_completions_keep_refusing_structured_formats() {
    for format in [
        json!({"type":"json_object"}),
        json!({"type":"json_schema","json_schema":{"name":"x","schema":schema()}}),
    ] {
        let body = json!({"prompt":"hi","response_format":format});
        let error = prepare_generation(body.to_string().as_bytes(), false, true)
            .err()
            .expect("refused")
            .to_string();
        assert!(error.contains("not legacy Completions"), "{error}");
    }
    for format in [json!({"type":"text"}), Value::Null] {
        let body = json!({"prompt":"hi","response_format":format});
        let prepared = prepare_generation(body.to_string().as_bytes(), false, true).expect("text");
        let GenerationRequest::Completion(request) = prepared.request else {
            unreachable!("completion")
        };
        assert_eq!(request.response_format, ResponseFormat::Text);
    }
}

#[test]
fn responses_text_format_maps_and_is_echoed_as_accepted() {
    let terminal = |body: &Value| {
        let mut frames = Frames::new(responses_reply(body));
        let _ = frames.start();
        let _ = frames.event(Event::Content("{}".into()));
        let (out, _) = frames.event(Event::Finished(structured_stats(StopReason::Eos, true)));
        sse_events(&out).last().expect("terminal").clone()
    };
    let plain = json!({"input":"hi"});
    assert_eq!(
        terminal(&plain)["response"]["text"],
        json!({"format":{"type":"text"}})
    );
    let object = json!({"input":"hi","text":{"format":{"type":"json_object"}}});
    let prepared = prepare_responses(object.to_string().as_bytes(), true).expect("valid");
    assert_eq!(prepared.request.response_format, ResponseFormat::JsonObject);
    assert_eq!(
        terminal(&object)["response"]["text"]["format"],
        json!({"type":"json_object"})
    );
    // `strict` is reported as sent, `false` when omitted; enforced either way.
    let flat = json!({"input":"hi","text":{"format":{"type":"json_schema","name":"weather",
        "description":"A city.","schema":schema()},"verbosity":"medium"}});
    let prepared = prepare_responses(flat.to_string().as_bytes(), true).expect("valid");
    assert_eq!(
        prepared.request.response_format,
        ResponseFormat::JsonSchema(schema())
    );
    let event = terminal(&flat);
    assert_eq!(event["type"], "response.completed");
    assert_eq!(
        event["response"]["text"]["format"],
        json!({"type":"json_schema","name":"weather","description":"A city.",
            "schema":schema(),"strict":false})
    );
    let strict = json!({"input":"hi","text":{"format":{"type":"json_schema","name":"w",
        "schema":schema(),"strict":true}}});
    assert_eq!(
        terminal(&strict)["response"]["text"]["format"]["strict"],
        true
    );
}

#[test]
fn responses_text_format_envelope_is_strict() {
    for (format, needle) in [
        (json!("text"), "text.format must be an object"),
        (json!({"type":"python"}), "type \"python\" is not supported"),
        (
            json!({"type":"text","name":"x"}),
            "unknown field text.format.name",
        ),
        // The Chat shape is not the Responses one.
        (
            json!({"type":"json_schema","json_schema":{"name":"x","schema":schema()}}),
            "unknown field text.format.json_schema",
        ),
        (
            json!({"type":"json_schema","schema":schema()}),
            "text.format.name is required",
        ),
        (
            json!({"type":"json_schema","name":"x"}),
            "text.format.schema is required",
        ),
        (
            json!({"type":"json_schema","name":"x","schema":{},"strict":1}),
            "text.format.strict must be a boolean",
        ),
    ] {
        let error = responses_error(&json!({"input":"hi","text":{"format":format}}));
        assert!(error.contains(needle), "{format}: {error}");
    }
}

#[test]
fn an_unfinished_structured_answer_is_never_reported_complete() {
    let structured = json!({"input":"hi","text":{"format":{"type":"json_object"}}});
    // End of turn during reasoning: no answer at all, so the Response fails.
    let mut frames = Frames::new(responses_reply(&structured));
    let _ = frames.start();
    let _ = frames.event(Event::Reasoning("hmm".into()));
    let (out, terminal) = frames.event(Event::Finished(structured_stats(StopReason::Eos, false)));
    assert!(terminal);
    let events = sse_events(&out);
    let last = events.last().expect("terminal event");
    assert_eq!(last["type"], "response.failed");
    assert_eq!(last["response"]["status"], "failed");
    assert!(
        last["response"]["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("during reasoning"))
    );
    // The buffered body is the same failed document.
    let body = body_bytes(
        responses_reply(&structured),
        vec![
            Signal::Event(Event::Reasoning("hmm".into())),
            Signal::Event(Event::Finished(structured_stats(StopReason::Eos, false))),
        ],
    );
    let document: Value = serde_json::from_slice(&body).expect("a JSON document");
    assert_eq!(document["status"], "failed");
    // A token limit cut the answer short: incomplete, not completed.
    let mut frames = Frames::new(responses_reply(&structured));
    let _ = frames.start();
    let _ = frames.event(Event::Content("{\"a\":".into()));
    let (out, _) = frames.event(Event::Finished(structured_stats(
        StopReason::TokenLimit,
        false,
    )));
    let events = sse_events(&out);
    let last = events.last().expect("terminal event");
    assert_eq!(last["type"], "response.incomplete");
    assert_eq!(last["response"]["output"][0]["status"], "incomplete");

    // Chat: the stream ends on an error frame, never a `stop` finish.
    let mut frames = Frames::new(chat_reply(true));
    let _ = frames.event(Event::Reasoning("hmm".into()));
    let (tail, terminal) = frames.event(Event::Finished(structured_stats(StopReason::Eos, false)));
    assert!(terminal);
    assert_eq!(tail.len(), 1, "{tail:?}");
    let error = parse_data(&tail[0]);
    assert!(
        error["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("during reasoning")),
        "{error}"
    );
    // The buffered Chat body is left truncated, not a short `stop` document.
    let body = body_bytes(
        chat_reply(false),
        vec![
            Signal::Event(Event::Reasoning("hmm".into())),
            Signal::Event(Event::Finished(structured_stats(StopReason::Eos, false))),
        ],
    );
    assert!(serde_json::from_slice::<Value>(&body).is_err());
    // A token limit keeps `length`; a complete document is an ordinary `stop`.
    for (stop, complete, reason) in [
        (StopReason::TokenLimit, false, "length"),
        (StopReason::Eos, true, "stop"),
    ] {
        let mut frames = Frames::new(chat_reply(false));
        let _ = frames.event(Event::Content("{".into()));
        let (tail, _) = frames.event(Event::Finished(structured_stats(stop, complete)));
        assert_eq!(parse_data(&tail[0])["choices"][0]["finish_reason"], reason);
    }
}
