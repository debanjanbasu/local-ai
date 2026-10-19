//! CPU-only protocol tests for tool calling and the Responses API: request
//! parsing, SSE framing and buffered documents, all driven by synthetic engine
//! events so no model or GPU is touched.

use std::sync::Arc;

use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{
    ChatRequest, Event, GenerationStats, PrefillProgress, ReasoningEffort, ResponseFormat, Signal,
    Stats, ToolCall, ToolCallDelta, ToolChoice,
};

use super::chunked::{Body, Flow, absorb};
use super::models::{Catalog, Installed};
use super::prepare;
use super::request::prepare_generation;
use super::response::{Protocol, Reply};
use super::responses::{ResponsesState, prepare_responses};
use super::sse::Frames;

fn chat_request(body: &Value, thinking: bool) -> crate::Result<ChatRequest> {
    Ok(prepare_generation(body.to_string().as_bytes(), thinking)?.request)
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
        "max_completion_tokens":20,
        "prompt_cache_key":"cache",
        "safety_identifier":"user-1"
    });
    let request = chat_request(&body, true).expect("valid chat request");
    assert_eq!(request.max_tokens, 20);
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
fn chat_session_is_prompt_cache_key_alone() {
    let request =
        chat_request(&json!({"messages":[user()],"prompt_cache_key":"k"}), true).expect("valid");
    assert_eq!(request.session.as_deref(), Some("k"));
    assert_eq!(
        chat_request(&json!({"messages":[user()]}), true)
            .expect("valid")
            .session,
        None
    );
}

/// Members the specification deprecates are honoured where they mean
/// something here, and accepted where they change nothing.
#[test]
fn chat_honours_deprecated_parameters() {
    let with = |extra: Value| {
        let mut body = json!({"messages":[user()]});
        body.as_object_mut()
            .expect("object")
            .extend(extra.as_object().expect("object").clone());
        chat_request(&body, true)
    };
    assert_eq!(with(json!({"max_tokens":7})).expect("valid").max_tokens, 7);
    assert_eq!(
        with(json!({"max_tokens":7,"max_completion_tokens":9}))
            .expect("valid")
            .max_tokens,
        9,
        "the current member wins"
    );
    assert_eq!(with(json!({"seed":3})).expect("valid").sampling.0.seed, 3);
    assert_eq!(
        with(json!({"seed":-1})).expect("valid").sampling.0.seed,
        u64::MAX,
        "a negative seed keeps its 64 bits"
    );
    assert!(chat_error(&json!({"messages":[user()],"seed":1.5})).contains("integer"));
    assert_eq!(
        with(json!({"session_id":"s"}))
            .expect("valid")
            .session
            .as_deref(),
        Some("s")
    );
    assert_eq!(
        with(json!({"session_id":"s","prompt_cache_key":"k"}))
            .expect("valid")
            .session
            .as_deref(),
        Some("k")
    );
    for ignored in [
        json!({"user":"u"}),
        json!({"prompt_cache_retention":"24h"}),
        json!({"functions":null,"function_call":null}),
    ] {
        assert!(with(ignored.clone()).is_ok(), "{ignored}");
    }
}

/// `functions` and `function_call` are function tools and a tool choice,
/// with one call at most, and a legacy history (an assistant
/// `function_call` answered by a `function` message) is a call and its
/// result.
#[test]
fn chat_maps_deprecated_functions_onto_tools() {
    let request = chat_request(
        &json!({
            "messages":[
                user(),
                {"role":"assistant","content":null,"function_call":{"name":"weather","arguments":"{\"city\":\"Oslo\"}"}},
                {"role":"function","name":"weather","content":"12C"},
                {"role":"user","content":"and tomorrow?"}
            ],
            "functions":[{"name":"weather","parameters":{"type":"object"}}],
            "function_call":{"name":"weather"}
        }),
        true,
    )
    .expect("valid");
    assert_eq!(request.tools[0].name, "weather");
    assert_eq!(request.tool_choice, ToolChoice::Function("weather".into()));
    assert!(!request.parallel_tool_calls);
    let assistant = &request.messages[1];
    assert_eq!(assistant.tool_calls[0].name, "weather");
    assert_eq!(assistant.tool_calls[0].arguments, json!({"city":"Oslo"}));
    let result = &request.messages[2];
    assert_eq!(result.role, "tool");
    assert_eq!(
        result.tool_call_id.as_deref(),
        Some(assistant.tool_calls[0].id.as_str())
    );
    assert_eq!(result.content, "12C");
    for (choice, expected) in [
        (json!("none"), ToolChoice::None),
        (json!("auto"), ToolChoice::Auto),
    ] {
        let request = chat_request(
            &json!({"messages":[user()],"functions":[{"name":"f"}],"function_call":choice}),
            true,
        )
        .expect("valid");
        assert_eq!(request.tool_choice, expected);
    }
    for (body, needle) in [
        (
            json!({"messages":[user()],"functions":[{"name":"f"}],"tools":[{"type":"function","function":{"name":"g"}}]}),
            "not both",
        ),
        (
            json!({"messages":[user()],"function_call":"auto"}),
            "tool_choice",
        ),
        (
            json!({"messages":[user(),{"role":"function","name":"f","content":"x"}]}),
            "answers no assistant function_call",
        ),
        (
            json!({"messages":[user()],"functions":[{"name":"f"}],"function_call":{"name":"g"}}),
            "not declared",
        ),
    ] {
        let error = chat_error(&body);
        assert!(error.contains(needle), "{needle}: {error}");
    }
}

/// vLLM- and llama.cpp-style members harnesses send to local servers.
#[test]
fn chat_accepts_local_server_extensions() {
    let request = chat_request(
        &json!({"messages":[user()],"frequency_penalty":0.5,"repetition_penalty":1.3,
                "chat_template_kwargs":{"enable_thinking":false}}),
        true,
    )
    .expect("valid");
    assert!((request.sampling.0.repetition_penalty - 1.3).abs() < f32::EPSILON);
    assert!(
        !request.thinking,
        "enable_thinking false turns reasoning off"
    );
    let request = chat_request(
        &json!({"messages":[user()],"reasoning_effort":"low",
                "chat_template_kwargs":{"enable_thinking":true}}),
        true,
    )
    .expect("valid");
    assert_eq!(
        request.thinking.then_some(request.effort),
        Some(ReasoningEffort::Low)
    );
}

#[test]
fn chat_reasoning_effort_runs_at_the_nearest_template_level() {
    let effort = |effort: &str, server: bool| {
        chat_request(
            &json!({"messages":[user()],"reasoning_effort":effort}),
            server,
        )
        .map(|request| request.thinking.then_some(request.effort))
    };
    for (requested, level) in [
        ("none", None),
        ("minimal", Some(ReasoningEffort::Low)),
        ("low", Some(ReasoningEffort::Low)),
        ("medium", Some(ReasoningEffort::Medium)),
        ("high", Some(ReasoningEffort::XHigh)),
        ("xhigh", Some(ReasoningEffort::XHigh)),
    ] {
        assert_eq!(
            effort(requested, true).expect(requested),
            level,
            "{requested}"
        );
    }
    assert_eq!(effort("none", false).expect("none without thinking"), None);
    for requested in ["low", "xhigh"] {
        let error = effort(requested, false).expect_err("operator disabled thinking");
        assert!(error.to_string().contains("--no-thinking"), "{error}");
    }
    for invalid in ["max", "bogus"] {
        let error = effort(invalid, true).expect_err(invalid);
        assert!(error.to_string().contains("reasoning effort"), "{error}");
    }
    assert!(
        chat_request(&json!({"messages":[user()]}), false).is_ok_and(|request| !request.thinking)
    );
}

#[test]
fn responses_input_images_keep_their_place_and_detail() {
    // "PNG" in base64: the bytes are only decoded by the engine.
    let body = json!({"input":[
        {"role":"user","content":[
            {"type":"input_text","text":"what is "},
            {"type":"input_image","image_url":"data:image/png;base64,UE5H","detail":"high"},
            {"type":"input_text","text":"?"},
        ]},
        {"type":"function_call","call_id":"c1","name":"look","arguments":"{}"},
        {"type":"function_call_output","call_id":"c1","output":[
            {"type":"input_image","image_url":"data:image/png;base64,UE5H"},
        ]},
    ],"tools":[{"type":"function","name":"look","parameters":{"type":"object"}}]});
    let request = prepare_responses(body.to_string().as_bytes(), true)
        .expect("valid")
        .request;
    let images = request
        .messages
        .iter()
        .map(|message| {
            let placed = message.images.iter();
            (
                message.role.as_str(),
                placed
                    .map(|image| (image.offset, image.detail, &*image.data))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        images,
        [
            (
                "user",
                vec![(8, local_engine::ImageDetail::High, &b"PNG"[..])]
            ),
            ("assistant", vec![]),
            (
                "tool",
                vec![(0, local_engine::ImageDetail::Auto, &b"PNG"[..])]
            ),
        ]
    );
    // file_id images need the Files store, and assistants show no images.
    for (content, role, needle) in [
        (
            json!([{"type":"input_image","file_id":"file-1"}]),
            "user",
            "Files API",
        ),
        (json!([{"type":"input_image"}]), "user", "exactly one"),
        (
            json!([{"type":"input_image","image_url":"data:image/png;base64,UE5H"}]),
            "assistant",
            "assistant",
        ),
    ] {
        let error = responses_error(&json!({"input":[{"role":role,"content":content}]}));
        assert!(error.contains(needle), "{content}: {error}");
    }
}

#[test]
fn responses_reasoning_context_and_effort_reach_the_engine() {
    let prepare = |reasoning: Value| {
        let body = json!({"input":"hi","reasoning":reasoning});
        let request = prepare_responses(body.to_string().as_bytes(), true)
            .expect("valid")
            .request;
        (
            request.thinking.then_some(request.effort),
            request.preserve_thinking,
        )
    };
    assert_eq!(
        prepare(json!({"effort":"minimal","context":"current_turn"})),
        (Some(ReasoningEffort::Low), false)
    );
    assert_eq!(
        prepare(json!({"effort":"medium","context":"all_turns"})),
        (Some(ReasoningEffort::Medium), true)
    );
    assert_eq!(
        prepare(json!({"context":"auto"})),
        (Some(ReasoningEffort::XHigh), true)
    );
}

#[test]
fn allowed_tools_choices_reach_the_engine_on_both_apis() {
    let chat_tools = json!([
        {"type":"function","function":{"name":"weather"}},
        {"type":"function","function":{"name":"time"}}
    ]);
    let chat = |choice: Value| {
        chat_request(
            &json!({"messages":[user()],"tools":chat_tools,"tool_choice":choice}),
            true,
        )
        .map(|request| request.tool_choice)
    };
    assert_eq!(
        chat(
            json!({"type":"allowed_tools","allowed_tools":{"mode":"required","tools":[
                {"type":"function","function":{"name":"time"}}
            ]}})
        )
        .expect("allowed"),
        ToolChoice::Allowed {
            tools: vec!["time".into()],
            required: true
        }
    );
    for invalid in [
        json!({"type":"allowed_tools","allowed_tools":{"tools":[]}}),
        json!({"type":"allowed_tools","allowed_tools":{"mode":"always","tools":[{"type":"function","function":{"name":"time"}}]}}),
        json!({"type":"allowed_tools","allowed_tools":{"tools":[{"type":"function","function":{"name":"nope"}}]}}),
    ] {
        assert!(chat(invalid.clone()).is_err(), "{invalid}");
    }
    let body = json!({"input":"hi","tools":[
        {"type":"function","name":"weather"},
        {"type":"custom","name":"patch"}
    ],"tool_choice":{"type":"allowed_tools","mode":"auto","tools":[
        {"type":"function","name":"weather"},{"type":"custom","name":"patch"}
    ]}});
    let request = prepare_responses(body.to_string().as_bytes(), true)
        .expect("valid")
        .request;
    assert_eq!(
        request.tool_choice,
        ToolChoice::Allowed {
            tools: vec!["weather".into(), "patch".into()],
            required: false
        }
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
        (json!({"stop":7}), "stop must be a string or an array"),
        (json!({"stop":["a",1]}), "stop must be a string or an array"),
        (json!({"stop":["a","b","c","d","e"]}), "at most 4 strings"),
        (json!({"logprobs":true}), "logprobs"),
        (json!({"logprobs":5}), "logprobs must be a boolean"),
        (json!({"logit_bias":{"1":2}}), "logit_bias"),
        (json!({"service_tier":"flex"}), "service_tier"),
        (json!({"store":true}), "store"),
        (
            json!({"verbosity":"loud"}),
            "verbosity must be low, medium or high",
        ),
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

/// `stop` is the engine's stop sequences; `verbosity` `low` and `high`
/// end the system turn with an instruction, after the client's own.
#[test]
fn chat_maps_stop_and_verbosity() {
    let with = |extra: Value| {
        let mut body = json!({"messages":[{"role":"system","content":"S"},
            {"role":"developer","content":"D"},user()]});
        body.as_object_mut()
            .expect("object")
            .extend(extra.as_object().expect("object").clone());
        chat_request(&body, true).expect("valid")
    };
    assert_eq!(with(json!({})).stop, Vec::<String>::new());
    assert_eq!(with(json!({"stop":"END"})).stop, ["END"]);
    assert_eq!(
        with(json!({"stop":["a","\n\nUser:","c","d"]})).stop,
        ["a", "\n\nUser:", "c", "d"]
    );
    let roles = |request: &ChatRequest| {
        request
            .messages
            .iter()
            .map(|message| (message.role.clone(), message.content.clone()))
            .collect::<Vec<_>>()
    };
    let plain = roles(&with(json!({})));
    assert_eq!(roles(&with(json!({"verbosity":"medium"}))), plain);
    let mut lengths = Vec::new();
    for level in ["low", "high"] {
        let messages = roles(&with(json!({"verbosity":level})));
        assert_eq!(messages.len(), plain.len() + 1, "{messages:?}");
        assert_eq!(messages[..2], plain[..2], "the client's instructions lead");
        assert_eq!(messages[2].0, "system");
        assert_eq!(messages[3..], plain[2..]);
        lengths.push(messages[2].1.clone());
    }
    assert_ne!(lengths[0], lengths[1]);
    // With no instructions of its own, the instruction is the system turn.
    let bare = chat_request(&json!({"messages":[user()],"verbosity":"low"}), true).expect("valid");
    assert_eq!(roles(&bare)[0], ("system".to_owned(), lengths[0].clone()));
    // A stop sequence that ends the text after calls still reports them.
    assert_eq!(
        super::response::chat_finish_reason(StopReason::Stopped, true),
        "tool_calls"
    );
    assert_eq!(
        super::response::chat_finish_reason(StopReason::Stopped, false),
        "stop"
    );
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

fn chat_reply(include_usage: bool) -> Reply {
    Reply {
        protocol: Protocol::Chat {
            include_usage,
            functions: false,
        },
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

fn delta(id: &str, name: &str, arguments: &str, text: Option<(&str, &str)>) -> Event {
    Event::ToolCallDelta(ToolCallDelta {
        id: id.into(),
        name: name.into(),
        arguments: arguments.into(),
        text: text.map(|(key, text)| (key.into(), text.into())),
    })
}

#[test]
fn streamed_call_arguments_reach_both_apis_and_end_in_the_validated_call() {
    let events = [
        delta("c1", "weather", "{\"city\":\"Os", Some(("city", "Os"))),
        delta("c1", "weather", "lo", Some(("city", "lo"))),
        Event::ToolCall(call("c1", "weather", json!({"city":"Oslo"}))),
    ];
    // Chat: the first fragment names the call, later ones add arguments, and
    // the validated call sends the rest; together they are the arguments.
    let mut frames = Frames::new(chat_reply(false));
    let mut out = Vec::new();
    for event in events {
        out.extend(frames.event(event).0);
    }
    out.extend(frames.event(Event::Finished(stats(StopReason::Eos))).0);
    let calls: Vec<Value> = out
        .iter()
        .map(|frame| parse_data(frame)["choices"][0]["delta"]["tool_calls"][0].clone())
        .filter(|call| !call.is_null())
        .collect();
    assert_eq!(calls.len(), 3, "{out:#?}");
    assert_eq!(calls[0]["id"], "c1");
    assert_eq!(calls[0]["function"]["name"], "weather");
    assert!(calls[1].get("id").is_none() && calls[2].get("id").is_none());
    assert!(calls.iter().all(|call| call["index"] == 0));
    let arguments: String = calls
        .iter()
        .filter_map(|call| call["function"]["arguments"].as_str())
        .collect();
    assert_eq!(arguments, r#"{"city":"Oslo"}"#);
    assert_eq!(
        parse_data(&out[out.len() - 2])["choices"][0]["finish_reason"],
        "tool_calls"
    );
    // Responses: a custom tool streams its raw input; a function call cut
    // short by the token limit ends incomplete with what streamed.
    let mut frames = Frames::new(responses_reply(&json!({"input":"hi","tools":[
        {"type":"custom","name":"apply_patch"},
        {"type":"function","name":"weather"}
    ]})));
    let mut out = frames.start();
    for event in [
        delta(
            "c2",
            "apply_patch",
            "{\"input\":\"*** Be",
            Some(("input", "*** Be")),
        ),
        delta("c2", "apply_patch", "gin\"", Some(("input", "gin"))),
        Event::ToolCall(call("c2", "apply_patch", json!({"input":"*** Begin"}))),
        delta("c3", "weather", "{\"ci", None),
        Event::Finished(stats(StopReason::TokenLimit)),
    ] {
        out.extend(frames.event(event).0);
    }
    let events = sse_events(&out);
    let kinds: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert_eq!(
        kinds[2..],
        [
            "response.output_item.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.incomplete",
        ]
    );
    assert_eq!(events[2]["item"]["input"], "");
    assert_eq!(events[3]["delta"], "*** Be");
    assert_eq!(events[4]["delta"], "gin");
    assert_eq!(events[5]["input"], "*** Begin");
    assert_eq!(events[6]["item"]["call_id"], "c2");
    let output = &events[11]["response"]["output"];
    assert_eq!(output[0]["type"], "custom_tool_call");
    assert_eq!(output[0]["input"], "*** Begin");
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["status"], "incomplete");
    assert_eq!(output[1]["arguments"], "{\"ci");
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
fn chat_stream_error_is_one_terminal_server_error_frame() {
    let mut failed = Frames::new(Reply::new(
        Protocol::Chat {
            include_usage: false,
            functions: false,
        },
        "m".into(),
    ));
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

/// A request that declared deprecated `functions` reads its call where such
/// a client looks: `function_call` in the stream and the message, finishing
/// with `function_call`.
#[test]
fn chat_answers_deprecated_functions_in_their_own_shape() {
    let reply = || Reply {
        protocol: Protocol::Chat {
            include_usage: false,
            functions: true,
        },
        id: "chatcmpl-1".into(),
        created: 5,
        model: "m".into(),
    };
    let mut frames = Frames::new(reply());
    let mut out = Vec::new();
    for event in [
        delta("c1", "weather", "{\"city\":\"Os", Some(("city", "Os"))),
        delta("c1", "weather", "lo", Some(("city", "lo"))),
        Event::ToolCall(call("c1", "weather", json!({"city":"Oslo"}))),
        Event::Finished(stats(StopReason::Eos)),
    ] {
        out.extend(frames.event(event).0);
    }
    let deltas: Vec<Value> = out
        .iter()
        .map(|frame| parse_data(frame)["choices"][0].clone())
        .collect();
    assert_eq!(
        deltas[0]["delta"],
        json!({"role":"assistant","function_call":{"name":"weather","arguments":"{\"city\":\"Os"}})
    );
    assert_eq!(
        deltas[1]["delta"],
        json!({"function_call":{"arguments":"lo"}})
    );
    assert_eq!(
        deltas[2]["delta"],
        json!({"function_call":{"arguments":"\"}"}})
    );
    assert_eq!(deltas[3]["finish_reason"], "function_call");
    let signals = vec![
        Signal::Event(Event::ToolCall(call(
            "c1",
            "weather",
            json!({"city":"Oslo"}),
        ))),
        Signal::Event(Event::Finished(stats(StopReason::Eos))),
    ];
    let document: Value =
        serde_json::from_slice(&body_bytes(reply(), signals)).expect("valid JSON");
    let choice = &document["choices"][0];
    assert_eq!(choice["finish_reason"], "function_call");
    assert_eq!(
        choice["message"]["function_call"],
        json!({"name":"weather","arguments":"{\"city\":\"Oslo\"}"})
    );
    assert!(choice["message"].get("tool_calls").is_none(), "{choice}");
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

/// The members the specification deprecates on `POST /v1/responses` are
/// accepted: `session_id` is the cache key, `generate_summary` asks for a
/// summary, and `user`, `prompt_cache_retention` and either `truncation`
/// change nothing (the engine never drops input).
#[test]
fn responses_accept_deprecated_parameters() {
    let prepared =
        |body: Value| prepare_responses(body.to_string().as_bytes(), true).expect("valid request");
    for extra in [
        json!({"user":"u"}),
        json!({"prompt_cache_retention":"24h"}),
        json!({"truncation":"disabled"}),
        json!({"truncation":"auto"}),
    ] {
        let mut body = json!({"input":"hi"});
        body.as_object_mut()
            .expect("object")
            .extend(extra.as_object().expect("object").clone());
        let _ = prepared(body);
    }
    assert!(responses_error(&json!({"input":"hi","truncation":"middle"})).contains("truncation"));
    assert_eq!(
        prepared(json!({"input":"hi","session_id":"s"}))
            .request
            .session
            .as_deref(),
        Some("s")
    );
    assert!(
        prepared(json!({"input":"hi","reasoning":{"generate_summary":"concise"}}))
            .echo
            .summarizes()
    );
    let penalties =
        prepared(json!({"input":"hi","presence_penalty":0.5,"frequency_penalty":0.25,"seed":9}))
            .request
            .sampling
            .0;
    assert!((penalties.presence_penalty - 0.5).abs() < f32::EPSILON);
    assert!((penalties.repetition_penalty - 1.25).abs() < f32::EPSILON);
    assert_eq!(penalties.seed, 9);
    let mut frames = Frames::new(responses_reply(&json!({"input":"hi","user":"u"})));
    let _ = frames.start();
    let (out, _) = frames.event(Event::Finished(stats(StopReason::Eos)));
    let response = &sse_events(&out).last().expect("terminal").clone()["response"];
    assert!(response.get("user").is_none(), "{response}");
    assert_eq!(response["truncation"], "disabled");
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
            json!({"input":"hi","tools":[{"type":"code_interpreter","container":{"type":"auto"}}]}),
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
            json!({"input":"hi","reasoning":{"effort":"max"}}),
            "reasoning effort",
        ),
        (
            json!({"input":"hi","reasoning":{"context":"some_turns"}}),
            "reasoning.context",
        ),
        (
            json!({"input":"hi","reasoning":{"summary":"verbose"}}),
            "reasoning.summary must be auto, concise or detailed",
        ),
        (
            json!({"input":"hi","include":["reasoning.encrypted_content"]}),
            "include",
        ),
        (
            json!({"input":"hi","text":{"verbosity":"loud"}}),
            "text.verbosity must be low, medium or high",
        ),
        (
            json!({"input":"hi","text":{"format":{"type":"text","extra":1}}}),
            "unknown field text.format.extra",
        ),
        (
            json!({"input":"hi","reasoning":{"effort":7,"vendor":1}}),
            "invalid JSON request",
        ),
        (
            json!({"input":"hi","temperature":"hot"}),
            "invalid JSON request",
        ),
        (
            json!({"input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]}),
            "must be a base64 data URL",
        ),
        (
            json!({"input":[{"role":"user","content":[{"type":"input_video","video_url":"x"}]}]}),
            "a video URL must be a base64 data URL",
        ),
        (
            json!({"input":[{"role":"user","content":[{"type":"input_video"}]}]}),
            "exactly one of video_url and file_id",
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
        "background":false,"include":[],"text":{"format":{"type":"text"}},
        "tool_choice":"auto","parallel_tool_calls":true,"service_tier":"default",
        "reasoning":{"effort":"xhigh","summary":null},"conversation":null,"client_metadata":null});
    assert!(prepare_responses(accepted.to_string().as_bytes(), true).is_ok());
    // Members this server does not know are ignored, as on Chat, at the top
    // level and inside the objects it reads.
    let vendor = json!({"input":"hi","frobnicate":1,"client_metadata":[],"extra_body":{"x":1},
        "reasoning":{"effort":"low","vendor":true},"text":{"verbosity":"low","vendor":1},
        "stream_options":{"vendor":1}});
    let prepared = prepare_responses(vendor.to_string().as_bytes(), true).expect("ignored");
    assert_eq!(prepared.request.effort, ReasoningEffort::Low);
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

    // Without --reasoning-key, encrypted reasoning is a semantic this server
    // lacks, and the request fails instead of being answered as if honoured.
    let mut encrypted = first.clone();
    encrypted["include"] = json!(["reasoning.encrypted_content"]);
    let error = responses_error(&encrypted);
    assert!(error.contains("include"), "{error:?}");
    // A reasoning summary is a second generation after the first.
    let mut summary = first;
    summary["reasoning"]["summary"] = json!("auto");
    assert_eq!(responses_error(&summary), "");
}

#[test]
fn only_current_generation_apis_are_routed() {
    use axum::http::Method;
    assert!(matches!(
        super::generation_api(&Method::POST, "/v1/chat/completions"),
        Some(super::Api::Chat)
    ));
    for path in ["/v1/responses", "/v1/responses/input_tokens"] {
        assert!(matches!(
            super::generation_api(&Method::POST, path),
            Some(super::Api::Responses)
        ));
    }
    // The legacy Completions API is a 404 like any unknown route.
    for (method, path) in [
        (Method::POST, "/v1/completions"),
        (Method::GET, "/v1/completions"),
        (Method::POST, "/v1/engines/local/completions"),
    ] {
        assert!(
            super::generation_api(&method, path).is_none(),
            "{method} {path}"
        );
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
    assert_eq!(request.messages.len(), 1);
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
            include_usage: true,
            functions: false
        }
    ));
}

#[test]
fn reply_ids_are_prefixed_and_unique() {
    let a = Reply::new(
        Protocol::Chat {
            include_usage: false,
            functions: false,
        },
        "m".into(),
    );
    let b = Reply::new(
        Protocol::Chat {
            include_usage: false,
            functions: false,
        },
        "m".into(),
    );
    assert!(a.id.starts_with("chatcmpl-") && a.id.len() == "chatcmpl-".len() + 32);
    assert_ne!(a.id, b.id);
}

#[test]
fn models_report_the_admitted_context() {
    let models = Catalog {
        bonsai: "bonsai",
        context: 32768,
        created: 1234,
        installed: Installed::default(),
    }
    .list();
    assert_eq!(models["data"][0]["id"], "bonsai");
    assert_eq!(models["data"][0]["created"], 1234);
    assert_eq!(models["data"][0]["context_length"], 32768);
    assert_eq!(models["data"][0]["max_model_len"], 32768);
    assert_eq!(models["data"].as_array().map(Vec::len), Some(1));
}

#[test]
fn models_list_every_installed_model() {
    let installed = Installed {
        embeddings: true,
        moderation: true,
        speech: true,
        transcription: true,
    };
    let models = Catalog {
        bonsai: "bonsai",
        context: 32768,
        created: 1234,
        installed,
    }
    .list();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|model| model["id"].as_str().expect("id"))
        .collect();
    assert_eq!(
        ids,
        [
            "bonsai",
            "embeddinggemma-2",
            "embeddinggemma-2-moderation-1",
            "kokoro-82m",
            "whisper-large-v3"
        ]
    );
    for model in models["data"].as_array().expect("data") {
        assert_eq!(model["object"], "model");
        assert_eq!(model["created"], 1234);
        assert_eq!(model["owned_by"], "local");
    }
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
        json!({"format":{"type":"text"},"verbosity":"medium"})
    );
    // `low` and `high` are echoed and asked for after the instructions.
    let plain_messages = prepare_responses(plain.to_string().as_bytes(), true)
        .expect("valid")
        .request
        .messages;
    for level in ["low", "high"] {
        let body = json!({"input":"hi","instructions":"sys","text":{"verbosity":level}});
        assert_eq!(terminal(&body)["response"]["text"]["verbosity"], level);
        let messages = prepare_responses(body.to_string().as_bytes(), true)
            .expect("valid")
            .request
            .messages;
        assert_eq!(messages.len(), plain_messages.len() + 2, "{messages:?}");
        assert_eq!(
            (messages[0].content.as_str(), messages[1].role.as_str()),
            ("sys", "system")
        );
        assert_eq!(messages[2].content, "hi");
    }
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
        // A stop sequence the client asked for cut the document short.
        (StopReason::Stopped, false, "stop"),
    ] {
        let mut frames = Frames::new(chat_reply(false));
        let _ = frames.event(Event::Content("{".into()));
        let (tail, _) = frames.event(Event::Finished(structured_stats(stop, complete)));
        assert_eq!(parse_data(&tail[0])["choices"][0]["finish_reason"], reason);
    }
}
