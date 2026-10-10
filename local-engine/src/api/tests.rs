use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{CompletionRequest, Event, PrefillProgress, Sampling, Signal};
use crate::Engine;

#[test]
fn a_silent_stream_times_out_without_closing_the_generation() {
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    let mut events = super::EventStream {
        receiver,
        progress: std::collections::VecDeque::new(),
        cancel: crate::bonsai_model::CancelToken::new(),
        not_sync: std::marker::PhantomData,
    };
    let (resume, waiting) = std::sync::mpsc::channel();
    let producer = std::thread::spawn(move || {
        // Close after one second if the wait is broken, so this regression
        // fails rather than hanging the suite indefinitely.
        if waiting.recv_timeout(Duration::from_secs(1)).is_ok() {
            sender
                .blocking_send(super::Delivered::Event(Event::Content("ready".into())))
                .expect("stream remains open after a timeout");
        }
    });
    assert!(matches!(
        events.next_timeout(Duration::from_millis(10)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    resume.send(()).expect("resume producer");
    assert!(matches!(events.next(), Some(Event::Content(text)) if text == "ready"));
    assert!(events.next().is_none());
    producer.join().expect("producer");
}

/// Prefill reports every chunk, in order, before the first event exists.
///
/// This is the claim the whole prefill heartbeat rests on: the model emits
/// nothing per token until decode starts, so a long prompt is minutes of silence
/// unless each chunk says it is still working. The prompt is made unique per run
/// because a prompt the cache already holds is prefilled as zero chunks, which
/// would make this vacuously true.
#[test]
#[ignore = "requires the pinned model; a real prefill is the only thing that reports"]
fn prefill_reports_every_chunk_before_the_first_event() {
    let handle = Engine::open().expect("open engine").into_handle();
    let marker = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after the epoch")
        .as_nanos();
    // 400 words is several chunks at `PREFILL_CHUNK` of 128, and one generated
    // token is enough because the assertion is about what arrives before it.
    let request = CompletionRequest {
        prompt: format!("{} {marker}", "word ".repeat(400)),
        max_tokens: 1,
        sampling: Sampling::default(),
        session: None,
        response_format: crate::ResponseFormat::default(),
    };
    let mut events = handle.complete(request).expect("queue");
    let mut boundaries: Vec<PrefillProgress> = Vec::new();
    let first_event = loop {
        match events.next_signal(None).expect("signal") {
            Some(Signal::Progress(progress)) => boundaries.push(progress),
            Some(Signal::Event(event)) => break event,
            None => break Event::Error("engine worker stopped".into()),
        }
    };
    assert!(
        boundaries.len() >= 2,
        "a multi-chunk prompt reported {boundaries:?}"
    );
    assert!(
        boundaries
            .windows(2)
            .all(|pair| pair[1].chunks == pair[0].chunks + 1),
        "chunk counts are not consecutive: {boundaries:?}"
    );
    // Tokens already in the K/V cache count, so the numbers climb and never
    // restart at the reusable boundary the way a per-call counter would.
    assert!(
        boundaries
            .windows(2)
            .all(|pair| pair[1].tokens >= pair[0].tokens),
        "token counts went backwards: {boundaries:?}"
    );
    assert!(
        boundaries.iter().all(|boundary| boundary.chunks >= 1),
        "the first chunk must be 1 rather than 0: {boundaries:?}"
    );
    assert!(
        !matches!(first_event, Event::Error(_)),
        "the request failed: {first_event:?}"
    );
}

/// A boundary is kept for `next_signal`, not dropped by `next_timeout`.
///
/// `next_timeout` is a wait on generated text and its deadline measures silence
/// from generated text, so a prefill boundary neither satisfies it nor resets it:
/// the protection for a long prefill is the *unbounded* first wait a caller has to
/// ask for, which is what `EventStream::next_signal(None)` is. What must hold on
/// this side is that the boundaries stay reachable afterwards, because the server
/// reads them from the same stream it read its events from, and a boundary dropped
/// here would silently disable the prefill heartbeat.
#[test]
#[ignore = "requires the pinned model; a real prefill is what reports"]
fn a_prefill_boundary_survives_a_wait_that_only_wanted_events() {
    let handle = Engine::open().expect("open engine").into_handle();
    let marker = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after the epoch")
        .as_nanos();
    let request = CompletionRequest {
        prompt: format!("{} {marker}", "word ".repeat(400)),
        max_tokens: 8,
        sampling: Sampling::default(),
        session: None,
        response_format: crate::ResponseFormat::default(),
    };
    let mut events = handle.complete(request).expect("queue");
    // Generous on purpose: this wait is allowed to cover the whole prefill, which
    // is the point. Only the *absence* of a deadline is what a caller needs there.
    let budget = Duration::from_secs(3600);
    loop {
        match events.next_timeout(budget) {
            Ok(Some(Event::Finished(_)) | None) => break,
            Ok(Some(_)) => {}
            Err(error) => panic!("an hour-long wait on a healthy generation expired: {error}"),
        }
    }
    // The generation is over and every boundary it reported is still here.
    let mut kept = 0;
    while let Ok(Some(Signal::Progress(_))) = events.next_signal(Some(Duration::from_millis(1))) {
        kept += 1;
    }
    assert!(
        kept >= 2,
        "a multi-chunk prefill left {kept} boundaries for `next_signal`"
    );
}

fn weather_tools() -> std::sync::Arc<crate::tools::ToolSet> {
    let tool = crate::ToolDefinition {
        name: "get_weather".into(),
        description: None,
        parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
        strict: false,
    };
    std::sync::Arc::new(crate::tools::ToolSet::new(&[tool]).expect("tools"))
}

const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

fn split(
    thinking: bool,
    tools: Option<std::sync::Arc<crate::tools::ToolSet>>,
    pieces: &[&str],
) -> (Vec<Event>, Option<String>) {
    let mut splitter = super::EventSplitter::new(thinking, tools);
    let mut events = Vec::new();
    let mut sink = |event| {
        events.push(event);
        std::ops::ControlFlow::Continue(())
    };
    for piece in pieces {
        if splitter.emit(piece, &mut sink).is_break() {
            break;
        }
    }
    if !splitter.failed() {
        let _ = splitter.finish(crate::bonsai_model::StopReason::Eos, &mut sink);
    }
    (events, splitter.take_failure())
}

#[test]
fn tool_calls_are_parsed_only_in_the_answer_of_a_request_with_tools() {
    let thought = format!("plan {CALL}</think>\n\nOK\n\n{CALL}");
    let (events, failure) = split(true, Some(weather_tools()), &[&thought]);
    assert_eq!(failure, None);
    assert!(matches!(&events[0], Event::Reasoning(text) if text.contains("<tool_call>")));
    assert!(matches!(&events[1], Event::Content(text) if text == "OK"));
    assert!(
        matches!(&events[2], Event::ToolCall(call) if call.arguments == serde_json::json!({"city":"Paris"}))
    );
    assert_eq!(events.len(), 3);

    let (events, failure) = split(false, None, &[CALL]);
    assert_eq!(failure, None);
    assert!(matches!(events.as_slice(), [Event::Content(text)] if text == CALL));
}

fn delivery(
    tools: Option<std::sync::Arc<crate::tools::ToolSet>>,
) -> (
    super::Delivery,
    tokio::sync::mpsc::Receiver<super::Delivered>,
) {
    let (events, receiver) = tokio::sync::mpsc::channel(16);
    let delivery = super::Delivery {
        events,
        cancel: crate::bonsai_model::CancelToken::new(),
        splitter: super::EventSplitter::new(false, tools),
        outbox: std::collections::VecDeque::new(),
    };
    (delivery, receiver)
}

fn generation() -> crate::bonsai_model::BonsaiGeneration {
    crate::bonsai_model::BonsaiGeneration {
        text: String::new(),
        token_ids: vec![1],
        stop_reason: crate::bonsai_model::StopReason::Eos,
        stats: crate::GenerationStats::default(),
        cache_source: crate::bonsai_model::PromptCacheSource::None,
    }
}

fn outbox_events(delivery: &super::Delivery) -> Vec<Event> {
    delivery
        .outbox
        .iter()
        .filter_map(|item| match item {
            super::Delivered::Event(event) => Some(event.clone()),
            super::Delivered::Progress(_) => None,
        })
        .collect()
}

#[test]
fn the_worker_streams_a_valid_call_and_then_finishes() {
    let (mut delivery, _receiver) = delivery(Some(weather_tools()));
    assert!(delivery.emit("Checking.\n\n"));
    assert!(delivery.emit(CALL));
    delivery.complete(Ok(generation()));
    let events = outbox_events(&delivery);
    assert!(matches!(&events[0], Event::Content(text) if text == "Checking."));
    assert!(matches!(&events[1], Event::ToolCall(call) if call.name == "get_weather"));
    assert!(matches!(events.last(), Some(Event::Finished(_))));
}

#[test]
fn usage_counts_reasoning_tokens_including_the_delimiter_not_text_chunks() {
    for (thinking, text, tokens, count) in [
        (
            true,
            "why</think>\n\nanswer",
            vec![7, 11, 248_069, 13, 17],
            3,
        ),
        (true, "unfinished thought", vec![7, 11], 2),
        (false, "literal </think>", vec![7, 248_069, 13], 0),
        (true, "", vec![], 0),
    ] {
        let (mut delivery, _receiver) = delivery(None);
        delivery.splitter = super::EventSplitter::new(thinking, None);
        assert!(delivery.emit(text));
        let mut output = generation();
        output.token_ids = tokens;
        delivery.complete(Ok(output));
        let events = outbox_events(&delivery);
        let Some(Event::Finished(stats)) = events.last() else {
            panic!("missing terminal usage: {events:?}");
        };
        assert_eq!(stats.reasoning_tokens, count);
    }
}

#[test]
fn the_worker_reports_invalid_and_truncated_calls_without_finishing() {
    let invalid = CALL.replace("city", "town");
    let truncated = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar";
    for (output, stops) in [(invalid.as_str(), true), (truncated, false)] {
        let (mut delivery, _receiver) = delivery(Some(weather_tools()));
        assert!(delivery.emit("Checking. "));
        assert_eq!(delivery.emit(output), !stops);
        delivery.complete(Ok(generation()));
        let events = outbox_events(&delivery);
        assert!(
            matches!(events.last(), Some(Event::Error(error)) if error.starts_with("invalid tool call")),
            "{events:?}"
        );
        assert!(
            events.iter().all(|event| !matches!(
                event,
                Event::ToolCall(_) | Event::Finished(_) | Event::TokenIds(_)
            )),
            "{events:?}"
        );
    }
}

#[test]
fn a_call_cut_off_by_the_budget_is_dropped_and_the_stop_reason_stands() {
    use crate::bonsai_model::StopReason;
    let strict = crate::ToolDefinition {
        name: "tag".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"names": {"type": "array", "items": {"type": "string"}}},
            "required": ["names"],
            "additionalProperties": false
        }),
        strict: true,
    };
    let strict = std::sync::Arc::new(crate::tools::ToolSet::new(&[strict]).expect("tools"));
    // A forced strict call stopped mid-JSON, and a valid call followed by
    // a second one the budget interrupted.
    let partial = "<tool_call>\n<function=tag>\n<parameter=names>\n[\"a";
    let after_call = format!("{CALL}\n<tool_call>\n<function=get_weather>\n<parameter=ci");
    for (tools, output, calls) in [
        (strict, partial.to_owned(), 0),
        (weather_tools(), after_call, 1),
    ] {
        for reason in [StopReason::TokenLimit, StopReason::Cancelled] {
            let (mut delivery, _receiver) = delivery(Some(std::sync::Arc::clone(&tools)));
            assert!(delivery.emit(&output));
            let mut generation = generation();
            generation.stop_reason = reason;
            delivery.complete(Ok(generation));
            let events = outbox_events(&delivery);
            let emitted = events
                .iter()
                .filter(
                    |event| matches!(event, Event::ToolCall(call) if call.name == "get_weather"),
                )
                .count();
            assert_eq!(emitted, calls, "{events:?}");
            assert!(
                events.iter().all(
                    |event| !matches!(event, Event::Error(_) | Event::Content(_))
                        && !matches!(event, Event::ToolCall(call) if call.name == "tag")
                ),
                "{events:?}"
            );
            assert!(
                matches!(events.last(), Some(Event::Finished(stats)) if stats.stop_reason == reason),
                "{events:?}"
            );
        }
    }
}

fn message(role: &str, content: &str) -> super::ChatMessage {
    super::ChatMessage {
        role: role.into(),
        content: content.into(),
        ..super::ChatMessage::default()
    }
}

fn weather_call() -> crate::ToolCall {
    crate::ToolCall {
        id: "call_1".into(),
        name: "get_weather".into(),
        arguments: serde_json::json!({"city":"Paris"}),
    }
}

/// A conversation exercising every rendered part: system prompt, prior
/// reasoning, a replayed call, its result, and a follow-up question.
fn tool_history(thinking: bool) -> super::ChatRequest {
    super::ChatRequest {
        messages: vec![
            message("system", "Be brief."),
            message("user", "Weather in Paris?"),
            super::ChatMessage {
                role: "assistant".into(),
                reasoning_content: Some("Need the tool.".into()),
                tool_calls: vec![weather_call()],
                ..super::ChatMessage::default()
            },
            super::ChatMessage {
                role: "tool".into(),
                content: "18C and sunny".into(),
                tool_call_id: Some("call_1".into()),
                ..super::ChatMessage::default()
            },
            message("user", "And tomorrow?"),
        ],
        max_tokens: 1,
        sampling: Sampling::default(),
        thinking,
        session: None,
        tools: vec![crate::ToolDefinition {
            name: "get_weather".into(),
            description: Some("Look up the weather.".into()),
            parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
            strict: false,
        }],
        tool_choice: crate::ToolChoice::Auto,
        parallel_tool_calls: true,
        response_format: crate::ResponseFormat::default(),
    }
}

#[test]
fn a_chat_count_is_the_length_of_the_prompt_generation_would_prefill() {
    let tokenizer = crate::bonsai_tokenizer::BonsaiTokenizer::tiny_for_tests();
    for thinking in [false, true] {
        let request = tool_history(thinking);
        let rendered = Engine::render_chat_with_tools(&request.messages, thinking, &request.tools)
            .expect("render");
        let (ids, tools) = super::chat_prompt_ids(&tokenizer, &request).expect("prepare");
        assert!(tools.is_some());
        assert_eq!(ids, tokenizer.encode(&rendered).expect("encode"));
        assert_eq!(
            super::count_chat(&tokenizer, &request).expect("count"),
            ids.len()
        );

        // Tool definitions are part of the prompt, so they are counted.
        let mut without_tools = request.clone();
        without_tools.tools.clear();
        without_tools
            .messages
            .retain(|turn| turn.role == "system" || turn.role == "user");
        let mut with_tools = without_tools.clone();
        with_tools.tools.clone_from(&request.tools);
        assert!(
            super::count_chat(&tokenizer, &with_tools).expect("with tools")
                > super::count_chat(&tokenizer, &without_tools).expect("without tools")
        );
    }
    // Reasoning mode changes the rendered system prompt and suffix.
    assert_ne!(
        super::count_chat(&tokenizer, &tool_history(true)).expect("thinking"),
        super::count_chat(&tokenizer, &tool_history(false)).expect("plain")
    );
}

#[test]
fn an_invalid_chat_fails_counting_exactly_as_it_fails_generation() {
    let tokenizer = crate::bonsai_tokenizer::BonsaiTokenizer::tiny_for_tests();
    let mut unanswered = tool_history(true);
    unanswered.messages.remove(3);
    let mut undefined_tool = tool_history(true);
    undefined_tool.messages[2].tool_calls[0].name = "get_time".into();
    let mut empty = tool_history(false);
    empty.messages.clear();
    for request in [unanswered, undefined_tool, empty] {
        let expected = super::prepare_chat(&request)
            .expect_err("generation rejects it")
            .to_string();
        let counted = super::count_chat(&tokenizer, &request)
            .expect_err("counting rejects it")
            .to_string();
        assert_eq!(counted, expected);
    }
}

#[test]
fn a_completion_count_parses_special_tokens_without_a_template() {
    let tokenizer = crate::bonsai_tokenizer::BonsaiTokenizer::tiny_for_tests();
    let request = |prompt: &str| CompletionRequest {
        prompt: prompt.into(),
        max_tokens: 1,
        sampling: Sampling::default(),
        session: None,
        response_format: crate::ResponseFormat::default(),
    };
    // `abc` is one merged token and `<|x|>` one special token.
    assert_eq!(
        super::count_completion(&tokenizer, &request("abc<|x|>")).expect("count"),
        2
    );
    assert_eq!(
        super::count_completion(&tokenizer, &request("")).expect("empty"),
        0
    );
}

#[test]
fn a_handle_can_count_from_any_thread() {
    const fn shareable<T: Clone + Send + Sync>() {}
    shareable::<super::EngineHandle>();
}

/// The counted prompt equals the prompt generation actually prefilled, for a
/// request with tools, a replayed call and reasoning, through both the
/// synchronous engine and a handle whose worker is otherwise idle.
#[test]
#[ignore = "requires the pinned model and the GPU"]
fn counted_prompt_tokens_match_generation_stats() {
    let engine = Engine::open().expect("open engine");
    let completion = CompletionRequest {
        prompt: "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n".into(),
        max_tokens: 1,
        sampling: Sampling::default(),
        session: None,
        response_format: crate::ResponseFormat::default(),
    };
    // Allow a complete tool call: a one-token budget can stop just after the
    // opening tag, which correctly fails parsing before Finished is emitted.
    let chats = [tool_history(true), tool_history(false)].map(|mut request| {
        request.max_tokens = 128;
        request.sampling.0.temperature = 0.0;
        request
    });
    let engine_counts = chats
        .iter()
        .map(|request| engine.count_chat_tokens(request).expect("engine count"))
        .collect::<Vec<_>>();
    let engine_completion = engine
        .count_completion_tokens(&completion)
        .expect("engine completion count");
    let handle = engine.into_handle();
    let finished = |mut events: super::EventStream| {
        events
            .find_map(|event| match event {
                Event::Finished(stats) => Some(stats.generation.prompt_tokens),
                Event::Error(error) => panic!("generation failed: {error}"),
                _ => None,
            })
            .expect("finished")
    };
    for (request, counted) in chats.into_iter().zip(engine_counts) {
        assert_eq!(handle.count_chat_tokens(&request).expect("count"), counted);
        let generated = finished(handle.chat(request).expect("queue chat"));
        assert_eq!(generated, counted);
    }
    assert_eq!(
        handle
            .count_completion_tokens(&completion)
            .expect("completion count"),
        engine_completion
    );
    let generated = finished(handle.complete(completion).expect("queue completion"));
    assert_eq!(generated, engine_completion);
}

#[test]
fn default_tool_settings_need_no_compiler_and_native_ones_compile_up_front() {
    // No end-of-sequence token: any attempt to compile would fail, so a
    // `None` here proves the default path builds no grammar at all.
    let plain = crate::bonsai_tokenizer::BonsaiTokenizer::tiny_for_tests();
    let mut request = tool_history(true);
    assert!(
        super::chat_grammar(&plain, &request)
            .expect("text")
            .is_none()
    );
    let constrained = crate::bonsai_tokenizer::BonsaiTokenizer::tiny_with_specials_for_tests(
        &["<|im_end|>", "</think>"],
        &["<|im_end|>"],
    );
    // Tools and a response format now ride together in one grammar.
    request.response_format = crate::ResponseFormat::JsonObject;
    assert!(
        super::chat_grammar(&constrained, &request)
            .expect("tools with a format")
            .is_some()
    );
    request.response_format = crate::ResponseFormat::Text;
    for (choice, parallel, strict) in [
        (crate::ToolChoice::Required, true, false),
        (crate::ToolChoice::None, true, false),
        (
            crate::ToolChoice::Function("get_weather".into()),
            true,
            false,
        ),
        (crate::ToolChoice::Auto, false, false),
        (crate::ToolChoice::Auto, true, true),
    ] {
        let mut native = request.clone();
        native.tool_choice = choice;
        native.parallel_tool_calls = parallel;
        native.tools[0].strict = strict;
        native.tools[0].parameters["additionalProperties"] = serde_json::Value::Bool(false);
        assert!(
            super::chat_grammar(&constrained, &native)
                .expect("native")
                .is_some()
        );
        // The prompt never depends on the tool policy.
        let mut default = native.clone();
        default.tool_choice = crate::ToolChoice::Auto;
        default.parallel_tool_calls = true;
        default.tools[0].strict = false;
        assert_eq!(
            super::count_chat(&plain, &native).expect("count"),
            super::count_chat(&plain, &default).expect("count")
        );
    }
    // Choices that cannot apply fail before any compiling.
    let mut unknown = request.clone();
    unknown.tool_choice = crate::ToolChoice::Function("get_time".into());
    let mut toolless = request;
    toolless.tools.clear();
    toolless.messages.truncate(2);
    toolless.tool_choice = crate::ToolChoice::Required;
    for invalid in [unknown, toolless] {
        assert!(matches!(
            super::chat_grammar(&plain, &invalid),
            Err(crate::Error::InvalidArgument(_))
        ));
    }
}

#[test]
fn a_strict_call_is_parsed_by_its_exact_layout_after_reasoning() {
    let tool = crate::ToolDefinition {
        name: "tag".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"names": {"type": "array", "items": {"type": "string"}}},
            "required": ["names"],
            "additionalProperties": false
        }),
        strict: true,
    };
    let tools = std::sync::Arc::new(crate::tools::ToolSet::new(&[tool]).expect("tools"));
    // A JSON string may hold `</tool_call>`: only the strict end marker,
    // which no value can contain, closes the call.
    let call = "<tool_call>\n<function=tag>\n<parameter=names>\n[\"</tool_call>\", \"b\"]\n</parameter>\n</function>\n</tool_call>";
    let text = format!("think</think>\n\n{call}");
    let pieces = text.split_inclusive(['>', '"']).collect::<Vec<_>>();
    let (events, failure) = split(true, Some(tools), &pieces);
    assert_eq!(failure, None);
    assert!(
        matches!(events.as_slice(), [Event::Reasoning(_), Event::ToolCall(call)]
            if call.arguments == serde_json::json!({"names": ["</tool_call>", "b"]})),
        "{events:?}"
    );
}
