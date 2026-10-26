use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{Event, PrefillProgress, Sampling, Signal};
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

/// A chat request whose user turn is several prefill chunks long (three
/// words per `PREFILL_CHUNK` token), unique to `marker` so the prompt cache cannot
/// already hold it, without reasoning.
fn long_prompt(marker: u128, max_tokens: usize) -> super::ChatRequest {
    super::ChatRequest {
        messages: vec![message(
            "user",
            &format!(
                "{} {marker}",
                "word ".repeat(3 * crate::resources::PREFILL_CHUNK)
            ),
        )],
        max_tokens,
        sampling: Sampling::default(),
        thinking: false,
        effort: crate::ReasoningEffort::default(),
        reasoning_instruction: false,
        preserve_thinking: true,
        session: None,
        tools: Vec::new(),
        tool_choice: crate::ToolChoice::Auto,
        parallel_tool_calls: true,
        response_format: crate::ResponseFormat::default(),
        stop: Vec::new(),
    }
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
    // The prompt is several chunks long, and one generated
    // token is enough because the assertion is about what arrives before it.
    let request = long_prompt(marker, 1);
    let mut events = handle.chat(request).expect("queue");
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
    let request = long_prompt(marker, 8);
    let mut events = handle.chat(request).expect("queue");
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
    let mut splitter = super::EventSplitter::new(thinking, tools, Vec::new());
    let mut events = Vec::new();
    let mut sink = |event| {
        events.push(event);
        std::ops::ControlFlow::Continue(())
    };
    for piece in pieces {
        if splitter.emit(piece, Vec::new(), &mut sink).is_break() {
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

/// Log-probabilities go just before the content their tokens produced, and
/// only answer tokens are reported: not reasoning, `</think>`, the dropped
/// whitespace around the answer and calls, call markup, or text from a stop
/// sequence on.
#[test]
fn logprobs_precede_the_content_their_tokens_produced() {
    let pieces = [
        "Plan",
        " it",
        "</think>",
        "\n\n",
        "Sure",
        ".",
        "\n\n",
        "<tool_call>",
        "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
        "</tool_call>",
        "\n",
        "Done",
        " now",
        " STOP",
        " after",
    ];
    let mut splitter = super::EventSplitter::new(true, Some(weather_tools()), vec!["STOP".into()]);
    let mut events = Vec::new();
    let mut sink = |event| {
        events.push(event);
        std::ops::ControlFlow::Continue(())
    };
    for (index, piece) in pieces.iter().enumerate() {
        let token = crate::TokenLogprob {
            token: index as u32,
            logprob: -1.0,
            bytes: piece.as_bytes().to_vec(),
            top: Vec::new(),
        };
        if splitter.emit(piece, vec![token], &mut sink).is_break() {
            break;
        }
    }
    let _ = splitter.close(crate::bonsai_model::StopReason::Eos, &mut sink);
    let mut reported = Vec::new();
    let mut content = String::new();
    for (index, event) in events.iter().enumerate() {
        match event {
            Event::Logprobs(tokens) => {
                assert!(matches!(events.get(index + 1), Some(Event::Content(_))));
                reported.extend(tokens.iter().map(|token| token.token));
            }
            Event::Content(text) => content.push_str(text),
            _ => {}
        }
    }
    assert_eq!(content, "Sure.Done now ");
    assert_eq!(reported, [4, 5, 11, 12, 13]);
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
        splitter: super::EventSplitter::new(false, tools, Vec::new()),
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
    assert!(delivery.emit("Checking.\n\n", Vec::new()));
    assert!(delivery.emit(CALL, Vec::new()));
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
        delivery.splitter = super::EventSplitter::new(thinking, None, Vec::new());
        assert!(delivery.emit(text, Vec::new()));
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
    let invalid = CALL.replace("get_weather", "get weather");
    let truncated = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar";
    for (output, stops) in [(invalid.as_str(), true), (truncated, false)] {
        let (mut delivery, _receiver) = delivery(Some(weather_tools()));
        assert!(delivery.emit("Checking. ", Vec::new()));
        assert_eq!(delivery.emit(output, Vec::new()), !stops);
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
            assert!(delivery.emit(&output, Vec::new()));
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

/// Stop sequences end the visible answer at their first match, excluding
/// it, and never stream part of one; reasoning and tool-call markup are not
/// matched, and a refusal that was not a match is the consumer cancelling.
#[test]
#[allow(clippy::too_many_lines)]
fn stop_sequences_end_the_visible_answer_without_streaming_part_of_one() {
    use crate::bonsai_model::StopReason;
    struct Case {
        thinking: bool,
        tools: bool,
        stop: &'static [&'static str],
        pieces: &'static [&'static str],
        /// The engine's own stop reason when every piece was accepted.
        engine: StopReason,
        content: &'static [&'static str],
        reasoning: &'static str,
        calls: usize,
        reason: StopReason,
    }
    let cases = [
        // A match split across pieces: its first half is held, never sent.
        Case {
            thinking: false,
            tools: false,
            stop: &["\n\nUser:"],
            pieces: &["Hello\n\nUs", "er: next", " ignored"],
            engine: StopReason::Eos,
            content: &["Hello"],
            reasoning: "",
            calls: 0,
            reason: StopReason::Stopped,
        },
        // A held start that does not become a sequence is released whole.
        Case {
            thinking: false,
            tools: false,
            stop: &["\n\nUser:"],
            pieces: &["a\n\nUs", "e it"],
            engine: StopReason::Eos,
            content: &["a", "\n\nUse it"],
            reasoning: "",
            calls: 0,
            reason: StopReason::Eos,
        },
        // The earliest of several sequences wins, across multibyte text.
        Case {
            thinking: false,
            tools: false,
            stop: &["", "END", "→E"],
            pieces: &["x →", "END"],
            engine: StopReason::TokenLimit,
            content: &["x "],
            reasoning: "",
            calls: 0,
            reason: StopReason::Stopped,
        },
        // Reasoning is never matched.
        Case {
            thinking: true,
            tools: false,
            stop: &["Paris"],
            pieces: &["Paris?</think>\n\nOK"],
            engine: StopReason::Eos,
            content: &["OK"],
            reasoning: "Paris?",
            calls: 0,
            reason: StopReason::Eos,
        },
        // Nor are a call's arguments: the call stands.
        Case {
            thinking: false,
            tools: true,
            stop: &["Paris"],
            pieces: &["Checking.\n\n", CALL],
            engine: StopReason::Eos,
            content: &["Checking."],
            reasoning: "",
            calls: 1,
            reason: StopReason::Eos,
        },
        // A match before a call means the call is never made.
        Case {
            thinking: false,
            tools: true,
            stop: &["king"],
            pieces: &["Checking.\n\n", CALL],
            engine: StopReason::Eos,
            content: &["Chec"],
            reasoning: "",
            calls: 0,
            reason: StopReason::Stopped,
        },
        // Text after a call is matched; the call made before it stands.
        Case {
            thinking: false,
            tools: true,
            stop: &["Done"],
            pieces: &[CALL, "\nAll Done."],
            engine: StopReason::Eos,
            content: &["All "],
            reasoning: "",
            calls: 1,
            reason: StopReason::Stopped,
        },
        // Text the tool parser holds to the end is matched when flushed.
        Case {
            thinking: false,
            tools: true,
            stop: &["<tool"],
            pieces: &["Done <tool"],
            engine: StopReason::Eos,
            content: &["Done", " "],
            reasoning: "",
            calls: 0,
            reason: StopReason::Stopped,
        },
        // A refusal without a match is the consumer stopping.
        Case {
            thinking: false,
            tools: false,
            stop: &[],
            pieces: &["partial"],
            engine: StopReason::Stopped,
            content: &["partial"],
            reasoning: "",
            calls: 0,
            reason: StopReason::Cancelled,
        },
    ];
    for case in cases {
        let tools = case.tools.then(weather_tools);
        let (mut delivery, _receiver) = delivery(None);
        delivery.splitter = super::EventSplitter::new(
            case.thinking,
            tools,
            case.stop.iter().map(|&stop| stop.to_owned()).collect(),
        );
        let mut engine = case.engine;
        for piece in case.pieces {
            if !delivery.emit(piece, Vec::new()) {
                // What the engine does when a piece is refused.
                engine = StopReason::Stopped;
                break;
            }
        }
        let mut output = generation();
        output.stop_reason = engine;
        delivery.complete(Ok(output));
        let events = outbox_events(&delivery);
        let label = format!("{:?} in {:?}: {events:?}", case.stop, case.pieces);
        let content: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, case.content, "{label}");
        let reasoning: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, case.reasoning, "{label}");
        let calls = events
            .iter()
            .filter(|event| matches!(event, Event::ToolCall(_)))
            .count();
        assert_eq!(calls, case.calls, "{label}");
        assert!(
            matches!(events.last(), Some(Event::Finished(stats)) if stats.stop_reason == case.reason),
            "{label}"
        );
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
        effort: crate::ReasoningEffort::default(),
        reasoning_instruction: false,
        preserve_thinking: true,
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
        stop: Vec::new(),
    }
}

/// Encoding without a vision tower, cancellation or context limit.
const fn no_vision() -> super::Encoding<'static> {
    super::Encoding {
        vision: None,
        activity: None,
        cancel: None,
        context: usize::MAX,
    }
}

#[test]
fn a_chat_count_is_the_length_of_the_prompt_generation_would_prefill() {
    let tokenizer = crate::bonsai_tokenizer::BonsaiTokenizer::tiny_for_tests();
    for thinking in [false, true] {
        let request = tool_history(thinking);
        let rendered = Engine::render_chat_with_tools(&request.messages, thinking, &request.tools)
            .expect("render");
        let tool_set = crate::tools::ToolSet::new(&request.tools).expect("tools");
        let super::ChatPrompt { ids, tools, .. } =
            super::chat_prompt(&tokenizer, &no_vision(), &request, tool_set).expect("prepare");
        assert!(tools.is_some());
        assert_eq!(ids, tokenizer.encode(&rendered).expect("encode"));
        assert_eq!(
            super::count_chat(&tokenizer, None, &request).expect("count"),
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
            super::count_chat(&tokenizer, None, &with_tools).expect("with tools")
                > super::count_chat(&tokenizer, None, &without_tools).expect("without tools")
        );
    }
    // Reasoning mode changes the rendered system prompt and suffix.
    assert_ne!(
        super::count_chat(&tokenizer, None, &tool_history(true)).expect("thinking"),
        super::count_chat(&tokenizer, None, &tool_history(false)).expect("plain")
    );
}

#[test]
fn an_invalid_chat_fails_counting_exactly_as_it_fails_generation() {
    let tokenizer = crate::bonsai_tokenizer::BonsaiTokenizer::tiny_for_tests();
    let mut unanswered = tool_history(true);
    unanswered.messages.remove(3);
    let mut not_object = tool_history(true);
    not_object.messages[2].tool_calls[0].arguments = serde_json::json!("Paris");
    let mut empty = tool_history(false);
    empty.messages.clear();
    for request in [unanswered, not_object, empty] {
        let expected = super::prepare_chat(&request)
            .expect_err("generation rejects it")
            .to_string();
        let counted = super::count_chat(&tokenizer, None, &request)
            .expect_err("counting rejects it")
            .to_string();
        assert_eq!(counted, expected);
    }
}

#[test]
fn images_render_as_the_templates_vision_markers_where_they_sit() {
    let part = |offset, video| super::ChatImage {
        offset,
        data: std::sync::Arc::from(&b"not decoded while rendering"[..]),
        detail: super::ImageDetail::Auto,
        video,
    };
    let mut user = message("user", "Compare  and ");
    user.images = vec![part(8, false), part(13, true), part(13, false)];
    let tools = crate::tools::ToolSet::default();
    let rendered =
        super::render_messages(std::slice::from_ref(&user), None, false, &tools).expect("render");
    let (image, video) = (
        crate::bonsai_vision::IMAGE_PLACEHOLDER,
        crate::bonsai_vision::VIDEO_PLACEHOLDER,
    );
    assert!(
        rendered.contains(&format!("Compare {image} and {video}{image}<|im_end|>")),
        "{rendered}"
    );

    // Images must follow the text in order, and system prompts hold none.
    let mut backwards = user.clone();
    backwards.images.swap(0, 1);
    let mut system = user.clone();
    system.role = "system".into();
    for messages in [vec![backwards], vec![system, message("user", "hi")]] {
        assert!(super::render_messages(&messages, None, false, &tools).is_err());
    }

    // Without the projector, image prompts fail to count as to generate.
    let tokenizer = crate::bonsai_tokenizer::BonsaiTokenizer::tiny_for_tests();
    let mut request = tool_history(false);
    request.messages = vec![user];
    let counted = super::count_chat(&tokenizer, None, &request).expect_err("no projector");
    let Err(generated) = super::chat_prompt(
        &tokenizer,
        &no_vision(),
        &request,
        crate::tools::ToolSet::default(),
    ) else {
        panic!("generation needs the projector");
    };
    assert_eq!(counted.to_string(), generated.to_string());
    assert!(counted.to_string().contains("mmproj"), "{counted}");
}

/// A request holding `parts`, each at the start of one user message.
fn media_request(parts: Vec<super::ChatImage>) -> super::ChatRequest {
    let mut request = tool_history(false);
    request.tools.clear();
    request.messages = vec![super::ChatMessage {
        role: "user".into(),
        content: "Look.".into(),
        images: parts,
        ..super::ChatMessage::default()
    }];
    request
}

fn image_part(data: &std::sync::Arc<[u8]>) -> super::ChatImage {
    super::ChatImage {
        offset: 0,
        data: std::sync::Arc::clone(data),
        detail: super::ImageDetail::Auto,
        video: false,
    }
}

/// A request's media are bounded before any pixel is decoded: at most 64
/// parts, 256 MiB and 512 MP of images, sized from their headers (these
/// PNGs hold no pixel data at all).
#[test]
fn media_beyond_the_request_limits_are_refused_before_decoding() {
    use crate::bonsai_vision::{MAX_REQUEST_MEDIA_BYTES, MAX_REQUEST_PARTS};
    use crate::embedding::vision::tests::png_header;
    let plan = |parts| super::plan_media(&media_request(parts), None).map(|planned| planned.len());
    let small: std::sync::Arc<[u8]> = png_header(64, 64).into();
    assert_eq!(
        plan(vec![image_part(&small); MAX_REQUEST_PARTS]).expect("within"),
        64
    );
    let error = plan(vec![image_part(&small); MAX_REQUEST_PARTS + 1]).expect_err("parts");
    assert!(error.to_string().contains("at most 64 images"), "{error}");
    // 64 MP each: eight are 512 MP, nine are over.
    let large: std::sync::Arc<[u8]> = png_header(8192, 8192).into();
    assert_eq!(plan(vec![image_part(&large); 8]).expect("within"), 8);
    let error = plan(vec![image_part(&large); 9]).expect_err("pixels");
    assert!(error.to_string().contains("pixels together"), "{error}");
    // Bytes are counted before anything is read.
    let bulky: std::sync::Arc<[u8]> = vec![0; MAX_REQUEST_MEDIA_BYTES / 4 + 1].into();
    let error = plan(vec![image_part(&bulky); 4]).expect_err("bytes");
    assert!(error.to_string().contains("bytes together"), "{error}");
}

/// Every chat is rendered and tokenized by the submitting thread, so the
/// worker that steps other generations only admits it, and an invalid one
/// fails the call instead of being queued.
#[test]
fn a_handle_queues_chats_already_tokenized() {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    let tokenizer = crate::bonsai_tokenizer::BonsaiTokenizer::tiny_for_tests();
    let handle = super::EngineHandle {
        sender,
        tokenizer: tokenizer.clone(),
        activity: std::sync::Arc::default(),
        prompt_cache: crate::prompt_cache::Closer::new(),
        vision: None,
        context: 32_768,
    };
    let mut invalid = tool_history(true);
    invalid.messages.clear();
    assert!(handle.chat(invalid).is_err());
    assert!(receiver.try_recv().is_err(), "nothing queued");
    let request = tool_history(true);
    let _events = handle.chat(request.clone()).expect("queued");
    let Ok(super::Job::Chat(job)) = receiver.try_recv() else {
        panic!("a queued chat");
    };
    let (prompt, _) = super::prepare_chat(&request).expect("render");
    assert_eq!(job.prompt.ids, tokenizer.encode(&prompt).expect("encode"));
    assert!(job.prompt.tools.is_some());
}

/// With the pinned model and projector, an image prompt over the context
/// fails at submission before its image is even decoded, and a cancelled
/// one fails before the tower submits anything.
#[test]
#[ignore = "requires the pinned model, the vision projector and the GPU"]
fn image_prompts_fit_the_context_and_observe_cancellation_before_encoding() {
    use crate::bonsai_vision::ImageFile;
    let handle = Engine::open().expect("open engine").into_handle();
    let vision = std::sync::Arc::clone(handle.vision.as_ref().expect("projector installed"));
    let bytes: std::sync::Arc<[u8]> = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/vision/syn_wide_1600x90.jpg"
    ))
    .expect("fixture")
    .into();
    let known = || {
        ImageFile::plan(&bytes, super::ImageDetail::Auto)
            .expect("plan")
            .known()
    };
    let mut request = media_request(vec![image_part(&bytes)]);
    request.thinking = false;
    request.max_tokens = handle.context;
    let Err(error) = handle.chat(request.clone()) else {
        panic!("over the context");
    };
    assert!(matches!(error, crate::Error::ContextOverflow(_)), "{error}");
    assert_eq!(known(), None, "not decoded");

    request.max_tokens = 4;
    let cancel = super::CancelHandle::new();
    cancel.cancel();
    let Err(error) = handle.chat_with_cancel(request.clone(), &cancel) else {
        panic!("cancelled");
    };
    assert!(error.to_string().contains("cancelled"), "{error}");
    let grid = known().expect("decoded");
    assert!(vision.cached(&grid.digest).is_none(), "not encoded");

    let finished = handle
        .chat(request)
        .expect("queued")
        .any(|event| matches!(event, Event::Finished(_)));
    assert!(finished);
    assert!(vision.cached(&grid.digest).is_some(), "encoded");
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
            .1
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
            .1
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
                .1
                .is_some()
        );
        // The prompt never depends on the tool policy.
        let mut default = native.clone();
        default.tool_choice = crate::ToolChoice::Auto;
        default.parallel_tool_calls = true;
        default.tools[0].strict = false;
        assert_eq!(
            super::count_chat(&plain, None, &native).expect("count"),
            super::count_chat(&plain, None, &default).expect("count")
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
    // Deltas report the call while it is generated; the call itself ends it.
    let events = events
        .into_iter()
        .filter(|event| !matches!(event, Event::ToolCallDelta(_)))
        .collect::<Vec<_>>();
    assert!(
        matches!(events.as_slice(), [Event::Reasoning(_), Event::ToolCall(call)]
            if call.arguments == serde_json::json!({"names": ["</tool_call>", "b"]})),
        "{events:?}"
    );
}

/// A request whose prompt and `max_tokens` can never fit the context fails at
/// once, instead of holding the queue until the request ahead of it finishes.
#[test]
#[ignore = "requires the pinned model and the GPU"]
fn a_request_that_can_never_fit_fails_without_waiting_for_the_engine() {
    let handle = Engine::open().expect("open engine").into_handle();
    let mut running = long_prompt(0, 2048);
    running.messages = vec![message(
        "user",
        "Write every number from 1 to 3000, one per line.",
    )];
    running.sampling.0.temperature = 0.0;
    let mut oversized = running.clone();
    oversized.max_tokens = 1 << 40;
    let mut first = handle.chat(running).expect("queue");
    assert!(
        matches!(first.next(), Some(Event::Content(_))),
        "the first request is decoding"
    );
    let mut events = handle.chat(oversized).expect("queue");
    match events.next_timeout(Duration::from_secs(20)) {
        Ok(Some(Event::Error(error))) => assert!(error.contains("exceed"), "{error}"),
        other => panic!("the oversized request was not refused at once: {other:?}"),
    }
    first.cancel_handle().cancel();
}

/// A finished request whose reader stopped reading does not keep the idle
/// worker from taking the next job: its last events wait for room while
/// later requests run.
#[test]
#[ignore = "requires the pinned model and the GPU"]
fn an_unread_finished_request_does_not_hold_up_the_next_one() {
    let handle = Engine::open().expect("open engine").into_handle();
    let request = |max_tokens| {
        let mut request = long_prompt(0, max_tokens);
        request.messages = vec![message(
            "user",
            "Write every number from 1 to 3000, one per line.",
        )];
        request.sampling.0.temperature = 0.0;
        request
    };
    // More events than its channel holds, never read: once it finishes beside
    // the second request, its last events are left undelivered.
    let unread = handle
        .chat(request(4 * crate::runtime::EVENT_BUFFER))
        .expect("queue");
    let second = handle
        .chat(request(8 * crate::runtime::EVENT_BUFFER))
        .expect("queue");
    let mut finished = false;
    for event in second {
        match event {
            Event::Finished(_) => finished = true,
            Event::Error(error) => panic!("the second request failed: {error}"),
            _ => {}
        }
    }
    assert!(finished, "the second request finished");
    let mut third = handle.chat(request(4)).expect("queue");
    match third.next_timeout(Duration::from_secs(60)) {
        Ok(Some(Event::Content(_))) => {}
        other => panic!("the next request did not start: {other:?}"),
    }
    drop(unread);
}
