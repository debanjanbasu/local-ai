use std::ops::ControlFlow;
use std::sync::Arc;

use serde_json::{Value, json};

use super::search_pattern;
use crate::api::Event;
use crate::bonsai_tokenizer::BonsaiTokenizer;
use crate::sampler::{Sampler, SamplingParams, verify_greedy_drafts};
use crate::structured::{Grammar, ResponseFormat};
use crate::tools::{ToolCall, ToolCallParser, ToolChoice, ToolDefinition, ToolSet};

const EOS: u32 = 258;
const THINK_END: u32 = 259;
const OPEN: u32 = 260;
const CLOSE: u32 = 261;

/// The tiny byte-level tokenizer with the checkpoint's call tags as special
/// tokens (ids 260 and 261), like the real vocabulary.
fn special_tags() -> BonsaiTokenizer {
    BonsaiTokenizer::tiny_with_specials_for_tests(
        &["<|im_end|>", "</think>", "<tool_call>", "</tool_call>"],
        &["<|im_end|>"],
    )
}

/// The tiny tokenizer without call-tag tokens: tags are literal text only.
fn literal_tags() -> BonsaiTokenizer {
    BonsaiTokenizer::tiny_with_specials_for_tests(&["<|im_end|>", "</think>"], &["<|im_end|>"])
}

fn tool(name: &str, parameters: Value, strict: bool) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: None,
        parameters,
        strict,
    }
}

fn weather(strict: bool) -> ToolDefinition {
    tool(
        "get_weather",
        json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "minLength": 1},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
                "units": {"enum": ["metric", "imperial"]}
            },
            "required": ["city"],
            "additionalProperties": false
        }),
        strict,
    )
}

fn matrix() -> ToolDefinition {
    tool(
        "plot",
        json!({
            "type": "object",
            "properties": {
                "rows": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"}
            },
            "required": ["rows", "tags", "note"],
            "additionalProperties": false
        }),
        true,
    )
}

fn compile(
    tokenizer: &BonsaiTokenizer,
    tools: &[ToolDefinition],
    choice: &ToolChoice,
    parallel: bool,
    format: &ResponseFormat,
    reasoning_end: Option<u32>,
) -> crate::Result<Grammar> {
    let tools = ToolSet::new(tools)?;
    tokenizer.compile_tools_ending(&tools, choice, parallel, format, reasoning_end)
}

fn constrained(tools: &[ToolDefinition], choice: &ToolChoice, parallel: bool) -> Grammar {
    compile(
        &special_tags(),
        tools,
        choice,
        parallel,
        &ResponseFormat::Text,
        None,
    )
    .expect("compile")
}

/// Token ids spelling `text` byte by byte, never as a special token.
fn literal(tokenizer: &BonsaiTokenizer, text: &str) -> Vec<u32> {
    let mut buffer = [0; 4];
    text.chars()
        .flat_map(|character| {
            tokenizer
                .encode(character.encode_utf8(&mut buffer))
                .expect("encode")
        })
        .collect()
}

/// Whether the grammar's next mask allows `token`.
fn allows(grammar: &mut Grammar, token: u32) -> bool {
    grammar.mask().expect("mask").is_allowed(token)
}

/// Select each token under the mask, failing on the first one it forbids.
fn feed(grammar: &mut Grammar, tokens: &[u32]) -> Result<(), String> {
    for (index, &token) in tokens.iter().enumerate() {
        let mask = grammar.mask()?;
        if !mask.is_allowed(token) {
            return Err(format!("token {index} ({token}) is forbidden"));
        }
        if !grammar.select(token, token == EOS, Some(&mask)) {
            return Err(grammar.failure().unwrap_or_default().to_owned());
        }
    }
    Ok(())
}

fn feed_text(grammar: &mut Grammar, tokenizer: &BonsaiTokenizer, text: &str) -> Result<(), String> {
    feed(grammar, &literal(tokenizer, text))
}

/// The ids for a call: special tag tokens around a literal body.
fn call_ids(tokenizer: &BonsaiTokenizer, body: &str) -> Vec<u32> {
    let mut ids = vec![OPEN];
    ids.extend(literal(tokenizer, body));
    ids.push(CLOSE);
    ids
}

const CITY_CALL: &str =
    "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n";

/// Parse decoded output the way the request's event splitter does.
fn parse(tools: &[ToolDefinition], text: &str) -> Result<(Vec<ToolCall>, String), String> {
    let mut parser = ToolCallParser::new(Arc::new(ToolSet::new(tools).expect("tools")));
    let mut calls = Vec::new();
    let mut content = String::new();
    let mut sink = |event| {
        match event {
            Event::ToolCall(call) => calls.push(call),
            Event::Content(text) => content.push_str(&text),
            _ => {}
        }
        ControlFlow::Continue(())
    };
    let _ = parser.feed(text, &mut sink);
    let _ = parser.finish(crate::bonsai_model::StopReason::Eos, &mut sink);
    parser
        .take_failure()
        .map_or_else(|| Ok((calls, content)), Err)
}

#[test]
fn the_default_policy_keeps_generation_unconstrained() {
    let tools = ToolSet::new(&[weather(false)]).expect("tools");
    assert!(!super::needs_grammar(&tools, &ToolChoice::Auto, true, true));
    for (choice, parallel, text) in [
        (ToolChoice::Required, true, true),
        (ToolChoice::None, true, true),
        (ToolChoice::Function("get_weather".into()), true, true),
        (ToolChoice::Auto, false, true),
        (ToolChoice::Auto, true, false),
    ] {
        assert!(super::needs_grammar(&tools, &choice, parallel, text));
    }
    let strict = ToolSet::new(&[weather(true)]).expect("tools");
    assert!(super::needs_grammar(&strict, &ToolChoice::Auto, true, true));
    let none = ToolSet::new(&[]).expect("tools");
    assert!(!super::needs_grammar(
        &none,
        &ToolChoice::Required,
        false,
        false
    ));
}

#[test]
fn a_required_call_cannot_end_or_wander_before_it_is_complete() {
    let tokenizer = special_tags();
    let tools = [weather(true)];
    let mut grammar = constrained(&tools, &ToolChoice::Required, false);
    let x = literal(&tokenizer, "x")[0];
    // High-logit EOS, free text and other special tokens are all out.
    assert!(!allows(&mut grammar, EOS));
    assert!(!allows(&mut grammar, x));
    assert!(!allows(&mut grammar, THINK_END));
    assert!(!allows(&mut grammar, CLOSE));
    // Blank lines may lead into the call, then only the call follows.
    feed_text(&mut grammar, &tokenizer, "\n\n").expect("lead");
    assert!(!allows(&mut grammar, EOS));
    let ids = call_ids(&tokenizer, CITY_CALL);
    for (index, &token) in ids.iter().enumerate() {
        assert!(
            !allows(&mut grammar, EOS),
            "EOS allowed before token {index}"
        );
        feed(&mut grammar, &[token]).expect("call token");
    }
    // Complete: end-of-sequence is now allowed, a second call is not.
    assert!(allows(&mut grammar, EOS));
    assert!(!allows(&mut grammar, OPEN));
    feed_text(&mut grammar, &tokenizer, "\n").expect("trailing whitespace");
    assert!(!allows(&mut grammar, literal(&tokenizer, "<")[0]));
    assert!(!allows(&mut grammar, OPEN));
    feed(&mut grammar, &[EOS]).expect("end");
    let text = tokenizer.decode(&ids, false).expect("decode");
    let (calls, content) = parse(&tools, &text).expect("parse");
    assert_eq!(content, "");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, json!({"city": "Paris"}));
}

#[test]
fn the_sampler_never_takes_a_forbidden_high_logit_token() {
    let tokenizer = special_tags();
    let tools = [weather(true)];
    for seed in 0..8 {
        let mut sampler = Sampler::new(
            262,
            SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                eos_tokens: vec![EOS],
                seed,
            },
        );
        sampler.constrain(constrained(&tools, &ToolChoice::Required, false));
        assert!(!sampler.selects_argmax());
        for &token in &call_ids(&tokenizer, CITY_CALL) {
            // Every special token outranks the intended one; none is ever
            // allowed except the tags at the two ends of the call.
            let mut logits = vec![0.0; 262];
            logits[EOS as usize] = 1000.0;
            logits[THINK_END as usize] = 950.0;
            logits[OPEN as usize] = 900.0;
            logits[CLOSE as usize] = 850.0;
            logits[token as usize] = 50.0;
            let selected = sampler.sample(&mut logits).expect("sample");
            assert_eq!(selected.token_id, token, "seed {seed}");
            assert!(!selected.is_eos);
        }
        let mut logits = vec![0.0; 262];
        logits[EOS as usize] = 50.0;
        logits[OPEN as usize] = 1000.0;
        assert!(sampler.sample(&mut logits).expect("end").is_eos);
        assert_eq!(sampler.response_format_complete(true), None);
        assert_eq!(sampler.tool_constraints_complete(true), Some(true));
    }
}

#[test]
fn malformed_triggers_names_and_parameters_are_forbidden() {
    let tokenizer = special_tags();
    let tools = [weather(true), matrix()];
    let required = || constrained(&tools, &ToolChoice::Required, true);
    // A misspelled literal trigger.
    let mut trigger = required();
    feed_text(&mut trigger, &tokenizer, "<tool_cal").expect("prefix");
    assert!(!allows(&mut trigger, literal(&tokenizer, "x")[0]));
    feed_text(&mut trigger, &tokenizer, "l>").expect("literal trigger");
    // The literal trigger spells the same call as the special token.
    feed_text(&mut trigger, &tokenizer, CITY_CALL).expect("body");
    feed_text(&mut trigger, &tokenizer, "</tool_call>").expect("literal close");
    assert!(allows(&mut trigger, EOS));

    let rejects = |prefix: &str, next: &str| {
        let mut grammar = required();
        feed(&mut grammar, &[OPEN]).expect("open");
        feed_text(&mut grammar, &tokenizer, prefix).expect("prefix");
        let next = literal(&tokenizer, next);
        assert!(
            feed(&mut grammar, &next).is_err(),
            "{prefix:?} then {:?} was accepted",
            tokenizer.decode(&next, false)
        );
    };
    // Unknown or misspelled names, and header layout.
    rejects("\n<function=get_w", "x");
    rejects("\n<function=", "unknown>");
    rejects("", "<function=get_weather>");
    rejects("\n<function=get_weather>", "<parameter=city>");
    // Missing required parameter, unknown parameter, out of order, duplicate.
    rejects("\n<function=get_weather>\n", "</function>");
    rejects("\n<function=get_weather>\n", "<parameter=town>");
    rejects("\n<function=get_weather>\n", "<parameter=days>");
    rejects(
        "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
        "<parameter=city>",
    );
    rejects(
        "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n",
        "<parameter=days>",
    );
    // Values: empty city (minLength), out-of-range or non-integer days,
    // an enum value outside the enum, JSON for a raw string is just text.
    rejects(
        "\n<function=get_weather>\n<parameter=city>\n",
        "\n</parameter>",
    );
    rejects(
        "\n<function=get_weather>\n<parameter=city>\nP\n</parameter>\n<parameter=days>\n",
        "15",
    );
    rejects(
        "\n<function=get_weather>\n<parameter=city>\nP\n</parameter>\n<parameter=days>\n",
        "\"3\"",
    );
    rejects(
        "\n<function=get_weather>\n<parameter=city>\nP\n</parameter>\n<parameter=units>\n",
        "kelvin",
    );
    // Delimiters inside a raw value.
    rejects(
        "\n<function=get_weather>\n<parameter=city>\nA",
        "</parameter>",
    );
    rejects(
        "\n<function=get_weather>\n<parameter=city>\nA",
        "</tool_call>",
    );
    // A truncated call has no closing tag yet: no end-of-sequence.
    let mut truncated = required();
    feed(&mut truncated, &[OPEN]).expect("open");
    feed_text(
        &mut truncated,
        &tokenizer,
        "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
    )
    .expect("body");
    assert!(!allows(&mut truncated, EOS));
}

#[test]
fn optional_parameters_utf8_raw_strings_and_nested_json_round_trip() {
    let tokenizer = special_tags();
    let tools = [weather(true), matrix()];
    let cases = [
        (
            "\n<function=get_weather>\n<parameter=city>\nZoë 😀\nline two </parameter \"x\" <tool_call> {}\n</parameter>\n<parameter=units>\nimperial\n</parameter>\n</function>\n",
            json!({"city": "Zoë 😀\nline two </parameter \"x\" <tool_call> {}", "units": "imperial"}),
        ),
        (
            "\n<function=get_weather>\n<parameter=city>\n\n\n</parameter>\n<parameter=days>\n14\n</parameter>\n</function>\n",
            json!({"city": "\n", "days": 14}),
        ),
        (
            "\n<function=plot>\n<parameter=note>\n[1, 2]\n</parameter>\n<parameter=rows>\n[[1, -2], [], [3]]\n</parameter>\n<parameter=tags>\n[\"</tool_call>\", \"a\\n</parameter>\", \"é\"]\n</parameter>\n</function>\n",
            json!({"tags": ["</tool_call>", "a\n</parameter>", "é"], "note": "[1, 2]", "rows": [[1, -2], [], [3]]}),
        ),
    ];
    for (body, expected) in cases {
        let mut grammar = constrained(&tools, &ToolChoice::Required, false);
        let ids = call_ids(&tokenizer, body);
        // UTF-8 splits across byte tokens: a tag byte is refused mid-character.
        feed(&mut grammar, &ids).unwrap_or_else(|error| panic!("{body:?}: {error}"));
        feed(&mut grammar, &[EOS]).expect("end");
        let text = tokenizer.decode(&ids, false).expect("decode");
        let (calls, _) = parse(&tools, &text).unwrap_or_else(|error| panic!("{text:?}: {error}"));
        assert_eq!(calls.len(), 1, "{text:?}");
        assert_eq!(calls[0].arguments, expected);
    }
    // Mid-character, nothing but the continuation byte is allowed.
    let mut grammar = constrained(&tools, &ToolChoice::Required, false);
    feed(&mut grammar, &[OPEN]).expect("open");
    feed_text(
        &mut grammar,
        &tokenizer,
        "\n<function=get_weather>\n<parameter=city>\n",
    )
    .expect("prefix");
    let accent = literal(&tokenizer, "é");
    feed(&mut grammar, &accent[..1]).expect("lead byte");
    assert!(!allows(&mut grammar, literal(&tokenizer, "\n")[0]));
    assert!(allows(&mut grammar, accent[1]));
    // JSON uses the template's separators exactly.
    let mut grammar = constrained(&tools, &ToolChoice::Required, false);
    feed(&mut grammar, &[OPEN]).expect("open");
    feed_text(
        &mut grammar,
        &tokenizer,
        "\n<function=plot>\n<parameter=note>\nn\n</parameter>\n<parameter=rows>\n[]\n</parameter>\n<parameter=tags>\n[\"a\"",
    )
    .expect("prefix");
    assert!(feed_text(&mut grammar, &tokenizer, ",\"b\"").is_err());
}

#[test]
fn at_most_one_call_bans_a_second_trigger_and_parallel_allows_it() {
    let tokenizer = special_tags();
    let tools = [weather(true)];
    for parallel in [false, true] {
        let mut grammar = constrained(&tools, &ToolChoice::Auto, parallel);
        feed_text(&mut grammar, &tokenizer, "Checking.\n\n").expect("preamble");
        feed(&mut grammar, &call_ids(&tokenizer, CITY_CALL)).expect("first call");
        assert!(allows(&mut grammar, EOS));
        feed_text(&mut grammar, &tokenizer, "\n").expect("gap");
        assert_eq!(allows(&mut grammar, OPEN), parallel);
        assert_eq!(allows(&mut grammar, literal(&tokenizer, "<")[0]), parallel);
        // Text after a call is never allowed in the constrained layout.
        assert!(!allows(&mut grammar, literal(&tokenizer, "D")[0]));
        if parallel {
            feed(&mut grammar, &call_ids(&tokenizer, CITY_CALL)).expect("second call");
            assert!(allows(&mut grammar, EOS));
        }
    }
    // The literal spelling of a second trigger is banned too.
    let mut grammar = constrained(&tools, &ToolChoice::Auto, false);
    feed(&mut grammar, &call_ids(&tokenizer, CITY_CALL)).expect("call");
    assert!(feed_text(&mut grammar, &tokenizer, "<tool_call>").is_err());
}

#[test]
fn auto_free_text_never_contains_the_trigger_outside_a_call() {
    for tokenizer in [special_tags(), literal_tags()] {
        let specials = tokenizer.vocab_size() > 260;
        let tools = [weather(false)];
        let mut text = compile(
            &tokenizer,
            &tools,
            &ToolChoice::Auto,
            false,
            &ResponseFormat::Text,
            None,
        )
        .expect("compile");
        // Plain text may end at any point.
        feed_text(&mut text, &tokenizer, "No call: <tool_cal").expect("text");
        assert!(allows(&mut text, EOS));
        // Special tokens are never text.
        assert!(!allows(&mut text, THINK_END));
        if specials {
            assert!(!allows(&mut text, CLOSE));
        }
        // Completing the literal trigger starts a call: only its header follows.
        feed_text(&mut text, &tokenizer, "l>").expect("trigger");
        assert!(!allows(&mut text, EOS));
        assert!(feed_text(&mut text, &tokenizer, " more text").is_err());
        feed_text(
            &mut text,
            &tokenizer,
            "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        )
        .expect("non-strict call body");
        assert!(allows(&mut text, EOS));
    }
    // A non-strict body is framed but its arguments are checked afterwards.
    let tokenizer = special_tags();
    let tools = [weather(false)];
    let mut grammar = constrained(&tools, &ToolChoice::Required, false);
    let ids = call_ids(
        &tokenizer,
        "\n<function=get_weather>\n<parameter=town>\nParis\n</parameter>\n</function>\n",
    );
    feed(&mut grammar, &ids).expect("framed call");
    let text = tokenizer.decode(&ids, false).expect("decode");
    assert!(parse(&tools, &text).is_err());
    // But it must still name a configured tool and close `</function>`.
    let mut grammar = constrained(&tools, &ToolChoice::Required, false);
    feed(&mut grammar, &[OPEN]).expect("open");
    assert!(feed_text(&mut grammar, &tokenizer, "\n<function=other>").is_err());
    let mut grammar = constrained(&tools, &ToolChoice::Required, false);
    feed(&mut grammar, &[OPEN]).expect("open");
    feed_text(
        &mut grammar,
        &tokenizer,
        "\n<function=get_weather>\nanything",
    )
    .expect("body");
    assert!(!allows(&mut grammar, CLOSE));
}

#[test]
fn tool_choice_none_and_named_restrict_the_answer() {
    let tokenizer = special_tags();
    let tools = [weather(true), matrix()];
    let mut none = constrained(&tools, &ToolChoice::None, true);
    assert!(!allows(&mut none, OPEN));
    feed_text(&mut none, &tokenizer, "Hi <tool_call").expect("text");
    assert!(!allows(&mut none, literal(&tokenizer, ">")[0]));
    assert!(allows(&mut none, EOS));

    let mut named = constrained(&tools, &ToolChoice::Function("plot".into()), true);
    feed(&mut named, &[OPEN]).expect("open");
    assert!(feed_text(&mut named, &tokenizer, "\n<function=get_weather>").is_err());
    let mut named = constrained(&tools, &ToolChoice::Function("plot".into()), true);
    feed(&mut named, &[OPEN]).expect("open");
    feed_text(&mut named, &tokenizer, "\n<function=plot>").expect("named header");
}

#[test]
fn a_response_format_is_the_final_answer_branch_beside_the_call() {
    let tokenizer = special_tags();
    let tools = [weather(true)];
    let format = ResponseFormat::JsonSchema(json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
    }));
    let auto =
        || compile(&tokenizer, &tools, &ToolChoice::Auto, false, &format, None).expect("compile");
    let mut answer = auto();
    assert!(!allows(&mut answer, EOS));
    assert!(!allows(&mut answer, literal(&tokenizer, "x")[0]));
    feed_text(&mut answer, &tokenizer, "\n\n{\"answer\":\"Sunny\"").expect("final");
    assert!(!allows(&mut answer, EOS));
    feed_text(&mut answer, &tokenizer, "}").expect("close");
    assert!(allows(&mut answer, EOS));
    assert!(!allows(&mut answer, OPEN));
    // Schema violations in the final answer are refused.
    let mut wrong = auto();
    assert!(feed_text(&mut wrong, &tokenizer, "{\"other\"").is_err());
    // The call branch still works, and needs no final answer after it.
    let mut call = auto();
    feed_text(&mut call, &tokenizer, "\n").expect("lead");
    feed(&mut call, &call_ids(&tokenizer, CITY_CALL)).expect("call");
    assert!(allows(&mut call, EOS));
    // With no tool call allowed, only the final answer remains.
    let mut none =
        compile(&tokenizer, &tools, &ToolChoice::None, true, &format, None).expect("compile");
    assert!(!allows(&mut none, OPEN));
    feed_text(&mut none, &tokenizer, "{\"answer\":\"x\"}").expect("final");
    assert!(allows(&mut none, EOS));
    // An invalid format still fails as a response_format error.
    let error = compile(
        &tokenizer,
        &tools,
        &ToolChoice::Auto,
        true,
        &ResponseFormat::JsonSchema(json!("object")),
        None,
    )
    .err()
    .expect("invalid format");
    assert!(
        error.to_string().contains("invalid response_format"),
        "{error}"
    );
}

#[test]
fn reasoning_stays_free_but_cannot_end_a_required_call() {
    let tokenizer = special_tags();
    let tools = [weather(true)];
    let mut grammar = compile(
        &tokenizer,
        &tools,
        &ToolChoice::Required,
        false,
        &ResponseFormat::Text,
        Some(THINK_END),
    )
    .expect("compile");
    assert!(grammar.masking());
    assert!(!grammar.answering());
    assert!(!allows(&mut grammar, EOS));
    assert!(allows(&mut grammar, OPEN));
    feed_text(&mut grammar, &tokenizer, "plan <tool_call> x").expect("reasoning");
    feed(&mut grammar, &[OPEN, THINK_END]).expect("reasoning ends");
    assert!(grammar.answering());
    assert!(!allows(&mut grammar, EOS));
    assert!(!allows(&mut grammar, literal(&tokenizer, "x")[0]));
    feed_text(&mut grammar, &tokenizer, "\n\n").expect("lead");
    feed(&mut grammar, &call_ids(&tokenizer, CITY_CALL)).expect("call");
    assert!(allows(&mut grammar, EOS));

    // Auto keeps the old reasoning behaviour: end-of-sequence is the model's.
    let mut auto = compile(
        &tokenizer,
        &tools,
        &ToolChoice::Auto,
        false,
        &ResponseFormat::Text,
        Some(THINK_END),
    )
    .expect("compile");
    assert!(!auto.masking());
    assert!(auto.select(EOS, true, None));
    assert!(auto.tools_complete());
    assert!(!auto.response_format);
}

#[test]
fn speculative_rows_across_the_reasoning_end_equal_sequential_selections() {
    let tokenizer = special_tags();
    let tools = [weather(true)];
    let x = literal(&tokenizer, "x")[0];
    let mut intended = vec![x, THINK_END];
    intended.extend(literal(&tokenizer, "\n"));
    intended.extend(call_ids(&tokenizer, CITY_CALL));
    let compiled = || {
        compile(
            &tokenizer,
            &tools,
            &ToolChoice::Required,
            false,
            &ResponseFormat::Text,
            Some(THINK_END),
        )
        .expect("compile")
    };
    // Logits favour EOS and the opening tag over the intended token.
    let logits = |row: usize| {
        let mut logits = vec![0.0; 262];
        logits[EOS as usize] = 100.0;
        logits[OPEN as usize] = 90.0;
        logits[intended[row] as usize] = 50.0;
        logits
    };
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        eos_tokens: vec![EOS],
        seed: 1,
    };
    let mut sequential = Sampler::new(262, params.clone());
    sequential.constrain(compiled());
    let expected = (0..intended.len())
        .map(|row| sequential.sample(&mut logits(row)).expect("sequential"))
        .collect::<Vec<_>>();
    // Row 0 is reasoning, where only EOS is masked: the opening tag wins.
    assert_eq!(expected[0].token_id, OPEN);
    assert!(expected.iter().all(|sample| !sample.is_eos));
    for drafted in [2, 4, 7] {
        let mut speculative = Sampler::new(262, params.clone());
        speculative.constrain(compiled());
        let mut drafts = expected[..drafted]
            .iter()
            .map(|sample| sample.token_id)
            .collect::<Vec<_>>();
        // A draft the grammar refuses right after the reasoning delimiter.
        if drafted > 3 {
            drafts[2] = x;
        }
        let verified = verify_greedy_drafts(&mut speculative, &drafts, |row, target| {
            target.sample(&mut logits(row))
        })
        .expect("verification");
        let accepted = verified.accepted;
        assert_eq!(verified.samples, expected[..=accepted], "drafted {drafted}");
        for (row, sample) in expected.iter().enumerate().skip(accepted + 1) {
            assert_eq!(
                speculative.sample(&mut logits(row)).expect("continue"),
                *sample,
                "drafted {drafted} row {row}"
            );
        }
        assert!(speculative.grammar_failure().is_none());
    }
}

#[test]
#[allow(clippy::too_many_lines)] // One row per refused setting.
fn unenforceable_settings_fail_early_with_the_reason() {
    let tokenizer = special_tags();
    let strict = |parameters: Value| tool("f", parameters, true);
    let closed = |properties: Value| {
        strict(json!({"type": "object", "properties": properties, "additionalProperties": false}))
    };
    for (definition, reason) in [
        (
            strict(json!({"type": "object", "properties": {"a": {"type": "string"}}})),
            "additionalProperties",
        ),
        (
            closed(json!({"a": {"type": "string", "format": "email"}})),
            "\"format\"",
        ),
        (
            closed(json!({"a": {"type": ["string", "integer"]}})),
            "ambiguous",
        ),
        (closed(json!({"a": {}})), "declare \"type\""),
        (closed(json!({"a": {"enum": ["x", 1]}})), "mixing strings"),
        (
            closed(json!({"a": {"type": "string", "pattern": "^a|b$"}})),
            "alternation",
        ),
        (
            closed(json!({"a": {"type": "string", "pattern": "a(?=b)"}})),
            "only (?:...)",
        ),
        (
            closed(json!({"a": {"type": "string", "pattern": "\\bword"}})),
            "word boundaries",
        ),
        (
            closed(json!({"a": {"type": "string", "pattern": "a^b"}})),
            "anchors",
        ),
        (
            closed(json!({"a": {"type": "string", "enum": ["ab"], "minLength": 3}})),
            "no raw value",
        ),
        (
            closed(json!({"a": {"type": "string", "const": "x</parameter>"}})),
            "no raw value",
        ),
        (
            closed(json!({"a": {"type": "string", "maxLength": 100_000}})),
            "maxLength",
        ),
        (
            closed(json!({"a": {"type": "string", "minLength": 3, "maxLength": 2}})),
            "minLength",
        ),
        (
            closed(json!({"a": {"$ref": "#/properties/b"}, "b": {"type": "string"}})),
            "$defs or definitions",
        ),
        (
            closed(
                json!({"a": {"type": "object", "properties": {"b": {"$ref": "#/properties/a"}}}}),
            ),
            "$ref",
        ),
        (
            closed(json!({"a": {"type": "array", "uniqueItems": true}})),
            "uniqueItems",
        ),
        (
            closed(json!({"a": {"type": "integer", "x-guidance": {}}})),
            "x-guidance",
        ),
        (
            strict(
                json!({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false, "allOf": [{}]}),
            ),
            "\"allOf\"",
        ),
        (
            strict(
                json!({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false, "minProperties": 1}),
            ),
            "minProperties",
        ),
        (
            strict(
                json!({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false, "dependentRequired": {"a": []}}),
            ),
            "dependentRequired",
        ),
    ] {
        let error = compile(
            &tokenizer,
            std::slice::from_ref(&definition),
            &ToolChoice::Auto,
            true,
            &ResponseFormat::Text,
            None,
        )
        .err()
        .unwrap_or_else(|| panic!("{} compiled", definition.parameters));
        let message = error.to_string();
        assert!(
            matches!(error, crate::Error::InvalidArgument(_)) && message.contains(reason),
            "{}: {message}",
            definition.parameters
        );
        assert!(
            message.contains("invalid tool constraints: strict tool \"f\""),
            "{message}"
        );
        // The tool still renders and counts: only constraining it fails.
        assert!(ToolSet::new(std::slice::from_ref(&definition)).is_ok());
    }
    // Choices that cannot apply.
    assert!(crate::tools::check_choice(&[], &ToolChoice::Required).is_err());
    assert!(
        crate::tools::check_choice(&[weather(false)], &ToolChoice::Function("nope".into()))
            .is_err()
    );
    assert!(crate::tools::check_choice(&[], &ToolChoice::None).is_ok());
}

#[test]
fn supported_string_constraints_are_enforced_exactly() {
    let tokenizer = special_tags();
    let definition = tool(
        "code",
        json!({
            "type": "object",
            "properties": {
                "country": {"type": "string", "pattern": "^[A-Z]{2}$"},
                "id": {"type": ["string", "null"], "pattern": "\\d-\\d", "maxLength": 5},
                "mode": {"type": "string", "const": "a.b|c"}
            },
            "required": ["country", "id", "mode"],
            "additionalProperties": false,
            "$schema": "https://json-schema.org/draft/2020-12/schema"
        }),
        true,
    );
    let tools = [definition];
    let header = "\n<function=code>\n<parameter=country>\n";
    let accepted = [
        "FR\n</parameter>\n<parameter=id>\nx1-2\n</parameter>\n<parameter=mode>\na.b|c\n</parameter>\n</function>\n",
        "US\n</parameter>\n<parameter=id>\n0-0\n</parameter>\n<parameter=mode>\na.b|c\n</parameter>\n</function>\n",
    ];
    for body in accepted {
        let mut grammar = constrained(&tools, &ToolChoice::Required, false);
        let ids = call_ids(&tokenizer, &format!("{header}{body}"));
        feed(&mut grammar, &ids).unwrap_or_else(|error| panic!("{body:?}: {error}"));
        let text = tokenizer.decode(&ids, false).expect("decode");
        let (calls, _) = parse(&tools, &text).expect("parse");
        assert_eq!(calls.len(), 1);
    }
    for (prefix, next) in [
        ("", "F1"),
        ("FRA", ""),
        ("FR\n</parameter>\n<parameter=id>\n", "1-a"),
        ("FR\n</parameter>\n<parameter=id>\n", "١-٢"),
        ("FR\n</parameter>\n<parameter=id>\n", "xxx1-2"),
        (
            "FR\n</parameter>\n<parameter=id>\n1-2\n</parameter>\n<parameter=mode>\n",
            "aXb",
        ),
    ] {
        let mut grammar = constrained(&tools, &ToolChoice::Required, false);
        feed(&mut grammar, &[OPEN]).expect("open");
        feed_text(&mut grammar, &tokenizer, header).expect("header");
        let attempt = format!("{prefix}{next}\n</parameter>");
        assert!(
            feed_text(&mut grammar, &tokenizer, &attempt).is_err(),
            "{attempt:?} was accepted"
        );
    }
}

#[test]
fn patterns_translate_with_search_semantics() {
    assert_eq!(search_pattern("^ab$").expect("anchored"), "(?:ab)");
    assert_eq!(
        search_pattern("a|b").expect("search"),
        "(?s:.*)(?:a|b)(?s:.*)"
    );
    assert_eq!(
        search_pattern(r"^\d/.$").expect("ascii"),
        r"(?:[0-9]\/[^\n\r\x{2028}\x{2029}])"
    );
    assert!(search_pattern("^a|b").is_err());
    assert!(search_pattern("(?i)a").is_err());
    assert!(search_pattern(r"(a)\1").is_err());
    assert!(
        search_pattern("[a&&b]")
            .expect("literal ampersands")
            .contains(r"\&\&")
    );
}

/// The real checkpoint tokenizer: the call tags are its special tokens 248058
/// and 248059, which the prompt also uses, and the grammar accepts the call
/// the way the template tokenizes it.
#[test]
#[ignore = "requires the pinned model (CPU only)"]
fn real_tokenizer_constrains_calls_in_its_own_tokens() {
    use crate::bonsai::BonsaiPackage;
    let package = BonsaiPackage::open(crate::bonsai::DEFAULT_BONSAI_GGUF).expect("package");
    let tokenizer = BonsaiTokenizer::from_package(&package).expect("tokenizer");
    let eos = tokenizer.eos_ids()[0];
    let tools = [weather(true)];
    let text = format!("<tool_call>{CITY_CALL}</tool_call>");
    let ids = tokenizer.encode(&text).expect("encode");
    assert_eq!(ids.first(), Some(&248_058));
    assert_eq!(ids.last(), Some(&248_059));
    let started = std::time::Instant::now();
    let mut required = compile(
        &tokenizer,
        &tools,
        &ToolChoice::Required,
        false,
        &ResponseFormat::Text,
        Some(248_069),
    )
    .expect("compile");
    eprintln!("first compile (tables + grammar): {:?}", started.elapsed());
    assert!(!required.mask().expect("mask").is_allowed(eos));
    let think = tokenizer.encode("plan</think>\n\n").expect("think");
    let feed_real = |grammar: &mut Grammar, ids: &[u32]| -> Result<(), String> {
        for &token in ids {
            let mask = grammar.mask()?;
            if !mask.is_allowed(token) {
                return Err(format!("{token} forbidden"));
            }
            assert!(grammar.select(token, token == eos, Some(&mask)));
        }
        Ok(())
    };
    feed_real(&mut required, &think).expect("reasoning");
    let started = std::time::Instant::now();
    feed_real(&mut required, &ids).expect("call");
    eprintln!(
        "{} masked call selections: {:?} each",
        ids.len(),
        started.elapsed() / ids.len() as u32
    );
    assert!(required.mask().expect("mask").is_allowed(eos));
    let decoded = tokenizer.decode(&ids, false).expect("decode");
    let (calls, _) = parse(&tools, &decoded).expect("parse");
    assert_eq!(calls[0].arguments, json!({"city": "Paris"}));

    // Free text under `auto`: the cost of the global trigger exclusion.
    let mut auto = compile(
        &tokenizer,
        &tools,
        &ToolChoice::Auto,
        false,
        &ResponseFormat::Text,
        None,
    )
    .expect("compile");
    let prose = tokenizer
        .encode("Paris is sunny today, so no lookup is needed; <tool_cal is not a call.")
        .expect("prose");
    let started = std::time::Instant::now();
    feed_real(&mut auto, &prose).expect("prose");
    eprintln!(
        "{} masked free-text selections: {:?} each",
        prose.len(),
        started.elapsed() / prose.len() as u32
    );
    let mask = auto.mask().expect("mask");
    assert!(mask.is_allowed(eos));
    assert!(mask.is_allowed(248_058));
    assert!(!mask.is_allowed(248_059));
    assert!(!mask.is_allowed(248_069));
}

#[test]
fn a_strict_tool_without_parameters_takes_an_empty_call_only() {
    let tokenizer = special_tags();
    let tools = [tool("ping", Value::Null, true)];
    let mut grammar = constrained(&tools, &ToolChoice::Required, false);
    let ids = call_ids(&tokenizer, "\n<function=ping>\n</function>\n");
    feed(&mut grammar, &ids).expect("empty call");
    assert!(allows(&mut grammar, EOS));
    let text = tokenizer.decode(&ids, false).expect("decode");
    let (calls, _) = parse(&tools, &text).expect("parse");
    assert_eq!(calls[0].arguments, json!({}));
    let mut grammar = constrained(&tools, &ToolChoice::Required, false);
    feed(&mut grammar, &[OPEN]).expect("open");
    assert!(feed_text(&mut grammar, &tokenizer, "\n<function=ping>\n<parameter=x>").is_err());
}
