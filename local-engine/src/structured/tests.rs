use serde_json::json;

use super::ResponseFormat;
use crate::bonsai_tokenizer::BonsaiTokenizer;
use crate::sampler::{Sampler, SamplingParams, SamplingResult, verify_greedy_drafts};

/// The tiny byte-level tokenizer with an end-of-turn and a reasoning
/// delimiter: 256 byte tokens, `ab`, `abc`, then ids 258 and 259.
fn tokenizer() -> BonsaiTokenizer {
    BonsaiTokenizer::tiny_with_specials_for_tests(&["<|im_end|>", "</think>"], &["<|im_end|>"])
}

const VOCAB: usize = 260;
const EOS: u32 = 258;
const THINK_END: u32 = 259;

fn ids(tokenizer: &BonsaiTokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text).expect("encode")
}

fn id(tokenizer: &BonsaiTokenizer, text: &str) -> u32 {
    let ids = ids(tokenizer, text);
    assert_eq!(ids.len(), 1, "{text:?} is {ids:?}");
    ids[0]
}

fn params(temperature: f32, seed: u64) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        eos_tokens: vec![EOS],
        seed,
    }
}

fn constrained(
    tokenizer: &BonsaiTokenizer,
    format: &ResponseFormat,
    reasoning_end: Option<u32>,
    params: SamplingParams,
) -> Sampler {
    let mut sampler = Sampler::new(VOCAB, params);
    let grammar = tokenizer
        .compile_format_ending(format, reasoning_end)
        .expect("compile")
        .expect("a constraint");
    sampler.constrain(grammar);
    sampler
}

/// Logits ranking `ranked` first to last above every other token.
fn ranking(ranked: &[u32]) -> Vec<f32> {
    let mut logits = vec![0.0; VOCAB];
    for (rank, &token) in ranked.iter().enumerate() {
        logits[token as usize] = 1000.0 - rank as f32;
    }
    logits
}

/// Select `token` greedily while `EOS` and `x` outrank everything else.
fn take(sampler: &mut Sampler, token: u32, x: u32) -> SamplingResult {
    sampler
        .sample(&mut ranking(&[EOS, x, token]))
        .expect("sample")
}

fn object_schema() -> ResponseFormat {
    ResponseFormat::JsonSchema(json!({
        "type": "object",
        "properties": {"s": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["s", "n"],
        "additionalProperties": false
    }))
}

#[test]
fn text_compiles_to_nothing_and_json_object_is_a_constraint() {
    let tokenizer = tokenizer();
    assert!(
        tokenizer
            .compile_format_ending(&ResponseFormat::Text, None)
            .expect("text")
            .is_none()
    );
    assert!(
        tokenizer
            .compile_format_ending(&ResponseFormat::JsonObject, None)
            .expect("object")
            .is_some()
    );
    let local = json!({
        "$defs": {"leaf": {"type": "integer"}},
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/leaf"}}
    });
    assert!(
        tokenizer
            .compile_format_ending(&ResponseFormat::JsonSchema(local), None)
            .expect("document-local $ref")
            .is_some()
    );
}

#[test]
fn invalid_and_unsupported_schemas_fail_before_generation() {
    let tokenizer = tokenizer();
    for schema in [
        json!("object"),
        json!(true),
        json!({"type": "object", "x-guidance": {"lenient": true}}),
        json!({"$ref": "https://example.com/schema.json"}),
        json!({"$ref": "file:///etc/schema.json"}),
        json!({"type": "string", "format": "made-up"}),
        json!({"oneOf": [{"type": "integer"}, {"type": "number"}]}),
        json!({"type": "array", "uniqueItems": true}),
        json!({"type": "array", "contains": {"type": "integer"}}),
        json!({"not": {"type": "string"}}),
        json!({"type": "string", "pattern": "("}),
        json!({"type": "integer", "minimum": 5, "maximum": 1}),
        json!({"type": "wat"}),
    ] {
        let result =
            tokenizer.compile_format_ending(&ResponseFormat::JsonSchema(schema.clone()), None);
        assert!(
            matches!(result, Err(crate::Error::InvalidArgument(_))),
            "{schema} compiled: {:?}",
            result.as_ref().map(Option::is_some)
        );
    }
}

#[test]
fn a_tokenizer_without_end_of_sequence_cannot_be_constrained() {
    let tokenizer = BonsaiTokenizer::tiny_for_tests();
    assert!(
        tokenizer
            .compile_format_ending(&ResponseFormat::JsonObject, None)
            .is_err()
    );
}

#[test]
fn impossible_high_logit_tokens_are_forbidden() {
    let tokenizer = tokenizer();
    let x = id(&tokenizer, "x");
    let brace = id(&tokenizer, "{");
    let whitespace = [" ", "\n", "\r", "\t"].map(|text| id(&tokenizer, text));
    let mut greedy = constrained(
        &tokenizer,
        &ResponseFormat::JsonObject,
        None,
        params(0.0, 1),
    );
    // A masking request never takes a precomputed, unmasked argmax.
    assert!(!greedy.selects_argmax());
    assert!(greedy.greedy_draft().selects_argmax());
    let first = take(&mut greedy, brace, x);
    assert_eq!(
        first,
        SamplingResult {
            token_id: brace,
            is_eos: false
        }
    );
    // Sampling from a flat distribution where forbidden tokens are favoured.
    for seed in 0..32 {
        let mut sampler = constrained(
            &tokenizer,
            &ResponseFormat::JsonObject,
            None,
            SamplingParams {
                top_k: 20,
                top_p: 0.9,
                presence_penalty: 0.5,
                repetition_penalty: 1.2,
                ..params(1.0, seed)
            },
        );
        sampler.observe(&[brace]);
        let mut logits = vec![5.0; VOCAB];
        logits[brace as usize] = 0.0;
        for token in whitespace {
            logits[token as usize] = -1.0;
        }
        let result = sampler.sample(&mut logits).expect("sample");
        assert!(
            result.token_id == brace || whitespace.contains(&result.token_id),
            "seed {seed} selected {result:?}"
        );
        assert!(sampler.grammar_failure().is_none());
    }
}

#[test]
fn eos_is_denied_until_the_grammar_accepts() {
    let tokenizer = tokenizer();
    let x = id(&tokenizer, "x");
    let mut sampler = constrained(&tokenizer, &object_schema(), None, params(0.0, 1));
    for token in ids(&tokenizer, r#"{"s":"ok","n":12"#) {
        let result = take(&mut sampler, token, EOS);
        assert_eq!(
            result,
            SamplingResult {
                token_id: token,
                is_eos: false
            }
        );
    }
    // A token limit here leaves the document incomplete.
    assert_eq!(sampler.response_format_complete(false), Some(false));
    let close = id(&tokenizer, "}");
    assert_eq!(take(&mut sampler, close, x).token_id, close);
    let end = sampler.sample(&mut ranking(&[EOS, x])).expect("end");
    assert_eq!(
        end,
        SamplingResult {
            token_id: EOS,
            is_eos: true
        }
    );
    assert_eq!(sampler.response_format_complete(true), Some(true));
    // A token limit at the same point is still reported as incomplete.
    assert_eq!(sampler.response_format_complete(false), Some(false));
}

#[test]
fn split_utf8_and_escapes_follow_the_grammar_byte_by_byte() {
    let tokenizer = tokenizer();
    let quote = id(&tokenizer, "\"");
    let mut sampler = constrained(&tokenizer, &object_schema(), None, params(0.0, 3));
    let text = "{\"s\":\"é\\u001f\\\"😀\\n\",\"n\":-7}";
    let tokens = ids(&tokenizer, text);
    // Byte-level BPE splits é into two and 😀 into four tokens.
    assert!(tokens.len() > text.chars().count());
    let prefix = ids(&tokenizer, "{\"s\":\"");
    for &token in &prefix {
        assert_eq!(take(&mut sampler, token, EOS).token_id, token);
    }
    let accent = ids(&tokenizer, "é");
    assert_eq!(accent.len(), 2);
    assert_eq!(take(&mut sampler, accent[0], EOS).token_id, accent[0]);
    // Mid-character, a closing quote (or EOS) would leave invalid UTF-8.
    let continuation = sampler
        .sample(&mut ranking(&[EOS, quote, accent[1]]))
        .expect("continuation");
    assert_eq!(continuation.token_id, accent[1]);
    for &token in &tokens[prefix.len() + 2..] {
        assert_eq!(take(&mut sampler, token, EOS).token_id, token);
    }
    let end = sampler.sample(&mut ranking(&[EOS])).expect("end");
    assert!(end.is_eos);
    let decoded = tokenizer.decode(&tokens, false).expect("decode");
    let value: serde_json::Value = serde_json::from_str(&decoded).expect("valid JSON");
    assert_eq!(value, json!({"s": "é\u{1f}\"😀\n", "n": -7}));
    // An escape the grammar does not know is refused.
    let mut sampler = constrained(&tokenizer, &object_schema(), None, params(0.0, 3));
    for token in ids(&tokenizer, "{\"s\":\"\\") {
        assert_eq!(take(&mut sampler, token, EOS).token_id, token);
    }
    let q = id(&tokenizer, "q");
    let escaped = sampler.sample(&mut ranking(&[q, quote])).expect("escape");
    assert_eq!(escaped.token_id, quote);
}

#[test]
fn matchers_are_independent_across_requests_and_threads() {
    let tokenizer = tokenizer();
    let x = id(&tokenizer, "x");
    let quote = id(&tokenizer, "\"");
    let brace = id(&tokenizer, "{");
    let mut ahead = constrained(&tokenizer, &object_schema(), None, params(0.0, 1));
    let mut fresh = constrained(&tokenizer, &object_schema(), None, params(0.0, 1));
    take(&mut ahead, brace, x);
    // `ahead` now needs a key; `fresh` still needs the opening brace.
    assert_eq!(take(&mut ahead, quote, x).token_id, quote);
    assert_eq!(
        fresh
            .sample(&mut ranking(&[quote, brace]))
            .expect("fresh")
            .token_id,
        brace
    );
    let document = r#"{"s":"thread","n":1}"#;
    std::thread::scope(|scope| {
        let workers = (0..4)
            .map(|worker| {
                let tokenizer = tokenizer.clone();
                scope.spawn(move || {
                    let mut sampler = constrained(
                        &tokenizer,
                        &object_schema(),
                        (worker % 2 == 1).then_some(THINK_END),
                        params(0.0, worker),
                    );
                    if worker % 2 == 1 {
                        let thought = sampler.sample(&mut ranking(&[x])).expect("thought");
                        assert_eq!(thought.token_id, x);
                        let end = sampler.sample(&mut ranking(&[THINK_END])).expect("end");
                        assert_eq!(end.token_id, THINK_END);
                    }
                    for token in ids(&tokenizer, document) {
                        assert_eq!(take(&mut sampler, token, EOS).token_id, token);
                    }
                    sampler.sample(&mut ranking(&[EOS])).expect("end")
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            assert!(worker.join().expect("worker").is_eos);
        }
    });
}

#[test]
fn reasoning_is_unconstrained_until_the_delimiter() {
    let tokenizer = tokenizer();
    let x = id(&tokenizer, "x");
    let brace = id(&tokenizer, "{");
    let mut sampler = constrained(
        &tokenizer,
        &ResponseFormat::JsonObject,
        Some(THINK_END),
        params(0.0, 1),
    );
    assert!(sampler.selects_argmax());
    assert_eq!(
        sampler.sample(&mut ranking(&[x])).expect("think").token_id,
        x
    );
    // The block's argmax is still fine while reasoning, and records the delimiter.
    assert_eq!(sampler.greedy_result(THINK_END).token_id, THINK_END);
    assert!(!sampler.selects_argmax());
    assert_eq!(take(&mut sampler, brace, x).token_id, brace);
    // A precomputed argmax can never bypass the mask: it fails the request.
    let bypass = sampler.greedy_result(x);
    assert!(bypass.is_eos);
    assert!(sampler.grammar_failure().is_some());
    assert_eq!(sampler.response_format_complete(true), Some(false));

    // End-of-sequence while reasoning ends with no answer at all.
    let mut early = constrained(
        &tokenizer,
        &ResponseFormat::JsonObject,
        Some(THINK_END),
        params(0.0, 1),
    );
    assert!(early.sample(&mut ranking(&[EOS])).expect("eos").is_eos);
    assert!(early.grammar_failure().is_none());
    assert_eq!(early.response_format_complete(true), Some(false));
}

#[test]
fn grammar_errors_are_reported_not_turned_into_success() {
    let tokenizer = tokenizer();
    let mut grammar = tokenizer
        .compile_format_ending(&ResponseFormat::JsonObject, None)
        .expect("compile")
        .expect("constraint");
    grammar.trigger_error_for_tests();
    let mut sampler = Sampler::new(VOCAB, params(0.0, 1));
    sampler.constrain(grammar);
    let brace = id(&tokenizer, "{");
    let verified = verify_greedy_drafts(&mut sampler, &[brace, brace], |_, target| {
        target.sample(&mut ranking(&[brace]))
    })
    .expect("verification");
    // The failed row stops verification and is flagged on the sampler.
    assert_eq!(verified.accepted, 0);
    assert_eq!(verified.samples.len(), 1);
    assert!(verified.samples[0].is_eos);
    assert!(sampler.grammar_failure().is_some());
    assert_eq!(sampler.response_format_complete(true), Some(false));
}

/// Deterministic logits for row `row` of a speculative round, favouring
/// forbidden tokens so the mask decides.
fn row_logits(tokenizer: &BonsaiTokenizer, row: usize, seed: u64) -> Vec<f32> {
    let x = id(tokenizer, "x") as usize;
    let mut logits = (0..VOCAB)
        .map(|token| ((token * 31 + row * 17 + seed as usize * 7) % 23) as f32 / 4.0)
        .collect::<Vec<_>>();
    logits[x] = 50.0;
    logits[EOS as usize] = -50.0;
    logits
}

#[test]
fn constrained_speculative_rejection_equals_sequential_target_selections() {
    let tokenizer = tokenizer();
    let x = id(&tokenizer, "x");
    let format = object_schema();
    let rows = 5;
    for temperature in [0.0, 0.8] {
        for seed in 0..8 {
            let sampling = SamplingParams {
                top_k: 40,
                presence_penalty: 0.3,
                repetition_penalty: 1.1,
                ..params(temperature, seed)
            };
            let logits = (0..rows + 3)
                .map(|row| row_logits(&tokenizer, row, seed))
                .collect::<Vec<_>>();
            // The target alone, one selection per row.
            let sequential = |count: usize| {
                let mut sampler = constrained(&tokenizer, &format, None, sampling.clone());
                let selected = (0..count)
                    .map(|row| {
                        let sample = sampler.sample(&mut logits[row].clone()).expect("target");
                        sampler.observe(&[sample.token_id]);
                        sample
                    })
                    .collect::<Vec<_>>();
                (sampler, selected)
            };
            let (_, expected) = sequential(rows);
            assert!(
                expected
                    .iter()
                    .all(|sample| !sample.is_eos && sample.token_id != x)
            );
            for accepted in 0..rows {
                let mut drafts = expected[..rows - 1]
                    .iter()
                    .map(|sample| sample.token_id)
                    .collect::<Vec<_>>();
                if accepted < drafts.len() {
                    // A draft the grammar forbids: the target can never agree.
                    drafts[accepted] = x;
                }
                let mut speculative = constrained(&tokenizer, &format, None, sampling.clone());
                // Unconstrained drafting must leave the target's matcher alone.
                let mut draft = speculative.greedy_draft();
                assert_eq!(
                    draft
                        .sample(&mut logits[0].clone())
                        .expect("draft")
                        .token_id,
                    x
                );
                let verified = verify_greedy_drafts(&mut speculative, &drafts, |row, target| {
                    target.sample(&mut logits[row].clone())
                })
                .expect("verification");
                assert_eq!(verified.accepted, accepted, "seed {seed} t {temperature}");
                assert_eq!(verified.samples, expected[..=accepted]);
                // Both continue identically from the committed rows.
                let corrective = verified.samples[accepted].token_id;
                speculative.observe(&[corrective]);
                let (mut reference, _) = sequential(accepted + 1);
                for (row, next) in logits.iter().enumerate().skip(accepted + 1).take(3) {
                    assert_eq!(
                        speculative.sample(&mut next.clone()).expect("speculative"),
                        reference.sample(&mut next.clone()).expect("reference"),
                        "seed {seed} t {temperature} accepted {accepted} row {row}"
                    );
                }
                assert!(speculative.grammar_failure().is_none());
            }
        }
    }
}

/// The real checkpoint tokenizer: exact token bytes, special tokens out of
/// every grammar, and the one-time table build and per-token mask costs.
#[test]
#[ignore = "requires the pinned model (CPU only)"]
fn real_tokenizer_builds_exact_constraint_tables() {
    use crate::bonsai::BonsaiPackage;
    let package = BonsaiPackage::open(crate::bonsai::DEFAULT_BONSAI_GGUF).expect("package");
    let tokenizer = BonsaiTokenizer::from_package(&package).expect("tokenizer");
    let started = std::time::Instant::now();
    let grammar = tokenizer
        .compile_format(&ResponseFormat::JsonObject, true)
        .expect("compile")
        .expect("constraint");
    eprintln!("first compile (tables + grammar): {:?}", started.elapsed());
    let started = std::time::Instant::now();
    let _ = tokenizer
        .compile_format(&object_schema(), false)
        .expect("compile");
    eprintln!("later compile: {:?}", started.elapsed());
    let mut sampler = Sampler::new(
        crate::bonsai::VOCAB,
        SamplingParams {
            eos_tokens: tokenizer.eos_ids().to_vec(),
            ..params(0.0, 1)
        },
    );
    sampler.constrain(grammar);
    let think_end = 248_069;
    let mut logits = vec![0.0; crate::bonsai::VOCAB];
    logits[think_end] = 10.0;
    assert_eq!(
        sampler.sample(&mut logits).expect("think").token_id,
        think_end as u32
    );
    // llguidance allows no whitespace before the document: the answer opens
    // with `{` right after `</think>`, not the template's usual blank line.
    let answer = "{\"name\": \"Zoë 😀\", \"n\": [1, 2]}";
    let tokens = tokenizer.encode(answer).expect("encode");
    let started = std::time::Instant::now();
    for &token in &tokens {
        let mut logits = vec![0.0; crate::bonsai::VOCAB];
        logits[token as usize] = 10.0;
        // Every special token, end-of-turn included, is out of reach mid-answer.
        logits[think_end] = 1000.0;
        logits[tokenizer.eos_ids()[0] as usize] = 1000.0;
        assert_eq!(sampler.sample(&mut logits).expect("answer").token_id, token);
    }
    eprintln!(
        "{} masked selections: {:?} each",
        tokens.len(),
        started.elapsed() / tokens.len() as u32
    );
    let mut logits = vec![0.0; crate::bonsai::VOCAB];
    logits[tokenizer.eos_ids()[0] as usize] = 1.0;
    assert!(sampler.sample(&mut logits).expect("end").is_eos);
    assert_eq!(sampler.response_format_complete(true), Some(true));
}
