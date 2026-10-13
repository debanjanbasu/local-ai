use std::path::Path;

use serde_json::{Value, json};

use super::{
    CODEREVIEWER_QUESTION, CalibratedHead, DECLARED_SCOPE, DecisionFamily, Json, Refusal,
    sha256_hex,
};
use crate::judgment::{
    DECISION_RENDERER, Decision, DecisionKind, DecisionProbability, DecisionRequest, DecisionValue,
    JUDGMENT_HEAD_FORMAT,
};

/// Python's `judgment_calibrate.canonical_sha256` of this value (computed
/// with CPython 3: sorted keys, compact separators, `ensure_ascii=False`,
/// numbers as `repr` writes them).
const PYTHON_FIXTURE: &str = r#"{"b": [1e-07, 2.0, 0.1, -3, 1e+16, 1.5e-05, 123456789012345.6, 0.0001], "a": {"z": "\u00e9\"\\ \n\t\u001f\u2028x", "y": null, "x": true}, "c": []}"#;
const PYTHON_FIXTURE_SHA256: &str =
    "e233583fb82513ce05b6de4023f3491a5c04c420adebb34897fe462f1e5e0f8f";

#[test]
fn canonical_json_matches_python() {
    let value = Json::parse(PYTHON_FIXTURE).unwrap();
    assert_eq!(
        sha256_hex(value.canonical().as_bytes()),
        PYTHON_FIXTURE_SHA256
    );
    assert!(Json::parse(r#"{"a": 1, "a": 2}"#).is_err());
    assert!(Json::parse("[1, 2").is_err());
    assert!(Json::parse("{} x").is_err());
}

const WIDTH: usize = 4;

fn safetensors(tensors: &[(&str, Vec<usize>, Vec<f32>)], metadata: &Value) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut body = Vec::new();
    for (name, shape, values) in tensors {
        let start = body.len();
        body.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(
            (*name).to_owned(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, body.len()]}),
        );
    }
    header.insert("__metadata__".into(), metadata.clone());
    let encoded = serde_json::to_vec(&header).unwrap();
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.extend(body);
    bytes
}

/// Head: q reads dims 0-1 of the decision row, k reads dims 0-1 of each
/// option row, so `logit_i = (option_i · decide)[0..2] / sqrt(2) / 2`.
fn head_bytes() -> Vec<u8> {
    let identity = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    safetensors(
        &[
            ("q.weight", vec![2, WIDTH], identity.clone()),
            ("q.bias", vec![2], vec![0.0, 0.0]),
            ("k.weight", vec![2, WIDTH], identity),
            ("k.bias", vec![2], vec![0.0, 0.0]),
            ("temperature", vec![1], vec![2.0]),
        ],
        &json!({"format": JUDGMENT_HEAD_FORMAT, "width": "4", "head_dim": "2",
                "experimental": "true"}),
    )
}

/// Gate: one component along dim 0 (variance 4), residual variance 1, mean 0.
fn gate_bytes() -> Vec<u8> {
    safetensors(
        &[
            ("components", vec![1, WIDTH], vec![1.0, 0.0, 0.0, 0.0]),
            ("mean", vec![WIDTH], vec![0.0; WIDTH]),
            ("residual_variance", vec![1], vec![1.0]),
            ("variance", vec![1], vec![4.0]),
        ],
        &json!({"format": super::GATE_FORMAT, "components": "1"}),
    )
}

fn body(head: &[u8], gate: &[u8]) -> Value {
    json!({
        "format": super::CALIBRATION_FORMAT,
        "head_sha256": sha256_hex(head),
        "gate_sha256": sha256_hex(gate),
        "families": {
            "codereviewer_needs_comment": {
                "calibration": {"method": "platt", "slope": 1.5, "bias": -0.25,
                                "input": "raw logit(Yes) - raw logit(No)"},
                "threshold": null},
            "commitpackft_message": {
                "calibration": {"method": "temperature", "temperature": 0.5,
                                "input": "raw logits"},
                "threshold": 0.75}
        },
        "refusal": {"target_risk": 0.1, "min_answered": 50},
        "gate": {"file_sha256": sha256_hex(gate), "threshold": 2.0, "percentile": 0.99},
        "scope": serde_json::from_str::<Value>(DECLARED_SCOPE).unwrap(),
        "provenance": {"note": "synthetic test artifact"},
        "acceptance": {"A1": true, "passed": true}
    })
}

/// `body` with its content hash, as Python writes it (indented).
fn sealed(body: &Value) -> String {
    let text = serde_json::to_string(body).unwrap();
    let hash = sha256_hex(Json::parse(&text).unwrap().canonical().as_bytes());
    let mut value = body.clone();
    value["content_sha256"] = json!(hash);
    serde_json::to_string_pretty(&value).unwrap()
}

fn write_dir(dir: &Path, artifact: &str, head: &[u8], gate: &[u8]) {
    std::fs::write(dir.join("calibration.json"), artifact).unwrap();
    std::fs::write(dir.join("head.safetensors"), head).unwrap();
    std::fs::write(dir.join("gate.safetensors"), gate).unwrap();
}

fn open_with(change: impl FnOnce(&mut Value)) -> crate::Result<CalibratedHead> {
    let (head, gate) = (head_bytes(), gate_bytes());
    let mut value = body(&head, &gate);
    change(&mut value);
    let dir = tempfile::tempdir().unwrap();
    write_dir(dir.path(), &sealed(&value), &head, &gate);
    CalibratedHead::open(dir.path())
}

fn message(error: crate::Result<CalibratedHead>) -> String {
    error.expect_err("refused").to_string()
}

#[test]
fn a_sealed_artifact_opens_and_reports_its_hashes() {
    let head = open_with(|_| {}).unwrap();
    assert_eq!(head.head_sha256(), sha256_hex(&head_bytes()));
    assert_eq!(head.gate_sha256(), sha256_hex(&gate_bytes()));
    assert_eq!(
        head.threshold(DecisionFamily::CodeReviewerNeedsComment),
        None
    );
    assert_eq!(
        head.threshold(DecisionFamily::CommitMessageMatch),
        Some(0.75)
    );
    assert_eq!(head.head().width(), WIDTH);
}

#[test]
fn tampering_with_content_head_or_gate_is_refused() {
    let (head, gate) = (head_bytes(), gate_bytes());
    let sealed_text = sealed(&body(&head, &gate));
    let dir = tempfile::tempdir().unwrap();

    let tampered = sealed_text.replace("0.75", "0.5");
    write_dir(dir.path(), &tampered, &head, &gate);
    assert!(message(CalibratedHead::open(dir.path())).contains("content hash"));

    let mut other_head = head.clone();
    *other_head.last_mut().unwrap() = 0; // temperature bytes change
    write_dir(dir.path(), &sealed_text, &other_head, &gate);
    assert!(message(CalibratedHead::open(dir.path())).contains("head_sha256"));

    let mut other_gate = gate.clone();
    let last = other_gate.len() - 1;
    other_gate[last] ^= 1;
    write_dir(dir.path(), &sealed_text, &head, &other_gate);
    assert!(message(CalibratedHead::open(dir.path())).contains("gate_sha256"));

    write_dir(dir.path(), &sealed_text, &head, &gate);
    assert!(CalibratedHead::open(dir.path()).is_ok());
}

#[test]
fn a_failed_acceptance_another_scope_or_missing_family_is_refused() {
    assert!(
        message(open_with(|v| v["acceptance"]["passed"] = json!(false))).contains("acceptance")
    );
    assert!(
        message(open_with(|v| v["scope"]["max_prompt_tokens"] = json!(4096))).contains("scope")
    );
    assert!(
        message(open_with(|v| {
            v["families"]
                .as_object_mut()
                .unwrap()
                .remove("commitpackft_message");
        }))
        .contains("families")
    );
    assert!(message(open_with(|v| v["format"] = json!("v1"))).contains("format"));
    assert!(message(open_with(|v| v["gate"]["file_sha256"] = json!("0"))).contains("gate_sha256"));
}

fn request(state: &str, question: &str, kind: DecisionKind) -> DecisionRequest {
    DecisionRequest {
        state: state.into(),
        question: question.into(),
        kind,
    }
}

#[test]
fn only_the_declared_questions_over_diffs_are_in_scope() {
    let cm = "Does this commit message describe this diff? Message: \"Fix the \"x\" bug\"";
    let of = |state: &str, question: &str| {
        DecisionFamily::of(&request(state, question, DecisionKind::Predicate))
    };
    assert_eq!(
        of("Diff:\n@@", CODEREVIEWER_QUESTION),
        Some(DecisionFamily::CodeReviewerNeedsComment)
    );
    assert_eq!(
        of(
            "Context before hunk:\nx\n\nDiff:\n@@",
            CODEREVIEWER_QUESTION
        ),
        Some(DecisionFamily::CodeReviewerNeedsComment)
    );
    assert_eq!(
        of("Diff:\n--- a", cm),
        Some(DecisionFamily::CommitMessageMatch)
    );
    assert_eq!(of("diff:\n@@", CODEREVIEWER_QUESTION), None);
    assert_eq!(
        of("Diff:\n@@", "Does this change need a reviewer comment"),
        None
    );
    assert_eq!(
        of(
            "Diff:\n@@",
            "Does this commit message describe this diff? Message: \"\""
        ),
        None
    );
    assert_eq!(
        of(
            "Diff:\n@@",
            "Does this commit message describe this diff? Message: \"a\nb\""
        ),
        None
    );
    assert_eq!(
        of("Is this request unsafe?", "Is this request unsafe?"),
        None
    );
    let choice = DecisionKind::Choice(vec![DecisionValue::Bool(true), DecisionValue::Bool(false)]);
    assert_eq!(
        DecisionFamily::of(&request("Diff:\n@@", CODEREVIEWER_QUESTION, choice)),
        None
    );
}

/// A predicate decision with option rows `yes`/`no` and decision row `decide`,
/// scored by the synthetic head.
fn decision(head: &CalibratedHead, yes: [f32; 4], no: [f32; 4], decide: [f32; 4]) -> Decision {
    let features: Vec<f32> = yes.iter().chain(&no).chain(&decide).copied().collect();
    let scores = head.head().score(&features).unwrap();
    Decision {
        options: [DecisionValue::Bool(true), DecisionValue::Bool(false)]
            .into_iter()
            .zip(scores.probabilities.iter().zip(&scores.logits))
            .map(|(option, (&probability, &logit))| DecisionProbability {
                option,
                probability,
                logit: Some(logit),
            })
            .collect(),
        argmax: 0,
        value: DecisionValue::Bool(true),
        score: None,
        prompt_tokens: 100,
        captured: true,
        renderer: DECISION_RENDERER,
        renderer_declared: true,
        calibration_scope: String::new(),
        decide_row: decide.to_vec(),
    }
}

#[test]
fn judgments_follow_calibration_thresholds_and_the_gate() {
    let head = open_with(|_| {}).unwrap();
    let decide = [1.0, 1.0, 0.0, 0.0];
    // raw = logit * temperature = (option . decide)[0..2] / sqrt(2)
    let strong = decision(&head, [2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], decide);
    let margin = 3.0 / 2f64.sqrt();

    let cm = head
        .judge(DecisionFamily::CommitMessageMatch, &strong)
        .unwrap();
    let expected = 1.0 / (1.0 + (-margin / 0.5).exp());
    assert!((cm.probability_yes - expected).abs() < 1e-12, "{cm:?}");
    assert!((cm.confidence - expected).abs() < 1e-12);
    // distance: z = 1 on the component (variance 4), residual energy 1, / width
    assert!((cm.gate_distance - (0.25 + 1.0) / 4.0).abs() < 1e-12);
    assert_eq!(cm.refusal, None);

    let cr = head
        .judge(DecisionFamily::CodeReviewerNeedsComment, &strong)
        .unwrap();
    let expected = 1.0 / (1.0 + (-(1.5 * margin - 0.25)).exp());
    assert!((cr.probability_yes - expected).abs() < 1e-12);
    assert_eq!(cr.refusal, Some(Refusal::FamilyRefused));

    let weak = decision(&head, [0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], decide);
    let low = head
        .judge(DecisionFamily::CommitMessageMatch, &weak)
        .unwrap();
    assert!(
        matches!(low.refusal, Some(Refusal::LowConfidence { threshold, .. })
        if threshold == 0.75)
    );

    let far = decision(&head, [2.0, 1.0, 0.0, 0.0], [0.0; 4], [1.0, 1.0, 9.0, 0.0]);
    let gated = head
        .judge(DecisionFamily::CommitMessageMatch, &far)
        .unwrap();
    assert!(
        matches!(gated.refusal, Some(Refusal::Gate { .. })),
        "{gated:?}"
    );

    let mut long = strong.clone();
    long.prompt_tokens = 2049;
    let too_long = head
        .judge(DecisionFamily::CommitMessageMatch, &long)
        .unwrap();
    assert_eq!(
        too_long.refusal,
        Some(Refusal::PromptTooLong {
            tokens: 2049,
            limit: 2048
        })
    );

    let mut bare = strong;
    bare.decide_row.clear();
    assert!(
        head.judge(DecisionFamily::CommitMessageMatch, &bare)
            .is_err()
    );
}

/// Rows of a JSONL file, by `id`.
fn jsonl_by_id(path: &str) -> std::collections::HashMap<String, Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let row: Value = serde_json::from_str(line).unwrap();
            (row["id"].as_str().unwrap().to_owned(), row)
        })
        .collect()
}

/// Offline parity with the Python evaluation: every captured row of a
/// feature directory, judged here from its stored (serving-path) features,
/// gets the decision `judgment_reeval.py offline` computed from the same
/// features and artifact. Run with
/// `LOCAL_AI_DECISIONS_ARTIFACT=DIR LOCAL_AI_DECISIONS_FEATURES=DIR
/// LOCAL_AI_DECISIONS_REQUESTS=requests.jsonl LOCAL_AI_DECISIONS_EXPECTED=offline.jsonl
/// cargo test -p local-engine --release -- --ignored offline_parity`.
#[test]
#[ignore = "needs an evaluated artifact, captured features and their offline decisions"]
fn offline_parity_with_the_python_evaluation() {
    let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("set {name}"));
    let head = CalibratedHead::open(var("LOCAL_AI_DECISIONS_ARTIFACT")).unwrap();
    let features_dir = std::path::PathBuf::from(var("LOCAL_AI_DECISIONS_FEATURES"));
    let requests = jsonl_by_id(&var("LOCAL_AI_DECISIONS_REQUESTS"));
    let expected = jsonl_by_id(&var("LOCAL_AI_DECISIONS_EXPECTED"));
    let manifest = std::fs::read_to_string(features_dir.join("features.jsonl")).unwrap();
    let (mut rows, mut answered, mut worst) = (0, 0, 0f64);
    for line in manifest.lines().filter(|line| !line.is_empty()) {
        let row: Value = serde_json::from_str(line).unwrap();
        let id = row["id"].as_str().unwrap();
        let (request, want) = (&requests[id], &expected[id]);
        let kind = if request["kind"] == "predicate" {
            DecisionKind::Predicate
        } else {
            DecisionKind::Choice(Vec::new())
        };
        let native = DecisionRequest {
            state: request["state"].as_str().unwrap().into(),
            question: request["question"].as_str().unwrap().into(),
            kind,
        };
        let family = DecisionFamily::of(&native);
        assert_eq!(family.is_some(), want["in_scope"] == true, "{id}: scope");
        rows += 1;
        let Some(family) = family else { continue };
        let bytes =
            std::fs::read(features_dir.join(row["feature_file"].as_str().unwrap())).unwrap();
        let features: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| half::f16::from_le_bytes(b).to_f32())
            .collect();
        let scores = head.head().score(&features).unwrap();
        // The native order is Yes, No (a predicate request).
        let decision = Decision {
            options: [DecisionValue::Bool(true), DecisionValue::Bool(false)]
                .into_iter()
                .zip(scores.probabilities.iter().zip(&scores.logits))
                .map(|(option, (&probability, &logit))| DecisionProbability {
                    option,
                    probability,
                    logit: Some(logit),
                })
                .collect(),
            argmax: 0,
            value: DecisionValue::Bool(true),
            score: None,
            prompt_tokens: row["token_ids"].as_array().unwrap().len(),
            captured: true,
            renderer: DECISION_RENDERER,
            renderer_declared: true,
            calibration_scope: String::new(),
            decide_row: features[features.len() - head.head().width()..].to_vec(),
        };
        let judged = head.judge(family, &decision).unwrap();
        let mut close = |a: f64, b: &Value| {
            let b = b.as_f64().unwrap();
            let error = (a - b).abs() / b.abs().max(1.0);
            worst = worst.max(error);
            error < 1e-9
        };
        assert!(
            close(judged.probability_yes, &want["p_yes"]),
            "{id}: {judged:?} vs {want}"
        );
        assert!(close(judged.confidence, &want["confidence"]), "{id}");
        assert!(close(judged.gate_distance, &want["gate_distance"]), "{id}");
        let kind = match judged.refusal {
            None => Value::Null,
            Some(Refusal::PromptTooLong { .. }) => json!("prompt_too_long"),
            Some(Refusal::Gate { .. }) => json!("gate"),
            Some(Refusal::FamilyRefused) => json!("family_refused"),
            Some(Refusal::LowConfidence { .. }) => json!("low_confidence"),
        };
        assert_eq!(kind, want["refusal"], "{id}");
        answered += usize::from(judged.refusal.is_none());
    }
    eprintln!("offline parity: {rows} rows, {answered} answered, worst relative error {worst:e}");
}

/// Native parity: the same requests decided on the loaded model by
/// `Engine::decide` with the artifact's head and judged here get the
/// decisions `judgment_reeval.py offline` computed from their captured
/// features. Run under the GPU lock with
/// `LOCAL_AI_DECISIONS_ARTIFACT=DIR LOCAL_AI_DECISIONS_REQUESTS=a.jsonl[:b.jsonl…]
/// LOCAL_AI_DECISIONS_EXPECTED=offline.jsonl LOCAL_AI_DECISIONS_IDS=ids.txt
/// cargo test -p local-engine --release -- --ignored native_parity`.
#[test]
#[ignore = "requires the pinned model and an evaluated artifact"]
fn native_parity_with_the_offline_decisions() {
    let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("set {name}"));
    let head = CalibratedHead::open(var("LOCAL_AI_DECISIONS_ARTIFACT")).unwrap();
    let mut requests = std::collections::HashMap::new();
    for path in var("LOCAL_AI_DECISIONS_REQUESTS").split(':') {
        requests.extend(jsonl_by_id(path));
    }
    let expected = jsonl_by_id(&var("LOCAL_AI_DECISIONS_EXPECTED"));
    let ids = std::fs::read_to_string(var("LOCAL_AI_DECISIONS_IDS")).unwrap();
    let model = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    );
    let mut engine = crate::Engine::open_model(model).unwrap();
    let (mut rows, mut answered) = (0, 0);
    for id in ids.lines().filter(|line| !line.is_empty()) {
        let (request, want) = (&requests[id], &expected[id]);
        let native = DecisionRequest {
            state: request["state"].as_str().unwrap().into(),
            question: request["question"].as_str().unwrap().into(),
            kind: if request["kind"] == "predicate" {
                DecisionKind::Predicate
            } else {
                DecisionKind::Choice(
                    request["options"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| DecisionValue::Text(v.as_str().unwrap().into()))
                        .collect(),
                )
            },
        };
        rows += 1;
        let Some(family) = DecisionFamily::of(&native) else {
            assert_eq!(want["in_scope"], false, "{id}");
            continue;
        };
        let decision = engine.decide(&head.head(), &native).unwrap();
        let judged = head.judge(family, &decision).unwrap();
        let p = want["p_yes"].as_f64().unwrap();
        assert!(
            (judged.probability_yes - p).abs() < 1e-9,
            "{id}: {judged:?} vs {want}"
        );
        assert!(
            (judged.gate_distance - want["gate_distance"].as_f64().unwrap()).abs() < 1e-9,
            "{id}"
        );
        assert_eq!(judged.refusal.is_none(), want["refusal"].is_null(), "{id}");
        answered += usize::from(judged.refusal.is_none());
    }
    eprintln!("native parity: {rows} rows, {answered} answered");
}
