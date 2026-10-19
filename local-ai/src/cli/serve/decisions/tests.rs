use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json};
use sha2_free::sha256_hex;
use tokio::sync::watch;

use local_engine::judgment::{
    CalibratedHead, DECLARED_SCOPE, Decision, DecisionProbability, DecisionRequest, DecisionValue,
    JUDGMENT_HEAD_FORMAT,
};

use super::{FLAG, ROUTE, answer, prepare, refusal_message, response_json};

/// SHA-256 through `ring`, already a dependency of this crate.
mod sha2_free {
    pub fn sha256_hex(bytes: &[u8]) -> String {
        use std::fmt::Write as _;
        ring::digest::digest(&ring::digest::SHA256, bytes)
            .as_ref()
            .iter()
            .fold(String::new(), |mut out, b| {
                let _ = write!(out, "{b:02x}");
                out
            })
    }
}

const SCHEMAS: &str = include_str!("../decisions_openapi_schemas.json");
const WIDTH: usize = 4;
const CR: &str = "Does this change need a reviewer comment?";
const CM: &str = "Does this commit message describe this diff? Message: \"Add a test\"";

fn safetensors(tensors: &[(&str, Vec<usize>, Vec<f32>)], metadata: &Value) -> Vec<u8> {
    let mut header = Map::new();
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

/// Python's canonical JSON for the ASCII-only, integer-free-of-exponent
/// values used here: sorted keys, compact separators.
fn canonical(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let fields: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}:{}", Value::String(k.clone()), canonical(&map[k])))
                .collect();
            format!("{{{}}}", fields.join(","))
        }
        Value::Array(items) => {
            format!(
                "[{}]",
                items.iter().map(canonical).collect::<Vec<_>>().join(",")
            )
        }
        other => other.to_string(),
    }
}

/// A synthetic, sealed artifact directory: the head reads dims 0-1, the gate
/// is one component along dim 0; `CodeReviewer` has no threshold,
/// `CommitPackFT` 0.75.
fn artifact(dir: &Path, passed: bool) {
    let identity = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let head = safetensors(
        &[
            ("q.weight", vec![2, WIDTH], identity.clone()),
            ("q.bias", vec![2], vec![0.0, 0.0]),
            ("k.weight", vec![2, WIDTH], identity),
            ("k.bias", vec![2], vec![0.0, 0.0]),
            ("temperature", vec![1], vec![2.0]),
        ],
        &json!({"format": JUDGMENT_HEAD_FORMAT, "width": "4", "head_dim": "2",
                "experimental": "true"}),
    );
    let gate = safetensors(
        &[
            ("components", vec![1, WIDTH], vec![1.0, 0.0, 0.0, 0.0]),
            ("mean", vec![WIDTH], vec![0.0; WIDTH]),
            ("residual_variance", vec![1], vec![1.0]),
            ("variance", vec![1], vec![4.0]),
        ],
        &json!({"format": "local-ai.judgment-gate.v1"}),
    );
    let mut body = json!({
        "format": "local-ai.judgment-calibration.v2",
        "head_sha256": sha256_hex(&head),
        "gate_sha256": sha256_hex(&gate),
        "families": {
            "codereviewer_needs_comment": {
                "calibration": {"method": "platt", "slope": 1.5, "bias": -0.25}, "threshold": null},
            "commitpackft_message": {
                "calibration": {"method": "temperature", "temperature": 0.5}, "threshold": 0.75}
        },
        "gate": {"file_sha256": sha256_hex(&gate), "threshold": 2.0},
        "scope": serde_json::from_str::<Value>(DECLARED_SCOPE).unwrap(),
        "acceptance": {"passed": passed}
    });
    let hash = sha256_hex(canonical(&body).as_bytes());
    body["content_sha256"] = json!(hash);
    std::fs::write(
        dir.join("calibration.json"),
        serde_json::to_vec_pretty(&body).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("head.safetensors"), head).unwrap();
    std::fs::write(dir.join("gate.safetensors"), gate).unwrap();
}

fn calibrated() -> CalibratedHead {
    let dir = tempfile::tempdir().unwrap();
    artifact(dir.path(), true);
    CalibratedHead::open(dir.path()).expect("synthetic artifact opens")
}

/// The native outcome for `request` as the engine would assemble it from
/// features: option rows `yes`/`no` and decision row `decide`.
fn native(head: &CalibratedHead, yes: f32, decide: [f32; 4]) -> Decision {
    let features: Vec<f32> = [yes, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        .into_iter()
        .chain(decide)
        .collect();
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
        prompt_tokens: 40,
        captured: true,
        decide_row: decide.to_vec(),
    }
}

/// Check `value` against the pinned schema `name`.
fn conforms(name: &str, value: &Value) {
    let spec: Value = serde_json::from_str(SCHEMAS).expect("schema fixture");
    let mut root = Map::new();
    root.insert("$ref".into(), json!(format!("#/components/schemas/{name}")));
    root.insert("components".into(), spec["components"].clone());
    let validator = jsonschema::options()
        .build(&Value::Object(root))
        .expect("the pinned schema compiles");
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|error| error.to_string())
        .collect();
    assert!(errors.is_empty(), "{name}: {errors:#?}\n{value:#}");
}

fn request_body() -> Value {
    json!({
        "model": "bonsai",
        "input": "Diff:\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b",
        "questions": [
            {"type": "predicate", "name": "message", "instructions": CM},
            {"type": "predicate", "name": "review", "instructions": CR},
            {"type": "predicate", "instructions": "Is this diff large?"},
            {"type": "choice", "name": "kind", "instructions": "Kind?",
             "choices": [{"value": "fix"}, {"value": "feature"}]},
            {"type": "score", "name": "risk", "instructions": "Risk?",
             "levels": [{"label": "low"}, {"label": "high"}]},
            {"type": "predicate", "name": "weak", "instructions": CM},
            {"type": "predicate", "name": "far", "instructions": CM}
        ]
    })
}

#[tokio::test]
async fn answers_are_calibrated_refusals_are_explicit_and_both_match_the_spec() {
    let head = Arc::new(calibrated());
    let body = request_body();
    conforms("DecisionRequest", &body);
    let prepared = prepare(&serde_json::to_vec(&body).unwrap()).expect("valid request");
    // Only in-scope questions reach the engine, in order: CM, CR, weak CM, far CM.
    let outcomes = [
        (2.0, [1.0, 1.0, 0.0, 0.0]),
        (2.0, [1.0, 1.0, 0.0, 0.0]),
        (0.1, [1.0, 1.0, 0.0, 0.0]),
        (2.0, [1.0, 1.0, 9.0, 0.0]),
    ];
    let seen: Arc<Mutex<Vec<DecisionRequest>>> = Arc::default();
    let (head_ref, seen_ref) = (Arc::clone(&head), Arc::clone(&seen));
    let mut index = 0;
    let submit = move |request: DecisionRequest| {
        let (yes, decide) = outcomes[index];
        index += 1;
        seen_ref.lock().unwrap().push(request);
        let decision = native(&head_ref, yes, decide);
        std::future::ready(Ok(std::future::ready(Ok(Some(decision)))))
    };
    let (_closing, receiver) = watch::channel(false);
    let answered = answer(&prepared, &head, submit, receiver)
        .await
        .expect("answered");
    let value = response_json("bonsai", answered);
    conforms("DecisionResponse", &value);
    let answers = value["answers"].as_array().unwrap();
    let kinds: Vec<&str> = answers
        .iter()
        .map(|a| a["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "predicate",
            "refusal",
            "refusal",
            "refusal",
            "refusal",
            "refusal",
            "refusal"
        ]
    );
    let names: Vec<&Value> = answers.iter().map(|a| &a["name"]).collect();
    assert_eq!(names[2], &Value::Null);
    assert_eq!(names[0], "message");
    // HTTP answer == the native judgment of the same decision.
    let judged = head
        .judge(
            local_engine::judgment::DecisionFamily::CommitMessageMatch,
            &native(&head, 2.0, [1.0, 1.0, 0.0, 0.0]),
        )
        .unwrap();
    assert_eq!(answers[0]["probability"], json!(judged.probability_yes));
    assert_eq!(value["usage"]["input_tokens"], 160);
    let seen = std::mem::take(&mut *seen.lock().unwrap());
    assert_eq!(seen.len(), 4);
    assert!(seen.iter().all(|r| r.state.starts_with("Diff:\n")));
}

#[tokio::test]
async fn message_inputs_and_bad_bodies() {
    let head = calibrated();
    let body = json!({"model": "m", "input": [{"role": "user", "content": "Diff:\nx"}],
                      "questions": [{"type": "predicate", "instructions": CR}]});
    conforms("DecisionRequest", &body);
    let prepared = prepare(&serde_json::to_vec(&body).unwrap()).unwrap();
    let submit = |_: DecisionRequest| -> std::future::Ready<
        crate::Result<std::future::Ready<crate::Result<Option<Decision>>>>,
    > { unreachable!("message inputs never reach the engine") };
    let (_closing, receiver) = watch::channel(false);
    let value = response_json(
        "m",
        answer(&prepared, &head, submit, receiver).await.unwrap(),
    );
    assert_eq!(value["answers"], json!([{"type": "refusal", "name": null}]));
    assert_eq!(value["usage"]["total_tokens"], 0);

    for bad in [
        json!({"input": "x", "questions": [{"type": "predicate", "instructions": "q"}]}),
        json!({"model": "m", "input": "x", "questions": []}),
        json!({"model": "m", "input": 3, "questions": [{"type": "predicate", "instructions": "q"}]}),
        json!({"model": "m", "input": "x", "questions": [{"type": "predicate",
               "instructions": "q", "extra": 1}]}),
    ] {
        assert!(
            prepare(&serde_json::to_vec(&bad).unwrap()).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn the_artifact_is_verified_before_serving() {
    let dir = tempfile::tempdir().unwrap();
    artifact(dir.path(), false);
    let error = super::Decisions::open(dir.path(), WIDTH)
        .err()
        .expect("a failed artifact is refused")
        .to_string();
    assert!(error.contains("acceptance"), "{error}");
    artifact(dir.path(), true);
    assert!(super::Decisions::open(dir.path(), WIDTH).is_ok());
    let error = super::Decisions::open(dir.path(), 5120)
        .err()
        .expect("wrong width")
        .to_string();
    assert!(error.contains("width"), "{error}");
    std::fs::write(dir.path().join("head.safetensors"), b"tampered").unwrap();
    let error = super::Decisions::open(dir.path(), WIDTH)
        .err()
        .expect("tampered head")
        .to_string();
    assert!(error.contains("head_sha256"), "{error}");
}

#[test]
fn the_flag_is_explicit_and_the_route_explains_itself_without_it() {
    use super::super::options::parse;
    let args = |values: &[&str]| values.iter().map(|v| (*v).to_owned()).collect::<Vec<_>>();
    assert_eq!(
        parse(&args(&[])).expect("defaults").decisions_artifact,
        None
    );
    let parsed = parse(&args(&[FLAG, "/tmp/artifact"])).expect("flag");
    assert_eq!(
        parsed.decisions_artifact.as_deref(),
        Some(Path::new("/tmp/artifact"))
    );
    assert!(parse(&args(&[FLAG])).is_err());
    assert!(parse(&args(&[FLAG, ""])).is_err());
    let message = refusal_message();
    for expected in [ROUTE, FLAG, "docs/DECISIONS_EVALUATION.md", "refusal"] {
        assert!(message.contains(expected), "{message}");
    }
}
