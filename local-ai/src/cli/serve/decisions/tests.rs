use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use tokio::sync::{oneshot, watch};

use local_engine::judgment::{
    DECISION_RENDERER, Decision, DecisionKind, DecisionProbability, DecisionRequest, DecisionValue,
    UNKNOWN_CALIBRATION_SCOPE,
};

use super::{Decisions, FLAG, Failure, MAX_CONCURRENT, Prepared, answer, prepare, response_json};

/// The synthetic calibration scope [`decision`] reports, as a head's
/// `calibration_scope` metadata would.
const SCOPE: &str = "synthetic test scope: calibration split of a test set only";

fn prepared(body: &Value) -> Prepared {
    prepare(&serde_json::to_vec(body).expect("encode")).expect("valid request")
}

fn refused(body: &Value) -> String {
    prepare(&serde_json::to_vec(body).expect("encode")).expect_err("refused")
}

/// The native outcome for `options` with `probabilities`, as the engine
/// assembles it.
fn decision(kind: &DecisionKind, probabilities: &[f64], tokens: usize) -> Decision {
    let options = match kind {
        DecisionKind::Predicate => vec![DecisionValue::Bool(true), DecisionValue::Bool(false)],
        DecisionKind::Choice(values) => values.clone(),
        DecisionKind::Score(labels) => labels.iter().cloned().map(DecisionValue::Text).collect(),
    };
    let argmax = (0..probabilities.len()).fold(0, |best, i| {
        if probabilities[i] > probabilities[best] {
            i
        } else {
            best
        }
    });
    let score = matches!(kind, DecisionKind::Score(_)).then(|| {
        probabilities
            .iter()
            .enumerate()
            .map(|(i, p)| i as f64 * p)
            .sum()
    });
    Decision {
        options: options
            .iter()
            .zip(probabilities)
            .map(|(option, &probability)| DecisionProbability {
                option: option.clone(),
                probability,
                logit: Some(probability.ln()),
            })
            .collect(),
        argmax,
        value: options[argmax].clone(),
        score,
        prompt_tokens: tokens,
        captured: true,
        renderer: DECISION_RENDERER,
        renderer_declared: true,
        calibration_scope: SCOPE.into(),
    }
}

/// A service whose decisions resolve at once with `probabilities`.
#[allow(clippy::type_complexity)]
fn immediate(
    probabilities: &'static [&'static [f64]],
    seen: Arc<std::sync::Mutex<Vec<DecisionRequest>>>,
) -> impl FnMut(
    DecisionRequest,
)
    -> std::future::Ready<crate::Result<std::future::Ready<crate::Result<Option<Decision>>>>> {
    let mut index = 0;
    move |request: DecisionRequest| {
        let outcome = decision(&request.kind, probabilities[index], 10 + index);
        index += 1;
        seen.lock().expect("lock").push(request);
        std::future::ready(Ok(std::future::ready(Ok(Some(outcome)))))
    }
}

fn open() -> watch::Receiver<bool> {
    // Leaked so the service never looks closed.
    let (sender, receiver) = watch::channel(false);
    std::mem::forget(sender);
    receiver
}

#[tokio::test]
async fn maps_each_question_type_without_confidence_or_refusal() {
    let request = prepared(&json!({
        "model": "ignored",
        "input": "Diff:\n-a\n+b",
        "questions": [
            {"type": "predicate", "name": "needs_comment", "instructions": "Does this need a comment?"},
            {"type": "choice", "instructions": "Which?", "choices": [{"value": true}, {"value": "maybe"}]},
            {"type": "score", "name": "risk", "instructions": "How risky?", "levels": [
                {"label": "Low"}, {"label": "Medium"}, {"label": "High"}
            ]},
        ],
    }));
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let answered = answer(
        &request,
        immediate(
            &[&[0.25, 0.75], &[0.6, 0.4], &[0.1, 0.7, 0.2]],
            Arc::clone(&seen),
        ),
        open(),
    )
    .await
    .expect("answered");
    let body = response_json("bonsai", answered);
    assert_eq!(body["object"], "experimental.decision");
    assert_eq!(body["model"], "bonsai");
    assert_eq!(
        body["answers"][0],
        json!({"type": "predicate", "name": "needs_comment", "probability": 0.25})
    );
    assert_eq!(
        body["answers"][1],
        json!({"type": "choice", "name": null, "choice": true, "probabilities": [
            {"value": true, "probability": 0.6}, {"value": "maybe", "probability": 0.4}
        ]})
    );
    let score = &body["answers"][2];
    assert_eq!(score["type"], "score");
    assert_eq!(score["name"], "risk");
    assert!((score["score"].as_f64().expect("score") - 1.1).abs() < 1e-12);
    assert_eq!(
        score["probabilities"],
        json!([
            {"value": 0, "label": "Low", "probability": 0.1},
            {"value": 1, "label": "Medium", "probability": 0.7},
            {"value": 2, "label": "High", "probability": 0.2},
        ])
    );
    for answer in body["answers"].as_array().expect("answers") {
        assert!(answer.get("confidence").is_none(), "{answer}");
        assert_ne!(answer["type"], "refusal");
    }
    assert_eq!(
        body["usage"],
        json!({"input_tokens": 33, "output_tokens": 0, "total_tokens": 33})
    );
    assert_eq!(body["experimental"]["calibration_scope"], SCOPE);
    assert_eq!(body["experimental"]["renderer"], DECISION_RENDERER);
    assert_eq!(body["experimental"]["renderer_source"], "artifact");
    // Each question is one native request over the same input.
    let seen = seen.lock().expect("lock");
    assert_eq!(seen.len(), 3);
    assert!(seen.iter().all(|request| request.state == "Diff:\n-a\n+b"));
    assert_eq!(seen[0].question, "Does this need a comment?");
    assert_eq!(seen[0].kind, DecisionKind::Predicate);
    assert_eq!(
        seen[1].kind,
        DecisionKind::Choice(vec![
            DecisionValue::Bool(true),
            DecisionValue::Text("maybe".into())
        ])
    );
    assert_eq!(
        seen[2].kind,
        DecisionKind::Score(vec!["Low".into(), "Medium".into(), "High".into()])
    );
}

#[test]
fn refuses_what_the_head_cannot_answer_faithfully() {
    let question = json!({"type": "predicate", "instructions": "Q?"});
    let with = |input: Value, questions: Value| json!({"input": input, "questions": questions});
    for (body, expected) in [
        (
            with(json!([{"role": "user", "content": "x"}]), json!([question])),
            "input must be a text string: messages and images",
        ),
        (with(json!("  "), json!([question])), "input is empty"),
        (with(json!("x"), json!([])), "questions must hold 1..=200"),
        (
            with(
                json!("x"),
                json!([{"type": "choice", "instructions": "Q?", "choices": [
                    {"value": "a", "description": "the first"}, {"value": "b"}
                ]}]),
            ),
            "questions[0]: choice descriptions are not supported",
        ),
        (
            with(
                json!("x"),
                json!([{"type": "score", "instructions": "Q?", "levels": [
                    {"label": "a"}, {"label": "b", "description": "high"}
                ]}]),
            ),
            "level descriptions are not supported",
        ),
        (
            with(
                json!("x"),
                json!([{"type": "score", "instructions": "Q?", "levels": [{"label": "a"}]}]),
            ),
            "levels must hold 2..=10 levels, got 1",
        ),
        (
            with(
                json!("x"),
                json!([{"type": "choice", "instructions": "Q?", "choices":
                    (0..27).map(|i| json!({"value": format!("v{i}")})).collect::<Vec<_>>()
                }]),
            ),
            "choices must hold 2..=26 values",
        ),
        (
            with(
                json!("x"),
                json!([question, {"type": "choice", "instructions": "Q?", "choices": [
                    {"value": "a\nb"}, {"value": "c"}
                ]}]),
            ),
            "questions[1]: invalid argument: decision: option 0 spans more than one line",
        ),
        (
            with(
                json!("x"),
                json!([{"type": "choice", "instructions": "Q?", "choices": [
                    {"value": true}, {"value": "Yes"}
                ]}]),
            ),
            "duplicate",
        ),
        (
            with(
                json!("x"),
                json!([{"type": "predicate", "instructions": " "}]),
            ),
            "question is empty",
        ),
        (
            with(
                json!("x"),
                json!([{"type": "predicate", "instructions": "Q?", "confidence": 0.9}]),
            ),
            "unknown field `confidence`",
        ),
        (
            with(
                json!("x"),
                json!([{"type": "refusal", "instructions": "Q?"}]),
            ),
            "unknown variant `refusal`",
        ),
        (
            json!({"input": "x", "questions": [question], "stream": true}),
            "unknown field `stream`",
        ),
    ] {
        let error = refused(&body);
        assert!(error.contains(expected), "{error:?} lacks {expected:?}");
    }
    // Empty descriptions say nothing, so they are not refused.
    prepared(
        &json!({"input": "x", "questions": [{"type": "choice", "instructions": "Q?",
        "choices": [{"value": "a", "description": ""}, {"value": "b"}]}]}),
    );
}

/// A pending decision that never resolves and reports when it is dropped.
struct Held {
    _dropped: oneshot::Sender<()>,
}

impl Future for Held {
    type Output = crate::Result<Option<Decision>>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        std::task::Poll::Pending
    }
}

#[tokio::test]
async fn shutdown_ends_a_waiting_request_and_cancels_its_decision() {
    let request = prepared(&json!({"input": "x", "questions": [
        {"type": "predicate", "instructions": "A?"}, {"type": "predicate", "instructions": "B?"}
    ]}));
    let (dropped_sender, dropped) = oneshot::channel();
    let mut dropped_sender = Some(dropped_sender);
    let submitted = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&submitted);
    let submit = move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        let held = Held {
            _dropped: dropped_sender.take().expect("one submission"),
        };
        std::future::ready(Ok(held))
    };
    let (closing, receiver) = watch::channel(false);
    let task = tokio::spawn(async move { answer(&request, submit, receiver).await });
    tokio::task::yield_now().await;
    closing.send_replace(true);
    let outcome = task.await.expect("task");
    assert!(matches!(outcome, Err(Failure::ShuttingDown)), "{outcome:?}");
    // The decision was dropped, which is what cancels it in the engine, and
    // the second question was never submitted.
    assert!(dropped.await.is_err());
    assert_eq!(submitted.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_dropped_request_drops_its_decision() {
    let request = prepared(&json!({"input": "x", "questions": [
        {"type": "predicate", "instructions": "A?"}
    ]}));
    let (dropped_sender, mut dropped) = oneshot::channel();
    let mut dropped_sender = Some(dropped_sender);
    let submit = move |_| {
        std::future::ready(Ok(Held {
            _dropped: dropped_sender.take().expect("one submission"),
        }))
    };
    let task = tokio::spawn(async move { answer(&request, submit, open()).await });
    tokio::task::yield_now().await;
    assert!(
        dropped
            .try_recv()
            .is_err_and(|e| e == oneshot::error::TryRecvError::Empty)
    );
    task.abort();
    assert!(task.await.is_err_and(|error| error.is_cancelled()));
    assert!(dropped.await.is_err());
}

#[tokio::test]
async fn questions_are_submitted_one_at_a_time() {
    let request = prepared(&json!({"input": "x", "questions": [
        {"type": "predicate", "instructions": "A?"}, {"type": "predicate", "instructions": "B?"}
    ]}));
    let (first_sender, first) = oneshot::channel::<crate::Result<Option<Decision>>>();
    let mut first = Some(first);
    let submitted = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&submitted);
    let submit = move |request: DecisionRequest| {
        counter.fetch_add(1, Ordering::SeqCst);
        let receiver = first.take();
        let kind = request.kind;
        std::future::ready(Ok(async move {
            match receiver {
                Some(receiver) => receiver.await.expect("sent"),
                None => Ok(Some(decision(&kind, &[0.5, 0.5], 1))),
            }
        }))
    };
    let task = tokio::spawn(async move { answer(&request, submit, open()).await });
    tokio::task::yield_now().await;
    assert_eq!(submitted.load(Ordering::SeqCst), 1);
    let _ = first_sender.send(Ok(Some(decision(&DecisionKind::Predicate, &[0.9, 0.1], 4))));
    let answered = task.await.expect("task").expect("answered");
    assert_eq!(submitted.load(Ordering::SeqCst), 2);
    let body = response_json("m", answered);
    assert_eq!(body["answers"][0]["probability"], 0.9);
    assert_eq!(body["usage"]["input_tokens"], 5);
}

#[tokio::test]
async fn engine_outcomes_map_to_failures() {
    let request = prepared(&json!({"input": "x", "questions": [
        {"type": "predicate", "instructions": "A?"}
    ]}));
    let full = |_| std::future::ready(Err::<std::future::Pending<_>, _>(crate::Error::QueueFull));
    let outcome = answer(&request, full, open()).await;
    assert!(matches!(outcome, Err(Failure::QueueFull)), "{outcome:?}");
    let cancelled = |_| std::future::ready(Ok(std::future::ready(Ok(None))));
    let outcome = answer(&request, cancelled, open()).await;
    assert!(matches!(outcome, Err(Failure::Cancelled)), "{outcome:?}");
    let invalid = |_| {
        std::future::ready(Err::<std::future::Pending<_>, _>(
            crate::Error::InvalidArgument("decision: text contains a special-token string".into()),
        ))
    };
    let outcome = answer(&request, invalid, open()).await;
    assert!(
        matches!(
            &outcome,
            Err(Failure::Engine(crate::Error::InvalidArgument(_)))
        ),
        "{outcome:?}"
    );
    let nan = |request: DecisionRequest| {
        std::future::ready(Ok(std::future::ready(Ok(Some(decision(
            &request.kind,
            &[f64::NAN, 0.5],
            1,
        ))))))
    };
    let outcome = answer(&request, nan, open()).await;
    assert!(matches!(outcome, Err(Failure::Internal(_))), "{outcome:?}");
}

/// The response reports the provenance the decisions carried, not a fixed
/// claim: a legacy head's assumed renderer and unknown scope pass through.
#[tokio::test]
async fn the_response_reports_the_decisions_provenance() {
    let request = prepared(&json!({"input": "x", "questions": [
        {"type": "predicate", "instructions": "A?"},
        {"type": "predicate", "instructions": "B?"},
    ]}));
    let legacy = |request: DecisionRequest| {
        let mut outcome = decision(&request.kind, &[0.5, 0.5], 1);
        outcome.renderer_declared = false;
        outcome.calibration_scope = UNKNOWN_CALIBRATION_SCOPE.into();
        std::future::ready(Ok(std::future::ready(Ok(Some(outcome)))))
    };
    let body = response_json(
        "bonsai",
        answer(&request, legacy, open()).await.expect("answered"),
    );
    assert_eq!(body["experimental"]["renderer"], DECISION_RENDERER);
    assert_eq!(body["experimental"]["renderer_source"], "assumed");
    assert_eq!(
        body["experimental"]["calibration_scope"],
        UNKNOWN_CALIBRATION_SCOPE
    );
}

/// A minimal valid judgment-head file of `width`, removed on drop.
struct HeadFile(std::path::PathBuf);

impl HeadFile {
    fn new(width: usize) -> Self {
        Self::with_metadata(width, &[])
    }

    /// A head whose metadata also carries `extra` key/value pairs.
    fn with_metadata(width: usize, extra: &[(&str, &str)]) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let tensors = [
            ("q.weight", vec![1, width]),
            ("q.bias", vec![1]),
            ("k.weight", vec![1, width]),
            ("k.bias", vec![1]),
            ("temperature", vec![1]),
        ];
        let mut header = serde_json::Map::new();
        let mut metadata = json!({
            "format": local_engine::judgment::JUDGMENT_HEAD_FORMAT,
            "experimental": "true",
            "width": width.to_string(),
            "head_dim": "1",
        });
        for (key, value) in extra {
            metadata[*key] = json!(value);
        }
        header.insert("__metadata__".into(), metadata);
        let mut data = Vec::new();
        for (name, shape) in tensors {
            let count: usize = shape.iter().product();
            let start = data.len();
            for _ in 0..count {
                data.extend_from_slice(&1.0f32.to_le_bytes());
            }
            header.insert(
                name.into(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&header).expect("header");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        let path = std::env::temp_dir().join(format!(
            "local-ai-decision-head-{}-{}.safetensors",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::write(&path, bytes).expect("write head");
        Self(path)
    }
}

impl Drop for HeadFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn the_head_is_checked_at_startup_and_admission_is_bounded() {
    let head = HeadFile::new(4);
    let error = Decisions::open(&head.0, 8)
        .err()
        .expect("a head of the wrong width is refused")
        .to_string();
    assert!(
        error.contains(FLAG) && error.contains("does not match the model width 8"),
        "{error}"
    );
    let missing = head.0.with_extension("missing");
    let error = Decisions::open(&missing, 4)
        .err()
        .expect("a missing head is refused")
        .to_string();
    assert!(
        error.contains("cannot open --experimental-decision-head"),
        "{error}"
    );

    let decisions = Decisions::open(&head.0, 4).expect("valid head");
    let permits: Vec<_> = (0..MAX_CONCURRENT)
        .map(|_| decisions.admit().expect("a free slot"))
        .collect();
    assert!(decisions.admit().is_none(), "admission is bounded");
    drop(permits);
    assert!(decisions.admit().is_some(), "slots are released");

    assert!(!decisions.closing());
    let receiver = decisions.closing.subscribe();
    decisions.close();
    decisions.close();
    assert!(decisions.closing() && *receiver.borrow());
}

/// Startup reports the file's own provenance and refuses a renderer the
/// native decisions do not reproduce; a legacy file's is assumed, its
/// calibration unknown.
#[test]
fn startup_reports_the_heads_own_provenance() {
    let legacy = Decisions::open(&HeadFile::new(4).0, 4).expect("legacy head");
    let notice = legacy.provenance();
    assert!(
        notice.contains("assumed") && notice.contains(UNKNOWN_CALIBRATION_SCOPE),
        "{notice}"
    );

    let staged = HeadFile::with_metadata(
        4,
        &[
            ("mode", "development"),
            ("renderer", DECISION_RENDERER),
            ("calibration_scope", SCOPE),
        ],
    );
    let notice = Decisions::open(&staged.0, 4)
        .expect("declared head")
        .provenance();
    assert!(
        notice.contains("declared by the head") && notice.contains(SCOPE),
        "{notice}"
    );

    let unscoped = HeadFile::with_metadata(4, &[("mode", "development")]);
    let notice = Decisions::open(&unscoped.0, 4)
        .expect("unscoped head")
        .provenance();
    assert!(notice.contains(UNKNOWN_CALIBRATION_SCOPE), "{notice}");

    let foreign = HeadFile::with_metadata(4, &[("renderer", "kev-hard-v1-judgment-render.v1")]);
    let error = Decisions::open(&foreign.0, 4)
        .err()
        .expect("a foreign renderer is refused")
        .to_string();
    assert!(
        error.contains(FLAG) && error.contains("kev-hard-v1-judgment-render.v1"),
        "{error}"
    );
}

#[test]
fn the_flag_is_explicit_and_the_openai_route_explains_itself() {
    use super::super::options::parse;
    let args = |values: &[&str]| values.iter().map(|v| (*v).to_owned()).collect::<Vec<_>>();
    assert_eq!(parse(&args(&[])).expect("defaults").decision_head, None);
    let parsed = parse(&args(&[FLAG, "/tmp/head.safetensors"])).expect("flag");
    assert_eq!(
        parsed.decision_head.as_deref(),
        Some(std::path::Path::new("/tmp/head.safetensors"))
    );
    assert!(parse(&args(&[FLAG])).is_err());
    assert!(parse(&args(&[FLAG, ""])).is_err());

    let message = super::openai_refusal();
    for expected in [
        "/v1/decisions",
        "confidence",
        "refusals",
        FLAG,
        super::ROUTE,
    ] {
        assert!(message.contains(expected), "{message}");
    }
}
