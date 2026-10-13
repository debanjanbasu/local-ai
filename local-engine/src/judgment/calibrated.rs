//! Calibrated, scope-gated judgment decisions: what `POST /v1/decisions`
//! answers with.
//!
//! A [`CalibratedHead`] is opened from a directory holding three files the
//! pre-registered evaluation produced (`tools/judgment_reeval.py`,
//! `docs/DECISIONS_EVALUATION.md` part 3):
//!
//! - `head.safetensors`: the judgment pointer head ([`JudgmentHead`]);
//! - `gate.safetensors`: the out-of-scope feature gate, a PCA of training
//!   `Decision:` hidden states (format [`GATE_FORMAT`]);
//! - `calibration.json`: per-family calibration and refusal thresholds fitted
//!   on the calibration split, the gate threshold, the declared scope, the
//!   evaluation's provenance and acceptance, and a content hash (format
//!   [`CALIBRATION_FORMAT`]).
//!
//! Opening refuses the directory unless the artifact's content hash matches
//! its canonical JSON (keys sorted, compact, UTF-8: what
//! `judgment_calibrate.canonical_sha256` computes), the head and gate files
//! hash to the artifact's `head_sha256` and `gate_sha256`, its acceptance
//! says it passed, and its declared scope is exactly [`DECLARED_SCOPE`], the
//! one this code checks.
//!
//! [`CalibratedHead::judge`] turns one native [`Decision`] of a question in
//! scope into a calibrated `P(Yes)`, its top-label confidence and either an
//! answer or a refusal: prompt too long, gate distance above the threshold,
//! a family with no qualifying threshold, or confidence below it. Questions
//! outside the declared scope ([`DecisionFamily::of`] is `None`) are refused
//! before any GPU work.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use super::{Decision, DecisionKind, DecisionRequest, DecisionValue, JudgmentHead};

/// The `format` of the calibration artifact this code reads.
pub const CALIBRATION_FORMAT: &str = "local-ai.judgment-calibration.v2";

/// The `format` metadata of the gate file this code reads.
pub const GATE_FORMAT: &str = "local-ai.judgment-gate.v1";

/// The only `CodeReviewer` question in scope, verbatim.
pub const CODEREVIEWER_QUESTION: &str = "Does this change need a reviewer comment?";

/// Every `CommitPackFT` question in scope starts with this, continues with a
/// non-empty single-line message and ends with `"`.
pub const COMMIT_MESSAGE_PREFIX: &str = "Does this commit message describe this diff? Message: \"";

/// Inputs in scope start with one of these (the rendered diff states).
pub const STATE_PREFIXES: [&str; 2] = ["Diff:\n", "Context before hunk:\n"];

/// The declared scope an artifact must carry.
///
/// This is `judgment_calibrate.py`'s `DEVTOOLS_SCOPE` as it serializes it.
/// [`DecisionFamily::of`] implements it
/// (predicates only, which is narrower than its `kinds`) and the prompt-token
/// limit is checked in [`CalibratedHead::judge`].
pub const DECLARED_SCOPE: &str = r#"{"kinds": ["predicate", "choice over the booleans true/false"], "option_sets": [["No", "Yes"]], "question_patterns": ["Does\\ this\\ change\\ need\\ a\\ reviewer\\ comment\\?", "Does this commit message describe this diff\\? Message: \"[^\\n]+\""], "state_prefixes": ["Diff:\n", "Context before hunk:\n"], "max_prompt_tokens": 2048}"#;

const ARTIFACT_FILE: &str = "calibration.json";
const HEAD_FILE: &str = "head.safetensors";
const GATE_FILE: &str = "gate.safetensors";
const MAX_HEADER: u64 = 1 << 20;

fn invalid<T>(message: impl Into<String>) -> crate::Result<T> {
    Err(crate::Error::InvalidFormat(format!(
        "decision calibration: {}",
        message.into()
    )))
}

/// A declared question family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionFamily {
    /// "Does this change need a reviewer comment?" over a `CodeReviewer` hunk.
    CodeReviewerNeedsComment,
    /// "Does this commit message describe this diff? Message: …" over a diff.
    CommitMessageMatch,
}

impl DecisionFamily {
    const ALL: [Self; 2] = [Self::CodeReviewerNeedsComment, Self::CommitMessageMatch];

    /// The family's key in the artifact.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::CodeReviewerNeedsComment => "codereviewer_needs_comment",
            Self::CommitMessageMatch => "commitpackft_message",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::CodeReviewerNeedsComment => 0,
            Self::CommitMessageMatch => 1,
        }
    }

    /// The declared family `request` belongs to, or `None` when it is out of
    /// scope: not a predicate, an input that is not a rendered diff state, or
    /// a question that is not one of the declared ones exactly.
    #[must_use]
    pub fn of(request: &DecisionRequest) -> Option<Self> {
        if request.kind != DecisionKind::Predicate
            || !STATE_PREFIXES
                .iter()
                .any(|prefix| request.state.starts_with(prefix))
        {
            return None;
        }
        let question = request.question.as_str();
        if question == CODEREVIEWER_QUESTION {
            return Some(Self::CodeReviewerNeedsComment);
        }
        let message = question
            .strip_prefix(COMMIT_MESSAGE_PREFIX)?
            .strip_suffix('"')?;
        (!message.is_empty() && !message.contains('\n')).then_some(Self::CommitMessageMatch)
    }
}

/// A fitted map from raw logits to a calibrated `P(Yes)`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Calibrator {
    /// `softmax(raw / T)`.
    Temperature(f64),
    /// `sigmoid(slope · (raw_yes − raw_no) + bias)`.
    Platt { slope: f64, bias: f64 },
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

impl Calibrator {
    // The same operations, in the same order, as the Python evaluation.
    #[allow(clippy::suboptimal_flops)]
    fn p_yes(self, raw_yes: f64, raw_no: f64) -> f64 {
        match self {
            Self::Temperature(t) => {
                // The two-option softmax, as judgment_calibrate computes it.
                let (a, b) = (raw_yes / t, raw_no / t);
                let peak = a.max(b);
                let total = peak + ((a - peak).exp() + (b - peak).exp()).ln();
                (a - total).exp()
            }
            Self::Platt { slope, bias } => sigmoid(slope * (raw_yes - raw_no) + bias),
        }
    }
}

/// Why a question in scope was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    /// The rendered prompt is longer than the declared limit.
    PromptTooLong { tokens: usize, limit: usize },
    /// The feature gate places the input outside the training distribution.
    Gate { distance: f64, threshold: f64 },
    /// The calibration split found no threshold meeting the target risk for
    /// this family, so it is always refused.
    FamilyRefused,
    /// The calibrated confidence is below the family's threshold.
    LowConfidence { confidence: f64, threshold: f64 },
}

/// One calibrated decision.
#[derive(Clone, Debug, PartialEq)]
pub struct Judgment {
    /// The declared family the question belongs to.
    pub family: DecisionFamily,
    /// Calibrated probability that the answer is Yes.
    pub probability_yes: f64,
    /// Calibrated probability of the more likely answer.
    pub confidence: f64,
    /// The gate's distance for this input.
    pub gate_distance: f64,
    /// `None` answers the question; otherwise why it is refused.
    pub refusal: Option<Refusal>,
}

/// The out-of-scope gate: PCA Mahalanobis distance plus residual energy.
#[derive(Debug)]
struct Gate {
    width: usize,
    mean: Vec<f64>,
    /// `components × width`, row-major.
    components: Vec<f64>,
    variance: Vec<f64>,
    residual_variance: f64,
}

impl Gate {
    fn open(bytes: &[u8]) -> crate::Result<Self> {
        let (metadata, mut tensors) = read_f32_safetensors(bytes)?;
        if metadata.get("format").and_then(|v| v.as_str()) != Some(GATE_FORMAT) {
            return invalid(format!("gate format must be {GATE_FORMAT}"));
        }
        let mut take = |name: &str| {
            tensors
                .iter()
                .position(|(n, _, _)| n == name)
                .map(|i| tensors.swap_remove(i))
                .ok_or_else(|| crate::Error::MissingTensor(name.to_owned()))
        };
        let (_, mean_shape, mean) = take("mean")?;
        let (_, comp_shape, components) = take("components")?;
        let (_, var_shape, variance) = take("variance")?;
        let (_, res_shape, residual) = take("residual_variance")?;
        let [width] = mean_shape[..] else {
            return invalid("gate mean must be one-dimensional");
        };
        let [k, comp_width] = comp_shape[..] else {
            return invalid("gate components must be two-dimensional");
        };
        if comp_width != width || var_shape != [k] || res_shape != [1] || k == 0 || k >= width {
            return invalid("gate tensor shapes do not agree");
        }
        let residual_variance = f64::from(residual[0]);
        if residual_variance <= 0.0 || variance.iter().any(|&v| v <= 0.0) {
            return invalid("gate variances must be positive");
        }
        Ok(Self {
            width,
            mean: mean.into_iter().map(f64::from).collect(),
            components: components.into_iter().map(f64::from).collect(),
            variance: variance.into_iter().map(f64::from).collect(),
            residual_variance,
        })
    }

    // The same operations, in the same order, as the Python evaluation.
    #[allow(clippy::suboptimal_flops)]
    fn distance(&self, decide: &[f32]) -> f64 {
        let y: Vec<f64> = decide
            .iter()
            .zip(&self.mean)
            .map(|(&x, &m)| f64::from(x) - m)
            .collect();
        let mut residual = y.clone();
        let mut total = 0.0;
        for (row, &var) in self.components.chunks_exact(self.width).zip(&self.variance) {
            let z: f64 = row.iter().zip(&y).map(|(a, b)| a * b).sum();
            total += z * z / var;
            for (r, &a) in residual.iter_mut().zip(row) {
                *r -= a * z;
            }
        }
        let energy: f64 = residual.iter().map(|r| r * r).sum();
        (total + energy / self.residual_variance) / self.width as f64
    }
}

/// A head with the calibration, thresholds, gate and scope it passed with.
#[derive(Debug)]
pub struct CalibratedHead {
    head: Arc<JudgmentHead>,
    gate: Gate,
    calibrators: [Calibrator; 2],
    thresholds: [Option<f64>; 2],
    gate_threshold: f64,
    max_prompt_tokens: usize,
    content_sha256: String,
    head_sha256: String,
    gate_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

fn finite(value: Option<&Json>, what: &str) -> crate::Result<f64> {
    match value.and_then(Json::as_f64) {
        Some(v) if v.is_finite() => Ok(v),
        _ => invalid(format!("{what} must be a finite number")),
    }
}

impl CalibratedHead {
    /// Open and verify the artifact directory `dir` (see the module docs).
    pub fn open(dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = dir.as_ref();
        let text = std::fs::read_to_string(dir.join(ARTIFACT_FILE))?;
        let head_bytes = std::fs::read(dir.join(HEAD_FILE))?;
        let gate_bytes = std::fs::read(dir.join(GATE_FILE))?;
        Self::from_parts(&text, &head_bytes, &gate_bytes, &dir.join(HEAD_FILE))
    }

    fn from_parts(
        artifact: &str,
        head_bytes: &[u8],
        gate_bytes: &[u8],
        head_path: &Path,
    ) -> crate::Result<Self> {
        let artifact = Json::parse(artifact)?;
        let Json::Object(fields) = &artifact else {
            return invalid("calibration.json must be an object");
        };
        let claimed = artifact.get("content_sha256").and_then(Json::as_str);
        let body = Json::Object(
            fields
                .iter()
                .filter(|(key, _)| key != "content_sha256")
                .cloned()
                .collect(),
        );
        let content_sha256 = sha256_hex(body.canonical().as_bytes());
        if claimed != Some(content_sha256.as_str()) {
            return invalid("content hash does not match the artifact");
        }
        if artifact.get("format").and_then(Json::as_str) != Some(CALIBRATION_FORMAT) {
            return invalid(format!("format must be {CALIBRATION_FORMAT}"));
        }
        let passed = artifact
            .get("acceptance")
            .and_then(|a| a.get("passed"))
            .and_then(Json::as_bool);
        if passed != Some(true) {
            return invalid(
                "the artifact did not pass its pre-registered acceptance criteria; refusing to \
                 serve it",
            );
        }
        let head_sha256 = sha256_hex(head_bytes);
        if artifact.get("head_sha256").and_then(Json::as_str) != Some(head_sha256.as_str()) {
            return invalid("head.safetensors does not hash to the artifact's head_sha256");
        }
        let gate_sha256 = sha256_hex(gate_bytes);
        let gate_declared = artifact.get("gate");
        if artifact.get("gate_sha256").and_then(Json::as_str) != Some(gate_sha256.as_str())
            || gate_declared
                .and_then(|g| g.get("file_sha256"))
                .and_then(Json::as_str)
                != Some(gate_sha256.as_str())
        {
            return invalid("gate.safetensors does not hash to the artifact's gate_sha256");
        }
        let scope = artifact.get("scope").map(Json::canonical);
        if scope != Some(Json::parse(DECLARED_SCOPE)?.canonical()) {
            return invalid("declared scope is not the one this server implements");
        }
        let max_prompt_tokens = finite(
            artifact
                .get("scope")
                .and_then(|s| s.get("max_prompt_tokens")),
            "scope.max_prompt_tokens",
        )? as usize;
        let gate_threshold = finite(
            gate_declared.and_then(|g| g.get("threshold")),
            "gate.threshold",
        )?;
        let (calibrators, thresholds) = families(&artifact)?;
        let head = JudgmentHead::open(head_path)?;
        let gate = Gate::open(gate_bytes)?;
        if gate.width != head.width() {
            return invalid("gate width does not match the head width");
        }
        Ok(Self {
            head: Arc::new(head),
            gate,
            calibrators,
            thresholds,
            gate_threshold,
            max_prompt_tokens,
            content_sha256,
            head_sha256,
            gate_sha256,
        })
    }

    /// The head native decisions are scored with.
    #[must_use]
    pub fn head(&self) -> Arc<JudgmentHead> {
        Arc::clone(&self.head)
    }

    /// The artifact's verified content hash.
    #[must_use]
    pub fn content_sha256(&self) -> &str {
        &self.content_sha256
    }

    /// The verified head file hash.
    #[must_use]
    pub fn head_sha256(&self) -> &str {
        &self.head_sha256
    }

    /// The verified gate file hash.
    #[must_use]
    pub fn gate_sha256(&self) -> &str {
        &self.gate_sha256
    }

    /// The family's refusal threshold; `None` refuses the whole family.
    #[must_use]
    pub const fn threshold(&self, family: DecisionFamily) -> Option<f64> {
        self.thresholds[family.index()]
    }

    /// Calibrate a native predicate `decision` of a `family` question.
    ///
    /// The decision must come from [`head`](Self::head) and carry its
    /// `Decision:` row ([`Decision::decide_row`]).
    pub fn judge(&self, family: DecisionFamily, decision: &Decision) -> crate::Result<Judgment> {
        if decision.decide_row.len() != self.gate.width {
            return Err(crate::Error::InvalidArgument(format!(
                "decision carries {} decision-row values, the gate needs {}",
                decision.decide_row.len(),
                self.gate.width
            )));
        }
        let raw = |value: bool| {
            decision
                .options
                .iter()
                .find(|option| option.option == DecisionValue::Bool(value))
                .and_then(|option| option.logit)
                .map(|logit| logit * f64::from(self.head.temperature()))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument("decision is not a scored predicate".into())
                })
        };
        let probability_yes = self.calibrators[family.index()].p_yes(raw(true)?, raw(false)?);
        let confidence = probability_yes.max(1.0 - probability_yes);
        let gate_distance = self.gate.distance(&decision.decide_row);
        let refusal = if decision.prompt_tokens > self.max_prompt_tokens {
            Some(Refusal::PromptTooLong {
                tokens: decision.prompt_tokens,
                limit: self.max_prompt_tokens,
            })
        } else if gate_distance.is_nan() || gate_distance > self.gate_threshold {
            Some(Refusal::Gate {
                distance: gate_distance,
                threshold: self.gate_threshold,
            })
        } else {
            match self.thresholds[family.index()] {
                None => Some(Refusal::FamilyRefused),
                Some(threshold) if confidence < threshold => Some(Refusal::LowConfidence {
                    confidence,
                    threshold,
                }),
                Some(_) => None,
            }
        };
        Ok(Judgment {
            family,
            probability_yes,
            confidence,
            gate_distance,
            refusal,
        })
    }
}

/// The per-family calibrators and thresholds of `artifact`.
fn families(artifact: &Json) -> crate::Result<([Calibrator; 2], [Option<f64>; 2])> {
    let Some(Json::Object(families)) = artifact.get("families") else {
        return invalid("families must be an object");
    };
    if families.len() != DecisionFamily::ALL.len() {
        return invalid("families must be exactly the declared ones");
    }
    let mut calibrators = [Calibrator::Temperature(1.0); 2];
    let mut thresholds = [None; 2];
    for family in DecisionFamily::ALL {
        let entry = artifact
            .get("families")
            .and_then(|f| f.get(family.key()))
            .ok_or_else(|| {
                crate::Error::InvalidFormat(format!(
                    "decision calibration: family {} is missing",
                    family.key()
                ))
            })?;
        let calibration = entry.get("calibration");
        let method = calibration
            .and_then(|c| c.get("method"))
            .and_then(Json::as_str);
        calibrators[family.index()] = match method {
            Some("temperature") => {
                let t = finite(
                    calibration.and_then(|c| c.get("temperature")),
                    "temperature",
                )?;
                if t <= 0.0 {
                    return invalid("temperature must be positive");
                }
                Calibrator::Temperature(t)
            }
            Some("platt") => Calibrator::Platt {
                slope: finite(calibration.and_then(|c| c.get("slope")), "slope")?,
                bias: finite(calibration.and_then(|c| c.get("bias")), "bias")?,
            },
            _ => return invalid(format!("{}: unknown calibration method", family.key())),
        };
        thresholds[family.index()] = match entry.get("threshold") {
            Some(Json::Null) => None,
            value => {
                let t = finite(value, "threshold")?;
                if !(0.0..=1.0).contains(&t) {
                    return invalid("threshold must be in [0, 1]");
                }
                Some(t)
            }
        };
    }
    Ok((calibrators, thresholds))
}

/// `(metadata, [(name, shape, values)])` of an F32-only safetensors file.
#[allow(clippy::type_complexity)]
fn read_f32_safetensors(
    bytes: &[u8],
) -> crate::Result<(
    serde_json::Map<String, serde_json::Value>,
    Vec<(String, Vec<usize>, Vec<f32>)>,
)> {
    let Some(header_len) = bytes.first_chunk::<8>().map(|b| u64::from_le_bytes(*b)) else {
        return invalid("gate file is too short");
    };
    if header_len == 0 || header_len > MAX_HEADER || header_len + 8 > bytes.len() as u64 {
        return invalid("gate safetensors header length is invalid");
    }
    let start = 8 + header_len as usize;
    let data = &bytes[start..];
    let mut header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..start])?;
    let Some(serde_json::Value::Object(metadata)) = header.remove("__metadata__") else {
        return invalid("gate metadata is missing");
    };
    let mut tensors = Vec::with_capacity(header.len());
    let mut covered = 0usize;
    for (name, entry) in header {
        let dtype = entry.get("dtype").and_then(serde_json::Value::as_str);
        let shape: Vec<usize> = serde_json::from_value(entry["shape"].clone())?;
        let [begin, end]: [usize; 2] = serde_json::from_value(entry["data_offsets"].clone())?;
        let expected = shape.iter().try_fold(4usize, |n, &d| n.checked_mul(d));
        if dtype != Some("F32") || end < begin || Some(end - begin) != expected || end > data.len()
        {
            return invalid(format!("gate tensor {name} is not a valid F32 tensor"));
        }
        let values: Vec<f32> = data[begin..end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        if !values.iter().all(|v| v.is_finite()) {
            return invalid(format!("gate tensor {name} holds a non-finite value"));
        }
        covered += end - begin;
        tensors.push((name, shape, values));
    }
    if covered != data.len() {
        return invalid("gate tensor data has gaps or overlaps");
    }
    Ok((metadata, tensors))
}

/// A JSON value that keeps each number's text, so an artifact written by
/// Python re-serializes to the same canonical bytes Python hashed.
#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl Json {
    fn parse(text: &str) -> crate::Result<Self> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            at: 0,
        };
        let value = parser.value(0)?;
        parser.space();
        if parser.at != parser.bytes.len() {
            return invalid("trailing characters after the JSON value");
        }
        Ok(value)
    }

    fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(text) => text.parse().ok(),
            _ => None,
        }
    }

    /// Python's `json.dumps(value, sort_keys=True, separators=(",", ":"),
    /// ensure_ascii=False)`.
    fn canonical(&self) -> String {
        let mut out = String::new();
        self.write_canonical(&mut out);
        out
    }

    fn write_canonical(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Number(text) => out.push_str(text),
            Self::String(text) => write_string(text, out),
            Self::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write_canonical(out);
                }
                out.push(']');
            }
            Self::Object(fields) => {
                let mut sorted: Vec<&(String, Self)> = fields.iter().collect();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                out.push('{');
                for (index, (key, value)) in sorted.into_iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write_canonical(out);
                }
                out.push('}');
            }
        }
    }
}

/// A string as Python's `json` writes it with `ensure_ascii=False`.
fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

const MAX_DEPTH: usize = 64;

impl Parser<'_> {
    fn space(&mut self) {
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.at += 1;
        }
    }

    fn eat(&mut self, literal: &str) -> bool {
        if self.bytes[self.at..].starts_with(literal.as_bytes()) {
            self.at += literal.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> crate::Result<Json> {
        if depth > MAX_DEPTH {
            return invalid("JSON nests too deeply");
        }
        self.space();
        match self.bytes.get(self.at) {
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'"') => self.string().map(Json::String),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if self.eat("]") {
                    return Ok(Json::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.space();
                    if self.eat("]") {
                        return Ok(Json::Array(items));
                    }
                    if !self.eat(",") {
                        return invalid("expected , or ] in an array");
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut fields: Vec<(String, Json)> = Vec::new();
                self.space();
                if self.eat("}") {
                    return Ok(Json::Object(fields));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    self.space();
                    if !self.eat(":") {
                        return invalid("expected : in an object");
                    }
                    let value = self.value(depth + 1)?;
                    if fields.iter().any(|(k, _)| *k == key) {
                        return invalid(format!("duplicate key {key:?}"));
                    }
                    fields.push((key, value));
                    self.space();
                    if self.eat("}") {
                        return Ok(Json::Object(fields));
                    }
                    if !self.eat(",") {
                        return invalid("expected , or } in an object");
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.at;
                while self
                    .bytes
                    .get(self.at)
                    .is_some_and(|b| matches!(b, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
                {
                    self.at += 1;
                }
                let text = std::str::from_utf8(&self.bytes[start..self.at])
                    .map_err(|_| crate::Error::InvalidFormat("number is not UTF-8".into()))?;
                if text.parse::<f64>().is_err() {
                    return invalid(format!("invalid number {text:?}"));
                }
                Ok(Json::Number(text.to_owned()))
            }
            _ => invalid("unexpected character in JSON"),
        }
    }

    fn hex4(&mut self) -> crate::Result<u32> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .and_then(|d| std::str::from_utf8(d).ok())
            .and_then(|d| u32::from_str_radix(d, 16).ok());
        self.at += 4;
        digits.map_or_else(|| invalid("invalid \\u escape"), Ok)
    }

    fn string(&mut self) -> crate::Result<String> {
        if !self.eat("\"") {
            return invalid("expected a string");
        }
        let mut out = String::new();
        loop {
            let rest = &self.bytes[self.at..];
            let Some(stop) = rest.iter().position(|&b| b == b'"' || b == b'\\') else {
                return invalid("unterminated string");
            };
            out.push_str(
                std::str::from_utf8(&rest[..stop])
                    .map_err(|_| crate::Error::InvalidFormat("string is not UTF-8".into()))?,
            );
            self.at += stop;
            if self.eat("\"") {
                return Ok(out);
            }
            self.at += 1; // the backslash
            let escape = self.bytes.get(self.at).copied();
            self.at += 1;
            match escape {
                Some(b'"') => out.push('"'),
                Some(b'\\') => out.push('\\'),
                Some(b'/') => out.push('/'),
                Some(b'b') => out.push('\u{8}'),
                Some(b'f') => out.push('\u{c}'),
                Some(b'n') => out.push('\n'),
                Some(b'r') => out.push('\r'),
                Some(b't') => out.push('\t'),
                Some(b'u') => {
                    let mut code = self.hex4()?;
                    if (0xD800..0xDC00).contains(&code) && self.eat("\\u") {
                        let low = self.hex4()?;
                        if !(0xDC00..0xE000).contains(&low) {
                            return invalid("unpaired surrogate");
                        }
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    }
                    out.push(
                        char::from_u32(code).ok_or_else(|| {
                            crate::Error::InvalidFormat("invalid \\u escape".into())
                        })?,
                    );
                }
                _ => return invalid("invalid escape"),
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::doc_markdown
)]
mod tests;
