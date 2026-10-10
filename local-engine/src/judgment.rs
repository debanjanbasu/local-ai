//! EXPERIMENTAL CPU scorer for the head-only judgment pointer artifact.
//!
//! [`JudgmentHead::open`] loads the `head.safetensors` exported by
//! `tools/judgment_train.py` (format [`JUDGMENT_HEAD_FORMAT`]) and
//! [`JudgmentHead::score`] reproduces its documented formula on the CPU:
//!
//! ```text
//! q        = Q h_decide + b_q             Q: [head_dim, width], b_q: [head_dim]
//! k_i      = K h_option_i + b_k           K: [head_dim, width], b_k: [head_dim]
//! logit_i  = dot(k_i, q) / sqrt(head_dim) / temperature
//! p        = softmax(logit) over the options, in row order
//! ```
//!
//! # Artifact
//!
//! Exactly five little-endian `F32` tensors, row-major: `q.weight` and
//! `k.weight` `[head_dim, width]`, `q.bias` and `k.bias` `[head_dim]`,
//! `temperature` `[1]`. Metadata must carry `format` =
//! [`JUDGMENT_HEAD_FORMAT`], `experimental` = `"true"` and decimal `width` and
//! `head_dim` matching the tensor shapes. Every value must be finite and the
//! temperature strictly positive. Anything else is refused.
//!
//! Two optional metadata strings are retained as the artifact's provenance:
//! `renderer` (the prompt renderer its training rows used,
//! [`JudgmentHead::renderer`]) and `calibration_scope` (what its temperature
//! was fit on, [`JudgmentHead::calibration_scope`]). When present each must
//! be non-empty. Absent values stay unknown: nothing is inferred from other
//! metadata, and the CPU scorer accepts any renderer, since it only scores
//! features. Native decisions are stricter; see [`DecisionRequest`].
//!
//! # Feature representation
//!
//! `features` is `rows × width` `f32`s, position-major: the option rows first,
//! in option order, and the final decision row last; at least two options
//! (the training tool's minimum). Each row must be the output-normalized,
//! unrotated (Qwen-basis) hidden state *after* the token at that position —
//! exactly what `bonsai_native` `capture_features` returns for the same
//! positions. The current model's width is 5120; the caller must check that
//! [`JudgmentHead::width`] matches the model it captured from.
//!
//! Training features were stored as fp16, so for parity with the trained
//! distribution round each captured value through `half::f16` before scoring
//! (`f16::from_f32(x).to_f32()`). The scorer itself does no rounding; it
//! accumulates in `f64`.
//!
//! # Limitations
//!
//! The temperature was fit on one dataset's calibration split only. The
//! returned probabilities are the head's softmax output, not a calibrated
//! confidence on other data, and the scorer defines no argmax, threshold or
//! refusal policy.
//!
//! # Native decisions
//!
//! [`DecisionRequest`] renders a typed text question exactly as the training
//! rows were rendered; [`crate::Engine::decide`] and
//! [`crate::EngineHandle::decide`] capture its features on the loaded model
//! and score them with a caller-supplied head (never loaded automatically).
//! A [`Decision`] reports the head's probabilities per caller option, the
//! argmax and, for scores, the probability-weighted level index, together
//! with the loaded head's own `calibration_scope` (or
//! [`UNKNOWN_CALIBRATION_SCOPE`]) for what those probabilities are not.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

mod decision;

pub(crate) use self::decision::PreparedDecision;
pub use self::decision::{
    DECISION_CAPTURE_ROWS, DECISION_RENDERER, DECISION_RENDERERS, Decision, DecisionKind,
    DecisionProbability, DecisionRequest, DecisionValue, MAX_DECISION_OPTIONS, RenderedDecision,
    UNKNOWN_CALIBRATION_SCOPE,
};

/// The `format` metadata value the loader accepts.
pub const JUDGMENT_HEAD_FORMAT: &str = "local-ai.judgment-pointer-head.v0-experimental";

/// Largest safetensors JSON header accepted.
const MAX_HEADER: u64 = 1 << 20;

/// Fewest option rows a scored sample may hold (matches training).
const MIN_OPTIONS: usize = 2;

/// A loaded judgment pointer head. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct JudgmentHead {
    width: usize,
    head_dim: usize,
    q_weight: Vec<f32>,
    q_bias: Vec<f32>,
    k_weight: Vec<f32>,
    k_bias: Vec<f32>,
    temperature: f32,
    /// `renderer` metadata, if the artifact declared one.
    renderer: Option<String>,
    /// `calibration_scope` metadata, if the artifact declared one.
    calibration_scope: Option<String>,
}

/// Scores for one sample, one entry per option in the supplied row order.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgmentScores {
    /// Temperature-scaled logits.
    pub logits: Vec<f64>,
    /// Softmax of `logits` (sums to 1).
    pub probabilities: Vec<f64>,
}

fn invalid<T>(message: impl Into<String>) -> crate::Result<T> {
    Err(crate::Error::InvalidFormat(format!(
        "judgment head: {}",
        message.into()
    )))
}

/// Optional provenance metadata `key`: absent stays unknown, present must be
/// non-empty.
fn provenance(metadata: &HashMap<String, String>, key: &str) -> crate::Result<Option<String>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(value) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(_) => invalid(format!("{key} must not be empty when present")),
    }
}

impl JudgmentHead {
    /// Read and validate the artifact at `path`.
    pub fn open(path: impl AsRef<Path>) -> crate::Result<Self> {
        #[derive(Deserialize)]
        struct Entry {
            dtype: String,
            shape: Vec<usize>,
            data_offsets: [usize; 2],
        }
        let bytes = std::fs::read(path)?;
        let Some(header_len) = bytes.first_chunk::<8>().map(|b| u64::from_le_bytes(*b)) else {
            return invalid("file is too short");
        };
        if header_len == 0 || header_len > MAX_HEADER || header_len + 8 > bytes.len() as u64 {
            return invalid("safetensors header length is invalid");
        }
        let data_start = 8 + header_len as usize;
        let data = &bytes[data_start..];
        let mut entries: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..data_start])?;
        let metadata: HashMap<String, String> = match entries.remove("__metadata__") {
            Some(value) => serde_json::from_value(value)?,
            None => return invalid("metadata is missing"),
        };
        let field = |key: &str| metadata.get(key).map(String::as_str);
        if field("format") != Some(JUDGMENT_HEAD_FORMAT) {
            return invalid(format!("format must be {JUDGMENT_HEAD_FORMAT}"));
        }
        if field("experimental") != Some("true") {
            return invalid("experimental must be \"true\"");
        }
        let dimension = |key: &str| match field(key).and_then(|v| v.parse::<usize>().ok()) {
            Some(value) if value > 0 => Ok(value),
            _ => invalid(format!("{key} must be a positive integer")),
        };
        let (width, head_dim) = (dimension("width")?, dimension("head_dim")?);
        let renderer = provenance(&metadata, "renderer")?;
        let calibration_scope = provenance(&metadata, "calibration_scope")?;

        let names = ["q.weight", "q.bias", "k.weight", "k.bias", "temperature"];
        if entries.len() != names.len() {
            return invalid(format!(
                "must contain exactly {} tensors, found {}",
                names.len(),
                entries.len()
            ));
        }
        let mut ranges = Vec::with_capacity(names.len());
        let mut read = |name: &str, shape: &[usize]| -> crate::Result<Vec<f32>> {
            let value = entries
                .remove(name)
                .ok_or_else(|| crate::Error::MissingTensor(name.to_owned()))?;
            let entry: Entry = serde_json::from_value(value)?;
            if entry.dtype != "F32" || entry.shape != shape {
                return invalid(format!("tensor {name} must be F32 {shape:?}"));
            }
            let [start, end] = entry.data_offsets;
            let byte_len = shape.iter().try_fold(4usize, |n, &dim| n.checked_mul(dim));
            if end < start || Some(end - start) != byte_len || end > data.len() {
                return invalid(format!("tensor {name} has an invalid byte range"));
            }
            ranges.push([start, end]);
            let values: Vec<f32> = data[start..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect();
            if !values.iter().all(|v| v.is_finite()) {
                return invalid(format!("tensor {name} holds a non-finite value"));
            }
            Ok(values)
        };
        let q_weight = read("q.weight", &[head_dim, width])?;
        let q_bias = read("q.bias", &[head_dim])?;
        let k_weight = read("k.weight", &[head_dim, width])?;
        let k_bias = read("k.bias", &[head_dim])?;
        let temperature = read("temperature", &[1])?[0];
        ranges.sort_unstable();
        let mut cursor = 0;
        for [start, end] in ranges {
            if start != cursor {
                return invalid("tensor byte ranges overlap or leave a gap");
            }
            cursor = end;
        }
        if cursor != data.len() {
            return invalid("unreferenced bytes after tensor data");
        }
        if temperature <= 0.0 {
            return invalid("temperature must be positive");
        }
        Ok(Self {
            width,
            head_dim,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            temperature,
            renderer,
            calibration_scope,
        })
    }

    /// A head from raw parts, with no provenance metadata, for tests
    /// elsewhere in the crate.
    #[cfg(test)]
    pub(crate) const fn for_tests(
        width: usize,
        head_dim: usize,
        q_weight: Vec<f32>,
        q_bias: Vec<f32>,
        k_weight: Vec<f32>,
        k_bias: Vec<f32>,
        temperature: f32,
    ) -> Self {
        Self {
            width,
            head_dim,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            temperature,
            renderer: None,
            calibration_scope: None,
        }
    }

    /// Hidden width each feature row must have.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Projection dimension.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The exported (calibration-split) temperature.
    #[must_use]
    pub const fn temperature(&self) -> f32 {
        self.temperature
    }

    /// The artifact's `renderer` metadata: the prompt renderer its training
    /// rows used. `None` for a legacy artifact that does not declare one.
    #[must_use]
    pub fn renderer(&self) -> Option<&str> {
        self.renderer.as_deref()
    }

    /// The artifact's `calibration_scope` metadata: what its temperature was
    /// fit on. `None` when the artifact does not declare one, in which case
    /// its calibration provenance is unknown.
    #[must_use]
    pub fn calibration_scope(&self) -> Option<&str> {
        self.calibration_scope.as_deref()
    }

    /// Score `features` (option rows, then the decision row; see the
    /// [module docs](self)) and return one logit and probability per option.
    pub fn score(&self, features: &[f32]) -> crate::Result<JudgmentScores> {
        let width = self.width;
        if features.is_empty() || !features.len().is_multiple_of(width) {
            return Err(crate::Error::InvalidArgument(format!(
                "judgment features must be a whole number of {width}-wide rows, got {} values",
                features.len()
            )));
        }
        let rows = features.len() / width;
        if rows < MIN_OPTIONS + 1 {
            return Err(crate::Error::InvalidArgument(format!(
                "judgment features need at least {MIN_OPTIONS} option rows plus the decision \
                 row, got {rows} rows"
            )));
        }
        if !features.iter().all(|v| v.is_finite()) {
            return Err(crate::Error::InvalidArgument(
                "judgment features hold a non-finite value".into(),
            ));
        }
        let dot = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum()
        };
        let (options, decide) = features.split_at((rows - 1) * width);
        let q: Vec<f64> = self
            .q_weight
            .chunks_exact(width)
            .zip(&self.q_bias)
            .map(|(row, &bias)| dot(row, decide) + f64::from(bias))
            .collect();
        let scale = (self.head_dim as f64).sqrt() * f64::from(self.temperature);
        let logits: Vec<f64> = options
            .chunks_exact(width)
            .map(|option| {
                self.k_weight
                    .chunks_exact(width)
                    .zip(&self.k_bias)
                    .zip(&q)
                    .map(|((row, &bias), &q)| (dot(row, option) + f64::from(bias)) * q)
                    .sum::<f64>()
                    / scale
            })
            .collect();
        let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = logits.iter().map(|&z| (z - peak).exp()).collect();
        let total: f64 = exp.iter().sum();
        let probabilities = exp.into_iter().map(|e| e / total).collect();
        Ok(JudgmentScores {
            logits,
            probabilities,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::{JUDGMENT_HEAD_FORMAT, JudgmentHead};
    use serde_json::{Value, json};

    /// width 3, `head_dim` 2; asymmetric so transposition, row swaps and a
    /// dropped bias or temperature each change the result.
    const Q: [f32; 6] = [1.0, 0.0, 2.0, 0.0, -1.0, 1.0];
    const QB: [f32; 2] = [0.5, -1.0];
    const K: [f32; 6] = [2.0, 1.0, 0.0, -1.0, 0.0, 3.0];
    const KB: [f32; 2] = [1.0, 0.25];
    /// Options [1,2,0], [0,1,-1], [0,0,1], then the decision row [1,-1,2].
    const FEATURES: [f32; 12] = [1.0, 2.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 2.0];

    /// `(name, dtype, shape, values)`.
    type Tensor = (&'static str, &'static str, Vec<usize>, Vec<f32>);

    fn tensors() -> Vec<Tensor> {
        vec![
            ("q.weight", "F32", vec![2, 3], Q.to_vec()),
            ("q.bias", "F32", vec![2], QB.to_vec()),
            ("k.weight", "F32", vec![2, 3], K.to_vec()),
            ("k.bias", "F32", vec![2], KB.to_vec()),
            ("temperature", "F32", vec![1], vec![8.0]),
        ]
    }

    fn metadata() -> Value {
        json!({"format": JUDGMENT_HEAD_FORMAT, "width": "3", "head_dim": "2",
               "experimental": "true", "tool_sha256": "test"})
    }

    fn write(tensors: &[Tensor], metadata: &Value) -> tempfile::NamedTempFile {
        let mut header = serde_json::Map::new();
        let mut body = Vec::new();
        for (name, dtype, shape, values) in tensors {
            let start = body.len();
            body.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            header.insert(
                (*name).to_owned(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, body.len()]}),
            );
        }
        header.insert("__metadata__".into(), metadata.clone());
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend(encoded);
        bytes.extend(body);
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), bytes).unwrap();
        file
    }

    fn open(tensors: &[Tensor], metadata: &Value) -> crate::Result<JudgmentHead> {
        JudgmentHead::open(write(tensors, metadata).path())
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn scores_options_against_the_final_row_in_order() {
        let head = open(&tensors(), &metadata()).unwrap();
        assert_eq!(
            (head.width(), head.head_dim(), head.temperature()),
            (3, 2, 8.0)
        );
        let scores = head.score(&FEATURES).unwrap();
        // Hand-derived: q = [5.5, 2]; k = [5,-0.75], [2,-2.75], [1,3.25];
        // dots 26, 5.5, 12, each divided by sqrt(2) * 8.
        assert_close(
            &scores.logits,
            &[
                2.298_097_038_856_279_4,
                0.486_135_912_065_751_4,
                1.060_660_171_779_821_2,
            ],
        );
        assert_close(
            &scores.probabilities,
            &[
                0.688_013_238_623_646_3,
                0.112_375_613_608_667_78,
                0.199_611_147_767_686,
            ],
        );
    }

    /// Declared provenance round-trips verbatim; absent provenance stays
    /// unknown (a `mode` alone implies nothing); empty values are refused;
    /// and the CPU scorer still scores a head declaring another renderer.
    #[test]
    fn retains_declared_provenance_and_scores_any_renderer() {
        let legacy = open(&tensors(), &metadata()).unwrap();
        assert_eq!(
            (legacy.renderer(), legacy.calibration_scope()),
            (None, None)
        );

        let mut staged = metadata();
        staged["mode"] = json!("development");
        staged["renderer"] = json!("kev-hard-v1-judgment-render.v1");
        staged["calibration_scope"] = json!("calibration split of a synthetic set only");
        let head = open(&tensors(), &staged).unwrap();
        assert_eq!(head.renderer(), Some("kev-hard-v1-judgment-render.v1"));
        assert_eq!(
            head.calibration_scope(),
            Some("calibration split of a synthetic set only")
        );
        assert_eq!(
            head.score(&FEATURES).unwrap(),
            legacy.score(&FEATURES).unwrap()
        );

        let mut mode_only = metadata();
        mode_only["mode"] = json!("development");
        let head = open(&tensors(), &mode_only).unwrap();
        assert_eq!((head.renderer(), head.calibration_scope()), (None, None));

        for key in ["renderer", "calibration_scope"] {
            let mut empty = metadata();
            empty[key] = json!(" ");
            let error = open(&tensors(), &empty).unwrap_err().to_string();
            assert!(error.contains(key), "{error}");
        }
    }

    #[test]
    fn refuses_invalid_artifacts() {
        let with = |name: &str, change: &dyn Fn(&mut Tensor)| {
            let mut t = tensors();
            change(t.iter_mut().find(|t| t.0 == name).unwrap());
            open(&t, &metadata())
        };
        let meta = |key: &str, value: Value| {
            let mut m = metadata();
            m[key] = value;
            open(&tensors(), &m)
        };
        let bad = [
            meta("format", json!("local-ai.judgment-pointer-head.v1")),
            meta("experimental", json!("false")),
            meta("width", json!("4")),
            meta("head_dim", json!("0")),
            with("q.weight", &|t| t.1 = "F16"),
            with("k.weight", &|t| t.2 = vec![3, 2]),
            with("k.bias", &|t| t.3[1] = f32::NAN),
            with("q.weight", &|t| t.3[4] = f32::INFINITY),
            with("temperature", &|t| t.3[0] = 0.0),
            with("temperature", &|t| t.3[0] = -1.0),
            with("temperature", &|t| t.3[0] = f32::INFINITY),
            open(&tensors()[..4], &metadata()),
            open(
                &[tensors(), vec![("extra", "F32", vec![1], vec![0.0])]].concat(),
                &metadata(),
            ),
        ];
        for (index, result) in bad.into_iter().enumerate() {
            assert!(result.is_err(), "case {index} was accepted");
        }
        let file = write(&tensors(), &metadata());
        let bytes = std::fs::read(file.path()).unwrap();
        std::fs::write(file.path(), &bytes[..bytes.len() - 1]).unwrap();
        assert!(
            JudgmentHead::open(file.path()).is_err(),
            "truncated file was accepted"
        );
    }

    #[test]
    fn refuses_invalid_features() {
        let head = open(&tensors(), &metadata()).unwrap();
        assert!(head.score(&FEATURES[..11]).is_err());
        assert!(head.score(&FEATURES[3..]).is_ok());
        assert!(
            head.score(&FEATURES[6..]).is_err(),
            "one option is below the minimum"
        );
        assert!(head.score(&[]).is_err());
        let mut features = FEATURES;
        features[10] = f32::NAN;
        assert!(head.score(&features).is_err());
    }

    #[test]
    fn refuses_invalid_tensor_layouts() {
        for case in ["overflow", "overlap", "gap", "trailing"] {
            let file = write(&tensors(), &metadata());
            let bytes = std::fs::read(file.path()).unwrap();
            let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let mut header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
            let mut body = bytes[8 + header_len..].to_vec();
            match case {
                "overflow" => {
                    header["__metadata__"]["width"] = json!(usize::MAX.to_string());
                    header["q.weight"]["shape"] = json!([2, usize::MAX]);
                }
                "overlap" => header["k.weight"]["data_offsets"] = json!([0, 24]),
                "gap" => {
                    header["temperature"]["data_offsets"] = json!([68, 72]);
                    body.extend(1.0_f32.to_le_bytes());
                }
                _ => body.extend(1.0_f32.to_le_bytes()),
            }
            let encoded = serde_json::to_vec(&header).unwrap();
            let mut malformed = (encoded.len() as u64).to_le_bytes().to_vec();
            malformed.extend(encoded);
            malformed.extend(body);
            std::fs::write(file.path(), malformed).unwrap();
            assert!(JudgmentHead::open(file.path()).is_err(), "accepted {case}");
        }
    }
}
