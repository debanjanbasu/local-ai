//! Text moderation heads on frozen `EmbeddingGemma` 2 embeddings.
//!
//! A moderation artifact (`models/moderation/moderation-heads.json`, see
//! `docs/MODERATION.md`) holds one calibrated logistic-regression head per
//! category of the `OpenAI` moderation taxonomy that passed its pre-registered
//! evaluation, and marks every other category unsupported with a reason. It
//! is opt-in by installation and validated on load: the file's SHA-256 must
//! equal [`MODERATION_HEADS_SHA256`], it must name the exact embedding payload
//! it was trained on (checked against the loaded encoder), and every head must
//! be well formed.
//!
//! Scoring: the text is prefixed with [`PROMPT`] (the model card's
//! classification prompt) and tokenized; inputs longer than
//! [`WINDOW_TOKENS`] tokens are split into windows of that many tokens, each
//! carrying the same prompt (see [`window_ids`]). For a head with weight `w`,
//! bias `b`, Platt scale `a`, offset `c` and threshold `t`, a window with
//! normalized 768-d embedding `x` scores `p = sigmoid(a (w·x + b) + c)`; the
//! input's score is the maximum over its windows and it is flagged when
//! `p >= t`. The heads were fitted on the first window of each training text.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use crate::Error;
use crate::embedding::{EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_ID, EmbeddingModel};

/// Folder of the artifact inside the models directory.
pub const MODERATION_DIR: &str = "moderation";
/// The artifact file name.
pub const MODERATION_ARTIFACT: &str = "moderation-heads.json";
/// SHA-256 of the one artifact this build accepts.
pub const MODERATION_HEADS_SHA256: &str =
    "b74727bf58afce5adad11dc3aa903388ad550fffc9c38b099d14995fc760bdc4";
/// Id this build reports as the moderation `model`.
pub const MODERATION_MODEL_ID: &str = "embeddinggemma-2-moderation-1";
/// The model card's classification prompt, prepended to every input.
pub const PROMPT: &str = "task: classification | query: ";
/// Tokens per window, `<bos>`, prompt and `<eos>` included.
pub const WINDOW_TOKENS: usize = 512;
const FORMAT: &str = "local-ai-moderation-heads";

/// The `OpenAI` moderation categories, in the pinned spec's order.
pub const CATEGORIES: [&str; 13] = [
    "hate",
    "hate/threatening",
    "harassment",
    "harassment/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadFile {
    weight: Vec<f32>,
    bias: f64,
    platt_scale: f64,
    platt_offset: f64,
    threshold: f64,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CategoryFile {
    Head(HeadFile),
    Unsupported { unsupported: String },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactFile {
    format: String,
    version: u32,
    model: String,
    embedding_model: String,
    embedding_payload_sha256: String,
    prompt: String,
    window_tokens: usize,
    dimensions: usize,
    categories: std::collections::BTreeMap<String, CategoryFile>,
    #[allow(dead_code)]
    provenance: serde_json::Value,
    #[allow(dead_code)]
    evaluation: serde_json::Value,
}

/// One calibrated head.
#[derive(Debug, Clone)]
pub struct Head {
    weight: Vec<f32>,
    bias: f64,
    scale: f64,
    offset: f64,
    threshold: f64,
}

impl Head {
    /// Calibrated probability for one normalized embedding.
    #[must_use]
    pub fn probability(&self, embedding: &[f32]) -> f64 {
        let logit = self
            .weight
            .iter()
            .zip(embedding)
            .map(|(w, x)| f64::from(*w) * f64::from(*x))
            .sum::<f64>()
            + self.bias;
        let z = self.scale.mul_add(logit, self.offset);
        1.0 / (1.0 + (-z).exp())
    }

    /// The `flagged` threshold chosen on the calibration split.
    #[must_use]
    pub const fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// Per-category result of one input: `None` for unsupported categories.
#[derive(Debug, Clone, PartialEq)]
pub struct Scores {
    pub scores: [Option<f64>; 13],
    pub flags: [Option<bool>; 13],
}

impl Scores {
    /// Whether any supported category is flagged.
    #[must_use]
    pub fn flagged(&self) -> bool {
        self.flags.iter().any(|f| *f == Some(true))
    }
}

/// The validated heads.
#[derive(Debug)]
pub struct ModerationHeads {
    heads: Vec<Option<Head>>,
    reasons: Vec<Option<String>>,
    sha256: String,
}

/// The artifact path inside `models_dir`, if installed.
#[must_use]
pub fn installed_heads(models_dir: &Path) -> Option<PathBuf> {
    let path = models_dir.join(MODERATION_DIR).join(MODERATION_ARTIFACT);
    path.is_file().then_some(path)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidArgument(format!("moderation artifact: {}", message.into()))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

impl ModerationHeads {
    /// Load and validate the installed artifact against the pinned SHA-256
    /// and the loaded encoder's payload SHA-256.
    ///
    /// # Errors
    ///
    /// I/O errors, a hash mismatch, or a malformed artifact.
    pub fn open(path: &Path, embedding_payload_sha256: &str) -> crate::Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|error| invalid(format!("cannot read {}: {error}", path.display())))?;
        Self::from_bytes(&bytes, MODERATION_HEADS_SHA256, embedding_payload_sha256)
    }

    /// Validate an artifact's bytes against an expected file SHA-256 and the
    /// encoder payload SHA-256.
    ///
    /// # Errors
    ///
    /// A hash mismatch or a malformed artifact.
    pub fn from_bytes(
        bytes: &[u8],
        expected_sha256: &str,
        embedding_payload_sha256: &str,
    ) -> crate::Result<Self> {
        let sha256 = hex(Sha256::digest(bytes).as_slice());
        if sha256 != expected_sha256 {
            return Err(invalid(format!(
                "sha256 {sha256} is not the pinned {expected_sha256}"
            )));
        }
        let file: ArtifactFile =
            serde_json::from_slice(bytes).map_err(|error| invalid(error.to_string()))?;
        if file.format != FORMAT || file.version != 1 {
            return Err(invalid(format!(
                "format {:?} version {} is not {FORMAT:?} version 1",
                file.format, file.version
            )));
        }
        if file.model != MODERATION_MODEL_ID || file.embedding_model != EMBEDDING_MODEL_ID {
            return Err(invalid(format!(
                "model {:?} on {:?} is not {MODERATION_MODEL_ID:?} on {EMBEDDING_MODEL_ID:?}",
                file.model, file.embedding_model
            )));
        }
        if file.embedding_payload_sha256 != embedding_payload_sha256 {
            return Err(invalid(format!(
                "trained on embedding payload {}, but the loaded encoder is {embedding_payload_sha256}",
                file.embedding_payload_sha256
            )));
        }
        if file.prompt != PROMPT
            || file.window_tokens != WINDOW_TOKENS
            || file.dimensions != EMBEDDING_DIMENSIONS
        {
            return Err(invalid(
                "prompt, window or dimensions differ from this build's",
            ));
        }
        if file.categories.len() != CATEGORIES.len()
            || CATEGORIES.iter().any(|c| !file.categories.contains_key(*c))
        {
            return Err(invalid("it must list exactly the 13 spec categories"));
        }
        let mut heads = Vec::with_capacity(CATEGORIES.len());
        let mut reasons = Vec::with_capacity(CATEGORIES.len());
        for name in CATEGORIES {
            match &file.categories[name] {
                CategoryFile::Head(head) => {
                    let finite = head.weight.iter().all(|w| w.is_finite())
                        && head.bias.is_finite()
                        && head.platt_offset.is_finite();
                    if head.weight.len() != EMBEDDING_DIMENSIONS
                        || !finite
                        || !(head.platt_scale.is_finite() && head.platt_scale > 0.0)
                        || !(head.threshold > 0.0 && head.threshold < 1.0)
                    {
                        return Err(invalid(format!("head {name:?} is malformed")));
                    }
                    heads.push(Some(Head {
                        weight: head.weight.clone(),
                        bias: head.bias,
                        scale: head.platt_scale,
                        offset: head.platt_offset,
                        threshold: head.threshold,
                    }));
                    reasons.push(None);
                }
                CategoryFile::Unsupported { unsupported } => {
                    if unsupported.trim().is_empty() {
                        return Err(invalid(format!("{name:?} needs an unsupported reason")));
                    }
                    heads.push(None);
                    reasons.push(Some(unsupported.clone()));
                }
            }
        }
        if heads.iter().all(Option::is_none) {
            return Err(invalid("no category is supported"));
        }
        Ok(Self {
            heads,
            reasons,
            sha256,
        })
    }

    /// SHA-256 of the artifact file.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Head of a category, `None` when unsupported or unknown.
    #[must_use]
    pub fn head(&self, category: &str) -> Option<&Head> {
        let index = CATEGORIES.iter().position(|c| *c == category)?;
        self.heads[index].as_ref()
    }

    /// Why a category is unsupported, `None` when it is supported.
    #[must_use]
    pub fn unsupported_reason(&self, category: &str) -> Option<&str> {
        let index = CATEGORIES.iter().position(|c| *c == category)?;
        self.reasons[index].as_deref()
    }

    /// Names of the supported categories, in spec order.
    pub fn supported(&self) -> impl Iterator<Item = &'static str> + '_ {
        CATEGORIES
            .iter()
            .zip(&self.heads)
            .filter(|(_, head)| head.is_some())
            .map(|(name, _)| *name)
    }

    /// Score one input from the embeddings of its windows (maximum over
    /// windows per category).
    #[must_use]
    pub fn score(&self, windows: &[Vec<f32>]) -> Scores {
        let mut scores = [None; 13];
        let mut flags = [None; 13];
        for (index, head) in self.heads.iter().enumerate() {
            if let Some(head) = head {
                let p = windows
                    .iter()
                    .map(|x| head.probability(x))
                    .fold(0.0_f64, f64::max);
                scores[index] = Some(p);
                flags[index] = Some(p >= head.threshold);
            }
        }
        Scores { scores, flags }
    }
}

/// Split the token ids of `PROMPT + text` into windows of at most
/// [`WINDOW_TOKENS`] ids.
///
/// `full` is the tokenization of the prompted text (`<bos>` … `<eos>`),
/// `prompt` that of [`PROMPT`] alone. Short inputs are one window, exactly
/// `full`. Longer ones keep the ids shared with the prompt's tokenization
/// (`<bos>` and the prompt) at the front of every window and the final
/// `<eos>` at the end, and split the rest into consecutive chunks.
#[must_use]
pub fn window_ids(full: &[u32], prompt: &[u32]) -> Vec<Vec<u32>> {
    if full.len() <= WINDOW_TOKENS || full.len() < 2 {
        return vec![full.to_vec()];
    }
    let shared = full[..full.len() - 1]
        .iter()
        .zip(&prompt[..prompt.len().saturating_sub(1)])
        .take_while(|(a, b)| a == b)
        .count();
    let head = &full[..shared];
    let eos = full[full.len() - 1];
    let body = &full[shared..full.len() - 1];
    let chunk = WINDOW_TOKENS - shared - 1;
    body.chunks(chunk)
        .map(|part| {
            let mut ids = Vec::with_capacity(WINDOW_TOKENS);
            ids.extend_from_slice(head);
            ids.extend_from_slice(part);
            ids.push(eos);
            ids
        })
        .collect()
}

/// Token-id windows of one input text, prompt applied.
///
/// # Errors
///
/// Tokenizer failures.
pub fn windows(model: &EmbeddingModel, text: &str) -> crate::Result<Vec<Vec<u32>>> {
    let full = model.tokenize(&format!("{PROMPT}{text}"))?;
    if full.len() <= WINDOW_TOKENS {
        return Ok(vec![full]);
    }
    let prompt = model.tokenize(PROMPT)?;
    Ok(window_ids(&full, &prompt))
}

#[cfg(test)]
mod tests;
