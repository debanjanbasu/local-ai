//! EXPERIMENTAL native text decisions over an explicitly supplied
//! [`JudgmentHead`].
//!
//! A [`DecisionRequest`] (state text, question text, options) is rendered
//! exactly as `tools/judgment_prepare.py` renders a training row
//! ([`DECISION_RENDERER`]):
//!
//! ```text
//! State:\n<state>\n\nQuestion: <question>\nOptions:\nA) <option>\nB) <option>\n...Decision:
//! ```
//!
//! Each option endpoint is the end of its line *including* its `\n`; the
//! decision endpoint is the end of `Decision:`, the last byte. The text is
//! encoded once with the checkpoint tokenizer and every endpoint must resolve
//! to the exact token ending there (`encode_with_token_ends`), never rounded.
//! The hidden after each endpoint token is captured in blocks of
//! [`DECISION_CAPTURE_ROWS`] rows (the `mtp-capture features` default the
//! training features were captured with), rounded through FP16 as the
//! training features were stored, and scored by the head.
//!
//! # Renderer
//!
//! A head declaring a `renderer` other than [`DECISION_RENDERER`] is refused
//! before tokenization or GPU work (its features can still be scored with
//! [`JudgmentHead::score`]). A head that declares none, such as the
//! `models/decisions-v2` head, is rendered with [`DECISION_RENDERER`].
//!
//! # Scope
//!
//! Probabilities are the head's softmax output, nothing more: no confidence,
//! threshold or refusal is derived here, and nothing claims production
//! quality. Questions, state text or option counts unlike the head's
//! training rows are outside any calibration it has.

use super::{JudgmentHead, JudgmentScores};
use crate::bonsai_model::{BonsaiEngine, CancelToken};
use crate::bonsai_tokenizer::BonsaiTokenizer;

/// Renderer identity: byte-identical to `tools/judgment_prepare.py`
/// `RENDERER`.
pub const DECISION_RENDERER: &str = "kev-devtools-v1-judgment-render.v1";

/// Most options a decision may render: option letters are fixed `A`..`Z`.
pub const MAX_DECISION_OPTIONS: usize = 26;

/// Capture block width, matching the training features' `mtp-capture`
/// `--rows` default. Block boundaries change FP16 recurrent-state rounding,
/// so this is part of the feature contract.
pub const DECISION_CAPTURE_ROWS: usize = 60;

/// A typed option value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionValue {
    /// Rendered verbatim.
    Text(String),
    /// Rendered `Yes` (`true`) or `No` (`false`), the training labels.
    Bool(bool),
}

impl DecisionValue {
    /// The text this option renders as.
    #[must_use]
    pub fn rendered(&self) -> &str {
        match self {
            Self::Text(text) => text,
            Self::Bool(true) => "Yes",
            Self::Bool(false) => "No",
        }
    }
}

/// The question's answer space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionKind {
    /// Yes/No: rendered `A) Yes`, `B) No`; options are
    /// `[Bool(true), Bool(false)]`. The training rows used both orders.
    Predicate,
    /// One of the caller's options, rendered in the given order: from two
    /// to [`MAX_DECISION_OPTIONS`] distinct single-line values.
    Choice(Vec<DecisionValue>),
    /// Ordered levels, lowest first, rendered in that order: from one to
    /// [`MAX_DECISION_OPTIONS`] distinct single-line labels. The
    /// [`Decision::score`] is the probability-weighted mean level index. A
    /// single level is answered deterministically without GPU work.
    Score(Vec<String>),
}

/// A typed text question.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionRequest {
    /// State text, rendered verbatim after `State:\n` (training states were
    /// `Diff:\n<diff>`, optionally preceded by `Context before hunk:\n...`).
    pub state: String,
    /// Question text, rendered verbatim after `Question: `.
    pub question: String,
    /// The answer space.
    pub kind: DecisionKind,
}

/// A rendered request: the exact text and its exclusive UTF-8 byte
/// endpoints, one per option line and then the decision cue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedDecision {
    pub text: String,
    pub byte_ends: Vec<usize>,
    /// Option values in render (and native) order.
    pub options: Vec<DecisionValue>,
}

/// One option's outcome, in the caller's option order.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionProbability {
    pub option: DecisionValue,
    /// Head softmax probability; the options' probabilities sum to 1.
    pub probability: f64,
    /// Temperature-scaled head logit; `None` when nothing was captured
    /// (single-level score).
    pub logit: Option<f64>,
}

/// A decision's typed outcome. No confidence or refusal is implied.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    /// One entry per option, in the caller's (render) order.
    pub options: Vec<DecisionProbability>,
    /// Index of the most probable option; ties go to the lowest index.
    pub argmax: usize,
    /// `options[argmax].option`; a [`DecisionKind::Predicate`] yields
    /// [`DecisionValue::Bool`].
    pub value: DecisionValue,
    /// [`DecisionKind::Score`] only: `Σ index · probability`, in
    /// `0..=levels-1`.
    pub score: Option<f64>,
    /// Tokens of the rendered prompt.
    pub prompt_tokens: usize,
    /// Whether hidden states were captured and scored.
    pub captured: bool,
    /// The FP16-rounded hidden state of the `Decision:` row the head scored
    /// (`width` values), for an out-of-scope gate
    /// ([`super::CalibratedHead::judge`]); empty when nothing was captured.
    pub decide_row: Vec<f32>,
}

impl DecisionRequest {
    /// The request's options in render order, validated.
    fn options(&self) -> crate::Result<Vec<DecisionValue>> {
        let (options, minimum) = match &self.kind {
            DecisionKind::Predicate => (
                vec![DecisionValue::Bool(true), DecisionValue::Bool(false)],
                2,
            ),
            DecisionKind::Choice(options) => (options.clone(), 2),
            DecisionKind::Score(levels) => {
                (levels.iter().cloned().map(DecisionValue::Text).collect(), 1)
            }
        };
        if !(minimum..=MAX_DECISION_OPTIONS).contains(&options.len()) {
            return Err(invalid(format!(
                "needs {minimum}..={MAX_DECISION_OPTIONS} options, got {}",
                options.len()
            )));
        }
        for (index, option) in options.iter().enumerate() {
            let text = option.rendered();
            if text.trim().is_empty() {
                return Err(invalid(format!("option {index} is empty")));
            }
            if text.contains(['\n', '\r']) {
                return Err(invalid(format!("option {index} spans more than one line")));
            }
            if options[..index]
                .iter()
                .any(|other| other.rendered() == text)
            {
                return Err(invalid(format!("option {index} ({text:?}) is a duplicate")));
            }
        }
        Ok(options)
    }

    /// Validate and render exactly as `tools/judgment_prepare.py` does.
    pub fn render(&self) -> crate::Result<RenderedDecision> {
        if self.state.trim().is_empty() {
            return Err(invalid("state is empty"));
        }
        if self.question.trim().is_empty() {
            return Err(invalid("question is empty"));
        }
        let options = self.options()?;
        let mut text = String::new();
        for piece in [
            "State:\n",
            &self.state,
            "\n\nQuestion: ",
            &self.question,
            "\nOptions:\n",
        ] {
            text.push_str(piece);
        }
        let mut byte_ends = Vec::with_capacity(options.len() + 1);
        for (letter, option) in ('A'..='Z').zip(&options) {
            text.push(letter);
            text.push_str(") ");
            text.push_str(option.rendered());
            text.push('\n');
            byte_ends.push(text.len());
        }
        text.push_str("Decision:");
        byte_ends.push(text.len());
        Ok(RenderedDecision {
            text,
            byte_ends,
            options,
        })
    }
}

fn invalid(message: impl std::fmt::Display) -> crate::Error {
    crate::Error::InvalidArgument(format!("decision: {message}"))
}

/// A validated, tokenized decision, ready to run on the engine.
pub struct PreparedDecision {
    options: Vec<DecisionValue>,
    score: bool,
    tokens: Vec<u32>,
    /// Token index of each option endpoint, then the decision endpoint.
    positions: Vec<usize>,
}

impl PreparedDecision {
    /// Render, tokenize and validate `request` for `head`, on the CPU.
    ///
    /// Rejects a head whose width is not the model's or whose declared
    /// renderer is not [`DECISION_RENDERER`], any input whose text tokenizes
    /// to a special token, and any endpoint that is not an exact token
    /// boundary.
    pub(crate) fn new(
        tokenizer: &BonsaiTokenizer,
        head: &JudgmentHead,
        request: &DecisionRequest,
    ) -> crate::Result<Self> {
        Self::with_width(tokenizer, head, crate::bonsai::WIDTH, request)
    }

    fn with_width(
        tokenizer: &BonsaiTokenizer,
        head: &JudgmentHead,
        width: usize,
        request: &DecisionRequest,
    ) -> crate::Result<Self> {
        if head.width() != width {
            return Err(invalid(format!(
                "judgment head width {} does not match the model width {width}",
                head.width()
            )));
        }
        if let Some(declared) = head.renderer()
            && declared != DECISION_RENDERER
        {
            return Err(invalid(format!(
                "judgment head renderer {declared:?} is not supported natively; score its \
                 features with JudgmentHead::score instead"
            )));
        }
        let (rendered, tokens, positions) = tokenize(tokenizer, request)?;
        Ok(Self {
            options: rendered.options,
            score: matches!(request.kind, DecisionKind::Score(_)),
            tokens,
            positions,
        })
    }

    /// Whether the decision needs hidden capture (and so the engine).
    pub(crate) const fn needs_capture(&self) -> bool {
        self.options.len() >= 2
    }

    /// The prompt tokens.
    #[cfg(test)]
    pub(crate) fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Answer a decision that needs no capture.
    pub(crate) fn ready(&self) -> crate::Result<Decision> {
        if self.needs_capture() {
            return Err(crate::Error::Generation(
                "decision needs hidden capture".into(),
            ));
        }
        self.assemble(None)
    }

    /// Capture on `engine` (which must have no active generation) and score.
    /// `Ok(None)` when `cancel` tripped first.
    pub(crate) fn run(
        &self,
        engine: &mut BonsaiEngine,
        head: &JudgmentHead,
        cancel: &CancelToken,
    ) -> crate::Result<Option<Decision>> {
        if !self.needs_capture() {
            return self.ready().map(Some);
        }
        if cancel.is_cancelled() {
            return Ok(None);
        }
        let Some(features) = capture(engine, &self.tokens, &self.positions, cancel)? else {
            return Ok(None);
        };
        let scores = head.score(&features)?;
        let mut decision = self.assemble(Some(&scores))?;
        decision.decide_row = features[features.len() - head.width()..].to_vec();
        Ok(Some(decision))
    }

    /// The typed outcome from the head's `scores`, or the deterministic one
    /// of a single option.
    fn assemble(&self, scores: Option<&JudgmentScores>) -> crate::Result<Decision> {
        let count = self.options.len();
        let (probabilities, logits) = match scores {
            Some(scores) => (
                scores.probabilities.clone(),
                scores.logits.iter().copied().map(Some).collect(),
            ),
            None if count == 1 => (vec![1.0], vec![None]),
            None => {
                return Err(crate::Error::Generation("decision has no scores".into()));
            }
        };
        if probabilities.len() != count || logits.len() != count {
            return Err(crate::Error::Generation(format!(
                "judgment head scored {} options, expected {count}",
                probabilities.len()
            )));
        }
        let argmax = probabilities
            .iter()
            .enumerate()
            .fold(
                0,
                |best, (index, &p)| if p > probabilities[best] { index } else { best },
            );
        let score = self.score.then(|| {
            probabilities
                .iter()
                .enumerate()
                .map(|(index, &p)| index as f64 * p)
                .sum()
        });
        Ok(Decision {
            options: self
                .options
                .iter()
                .zip(probabilities)
                .zip(logits)
                .map(|((option, probability), logit)| DecisionProbability {
                    option: option.clone(),
                    probability,
                    logit,
                })
                .collect(),
            argmax,
            value: self.options[argmax].clone(),
            score,
            prompt_tokens: self.tokens.len(),
            captured: scores.is_some(),
            decide_row: Vec::new(),
        })
    }
}

/// Render `request`, encode it once and resolve every endpoint to the exact
/// token ending there; text spelling a special token is refused.
fn tokenize(
    tokenizer: &BonsaiTokenizer,
    request: &DecisionRequest,
) -> crate::Result<(RenderedDecision, Vec<u32>, Vec<usize>)> {
    let rendered = request.render()?;
    let (tokens, positions) =
        tokenizer.encode_with_token_ends(&rendered.text, &rendered.byte_ends)?;
    // Every added token is special, so skipping specials changes the
    // decode exactly when the text spelled one.
    if tokenizer.decode(&tokens, true)? != tokenizer.decode(&tokens, false)? {
        return Err(invalid("text contains a special-token string"));
    }
    Ok((rendered, tokens, positions))
}

/// Capture the hidden after each of `positions` in
/// [`DECISION_CAPTURE_ROWS`]-row blocks and round it through FP16, as the
/// training features were stored. The one capture path of both scored
/// decisions and [`DecisionCapture`]. `Ok(None)` when `cancel` tripped.
fn capture(
    engine: &mut BonsaiEngine,
    tokens: &[u32],
    positions: &[usize],
    cancel: &CancelToken,
) -> crate::Result<Option<Vec<f32>>> {
    let Some(mut features) =
        engine.capture_hidden(tokens, positions, DECISION_CAPTURE_ROWS, cancel)?
    else {
        return Ok(None);
    };
    crate::bonsai_native::capture::round_features_f16(&mut features)?;
    Ok(Some(features))
}

/// Round `features` through FP16 and score them, as a captured decision is.
#[cfg(test)]
fn score_features(head: &JudgmentHead, mut features: Vec<f32>) -> crate::Result<JudgmentScores> {
    crate::bonsai_native::capture::round_features_f16(&mut features)?;
    head.score(&features)
}

/// A decision's head-independent capture: what a scored decision would read,
/// before any head. See [`crate::Engine::capture_decision`].
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionCapture {
    /// The native rendering: text, byte endpoints and options.
    pub rendered: RenderedDecision,
    /// Prompt token ids.
    pub tokens: Vec<u32>,
    /// Token index of each option endpoint, then the decision endpoint.
    pub positions: Vec<usize>,
    /// `positions.len()` rows of [`crate::bonsai::WIDTH`] FP16-rounded
    /// values, option rows first and the decision row last: exactly the
    /// features a scored decision passes to [`JudgmentHead::score`].
    pub features: Vec<f32>,
}

impl DecisionCapture {
    /// Render and tokenize `request` on the CPU, with every check a capture
    /// makes before GPU work; [`features`](Self::features) stay empty. A
    /// request with fewer than two options has no head features and is
    /// refused. Use it to validate a batch before loading the model.
    pub fn prepare(tokenizer: &BonsaiTokenizer, request: &DecisionRequest) -> crate::Result<Self> {
        let (rendered, tokens, positions) = tokenize(tokenizer, request)?;
        if rendered.options.len() < 2 {
            return Err(invalid("capture needs at least two options"));
        }
        Ok(Self {
            rendered,
            tokens,
            positions,
            features: Vec::new(),
        })
    }

    /// Capture on `engine`, which must have no active generation.
    pub(crate) fn run(mut self, engine: &mut BonsaiEngine) -> crate::Result<Self> {
        self.features = capture(engine, &self.tokens, &self.positions, &CancelToken::new())?
            .ok_or_else(|| crate::Error::Generation("decision capture was cancelled".into()))?;
        Ok(self)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::{
        DECISION_RENDERER, DecisionCapture, DecisionKind, DecisionRequest, DecisionValue,
        JudgmentHead, MAX_DECISION_OPTIONS, PreparedDecision, score_features,
    };
    use crate::bonsai_tokenizer::BonsaiTokenizer;

    fn request(kind: DecisionKind) -> DecisionRequest {
        DecisionRequest {
            state: "Diff:\n-a\n+b".into(),
            question: "Does this need a comment?".into(),
            kind,
        }
    }

    fn texts(values: &[&str]) -> Vec<DecisionValue> {
        values
            .iter()
            .map(|&value| DecisionValue::Text(value.into()))
            .collect()
    }

    /// width 3, `head_dim` 2, the scorer's own hand-checked fixture.
    fn head() -> JudgmentHead {
        JudgmentHead::for_tests(
            3,
            2,
            vec![1.0, 0.0, 2.0, 0.0, -1.0, 1.0],
            vec![0.5, -1.0],
            vec![2.0, 1.0, 0.0, -1.0, 0.0, 3.0],
            vec![1.0, 0.25],
            8.0,
        )
    }

    fn prepare(request: &DecisionRequest) -> crate::Result<PreparedDecision> {
        PreparedDecision::with_width(&BonsaiTokenizer::tiny_for_tests(), &head(), 3, request)
    }

    /// The exact bytes and endpoints of `tools/judgment_prepare.py::render`
    /// for ("Yes", "No") options: each option end includes its newline.
    #[test]
    fn renders_exactly_like_the_training_tool() -> crate::Result<()> {
        let rendered = request(DecisionKind::Predicate).render()?;
        let expected = "State:\nDiff:\n-a\n+b\n\nQuestion: Does this need a comment?\n\
                        Options:\nA) Yes\nB) No\nDecision:";
        assert_eq!(rendered.text, expected);
        let a = expected.find("A) Yes\n").expect("A") + "A) Yes\n".len();
        let b = expected.find("B) No\n").expect("B") + "B) No\n".len();
        assert_eq!(rendered.byte_ends, vec![a, b, expected.len()]);
        assert_eq!(
            rendered.options,
            vec![DecisionValue::Bool(true), DecisionValue::Bool(false)]
        );
        assert_eq!(DECISION_RENDERER, "kev-devtools-v1-judgment-render.v1");

        // UTF-8 endpoints are byte offsets, and letters run A..Z.
        let mut options = texts(&["é"]);
        options.extend((1..MAX_DECISION_OPTIONS).map(|i| DecisionValue::Text(format!("o{i}"))));
        let rendered = request(DecisionKind::Choice(options)).render()?;
        assert!(rendered.text.contains("\nA) é\nB) o1\n"));
        assert!(rendered.text.ends_with("\nZ) o25\nDecision:"));
        let a = rendered.text.find("A) é\n").expect("A") + "A) é\n".len();
        assert_eq!(rendered.byte_ends[0], a);
        assert_eq!(rendered.byte_ends.len(), MAX_DECISION_OPTIONS + 1);
        Ok(())
    }

    #[test]
    fn invalid_requests_are_refused_before_any_work() {
        let many = (0..=MAX_DECISION_OPTIONS)
            .map(|i| DecisionValue::Text(format!("o{i}")))
            .collect();
        let cases = [
            DecisionRequest {
                state: " \n".into(),
                ..request(DecisionKind::Predicate)
            },
            DecisionRequest {
                question: String::new(),
                ..request(DecisionKind::Predicate)
            },
            request(DecisionKind::Choice(texts(&["only"]))),
            request(DecisionKind::Choice(many)),
            request(DecisionKind::Choice(texts(&["a", " "]))),
            request(DecisionKind::Choice(texts(&["a", "b\nc"]))),
            request(DecisionKind::Choice(texts(&["a", "b\r"]))),
            request(DecisionKind::Choice(vec![
                DecisionValue::Text("Yes".into()),
                DecisionValue::Bool(true),
            ])),
            request(DecisionKind::Score(Vec::new())),
            request(DecisionKind::Score(vec!["low".into(), "low".into()])),
        ];
        for (index, case) in cases.iter().enumerate() {
            assert!(case.render().is_err(), "case {index} rendered");
            assert!(prepare(case).is_err(), "case {index} prepared");
        }
    }

    /// Endpoints resolve to the tokens ending each option line and the cue,
    /// through the tokenizer's exact boundary check.
    #[test]
    fn endpoints_resolve_to_exact_tokens() -> crate::Result<()> {
        let tokenizer = BonsaiTokenizer::tiny_for_tests();
        let request = request(DecisionKind::Choice(texts(&["abc", "é"])));
        let prepared = prepare(&request)?;
        let rendered = request.render()?;
        assert_eq!(prepared.tokens(), tokenizer.encode(&rendered.text)?);
        let count = prepared.tokens.len();
        assert_eq!(prepared.positions.len(), 3);
        assert_eq!(prepared.positions[2], count - 1);
        for (&position, &end) in prepared.positions.iter().zip(&rendered.byte_ends) {
            let prefix = tokenizer.decode(&prepared.tokens[..=position], false)?;
            assert_eq!(prefix.len(), end);
            assert!(prefix.ends_with('\n') || prefix.ends_with("Decision:"));
        }
        Ok(())
    }

    #[test]
    fn special_token_strings_and_foreign_heads_are_refused() {
        for field in 0..3 {
            let mut case = request(DecisionKind::Choice(texts(&["a", "b"])));
            match field {
                0 => case.state.push_str("<|x|>"),
                1 => case.question.push_str("<|x|>"),
                _ => case.kind = DecisionKind::Choice(texts(&["a", "b<|x|>"])),
            }
            let error = prepare(&case).err().map(|e| e.to_string());
            assert!(
                error
                    .as_deref()
                    .is_some_and(|e| e.contains("special-token")),
                "field {field}: {error:?}"
            );
        }
        let ok = request(DecisionKind::Predicate);
        assert!(prepare(&ok).is_ok());
        assert!(
            PreparedDecision::with_width(&BonsaiTokenizer::tiny_for_tests(), &head(), 4, &ok)
                .is_err()
        );
        assert!(PreparedDecision::new(&BonsaiTokenizer::tiny_for_tests(), &head(), &ok).is_err());
    }

    /// Features are FP16-rounded before scoring; probabilities keep the
    /// caller's order, the argmax and predicate value follow them, and a
    /// score is the index-weighted mean.
    #[test]
    fn scores_map_to_typed_outcomes() -> crate::Result<()> {
        // Options [1,2,0], [0,1,-1], [0,0,1], decision [1,-1,2]: the
        // scorer's fixture, exact in FP16.
        let features = vec![1.0, 2.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 2.0];
        let head = head();
        let exact = head.score(&features)?;
        let scores = score_features(&head, features.clone())?;
        assert_eq!(scores, exact);
        // A value FP16 cannot hold exactly is rounded first.
        let mut nudged = features;
        nudged[0] = 1.000_1;
        assert_eq!(score_features(&head, nudged)?, exact);
        assert!(score_features(&head, vec![1.0e6; 12]).is_err());

        let levels = vec!["low".to_owned(), "mid".to_owned(), "high".to_owned()];
        let prepared = prepare(&request(DecisionKind::Score(levels)))?;
        let decision = prepared.assemble(Some(&scores))?;
        assert_eq!(decision.argmax, 0);
        assert_eq!(decision.value, DecisionValue::Text("low".into()));
        let p: Vec<f64> = decision.options.iter().map(|o| o.probability).collect();
        assert_eq!(p, exact.probabilities);
        let mean = 2.0_f64.mul_add(p[2], p[1]);
        assert!((decision.score.expect("score") - mean).abs() < 1e-12);
        assert!(decision.captured);
        assert_eq!(decision.options[2].logit, Some(exact.logits[2]));

        let two = crate::judgment::JudgmentScores {
            logits: vec![-1.0, 1.0],
            probabilities: vec![0.25, 0.75],
        };
        let predicate = prepare(&request(DecisionKind::Predicate))?.assemble(Some(&two))?;
        assert_eq!((predicate.argmax, predicate.score), (1, None));
        assert_eq!(predicate.value, DecisionValue::Bool(false));
        let tie = crate::judgment::JudgmentScores {
            logits: vec![0.0, 0.0],
            probabilities: vec![0.5, 0.5],
        };
        let choice = prepare(&request(DecisionKind::Choice(texts(&["x", "y"]))))?;
        assert_eq!(
            choice.assemble(Some(&tie))?.argmax,
            0,
            "ties keep the first"
        );
        assert!(choice.assemble(Some(&scores)).is_err(), "count mismatch");
        assert!(choice.ready().is_err(), "two options need capture");
        Ok(())
    }

    /// A head declaring no renderer or [`DECISION_RENDERER`] is served; any
    /// other declared renderer is refused before any work.
    #[test]
    fn only_the_training_renderer_is_served() {
        let tokenizer = BonsaiTokenizer::tiny_for_tests();
        let with = |renderer: Option<&str>| JudgmentHead {
            renderer: renderer.map(str::to_owned),
            ..head()
        };
        let predicate = request(DecisionKind::Predicate);
        let prepare = |head: &JudgmentHead| {
            PreparedDecision::with_width(&tokenizer, head, 3, &predicate).map(|_| ())
        };
        assert!(prepare(&with(None)).is_ok());
        assert!(prepare(&with(Some(DECISION_RENDERER))).is_ok());
        for renderer in ["kev-hard-v1-judgment-render.v1", "unknown-renderer"] {
            let error = prepare(&with(Some(renderer))).err().map(|e| e.to_string());
            assert!(
                error.as_deref().is_some_and(|e| e.contains(renderer)),
                "{renderer}: {error:?}"
            );
        }
    }

    /// A head-independent capture prepares exactly the tokens and positions
    /// a scored decision captures, needs no head, and refuses what a scored
    /// decision would refuse.
    #[test]
    fn captures_prepare_the_scored_decisions_tokens() -> crate::Result<()> {
        let tokenizer = BonsaiTokenizer::tiny_for_tests();
        for kind in [
            DecisionKind::Predicate,
            DecisionKind::Choice(vec![DecisionValue::Bool(false), DecisionValue::Bool(true)]),
            DecisionKind::Score(vec!["low".into(), "high".into()]),
        ] {
            let request = request(kind);
            let capture = DecisionCapture::prepare(&tokenizer, &request)?;
            let prepared = prepare(&request)?;
            assert_eq!(capture.tokens, prepared.tokens);
            assert_eq!(capture.positions, prepared.positions);
            assert_eq!(capture.rendered, request.render()?);
            assert_eq!(capture.features, Vec::<f32>::new());
        }
        let single = request(DecisionKind::Score(vec!["only".into()]));
        assert!(DecisionCapture::prepare(&tokenizer, &single).is_err());
        let mut special = request(DecisionKind::Predicate);
        special.state.push_str("<|x|>");
        assert!(DecisionCapture::prepare(&tokenizer, &special).is_err());
        Ok(())
    }

    #[test]
    fn a_single_level_score_is_deterministic_without_capture() -> crate::Result<()> {
        let prepared = prepare(&request(DecisionKind::Score(vec!["only".into()])))?;
        assert!(!prepared.needs_capture());
        let decision = prepared.ready()?;
        assert_eq!(decision.argmax, 0);
        assert_eq!(decision.score, Some(0.0));
        assert_eq!(decision.options[0].probability, 1.0);
        assert_eq!(decision.options[0].logit, None);
        assert!(!decision.captured);
        assert!(decision.prompt_tokens > 0);
        Ok(())
    }
}
