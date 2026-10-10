//! Shared generation parameters and measurements for the Bonsai engine.

use std::time::Duration;

pub const DEFAULT_MAX_OUTPUT_TOKENS: usize = 8_192;
pub const DEFAULT_TEMPERATURE: f32 = 1.0;
pub const DEFAULT_TOP_P: f32 = 0.95;
pub const DEFAULT_TOP_K: usize = 20;
pub const DEFAULT_MIN_P: f32 = 0.0;

/// Capacity of the per-request event channel fed by the engine worker.
///
/// Capacity 1 bought no useful backpressure, because the engine's own decode
/// pace is the rate limiter; it only cost cross-client isolation. A real buffer
/// stops one slow consumer from stalling the engine worker, which serves every
/// running and queued request. Events are single-token text
/// pieces, or one `Vec<u32>` for the whole generation, so 64 stays well under
/// one megabyte even in the worst case.
pub const EVENT_BUFFER: usize = 64;

/// One prefill-chunk boundary: the engine is about to submit this many tokens.
///
/// Prefill is the only phase of a generation that emits nothing per token. The
/// chunk is one forward pass over up to `PREFILL_CHUNK` tokens, so a prompt
/// longer than one chunk is silent for the whole of that pass — about 35 s per
/// 128 tokens on this M2, or roughly 14 minutes for a 3000-token prompt. A
/// consumer that watches only for generated text cannot tell a working engine
/// from an abandoned socket during it, so each boundary is reported instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillProgress {
    /// Prompt tokens committed before this chunk was submitted. Tokens restored
    /// from the prompt cache are counted, because they are already in the K/V
    /// cache this chunk continues from.
    pub tokens: usize,
    /// Chunks this request has entered, counting the one in flight, so the first
    /// is 1 rather than 0.
    pub chunks: usize,
}

#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    pub seed: u64,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            top_k: DEFAULT_TOP_K,
            min_p: DEFAULT_MIN_P,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            seed: 0x5eed_5eed_cafe_f00d,
        }
    }
}

/// Speculative-decoding counters accumulated over one generation request.
#[derive(Debug, Clone, Default)]
pub struct MtpStats {
    pub rounds: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub verified_tokens: usize,
    pub drafting: Duration,
    pub verification: Duration,
    pub commit: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct NgramStats {
    pub rounds: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub lookup: Duration,
}

/// Measurements from actual token IDs, not a re-tokenized output string.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub reused_prompt_tokens: usize,
    pub generated_tokens: usize,
    /// Includes EOS and any verified samples not emitted after cancellation.
    /// `generated_tokens` counts only emitted token IDs, excluding EOS.
    pub sampled_tokens: usize,
    pub prefill: Duration,
    pub sampling: Duration,
    /// GPU-busy time from Metal command-buffer timestamps over the whole
    /// request (native backend; zero elsewhere). `elapsed - gpu` is CPU
    /// encoding, sampling and submission gaps.
    pub gpu: Duration,
    pub mtp: MtpStats,
    pub ngram: NgramStats,
    /// Tokens decoded in a batched step beside other requests.
    pub batched_tokens: usize,
    /// Time from request start to the first sampled token, even without visible text.
    pub first_token: Option<Duration>,
    pub elapsed: Duration,
    /// For a request with a [`ResponseFormat`](crate::ResponseFormat) other
    /// than text: whether its answer is a complete document the format
    /// accepts, which holds exactly when generation stopped at end-of-sequence
    /// after the constraint began. `Some(false)` for a token limit,
    /// cancellation, or end-of-sequence during reasoning (no answer at all).
    /// `None` for unconstrained requests.
    pub response_format_complete: Option<bool>,
}

pub(crate) fn validate_generation_params(params: &GenerateParams) -> crate::Result<()> {
    if !params.temperature.is_finite() || params.temperature < 0.0 {
        return Err(crate::Error::InvalidArgument(
            "temperature must be finite and non-negative".into(),
        ));
    }
    if !(0.0..=1.0).contains(&params.top_p) || !(0.0..=1.0).contains(&params.min_p) {
        return Err(crate::Error::InvalidArgument(
            "top-p and min-p must be between zero and one".into(),
        ));
    }
    if !params.repetition_penalty.is_finite() || params.repetition_penalty <= 0.0 {
        return Err(crate::Error::InvalidArgument(
            "repetition penalty must be positive".into(),
        ));
    }
    if !params.presence_penalty.is_finite() {
        return Err(crate::Error::InvalidArgument(
            "presence penalty must be finite".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_out_of_range_generation_params() {
        let defaults = GenerateParams::default();
        assert!(validate_generation_params(&defaults).is_ok());
        let cases = [
            GenerateParams {
                temperature: -0.5,
                ..defaults.clone()
            },
            GenerateParams {
                temperature: f32::NAN,
                ..defaults.clone()
            },
            GenerateParams {
                top_p: 1.5,
                ..defaults.clone()
            },
            GenerateParams {
                min_p: -0.1,
                ..defaults.clone()
            },
            GenerateParams {
                repetition_penalty: 0.0,
                ..defaults.clone()
            },
            GenerateParams {
                presence_penalty: f32::INFINITY,
                ..defaults
            },
        ];
        for params in cases {
            assert!(
                matches!(
                    validate_generation_params(&params),
                    Err(crate::Error::InvalidArgument(_))
                ),
                "{params:?}"
            );
        }
    }
}
