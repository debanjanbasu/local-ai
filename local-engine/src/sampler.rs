//! CPU-side token sampling from logits.
//!
//! Supports greedy, temperature, top-k, top-p, and repetition penalty.
//! Applies `final_logit_softcapping` before sampling per Gemma 4 spec.

/// Parameters controlling token sampling behavior.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for sampling. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Top-k filtering. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) filtering. 1.0 = disabled.
    pub top_p: f32,
    /// Repetition penalty. 1.0 = disabled.
    pub repetition_penalty: f32,
    /// Logit softcap value. 0.0 = disabled, typically 30.0.
    pub logit_softcap: f32,
    /// EOS token IDs for stop detection.
    pub eos_tokens: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.0,
            logit_softcap: 30.0,
            eos_tokens: vec![1, 106],
        }
    }
}

/// Result of a sampling operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingResult {
    /// The sampled token ID.
    pub token_id: u32,
    /// Whether this token is an end-of-sequence token.
    pub is_eos: bool,
}

#[derive(Debug)]
struct ActiveSupport {
    indices: Vec<usize>,
    dropped: Vec<usize>,
}

/// Sample a token from logits.
///
/// # Errors
///
/// Returns `Sampling` error if logits are empty or sampling fails.
pub fn sample(
    logits: &mut [f32],
    params: &SamplingParams,
    context_tokens: &[u32],
    rng: &mut impl FnMut() -> f32,
) -> crate::Result<SamplingResult> {
    if logits.is_empty() {
        return Err(crate::Error::Sampling("empty logits".into()));
    }

    // 1. Logit softcapping.
    //
    // `cap * tanh(x / cap)` is strictly monotonic, so it never changes the
    // argmax. The only path where it affects the *result* is when repetition
    // penalty is active (the penalty scales the capped magnitude). For greedy
    // decode (temperature == 0 or top_k == 1) with no repetition penalty we
    // therefore skip the full-vocab tanh pass entirely — at 262k entries that
    // is the dominant CPU cost of sampling.
    let greedy = params.temperature <= f32::EPSILON || params.top_k == 1;
    let rep_active = (params.repetition_penalty - 1.0).abs() > f32::EPSILON;
    if params.logit_softcap > 0.0 && (!greedy || rep_active) {
        apply_logit_softcap(logits, params.logit_softcap);
    }

    // 2. Repetition penalty
    if (params.repetition_penalty - 1.0).abs() > f32::EPSILON {
        apply_repetition_penalty(logits, context_tokens, params.repetition_penalty);
    }

    // 3. Greedy
    if params.temperature <= f32::EPSILON {
        let token_id = argmax(logits);
        return Ok(SamplingResult {
            token_id,
            is_eos: params.eos_tokens.contains(&token_id),
        });
    }

    // 4. Deterministic top-k shortcut
    // Positive temperature scaling preserves ordering, so top_k=1 can return
    // the argmax before softmax/top-p/rng work.
    if params.top_k == 1 {
        let token_id = argmax(logits);
        return Ok(SamplingResult {
            token_id,
            is_eos: params.eos_tokens.contains(&token_id),
        });
    }

    // 5. Temperature scaling
    let inv_temp = 1.0 / params.temperature;
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }

    // 6. Top-k filtering
    let mut active_support = if params.top_k > 0 {
        apply_top_k(logits, params.top_k)
    } else {
        None
    };

    // 7. Softmax
    if let Some(support) = active_support.as_ref() {
        softmax_active_in_place(logits, support);
    } else {
        softmax_in_place(logits);
    }

    // 8. Top-p filtering + renormalize
    if params.top_p < 1.0 {
        if let Some(support) = active_support.as_mut() {
            apply_top_p_active(logits, support, params.top_p);
        } else {
            apply_top_p(logits, params.top_p);
        }
    }

    // 9. Sample from distribution
    let r = rng();
    let token_id = active_support.as_ref().map_or_else(
        || sample_from_probs(logits, r),
        |support| sample_from_active_support(logits, support, r),
    );

    // 10. EOS check
    Ok(SamplingResult {
        token_id,
        is_eos: params.eos_tokens.contains(&token_id),
    })
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0_u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn apply_logit_softcap(logits: &mut [f32], cap: f32) {
    for logit in logits.iter_mut() {
        *logit = cap * (*logit / cap).tanh();
    }
}

fn apply_repetition_penalty(logits: &mut [f32], context_tokens: &[u32], penalty: f32) {
    for &token in context_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn apply_top_k(logits: &mut [f32], k: usize) -> Option<ActiveSupport> {
    if k == 0 || k >= logits.len() {
        return None;
    }

    let mut indices: Vec<usize> = (0..logits.len()).collect();
    let (active, cutoff, dropped_tail) =
        indices.select_nth_unstable_by(k, |&a, &b| logits[b].total_cmp(&logits[a]));

    let mut dropped = Vec::with_capacity(dropped_tail.len() + 1);
    dropped.push(*cutoff);
    dropped.extend_from_slice(dropped_tail);

    for &idx in &dropped {
        logits[idx] = f32::NEG_INFINITY;
    }

    Some(ActiveSupport {
        indices: active.to_vec(),
        dropped,
    })
}

fn softmax_in_place(logits: &mut [f32]) {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0_f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max_val).exp();
        sum += *logit;
    }

    if sum > 0.0 {
        for logit in logits.iter_mut() {
            *logit /= sum;
        }
    }
}

fn softmax_active_in_place(logits: &mut [f32], support: &ActiveSupport) {
    let max_val = support
        .indices
        .iter()
        .map(|&idx| logits[idx])
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0_f32;
    for &idx in &support.indices {
        logits[idx] = (logits[idx] - max_val).exp();
        sum += logits[idx];
    }

    if sum > 0.0 {
        for &idx in &support.indices {
            logits[idx] /= sum;
        }
    }

    for &idx in &support.dropped {
        logits[idx] = 0.0;
    }
}

fn apply_top_p(probs: &mut [f32], p: f32) {
    let mut indices: Vec<usize> = (0..probs.len()).filter(|&idx| probs[idx] > 0.0).collect();
    if indices.is_empty() {
        return;
    }

    indices.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

    let mut cumulative = 0.0_f32;
    let mut cutoff_idx = indices.len();
    for (rank, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= p {
            cutoff_idx = rank + 1;
            break;
        }
    }

    for &idx in &indices[cutoff_idx..] {
        probs[idx] = 0.0;
    }

    let sum: f32 = indices[..cutoff_idx].iter().map(|&idx| probs[idx]).sum();
    if sum > 0.0 {
        for &idx in &indices[..cutoff_idx] {
            probs[idx] /= sum;
        }
    }
}

fn apply_top_p_active(probs: &mut [f32], support: &mut ActiveSupport, p: f32) {
    if support.indices.is_empty() {
        return;
    }

    support
        .indices
        .sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

    let mut cumulative = 0.0_f32;
    let mut cutoff_idx = support.indices.len();
    for (rank, &idx) in support.indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= p {
            cutoff_idx = rank + 1;
            break;
        }
    }

    let dropped = support.indices.split_off(cutoff_idx);
    for &idx in &dropped {
        probs[idx] = 0.0;
    }

    let sum: f32 = support.indices.iter().map(|&idx| probs[idx]).sum();
    if sum > 0.0 {
        for &idx in &support.indices {
            probs[idx] /= sum;
        }
    }

    support.dropped.extend(dropped);
}

fn sample_from_probs(probs: &[f32], r: f32) -> u32 {
    let mut cumulative = 0.0_f32;
    let mut token_id = 0_u32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative > r {
            return i as u32;
        }
        if i == probs.len() - 1 {
            token_id = i as u32;
        }
    }
    token_id
}

fn sample_from_active_support(probs: &[f32], support: &ActiveSupport, r: f32) -> u32 {
    let mut cumulative = 0.0_f32;
    let mut token_id = 0_u32;
    for &idx in &support.indices {
        cumulative += probs[idx];
        token_id = idx as u32;
        if cumulative > r {
            return token_id;
        }
    }
    token_id
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::approx_constant,
    clippy::too_many_lines,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::string_slice,
    clippy::manual_let_else,
    clippy::single_match_else,
    clippy::must_use_candidate,
    clippy::missing_panics_doc,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::cast_lossless,
    clippy::redundant_clone,
    clippy::useless_vec,
    clippy::similar_names,
    clippy::unnecessary_trailing_comma
)]
mod tests {
    use super::*;

    fn deterministic_rng(value: f32) -> impl FnMut() -> f32 {
        move || value
    }

    #[test]
    fn test_greedy_always_argmax() -> crate::Result<()> {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            eos_tokens: vec![],
        };
        let result = sample(&mut logits, &params, &[], &mut deterministic_rng(0.0))?;
        assert_eq!(result.token_id, 1);
        assert!(!result.is_eos);
        Ok(())
    }

    #[test]
    fn test_top_k_1_always_max() -> crate::Result<()> {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            eos_tokens: vec![],
        };
        // With top_k=1, only the max token survives, so any rng value picks it
        let result = sample(&mut logits, &params, &[], &mut deterministic_rng(0.5))?;
        assert_eq!(result.token_id, 1);
        Ok(())
    }

    #[test]
    fn test_top_k_1_skips_rng_and_top_p_sampling() -> crate::Result<()> {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 1,
            top_p: 0.1,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            eos_tokens: vec![],
        };

        let result = sample(&mut logits, &params, &[], &mut || -> f32 {
            panic!("top_k=1 should short-circuit before calling rng");
        })?;

        assert_eq!(result.token_id, 1);
        Ok(())
    }

    #[test]
    fn test_top_k_support_path_zeroes_filtered_logits() -> crate::Result<()> {
        let mut logits = vec![4.0, 3.0, 2.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            eos_tokens: vec![],
        };

        let result = sample(&mut logits, &params, &[], &mut deterministic_rng(0.99))?;

        assert!(
            result.token_id <= 1,
            "top-k support should exclude lower-ranked tokens"
        );
        assert!(logits[2].abs() < 1e-6);
        assert!(logits[3].abs() < 1e-6);
        assert!((logits[0] + logits[1] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_top_p_filters_low_probability() -> crate::Result<()> {
        // Token 0 dominates with huge logit
        let mut logits = vec![100.0, 1.0, 1.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            eos_tokens: vec![],
        };
        let result = sample(&mut logits, &params, &[], &mut deterministic_rng(0.0))?;
        // Token 0 dominates so it should be selected
        assert_eq!(result.token_id, 0);
        Ok(())
    }

    #[test]
    fn test_apply_top_p_exact_threshold_keeps_minimal_nucleus() {
        let mut probs = vec![0.5, 0.25, 0.25];
        apply_top_p(&mut probs, 0.75);

        assert!((probs[0] - (2.0 / 3.0)).abs() < 1e-6);
        assert!((probs[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!(probs[2].abs() < 1e-6);
    }

    #[test]
    fn test_sample_top_p_exact_threshold_never_returns_dropped_token() -> crate::Result<()> {
        let mut logits = vec![0.0, -0.6931472, -0.6931472];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.75,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            eos_tokens: vec![],
        };

        let result = sample(&mut logits, &params, &[], &mut deterministic_rng(0.99))?;
        assert_eq!(result.token_id, 1);
        Ok(())
    }

    #[test]
    fn test_repetition_penalty_reduces_probability() -> crate::Result<()> {
        // Token 0 is slightly ahead, but penalized by repetition
        let mut logits = vec![5.0, 4.9, 0.0, 0.0];
        let params = SamplingParams {
            temperature: 0.0, // greedy to make deterministic
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 2.0,
            logit_softcap: 0.0,
            eos_tokens: vec![],
        };
        // Token 0 appears in context → gets penalized: 5.0/2.0 = 2.5
        // Token 1 stays at 4.9 → wins
        let result = sample(&mut logits, &params, &[0], &mut deterministic_rng(0.0))?;
        assert_eq!(result.token_id, 1);
        Ok(())
    }

    #[test]
    fn test_temperature_affects_distribution() -> crate::Result<()> {
        // With very low temperature, should behave like greedy
        let mut logits = vec![1.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.01,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            eos_tokens: vec![],
        };
        let result = sample(&mut logits, &params, &[], &mut deterministic_rng(0.0))?;
        // Very low temperature makes distribution peaky → token 1 wins
        assert_eq!(result.token_id, 1);
        Ok(())
    }

    #[test]
    fn test_logit_softcapping() -> crate::Result<()> {
        let mut logits = vec![0.0, 1000.0, -1000.0];
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            logit_softcap: 30.0,
            eos_tokens: vec![],
        };
        // After softcap: 1000/30 is huge → tanh ≈ 1.0 → 30 * 1.0 ≈ 30
        // -1000/30 → tanh ≈ -1.0 → 30 * -1.0 ≈ -30
        let result = sample(&mut logits, &params, &[], &mut deterministic_rng(0.0))?;
        assert_eq!(result.token_id, 1);

        // Verify capping actually happened by checking values directly
        let mut test_logits = vec![0.0_f32, 1000.0, -1000.0];
        apply_logit_softcap(&mut test_logits, 30.0);
        assert!((test_logits[0]).abs() < 0.001);
        assert!((test_logits[1] - 30.0).abs() < 0.01);
        assert!((test_logits[2] + 30.0).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_eos_detection() -> crate::Result<()> {
        let mut logits = vec![0.0, 0.0, 100.0]; // Token 2 wins
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            logit_softcap: 0.0,
            eos_tokens: vec![1, 2, 50], // Token 2 is EOS
        };
        let result = sample(&mut logits, &params, &[], &mut deterministic_rng(0.0))?;
        assert_eq!(result.token_id, 2);
        assert!(result.is_eos);

        // Token 0 is not EOS
        let mut logits2 = vec![100.0, 0.0, 0.0];
        let result2 = sample(&mut logits2, &params, &[], &mut deterministic_rng(0.0))?;
        assert_eq!(result2.token_id, 0);
        assert!(!result2.is_eos);
        Ok(())
    }
}
