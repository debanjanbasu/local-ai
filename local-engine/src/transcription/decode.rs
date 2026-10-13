//! Whisper decoding and long-form transcription, ported from the original
//! `openai-whisper` (20250625) `decoding.py` and `transcribe.py`: greedy or
//! temperature sampling with the reference's logit filters (blank and
//! non-speech suppression, timestamp rules), the temperature fallback on
//! compression ratio and average log-probability, the no-speech skip, and
//! the 30-second sliding window that seeks to the last complete timestamp.
//! Beam search, word timestamps and clip timestamps are not ported.
//!
//! Arithmetic on times and probabilities is written as the reference writes
//! it (no fused multiply-add, exact float comparisons), so segment times
//! match it bit for bit.
#![allow(clippy::suboptimal_flops, clippy::float_cmp)]

use std::io::Write as _;

use super::mel::{LogMel, WINDOW_FRAMES};
use super::tokens::WhisperTokenizer;
use super::{Pace, Segment, Task, Temperature};
use crate::Error;

/// Text context; prompts keep at most half of it.
const N_TEXT_CTX: usize = 448;
const SAMPLE_LEN: usize = N_TEXT_CTX / 2;
const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;
const LOGPROB_THRESHOLD: f64 = -1.0;
const NO_SPEECH_THRESHOLD: f64 = 0.6;
/// Mel frames per encoder output frame, and seconds per timestamp step.
const INPUT_STRIDE: usize = 2;
const TIME_PRECISION: f64 = (INPUT_STRIDE * 160) as f64 / 16_000.0;
const FALLBACK_TEMPERATURES: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

/// The model operations decoding needs (the Metal model, or a script in
/// tests).
pub trait Forward {
    /// Encode one 30-second window (`[3000][n_mels]`, time-major).
    fn encode(&mut self, window: &[f32], pace: &mut Pace<'_>) -> crate::Result<()>;
    /// Reset the decoder, run `tokens` from position 0 against the last
    /// encoded window, and return the logits at `rows`.
    fn prefill(
        &mut self,
        tokens: &[u32],
        rows: &[usize],
        pace: &mut Pace<'_>,
    ) -> crate::Result<Vec<Vec<f32>>>;
    /// Run one more token at the next position; its logits.
    fn step(&mut self, token: u32, pace: &mut Pace<'_>) -> crate::Result<Vec<f32>>;
}

/// Counters for usage reporting.
#[derive(Default, Clone, Copy, Debug)]
pub struct Work {
    pub windows: usize,
    pub prompt_tokens: usize,
    pub sampled_tokens: usize,
}

pub struct DecodingResult {
    /// Sampled tokens, end-of-text excluded.
    pub tokens: Vec<u32>,
    /// Log-probability of each sampled token (filtered distribution).
    pub logprobs: Vec<f32>,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f32,
    pub compression_ratio: f64,
}

/// xorshift64* for temperature sampling.
pub struct Rng(u64);

impl Rng {
    pub const fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let value = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (value >> 11) as f64 / (1_u64 << 53) as f64
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max = f64::from(max);
    let sum: f64 = logits.iter().map(|v| (f64::from(*v) - max).exp()).sum();
    let log_sum = max + sum.ln();
    logits.iter().map(|v| f64::from(*v) - log_sum).collect()
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// `len(utf8) / len(zlib.compress(utf8))` (zlib's default level), the
/// reference's repetition measure.
pub fn compression_ratio(text: &str) -> f64 {
    let bytes = text.as_bytes();
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    let compressed = encoder
        .write_all(bytes)
        .and_then(|()| encoder.finish())
        .map_or(1, |out| out.len().max(1));
    bytes.len() as f64 / compressed as f64
}

/// The reference's logit filters, in order: `SuppressBlank`,
/// `SuppressTokens`, `ApplyTimestampRules`.
pub fn apply_filters(
    logits: &mut [f32],
    tokens: &[u32],
    sample_begin: usize,
    tokenizer: &WhisperTokenizer,
    suppress: &[u32],
) {
    let sp = &tokenizer.special;
    let ts = sp.timestamp_begin as usize;
    let eot = sp.eot as usize;
    let at_begin = tokens.len() == sample_begin;
    if at_begin {
        for id in &sp.begin_suppress_tokens {
            logits[*id as usize] = f32::NEG_INFINITY;
        }
    }
    for id in suppress {
        logits[*id as usize] = f32::NEG_INFINITY;
    }
    // Timestamp rules.
    logits[sp.no_timestamps as usize] = f32::NEG_INFINITY;
    let sampled = &tokens[sample_begin..];
    let is_ts = |t: u32| t >= sp.timestamp_begin;
    let last_was_timestamp = sampled.last().is_some_and(|t| is_ts(*t));
    let penultimate_was_timestamp = sampled.len() < 2 || is_ts(sampled[sampled.len() - 2]);
    if last_was_timestamp {
        if penultimate_was_timestamp {
            logits[ts..].fill(f32::NEG_INFINITY);
        } else {
            logits[..eot].fill(f32::NEG_INFINITY);
        }
    }
    if let Some(last) = sampled.iter().rev().find(|t| is_ts(**t)) {
        let last = *last as usize;
        let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
            last
        } else {
            last + 1
        };
        let end = timestamp_last.min(logits.len());
        if end > ts {
            logits[ts..end].fill(f32::NEG_INFINITY);
        }
    }
    if at_begin {
        logits[..ts].fill(f32::NEG_INFINITY);
        let last_allowed = ts + sp.max_initial_timestamp_index as usize;
        if last_allowed + 1 < logits.len() {
            logits[last_allowed + 1..].fill(f32::NEG_INFINITY);
        }
    }
    let logprobs = log_softmax(logits);
    let timestamp_logprob = {
        let tail = &logprobs[ts..];
        let max = tail.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            max + tail.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
        }
    };
    let max_text = logprobs[..ts]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if timestamp_logprob > max_text {
        logits[..ts].fill(f32::NEG_INFINITY);
    }
}

/// One `DecodingTask.run` over the last encoded window: greedy at
/// temperature 0, sampling otherwise; timestamps on.
#[allow(clippy::too_many_arguments)]
pub fn decode_window(
    forward: &mut dyn Forward,
    tokenizer: &WhisperTokenizer,
    task: Task,
    language: u32,
    prompt: &[u32],
    temperature: f32,
    rng: &mut Rng,
    pace: &mut Pace<'_>,
    work: &mut Work,
) -> crate::Result<DecodingResult> {
    let sp = &tokenizer.special;
    let mut initial = Vec::with_capacity(prompt.len() + 4);
    if !prompt.is_empty() {
        initial.push(sp.sot_prev);
        initial.extend_from_slice(&prompt[prompt.len().saturating_sub(N_TEXT_CTX / 2 - 1)..]);
    }
    let sot_index = initial.len();
    initial.extend([
        sp.sot,
        language,
        match task {
            Task::Transcribe => sp.transcribe,
            Task::Translate => sp.translate,
        },
    ]);
    let sample_begin = initial.len();
    work.prompt_tokens += initial.len();
    let suppress = sp.suppressed();
    let mut rows = forward.prefill(&initial, &[sot_index, sample_begin - 1], pace)?;
    let mut logits = rows
        .pop()
        .ok_or_else(|| Error::Generation("no logits".into()))?;
    let at_sot = rows
        .pop()
        .ok_or_else(|| Error::Generation("no logits".into()))?;
    let no_speech_prob = log_softmax(&at_sot)[sp.no_speech as usize].exp();
    let mut tokens = initial;
    let mut sum_logprob = 0.0_f64;
    let mut logprobs = Vec::new();
    for i in 0..SAMPLE_LEN {
        if i > 0 {
            let last = *tokens
                .last()
                .ok_or_else(|| Error::Generation("empty".into()))?;
            logits = forward.step(last, pace)?;
        }
        apply_filters(&mut logits, &tokens, sample_begin, tokenizer, &suppress);
        let next = if temperature == 0.0 {
            argmax(&logits)
        } else {
            let scaled: Vec<f32> = logits.iter().map(|v| v / temperature).collect();
            let probs = log_softmax(&scaled);
            let mut target = rng.next_f64();
            let mut chosen = argmax(&logits);
            for (id, lp) in probs.iter().enumerate() {
                let p = lp.exp();
                if p > 0.0 {
                    if target < p {
                        chosen = id;
                        break;
                    }
                    target -= p;
                }
            }
            chosen
        };
        let logprob = log_softmax(&logits)[next];
        sum_logprob += logprob;
        logprobs.push(logprob as f32);
        tokens.push(next as u32);
        work.sampled_tokens += 1;
        if next as u32 == sp.eot || tokens.len() > N_TEXT_CTX {
            break;
        }
    }
    let mut sampled = tokens[sample_begin..].to_vec();
    if let Some(end) = sampled.iter().position(|t| *t == sp.eot) {
        sampled.truncate(end);
    }
    logprobs.truncate(sampled.len());
    let text = tokenizer.decode(&sampled);
    let avg_logprob = sum_logprob / (sampled.len() + 1) as f64;
    Ok(DecodingResult {
        compression_ratio: compression_ratio(text.trim()),
        tokens: sampled,
        logprobs,
        avg_logprob,
        no_speech_prob,
        temperature,
    })
}

/// Language probabilities over Whisper's languages for the encoded window
/// (`detect_language`: the logits after `<|startoftranscript|>` restricted to
/// language tokens), most likely first.
pub fn language_probabilities(
    forward: &mut dyn Forward,
    tokenizer: &WhisperTokenizer,
    pace: &mut Pace<'_>,
) -> crate::Result<Vec<(String, f64)>> {
    let sp = &tokenizer.special;
    let logits = forward
        .prefill(&[sp.sot], &[0], pace)?
        .pop()
        .ok_or_else(|| Error::Generation("no logits".into()))?;
    let ids: Vec<(String, u32)> = sp
        .languages
        .iter()
        .map(|(code, id)| (code.clone(), *id))
        .collect();
    let picked: Vec<f32> = ids.iter().map(|(_, id)| logits[*id as usize]).collect();
    let lp = log_softmax(&picked);
    let mut out: Vec<(String, f64)> = ids
        .into_iter()
        .zip(lp)
        .map(|((code, _), lp)| (code, lp.exp()))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(out)
}

fn decode_with_fallback(
    forward: &mut dyn Forward,
    tokenizer: &WhisperTokenizer,
    options: &WindowOptions,
    prompt: &[u32],
    rng: &mut Rng,
    pace: &mut Pace<'_>,
    work: &mut Work,
) -> crate::Result<DecodingResult> {
    let temperatures: &[f32] = match &options.temperature {
        Temperature::Fallback => &FALLBACK_TEMPERATURES,
        Temperature::Fixed(t) => std::slice::from_ref(t),
    };
    let mut last = None;
    for &t in temperatures {
        let result = decode_window(
            forward,
            tokenizer,
            options.task,
            options.language,
            prompt,
            t,
            rng,
            pace,
            work,
        )?;
        let silence =
            result.no_speech_prob > NO_SPEECH_THRESHOLD && result.avg_logprob < LOGPROB_THRESHOLD;
        let needs_fallback = !silence
            && (result.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                || result.avg_logprob < LOGPROB_THRESHOLD);
        last = Some(result);
        if !needs_fallback {
            break;
        }
    }
    last.ok_or_else(|| Error::Generation("no temperature to decode with".into()))
}

pub struct WindowOptions {
    pub task: Task,
    pub language: u32,
    pub temperature: Temperature,
    pub condition_on_previous_text: bool,
}

#[allow(clippy::too_many_arguments)]
fn new_segment(
    tokenizer: &WhisperTokenizer,
    id: usize,
    seek: usize,
    start: f64,
    end: f64,
    tokens: &[u32],
    logprobs: &[f32],
    result: &DecodingResult,
) -> Segment {
    let text_tokens: Vec<u32> = tokens
        .iter()
        .copied()
        .filter(|t| *t < tokenizer.special.eot)
        .collect();
    Segment {
        id,
        seek,
        start,
        end,
        text: tokenizer.decode(&text_tokens),
        tokens: tokens.to_vec(),
        logprobs: logprobs.to_vec(),
        temperature: result.temperature,
        avg_logprob: result.avg_logprob,
        compression_ratio: result.compression_ratio,
        no_speech_prob: result.no_speech_prob,
    }
}

/// The reference's `transcribe()` with `clip_timestamps="0"`, no word
/// timestamps and no hallucination-silence threshold. Returns the segments
/// and the prompt-free text tokens; `on_segments` sees each window's
/// segments as soon as they are final. `first_window_encoded`: the forward
/// already holds the encoding of the full first window (`mel.window(0,
/// 3000)`, as language detection leaves it).
#[allow(clippy::too_many_arguments)]
pub fn transcribe(
    forward: &mut dyn Forward,
    tokenizer: &WhisperTokenizer,
    mel: &LogMel,
    first_window_encoded: bool,
    options: &WindowOptions,
    initial_prompt: &[u32],
    rng: &mut Rng,
    pace: &mut Pace<'_>,
    work: &mut Work,
    on_segments: &mut dyn FnMut(&[Segment]) -> crate::Result<()>,
) -> crate::Result<(Vec<Segment>, Vec<u32>)> {
    let sp = &tokenizer.special;
    let content_frames = mel.frames.saturating_sub(WINDOW_FRAMES);
    let mut seek = 0_usize;
    let mut all_tokens: Vec<u32> = initial_prompt.to_vec();
    let mut all_segments: Vec<Segment> = Vec::new();
    let mut prompt_reset_since = 0_usize;
    while seek < content_frames {
        let time_offset = (seek * 160) as f64 / 16_000.0;
        let segment_size = WINDOW_FRAMES.min(content_frames - seek);
        let segment_duration = (segment_size * 160) as f64 / 16_000.0;
        // Language detection already encoded the first full window.
        if !(seek == 0 && segment_size == WINDOW_FRAMES && first_window_encoded) {
            forward.encode(&mel.window(seek, segment_size), pace)?;
            work.windows += 1;
        }
        let prompt = all_tokens[prompt_reset_since..].to_vec();
        let result = decode_with_fallback(forward, tokenizer, options, &prompt, rng, pace, work)?;
        let should_skip =
            result.no_speech_prob > NO_SPEECH_THRESHOLD && result.avg_logprob <= LOGPROB_THRESHOLD;
        if should_skip {
            seek += segment_size;
            continue;
        }
        let tokens = &result.tokens;
        let logprobs = &result.logprobs;
        let is_ts: Vec<bool> = tokens.iter().map(|t| *t >= sp.timestamp_begin).collect();
        let single_timestamp_ending =
            is_ts.len() >= 2 && !is_ts[is_ts.len() - 2] && is_ts[is_ts.len() - 1];
        let consecutive: Vec<usize> = (0..is_ts.len().saturating_sub(1))
            .filter(|i| is_ts[*i] && is_ts[i + 1])
            .map(|i| i + 1)
            .collect();
        let mut current = Vec::new();
        let first_id = all_segments.len();
        if consecutive.is_empty() {
            let duration = match tokens.iter().rev().find(|t| **t >= sp.timestamp_begin) {
                Some(last) if *last != sp.timestamp_begin => {
                    f64::from(*last - sp.timestamp_begin) * TIME_PRECISION
                }
                _ => segment_duration,
            };
            current.push(new_segment(
                tokenizer,
                first_id,
                seek,
                time_offset,
                time_offset + duration,
                tokens,
                logprobs,
                &result,
            ));
            seek += segment_size;
        } else {
            let mut cuts = consecutive;
            if single_timestamp_ending {
                cuts.push(tokens.len());
            }
            let mut last_slice = 0;
            for &slice in &cuts {
                let piece = &tokens[last_slice..slice];
                let start = f64::from(piece[0] - sp.timestamp_begin);
                let end = f64::from(piece[piece.len() - 1] - sp.timestamp_begin);
                current.push(new_segment(
                    tokenizer,
                    first_id + current.len(),
                    seek,
                    time_offset + start * TIME_PRECISION,
                    time_offset + end * TIME_PRECISION,
                    piece,
                    &logprobs[last_slice..slice],
                    &result,
                ));
                last_slice = slice;
            }
            if single_timestamp_ending {
                seek += segment_size;
            } else {
                let last_timestamp = (tokens[last_slice - 1] - sp.timestamp_begin) as usize;
                seek += last_timestamp * INPUT_STRIDE;
            }
        }
        for segment in &mut current {
            if segment.start == segment.end || segment.text.trim().is_empty() {
                segment.text.clear();
                segment.tokens.clear();
                segment.logprobs.clear();
            }
        }
        all_tokens.extend(current.iter().flat_map(|s| s.tokens.iter().copied()));
        if !options.condition_on_previous_text || result.temperature > 0.5 {
            prompt_reset_since = all_tokens.len();
        }
        on_segments(&current)?;
        all_segments.extend(current);
    }
    let text_tokens = all_tokens[initial_prompt.len()..].to_vec();
    Ok((all_segments, text_tokens))
}

#[cfg(test)]
mod tests;
