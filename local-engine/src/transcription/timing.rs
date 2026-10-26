//! Word timestamps, ported from openai-whisper 20250625 `timing.py`: the
//! cross-attention of the model's alignment heads over a forced decode of
//! the window's text, normalized per frame, median-filtered (width 7) and
//! averaged, aligned to time by dynamic time warping; words split as the
//! reference tokenizer splits them, punctuation merged into neighbours, and
//! the reference's duration heuristics applied.
//!
//! Arithmetic follows the reference's order (no fused multiply-add, exact
//! comparisons, F32 DTW costs) so that alignments match it.

use super::decode::Forward;
use super::tokens::WhisperTokenizer;
use super::{Pace, Segment, Word};

const MEDFILT_WIDTH: usize = 7;
const TOKENS_PER_SECOND: f64 = 50.0;
const PREPEND_PUNCTUATIONS: &str = "\"'“¿([{-";
const APPEND_PUNCTUATIONS: &str = "\"'.。,，!！?？:：”)]}、";
const ASCII_PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
/// Languages split into words at Unicode boundaries rather than spaces.
const NO_SPACE_LANGUAGES: [&str; 6] = ["zh", "ja", "th", "lo", "my", "yue"];

/// Python's `round(x, 2)`: correctly rounded, ties to even.
fn round2(x: f64) -> f64 {
    format!("{x:.2}").parse().unwrap_or(x)
}

#[derive(Clone, Debug)]
struct WordTiming {
    word: String,
    tokens: Vec<u32>,
    start: f64,
    end: f64,
    probability: f64,
}

/// Median of each `width` window along the rows of `x` (`rows` x `cols`),
/// reflect-padded, as `whisper.timing.median_filter`.
fn median_filter(x: &mut [f64], rows: usize, cols: usize, width: usize) {
    let pad = width / 2;
    if cols <= pad {
        return;
    }
    let mut padded = vec![0.0; cols + 2 * pad];
    let mut window = vec![0.0; width];
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        for (i, slot) in padded.iter_mut().enumerate() {
            let k = if i < pad {
                pad - i
            } else if i - pad >= cols {
                2 * (cols - 1) - (i - pad)
            } else {
                i - pad
            };
            *slot = row[k];
        }
        for (c, out) in row.iter_mut().enumerate() {
            window.copy_from_slice(&padded[c..c + width]);
            window.sort_by(f64::total_cmp);
            *out = window[pad];
        }
    }
}

/// `whisper.timing.dtw_cpu` + `backtrace` on an `n` x `m` cost matrix:
/// (text indices, time indices).
#[expect(clippy::many_single_char_names)]
fn dtw(x: &[f64], n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let w = m + 1;
    let mut cost = vec![f32::INFINITY; (n + 1) * w];
    let mut trace = vec![-1_i8; (n + 1) * w];
    cost[0] = 0.0;
    for j in 1..=m {
        for i in 1..=n {
            let c0 = cost[(i - 1) * w + j - 1];
            let c1 = cost[(i - 1) * w + j];
            let c2 = cost[i * w + j - 1];
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            cost[i * w + j] = (x[(i - 1) * m + j - 1] + f64::from(c)) as f32;
            trace[i * w + j] = t;
        }
    }
    trace[..w].fill(2);
    for i in 0..=n {
        trace[i * w] = 1;
    }
    let (mut i, mut j) = (n, m);
    let mut path = Vec::new();
    while i > 0 || j > 0 {
        path.push((i.wrapping_sub(1), j.wrapping_sub(1)));
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path.into_iter().unzip()
}

/// Text of tokens as tiktoken's `decode_with_timestamps` renders it (special
/// tokens as their names; only end-of-text occurs here).
fn decode_with_specials(tokenizer: &WhisperTokenizer, tokens: &[u32]) -> String {
    let eot = tokenizer.special.eot;
    let mut bytes = Vec::new();
    for token in tokens {
        if *token == eot {
            bytes.extend_from_slice(b"<|endoftext|>");
        } else {
            bytes.extend_from_slice(tokenizer.token_bytes(*token));
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn split_tokens_on_unicode(
    tokenizer: &WhisperTokenizer,
    tokens: &[u32],
) -> (Vec<String>, Vec<Vec<u32>>) {
    let full: Vec<char> = decode_with_specials(tokenizer, tokens).chars().collect();
    let mut words = Vec::new();
    let mut word_tokens = Vec::new();
    let mut current = Vec::new();
    let mut offset = 0;
    for token in tokens {
        current.push(*token);
        let decoded = decode_with_specials(tokenizer, &current);
        let chars: Vec<char> = decoded.chars().collect();
        let complete = chars
            .iter()
            .position(|c| *c == '\u{fffd}')
            .is_none_or(|index| full.get(offset + index) == Some(&'\u{fffd}'));
        if complete {
            offset += chars.len();
            words.push(decoded);
            word_tokens.push(std::mem::take(&mut current));
        }
    }
    (words, word_tokens)
}

fn split_to_word_tokens(
    tokenizer: &WhisperTokenizer,
    tokens: &[u32],
    language: &str,
) -> (Vec<String>, Vec<Vec<u32>>) {
    let (subwords, subword_tokens) = split_tokens_on_unicode(tokenizer, tokens);
    if NO_SPACE_LANGUAGES.contains(&language) {
        return (subwords, subword_tokens);
    }
    let mut words: Vec<String> = Vec::new();
    let mut word_tokens: Vec<Vec<u32>> = Vec::new();
    for (subword, toks) in subwords.into_iter().zip(subword_tokens) {
        let special = toks[0] >= tokenizer.special.eot;
        let with_space = subword.starts_with(' ');
        let punctuation = ASCII_PUNCTUATION.contains(subword.trim());
        if special || with_space || punctuation || words.is_empty() {
            words.push(subword);
            word_tokens.push(toks);
        } else if let (Some(word), Some(last)) = (words.last_mut(), word_tokens.last_mut()) {
            word.push_str(&subword);
            last.extend(toks);
        }
    }
    (words, word_tokens)
}

/// `find_alignment` for the window the forward holds.
#[expect(clippy::too_many_arguments)]
fn find_alignment(
    forward: &mut dyn Forward,
    tokenizer: &WhisperTokenizer,
    language: &str,
    language_token: u32,
    task_token: u32,
    text_tokens: &[u32],
    num_frames: usize,
    pace: &mut Pace<'_>,
) -> crate::Result<Vec<WordTiming>> {
    if text_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let sp = &tokenizer.special;
    let mut tokens = vec![sp.sot, language_token, task_token, sp.no_timestamps];
    tokens.extend_from_slice(text_tokens);
    tokens.push(sp.eot);
    let frames = num_frames / 2;
    let heads = &sp.alignment_heads;
    let (weights, logits) =
        forward.alignment(&tokens, heads, frames, 3, text_tokens.len(), pace)?;
    let n_tokens = tokens.len();
    // Token probabilities of the text under the forced decode.
    let eot = sp.eot as usize;
    let probabilities: Vec<f64> = text_tokens
        .iter()
        .zip(&logits)
        .map(|(token, row)| {
            let row = &row[..eot];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = row.iter().map(|v| f64::from(v - max).exp()).sum();
            f64::from(row[*token as usize] - max).exp() / sum
        })
        .collect();
    // Softmax over frames, normalize over tokens, median filter, mean of heads.
    let mut matrix = vec![0.0_f64; n_tokens * frames];
    let mut head = vec![0.0_f64; n_tokens * frames];
    for h in 0..heads.len() {
        for t in 0..n_tokens {
            let row = &weights[(h * n_tokens + t) * frames..][..frames];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = row.iter().map(|v| f64::from(v - max).exp()).sum();
            for f in 0..frames {
                head[t * frames + f] = f64::from(row[f] - max).exp() / sum;
            }
        }
        for f in 0..frames {
            let mean = (0..n_tokens).map(|t| head[t * frames + f]).sum::<f64>() / n_tokens as f64;
            let var = (0..n_tokens)
                .map(|t| (head[t * frames + f] - mean).powi(2))
                .sum::<f64>()
                / n_tokens as f64;
            let std = var.sqrt();
            for t in 0..n_tokens {
                head[t * frames + f] = (head[t * frames + f] - mean) / std;
            }
        }
        median_filter(&mut head, n_tokens, frames, MEDFILT_WIDTH);
        for (m, v) in matrix.iter_mut().zip(&head) {
            *m += v;
        }
    }
    let count = heads.len() as f64;
    // Rows from <|notimestamps|> to the last text token, negated.
    let first = 3;
    let rows = n_tokens - 1 - first;
    let cost: Vec<f64> = matrix[first * frames..(first + rows) * frames]
        .iter()
        .map(|v| -(v / count))
        .collect();
    let (text_indices, time_indices) = dtw(&cost, rows, frames);
    let mut with_eot = text_tokens.to_vec();
    with_eot.push(sp.eot);
    let (words, word_tokens) = split_to_word_tokens(tokenizer, &with_eot, language);
    if word_tokens.len() <= 1 {
        return Ok(Vec::new());
    }
    let mut boundaries = vec![0_usize];
    for toks in &word_tokens[..word_tokens.len() - 1] {
        boundaries.push(boundaries[boundaries.len() - 1] + toks.len());
    }
    let jump_times: Vec<f64> = (0..text_indices.len())
        .filter(|k| *k == 0 || text_indices[*k] != text_indices[k - 1])
        .map(|k| time_indices[k] as f64 / TOKENS_PER_SECOND)
        .collect();
    let mut out = Vec::new();
    for (k, (word, toks)) in words.into_iter().zip(word_tokens).enumerate() {
        if k + 1 >= boundaries.len() {
            break;
        }
        let (a, b) = (boundaries[k], boundaries[k + 1]);
        let probability = if b > a {
            probabilities[a..b.min(probabilities.len())]
                .iter()
                .sum::<f64>()
                / (b - a) as f64
        } else {
            f64::NAN
        };
        out.push(WordTiming {
            word,
            tokens: toks,
            start: jump_times.get(a).copied().unwrap_or(0.0),
            end: jump_times.get(b).copied().unwrap_or(0.0),
            probability,
        });
    }
    Ok(out)
}

fn merge_punctuations(alignment: &mut [WordTiming]) {
    if alignment.len() < 2 {
        return;
    }
    let mut j = alignment.len() - 1;
    for p in (0..alignment.len() - 1).rev() {
        if alignment[p].word.starts_with(' ')
            && PREPEND_PUNCTUATIONS.contains(alignment[p].word.trim())
        {
            let previous = std::mem::take(&mut alignment[p].word);
            let previous_tokens = std::mem::take(&mut alignment[p].tokens);
            let following = &mut alignment[j];
            following.word = previous + &following.word;
            let mut tokens = previous_tokens;
            tokens.extend(&following.tokens);
            following.tokens = tokens;
        } else {
            j = p;
        }
    }
    let (mut i, mut j) = (0, 1);
    while j < alignment.len() {
        if !alignment[i].word.ends_with(' ')
            && APPEND_PUNCTUATIONS.contains(alignment[j].word.as_str())
        {
            let following = std::mem::take(&mut alignment[j].word);
            let following_tokens = std::mem::take(&mut alignment[j].tokens);
            alignment[i].word.push_str(&following);
            alignment[i].tokens.extend(following_tokens);
        } else {
            i = j;
        }
        j += 1;
    }
}

/// `add_word_timestamps` over one window's segments (which must not be
/// empty of text tokens for words to appear).
#[expect(clippy::too_many_arguments)]
pub fn add_word_timestamps(
    segments: &mut [Segment],
    forward: &mut dyn Forward,
    tokenizer: &WhisperTokenizer,
    language: &str,
    language_token: u32,
    task_token: u32,
    num_frames: usize,
    mut last_speech_timestamp: f64,
    pace: &mut Pace<'_>,
) -> crate::Result<()> {
    if segments.is_empty() {
        return Ok(());
    }
    let eot = tokenizer.special.eot;
    let per_segment: Vec<Vec<u32>> = segments
        .iter()
        .map(|s| s.tokens.iter().copied().filter(|t| *t < eot).collect())
        .collect();
    let text_tokens: Vec<u32> = per_segment.iter().flatten().copied().collect();
    let mut alignment = find_alignment(
        forward,
        tokenizer,
        language,
        language_token,
        task_token,
        &text_tokens,
        num_frames,
        pace,
    )?;
    let mut durations: Vec<f64> = alignment
        .iter()
        .map(|t| t.end - t.start)
        .filter(|d| *d != 0.0)
        .collect();
    durations.sort_by(f64::total_cmp);
    let median = match durations.len() {
        0 => 0.0,
        n if n % 2 == 1 => durations[n / 2],
        n => f64::midpoint(durations[n / 2 - 1], durations[n / 2]),
    };
    let median_duration = median.min(0.7);
    let max_duration = median_duration * 2.0;
    if !durations.is_empty() {
        let sentence_end_marks = ".。!！?？";
        for i in 1..alignment.len() {
            if alignment[i].end - alignment[i].start > max_duration {
                if sentence_end_marks.contains(alignment[i].word.as_str()) {
                    alignment[i].end = alignment[i].start + max_duration;
                } else if sentence_end_marks.contains(alignment[i - 1].word.as_str()) {
                    alignment[i].start = alignment[i].end - max_duration;
                }
            }
        }
    }
    merge_punctuations(&mut alignment);
    let time_offset = (segments[0].seek * 160) as f64 / 16_000.0;
    let mut word_index = 0;
    for (segment, text_tokens) in segments.iter_mut().zip(&per_segment) {
        let mut saved_tokens = 0;
        let mut words: Vec<Word> = Vec::new();
        while word_index < alignment.len() && saved_tokens < text_tokens.len() {
            let timing = &alignment[word_index];
            if !timing.word.is_empty() {
                words.push(Word {
                    word: timing.word.clone(),
                    start: round2(time_offset + timing.start),
                    end: round2(time_offset + timing.end),
                    probability: timing.probability,
                });
            }
            saved_tokens += timing.tokens.len();
            word_index += 1;
        }
        if !words.is_empty() {
            let n = words.len();
            if words[0].end - last_speech_timestamp > median_duration * 4.0
                && (words[0].end - words[0].start > max_duration
                    || (n > 1 && words[1].end - words[0].start > max_duration * 2.0))
            {
                if n > 1 && words[1].end - words[1].start > max_duration {
                    let boundary = (words[1].end / 2.0).max(words[1].end - max_duration);
                    words[0].end = boundary;
                    words[1].start = boundary;
                }
                words[0].start = (words[0].end - max_duration).max(0.0);
            }
            if segment.start < words[0].end && segment.start - 0.5 > words[0].start {
                words[0].start = (words[0].end - median_duration).min(segment.start).max(0.0);
            } else {
                segment.start = words[0].start;
            }
            let last = words.len() - 1;
            if segment.end > words[last].start && segment.end + 0.5 < words[last].end {
                words[last].end = (words[last].start + median_duration).max(segment.end);
            } else {
                segment.end = words[last].end;
            }
            last_speech_timestamp = segment.end;
        }
        segment.words = words;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_rounding_is_reproduced() {
        assert_eq!(round2(1.005), 1.0); // 1.005 is 1.00499999999999989...
        assert_eq!(round2(2.675), 2.67);
        assert_eq!(round2(0.125), 0.12);
        assert_eq!(round2(7.777), 7.78);
    }

    #[test]
    fn median_filter_reflects_at_the_edges() {
        let mut x = vec![5.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0];
        median_filter(&mut x, 1, 8, 3);
        assert_eq!(x, vec![1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 7.0]);
        let mut short = vec![1.0, 2.0, 3.0];
        median_filter(&mut short, 1, 3, 7);
        assert_eq!(short, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dtw_walks_the_cheapest_monotonic_path() {
        // Two rows over four frames: row 0 cheap at frames 0-1, row 1 at 2-3.
        // Ties go to the left move, as in the reference (a vertical step at
        // frame 1 here).
        let cost = vec![-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0];
        let (text, time) = dtw(&cost, 2, 4);
        assert_eq!(text, vec![0, 0, 1, 1, 1]);
        assert_eq!(time, vec![0, 1, 1, 2, 3]);
    }

    #[test]
    fn punctuation_merges_into_its_neighbours() {
        let timing = |word: &str| WordTiming {
            word: word.into(),
            tokens: vec![1],
            start: 0.0,
            end: 0.0,
            probability: 1.0,
        };
        let mut words = vec![
            timing(" \""),
            timing("Hello"),
            timing(","),
            timing(" world"),
            timing("."),
        ];
        merge_punctuations(&mut words);
        let texts: Vec<&str> = words.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(texts, vec!["", " \"Hello,", "", " world.", ""]);
        assert_eq!(words[1].tokens.len(), 3);
    }
}
