//! The deterministic grader arithmetic: `string_check` and the
//! `text_similarity` metrics this server computes exactly.
//!
//! `OpenAI` names the libraries behind its metrics but not every parameter,
//! so each metric here is pinned to one published reference implementation
//! (the versions in `OpenAI`'s documented grader image) and tested against
//! values that implementation computes:
//!
//! | metric        | reference                                                        |
//! |---------------|------------------------------------------------------------------|
//! | `fuzzy_match` | `rapidfuzz` 3.10.1 `fuzz.WRatio(input, reference, processor=utils.default_process) / 100` |
//! | `bleu`        | `nltk` 3.9.1 `sentence_bleu([reference.split()], input.split())` (BLEU-4, uniform weights, no smoothing) |
//! | `gleu`        | `nltk` 3.9.1 `sentence_gleu([reference.split()], input.split())` (1- to 4-grams) |
//! | `rouge_1`..`rouge_5`, `rouge_l` | `rouge-score` 0.1.2 `RougeScorer([...], use_stemmer=False)` F-measure |
//!
//! `bleu` and `gleu` split on whitespace and are case-sensitive; ROUGE uses
//! its own tokenizer (lowercase, ASCII letters and digits only).
//! `fuzzy_match` treats letters, digits and `_` as word characters, as
//! Python's `\w` does; the two can disagree only on rare non-ASCII
//! characters whose Unicode properties Python and Rust classify differently.
//!
//! `meteor` (which needs `WordNet` synonyms and a stemmer) and `cosine` (which
//! needs a validated embedding model) are not computed: graders using them
//! are refused when an eval is created.

use std::collections::{HashMap, HashSet};

/// A `string_check` operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Eq,
    Ne,
    Like,
    Ilike,
}

/// Whether `input` passes the check against `reference`: `eq` and `ne`
/// compare exactly, `like` tests that `input` contains `reference`, and
/// `ilike` does so ignoring case.
#[must_use]
pub fn string_check(operation: Operation, input: &str, reference: &str) -> bool {
    match operation {
        Operation::Eq => input == reference,
        Operation::Ne => input != reference,
        Operation::Like => input.contains(reference),
        Operation::Ilike => input.to_lowercase().contains(&reference.to_lowercase()),
    }
}

/// A `text_similarity` metric this server computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    FuzzyMatch,
    Bleu,
    Gleu,
    /// ROUGE-N for N in 1 to 5.
    Rouge(usize),
    RougeL,
}

/// The similarity of `input` to `reference`, from 0 to 1.
#[must_use]
pub fn similarity(metric: Metric, input: &str, reference: &str) -> f64 {
    match metric {
        Metric::FuzzyMatch => w_ratio(&default_process(input), &default_process(reference)) / 100.0,
        Metric::Bleu => bleu(&words(input), &words(reference)),
        Metric::Gleu => gleu(&words(input), &words(reference)),
        Metric::Rouge(n) => rouge_n(&rouge_tokens(input), &rouge_tokens(reference), n),
        Metric::RougeL => rouge_l(&rouge_tokens(input), &rouge_tokens(reference)),
    }
}

fn words(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

// ---------------------------------------------------------------- BLEU, GLEU

/// Counts of every `n`-gram of `tokens`.
fn ngrams<T: Eq + std::hash::Hash>(tokens: &[T], n: usize) -> HashMap<&[T], usize> {
    let mut counts = HashMap::new();
    if n > 0 && tokens.len() >= n {
        for gram in tokens.windows(n) {
            *counts.entry(gram).or_insert(0) += 1;
        }
    }
    counts
}

/// Matches of `hypothesis` n-grams clipped by `reference` counts, and the
/// hypothesis n-gram total.
fn clipped<T: Eq + std::hash::Hash>(hypothesis: &[T], reference: &[T], n: usize) -> (usize, usize) {
    let hyp = ngrams(hypothesis, n);
    let reference = ngrams(reference, n);
    let matched = hyp
        .iter()
        .map(|(gram, count)| (*count).min(reference.get(gram).copied().unwrap_or(0)))
        .sum();
    (matched, hyp.values().sum())
}

/// `nltk` `sentence_bleu` with one reference, BLEU-4 weights and its default
/// (no) smoothing: an n-gram order without matches contributes the smallest
/// positive float, so the score is then effectively zero.
fn bleu(hypothesis: &[&str], reference: &[&str]) -> f64 {
    let mut precisions = [0.0_f64; 4];
    let mut unigram_matches = 0;
    for (index, slot) in precisions.iter_mut().enumerate() {
        let (matched, total) = clipped(hypothesis, reference, index + 1);
        if index == 0 {
            unigram_matches = matched;
        }
        *slot = if matched == 0 {
            f64::MIN_POSITIVE
        } else {
            matched as f64 / total.max(1) as f64
        };
    }
    if unigram_matches == 0 {
        return 0.0;
    }
    let (hyp_len, ref_len) = (hypothesis.len(), reference.len());
    let penalty = if hyp_len > ref_len {
        1.0
    } else if hyp_len == 0 {
        0.0
    } else {
        (1.0 - ref_len as f64 / hyp_len as f64).exp()
    };
    let sum: f64 = precisions.iter().map(|p| 0.25 * p.ln()).sum();
    penalty * sum.exp()
}

/// `nltk` `sentence_gleu` with one reference over 1- to 4-grams: matched
/// n-grams over the larger of the hypothesis and reference n-gram totals.
fn gleu(hypothesis: &[&str], reference: &[&str]) -> f64 {
    let (mut matched, mut hyp_total, mut ref_total) = (0, 0, 0);
    for n in 1..=4 {
        let (hit, total) = clipped(hypothesis, reference, n);
        matched += hit;
        hyp_total += total;
        ref_total += reference.len().saturating_sub(n - 1);
    }
    let all = hyp_total.max(ref_total);
    if all == 0 {
        0.0
    } else {
        matched as f64 / all as f64
    }
}

// --------------------------------------------------------------------- ROUGE

/// The `rouge-score` default tokenizer without stemming.
fn rouge_tokens(text: &str) -> Vec<String> {
    let lowered: String = text
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() {
                c
            } else {
                ' '
            }
        })
        .collect();
    lowered.split_whitespace().map(str::to_owned).collect()
}

fn f_measure(precision: f64, recall: f64) -> f64 {
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

fn rouge_n(prediction: &[String], target: &[String], n: usize) -> f64 {
    let predicted = ngrams(prediction, n);
    let targets = ngrams(target, n);
    let overlap: usize = targets
        .iter()
        .map(|(gram, count)| (*count).min(predicted.get(gram).copied().unwrap_or(0)))
        .sum();
    let predicted_total: usize = predicted.values().sum();
    let target_total: usize = targets.values().sum();
    let precision = overlap as f64 / predicted_total.max(1) as f64;
    let recall = overlap as f64 / target_total.max(1) as f64;
    f_measure(precision, recall)
}

fn rouge_l(prediction: &[String], target: &[String]) -> f64 {
    if prediction.is_empty() || target.is_empty() {
        return 0.0;
    }
    let mut previous = vec![0_usize; prediction.len() + 1];
    let mut current = vec![0_usize; prediction.len() + 1];
    for token in target {
        for (j, candidate) in prediction.iter().enumerate() {
            current[j + 1] = if token == candidate {
                previous[j] + 1
            } else {
                previous[j + 1].max(current[j])
            };
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let lcs = previous[prediction.len()] as f64;
    f_measure(lcs / prediction.len() as f64, lcs / target.len() as f64)
}

// ------------------------------------------------------------ fuzzy matching

/// `rapidfuzz.utils.default_process`: every non-word character becomes a
/// space, then the text is trimmed and lowercased.
fn default_process(text: &str) -> Vec<char> {
    let replaced: String = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                ' '
            }
        })
        .collect();
    replaced.trim().to_lowercase().chars().collect()
}

/// The longest common subsequence of `pattern` and `text`, bit-parallel
/// (Hyyrö) over 64-bit words.
struct Pattern {
    len: usize,
    masks: HashMap<char, Vec<u64>>,
    words: usize,
}

impl Pattern {
    fn new(pattern: &[char]) -> Self {
        let words = pattern.len().div_ceil(64).max(1);
        let mut masks: HashMap<char, Vec<u64>> = HashMap::new();
        for (index, c) in pattern.iter().enumerate() {
            let mask = masks.entry(*c).or_insert_with(|| vec![0; words]);
            mask[index / 64] |= 1 << (index % 64);
        }
        Self {
            len: pattern.len(),
            masks,
            words,
        }
    }

    fn lcs(&self, text: &[char]) -> usize {
        if self.len == 0 || text.is_empty() {
            return 0;
        }
        let mut state = vec![u64::MAX; self.words];
        let zero = vec![0; self.words];
        for c in text {
            let mask = self.masks.get(c).unwrap_or(&zero);
            let mut carry = false;
            for (word, m) in state.iter_mut().zip(mask) {
                let matched = *word & m;
                let (sum, first) = word.overflowing_add(matched);
                let (sum, second) = sum.overflowing_add(u64::from(carry));
                carry = first || second;
                *word = sum | (*word - matched);
            }
        }
        let mut zeros = 0;
        for (index, word) in state.iter().enumerate() {
            let bits = (self.len - index * 64).min(64);
            let valid = if bits == 64 {
                u64::MAX
            } else {
                (1 << bits) - 1
            };
            zeros += (!word & valid).count_ones() as usize;
        }
        zeros
    }

    /// The normalized Indel similarity of the pattern and `text`, 0 to 1.
    fn similarity(&self, text: &[char]) -> f64 {
        let total = self.len + text.len();
        if total == 0 {
            return 1.0;
        }
        let distance = total - 2 * self.lcs(text);
        1.0 - distance as f64 / total as f64
    }
}

/// `fuzz.ratio`: the normalized Indel similarity, 0 to 100.
fn ratio(a: &[char], b: &[char]) -> f64 {
    Pattern::new(a).similarity(b) * 100.0
}

fn indel_distance(a: &[char], b: &[char]) -> usize {
    a.len() + b.len() - 2 * Pattern::new(a).lcs(b)
}

/// `_partial_ratio_impl`: the best alignment of the shorter `needle` in
/// `haystack`, 0 to 100.
fn partial_ratio_impl(needle: &[char], haystack: &[char]) -> f64 {
    let pattern = Pattern::new(needle);
    let present: HashSet<char> = needle.iter().copied().collect();
    let (len1, len2) = (needle.len(), haystack.len());
    let mut best = 0.0_f64;
    let mut consider = |window: &[char]| {
        let score = pattern.similarity(window);
        if score > best {
            best = score;
        }
        best >= 1.0
    };
    for i in 1..len1 {
        if present.contains(&haystack[i - 1]) && consider(&haystack[..i]) {
            return 100.0;
        }
    }
    for i in 0..len2 - len1 {
        if present.contains(&haystack[i + len1 - 1]) && consider(&haystack[i..i + len1]) {
            return 100.0;
        }
    }
    for i in len2 - len1..len2 {
        if present.contains(&haystack[i]) && consider(&haystack[i..]) {
            return 100.0;
        }
    }
    best * 100.0
}

/// `fuzz.partial_ratio`, 0 to 100.
fn partial_ratio(a: &[char], b: &[char]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 100.0;
    }
    let (shorter, longer) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if shorter.is_empty() {
        return 0.0;
    }
    let mut score = partial_ratio_impl(shorter, longer);
    if score < 100.0 && a.len() == b.len() {
        score = score.max(partial_ratio_impl(longer, shorter));
    }
    score
}

fn split(text: &[char]) -> Vec<String> {
    text.iter()
        .collect::<String>()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

fn join<'a>(tokens: impl IntoIterator<Item = &'a String>) -> Vec<char> {
    let mut joined = Vec::new();
    for (index, token) in tokens.into_iter().enumerate() {
        if index > 0 {
            joined.push(' ');
        }
        joined.extend(token.chars());
    }
    joined
}

fn sorted_join(mut tokens: Vec<&String>) -> Vec<char> {
    tokens.sort();
    join(tokens)
}

fn token_sort_ratio(a: &[char], b: &[char]) -> f64 {
    let (a, b) = (split(a), split(b));
    ratio(
        &sorted_join(a.iter().collect()),
        &sorted_join(b.iter().collect()),
    )
}

fn norm_distance(distance: usize, total: usize) -> f64 {
    if total == 0 {
        100.0
    } else {
        100.0 - 100.0 * distance as f64 / total as f64
    }
}

fn token_set_ratio(a: &[char], b: &[char]) -> f64 {
    let (a, b) = (split(a), split(b));
    let tokens_a: HashSet<&String> = a.iter().collect();
    let tokens_b: HashSet<&String> = b.iter().collect();
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }
    let intersection: Vec<&String> = tokens_a.intersection(&tokens_b).copied().collect();
    let diff_ab: Vec<&String> = tokens_a.difference(&tokens_b).copied().collect();
    let diff_ba: Vec<&String> = tokens_b.difference(&tokens_a).copied().collect();
    if !intersection.is_empty() && (diff_ab.is_empty() || diff_ba.is_empty()) {
        return 100.0;
    }
    let diff_ab = sorted_join(diff_ab);
    let diff_ba = sorted_join(diff_ba);
    let sect_len = join(intersection.iter().copied()).len();
    let separator = usize::from(sect_len != 0);
    let sect_ab_len = sect_len + separator + diff_ab.len();
    let sect_ba_len = sect_len + separator + diff_ba.len();
    let result = norm_distance(
        indel_distance(&diff_ab, &diff_ba),
        sect_ab_len + sect_ba_len,
    );
    if sect_len == 0 {
        return result;
    }
    let sect_ab = norm_distance(separator + diff_ab.len(), sect_len + sect_ab_len);
    let sect_ba = norm_distance(separator + diff_ba.len(), sect_len + sect_ba_len);
    result.max(sect_ab).max(sect_ba)
}

fn token_ratio(a: &[char], b: &[char]) -> f64 {
    token_set_ratio(a, b).max(token_sort_ratio(a, b))
}

fn partial_token_ratio(a: &[char], b: &[char]) -> f64 {
    let (split_a, split_b) = (split(a), split(b));
    let tokens_a: HashSet<&String> = split_a.iter().collect();
    let tokens_b: HashSet<&String> = split_b.iter().collect();
    if tokens_a.intersection(&tokens_b).next().is_some() {
        return 100.0;
    }
    let diff_ab: Vec<&String> = tokens_a.difference(&tokens_b).copied().collect();
    let diff_ba: Vec<&String> = tokens_b.difference(&tokens_a).copied().collect();
    let result = partial_ratio(
        &sorted_join(split_a.iter().collect()),
        &sorted_join(split_b.iter().collect()),
    );
    if split_a.len() == diff_ab.len() && split_b.len() == diff_ba.len() {
        return result;
    }
    result.max(partial_ratio(&sorted_join(diff_ab), &sorted_join(diff_ba)))
}

/// `fuzz.WRatio` on already processed text, 0 to 100.
fn w_ratio(a: &[char], b: &[char]) -> f64 {
    const UNBASE_SCALE: f64 = 0.95;
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (len1, len2) = (a.len() as f64, b.len() as f64);
    let len_ratio = if len1 > len2 {
        len1 / len2
    } else {
        len2 / len1
    };
    let end_ratio = ratio(a, b);
    if len_ratio < 1.5 {
        return end_ratio.max(token_ratio(a, b) * UNBASE_SCALE);
    }
    let partial_scale = if len_ratio < 8.0 { 0.9 } else { 0.6 };
    let end_ratio = end_ratio.max(partial_ratio(a, b) * partial_scale);
    end_ratio.max(partial_token_ratio(a, b) * UNBASE_SCALE * partial_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{what}: got {actual}, expected {expected}"
        );
    }

    #[test]
    fn string_checks_follow_the_documented_operations() {
        assert!(string_check(Operation::Eq, "Paris", "Paris"));
        assert!(!string_check(Operation::Eq, "Paris", "paris"));
        assert!(string_check(Operation::Ne, "Paris", "paris"));
        assert!(!string_check(Operation::Ne, "Paris", "Paris"));
        assert!(string_check(
            Operation::Like,
            "The capital is Paris.",
            "Paris"
        ));
        assert!(!string_check(
            Operation::Like,
            "The capital is PARIS.",
            "Paris"
        ));
        assert!(string_check(
            Operation::Ilike,
            "The capital is PARIS.",
            "paris"
        ));
        assert!(!string_check(
            Operation::Ilike,
            "The capital is Rome.",
            "paris"
        ));
        assert!(string_check(Operation::Like, "anything", ""));
    }

    /// (input, reference) pairs and the values the reference libraries
    /// compute for them (`rapidfuzz` 3.10.1, `nltk` 3.9.1, `rouge-score`
    /// 0.1.2), in the order fuzzy, bleu, gleu, rouge 1-5, rouge L.
    #[allow(clippy::unreadable_literal)] // as the reference libraries print them
    const CASES: &[(&str, &str, [f64; 9])] = &[
        (
            "It is a guide to action which ensures that the military always obeys the commands of the party",
            "It is a guide to action that ensures that the military will forever heed Party commands",
            [
                0.832876712328767,
                0.41180376356915777,
                0.4393939393939394,
                0.7058823529411765,
                0.5,
                0.39999999999999997,
                0.28571428571428575,
                0.15384615384615383,
                0.6470588235294118,
            ],
        ),
        (
            "the cat sat on the mat",
            "the cat is on the mat",
            [
                0.9302325581395349,
                7.262123179505913e-78,
                0.5,
                0.8333333333333334,
                0.6,
                0.25,
                0.0,
                0.0,
                0.8333333333333334,
            ],
        ),
        (
            "Paris",
            "The capital of France is Paris.",
            [
                0.9,
                0.0,
                0.0,
                0.2857142857142857,
                0.0,
                0.0,
                0.0,
                0.0,
                0.2857142857142857,
            ],
        ),
        (
            "New York Mets vs Atlanta Braves",
            "Atlanta Braves vs New York Mets",
            [
                0.95,
                7.600793306264725e-78,
                0.5555555555555556,
                1.0,
                0.6,
                0.25,
                0.0,
                0.0,
                0.5,
            ],
        ),
        (
            "fuzzy wuzzy was a bear",
            "wuzzy fuzzy was a bear",
            [
                0.95,
                7.803659209895733e-78,
                0.5714285714285714,
                1.0,
                0.5,
                0.3333333333333333,
                0.0,
                0.0,
                0.8000000000000002,
            ],
        ),
        (
            "",
            "anything",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "Le café coûte 3,50 € à Zürich!",
            "le cafe coute 3.50 euros a zurich",
            [
                0.7741935483870968,
                0.0,
                0.0,
                0.375,
                0.14285714285714285,
                0.0,
                0.0,
                0.0,
                0.375,
            ],
        ),
        (
            "Paris is the capital and most populous city of France, with an estimated population of 2,102,650 residents",
            "Paris",
            [
                0.6,
                8.972141065609098e-232,
                0.016129032258064516,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
            ],
        ),
        (
            "the the the the the the the",
            "the cat is on the mat",
            [
                0.95,
                1.331960397810445e-231,
                0.09090909090909091,
                0.30769230769230765,
                0.0,
                0.0,
                0.0,
                0.0,
                0.30769230769230765,
            ],
        ),
        (
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        (
            "ok",
            "the quick brown fox jumps over the lazy dog and keeps running far beyond the hills",
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ];

    /// The worked examples the reference libraries publish: the
    /// `rouge-score` README (`fmeasure` 0.70588235 and 0.58823529, that is
    /// 12/17 and 10/17) and the `nltk` `sentence_gleu` doctests (`0.4393...`,
    /// `0.0909...`).
    #[test]
    fn published_examples_are_reproduced() {
        let (target, prediction) = (
            "The quick brown fox jumps over the lazy dog",
            "The quick brown dog jumps on the log.",
        );
        close(
            similarity(Metric::Rouge(1), prediction, target),
            12.0 / 17.0,
            "README rouge1",
        );
        close(
            similarity(Metric::RougeL, prediction, target),
            10.0 / 17.0,
            "README rougeL",
        );
        let truncated = |value: f64| (value * 10_000.0).floor();
        let reference = "It is a guide to action that ensures that the military will forever heed \
                         Party commands";
        let hypothesis = "It is a guide to action which ensures that the military always obeys \
                          the commands of the party";
        close(
            truncated(similarity(Metric::Gleu, hypothesis, reference)),
            4393.0,
            "gleu doctest",
        );
        let repeated = similarity(
            Metric::Gleu,
            "the the the the the the the",
            "the cat is on the mat",
        );
        close(truncated(repeated), 909.0, "gleu doctest");
    }

    #[test]
    fn similarity_metrics_match_the_reference_implementations() {
        let metrics = [
            Metric::FuzzyMatch,
            Metric::Bleu,
            Metric::Gleu,
            Metric::Rouge(1),
            Metric::Rouge(2),
            Metric::Rouge(3),
            Metric::Rouge(4),
            Metric::Rouge(5),
            Metric::RougeL,
        ];
        for (input, reference, expected) in CASES {
            for (metric, expected) in metrics.iter().zip(expected) {
                close(
                    similarity(*metric, input, reference),
                    *expected,
                    &format!("{metric:?} of {input:?} against {reference:?}"),
                );
            }
        }
    }
}
