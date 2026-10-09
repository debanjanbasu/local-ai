//! Lossless suffix-lookup speculative drafting.

use std::collections::HashMap;

pub const DEFAULT_NGRAM_ENABLED: bool = true;
pub const DEFAULT_NGRAM_MAX: usize = 63;
/// The suffix length that must match before lookup drafts instead of MTP.
/// Measured on the M4 Pro, greedy, byte-identical text: three edit prompts that
/// echo quoted code decoded at 55.9 tok/s geomean with 12 against 55.1 with 16
/// and 52.8 with 24; 4 and 8 fired spurious lookups on novel prose, while 12
/// fired none on four novel prompts. Re-measured on the current kernels with
/// two rename-an-identifier edits of ~50 quoted lines (600 tokens): 8, 10, 12
/// and 16 gave 60.3, 58.9, 60.1 and 58.5 tok/s geomean against 47.5 without
/// lookup. 8 tied 12 only because the two edits disagreed by ±10 %, so 12
/// stays.
pub const DEFAULT_NGRAM_MIN_MATCH: usize = 12;
/// 63 drafts plus the seed fill one 64-token verify tile; a 65th row would
/// spill into the 128-token tile and cost half as much again.
pub const MAX_NGRAM_MAX: usize = 63;
pub const MAX_STORED_REQUESTS: usize = 8;
pub const MAX_STORED_TOKENS: usize = 131_072;

/// One place where the `min_match`-token anchor ends.
#[derive(Clone, Copy, Debug)]
struct Location {
    sequence: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NgramSettings {
    pub enabled: bool,
    pub max_drafts: usize,
    pub min_match: usize,
}

impl Default for NgramSettings {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_NGRAM_ENABLED,
            max_drafts: DEFAULT_NGRAM_MAX,
            min_match: DEFAULT_NGRAM_MIN_MATCH,
        }
    }
}

/// The verifier's current lookup capacity. This is deliberately the single
/// policy hook to repoint when the native verifier exposes its row capacity.
pub const fn max_lookup_drafts() -> usize {
    MAX_NGRAM_MAX
}

/// Bounded request-level corpus retained by one engine (and therefore by one
/// server worker, chat session, or `bonsai --repeat` process).
#[derive(Default)]
pub struct SuffixStore {
    requests: std::collections::VecDeque<Vec<u32>>,
    tokens: usize,
}

impl SuffixStore {
    pub fn session(&self, prompt: &[u32], min_match: usize) -> SuffixSession {
        let mut sequences: Vec<Vec<u32>> = self.requests.iter().cloned().collect();
        sequences.push(prompt.to_vec());
        SuffixSession::new(sequences, min_match)
    }

    pub fn remember(&mut self, mut request: Vec<u32>) {
        if request.len() > MAX_STORED_TOKENS {
            request.drain(..request.len() - MAX_STORED_TOKENS);
        }
        self.tokens += request.len();
        self.requests.push_back(request);
        while self.requests.len() > MAX_STORED_REQUESTS || self.tokens > MAX_STORED_TOKENS {
            if let Some(removed) = self.requests.pop_front() {
                self.tokens -= removed.len();
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct LookupDraft {
    pub tokens: Vec<u32>,
    pub match_len: usize,
    pub occurrences: usize,
}

/// Incremental exact matcher. A fixed-size anchor finds candidate occurrences
/// in O(1) expected time; candidates are extended backwards to obtain the exact
/// longest suffix. This avoids rescanning a 100K-token prompt each decode step.
pub struct SuffixSession {
    sequences: Vec<Vec<u32>>,
    anchors: HashMap<Vec<u32>, Vec<Location>>,
    min_match: usize,
}

impl SuffixSession {
    fn new(sequences: Vec<Vec<u32>>, min_match: usize) -> Self {
        let mut result = Self {
            sequences,
            anchors: HashMap::new(),
            min_match,
        };
        for sequence in 0..result.sequences.len() {
            for end in min_match..=result.sequences[sequence].len() {
                result.index(sequence, end);
            }
        }
        result
    }

    fn index(&mut self, sequence: usize, end: usize) {
        if self.min_match == 0 || end < self.min_match {
            return;
        }
        let key = self.sequences[sequence][end - self.min_match..end].to_vec();
        self.anchors
            .entry(key)
            .or_default()
            .push(Location { sequence, end });
    }

    pub fn append(&mut self, token: u32) {
        let sequence = self.sequences.len() - 1;
        self.sequences[sequence].push(token);
        let end = self.sequences[sequence].len();
        self.index(sequence, end);
    }

    pub fn history(&self) -> &[u32] {
        &self.sequences[self.sequences.len() - 1]
    }

    pub fn find(&self, max_drafts: usize) -> Option<LookupDraft> {
        let current_index = self.sequences.len() - 1;
        let current = &self.sequences[current_index];
        if self.min_match == 0 || max_drafts == 0 || current.len() < self.min_match {
            return None;
        }
        let key = &current[current.len() - self.min_match..];
        let locations = self.anchors.get(key)?;
        let mut best_match = 0;
        let mut candidates = Vec::new();
        for &location in locations.iter().rev() {
            let source = &self.sequences[location.sequence];
            if location.end >= source.len()
                || (location.sequence == current_index && location.end == current.len())
            {
                continue;
            }
            let available = location.end.min(current.len());
            let mut matched = self.min_match;
            while matched < available
                && source[location.end - matched - 1] == current[current.len() - matched - 1]
            {
                matched += 1;
            }
            // Do not copy tokens from the suffix currently being matched.
            let mut continuation_end = (location.end + max_drafts).min(source.len());
            if location.sequence == current_index {
                continuation_end = continuation_end.min(current.len().saturating_sub(matched));
            }
            // A long self-referential match can pull `continuation_end` below
            // `location.end`; the continuation must be dropped, never sliced
            // backwards.
            if continuation_end <= location.end {
                continue;
            }
            if matched > best_match {
                best_match = matched;
                candidates.clear();
            }
            if matched == best_match {
                candidates.push(source[location.end..continuation_end].to_vec());
            }
        }
        // Frequency wins, then continuation length, then recency (insertion order).
        let mut counts: HashMap<&[u32], usize> = HashMap::new();
        for candidate in &candidates {
            *counts.entry(candidate).or_default() += 1;
        }
        let (index, tokens) = candidates
            .iter()
            .enumerate()
            .max_by_key(|(index, candidate)| {
                (
                    counts.get(candidate.as_slice()).copied().unwrap_or(0),
                    candidate.len(),
                    *index,
                )
            })?;
        let _ = index;
        Some(LookupDraft {
            occurrences: counts.get(tokens.as_slice()).copied().unwrap_or(1),
            tokens: tokens.clone(),
            match_len: best_match,
        })
    }
}

/// Verify cost is flat within a tensor tile (see
/// `local_metal::bonsai::DEFAULT_SMALL_BATCH_MAX`): past the small-batch
/// range, extend a draft to fill its 32- or 64-token tile when the matched
/// continuation has the tokens, since the extra rows cost nothing.
pub fn fill_verify_tile(depth: usize, available: usize) -> usize {
    let rows = depth + 1;
    if rows <= local_metal::bonsai::DEFAULT_SMALL_BATCH_MAX as usize {
        return depth;
    }
    let tile = if rows <= 32 { 32 } else { 64 };
    available.min((tile - 1).max(depth))
}

/// Adaptive length/backoff policy. Full acceptance grows one row; any
/// rejection halves the next lookup. A long exact match is the confidence
/// gate; otherwise the caller falls back to MTP.
pub struct LookupPolicy {
    limit: usize,
}

impl LookupPolicy {
    pub fn new(max_drafts: usize) -> Self {
        Self {
            limit: max_drafts.min(max_lookup_drafts()),
        }
    }

    pub fn depth(&self, matched: usize, configured_max: usize) -> usize {
        // alpha=1/2 leaves verification work proportional to evidence.
        self.limit.min(configured_max).min(matched / 2)
    }

    pub fn observe(&mut self, accepted: usize, proposed: usize, configured_max: usize) {
        if accepted == proposed {
            self.limit = (self.limit * 2)
                .min(configured_max)
                .min(max_lookup_drafts());
        } else {
            self.limit = (accepted + 1).max(self.limit / 2).max(1);
        }
    }
}

#[cfg(test)]
mod tests;
