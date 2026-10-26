//! Lossless suffix-lookup speculative drafting.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

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
///
/// Each request's anchors are indexed once, the first time a session needs
/// them, and shared by every session made after that: admitting a request
/// indexes only its own prompt.
#[derive(Default)]
pub struct SuffixStore {
    /// Oldest first.
    requests: std::collections::VecDeque<Arc<Anchors>>,
    tokens: usize,
}

impl SuffixStore {
    pub fn session(&mut self, prompt: &[u32], min_match: usize) -> SuffixSession {
        for request in &mut self.requests {
            if request.min_match != min_match {
                *request = Arc::new(Anchors::new(request.tokens.clone(), min_match));
            }
        }
        SuffixSession {
            stored: self.requests.iter().rev().cloned().collect(),
            current: Anchors::new(prompt.to_vec(), min_match),
        }
    }

    pub fn remember(&mut self, mut request: Vec<u32>) {
        if request.len() > MAX_STORED_TOKENS {
            request.drain(..request.len() - MAX_STORED_TOKENS);
        }
        self.tokens += request.len();
        self.requests
            .push_back(Arc::new(Anchors::unindexed(request)));
        while self.requests.len() > MAX_STORED_REQUESTS || self.tokens > MAX_STORED_TOKENS {
            if let Some(removed) = self.requests.pop_front() {
                self.tokens -= removed.tokens.len();
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

/// No earlier anchor with the same hash.
const NO_ANCHOR: u32 = u32::MAX;
/// Multiplier of the rolling anchor hash (odd, so no token is lost).
const HASH_BASE: u64 = 0x9e37_79b9_7f4a_7c15;

/// Hasher for keys that are already mixed hashes.
#[derive(Default)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 << 8) | u64::from(byte);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

/// Spread a rolling hash over every bit (`SplitMix64`'s finalizer).
const fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// One token sequence and every place a `min_match`-token anchor ends in it,
/// by a rolling hash of the anchor: the newest end of each hash, and from each
/// end the previous one with the same hash. Collisions are told apart by
/// comparing the tokens on lookup.
struct Anchors {
    tokens: Vec<u32>,
    min_match: usize,
    newest: HashMap<u64, u32, BuildHasherDefault<PassThrough>>,
    /// The previous end with the same hash as end `min_match + index`.
    previous: Vec<u32>,
    /// Rolling hash of the last `min_match` tokens.
    hash: u64,
    /// `HASH_BASE` to the `min_match`, which drops a token from `hash`.
    drop_factor: u64,
}

impl Anchors {
    /// Tokens awaiting an index (see [`SuffixStore::session`]).
    fn unindexed(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            min_match: usize::MAX,
            newest: HashMap::default(),
            previous: Vec::new(),
            hash: 0,
            drop_factor: 0,
        }
    }

    fn new(tokens: Vec<u32>, min_match: usize) -> Self {
        let mut anchors = Self {
            tokens: Vec::with_capacity(tokens.len()),
            min_match,
            newest: HashMap::with_capacity_and_hasher(
                tokens.len().saturating_sub(min_match) + 1,
                BuildHasherDefault::default(),
            ),
            previous: Vec::with_capacity(tokens.len().saturating_sub(min_match) + 1),
            hash: 0,
            drop_factor: (0..min_match).fold(1, |power: u64, _| power.wrapping_mul(HASH_BASE)),
        };
        for token in tokens {
            anchors.push(token);
        }
        anchors
    }

    fn push(&mut self, token: u32) {
        self.tokens.push(token);
        if self.min_match == 0 {
            return;
        }
        let end = self.tokens.len();
        self.hash = self
            .hash
            .wrapping_mul(HASH_BASE)
            .wrapping_add(u64::from(token) + 1);
        if end > self.min_match {
            let dropped = u64::from(self.tokens[end - 1 - self.min_match]) + 1;
            self.hash = self
                .hash
                .wrapping_sub(dropped.wrapping_mul(self.drop_factor));
        }
        if end >= self.min_match {
            let previous = self
                .newest
                .insert(mix(self.hash), end as u32)
                .unwrap_or(NO_ANCHOR);
            self.previous.push(previous);
        }
    }

    /// Ends of the anchors equal to `anchor` (whose hash key is `key`),
    /// newest first.
    fn ends<'a>(&'a self, key: u64, anchor: &'a [u32]) -> impl Iterator<Item = usize> + 'a {
        let first = self.newest.get(&key).copied().unwrap_or(NO_ANCHOR);
        std::iter::successors((first != NO_ANCHOR).then_some(first), |&end| {
            let previous = self.previous[end as usize - self.min_match];
            (previous != NO_ANCHOR).then_some(previous)
        })
        .map(|end| end as usize)
        .filter(move |&end| &self.tokens[end - self.min_match..end] == anchor)
    }
}

/// Incremental exact matcher. A fixed-size anchor finds candidate occurrences
/// in O(1) expected time; candidates are extended backwards to obtain the exact
/// longest suffix. This avoids rescanning a 100K-token prompt each decode step.
pub struct SuffixSession {
    /// Remembered requests, newest first, shared with the store.
    stored: Vec<Arc<Anchors>>,
    /// The prompt and every token appended since.
    current: Anchors,
}

impl Default for SuffixSession {
    /// A session over nothing, which drafts nothing and costs nothing to make.
    fn default() -> Self {
        Self {
            stored: Vec::new(),
            current: Anchors::new(Vec::new(), 0),
        }
    }
}

impl SuffixSession {
    pub fn append(&mut self, token: u32) {
        self.current.push(token);
    }

    pub fn history(&self) -> &[u32] {
        &self.current.tokens
    }

    pub fn find(&self, max_drafts: usize) -> Option<LookupDraft> {
        let min_match = self.current.min_match;
        let current = &self.current.tokens;
        if min_match == 0 || max_drafts == 0 || current.len() < min_match {
            return None;
        }
        let anchor = &current[current.len() - min_match..];
        let key = mix(self.current.hash);
        let mut best_match = 0;
        let mut candidates = Vec::new();
        // Newest first: the current sequence, then remembered requests from
        // the most recent.
        let locations = std::iter::once((true, &self.current))
            .chain(self.stored.iter().map(|stored| (false, &**stored)))
            .flat_map(|(own, sequence)| {
                sequence
                    .ends(key, anchor)
                    .map(move |end| (own, &sequence.tokens, end))
            });
        for (own, source, end) in locations {
            if end >= source.len() || (own && end == current.len()) {
                continue;
            }
            let available = end.min(current.len());
            let mut matched = min_match;
            while matched < available
                && source[end - matched - 1] == current[current.len() - matched - 1]
            {
                matched += 1;
            }
            // Do not copy tokens from the suffix currently being matched.
            let mut continuation_end = (end + max_drafts).min(source.len());
            if own {
                continuation_end = continuation_end.min(current.len().saturating_sub(matched));
            }
            // A long self-referential match can pull `continuation_end` below
            // `end`; the continuation must be dropped, never sliced backwards.
            if continuation_end <= end {
                continue;
            }
            // Image tokens are prompt-only: a draft never proposes one.
            if let Some(image) = source[end..continuation_end]
                .iter()
                .position(|&token| crate::bonsai_vision::is_image_token(token))
            {
                continuation_end = end + image;
                if image == 0 {
                    continue;
                }
            }
            if matched > best_match {
                best_match = matched;
                candidates.clear();
            }
            if matched == best_match {
                candidates.push(&source[end..continuation_end]);
            }
        }
        // Frequency wins, then continuation length, then the oldest
        // occurrence (the last one found).
        let mut counts: HashMap<&[u32], usize> = HashMap::new();
        for &candidate in &candidates {
            *counts.entry(candidate).or_default() += 1;
        }
        let (_, tokens) = candidates
            .iter()
            .enumerate()
            .max_by_key(|&(index, candidate)| {
                (
                    counts.get(candidate).copied().unwrap_or(0),
                    candidate.len(),
                    index,
                )
            })?;
        Some(LookupDraft {
            occurrences: counts.get(tokens).copied().unwrap_or(1),
            tokens: tokens.to_vec(),
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
