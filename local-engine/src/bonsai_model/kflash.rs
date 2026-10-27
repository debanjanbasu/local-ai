//! Which source drafts a single sequence's next cycle, and how many tree
//! rows it verifies.
//!
//! Three sources compete each cycle: the `DFlash2` tree (one drafting pass,
//! then a verify of `n` of its best-first rows), the MTP head's gated chain,
//! and suffix lookup when it has a match. Each one's tokens per millisecond
//! is measured on the cycles it runs (tokens emitted over the cycle's wall
//! time, both averaged), and the cycle goes to the source that leads. A
//! source that has not run yet runs first, and a trailing one runs again
//! after a number of cycles that grows with the cube of how far it
//! trails, so a change in the text (prose turning into code) is noticed
//! without paying for it on every cycle. The choice comes before any
//! drafting, so no drafting pass is ever thrown away.
//!
//! A tree's size follows Kapitsa's budget: the first `n` rows of the
//! best-first tree are the best `n`-row tree, and the cycle verifies the `n`
//! with the largest `E(n) − r · cost(n)`, where `E(n)` is the calibrated
//! expected tokens of those rows, `cost(n)` the measured verify time of `n`
//! rows at the current context, and `r` the recent tokens per millisecond:
//! a row goes in when its expected tokens beat what its milliseconds would
//! earn at the going rate. Before `r` exists the cycle maximizes
//! `E(n) / (draft + cost(n))`.

use crate::bonsai_dflash::DflashTree;
use crate::bonsai_native::{KflashCosts, expected_tokens};

/// A cycle's draft source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Source {
    Tree = 0,
    Mtp = 1,
    Lookup = 2,
}

const SOURCES: [Source; 3] = [Source::Lookup, Source::Tree, Source::Mtp];
/// Least weight of a new cycle in a source's averages: the first ten
/// cycles average evenly, later ones decay. A tree cycle emits 1 to 8 tokens,
/// so a heavier weight lets a few short cycles hand the lead to a slower
/// source.
const WEIGHT: f64 = 0.1;
/// Weight of each new cycle in the going rate.
const RATE_WEIGHT: f64 = 0.25;
/// Cycles a source sits out, at an equal rate, before it runs again (times
/// the cube of how far it trails). On an essay, where the head chain leads
/// the tree by about 1.3x, a tree every 14 cycles cost 3.7% of the decode
/// rate; every 36 costs about 1.5%.
const EXPLORE: [usize; 3] = [16, 32, 8];
/// The longest a trailing source sits out.
const MAX_EXPLORE: usize = 512;
/// Cycles before the going rate prices tree rows.
const RATE_CYCLES: usize = 4;

/// One sequence's measured rates.
#[derive(Clone, Debug, Default)]
pub(super) struct KflashBudget {
    tokens: [f64; 3],
    ms: [f64; 3],
    runs: [usize; 3],
    since: [usize; 3],
    rate_tokens: f64,
    rate_ms: f64,
    cycles: usize,
}

impl KflashBudget {
    fn rate(&self, source: Source) -> Option<f64> {
        let index = source as usize;
        (self.runs[index] > 0 && self.ms[index] > 0.0).then(|| self.tokens[index] / self.ms[index])
    }

    /// Recent tokens per millisecond over every source, once a few cycles ran.
    pub(super) fn going_rate(&self) -> Option<f64> {
        (self.cycles >= RATE_CYCLES && self.rate_ms > 0.0).then(|| self.rate_tokens / self.rate_ms)
    }

    /// The source for the next cycle among the `available` ones (indexed by
    /// [`Source`]), or `None` when none is.
    pub(super) fn choose(&self, available: [bool; 3]) -> Option<Source> {
        let candidates = SOURCES
            .into_iter()
            .filter(|&source| available[source as usize])
            .collect::<Vec<_>>();
        if let Some(&untried) = candidates
            .iter()
            .find(|&&source| self.rate(source).is_none())
        {
            return Some(untried);
        }
        let rate = |source: Source| self.rate(source).unwrap_or(0.0);
        let best = candidates
            .iter()
            .copied()
            .max_by(|&a, &b| rate(a).total_cmp(&rate(b)))?;
        let due = candidates.iter().copied().find(|&source| {
            let index = source as usize;
            let behind = rate(best) / rate(source).max(1e-9);
            let interval = (EXPLORE[index] as f64 * behind.powi(3)).min(MAX_EXPLORE as f64);
            source != best && self.since[index] as f64 >= interval
        });
        Some(due.unwrap_or(best))
    }

    /// Record a cycle of `source` that emitted `tokens` in `ms`.
    pub(super) fn observe(&mut self, source: Source, tokens: usize, ms: f64) {
        let index = source as usize;
        let tokens = tokens as f64;
        self.runs[index] += 1;
        let weight = (1.0 / self.runs[index] as f64).max(WEIGHT);
        self.tokens[index] = weight.mul_add(tokens - self.tokens[index], self.tokens[index]);
        self.ms[index] = weight.mul_add(ms - self.ms[index], self.ms[index]);
        for since in &mut self.since {
            *since += 1;
        }
        self.since[index] = 0;
        self.cycles += 1;
        let weight = (1.0 / self.cycles as f64).max(RATE_WEIGHT);
        self.rate_tokens = weight.mul_add(tokens - self.rate_tokens, self.rate_tokens);
        self.rate_ms = weight.mul_add(ms - self.rate_ms, self.rate_ms);
    }
}

/// How many of `rows` (a prefix-closed list of tree rows, anchor first) to
/// verify: the largest `E(n) − rate · cost(n)`, or before a going rate
/// exists the largest `E(n) / (draft + cost(n))`; ties to fewer rows.
pub(super) fn tree_size(
    tree: &DflashTree,
    rows: &[usize],
    costs: &KflashCosts,
    rate: Option<f64>,
) -> usize {
    let mut best = (1, f64::NEG_INFINITY);
    for n in 1..=rows.len() {
        let expected = expected_tokens(tree, &rows[..n]);
        let cost = costs.verify_ms(n);
        let score = rate.map_or_else(
            || expected / (costs.draft_ms + cost),
            |rate| rate.mul_add(-cost, expected),
        );
        if score > best.1 {
            best = (n, score);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untried_sources_run_first_then_the_fastest_with_exploration() {
        let mut budget = KflashBudget::default();
        let all = [true, true, false];
        assert_eq!(budget.choose(all), Some(Source::Tree));
        budget.observe(Source::Tree, 6, 100.0);
        assert_eq!(budget.choose(all), Some(Source::Mtp));
        budget.observe(Source::Mtp, 2, 60.0);
        // The tree leads (0.06 against 0.033 tokens per ms).
        for _ in 0..20 {
            assert_eq!(budget.choose(all), Some(Source::Tree));
            budget.observe(Source::Tree, 6, 100.0);
        }
        // Trailing by 1.8x, the head chain is due after 32 · 5.83 cycles.
        for _ in 21..190 {
            budget.observe(Source::Tree, 6, 100.0);
        }
        assert_eq!(budget.choose(all), Some(Source::Mtp));
        // Lookup, available at last and never tried, runs at once.
        assert_eq!(budget.choose([true, true, true]), Some(Source::Lookup));
        assert_eq!(budget.choose([false, false, false]), None);
    }

    #[test]
    fn rates_follow_recent_cycles() {
        let mut budget = KflashBudget::default();
        budget.observe(Source::Tree, 2, 100.0);
        budget.observe(Source::Mtp, 2, 50.0);
        assert_eq!(budget.choose([true, true, false]), Some(Source::Mtp));
        for _ in 0..8 {
            budget.observe(Source::Tree, 8, 100.0);
        }
        assert_eq!(budget.choose([true, true, false]), Some(Source::Tree));
        assert!(budget.going_rate().is_some_and(|rate| rate > 0.05));
    }
}
