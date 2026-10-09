//! Which sequences of a batched step verify drafts, by a cost model.
//!
//! A batched pass costs roughly what its stacked rows cost: one sequence's
//! drafts add rows to everybody's step. Drafts are worth their rows only when
//! the tokens they are expected to add raise the step's tokens per second.
//! [`choose`] adds candidates in order of expected tokens per added row while
//! each one raises `expected tokens / (pass time + drafting time)`.

/// Measured time of one batched pass by stacked rows (M4 Pro, every row
/// projected to logits, `batched_verify_timings` and `batched_step_timings`);
/// linear between points. Rows five to eight cost little more than four, and
/// from there a pass costs about 75 ms per eight rows.
const PASS_MS: [(usize, f64); 14] = [
    (1, 32.4),
    (2, 48.4),
    (3, 64.0),
    (4, 75.0),
    (5, 78.0),
    (6, 80.0),
    (7, 82.0),
    (8, 84.0),
    (10, 125.0),
    (12, 148.0),
    (16, 154.0),
    (24, 229.0),
    (32, 303.0),
    (64, 602.0),
];

/// Replaying one verifying sequence's kept rows after the pass.
pub(super) const COMMIT_MS: f64 = 1.5;

/// Time to catch a sequence's MTP head up and draft one chain with it.
pub(super) const MTP_DRAFT_MS: f64 = 15.0;

/// Milliseconds of one batched pass over `rows` stacked rows.
pub(super) fn pass_ms(rows: usize) -> f64 {
    let rows = rows.max(1);
    let mut previous = PASS_MS[0];
    for &(at, ms) in &PASS_MS {
        if rows <= at {
            if at == previous.0 {
                return ms;
            }
            let fraction = (rows - previous.0) as f64 / (at - previous.0) as f64;
            return fraction.mul_add(ms - previous.1, previous.1);
        }
        previous = (at, ms);
    }
    // Beyond the table, at the last segment's slope.
    let (last, ms) = PASS_MS[PASS_MS.len() - 1];
    let (before, before_ms) = PASS_MS[PASS_MS.len() - 2];
    ms + (rows - last) as f64 * (ms - before_ms) / (last - before) as f64
}

/// One sequence's possible drafts.
#[derive(Clone, Copy, Debug)]
pub(super) struct Candidate {
    /// Most drafts the sequence can verify this step.
    pub(super) depth: usize,
    /// Estimated chance each draft is accepted given the previous one was.
    pub(super) acceptance: f64,
    /// Time to produce the drafts, paid only if they are used.
    pub(super) draft_ms: f64,
}

/// Expected accepted drafts of a chain of `depth` drafts.
fn expected(acceptance: f64, depth: usize) -> f64 {
    let mut total = 0.0;
    let mut chance = 1.0;
    for _ in 0..depth {
        chance *= acceptance;
        total += chance;
    }
    total
}

/// Drafts to verify per sequence (0 for none) for a step of `sequences`
/// sequences, each contributing one row plus its chosen drafts, within
/// `max_rows` stacked rows.
pub(super) fn choose(
    sequences: usize,
    candidates: &[Option<Candidate>],
    max_rows: usize,
) -> Vec<usize> {
    let mut chosen = vec![0; candidates.len()];
    let mut tokens = sequences as f64;
    let mut rows = sequences;
    let mut extra = 0.0;
    let mut order = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| Some((index, (*candidate)?)))
        .filter(|(_, candidate)| candidate.depth > 0)
        .collect::<Vec<_>>();
    // Most expected tokens per row first.
    order.sort_by(|(_, a), (_, b)| {
        let a = expected(a.acceptance, a.depth) / a.depth as f64;
        let b = expected(b.acceptance, b.depth) / b.depth as f64;
        b.total_cmp(&a)
    });
    for (index, candidate) in order {
        let rate = tokens / (pass_ms(rows) + extra);
        let mut best = None;
        for depth in 1..=candidate.depth.min(max_rows.saturating_sub(rows)) {
            let gain = expected(candidate.acceptance, depth);
            let time = pass_ms(rows + depth) + extra + candidate.draft_ms;
            let candidate_rate = (tokens + gain) / time;
            if candidate_rate > best.map_or(rate, |(_, best_rate, _)| best_rate) {
                best = Some((depth, candidate_rate, gain));
            }
        }
        if let Some((depth, _, gain)) = best {
            chosen[index] = depth;
            tokens += gain;
            rows += depth;
            extra += candidate.draft_ms;
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_time_interpolates_and_extrapolates() {
        assert!((pass_ms(4) - 75.0).abs() < 1e-9);
        assert!(pass_ms(10) > pass_ms(8) && pass_ms(10) < pass_ms(16));
        assert!(pass_ms(80) > pass_ms(64));
    }

    #[test]
    fn confident_drafts_fill_cheap_rows_and_doubtful_ones_do_not() {
        // Five sequences: rows six to eight cost little, so confident
        // n-gram drafts of one sequence are taken.
        let confident = Candidate {
            depth: 3,
            acceptance: 0.9,
            draft_ms: 0.0,
        };
        let chosen = choose(5, &[Some(confident), None, None, None, None], 64);
        assert_eq!(chosen[0], 3);
        // A drafted chain that is rarely accepted is not worth its rows.
        let doubtful = Candidate {
            depth: 3,
            acceptance: 0.1,
            draft_ms: 15.0,
        };
        assert_eq!(choose(2, &[Some(doubtful), None], 64), vec![0, 0]);
    }

    #[test]
    fn chosen_rows_respect_the_limit() {
        let long = Candidate {
            depth: 40,
            acceptance: 0.99,
            draft_ms: 0.0,
        };
        let chosen = choose(4, &[Some(long), Some(long), None, None], 64);
        assert!(4 + chosen.iter().sum::<usize>() <= 64);
    }
}
