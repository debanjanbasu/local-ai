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

/// A batched MTP draft round (`batched_head_draft_timings`, M4 Pro) runs one
/// stacked head pass per draft depth for every drafting sequence together:
/// a pass costs about 2.1 ms plus 0.5 ms per drafting sequence (2.5 ms for
/// one sequence, 6.1 ms for eight, against 3.0 and 23.5 ms drafting each
/// alone in turn). The fixed part is paid once per depth the deepest chosen
/// chain reaches; the per-sequence part goes into each candidate's
/// `draft_ms`.
pub(super) const HEAD_PASS_MS: f64 = 2.1;

/// What one sequence adds to each head pass of a batched draft round.
pub(super) const HEAD_ROW_MS: f64 = 0.5;

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
    /// Drafted by the sequence's MTP head: each draft also costs
    /// [`HEAD_ROW_MS`], and the step pays [`HEAD_PASS_MS`] for every depth
    /// of its deepest head chain, shared by every head-drafting sequence.
    pub(super) head: bool,
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

/// Expected tokens per millisecond of a step that verifies `plan[i]` drafts
/// of candidate `i`, or `None` past `max_rows` stacked rows.
fn rate(
    sequences: usize,
    rows: usize,
    candidates: &[Option<Candidate>],
    plan: &[usize],
    max_rows: usize,
) -> Option<f64> {
    let rows_before = rows;
    let mut tokens = sequences as f64;
    let mut rows = rows.max(sequences);
    let mut time = 0.0;
    let mut head_depth = 0;
    for (candidate, &depth) in candidates.iter().zip(plan) {
        let Some(candidate) = candidate.filter(|_| depth > 0) else {
            continue;
        };
        tokens += expected(candidate.acceptance, depth);
        rows += depth;
        time += candidate.draft_ms;
        if candidate.head {
            time = (depth as f64).mul_add(HEAD_ROW_MS, time);
            head_depth = head_depth.max(depth);
        }
    }
    // A prompt chunk in the pass would have cost its own pass: the decoding
    // rows and drafts are charged only what they add to it.
    let chunk = rows_before.saturating_sub(sequences);
    let prefill = if chunk > 0 { pass_ms(chunk) } else { 0.0 };
    (rows <= max_rows)
        .then(|| tokens / (head_depth as f64).mul_add(HEAD_PASS_MS, pass_ms(rows) - prefill + time))
}

/// Drafts to verify per sequence (0 for none) for a step of `sequences`
/// sequences, each contributing one row plus its chosen drafts, beside
/// `rows - sequences` rows of a prompt chunk, within `max_rows` stacked rows.
///
/// Pass time is not linear in rows (rows five to eight cost little more
/// than four, the ninth starts another group), so a draft that does not pay
/// alone may pay beside another's. Candidates are added in order of expected
/// tokens per row, each at its best depth whether or not it pays yet, and
/// the best prefix kept; then each choice is revised given the others.
pub(super) fn choose(
    sequences: usize,
    rows: usize,
    candidates: &[Option<Candidate>],
    max_rows: usize,
) -> Vec<usize> {
    let rate = |plan: &[usize]| rate(sequences, rows, candidates, plan, max_rows);
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
    // The depth of `index` that maximizes the rate, from `least`.
    let best_depth = |plan: &mut Vec<usize>, index: usize, depth: usize, least: usize| {
        let mut best = None;
        for candidate in least..=depth {
            plan[index] = candidate;
            if let Some(value) = rate(plan)
                && best.is_none_or(|(_, best_value)| value > best_value)
            {
                best = Some((candidate, value));
            }
        }
        plan[index] = best.map_or(0, |(candidate, _)| candidate);
        best.map(|(_, value)| value)
    };
    let mut plan = vec![0; candidates.len()];
    let mut best = (rate(&plan).unwrap_or(0.0), plan.clone());
    for &(index, candidate) in &order {
        if let Some(value) = best_depth(&mut plan, index, candidate.depth, 1)
            && value > best.0
        {
            best = (value, plan.clone());
        }
    }
    let mut plan = best.1;
    for _ in 0..2 {
        for &(index, candidate) in &order {
            best_depth(&mut plan, index, candidate.depth, 0);
        }
    }
    plan
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
            head: false,
        };
        let chosen = choose(5, 5, &[Some(confident), None, None, None, None], 64);
        assert_eq!(chosen[0], 3);
        // A drafted chain that is rarely accepted is not worth its rows.
        let doubtful = Candidate {
            depth: 3,
            acceptance: 0.1,
            draft_ms: COMMIT_MS,
            head: true,
        };
        assert_eq!(choose(2, 2, &[Some(doubtful), None], 64), vec![0, 0]);
    }

    #[test]
    fn head_drafts_share_their_passes() {
        // Two prose streams whose heads are accepted 65% of the time: the
        // head passes are shared, so both chains pay.
        let head = Candidate {
            depth: 3,
            acceptance: 0.65,
            draft_ms: COMMIT_MS,
            head: true,
        };
        let chosen = choose(2, 2, &[Some(head), Some(head)], 64);
        assert!(chosen.iter().all(|&depth| depth > 0), "{chosen:?}");
        // Eight streams fill their eight-row group with seeds: drafts would
        // double the pass.
        assert_eq!(choose(8, 8, &[Some(head); 8], 64), vec![0; 8]);
        // Beside a prompt chunk the pass is long and rows cost their full
        // price; uncertain drafts are not worth them.
        let doubtful = Candidate {
            acceptance: 0.3,
            ..head
        };
        assert_eq!(choose(2, 40, &[Some(doubtful); 2], 64), vec![0, 0]);
    }

    #[test]
    fn chosen_rows_respect_the_limit() {
        let long = Candidate {
            depth: 40,
            acceptance: 0.99,
            draft_ms: 0.0,
            head: false,
        };
        let chosen = choose(4, 4, &[Some(long), Some(long), None, None], 64);
        assert!(4 + chosen.iter().sum::<usize>() <= 64);
    }
}
