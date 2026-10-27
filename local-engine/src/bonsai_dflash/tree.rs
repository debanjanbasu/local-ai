//! The host reference of the GPU tree builder, and the acceptance records a
//! verified tree yields.

use local_metal::dflash::{DFLASH_SLOTS, DflashCalibration, DflashLattice, DflashTree};

/// One row of a reference tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeNode {
    /// Parent row; -1 for the anchor.
    pub parent: i32,
    /// Positions after the anchor.
    pub depth: u32,
    /// Candidate index in lattice position `depth - 1`.
    pub slot: u32,
    pub token: u32,
    /// Conditional drafter log probability.
    pub log_q: f64,
    /// The path's calibrated log acceptance probability.
    pub path_log_p: f64,
}

/// Log of the calibrated acceptance probability `σ(α + β · logit q)` of a
/// node of log probability `lq`; `α = 0, β = 1` returns `lq` unchanged
/// (`p = q`). This is `sim.py`'s `calibrate` and the kernel's `df_lcal`.
#[must_use]
#[expect(clippy::float_cmp, reason = "the identity calibration is exact")]
pub fn calibrated(lq: f64, alpha: f32, beta: f32) -> f64 {
    if alpha == 0.0 && beta == 1.0 {
        return lq;
    }
    let q = lq.exp().min(0.999_999);
    let z = f64::from(beta).mul_add(lq - (1.0 - q).ln(), f64::from(alpha));
    if z > -30.0 { -(-z).exp().ln_1p() } else { z }
}

/// Best-first tree of up to `max_rows` rows over `lattice`, in F64.
///
/// The frontier node with the highest path score becomes the next row (ties: the
/// earlier frontier entry, as `sim.py`'s insertion counter) and its children
/// join the frontier. The GPU kernel builds the same tree in F32.
#[must_use]
pub fn build_tree_reference(
    lattice: &DflashLattice,
    anchor: u32,
    calibration: &DflashCalibration,
    max_rows: usize,
) -> Vec<TreeNode> {
    struct Entry {
        score: f64,
        log_q: f64,
        parent: usize,
        slot: usize,
        depth: usize,
        taken: bool,
    }
    let mut nodes = vec![TreeNode {
        parent: -1,
        depth: 0,
        slot: 0,
        token: anchor,
        log_q: 0.0,
        path_log_p: 0.0,
    }];
    let mut frontier: Vec<Entry> = Vec::new();
    let push = |frontier: &mut Vec<Entry>, parent: usize, score: f64, depth: usize, row: usize| {
        for (c, &lq) in lattice.cond_logq[depth - 1][row].iter().enumerate() {
            let lq = f64::from(lq);
            frontier.push(Entry {
                score: score + calibrated(lq, calibration.alpha[depth], calibration.beta[depth]),
                log_q: lq,
                parent,
                slot: c,
                depth,
                taken: false,
            });
        }
    };
    push(&mut frontier, 0, 0.0, 1, 0);
    while nodes.len() < max_rows {
        let Some(best) = frontier
            .iter()
            .enumerate()
            .filter(|(_, entry)| !entry.taken)
            .fold(None::<(usize, f64)>, |best, (index, entry)| match best {
                Some((_, score)) if entry.score <= score => best,
                _ => Some((index, entry.score)),
            })
            .map(|(index, _)| index)
        else {
            break;
        };
        frontier[best].taken = true;
        let entry = &frontier[best];
        let (depth, slot, score) = (entry.depth, entry.slot, entry.score);
        nodes.push(TreeNode {
            parent: i32::try_from(entry.parent).unwrap_or(-1),
            depth: depth as u32,
            slot: slot as u32,
            token: lattice.cand[depth - 1][slot],
            log_q: entry.log_q,
            path_log_p: score,
        });
        if depth < DFLASH_SLOTS {
            push(&mut frontier, nodes.len() - 1, score, depth + 1, slot);
        }
    }
    nodes
}

/// What one verified tree says about the drafter: the drafts accepted, and
/// one `(depth, log q, accepted)` record per tested node.
///
/// `rows` is how many of the tree's rows were verified; `target[i]` is the
/// target's own token after the `i`-th row of the accepted path (`target[0]`
/// after the anchor). Acceptance walks from the anchor into the child whose
/// token is the target's, for as long as there is one; a node is tested when
/// its parent was accepted.
#[must_use]
pub fn tested_nodes(
    tree: &DflashTree,
    rows: usize,
    target: &[u32],
) -> (usize, Vec<(usize, f32, bool)>) {
    let rows = rows.min(tree.rows as usize);
    let mut path = vec![0_usize];
    let mut records = Vec::new();
    loop {
        let depth = path.len() - 1;
        let Some(&want) = target.get(depth) else {
            break;
        };
        let parent = i32::try_from(path[depth]).unwrap_or(-1);
        let mut next = None;
        for row in 1..rows {
            if tree.parent[row] == parent {
                let hit = tree.token[row] == want;
                records.push((depth + 1, tree.log_q[row], hit));
                if hit && next.is_none() {
                    next = Some(row);
                }
            }
        }
        match next {
            Some(row) => path.push(row),
            None => break,
        }
    }
    (path.len() - 1, records)
}
