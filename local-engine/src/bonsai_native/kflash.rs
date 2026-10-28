//! `KFlash` drafting for the resident sequence: the `DFlash2` drafter's
//! context kept on the target's residual taps, one drafting pass per cycle
//! and the verified tree's commit and calibration.
//!
//! The drafter holds one context. Its rows come from the tap ring, which
//! holds the resident sequence's newest committed positions while they are
//! consecutive ([`BonsaiModel::tap_span`]). Appends are lazy: a cycle that
//! drafts first appends every position committed since the last append (by
//! any source), in the drafting pass's own command buffer, so cycles that
//! verify head or lookup drafts pay nothing and a run of them is appended in
//! a few 8-row chunks rather than one pass per cycle. When the span the
//! drafter continued ends (another sequence's rows, a batched pass, a
//! restored prefix), the drafter restarts at the oldest position the ring
//! still holds, at most a window back, and attends only what it has since;
//! unless that starts at position 0 it first waits for [`RESTART_ROWS`]
//! tapped positions.

use std::path::Path;
use std::time::Duration;

use local_metal::dflash::DFLASH_WINDOW;

use super::{BonsaiMetalTensor, BonsaiModel, CommandBatch, TAP_ROW_FLOATS, TAP_WINDOW};
use crate::bonsai_dflash::{
    Calibrator, DFLASH_MAX_TREE_ROWS, DflashDrafter, DflashPrecision, DflashShared, DflashTree,
    DraftAnchor, TapFormat, calibration_cache_path,
};
use crate::bonsai_native::tree::{DraftTree, MAX_TREE_ROWS};
use crate::sampler::{Sampler, SamplingResult};

/// A catch-up longer than this many rows runs in a command buffer of its
/// own before the cycle, so the cycle's measured time stays a cycle's.
pub const CATCH_UP_ROWS: usize = 64;
/// Positions a restarted drafter context needs before it drafts: with only
/// the few positions tapped since a batched pass it would draft from almost
/// no context, and its poor trees would leave the tree trailing in the
/// budget for a long stretch.
const RESTART_ROWS: usize = 64;
/// Weight of each new measurement in the cost averages.
const COST_WEIGHT: f64 = 0.125;
/// Verify cost repetitions at load, after a warm-up pass of each size: the
/// running drift corrects what two samples miss.
const COST_REPETITIONS: usize = 1;

/// Bytes `KFlash` allocates beside the drafter's mapped weights, for the
/// memory plan: the drafter's context ring and pass scratch (about 60 MB),
/// the tap ring and its verify staging rows (210 MB plus 100 KB per row of
/// the largest verify), and the tree verifier (about 105 MB).
pub(super) fn resident_bytes(verify_depth: usize) -> usize {
    const DRAFTER_STATE: usize = 64 << 20;
    let taps = (TAP_WINDOW + verify_depth.max(MAX_TREE_ROWS - 1) + 1) * TAP_ROW_FLOATS * 4;
    let tree = super::Verifier::checkpoint_bytes(MAX_TREE_ROWS - 1)
        + MAX_TREE_ROWS * super::VOCAB * size_of::<f32>();
    DRAFTER_STATE + taps + tree
}

/// The drafter, its calibration and the measured cycle costs.
pub(super) struct Kflash {
    /// Keeps `embedding`, which only the drafter reads on the GPU, resident
    /// beside the drafter's own set (see [`DflashDrafter`]); dropped
    /// first.
    _resident: Option<local_metal::context::ResidencySet>,
    drafter: DflashDrafter,
    calibrator: Calibrator,
    /// The target's `token_embd`, bound for the drafter's anchor gather.
    embedding: BonsaiMetalTensor,
    /// The tap span the drafter's context continues.
    epoch: Option<u64>,
    /// GPU ms of a verify of `n + 1` rows, measured at load.
    verify_ms: [f64; MAX_TREE_ROWS],
    /// How much each verified row has cost beyond `verify_ms` lately: the
    /// history the rows attend grows with the context.
    row_drift_ms: f64,
    /// GPU ms of a drafting pass with its appends, lately.
    draft_ms: f64,
}

/// What the policy reports about the loaded drafter.
#[derive(Clone, Debug)]
pub struct KflashInfo {
    pub path: std::path::PathBuf,
    pub payload_sha256: String,
    /// "int8" or "int4", how the drafter's matrices are stored.
    pub precision: &'static str,
    pub weight_bytes: u64,
    pub state_bytes: usize,
    pub draft_ms: f64,
    pub verify_ms: [f64; MAX_TREE_ROWS],
    pub calibration: Option<std::path::PathBuf>,
}

/// The cycle costs the budget prices candidates with.
#[derive(Clone, Copy, Debug)]
pub struct KflashCosts {
    /// GPU ms of a drafting pass, its appends included.
    pub draft_ms: f64,
    verify_ms: [f64; MAX_TREE_ROWS],
    row_drift_ms: f64,
}

impl KflashCosts {
    /// GPU ms of verifying `rows` rows (a tree up to eight, a chain beyond).
    #[must_use]
    pub fn verify_ms(&self, rows: usize) -> f64 {
        let rows = rows.max(1);
        let base = if rows <= MAX_TREE_ROWS {
            self.verify_ms[rows - 1]
        } else {
            // Past eight rows a pass costs about 9.4 ms per row.
            9.4f64.mul_add(
                (rows - MAX_TREE_ROWS) as f64,
                self.verify_ms[MAX_TREE_ROWS - 1],
            )
        };
        (rows as f64).mul_add(self.row_drift_ms, base)
    }
}

/// One verified `KFlash` cycle.
pub struct KflashCycle {
    /// The target's own tokens: the accepted drafts and the bonus token.
    pub samples: Vec<SamplingResult>,
    pub accepted: usize,
    /// Whether the accepted path left the tree's leading rows.
    pub off_chain: bool,
    pub verification: Duration,
    pub commit: Duration,
}

/// Rows of `tree` with depth at most `max_depth`, in tree order (parents
/// before children, so every prefix is a tree).
#[must_use]
pub fn tree_rows(tree: &DflashTree, max_depth: usize) -> Vec<usize> {
    (0..(tree.rows as usize).min(DFLASH_MAX_TREE_ROWS))
        .filter(|&row| tree.depth[row] as usize <= max_depth)
        .collect()
}

/// Tokens a verify of `rows` (a prefix of [`tree_rows`]) is expected to
/// emit: the target's own token plus each node's calibrated path
/// probability.
#[must_use]
pub fn expected_tokens(tree: &DflashTree, rows: &[usize]) -> f64 {
    1.0 + rows
        .iter()
        .skip(1)
        .map(|&row| f64::from(tree.path_log_p[row]).exp())
        .sum::<f64>()
}

/// `tree` cut to `rows` (a prefix of [`tree_rows`]), parents renumbered.
fn subtree(tree: &DflashTree, rows: &[usize]) -> DflashTree {
    let mut cut = *tree;
    cut.rows = rows.len() as u32;
    for (index, &row) in rows.iter().enumerate() {
        cut.token[index] = tree.token[row];
        cut.parent[index] = rows
            .iter()
            .position(|&kept| i32::try_from(kept).ok() == Some(tree.parent[row]))
            .and_then(|parent| i32::try_from(parent).ok())
            .unwrap_or(-1);
        cut.depth[index] = tree.depth[row];
        cut.slot[index] = tree.slot[row];
        cut.log_q[index] = tree.log_q[row];
        cut.path_log_p[index] = tree.path_log_p[row];
        cut.expected[index] = expected_tokens(tree, &rows[..=index]) as f32;
    }
    cut
}

impl BonsaiModel {
    /// Load the drafter at `path` for `KFlash` cycles: turn on the residual
    /// taps and tree verify, and measure the verify cost of 1 to 8 rows and
    /// the drafting pass on this machine. The calibration persists per
    /// `model` name, drafter and chip. Leaves the model reset.
    pub(crate) fn enable_kflash(&mut self, path: &Path, model: &str) -> crate::Result<()> {
        self.enable_tree_verify()?;
        self.set_taps(true)?;
        let shaders = local_metal::shaders::ShaderLibrary::new(self.context.device())?;
        let drafter = DflashDrafter::load(&self.context, &shaders, path)?;
        let calibrator =
            calibration_cache_path(model, drafter.payload_sha256(), &self.context.caps().name)
                .map_or_else(Calibrator::default, Calibrator::persistent);
        drafter.set_calibration(calibrator.calibration());
        self.info.state_bytes += drafter.state_bytes();
        let embedding = self
            .package
            .metal_tensor(self.context.device(), "token_embd.weight")?;
        self.reset();
        // The first pass of each shape also builds pipelines and pages
        // weights in: keep the fastest of every pass.
        let warm = self.tree_verify_costs(0, 1)?;
        let mut costs = self.tree_verify_costs(0, COST_REPETITIONS)?;
        for (cost, warm) in costs.iter_mut().zip(warm) {
            *cost = (*cost).min(warm);
        }
        self.reset();
        let resident = self.context.keep_resident([embedding.buffer()]);
        let mut kflash = Kflash {
            drafter,
            calibrator,
            embedding,
            epoch: None,
            verify_ms: costs.map(|cost| cost.as_secs_f64() * 1e3),
            row_drift_ms: 0.0,
            draft_ms: 0.0,
            _resident: resident,
        };
        let draft = {
            let Kflash {
                drafter, embedding, ..
            } = &mut kflash;
            let shared = DflashShared {
                kernels: &self.kernels,
                embedding: embedding.ptq1_matrix()?,
                output: self.output.ptq1_matrix()?,
                rotation: &self.input_rotation,
            };
            let draft = drafter.measure_draft_time(&self.context, &shared, 3)?;
            drafter.reset();
            draft
        };
        // One appended row rides with every drafting pass.
        kflash.draft_ms = draft.as_secs_f64().mul_add(1e3, 0.83);
        self.take_gpu_time();
        self.kflash = Some(kflash);
        Ok(())
    }

    /// The loaded drafter and its measured costs, for the policy record.
    pub(crate) fn kflash_info(&self) -> Option<KflashInfo> {
        let kflash = self.kflash.as_ref()?;
        Some(KflashInfo {
            path: kflash.drafter.path().to_owned(),
            payload_sha256: kflash.drafter.payload_sha256().to_owned(),
            precision: match kflash.drafter.precision() {
                DflashPrecision::Int8 => "int8",
                DflashPrecision::Int4 => "int4",
            },
            weight_bytes: kflash.drafter.weight_bytes(),
            state_bytes: kflash.drafter.state_bytes(),
            draft_ms: kflash.draft_ms,
            verify_ms: kflash.verify_ms,
            calibration: kflash.calibrator.path().map(Path::to_owned),
        })
    }

    /// The current cycle costs, while `KFlash` is loaded.
    pub(crate) fn kflash_costs(&self) -> Option<KflashCosts> {
        self.kflash.as_ref().map(|kflash| KflashCosts {
            draft_ms: kflash.draft_ms,
            verify_ms: kflash.verify_ms,
            row_drift_ms: kflash.row_drift_ms,
        })
    }

    /// Whether the resident sequence can draft now: the drafter is loaded,
    /// the ring holds its newest committed positions, and the drafter's
    /// context continues them, would start at position 0, or would restart
    /// with at least [`RESTART_ROWS`] of them. A context restarted after
    /// decoding without taps (a batched pass, a restored prefix) waits until
    /// the head's or lookup's committed rows have tapped that many again.
    pub(crate) fn kflash_ready(&self) -> bool {
        let (Some(kflash), Some(span)) = (self.kflash.as_ref(), self.tap_span()) else {
            return false;
        };
        let start = Self::restart_position(span);
        Self::continues(kflash, span) || start == 0 || span.end - start >= RESTART_ROWS
    }

    /// Positions the next drafting pass must append first.
    pub(crate) fn kflash_pending(&self) -> Option<usize> {
        let kflash = self.kflash.as_ref()?;
        let span = self.tap_span()?;
        let len = kflash.drafter.context_len();
        Some(if Self::continues(kflash, span) {
            span.end - len
        } else {
            span.end - Self::restart_position(span)
        })
    }

    fn continues(kflash: &Kflash, span: super::taps::TapSpan) -> bool {
        let len = kflash.drafter.context_len();
        kflash.epoch == Some(span.epoch) && (span.start..=span.end).contains(&len)
    }

    /// Where a restarted context starts: the oldest tapped position, at most
    /// the window back.
    fn restart_position(span: super::taps::TapSpan) -> usize {
        span.start.max(span.end.saturating_sub(DFLASH_WINDOW - 1))
    }

    /// Continue the drafter's context from `prefix` positions it already
    /// holds: the resident sequence restored a prompt-cache prefix whose
    /// tokens are the first `prefix` of the drafter's context, and the ring
    /// holds what was prefilled after it. Returns whether it could; if not,
    /// the next draft restarts the context at the first tapped position.
    pub(crate) fn kflash_adopt(&mut self, prefix: usize) -> bool {
        let Some(span) = self.tap_span() else {
            return false;
        };
        let Some(kflash) = self.kflash.as_mut() else {
            return false;
        };
        if span.start != prefix || kflash.drafter.rewind_to(prefix).is_err() {
            return false;
        }
        kflash.epoch = Some(span.epoch);
        true
    }

    /// Append the positions the drafter lacks, in their own command buffer,
    /// when there are more than `threshold`.
    pub(crate) fn kflash_catch_up(&mut self, threshold: usize) -> crate::Result<()> {
        if self
            .kflash_pending()
            .is_none_or(|pending| pending <= threshold)
        {
            return Ok(());
        }
        let Some(mut kflash) = self.kflash.take() else {
            return Ok(());
        };
        let result = (|| {
            let mut batch = CommandBatch::new(&self.context)?;
            self.encode_sync(&mut kflash, &mut batch)?;
            self.finish(batch)
        })();
        self.kflash = Some(kflash);
        result
    }

    /// Encode the appends that bring the drafter's context to
    /// [`Self::position`]; `false` when the ring holds no usable span.
    fn encode_sync(&self, kflash: &mut Kflash, batch: &mut CommandBatch) -> crate::Result<bool> {
        let (Some(span), Some(ring)) = (self.tap_span(), self.tap_ring()) else {
            return Ok(false);
        };
        if !Self::continues(kflash, span) {
            match Self::restart_position(span) {
                0 => kflash.drafter.reset(),
                start => kflash.drafter.restart_at(start),
            }
            kflash.epoch = Some(span.epoch);
        }
        let Kflash {
            drafter, embedding, ..
        } = kflash;
        let shared = DflashShared {
            kernels: &self.kernels,
            embedding: embedding.ptq1_matrix()?,
            output: self.output.ptq1_matrix()?,
            rotation: &self.input_rotation,
        };
        let row_bytes = TAP_ROW_FLOATS * size_of::<f32>();
        while drafter.context_len() < span.end {
            let position = drafter.context_len();
            let row = position % TAP_WINDOW;
            let rows = (span.end - position).min(TAP_WINDOW - row);
            drafter.append_context(batch, &shared, ring, row * row_bytes, rows, TapFormat::F32)?;
        }
        Ok(true)
    }

    /// Draft at `anchor`, the next token at [`Self::position`]: append what
    /// the drafter lacks, run the drafting pass and build the 8-row tree in
    /// one command buffer, and return the tree. `None` when the request was
    /// cancelled or the ring holds no usable span.
    pub(crate) fn kflash_draft(&mut self, anchor: u32) -> crate::Result<Option<DflashTree>> {
        if self.stop_for_cancel() {
            return Ok(None);
        }
        let Some(mut kflash) = self.kflash.take() else {
            return Ok(None);
        };
        let result = self.draft_tree(&mut kflash, anchor);
        self.kflash = Some(kflash);
        result
    }

    fn draft_tree(&self, kflash: &mut Kflash, anchor: u32) -> crate::Result<Option<DflashTree>> {
        let mut batch = CommandBatch::new(&self.context)?;
        if !self.encode_sync(kflash, &mut batch)? {
            return Ok(None);
        }
        let Kflash {
            drafter, embedding, ..
        } = &mut *kflash;
        let shared = DflashShared {
            kernels: &self.kernels,
            embedding: embedding.ptq1_matrix()?,
            output: self.output.ptq1_matrix()?,
            rotation: &self.input_rotation,
        };
        drafter.draft(
            &mut batch,
            &shared,
            &DraftAnchor::Token(anchor),
            self.position,
        )?;
        drafter.build_tree(&mut batch, DFLASH_MAX_TREE_ROWS)?;
        let gpu = batch.commit_async().wait_with_gpu_time()?;
        self.gpu_time.set(self.gpu_time.get().saturating_add(gpu));
        let ms = gpu.as_secs_f64() * 1e3;
        kflash.draft_ms = COST_WEIGHT.mul_add(ms - kflash.draft_ms, kflash.draft_ms);
        Ok(Some(kflash.drafter.read_tree()))
    }

    /// Verify `rows` of `tree` (a prefix of [`tree_rows`], anchored at
    /// [`Self::position`]), commit the longest path the target agrees with,
    /// and learn from it: the calibration from the nodes the target tested,
    /// the verify cost from the pass's GPU time. `None` when the request was
    /// cancelled before the verify was submitted.
    pub(crate) fn kflash_verify(
        &mut self,
        tree: &DflashTree,
        rows: &[usize],
        sampler: &mut Sampler,
    ) -> crate::Result<Option<KflashCycle>> {
        if self.stop_for_cancel() {
            return Ok(None);
        }
        let cut = subtree(tree, rows);
        let n = rows.len();
        let draft = DraftTree::from_parents(&cut.token[..n], &cut.parent[..n])?;
        let before = self.take_gpu_time();
        let started = std::time::Instant::now();
        self.tree_verify(&draft)?;
        let verify_gpu = self.take_gpu_time();
        let acceptance = self.tree_accept(sampler)?;
        let verification = started.elapsed();
        let committed = std::time::Instant::now();
        self.tree_commit(&acceptance.path)?;
        let commit = committed.elapsed();
        self.gpu_time.set(
            self.gpu_time
                .get()
                .saturating_add(before)
                .saturating_add(verify_gpu),
        );
        if let Some(kflash) = self.kflash.as_mut() {
            let excess = verify_gpu
                .as_secs_f64()
                .mul_add(1e3, -kflash.verify_ms[n - 1])
                / n as f64;
            kflash.row_drift_ms =
                COST_WEIGHT.mul_add(excess - kflash.row_drift_ms, kflash.row_drift_ms);
            let target = acceptance
                .samples
                .iter()
                .map(|sample| sample.token_id)
                .collect::<Vec<_>>();
            let (_, refit) = kflash.calibrator.record_cycle(&cut, n, &target);
            if refit {
                kflash
                    .drafter
                    .set_calibration(kflash.calibrator.calibration());
            }
        }
        Ok(Some(KflashCycle {
            accepted: acceptance.path.len() - 1,
            off_chain: acceptance
                .path
                .iter()
                .enumerate()
                .any(|(index, &row)| index != row),
            samples: acceptance.samples,
            verification,
            commit,
        }))
    }
}
