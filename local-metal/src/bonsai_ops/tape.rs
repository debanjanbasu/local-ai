//! The GDN tape: a verify step's kept rows folded into its sequence's
//! recurrent state at the start of the sequence's next step, instead of a
//! replay pass after the verify (`bo_*_fold` in `bonsai_ops.metal`).
//!
//! A step that runs `tokens` rows and keeps fewer leaves the kept rows' raw
//! QKV projection rows, decay and beta on the sequence's tape and the state
//! and history as they were before them. The next step's convolution kernel
//! convolves the pending rows from the committed history (which then holds
//! the window they leave) into a scratch block, keys normalized, and its
//! recurrence kernel runs them through each value row's state before its own
//! rows, storing the folded state once. That is the replay's arithmetic, so
//! the state is bitwise the replayed one.

use super::TREE_DESCRIPTOR_WORDS;
use super::{
    BonsaiOps, CommandBatch, GDN_HEADS, GdnStateFormat, MetalBuffer, arg, need, no_alias,
    sequence_bytes,
};

const QKV: usize = 10_240;
const MIXED: usize = 6144;

/// Most rows a tape holds (the largest step it records).
pub const TAPE_ROWS: u32 = 8;

/// Indexes in `BonsaiOps::p` of the fold pipelines (appended to its names).
const CONV_FOLD: usize = 37;
const CONV_TREE_FOLD: usize = 38;

impl GdnStateFormat {
    /// Indexes of the one-row, four-row and tree fold pipelines.
    const fn fold_pipelines(self) -> (usize, usize, usize) {
        match self {
            Self::F32 => (39, 40, 41),
            Self::F16 => (42, 43, 44),
            Self::Bf16 => (45, 46, 47),
        }
    }
}

/// One layer of a sequence's tape whose `rows` pending rows a step folds in
/// first.
#[derive(Clone, Copy)]
pub struct TapeFold<'a> {
    /// The pending rows' raw QKV projection rows (10,240 F32 each).
    pub raw: &'a MetalBuffer,
    /// Their decay and beta (48 F32 each).
    pub decay: &'a MetalBuffer,
    pub beta: &'a MetalBuffer,
    /// Pending rows, at most [`TAPE_ROWS`]; 0 folds nothing.
    pub rows: u32,
    /// Where the convolution leaves the pending rows for the recurrence:
    /// room for `rows` rows of 10,240 F32 from this byte offset.
    pub convolved: (&'a MetalBuffer, usize),
}

impl TapeFold<'_> {
    fn check(&self) -> crate::Result<()> {
        if self.rows > TAPE_ROWS {
            return Err(arg("a tape folds at most 8 rows"));
        }
        let rows = self.rows.max(1);
        need(self.raw, sequence_bytes(rows, QKV as u32)?)?;
        need(self.decay, sequence_bytes(rows, GDN_HEADS)?)?;
        need(self.beta, sequence_bytes(rows, GDN_HEADS)?)?;
        need(
            self.convolved.0,
            self.convolved.1 + sequence_bytes(rows, QKV as u32)?,
        )?;
        no_alias(self.convolved.0, &[self.raw, self.decay, self.beta])
    }
}

/// A buffer and byte offset.
pub type At<'a> = (&'a MetalBuffer, usize);

/// Where [`BonsaiOps::conv_l2_decay_fold`] reads a step's rows and leaves
/// what its recurrence reads.
#[derive(Clone, Copy)]
pub struct ConvRows<'a> {
    /// The step's raw QKV rows.
    pub input: At<'a>,
    /// Their convolved, normalized rows.
    pub out: At<'a>,
    /// The layer's `[a, alpha, dt, raw beta]`: `a` and `dt` per head, alpha
    /// and raw beta per row.
    pub decay_inputs: [At<'a>; 4],
    /// Each row's decay and beta.
    pub decay: At<'a>,
    pub beta: At<'a>,
    /// Where the step's raw rows are copied too (a batched sequence's tape),
    /// if anywhere.
    pub raw_copy: Option<&'a MetalBuffer>,
}

impl BonsaiOps {
    /// [`Self::conv_l2_decay_into`] over `tokens` rows after folding
    /// `fold`'s pending rows into `history` in place: the step reads the
    /// history its pending rows leave and writes its own final history to
    /// `final_history` (which may be `history`).
    pub fn conv_l2_decay_fold(
        &self,
        b: &mut CommandBatch,
        weights: &MetalBuffer,
        histories: [&MetalBuffer; 2],
        rows: ConvRows<'_>,
        epsilon: f32,
        fold: TapeFold<'_>,
        tokens: u32,
    ) -> crate::Result<()> {
        let [history, final_history] = histories;
        let [a, alpha, dt, beta_raw] = rows.decay_inputs;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(arg("invalid L2 epsilon"));
        }
        fold.check()?;
        let wide = sequence_bytes(tokens, QKV as u32)?;
        let scalar = sequence_bytes(tokens, GDN_HEADS)?;
        need(rows.input.0, rows.input.1 + wide)?;
        need(rows.out.0, rows.out.1 + wide)?;
        need(weights, QKV * 4 * 4)?;
        need(history, QKV * 3 * 4)?;
        need(final_history, QKV * 3 * 4)?;
        for v in [a, dt] {
            need(v.0, v.1 + GDN_HEADS as usize * 4)?;
        }
        for v in [alpha, beta_raw, rows.decay, rows.beta] {
            need(v.0, v.1 + scalar)?;
        }
        if let Some(copy) = rows.raw_copy {
            need(copy, wide)?;
            no_alias(copy, &[rows.input.0, history, rows.out.0, fold.raw])?;
        }
        let inputs = [rows.input.0, weights, a.0, alpha.0, dt.0, beta_raw.0];
        no_alias(rows.out.0, &[history, final_history, fold.raw])?;
        no_alias(rows.out.0, &inputs)?;
        for history in [history, final_history] {
            no_alias(history, &inputs)?;
            no_alias(history, &[rows.decay.0, rows.beta.0, fold.raw])?;
        }
        no_alias(rows.decay.0, &inputs)?;
        no_alias(rows.beta.0, &inputs)?;
        let copy = rows.raw_copy.unwrap_or(rows.out.0);
        self.go(
            b,
            CONV_FOLD,
            &[
                rows.input,
                (weights, 0),
                (history, 0),
                rows.out,
                a,
                alpha,
                dt,
                beta_raw,
                rows.decay,
                rows.beta,
                (final_history, 0),
                (fold.raw, 0),
                fold.convolved,
                (copy, 0),
            ],
            &[tokens, fold.rows, u32::from(rows.raw_copy.is_some())],
            &[epsilon],
            QKV / 128 + 1,
            128,
        );
        Ok(())
    }

    /// [`Self::conv_l2_decay_tree`] after folding `fold`'s pending rows into
    /// `history` in place.
    pub fn conv_l2_decay_tree_fold(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        history: &MetalBuffer,
        out: &MetalBuffer,
        epsilon: f32,
        decay_inputs: [&MetalBuffer; 4],
        outputs: [&MetalBuffer; 2],
        tree: &MetalBuffer,
        fold: TapeFold<'_>,
        tokens: u32,
    ) -> crate::Result<()> {
        let [a, alpha, dt, beta_raw] = decay_inputs;
        let [decay, beta] = outputs;
        if !(1..=8).contains(&tokens) || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(arg("invalid tree rows or L2 epsilon"));
        }
        fold.check()?;
        let bytes = sequence_bytes(tokens, QKV as u32)?;
        need(input, bytes)?;
        need(weights, QKV * 4 * 4)?;
        need(history, QKV * 3 * 4)?;
        need(out, bytes)?;
        need(tree, TREE_DESCRIPTOR_WORDS * 4)?;
        for v in [a, dt] {
            need(v, 48 * 4)?;
        }
        for v in [alpha, beta_raw, decay, beta] {
            need(v, sequence_bytes(tokens, 48)?)?;
        }
        let inputs = [input, weights, a, alpha, dt, beta_raw, tree, fold.raw];
        for written in [out, history, decay, beta] {
            no_alias(written, &inputs)?;
        }
        no_alias(out, &[history, decay, beta])?;
        no_alias(history, &[decay, beta])?;
        self.go(
            b,
            CONV_TREE_FOLD,
            &[
                (input, 0),
                (weights, 0),
                (history, 0),
                (out, 0),
                (a, 0),
                (alpha, 0),
                (dt, 0),
                (beta_raw, 0),
                (decay, 0),
                (beta, 0),
                (tree, 0),
                (fold.raw, 0),
                fold.convolved,
            ],
            &[tokens, fold.rows],
            &[epsilon],
            QKV / 128 + 1,
            128,
        );
        Ok(())
    }

    /// [`Self::gdn_sequence_into`] over `tokens` rows (`qkv`, `decay` and
    /// `beta` at their byte offsets) after folding `fold`'s pending rows into
    /// `state` in place: the step starts from the folded state and writes its
    /// final state to `final_state` (which may be `state`).
    pub fn gdn_fold(
        &self,
        b: &mut CommandBatch,
        format: GdnStateFormat,
        inputs: [At<'_>; 3],
        states: [&MetalBuffer; 2],
        out: At<'_>,
        fold: TapeFold<'_>,
        tokens: u32,
    ) -> crate::Result<()> {
        let [qkv, decay, beta] = inputs;
        let [state, final_state] = states;
        fold.check()?;
        need(qkv.0, qkv.1 + sequence_bytes(tokens, QKV as u32)?)?;
        need(decay.0, decay.1 + sequence_bytes(tokens, GDN_HEADS)?)?;
        need(beta.0, beta.1 + sequence_bytes(tokens, GDN_HEADS)?)?;
        need(state, format.state_bytes())?;
        need(final_state, format.state_bytes())?;
        need(out.0, out.1 + sequence_bytes(tokens, MIXED as u32)?)?;
        let read = [
            qkv.0,
            decay.0,
            beta.0,
            fold.convolved.0,
            fold.decay,
            fold.beta,
        ];
        no_alias(out.0, &read)?;
        no_alias(out.0, &[state, final_state])?;
        no_alias(state, &read)?;
        no_alias(final_state, &read)?;
        let (single, four, _) = format.fold_pipelines();
        let (pipeline, threads) = if tokens > 1 {
            (four, 32)
        } else {
            (single, 128)
        };
        self.go(
            b,
            pipeline,
            &[
                qkv,
                decay,
                beta,
                (state, 0),
                out,
                (final_state, 0),
                fold.convolved,
                (fold.decay, 0),
                (fold.beta, 0),
            ],
            &[tokens, fold.rows],
            &[],
            48 * 128 / 4,
            threads,
        );
        Ok(())
    }

    /// [`Self::gdn_tree`] after folding `fold`'s pending rows into `state` in
    /// place; the tree's rows start from the folded state.
    pub fn gdn_tree_fold(
        &self,
        b: &mut CommandBatch,
        format: GdnStateFormat,
        qkv: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        out: &MetalBuffer,
        tree: &MetalBuffer,
        fold: TapeFold<'_>,
        tokens: u32,
    ) -> crate::Result<()> {
        if !(1..=8).contains(&tokens) {
            return Err(arg("a tree verify holds 1..=8 rows"));
        }
        fold.check()?;
        need(qkv, sequence_bytes(tokens, QKV as u32)?)?;
        need(decay, sequence_bytes(tokens, GDN_HEADS)?)?;
        need(beta, sequence_bytes(tokens, GDN_HEADS)?)?;
        need(state, format.state_bytes())?;
        need(out, sequence_bytes(tokens, MIXED as u32)?)?;
        need(tree, TREE_DESCRIPTOR_WORDS * 4)?;
        let read = [
            qkv,
            decay,
            beta,
            tree,
            fold.convolved.0,
            fold.decay,
            fold.beta,
        ];
        no_alias(out, &read)?;
        no_alias(out, &[state])?;
        no_alias(state, &read)?;
        let (_, _, pipeline) = format.fold_pipelines();
        self.go(
            b,
            pipeline,
            &[
                (qkv, 0),
                (decay, 0),
                (beta, 0),
                (state, 0),
                (out, 0),
                (tree, 0),
                fold.convolved,
                (fold.decay, 0),
                (fold.beta, 0),
            ],
            &[tokens, fold.rows],
            &[],
            48 * 128 / 4,
            32,
        );
        Ok(())
    }
}
