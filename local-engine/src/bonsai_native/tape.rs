//! A sequence's GDN tape: the rows a verify step kept, folded into the
//! recurrent state at the start of the sequence's next step instead of
//! replayed after the verify.
//!
//! A verify step of at most [`TAPE_ROWS`] rows records every row's raw QKV
//! projection row, decay and beta in the tape half the pending rows are not
//! in, and leaves the recurrent state and history at the start of the step
//! (with the earlier pending rows folded in). Its commit only marks the kept
//! rows pending: no GPU work. The sequence's next decode, verify, tree or
//! batched step folds them in with its own first kernels
//! ([`local_metal::bonsai_ops::TapeFold`]); anything else that reads the
//! state (a long prefill block, a checkpoint, a snapshot) first replays them
//! ([`BonsaiModel::encode_settle`]). Both produce bitwise the state the
//! replay after the verify produced.

use std::cell::Cell;

use local_metal::bonsai_ops::{TAPE_ROWS as METAL_TAPE_ROWS, TapeFold};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;

use super::{AttentionLayer, BonsaiModel, CommandBatch};

/// Most rows a step records on the tape; a longer verify replays its kept
/// rows after the step, as before.
pub(super) const TAPE_ROWS: usize = METAL_TAPE_ROWS as usize;
const QKV: usize = 10_240;
const HEADS: usize = 48;

/// One recurrent layer's rows of a tape half.
pub(super) struct TapeLayer {
    /// Raw QKV projection rows.
    pub(super) raw: MetalBuffer,
    pub(super) decay: MetalBuffer,
    pub(super) beta: MetalBuffer,
}

/// A sequence's tape: two halves, so a step records its rows in one while it
/// folds the pending rows of the other.
pub struct Tape {
    halves: [Vec<TapeLayer>; 2],
    /// The half holding the pending rows.
    current: Cell<usize>,
    /// Rows of the current half the state does not hold yet.
    pending: Cell<usize>,
}

impl Tape {
    pub(super) fn new(context: &MetalContext, layers: usize) -> crate::Result<Self> {
        let floats = |width: usize| {
            MetalBuffer::empty(context.device(), TAPE_ROWS * width * size_of::<f32>())
        };
        let half = || {
            (0..layers)
                .map(|_| {
                    Ok(TapeLayer {
                        raw: floats(QKV)?,
                        decay: floats(HEADS)?,
                        beta: floats(HEADS)?,
                    })
                })
                .collect::<crate::Result<Vec<_>>>()
        };
        Ok(Self {
            halves: [half()?, half()?],
            current: Cell::new(0),
            pending: Cell::new(0),
        })
    }

    /// Bytes of a tape for `layers` recurrent layers.
    pub(crate) const fn bytes(layers: usize) -> usize {
        2 * layers * TAPE_ROWS * (QKV + 2 * HEADS) * size_of::<f32>()
    }

    /// Rows the state does not hold yet.
    pub(super) const fn pending(&self) -> usize {
        self.pending.get()
    }

    /// Recurrent layer `layer`'s pending rows, which a step's convolution
    /// leaves convolved at `convolved` for its recurrence.
    pub(super) fn fold<'a>(
        &'a self,
        layer: usize,
        convolved: (&'a MetalBuffer, usize),
    ) -> TapeFold<'a> {
        let pending = &self.halves[self.current.get()][layer];
        TapeFold {
            raw: &pending.raw,
            decay: &pending.decay,
            beta: &pending.beta,
            rows: self.pending.get() as u32,
            convolved,
        }
    }

    /// Recurrent layer `layer` of the half a step records its rows in.
    pub(super) fn step(&self, layer: usize) -> &TapeLayer {
        &self.halves[1 - self.current.get()][layer]
    }

    /// The state holds every pending row (a step folded them in, or it was
    /// replaced).
    pub(super) fn clear(&self) {
        self.pending.set(0);
    }

    /// The step that recorded its rows kept its first `kept`: they are
    /// pending.
    pub(super) fn keep(&self, kept: usize) {
        self.current.set(1 - self.current.get());
        self.pending.set(kept);
    }
}

impl BonsaiModel {
    /// Replay the resident sequence's pending tape rows into its recurrent
    /// state and history, into `batch`; the state then holds them.
    pub(super) fn encode_settle(&self, batch: &mut CommandBatch) -> crate::Result<()> {
        let rows = self.tape.pending() as u32;
        if rows == 0 {
            return Ok(());
        }
        let scratch = &self.scratch;
        let recurrent = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attention {
                AttentionLayer::Recurrent(recurrent) => Some(recurrent),
                AttentionLayer::Full(_) => None,
            });
        for (index, layer) in recurrent.enumerate() {
            let fold = self.tape.fold(index, (&scratch.convolved, 0));
            self.ops.conv_sequence(
                batch,
                fold.raw,
                &layer.convolution,
                &layer.history,
                &scratch.convolved,
                rows,
            )?;
            self.ops
                .l2_normalize_qk_rows(batch, &scratch.convolved, self.epsilon, rows)?;
            self.ops.gdn_sequence_into(
                batch,
                self.state_format,
                &scratch.convolved,
                fold.decay,
                fold.beta,
                &layer.state,
                &layer.state,
                &scratch.recurrent_output,
                rows,
            )?;
        }
        self.tape.clear();
        Ok(())
    }

    /// [`Self::encode_settle`] in a command buffer of its own, when rows are
    /// pending: before the state is read or copied.
    pub(super) fn settle_tape(&self) -> crate::Result<()> {
        if self.tape.pending() == 0 {
            return Ok(());
        }
        let mut batch = CommandBatch::new(&self.context)?;
        self.encode_settle(&mut batch)?;
        self.finish(batch)
    }
}
