//! Residual-stream taps for a block drafter (`DFlash2`): copies of the target's
//! residual stream after five layers, for every committed position.
//!
//! The residual stream is in the standard Qwen basis (only linear inputs are
//! Hadamard-rotated). Tap `i` of a position is the residual after the
//! zero-based layer `TAP_LAYERS[i]`: the sum the next layer's input norm
//! reads, as the validated drafter trace captured it. Taps are off, and cost
//! nothing, until [`BonsaiModel::set_taps`] turns them on.

use std::cell::Cell;

use super::{BonsaiModel, CommandBatch, MetalBuffer, WIDTH};

/// Target layers whose output the drafter reads, in tap order.
pub const TAP_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];
/// Committed positions the tap ring holds; prefill taps only its last
/// `TAP_WINDOW` positions.
pub const TAP_WINDOW: usize = 2048;
/// Floats of one position's taps: the five residual rows, in tap order.
pub const TAP_ROW_FLOATS: usize = TAP_LAYERS.len() * WIDTH;

/// The tap ring and the staging rows of a verify pass awaiting its commit.
pub(super) struct Taps {
    /// `[TAP_WINDOW][TAP_LAYERS.len()][WIDTH]` F32: position `p` in row
    /// `p % TAP_WINDOW`.
    ring: MetalBuffer,
    /// The last verify pass's rows, laid out as ring rows, until the commit
    /// copies the accepted ones to their positions.
    staging: MetalBuffer,
    staging_rows: usize,
    /// Prefill leaves positions below this untapped.
    pub(super) floor: usize,
    /// The consecutive positions the ring holds, if any.
    span: Cell<Option<TapSpan>>,
    /// Identity of the next span.
    next_epoch: Cell<u64>,
}

/// Consecutive committed positions `start..end` of one buffer set whose taps
/// the ring holds (only the last [`TAP_WINDOW`] are still there).
///
/// A write that does not continue the span (another buffer set, or a jump
/// in position: rows decoded without taps, a restored prefix) starts a new
/// span with a new `epoch`, so a reader that recorded the epoch it read up
/// to knows whether the rows it has not read yet continue what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TapSpan {
    pub epoch: u64,
    state: u64,
    pub start: usize,
    pub end: usize,
}

impl Taps {
    /// Record that the ring now holds `start..end` of buffer set `state`.
    fn note(&self, state: u64, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let span = match self.span.get() {
            Some(span) if span.state == state && span.end == start => TapSpan { end, ..span },
            _ => {
                let epoch = self.next_epoch.get();
                self.next_epoch.set(epoch + 1);
                TapSpan {
                    epoch,
                    state,
                    start,
                    end,
                }
            }
        };
        self.span.set(Some(span));
    }
}

/// The tap slot that layer `layer`'s input norm fills (the residual after
/// the layer before it), if any.
fn slot(layer: usize) -> Option<usize> {
    TAP_LAYERS.iter().position(|&tapped| tapped + 1 == layer)
}

impl BonsaiModel {
    /// Turn the residual taps on (allocating the 210 MB ring and the verify
    /// staging rows) or off (freeing them). Prefill, plain decode steps and
    /// committed verify rows of the resident sequence write their positions
    /// from then on; batched passes do not.
    pub fn set_taps(&mut self, enabled: bool) -> crate::Result<()> {
        if !enabled {
            if let Some(taps) = self.taps.take() {
                self.info.state_bytes -= taps.ring.length() + taps.staging.length();
            }
            return Ok(());
        }
        if self.taps.is_some() {
            return Ok(());
        }
        let staging_rows = self
            .speculation
            .iter()
            .map(|speculation| speculation.verifier.max_rows)
            .chain(self.ngram_verifier.iter().map(|verifier| verifier.max_rows))
            .chain(self.tree.iter().map(|tree| tree.verifier.max_rows))
            .max()
            .unwrap_or(1);
        let floats = |rows: usize| rows * TAP_ROW_FLOATS * size_of::<f32>();
        self.info.state_bytes += floats(TAP_WINDOW) + floats(staging_rows);
        self.taps = Some(Taps {
            ring: MetalBuffer::empty(self.context.device(), floats(TAP_WINDOW))?,
            staging: MetalBuffer::empty(self.context.device(), floats(staging_rows))?,
            staging_rows,
            floor: 0,
            span: Cell::new(None),
            next_epoch: Cell::new(0),
        });
        Ok(())
    }

    /// The consecutive committed positions of the resident sequence whose
    /// taps the ring holds, ending at [`Self::position`]: `None` while taps
    /// are off or when the newest committed positions were not tapped
    /// (decoded in a batched pass, or restored from a prompt checkpoint).
    /// `start` is clamped to the ring's last [`TAP_WINDOW`] positions.
    #[must_use]
    pub fn tap_span(&self) -> Option<TapSpan> {
        self.tap_span_of(self.state_id, self.position)
    }

    /// [`Self::tap_span`] of buffer set `state` at `position`: a parked
    /// sequence whose batched rows the ring holds.
    #[must_use]
    pub fn tap_span_of(&self, state: u64, position: usize) -> Option<TapSpan> {
        let span = self.taps.as_ref()?.span.get()?;
        (span.state == state && span.end == position).then(|| TapSpan {
            start: span.start.max(span.end.saturating_sub(TAP_WINDOW)),
            ..span
        })
    }

    /// Copy the `rows` residual rows from stacked row `first` of
    /// `scratch.hidden`, at the input norm of layer `layer`, to the ring at
    /// positions from `position`: one batched sequence's rows, before its
    /// commit settles how many it keeps ([`Self::note_batch_taps`]).
    pub(super) fn encode_batch_taps(
        &self,
        batch: &mut CommandBatch,
        layer: usize,
        first: usize,
        rows: usize,
        position: usize,
    ) -> crate::Result<()> {
        let (Some(taps), Some(slot)) = (&self.taps, slot(layer)) else {
            return Ok(());
        };
        Ok(self.ops.copy_rows(
            batch,
            (&self.scratch.hidden, first * WIDTH * size_of::<f32>()),
            (&taps.ring, slot * WIDTH * size_of::<f32>()),
            WIDTH,
            rows,
            [WIDTH, TAP_ROW_FLOATS],
            position % TAP_WINDOW,
            TAP_WINDOW,
        )?)
    }

    /// Record that a batched pass left the taps of positions `start..kept`
    /// of buffer set `state` in the ring, having written rows up to
    /// `written` (rejected drafts past `kept`, which overwrote the ring's
    /// oldest rows).
    pub(super) fn note_batch_taps(&self, state: u64, start: usize, kept: usize, written: usize) {
        let Some(taps) = &self.taps else {
            return;
        };
        taps.note(state, start, kept);
        if let Some(span) = taps.span.get() {
            taps.span.set(Some(TapSpan {
                start: span.start.max(written.saturating_sub(TAP_WINDOW)),
                ..span
            }));
        }
    }

    /// Forget which positions the ring holds: the committed state was
    /// replaced (a reset or a restored prefix), so the rows describe
    /// positions of some other text.
    pub(super) fn forget_taps(&self) {
        if let Some(taps) = &self.taps {
            taps.span.set(None);
        }
    }

    /// Keep only the newest `TAP_WINDOW - 1` positions of the current span:
    /// a row was written at the position just past it and then dropped (the
    /// cost table's decode steps), overwriting the oldest row of the window.
    pub(super) fn trim_taps(&self, span: Option<TapSpan>) {
        if let Some(taps) = &self.taps {
            taps.span.set(span.map(|span| TapSpan {
                start: span.start.max((span.end + 1).saturating_sub(TAP_WINDOW)),
                ..span
            }));
        }
    }

    /// The tap ring, while taps are on: `[TAP_WINDOW][TAP_LAYERS.len()][WIDTH]`
    /// F32, committed position `p` in row `p % TAP_WINDOW`. Rows of positions
    /// at or past [`Self::position`] are not committed yet.
    #[must_use]
    pub fn tap_ring(&self) -> Option<&MetalBuffer> {
        self.taps.as_ref().map(|taps| &taps.ring)
    }

    /// A host copy of the taps of the `count` committed positions from
    /// `position`, each [`TAP_ROW_FLOATS`] floats; `None` while taps are off
    /// or when the ring no longer (or does not yet) hold them all.
    #[expect(unsafe_code)]
    #[must_use]
    pub fn tap_rows(&self, position: usize, count: usize) -> Option<Vec<f32>> {
        let taps = self.taps.as_ref()?;
        let end = position.checked_add(count)?;
        if end > self.position || self.position - position > TAP_WINDOW {
            return None;
        }
        // SAFETY: every block waits for its GPU work before returning, and
        // `&self` keeps the model from encoding while this copy is made.
        let ring = unsafe { taps.ring.as_slice::<f32>() };
        let mut rows = Vec::with_capacity(count * TAP_ROW_FLOATS);
        for position in position..end {
            let start = position % TAP_WINDOW * TAP_ROW_FLOATS;
            rows.extend_from_slice(&ring[start..start + TAP_ROW_FLOATS]);
        }
        Some(rows)
    }

    /// Copy the residual rows `scratch.hidden` holds at the input norm of
    /// layer `layer` (the residual after the layer before it) to their tap,
    /// when that layer is tapped: the `rows` rows of a verify pass to the
    /// staging rows (`staged`), otherwise to the ring at their positions from
    /// [`Self::position`], skipping those below the prefill floor.
    pub(super) fn encode_taps(
        &self,
        batch: &mut CommandBatch,
        layer: usize,
        rows: usize,
        staged: bool,
    ) -> crate::Result<()> {
        let (Some(taps), Some(slot)) = (&self.taps, slot(layer)) else {
            return Ok(());
        };
        let column = slot * WIDTH * size_of::<f32>();
        if staged {
            if rows > taps.staging_rows {
                return Err(crate::Error::InvalidArgument(
                    "verify block exceeds the tap staging rows".into(),
                ));
            }
            return Ok(self.ops.copy_rows(
                batch,
                (&self.scratch.hidden, 0),
                (&taps.staging, column),
                WIDTH,
                rows,
                [WIDTH, TAP_ROW_FLOATS],
                0,
                taps.staging_rows,
            )?);
        }
        self.encode_taps_at(batch, layer, rows, &self.scratch.hidden, self.position)
    }

    /// [`Self::encode_taps`] for `rows` committed rows held in `source` from
    /// `position`: a Neural Engine wavefront half, whose residual stream and
    /// first position are its own. Halves must be tapped in position order,
    /// so the ring's span grows contiguously.
    pub(super) fn encode_taps_at(
        &self,
        batch: &mut CommandBatch,
        layer: usize,
        rows: usize,
        source: &MetalBuffer,
        position: usize,
    ) -> crate::Result<()> {
        let (Some(taps), Some(slot)) = (&self.taps, slot(layer)) else {
            return Ok(());
        };
        let column = slot * WIDTH * size_of::<f32>();
        let skip = taps.floor.saturating_sub(position).min(rows);
        if slot == 0 {
            taps.note(self.state_id, position + skip, position + rows);
        }
        Ok(self.ops.copy_rows(
            batch,
            (source, skip * WIDTH * size_of::<f32>()),
            (&taps.ring, column),
            WIDTH,
            rows - skip,
            [WIDTH, TAP_ROW_FLOATS],
            (position + skip) % TAP_WINDOW,
            TAP_WINDOW,
        )?)
    }

    /// Copy the staged taps of verify rows `rows` (in commit order) to the
    /// ring positions from `start`: a commit's accepted rows.
    pub(super) fn encode_tap_commit(
        &self,
        batch: &mut CommandBatch,
        rows: &[usize],
        start: usize,
    ) -> crate::Result<()> {
        let Some(taps) = &self.taps else {
            return Ok(());
        };
        taps.note(self.state_id, start, start + rows.len());
        let mut index = 0;
        while index < rows.len() {
            // Consecutive staged rows go to consecutive positions in one copy.
            let mut run = 1;
            while index + run < rows.len() && rows[index + run] == rows[index] + run {
                run += 1;
            }
            if rows[index] + run > taps.staging_rows {
                return Err(crate::Error::InvalidArgument(
                    "committed rows exceed the tap staging rows".into(),
                ));
            }
            self.ops.copy_rows(
                batch,
                (
                    &taps.staging,
                    rows[index] * TAP_ROW_FLOATS * size_of::<f32>(),
                ),
                (&taps.ring, 0),
                TAP_ROW_FLOATS,
                run,
                [TAP_ROW_FLOATS, TAP_ROW_FLOATS],
                (start + index) % TAP_WINDOW,
                TAP_WINDOW,
            )?;
            index += run;
        }
        Ok(())
    }
}
