//! Residual-stream taps for a block drafter (`DFlash2`): copies of the target's
//! residual stream after five layers, for every committed position.
//!
//! The residual stream is in the standard Qwen basis (only linear inputs are
//! Hadamard-rotated). Tap `i` of a position is the residual after the
//! zero-based layer `TAP_LAYERS[i]`: the sum the next layer's input norm
//! reads, as the validated drafter trace captured it. Taps are off, and cost
//! nothing, until [`BonsaiModel::set_taps`] turns them on.

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
        });
        Ok(())
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
        let skip = taps.floor.saturating_sub(self.position).min(rows);
        Ok(self.ops.copy_rows(
            batch,
            (&self.scratch.hidden, skip * WIDTH * size_of::<f32>()),
            (&taps.ring, column),
            WIDTH,
            rows - skip,
            [WIDTH, TAP_ROW_FLOATS],
            (self.position + skip) % TAP_WINDOW,
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
