//! Self-distillation capture for the MTP draft head.
//!
//! The head is trained offline (`tools/mtp_train`) on the target's own
//! behaviour, so everything here reproduces exactly what the runtime feeds it:
//!
//! - **hidden input**: the target's output-normalized final hidden
//!   (`scratch.normalized`, written by `normalize_input` with `output_norm`).
//!   The residual stream is kept in the unrotated (Qwen) basis — only the copy in
//!   `scratch.rotated_hidden` is Hadamard-rotated for the PTQ1 projections — so
//!   this is a plain Qwen-basis vector.
//! - **embedding input**: the decoded PTQ1 `token_embd` row, which the checkpoint
//!   stores rotated; the head applies the inverse rotation itself
//!   (`bonsai_mtp/head/step.rs`), so the trainer's table holds the
//!   inverse-rotated rows.
//! - **logits**: the head output is forward-rotated and multiplied by the PTQ1
//!   `output` matrix. The rotation is orthonormal, so `W · R x = (Rᵀ wᵣ) · x`
//!   and the dense Qwen-basis `lm_head` is the inverse rotation of every decoded
//!   output row.
//!
//! A capture shard follows the `CAPTURE_FORMAT.md` layout of
//! `xkm/qwen3.8-27b-mtp-head-retrained`: `[u32 magic][u32 count][u32 width][u32 k]`
//! then `count` records `{i32 token; k × i32 ids; k × f32 logits; width × f16 hidden}`.
//! Record `p` is the head row the runtime encodes at absolute position `p`: the
//! token at `p`, the target's top-k logits of row `p` (its argmax is the verify
//! accept criterion for the draft this row makes), and the normalized hidden of
//! row `p - 1` — zero for `p = 0`, exactly as `BonsaiMtp::reset` leaves it.
//!
//! [`MtpCapture::capture_features`] reads the same normalized hidden for
//! frozen-target probes, but unshifted: row `p` itself at explicitly selected
//! positions, the final token included.

use std::io::Write;
use std::path::Path;

use super::{
    BlockOutput, BonsaiModel, CommandBatch, KvOptions, MAX_PREFILL_TOKENS, PTQ1_BLOCK_BYTES,
    PTQ1_BLOCK_ELEMENTS, VOCAB, Verifier, WIDTH, decode_embeddings, decode_ptq1_row,
};
use crate::bonsai::{BonsaiPackage, BonsaiTensorType};
use crate::bonsai_mtp::MtpSettings;
use crate::bonsai_ngram::NgramSettings;

/// `MTPD`, little-endian, as the xkm shard reader expects.
pub const CAPTURE_MAGIC: u32 = 0x4D54_5044;
const HADAMARD_BLOCK: usize = 1024;

/// One draft step of the runtime head, for the parity check.
#[derive(Clone, Debug, serde::Serialize)]
pub struct DraftStep {
    /// Absolute position the head row is encoded at.
    pub position: usize,
    /// Token whose embedding the row fuses.
    pub input_token: u32,
    /// Top draft ids, best first, and their logits.
    pub top_ids: Vec<u32>,
    pub top_logits: Vec<f32>,
}

/// The target (and optionally the head) loaded for capture.
pub struct MtpCapture {
    model: BonsaiModel,
    verifier: Verifier,
    rows: usize,
}

impl MtpCapture {
    /// Maximum capture-block width accepted by [`Self::open`].
    pub const MAX_ROWS: usize = MAX_PREFILL_TOKENS as usize;

    /// Load the target for capture blocks of up to `rows` tokens and documents of
    /// up to `context` tokens. A `head` is only needed for [`Self::draft_chain`].
    pub fn open(
        model: &Path,
        head: Option<&MtpSettings>,
        rows: usize,
        context: usize,
    ) -> crate::Result<Self> {
        if !(3..=Self::MAX_ROWS).contains(&rows) {
            return Err(crate::Error::InvalidArgument(format!(
                "capture rows must be within 3..={MAX_PREFILL_TOKENS}"
            )));
        }
        let package = BonsaiPackage::open(model)?;
        let model = BonsaiModel::load(
            package,
            context,
            rows,
            None,
            head,
            NgramSettings {
                enabled: false,
                ..NgramSettings::default()
            },
            KvOptions::default(),
        )?;
        let verifier = Verifier::new(&model.context, model.state_format, rows - 1, rows - 1)?;
        Ok(Self {
            model,
            verifier,
            rows,
        })
    }

    /// Tokens one document may hold.
    pub const fn context(&self) -> usize {
        self.model.info.context
    }

    /// Run `tokens` through the target from an empty cache and write one shard
    /// of `tokens.len()` records to `out`.
    pub fn capture_document(
        &mut self,
        tokens: &[u32],
        top_k: usize,
        out: &mut impl Write,
    ) -> crate::Result<()> {
        if tokens.len() < 2 || tokens.len() > self.context() {
            return Err(crate::Error::InvalidArgument(
                "capture documents need 2..=context tokens".into(),
            ));
        }
        if top_k == 0 || top_k > 64 {
            return Err(crate::Error::InvalidArgument(
                "capture top-k must be within 1..=64".into(),
            ));
        }
        let count = u32::try_from(tokens.len())
            .map_err(|_| crate::Error::InvalidArgument("document too long".into()))?;
        let mut header = Vec::with_capacity(16);
        for value in [CAPTURE_MAGIC, count, WIDTH as u32, top_k as u32] {
            header.extend_from_slice(&value.to_le_bytes());
        }
        out.write_all(&header)?;
        self.model.reset();
        let mut previous = vec![0.0_f32; WIDTH];
        let record_bytes = 4 + top_k * 8 + WIDTH * 2;
        let mut start = 0;
        for take in block_sizes(tokens.len(), self.rows) {
            let block = &tokens[start..start + take];
            self.model
                .forward_block(block, BlockOutput::Verify(&self.verifier))?;
            self.model.commit_verified(&mut self.verifier, take, take)?;
            let logits = &self.verifier.verify_logits.as_slice::<f32>()[..take * VOCAB];
            let normalized = &self.model.scratch.normalized.as_slice::<f32>()[..take * WIDTH];
            let tops = top_k_rows(logits, top_k)?;
            let mut bytes = Vec::with_capacity(take * record_bytes);
            for (row, &token) in block.iter().enumerate() {
                bytes.extend_from_slice(&token.cast_signed().to_le_bytes());
                let (ids, values) = &tops[row];
                for &id in ids {
                    bytes.extend_from_slice(&id.cast_signed().to_le_bytes());
                }
                for &value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                for &value in &previous {
                    bytes.extend_from_slice(&half::f16::from_f32(value).to_le_bytes());
                }
                previous.copy_from_slice(&normalized[row * WIDTH..(row + 1) * WIDTH]);
            }
            out.write_all(&bytes)?;
            start += take;
        }
        Ok(())
    }

    /// Run `tokens` through the target from an empty cache and return the
    /// output-normalized, unrotated (Qwen-basis) hidden of each requested row:
    /// `positions.len() × WIDTH` `f32`s, position-major.
    ///
    /// Row `p` is the hidden *after* the token at zero-based index `p` — the
    /// one whose logits predict token `p + 1` — read from `scratch.normalized`
    /// on committed hidden-only blocks, without vocabulary projection or
    /// verification rollback. Unlike a shard record (which holds row `p - 1`),
    /// there is no shift and no dummy
    /// token: `tokens.len() - 1` selects the final token's own hidden.
    ///
    /// `tokens` must hold `2..=context` valid ids; `positions` must be non-empty,
    /// strictly increasing and below `tokens.len()`. Nothing is truncated.
    pub fn capture_features(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> crate::Result<Vec<f32>> {
        // The capture model's cancel token is never tripped.
        self.model
            .capture_hidden(tokens, positions, self.rows)?
            .ok_or_else(|| crate::Error::Generation("feature capture was cancelled".into()))
    }

    /// [`Self::capture_features`], written to `out` as consecutive
    /// little-endian FP16 vectors of `WIDTH` values, one per position in
    /// order. Nothing is written unless every row was captured and is finite
    /// in FP16.
    pub fn write_features(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        out: &mut impl Write,
    ) -> crate::Result<()> {
        let features = self.capture_features(tokens, positions)?;
        out.write_all(&features_f16_bytes(&features)?)?;
        Ok(())
    }

    /// Draft `depth` tokens with the runtime head after committing
    /// `tokens[..position]`, exactly as a speculative round does: step 0 fuses
    /// `tokens[position]` with the committed hidden, every deeper step fuses the
    /// previous step's argmax with the head's own normalized output.
    pub fn draft_chain(
        &mut self,
        tokens: &[u32],
        position: usize,
        depth: usize,
        top_k: usize,
    ) -> crate::Result<Vec<DraftStep>> {
        if position == 0 || position >= tokens.len() || depth == 0 || top_k == 0 {
            return Err(crate::Error::InvalidArgument(
                "draft chain needs 1 <= position < tokens and depth > 0".into(),
            ));
        }
        self.model.reset();
        self.model.prefill(&tokens[..position], &mut |_| {})?;
        let Some(mut speculation) = self.model.speculation.take() else {
            return Err(crate::Error::InvalidArgument(
                "draft chain needs an MTP head".into(),
            ));
        };
        let mut run = || -> crate::Result<Vec<DraftStep>> {
            self.model
                .reserve_kv(position + depth + 1, Some(&mut speculation))?;
            let mut steps = Vec::with_capacity(depth);
            let mut token = tokens[position];
            for step in 0..depth {
                decode_embeddings(
                    &self.model.package,
                    &[token],
                    speculation.mtp.embedding_rows(1)?,
                )?;
                let mtp = &speculation.mtp;
                let hidden = if step == 0 {
                    mtp.prev_hidden()
                } else {
                    mtp.predicted()
                };
                let mut batch = CommandBatch::new(&self.model.context)?;
                mtp.encode(
                    &mut batch,
                    &self.model.mtp_shared(),
                    hidden,
                    position + step,
                    1,
                    true,
                )?;
                self.model.finish(batch)?;
                let logits = &mtp.logits.as_slice::<f32>()[..VOCAB];
                let (top_ids, top_logits) = top_k_row(logits, top_k);
                steps.push(DraftStep {
                    position: position + step,
                    input_token: token,
                    top_ids: top_ids.clone(),
                    top_logits,
                });
                token = top_ids[0];
            }
            Ok(steps)
        };
        let result = run();
        self.model.speculation = Some(speculation);
        result
    }
}

impl BonsaiModel {
    /// The body of [`MtpCapture::capture_features`], on whichever sequence
    /// buffers are resident: reset them, run `tokens` in committed
    /// hidden-only blocks of at most `rows` (split by [`block_sizes`], so the
    /// same `rows` reproduces the same blocks and FP16 state rounding), and
    /// return the output-normalized, unrotated hidden of each of
    /// `positions`, `positions.len() × WIDTH` `f32`s, position-major.
    ///
    /// The installed cancel token is polled before every block; a trip
    /// returns `Ok(None)` and submits nothing further. Either way the
    /// resident buffers are left holding a partial capture, not any earlier
    /// sequence: the caller owns invalidating whatever described them.
    pub(crate) fn capture_hidden(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        rows: usize,
    ) -> crate::Result<Option<Vec<f32>>> {
        validate_feature_request(tokens, positions, self.info.context)?;
        if !(3..=self.block_rows).contains(&rows) {
            return Err(crate::Error::InvalidArgument(format!(
                "capture rows must be within 3..={}",
                self.block_rows
            )));
        }
        self.reset();
        let mut features = Vec::with_capacity(positions.len() * WIDTH);
        let mut next = 0;
        let mut start = 0;
        for take in block_sizes(tokens.len(), rows) {
            if next == positions.len() {
                // Causal: later blocks cannot change rows already read.
                break;
            }
            if self.stop_for_cancel() {
                self.cancel_observed = false;
                return Ok(None);
            }
            let block = &tokens[start..start + take];
            self.forward_block(block, BlockOutput::Hidden)?;
            let normalized = &self.scratch.normalized.as_slice::<f32>()[..take * WIDTH];
            next = select_rows(normalized, start, positions, next, &mut features);
            start += take;
        }
        if next != positions.len() || features.len() != positions.len() * WIDTH {
            return Err(crate::Error::Generation(
                "feature capture did not reach every requested position".into(),
            ));
        }
        Ok(Some(features))
    }
}

/// Block lengths covering `count` tokens with at most `rows` each and never a
/// single-row block, which the verify path rejects.
fn block_sizes(count: usize, rows: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut remaining = count;
    while remaining > 0 {
        let mut take = remaining.min(rows);
        if remaining - take == 1 {
            take -= 1;
        }
        sizes.push(take);
        remaining -= take;
    }
    sizes
}

/// Check a [`MtpCapture::capture_features`] request before any GPU work.
fn validate_feature_request(
    tokens: &[u32],
    positions: &[usize],
    context: usize,
) -> crate::Result<()> {
    if tokens.len() < 2 {
        return Err(crate::Error::InvalidArgument(
            "feature capture needs at least 2 tokens".into(),
        ));
    }
    if tokens.len() > context {
        return Err(crate::Error::ContextOverflow(format!(
            "feature capture has {} tokens, context is {context}",
            tokens.len()
        )));
    }
    if let Some(index) = tokens.iter().position(|&token| token as usize >= VOCAB) {
        return Err(crate::Error::InvalidArgument(format!(
            "token {} at index {index} is outside the {VOCAB}-entry vocabulary",
            tokens[index]
        )));
    }
    if positions.is_empty() {
        return Err(crate::Error::InvalidArgument(
            "feature capture needs at least one position".into(),
        ));
    }
    if let Some(&[before, after]) = positions
        .array_windows::<2>()
        .find(|[before, after]| before >= after)
    {
        return Err(crate::Error::InvalidArgument(format!(
            "feature positions must be strictly increasing: {before} then {after}"
        )));
    }
    let last = positions[positions.len() - 1];
    if last >= tokens.len() {
        return Err(crate::Error::InvalidArgument(format!(
            "feature position {last} is past the final token index {}",
            tokens.len() - 1
        )));
    }
    Ok(())
}

/// Append to `out` the rows of one block — `normalized` holds the block's
/// rows, the first at absolute position `start` — for `positions[next..]`
/// that fall inside it, returning the index of the first position past it.
/// `positions` is strictly increasing and `positions[next] >= start`.
fn select_rows(
    normalized: &[f32],
    start: usize,
    positions: &[usize],
    mut next: usize,
    out: &mut Vec<f32>,
) -> usize {
    let end = start + normalized.len() / WIDTH;
    while let Some(&position) = positions.get(next) {
        if position >= end {
            break;
        }
        let row = position - start;
        out.extend_from_slice(&normalized[row * WIDTH..(row + 1) * WIDTH]);
        next += 1;
    }
    next
}

/// Little-endian FP16 bytes of `values`, rejecting any value FP16 cannot
/// hold finitely rather than writing `inf`/`NaN` features.
fn features_f16_bytes(values: &[f32]) -> crate::Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for (index, &value) in values.iter().enumerate() {
        let half = half::f16::from_f32(value);
        if !half.is_finite() {
            return Err(crate::Error::Generation(format!(
                "feature value {value} at element {index} is not finite in FP16"
            )));
        }
        bytes.extend_from_slice(&half.to_le_bytes());
    }
    Ok(bytes)
}

/// Round `values` in place through FP16, exactly as [`features_f16_bytes`]
/// stores them for training, rejecting the same non-finite values.
pub(crate) fn round_features_f16(values: &mut [f32]) -> crate::Result<()> {
    for (index, value) in values.iter_mut().enumerate() {
        let half = half::f16::from_f32(*value);
        if !half.is_finite() {
            return Err(crate::Error::Generation(format!(
                "feature value {value} at element {index} is not finite in FP16"
            )));
        }
        *value = half.to_f32();
    }
    Ok(())
}

/// Highest `k` entries of one row, best first; ties keep the lower id, as the
/// greedy sampler does.
fn top_k_row(row: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (id, &value) in row.iter().enumerate() {
        if best.len() == k && value <= best[k - 1].0 {
            continue;
        }
        let at = best.partition_point(|&(other, _)| other >= value);
        best.insert(at, (value, id as u32));
        best.truncate(k);
    }
    best.into_iter().map(|(value, id)| (id, value)).unzip()
}

type TopK = (Vec<u32>, Vec<f32>);

fn top_k_rows(logits: &[f32], k: usize) -> crate::Result<Vec<TopK>> {
    let rows: Vec<&[f32]> = logits
        .as_chunks::<VOCAB>()
        .0
        .iter()
        .map(<[f32; VOCAB]>::as_slice)
        .collect();
    let threads = std::thread::available_parallelism()
        .map_or(4, std::num::NonZero::get)
        .min(rows.len().max(1));
    let per = rows.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(per.max(1))
            .map(|group| {
                scope.spawn(move || {
                    group
                        .iter()
                        .map(|row| top_k_row(row, k))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut tops = Vec::with_capacity(rows.len());
        for handle in handles {
            tops.extend(
                handle
                    .join()
                    .map_err(|_| crate::Error::Generation("top-k thread panicked".into()))?,
            );
        }
        Ok(tops)
    })
}

/// Undo the checkpoint's activation rotation on one row in place: the
/// normalized 1024-block Walsh-Hadamard transform, then the signs — the CPU
/// twin of `bonsai_fwht_inverse`.
fn inverse_rotate(row: &mut [f32], signs: &[f32]) {
    for block in row.as_chunks_mut::<HADAMARD_BLOCK>().0 {
        let mut half = 1;
        while half < HADAMARD_BLOCK {
            for pair in block.chunks_exact_mut(2 * half) {
                let (low, high) = pair.split_at_mut(half);
                for (a, b) in low.iter_mut().zip(high) {
                    let (x, y) = (*a, *b);
                    *a = x + y;
                    *b = x - y;
                }
            }
            half *= 2;
        }
    }
    let scale = 1.0 / (HADAMARD_BLOCK as f32).sqrt();
    for (value, sign) in row.iter_mut().zip(signs) {
        *value *= sign * scale;
    }
}

/// Input widths of the MTP head's matrices: the hidden width (fc halves,
/// q/k/v, gate/up), the attention output width (`o_proj`) and the MLP width
/// (`down_proj`).
pub const HEAD_SIGN_WIDTHS: [u32; 3] = [5120, 6144, 17408];

/// Write the checkpoint's Hadamard signs for each of `widths` as
/// `signs-{width}.bin` (raw `i8` ±1, `width` bytes), the rotation basis a
/// ternary head is trained in (`tools/mtp_train/ternary.py`).
///
/// Beside each goes `rotate-check-{width}.bin`: a deterministic `f32` test
/// vector `x` followed by `inverse_rotate(x)`, so the trainer can check its
/// rotation against this file's CPU twin of the Metal transform.
pub fn export_signs(
    model: &Path,
    directory: &Path,
    widths: &[u32],
) -> crate::Result<serde_json::Value> {
    let package = BonsaiPackage::open(model)?;
    std::fs::create_dir_all(directory)?;
    let mut written = Vec::new();
    for &width in widths {
        let signs = package.hadamard().signs(width)?;
        let codes = sign_codes(signs, width)?;
        let path = directory.join(format!("signs-{width}.bin"));
        let bytes: Vec<u8> = codes.iter().map(|&sign| sign.cast_unsigned()).collect();
        std::fs::write(&path, bytes)?;
        let input = rotate_check_input(width as usize);
        let mut rotated = input.clone();
        inverse_rotate(&mut rotated, signs);
        let bytes: Vec<u8> = input
            .iter()
            .chain(&rotated)
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(directory.join(format!("rotate-check-{width}.bin")), bytes)?;
        written.push(path.display().to_string());
    }
    Ok(serde_json::json!({
        "signs": written,
        "model": model.display().to_string(),
    }))
}

/// The ±1 signs as `i8`, rejecting a width the rotation cannot apply to or a
/// value that is not exactly ±1.
fn sign_codes(signs: &[f32], width: u32) -> crate::Result<Vec<i8>> {
    if signs.len() != width as usize || !signs.len().is_multiple_of(HADAMARD_BLOCK) {
        return Err(crate::Error::InvalidFormat(format!(
            "width {width}: {} signs, need a multiple of {HADAMARD_BLOCK} equal to the width",
            signs.len()
        )));
    }
    signs
        .iter()
        .map(|&sign| match sign {
            1.0 => Ok(1),
            -1.0 => Ok(-1),
            other => Err(crate::Error::InvalidFormat(format!(
                "width {width}: sign {other} is not ±1"
            ))),
        })
        .collect()
}

/// A deterministic, non-symmetric test vector for the rotation check.
fn rotate_check_input(width: usize) -> Vec<f32> {
    (0..width)
        .map(|i| (((i * 7919 + 13) % 1009) as f32 - 504.0) / 1009.0)
        .collect()
}

/// Write the frozen tables the trainer needs, in the head's unrotated basis.
///
/// `embed_tokens.safetensors` and `lm_head.safetensors` hold one F16 `weight`
/// tensor of `[VOCAB, WIDTH]` each; `meta.json` records the norm epsilon and
/// `RoPE` base the runtime head uses.
pub fn export_head_tables(model: &Path, directory: &Path) -> crate::Result<serde_json::Value> {
    let package = BonsaiPackage::open(model)?;
    let signs = package.hadamard().signs(WIDTH as u32)?.to_vec();
    std::fs::create_dir_all(directory)?;
    let row_bytes = WIDTH / PTQ1_BLOCK_ELEMENTS * PTQ1_BLOCK_BYTES;
    let mut files = Vec::new();
    for (tensor, file) in [
        ("token_embd.weight", "embed_tokens.safetensors"),
        ("output.weight", "lm_head.safetensors"),
    ] {
        let info = package.tensor(tensor)?;
        let packed = package.bytes(tensor)?;
        if !matches!(info.tensor_type(), BonsaiTensorType::Ptq1)
            || packed.len() != VOCAB * row_bytes
        {
            return Err(crate::Error::InvalidFormat(format!(
                "{tensor} is not a [{VOCAB}, {WIDTH}] PTQ1 matrix"
            )));
        }
        let path = directory.join(file);
        write_rotated_table(&path, packed, row_bytes, &signs)?;
        files.push(path.display().to_string());
    }
    let meta = serde_json::json!({
        "basis": "unrotated Qwen basis: rows are inverse_hadamard(decoded PTQ1 row)",
        "embed_tokens": files[0],
        "lm_head": files[1],
        "vocab": VOCAB,
        "hidden": WIDTH,
        "rms_norm_eps": package.metadata_f32("qwen35.attention.layer_norm_rms_epsilon")?,
        "rope_theta": package.metadata_f32("qwen35.rope.freq_base")?,
        "model": model.display().to_string(),
    });
    std::fs::write(
        directory.join("meta.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    Ok(meta)
}

fn write_rotated_table(
    path: &Path,
    packed: &[u8],
    row_bytes: usize,
    signs: &[f32],
) -> crate::Result<()> {
    let data_bytes = VOCAB * WIDTH * 2;
    let mut header = serde_json::json!({
        "weight": {"dtype": "F16", "shape": [VOCAB, WIDTH], "data_offsets": [0, data_bytes]}
    })
    .to_string();
    while !(header.len() + 8).is_multiple_of(8) {
        header.push(' ');
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(&(header.len() as u64).to_le_bytes())?;
    out.write_all(header.as_bytes())?;
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    let chunk_rows = 4096;
    for chunk in packed.chunks(chunk_rows * row_bytes) {
        let per = chunk.len() / row_bytes;
        let per_thread = per.div_ceil(threads) * row_bytes;
        // Every worker is spawned before any is joined; joining lazily would
        // run them one at a time.
        #[allow(clippy::needless_collect)]
        let parts = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .chunks(per_thread.max(row_bytes))
                .map(|part| {
                    scope.spawn(move || -> crate::Result<Vec<u8>> {
                        let mut bytes = Vec::with_capacity(part.len() / row_bytes * WIDTH * 2);
                        let mut row = vec![0.0_f32; WIDTH];
                        for packed_row in part.chunks_exact(row_bytes) {
                            decode_ptq1_row(packed_row, &mut row)?;
                            inverse_rotate(&mut row, signs);
                            for &value in &row {
                                bytes.extend_from_slice(&half::f16::from_f32(value).to_le_bytes());
                            }
                        }
                        Ok(bytes)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().map_err(|_| {
                        crate::Error::Generation("table export thread panicked".into())
                    })?
                })
                .collect::<crate::Result<Vec<_>>>()
        })?;
        for part in parts {
            out.write_all(&part)?;
        }
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        VOCAB, WIDTH, block_sizes, features_f16_bytes, inverse_rotate, select_rows, sign_codes,
        top_k_row, validate_feature_request,
    };

    #[test]
    #[ignore = "requires the pinned model; selected features must match MTP capture row p+1"]
    fn selected_features_match_model_capture_including_final_token() -> crate::Result<()> {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
        ));
        let mut capture = super::MtpCapture::open(path, None, 4, 64)?;
        let tokens = [7, 1503, 4095, 83, 700, 38, 127, 512];
        let positions = [0, 3, 4, 7];
        let features = capture.capture_features(&tokens, &positions)?;
        let mut extended = tokens.to_vec();
        extended.push(19);
        let mut shard = Vec::new();
        capture.capture_document(&extended, 1, &mut shard)?;
        let record_bytes = 12 + WIDTH * 2;
        for (row, &position) in positions.iter().enumerate() {
            // MTP record p+1 stores hidden p; unlike selected capture its final
            // hidden needs an appended token. Appending also changes the last
            // block's size, so allow its existing FP16 state rounding error.
            let offset = 16 + (position + 1) * record_bytes + 12;
            let expected: Vec<f32> = shard[offset..offset + WIDTH * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| half::f16::from_le_bytes([pair[0], pair[1]]).to_f32())
                .collect();
            let actual = &features[row * WIDTH..(row + 1) * WIDTH];
            let squared_error: f64 = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum();
            let energy: f64 = expected.iter().map(|&v| f64::from(v).powi(2)).sum();
            assert!(energy > 0.0);
            let relative_rms = (squared_error / energy).sqrt();
            assert!(
                relative_rms < 0.003,
                "position {position}: relative RMS {relative_rms}"
            );
        }
        Ok(())
    }

    /// Feature selection over the real block split reads row `p` itself — not
    /// `p - 1` as shard records do — including the first and final tokens and
    /// rows on either side of a block boundary.
    #[test]
    fn feature_rows_are_selected_without_shift() {
        let rows = 4;
        for count in [2, 3, 5, 9, 10] {
            // Every value of row `p` is `p`, so a shifted read is visible.
            let all: Vec<f32> = (0..count)
                .flat_map(|p| std::iter::repeat_n(p as f32, WIDTH))
                .collect();
            let positions: Vec<usize> = (0..count)
                .filter(|p| p % 2 == 0 || p + 1 == count)
                .collect();
            assert_eq!(positions.last(), Some(&(count - 1)));
            let (mut out, mut next, mut start) = (Vec::new(), 0, 0);
            for take in block_sizes(count, rows) {
                let block = &all[start * WIDTH..(start + take) * WIDTH];
                next = select_rows(block, start, &positions, next, &mut out);
                start += take;
            }
            assert_eq!(next, positions.len());
            let want: Vec<f32> = positions
                .iter()
                .flat_map(|&p| std::iter::repeat_n(p as f32, WIDTH))
                .collect();
            assert!(out == want, "count {count}: rows read shifted");
        }
    }

    #[test]
    fn feature_requests_are_validated() {
        let tokens = [1, 2, 3, 4];
        assert!(validate_feature_request(&tokens, &[0, 3], 4).is_ok());
        assert!(validate_feature_request(&tokens, &[3], 4).is_ok());
        assert!(validate_feature_request(&tokens, &[], 4).is_err());
        assert!(validate_feature_request(&tokens, &[2, 1], 4).is_err());
        assert!(validate_feature_request(&tokens, &[1, 1], 4).is_err());
        assert!(validate_feature_request(&tokens, &[4], 4).is_err());
        assert!(validate_feature_request(&tokens, &[0], 3).is_err());
        assert!(validate_feature_request(&[1], &[0], 4).is_err());
        assert!(validate_feature_request(&[1, VOCAB as u32], &[0], 4).is_err());
    }

    #[test]
    fn feature_bytes_are_little_endian_f16_and_finite() {
        assert_eq!(
            features_f16_bytes(&[1.0, -2.0]).ok(),
            Some(vec![0x00, 0x3C, 0x00, 0xC0])
        );
        assert!(features_f16_bytes(&[1.0e6]).is_err());
        assert!(features_f16_bytes(&[f32::NAN]).is_err());
    }

    /// In-place rounding matches the stored FP16 bytes value for value.
    #[test]
    fn feature_rounding_matches_stored_f16() -> crate::Result<()> {
        let values = [1.0e-3_f32, -2.5, 0.1, 65_504.0];
        let mut rounded = values;
        super::round_features_f16(&mut rounded)?;
        let stored: Vec<f32> = features_f16_bytes(&values)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| half::f16::from_le_bytes(*pair).to_f32())
            .collect();
        assert_eq!(rounded.to_vec(), stored);
        assert!(
            (rounded[2] - 0.1).abs() > 1.0e-6,
            "0.1 is not exact in FP16"
        );
        assert!(super::round_features_f16(&mut [1.0e6]).is_err());
        Ok(())
    }

    /// The butterflies are the natural (Sylvester) order Walsh-Hadamard
    /// transform, H[i][j] = (-1)^popcount(i & j), scaled by 1/32 — the matrix
    /// `tools/mtp_train/ternary.py` builds.
    #[test]
    fn inverse_rotation_is_sylvester_order() {
        let width = 2048;
        let signs: Vec<f32> = (0..width)
            .map(|i| if (i * 5 + 1) % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let input: Vec<f32> = (0..width).map(|i| ((i * 31) % 17) as f32 - 8.0).collect();
        let mut rotated = input.clone();
        inverse_rotate(&mut rotated, &signs);
        for (block, outputs) in rotated.as_chunks::<1024>().0.iter().enumerate() {
            let inputs = &input[block * 1024..(block + 1) * 1024];
            for (i, &value) in outputs.iter().enumerate() {
                let sum: f64 = inputs
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        let sign = if (i & j).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        sign * f64::from(x)
                    })
                    .sum();
                let expected = sum / 32.0 * f64::from(signs[block * 1024 + i]);
                assert!(
                    (f64::from(value) - expected).abs() < 1e-3,
                    "{i}: {value} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn sign_codes_reject_bad_widths_and_values() {
        assert_eq!(sign_codes(&[1.0; 1024], 1024).ok(), Some(vec![1; 1024]));
        assert!(sign_codes(&[1.0; 1024], 2048).is_err());
        assert!(sign_codes(&[1.0; 1000], 1000).is_err());
        let mut signs = vec![-1.0; 1024];
        signs[3] = 0.5;
        assert!(sign_codes(&signs, 1024).is_err());
    }

    #[test]
    fn blocks_cover_without_single_rows() {
        assert_eq!(block_sizes(2, 60), vec![2]);
        assert_eq!(block_sizes(61, 60), vec![59, 2]);
        assert_eq!(block_sizes(120, 60), vec![60, 60]);
        assert_eq!(block_sizes(121, 60), vec![60, 59, 2]);
        for count in 2..400 {
            let sizes = block_sizes(count, 60);
            assert_eq!(sizes.iter().sum::<usize>(), count);
            assert!(sizes.iter().all(|&size| (2..=60).contains(&size)));
        }
    }

    #[test]
    fn top_k_is_sorted_and_stable() {
        let row = [1.0, 5.0, 3.0, 5.0, -2.0, 4.0];
        let (ids, values) = top_k_row(&row, 3);
        assert_eq!(ids, vec![1, 3, 5]);
        assert_eq!(values, vec![5.0, 5.0, 4.0]);
    }

    /// The inverse rotation is orthonormal and inverts the forward one
    /// (signs, then the normalized transform).
    #[test]
    fn inverse_rotation_inverts_forward() {
        let width = 2048;
        let signs: Vec<f32> = (0..width)
            .map(|i| if (i * 7 + 3) % 5 < 2 { -1.0 } else { 1.0 })
            .collect();
        let original: Vec<f32> = (0..width).map(|i| ((i * 37) % 101) as f32 - 50.0).collect();
        // Forward: multiply by signs, then the normalized transform, which is
        // the inverse with unit signs.
        let mut rotated: Vec<f32> = original.iter().zip(&signs).map(|(v, s)| v * s).collect();
        inverse_rotate(&mut rotated, &vec![1.0; width]);
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm(&rotated) - norm(&original)).abs() < 1e-2 * norm(&original));
        inverse_rotate(&mut rotated, &signs);
        for (a, b) in rotated.iter().zip(&original) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }
}
