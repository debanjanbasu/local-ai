use std::collections::HashMap;
use std::path::Path;

use half::f16;
use image::DynamicImage;
use image::GenericImageView;
use image::RgbImage;
use image::imageops::FilterType;
use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::kernels::Kernels;

use crate::Error;
use crate::layer::QuantWeight;
use crate::multimodal::DecodedRgbImage;

const VISION_HIDDEN: usize = 768;
const TEXT_HIDDEN: usize = 1536;
const VISION_LAYERS: usize = 16;
const VISION_HEADS: usize = 12;
const VISION_HEAD_DIM: usize = 64;
const VISION_INTERMEDIATE: usize = 3072;
const RGB_CHANNELS: usize = 3;
const PATCH_SIZE: usize = 16;
const POOLING_KERNEL: usize = 3;
const F16: usize = std::mem::size_of::<f16>();

/// Encode one image into Gemma 4 image-slot soft embeddings.
///
/// The current path keeps model math on Metal: patch embedding, learned 2D
/// position embedding, the 16-block SigLIP-style image encoder, pooling,
/// pooler scaling/RMSNorm, and the `mm.input_projection` projector. The CPU
/// side only decodes/resizes the user image into the processor's RGB tensor.
pub fn encode_image_soft_tokens(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    path: &Path,
    soft_token_budget: usize,
) -> crate::Result<Vec<Vec<f16>>> {
    if soft_token_budget == 0 {
        return Err(Error::InvalidArgument(
            "image soft-token budget must be non-zero".into(),
        ));
    }

    let (pixels, height, width) = load_resized_chw(path, soft_token_budget)?;
    encode_chw_soft_tokens(
        ctx,
        kernels,
        weights,
        &pixels,
        height,
        width,
        soft_token_budget,
    )
}

/// Encode an already-decoded RGB image.
///
/// This is the app-facing fast path for OS media stacks (for example
/// `AVFoundation` frame extraction) that should not
/// have to round-trip through temporary files.
pub fn encode_decoded_rgb_soft_tokens(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    image: &DecodedRgbImage,
    soft_token_budget: usize,
) -> crate::Result<Vec<Vec<f16>>> {
    if soft_token_budget == 0 {
        return Err(Error::InvalidArgument(
            "image soft-token budget must be non-zero".into(),
        ));
    }

    let (pixels, height, width) = decoded_rgb_to_resized_chw(image, soft_token_budget)?;
    encode_chw_soft_tokens(
        ctx,
        kernels,
        weights,
        &pixels,
        height,
        width,
        soft_token_budget,
    )
}

#[allow(clippy::too_many_lines)]
fn encode_chw_soft_tokens(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    pixels: &[f16],
    height: usize,
    width: usize,
    soft_token_budget: usize,
) -> crate::Result<Vec<Vec<f16>>> {
    let device = ctx.device();
    let input = MetalBuffer::from_slice(device, pixels)
        .map_err(|e| Error::InvalidArgument(format!("image input buffer: {e}")))?;

    let patch_weight = weights
        .get("v.patch_embd.weight")
        .ok_or_else(|| Error::InvalidFormat("missing v.patch_embd.weight".into()))?
        .as_f16_buffer("v.patch_embd.weight")?;
    let patch_bias = MetalBuffer::from_slice(device, &vec![f16::ZERO; VISION_HIDDEN])
        .map_err(|e| Error::InvalidArgument(format!("vision patch bias buffer: {e}")))?;

    let patch_rows = height / PATCH_SIZE;
    let patch_cols = width / PATCH_SIZE;
    let patch_count = patch_rows * patch_cols;
    let mut state = MetalBuffer::empty(device, patch_count * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision patch output buffer: {e}")))?;
    kernels
        .vision_patch_embed(
            ctx,
            &input,
            patch_weight,
            &patch_bias,
            &state,
            VISION_HIDDEN as u32,
            PATCH_SIZE as u32,
            height as u32,
            width as u32,
        )
        .map_err(Error::Metal)?;

    let position_table = weights
        .get("v.position_embd.weight")
        .ok_or_else(|| Error::InvalidFormat("missing v.position_embd.weight".into()))?
        .as_f16_buffer("v.position_embd.weight")?;
    kernels
        .vision_add_position_embedding(
            ctx,
            &state,
            position_table,
            VISION_HIDDEN as u32,
            10_240,
            patch_rows as u32,
            patch_cols as u32,
        )
        .map_err(Error::Metal)?;

    run_vision_encoder(ctx, kernels, weights, patch_rows, patch_cols, &mut state)?;

    let soft_rows = patch_rows / POOLING_KERNEL;
    let soft_cols = patch_cols / POOLING_KERNEL;
    let soft_tokens = soft_rows * soft_cols;
    if soft_tokens != soft_token_budget {
        return Err(Error::InvalidState(format!(
            "image processor produced {soft_tokens} soft tokens, expected {soft_token_budget}"
        )));
    }

    let pooled = MetalBuffer::empty(device, soft_tokens * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision pool output buffer: {e}")))?;
    kernels
        .vision_avg_pool_2d(
            ctx,
            &state,
            &pooled,
            VISION_HIDDEN as u32,
            patch_rows as u32,
            patch_cols as u32,
            POOLING_KERNEL as u32,
            POOLING_KERNEL as u32,
        )
        .map_err(Error::Metal)?;

    kernels
        .scale_in_place_gpu(
            ctx,
            &pooled,
            (VISION_HIDDEN as f32).sqrt(),
            (soft_tokens * VISION_HIDDEN) as u32,
        )
        .map_err(Error::Metal)?;

    let ones = MetalBuffer::from_slice(device, &vec![f16::from_f32(1.0); VISION_HIDDEN])
        .map_err(|e| Error::InvalidArgument(format!("vision norm weight buffer: {e}")))?;
    let normalized = MetalBuffer::empty(device, soft_tokens * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision norm output buffer: {e}")))?;
    kernels
        .rms_norm(
            ctx,
            &pooled,
            &ones,
            &normalized,
            VISION_HIDDEN as u32,
            soft_tokens as u32,
            1e-6,
        )
        .map_err(Error::Metal)?;

    let projected = MetalBuffer::empty(device, soft_tokens * TEXT_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision projection output buffer: {e}")))?;
    let projector = weights
        .get("mm.input_projection.weight")
        .ok_or_else(|| Error::InvalidFormat("missing mm.input_projection.weight".into()))?;
    if projector.cols() as usize != VISION_HIDDEN || projector.rows() as usize != TEXT_HIDDEN {
        return Err(Error::InvalidFormat(format!(
            "mm.input_projection.weight has shape [{}, {}], expected [{TEXT_HIDDEN}, {VISION_HIDDEN}]",
            projector.rows(),
            projector.cols()
        )));
    }
    let mut batch = CommandBatch::new(ctx).map_err(Error::Metal)?;
    projector.matmul_nt_into(
        &mut batch,
        kernels,
        &normalized,
        &projected,
        soft_tokens as u32,
    )?;
    batch.commit_and_wait().map_err(Error::Metal)?;

    let flat = projected.as_slice::<f16>();
    Ok(flat
        .chunks_exact(TEXT_HIDDEN)
        .map(<[f16]>::to_vec)
        .collect())
}

#[allow(clippy::too_many_lines)]
fn run_vision_encoder(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    patch_rows: usize,
    patch_cols: usize,
    state: &mut MetalBuffer,
) -> crate::Result<()> {
    let seq_len = patch_rows * patch_cols;
    let device = ctx.device();
    let mut residual_out = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision residual buffer: {e}")))?;
    let norm = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision norm buffer: {e}")))?;
    let q = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision q buffer: {e}")))?;
    let k = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision k buffer: {e}")))?;
    let q_normed = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision q norm buffer: {e}")))?;
    let k_normed = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision k norm buffer: {e}")))?;
    let v = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision v buffer: {e}")))?;
    let v_normed = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision v norm buffer: {e}")))?;
    // Gemma4V applies an unweighted per-head RMSNorm to V before attention
    // (`V = rms_norm(V)` with no learned scale), unlike plain SigLIP towers.
    let ones_head = MetalBuffer::from_slice(device, &[f16::from_f32(1.0); VISION_HEAD_DIM])
        .map_err(|e| Error::InvalidArgument(format!("vision v norm weight buffer: {e}")))?;
    let attn = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision attention buffer: {e}")))?;
    let proj = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision projection buffer: {e}")))?;
    let post = MetalBuffer::empty(device, seq_len * VISION_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision post-norm buffer: {e}")))?;
    let gate = MetalBuffer::empty(device, seq_len * VISION_INTERMEDIATE * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision ffn gate buffer: {e}")))?;
    let up = MetalBuffer::empty(device, seq_len * VISION_INTERMEDIATE * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision ffn up buffer: {e}")))?;
    let gated = MetalBuffer::empty(device, seq_len * VISION_INTERMEDIATE * F16)
        .map_err(|e| Error::InvalidArgument(format!("vision ffn gated buffer: {e}")))?;
    let mask = MetalBuffer::from_slice(device, &vec![f16::ZERO; seq_len * seq_len])
        .map_err(|e| Error::InvalidArgument(format!("vision attention mask buffer: {e}")))?;

    for layer in 0..VISION_LAYERS {
        let prefix = format!("v.blk.{layer}");
        let mut batch = CommandBatch::new(ctx).map_err(Error::Metal)?;
        rms_norm_into(
            &mut batch,
            kernels,
            weights,
            state,
            &norm,
            &format!("{prefix}.ln1.weight"),
            seq_len,
            VISION_HIDDEN,
        )?;
        clipped_linear_into(
            &mut batch,
            kernels,
            weights,
            &norm,
            &q,
            seq_len,
            VISION_HIDDEN,
            VISION_HIDDEN,
            &format!("{prefix}.attn_q"),
        )?;
        clipped_linear_into(
            &mut batch,
            kernels,
            weights,
            &norm,
            &k,
            seq_len,
            VISION_HIDDEN,
            VISION_HIDDEN,
            &format!("{prefix}.attn_k"),
        )?;
        clipped_linear_into(
            &mut batch,
            kernels,
            weights,
            &norm,
            &v,
            seq_len,
            VISION_HIDDEN,
            VISION_HIDDEN,
            &format!("{prefix}.attn_v"),
        )?;
        rms_norm_into(
            &mut batch,
            kernels,
            weights,
            &q,
            &q_normed,
            &format!("{prefix}.attn_q_norm.weight"),
            seq_len * VISION_HEADS,
            VISION_HEAD_DIM,
        )?;
        rms_norm_into(
            &mut batch,
            kernels,
            weights,
            &k,
            &k_normed,
            &format!("{prefix}.attn_k_norm.weight"),
            seq_len * VISION_HEADS,
            VISION_HEAD_DIM,
        )?;
        kernels
            .vision_rope_into(
                &mut batch,
                &q_normed,
                &k_normed,
                VISION_HEAD_DIM as u32,
                100.0,
                VISION_HEADS as u32,
                seq_len as u32,
                patch_rows as u32,
                patch_cols as u32,
            )
            .map_err(Error::Metal)?;
        kernels
            .rms_norm_into(
                &mut batch,
                &v,
                &ones_head,
                &v_normed,
                VISION_HEAD_DIM as u32,
                (seq_len * VISION_HEADS) as u32,
                1e-6,
            )
            .map_err(Error::Metal)?;
        kernels
            .flash_attention_prefill_masked_into(
                &mut batch,
                &q_normed,
                &k_normed,
                &v_normed,
                &attn,
                &mask,
                seq_len as u32,
                seq_len as u32,
                VISION_HEADS as u32,
                VISION_HEADS as u32,
                VISION_HEAD_DIM as u32,
                1.0,
            )
            .map_err(Error::Metal)?;
        clipped_linear_into(
            &mut batch,
            kernels,
            weights,
            &attn,
            &proj,
            seq_len,
            VISION_HIDDEN,
            VISION_HIDDEN,
            &format!("{prefix}.attn_out"),
        )?;
        rms_norm_into(
            &mut batch,
            kernels,
            weights,
            &proj,
            &post,
            &format!("{prefix}.attn_post_norm.weight"),
            seq_len,
            VISION_HIDDEN,
        )?;
        kernels
            .residual_add_into(
                &mut batch,
                state,
                &post,
                &residual_out,
                (seq_len * VISION_HIDDEN) as u32,
            )
            .map_err(Error::Metal)?;
        std::mem::swap(state, &mut residual_out);

        rms_norm_into(
            &mut batch,
            kernels,
            weights,
            state,
            &norm,
            &format!("{prefix}.ln2.weight"),
            seq_len,
            VISION_HIDDEN,
        )?;
        clipped_linear_into(
            &mut batch,
            kernels,
            weights,
            &norm,
            &gate,
            seq_len,
            VISION_HIDDEN,
            VISION_INTERMEDIATE,
            &format!("{prefix}.ffn_gate"),
        )?;
        clipped_linear_into(
            &mut batch,
            kernels,
            weights,
            &norm,
            &up,
            seq_len,
            VISION_HIDDEN,
            VISION_INTERMEDIATE,
            &format!("{prefix}.ffn_up"),
        )?;
        kernels
            .gelu_into(
                &mut batch,
                &gate,
                &gate,
                (seq_len * VISION_INTERMEDIATE) as u32,
            )
            .map_err(Error::Metal)?;
        kernels
            .elementwise_mul_into(
                &mut batch,
                &gate,
                &up,
                &gated,
                (seq_len * VISION_INTERMEDIATE) as u32,
            )
            .map_err(Error::Metal)?;
        clipped_linear_into(
            &mut batch,
            kernels,
            weights,
            &gated,
            &proj,
            seq_len,
            VISION_INTERMEDIATE,
            VISION_HIDDEN,
            &format!("{prefix}.ffn_down"),
        )?;
        rms_norm_into(
            &mut batch,
            kernels,
            weights,
            &proj,
            &post,
            &format!("{prefix}.ffn_post_norm.weight"),
            seq_len,
            VISION_HIDDEN,
        )?;
        kernels
            .residual_add_into(
                &mut batch,
                state,
                &post,
                &residual_out,
                (seq_len * VISION_HIDDEN) as u32,
            )
            .map_err(Error::Metal)?;
        std::mem::swap(state, &mut residual_out);
        batch.commit_and_wait().map_err(Error::Metal)?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rms_norm_into(
    batch: &mut CommandBatch,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    input: &MetalBuffer,
    output: &MetalBuffer,
    weight_name: &str,
    rows: usize,
    dim: usize,
) -> crate::Result<()> {
    let weight = weights
        .get(weight_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {weight_name}")))?
        .as_f16_buffer(weight_name)?;
    kernels
        .rms_norm_into(batch, input, weight, output, dim as u32, rows as u32, 1e-6)
        .map_err(Error::Metal)
}

#[allow(clippy::too_many_arguments)]
fn clipped_linear_into(
    batch: &mut CommandBatch,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    input: &MetalBuffer,
    output: &MetalBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    prefix: &str,
) -> crate::Result<()> {
    let weight_name = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {weight_name}")))?;
    if weight.cols() as usize != input_dim || weight.rows() as usize != output_dim {
        return Err(Error::InvalidFormat(format!(
            "{weight_name} has shape [{}, {}], expected [{output_dim}, {input_dim}]",
            weight.rows(),
            weight.cols()
        )));
    }
    kernels
        .clipped_linear_tiled_nt_into(
            batch,
            input,
            weight.as_f16_buffer(&weight_name)?,
            None,
            output,
            rows as u32,
            input_dim as u32,
            output_dim as u32,
            scalar(weights, &format!("{prefix}.input_min"))?,
            scalar(weights, &format!("{prefix}.input_max"))?,
            scalar(weights, &format!("{prefix}.output_min"))?,
            scalar(weights, &format!("{prefix}.output_max"))?,
        )
        .map_err(Error::Metal)
}

fn scalar(weights: &HashMap<String, QuantWeight>, name: &str) -> crate::Result<f32> {
    let buf = weights
        .get(name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {name}")))?
        .as_f16_buffer(name)?;
    Ok(buf
        .as_slice::<f16>()
        .first()
        .copied()
        .unwrap_or(f16::ZERO)
        .to_f32())
}

fn load_resized_chw(
    path: &Path,
    soft_token_budget: usize,
) -> crate::Result<(Vec<f16>, usize, usize)> {
    let image = image::open(path).map_err(|e| {
        Error::InvalidArgument(format!("failed to decode image {}: {e}", path.display()))
    })?;
    dynamic_image_to_resized_chw(
        &image,
        soft_token_budget,
        &format!("image {}", path.display()),
    )
}

fn decoded_rgb_to_resized_chw(
    image: &DecodedRgbImage,
    soft_token_budget: usize,
) -> crate::Result<(Vec<f16>, usize, usize)> {
    let expected_len = usize::try_from(image.width)
        .ok()
        .and_then(|w| {
            usize::try_from(image.height)
                .ok()
                .map(|h| w * h * RGB_CHANNELS)
        })
        .ok_or_else(|| Error::InvalidArgument("decoded image dimensions overflow".into()))?;
    if image.rgb.len() != expected_len {
        return Err(Error::InvalidArgument(format!(
            "decoded RGB image has {} bytes, expected {expected_len} for {}x{} RGB888",
            image.rgb.len(),
            image.width,
            image.height
        )));
    }
    let rgb =
        RgbImage::from_raw(image.width, image.height, image.rgb.clone()).ok_or_else(|| {
            Error::InvalidArgument("decoded RGB image buffer shape is invalid".into())
        })?;
    dynamic_image_to_resized_chw(
        &DynamicImage::ImageRgb8(rgb),
        soft_token_budget,
        "decoded RGB image",
    )
}

fn dynamic_image_to_resized_chw(
    image: &DynamicImage,
    soft_token_budget: usize,
    label: &str,
) -> crate::Result<(Vec<f16>, usize, usize)> {
    let (src_w, src_h) = image.dimensions();
    if src_w == 0 || src_h == 0 {
        return Err(Error::InvalidArgument(format!(
            "{label} has empty dimensions"
        )));
    }
    let (unit_rows, unit_cols) = choose_soft_grid(src_w, src_h, soft_token_budget);
    let height = unit_rows * POOLING_KERNEL * PATCH_SIZE;
    let width = unit_cols * POOLING_KERNEL * PATCH_SIZE;
    let resized = image
        .resize_exact(width as u32, height as u32, FilterType::CatmullRom)
        .to_rgb8();

    let plane = height * width;
    let mut out = vec![f16::ZERO; RGB_CHANNELS * plane];
    for y in 0..height {
        for x in 0..width {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            let idx = y * width + x;
            for c in 0..RGB_CHANNELS {
                let rescaled = f32::from(pixel[c]) / 255.0;
                // Gemma4VisionPatchEmbedder applies `2 * (x - 0.5)` after the
                // processor's [0,1] rescale.
                out[c * plane + idx] = f16::from_f32(2.0 * (rescaled - 0.5));
            }
        }
    }
    Ok((out, height, width))
}

fn choose_soft_grid(src_w: u32, src_h: u32, soft_token_budget: usize) -> (usize, usize) {
    let aspect = f64::from(src_w) / f64::from(src_h);
    let mut best = (1usize, soft_token_budget.max(1));
    let mut best_err = f64::INFINITY;
    for rows in 1..=soft_token_budget.max(1) {
        if !soft_token_budget.is_multiple_of(rows) {
            continue;
        }
        let cols = soft_token_budget / rows;
        let candidate_aspect = cols as f64 / rows as f64;
        let err = (candidate_aspect.ln() - aspect.ln()).abs();
        if err < best_err {
            best = (rows, cols);
            best_err = err;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_grid_preserves_budget_and_orientation() {
        let (rows, cols) = choose_soft_grid(1600, 900, 280);
        assert_eq!(rows * cols, 280);
        assert!(cols > rows);

        let (rows, cols) = choose_soft_grid(900, 1600, 280);
        assert_eq!(rows * cols, 280);
        assert!(rows > cols);
    }
}
