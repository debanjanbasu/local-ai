use core::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice, MTLSize,
};

use crate::Error;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::ShaderLibrary;

/// Holds pre-built compute pipelines for utility shader kernels.
#[allow(clippy::struct_field_names)]
pub struct Kernels {
    rms_norm_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gelu_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gelu_mul_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rms_norm_add_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gelu_f32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    softmax_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    embedding_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rope_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matvec_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matvec_f32out_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matvec_f32in_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    attention_sliding_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    attention_sliding_paged_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    attention_full_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    attention_full_paged_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    flash_attention_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    flash_attention_paged_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    flash_decoding_split_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_decoding_reduce_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attention_windowed_split_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    elementwise_mul_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    elementwise_mul_f32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    residual_add_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scale_in_place_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    logit_softcap_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gather_strided_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    qk_norm_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    qk_norm_rope_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    depthwise_conv1d_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    batch_norm_1d_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    silu_inplace_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    vision_patch_embed_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    vision_add_position_embedding_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    clipped_linear_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul_f16_bias_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    vision_avg_pool_2d_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    vision_rope_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    audio_subsample_conv2d_ln_relu_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    audio_pack_frontend_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    audio_chunked_attention_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    audio_glu_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    tq2_dequant_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    write_kv_cache_decode_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attention_decode_batched_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    clipped_linear_f16_nt_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attn_prefill_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attn_prefill_tiled_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attn_prefill_mqa_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attn_prefill_masked_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    rope_batch_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    rope_batch_decode_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    bf16_to_fp16_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    dequantize_kv_turboquant_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    encode_kv_turboquant_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    encode_kv_turboquant_batched_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    hadamard_rotate_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    hadamard_rotate_hf_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    hadamard_rotate_fh_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attention_tq_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attention_tq_prefill_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attention_tq_prefill_tiled_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    flash_attention_tq_batched_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    matmul_f16_nt_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    matvec_q4_0_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matvec_q4_0_f32out_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matvec_tq2_0_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matvec_tq2_0_soa_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    matvec_tq2_0_f32out_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul_nt_q4_0_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul_nt_tq2_0_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul_nt_q4_0_mma_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    matmul_nt_tq2_0_mma_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    matmul_nt_q4_0_smallm_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    matmul_nt_tq2_0_smallm_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    matmul_nt_tq2_0_batchm_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    multivec_nt_q4_0_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    multivec_nt_tq2_0_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Kernels {
    /// Build compute pipelines for all utility kernels.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShaderNotFound`] or [`Error::PipelineCreation`] on failure.
    #[allow(clippy::too_many_lines)]
    pub fn new(ctx: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        let make_pipeline =
            |name: &str| -> crate::Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
                let func = shaders.get_function(name)?;
                ctx.device()
                    .newComputePipelineStateWithFunction_error(&func)
                    .map_err(|e| Error::PipelineCreation(e.to_string()))
            };

        Ok(Self {
            rms_norm_pipeline: make_pipeline("rms_norm")?,
            gelu_pipeline: make_pipeline("gelu_tanh")?,
            gelu_mul_pipeline: make_pipeline("gelu_mul_f16")?,
            rms_norm_add_pipeline: make_pipeline("rms_norm_add")?,
            gelu_f32_pipeline: make_pipeline("gelu_tanh_f32")?,
            softmax_pipeline: make_pipeline("softmax")?,
            embedding_pipeline: make_pipeline("embedding_lookup")?,
            rope_pipeline: make_pipeline("rope")?,
            matvec_pipeline: make_pipeline("matvec_f16")?,
            matvec_f32out_pipeline: make_pipeline("matvec_f16w_f32out")?,
            matvec_f32in_pipeline: make_pipeline("matvec_f16w_f32in")?,
            attention_sliding_pipeline: make_pipeline("attention_sliding")?,
            attention_sliding_paged_pipeline: make_pipeline("attention_sliding_paged")?,
            attention_full_pipeline: make_pipeline("attention_full")?,
            attention_full_paged_pipeline: make_pipeline("attention_full_paged")?,
            flash_attention_pipeline: make_pipeline("flash_attention")?,
            flash_attention_paged_pipeline: make_pipeline("flash_attention_paged")?,
            flash_decoding_split_pipeline: make_pipeline("flash_decoding_split").ok(),
            flash_decoding_reduce_pipeline: make_pipeline("flash_decoding_reduce").ok(),
            flash_attention_windowed_split_pipeline: make_pipeline(
                "flash_attention_windowed_split",
            )
            .ok(),
            elementwise_mul_pipeline: make_pipeline("elementwise_mul_f16")?,
            elementwise_mul_f32_pipeline: make_pipeline("elementwise_mul_f32")?,
            residual_add_pipeline: make_pipeline("residual_add")?,
            scale_in_place_pipeline: make_pipeline("scale_in_place")?,
            logit_softcap_pipeline: make_pipeline("logit_softcap_f32")?,
            gather_strided_pipeline: make_pipeline("gather_strided_f16")?,
            qk_norm_pipeline: make_pipeline("qk_norm")?,
            qk_norm_rope_pipeline: make_pipeline("qk_norm_rope").ok(),
            depthwise_conv1d_pipeline: make_pipeline("depthwise_conv1d").ok(),
            batch_norm_1d_pipeline: make_pipeline("batch_norm_1d").ok(),
            silu_inplace_pipeline: make_pipeline("silu_inplace").ok(),
            vision_patch_embed_pipeline: make_pipeline("vision_patch_embed").ok(),
            vision_add_position_embedding_pipeline: make_pipeline("vision_add_position_embedding")
                .ok(),
            clipped_linear_pipeline: make_pipeline("clipped_linear")?,
            matmul_f16_bias_pipeline: make_pipeline("matmul_f16_bias")?,
            vision_avg_pool_2d_pipeline: make_pipeline("vision_avg_pool_2d").ok(),
            vision_rope_pipeline: make_pipeline("vision_rope").ok(),
            audio_subsample_conv2d_ln_relu_pipeline: make_pipeline(
                "audio_subsample_conv2d_ln_relu",
            )
            .ok(),
            audio_pack_frontend_pipeline: make_pipeline("audio_pack_frontend").ok(),
            audio_chunked_attention_pipeline: make_pipeline("audio_chunked_attention").ok(),
            audio_glu_pipeline: make_pipeline("audio_glu").ok(),
            clipped_linear_f16_nt_pipeline: make_pipeline("clipped_linear_f16_nt").ok(),
            flash_attn_prefill_pipeline: make_pipeline("flash_attention_prefill").ok(),
            flash_attn_prefill_tiled_pipeline: make_pipeline("flash_attention_prefill_tiled").ok(),
            flash_attn_prefill_mqa_pipeline: make_pipeline("flash_attention_prefill_mqa").ok(),
            flash_attn_prefill_masked_pipeline: make_pipeline("flash_attention_prefill_masked")
                .ok(),
            rope_batch_pipeline: make_pipeline("rope_batch").ok(),
            rope_batch_decode_pipeline: make_pipeline("rope_batch_decode").ok(),
            bf16_to_fp16_pipeline: make_pipeline("bf16_to_fp16").ok(),
            dequantize_kv_turboquant_pipeline: make_pipeline("dequantize_kv_turboquant").ok(),
            encode_kv_turboquant_pipeline: make_pipeline("encode_kv_turboquant").ok(),
            encode_kv_turboquant_batched_pipeline: make_pipeline("encode_kv_turboquant_batched")
                .ok(),
            hadamard_rotate_pipeline: make_pipeline("hadamard_rotate").ok(),
            hadamard_rotate_hf_pipeline: make_pipeline("hadamard_rotate_hf").ok(),
            hadamard_rotate_fh_pipeline: make_pipeline("hadamard_rotate_fh").ok(),
            flash_attention_tq_pipeline: make_pipeline("flash_attention_tq").ok(),
            flash_attention_tq_prefill_pipeline: make_pipeline("flash_attention_tq_prefill").ok(),
            flash_attention_tq_prefill_tiled_pipeline: make_pipeline(
                "flash_attention_tq_prefill_tiled",
            )
            .ok(),
            flash_attention_tq_batched_pipeline: make_pipeline("flash_attention_tq_batched").ok(),
            matmul_f16_nt_pipeline: make_pipeline("matmul_f16_nt").ok(),
            tq2_dequant_pipeline: make_pipeline("tq2_dequant").ok(),
            write_kv_cache_decode_pipeline: make_pipeline("write_kv_cache_decode").ok(),
            flash_attention_decode_batched_pipeline: make_pipeline(
                "flash_attention_decode_batched",
            )
            .ok(),
            matvec_q4_0_pipeline: make_pipeline("matvec_q4_0")?,
            matvec_q4_0_f32out_pipeline: make_pipeline("matvec_q4_0_f32out")?,
            matvec_tq2_0_pipeline: make_pipeline("matvec_tq2_0")?,
            matvec_tq2_0_soa_pipeline: make_pipeline("matvec_tq2_0_soa").ok(),
            matvec_tq2_0_f32out_pipeline: make_pipeline("matvec_tq2_0_f32out")?,
            matmul_nt_q4_0_pipeline: make_pipeline("matmul_nt_q4_0")?,
            matmul_nt_tq2_0_pipeline: make_pipeline("matmul_nt_tq2_0")?,
            matmul_nt_q4_0_mma_pipeline: make_pipeline("matmul_nt_q4_0_mma").ok(),
            matmul_nt_q4_0_smallm_pipeline: make_pipeline("matmul_nt_q4_0_smallm").ok(),
            matmul_nt_tq2_0_smallm_pipeline: make_pipeline("matmul_nt_tq2_0_smallm").ok(),
            matmul_nt_tq2_0_batchm_pipeline: make_pipeline("matmul_nt_tq2_0_batchm").ok(),
            matmul_nt_tq2_0_mma_pipeline: make_pipeline("matmul_nt_tq2_0_mma").ok(),
            multivec_nt_q4_0_pipeline: make_pipeline("multivec_nt_q4_0")?,
            multivec_nt_tq2_0_pipeline: make_pipeline("multivec_nt_tq2_0")?,
        })
    }

    /// Dispatch RMS-norm: `output[row] = input[row] * weight / rms(input[row])`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn rms_norm(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        dim: u32,
        rows: u32,
        eps: f32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.rms_norm_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(eps).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                4,
            );
        }

        let tg_size = dim.min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch GELU (tanh approximation): `output[i] = gelu(input[i])`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code)]
    pub fn gelu(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        output: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.gelu_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch softmax: row-wise softmax over `dim`-wide rows.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code)]
    pub fn softmax(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        output: &MetalBuffer,
        dim: u32,
        rows: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.softmax_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
        }

        let tg_size = dim.min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }
    /// Dispatch embedding lookup: `output[token_idx] = table[token_id]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn embedding(
        &self,
        ctx: &MetalContext,
        table: &MetalBuffer,
        token_ids: &MetalBuffer,
        output: &MetalBuffer,
        hidden_dim: u32,
        num_tokens: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.embedding_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(table.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(token_ids.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(hidden_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: hidden_dim.div_ceil(256) as usize,
                height: num_tokens as usize,
                depth: 1,
            },
            MTLSize {
                width: hidden_dim.min(256) as usize,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch `RoPE` on a single head vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn rope(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        position: u32,
        theta: f32,
        partial_rotary_factor: f32,
        num_heads: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.rope_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(position).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(theta).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(partial_rotary_factor)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<f32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
        }

        let half_dim = (head_dim / 2) as usize;
        let tg_width = half_dim.min(256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: half_dim.div_ceil(tg_width),
                height: num_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`rope`](Self::rope) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn rope_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        position: u32,
        theta: f32,
        partial_rotary_factor: f32,
        num_heads: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.rope_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(position).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(theta).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(partial_rotary_factor)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<f32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
        }

        let half_dim = (head_dim / 2) as usize;
        let tg_width = half_dim.min(256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: half_dim.div_ceil(tg_width),
                height: num_heads as usize,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Dispatch FP16 matrix-vector multiply: `output = matrix × vector`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matvec(
        &self,
        ctx: &MetalContext,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.matvec_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(matrix.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(vector.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
        }

        let rows_per_tg = Self::F16_MATVEC_ROWS_PER_TG;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (rows as usize).div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: rows_per_tg * 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch elementwise multiply: `out[i] = a[i] * b[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code)]
    pub fn elementwise_mul(
        &self,
        ctx: &MetalContext,
        a: &MetalBuffer,
        b: &MetalBuffer,
        out: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.elementwise_mul_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch sliding-window causal attention with GQA.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn attention_sliding(
        &self,
        ctx: &MetalContext,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        window: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.attention_sliding_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(current_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(window).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                9,
            );
        }

        let threads = MTLSize {
            width: head_dim as usize,
            height: num_q_heads as usize,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: head_dim as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(threads, threads_per_tg);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn attention_sliding_paged(
        &self,
        ctx: &MetalContext,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        page_table: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        window: u32,
        page_size_tokens: u64,
        num_q_heads: u32,
        num_kv_heads: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.attention_sliding_paged_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(page_table.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(current_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(window).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(page_size_tokens)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<u64>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                10,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                11,
            );
        }

        let threads = MTLSize {
            width: head_dim as usize,
            height: num_q_heads as usize,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: head_dim as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(threads, threads_per_tg);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn attention_full(
        &self,
        ctx: &MetalContext,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.attention_full_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(current_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                8,
            );
        }

        let threads = MTLSize {
            width: head_dim as usize,
            height: num_q_heads as usize,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: head_dim as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(threads, threads_per_tg);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Flash Attention v2 for single-query decode.
    ///
    /// Processes KV in tiles with online softmax, enabling arbitrarily long
    /// sequences without large threadgroup memory allocations.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn attention_full_paged(
        &self,
        ctx: &MetalContext,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        page_table: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        page_size_tokens: u64,
        num_q_heads: u32,
        num_kv_heads: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.attention_full_paged_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(page_table.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(current_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(page_size_tokens)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<u64>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                10,
            );
        }

        let threads = MTLSize {
            width: head_dim as usize,
            height: num_q_heads as usize,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: head_dim as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(threads, threads_per_tg);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// `window`: sliding-window size — the query attends keys `k` with
    /// `current_pos - window < k <= current_pos`; `0` means unlimited.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention(
        &self,
        ctx: &MetalContext,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        window: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.flash_attention_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(current_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(window).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                9,
            );
        }

        let tg_width = 256usize; // 8 simdgroups per head (split-K barrier-free flash decode)
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_q_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`flash_attention`](Self::flash_attention) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    /// Always runs unwindowed (binds `window = 0`).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.flash_attention_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(current_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                8,
            );
            // The kernel reads `window` at buffer(9); this batched path does
            // not use sliding windows, so bind 0 (= unlimited).
            let window: u32 = 0;
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(window).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                9,
            );
        }

        let tg_width = 256usize; // 8 simdgroups per head (split-K barrier-free flash decode)
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_q_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Like [`flash_attention_into`](Self::flash_attention_into) but with an
    /// explicit sliding `window` (the query attends keys `k` with
    /// `current_pos - window < k <= current_pos`; `0` means unlimited). Used by
    /// the FP16 KV decode path, which reads K/V straight out of the resident
    /// cache buffers.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_windowed_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        window: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.flash_attention_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(current_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(window).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                9,
            );
        }
        let tg_width = 256usize; // 8 simdgroups per head (split-K barrier-free flash decode)
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_q_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Number of KV splits to use for [`Self::flash_attention_windowed_split_into`]
    /// given the effective (post-window) sequence length. Returns `1` when
    /// splitting would not help (short context) so the caller can fall back to
    /// the single-pass kernel. `0` means "no split path available".
    #[must_use]
    pub fn flash_split_count(&self, eff_seq_len: u32) -> u32 {
        const CHUNK: u32 = 16;
        const MAX_SPLITS: u32 = 16;

        if self.flash_attention_windowed_split_pipeline.is_none()
            || self.flash_decoding_reduce_pipeline.is_none()
        {
            return 0;
        }
        // Single-query decode launches only `num_q_heads` threadgroups; splitting
        // the KV range into ~128-key chunks recovers occupancy. Cap the split
        // count so the partial scratch / reduce overhead stays small.
        (eff_seq_len.div_ceil(CHUNK)).clamp(1, MAX_SPLITS)
    }

    /// Flash-decoding with sliding-window support: splits the KV range across
    /// `num_splits` threadgroups per head (raising GPU occupancy for the
    /// otherwise `num_q_heads`-threadgroup decode), writing per-split partials,
    /// then reduces them. `partial_max`/`partial_sum` must hold at least
    /// `num_q_heads * num_splits` f32; `partial_acc` at least
    /// `num_q_heads * num_splits * head_dim` f32.
    ///
    /// # Errors
    ///
    /// Returns an error if the split/reduce pipelines are unavailable or encoding
    /// fails.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_windowed_split_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        partial_max: &MetalBuffer,
        partial_sum: &MetalBuffer,
        partial_acc: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        window: u32,
        num_splits: u32,
    ) -> crate::Result<()> {
        let split = self
            .flash_attention_windowed_split_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("flash_attention_windowed_split".into()))?;
        let reduce = self
            .flash_decoding_reduce_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("flash_decoding_reduce".into()))?;

        // Split pass: grid (num_q_heads * num_splits), 256 threads (FLASH_NSG*32).
        let encoder = batch.encoder();
        encoder.setComputePipelineState(split);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(partial_max.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(partial_sum.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(partial_acc.raw()), 0, 5);
            let scalars = [
                head_dim,
                kv_len,
                current_pos,
                num_q_heads,
                num_kv_heads,
                window,
                num_splits,
            ];
            for (i, s) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(std::ptr::addr_of!(*s).cast_mut().cast::<c_void>()),
                    size_of::<u32>(),
                    6 + i,
                );
            }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (num_q_heads * num_splits) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();

        // Reduce pass: grid (num_q_heads). Serial encoder orders it after split.
        let encoder = batch.encoder();
        encoder.setComputePipelineState(reduce);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(partial_max.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(partial_sum.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(partial_acc.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            let scalars = [head_dim, num_splits, num_q_heads];
            for (i, s) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(std::ptr::addr_of!(*s).cast_mut().cast::<c_void>()),
                    size_of::<u32>(),
                    4 + i,
                );
            }
        }
        let red_width = (head_dim as usize).min(256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_q_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: red_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Flash-decoding: splits KV across threadgroups for parallel decode attention.
    /// Falls back to `flash_attention_into` for short sequences (`kv_len` <= 256).
    #[allow(unsafe_code, clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn flash_decoding_into(
        &self,
        ctx: &MetalContext,
        batch: &mut crate::batch::CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
    ) -> crate::Result<()> {
        let seq_len = kv_len.min(current_pos + 1);

        // For short sequences, delegate to standard flash attention
        if seq_len <= 256 {
            return self.flash_attention_into(
                batch,
                q,
                k,
                v,
                output,
                head_dim,
                kv_len,
                current_pos,
                num_q_heads,
                num_kv_heads,
            );
        }

        let split_pipeline = self
            .flash_decoding_split_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("flash_decoding_split".to_owned()))?;
        let reduce_pipeline = self
            .flash_decoding_reduce_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("flash_decoding_reduce".to_owned()))?;

        let num_splits = seq_len.div_ceil(256);

        // Allocate intermediate buffers
        let partial_max = MetalBuffer::empty(
            ctx.device(),
            (num_q_heads * num_splits) as usize * size_of::<f32>(),
        )?;
        let partial_sum = MetalBuffer::empty(
            ctx.device(),
            (num_q_heads * num_splits) as usize * size_of::<f32>(),
        )?;
        let partial_acc = MetalBuffer::empty(
            ctx.device(),
            (num_q_heads * num_splits * head_dim) as usize * size_of::<f32>(),
        )?;

        // Dispatch split kernel
        let encoder = batch.encoder();
        encoder.setComputePipelineState(split_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(partial_max.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(partial_sum.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(partial_acc.raw()), 0, 5);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(current_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                10,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_splits).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                11,
            );
        }

        let tg_width = (head_dim as usize).min(256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (num_q_heads * num_splits) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();

        // Barrier: end current encoder, start a new one on same command buffer
        batch.submit_and_renew(ctx)?;

        // Dispatch reduce kernel
        let encoder = batch.encoder();
        encoder.setComputePipelineState(reduce_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(partial_max.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(partial_sum.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(partial_acc.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_splits).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_q_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_paged(
        &self,
        ctx: &MetalContext,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        page_table: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        page_size_tokens: u64,
        num_q_heads: u32,
        num_kv_heads: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.flash_attention_paged_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(page_table.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(current_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(page_size_tokens)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<u64>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                10,
            );
        }

        let tg_width = (head_dim as usize).min(256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_q_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`matvec`](Self::matvec) but reads the matrix at a byte offset within `matrix`.
    ///
    /// This supports reading weight matrices from a packed pool buffer where
    /// each tensor starts at a different `pool_offset`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matvec_offset(
        &self,
        ctx: &MetalContext,
        matrix: &MetalBuffer,
        matrix_offset: usize,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.matvec_pipeline);

        // SAFETY: pointers are valid for the duration of the encode; Metal reads
        // them during dispatch which completes before we return.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(matrix.raw()), matrix_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(vector.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
        }

        let rows_per_tg = Self::F16_MATVEC_ROWS_PER_TG;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (rows as usize).div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: rows_per_tg * 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`matvec_offset`](Self::matvec_offset) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matvec_offset_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        matrix: &MetalBuffer,
        matrix_offset: usize,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.matvec_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(matrix.raw()), matrix_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(vector.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
        }

        let rows_per_tg = Self::F16_MATVEC_ROWS_PER_TG;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (rows as usize).div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: rows_per_tg * 32,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Dispatch FP16-weights × FP16-vector → float32-output matvec with matrix offset.
    ///
    /// Used for MLP gate/up projections where the intermediate result must stay
    /// in float32 to avoid FP16 overflow in downstream elementwise multiply.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matvec_f32out_offset(
        &self,
        ctx: &MetalContext,
        matrix: &MetalBuffer,
        matrix_offset: usize,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.matvec_f32out_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(matrix.raw()), matrix_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(vector.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
        }

        let rows_per_tg = Self::F16_MATVEC_ROWS_PER_TG;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (rows as usize).div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: rows_per_tg * 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`matvec_f32out_offset`](Self::matvec_f32out_offset) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matvec_f32out_offset_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        matrix: &MetalBuffer,
        matrix_offset: usize,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.matvec_f32out_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(matrix.raw()), matrix_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(vector.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
        }

        let rows_per_tg = Self::F16_MATVEC_ROWS_PER_TG;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (rows as usize).div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: rows_per_tg * 32,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Dispatch FP16-weights × float32-vector → FP16-output matvec with matrix offset.
    ///
    /// Used for MLP down projection: the input vector is the float32 gate*up
    /// product, and the output can safely be stored as FP16.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matvec_f32in_offset(
        &self,
        ctx: &MetalContext,
        matrix: &MetalBuffer,
        matrix_offset: usize,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.matvec_f32in_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(matrix.raw()), matrix_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(vector.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
        }

        let rows_per_tg = Self::F16_MATVEC_ROWS_PER_TG;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (rows as usize).div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: rows_per_tg * 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch float32 GELU (tanh approximation) activation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code)]
    pub fn gelu_f32(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        output: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.gelu_f32_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch float32 elementwise multiply.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code)]
    pub fn elementwise_mul_f32(
        &self,
        ctx: &MetalContext,
        a: &MetalBuffer,
        b: &MetalBuffer,
        out: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.elementwise_mul_f32_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`rms_norm`](Self::rms_norm) but reads the weight vector at a byte offset.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn rms_norm_offset(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        weight_offset: usize,
        output: &MetalBuffer,
        dim: u32,
        rows: u32,
        eps: f32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.rms_norm_pipeline);

        // SAFETY: pointers are valid for the duration of the encode.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), weight_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(eps).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                4,
            );
        }

        let tg_size = dim.min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`rms_norm_offset`](Self::rms_norm_offset) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn rms_norm_offset_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        weight_offset: usize,
        output: &MetalBuffer,
        dim: u32,
        rows: u32,
        eps: f32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.rms_norm_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), weight_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(eps).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                4,
            );
        }

        let tg_size = dim.min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Like [`rms_norm`](Self::rms_norm) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn rms_norm_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        dim: u32,
        rows: u32,
        eps: f32,
    ) -> crate::Result<()> {
        self.rms_norm_offset_into(batch, input, weight, 0, output, dim, rows, eps)
    }

    // ── QK Norm (per-head RMSNorm) ─────────────────────────────────────

    /// Dispatch per-head `RMSNorm` with learned weights (in-place on `qk_data`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn qk_norm_gpu(
        &self,
        ctx: &MetalContext,
        qk_data: &MetalBuffer,
        weight: &MetalBuffer,
        head_dim: u32,
        num_heads: u32,
        eps: f32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.qk_norm_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(qk_data.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(eps).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                3,
            );
        }

        let tg_size = head_dim.min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`qk_norm_gpu`](Self::qk_norm_gpu) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn qk_norm_gpu_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        qk_data: &MetalBuffer,
        weight: &MetalBuffer,
        head_dim: u32,
        num_heads: u32,
        eps: f32,
    ) -> crate::Result<()> {
        self.qk_norm_offset_into(batch, qk_data, weight, 0, head_dim, num_heads, eps)
    }

    /// Like [`qk_norm_gpu`](Self::qk_norm_gpu) but encodes into a batch with a weight buffer offset.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn qk_norm_offset_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        qk_data: &MetalBuffer,
        weight_buf: &MetalBuffer,
        weight_offset: usize,
        head_dim: u32,
        num_heads: u32,
        eps: f32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.qk_norm_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(qk_data.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight_buf.raw()), weight_offset, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(eps).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                3,
            );
        }

        let tg_size = head_dim.min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Largest `head_dim` the fused [`Self::qk_norm_rope_into`] kernel supports;
    /// must match `QK_NORM_MAX_HEAD_DIM` in `shaders/qk_norm.metal`.
    pub const QK_NORM_ROPE_MAX_HEAD_DIM: u32 = 256;

    /// Whether the fused qk-norm+RoPE kernel is available (compiled) and the
    /// `head_dim` is within its staging-array bound.
    #[must_use]
    pub fn qk_norm_rope_available(&self, head_dim: u32) -> bool {
        // Cached env opt-out for A/B (`LOCAL_AI_NO_FUSE_QKROPE=1`); read once to
        // keep this off the per-layer hot path.
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        let disabled = *DISABLED
            .get_or_init(|| std::env::var("LOCAL_AI_NO_FUSE_QKROPE").as_deref() == Ok("1"));
        !disabled
            && self.qk_norm_rope_pipeline.is_some()
            && head_dim <= Self::QK_NORM_ROPE_MAX_HEAD_DIM
    }

    /// Fused per-head `RMSNorm` + `NEOX` `RoPE`, encoded into a batch — equivalent to
    /// [`Self::qk_norm_gpu_into`] immediately followed by [`Self::rope_into`],
    /// in a single dispatch. Caller must check [`Self::qk_norm_rope_available`].
    ///
    /// # Errors
    ///
    /// Returns an error if the fused pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn qk_norm_rope_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        output: &MetalBuffer,
        weight: &MetalBuffer,
        head_dim: u32,
        num_heads: u32,
        eps: f32,
        position: u32,
        theta: f32,
        partial_rotary_factor: f32,
    ) -> crate::Result<()> {
        let pipeline = self.qk_norm_rope_pipeline.as_ref().ok_or_else(|| {
            Error::InvalidArgument("qk_norm_rope pipeline unavailable".to_owned())
        })?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(eps).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(position).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(theta).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(partial_rotary_factor)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<f32>(),
                7,
            );
        }
        let tg_size = head_dim.min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    // ── Residual Add & Scale ─────────────────────────────────────────────

    /// Dispatch elementwise FP16 addition: `output[i] = a[i] + b[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code)]
    pub fn residual_add(
        &self,
        ctx: &MetalContext,
        a: &MetalBuffer,
        b: &MetalBuffer,
        output: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.residual_add_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Scatter one new K/V row per lane into the unified by-lane KV pool, with
    /// each lane writing at its own `positions[lane]`. `k_src`/`v_src` are
    /// `[n_lanes, row]` f16; `k_cache`/`v_cache` are `[n_lanes, lane_capacity,
    /// row]` f16. One dispatch handles all lanes. See the
    /// `write_kv_cache_decode` shader.
    ///
    /// # Errors
    /// Returns an error if the `write_kv_cache_decode` pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn write_kv_cache_decode_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        k_src: &MetalBuffer,
        v_src: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        positions: &MetalBuffer,
        row: u32,
        lane_capacity: u32,
        n_lanes: u32,
    ) -> crate::Result<()> {
        let pipeline = self
            .write_kv_cache_decode_pipeline
            .as_ref()
            .ok_or_else(|| {
                Error::CommandBuffer("write_kv_cache_decode pipeline unavailable".to_owned())
            })?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(k_src.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(v_src.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k_cache.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v_cache.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(positions.raw()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(row).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(lane_capacity)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n_lanes).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
        }
        let threads_per_tg = 256_usize;
        let total = (n_lanes * row) as usize;
        let num_tg = total.div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Batched-decode flash attention over `n_lanes` independent sequences in
    /// one dispatch. `q`/`output` are `[n_lanes, num_q_heads, head_dim]` f16;
    /// `k`/`v` are the unified pool `[n_lanes, lane_capacity, num_kv_heads,
    /// head_dim]` f16; `positions` holds each lane's current absolute position.
    /// See the `flash_attention_decode_batched` shader.
    ///
    /// # Errors
    /// Returns an error if the `flash_attention_decode_batched` pipeline is
    /// unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_decode_batched_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        positions: &MetalBuffer,
        head_dim: u32,
        lane_capacity: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        window: u32,
        n_lanes: u32,
    ) -> crate::Result<()> {
        let pipeline = self
            .flash_attention_decode_batched_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("flash_attention_decode_batched".into()))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(positions.raw()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(lane_capacity)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_q_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(window).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n_lanes).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                10,
            );
        }
        let tg_width = 256usize; // FLASH_NSG(8) simdgroups * 32 lanes
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (num_q_heads * n_lanes) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Like [`residual_add`](Self::residual_add) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code)]
    pub fn residual_add_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        a: &MetalBuffer,
        b: &MetalBuffer,
        output: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.residual_add_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Dispatch in-place FP16 scaling: `data[i] *= scalar`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code)]
    pub fn scale_in_place_gpu(
        &self,
        ctx: &MetalContext,
        data: &MetalBuffer,
        scalar: f32,
        count: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.scale_in_place_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(data.raw()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(scalar).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// In-place Gemma final-logit soft-capping over an f32 buffer:
    /// `data[i] = cap * tanh(data[i] / cap)`. Commits and waits, so the CPU
    /// sampler can read the softcapped logits directly afterward.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code)]
    pub fn logit_softcap_gpu(
        &self,
        ctx: &MetalContext,
        data: &MetalBuffer,
        cap: f32,
        count: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.logit_softcap_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(data.raw()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cap).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`logit_softcap_gpu`](Self::logit_softcap_gpu) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code)]
    pub fn logit_softcap_gpu_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        data: &MetalBuffer,
        cap: f32,
        count: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.logit_softcap_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(data.raw()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cap).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Strided f16 gather into a [`CommandBatch`](crate::batch::CommandBatch):
    /// `dst[i*pld + p] = src[i*row_stride + base_off + p]` for
    /// `i in 0..n_rows`, `p in 0..pld`. Extracts one layer's PLE slice from the
    /// packed `[n_rows × n_layer × pld]` batch tensor on-device.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn gather_strided_f16_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        row_stride: u32,
        base_off: u32,
        pld: u32,
        n_rows: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gather_strided_pipeline);
        let count = pld * n_rows;
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.raw()), 0, 1);
            for (i, v) in [row_stride, base_off, pld, n_rows].into_iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(std::ptr::addr_of!(v).cast_mut().cast::<c_void>()),
                    size_of::<u32>(),
                    2 + i,
                );
            }
        }
        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Like [`scale_in_place_gpu`](Self::scale_in_place_gpu) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code)]
    pub fn scale_in_place_gpu_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        data: &MetalBuffer,
        scalar: f32,
        count: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.scale_in_place_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(data.raw()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(scalar).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
        }

        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    // ── Vision Patch Embedding ─────────────────────────────────────────────

    /// Dispatch vision patch embedding: `Conv2D` with `kernel_size=stride=patch_size`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn vision_patch_embed(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        bias: &MetalBuffer,
        output: &MetalBuffer,
        hidden_size: u32,
        patch_size: u32,
        img_h: u32,
        img_w: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        let pipeline = self
            .vision_patch_embed_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("vision_patch_embed".into()))?;
        encoder.setComputePipelineState(pipeline);

        let num_patches_w = img_w / patch_size;
        let num_patches_h = img_h / patch_size;
        let num_patches = num_patches_w * num_patches_h;

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(bias.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(hidden_size).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(patch_size).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(img_h).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(img_w).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_patches_w)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<u32>(),
                8,
            );
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: hidden_size as usize,
                height: num_patches as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Add Gemma 4 learned 2D vision position embeddings in-place.
    ///
    /// Hidden states layout: `[patch_rows * patch_cols, hidden_size]` FP16.
    /// Position table layout: `[2, position_count, hidden_size]` FP16 in
    /// contiguous GGUF order.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn vision_add_position_embedding(
        &self,
        ctx: &MetalContext,
        hidden_states: &MetalBuffer,
        position_table: &MetalBuffer,
        hidden_size: u32,
        position_count: u32,
        patch_rows: u32,
        patch_cols: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        let pipeline = self
            .vision_add_position_embedding_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("vision_add_position_embedding".into()))?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden_states.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(position_table.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(hidden_size).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(position_count)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(patch_rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(patch_cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: hidden_size as usize,
                height: (patch_rows * patch_cols) as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    // ── Clipped Linear / MatMul+Bias ───────────────────────────────────────

    /// Dispatch clipped linear: `y = clamp(clamp(x, in_min, in_max) @ W^T + bias, out_min, out_max)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn clipped_linear(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        bias: Option<&MetalBuffer>,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
        input_min: f32,
        input_max: f32,
        output_min: f32,
        output_max: f32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.clipped_linear_pipeline);

        let has_bias: u32 = u32::from(bias.is_some());

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            if let Some(b) = bias {
                encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 2);
            } else {
                encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 2); // dummy
            }
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(input_min).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(input_max).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(output_min).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(output_max).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                10,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(has_bias).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                11,
            );
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n as usize,
                height: m as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`clipped_linear`](Self::clipped_linear) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn clipped_linear_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        bias: Option<&MetalBuffer>,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
        input_min: f32,
        input_max: f32,
        output_min: f32,
        output_max: f32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.clipped_linear_pipeline);

        let has_bias: u32 = u32::from(bias.is_some());

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            if let Some(b) = bias {
                encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 2);
            } else {
                encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 2);
            }
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(input_min).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(input_max).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(output_min).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(output_max).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                10,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(has_bias).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                11,
            );
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n as usize,
                height: m as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Tiled clipped linear for large `[M, K] × [N, K]ᵀ` projections.
    /// This preserves the same clamp semantics as [`Self::clipped_linear_into`]
    /// but uses 32×32 threadgroup tiles instead of one serial dot product per
    /// output element.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn clipped_linear_tiled_nt_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        bias: Option<&MetalBuffer>,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
        input_min: f32,
        input_max: f32,
        output_min: f32,
        output_max: f32,
    ) -> crate::Result<()> {
        let pipeline = self
            .clipped_linear_f16_nt_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("clipped_linear_f16_nt".into()))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        let has_bias: u32 = u32::from(bias.is_some());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            if let Some(b) = bias {
                encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 2);
            } else {
                encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 2);
            }
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(input_min).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(input_max).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(output_min).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(output_max).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                10,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(has_bias).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                11,
            );
        }
        let tile = 32_usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(tile),
                height: (m as usize).div_ceil(tile),
                depth: 1,
            },
            MTLSize {
                width: tile,
                height: tile,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Dispatch matmul with optional bias (no clamping): y = x @ W^T + bias.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matmul_f16_bias(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        bias: Option<&MetalBuffer>,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(&self.matmul_f16_bias_pipeline);

        let has_bias: u32 = u32::from(bias.is_some());

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            if let Some(b) = bias {
                encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 2);
            } else {
                encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 2);
            }
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(has_bias).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n as usize,
                height: m as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`matmul_f16_bias`](Self::matmul_f16_bias) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matmul_f16_bias_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        bias: Option<&MetalBuffer>,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        encoder.setComputePipelineState(&self.matmul_f16_bias_pipeline);

        let has_bias: u32 = u32::from(bias.is_some());

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            if let Some(b) = bias {
                encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 2);
            } else {
                encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 2);
            }
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(has_bias).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n as usize,
                height: m as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    // ── Bidirectional Attention ─────────────────────────────────────────────

    // ── Vision Average Pooling 2D ──────────────────────────────────────────

    /// Dispatch 2D average pooling for vision patch grid.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn vision_avg_pool_2d(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        output: &MetalBuffer,
        hidden_size: u32,
        in_rows: u32,
        in_cols: u32,
        kernel_size: u32,
        stride: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        let pipeline = self
            .vision_avg_pool_2d_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("vision_avg_pool_2d".into()))?;
        encoder.setComputePipelineState(pipeline);

        let out_rows = (in_rows - kernel_size) / stride + 1;
        let out_cols = (in_cols - kernel_size) / stride + 1;
        let out_patches = out_rows * out_cols;

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(hidden_size).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(in_rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(in_cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kernel_size).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(stride).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
        }

        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: hidden_size as usize,
                height: out_patches as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    // ── Vision Batched RoPE ────────────────────────────────────────────────

    /// Dispatch Gemma 4 multidimensional vision `RoPE` in-place for Q and K.
    ///
    /// Q/K layout: `[seq_len, num_heads * head_dim]` FP16 (interleaved even/odd pairs).
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] on encoding failure.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn vision_rope(
        &self,
        ctx: &MetalContext,
        q: &MetalBuffer,
        k: &MetalBuffer,
        head_dim: u32,
        theta: f32,
        num_heads: u32,
        seq_len: u32,
        patch_rows: u32,
        patch_cols: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        let pipeline = self
            .vision_rope_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("vision_rope".into()))?;
        encoder.setComputePipelineState(pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(theta).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(patch_rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(patch_cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
        }

        let spatial_pairs = (head_dim / 2) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: spatial_pairs,
                height: num_heads as usize,
                depth: seq_len as usize,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Like [`vision_rope`](Self::vision_rope) but encodes into a [`CommandBatch`](crate::batch::CommandBatch).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn vision_rope_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        head_dim: u32,
        theta: f32,
        num_heads: u32,
        seq_len: u32,
        patch_rows: u32,
        patch_cols: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();

        let pipeline = self
            .vision_rope_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("vision_rope".into()))?;
        encoder.setComputePipelineState(pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(theta).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(patch_rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(patch_cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
        }

        let spatial_pairs = (head_dim / 2) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: spatial_pairs,
                height: num_heads as usize,
                depth: seq_len as usize,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Dispatch Gemma 4 audio subsampling conv + channel `LayerNorm` + `ReLU`.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn audio_subsample_conv2d_ln_relu(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        norm_weight: &MetalBuffer,
        output: &MetalBuffer,
        in_channels: u32,
        out_channels: u32,
        in_time: u32,
        in_freq: u32,
        out_time: u32,
        out_freq: u32,
        eps: f32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        let pipeline = self
            .audio_subsample_conv2d_ln_relu_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("audio_subsample_conv2d_ln_relu".into()))?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(norm_weight.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(in_channels).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(out_channels).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(in_time).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(in_freq).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(out_time).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(out_freq).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(eps).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                10,
            );
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: out_time as usize,
                height: out_freq as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Pack Gemma 4 audio frontend output from `[C, T, F]` to `[T, F*C]`.
    #[allow(unsafe_code)]
    pub fn audio_pack_frontend(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        output: &MetalBuffer,
        channels: u32,
        time: u32,
        freq: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        let pipeline = self
            .audio_pack_frontend_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("audio_pack_frontend".into()))?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(channels).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(time).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(freq).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: time as usize,
                height: (freq * channels) as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch Gemma 4 audio chunked local attention for one batch.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn audio_chunked_attention(
        &self,
        ctx: &MetalContext,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        rel_k: &MetalBuffer,
        per_dim_scale: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        chunk_size: u32,
        left_context: u32,
        softcap: f32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        let pipeline = self
            .audio_chunked_attention_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("audio_chunked_attention".into()))?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(rel_k.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(per_dim_scale.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 5);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(seq_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(chunk_size).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(left_context).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                10,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(softcap).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                11,
            );
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: head_dim as usize,
                height: seq_len as usize,
                depth: num_heads as usize,
            },
            MTLSize {
                width: head_dim.min(128) as usize,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch GLU over `[rows, hidden * 2] -> [rows, hidden]`.
    #[allow(unsafe_code)]
    pub fn audio_glu(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        hidden: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        let pipeline = self
            .audio_glu_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("audio_glu".into()))?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(hidden).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: hidden as usize,
                height: rows as usize,
                depth: 1,
            },
            MTLSize {
                width: hidden.min(256) as usize,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    // ── IQ Generic Dispatch ────────────────────────────────────────────────

    // ── Batch Prefill Kernels ──────────────────────────────────────────────

    /// Dispatch Flash Attention v2 for batch prefill.
    ///
    /// `window`: sliding-window size — a query at global position `q` attends
    /// keys `k` with `q - window < k <= q`; `0` means unlimited.
    ///
    /// # Panics
    ///
    /// Does not panic; unavailable pipelines are returned as errors.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_prefill_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        kv_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
        window: u32,
    ) -> crate::Result<()> {
        // Kernel selection (fastest → fallback):
        //  - MQA kernel: one threadgroup per query block processes ALL heads,
        //    loading each K/V tile once and reusing it across heads AND rows.
        //    Only valid for MQA (num_kv_heads == 1) with num_heads <= 8.
        //    Gated by `LOCAL_AI_FLASH_MQA` (default on). Grid omits the head
        //    dimension (one TG per q_block).
        //  - Tiled kernel: one TG per (q_block, head); K/V tile reused across
        //    the block's BR rows. Gated by `LOCAL_AI_FLASH_TILED`.
        //  - Per-row kernel: original fallback, re-reads K/V per query row.
        let use_mqa = Self::flash_mqa_enabled()
            && num_kv_heads == 1
            && num_heads <= 8
            && self.flash_attn_prefill_mqa_pipeline.is_some();
        let tiled = (!use_mqa && Self::flash_tiled_enabled())
            .then_some(self.flash_attn_prefill_tiled_pipeline.as_ref())
            .flatten();

        let (pipeline, tg_width, br, heads_per_tg) = if use_mqa {
            let p = self
                .flash_attn_prefill_mqa_pipeline
                .as_ref()
                .ok_or_else(|| Error::ShaderNotFound("flash_attention_prefill_mqa".into()))?;
            (p, 8 * 32_usize, 8_usize, 1_usize)
        } else if let Some(p) = tiled {
            (p, 8 * 32_usize, 8_usize, num_heads as usize)
        } else {
            let p = self
                .flash_attn_prefill_pipeline
                .as_ref()
                .ok_or_else(|| Error::ShaderNotFound("flash_attention_prefill".into()))?;
            (
                p,
                (head_dim as usize).min(256),
                32_usize,
                num_heads as usize,
            )
        };
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(seq_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(scale).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                9,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(window).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                10,
            );
        }
        let total_q_blocks = (seq_len as usize).div_ceil(br);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: total_q_blocks * heads_per_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Whether the tiled flash-attention prefill kernel is enabled
    /// (`LOCAL_AI_FLASH_TILED` != "0", default on).
    #[must_use]
    pub fn flash_tiled_enabled() -> bool {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("LOCAL_AI_FLASH_TILED").ok().as_deref() != Some("0"))
    }

    /// Whether the MQA multi-head flash-attention prefill kernel is enabled
    /// (`LOCAL_AI_FLASH_MQA` == "1", default **off**). This kernel processes all
    /// heads in one threadgroup to cut K/V device traffic by `num_heads`×, but
    /// holding per-head online-softmax accumulators in registers
    /// (`[num_heads][head_dim/32]`) blows up register pressure — at `head_dim=512`
    /// that is 128 floats/lane for `acc` alone — which collapses occupancy and
    /// makes it ~1.8× *slower* than the per-head tiled kernel on this model
    /// (5.6 s vs 3.1 s / 936 tokens). After row-tiling, prefill attention is
    /// compute/occupancy-bound, not K/V-bandwidth-bound, so this avenue loses.
    /// Kept opt-in for the record; the tiled kernel is the default.
    #[must_use]
    pub fn flash_mqa_enabled() -> bool {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("LOCAL_AI_FLASH_MQA").ok().as_deref() == Some("1"))
    }

    /// Flash attention prefill with an explicit additive mask buffer.
    ///
    /// The mask is `[seq_len, seq_len]` FP16 where `0.0` = attend, `-inf` = block.
    /// This is used for vision models where image tokens attend bidirectionally.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_prefill_masked_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        mask: &MetalBuffer,
        seq_len: u32,
        kv_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> crate::Result<()> {
        let pipeline = self
            .flash_attn_prefill_masked_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("flash_attention_prefill_masked".into()))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(seq_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kv_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_kv_heads).cast_mut().cast::<c_void>(),
                ),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(scale).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                9,
            );
            encoder.setBuffer_offset_atIndex(Some(mask.raw()), 0, 10);
        }
        let br = 32_usize;
        let tg_width = (head_dim as usize).min(256);
        let total_q_blocks = (seq_len as usize).div_ceil(br);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: total_q_blocks * num_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Dispatch batch `RoPE` in-place on [`seq_len`, `num_heads`, `head_dim`].
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn rope_batch_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        qk: &MetalBuffer,
        start_pos: u32,
        theta: f32,
        head_dim: u32,
        num_heads: u32,
        seq_len: u32,
        partial_rotary_factor: f32,
    ) -> crate::Result<()> {
        let pipeline = self
            .rope_batch_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("rope_batch".into()))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(qk.raw()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(start_pos).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(theta).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(seq_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(partial_rotary_factor)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<f32>(),
                6,
            );
        }
        let half_dim = (head_dim / 2) as usize;
        let tg_width = half_dim.min(256);
        let total = num_heads as usize * seq_len as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: half_dim.div_ceil(tg_width),
                height: total,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Batched-**decode** `RoPE`: like [`rope_batch_into`](Self::rope_batch_into)
    /// but each row (lane) reads its own absolute position from `positions`
    /// (`seq_len` u32 entries) instead of `start_pos + row`. Required to decode
    /// N independent sequences — each at a different context length — in a
    /// single batched forward.
    ///
    /// # Errors
    /// Returns [`Error::ShaderNotFound`] if the `rope_batch_decode` pipeline is
    /// unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn rope_batch_decode_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        qk: &MetalBuffer,
        positions: &MetalBuffer,
        theta: f32,
        head_dim: u32,
        num_heads: u32,
        seq_len: u32,
        partial_rotary_factor: f32,
    ) -> crate::Result<()> {
        let pipeline = self
            .rope_batch_decode_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("rope_batch_decode".into()))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(qk.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(positions.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(theta).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(seq_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(partial_rotary_factor)
                        .cast_mut()
                        .cast::<c_void>(),
                ),
                size_of::<f32>(),
                6,
            );
        }
        let half_dim = (head_dim / 2) as usize;
        let tg_width = half_dim.min(256);
        let total = num_heads as usize * seq_len as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: half_dim.div_ceil(tg_width),
                height: total,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    // ── Mamba-2 SSM Step ───────────────────────────────────────────────────

    // ── M-RoPE ─────────────────────────────────────────────────────────────

    // ── Gated Attention Output ─────────────────────────────────────────────

    // ── Top-K MoE Routing ──────────────────────────────────────────────────

    // ── BF16 → FP16 Conversion ────────────────────────────────────────────

    /// Convert a BF16 tensor to FP16 on the GPU.
    ///
    /// `src` contains `num_elements` BF16 values (2 bytes each, stored as `ushort`).
    /// `dst` receives `num_elements` FP16 values (2 bytes each).
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline is missing or command buffer creation fails.
    #[allow(unsafe_code)]
    pub fn bf16_to_fp16(
        &self,
        ctx: &MetalContext,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        num_elements: u32,
    ) -> crate::Result<()> {
        let pipeline = self
            .bf16_to_fp16_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("bf16_to_fp16".to_owned()))?;
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.raw()), 0, 1);
        }

        let threads_per_tg = 256_usize;
        let num_tg = (num_elements as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    // ── Audio Conformer Kernels ────────────────────────────────────────────

    /// Dispatch 1D depthwise convolution with same-padding.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn depthwise_conv1d(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        channels: u32,
        kernel_size: u32,
    ) -> crate::Result<()> {
        let pipeline = self
            .depthwise_conv1d_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("depthwise_conv1d".into()))?;
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(seq_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(channels).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(kernel_size).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: channels as usize,
                height: seq_len as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch 1D batch normalization (inference mode).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn batch_norm_1d(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        running_mean: &MetalBuffer,
        running_var: &MetalBuffer,
        weight: &MetalBuffer,
        bias: &MetalBuffer,
        output: &MetalBuffer,
        seq_len: u32,
        channels: u32,
        eps: f32,
    ) -> crate::Result<()> {
        let pipeline = self
            .batch_norm_1d_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("batch_norm_1d".into()))?;
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(running_mean.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(running_var.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(bias.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 5);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(seq_len).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(channels).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(eps).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                8,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: channels as usize,
                height: seq_len as usize,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Dispatch in-place `SiLU` activation.
    #[allow(unsafe_code)]
    pub fn silu_inplace(
        &self,
        ctx: &MetalContext,
        data: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let pipeline = self
            .silu_inplace_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("silu_inplace".into()))?;
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(data.raw()), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                1,
            );
        }
        let threads_per_tg = 256_usize;
        let num_tg = (count as usize).div_ceil(threads_per_tg);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    // ── Conv1d Step (depthwise, single-token decode) ──────────────────────

    // ── Gated Delta Rule Step (GDR single-token decode) ─────────────────

    // ── RMSNorm Gated (norm * SiLU(gate)) ──────────────────────────────

    // ── RMSNorm Centered (1+w) ─────────────────────────────────────────

    /// Dequantize a `TurboQuant` KV cache (rotation + Lloyd–Max codes + norm)
    /// into FP16, applying the inverse randomized Hadamard transform on-GPU.
    ///
    /// One threadgroup per `(token, head)`; `head_dim` threads cooperate on the
    /// inverse FWHT. `head_dim` must be a power of two ≤ 512.
    ///
    /// * `packed` — `[num_tokens * n_kv_heads * code_bytes]` LSB-first codes
    /// * `norms` — `[num_tokens * n_kv_heads]` FP16 vector norms
    /// * `levels` — `[1 << bits]` f32 reconstruction levels (pre-scaled by `1/√d`)
    /// * `signs` — `[head_dim]` f32 `±1` rotation sign flips
    /// * `output` — `[num_tokens * n_kv_heads * head_dim]` FP16
    /// * `token_start` — first token to decode; tokens `[token_start, num_tokens)`
    ///   are written into their absolute output slots, leaving earlier slots
    ///   untouched. Sliding-window layers pass the window start so only the
    ///   active window is dequantized; full-attention layers pass `0`.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline is unavailable or encoding fails.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn dequantize_kv_turboquant(
        &self,
        ctx: &MetalContext,
        packed: &MetalBuffer,
        norms: &MetalBuffer,
        levels: &MetalBuffer,
        signs: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        n_kv_heads: u32,
        num_tokens: u32,
        bits: u32,
        token_start: u32,
        ring_capacity: u32,
    ) -> crate::Result<()> {
        #[repr(C)]
        struct TqDequantParams {
            head_dim: u32,
            n_kv_heads: u32,
            num_tokens: u32,
            bits: u32,
            code_bytes: u32,
            group_offset: u32,
            ring_capacity: u32,
        }

        debug_assert!(head_dim.is_power_of_two() && head_dim <= 512);
        let token_start = token_start.min(num_tokens);
        let pipeline = self
            .dequantize_kv_turboquant_pipeline
            .as_ref()
            .ok_or_else(|| {
                Error::PipelineCreation("dequantize_kv_turboquant pipeline not available".into())
            })?;

        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(pipeline);

        let params = TqDequantParams {
            head_dim,
            n_kv_heads,
            num_tokens,
            bits,
            code_bytes: (head_dim * bits).div_ceil(8),
            group_offset: token_start * n_kv_heads,
            ring_capacity,
        };

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(norms.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(levels.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(signs.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(params).cast_mut().cast::<c_void>()),
                size_of::<TqDequantParams>(),
                5,
            );
        }

        let groups = (n_kv_heads * (num_tokens - token_start)) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.max(1),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Encode a `TurboQuant` KV dequantization into an existing [`crate::batch::CommandBatch`]
    /// (same semantics as [`Self::dequantize_kv_turboquant`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn dequantize_kv_turboquant_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        packed: &MetalBuffer,
        norms: &MetalBuffer,
        levels: &MetalBuffer,
        signs: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        n_kv_heads: u32,
        num_tokens: u32,
        bits: u32,
        token_start: u32,
        ring_capacity: u32,
    ) -> crate::Result<()> {
        #[repr(C)]
        struct TqDequantParams {
            head_dim: u32,
            n_kv_heads: u32,
            num_tokens: u32,
            bits: u32,
            code_bytes: u32,
            group_offset: u32,
            ring_capacity: u32,
        }

        let token_start = token_start.min(num_tokens);
        let pipeline = self
            .dequantize_kv_turboquant_pipeline
            .as_ref()
            .ok_or_else(|| {
                Error::PipelineCreation("dequantize_kv_turboquant pipeline not available".into())
            })?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);

        let params = TqDequantParams {
            head_dim,
            n_kv_heads,
            num_tokens,
            bits,
            code_bytes: (head_dim * bits).div_ceil(8),
            group_offset: token_start * n_kv_heads,
            ring_capacity,
        };
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(norms.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(levels.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(signs.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(params).cast_mut().cast::<c_void>()),
                size_of::<TqDequantParams>(),
                5,
            );
        }
        let groups = (n_kv_heads * (num_tokens - token_start)) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.max(1),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// GPU `TurboQuant` KV **encode** — the inverse of
    /// [`Self::dequantize_kv_turboquant`]. Quantizes `num_tokens × n_kv_heads`
    /// freshly projected FP16 K (or V) vectors (`input`, read relative to
    /// element 0) into the packed code buffer + per-head `norms`, writing to the
    /// absolute slots starting at `position * n_kv_heads`. This keeps the KV
    /// write entirely on the GPU, removing the CPU readback + encode round-trip.
    ///
    /// `levels`/`signs` are the same codec buffers the decode kernel uses.
    ///
    /// # Errors
    ///
    /// Returns an error if the `encode_kv_turboquant` pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn encode_kv_turboquant_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        levels: &MetalBuffer,
        signs: &MetalBuffer,
        packed: &MetalBuffer,
        norms: &MetalBuffer,
        head_dim: u32,
        n_kv_heads: u32,
        num_tokens: u32,
        bits: u32,
        position: u32,
        ring_capacity: u32,
    ) -> crate::Result<()> {
        #[repr(C)]
        struct TqEncodeParams {
            head_dim: u32,
            n_kv_heads: u32,
            num_tokens: u32,
            bits: u32,
            code_bytes: u32,
            group_offset: u32,
            ring_capacity: u32,
        }

        debug_assert!(head_dim.is_power_of_two() && head_dim <= 512);
        let pipeline = self.encode_kv_turboquant_pipeline.as_ref().ok_or_else(|| {
            Error::PipelineCreation("encode_kv_turboquant pipeline not available".into())
        })?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);

        let params = TqEncodeParams {
            head_dim,
            n_kv_heads,
            num_tokens,
            bits,
            code_bytes: (head_dim * bits).div_ceil(8),
            group_offset: position * n_kv_heads,
            ring_capacity,
        };

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(levels.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(signs.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(packed.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(norms.raw()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(params).cast_mut().cast::<c_void>()),
                size_of::<TqEncodeParams>(),
                5,
            );
        }

        let groups = (n_kv_heads * num_tokens) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.max(1),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Batched per-lane `TurboQuant` encode + scatter into a unified by-lane pool
    /// (`encode_kv_turboquant_batched` shader). `input` is `[n_lanes, row]`
    /// (row = `n_kv_heads * head_dim`) f16; each lane's freshly projected K (or
    /// V) is encoded and written to its own pool slot at `positions[lane]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn encode_kv_turboquant_batched_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        levels: &MetalBuffer,
        signs: &MetalBuffer,
        packed: &MetalBuffer,
        norms: &MetalBuffer,
        positions: &MetalBuffer,
        head_dim: u32,
        n_kv_heads: u32,
        lane_capacity: u32,
        bits: u32,
        n_lanes: u32,
        ring_capacity: u32,
    ) -> crate::Result<()> {
        #[repr(C)]
        struct TqEncodeBatchedParams {
            head_dim: u32,
            n_kv_heads: u32,
            lane_capacity: u32,
            bits: u32,
            code_bytes: u32,
            n_lanes: u32,
            ring_capacity: u32,
        }

        debug_assert!(head_dim.is_power_of_two() && head_dim <= 512);
        let pipeline = self
            .encode_kv_turboquant_batched_pipeline
            .as_ref()
            .ok_or_else(|| {
                Error::PipelineCreation(
                    "encode_kv_turboquant_batched pipeline not available".into(),
                )
            })?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);

        let params = TqEncodeBatchedParams {
            head_dim,
            n_kv_heads,
            lane_capacity,
            bits,
            code_bytes: (head_dim * bits).div_ceil(8),
            n_lanes,
            ring_capacity,
        };

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(levels.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(signs.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(packed.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(norms.raw()), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(params).cast_mut().cast::<c_void>()),
                size_of::<TqEncodeBatchedParams>(),
                5,
            );
            encoder.setBuffer_offset_atIndex(Some(positions.raw()), 0, 6);
        }

        let groups = (n_lanes * n_kv_heads) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.max(1),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Standalone (own command buffer) variant of
    /// [`Self::encode_kv_turboquant_into`].
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline is unavailable or encoding fails.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_kv_turboquant(
        &self,
        ctx: &MetalContext,
        input: &MetalBuffer,
        levels: &MetalBuffer,
        signs: &MetalBuffer,
        packed: &MetalBuffer,
        norms: &MetalBuffer,
        head_dim: u32,
        n_kv_heads: u32,
        num_tokens: u32,
        bits: u32,
        position: u32,
        ring_capacity: u32,
    ) -> crate::Result<()> {
        let mut batch = crate::batch::CommandBatch::new(ctx)?;
        self.encode_kv_turboquant_into(
            &mut batch,
            input,
            levels,
            signs,
            packed,
            norms,
            head_dim,
            n_kv_heads,
            num_tokens,
            bits,
            position,
            ring_capacity,
        )?;
        batch.commit_and_wait()?;
        Ok(())
    }

    /// Randomized Hadamard rotation per head (the query rotation and its inverse
    /// for fused `TurboQuant` attention). `pre_sign`/`post_sign` select whether
    /// the `±1` rotation signs are applied before or after the butterflies;
    /// both directions scale by `1/√head_dim`. `input`/`output` are f32
    /// `[num_heads * head_dim]` and may alias.
    ///
    /// # Errors
    ///
    /// Returns an error if the `hadamard_rotate` pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn hadamard_rotate_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        signs: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        num_heads: u32,
        pre_sign: bool,
        post_sign: bool,
    ) -> crate::Result<()> {
        let pipeline = self.hadamard_rotate_pipeline.as_ref().ok_or_else(|| {
            Error::PipelineCreation("hadamard_rotate pipeline not available".into())
        })?;
        Self::hadamard_rotate_dispatch(
            batch, pipeline, input, signs, output, head_dim, num_heads, pre_sign, post_sign,
        );
        Ok(())
    }

    /// f16-input → f32-output Hadamard rotation: rotate the `RoPE`'d query
    /// (`half`) into the f32 layout the fused `TurboQuant` attention kernel
    /// reads, with no separate dtype-conversion pass.
    ///
    /// # Errors
    ///
    /// Returns an error if the `hadamard_rotate_hf` pipeline is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn hadamard_rotate_hf_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        signs: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        num_heads: u32,
        pre_sign: bool,
        post_sign: bool,
    ) -> crate::Result<()> {
        let pipeline = self.hadamard_rotate_hf_pipeline.as_ref().ok_or_else(|| {
            Error::PipelineCreation("hadamard_rotate_hf pipeline not available".into())
        })?;
        Self::hadamard_rotate_dispatch(
            batch, pipeline, input, signs, output, head_dim, num_heads, pre_sign, post_sign,
        );
        Ok(())
    }

    /// f32-input → f16-output Hadamard rotation: inverse-rotate the fused
    /// attention value accumulation (f32) straight into a `half` attention
    /// output buffer the output projection consumes.
    ///
    /// # Errors
    ///
    /// Returns an error if the `hadamard_rotate_fh` pipeline is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn hadamard_rotate_fh_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        signs: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        num_heads: u32,
        pre_sign: bool,
        post_sign: bool,
    ) -> crate::Result<()> {
        let pipeline = self.hadamard_rotate_fh_pipeline.as_ref().ok_or_else(|| {
            Error::PipelineCreation("hadamard_rotate_fh pipeline not available".into())
        })?;
        Self::hadamard_rotate_dispatch(
            batch, pipeline, input, signs, output, head_dim, num_heads, pre_sign, post_sign,
        );
        Ok(())
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    fn hadamard_rotate_dispatch(
        batch: &mut crate::batch::CommandBatch,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        input: &MetalBuffer,
        signs: &MetalBuffer,
        output: &MetalBuffer,
        head_dim: u32,
        num_heads: u32,
        pre_sign: bool,
        post_sign: bool,
    ) {
        debug_assert!(head_dim.is_power_of_two() && head_dim <= 512);
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        let pre: u32 = pre_sign.into();
        let post: u32 = post_sign.into();
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(signs.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(head_dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(num_heads).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(pre).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(post).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                6,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    /// Fused `TurboQuant` flash-attention decode: computes attention logits and
    /// the value accumulation directly from the packed K/V codes (no FP16
    /// expansion). `rq` is the pre-rotated query (see [`Self::hadamard_rotate_into`]
    /// with `pre_sign`); `out` receives the value accumulation in rotated space
    /// and must be inverse-rotated (`post_sign`) to obtain the attention output.
    /// `window == 0` disables the sliding window.
    ///
    /// # Errors
    ///
    /// Returns an error if the `flash_attention_tq` pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_tq_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        rq: &MetalBuffer,
        k_codes: &MetalBuffer,
        k_norms: &MetalBuffer,
        v_codes: &MetalBuffer,
        v_norms: &MetalBuffer,
        levels: &MetalBuffer,
        out: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        current_pos: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        window: u32,
        bits: u32,
        ring_capacity: u32,
    ) -> crate::Result<()> {
        let pipeline = self.flash_attention_tq_pipeline.as_ref().ok_or_else(|| {
            Error::PipelineCreation("flash_attention_tq pipeline not available".into())
        })?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        let code_bytes = (head_dim * bits).div_ceil(8);
        let scalars = [
            head_dim,
            kv_len,
            current_pos,
            num_q_heads,
            num_kv_heads,
            window,
            bits,
            code_bytes,
            ring_capacity,
        ];
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(rq.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_codes.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k_norms.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v_codes.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(v_norms.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(levels.raw()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(out.raw()), 0, 6);
            for (i, s) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(std::ptr::addr_of!(*s).cast_mut().cast::<c_void>()),
                    size_of::<u32>(),
                    7 + i,
                );
            }
        }
        let tg_width = 256usize; // FLASH_TQ_NSG (8) simdgroups per head
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_q_heads as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Fused `TurboQuant` flash-attention PREFILL / multi-row verify
    /// (`flash_attention_tq_prefill` shader): attends `n_rows` consecutive query
    /// rows (row `r` at absolute position `start_pos + r`) directly over the
    /// packed K/V codes, applying a per-row causal cutoff and sliding window, so
    /// the cache is never expanded to FP16. `rq` holds the rotated queries
    /// `[n_rows, num_q_heads, head_dim]` f32; `out` receives the rotated-space
    /// value accumulation in the same layout (inverse-rotate with
    /// `hadamard_rotate_fh` to recover the attention output). `kv_len` is the
    /// total cache length after this batch was written (`start_pos + n_rows`).
    ///
    /// # Errors
    ///
    /// Returns an error if the `flash_attention_tq_prefill` pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_tq_prefill_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        rq: &MetalBuffer,
        k_codes: &MetalBuffer,
        k_norms: &MetalBuffer,
        v_codes: &MetalBuffer,
        v_norms: &MetalBuffer,
        levels: &MetalBuffer,
        out: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        start_pos: u32,
        n_rows: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        window: u32,
        bits: u32,
        ring_capacity: u32,
    ) -> crate::Result<()> {
        let pipeline = self
            .flash_attention_tq_prefill_pipeline
            .as_ref()
            .ok_or_else(|| {
                Error::PipelineCreation("flash_attention_tq_prefill pipeline not available".into())
            })?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        let code_bytes = (head_dim * bits).div_ceil(8);
        let scalars = [
            head_dim,
            kv_len,
            start_pos,
            n_rows,
            num_q_heads,
            num_kv_heads,
            window,
            bits,
            code_bytes,
            ring_capacity,
        ];
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(rq.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_codes.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k_norms.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v_codes.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(v_norms.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(levels.raw()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(out.raw()), 0, 6);
            for (i, s) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(std::ptr::addr_of!(*s).cast_mut().cast::<c_void>()),
                    size_of::<u32>(),
                    7 + i,
                );
            }
        }
        let tg_width = 256usize; // FLASH_TQ_NSG (8) simdgroups per (row, head)
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (num_q_heads * n_rows) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Query-tiled fused `TurboQuant` flash-attention prefill
    /// (`flash_attention_tq_prefill_tiled` shader): same per-row causal,
    /// rotated-space result as [`Self::flash_attention_tq_prefill_into`], but
    /// each K/V code tile is dequantized into threadgroup memory once and reused
    /// across `BR = 8` query rows, cutting the quadratic K/V code bandwidth ~8×.
    /// Identical buffer/scalar contract to the non-tiled variant.
    ///
    /// # Errors
    ///
    /// Returns an error if the `flash_attention_tq_prefill_tiled` pipeline is
    /// unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_tq_prefill_tiled_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        rq: &MetalBuffer,
        k_codes: &MetalBuffer,
        k_norms: &MetalBuffer,
        v_codes: &MetalBuffer,
        v_norms: &MetalBuffer,
        levels: &MetalBuffer,
        out: &MetalBuffer,
        head_dim: u32,
        kv_len: u32,
        start_pos: u32,
        n_rows: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        window: u32,
        bits: u32,
        ring_capacity: u32,
    ) -> crate::Result<()> {
        const BR: u32 = 8; // FLASH_TQ_TILED_BR
        let pipeline = self
            .flash_attention_tq_prefill_tiled_pipeline
            .as_ref()
            .ok_or_else(|| {
                Error::PipelineCreation(
                    "flash_attention_tq_prefill_tiled pipeline not available".into(),
                )
            })?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        let code_bytes = (head_dim * bits).div_ceil(8);
        let scalars = [
            head_dim,
            kv_len,
            start_pos,
            n_rows,
            num_q_heads,
            num_kv_heads,
            window,
            bits,
            code_bytes,
            ring_capacity,
        ];
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(rq.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_codes.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k_norms.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v_codes.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(v_norms.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(levels.raw()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(out.raw()), 0, 6);
            for (i, s) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(std::ptr::addr_of!(*s).cast_mut().cast::<c_void>()),
                    size_of::<u32>(),
                    7 + i,
                );
            }
        }
        let q_blocks = n_rows.div_ceil(BR);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (num_q_heads * q_blocks) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: (BR * 32) as usize,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Batched fused `TurboQuant` flash-attention decode (`flash_attention_tq_batched`
    /// shader): one new query token per lane, K/V read from a unified by-lane
    /// pool. `rq` holds the per-lane rotated queries `[n_lanes, num_q_heads,
    /// head_dim]` f32; `out` receives the per-lane rotated-space value
    /// accumulation in the same layout (inverse-rotate with `hadamard_rotate` to
    /// recover the attention output). Turbo analogue of
    /// [`Self::flash_attention_decode_batched_into`].
    ///
    /// # Errors
    ///
    /// Returns an error if the `flash_attention_tq_batched` pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn flash_attention_tq_batched_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        rq: &MetalBuffer,
        k_codes: &MetalBuffer,
        k_norms: &MetalBuffer,
        v_codes: &MetalBuffer,
        v_norms: &MetalBuffer,
        levels: &MetalBuffer,
        out: &MetalBuffer,
        positions: &MetalBuffer,
        head_dim: u32,
        lane_capacity: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        window: u32,
        bits: u32,
        n_lanes: u32,
        ring_capacity: u32,
    ) -> crate::Result<()> {
        let pipeline = self
            .flash_attention_tq_batched_pipeline
            .as_ref()
            .ok_or_else(|| {
                Error::PipelineCreation("flash_attention_tq_batched pipeline not available".into())
            })?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        let code_bytes = (head_dim * bits).div_ceil(8);
        let scalars = [
            head_dim,
            lane_capacity,
            num_q_heads,
            num_kv_heads,
            window,
            bits,
            code_bytes,
            n_lanes,
            ring_capacity,
        ];
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(rq.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k_codes.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(k_norms.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v_codes.raw()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(v_norms.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(levels.raw()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(out.raw()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(positions.raw()), 0, 7);
            for (i, s) in scalars.iter().enumerate() {
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(std::ptr::addr_of!(*s).cast_mut().cast::<c_void>()),
                    size_of::<u32>(),
                    8 + i,
                );
            }
        }
        let tg_width = 256usize; // FLASH_TQ_NSG (8) simdgroups per head
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (num_q_heads * n_lanes) as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Encode a tiled `[M, K] × [N, K]ᵀ → [M, N]` matmul into a
    /// [`crate::batch::CommandBatch`]. `weight` is row-major `[N, K]` — the
    /// engine's standard `[out, in]` weight layout — so a batch of `m` token
    /// rows streams the weights once instead of `m` matvec passes.
    ///
    /// # Errors
    ///
    /// Returns an error if the `matmul_f16_nt` pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matmul_nt_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> crate::Result<()> {
        let pipeline = self
            .matmul_f16_nt_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("matmul_f16_nt".into()))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
        }
        let tile = 32_usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(tile),
                height: (m as usize).div_ceil(tile),
                depth: 1,
            },
            MTLSize {
                width: tile,
                height: tile,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Output rows reduced per threadgroup in the quantized matvec kernels
    /// (one SIMD-group per row). Must match `ROWS_PER_TG` in
    /// `shaders/matvec_quant.metal`.
    const MATVEC_ROWS_PER_TG: usize = 4;

    /// Output rows each SIMD-group reduces for the `TQ2_0` matvec. Must match
    /// `ROWS_PER_SG` in `shaders/matvec_quant.metal`. Tuned via the
    /// `matvec_decode_timing` microbench on M4 Pro. With the no-shift dequant
    /// (pre-scaled activations + single `-1` zero-point per row), the per-block
    /// activation prep is now the amortizable cost, so reusing it across a few
    /// rows wins: 1 row ~100 GB/s, 2 ~112, 3 ~114, 4 ~107. 3 is the sweet spot
    /// (beyond it register pressure drops occupancy). Must match `ROWS_PER_SG`
    /// in `shaders/matvec_quant.metal`.
    const MATVEC_TQ2_ROWS_PER_SG: usize = 3;

    /// Output rows each SIMD-group reduces for the `Q4_0` matvec. Must match
    /// `Q4_NR0` in `shaders/matvec_quant.metal` (llama.cpp-style multi-row
    /// kernel with `ushort` block loads and activation reuse).
    const MATVEC_Q4_NR0: usize = 4;

    /// Encode a quantized-weight matvec (`output = matrix × vector`) onto an
    /// encoder. One SIMD-group (32 lanes) reduces each output row and
    /// [`Self::MATVEC_ROWS_PER_TG`] rows share a threadgroup, so the grid is
    /// `ceil(rows / ROWS_PER_TG)` threadgroups of `ROWS_PER_TG * 32` threads.
    /// `units_per_row` is unused by the dispatch but kept so callers can
    /// validate block alignment at the call site.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    fn encode_matvec_quant(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
        _units_per_row: usize,
        rows_per_sg: usize,
    ) {
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(matrix.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(vector.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
        }
        let simdgroups_per_threadgroup = Self::MATVEC_ROWS_PER_TG;
        // Each threadgroup has `simdgroups_per_threadgroup` simdgroups; each simdgroup reduces
        // `rows_per_sg` output rows, so one threadgroup covers
        // `simdgroups_per_threadgroup * rows_per_sg` rows.
        let rows_per_group = simdgroups_per_threadgroup * rows_per_sg;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (rows as usize).div_ceil(rows_per_group),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: simdgroups_per_threadgroup * 32,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Dispatch a one-command-buffer quantized matvec and wait.
    #[allow(clippy::too_many_arguments)]
    fn run_matvec_quant(
        ctx: &MetalContext,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
        units_per_row: usize,
        rows_per_sg: usize,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        Self::encode_matvec_quant(
            pipeline,
            &encoder,
            matrix,
            vector,
            output,
            rows,
            cols,
            units_per_row,
            rows_per_sg,
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matvec_quant_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        batch: &mut crate::batch::CommandBatch,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
        units_per_row: usize,
        rows_per_sg: usize,
    ) {
        Self::encode_matvec_quant(
            pipeline,
            batch.encoder(),
            matrix,
            vector,
            output,
            rows,
            cols,
            units_per_row,
            rows_per_sg,
        );
        batch.record_dispatch();
    }

    /// `Q4_0`-weight matvec: `output[rows] (f16) = matrix[rows, cols] × vector[cols]`.
    ///
    /// `matrix` holds verbatim GGUF `Q4_0` blocks (18 bytes / 32 elems);
    /// `cols` must be a multiple of 32.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not block-aligned or encoding fails.
    pub fn matvec_q4_0(
        &self,
        ctx: &MetalContext,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        if !cols.is_multiple_of(32) {
            return Err(Error::InvalidArgument(format!(
                "matvec_q4_0: cols {cols} not /32"
            )));
        }
        let units = (cols as usize / 32) * 16;
        Self::run_matvec_quant(
            ctx,
            &self.matvec_q4_0_pipeline,
            matrix,
            vector,
            output,
            rows,
            cols,
            units,
            Self::MATVEC_Q4_NR0,
        )
    }

    /// Encode a `Q4_0`-weight matvec into a command batch.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not block-aligned.
    pub fn matvec_q4_0_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        if !cols.is_multiple_of(32) {
            return Err(Error::InvalidArgument(format!(
                "matvec_q4_0: cols {cols} not /32"
            )));
        }
        let units = (cols as usize / 32) * 16;
        Self::matvec_quant_into(
            &self.matvec_q4_0_pipeline,
            batch,
            matrix,
            vector,
            output,
            rows,
            cols,
            units,
            Self::MATVEC_Q4_NR0,
        );
        Ok(())
    }

    /// `Q4_0`-weight matvec with float32 output (tied-embedding logits).
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not block-aligned or encoding fails.
    pub fn matvec_q4_0_f32out(
        &self,
        ctx: &MetalContext,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        if !cols.is_multiple_of(32) {
            return Err(Error::InvalidArgument(format!(
                "matvec_q4_0_f32out: cols {cols} not /32"
            )));
        }
        let units = (cols as usize / 32) * 16;
        Self::run_matvec_quant(
            ctx,
            &self.matvec_q4_0_f32out_pipeline,
            matrix,
            vector,
            output,
            rows,
            cols,
            units,
            Self::MATVEC_Q4_NR0,
        )
    }

    /// Encode a `Q4_0`-weight matvec with float32 output into a command batch.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not block-aligned.
    pub fn matvec_q4_0_f32out_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        if !cols.is_multiple_of(32) {
            return Err(Error::InvalidArgument(format!(
                "matvec_q4_0_f32out: cols {cols} not /32"
            )));
        }
        let units = (cols as usize / 32) * 16;
        Self::matvec_quant_into(
            &self.matvec_q4_0_f32out_pipeline,
            batch,
            matrix,
            vector,
            output,
            rows,
            cols,
            units,
            Self::MATVEC_Q4_NR0,
        );
        Ok(())
    }

    /// `TQ2_0`-weight matvec: `output[rows] (f16) = matrix[rows, cols] × vector[cols]`.
    ///
    /// `matrix` holds verbatim GGUF `TQ2_0` blocks (66 bytes / 256 elems);
    /// `cols` must be a multiple of 256.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not block-aligned or encoding fails.
    pub fn matvec_tq2_0(
        &self,
        ctx: &MetalContext,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        if !cols.is_multiple_of(256) {
            return Err(Error::InvalidArgument(format!(
                "matvec_tq2_0: cols {cols} not /256"
            )));
        }
        let units = (cols as usize / 256) * 64;
        Self::run_matvec_quant(
            ctx,
            &self.matvec_tq2_0_pipeline,
            matrix,
            vector,
            output,
            rows,
            cols,
            units,
            Self::MATVEC_TQ2_ROWS_PER_SG,
        )
    }

    /// Encode a `TQ2_0`-weight matvec into a command batch.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not block-aligned.
    pub fn matvec_tq2_0_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        if !cols.is_multiple_of(256) {
            return Err(Error::InvalidArgument(format!(
                "matvec_tq2_0: cols {cols} not /256"
            )));
        }
        let units = (cols as usize / 256) * 64;
        Self::matvec_quant_into(
            &self.matvec_tq2_0_pipeline,
            batch,
            matrix,
            vector,
            output,
            rows,
            cols,
            units,
            Self::MATVEC_TQ2_ROWS_PER_SG,
        );
        Ok(())
    }

    /// Struct-of-arrays `TQ2_0` matvec: `qs` holds aligned 64-byte block
    /// payloads (`qs[(row*blocks+b)*64]`) and `scales` the fp16 per-block scales
    /// (`scales[row*blocks+b]`). See `matvec_tq2_0_soa` in the shader.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not a multiple of 256, encoding fails, or
    /// the split-buffer pipeline is unavailable.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn matvec_tq2_0_soa_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        qs: &MetalBuffer,
        scales: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        if !cols.is_multiple_of(256) {
            return Err(Error::InvalidArgument(format!(
                "matvec_tq2_0_soa: cols {cols} not /256"
            )));
        }
        let pipeline = self
            .matvec_tq2_0_soa_pipeline
            .as_ref()
            .ok_or_else(|| Error::ShaderNotFound("matvec_tq2_0_soa".into()))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(qs.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(vector.raw()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(rows).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(cols).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
        }
        let rows_per_tg = Self::MATVEC_ROWS_PER_TG;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (rows as usize).div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: rows_per_tg * 32,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// `TQ2_0`-weight matvec with float32 output (tied-embedding logits).
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not block-aligned or encoding fails.
    pub fn matvec_tq2_0_f32out(
        &self,
        ctx: &MetalContext,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        if !cols.is_multiple_of(256) {
            return Err(Error::InvalidArgument(format!(
                "matvec_tq2_0_f32out: cols {cols} not /256"
            )));
        }
        let units = (cols as usize / 256) * 64;
        Self::run_matvec_quant(
            ctx,
            &self.matvec_tq2_0_f32out_pipeline,
            matrix,
            vector,
            output,
            rows,
            cols,
            units,
            Self::MATVEC_TQ2_ROWS_PER_SG,
        )
    }

    /// Encode a `TQ2_0`-weight matvec with float32 output into a command batch.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is not block-aligned.
    pub fn matvec_tq2_0_f32out_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        matrix: &MetalBuffer,
        vector: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        cols: u32,
    ) -> crate::Result<()> {
        if !cols.is_multiple_of(256) {
            return Err(Error::InvalidArgument(format!(
                "matvec_tq2_0_f32out: cols {cols} not /256"
            )));
        }
        let units = (cols as usize / 256) * 64;
        Self::matvec_quant_into(
            &self.matvec_tq2_0_f32out_pipeline,
            batch,
            matrix,
            vector,
            output,
            rows,
            cols,
            units,
            Self::MATVEC_TQ2_ROWS_PER_SG,
        );
        Ok(())
    }

    /// Encode a tiled `[M, K] × [N, K]ᵀ → [M, N]` matmul with quantized
    /// weights into a [`crate::batch::CommandBatch`] — the quantized analogue
    /// of [`Self::matmul_nt_into`]. `weight` holds verbatim GGUF blocks of
    /// the given format (`Q4_0` needs `k % 32 == 0`, `TQ2_0` `k % 256 == 0`).
    ///
    /// # Errors
    ///
    /// Returns an error if `k` is not block-aligned for the format.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    fn matmul_nt_quant_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
        }
        let tile = 32_usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(tile),
                height: (m as usize).div_ceil(tile),
                depth: 1,
            },
            MTLSize {
                width: tile,
                height: tile,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    /// `MMA` (`simdgroup_matrix`) tiled quantized matmul dispatch. 64×32 output
    /// tile per threadgroup, 256 threads (8 simdgroups). Buffer/scalar binding
    /// matches [`Self::matmul_nt_quant_into`].
    #[allow(unsafe_code, clippy::too_many_arguments)]
    fn matmul_nt_quant_mma_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
        }
        // 64×32 output tile (BM×BN), 8 simdgroups × 32 lanes. Must match
        // MMA_BM/MMA_BN/MMA_THREADS in `shaders/matvec_quant.metal`.
        let tile_m = 64_usize;
        let tile_n = 32_usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(tile_n),
                height: (m as usize).div_ceil(tile_m),
                depth: 1,
            },
            MTLSize {
                width: 256, // 8 simdgroups × 32 lanes
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    /// Largest M routed to the small-M `Q4_0` matmul (`matmul_nt_q4_0_smallm`).
    /// Its 8×64 output tile pads M only to 8, so it is the fast path for small
    /// batched-decode batches (M≈2–7); larger M go to the 64×32 MMA kernel.
    pub const SMALLM_MAX_M: u32 = 8;
    /// Upper M bound for the batched fused-matvec (`qmv`-style) `TQ2_0` path. It
    /// beats the MMA small-M kernel for M≤3 and ties at M=4 (measured on e2b
    /// TQ2 verify); above that its per-M scalar ALU loses to the MMA tile, so
    /// M≥5 falls through to `matmul_nt_tq2_0_smallm`.
    pub const BATCHM_MAX_M: u32 = 4;

    /// Dispatch a small-M quantized matmul (`Q4_0` or `TQ2_0`): `BM=8, BN=64`,
    /// 8 simdgroups (256 threads) per threadgroup. Grid is
    /// `(ceil(N/64), ceil(M/8))`. The buffer/scalar binding and grid are
    /// identical across the `Q4_0` (`BK=32`) and `TQ2_0` (`BK=128`) small-M
    /// kernels, so both route through this helper with their own pipeline.
    /// Buffer/scalar binding matches [`Self::matmul_nt_quant_mma_into`].
    // Metal kernels require one buffer/scalar binding per shader argument.
    #[allow(unsafe_code)]
    #[allow(clippy::too_many_arguments)]
    fn matmul_nt_quant_smallm_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(64),
                height: (m as usize).div_ceil(8),
                depth: 1,
            },
            MTLSize {
                width: 256, // 8 simdgroups × 32 lanes
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    /// Batched fused `TQ2_0` matvec dispatch (small M). Same grid as the quantized
    /// matvec — one simdgroup per `BATCHM_R`(=2) weight rows, `ROWS_PER_TG`(=4)
    /// simdgroups per threadgroup → 8 weight rows/TG, 128 threads/TG — so the
    /// quant weight streams once and is reused across all `m` input vectors
    /// (mirrors MLX's `qmv` small-M routing). Buffer/scalar binding matches
    /// [`Self::matmul_nt_quant_smallm_into`].
    #[allow(unsafe_code, clippy::too_many_arguments)]
    fn matmul_nt_quant_batchm_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
        }
        // ROWS_PER_TG(4) × BATCHM_R(2) = 8 weight rows per threadgroup.
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(8),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128, // 4 simdgroups × 32 lanes
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    /// Whether the batched-matvec small-M `TQ2_0` path is enabled (cached read of
    /// `LOCAL_AI_BATCHM`; defaults on). Set `LOCAL_AI_BATCHM=0` to fall back to
    /// the MMA small-M kernel for A/B comparison.
    #[must_use]
    pub fn batchm_enabled() -> bool {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("LOCAL_AI_BATCHM").ok().as_deref() != Some("0"))
    }

    /// Whether the MMA matmul path is enabled (cached read of `LOCAL_AI_MMA`;
    /// defaults on). Set `LOCAL_AI_MMA=0` to force the scalar tiled / multivec
    /// kernels. The `simdgroup_matrix` GEMM beats both the scalar tiled kernel
    /// (M>16) and the multivec GEMV-style kernel (M≤16) at every batch size
    /// measured (see `matmul_prefill_timing`), so callers route all batch
    /// matmuls through it when enabled.
    #[must_use]
    pub fn mma_enabled() -> bool {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("LOCAL_AI_MMA").ok().as_deref() != Some("0"))
    }

    /// Whether the small-M `Q4_0` matmul fast path is enabled (cached read of
    /// `LOCAL_AI_SMALLM`; defaults on). Set `LOCAL_AI_SMALLM=0` to fall back to
    /// the 64×32 MMA / scalar tiled kernels for A/B comparison.
    #[must_use]
    pub fn smallm_enabled() -> bool {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("LOCAL_AI_SMALLM").ok().as_deref() != Some("0"))
    }

    /// Small-batch multi-vector matvec dispatch: one SIMD-group per weight row
    /// (`n`), [`Self::MATVEC_ROWS_PER_TG`] rows per threadgroup — the same grid
    /// as the quantized matvec, so the weight streams once and is reused across
    /// all `m` input vectors. Buffer/scalar binding matches
    /// [`Self::matmul_nt_quant_into`].
    #[allow(unsafe_code, clippy::too_many_arguments)]
    fn multivec_nt_quant_into(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(m).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(k).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(n).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                5,
            );
        }
        let rows_per_tg = Self::MATVEC_ROWS_PER_TG;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: rows_per_tg * 32,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    /// Largest batch the multi-vector matvec path accepts; must match
    /// `MULTIVEC_MAX_M` in `shaders/matvec_quant.metal`.
    pub const MULTIVEC_MAX_M: u32 = 16;

    /// Output rows packed per threadgroup by the FP16 matvec kernels
    /// (`matvec_f16`, `matvec_f16w_f32out`, `matvec_f16w_f32in`). Each row is
    /// owned by one simdgroup (32 lanes), so a threadgroup runs
    /// `F16_MATVEC_ROWS_PER_TG * 32` threads and the grid uses
    /// `rows.div_ceil(F16_MATVEC_ROWS_PER_TG)` threadgroups.
    const F16_MATVEC_ROWS_PER_TG: usize = 8;

    /// `Q4_0`-weight small-batch multi-vector matvec into a command batch
    /// (`m <= MULTIVEC_MAX_M`).
    ///
    /// # Errors
    ///
    /// Returns an error if `k` is not a multiple of 32 or `m` is out of range.
    #[allow(clippy::too_many_arguments)]
    pub fn multivec_nt_q4_0_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> crate::Result<()> {
        if !k.is_multiple_of(32) {
            return Err(Error::InvalidArgument(format!(
                "multivec_nt_q4_0: k {k} not /32"
            )));
        }
        if m == 0 || m > Self::MULTIVEC_MAX_M {
            return Err(Error::InvalidArgument(format!(
                "multivec_nt_q4_0: m {m} out of 1..={}",
                Self::MULTIVEC_MAX_M
            )));
        }
        Self::multivec_nt_quant_into(
            &self.multivec_nt_q4_0_pipeline,
            batch,
            input,
            weight,
            output,
            m,
            k,
            n,
        );
        Ok(())
    }

    /// `TQ2_0`-weight small-batch multi-vector matvec into a command batch
    /// (`m <= MULTIVEC_MAX_M`).
    ///
    /// # Errors
    ///
    /// Returns an error if `k` is not a multiple of 256 or `m` is out of range.
    #[allow(clippy::too_many_arguments)]
    pub fn multivec_nt_tq2_0_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> crate::Result<()> {
        if !k.is_multiple_of(256) {
            return Err(Error::InvalidArgument(format!(
                "multivec_nt_tq2_0: k {k} not /256"
            )));
        }
        if m == 0 || m > Self::MULTIVEC_MAX_M {
            return Err(Error::InvalidArgument(format!(
                "multivec_nt_tq2_0: m {m} out of 1..={}",
                Self::MULTIVEC_MAX_M
            )));
        }
        Self::multivec_nt_quant_into(
            &self.multivec_nt_tq2_0_pipeline,
            batch,
            input,
            weight,
            output,
            m,
            k,
            n,
        );
        Ok(())
    }

    /// `Q4_0`-weight batched matmul into a command batch.
    ///
    /// # Errors
    ///
    /// Returns an error if `k` is not a multiple of 32.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_nt_q4_0_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> crate::Result<()> {
        if !k.is_multiple_of(32) {
            return Err(Error::InvalidArgument(format!(
                "matmul_nt_q4_0: k {k} not /32"
            )));
        }
        // Small-M fast path: pads M only to 8 (vs the 64×32 MMA kernel) and
        // dequantizes each Q4_0 block once into a half threadgroup tile. This
        // is the small batched-decode batch (M≈2–7). Opt out with LOCAL_AI_SMALLM=0.
        if m <= Self::SMALLM_MAX_M
            && Self::smallm_enabled()
            && let Some(sm) = self.matmul_nt_q4_0_smallm_pipeline.as_ref()
        {
            Self::matmul_nt_quant_smallm_into(sm, batch, input, weight, output, m, k, n);
            return Ok(());
        }
        if Self::mma_enabled()
            && let Some(mma) = self.matmul_nt_q4_0_mma_pipeline.as_ref()
        {
            Self::matmul_nt_quant_mma_into(mma, batch, input, weight, output, m, k, n);
            return Ok(());
        }
        Self::matmul_nt_quant_into(
            &self.matmul_nt_q4_0_pipeline,
            batch,
            input,
            weight,
            output,
            m,
            k,
            n,
        );
        Ok(())
    }

    /// `TQ2_0`-weight batched matmul into a command batch.
    ///
    /// # Errors
    ///
    /// Returns an error if `k` is not a multiple of 256.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_nt_tq2_0_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> crate::Result<()> {
        if !k.is_multiple_of(256) {
            return Err(Error::InvalidArgument(format!(
                "matmul_nt_tq2_0: k {k} not /256"
            )));
        }
        // Small-M fast path #1 (batched fused matvec): keeps the single-token
        // matvec's fused no-shift dequant and coalesced weight read, reusing the
        // streamed-once quant weights across all M activation vectors — no
        // threadgroup round-trip or barriers. This is the continuous-batch
        // small-M batch (M≈2–8) and is the MLX `qmv`-style routing.
        // Opt out with LOCAL_AI_BATCHM=0 (falls through to the MMA small-M path).
        if m <= Self::BATCHM_MAX_M
            && Self::batchm_enabled()
            && let Some(bm) = self.matmul_nt_tq2_0_batchm_pipeline.as_ref()
        {
            Self::matmul_nt_quant_batchm_into(bm, batch, input, weight, output, m, k, n);
            return Ok(());
        }
        // Small-M fast path #2 (MMA): dequantizes each TQ2_0 half-block once into
        // a half threadgroup tile, then simdgroup-MMA. Opt out: LOCAL_AI_SMALLM=0.
        if m <= Self::SMALLM_MAX_M
            && Self::smallm_enabled()
            && let Some(sm) = self.matmul_nt_tq2_0_smallm_pipeline.as_ref()
        {
            Self::matmul_nt_quant_smallm_into(sm, batch, input, weight, output, m, k, n);
            return Ok(());
        }
        if Self::mma_enabled()
            && let Some(mma) = self.matmul_nt_tq2_0_mma_pipeline.as_ref()
        {
            Self::matmul_nt_quant_mma_into(mma, batch, input, weight, output, m, k, n);
            return Ok(());
        }
        Self::matmul_nt_quant_into(
            &self.matmul_nt_tq2_0_pipeline,
            batch,
            input,
            weight,
            output,
            m,
            k,
            n,
        );
        Ok(())
    }

    /// Encode GELU (tanh approximation) into a [`crate::batch::CommandBatch`].
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Result` for API symmetry.
    #[allow(unsafe_code)]
    pub fn gelu_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        output: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gelu_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                2,
            );
        }
        let threads_per_tg = 256_usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (count as usize).div_ceil(threads_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Fused `GeGLU` elementwise `out[i] = gelu(a[i]) * b[i]`, reading `b` at
    /// `b_offset` bytes (use 0 for no offset). Replaces a `gelu` + `mul` pair.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn gelu_mul_b_offset_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        a: &MetalBuffer,
        b: &MetalBuffer,
        b_offset: usize,
        out: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gelu_mul_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.raw()), b_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
        }
        let threads_per_tg = 256_usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (count as usize).div_ceil(threads_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Fused `GeGLU` elementwise `out[i] = gelu(a[i]) * b[i]`.
    pub fn gelu_mul_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        a: &MetalBuffer,
        b: &MetalBuffer,
        out: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        self.gelu_mul_b_offset_into(batch, a, b, 0, out, count)
    }

    /// Fused `RMSNorm` + residual: `out[i] = rms_norm(input, weight)[i] + residual[i]`.
    /// Replaces an `rms_norm` + `residual_add` pair (one fewer dispatch and one
    /// fewer scratch round-trip per use).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn rms_norm_add_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        residual: &MetalBuffer,
        output: &MetalBuffer,
        dim: u32,
        rows: u32,
        eps: f32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.rms_norm_add_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(dim).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(eps).cast_mut().cast::<c_void>()),
                size_of::<f32>(),
                4,
            );
            encoder.setBuffer_offset_atIndex(Some(residual.raw()), 0, 5);
        }
        let tg_size = dim.min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Encode an element-wise f16 multiply into a [`crate::batch::CommandBatch`].
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Result` for API symmetry.
    #[allow(unsafe_code)]
    pub fn elementwise_mul_into(
        &self,
        batch: &mut crate::batch::CommandBatch,
        a: &MetalBuffer,
        b: &MetalBuffer,
        out: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.elementwise_mul_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(std::ptr::addr_of!(count).cast_mut().cast::<c_void>()),
                size_of::<u32>(),
                3,
            );
        }
        let threads_per_tg = 256_usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (count as usize).div_ceil(threads_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn tq2_dequant(
        &self,
        ctx: &MetalContext,
        packed: &MetalBuffer,
        scales: &MetalBuffer,
        output: &MetalBuffer,
        num_elements: u32,
    ) -> crate::Result<()> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;

        encoder.setComputePipelineState(
            self.tq2_dequant_pipeline
                .as_ref()
                .ok_or_else(|| Error::ShaderNotFound("tq2_dequant".to_owned()))?,
        );

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed.raw()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales.raw()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(
                    std::ptr::addr_of!(num_elements).cast_mut().cast::<c_void>(),
                ),
                std::mem::size_of::<u32>(),
                3,
            );
        }

        let tg_width = (num_elements as usize).min(256).next_power_of_two();
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]
mod tests {
    use super::*;
    use half::f16;

    fn f16_vec(vals: &[f32]) -> Vec<f16> {
        vals.iter().map(|&v| f16::from_f32(v)).collect()
    }

    fn read_f16(buf: &MetalBuffer) -> Vec<f32> {
        buf.as_slice::<f16>().iter().map(|v| v.to_f32()).collect()
    }

    fn setup() -> (MetalContext, Kernels) {
        let ctx = MetalContext::new().expect("Metal context");
        let shaders = ShaderLibrary::new(ctx.device()).expect("shader library");
        let kernels = Kernels::new(&ctx, &shaders).expect("kernels");
        (ctx, kernels)
    }

    #[test]
    fn vision_patch_embed_known_patch() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let input = MetalBuffer::from_slice(
            device,
            &f16_vec(&[
                1.0, 2.0, 3.0, 4.0, // channel 0
                10.0, 20.0, 30.0, 40.0, // channel 1
                100.0, 200.0, 300.0, 400.0, // channel 2
            ]),
        )
        .expect("input");
        // Weight layout is GGUF-order [patch, patch, 3, hidden]. With one
        // hidden channel and patch_size=2, this sums all pixels/channels.
        let weight = MetalBuffer::from_slice(device, &f16_vec(&[1.0; 12])).expect("weight");
        let bias = MetalBuffer::from_slice(device, &f16_vec(&[5.0])).expect("bias");
        let output = MetalBuffer::empty(device, size_of::<f16>()).expect("output");

        kernels
            .vision_patch_embed(&ctx, &input, &weight, &bias, &output, 1, 2, 2, 2)
            .expect("vision_patch_embed");

        let out = read_f16(&output);
        assert!((out[0] - 1115.0).abs() < 0.5, "patch embed got {out:?}");
    }

    #[test]
    fn vision_avg_pool_2d_known_grid() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // [rows=2, cols=2, hidden=2]
        let input = MetalBuffer::from_slice(
            device,
            &f16_vec(&[1.0, 10.0, 3.0, 30.0, 5.0, 50.0, 7.0, 70.0]),
        )
        .expect("input");
        let output = MetalBuffer::empty(device, 2 * size_of::<f16>()).expect("output");

        kernels
            .vision_avg_pool_2d(&ctx, &input, &output, 2, 2, 2, 2, 2)
            .expect("vision_avg_pool_2d");

        let out = read_f16(&output);
        assert!((out[0] - 4.0).abs() < 0.01, "avg h0 got {out:?}");
        assert!((out[1] - 40.0).abs() < 0.01, "avg h1 got {out:?}");
    }

    #[test]
    fn vision_add_position_embedding_known_grid() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let hidden =
            MetalBuffer::from_slice(device, &f16_vec(&[0.0, 0.0, 0.0, 0.0])).expect("hidden");
        // [axis=2, pos=4, hidden=2]: x0, x1, x2, x3, y0, y1, y2, y3.
        let table = MetalBuffer::from_slice(
            device,
            &f16_vec(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
                80.0,
            ]),
        )
        .expect("position table");

        kernels
            .vision_add_position_embedding(&ctx, &hidden, &table, 2, 4, 1, 2)
            .expect("vision_add_position_embedding");

        let out = read_f16(&hidden);
        assert_eq!(out, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn vision_rope_uses_2d_rotate_half_layout() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let token0 = vec![7.0; 64];
        let mut token1 = vec![3.0; 64];
        token1[0] = 1.0;
        token1[16] = 2.0;
        let mut data = token0;
        data.extend_from_slice(&token1);
        let q = MetalBuffer::from_slice(device, &f16_vec(&data)).expect("q");
        let k = MetalBuffer::from_slice(device, &f16_vec(&data)).expect("k");

        kernels
            .vision_rope(&ctx, &q, &k, 64, 100.0, 1, 2, 1, 2)
            .expect("vision_rope");

        let out = read_f16(&q);
        assert!((out[0] - 7.0).abs() < 0.01, "x=0 should not rotate");
        let c = 1.0_f32.cos();
        let s = 1.0_f32.sin();
        assert!((out[64] - (1.0 * c - 2.0 * s)).abs() < 0.01);
        assert!((out[80] - (2.0 * c + 1.0 * s)).abs() < 0.01);
        assert!((out[96] - 3.0).abs() < 0.01, "y=0 should not rotate");
        assert_eq!(out, read_f16(&k));
    }

    #[test]
    fn clipped_linear_tiled_matches_reference_kernel() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let m = 5;
        let k = 7;
        let n = 6;
        let input_data: Vec<f32> = (0..m * k).map(|i| (i as f32 % 11.0) / 5.0 - 1.0).collect();
        let weight_data: Vec<f32> = (0..n * k).map(|i| (i as f32 % 13.0) / 7.0 - 0.8).collect();
        let input = MetalBuffer::from_slice(device, &f16_vec(&input_data)).expect("input");
        let weight = MetalBuffer::from_slice(device, &f16_vec(&weight_data)).expect("weight");
        let reference = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("reference");
        let tiled = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("tiled");

        kernels
            .clipped_linear(
                &ctx, &input, &weight, None, &reference, m as u32, k as u32, n as u32, -0.6, 0.7,
                -1.25, 1.5,
            )
            .expect("reference clipped_linear");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .clipped_linear_tiled_nt_into(
                &mut batch, &input, &weight, None, &tiled, m as u32, k as u32, n as u32, -0.6, 0.7,
                -1.25, 1.5,
            )
            .expect("tiled clipped_linear");
        batch.commit_and_wait().expect("commit");

        assert_close(
            &read_f16(&tiled),
            &read_f16(&reference),
            0.02,
            "clipped_linear_tiled",
        );
    }

    #[test]
    fn audio_subsample_conv2d_ln_relu_known_point() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let input = MetalBuffer::from_slice(device, &f16_vec(&[2.0])).expect("input");
        let mut weights = vec![0.0; 3 * 3 * 2];
        weights[8] = 1.0;
        weights[9] = 3.0;
        let weight = MetalBuffer::from_slice(device, &f16_vec(&weights)).expect("weight");
        let norm = MetalBuffer::from_slice(device, &f16_vec(&[1.0, 1.0])).expect("norm");
        let output = MetalBuffer::empty(device, 2 * size_of::<f16>()).expect("output");

        kernels
            .audio_subsample_conv2d_ln_relu(
                &ctx, &input, &weight, &norm, &output, 1, 2, 1, 1, 1, 1, 1e-6,
            )
            .expect("audio_subsample_conv2d_ln_relu");

        let out = read_f16(&output);
        assert!(
            out[0].abs() < 0.01,
            "negative normalized channel should ReLU to 0: {out:?}"
        );
        assert!(
            (out[1] - 1.0).abs() < 0.01,
            "positive channel should normalize to 1: {out:?}"
        );
    }

    #[test]
    fn audio_pack_frontend_matches_hf_reshape_order() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let input =
            MetalBuffer::from_slice(device, &f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
                .expect("input");
        let output = MetalBuffer::empty(device, 8 * size_of::<f16>()).expect("output");

        kernels
            .audio_pack_frontend(&ctx, &input, &output, 2, 2, 2)
            .expect("audio_pack_frontend");

        assert_eq!(
            read_f16(&output),
            vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]
        );
    }

    #[test]
    fn audio_glu_matches_left_times_sigmoid_right() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let input =
            MetalBuffer::from_slice(device, &f16_vec(&[2.0, 4.0, 0.0, 1.0])).expect("input");
        let output = MetalBuffer::empty(device, 2 * size_of::<f16>()).expect("output");

        kernels
            .audio_glu(&ctx, &input, &output, 1, 2)
            .expect("audio_glu");

        let out = read_f16(&output);
        let expected = [2.0 * 0.5, 4.0 / (1.0 + (-1.0_f32).exp())];
        for (got, exp) in out.iter().zip(expected) {
            assert!((got - exp).abs() < 0.01, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn audio_depthwise_conv1d_is_causal_left_padded() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let input = MetalBuffer::from_slice(device, &f16_vec(&[1.0, 2.0, 3.0])).expect("input");
        let weight = MetalBuffer::from_slice(device, &f16_vec(&[10.0, 1.0])).expect("weight");
        let output = MetalBuffer::empty(device, 3 * size_of::<f16>()).expect("output");

        kernels
            .depthwise_conv1d(&ctx, &input, &weight, &output, 3, 1, 2)
            .expect("depthwise_conv1d");

        assert_eq!(read_f16(&output), vec![1.0, 12.0, 23.0]);
    }

    #[test]
    fn rms_norm_known_vector() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // input = [1, 2, 3, 4], weight = [1, 1, 1, 1]
        // rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // output[i] = input[i] / rms * weight[i]
        let input_data = f16_vec(&[1.0, 2.0, 3.0, 4.0]);
        let weight_data = f16_vec(&[1.0, 1.0, 1.0, 1.0]);
        let input = MetalBuffer::from_slice(device, &input_data).expect("input buf");
        let weight = MetalBuffer::from_slice(device, &weight_data).expect("weight buf");
        let output = MetalBuffer::empty(device, 4 * size_of::<f16>()).expect("output buf");
        kernels
            .rms_norm(&ctx, &input, &weight, &output, 4, 1, 1e-6)
            .expect("rms_norm");
        let out = read_f16(&output);
        let rms = (7.5_f32).sqrt();
        for (i, &v) in out.iter().enumerate() {
            let expected = (i as f32 + 1.0) / rms;
            assert!(
                (v - expected).abs() < 0.02,
                "rms_norm[{i}]: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn gelu_known_values() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let vals = [0.0_f32, 1.0, -1.0, 2.0];
        let input_data = f16_vec(&vals);
        let input = MetalBuffer::from_slice(device, &input_data).expect("input buf");
        let output = MetalBuffer::empty(device, vals.len() * size_of::<f16>()).expect("output buf");
        kernels
            .gelu(&ctx, &input, &output, vals.len() as u32)
            .expect("gelu");
        let out = read_f16(&output);
        // GELU(0) ≈ 0, GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1588, GELU(2) ≈ 1.9545
        let expected = [0.0, 0.8412, -0.1588, 1.9545];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.02,
                "gelu[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn gelu_large_values_no_nan() {
        // Regression test: Metal tanh() overflows for large inputs if inner
        // value is not clamped. GELU(x) ≈ x for x >> 0 and ≈ 0 for x << 0.
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let vals = [10.0_f32, 15.0, 19.0, -10.0, -15.0, -19.0, 5.0, 65000.0];
        let input_data = f16_vec(&vals);
        let input = MetalBuffer::from_slice(device, &input_data).expect("input buf");
        let output = MetalBuffer::empty(device, vals.len() * size_of::<f16>()).expect("output buf");
        kernels
            .gelu(&ctx, &input, &output, vals.len() as u32)
            .expect("gelu");
        let out = read_f16(&output);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "gelu_large[{i}] (input={:.1}): got NaN/Inf",
                vals[i]
            );
        }
        // GELU(10) ≈ 10, GELU(-10) ≈ 0
        assert!((out[0] - 10.0).abs() < 0.1, "GELU(10) ≈ 10, got {}", out[0]);
        assert!(out[3].abs() < 0.01, "GELU(-10) ≈ 0, got {}", out[3]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let input_data = f16_vec(&[1.0, 2.0, 3.0, 4.0]);
        let input = MetalBuffer::from_slice(device, &input_data).expect("input buf");
        let output = MetalBuffer::empty(device, 4 * size_of::<f16>()).expect("output buf");
        kernels
            .softmax(&ctx, &input, &output, 4, 1)
            .expect("softmax");
        let out = read_f16(&output);
        let sum: f32 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "softmax sum: got {sum}, expected 1.0"
        );
        // Should be monotonically increasing
        for i in 1..out.len() {
            assert!(out[i] > out[i - 1], "softmax not monotonic at {i}");
        }
    }

    #[test]
    fn softmax_known_values() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let input_data = f16_vec(&[1.0, 2.0, 3.0, 4.0]);
        let input = MetalBuffer::from_slice(device, &input_data).expect("input buf");
        let output = MetalBuffer::empty(device, 4 * size_of::<f16>()).expect("output buf");
        kernels
            .softmax(&ctx, &input, &output, 4, 1)
            .expect("softmax");
        let out = read_f16(&output);
        let expected = [0.0321, 0.0871, 0.2369, 0.6439];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.02,
                "softmax[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn embedding_lookup_single() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // 4 tokens × 3 dims: [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
        let table_data = f16_vec(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        let token_ids: Vec<u32> = vec![2];
        let table = MetalBuffer::from_slice(device, &table_data).expect("table buf");
        let ids = MetalBuffer::from_slice(device, &token_ids).expect("ids buf");
        let output = MetalBuffer::empty(device, 3 * size_of::<f16>()).expect("output buf");
        kernels
            .embedding(&ctx, &table, &ids, &output, 3, 1)
            .expect("embedding");
        let out = read_f16(&output);
        let expected = [7.0, 8.0, 9.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.01,
                "embedding_single[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn embedding_lookup_multiple() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // 3 tokens × 2 dims: [[1,2],[3,4],[5,6]]
        let table_data = f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let token_ids: Vec<u32> = vec![0, 2];
        let table = MetalBuffer::from_slice(device, &table_data).expect("table buf");
        let ids = MetalBuffer::from_slice(device, &token_ids).expect("ids buf");
        let output = MetalBuffer::empty(device, 4 * size_of::<f16>()).expect("output buf");
        kernels
            .embedding(&ctx, &table, &ids, &output, 2, 2)
            .expect("embedding");
        let out = read_f16(&output);
        let expected = [1.0, 2.0, 5.0, 6.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.01,
                "embedding_multiple[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let input_data = f16_vec(&[1.0, 2.0, 3.0, 4.0]);
        let input = MetalBuffer::from_slice(device, &input_data).expect("input buf");
        let output = MetalBuffer::empty(device, 4 * size_of::<f16>()).expect("output buf");
        kernels
            .rope(&ctx, &input, &output, 4, 0, 10000.0, 1.0, 1)
            .expect("rope");
        let out = read_f16(&output);
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.01,
                "rope_pos0[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn rope_position_nonzero_changes_values() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let input_data = f16_vec(&[1.0, 2.0, 3.0, 4.0]);
        let input = MetalBuffer::from_slice(device, &input_data).expect("input buf");
        let output = MetalBuffer::empty(device, 4 * size_of::<f16>()).expect("output buf");
        kernels
            .rope(&ctx, &input, &output, 4, 100, 10000.0, 1.0, 1)
            .expect("rope");
        let out = read_f16(&output);
        let inp = [1.0_f32, 2.0, 3.0, 4.0];
        let mut changed = false;
        for (i, &v) in out.iter().enumerate() {
            if (v - inp[i]).abs() > 0.01 {
                changed = true;
            }
        }
        assert!(changed, "rope at position 100 should change values");
    }

    #[test]
    fn rope_partial_rotary_leaves_suffix_unchanged() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let vals = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input_data = f16_vec(&vals);
        let input = MetalBuffer::from_slice(device, &input_data).expect("input buf");
        let output = MetalBuffer::empty(device, 8 * size_of::<f16>()).expect("output buf");
        // HF proportional RoPE: factor 0.25 on head_dim=8 → rope_angles = 1,
        // so only NEOX pair (0, 4) rotates, with the full-head frequency
        // schedule freq = 1/10000^(0/8) = 1. Pairs (1,5), (2,6), (3,7) have
        // zero inverse frequency and pass through unchanged.
        kernels
            .rope(&ctx, &input, &output, 8, 100, 10000.0, 0.25, 1)
            .expect("rope");
        let out = read_f16(&output);
        let (sin, cos) = (100.0_f32).sin_cos();
        let want0 = vals[0].mul_add(cos, -(vals[4] * sin));
        let want4 = vals[0].mul_add(sin, vals[4] * cos);
        assert!(
            (out[0] - want0).abs() < 0.02,
            "rotated[0]: got {}, want {want0}",
            out[0]
        );
        assert!(
            (out[4] - want4).abs() < 0.02,
            "rotated[4]: got {}, want {want4}",
            out[4]
        );
        for i in [1usize, 2, 3, 5, 6, 7] {
            assert!(
                (out[i] - vals[i]).abs() < 0.01,
                "rope_proportional[{i}]: got {}, expected {} (should be unchanged)",
                out[i],
                vals[i]
            );
        }
    }

    #[test]
    fn matvec_identity() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // 3×3 identity × [5,7,9] = [5,7,9]
        let matrix_data = f16_vec(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let vector_data = f16_vec(&[5.0, 7.0, 9.0]);
        let matrix = MetalBuffer::from_slice(device, &matrix_data).expect("matrix buf");
        let vector = MetalBuffer::from_slice(device, &vector_data).expect("vector buf");
        let output = MetalBuffer::empty(device, 3 * size_of::<f16>()).expect("output buf");
        kernels
            .matvec(&ctx, &matrix, &vector, &output, 3, 3)
            .expect("matvec");
        let out = read_f16(&output);
        let expected = [5.0, 7.0, 9.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.01,
                "matvec_identity[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn matvec_known_product() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // [[1,2],[3,4]] × [5,6] = [17,39]
        let matrix_data = f16_vec(&[1.0, 2.0, 3.0, 4.0]);
        let vector_data = f16_vec(&[5.0, 6.0]);
        let matrix = MetalBuffer::from_slice(device, &matrix_data).expect("matrix buf");
        let vector = MetalBuffer::from_slice(device, &vector_data).expect("vector buf");
        let output = MetalBuffer::empty(device, 2 * size_of::<f16>()).expect("output buf");
        kernels
            .matvec(&ctx, &matrix, &vector, &output, 2, 2)
            .expect("matvec");
        let out = read_f16(&output);
        let expected = [17.0, 39.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.1,
                "matvec_known[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn matvec_rectangular() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // [[1,2,3,4],[5,6,7,8]] × [1,1,1,1] = [10,26]
        let matrix_data = f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let vector_data = f16_vec(&[1.0, 1.0, 1.0, 1.0]);
        let matrix = MetalBuffer::from_slice(device, &matrix_data).expect("matrix buf");
        let vector = MetalBuffer::from_slice(device, &vector_data).expect("vector buf");
        let output = MetalBuffer::empty(device, 2 * size_of::<f16>()).expect("output buf");
        kernels
            .matvec(&ctx, &matrix, &vector, &output, 2, 4)
            .expect("matvec");
        let out = read_f16(&output);
        let expected = [10.0, 26.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.1,
                "matvec_rect[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn attention_sliding_basic() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // 1 Q head, 1 KV head, head_dim=4, kv_len=4, window=512
        // Q = [1, 0, 0, 0] — should attend most to K[0]=[1,0,0,0]
        // K: 4 keys, each is a one-hot basis vector
        // V: same as K (identity-like)
        let head_dim: u32 = 4;
        let kv_len: u32 = 4;
        let num_q_heads: u32 = 1;
        let num_kv_heads: u32 = 1;

        let q_data = f16_vec(&[1.0, 0.0, 0.0, 0.0]);
        // K layout: [kv_len * num_kv_heads * head_dim]
        let k_data = f16_vec(&[
            1.0, 0.0, 0.0, 0.0, // K[0]
            0.0, 1.0, 0.0, 0.0, // K[1]
            0.0, 0.0, 1.0, 0.0, // K[2]
            0.0, 0.0, 0.0, 1.0, // K[3]
        ]);
        let v_data = f16_vec(&[
            1.0, 0.0, 0.0, 0.0, // V[0]
            0.0, 1.0, 0.0, 0.0, // V[1]
            0.0, 0.0, 1.0, 0.0, // V[2]
            0.0, 0.0, 0.0, 1.0, // V[3]
        ]);

        let q = MetalBuffer::from_slice(device, &q_data).expect("q buf");
        let k = MetalBuffer::from_slice(device, &k_data).expect("k buf");
        let v = MetalBuffer::from_slice(device, &v_data).expect("v buf");
        let output =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * size_of::<f16>())
                .expect("output buf");

        kernels
            .attention_sliding(
                &ctx,
                &q,
                &k,
                &v,
                &output,
                head_dim,
                kv_len,
                kv_len - 1,
                512,
                num_q_heads,
                num_kv_heads,
            )
            .expect("attention_sliding");

        let out = read_f16(&output);
        // Q=[1,0,0,0] dots with K[0]=[1,0,0,0] gives highest score
        // scale=1/sqrt(4)=0.5, so score0=0.5, others=0 → softmax≈0.36
        // Output[0] should be the largest component
        assert!(
            out[0] > out[1] && out[0] > out[2] && out[0] > out[3],
            "attention_sliding: out[0]={} should be largest, got {:?}",
            out[0],
            &out[..4]
        );
    }

    #[test]
    fn attention_full_causal() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // 1 Q head, 1 KV head, head_dim=4, kv_len=4
        // Uniform Q against identity K → equal scores → output = avg of V rows
        let head_dim: u32 = 4;
        let kv_len: u32 = 4;
        let num_q_heads: u32 = 1;
        let num_kv_heads: u32 = 1;

        let q_data = f16_vec(&[0.25, 0.25, 0.25, 0.25]);
        let k_data = f16_vec(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let v_data = f16_vec(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);

        let q = MetalBuffer::from_slice(device, &q_data).expect("q buf");
        let k = MetalBuffer::from_slice(device, &k_data).expect("k buf");
        let v = MetalBuffer::from_slice(device, &v_data).expect("v buf");
        let output =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * size_of::<f16>())
                .expect("output buf");

        kernels
            .attention_full(
                &ctx,
                &q,
                &k,
                &v,
                &output,
                head_dim,
                kv_len,
                kv_len - 1,
                num_q_heads,
                num_kv_heads,
            )
            .expect("attention_full");

        let out = read_f16(&output);
        // Equal attention scores → output ≈ [0.25, 0.25, 0.25, 0.25]
        for (i, &val) in out.iter().enumerate() {
            assert!(
                (val - 0.25).abs() < 0.2,
                "attention_full_causal[{i}]: got {val}, expected ~0.25"
            );
        }
    }

    #[test]
    fn attention_full_paged_respects_page_table() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let head_dim: u32 = 4;
        let kv_len: u32 = 2;
        let num_q_heads: u32 = 1;
        let num_kv_heads: u32 = 1;
        let page_size_tokens: u64 = 1;
        let page_table: [u64; 2] = [1, 0];

        let q_data = f16_vec(&[1.0, 0.0, 0.0, 0.0]);
        let k_data = f16_vec(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let v_data = f16_vec(&[10.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0]);

        let q = MetalBuffer::from_slice(device, &q_data).expect("q buf");
        let k = MetalBuffer::from_slice(device, &k_data).expect("k buf");
        let v = MetalBuffer::from_slice(device, &v_data).expect("v buf");
        let page_table_buf = MetalBuffer::from_slice(device, &page_table).expect("page table");
        let output =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * size_of::<f16>())
                .expect("output buf");

        kernels
            .attention_full_paged(
                &ctx,
                &q,
                &k,
                &v,
                &page_table_buf,
                &output,
                head_dim,
                kv_len,
                0,
                page_size_tokens,
                num_q_heads,
                num_kv_heads,
            )
            .expect("attention_full_paged");

        let out = read_f16(&output);
        assert!(
            out[1] > out[0],
            "paged full attention should read logical token 0 from physical page 1, got {out:?}"
        );
    }

    #[test]
    fn attention_sliding_paged_respects_page_table() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let head_dim: u32 = 4;
        let kv_len: u32 = 2;
        let num_q_heads: u32 = 1;
        let num_kv_heads: u32 = 1;
        let page_size_tokens: u64 = 1;
        let page_table: [u64; 2] = [1, 0];

        let q_data = f16_vec(&[1.0, 0.0, 0.0, 0.0]);
        let k_data = f16_vec(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let v_data = f16_vec(&[10.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0]);

        let q = MetalBuffer::from_slice(device, &q_data).expect("q buf");
        let k = MetalBuffer::from_slice(device, &k_data).expect("k buf");
        let v = MetalBuffer::from_slice(device, &v_data).expect("v buf");
        let page_table_buf = MetalBuffer::from_slice(device, &page_table).expect("page table");
        let output =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * size_of::<f16>())
                .expect("output buf");

        kernels
            .attention_sliding_paged(
                &ctx,
                &q,
                &k,
                &v,
                &page_table_buf,
                &output,
                head_dim,
                kv_len,
                0,
                2,
                page_size_tokens,
                num_q_heads,
                num_kv_heads,
            )
            .expect("attention_sliding_paged");

        let out = read_f16(&output);
        assert!(
            out[1] > out[0],
            "paged sliding attention should read logical token 0 from physical page 1, got {out:?}"
        );
    }

    #[test]
    fn elementwise_mul_known_vectors() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let a_data = f16_vec(&[2.0, 3.0, 4.0]);
        let b_data = f16_vec(&[0.5, 2.0, 0.25]);
        let a = MetalBuffer::from_slice(device, &a_data).expect("a buf");
        let b = MetalBuffer::from_slice(device, &b_data).expect("b buf");
        let out = MetalBuffer::empty(device, 3 * size_of::<f16>()).expect("out buf");
        kernels
            .elementwise_mul(&ctx, &a, &b, &out, 3)
            .expect("elementwise_mul");
        let result = read_f16(&out);
        assert!((result[0] - 1.0).abs() < 0.01, "2*0.5=1, got {}", result[0]);
        assert!((result[1] - 6.0).abs() < 0.01, "3*2=6, got {}", result[1]);
        assert!(
            (result[2] - 1.0).abs() < 0.01,
            "4*0.25=1, got {}",
            result[2]
        );
    }

    #[test]
    fn attention_gqa_multi_head() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // 2 Q heads sharing 1 KV head, head_dim=2, kv_len=2
        // Q head 0 = [1, 0] → attends to K[0]=[1,0]
        // Q head 1 = [0, 1] → attends to K[1]=[0,1]
        let head_dim: u32 = 2;
        let kv_len: u32 = 2;
        let num_q_heads: u32 = 2;
        let num_kv_heads: u32 = 1;

        // Q: [num_q_heads * head_dim] = [head0_d0, head0_d1, head1_d0, head1_d1]
        let q_data = f16_vec(&[1.0, 0.0, 0.0, 1.0]);
        // K: [kv_len * num_kv_heads * head_dim]
        let k_data = f16_vec(&[1.0, 0.0, 0.0, 1.0]);
        // V: [kv_len * num_kv_heads * head_dim]
        let v_data = f16_vec(&[10.0, 0.0, 0.0, 10.0]);

        let q = MetalBuffer::from_slice(device, &q_data).expect("q buf");
        let k = MetalBuffer::from_slice(device, &k_data).expect("k buf");
        let v = MetalBuffer::from_slice(device, &v_data).expect("v buf");
        let output =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * size_of::<f16>())
                .expect("output buf");

        kernels
            .attention_full(
                &ctx,
                &q,
                &k,
                &v,
                &output,
                head_dim,
                kv_len,
                kv_len - 1,
                num_q_heads,
                num_kv_heads,
            )
            .expect("attention_full gqa");

        let out = read_f16(&output);
        // Head 0: Q=[1,0] attends mostly to K[0]=[1,0] → output ≈ V[0]=[10,0]
        assert!(
            out[0] > 5.0,
            "gqa head0 dim0: got {}, expected > 5.0",
            out[0]
        );
        // Head 1: Q=[0,1] attends mostly to K[1]=[0,1] → output ≈ V[1]=[0,10]
        assert!(
            out[3] > 5.0,
            "gqa head1 dim1: got {}, expected > 5.0",
            out[3]
        );
    }

    #[test]
    fn rms_norm_into_matches_standalone() {
        use crate::batch::CommandBatch;
        use std::sync::Arc;

        let ctx = Arc::new(MetalContext::new().expect("ctx"));
        let shaders = ShaderLibrary::new(ctx.device()).expect("shaders");
        let kernels = Kernels::new(&ctx, &shaders).expect("kernels");

        let dim = 64_u32;
        let input_data: Vec<f16> = (0..dim).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
        let weight_data: Vec<f16> = (0..dim).map(|_| f16::from_f32(1.0)).collect();
        let input = MetalBuffer::from_slice(ctx.device(), &input_data).expect("input");
        let weight = MetalBuffer::from_slice(ctx.device(), &weight_data).expect("weight");
        let output_standalone = MetalBuffer::empty(ctx.device(), dim as usize * 2).expect("out1");
        let output_batched = MetalBuffer::empty(ctx.device(), dim as usize * 2).expect("out2");

        // Standalone
        kernels
            .rms_norm(&ctx, &input, &weight, &output_standalone, dim, 1, 1e-6)
            .expect("standalone");

        // Batched
        let mut batch = CommandBatch::new(&ctx).expect("batch");
        kernels
            .rms_norm_into(&mut batch, &input, &weight, &output_batched, dim, 1, 1e-6)
            .expect("into");
        batch.commit_and_wait().expect("commit");

        // Compare
        let s = output_standalone.as_slice::<f16>();
        let b = output_batched.as_slice::<f16>();
        for i in 0..dim as usize {
            let diff = (s[i].to_f32() - b[i].to_f32()).abs();
            assert!(
                diff < 1e-4,
                "mismatch at {i}: standalone={}, batched={}",
                s[i],
                b[i]
            );
        }
    }

    #[test]
    fn residual_add_gpu_matches_cpu() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let count = 2304_u32;
        let a_data: Vec<f16> = (0..count).map(|i| f16::from_f32(i as f32 * 0.01)).collect();
        let b_data: Vec<f16> = (0..count)
            .map(|i| f16::from_f32(i as f32 * -0.005))
            .collect();
        let a = MetalBuffer::from_slice(device, &a_data).expect("a");
        let b = MetalBuffer::from_slice(device, &b_data).expect("b");
        let output = MetalBuffer::empty(device, count as usize * 2).expect("output");

        kernels
            .residual_add(&ctx, &a, &b, &output, count)
            .expect("residual_add");

        let result = output.as_slice::<f16>();
        for i in 0..count as usize {
            let expected = a_data[i].to_f32() + b_data[i].to_f32();
            let diff = (result[i].to_f32() - expected).abs();
            assert!(
                diff < 1e-3,
                "mismatch at {i}: got {}, expected {}",
                result[i],
                expected
            );
        }
    }

    #[test]
    fn scale_in_place_gpu_matches_cpu() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let count = 1536_u32;
        let scalar = 0.0178_f32;
        let data: Vec<f16> = (0..count).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
        let buf = MetalBuffer::from_slice(device, &data).expect("buf");

        kernels
            .scale_in_place_gpu(&ctx, &buf, scalar, count)
            .expect("scale");

        let result = buf.as_slice::<f16>();
        for i in 0..count as usize {
            let expected = data[i].to_f32() * scalar;
            let diff = (result[i].to_f32() - expected).abs();
            assert!(
                diff < 1e-2,
                "mismatch at {i}: got {}, expected {}",
                result[i],
                expected
            );
        }
    }

    #[test]
    fn gather_strided_f16_extracts_layer_slice() {
        use crate::batch::CommandBatch;
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // Packed [n_rows × n_layer × pld]; extract layer `L`'s [n_rows × pld].
        let (n_rows, n_layer, pld, l) = (5usize, 7usize, 4usize, 3usize);
        let src: Vec<f16> = (0..n_rows * n_layer * pld)
            .map(|i| f16::from_f32(i as f32))
            .collect();
        let src_buf = MetalBuffer::from_slice(device, &src).expect("src");
        let dst_buf = MetalBuffer::empty(device, n_rows * pld * 2).expect("dst");
        let mut batch = CommandBatch::new(&ctx).expect("batch");
        kernels
            .gather_strided_f16_into(
                &mut batch,
                &src_buf,
                &dst_buf,
                (n_layer * pld) as u32,
                (l * pld) as u32,
                pld as u32,
                n_rows as u32,
            )
            .expect("gather");
        batch.commit_and_wait().expect("commit");
        let got = dst_buf.as_slice::<f16>();
        for i in 0..n_rows {
            for p in 0..pld {
                let expected = src[i * n_layer * pld + l * pld + p];
                let got_value = got[i * pld + p].to_f32();
                let expected_value = expected.to_f32();
                assert!(
                    (got_value - expected_value).abs() <= f32::EPSILON,
                    "mismatch at row {i} pos {p}: got {got_value}, expected {expected_value}"
                );
            }
        }
    }

    #[test]
    fn logit_softcap_gpu_matches_cpu() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let count = 4096_u32;
        let cap = 30.0_f32;
        // Spread across a wide range so tanh saturates on both ends.
        let data: Vec<f32> = (0..count).map(|i| (i as f32 - 2048.0) * 0.05).collect();
        let buf = MetalBuffer::from_slice(device, &data).expect("buf");

        kernels
            .logit_softcap_gpu(&ctx, &buf, cap, count)
            .expect("softcap");

        let result = buf.as_slice::<f32>();
        for i in 0..count as usize {
            let expected = cap * (data[i] / cap).tanh();
            let diff = (result[i] - expected).abs();
            assert!(
                diff < 1e-4,
                "mismatch at {i}: got {}, expected {}",
                result[i],
                expected
            );
        }
    }

    #[test]
    fn qk_norm_gpu_matches_cpu() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let num_heads = 4_u32;
        let head_dim = 64_u32;
        let eps = 1e-6_f32;
        let total = (num_heads * head_dim) as usize;

        let qk_data: Vec<f16> = (0..total)
            .map(|i| f16::from_f32((i as f32 * 0.037).sin()))
            .collect();
        let norm_weights: Vec<f16> = (0..head_dim as usize)
            .map(|i| f16::from_f32(1.0 + i as f32 * 0.01))
            .collect();

        // GPU version
        let qk_buf = MetalBuffer::from_slice(device, &qk_data).expect("qk");
        let norm_buf = MetalBuffer::from_slice(device, &norm_weights).expect("norm");
        kernels
            .qk_norm_gpu(&ctx, &qk_buf, &norm_buf, head_dim, num_heads, eps)
            .expect("qk_norm");
        let gpu_result: Vec<f32> = qk_buf
            .as_slice::<f16>()
            .iter()
            .map(|v| v.to_f32())
            .collect();

        // CPU reference
        let mut cpu_data: Vec<f32> = qk_data.iter().map(|v| v.to_f32()).collect();
        let cpu_weights: Vec<f32> = norm_weights.iter().map(|v| v.to_f32()).collect();
        for h in 0..num_heads as usize {
            let base = h * head_dim as usize;
            let mut sum_sq = 0.0_f32;
            for j in 0..head_dim as usize {
                sum_sq += cpu_data[base + j] * cpu_data[base + j];
            }
            let rms = (sum_sq / head_dim as f32 + eps).sqrt().recip();
            for j in 0..head_dim as usize {
                cpu_data[base + j] = cpu_data[base + j] * rms * cpu_weights[j];
            }
        }

        for i in 0..total {
            let diff = (gpu_result[i] - cpu_data[i]).abs();
            assert!(
                diff < 0.01,
                "mismatch at {i}: gpu={}, cpu={}",
                gpu_result[i],
                cpu_data[i]
            );
        }
    }

    #[test]
    fn qk_norm_rope_matches_separate_kernels() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        assert!(
            kernels.qk_norm_rope_pipeline.is_some(),
            "qk_norm_rope pipeline failed to compile"
        );

        let num_heads = 8_u32;
        let head_dim = 256_u32; // Gemma 4 head_dim; exercises the staging array
        let eps = 1e-6_f32;
        let theta = 1_000_000.0_f32;
        let position = 137_u32;
        let total = (num_heads * head_dim) as usize;

        for &prf in &[1.0_f32, 0.5_f32] {
            let data: Vec<f16> = (0..total)
                .map(|i| f16::from_f32((i as f32 * 0.013).sin() * 0.8))
                .collect();
            let weights: Vec<f16> = (0..head_dim as usize)
                .map(|i| f16::from_f32(1.0 + (i as f32 * 0.005).cos() * 0.1))
                .collect();
            let wbuf = MetalBuffer::from_slice(device, &weights).expect("w");

            // Reference: separate qk_norm then rope, in two dispatches.
            let ref_buf = MetalBuffer::from_slice(device, &data).expect("ref");
            let mut b1 = crate::batch::CommandBatch::new(&ctx).expect("b1");
            kernels
                .qk_norm_gpu_into(&mut b1, &ref_buf, &wbuf, head_dim, num_heads, eps)
                .expect("qk_norm");
            kernels
                .rope_into(
                    &mut b1, &ref_buf, &ref_buf, head_dim, position, theta, prf, num_heads,
                )
                .expect("rope");
            b1.commit_and_wait().expect("b1 commit");

            // Fused: single dispatch.
            let fused_buf = MetalBuffer::from_slice(device, &data).expect("fused");
            let mut b2 = crate::batch::CommandBatch::new(&ctx).expect("b2");
            kernels
                .qk_norm_rope_into(
                    &mut b2, &fused_buf, &fused_buf, &wbuf, head_dim, num_heads, eps, position,
                    theta, prf,
                )
                .expect("fused");
            b2.commit_and_wait().expect("b2 commit");

            // Fused staging rounds to FP16 between norm and RoPE exactly as the
            // two-kernel path does, so the result is bit-identical.
            assert_eq!(
                read_f16(&fused_buf),
                read_f16(&ref_buf),
                "qk_norm_rope (prf={prf}) must be bit-identical to qk_norm+rope"
            );
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn batched_decode_attention_matches_independent_single_decode() {
        // The core continuous-batching correctness property: a single batched
        // dispatch over N lanes (write_kv_cache_decode + flash_attention_decode_
        // batched) must equal running each lane independently through the proven
        // single-sequence flash_attention. Each lane sits at a *different*
        // position, with its own KV history, exactly like N concurrent agents.
        let (ctx, kernels) = setup();
        let device = ctx.device();
        assert!(
            kernels.flash_attention_decode_batched_pipeline.is_some(),
            "flash_attention_decode_batched pipeline failed to compile"
        );
        assert!(
            kernels.write_kv_cache_decode_pipeline.is_some(),
            "write_kv_cache_decode pipeline failed to compile"
        );

        let num_q_heads = 8_u32;
        let num_kv_heads = 2_u32;
        let head_dim = 64_u32;
        let lane_capacity = 16_u32;
        let row = num_kv_heads * head_dim; // KV row elements per token
        let lane_positions = [4_u32, 9_u32, 0_u32];
        let n_lanes = lane_positions.len() as u32;

        let mkv = |seed: f32, n: usize| -> Vec<f16> {
            (0..n)
                .map(|i| f16::from_f32(((i as f32 + seed) * 0.017).sin() * 0.4))
                .collect()
        };

        // Per-lane query rows: [n_lanes, num_q_heads, head_dim].
        let q_all: Vec<f16> = mkv(1.0, (n_lanes * num_q_heads * head_dim) as usize);
        // New K/V row per lane: [n_lanes, row].
        let k_new: Vec<f16> = mkv(2.0, (n_lanes * row) as usize);
        let v_new: Vec<f16> = mkv(3.0, (n_lanes * row) as usize);

        // Unified pool, zeroed, pre-filled with each lane's history [0..pos).
        let pool_elems = (n_lanes * lane_capacity * row) as usize;
        let mut k_pool_storage = vec![f16::from_f32(0.0); pool_elems];
        let mut v_pool_storage = vec![f16::from_f32(0.0); pool_elems];
        for (l, &pos) in lane_positions.iter().enumerate() {
            let region = l * (lane_capacity * row) as usize;
            for t in 0..pos as usize {
                let hist_k = mkv(10.0 + l as f32 + t as f32 * 0.5, row as usize);
                let hist_v = mkv(20.0 + l as f32 + t as f32 * 0.5, row as usize);
                let off = region + t * row as usize;
                k_pool_storage[off..off + row as usize].copy_from_slice(&hist_k);
                v_pool_storage[off..off + row as usize].copy_from_slice(&hist_v);
            }
        }

        // ---- Batched path ----
        let q_buf = MetalBuffer::from_slice(device, &q_all).expect("q");
        let ksrc = MetalBuffer::from_slice(device, &k_new).expect("ksrc");
        let vsrc = MetalBuffer::from_slice(device, &v_new).expect("vsrc");
        let kpool = MetalBuffer::from_slice(device, &k_pool_storage).expect("kpool");
        let vpool = MetalBuffer::from_slice(device, &v_pool_storage).expect("vpool");
        let pos_buf = MetalBuffer::from_slice(device, &lane_positions).expect("pos");
        let out_batched =
            MetalBuffer::empty(device, (n_lanes * num_q_heads * head_dim) as usize * 2)
                .expect("outb");
        let mut b = crate::batch::CommandBatch::new(&ctx).expect("b");
        kernels
            .write_kv_cache_decode_into(
                &mut b,
                &ksrc,
                &vsrc,
                &kpool,
                &vpool,
                &pos_buf,
                row,
                lane_capacity,
                n_lanes,
            )
            .expect("scatter");
        kernels
            .flash_attention_decode_batched_into(
                &mut b,
                &q_buf,
                &kpool,
                &vpool,
                &out_batched,
                &pos_buf,
                head_dim,
                lane_capacity,
                num_q_heads,
                num_kv_heads,
                0,
                n_lanes,
            )
            .expect("batched flash");
        b.commit_and_wait().expect("commit batched");
        let got = read_f16(&out_batched);

        // ---- Reference path: each lane via single flash_attention ----
        let qrow = (num_q_heads * head_dim) as usize;
        for (l, &pos) in lane_positions.iter().enumerate() {
            let kv_len = pos + 1;
            // Build [kv_len, row] cache = history[0..pos] + new row at pos.
            let region = l * (lane_capacity * row) as usize;
            let mut kref = vec![f16::from_f32(0.0); (kv_len * row) as usize];
            let mut vref = vec![f16::from_f32(0.0); (kv_len * row) as usize];
            for t in 0..pos as usize {
                let off = region + t * row as usize;
                let d = t * row as usize;
                kref[d..d + row as usize].copy_from_slice(&k_pool_storage[off..off + row as usize]);
                vref[d..d + row as usize].copy_from_slice(&v_pool_storage[off..off + row as usize]);
            }
            let dn = pos as usize * row as usize;
            let sn = l * row as usize;
            kref[dn..dn + row as usize].copy_from_slice(&k_new[sn..sn + row as usize]);
            vref[dn..dn + row as usize].copy_from_slice(&v_new[sn..sn + row as usize]);

            let qlane = &q_all[l * qrow..(l + 1) * qrow];
            let q1 = MetalBuffer::from_slice(device, qlane).expect("q1");
            let k1 = MetalBuffer::from_slice(device, &kref).expect("k1");
            let v1 = MetalBuffer::from_slice(device, &vref).expect("v1");
            let out1 = MetalBuffer::empty(device, qrow * 2).expect("out1");
            let mut b1 = crate::batch::CommandBatch::new(&ctx).expect("b1");
            kernels
                .flash_attention_into(
                    &mut b1,
                    &q1,
                    &k1,
                    &v1,
                    &out1,
                    head_dim,
                    kv_len,
                    pos,
                    num_q_heads,
                    num_kv_heads,
                )
                .expect("single flash");
            b1.commit_and_wait().expect("commit single");

            assert_close(
                &got[l * qrow..(l + 1) * qrow],
                &read_f16(&out1),
                0.01,
                &format!("lane {l} (pos {pos}) batched decode vs single flash"),
            );
        }
    }

    #[test]
    fn rope_batch_decode_matches_rope_batch_for_contiguous_positions() {
        // When the per-lane positions are exactly [start_pos, start_pos+1, ...]
        // the batched-decode RoPE must produce bit-identical output to the
        // scalar-start_pos `rope_batch`. This pins the new primitive to the
        // proven kernel before it is used for independent-lane decode.
        let (ctx, kernels) = setup();
        let device = ctx.device();
        assert!(
            kernels.rope_batch_decode_pipeline.is_some(),
            "rope_batch_decode pipeline failed to compile"
        );

        let num_heads = 8_u32;
        let head_dim = 256_u32;
        let seq_len = 5_u32;
        let theta = 1_000_000.0_f32;
        let start_pos = 137_u32;
        let total = (num_heads * head_dim * seq_len) as usize;

        for &prf in &[1.0_f32, 0.5_f32] {
            let data: Vec<f16> = (0..total)
                .map(|i| f16::from_f32((i as f32 * 0.013).sin() * 0.8))
                .collect();

            // Reference: scalar start_pos rope_batch (pos = start_pos + row).
            let ref_buf = MetalBuffer::from_slice(device, &data).expect("ref");
            let mut b1 = crate::batch::CommandBatch::new(&ctx).expect("b1");
            kernels
                .rope_batch_into(
                    &mut b1, &ref_buf, start_pos, theta, head_dim, num_heads, seq_len, prf,
                )
                .expect("rope_batch");
            b1.commit_and_wait().expect("b1 commit");

            // Decode variant: explicit per-lane positions, contiguous.
            let positions: Vec<u32> = (0..seq_len).map(|r| start_pos + r).collect();
            let pos_buf = MetalBuffer::from_slice(device, &positions).expect("pos");
            let dec_buf = MetalBuffer::from_slice(device, &data).expect("dec");
            let mut b2 = crate::batch::CommandBatch::new(&ctx).expect("b2");
            kernels
                .rope_batch_decode_into(
                    &mut b2, &dec_buf, &pos_buf, theta, head_dim, num_heads, seq_len, prf,
                )
                .expect("rope_batch_decode");
            b2.commit_and_wait().expect("b2 commit");

            assert_eq!(
                read_f16(&dec_buf),
                read_f16(&ref_buf),
                "rope_batch_decode (prf={prf}) must equal rope_batch for contiguous positions"
            );
        }
    }

    #[test]
    fn rope_batch_decode_applies_independent_positions_per_lane() {
        // Two lanes carrying identical head data but different positions must
        // come out differently, and each must match a single-row rope at its
        // own position. This is the property batched decode actually relies on.
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let num_heads = 4_u32;
        let head_dim = 128_u32;
        let theta = 1_000_000.0_f32;
        let prf = 1.0_f32;
        let row = (num_heads * head_dim) as usize;

        let one_row: Vec<f16> = (0..row)
            .map(|i| f16::from_f32((i as f32 * 0.021).cos() * 0.5))
            .collect();

        let lane_positions = [3_u32, 91_u32];

        // Batched: 2 lanes, same data, positions [3, 91].
        let mut two: Vec<f16> = Vec::with_capacity(row * 2);
        two.extend_from_slice(&one_row);
        two.extend_from_slice(&one_row);
        let pos_buf = MetalBuffer::from_slice(device, &lane_positions).expect("pos");
        let dec_buf = MetalBuffer::from_slice(device, &two).expect("dec");
        let mut b = crate::batch::CommandBatch::new(&ctx).expect("b");
        kernels
            .rope_batch_decode_into(
                &mut b, &dec_buf, &pos_buf, theta, head_dim, num_heads, 2, prf,
            )
            .expect("rope_batch_decode");
        b.commit_and_wait().expect("commit");
        let got = read_f16(&dec_buf);

        for (lane, &pos) in lane_positions.iter().enumerate() {
            let single = MetalBuffer::from_slice(device, &one_row).expect("single");
            let mut bs = crate::batch::CommandBatch::new(&ctx).expect("bs");
            kernels
                .rope_batch_into(&mut bs, &single, pos, theta, head_dim, num_heads, 1, prf)
                .expect("rope_batch single");
            bs.commit_and_wait().expect("commit single");
            assert_eq!(
                &got[lane * row..(lane + 1) * row],
                read_f16(&single).as_slice(),
                "lane {lane} (pos {pos}) must match single-row rope at that position"
            );
        }
        assert_ne!(
            &got[0..row],
            &got[row..2 * row],
            "lanes at different positions must differ"
        );
    }

    #[test]
    fn flash_attention_matches_standard() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let num_q_heads: u32 = 8;
        let num_kv_heads: u32 = 1;
        let head_dim: u32 = 64;
        let kv_len: u32 = 128;
        let current_pos: u32 = kv_len - 1;

        let q_data: Vec<f16> = (0..num_q_heads * head_dim)
            .map(|i| f16::from_f32(((i as f32 * 0.1).sin()) * 0.5))
            .collect();
        let kv_data: Vec<f16> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| f16::from_f32(((i as f32 * 0.07).cos()) * 0.3))
            .collect();

        let q = MetalBuffer::from_slice(device, &q_data).expect("q");
        let k = MetalBuffer::from_slice(device, &kv_data).expect("k");
        let v = MetalBuffer::from_slice(device, &kv_data).expect("v");

        let output_std =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * 2).expect("out_std");
        let output_flash =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * 2).expect("out_flash");

        kernels
            .attention_full(
                &ctx,
                &q,
                &k,
                &v,
                &output_std,
                head_dim,
                kv_len,
                current_pos,
                num_q_heads,
                num_kv_heads,
            )
            .expect("std attn");

        kernels
            .flash_attention(
                &ctx,
                &q,
                &k,
                &v,
                &output_flash,
                head_dim,
                kv_len,
                current_pos,
                num_q_heads,
                num_kv_heads,
                0,
            )
            .expect("flash attn");

        let std_out = output_std.as_slice::<f16>();
        let flash_out = output_flash.as_slice::<f16>();
        for i in 0..(num_q_heads * head_dim) as usize {
            let a = std_out[i].to_f32();
            let b = flash_out[i].to_f32();
            assert!((a - b).abs() < 0.05, "mismatch at {i}: std={a}, flash={b}");
        }
    }

    #[test]
    fn flash_attention_windowed_split_matches_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // Real-model-ish: 8 q heads, 1 kv head (MQA), head_dim 256, with a
        // sliding window so win_start > 0 and the split must respect it.
        let num_q_heads: u32 = 8;
        let num_kv_heads: u32 = 1;
        let head_dim: u32 = 256;
        let kv_len: u32 = 700;
        let current_pos: u32 = kv_len - 1;
        let window: u32 = 512;

        let q_data: Vec<f16> = (0..num_q_heads * head_dim)
            .map(|i| f16::from_f32((i as f32 * 0.1).sin() * 0.5))
            .collect();
        let kv_data: Vec<f16> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| f16::from_f32((i as f32 * 0.07).cos() * 0.3))
            .collect();
        let q = MetalBuffer::from_slice(device, &q_data).expect("q");
        let k = MetalBuffer::from_slice(device, &kv_data).expect("k");
        let v = MetalBuffer::from_slice(device, &kv_data).expect("v");

        // Reference: single-pass windowed flash attention.
        let out_ref =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * 2).expect("ref");
        let mut b0 = crate::batch::CommandBatch::new(&ctx).expect("b0");
        kernels
            .flash_attention_windowed_into(
                &mut b0,
                &q,
                &k,
                &v,
                &out_ref,
                head_dim,
                kv_len,
                current_pos,
                num_q_heads,
                num_kv_heads,
                window,
            )
            .expect("ref attn");
        b0.commit_and_wait().expect("commit ref");

        // Split path with several splits (eff seq len = window = 512).
        let num_splits = kernels.flash_split_count(window);
        assert!(num_splits > 1, "expected multi-split, got {num_splits}");
        let pm = MetalBuffer::empty(device, (num_q_heads * num_splits) as usize * 4).expect("pm");
        let ps = MetalBuffer::empty(device, (num_q_heads * num_splits) as usize * 4).expect("ps");
        let pa = MetalBuffer::empty(device, (num_q_heads * num_splits * head_dim) as usize * 4)
            .expect("pa");
        let out_split =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * 2).expect("split");
        let mut b1 = crate::batch::CommandBatch::new(&ctx).expect("b1");
        kernels
            .flash_attention_windowed_split_into(
                &mut b1,
                &q,
                &k,
                &v,
                &out_split,
                &pm,
                &ps,
                &pa,
                head_dim,
                kv_len,
                current_pos,
                num_q_heads,
                num_kv_heads,
                window,
                num_splits,
            )
            .expect("split attn");
        b1.commit_and_wait().expect("commit split");

        let a = out_ref.as_slice::<f16>();
        let b = out_split.as_slice::<f16>();
        for i in 0..(num_q_heads * head_dim) as usize {
            let x = a[i].to_f32();
            let y = b[i].to_f32();
            assert!((x - y).abs() < 0.02, "mismatch at {i}: ref={x}, split={y}");
        }
    }

    #[test]
    fn flash_decoding_matches_standard_attention() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let num_q_heads: u32 = 8;
        let num_kv_heads: u32 = 2;
        let head_dim: u32 = 128;
        let kv_len: u32 = 1024;
        let current_pos: u32 = kv_len - 1;

        let q_data: Vec<f16> = (0..num_q_heads * head_dim)
            .map(|i| f16::from_f32(((i as f32 * 0.1).sin()) * 0.5))
            .collect();
        let kv_data: Vec<f16> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| f16::from_f32(((i as f32 * 0.07).cos()) * 0.3))
            .collect();

        let q = MetalBuffer::from_slice(device, &q_data).expect("q");
        let k = MetalBuffer::from_slice(device, &kv_data).expect("k");
        let v = MetalBuffer::from_slice(device, &kv_data).expect("v");

        let output_std =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * 2).expect("out_std");
        let output_fd =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * 2).expect("out_fd");

        // Run standard flash attention
        kernels
            .flash_attention(
                &ctx,
                &q,
                &k,
                &v,
                &output_std,
                head_dim,
                kv_len,
                current_pos,
                num_q_heads,
                num_kv_heads,
                0,
            )
            .expect("flash attn");

        // Run flash decoding
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("CommandBatch");
        kernels
            .flash_decoding_into(
                &ctx,
                &mut batch,
                &q,
                &k,
                &v,
                &output_fd,
                head_dim,
                kv_len,
                current_pos,
                num_q_heads,
                num_kv_heads,
            )
            .expect("flash decoding");
        batch.commit_and_wait().expect("commit");

        let std_out = output_std.as_slice::<f16>();
        let fd_out = output_fd.as_slice::<f16>();
        for i in 0..(num_q_heads * head_dim) as usize {
            let a = std_out[i].to_f32();
            let b = fd_out[i].to_f32();
            assert!(
                (a - b).abs() < 0.05,
                "mismatch at {i}: std={a}, flash_decoding={b}",
            );
        }
    }

    #[test]
    fn flash_attention_paged_respects_page_table() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let num_q_heads: u32 = 1;
        let num_kv_heads: u32 = 1;
        let head_dim: u32 = 64;
        let kv_len: u32 = 2;
        let page_size_tokens: u64 = 1;
        let page_table: [u64; 2] = [1, 0];

        let mut q_data = vec![f16::from_f32(0.0); head_dim as usize];
        q_data[1] = f16::from_f32(1.0);
        let mut k_data = vec![f16::from_f32(0.0); (kv_len * head_dim) as usize];
        k_data[0] = f16::from_f32(1.0);
        k_data[head_dim as usize + 1] = f16::from_f32(1.0);
        let mut v_data = vec![f16::from_f32(0.0); (kv_len * head_dim) as usize];
        v_data[0] = f16::from_f32(10.0);
        v_data[head_dim as usize + 1] = f16::from_f32(20.0);

        let q = MetalBuffer::from_slice(device, &q_data).expect("q");
        let k = MetalBuffer::from_slice(device, &k_data).expect("k");
        let v = MetalBuffer::from_slice(device, &v_data).expect("v");
        let page_table_buf = MetalBuffer::from_slice(device, &page_table).expect("page table");
        let output =
            MetalBuffer::empty(device, (num_q_heads * head_dim) as usize * 2).expect("output");

        kernels
            .flash_attention_paged(
                &ctx,
                &q,
                &k,
                &v,
                &page_table_buf,
                &output,
                head_dim,
                kv_len,
                0,
                page_size_tokens,
                num_q_heads,
                num_kv_heads,
            )
            .expect("flash paged");

        let out = output.as_slice::<f16>();
        assert!(
            out[1].to_f32() > out[0].to_f32(),
            "paged flash attention should read logical token 0 from physical page 1"
        );
    }

    #[test]
    fn flash_attention_window_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let num_q_heads: u32 = 2;
        let num_kv_heads: u32 = 1;
        let head_dim: u32 = 64;
        let kv_len: u32 = 20;
        let current_pos: u32 = 19;

        let q_data: Vec<f16> = (0..num_q_heads * head_dim)
            .map(|i| f16::from_f32((i as f32 * 0.13).sin() * 0.5))
            .collect();
        let k_data: Vec<f16> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| f16::from_f32((i as f32 * 0.07).cos() * 0.3))
            .collect();
        let v_data: Vec<f16> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| f16::from_f32((i as f32 * 0.11).sin() * 0.4))
            .collect();

        let q = MetalBuffer::from_slice(device, &q_data).expect("q");
        let k = MetalBuffer::from_slice(device, &k_data).expect("k");
        let v = MetalBuffer::from_slice(device, &v_data).expect("v");

        let nh = num_q_heads as usize;
        let nkv = num_kv_heads as usize;
        let hd = head_dim as usize;

        // CPU reference: scores over [win_start, current_pos], softmax
        // (kernel uses scale 1.0), weighted V sum. window == 0 is unlimited.
        let cpu_reference = |window: u32| -> Vec<f32> {
            let seq_len = kv_len.min(current_pos + 1) as usize;
            let win_start = if window > 0 && current_pos + 1 > window {
                (current_pos + 1 - window) as usize
            } else {
                0
            };
            let mut out = vec![0.0f32; nh * hd];
            for h in 0..nh {
                let kv_head = h / (nh / nkv);
                let scores: Vec<f32> = (win_start..seq_len)
                    .map(|t| {
                        (0..hd)
                            .map(|d| {
                                q_data[h * hd + d].to_f32()
                                    * k_data[(t * nkv + kv_head) * hd + d].to_f32()
                            })
                            .sum()
                    })
                    .collect();
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = weights.iter().sum();
                for (wi, t) in (win_start..seq_len).enumerate() {
                    let w = weights[wi] / sum_exp;
                    for d in 0..hd {
                        out[h * hd + d] += w * v_data[(t * nkv + kv_head) * hd + d].to_f32();
                    }
                }
            }
            out
        };

        // window = 8 restricts attention to positions [12, 19];
        // window = 0 attends all 20 positions (unlimited).
        for window in [8_u32, 0_u32] {
            let output = MetalBuffer::empty(device, nh * hd * 2).expect("out");
            kernels
                .flash_attention(
                    &ctx,
                    &q,
                    &k,
                    &v,
                    &output,
                    head_dim,
                    kv_len,
                    current_pos,
                    num_q_heads,
                    num_kv_heads,
                    window,
                )
                .expect("flash attn");
            let want = cpu_reference(window);
            assert_close(
                &read_f16(&output),
                &want,
                0.02,
                &format!("flash_attention window={window}"),
            );
        }
    }

    #[test]
    fn flash_attention_prefill_window_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let seq_len: u32 = 6;
        let kv_len: u32 = 10;
        let num_heads: u32 = 2;
        let num_kv_heads: u32 = 1;
        let head_dim: u32 = 32;
        let scale: f32 = 1.0;

        // Q [seq_len, num_heads, head_dim]; K/V [kv_len, num_kv_heads, head_dim]
        let q_data: Vec<f16> = (0..seq_len * num_heads * head_dim)
            .map(|i| f16::from_f32((i as f32 * 0.19).sin() * 0.5))
            .collect();
        let k_data: Vec<f16> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| f16::from_f32((i as f32 * 0.07).cos() * 0.3))
            .collect();
        let v_data: Vec<f16> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| f16::from_f32((i as f32 * 0.11).sin() * 0.4))
            .collect();

        let q = MetalBuffer::from_slice(device, &q_data).expect("q");
        let k = MetalBuffer::from_slice(device, &k_data).expect("k");
        let v = MetalBuffer::from_slice(device, &v_data).expect("v");

        let sl = seq_len as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let hd = head_dim as usize;

        // CPU reference: query row qi sits at global position
        // kv_len - seq_len + qi and attends [win_start, q_global].
        let cpu_reference = |window: u32| -> Vec<f32> {
            let hist = (kv_len - seq_len) as usize;
            let mut out = vec![0.0f32; sl * nh * hd];
            for qi in 0..sl {
                let q_global = hist + qi;
                let win_start = if window > 0 && q_global + 1 > window as usize {
                    q_global + 1 - window as usize
                } else {
                    0
                };
                for h in 0..nh {
                    let kv_head = h / (nh / nkv);
                    let q_off = (qi * nh + h) * hd;
                    let scores: Vec<f32> = (win_start..=q_global)
                        .map(|t| {
                            scale
                                * (0..hd)
                                    .map(|d| {
                                        q_data[q_off + d].to_f32()
                                            * k_data[(t * nkv + kv_head) * hd + d].to_f32()
                                    })
                                    .sum::<f32>()
                        })
                        .collect();
                    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = weights.iter().sum();
                    for (wi, t) in (win_start..=q_global).enumerate() {
                        let w = weights[wi] / sum_exp;
                        for d in 0..hd {
                            out[q_off + d] += w * v_data[(t * nkv + kv_head) * hd + d].to_f32();
                        }
                    }
                }
            }
            out
        };

        // window = 4: row qi (global 4 + qi) attends [qi + 1, qi + 4];
        // window = 0 is the full causal reference.
        for window in [4_u32, 0_u32] {
            let output = MetalBuffer::empty(device, sl * nh * hd * 2).expect("out");
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
            kernels
                .flash_attention_prefill_into(
                    &mut batch,
                    &q,
                    &k,
                    &v,
                    &output,
                    seq_len,
                    kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    window,
                )
                .expect("prefill");
            batch.commit_and_wait().expect("commit");
            let want = cpu_reference(window);
            assert_close(
                &read_f16(&output),
                &want,
                0.02,
                &format!("flash_attention_prefill window={window}"),
            );
        }
    }

    /// The tiled prefill kernel must match the CPU online-softmax reference at
    /// the production Gemma shapes: `MQA` (1 KV head, 8 query heads), `head_dim` 256
    /// and 512, and a `kv_len` that spans several K/V tiles (tile = 4096/hd keys
    /// ⇒ 16 keys for hd=256, 8 for hd=512) so tile-boundary masking is covered.
    #[test]
    fn flash_attention_prefill_tiled_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        assert!(
            kernels.flash_attn_prefill_tiled_pipeline.is_some(),
            "tiled prefill pipeline missing"
        );

        let num_heads: u32 = 8;
        let num_kv_heads: u32 = 1;

        // hd=256 ⇒ 16 keys/tile, hd=512 ⇒ 8 keys/tile. seq_len=20 also crosses
        // the BR=8 query-block boundary; kv_len=37 spans ≥3 tiles each.
        for &(head_dim, scale) in &[(256_u32, 0.0625_f32), (512_u32, 0.0442_f32)] {
            let seq_len: u32 = 20;
            let kv_len: u32 = 37;
            let hd = head_dim as usize;
            let nh = num_heads as usize;
            let nkv = num_kv_heads as usize;
            let sl = seq_len as usize;

            let q_data: Vec<f16> = (0..seq_len * num_heads * head_dim)
                .map(|i| f16::from_f32((i as f32 * 0.013).sin() * 0.5))
                .collect();
            let k_data: Vec<f16> = (0..kv_len * num_kv_heads * head_dim)
                .map(|i| f16::from_f32((i as f32 * 0.009).cos() * 0.3))
                .collect();
            let v_data: Vec<f16> = (0..kv_len * num_kv_heads * head_dim)
                .map(|i| f16::from_f32((i as f32 * 0.017).sin() * 0.4))
                .collect();

            let q = MetalBuffer::from_slice(device, &q_data).expect("q");
            let k = MetalBuffer::from_slice(device, &k_data).expect("k");
            let v = MetalBuffer::from_slice(device, &v_data).expect("v");

            let cpu_reference = |window: u32| -> Vec<f32> {
                let hist = (kv_len - seq_len) as usize;
                let mut out = vec![0.0f32; sl * nh * hd];
                for qi in 0..sl {
                    let q_global = hist + qi;
                    let win_start = if window > 0 && q_global + 1 > window as usize {
                        q_global + 1 - window as usize
                    } else {
                        0
                    };
                    for h in 0..nh {
                        let kv_head = h / (nh / nkv);
                        let q_off = (qi * nh + h) * hd;
                        let scores: Vec<f32> = (win_start..=q_global)
                            .map(|t| {
                                scale
                                    * (0..hd)
                                        .map(|d| {
                                            q_data[q_off + d].to_f32()
                                                * k_data[(t * nkv + kv_head) * hd + d].to_f32()
                                        })
                                        .sum::<f32>()
                            })
                            .collect();
                        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let weights: Vec<f32> =
                            scores.iter().map(|s| (s - max_score).exp()).collect();
                        let sum_exp: f32 = weights.iter().sum();
                        for (wi, t) in (win_start..=q_global).enumerate() {
                            let w = weights[wi] / sum_exp;
                            for d in 0..hd {
                                out[q_off + d] += w * v_data[(t * nkv + kv_head) * hd + d].to_f32();
                            }
                        }
                    }
                }
                out
            };

            for window in [12_u32, 0_u32] {
                let output = MetalBuffer::empty(device, sl * nh * hd * 2).expect("out");
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
                kernels
                    .flash_attention_prefill_into(
                        &mut batch,
                        &q,
                        &k,
                        &v,
                        &output,
                        seq_len,
                        kv_len,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        scale,
                        window,
                    )
                    .expect("prefill");
                batch.commit_and_wait().expect("commit");
                let want = cpu_reference(window);
                assert_close(
                    &read_f16(&output),
                    &want,
                    0.02,
                    &format!("flash_attention_prefill_tiled hd={head_dim} window={window}"),
                );
            }
        }
    }

    // ---- quantized-weight kernels ----------------------------------------

    /// Deterministic pseudo-random byte stream for quant payloads.
    fn xorshift_bytes(seed: u32, n: usize) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 8) as u8
            })
            .collect()
    }

    /// Build a `Q4_0` payload of `rows × cols` (`cols % 32 == 0`) and its FP32
    /// dequantized expansion (row-major), with sane scales.
    fn q4_0_payload(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let blocks_per_row = cols / 32;
        let mut raw = Vec::with_capacity(rows * blocks_per_row * 18);
        let mut dq = vec![0.0f32; rows * cols];
        let rnd = xorshift_bytes(seed, rows * blocks_per_row * 17);
        let mut r = rnd.iter().copied();
        for row in 0..rows {
            for b in 0..blocks_per_row {
                let d = 0.01 + f32::from(r.next().unwrap()) / 2550.0;
                raw.extend_from_slice(&f16::from_f32(d).to_le_bytes());
                let d = f16::from_f32(d).to_f32();
                for i in 0..16 {
                    let q = r.next().unwrap();
                    raw.push(q);
                    dq[row * cols + b * 32 + i] = (f32::from(q & 0x0F) - 8.0) * d;
                    dq[row * cols + b * 32 + 16 + i] = (f32::from(q >> 4) - 8.0) * d;
                }
            }
        }
        (raw, dq)
    }

    /// Build a `TQ2_0` payload of `rows × cols` (`cols % 256 == 0`) and its FP32
    /// dequantized expansion (row-major).
    fn tq2_0_payload(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let blocks_per_row = cols / 256;
        let mut raw = Vec::with_capacity(rows * blocks_per_row * 66);
        let mut dq = vec![0.0f32; rows * cols];
        let rnd = xorshift_bytes(seed, rows * blocks_per_row * 65);
        let mut r = rnd.iter().copied();
        for row in 0..rows {
            for b in 0..blocks_per_row {
                let mut qs = [0u8; 64];
                for q in &mut qs {
                    *q = r.next().unwrap();
                }
                let d = 0.05 + f32::from(r.next().unwrap()) / 1275.0;
                raw.extend_from_slice(&qs);
                raw.extend_from_slice(&f16::from_f32(d).to_le_bytes());
                let d = f16::from_f32(d).to_f32();
                // elem (j>>5)*128 + l*32 + (j&31) ← bits 2l..2l+1 of qs[j]
                for (j, &q) in qs.iter().enumerate() {
                    let base = b * 256 + (j >> 5) * 128 + (j & 31);
                    for l in 0..4 {
                        dq[row * cols + base + l * 32] = (f32::from((q >> (2 * l)) & 3) - 1.0) * d;
                    }
                }
            }
        }
        (raw, dq)
    }

    /// Repack an `AoS` `TQ2_0` payload (66-byte interleaved blocks) into `SoA`:
    /// returns `(qs, scales)` where `qs` is `rows*blocks*64` aligned bytes and
    /// `scales` is `rows*blocks` fp16 values.
    fn tq2_0_repack_soa(raw: &[u8], rows: usize, cols: usize) -> (Vec<u8>, Vec<f16>) {
        let blocks = cols / 256;
        let mut qs = Vec::with_capacity(rows * blocks * 64);
        let mut scales = Vec::with_capacity(rows * blocks);
        for blk in 0..rows * blocks {
            let off = blk * 66;
            qs.extend_from_slice(&raw[off..off + 64]);
            scales.push(f16::from_le_bytes([raw[off + 64], raw[off + 65]]));
        }
        (qs, scales)
    }

    fn test_vector(cols: usize) -> (Vec<f16>, Vec<f32>) {
        let v16: Vec<f16> = (0..cols)
            .map(|i| f16::from_f32((i as f32 * 0.731).sin()))
            .collect();
        let v32 = v16.iter().map(|v| v.to_f32()).collect();
        (v16, v32)
    }

    fn cpu_matvec(dq: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|r| (0..cols).map(|c| dq[r * cols + c] * v[c]).sum())
            .collect()
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let scale = w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol * scale,
                "{label}[{i}]: got {g}, want {w}"
            );
        }
    }

    /// Run with: `cargo test -p local-metal --release matvec_decode_timing -- --ignored --nocapture`
    #[test]
    #[ignore = "perf timing, run manually"]
    #[allow(clippy::too_many_lines)]
    fn matvec_decode_timing() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // Build the per-layer FFN + attn weight buffers once (TQ2_0), reused.
        let mk = |rows: usize, cols: usize| {
            let (raw, _) = tq2_0_payload(rows, cols, (rows * 7 + cols) as u32);
            MetalBuffer::from_slice(device, &raw).expect("m")
        };
        let h = 1536;
        let inter = 6144;
        let layers = 35;
        let iters = 50;
        // 35 DISTINCT weight sets so the kernel reads cold from RAM each layer
        // (the real decode path has no weight reuse within a token).
        let gate: Vec<_> = (0..layers).map(|_| mk(inter, h)).collect();
        let up: Vec<_> = (0..layers).map(|_| mk(inter, h)).collect();
        let down: Vec<_> = (0..layers).map(|_| mk(h, inter)).collect();
        let q: Vec<_> = (0..layers).map(|_| mk(2048, h)).collect();
        let o: Vec<_> = (0..layers).map(|_| mk(h, 2048)).collect();
        let vin = MetalBuffer::from_slice(device, &test_vector(h).0).expect("vin");
        let vinter = MetalBuffer::from_slice(device, &test_vector(inter).0).expect("vinter");
        let v2048 = MetalBuffer::from_slice(device, &test_vector(2048).0).expect("v2048");
        let o_inter = MetalBuffer::empty(device, inter * size_of::<f16>()).expect("oi");
        let o_h = MetalBuffer::empty(device, h * size_of::<f16>()).expect("oh");
        let o_2048 = MetalBuffer::empty(device, 2048 * size_of::<f16>()).expect("o2048");

        // Warmup.
        for _ in 0..3 {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
            for l in 0..layers {
                kernels
                    .matvec_tq2_0_into(&mut batch, &gate[l], &vin, &o_inter, inter as u32, h as u32)
                    .unwrap();
                kernels
                    .matvec_tq2_0_into(&mut batch, &up[l], &vin, &o_inter, inter as u32, h as u32)
                    .unwrap();
                kernels
                    .matvec_tq2_0_into(&mut batch, &down[l], &vinter, &o_h, h as u32, inter as u32)
                    .unwrap();
                kernels
                    .matvec_tq2_0_into(&mut batch, &q[l], &vin, &o_2048, 2048, h as u32)
                    .unwrap();
                kernels
                    .matvec_tq2_0_into(&mut batch, &o[l], &v2048, &o_h, h as u32, 2048)
                    .unwrap();
            }
            batch.commit_and_wait().unwrap();
        }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
            for l in 0..layers {
                kernels
                    .matvec_tq2_0_into(&mut batch, &gate[l], &vin, &o_inter, inter as u32, h as u32)
                    .unwrap();
                kernels
                    .matvec_tq2_0_into(&mut batch, &up[l], &vin, &o_inter, inter as u32, h as u32)
                    .unwrap();
                kernels
                    .matvec_tq2_0_into(&mut batch, &down[l], &vinter, &o_h, h as u32, inter as u32)
                    .unwrap();
                kernels
                    .matvec_tq2_0_into(&mut batch, &q[l], &vin, &o_2048, 2048, h as u32)
                    .unwrap();
                kernels
                    .matvec_tq2_0_into(&mut batch, &o[l], &v2048, &o_h, h as u32, 2048)
                    .unwrap();
            }
            batch.commit_and_wait().unwrap();
        }
        let per_tok = start.elapsed().as_secs_f64() / f64::from(iters);
        let bytes =
            (layers as f64) * ((inter * h * 2 + h * inter + 2048 * h + h * 2048) as f64) * 66.0
                / 256.0;
        eprintln!(
            "[timing] FFN+attn matvecs only: {:.3} ms/token => {:.1} tok/s, weight read {:.1} MB, {:.0} GB/s",
            per_tok * 1e3,
            1.0 / per_tok,
            bytes / 1e6,
            bytes / per_tok / 1e9
        );

        // Q4_0 variant at identical shapes: tells us whether TQ2's effective
        // bandwidth is capped by its heavier 2-bit unpack ALU vs Q4_0's
        // pre-scaled no-shift nibble path. Q4_0 block = 32 elems / 18 B.
        let mkq = |rows: usize, cols: usize| {
            let (raw, _) = q4_0_payload(rows, cols, (rows * 13 + cols) as u32);
            MetalBuffer::from_slice(device, &raw).expect("mq")
        };
        let gate_q: Vec<_> = (0..layers).map(|_| mkq(inter, h)).collect();
        let up_q: Vec<_> = (0..layers).map(|_| mkq(inter, h)).collect();
        let down_q: Vec<_> = (0..layers).map(|_| mkq(h, inter)).collect();
        let q_q: Vec<_> = (0..layers).map(|_| mkq(2048, h)).collect();
        let o_q: Vec<_> = (0..layers).map(|_| mkq(h, 2048)).collect();
        let run_q4 = |batch: &mut crate::batch::CommandBatch| {
            for l in 0..layers {
                kernels
                    .matvec_q4_0_into(batch, &gate_q[l], &vin, &o_inter, inter as u32, h as u32)
                    .unwrap();
                kernels
                    .matvec_q4_0_into(batch, &up_q[l], &vin, &o_inter, inter as u32, h as u32)
                    .unwrap();
                kernels
                    .matvec_q4_0_into(batch, &down_q[l], &vinter, &o_h, h as u32, inter as u32)
                    .unwrap();
                kernels
                    .matvec_q4_0_into(batch, &q_q[l], &vin, &o_2048, 2048, h as u32)
                    .unwrap();
                kernels
                    .matvec_q4_0_into(batch, &o_q[l], &v2048, &o_h, h as u32, 2048)
                    .unwrap();
            }
        };
        for _ in 0..3 {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
            run_q4(&mut batch);
            batch.commit_and_wait().unwrap();
        }
        let start_q = std::time::Instant::now();
        for _ in 0..iters {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
            run_q4(&mut batch);
            batch.commit_and_wait().unwrap();
        }
        let per_tok_q = start_q.elapsed().as_secs_f64() / f64::from(iters);
        let bytes_q =
            (layers as f64) * ((inter * h * 2 + h * inter + 2048 * h + h * 2048) as f64) * 18.0
                / 32.0;
        eprintln!(
            "[timing] Q4_0 matvecs (same shapes): {:.3} ms/token => {:.1} tok/s, {:.1} MB, {:.0} GB/s",
            per_tok_q * 1e3,
            1.0 / per_tok_q,
            bytes_q / 1e6,
            bytes_q / per_tok_q / 1e9
        );

        // SoA TQ2_0 variant: same shapes, repacked into aligned qs + separate
        // scales, to test whether removing the 66-byte stub (sector straddle)
        // and the 32× redundant scale loads lifts achieved bandwidth.
        let mk_soa = |rows: usize, cols: usize| {
            let (raw, _) = tq2_0_payload(rows, cols, (rows * 7 + cols) as u32);
            let (qs, scales) = tq2_0_repack_soa(&raw, rows, cols);
            (
                MetalBuffer::from_slice(device, &qs).expect("qs"),
                MetalBuffer::from_slice(device, &scales).expect("sc"),
            )
        };
        let gate_s: Vec<_> = (0..layers).map(|_| mk_soa(inter, h)).collect();
        let up_s: Vec<_> = (0..layers).map(|_| mk_soa(inter, h)).collect();
        let down_s: Vec<_> = (0..layers).map(|_| mk_soa(h, inter)).collect();
        let q_s: Vec<_> = (0..layers).map(|_| mk_soa(2048, h)).collect();
        let o_s: Vec<_> = (0..layers).map(|_| mk_soa(h, 2048)).collect();
        let run_soa = |batch: &mut crate::batch::CommandBatch| {
            for l in 0..layers {
                kernels
                    .matvec_tq2_0_soa_into(
                        batch,
                        &gate_s[l].0,
                        &gate_s[l].1,
                        &vin,
                        &o_inter,
                        inter as u32,
                        h as u32,
                    )
                    .unwrap();
                kernels
                    .matvec_tq2_0_soa_into(
                        batch,
                        &up_s[l].0,
                        &up_s[l].1,
                        &vin,
                        &o_inter,
                        inter as u32,
                        h as u32,
                    )
                    .unwrap();
                kernels
                    .matvec_tq2_0_soa_into(
                        batch,
                        &down_s[l].0,
                        &down_s[l].1,
                        &vinter,
                        &o_h,
                        h as u32,
                        inter as u32,
                    )
                    .unwrap();
                kernels
                    .matvec_tq2_0_soa_into(
                        batch, &q_s[l].0, &q_s[l].1, &vin, &o_2048, 2048, h as u32,
                    )
                    .unwrap();
                kernels
                    .matvec_tq2_0_soa_into(
                        batch, &o_s[l].0, &o_s[l].1, &v2048, &o_h, h as u32, 2048,
                    )
                    .unwrap();
            }
        };
        for _ in 0..3 {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
            run_soa(&mut batch);
            batch.commit_and_wait().unwrap();
        }
        let start_s = std::time::Instant::now();
        for _ in 0..iters {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
            run_soa(&mut batch);
            batch.commit_and_wait().unwrap();
        }
        let per_tok_s = start_s.elapsed().as_secs_f64() / f64::from(iters);
        // SoA reads 64 qs bytes + 2 scale bytes per block (vs 66 AoS) — same
        // logical bytes, but now aligned.
        eprintln!(
            "[timing] SoA TQ2_0 matvecs only: {:.3} ms/token => {:.1} tok/s, {:.0} GB/s ({:+.1}% vs AoS)",
            per_tok_s * 1e3,
            1.0 / per_tok_s,
            bytes / per_tok_s / 1e9,
            (per_tok / per_tok_s - 1.0) * 100.0
        );

        // Pure launch-overhead probe: many trivial scale dispatches on a small
        // buffer, batched into one command buffer (like the real decode token).
        let small = MetalBuffer::empty(device, h * size_of::<f16>()).expect("small");
        let n_disp = 900usize;
        for _ in 0..3 {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
            for _ in 0..n_disp {
                kernels
                    .scale_in_place_gpu_into(&mut batch, &small, 1.0001, h as u32)
                    .unwrap();
            }
            batch.commit_and_wait().unwrap();
        }
        let s2 = std::time::Instant::now();
        for _ in 0..iters {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
            for _ in 0..n_disp {
                kernels
                    .scale_in_place_gpu_into(&mut batch, &small, 1.0001, h as u32)
                    .unwrap();
            }
            batch.commit_and_wait().unwrap();
        }
        let per = s2.elapsed().as_secs_f64() / f64::from(iters);
        eprintln!(
            "[timing] {n_disp} trivial dispatches: {:.3} ms => {:.2} us/dispatch",
            per * 1e3,
            per / n_disp as f64 * 1e6
        );
    }

    /// Run with: `cargo test -p local-metal --release matmul_prefill_timing -- --ignored --nocapture`
    /// Characterizes the tiled quantized matmul (prefill path) across M to see
    /// whether weight reads amortize across the token dimension.
    #[test]
    #[ignore = "perf timing, run manually"]
    fn matmul_prefill_timing() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // gate-like shape: N output rows, K hidden.
        let n = 6144usize;
        let k = 1536usize;
        let layers = 35usize;
        let iters = 30usize;
        let (raw, _) = q4_0_payload(n, k, 12345);
        let w: Vec<_> = (0..layers)
            .map(|_| MetalBuffer::from_slice(device, &raw).expect("w"))
            .collect();
        for &m in &[8usize, 16, 17, 24, 32, 48, 64] {
            let a: Vec<f16> = (0..m * k)
                .map(|i| f16::from_f32((i as f32 * 0.013).sin()))
                .collect();
            let abuf = MetalBuffer::from_slice(device, &a).expect("a");
            let out = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("o");
            for _ in 0..3 {
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
                for weight in w.iter().take(layers) {
                    kernels
                        .matmul_nt_q4_0_into(
                            &mut batch, &abuf, weight, &out, m as u32, k as u32, n as u32,
                        )
                        .unwrap();
                }
                batch.commit_and_wait().unwrap();
            }
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
                for weight in w.iter().take(layers) {
                    kernels
                        .matmul_nt_q4_0_into(
                            &mut batch, &abuf, weight, &out, m as u32, k as u32, n as u32,
                        )
                        .unwrap();
                }
                batch.commit_and_wait().unwrap();
            }
            let per = start.elapsed().as_secs_f64() / iters as f64;
            let wbytes = layers as f64 * (n * k) as f64 * 18.0 / 32.0;
            eprintln!(
                "[matmul Q4_0 MMA] M={m:>3}: {:.2} ms/forward ({:.3} ms/tok, {:.0} GB/s weight)",
                per * 1e3,
                per * 1e3 / m as f64,
                wbytes / per / 1e9
            );
        }
        // Compare the multivec (GEMV-style) small-batch path at M<=16.
        for &m in &[4usize, 8, 12, 16] {
            let a: Vec<f16> = (0..m * k)
                .map(|i| f16::from_f32((i as f32 * 0.013).sin()))
                .collect();
            let abuf = MetalBuffer::from_slice(device, &a).expect("a");
            let out = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("o");
            for _ in 0..3 {
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
                for weight in w.iter().take(layers) {
                    kernels
                        .multivec_nt_q4_0_into(
                            &mut batch, &abuf, weight, &out, m as u32, k as u32, n as u32,
                        )
                        .unwrap();
                }
                batch.commit_and_wait().unwrap();
            }
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("b");
                for weight in w.iter().take(layers) {
                    kernels
                        .multivec_nt_q4_0_into(
                            &mut batch, &abuf, weight, &out, m as u32, k as u32, n as u32,
                        )
                        .unwrap();
                }
                batch.commit_and_wait().unwrap();
            }
            let per = start.elapsed().as_secs_f64() / iters as f64;
            eprintln!(
                "[matmul Q4_0 MVEC] M={m:>3}: {:.2} ms/forward ({:.3} ms/tok)",
                per * 1e3,
                per * 1e3 / m as f64,
            );
        }
    }

    #[test]
    fn matvec_q4_0_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (rows, cols) = (5, 96); // 3 blocks per row
        let (raw, dq) = q4_0_payload(rows, cols, 0xBEEF);
        let (v16, v32) = test_vector(cols);

        let m = MetalBuffer::from_slice(device, &raw).expect("matrix");
        let v = MetalBuffer::from_slice(device, &v16).expect("vector");
        let out = MetalBuffer::empty(device, rows * size_of::<f16>()).expect("out");
        kernels
            .matvec_q4_0(&ctx, &m, &v, &out, rows as u32, cols as u32)
            .expect("matvec_q4_0");

        let want = cpu_matvec(&dq, &v32, rows, cols);
        assert_close(&read_f16(&out), &want, 0.02, "matvec_q4_0");
    }

    #[test]
    fn matvec_q4_0_f32out_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (rows, cols) = (7, 64);
        let (raw, dq) = q4_0_payload(rows, cols, 0xCAFE);
        let (v16, v32) = test_vector(cols);

        let m = MetalBuffer::from_slice(device, &raw).expect("matrix");
        let v = MetalBuffer::from_slice(device, &v16).expect("vector");
        let out = MetalBuffer::empty(device, rows * size_of::<f32>()).expect("out");
        kernels
            .matvec_q4_0_f32out(&ctx, &m, &v, &out, rows as u32, cols as u32)
            .expect("matvec_q4_0_f32out");

        let want = cpu_matvec(&dq, &v32, rows, cols);
        assert_close(out.as_slice::<f32>(), &want, 0.01, "matvec_q4_0_f32out");
    }

    #[test]
    fn matvec_q4_0_into_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (rows, cols) = (5, 96);
        let (raw, dq) = q4_0_payload(rows, cols, 0xB47C);
        let (v16, v32) = test_vector(cols);

        let m = MetalBuffer::from_slice(device, &raw).expect("matrix");
        let v = MetalBuffer::from_slice(device, &v16).expect("vector");
        let out = MetalBuffer::empty(device, rows * size_of::<f16>()).expect("out");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .matvec_q4_0_into(&mut batch, &m, &v, &out, rows as u32, cols as u32)
            .expect("matvec_q4_0_into");
        batch.commit_and_wait().expect("commit");

        let want = cpu_matvec(&dq, &v32, rows, cols);
        assert_close(&read_f16(&out), &want, 0.02, "matvec_q4_0_into");
    }

    #[test]
    fn matvec_q4_0_f32out_into_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (rows, cols) = (7, 64);
        let (raw, dq) = q4_0_payload(rows, cols, 0xF32A);
        let (v16, v32) = test_vector(cols);

        let m = MetalBuffer::from_slice(device, &raw).expect("matrix");
        let v = MetalBuffer::from_slice(device, &v16).expect("vector");
        let out = MetalBuffer::empty(device, rows * size_of::<f32>()).expect("out");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .matvec_q4_0_f32out_into(&mut batch, &m, &v, &out, rows as u32, cols as u32)
            .expect("matvec_q4_0_f32out_into");
        batch.commit_and_wait().expect("commit");

        let want = cpu_matvec(&dq, &v32, rows, cols);
        assert_close(
            out.as_slice::<f32>(),
            &want,
            0.01,
            "matvec_q4_0_f32out_into",
        );
    }

    #[test]
    fn matvec_tq2_0_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (rows, cols) = (4, 512); // 2 blocks per row
        let (raw, dq) = tq2_0_payload(rows, cols, 0xFEED);
        let (v16, v32) = test_vector(cols);

        let m = MetalBuffer::from_slice(device, &raw).expect("matrix");
        let v = MetalBuffer::from_slice(device, &v16).expect("vector");
        let out = MetalBuffer::empty(device, rows * size_of::<f16>()).expect("out");
        kernels
            .matvec_tq2_0(&ctx, &m, &v, &out, rows as u32, cols as u32)
            .expect("matvec_tq2_0");

        let want = cpu_matvec(&dq, &v32, rows, cols);
        assert_close(&read_f16(&out), &want, 0.02, "matvec_tq2_0");
    }

    #[test]
    fn matvec_tq2_0_f32out_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (rows, cols) = (6, 256);
        let (raw, dq) = tq2_0_payload(rows, cols, 0xD00D);
        let (v16, v32) = test_vector(cols);

        let m = MetalBuffer::from_slice(device, &raw).expect("matrix");
        let v = MetalBuffer::from_slice(device, &v16).expect("vector");
        let out = MetalBuffer::empty(device, rows * size_of::<f32>()).expect("out");
        kernels
            .matvec_tq2_0_f32out(&ctx, &m, &v, &out, rows as u32, cols as u32)
            .expect("matvec_tq2_0_f32out");

        let want = cpu_matvec(&dq, &v32, rows, cols);
        assert_close(out.as_slice::<f32>(), &want, 0.01, "matvec_tq2_0_f32out");
    }

    #[test]
    fn matvec_tq2_0_into_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (rows, cols) = (4, 512);
        let (raw, dq) = tq2_0_payload(rows, cols, 0xA551);
        let (v16, v32) = test_vector(cols);

        let m = MetalBuffer::from_slice(device, &raw).expect("matrix");
        let v = MetalBuffer::from_slice(device, &v16).expect("vector");
        let out = MetalBuffer::empty(device, rows * size_of::<f16>()).expect("out");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .matvec_tq2_0_into(&mut batch, &m, &v, &out, rows as u32, cols as u32)
            .expect("matvec_tq2_0_into");
        batch.commit_and_wait().expect("commit");

        let want = cpu_matvec(&dq, &v32, rows, cols);
        assert_close(&read_f16(&out), &want, 0.02, "matvec_tq2_0_into");
    }

    #[test]
    fn matvec_tq2_0_soa_into_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (rows, cols) = (6, 512);
        let (raw, dq) = tq2_0_payload(rows, cols, 0x50A1);
        let (qs, scales) = tq2_0_repack_soa(&raw, rows, cols);
        let (v16, v32) = test_vector(cols);

        let qbuf = MetalBuffer::from_slice(device, &qs).expect("qs");
        let sbuf = MetalBuffer::from_slice(device, &scales).expect("scales");
        let v = MetalBuffer::from_slice(device, &v16).expect("vector");
        let out = MetalBuffer::empty(device, rows * size_of::<f16>()).expect("out");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .matvec_tq2_0_soa_into(&mut batch, &qbuf, &sbuf, &v, &out, rows as u32, cols as u32)
            .expect("matvec_tq2_0_soa_into");
        batch.commit_and_wait().expect("commit");

        let want = cpu_matvec(&dq, &v32, rows, cols);
        assert_close(&read_f16(&out), &want, 0.02, "matvec_tq2_0_soa_into");
    }

    #[test]
    fn matvec_tq2_0_f32out_into_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (rows, cols) = (6, 256);
        let (raw, dq) = tq2_0_payload(rows, cols, 0xF320);
        let (v16, v32) = test_vector(cols);

        let m = MetalBuffer::from_slice(device, &raw).expect("matrix");
        let v = MetalBuffer::from_slice(device, &v16).expect("vector");
        let out = MetalBuffer::empty(device, rows * size_of::<f32>()).expect("out");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .matvec_tq2_0_f32out_into(&mut batch, &m, &v, &out, rows as u32, cols as u32)
            .expect("matvec_tq2_0_f32out_into");
        batch.commit_and_wait().expect("commit");

        let want = cpu_matvec(&dq, &v32, rows, cols);
        assert_close(
            out.as_slice::<f32>(),
            &want,
            0.01,
            "matvec_tq2_0_f32out_into",
        );
    }

    /// Small-M `Q4_0` matmul (`matmul_nt_q4_0_smallm`, the M<=8 batched-decode
    /// fast path) matches the CPU reference across M=1..8 and N not a multiple of
    /// the 64-wide tile, at model-realistic K (gate/up K=1536, down K=6144).
    #[test]
    fn matmul_nt_q4_0_smallm_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        assert!(
            kernels.matmul_nt_q4_0_smallm_pipeline.is_some(),
            "Q4_0 small-M pipeline failed to compile"
        );
        let device = ctx.device();
        // (k, n): gate/up-like (N>K), down-like (K>N), and non-tile-aligned N.
        for &(k, n) in &[
            (1536usize, 6144usize),
            (6144usize, 1536usize),
            (1536usize, 100usize),
        ] {
            let (raw, dq) = q4_0_payload(n, k, 0x5A1Du32.wrapping_add(n as u32));
            let w = MetalBuffer::from_slice(device, &raw).expect("w");
            for m in 1..=8usize {
                let a16: Vec<f16> = (0..m * k)
                    .map(|i| f16::from_f32(((i as f32) * 0.011).sin() * 0.6))
                    .collect();
                let a32: Vec<f32> = a16.iter().map(|v| v.to_f32()).collect();
                let a = MetalBuffer::from_slice(device, &a16).expect("a");
                let c = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("c");
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
                kernels
                    .matmul_nt_q4_0_into(&mut batch, &a, &w, &c, m as u32, k as u32, n as u32)
                    .expect("q4 smallm");
                batch.commit_and_wait().expect("commit");
                let want = cpu_matmul_nt(&a32, &dq, m, k, n);
                assert_close(&read_f16(&c), &want, 0.03, "matmul_nt_q4_0_smallm");
            }
        }
    }

    /// Small-M `TQ2_0` matmul (`matmul_nt_tq2_0_smallm`, the M<=8 batched-decode
    /// fast path) matches the CPU reference across M=1..8 and N not a multiple of
    /// the 64-wide tile, at model-realistic K (K multiple of 256). This kernel
    /// makes small batched matmuls bandwidth-bound instead of re-decoding each
    /// quant block per output element.
    #[test]
    fn matmul_nt_tq2_0_smallm_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        assert!(
            kernels.matmul_nt_tq2_0_smallm_pipeline.is_some(),
            "TQ2_0 small-M pipeline failed to compile"
        );
        let device = ctx.device();
        // (k, n): gate/up-like (N>K), down-like (K>N), and non-tile-aligned N.
        // K must be a multiple of 256 for TQ2_0.
        for &(k, n) in &[
            (1536usize, 6144usize),
            (6144usize, 1536usize),
            (1536usize, 100usize),
        ] {
            let (raw, dq) = tq2_0_payload(n, k, 0x7C3Fu32.wrapping_add(n as u32));
            let w = MetalBuffer::from_slice(device, &raw).expect("w");
            for m in 1..=8usize {
                let a16: Vec<f16> = (0..m * k)
                    .map(|i| f16::from_f32(((i as f32) * 0.011).sin() * 0.6))
                    .collect();
                let a32: Vec<f32> = a16.iter().map(|v| v.to_f32()).collect();
                let a = MetalBuffer::from_slice(device, &a16).expect("a");
                let c = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("c");
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
                kernels
                    .matmul_nt_tq2_0_into(&mut batch, &a, &w, &c, m as u32, k as u32, n as u32)
                    .expect("tq2 smallm");
                batch.commit_and_wait().expect("commit");
                let want = cpu_matmul_nt(&a32, &dq, m, k, n);
                assert_close(&read_f16(&c), &want, 0.03, "matmul_nt_tq2_0_smallm");
            }
        }
    }

    #[test]
    fn matmul_nt_tq2_0_batchm_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let bm = kernels
            .matmul_nt_tq2_0_batchm_pipeline
            .as_ref()
            .expect("TQ2_0 batchm pipeline failed to compile");
        let device = ctx.device();
        // (k, n): gate/up-like (N>K), down-like (K>N), and non-tile-aligned N.
        for &(k, n) in &[
            (1536usize, 6144usize),
            (6144usize, 1536usize),
            (1536usize, 100usize),
        ] {
            let (raw, dq) = tq2_0_payload(n, k, 0x51A3u32.wrapping_add(n as u32));
            let w = MetalBuffer::from_slice(device, &raw).expect("w");
            for m in 1..=8usize {
                let a16: Vec<f16> = (0..m * k)
                    .map(|i| f16::from_f32(((i as f32) * 0.013).cos() * 0.55))
                    .collect();
                let a32: Vec<f32> = a16.iter().map(|v| v.to_f32()).collect();
                let a = MetalBuffer::from_slice(device, &a16).expect("a");
                let c = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("c");
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
                Kernels::matmul_nt_quant_batchm_into(
                    bm, &mut batch, &a, &w, &c, m as u32, k as u32, n as u32,
                );
                batch.commit_and_wait().expect("commit");
                let want = cpu_matmul_nt(&a32, &dq, m, k, n);
                assert_close(&read_f16(&c), &want, 0.03, "matmul_nt_tq2_0_batchm");
            }
        }
    }

    fn cpu_matmul_nt(a: &[f32], dq: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k).map(|p| a[i * k + p] * dq[j * k + p]).sum();
            }
        }
        c
    }

    /// MMA quantized matmul correctness across tile boundaries (M, N, K each
    /// span multiple 32-tiles, with non-aligned M). Exercises the
    /// `matmul_nt_*_mma` kernels that back `matmul_nt_*_into` when
    /// `LOCAL_AI_MMA` is enabled (the default).
    #[test]
    fn matmul_nt_mma_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        assert!(
            kernels.matmul_nt_q4_0_mma_pipeline.is_some(),
            "Q4_0 MMA pipeline failed to compile"
        );
        assert!(
            kernels.matmul_nt_tq2_0_mma_pipeline.is_some(),
            "TQ2_0 MMA pipeline failed to compile"
        );
        let device = ctx.device();
        // m=70 (3 tiles, non-aligned), n=96 (3 tiles), k spans tiles.
        let m = 70usize;
        for &(k, n) in &[(64usize, 96usize), (256usize, 128usize)] {
            // Q4_0
            let (raw, dq) = q4_0_payload(n, k, 0x51A7);
            let a16: Vec<f16> = (0..m * k)
                .map(|i| f16::from_f32(((i as f32) * 0.013).sin() * 0.7))
                .collect();
            let a32: Vec<f32> = a16.iter().map(|v| v.to_f32()).collect();
            let a = MetalBuffer::from_slice(device, &a16).expect("a");
            let w = MetalBuffer::from_slice(device, &raw).expect("w");
            let c = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("c");
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
            kernels
                .matmul_nt_q4_0_into(&mut batch, &a, &w, &c, m as u32, k as u32, n as u32)
                .expect("q4 mma");
            batch.commit_and_wait().expect("commit");
            let want = cpu_matmul_nt(&a32, &dq, m, k, n);
            assert_close(&read_f16(&c), &want, 0.03, "matmul_nt_q4_0_mma");
        }
        // TQ2_0 needs k multiple of 256.
        for &(k, n) in &[(256usize, 96usize), (512usize, 128usize)] {
            let (raw, dq) = tq2_0_payload(n, k, 0x2B19);
            let a16: Vec<f16> = (0..m * k)
                .map(|i| f16::from_f32(((i as f32) * 0.017).cos() * 0.6))
                .collect();
            let a32: Vec<f32> = a16.iter().map(|v| v.to_f32()).collect();
            let a = MetalBuffer::from_slice(device, &a16).expect("a");
            let w = MetalBuffer::from_slice(device, &raw).expect("w");
            let c = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("c");
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
            kernels
                .matmul_nt_tq2_0_into(&mut batch, &a, &w, &c, m as u32, k as u32, n as u32)
                .expect("tq2 mma");
            batch.commit_and_wait().expect("commit");
            let want = cpu_matmul_nt(&a32, &dq, m, k, n);
            assert_close(&read_f16(&c), &want, 0.04, "matmul_nt_tq2_0_mma");
        }
    }

    #[test]
    fn matmul_nt_q4_0_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // Deliberately not tile-aligned in m or n.
        let (m, k, n) = (5usize, 64usize, 37usize);
        let (raw, dq) = q4_0_payload(n, k, 0xABCD);
        let a16: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((i as f32 * 0.117).cos() * 0.5))
            .collect();
        let a32: Vec<f32> = a16.iter().map(|v| v.to_f32()).collect();

        let a = MetalBuffer::from_slice(device, &a16).expect("a");
        let w = MetalBuffer::from_slice(device, &raw).expect("w");
        let c = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("c");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .matmul_nt_q4_0_into(&mut batch, &a, &w, &c, m as u32, k as u32, n as u32)
            .expect("matmul_nt_q4_0");
        batch.commit_and_wait().expect("commit");

        let want = cpu_matmul_nt(&a32, &dq, m, k, n);
        assert_close(&read_f16(&c), &want, 0.03, "matmul_nt_q4_0");
    }

    #[test]
    fn matmul_nt_tq2_0_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (m, k, n) = (3usize, 256usize, 33usize);
        let (raw, dq) = tq2_0_payload(n, k, 0x1234);
        let a16: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((i as f32 * 0.213).sin() * 0.5))
            .collect();
        let a32: Vec<f32> = a16.iter().map(|v| v.to_f32()).collect();

        let a = MetalBuffer::from_slice(device, &a16).expect("a");
        let w = MetalBuffer::from_slice(device, &raw).expect("w");
        let c = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("c");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .matmul_nt_tq2_0_into(&mut batch, &a, &w, &c, m as u32, k as u32, n as u32)
            .expect("matmul_nt_tq2_0");
        batch.commit_and_wait().expect("commit");

        let want = cpu_matmul_nt(&a32, &dq, m, k, n);
        assert_close(&read_f16(&c), &want, 0.03, "matmul_nt_tq2_0");
    }

    #[test]
    fn multivec_nt_q4_0_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        // Small-batch verify shape (m≈5), n deliberately not tile-aligned.
        let (m, k, n) = (5usize, 128usize, 37usize);
        let (raw, dq) = q4_0_payload(n, k, 0xABCD);
        let a16: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((i as f32 * 0.117).cos() * 0.5))
            .collect();
        let a32: Vec<f32> = a16.iter().map(|v| v.to_f32()).collect();

        let a = MetalBuffer::from_slice(device, &a16).expect("a");
        let w = MetalBuffer::from_slice(device, &raw).expect("w");
        let c = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("c");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .multivec_nt_q4_0_into(&mut batch, &a, &w, &c, m as u32, k as u32, n as u32)
            .expect("multivec_nt_q4_0");
        batch.commit_and_wait().expect("commit");

        let want = cpu_matmul_nt(&a32, &dq, m, k, n);
        assert_close(&read_f16(&c), &want, 0.03, "multivec_nt_q4_0");
    }

    #[test]
    fn multivec_nt_tq2_0_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let (m, k, n) = (6usize, 512usize, 33usize);
        let (raw, dq) = tq2_0_payload(n, k, 0x1234);
        let a16: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((i as f32 * 0.213).sin() * 0.5))
            .collect();
        let a32: Vec<f32> = a16.iter().map(|v| v.to_f32()).collect();

        let a = MetalBuffer::from_slice(device, &a16).expect("a");
        let w = MetalBuffer::from_slice(device, &raw).expect("w");
        let c = MetalBuffer::empty(device, m * n * size_of::<f16>()).expect("c");
        let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
        kernels
            .multivec_nt_tq2_0_into(&mut batch, &a, &w, &c, m as u32, k as u32, n as u32)
            .expect("multivec_nt_tq2_0");
        batch.commit_and_wait().expect("commit");

        let want = cpu_matmul_nt(&a32, &dq, m, k, n);
        assert_close(&read_f16(&c), &want, 0.03, "multivec_nt_tq2_0");
    }

    /// Same comparison at the model's real layer dimensions, against the
    /// FP16 matvec on the dequantized expansion of the same payload.
    #[test]
    fn quant_matvec_matches_f16_matvec_at_model_dims() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        for &(rows, cols, is_tq2) in &[
            (2048usize, 1536usize, false), // attn_q
            (1536, 2048, false),           // attn_output
            (12288, 1536, true),           // ffn_gate/up (wide layers)
            (1536, 12288, true),           // ffn_down (wide layers)
        ] {
            let (raw, dq) = if is_tq2 {
                tq2_0_payload(rows, cols, 0x5EED)
            } else {
                q4_0_payload(rows, cols, 0x5EED)
            };
            let dq16: Vec<f16> = dq.iter().map(|&v| f16::from_f32(v)).collect();
            let (v16, _) = test_vector(cols);

            let mq = MetalBuffer::from_slice(device, &raw).expect("quant matrix");
            let mf = MetalBuffer::from_slice(device, &dq16).expect("f16 matrix");
            let v = MetalBuffer::from_slice(device, &v16).expect("vector");
            let out_q = MetalBuffer::empty(device, rows * size_of::<f16>()).expect("out q");
            let out_f = MetalBuffer::empty(device, rows * size_of::<f16>()).expect("out f");

            if is_tq2 {
                kernels
                    .matvec_tq2_0(&ctx, &mq, &v, &out_q, rows as u32, cols as u32)
                    .expect("quant matvec");
            } else {
                kernels
                    .matvec_q4_0(&ctx, &mq, &v, &out_q, rows as u32, cols as u32)
                    .expect("quant matvec");
            }
            kernels
                .matvec(&ctx, &mf, &v, &out_f, rows as u32, cols as u32)
                .expect("f16 matvec");

            let got = read_f16(&out_q);
            let want = read_f16(&out_f);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let scale = w.abs().max(1.0);
                assert!(
                    (g - w).abs() <= 0.05 * scale,
                    "[{rows}x{cols} tq2={is_tq2}] row {i}: quant {g} vs f16 {w}"
                );
            }
        }
    }

    #[test]
    fn quant_kernels_reject_misaligned_cols() {
        let (ctx, kernels) = setup();
        let device = ctx.device();
        let m = MetalBuffer::empty(device, 64).expect("m");
        let v = MetalBuffer::empty(device, 64).expect("v");
        let o = MetalBuffer::empty(device, 64).expect("o");
        assert!(kernels.matvec_q4_0(&ctx, &m, &v, &o, 1, 33).is_err());
        assert!(kernels.matvec_tq2_0(&ctx, &m, &v, &o, 1, 128).is_err());
    }

    /// Unnormalized FWHT butterflies (matches the shader; the 1/√d is applied
    /// separately by the caller).
    fn fwht_butterflies(v: &mut [f32]) {
        let n = v.len();
        let mut len = 1;
        while len < n {
            let mut i = 0;
            while i < n {
                for j in i..i + len {
                    let a = v[j];
                    let b = v[j + len];
                    v[j] = a + b;
                    v[j + len] = a - b;
                }
                i += len << 1;
            }
            len <<= 1;
        }
    }

    /// CPU reference for the GPU `encode_kv_turboquant` kernel: rotate, quantize
    /// to nearest level (lowest-index tie break), and LSB-first bit pack.
    fn cpu_tq_encode(x: &[f32], signs: &[f32], levels: &[f32], bits: u32) -> (Vec<u8>, f32) {
        let d = x.len();
        let norm = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let mut rot: Vec<f32> = x.iter().zip(signs).map(|(&v, &s)| v * inv * s).collect();
        fwht_butterflies(&mut rot);
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let code_bytes = (d * bits as usize).div_ceil(8);
        let mut codes = vec![0u8; code_bytes];
        let mut acc = 0u32;
        let mut nbits = 0u32;
        let mut idx = 0usize;
        for &r in &rot {
            let y = r * inv_sqrt_d;
            let mut best = 0u32;
            let mut bd = f32::INFINITY;
            for (i, &l) in levels.iter().enumerate() {
                let dd = (y - l).abs();
                if dd < bd {
                    bd = dd;
                    best = i as u32;
                }
            }
            acc |= best << nbits;
            nbits += bits;
            while nbits >= 8 {
                codes[idx] = (acc & 0xFF) as u8;
                idx += 1;
                acc >>= 8;
                nbits -= 8;
            }
        }
        if nbits > 0 {
            codes[idx] = (acc & 0xFF) as u8;
        }
        (codes, norm)
    }

    #[test]
    fn encode_kv_turboquant_matches_cpu_reference() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let head_dim = 128usize;
        let n_kv_heads = 4usize;
        let num_tokens = 3usize;
        let bits = 4u32;
        let k = 1usize << bits;

        // Deterministic ±1 signs and a symmetric monotonic level grid (the
        // kernel only needs monotone levels; optimality is irrelevant here).
        let signs: Vec<f32> = (0..head_dim)
            .map(|i| {
                if ((i * 2_654_435_761usize) >> 13) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
        let levels: Vec<f32> = (0..k)
            .map(|i| ((i as f32 + 0.5) / k as f32 - 0.5) * 6.0 * inv_sqrt_d)
            .collect();

        let rows = num_tokens * n_kv_heads;
        let input: Vec<f32> = (0..rows * head_dim)
            .map(|i| ((i as f32 * 0.123).sin() + (i as f32 * 0.017).cos()) * 0.7)
            .collect();

        let code_bytes = (head_dim * bits as usize).div_ceil(8);
        // Allocate at an absolute base so group_offset / position is exercised.
        let position = 2u32;
        let total_slots = (position as usize + num_tokens) * n_kv_heads;

        let input_buf = MetalBuffer::from_slice(device, &f16_vec(&input)).expect("input");
        let levels_buf = MetalBuffer::from_slice(device, &levels).expect("levels");
        let signs_buf = MetalBuffer::from_slice(device, &signs).expect("signs");
        let packed_buf = MetalBuffer::empty(device, total_slots * code_bytes).expect("packed");
        let norms_buf =
            MetalBuffer::empty(device, total_slots * std::mem::size_of::<f16>()).expect("norms");

        kernels
            .encode_kv_turboquant(
                &ctx,
                &input_buf,
                &levels_buf,
                &signs_buf,
                &packed_buf,
                &norms_buf,
                head_dim as u32,
                n_kv_heads as u32,
                num_tokens as u32,
                bits,
                position,
                0,
            )
            .expect("encode");

        let gpu_codes = packed_buf.as_slice::<u8>();
        let gpu_norms = norms_buf.as_slice::<f16>();
        let slot_base = position as usize * n_kv_heads;

        for row in 0..rows {
            let x = &input[row * head_dim..(row + 1) * head_dim];
            let (cpu_codes, cpu_norm) = cpu_tq_encode(x, &signs, &levels, bits);
            let g = slot_base + row;

            // Norm matches within f16 precision.
            let gn = gpu_norms[g].to_f32();
            assert!(
                (gn - cpu_norm).abs() <= 1e-2 * cpu_norm.max(1.0),
                "row {row}: norm gpu={gn} cpu={cpu_norm}"
            );

            // Compare per coordinate. The GPU butterfly order (lane ^ len) sums
            // in a different order than the CPU's sequential FWHT, so values at
            // an exact quantization midpoint can flip to the adjacent level.
            // Require that any difference is at most one level and that such
            // boundary flips are rare (< 2% of coordinates).
            let gc = &gpu_codes[g * code_bytes..(g + 1) * code_bytes];
            let unpack = |codes: &[u8], i: usize| -> u32 {
                let bit_off = i * bits as usize;
                let byte = bit_off / 8;
                let shift = (bit_off % 8) as u32;
                let mut v = u32::from(codes[byte]);
                if shift + bits > 8 {
                    v |= u32::from(codes[byte + 1]) << 8;
                }
                (v >> shift) & ((1 << bits) - 1)
            };
            let mut flips = 0usize;
            for i in 0..head_dim {
                let a = i64::from(unpack(gc, i));
                let b = i64::from(unpack(&cpu_codes, i));
                assert!(
                    (a - b).abs() <= 1,
                    "row {row} coord {i}: gpu code {a} vs cpu {b} differ by >1 level"
                );
                if a != b {
                    flips += 1;
                }
            }
            assert!(
                flips * 50 < head_dim,
                "row {row}: {flips} boundary flips out of {head_dim} coords (too many)"
            );
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn fused_tq_attention_matches_dequant_then_flash() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let head_dim = 128usize;
        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let kv_len = 24usize;
        let current_pos = (kv_len - 1) as u32;
        let window = 0u32;
        let bits = 4u32;
        let k = 1usize << bits;

        let signs: Vec<f32> = (0..head_dim)
            .map(|i| {
                if ((i * 2_246_822_519usize) >> 11) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
        let levels: Vec<f32> = (0..k)
            .map(|i| ((i as f32 + 0.5) / k as f32 - 0.5) * 5.0 * inv_sqrt_d)
            .collect();

        // Random K/V, encoded once; both paths consume the same codes.
        let kv_rows = kv_len * num_kv_heads;
        let kdata: Vec<f32> = (0..kv_rows * head_dim)
            .map(|i| (i as f32 * 0.07).sin() * 0.5)
            .collect();
        let vdata: Vec<f32> = (0..kv_rows * head_dim)
            .map(|i| (i as f32 * 0.041).cos() * 0.5)
            .collect();
        let qdata: Vec<f32> = (0..num_q_heads * head_dim)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();

        let code_bytes = (head_dim * bits as usize).div_ceil(8);
        let levels_buf = MetalBuffer::from_slice(device, &levels).expect("levels");
        let signs_buf = MetalBuffer::from_slice(device, &signs).expect("signs");

        let kbuf = MetalBuffer::from_slice(device, &f16_vec(&kdata)).expect("kbuf");
        let vbuf = MetalBuffer::from_slice(device, &f16_vec(&vdata)).expect("vbuf");
        let kcodes = MetalBuffer::empty(device, kv_rows * code_bytes).expect("kcodes");
        let knorms = MetalBuffer::empty(device, kv_rows * std::mem::size_of::<f16>()).expect("kn");
        let vcodes = MetalBuffer::empty(device, kv_rows * code_bytes).expect("vcodes");
        let vnorms = MetalBuffer::empty(device, kv_rows * std::mem::size_of::<f16>()).expect("vn");

        for (src, codes, norms) in [(&kbuf, &kcodes, &knorms), (&vbuf, &vcodes, &vnorms)] {
            kernels
                .encode_kv_turboquant(
                    &ctx,
                    src,
                    &levels_buf,
                    &signs_buf,
                    codes,
                    norms,
                    head_dim as u32,
                    num_kv_heads as u32,
                    kv_len as u32,
                    bits,
                    0,
                    0,
                )
                .expect("encode");
        }

        // Reference: dequantize to FP16, then the standard flash attention.
        let k_deq = MetalBuffer::empty(device, kv_rows * head_dim * std::mem::size_of::<f16>())
            .expect("kd");
        let v_deq = MetalBuffer::empty(device, kv_rows * head_dim * std::mem::size_of::<f16>())
            .expect("vd");
        for (codes, norms, out) in [(&kcodes, &knorms, &k_deq), (&vcodes, &vnorms, &v_deq)] {
            kernels
                .dequantize_kv_turboquant(
                    &ctx,
                    codes,
                    norms,
                    &levels_buf,
                    &signs_buf,
                    out,
                    head_dim as u32,
                    num_kv_heads as u32,
                    kv_len as u32,
                    bits,
                    0,
                    0,
                )
                .expect("dequant");
        }
        let q_f16 = MetalBuffer::from_slice(device, &f16_vec(&qdata)).expect("qf16");
        let out_ref =
            MetalBuffer::empty(device, num_q_heads * head_dim * std::mem::size_of::<f16>())
                .expect("oref");
        kernels
            .flash_attention(
                &ctx,
                &q_f16,
                &k_deq,
                &v_deq,
                &out_ref,
                head_dim as u32,
                kv_len as u32,
                current_pos,
                num_q_heads as u32,
                num_kv_heads as u32,
                window,
            )
            .expect("flash ref");
        let ref_out = read_f16(&out_ref);

        // Fused: rotate query, fused attention over codes, inverse-rotate.
        let q_f32 = MetalBuffer::from_slice(device, &qdata).expect("qf32");
        let rq = MetalBuffer::empty(device, num_q_heads * head_dim * 4).expect("rq");
        let vacc = MetalBuffer::empty(device, num_q_heads * head_dim * 4).expect("vacc");
        let out_fused = MetalBuffer::empty(device, num_q_heads * head_dim * 4).expect("of");
        {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
            kernels
                .hadamard_rotate_into(
                    &mut batch,
                    &q_f32,
                    &signs_buf,
                    &rq,
                    head_dim as u32,
                    num_q_heads as u32,
                    true,
                    false,
                )
                .expect("rotate q");
            kernels
                .flash_attention_tq_into(
                    &mut batch,
                    &rq,
                    &kcodes,
                    &knorms,
                    &vcodes,
                    &vnorms,
                    &levels_buf,
                    &vacc,
                    head_dim as u32,
                    kv_len as u32,
                    current_pos,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    window,
                    bits,
                    0,
                )
                .expect("flash tq");
            kernels
                .hadamard_rotate_into(
                    &mut batch,
                    &vacc,
                    &signs_buf,
                    &out_fused,
                    head_dim as u32,
                    num_q_heads as u32,
                    false,
                    true,
                )
                .expect("inverse rotate");
            batch.commit_and_wait().expect("commit");
        }
        let fused_out: Vec<f32> = out_fused.as_slice::<f32>().to_vec();

        // The two paths differ only in f16 vs f32 intermediate precision.
        let mut max_abs = 0.0f32;
        let mut ref_norm = 0.0f32;
        for (a, b) in fused_out.iter().zip(&ref_out) {
            max_abs = max_abs.max((a - b).abs());
            ref_norm = ref_norm.max(b.abs());
        }
        assert!(
            max_abs <= 0.02 * ref_norm.max(1e-3),
            "fused vs reference attention diverged: max_abs={max_abs}, ref_norm={ref_norm}"
        );
    }

    /// The fused multi-row prefill kernel (`flash_attention_tq_prefill`) must
    /// reproduce, for every query row `r`, exactly what the proven single-token
    /// decode kernel (`flash_attention_tq`) computes at absolute position `r`
    /// over the same packed cache. Both produce rotated-space accumulation from
    /// identical codes, so they should agree to f32 reduction-order noise. This
    /// is the correctness gate for the fused-TQ prefill path that replaces the
    /// FP16-dequant + `flash_attention_prefill` pair.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn fused_tq_prefill_matches_per_row_decode() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let head_dim = 128usize;
        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let n_rows = 12usize; // query rows; row r sits at absolute position r
        let kv_len = n_rows; // start_pos = 0, so the cache holds exactly n_rows
        let window = 0u32;
        let bits = 4u32;
        let k = 1usize << bits;

        let signs: Vec<f32> = (0..head_dim)
            .map(|i| {
                if ((i * 2_246_822_519usize) >> 11) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
        let levels: Vec<f32> = (0..k)
            .map(|i| ((i as f32 + 0.5) / k as f32 - 0.5) * 5.0 * inv_sqrt_d)
            .collect();

        let kv_rows = kv_len * num_kv_heads;
        let kdata: Vec<f32> = (0..kv_rows * head_dim)
            .map(|i| (i as f32 * 0.07).sin() * 0.5)
            .collect();
        let vdata: Vec<f32> = (0..kv_rows * head_dim)
            .map(|i| (i as f32 * 0.041).cos() * 0.5)
            .collect();
        // One query per (row, head), laid out [n_rows, num_q_heads, head_dim].
        let qdata: Vec<f32> = (0..n_rows * num_q_heads * head_dim)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();

        let code_bytes = (head_dim * bits as usize).div_ceil(8);
        let levels_buf = MetalBuffer::from_slice(device, &levels).expect("levels");
        let signs_buf = MetalBuffer::from_slice(device, &signs).expect("signs");

        let kbuf = MetalBuffer::from_slice(device, &f16_vec(&kdata)).expect("kbuf");
        let vbuf = MetalBuffer::from_slice(device, &f16_vec(&vdata)).expect("vbuf");
        let kcodes = MetalBuffer::empty(device, kv_rows * code_bytes).expect("kcodes");
        let knorms = MetalBuffer::empty(device, kv_rows * std::mem::size_of::<f16>()).expect("kn");
        let vcodes = MetalBuffer::empty(device, kv_rows * code_bytes).expect("vcodes");
        let vnorms = MetalBuffer::empty(device, kv_rows * std::mem::size_of::<f16>()).expect("vn");
        for (src, codes, norms) in [(&kbuf, &kcodes, &knorms), (&vbuf, &vcodes, &vnorms)] {
            kernels
                .encode_kv_turboquant(
                    &ctx,
                    src,
                    &levels_buf,
                    &signs_buf,
                    codes,
                    norms,
                    head_dim as u32,
                    num_kv_heads as u32,
                    kv_len as u32,
                    bits,
                    0,
                    0,
                )
                .expect("encode");
        }

        // Rotate all query rows up front: rq[n_rows * num_q_heads * head_dim].
        let q_f32 = MetalBuffer::from_slice(device, &qdata).expect("qf32");
        let vecs = (n_rows * num_q_heads) as u32;
        let rq_all = MetalBuffer::empty(device, qdata.len() * 4).expect("rq_all");
        let out_prefill = MetalBuffer::empty(device, qdata.len() * 4).expect("out_prefill");
        {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
            kernels
                .hadamard_rotate_into(
                    &mut batch, &q_f32, &signs_buf, &rq_all, head_dim as u32, vecs, true, false,
                )
                .expect("rotate all");
            kernels
                .flash_attention_tq_prefill_into(
                    &mut batch,
                    &rq_all,
                    &kcodes,
                    &knorms,
                    &vcodes,
                    &vnorms,
                    &levels_buf,
                    &out_prefill,
                    head_dim as u32,
                    kv_len as u32,
                    0, // start_pos
                    n_rows as u32,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    window,
                    bits,
                    0, // ring_capacity (full attention)
                )
                .expect("flash tq prefill");
            batch.commit_and_wait().expect("commit");
        }
        let prefill_out: Vec<f32> = out_prefill.as_slice::<f32>().to_vec();

        // Reference: for each row, the proven single-token decode kernel at that
        // absolute position over the same codes (rotated-space output too).
        let row_elems = num_q_heads * head_dim;
        for row in 0..n_rows {
            let rq_row = MetalBuffer::from_slice(
                device,
                &rq_all.as_slice::<f32>()[row * row_elems..(row + 1) * row_elems],
            )
            .expect("rq_row");
            let vacc = MetalBuffer::empty(device, row_elems * 4).expect("vacc");
            {
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
                kernels
                    .flash_attention_tq_into(
                        &mut batch,
                        &rq_row,
                        &kcodes,
                        &knorms,
                        &vcodes,
                        &vnorms,
                        &levels_buf,
                        &vacc,
                        head_dim as u32,
                        kv_len as u32,
                        row as u32, // current_pos
                        num_q_heads as u32,
                        num_kv_heads as u32,
                        window,
                        bits,
                        0,
                    )
                    .expect("flash tq decode");
                batch.commit_and_wait().expect("commit");
            }
            let ref_row: Vec<f32> = vacc.as_slice::<f32>().to_vec();
            let got_row = &prefill_out[row * row_elems..(row + 1) * row_elems];
            let mut max_abs = 0.0f32;
            let mut ref_norm = 0.0f32;
            for (a, b) in got_row.iter().zip(&ref_row) {
                max_abs = max_abs.max((a - b).abs());
                ref_norm = ref_norm.max(b.abs());
            }
            assert!(
                max_abs <= 1e-4 * ref_norm.max(1e-3),
                "row {row}: prefill vs per-row decode diverged: max_abs={max_abs}, ref_norm={ref_norm}"
            );
        }
    }

    /// The query-tiled fused prefill kernel (`flash_attention_tq_prefill_tiled`)
    /// must reproduce the validated non-tiled prefill
    /// (`flash_attention_tq_prefill`) bit-for-bit-close for both full attention
    /// and a sliding window, exercising the ring-slot remapping and the
    /// per-block tile alignment / causal cutoffs. The tile dequant uses `half`
    /// like the non-tiled path, so the tolerance matches the per-row gate.
    #[test]
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn fused_tq_prefill_tiled_matches_nontiled() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        // head_dim 256 → bk = 4096/256 = 16 keys per tile; n_rows spans several
        // query blocks (BR = 8) so the q_block/head grid mapping is exercised.
        let head_dim = 256usize;
        let num_q_heads = 8usize;
        let num_kv_heads = 1usize; // Gemma E2B is MQA-favorable (1 KV head).
        let n_rows = 20usize;
        let kv_len = n_rows; // start_pos = 0
        let bits = 2u32; // TurboQuant ships 2-bit KV.
        let k = 1usize << bits;

        let signs: Vec<f32> = (0..head_dim)
            .map(|i| {
                if ((i * 2_246_822_519usize) >> 11) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
        let levels: Vec<f32> = (0..k)
            .map(|i| ((i as f32 + 0.5) / k as f32 - 0.5) * 5.0 * inv_sqrt_d)
            .collect();

        let kv_rows = kv_len * num_kv_heads;
        let kdata: Vec<f32> = (0..kv_rows * head_dim)
            .map(|i| (i as f32 * 0.07).sin() * 0.5)
            .collect();
        let vdata: Vec<f32> = (0..kv_rows * head_dim)
            .map(|i| (i as f32 * 0.041).cos() * 0.5)
            .collect();
        let qdata: Vec<f32> = (0..n_rows * num_q_heads * head_dim)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();

        let code_bytes = (head_dim * bits as usize).div_ceil(8);
        let levels_buf = MetalBuffer::from_slice(device, &levels).expect("levels");
        let signs_buf = MetalBuffer::from_slice(device, &signs).expect("signs");

        let kbuf = MetalBuffer::from_slice(device, &f16_vec(&kdata)).expect("kbuf");
        let vbuf = MetalBuffer::from_slice(device, &f16_vec(&vdata)).expect("vbuf");
        let kcodes = MetalBuffer::empty(device, kv_rows * code_bytes).expect("kcodes");
        let knorms = MetalBuffer::empty(device, kv_rows * std::mem::size_of::<f16>()).expect("kn");
        let vcodes = MetalBuffer::empty(device, kv_rows * code_bytes).expect("vcodes");
        let vnorms = MetalBuffer::empty(device, kv_rows * std::mem::size_of::<f16>()).expect("vn");
        for (src, codes, norms) in [(&kbuf, &kcodes, &knorms), (&vbuf, &vcodes, &vnorms)] {
            kernels
                .encode_kv_turboquant(
                    &ctx,
                    src,
                    &levels_buf,
                    &signs_buf,
                    codes,
                    norms,
                    head_dim as u32,
                    num_kv_heads as u32,
                    kv_len as u32,
                    bits,
                    0,
                    0,
                )
                .expect("encode");
        }

        let q_f32 = MetalBuffer::from_slice(device, &qdata).expect("qf32");
        let vecs = (n_rows * num_q_heads) as u32;
        let rq_all = MetalBuffer::empty(device, qdata.len() * 4).expect("rq_all");
        {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
            kernels
                .hadamard_rotate_into(
                    &mut batch, &q_f32, &signs_buf, &rq_all, head_dim as u32, vecs, true, false,
                )
                .expect("rotate all");
            batch.commit_and_wait().expect("commit");
        }

        // Compare tiled vs non-tiled for full attention and a sliding window.
        for window in [0u32, 6u32] {
            let out_tiled = MetalBuffer::empty(device, qdata.len() * 4).expect("out_tiled");
            let out_ref = MetalBuffer::empty(device, qdata.len() * 4).expect("out_ref");
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
            kernels
                .flash_attention_tq_prefill_into(
                    &mut batch,
                    &rq_all,
                    &kcodes,
                    &knorms,
                    &vcodes,
                    &vnorms,
                    &levels_buf,
                    &out_ref,
                    head_dim as u32,
                    kv_len as u32,
                    0,
                    n_rows as u32,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    window,
                    bits,
                    0,
                )
                .expect("nontiled prefill");
            kernels
                .flash_attention_tq_prefill_tiled_into(
                    &mut batch,
                    &rq_all,
                    &kcodes,
                    &knorms,
                    &vcodes,
                    &vnorms,
                    &levels_buf,
                    &out_tiled,
                    head_dim as u32,
                    kv_len as u32,
                    0,
                    n_rows as u32,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    window,
                    bits,
                    0,
                )
                .expect("tiled prefill");
            batch.commit_and_wait().expect("commit");

            let got: Vec<f32> = out_tiled.as_slice::<f32>().to_vec();
            let reference: Vec<f32> = out_ref.as_slice::<f32>().to_vec();
            let mut max_abs = 0.0f32;
            let mut ref_norm = 0.0f32;
            for (a, b) in got.iter().zip(&reference) {
                max_abs = max_abs.max((a - b).abs());
                ref_norm = ref_norm.max(b.abs());
            }
            // The tiled path stores each dequantized `norm · level` K/V tile as
            // `half` (matching the legacy FP16 attention), whereas the non-tiled
            // path keeps the product in f32; the gap is pure f16 rounding.
            assert!(
                max_abs <= 5e-3 * ref_norm.max(1e-3),
                "window {window}: tiled vs non-tiled diverged: max_abs={max_abs}, ref_norm={ref_norm}"
            );
        }
    }

    /// The batched `TurboQuant` decode kernels (`encode_kv_turboquant_batched` +
    /// `flash_attention_tq_batched`) over a unified by-lane pool must reproduce
    /// the proven single-sequence turbo path for every lane decoding at its own
    /// independent context length. This is the correctness gate for porting
    /// continuous batching / serve onto the `TurboQuant` KV backing.
    #[test]
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn batched_tq_decode_matches_independent_single_seq() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let head_dim = 128usize;
        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let bits = 4u32;
        let k = 1usize << bits;
        let n_lanes = 2usize;
        let lane_capacity = 16usize;
        let window = 0u32;
        let code_bytes = (head_dim * bits as usize).div_ceil(8);
        // Each lane decodes at its own context length.
        let lane_pos = [10usize, 6usize];
        let max_pos = *lane_pos.iter().max().unwrap();

        let signs: Vec<f32> = (0..head_dim)
            .map(|i| {
                if ((i * 2_246_822_519usize) >> 11) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
        let levels: Vec<f32> = (0..k)
            .map(|i| ((i as f32 + 0.5) / k as f32 - 0.5) * 5.0 * inv_sqrt_d)
            .collect();
        let levels_buf = MetalBuffer::from_slice(device, &levels).expect("levels");
        let signs_buf = MetalBuffer::from_slice(device, &signs).expect("signs");

        // Deterministic per-(lane, token, head, dim) K/V values.
        let row = num_kv_heads * head_dim;
        let kval =
            |l: usize, t: usize, e: usize| (((l * 131 + t * 37 + e) as f32) * 0.011).sin() * 0.5;
        let vval =
            |l: usize, t: usize, e: usize| (((l * 91 + t * 53 + e) as f32) * 0.009).cos() * 0.5;

        // ---- Build the unified pool by replaying decode steps through the
        //      batched encode kernel (one new token per lane per step). ----
        let pool_elems = n_lanes * lane_capacity * num_kv_heads;
        let pool_kcodes = MetalBuffer::empty(device, pool_elems * code_bytes).expect("pk");
        let pool_vcodes = MetalBuffer::empty(device, pool_elems * code_bytes).expect("pv");
        let pool_knorms =
            MetalBuffer::empty(device, pool_elems * std::mem::size_of::<f16>()).expect("pkn");
        let pool_vnorms =
            MetalBuffer::empty(device, pool_elems * std::mem::size_of::<f16>()).expect("pvn");

        for s in 0..=max_pos {
            let mut k_in = vec![0.0f32; n_lanes * row];
            let mut v_in = vec![0.0f32; n_lanes * row];
            for l in 0..n_lanes {
                if s <= lane_pos[l] {
                    for e in 0..row {
                        k_in[l * row + e] = kval(l, s, e);
                        v_in[l * row + e] = vval(l, s, e);
                    }
                }
            }
            let k_buf = MetalBuffer::from_slice(device, &f16_vec(&k_in)).expect("kin");
            let v_buf = MetalBuffer::from_slice(device, &f16_vec(&v_in)).expect("vin");
            let pos = [s as u32; 2];
            let pos_buf = MetalBuffer::from_slice(device, &pos).expect("pos");
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
            for (src, codes, norms) in [
                (&k_buf, &pool_kcodes, &pool_knorms),
                (&v_buf, &pool_vcodes, &pool_vnorms),
            ] {
                kernels
                    .encode_kv_turboquant_batched_into(
                        &mut batch,
                        src,
                        &levels_buf,
                        &signs_buf,
                        codes,
                        norms,
                        &pos_buf,
                        head_dim as u32,
                        num_kv_heads as u32,
                        lane_capacity as u32,
                        bits,
                        n_lanes as u32,
                        0,
                    )
                    .expect("encode batched");
            }
            batch.commit_and_wait().expect("commit encode");
        }

        // ---- Rotated queries: lane-major [n_lanes, num_q_heads, head_dim]. ----
        let qval = |l: usize, h: usize, d: usize| (((l * 17 + h * 5 + d) as f32) * 0.013).sin();
        let mut q_all = vec![0.0f32; n_lanes * num_q_heads * head_dim];
        for l in 0..n_lanes {
            for h in 0..num_q_heads {
                for d in 0..head_dim {
                    q_all[(l * num_q_heads + h) * head_dim + d] = qval(l, h, d);
                }
            }
        }
        let q_all_buf = MetalBuffer::from_slice(device, &q_all).expect("qall");
        let rq_all = MetalBuffer::empty(device, q_all.len() * 4).expect("rqall");
        let out_all = MetalBuffer::empty(device, q_all.len() * 4).expect("outall");
        {
            let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
            kernels
                .hadamard_rotate_into(
                    &mut batch,
                    &q_all_buf,
                    &signs_buf,
                    &rq_all,
                    head_dim as u32,
                    (n_lanes * num_q_heads) as u32,
                    true,
                    false,
                )
                .expect("rotate q all");
            let pos = [lane_pos[0] as u32, lane_pos[1] as u32];
            let pos_buf = MetalBuffer::from_slice(device, &pos).expect("pos");
            kernels
                .flash_attention_tq_batched_into(
                    &mut batch,
                    &rq_all,
                    &pool_kcodes,
                    &pool_knorms,
                    &pool_vcodes,
                    &pool_vnorms,
                    &levels_buf,
                    &out_all,
                    &pos_buf,
                    head_dim as u32,
                    lane_capacity as u32,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    window,
                    bits,
                    n_lanes as u32,
                    0,
                )
                .expect("flash tq batched");
            batch.commit_and_wait().expect("commit attn");
        }
        let batched_out: Vec<f32> = out_all.as_slice::<f32>().to_vec();

        // ---- Per-lane reference: encode that lane's tokens with the single-seq
        //      kernel and run single-seq fused attention; compare both the codes
        //      and the rotated-space attention output. ----
        for l in 0..n_lanes {
            let seq = lane_pos[l] + 1;
            let mut k_ref = vec![0.0f32; seq * row];
            let mut v_ref = vec![0.0f32; seq * row];
            for t in 0..seq {
                for e in 0..row {
                    k_ref[t * row + e] = kval(l, t, e);
                    v_ref[t * row + e] = vval(l, t, e);
                }
            }
            let k_ref_buf = MetalBuffer::from_slice(device, &f16_vec(&k_ref)).expect("kref");
            let v_ref_buf = MetalBuffer::from_slice(device, &f16_vec(&v_ref)).expect("vref");
            let kc = MetalBuffer::empty(device, seq * num_kv_heads * code_bytes).expect("kc");
            let vc = MetalBuffer::empty(device, seq * num_kv_heads * code_bytes).expect("vc");
            let kn = MetalBuffer::empty(device, seq * num_kv_heads * std::mem::size_of::<f16>())
                .expect("kn");
            let vn = MetalBuffer::empty(device, seq * num_kv_heads * std::mem::size_of::<f16>())
                .expect("vn");
            for (src, codes, norms) in [(&k_ref_buf, &kc, &kn), (&v_ref_buf, &vc, &vn)] {
                kernels
                    .encode_kv_turboquant(
                        &ctx,
                        src,
                        &levels_buf,
                        &signs_buf,
                        codes,
                        norms,
                        head_dim as u32,
                        num_kv_heads as u32,
                        seq as u32,
                        bits,
                        0,
                        0,
                    )
                    .expect("encode ref");
            }

            // Codes/norms for this lane's used region must match the pool exactly.
            let pool_off = l * lane_capacity * num_kv_heads;
            let used = seq * num_kv_heads;
            let pk = &pool_kcodes.as_slice::<u8>()
                [pool_off * code_bytes..(pool_off + used) * code_bytes];
            assert_eq!(
                pk,
                kc.as_slice::<u8>(),
                "lane {l} key codes differ from single-seq"
            );
            let pkn = &read_f16(&pool_knorms)[pool_off..pool_off + used];
            assert_eq!(pkn, read_f16(&kn).as_slice(), "lane {l} key norms differ");

            // Single-seq rotated query for this lane (slice of q_all).
            let q_lane = &q_all[l * num_q_heads * head_dim..(l + 1) * num_q_heads * head_dim];
            let q_lane_buf = MetalBuffer::from_slice(device, q_lane).expect("qlane");
            let rq_lane = MetalBuffer::empty(device, q_lane.len() * 4).expect("rqlane");
            let vacc = MetalBuffer::empty(device, q_lane.len() * 4).expect("vacc");
            {
                let mut batch = crate::batch::CommandBatch::new(&ctx).expect("batch");
                kernels
                    .hadamard_rotate_into(
                        &mut batch,
                        &q_lane_buf,
                        &signs_buf,
                        &rq_lane,
                        head_dim as u32,
                        num_q_heads as u32,
                        true,
                        false,
                    )
                    .expect("rotate q lane");
                kernels
                    .flash_attention_tq_into(
                        &mut batch,
                        &rq_lane,
                        &kc,
                        &kn,
                        &vc,
                        &vn,
                        &levels_buf,
                        &vacc,
                        head_dim as u32,
                        seq as u32,
                        lane_pos[l] as u32,
                        num_q_heads as u32,
                        num_kv_heads as u32,
                        window,
                        bits,
                        0,
                    )
                    .expect("flash tq single");
                batch.commit_and_wait().expect("commit single");
            }
            let ref_out: Vec<f32> = vacc.as_slice::<f32>().to_vec();
            let lane_out =
                &batched_out[l * num_q_heads * head_dim..(l + 1) * num_q_heads * head_dim];

            let mut max_abs = 0.0f32;
            let mut ref_norm = 0.0f32;
            for (a, b) in lane_out.iter().zip(&ref_out) {
                max_abs = max_abs.max((a - b).abs());
                ref_norm = ref_norm.max(b.abs());
            }
            assert!(
                max_abs <= 1e-4 * ref_norm.max(1e-3),
                "lane {l} batched vs single attention diverged: max_abs={max_abs}, ref_norm={ref_norm}"
            );
        }
    }

    #[test]
    fn encode_then_decode_roundtrips() {
        let (ctx, kernels) = setup();
        let device = ctx.device();

        let head_dim = 128usize;
        let n_kv_heads = 2usize;
        let num_tokens = 4usize;
        let bits = 4u32;
        let k = 1usize << bits;

        let signs: Vec<f32> = (0..head_dim)
            .map(|i| {
                if ((i * 40_503usize) >> 7) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
        // A Gaussian-ish level grid so 4-bit reconstruction is reasonably tight.
        let levels: Vec<f32> = (0..k)
            .map(|i| {
                let t = (i as f32 + 0.5) / k as f32 - 0.5;
                t * 5.0 * inv_sqrt_d
            })
            .collect();

        let rows = num_tokens * n_kv_heads;
        let input: Vec<f32> = (0..rows * head_dim)
            .map(|i| (i as f32 * 0.0911).sin())
            .collect();

        let code_bytes = (head_dim * bits as usize).div_ceil(8);
        let input_buf = MetalBuffer::from_slice(device, &f16_vec(&input)).expect("input");
        let levels_buf = MetalBuffer::from_slice(device, &levels).expect("levels");
        let signs_buf = MetalBuffer::from_slice(device, &signs).expect("signs");
        let packed_buf = MetalBuffer::empty(device, rows * code_bytes).expect("packed");
        let norms_buf =
            MetalBuffer::empty(device, rows * std::mem::size_of::<f16>()).expect("norms");
        let out_buf =
            MetalBuffer::empty(device, rows * head_dim * std::mem::size_of::<f16>()).expect("out");

        kernels
            .encode_kv_turboquant(
                &ctx,
                &input_buf,
                &levels_buf,
                &signs_buf,
                &packed_buf,
                &norms_buf,
                head_dim as u32,
                n_kv_heads as u32,
                num_tokens as u32,
                bits,
                0,
                0,
            )
            .expect("encode");

        kernels
            .dequantize_kv_turboquant(
                &ctx,
                &packed_buf,
                &norms_buf,
                &levels_buf,
                &signs_buf,
                &out_buf,
                head_dim as u32,
                n_kv_heads as u32,
                num_tokens as u32,
                bits,
                0,
                0,
            )
            .expect("decode");

        let recon = read_f16(&out_buf);
        // Per-row relative reconstruction error should be small for 4-bit.
        for row in 0..rows {
            let x = &input[row * head_dim..(row + 1) * head_dim];
            let xr = &recon[row * head_dim..(row + 1) * head_dim];
            let num: f32 = x.iter().zip(xr).map(|(a, b)| (a - b).powi(2)).sum();
            let den: f32 = x.iter().map(|a| a * a).sum::<f32>().max(1e-6);
            let rel = (num / den).sqrt();
            assert!(rel < 0.25, "row {row}: relative recon error {rel} too high");
        }
    }
}
