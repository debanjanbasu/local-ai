use super::{
    BonsaiPackage, BonsaiTensorType, FFN, FULL_INTERVAL, LAYERS, TRAINING_CONTEXT, VOCAB, WIDTH,
    invalid,
};

/// The exact tensor set, shapes, dtypes and hyperparameters of the pinned
/// Bonsai 2 27B PTQ1 checkpoint; anything else is refused rather than guessed.
#[allow(clippy::too_many_lines)]
pub fn validate_profile(package: &BonsaiPackage) -> crate::Result<()> {
    use BonsaiTensorType::{Bf16, F32, Ptq1};

    for (key, expected) in [
        ("block_count", LAYERS),
        ("context_length", TRAINING_CONTEXT),
        ("embedding_length", WIDTH),
        ("feed_forward_length", FFN),
        ("attention.head_count", 24),
        ("attention.head_count_kv", 4),
        ("attention.key_length", 256),
        ("attention.value_length", 256),
        ("ssm.conv_kernel", 4),
        ("ssm.state_size", 128),
        ("ssm.group_count", 16),
        ("ssm.time_step_rank", 48),
        ("ssm.inner_size", 6144),
        ("full_attention_interval", FULL_INTERVAL),
        ("rope.dimension_count", 64),
    ] {
        if package.metadata_u32(&format!("qwen35.{key}"))? as usize != expected {
            return invalid(format!("unsupported Bonsai qwen35.{key}"));
        }
    }
    if package.metadata_i32_array("qwen35.rope.dimension_sections")? != [11, 11, 10, 0]
        || package.metadata_f32("qwen35.rope.freq_base")?.to_bits() != 10_000_000.0_f32.to_bits()
        || package
            .metadata_f32("qwen35.attention.layer_norm_rms_epsilon")?
            .to_bits()
            != 1e-6_f32.to_bits()
        || !package.hadamard().gdn_v_grouped()
        || package.metadata_count() != 49
    {
        return invalid("unsupported Bonsai normalization, RoPE, or metadata profile");
    }
    let mut checked = 0;
    let mut require = |name: &str, shape: &[u64], dtype| -> crate::Result<()> {
        let tensor = package.tensor(name)?;
        if tensor.dimensions() != shape || tensor.tensor_type() != dtype {
            return invalid(format!("unexpected shape/type for Bonsai {name}"));
        }
        checked += 1;
        Ok(())
    };
    require("token_embd.weight", &[WIDTH as u64, VOCAB as u64], Ptq1)?;
    require("output.weight", &[WIDTH as u64, VOCAB as u64], Ptq1)?;
    require("output_norm.weight", &[WIDTH as u64], F32)?;
    for index in 0..LAYERS {
        let mut require =
            |suffix, shape: &[u64], dtype| require(&format!("blk.{index}.{suffix}"), shape, dtype);
        require("attn_norm.weight", &[WIDTH as u64], F32)?;
        require("post_attention_norm.weight", &[WIDTH as u64], F32)?;
        require("ffn_gate.weight", &[WIDTH as u64, FFN as u64], Ptq1)?;
        require("ffn_up.weight", &[WIDTH as u64, FFN as u64], Ptq1)?;
        require("ffn_down.weight", &[FFN as u64, WIDTH as u64], Ptq1)?;
        if (index + 1).is_multiple_of(FULL_INTERVAL) {
            require("attn_q.weight", &[WIDTH as u64, 12288], Ptq1)?;
            require("attn_k.weight", &[WIDTH as u64, 1024], Ptq1)?;
            require("attn_v.weight", &[WIDTH as u64, 1024], Ptq1)?;
            require("attn_q_norm.weight", &[256], F32)?;
            require("attn_k_norm.weight", &[256], F32)?;
            require("attn_output.weight", &[6144, WIDTH as u64], Ptq1)?;
        } else {
            require("attn_qkv.weight", &[WIDTH as u64, 10240], Ptq1)?;
            require("attn_gate.weight", &[WIDTH as u64, 6144], Ptq1)?;
            require("ssm_alpha.weight", &[WIDTH as u64, 48], Bf16)?;
            require("ssm_beta.weight", &[WIDTH as u64, 48], Bf16)?;
            require("ssm_conv1d.weight", &[4, 10240], F32)?;
            require("ssm_a", &[48], F32)?;
            require("ssm_dt.bias", &[48], F32)?;
            require("ssm_norm.weight", &[128], F32)?;
            require("ssm_out.weight", &[6144, WIDTH as u64], Ptq1)?;
        }
    }
    if checked != package.tensors().len() {
        return invalid("unhandled tensor in Bonsai checkpoint");
    }
    Ok(())
}
