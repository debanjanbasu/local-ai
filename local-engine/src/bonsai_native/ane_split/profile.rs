//! GPU time of each operation of a prefill block, for attributing where
//! prefill time goes (measurement only; nothing is asserted).

use std::time::Instant;

use local_metal::batch::submit_buffer_copies;

use super::super::layers::decay_projection;
use super::super::{BonsaiPackage, KvOptions, NgramSettings};
use super::*;
use crate::bonsai::DEFAULT_BONSAI_GGUF;

/// Timed repeats of each operation after one warm-up; the median is kept.
const REPEATS: usize = 5;
/// Rows of a half block (the wavefront) and of a GPU block.
const ROWS: u32 = 512;
/// Output projection rows on the ANE in the wavefront (the largest step).
const SPLIT_ROWS: u32 = 3072;

/// Median GPU time (ms) of `encode` in a command buffer of its own.
fn gpu_ms(
    model: &BonsaiModel,
    mut encode: impl FnMut(&mut CommandBatch) -> crate::Result<()>,
) -> f64 {
    let mut times = (0..=REPEATS)
        .map(|_| {
            let mut batch = CommandBatch::new(&model.context).expect("batch");
            encode(&mut batch).expect("encode");
            batch
                .commit_async()
                .wait_with_gpu_time()
                .expect("wait")
                .as_secs_f64()
                * 1e3
        })
        .skip(1)
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    times[REPEATS / 2]
}

/// Per-operation GPU time of one 512-row block at each position of
/// `PROFILE_POSITIONS` (default 4,096, 32,768 and 131,072), on the GPU path
/// and as one half of a Neural Engine wavefront (and that half's mean at the
/// attention Q/K/V step), plus the whole block's wall time on the GPU and as
/// a wavefront at the largest output split without and with the Q/K/V. The
/// K/V caches hold a real 4,096-token prefix repeated up to the longest
/// position.
#[test]
#[ignore = "measurement; requires the pinned model, a Metal device and the Neural Engine"]
#[expect(clippy::too_many_lines)]
fn prefill_op_timings() {
    let positions = std::env::var("PROFILE_POSITIONS").map_or_else(
        |_| vec![4096, 32_768, 131_072],
        |list| {
            list.split(',')
                .filter_map(|value| value.parse::<usize>().ok())
                .collect()
        },
    );
    let seed = 4096;
    let longest = positions.iter().copied().max().expect("positions");
    let context = longest + WAVE_ROWS + 64;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let text = std::fs::read_to_string(format!("{root}/docs/BONSAI.md")).expect("corpus");
    let package = BonsaiPackage::open(DEFAULT_BONSAI_GGUF).expect("open");
    let tokenizer = crate::bonsai_tokenizer::BonsaiTokenizer::from_package(&package).expect("tok");
    let corpus = tokenizer.encode(&text).expect("encode");
    let mut model = BonsaiModel::load(
        package,
        context,
        512,
        None,
        None,
        NgramSettings::default(),
        KvOptions {
            initial_tokens: context,
            ..KvOptions::default()
        },
    )
    .expect("load");
    model
        .prefill(&corpus[..seed], &mut |_| {})
        .expect("prefill");
    assert!(model.ane_active(), "{}", model.neural_engine_policy());
    // Repeat the prefix's K/V up to the context.
    let mut copies = Vec::new();
    for layer in &model.layers {
        let AttentionLayer::Full(full) = &layer.attention else {
            continue;
        };
        for (cache, token_bytes) in [
            (&full.key_cache, model.kv_layout.key.token_bytes()),
            (&full.value_cache, model.kv_layout.value.token_bytes()),
        ] {
            for start in (seed..context).step_by(seed) {
                copies.push(BufferCopyRequest {
                    source: cache,
                    source_offset: 0,
                    destination: cache,
                    destination_offset: start * token_bytes,
                    size: seed.min(context - start) * token_bytes,
                });
            }
        }
    }
    submit_buffer_copies(&model.context, &copies)
        .expect("copies")
        .wait()
        .expect("copies");
    drop(copies);
    let split = model.take_ane().expect("the ANE path");
    let rows = ROWS;
    let s = &model.scratch;
    let eps = model.epsilon;
    let (recurrent, recurrent_layer) = model
        .layers
        .iter()
        .find_map(|layer| match &layer.attention {
            AttentionLayer::Recurrent(recurrent) => Some((recurrent, layer)),
            AttentionLayer::Full(_) => None,
        })
        .expect("a recurrent layer");
    let full = model
        .layers
        .iter()
        .find_map(|layer| match &layer.attention {
            AttentionLayer::Full(full) => Some(full),
            AttentionLayer::Recurrent(_) => None,
        })
        .expect("a full-attention layer");
    let m = &model;
    let mut fixed = Vec::<(&str, f64)>::new();
    let mut time = |name, encode: &mut dyn FnMut(&mut CommandBatch) -> crate::Result<()>| {
        let ms = gpu_ms(m, encode);
        fixed.push((name, ms));
    };
    time("norm", &mut |b| {
        m.ops.residual_normalize_transform(
            b,
            &m.input_rotation,
            &s.hidden_alt,
            &s.branch,
            &recurrent_layer.attention_norm,
            [&s.hidden, &s.normalized, &s.rotated_hidden],
            rows,
            eps,
        )?;
        Ok(())
    });
    time("gdn.qkv", &mut |b| {
        m.project(b, &recurrent.qkv, &s.rotated_hidden, &s.query_gate, rows)
    });
    time("gdn.gate", &mut |b| {
        m.project(b, &recurrent.gate, &s.rotated_hidden, &s.gate, rows)
    });
    time("gdn.alpha_beta", &mut |b| {
        m.ops.bf16_matmul_pair(
            b,
            [
                (decay_projection(&recurrent.alpha), &s.alpha),
                (decay_projection(&recurrent.beta), &s.raw_beta),
            ],
            &s.normalized,
            rows,
        )?;
        Ok(())
    });
    time("gdn.conv", &mut |b| {
        m.ops.conv_sequence_into(
            b,
            &s.query_gate,
            &recurrent.convolution,
            &recurrent.history,
            &recurrent.history,
            &s.convolved,
            rows,
        )?;
        Ok(())
    });
    time("gdn.l2", &mut |b| {
        m.ops.l2_normalize_qk_rows(b, &s.convolved, eps, rows)?;
        Ok(())
    });
    time("gdn.decay", &mut |b| {
        m.ops.decay_beta_rows(
            b,
            &recurrent.decay,
            &s.alpha,
            &recurrent.dt,
            &s.raw_beta,
            &s.decay,
            &s.beta,
            rows,
        )?;
        Ok(())
    });
    time("gdn.scan", &mut |b| {
        m.ops.gdn_sequence_into(
            b,
            m.state_format,
            &s.convolved,
            &s.decay,
            &s.beta,
            &recurrent.state,
            &recurrent.state,
            &s.recurrent_output,
            rows,
        )?;
        Ok(())
    });
    time("gdn.post", &mut |b| {
        m.ops.gdn_postprocess_transform(
            b,
            &s.recurrent_output,
            &s.gate,
            &recurrent.norm,
            &m.attention_rotation,
            &s.rotated_attention,
            eps,
            rows,
        )?;
        Ok(())
    });
    time("out.all_rows", &mut |b| {
        m.project(b, &recurrent.output, &s.rotated_attention, &s.branch, rows)
    });
    time("out.gpu_rows", &mut |b| {
        let matrix = recurrent.output.ptq1_matrix()?;
        m.kernels.matmul(
            b,
            matrix.row_slice(SPLIT_ROWS, matrix.rows() - SPLIT_ROWS)?,
            &s.rotated_attention,
            &s.ffn_gate,
            rows,
        )?;
        Ok(())
    });
    time("attn.q", &mut |b| {
        m.project(b, &full.query_gate, &s.rotated_hidden, &s.query_gate, rows)
    });
    time("attn.k", &mut |b| {
        m.project(b, &full.key, &s.rotated_hidden, &s.key, rows)
    });
    time("attn.v", &mut |b| {
        m.project(b, &full.value, &s.rotated_hidden, &s.value, rows)
    });
    time("attn.prep", &mut |b| {
        m.ops.prepare_attention_rows_kv(
            m.kv_layout,
            b,
            &s.query_gate,
            &s.key,
            &s.value,
            &full.query_norm,
            &full.key_norm,
            &s.query,
            &s.gate,
            &full.key_cache,
            &full.value_cache,
            positions[0] as u32,
            m.kv_allocated as u32,
            eps,
            m.rope_base,
            rows,
            Rope::TEXT,
        )?;
        Ok(())
    });
    time("attn.rotate", &mut |b| m.rotate_attention(b, rows));
    time("ffn.gate", &mut |b| {
        m.project(
            b,
            &recurrent_layer.gate,
            &s.rotated_hidden,
            &s.ffn_gate,
            rows,
        )
    });
    time("ffn.up", &mut |b| {
        m.project(b, &recurrent_layer.up, &s.rotated_hidden, &s.ffn_up, rows)
    });
    time("ffn.swiglu_rotate", &mut |b| {
        m.kernels.swiglu_transform(
            b,
            &m.ffn_rotation,
            &s.ffn_gate,
            &s.ffn_up,
            &s.rotated_ffn,
            rows,
        )?;
        Ok(())
    });
    time("ffn.down", &mut |b| {
        m.project(b, &recurrent_layer.down, &s.rotated_ffn, &s.branch, rows)
    });
    let lane = &split.lanes[0];
    time("ane.stage_ffn", &mut |b| split.stage(b, recurrent_layer, 0));
    time("ane.stage_out", &mut |b| {
        split.stage_out(b, &recurrent.output, 0, SPLIT_ROWS)
    });
    time("ane.stage_qkv", &mut |b| split.stage_qkv(b, full, 0));
    time("ane.join_qkv", &mut |b| {
        split.join_qkv(b, rows, [&s.query_gate, &s.key, &s.value])
    });
    time("ane.pack_ffn", &mut |b| {
        split.kernels.pack(
            b,
            &s.rotated_hidden,
            WIDTH as u32,
            rows,
            ROWS,
            &lane.tx,
            [&lane.x[0], &lane.x[1]],
        )?;
        Ok(())
    });
    time("ane.pack_out", &mut |b| {
        split.kernels.pack(
            b,
            &s.rotated_attention,
            MIXED as u32,
            rows,
            ROWS,
            &lane.tx,
            [&lane.o[0], &lane.o[1]],
        )?;
        Ok(())
    });
    time("ane.join_ffn", &mut |b| {
        split.kernels.join(
            b,
            &lane.y,
            &lane.tx,
            &lane.ffn,
            &split.status,
            WIDTH as u32,
            rows,
        )?;
        Ok(())
    });
    time("ane.join_out", &mut |b| {
        split.kernels.join_split(
            b,
            &lane.y,
            0,
            &lane.tx,
            &s.ffn_gate,
            &s.branch,
            &split.status,
            WIDTH as u32,
            SPLIT_ROWS,
            rows,
        )?;
        Ok(())
    });
    let get = |name: &str| {
        fixed
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|&(_, ms)| ms)
            .expect(name)
    };
    let sum = |names: &[&str]| names.iter().map(|name| get(name)).sum::<f64>();
    for (name, ms) in &fixed {
        println!(
            "{}",
            serde_json::json!({"op": name, "rows": rows, "ms": ms})
        );
    }
    let gdn_mixer = sum(&[
        "gdn.qkv",
        "gdn.gate",
        "gdn.alpha_beta",
        "gdn.conv",
        "gdn.l2",
        "gdn.decay",
        "gdn.scan",
        "gdn.post",
    ]);
    let attention_fixed = sum(&["attn.q", "attn.k", "attn.v", "attn.prep", "attn.rotate"]);
    let ffn = sum(&["ffn.gate", "ffn.up", "ffn.swiglu_rotate", "ffn.down"]);
    let wave_extra = sum(&[
        "ane.pack_ffn",
        "ane.pack_out",
        "ane.join_ffn",
        "ane.join_out",
    ]);
    let stage = sum(&["ane.stage_ffn", "ane.stage_out"]);
    let norm = get("norm");
    for &position in &positions {
        let core = gpu_ms(m, |b: &mut CommandBatch| {
            m.ops.attention_block_kv(
                m.kv_layout,
                b,
                &s.query,
                &full.key_cache,
                &full.value_cache,
                Some(&s.gate),
                &s.attention_output,
                position as u32,
                rows,
                &s.attention,
            )?;
            Ok(())
        });
        // A GPU block: every layer's two normalizations, mixer, output
        // projection and FFN.
        let gpu = [
            ("norms", 128.0 * norm),
            ("gdn_mixers", 48.0 * gdn_mixer),
            ("attention_projections_prep", 16.0 * attention_fixed),
            ("attention_core", 16.0 * core),
            ("output_projections", 64.0 * get("out.all_rows")),
            ("ffn", 64.0 * ffn),
        ];
        // A wavefront half: the same mixers, the GPU's output rows, packs and
        // joins, and half of each layer's weight conversion.
        let wave = [
            ("norms", 128.0 * norm),
            ("gdn_mixers", 48.0 * gdn_mixer),
            ("attention_projections_prep", 16.0 * attention_fixed),
            ("attention_core", 16.0 * core),
            ("output_gpu_rows", 64.0 * get("out.gpu_rows")),
            ("packs_joins", 64.0 * wave_extra),
            ("weight_conversion_half", 32.0 * stage),
        ];
        // The mean of a wavefront's halves at `QKV_STEP`: half B's attention
        // layers take Q, K and V from the ANE (one more pack and the join)
        // and their weights take converting.
        let qkv_change = 8.0
            * (sum(&["ane.stage_qkv", "ane.pack_ffn", "ane.join_qkv"])
                - sum(&["attn.q", "attn.k", "attn.v"]));
        let total = |parts: &[(&str, f64)]| parts.iter().map(|&(_, ms)| ms).sum::<f64>();
        println!(
            "{}",
            serde_json::json!({
                "position": position,
                "attention_core_ms": core,
                "gpu_block_ms": total(&gpu),
                "gpu_block": gpu.iter().map(|&(k, v)| (k, v)).collect::<std::collections::BTreeMap<_, _>>(),
                "wave_half_gpu_ms": total(&wave),
                "wave_half": wave.iter().map(|&(k, v)| (k, v)).collect::<std::collections::BTreeMap<_, _>>(),
                "wave_half_qkv_ms": total(&wave) + qkv_change,
            })
        );
    }
    drop(fixed);
    model.ane = super::AnePrefill::Ready(split);
    // Whole blocks at each position: one GPU block of 512 rows and one
    // wavefront of two halves at the largest output split, without and with
    // the attention Q/K/V on the ANE, wall time and summed command-buffer
    // GPU time.
    for &position in &positions {
        let tokens = &corpus[seed..seed + WAVE_ROWS];
        model.position = position;
        model.take_gpu_time();
        let started = Instant::now();
        model
            .forward_block(&tokens[..ROWS as usize], BlockOutput::None)
            .expect("GPU block");
        let gpu_wall = started.elapsed().as_secs_f64() * 1e3;
        let gpu_busy = model.take_gpu_time().as_secs_f64() * 1e3;
        let mut wave = |step: usize| {
            OUT_STEP_OVERRIDE.store(step + 1, std::sync::atomic::Ordering::Relaxed);
            model.position = position;
            let started = Instant::now();
            model
                .encode_prefill_block(tokens, BlockOutput::None)
                .expect("wavefront");
            let wall = started.elapsed().as_secs_f64() * 1e3;
            let busy = model.take_gpu_time().as_secs_f64() * 1e3;
            assert!(model.ane_active(), "{}", model.neural_engine_policy());
            (wall, busy)
        };
        // Each twice, alternating; the better of each.
        let mut best = [(f64::INFINITY, 0.0); 2];
        for _ in 0..2 {
            for (slot, step) in [OUT_STEPS.len() - 1, QKV_STEP].into_iter().enumerate() {
                let (wall, busy) = wave(step);
                if wall < best[slot].0 {
                    best[slot] = (wall, busy);
                }
            }
        }
        OUT_STEP_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
        println!(
            "{}",
            serde_json::json!({
                "position": position,
                "gpu_block_wall_ms": gpu_wall,
                "gpu_block_busy_ms": gpu_busy,
                "wave_block_wall_ms": best[0].0,
                "wave_block_busy_ms": best[0].1,
                "wave_qkv_block_wall_ms": best[1].0,
                "wave_qkv_block_busy_ms": best[1].1,
            })
        );
    }
}
