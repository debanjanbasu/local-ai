#![allow(clippy::expect_used, clippy::too_many_lines)]

use local_metal::{
    batch::CommandBatch,
    bonsai_ops::{AttentionKernel, AttentionWorkspace, Bf16Matrix, BonsaiOps, MAX_PREFILL_TOKENS},
    buffer::MetalBuffer,
    context::MetalContext,
    shaders::ShaderLibrary,
};

/// `None` when this machine exposes no Metal device, so GPU tests skip
/// instead of failing. Any other failure is still a failure: a missing GPU
/// is an environment, a broken context is a bug.
fn gpu_or_skip() -> Option<MetalContext> {
    let context = MetalContext::new();
    if matches!(context, Err(local_metal::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    Some(context.expect("Metal context failed for a reason other than a missing device"))
}

fn setup() -> Option<(MetalContext, BonsaiOps)> {
    let context = gpu_or_skip()?;
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let ops = BonsaiOps::new(&context, &shaders).expect("ops");
    Some((context, ops))
}

fn floats(context: &MetalContext, values: &[f32]) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), values).expect("float buffer")
}

fn guarded(context: &MetalContext, count: usize) -> MetalBuffer {
    floats(context, &vec![f32::NAN; count + 7])
}

fn signal(index: usize, salt: usize) -> f32 {
    ((index * 17 + index / 29 * 11 + salt * 37) % 251) as f32 / 131.0 - 0.91
}

fn check(buffer: &MetalBuffer, expected: &[f64], tolerance: f64) {
    for (i, (&actual, &wanted)) in buffer.as_slice::<f32>().iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite()
                && (f64::from(actual) - wanted).abs() <= tolerance * (1.0 + wanted.abs()),
            "element {i}: {actual} != {wanted}"
        );
    }
    assert!(
        buffer.as_slice::<f32>()[expected.len()..]
            .iter()
            .all(|v| v.is_nan())
    );
}

#[test]
fn bf16_batch_uses_each_unaligned_row_and_matrix_offset() {
    let Some((ctx, ops)) = setup() else { return };
    let (tokens, rows, columns) = (4, 7, 35);
    let mut packed = vec![0xdead_u16; 3];
    packed.extend((0..rows * columns).map(|i| {
        let value = if i % 41 == 0 {
            if i % 82 == 0 { 100_352.0 } else { -99_840.0 }
        } else {
            signal(i, 3) * 9.0
        };
        half::bf16::from_f32(value).to_bits()
    }));
    let x_data = (0..tokens * columns)
        .map(|i| ((i / columns) as f32).mul_add(0.17, signal(i, 13)))
        .collect::<Vec<_>>();
    let weights = MetalBuffer::from_slice(ctx.device(), &packed).expect("weights");
    let x = floats(&ctx, &x_data);
    let y = guarded(&ctx, tokens * rows);
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.bf16_matmul(
        &mut batch,
        Bf16Matrix {
            buffer: &weights,
            offset: 6,
            rows: rows as u32,
            columns: columns as u32,
        },
        &x,
        &y,
        tokens as u32,
    )
    .expect("encode");
    batch.commit_and_wait().expect("completion");
    let expected = (0..tokens)
        .flat_map(|token| {
            let packed = &packed;
            let x_data = &x_data;
            (0..rows).map(move |row| {
                (0..columns)
                    .map(|column| {
                        f64::from(f32::from_bits(
                            u32::from(packed[3 + row * columns + column]) << 16,
                        )) * f64::from(x_data[token * columns + column])
                    })
                    .sum()
            })
        })
        .collect::<Vec<_>>();
    check(&y, &expected, 4e-5);
}

#[test]
fn decay_rows_vary_by_token_and_head() {
    let Some((ctx, ops)) = setup() else { return };
    let tokens = 5;
    let a_data = (0..48)
        .map(|h| (h as f32).mul_add(0.013, -2.1))
        .collect::<Vec<_>>();
    let dt_data = (0..48)
        .map(|h| (h as f32).mul_add(0.021, -0.7))
        .collect::<Vec<_>>();
    let alpha_data = (0..tokens * 48)
        .map(|i| signal(i, 5) * 1.3)
        .collect::<Vec<_>>();
    let raw_data = (0..tokens * 48)
        .map(|i| signal(i, 9) * 2.1)
        .collect::<Vec<_>>();
    let (a, dt) = (floats(&ctx, &a_data), floats(&ctx, &dt_data));
    let (alpha, raw) = (floats(&ctx, &alpha_data), floats(&ctx, &raw_data));
    let (decay, beta) = (guarded(&ctx, tokens * 48), guarded(&ctx, tokens * 48));
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.decay_beta_rows(
        &mut batch,
        &a,
        &alpha,
        &dt,
        &raw,
        &decay,
        &beta,
        tokens as u32,
    )
    .expect("encode");
    batch.commit_and_wait().expect("completion");
    let expected_decay = (0..tokens * 48)
        .map(|i| {
            let h = i % 48;
            (f64::from(a_data[h]) * (f64::from(alpha_data[i] + dt_data[h])).exp().ln_1p()).exp()
        })
        .collect::<Vec<_>>();
    let expected_beta = raw_data
        .iter()
        .map(|&v| 1.0 / (1.0 + (-f64::from(v)).exp()))
        .collect::<Vec<_>>();
    check(&decay, &expected_decay, 2e-6);
    check(&beta, &expected_beta, 2e-6);
}

#[test]
fn batched_conv_l2_gdn_and_postprocess_match_causal_f64_recurrence() {
    let Some((ctx, ops)) = setup() else { return };
    let tokens = 5;
    let input_data = (0..tokens * 10240)
        .map(|i| signal(i, 17))
        .collect::<Vec<_>>();
    let weight_data = (0..10240 * 4)
        .map(|i| signal(i, 3) * 0.23)
        .collect::<Vec<_>>();
    let initial_history = (0..10240 * 3)
        .map(|i| signal(i, 7) * 0.08)
        .collect::<Vec<_>>();
    let initial_state = (0..48 * 128 * 128)
        .map(|i| signal(i, 8) * 0.012)
        .collect::<Vec<_>>();
    let decay_data = (0..tokens * 48)
        .map(|i| ((i / 48) as f32).mul_add(0.013, ((i % 48) as f32).mul_add(0.009, 0.31)))
        .collect::<Vec<_>>();
    let beta_data = (0..tokens * 48)
        .map(|i| ((i % 11) as f32).mul_add(0.047, 0.11))
        .collect::<Vec<_>>();
    let z_data = (0..tokens * 6144)
        .map(|i| signal(i, 29) * 2.0)
        .collect::<Vec<_>>();
    let norm_data = (0..128)
        .map(|i| signal(i, 1).mul_add(0.1, 0.6))
        .collect::<Vec<_>>();
    let mut history_ref = initial_history
        .iter()
        .copied()
        .map(f64::from)
        .collect::<Vec<_>>();
    let mut state_ref = initial_state
        .iter()
        .copied()
        .map(f64::from)
        .collect::<Vec<_>>();
    let mut qkv_ref = vec![0.0; tokens * 10240];
    let mut recurrent_ref = vec![0.0; tokens * 6144];
    let mut grouped_ref = vec![0.0; tokens * 6144];
    let mut prefix_reference = None;
    for token in 0..tokens {
        let qkv = &mut qkv_ref[token * 10240..][..10240];
        for c in 0..10240 {
            let sum = f64::mul_add(
                f64::from(input_data[token * 10240 + c]),
                f64::from(weight_data[c * 4 + 3]),
                (0..3)
                    .map(|j| history_ref[c * 3 + j] * f64::from(weight_data[c * 4 + j]))
                    .sum::<f64>(),
            );
            qkv[c] = sum / (1.0 + (-sum).exp());
            history_ref.copy_within(c * 3 + 1..c * 3 + 3, c * 3);
            history_ref[c * 3 + 2] = f64::from(input_data[token * 10240 + c]);
        }
        for head in 0..32 {
            let row = &mut qkv[head * 128..][..128];
            let scale = row.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-6);
            for value in row {
                *value /= scale;
            }
        }
        let output = &mut recurrent_ref[token * 6144..][..6144];
        for head in 0..48 {
            let q = &qkv[(head % 16) * 128..][..128];
            let k = &qkv[2048 + (head % 16) * 128..][..128];
            for d in 0..128 {
                let row = head * 128 + d;
                let memory = &mut state_ref[row * 128..][..128];
                for value in &mut *memory {
                    *value *= f64::from(decay_data[token * 48 + head]);
                }
                let prediction = memory.iter().zip(k).map(|(s, k)| s * k).sum::<f64>();
                let correction =
                    (qkv[4096 + row] - prediction) * f64::from(beta_data[token * 48 + head]);
                for (s, k) in memory.iter_mut().zip(k) {
                    *s += k * correction;
                }
                output[row] =
                    memory.iter().zip(q).map(|(s, q)| s * q).sum::<f64>() / 128.0_f64.sqrt();
            }
        }
        let grouped = &mut grouped_ref[token * 6144..][..6144];
        for group in 0..16 {
            for replica in 0..3 {
                let head = group + replica * 16;
                let values = &output[head * 128..][..128];
                let rms = (values.iter().map(|v| v * v).sum::<f64>() / 128.0 + 1e-6).sqrt();
                for d in 0..128 {
                    let gate = f64::from(z_data[token * 6144 + head * 128 + d]);
                    grouped[(group * 3 + replica) * 128 + d] =
                        values[d] / rms * f64::from(norm_data[d]) * gate / (1.0 + (-gate).exp());
                }
            }
        }
        if token == 1 {
            prefix_reference = Some((history_ref.clone(), state_ref.clone()));
        }
    }
    let (prefix_history, prefix_state) = prefix_reference.expect("two-token reference");
    let weights = floats(&ctx, &weight_data);
    let mut history = floats(&ctx, &initial_history);
    let mut state = floats(&ctx, &initial_state);
    let norm = floats(&ctx, &norm_data);
    // Whole block, split continuation, then shorter reuse after reset. Inputs
    // retain future rows, while output guards expose any writes past the block.
    for (start, count, reset) in [(0, 5, true), (0, 2, true), (2, 3, false), (0, 2, true)] {
        if reset {
            history
                .as_mut_slice::<f32>()
                .copy_from_slice(&initial_history);
            state.as_mut_slice::<f32>().copy_from_slice(&initial_state);
        }
        let input = floats(&ctx, &input_data[start * 10240..]);
        let decay = floats(&ctx, &decay_data[start * 48..]);
        let beta = floats(&ctx, &beta_data[start * 48..]);
        let z = floats(&ctx, &z_data[start * 6144..]);
        let qkv = guarded(&ctx, count * 10240);
        let recurrent = guarded(&ctx, count * 6144);
        let grouped = guarded(&ctx, count * 6144);
        let mut batch = CommandBatch::new(&ctx).expect("batch");
        ops.conv_sequence(&mut batch, &input, &weights, &history, &qkv, count as u32)
            .expect("conv");
        ops.l2_normalize_qk_rows(&mut batch, &qkv, 1e-6, count as u32)
            .expect("l2");
        ops.gdn_sequence(
            &mut batch,
            &qkv,
            &decay,
            &beta,
            &state,
            &recurrent,
            count as u32,
        )
        .expect("gdn");
        ops.gdn_postprocess_rows(
            &mut batch,
            &recurrent,
            &z,
            &norm,
            &grouped,
            1e-6,
            count as u32,
        )
        .expect("post");
        batch.commit_and_wait().expect("completion");
        let end = start + count;
        check(&qkv, &qkv_ref[start * 10240..end * 10240], 3e-6);
        check(&recurrent, &recurrent_ref[start * 6144..end * 6144], 3e-6);
        check(&grouped, &grouped_ref[start * 6144..end * 6144], 4e-5);
        if end == 2 {
            check(&history, &prefix_history, 0.0);
            check(&state, &prefix_state, 4e-6);
        } else {
            check(&history, &history_ref, 0.0);
            check(&state, &state_ref, 4e-6);
        }
    }
}

fn normalized_rope(values: &[f32], weights: &[f32], position: u32) -> Vec<f64> {
    let inverse = (values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / 256.0 + 1e-6)
        .sqrt()
        .recip();
    let mut result = values
        .iter()
        .zip(weights)
        .map(|(&v, &w)| f64::from(v) * inverse * f64::from(w))
        .collect::<Vec<_>>();
    for pair in 0..32 {
        let angle = f64::from(position) / 10_000_000.0_f64.powf(pair as f64 / 32.0);
        let (a, b) = (result[pair], result[pair + 32]);
        result[pair] = f64::mul_add(b, -angle.sin(), a * angle.cos());
        result[pair + 32] = f64::mul_add(b, angle.cos(), a * angle.sin());
    }
    result
}

#[test]
fn attention_prefill_rows_use_row_offsets_causal_prefixes_and_one_workspace() {
    let Some((ctx, ops)) = setup() else { return };
    let (tokens, position, capacity) = (3, 126_u32, 130_u32);
    let qg_data = (0..tokens * 12288)
        .map(|i| ((i / 12288) as f32).mul_add(0.2, signal(i, 5)))
        .collect::<Vec<_>>();
    let k_data = (0..tokens * 1024)
        .map(|i| signal(i, 11))
        .collect::<Vec<_>>();
    let v_data = (0..tokens * 1024)
        .map(|i| signal(i, 31) * 1.4)
        .collect::<Vec<_>>();
    let query_norm_weights = (0..256)
        .map(|i| signal(i, 19).mul_add(0.2, 0.7))
        .collect::<Vec<_>>();
    let key_norm_weights = (0..256)
        .map(|i| signal(i, 23).mul_add(0.2, 0.8))
        .collect::<Vec<_>>();
    let old_keys = (0..capacity as usize * 1024)
        .map(|i| half::f16::from_f32(signal(i, 41)))
        .collect::<Vec<_>>();
    let old_values = (0..capacity as usize * 1024)
        .map(|i| half::f16::from_f32(signal(i, 43)))
        .collect::<Vec<_>>();
    let qg = floats(&ctx, &qg_data);
    let k = floats(&ctx, &k_data);
    let v = floats(&ctx, &v_data);
    let qw = floats(&ctx, &query_norm_weights);
    let kw = floats(&ctx, &key_norm_weights);
    let query = guarded(&ctx, tokens * 6144);
    let gate = guarded(&ctx, tokens * 6144);
    let output = guarded(&ctx, tokens * 6144);
    let ungated_output = guarded(&ctx, tokens * 6144);
    let kc = MetalBuffer::from_slice(ctx.device(), &old_keys).expect("kc");
    let vc = MetalBuffer::from_slice(ctx.device(), &old_values).expect("vc");
    let workspace = AttentionWorkspace::new(&ctx, 129).expect("workspace");
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.prepare_attention_rows(
        &mut batch,
        &qg,
        &k,
        &v,
        &qw,
        &kw,
        &query,
        &gate,
        &kc,
        &vc,
        position,
        capacity,
        1e-6,
        10_000_000.0,
        tokens as u32,
    )
    .expect("prep");
    ops.attention_block(
        &mut batch,
        &query,
        &kc,
        &vc,
        Some(&gate),
        &output,
        position,
        tokens as u32,
        &workspace,
    )
    .expect("attention block");
    ops.attention_block(
        &mut batch,
        &query,
        &kc,
        &vc,
        None,
        &ungated_output,
        position,
        tokens as u32,
        &workspace,
    )
    .expect("ungated attention block");
    batch.commit_and_wait().expect("completion");
    let mut expected_q = Vec::new();
    let mut expected_gate = Vec::new();
    let mut expected_keys = old_keys.clone();
    let mut expected_values = old_values.clone();
    for token in 0..tokens {
        for head in 0..24 {
            expected_q.extend(normalized_rope(
                &qg_data[token * 12288 + head * 512..][..256],
                &query_norm_weights,
                position + token as u32,
            ));
            expected_gate.extend(
                qg_data[token * 12288 + head * 512 + 256..][..256]
                    .iter()
                    .copied()
                    .map(f64::from),
            );
        }
        for head in 0..4 {
            let key = normalized_rope(
                &k_data[token * 1024 + head * 256..][..256],
                &key_norm_weights,
                position + token as u32,
            );
            for d in 0..256 {
                let index = (position as usize + token) * 1024 + head * 256 + d;
                expected_keys[index] = half::f16::from_f64(key[d]);
                expected_values[index] = half::f16::from_f32(v_data[token * 1024 + head * 256 + d]);
            }
        }
    }
    check(&query, &expected_q, 3e-5);
    check(&gate, &expected_gate, 0.0);
    for i in 0..capacity as usize * 1024 {
        if i < position as usize * 1024 || i >= (position as usize + tokens) * 1024 {
            assert_eq!(kc.as_slice::<half::f16>()[i], old_keys[i]);
            assert_eq!(vc.as_slice::<half::f16>()[i], old_values[i]);
        } else {
            let local = i - position as usize * 1024;
            let token = local / 1024;
            let coordinate = local % 1024;
            let head = coordinate / 256;
            let expected_k = normalized_rope(
                &k_data[token * 1024 + head * 256..][..256],
                &key_norm_weights,
                position + token as u32,
            );
            assert!(
                (f64::from(kc.as_slice::<half::f16>()[i].to_f32()) - expected_k[coordinate % 256])
                    .abs()
                    < 0.002
            );
            assert_eq!(
                vc.as_slice::<half::f16>()[i],
                half::f16::from_f32(v_data[local])
            );
        }
    }
    for row in 0..tokens {
        for head in 0..24 {
            let prefix = position as usize + row + 1;
            let mut scores = (0..prefix)
                .map(|t| {
                    (0..256)
                        .map(|d| {
                            expected_q[row * 6144 + head * 256 + d]
                                * f64::from(expected_keys[(t * 4 + head / 6) * 256 + d].to_f32())
                        })
                        .sum::<f64>()
                        / 16.0
                })
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores {
                *score = (*score - max).exp();
            }
            let denominator = scores.iter().sum::<f64>();
            for d in 0..256 {
                let attention = scores
                    .iter()
                    .enumerate()
                    .map(|(t, s)| {
                        s * f64::from(expected_values[(t * 4 + head / 6) * 256 + d].to_f32())
                    })
                    .sum::<f64>()
                    / denominator;
                let g = expected_gate[row * 6144 + head * 256 + d];
                let wanted = attention / (1.0 + (-g).exp());
                let actual = output.as_slice::<f32>()[row * 6144 + head * 256 + d];
                assert!((f64::from(actual) - wanted).abs() < 3e-5 * (1.0 + wanted.abs()));
                let actual = ungated_output.as_slice::<f32>()[row * 6144 + head * 256 + d];
                assert!((f64::from(actual) - attention).abs() < 3e-5 * (1.0 + attention.abs()));
            }
        }
    }
    assert!(
        output.as_slice::<f32>()[tokens * 6144..]
            .iter()
            .all(|v| v.is_nan())
    );
    assert!(
        ungated_output.as_slice::<f32>()[tokens * 6144..]
            .iter()
            .all(|v| v.is_nan())
    );
}

#[test]
fn attention_block_rejects_invalid_extents_and_aliases_without_dispatch() {
    let Some((ctx, ops)) = setup() else { return };
    let workspace = AttentionWorkspace::new(&ctx, 129).expect("workspace");
    let q = guarded(&ctx, 128 * 6144);
    let k =
        MetalBuffer::from_slice(ctx.device(), &vec![half::f16::ZERO; 129 * 1024]).expect("keys");
    let v =
        MetalBuffer::from_slice(ctx.device(), &vec![half::f16::ZERO; 129 * 1024]).expect("values");
    let out = guarded(&ctx, 128 * 6144);
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    for (position, tokens) in [(0, 0), (0, 129), (128, 2), (u32::MAX, 1)] {
        assert!(
            ops.attention_block(
                &mut batch, &q, &k, &v, None, &out, position, tokens, &workspace,
            )
            .is_err()
        );
    }
    assert!(
        ops.attention_block(&mut batch, &q, &k, &v, None, &q, 0, 1, &workspace)
            .is_err()
    );
    assert!(
        ops.attention_block(&mut batch, &q, &k, &v, Some(&out), &out, 0, 1, &workspace,)
            .is_err()
    );
    assert_eq!(batch.dispatch_count(), 0);
}

#[test]
fn attention_block_handles_all_query_tile_tails_and_ignores_future_cache() {
    let Some((ctx, ops)) = setup() else { return };
    let position = 129_u32;
    let capacity = position as usize + 128;
    let workspace = AttentionWorkspace::new(&ctx, capacity as u32).expect("workspace");
    let keys = MetalBuffer::from_slice(ctx.device(), &vec![half::f16::ZERO; capacity * 1024])
        .expect("keys");
    let values_data = (0..capacity * 1024)
        .map(|i| {
            let token = i / 1024;
            let coordinate = i % 1024;
            let value = if token < position as usize {
                ((coordinate % 29) as f32).mul_add(0.003_906_25, (token % 17) as f32 * 0.03125)
            } else {
                // Future rows are conspicuous, but become visible one at a time.
                40.0 + (token - position as usize) as f32
            };
            half::f16::from_f32(value)
        })
        .collect::<Vec<_>>();
    let values = MetalBuffer::from_slice(ctx.device(), &values_data).expect("values");
    for tokens in [1_u32, 7, 8, 9, 31, 32, 128] {
        let query = floats(&ctx, &vec![0.0; tokens as usize * 6144]);
        let output = guarded(&ctx, tokens as usize * 6144);
        let mut batch = CommandBatch::new(&ctx).expect("batch");
        ops.attention_block(
            &mut batch, &query, &keys, &values, None, &output, position, tokens, &workspace,
        )
        .expect("attention block tail");
        batch.commit_and_wait().expect("completion");
        for row in 0..tokens as usize {
            let prefix = position as usize + row + 1;
            for head in 0..24 {
                for d in 0..256 {
                    let coordinate = (head / 6) * 256 + d;
                    let expected = (0..prefix)
                        .map(|token| f64::from(values_data[token * 1024 + coordinate].to_f32()))
                        .sum::<f64>()
                        / prefix as f64;
                    let actual = f64::from(output.as_slice::<f32>()[(row * 24 + head) * 256 + d]);
                    assert!((actual - expected).abs() < 3e-5 * (1.0 + expected.abs()));
                }
            }
        }
        assert!(
            output.as_slice::<f32>()[tokens as usize * 6144..]
                .iter()
                .all(|v| v.is_nan())
        );
    }
}

#[test]
fn attention_block_does_not_clip_large_finite_logits() {
    let Some((ctx, ops)) = setup() else { return };
    let (position, tokens) = (126_usize, 3_usize);
    let capacity = position + tokens;
    let workspace = AttentionWorkspace::new(&ctx, capacity as u32).expect("workspace");
    let mut q = vec![0.0_f32; tokens * 6144];
    let mut k = vec![half::f16::ZERO; capacity * 1024];
    let values = (0..capacity * 1024)
        .map(|i| half::f16::from_f32(signal(i, 67)))
        .collect::<Vec<_>>();
    for row in 0..tokens {
        for head in 0..24 {
            q[(row * 24 + head) * 256] = 16.0;
        }
        for head in 0..4 {
            k[((position + row) * 4 + head) * 256] = half::f16::from_f32((2048 << row) as f32);
        }
    }
    let q = floats(&ctx, &q);
    let k = MetalBuffer::from_slice(ctx.device(), &k).expect("keys");
    let v = MetalBuffer::from_slice(ctx.device(), &values).expect("values");
    let out = guarded(&ctx, tokens * 6144);
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.attention_block(
        &mut batch,
        &q,
        &k,
        &v,
        None,
        &out,
        position as u32,
        tokens as u32,
        &workspace,
    )
    .expect("attention");
    batch.commit_and_wait().expect("completion");
    // Each newly visible row has a logit at least 2048 above every other key.
    // F64 softmax underflows all other probabilities to zero; clipping logits
    // instead would average different values and fail this check.
    let expected = (0..tokens * 6144)
        .map(|i| {
            let row = i / 6144;
            let head = (i % 6144) / 256;
            f64::from(values[(position + row) * 1024 + (head / 6) * 256 + i % 256])
        })
        .collect::<Vec<_>>();
    check(&out, &expected, 1e-6);
}

#[test]
fn attention_preserves_f32_queries_and_tiny_probabilities_across_key_and_query_tiles() {
    let Some(ctx) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(ctx.device()).expect("shaders");
    let portable = BonsaiOps::new_with_attention_kernel(&ctx, &shaders, AttentionKernel::SimdF32)
        .expect("portable ops");
    let selected = BonsaiOps::new(&ctx, &shaders).expect("selected ops");
    for (position, tokens) in [
        (0, 1),
        (55, 17),
        (63, 2),
        (64, 7),
        (65, 8),
        (127, 9),
        (257, 128),
    ] {
        let end = position + tokens;
        let workspace = AttentionWorkspace::new(&ctx, end as u32).expect("workspace");
        let mut queries = vec![0.0_f32; tokens * 6144];
        // Readable poisoned cache tails distinguish true tensor bounds from
        // merely allocating a sufficiently large buffer for an unmasked load.
        let mut keys = vec![half::f16::NAN; (end + 64) * 1024];
        keys[..end * 1024].fill(half::f16::ZERO);
        let mut values = vec![half::f16::NAN; (end + 64) * 1024];
        for row in 0..tokens {
            for head in 0..24 {
                queries[row * 6144 + head * 256] = 70_000.25 + (row * 13 + head * 37) as f32;
            }
        }
        for token in 0..end {
            for head in 0..4 {
                keys[token * 1024 + head * 256] = half::f16::from_f32(if token == 0 {
                    0.0
                } else {
                    (((token * 3 + head) % 7) as f32).mul_add(-0.000_001, -0.00388)
                });
                for d in 0..256 {
                    values[token * 1024 + head * 256 + d] = half::f16::from_f32(if token == 0 {
                        0.0
                    } else {
                        (((token * 17 + head * 31 + d * 11) % 301) as f32).mul_add(64.0, 20_000.0)
                    });
                }
            }
        }
        let mut expected = Vec::with_capacity(tokens * 6144);
        for row in 0..tokens {
            for head in 0..24 {
                let q = f64::from(queries[row * 6144 + head * 256]);
                let probabilities = (0..=position + row)
                    .map(|token| {
                        (q * f64::from(keys[token * 1024 + (head / 6) * 256]) / 16.0).exp()
                    })
                    .collect::<Vec<_>>();
                let denominator = probabilities.iter().sum::<f64>();
                for d in 0..256 {
                    expected.push(
                        probabilities
                            .iter()
                            .enumerate()
                            .map(|(token, &p)| {
                                p * f64::from(values[token * 1024 + (head / 6) * 256 + d])
                            })
                            .sum::<f64>()
                            / denominator,
                    );
                }
            }
        }
        let q = floats(&ctx, &queries);
        let k = MetalBuffer::from_slice(ctx.device(), &keys).expect("keys");
        let v = MetalBuffer::from_slice(ctx.device(), &values).expect("values");
        for ops in [&portable, &selected] {
            let out = guarded(&ctx, tokens * 6144);
            let mut batch = CommandBatch::new(&ctx).expect("batch");
            ops.attention_block(
                &mut batch,
                &q,
                &k,
                &v,
                None,
                &out,
                position as u32,
                tokens as u32,
                &workspace,
            )
            .expect("attention");
            batch.commit_and_wait().expect("completion");
            // Half queries overflow; half probabilities round these ~4e-8
            // weights badly. Large finite V exposes either precision downgrade.
            check(&out, &expected, 5e-6);
        }
    }
}

#[test]
#[cfg(any())]
fn mixed_attention_matches_explicit_query_and_probability_rounding() {
    let Some((ctx, selected)) = setup() else {
        return;
    };
    if selected.attention_kernel() == AttentionKernel::SimdF32 {
        eprintln!("mixed attention requires Metal 4 tensor support");
        return;
    }
    let shaders = ShaderLibrary::new(ctx.device()).expect("shaders");
    let kernels = [
        AttentionKernel::TensorF32,
        AttentionKernel::TensorF16Q,
        AttentionKernel::TensorF16QP,
    ];
    let implementations = kernels.map(|kernel| {
        BonsaiOps::new_with_attention_kernel(&ctx, &shaders, kernel).expect("explicit kernel")
    });
    let mut precision_witnesses = [false; 2];
    for (position, tokens) in [(0, 1), (55, 17), (63, 2), (65, 8), (127, 9), (257, 128)] {
        let end = position + tokens;
        let workspace = AttentionWorkspace::new(&ctx, end as u32).expect("workspace");
        let mut queries = vec![0.0; tokens * 6144];
        let mut keys = vec![half::f16::NAN; (end + 64) * 1024];
        keys[..end * 1024].fill(half::f16::ZERO);
        let mut values = vec![half::f16::NAN; (end + 64) * 1024];
        let gates = (0..queries.len())
            .map(|i| signal(i, 57))
            .collect::<Vec<_>>();
        for row in 0..tokens {
            for head in 0..24 {
                queries[row * 6144 + head * 256] =
                    (head as f32).mul_add(0.03113, ((row % 3) as f32).mul_add(0.27, 31.2345));
            }
        }
        for token in 0..end {
            for head in 0..4 {
                // The first key is always the maximum, so a single F64 softmax
                // independently predicts both operand-rounding policies without
                // copying the GPU's blocked online recurrence into the oracle.
                keys[token * 1024 + head * 256] = half::f16::from_f32(if token == 0 {
                    0.0
                } else {
                    -0.037 * ((token * 3 + head) % 19 + 1) as f32
                });
                for d in 0..256 {
                    values[token * 1024 + head * 256 + d] =
                        half::f16::from_f32(signal(token * 1024 + head * 256 + d, 43) * 100.0);
                }
            }
        }
        let query_buffer = floats(&ctx, &queries);
        let key_buffer = MetalBuffer::from_slice(ctx.device(), &keys).expect("keys");
        let value_buffer = MetalBuffer::from_slice(ctx.device(), &values).expect("values");
        let gate_buffer = floats(&ctx, &gates);
        let mut previous: Option<Vec<f64>> = None;
        for (method, ops) in implementations.iter().enumerate() {
            let mut expected = Vec::with_capacity(queries.len());
            for row in 0..tokens {
                for head in 0..24 {
                    let q = queries[row * 6144 + head * 256];
                    let q = if method == 0 {
                        f64::from(q)
                    } else {
                        f64::from(half::f16::from_f32(q))
                    };
                    let probabilities = (0..=position + row)
                        .map(|token| {
                            (q * f64::from(keys[token * 1024 + (head / 6) * 256]) / 16.0).exp()
                        })
                        .collect::<Vec<_>>();
                    let denominator = probabilities.iter().sum::<f64>();
                    for d in 0..256 {
                        let numerator = probabilities
                            .iter()
                            .enumerate()
                            .map(|(token, &p)| {
                                let p = if method == 2 {
                                    f64::from(half::f16::from_f64(p))
                                } else {
                                    p
                                };
                                p * f64::from(values[token * 1024 + (head / 6) * 256 + d])
                            })
                            .sum::<f64>();
                        let gate = f64::from(gates[row * 6144 + head * 256 + d]);
                        expected.push(numerator / denominator / (1.0 + (-gate).exp()));
                    }
                }
            }
            if let Some(prior) = &previous {
                precision_witnesses[method - 1] |= prior
                    .iter()
                    .zip(&expected)
                    .any(|(&a, &b)| (a - b).abs() > 2e-5 * (1.0 + a.abs() + b.abs()));
            }
            let out = guarded(&ctx, queries.len());
            let mut batch = CommandBatch::new(&ctx).expect("batch");
            ops.attention_block(
                &mut batch,
                &query_buffer,
                &key_buffer,
                &value_buffer,
                Some(&gate_buffer),
                &out,
                position as u32,
                tokens as u32,
                &workspace,
            )
            .expect("mixed attention");
            batch.commit_and_wait().expect("completion");
            check(&out, &expected, 1e-5);
            previous = Some(expected);
        }
    }
    // These inputs must distinguish no rounding, Q rounding and Q+P rounding:
    // silently selecting F32 instead of the requested mixed kernel must fail.
    assert!(precision_witnesses.into_iter().all(|witness| witness));
}

#[test]
fn recurrence_alias_checks_have_otherwise_valid_buffer_sizes() {
    let Some((ctx, ops)) = setup() else { return };
    let input = guarded(&ctx, 3 * 10240);
    let weights = guarded(&ctx, 4 * 10240);
    let history = guarded(&ctx, 3 * 10240);
    let convolution = guarded(&ctx, 3 * 10240);
    let state = guarded(&ctx, 48 * 128 * 128);
    let output = guarded(&ctx, 3 * 6144);
    let a = guarded(&ctx, 48);
    let dt = guarded(&ctx, 48);
    let alpha = guarded(&ctx, 3 * 48);
    let beta_raw = guarded(&ctx, 3 * 48);
    let decay = guarded(&ctx, 3 * 48);
    let beta = guarded(&ctx, 3 * 48);
    let norm = guarded(&ctx, 128);
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    assert!(
        ops.conv_sequence(&mut batch, &input, &weights, &history, &input, 3)
            .is_err()
    );
    assert!(
        ops.conv_sequence(&mut batch, &history, &weights, &history, &convolution, 3)
            .is_err()
    );
    assert!(
        ops.decay_beta_rows(&mut batch, &a, &alpha, &dt, &beta_raw, &decay, &decay, 3,)
            .is_err()
    );
    assert!(
        ops.decay_beta_rows(&mut batch, &a, &alpha, &dt, &beta_raw, &alpha, &beta, 3)
            .is_err()
    );
    assert!(
        ops.gdn_sequence(&mut batch, &input, &decay, &beta, &state, &input, 3)
            .is_err()
    );
    assert!(
        ops.gdn_sequence(&mut batch, &state, &decay, &beta, &state, &output, 3)
            .is_err()
    );
    assert!(
        ops.gdn_postprocess_rows(&mut batch, &input, &output, &norm, &output, 1e-6, 3)
            .is_err()
    );
    assert_eq!(batch.dispatch_count(), 0);
}

#[test]
fn batch_validation_rejects_bounds_short_buffers_aliases_and_overflow_before_dispatch() {
    let Some((ctx, ops)) = setup() else { return };
    let tiny = floats(&ctx, &[0.0; 8]);
    let workspace = AttentionWorkspace::new(&ctx, 128).expect("workspace");
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    let matrix = Bf16Matrix {
        buffer: &tiny,
        offset: 0,
        rows: 1,
        columns: 1,
    };
    assert!(
        ops.bf16_matmul(&mut batch, matrix, &tiny, &tiny, 0)
            .is_err()
    );
    assert!(
        ops.bf16_matmul(&mut batch, matrix, &tiny, &tiny, MAX_PREFILL_TOKENS + 1)
            .is_err()
    );
    let one_bf16 = MetalBuffer::from_slice(ctx.device(), &[0x3f80_u16]).expect("bf16");
    let one_float = floats(&ctx, &[1.0]);
    assert!(
        ops.bf16_matmul(
            &mut batch,
            Bf16Matrix {
                buffer: &one_bf16,
                offset: 0,
                rows: 1,
                columns: 1,
            },
            &one_float,
            &one_float,
            1,
        )
        .is_err()
    );
    assert!(
        ops.conv_sequence(&mut batch, &tiny, &tiny, &tiny, &tiny, 1)
            .is_err()
    );
    assert!(
        ops.decay_beta_rows(&mut batch, &tiny, &tiny, &tiny, &tiny, &tiny, &tiny, 1)
            .is_err()
    );
    assert!(
        ops.gdn_sequence(&mut batch, &tiny, &tiny, &tiny, &tiny, &tiny, 1)
            .is_err()
    );
    assert!(
        ops.gdn_postprocess_rows(&mut batch, &tiny, &tiny, &tiny, &tiny, 1e-6, 1)
            .is_err()
    );
    assert!(
        ops.prepare_attention_rows(
            &mut batch,
            &tiny,
            &tiny,
            &tiny,
            &tiny,
            &tiny,
            &tiny,
            &tiny,
            &tiny,
            &tiny,
            u32::MAX,
            1,
            1e-6,
            10_000.0,
            2
        )
        .is_err()
    );
    assert!(
        ops.attention_row(
            &mut batch,
            &tiny,
            &tiny,
            &tiny,
            None,
            &tiny,
            129,
            &workspace,
            u32::MAX
        )
        .is_err()
    );
    assert!(
        ops.attention_row(
            &mut batch,
            &tiny,
            &tiny,
            &tiny,
            None,
            &tiny,
            1,
            &workspace,
            u32::MAX,
        )
        .is_err()
    );
    assert_eq!(batch.dispatch_count(), 0);
}
