#![allow(clippy::expect_used)]

use local_metal::{
    batch::CommandBatch,
    bonsai_ops::{AttentionWorkspace, Bf16Matrix, BonsaiOps, KvFormat, KvLayout, RmsNormParams},
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

#[test]
fn bf16_expansion_preserves_values_outside_f16_range() {
    let Some((ctx, ops)) = setup() else { return };
    // 100000, -2, 0.5, and 65536 as exact BF16 bit patterns, with an offset.
    let weights = MetalBuffer::from_slice(
        ctx.device(),
        &[0_u16, 0_u16, 0x47c3, 0xc000, 0x3f00, 0x4780],
    )
    .expect("weights");
    let x = MetalBuffer::from_slice(ctx.device(), &[1.0_f32, 3.0, -4.0, 0.25]).expect("x");
    let y = MetalBuffer::from_slice(ctx.device(), &[f32::NAN, 12345.0]).expect("y");
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.bf16_matvec(
        &mut batch,
        Bf16Matrix {
            buffer: &weights,
            offset: 4,
            rows: 1,
            columns: 4,
        },
        &x,
        &y,
    )
    .expect("encode");
    batch.commit_and_wait().expect("run");
    let expected = f64::from(f32::from_bits(0x47c3_0000)) - 6.0 - 2.0 + 16384.0;
    assert!((f64::from(y.as_slice::<f32>()[0]) - expected).abs() < 0.02);
    assert_eq!(y.as_slice::<f32>()[1], 12345.0, "guard tail changed");
}

#[test]
fn signed_decay_and_tiny_l2_use_pinned_formulas() {
    let Some((ctx, ops)) = setup() else { return };
    let a = MetalBuffer::from_slice(ctx.device(), &[-2.0_f32; 48]).expect("a");
    let alpha = MetalBuffer::from_slice(ctx.device(), &[0.25_f32; 48]).expect("alpha");
    let dt = MetalBuffer::from_slice(ctx.device(), &[-0.5_f32; 48]).expect("dt");
    let raw = MetalBuffer::from_slice(ctx.device(), &[-1.0_f32; 48]).expect("raw");
    let decay = MetalBuffer::from_slice(ctx.device(), &[0.0_f32; 48]).expect("decay");
    let beta = MetalBuffer::from_slice(ctx.device(), &[0.0_f32; 48]).expect("beta");
    let mut qv = vec![0.0_f32; 32 * 128 + 7];
    for head in 0..32 {
        qv[head * 128 + head] = if head % 2 == 0 { 1.0e-9 } else { -1.0e-15 };
    }
    let q = MetalBuffer::from_slice(ctx.device(), &qv).expect("q");
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.decay_beta(&mut batch, &a, &alpha, &dt, &raw, &decay, &beta)
        .expect("decay");
    ops.l2_normalize_qk(&mut batch, &q, 1.0e-12).expect("l2");
    batch.commit_and_wait().expect("run");
    let softplus = (-0.25_f64).exp().ln_1p();
    assert!((f64::from(decay.as_slice::<f32>()[0]) - (-2.0 * softplus).exp()).abs() < 2e-6);
    assert!((f64::from(beta.as_slice::<f32>()[0]) - 1.0 / (1.0 + 1.0_f64.exp())).abs() < 2e-6);
    for head in 0..32 {
        let expected = if head % 2 == 0 { 1.0_f32 } else { -0.001 };
        assert!(
            (q.as_slice::<f32>()[head * 128 + head] - expected).abs() < 1e-6,
            "head {head}: must clamp sqrt, not add epsilon"
        );
    }
    assert_eq!(&q.as_slice::<f32>()[4096..], &[0.0; 7]);
}

#[test]
fn validation_happens_before_dispatch_and_workspace_is_linear() {
    let Some((ctx, ops)) = setup() else { return };
    let workspace = AttentionWorkspace::new(&ctx, 129).expect("workspace");
    assert_eq!(workspace.byte_len(), 2 * 24 * 258 * 4);
    let tiny = MetalBuffer::from_slice(ctx.device(), &[0.0_f32; 4]).expect("tiny");
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    assert!(ops.sigmoid_mul(&mut batch, &tiny, &tiny, &tiny, 5).is_err());
    assert!(
        ops.attention(
            &mut batch, &tiny, &tiny, &tiny, None, &tiny, 130, &workspace
        )
        .is_err()
    );
    assert_eq!(batch.dispatch_count(), 0);
}

fn floats(context: &MetalContext, values: &[f32]) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), values).expect("float buffer")
}

fn guarded(context: &MetalContext, count: usize) -> MetalBuffer {
    floats(context, &vec![f32::NAN; count + 7])
}

fn check(buffer: &MetalBuffer, expected: &[f64], tolerance: f64) {
    for (index, (&actual, &expected)) in buffer.as_slice::<f32>().iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite()
                && (f64::from(actual) - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "element {index}: {actual} != {expected}"
        );
    }
    assert!(
        buffer.as_slice::<f32>()[expected.len()..]
            .iter()
            .all(|v| v.is_nan()),
        "tail overwritten"
    );
}

fn signal(index: usize, salt: usize) -> f32 {
    ((index * 17 + index / 31 * 13 + salt * 37) % 241) as f32 / 127.0 - 0.87
}

#[test]
fn rms_all_rows_and_elementwise_tails_match_f64() {
    let Some((ctx, ops)) = setup() else { return };
    let (rows, dimension, stride) = (9, 271, 288);
    let input_data = (0..rows * stride).map(|i| signal(i, 1)).collect::<Vec<_>>();
    let weights_data = (0..dimension + 2).map(|i| signal(i, 9)).collect::<Vec<_>>();
    let input = floats(&ctx, &input_data);
    let weights = floats(&ctx, &weights_data);
    let output = guarded(&ctx, rows * stride);
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.rms_norm(
        &mut batch,
        &input,
        &weights,
        &output,
        RmsNormParams {
            dimension: dimension as u32,
            rows: rows as u32,
            stride: stride as u32,
            epsilon: 1e-6,
            weight_offset: 8,
        },
    )
    .expect("RMSNorm");
    batch.commit_and_wait().expect("RMSNorm completion");
    for row in 0..rows {
        let values = &input_data[row * stride..][..dimension];
        let sum = values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let inverse = (sum / dimension as f64 + f64::from(1e-6_f32))
            .sqrt()
            .recip();
        for (i, &value) in values.iter().enumerate() {
            let expected = f64::from(value) * inverse * f64::from(weights_data[i + 2]);
            let actual = output.as_slice::<f32>()[row * stride + i];
            assert!(actual.is_finite() && (f64::from(actual) - expected).abs() < 2e-6);
        }
        assert!(
            output.as_slice::<f32>()[row * stride + dimension..(row + 1) * stride]
                .iter()
                .all(|v| v.is_nan())
        );
    }
    assert!(
        output.as_slice::<f32>()[rows * stride..]
            .iter()
            .all(|v| v.is_nan())
    );

    let count = 519;
    let gate_data = (0..count).map(|i| signal(i, 4) * 5.0).collect::<Vec<_>>();
    let up_data = (0..count).map(|i| signal(i, 8)).collect::<Vec<_>>();
    let residual_data = (0..count).map(|i| signal(i, 11)).collect::<Vec<_>>();
    let gate = floats(&ctx, &gate_data);
    let up = floats(&ctx, &up_data);
    let residual = guarded(&ctx, count);
    residual.copy_from_bytes(bytemuck::cast_slice(&residual_data), 0);
    let activated = guarded(&ctx, count);
    let sigmoid = guarded(&ctx, count);
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.swiglu(&mut batch, &gate, &up, &activated, count as u32)
        .expect("SwiGLU");
    ops.residual_add(&mut batch, &activated, &residual, &residual, count as u32)
        .expect("residual");
    ops.sigmoid_mul(&mut batch, &up, &gate, &sigmoid, count as u32)
        .expect("sigmoid gate");
    batch.commit_and_wait().expect("elementwise completion");
    let expected = gate_data
        .iter()
        .zip(&up_data)
        .map(|(&g, &u)| f64::from(g) / (1.0 + (-f64::from(g)).exp()) * f64::from(u))
        .collect::<Vec<_>>();
    check(&activated, &expected, 2e-6);
    check(
        &residual,
        &expected
            .iter()
            .zip(residual_data)
            .map(|(a, r)| a + f64::from(r))
            .collect::<Vec<_>>(),
        2e-6,
    );
    check(
        &sigmoid,
        &gate_data
            .iter()
            .zip(&up_data)
            .map(|(&g, &u)| f64::from(u) / (1.0 + (-f64::from(g)).exp()))
            .collect::<Vec<_>>(),
        2e-6,
    );
}

#[test]
fn bf16_matvec_checks_every_row_and_partial_simd_width() {
    let Some((ctx, ops)) = setup() else { return };
    let (rows, columns) = (37, 519);
    let mut packed = vec![0xffff_u16; 2];
    packed.extend((0..rows * columns).map(|i| half::bf16::from_f32(signal(i, 2) * 10.0).to_bits()));
    let input_data = (0..columns).map(|i| signal(i, 5)).collect::<Vec<_>>();
    let weights = MetalBuffer::from_slice(ctx.device(), &packed).expect("BF16 weights");
    let input = floats(&ctx, &input_data);
    let output = guarded(&ctx, rows);
    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.bf16_matvec(
        &mut batch,
        Bf16Matrix {
            buffer: &weights,
            offset: 4,
            rows: rows as u32,
            columns: columns as u32,
        },
        &input,
        &output,
    )
    .expect("BF16 matvec");
    batch.commit_and_wait().expect("matvec completion");
    let expected = (0..rows)
        .map(|r| {
            (0..columns)
                .map(|c| {
                    f64::from(f32::from_bits(u32::from(packed[2 + r * columns + c]) << 16))
                        * f64::from(input_data[c])
                })
                .sum()
        })
        .collect::<Vec<_>>();
    check(&output, &expected, 3e-5);
}

/// Multi-row BF16 projections (verify, batched decode and prefill blocks)
/// must agree with the F64 product and be bitwise the single-row matvec of
/// each row, for one matrix and for the alpha/beta pair, on partial row
/// groups (1 to 41 rows against four per SIMD group), an offset matrix, a
/// column count with a partial 256-column step, and must leave the guard
/// tails untouched.
#[test]
fn bf16_matmul_rows_match_f64_and_are_bitwise_the_matvec() {
    let Some((ctx, ops)) = setup() else { return };
    let (rows, columns) = (37_usize, 320_usize);
    let matrix_data = |salt: usize| {
        let mut packed = vec![0xffff_u16; 2];
        packed.extend(
            (0..rows * columns).map(|i| half::bf16::from_f32(signal(i, salt) * 10.0).to_bits()),
        );
        packed
    };
    let packed = [matrix_data(3), matrix_data(5)];
    let weights = packed
        .iter()
        .map(|data| MetalBuffer::from_slice(ctx.device(), data).expect("BF16 weights"))
        .collect::<Vec<_>>();
    let matrix = |index: usize| Bf16Matrix {
        buffer: &weights[index],
        offset: 4,
        rows: rows as u32,
        columns: columns as u32,
    };
    for tokens in [1_usize, 2, 3, 4, 5, 8, 41] {
        let input_data = (0..tokens * columns)
            .map(|i| signal(i, 7 + tokens))
            .collect::<Vec<_>>();
        let input = floats(&ctx, &input_data);
        let single = guarded(&ctx, rows * tokens);
        let pair = [guarded(&ctx, rows * tokens), guarded(&ctx, rows * tokens)];
        let mut rows_by_matvec = [guarded(&ctx, rows * tokens), guarded(&ctx, rows * tokens)];
        let mut batch = CommandBatch::new(&ctx).expect("batch");
        ops.bf16_matmul(&mut batch, matrix(0), &input, &single, tokens as u32)
            .expect("BF16 matmul");
        ops.bf16_matmul_pair(
            &mut batch,
            [(matrix(0), &pair[0]), (matrix(1), &pair[1])],
            &input,
            tokens as u32,
        )
        .expect("BF16 pair");
        batch.commit_and_wait().expect("matmul completion");
        for (index, by_matvec) in rows_by_matvec.iter_mut().enumerate() {
            for token in 0..tokens {
                let row_input = floats(&ctx, &input_data[token * columns..(token + 1) * columns]);
                let row_output = guarded(&ctx, rows);
                let mut batch = CommandBatch::new(&ctx).expect("batch");
                ops.bf16_matvec(&mut batch, matrix(index), &row_input, &row_output)
                    .expect("BF16 matvec");
                batch.commit_and_wait().expect("matvec completion");
                by_matvec.as_mut_slice::<f32>()[token * rows..(token + 1) * rows]
                    .copy_from_slice(&row_output.as_slice::<f32>()[..rows]);
            }
        }
        for (index, packed) in packed.iter().enumerate() {
            let expected = (0..tokens)
                .flat_map(|t| {
                    let input_data = &input_data;
                    (0..rows).map(move |r| {
                        (0..columns)
                            .map(|c| {
                                f64::from(f32::from_bits(
                                    u32::from(packed[2 + r * columns + c]) << 16,
                                )) * f64::from(input_data[t * columns + c])
                            })
                            .sum::<f64>()
                    })
                })
                .collect::<Vec<_>>();
            check(&pair[index], &expected, 3e-5);
            let bits = |buffer: &MetalBuffer| {
                buffer.as_slice::<f32>()[..rows * tokens]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(&pair[index]),
                bits(&rows_by_matvec[index]),
                "{tokens} rows"
            );
            if index == 0 {
                check(&single, &expected, 3e-5);
                assert_eq!(bits(&single), bits(&rows_by_matvec[0]), "{tokens} rows");
            }
        }
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn recurrent_sequence_preserves_all_heads_state_history_and_grouped_output() {
    let Some((ctx, ops)) = setup() else { return };
    let weight_data = (0..10240 * 4)
        .map(|i| signal(i, 3) * 0.31)
        .collect::<Vec<_>>();
    let weights = floats(&ctx, &weight_data);
    let mut history_ref = (0..10240 * 3)
        .map(|i| f64::from(signal(i, 7) * 0.1))
        .collect::<Vec<_>>();
    let mut state_ref = (0..6144 * 128)
        .map(|i| f64::from(signal(i, 8) * 0.02))
        .collect::<Vec<_>>();
    let history = floats(
        &ctx,
        &history_ref.iter().map(|v| *v as f32).collect::<Vec<_>>(),
    );
    let state = floats(
        &ctx,
        &state_ref.iter().map(|v| *v as f32).collect::<Vec<_>>(),
    );
    let convolved = guarded(&ctx, 10240);
    let output = guarded(&ctx, 6144);
    let grouped = guarded(&ctx, 6144);
    let norm_data = (0..128)
        .map(|i| signal(i, 1).mul_add(0.2, 0.7))
        .collect::<Vec<_>>();
    let norm = floats(&ctx, &norm_data);
    let decay_data = (0..48)
        .map(|i| (i as f32).mul_add(0.011, 0.35))
        .collect::<Vec<_>>();
    let beta_data = (0..48)
        .map(|i| ((i % 9) as f32).mul_add(0.061, 0.13))
        .collect::<Vec<_>>();
    let decay = floats(&ctx, &decay_data);
    let beta = floats(&ctx, &beta_data);
    for step in 0..5 {
        if step == 3 {
            state.clear();
            history.clear();
            state_ref.fill(0.0);
            history_ref.fill(0.0);
        }
        let input_data = (0..10240).map(|i| signal(i, step + 13)).collect::<Vec<_>>();
        let gate_data = (0..6144)
            .map(|i| signal(i, step + 27) * 2.0)
            .collect::<Vec<_>>();
        let input = floats(&ctx, &input_data);
        let gate = floats(&ctx, &gate_data);
        let mut qkv = vec![0.0; 10240];
        for c in 0..10240 {
            let sum = f64::mul_add(
                f64::from(input_data[c]),
                f64::from(weight_data[c * 4 + 3]),
                (0..3)
                    .map(|t| history_ref[c * 3 + t] * f64::from(weight_data[c * 4 + t]))
                    .sum::<f64>(),
            );
            qkv[c] = sum / (1.0 + (-sum).exp());
            history_ref[c * 3] = history_ref[c * 3 + 1];
            history_ref[c * 3 + 1] = history_ref[c * 3 + 2];
            history_ref[c * 3 + 2] = f64::from(input_data[c]);
        }
        for head in 0..32 {
            let group = &mut qkv[head * 128..(head + 1) * 128];
            let length = group
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                .max(f64::from(1e-6_f32));
            for value in group {
                *value /= length;
            }
        }
        let mut expected_output = vec![0.0; 6144];
        for head in 0..48 {
            let q = &qkv[(head % 16) * 128..][..128];
            let k = &qkv[2048 + (head % 16) * 128..][..128];
            for value_dimension in 0..128 {
                let row = head * 128 + value_dimension;
                let memory = &mut state_ref[row * 128..][..128];
                for value in &mut *memory {
                    *value *= f64::from(decay_data[head]);
                }
                let prediction = memory.iter().zip(k).map(|(s, k)| s * k).sum::<f64>();
                let correction = (qkv[4096 + row] - prediction) * f64::from(beta_data[head]);
                for (value, k) in memory.iter_mut().zip(k) {
                    *value += k * correction;
                }
                expected_output[row] =
                    memory.iter().zip(q).map(|(s, q)| s * q).sum::<f64>() / 128.0_f64.sqrt();
            }
        }
        let mut expected_grouped = Vec::with_capacity(6144);
        for key_group in 0..16 {
            for replica in 0..3 {
                let head = key_group + 16 * replica;
                let values = &expected_output[head * 128..][..128];
                let rms = (values.iter().map(|v| v * v).sum::<f64>() / 128.0 + f64::from(1e-6_f32))
                    .sqrt();
                for i in 0..128 {
                    let gate = f64::from(gate_data[head * 128 + i]);
                    expected_grouped.push(
                        values[i] / rms * f64::from(norm_data[i]) * gate / (1.0 + (-gate).exp()),
                    );
                }
            }
        }
        let mut batch = CommandBatch::new(&ctx).expect("batch");
        ops.conv_step(&mut batch, &input, &weights, &history, &convolved)
            .expect("conv");
        ops.l2_normalize_qk(&mut batch, &convolved, 1e-6)
            .expect("L2");
        ops.gdn_step(&mut batch, &convolved, &decay, &beta, &state, &output)
            .expect("GDN");
        ops.gdn_postprocess(&mut batch, &output, &gate, &norm, &grouped, 1e-6)
            .expect("postprocess");
        batch.commit_and_wait().expect("recurrence completion");
        check(&convolved, &qkv, 2e-6);
        check(&output, &expected_output, 2e-6);
        check(&state, &state_ref, 3e-6);
        check(&history, &history_ref, 0.0);
        check(&grouped, &expected_grouped, 3e-5);
    }
}

fn normalized_rope(values: &[f32], weights: &[f32], position: u32) -> Vec<f64> {
    let inverse = (values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / 256.0
        + f64::from(1e-6_f32))
    .sqrt()
    .recip();
    let mut result = values
        .iter()
        .zip(weights)
        .map(|(&v, &w)| f64::from(v) * inverse * f64::from(w))
        .collect::<Vec<_>>();
    for pair in 0..32 {
        let angle = f64::from(position) / 10_000_000.0_f64.powf(pair as f64 / 32.0);
        let (first, second) = (result[pair], result[pair + 32]);
        result[pair] = f64::mul_add(second, -angle.sin(), first * angle.cos());
        result[pair + 32] = f64::mul_add(second, angle.cos(), first * angle.sin());
    }
    result
}

#[test]
fn attention_preparation_splits_every_head_and_rotates_only_first64() {
    let Some((ctx, ops)) = setup() else { return };
    let qg_data = (0..12288).map(|i| signal(i, 5)).collect::<Vec<_>>();
    let k_data = (0..1024).map(|i| signal(i, 11)).collect::<Vec<_>>();
    let v_data = (0..1024).map(|i| signal(i, 31)).collect::<Vec<_>>();
    let query_norm_data = (0..256).map(|i| signal(i, 19)).collect::<Vec<_>>();
    let key_norm_data = (0..256).map(|i| signal(i, 23)).collect::<Vec<_>>();
    let qg = floats(&ctx, &qg_data);
    let key = floats(&ctx, &k_data);
    let value = floats(&ctx, &v_data);
    let qw = floats(&ctx, &query_norm_data);
    let kw = floats(&ctx, &key_norm_data);
    let query = guarded(&ctx, 6144);
    let gate = guarded(&ctx, 6144);
    for position in [0, 1, 131] {
        let capacity = position + 2;
        let nan_cache = vec![half::f16::NAN; capacity as usize * 1024 + 8];
        let kc = MetalBuffer::from_slice(ctx.device(), &nan_cache).expect("K cache");
        let vc = MetalBuffer::from_slice(ctx.device(), &nan_cache).expect("V cache");
        let mut batch = CommandBatch::new(&ctx).expect("batch");
        ops.prepare_attention(
            &mut batch,
            &qg,
            &key,
            &value,
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
        )
        .expect("prepare attention");
        batch.commit_and_wait().expect("prepare completion");
        let expected_query = (0..24)
            .flat_map(|h| normalized_rope(&qg_data[h * 512..][..256], &query_norm_data, position))
            .collect::<Vec<_>>();
        let expected_gate = (0..24)
            .flat_map(|h| {
                qg_data[h * 512 + 256..][..256]
                    .iter()
                    .copied()
                    .map(f64::from)
            })
            .collect::<Vec<_>>();
        check(&query, &expected_query, 3e-5);
        check(&gate, &expected_gate, 0.0);
        let expected_key = (0..4)
            .flat_map(|h| normalized_rope(&k_data[h * 256..][..256], &key_norm_data, position))
            .collect::<Vec<_>>();
        for (i, (&actual_k, &actual_v)) in kc
            .as_slice::<half::f16>()
            .iter()
            .zip(vc.as_slice::<half::f16>())
            .enumerate()
        {
            if (position as usize * 1024..(position as usize + 1) * 1024).contains(&i) {
                let coordinate = i - position as usize * 1024;
                assert!((f64::from(actual_k.to_f32()) - expected_key[coordinate]).abs() < 0.002);
                assert_eq!(actual_v, half::f16::from_f32(v_data[coordinate]));
            } else {
                assert!(actual_k.is_nan() && actual_v.is_nan(), "cache guard {i}");
            }
        }
    }
}

/// K/V-only preparation must write exactly the full preparation's cache rows.
///
/// Bit for bit and nothing else: a multi-token block at a nonzero position (so
/// `RoPE` angles differ per row), asymmetric rows, guarded neighbours, and
/// rejected alias/overflow/short-input calls.
#[test]
#[allow(clippy::too_many_lines)]
fn kv_preparation_matches_full_preparation_rows_bitwise_and_touches_nothing_else() {
    let Some((ctx, ops)) = setup() else { return };
    let tokens = 3u32;
    let position = 129u32;
    let capacity = position + tokens + 2;
    let qg_data = (0..12288 * tokens as usize)
        .map(|i| signal(i, 5))
        .collect::<Vec<_>>();
    let k_data = (0..1024 * tokens as usize)
        .map(|i| signal(i, 11) * (1.0 + (i / 1024) as f32))
        .collect::<Vec<_>>();
    let v_data = (0..1024 * tokens as usize)
        .map(|i| signal(i, 31) - (i / 1024) as f32)
        .collect::<Vec<_>>();
    let query_norm_data = (0..256).map(|i| signal(i, 19)).collect::<Vec<_>>();
    let key_norm_data = (0..256).map(|i| signal(i, 23)).collect::<Vec<_>>();
    let qg = floats(&ctx, &qg_data);
    let key = floats(&ctx, &k_data);
    let value = floats(&ctx, &v_data);
    let qw = floats(&ctx, &query_norm_data);
    let kw = floats(&ctx, &key_norm_data);
    let query = guarded(&ctx, 6144 * tokens as usize);
    let gate = guarded(&ctx, 6144 * tokens as usize);
    let nan_cache = vec![half::f16::NAN; capacity as usize * 1024 + 8];
    let cache = || MetalBuffer::from_slice(ctx.device(), &nan_cache).expect("cache");
    let (full_k, full_v, only_k, only_v) = (cache(), cache(), cache(), cache());

    let mut batch = CommandBatch::new(&ctx).expect("batch");
    ops.prepare_attention_rows(
        &mut batch,
        &qg,
        &key,
        &value,
        &qw,
        &kw,
        &query,
        &gate,
        &full_k,
        &full_v,
        position,
        capacity,
        1e-6,
        10_000_000.0,
        tokens,
    )
    .expect("full preparation");
    ops.prepare_kv_rows(
        &mut batch,
        &key,
        &value,
        &kw,
        &only_k,
        &only_v,
        position,
        capacity,
        1e-6,
        10_000_000.0,
        tokens,
    )
    .expect("kv preparation");
    batch.commit_and_wait().expect("preparation completion");

    let written = position as usize * 1024..(position + tokens) as usize * 1024;
    let expected_key = (0..tokens as usize)
        .flat_map(|token| {
            let (key_rows, key_norm) = (&k_data, &key_norm_data);
            (0..4).flat_map(move |h| {
                normalized_rope(
                    &key_rows[token * 1024 + h * 256..][..256],
                    key_norm,
                    position + token as u32,
                )
            })
        })
        .collect::<Vec<_>>();
    let only_k = only_k.as_slice::<half::f16>();
    let only_v = only_v.as_slice::<half::f16>();
    let full_k = full_k.as_slice::<half::f16>();
    let full_v = full_v.as_slice::<half::f16>();
    assert_eq!(only_k.len(), nan_cache.len());
    for i in 0..nan_cache.len() {
        if written.contains(&i) {
            let coordinate = i - written.start;
            assert_eq!(only_k[i].to_bits(), full_k[i].to_bits(), "key {i}");
            assert_eq!(only_v[i].to_bits(), full_v[i].to_bits(), "value {i}");
            assert!(
                (f64::from(only_k[i].to_f32()) - expected_key[coordinate]).abs() < 0.004,
                "key {i}: {} != {}",
                only_k[i],
                expected_key[coordinate]
            );
            assert_eq!(only_v[i], half::f16::from_f32(v_data[coordinate]));
        } else {
            assert!(only_k[i].is_nan() && only_v[i].is_nan(), "cache guard {i}");
        }
    }
    // Rows really differ across the block, so a kernel that ignored the token
    // index or the position offset would have been caught above.
    assert_ne!(
        only_k[written.start..][..1024]
            .iter()
            .copied()
            .map(half::f16::to_bits)
            .collect::<Vec<_>>(),
        only_k[written.start + 1024..][..1024]
            .iter()
            .copied()
            .map(half::f16::to_bits)
            .collect::<Vec<_>>()
    );

    let mut batch = CommandBatch::new(&ctx).expect("batch");
    let same_cache = cache();
    assert!(
        ops.prepare_kv_rows(
            &mut batch,
            &key,
            &value,
            &kw,
            &same_cache,
            &same_cache,
            position,
            capacity,
            1e-6,
            10_000_000.0,
            tokens,
        )
        .is_err(),
        "aliased K/V caches must be rejected"
    );
    assert!(
        ops.prepare_kv_rows(
            &mut batch,
            &key,
            &value,
            &kw,
            &same_cache,
            &cache(),
            capacity - tokens + 1,
            capacity,
            1e-6,
            10_000_000.0,
            tokens,
        )
        .is_err(),
        "a block ending past capacity must be rejected"
    );
    assert!(
        ops.prepare_kv_rows(
            &mut batch,
            &floats(&ctx, &k_data[..1024 * (tokens as usize - 1)]),
            &value,
            &kw,
            &same_cache,
            &cache(),
            position,
            capacity,
            1e-6,
            10_000_000.0,
            tokens,
        )
        .is_err(),
        "a short key block must be rejected"
    );
}

#[test]
fn split_gqa_matches_f64_across_boundaries_and_shorter_workspace_reuse() {
    let Some((ctx, ops)) = setup() else { return };
    let capacity = 4105;
    let workspace = AttentionWorkspace::new(&ctx, capacity).expect("workspace");
    let query_data = (0..6144).map(|i| signal(i, 19) * 1.7).collect::<Vec<_>>();
    let gate_data = (0..6144).map(|i| signal(i, 23) * 3.0).collect::<Vec<_>>();
    let keys = (0..capacity as usize * 1024)
        .map(|i| half::f16::from_f32(signal(i, 11) * 2.0))
        .collect::<Vec<_>>();
    let values = (0..capacity as usize * 1024)
        .map(|i| half::f16::from_f32(signal(i, 7) + (i / 1024) as f32 / 3000.0))
        .collect::<Vec<_>>();
    let query = floats(&ctx, &query_data);
    let gate = floats(&ctx, &gate_data);
    let key = MetalBuffer::from_slice(ctx.device(), &keys).expect("keys");
    let value = MetalBuffer::from_slice(ctx.device(), &values).expect("values");
    let output = guarded(&ctx, 6144);
    // 1,100 and 4,000 take 64-token tensor splits on tensor builds: more
    // partial records than a 4,105-token workspace has 128-token splits.
    for prefix in [1, 2, 127, 128, 129, 257, 1100, 4000, 4097, 7] {
        let mut expected = vec![0.0; 6144];
        for head in 0..24 {
            let mut scores = (0..prefix as usize)
                .map(|t| {
                    (0..256)
                        .map(|i| {
                            f64::from(query_data[head * 256 + i])
                                * f64::from(keys[(t * 4 + head / 6) * 256 + i].to_f32())
                        })
                        .sum::<f64>()
                        / 16.0
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores {
                *score = (*score - maximum).exp();
            }
            let denominator = scores.iter().sum::<f64>();
            for i in 0..256 {
                expected[head * 256 + i] = scores
                    .iter()
                    .enumerate()
                    .map(|(t, p)| p * f64::from(values[(t * 4 + head / 6) * 256 + i].to_f32()))
                    .sum::<f64>()
                    / denominator;
            }
        }
        for gated in [false, true] {
            let mut batch = CommandBatch::new(&ctx).expect("batch");
            ops.attention(
                &mut batch,
                &query,
                &key,
                &value,
                gated.then_some(&gate),
                &output,
                prefix,
                &workspace,
            )
            .expect("attention");
            batch.commit_and_wait().expect("attention completion");
            let reference = expected
                .iter()
                .zip(&gate_data)
                .map(|(&v, &g)| {
                    if gated {
                        v / (1.0 + (-f64::from(g)).exp())
                    } else {
                        v
                    }
                })
                .collect::<Vec<_>>();
            check(&output, &reference, 2e-5);
        }
    }
}

/// Orthonormal 256-point Walsh-Hadamard transform of every 256-value row, the
/// basis quantized caches are stored in (`bonsai_hadamard256`). Symmetric and
/// self-inverse.
fn hadamard_rows(values: &[f64]) -> Vec<f64> {
    let mut out = values.to_vec();
    for row in out.chunks_mut(256) {
        let mut distance = 1;
        while distance < 256 {
            for i in 0..256 {
                if i & distance == 0 {
                    let (a, b) = (row[i], row[i | distance]);
                    row[i] = a + b;
                    row[i | distance] = a - b;
                }
            }
            distance <<= 1;
        }
        for value in row.iter_mut() {
            *value /= 16.0;
        }
    }
    out
}

/// Independent model of the quantized cache row layout (`bonsai_kv_store`):
/// per KV head, eight 32-value blocks; Q8 keeps one int8 per value then 32 F16
/// scales after byte 1024.
fn quantize_row(format: KvFormat, values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len(), 1024);
    let mut row = vec![0u8; format.token_bytes()];
    match format {
        KvFormat::F16 => {
            for (index, &value) in values.iter().enumerate() {
                row[index * 2..][..2].copy_from_slice(&half::f16::from_f32(value).to_le_bytes());
            }
        }
        KvFormat::Q8 => {
            let (levels, code_bytes) = (127.0f32, 1024);
            for block in 0..32 {
                let chunk = &values[block * 32..][..32];
                let peak = chunk.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                let scale = half::f16::from_f32(peak / levels);
                let inverse = if scale > half::f16::ZERO {
                    1.0 / scale.to_f32()
                } else {
                    0.0
                };
                row[code_bytes + block * 2..][..2].copy_from_slice(&scale.to_le_bytes());
                for (lane, &value) in chunk.iter().enumerate() {
                    let quantized = (value * inverse).round_ties_even();
                    row[block * 32 + lane] = quantized.clamp(-127.0, 127.0) as i8 as u8;
                }
            }
        }
    }
    row
}

fn dequantize_row(format: KvFormat, row: &[u8]) -> Vec<f32> {
    assert_eq!(row.len(), format.token_bytes());
    let scale = |offset: usize, block: usize| {
        half::f16::from_le_bytes([row[offset + block * 2], row[offset + block * 2 + 1]]).to_f32()
    };
    (0..1024)
        .map(|index| match format {
            KvFormat::F16 => {
                half::f16::from_le_bytes([row[index * 2], row[index * 2 + 1]]).to_f32()
            }
            KvFormat::Q8 => f32::from(row[index].cast_signed()) * scale(1024, index / 32),
        })
        .collect()
}

#[test]
#[allow(clippy::too_many_lines)]
fn quantized_kv_rows_written_by_gpu_decode_within_half_scale_and_touch_nothing_else() {
    let Some((ctx, ops)) = setup() else { return };
    let tokens = 3u32;
    let position = 129u32;
    let capacity = position + tokens + 2;
    let k_data = (0..1024 * tokens as usize)
        .map(|i| signal(i, 11) * (1.0 + (i / 1024) as f32))
        .collect::<Vec<_>>();
    let v_data = (0..1024 * tokens as usize)
        .map(|i| signal(i, 31) - (i / 1024) as f32)
        .collect::<Vec<_>>();
    let key_norm_data = (0..256).map(|i| signal(i, 23)).collect::<Vec<_>>();
    let qg = floats(
        &ctx,
        &(0..12288 * tokens as usize)
            .map(|i| signal(i, 5))
            .collect::<Vec<_>>(),
    );
    let qw = floats(&ctx, &(0..256).map(|i| signal(i, 19)).collect::<Vec<_>>());
    let key = floats(&ctx, &k_data);
    let value = floats(&ctx, &v_data);
    let kw = floats(&ctx, &key_norm_data);
    let query = guarded(&ctx, 6144 * tokens as usize);
    let gate = guarded(&ctx, 6144 * tokens as usize);
    let expected_key = (0..tokens as usize)
        .flat_map(|token| {
            let (key_rows, key_norm) = (&k_data, &key_norm_data);
            (0..4).flat_map(move |h| {
                normalized_rope(
                    &key_rows[token * 1024 + h * 256..][..256],
                    key_norm,
                    position + token as u32,
                )
            })
        })
        .collect::<Vec<_>>();
    // Quantized caches store Hadamard-rotated rows.
    let expected_key = hadamard_rows(&expected_key);
    let expected_value = hadamard_rows(&v_data.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
    for layout in ["q8"].map(|name| KvLayout::parse(name).expect("layout")) {
        let cache = |format: KvFormat| {
            MetalBuffer::from_slice(
                ctx.device(),
                &vec![0xFFu8; capacity as usize * format.token_bytes() + 16],
            )
            .expect("cache")
        };
        let (full_k, full_v, only_k, only_v) = (
            cache(layout.key),
            cache(layout.value),
            cache(layout.key),
            cache(layout.value),
        );
        let mut batch = CommandBatch::new(&ctx).expect("batch");
        ops.prepare_attention_rows_kv(
            layout,
            &mut batch,
            &qg,
            &key,
            &value,
            &qw,
            &kw,
            &query,
            &gate,
            &full_k,
            &full_v,
            position,
            capacity,
            1e-6,
            10_000_000.0,
            tokens,
        )
        .expect("full preparation");
        ops.prepare_kv_rows_kv(
            layout,
            &mut batch,
            &key,
            &value,
            &kw,
            &only_k,
            &only_v,
            position,
            capacity,
            1e-6,
            10_000_000.0,
            tokens,
        )
        .expect("kv preparation");
        batch.commit_and_wait().expect("preparation completion");
        assert_eq!(
            only_k.as_slice::<u8>(),
            full_k.as_slice::<u8>(),
            "{layout:?} keys"
        );
        assert_eq!(
            only_v.as_slice::<u8>(),
            full_v.as_slice::<u8>(),
            "{layout:?} values"
        );

        for (format, cache, expected, label) in [
            (layout.key, &only_k, &expected_key, "key"),
            (layout.value, &only_v, &expected_value, "value"),
        ] {
            let bytes = cache.as_slice::<u8>();
            let row_bytes = format.token_bytes();
            let written = position as usize * row_bytes..(position + tokens) as usize * row_bytes;
            assert!(
                bytes[..written.start]
                    .iter()
                    .chain(&bytes[written.end..])
                    .all(|&b| b == 0xFF),
                "{layout:?} {label} guard bytes"
            );
            // Largest level; every format rounds to within half a scale step.
            let (levels, half_step) = match format {
                KvFormat::F16 | KvFormat::Q8 => (127.0, 0.5),
            };
            for token in 0..tokens as usize {
                let row = &bytes[(position as usize + token) * row_bytes..][..row_bytes];
                let decoded = dequantize_row(format, row);
                let wanted = &expected[token * 1024..][..1024];
                for block in 0..32 {
                    let peak = wanted[block * 32..][..32]
                        .iter()
                        .fold(0.0f64, |acc, v| acc.max(v.abs()));
                    let scale = f64::from(half::f16::from_f64(peak / levels).to_f32());
                    assert!(
                        scale > 0.0,
                        "{layout:?} {label} token {token} block {block} scale"
                    );
                    for lane in 0..32 {
                        let index = block * 32 + lane;
                        let error = (f64::from(decoded[index]) - wanted[index]).abs();
                        // Half a step of rounding, plus the F32 prep kernel's own
                        // error (the F16 test allows 0.004) which can shift the
                        // scale by an F16 ulp.
                        let bound = scale.mul_add(2e-3f64.mul_add(levels, half_step), 0.004);
                        assert!(
                            error <= bound,
                            "{layout:?} {label} token {token} element {index}: {} vs {} (bound {bound})",
                            decoded[index],
                            wanted[index]
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn quantized_attention_matches_f64_over_dequantized_cache_for_rows_and_blocks() {
    let Some((ctx, ops)) = setup() else { return };
    let capacity = 1100u32;
    let workspace = AttentionWorkspace::new(&ctx, capacity).expect("workspace");
    let query_rows = 9usize;
    let query_data = (0..6144 * query_rows)
        .map(|i| signal(i, 19) * 1.7)
        .collect::<Vec<_>>();
    let gate_data = (0..6144 * query_rows)
        .map(|i| signal(i, 23) * 3.0)
        .collect::<Vec<_>>();
    let keys = (0..capacity as usize * 1024)
        .map(|i| signal(i, 11) * 2.0)
        .collect::<Vec<_>>();
    let values = (0..capacity as usize * 1024)
        .map(|i| signal(i, 7) + (i / 1024) as f32 / 3000.0)
        .collect::<Vec<_>>();
    let query = floats(&ctx, &query_data);
    let gate = floats(&ctx, &gate_data);
    let mut output = guarded(&ctx, 6144 * query_rows);
    let reference_raw = |row: usize, prefix: usize, k: &[f32], v: &[f32], gated: bool| {
        let mut expected = vec![0.0; 6144];
        for head in 0..24 {
            let mut scores = (0..prefix)
                .map(|t| {
                    (0..256)
                        .map(|i| {
                            f64::from(query_data[row * 6144 + head * 256 + i])
                                * f64::from(k[(t * 4 + head / 6) * 256 + i])
                        })
                        .sum::<f64>()
                        / 16.0
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores {
                *score = (*score - maximum).exp();
            }
            let denominator = scores.iter().sum::<f64>();
            for i in 0..256 {
                let index = head * 256 + i;
                let mixed = scores
                    .iter()
                    .enumerate()
                    .map(|(t, p)| p * f64::from(v[(t * 4 + head / 6) * 256 + i]))
                    .sum::<f64>()
                    / denominator;
                expected[index] = if gated {
                    mixed / (1.0 + (-f64::from(gate_data[row * 6144 + index])).exp())
                } else {
                    mixed
                };
            }
        }
        expected
    };
    // A quantized cache is taken to hold rotated rows: its kernels attend in
    // that basis and `bo_attn_unrotate` rotates each head's output back before
    // gating. The F16 path gates directly.
    let reference_in =
        |row: usize, prefix: usize, k: &[f32], v: &[f32], gated: bool, rotated: bool| {
            let mut expected = reference_raw(row, prefix, k, v, false);
            if rotated {
                expected = hadamard_rows(&expected);
            }
            if gated {
                for (index, value) in expected.iter_mut().enumerate() {
                    *value /= 1.0 + (-f64::from(gate_data[row * 6144 + index])).exp();
                }
            }
            expected
        };
    for layout in ["q8", "f16"].map(|name| KvLayout::parse(name).expect("layout")) {
        let encode = |format: KvFormat, data: &[f32]| {
            let bytes = data
                .chunks(1024)
                .flat_map(|row| quantize_row(format, row))
                .collect::<Vec<_>>();
            let decoded = bytes
                .chunks(format.token_bytes())
                .flat_map(|row| dequantize_row(format, row))
                .collect::<Vec<_>>();
            (
                MetalBuffer::from_slice(ctx.device(), &bytes).expect("cache"),
                decoded,
            )
        };
        let (k_cache, k_seen) = encode(layout.key, &keys);
        let (v_cache, v_seen) = encode(layout.value, &values);
        // Quantization must actually change what the kernel reads, or this
        // test could pass on a kernel that ignored the format.
        if layout != KvLayout::F16 {
            assert!(
                k_seen.iter().zip(&keys).any(|(a, b)| (a - b).abs() > 1e-6),
                "{layout:?} keys rounded"
            );
        }
        // Every attention path dequantizes a quantized value and rounds it once
        // to half (the tensor kernels' operand type, matched by the SIMD
        // reader). That rounding is the kernels' contract, so the reference
        // applies it too rather than the bound being loosened.
        let half_operands = layout != KvLayout::F16;
        let rounded = |data: &[f32]| {
            data.iter()
                .map(|&value| half::f16::from_f32(value).to_f32())
                .collect::<Vec<_>>()
        };
        let (k_half, v_half) = (rounded(&k_seen), rounded(&v_seen));
        for prefix in [1u32, 129, 257, 300, 1100] {
            let (k_read, v_read) = if half_operands {
                (&k_half, &v_half)
            } else {
                (&k_seen, &v_seen)
            };
            for gated in [false, true] {
                let mut batch = CommandBatch::new(&ctx).expect("batch");
                ops.attention_row_kv(
                    layout,
                    &mut batch,
                    &query,
                    &k_cache,
                    &v_cache,
                    gated.then_some(&gate),
                    &output,
                    prefix,
                    &workspace,
                    0,
                )
                .expect("attention row");
                batch.commit_and_wait().expect("attention completion");
                let expected = reference_in(
                    0,
                    prefix as usize,
                    k_read,
                    v_read,
                    gated,
                    layout != KvLayout::F16,
                );
                for (index, (&actual, &wanted)) in output.as_slice::<f32>()[..6144]
                    .iter()
                    .zip(&expected)
                    .enumerate()
                {
                    assert!(
                        (f64::from(actual) - wanted).abs() <= 2e-5 * (1.0 + wanted.abs()),
                        "{layout:?} prefix {prefix} gated {gated} element {index}: {actual} != {wanted}"
                    );
                }
            }
        }
        // A nine-row block whose rows straddle the 256-key tile boundary and
        // a two-row verification-sized block at a short prefix (block
        // kernels), then the same two rows past the 1,024-token routing
        // threshold, which every layout attends row by row through the
        // split kernel.
        for (position, rows) in [(250u32, query_rows), (255, 2), (1023, 2)] {
            output.as_mut_slice::<f32>().fill(f32::NAN);
            let mut batch = CommandBatch::new(&ctx).expect("batch");
            ops.attention_block_kv(
                layout,
                &mut batch,
                &query,
                &k_cache,
                &v_cache,
                Some(&gate),
                &output,
                position,
                rows as u32,
                &workspace,
            )
            .expect("attention block");
            batch.commit_and_wait().expect("block completion");
            let out = output.as_slice::<f32>();
            for row in 0..rows {
                let prefix = position as usize + row + 1;
                let (k_read, v_read) = if half_operands {
                    (&k_half, &v_half)
                } else {
                    (&k_seen, &v_seen)
                };
                let expected =
                    reference_in(row, prefix, k_read, v_read, true, layout != KvLayout::F16);
                for (index, (&actual, &wanted)) in
                    out[row * 6144..][..6144].iter().zip(&expected).enumerate()
                {
                    assert!(
                        (f64::from(actual) - wanted).abs() <= 2e-5 * (1.0 + wanted.abs()),
                        "{layout:?} block at {position} row {row} element {index}: {actual} != {wanted}"
                    );
                }
            }
            assert!(
                out[6144 * rows..].iter().all(|v| v.is_nan()),
                "{layout:?} block at {position}: tail overwritten"
            );
        }
    }
}
