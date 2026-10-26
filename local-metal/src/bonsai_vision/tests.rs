#![expect(unsafe_code, clippy::cast_precision_loss, clippy::suboptimal_flops)]

use super::*;

fn gpu_or_skip() -> Option<(MetalContext, VisionKernels)> {
    let context = MetalContext::new();
    if matches!(context, Err(crate::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    let context = context.expect("Metal context failed for a reason other than a missing device");
    let shaders = ShaderLibrary::new(context.device()).expect("shader library");
    let kernels = VisionKernels::new(&context, &shaders).expect("vision kernels");
    Some((context, kernels))
}

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 20_000) as f32 / 10_000.0 - 1.0
        })
        .collect()
}

fn buffer<T: bytemuck::NoUninit>(context: &MetalContext, data: &[T]) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), data).expect("buffer")
}

fn bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn gelu_erf(x: f64) -> f64 {
    // 0.5 x erfc(-x / sqrt 2): erf's Taylor series near zero, erfc's
    // continued fraction in the tails (where the series cancels).
    let z = -x / std::f64::consts::SQRT_2;
    let erfc = if z.abs() < 2.0 {
        let mut term = z;
        let mut sum = z;
        for n in 1..60 {
            term *= -z * z / f64::from(n);
            sum += term / f64::from(2 * n + 1);
        }
        1.0 - 2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        let a = z.abs();
        let mut fraction = a;
        for n in (1..200).rev() {
            fraction = a + f64::from(n) / 2.0 / fraction;
        }
        let tail = (-a * a).exp() / (std::f64::consts::PI.sqrt() * fraction);
        if z > 0.0 { tail } else { 2.0 - tail }
    };
    0.5 * x * erfc
}

fn gelu_tanh(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh())
}

#[test]
fn gemm_handles_ragged_tiles_column_tails_and_every_epilogue() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    // 40 columns: one full 32-column tile and an 8-column tail; 37 rows and
    // 45 tokens: ragged tiles on both output axes.
    let (rows, columns, tokens) = (37_usize, 40_usize, 45_usize);
    let full = values(rows * columns, 1);
    let weights = full.iter().copied().map(bf16).collect::<Vec<_>>();
    let bias = values(rows, 2);
    let input = values(tokens * columns, 3);
    let residual = values(tokens * rows, 4);
    let wb = buffer(&context, &weights);
    let fb = buffer(&context, &full);
    let bb = buffer(&context, &bias);
    let ib = buffer(&context, &input);
    // BF16 weights, and the same values unrounded as F32 weights.
    for (f32_weights, wbuf) in [(false, &wb), (true, &fb)] {
        let linear = Linear {
            f32: f32_weights,
            weights: wbuf,
            weights_offset: 0,
            bias: &bb,
            bias_offset: 0,
            rows: rows as u32,
            columns: columns as u32,
        };
        for epilogue in [
            Epilogue::Store,
            Epilogue::GeluTanh,
            Epilogue::GeluErf,
            Epilogue::Accumulate,
        ] {
            let out = buffer(&context, &residual);
            let mut batch = CommandBatch::new(&context).expect("batch");
            kernels
                .gemm(&mut batch, linear, &ib, &out, tokens as u32, epilogue)
                .expect("gemm");
            batch.commit_and_wait().expect("wait");
            // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
            let got = unsafe { out.as_slice::<f32>() };
            for t in 0..tokens {
                for r in 0..rows {
                    let mut value = f64::from(bias[r]);
                    for c in 0..columns {
                        let weight = if f32_weights {
                            full[r * columns + c]
                        } else {
                            widen(weights[r * columns + c])
                        };
                        value += f64::from(weight) * f64::from(input[t * columns + c]);
                    }
                    let expected = match epilogue {
                        Epilogue::Store => value,
                        Epilogue::GeluTanh => gelu_tanh(value),
                        Epilogue::GeluErf => gelu_erf(value),
                        Epilogue::Accumulate => f64::from(residual[t * rows + r]) + value,
                    };
                    let actual = f64::from(got[t * rows + r]);
                    assert!(
                        (actual - expected).abs() < 2e-5,
                        "{epilogue:?} f32={f32_weights} t={t} r={r}: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn layernorm_matches_reference() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let (tokens, width) = (5_usize, 1152_usize);
    // An offset mean makes a one-pass variance lose precision.
    let input = values(tokens * width, 7)
        .into_iter()
        .map(|v| v * 3.0 + 40.0)
        .collect::<Vec<_>>();
    let weight = values(width, 8);
    let bias = values(width, 9);
    let (ib, wb, bb) = (
        buffer(&context, &input),
        buffer(&context, &weight),
        buffer(&context, &bias),
    );
    let out = MetalBuffer::empty(context.device(), tokens * width * 4).expect("out");
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .layernorm(
            &mut batch,
            &ib,
            (&wb, 0),
            (&bb, 0),
            &out,
            tokens as u32,
            width as u32,
            1e-6,
        )
        .expect("layernorm");
    batch.commit_and_wait().expect("wait");
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    let got = unsafe { out.as_slice::<f32>() };
    for t in 0..tokens {
        let row = &input[t * width..(t + 1) * width];
        let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / width as f64;
        let var = row
            .iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / width as f64;
        for c in 0..width {
            let expected = (f64::from(row[c]) - mean) / (var + 1e-6).sqrt() * f64::from(weight[c])
                + f64::from(bias[c]);
            assert!((f64::from(got[t * width + c]) - expected).abs() < 1e-4);
        }
    }
}

#[test]
fn rope_rotates_q_and_k_halves_only() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let tokens = 3_usize;
    let row = 3 * VISION_WIDTH as usize;
    let qkv = values(tokens * row, 11);
    let angles = values(tokens * 36, 12)
        .into_iter()
        .map(|a| a * 3.0)
        .collect::<Vec<_>>();
    let cosines = angles.iter().map(|a| a.cos()).collect::<Vec<_>>();
    let sines = angles.iter().map(|a| a.sin()).collect::<Vec<_>>();
    let qb = buffer(&context, &qkv);
    let (cb, sb) = (buffer(&context, &cosines), buffer(&context, &sines));
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .rope(&mut batch, &qb, &cb, &sb, tokens as u32)
        .expect("rope");
    batch.commit_and_wait().expect("wait");
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    let got = unsafe { qb.as_slice::<f32>() };
    for t in 0..tokens {
        for part in 0..3 {
            for head in 0..16 {
                let base = t * row + part * 1152 + head * 72;
                for j in 0..36 {
                    let (x1, x2) = (qkv[base + j], qkv[base + j + 36]);
                    let (c, s) = (cosines[t * 36 + j], sines[t * 36 + j]);
                    let (e1, e2) = if part == 2 {
                        (x1, x2)
                    } else {
                        (x1 * c - x2 * s, x2 * c + x1 * s)
                    };
                    assert!((got[base + j] - e1).abs() < 1e-6);
                    assert!((got[base + j + 36] - e2).abs() < 1e-6);
                }
            }
        }
    }
}

#[test]
fn attention_matches_reference_over_ragged_counts() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let width = VISION_WIDTH as usize;
    let d = VISION_HEAD_DIM as usize;
    let scale = 1.0 / (d as f32).sqrt();
    // 77 patches: a full 64-query block and a ragged one, two full 32-key
    // tiles and a ragged one; the second image starts at row offset 77 of the
    // same buffers.
    for (offset_rows, count) in [(0_usize, 77_usize), (77, 7)] {
        let total = offset_rows + count;
        let qkv = values(total * 3 * width, 21)
            .into_iter()
            .map(|v| v * 2.0)
            .collect::<Vec<_>>();
        let qb = buffer(&context, &qkv);
        let out = MetalBuffer::empty(context.device(), total * width * 4).expect("out");
        // The first image's queries split over two dispatches.
        let mut batch = CommandBatch::new(&context).expect("batch");
        let split = (count as u32).min(64);
        for queries in [0..split, split..count as u32] {
            if queries.is_empty() {
                continue;
            }
            kernels
                .attention(
                    &mut batch,
                    &qb,
                    offset_rows * 3 * width * 4,
                    &out,
                    offset_rows * width * 4,
                    count as u32,
                    queries,
                    scale,
                )
                .expect("attention");
        }
        batch.commit_and_wait().expect("wait");
        // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
        let got = unsafe { out.as_slice::<f32>() };
        let at = |t: usize, part: usize, h: usize, j: usize| {
            f64::from(qkv[(offset_rows + t) * 3 * width + part * width + h * d + j])
        };
        for h in 0..16 {
            for i in 0..count {
                let scores = (0..count)
                    .map(|k| (0..d).map(|j| at(i, 0, h, j) * at(k, 1, h, j)).sum::<f64>())
                    .map(|s| s * f64::from(scale))
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                let total_weight = weights.iter().sum::<f64>();
                for j in 0..d {
                    let expected =
                        (0..count).map(|k| weights[k] * at(k, 2, h, j)).sum::<f64>() / total_weight;
                    let actual = f64::from(got[(offset_rows + i) * width + h * d + j]);
                    assert!(
                        (actual - expected).abs() < 1e-4,
                        "h={h} i={i} j={j}: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}
