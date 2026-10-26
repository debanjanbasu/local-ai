#![expect(
    unsafe_code,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]

use super::*;

fn gpu_or_skip() -> Option<(MetalContext, EmbeddingKernels)> {
    let context = MetalContext::new();
    if matches!(context, Err(crate::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    let context = context.expect("Metal context failed for a reason other than a missing device");
    let shaders = ShaderLibrary::new(context.device()).expect("shader library");
    let kernels = EmbeddingKernels::new(&context, &shaders).expect("embedding kernels");
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

fn buffer(context: &MetalContext, data: &[f32]) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), data).expect("buffer")
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: {a} vs {e}"
        );
    }
}

#[test]
fn bf16_gemm_matches_reference_on_ragged_tiles() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let (rows, columns, tokens) = (70_usize, 96_usize, 37_usize);
    // BF16 bit patterns of the test values (exact: low mantissa bits dropped).
    let weights: Vec<u16> = values(rows * columns, 1)
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect();
    let input = values(tokens * columns, 3);
    let packed = MetalBuffer::from_slice(context.device(), &weights).expect("weights");
    let x = buffer(&context, &input);
    let y = MetalBuffer::empty(context.device(), tokens * rows * 4).expect("output");
    let mut batch = CommandBatch::new(&context).expect("batch");
    let matrix = Bf16Rows {
        weights: &packed,
        offset: 0,
        rows: rows as u32,
        columns: columns as u32,
    };
    kernels
        .bf16_gemm(&mut batch, matrix, &x, &y, tokens as u32)
        .expect("gemm");
    batch.commit_and_wait().expect("commit");
    let mut expected = vec![0.0_f32; tokens * rows];
    for t in 0..tokens {
        for r in 0..rows {
            expected[t * rows + r] = (0..columns)
                .map(|c| {
                    f32::from_bits(u32::from(weights[r * columns + c]) << 16)
                        * input[t * columns + c]
                })
                .sum();
        }
    }
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    close(
        &unsafe { y.as_slice::<f32>() }[..tokens * rows],
        &expected,
        1e-4,
    );
}

#[test]
fn rms_rope_and_gelu_match_reference() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let (rows, dim) = (6_usize, 256_usize);
    let input = values(rows * dim, 4);
    let weight = values(dim, 5);
    let x = buffer(&context, &input);
    let w = buffer(&context, &weight);
    let normed = MetalBuffer::empty(context.device(), rows * dim * 4).expect("normed");
    let positions =
        MetalBuffer::from_slice(context.device(), &[0_u32, 1, 2, 7, 0, 300]).expect("positions");
    let gated = MetalBuffer::empty(context.device(), rows * dim * 4).expect("gated");
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .rms(
            &mut batch,
            &x,
            Some((&w, 0)),
            &normed,
            rows as u32,
            dim as u32,
            0.5,
            1e-6,
        )
        .expect("rms");
    kernels
        .rope(
            &mut batch,
            &normed,
            &positions,
            rows as u32,
            1,
            dim as u32,
            10_000.0,
        )
        .expect("rope");
    kernels
        .gelu_mul(
            &mut batch,
            &normed,
            &x,
            &gated,
            rows as u32,
            dim as u32,
            dim as u32,
            0,
        )
        .expect("gelu");
    batch.commit_and_wait().expect("commit");
    let pos = [0_u32, 1, 2, 7, 0, 300];
    let mut expected = vec![0.0_f32; rows * dim];
    for r in 0..rows {
        let row = &input[r * dim..(r + 1) * dim];
        let mean = row.iter().map(|v| (v * 0.5) * (v * 0.5)).sum::<f32>() / dim as f32;
        let inv = (mean + 1e-6).powf(-0.5);
        let n: Vec<f32> = row
            .iter()
            .zip(&weight)
            .map(|(v, w)| v * 0.5 * inv * w)
            .collect();
        let half = dim / 2;
        for i in 0..half {
            let freq = 1.0 / 10_000_f32.powf((2 * i) as f32 / dim as f32);
            let angle = pos[r] as f32 * freq;
            let (s, c) = angle.sin_cos();
            expected[r * dim + i] = n[i] * c - n[i + half] * s;
            expected[r * dim + i + half] = n[i + half] * c + n[i] * s;
        }
    }
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    close(
        &unsafe { normed.as_slice::<f32>() }[..rows * dim],
        &expected,
        2e-4,
    );
    let gelu: Vec<f32> = expected
        .iter()
        .zip(&input)
        .map(|(g, o)| {
            let inner = 0.797_884_6 * (g + 0.044_715 * g * g * g);
            0.5 * g * (1.0 + inner.tanh()) * o
        })
        .collect();
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    close(
        &unsafe { gated.as_slice::<f32>() }[..rows * dim],
        &gelu,
        2e-4,
    );
}

#[test]
fn packed_attention_respects_sequence_bounds_and_window() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    // Two sequences (5 and 3 tokens), 4 query heads over 2 KV heads, window 2.
    let (tokens, q_heads, kv_heads, dim, window) = (8_usize, 4_usize, 2_usize, 256_usize, 2_usize);
    let q = values(tokens * q_heads * dim, 6);
    let k = values(tokens * kv_heads * dim, 7);
    let v = values(tokens * kv_heads * dim, 8);
    let bounds: Vec<u32> = (0..tokens)
        .flat_map(|t| if t < 5 { [0, 5] } else { [5, 8] })
        .collect();
    let (qb, kb, vb) = (
        buffer(&context, &q),
        buffer(&context, &k),
        buffer(&context, &v),
    );
    let bb = MetalBuffer::from_slice(context.device(), &bounds).expect("bounds");
    let out = MetalBuffer::empty(context.device(), q.len() * 4).expect("out");
    let mut batch = CommandBatch::new(&context).expect("batch");
    let shape = AttentionShape {
        tokens: tokens as u32,
        q_heads: q_heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: dim as u32,
        window: window as u32,
    };
    kernels
        .attention(&mut batch, shape, &qb, &kb, &vb, &bb, &out)
        .expect("attention");
    batch.commit_and_wait().expect("commit");
    let mut expected = vec![0.0_f32; q.len()];
    for t in 0..tokens {
        let (first, end) = (bounds[2 * t] as usize, bounds[2 * t + 1] as usize);
        for h in 0..q_heads {
            let kvh = h / (q_heads / kv_heads);
            let keys: Vec<usize> = (first..end).filter(|j| t.abs_diff(*j) <= window).collect();
            let scores: Vec<f32> = keys
                .iter()
                .map(|&j| {
                    (0..dim)
                        .map(|d| q[(t * q_heads + h) * dim + d] * k[(j * kv_heads + kvh) * dim + d])
                        .sum()
                })
                .collect();
            let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let total: f32 = weights.iter().sum();
            for d in 0..dim {
                expected[(t * q_heads + h) * dim + d] = keys
                    .iter()
                    .zip(&weights)
                    .map(|(&j, w)| w * v[(j * kv_heads + kvh) * dim + d])
                    .sum::<f32>()
                    / total;
            }
        }
    }
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    close(
        &unsafe { out.as_slice::<f32>() }[..q.len()],
        &expected,
        1e-4,
    );
}
