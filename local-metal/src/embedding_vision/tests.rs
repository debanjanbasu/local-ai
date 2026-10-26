#![expect(unsafe_code, clippy::cast_precision_loss)]

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

#[test]
fn attention_matches_reference_within_each_segment() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let heads = 3_usize;
    let d = VISION_HEAD_DIM as usize;
    // Two segments with ragged lengths (not multiples of the 32-token tiles).
    let segments = [(0_usize, 45_usize), (45, 70)];
    let tokens = 115;
    let scale = |v: Vec<f32>| v.into_iter().map(|x| x * 3.0).collect::<Vec<_>>();
    let q = scale(values(tokens * heads * d, 1));
    let k = scale(values(tokens * heads * d, 2));
    let v = values(tokens * heads * d, 3);
    let (qb, kb, vb) = (
        buffer(&context, &q),
        buffer(&context, &k),
        buffer(&context, &v),
    );
    let out = MetalBuffer::empty(context.device(), tokens * heads * d * 4).expect("out");
    let mut batch = CommandBatch::new(&context).expect("batch");
    for (first, count) in segments {
        kernels
            .attention(
                &mut batch,
                &qb,
                &kb,
                &vb,
                &out,
                first as u32,
                count as u32,
                heads as u32,
            )
            .expect("attention");
    }
    batch.commit_and_wait().expect("commit");
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    let got = unsafe { out.as_slice::<f32>() };
    for (first, count) in segments {
        for t in first..first + count {
            for h in 0..heads {
                let row =
                    |x: &[f32], j: usize| x[(j * heads + h) * d..(j * heads + h + 1) * d].to_vec();
                let qt = row(&q, t);
                let scores: Vec<f64> = (first..first + count)
                    .map(|j| {
                        qt.iter()
                            .zip(row(&k, j))
                            .map(|(a, b)| f64::from(*a) * f64::from(b))
                            .sum()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for e in 0..d {
                    let expected: f64 = (first..first + count)
                        .zip(&weights)
                        .map(|(j, w)| w * f64::from(v[(j * heads + h) * d + e]))
                        .sum::<f64>()
                        / total;
                    let actual = f64::from(got[(t * heads + h) * d + e]);
                    assert!(
                        (actual - expected).abs() < 1e-4,
                        "t {t} h {h} e {e}: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn positions_add_and_pooling_match_reference() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let (width, table_rows, tokens) = (64_usize, 10_usize, 9_usize);
    let table: Vec<u16> = values(2 * table_rows * width, 5)
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect();
    let hidden = values(tokens * width, 6);
    let positions: Vec<u32> = (0..tokens as u32)
        .flat_map(|i| [i % 3, i / 3 + 4])
        .collect();
    let hb = buffer(&context, &hidden);
    let tb = buffer(&context, &table);
    let pb = buffer(&context, &positions);
    let members: Vec<u32> = (0..9).collect();
    let mb = buffer(&context, &members);
    let pooled = MetalBuffer::empty(context.device(), width * 4).expect("pooled");
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .add_positions(
            &mut batch,
            &hb,
            (&tb, 0),
            table_rows as u32,
            &pb,
            tokens as u32,
            width as u32,
        )
        .expect("positions");
    kernels
        .pool(&mut batch, &hb, &mb, &pooled, 1, width as u32, 2.0)
        .expect("pool");
    batch.commit_and_wait().expect("commit");
    let bf = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let mut expected = hidden;
    for t in 0..tokens {
        let (x, y) = (positions[2 * t] as usize, positions[2 * t + 1] as usize);
        for c in 0..width {
            let e = bf(table[x * width + c]) + bf(table[(table_rows + y) * width + c]);
            expected[t * width + c] += e;
        }
    }
    // Fast-math may reassociate the three-term sum: allow a rounding step.
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    for (a, e) in unsafe { hb.as_slice::<f32>() }.iter().zip(&expected) {
        assert!((a - e).abs() <= 1e-6 * (1.0 + e.abs()), "{a} vs {e}");
    }
    for c in 0..width {
        let mean: f32 = (0..tokens)
            .map(|t| expected[t * width + c] * (1.0 / 9.0))
            .sum::<f32>()
            * 2.0;
        // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
        assert!((unsafe { pooled.as_slice::<f32>() }[c] - mean).abs() < 1e-5);
    }
}
