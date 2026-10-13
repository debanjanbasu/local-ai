#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]

use half::f16;

use super::*;

fn gpu_or_skip() -> Option<(MetalContext, WhisperKernels)> {
    let context = MetalContext::new();
    if matches!(context, Err(crate::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    let context = context.expect("Metal context failed for a reason other than a missing device");
    let shaders = ShaderLibrary::new(context.device()).expect("shader library");
    let kernels = WhisperKernels::new(&context, &shaders).expect("whisper kernels");
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

fn gelu(x: f64) -> f64 {
    0.5 * x * erfc(-x / std::f64::consts::SQRT_2)
}

/// erfc with fractional error below 1.2e-7 (Numerical Recipes), in F64.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = [
        -1.265_512_23,
        1.000_023_68,
        0.374_091_96,
        0.096_784_18,
        -0.186_288_06,
        0.278_868_07,
        -1.135_203_98,
        1.488_515_87,
        -0.822_152_23,
        0.170_872_77,
    ];
    let horner = poly.iter().rev().fold(0.0, |acc, c| acc * t + c);
    let r = t * (-z * z + horner).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

fn f16_matrix(rows: usize, columns: usize, seed: u32) -> (Vec<f16>, Vec<f32>) {
    let halves: Vec<f16> = values(rows * columns, seed)
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let widened = halves.iter().map(|h| h.to_f32()).collect();
    (halves, widened)
}

#[test]
fn gemm_and_gemv_match_reference_with_every_epilogue() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let (rows, columns, tokens) = (70_usize, 128_usize, 37_usize);
    let (halves, w) = f16_matrix(rows, columns, 1);
    let input = values(tokens * columns, 3);
    let bias = values(rows, 5);
    let residual = values(tokens * rows, 7);
    let packed = MetalBuffer::from_slice(context.device(), &halves).expect("weights");
    let x = buffer(&context, &input);
    let b = buffer(&context, &bias);
    let matrix = F16Rows {
        weights: &packed,
        offset: 0,
        rows: rows as u32,
        columns: columns as u32,
    };
    for (use_bias, use_gelu, use_residual) in [
        (false, false, false),
        (true, false, false),
        (true, true, false),
        (true, false, true),
        (false, true, true),
    ] {
        let y = buffer(&context, &residual);
        let mut batch = CommandBatch::new(&context).expect("batch");
        let epilogue = Epilogue {
            bias: use_bias.then_some(Region::whole(&b)),
            gelu: use_gelu,
            residual: use_residual.then_some(Region::whole(&y)),
        };
        kernels
            .gemm(
                &mut batch,
                matrix,
                Region::whole(&x),
                Region::whole(&y),
                tokens as u32,
                &epilogue,
            )
            .expect("gemm");
        // The GEMV of the last token into a separate vector.
        let v = buffer(&context, &residual[(tokens - 1) * rows..]);
        let epilogue_v = Epilogue {
            bias: use_bias.then_some(Region::whole(&b)),
            gelu: use_gelu,
            residual: use_residual.then_some(Region::whole(&v)),
        };
        kernels
            .gemv(
                &mut batch,
                matrix,
                Region::at(&x, (tokens - 1) * columns * 4),
                Region::whole(&v),
                &epilogue_v,
            )
            .expect("gemv");
        batch.commit_and_wait().expect("commit");
        let mut expected = vec![0.0_f32; tokens * rows];
        for t in 0..tokens {
            for r in 0..rows {
                let mut acc: f64 = (0..columns)
                    .map(|c| f64::from(w[r * columns + c]) * f64::from(input[t * columns + c]))
                    .sum();
                if use_bias {
                    acc += f64::from(bias[r]);
                }
                if use_gelu {
                    acc = gelu(acc);
                }
                if use_residual {
                    acc += f64::from(residual[t * rows + r]);
                }
                expected[t * rows + r] = acc as f32;
            }
        }
        close(&y.as_slice::<f32>()[..tokens * rows], &expected, 2e-5);
        close(
            &v.as_slice::<f32>()[..rows],
            &expected[(tokens - 1) * rows..],
            2e-5,
        );
    }
}

#[test]
fn layernorm_and_im2col_match_reference() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let (rows, dim) = (5_usize, 1280_usize);
    let input: Vec<f32> = values(rows * dim, 11)
        .iter()
        .map(|v| v * 3.0 + 0.5)
        .collect();
    let weight = values(dim, 12);
    let bias = values(dim, 13);
    let x = buffer(&context, &input);
    let wb = buffer(&context, &weight);
    let bb = buffer(&context, &bias);
    let y = MetalBuffer::empty(context.device(), rows * dim * 4).expect("y");
    let (t_in, channels) = (9_usize, 4_usize);
    let conv_in = values(t_in * channels, 14);
    let ci = buffer(&context, &conv_in);
    let t_out = 5_usize; // stride 2
    let co = MetalBuffer::empty(context.device(), t_out * channels * 3 * 4).expect("cols");
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .layernorm(
            &mut batch,
            Region::whole(&x),
            Region::whole(&wb),
            Region::whole(&bb),
            Region::whole(&y),
            rows as u32,
            dim as u32,
            1e-5,
        )
        .expect("layernorm");
    kernels
        .im2col3(
            &mut batch,
            Region::whole(&ci),
            Region::whole(&co),
            channels as u32,
            t_in as u32,
            t_out as u32,
            2,
        )
        .expect("im2col");
    batch.commit_and_wait().expect("commit");
    let mut expected = vec![0.0_f32; rows * dim];
    for r in 0..rows {
        let row = &input[r * dim..(r + 1) * dim];
        let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / dim as f64;
        let var = row
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / dim as f64;
        for i in 0..dim {
            expected[r * dim + i] = ((f64::from(row[i]) - mean) / (var + 1e-5).sqrt()
                * f64::from(weight[i])
                + f64::from(bias[i])) as f32;
        }
    }
    close(&y.as_slice::<f32>()[..rows * dim], &expected, 1e-5);
    let cols = &co.as_slice::<f32>()[..t_out * channels * 3];
    for t in 0..t_out {
        for c in 0..channels {
            for k in 0..3 {
                let src = (t * 2 + k).checked_sub(1).filter(|s| *s < t_in);
                let want = src.map_or(0.0, |s| conv_in[s * channels + c]);
                assert_eq!(cols[t * channels * 3 + c * 3 + k], want, "t{t} c{c} k{k}");
            }
        }
    }
}

fn attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    query: usize,
    keys: usize,
) -> Vec<f32> {
    let width = heads * 64;
    let mut out = vec![0.0_f32; width];
    for h in 0..heads {
        let scores: Vec<f64> = (0..keys)
            .map(|j| {
                (0..64)
                    .map(|e| {
                        f64::from(q[query * width + h * 64 + e])
                            * f64::from(k[j * width + h * 64 + e])
                    })
                    .sum::<f64>()
                    / 8.0
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        for e in 0..64 {
            out[h * 64 + e] = (weights
                .iter()
                .enumerate()
                .map(|(j, p)| p * f64::from(v[j * width + h * 64 + e]))
                .sum::<f64>()
                / total) as f32;
        }
    }
    out
}

#[test]
fn encoder_attention_matches_reference_across_split_queries() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let (tokens, heads) = (161_usize, 3_usize);
    let width = heads * 64;
    let scale = |v: Vec<f32>| v.into_iter().map(|x| x * 4.0).collect::<Vec<_>>();
    let q = scale(values(tokens * width, 21));
    let k = scale(values(tokens * width, 22));
    let v = values(tokens * width, 23);
    let (qb, kb, vb) = (
        buffer(&context, &q),
        buffer(&context, &k),
        buffer(&context, &v),
    );
    let out = MetalBuffer::empty(context.device(), tokens * width * 4).expect("out");
    let mut batch = CommandBatch::new(&context).expect("batch");
    for queries in [(0, 100), (100, 61)] {
        kernels
            .encoder_attention(
                &mut batch,
                Region::whole(&qb),
                Region::whole(&kb),
                Region::whole(&vb),
                Region::whole(&out),
                tokens as u32,
                heads as u32,
                queries,
            )
            .expect("attention");
    }
    batch.commit_and_wait().expect("commit");
    let got = out.as_slice::<f32>();
    for query in [0, 1, 99, 100, 127, 128, 160] {
        let want = attention_reference(&q, &k, &v, heads, query, tokens);
        close(&got[query * width..(query + 1) * width], &want, 2e-5);
    }
}

#[test]
fn decoder_attention_is_causal_over_the_cache_and_full_over_frames() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let (keys, heads, queries) = (700_usize, 2_usize, 3_usize);
    let width = heads * 64;
    let q = values(queries * width, 31)
        .into_iter()
        .map(|x| x * 4.0)
        .collect::<Vec<_>>();
    let k = values(keys * width, 32)
        .into_iter()
        .map(|x| x * 4.0)
        .collect::<Vec<_>>();
    let v = values(keys * width, 33);
    let (qb, kb, vb) = (
        buffer(&context, &q),
        buffer(&context, &k),
        buffer(&context, &v),
    );
    let causal = MetalBuffer::empty(context.device(), queries * width * 4).expect("out");
    let full = MetalBuffer::empty(context.device(), queries * width * 4).expect("out");
    let mut batch = CommandBatch::new(&context).expect("batch");
    let base = 40_u32;
    kernels
        .decoder_attention(
            &mut batch,
            Region::whole(&qb),
            Region::whole(&kb),
            Region::whole(&vb),
            Region::whole(&causal),
            queries as u32,
            heads as u32,
            DecoderKeys::Causal { base },
        )
        .expect("causal");
    kernels
        .decoder_attention(
            &mut batch,
            Region::whole(&qb),
            Region::whole(&kb),
            Region::whole(&vb),
            Region::whole(&full),
            queries as u32,
            heads as u32,
            DecoderKeys::All { keys: keys as u32 },
        )
        .expect("full");
    batch.commit_and_wait().expect("commit");
    for i in 0..queries {
        let want = attention_reference(&q, &k, &v, heads, i, base as usize + i + 1);
        close(
            &causal.as_slice::<f32>()[i * width..(i + 1) * width],
            &want,
            2e-5,
        );
        let want = attention_reference(&q, &k, &v, heads, i, keys);
        close(
            &full.as_slice::<f32>()[i * width..(i + 1) * width],
            &want,
            2e-5,
        );
    }
}

/// Throughput of the encoder's largest GEMM shape (run explicitly).
#[test]
#[ignore = "benchmark; run with --ignored under the GPU lock"]
fn gemm_throughput_on_encoder_shapes() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    for (rows, columns, tokens) in [
        (5120_usize, 1280_usize, 1500_usize),
        (1280, 5120, 1500),
        (1280, 1280, 1500),
    ] {
        let (halves, _) = f16_matrix(rows, columns, 1);
        let packed = MetalBuffer::from_slice(context.device(), &halves).expect("weights");
        let x = buffer(&context, &values(tokens * columns, 3));
        let y = MetalBuffer::empty(context.device(), tokens * rows * 4).expect("y");
        let matrix = F16Rows {
            weights: &packed,
            offset: 0,
            rows: rows as u32,
            columns: columns as u32,
        };
        let mut best = f64::MAX;
        for _ in 0..5 {
            let mut batch = CommandBatch::new(&context).expect("batch");
            for _ in 0..10 {
                kernels
                    .gemm(
                        &mut batch,
                        matrix,
                        Region::whole(&x),
                        Region::whole(&y),
                        tokens as u32,
                        &Epilogue::default(),
                    )
                    .expect("gemm");
            }
            let started = std::time::Instant::now();
            batch.commit_and_wait().expect("commit");
            best = best.min(started.elapsed().as_secs_f64() / 10.0);
        }
        let tflops = 2.0 * (rows * columns * tokens) as f64 / best / 1e12;
        eprintln!(
            "gemm {rows}x{columns} x {tokens} tokens: {:.3} ms, {tflops:.2} TFLOPS",
            best * 1e3
        );
    }
}
