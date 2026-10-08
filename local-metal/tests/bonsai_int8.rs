#![allow(clippy::expect_used)]

use local_metal::batch::CommandBatch;
use local_metal::bonsai::{BonsaiKernels, Int8Matrix, PTQ1_BLOCK_BYTES, Ptq1Matrix};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::shaders::ShaderLibrary;

/// `None` when this machine exposes no Metal device, so GPU tests skip
/// instead of failing. Any other failure is still a failure.
fn gpu_or_skip() -> Option<MetalContext> {
    let context = MetalContext::new();
    if matches!(context, Err(local_metal::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    Some(context.expect("Metal context failed for a reason other than a missing device"))
}

fn setup() -> Option<(MetalContext, BonsaiKernels)> {
    let context = gpu_or_skip()?;
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let kernels = BonsaiKernels::new(&context, &shaders).expect("Bonsai kernels");
    Some((context, kernels))
}

const fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

struct Matrix {
    codes: Vec<i8>,
    scales: Vec<f32>,
    weights: MetalBuffer,
    scale_buffer: MetalBuffer,
    rows: usize,
    columns: usize,
}

impl Matrix {
    fn random(context: &MetalContext, state: &mut u64, rows: usize, columns: usize) -> Self {
        // Every int8 value the contract allows, including both extremes.
        let codes: Vec<i8> = (0..rows * columns)
            .map(|_| ((xorshift(state) % 255) as i16 - 127) as i8)
            .collect();
        let scales: Vec<f32> = (0..rows)
            .map(|_| ((xorshift(state) % 1000) as f32).mul_add(2e-6, 1e-4))
            .collect();
        Self {
            weights: MetalBuffer::from_slice(context.device(), &codes).expect("weights"),
            scale_buffer: MetalBuffer::from_slice(context.device(), &scales).expect("scales"),
            codes,
            scales,
            rows,
            columns,
        }
    }

    fn view(&self) -> Int8Matrix<'_> {
        Int8Matrix::new(
            &self.weights,
            &self.scale_buffer,
            self.rows as u32,
            self.columns as u32,
        )
        .expect("int8 view")
    }

    fn project(&self, input: &[f32], row: usize) -> f64 {
        let weights = &self.codes[row * self.columns..][..self.columns];
        weights
            .iter()
            .zip(input)
            .map(|(&w, &x)| f64::from(w) * f64::from(x))
            .sum::<f64>()
            * f64::from(self.scales[row])
    }
}

/// NaN-filled output with a guard tail, so a short or stray write shows.
fn guarded(context: &MetalContext, count: usize) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), &vec![f32::NAN; count + 17]).expect("output")
}

fn assert_close(label: &str, actual: &MetalBuffer, expected: &[f64]) -> f64 {
    let actual = actual.as_slice::<f32>();
    let peak = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(got) - want).abs() / peak;
        assert!(
            got.is_finite() && error < 3e-5,
            "{label} element {index}: {got} vs {want}"
        );
        worst = worst.max(error);
    }
    assert!(
        actual[expected.len()..].iter().all(|value| value.is_nan()),
        "{label}: tail overwritten"
    );
    worst
}

/// Every int8 kernel equals the F64 contract `y[t, r] = scale[r] · Σ int8 · x`
/// at the head's three input widths, for the row counts drafts (1), verify and
/// commit blocks (2..=5), chunked blocks (12) and K/V-only prefill ingestion
/// (40, 128) use; the fused concat and `SwiGLU` kernels match their parts.
#[test]
#[allow(clippy::too_many_lines)]
fn int8_projections_match_f64_reference() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    let mut state = 0x9e37_79b9_7f4a_7c15;
    // Not a multiple of four, so every kernel sees a partial row group.
    let rows = 70;
    let mut worst = 0.0_f64;
    for columns in [5120, 6144, 17408] {
        let matrices: Vec<Matrix> = [rows, 37, 9]
            .into_iter()
            .map(|count| Matrix::random(&context, &mut state, count, columns))
            .collect();
        for tokens in [1, 2, 3, 4, 5, 8, 12, 40, 128] {
            let input: Vec<f32> = (0..tokens * columns)
                .map(|_| (xorshift(&mut state) % 20_001) as f32 / 10_000.0 - 1.0)
                .collect();
            let input_buffer = MetalBuffer::from_slice(context.device(), &input).expect("input");
            let expected: Vec<f64> = (0..tokens)
                .flat_map(|token| {
                    let row_input = &input[token * columns..][..columns];
                    (0..rows).map(|row| matrices[0].project(row_input, row))
                })
                .collect();
            let output = guarded(&context, tokens * rows);
            let mut batch = CommandBatch::new(&context).expect("batch");
            kernels
                .int8_matmul(
                    &mut batch,
                    matrices[0].view(),
                    &input_buffer,
                    &output,
                    tokens as u32,
                )
                .expect("int8 matmul");
            batch.commit_and_wait().expect("run");
            worst = worst.max(assert_close(
                &format!("width {columns} tokens {tokens}"),
                &output,
                &expected,
            ));
            if tokens != 1 {
                continue;
            }
            let outputs: Vec<MetalBuffer> = matrices
                .iter()
                .map(|matrix| guarded(&context, matrix.rows))
                .collect();
            let product = guarded(&context, rows);
            let mut batch = CommandBatch::new(&context).expect("batch");
            let projections: Vec<_> = matrices
                .iter()
                .zip(&outputs)
                .map(|(matrix, output)| (matrix.view(), output))
                .collect();
            kernels
                .int8_matvec_concat(&mut batch, &projections, &input_buffer)
                .expect("concat");
            let up = Matrix::random(&context, &mut state, rows, columns);
            kernels
                .int8_matvec_swiglu(
                    &mut batch,
                    matrices[0].view(),
                    up.view(),
                    &input_buffer,
                    &product,
                )
                .expect("swiglu");
            batch.commit_and_wait().expect("run fused");
            for (matrix, output) in matrices.iter().zip(&outputs) {
                let expected: Vec<f64> = (0..matrix.rows)
                    .map(|row| matrix.project(&input, row))
                    .collect();
                assert_close(&format!("concat width {columns}"), output, &expected);
            }
            let expected: Vec<f64> = (0..rows)
                .map(|row| {
                    let gate = matrices[0].project(&input, row);
                    gate / (1.0 + (-gate).exp()) * up.project(&input, row)
                })
                .collect();
            assert_close(&format!("swiglu width {columns}"), &product, &expected);
        }
    }
    eprintln!("worst relative int8 projection error {worst:e}");

    // Shapes the kernels cannot serve are refused, not misread.
    let small = Matrix::random(&context, &mut state, 4, 64);
    assert!(Int8Matrix::new(&small.weights, &small.scale_buffer, 4, 32).is_err());
    assert!(Int8Matrix::new(&small.weights, &small.scale_buffer, 5, 64).is_err());
    assert!(Int8Matrix::new(&small.weights, &small.weights, 4, 64).is_err());
}

/// GPU time per projection for the head's matrix shapes: int8 against the
/// `PTQ1_0` kernels on the same shape. Several copies of each matrix are
/// cycled so the system cache cannot serve repeated reads.
#[test]
#[ignore = "benchmark: run alone on an idle GPU"]
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn int8_and_ptq1_projection_throughput() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    let repeats = 24;
    let shapes = [
        (1024, 5120),
        (5120, 5120),
        (12288, 5120),
        (5120, 6144),
        (17408, 5120),
        (5120, 17408),
    ];
    // Raise the GPU clocks before the first measurement, which otherwise
    // reads two to three times slow.
    let warm = MetalBuffer::from_slice(context.device(), &vec![1_u8; 5120 * 5120]).expect("warm");
    let warm_scales =
        MetalBuffer::from_slice(context.device(), &vec![1e-3_f32; 5120]).expect("warm scales");
    let warm_input =
        MetalBuffer::from_slice(context.device(), &vec![0.5_f32; 5120]).expect("warm input");
    let warm_output = MetalBuffer::empty(context.device(), 5120 * 4).expect("warm output");
    let mut batch = CommandBatch::new(&context).expect("batch");
    for _ in 0..2000 {
        kernels
            .int8_matvec(
                &mut batch,
                Int8Matrix::new(&warm, &warm_scales, 5120, 5120).expect("warm matrix"),
                &warm_input,
                &warm_output,
            )
            .expect("warm");
    }
    batch.commit_and_wait().expect("warm run");
    for (rows, columns) in shapes {
        let int8_bytes = rows * columns;
        let ptq1_bytes = rows * columns / 128 * PTQ1_BLOCK_BYTES;
        let copies = (256_usize << 20).div_ceil(int8_bytes).clamp(2, 32);
        let int8: Vec<(MetalBuffer, MetalBuffer)> = (0..copies)
            .map(|copy| {
                let weights: Vec<u8> = (0..int8_bytes)
                    .map(|i| (i.wrapping_mul(31) ^ copy) as u8)
                    .collect();
                (
                    MetalBuffer::from_slice(context.device(), &weights).expect("weights"),
                    MetalBuffer::from_slice(context.device(), &vec![1e-3_f32; rows])
                        .expect("scales"),
                )
            })
            .collect();
        let ptq1: Vec<MetalBuffer> = (0..copies)
            .map(|copy| {
                let mut packed: Vec<u8> = (0..ptq1_bytes)
                    .map(|i| (i.wrapping_mul(17) ^ copy) as u8)
                    .collect();
                for block in packed.as_chunks_mut::<PTQ1_BLOCK_BYTES>().0 {
                    block[26..].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
                }
                MetalBuffer::from_slice(context.device(), &packed).expect("packed")
            })
            .collect();
        for tokens in [1_u32, 2, 3, 4, 5, 8, 40, 128] {
            let input = MetalBuffer::from_slice(
                context.device(),
                &vec![0.5_f32; tokens as usize * columns],
            )
            .expect("input");
            let output =
                MetalBuffer::empty(context.device(), tokens as usize * rows * 4).expect("output");
            let time = |int8_format: bool| -> f64 {
                let mut best = f64::INFINITY;
                for _ in 0..3 {
                    let mut batch = CommandBatch::new(&context).expect("batch");
                    for repeat in 0..repeats {
                        let copy = repeat % copies;
                        if int8_format {
                            let (weights, scales) = &int8[copy];
                            let matrix =
                                Int8Matrix::new(weights, scales, rows as u32, columns as u32)
                                    .expect("int8");
                            kernels
                                .int8_matmul(&mut batch, matrix, &input, &output, tokens)
                                .expect("int8 matmul");
                        } else {
                            let matrix =
                                Ptq1Matrix::new(&ptq1[copy], 0, rows as u32, columns as u32)
                                    .expect("ptq1");
                            kernels
                                .matmul(&mut batch, matrix, &input, &output, tokens)
                                .expect("ptq1 matmul");
                        }
                    }
                    let elapsed = batch
                        .commit_async()
                        .wait_with_gpu_time()
                        .expect("timing")
                        .as_secs_f64();
                    best = best.min(elapsed / f64::from(repeats as u32));
                }
                best
            };
            let int8_seconds = time(true);
            let ptq1_seconds = time(false);
            eprintln!(
                "{rows:>5}x{columns:<5} rows {tokens:>3}: int8 {:>8.1} us {:>6.1} GB/s | \
                 ptq1 {:>8.1} us {:>6.1} GB/s",
                int8_seconds * 1e6,
                int8_bytes as f64 / int8_seconds / 1e9,
                ptq1_seconds * 1e6,
                ptq1_bytes as f64 / ptq1_seconds / 1e9,
            );
        }
    }
}

fn i8m(set: &[(MetalBuffer, MetalBuffer)], copy: usize, rows: usize) -> Int8Matrix<'_> {
    Int8Matrix::new(&set[copy].0, &set[copy].1, rows as u32, 5120).expect("int8")
}
fn p1m(set: &[MetalBuffer], copy: usize, rows: usize) -> Ptq1Matrix<'_> {
    Ptq1Matrix::new(&set[copy], 0, rows as u32, 5120).expect("ptq1")
}

/// GPU time of the head's single-row projection groups by format: q/k/v and
/// gate/up all ternary (fused), all int8 (fused), and mixed (one fused
/// dispatch per format, or two projections and the elementwise `SwiGLU`),
/// against the sum of the same projections run alone. The difference is what
/// a cross-format fused kernel could at most recover.
#[test]
#[ignore = "benchmark: run alone on an idle GPU"]
#[allow(clippy::too_many_lines)]
fn mixed_group_dispatch_costs() {
    use local_metal::bonsai_ops::BonsaiOps;

    let Some((context, kernels)) = setup() else {
        return;
    };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let ops = BonsaiOps::new(&context, &shaders).expect("ops");
    let width = 5120_usize;
    let copies = 4;
    let int8 = |rows: usize| -> Vec<(MetalBuffer, MetalBuffer)> {
        (0..copies)
            .map(|copy| {
                let weights: Vec<u8> = (0..rows * width)
                    .map(|i| (i.wrapping_mul(29) ^ copy) as u8 & 0x7f)
                    .collect();
                (
                    MetalBuffer::from_slice(context.device(), &weights).expect("weights"),
                    MetalBuffer::from_slice(context.device(), &vec![1e-3_f32; rows])
                        .expect("scales"),
                )
            })
            .collect()
    };
    let ptq1 = |rows: usize| -> Vec<MetalBuffer> {
        (0..copies)
            .map(|copy| {
                let mut packed: Vec<u8> = (0..rows * width / 128 * PTQ1_BLOCK_BYTES)
                    .map(|i| (i.wrapping_mul(13) ^ copy) as u8)
                    .collect();
                for block in packed.as_chunks_mut::<PTQ1_BLOCK_BYTES>().0 {
                    block[26..].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
                }
                MetalBuffer::from_slice(context.device(), &packed).expect("packed")
            })
            .collect()
    };
    let (q_rows, kv_rows, ffn_rows) = (12288, 1024, 17408);
    let (q8, k8, v8, g8, u8_) = (
        int8(q_rows),
        int8(kv_rows),
        int8(kv_rows),
        int8(ffn_rows),
        int8(ffn_rows),
    );
    let (q1, k1, v1, g1, u1) = (
        ptq1(q_rows),
        ptq1(kv_rows),
        ptq1(kv_rows),
        ptq1(ffn_rows),
        ptq1(ffn_rows),
    );
    let input = MetalBuffer::from_slice(context.device(), &vec![0.5_f32; width]).expect("input");
    let out = |rows: usize| MetalBuffer::empty(context.device(), rows * 4).expect("output");
    let (q_out, k_out, v_out, g_out, u_out, product) = (
        out(q_rows),
        out(kv_rows),
        out(kv_rows),
        out(ffn_rows),
        out(ffn_rows),
        out(ffn_rows),
    );
    let time = |encode: &dyn Fn(&mut CommandBatch, usize)| -> f64 {
        let repeats = 32;
        let mut best = f64::INFINITY;
        for _ in 0..4 {
            let mut batch = CommandBatch::new(&context).expect("batch");
            for repeat in 0..repeats {
                encode(&mut batch, repeat % copies);
            }
            let elapsed = batch
                .commit_async()
                .wait_with_gpu_time()
                .expect("timing")
                .as_secs_f64();
            best = best.min(elapsed / f64::from(repeats as u32));
        }
        best * 1e6
    };
    // Warm the clocks up.
    time(&|batch, copy| {
        kernels
            .int8_matvec(batch, i8m(&g8, copy, ffn_rows), &input, &g_out)
            .expect("warm");
    });
    let report = |label: &str, group: f64, parts: f64| {
        eprintln!("{label:<34} {group:>7.1} us (parts alone {parts:>7.1} us)");
    };
    let qkv_ternary = time(&|batch, c| {
        kernels
            .matvec_concat(
                batch,
                &[
                    (p1m(&q1, c, q_rows), &q_out),
                    (p1m(&k1, c, kv_rows), &k_out),
                    (p1m(&v1, c, kv_rows), &v_out),
                ],
                &input,
            )
            .expect("concat");
    });
    let qkv_int8 = time(&|batch, c| {
        kernels
            .int8_matvec_concat(
                batch,
                &[
                    (i8m(&q8, c, q_rows), &q_out),
                    (i8m(&k8, c, kv_rows), &k_out),
                    (i8m(&v8, c, kv_rows), &v_out),
                ],
                &input,
            )
            .expect("concat");
    });
    let q_int8 = time(&|batch, c| {
        kernels
            .int8_matvec(batch, i8m(&q8, c, q_rows), &input, &q_out)
            .expect("q");
    });
    let kv_ternary = time(&|batch, c| {
        kernels
            .matvec_concat(
                batch,
                &[
                    (p1m(&k1, c, kv_rows), &k_out),
                    (p1m(&v1, c, kv_rows), &v_out),
                ],
                &input,
            )
            .expect("kv");
    });
    let qkv_mixed = time(&|batch, c| {
        kernels
            .matvec_concat(
                batch,
                &[
                    (p1m(&k1, c, kv_rows), &k_out),
                    (p1m(&v1, c, kv_rows), &v_out),
                ],
                &input,
            )
            .expect("kv");
        kernels
            .int8_matvec_concat(batch, &[(i8m(&q8, c, q_rows), &q_out)], &input)
            .expect("q");
    });
    report("q/k/v ternary (one dispatch)", qkv_ternary, qkv_ternary);
    report("q/k/v int8 (one dispatch)", qkv_int8, qkv_int8);
    report("q int8 + k/v ternary (two)", qkv_mixed, q_int8 + kv_ternary);

    let ffn_ternary = time(&|batch, c| {
        kernels
            .matvec_swiglu(
                batch,
                p1m(&g1, c, ffn_rows),
                p1m(&u1, c, ffn_rows),
                &input,
                &product,
            )
            .expect("swiglu");
    });
    let ffn_int8 = time(&|batch, c| {
        kernels
            .int8_matvec_swiglu(
                batch,
                i8m(&g8, c, ffn_rows),
                i8m(&u8_, c, ffn_rows),
                &input,
                &product,
            )
            .expect("swiglu");
    });
    let gate_int8 = time(&|batch, c| {
        kernels
            .int8_matvec(batch, i8m(&g8, c, ffn_rows), &input, &g_out)
            .expect("gate");
    });
    let up_ternary = time(&|batch, c| {
        kernels
            .matvec(batch, p1m(&u1, c, ffn_rows), &input, &u_out)
            .expect("up");
    });
    let ffn_mixed = time(&|batch, c| {
        kernels
            .int8_matvec(batch, i8m(&g8, c, ffn_rows), &input, &g_out)
            .expect("gate");
        kernels
            .matvec(batch, p1m(&u1, c, ffn_rows), &input, &u_out)
            .expect("up");
        ops.swiglu(batch, &g_out, &u_out, &product, ffn_rows as u32)
            .expect("swiglu");
    });
    report("gate/up ternary (fused SwiGLU)", ffn_ternary, ffn_ternary);
    report("gate/up int8 (fused SwiGLU)", ffn_int8, 2.0 * gate_int8);
    report(
        "gate int8 + up ternary (three)",
        ffn_mixed,
        gate_int8 + up_ternary,
    );
}
