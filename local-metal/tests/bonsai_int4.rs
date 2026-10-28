#![expect(unsafe_code, clippy::expect_used)]

use half::f16;
use local_metal::batch::CommandBatch;
use local_metal::bonsai::{BonsaiKernels, INT4_GROUP, Int4Matrix, Int8Matrix};
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

const GROUP: usize = INT4_GROUP as usize;

const fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

struct Matrix {
    codes: Vec<u8>,
    /// `(scale, bias)` per group, as stored.
    pairs: Vec<[f16; 2]>,
    code_buffer: MetalBuffer,
    group_buffer: MetalBuffer,
    rows: usize,
    columns: usize,
}

impl Matrix {
    fn random(context: &MetalContext, state: &mut u64, rows: usize, columns: usize) -> Self {
        // Every code, and scales and biases of either sign.
        let codes: Vec<u8> = (0..rows * columns / 2)
            .map(|_| xorshift(state) as u8)
            .collect();
        let pairs: Vec<[f16; 2]> = (0..rows * columns / GROUP)
            .map(|_| {
                let scale = ((xorshift(state) % 1000) as f32).mul_add(2e-6, 1e-4);
                let bias = ((xorshift(state) % 2001) as f32 - 1000.0) * 1.5e-5;
                [f16::from_f32(scale), f16::from_f32(bias)]
            })
            .collect();
        let flat: Vec<f16> = pairs.iter().flatten().copied().collect();
        Self {
            code_buffer: MetalBuffer::from_slice(context.device(), &codes).expect("codes"),
            group_buffer: MetalBuffer::from_slice(context.device(), &flat).expect("groups"),
            codes,
            pairs,
            rows,
            columns,
        }
    }

    fn view(&self) -> Int4Matrix<'_> {
        Int4Matrix::new(
            &self.code_buffer,
            &self.group_buffer,
            self.rows as u32,
            self.columns as u32,
        )
        .expect("int4 view")
    }

    fn weight(&self, row: usize, column: usize) -> f64 {
        let element = row * self.columns + column;
        let byte = self.codes[element / 2];
        let code = if column.is_multiple_of(2) {
            byte & 15
        } else {
            byte >> 4
        };
        let [scale, bias] = self.pairs[element / GROUP];
        f64::from(scale.to_f32()).mul_add(f64::from(code), f64::from(bias.to_f32()))
    }

    fn project(&self, input: &[f32], row: usize) -> f64 {
        (0..self.columns)
            .map(|column| self.weight(row, column) * f64::from(input[column]))
            .sum()
    }
}

/// NaN-filled output with a guard tail, so a short or stray write shows.
fn guarded(context: &MetalContext, count: usize) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), &vec![f32::NAN; count + 17]).expect("output")
}

fn assert_close(label: &str, actual: &MetalBuffer, expected: &[f64]) -> f64 {
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    let actual = unsafe { actual.as_slice::<f32>() };
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

/// Every 4-bit kernel equals the F64 contract `y[t, r] = Σ (scale · code +
/// bias) · x` at the drafter's input widths, for the
/// row counts drafting (8, the selector's 7), appending committed positions
/// (1..=5) and appending a prompt in chunks (12, 64) use.
#[test]
fn int4_projections_match_f64_reference() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    let mut state = 0x9e37_79b9_7f4a_7c15;
    let mut worst = 0.0_f64;
    for columns in [4096, 5120, 17408, 25600] {
        // Not a multiple of four, eight or sixteen, so every kernel sees a
        // partial row group; 4100 rows take the paired-tile wide kernel.
        for rows in [70, 4100] {
            if rows == 4100 && columns > 5120 {
                continue;
            }
            let matrix = Matrix::random(&context, &mut state, rows, columns);
            let token_counts: &[usize] = if rows == 70 {
                &[1, 2, 3, 4, 5, 7, 8, 12, 64]
            } else {
                &[1, 5, 8]
            };
            for &tokens in token_counts {
                // At eight rows the activations and sums exceed the F16 range.
                let magnitude = if tokens == 8 { 1e5 } else { 1.0 };
                let input: Vec<f32> = (0..tokens * columns)
                    .map(|_| ((xorshift(&mut state) % 20_001) as f32 / 10_000.0 - 1.0) * magnitude)
                    .collect();
                let input_buffer =
                    MetalBuffer::from_slice(context.device(), &input).expect("input");
                let expected: Vec<f64> = (0..tokens)
                    .flat_map(|token| {
                        let row_input = &input[token * columns..][..columns];
                        (0..rows).map(|row| matrix.project(row_input, row))
                    })
                    .collect();
                let output = guarded(&context, tokens * rows);
                let mut batch = CommandBatch::new(&context).expect("batch");
                kernels
                    .int4_matmul(
                        &mut batch,
                        matrix.view(),
                        &input_buffer,
                        &output,
                        tokens as u32,
                    )
                    .expect("int4 matmul");
                batch.commit_and_wait().expect("run");
                worst = worst.max(assert_close(
                    &format!("{rows}x{columns} tokens {tokens}"),
                    &output,
                    &expected,
                ));
            }
        }
    }
    eprintln!("worst relative int4 projection error {worst:e}");

    // Shapes the kernels cannot serve are refused, not misread.
    let small = Matrix::random(&context, &mut state, 4, 128);
    let (codes, groups) = (&small.code_buffer, &small.group_buffer);
    assert!(Int4Matrix::new(codes, groups, 4, 128).is_ok());
    assert!(Int4Matrix::new(codes, groups, 8, 64).is_err());
    assert!(Int4Matrix::new(codes, groups, 5, 128).is_err());
    assert!(Int4Matrix::new(codes, codes, 4, 128).is_err());
}

/// GPU time per projection for the drafter's matrix shapes, 4-bit against
/// int8 on the same shape. Several copies of each matrix
/// are cycled so the system cache cannot serve repeated reads.
#[test]
#[ignore = "benchmark: run alone on an idle GPU"]
#[expect(clippy::too_many_lines, clippy::cast_precision_loss)]
fn int4_and_int8_projection_throughput() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    let repeats = 24;
    let shapes = [
        (17408, 5120),
        (5120, 17408),
        (5120, 25600),
        (4096, 5120),
        (5120, 4096),
        (1024, 5120),
        (256, 5120),
    ];
    let token_counts: Vec<u32> = std::env::var("INT4_TOKENS").map_or_else(
        |_| vec![1, 2, 3, 5, 8, 64],
        |list| {
            list.split(',')
                .map(|n| n.parse().expect("token count"))
                .collect()
        },
    );
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
        let copies = (256_usize << 20).div_ceil(int8_bytes / 2).clamp(2, 32);
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
        let int4: Vec<(MetalBuffer, MetalBuffer)> = (0..copies)
            .map(|copy| {
                let codes: Vec<u8> = (0..int8_bytes / 2)
                    .map(|i| (i.wrapping_mul(29) ^ copy) as u8)
                    .collect();
                let pairs = vec![f16::from_f32(1e-3); 2 * rows * columns / GROUP];
                (
                    MetalBuffer::from_slice(context.device(), &codes).expect("codes"),
                    MetalBuffer::from_slice(context.device(), &pairs).expect("groups"),
                )
            })
            .collect();
        for &tokens in &token_counts {
            let input = MetalBuffer::from_slice(
                context.device(),
                &vec![0.5_f32; tokens as usize * columns],
            )
            .expect("input");
            let output =
                MetalBuffer::empty(context.device(), tokens as usize * rows * 4).expect("output");
            let time = |int4_format: bool| -> f64 {
                let mut best = f64::INFINITY;
                for _ in 0..3 {
                    let mut batch = CommandBatch::new(&context).expect("batch");
                    for repeat in 0..repeats {
                        let copy = repeat % copies;
                        if int4_format {
                            let (codes, groups) = &int4[copy];
                            let matrix =
                                Int4Matrix::new(codes, groups, rows as u32, columns as u32)
                                    .expect("int4");
                            kernels
                                .int4_matmul(&mut batch, matrix, &input, &output, tokens)
                                .expect("int4 matmul");
                        } else {
                            let (weights, scales) = &int8[copy];
                            let matrix =
                                Int8Matrix::new(weights, scales, rows as u32, columns as u32)
                                    .expect("int8");
                            kernels
                                .int8_matmul(&mut batch, matrix, &input, &output, tokens)
                                .expect("int8 matmul");
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
            let int8_seconds = time(false);
            let int4_seconds = time(true);
            let int4_bytes = int8_bytes / 2 + rows * columns / GROUP * 4;
            eprintln!(
                "{rows:>5}x{columns:<5} rows {tokens:>3}: int8 {:>7.1} us {:>5.0} GB/s | int4 \
                 {:>7.1} us {:>5.0} GB/s",
                int8_seconds * 1e6,
                int8_bytes as f64 / int8_seconds / 1e9,
                int4_seconds * 1e6,
                int4_bytes as f64 / int4_seconds / 1e9,
            );
        }
    }
}
