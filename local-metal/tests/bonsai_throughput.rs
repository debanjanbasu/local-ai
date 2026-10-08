#![allow(clippy::expect_used, clippy::cast_precision_loss)]

use local_metal::batch::CommandBatch;
use local_metal::bonsai::{BonsaiKernels, PTQ1_BLOCK_BYTES, Ptq1Matrix};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::shaders::ShaderLibrary;

/// `None` when this machine exposes no Metal device.
fn gpu_or_skip() -> Option<MetalContext> {
    let context = MetalContext::new();
    if matches!(context, Err(local_metal::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    Some(context.expect("Metal context failed for a reason other than a missing device"))
}

/// Pseudo-random packed blocks whose every byte is a valid five-trit code and
/// whose scales are a plausible F16 magnitude, so timing sees real decoding.
fn packed(rows: usize, columns: usize, salt: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt as u64;
    let mut bytes = Vec::with_capacity(rows * columns / 128 * PTQ1_BLOCK_BYTES);
    for _ in 0..rows * columns / 128 {
        let mut block = [0_u8; PTQ1_BLOCK_BYTES];
        for byte in &mut block[..26] {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = ((state % 243) * 256).div_ceil(243) as u8;
        }
        block[26..].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        bytes.extend_from_slice(&block);
    }
    bytes
}

/// GPU time per `PTQ1_0` projection on every shape the model runs, single
/// row (plain and fused matvecs) and 2..8 verify rows. Copies of each matrix
/// rotate through at least 512 MB so the system cache cannot serve repeats.
/// Run alone on an idle GPU:
/// `cargo test --release -p local-metal --test bonsai_throughput -- --ignored --nocapture`
#[test]
#[ignore = "benchmark: run alone on an idle GPU"]
#[allow(clippy::too_many_lines)]
fn ptq1_projection_throughput() {
    let Some(context) = gpu_or_skip() else {
        return;
    };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let kernels = BonsaiKernels::new(&context, &shaders).expect("kernels");
    let filter = std::env::var("PTQ1_BENCH_TOKENS").ok();
    let token_counts: Vec<u32> = filter.as_deref().map_or_else(
        || vec![1, 2, 3, 4, 5, 8],
        |list| {
            list.split(',')
                .map(|t| t.parse().expect("token count"))
                .collect()
        },
    );
    let shapes = [
        (17408_usize, 5120_usize),
        (5120, 17408),
        (12288, 5120),
        (10240, 5120),
        (6144, 5120),
        (1024, 5120),
        (5120, 6144),
        (248_320, 5120),
    ];
    let time = |encode: &dyn Fn(&mut CommandBatch, usize), copies: usize| -> f64 {
        let repeats = (copies * 4).max(24);
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let mut batch = CommandBatch::new(&context).expect("batch");
            for repeat in 0..repeats {
                encode(&mut batch, repeat % copies);
            }
            let elapsed = batch
                .commit_async()
                .wait_with_gpu_time()
                .expect("timing")
                .as_secs_f64();
            best = best.min(elapsed / repeats as f64);
        }
        best
    };
    // Raise the GPU clocks before the first measurement.
    {
        let weights = MetalBuffer::from_slice(context.device(), &packed(4096, 5120, 0)).expect("w");
        let input = MetalBuffer::from_slice(context.device(), &vec![0.5_f32; 5120]).expect("x");
        let output = MetalBuffer::empty(context.device(), 4096 * 4).expect("y");
        let matrix = Ptq1Matrix::new(&weights, 0, 4096, 5120).expect("matrix");
        time(
            &|batch, _| {
                for _ in 0..50 {
                    kernels
                        .matvec(batch, matrix, &input, &output)
                        .expect("warm");
                }
            },
            1,
        );
    }
    let buffers = |rows: usize, columns: usize, salt: usize| -> Vec<MetalBuffer> {
        let bytes = rows * columns / 128 * PTQ1_BLOCK_BYTES;
        let copies = (512_usize << 20).div_ceil(bytes).clamp(1, 32);
        let data = packed(rows, columns, salt);
        (0..copies)
            .map(|copy| {
                let buffer = MetalBuffer::from_slice(context.device(), &data).expect("weights");
                // Distinct contents per copy without regenerating the matrix.
                buffer.copy_from_bytes(&[copy as u8 * 3 + 1], 0);
                buffer
            })
            .collect()
    };
    for (rows, columns) in shapes {
        let weights = buffers(rows, columns, rows ^ columns);
        let bytes = (rows * columns / 128 * PTQ1_BLOCK_BYTES) as f64;
        for &tokens in &token_counts {
            if rows > 100_000 && tokens > 1 {
                continue;
            }
            let input = MetalBuffer::from_slice(
                context.device(),
                &vec![0.25_f32; tokens as usize * columns],
            )
            .expect("input");
            let output =
                MetalBuffer::empty(context.device(), tokens as usize * rows * 4).expect("output");
            let seconds = time(
                &|batch, copy| {
                    let matrix = Ptq1Matrix::new(&weights[copy], 0, rows as u32, columns as u32)
                        .expect("matrix");
                    kernels
                        .matmul(batch, matrix, &input, &output, tokens)
                        .expect("matmul");
                },
                weights.len(),
            );
            eprintln!(
                "{rows:>6}x{columns:<5} rows {tokens}: {:>8.1} us {:>6.1} GB/s",
                seconds * 1e6,
                bytes / seconds / 1e9
            );
        }
    }
    if token_counts.contains(&1) {
        // The fused single-row dispatches decode actually runs.
        let gate = buffers(17408, 5120, 1);
        let up = buffers(17408, 5120, 2);
        let input = MetalBuffer::from_slice(context.device(), &vec![0.25_f32; 5120]).expect("x");
        let product = MetalBuffer::empty(context.device(), 17408 * 4).expect("y");
        let seconds = time(
            &|batch, copy| {
                let gate = Ptq1Matrix::new(&gate[copy], 0, 17408, 5120).expect("gate");
                let up = Ptq1Matrix::new(&up[copy], 0, 17408, 5120).expect("up");
                kernels
                    .matvec_swiglu(batch, gate, up, &input, &product)
                    .expect("swiglu");
            },
            gate.len(),
        );
        let bytes = 2.0 * (17408 * 40 * PTQ1_BLOCK_BYTES) as f64;
        eprintln!(
            "swiglu 2x17408x5120:    {:>8.1} us {:>6.1} GB/s",
            seconds * 1e6,
            bytes / seconds / 1e9
        );
        for (name, segments) in [
            ("concat gdn 10240+6144", &[10240_usize, 6144][..]),
            ("concat attn 12288+1024+1024", &[12288, 1024, 1024][..]),
        ] {
            let sets: Vec<Vec<MetalBuffer>> = segments
                .iter()
                .enumerate()
                .map(|(i, &rows)| buffers(rows, 5120, 7 + i))
                .collect();
            let outputs: Vec<MetalBuffer> = segments
                .iter()
                .map(|&rows| MetalBuffer::empty(context.device(), rows * 4).expect("y"))
                .collect();
            let copies = sets.iter().map(Vec::len).min().expect("copies");
            let seconds = time(
                &|batch, copy| {
                    let projections: Vec<(Ptq1Matrix<'_>, &MetalBuffer)> = segments
                        .iter()
                        .zip(&sets)
                        .zip(&outputs)
                        .map(|((&rows, set), output)| {
                            (
                                Ptq1Matrix::new(&set[copy], 0, rows as u32, 5120).expect("matrix"),
                                output,
                            )
                        })
                        .collect();
                    kernels
                        .matvec_concat(batch, &projections, &input)
                        .expect("concat");
                },
                copies,
            );
            let bytes = (segments.iter().sum::<usize>() * 40 * PTQ1_BLOCK_BYTES) as f64;
            eprintln!(
                "{name:<28} {:>8.1} us {:>6.1} GB/s",
                seconds * 1e6,
                bytes / seconds / 1e9
            );
        }
    }
}
