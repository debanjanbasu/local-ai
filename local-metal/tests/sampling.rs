#![allow(unsafe_code, clippy::expect_used)]

use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::sampling::{
    GpuTopK, LogprobRows, MAX_TOP_K, MAX_TOP_LOGPROBS, TopKCandidate, TopKRows,
};
use local_metal::shaders::ShaderLibrary;

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

fn reference(values: &[f32], k: usize) -> Vec<TopKCandidate> {
    let mut ids: Vec<usize> = (0..values.len()).collect();
    ids.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
    ids.into_iter()
        .take(k)
        .map(|id| TopKCandidate {
            token_id: id as u32,
            logit: values[id],
        })
        .collect()
}

fn assert_exact(actual: &[TopKCandidate], expected: &[TopKCandidate]) {
    assert_eq!(actual.len(), expected.len());
    for (rank, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.token_id, expected.token_id, "token at rank {rank}");
        assert_eq!(
            actual.logit.to_bits(),
            expected.logit.to_bits(),
            "bits at rank {rank}"
        );
    }
}

fn values(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let bits = (i as u32).wrapping_mul(0x9e37_79b9);
            ((bits & 0xffff) as f32 - 32768.0) / 127.0
        })
        .collect()
}

#[test]
fn gpu_matches_total_cmp_for_tiny_odd_and_full_vocab() {
    let Some(context) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let topk = GpuTopK::new(&context, &shaders, 248_320).expect("top-k");
    assert_eq!(topk.maximum_supported_k(), MAX_TOP_K);
    assert!(topk.allocated_bytes() >= MAX_TOP_K * size_of::<TopKCandidate>());

    for (count, k) in [(1, 1), (17, 1), (257, 20), (1_003, 64), (248_320, 64)] {
        let input = values(count);
        let padded = [vec![-999.0; 3], input.clone()].concat();
        let logits = MetalBuffer::from_slice(context.device(), &padded).expect("logits");
        let mut batch = CommandBatch::new(&context).expect("batch");
        topk.encode(&mut batch, &logits, 3 * size_of::<f32>(), count, k)
            .expect("encode");
        batch.commit_and_wait().expect("wait");
        assert_exact(&topk.candidates(k), &reference(&input, k));
        assert_eq!(
            // SAFETY: the batch has completed and nothing else touches the buffer.
            unsafe { logits.as_slice::<f32>() }
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            padded.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn preserves_special_bits_and_lowest_id_ties_across_reuse() {
    let Some(context) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let topk = GpuTopK::new(&context, &shaders, 300).expect("top-k");
    let special = [
        f32::from_bits(0xffc0_0002),
        f32::NEG_INFINITY,
        -3.0,
        -0.0,
        0.0,
        3.0,
        f32::INFINITY,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0x7fc0_0002),
        3.0,
        -0.0,
        0.0,
    ];

    for (input, k) in [
        (special.to_vec(), 12),
        (vec![7.0; 300], 20),
        (values(65), 64),
    ] {
        let logits = MetalBuffer::from_slice(context.device(), &input).expect("logits");
        let mut batch = CommandBatch::new(&context).expect("batch");
        topk.encode(&mut batch, &logits, 0, input.len(), k)
            .expect("encode");
        batch.commit_and_wait().expect("wait");
        assert_exact(&topk.candidates(k), &reference(&input, k));
    }
}

/// Results are copies: encoding into the same workspace later must not
/// change candidates already read from it.
#[test]
fn candidates_survive_a_later_encode() {
    let Some(context) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let topk = GpuTopK::new(&context, &shaders, 300).expect("top-k");
    let run = |input: &[f32]| {
        let logits = MetalBuffer::from_slice(context.device(), input).expect("logits");
        let mut batch = CommandBatch::new(&context).expect("batch");
        topk.encode(&mut batch, &logits, 0, input.len(), 8)
            .expect("encode");
        batch.commit_and_wait().expect("wait");
    };
    let first_input = values(300);
    run(&first_input);
    let first = topk.candidates(8);
    run(&first_input.iter().map(|value| -value).collect::<Vec<_>>());
    assert_exact(&first, &reference(&first_input, 8));
}

#[test]
fn rejects_invalid_bounds_before_dispatch() {
    let Some(context) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    assert!(GpuTopK::new(&context, &shaders, 0).is_err());
    let topk = GpuTopK::new(&context, &shaders, 100).expect("top-k");
    let short = MetalBuffer::from_slice(context.device(), &[0.0_f32; 9]).expect("logits");

    for (vocab, k) in [(10, 0), (10, 65), (10, 11), (0, 1), (101, 1), (10, 1)] {
        let mut batch = CommandBatch::new(&context).expect("batch");
        assert!(topk.encode(&mut batch, &short, 0, vocab, k).is_err());
        assert_eq!(batch.dispatch_count(), 0);
        batch.commit_and_wait().expect("wait");
    }
}

/// Every row of a multi-row selection is the row's own exact top k, ties to
/// the lower id, special bit patterns preserved, at every k up to 64.
#[test]
fn topk_rows_match_total_cmp_per_row() {
    let Some(context) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    for vocab in [65, 4_097, 248_320] {
        let rows = 4;
        let workspace = TopKRows::new(&context, &shaders, vocab, 8).expect("workspace");
        let mut input = Vec::with_capacity(rows * vocab);
        input.extend(values(vocab));
        input.extend((0..vocab).map(|i| (i % 7) as f32 - 30.0));
        input.extend((0..vocab).map(|i| match i % 97 {
            0 => f32::from_bits(0x7fc0_0001),
            1 => f32::INFINITY,
            2 => f32::from_bits(0xffc0_0002),
            3 => -0.0,
            4 => 0.0,
            _ => -(i as f32 % 13.0),
        }));
        input.extend(values(vocab).iter().rev().map(|value| value / 8.0));
        let logits = MetalBuffer::from_slice(context.device(), &input).expect("logits");
        for k in [1, 20, MAX_TOP_K] {
            let mut batch = CommandBatch::new(&context).expect("batch");
            workspace
                .encode(&mut batch, &logits, rows, k)
                .expect("encode");
            batch.commit_and_wait().expect("wait");
            for row in 0..rows {
                assert_exact(
                    &workspace.row(row, k),
                    &reference(&input[row * vocab..(row + 1) * vocab], k),
                );
            }
        }
    }
}

/// Log-sum-exp in f64, independent of the kernel's grouping and order.
fn reference_lse(values: &[f32]) -> f64 {
    let top = f64::from(values.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    top + values
        .iter()
        .map(|&value| (f64::from(value) - top).exp())
        .sum::<f64>()
        .ln()
}

#[test]
fn logprob_rows_match_a_float64_reference() {
    let Some(context) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    for vocab in [17, 4_097, 248_320] {
        let rows = 3;
        let workspace = LogprobRows::new(&context, &shaders, vocab, 4).expect("workspace");
        // Row 0: spread values, row 1: many exact ties (lowest id must win),
        // row 2: one dominant logit whose probability is nearly 1.
        let mut input = Vec::with_capacity(rows * vocab);
        input.extend(values(vocab).iter().map(|value| value / 16.0));
        input.extend((0..vocab).map(|i| (i % 7) as f32 - 30.0));
        input.extend((0..vocab).map(|i| {
            if i == vocab / 2 {
                40.0
            } else {
                -(i as f32 % 13.0)
            }
        }));
        let logits = MetalBuffer::from_slice(context.device(), &input).expect("logits");
        for k in [0, 1, MAX_TOP_LOGPROBS] {
            let mut batch = CommandBatch::new(&context).expect("batch");
            workspace
                .encode(&mut batch, &logits, rows, k)
                .expect("encode");
            batch.commit_and_wait().expect("wait");
            for row in 0..rows {
                let values = &input[row * vocab..(row + 1) * vocab];
                let expected = reference_lse(values);
                let actual = f64::from(workspace.log_sum_exp(row));
                assert!(
                    (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                    "vocab {vocab} row {row}: lse {actual} vs {expected}"
                );
                assert_exact(
                    &workspace.top(row, k.min(vocab)),
                    &reference(values, k.min(vocab)),
                );
            }
        }
    }
}
