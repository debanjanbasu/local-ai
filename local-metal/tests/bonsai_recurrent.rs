#![allow(clippy::expect_used, clippy::too_many_lines)]

use std::ptr::NonNull;

use local_metal::{
    batch::CommandBatch, buffer::MetalBuffer, context::MetalContext, shaders::ShaderLibrary,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

const STATE: usize = 48 * 128 * 128;
const QKV: usize = 10240;
const OUTPUT: usize = 6144;

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

struct Kernel {
    rows: usize,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

fn kernels(context: &MetalContext) -> Vec<Kernel> {
    named_kernels(context, [("bo_gdn", 1), ("bo_gdn_rows_4", 4)])
}

fn named_kernels(context: &MetalContext, names: [(&str, usize); 2]) -> Vec<Kernel> {
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    names
        .into_iter()
        .map(|(name, rows)| {
            let function = shaders.get_function(name).expect("function");
            Kernel {
                rows,
                pipeline: context
                    .device()
                    .newComputePipelineStateWithFunction_error(&function)
                    .expect("pipeline"),
            }
        })
        .collect()
}

fn encode(batch: &mut CommandBatch, kernel: &Kernel, buffers: [&MetalBuffer; 5], tokens: u32) {
    // In place: the final state overwrites the initial one.
    let [qkv, decay, beta, state, output] = buffers;
    encode_into(
        batch,
        kernel,
        [qkv, decay, beta, state, output, state],
        tokens,
    );
}

/// Buffers: QKV, decay, beta, initial state, output, final state.
#[allow(unsafe_code)]
fn encode_into(batch: &mut CommandBatch, kernel: &Kernel, buffers: [&MetalBuffer; 6], tokens: u32) {
    let encoder = batch.encoder();
    encoder.setComputePipelineState(&kernel.pipeline);
    // SAFETY: callers allocate the fixed GDN geometry; the owned buffers remain
    // alive until completion. Metal copies the scalar argument during encoding.
    unsafe {
        for (index, buffer) in buffers.into_iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buffer.raw()), 0, index);
        }
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(std::ptr::from_ref(&tokens).cast_mut().cast()),
            size_of::<u32>(),
            6,
        );
    }
    // The single-row kernels run four SIMD groups (one row each) per
    // threadgroup; the tiled ones one SIMD group of `rows` rows.
    let simds = if kernel.rows == 1 { 4 } else { 1 };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: OUTPUT / (kernel.rows * simds),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32 * simds,
            height: 1,
            depth: 1,
        },
    );
    batch.record_dispatch();
}

fn floats(context: &MetalContext, values: &[f32]) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), values).expect("float buffer")
}

fn signal(index: usize, salt: usize) -> f32 {
    ((index * 17 + index / 29 * 11 + salt * 37) % 251) as f32 / 131.0 - 0.91
}

fn inputs(tokens: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let qkv = (0..tokens * QKV)
        .map(|i| signal(i, 19) * if i % QKV < 4096 { 0.09 } else { 0.7 })
        .collect();
    let decay = (0..tokens * 48)
        .map(|i| match i % 53 {
            0 => 0.0,
            1 => 1.0,
            _ => ((i % 29) as f32).mul_add(0.0011, 0.966),
        })
        .collect();
    let beta = (0..tokens * 48)
        .map(|i| ((i * 7 % 31) as f32) / 30.0)
        .collect();
    let state = (0..STATE).map(|i| signal(i, 5) * 0.03).collect();
    (qkv, decay, beta, state)
}

fn check(buffer: &MetalBuffer, expected: &[f64]) {
    for (i, (&actual, &wanted)) in buffer.as_slice::<f32>().iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (f64::from(actual) - wanted).abs() < 5e-6 * (1.0 + wanted.abs()),
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
fn tiled_recurrence_matches_f64_for_every_state_row_and_causal_continuation() {
    let Some(context) = gpu_or_skip() else { return };
    let kernels = kernels(&context);
    let tokens = 128;
    let (qkv, decay, beta, initial) = inputs(tokens);
    let mut state_reference = initial.iter().copied().map(f64::from).collect::<Vec<_>>();
    let mut output_reference = vec![0.0; tokens * OUTPUT];
    let mut checkpoints = Vec::new();
    for token in 0..tokens {
        let x = &qkv[token * QKV..][..QKV];
        for head in 0..48 {
            let q = &x[(head % 16) * 128..][..128];
            let k = &x[2048 + (head % 16) * 128..][..128];
            for d in 0..128 {
                let row = head * 128 + d;
                let memory = &mut state_reference[row * 128..][..128];
                for value in &mut *memory {
                    *value *= f64::from(decay[token * 48 + head]);
                }
                let prediction = memory
                    .iter()
                    .zip(k)
                    .map(|(s, &k)| s * f64::from(k))
                    .sum::<f64>();
                let correction =
                    (f64::from(x[4096 + row]) - prediction) * f64::from(beta[token * 48 + head]);
                for (s, &k) in memory.iter_mut().zip(k) {
                    *s = f64::mul_add(f64::from(k), correction, *s);
                }
                output_reference[token * OUTPUT + row] = memory
                    .iter()
                    .zip(q)
                    .map(|(s, &q)| s * f64::from(q))
                    .sum::<f64>()
                    / 128.0_f64.sqrt();
            }
        }
        if [1, 2, 37, 128].contains(&(token + 1)) {
            checkpoints.push((token + 1, state_reference.clone()));
        }
    }
    for kernel in &kernels {
        let mut state = floats(&context, &vec![f32::NAN; STATE + 7]);
        // All head boundaries, the maximum block, split continuation, and a
        // shorter reset after a full block. Future input rows remain readable;
        // NaN guards and F64 outputs catch accidental noncausal processing.
        for (start, count, reset) in [
            (0, 1, true),
            (0, 128, true),
            (0, 37, true),
            (37, 91, false),
            (0, 2, true),
        ] {
            if reset {
                state.as_mut_slice::<f32>()[..STATE].copy_from_slice(&initial);
            }
            let x = floats(&context, &qkv[start * QKV..]);
            let a = floats(&context, &decay[start * 48..]);
            let b = floats(&context, &beta[start * 48..]);
            let out = floats(&context, &vec![f32::NAN; count * OUTPUT + 7]);
            let mut batch = CommandBatch::new(&context).expect("batch");
            encode(&mut batch, kernel, [&x, &a, &b, &state, &out], count as u32);
            batch.commit_and_wait().expect("completion");
            let end = start + count;
            check(&out, &output_reference[start * OUTPUT..end * OUTPUT]);
            check(
                &state,
                &checkpoints
                    .iter()
                    .find(|(position, _)| *position == end)
                    .expect("state checkpoint")
                    .1,
            );
        }
    }
}

/// F64 recurrence over `tokens` rows from `initial`: outputs and final state.
fn reference(
    qkv: &[f32],
    decay: &[f32],
    beta: &[f32],
    initial: &[f32],
    tokens: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut state = initial.iter().copied().map(f64::from).collect::<Vec<_>>();
    let mut output = vec![0.0; tokens * OUTPUT];
    for token in 0..tokens {
        let x = &qkv[token * QKV..][..QKV];
        for head in 0..48 {
            let q = &x[(head % 16) * 128..][..128];
            let k = &x[2048 + (head % 16) * 128..][..128];
            for d in 0..128 {
                let row = head * 128 + d;
                let memory = &mut state[row * 128..][..128];
                for value in &mut *memory {
                    *value *= f64::from(decay[token * 48 + head]);
                }
                let prediction = memory
                    .iter()
                    .zip(k)
                    .map(|(s, &k)| s * f64::from(k))
                    .sum::<f64>();
                let correction =
                    (f64::from(x[4096 + row]) - prediction) * f64::from(beta[token * 48 + head]);
                for (s, &k) in memory.iter_mut().zip(k) {
                    *s = f64::mul_add(f64::from(k), correction, *s);
                }
                output[token * OUTPUT + row] = memory
                    .iter()
                    .zip(q)
                    .map(|(s, &q)| s * f64::from(q))
                    .sum::<f64>()
                    / 128.0_f64.sqrt();
            }
        }
    }
    (output, state)
}

/// F16-stored state: both kernels widen the initial state, run the block in
/// F32 registers, and round only the final state, written to a separate
/// buffer so the initial one survives for a verification rollback.
#[test]
fn f16_state_rounds_once_per_block_and_leaves_a_separate_initial_state_intact() {
    let Some(context) = gpu_or_skip() else { return };
    let kernels = named_kernels(&context, [("bo_gdn_f16", 1), ("bo_gdn_rows_4_f16", 4)]);
    let tokens = 37;
    let (qkv, decay, beta, initial) = inputs(tokens);
    let initial = initial
        .iter()
        .map(|&value| half::f16::from_f32(value))
        .collect::<Vec<_>>();
    let widened = initial
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let (output_reference, state_reference) = reference(&qkv, &decay, &beta, &widened, tokens);
    for kernel in &kernels {
        let start = MetalBuffer::from_slice(context.device(), &initial).expect("initial");
        let end =
            MetalBuffer::from_slice(context.device(), &vec![half::f16::NAN; STATE]).expect("final");
        let x = floats(&context, &qkv);
        let a = floats(&context, &decay);
        let b = floats(&context, &beta);
        let out = floats(&context, &vec![f32::NAN; tokens * OUTPUT + 7]);
        let mut batch = CommandBatch::new(&context).expect("batch");
        encode_into(
            &mut batch,
            kernel,
            [&x, &a, &b, &start, &out, &end],
            tokens as u32,
        );
        batch.commit_and_wait().expect("completion");
        check(&out, &output_reference);
        assert!(
            start
                .as_slice::<half::f16>()
                .iter()
                .zip(&initial)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the initial state was modified"
        );
        for (i, (actual, &wanted)) in end
            .as_slice::<half::f16>()
            .iter()
            .zip(&state_reference)
            .enumerate()
        {
            // One F16 rounding of an F32 value within 5e-6 of the F64 one.
            let tolerance = 5e-6f64.mul_add(1.0 + wanted.abs(), wanted.abs() * 2.0_f64.powi(-11));
            assert!(
                (f64::from(actual.to_f32()) - wanted).abs() <= tolerance,
                "state {i}: {actual} != {wanted}"
            );
        }
    }
}
