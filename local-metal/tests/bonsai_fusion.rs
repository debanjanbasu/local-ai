#![allow(clippy::expect_used, clippy::too_many_lines)]

//! Every fused decode kernel must reproduce, bit for bit, the separate
//! dispatches it replaces: greedy output is only unchanged if logits are.

use half::{bf16, f16};
use local_metal::batch::CommandBatch;
use local_metal::bonsai::{
    BonsaiKernels, HadamardDirection, PTQ1_BLOCK_BYTES, Ptq1Matrix, SignedHadamard,
};
use local_metal::bonsai_ops::{Bf16Matrix, BonsaiOps, RmsNormParams};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::shaders::ShaderLibrary;

struct Rng(u64);

impl Rng {
    const fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1_u64 << 24) as f32).mul_add(2.0, -1.0)
    }

    fn floats(&mut self, context: &MetalContext, count: usize) -> MetalBuffer {
        let values: Vec<f32> = (0..count).map(|_| self.unit()).collect();
        MetalBuffer::from_slice(context.device(), &values).expect("floats")
    }

    /// Arbitrary bytes are valid PTQ1 codes; scales are kept finite.
    fn packed(&mut self, context: &MetalContext, rows: usize, columns: usize) -> MetalBuffer {
        let mut bytes = vec![0_u8; rows * columns / 128 * PTQ1_BLOCK_BYTES];
        for block in bytes.as_chunks_mut::<PTQ1_BLOCK_BYTES>().0 {
            for byte in &mut block[..26] {
                *byte = self.next() as u8;
            }
            let scale = f16::from_f32((self.next() % 53 + 1) as f32 / 2048.0);
            block[26..].copy_from_slice(&scale.to_bits().to_le_bytes());
        }
        MetalBuffer::from_slice(context.device(), &bytes).expect("packed")
    }

    fn signs(&mut self, context: &MetalContext, columns: usize) -> SignedHadamard {
        let signs: Vec<f32> = (0..columns)
            .map(|_| if self.next() & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        SignedHadamard::new(context, &signs).expect("signs")
    }
}

fn setup() -> Option<(MetalContext, BonsaiKernels, BonsaiOps)> {
    let context = MetalContext::new();
    if matches!(context, Err(local_metal::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    let context = context.expect("Metal context failed for a reason other than a missing device");
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let kernels = BonsaiKernels::new(&context, &shaders).expect("kernels");
    let ops = BonsaiOps::new(&context, &shaders).expect("ops");
    Some((context, kernels, ops))
}

fn copy(context: &MetalContext, buffer: &MetalBuffer) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), buffer.as_slice::<u32>()).expect("copy")
}

fn assert_bits(name: &str, actual: &MetalBuffer, expected: &MetalBuffer, count: usize) {
    let actual = &actual.as_slice::<u32>()[..count];
    let expected = &expected.as_slice::<u32>()[..count];
    let differ = actual.iter().zip(expected).filter(|(a, e)| a != e).count();
    assert_eq!(differ, 0, "{name}: {differ} of {count} values differ");
}

#[test]
fn fused_projections_match_separate_dispatches_bitwise() {
    let Some((context, kernels, ops)) = setup() else {
        return;
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    // Odd row counts leave partial four-row groups in every segment.
    let (rows, columns) = (37_u32, 512_u32);
    let gate_bytes = rng.packed(&context, rows as usize, columns as usize);
    let up_bytes = rng.packed(&context, rows as usize, columns as usize);
    let gate = Ptq1Matrix::new(&gate_bytes, 0, rows, columns).expect("gate");
    let up = Ptq1Matrix::new(&up_bytes, 0, rows, columns).expect("up");
    let third_bytes = rng.packed(&context, 6, columns as usize);
    let third = Ptq1Matrix::new(&third_bytes, 0, 6, columns).expect("third");
    let input = rng.floats(&context, columns as usize);
    let bf16_input = rng.floats(&context, 96);
    let bf16_values: Vec<u16> = (0..2 * 5 * 96)
        .map(|_| bf16::from_f32(rng.unit()).to_bits())
        .collect();
    let bf16_bytes = MetalBuffer::from_slice(context.device(), &bf16_values).expect("bf16");
    let bf16_matrix = |index: usize| Bf16Matrix {
        buffer: &bf16_bytes,
        offset: index * 5 * 96 * 2,
        rows: 5,
        columns: 96,
    };
    let out: Vec<_> = (0..14).map(|_| rng.floats(&context, 64)).collect();

    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .matvec(&mut batch, gate, &input, &out[0])
        .expect("gate");
    kernels.matvec(&mut batch, up, &input, &out[1]).expect("up");
    kernels
        .matvec(&mut batch, third, &input, &out[2])
        .expect("third");
    ops.swiglu(&mut batch, &out[0], &out[1], &out[3], rows)
        .expect("swiglu");
    ops.bf16_matvec(&mut batch, bf16_matrix(0), &bf16_input, &out[4])
        .expect("alpha");
    ops.bf16_matvec(&mut batch, bf16_matrix(1), &bf16_input, &out[5])
        .expect("beta");
    kernels
        .matvec_swiglu(&mut batch, gate, up, &input, &out[6])
        .expect("fused swiglu");
    kernels
        .matvec_concat(
            &mut batch,
            &[(gate, &out[7]), (up, &out[8]), (third, &out[9])],
            &input,
        )
        .expect("concat");
    kernels
        .matvec_concat_bf16(
            &mut batch,
            &[(gate, &out[10]), (up, &out[11])],
            &input,
            [(bf16_matrix(0), &out[12]), (bf16_matrix(1), &out[13])],
            &bf16_input,
        )
        .expect("concat bf16");
    batch.commit_and_wait().expect("run");
    let rows = rows as usize;
    assert_bits("swiglu", &out[6], &out[3], rows);
    for (name, fused, separate, count) in [
        ("concat 0", 7, 0, rows),
        ("concat 1", 8, 1, rows),
        ("concat 2", 9, 2, 6),
        ("concat bf16 0", 10, 0, rows),
        ("concat bf16 1", 11, 1, rows),
        ("alpha", 12, 4, 5),
        ("beta", 13, 5, 5),
    ] {
        assert_bits(name, &out[fused], &out[separate], count);
    }

    // Outputs must not alias the input or a weight buffer.
    let mut batch = CommandBatch::new(&context).expect("batch");
    assert!(
        kernels
            .normalize_transform(
                &mut batch,
                &rng.signs(&context, 1024),
                &input,
                &input,
                &out[0],
                &input,
                1,
                1e-6,
            )
            .is_err()
    );
    assert!(
        kernels
            .matvec_swiglu(&mut batch, gate, third, &input, &out[0])
            .is_err()
    );
    assert!(
        kernels
            .matvec_concat(&mut batch, &[(gate, &input)], &input)
            .is_err()
    );
    assert_eq!(batch.dispatch_count(), 0);
}

#[test]
fn fused_normalization_and_rotation_match_separate_dispatches_bitwise() {
    let Some((context, kernels, ops)) = setup() else {
        return;
    };
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for tokens in [1_u32, 3] {
        let t = tokens as usize;
        // RMSNorm + rotation over two 1024-wide blocks per row.
        let rotation = rng.signs(&context, 2048);
        let hidden = rng.floats(&context, 2048 * t);
        let norm = rng.floats(&context, 2048);
        let rows: Vec<_> = (0..4).map(|_| rng.floats(&context, 2048 * t)).collect();
        let mut batch = CommandBatch::new(&context).expect("batch");
        ops.rms_norm(
            &mut batch,
            &hidden,
            &norm,
            &rows[0],
            RmsNormParams {
                dimension: 2048,
                rows: tokens,
                stride: 2048,
                epsilon: 1e-6,
                weight_offset: 0,
            },
        )
        .expect("rms");
        kernels
            .transform(
                &mut batch,
                &rotation,
                &rows[0],
                &rows[1],
                tokens,
                HadamardDirection::Forward,
            )
            .expect("rotate");
        kernels
            .normalize_transform(
                &mut batch, &rotation, &hidden, &norm, &rows[2], &rows[3], tokens, 1e-6,
            )
            .expect("fused rms");
        batch.commit_and_wait().expect("run");
        assert_bits("normalized", &rows[2], &rows[0], 2048 * t);
        assert_bits("rotated", &rows[3], &rows[1], 2048 * t);

        // Convolution + Q/K L2 normalization + decay/beta.
        let qkv = rng.floats(&context, 10240 * t);
        let weights = rng.floats(&context, 10240 * 4);
        let history = rng.floats(&context, 10240 * 3);
        let fused_history = copy(&context, &history);
        let scalars: Vec<_> = (0..8).map(|_| rng.floats(&context, 48 * t)).collect();
        let convolved = [
            rng.floats(&context, 10240 * t),
            rng.floats(&context, 10240 * t),
        ];
        let mut batch = CommandBatch::new(&context).expect("batch");
        ops.conv_sequence(&mut batch, &qkv, &weights, &history, &convolved[0], tokens)
            .expect("conv");
        ops.l2_normalize_qk_rows(&mut batch, &convolved[0], 1e-6, tokens)
            .expect("l2");
        ops.decay_beta_rows(
            &mut batch,
            &scalars[0],
            &scalars[1],
            &scalars[2],
            &scalars[3],
            &scalars[4],
            &scalars[5],
            tokens,
        )
        .expect("decay");
        ops.conv_l2_decay(
            &mut batch,
            &qkv,
            &weights,
            &fused_history,
            &convolved[1],
            1e-6,
            [&scalars[0], &scalars[1], &scalars[2], &scalars[3]],
            &scalars[6],
            &scalars[7],
            tokens,
        )
        .expect("fused conv");
        batch.commit_and_wait().expect("run");
        assert_bits("convolved", &convolved[1], &convolved[0], 10240 * t);
        assert_bits("history", &fused_history, &history, 10240 * 3);
        assert_bits("decay", &scalars[6], &scalars[4], 48 * t);
        assert_bits("beta", &scalars[7], &scalars[5], 48 * t);
    }
}

#[test]
fn fused_residual_and_gdn_output_chains_match_separate_dispatches_bitwise() {
    let Some((context, kernels, ops)) = setup() else {
        return;
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for tokens in [1_u32, 3] {
        let t = tokens as usize;
        // Residual add + RMSNorm + rotation over five 1024-wide blocks per row.
        let rotation = rng.signs(&context, 5120);
        let hidden = rng.floats(&context, 5120 * t);
        let branch = rng.floats(&context, 5120 * t);
        let norm = rng.floats(&context, 5120);
        let rows: Vec<_> = (0..6).map(|_| rng.floats(&context, 5120 * t)).collect();
        let mut batch = CommandBatch::new(&context).expect("batch");
        ops.residual_add(&mut batch, &branch, &hidden, &rows[0], 5120 * tokens)
            .expect("add");
        kernels
            .normalize_transform(
                &mut batch, &rotation, &rows[0], &norm, &rows[1], &rows[2], tokens, 1e-6,
            )
            .expect("rms rotate");
        ops.residual_normalize_transform(
            &mut batch,
            &rotation,
            &hidden,
            &branch,
            &norm,
            [&rows[3], &rows[4], &rows[5]],
            tokens,
            1e-6,
        )
        .expect("fused residual");
        assert!(
            ops.residual_normalize_transform(
                &mut batch,
                &rotation,
                &hidden,
                &branch,
                &norm,
                [&hidden, &rows[4], &rows[5]],
                tokens,
                1e-6,
            )
            .is_err(),
            "the sum may not overwrite rows other threadgroups still read"
        );
        batch.commit_and_wait().expect("run");
        assert_bits("sum", &rows[3], &rows[0], 5120 * t);
        assert_bits("normalized", &rows[4], &rows[1], 5120 * t);
        assert_bits("rotated", &rows[5], &rows[2], 5120 * t);

        // GDN output norm + gate + grouped-head reorder + rotation.
        let rotation = rng.signs(&context, 6144);
        let recurrent = rng.floats(&context, 6144 * t);
        let gate = rng.floats(&context, 6144 * t);
        let norm = rng.floats(&context, 128);
        let rows: Vec<_> = (0..3).map(|_| rng.floats(&context, 6144 * t)).collect();
        let mut batch = CommandBatch::new(&context).expect("batch");
        ops.gdn_postprocess_rows(&mut batch, &recurrent, &gate, &norm, &rows[0], 1e-6, tokens)
            .expect("post");
        kernels
            .transform(
                &mut batch,
                &rotation,
                &rows[0],
                &rows[1],
                tokens,
                HadamardDirection::Forward,
            )
            .expect("rotate");
        ops.gdn_postprocess_transform(
            &mut batch, &recurrent, &gate, &norm, &rotation, &rows[2], 1e-6, tokens,
        )
        .expect("fused post");
        batch.commit_and_wait().expect("run");
        assert_bits("rotated GDN output", &rows[2], &rows[1], 6144 * t);
    }
}

#[test]
fn fused_swiglu_and_rotation_match_separate_dispatches_bitwise() {
    let Some((context, kernels, ops)) = setup() else {
        return;
    };
    let mut rng = Rng(0x6a09_e667_f3bc_c908);
    // Three 1024-wide blocks per row, each with its own signs.
    let columns = 3072_usize;
    let rotation = rng.signs(&context, columns);
    for tokens in [2_u32, 4, 8, 64] {
        let count = columns * tokens as usize;
        // Asymmetric operands: gate spans the sigmoid's curved and saturating
        // ranges with a positive bias; up has its own scale and offset.
        let gate_values: Vec<f32> = (0..count).map(|_| rng.unit().mul_add(9.0, 1.5)).collect();
        let up_values: Vec<f32> = (0..count)
            .map(|_| rng.unit().mul_add(2.5, -0.375))
            .collect();
        let gate = MetalBuffer::from_slice(context.device(), &gate_values).expect("gate");
        let up = MetalBuffer::from_slice(context.device(), &up_values).expect("up");
        // The engine rotates in place over the gate buffer.
        let in_place = copy(&context, &gate);
        let rows: Vec<_> = (0..3).map(|_| rng.floats(&context, count)).collect();
        let mut batch = CommandBatch::new(&context).expect("batch");
        ops.swiglu(&mut batch, &gate, &up, &rows[0], count as u32)
            .expect("swiglu");
        kernels
            .transform(
                &mut batch,
                &rotation,
                &rows[0],
                &rows[1],
                tokens,
                HadamardDirection::Forward,
            )
            .expect("rotate");
        kernels
            .swiglu_transform(&mut batch, &rotation, &gate, &up, &rows[2], tokens)
            .expect("fused swiglu rotate");
        kernels
            .swiglu_transform(&mut batch, &rotation, &in_place, &up, &in_place, tokens)
            .expect("in-place fused swiglu rotate");
        batch.commit_and_wait().expect("run");
        assert_bits("rotated SwiGLU", &rows[2], &rows[1], count);
        assert_bits("in-place rotated SwiGLU", &in_place, &rows[1], count);
        assert!(
            rows[1].as_slice::<f32>()[..count]
                .iter()
                .any(|&value| value != 0.0),
            "the reference must be nontrivial"
        );
    }

    // Short buffers and empty shapes encode nothing.
    let short = rng.floats(&context, columns);
    let mut batch = CommandBatch::new(&context).expect("batch");
    assert!(
        kernels
            .swiglu_transform(&mut batch, &rotation, &short, &short, &short, 2)
            .is_err()
    );
    assert!(
        kernels
            .swiglu_transform(&mut batch, &rotation, &short, &short, &short, 0)
            .is_err()
    );
    assert_eq!(batch.dispatch_count(), 0);
}

#[test]
fn concurrent_batch_orders_everything_outside_independent_groups() {
    let Some((context, kernels, ops)) = setup() else {
        return;
    };
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let (rows, columns) = (1024_u32, 1024_u32);
    let gate_bytes = rng.packed(&context, rows as usize, columns as usize);
    let up_bytes = rng.packed(&context, rows as usize, columns as usize);
    let gate = Ptq1Matrix::new(&gate_bytes, 0, rows, columns).expect("gate");
    let up = Ptq1Matrix::new(&up_bytes, 0, rows, columns).expect("up");
    let start = rng.floats(&context, columns as usize);
    let mut results = Vec::new();
    for concurrent in [false, true] {
        let hidden = copy(&context, &start);
        let scratch: Vec<_> = (0..3)
            .map(|_| rng.floats(&context, rows as usize))
            .collect();
        let mut batch = if concurrent {
            CommandBatch::new_concurrent(&context)
        } else {
            CommandBatch::new(&context)
        }
        .expect("batch");
        // A dependent chain: every step reads what the previous one wrote.
        for _ in 0..32 {
            batch
                .independent(|batch| {
                    kernels.matvec(batch, gate, &hidden, &scratch[0])?;
                    kernels.matvec(batch, up, &hidden, &scratch[1])
                })
                .expect("projections");
            ops.swiglu(&mut batch, &scratch[0], &scratch[1], &scratch[2], rows)
                .expect("swiglu");
            ops.residual_add(&mut batch, &scratch[2], &hidden, &hidden, rows)
                .expect("residual");
        }
        batch.commit_and_wait().expect("run");
        results.push(hidden);
    }
    assert_bits("concurrent chain", &results[1], &results[0], rows as usize);
}
