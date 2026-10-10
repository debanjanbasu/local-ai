//! SIMD split attention on both sides of the 1,024-token threshold where
//! splits grow from 32 to 128 tokens: results match an F64 reference, stay
//! causal, and write only their own rows, with a workspace sized tightly
//! below the threshold.

#![allow(clippy::expect_used)]

use local_metal::{
    batch::CommandBatch,
    bonsai_ops::{AttentionKernel, AttentionWorkspace, BonsaiOps, KvFormat, KvLayout},
    buffer::MetalBuffer,
    context::MetalContext,
    shaders::ShaderLibrary,
};

const CAPACITY: u32 = 1300;
/// Below `SPLIT_TENSOR_MIN_PREFIX`; sized tightly so the 32-token splits'
/// partial records must come from the workspace's own reservation.
const SHORT_CAPACITY: u32 = 1023;
const ROWS: usize = 9;
/// Seen only by prefixes above it, so a causal leak moves every head.
const MARKED_TOKEN: usize = 1019;

fn gpu_or_skip() -> Option<MetalContext> {
    let context = MetalContext::new();
    if matches!(context, Err(local_metal::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    Some(context.expect("Metal context failed for a reason other than a missing device"))
}

/// The SIMD build always, and the tensor build where this GPU has one: its
/// short prefixes take the same SIMD split kernel.
fn kernels(context: &MetalContext) -> Vec<BonsaiOps> {
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let mut ops = vec![
        BonsaiOps::new_with_attention_kernel(context, &shaders, AttentionKernel::SimdF32)
            .expect("SIMD ops"),
    ];
    match BonsaiOps::new_with_attention_kernel(context, &shaders, AttentionKernel::TensorF32) {
        Ok(tensor) => ops.push(tensor),
        Err(error) => eprintln!("tensor attention unavailable, SIMD only: {error}"),
    }
    ops
}

fn signal(index: usize, salt: usize) -> f32 {
    ((index * 17 + index / 31 * 13 + salt * 37) % 241) as f32 / 127.0 - 0.87
}

fn hadamard_rows(values: &[f64]) -> Vec<f64> {
    let mut out = values.to_vec();
    for row in out.chunks_mut(256) {
        let mut distance = 1;
        while distance < 256 {
            for i in 0..256 {
                if i & distance == 0 {
                    let (a, b) = (row[i], row[i | distance]);
                    row[i] = a + b;
                    row[i | distance] = a - b;
                }
            }
            distance <<= 1;
        }
        for value in row.iter_mut() {
            *value /= 16.0;
        }
    }
    out
}

/// Independent model of `bonsai_kv_store` rows; returns bytes and the values
/// the kernels read back (Q8 rounded once more to half, their operand type).
fn encode(format: KvFormat, data: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(data.len() / 1024 * format.token_bytes());
    let mut seen = Vec::with_capacity(data.len());
    for values in data.chunks(1024) {
        let mut row = vec![0u8; format.token_bytes()];
        match format {
            KvFormat::F16 => {
                for (index, &value) in values.iter().enumerate() {
                    let half = half::f16::from_f32(value);
                    row[index * 2..][..2].copy_from_slice(&half.to_le_bytes());
                    seen.push(half.to_f32());
                }
            }
            KvFormat::Q8 => {
                for block in 0..32 {
                    let chunk = &values[block * 32..][..32];
                    let peak = chunk.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                    let scale = half::f16::from_f32(peak / 127.0);
                    let inverse = if scale > half::f16::ZERO {
                        1.0 / scale.to_f32()
                    } else {
                        0.0
                    };
                    row[1024 + block * 2..][..2].copy_from_slice(&scale.to_le_bytes());
                    for (lane, &value) in chunk.iter().enumerate() {
                        let code = (value * inverse).round_ties_even().clamp(-127.0, 127.0) as i8;
                        row[block * 32 + lane] = code as u8;
                    }
                }
                let (codes, scales) = row.split_at(1024);
                for (index, &code) in codes.iter().enumerate() {
                    let scale = half::f16::from_le_bytes([
                        scales[index / 32 * 2],
                        scales[index / 32 * 2 + 1],
                    ]);
                    let value = f32::from(code.cast_signed()) * scale.to_f32();
                    seen.push(half::f16::from_f32(value).to_f32());
                }
            }
        }
        bytes.extend_from_slice(&row);
    }
    (bytes, seen)
}

/// F64 attention of one query row over `prefix` tokens, before any rotation
/// back or gating.
fn reference(query: &[f32], prefix: usize, keys: &[f32], values: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0; 6144];
    for head in 0..24 {
        let kv = head / 6;
        let q = &query[head * 256..][..256];
        let mut scores = (0..prefix)
            .map(|t| {
                let k = &keys[(t * 4 + kv) * 256..][..256];
                q.iter()
                    .zip(k)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
                    / 16.0
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for score in &mut scores {
            *score = (*score - maximum).exp();
        }
        let denominator = scores.iter().sum::<f64>();
        let row = &mut out[head * 256..][..256];
        for (t, p) in scores.iter().enumerate() {
            let v = &values[(t * 4 + kv) * 256..][..256];
            for (o, &x) in row.iter_mut().zip(v) {
                *o = p.mul_add(f64::from(x), *o);
            }
        }
        for o in row {
            *o /= denominator;
        }
    }
    out
}

fn finish(mut expected: Vec<f64>, rotated: bool, gate: Option<&[f32]>) -> Vec<f64> {
    if rotated {
        expected = hadamard_rows(&expected);
    }
    if let Some(gate) = gate {
        for (value, &g) in expected.iter_mut().zip(gate) {
            *value /= 1.0 + (-f64::from(g)).exp();
        }
    }
    expected
}

/// Asserts every element within `tolerance` relative error.
fn check(actual: &[f32], expected: &[f64], tolerance: f64, what: &str) {
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(a) - e).abs() / (1.0 + e.abs());
        assert!(
            a.is_finite() && error <= tolerance,
            "{what} element {index}: {a} != {e}"
        );
    }
}

struct Inputs {
    query: Vec<f32>,
    gate: Vec<f32>,
    query_buffer: MetalBuffer,
    gate_buffer: MetalBuffer,
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// Distinct, asymmetric rows: every query row and head has its own pattern
/// and sharpness, keys differ per token and KV head, and values drift with
/// position so a dropped or misplaced split moves the mean.
fn inputs(context: &MetalContext) -> Inputs {
    let query = (0..6144 * ROWS)
        .map(|i| signal(i + i / 6144 * 6151, 19) * 0.05f32.mul_add((i % 6144 / 256) as f32, 1.2))
        .collect::<Vec<_>>();
    let gate = (0..6144 * ROWS)
        .map(|i| signal(i + 3, 23) * 3.0)
        .collect::<Vec<_>>();
    let keys = (0..CAPACITY as usize * 1024)
        .map(|i| signal(i / 1024 * 1031 + i % 1024, 11) * 2.0)
        .collect::<Vec<_>>();
    let values = (0..CAPACITY as usize * 1024)
        .map(|i| {
            if i / 1024 == MARKED_TOKEN {
                4.0 + (i % 17) as f32 / 8.0
            } else {
                signal(i, 7) + (i / 1024) as f32 / 3000.0
            }
        })
        .collect::<Vec<_>>();
    Inputs {
        query_buffer: MetalBuffer::from_slice(context.device(), &query).expect("query"),
        gate_buffer: MetalBuffer::from_slice(context.device(), &gate).expect("gate"),
        query,
        gate,
        keys,
        values,
    }
}

fn guarded(context: &MetalContext, count: usize) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), &vec![f32::NAN; count + 7]).expect("output")
}

#[test]
#[allow(clippy::too_many_lines)]
fn short_simd_splits_track_the_reference_on_both_sides_of_the_threshold() {
    let Some(ctx) = gpu_or_skip() else { return };
    let all_ops = kernels(&ctx);
    let data = inputs(&ctx);
    let short = AttentionWorkspace::new(&ctx, SHORT_CAPACITY).expect("short workspace");
    let wide = AttentionWorkspace::new(&ctx, CAPACITY).expect("wide workspace");
    let workspace_for = |end: u32| if end <= SHORT_CAPACITY { &short } else { &wide };
    let mut output = guarded(&ctx, 6144 * ROWS);
    for layout in [KvLayout::F16, KvLayout::Q8] {
        let rotated = layout != KvLayout::F16;
        let (k_bytes, keys) = encode(layout.key, &data.keys);
        let (v_bytes, values) = encode(layout.value, &data.values);
        let k_cache = MetalBuffer::from_slice(ctx.device(), &k_bytes).expect("keys");
        let v_cache = MetalBuffer::from_slice(ctx.device(), &v_bytes).expect("values");
        // Split-boundary and non-multiple prefixes on both sides of the
        // threshold; the query row (and so its offset) varies with them.
        for (case, prefix) in [
            1u32, 2, 31, 32, 33, 63, 64, 65, 97, 127, 128, 129, 255, 300, 511, 777, 1000, 1019,
            1020, 1023, 1024, 1025, 1100, 1300,
        ]
        .into_iter()
        .enumerate()
        {
            let row = case % ROWS;
            let query = &data.query[row * 6144..][..6144];
            let raw = reference(query, prefix as usize, &keys, &values);
            for ops in &all_ops {
                for gated in [false, true] {
                    let gate = gated.then_some(&data.gate[row * 6144..][..6144]);
                    let expected = finish(raw.clone(), rotated, gate);
                    output.as_mut_slice::<f32>().fill(f32::NAN);
                    let mut batch = CommandBatch::new(&ctx).expect("batch");
                    ops.attention_row_kv(
                        layout,
                        &mut batch,
                        &data.query_buffer,
                        &k_cache,
                        &v_cache,
                        gated.then_some(&data.gate_buffer),
                        &output,
                        prefix,
                        workspace_for(prefix),
                        row as u32,
                    )
                    .expect("attention row");
                    batch.commit_and_wait().expect("attention completion");
                    let out = output.as_slice::<f32>();
                    let what = format!(
                        "{:?} {layout:?} prefix {prefix} row {row} gated {gated}",
                        ops.attention_kernel()
                    );
                    check(&out[row * 6144..][..6144], &expected, 2e-5, &what);
                    assert!(
                        out[..row * 6144]
                            .iter()
                            .chain(&out[(row + 1) * 6144..])
                            .all(|v| v.is_nan()),
                        "{what}: wrote outside its row"
                    );
                }
            }
        }
        // Short verification blocks: rows on both sides of the threshold in
        // one call (row path, end > 1,024), and one ending at 1,008 that stays
        // on the block kernel.
        for (position, rows, gated) in [
            (1017u32, 8u32, true),
            (1020, 7, false),
            (1022, 3, true),
            (1000, 8, true),
        ] {
            let expected = (0..rows as usize)
                .map(|row| {
                    let query = &data.query[row * 6144..][..6144];
                    let prefix = position as usize + row + 1;
                    let gate = gated.then_some(&data.gate[row * 6144..][..6144]);
                    finish(reference(query, prefix, &keys, &values), rotated, gate)
                })
                .collect::<Vec<_>>();
            for ops in &all_ops {
                output.as_mut_slice::<f32>().fill(f32::NAN);
                let mut batch = CommandBatch::new(&ctx).expect("batch");
                ops.attention_block_kv(
                    layout,
                    &mut batch,
                    &data.query_buffer,
                    &k_cache,
                    &v_cache,
                    gated.then_some(&data.gate_buffer),
                    &output,
                    position,
                    rows,
                    workspace_for(position + rows),
                )
                .expect("attention block");
                batch.commit_and_wait().expect("block completion");
                let out = output.as_slice::<f32>();
                for (row, expected) in expected.iter().enumerate() {
                    let what = format!(
                        "{:?} {layout:?} block at {position} row {row}",
                        ops.attention_kernel()
                    );
                    check(&out[row * 6144..][..6144], expected, 2e-5, &what);
                }
                assert!(
                    out[rows as usize * 6144..].iter().all(|v| v.is_nan()),
                    "block at {position}: tail overwritten"
                );
            }
        }
    }
}
