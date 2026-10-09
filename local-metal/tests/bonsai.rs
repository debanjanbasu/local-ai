#![allow(clippy::expect_used, clippy::too_many_lines)]

use half::f16;
use local_metal::batch::CommandBatch;
use local_metal::bonsai::{
    BonsaiKernels, HADAMARD_BLOCK_ELEMENTS, HadamardDirection, PTQ1_BLOCK_BYTES,
    PTQ1_BLOCK_ELEMENTS, PrefillKernel, Ptq1Matrix, SignedHadamard, decode_ptq1_row,
};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::shaders::ShaderLibrary;

const GUARD: usize = 17;
const SENTINEL: u32 = 0x7fc0_12ab;

// Sequential stage traversal from the pinned upstream decoder, deliberately not
// the element-index formula used by production. Accumulate references in F64.
fn reference_decode(packed: &[u8]) -> Vec<f64> {
    let mut result = Vec::new();
    for block in packed.as_chunks::<28>().0 {
        let scale = f64::from(f16::from_bits(u16::from_le_bytes([block[26], block[27]])));
        let mut offset = 0;
        for width in [16, 8] {
            let mut power = 1_u16;
            for _ in 0..5 {
                for &byte in &block[offset..offset + width] {
                    let wrapped = (u16::from(byte) * power) % 256;
                    result.push((f64::from((wrapped * 3) / 256) - 1.0) * scale);
                }
                power *= 3;
            }
            offset += width;
        }
        let mut power = 1_u16;
        for _ in 0..4 {
            for &byte in &block[24..26] {
                let wrapped = (u16::from(byte) * power) % 256;
                result.push((f64::from((wrapped * 3) / 256) - 1.0) * scale);
            }
            power *= 3;
        }
    }
    result
}

// Independent integer encoder: accept already-ternary values, not floats to
// re-quantize. The caller's scales are preserved bit-for-bit.
fn encode_trits(trits: &[i8; 128], scale: f16) -> [u8; 28] {
    let mut result = [0_u8; 28];
    let mut input_start = 0;
    let mut output_start = 0;
    for width in [16, 8] {
        for column in 0..width {
            let mut rank = 0_u16;
            for digit in 0..5 {
                rank = 3 * rank + (trits[input_start + digit * width + column] + 1) as u16;
            }
            result[output_start + column] = (rank * 256).div_ceil(243) as u8;
        }
        input_start += width * 5;
        output_start += width;
    }
    for column in 0..2 {
        let mut rank = 0_u16;
        for digit in 0..4 {
            rank = 3 * rank + (trits[120 + digit * 2 + column] + 1) as u16;
        }
        result[24 + column] = (rank * 3 * 256).div_ceil(243) as u8;
    }
    result[26..].copy_from_slice(&scale.to_bits().to_le_bytes());
    result
}

const fn hash(mut value: u32) -> u32 {
    value = (value ^ (value >> 16)).wrapping_mul(0x7feb_352d);
    value = (value ^ (value >> 15)).wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn signs(columns: usize) -> Vec<f32> {
    (0..columns)
        .map(|i| {
            if hash(i as u32 + 71) & 1 == 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect()
}

fn activations(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (hash(i as u32 + 173) % 4093) as f32 / 2048.0 - 0.781_25)
        .collect()
}

// Dense Walsh matrix multiplication, not another butterfly implementation.
fn reference_transform(input: &[f64], signs: &[f32], inverse: bool) -> Vec<f64> {
    input
        .as_chunks::<1024>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(block, values)| {
            let signs = &signs[(block * 1024) % signs.len()..][..1024];
            (0..1024).map(move |row| {
                let sum = values
                    .iter()
                    .enumerate()
                    .map(|(column, &value)| {
                        let walsh = if (row & column).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        value
                            * walsh
                            * if inverse {
                                1.0
                            } else {
                                f64::from(signs[column])
                            }
                    })
                    .sum::<f64>()
                    / 32.0;
                sum * if inverse { f64::from(signs[row]) } else { 1.0 }
            })
        })
        .collect()
}

fn guarded(context: &MetalContext, elements: usize) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), &vec![SENTINEL; elements + GUARD])
        .expect("guarded output")
}

fn assert_guards(buffer: &MetalBuffer, elements: usize) {
    assert!(
        buffer.as_slice::<u32>()[elements..]
            .iter()
            .all(|&v| v == SENTINEL)
    );
}

fn assert_close(actual: f32, expected: f64, tolerance: f64) {
    assert!(actual.is_finite());
    assert!(
        (f64::from(actual) - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

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

fn setup() -> Option<(MetalContext, BonsaiKernels)> {
    let context = gpu_or_skip()?;
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let kernels = BonsaiKernels::new(&context, &shaders).expect("Bonsai kernels");
    Some((context, kernels))
}

fn prefill_kernels(context: &MetalContext) -> Vec<BonsaiKernels> {
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    [PrefillKernel::SimdF32, PrefillKernel::TensorF32]
        .into_iter()
        .filter_map(|kernel| {
            BonsaiKernels::new_with_prefill_kernel(context, &shaders, kernel)
                .map(|kernels| kernels.with_small_batch_max(1))
                .map_err(|error| eprintln!("{} prefill unavailable: {error}", kernel.name()))
                .ok()
        })
        .collect()
}

#[test]
fn row_decoder_preserves_all_ternary_combinations_and_independent_group_scales() {
    // Every base-three rank at every lane, with asymmetric neighboring groups.
    let mut packed = Vec::new();
    let mut expected = Vec::new();
    for block in 0..243 {
        let mut trits = [0_i8; 128];
        for (start, width, digits) in [(0, 16, 5), (80, 8, 5), (120, 2, 4)] {
            for lane in 0..width {
                let mut rank = (block + lane * 71 + start) % 3_usize.pow(digits as u32);
                for digit in (0..digits).rev() {
                    trits[start + digit * width + lane] = (rank % 3) as i8 - 1;
                    rank /= 3;
                }
            }
        }
        let scale = [f16::from_f32(0.375), f16::from_f32(-1.625), f16::ZERO][block % 3];
        packed.extend_from_slice(&encode_trits(&trits, scale));
        expected.extend(trits.map(|trit| f32::from(trit) * scale.to_f32()));
    }
    let mut actual = vec![f32::NAN; expected.len()];
    decode_ptq1_row(&packed, &mut actual).expect("decode");
    assert_eq!(actual, expected);
    assert_eq!(
        reference_decode(&packed),
        expected.iter().copied().map(f64::from).collect::<Vec<_>>()
    );
    for (packed_bytes, elements) in [(0, 0), (28, 127), (27, 128), (29, 128), (56, 128)] {
        assert!(decode_ptq1_row(&vec![0; packed_bytes], &mut vec![0.0; elements]).is_err());
    }
}

#[test]
fn cpu_and_gpu_decoders_cover_every_byte_code_at_every_element() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    let mut packed = Vec::new();
    for code in 0..256 {
        let mut block = [0_u8; 28];
        for (position, byte) in block[..26].iter_mut().enumerate() {
            *byte = (code + position * 37) as u8;
        }
        // Nonzero for every code: a zero scale would hide a decode error.
        let sign = if code % 2 == 0 { -1.0 } else { 1.0 };
        let scale = f16::from_f32(sign * ((code % 53 + 1) as f32 / 32.0));
        block[26..].copy_from_slice(&scale.to_bits().to_le_bytes());
        packed.extend_from_slice(&block);
    }
    let expected = reference_decode(&packed);
    let mut cpu = vec![f32::NAN; expected.len()];
    decode_ptq1_row(&packed, &mut cpu).expect("CPU decode");
    assert_eq!(
        cpu.iter().copied().map(f64::from).collect::<Vec<_>>(),
        expected
    );

    // Blocks start at a word-aligned offset but are not 16-byte aligned. Prefix/
    // suffix bytes must not be mistaken for a block, and output guards must not
    // change.
    let mut data = vec![0xcd; 4];
    data.extend_from_slice(&packed);
    data.extend_from_slice(&[0xba; 30]);
    let weights = MetalBuffer::from_slice(context.device(), &data).expect("weights");
    let matrix = Ptq1Matrix::new(&weights, 4, 256, 128).expect("matrix");
    let outputs = (0..128).map(|_| guarded(&context, 256)).collect::<Vec<_>>();
    let mut batch = CommandBatch::new(&context).expect("batch");
    for (element, output) in outputs.iter().enumerate() {
        let mut basis = [0.0_f32; 128];
        basis[element] = 1.0;
        let input = MetalBuffer::from_slice(context.device(), &basis).expect("basis");
        kernels
            .matvec(&mut batch, matrix, &input, output)
            .expect("packed matvec");
    }
    batch.commit_and_wait().expect("GPU completion");
    for (element, output) in outputs.iter().enumerate() {
        assert_guards(output, 256);
        for (row, &actual) in output.as_slice::<f32>()[..256].iter().enumerate() {
            assert_close(actual, expected[row * 128 + element], 0.0);
        }
    }
    // The small-batch decoders too: consecutive basis rows as one block of
    // each exact-token kernel and of the wide kernel.
    for tokens in [2_usize, 3, 4, 5, 8] {
        let starts = (0..128 - tokens)
            .step_by(tokens)
            .chain([128 - tokens])
            .collect::<Vec<_>>();
        let outputs = starts
            .iter()
            .map(|_| guarded(&context, tokens * 256))
            .collect::<Vec<_>>();
        let mut batch = CommandBatch::new(&context).expect("batch");
        for (&start, output) in starts.iter().zip(&outputs) {
            let mut basis = vec![0.0_f32; tokens * 128];
            for token in 0..tokens {
                basis[token * 128 + start + token] = 1.0;
            }
            let input = MetalBuffer::from_slice(context.device(), &basis).expect("basis");
            kernels
                .matmul(&mut batch, matrix, &input, output, tokens as u32)
                .expect("small batch");
        }
        batch.commit_and_wait().expect("GPU completion");
        for (&start, output) in starts.iter().zip(&outputs) {
            assert_guards(output, tokens * 256);
            let values = output.as_slice::<f32>();
            for token in 0..tokens {
                for row in 0..256 {
                    assert_close(
                        values[token * 256 + row],
                        expected[row * 128 + start + token],
                        0.0,
                    );
                }
            }
        }
    }
    assert_eq!(weights.as_slice::<u8>(), data);
}

#[test]
fn signed_fwht_matches_dense_walsh_reference_and_true_inverse() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    let columns = 2048;
    let tokens = 3;
    let signs = signs(columns);
    let rotation = SignedHadamard::new(&context, &signs).expect("signs");
    assert_eq!(rotation.columns(), columns as u32);
    let values = activations(columns * tokens);
    let reference_values = values.iter().copied().map(f64::from).collect::<Vec<_>>();
    let expected_forward = reference_transform(&reference_values, &signs, false);
    let expected_inverse = reference_transform(&reference_values, &signs, true);
    let input = MetalBuffer::from_slice(context.device(), &values).expect("input");
    let forward = guarded(&context, values.len());
    let inverse = guarded(&context, values.len());
    let roundtrip = guarded(&context, values.len());
    let wrong_inverse = guarded(&context, values.len());
    let mut batch = CommandBatch::new(&context).expect("batch");
    for (source, destination, direction) in [
        (&input, &forward, HadamardDirection::Forward),
        (&input, &inverse, HadamardDirection::Inverse),
        (&forward, &roundtrip, HadamardDirection::Inverse),
        (&forward, &wrong_inverse, HadamardDirection::Forward),
    ] {
        kernels
            .transform(
                &mut batch,
                &rotation,
                source,
                destination,
                tokens as u32,
                direction,
            )
            .expect("transform");
    }
    batch.commit_and_wait().expect("completion");
    for (buffer, reference) in [
        (&forward, &expected_forward),
        (&inverse, &expected_inverse),
        (&roundtrip, &reference_values),
    ] {
        assert_guards(buffer, values.len());
        for (&actual, &expected) in buffer.as_slice::<f32>().iter().zip(reference) {
            assert_close(actual, expected, 2e-5);
        }
    }
    assert_guards(&wrong_inverse, values.len());
    assert!(
        wrong_inverse
            .as_slice::<f32>()
            .iter()
            .zip(&values)
            .filter(|(a, b)| (*a - *b).abs() > 0.1)
            .count()
            > values.len() / 2
    );

    // No hidden scratch allocation is required for reconstruction in place.
    let mut batch = CommandBatch::new(&context).expect("in-place batch");
    for direction in [HadamardDirection::Forward, HadamardDirection::Inverse] {
        kernels
            .transform(
                &mut batch,
                &rotation,
                &input,
                &input,
                tokens as u32,
                direction,
            )
            .expect("in-place transform");
    }
    batch.commit_and_wait().expect("in-place completion");
    for (&actual, &expected) in input.as_slice::<f32>().iter().zip(&reference_values) {
        assert_close(actual, expected, 2e-5);
    }
}

fn packed_fixture(rows: usize, columns: usize, salt: u32) -> Vec<u8> {
    let mut packed = Vec::with_capacity(rows * columns / 128 * 28);
    for block in 0..rows * columns / 128 {
        let seed = hash(block as u32 ^ salt);
        let mut bytes = [0_u8; 28];
        for (lane, byte) in bytes[..24].iter_mut().enumerate() {
            let rank = hash(seed.wrapping_add(lane as u32)) % 243;
            *byte = (rank * 256).div_ceil(243) as u8;
        }
        for (lane, byte) in bytes[24..26].iter_mut().enumerate() {
            let rank = hash(seed.wrapping_add(31 + lane as u32)) % 81;
            *byte = (rank * 3 * 256).div_ceil(243) as u8;
        }
        let scale = f16::from_f32((seed % 53 + 1) as f32 / 2048.0);
        bytes[26..].copy_from_slice(&scale.to_bits().to_le_bytes());
        packed.extend_from_slice(&bytes);
    }
    packed
}

#[test]
fn rotated_matvec_matches_unrotated_weights_and_real_projection_widths() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    for (rows, columns) in [(1, 1024), (3, 2048), (7, 5120), (9, 17408)] {
        let packed = packed_fixture(rows, columns, 19);
        let decoded = reference_decode(&packed);
        let signs = signs(columns);
        let rotation = SignedHadamard::new(&context, &signs).expect("rotation");
        let values = activations(columns);
        let values64 = values.iter().copied().map(f64::from).collect::<Vec<_>>();
        let rotated64 = reference_transform(&values64, &signs, false);
        let weights = MetalBuffer::from_slice(context.device(), &packed).expect("weights");
        let matrix = Ptq1Matrix::new(&weights, 0, rows as u32, columns as u32).expect("matrix");
        let input = MetalBuffer::from_slice(context.device(), &values).expect("input");
        let rotated = guarded(&context, columns);
        let output = guarded(&context, rows);
        let mut batch = CommandBatch::new(&context).expect("batch");
        kernels
            .transform(
                &mut batch,
                &rotation,
                &input,
                &rotated,
                1,
                HadamardDirection::Forward,
            )
            .expect("activation transform");
        kernels
            .matvec(&mut batch, matrix, &rotated, &output)
            .expect("matvec");
        batch.commit_and_wait().expect("completion");
        assert_guards(&rotated, columns);
        for (row, weights) in decoded.chunks_exact(columns).enumerate() {
            let expected = weights
                .iter()
                .zip(&rotated64)
                .map(|(w, x)| w * x)
                .sum::<f64>();
            // An independent unrotated-weight calculation also detects using HD
            // instead of DH to reconstruct token-embedding rows.
            if columns <= 2048 {
                let original_weights = reference_transform(weights, &signs, true);
                let unrotated = original_weights
                    .iter()
                    .zip(&values64)
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
                assert!((unrotated - expected).abs() < 1e-9);
            }
            assert_guards(&output, rows);
            assert_close(
                output.as_slice::<f32>()[row],
                expected,
                f64::mul_add(expected.abs(), 1e-5, 3e-5),
            );
        }
    }
}

#[test]
fn matvec_handles_partial_row_groups_and_partial_four_block_iterations() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    // Widths deliberately not restricted to 1024: the four-block work layout
    // must also handle fewer than four blocks and incomplete final iterations.
    for (rows, columns) in [(1, 128), (2, 256), (5, 384), (6, 640), (17, 896)] {
        let packed = packed_fixture(rows, columns, 127);
        let decoded = reference_decode(&packed);
        let mut data = vec![0xcd; 12];
        data.extend_from_slice(&packed);
        data.extend_from_slice(&[0xba; 30]);
        let weights = MetalBuffer::from_slice(context.device(), &data).expect("weights");
        let matrix = Ptq1Matrix::new(&weights, 12, rows as u32, columns as u32).expect("matrix");
        let values = activations(columns);
        let mut padded = values.clone();
        padded.extend_from_slice(&[f32::NAN; 512]);
        let input = MetalBuffer::from_slice(context.device(), &padded).expect("input");
        let output = guarded(&context, rows);
        let mut batch = CommandBatch::new(&context).expect("batch");
        kernels
            .matvec(&mut batch, matrix, &input, &output)
            .expect("matvec");
        batch.commit_and_wait().expect("completion");
        for (row, weights) in decoded.chunks_exact(columns).enumerate() {
            let expected = weights
                .iter()
                .zip(&values)
                .map(|(w, x)| w * f64::from(*x))
                .sum::<f64>();
            assert_guards(&output, rows);
            assert_close(
                output.as_slice::<f32>()[row],
                expected,
                f64::mul_add(expected.abs(), 1e-5, 3e-5),
            );
        }
        assert_eq!(weights.as_slice::<u8>(), data);
    }
}

#[test]
fn packed_matmul_preserves_f32_inputs_and_all_tile_tails() {
    let Some((context, _)) = setup() else {
        return;
    };
    // Tile tails are the subject; small-batch shapes have their own test. Exercise
    // both production prefill implementations directly against the F64 reference.
    let kernels = prefill_kernels(&context);
    for (tokens, rows, columns) in [
        (1, 3, 128),
        (2, 1, 128),
        (3, 7, 384),
        (7, 31, 5120),
        (8, 32, 1024),
        (9, 33, 2048),
        (17, 65, 896),
        (9, 7, 17408),
        (31, 7, 128),
        (32, 33, 256),
        (33, 3, 5120),
        (63, 7, 128),
        (64, 32, 128),
        (65, 33, 256),
        (127, 3, 5120),
        (128, 5, 1024),
        (129, 65, 256),
        (129, 129, 128),
    ] {
        let packed = packed_fixture(rows, columns, 913);
        let decoded = reference_decode(&packed);
        // Multi-token paths accept a half-aligned view; one token is a matvec,
        // which needs a word-aligned one.
        let offset = if tokens == 1 { 8 } else { 6 };
        let mut data = vec![0xcd; offset];
        data.extend_from_slice(&packed);
        data.extend_from_slice(&[0xba; 30]);
        let weights = MetalBuffer::from_slice(context.device(), &data).expect("weights");
        let matrix =
            Ptq1Matrix::new(&weights, offset, rows as u32, columns as u32).expect("matrix");
        // Non-dyadic inputs distinguish F32 arithmetic from half-rounded operands.
        let mut values = activations(tokens * columns)
            .into_iter()
            .map(|value| value / 1.37 + 0.01)
            .collect::<Vec<_>>();
        // F16 operands would overflow; no hidden precision reduction is allowed.
        values[5] = 100_000.0;
        values[columns - 7] = -75_000.0;
        let mut padded = values.clone();
        padded.extend_from_slice(&[f32::NAN; 512]);
        let input = MetalBuffer::from_slice(context.device(), &padded).expect("input");
        let outputs = (0..kernels.len())
            .map(|_| guarded(&context, tokens * rows))
            .collect::<Vec<_>>();
        let mut batch = CommandBatch::new(&context).expect("batch");
        for (kernels, output) in kernels.iter().zip(&outputs) {
            kernels
                .matmul(&mut batch, matrix, &input, output, tokens as u32)
                .expect("matmul");
        }
        batch.commit_and_wait().expect("completion");
        for (token, input_row) in values.chunks_exact(columns).enumerate() {
            for (row, weights) in decoded.chunks_exact(columns).enumerate() {
                let expected = weights
                    .iter()
                    .zip(input_row)
                    .map(|(weight, &input)| weight * f64::from(input))
                    .sum::<f64>();
                // Large cancelling terms amplify F32 summation roundoff even
                // in the control kernels; bound that by product magnitudes,
                // not just the small residual. Half conversion is tested below.
                let magnitude = weights
                    .iter()
                    .zip(input_row)
                    .map(|(weight, &input)| (weight * f64::from(input)).abs())
                    .sum::<f64>();
                let tolerance = f64::mul_add(expected.abs(), 2e-5, 3e-4)
                    .max(8.0 * f64::from(f32::EPSILON) * magnitude);
                for (kernels, output) in kernels.iter().zip(&outputs) {
                    let actual = f64::from(output.as_slice::<f32>()[token * rows + row]);
                    let name = kernels.prefill_kernel().name();
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= tolerance,
                        "{name}: shape ({tokens}, {rows}, {columns}), token {token}, row {row}: {actual} != {expected}, tolerance {tolerance}",
                    );
                }
            }
        }
        for output in &outputs {
            assert_guards(output, tokens * rows);
        }
        assert_eq!(weights.as_slice::<u8>(), data);
    }
}

#[test]
fn matmul_keeps_f32_operand_bits_and_accumulates_beyond_f16_range() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    // The tile path is the subject; the small-batch route is checked below.
    let kernels = kernels.with_small_batch_max(1);
    let mut selected = [0_i8; 128];
    selected[7] = 1;
    let mut packed = encode_trits(&selected, f16::from_f32(2048.0)).to_vec();
    packed.extend_from_slice(&encode_trits(&[1_i8; 128], f16::from_f32(2.0)));
    let weights = MetalBuffer::from_slice(context.device(), &packed).expect("weights");
    let matrix = Ptq1Matrix::new(&weights, 0, 2, 128).expect("matrix");
    let mut values = (0..3 * 128)
        .map(|i| 1000.125 + i as f32 / 31.0)
        .collect::<Vec<_>>();
    for token in 0..3 {
        values[token * 128 + 7] = (token + 1) as f32 * 1.000_3;
    }
    let input = MetalBuffer::from_slice(context.device(), &values).expect("input");
    let output = guarded(&context, 3 * 2);
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .matmul(&mut batch, matrix, &input, &output, 3)
        .expect("matmul");
    batch.commit_and_wait().expect("completion");
    for token in 0..3 {
        let expected = f64::from(values[token * 128 + 7]) * 2048.0;
        assert_close(
            output.as_slice::<f32>()[token * 2],
            expected,
            2e-6 * expected,
        );
        let sum = 2.0
            * values[token * 128..(token + 1) * 128]
                .iter()
                .copied()
                .map(f64::from)
                .sum::<f64>();
        assert!(sum > 65_504.0);
        assert_close(output.as_slice::<f32>()[token * 2 + 1], sum, 2e-6 * sum);
    }
    assert_guards(&output, 3 * 2);

    // Small-batch decodes the actual trits before the dot product. The former
    // telescoped factors lost about 72.6 on this lone-trit row; the remaining
    // decode matvec is retained as a diagnostic, not as the expected answer.
    // Row 1 distinguishes F32 from F16 operands (spacing at 1000 is 0.5)
    // and must accumulate beyond half's range.
    let small = kernels_with_small_batch_max(&context, 8);
    let small_output = guarded(&context, 3 * 2);
    let matvec_output = guarded(&context, 2);
    let mut batch = CommandBatch::new(&context).expect("batch");
    small
        .matmul(&mut batch, matrix, &input, &small_output, 3)
        .expect("small batch");
    small
        .matvec(&mut batch, matrix, &input, &matvec_output)
        .expect("matvec");
    batch.commit_and_wait().expect("completion");
    for token in 0..3 {
        let sum = 2.0
            * values[token * 128..(token + 1) * 128]
                .iter()
                .copied()
                .map(f64::from)
                .sum::<f64>();
        assert_close(
            small_output.as_slice::<f32>()[token * 2 + 1],
            sum,
            2e-6 * sum,
        );
        let expected = f64::from(values[token * 128 + 7]) * 2048.0;
        assert_close(
            small_output.as_slice::<f32>()[token * 2],
            expected,
            2e-6 * expected,
        );
    }
    eprintln!(
        "offset-heavy lone-trit row: small batch {} vs decode matvec {} vs exact {}",
        small_output.as_slice::<f32>()[0],
        matvec_output.as_slice::<f32>()[0],
        f64::from(values[7]) * 2048.0
    );
    assert_guards(&small_output, 3 * 2);
}

#[test]
fn rejects_invalid_geometry_signs_aliasing_and_short_buffers_before_dispatch() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    assert_eq!(PTQ1_BLOCK_ELEMENTS, 128);
    assert_eq!(PTQ1_BLOCK_BYTES, 28);
    assert_eq!(HADAMARD_BLOCK_ELEMENTS, 1024);
    let weights = MetalBuffer::empty(context.device(), 64).expect("weights");
    for (offset, rows, columns) in [
        (0, 0, 128),
        (0, 1, 0),
        (0, 1, 127),
        (0, 1, 129),
        (1, 1, 128),
        (38, 1, 128),
        (usize::MAX - 1, 1, 128),
        (0, u32::MAX, u32::MAX - 127),
    ] {
        assert!(Ptq1Matrix::new(&weights, offset, rows, columns).is_err());
    }
    for columns in [0, 1023, 1025] {
        assert!(SignedHadamard::new(&context, &vec![1.0; columns]).is_err());
    }
    for value in [0.0, -0.0, 0.5, f32::NAN, f32::INFINITY] {
        let mut signs = vec![1.0; 1024];
        signs[917] = value;
        assert!(SignedHadamard::new(&context, &signs).is_err());
    }
    let rotation = SignedHadamard::new(&context, &signs(1024)).expect("rotation");
    let input = MetalBuffer::empty(context.device(), 4096).expect("input");
    let output = MetalBuffer::empty(context.device(), 4096).expect("output");
    let short = MetalBuffer::empty(context.device(), 4092).expect("short");
    let mut batch = CommandBatch::new(&context).expect("batch");
    for (source, destination, tokens) in [
        (&input, &output, 0),
        (&input, &output, u32::MAX),
        (&short, &output, 1),
        (&input, &short, 1),
    ] {
        assert!(
            kernels
                .transform(
                    &mut batch,
                    &rotation,
                    source,
                    destination,
                    tokens,
                    HadamardDirection::Forward
                )
                .is_err()
        );
    }
    let matrix = Ptq1Matrix::new(&weights, 2, 2, 128).expect("matrix");
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 128);
    let short_input = MetalBuffer::empty(context.device(), 508).expect("short input");
    let short_output = MetalBuffer::empty(context.device(), 4).expect("short output");
    for (source, destination) in [
        (&short_input, &output),
        (&input, &short_output),
        (&input, &input),
        (&input, &weights),
    ] {
        assert!(
            kernels
                .matvec(&mut batch, matrix, source, destination)
                .is_err()
        );
    }
    let alias = input.clone();
    assert!(kernels.matvec(&mut batch, matrix, &input, &alias).is_err());
    // Half-aligned views are valid, but the single-token kernels read words.
    assert!(kernels.matvec(&mut batch, matrix, &input, &output).is_err());
    assert!(
        kernels
            .matvec_swiglu(&mut batch, matrix, matrix, &input, &output)
            .is_err()
    );
    assert!(
        kernels
            .matvec_concat(&mut batch, &[(matrix, &output)], &input)
            .is_err()
    );
    for (source, destination, tokens) in [
        (&input, &output, 0),
        (&input, &output, u32::MAX),
        (&short_input, &output, 2),
        (&input, &short_output, 2),
        (&input, &alias, 2),
        (&input, &weights, 2),
    ] {
        assert!(
            kernels
                .matmul(&mut batch, matrix, source, destination, tokens)
                .is_err()
        );
    }
    assert_eq!(batch.dispatch_count(), 0);
}

fn kernels_with_small_batch_max(context: &MetalContext, tokens: u32) -> BonsaiKernels {
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    BonsaiKernels::new(context, &shaders)
        .expect("Bonsai kernels")
        .with_small_batch_max(tokens)
}

// Every exact-token kernel, the wide kernel alone and with each scalar tail
// (9 -> 7 + 2, 12 -> 8 + 4, 17 -> 8 + 7 + 2), partial row groups, partial
// four-block iterations and real projection widths. The
// small-batch path repeats the decode matvec's arithmetic per token, but the
// compiler contracts the per-token FMAs differently, so results are not bit
// identical (first run: 3.7e-7 gap on a 1.4-magnitude row); the gap is bounded
// by the reference tolerance and the worst observed value is printed. It must also match the F64
// reference and agree with the prefill tile within its tolerance.
#[test]
fn small_batch_matches_tokenwise_matvec_reference_and_tile_for_all_chunks() {
    let Some((context, kernels)) = setup() else {
        return;
    };
    let small = kernels_with_small_batch_max(&context, 64);
    let tile = kernels_with_small_batch_max(&context, 1);
    assert_eq!(tile.small_batch_max(), 1);
    let mut worst_matvec_ulps = 0.0_f64;
    let mut worst_matvec_gap = 0.0_f64;
    for (tokens, rows, columns) in [
        (2, 1, 128),
        (3, 7, 384),
        (4, 5, 640),
        (5, 33, 256),
        (6, 3, 5120),
        (7, 31, 896),
        (8, 32, 1024),
        (9, 7, 2048),
        (12, 5, 384),
        (16, 2, 128),
        (17, 65, 640),
        (23, 3, 128),
        (64, 7, 256),
    ] {
        let packed = packed_fixture(rows, columns, 431);
        let decoded = reference_decode(&packed);
        let mut data = vec![0xcd; 6];
        data.extend_from_slice(&packed);
        data.extend_from_slice(&[0xba; 30]);
        let weights = MetalBuffer::from_slice(context.device(), &data).expect("weights");
        let matrix = Ptq1Matrix::new(&weights, 6, rows as u32, columns as u32).expect("matrix");
        // The matvec reads words, so it gets a word-aligned copy.
        let aligned = MetalBuffer::from_slice(context.device(), &data[2..]).expect("aligned");
        let word_matrix =
            Ptq1Matrix::new(&aligned, 4, rows as u32, columns as u32).expect("word matrix");
        // Token-dependent, non-dyadic values so rows are distinguishable, with
        // outliers wide enough to expose row/offset mixups. F16 operand
        // rounding and beyond-half sums are covered by the tests above; larger
        // outliers push the reuse trick's 243x intermediates past F32's exact
        // integer range and merely measure shared roundoff.
        let mut values = activations(tokens * columns)
            .into_iter()
            .enumerate()
            .map(|(i, value)| ((i / columns) as f32).mul_add(0.11, value / 1.37 + 0.01))
            .collect::<Vec<_>>();
        values[columns - 1] = 900.0;
        values[tokens * columns - 3] = -700.0;
        let mut padded = values.clone();
        padded.extend_from_slice(&[f32::NAN; 512]);
        let input = MetalBuffer::from_slice(context.device(), &padded).expect("input");
        let token_inputs = values
            .chunks_exact(columns)
            .map(|row| MetalBuffer::from_slice(context.device(), row).expect("token input"))
            .collect::<Vec<_>>();
        let small_output = guarded(&context, tokens * rows);
        let tile_output = guarded(&context, tokens * rows);
        let matvec_outputs = (0..tokens)
            .map(|_| guarded(&context, rows))
            .collect::<Vec<_>>();
        let mut batch = CommandBatch::new(&context).expect("batch");
        small
            .matmul(&mut batch, matrix, &input, &small_output, tokens as u32)
            .expect("small batch");
        tile.matmul(&mut batch, matrix, &input, &tile_output, tokens as u32)
            .expect("tile");
        for (token_input, output) in token_inputs.iter().zip(&matvec_outputs) {
            kernels
                .matvec(&mut batch, word_matrix, token_input, output)
                .expect("matvec");
        }
        batch.commit_and_wait().expect("completion");
        for (token, input_row) in values.chunks_exact(columns).enumerate() {
            for (row, weights) in decoded.chunks_exact(columns).enumerate() {
                let actual = small_output.as_slice::<f32>()[token * rows + row];
                let tokenwise = matvec_outputs[token].as_slice::<f32>()[row];
                let expected = weights
                    .iter()
                    .zip(input_row)
                    .map(|(weight, &input)| weight * f64::from(input))
                    .sum::<f64>();
                let magnitude = weights
                    .iter()
                    .zip(input_row)
                    .map(|(weight, &input)| (weight * f64::from(input)).abs())
                    .sum::<f64>();
                let ulp = f64::from(f32::EPSILON) * magnitude;
                let tolerance = f64::mul_add(expected.abs(), 2e-5, 3e-4).max(8.0 * ulp);
                // The reuse trick sums terms up to 243x larger than the products,
                // so both kernels carry roundoff well above one product ulp; the
                // gap is bounded by the reference tolerance and reported below.
                let matvec_gap = (f64::from(actual) - f64::from(tokenwise)).abs();
                worst_matvec_gap = worst_matvec_gap.max(matvec_gap);
                worst_matvec_ulps = worst_matvec_ulps.max(matvec_gap / ulp);
                assert!(
                    actual.is_finite() && matvec_gap <= tolerance,
                    "shape ({tokens}, {rows}, {columns}), token {token}, row {row}: small batch {actual} vs matvec {tokenwise}, gap {matvec_gap} > tolerance {tolerance}"
                );
                assert!(
                    (f64::from(actual) - expected).abs() <= tolerance,
                    "shape ({tokens}, {rows}, {columns}), token {token}, row {row}: {actual} != reference {expected}, tolerance {tolerance}"
                );
                let tiled = tile_output.as_slice::<f32>()[token * rows + row];
                for (name, value) in [("matvec", tokenwise), ("tile", tiled)] {
                    assert!(
                        value.is_finite() && (f64::from(value) - expected).abs() <= tolerance,
                        "shape ({tokens}, {rows}, {columns}), token {token}, row {row}: {name} {value} != reference {expected}, tolerance {tolerance}"
                    );
                }
            }
        }
        assert_guards(&small_output, tokens * rows);
        assert_guards(&tile_output, tokens * rows);
        assert_eq!(weights.as_slice::<u8>(), data);
    }
    eprintln!(
        "small batch vs tokenwise matvec: worst gap {worst_matvec_gap} abs, {worst_matvec_ulps} ulp of product magnitude"
    );
}
