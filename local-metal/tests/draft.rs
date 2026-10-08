#![allow(clippy::expect_used)]

use half::f16;
use local_metal::batch::CommandBatch;
use local_metal::bonsai::{
    BonsaiKernels, HadamardDirection, Ptq1Matrix, SignedHadamard, decode_ptq1_row,
};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::draft::{DraftKernels, DraftTopTwo, GreedyRows};
use local_metal::shaders::ShaderLibrary;

fn gpu_or_skip() -> Option<MetalContext> {
    let context = MetalContext::new();
    if matches!(context, Err(local_metal::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    Some(context.expect("Metal context failed for a reason other than a missing device"))
}

const fn hash(mut value: u32) -> u32 {
    value = (value ^ (value >> 16)).wrapping_mul(0x7feb_352d);
    value = (value ^ (value >> 15)).wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

/// Every byte value, not only canonical trit codes: the codec is defined on all.
fn packed_rows(rows: usize, columns: usize) -> Vec<u8> {
    let mut packed = Vec::new();
    for block in 0..rows * columns / 128 {
        let seed = hash(block as u32 + 5);
        let mut bytes = [0_u8; 28];
        for (lane, byte) in bytes[..26].iter_mut().enumerate() {
            *byte = hash(seed.wrapping_add(lane as u32)) as u8;
        }
        let scale =
            f16::from_f32((seed % 97 + 1) as f32 / 512.0 * if seed & 1 == 0 { 1.0 } else { -1.0 });
        bytes[26..].copy_from_slice(&scale.to_bits().to_le_bytes());
        packed.extend_from_slice(&bytes);
    }
    packed
}

#[test]
fn embed_inverse_matches_cpu_decode_then_inverse_rotation_bit_for_bit() {
    let Some(context) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let kernels = BonsaiKernels::new(&context, &shaders).expect("kernels");
    let draft = DraftKernels::new(&context, &shaders, 1000).expect("draft");
    let (rows, columns) = (37_usize, 5120_usize);
    let signs = (0..columns)
        .map(|i| {
            if hash(i as u32 + 71) & 1 == 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect::<Vec<f32>>();
    let rotation = SignedHadamard::new(&context, &signs).expect("signs");
    let mut padded = vec![0_u8; 2];
    padded.extend(packed_rows(rows, columns));
    let table = MetalBuffer::from_slice(context.device(), &padded).expect("table");
    let matrix = Ptq1Matrix::new(&table, 2, rows as u32, columns as u32).expect("matrix");
    let row_bytes = columns / 128 * 28;
    let tokens = MetalBuffer::from_slice(context.device(), &[3_u32, 0, 36, 99]).expect("tokens");
    for (index, token) in [3_usize, 0, 36, 99].into_iter().enumerate() {
        let mut decoded = vec![0.0_f32; columns];
        if token < rows {
            decode_ptq1_row(&padded[2 + token * row_bytes..][..row_bytes], &mut decoded)
                .expect("decode");
        }
        let staged = MetalBuffer::from_slice(context.device(), &decoded).expect("staged");
        let expected = MetalBuffer::empty(context.device(), columns * 4).expect("expected");
        let actual = MetalBuffer::empty(context.device(), columns * 4).expect("actual");
        let mut batch = CommandBatch::new(&context).expect("batch");
        kernels
            .transform(
                &mut batch,
                &rotation,
                &staged,
                &expected,
                1,
                HadamardDirection::Inverse,
            )
            .expect("transform");
        draft
            .embed_inverse(
                &mut batch,
                matrix,
                &rotation,
                &tokens,
                index as u32,
                &actual,
            )
            .expect("embed");
        batch.commit_and_wait().expect("wait");
        let expected = expected.as_slice::<f32>();
        let actual = actual.as_slice::<f32>();
        for column in 0..columns {
            assert_eq!(
                actual[column].to_bits(),
                expected[column].to_bits(),
                "token {token} column {column}"
            );
        }
    }
}

fn reference(values: &[f32]) -> DraftTopTwo {
    let mut ids: Vec<usize> = (0..values.len()).collect();
    ids.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
    DraftTopTwo {
        best_id: ids[0] as u32,
        best_logit: values[ids[0]],
        second_id: ids[1] as u32,
        second_logit: values[ids[1]],
    }
}

#[test]
fn top_two_matches_total_cmp_with_ties_and_chains_through_the_token_slot() {
    let Some(context) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let vocab_max = 248_320;
    let draft = DraftKernels::new(&context, &shaders, vocab_max).expect("draft");
    let cases: Vec<Vec<f32>> = vec![
        // Coarse values: many exact ties across groups and lanes.
        (0..vocab_max)
            .map(|i| (hash(i as u32) % 50) as f32)
            .collect(),
        (0..vocab_max)
            .map(|i| ((hash(i as u32 + 9) & 0xffff) as f32 - 32768.0) / 127.0)
            .collect(),
        // Winner in the last, partial group; negative zero against zero.
        {
            let mut v = vec![-1.0_f32; 3001];
            v[3000] = 2.0;
            v[17] = 0.0;
            v[16] = -0.0;
            v
        },
        {
            let mut v = vec![0.0_f32; 2];
            v[0] = -0.0;
            v
        },
        // Infinities and every value equal.
        {
            let mut v = vec![f32::NEG_INFINITY; 70_001];
            v[12_345] = f32::INFINITY;
            v[54_321] = f32::INFINITY;
            v
        },
        vec![1.5_f32; 248_320],
    ];
    let tokens = MetalBuffer::empty(context.device(), 4 * 8).expect("tokens");
    let results =
        MetalBuffer::empty(context.device(), size_of::<DraftTopTwo>() * 8).expect("results");
    for (index, values) in cases.iter().enumerate() {
        let logits = MetalBuffer::from_slice(context.device(), values).expect("logits");
        let mut batch = CommandBatch::new(&context).expect("batch");
        draft
            .top_two(
                &mut batch,
                &logits,
                values.len(),
                &tokens,
                index as u32 + 1,
                &results,
                index as u32,
            )
            .expect("top two");
        batch.commit_and_wait().expect("wait");
        let actual = results.as_slice::<DraftTopTwo>()[index];
        let expected = reference(values);
        assert_eq!(actual.best_id, expected.best_id, "case {index}");
        assert_eq!(actual.second_id, expected.second_id, "case {index}");
        assert_eq!(actual.best_logit.to_bits(), expected.best_logit.to_bits());
        assert_eq!(
            actual.second_logit.to_bits(),
            expected.second_logit.to_bits()
        );
        assert_eq!(tokens.as_slice::<u32>()[index + 1], expected.best_id);
    }
}

/// Every row of a block selects the host's `total_cmp` argmax (ties to the
/// lower id, signed zeros and infinities ordered), and a non-finite logit
/// anywhere in a row flags exactly that row.
#[test]
fn greedy_rows_match_total_cmp_argmax_and_flag_nonfinite_rows() {
    let Some(context) = gpu_or_skip() else { return };
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    // A vocabulary that leaves the last 1024-wide group partial.
    let (vocab, rows) = (248_320_usize, 7_usize);
    let selector = GreedyRows::new(&context, &shaders, vocab, 8).expect("greedy rows");
    let mut values = (0..rows * vocab)
        .map(|i| (hash(i as u32 + 3) % 20_000) as f32 / 1000.0 - 10.0)
        .collect::<Vec<f32>>();
    // Row 1: a tie at the maximum between a late and an early id.
    values[vocab + 200_000] = 50.0;
    values[vocab + 1_234] = 50.0;
    // Row 2: +0.0 beats -0.0 under total_cmp when everything else is negative.
    for value in &mut values[2 * vocab..3 * vocab] {
        *value = -value.abs() - 1.0;
    }
    values[2 * vocab + 9] = -0.0;
    values[2 * vocab + vocab - 1] = 0.0;
    // Row 3: +inf wins and is flagged; row 4: one NaN in the last group.
    values[3 * vocab + 77] = f32::INFINITY;
    values[4 * vocab + vocab - 3] = f32::NAN;
    // Row 5: the maximum in the partial final group.
    values[5 * vocab + vocab - 1] = 99.0;
    let logits = MetalBuffer::from_slice(context.device(), &values).expect("logits");
    let mut batch = CommandBatch::new(&context).expect("batch");
    selector.encode(&mut batch, &logits, rows).expect("encode");
    batch.commit_and_wait().expect("run");
    let results = selector.results(rows);
    for (row, result) in results.iter().enumerate() {
        let row_values = &values[row * vocab..(row + 1) * vocab];
        let expected = row_values
            .iter()
            .enumerate()
            .max_by(|(a_index, a), (b_index, b)| a.total_cmp(b).then_with(|| b_index.cmp(a_index)))
            .map(|(index, _)| index as u32)
            .expect("vocab");
        assert_eq!(result.best_id, expected, "row {row}");
        assert_eq!(
            result.best_logit.to_bits(),
            row_values[expected as usize].to_bits(),
            "row {row}"
        );
        let nonfinite = row_values.iter().any(|value| !value.is_finite());
        assert_eq!(result.nonfinite != 0, nonfinite, "row {row}");
    }
    assert_eq!(results[1].best_id, 1_234);
    assert_eq!(results[2].best_id, vocab as u32 - 1);
    assert_eq!(results[5].best_id, vocab as u32 - 1);
    assert!(
        selector
            .encode(&mut CommandBatch::new(&context).expect("batch"), &logits, 9)
            .is_err()
    );
}
