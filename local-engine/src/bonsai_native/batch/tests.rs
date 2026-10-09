use std::time::Instant;

use super::super::{BlockOutput, KvOptions, NgramSettings};
use super::*;
use crate::bonsai::{BonsaiPackage, DEFAULT_BONSAI_GGUF};

const PREFIXES: [&[u32]; 4] = [
    &[248_045, 846, 198, 814, 20139, 1204, 264, 5010, 2336, 13081],
    &[
        248_045, 846, 198, 3742, 220, 18, 24, 16, 264, 9944, 1324, 30,
    ],
    &[248_045, 846, 198, 45_776, 11, 264, 198, 13, 3010],
    &[248_045, 846, 198, 1204, 264, 2336, 5010, 814],
];
const STEPS: usize = 16;

fn load(context: usize, ngram: bool) -> BonsaiModel {
    let package = BonsaiPackage::open(DEFAULT_BONSAI_GGUF).expect("open Bonsai GGUF");
    BonsaiModel::load(
        package,
        context,
        16,
        None,
        None,
        NgramSettings {
            enabled: ngram,
            ..NgramSettings::default()
        },
        KvOptions::default(),
    )
    .expect("load")
}

/// Teacher-forced tokens for sequence `index` at step `step`.
fn forced(index: usize, step: usize) -> u32 {
    PREFIXES[index][(step * 3 + 1) % PREFIXES[index].len()] + step as u32
}

/// Fresh parked sequences, each prefilled with its prefix.
fn prefilled(model: &mut BonsaiModel, count: usize) -> Vec<SequenceState> {
    (0..count)
        .map(|index| {
            let mut state = model.new_sequence().expect("sequence");
            model.swap_sequence(&mut state).expect("swap in");
            model
                .forward_block(PREFIXES[index], BlockOutput::LastLogits)
                .expect("prefill");
            model.swap_sequence(&mut state).expect("swap out");
            state
        })
        .collect()
}

/// Logits of every step for the first `count` sequences, decoded alone
/// (`batched == false`) or all together.
fn trajectories(model: &mut BonsaiModel, count: usize, batched: bool) -> Vec<Vec<Vec<f32>>> {
    let mut states = prefilled(model, count);
    let mut logits = vec![Vec::new(); count];
    for step in 0..STEPS {
        if batched {
            let mut rows = states
                .iter_mut()
                .enumerate()
                .map(|(index, state)| BatchRow {
                    token: forced(index, step),
                    state: Some(state),
                    lag: None,
                })
                .collect::<Vec<_>>();
            model.decode_batch(&mut rows).expect("batched step");
            let output = model.batch.as_ref().expect("batch scratch");
            for (index, row) in logits.iter_mut().enumerate() {
                row.push(
                    output.logits.as_slice::<f32>()[index * VOCAB..(index + 1) * VOCAB].to_vec(),
                );
            }
        } else {
            for (index, state) in states.iter_mut().enumerate() {
                model.swap_sequence(state).expect("swap in");
                model.decode(forced(index, step)).expect("decode");
                logits[index].push(model.scratch.logits.as_slice::<f32>().to_vec());
                model.swap_sequence(state).expect("swap out");
            }
        }
    }
    logits
}

fn softmax(values: &[f32]) -> Vec<f64> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = values
        .iter()
        .map(|&value| f64::from(value - max).exp())
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f64>();
    exp.into_iter().map(|value| value / sum).collect()
}

fn kl(reference: &[f32], actual: &[f32]) -> f64 {
    let p = softmax(reference);
    let q = softmax(actual);
    p.iter()
        .zip(&q)
        .filter(|(p, _)| **p > 0.0)
        .map(|(p, q)| p * (p / q.max(1e-300)).ln())
        .sum()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(index, _)| index)
}

/// What batching changes numerically: one sequence's logits decoded alone
/// (single-row kernels) against decoded beside 1, 2 and 3 others
/// (multi-row kernels), teacher-forced so the comparison never diverges in
/// tokens. Rows of one batch are independent of each other: the same
/// sequence's logits are bitwise equal whatever else shares the batch, as
/// long as the row count selects the same projection kernels.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn batched_rows_track_single_row_decode() {
    let mut model = load(1024, false);
    let alone = trajectories(&mut model, 4, false);
    let mut report = Vec::new();
    for count in 2..=4 {
        let batched = trajectories(&mut model, count, true);
        let mut worst_kl = 0.0f64;
        let mut mean_kl = 0.0f64;
        let mut agree = 0;
        let mut compared = 0u32;
        let mut max_abs = 0.0f32;
        for (sequence, steps) in batched.iter().enumerate() {
            for (step, logits) in steps.iter().enumerate() {
                let reference = &alone[sequence][step];
                let divergence = kl(reference, logits);
                worst_kl = worst_kl.max(divergence);
                mean_kl += divergence;
                compared += 1;
                agree += u32::from(argmax(reference) == argmax(logits));
                max_abs = reference
                    .iter()
                    .zip(logits)
                    .map(|(a, b)| (a - b).abs())
                    .fold(max_abs, f32::max);
            }
        }
        mean_kl /= f64::from(compared);
        report.push(serde_json::json!({
            "rows": count, "mean_kl": mean_kl, "max_kl": worst_kl,
            "top1_agreement": format!("{agree}/{compared}"), "max_abs_logit": max_abs,
        }));
        assert!(mean_kl < 1e-3, "rows {count}: mean KL {mean_kl}");
        assert!(
            agree * 100 >= compared * 95,
            "rows {count}: top-1 {agree}/{compared}"
        );
    }
    // Row independence: sequence 0 beside one other or beside three others.
    let two = trajectories(&mut model, 2, true);
    let three = trajectories(&mut model, 3, true);
    let four = trajectories(&mut model, 4, true);
    let bitwise = |a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>| {
        a.iter()
            .zip(b)
            .all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()))
    };
    report.push(serde_json::json!({
        "sequence0_bitwise_2_vs_3": bitwise(&two[0], &three[0]),
        "sequence0_bitwise_2_vs_4": bitwise(&two[0], &four[0]),
        "sequence0_bitwise_3_vs_4": bitwise(&three[0], &four[0]),
    }));
    println!("{}", serde_json::Value::Array(report));
}

/// Step time of batched decode at 1..=8 sequences, and of one sequence's
/// block of `rows` rows (the cost model of a batched speculative verify).
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device; prints timings"]
fn batched_step_timings() {
    let mut model = load(4096, true);
    let repeats = 24u32;
    let mut lines = Vec::new();
    for count in 1..=MAX_BATCH_SEQUENCES {
        let mut states = prefilled(&mut model, count.min(4));
        while states.len() < count {
            states.push(model.new_sequence().expect("sequence"));
        }
        let mut elapsed = 0.0;
        for step in 0..repeats as usize + 4 {
            let mut rows = states
                .iter_mut()
                .enumerate()
                .map(|(index, state)| BatchRow {
                    token: 1000 + (index * 31 + step) as u32,
                    state: Some(state),
                    lag: None,
                })
                .collect::<Vec<_>>();
            let started = Instant::now();
            model.decode_batch(&mut rows).expect("batched step");
            if step >= 4 {
                elapsed += started.elapsed().as_secs_f64();
            }
        }
        let step = elapsed / f64::from(repeats);
        lines.push(serde_json::json!({
            "sequences": count, "step_ms": step * 1e3,
            "aggregate_tok_s": count as f64 / step,
        }));
    }
    let mut solo = Vec::new();
    model.reset();
    for _ in 0..4 {
        model.decode(1000).expect("warm");
    }
    let started = Instant::now();
    for step in 0..repeats {
        model.decode(1000 + step).expect("decode");
    }
    solo.push(serde_json::json!({
        "solo_decode_ms": started.elapsed().as_secs_f64() * 1e3 / f64::from(repeats),
    }));
    // Verify blocks project every row to logits, as a batched verify would.
    let mut verifier = model.ngram_verifier.take().expect("n-gram verifier");
    for rows in [2usize, 4, 5, 8, 10, 12, 16, 20, 24, 32] {
        let tokens = (0..rows).map(|row| 1000 + row as u32).collect::<Vec<_>>();
        verifier.reserve(&model.context, rows).expect("reserve");
        let mut elapsed = 0.0;
        for repeat in 0..8 {
            model.reset();
            model
                .forward_block(PREFIXES[0], BlockOutput::LastLogits)
                .expect("prefix");
            let started = Instant::now();
            model
                .forward_block(&tokens, BlockOutput::Verify(&verifier))
                .expect("block");
            if repeat >= 2 {
                elapsed += started.elapsed().as_secs_f64();
            }
        }
        solo.push(serde_json::json!({"verify_rows": rows, "verify_ms": elapsed / 6.0 * 1e3}));
    }
    model.ngram_verifier = Some(verifier);
    println!("{}", serde_json::json!({"batched": lines, "blocks": solo}));
}
