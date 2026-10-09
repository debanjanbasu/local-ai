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
                    drafts: Vec::new(),
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
                    drafts: Vec::new(),
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

/// Time of one prefill block of `rows` rows (no logits), the cost of one
/// interleaved prefill chunk beside batched decode.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device; prints timings"]
fn prefill_block_timings() {
    let package = BonsaiPackage::open(DEFAULT_BONSAI_GGUF).expect("open Bonsai GGUF");
    let mut model = BonsaiModel::load(
        package,
        8192,
        128,
        None,
        None,
        NgramSettings::default(),
        KvOptions::default(),
    )
    .expect("load");
    let mut lines = Vec::new();
    for rows in [8usize, 16, 32, 48, 64, 96, 128] {
        let tokens = (0..rows).map(|row| 1000 + row as u32).collect::<Vec<_>>();
        let mut elapsed = 0.0;
        for repeat in 0..6 {
            model.reset();
            model
                .forward_block(&tokens, BlockOutput::None)
                .expect("warm block");
            let started = Instant::now();
            model
                .forward_block(&tokens, BlockOutput::None)
                .expect("block");
            if repeat >= 1 {
                elapsed += started.elapsed().as_secs_f64();
            }
        }
        let block = elapsed / 5.0;
        lines.push(serde_json::json!({
            "rows": rows, "block_ms": block * 1e3, "tok_s": rows as f64 / block,
        }));
    }
    println!("{}", serde_json::json!({"prefill_blocks": lines}));
}

/// Seeds and drafts verified for three sequences in one batched pass: one
/// with four rows, one with a single row, one with three. Each verify row's
/// logits track the same sequence's own verify block, and after each keeps
/// the rows a commit names (a partial commit, a plain row and a full
/// commit), the next batched step tracks each sequence decoded alone from
/// the same kept rows, so the replayed state and positions are exact.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn batched_verify_tracks_each_sequence_alone() {
    let mut model = load(1024, true);
    let drafts: [&[u32]; 3] = [&[264, 5010, 2336], &[], &[11, 13]];
    let kept = [2usize, 1, 3];
    let next = [3010u32, 846, 198];
    // Reference: each sequence alone, verify block then kept rows then one step.
    let mut verifier = model.ngram_verifier.take().expect("n-gram verifier");
    let mut reference = Vec::new();
    let mut states = prefilled(&mut model, 3);
    for (index, state) in states.iter_mut().enumerate() {
        model.swap_sequence(state).expect("swap in");
        let inputs = std::iter::once(forced(index, 0))
            .chain(drafts[index].iter().copied())
            .collect::<Vec<_>>();
        let start = model.position;
        let mut rows = Vec::new();
        if inputs.len() > 1 {
            verifier
                .reserve(&model.context, inputs.len())
                .expect("reserve");
            model
                .forward_block(&inputs, BlockOutput::Verify(&verifier))
                .expect("verify block");
            for row in 0..inputs.len() {
                rows.push(
                    verifier.verify_logits.as_slice::<f32>()[row * VOCAB..(row + 1) * VOCAB]
                        .to_vec(),
                );
            }
            model
                .commit_verified(&mut verifier, inputs.len(), kept[index])
                .expect("commit");
            model.position = start + kept[index];
        } else {
            model.decode(inputs[0]).expect("decode");
            rows.push(model.scratch.logits.as_slice::<f32>()[..VOCAB].to_vec());
        }
        model.decode(next[index]).expect("next");
        let after = model.scratch.logits.as_slice::<f32>()[..VOCAB].to_vec();
        model.swap_sequence(state).expect("swap out");
        reference.push((rows, after));
    }
    model.ngram_verifier = Some(verifier);
    // Batched: the same rows in one pass, the same commits, one plain step.
    let mut states = prefilled(&mut model, 3);
    let mut rows = states
        .iter_mut()
        .enumerate()
        .map(|(index, state)| BatchRow {
            token: forced(index, 0),
            drafts: drafts[index].to_vec(),
            state: Some(state),
            lag: None,
        })
        .collect::<Vec<_>>();
    model.decode_batch(&mut rows).expect("batched verify");
    let mut worst = 0.0f64;
    for (index, (expected, _)) in reference.iter().enumerate() {
        let start = model.batch_row_start(index).expect("start");
        let output = model.batch.as_ref().expect("batch scratch");
        for (row, expected) in expected.iter().enumerate() {
            let actual =
                &output.logits.as_slice::<f32>()[(start + row) * VOCAB..(start + row + 1) * VOCAB];
            worst = worst.max(kl(expected, actual));
            assert_eq!(
                argmax(expected),
                argmax(actual),
                "sequence {index} row {row}"
            );
        }
    }
    model.commit_batch(&mut rows, &kept).expect("commit");
    drop(rows);
    for (index, state) in states.iter().enumerate() {
        assert_eq!(state.position, PREFIXES[index].len() + kept[index]);
    }
    let mut rows = states
        .iter_mut()
        .enumerate()
        .map(|(index, state)| BatchRow {
            token: next[index],
            drafts: Vec::new(),
            state: Some(state),
            lag: None,
        })
        .collect::<Vec<_>>();
    model.decode_batch(&mut rows).expect("batched step");
    let output = model.batch.as_ref().expect("batch scratch");
    for (index, (_, expected)) in reference.iter().enumerate() {
        let actual = &output.logits.as_slice::<f32>()[index * VOCAB..(index + 1) * VOCAB];
        worst = worst.max(kl(expected, actual));
        assert_eq!(
            argmax(expected),
            argmax(actual),
            "sequence {index} after commit"
        );
    }
    println!("{}", serde_json::json!({"batched_verify_max_kl": worst}));
    assert!(worst < 1e-3, "max KL {worst}");
}

/// Time of one batched pass that verifies drafts: `sequences` sequences of
/// `1 + drafts` rows each, then the commit of every row (the cost model of
/// batched speculation).
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device; prints timings"]
fn batched_verify_timings() {
    let package = BonsaiPackage::open(DEFAULT_BONSAI_GGUF).expect("open Bonsai GGUF");
    let mut model = BonsaiModel::load(
        package,
        4096,
        128,
        None,
        None,
        NgramSettings::default(),
        KvOptions::default(),
    )
    .expect("load");
    let mut lines = Vec::new();
    for (sequences, drafts) in [
        (1, 0),
        (2, 0),
        (4, 0),
        (8, 0),
        (2, 1),
        (2, 3),
        (4, 1),
        (4, 3),
        (4, 7),
        (8, 1),
        (8, 3),
        (8, 7),
        (2, 15),
        (4, 15),
        (2, 31),
    ] {
        let mut states = prefilled(&mut model, sequences.min(4));
        while states.len() < sequences {
            states.push(model.new_sequence().expect("sequence"));
        }
        let repeats = 6u32;
        let mut pass = 0.0;
        let mut commit = 0.0;
        for repeat in 0..repeats + 2 {
            let mut rows = states
                .iter_mut()
                .enumerate()
                .map(|(index, state)| BatchRow {
                    token: 1000 + index as u32 * 31 + repeat,
                    drafts: (0..drafts).map(|draft| 2000 + draft as u32).collect(),
                    state: Some(state),
                    lag: None,
                })
                .collect::<Vec<_>>();
            let started = Instant::now();
            model.decode_batch(&mut rows).expect("batched pass");
            let middle = Instant::now();
            let kept = rows
                .iter()
                .map(|row| row.drafts.len() + 1)
                .collect::<Vec<_>>();
            model.commit_batch(&mut rows, &kept).expect("commit");
            if repeat >= 2 {
                pass += middle.duration_since(started).as_secs_f64();
                commit += middle.elapsed().as_secs_f64();
            }
        }
        lines.push(serde_json::json!({
            "sequences": sequences, "drafts": drafts, "rows": sequences * (drafts + 1),
            "pass_ms": pass / f64::from(repeats) * 1e3,
            "commit_ms": commit / f64::from(repeats) * 1e3,
        }));
    }
    println!("{}", serde_json::json!({"batched_verify": lines}));
}

fn load_with_head(context: usize, block_rows: usize) -> BonsaiModel {
    let settings = crate::bonsai_mtp::MtpSettings::new(
        crate::bonsai_mtp::DEFAULT_BONSAI_MTP_ARTIFACT.into(),
        crate::bonsai_mtp::DEFAULT_MTP_DEPTH,
    )
    .expect("head settings");
    let package = BonsaiPackage::open(DEFAULT_BONSAI_GGUF).expect("open Bonsai GGUF");
    BonsaiModel::load(
        package,
        context,
        block_rows,
        None,
        Some(&settings),
        NgramSettings::default(),
        KvOptions::default(),
    )
    .expect("load")
}

fn greedy_sampler() -> Sampler {
    Sampler::new(
        VOCAB,
        crate::sampler::SamplingParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            eos_tokens: Vec::new(),
            seed: 0,
        },
    )
}

/// Natural text, so the head's chains run deep: each sequence's prompt,
/// then the continuation its batched steps are teacher-forced through.
fn natural_texts(model: &BonsaiModel) -> Vec<Vec<u32>> {
    let tokenizer =
        crate::bonsai_tokenizer::BonsaiTokenizer::from_package(&model.package).expect("tokenizer");
    [
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog again, and then the dog wakes up and chases the fox across the field until both of them are tired.",
        "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty, twenty-one, twenty-two.",
        "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n\n\ndef multiply(a, b):\n    return a * b\n\n\ndef divide(a, b):\n    return a / b\n",
        "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday. January, February, March, April, May, June, July, August, September, October, November, December.",
    ]
    .iter()
    .map(|text| tokenizer.encode(text).expect("encode"))
    .collect()
}

/// Parked sequences prefilled (head included) with the first eight tokens
/// of their texts, then advanced `steps` teacher-forced batched steps
/// through the text that leave their heads behind by those rows, with each
/// sequence's next seed.
fn lagged(
    model: &mut BonsaiModel,
    count: usize,
    steps: usize,
) -> Vec<(SequenceState, HeadLag, u32)> {
    const PROMPT: usize = 8;
    let texts = natural_texts(model);
    let text = |index: usize| &texts[index % texts.len()];
    let mut sequences = (0..count)
        .map(|index| {
            let mut state = model.new_sequence().expect("sequence");
            model.swap_sequence(&mut state).expect("swap in");
            model
                .prefill_segment(&text(index)[..PROMPT], true, &mut |_| {})
                .expect("prefill");
            model.swap_sequence(&mut state).expect("swap out");
            (state, HeadLag::default(), 0)
        })
        .collect::<Vec<_>>();
    for step in 0..steps {
        let mut rows = sequences
            .iter_mut()
            .enumerate()
            .map(|(index, (state, lag, _))| BatchRow {
                token: text(index)[PROMPT + step],
                drafts: Vec::new(),
                state: Some(state),
                lag: Some(lag),
            })
            .collect::<Vec<_>>();
        model.decode_batch(&mut rows).expect("batched step");
    }
    for (index, sequence) in sequences.iter_mut().enumerate() {
        sequence.2 = text(index)[PROMPT + steps];
    }
    sequences
}

/// Drafts of several sequences made together, each head fed the rows it
/// missed in the same passes (a lag too long for one pass first, alone), are
/// each sequence's own chain: what its solo catch-up and draft propose from
/// the same state.
#[test]
#[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device"]
fn batched_head_drafts_match_each_alone() {
    // Sixteen-row passes: three sequences owing nine rows each feed part of
    // their lags ahead of the drafting pass.
    let mut model = load_with_head(4096, 16);
    let sampler = greedy_sampler();
    let count = 3;
    let mut compared = 0;
    let mut matching = 0;
    for steps in [0usize, 1, 9, 20] {
        let mut together = lagged(&mut model, count, steps);
        let mut alone = lagged(&mut model, count, steps);
        // The first sequence drafts resident, the others parked.
        model.swap_sequence(&mut together[0].0).expect("swap in");
        let mut entries = together
            .iter_mut()
            .enumerate()
            .map(|(index, (state, lag, seed))| HeadDraft {
                seed: *seed,
                depth: 3,
                state: (index > 0).then_some(state),
                lag,
                sampler: &sampler,
            })
            .collect::<Vec<_>>();
        let batched = model.draft_heads(&mut entries).expect("batched drafts");
        drop(entries);
        model.swap_sequence(&mut together[0].0).expect("swap out");
        assert!(together.iter().all(|(_, lag, _)| lag.is_empty()));
        for (index, (state, lag, seed)) in alone.iter_mut().enumerate() {
            model.swap_sequence(state).expect("swap in");
            let solo = model
                .draft_resident(lag, *seed, &sampler, 4)
                .expect("solo drafts");
            model.swap_sequence(state).expect("swap out");
            println!(
                "{}",
                serde_json::json!({
                    "lag": steps, "sequence": index,
                    "solo": solo, "batched": batched[index],
                })
            );
            assert!(!batched[index].is_empty(), "no drafts at lag {steps}");
            assert_eq!(
                batched[index][0], solo[0],
                "first draft differs at lag {steps}, sequence {index}"
            );
            compared += 1;
            matching += usize::from(batched[index] == solo);
        }
        // The heads were fed exactly: a second round from the next seeds
        // (no lag now) drafts as the solo heads do.
        let texts = natural_texts(&model);
        let next = (0..count)
            .map(|index| texts[index][9 + steps])
            .collect::<Vec<_>>();
        let mut entries = together
            .iter_mut()
            .zip(next.iter().copied())
            .map(|((state, lag, _), seed)| HeadDraft {
                seed,
                depth: 3,
                state: Some(state),
                lag,
                sampler: &sampler,
            })
            .collect::<Vec<_>>();
        // Drafting writes no committed state: position stays, so the same
        // positions are drafted again from the same committed hidden.
        let again = model.draft_heads(&mut entries).expect("second round");
        drop(entries);
        for (index, ((state, lag, _), seed)) in alone.iter_mut().zip(next).enumerate() {
            model.swap_sequence(state).expect("swap in");
            let solo = model
                .draft_resident(lag, seed, &sampler, 4)
                .expect("solo drafts");
            model.swap_sequence(state).expect("swap out");
            assert_eq!(again[index][0], solo[0], "second round, sequence {index}");
            compared += 1;
            matching += usize::from(again[index] == solo);
        }
    }
    println!(
        "{}",
        serde_json::json!({"chains": compared, "identical": matching})
    );
    assert!(
        matching * 10 >= compared * 8,
        "{matching} of {compared} chains identical"
    );
}

/// Time of one batched MTP draft round at 1..=8 sequences, each owing its
/// head one row (a sequence that drafted last step) or eight, against each
/// sequence drafting alone in turn (the cost model of batched head drafts).
#[test]
#[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device; prints timings"]
fn batched_head_draft_timings() {
    let mut model = load_with_head(4096, 128);
    let sampler = greedy_sampler();
    let mut lines = Vec::new();
    for lag in [1usize, 8] {
        for count in [1usize, 2, 3, 4, 6, 8] {
            let mut sequences = lagged(&mut model, count, 0);
            let repeats = 8;
            let mut batched = 0.0;
            let mut alone = 0.0;
            let mut steps = 0usize;
            let mut chain = 0usize;
            for repeat in 0..repeats + 2 {
                for step in 0..lag {
                    let mut rows = sequences
                        .iter_mut()
                        .enumerate()
                        .map(|(index, (state, lag, _))| BatchRow {
                            token: 1000 + (index * 31 + repeat * 7 + step) as u32,
                            drafts: Vec::new(),
                            state: Some(state),
                            lag: Some(lag),
                        })
                        .collect::<Vec<_>>();
                    model.decode_batch(&mut rows).expect("batched step");
                }
                let alone_round = repeat % 2 == 1;
                let started = Instant::now();
                let drafted = if alone_round {
                    sequences
                        .iter_mut()
                        .map(|(state, lag, _)| {
                            model.swap_sequence(state).expect("swap in");
                            let drafts = model
                                .draft_resident(lag, 13, &sampler, 4)
                                .expect("solo drafts");
                            model.swap_sequence(state).expect("swap out");
                            drafts
                        })
                        .collect::<Vec<_>>()
                } else {
                    let mut entries = sequences
                        .iter_mut()
                        .map(|(state, lag, _)| HeadDraft {
                            seed: 13,
                            depth: 3,
                            state: Some(state),
                            lag,
                            sampler: &sampler,
                        })
                        .collect::<Vec<_>>();
                    model.draft_heads(&mut entries).expect("batched drafts")
                };
                let elapsed = started.elapsed().as_secs_f64();
                if repeat >= 2 {
                    if alone_round {
                        alone += elapsed;
                    } else {
                        batched += elapsed;
                        steps += drafted.iter().map(Vec::len).max().unwrap_or(0);
                        chain += drafted.iter().map(Vec::len).sum::<usize>();
                    }
                }
            }
            let rounds = (repeats / 2) as f64;
            lines.push(serde_json::json!({
                "sequences": count, "lag": lag,
                "batched_ms": batched / rounds * 1e3,
                "alone_ms": alone / rounds * 1e3,
                "depth_passes": steps as f64 / rounds,
                "drafts_per_sequence": chain as f64 / rounds / count as f64,
            }));
        }
    }
    println!("{}", serde_json::json!({"head_drafts": lines}));
}

/// A prompt chunk prefilled inside a batched pass, after two decoding
/// sequences' rows, leaves its sequence (target and MTP head) where prefill
/// blocks leave it: the next token's logits track the reference and the head
/// drafts the same chain. The decoding rows beside it track the same step
/// without the chunk.
#[test]
#[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device"]
#[allow(clippy::too_many_lines)]
fn prefill_rows_in_a_batched_pass_track_prefill_blocks() {
    let mut model = load_with_head(4096, 64);
    let sampler = greedy_sampler();
    let text = natural_texts(&model).swap_remove(0);
    let prefix = &text[..8];
    let chunk = text[8..38].to_vec();
    let next = text[38];
    // Reference: prefill blocks, the head ingesting them as prefill does.
    let mut reference = model.new_sequence().expect("sequence");
    model.swap_sequence(&mut reference).expect("swap in");
    model
        .prefill_segment(prefix, false, &mut |_| {})
        .expect("prefix");
    model
        .prefill_segment(&chunk, false, &mut |_| {})
        .expect("chunk");
    let reference_drafts = model
        .draft_resident(&mut HeadLag::default(), next, &sampler, 4)
        .expect("reference drafts");
    model.decode(next).expect("decode");
    let reference_logits = model.scratch.logits.as_slice::<f32>()[..VOCAB].to_vec();
    model.swap_sequence(&mut reference).expect("swap out");
    // The same chunk inside a batched pass.
    let mut decoding = lagged(&mut model, 2, 0);
    let mut alone = lagged(&mut model, 2, 0);
    let mut folded = model.new_sequence().expect("sequence");
    model.swap_sequence(&mut folded).expect("swap in");
    model
        .prefill_segment(prefix, false, &mut |_| {})
        .expect("prefix");
    model.swap_sequence(&mut folded).expect("swap out");
    for resident in [false, true] {
        if resident {
            // Again from the start, the prefilling sequence resident.
            folded = model.new_sequence().expect("sequence");
            model.swap_sequence(&mut folded).expect("swap in");
            model
                .prefill_segment(prefix, false, &mut |_| {})
                .expect("prefix");
            decoding = lagged(&mut model, 2, 0);
            alone = lagged(&mut model, 2, 0);
        }
        let mut rows = decoding
            .iter_mut()
            .enumerate()
            .map(|(index, (state, lag, _))| BatchRow {
                token: forced(index, 0),
                drafts: Vec::new(),
                state: Some(state),
                lag: Some(lag),
            })
            .collect::<Vec<_>>();
        model
            .decode_batch_prefilling(
                &mut rows,
                Some(PrefillRows {
                    tokens: &chunk,
                    state: (!resident).then_some(&mut folded),
                }),
            )
            .expect("batched pass with a prompt chunk");
        drop(rows);
        let output = model.batch.as_ref().expect("batch scratch");
        assert_eq!(output.rows, 2, "only the decoding rows have logits");
        let with_chunk = output.logits.as_slice::<f32>()[..2 * VOCAB].to_vec();
        let mut rows = alone
            .iter_mut()
            .enumerate()
            .map(|(index, (state, lag, _))| BatchRow {
                token: forced(index, 0),
                drafts: Vec::new(),
                state: Some(state),
                lag: Some(lag),
            })
            .collect::<Vec<_>>();
        model.decode_batch(&mut rows).expect("batched pass alone");
        drop(rows);
        let output = model.batch.as_ref().expect("batch scratch");
        let without = output.logits.as_slice::<f32>()[..2 * VOCAB].to_vec();
        for row in 0..2 {
            let range = row * VOCAB..(row + 1) * VOCAB;
            let divergence = kl(&without[range.clone()], &with_chunk[range.clone()]);
            assert!(divergence < 1e-4, "decoding row {row}: KL {divergence}");
            assert_eq!(argmax(&without[range.clone()]), argmax(&with_chunk[range]));
        }
        if !resident {
            model.swap_sequence(&mut folded).expect("swap in");
        }
        assert_eq!(model.position, prefix.len() + chunk.len());
        let drafts = model
            .draft_resident(&mut HeadLag::default(), next, &sampler, 4)
            .expect("drafts");
        model.decode(next).expect("decode");
        let logits = model.scratch.logits.as_slice::<f32>()[..VOCAB].to_vec();
        model.swap_sequence(&mut folded).expect("swap out");
        let divergence = kl(&reference_logits, &logits);
        println!(
            "{}",
            serde_json::json!({
                "resident": resident, "next_token_kl": divergence,
                "reference_drafts": reference_drafts, "drafts": drafts,
            })
        );
        assert!(divergence < 1e-4, "next-token KL {divergence}");
        assert_eq!(argmax(&reference_logits), argmax(&logits));
        assert_eq!(drafts.first(), reference_drafts.first());
    }
}

/// Time of a batched step of `sequences` decoding rows with a prompt chunk
/// folded into its pass, against the same step followed by the chunk as its
/// own prefill block (the cost model of prefilling beside decoding).
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device; prints timings"]
fn prefill_fold_timings() {
    let package = BonsaiPackage::open(DEFAULT_BONSAI_GGUF).expect("open Bonsai GGUF");
    let mut model = BonsaiModel::load(
        package,
        8192,
        128,
        None,
        None,
        NgramSettings::default(),
        KvOptions::default(),
    )
    .expect("load");
    let mut lines = Vec::new();
    for (sequences, chunk) in [
        (2usize, 38usize),
        (4, 36),
        (4, 28),
        (4, 20),
        (8, 32),
        (8, 24),
    ] {
        let mut states = prefilled(&mut model, sequences.min(4));
        while states.len() < sequences {
            states.push(model.new_sequence().expect("sequence"));
        }
        let mut prompt = model.new_sequence().expect("sequence");
        let tokens = (0..chunk).map(|row| 1000 + row as u32).collect::<Vec<_>>();
        let repeats = 6u32;
        let mut folded = 0.0;
        let mut separate = 0.0;
        for repeat in 0..repeats + 2 {
            for fold in [true, false] {
                let mut rows = states
                    .iter_mut()
                    .enumerate()
                    .map(|(index, state)| BatchRow {
                        token: 1000 + index as u32 * 31 + repeat,
                        drafts: Vec::new(),
                        state: Some(state),
                        lag: None,
                    })
                    .collect::<Vec<_>>();
                let started = Instant::now();
                if fold {
                    model
                        .decode_batch_prefilling(
                            &mut rows,
                            Some(PrefillRows {
                                tokens: &tokens,
                                state: Some(&mut prompt),
                            }),
                        )
                        .expect("folded pass");
                } else {
                    model.decode_batch(&mut rows).expect("batched pass");
                    drop(rows);
                    model.swap_sequence(&mut prompt).expect("swap in");
                    model
                        .forward_block(&tokens, BlockOutput::None)
                        .expect("prefill block");
                    model.swap_sequence(&mut prompt).expect("swap out");
                }
                let elapsed = started.elapsed().as_secs_f64();
                if repeat >= 2 {
                    if fold {
                        folded += elapsed;
                    } else {
                        separate += elapsed;
                    }
                }
            }
        }
        lines.push(serde_json::json!({
            "sequences": sequences, "chunk": chunk,
            "folded_ms": folded / f64::from(repeats) * 1e3,
            "separate_ms": separate / f64::from(repeats) * 1e3,
        }));
    }
    println!("{}", serde_json::json!({"prefill_fold": lines}));
}
