use std::collections::HashMap;

use super::super::DEFAULT_PREFILL_CHUNK;
use super::*;
use crate::bonsai_model::StopReason;
use crate::bonsai_mtp::MtpMode;
use crate::bonsai_native::KvOptions;
use crate::bonsai_ngram::NgramSettings;

const PROMPTS: [&str; 3] = [
    "Explain how a hash map handles collisions.",
    "Write a short poem about the sea at night.",
    "List five facts about the planet Mars.",
];

fn open(mtp: &MtpMode) -> BonsaiEngine {
    BonsaiEngine::open_with_options(
        std::path::Path::new(crate::bonsai::DEFAULT_BONSAI_GGUF),
        Some(4096),
        DEFAULT_PREFILL_CHUNK,
        None,
        mtp,
        NgramSettings {
            enabled: false,
            ..NgramSettings::default()
        },
        KvOptions::default(),
    )
    .expect("load model")
}

fn params(max_tokens: usize) -> GenerateParams {
    GenerateParams {
        temperature: 0.0,
        max_tokens,
        ..GenerateParams::default()
    }
}

fn prompt(engine: &BonsaiEngine, index: usize) -> Vec<u32> {
    engine
        .encode_prompt(PROMPTS[index], false, false)
        .expect("prompt")
}

/// Each prompt generated alone, the reference every batched run must equal.
fn alone(engine: &mut BonsaiEngine, max_tokens: usize) -> Vec<Vec<u32>> {
    (0..PROMPTS.len())
        .map(|index| {
            let ids = prompt(engine, index);
            engine
                .generate(&ids, &params(max_tokens), |_| true)
                .expect("generate alone")
                .token_ids
        })
        .collect()
}

fn admit(engine: &mut BonsaiEngine, index: usize, max_tokens: usize, cancel: CancelToken) -> u64 {
    let ids = prompt(engine, index);
    engine
        .admit(
            &ids,
            &params(max_tokens),
            None,
            cancel,
            &mut |_| {},
            &mut |_| true,
        )
        .expect("admit")
}

/// Step until every generation finished, collecting outcomes by id.
fn drain(
    engine: &mut BonsaiEngine,
    outcomes: &mut HashMap<u64, BonsaiGeneration>,
    mut between: impl FnMut(&mut BonsaiEngine, usize),
) {
    let mut steps = 0;
    while engine.active_generations() > 0 {
        for (id, result) in engine.step(&mut |_, _| true) {
            outcomes.insert(id, result.expect("generation"));
        }
        steps += 1;
        between(engine, steps);
    }
}

fn assert_same(label: &str, batched: &[u32], reference: &[u32]) {
    let first = batched
        .iter()
        .zip(reference)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| batched.len().min(reference.len()));
    assert_eq!(
        batched, reference,
        "{label}: first difference at token {first}"
    );
}

/// Three sequences decoded together produce exactly the tokens each produces
/// alone with the same (plain-decode) configuration.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn batched_sequences_match_each_alone() {
    let max_tokens = 40;
    let mut engine = open(&MtpMode::Off(None));
    let reference = alone(&mut engine, max_tokens);
    let ids = (0..PROMPTS.len())
        .map(|index| admit(&mut engine, index, max_tokens, CancelToken::new()))
        .collect::<Vec<_>>();
    let mut outcomes = HashMap::new();
    drain(&mut engine, &mut outcomes, |_, _| {});
    for (index, id) in ids.iter().enumerate() {
        let output = &outcomes[id];
        assert!(
            output.stats.batched_tokens > 0,
            "prompt {index} never batched"
        );
        assert_same(
            &format!("prompt {index}"),
            &output.token_ids,
            &reference[index],
        );
    }
}

/// Sequences joining and leaving mid-stream, with MTP speculation whenever a
/// sequence runs alone (its head caught up after batched steps), still produce
/// each prompt's own tokens.
#[test]
#[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device"]
fn joining_and_leaving_keeps_each_sequence_exact() {
    let mut engine = open(&MtpMode::default());
    let lengths = [48, 16, 32];
    let reference = (0..PROMPTS.len())
        .map(|index| {
            let ids = prompt(&engine, index);
            engine
                .generate(&ids, &params(lengths[index]), |_| true)
                .expect("generate alone")
                .token_ids
        })
        .collect::<Vec<_>>();
    let mut outcomes = HashMap::new();
    let first = admit(&mut engine, 0, lengths[0], CancelToken::new());
    let mut ids = vec![first];
    // The first runs alone (speculating) for a few rounds, the second joins,
    // and the third joins once the second has left.
    let mut joined = 1;
    drain(&mut engine, &mut outcomes, |engine, steps| {
        if joined == 1 && steps == 3 {
            ids.push(admit(engine, 1, lengths[1], CancelToken::new()));
            joined = 2;
        } else if joined == 2 && engine.active_generations() == 1 {
            ids.push(admit(engine, 2, lengths[2], CancelToken::new()));
            joined = 3;
        }
    });
    assert_eq!(ids.len(), 3);
    for (index, id) in ids.iter().enumerate() {
        assert_same(
            &format!("prompt {index}"),
            &outcomes[id].token_ids,
            &reference[index],
        );
    }
    assert!(outcomes[&ids[1]].stats.batched_tokens > 0);
    assert!(outcomes[&ids[0]].stats.mtp.rounds > 0);
}

/// Cancelling one sequence of a batch stops only that one; the others finish
/// exactly as they would alone.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn cancelling_one_sequence_leaves_the_batch_exact() {
    let max_tokens = 32;
    let mut engine = open(&MtpMode::Off(None));
    let reference = alone(&mut engine, max_tokens);
    let cancels = (0..PROMPTS.len())
        .map(|_| CancelToken::new())
        .collect::<Vec<_>>();
    let ids = (0..PROMPTS.len())
        .map(|index| admit(&mut engine, index, max_tokens, cancels[index].clone()))
        .collect::<Vec<_>>();
    let mut outcomes = HashMap::new();
    drain(&mut engine, &mut outcomes, |_, steps| {
        if steps == 5 {
            cancels[1].cancel();
        }
    });
    let cancelled = &outcomes[&ids[1]];
    assert_eq!(cancelled.stop_reason, StopReason::Cancelled);
    assert!(cancelled.token_ids.len() < max_tokens);
    assert_eq!(
        cancelled.token_ids[..],
        reference[1][..cancelled.token_ids.len()]
    );
    for index in [0, 2] {
        assert_same(
            &format!("prompt {index}"),
            &outcomes[&ids[index]].token_ids,
            &reference[index],
        );
    }
    // The engine is reusable afterwards, alone.
    let ids = prompt(&engine, 0);
    let again = engine
        .generate(&ids, &params(max_tokens), |_| true)
        .expect("generate after batch");
    assert_same("after batch", &again.token_ids, &reference[0]);
}
