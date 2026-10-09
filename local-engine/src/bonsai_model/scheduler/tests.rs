use std::collections::HashMap;

use super::super::DEFAULT_PREFILL_CHUNK;
use super::*;
use crate::bonsai_model::StopReason;
use crate::bonsai_mtp::MtpMode;
use crate::bonsai_native::KvOptions;
use crate::bonsai_ngram::{NgramSettings, SuffixStore};

const PROMPTS: [&str; 3] = [
    "Explain how a hash map handles collisions.",
    "Write a short poem about the sea at night.",
    "List five facts about the planet Mars.",
];

fn open(mtp: &MtpMode) -> BonsaiEngine {
    open_with(mtp, false)
}

fn open_with(mtp: &MtpMode, ngram: bool) -> BonsaiEngine {
    BonsaiEngine::open_with_options(
        std::path::Path::new(crate::bonsai::DEFAULT_BONSAI_GGUF),
        Some(4096),
        DEFAULT_PREFILL_CHUNK,
        None,
        mtp,
        NgramSettings {
            enabled: ngram,
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
        .admit(&ids, &params(max_tokens), None, cancel)
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
        for (id, result) in engine.step(&mut |_, _| true, &mut |_, _| {}) {
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

/// A prompt of several hundred tokens: many prefill chunks.
fn long_prompt(engine: &BonsaiEngine) -> Vec<u32> {
    let items = (0..60)
        .map(|index| format!("Entry {index}: the {index}th lighthouse keeper logged calm seas."))
        .collect::<Vec<_>>()
        .join("\n");
    engine
        .encode_prompt(
            &format!("{items}\n\nWhich entry number is the largest?"),
            false,
            false,
        )
        .expect("prompt")
}

/// Prompts whose answers copy text from the prompt, so suffix lookup drafts.
fn copy_prompt(engine: &BonsaiEngine, index: usize) -> Vec<u32> {
    let fruit = [
        "apple", "pear", "plum", "fig", "lime", "kiwi", "date", "grape",
    ];
    let lines = (0..24)
        .map(|line| {
            format!(
                "{} {} costs {} cents",
                fruit[(line + index) % 8],
                line,
                10 + line * 3
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    engine
        .encode_prompt(
            &format!("Copy this list exactly, one line per entry, and nothing else:\n{lines}"),
            false,
            false,
        )
        .expect("prompt")
}

/// What a stepping loop observed: outcomes, and per step the ids that
/// emitted text (with how many pieces) and the ids that reported prefill.
#[derive(Default)]
struct Observed {
    outcomes: HashMap<u64, BonsaiGeneration>,
    steps: Vec<(HashMap<u64, usize>, Vec<u64>)>,
}

fn observe(
    engine: &mut BonsaiEngine,
    mut between: impl FnMut(&mut BonsaiEngine, &Observed),
) -> Observed {
    let mut observed = Observed::default();
    while engine.active_generations() > 0 {
        let mut emitted = HashMap::new();
        let mut prefilled = Vec::new();
        let finished = engine.step(
            &mut |id, _| {
                *emitted.entry(id).or_insert(0) += 1;
                true
            },
            &mut |id, _| prefilled.push(id),
        );
        observed.steps.push((emitted, prefilled));
        for (id, result) in finished {
            observed.outcomes.insert(id, result.expect("generation"));
        }
        between(engine, &observed);
    }
    observed
}

fn admit_ids(
    engine: &mut BonsaiEngine,
    ids: &[u32],
    max_tokens: usize,
    cancel: CancelToken,
) -> u64 {
    engine
        .admit(ids, &params(max_tokens), None, cancel)
        .expect("admit")
}

/// A long prompt arriving beside two decoding requests prefills in chunks
/// between their steps: they keep emitting tokens while it prefills, and all
/// three produce exactly their tokens alone.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn chunked_prefill_beside_decoding_matches_alone() {
    let mut engine = open(&MtpMode::Off(None));
    let long = long_prompt(&engine);
    assert!(
        long.len() > 4 * DEFAULT_PREFILL_CHUNK,
        "{} tokens",
        long.len()
    );
    let reference = alone(&mut engine, 48);
    let long_reference = engine
        .generate(&long, &params(16), |_| true)
        .expect("long alone")
        .token_ids;
    let first = admit(&mut engine, 0, 48, CancelToken::new());
    let second = admit(&mut engine, 1, 48, CancelToken::new());
    let mut long_id = None;
    let observed = observe(&mut engine, |engine, observed| {
        if long_id.is_none() && observed.steps.len() == 3 {
            long_id = Some(admit_ids(engine, &long, 16, CancelToken::new()));
        }
    });
    let long_id = long_id.expect("long admitted");
    let interleaved = observed
        .steps
        .iter()
        .filter(|(emitted, prefilled)| {
            prefilled.contains(&long_id)
                && emitted.contains_key(&first)
                && emitted.contains_key(&second)
        })
        .count();
    assert!(
        interleaved >= 4,
        "only {interleaved} steps interleaved prefill and decode"
    );
    assert_same("first", &observed.outcomes[&first].token_ids, &reference[0]);
    assert_same(
        "second",
        &observed.outcomes[&second].token_ids,
        &reference[1],
    );
    assert_same(
        "long",
        &observed.outcomes[&long_id].token_ids,
        &long_reference,
    );
}

/// Cancelling a request between two of its prefill chunks ends it with no
/// tokens, leaves the request decoding beside it exact, and leaves no
/// prompt-cache state behind that would change the prompt's output later.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn cancelling_mid_prefill_leaves_the_rest_exact() {
    let mut engine = open(&MtpMode::Off(None));
    let long = long_prompt(&engine);
    let reference = alone(&mut engine, 40);
    let long_reference = engine
        .generate(&long, &params(12), |_| true)
        .expect("long alone")
        .token_ids;
    let short = admit(&mut engine, 2, 40, CancelToken::new());
    let cancel = CancelToken::new();
    let long_id = admit_ids(&mut engine, &long, 12, cancel.clone());
    let observed = observe(&mut engine, |_, observed| {
        let chunks = observed
            .steps
            .iter()
            .filter(|(_, prefilled)| prefilled.contains(&long_id))
            .count();
        if chunks == 2 {
            cancel.cancel();
        }
    });
    let cancelled = &observed.outcomes[&long_id];
    assert_eq!(cancelled.stop_reason, StopReason::Cancelled);
    assert_eq!(cancelled.token_ids, Vec::<u32>::new());
    assert_same("short", &observed.outcomes[&short].token_ids, &reference[2]);
    let again = engine
        .generate(&long, &params(12), |_| true)
        .expect("long after cancellation");
    assert_same("long after cancellation", &again.token_ids, &long_reference);
}

/// Batched steps that verify every available draft — suffix-lookup drafts
/// of copying prompts, MTP drafts otherwise — produce each prompt's tokens
/// alone, and do verify more than one token per step.
#[test]
#[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device"]
fn batched_speculation_matches_alone() {
    let mut engine = open_with(&MtpMode::default(), true);
    engine.force_batched_drafts = true;
    let max_tokens = 256;
    let prompts = [
        copy_prompt(&engine, 0),
        copy_prompt(&engine, 3),
        prompt(&engine, 0),
    ];
    let reference = prompts
        .iter()
        .map(|ids| {
            engine
                .generate(ids, &params(max_tokens), |_| true)
                .expect("alone")
                .token_ids
        })
        .collect::<Vec<_>>();
    // Lookup drafts come from the prompts, not from the references' outputs.
    engine.suffix_store = SuffixStore::default();
    let ids = prompts
        .iter()
        .map(|prompt| admit_ids(&mut engine, prompt, max_tokens, CancelToken::new()))
        .collect::<Vec<_>>();
    let observed = observe(&mut engine, |_, _| {});
    let multi = observed
        .steps
        .iter()
        .filter(|(emitted, _)| emitted.len() >= 2 && emitted.values().any(|&pieces| pieces > 1))
        .count();
    assert!(multi > 0, "no batched step committed more than one token");
    for (index, id) in ids.iter().enumerate() {
        let output = &observed.outcomes[id];
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
    let lookups = ids
        .iter()
        .map(|id| observed.outcomes[id].stats.ngram.rounds)
        .sum::<usize>();
    assert!(lookups > 0, "no lookup drafts were verified");
}

/// Requests joining and leaving while the others verify drafts in batched
/// steps still produce their own tokens.
#[test]
#[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device"]
fn joining_and_leaving_during_batched_verify() {
    let mut engine = open_with(&MtpMode::default(), true);
    engine.force_batched_drafts = true;
    let prompts = [
        prompt(&engine, 2),
        copy_prompt(&engine, 1),
        copy_prompt(&engine, 5),
    ];
    let lengths = [256, 24, 64];
    let reference = prompts
        .iter()
        .zip(lengths)
        .map(|(ids, length)| {
            engine
                .generate(ids, &params(length), |_| true)
                .expect("alone")
                .token_ids
        })
        .collect::<Vec<_>>();
    engine.suffix_store = SuffixStore::default();
    let mut ids = vec![
        admit_ids(&mut engine, &prompts[0], lengths[0], CancelToken::new()),
        admit_ids(&mut engine, &prompts[1], lengths[1], CancelToken::new()),
    ];
    let observed = observe(&mut engine, |engine, observed| {
        // The third joins while the first two verify drafts together; the
        // second leaves while the third prefills or decodes beside the first.
        if ids.len() == 2 && observed.steps.len() == 3 {
            ids.push(admit_ids(
                engine,
                &prompts[2],
                lengths[2],
                CancelToken::new(),
            ));
        }
    });
    assert_eq!(ids.len(), 3);
    for (index, id) in ids.iter().enumerate() {
        assert_same(
            &format!("prompt {index}"),
            &observed.outcomes[id].token_ids,
            &reference[index],
        );
    }
    assert!(observed.outcomes[&ids[2]].stats.batched_tokens > 0);
}
