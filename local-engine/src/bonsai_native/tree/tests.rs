use std::collections::HashMap;

use super::super::{KvOptions, NgramSettings, PromptSnapshot, TAP_ROW_FLOATS};
use super::*;
use crate::bonsai::{BonsaiPackage, DEFAULT_BONSAI_GGUF};
use crate::sampler::SamplingParams;

/// A short chat-like prompt (the batch tests' first prefix, extended).
const PROMPT: [u32; 18] = [
    248_045, 846, 198, 814, 20139, 1204, 264, 5010, 2336, 13081, 11, 264, 198, 13, 3010, 1204, 264,
    2336,
];
/// History past `SPLIT_TENSOR_MIN_PREFIX` (1,024), so tree attention takes the
/// tensor split kernels for its history.
const LONG_PREFIX: usize = 1100;
/// Verify rows against plain decoding: the chain verify test's Q8 bound.
const TOLERANCE: f32 = 3e-2;

fn load(context: usize) -> BonsaiModel {
    let package = BonsaiPackage::open(DEFAULT_BONSAI_GGUF).expect("open Bonsai GGUF");
    let mut model = BonsaiModel::load(
        package,
        context,
        64,
        None,
        None,
        NgramSettings {
            enabled: false,
            ..NgramSettings::default()
        },
        KvOptions::default(),
    )
    .expect("load");
    model.enable_tree_verify().expect("tree verify");
    model.set_device_greedy(true);
    model
}

fn long_prefix() -> Vec<u32> {
    PROMPT
        .iter()
        .copied()
        .chain((0..LONG_PREFIX - PROMPT.len()).map(|index| 300 + (index as u32 * 7919) % 90_000))
        .collect()
}

/// `f32::total_cmp` maximum, ties to the lower id: the GPU argmax's order.
fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(a_index, a), (b_index, b)| a.total_cmp(b).then_with(|| b_index.cmp(a_index)))
        .map_or(0, |(index, _)| index as u32)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn greedy_sampler() -> Sampler {
    Sampler::new(
        VOCAB,
        SamplingParams {
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

/// A prefilled prefix: the state after it, the anchor (the prefix's greedy
/// next token, not yet run) and plain greedy decoding's next tokens.
struct Fixture {
    snapshot: PromptSnapshot,
    anchor: u32,
    greedy: Vec<u32>,
    /// Plain decoding's logits of each path tried so far.
    plain: HashMap<Vec<u32>, Vec<f32>>,
}

impl Fixture {
    fn new(model: &mut BonsaiModel, prefix: &[u32], continuation: usize) -> Self {
        model.reset();
        model.prefill(prefix, &mut |_| {}).expect("prefill");
        let anchor = argmax(&model.last_logits());
        let snapshot = model.prompt_snapshot().expect("snapshot");
        let mut fixture = Self {
            snapshot,
            anchor,
            greedy: Vec::new(),
            plain: HashMap::new(),
        };
        let mut path = vec![anchor];
        for _ in 0..continuation {
            let next = argmax(&fixture.plain(model, &path));
            fixture.greedy.push(next);
            path.push(next);
        }
        fixture
    }

    fn restore(&self, model: &mut BonsaiModel) {
        model
            .restore_prompt_snapshot(&self.snapshot)
            .expect("restore");
    }

    /// Plain single-row decoding's logits after `path` (the anchor first).
    fn plain(&mut self, model: &mut BonsaiModel, path: &[u32]) -> Vec<f32> {
        if let Some(logits) = self.plain.get(path) {
            return logits.clone();
        }
        self.restore(model);
        for &token in path {
            model.forward(token, true).expect("plain decode");
        }
        let logits = model.last_logits();
        self.plain.insert(path.to_vec(), logits.clone());
        logits
    }
}

/// Deterministic pseudo-random numbers (xorshift64).
struct Rng(u64);

impl Rng {
    fn next(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % bound as u64) as usize
    }
}

/// A random tree of `rows` rows whose rows on the plain greedy path hold
/// its tokens about half the time, so trees hold real continuations as well
/// as random tokens.
fn random_tree(rng: &mut Rng, rows: usize, anchor: u32, greedy: &[u32]) -> DraftTree {
    let mut tokens = vec![anchor];
    let mut parents = vec![-1];
    let mut on_greedy = vec![true];
    for row in 1..rows {
        let parent = rng.next(row);
        let depth = tree_depth(&parents, parent) + 1;
        let follow = on_greedy[parent] && depth <= greedy.len() && rng.next(2) == 0;
        tokens.push(if follow {
            greedy[depth - 1]
        } else {
            10 + rng.next(150_000) as u32
        });
        parents.push((parent as u32).cast_signed());
        on_greedy.push(follow);
    }
    DraftTree::from_parents(&tokens, &parents).expect("random tree")
}

fn tree_depth(parents: &[i32], mut row: usize) -> usize {
    let mut depth = 0;
    while let Ok(parent) = usize::try_from(parents[row]) {
        row = parent;
        depth += 1;
    }
    depth
}

/// The tokens of `row`'s path, root first.
fn path_tokens(tree: &DraftTree, row: usize) -> Vec<u32> {
    let mut path = vec![tree.tokens[row]];
    let mut row = row;
    while let Ok(parent) = usize::try_from(tree.parents[row]) {
        path.push(tree.tokens[parent]);
        row = parent;
    }
    path.reverse();
    path
}

/// Every row of random trees of one to eight rows, verified in one pass,
/// tracks plain single-row decoding of that row's own path: the same GPU
/// argmax and logits within the chain verify's tolerance, at a short history
/// (SIMD history splits) and one past 1,024 tokens (tensor splits). The
/// negative control proves the tree's structure is used: a branching row
/// read as the chain of the rows before it differs by far more.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn tree_rows_match_plain_decoding_of_their_paths() {
    let mut model = load(LONG_PREFIX + 64);
    let mut evidence = Vec::new();
    let mut control = 0.0f32;
    for (label, prefix) in [("short", PROMPT.to_vec()), ("long", long_prefix())] {
        let mut fixture = Fixture::new(&mut model, &prefix, MAX_TREE_ROWS);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ prefix.len() as u64);
        let mut worst = 0.0f32;
        let mut rows_checked = 0;
        let shapes = if label == "short" { 3 } else { 1 };
        for rows in 1..=MAX_TREE_ROWS {
            for _ in 0..shapes {
                let tree = random_tree(&mut rng, rows, fixture.anchor, &fixture.greedy);
                fixture.restore(&mut model);
                model.tree_verify(&tree).expect("tree verify");
                assert_eq!(model.position, prefix.len() + usize::from(rows == 1));
                let argmax_rows = model.tree_argmax().expect("argmax");
                let verified = (0..rows)
                    .map(|row| model.tree_logits(row).expect("row logits"))
                    .collect::<Vec<_>>();
                for (row, logits) in verified.iter().enumerate() {
                    let path = path_tokens(&tree, row);
                    let expected = fixture.plain(&mut model, &path);
                    let difference = max_abs(logits, &expected);
                    worst = worst.max(difference);
                    rows_checked += 1;
                    assert!(
                        argmax_rows[row] == argmax(&expected) && difference < TOLERANCE,
                        "{label} tree {tree:?} row {row}: argmax {} against plain {}, \
                         max |difference| {difference}",
                        argmax_rows[row],
                        argmax(&expected)
                    );
                    if tree.parents[row] >= 0 && tree.parents[row] != (row as u32).cast_signed() - 1
                    {
                        let chain = tree.tokens[..=row].to_vec();
                        control = control.max(max_abs(logits, &fixture.plain(&mut model, &chain)));
                    }
                }
            }
        }
        evidence.push(serde_json::json!({
            "history": label, "prefix": prefix.len(), "rows_checked": rows_checked,
            "worst_max_abs": worst,
        }));
    }
    assert!(
        control > 10.0 * TOLERANCE,
        "rows read as chains matched their tree paths: the tree went unused ({control})"
    );
    eprintln!(
        "{:#}",
        serde_json::json!({"rows": evidence, "chain_misread_control": control})
    );
}

/// The same path verified under different trees gives bitwise the same
/// logits: whatever rows sit beside it and wherever its rows sit in the
/// pass, among trees whose row counts take the same projection kernels
/// (three or four rows; five to eight). Across those classes the numbers
/// differ at rounding level, as chain verify blocks of different row counts
/// do; that difference is reported, not asserted.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn a_path_verifies_bitwise_alike_under_every_tree_schedule() {
    let mut model = load(LONG_PREFIX + 64);
    let mut evidence = Vec::new();
    for (label, prefix) in [("short", PROMPT.to_vec()), ("long", long_prefix())] {
        let fixture = Fixture::new(&mut model, &prefix, 3);
        let [g1, g2, g3] = [fixture.greedy[0], fixture.greedy[1], fixture.greedy[2]];
        let a = fixture.anchor;
        let decoy = |index: u32| 1000 + 37 * index;
        // (rows of the path [a, g1, g2, g3] in the tree, tree)
        let schedules = [
            // Eight rows: the path first, then branches.
            (
                vec![0, 1, 2, 3],
                DraftTree::from_parents(
                    &[a, g1, g2, g3, decoy(1), decoy(2), decoy(3), decoy(4)],
                    &[-1, 0, 1, 2, 0, 1, 2, 4],
                ),
            ),
            // Eight rows: the path interleaved with decoys.
            (
                vec![0, 2, 4, 7],
                DraftTree::from_parents(
                    &[a, decoy(5), g1, decoy(6), g2, decoy(7), decoy(8), g3],
                    &[-1, 0, 0, 1, 2, 4, 2, 4],
                ),
            ),
            // Six rows: the path last.
            (
                vec![0, 3, 4, 5],
                DraftTree::from_parents(
                    &[a, decoy(9), decoy(10), g1, g2, g3],
                    &[-1, 0, 1, 0, 3, 4],
                ),
            ),
            // Four rows.
            (
                vec![0, 2, 3],
                DraftTree::from_parents(&[a, decoy(11), g1, g2], &[-1, 0, 0, 2]),
            ),
            (
                vec![0, 1, 2],
                DraftTree::from_parents(&[a, g1, g2, decoy(12)], &[-1, 0, 1, 0]),
            ),
            // Three rows.
            (
                vec![0, 2],
                DraftTree::from_parents(&[a, decoy(13), g1], &[-1, 0, 0]),
            ),
        ];
        let mut runs = Vec::new();
        for (path, tree) in schedules {
            let tree = tree.expect("schedule tree");
            fixture.restore(&mut model);
            model.tree_verify(&tree).expect("tree verify");
            let logits = path
                .iter()
                .map(|&row| model.tree_logits(row).expect("row logits"))
                .collect::<Vec<_>>();
            runs.push((tree.len(), logits));
        }
        let class = |rows: usize| usize::from(rows >= 5);
        let mut compared = 0;
        let mut across = 0.0f32;
        for (index, (rows, logits)) in runs.iter().enumerate() {
            for (other_rows, other) in &runs[..index] {
                for (depth, (row, other_row)) in logits.iter().zip(other).enumerate() {
                    if class(*rows) == class(*other_rows) {
                        assert_eq!(
                            bits(row),
                            bits(other_row),
                            "{label}: depth {depth} differs between trees of {rows} and \
                             {other_rows} rows"
                        );
                        compared += 1;
                    } else {
                        across = across.max(max_abs(row, other_row));
                    }
                }
            }
        }
        evidence.push(serde_json::json!({
            "history": label, "bitwise_pairs": compared,
            "max_abs_across_kernel_classes": across,
        }));
    }
    eprintln!("{:#}", serde_json::json!(evidence));
}

/// Speculative decoding through trees emits plain decoding's tokens, and
/// commits leave the model where plain decoding of the accepted tokens
/// leaves it: trees hold plain decoding's continuation in rows off their
/// first branch (so commits move K/V rows and replay recurrences from
/// non-consecutive rows), on it, and partly (so some rounds accept only a
/// prefix); after the rounds, plain decode steps continue with plain
/// decoding's tokens and track its logits. Greedy, and sampled with the
/// random numbers plain sampling draws.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
#[expect(clippy::too_many_lines)]
fn tree_commits_continue_like_plain_decoding() {
    let mut model = load(256);
    for temperature in [0.0f32, 0.8] {
        let params = SamplingParams {
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            top_p: 0.95,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            eos_tokens: Vec::new(),
            seed: 17,
        };
        model.set_device_greedy(temperature == 0.0);
        // Plain decoding's tokens and logits, sampled one step at a time.
        let fixture = Fixture::new(&mut model, &PROMPT, 0);
        let anchor = fixture.anchor;
        fixture.restore(&mut model);
        let mut plain = Sampler::new(VOCAB, params.clone());
        let mut stream = vec![anchor];
        let mut plain_logits = Vec::new();
        for _ in 0..40 {
            model
                .forward(*stream.last().expect("token"), true)
                .expect("plain decode");
            plain_logits.push(model.last_logits());
            let next = model.sample(&mut plain).expect("plain sample").token_id;
            plain.observe(&[next]);
            stream.push(next);
        }

        fixture.restore(&mut model);
        let mut sampler = Sampler::new(VOCAB, params);
        let mut emitted = vec![anchor];
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut round = 0;
        let mut rounds = Vec::new();
        while emitted.len() < 28 {
            let done = emitted.len() - 1;
            let next = |offset: usize| stream[done + 1 + offset];
            let decoy = |offset: u32| 2000 + 101 * (round as u32 + offset);
            let tree = match round % 4 {
                // The continuation off the first branch, three deep.
                0 => DraftTree::from_parents(
                    &[
                        emitted[done],
                        decoy(0),
                        next(0),
                        decoy(1),
                        next(1),
                        next(2),
                        decoy(2),
                    ],
                    &[-1, 0, 0, 1, 2, 4, 4],
                ),
                // On the first branch.
                1 => DraftTree::from_parents(
                    &[emitted[done], next(0), next(1), decoy(3), decoy(4)],
                    &[-1, 0, 1, 0, 2],
                ),
                // Only one level deep, then decoys: a partial acceptance.
                2 => DraftTree::from_parents(
                    &[
                        emitted[done],
                        decoy(5),
                        next(0),
                        decoy(6),
                        decoy(7),
                        decoy(8),
                    ],
                    &[-1, 0, 0, 2, 2, 1],
                ),
                // A random tree.
                _ => Ok(random_tree(&mut rng, 8, emitted[done], &stream[done + 1..])),
            }
            .expect("round tree");
            let acceptance = model.tree_step(&tree, &mut sampler).expect("tree step");
            assert_eq!(acceptance.path.len(), acceptance.samples.len());
            assert_eq!(model.position, PROMPT.len() + done + acceptance.path.len());
            rounds.push(acceptance.path.clone());
            emitted.extend(acceptance.samples.iter().map(|sample| sample.token_id));
            round += 1;
        }
        let accepted = emitted.len() - 1;
        assert_eq!(
            emitted[..],
            stream[..=accepted],
            "temperature {temperature}: tree rounds {rounds:?}"
        );
        assert!(
            rounds.iter().any(|path| path.len() > 2 && path[1] != 1),
            "no round committed a path off the first branch: {rounds:?}"
        );
        // Plain decode steps after the commits.
        let mut worst = 0.0f32;
        for step in accepted..accepted + 8 {
            model
                .forward(stream[step], true)
                .expect("decode after commits");
            let logits = model.last_logits();
            worst = worst.max(max_abs(&logits, &plain_logits[step]));
            let next = model.sample(&mut sampler).expect("sample").token_id;
            sampler.observe(&[next]);
            assert_eq!(
                next,
                stream[step + 1],
                "temperature {temperature} step {step}"
            );
        }
        assert!(worst < TOLERANCE, "decode after commits drifted: {worst}");
        eprintln!(
            "{}",
            serde_json::json!({
                "temperature": temperature, "rounds": rounds, "tokens": accepted,
                "continuation_max_abs": worst,
            })
        );
    }
}

/// Taps hold the residual after layers 5, 19, 33, 47 and 61 of every
/// committed position: prefill rows, plain decode steps and accepted tree
/// rows (moved to their positions), tracking the same positions decoded
/// one row at a time. Turning taps on changes no logit, of a chain verify or
/// a tree verify.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn taps_follow_committed_positions_and_change_no_logit() {
    let mut model = load(256);
    let fixture = Fixture::new(&mut model, &PROMPT, 4);
    let (a, greedy) = (fixture.anchor, fixture.greedy.clone());
    let tree = DraftTree::from_parents(
        &[a, 1234, greedy[0], greedy[1], 4321, greedy[2]],
        &[-1, 0, 0, 2, 1, 3],
    )
    .expect("tree");
    let chain = DraftTree::chain(&[a, greedy[0], 777, greedy[2]]);
    let verify = |model: &mut BonsaiModel, tree: &DraftTree| {
        fixture.restore(model);
        model.tree_verify(tree).expect("verify");
        (0..tree.len())
            .map(|row| bits(&model.tree_logits(row).expect("logits")))
            .collect::<Vec<_>>()
    };
    let untapped = (verify(&mut model, &tree), verify(&mut model, &chain));
    model.set_taps(true).expect("taps on");
    let tapped = (verify(&mut model, &tree), verify(&mut model, &chain));
    assert!(untapped == tapped, "taps changed verify logits");

    // Plain row-at-a-time taps of the prompt, the anchor and the continuation.
    model.reset();
    let sequence = PROMPT
        .iter()
        .copied()
        .chain([a, greedy[0], greedy[1], greedy[2]])
        .collect::<Vec<_>>();
    for &token in &sequence {
        model.forward(token, true).expect("plain decode");
    }
    let plain = model.tap_rows(0, sequence.len()).expect("plain taps");
    assert!(plain.iter().all(|value| value.is_finite()));
    assert!(model.tap_rows(0, sequence.len() + 1).is_none());

    // Prefill, then a tree whose accepted path skips rows.
    model.reset();
    model.prefill(&PROMPT, &mut |_| {}).expect("prefill");
    model.tree_verify(&tree).expect("tree verify");
    let acceptance = model.tree_accept(&mut greedy_sampler()).expect("accept");
    assert_eq!(acceptance.path, [0, 2, 3, 5]);
    model.tree_commit(&acceptance.path).expect("commit");
    let tapped = model.tap_rows(0, sequence.len()).expect("tree taps");
    let mut worst = Vec::new();
    for (position, (plain, tapped)) in plain
        .chunks(TAP_ROW_FLOATS)
        .zip(tapped.chunks(TAP_ROW_FLOATS))
        .enumerate()
    {
        for (slot, (plain, tapped)) in plain.chunks(WIDTH).zip(tapped.chunks(WIDTH)).enumerate() {
            let scale = plain
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()));
            let relative = max_abs(plain, tapped) / scale;
            assert!(
                relative < 1e-2,
                "position {position} tap {slot}: relative difference {relative}"
            );
            if position == 0 || position >= PROMPT.len() {
                worst.push((position, slot, relative));
            }
        }
        // Layers deepen the residual: each tap differs from the one before.
        assert!(
            tapped
                .chunks(WIDTH)
                .zip(tapped.chunks(WIDTH).skip(1))
                .all(|(a, b)| max_abs(a, b) > 1e-3)
        );
    }
    eprintln!(
        "{}",
        serde_json::json!({ "relative_tap_differences": worst })
    );
    model.set_taps(false).expect("taps off");
    assert!(model.tap_ring().is_none());
}

/// GPU time of tree verifies of one to eight rows against chain verifies of
/// the same row counts, and with the residual taps on, at a short history
/// and at 2,048 and 8,192 tokens (median of nine passes each). Hold the GPU
/// lock while it runs.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device; a timing run"]
fn tree_verify_cost_table() {
    let histories = std::env::var("TREE_COST_HISTORIES").map_or_else(
        |_| vec![PROMPT.len(), 2048, 8192],
        |text| {
            text.split(',')
                .map(|value| value.trim().parse().expect("history length"))
                .collect()
        },
    );
    let longest = histories.iter().copied().max().unwrap_or(PROMPT.len());
    let mut model = load(longest + 64);
    let mut table = Vec::new();
    for &history in &histories {
        let prefix = PROMPT
            .iter()
            .copied()
            .chain((PROMPT.len()..history).map(|index| 300 + (index as u32 * 7919) % 90_000))
            .collect::<Vec<_>>();
        model.reset();
        model.prefill(&prefix, &mut |_| {}).expect("prefill");
        let anchor = argmax(&model.last_logits());
        // Warm up both paths.
        model.tree_verify_costs(anchor, 2).expect("warm-up");
        let tree = model.tree_verify_costs(anchor, 9).expect("tree costs");
        let mut chain = [0.0f64; MAX_TREE_ROWS];
        chain[0] = tree[0].as_secs_f64() * 1e3;
        for rows in 2..=MAX_TREE_ROWS {
            let mut times = (0..9)
                .map(|_| {
                    model.take_gpu_time();
                    model
                        .tree_verify(&DraftTree::chain(&vec![anchor; rows]))
                        .expect("chain verify");
                    model.take_gpu_time().as_secs_f64() * 1e3
                })
                .collect::<Vec<_>>();
            times.sort_by(f64::total_cmp);
            chain[rows - 1] = times[times.len() / 2];
        }
        model.drop_tree_verify();
        // The same passes with the residual taps on.
        model.set_taps(true).expect("taps on");
        let tapped = model.tree_verify_costs(anchor, 9).expect("tapped costs");
        model.set_taps(false).expect("taps off");
        let tree = tree.map(|time| time.as_secs_f64() * 1e3);
        table.push(serde_json::json!({
            "history": history,
            "tree_ms": tree,
            "tree_ms_with_taps": tapped.map(|time| time.as_secs_f64() * 1e3),
            "chain_ms": chain,
            "tree_over_chain": tree.iter().zip(&chain).map(|(t, c)| t / c - 1.0).collect::<Vec<_>>(),
        }));
    }
    eprintln!("{:#}", serde_json::json!(table));
}
