use super::*;
use crate::bonsai::DEFAULT_BONSAI_GGUF;
use crate::bonsai_tokenizer::BonsaiTokenizer;

#[test]
fn automatic_context_reserves_ten_percent_and_caps_at_training_limit() {
    let gib = 1024_u64.pow(3);
    let fixed = 8 * gib;
    let per_token = 64 * 1024;
    assert_eq!(fit_context(fixed, per_token, 16 * gib), Some(104_857));
    assert_eq!(fit_context(fixed, per_token, 24 * gib), Some(222_822));
    assert_eq!(
        fit_context(fixed, per_token, 32 * gib),
        Some(TRAINING_CONTEXT)
    );
    assert_eq!(fit_context(15 * gib, per_token, 16 * gib), None);
}

#[test]
fn speculation_byte_budget_counts_checkpoints_kv_and_logit_rows() {
    // 48 recurrent layers x (one state+history snapshot + compact rows).
    assert_eq!(Speculation::checkpoint_bytes(1), 158_877_696);
    assert_eq!(Speculation::checkpoint_bytes(2), 160_862_208);
    assert_eq!(Speculation::checkpoint_bytes(0), 0);
    // Head KV for 8 tokens (2 x 8 x 2048 B) + depth-2 rollback
    // + one contiguous verify-logit block with depth + 1 rows.
    assert_eq!(
        Speculation::extra_bytes(2, 8),
        32_768 + 160_862_208 + 3 * 248_320 * 4
    );
    assert_eq!(Speculation::extra_bytes(2, 8), 163_874_816);
}

/// Plain tokenwise decoding: reset, run `prefix` as one block, then feed
/// `steps` one token at a time, returning the target logits after each.
fn tokenwise_logits(model: &mut BonsaiModel, prefix: &[u32], steps: &[u32]) -> Vec<Vec<f32>> {
    model.reset();
    model
        .forward_block(prefix, BlockOutput::None)
        .expect("prefix block");
    steps
        .iter()
        .map(|&token| {
            model.forward(token, true).expect("tokenwise decode");
            model.scratch.logits.as_slice::<f32>().to_vec()
        })
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(index, _)| index)
}

fn compare(actual: &[f32], expected: &[f32]) -> serde_json::Value {
    assert_eq!(actual.len(), VOCAB);
    assert_eq!(expected.len(), VOCAB);
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    serde_json::json!({
        "max_abs": max_abs,
        "scaled": max_abs / scale,
        "argmax_actual": argmax(actual),
        "argmax_expected": argmax(expected),
        "same_argmax": argmax(actual) == argmax(expected),
    })
}

/// Head-independent check of the riskiest MTP state path: a verify block
/// must reproduce tokenwise logits row by row, and restoring the
/// checkpoint after `committed` rows must leave the target exactly where
/// ordinary decoding of those `committed` tokens would, for every prefix.
/// Skipping the restore must be observable, or the test proves nothing.
/// On-demand K/V growth must be invisible: a model whose caches start at
/// two tokens and double repeatedly must produce bitwise the logits of
/// one allocated for the whole context, through prefill blocks, plain
/// decode steps, and a speculative round (which grows the head's cache
/// too). The growth path really runs: allocation is checked at each stage.
#[test]
#[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device"]
fn kv_growth_reproduces_full_allocation_bitwise_across_prefill_decode_and_speculation() {
    let path = std::env::var("BONSAI_GGUF").unwrap_or_else(|_| DEFAULT_BONSAI_GGUF.into());
    let head = std::env::var("BONSAI_MTP_HEAD")
        .unwrap_or_else(|_| crate::bonsai_mtp::DEFAULT_BONSAI_MTP_HEAD.into());
    let settings = MtpSettings::new(head.into(), 1).expect("head settings");
    let context = 40;
    let load = |initial_tokens| {
        let package = BonsaiPackage::open(&path).expect("open Bonsai GGUF");
        BonsaiModel::load(
            package,
            context,
            8,
            None,
            Some(&settings),
            NgramSettings::default(),
            KvOptions {
                layout: KvLayout::F16,
                initial_tokens,
            },
        )
        .expect("load")
    };
    let mut full = load(context);
    let mut growing = load(2);
    assert_eq!(full.kv_allocated(), context);
    assert_eq!(growing.kv_allocated(), 2);

    let prompt = [5u32, 8, 12, 33, 21, 7, 44, 91, 3, 15, 27, 60, 2];
    let steps = [9u32, 18, 36];
    let params = crate::sampler::SamplingParams {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        eos_tokens: Vec::new(),
        seed: 0,
    };
    let run = |model: &mut BonsaiModel| -> (Vec<Vec<f32>>, Vec<u32>, Vec<usize>) {
        model.reset();
        let mut logits = Vec::new();
        let mut allocations = Vec::new();
        model.prefill(&prompt, &mut |_| {}).expect("prefill");
        logits.push(model.scratch.logits.as_slice::<f32>().to_vec());
        allocations.push(model.kv_allocated());
        for &step in &steps {
            model.decode(step).expect("decode");
            logits.push(model.scratch.logits.as_slice::<f32>().to_vec());
            allocations.push(model.kv_allocated());
        }
        let mut sampler = Sampler::new(VOCAB, params.clone());
        let mut committed = Vec::new();
        for seed in [11u32, 23] {
            let batch = model
                .speculative_step(seed, &mut sampler, 8)
                .expect("speculative round");
            committed.extend(batch.samples.iter().map(|sample| sample.token_id));
            allocations.push(model.kv_allocated());
        }
        (logits, committed, allocations)
    };
    let (full_logits, full_tokens, full_allocations) = run(&mut full);
    let (grown_logits, grown_tokens, grown_allocations) = run(&mut growing);

    assert!(full_allocations.iter().all(|&tokens| tokens == context));
    // 2 -> 4 -> 8 -> 16 after the 13-token prompt, then 32 once decode
    // passes 16, then the context (40) when speculation needs row 33+.
    assert_eq!(grown_allocations[0], 16, "{grown_allocations:?}");
    assert_eq!(grown_allocations[3], 16, "{grown_allocations:?}");
    assert!(
        *grown_allocations.last().expect("rounds") > 16,
        "{grown_allocations:?}"
    );
    assert_eq!(growing.position, full.position);
    assert!(full_tokens.len() >= 2, "{full_tokens:?}");
    assert_eq!(grown_tokens, full_tokens);
    for (stage, (full, grown)) in full_logits.iter().zip(&grown_logits).enumerate() {
        let bits = |row: &Vec<f32>| row.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(full), bits(grown), "logits differ at stage {stage}");
    }
    // Growth must not have discarded written rows: continue both models
    // and require agreement again after the final growth step.
    full.decode(4).expect("decode");
    growing.decode(4).expect("decode");
    assert_eq!(
        full.scratch.logits.as_slice::<f32>(),
        growing.scratch.logits.as_slice::<f32>()
    );
}

#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
#[allow(clippy::too_many_lines)]
fn verify_block_restores_every_accepted_prefix_like_tokenwise_decoding() {
    // Block-vs-tokenwise F32 kernels differ slightly; a wrong restore
    // differs by whole units (see the negative control below).
    const ROW_TOLERANCE: f32 = 1e-2;
    let path = std::env::var("BONSAI_GGUF").unwrap_or_else(|_| DEFAULT_BONSAI_GGUF.into());
    let package = BonsaiPackage::open(&path).expect("open Bonsai GGUF");
    let mut model = BonsaiModel::load(
        package,
        96,
        8,
        None,
        None,
        NgramSettings {
            enabled: true,
            max_drafts: 64,
            min_match: 8,
        },
        KvOptions::default(),
    )
    .expect("load target");
    let prefix = [5u32, 8, 12];
    let block = (0..65)
        .map(|row| 7 + (row * 13 % 101) as u32)
        .collect::<Vec<_>>(); // seed plus the maximum 64 drafts
    let next = 44u32;
    let mut verifier = model.ngram_verifier.take().expect("verifier");
    verifier
        .reserve(&model.context, block.len())
        .expect("grow verifier");

    let rows = tokenwise_logits(&mut model, &prefix, &block);
    let accepted_prefixes = [1usize, 2, 4, 8, 16, 32, 64, 65];
    let continuations = accepted_prefixes
        .iter()
        .copied()
        .map(|committed| {
            let mut steps = block[..committed].to_vec();
            steps.push(next);
            tokenwise_logits(&mut model, &prefix, &steps)
                .pop()
                .expect("continuation logits")
        })
        .collect::<Vec<_>>();

    let run_verify_block = |model: &mut BonsaiModel| -> usize {
        model.reset();
        model
            .forward_block(&prefix, BlockOutput::None)
            .expect("prefix block");
        let start = model.position;
        model
            .forward_block(&block, BlockOutput::Verify(&verifier))
            .expect("verify block");
        assert_eq!(model.position, start + block.len());
        start
    };

    let mut evidence = serde_json::Map::new();
    run_verify_block(&mut model);
    let row_reports = rows
        .iter()
        .enumerate()
        .map(|(row, expected)| {
            let start = row * VOCAB;
            compare(
                &verifier.verify_logits.as_slice::<f32>()[start..start + VOCAB],
                expected,
            )
        })
        .collect::<Vec<_>>();
    for report in &row_reports {
        assert!(
            report["same_argmax"].as_bool() == Some(true)
                && report["max_abs"].as_f64().expect("max_abs") < f64::from(ROW_TOLERANCE),
            "verify row differs from tokenwise decoding: {report}"
        );
    }
    evidence.insert("verify_rows".into(), row_reports.into());

    let mut rollback_reports = Vec::new();
    for (committed, expected) in accepted_prefixes.into_iter().zip(&continuations) {
        let start = run_verify_block(&mut model);
        if committed < block.len() {
            model
                .restore_checkpoint(&verifier, committed)
                .expect("restore checkpoint");
            model.position = start + committed;
        }
        model.forward(next, true).expect("decode after rollback");
        let report = compare(model.scratch.logits.as_slice::<f32>(), expected);
        assert!(
            report["same_argmax"].as_bool() == Some(true)
                && report["max_abs"].as_f64().expect("max_abs") < f64::from(ROW_TOLERANCE),
            "decode after committing {committed} rows differs from tokenwise: {report}"
        );
        rollback_reports.push(serde_json::json!({"committed": committed, "report": report}));
    }
    evidence.insert("rollbacks".into(), rollback_reports.into());

    // Negative control: rewind the position but keep the recurrent state
    // of all three rows. The recurrent layers must make this visible.
    let start = run_verify_block(&mut model);
    model.position = start + 1;
    model.forward(next, true).expect("decode without restore");
    let control = compare(model.scratch.logits.as_slice::<f32>(), &continuations[0]);
    assert!(
        control["max_abs"].as_f64().expect("max_abs") > 10.0 * f64::from(ROW_TOLERANCE),
        "skipping the restore was not observable: {control}"
    );
    evidence.insert("no_restore_control".into(), control);
    evidence.insert(
        "setup".into(),
        serde_json::json!({
            "model": path, "context": 96, "prefill_chunk": 8,
            "prefix": prefix, "block": block, "next": next,
            "row_tolerance": ROW_TOLERANCE,
        }),
    );
    let evidence = serde_json::Value::Object(evidence);
    eprintln!("{evidence:#}");
    if let Some(output) = std::env::var_os("BONSAI_VERIFY_EVIDENCE") {
        std::fs::write(output, format!("{evidence:#}\n")).expect("write evidence");
    }
}

/// Poison the head's caches and predicted scratch so anything the next
/// encode leaves untouched is visibly NaN.
fn poison_head(speculation: &mut Speculation) {
    let (keys, values, predicted) = speculation.mtp.cache_buffers_mut();
    for cache in [keys, values] {
        cache.as_mut_slice::<half::f16>().fill(half::f16::NAN);
    }
    predicted.as_mut_slice::<f32>().fill(f32::NAN);
}

/// Key cache bits, value cache bits, and whether each predicted element is NaN.
fn head_bits(speculation: &Speculation) -> (Vec<u16>, Vec<u16>, Vec<bool>) {
    let bits = |cache: &MetalBuffer| {
        cache
            .as_slice::<half::f16>()
            .iter()
            .copied()
            .map(half::f16::to_bits)
            .collect::<Vec<_>>()
    };
    let (keys, values) = speculation.mtp.kv_caches();
    (
        bits(keys),
        bits(values),
        speculation
            .mtp
            .predicted()
            .as_slice::<f32>()
            .iter()
            .map(|value| value.is_nan())
            .collect(),
    )
}

/// Draft one token from the head's current committed hidden and K/V
/// state, exactly as `speculative_round` does for its first draft row.
fn draft_logits(model: &mut BonsaiModel, seed: u32) -> Vec<f32> {
    let mut speculation = model.speculation.take().expect("speculation");
    decode_embeddings(
        &model.package,
        &[seed],
        speculation.mtp.embedding_rows(1).expect("embedding row"),
    )
    .expect("seed embedding");
    let mut batch = CommandBatch::new(&model.context).expect("batch");
    speculation
        .mtp
        .encode(
            &mut batch,
            &model.mtp_shared(),
            speculation.mtp.prev_hidden(),
            model.position,
            1,
            true,
        )
        .expect("draft encode");
    batch.commit_and_wait().expect("draft completion");
    let logits = speculation.mtp.logits.as_slice::<f32>().to_vec();
    model.speculation = Some(speculation);
    logits
}

/// K/V-only ingestion must leave the head exactly where the full layer would.
///
/// Checked: bit-identical F16 cache rows for the committed prefix, nothing
/// written elsewhere, `predicted` untouched, and the same next draft. A
/// different last prefix token must change that draft, or equal drafts
/// would prove nothing about the cache.
#[test]
#[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device"]
#[allow(clippy::too_many_lines)]
fn kv_only_ingestion_matches_full_head_cache_and_next_draft() {
    let path = std::env::var("BONSAI_GGUF").unwrap_or_else(|_| DEFAULT_BONSAI_GGUF.into());
    let head = std::env::var("BONSAI_MTP_HEAD")
        .unwrap_or_else(|_| crate::bonsai_mtp::DEFAULT_BONSAI_MTP_HEAD.into());
    let settings = MtpSettings::new(head.clone().into(), 1).expect("head settings");
    let package = BonsaiPackage::open(&path).expect("open Bonsai GGUF");
    let mut model = BonsaiModel::load(
        package,
        32,
        8,
        None,
        Some(&settings),
        NgramSettings::default(),
        KvOptions::default(),
    )
    .expect("load");
    let prefix = [5u32, 8, 12];
    let control_prefix = [5u32, 8, 200];
    let seed = 7u32;
    let cache_rows = KV_TOKEN_BYTES / 2;
    let written = 0..prefix.len() * cache_rows;

    // K/V-only path: what generation actually runs after each prefill block.
    model.reset();
    model
        .forward_block(&prefix, BlockOutput::Hidden)
        .expect("prefix block");
    poison_head(model.speculation.as_mut().expect("speculation"));
    let started = Instant::now();
    model.ingest_committed(&prefix).expect("kv-only ingestion");
    let kv_only_seconds = started.elapsed().as_secs_f64();
    let (kv_keys, kv_values, kv_predicted_nan) =
        head_bits(model.speculation.as_ref().expect("speculation"));
    let kv_prev_hidden = model
        .speculation
        .as_ref()
        .expect("speculation")
        .mtp
        .prev_hidden()
        .as_slice::<f32>()
        .to_vec();
    let kv_draft = draft_logits(&mut model, seed);

    // Full head over the same rows: the previous behaviour of `logits=false`.
    model.reset();
    model
        .forward_block(&prefix, BlockOutput::Hidden)
        .expect("prefix block");
    let mut speculation = model.speculation.take().expect("speculation");
    poison_head(&mut speculation);
    let started = Instant::now();
    decode_embeddings(
        &model.package,
        &prefix,
        speculation
            .mtp
            .embedding_rows(prefix.len())
            .expect("embedding rows"),
    )
    .expect("prefix embeddings");
    {
        let mtp = &speculation.mtp;
        let mut batch = CommandBatch::new(&model.context).expect("batch");
        mtp.stage_hidden(&mut batch, &model.scratch.normalized, prefix.len())
            .expect("stage hidden");
        mtp.encode(
            &mut batch,
            &model.mtp_shared(),
            mtp.hidden_in(),
            0,
            prefix.len(),
            true,
        )
        .expect("full encode");
        mtp.commit_hidden(&mut batch, &model.scratch.normalized, prefix.len() - 1)
            .expect("commit hidden");
        batch.commit_and_wait().expect("full completion");
    }
    let full_seconds = started.elapsed().as_secs_f64();
    let (full_keys, full_values, full_predicted_nan) = head_bits(&speculation);
    let full_prev_hidden = speculation.mtp.prev_hidden().as_slice::<f32>().to_vec();
    model.speculation = Some(speculation);
    let full_draft = draft_logits(&mut model, seed);

    assert_eq!(kv_keys.len(), full_keys.len());
    for i in 0..kv_keys.len() {
        if written.contains(&i) {
            assert_eq!(kv_keys[i], full_keys[i], "key cache element {i}");
            assert_eq!(kv_values[i], full_values[i], "value cache element {i}");
            assert!(
                !half::f16::from_bits(kv_keys[i]).is_nan()
                    && !half::f16::from_bits(kv_values[i]).is_nan(),
                "committed row left poisoned at {i}"
            );
        } else {
            assert_eq!(
                kv_keys[i],
                half::f16::NAN.to_bits(),
                "kv-only path wrote outside the block at {i}"
            );
            assert_eq!(
                kv_values[i],
                half::f16::NAN.to_bits(),
                "kv-only path wrote outside the block at {i}"
            );
        }
    }
    assert!(
        kv_predicted_nan.iter().all(|&nan| nan),
        "kv-only path must not touch the predicted hidden"
    );
    // The full layer writes exactly one predicted row per encoded token;
    // the scratch tail beyond the block stays poisoned on both paths.
    let predicted_rows = prefix.len() * WIDTH;
    assert!(
        full_predicted_nan[..predicted_rows].iter().all(|&nan| !nan),
        "full path must fill the predicted hidden for every encoded row"
    );
    assert!(
        full_predicted_nan[predicted_rows..].iter().all(|&nan| nan),
        "full path wrote predicted rows beyond the block"
    );
    assert_eq!(
        kv_prev_hidden
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        full_prev_hidden
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        "committed hidden handoff differs"
    );
    // Identical caches and hidden handoff feed identical draft kernels, so
    // the drafts should agree far below the control's separation; the
    // observed max_abs is recorded in the evidence rather than assumed zero.
    let draft_report = compare(&kv_draft, &full_draft);
    assert!(
        draft_report["same_argmax"].as_bool() == Some(true)
            && draft_report["max_abs"].as_f64().expect("max_abs") <= 1e-5,
        "next draft differs between kv-only and full ingestion: {draft_report}"
    );

    // Negative control: a different committed token must reach the draft
    // through the cache, so a path that wrote nothing would fail here.
    model.reset();
    model
        .forward_block(&control_prefix, BlockOutput::Hidden)
        .expect("control block");
    model
        .ingest_committed(&control_prefix)
        .expect("control ingestion");
    let control_draft = draft_logits(&mut model, seed);
    let control = compare(&control_draft, &full_draft);
    assert!(
        control["max_abs"].as_f64().expect("max_abs") > 1e-2,
        "changing a committed token was not visible in the draft: {control}"
    );

    let evidence = serde_json::json!({
        "setup": {"model": path, "head": head, "context": 32, "prefill_chunk": 8,
                  "prefix": prefix, "control_prefix": control_prefix, "seed": seed},
        "cache_rows_bitwise_equal": prefix.len(),
        "draft": draft_report,
        "control": control,
        "ingest_seconds": {"kv_only": kv_only_seconds, "full_head": full_seconds},
    });
    eprintln!("{evidence:#}");
    if let Some(output) = std::env::var_os("BONSAI_KV_EVIDENCE") {
        std::fs::write(output, format!("{evidence:#}\n")).expect("write evidence");
    }
}

/// Chunked prefill with the head fused into each target batch must leave
/// the head exactly where separate ingestion batches do, and within F16
/// rounding of one block over the whole prompt.
///
/// Intermediate chunks produce no logits, so they must still output-
/// normalize every row for the head. Bitwise equality is only expected
/// between paths with the same chunking (the target's prefill kernels
/// reduce differently per block width); the single-block reference bounds
/// the ingested rows instead. The witness reproduces the former behaviour
/// (`BlockOutput::None` before ingestion, which leaves the last layer's
/// post-attention-norm rows in `scratch.normalized`) and must land far
/// outside that bound, or a loose bound would prove nothing.
#[test]
#[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device"]
#[allow(clippy::too_many_lines)]
fn chunked_prefill_ingests_output_normalized_rows_for_every_chunk() {
    struct HeadState {
        keys: Vec<f32>,
        values: Vec<f32>,
        prev_hidden: Vec<f32>,
        draft: Vec<f32>,
    }
    let path = std::env::var("BONSAI_GGUF").unwrap_or_else(|_| DEFAULT_BONSAI_GGUF.into());
    let head = std::env::var("BONSAI_MTP_HEAD")
        .unwrap_or_else(|_| crate::bonsai_mtp::DEFAULT_BONSAI_MTP_HEAD.into());
    let settings = MtpSettings::new(head.into(), 1).expect("head settings");
    let package = BonsaiPackage::open(&path).expect("open Bonsai GGUF");
    let mut model = BonsaiModel::load(
        package,
        32,
        8,
        None,
        Some(&settings),
        NgramSettings::default(),
        KvOptions::default(),
    )
    .expect("load");
    let prompt = [5u32, 8, 12, 17, 23, 31, 40, 52, 61, 77, 90, 101];
    let chunk = model.info.prefill_chunk_size;
    assert!(prompt.len() > chunk);
    let seed = 7u32;
    let cache_rows = KV_TOKEN_BYTES / 2;
    let written = 0..prompt.len() * cache_rows;

    let snapshot = |model: &mut BonsaiModel| {
        let speculation = model.speculation.as_ref().expect("speculation");
        let (keys, values) = speculation.mtp.kv_caches();
        let floats = |cache: &MetalBuffer| {
            cache.as_slice::<half::f16>()[written.clone()]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>()
        };
        HeadState {
            keys: floats(keys),
            values: floats(values),
            prev_hidden: speculation.mtp.prev_hidden().as_slice::<f32>().to_vec(),
            draft: draft_logits(model, seed),
        }
    };
    let max_abs = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();

    // Production path: chunked prefill with fused head ingestion.
    model.reset();
    poison_head(model.speculation.as_mut().expect("speculation"));
    model
        .prefill(&prompt, &mut |_| {})
        .expect("chunked prefill");
    assert_eq!(model.position, prompt.len());
    let fused = snapshot(&mut model);
    assert!(
        fused
            .keys
            .iter()
            .chain(&fused.values)
            .all(|v| v.is_finite()),
        "poisoned cache rows survived fused ingestion"
    );

    // Same chunking, separate ingestion batches: must be bit-identical.
    let (first, rest) = prompt.split_at(chunk);
    model.reset();
    poison_head(model.speculation.as_mut().expect("speculation"));
    model
        .forward_block(first, BlockOutput::Hidden)
        .expect("two-step first chunk");
    model.ingest_committed(first).expect("two-step ingestion");
    model
        .forward_block(rest, BlockOutput::LastLogits)
        .expect("two-step last chunk");
    model
        .ingest_committed(rest)
        .expect("two-step last ingestion");
    let two_step = snapshot(&mut model);
    assert_eq!(bits(&fused.keys), bits(&two_step.keys), "fused key cache");
    assert_eq!(
        bits(&fused.values),
        bits(&two_step.values),
        "fused value cache"
    );
    assert_eq!(
        bits(&fused.prev_hidden),
        bits(&two_step.prev_hidden),
        "fused hidden handoff"
    );
    let fused_draft = compare(&fused.draft, &two_step.draft);
    assert!(
        fused_draft["same_argmax"].as_bool() == Some(true)
            && fused_draft["max_abs"].as_f64().expect("max_abs") <= 1e-5,
        "fused ingestion changed the next draft: {fused_draft}"
    );

    // Witness of the former behaviour: no output norm before ingestion.
    model.reset();
    poison_head(model.speculation.as_mut().expect("speculation"));
    model
        .forward_block(first, BlockOutput::None)
        .expect("witness first chunk");
    model.ingest_committed(first).expect("witness ingestion");
    model
        .forward_block(rest, BlockOutput::LastLogits)
        .expect("witness last chunk");
    model
        .ingest_committed(rest)
        .expect("witness last ingestion");
    let witness = snapshot(&mut model);
    drop(model);

    // Single-block reference from a model whose block holds the prompt.
    let package = BonsaiPackage::open(&path).expect("open Bonsai GGUF");
    let mut wide = BonsaiModel::load(
        package,
        32,
        16,
        None,
        Some(&settings),
        NgramSettings::default(),
        KvOptions::default(),
    )
    .expect("load");
    poison_head(wide.speculation.as_mut().expect("speculation"));
    wide.forward_block(&prompt, BlockOutput::LastLogits)
        .expect("single block");
    wide.ingest_committed(&prompt).expect("single ingestion");
    let single = snapshot(&mut wide);

    let first_rows = 0..first.len() * cache_rows;
    let fused_key_gap = max_abs(
        &fused.keys[first_rows.clone()],
        &single.keys[first_rows.clone()],
    );
    let fused_value_gap = max_abs(
        &fused.values[first_rows.clone()],
        &single.values[first_rows.clone()],
    );
    let witness_key_gap = max_abs(
        &witness.keys[first_rows.clone()],
        &single.keys[first_rows.clone()],
    );
    let witness_value_gap = max_abs(
        &witness.values[first_rows.clone()],
        &single.values[first_rows],
    );
    let key_scale = single.keys.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let value_scale = single.values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let tolerance = 2e-2;
    assert!(
        fused_key_gap <= tolerance * key_scale && fused_value_gap <= tolerance * value_scale,
        "chunked ingestion left the single-block reference: keys {fused_key_gap} of {key_scale}, values {fused_value_gap} of {value_scale}"
    );
    assert!(
        witness_key_gap > 10.0 * fused_key_gap.max(1e-3 * key_scale)
            && witness_value_gap > 10.0 * fused_value_gap.max(1e-3 * value_scale),
        "post-attention-norm rows were indistinguishable from output-normalized rows: witness keys {witness_key_gap}, values {witness_value_gap}"
    );
    let single_draft = compare(&fused.draft, &single.draft);
    assert!(
        single_draft["same_argmax"].as_bool() == Some(true),
        "chunked prefill drafts differently from a single block: {single_draft}"
    );
    eprintln!(
        "{:#}",
        serde_json::json!({
            "setup": {"model": path, "context": 32, "prefill_chunk": chunk, "prompt": prompt},
            "fused_vs_two_step": {"bitwise_equal_rows": prompt.len(), "draft": fused_draft},
            "fused_vs_single_block": {
                "key_gap": fused_key_gap, "value_gap": fused_value_gap,
                "key_scale": key_scale, "value_scale": value_scale, "draft": single_draft,
            },
            "witness_vs_single_block": {"key_gap": witness_key_gap, "value_gap": witness_value_gap},
        })
    );
}

/// Diagnostic, not a check: prefill `BONSAI_KV_DUMP_PROMPT` (chat-wrapped,
/// thinking on, like the CLI) and write every full-attention layer's F16
/// K/V prefix to `BONSAI_KV_DUMP_DIR/layer_<index>_{k,v}.f16` so the
/// cache's real bit-pattern statistics can be measured offline.
#[test]
#[ignore = "requires the Bonsai GGUF, a Metal device and BONSAI_KV_DUMP_DIR"]
fn dump_full_attention_kv_caches() {
    let Ok(dir) = std::env::var("BONSAI_KV_DUMP_DIR") else {
        eprintln!("BONSAI_KV_DUMP_DIR unset; nothing dumped");
        return;
    };
    let prompt_path = std::env::var("BONSAI_KV_DUMP_PROMPT").expect("BONSAI_KV_DUMP_PROMPT");
    let path = std::env::var("BONSAI_GGUF").unwrap_or_else(|_| DEFAULT_BONSAI_GGUF.into());
    let package = BonsaiPackage::open(&path).expect("open Bonsai GGUF");
    let tokenizer = BonsaiTokenizer::from_package(&package).expect("tokenizer");
    let text = std::fs::read_to_string(prompt_path).expect("prompt text");
    let prompt = tokenizer
        .encode(&BonsaiTokenizer::chat_prompt(&text, true).expect("chat prompt"))
        .expect("encode");
    let capacity = prompt.len().next_multiple_of(128);
    let mut model = BonsaiModel::load(
        package,
        capacity,
        128,
        None,
        None,
        NgramSettings::default(),
        KvOptions::default(),
    )
    .expect("load");
    model.prefill(&prompt, &mut |_| {}).expect("prefill");
    std::fs::create_dir_all(&dir).expect("dump dir");
    let bytes = prompt.len() * KV_TOKEN_BYTES;
    let mut dumped = 0usize;
    for (index, layer) in model.layers.iter().enumerate() {
        let AttentionLayer::Full(full) = &layer.attention else {
            continue;
        };
        for (name, cache) in [("k", &full.key_cache), ("v", &full.value_cache)] {
            let file = format!("{dir}/layer_{index:02}_{name}.f16");
            std::fs::write(&file, &cache.as_slice::<u8>()[..bytes]).expect("write dump");
        }
        dumped += 1;
    }
    eprintln!(
        "dumped {dumped} full-attention layers × {} tokens × {} bytes to {dir}",
        prompt.len(),
        KV_TOKEN_BYTES
    );
    assert_eq!(dumped, LAYERS / FULL_INTERVAL);
}
