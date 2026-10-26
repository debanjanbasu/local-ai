use super::*;
use crate::bonsai_mtp::MtpMode;
use crate::bonsai_native::KvOptions;
use crate::bonsai_ngram::NgramSettings;
use std::fmt::Write as _;

#[test]
fn requests_reject_truncation_overflow_and_invalid_ids() {
    assert!(validate_request(&[5, 8, 12], 5, 8, None).is_ok());
    assert!(validate_request(&[5, 8, 12], 6, 8, None).is_err());
    assert!(validate_request(&[5], usize::MAX, 8, None).is_err());
    assert!(validate_request(&[], 5, 8, None).is_err());
    assert!(validate_request(&[VOCAB as u32], 1, 8, None).is_err());
    assert!(validate_request(&[VOCAB as u32 - 1], 0, 1, None).is_ok());
}

#[test]
fn draft_depth_is_bounded_by_request_then_context() {
    // Plenty of room on both sides: the configured depth wins.
    assert_eq!(draft_depth(2, 10, 100, 0), 2);
    assert_eq!(draft_depth(4, 10, 100, 0), 4);
    // Only the final token remains (or nothing): no drafting, no underflow.
    assert_eq!(draft_depth(2, 1, 100, 0), 0);
    assert_eq!(draft_depth(2, 0, 100, 0), 0);
    // Two tokens remain: exactly one draft rides along with the seed.
    assert_eq!(draft_depth(4, 2, 100, 0), 1);
    // Seed lands on the last context slot or beyond: nothing may follow it.
    assert_eq!(draft_depth(2, 10, 100, 99), 0);
    assert_eq!(draft_depth(2, 10, 100, 100), 0);
    assert_eq!(draft_depth(2, 10, 100, 250), 0);
    // One slot left after the seed.
    assert_eq!(draft_depth(4, 10, 100, 98), 1);
    // Both limits bind, asymmetrically: request allows 2, context allows 1.
    assert_eq!(draft_depth(4, 3, 100, 98), 1);
    // ... and the other way round: context allows 2, request allows 1.
    assert_eq!(draft_depth(4, 2, 100, 97), 1);
}

#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn bonsai_ngram_repetitive_greedy_matches_plain_tokens() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let prompt =
        "Repeat this sequence exactly twice: alpha beta gamma delta alpha beta gamma delta";
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 48,
        ..GenerateParams::default()
    };
    let run = |ngram| {
        let mut engine = BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(512),
            DEFAULT_PREFILL_CHUNK,
            None,
            &MtpMode::Off(None),
            ngram,
            KvOptions::default(),
        )
        .expect("load model");
        let ids = engine.encode_prompt(prompt, false, false).expect("prompt");
        engine
            .generate(&ids, &params, |_| true)
            .expect("generate")
            .token_ids
    };
    let plain = run(NgramSettings {
        enabled: false,
        ..NgramSettings::default()
    });
    let ngram = run(NgramSettings {
        enabled: true,
        ..NgramSettings::default()
    });
    assert_eq!(plain, ngram);
}

/// A shared prefix and a mid-prompt edit both decode to exactly the tokens a
/// cold engine produces for the same prompt.
///
/// Reuse *counts* are deliberately not asserted once a second engine is live.
/// `set_prompt_cache_checkpoints(4)` is a purgeable LRU and every cold
/// reference below holds another full engine open, whose unified-memory
/// pressure can evict those volatile checkpoints mid-test, so
/// `reused_prompt_tokens > 0` measures this test's own memory footprint rather
/// than cache correctness. What must hold either way is token equality.
/// `purged_gpu_and_host_snapshots_fall_back_to_disk_with_identical_tokens`
/// covers the purge itself and asserts the tokens stay identical to cold.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn prompt_checkpoints_match_cold_greedy_for_extension_and_mid_prompt_edit() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 8,
        ..GenerateParams::default()
    };
    let settings = NgramSettings {
        enabled: false,
        ..NgramSettings::default()
    };
    let open = || {
        BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(512),
            DEFAULT_PREFILL_CHUNK,
            None,
            &MtpMode::default(),
            settings,
            KvOptions::default(),
        )
        .expect("load model")
    };
    let extended = "prefix shared by all requests: alpha beta gamma delta. More context follows: epsilon zeta eta. Answer briefly:";
    let edited = "prefix shared by all requests: alpha beta gamma delta. More context follows: EPSILON CHANGED eta. Answer briefly:";
    let mut cached = open();
    cached
        .set_prompt_cache_checkpoints(4)
        .expect("enable cache");
    let extended_ids = cached
        .encode_prompt(extended, true, false)
        .expect("extended prompt");
    let first_ids = extended_ids[..extended_ids.len() / 2].to_vec();
    cached
        .generate(&first_ids, &params, |_| true)
        .expect("prime cache");
    let repeated = cached
        .generate(&first_ids, &params, |_| true)
        .expect("repeat prompt");
    // Only the upper bound is asserted. The exact count depends on volatile GPU
    // checkpoints surviving, and every ignored test in this binary maps and
    // uploads the whole checkpoint into a 16 GB unified memory, so cumulative
    // pressure from earlier tests in the same process can evict them. The bound
    // holds unconditionally because `reusable` is `lcp.min(len - 1)`, while the
    // exact figure is a residency diagnostic, not a correctness property. Real
    // reuse counts are asserted by
    // `shared_system_prefix_reuses_across_sessions_and_restart`.
    assert!(repeated.stats.reused_prompt_tokens < first_ids.len());
    // Cold references are temporaries: an extra live engine adds memory
    // pressure that can purge the cached engine's volatile checkpoints, which
    // is why no reuse count below is asserted past the first two requests.
    assert_eq!(
        repeated.token_ids,
        open()
            .generate(&first_ids, &params, |_| true)
            .expect("cold repeat")
            .token_ids
    );
    for ids in [
        extended_ids,
        cached
            .encode_prompt(edited, true, false)
            .expect("edited prompt"),
    ] {
        let cached_output = cached
            .generate(&ids, &params, |_| true)
            .expect("cached run");
        // Reuse may be zero here — the cold engine below can purge the volatile
        // checkpoints — but it can never exceed the prompt's own reusable
        // prefix, so the bound survives a purge while still catching a
        // bookkeeping count that claims tokens this prompt does not contain.
        assert!(cached_output.stats.reused_prompt_tokens < ids.len());
        let cold_output = open().generate(&ids, &params, |_| true).expect("cold run");
        assert_eq!(cached_output.token_ids, cold_output.token_ids);
    }
}

#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn interleaved_and_restarted_session_snapshots_match_cold_tokens() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 4,
        ..GenerateParams::default()
    };
    let open = || {
        BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(512),
            DEFAULT_PREFILL_CHUNK,
            None,
            &MtpMode::Off(None),
            NgramSettings::default(),
            KvOptions::default(),
        )
        .expect("load model")
    };
    let cache = tempfile::tempdir().expect("cache directory");
    let mut engine = open();
    engine.set_prompt_cache_checkpoints(4).expect("checkpoints");
    engine.set_session_cache(1024, Some(cache.path().to_owned()));
    let a = engine
        .encode_prompt("session A exact prefix", true, false)
        .expect("A");
    let mut extended_a = a.clone();
    extended_a.push(42);
    let b = engine
        .encode_prompt("session B independent prefix", true, false)
        .expect("B");
    engine
        .generate_session(&a, &params, Some("a"), |_| true)
        .expect("A1");
    engine
        .generate_session(&b, &params, Some("b"), |_| true)
        .expect("B1");
    let resumed = engine
        .generate_session(&extended_a, &params, Some("a"), |_| true)
        .expect("A2");
    // In-process resume may land on either tier. The host snapshot is preferred,
    // but restoring it copies KV state to the GPU, so under memory pressure the
    // restore can fail and `prepare_prompt` falls through to the disk snapshot it
    // also wrote. Both are correct resumes; the tokens are what must match.
    assert!(
        matches!(
            resumed.cache_source,
            PromptCacheSource::Host | PromptCacheSource::Disk
        ),
        "in-process resume used {:?}",
        resumed.cache_source
    );
    let cold_tokens = open()
        .generate(&extended_a, &params, |_| true)
        .expect("cold A2")
        .token_ids;
    assert_eq!(resumed.token_ids, cold_tokens);

    drop(engine);
    let mut restarted = open();
    restarted
        .set_prompt_cache_checkpoints(4)
        .expect("checkpoints");
    restarted.set_session_cache(1024, Some(cache.path().to_owned()));
    let disk = restarted
        .generate_session(&extended_a, &params, Some("a"), |_| true)
        .expect("disk A");
    assert_eq!(disk.cache_source, PromptCacheSource::Disk);
    assert_eq!(disk.token_ids, cold_tokens);
}

#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn shared_system_prefix_reuses_across_sessions_and_restart() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 4,
        ..GenerateParams::default()
    };
    let open = || {
        BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(1024),
            DEFAULT_PREFILL_CHUNK,
            None,
            &MtpMode::Off(None),
            NgramSettings::default(),
            KvOptions::default(),
        )
        .expect("load model")
    };
    let system = "Rust ownership, borrowing, lifetimes, traits, and safe concurrency. ".repeat(48);
    let rendered = |question: &str| {
        format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    };
    let cache = tempfile::tempdir().expect("cache directory");
    let mut cached = open();
    cached.set_prompt_cache_checkpoints(4).expect("checkpoints");
    cached.set_session_cache(1024, Some(cache.path().to_owned()));
    cached.prompt_cache_disk_bytes = u64::MAX;
    let first = cached
        .tokenizer
        .encode(&rendered("Explain Arc briefly."))
        .expect("first prompt");
    let second = cached
        .tokenizer
        .encode(&rendered("Explain Pin briefly."))
        .expect("second prompt");
    let boundary = first
        .iter()
        .position(|&token| token == 248_046)
        .expect("system end")
        + 1;
    assert!(boundary >= super::checkpoints::SHARED_PREFIX_MIN_TOKENS);
    cached
        .generate_session(&first, &params, Some("first"), |_| true)
        .expect("prime shared prefix");
    let reused = cached
        .generate_session(&second, &params, Some("second"), |_| true)
        .expect("reuse shared prefix");
    assert!(reused.stats.reused_prompt_tokens >= boundary);
    assert_eq!(
        reused.token_ids,
        open()
            .generate(&second, &params, |_| true)
            .expect("cold reference")
            .token_ids
    );

    drop(cached);
    let mut restarted = open();
    restarted.set_session_cache(1024, Some(cache.path().to_owned()));
    let restored = restarted
        .generate_session(&second, &params, Some("restart"), |_| true)
        .expect("disk shared prefix");
    assert_eq!(restored.cache_source, PromptCacheSource::Disk);
    assert!(restored.stats.reused_prompt_tokens >= boundary);
    assert_eq!(
        restored.token_ids,
        open()
            .generate(&second, &params, |_| true)
            .expect("cold restart reference")
            .token_ids
    );
}

/// An agent session of thinking requests: each prompt ends with the
/// generation opener `<think>\n`, which the next turn's history re-renders
/// without, so every turn diverges from the last prompt just before its end
/// and checkpoints a divergence boundary that barely extends the previous
/// one. Once such boundaries filled the GPU tier, `b09e22f` evicted every
/// new checkpoint at once and restored the newest boundary it still held, a
/// few turns old, ahead of the newer one on disk, so each turn prefilled
/// more of the conversation again (seen after a compaction in a 64K
/// session: steps grew from 30 s to 179 s). Each turn must restore the
/// boundary the previous turn checkpointed, from whichever tier holds it.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn agent_turns_keep_restoring_the_previous_prompt() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let open = || {
        BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(2048),
            DEFAULT_PREFILL_CHUNK,
            None,
            &MtpMode::Off(None),
            NgramSettings::default(),
            KvOptions::default(),
        )
        .expect("load model")
    };
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 1,
        ..GenerateParams::default()
    };
    let system = "You are a careful filing clerk who keeps an inventory. ".repeat(24);
    let mut history = format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nFile the items.<|im_end|>\n"
    );
    let cache = tempfile::tempdir().expect("cache directory");
    let mut engine = open();
    engine.set_prompt_cache_checkpoints(4).expect("checkpoints");
    engine.set_session_cache(0, Some(cache.path().to_owned()));
    let common = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut last = None;
    for turn in 1..=10 {
        let prompt = engine
            .tokenizer
            .encode(&format!("{history}<|im_start|>assistant\n<think>\n"))
            .expect("prompt");
        let run = engine
            .generate_session(&prompt, &params, Some("agent"), |_| true)
            .expect("turn");
        if let [.., before, previous] = prompts.as_slice() {
            // The boundary the previous turn checkpointed, where it diverged
            // from the turn before it.
            let boundary = common(before, previous);
            assert!(
                run.stats.reused_prompt_tokens >= boundary,
                "turn {turn} of {} tokens reused {} from {:?}, the previous boundary is {boundary}",
                prompt.len(),
                run.stats.reused_prompt_tokens,
                run.cache_source
            );
        }
        last = Some((prompt.clone(), run.token_ids));
        prompts.push(prompt);
        write!(
            history,
            "<|im_start|>assistant\nNoted: item {turn} sits in drawer {}.<|im_end|>\n\
             <|im_start|>user\nItem {} arrived; it is a box of {} pencils.<|im_end|>\n",
            turn * 7,
            turn + 1,
            turn * 12
        )
        .expect("writes to a String");
    }
    let (prompt, tokens) = last.expect("turns ran");
    assert_eq!(
        tokens,
        open()
            .generate(&prompt, &params, |_| true)
            .expect("cold reference")
            .token_ids
    );
}

/// A request that restores a shared boundary and checkpoints a new one fills
/// the GPU tier: its boundary, penultimate token, end and output end, beside
/// the restored boundary. The next turn re-renders the last prompt token, so
/// it needs the penultimate checkpoint, and the disk tier has none (a request
/// that persisted a boundary does not persist its penultimate snapshot).
/// With the prompt's end also checkpointed, the penultimate checkpoint was
/// the oldest request tail outside the two newest and was evicted, and the
/// turn restored the new boundary instead (three steps after a compaction in
/// a 64K session prefilled one turn too many).
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn a_turn_after_a_new_boundary_restores_the_penultimate_checkpoint() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let open = || {
        BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(2048),
            DEFAULT_PREFILL_CHUNK,
            None,
            &MtpMode::Off(None),
            NgramSettings::default(),
            KvOptions::default(),
        )
        .expect("load model")
    };
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 1,
        ..GenerateParams::default()
    };
    let system = "You are a careful filing clerk who keeps an inventory. ".repeat(24);
    let task = format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{}",
        "File every item in the order it arrives, and note its drawer. ".repeat(8)
    );
    let cache = tempfile::tempdir().expect("cache directory");
    let mut engine = open();
    engine.set_prompt_cache_checkpoints(4).expect("checkpoints");
    engine.set_session_cache(0, Some(cache.path().to_owned()));
    let encode = |engine: &BonsaiEngine, text: &str| engine.tokenizer.encode(text).expect("prompt");
    // The conversation before a compaction, then the compacted one: they
    // share the system turn and the task, so the second restores the
    // system-turn boundary and checkpoints its divergence from the first.
    let before = encode(
        &engine,
        &format!("{task}Item 1 is a stapler.<|im_end|>\n<|im_start|>assistant\n"),
    );
    let compacted = encode(
        &engine,
        &format!("{task}Summary: three items are filed.<|im_end|>\n<|im_start|>assistant\n"),
    );
    for prompt in [&before, &compacted] {
        engine
            .generate_session(prompt, &params, Some("agent"), |_| true)
            .expect("turn");
    }
    let mut next = compacted[..compacted.len() - 1].to_vec();
    next.extend(encode(
        &engine,
        " Item 4 is a lamp.<|im_end|>\n<|im_start|>assistant\n",
    ));
    let run = engine
        .generate_session(&next, &params, Some("agent"), |_| true)
        .expect("next turn");
    assert!(
        run.stats.reused_prompt_tokens >= compacted.len() - 1
            && run.cache_source == PromptCacheSource::Gpu,
        "reused {} from {:?}, the penultimate checkpoint is at {}",
        run.stats.reused_prompt_tokens,
        run.cache_source,
        compacted.len() - 1
    );
    assert_eq!(
        run.token_ids,
        open()
            .generate(&next, &params, |_| true)
            .expect("cold reference")
            .token_ids
    );
}

/// The GPU tier describes one buffer set, so two sessions whose turns run at
/// once take it from each other: each request's milestones and end claim it
/// for its own set and drop the other's checkpoints. With no host tier, the
/// penultimate checkpoint of a request that persisted its system-turn
/// boundary was the only copy of its tail, so the session whose turn ended
/// first lost it, and its next turn restored only the system turn. Each
/// session's next turn must restore its previous prompt's tail.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn concurrent_sessions_each_restore_their_previous_tail() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let open = || {
        BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(2048),
            DEFAULT_PREFILL_CHUNK,
            None,
            &MtpMode::Off(None),
            NgramSettings::default(),
            KvOptions::default(),
        )
        .expect("load model")
    };
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 4,
        ..GenerateParams::default()
    };
    let cache = tempfile::tempdir().expect("cache directory");
    let mut engine = open();
    engine.set_prompt_cache_checkpoints(4).expect("checkpoints");
    engine.set_session_cache(0, Some(cache.path().to_owned()));
    let encode = |engine: &BonsaiEngine, text: &str| engine.tokenizer.encode(text).expect("prompt");
    // Distinct system turns, so each request persists its own boundary.
    let sessions = [
        "You are a careful filing clerk who keeps an inventory. ",
        "You are a patient librarian who shelves returned books. ",
    ]
    .map(|role| {
        encode(
            &engine,
            &format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n\
                 <|im_start|>assistant\n",
                role.repeat(24),
                "Describe the next task in one line. ".repeat(4)
            ),
        )
    });
    let ids = sessions
        .iter()
        .zip(["a", "b"])
        .map(|(prompt, session)| {
            engine
                .admit(prompt, &params, Some(session), CancelToken::new())
                .expect("admit")
        })
        .collect::<Vec<_>>();
    let mut finished = 0;
    while finished < ids.len() {
        for (id, result) in engine.step(&mut |_, _, _| true, &mut |_, _| {}) {
            assert!(ids.contains(&id));
            result.expect("first turn");
            finished += 1;
        }
    }
    for (prompt, session) in sessions.iter().zip(["a", "b"]) {
        let mut next = prompt[..prompt.len() - 1].to_vec();
        next.extend(encode(
            &engine,
            " Then file it.<|im_end|>\n<|im_start|>assistant\n",
        ));
        let run = engine
            .generate_session(&next, &params, Some(session), |_| true)
            .expect("next turn");
        assert!(
            run.stats.reused_prompt_tokens >= prompt.len() - 1,
            "session {session} reused {} from {:?}; its previous tail is at {}",
            run.stats.reused_prompt_tokens,
            run.cache_source,
            prompt.len() - 1
        );
        assert_eq!(
            run.token_ids,
            open()
                .generate(&next, &params, |_| true)
                .expect("cold reference")
                .token_ids
        );
    }
}

#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn purged_gpu_and_host_snapshots_fall_back_to_disk_with_identical_tokens() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let open = || {
        BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(512),
            DEFAULT_PREFILL_CHUNK,
            None,
            &MtpMode::Off(None),
            NgramSettings::default(),
            KvOptions::default(),
        )
        .expect("load model")
    };
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 4,
        ..GenerateParams::default()
    };
    let cache = tempfile::tempdir().expect("cache directory");
    let mut cached = open();
    cached.set_prompt_cache_checkpoints(4).expect("checkpoints");
    cached.set_session_cache(1024, Some(cache.path().to_owned()));
    cached.prompt_cache_disk_bytes = u64::MAX;
    let prompt = cached
        .encode_prompt("purgeable cache prefix", true, false)
        .expect("prompt");
    cached
        .generate_session(&prompt, &params, Some("purge"), |_| true)
        .expect("prime cache");
    for checkpoint in &cached.prompt_checkpoints {
        checkpoint.state.discard();
    }
    for snapshot in &cached.session_snapshots {
        snapshot.state.discard();
    }
    let mut extended = prompt.clone();
    extended.push(42);
    let resumed = cached
        .generate_session(&extended, &params, Some("purge"), |_| true)
        .expect("disk fallback");
    assert_eq!(resumed.cache_source, PromptCacheSource::Disk);
    let cold = open()
        .generate(&extended, &params, |_| true)
        .expect("cold run");
    assert_eq!(resumed.token_ids, cold.token_ids);
}

/// Reading and verifying a disk snapshot (about 34 KiB a token here) used to
/// run on the engine thread inside the restoring request's admission, so a
/// stream decoding beside it emitted nothing until the whole file was read
/// and hashed. The read now runs on a helper thread: the stream keeps
/// emitting while the request waits out of prefill, the restore that follows
/// is only a copy, and both produce exactly their tokens alone. Prints the
/// stream's longest gap while the snapshot restores.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device; prints timings"]
#[allow(clippy::too_many_lines)]
fn a_stream_keeps_decoding_while_another_request_loads_a_disk_snapshot() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let mut engine = BonsaiEngine::open_with_options(
        Path::new(&path),
        Some(16_384),
        DEFAULT_PREFILL_CHUNK,
        None,
        &MtpMode::Off(None),
        NgramSettings {
            enabled: false,
            ..NgramSettings::default()
        },
        KvOptions::default(),
    )
    .expect("load model");
    let cache = tempfile::tempdir().expect("cache directory");
    engine.set_prompt_cache_checkpoints(4).expect("checkpoints");
    engine.set_session_cache(0, Some(cache.path().to_owned()));
    engine.prompt_cache_disk_bytes = u64::MAX;
    let stream_prompt = engine
        .encode_prompt(
            "Write a long story about a lighthouse keeper.",
            false,
            false,
        )
        .expect("stream prompt");
    let stream_params = GenerateParams {
        temperature: 0.0,
        max_tokens: 160,
        ..GenerateParams::default()
    };
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 4,
        ..GenerateParams::default()
    };
    let text = (0..360)
        .map(|index| format!("Line {index}: the keeper of lighthouse {index} logged calm seas."))
        .collect::<Vec<_>>()
        .join("\n");
    let long = engine
        .encode_prompt(&text, true, false)
        .expect("long prompt");
    assert!(long.len() > 6_000, "{} tokens", long.len());
    // Cold references, the second leaving the long prompt's tail on disk.
    let stream_alone = engine
        .generate(&stream_prompt, &stream_params, |_| true)
        .expect("stream alone")
        .token_ids;
    let long_alone = engine
        .generate(&long, &params, |_| true)
        .expect("long alone cold");
    assert_eq!(long_alone.stats.reused_prompt_tokens, 0);
    engine.flush_prompt_cache();
    let snapshot = engine
        .disk_snapshots
        .iter()
        .find(|entry| entry.tokens.as_slice() == &long[..long.len() - 1])
        .expect("the long prompt's tail on disk")
        .path
        .clone();
    let snapshot_bytes = std::fs::metadata(&snapshot).expect("snapshot").len();
    engine.clear_gpu_cache();

    let stream = engine
        .admit(&stream_prompt, &stream_params, None, CancelToken::new())
        .expect("admit stream");
    let mut stream_emitted = Vec::new();
    let mut outcomes = std::collections::HashMap::new();
    let step = |engine: &mut BonsaiEngine,
                emitted: &mut Vec<Instant>,
                outcomes: &mut std::collections::HashMap<u64, BonsaiGeneration>| {
        let mut others = 0;
        for (id, result) in engine.step(
            &mut |id, _, _| {
                if id == stream {
                    emitted.push(Instant::now());
                } else {
                    others += 1;
                }
                true
            },
            &mut |_, _| {},
        ) {
            outcomes.insert(id, result.expect("generation"));
        }
        others
    };
    while stream_emitted.len() < 8 {
        step(&mut engine, &mut stream_emitted, &mut outcomes);
    }
    let admitted = Instant::now();
    let restoring = engine
        .admit(&long, &params, None, CancelToken::new())
        .expect("admit restoring request");
    let admission = admitted.elapsed();
    let mut steps_while_loading = 0;
    let mut first_token = None;
    while engine.active_generations() > 0 {
        let loading = engine
            .active
            .iter()
            .any(|generation| generation.id == restoring && generation.awaiting_disk());
        let before = stream_emitted.len();
        let others = step(&mut engine, &mut stream_emitted, &mut outcomes);
        if loading && stream_emitted.len() > before {
            steps_while_loading += 1;
        }
        if others > 0 || outcomes.contains_key(&restoring) {
            first_token.get_or_insert_with(Instant::now);
        }
    }
    let first_token = first_token.expect("restoring request's first token");
    let window = stream_emitted
        .iter()
        .copied()
        .filter(|&at| at > admitted && at <= first_token)
        .collect::<Vec<_>>();
    let longest_gap = std::iter::once(admitted)
        .chain(window.iter().copied())
        .zip(window.iter().copied())
        .map(|(earlier, later)| later - earlier)
        .max()
        .unwrap_or_else(|| first_token - admitted);
    let started = Instant::now();
    prompt_cache::load(&snapshot, &engine.prompt_cache_model_key).expect("load snapshot");
    let load = started.elapsed();
    eprintln!(
        "{} MB snapshot ({} tokens): load and verify {:.0} ms; admission {:.1} ms; \
         stream emitted {} tokens in {} steps while it loaded, longest gap until the \
         restoring request's first token {:.0} ms ({:.0} ms after admission)",
        snapshot_bytes / 1_000_000,
        long.len() - 1,
        load.as_secs_f64() * 1e3,
        admission.as_secs_f64() * 1e3,
        window.len(),
        steps_while_loading,
        longest_gap.as_secs_f64() * 1e3,
        (first_token - admitted).as_secs_f64() * 1e3,
    );
    let restored = &outcomes[&restoring];
    assert_eq!(restored.cache_source, PromptCacheSource::Disk);
    assert_eq!(restored.stats.reused_prompt_tokens, long.len() - 1);
    assert!(
        steps_while_loading > 0,
        "the stream emitted nothing while the snapshot loaded"
    );
    assert_eq!(restored.token_ids, long_alone.token_ids);
    assert_eq!(outcomes[&stream].token_ids, stream_alone);
}

/// Greedy plain decode commits each token's successor pass before waiting
/// for the token (`BonsaiModel::decode_pipelined`). It must produce exactly
/// the tokens of one pass at a time, and a request ending at end of
/// sequence (whose guessed pass is rolled back), at its token limit or by
/// its client must leave the same closing checkpoint, from which the next
/// turn generates the same tokens.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn pipelined_plain_decode_matches_one_pass_at_a_time() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let mut engine = BonsaiEngine::open_with_options(
        Path::new(&path),
        Some(4096),
        DEFAULT_PREFILL_CHUNK,
        None,
        &MtpMode::Off(None),
        NgramSettings {
            enabled: false,
            ..NgramSettings::default()
        },
        KvOptions::default(),
    )
    .expect("load model");
    engine.set_prompt_cache_checkpoints(4).expect("checkpoints");
    let cases = [
        ("What is 17 times 23? Reply with the number only.", 64, None),
        ("Name three primary colours, comma separated.", 64, None),
        (
            "Write a long essay about the history of lighthouses.",
            48,
            None,
        ),
        ("Explain how a hash map handles collisions.", 64, Some(6)),
    ];
    let mut eos = 0;
    for (text, max_tokens, stop_after) in cases {
        let prompt = engine.encode_prompt(text, false, false).expect("prompt");
        let params = GenerateParams {
            temperature: 0.0,
            max_tokens,
            ..GenerateParams::default()
        };
        let mut runs = Vec::new();
        for unpipelined in [true, false] {
            engine.unpipelined_decode = unpipelined;
            engine.clear_gpu_cache();
            let mut pieces = 0;
            let run = engine
                .generate(&prompt, &params, |_| {
                    pieces += 1;
                    stop_after.is_none_or(|limit| pieces < limit)
                })
                .expect("generate");
            let represented = [prompt.as_slice(), &run.token_ids].concat();
            let closing = engine.holds_checkpoint(&represented);
            let mut next = represented;
            next.extend(
                engine
                    .encode_prompt(
                        "<|im_end|>\n<|im_start|>user\nAnd why?<|im_end|>\n\
                         <|im_start|>assistant\n",
                        true,
                        false,
                    )
                    .expect("next turn"),
            );
            let followed = engine
                .generate(&next, &params, |_| true)
                .expect("next turn");
            runs.push((
                run.token_ids,
                run.stop_reason,
                closing,
                followed.token_ids,
                followed.stats.reused_prompt_tokens,
            ));
        }
        let (tokens, stop_reason, closing, followed, reused) = &runs[0];
        eos += usize::from(*stop_reason == StopReason::Eos);
        assert_eq!(runs[0], runs[1], "{text}");
        assert!(
            *closing && *reused >= prompt.len() + tokens.len(),
            "{text}: {stop_reason:?} closing checkpoint {closing}, next turn reused {reused} \
             of {} ({} tokens followed)",
            prompt.len() + tokens.len(),
            followed.len()
        );
    }
    assert!(eos > 0, "no request ended at end of sequence");
}

/// Deep cancellation: a tripped [`CancelToken`] stops a request between GPU
/// dispatches and is a successful [`StopReason::Cancelled`], never an error.
/// It drops only the checkpoints past where it stopped: the one it restored
/// still describes intact K/V rows, so a retry restores it and generates
/// exactly what a cold run does. (It used to clear the whole GPU tier, and the
/// retry prefilled from scratch.)
///
/// The trip is taken **during prefill**. With speculation and n-gram drafting
/// off, decode calls `BonsaiModel::decode`, which polls nothing, so prefill's
/// chunk loop is the only reachable `is_cancelled()` site and a trip
/// that arrived later would run the request to `TokenLimit` and fail loudly
/// rather than pass silently. Prefill is GPU-paced — every chunk waits on its
/// command buffer — so a 250 ms producer delay lands inside a request whose
/// prefill spans four chunks, with more than an order of magnitude of margin on
/// both sides of the last poll.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn deep_cancellation_stops_in_prefill_and_a_retry_restores_what_it_kept() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let open = || {
        BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(4096),
            DEFAULT_PREFILL_CHUNK,
            None,
            &MtpMode::Off(None),
            NgramSettings {
                enabled: false,
                ..NgramSettings::default()
            },
            KvOptions::default(),
        )
        .expect("load model")
    };
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 12,
        ..GenerateParams::default()
    };
    let mut engine = open();
    engine.set_prompt_cache_checkpoints(4).expect("checkpoints");
    let text = "Rust ownership, borrowing, lifetimes, traits, and safe concurrency. \
                Explain briefly why a shared prefix cache cannot shorten the last \
                prompt token. "
        .repeat(96);
    let full = engine.encode_prompt(&text, true, false).expect("prompt");
    assert!(
        full.len() > 4 * DEFAULT_PREFILL_CHUNK,
        "the cancelled prompt must span several prefill chunks: {} tokens",
        full.len()
    );
    // Diverge from the cancelled prompt at exactly one chunk boundary. The
    // cancelled request therefore restores the warmup's penultimate checkpoint
    // and prefills past it over rows no checkpoint it keeps relies on.
    let mut warmup = full[..DEFAULT_PREFILL_CHUNK].to_vec();
    let sentinel = engine
        .encode_prompt(" alpha", true, false)
        .expect("sentinel")[0];
    assert_ne!(sentinel, full[DEFAULT_PREFILL_CHUNK]);
    warmup.push(sentinel);
    let extended = full[..4 * DEFAULT_PREFILL_CHUNK].to_vec();

    engine
        .generate(&warmup, &params, |_| true)
        .expect("warm the checkpoint tier");

    let cancel = CancelToken::new();
    let producer = {
        let token = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(250));
            token.cancel();
        })
    };
    let cancelled = engine
        .generate_session_cancellable(&extended, &params, None, |_| true, &cancel)
        .expect("cancelled request");
    producer.join().expect("cancel producer");
    assert_eq!(
        cancelled.stop_reason,
        StopReason::Cancelled,
        "a tripped token is a successful outcome, not an error"
    );
    assert!(
        cancelled.token_ids.len() < params.max_tokens,
        "cancellation stopped short of the token budget: {} tokens",
        cancelled.token_ids.len()
    );

    // The retry restores from the GPU tier (no session cache is configured,
    // so host and disk tiers are empty by construction): at least the
    // checkpoint the cancelled request restored.
    let retry = engine
        .generate(&extended, &params, |_| true)
        .expect("retry after cancellation");
    assert!(
        retry.stats.reused_prompt_tokens >= DEFAULT_PREFILL_CHUNK
            && retry.cache_source == PromptCacheSource::Gpu,
        "the retry reused {} tokens from {:?}; the checkpoint the cancelled request kept is at {}",
        retry.stats.reused_prompt_tokens,
        retry.cache_source,
        DEFAULT_PREFILL_CHUNK
    );
    assert_eq!(
        retry.token_ids,
        open()
            .generate(&extended, &params, |_| true)
            .expect("cold reference")
            .token_ids,
        "a cancelled request must not corrupt the next request's output"
    );
}
