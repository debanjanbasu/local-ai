use super::*;
use crate::bonsai_mtp::MtpMode;
use crate::bonsai_native::KvOptions;
use crate::bonsai_ngram::NgramSettings;

#[test]
fn requests_reject_truncation_overflow_and_invalid_ids() {
    assert!(validate_request(&[5, 8, 12], 5, 8).is_ok());
    assert!(validate_request(&[5, 8, 12], 6, 8).is_err());
    assert!(validate_request(&[5], usize::MAX, 8).is_err());
    assert!(validate_request(&[], 5, 8).is_err());
    assert!(validate_request(&[VOCAB as u32], 1, 8).is_err());
    assert!(validate_request(&[VOCAB as u32 - 1], 0, 1).is_ok());
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
    assert!(boundary >= DEFAULT_PREFILL_CHUNK);
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

/// Deep cancellation: a tripped [`CancelToken`] stops a request between GPU
/// dispatches and is a successful [`StopReason::Cancelled`], never an error,
/// and it must leave the reusable prompt-cache tiers empty.
///
/// The trip is taken **during prefill**. With speculation and n-gram drafting
/// off, decode calls `BonsaiModel::decode`, which polls nothing, so prefill's
/// 128-token chunk loop is the only reachable `is_cancelled()` site and a trip
/// that arrived later would run the request to `TokenLimit` and fail loudly
/// rather than pass silently. Prefill is GPU-paced — every chunk waits on its
/// command buffer — so a 250 ms producer delay lands inside a request whose
/// prefill spans four chunks, with more than an order of magnitude of margin on
/// both sides of the last poll.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
fn deep_cancellation_stops_in_prefill_and_leaves_the_prompt_cache_empty() {
    let path = String::from(crate::bonsai::DEFAULT_BONSAI_GGUF);
    let open = || {
        BonsaiEngine::open_with_options(
            Path::new(&path),
            Some(1024),
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
        .repeat(48);
    let full = engine.encode_prompt(&text, true, false).expect("prompt");
    assert!(
        full.len() > 4 * DEFAULT_PREFILL_CHUNK,
        "the cancelled prompt must span several prefill chunks: {} tokens",
        full.len()
    );
    // Diverge from the cancelled prompt at exactly one chunk boundary. The
    // cancelled request therefore restores a shared-prefix checkpoint and then
    // commits no further chunk, leaving that checkpoint behind unless the
    // `Cancelled` tail clears it.
    let mut warmup = full[..DEFAULT_PREFILL_CHUNK].to_vec();
    let sentinel = engine
        .encode_prompt(" alpha", true, false)
        .expect("sentinel")[0];
    assert_ne!(sentinel, full[DEFAULT_PREFILL_CHUNK]);
    warmup.push(sentinel);
    let extended = full[..4 * DEFAULT_PREFILL_CHUNK].to_vec();
    let mut follow_up = extended.clone();
    follow_up.push(sentinel);

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

    // The follow-up shares the cancelled prompt's whole prefix, so any tier
    // the cancelled request failed to clear would be restored here. No session
    // cache is configured, so host and disk tiers are empty by construction and
    // `PromptCacheSource::None` is the only reachable outcome after the clear.
    let after = engine
        .generate(&follow_up, &params, |_| true)
        .expect("request after cancellation");
    assert_eq!(
        after.stats.reused_prompt_tokens, 0,
        "a cancelled request must leave no reusable prompt prefix behind"
    );
    assert_eq!(
        after.cache_source,
        PromptCacheSource::None,
        "a cancelled request must leave no prompt-cache tier behind"
    );
    assert_eq!(
        after.token_ids,
        open()
            .generate(&follow_up, &params, |_| true)
            .expect("cold reference")
            .token_ids,
        "a cancelled request must not corrupt the next request's output"
    );
}
