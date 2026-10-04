use super::*;
use local_metal::shaders::ShaderLibrary;

fn params() -> SamplingParams {
    SamplingParams {
        temperature: 0.8,
        top_k: 20,
        top_p: 0.91,
        min_p: 0.08,
        presence_penalty: 0.25,
        repetition_penalty: 1.13,
        eos_tokens: vec![3],
        seed: 123,
    }
}

#[test]
fn seeded_gpu_sampling_preserves_penalties_filters_ties_and_cpu_fallbacks() {
    let context = MetalContext::new().expect("context");
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let topk = GpuTopK::new(&context, &shaders, 248_320).expect("top-k");
    for (vocab, k, temperature, top_p, min_p) in [
        (33, 20, 0.8, 0.91, 0.08),
        (4_099, 20, 0.8, 0.91, 0.08),
        (4_099, 64, 1.1, 0.0, 0.0),
        (4_099, 64, 1.1, 1.0, 1.0),
        (4_099, 65, 0.8, 0.91, 0.08),
        (4_099, 0, 0.8, 0.91, 0.08),
        (248_320, 20, 0.8, 0.91, 0.08),
        (248_320, 20, 0.0, 1.0, 0.0),
        (4_099, 1, 1.0, 1.0, 0.0),
    ] {
        let config = SamplingParams {
            top_k: k,
            temperature,
            top_p,
            min_p,
            ..params()
        };
        let mut cpu = Sampler::new(vocab, config.clone());
        let mut gpu = Sampler::new(vocab, config);
        for step in 0..8 {
            let history = [1, 1, 3, (step * 19) as u32, vocab as u32 + 1];
            cpu.observe(&history);
            gpu.observe(&history);
            let original = (0..vocab)
                .map(|i| ((i * 173 + step * 13) % 719) as f32 / 113.0 - 3.0)
                .collect::<Vec<_>>();
            let mut logits = original.clone();
            let mut buffer = MetalBuffer::from_slice(context.device(), &original).expect("logits");
            let expected = cpu.sample(&mut logits).expect("CPU sample");
            let actual = gpu
                .sample_buffer(&mut buffer, 0, &context, &topk)
                .expect("GPU sample");
            assert_eq!(actual, expected, "vocab={vocab} k={k} step={step}");
            cpu.observe(&[expected.token_id]);
            gpu.observe(&[actual.token_id]);
        }
        assert_eq!(cpu.seen_tokens, gpu.seen_tokens);
    }
}

#[test]
fn draft_margin_is_the_penalized_lead_over_the_runner_up_on_gpu_and_cpu() {
    let context = MetalContext::new().expect("context");
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let topk = GpuTopK::new(&context, &shaders, 248_320).expect("top-k");
    // 33 selects on the CPU, 248_320 through the GPU top-k.
    for vocab in [33_usize, 248_320] {
        let mut draft = Sampler::new(vocab, params()).greedy_draft();
        let mut logits = vec![-2.0_f32; vocab];
        // Unpenalized the winner leads by 1.5 (8.0 over 6.5); the
        // presence and repetition penalties on the seen token 5
        // ((8.0 - 0.25) / 1.13 = 6.86) shrink its lead to 0.36.
        logits[5] = 8.0;
        logits[9] = 6.5;
        draft.observe(&[5]);
        let mut buffer = MetalBuffer::from_slice(context.device(), &logits).expect("logits");
        let (sample, margin) = draft
            .sample_buffer_with_margin(&mut buffer, 0, &context, &topk)
            .expect("margin");
        assert_eq!(sample.token_id, 5, "vocab={vocab}");
        let expected = (8.0_f32 - 0.25) / 1.13 - 6.5;
        assert!((margin - expected).abs() < 1e-5, "vocab={vocab} {margin}");
        assert_eq!(
            sample,
            draft.sample(&mut logits).expect("plain greedy"),
            "the margin path selects the same token as greedy sampling"
        );

        // Ties resolve to the lower token ID with a zero margin, as the
        // GPU top-k orders them.
        let mut tied = vec![0.0_f32; vocab];
        tied[7] = 3.0;
        tied[2] = 3.0;
        let mut buffer = MetalBuffer::from_slice(context.device(), &tied).expect("logits");
        let fresh = Sampler::new(vocab, params()).greedy_draft();
        let (sample, margin) = fresh
            .sample_buffer_with_margin(&mut buffer, 0, &context, &topk)
            .expect("tie");
        assert_eq!((sample.token_id, margin), (2, 0.0), "vocab={vocab}");
    }
    // Only greedy samplers have a single proposal to measure.
    let stochastic = Sampler::new(33, params());
    let mut buffer = MetalBuffer::from_slice(context.device(), &[0.0_f32; 33]).expect("logits");
    assert!(
        stochastic
            .sample_buffer_with_margin(&mut buffer, 0, &context, &topk)
            .is_err()
    );
    assert_eq!(top_two(&[1.0, 4.0, 4.0, -1.0]), ((1, 4.0), (2, 4.0)));
    assert_eq!(top_two(&[9.0, 4.0]), ((0, 9.0), (1, 4.0)));
}

#[test]
fn history_is_deduplicated_and_invalid_logit_width_is_rejected() {
    let mut sampler = Sampler::new(4, params());
    sampler.observe(&[1, 1, 3, 1, u32::MAX]);
    assert_eq!(sampler.seen_tokens, [1, 3]);
    assert!(sampler.sample(&mut [0.0; 3]).is_err());
    assert!(sampler.sample(&mut []).is_err());
}

#[test]
fn verification_preserves_target_samples_history_and_rng_at_every_rejection() {
    let logits = (0..5)
        .map(|row| {
            (0..32)
                .map(|token| ((token * 17 + row * 23) % 47) as f32 / 13.0 - 2.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for temperature in [0.0, 0.8] {
        for seed in 0..16 {
            let mut initial = Sampler::new(
                32,
                SamplingParams {
                    temperature,
                    seed,
                    eos_tokens: Vec::new(),
                    ..params()
                },
            );
            initial.observe(&[1, 2, 1, 31]);
            let mut sequential = initial.clone();
            let mut expected = Vec::new();
            for row in &logits {
                let result = sequential.sample(&mut row.clone()).expect("target");
                expected.push((result, sequential.clone()));
                sequential.observe(&[result.token_id]);
            }
            for accepted in 0..=4 {
                let mut drafts = expected
                    .iter()
                    .take(4)
                    .map(|(result, _)| result.token_id)
                    .collect::<Vec<_>>();
                if accepted < drafts.len() {
                    drafts[accepted] = (drafts[accepted] + 1) % 32;
                }
                let mut speculative = initial.clone();
                let mut rows = Vec::new();
                let verified = verify_greedy_drafts(&mut speculative, &drafts, |row, target| {
                    rows.push(row);
                    target.sample(&mut logits[row].clone())
                })
                .expect("verification");
                assert_eq!(verified.accepted, accepted);
                assert_eq!(rows, (0..=accepted).collect::<Vec<_>>());
                assert_eq!(
                    verified.samples,
                    expected
                        .iter()
                        .take(accepted + 1)
                        .map(|(result, _)| *result)
                        .collect::<Vec<_>>()
                );
                let reference = &expected[accepted].1;
                assert_eq!(speculative.seen_tokens, reference.seen_tokens);
                assert_eq!(speculative.observed, reference.observed);
                assert_eq!(speculative.rng.0, reference.rng.0);
            }
        }
    }
}

#[test]
fn eos_stops_verification_without_observing_or_accepting_it() {
    for stop in 0..=4 {
        let mut drafts = [0, 1, 2, 0];
        if stop < drafts.len() {
            drafts[stop] = 3;
        }
        let mut target = Sampler::new(
            4,
            SamplingParams {
                temperature: 0.0,
                ..params()
            },
        );
        let verified = verify_greedy_drafts(&mut target, &drafts, |row, distribution| {
            let mut logits = [-10.0; 4];
            logits[if row == stop { 3 } else { drafts[row] as usize }] = 10.0;
            distribution.sample(&mut logits)
        })
        .expect("EOS verification");
        assert_eq!(verified.accepted, stop);
        assert_eq!(verified.samples.len(), stop + 1);
        assert_eq!(
            verified.samples.last(),
            Some(&SamplingResult {
                token_id: 3,
                is_eos: true
            })
        );
        assert!(!target.observed[3]);
    }
}

#[test]
fn empty_draft_uses_exactly_one_target_row_and_sampling_errors_propagate() {
    let mut target = Sampler::new(4, params());
    let verified = verify_greedy_drafts(&mut target, &[], |row, distribution| {
        assert_eq!(row, 0);
        distribution.sample(&mut [3.0, 2.0, 1.0, -10.0])
    })
    .expect("single target row");
    assert_eq!(verified.accepted, 0);
    assert_eq!(verified.samples.len(), 1);
    assert_eq!(target.seen_tokens, [] as [usize; 0]);
    assert!(
        verify_greedy_drafts(&mut target, &[0], |_, distribution| {
            distribution.sample(&mut [])
        })
        .is_err()
    );
}
