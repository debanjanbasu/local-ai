use super::*;

#[test]
fn exact_longest_suffix_and_frequency() {
    let store = SuffixStore::default();
    let mut session = store.session(&[1, 2, 3, 9, 1, 2, 3], 3);
    assert_eq!(session.find(8).map(|draft| draft.tokens), Some(vec![9]));
    session.append(9);
    assert!(session.find(8).is_none());
}

#[test]
fn a_long_self_referential_match_never_slices_backwards() {
    // An anchor key that repeats inside the *current* sequence can match
    // backwards far enough that `current.len() - matched` drops below the
    // location end. The continuation must then be dropped, not sliced
    // backwards: taking `source[end..continuation_end]` with the end first is
    // the panic that killed the engine worker.
    let store = SuffixStore::default();
    let session = store.session(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3);
    assert_eq!(
        session.find(8).map(|draft| (draft.tokens, draft.match_len)),
        Some((vec![1, 1], 5))
    );
}

#[test]
fn store_is_bounded_and_reused() {
    let mut store = SuffixStore::default();
    store.remember(vec![1, 2, 3, 4]);
    let session = store.session(&[8, 1, 2, 3], 3);
    assert_eq!(session.find(8).map(|draft| draft.tokens), Some(vec![4]));
    for token in 0..MAX_STORED_REQUESTS + 2 {
        store.remember(vec![token as u32]);
    }
    assert!(store.requests.len() <= MAX_STORED_REQUESTS);
}

#[test]
fn drafts_fill_their_verify_tile() {
    // Small-batch cost grows with rows through DEFAULT_SMALL_BATCH_MAX (128),
    // which covers every verify block (at most 64 drafts), so no draft is
    // padded.
    assert_eq!(fill_verify_tile(10, 40), 10);
    assert_eq!(fill_verify_tile(40, 63), 40);
    assert_eq!(fill_verify_tile(59, 63), 59);
    assert_eq!(fill_verify_tile(60, 63), 60);
    assert_eq!(fill_verify_tile(64, 64), 64);
    assert_eq!(fill_verify_tile(61, 20), 61);
}

#[test]
fn policy_backs_off_and_recovers() {
    let mut policy = LookupPolicy::new(16);
    assert_eq!(policy.depth(24, 16), 12);
    // A rejection halves the limit but keeps what was accepted.
    policy.observe(0, 12, 16);
    assert_eq!(policy.depth(24, 16), 8);
    policy.observe(2, 8, 16);
    assert_eq!(policy.depth(24, 16), 4);
    // A full accept doubles it again.
    policy.observe(4, 4, 16);
    assert_eq!(policy.depth(24, 16), 8);
}

#[test]
fn lookup_scales_to_long_histories() {
    let prompt: Vec<u32> = (0..100_000).map(|token| token as u32).collect();
    let session = SuffixStore::default().session(&prompt, 24);
    let started = std::time::Instant::now();
    for _ in 0..10_000 {
        assert!(session.find(16).is_none());
    }
    let micros = started.elapsed().as_micros() as usize / 10_000;
    eprintln!("100K-history suffix lookup: {micros} us/step");
    assert!(micros < 100, "suffix lookup took {micros} us/step");
}
