use super::writer::{Completion, Submitted};
use super::*;
use std::time::{Duration, Instant, SystemTime};

fn snapshot_fixture() -> PromptSnapshot {
    PromptSnapshot {
        position: 3,
        layout: "f16".into(),
        recurrent: PageBytes::concat([&[1, 2, 3][..]]).expect("bytes"),
        mtp_prev_hidden: PageBytes::concat([&[4, 5][..]]).expect("bytes"),
        target_kv: PageBytes::concat([&[6, 7, 8, 9][..]]).expect("bytes"),
        mtp_kv: PageBytes::concat([&[10, 11][..]]).expect("bytes"),
    }
}

#[test]
fn disk_snapshot_round_trip_and_corruption_rejection() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snapshot = snapshot_fixture();
    let path = store(
        dir.path(),
        "model",
        &[12, 34, 56],
        Some("a"),
        &snapshot,
        true,
    )
    .expect("store snapshot");
    let entries = discover(dir.path(), "model");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].tokens, [12, 34, 56]);
    assert!(entries[0].reusable_boundary);
    let loaded = load(&path, "model").expect("load snapshot");
    assert_eq!(loaded.position, snapshot.position);
    assert_eq!(*loaded.target_kv, *snapshot.target_kv);
    corrupt_last_payload_byte(&path);
    assert!(load(&path, "model").is_err());
}

#[test]
fn a_background_load_verifies_exactly_as_a_load_does() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snapshot = snapshot_fixture();
    let path =
        store(dir.path(), "model", &[12, 34, 56], None, &snapshot, false).expect("store snapshot");
    let started = Load::spawn(path.clone(), "model".into()).expect("spawn load");
    assert_eq!(started.path(), path);
    let loaded = started.wait().expect("loaded snapshot");
    assert_eq!(loaded.position, snapshot.position);
    assert_eq!(*loaded.recurrent, *snapshot.recurrent);
    assert_eq!(*loaded.mtp_kv, *snapshot.mtp_kv);
    assert!(
        Load::spawn(path.clone(), "other".into())
            .expect("spawn load")
            .wait()
            .is_err()
    );
    corrupt_last_payload_byte(&path);
    assert!(
        Load::spawn(path, "model".into())
            .expect("spawn load")
            .wait()
            .is_err()
    );
}

/// Flip a bit of the last byte of the last section, which the padding after
/// it would otherwise hide from a test that flips the file's last byte.
fn corrupt_last_payload_byte(path: &Path) {
    let (start, length) = *section_spans(path).last().expect("sections");
    let mut bytes = fs::read(path).expect("read snapshot");
    bytes[start + length - 1] ^= 1;
    fs::write(path, bytes).expect("corrupt snapshot");
}

fn names_with_extension(dir: &Path, extension: &str) -> Vec<String> {
    let mut names = fs::read_dir(dir)
        .expect("read cache dir")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(extension))
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[test]
fn failed_store_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    let snapshot = snapshot_fixture();
    let path = store(
        dir.path(),
        "model",
        &[12, 34, 56],
        Some("a"),
        &snapshot,
        true,
    )
    .expect("first store succeeds");
    // Occupy the commit name with a directory so `fs::rename` must fail after the
    // payload is already written and synced: the exact "partway" window.
    fs::remove_file(&path).expect("remove committed snapshot");
    fs::create_dir(&path).expect("occupy commit name with a directory");

    assert!(
        store(
            dir.path(),
            "model",
            &[12, 34, 56],
            Some("a"),
            &snapshot,
            true
        )
        .is_err()
    );

    assert!(
        names_with_extension(&model_dir, ".tmp").is_empty(),
        "a failed store must not leave a partial temporary"
    );
    assert!(discover(dir.path(), "model").is_empty());
    fs::remove_dir(&path).expect("clear occupied commit name");
}

#[test]
fn discover_and_trim_ignore_zero_byte_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    fs::create_dir_all(&model_dir).expect("create model dir");
    let debris = model_dir.join("0-deadbeefdeadbeef.bpc");
    fs::write(&debris, b"").expect("write zero-byte entry");
    store(
        dir.path(),
        "model",
        &[12, 34, 56],
        Some("a"),
        &snapshot_fixture(),
        true,
    )
    .expect("store snapshot");

    assert_eq!(discover(dir.path(), "model").len(), 1);

    trim(dir.path(), "model", 0);

    assert!(debris.exists(), "zero-byte entry is never a trim victim");
    assert_eq!(discover(dir.path(), "model").len(), 1);
}

/// Backdate a file so the age guard can be exercised without sleeping.
/// `File::set_modified` takes a `SystemTime`, so this needs no new dependency.
fn backdate(path: &Path, age: Duration) {
    let stamp = SystemTime::now()
        .checked_sub(age)
        .expect("clock reads before the epoch");
    File::options()
        .write(true)
        .open(path)
        .expect("open to set mtime")
        .set_modified(stamp)
        .expect("set mtime");
}

/// Liveness is the writer's lock, not age: `IN_FLIGHT_AGE` is a `store`
/// partway through a long payload; `ABANDONED_FOR` is what SIGKILL leaves
/// behind by the time the app is reopened. Neither age decides anything.
const IN_FLIGHT_AGE: Duration = Duration::from_secs(75);
const ABANDONED_FOR: Duration = Duration::from_mins(30);

fn snapshot_paths(dir: &Path, model_key: &str) -> Vec<PathBuf> {
    let mut paths = discover(dir, model_key)
        .into_iter()
        .map(|entry| entry.path)
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn a_temporary_being_written_is_not_reclaimed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    fs::create_dir_all(&model_dir).expect("create model dir");
    // Stand in for a `store` mid-flight: a real temporary, still being
    // written, so its writer holds the lock — however old it looks.
    let in_flight = model_dir.join("3-0123456789abcdef.tmp");
    let mut writing = create_locked(&in_flight).expect("create in-flight temporary");
    writing
        .write_all(&[7_u8; 4096])
        .expect("write into in-flight temporary");
    backdate(&in_flight, ABANDONED_FOR);

    trim(dir.path(), "model", 0);

    assert!(
        in_flight.exists(),
        "a temporary whose writer holds its lock belongs to a running store and must survive"
    );
    drop(writing);
    trim(dir.path(), "model", 0);
    assert!(
        !in_flight.exists(),
        "once its writer is gone the same temporary is an orphan"
    );
}

#[test]
fn a_temporary_orphaned_by_a_crash_is_reclaimed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    fs::create_dir_all(&model_dir).expect("create model dir");
    // SIGKILL or power loss: the payload was written but `fs::rename` never ran
    // and no cleanup code ever will.
    let orphan = model_dir.join("3-fedcba9876543210.tmp");
    fs::write(&orphan, vec![7_u8; 4096]).expect("write orphaned temporary");
    backdate(&orphan, ABANDONED_FOR);

    trim(dir.path(), "model", 0);

    assert!(
        !orphan.exists(),
        "a temporary older than the threshold can never be a live write and must be reclaimed"
    );
}

#[test]
fn a_young_orphaned_temporary_is_reclaimed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    fs::create_dir_all(&model_dir).expect("create model dir");
    // A server stopped a second after a long answer: the write was killed
    // with the process, so its temporary is seconds old, not minutes.
    let orphan = model_dir.join("249350-fedcba9876543210.tmp");
    fs::write(&orphan, vec![7_u8; 4096]).expect("write orphaned temporary");

    assert_eq!(
        reclaim(dir.path(), "model", u64::MAX),
        Vec::<PathBuf>::new()
    );

    assert!(
        !orphan.exists(),
        "an unlocked temporary is an orphan whatever its age, and the next start reclaims it"
    );
}

#[test]
fn the_temporary_sweep_never_reclaims_a_committed_snapshot() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    fs::create_dir_all(&model_dir).expect("create model dir");
    let committed = store(
        dir.path(),
        "model",
        &[12, 34, 56],
        Some("a"),
        &snapshot_fixture(),
        true,
    )
    .expect("store snapshot");
    // Age the snapshot well past the threshold, then sweep alongside a young and
    // a stale temporary: only the stale temporary may go.
    backdate(&committed, Duration::from_hours(720));
    let young = model_dir.join("3-000000000000000a.tmp");
    let stale = model_dir.join("3-000000000000000b.tmp");
    let mut live = create_locked(&young).expect("create live temporary");
    live.write_all(b"young").expect("write live temporary");
    fs::write(&stale, b"stale").expect("write stale temporary");
    backdate(&stale, ABANDONED_FOR);
    let bytes = fs::metadata(&committed).expect("metadata").len();

    // A budget of `u64::MAX` means no `.bpc` eviction is due, so anything this
    // test finds missing was removed by the temporary sweep.
    trim(dir.path(), "model", u64::MAX);

    assert!(!stale.exists(), "the stale temporary is reclaimed");
    assert!(young.exists(), "the live temporary survives");
    drop(live);
    assert!(
        committed.exists(),
        "a committed snapshot is never a temporary-sweep victim, at any age"
    );
    assert_eq!(fs::metadata(&committed).expect("metadata").len(), bytes);
    assert_eq!(
        snapshot_paths(dir.path(), "model"),
        std::slice::from_ref(&committed)
    );
    load(&committed, "model").expect("snapshot is still intact and loadable");
}

#[test]
fn temporaries_do_not_perturb_the_disk_budget() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    fs::create_dir_all(&model_dir).expect("create model dir");
    let first = store(
        dir.path(),
        "model",
        &[12, 34, 56],
        Some("a"),
        &snapshot_fixture(),
        true,
    )
    .expect("store first snapshot");
    let second = store(
        dir.path(),
        "model",
        &[78, 90, 12, 34],
        Some("a"),
        &snapshot_fixture(),
        false,
    )
    .expect("store second snapshot");
    assert_ne!(first, second);
    let before = snapshot_paths(dir.path(), "model");
    assert_eq!(before.len(), 2);
    // A budget of exactly the two snapshots: `bytes > budget` is false, so
    // nothing is evicted. Temporaries that counted would blow this budget many
    // times over and push `trim` into evicting a real snapshot.
    let budget = before
        .iter()
        .map(|path| fs::metadata(path).expect("metadata").len())
        .sum::<u64>();

    // Temporaries both younger and older than the threshold, in bulk.
    for index in 0..4_u32 {
        let temporary = model_dir.join(format!("3-aaaaaaaaaaaaaaa{index}.tmp"));
        fs::write(&temporary, vec![9_u8; 1 << 20]).expect("write temporary");
        let age = if index % 2 == 0 {
            IN_FLIGHT_AGE
        } else {
            ABANDONED_FOR
        };
        backdate(&temporary, age);
    }

    assert_eq!(
        snapshot_paths(dir.path(), "model"),
        before,
        "discover reports exactly the committed snapshots, whatever temporaries sit beside them"
    );

    trim(dir.path(), "model", budget);

    assert_eq!(
        snapshot_paths(dir.path(), "model"),
        before,
        "the budget arithmetic is unchanged by temporaries"
    );
    assert!(
        first.exists() && second.exists(),
        "no committed snapshot was evicted"
    );
    load(&first, "model").expect("first snapshot intact");
    load(&second, "model").expect("second snapshot intact");
}

fn job(root: &Path, tokens: &[u32], reusable_boundary: bool, budget: u64) -> StoreJob {
    StoreJob {
        root: root.to_owned(),
        model_key: "model".into(),
        tokens: tokens.to_vec(),
        session_id: Some("a".into()),
        snapshot: snapshot_fixture(),
        reusable_boundary,
        budget,
    }
}

fn committed(completions: Vec<Completion>) -> Vec<DiskEntry> {
    completions
        .into_iter()
        .map(|completion| {
            completion
                .stored
                .expect("background store")
                .expect("admitted")
        })
        .collect()
}

#[test]
fn a_background_snapshot_is_reported_only_once_committed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let writer = Writer::spawn().expect("writer");
    // Large enough that the write is still running when the first poll lands.
    let mut large = job(dir.path(), &[1, 2, 3], true, u64::MAX);
    large.snapshot.target_kv = PageBytes::zeroed(64 << 20).expect("bytes");
    assert_eq!(writer.submit(large), Submitted::Queued);
    let mut reported = Vec::new();
    while reported.is_empty() {
        // Whatever the timing, a reported entry is a renamed, loadable file.
        for entry in committed(writer.take_completed()) {
            assert_eq!(entry.path.extension().and_then(|e| e.to_str()), Some("bpc"));
            load(&entry.path, "model").expect("a reported snapshot is complete");
            reported.push(entry);
        }
        std::thread::yield_now();
    }
    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].tokens, [1, 2, 3]);
    assert!(reported[0].reusable_boundary);
    assert_eq!(
        snapshot_paths(dir.path(), "model"),
        [reported[0].path.clone()]
    );
    assert_eq!(
        names_with_extension(&dir.path().join("model"), ".tmp"),
        Vec::<String>::new()
    );
}

#[test]
fn wait_for_returns_once_the_matching_write_is_committed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let writer = Writer::spawn().expect("writer");
    writer.submit(job(dir.path(), &[7, 8, 9], false, u64::MAX));
    writer.wait_for(|tokens| tokens == [7, 8, 9]);
    let entries = committed(writer.take_completed());
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].tokens, [7, 8, 9]);
    load(&entries[0].path, "model").expect("committed");
    // Nothing pending matches, so this returns at once.
    writer.wait_for(|_| true);
}

#[test]
fn dropping_the_writer_finishes_pending_writes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let writer = Writer::spawn().expect("writer");
    let mut submitted = Vec::new();
    for tokens in [[1, 1], [2, 2], [3, 3]] {
        submitted.push(writer.submit(job(dir.path(), &tokens, false, u64::MAX)));
    }
    drop(writer);
    // Each `Replaced` is one job that was never written; every other accepted
    // job is on disk once the writer is gone, whatever the thread's timing.
    let replaced = submitted
        .iter()
        .filter(|&&outcome| outcome == Submitted::Replaced)
        .count();
    assert!(!submitted.contains(&Submitted::Dropped));
    assert_eq!(snapshot_paths(dir.path(), "model").len(), 3 - replaced);
    assert_eq!(
        names_with_extension(&dir.path().join("model"), ".tmp"),
        Vec::<String>::new()
    );
    // The newest job is never the one replaced.
    assert!(
        discover(dir.path(), "model")
            .iter()
            .any(|entry| entry.tokens == [3, 3])
    );
}

#[test]
fn the_queue_holds_one_job_and_never_lets_a_tail_displace_a_boundary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut queued = None;
    let tokens = |slot: &Option<StoreJob>| slot.as_ref().map(|job| job.tokens.clone());
    assert_eq!(
        writer::enqueue(&mut queued, job(dir.path(), &[1], false, 0)),
        Submitted::Queued
    );
    assert_eq!(
        writer::enqueue(&mut queued, job(dir.path(), &[2], true, 0)),
        Submitted::Replaced
    );
    assert_eq!(tokens(&queued), Some(vec![2]));
    assert_eq!(
        writer::enqueue(&mut queued, job(dir.path(), &[3], false, 0)),
        Submitted::Dropped,
        "a request tail never displaces a queued shared-prefix boundary"
    );
    assert_eq!(tokens(&queued), Some(vec![2]));
    assert_eq!(
        writer::enqueue(&mut queued, job(dir.path(), &[4], true, 0)),
        Submitted::Replaced,
        "a newer boundary replaces an older one"
    );
    assert_eq!(tokens(&queued), Some(vec![4]));
}

#[test]
fn the_writer_trims_from_its_index_and_reports_evictions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let writer = Writer::spawn().expect("writer");
    let mut stored = Vec::new();
    let mut evicted = Vec::new();
    // A zero budget keeps exactly one snapshot, preferring the boundary: two
    // tails evict each other in turn, and the boundary evicts the last.
    for (tokens, boundary) in [([1, 1], false), ([2, 2], false), ([3, 3], true)] {
        writer.submit(job(dir.path(), &tokens, boundary, 0));
        writer.flush();
        for completion in writer.take_completed() {
            stored.push(completion.stored.expect("store").expect("admitted").path);
            evicted.extend(completion.evicted);
        }
    }
    assert_eq!(stored.len(), 3);
    assert_eq!(snapshot_paths(dir.path(), "model"), [stored[2].clone()]);
    assert_eq!(evicted, [stored[0].clone(), stored[1].clone()]);
    // A tail behind it would be deleted again at once, so it is not written.
    writer.submit(job(dir.path(), &[4, 4], false, 0));
    writer.flush();
    let completions = writer.take_completed();
    assert!(matches!(
        completions[..],
        [Completion {
            stored: Ok(None),
            ..
        }]
    ));
    assert_eq!(completions[0].evicted, Vec::<PathBuf>::new());
    assert_eq!(snapshot_paths(dir.path(), "model"), [stored[2].clone()]);
}

#[test]
fn a_failed_background_store_is_reported_not_raised() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A file where the cache directory should be: `create_dir_all` fails.
    let root = dir.path().join("not-a-directory");
    fs::write(&root, b"").expect("occupy root");
    let writer = Writer::spawn().expect("writer");
    writer.submit(job(&root, &[5], true, u64::MAX));
    writer.flush();
    let completions = writer.take_completed();
    assert_eq!(completions.len(), 1);
    assert!(completions[0].stored.is_err());
    // The thread survives the failure.
    writer.submit(job(dir.path(), &[6], true, u64::MAX));
    writer.flush();
    assert_eq!(committed(writer.take_completed()).len(), 1);
}

fn tokens(range: std::ops::Range<u32>) -> Vec<u32> {
    range.collect()
}

fn extended(prefix: &[u32], extra: std::ops::Range<u32>) -> Vec<u32> {
    prefix.iter().copied().chain(extra).collect()
}

/// A budget that holds one fixture snapshot of `tokens` but not two, as the
/// automatic 5.9 GB budget holds one 5.06 GB 128K-token snapshot.
fn one_snapshot_budget(tokens: &[u32]) -> u64 {
    stored_bytes("model", tokens, Some("a"), &snapshot_fixture(), true) * 11 / 10
}

fn stored_lengths(dir: &Path) -> Vec<usize> {
    let mut lengths = discover(dir, "model")
        .into_iter()
        .map(|entry| entry.tokens.len())
        .collect::<Vec<_>>();
    lengths.sort_unstable();
    lengths
}

/// The long-context evaluation's third run, in miniature: a staged prefix
/// (a request tail), then the system-turn boundary every question shares,
/// then the divergence boundaries later questions observe a few tokens into
/// the user turn (the turn opener, then the first words of one kind of
/// question). Each fills the budget alone. Before, each newer boundary
/// evicted the older one, so the system-turn boundary was gone and every
/// question of another kind re-prefilled the whole prompt.
#[test]
fn a_near_duplicate_boundary_never_evicts_the_shared_prefix_it_extends() {
    let dir = tempfile::tempdir().expect("tempdir");
    let writer = Writer::spawn().expect("writer");
    let stage = tokens(0..1100);
    let system = tokens(0..2000);
    let opener = extended(&system, 5000..5004);
    let recall = extended(&opener, 6000..6009);
    let budget = one_snapshot_budget(&recall);
    let mut outcomes = Vec::new();
    for (tokens, boundary) in [
        (&stage, false),
        (&system, true),
        (&opener, true),
        (&recall, true),
    ] {
        writer.submit(job(dir.path(), tokens, boundary, budget));
        writer.flush();
        outcomes.extend(writer.take_completed().into_iter().map(|completion| {
            (
                completion
                    .stored
                    .expect("store")
                    .map(|entry| entry.tokens.len()),
                completion.evicted.len(),
            )
        }));
    }
    assert_eq!(
        outcomes,
        [
            (Some(1100), 0),
            // The system-turn boundary replaces the staged prefix it extends.
            (Some(2000), 1),
            // The near-duplicates are not written and evict nothing.
            (None, 0),
            (None, 0),
        ]
    );
    assert_eq!(stored_lengths(dir.path()), [2000]);
}

/// Controls for the admission rule: a boundary that extends its ancestor by
/// more than the ancestor holds (a long shared document after a short system
/// prompt) still replaces it, and request tails, staged prefixes included,
/// still replace older tails.
#[test]
fn a_far_extension_and_newer_tails_still_replace_what_they_extend() {
    let dir = tempfile::tempdir().expect("tempdir");
    let writer = Writer::spawn().expect("writer");
    let system = tokens(0..200);
    let document = extended(&system, 1000..1300);
    let budget = one_snapshot_budget(&document);
    for (tokens, boundary) in [(&system, true), (&document, true)] {
        writer.submit(job(dir.path(), tokens, boundary, budget));
        writer.flush();
    }
    assert_eq!(stored_lengths(dir.path()), [500]);

    let staged = tempfile::tempdir().expect("tempdir");
    let first = tokens(0..1000);
    let second = tokens(0..1650);
    let budget = one_snapshot_budget(&second);
    for stage in [&first, &second] {
        writer.submit(job(staged.path(), stage, false, budget));
        writer.flush();
    }
    assert_eq!(stored_lengths(staged.path()), [1650]);
    assert_eq!(committed(writer.take_completed()).len(), 4);
}

#[test]
fn eviction_keeps_the_shortest_shared_ancestor_of_near_duplicates() {
    let entry = |tokens: Vec<u32>, reusable_boundary| Indexed {
        path: PathBuf::new(),
        tokens,
        bytes: 10,
        reusable_boundary,
    };
    let system = tokens(0..100);
    // Oldest first, whatever order the near-duplicates arrived in.
    let entries = [
        entry(extended(&system, 500..513), true),
        entry(system.clone(), true),
        entry(extended(&system, 600..604), true),
        entry(tokens(0..50), false),
    ];
    assert_eq!(eviction_order(&entries, 10), [3, 2, 0]);
    // Unrelated boundaries still go oldest first.
    let entries = [entry(tokens(0..100), true), entry(tokens(1..101), true)];
    assert_eq!(eviction_order(&entries, 10), [0]);
}

/// A restarted server finds what its predecessor left: two staged prefixes
/// over the budget together (100K and 165K tokens, 10.5 GB against 5.9 GB)
/// and a temporary from a write killed with the process. Before, both stayed
/// until a later store trimmed them, and the temporary until it was ten minutes old.
#[test]
fn reclaim_at_startup_trims_to_the_budget_and_removes_orphans() {
    let dir = tempfile::tempdir().expect("tempdir");
    let older = tokens(0..1000);
    let newer = tokens(0..1650);
    store(
        dir.path(),
        "model",
        &older,
        None,
        &snapshot_fixture(),
        false,
    )
    .expect("store");
    let kept = store(
        dir.path(),
        "model",
        &newer,
        None,
        &snapshot_fixture(),
        false,
    )
    .expect("store");
    let orphan = dir.path().join("model").join("2000-0123456789abcdef.tmp");
    fs::write(&orphan, b"partial").expect("write orphan");
    let budget = one_snapshot_budget(&newer);

    let evicted = reclaim(dir.path(), "model", budget);

    assert_eq!(evicted.len(), 1);
    assert_eq!(snapshot_paths(dir.path(), "model"), [kept]);
    assert!(!orphan.exists());
}

#[test]
fn close_finishes_accepted_writes_and_then_refuses_new_ones() {
    let dir = tempfile::tempdir().expect("tempdir");
    let closer = Closer::new();
    let writer = Writer::spawn_with(&closer, None).expect("writer");
    let mut large = job(dir.path(), &[1, 2, 3], true, u64::MAX);
    large.snapshot.target_kv = PageBytes::zeroed(64 << 20).expect("bytes");
    assert_eq!(writer.submit(large), Submitted::Queued);

    assert_eq!(closer.close(Duration::from_mins(1)), Closed::Flushed);

    assert_eq!(stored_lengths(dir.path()), [3]);
    assert_eq!(
        names_with_extension(&dir.path().join("model"), ".tmp"),
        Vec::<String>::new()
    );
    assert_eq!(
        writer.submit(job(dir.path(), &[4, 5, 6], true, u64::MAX)),
        Submitted::Dropped
    );
    drop(writer);
    assert_eq!(stored_lengths(dir.path()), [3]);
}

/// A server stopped with SIGINT right after a long answer used to exit with
/// the ~10 GB snapshot write partway, leaving a 4.66 GB temporary. Past its
/// deadline, close aborts the write and removes the temporary instead.
#[test]
fn close_past_its_deadline_aborts_and_leaves_no_temporary() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(dir.path().join("model")).expect("create model dir");
    let closer = Closer::new();
    let writer = Writer::spawn_with(&closer, None).expect("writer");
    for tokens in [[1, 1], [2, 2]] {
        let mut large = job(dir.path(), &tokens, true, u64::MAX);
        large.snapshot.target_kv = PageBytes::zeroed(512 << 20).expect("bytes");
        writer.submit(large);
    }

    assert_eq!(closer.close(Duration::ZERO), Closed::Aborted);

    assert_eq!(
        names_with_extension(&dir.path().join("model"), ".tmp"),
        Vec::<String>::new()
    );
    // An abort that lands during the final `fsync` still commits; whatever
    // is committed is complete.
    for path in snapshot_paths(dir.path(), "model") {
        load(&path, "model").expect("a committed snapshot is complete");
    }
    drop(writer);
    assert_eq!(
        names_with_extension(&dir.path().join("model"), ".tmp"),
        Vec::<String>::new()
    );
}

/// A snapshot whose sections span several checksum blocks and end mid-block
/// and mid-page, filled with a position-dependent pattern.
fn multi_block_fixture() -> PromptSnapshot {
    let filled = |length: usize, seed: u8| {
        let bytes = (0..length)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed))
            .collect::<Vec<_>>();
        PageBytes::concat([bytes.as_slice()]).expect("bytes")
    };
    PromptSnapshot {
        position: 5,
        layout: "q8".into(),
        recurrent: filled(blocks::BLOCK * 2 + 4_097, 1),
        mtp_prev_hidden: filled(20_480, 2),
        target_kv: filled(blocks::BLOCK + 3, 3),
        mtp_kv: filled(0, 4),
    }
}

#[test]
fn crc32c_matches_the_reference() {
    // The check value of CRC-32C (iSCSI), RFC 3720 B.4.
    assert_eq!(blocks::crc32c(b"123456789"), 0xE306_9283);
    assert_eq!(blocks::crc32c(&[0; 32]), 0x8A91_36AA);
    assert_eq!(blocks::crc32c(&[]), 0);
    let bytes = (0..4_099_u32)
        .map(|index| (index.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect::<Vec<_>>();
    for length in [1, 7, 8, 9, 63, 1_024, 4_099] {
        assert_eq!(
            blocks::crc32c(&bytes[..length]),
            blocks::crc32c_bitwise(&bytes[..length]),
            "{length} bytes"
        );
    }
}

#[test]
fn a_block_snapshot_round_trips_aligned_and_sized_as_predicted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snapshot = multi_block_fixture();
    let tokens = [7, 8, 9, 10, 11];
    let path = store(dir.path(), "model", &tokens, Some("s"), &snapshot, true).expect("store");
    let bytes = fs::read(&path).expect("read snapshot");
    assert_eq!(
        bytes.len() as u64,
        stored_bytes("model", &tokens, Some("s"), &snapshot, true)
    );
    assert_eq!(bytes.len() % blocks::ALIGN, 0);
    let header_bytes = u64::from_le_bytes(bytes[8..16].try_into().expect("length")) as usize;
    assert_eq!((16 + header_bytes) % blocks::ALIGN, 0, "payload is aligned");
    let entries = discover(dir.path(), "model");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].tokens, tokens);
    let loaded = load(&path, "model").expect("load");
    assert_eq!(loaded.position, snapshot.position);
    assert_eq!(*loaded.recurrent, *snapshot.recurrent);
    assert_eq!(*loaded.mtp_prev_hidden, *snapshot.mtp_prev_hidden);
    assert_eq!(*loaded.target_kv, *snapshot.target_kv);
    assert!(loaded.mtp_kv.is_empty());
}

/// Where each section starts in a stored block snapshot, and its length.
fn section_spans(path: &Path) -> Vec<(usize, usize)> {
    let mut file = File::open(path).expect("open");
    let header = read_header(&mut file).expect("header");
    let mut start = usize::try_from(file.stream_position().expect("position")).expect("offset");
    [
        header.token_count * 4,
        header.recurrent,
        header.mtp_hidden,
        header.target_kv,
        header.mtp_kv,
    ]
    .into_iter()
    .map(|length| {
        let span = (start, length);
        start += blocks::padded(length);
        span
    })
    .collect()
}

#[test]
fn a_damaged_truncated_or_extended_block_snapshot_is_never_loaded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snapshot = multi_block_fixture();
    let path = store(
        dir.path(),
        "model",
        &[1, 2, 3, 4, 5],
        None,
        &snapshot,
        false,
    )
    .expect("store");
    let pristine = fs::read(&path).expect("read snapshot");
    let rejects = |bytes: &[u8], what: &str| {
        fs::write(&path, bytes).expect("rewrite snapshot");
        assert!(load(&path, "model").is_err(), "{what} was loaded");
        assert!(
            Load::spawn(path.clone(), "model".into())
                .expect("spawn load")
                .wait()
                .is_err(),
            "{what} was loaded in the background"
        );
    };
    // One flipped bit in the first, a middle and the last byte of every
    // section that has bytes, the token ids included.
    for (start, length) in section_spans(&path) {
        for offset in [0, length / 2, length.saturating_sub(1)] {
            if length == 0 {
                continue;
            }
            let mut bytes = pristine.clone();
            bytes[start + offset] ^= 0x10;
            rejects(&bytes, &format!("a bit flipped at {}", start + offset));
        }
    }
    // A damaged checksum digit in the header.
    let text = String::from_utf8_lossy(&pristine[16..]).into_owned();
    let digit = 16 + text.find("\"payload_crc32c\":\"").expect("checksums") + 18;
    let mut bytes = pristine.clone();
    bytes[digit] = if bytes[digit] == b'0' { b'1' } else { b'0' };
    rejects(&bytes, "a damaged checksum");
    // A header claiming a section of `usize::MAX` bytes, its padding
    // shortened to keep the payload where it was.
    let field = format!("\"recurrent\":{}", snapshot.recurrent.len());
    let huge = format!("\"recurrent\":{}", usize::MAX);
    let at = 16 + text.find(&field).expect("recurrent length");
    let header_end = section_spans(&path)[0].0;
    let mut bytes = pristine[..at].to_vec();
    bytes.extend_from_slice(huge.as_bytes());
    bytes.extend_from_slice(&pristine[at + field.len()..header_end - (huge.len() - field.len())]);
    bytes.extend_from_slice(&pristine[header_end..]);
    assert_eq!(bytes.len(), pristine.len());
    rejects(&bytes, "an impossible section length");
    // Cut short by a byte, by a page, and to the header alone; and extended.
    rejects(&pristine[..pristine.len() - 1], "a file a byte short");
    rejects(
        &pristine[..pristine.len() - blocks::ALIGN],
        "a file a page short",
    );
    rejects(&pristine[..section_spans(&path)[0].0], "the header alone");
    let mut longer = pristine.clone();
    longer.extend_from_slice(&[0; blocks::ALIGN]);
    rejects(&longer, "a file a page long");
    // Intact again, it loads.
    fs::write(&path, &pristine).expect("restore snapshot");
    assert!(load(&path, "model").is_ok());
}

/// A snapshot as stores before the block layout wrote it: a contiguous
/// payload right after the header, checked by one SHA-256.
fn store_sha256(dir: &Path, tokens: &[u32], snapshot: &PromptSnapshot) -> PathBuf {
    let stored_tokens = token_bytes(tokens);
    let mut payload = Vec::new();
    for section in sections(&stored_tokens, snapshot) {
        payload.extend_from_slice(section);
    }
    let mut header = header("model", tokens, None, snapshot, true, String::new());
    header.payload_crc32c = None;
    header.payload_sha256 = Some(hex(&Sha256::digest(&payload)));
    let encoded = serde_json::to_vec(&header).expect("header");
    let dir = dir.join("model");
    fs::create_dir_all(&dir).expect("model dir");
    let path = dir.join(format!("{}-legacy.bpc", snapshot.position));
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&encoded);
    bytes.extend_from_slice(&payload);
    fs::write(&path, bytes).expect("write legacy snapshot");
    path
}

#[test]
fn snapshots_written_with_sha256_still_load_and_are_still_checked() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snapshot = multi_block_fixture();
    let path = store_sha256(dir.path(), &[4, 5, 6, 7, 8], &snapshot);
    assert_eq!(discover(dir.path(), "model")[0].tokens, [4, 5, 6, 7, 8]);
    let loaded = load(&path, "model").expect("load legacy snapshot");
    assert_eq!(*loaded.recurrent, *snapshot.recurrent);
    assert_eq!(*loaded.target_kv, *snapshot.target_kv);
    let pristine = fs::read(&path).expect("read");
    let mut bytes = pristine.clone();
    let middle = bytes.len() - blocks::BLOCK;
    bytes[middle] ^= 1;
    fs::write(&path, bytes).expect("corrupt");
    assert!(load(&path, "model").is_err());
    fs::write(&path, &pristine[..pristine.len() - 1]).expect("truncate");
    assert!(load(&path, "model").is_err());
}

#[test]
fn the_write_budget_refills_daily_and_admits_one_overdraft() {
    let start = Instant::now();
    let mut budget = writer::WriteBudget::new(4_000, start);
    // A quarter of a day's at the start.
    assert!(budget.admits(start));
    budget.charge(999);
    assert!(budget.admits(start));
    // One write larger than what is left still goes through, overdrawing it.
    budget.charge(5_000);
    assert!(!budget.admits(start));
    // A quarter of a day later 1,000 bytes have come back: still overdrawn.
    assert!(!budget.admits(start + Duration::from_hours(6)));
    // A day later it is back in credit, and never holds more than a day's.
    assert!(budget.admits(start + Duration::from_hours(30)));
    assert!(budget.admits(start + Duration::from_hours(1_000)));
    budget.charge(4_000);
    assert!(!budget.admits(start + Duration::from_hours(1_000)));
}

#[test]
fn a_writer_over_its_daily_budget_skips_writes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let closer = Closer::new();
    // Starts with a quarter of a day's: one byte, so the first snapshot is
    // written, overdrawing it, and the next is not.
    let writer = Writer::spawn_with(&closer, Some(4)).expect("writer");
    assert!(writer.within_budget());
    writer.submit(job(dir.path(), &[1, 2, 3], true, u64::MAX));
    writer.flush();
    assert!(!writer.within_budget());
    writer.submit(job(dir.path(), &[4, 5, 6], true, u64::MAX));
    writer.flush();
    let stored = writer
        .take_completed()
        .into_iter()
        .map(|completion| completion.stored.expect("background store"))
        .map(|entry| entry.map(|entry| entry.tokens))
        .collect::<Vec<_>>();
    assert_eq!(stored, [Some(vec![1, 2, 3]), None]);
    assert_eq!(discover(dir.path(), "model").len(), 1);
}

#[test]
fn the_daily_write_budget_is_a_quarter_of_the_volume() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (daily, reason) = daily_write_budget(&dir.path().join("not/yet/created"));
    let daily = daily.expect("a readable volume");
    assert!(daily > 0, "{reason}");
    assert!(reason.contains("a quarter"), "{reason}");
}
