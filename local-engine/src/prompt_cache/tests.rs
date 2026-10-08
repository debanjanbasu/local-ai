use super::writer::{Completion, Submitted};
use super::*;

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
    let mut bytes = fs::read(&path).expect("read snapshot");
    let last = bytes.last_mut().expect("non-empty snapshot");
    *last ^= 1;
    fs::write(&path, bytes).expect("corrupt snapshot");
    assert!(load(&path, "model").is_err());
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

/// Ages are absolute, not multiples of `ORPHANED_TEMPORARY_AGE`, so that retuning
/// the constant — upward to something absurd, downward to zero — breaks these
/// tests instead of sliding along with it. `IN_FLIGHT_AGE` is a `store` partway
/// through its ~152 MiB payload; `ABANDONED_FOR` is what SIGKILL leaves behind
/// by the time the app is reopened.
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
    // Stand in for a `store` mid-flight: a real temporary, still being written.
    let in_flight = model_dir.join("3-0123456789abcdef.tmp");
    File::create(&in_flight).expect("create in-flight temporary");
    fs::write(&in_flight, vec![7_u8; 4096]).expect("write into in-flight temporary");
    backdate(&in_flight, IN_FLIGHT_AGE);

    trim(dir.path(), "model", 0);

    assert!(
        in_flight.exists(),
        "a temporary younger than the threshold belongs to a running store and must survive"
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
    fs::write(&young, b"young").expect("write young temporary");
    fs::write(&stale, b"stale").expect("write stale temporary");
    backdate(&stale, ABANDONED_FOR);
    let bytes = fs::metadata(&committed).expect("metadata").len();

    // A budget of `u64::MAX` means no `.bpc` eviction is due, so anything this
    // test finds missing was removed by the temporary sweep.
    trim(dir.path(), "model", u64::MAX);

    assert!(!stale.exists(), "the stale temporary is reclaimed");
    assert!(young.exists(), "the young temporary survives");
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
        .map(|completion| completion.stored.expect("background store"))
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
    // A zero budget keeps exactly one snapshot, preferring the boundary.
    for (tokens, boundary) in [([1, 1], true), ([2, 2], false), ([3, 3], false)] {
        writer.submit(job(dir.path(), &tokens, boundary, 0));
        writer.flush();
        for completion in writer.take_completed() {
            stored.push(completion.stored.expect("store").path);
            evicted.extend(completion.evicted);
        }
    }
    assert_eq!(stored.len(), 3);
    assert_eq!(snapshot_paths(dir.path(), "model"), [stored[0].clone()]);
    evicted.sort();
    let mut expected = vec![stored[1].clone(), stored[2].clone()];
    expected.sort();
    assert_eq!(evicted, expected);
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
