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
