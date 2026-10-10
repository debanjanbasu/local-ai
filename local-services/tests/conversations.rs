//! Public-interface tests for the durable store and the Conversations API.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines
)]

use std::sync::Barrier;

use local_services::{Append, Error, ListItems, Metadata, Order, Store};
use serde_json::{Value, json};

fn store() -> (tempfile::TempDir, Store) {
    let scratch = tempfile::tempdir().expect("tempdir");
    let store = Store::open(scratch.path().join("services")).expect("store opens");
    (scratch, store)
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn metadata(pairs: &[(&str, &str)]) -> Metadata {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

fn ids(items: &[local_services::Item]) -> Vec<String> {
    items.iter().map(|item| item.id.clone()).collect()
}

#[cfg(unix)]
fn denied(result: Result<Store, Error>) -> bool {
    matches!(result, Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::PermissionDenied)
}

#[test]
fn conversations_persist_across_reopen_in_normalized_form() {
    let scratch = tempfile::tempdir().expect("tempdir");
    let root = scratch.path().join("services");
    let created = {
        let store = Store::open(&root).expect("store opens");
        store
            .create_conversation(
                &metadata(&[("topic", "demo")]),
                vec![
                    user("hello"),
                    json!({"type": "message", "role": "assistant", "content": "hi"}),
                    json!({"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"}),
                    json!({"type": "function_call_output", "call_id": "call_1", "output": "ok"}),
                    json!({"type": "reasoning", "content": [{"type": "reasoning_text", "text": "hm"}]}),
                ],
            )
            .expect("created")
    };
    assert!(created.id.starts_with("conv_"));
    assert_eq!(created.version, 1);
    assert_eq!(
        serde_json::to_value(&created).expect("serializes"),
        json!({"id": created.id, "object": "conversation", "created_at": created.created_at,
               "metadata": {"topic": "demo"}})
    );

    let store = Store::open(&root).expect("store reopens");
    assert_eq!(store.get_conversation(&created.id).expect("get"), created);
    let history = store.conversation_history(&created.id).expect("history");
    assert_eq!(history.conversation, created);
    let kinds: Vec<&str> = history
        .items
        .iter()
        .map(local_services::Item::kind)
        .collect();
    assert_eq!(
        kinds,
        [
            "message",
            "message",
            "function_call",
            "function_call_output",
            "reasoning"
        ]
    );
    let first = serde_json::to_value(&history.items[0]).expect("serializes");
    assert!(first["id"].as_str().expect("id").starts_with("msg_"));
    assert_eq!(
        first["content"],
        json!([{"type": "input_text", "text": "hello"}])
    );
    assert_eq!(first["status"], "completed");
    let reply = serde_json::to_value(&history.items[1]).expect("serializes");
    assert_eq!(
        reply["content"],
        json!([{"type": "output_text", "text": "hi", "annotations": [], "logprobs": []}])
    );
    assert_eq!(reply["status"], "completed");
    assert_eq!(history.items[4].body["summary"], json!([]));
    assert_eq!(
        store
            .get_item(&created.id, &history.items[2].id)
            .expect("item"),
        history.items[2]
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = |path: &std::path::Path| {
            std::fs::metadata(path).expect("meta").permissions().mode() & 0o777
        };
        assert_eq!(mode(&root), 0o700);
        assert_eq!(mode(&root.join("services.sqlite3")), 0o600);
    }
}

#[test]
fn metadata_is_bounded_and_replaced_on_update() {
    let (_scratch, store) = store();
    let too_many: Metadata = (0..17).map(|n| (format!("k{n}"), String::new())).collect();
    assert!(matches!(
        store.create_conversation(&too_many, Vec::new()),
        Err(Error::InvalidArgument(_))
    ));
    let long_key = metadata(&[(&"k".repeat(65), "v")]);
    assert!(matches!(
        store.create_conversation(&long_key, Vec::new()),
        Err(Error::InvalidArgument(_))
    ));
    let long_value = metadata(&[("k", &"é".repeat(513))]);
    assert!(matches!(
        store.create_conversation(&long_value, Vec::new()),
        Err(Error::InvalidArgument(_))
    ));
    let limit = metadata(&[(&"k".repeat(64), &"é".repeat(512))]);
    let conversation = store
        .create_conversation(&limit, Vec::new())
        .expect("limits are inclusive");
    assert_eq!(conversation.version, 0);
    let updated = store
        .update_conversation(&conversation.id, &metadata(&[("a", "b")]))
        .expect("updated");
    assert_eq!(updated.metadata, metadata(&[("a", "b")]));
    assert_eq!(
        store.get_conversation(&conversation.id).expect("get"),
        updated
    );
}

#[test]
fn items_list_newest_first_and_page_after_a_cursor() {
    let (_scratch, store) = store();
    let conversation = store
        .create_conversation(&Metadata::new(), Vec::new())
        .expect("created");
    let mut added = Vec::new();
    for batch in [0..20, 20..25] {
        let page = store
            .add_items(
                &conversation.id,
                batch.map(|n| user(&n.to_string())).collect(),
            )
            .expect("added");
        assert!(!page.has_more);
        assert_eq!(
            page.first_id.as_ref(),
            page.data.first().map(|item| &item.id)
        );
        added.extend(ids(&page.data));
    }
    assert_eq!(added.len(), 25);
    let newest_first: Vec<String> = added.iter().rev().cloned().collect();

    let first = store
        .list_items(&conversation.id, &ListItems::default())
        .expect("first page");
    assert_eq!(ids(&first.data), newest_first[..20]);
    assert!(first.has_more);
    assert_eq!(first.last_id.as_deref(), Some(newest_first[19].as_str()));
    let rest = store
        .list_items(
            &conversation.id,
            &ListItems {
                after: first.last_id,
                ..ListItems::default()
            },
        )
        .expect("second page");
    assert_eq!(ids(&rest.data), newest_first[20..]);
    assert!(!rest.has_more);

    let ascending = store
        .list_items(
            &conversation.id,
            &ListItems {
                limit: Some(10),
                order: Order::Asc,
                after: Some(added[4].clone()),
            },
        )
        .expect("ascending page");
    assert_eq!(ids(&ascending.data), added[5..15]);
    assert!(ascending.has_more);

    for limit in [0, 101] {
        let query = ListItems {
            limit: Some(limit),
            ..ListItems::default()
        };
        assert!(matches!(
            store.list_items(&conversation.id, &query),
            Err(Error::InvalidArgument(_))
        ));
    }
    let unknown = ListItems {
        after: Some("msg_unknown".to_owned()),
        ..ListItems::default()
    };
    assert!(matches!(
        store.list_items(&conversation.id, &unknown),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn an_invalid_item_rolls_back_the_whole_batch() {
    let (_scratch, store) = store();
    let conversation = store
        .create_conversation(&Metadata::new(), vec![user("kept")])
        .expect("created");
    let kept = store
        .conversation_history(&conversation.id)
        .expect("history")
        .items;
    let unsupported = [
        json!({"role": "user", "content": [{"type": "input_image", "image_url": "https://x/y.png"}]}),
        json!({"type": "web_search_call", "id": "ws_1", "status": "completed"}),
        json!({"role": "robot", "content": "hi"}),
        json!({"role": "user", "content": "hi", "extra": true}),
        json!({"type": "function_call", "call_id": "", "name": "f", "arguments": "{}"}),
        json!({"type": "reasoning", "summary": [], "encrypted_content": "opaque"}),
        json!({"role": "user", "content": "hi", "id": "bad id"}),
        json!("not an object"),
    ];
    for bad in unsupported {
        assert!(
            matches!(
                store.add_items(&conversation.id, vec![user("dropped"), bad.clone()]),
                Err(Error::InvalidArgument(_))
            ),
            "{bad} is refused"
        );
    }
    assert!(matches!(
        store.add_items(&conversation.id, (0..21).map(|_| user("x")).collect()),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        store.add_items(&conversation.id, Vec::new()),
        Err(Error::InvalidArgument(_))
    ));
    // An ID already in the conversation is a conflict, also atomically.
    let duplicate = json!({"role": "user", "content": "again", "id": kept[0].id});
    assert!(matches!(
        store.add_items(&conversation.id, vec![user("dropped"), duplicate]),
        Err(Error::Conflict(_))
    ));
    let same_batch = vec![
        json!({"role": "user", "content": "a", "id": "msg_same"}),
        json!({"role": "user", "content": "b", "id": "msg_same"}),
    ];
    assert!(matches!(
        store.add_items(&conversation.id, same_batch),
        Err(Error::InvalidArgument(_))
    ));

    let after = store
        .conversation_history(&conversation.id)
        .expect("history");
    assert_eq!(after.items, kept);
    assert_eq!(after.conversation.version, conversation.version);
}

#[test]
fn deletion_hides_the_conversation_but_keeps_its_items() {
    let (scratch, store) = store();
    let conversation = store
        .create_conversation(&Metadata::new(), vec![user("one"), user("two")])
        .expect("created");
    let items = store
        .conversation_history(&conversation.id)
        .expect("history")
        .items;

    let parent = store
        .delete_item(&conversation.id, &items[0].id)
        .expect("item deleted");
    assert_eq!(parent.id, conversation.id);
    assert_eq!(parent.version, conversation.version + 1);
    assert!(matches!(
        store.get_item(&conversation.id, &items[0].id),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        store.delete_item(&conversation.id, &items[0].id),
        Err(Error::NotFound(_))
    ));
    let listed = store
        .list_items(&conversation.id, &ListItems::default())
        .expect("listed");
    assert_eq!(ids(&listed.data), [items[1].id.clone()]);

    let deleted = store
        .delete_conversation(&conversation.id)
        .expect("conversation deleted");
    assert!(deleted.deleted);
    assert_eq!(
        serde_json::to_value(&deleted).expect("serializes"),
        json!({"id": conversation.id, "object": "conversation.deleted", "deleted": true})
    );
    let id = conversation.id.as_str();
    let not_found = |result: Result<(), Error>| matches!(result, Err(Error::NotFound(_)));
    assert!(not_found(store.get_conversation(id).map(drop)));
    assert!(not_found(
        store.update_conversation(id, &Metadata::new()).map(drop)
    ));
    assert!(not_found(store.delete_conversation(id).map(drop)));
    assert!(not_found(store.add_items(id, vec![user("x")]).map(drop)));
    assert!(not_found(
        store.list_items(id, &ListItems::default()).map(drop)
    ));
    assert!(not_found(store.get_item(id, &items[1].id).map(drop)));
    assert!(not_found(store.delete_item(id, &items[1].id).map(drop)));
    assert!(not_found(store.conversation_history(id).map(drop)));
    let append = Append {
        request_id: "resp_1".to_owned(),
        expected_version: parent.version,
        items: vec![user("x")],
    };
    assert!(not_found(store.append_items(id, append).map(drop)));
    assert!(not_found(store.get_conversation("conv_missing").map(drop)));

    // The surviving item is still stored, only unreachable.
    let database = rusqlite::Connection::open(scratch.path().join("services/services.sqlite3"))
        .expect("database opens");
    let stored: i64 = database
        .query_row(
            "SELECT count(*) FROM conversation_items WHERE conversation_id = ?1",
            [id],
            |row| row.get(0),
        )
        .expect("counted");
    assert_eq!(stored, 1);
}

#[test]
fn appends_are_idempotent_and_refuse_stale_versions() {
    let (_scratch, store) = store();
    let conversation = store
        .create_conversation(&Metadata::new(), vec![user("q")])
        .expect("created");
    let version = store
        .conversation_history(&conversation.id)
        .expect("history")
        .conversation
        .version;
    let turn = vec![
        user("next"),
        json!({"type": "message", "role": "assistant", "content": "answer"}),
    ];
    let append = |request_id: &str, expected_version: u64, items: Vec<Value>| {
        store.append_items(
            &conversation.id,
            Append {
                request_id: request_id.to_owned(),
                expected_version,
                items,
            },
        )
    };

    let first = append("resp_a", version, turn.clone()).expect("appended");
    assert!(!first.replayed);
    assert_eq!(first.items.len(), 2);
    assert_eq!(first.conversation.version, version + 1);

    let replay = append("resp_a", version, turn.clone()).expect("replayed");
    assert!(replay.replayed);
    assert_eq!(replay.items, first.items);
    assert_eq!(replay.conversation.version, version + 1);
    assert!(matches!(
        append("resp_a", version + 1, vec![user("different")]),
        Err(Error::Conflict(_))
    ));
    assert!(matches!(
        append("resp_b", version, turn.clone()),
        Err(Error::Conflict(_))
    ));
    assert!(matches!(
        append("bad id", version + 1, turn),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(
        store
            .conversation_history(&conversation.id)
            .expect("history")
            .items
            .len(),
        3
    );

    // Two generations against one version, through separate handles as two
    // processes would: exactly one appends, the other learns it is stale.
    let current = version + 1;
    let other = Store::open(store.root()).expect("second handle");
    let barrier = Barrier::new(2);
    let results: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = [(store.clone(), "resp_x"), (other, "resp_y")]
            .into_iter()
            .map(|(handle, request_id)| {
                let barrier = &barrier;
                let id = conversation.id.clone();
                scope.spawn(move || {
                    barrier.wait();
                    handle.append_items(
                        &id,
                        Append {
                            request_id: request_id.to_owned(),
                            expected_version: current,
                            items: vec![user(request_id)],
                        },
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("thread"))
            .collect()
    });
    let won = results.iter().filter(|result| result.is_ok()).count();
    let stale = results
        .iter()
        .filter(|result| matches!(result, Err(Error::Conflict(_))))
        .count();
    assert_eq!((won, stale), (1, 1), "{results:?}");
    let history = store
        .conversation_history(&conversation.id)
        .expect("history");
    assert_eq!(history.items.len(), 4);
    assert_eq!(history.conversation.version, current + 1);
}

#[cfg(unix)]
#[test]
fn symlinks_and_shared_modes_are_refused() {
    use std::os::unix::fs::{PermissionsExt as _, symlink};

    let scratch = tempfile::tempdir().expect("tempdir");
    let base = scratch.path();
    let set_mode = |path: &std::path::Path, mode: u32| {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).expect("chmod");
    };

    let shared = base.join("shared");
    std::fs::create_dir(&shared).expect("mkdir");
    set_mode(&shared, 0o755);
    assert!(denied(Store::open(&shared)), "group-readable directory");

    let linked_root = base.join("linked-root");
    let real = base.join("real");
    Store::open(&real).expect("real store");
    symlink(&real, &linked_root).expect("symlink");
    assert!(denied(Store::open(&linked_root)), "symlinked directory");

    let planted = base.join("planted");
    std::fs::create_dir(&planted).expect("mkdir");
    set_mode(&planted, 0o700);
    let target = base.join("target.sqlite3");
    std::fs::write(&target, b"").expect("target");
    set_mode(&target, 0o600);
    symlink(&target, planted.join("services.sqlite3")).expect("symlink");
    assert!(denied(Store::open(&planted)), "symlinked database");
    let dangling = base.join("dangling");
    std::fs::create_dir(&dangling).expect("mkdir");
    set_mode(&dangling, 0o700);
    symlink(base.join("missing"), dangling.join("services.sqlite3")).expect("symlink");
    assert!(
        denied(Store::open(&dangling)),
        "dangling symlinked database"
    );
    assert!(!base.join("missing").exists(), "the link was not followed");

    let readable = base.join("readable");
    let store = Store::open(&readable).expect("store");
    set_mode(&readable.join("services.sqlite3"), 0o644);
    assert!(denied(Store::open(&readable)), "group-readable database");
    assert!(
        matches!(
            store.get_conversation("conv_x"),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::PermissionDenied
        ),
        "an open handle re-checks on every connection"
    );
}
