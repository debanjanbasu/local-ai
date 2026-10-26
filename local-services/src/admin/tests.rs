use super::usage::{DAY, HOUR, MINUTE, UsageKind, UsageQuery, UsageRecord, bucket_spans};
use super::*;

fn open() -> (tempfile::TempDir, Admin) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("store")).unwrap();
    (dir, Admin::new(store))
}

fn granted(admin: &Admin, token: &str) -> Credential {
    match admin.authenticate(Some(token)).unwrap() {
        Verdict::Granted(credential) => credential,
        other => panic!("expected a grant, got {other:?}"),
    }
}

/// Every byte of the store directory (database, WAL and sidecars).
fn store_bytes(admin: &Admin) -> Vec<u8> {
    let mut bytes = Vec::new();
    for entry in std::fs::read_dir(admin.store().root()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            bytes.extend(std::fs::read(path).unwrap());
        }
    }
    bytes
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

#[test]
fn authentication_is_off_until_the_first_admin_key_and_then_required() {
    let (_dir, admin) = open();
    assert_eq!(admin.authenticate(None).unwrap(), Verdict::Open);
    assert!(!admin.auth_enabled().unwrap());
    let created = admin
        .create_admin_key("first", None, &Actor::Owner)
        .unwrap();
    assert!(admin.auth_enabled().unwrap());
    assert!(matches!(
        admin.authenticate(None).unwrap(),
        Verdict::Refused(_)
    ));
    assert!(matches!(
        admin.authenticate(Some("sk-nonsense")).unwrap(),
        Verdict::Refused(_)
    ));
    assert_eq!(
        granted(&admin, &created.value),
        Credential::Admin {
            key_id: created.object.id.clone()
        }
    );
    // Deleting every admin key keeps authentication on (fail closed).
    admin
        .delete_admin_key(&created.object.id, &Actor::Owner)
        .unwrap();
    assert!(matches!(
        admin.authenticate(Some(&created.value)).unwrap(),
        Verdict::Refused(_)
    ));
    assert!(matches!(
        admin.authenticate(None).unwrap(),
        Verdict::Refused(_)
    ));
    assert!(matches!(admin.enable_auth(), Err(Error::Conflict(_))));
    admin.disable_auth().unwrap();
    assert_eq!(admin.authenticate(None).unwrap(), Verdict::Open);
}

#[test]
fn a_busy_last_used_write_never_fails_authentication() {
    let (_dir, admin) = open();
    let created = admin
        .create_admin_key("first", None, &Actor::Owner)
        .unwrap();
    // Another writer holds the write lock throughout; the first use of the
    // key wants to record `last_used_at`.
    let writer = rusqlite::Connection::open(admin.store().root().join(crate::DATABASE)).unwrap();
    writer.execute_batch("BEGIN IMMEDIATE").unwrap();
    let started = std::time::Instant::now();
    assert_eq!(
        granted(&admin, &created.value),
        Credential::Admin {
            key_id: created.object.id.clone()
        }
    );
    assert!(
        started.elapsed() < crate::BUSY_TIMEOUT,
        "{:?}",
        started.elapsed()
    );
    writer.execute_batch("ROLLBACK").unwrap();
}

#[test]
fn keys_are_stored_only_as_salted_hmacs_and_shown_redacted() {
    let (_dir, admin) = open();
    let first = admin.create_admin_key("a", None, &Actor::Owner).unwrap();
    let project = admin.create_project("p", None, &Actor::Owner).unwrap();
    let account = admin
        .create_service_account(&project.id, "bot", false, None, &Actor::Owner)
        .unwrap();
    let project_token = account.api_key.unwrap().value;
    for (token, prefix) in [(&first.value, "sk-admin-"), (&project_token, "sk-proj-")] {
        assert!(token.starts_with(prefix));
        assert_eq!(token.len(), prefix.len() + 16 + 1 + 64);
        let secret = token.rsplit('_').next().unwrap();
        let bytes = store_bytes(&admin);
        assert!(!contains(&bytes, token.as_bytes()), "plaintext key at rest");
        assert!(
            !contains(&bytes, secret.as_bytes()),
            "plaintext secret at rest"
        );
    }
    let tail: String = first.value.chars().skip(first.value.len() - 3).collect();
    assert_eq!(first.object.redacted_value, format!("sk-admin...{tail}"));
    let listed = admin
        .list_project_keys(&project.id, &Page::default(), None)
        .unwrap();
    assert!(listed.data[0].redacted_value.starts_with("sk-proj..."));
    let json = serde_json::to_string(&listed).unwrap();
    assert!(!json.contains(&project_token));
    let json = serde_json::to_string(&admin.get_admin_key(&first.object.id).unwrap()).unwrap();
    assert!(!json.contains(&first.value) && !json.contains("\"value\""));
    // Same key text under two salts gives different tags.
    let salt_a = random_bytes::<32>().unwrap();
    let salt_b = random_bytes::<32>().unwrap();
    assert_ne!(tag(&salt_a, &first.value), tag(&salt_b, &first.value));
}

#[test]
fn verification_uses_the_constant_time_hmac_comparison() {
    let salt = random_bytes::<32>().unwrap();
    let minted = mint(KeyKind::Admin).unwrap();
    let stored = tag(&salt, &minted.token);
    assert!(verify(&salt, &stored, &minted.token));
    // One changed final digit, same lookup: refused.
    let mut altered = minted.token;
    let last = altered.pop().unwrap();
    altered.push(if last == '0' { '1' } else { '0' });
    assert!(!verify(&salt, &stored, &altered));
    // The comparison is ring's constant-time HMAC verification, never `==`.
    let source = include_str!("../admin.rs");
    assert!(source.contains("ring::hmac::verify(&key, token.as_bytes(), stored_tag)"));
    assert!(!source.contains("stored_tag =="));
    assert!(!source.contains("== stored_tag"));
}

#[test]
fn project_keys_authorize_their_project_until_revoked_archived_or_expired() {
    let (_dir, admin) = open();
    admin.create_admin_key("a", None, &Actor::Owner).unwrap();
    let project = admin.create_project("p", None, &Actor::Owner).unwrap();
    let account = admin
        .create_service_account(&project.id, "bot", false, None, &Actor::Owner)
        .unwrap();
    assert_eq!(account.account.role, "member");
    let first = account.api_key.unwrap();
    assert_eq!(
        granted(&admin, &first.value),
        Credential::Project {
            key_id: first.id.clone(),
            project_id: project.id.clone(),
            service_account_id: account.account.id.clone(),
        }
    );
    assert!(
        admin
            .get_project_key(&project.id, &first.id)
            .unwrap()
            .last_used_at
            .is_some()
    );
    let second = admin
        .create_service_account_key(
            &project.id,
            &account.account.id,
            Some("two"),
            None,
            &Actor::Owner,
        )
        .unwrap();
    admin
        .delete_project_key(&project.id, &first.id, &Actor::Owner)
        .unwrap();
    assert!(matches!(
        admin.authenticate(Some(&first.value)).unwrap(),
        Verdict::Refused(_)
    ));
    granted(&admin, &second.value);
    // An expired key is refused.
    admin
        .store()
        .connection()
        .unwrap()
        .execute(
            "UPDATE admin_project_keys SET expires_at = 1 WHERE id = ?1",
            [&second.id],
        )
        .unwrap();
    assert_eq!(
        admin.authenticate(Some(&second.value)).unwrap(),
        Verdict::Refused("the API key has expired")
    );
    let third = admin
        .create_service_account_key(
            &project.id,
            &account.account.id,
            None,
            Some(3600),
            &Actor::Owner,
        )
        .unwrap();
    assert_eq!(third.expires_at, Some(third.created_at + 3600));
    granted(&admin, &third.value);
    let archived = admin.archive_project(&project.id, &Actor::Owner).unwrap();
    assert_eq!(archived.status, "archived");
    assert!(matches!(
        admin.authenticate(Some(&third.value)).unwrap(),
        Verdict::Refused(_)
    ));
    let listed = admin
        .list_project_keys(&project.id, &Page::default(), Some("inactive"))
        .unwrap();
    assert_eq!(listed.data.len(), 2);
    assert!(
        listed
            .data
            .iter()
            .all(|key| key.owner_project_access == "inactive")
    );
    assert!(matches!(
        admin.create_service_account(&project.id, "x", false, None, &Actor::Owner),
        Err(Error::Conflict(_))
    ));
}

#[test]
fn deleting_a_service_account_revokes_its_keys() {
    let (_dir, admin) = open();
    admin.create_admin_key("a", None, &Actor::Owner).unwrap();
    let project = admin.create_project("p", None, &Actor::Owner).unwrap();
    let account = admin
        .create_service_account(&project.id, "bot", false, None, &Actor::Owner)
        .unwrap();
    let token = account.api_key.unwrap().value;
    admin
        .delete_service_account(&project.id, &account.account.id, &Actor::Owner)
        .unwrap();
    assert!(matches!(
        admin.authenticate(Some(&token)).unwrap(),
        Verdict::Refused(_)
    ));
    assert!(matches!(
        admin.get_service_account(&project.id, &account.account.id),
        Err(Error::NotFound(_))
    ));
    let alone = admin
        .create_service_account(&project.id, "alone", true, None, &Actor::Owner)
        .unwrap();
    assert!(alone.api_key.is_none());
    assert_eq!(alone.account.role, "none");
    assert!(matches!(
        admin.create_service_account(&project.id, "bad", true, Some(10), &Actor::Owner),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn lists_page_in_creation_order_with_cursors() {
    let (_dir, admin) = open();
    let ids: Vec<String> = (0..5)
        .map(|index| {
            admin
                .create_project(&format!("p{index}"), None, &Actor::Owner)
                .unwrap()
                .id
        })
        .collect();
    let first = admin
        .list_projects(
            &Page {
                limit: Some(2),
                ..Page::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(
        first.data.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        ids[..2]
    );
    assert!(first.has_more);
    assert_eq!(first.last_id.as_deref(), Some(ids[1].as_str()));
    let rest = admin
        .list_projects(
            &Page {
                limit: Some(10),
                after: first.last_id,
                ..Page::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(rest.data.len(), 3);
    assert!(!rest.has_more);
    admin.archive_project(&ids[0], &Actor::Owner).unwrap();
    assert_eq!(
        admin
            .list_projects(&Page::default(), false)
            .unwrap()
            .data
            .len(),
        4
    );
    assert_eq!(
        admin
            .list_projects(&Page::default(), true)
            .unwrap()
            .data
            .len(),
        5
    );
    assert!(matches!(
        admin.list_projects(
            &Page {
                limit: Some(0),
                ..Page::default()
            },
            true
        ),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        admin.list_projects(
            &Page {
                after: Some("proj_missing".into()),
                ..Page::default()
            },
            true
        ),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        admin.modify_project(&ids[0], &ProjectChanges::default(), &Actor::Owner),
        Err(Error::Conflict(_))
    ));
    let renamed = admin
        .modify_project(
            &ids[1],
            &ProjectChanges {
                name: Some("renamed".into()),
                external_key_id: None,
            },
            &Actor::Owner,
        )
        .unwrap();
    assert_eq!(renamed.name, "renamed");
    assert!(matches!(
        admin.get_project("proj_x"),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        admin.create_project(" ", None, &Actor::Owner),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn audit_logs_record_lifecycle_events_newest_first_with_filters() {
    let (_dir, admin) = open();
    let key = admin.create_admin_key("a", None, &Actor::Owner).unwrap();
    let actor = Actor::AdminKey(key.object.id.clone());
    let project = admin.create_project("p", None, &actor).unwrap();
    admin
        .modify_project(
            &project.id,
            &ProjectChanges {
                name: Some("q".into()),
                external_key_id: None,
            },
            &actor,
        )
        .unwrap();
    let account = admin
        .create_service_account(&project.id, "bot", false, None, &actor)
        .unwrap();
    admin.archive_project(&project.id, &actor).unwrap();
    let all = admin.list_audit_logs(&AuditQuery::default()).unwrap();
    let kinds: Vec<&str> = all
        .data
        .iter()
        .map(|log| log.value["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "project.archived",
            "api_key.created",
            "service_account.created",
            "project.updated",
            "project.created",
            "api_key.created"
        ]
    );
    let created = &all.data[4].value;
    assert_eq!(created["project"]["id"], project.id);
    assert_eq!(created["project.created"]["data"]["name"], "p");
    assert_eq!(created["actor"]["type"], "api_key");
    assert_eq!(created["actor"]["api_key"]["id"], key.object.id);
    assert_eq!(all.data[5].value["actor"]["type"], "session");
    let filtered = admin
        .list_audit_logs(&AuditQuery {
            event_types: vec!["service_account.created".into()],
            ..AuditQuery::default()
        })
        .unwrap();
    assert_eq!(filtered.data.len(), 1);
    assert_eq!(
        filtered.data[0].value["service_account.created"]["id"],
        account.account.id
    );
    let by_actor = admin
        .list_audit_logs(&AuditQuery {
            actor_ids: vec![key.object.id],
            ..AuditQuery::default()
        })
        .unwrap();
    assert_eq!(by_actor.data.len(), 5);
    let page = admin
        .list_audit_logs(&AuditQuery {
            limit: Some(2),
            ..AuditQuery::default()
        })
        .unwrap();
    assert!(page.has_more);
    let next = admin
        .list_audit_logs(&AuditQuery {
            limit: Some(2),
            after: page.last_id,
            ..AuditQuery::default()
        })
        .unwrap();
    assert_eq!(next.data[0].value["type"], "service_account.created");
    let before = admin
        .list_audit_logs(&AuditQuery {
            before: next.first_id,
            ..AuditQuery::default()
        })
        .unwrap();
    assert_eq!(before.data.len(), 2);
    assert_eq!(before.data[0].value["type"], "project.archived");
    let none = admin
        .list_audit_logs(&AuditQuery {
            actor_emails: vec!["x@example.com".into()],
            ..AuditQuery::default()
        })
        .unwrap();
    assert!(none.data.is_empty());
}

#[test]
fn bucket_spans_align_to_utc_widths_and_page() {
    // 2024-11-01T00:00:00Z plus 90 minutes, to 2024-11-01T05:00:00Z.
    let midnight = 1_730_419_200;
    let start = midnight + 90 * MINUTE;
    let end = midnight + 5 * HOUR;
    let (spans, next) = bucket_spans(start, end, HOUR, 2, None).unwrap();
    assert_eq!(
        spans,
        [
            (start, midnight + 2 * HOUR),
            (midnight + 2 * HOUR, midnight + 3 * HOUR)
        ]
    );
    let next = next.unwrap();
    let (spans, last) = bucket_spans(start, end, HOUR, 24, Some(&next)).unwrap();
    assert_eq!(
        spans,
        [
            (midnight + 3 * HOUR, midnight + 4 * HOUR),
            (midnight + 4 * HOUR, end)
        ]
    );
    assert!(last.is_none());
    let (days, _) = bucket_spans(midnight, midnight + 3 * DAY, DAY, 7, None).unwrap();
    assert_eq!(days.len(), 3);
    assert!(days.iter().all(|(a, b)| b - a == DAY && a % DAY == 0));
    assert!(bucket_spans(end, start, HOUR, 1, None).is_err());
    assert!(bucket_spans(start, end, HOUR, 1, Some("page_1")).is_err());
    assert!(bucket_spans(start, end, HOUR, 1, Some("garbage")).is_err());
}

fn completion(
    at: i64,
    project: Option<&str>,
    key: Option<&str>,
    input: i64,
    output: i64,
) -> UsageRecord {
    UsageRecord {
        at: Some(at),
        project_id: project.map(Into::into),
        api_key_id: key.map(Into::into),
        model: Some("bonsai".into()),
        input_tokens: input,
        input_cached_tokens: input / 2,
        output_tokens: output,
        ..UsageRecord::new(UsageKind::Completions)
    }
}

fn query(kind: UsageKind, start: i64, end: i64, width: i64) -> UsageQuery {
    UsageQuery {
        kind,
        start_time: start,
        end_time: Some(end),
        bucket_width: width,
        limit: None,
        page: None,
        project_ids: Vec::new(),
        user_ids: Vec::new(),
        api_key_ids: Vec::new(),
        models: Vec::new(),
        batch: None,
        group_by: Vec::new(),
    }
}

#[test]
fn usage_sums_records_per_bucket_and_group() {
    let (_dir, admin) = open();
    let day = 1_730_419_200;
    for record in [
        completion(day + 10, Some("proj_a"), Some("key_a"), 100, 10),
        completion(day + 20, Some("proj_b"), Some("key_b"), 50, 5),
        completion(day + HOUR + 1, Some("proj_a"), Some("key_a"), 8, 2),
        completion(day + 2 * DAY, None, None, 1, 1),
        completion(day - 1, Some("proj_a"), None, 1000, 1000),
    ] {
        admin.record_usage(&record).unwrap();
    }
    admin
        .record_usage(&UsageRecord {
            at: Some(day + 30),
            input_tokens: 7,
            ..UsageRecord::new(UsageKind::Embeddings)
        })
        .unwrap();
    let page = admin
        .usage(&query(UsageKind::Completions, day, day + 3 * DAY, DAY))
        .unwrap();
    assert_eq!(page.object, "page");
    assert_eq!(page.data.len(), 3);
    assert!(!page.has_more);
    let first = &page.data[0].results;
    assert_eq!(first.len(), 1);
    assert_eq!(first[0]["input_tokens"], 158);
    assert_eq!(first[0]["input_cached_tokens"], 50 + 25 + 4);
    assert_eq!(first[0]["input_uncached_tokens"], 158 - 79);
    assert_eq!(first[0]["output_tokens"], 17);
    assert_eq!(first[0]["num_model_requests"], 3);
    assert_eq!(first[0]["project_id"], Value::Null);
    assert_eq!(page.data[1].results.len(), 0);
    assert_eq!(page.data[2].results[0]["num_model_requests"], 1);

    let mut grouped = query(UsageKind::Completions, day, day + DAY, HOUR);
    grouped.group_by = vec!["project_id".into(), "api_source".into()];
    grouped.limit = Some(3);
    let page = admin.usage(&grouped).unwrap();
    assert_eq!(page.data.len(), 3);
    assert!(page.has_more);
    assert_eq!(page.next_page.as_deref(), Some("page_1730430000"));
    let hour = &page.data[0].results;
    assert_eq!(hour.len(), 2);
    assert_eq!(hour[0]["project_id"], "proj_a");
    assert_eq!(hour[0]["input_tokens"], 100);
    assert_eq!(hour[0]["api_source"], "unlabeled");
    assert_eq!(hour[1]["project_id"], "proj_b");
    assert_eq!(page.data[1].results[0]["input_tokens"], 8);

    let mut filtered = query(UsageKind::Completions, day, day + DAY, DAY);
    filtered.api_key_ids = vec!["key_b".into()];
    let page = admin.usage(&filtered).unwrap();
    assert_eq!(page.data[0].results[0]["input_tokens"], 50);

    let page = admin
        .usage(&query(UsageKind::Embeddings, day, day + DAY, DAY))
        .unwrap();
    assert_eq!(
        page.data[0].results[0]["object"],
        "organization.usage.embeddings.result"
    );
    assert_eq!(page.data[0].results[0]["input_tokens"], 7);

    let mut bad = query(UsageKind::Embeddings, day, day + DAY, DAY);
    bad.group_by = vec!["batch".into()];
    assert!(matches!(admin.usage(&bad), Err(Error::InvalidArgument(_))));
    let mut bad = query(UsageKind::Completions, day, day + DAY, DAY);
    bad.limit = Some(32);
    assert!(matches!(admin.usage(&bad), Err(Error::InvalidArgument(_))));
}
