# local-services

Durable native services shared by the `local-ai` server and native Rust
callers: Conversations, Files and Uploads, Batches, and a read-only repository
Workspace. Nothing here loads or needs a model, and nothing reaches the
network. Batches only queue and record lines; the server's worker generates
them on the one loaded model.

There is one fixed implementation per job, not a set of optional backends:

| Job | Implementation |
| --- | --- |
| Durable state | One bundled SQLite database through [`rusqlite` 0.40.2](https://docs.rs/rusqlite/0.40.2/rusqlite/) (`bundled` feature, no other features) |
| File bytes | Owner-only blob files beside the database, tracked by SQLite rows |
| Ranked repository search | A fresh in-RAM [Tantivy 0.26.2](https://docs.rs/tantivy/0.26.2/tantivy/) index built for each call (default features off) |
| Exact or regex matches | ripgrep's [`grep-searcher` 0.1.17](https://docs.rs/grep-searcher/0.1.17/grep_searcher/) and [`grep-regex` 0.1.14](https://docs.rs/grep-regex/0.1.14/grep_regex/) |
| Discovery walk | [`ignore` 0.4.33](https://docs.rs/ignore/0.4.33/ignore/) (`.gitignore`, `.ignore`, `.git/info/exclude`) |
| Rooted file access | [`cap-std` 4.0.3](https://docs.rs/cap-std/4.0.3/cap_std/fs/struct.Dir.html) `Dir` handles |

Every method blocks. Async callers should run them on a bounded blocking pool,
as the server does.

## Native usage

No model, server or HTTP is involved. Add `local-services` (a path dependency
in this workspace) and `serde_json`:

```rust,no_run
use local_services::workspace::{ReadOptions, SearchOptions, Workspace};
use local_services::{Metadata, Store};
use serde_json::json;

fn main() -> Result<(), local_services::Error> {
    // Creates the directory (0700) and its database (0600) if needed.
    let store = Store::open(std::env::temp_dir().join("local-services-demo"))?;
    let mut metadata = Metadata::new();
    metadata.insert("topic".into(), "demo".into());
    let conversation = store.create_conversation(
        &metadata,
        vec![json!({"role": "user", "content": "Where is ranked search?"})],
    )?;
    let history = store.conversation_history(&conversation.id)?;
    println!("{}: {} item(s)", conversation.id, history.items.len());

    let workspace = Workspace::open(".")?;
    // Ranked: BM25 over a fresh in-memory index of the visible text files.
    for file in workspace.search("ranked search")?.files {
        println!("{:.2} {}:{} {}", file.score, file.path, file.line_number, file.snippet);
    }
    // Exact: every matching line, in path order.
    let found = workspace.find_matches(&SearchOptions::literal("fn search"))?;
    if let Some(hit) = found.matches.first() {
        let read = workspace.read_file(
            &hit.path,
            &ReadOptions {
                start_line: hit.line_number,
                end_line: Some(hit.line_number + 5),
                ..ReadOptions::default()
            },
        )?;
        for line in read.lines {
            println!("{:>5} {}", line.number, line.text);
        }
    }
    Ok(())
}
```

## Store

`Store::open(dir)` creates or opens one database, `services.sqlite3`, and
applies each component's versioned migrations exactly once inside one
immediate transaction. A database written by a newer build is refused. The
directory must be `0700` and the database and its `-wal`, `-shm` and
`-journal` sidecars `0600`. A symlink, the wrong file type, or any group or
other permission bit is refused with `PermissionDenied`, never repaired. Every
operation opens its own connection with:

- [`SQLITE_OPEN_NOFOLLOW`](https://sqlite.org/c3ref/open.html) and a 5 s
  [busy timeout](https://sqlite.org/c3ref/busy_timeout.html);
- [WAL](https://sqlite.org/wal.html) journaling, required: if WAL cannot be
  enabled the open fails;
- [`synchronous = FULL`](https://sqlite.org/pragma.html#pragma_synchronous),
  which SQLite documents as ACID in WAL mode, plus
  [`fullfsync`](https://sqlite.org/pragma.html#pragma_fullfsync) and
  [`checkpoint_fullfsync`](https://sqlite.org/pragma.html#pragma_checkpoint_fullfsync)
  (macOS `F_FULLFSYNC`);
- [`foreign_keys = ON`](https://sqlite.org/pragma.html#pragma_foreign_keys) and
  [`trusted_schema = OFF`](https://sqlite.org/pragma.html#pragma_trusted_schema).

WAL lets several handles, threads and processes share the store. Identifiers
carry 192 bits of OS randomness.

**Data stays local but is stored as plaintext.** Owner-only permissions are the
only protection. Nothing is encrypted at rest.

`local-ai serve` always opens
`$HOME/Library/Application Support/local-ai/services`, with no flag or backend
choice. It refuses to start if that store cannot be opened. The model,
checkpoint and every other model artifact are unchanged.

## Conversations

`create_conversation`, `get_conversation`, `update_conversation` (which replaces
the metadata), `delete_conversation`, `add_items`, `list_items`, `get_item`,
`delete_item`, `conversation_history` and `append_items`.

- Items are checked against the forms local generation implements:
  `message` (`user`, `assistant`, `system`, `developer`, with text content),
  `function_call`, `function_call_output` and `reasoning`. Images, files and
  any other item type are refused. Nothing is stored and silently ignored.
- Limits: 16 metadata pairs (64-character keys, 512-character values). Up to
  20 items per create or add, 256 per `append_items`, and 4 MiB per item.
  Pages hold 1 to 100 items, 20 by default, newest first. IDs and request
  keys are at most 128 characters.
- Every insertion or deletion bumps an internal `version`. `append_items`
  takes an idempotency `request_id` and an `expected_version`. A retry with
  the same request returns the stored items with `replayed: true`. A stale
  version, or the same key with different items, is a `Conflict`, and nothing
  is stored.
- Deletion is logical. The conversation and its items become unreachable,
  but its item rows stay in the database. No purge exists for them.

## Files and Uploads

`create_file` streams content into a new file. Uploads use `create_upload`,
`add_upload_part`, `complete_upload` (with part order and an optional hex
`md5` that is checked against the assembled bytes before completion) and
`cancel_upload`.

- Limits: 512 MiB per file (200 MiB for `batch`), 64 MiB per part, and 8 GiB
  per upload. An upload expires one hour after it is created. File
  `expires_after` accepts 3600 to 2,592,000 s, and `batch` files default to 30
  days. File pages hold up to 10,000 entries.
- A `purpose` is checked and stored verbatim. `vision`, `fine-tune`, `evals`
  and the other purposes enable no processing. `batch_output` is internal:
  callers cannot create it.
- Crash safety: the metadata row is committed before any bytes are written.
  Bytes go to a temporary file, are `fsync`ed and renamed into
  `file-store/blobs/`, then the directory is `fsync`ed and the row is
  committed. Removal deletes metadata before bytes. User filenames never
  become paths.
- **Expiry is enforced on access.** Expired files report not found and are
  left out of listings, and expired uploads refuse every change. Bytes are
  removed physically only by `purge_file_storage()`, which also deletes
  expired upload parts, staging blobs abandoned for more than 24 hours, and
  orphaned disk entries. The server calls it once at startup. There is no
  background sweeper. Upload rows remain as small metadata records.

## Batches

`create_batch` validates the whole `purpose=batch` JSONL file before storing
anything. A file is at most 50,000 lines of up to 2 MiB each. Every line is
`{custom_id, method: "POST", url, body}` with a unique `custom_id` (512
characters at most), the batch's endpoint, one shared `body.model` that must
be the caller's installed model, and neither `stream` nor `background`. Only
`/v1/responses`, `/v1/chat/completions` and `/v1/completions` are accepted, and
`completion_window` must be `24h`. A valid batch is created `in_progress`.
There is no asynchronous `validating` phase.

Execution uses a lease and settles each line once:

1. `lease_batch` takes an exclusive advisory lock on an owner-only file under
   `batch-locks/`. The OS releases it when the process dies, so another
   process or a later run can reclaim the batch.
2. `next_batch_line` returns the lowest pending line, and `settle_batch_line`
   records its result only if the line is still pending.
3. `finish_batch` freezes the lines and writes the output and error files. The
   files have deterministic IDs (`file_<batch>_output`, `file_<batch>_error`)
   and the internal `batch_output` purpose, are capped at 512 MiB each, and
   expire after 30 days. Each line's output record ID is fixed when the batch
   is created.

Settlement happens exactly once. Generation does not: a crash between
generating and settling a line generates that line again on resume. A result
file that was committed is reused by an interrupted finish and is never
recreated after deletion or expiry. Expiry and cancellation are observed at
line boundaries.

## Workspace

`Workspace::open(root)` gives synchronous, read-only access to one directory
tree. It never writes, executes or spawns processes.

- **Rooted access.** Every file whose contents are returned is opened one
  component at a time through a `cap-std` `Dir` on the root. `..`, absolute
  paths and any `.git` component are refused. A symlink component, including
  one swapped in mid-open, is refused rather than followed. A discovered file
  is re-opened and its inode checked, so a file swapped after discovery is
  skipped.
- **Discovery** (`list_files`, `search`, `find_matches`) honours `.gitignore`,
  `.ignore` and `.git/info/exclude`. It skips hidden entries and symlinks and
  never enters `.git`. It deliberately ignores the user's global gitignore.
  `read_file` returns gitignored or hidden paths only with
  `include_ignored: true`, and still refuses symlinks, `.git` and escapes.
- **`search(query)`**, the default ranked search, indexes the current visible
  text files into a new `Index::create_in_ram` index on every call. It uses
  `path` and `body` fields, Tantivy's
  [default tokenizer](https://docs.rs/tantivy/0.26.2/tantivy/tokenizer/index.html)
  and one indexing thread that is joined before the call returns. Edits,
  deletions and new files are therefore always reflected, and nothing is
  persisted, cached or left running. The query is never parsed as query
  syntax: it is split by the same tokenizer (`EngineHandle::decide` becomes
  `enginehandle`, `decide`). Files are ranked by BM25, a query equal to a file's
  path gets a boost, and equal scores are ordered by path. Each result carries
  a score (comparable only within one result set), a line number and a snippet.
  This search has no path scope. Each call pays the full indexing cost again.
- **`find_matches(&SearchOptions)`** is a separate operation. It reports every
  line that matches an exact literal or a Rust `regex` (optionally
  case-insensitive, optionally scoped to a path), with line and column, in
  path then line order. Matches never span lines.
- **`read_file(path, &ReadOptions)`** returns a 1-based inclusive line range
  with `next_line` and `eof`. Invalid UTF-8 becomes U+FFFD, and binary files
  (a NUL byte in the content) are refused by reads and skipped by `search` and
  `find_matches`.

Default `WorkspaceLimits`: 100,000 walked entries, 4 MiB per file, 256 KiB per
read, 10,000 listed entries, 1,000 matches, 20 ranked files, 20,000 files and
32 MiB indexed per search, 1 MiB of output, 4 KiB per line or snippet, 4 KiB
per query or pattern, and a 1 MiB compiled regex. Whenever a limit cuts a
result short, the result says so through `truncated` and a `Truncation`
reason. Per-call options can lower these limits but never raise them.

Workspace has **no HTTP endpoint**: the server exposes no way to read or search
the disk. It is a native API for in-process callers.

### Search evaluation

Before implementation, a local harness (`.amp/in/artifacts/search-eval-*`,
not committed) compared prototypes on a copy of `local-ai` and
`local-services`: 36 files, 844 KiB. It ran 24 identifier, concept and
natural-language queries, with expected files fixed before the runs.

| Engine | MRR | hit@1 | hit@5 |
| --- | ---: | ---: | ---: |
| literal grep (path order) | 0.25 | 5 | 8 |
| case-insensitive OR-regex grep | 0.34 | 5 | 12 |
| Tantivy, raw query | 0.76 | 16 | 21 |
| Tantivy, query punctuation removed | 0.81 | 17 | 22 |

Literal grep found no expected file for any concept or natural-language query.
Raw Tantivy query syntax failed on `EngineHandle::decide`, which is why the
shipped search tokenizes the query instead of parsing it. The corpus is small
and the query set was written by the same author, so this shows a direction,
not general retrieval quality.

The prototype used a persisted index, so its timings do not describe the
shipped per-call in-RAM index, which has not been benchmarked. The same
evaluation recommended an index-free BM25 scan (MRR 0.74). The project chose
the one Tantivy path instead.

## Not provided

The following are not part of this crate:

- Encryption at rest.
- A background sweeper or poller.
- A persistent search index, embeddings or vector stores.
- Physical purge of deleted conversations.
- Compaction.
- Model training.
- Coding-quality benchmarks.
- Any change to the model.

For server behaviour (routes, Responses `conversation`, and the batch worker),
see [docs/BONSAI.md](../docs/BONSAI.md#native-services).
