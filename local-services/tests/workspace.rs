#![allow(clippy::expect_used)]

//! Observable behaviour of the read-only workspace tools on throwaway trees.

use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;

use local_services::Error;
use local_services::workspace::{
    ListOptions, PatternMode, ReadOptions, SearchOptions, Truncation, Workspace, WorkspaceLimits,
};
use tempfile::TempDir;

fn write(root: &Path, rel: &str, contents: impl AsRef<[u8]>) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent directories");
    }
    fs::write(path, contents).expect("write test file");
}

/// A small repository with ignored, hidden and `.git` content that must stay
/// invisible to discovery.
fn repo() -> TempDir {
    let dir = tempfile::tempdir().expect("create temp dir");
    let root = dir.path();
    write(root, ".gitignore", "*.log\nbuild/\n");
    write(root, ".git/config", "token = SECRET\n");
    write(root, ".env", "API_KEY=SECRET\n");
    write(root, "debug.log", "SECRET in a log\n");
    write(root, "build/out.txt", "SECRET build output\n");
    write(root, "b.txt", "beta\n");
    write(root, "a.txt", "alpha\nneedle one\n");
    write(root, "src/main.rs", "fn main() {\n    // needle two\n}\n");
    write(root, "src/lib/z.rs", "needle three\n");
    dir
}

fn paths(workspace: &Workspace, options: &ListOptions) -> Vec<String> {
    workspace
        .list_files(options)
        .expect("list files")
        .files
        .into_iter()
        .map(|entry| entry.path)
        .collect()
}

fn read(workspace: &Workspace, path: &str, options: &ReadOptions) -> Vec<String> {
    workspace
        .read_file(path, options)
        .expect("read file")
        .lines
        .into_iter()
        .map(|line| line.text)
        .collect()
}

fn explicit() -> ReadOptions {
    ReadOptions {
        include_ignored: true,
        ..ReadOptions::default()
    }
}

#[test]
fn open_rejects_missing_or_non_directory_roots() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let missing = Workspace::open(dir.path().join("missing"));
    assert!(matches!(missing, Err(Error::NotFound(_))));
    write(dir.path(), "file.txt", "x");
    let file = Workspace::open(dir.path().join("file.txt"));
    assert!(matches!(file, Err(Error::InvalidArgument(_))));
}

#[test]
fn listing_honours_gitignore_hides_dotfiles_and_is_sorted() {
    let dir = repo();
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    let list = workspace.list_files(&ListOptions::default()).expect("list");
    let listed: Vec<_> = list.files.iter().map(|entry| entry.path.as_str()).collect();
    assert_eq!(listed, ["a.txt", "b.txt", "src/lib/z.rs", "src/main.rs"]);
    assert!(!list.truncated);
    assert_eq!(list.truncation, None);
    assert_eq!(list.files[1].size, 5);
}

#[test]
fn listing_is_identical_regardless_of_creation_order() {
    let first = tempfile::tempdir().expect("create temp dir");
    let second = tempfile::tempdir().expect("create temp dir");
    let names = ["m.txt", "a/z.txt", "a/b.txt", "Z.txt", "a.txt", "a-b.txt"];
    for name in names {
        write(first.path(), name, "x");
    }
    for name in names.iter().rev() {
        write(second.path(), name, "x");
    }
    let one = Workspace::open(first.path()).expect("open first");
    let two = Workspace::open(second.path()).expect("open second");
    let expected = ["Z.txt", "a-b.txt", "a.txt", "a/b.txt", "a/z.txt", "m.txt"];
    assert_eq!(paths(&one, &ListOptions::default()), expected);
    assert_eq!(paths(&two, &ListOptions::default()), expected);
    assert_eq!(paths(&one, &ListOptions::default()), expected);
}

#[test]
fn listing_scopes_to_a_subdirectory_with_root_ignore_rules() {
    let dir = repo();
    write(
        dir.path(),
        "src/trace.log",
        "ignored by the root .gitignore",
    );
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    let scoped = ListOptions {
        path: Some("./src".to_owned()),
        ..ListOptions::default()
    };
    assert_eq!(paths(&workspace, &scoped), ["src/lib/z.rs", "src/main.rs"]);
    let missing = workspace.list_files(&ListOptions {
        path: Some("nope".to_owned()),
        ..ListOptions::default()
    });
    assert!(matches!(missing, Err(Error::NotFound(_))));
}

#[test]
fn listing_reports_entry_and_walk_truncation() {
    let dir = tempfile::tempdir().expect("create temp dir");
    for index in 0..10 {
        write(dir.path(), &format!("f{index}.txt"), "x");
    }
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    let list = workspace
        .list_files(&ListOptions {
            max_entries: Some(3),
            ..ListOptions::default()
        })
        .expect("list");
    let listed: Vec<_> = list.files.iter().map(|entry| entry.path.as_str()).collect();
    assert_eq!(listed, ["f0.txt", "f1.txt", "f2.txt"]);
    assert!(list.truncated);
    assert_eq!(list.truncation, Some(Truncation::Entries));

    let small_walk = Workspace::open_with_limits(
        dir.path(),
        WorkspaceLimits {
            max_walk_entries: 4,
            ..WorkspaceLimits::default()
        },
    )
    .expect("open workspace");
    let walked = small_walk
        .list_files(&ListOptions::default())
        .expect("list");
    assert_eq!(walked.files.len(), 4);
    assert!(walked.truncated);
    assert_eq!(walked.truncation, Some(Truncation::WalkEntries));
    let again = small_walk
        .list_files(&ListOptions::default())
        .expect("list");
    assert_eq!(walked, again);

    let zero = workspace.list_files(&ListOptions {
        max_entries: Some(0),
        ..ListOptions::default()
    });
    assert!(matches!(zero, Err(Error::InvalidArgument(_))));
}

#[test]
fn read_file_returns_inclusive_line_ranges() {
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "lines.txt", "one\ntwo\r\nthree\nfour");
    let workspace = Workspace::open(dir.path()).expect("open workspace");

    let all = workspace
        .read_file("lines.txt", &ReadOptions::default())
        .expect("read");
    assert_eq!(all.path, "lines.txt");
    let numbers: Vec<_> = all.lines.iter().map(|line| line.number).collect();
    assert_eq!(numbers, [1, 2, 3, 4]);
    assert!(all.eof && !all.truncated && all.next_line.is_none());

    let middle = workspace
        .read_file(
            "./lines.txt",
            &ReadOptions {
                start_line: 2,
                end_line: Some(3),
                ..ReadOptions::default()
            },
        )
        .expect("read");
    let texts: Vec<_> = middle.lines.iter().map(|line| line.text.as_str()).collect();
    assert_eq!(texts, ["two", "three"]);
    assert_eq!(middle.next_line, Some(4));
    assert!(!middle.eof && !middle.truncated);

    let past_end = workspace
        .read_file(
            "lines.txt",
            &ReadOptions {
                start_line: 99,
                ..ReadOptions::default()
            },
        )
        .expect("read");
    assert!(past_end.lines.is_empty() && past_end.eof && !past_end.truncated);

    for (start_line, end_line) in [(0, None), (3, Some(2))] {
        let invalid = workspace.read_file(
            "lines.txt",
            &ReadOptions {
                start_line,
                end_line,
                ..ReadOptions::default()
            },
        );
        assert!(matches!(invalid, Err(Error::InvalidArgument(_))));
    }
    let missing = workspace.read_file("absent.txt", &ReadOptions::default());
    assert!(matches!(missing, Err(Error::NotFound(_))));
    let directory = workspace.read_file(".", &ReadOptions::default());
    assert!(matches!(directory, Err(Error::InvalidArgument(_))));
}

#[test]
fn read_file_reports_byte_and_file_size_truncation() {
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "lines.txt", "aaaa\nbbbb\ncccc\n");
    write(dir.path(), "long.txt", "é".repeat(10));
    let workspace = Workspace::open(dir.path()).expect("open workspace");

    let budget = workspace
        .read_file(
            "lines.txt",
            &ReadOptions {
                max_bytes: Some(10),
                ..ReadOptions::default()
            },
        )
        .expect("read");
    assert_eq!(budget.lines.len(), 2);
    assert!(budget.truncated);
    assert_eq!(budget.truncation, Some(Truncation::OutputBytes));
    assert_eq!(budget.next_line, Some(3));

    let cut = workspace
        .read_file(
            "long.txt",
            &ReadOptions {
                max_bytes: Some(5),
                ..ReadOptions::default()
            },
        )
        .expect("read");
    assert_eq!(cut.lines[0].text, "éé");
    assert!(cut.lines[0].truncated && cut.truncated);

    let capped = Workspace::open_with_limits(
        dir.path(),
        WorkspaceLimits {
            max_file_bytes: 7,
            ..WorkspaceLimits::default()
        },
    )
    .expect("open workspace");
    let partial = capped
        .read_file("lines.txt", &ReadOptions::default())
        .expect("read");
    assert_eq!(
        read(&capped, "lines.txt", &ReadOptions::default()),
        ["aaaa", "bb"]
    );
    assert!(partial.truncated && !partial.eof);
    assert_eq!(partial.truncation, Some(Truncation::FileBytes));
}

#[test]
fn read_file_requires_explicit_opt_in_for_ignored_and_hidden_files() {
    let dir = repo();
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    for path in ["debug.log", "build/out.txt", ".env", ".gitignore"] {
        let refused = workspace.read_file(path, &ReadOptions::default());
        assert!(
            matches!(refused, Err(Error::InvalidArgument(_))),
            "{path} must need include_ignored"
        );
    }
    assert_eq!(
        read(&workspace, "debug.log", &explicit()),
        ["SECRET in a log"]
    );
    assert_eq!(read(&workspace, ".env", &explicit()), ["API_KEY=SECRET"]);
    for path in [".git/config", "./.git/config", ".GIT/config"] {
        let refused = workspace.read_file(path, &explicit());
        assert!(
            matches!(refused, Err(Error::InvalidArgument(_))),
            "{path} must never be readable"
        );
    }
}

#[test]
fn non_utf8_text_is_lossy_and_binary_files_are_refused_or_skipped() {
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "latin1.txt", b"caf\xe9 needle\n");
    write(dir.path(), "blob.bin", b"needle\0\x01\x02needle\n");
    let workspace = Workspace::open(dir.path()).expect("open workspace");

    let latin1 = workspace
        .read_file("latin1.txt", &ReadOptions::default())
        .expect("read");
    assert_eq!(latin1.lines[0].text, "caf\u{fffd} needle");
    assert!(latin1.invalid_utf8);
    let binary = workspace.read_file("blob.bin", &ReadOptions::default());
    assert!(matches!(binary, Err(Error::InvalidArgument(_))));

    let results = workspace
        .find_matches(&SearchOptions::literal("needle"))
        .expect("search");
    let hits: Vec<_> = results.matches.iter().map(|m| m.path.as_str()).collect();
    assert_eq!(hits, ["latin1.txt"]);
    assert_eq!(results.matches[0].line, "caf\u{fffd} needle");
    assert_eq!(results.skipped_binary_files, 1);
    assert_eq!(results.files_searched, 1);
}

#[test]
fn find_matches_distinguishes_literal_and_regex_patterns() {
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "a.txt", "abc\na.c\nABC\n");
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    let lines = |options: &SearchOptions| -> Vec<(u64, u64)> {
        workspace
            .find_matches(options)
            .expect("search")
            .matches
            .iter()
            .map(|m| (m.line_number, m.column))
            .collect()
    };
    assert_eq!(lines(&SearchOptions::literal("a.c")), [(2, 1)]);
    assert_eq!(lines(&SearchOptions::regex("a.c")), [(1, 1), (2, 1)]);
    assert_eq!(lines(&SearchOptions::regex("c$")), [(1, 3), (2, 3)]);
    let insensitive = SearchOptions {
        case_insensitive: true,
        ..SearchOptions::literal("bc")
    };
    assert_eq!(lines(&insensitive), [(1, 2), (3, 2)]);
    let scoped = SearchOptions {
        path: Some("a.txt".to_owned()),
        mode: PatternMode::Regex,
        ..SearchOptions::literal("^A")
    };
    assert_eq!(lines(&scoped), [(3, 1)]);
}

#[test]
fn find_matches_rejects_invalid_empty_and_oversized_patterns() {
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "a.txt", "abc\n");
    let workspace = Workspace::open_with_limits(
        dir.path(),
        WorkspaceLimits {
            max_pattern_bytes: 64,
            max_regex_bytes: 4 * 1024,
            ..WorkspaceLimits::default()
        },
    )
    .expect("open workspace");
    for options in [
        SearchOptions::regex("(unclosed"),
        SearchOptions::literal(""),
        SearchOptions::literal("a".repeat(65)),
        SearchOptions::regex(r"\w{1000}"),
        SearchOptions::literal("a\nb"),
    ] {
        let result = workspace.find_matches(&options);
        assert!(
            matches!(result, Err(Error::InvalidArgument(_))),
            "{:?} must be rejected",
            options.pattern
        );
    }
    assert_eq!(
        workspace
            .find_matches(&SearchOptions::literal("(unclosed"))
            .expect("literal parentheses are fine")
            .matches
            .len(),
        0
    );
}

#[test]
fn find_matches_never_reads_ignored_hidden_or_git_content() {
    let dir = repo();
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    let results = workspace
        .find_matches(&SearchOptions::literal("SECRET"))
        .expect("search");
    assert_eq!(results.matches.len(), 0);
    assert!(!results.truncated);

    let needles = workspace
        .find_matches(&SearchOptions::literal("needle"))
        .expect("search");
    let hits: Vec<_> = needles
        .matches
        .iter()
        .map(|m| (m.path.as_str(), m.line_number))
        .collect();
    assert_eq!(
        hits,
        [("a.txt", 2), ("src/lib/z.rs", 1), ("src/main.rs", 2)]
    );
    assert_eq!(needles.files_searched, 4);
}

#[test]
fn find_matches_reports_result_output_and_size_limits() {
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "a.txt", "hit\nhit\nhit\n");
    write(dir.path(), "b.txt", "hit\nhit\n");
    write(dir.path(), "c.txt", format!("hit {}\n", "x".repeat(100)));
    let workspace = Workspace::open(dir.path()).expect("open workspace");

    let limited = workspace
        .find_matches(&SearchOptions {
            max_results: Some(4),
            ..SearchOptions::literal("hit")
        })
        .expect("search");
    let hits: Vec<_> = limited
        .matches
        .iter()
        .map(|m| (m.path.as_str(), m.line_number))
        .collect();
    assert_eq!(
        hits,
        [("a.txt", 1), ("a.txt", 2), ("a.txt", 3), ("b.txt", 1)]
    );
    assert!(limited.truncated);
    assert_eq!(limited.truncation, Some(Truncation::Results));

    let exact = workspace
        .find_matches(&SearchOptions {
            max_results: Some(6),
            ..SearchOptions::literal("hit")
        })
        .expect("search");
    assert_eq!(exact.matches.len(), 6);
    assert!(!exact.truncated);

    let budget = workspace
        .find_matches(&SearchOptions {
            max_output_bytes: Some(60),
            ..SearchOptions::literal("hit")
        })
        .expect("search");
    assert!(budget.truncated);
    assert_eq!(budget.truncation, Some(Truncation::OutputBytes));
    assert!(budget.matches.len() < 6);

    let small = Workspace::open_with_limits(
        dir.path(),
        WorkspaceLimits {
            max_file_bytes: 20,
            max_line_bytes: 10,
            ..WorkspaceLimits::default()
        },
    )
    .expect("open workspace");
    let skipped = small
        .find_matches(&SearchOptions::literal("hit"))
        .expect("search");
    assert_eq!(skipped.matches.len(), 5);
    assert_eq!(skipped.skipped_large_files, 1);

    let short_lines = Workspace::open_with_limits(
        dir.path(),
        WorkspaceLimits {
            max_line_bytes: 10,
            ..WorkspaceLimits::default()
        },
    )
    .expect("open workspace");
    let long = short_lines
        .find_matches(&SearchOptions {
            path: Some("c.txt".to_owned()),
            ..SearchOptions::literal("hit")
        })
        .expect("search");
    assert_eq!(long.matches[0].line, "hit xxxxxx");
    assert!(long.matches[0].line_truncated);
}

#[test]
fn paths_that_escape_the_root_are_rejected() {
    let outside = tempfile::tempdir().expect("create temp dir");
    write(outside.path(), "secret.txt", "OUTSIDE needle\n");
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "inside/ok.txt", "ok\n");
    symlink(
        outside.path().join("secret.txt"),
        dir.path().join("escape.txt"),
    )
    .expect("symlink");
    symlink(outside.path(), dir.path().join("escape_dir")).expect("symlink");
    symlink("../../", dir.path().join("inside/up")).expect("symlink");
    let workspace = Workspace::open(dir.path()).expect("open workspace");

    let absolute = outside.path().join("secret.txt");
    let absolute = absolute.to_str().expect("utf-8 temp path");
    for path in [
        "../secret.txt",
        "inside/../../secret.txt",
        absolute,
        "/etc/passwd",
        "escape.txt",
        "escape_dir/secret.txt",
        "inside/up/secret.txt",
    ] {
        for options in [ReadOptions::default(), explicit()] {
            let result = workspace.read_file(path, &options);
            assert!(result.is_err(), "{path} must not be readable");
            assert!(
                !matches!(result, Err(Error::Io(_))),
                "{path} must be refused, not fail with an I/O error"
            );
        }
    }
    let scope = workspace.list_files(&ListOptions {
        path: Some("../".to_owned()),
        ..ListOptions::default()
    });
    assert!(matches!(scope, Err(Error::InvalidArgument(_))));

    assert_eq!(
        paths(&workspace, &ListOptions::default()),
        ["inside/ok.txt"]
    );
    let results = workspace
        .find_matches(&SearchOptions::literal("OUTSIDE"))
        .expect("search");
    assert_eq!(results.matches.len(), 0);
}

#[test]
fn symlinks_are_never_discovered_or_followed_even_when_explicit() {
    let outside = tempfile::tempdir().expect("create temp dir");
    write(outside.path(), "secret.txt", "OUTSIDE\n");
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "real/target.txt", "needle\n");
    write(dir.path(), ".git/config", "token = SECRET\n");
    symlink("real/target.txt", dir.path().join("link.txt")).expect("symlink");
    symlink("real", dir.path().join("linkdir")).expect("symlink");
    symlink(".git/config", dir.path().join("gitlink")).expect("symlink");
    symlink(".git", dir.path().join("gitdir")).expect("symlink");
    symlink(
        outside.path().join("secret.txt"),
        dir.path().join("outside.txt"),
    )
    .expect("symlink");
    let workspace = Workspace::open(dir.path()).expect("open workspace");

    assert_eq!(
        paths(&workspace, &ListOptions::default()),
        ["real/target.txt"]
    );
    let results = workspace
        .find_matches(&SearchOptions::literal("needle"))
        .expect("find matches");
    let hits: Vec<_> = results.matches.iter().map(|m| m.path.as_str()).collect();
    assert_eq!(hits, ["real/target.txt"]);
    assert_eq!(ranked(&workspace, "needle"), ["real/target.txt"]);
    for query in ["SECRET token", "OUTSIDE"] {
        assert_eq!(ranked(&workspace, query), NO_HITS, "{query} leaked");
    }

    for path in [
        "link.txt",
        "linkdir/target.txt",
        "gitlink",
        "gitdir/config",
        "outside.txt",
    ] {
        for options in [ReadOptions::default(), explicit()] {
            let refused = workspace.read_file(path, &options);
            assert!(
                matches!(refused, Err(Error::InvalidArgument(_))),
                "{path} must never be followed: {refused:?}"
            );
        }
    }
    assert_eq!(read(&workspace, "real/target.txt", &explicit()), ["needle"]);
}

const NO_HITS: [&str; 0] = [];

fn ranked(workspace: &Workspace, query: &str) -> Vec<String> {
    workspace
        .search(query)
        .expect("ranked search")
        .files
        .into_iter()
        .map(|file| file.path)
        .collect()
}

/// A small code-like tree for ranked search.
fn code_repo() -> TempDir {
    let dir = tempfile::tempdir().expect("create temp dir");
    let root = dir.path();
    write(
        root,
        "src/cipher.rs",
        "//! Reasoning cipher: encrypts reasoning items at rest.\n\
         pub struct ReasoningCipher;\n\
         impl ReasoningCipher {\n    pub fn seal(&self) {}\n}\n",
    );
    write(
        root,
        "src/engine.rs",
        "pub struct EngineHandle;\n\
         impl EngineHandle {\n    pub fn decide(&self) -> bool { true }\n}\n",
    );
    write(
        root,
        "src/limits.rs",
        "pub struct Limits {\n    pub max_walk_entries: usize,\n}\n",
    );
    write(root, "src/main.rs", "fn main() { start(); }\n");
    write(
        root,
        "src/main_helpers.rs",
        "fn start() {}\n// called from main\n",
    );
    write(
        root,
        "docs/http.md",
        "# HTTP server\n\nThe server answers HTTP/3 requests on a socket.\n",
    );
    let mut deep = "filler words\n".repeat(60);
    deep.push_str("the quokka lives on one line\n");
    write(root, "notes/deep.txt", deep);
    dir
}

#[test]
fn search_ranks_files_by_concept_words() {
    let dir = code_repo();
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    assert_eq!(ranked(&workspace, "reasoning cipher")[0], "src/cipher.rs");
    assert_eq!(ranked(&workspace, "cipher reasoning")[0], "src/cipher.rs");
    assert_eq!(
        ranked(&workspace, "http server requests")[0],
        "docs/http.md"
    );

    let none = workspace.search("zebra unicorn").expect("search");
    assert!(none.files.is_empty() && none.total_matches == 0);
    assert_eq!(none.files_indexed, 7);
    assert!(!none.truncated && none.truncation.is_none());

    let quokka = workspace.search("Quokka").expect("search");
    assert_eq!(quokka.files.len(), 1);
    let hit = &quokka.files[0];
    assert_eq!(hit.path, "notes/deep.txt");
    assert!(hit.score > 0.0);
    assert!(hit.snippet.contains("quokka"), "{:?}", hit.snippet);
    let line = workspace
        .read_file(
            &hit.path,
            &ReadOptions {
                start_line: hit.line_number,
                end_line: Some(hit.line_number),
                ..ReadOptions::default()
            },
        )
        .expect("read snippet line");
    let first = hit.snippet.lines().next().expect("snippet has a line");
    assert!(
        line.lines[0].text.contains(first),
        "line {}",
        hit.line_number
    );
}

#[test]
fn search_handles_identifiers_paths_and_punctuation() {
    let dir = code_repo();
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    assert_eq!(
        ranked(&workspace, "EngineHandle::decide")[0],
        "src/engine.rs"
    );
    assert_eq!(ranked(&workspace, "max_walk_entries"), ["src/limits.rs"]);
    assert_eq!(ranked(&workspace, "src/main.rs")[0], "src/main.rs");
    assert_eq!(ranked(&workspace, "./src/main.rs")[0], "src/main.rs");
    assert_eq!(ranked(&workspace, "main_helpers")[0], "src/main_helpers.rs");
    // Query syntax is never interpreted: these are just words.
    assert_eq!(ranked(&workspace, "#[must_use] AND (x OR"), NO_HITS);
    let engine = workspace.search("fn decide(&self)").expect("search");
    assert_eq!(engine.files[0].path, "src/engine.rs");
    assert!(engine.files[0].snippet.contains("decide"));

    for query in ["", "   ", "::&*", "a\nb"] {
        let refused = workspace.search(query);
        assert!(
            matches!(refused, Err(Error::InvalidArgument(_))),
            "{query:?} must be rejected"
        );
    }
}

#[test]
fn search_orders_ties_by_path_and_is_repeatable() {
    let dir = tempfile::tempdir().expect("create temp dir");
    for name in ["c.txt", "a.txt", "b/x.txt", "b.txt"] {
        write(dir.path(), name, "same shared words\n");
    }
    write(dir.path(), "z.txt", "unrelated\n");
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    let first = workspace.search("shared").expect("search");
    let order: Vec<_> = first.files.iter().map(|file| file.path.as_str()).collect();
    assert_eq!(order, ["a.txt", "b.txt", "b/x.txt", "c.txt"]);
    assert!(
        first
            .files
            .windows(2)
            .all(|pair| pair[0].score.to_bits() == pair[1].score.to_bits())
    );
    assert_eq!(workspace.search("shared").expect("search"), first);
}

#[test]
fn search_reflects_edits_deletes_and_new_files_immediately() {
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "notes/a.md", "alpha topic\n");
    write(dir.path(), "notes/b.md", "beta topic\n");
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    assert_eq!(ranked(&workspace, "gamma"), NO_HITS);

    write(dir.path(), "notes/a.md", "gamma topic\n");
    assert_eq!(ranked(&workspace, "gamma"), ["notes/a.md"]);
    assert_eq!(ranked(&workspace, "alpha"), NO_HITS);

    fs::remove_file(dir.path().join("notes/a.md")).expect("delete");
    assert_eq!(ranked(&workspace, "gamma"), NO_HITS);

    write(dir.path(), "notes/c.md", "gamma again\n");
    assert_eq!(ranked(&workspace, "gamma"), ["notes/c.md"]);
    assert_eq!(ranked(&workspace, "topic"), ["notes/b.md"]);
}

#[test]
fn search_never_indexes_ignored_hidden_or_git_content() {
    let dir = repo();
    let workspace = Workspace::open(dir.path()).expect("open workspace");
    let secret = workspace.search("SECRET API_KEY token").expect("search");
    assert_eq!(secret.files, []);
    assert_eq!(secret.files_indexed, 4);
    let mut needles = ranked(&workspace, "needle");
    needles.sort();
    assert_eq!(needles, ["a.txt", "src/lib/z.rs", "src/main.rs"]);
}

#[test]
fn search_discloses_index_result_and_output_limits() {
    let dir = tempfile::tempdir().expect("create temp dir");
    write(dir.path(), "a.bin", b"hit\0hit");
    write(dir.path(), "big.txt", "hit ".repeat(30));
    for index in 0..5 {
        write(dir.path(), &format!("f{index}.txt"), "hit\n");
    }
    let open = |limits: WorkspaceLimits| {
        Workspace::open_with_limits(
            dir.path(),
            WorkspaceLimits {
                max_file_bytes: 64,
                ..limits
            },
        )
        .expect("open workspace")
    };
    let defaults = WorkspaceLimits::default();
    assert_eq!(defaults.max_ranked_results, 20);
    assert_eq!(defaults.max_index_bytes, 32 * 1024 * 1024);

    let few = open(WorkspaceLimits {
        max_ranked_results: 2,
        ..defaults
    })
    .search("hit")
    .expect("search");
    let order: Vec<_> = few.files.iter().map(|file| file.path.as_str()).collect();
    assert_eq!(order, ["f0.txt", "f1.txt"]);
    assert_eq!(few.total_matches, 5);
    assert_eq!((few.files_indexed, few.bytes_indexed), (5, 20));
    assert_eq!((few.skipped_binary_files, few.skipped_large_files), (1, 1));
    assert_eq!(few.skipped_unreadable_files, 0);
    assert!(few.truncated);
    assert_eq!(few.truncation, Some(Truncation::Results));

    let bytes = open(WorkspaceLimits {
        max_index_bytes: 10,
        ..defaults
    })
    .search("hit")
    .expect("search");
    assert_eq!((bytes.files_indexed, bytes.bytes_indexed), (2, 8));
    assert_eq!(bytes.truncation, Some(Truncation::IndexBytes));

    let files = open(WorkspaceLimits {
        max_index_files: 3,
        ..defaults
    })
    .search("hit")
    .expect("search");
    assert_eq!(files.files_indexed, 3);
    assert_eq!(files.truncation, Some(Truncation::IndexFiles));

    let output = open(WorkspaceLimits {
        max_output_bytes: 30,
        ..defaults
    })
    .search("hit")
    .expect("search");
    assert_eq!(output.files.len(), 1);
    assert_eq!(output.truncation, Some(Truncation::OutputBytes));

    let long_query = open(WorkspaceLimits {
        max_pattern_bytes: 8,
        ..defaults
    })
    .search(&"hit ".repeat(3));
    assert!(matches!(long_query, Err(Error::InvalidArgument(_))));

    write(dir.path(), "long.txt", format!("hit {}\n", "x".repeat(50)));
    let short = open(WorkspaceLimits {
        max_line_bytes: 12,
        ..defaults
    })
    .search("hit")
    .expect("search");
    assert!(short.files.iter().all(|file| file.snippet.len() <= 12));
    assert!(
        short
            .files
            .iter()
            .all(|file| file.snippet.starts_with("hit"))
    );
}
