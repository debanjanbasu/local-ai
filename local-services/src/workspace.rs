//! Deterministic, bounded, read-only access to a repository working tree.
//!
//! A [`Workspace`] is rooted at one directory. Every file whose contents are
//! returned is opened through a capability handle on that directory
//! ([`cap_std::fs::Dir`]), one path component at a time: `..` and absolute
//! paths are rejected lexically, any `.git` component is refused, and a
//! component that is a symlink (or is swapped for one while being opened) is
//! refused rather than followed, so a path can neither leave the root nor
//! reach `.git` through a link.
//!
//! Discovery (listing, search and matching) walks the tree with the `ignore`
//! crate, honouring `.gitignore`, `.ignore` and `.git/info/exclude`, skipping
//! hidden entries, never descending into `.git`, and never following symlinks.
//! The user's global gitignore is deliberately not consulted so results depend
//! only on the repository itself. Discovered paths are re-opened through the
//! capability handle and checked against the inode the walker saw, so a file
//! swapped between discovery and read is skipped.
//!
//! Two kinds of search exist with different semantics:
//! - [`Workspace::search`] is the default ranked search. Each call builds a
//!   fresh, bounded, in-memory [Tantivy](tantivy) index of the visible text
//!   files (`path` and `body` fields, Tantivy's default tokenizer) and ranks
//!   files by BM25 for the query's words, so edits are reflected immediately
//!   and nothing is persisted, cached or left running.
//! - [`Workspace::find_matches`] reports every line matching an exact literal
//!   or regex pattern, in path order.
//!
//! Every operation is synchronous and bounded: the number of walked entries,
//! the size of files scanned or indexed, the bytes returned, the number of
//! results and the compiled regex size are all capped by [`WorkspaceLimits`].
//! Whenever a cap cuts output short the result says so through an explicit
//! `truncated` flag and a [`Truncation`] reason. Output order is fully
//! determined by the tree (path order, or score then path for ranked search).
//!
//! This module never writes to the tree, executes, or spawns processes. The
//! only threads are Tantivy's single indexing worker and segment updater,
//! which are joined before [`Workspace::search`] returns.

use std::collections::BTreeSet;
use std::fs::{File, Metadata};
use std::io::{self, BufRead, BufReader, Read};
use std::ops::ControlFlow;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use cap_std::ambient_authority;
use cap_std::fs::{Dir, MetadataExt as _};
use grep_matcher::Matcher;
use grep_regex::{RegexMatcher, RegexMatcherBuilder};
use grep_searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkMatch};
use ignore::WalkBuilder;
use tantivy::collector::TopDocs;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::{BooleanQuery, BoostQuery, Occur, Query, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions, Value,
};
use tantivy::snippet::SnippetGenerator;
use tantivy::tokenizer::TokenStream;
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument, Term, doc};

use crate::{Error, Result};

/// Bytes inspected up front for a NUL byte before a file is treated as text.
const BINARY_SNIFF_BYTES: usize = 8 * 1024;
/// Memory arena of Tantivy's single indexing thread: its documented minimum
/// (`MEMORY_BUDGET_NUM_BYTES_MIN`, 15 MB). Larger corpora flush extra
/// in-memory segments instead of growing the arena.
const INDEX_WRITER_MEMORY_BYTES: usize = 15_000_000;
/// Characters of context Tantivy picks around the best-matching words.
const SNIPPET_CHARS: usize = 240;
/// Boost for a query that is exactly a file's workspace-relative path.
const EXACT_PATH_BOOST: f32 = 4.0;
/// Fixed per-entry overhead charged against output budgets (separators,
/// line numbers), so many tiny results still exhaust the budget.
const ENTRY_OVERHEAD_BYTES: usize = 16;
/// Regex nesting depth cap; deeper patterns are rejected at compile time.
const REGEX_NEST_LIMIT: u32 = 64;

/// Hard caps applied to every operation on a [`Workspace`].
///
/// Per-call options may lower these but never raise them. All values must be
/// non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceLimits {
    /// Maximum directory entries (files and directories) visited by one walk.
    pub max_walk_entries: usize,
    /// Files larger than this are skipped by `search` and `find_matches`;
    /// `read_file` scans at most this many bytes of a file.
    pub max_file_bytes: u64,
    /// Maximum bytes of line text returned by one `read_file` call.
    pub max_read_bytes: usize,
    /// Maximum entries returned by one `list_files` call.
    pub max_list_entries: usize,
    /// Maximum matches returned by one `find_matches` call.
    pub max_results: usize,
    /// Maximum ranked files returned by one `search` call.
    pub max_ranked_results: usize,
    /// Maximum files indexed by one `search` call.
    pub max_index_files: usize,
    /// Maximum total file bytes indexed by one `search` call.
    pub max_index_bytes: usize,
    /// Maximum bytes of paths and text returned by `list_files`, `search` or
    /// `find_matches`.
    pub max_output_bytes: usize,
    /// Matched lines and snippets longer than this are cut (on a UTF-8
    /// boundary).
    pub max_line_bytes: usize,
    /// Maximum length of a search query or pattern.
    pub max_pattern_bytes: usize,
    /// Approximate cap on the compiled regex program and its DFA cache.
    pub max_regex_bytes: usize,
}

impl Default for WorkspaceLimits {
    fn default() -> Self {
        Self {
            max_walk_entries: 100_000,
            max_file_bytes: 4 * 1024 * 1024,
            max_read_bytes: 256 * 1024,
            max_list_entries: 10_000,
            max_results: 1_000,
            max_ranked_results: 20,
            max_index_files: 20_000,
            max_index_bytes: 32 * 1024 * 1024,
            max_output_bytes: 1024 * 1024,
            max_line_bytes: 4 * 1024,
            max_pattern_bytes: 4 * 1024,
            max_regex_bytes: 1024 * 1024,
        }
    }
}

impl WorkspaceLimits {
    fn validate(&self) -> Result<()> {
        let all_set = [
            self.max_walk_entries,
            self.max_read_bytes,
            self.max_list_entries,
            self.max_results,
            self.max_ranked_results,
            self.max_index_files,
            self.max_index_bytes,
            self.max_output_bytes,
            self.max_line_bytes,
            self.max_pattern_bytes,
            self.max_regex_bytes,
        ]
        .iter()
        .all(|&value| value > 0)
            && self.max_file_bytes > 0;
        if all_set {
            Ok(())
        } else {
            Err(Error::InvalidArgument(
                "workspace limits must all be non-zero".to_owned(),
            ))
        }
    }
}

/// Why an operation returned less than the full answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truncation {
    /// The directory walk hit [`WorkspaceLimits::max_walk_entries`]; files past
    /// that point were not considered.
    WalkEntries,
    /// More files matched than the entry limit allows.
    Entries,
    /// More lines (or ranked files) matched than the result limit allows.
    Results,
    /// [`WorkspaceLimits::max_index_files`] files were indexed; files past
    /// that point (in path order) were not searched.
    IndexFiles,
    /// The next file would exceed [`WorkspaceLimits::max_index_bytes`]; it and
    /// the files after it (in path order) were not searched.
    IndexBytes,
    /// The output byte budget ran out.
    OutputBytes,
    /// The file is larger than [`WorkspaceLimits::max_file_bytes`]; content
    /// past that point was not scanned.
    FileBytes,
}

/// Options for [`Workspace::list_files`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ListOptions {
    /// Workspace-relative directory (or file) to list under; `None`, `""` or
    /// `"."` mean the whole workspace.
    pub path: Option<String>,
    /// Lower the entry limit for this call.
    pub max_entries: Option<usize>,
    /// Lower the output byte budget for this call.
    pub max_output_bytes: Option<usize>,
}

/// One regular file found by [`Workspace::list_files`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// Workspace-relative path using `/` separators.
    pub path: String,
    /// Size in bytes at discovery time.
    pub size: u64,
}

/// Result of [`Workspace::list_files`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileList {
    /// Files sorted by path.
    pub files: Vec<FileEntry>,
    /// `true` when `files` is not the complete set.
    pub truncated: bool,
    /// The first limit that cut the result short.
    pub truncation: Option<Truncation>,
}

/// Options for [`Workspace::read_file`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadOptions {
    /// First line to return, 1-based. Must be at least 1.
    pub start_line: u64,
    /// Last line to return, inclusive; `None` reads to the end (within limits).
    pub end_line: Option<u64>,
    /// Lower the returned-bytes limit for this call.
    pub max_bytes: Option<usize>,
    /// Explicitly allow reading a gitignored or hidden path that discovery
    /// would not surface. Symlinks are still never followed, and the path
    /// still may not leave the root or name a `.git` component.
    pub include_ignored: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            start_line: 1,
            end_line: None,
            max_bytes: None,
            include_ignored: false,
        }
    }
}

/// One line returned by [`Workspace::read_file`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileLine {
    /// 1-based line number.
    pub number: u64,
    /// Line text without its terminator; invalid UTF-8 is replaced with
    /// U+FFFD.
    pub text: String,
    /// `true` when the line was cut to fit the byte budget.
    pub truncated: bool,
}

/// Result of [`Workspace::read_file`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRead {
    /// Normalised workspace-relative path using `/` separators.
    pub path: String,
    /// The requested lines that fit, in order.
    pub lines: Vec<FileLine>,
    /// The next unread line, when the file is known to continue past the
    /// returned lines. `None` at end of file or when scanning stopped at
    /// [`WorkspaceLimits::max_file_bytes`].
    pub next_line: Option<u64>,
    /// `true` when the end of the file was reached.
    pub eof: bool,
    /// `true` when the requested range was not returned in full.
    pub truncated: bool,
    /// The limit that cut the result short.
    pub truncation: Option<Truncation>,
    /// `true` when any returned line contained invalid UTF-8.
    pub invalid_utf8: bool,
}

/// How [`SearchOptions::pattern`] is interpreted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PatternMode {
    /// Match the pattern text exactly.
    #[default]
    Literal,
    /// Rust `regex` syntax, matched within single lines.
    Regex,
}

/// Options for [`Workspace::find_matches`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchOptions {
    /// Non-empty pattern; matched line by line, never across lines.
    pub pattern: String,
    /// Literal or regex interpretation.
    pub mode: PatternMode,
    /// Match letters regardless of case.
    pub case_insensitive: bool,
    /// Workspace-relative directory (or file) to search under; `None`, `""` or
    /// `"."` mean the whole workspace.
    pub path: Option<String>,
    /// Lower the result limit for this call.
    pub max_results: Option<usize>,
    /// Lower the output byte budget for this call.
    pub max_output_bytes: Option<usize>,
}

impl SearchOptions {
    /// A case-sensitive literal search over the whole workspace.
    #[must_use]
    pub fn literal(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            mode: PatternMode::Literal,
            ..Self::default()
        }
    }

    /// A case-sensitive regex search over the whole workspace.
    #[must_use]
    pub fn regex(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            mode: PatternMode::Regex,
            ..Self::default()
        }
    }
}

/// One matching line found by [`Workspace::find_matches`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchMatch {
    /// Workspace-relative path using `/` separators.
    pub path: String,
    /// 1-based line number.
    pub line_number: u64,
    /// 1-based byte column of the first match on the line.
    pub column: u64,
    /// Line text without its terminator; invalid UTF-8 is replaced with
    /// U+FFFD.
    pub line: String,
    /// `true` when `line` was cut to [`WorkspaceLimits::max_line_bytes`].
    pub line_truncated: bool,
}

/// Result of [`Workspace::find_matches`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResults {
    /// Matches sorted by path, then line number.
    pub matches: Vec<SearchMatch>,
    /// Text files fully or partially searched.
    pub files_searched: usize,
    /// Files skipped because they contain a NUL byte.
    pub skipped_binary_files: usize,
    /// Files skipped because they exceed [`WorkspaceLimits::max_file_bytes`].
    pub skipped_large_files: usize,
    /// Files skipped because they could not be opened, changed after
    /// discovery, or failed to read.
    pub skipped_unreadable_files: usize,
    /// `true` when `matches` is not the complete set.
    pub truncated: bool,
    /// The first limit that cut the result short.
    pub truncation: Option<Truncation>,
}

/// One file ranked by [`Workspace::search`].
#[derive(Debug, Clone, PartialEq)]
pub struct RankedFile {
    /// Workspace-relative path using `/` separators.
    pub path: String,
    /// Tantivy BM25 relevance; higher is better. Only comparable within one
    /// result set.
    pub score: f32,
    /// 1-based line on which `snippet` starts.
    pub line_number: u64,
    /// Bounded excerpt around the best-matching words, or the first non-blank
    /// line when only the path matched. Invalid UTF-8 is replaced with U+FFFD.
    pub snippet: String,
}

/// Result of [`Workspace::search`].
#[derive(Debug, Clone, PartialEq)]
pub struct RankedResults {
    /// Best files first; equal scores are ordered by path.
    pub files: Vec<RankedFile>,
    /// Indexed files matching at least one query word (or the exact path).
    pub total_matches: usize,
    /// Text files indexed for this query.
    pub files_indexed: usize,
    /// Bytes of file content indexed.
    pub bytes_indexed: u64,
    /// Files skipped because they contain a NUL byte.
    pub skipped_binary_files: usize,
    /// Files skipped because they exceed [`WorkspaceLimits::max_file_bytes`].
    pub skipped_large_files: usize,
    /// Files skipped because they could not be opened, changed after
    /// discovery, or failed to read.
    pub skipped_unreadable_files: usize,
    /// `true` when `files` is not the complete ranking of every visible file.
    pub truncated: bool,
    /// The first limit that cut the result short.
    pub truncation: Option<Truncation>,
}

/// A read-only, capability-rooted view of a repository working tree.
#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
    dir: Dir,
    limits: WorkspaceLimits,
}

/// A regular file seen by the discovery walk.
struct Candidate {
    rel: PathBuf,
    display: String,
    size: u64,
    ino: Option<u64>,
}

impl Workspace {
    /// Open the directory at `root` with [`WorkspaceLimits::default`].
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(root, WorkspaceLimits::default())
    }

    /// Open the directory at `root` with explicit limits.
    pub fn open_with_limits(root: impl AsRef<Path>, limits: WorkspaceLimits) -> Result<Self> {
        limits.validate()?;
        let root = root.as_ref();
        let root = std::fs::canonicalize(root).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                Error::NotFound(format!(
                    "workspace root `{}` does not exist",
                    root.display()
                ))
            } else {
                Error::Io(error)
            }
        })?;
        if !root.is_dir() {
            return Err(Error::InvalidArgument(format!(
                "workspace root `{}` is not a directory",
                root.display()
            )));
        }
        let dir = Dir::open_ambient_dir(&root, ambient_authority())?;
        Ok(Self { root, dir, limits })
    }

    /// The canonical root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The hard limits in force.
    #[must_use]
    pub const fn limits(&self) -> &WorkspaceLimits {
        &self.limits
    }

    /// List regular files visible to discovery, sorted by path.
    ///
    /// Ignored, hidden and `.git` entries and symlinks are omitted.
    pub fn list_files(&self, options: &ListOptions) -> Result<FileList> {
        let max_entries = clamp(
            options.max_entries,
            self.limits.max_list_entries,
            "max_entries",
        )?;
        let max_output = clamp(
            options.max_output_bytes,
            self.limits.max_output_bytes,
            "max_output_bytes",
        )?;
        let scope = self.scope(options.path.as_deref())?;
        let (mut candidates, walk_truncated) = self.candidates(&scope);
        candidates.sort_by(|a, b| a.display.cmp(&b.display));

        let mut truncation = walk_truncated.then_some(Truncation::WalkEntries);
        let mut files = Vec::new();
        let mut used = 0usize;
        for candidate in candidates {
            if files.len() == max_entries {
                truncation.get_or_insert(Truncation::Entries);
                break;
            }
            let cost = candidate.display.len() + ENTRY_OVERHEAD_BYTES;
            if used + cost > max_output {
                truncation.get_or_insert(Truncation::OutputBytes);
                break;
            }
            used += cost;
            files.push(FileEntry {
                path: candidate.display,
                size: candidate.size,
            });
        }
        Ok(FileList {
            files,
            truncated: truncation.is_some(),
            truncation,
        })
    }

    /// Read a 1-based, inclusive line range of one text file.
    ///
    /// Paths that discovery would not surface (gitignored or hidden) are
    /// refused unless [`ReadOptions::include_ignored`] is set. A path with a
    /// symlink in any component is always refused. Binary files (containing a
    /// NUL byte in the scanned content) are refused.
    pub fn read_file(&self, path: &str, options: &ReadOptions) -> Result<FileRead> {
        if options.start_line == 0 {
            return Err(Error::InvalidArgument(
                "start_line is 1-based and must be at least 1".to_owned(),
            ));
        }
        if options.end_line.is_some_and(|end| end < options.start_line) {
            return Err(Error::InvalidArgument(
                "end_line must not be before start_line".to_owned(),
            ));
        }
        let max_bytes = clamp(options.max_bytes, self.limits.max_read_bytes, "max_bytes")?;
        let rel = relative_path(path, false)?;
        let display = display_path(&rel);

        let (file, metadata) = self.open_file(&rel, &display)?;
        if !options.include_ignored {
            self.check_discoverable(&rel, &display, &metadata)?;
        }

        let mut reader = BufReader::with_capacity(64 * 1024, file.take(self.limits.max_file_bytes));
        if reader
            .fill_buf()?
            .iter()
            .take(BINARY_SNIFF_BYTES)
            .any(|&b| b == 0)
        {
            return Err(binary_error(&display));
        }
        let mut result = read_lines(&mut reader, options, max_bytes, &display)?;
        if result.eof && metadata.len() > self.limits.max_file_bytes {
            // Scanning stopped at the byte cap, not at the real end of file.
            result.eof = false;
            result.truncation.get_or_insert(Truncation::FileBytes);
        }
        result.truncated = result.truncation.is_some();
        result.path = display;
        Ok(result)
    }

    /// Rank visible text files by relevance to a free-text `query`.
    ///
    /// This is the default search. Each call indexes the current visible text
    /// files in path order into a fresh in-memory Tantivy index (so edits,
    /// deletions and new files are always reflected), within
    /// [`WorkspaceLimits::max_index_files`] and
    /// [`WorkspaceLimits::max_index_bytes`]. The query is never parsed as
    /// query syntax: it is split into words by the same tokenizer as the
    /// index (`EngineHandle::decide` becomes `enginehandle` and `decide`,
    /// `max_walk_entries` becomes `max`, `walk` and `entries`, all
    /// lowercased), and a file matches when its path or body contains any
    /// word. Files are scored with BM25; a query equal to a file's path also
    /// boosts that file. At most [`WorkspaceLimits::max_ranked_results`] files
    /// are returned, best first, ties ordered by path.
    ///
    /// Binary files, files over the size cap, ignored, hidden and `.git`
    /// entries and symlinks are never indexed.
    pub fn search(&self, query: &str) -> Result<RankedResults> {
        let query = query.trim();
        self.check_pattern(query, "search query")?;
        let rank = RankIndex::new();
        let words = rank.words(query)?;
        if words.is_empty() {
            return Err(Error::InvalidArgument(
                "search query contains no searchable words".to_owned(),
            ));
        }
        let mut results = RankedResults {
            files: Vec::new(),
            total_matches: 0,
            files_indexed: 0,
            bytes_indexed: 0,
            skipped_binary_files: 0,
            skipped_large_files: 0,
            skipped_unreadable_files: 0,
            truncated: false,
            truncation: None,
        };
        let docs = self.index_files(&rank, &mut results)?;
        let query = rank.query(query, &words);
        let searcher = rank
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(index_error)?
            .searcher();
        let hits = if docs.is_empty() {
            Vec::new()
        } else {
            let top = TopDocs::with_limit(docs.len()).order_by_score();
            searcher.search(&query, &top).map_err(index_error)?
        };
        let mut ranked = Vec::with_capacity(hits.len());
        for (score, address) in hits {
            let doc: TantivyDocument = searcher.doc(address).map_err(index_error)?;
            let ord = doc.get_first(rank.ord).and_then(|value| value.as_u64());
            if let Some(ord) = ord.and_then(|ord| usize::try_from(ord).ok()) {
                ranked.push((score, ord));
            }
        }
        // Documents were added in path order, so `ord` breaks ties by path.
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        results.total_matches = ranked.len();

        let mut snippets =
            SnippetGenerator::create(&searcher, &query, rank.body).map_err(index_error)?;
        snippets.set_max_num_chars(SNIPPET_CHARS);
        let mut used = 0usize;
        for (score, ord) in ranked {
            if results.files.len() == self.limits.max_ranked_results {
                results.truncation.get_or_insert(Truncation::Results);
                break;
            }
            let Some((path, text)) = docs.get(ord) else {
                continue;
            };
            let (line_number, snippet) = self.snippet(&snippets, text);
            let cost = path.len() + snippet.len() + ENTRY_OVERHEAD_BYTES;
            if used + cost > self.limits.max_output_bytes {
                results.truncation.get_or_insert(Truncation::OutputBytes);
                break;
            }
            used += cost;
            results.files.push(RankedFile {
                path: path.clone(),
                score,
                line_number,
                snippet,
            });
        }
        results.truncated = results.truncation.is_some();
        Ok(results)
    }

    /// Index the visible text files in path order; returns each indexed
    /// document's `(path, text)`, positioned by its `ord` field.
    fn index_files(
        &self,
        rank: &RankIndex,
        results: &mut RankedResults,
    ) -> Result<Vec<(String, String)>> {
        let mut writer: IndexWriter = rank
            .index
            .writer_with_num_threads(1, INDEX_WRITER_MEMORY_BYTES)
            .map_err(index_error)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let (mut candidates, walk_truncated) = self.candidates(Path::new(""));
        candidates.sort_by(|a, b| a.display.cmp(&b.display));
        results.truncation = walk_truncated.then_some(Truncation::WalkEntries);

        let max_bytes = self.limits.max_index_bytes as u64;
        let mut docs = Vec::new();
        for candidate in candidates {
            if docs.len() == self.limits.max_index_files {
                results.truncation.get_or_insert(Truncation::IndexFiles);
                break;
            }
            if candidate.size > self.limits.max_file_bytes {
                results.skipped_large_files += 1;
                continue;
            }
            let remaining = max_bytes - results.bytes_indexed;
            if candidate.size > remaining {
                results.truncation.get_or_insert(Truncation::IndexBytes);
                break;
            }
            let Some(bytes) = self.load(&candidate, remaining) else {
                results.skipped_unreadable_files += 1;
                continue;
            };
            let len = bytes.len() as u64;
            if len > self.limits.max_file_bytes {
                results.skipped_large_files += 1;
                continue;
            }
            if len > remaining {
                // The file grew since discovery past the remaining budget.
                results.truncation.get_or_insert(Truncation::IndexBytes);
                break;
            }
            if bytes.contains(&0) {
                results.skipped_binary_files += 1;
                continue;
            }
            let text = String::from_utf8_lossy(&bytes).into_owned();
            writer
                .add_document(doc!(
                    rank.path => candidate.display.as_str(),
                    rank.path_key => candidate.display.as_str(),
                    rank.body => text.as_str(),
                    rank.ord => docs.len() as u64,
                ))
                .map_err(index_error)?;
            results.bytes_indexed += len;
            docs.push((candidate.display, text));
        }
        results.files_indexed = docs.len();
        writer.commit().map_err(index_error)?;
        writer.wait_merging_threads().map_err(index_error)?;
        Ok(docs)
    }

    /// Read at most `min(max_file_bytes, budget) + 1` bytes of a discovered
    /// file, so the caller can tell an over-limit file from one that fits.
    fn load(&self, candidate: &Candidate, budget: u64) -> Option<Vec<u8>> {
        let file = self.open_candidate(candidate)?;
        let cap = self.limits.max_file_bytes.min(budget).saturating_add(1);
        let mut bytes = Vec::new();
        file.take(cap).read_to_end(&mut bytes).ok()?;
        Some(bytes)
    }

    /// The best excerpt of `text` for the query and the line it starts on.
    fn snippet(&self, generator: &SnippetGenerator, text: &str) -> (u64, String) {
        let snippet = generator.snippet(text);
        let fragment = snippet.fragment().trim();
        let (offset, fragment) = match text.find(fragment) {
            Some(offset) if !fragment.is_empty() => (offset, fragment),
            // Only the path matched: show the first non-blank line instead.
            _ => {
                let line = text.lines().find(|line| !line.trim().is_empty());
                let line = line.unwrap_or("").trim();
                (text.find(line).unwrap_or(0), line)
            }
        };
        let limit = self.limits.max_line_bytes.min(SNIPPET_CHARS * 4);
        let (snippet, _) = lossy_bounded(fragment.as_bytes(), limit);
        let before = text.get(..offset).unwrap_or_default();
        let line_number = before.matches('\n').count() as u64 + 1;
        (line_number, snippet)
    }

    /// Search visible text files line by line for an exact literal or regex
    /// pattern, in path order.
    ///
    /// Binary files, files over the size cap, ignored, hidden and `.git`
    /// entries and symlinks are never searched.
    pub fn find_matches(&self, options: &SearchOptions) -> Result<SearchResults> {
        let max_results = clamp(options.max_results, self.limits.max_results, "max_results")?;
        let max_output = clamp(
            options.max_output_bytes,
            self.limits.max_output_bytes,
            "max_output_bytes",
        )?;
        let matcher = self.matcher(options)?;
        let scope = self.scope(options.path.as_deref())?;
        let (mut candidates, walk_truncated) = self.candidates(&scope);
        candidates.sort_by(|a, b| a.display.cmp(&b.display));

        let heap_limit = usize::try_from(self.limits.max_file_bytes)
            .unwrap_or(usize::MAX)
            .saturating_add(64 * 1024);
        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .binary_detection(BinaryDetection::quit(0))
            .bom_sniffing(false)
            .heap_limit(Some(heap_limit))
            .build();

        let mut results = SearchResults {
            matches: Vec::new(),
            files_searched: 0,
            skipped_binary_files: 0,
            skipped_large_files: 0,
            skipped_unreadable_files: 0,
            truncated: false,
            truncation: walk_truncated.then_some(Truncation::WalkEntries),
        };
        let mut used = 0usize;
        for candidate in &candidates {
            if candidate.size > self.limits.max_file_bytes {
                results.skipped_large_files += 1;
                continue;
            }
            let remaining = max_results - results.matches.len();
            let Some(found) = self.search_file(&mut searcher, &matcher, candidate, remaining + 1)
            else {
                results.skipped_unreadable_files += 1;
                continue;
            };
            let found = match found {
                FileOutcome::Large => {
                    results.skipped_large_files += 1;
                    continue;
                }
                FileOutcome::Binary => {
                    results.skipped_binary_files += 1;
                    continue;
                }
                FileOutcome::Text(found) => found,
            };
            results.files_searched += 1;
            if self
                .collect(
                    &mut results,
                    found,
                    candidate,
                    max_results,
                    max_output,
                    &mut used,
                )
                .is_break()
            {
                break;
            }
        }
        results.truncated = results.truncation.is_some();
        Ok(results)
    }

    /// Append one file's matches, stopping at the first exhausted budget.
    fn collect(
        &self,
        results: &mut SearchResults,
        found: Vec<RawMatch>,
        candidate: &Candidate,
        max_results: usize,
        max_output: usize,
        used: &mut usize,
    ) -> ControlFlow<()> {
        for raw in found {
            if results.matches.len() == max_results {
                results.truncation.get_or_insert(Truncation::Results);
                return ControlFlow::Break(());
            }
            let (line, line_truncated) = lossy_bounded(&raw.line, self.limits.max_line_bytes);
            let cost = candidate.display.len() + line.len() + ENTRY_OVERHEAD_BYTES;
            if *used + cost > max_output {
                results.truncation.get_or_insert(Truncation::OutputBytes);
                return ControlFlow::Break(());
            }
            *used += cost;
            results.matches.push(SearchMatch {
                path: candidate.display.clone(),
                line_number: raw.line_number,
                column: raw.column,
                line,
                line_truncated,
            });
        }
        ControlFlow::Continue(())
    }

    /// Search one discovered file; `None` when it cannot be read safely.
    fn search_file(
        &self,
        searcher: &mut Searcher,
        matcher: &RegexMatcher,
        candidate: &Candidate,
        limit: usize,
    ) -> Option<FileOutcome> {
        let file = self.open_candidate(candidate)?;
        if file.metadata().ok()?.len() > self.limits.max_file_bytes {
            return Some(FileOutcome::Large);
        }
        let mut sink = FileSink {
            matcher,
            matches: Vec::new(),
            binary: false,
            limit,
        };
        searcher
            .search_reader(matcher, file.take(self.limits.max_file_bytes), &mut sink)
            .ok()?;
        Some(if sink.binary {
            FileOutcome::Binary
        } else {
            FileOutcome::Text(sink.matches)
        })
    }

    /// Open a discovered file without following symlinks, confirming it is
    /// still the regular file the walker saw; `None` when it is not.
    fn open_candidate(&self, candidate: &Candidate) -> Option<File> {
        let (file, metadata) = self.open_file(&candidate.rel, &candidate.display).ok()?;
        candidate
            .ino
            .is_none_or(|ino| ino == metadata.ino())
            .then_some(file)
    }

    /// Open a regular file under the root without following a symlink at any
    /// component. Each component is inspected with `lstat`, opened, and
    /// checked to be the same inode, so one swapped for a symlink between the
    /// two steps is refused rather than followed.
    fn open_file(&self, rel: &Path, display: &str) -> Result<(File, Metadata)> {
        let changed = || Error::Conflict(format!("`{display}` changed while it was being opened"));
        let mut parent: Option<Dir> = None;
        let mut names = rel.iter().peekable();
        while let Some(name) = names.next() {
            let dir = parent.as_ref().unwrap_or(&self.dir);
            let seen = dir
                .symlink_metadata(name)
                .map_err(|error| open_error(display, error))?;
            if seen.file_type().is_symlink() {
                return Err(Error::InvalidArgument(format!(
                    "`{display}` is reached through a symlink, which is never followed"
                )));
            }
            if names.peek().is_some() {
                let next = dir
                    .open_dir(name)
                    .map_err(|error| open_error(display, error))?;
                let opened = next.dir_metadata()?;
                if (opened.dev(), opened.ino()) != (seen.dev(), seen.ino()) {
                    return Err(changed());
                }
                parent = Some(next);
                continue;
            }
            if !seen.is_file() {
                return Err(Error::InvalidArgument(format!(
                    "`{display}` is not a regular file"
                )));
            }
            let file = dir
                .open(name)
                .map_err(|error| open_error(display, error))?
                .into_std();
            let metadata = file.metadata()?;
            if !metadata.is_file() || (metadata.dev(), metadata.ino()) != (seen.dev(), seen.ino()) {
                return Err(changed());
            }
            return Ok((file, metadata));
        }
        Err(Error::InvalidArgument("path must name a file".to_owned()))
    }

    /// Validate a search query or pattern: non-empty, within the length limit,
    /// on one line.
    fn check_pattern(&self, pattern: &str, what: &str) -> Result<()> {
        if pattern.is_empty() {
            return Err(Error::InvalidArgument(format!("{what} must not be empty")));
        }
        if pattern.len() > self.limits.max_pattern_bytes {
            return Err(Error::InvalidArgument(format!(
                "{what} is {} bytes; the limit is {}",
                pattern.len(),
                self.limits.max_pattern_bytes
            )));
        }
        if pattern.contains(['\n', '\r', '\0']) {
            return Err(Error::InvalidArgument(format!(
                "{what} must not contain line terminators or NUL bytes"
            )));
        }
        Ok(())
    }

    fn matcher(&self, options: &SearchOptions) -> Result<RegexMatcher> {
        self.check_pattern(&options.pattern, "search pattern")?;
        RegexMatcherBuilder::new()
            .case_insensitive(options.case_insensitive)
            .fixed_strings(options.mode == PatternMode::Literal)
            .line_terminator(Some(b'\n'))
            .size_limit(self.limits.max_regex_bytes)
            .dfa_size_limit(self.limits.max_regex_bytes)
            .nest_limit(REGEX_NEST_LIMIT)
            .build(&options.pattern)
            .map_err(|error| Error::InvalidArgument(format!("invalid search pattern: {error}")))
    }

    /// Validate an optional scope path and confirm it exists inside the root.
    fn scope(&self, path: Option<&str>) -> Result<PathBuf> {
        let rel = relative_path(path.unwrap_or(""), true)?;
        if !rel.as_os_str().is_empty() {
            self.dir
                .metadata(&rel)
                .map_err(|error| open_error(&display_path(&rel), error))?;
        }
        Ok(rel)
    }

    /// Regular files under `scope`, in walk order, plus whether the walk was
    /// cut short.
    fn candidates(&self, scope: &Path) -> (Vec<Candidate>, bool) {
        let mut candidates = Vec::new();
        let truncated = self.walk(scope, |entry, rel| {
            if entry.file_type().is_some_and(|kind| kind.is_file()) {
                let size = entry.metadata().map_or(0, |metadata| metadata.len());
                candidates.push(Candidate {
                    display: display_path(&rel),
                    rel,
                    size,
                    ino: entry.ino(),
                });
            }
            ControlFlow::Continue(())
        });
        (candidates, truncated)
    }

    /// Refuse a path that the discovery walk would not surface as a regular
    /// file, or whose inode differs from the one just opened.
    fn check_discoverable(&self, rel: &Path, display: &str, opened: &Metadata) -> Result<()> {
        let mut seen = None;
        self.walk(rel, |entry, entry_rel| {
            if entry_rel == rel {
                seen = Some((
                    entry.file_type().is_some_and(|kind| kind.is_file()),
                    entry.ino(),
                ));
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        });
        match seen {
            Some((true, ino)) if ino.is_none_or(|ino| ino == opened.ino()) => Ok(()),
            Some((true, _)) => Err(Error::Conflict(format!(
                "`{display}` changed while it was being opened"
            ))),
            _ => Err(Error::InvalidArgument(format!(
                "`{display}` is ignored, hidden, or reached through a symlink; \
                 set include_ignored to read it explicitly"
            ))),
        }
    }

    /// Walk the visible tree, restricted to `scope` and its ancestors, in a
    /// deterministic order. Returns `true` when the entry limit cut the walk
    /// short.
    fn walk(
        &self,
        scope: &Path,
        mut visit: impl FnMut(&ignore::DirEntry, PathBuf) -> ControlFlow<()>,
    ) -> bool {
        let scope_abs = self.root.join(scope);
        let mut builder = WalkBuilder::new(&self.root);
        builder
            .standard_filters(true)
            .hidden(true)
            .parents(false)
            .ignore(true)
            .git_ignore(true)
            .git_exclude(true)
            .git_global(false)
            .require_git(false)
            .follow_links(false)
            .sort_by_file_name(std::cmp::Ord::cmp)
            .filter_entry(move |entry| {
                let path = entry.path();
                !is_git_dir_name(entry.file_name())
                    && (path.starts_with(&scope_abs) || scope_abs.starts_with(path))
            });
        let mut visited = 0usize;
        for entry in builder.build() {
            // Unreadable directories and malformed ignore files are skipped so
            // one bad entry does not fail the whole operation.
            let Ok(entry) = entry else { continue };
            if entry.depth() == 0 {
                continue;
            }
            visited += 1;
            if visited > self.limits.max_walk_entries {
                return true;
            }
            let Some(rel) = entry
                .path()
                .strip_prefix(&self.root)
                .ok()
                .and_then(utf8_relative)
            else {
                continue;
            };
            if visit(&entry, rel).is_break() {
                break;
            }
        }
        false
    }
}

/// A fresh in-memory Tantivy index for one ranked search.
struct RankIndex {
    index: Index,
    /// Workspace-relative path, tokenized like the body.
    path: Field,
    /// The exact path, untokenized, for exact-path queries.
    path_key: Field,
    /// File text, tokenized; frequencies but no positions (BM25 only).
    body: Field,
    /// Position of the document in path order; the only stored value.
    ord: Field,
}

impl RankIndex {
    fn new() -> Self {
        let words = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("default")
                .set_index_option(IndexRecordOption::WithFreqs),
        );
        let mut schema = Schema::builder();
        let path = schema.add_text_field("path", words.clone());
        let path_key = schema.add_text_field("path_key", STRING);
        let body = schema.add_text_field("body", words);
        let ord = schema.add_u64_field("ord", STORED);
        Self {
            index: Index::create_in_ram(schema.build()),
            path,
            path_key,
            body,
            ord,
        }
    }

    /// The distinct words of `text` under the index tokenizer, sorted.
    fn words(&self, text: &str) -> Result<BTreeSet<String>> {
        let mut analyzer = self
            .index
            .tokenizer_for_field(self.body)
            .map_err(index_error)?;
        let mut stream = analyzer.token_stream(text);
        let mut words = BTreeSet::new();
        while stream.advance() {
            words.insert(stream.token().text.clone());
        }
        Ok(words)
    }

    /// Any word in the path or body, plus a boosted exact-path match.
    fn query(&self, text: &str, words: &BTreeSet<String>) -> BooleanQuery {
        let exact = TermQuery::new(
            Term::from_field_text(self.path_key, text.trim_start_matches("./")),
            IndexRecordOption::Basic,
        );
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = vec![(
            Occur::Should,
            Box::new(BoostQuery::new(Box::new(exact), EXACT_PATH_BOOST)),
        )];
        for word in words {
            for field in [self.path, self.body] {
                let term = Term::from_field_text(field, word);
                let query = TermQuery::new(term, IndexRecordOption::WithFreqs);
                clauses.push((Occur::Should, Box::new(query)));
            }
        }
        BooleanQuery::new(clauses)
    }
}

fn index_error(error: tantivy::TantivyError) -> Error {
    Error::Io(io::Error::other(error))
}

/// Outcome of searching one file.
enum FileOutcome {
    Text(Vec<RawMatch>),
    Binary,
    Large,
}

struct RawMatch {
    line_number: u64,
    column: u64,
    line: Vec<u8>,
}

/// Collects one file's matches; a NUL byte anywhere searched marks the file
/// binary and its matches are discarded by the caller.
struct FileSink<'a> {
    matcher: &'a RegexMatcher,
    matches: Vec<RawMatch>,
    binary: bool,
    limit: usize,
}

impl Sink for FileSink<'_> {
    type Error = io::Error;

    fn matched(&mut self, _searcher: &Searcher, mat: &SinkMatch<'_>) -> io::Result<bool> {
        let line = trim_terminator(mat.bytes());
        let start = self
            .matcher
            .find(line)
            .ok()
            .flatten()
            .map_or(0, |found| found.start());
        self.matches.push(RawMatch {
            line_number: mat.line_number().unwrap_or(0),
            column: start as u64 + 1,
            line: line.to_vec(),
        });
        Ok(self.matches.len() < self.limit)
    }

    fn binary_data(&mut self, _searcher: &Searcher, _offset: u64) -> io::Result<bool> {
        self.binary = true;
        Ok(false)
    }
}

/// Stream lines from `reader`, keeping the requested range within `max_bytes`.
fn read_lines(
    reader: &mut impl BufRead,
    options: &ReadOptions,
    max_bytes: usize,
    display: &str,
) -> Result<FileRead> {
    let mut result = FileRead {
        path: String::new(),
        lines: Vec::new(),
        next_line: None,
        eof: false,
        truncated: false,
        truncation: None,
        invalid_utf8: false,
    };
    let mut buf = Vec::new();
    let mut number = 0u64;
    let mut used = 0usize;
    loop {
        buf.clear();
        if reader.read_until(b'\n', &mut buf)? == 0 {
            result.eof = true;
            return Ok(result);
        }
        if buf.contains(&0) {
            return Err(binary_error(display));
        }
        number += 1;
        if number < options.start_line {
            continue;
        }
        if options.end_line.is_some_and(|end| number > end) {
            result.next_line = Some(number);
            return Ok(result);
        }
        let raw = trim_terminator(&buf);
        let remaining = max_bytes - used;
        if raw.len() + 1 > remaining && !result.lines.is_empty() {
            result.next_line = Some(number);
            result.truncation = Some(Truncation::OutputBytes);
            return Ok(result);
        }
        let (text, cut) = lossy_bounded(raw, remaining);
        result.invalid_utf8 |= std::str::from_utf8(raw).is_err();
        used += text.len() + 1;
        result.lines.push(FileLine {
            number,
            text,
            truncated: cut,
        });
        if cut {
            result.next_line = Some(number + 1);
            result.truncation = Some(Truncation::OutputBytes);
            return Ok(result);
        }
    }
}

/// Lossily decode `bytes`, cutting the result to at most `max` bytes on a
/// character boundary. Returns whether anything was cut.
fn lossy_bounded(bytes: &[u8], max: usize) -> (String, bool) {
    let mut text = String::from_utf8_lossy(bytes).into_owned();
    if text.len() <= max {
        return (text, false);
    }
    let cut = text
        .char_indices()
        .map(|(index, _)| index)
        .take_while(|&index| index <= max)
        .last()
        .unwrap_or(0);
    text.truncate(cut);
    (text, true)
}

fn trim_terminator(line: &[u8]) -> &[u8] {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    line.strip_suffix(b"\r").unwrap_or(line)
}

/// Lower a hard limit with an optional per-call request, rejecting zero.
fn clamp(requested: Option<usize>, hard: usize, name: &str) -> Result<usize> {
    match requested {
        Some(0) => Err(Error::InvalidArgument(format!("{name} must be at least 1"))),
        Some(value) => Ok(value.min(hard)),
        None => Ok(hard),
    }
}

/// Normalise a caller-supplied workspace-relative path, rejecting absolute
/// paths, `..`, NUL bytes and any `.git` component.
fn relative_path(input: &str, allow_root: bool) -> Result<PathBuf> {
    if input.contains('\0') {
        return Err(Error::InvalidArgument(
            "path must not contain NUL bytes".to_owned(),
        ));
    }
    let mut rel = PathBuf::new();
    for component in Path::new(input).components() {
        match component {
            Component::Normal(part) if is_git_dir_name(part) => {
                return Err(Error::InvalidArgument(format!(
                    "`{input}` is inside .git, which is never readable"
                )));
            }
            Component::Normal(part) => rel.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(Error::InvalidArgument(format!(
                    "`{input}` must be a relative path that stays inside the workspace"
                )));
            }
        }
    }
    if rel.as_os_str().is_empty() && !allow_root {
        return Err(Error::InvalidArgument("path must name a file".to_owned()));
    }
    Ok(rel)
}

/// `.git` compared case-insensitively: macOS file systems usually are.
fn is_git_dir_name(name: &std::ffi::OsStr) -> bool {
    name.as_encoded_bytes().eq_ignore_ascii_case(b".git")
}

/// Re-check a walker-relative path is plain UTF-8 components; names that are
/// not UTF-8 cannot round-trip through the string API and are skipped.
fn utf8_relative(path: &Path) -> Option<PathBuf> {
    path.components()
        .map(|component| match component {
            Component::Normal(part) => part.to_str().map(|_| part),
            _ => None,
        })
        .collect::<Option<PathBuf>>()
}

fn display_path(rel: &Path) -> String {
    rel.components()
        .filter_map(|component| match component {
            Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn open_error(display: &str, error: io::Error) -> Error {
    match error.kind() {
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory => {
            Error::NotFound(format!("`{display}` does not exist in the workspace"))
        }
        // cap-std reports a path that resolves outside the root as
        // PermissionDenied.
        io::ErrorKind::PermissionDenied => Error::InvalidArgument(format!(
            "`{display}` cannot be opened inside the workspace root: {error}"
        )),
        _ => Error::Io(error),
    }
}

fn binary_error(display: &str) -> Error {
    Error::InvalidArgument(format!("`{display}` is a binary file"))
}
