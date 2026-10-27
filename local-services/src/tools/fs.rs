//! Filesystem tools inside the authorized root.
//!
//! Reading, listing and searching go through the root's
//! [`Workspace`](crate::workspace::Workspace), which refuses `..`, absolute
//! paths, `.git` and symlinks and bounds every walk and result. Writes and
//! edits resolve their path the same way, one component at a time through a
//! capability handle on the root: every existing component is checked with
//! `lstat` and refused if it is a symlink, a missing parent directory is
//! created, and the opened directory must be the inode that was checked.
//! `cap-std` itself never resolves a path outside the root, so even a
//! component swapped for a symlink between the check and the open stays
//! inside it. The file itself is opened with `O_NOFOLLOW` and checked
//! through the open handle (a regular file with one link, the inode that was
//! checked), which is then truncated, read and written; it is never reopened
//! by name.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{ErrorKind, Read as _};
use std::os::unix::fs::{FileExt as _, MetadataExt as _};
use std::path::Path;

use cap_std::ambient_authority;
use cap_std::fs::{Dir, MetadataExt as _};
use rustix::fs::{Mode, OFlags};
use rustix::io::Errno;

use super::{Inner, Output, Request, SearchMode, ToolError, truncate_marked};
use crate::Error;
use crate::workspace::{
    ListOptions, PatternMode, ReadOptions, SearchOptions, Truncation, relative_path,
};

/// Run a filesystem request. Blocking.
// One flat dispatch over the request kinds, each a few lines of formatting.
#[expect(clippy::too_many_lines)]
pub(super) fn run(inner: &Inner, request: Request) -> Result<Output, ToolError> {
    let workspace = &inner.workspace;
    let max = inner.limits.max_output_bytes;
    match request {
        Request::List { path } => {
            let list = workspace
                .list_files(&ListOptions {
                    path,
                    max_entries: None,
                    max_output_bytes: Some(max),
                })
                .map_err(service_error)?;
            let mut text = String::new();
            for file in &list.files {
                let _ = writeln!(text, "{}\t{} bytes", file.path, file.size);
            }
            if list.files.is_empty() {
                text.push_str("(no files)\n");
            }
            Ok(marked(&text, list.truncation, max))
        }
        Request::Read {
            path,
            start_line,
            end_line,
        } => {
            let start_line = start_line.unwrap_or(1);
            if start_line == 0 || end_line.is_some_and(|end| end < start_line) {
                return Err(ToolError::Invalid(
                    "start_line must be at least 1 and end_line not before it".into(),
                ));
            }
            let read = workspace
                .read_file(
                    &path,
                    &ReadOptions {
                        start_line,
                        end_line,
                        max_bytes: Some(max),
                        include_ignored: false,
                    },
                )
                .map_err(service_error)?;
            let mut text = String::new();
            for line in &read.lines {
                let _ = writeln!(text, "{}: {}", line.number, line.text);
            }
            let mut output = marked(&text, read.truncation, max);
            if let (Some(next), Output::Text { text, .. }) = (read.next_line, &mut output) {
                let _ = writeln!(text, "[more lines follow; continue at start_line {next}]");
            }
            Ok(output)
        }
        Request::Search {
            pattern,
            path,
            mode: SearchMode::Ranked,
        } => {
            if path.is_some() {
                return Err(ToolError::Invalid(
                    "ranked search covers the whole root; omit path".into(),
                ));
            }
            let ranked = workspace.search(&pattern).map_err(service_error)?;
            let mut text = String::new();
            for file in &ranked.files {
                let _ = writeln!(
                    text,
                    "{}:{}: {}",
                    file.path,
                    file.line_number,
                    file.snippet.replace('\n', " ")
                );
            }
            if ranked.files.is_empty() {
                text.push_str("(no matches)\n");
            }
            Ok(marked(&text, ranked.truncation, max))
        }
        Request::Search {
            pattern,
            path,
            mode,
        } => {
            let found = workspace
                .find_matches(&SearchOptions {
                    pattern,
                    mode: if mode == SearchMode::Regex {
                        PatternMode::Regex
                    } else {
                        PatternMode::Literal
                    },
                    case_insensitive: false,
                    path,
                    max_results: None,
                    max_output_bytes: Some(max),
                })
                .map_err(service_error)?;
            let mut text = String::new();
            for hit in &found.matches {
                let _ = writeln!(text, "{}:{}: {}", hit.path, hit.line_number, hit.line);
            }
            if found.matches.is_empty() {
                text.push_str("(no matches)\n");
            }
            Ok(marked(&text, found.truncation, max))
        }
        Request::Write { path, content } => write_file(inner, &path, &content),
        Request::Edit { path, old, new } => edit_file(inner, &path, &old, &new),
        Request::Shell { .. } | Request::Fetch { .. } => {
            Err(ToolError::Failed("not a filesystem request".into()))
        }
    }
}

/// Create or replace `path` with `content`.
fn write_file(inner: &Inner, path: &str, content: &str) -> Result<Output, ToolError> {
    check_size(content.len(), inner.limits.max_write_bytes)?;
    let root = root_dir(inner)?;
    let rel = relative_path(path, false).map_err(service_error)?;
    let (parent, name) = open_parent(&root, &rel, true)?;
    let seen = check_target(&parent, &name, path, true)?;
    let (file, _) = open_target(&parent, &name, path, seen)?;
    replace_contents(&file, content.as_bytes(), path)?;
    Ok(Output::Text {
        text: format!("wrote {} bytes to {path}", content.len()),
        truncated: false,
    })
}

/// Replace the single occurrence of `old` in `path` with `new`.
fn edit_file(inner: &Inner, path: &str, old: &str, new: &str) -> Result<Output, ToolError> {
    if old.is_empty() {
        return Err(ToolError::Invalid("old text must not be empty".into()));
    }
    let max = inner.limits.max_write_bytes;
    let root = root_dir(inner)?;
    let rel = relative_path(path, false).map_err(service_error)?;
    let (parent, name) = open_parent(&root, &rel, false)?;
    let seen = check_target(&parent, &name, path, false)?;
    let (mut file, size) = open_target(&parent, &name, path, seen)?;
    check_size(usize::try_from(size).unwrap_or(usize::MAX), max)?;
    let mut bytes = Vec::new();
    (&mut file)
        .take(max as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, &error))?;
    check_size(bytes.len(), max)?;
    let current = String::from_utf8(bytes)
        .map_err(|_| ToolError::Invalid(format!("{path} is not UTF-8 text")))?;
    let count = current.matches(old).count();
    if count != 1 {
        return Err(ToolError::Invalid(format!(
            "old text occurs {count} times in {path}; it must occur exactly once"
        )));
    }
    let edited = current.replacen(old, new, 1);
    check_size(edited.len(), max)?;
    replace_contents(&file, edited.as_bytes(), path)?;
    Ok(Output::Text {
        text: format!("edited {path}: replaced 1 occurrence"),
        truncated: false,
    })
}

/// `text` capped at `max` bytes, with a marker when a workspace limit or the
/// cap cut it short.
fn marked(text: &str, truncation: Option<Truncation>, max: usize) -> Output {
    let (mut text, cut) = truncate_marked(text, max);
    if let Some(reason) = truncation {
        let _ = writeln!(text, "[truncated: result limited ({reason:?})]");
    }
    Output::Text {
        truncated: cut || truncation.is_some(),
        text,
    }
}

fn check_size(len: usize, max: usize) -> Result<(), ToolError> {
    if len > max {
        return Err(ToolError::Invalid(format!(
            "the file would be {len} bytes; the limit is {max}"
        )));
    }
    Ok(())
}

fn root_dir(inner: &Inner) -> Result<Dir, ToolError> {
    Dir::open_ambient_dir(&inner.root, ambient_authority())
        .map_err(|error| ToolError::Failed(format!("cannot open the tools root: {error}")))
}

/// The directory holding `rel`'s last component and that component's name,
/// never following a symlink; missing directories are created when `create`.
fn open_parent(root: &Dir, rel: &Path, create: bool) -> Result<(Dir, OsString), ToolError> {
    let display = rel.display().to_string();
    let mut parts: Vec<&std::ffi::OsStr> = rel.iter().collect();
    let name = parts
        .pop()
        .ok_or_else(|| ToolError::Invalid("path must name a file".into()))?
        .to_owned();
    let mut dir = root
        .try_clone()
        .map_err(|error| ToolError::Failed(error.to_string()))?;
    for part in parts {
        let seen = match dir.symlink_metadata(part) {
            Ok(seen) => seen,
            Err(error) if error.kind() == ErrorKind::NotFound && create => {
                dir.create_dir(part)
                    .map_err(|error| io_error(&display, &error))?;
                dir.symlink_metadata(part)
                    .map_err(|error| io_error(&display, &error))?
            }
            Err(error) => return Err(io_error(&display, &error)),
        };
        if seen.file_type().is_symlink() {
            return Err(ToolError::Denied(format!(
                "{display} is reached through a symlink, which is never followed"
            )));
        }
        if !seen.is_dir() {
            return Err(ToolError::Invalid(format!(
                "{display}: a parent component is not a directory"
            )));
        }
        let next = dir
            .open_dir(part)
            .map_err(|error| io_error(&display, &error))?;
        let opened = next
            .dir_metadata()
            .map_err(|error| io_error(&display, &error))?;
        if (opened.dev(), opened.ino()) != (seen.dev(), seen.ino()) {
            return Err(ToolError::Failed(format!(
                "{display} changed while it was being opened"
            )));
        }
        dir = next;
    }
    Ok((dir, name))
}

/// Check the final component: never a symlink, and a regular file (or, when
/// `may_create`, absent). Returns its device and inode (`None` when absent).
fn check_target(
    parent: &Dir,
    name: &OsString,
    display: &str,
    may_create: bool,
) -> Result<Option<(u64, u64)>, ToolError> {
    match parent.symlink_metadata(name) {
        Ok(seen) if seen.file_type().is_symlink() => Err(ToolError::Denied(format!(
            "{display} is a symlink, which is never followed"
        ))),
        // Another name for the same file may live outside the root.
        Ok(seen) if seen.is_file() && seen.nlink() > 1 => Err(more_links(display)),
        Ok(seen) if seen.is_file() => Ok(Some((seen.dev(), seen.ino()))),
        Ok(_) => Err(ToolError::Invalid(format!(
            "{display} is not a regular file"
        ))),
        Err(error) if error.kind() == ErrorKind::NotFound && may_create => Ok(None),
        Err(error) => Err(io_error(display, &error)),
    }
}

/// Open the final component `name` of `parent` for reading and writing,
/// creating it when `seen` (what [`check_target`] found) is `None`, and
/// return it with its size. Nothing is truncated yet.
///
/// `O_NOFOLLOW` refuses a symlink swapped in since the check, and a single
/// component opened relative to `parent` cannot leave it. The opened handle
/// must again be a regular file with one link, and the checked inode when
/// there was one; `O_NONBLOCK` keeps a swapped-in FIFO from blocking the
/// open.
fn open_target(
    parent: &Dir,
    name: &OsString,
    display: &str,
    seen: Option<(u64, u64)>,
) -> Result<(File, u64), ToolError> {
    let mut flags = OFlags::RDWR | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
    if seen.is_none() {
        flags |= OFlags::CREATE;
    }
    let mode = Mode::RUSR | Mode::WUSR | Mode::RGRP | Mode::WGRP | Mode::ROTH | Mode::WOTH;
    let file = match rustix::fs::openat(parent, name.as_os_str(), flags, mode) {
        Ok(fd) => File::from(fd),
        Err(Errno::LOOP) => {
            return Err(ToolError::Denied(format!(
                "{display} is a symlink, which is never followed"
            )));
        }
        Err(errno) => return Err(io_error(display, &errno.into())),
    };
    let opened = file.metadata().map_err(|error| io_error(display, &error))?;
    if !opened.is_file() {
        return Err(ToolError::Invalid(format!(
            "{display} is not a regular file"
        )));
    }
    if opened.nlink() > 1 {
        return Err(more_links(display));
    }
    if seen.is_some_and(|seen| seen != (opened.dev(), opened.ino())) {
        return Err(ToolError::Failed(format!(
            "{display} changed while it was being opened"
        )));
    }
    Ok((file, opened.len()))
}

/// Make `file` hold exactly `content`.
fn replace_contents(file: &File, content: &[u8], display: &str) -> Result<(), ToolError> {
    file.set_len(0)
        .and_then(|()| file.write_all_at(content, 0))
        .map_err(|error| io_error(display, &error))
}

fn more_links(display: &str) -> ToolError {
    ToolError::Denied(format!(
        "{display} has other hard links, so writing it could change files outside the root"
    ))
}

fn io_error(display: &str, error: &std::io::Error) -> ToolError {
    match error.kind() {
        ErrorKind::NotFound => ToolError::Invalid(format!("{display} does not exist")),
        // cap-std reports a path that would leave the root this way.
        ErrorKind::PermissionDenied => ToolError::Denied(format!(
            "{display} cannot be opened inside the tools root: {error}"
        )),
        _ => ToolError::Failed(format!("{display}: {error}")),
    }
}

fn service_error(error: Error) -> ToolError {
    match error {
        Error::InvalidArgument(message) | Error::NotFound(message) | Error::Conflict(message) => {
            ToolError::Invalid(message)
        }
        other => ToolError::Failed(other.to_string()),
    }
}
