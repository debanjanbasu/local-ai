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
//! inside it.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::ErrorKind;
use std::path::Path;

use cap_std::ambient_authority;
use cap_std::fs::{Dir, MetadataExt as _};

use super::{Inner, Output, Request, SearchMode, ToolError, truncate_marked};
use crate::Error;
use crate::workspace::{
    ListOptions, PatternMode, ReadOptions, SearchOptions, Truncation, relative_path,
};

/// Run a filesystem request. Blocking.
// One flat dispatch over the request kinds, each a few lines of formatting.
#[allow(clippy::too_many_lines)]
pub(super) fn run(inner: &Inner, request: Request) -> Result<Output, ToolError> {
    let workspace = inner
        .workspace
        .as_ref()
        .ok_or_else(|| ToolError::Denied("no tools root is configured".into()))?;
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
    check_target(&parent, &name, path, true)?;
    parent
        .write(&name, content.as_bytes())
        .map_err(|error| io_error(path, &error))?;
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
    let root = root_dir(inner)?;
    let rel = relative_path(path, false).map_err(service_error)?;
    let (parent, name) = open_parent(&root, &rel, false)?;
    let size = check_target(&parent, &name, path, false)?;
    check_size(
        usize::try_from(size).unwrap_or(usize::MAX),
        inner.limits.max_write_bytes,
    )?;
    let bytes = parent.read(&name).map_err(|error| io_error(path, &error))?;
    let current = String::from_utf8(bytes)
        .map_err(|_| ToolError::Invalid(format!("{path} is not UTF-8 text")))?;
    let count = current.matches(old).count();
    if count != 1 {
        return Err(ToolError::Invalid(format!(
            "old text occurs {count} times in {path}; it must occur exactly once"
        )));
    }
    let edited = current.replacen(old, new, 1);
    check_size(edited.len(), inner.limits.max_write_bytes)?;
    parent
        .write(&name, edited.as_bytes())
        .map_err(|error| io_error(path, &error))?;
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
    let root = inner
        .root
        .as_ref()
        .ok_or_else(|| ToolError::Denied("no tools root is configured".into()))?;
    Dir::open_ambient_dir(root, ambient_authority())
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
/// `may_create`, absent). Returns its size (0 when absent).
fn check_target(
    parent: &Dir,
    name: &OsString,
    display: &str,
    may_create: bool,
) -> Result<u64, ToolError> {
    match parent.symlink_metadata(name) {
        Ok(seen) if seen.file_type().is_symlink() => Err(ToolError::Denied(format!(
            "{display} is a symlink, which is never followed"
        ))),
        // Another name for the same file may live outside the root.
        Ok(seen) if seen.is_file() && seen.nlink() > 1 => Err(ToolError::Denied(format!(
            "{display} has other hard links, so writing it could change files outside the root"
        ))),
        Ok(seen) if seen.is_file() => Ok(seen.len()),
        Ok(_) => Err(ToolError::Invalid(format!(
            "{display} is not a regular file"
        ))),
        Err(error) if error.kind() == ErrorKind::NotFound && may_create => Ok(0),
        Err(error) => Err(io_error(display, &error)),
    }
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
