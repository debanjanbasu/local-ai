//! Hosted tools: native executors a server may run on the model's behalf,
//! each only under an explicit, opt-in [`Permissions`] grant.
//!
//! Nothing here is reachable by default: [`Permissions::default`] grants no
//! capability at all, and every request names exactly one capability that is
//! checked before anything is touched. The executors are:
//!
//! - **Filesystem** ([`fs`]): list, read and search inside one authorized
//!   root through the capability-rooted [`Workspace`](crate::workspace::Workspace)
//!   (so `..`, absolute paths, `.git` and symlinks are refused), plus whole-file
//!   writes and exact-text edits, which need the separate `write` grant.
//! - **Shell** ([`shell`]): `/bin/sh -c` under a macOS `sandbox-exec`
//!   profile that denies all network access and every file write outside a
//!   per-call scratch directory (and, with `write`, the root), with CPU-time,
//!   file-size, open-file and process-count `setrlimit`s applied in the child
//!   before `exec`, a wall-clock deadline, output byte caps and termination
//!   of the child's own process group.
//! - **Web fetch** ([`fetch`]): one `GET` to a host on an explicit domain
//!   allowlist, refusing private, loopback and other non-public addresses,
//!   never following redirects, with time and size limits.
//!
//! Every result that was cut short says so with an explicit marker, so a
//! truncated answer is never mistaken for a complete one.
//!
//! What the OS cannot enforce is documented rather than imitated: macOS
//! rejects `RLIMIT_AS`, so a sandboxed command's memory is bounded only by
//! its CPU time and deadline, and a process the command starts that moves
//! itself into a new session or group (`setsid`, `setpgid`) leaves the
//! process group that is terminated at the end of the call. See
//! `docs/TOOLS.md`.

pub mod fetch;
pub mod fs;
pub mod shell;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::workspace::Workspace;

/// Default wall-clock deadline of one hosted call.
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(20);
/// Default bytes of output one call may return to the model.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 32 * 1024;

/// Which hosted capabilities a server grants. The default grants none.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Permissions {
    /// The only directory filesystem tools and the shell may use. Without
    /// it, neither is available.
    pub root: Option<PathBuf>,
    /// Allow writes and edits under `root` (and shell writes there).
    pub write: bool,
    /// Allow sandboxed shell commands, run in `root`.
    pub shell: bool,
    /// Hosts web fetch may contact (each also admits its subdomains). Empty
    /// disables web fetch.
    pub fetch_domains: Vec<String>,
}

/// Bounds applied to every hosted call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Wall-clock deadline of one call; a shell command is killed at it.
    pub call_timeout: Duration,
    /// Bytes of output one call may return (each shell stream separately).
    pub max_output_bytes: usize,
    /// `RLIMIT_CPU` of a shell command, in seconds.
    pub shell_cpu_seconds: u64,
    /// `RLIMIT_FSIZE` of a shell command: the largest file it may write.
    pub shell_file_bytes: u64,
    /// `RLIMIT_NOFILE` of a shell command.
    pub shell_open_files: u64,
    /// New processes a shell command may create beyond those its user
    /// already runs (`RLIMIT_NPROC` is per user, so it is set to the user's
    /// current count plus this).
    pub shell_processes: u64,
    /// Bytes of body web fetch reads before it stops and marks truncation.
    pub fetch_max_bytes: usize,
    /// Largest file a write or edit may produce.
    pub max_write_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            call_timeout: DEFAULT_CALL_TIMEOUT,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            shell_cpu_seconds: 10,
            shell_file_bytes: 16 * 1024 * 1024,
            shell_open_files: 256,
            shell_processes: 64,
            fetch_max_bytes: 256 * 1024,
            max_write_bytes: 4 * 1024 * 1024,
        }
    }
}

/// One hosted operation, as the model requested it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// List files under a root-relative directory (`None`: the whole root).
    List { path: Option<String> },
    /// Read lines of a root-relative file.
    Read {
        path: String,
        start_line: Option<u64>,
        end_line: Option<u64>,
    },
    /// Find lines matching `pattern` (literal unless `regex`), or with
    /// `ranked`, rank files for the query's words.
    Search {
        pattern: String,
        path: Option<String>,
        mode: SearchMode,
    },
    /// Create or replace a root-relative file with `content`.
    Write { path: String, content: String },
    /// Replace the one occurrence of `old` in a root-relative file with `new`.
    Edit {
        path: String,
        old: String,
        new: String,
    },
    /// Run each command in turn under the shell sandbox.
    Shell {
        commands: Vec<String>,
        /// Per-command deadline asked for; never above the call deadline.
        timeout: Option<Duration>,
        /// Per-stream byte cap asked for; never above the output limit.
        max_output_bytes: Option<usize>,
    },
    /// `GET` one URL on the allowlist.
    Fetch { url: String },
}

/// How [`Request::Search`] matches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchMode {
    #[default]
    Literal,
    Regex,
    Ranked,
}

/// The capability a request needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Read,
    Write,
    Shell,
    Fetch,
}

impl Request {
    /// The grant this request needs.
    #[must_use]
    pub const fn capability(&self) -> Capability {
        match self {
            Self::List { .. } | Self::Read { .. } | Self::Search { .. } => Capability::Read,
            Self::Write { .. } | Self::Edit { .. } => Capability::Write,
            Self::Shell { .. } => Capability::Shell,
            Self::Fetch { .. } => Capability::Fetch,
        }
    }
}

/// A successful call's result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Output {
    /// Text for the model; `truncated` when a limit cut it (the text then
    /// ends with a marker saying so).
    Text { text: String, truncated: bool },
    /// One run per shell command, in order.
    Shell(Vec<shell::CommandRun>),
}

impl Output {
    /// The text the model reads back.
    #[must_use]
    pub fn model_text(&self) -> String {
        match self {
            Self::Text { text, .. } => text.clone(),
            Self::Shell(runs) => shell::model_text(runs),
        }
    }
}

/// Why a hosted call produced no result.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolError {
    /// The server's permissions do not allow this. Nothing was run.
    #[error("refused: {0}")]
    Denied(String),
    /// The arguments are malformed or name something that does not exist.
    #[error("invalid arguments: {0}")]
    Invalid(String),
    /// The call ran out of time and was stopped.
    #[error("timed out after {0:?}")]
    TimedOut(Duration),
    /// The operation itself failed.
    #[error("failed: {0}")]
    Failed(String),
}

/// The configured executors. Cheap to clone; share one per server.
#[derive(Debug, Clone)]
pub struct Toolbox {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    permissions: Permissions,
    limits: Limits,
    /// The canonical root, when one was granted.
    root: Option<PathBuf>,
    workspace: Option<Workspace>,
}

impl Toolbox {
    /// Validate `permissions` and open the root. A grant that needs a root
    /// without one, or a root that is not a directory, is an error here,
    /// before anything can run.
    pub fn new(permissions: Permissions, limits: Limits) -> Result<Self, ToolError> {
        if limits.call_timeout.is_zero()
            || limits.max_output_bytes == 0
            || limits.fetch_max_bytes == 0
            || limits.max_write_bytes == 0
            || limits.shell_cpu_seconds == 0
            || limits.shell_open_files < 16
        {
            return Err(ToolError::Invalid(
                "tool limits must be non-zero (and allow at least 16 open files)".into(),
            ));
        }
        if (permissions.write || permissions.shell) && permissions.root.is_none() {
            return Err(ToolError::Invalid(
                "write and shell permissions need a tools root directory".into(),
            ));
        }
        for domain in &permissions.fetch_domains {
            fetch::check_domain(domain)?;
        }
        let workspace = permissions
            .root
            .as_ref()
            .map(Workspace::open)
            .transpose()
            .map_err(|error| ToolError::Invalid(format!("tools root: {error}")))?;
        let root = workspace
            .as_ref()
            .map(|workspace| workspace.root().to_owned());
        Ok(Self {
            inner: Arc::new(Inner {
                permissions,
                limits,
                root,
                workspace,
            }),
        })
    }

    /// Whether `capability` is granted.
    #[must_use]
    pub fn allows(&self, capability: Capability) -> bool {
        let permissions = &self.inner.permissions;
        match capability {
            Capability::Read => self.inner.root.is_some(),
            Capability::Write => self.inner.root.is_some() && permissions.write,
            Capability::Shell => self.inner.root.is_some() && permissions.shell,
            Capability::Fetch => !permissions.fetch_domains.is_empty(),
        }
    }

    /// Whether any capability is granted.
    #[must_use]
    pub fn any(&self) -> bool {
        [
            Capability::Read,
            Capability::Write,
            Capability::Shell,
            Capability::Fetch,
        ]
        .into_iter()
        .any(|capability| self.allows(capability))
    }

    /// The canonical root, when one was granted.
    #[must_use]
    pub fn root(&self) -> Option<&Path> {
        self.inner.root.as_deref()
    }

    #[must_use]
    pub fn limits(&self) -> &Limits {
        &self.inner.limits
    }

    #[must_use]
    pub fn permissions(&self) -> &Permissions {
        &self.inner.permissions
    }

    /// Run one request under the deadline `timeout` (never above the call
    /// limit). Refused before anything happens when its capability is not
    /// granted. Dropping the future stops the work: a shell command's process
    /// group is killed, a fetch's connection is closed. Filesystem work is
    /// synchronous and bounded by the workspace limits, so it finishes on its
    /// blocking thread even when abandoned.
    pub async fn run(&self, request: Request, timeout: Duration) -> Result<Output, ToolError> {
        let capability = request.capability();
        if !self.allows(capability) {
            return Err(ToolError::Denied(denial(capability)));
        }
        let timeout = timeout.min(self.inner.limits.call_timeout);
        if timeout.is_zero() {
            return Err(ToolError::TimedOut(timeout));
        }
        if let Request::Shell {
            commands,
            timeout: asked,
            max_output_bytes,
        } = request
        {
            // The shell enforces its own deadlines, killing the process group
            // and the leader and keeping the output so far. The outer bound
            // only catches a command that outlives that kill: dropping the
            // run kills the group and, by PID, the leader again.
            let root = self.root_path()?;
            let policy = shell::Policy::new(root, self.inner.permissions.write, &self.inner.limits);
            let per_command = asked.map_or(timeout, |asked| asked.min(timeout));
            let cap = max_output_bytes.map_or(self.inner.limits.max_output_bytes, |asked| {
                asked.clamp(1, self.inner.limits.max_output_bytes)
            });
            let bound = timeout + shell::DRAIN_GRACE + shell::REAP_GRACE;
            return tokio::time::timeout(
                bound,
                shell::run_all(&policy, &commands, per_command, timeout, cap),
            )
            .await
            .unwrap_or(Err(ToolError::TimedOut(timeout)))
            .map(Output::Shell);
        }
        let work = async {
            match request {
                Request::Fetch { url } => {
                    fetch::fetch(
                        &url,
                        &self.inner.permissions.fetch_domains,
                        self.inner.limits.fetch_max_bytes,
                    )
                    .await
                }
                other => {
                    let inner = Arc::clone(&self.inner);
                    tokio::task::spawn_blocking(move || fs::run(&inner, other))
                        .await
                        .map_err(|error| ToolError::Failed(format!("tool task failed: {error}")))?
                }
            }
        };
        tokio::time::timeout(timeout, work)
            .await
            .unwrap_or(Err(ToolError::TimedOut(timeout)))
    }

    fn root_path(&self) -> Result<&Path, ToolError> {
        self.inner
            .root
            .as_deref()
            .ok_or_else(|| ToolError::Denied("no tools root is configured".into()))
    }
}

/// Why `capability` is refused, naming the server flag that would grant it.
#[must_use]
pub fn denial(capability: Capability) -> String {
    match capability {
        Capability::Read => "filesystem tools are disabled on this server (no --tools-root)".into(),
        Capability::Write => {
            "file writes are disabled on this server (start it with --tools-root and \
             --tools-allow-write)"
                .into()
        }
        Capability::Shell => {
            "the shell is disabled on this server (start it with --tools-root and --tools-shell)"
                .into()
        }
        Capability::Fetch => {
            "web fetch is disabled on this server (start it with --tools-fetch-domain)".into()
        }
    }
}

/// Cut `text` to at most `max` bytes on a character boundary, appending a
/// marker that says how much was dropped. Returns the text and whether it
/// was cut.
#[must_use]
pub fn truncate_marked(text: &str, max: usize) -> (String, bool) {
    if text.len() <= max {
        return (text.to_owned(), false);
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let kept = text.get(..end).unwrap_or_default();
    (
        format!(
            "{kept}\n[truncated: {} of {} bytes shown]",
            kept.len(),
            text.len()
        ),
        true,
    )
}
