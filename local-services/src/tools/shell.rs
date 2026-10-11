//! Sandboxed shell commands.
//!
//! Rust's memory safety says nothing about what a spawned program does, so
//! every command is confined by the operating system instead:
//!
//! - **`sandbox-exec`** runs `/bin/sh -c COMMAND` under a deny-by-default
//!   Seatbelt profile ([`profile`]): no network of any kind, no Mach service
//!   lookups (so nothing can ask `launchd` to start a process outside the
//!   sandbox), file reads only of the system directories a shell needs, the
//!   root and a per-call scratch directory (metadata is readable everywhere),
//!   and file writes only to the scratch directory, `/dev/null` and, with the
//!   `write` grant, the root outside `.git`.
//! - **`setrlimit`** in the child, before `exec`: `RLIMIT_CPU`,
//!   `RLIMIT_FSIZE`, `RLIMIT_NOFILE`, `RLIMIT_CORE` (zero) and
//!   `RLIMIT_NPROC`, soft and hard alike so the command cannot raise them.
//!   `RLIMIT_NPROC` counts every process of the user, so it is set to the
//!   user's current count plus [`super::Limits::shell_processes`].
//! - **A wall-clock deadline**, after which the child's own process group
//!   (it is started as the leader of a new one) receives `SIGKILL`. The group
//!   is also killed when the command exits, so nothing it left in the
//!   background outlives the call, and when the caller drops the future. The
//!   group is always signalled while its leader is still unreaped, so its ID
//!   cannot have been reused by an unrelated process. Nothing else is ever
//!   signalled.
//! - **Output caps**: each stream keeps at most the cap and counts the rest,
//!   and a cut stream ends with a marker giving both numbers.
//! - A clean environment (`PATH`, `HOME` and `TMPDIR` pointing at the
//!   scratch directory, `LANG`) and `stdin` from `/dev/null`.
//!
//! Gaps, stated rather than papered over: macOS rejects `RLIMIT_AS` and does
//! not enforce `RLIMIT_DATA`/`RLIMIT_RSS`, so memory is bounded only through
//! CPU time and the deadline; a process that calls `setsid` leaves the
//! process group and is not killed at the deadline (it remains confined by
//! the sandbox, its CPU limit and the process limit); and file reads of
//! system directories are allowed, so a command can read what any local
//! user can read there.

use std::fmt::Write as _;
use std::os::unix::process::ExitStatusExt as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use rustix::process::{Pid, Resource, Rlimit, Signal, WaitId, WaitIdOptions};
use tokio::io::{AsyncRead, AsyncReadExt as _};
use tokio::time::Instant;

use super::{Limits, ToolError};

/// The system's Seatbelt launcher.
const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";
/// The only shell commands run under.
const SHELL: &str = "/bin/sh";
/// Commands one call may run.
pub const MAX_COMMANDS: usize = 16;
/// Bytes one command string may hold.
pub const MAX_COMMAND_BYTES: usize = 16 * 1024;
/// How long output is still collected after the group was killed, for a
/// process that escaped the group while holding a pipe.
const DRAIN_GRACE: Duration = Duration::from_secs(1);

/// What one command may do, fixed by the server's grants.
#[derive(Debug, Clone)]
pub struct Policy {
    root: PathBuf,
    write: bool,
    cpu_seconds: u64,
    file_bytes: u64,
    open_files: u64,
    processes: u64,
}

impl Policy {
    /// The policy for commands run in `root` (canonical), writing there only
    /// with `write`.
    #[must_use]
    pub fn new(root: &Path, write: bool, limits: &Limits) -> Self {
        Self {
            root: root.to_owned(),
            write,
            cpu_seconds: limits.shell_cpu_seconds,
            file_bytes: limits.shell_file_bytes,
            open_files: limits.shell_open_files,
            processes: limits.shell_processes,
        }
    }
}

/// How a command ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// It exited with this status.
    Exit(i32),
    /// It was ended by this signal (for instance `SIGXCPU` at the CPU limit).
    Signal(i32),
    /// It outlived its deadline and its process group was killed.
    TimedOut,
}

/// One command's result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandRun {
    pub command: String,
    /// Captured standard output, ending with a marker when it was cut.
    pub stdout: String,
    /// Captured standard error, ending with a marker when it was cut.
    pub stderr: String,
    pub outcome: Outcome,
    /// Whether either stream was cut at the cap.
    pub truncated: bool,
}

/// The text the model reads for `runs`.
#[must_use]
pub fn model_text(runs: &[CommandRun]) -> String {
    let mut text = String::new();
    for run in runs {
        let _ = writeln!(text, "$ {}", run.command);
        if !run.stdout.is_empty() {
            text.push_str(&run.stdout);
            if !run.stdout.ends_with('\n') {
                text.push('\n');
            }
        }
        if !run.stderr.is_empty() {
            let _ = writeln!(text, "[stderr]");
            text.push_str(&run.stderr);
            if !run.stderr.ends_with('\n') {
                text.push('\n');
            }
        }
        let _ = match run.outcome {
            Outcome::Exit(code) => writeln!(text, "[exit code {code}]"),
            Outcome::Signal(signal) => writeln!(text, "[terminated by signal {signal}]"),
            Outcome::TimedOut => writeln!(text, "[timed out; the command was killed]"),
        };
    }
    text
}

/// Check one command string before anything runs.
pub fn check_command(command: &str) -> Result<(), ToolError> {
    if command.trim().is_empty() {
        return Err(ToolError::Invalid(
            "a shell command must not be empty".into(),
        ));
    }
    if command.len() > MAX_COMMAND_BYTES {
        return Err(ToolError::Invalid(format!(
            "a shell command may hold at most {MAX_COMMAND_BYTES} bytes"
        )));
    }
    if command.contains('\0') {
        return Err(ToolError::Invalid(
            "a shell command must not contain NUL bytes".into(),
        ));
    }
    Ok(())
}

/// Run `commands` in order, each under `per_command` and all within
/// `total`. A command that would start after `total` has passed is not run
/// and is reported as timed out.
pub async fn run_all(
    policy: &Policy,
    commands: &[String],
    per_command: Duration,
    total: Duration,
    cap: usize,
) -> Result<Vec<CommandRun>, ToolError> {
    if commands.is_empty() || commands.len() > MAX_COMMANDS {
        return Err(ToolError::Invalid(format!(
            "commands must hold 1 to {MAX_COMMANDS} commands"
        )));
    }
    for command in commands {
        check_command(command)?;
    }
    let deadline = Instant::now() + total;
    let mut runs = Vec::with_capacity(commands.len());
    for command in commands {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            runs.push(CommandRun {
                command: command.clone(),
                stdout: String::new(),
                stderr: "[not run: the call's time budget was used up]".into(),
                outcome: Outcome::TimedOut,
                truncated: false,
            });
            continue;
        }
        runs.push(run(policy, command, per_command.min(left), cap).await?);
    }
    Ok(runs)
}

/// The Seatbelt profile. `ROOT`, `SCRATCH` and `GIT` (the root's `.git`) are
/// passed as parameters, never spliced into the text.
#[must_use]
pub fn profile(write: bool) -> String {
    let mut profile = String::from(
        r#"(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow sysctl-read)
(allow file-read-metadata)
(allow file-read* (literal "/"))
(allow file-read*
  (subpath "/usr")
  (subpath "/bin")
  (subpath "/sbin")
  (subpath "/System")
  (subpath "/Library/Apple")
  (subpath "/private/etc")
  (subpath "/private/var/db/timezone")
  (subpath "/dev")
  (subpath (param "ROOT"))
  (subpath (param "SCRATCH")))
(allow file-write-data
  (require-all (path "/dev/null") (vnode-type CHARACTER-DEVICE)))
(allow file-write* (subpath (param "SCRATCH")))
"#,
    );
    if write {
        profile.push_str(
            "(allow file-write*\n  (require-all (subpath (param \"ROOT\")) \
             (require-not (subpath (param \"GIT\")))))\n",
        );
    }
    profile.push_str("(deny network*)\n");
    profile
}

/// Resource limits applied in the child before `exec`.
#[derive(Debug, Clone, Copy)]
struct ChildLimits {
    cpu_seconds: u64,
    file_bytes: u64,
    open_files: u64,
    processes: u64,
}

impl ChildLimits {
    /// The limits for `policy`, with the process limit counted now. Fails
    /// closed when the user's process count cannot be read.
    fn for_policy(policy: &Policy) -> Result<Self, ToolError> {
        let current = user_processes().ok_or_else(|| {
            ToolError::Failed("cannot count the user's processes to bound the shell".into())
        })?;
        let wanted = current.saturating_add(policy.processes);
        let hard = rustix::process::getrlimit(Resource::Nproc)
            .maximum
            .unwrap_or(u64::MAX);
        Ok(Self {
            cpu_seconds: policy.cpu_seconds,
            file_bytes: policy.file_bytes,
            open_files: policy.open_files.min(
                rustix::process::getrlimit(Resource::Nofile)
                    .maximum
                    .unwrap_or(u64::MAX),
            ),
            processes: wanted.min(hard),
        })
    }

    /// Apply the limits. Runs between `fork` and `exec`, so it only makes
    /// system calls and never allocates.
    fn apply(self) -> std::io::Result<()> {
        let both = |value: u64| Rlimit {
            current: Some(value),
            maximum: Some(value),
        };
        // CPU: the soft limit delivers SIGXCPU; the hard one, a second
        // later, SIGKILL for a process that ignores it.
        rustix::process::setrlimit(
            Resource::Cpu,
            Rlimit {
                current: Some(self.cpu_seconds),
                maximum: Some(self.cpu_seconds.saturating_add(1)),
            },
        )?;
        rustix::process::setrlimit(Resource::Fsize, both(self.file_bytes))?;
        rustix::process::setrlimit(Resource::Nofile, both(self.open_files))?;
        rustix::process::setrlimit(Resource::Core, both(0))?;
        rustix::process::setrlimit(Resource::Nproc, both(self.processes))?;
        Ok(())
    }
}

/// `PROC_RUID_ONLY` from `<sys/proc_info.h>`: processes by real user ID,
/// the ID `RLIMIT_NPROC` is counted against.
const PROC_RUID_ONLY: u32 = 5;

/// Processes the current user (by real user ID) runs now.
#[allow(unsafe_code)]
fn user_processes() -> Option<u64> {
    let uid = rustix::process::getuid().as_raw();
    // SAFETY: with a null buffer `proc_listpids` only returns the bytes a
    // buffer for every process would need; nothing is written.
    let needed = unsafe { libc::proc_listpids(PROC_RUID_ONLY, uid, std::ptr::null_mut(), 0) };
    let slots = usize::try_from(needed).ok()? / std::mem::size_of::<libc::c_int>() + 64;
    let mut pids: Vec<libc::c_int> = vec![0; slots];
    let bytes = libc::c_int::try_from(slots * std::mem::size_of::<libc::c_int>()).ok()?;
    // SAFETY: `pids` owns `bytes` writable bytes, the size passed, and
    // outlives the call; the kernel writes at most that many.
    let filled =
        unsafe { libc::proc_listpids(PROC_RUID_ONLY, uid, pids.as_mut_ptr().cast(), bytes) };
    let count = usize::try_from(filled).ok()? / std::mem::size_of::<libc::c_int>();
    (count > 0).then_some(count as u64)
}

/// The child's process group, killed when this is dropped unless disarmed.
///
/// Signals only that group, whose ID is the child's own PID, and only while
/// the child is unreaped.
struct Group {
    leader: Pid,
    armed: bool,
}

impl Group {
    fn kill(&self) {
        if self.armed {
            // ESRCH (the group is already empty) is the expected outcome
            // for a command that left nothing behind.
            let _ = rustix::process::kill_process_group(self.leader, Signal::KILL);
        }
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Bytes read from one stream: the first `cap`, and how many there were.
struct Capture {
    kept: Vec<u8>,
    total: u64,
}

impl Capture {
    fn text(&self, cap: usize) -> (String, bool) {
        let mut text = String::from_utf8_lossy(&self.kept).into_owned();
        let cut = self.total > self.kept.len() as u64;
        if cut {
            if !text.ends_with('\n') {
                text.push('\n');
            }
            let _ = write!(
                text,
                "[truncated: {} of {} bytes shown (cap {cap})]",
                self.kept.len(),
                self.total
            );
        }
        (text, cut)
    }
}

/// Read `stream` until it ends or `until`, keeping the first `cap` bytes.
async fn capture(stream: Option<impl AsyncRead + Unpin>, cap: usize, until: Instant) -> Capture {
    let mut capture = Capture {
        kept: Vec::new(),
        total: 0,
    };
    let Some(mut stream) = stream else {
        return capture;
    };
    let mut buffer = [0_u8; 8192];
    loop {
        match tokio::time::timeout_at(until, stream.read(&mut buffer)).await {
            Ok(Ok(0) | Err(_)) | Err(_) => return capture,
            Ok(Ok(read)) => {
                capture.total += read as u64;
                let room = cap.saturating_sub(capture.kept.len());
                capture
                    .kept
                    .extend_from_slice(buffer.get(..read.min(room)).unwrap_or_default());
            }
        }
    }
}

/// Run one command under `policy` for at most `timeout`.
pub async fn run(
    policy: &Policy,
    command: &str,
    timeout: Duration,
    cap: usize,
) -> Result<CommandRun, ToolError> {
    check_command(command)?;
    let failed = |what: &str, error: std::io::Error| ToolError::Failed(format!("{what}: {error}"));
    let scratch = tempfile::Builder::new()
        .prefix("local-ai-tool-")
        .tempdir()
        .map_err(|error| failed("cannot create the scratch directory", error))?;
    // Seatbelt matches resolved paths (`/var` is `/private/var`).
    let scratch_path = std::fs::canonicalize(scratch.path())
        .map_err(|error| failed("cannot resolve the scratch directory", error))?;
    let root = std::fs::canonicalize(&policy.root)
        .map_err(|error| failed("cannot resolve the tools root", error))?;
    let limits = ChildLimits::for_policy(policy)?;
    let mut builder = tokio::process::Command::new(SANDBOX_EXEC);
    builder
        .arg("-p")
        .arg(profile(policy.write))
        .arg("-D")
        .arg(param("ROOT", &root))
        .arg("-D")
        .arg(param("SCRATCH", &scratch_path))
        .arg("-D")
        .arg(param("GIT", &root.join(".git")))
        .arg(SHELL)
        .arg("-c")
        .arg(command)
        .current_dir(&root)
        .env_clear()
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .env("HOME", &scratch_path)
        .env("TMPDIR", &scratch_path)
        .env("LANG", "en_US.UTF-8")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // The leader of its own new group, so the group can be killed
        // without touching any other process.
        .process_group(0)
        .kill_on_drop(true);
    // SAFETY: the closure only calls `setrlimit`, which is async-signal-safe,
    // and touches nothing but the copied `limits`.
    #[allow(unsafe_code)]
    unsafe {
        builder.pre_exec(move || limits.apply());
    }
    let mut child = builder
        .spawn()
        .map_err(|error| failed("cannot start the sandboxed shell", error))?;
    let leader = child
        .id()
        .and_then(|id| i32::try_from(id).ok())
        .filter(|&id| id > 1)
        .and_then(Pid::from_raw)
        .ok_or_else(|| ToolError::Failed("the shell started without a usable PID".into()))?;
    // Declared after `child`, so it is dropped first: the group is killed
    // before the child handle reaps the leader.
    let mut group = Group {
        leader,
        armed: true,
    };
    let started = Instant::now();
    let deadline = started + timeout;
    let drain_until = deadline + DRAIN_GRACE;
    let stdout = tokio::spawn(capture(child.stdout.take(), cap, drain_until));
    let stderr = tokio::spawn(capture(child.stderr.take(), cap, drain_until));
    // Waits for the leader to exit without reaping it (`WNOWAIT`), so its
    // PID, and with it the group ID, stays reserved until the group is killed.
    let exited = tokio::task::spawn_blocking(move || {
        rustix::process::waitid(
            WaitId::Pid(leader),
            WaitIdOptions::EXITED | WaitIdOptions::NOWAIT,
        )
        .map(drop)
    });
    let timed_out = tokio::time::timeout_at(deadline, exited).await.is_err();
    group.kill();
    let stdout = stdout.await.unwrap_or(Capture {
        kept: Vec::new(),
        total: 0,
    });
    let stderr = stderr.await.unwrap_or(Capture {
        kept: Vec::new(),
        total: 0,
    });
    let status = child
        .wait()
        .await
        .map_err(|error| failed("cannot collect the shell's status", error))?;
    // Reaped: the group ID may now be reused, so it is never signalled again.
    group.armed = false;
    drop(scratch);
    let outcome = if timed_out {
        Outcome::TimedOut
    } else if let Some(code) = status.code() {
        Outcome::Exit(code)
    } else {
        Outcome::Signal(status.signal().unwrap_or_default())
    };
    let (stdout, out_cut) = stdout.text(cap);
    let (stderr, err_cut) = stderr.text(cap);
    Ok(CommandRun {
        command: command.to_owned(),
        stdout,
        stderr,
        outcome,
        truncated: out_cut || err_cut,
    })
}

fn param(name: &str, path: &Path) -> std::ffi::OsString {
    let mut param = std::ffi::OsString::from(format!("{name}="));
    param.push(path.as_os_str());
    param
}
