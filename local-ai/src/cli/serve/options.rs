use std::path::PathBuf;
use std::time::Duration;

use super::Args;

/// Default rounds of hosted tool calls one Response may run.
const DEFAULT_TOOL_ITERATIONS: usize = 8;
const MAX_TOOL_ITERATIONS: usize = 64;
/// Default seconds one hosted call may run.
const DEFAULT_TOOL_CALL_SECONDS: u64 = 20;
const MAX_TOOL_CALL_SECONDS: u64 = 300;
/// Default seconds all hosted calls of one Response may run together.
const DEFAULT_TOOL_BUDGET_SECONDS: u64 = 120;
const MAX_TOOL_BUDGET_SECONDS: u64 = 3600;

/// Where Responses are stored.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ResponseStoreArg {
    /// The user's data directory, pruned of responses older than 30 days.
    Default,
    /// `--response-store DIR`, kept until deleted.
    Dir(PathBuf),
    /// `--no-response-store`: nothing is kept.
    Off,
}

/// Hosted tool permissions and bounds. The default grants nothing.
#[derive(Debug)]
pub(super) struct ToolArgs {
    pub(super) root: Option<PathBuf>,
    pub(super) write: bool,
    pub(super) shell: bool,
    pub(super) fetch_domains: Vec<String>,
    pub(super) max_iterations: usize,
    pub(super) call_timeout: Duration,
    pub(super) budget: Duration,
}

impl Default for ToolArgs {
    fn default() -> Self {
        Self {
            root: None,
            write: false,
            shell: false,
            fetch_domains: Vec::new(),
            max_iterations: DEFAULT_TOOL_ITERATIONS,
            call_timeout: Duration::from_secs(DEFAULT_TOOL_CALL_SECONDS),
            budget: Duration::from_secs(DEFAULT_TOOL_BUDGET_SECONDS),
        }
    }
}

/// Default seconds a client may stop taking frames before its generation is
/// dropped.
///
/// Measured on this M2, generation runs at 5-17 tok/s, so 30 s is roughly
/// 150-500 tokens of slack over what a live consumer needs. The kernel socket
/// buffers are what a merely-slow client runs into first (about 551 KiB here),
/// so this budget is only reached by a client that has stopped reading.
const DEFAULT_STALL_SECONDS: u64 = 30;

/// Below this the budget would abandon generations that are merely slow.
///
/// One flag governs two different clocks. On a streaming response it is a
/// *consumer*-liveness budget, because a frame is only produced when the engine
/// emits one, so a small value is safe. A non-streaming response is bounded by
/// the same number as a *producer*-liveness budget instead, and the engine's own
/// inter-frame gaps are not small: a speculative round measured 3.076 s (p95
/// 0.728 s) on a six-token prompt here. A budget under that abandons healthy
/// non-streaming requests, so the floor sits an order of magnitude above the
/// worst gap actually observed rather than at a round number.
const MIN_STALL_SECONDS: i64 = 10;

/// Above this a single wedged client can hold a batch slot, its sequence state
/// and its undelivered output, for an hour.
const MAX_STALL_SECONDS: i64 = 3600;

pub(super) fn usage() {
    eprintln!("Usage: local-ai serve [options]");
    eprintln!();
    eprintln!("  --host IP          Bind address (default: 127.0.0.1)");
    eprintln!("  --port N           TCP/UDP port (default: 8080)");
    eprintln!(
        "  --api-key KEY      Require bearer authentication; KEY authorizes every API but \
         /v1/organization (admin keys: `local-ai admin create-key`, docs/ADMIN.md)"
    );
    eprintln!(
        "  --allow-origin ORIGIN  Admit browser requests from ORIGIN (e.g. \
         https://chat.example.com) while authentication is off; repeatable; * for every origin, \
         null for file: pages (default: loopback pages and local apps only; see docs/BONSAI.md)"
    );
    eprintln!(
        "  --allow-host NAME  Answer requests addressed to host NAME besides IP addresses, \
         localhost and *.local/*.internal names (DNS rebinding protection; applies on loopback, \
         and elsewhere while authentication is off); repeatable"
    );
    eprintln!("  --no-thinking      Disable thinking");
    eprintln!(
        "  --response-store DIR  Keep Responses API responses in DIR (owner-only, \
         unencrypted) until deleted (default: ~/Library/Application Support/local-ai/responses, \
         where responses older than 30 days are removed at startup)"
    );
    eprintln!(
        "  --no-response-store  Keep no Responses (no retrieval or previous_response_id; \
         background responses are kept temporarily)"
    );
    eprintln!(
        "  --reasoning-key FILE  Owner-only key for encrypted reasoning and compaction, created \
         if absent (default: ~/Library/Application Support/local-ai/reasoning.key)"
    );
    eprintln!(
        "  --stall-timeout SECONDS  Drop a generation or download whose client stopped \
         reading, or a request body that stopped arriving (default: {DEFAULT_STALL_SECONDS})"
    );
    eprintln!();
    eprintln!(
        "  Hosted tools (Responses API; default: none; file_search needs no flag; see docs/TOOLS.md):"
    );
    eprintln!(
        "  --tools-root DIR   Let requests list, read and search files under DIR \
         (local_fs_list/read/search)"
    );
    eprintln!("  --tools-allow-write  Also allow writes and edits under DIR (local_fs_write/edit)");
    eprintln!(
        "  --tools-shell      Allow sandboxed shell commands in DIR (the shell tool): no \
         network, writes only with --tools-allow-write"
    );
    eprintln!(
        "  --tools-fetch-domain DOMAIN  Allow local_web_fetch to DOMAIN and its subdomains \
         (repeatable; public addresses only)"
    );
    eprintln!(
        "  --tools-max-iterations N  Rounds of hosted calls per response \
         (default: {DEFAULT_TOOL_ITERATIONS})"
    );
    eprintln!(
        "  --tools-call-timeout SECONDS  Deadline of one hosted call \
         (default: {DEFAULT_TOOL_CALL_SECONDS})"
    );
    eprintln!(
        "  --tools-budget SECONDS  Combined time of a response's hosted calls \
         (default: {DEFAULT_TOOL_BUDGET_SECONDS})"
    );
}

pub(super) fn parse(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        host: "127.0.0.1".parse().map_err(|_| "invalid default host")?,
        port: 8080,
        thinking: true,
        api_key: None,
        stall: Duration::from_secs(DEFAULT_STALL_SECONDS),
        response_store: ResponseStoreArg::Default,
        reasoning_key: None,
        tools: ToolArgs::default(),
        allowed_origins: Vec::new(),
        allowed_hosts: Vec::new(),
    };
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if matches!(flag, "--help" | "-h") {
            return Err(String::new());
        }
        if flag == "--no-thinking" {
            parsed.thinking = false;
            index += 1;
            continue;
        }
        if flag == "--no-response-store" {
            parsed.response_store = ResponseStoreArg::Off;
            index += 1;
            continue;
        }
        if matches!(flag, "--tools-allow-write" | "--tools-shell") {
            if flag == "--tools-shell" {
                parsed.tools.shell = true;
            } else {
                parsed.tools.write = true;
            }
            index += 1;
            continue;
        }
        index += 1;
        // Recognise the flag before demanding its value, so an unknown option is
        // reported as unknown rather than as a missing argument.
        if !matches!(
            flag,
            "--host"
                | "--port"
                | "--api-key"
                | "--allow-origin"
                | "--allow-host"
                | "--stall-timeout"
                | "--response-store"
                | "--reasoning-key"
                | "--tools-root"
                | "--tools-fetch-domain"
                | "--tools-max-iterations"
                | "--tools-call-timeout"
                | "--tools-budget"
        ) {
            return Err(format!("unknown option: {flag}"));
        }
        let value = args
            .get(index)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--host" => parsed.host = value.parse().map_err(|_| "invalid --host")?,
            "--port" => parsed.port = value.parse().map_err(|_| "invalid --port")?,
            "--api-key" => parsed.api_key = Some(value.clone()),
            "--allow-origin" => parsed.allowed_origins.push(allowed_origin(value)?),
            "--allow-host" => parsed.allowed_hosts.push(allowed_host(value)?),
            "--stall-timeout" => parsed.stall = stall_timeout(value)?,
            "--reasoning-key" => {
                if value.is_empty() {
                    return Err("--reasoning-key requires a file path".into());
                }
                parsed.reasoning_key = Some(value.into());
            }
            "--response-store" => {
                if value.is_empty() {
                    return Err("--response-store requires a directory".into());
                }
                parsed.response_store = ResponseStoreArg::Dir(value.into());
            }
            tools if tools.starts_with("--tools-") => tool_value(&mut parsed.tools, flag, value)?,
            _ => return Err(format!("unknown option: {flag}")),
        }
        index += 1;
    }
    if (parsed.tools.write || parsed.tools.shell) && parsed.tools.root.is_none() {
        return Err("--tools-allow-write and --tools-shell need --tools-root DIR".into());
    }
    Ok(parsed)
}

/// An `--allow-origin` value: `*`, `null`, or an origin as browsers send it
/// (`scheme://host[:port]`, no path), compared exactly.
fn allowed_origin(value: &str) -> Result<String, String> {
    let origin = value.strip_suffix('/').unwrap_or(value);
    let valid = matches!(origin, "*" | "null")
        || origin.split_once("://").is_some_and(|(scheme, host)| {
            !scheme.is_empty() && !host.is_empty() && !host.contains('/')
        });
    if valid {
        Ok(origin.to_ascii_lowercase())
    } else {
        Err(format!(
            "--allow-origin must be *, null or an origin such as https://chat.example.com, not \
             {value:?}"
        ))
    }
}

/// An `--allow-host` value: a host name as a client addresses this server,
/// without scheme or port, compared exactly (lowercase, no trailing dot).
fn allowed_host(value: &str) -> Result<String, String> {
    let host = value
        .strip_suffix('.')
        .unwrap_or(value)
        .to_ascii_lowercase();
    let valid = !host.is_empty()
        && host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_'));
    if valid {
        Ok(host)
    } else {
        Err(format!(
            "--allow-host must be a host name such as ai.example.com, without scheme or port, \
             not {value:?}"
        ))
    }
}

/// Apply a hosted-tool flag that takes `value`.
fn tool_value(tools: &mut ToolArgs, flag: &str, value: &str) -> Result<(), String> {
    match flag {
        "--tools-root" => {
            if value.is_empty() {
                return Err("--tools-root requires a directory".into());
            }
            tools.root = Some(value.into());
        }
        "--tools-fetch-domain" => tools.fetch_domains.push(value.to_owned()),
        "--tools-max-iterations" => {
            tools.max_iterations = bounded(flag, value, 1, MAX_TOOL_ITERATIONS as u64)? as usize;
        }
        "--tools-call-timeout" => {
            tools.call_timeout =
                Duration::from_secs(bounded(flag, value, 1, MAX_TOOL_CALL_SECONDS)?);
        }
        "--tools-budget" => {
            tools.budget = Duration::from_secs(bounded(flag, value, 1, MAX_TOOL_BUDGET_SECONDS)?);
        }
        _ => return Err(format!("unknown option: {flag}")),
    }
    Ok(())
}

/// A whole number for `flag` between `min` and `max`.
fn bounded(flag: &str, value: &str, min: u64, max: u64) -> Result<u64, String> {
    value
        .parse::<u64>()
        .ok()
        .filter(|number| (min..=max).contains(number))
        .ok_or_else(|| format!("{flag} must be a whole number from {min} to {max}, not {value:?}"))
}

/// Parse a `--stall-timeout` value as whole seconds inside the useful range.
///
/// The range is checked here rather than at the point of use because both ends
/// are only discoverable by letting the server run: zero abandons every slow
/// generation, and an unbounded value leaves one wedged client owning a batch
/// slot for as long as it cares to hold the socket open.
fn stall_timeout(value: &str) -> Result<Duration, String> {
    let seconds: i64 = value.parse().map_err(|_| {
        format!("invalid --stall-timeout: {value:?} is not a whole number of seconds")
    })?;
    if !(MIN_STALL_SECONDS..=MAX_STALL_SECONDS).contains(&seconds) {
        return Err(format!(
            "--stall-timeout must be between {MIN_STALL_SECONDS} and {MAX_STALL_SECONDS} \
             seconds, not {seconds}"
        ));
    }
    Ok(Duration::from_secs(seconds.unsigned_abs()))
}
