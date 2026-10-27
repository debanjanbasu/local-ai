use std::path::PathBuf;
use std::time::Duration;

use super::Args;

/// Where Responses are stored.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ResponseStoreArg {
    /// The user's data directory, pruned of responses older than 30 days.
    Default,
    /// `--response-store DIR`, kept until deleted.
    Dir(PathBuf),
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
    eprintln!(
        "  --response-store DIR  Keep Responses API responses in DIR (owner-only, \
         unencrypted) until deleted (default: ~/Library/Application Support/local-ai/responses, \
         where responses older than 30 days are removed at startup)"
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
        "  Hosted tools (Responses API; see docs/TOOLS.md) are always on: file access and a \
         sandboxed shell with writes, rooted at the working directory, and web fetch of any \
         public host. A non-loopback --host therefore needs authentication (--api-key or \
         `local-ai admin create-key`)."
    );
}

pub(super) fn parse(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        host: "127.0.0.1".parse().map_err(|_| "invalid default host")?,
        port: 8080,
        api_key: None,
        stall: Duration::from_secs(DEFAULT_STALL_SECONDS),
        response_store: ResponseStoreArg::Default,
        reasoning_key: None,
        allowed_origins: Vec::new(),
        allowed_hosts: Vec::new(),
    };
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if matches!(flag, "--help" | "-h") {
            return Err(String::new());
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
        ) {
            return Err(format!("unknown option: {flag}"));
        }
        let value = args
            .get(index)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--host" => parsed.host = value.parse().map_err(|_| "invalid --host")?,
            "--port" => parsed.port = value.parse().map_err(|_| "invalid --port")?,
            "--api-key" => {
                if value.is_empty() {
                    return Err("--api-key requires a non-empty key".into());
                }
                parsed.api_key = Some(value.clone());
            }
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
            _ => return Err(format!("unknown option: {flag}")),
        }
        index += 1;
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
