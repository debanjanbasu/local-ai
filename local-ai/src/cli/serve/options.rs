use std::time::Duration;

use super::Args;

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
    eprintln!("  --api-key KEY      Require bearer authentication");
    eprintln!("  --no-thinking      Disable thinking");
    eprintln!(
        "  --stall-timeout SECONDS  Drop a generation whose client stopped reading \
         (default: {DEFAULT_STALL_SECONDS})"
    );
}

pub(super) fn parse(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        host: "127.0.0.1".parse().map_err(|_| "invalid default host")?,
        port: 8080,
        thinking: true,
        api_key: None,
        stall: Duration::from_secs(DEFAULT_STALL_SECONDS),
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
        index += 1;
        // Recognise the flag before demanding its value, so an unknown option is
        // reported as unknown rather than as a missing argument.
        if !matches!(flag, "--host" | "--port" | "--api-key" | "--stall-timeout") {
            return Err(format!("unknown option: {flag}"));
        }
        let value = args
            .get(index)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--host" => parsed.host = value.parse().map_err(|_| "invalid --host")?,
            "--port" => parsed.port = value.parse().map_err(|_| "invalid --port")?,
            "--api-key" => parsed.api_key = Some(value.clone()),
            "--stall-timeout" => parsed.stall = stall_timeout(value)?,
            _ => return Err(format!("unknown option: {flag}")),
        }
        index += 1;
    }
    Ok(parsed)
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
