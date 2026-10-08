//! Startup resource discovery and automatic policy choices.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::bonsai::DEFAULT_BONSAI_GGUF;
use crate::bonsai_mtp::{DEFAULT_MTP_DEPTH, MTP_HEAD_ARTIFACT, MtpMode, MtpSettings};

const MODEL_RELATIVE: &str = "models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf";
/// The MTP head's directory under the models root.
const MTP_DIRECTORY: &str = "bonsai2-27b-mtp";
pub const PREFILL_CHUNK: usize = 128;
pub const SERVE_QUEUE: usize = 8;
/// Divisor on free disk in the prompt-cache disk budget, so a fuller disk
/// shrinks it.
const FREE_DISK_SHARE: u64 = 8;
/// Divisor on physical memory in the prompt-cache disk budget, leaving three
/// quarters of memory to page cache the demand-paged checkpoint.
const PHYSICAL_MEMORY_SHARE: u64 = 4;

#[derive(Debug)]
pub struct Resources {
    pub model: PathBuf,
    pub model_reason: String,
    pub mtp: MtpMode,
    pub mtp_reason: String,
    pub prompt_cache_dir: Option<PathBuf>,
    pub disk_budget_bytes: u64,
    pub disk_reason: String,
    pub prompt_cache_write_bytes_per_second: Option<u64>,
    pub tls: Option<(PathBuf, PathBuf)>,
    pub tls_reason: String,
    pub physical_memory_bytes: Option<u64>,
}

impl Resources {
    /// [`Self::discover`] at the built-in draft depth, which is what every
    /// caller that is not measuring speculation wants.
    pub fn discover(model_override: Option<&Path>, speculation: bool) -> crate::Result<Self> {
        Self::discover_with_depth(model_override, speculation, DEFAULT_MTP_DEPTH)
    }

    /// [`Self::discover`] with an explicit MTP draft depth, for measuring what
    /// the shipped depth leaves on the table.
    ///
    /// [`MtpSettings::new`] owns the accepted range, so a depth outside it is
    /// rejected here rather than quietly decoding at some other depth.
    pub fn discover_with_depth(
        model_override: Option<&Path>,
        speculation: bool,
        mtp_depth: usize,
    ) -> crate::Result<Self> {
        let (model, model_reason) = discover_model(model_override)?;
        // One probe, read by both the budget and the reported field: `sysctl`
        // is spawned once per discovery rather than once per consumer.
        let memory = physical_memory_bytes();
        let model_bytes = file_bytes(&model);
        let head = model
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join(MTP_DIRECTORY).join(MTP_HEAD_ARTIFACT))
            .filter(|path| path.is_file());
        // The one head is the trained ternary artifact beside the model. An
        // invalid one is reported by the loader instead of quietly costing
        // throughput. Every branch below answers with a named head or with
        // `Off`; discovery never constructs `MtpMode::Auto`, which stays the
        // default for an embedder that has not called this.
        let (mtp, mtp_reason) = if !speculation {
            let reason = "disabled by --no-speculation";
            (MtpMode::Off(Some(reason.into())), reason.into())
        } else if let Some(path) = head {
            let reason = format!("ternary PTQ1 MTP head artifact {}", path.display());
            (MtpMode::Head(MtpSettings::new(path, mtp_depth)?), reason)
        } else {
            // Only what discovery can see: whether suffix/ngram lookup is on is
            // decided by `--no-speculation`, which took the branch above.
            let reason = "no MTP head artifact installed beside the model";
            (MtpMode::Off(Some(reason.into())), reason.into())
        };
        let cache = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("Library/Caches/local-ai/prompt-cache"));
        let (prompt_cache_dir, disk_budget_bytes, disk_reason, cache_write_rate) = cache
            .map_or_else(
                || {
                    (
                        None,
                        0,
                        "HOME is unavailable; disk prompt cache disabled".into(),
                        None,
                    )
                },
                |path| match writable_directory(&path) {
                    Ok(()) => {
                        let (budget, reason) =
                            disk_cache_budget(model_bytes, free_disk_bytes(&path), memory);
                        let rate = measure_write_rate(&path);
                        (Some(path), budget, reason, rate)
                    }
                    Err(reason) => (None, 0, reason, None),
                },
            );
        let tls_dir = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("Library/Application Support/local-ai/tls"));
        let tls = tls_dir.as_ref().and_then(|dir| {
            let pair = (dir.join("cert.pem"), dir.join("key.pem"));
            (pair.0.is_file() && pair.1.is_file()).then_some(pair)
        });
        let tls_reason = tls.as_ref().map_or_else(
            || "certificate/key pair not found; HTTP/3 disabled".into(),
            |(cert, _)| format!("certificate/key pair discovered at {}", cert.display()),
        );
        Ok(Self {
            model,
            model_reason,
            mtp,
            mtp_reason,
            prompt_cache_dir,
            disk_budget_bytes,
            disk_reason,
            prompt_cache_write_bytes_per_second: cache_write_rate,
            tls,
            tls_reason,
            physical_memory_bytes: memory,
        })
    }

    pub fn policy(&self) -> serde_json::Value {
        serde_json::json!({
            "resources": {
                "model": self.model,
                "model_reason": self.model_reason,
                "mtp_reason": self.mtp_reason,
                "physical_memory_bytes": self.physical_memory_bytes,
                "prompt_cache_directory": self.prompt_cache_dir,
                "prompt_cache_disk_budget_bytes": self.disk_budget_bytes,
                "prompt_cache_disk_reason": self.disk_reason,
                "prompt_cache_measured_write_bytes_per_second": self.prompt_cache_write_bytes_per_second,
                "http3": self.tls.is_some(),
                "http3_reason": self.tls_reason,
                "serve_queue": SERVE_QUEUE,
                "prefill_chunk": PREFILL_CHUNK,
            }
        })
    }
}

fn discover_model(override_path: Option<&Path>) -> crate::Result<(PathBuf, String)> {
    if let Some(path) = override_path {
        return path
            .is_file()
            .then(|| (path.to_owned(), "explicit Engine::open_model path".into()))
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("model not found: {}", path.display()))
            });
    }
    let mut candidates = vec![
        PathBuf::from(DEFAULT_BONSAI_GGUF),
        PathBuf::from(MODEL_RELATIVE),
    ];
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join(MODEL_RELATIVE));
        candidates.push(dir.join("../models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join(
            "Library/Caches/local-ai/models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        ));
    }
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .map(|path| {
            let reason = format!("first installed pinned model candidate: {}", path.display());
            (path, reason)
        })
        .ok_or_else(|| {
            crate::Error::InvalidArgument(
                "pinned Bonsai model not found; install it under ./models, beside the \
                 executable, or under ~/Library/Caches/local-ai/models"
                    .into(),
            )
        })
}

fn writable_directory(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path)
        .map_err(|error| format!("disk prompt cache disabled: {error}"))?;
    let probe = path.join(".write-probe");
    std::fs::write(&probe, [])
        .and_then(|()| std::fs::remove_file(probe))
        .map_err(|error| format!("disk prompt cache disabled: {error}"))
}

fn command_number(program: &str, args: &[&str]) -> Option<u64> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().parse().ok())
        .flatten()
}

/// The prompt-cache disk budget, and the reason string that reports it.
///
/// The budget is the smallest of three terms derived from machine facts: the
/// checkpoint's own bytes, an eighth of free disk, and a quarter of physical
/// memory. Measured on this M2 — a 5.95 GB checkpoint, 643.7 GiB free, 16 GiB of
/// memory — the reason reads `4 GiB = min(model 5 GiB, free disk 642 GiB / 8 =
/// 80 GiB, physical memory 16 GiB / 4 = 4 GiB), bounded by a quarter of
/// physical memory`, which holds about 27 snapshots. A fixed 512 GiB ceiling
/// against a quarter of free disk allowed 161 GiB and about 1,100 instead.
///
/// The memory term is the one that protects what the cache exists to serve.
/// [`crate::bonsai::BonsaiPackage`] maps the checkpoint with
/// `map_copy_read_only`, so the checkpoint is demand-paged and decode needs its
/// pages resident: on a 16 GiB part a cache sized only against free disk evicts
/// exactly those pages. That term is what reserves page cache for the
/// checkpoint. The in-RAM tier keeps its own independent policy and is untouched
/// by this bound.
///
/// The unit the budget is really counting is the snapshot, not the context.
/// Measured on this M2 across all 71 real snapshots, spanning six contexts from
/// 45,720 to 66,810 tokens and 10.56 GiB in total, every `.bpc` file is
/// ~152 MiB: 151.2-153.4 MiB per context, and no file outside the 100-200 MiB
/// band. A snapshot's size is therefore close to independent of the context it
/// holds, which is why no term here reads the context length, and why a budget
/// in bytes converts to a snapshot count without a guess.
///
/// Every term is reported in the reason beside the one that won, so the number
/// cannot drift from the arithmetic that produced it.
fn disk_cache_budget(
    model_bytes: Option<u64>,
    free_bytes: Option<u64>,
    memory_bytes: Option<u64>,
) -> (u64, String) {
    let model = model_bytes.unwrap_or(u64::MAX);
    let model_text = readable_bytes(model);
    let (free, free_text) = bounded_term(free_bytes, FREE_DISK_SHARE);
    let (memory, memory_text) = bounded_term(memory_bytes, PHYSICAL_MEMORY_SHARE);
    let budget = model.min(free).min(memory);
    // `u64::MAX` is the sentinel a failed probe contributes, so a budget sitting
    // on it means nothing was readable rather than a term having genuinely won.
    let bound = if budget == u64::MAX {
        "no machine fact was readable".to_owned()
    } else if budget == model {
        format!("the checkpoint it accelerates ({model_text})")
    } else if budget == free {
        format!("an eighth of free disk ({free_text})")
    } else {
        format!("a quarter of physical memory ({memory_text})")
    };
    (
        budget,
        format!(
            "{} = min(model {model_text}, free disk {free_text}, physical memory {memory_text}), \
             bounded by {bound}",
            readable_bytes(budget)
        ),
    )
}

/// One divided term of [`disk_cache_budget`], and how it reads in the reason.
///
/// A probe that failed contributes [`u64::MAX`] rather than zero. A missing
/// `sysctl` or `df` must remove a limit, because reading it as "no budget" would
/// silently turn a failed probe into a disabled cache.
fn bounded_term(bytes: Option<u64>, share: u64) -> (u64, String) {
    let Some(known) = bytes else {
        let text = format!("unbounded (probe failed; would be / {share})");
        return (u64::MAX, text);
    };
    let bounded = known / share;
    (
        bounded,
        format!(
            "{} / {share} = {}",
            readable_bytes(known),
            readable_bytes(bounded)
        ),
    )
}

/// A byte count in gibibytes, falling back to mebibytes below one gibibyte so a
/// small budget still reads as a number rather than as `0 GiB`.
fn readable_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    if bytes == u64::MAX {
        // The sentinel an unreadable probe contributes; naming it as a count
        // would report a budget nobody measured.
        "unbounded".to_owned()
    } else if bytes >= GIB {
        format!("{} GiB", bytes / GIB)
    } else {
        format!("{} MiB", bytes / MIB)
    }
}

/// The size of a file, or `None` when it cannot be measured.
fn file_bytes(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|metadata| metadata.len())
}

fn physical_memory_bytes() -> Option<u64> {
    command_number("sysctl", &["-n", "hw.memsize"])
}

fn free_disk_bytes(path: &Path) -> Option<u64> {
    let output = std::process::Command::new("df")
        .args(["-Pk", path.to_str()?])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    text.lines()
        .nth(1)?
        .split_whitespace()
        .nth(3)?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

fn measure_write_rate(path: &Path) -> Option<u64> {
    const SAMPLE_BYTES: usize = 16 * 1024 * 1024;
    let probe = path.join(".throughput-probe");
    let block = vec![0_u8; 1024 * 1024];
    let started = Instant::now();
    let result = (|| {
        let mut file = std::fs::File::create(&probe).ok()?;
        for _ in 0..SAMPLE_BYTES / block.len() {
            std::io::Write::write_all(&mut file, &block).ok()?;
        }
        file.sync_all().ok()?;
        Some(())
    })();
    let elapsed = started.elapsed();
    let _ = std::fs::remove_file(probe);
    result?;
    u64::try_from(SAMPLE_BYTES)
        .ok()?
        .checked_mul(1_000_000_000)?
        .checked_div(u64::try_from(elapsed.as_nanos()).ok()?.max(1))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{FREE_DISK_SHARE, PHYSICAL_MEMORY_SHARE, disk_cache_budget, readable_bytes};

    const GIB: u64 = 1024 * 1024 * 1024;
    /// The measured facts of the machine these terms were derived on.
    const MODEL: u64 = 6 * GIB;
    const DISK: u64 = 644 * GIB;
    const MEMORY: u64 = 16 * GIB;

    #[test]
    fn the_budget_is_the_smallest_of_the_three_machine_terms() {
        // A small checkpoint bounds the cache below both fractions, and the
        // reason says so rather than leaving it to be inferred.
        let (budget, reason) = disk_cache_budget(Some(GIB), Some(DISK), Some(MEMORY));
        assert_eq!(budget, GIB);
        assert!(reason.contains("the checkpoint it accelerates"), "{reason}");

        // A fuller disk bounds it even against a large checkpoint and 16 GiB.
        let (budget, reason) = disk_cache_budget(Some(64 * GIB), Some(8 * GIB), Some(MEMORY));
        assert_eq!(budget, GIB);
        assert!(reason.contains("an eighth of free disk"), "{reason}");

        // With disk and memory both plentiful, the memory term is left to bound
        // it, which is the case this machine is actually in.
        let (budget, reason) = disk_cache_budget(Some(MODEL), Some(DISK), Some(MEMORY));
        assert_eq!(budget, MEMORY / PHYSICAL_MEMORY_SHARE);
        assert!(reason.contains("a quarter of physical memory"), "{reason}");
    }

    #[test]
    fn the_reason_reports_every_term_and_the_budget_it_derived() {
        let (budget, reason) = disk_cache_budget(Some(MODEL), Some(DISK), Some(MEMORY));
        assert!(reason.starts_with(&readable_bytes(budget)), "{reason}");
        assert!(reason.contains("model 6 GiB"), "{reason}");
        assert!(
            reason.contains(&format!(
                "free disk {} GiB / {FREE_DISK_SHARE} = {} GiB",
                DISK / GIB,
                DISK / FREE_DISK_SHARE / GIB
            )),
            "{reason}"
        );
        assert!(
            reason.contains(&format!(
                "physical memory 16 GiB / {PHYSICAL_MEMORY_SHARE} = 4 GiB"
            )),
            "{reason}"
        );
    }

    #[test]
    fn a_failed_probe_drops_its_limit_instead_of_zeroing_the_budget() {
        // Every probe missing leaves the cache unbounded, never disabled.
        let (budget, reason) = disk_cache_budget(None, None, None);
        assert_eq!(budget, u64::MAX);
        assert!(reason.contains("no machine fact was readable"), "{reason}");

        // One probe answering still bounds the budget, and the terms that failed
        // read as unbounded rather than as a zero that would beat it.
        let (budget, reason) = disk_cache_budget(None, None, Some(MEMORY));
        assert_eq!(budget, MEMORY / PHYSICAL_MEMORY_SHARE);
        assert!(reason.contains("model unbounded"), "{reason}");
        assert!(
            reason.contains(&format!(
                "free disk unbounded (probe failed; would be / {FREE_DISK_SHARE})"
            )),
            "{reason}"
        );

        let (budget, reason) = disk_cache_budget(Some(MODEL), None, None);
        assert_eq!(budget, MODEL);
        assert!(reason.contains("bounded by the checkpoint"), "{reason}");

        let (budget, _) = disk_cache_budget(Some(MODEL), Some(DISK), None);
        assert_eq!(budget, MODEL);
    }

    #[test]
    fn a_byte_count_reads_in_gibibytes_and_a_failed_probe_reads_as_unbounded() {
        assert_eq!(readable_bytes(4 * GIB), "4 GiB");
        assert_eq!(readable_bytes(152 * 1024 * 1024), "152 MiB");
        assert_eq!(readable_bytes(u64::MAX), "unbounded");
    }
}
