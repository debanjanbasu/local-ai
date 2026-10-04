//! Startup resource discovery and automatic policy choices.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::bonsai::DEFAULT_BONSAI_GGUF;
use crate::bonsai_mtp::{DEFAULT_MTP_DEPTH, MTP_HEAD_ARTIFACT, MtpMode, MtpSettings};

const MODEL_RELATIVE: &str = "models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf";
const MTP_RELATIVE: &str = "bonsai2-27b-mtp/model_mtp.safetensors";
pub const PREFILL_CHUNK: usize = 128;
pub const SERVE_QUEUE: usize = 8;
/// Ceiling on the prompt-cache disk budget.
///
/// A disk snapshot stores the full prompt state, so its size scales with the
/// selected context rather than staying small: at a 45k-token context one
/// snapshot is roughly 1.7 GB. A 32 GiB ceiling therefore held under twenty
/// sessions, which evicted checkpoints long before the disk ran short, so the
/// ceiling was raised to 512 GiB, which holds about 290.
///
/// The free-space fraction in `Resources::discover` is the real governor; this
/// ceiling only stops a pathological report from authorising an unbounded
/// budget.
pub const DISK_CACHE_CAP_BYTES: u64 = 512 * 1024 * 1024 * 1024;

#[derive(Debug)]
pub struct Resources {
    pub model: PathBuf,
    pub model_reason: String,
    pub mtp: MtpMode,
    pub mtp_reason: String,
    /// The BF16 safetensors head beside the model, if one is installed. Exporting
    /// an int8 artifact needs it; loading an artifact does not.
    pub mtp_source: Option<PathBuf>,
    pub mtp_head_cache_dir: Option<PathBuf>,
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
        let mtp_root = model.parent().and_then(Path::parent);
        let source = mtp_root.map(|root| root.join(MTP_RELATIVE));
        // The artifact is the head's sibling by construction, so the two can never
        // drift onto different directories.
        let artifact = source
            .as_ref()
            .map(|head| head.with_file_name(MTP_HEAD_ARTIFACT));
        let mtp_source = source.filter(|path| path.is_file());
        let mtp_artifact = artifact.filter(|path| path.is_file());
        let mtp_cache = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("Library/Caches/local-ai/mtp-head"))
            .filter(|path| writable_directory(path).is_ok());
        // The artifact first: it loads without the 849 MB source, and an invalid
        // one is reported by the loader instead of quietly costing throughput.
        let (mtp, mtp_reason) = if !speculation {
            let reason = "disabled by --no-speculation";
            (MtpMode::Off(Some(reason.into())), reason.into())
        } else if let Some((path, reason)) =
            describe_head(mtp_artifact.as_ref(), mtp_source.as_ref())
        {
            (
                MtpMode::Head(
                    MtpSettings::new(path, mtp_depth)?.with_head_cache(mtp_cache.clone()),
                ),
                reason,
            )
        } else {
            // Only what discovery can see: whether suffix/ngram lookup is on is
            // decided by `--no-speculation`, which took the branch above.
            let reason = "no MTP head or int8 artifact installed beside the model";
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
                        let free = free_disk_bytes(&path).unwrap_or(0);
                        let budget = DISK_CACHE_CAP_BYTES.min(free / 4);
                        let rate = measure_write_rate(&path);
                        (
                            Some(path),
                            budget,
                            format!(
                                "min({} GiB, 25% of {free} free bytes)",
                                DISK_CACHE_CAP_BYTES / (1024 * 1024 * 1024)
                            ),
                            rate,
                        )
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
            mtp_source,
            mtp_head_cache_dir: mtp_cache,
            prompt_cache_dir,
            disk_budget_bytes,
            disk_reason,
            prompt_cache_write_bytes_per_second: cache_write_rate,
            tls,
            tls_reason,
            physical_memory_bytes: physical_memory_bytes(),
        })
    }

    pub fn policy(&self) -> serde_json::Value {
        serde_json::json!({
            "resources": {
                "model": self.model,
                "model_reason": self.model_reason,
                "mtp_reason": self.mtp_reason,
                "mtp_head_source": self.mtp_source,
                "mtp_head_cache_directory": self.mtp_head_cache_dir,
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

/// Name the head that will load and say what it is.
///
/// Discovery can only see that these paths are files; the loader classifies them
/// by content, so the reason reports what is installed rather than claiming a
/// winner it has not checked.
fn describe_head(
    artifact: Option<&PathBuf>,
    source: Option<&PathBuf>,
) -> Option<(PathBuf, String)> {
    match (artifact, source) {
        (Some(artifact), Some(source)) => Some((
            artifact.clone(),
            format!(
                "int8 MTP head artifact {}; {} kept as the rebuild source",
                artifact.display(),
                source.display()
            ),
        )),
        (Some(artifact), None) => Some((
            artifact.clone(),
            format!(
                "int8 MTP head artifact {}; no BF16 source installed",
                artifact.display()
            ),
        )),
        (None, Some(source)) => Some((
            source.clone(),
            format!(
                "BF16 MTP head {}; quantized once and cached",
                source.display()
            ),
        )),
        (None, None) => None,
    }
}

fn discover_model(override_path: Option<&Path>) -> crate::Result<(PathBuf, String)> {
    if let Some(path) = override_path {
        return path
            .is_file()
            .then(|| (path.to_owned(), "explicit --model override".into()))
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
                "pinned Bonsai model not found; install it under ./models or pass --model PATH"
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
