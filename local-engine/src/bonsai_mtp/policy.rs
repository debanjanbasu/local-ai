use std::path::{Path, PathBuf};

use super::cache::MTP_HEAD_ARTIFACT;

/// Native drafts per round, each verify row costing 30–45 % of a single-row
/// pass: the PTQ1 decode is shared, and the extra row adds one FMA per weight.
///
/// A draft earns its cost when the head's chained next token is accepted often
/// enough. Measured on the M4 Pro with lossless F16 caches (greedy, tokens
/// identical at every depth): unconditional depth 2 beat depth 1 by 1–9 % on
/// three parity cases and lost 11 % on the fourth (second-draft acceptance
/// 26 %); with the margin gate below it is ahead of depth 1 on the three short
/// cases and within noise on the 12K one.
///
/// Once the int8 head made drafting 20–30 % cheaper, gated depth 3 beat depth 2
/// on arithmetic, code, explanation and essay prompts (+1–3 %, lower GPU time
/// per token, identical tokens). That is the measurement behind the 3 here.
pub const DEFAULT_MTP_DEPTH: usize = 3;
pub const MAX_MTP_DEPTH: usize = 4;
/// The head drafts a further token only while its current proposal leads
/// the runner-up by at least this many logits. A deeper draft chains the
/// head's own hidden for the token it just proposed, so its acceptance
/// tracks the head's confidence in that token: below a lead of 1 the second
/// draft was accepted in 10–35 % of rounds, above 4 in 71–97 %. Rounds the
/// gate shortens show up as `proposed_tokens < rounds * depth`.
pub const DRAFT_CHAIN_MIN_MARGIN: f32 = 4.0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtpSettings {
    pub path: PathBuf,
    pub depth: usize,
    pub head_cache_dir: Option<PathBuf>,
}

impl MtpSettings {
    pub fn new(path: PathBuf, depth: usize) -> crate::Result<Self> {
        if !(1..=MAX_MTP_DEPTH).contains(&depth) {
            return Err(crate::Error::InvalidArgument(format!(
                "MTP draft depth must be within 1..={MAX_MTP_DEPTH}"
            )));
        }
        Ok(Self {
            path,
            depth,
            head_cache_dir: None,
        })
    }

    #[must_use]
    pub fn with_head_cache(mut self, directory: Option<PathBuf>) -> Self {
        self.head_cache_dir = directory;
        self
    }
}

/// Where the pinned community head is installed by default, beside the
/// pinned target checkpoint directory.
///
/// Shipped builds keep this repository-relative, and `MtpMode::resolve` reads
/// it as-is, so `Auto` speculation finds the head through a relative `./models`
/// beside the running binary. Test fixtures need the opposite:
/// `cargo test -p local-engine` runs with `local-engine/` as the working
/// directory, so the same relative string cannot resolve. The test build
/// therefore anchors the identical file to the workspace root, where nothing
/// joins it onto a working directory before opening it.
#[cfg(not(test))]
pub const DEFAULT_BONSAI_MTP_HEAD: &str = "models/bonsai2-27b-mtp/model_mtp.safetensors";
#[cfg(test)]
pub const DEFAULT_BONSAI_MTP_HEAD: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/bonsai2-27b-mtp/model_mtp.safetensors"
);

/// Where the shipped int8 head is installed by default, beside
/// [`DEFAULT_BONSAI_MTP_HEAD`] and with the same shipped/test anchoring.
///
/// Discovery prefers it: it loads without mapping the 849 MB BF16 source, and it
/// carries its own payload digest, so an install needs only this file.
#[cfg(not(test))]
pub const DEFAULT_BONSAI_MTP_ARTIFACT: &str = "models/bonsai2-27b-mtp/mtp-head-int8-v2.bin";
#[cfg(test)]
pub const DEFAULT_BONSAI_MTP_ARTIFACT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/bonsai2-27b-mtp/mtp-head-int8-v2.bin"
);

/// [`MTP_HEAD_ARTIFACT`] in the directory holding `head`.
///
/// Every head sits in its own directory, so the artifact is a sibling of
/// whichever head path is in play — the pinned default, or one the caller named.
#[must_use]
pub fn artifact_beside(head: &Path) -> Option<PathBuf> {
    head.parent()
        .map(|directory| directory.join(MTP_HEAD_ARTIFACT))
}

/// Whether `MtpMode::Auto` speculates when a head is available.
///
/// Greedy tokens were identical with and without the head on every measured
/// case; only the cost moved. On the M4 Pro the gated depth-3 default
/// decodes at 20–26 tok/s against 17.5 plain. Speculation is therefore on
/// by default; `bonsai --no-speculation` opts out for A/B runs.
///
/// Only the [`MtpMode::Auto`] arm reads this, and
/// [`Resources::discover`](crate::resources::Resources::discover) never builds an
/// `Auto`, so no shipped binary is decided by it: discovery takes an explicit
/// `speculation` argument and answers with a head or with `Off`. It is the
/// embedder's constant — what [`MtpMode::default`] follows, and what
/// [`MtpMode::resolve`] hands [`MtpMode::resolve_with_default`] as `auto_enabled`.
pub const DEFAULT_MTP_ENABLED: bool = true;

/// The record an `Off` mode falls back to when its constructor recorded none.
///
/// `Off` carries its own reason, so a caller that knows why it turned speculation
/// off — `Resources::discover` does, for both an explicit opt-out and a missing
/// install — says so instead of landing here. The request is the remaining way to
/// reach `Off`, and `--no-speculation` is the flag that asks for it: the CLI
/// rejects `--no-mtp` as an unknown option, so naming that would name a flag no
/// user can pass.
pub const MTP_OFF_REASON: &str = "speculation disabled by request (--no-speculation)";

/// Why `Auto` leaves speculation disabled when the build default is off.
///
/// No flag re-enables it. `--no-speculation` is the only one that reaches the
/// policy at all, and it asks for `Off`, which reports [`MTP_OFF_REASON`]
/// instead; the CLI rejects `--mtp` as an unknown option, so naming that would
/// name a flag no user can pass. [`DEFAULT_MTP_ENABLED`] is a compile-time
/// constant, so turning it back on means a build that sets it true, or naming
/// the head directly with [`MtpMode::Head`], which skips this policy.
///
/// Only the [`MtpMode::Auto`] arm reports this text, and only when
/// `auto_enabled` is false: so the reachers are an embedder calling
/// [`MtpMode::resolve_with_default`] with `false`, and [`MtpMode::resolve`] in a
/// build configured off. The shipping CLI reaches neither, because discovery
/// hands the engine an explicit [`MtpMode::Head`] or [`MtpMode::Off`]. Reaching
/// it through `Auto` is doubly guarded anyway: a head or artifact must be
/// installed, and `auto_enabled` must be false.
pub const MTP_DEFAULT_OFF_REASON: &str = "speculation is off by build default (DEFAULT_MTP_ENABLED); no flag re-enables it, so this needs a build with DEFAULT_MTP_ENABLED = true";

/// How a caller wants speculation.
///
/// `Auto` follows [`DEFAULT_MTP_ENABLED`] when a head is installed, preferring the
/// int8 artifact over the BF16 source. `Head` names a head file, which may be
/// either. `Off` never speculates and carries why, so the caller that built it —
/// which is the one that knows — is the only one that has to supply a reason.
/// Drafts never change the emitted tokens because the target verifies every one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MtpMode {
    /// Speculate from whichever head is found.
    ///
    /// This is the library-facing default — [`MtpMode::default`] returns it — and
    /// [`Resources::discover`](crate::resources::Resources::discover) never does:
    /// discovery resolves a head or reports `Off`, so the shipping CLI never
    /// resolves through this arm. An embedder who wants automatic behaviour
    /// should call `Resources::discover`; one who already knows the head can pass
    /// [`MtpMode::Head`] and skip discovery. The variant stays because it is
    /// public API and the default for a caller that never discovered anything.
    Auto {
        depth: usize,
    },
    Head(MtpSettings),
    Off(Option<String>),
}

impl Default for MtpMode {
    fn default() -> Self {
        Self::Auto {
            depth: DEFAULT_MTP_DEPTH,
        }
    }
}

/// Outcome of resolving an [`MtpMode`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MtpResolution {
    /// Load this head.
    Native(MtpSettings),
    /// Plain decoding, with the reason reported in the policy JSON.
    Disabled(String),
}

impl MtpMode {
    /// Decide whether to speculate.
    ///
    /// Every engine construction calls this with the mode it was handed, so it
    /// runs on the CLI as well — but with a mode
    /// [`Resources::discover`](crate::resources::Resources::discover) built, which
    /// is never [`MtpMode::Auto`]. The `Auto` arm here is therefore reached by a
    /// caller that asked for `Auto` or took [`MtpMode::default`].
    pub fn resolve(&self) -> crate::Result<MtpResolution> {
        self.resolve_with_default(Path::new(DEFAULT_BONSAI_MTP_HEAD), DEFAULT_MTP_ENABLED)
    }

    /// [`Self::resolve`] with an explicit `Auto` head location and policy.
    pub fn resolve_with_default(
        &self,
        default_head: &Path,
        auto_enabled: bool,
    ) -> crate::Result<MtpResolution> {
        match self {
            // A reason the constructor recorded is the record; `None` means the
            // caller had none to give, and the generic opt-out text stands in.
            Self::Off(carried) => Ok(MtpResolution::Disabled(
                carried.clone().unwrap_or_else(|| MTP_OFF_REASON.into()),
            )),
            Self::Head(settings) => {
                if !settings.path.is_file() {
                    return Err(crate::Error::InvalidArgument(format!(
                        "MTP head not found: {}",
                        settings.path.display()
                    )));
                }
                Ok(MtpResolution::Native(settings.clone()))
            }
            Self::Auto { depth } => {
                let mut settings = MtpSettings::new(default_head.to_path_buf(), *depth)?;
                // An int8 artifact beside the default head wins over it: it needs
                // neither the BF16 source nor any transform to load.
                match artifact_beside(default_head).filter(|path| path.is_file()) {
                    Some(artifact) => settings.path = artifact,
                    None if !default_head.is_file() => {
                        return Ok(MtpResolution::Disabled(format!(
                            "no MTP head or int8 artifact installed at {}",
                            default_head.display()
                        )));
                    }
                    None => {}
                }
                if auto_enabled {
                    Ok(MtpResolution::Native(settings))
                } else {
                    Ok(MtpResolution::Disabled(MTP_DEFAULT_OFF_REASON.into()))
                }
            }
        }
    }
}
