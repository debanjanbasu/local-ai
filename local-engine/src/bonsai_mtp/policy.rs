use std::path::{Path, PathBuf};

use super::cache::MTP_HEAD_ARTIFACT;

/// Native drafts per round. Each verify row costs 30–45 % of a single-row
/// pass (the PTQ1 decode is shared; the extra row adds one FMA per weight),
/// so a second draft pays off when the head's chained second token is
/// accepted often enough. Measured on the M4 Pro with lossless F16 caches
/// (greedy, tokens identical at every depth): unconditional depth 2 beat
/// depth 1 by 1–9 % on three parity cases and lost 11 % on the fourth
/// (second-draft acceptance 26 %); with the margin gate below it is ahead
/// of depth 1 on the three short cases and within noise on the 12K one.
/// Once the int8 head made drafting 20–30 % cheaper, gated depth 3 beat
/// depth 2 on arithmetic, code, explanation and essay prompts (+1–3 %,
/// lower GPU time per token, identical tokens).
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
pub const DEFAULT_MTP_ENABLED: bool = true;

/// Why an `Off` mode leaves speculation disabled, for the policy record.
pub const MTP_OFF_REASON: &str = "speculation disabled by request (--no-mtp)";

/// Why `Auto` leaves speculation disabled when the build default is off.
pub const MTP_DEFAULT_OFF_REASON: &str =
    "speculation is off by build default (DEFAULT_MTP_ENABLED); pass --mtp to enable";

/// Why an `Off` mode leaves speculation disabled when no head is installed.
///
/// `Resources::discover` builds `Off` both for an explicit opt-out and for a
/// missing install, so [`MTP_OFF_REASON`] covers only half of what `Off` means.
/// This names the absence, which holds whichever way `Off` was reached, and
/// names the request as the other way to the same state rather than claiming it
/// happened.
pub const MTP_ABSENT_REASON: &str = "speculation is off: no MTP head or int8 artifact installed beside the model, or the request disabled it";

/// How a caller wants speculation.
///
/// `Auto` follows [`DEFAULT_MTP_ENABLED`] when a head is installed, preferring the
/// int8 artifact over the BF16 source. `Head` names a head file, which may be
/// either. `Off` never speculates. Drafts never change the emitted tokens because
/// the target verifies every one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MtpMode {
    Auto { depth: usize },
    Head(MtpSettings),
    Off,
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

/// The head `crate::Resources::discover` looks for beside `model`.
///
/// Discovery resolves `models/bonsai2-27b-mtp/model_mtp.safetensors` next to the
/// model directory, so an engine opened on an explicit `--model` path has to ask
/// about that same place and not about the default under the working directory.
/// Both names are read back out of [`DEFAULT_BONSAI_MTP_HEAD`], leaving the
/// pinned layout named once; a model with no directory to look beside yields
/// `None`, which discovery accounts for as no head either.
fn head_beside_model(model: &Path) -> Option<PathBuf> {
    let default = Path::new(DEFAULT_BONSAI_MTP_HEAD);
    let root = model.parent().and_then(Path::parent)?;
    Some(
        root.join(default.parent()?.file_name()?)
            .join(default.file_name()?),
    )
}

impl MtpMode {
    /// Decide whether to speculate.
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
            Self::Off => Ok(MtpResolution::Disabled(MTP_OFF_REASON.into())),
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

    /// The reason to record for an `Off` mode whose model is `model`.
    ///
    /// `None` for every other mode, which keeps the reason its own resolution
    /// produced: `Auto`'s missing-head and build-default messages, and `Head`'s
    /// missing-file error, stay exactly as [`Self::resolve`] words them.
    ///
    /// An `Off` mode carries no record of which way it was reached, so
    /// [`Self::resolve`] can only name one of them. A head installed beside
    /// `model` is the discriminator: discovery would have speculated on it, so
    /// an opt-out is then the only explanation left. Its absence is the other
    /// explanation, and is reported as the fact it is.
    #[must_use]
    pub fn off_reason(&self, model: &Path) -> Option<String> {
        if !matches!(self, Self::Off) {
            return None;
        }
        Some(match head_beside_model(model) {
            Some(head)
                if head.is_file() || artifact_beside(&head).is_some_and(|path| path.is_file()) =>
            {
                MTP_OFF_REASON.into()
            }
            _ => MTP_ABSENT_REASON.into(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    /// A model at the pinned install path, so discovery would look for the head
    /// beside it exactly as it does for the shipped layout.
    fn model(root: &Path) -> PathBuf {
        root.join("models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    }

    /// Put `file` where the pinned default layout puts a head beside `model`.
    fn install(model: &Path, file: &str) {
        let directory = model
            .parent()
            .and_then(Path::parent)
            .expect("model root")
            .join(
                Path::new(DEFAULT_BONSAI_MTP_HEAD)
                    .parent()
                    .and_then(Path::file_name)
                    .expect("head directory"),
            );
        std::fs::create_dir_all(&directory).expect("head directory");
        std::fs::write(directory.join(file), b"stub").expect("stub head");
    }

    /// An `Off` mode is reached two ways, and the record must not confuse them.
    ///
    /// `Resources::discover` builds `Off` for an explicit `--no-speculation` and
    /// again when it finds no head, so a record that always names the request
    /// claims one that never happened.
    #[test]
    fn off_names_the_request_only_when_a_head_is_installed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = model(dir.path());

        // Nothing installed: the discoverer chose `Off` on its own.
        let absent = MtpMode::Off.off_reason(&path);
        assert_ne!(absent.as_deref(), Some(MTP_OFF_REASON), "{absent:?}");
        assert!(
            absent
                .as_deref()
                .is_some_and(|why| why.contains("no MTP head")),
            "{absent:?}"
        );

        // A source head beside the model means discovery would have speculated,
        // so `Off` can only be the request.
        install(&path, "model_mtp.safetensors");
        assert_eq!(
            MtpMode::Off.off_reason(&path).as_deref(),
            Some(MTP_OFF_REASON)
        );

        // An artifact-only install counts too: discovery prefers it and
        // speculates without the 849 MB source.
        let artifact_only = tempfile::tempdir().expect("tempdir");
        let bare = model(artifact_only.path());
        install(&bare, MTP_HEAD_ARTIFACT);
        assert_eq!(
            MtpMode::Off.off_reason(&bare).as_deref(),
            Some(MTP_OFF_REASON)
        );

        // Every other mode keeps the reason its own resolution produced, so
        // `Auto`'s missing-head and build-default messages, and `Head`'s missing
        // file error, stay as they were.
        assert!(
            MtpMode::Auto {
                depth: DEFAULT_MTP_DEPTH
            }
            .off_reason(&path)
            .is_none()
        );
        assert!(
            MtpMode::Head(MtpSettings::new(path.clone(), 1).expect("settings"))
                .off_reason(&path)
                .is_none()
        );
    }
}
