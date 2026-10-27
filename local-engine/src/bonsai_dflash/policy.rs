use std::path::{Path, PathBuf};

/// Where the drafter artifact is installed, beside the pinned target
/// checkpoint directory.
///
/// Discovery looks for exactly this file; the test
/// build anchors it to the workspace root, as the MTP head's path is.
#[cfg(not(test))]
pub const DEFAULT_BONSAI_DFLASH_ARTIFACT: &str =
    "models/bonsai2-27b-dflash2/dflash2-drafter-int8-v1.bin";
#[cfg(test)]
pub const DEFAULT_BONSAI_DFLASH_ARTIFACT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/bonsai2-27b-dflash2/dflash2-drafter-int8-v1.bin"
);

/// Payload digest of the pinned drafter.
///
/// It is `ProCreations/Ternary-Bonsai-2-27B-DFlash2` at revision `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b` (`model.safetensors`
/// SHA256 `708e141bf4a33b09620c4f86263f1c7edcec7882f5dd39448e83e0c0defdc230`),
/// converted by `local-ai bonsai --export dflash-head=DIR`. Any converted
/// `DFlash2` checkpoint of this architecture loads; the policy reports
/// whether the installed one is this.
pub const PINNED_DFLASH_SHA256: &str =
    "1b5dc3551406e677ec9f4321f8f5c20f7310eaae0cede95792ceaa2375dfc02a";

/// How a caller wants `KFlash` drafting. The drafter is required: `Auto`
/// drafts with the installed artifact and refuses to start without it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DflashMode {
    /// Use the drafter at the default location; its absence is an error.
    Auto,
    /// Use this artifact; its absence is an error.
    Drafter(PathBuf),
    /// Never draft with `DFlash2`, for this reason (for example
    /// `--no-speculation`).
    Off(String),
}

/// Outcome of resolving a [`DflashMode`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DflashResolution {
    /// Load this artifact with [`super::DflashDrafter::load`]; a corrupt
    /// file is then a hard error.
    Drafter(PathBuf),
    /// Do not draft with `DFlash2`, for the reason an [`DflashMode::Off`]
    /// carried.
    Fallback(String),
}

impl DflashMode {
    pub fn resolve(&self) -> crate::Result<DflashResolution> {
        self.resolve_with_default(Path::new(DEFAULT_BONSAI_DFLASH_ARTIFACT))
    }

    /// [`Self::resolve`] with an explicit `Auto` location.
    pub fn resolve_with_default(&self, default: &Path) -> crate::Result<DflashResolution> {
        let path = match self {
            Self::Off(reason) => return Ok(DflashResolution::Fallback(reason.clone())),
            Self::Drafter(path) => path.as_path(),
            Self::Auto => default,
        };
        if path.is_file() {
            Ok(DflashResolution::Drafter(path.to_owned()))
        } else {
            Err(crate::Error::InvalidArgument(format!(
                "DFlash2 drafter not found: {} (required; see README Model install)",
                path.display()
            )))
        }
    }
}
