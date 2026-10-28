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

/// Where the 4-bit drafter is installed, beside the int8 one; discovery
/// prefers it when it is there.
#[cfg(not(test))]
pub const DEFAULT_BONSAI_DFLASH_INT4_ARTIFACT: &str =
    "models/bonsai2-27b-dflash2/dflash2-drafter-int4-v1.bin";
#[cfg(test)]
pub const DEFAULT_BONSAI_DFLASH_INT4_ARTIFACT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/bonsai2-27b-dflash2/dflash2-drafter-int4-v1.bin"
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

/// Payload digest of the pinned 4-bit drafter.
///
/// It is the 4-bit conversion of the pinned int8 drafter above (`local-ai
/// bonsai --export dflash-head-int4=DIR` with only that artifact in `DIR`;
/// its matrices are dequantized exactly first).
pub const PINNED_DFLASH_INT4_SHA256: &str =
    "0b875f62c3550cd6c57245f8837fb4268340748343a6ac8078ffbac9ce5a0c96";

/// How a caller wants `KFlash` drafting. The drafter is required: `Auto`
/// drafts with the installed artifact (the 4-bit one when it is there) and
/// refuses to start without one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DflashMode {
    /// Use the drafter at the default location, the 4-bit one when it is
    /// installed; the absence of both is an error.
    Auto,
    /// Use this artifact; its absence is an error.
    Drafter(PathBuf),
    /// Never draft with `DFlash2`, for this reason (the plain-decoding
    /// reference of developer tools and tests; no command-line flag).
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
        self.resolve_with_default(&installed())
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

/// The installed drafter `Auto` resolves to: the 4-bit artifact when it is
/// installed, the int8 one otherwise (whether or not that exists).
#[must_use]
pub fn installed() -> PathBuf {
    let int4 = Path::new(DEFAULT_BONSAI_DFLASH_INT4_ARTIFACT);
    if int4.is_file() {
        int4.to_owned()
    } else {
        PathBuf::from(DEFAULT_BONSAI_DFLASH_ARTIFACT)
    }
}
