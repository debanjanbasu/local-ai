use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{FrameEntry, FrameKind, ShuffleMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OutputPolicy {
    pub create_backup: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
pub struct ArchiveOutputPlan {
    final_path: PathBuf,
    partial_path: PathBuf,
    manifest_path: PathBuf,
    backup_path: Option<PathBuf>,
}

impl ArchiveOutputPlan {
    #[must_use]
    pub fn new(final_path: &Path, policy: OutputPolicy) -> Self {
        let partial_path = append_suffix(final_path, ".partial");
        let manifest_path = append_suffix(final_path, ".manifest.json");
        let backup_path = policy
            .create_backup
            .then(|| append_suffix(final_path, ".bak"));
        Self {
            final_path: final_path.to_path_buf(),
            partial_path,
            manifest_path,
            backup_path,
        }
    }

    pub fn prepare(&self) -> crate::Result<()> {
        if let Some(parent) = self.final_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| crate::Error::Io(format!("create {}: {e}", parent.display())))?;
        }
        Ok(())
    }

    #[must_use]
    pub fn final_path(&self) -> &Path {
        &self.final_path
    }

    #[must_use]
    pub fn partial_path(&self) -> &Path {
        &self.partial_path
    }

    #[must_use]
    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    #[must_use]
    pub fn backup_path(&self) -> Option<&Path> {
        self.backup_path.as_deref()
    }

    pub fn publish(&self) -> crate::Result<()> {
        if let Some(backup_path) = &self.backup_path {
            if self.final_path.exists() {
                if backup_path.exists() {
                    std::fs::remove_file(backup_path).map_err(|e| {
                        crate::Error::Io(format!("remove {}: {e}", backup_path.display()))
                    })?;
                }
                std::fs::rename(&self.final_path, backup_path).map_err(|e| {
                    crate::Error::Io(format!(
                        "rename {} -> {}: {e}",
                        self.final_path.display(),
                        backup_path.display()
                    ))
                })?;
            }
        } else if self.final_path.exists() {
            std::fs::remove_file(&self.final_path).map_err(|e| {
                crate::Error::Io(format!("remove {}: {e}", self.final_path.display()))
            })?;
        }

        std::fs::rename(&self.partial_path, &self.final_path).map_err(|e| {
            crate::Error::Io(format!(
                "rename {} -> {}: {e}",
                self.partial_path.display(),
                self.final_path.display()
            ))
        })?;

        if self.manifest_path.exists() {
            std::fs::remove_file(&self.manifest_path).map_err(|e| {
                crate::Error::Io(format!("remove {}: {e}", self.manifest_path.display()))
            })?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResumeManifest {
    pub archive_version: u32,
    pub frames: Vec<ManifestFrameRecord>,
}

impl Default for ResumeManifest {
    fn default() -> Self {
        Self {
            archive_version: super::VERSION,
            frames: Vec::new(),
        }
    }
}

impl ResumeManifest {
    pub fn load_or_default(path: &Path) -> crate::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(path)
            .map_err(|e| crate::Error::Io(format!("read {}: {e}", path.display())))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| crate::Error::InvalidFormat(format!("manifest {}: {e}", path.display())))
    }

    pub fn save(&self, path: &Path) -> crate::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|e| crate::Error::InvalidFormat(format!("manifest encode: {e}")))?;
        std::fs::write(path, bytes)
            .map_err(|e| crate::Error::Io(format!("write {}: {e}", path.display())))
    }

    #[must_use]
    pub fn contains(&self, kind: FrameKind, id: u32) -> bool {
        self.frames
            .iter()
            .any(|frame| frame.kind == kind as u8 && frame.id == id)
    }

    pub fn record_entry(&mut self, entry: &FrameEntry) {
        if self.contains(entry.kind, entry.id) {
            return;
        }
        self.frames.push(ManifestFrameRecord::from_entry(entry));
    }

    #[must_use]
    pub fn frame_entries(&self) -> Vec<FrameEntry> {
        self.frames
            .iter()
            .filter_map(ManifestFrameRecord::to_entry)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFrameRecord {
    pub kind: u8,
    pub shuffle_mode: u8,
    pub quant_type_raw: u8,
    pub id: u32,
    pub file_offset: u64,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
}

impl ManifestFrameRecord {
    const fn from_entry(entry: &FrameEntry) -> Self {
        Self {
            kind: entry.kind as u8,
            shuffle_mode: entry.shuffle_mode as u8,
            quant_type_raw: entry.quant_type_raw,
            id: entry.id,
            file_offset: entry.file_offset,
            compressed_size: entry.compressed_size,
            uncompressed_size: entry.uncompressed_size,
        }
    }

    fn to_entry(&self) -> Option<FrameEntry> {
        Some(FrameEntry {
            kind: FrameKind::from_u8(self.kind)?,
            shuffle_mode: ShuffleMode::from_u8(self.shuffle_mode).unwrap_or(ShuffleMode::None),
            quant_type_raw: self.quant_type_raw,
            id: self.id,
            file_offset: self.file_offset,
            compressed_size: self.compressed_size,
            uncompressed_size: self.uncompressed_size,
        })
    }
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let file_name = path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    path.with_file_name(format!("{file_name}{suffix}"))
}
