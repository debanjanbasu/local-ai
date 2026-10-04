use super::{BonsaiEngine, PROMPT_CHECKPOINT_BYTES, PathBuf, prompt_cache};
pub const DEFAULT_PROMPT_CACHE_CHECKPOINTS: usize = 4;
pub const MAX_PROMPT_CACHE_CHECKPOINTS: usize = 4;
const SSD_FIRST_WRITE_RATE: u64 = 500 * 1024 * 1024;

impl BonsaiEngine {
    pub fn set_session_cache(&mut self, megabytes: usize, directory: Option<PathBuf>) {
        self.prompt_cache_bytes = megabytes.saturating_mul(1024 * 1024);
        self.prompt_cache_dir = directory;
        if self.prompt_cache_dir.is_some() && self.prompt_cache_disk_bytes == 0 {
            self.prompt_cache_disk_bytes = u64::MAX;
        }
        // Snapshots carry MTP head state, so runs with and without the head
        // must not share files.
        self.prompt_cache_model_key = format!(
            "v1-{}-{}-{}-{}",
            self.info.precision.replace('/', "_"),
            self.info.context,
            self.model.kv_layout().name().replace('/', "_"),
            if self.model.speculation().is_some() {
                "mtp"
            } else {
                "nomtp"
            }
        );
        self.disk_snapshots = if let Some(dir) = &self.prompt_cache_dir {
            prompt_cache::discover(dir, &self.prompt_cache_model_key)
        } else {
            Vec::new()
        };
        self.info.policy["prompt_cache"]["memory_budget_bytes"] = self.prompt_cache_bytes.into();
        self.info.policy["prompt_cache"]["directory"] = self
            .prompt_cache_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .into();
    }

    /// Derive all prompt-cache tiers from headroom after the fully-grown model.
    pub fn configure_prompt_cache(
        &mut self,
        directory: Option<PathBuf>,
        disk_budget_bytes: u64,
        measured_write_bytes_per_second: Option<u64>,
    ) -> crate::Result<()> {
        let headroom = self
            .info
            .working_set_limit
            .saturating_sub(self.info.estimated_working_set);
        let checkpoints = usize::try_from(headroom / PROMPT_CHECKPOINT_BYTES as u64)
            .unwrap_or(MAX_PROMPT_CACHE_CHECKPOINTS)
            .min(MAX_PROMPT_CACHE_CHECKPOINTS);
        self.set_prompt_cache_checkpoints(checkpoints)?;
        let remaining = self
            .info
            .working_set_limit
            .saturating_sub(self.info.estimated_working_set);
        let disk_is_fast = directory.is_some()
            && disk_budget_bytes > 0
            && measured_write_bytes_per_second.is_some_and(|rate| rate >= SSD_FIRST_WRITE_RATE);
        let host_bytes = if disk_is_fast {
            0
        } else {
            (remaining / 4).min(4 * 1024 * 1024 * 1024)
        };
        self.set_session_cache((host_bytes / (1024 * 1024)) as usize, directory);
        self.prompt_cache_disk_bytes = disk_budget_bytes;
        self.info.policy["prompt_cache"]["host_tier"] = serde_json::json!({
            "enabled": host_bytes > 0,
            "budget_bytes": host_bytes,
            "measured_disk_write_bytes_per_second": measured_write_bytes_per_second,
            "ssd_first_threshold_bytes_per_second": SSD_FIRST_WRITE_RATE,
            "reason": if disk_is_fast {
                "disabled: measured prompt-cache storage exceeds the SSD-first threshold"
            } else {
                "enabled: prompt-cache storage was unavailable or below the SSD-first threshold"
            },
        });
        self.info.policy["prompt_cache"]["reason"] = serde_json::json!(
            "GPU checkpoints use memory headroom; host snapshots are skipped on measured fast storage; shared-prefix checkpoints start at one prefill chunk"
        );
        self.info.policy["prompt_cache"]["shared_prefix"] = serde_json::json!({
            "enabled": checkpoints > 0 || host_bytes > 0 || disk_budget_bytes > 0,
            "minimum_tokens": self.info.prefill_chunk_size,
            "boundaries": ["system_turn_end", "observed_divergence"],
            "checkpoints_per_prefill": 1,
            "eviction_priority": "shared_prefix_before_request_tail",
        });
        self.info.policy["prompt_cache"]["disk_budget_bytes"] = disk_budget_bytes.into();
        Ok(())
    }

    pub fn set_prompt_cache_checkpoints(&mut self, count: usize) -> crate::Result<()> {
        if count > MAX_PROMPT_CACHE_CHECKPOINTS {
            return Err(crate::Error::InvalidArgument(format!(
                "prompt cache checkpoints must be within 0..={MAX_PROMPT_CACHE_CHECKPOINTS}"
            )));
        }
        self.max_prompt_checkpoints = count;
        self.prompt_checkpoints.clear();
        self.cached_tokens.clear();
        self.info.policy["prompt_cache"] = serde_json::json!({
            "checkpoints": count,
            "checkpoint_bytes": PROMPT_CHECKPOINT_BYTES,
            "purgeable": true,
        });
        Ok(())
    }
}
