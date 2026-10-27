use crate::bonsai_dflash::{DflashMode, DflashResolution};

use super::{
    AttentionKernel, BonsaiEngine, BonsaiInfo, BonsaiModel, BonsaiPackage, BonsaiTokenizer,
    DRAFT_CHAIN_MIN_MARGIN, KvOptions, MtpMode, MtpResolution, NgramSettings, Path, SuffixStore,
    VOCAB, prompt_cache,
};
impl BonsaiEngine {
    /// `max_context` of `None` chooses the largest context that fits safely.
    pub fn open(
        path: &Path,
        max_context: Option<usize>,
        prefill_chunk: usize,
    ) -> crate::Result<Self> {
        Self::open_with_options(
            path,
            max_context,
            prefill_chunk,
            None,
            &MtpMode::default(),
            NgramSettings::default(),
            KvOptions::default(),
        )
    }

    /// Speculation follows `mtp`: `Auto` applies the measured default when
    /// the default head is installed, `On` requires one, `Head` names a head
    /// file, and `Off` decodes plainly. The decision and its reason are
    /// reported in the policy JSON.
    pub fn open_with_options(
        path: &Path,
        max_context: Option<usize>,
        prefill_chunk: usize,
        attention_kernel: Option<AttentionKernel>,
        mtp: &MtpMode,
        ngram: NgramSettings,
        kv: KvOptions,
    ) -> crate::Result<Self> {
        Self::open_with_speculation(
            path,
            max_context,
            prefill_chunk,
            attention_kernel,
            mtp,
            &DflashMode::Off("not requested by this constructor".into()),
            ngram,
            kv,
        )
    }

    /// [`Self::open_with_options`] with `KFlash` drafting following
    /// `dflash`: a resolved drafter turns on the residual taps and tree
    /// verify beside the MTP head, which stays loaded for the cycles where
    /// its chain promises more. The decision and its reason are reported in
    /// the policy JSON.
    #[expect(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn open_with_speculation(
        path: &Path,
        max_context: Option<usize>,
        prefill_chunk: usize,
        attention_kernel: Option<AttentionKernel>,
        mtp: &MtpMode,
        dflash: &DflashMode,
        ngram: NgramSettings,
        kv: KvOptions,
    ) -> crate::Result<Self> {
        let package = BonsaiPackage::open(path)?;
        let drafter = match dflash.resolve()? {
            DflashResolution::Drafter(path) => Ok(path),
            DflashResolution::Fallback(reason) => Err(reason),
        };
        let model_name = path.file_stem().map_or_else(
            || "bonsai".into(),
            |stem| stem.to_string_lossy().into_owned(),
        );
        let context = max_context.unwrap_or(0);
        let (settings, disabled) = match mtp.resolve()? {
            MtpResolution::Native(settings) => (Some(settings), None),
            // Whatever the mode resolved with is the record; an `Off` mode carries
            // the reason that turned it off, so nothing has to re-derive it here.
            MtpResolution::Disabled(reason) => (None, Some(reason)),
        };
        let model = BonsaiModel::load_with_drafter(
            package,
            context,
            prefill_chunk,
            attention_kernel,
            settings.as_ref(),
            ngram,
            kv,
            drafter
                .as_ref()
                .ok()
                .map(|path| (path.as_path(), model_name.as_str())),
        )?;
        let tokenizer = BonsaiTokenizer::from_package(model.package())?;
        if tokenizer.vocab_size() != VOCAB {
            return Err(crate::Error::InvalidFormat(
                "Bonsai vocabulary does not match the output matrix".into(),
            ));
        }
        let mtp_policy = match (&settings, model.speculation()) {
            (Some(settings), Some(speculation)) => serde_json::json!({
                "enabled": true,
                "head": settings.path,
                "depth": speculation.depth,
                "max_draft_rows": speculation.max_draft_rows,
                "chain_margin": DRAFT_CHAIN_MIN_MARGIN,
                "head_bytes": speculation.head_bytes,
                "checkpoint_bytes": speculation.checkpoint_bytes,
            }),
            _ => serde_json::json!({"enabled": false, "reason": disabled}),
        };
        let kflash_policy = match (model.kflash_info(), &drafter) {
            (Some(kflash), _) => serde_json::json!({
                "enabled": true,
                "drafter": kflash.path,
                "pinned": kflash.payload_sha256 == crate::bonsai_dflash::PINNED_DFLASH_SHA256,
                "payload_sha256": kflash.payload_sha256,
                "weight_bytes": kflash.weight_bytes,
                "state_bytes": kflash.state_bytes,
                "tap_bytes": crate::bonsai_native::TAP_WINDOW
                    * crate::bonsai_native::TAP_ROW_FLOATS
                    * size_of::<f32>(),
                "max_tree_rows": crate::bonsai_native::MAX_TREE_ROWS,
                "draft_ms": kflash.draft_ms,
                "verify_ms": kflash.verify_ms,
                "calibration": kflash.calibration,
                "fallback": "MTP chain or suffix lookup when its measured tokens per millisecond lead",
            }),
            (None, Err(reason)) => serde_json::json!({"enabled": false, "reason": reason}),
            (None, Ok(_)) => serde_json::json!({"enabled": false}),
        };
        let layout = model.kv_layout();
        let info = BonsaiInfo {
            precision: format!(
                "checkpoint_ptq1_state_{}_kv_{}",
                model.state_format().name(),
                layout.name().replace('/', "_")
            ),
            policy: serde_json::json!({
                "attention_kernel": model.attention_kernel_name(),
                "prefill_kernel": model.prefill_kernel_name(),
                "kv_cache": {
                    "key": layout.key.name(),
                    "value": layout.value.name(),
                    "token_bytes_per_layer": layout.token_bytes(),
                    "initial_tokens": model.kv_allocated(),
                    "initial_bytes": model.kv_allocated_bytes(),
                    "max_bytes": model.kv_max_bytes(),
                },
                "mtp": mtp_policy,
                "kflash": kflash_policy,
                "neural_engine_prefill": model.neural_engine_policy(),
                "ngram": {
                    "enabled": ngram.enabled,
                    "max_drafts": ngram.max_drafts,
                    "min_match": ngram.min_match,
                },
            }),
            ..model.info().clone()
        };
        Ok(Self {
            tokenizer,
            model: Box::new(model),
            info,
            ngram,
            suffix_store: SuffixStore::default(),
            cached_tokens: Vec::new(),
            prompt_checkpoints: Vec::new(),
            max_prompt_checkpoints: 0,
            session_snapshots: Vec::new(),
            disk_snapshots: Vec::new(),
            disk_writer: None,
            disk_closer: prompt_cache::Closer::new(),
            prompt_cache_bytes: 0,
            prompt_cache_disk_bytes: 0,
            prompt_cache_daily_writes: None,
            prompt_cache_dir: None,
            prompt_cache_model_key: String::new(),
            active: Vec::new(),
            resident: None,
            pool: Vec::new(),
            cache_state: 0,
            next_generation: 0,
            kflash_owner: None,
            kflash_history: None,
            #[cfg(test)]
            force_batched_drafts: false,
            #[cfg(test)]
            fail_reservation: None,
        })
    }
}
