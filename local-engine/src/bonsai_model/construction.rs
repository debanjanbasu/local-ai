use super::{
    AttentionKernel, BonsaiEngine, BonsaiInfo, BonsaiModel, BonsaiPackage, BonsaiTokenizer,
    DRAFT_CHAIN_MIN_MARGIN, KvOptions, MtpMode, MtpResolution, NgramSettings, Path, SuffixStore,
    VOCAB,
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
        let package = BonsaiPackage::open(path)?;
        let context = max_context.unwrap_or(0);
        let (settings, disabled) = match mtp.resolve()? {
            MtpResolution::Native(settings) => (Some(settings), None),
            // Whatever the mode resolved with is the record; an `Off` mode carries
            // the reason that turned it off, so nothing has to re-derive it here.
            MtpResolution::Disabled(reason) => (None, Some(reason)),
        };
        let model = BonsaiModel::load(
            package,
            context,
            prefill_chunk,
            attention_kernel,
            settings.as_ref(),
            ngram,
            kv,
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
            prompt_cache_bytes: 0,
            prompt_cache_disk_bytes: 0,
            prompt_cache_dir: None,
            prompt_cache_model_key: String::new(),
        })
    }
}
