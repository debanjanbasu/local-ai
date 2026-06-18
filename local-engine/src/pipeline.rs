use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::UNIX_EPOCH;

use local_core::config::{Gemma4QATConfig, ModelConfig};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::kernels::Kernels;
use local_metal::shaders::ShaderLibrary;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use local_metal::batch::CommandBatch;

use crate::Error;

use crate::gemma4_audio;
use crate::gemma4_vision;
use crate::gguf::GGUFModel;
use crate::kv_cache::{
    QuantizedKvCache, QuantizedKvCacheSnapshot, kv_bits_from_env, swa_ring_capacity,
};
use crate::layer::{
    BatchScratch, LayerParams, LayerWeights, QuantWeight, ScratchBuffers, TransformerLayer,
};
use crate::multimodal::{
    DecodedVideoFrame, MediaInput, MultimodalPrompt, MultimodalSupport, PcmAudio,
    PreparedMultimodalPrompt, SoftTokenOverride, audio_placeholder_tokens,
    audio_soft_token_count_for_samples, image_placeholder_tokens, image_soft_token_count,
    sampled_video_frames, tensor_names_indicate_audio_support, tensor_names_indicate_image_support,
    video_frame_count, video_frame_placeholder_tokens, video_soft_token_count,
};
use crate::pipeline::init_gemma4_qat::{QuantRowTable, load_all_weights};
use crate::sampler::{SamplingParams, SamplingResult, sample};
use crate::tokenizer::Tokenizer;

mod batched;
mod init_gemma;
pub(crate) mod init_gemma4_qat;
mod runner;

pub use batched::BatchedDecodeState;
pub use runner::{
    BatchOutput, BatchRequest, EventSink, ServeEvent, ServeRequest, ServeResponse, StopReason,
};

const F16: usize = std::mem::size_of::<half::f16>();

/// The live KV cache backing. `TurboQuant` is the only backing: its ~4×-smaller
/// resident KV is what lets every device hold the model's full context window.
/// Each non-shared layer owns one [`QuantizedKvCache`] whose K/V is written and
/// read directly as packed codes by the fused `TurboQuant` kernels (no FP16
/// expansion).
type KvCaches = Vec<QuantizedKvCache>;

/// CPU prefix snapshots for the prompt-prefix cache, one per non-shared layer.
type KvSnapshots = Vec<QuantizedKvCacheSnapshot>;

pub struct Pipeline {
    config: Gemma4QATConfig,
    ctx: MetalContext,
    kernels: Kernels,
    layers: Vec<TransformerLayer>,
    tokenizer: Option<Tokenizer>,
    /// Tied embedding/logits table; stays in its quantized GGUF format. The
    /// GPU runs the quantized logits matvec on it while the CPU gathers and
    /// dequantizes single embedding rows from the same shared-memory buffer.
    token_embd: QuantWeight,
    output_norm: MetalBuffer,
    // Per-layer-input (PLE) tensors and scratch — present on Gemma4 E2B.
    // The token-embedding table is streamed row-by-row from CPU rather than held
    // resident in VRAM (exploits the E2B "embed-2B" structure).
    ple_table: Option<QuantRowTable>,
    per_layer_model_proj: Option<QuantWeight>,
    per_layer_proj_norm: Option<MetalBuffer>,
    per_layer_input: Option<MetalBuffer>,
    ple_tok: Option<MetalBuffer>,
    ple_proj: Option<MetalBuffer>,
    // KV caches: one per non-shared layer; `kv_index_map[layer]` gives the slot.
    kv_caches: KvCaches,
    kv_index_map: Vec<usize>,
    hidden_a: MetalBuffer,
    hidden_b: MetalBuffer,
    logits_buf: MetalBuffer,
    scratch: ScratchBuffers,
    position: usize,
    /// Effective context capacity: the device-budgeted KV allocation, never
    /// above the model's positional limit.
    max_context: usize,
    /// Gemma `final_logit_softcapping` value, applied on-device to `logits_buf`
    /// right after the output matvec so the CPU sampler can skip the ~1 ms
    /// vocab-wide tanh pass. Sourced from [`SamplingParams::default`] to keep
    /// identical numerics to the previous CPU-only path.
    logit_softcap: f32,
    /// Multi-row scratch shared by chunked prefill and batched decode.
    batch_scratch: BatchScratch,
    multimodal: MultimodalSupport,
    /// Device-resident Gemma 4 modality tower/projector tensors loaded from
    /// Unsloth's `mmproj-BF16.gguf` companion archive.
    modality_weights: HashMap<String, QuantWeight>,
    /// Small MRU prompt-prefix cache. Entries are CPU snapshots of quantized KV
    /// rows, so keep this bounded independently from the model context window.
    prompt_cache: VecDeque<PromptCacheEntry>,
    /// Small MRU of already-computed multimodal soft embeddings. This avoids
    /// rerunning expensive vision/audio towers when a chat sends follow-up text
    /// against the same decoded media in one engine session.
    media_cache: VecDeque<MediaCacheEntry>,
}

struct MediaCacheEntry {
    key: u64,
    embeddings: Vec<Vec<half::f16>>,
}

struct PromptCacheEntry {
    tokens: Vec<u32>,
    kv_caches: KvSnapshots,
    final_hidden: Vec<u8>,
    final_in_a: bool,
}

/// Cached `LOCAL_AI_TIMING` flag. Read once; safe to call on hot paths.
fn timing_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("LOCAL_AI_TIMING").is_ok())
}

/// Take the first present tensor among `names` (consuming it from the map).
fn take(buffers: &mut HashMap<String, QuantWeight>, names: &[&str]) -> Option<QuantWeight> {
    names.iter().find_map(|n| buffers.remove(*n))
}

/// Like [`take`] but unwraps to the FP16 buffer (norm vectors are never
/// quantized-at-rest).
fn take_f16(
    buffers: &mut HashMap<String, QuantWeight>,
    names: &[&str],
) -> crate::Result<Option<MetalBuffer>> {
    match take(buffers, names) {
        Some(w) => Ok(Some(w.into_f16(names[0])?)),
        None => Ok(None),
    }
}

/// Load one transformer layer's weights (GGUF naming).
fn load_layer_weights(
    buffers: &mut HashMap<String, QuantWeight>,
    layer_idx: usize,
) -> crate::Result<LayerWeights> {
    let prefix = format!("blk.{layer_idx}.");
    let req = |buffers: &mut HashMap<String, QuantWeight>, names: &[&str], label: &str| {
        let owned: Vec<String> = names.iter().map(|n| format!("{prefix}{n}")).collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        take(buffers, &refs).ok_or_else(|| Error::InvalidFormat(format!("missing {prefix}{label}")))
    };
    let req_f16 = |buffers: &mut HashMap<String, QuantWeight>, names: &[&str], label: &str| {
        req(buffers, names, label).and_then(|w| w.into_f16(label))
    };
    let opt = |buffers: &mut HashMap<String, QuantWeight>, names: &[&str]| {
        let owned: Vec<String> = names.iter().map(|n| format!("{prefix}{n}")).collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        take(buffers, &refs)
    };
    let opt_f16 = |buffers: &mut HashMap<String, QuantWeight>,
                   names: &[&str]|
     -> crate::Result<Option<MetalBuffer>> {
        opt(buffers, names)
            .map(|w| w.into_f16(names[0]))
            .transpose()
    };

    Ok(LayerWeights {
        attn_norm: req_f16(
            buffers,
            &["attn_norm.weight", "input_layernorm.weight"],
            "attn_norm",
        )?,
        attn_q: req(buffers, &["attn_q.weight"], "attn_q")?,
        attn_k: opt(buffers, &["attn_k.weight"]),
        attn_v: opt(buffers, &["attn_v.weight"]),
        q_norm: opt_f16(buffers, &["attn_q_norm.weight", "q_norm.weight"])?,
        k_norm: opt_f16(buffers, &["attn_k_norm.weight", "k_norm.weight"])?,
        attn_output: req(buffers, &["attn_output.weight"], "attn_output")?,
        attn_post_norm: req_f16(
            buffers,
            &[
                "post_attention_norm.weight",
                "post_attention_layernorm.weight",
            ],
            "post_attention_norm",
        )?,
        ffn_norm: req_f16(
            buffers,
            &["ffn_norm.weight", "pre_feedforward_layernorm.weight"],
            "ffn_norm",
        )?,
        ffn_gate: req(buffers, &["ffn_gate.weight"], "ffn_gate")?,
        ffn_up: req(buffers, &["ffn_up.weight"], "ffn_up")?,
        ffn_down: req(buffers, &["ffn_down.weight"], "ffn_down")?,
        ffn_post_norm: req_f16(
            buffers,
            &["post_ffw_norm.weight", "post_feedforward_layernorm.weight"],
            "post_ffw_norm",
        )?,
        per_layer_inp_gate: opt(buffers, &["inp_gate.weight"]),
        per_layer_proj: opt(buffers, &["proj.weight"]),
        per_layer_post_norm: opt_f16(buffers, &["post_norm.weight"])?,
        layer_output_scale: opt_f16(buffers, &["layer_output_scale.weight"])?,
    })
}

const QAT_GGUF: &str = "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf";

/// Load config + weight buffers (and any bundled tokenizer) from
/// `model.lma` or the Unsloth QAT GGUF.
fn load_config_and_weights(
    model_dir: &Path,
    device: &ProtocolObject<dyn MTLDevice>,
) -> crate::Result<crate::lma::LoadedLma> {
    let lma_path = model_dir.join("model.lma");
    if lma_path.exists() {
        return crate::lma::load_lma(&lma_path, device);
    }
    let config_path = model_dir.join("config.json");
    let config_json = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::Io(format!("{}: {e}", config_path.display())))?;
    let ModelConfig::Gemma4QAT(config) = ModelConfig::from_json(&config_json).map_err(|e| {
        Error::InvalidFormat(format!("failed to parse {}: {e}", config_path.display()))
    })?;
    let gguf = GGUFModel::open(model_dir.join(QAT_GGUF))?;
    let companion = GGUFModel::open(model_dir.join("mmproj-BF16.gguf")).ok();
    let tensor_names = gguf.tensors.keys().map(String::as_str);
    let companion_names = companion
        .as_ref()
        .into_iter()
        .flat_map(|model| model.tensors.keys().map(String::as_str));
    let has_image_tensors = tensor_names_indicate_image_support(tensor_names.clone())
        || tensor_names_indicate_image_support(companion_names.clone());
    let multimodal = MultimodalSupport {
        config_declares_image: config.multimodal.has_vision_config()
            || config.multimodal.image_token_id.is_some(),
        config_declares_audio: config.multimodal.has_audio_config()
            || config.multimodal.audio_token_id.is_some(),
        config_declares_video: config.multimodal.has_video_config(),
        has_image_tensors,
        has_audio_tensors: tensor_names_indicate_audio_support(tensor_names)
            || tensor_names_indicate_audio_support(companion_names),
        // Gemma 4 video reuses the image/vision companion tensors.
        has_video_tensors: has_image_tensors,
    };
    let mut weights = load_all_weights(&gguf, device)?;
    if let Some(companion) = companion.as_ref() {
        let companion_weights = load_all_weights(companion, device)?;
        for (name, weight) in companion_weights.buffers {
            weights.buffers.entry(name).or_insert(weight);
        }
    }
    Ok(crate::lma::LoadedLma {
        config,
        weights,
        tokenizer_json: std::fs::read(model_dir.join("tokenizer.json")).ok(),
        multimodal,
    })
}

/// Resident KV bytes for one non-shared layer's `(token, head)` row: the packed
/// key+value codes plus their f16 norms.
const fn kv_row_bytes(head_dim: usize, n_kv: usize, kv_bits: u8) -> usize {
    let code_bytes = (head_dim * kv_bits as usize).div_ceil(8);
    n_kv * (2 * code_bytes + 2 * F16 * 2)
}

/// Split the `TurboQuant` KV working set into a context-proportional rate and a
/// fixed floor, so the adaptive sizer can solve for the largest context that
/// fits. Full-attention layers grow with context (`per_position`); sliding-
/// window layers are capped at their ring and the shared dequant scratch is
/// constant, so both fold into `fixed`. With `ring_enabled` false every layer
/// is full attention and `fixed` holds only the scratch.
fn kv_memory_model(
    layers: &[TransformerLayer],
    n_kv: usize,
    kv_bits: u8,
    requested: usize,
    ring_enabled: bool,
) -> (usize, usize) {
    // Fused TurboQuant attention reads the packed codes directly, so there is
    // no longer a context-sized FP16 dequant scratch: only the quantized KV
    // codes of full-attention layers grow per position (added below).
    let mut per_position = 0;
    let mut fixed = 0;
    for l in layers.iter().filter(|l| !l.params.is_kv_shared) {
        let row = kv_row_bytes(l.params.head_dim, n_kv, kv_bits);
        if ring_enabled && l.params.sliding_window > 0 {
            let ring = swa_ring_capacity(l.params.sliding_window, requested);
            fixed += row * ring;
        } else {
            per_position += row;
        }
    }
    (per_position, fixed)
}

/// Clamp the requested context length to what actually fits: the lesser of
/// the device's recommended working set (minus the already-resident weights)
/// and the memory **free right now** (`os_proc_available_memory` on iOS, VM
/// statistics on macOS — queried after load, so weights are already excluded),
/// with fixed headroom for scratch, logits, and transients.
///
/// `LOCAL_AI_MEMORY_BUDGET` (MiB) overrides both signals — deterministic
/// rehearsal of small devices, or capping the engine on shared machines.
fn adaptive_context_length(
    caps: &local_metal::context::DeviceCaps,
    requested: usize,
    weight_bytes: usize,
    per_position: usize,
    fixed_bytes: usize,
) -> usize {
    const MIN_CONTEXT: usize = 512;
    if per_position == 0 {
        // No context-proportional KV (every layer is a capped ring): context is
        // bounded only by the model, not by growing KV.
        return requested;
    }
    let override_mib = std::env::var("LOCAL_AI_MEMORY_BUDGET")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    let budget = override_mib.map_or_else(
        || {
            let device_leg = if caps.recommended_working_set == 0 {
                u64::MAX
            } else {
                caps.recommended_working_set
                    .saturating_sub(weight_bytes as u64)
            };
            // Free-now already excludes the loaded weights.
            let free_leg = local_metal::memory::available_memory_now().unwrap_or(u64::MAX);
            device_leg.min(free_leg)
        },
        |mib| (mib * 1024 * 1024).saturating_sub(weight_bytes as u64),
    );
    if budget == u64::MAX {
        return requested;
    }
    // Reserve headroom and the fixed KV floor (capped sliding-window rings +
    // dequant scratch) before dividing the remainder by the per-position rate.
    let budget = budget
        .saturating_sub(dynamic_headroom(budget))
        .saturating_sub(fixed_bytes as u64);
    let fit = usize::try_from(budget / per_position as u64).unwrap_or(usize::MAX);
    fit.clamp(MIN_CONTEXT.min(requested), requested)
}

/// Memory to hold back from the KV budget for everything that is *not*
/// context-proportional: per-step logits, activation scratch, command-buffer
/// transients, and a safety cushion against the live free-memory signal drifting
/// during a long run. Scaled to the live budget (12.5%) rather than a fixed
/// constant so it tracks the device in real time — a few hundred MiB on a phone,
/// up to a 1 GiB ceiling on a workstation — instead of stranding context on
/// small devices or over-reserving on large ones.
fn dynamic_headroom(budget: u64) -> u64 {
    const MIN_HEADROOM: u64 = 256 * 1024 * 1024;
    const MAX_HEADROOM: u64 = 1024 * 1024 * 1024;
    (budget / 8).clamp(MIN_HEADROOM, MAX_HEADROOM)
}

/// Whether sliding-window layers are stored as a capped ring (the default).
/// `LOCAL_AI_SWA_RING=0` forces the full absolute allocation — an escape hatch
/// for debugging or for exotic attention patterns where the ring is unsafe.
fn swa_ring_enabled() -> bool {
    std::env::var("LOCAL_AI_SWA_RING").ok().as_deref() != Some("0")
}

/// Allocate KV caches: one per non-shared layer; shared layers map to their
/// source layer's cache slot. Returns `(caches, layer → slot map)`.
///
/// Caches are `TurboQuant`-compressed; `LOCAL_AI_KV_QUANT` (`tq2`/`tq3`/`tq4`)
/// selects the bit-width (default 2 — smallest footprint, same fused-path speed).
fn build_kv_caches(
    config: &Gemma4QATConfig,
    layers: &[TransformerLayer],
    device: &ProtocolObject<dyn MTLDevice>,
    max_context_length: usize,
) -> crate::Result<(KvCaches, Vec<usize>)> {
    let n_kv = config.num_key_value_heads;
    let bits = kv_bits_from_env();
    let ring_enabled = swa_ring_enabled();
    let mut kv_index_map = vec![0usize; config.num_hidden_layers];
    let mut slot_of = vec![None; config.num_hidden_layers];
    // One cache slot per non-shared layer; resolve indices once before
    // allocating. Each slot records its head dim and sliding window (`0` for
    // full attention) so sliding-window slots can allocate a small ring.
    let mut slots: Vec<(usize, usize)> = Vec::new();
    for (layer_idx, layer) in layers.iter().enumerate() {
        if config.is_kv_shared(layer_idx) {
            let src = config
                .kv_source_layer(layer_idx)
                .and_then(|s| slot_of[s])
                .ok_or_else(|| {
                    Error::InvalidFormat(format!("no KV source for shared layer {layer_idx}"))
                })?;
            kv_index_map[layer_idx] = src;
        } else {
            let slot = slots.len();
            let window = if config.is_full_attention(layer_idx) {
                0
            } else {
                config.sliding_window
            };
            slots.push((layer.params.head_dim, window));
            slot_of[layer_idx] = Some(slot);
            kv_index_map[layer_idx] = slot;
        }
    }
    let mut caches = Vec::with_capacity(slots.len());
    for (head_dim, window) in slots {
        let cache = if ring_enabled && window > 0 {
            // Sliding-window layer: physically allocate only the ring (window +
            // prefill slack), independent of the context length, so context can
            // grow without these layers' KV growing with it.
            let ring = swa_ring_capacity(window, max_context_length);
            QuantizedKvCache::new_with_bits_and_ring(
                device,
                n_kv,
                head_dim,
                max_context_length,
                ring,
                window,
                bits,
            )?
        } else {
            QuantizedKvCache::new_with_bits(device, n_kv, head_dim, max_context_length, bits)?
        };
        caches.push(cache);
    }
    Ok((caches, kv_index_map))
}

fn format_video_timestamp(seconds: usize) -> String {
    format!("{:02}:{:02} ", seconds / 60, seconds % 60)
}

fn append_media_placeholder(
    tokens: &mut Vec<u32>,
    ple_tokens: &mut Vec<u32>,
    span: &[u32],
    soft_token_id: Option<u32>,
    pad_token: u32,
) {
    let start = tokens.len();
    tokens.extend_from_slice(span);
    ple_tokens.extend(span.iter().map(|&tok| {
        if Some(tok) == soft_token_id {
            pad_token
        } else {
            tok
        }
    }));
    debug_assert_eq!(tokens.len(), ple_tokens.len());
    if let Some(soft_token_id) = soft_token_id {
        debug_assert!(tokens[start..].contains(&soft_token_id));
    }
}

fn append_video_frame_placeholders(
    tokenizer: &Tokenizer,
    tokens: &mut Vec<u32>,
    ple_tokens: &mut Vec<u32>,
    video_span: &[u32],
    timestamps: impl IntoIterator<Item = usize>,
    video_token_id: Option<u32>,
    pad_token: u32,
) -> crate::Result<()> {
    for timestamp_seconds in timestamps {
        let timestamp = format_video_timestamp(timestamp_seconds);
        let timestamp_tokens = tokenizer.encode(&timestamp, false)?;
        tokens.extend_from_slice(&timestamp_tokens);
        ple_tokens.extend_from_slice(&timestamp_tokens);
        append_media_placeholder(tokens, ple_tokens, video_span, video_token_id, pad_token);
    }
    Ok(())
}

impl Pipeline {
    /// Build the Gemma4 QAT runtime pipeline from a model directory.
    ///
    /// # Errors
    ///
    /// Returns an error if config, weights, tokenizer, or Metal resources fail.
    #[allow(clippy::too_many_lines)]
    pub fn new_qat(model_dir: &Path, max_context_length: usize) -> crate::Result<Self> {
        let ctx = MetalContext::new().map_err(|e| Error::InvalidArgument(e.to_string()))?;
        let device = ctx.device();
        let shader_lib =
            ShaderLibrary::new(device).map_err(|e| Error::InvalidArgument(e.to_string()))?;
        let kernels =
            Kernels::new(&ctx, &shader_lib).map_err(|e| Error::InvalidArgument(e.to_string()))?;

        let loaded = load_config_and_weights(model_dir, device)?;
        let (config, mut weight_map) = (loaded.config, loaded.weights);
        let bundled_tokenizer = loaded.tokenizer_json;
        let multimodal = loaded.multimodal;
        let ple_table = weight_map.ple_table.take();

        // Total resident model bytes (device weights + CPU-streamed PLE table)
        // — the fixed cost the context budget is computed against.
        let weight_bytes: usize = weight_map
            .buffers
            .values()
            .map(QuantWeight::device_bytes)
            .sum::<usize>()
            + ple_table.as_ref().map_or(0, QuantRowTable::bytes);

        let buffers = &mut weight_map.buffers;

        let token_embd = buffers
            .remove("token_embd.weight")
            .ok_or_else(|| Error::InvalidFormat("missing token_embd.weight".into()))?;
        let output_norm = buffers
            .remove("output_norm.weight")
            .ok_or_else(|| Error::InvalidFormat("missing output_norm.weight".into()))?
            .into_f16("output_norm.weight")?;
        let per_layer_model_proj = take(buffers, &["per_layer_model_proj.weight"]);
        let per_layer_proj_norm = take_f16(buffers, &["per_layer_proj_norm.weight"])?;

        let h = config.hidden_size;
        let n_head = config.num_attention_heads;
        let n_kv = config.num_key_value_heads;
        let pld = config.hidden_size_per_layer_input;
        let eps = config.rms_norm_eps as f32;

        // Build per-layer weights + params; derive per-layer head_dim / FFN width
        // from the actual tensor shapes (robust to the SWA/full + double-MLP mix).
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut max_head_dim = config.head_dim;
        let mut max_inter = config.intermediate_size;
        for layer_idx in 0..config.num_hidden_layers {
            let w = load_layer_weights(buffers, layer_idx)?;
            let head_dim = w.attn_q.rows() as usize / n_head;
            let intermediate_size = w.ffn_gate.rows() as usize;
            max_head_dim = max_head_dim.max(head_dim);
            max_inter = max_inter.max(intermediate_size);
            let params = LayerParams {
                layer_idx,
                hidden_size: h,
                intermediate_size,
                num_attention_heads: n_head,
                num_key_value_heads: n_kv,
                head_dim,
                per_layer_dim: pld,
                rope_theta: config.rope_theta(layer_idx) as f32,
                partial_rotary_factor: config.partial_rotary_factor(layer_idx) as f32,
                sliding_window: if config.is_full_attention(layer_idx) {
                    0
                } else {
                    config.sliding_window
                },
                rms_norm_eps: eps,
                is_kv_shared: config.is_kv_shared(layer_idx),
            };
            layers.push(TransformerLayer::new(params, w));
        }

        // Size the context window to the device: target the model's full
        // positional range by default (`0` = auto) — never beyond it — and
        // let the memory budget make the final call, since weights are fixed
        // and every remaining byte of the working set can go to KV.
        let requested = if max_context_length == 0 {
            config.max_position_embeddings
        } else {
            max_context_length.min(config.max_position_embeddings)
        };
        let caps = ctx.caps();
        // TurboQuant is the only KV backing: ~4× smaller resident KV is what
        // lets every device hold the model's full context window, the engine's
        // hard requirement for long-running agent sessions.
        let (per_pos, fixed_kv) = kv_memory_model(
            &layers,
            n_kv,
            kv_bits_from_env(),
            requested,
            swa_ring_enabled(),
        );
        let max_context_length =
            adaptive_context_length(&caps, requested, weight_bytes, per_pos, fixed_kv);
        let gib = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
        eprintln!(
            "[device] {} (Apple family {}, {}) working set {:.1} GiB, free now {}",
            caps.name,
            caps.apple_family,
            if caps.has_unified_memory {
                "unified"
            } else {
                "discrete"
            },
            gib(caps.recommended_working_set),
            local_metal::memory::available_memory_now()
                .map_or_else(|| "unknown".into(), |v| format!("{:.1} GiB", gib(v))),
        );
        eprintln!(
            "[memory] model resident {:.2} GiB; context {} positions ({:.2} GiB KV + scratch)",
            weight_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            max_context_length,
            (max_context_length * per_pos) as f64 / 1024.0 / 1024.0 / 1024.0,
        );

        let (kv_caches, kv_index_map) =
            build_kv_caches(&config, &layers, device, max_context_length)?;

        let alloc = |elems: usize| {
            MetalBuffer::empty(device, elems * F16)
                .map_err(|e| Error::InvalidArgument(e.to_string()))
        };
        let hidden_a = alloc(h)?;
        let hidden_b = alloc(h)?;
        let logits_buf = MetalBuffer::empty(device, config.vocab_size * std::mem::size_of::<f32>())
            .map_err(|e| Error::InvalidArgument(e.to_string()))?;
        let scratch = ScratchBuffers::new(device, h, max_inter, n_head, n_kv, max_head_dim, pld)?;
        // Shared multi-row scratch: prefill runs in chunks of up to 64 rows.
        let batch_scratch = BatchScratch::new(
            device,
            256,
            h,
            max_inter,
            n_head,
            n_kv,
            max_head_dim,
            pld,
            config.num_hidden_layers,
        )?;

        // PLE scratch is only allocated when the per-layer tensors are present.
        let ple_total = pld * config.num_hidden_layers;
        let (per_layer_input, ple_tok, ple_proj) = if ple_table.is_some()
            && per_layer_model_proj.is_some()
            && per_layer_proj_norm.is_some()
        {
            (
                Some(alloc(ple_total)?),
                Some(alloc(ple_total)?),
                Some(alloc(ple_total)?),
            )
        } else {
            (None, None, None)
        };

        let tokenizer = bundled_tokenizer
            .as_deref()
            .and_then(|b| Tokenizer::from_bytes(b).ok())
            .or_else(|| Tokenizer::from_model_dir(model_dir).ok());
        let modality_weights = std::mem::take(buffers);

        Ok(Self {
            config,
            ctx,
            kernels,
            layers,
            tokenizer,
            token_embd,
            output_norm,
            ple_table,
            per_layer_model_proj,
            per_layer_proj_norm,
            per_layer_input,
            ple_tok,
            ple_proj,
            kv_caches,
            kv_index_map,
            hidden_a,
            hidden_b,
            logits_buf,
            scratch,
            position: 0,
            max_context: max_context_length,
            logit_softcap: SamplingParams::default().logit_softcap,
            batch_scratch,
            multimodal,
            modality_weights,
            prompt_cache: VecDeque::new(),
            media_cache: VecDeque::new(),
        })
    }

    #[must_use]
    pub const fn config(&self) -> &Gemma4QATConfig {
        &self.config
    }

    /// The effective context window (positions) after the device-memory budget
    /// clamp — the largest per-sequence KV length this pipeline can serve.
    #[must_use]
    pub const fn max_effective_context(&self) -> usize {
        if self.config.max_position_embeddings < self.max_context {
            self.config.max_position_embeddings
        } else {
            self.max_context
        }
    }

    #[must_use]
    pub const fn multimodal_support(&self) -> MultimodalSupport {
        self.multimodal
    }

    fn effective_output_limit(&self, prompt_tokens: usize, requested: usize) -> usize {
        let max_pos = self.config.max_position_embeddings.min(self.max_context);
        // The first sampled token is produced from the prompt's final logits;
        // subsequent generated tokens need context slots when fed back in.
        let context_budget = max_pos.saturating_sub(prompt_tokens).saturating_add(1);
        requested.min(context_budget)
    }

    /// Reset token position and KV caches for a fresh sequence.
    pub fn reset(&mut self) {
        self.position = 0;
        self.kv_caches.iter().for_each(QuantizedKvCache::reset);
    }

    /// Precompute the per-layer input (PLE) vectors for the current token.
    /// Reads the scaled embedding in `hidden_a`; writes `per_layer_input`.
    ///
    /// The token's embedding row is dequantized on demand from the CPU-resident
    /// [`QuantRowTable`] rather than read from a VRAM tensor, so the ~2.35 B
    /// parameter table never occupies device memory.
    fn compute_per_layer_inputs(&self, token: u32) {
        let (Some(table), Some(model_proj), Some(proj_norm)) = (
            self.ple_table.as_ref(),
            self.per_layer_model_proj.as_ref(),
            self.per_layer_proj_norm.as_ref(),
        ) else {
            return;
        };
        let (Some(ple), Some(ple_tok), Some(ple_proj)) = (
            self.per_layer_input.as_ref(),
            self.ple_tok.as_ref(),
            self.ple_proj.as_ref(),
        ) else {
            return;
        };

        let pld = self.config.hidden_size_per_layer_input;
        let n_layer = self.config.num_hidden_layers;
        let total = (pld * n_layer) as u32;
        let eps = self.config.rms_norm_eps as f32;

        // ple_tok = per_layer_token_embd[token] * sqrt(pld): stream one row from
        // the CPU-side quantized table and upload it into the PLE scratch buffer.
        if let Ok(row) = table.dequant_row(token as usize) {
            ple_tok.copy_from_bytes(bytemuck::cast_slice(&row), 0);
        }
        let _ = self
            .kernels
            .scale_in_place_gpu(&self.ctx, ple_tok, (pld as f32).sqrt(), total);
        // ple_proj = rms_norm( (model_proj @ h) / sqrt(h) )
        let _ = model_proj.matvec(&self.ctx, &self.kernels, &self.hidden_a, ple_proj);
        let inv_sqrt_h = 1.0 / (self.config.hidden_size as f32).sqrt();
        let _ = self
            .kernels
            .scale_in_place_gpu(&self.ctx, ple_proj, inv_sqrt_h, total);
        let _ = self.kernels.rms_norm(
            &self.ctx,
            ple_proj,
            proj_norm,
            ple_proj,
            pld as u32,
            n_layer as u32,
            eps,
        );
        // per_layer_input = (ple_proj + ple_tok) * (1/sqrt(2))
        let _ = self
            .kernels
            .residual_add(&self.ctx, ple_proj, ple_tok, ple, total);
        let _ =
            self.kernels
                .scale_in_place_gpu(&self.ctx, ple, std::f32::consts::FRAC_1_SQRT_2, total);
    }

    /// Gather embedding rows for `tokens` from the (possibly quantized) tied
    /// embedding table — dequantized on the CPU straight out of the
    /// shared-memory device buffer — into consecutive FP16 rows of `dst`.
    fn gather_embeddings(&self, tokens: &[u32], dst: &MetalBuffer) -> crate::Result<()> {
        let h = self.config.hidden_size;
        // One scratch row reused across tokens instead of allocating per token.
        let mut row = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            self.token_embd.dequant_row_into(tok as usize, &mut row)?;
            dst.copy_from_bytes(bytemuck::cast_slice(&row[..h]), i * h * F16);
        }
        Ok(())
    }

    fn gather_embeddings_with_overrides(
        &self,
        tokens: &[u32],
        dst: &MetalBuffer,
        overrides: &[SoftTokenOverride],
        base_position: usize,
    ) -> crate::Result<()> {
        self.gather_embeddings(tokens, dst)?;
        let h = self.config.hidden_size;
        // Multimodal soft embeddings (from the vision/audio projectors) already
        // live at the *scaled* text-embedding magnitude — in HF Gemma they
        // replace `embed_tokens(ids) * sqrt(hidden)` directly and are NOT
        // re-scaled by the embedding normalizer. The callers multiply the whole
        // hidden buffer by `embed_scale` after gathering, so pre-divide the
        // overrides here to cancel that multiply.
        let inv_embed_scale = 1.0 / (h as f32).sqrt();
        for soft in overrides {
            let Some(row) = soft.position.checked_sub(base_position) else {
                continue;
            };
            if row >= tokens.len() {
                continue;
            }
            if soft.embedding.len() != h {
                return Err(Error::InvalidFormat(format!(
                    "soft token embedding at position {} has {} values, expected {h}",
                    soft.position,
                    soft.embedding.len()
                )));
            }
            let scaled: Vec<half::f16> = soft
                .embedding
                .iter()
                .map(|x| half::f16::from_f32(x.to_f32() * inv_embed_scale))
                .collect();
            dst.copy_from_bytes(bytemuck::cast_slice(&scaled), row * h * F16);
        }
        Ok(())
    }

    /// Embed `token`, scale by √hidden, precompute PLE, then run all layers.
    /// Returns `true` if the final hidden state is in `hidden_a`.
    fn forward_token(&mut self, token: u32, position: usize) -> crate::Result<bool> {
        self.forward_token_with_embedding(token, token, None, position)
    }

    fn forward_token_with_embedding(
        &mut self,
        token: u32,
        ple_token: u32,
        embedding: Option<&[half::f16]>,
        position: usize,
    ) -> crate::Result<bool> {
        self.gather_embeddings(&[token], &self.hidden_a)?;
        if let Some(embedding) = embedding {
            if embedding.len() != self.config.hidden_size {
                return Err(Error::InvalidFormat(format!(
                    "soft token embedding at position {position} has {} values, expected {}",
                    embedding.len(),
                    self.config.hidden_size
                )));
            }
            // Pre-divide by the embedding normalizer; `forward_token_fp16` /
            // `forward_token_turbo` scale the hidden buffer by `sqrt(hidden)`
            // and multimodal soft embeddings must not be re-scaled.
            let inv_embed_scale = 1.0 / (self.config.hidden_size as f32).sqrt();
            let scaled: Vec<half::f16> = embedding
                .iter()
                .map(|x| half::f16::from_f32(x.to_f32() * inv_embed_scale))
                .collect();
            self.hidden_a
                .copy_from_bytes(bytemuck::cast_slice(&scaled), 0);
        }
        self.forward_token_turbo(ple_token, position)
    }

    /// `TurboQuant` decode path: encode all layers into one command buffer by
    /// default and read/write K/V directly as packed codes via the fused kernels.
    fn forward_token_turbo(&mut self, ple_token: u32, position: usize) -> crate::Result<bool> {
        let h = self.config.hidden_size as u32;
        let scale = (self.config.hidden_size as f32).sqrt();
        let _ = self
            .kernels
            .scale_in_place_gpu(&self.ctx, &self.hidden_a, scale, h);
        self.compute_per_layer_inputs(ple_token);

        let caches = &mut self.kv_caches;
        let mut cur_in_a = true;
        let chunk = std::env::var("LOCAL_AI_DECODE_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(self.layers.len());
        let mut batch = CommandBatch::new(&self.ctx).map_err(Error::Metal)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let (inp, out) = if cur_in_a {
                (&self.hidden_a, &self.hidden_b)
            } else {
                (&self.hidden_b, &self.hidden_a)
            };
            let slot = self.kv_index_map[i];
            layer.forward_decode(
                &self.kernels,
                &mut batch,
                inp,
                out,
                position,
                &mut self.scratch,
                &mut caches[slot],
                self.per_layer_input.as_ref(),
                self.config.num_hidden_layers,
            )?;
            if (i + 1) % chunk == 0 && i + 1 < self.layers.len() {
                batch.commit_and_renew(&self.ctx).map_err(Error::Metal)?;
            }
            cur_in_a = !cur_in_a;
        }
        batch.commit_and_wait().map_err(Error::Metal)?;
        Ok(cur_in_a)
    }

    /// Final norm + tied-embedding logit projection into `logits_buf`.
    fn write_logits(&self, final_in_a: bool) {
        let h = self.config.hidden_size as u32;
        let vocab = self.config.vocab_size as u32;
        let final_hidden = if final_in_a {
            &self.hidden_a
        } else {
            &self.hidden_b
        };
        debug_assert_eq!(self.token_embd.rows(), vocab);
        debug_assert_eq!(self.token_embd.cols(), h);
        // Fuse output-norm + logits matvec + softcap into one command buffer so
        // the decode tail pays a single GPU round-trip instead of three.
        let Ok(mut batch) = CommandBatch::new(&self.ctx).map_err(Error::Metal) else {
            return;
        };
        let _ = self.kernels.rms_norm_into(
            &mut batch,
            final_hidden,
            &self.output_norm,
            &self.hidden_b,
            h,
            1,
            self.config.rms_norm_eps as f32,
        );
        let _ = self.token_embd.matvec_f32out_into(
            &mut batch,
            &self.kernels,
            &self.hidden_b,
            &self.logits_buf,
        );
        if self.logit_softcap > 0.0 {
            let _ = self.kernels.logit_softcap_gpu_into(
                &mut batch,
                &self.logits_buf,
                self.logit_softcap,
                vocab,
            );
        }
        let _ = batch.commit_and_wait().map_err(Error::Metal);
    }

    /// Sample the next token from the logits currently in `logits_buf`.
    fn sample_next(
        &mut self,
        context_tokens: &[u32],
        params: &SamplingParams,
    ) -> crate::Result<SamplingResult> {
        let logits = self.logits_buf.as_mut_slice::<f32>();
        sample(logits, params, context_tokens, &mut fastrand::f32)
            .map_err(|e| Error::Sampling(e.to_string()))
    }

    /// Batched forward over `tokens` at positions `start_pos..start_pos+n`,
    /// writing all KV rows and producing `[n × hidden]` pre-norm hidden rows
    /// in the shared batch scratch. Returns `true` if they ended in `hidden_a`.
    fn forward_tokens_batch(&mut self, tokens: &[u32], start_pos: usize) -> crate::Result<bool> {
        self.forward_tokens_batch_with_embeddings(tokens, tokens, &[], start_pos)
    }

    #[allow(clippy::too_many_lines)]
    fn forward_tokens_batch_with_embeddings(
        &mut self,
        tokens: &[u32],
        ple_tokens: &[u32],
        overrides: &[SoftTokenOverride],
        start_pos: usize,
    ) -> crate::Result<bool> {
        let n = tokens.len();
        if ple_tokens.len() != n {
            return Err(Error::InvalidArgument(format!(
                "PLE token count {} does not match token count {n}",
                ple_tokens.len()
            )));
        }
        let h = self.config.hidden_size as u32;
        let m = n as u32;
        let eps = self.config.rms_norm_eps as f32;
        let bs = &self.batch_scratch;
        debug_assert!(n >= 1 && n <= bs.max_batch);

        // Embedding rows gathered on the CPU (quantized table), then scale +
        // batched PLE in one command buffer.
        self.gather_embeddings_with_overrides(tokens, &bs.hidden_a, overrides, start_pos)?;
        let mut batch = CommandBatch::new(&self.ctx).map_err(Error::Metal)?;
        let embed_scale = (self.config.hidden_size as f32).sqrt();
        self.kernels
            .scale_in_place_gpu_into(&mut batch, &bs.hidden_a, embed_scale, m * h)?;

        let ple_ready = if let (Some(table), Some(model_proj), Some(proj_norm)) = (
            self.ple_table.as_ref(),
            self.per_layer_model_proj.as_ref(),
            self.per_layer_proj_norm.as_ref(),
        ) {
            let pld = self.config.hidden_size_per_layer_input;
            let nl = self.config.num_hidden_layers;
            let row = pld * nl;
            // One scratch row reused across tokens instead of allocating per token.
            let mut r = Vec::new();
            for (i, &tok) in ple_tokens.iter().enumerate() {
                if table.dequant_row_into(tok as usize, &mut r).is_ok() {
                    bs.ple_tok
                        .copy_from_bytes(bytemuck::cast_slice(&r), i * row * F16);
                }
            }
            let total = (n * row) as u32;
            self.kernels.scale_in_place_gpu_into(
                &mut batch,
                &bs.ple_tok,
                (pld as f32).sqrt(),
                total,
            )?;
            model_proj.matmul_nt_into(&mut batch, &self.kernels, &bs.hidden_a, &bs.ple_proj, m)?;
            self.kernels.scale_in_place_gpu_into(
                &mut batch,
                &bs.ple_proj,
                1.0 / embed_scale,
                total,
            )?;
            self.kernels.rms_norm_into(
                &mut batch,
                &bs.ple_proj,
                proj_norm,
                &bs.ple_proj,
                pld as u32,
                (n * nl) as u32,
                eps,
            )?;
            self.kernels.residual_add_into(
                &mut batch,
                &bs.ple_proj,
                &bs.ple_tok,
                &bs.ple_all,
                total,
            )?;
            self.kernels.scale_in_place_gpu_into(
                &mut batch,
                &bs.ple_all,
                std::f32::consts::FRAC_1_SQRT_2,
                total,
            )?;
            true
        } else {
            false
        };
        batch.commit_and_wait().map_err(Error::Metal)?;

        let mut cur_in_a = true;
        let ple_arg = ple_ready.then_some((&bs.ple_all, self.config.num_hidden_layers));
        let caches = &mut self.kv_caches;
        // One command buffer across layers, chunk-committed. `forward_batch`
        // is now fully in-stream (the per-layer PLE slice is gathered on the
        // GPU, not memcpy'd from the CPU into shared scratch), so any chunk
        // size is correct. Prefill is compute-bound on the per-chunk KV
        // dequant, so larger chunks only trim ~2% of per-layer commit overhead;
        // the default stays 1. Override with `LOCAL_AI_BATCH_CHUNK`.
        let chunk = std::env::var("LOCAL_AI_BATCH_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(1);
        let mut batch = CommandBatch::new(&self.ctx).map_err(Error::Metal)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let (inp, out) = if cur_in_a {
                (&bs.hidden_a, &bs.hidden_b)
            } else {
                (&bs.hidden_b, &bs.hidden_a)
            };
            let slot = self.kv_index_map[i];
            layer.forward_batch(
                &self.kernels,
                &mut batch,
                n,
                start_pos,
                inp,
                out,
                bs,
                &self.scratch.ones,
                &mut caches[slot],
                ple_arg,
            )?;
            if (i + 1) % chunk == 0 && i + 1 < self.layers.len() {
                batch.commit_and_renew(&self.ctx).map_err(Error::Metal)?;
            }
            cur_in_a = !cur_in_a;
        }
        batch.commit_and_wait().map_err(Error::Metal)?;
        Ok(cur_in_a)
    }

    fn prompt_cache_enabled() -> bool {
        std::env::var("LOCAL_AI_PROMPT_CACHE").ok().as_deref() != Some("0")
    }

    fn max_prompt_cache_tokens(&self) -> usize {
        std::env::var("LOCAL_AI_PROMPT_CACHE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(8192)
            .min(self.max_context)
    }

    fn max_prompt_cache_entries() -> usize {
        std::env::var("LOCAL_AI_PROMPT_CACHE_ENTRIES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(4)
    }

    fn media_cache_enabled() -> bool {
        std::env::var("LOCAL_AI_MEDIA_CACHE").ok().as_deref() != Some("0")
    }

    fn max_media_cache_entries() -> usize {
        std::env::var("LOCAL_AI_MEDIA_CACHE_ENTRIES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(8)
    }

    fn get_cached_media_embeddings(&mut self, key: u64) -> Option<Vec<Vec<half::f16>>> {
        if !Self::media_cache_enabled() {
            return None;
        }
        let idx = self.media_cache.iter().position(|entry| entry.key == key)?;
        let entry = self.media_cache.remove(idx)?;
        let embeddings = entry.embeddings.clone();
        self.media_cache.push_back(entry);
        Some(embeddings)
    }

    fn remember_media_embeddings(&mut self, key: u64, embeddings: &[Vec<half::f16>]) {
        if !Self::media_cache_enabled() {
            return;
        }
        if let Some(existing) = self.media_cache.iter().position(|entry| entry.key == key) {
            self.media_cache.remove(existing);
        }
        let max_entries = Self::max_media_cache_entries();
        while self.media_cache.len() >= max_entries {
            self.media_cache.pop_front();
        }
        self.media_cache.push_back(MediaCacheEntry {
            key,
            embeddings: embeddings.to_vec(),
        });
    }

    fn media_hash(
        tag: &str,
        soft_token_budget: usize,
    ) -> std::collections::hash_map::DefaultHasher {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tag.hash(&mut hasher);
        soft_token_budget.hash(&mut hasher);
        hasher
    }

    fn hash_path_fingerprint(hasher: &mut impl Hasher, path: &Path) {
        path.to_string_lossy().hash(hasher);
        let Ok(metadata) = std::fs::metadata(path) else {
            return;
        };
        metadata.len().hash(hasher);
        metadata.is_dir().hash(hasher);
        if let Ok(modified) = metadata.modified()
            && let Ok(duration) = modified.duration_since(UNIX_EPOCH)
        {
            duration.as_nanos().hash(hasher);
        }
        if metadata.is_dir() {
            let Ok(entries) = std::fs::read_dir(path) else {
                return;
            };
            let mut entries: Vec<_> = entries.filter_map(Result::ok).collect();
            entries.sort_by_key(std::fs::DirEntry::path);
            for entry in entries {
                entry.file_name().to_string_lossy().hash(hasher);
                if let Ok(entry_metadata) = entry.metadata() {
                    entry_metadata.len().hash(hasher);
                    if let Ok(modified) = entry_metadata.modified()
                        && let Ok(duration) = modified.duration_since(UNIX_EPOCH)
                    {
                        duration.as_nanos().hash(hasher);
                    }
                }
            }
        }
    }

    fn path_media_key(tag: &str, path: &Path, soft_token_budget: usize) -> u64 {
        let mut hasher = Self::media_hash(tag, soft_token_budget);
        Self::hash_path_fingerprint(&mut hasher, path);
        hasher.finish()
    }

    fn decoded_image_media_key(
        tag: &str,
        image: &crate::DecodedRgbImage,
        soft_token_budget: usize,
    ) -> u64 {
        let mut hasher = Self::media_hash(tag, soft_token_budget);
        image.width.hash(&mut hasher);
        image.height.hash(&mut hasher);
        image.rgb.hash(&mut hasher);
        hasher.finish()
    }

    fn pcm_audio_media_key(audio: &PcmAudio) -> u64 {
        let mut hasher = Self::media_hash("pcm_audio", 0);
        audio.sample_rate.hash(&mut hasher);
        audio.channels.hash(&mut hasher);
        audio.samples.len().hash(&mut hasher);
        for sample in &audio.samples {
            sample.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn common_cached_prompt_prefix(&self, tokens: &[u32]) -> Option<(usize, usize, bool)> {
        self.prompt_cache
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.kv_caches.len() == self.kv_caches.len())
            .filter_map(|(idx, entry)| {
                let common = tokens
                    .iter()
                    .zip(&entry.tokens)
                    .take_while(|(a, b)| a == b)
                    .count();
                if common == 0 {
                    return None;
                }

                // We only have a final hidden snapshot for exact cached prompts.
                // If the new prompt is a shorter prefix of a cached prompt,
                // reuse KV up to the previous token and recompute the last
                // prompt token's hidden.
                let exact = common == tokens.len() && common == entry.tokens.len();
                let usable = if exact || common < tokens.len() {
                    common
                } else {
                    common.saturating_sub(1)
                };
                (usable > 0).then_some((idx, usable, exact))
            })
            // Prefer the longest prefix; for ties, prefer the most recently
            // used entry (the back of the VecDeque).
            .max_by_key(|&(idx, usable, _)| (usable, idx))
    }

    fn restore_cached_prompt_prefix(
        &mut self,
        entry_idx: usize,
        positions: usize,
        exact: bool,
    ) -> Option<bool> {
        let entry = self.prompt_cache.remove(entry_idx)?;
        for (cache, snapshot) in self.kv_caches.iter().zip(&entry.kv_caches) {
            cache.restore_prefix(snapshot, positions);
        }
        self.position = positions - 1;
        let final_in_a = if exact {
            let dst = if entry.final_in_a {
                &self.hidden_a
            } else {
                &self.hidden_b
            };
            dst.copy_from_bytes(&entry.final_hidden, 0);
            Some(entry.final_in_a)
        } else {
            None
        };
        self.prompt_cache.push_back(entry);
        final_in_a
    }

    fn remember_prompt(&mut self, tokens: &[u32], final_in_a: bool) {
        if !Self::prompt_cache_enabled() || tokens.len() > self.max_prompt_cache_tokens() {
            return;
        }
        let src = if final_in_a {
            &self.hidden_a
        } else {
            &self.hidden_b
        };
        let hidden_bytes = self.config.hidden_size * F16;
        if let Some(existing) = self
            .prompt_cache
            .iter()
            .position(|entry| entry.tokens == tokens)
        {
            self.prompt_cache.remove(existing);
        }
        let max_entries = Self::max_prompt_cache_entries();
        while self.prompt_cache.len() >= max_entries {
            self.prompt_cache.pop_front();
        }
        self.prompt_cache.push_back(PromptCacheEntry {
            tokens: tokens.to_vec(),
            kv_caches: self
                .kv_caches
                .iter()
                .map(|c| c.snapshot_prefix(tokens.len()))
                .collect(),
            final_hidden: src.as_slice::<u8>()[..hidden_bytes].to_vec(),
            final_in_a,
        });
    }

    /// Prefill the prompt: every token but the last runs through the batched
    /// multi-row path in chunks (the weights stream once per chunk instead of
    /// once per token), then the last token takes the single-token path so
    /// the logits flow applies unchanged. Returns `true` if the
    /// final hidden landed in `hidden_a`. `LOCAL_AI_PREFILL_BATCH=0` forces
    /// token-by-token prefill; `LOCAL_AI_PREFILL_CHUNK` tunes the chunk size.
    fn prefill_prompt(&mut self, tokens: &[u32]) -> crate::Result<bool> {
        let last_idx = tokens.len() - 1;
        let start_idx = if Self::prompt_cache_enabled()
            && let Some((entry_idx, positions, exact)) = self.common_cached_prompt_prefix(tokens)
        {
            if let Some(final_in_a) = self.restore_cached_prompt_prefix(entry_idx, positions, exact)
            {
                return Ok(final_in_a);
            }
            positions
        } else {
            0
        };
        let batch_prefill = std::env::var("LOCAL_AI_PREFILL_BATCH").ok().as_deref() != Some("0");
        let chunk_len = std::env::var("LOCAL_AI_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(self.batch_scratch.max_batch)
            .min(self.batch_scratch.max_batch);
        if batch_prefill && start_idx < last_idx {
            let mut pos = start_idx;
            for chunk in tokens[start_idx..last_idx].chunks(chunk_len) {
                self.forward_tokens_batch(chunk, pos)?;
                pos += chunk.len();
            }
        } else {
            for (off, &token) in tokens[start_idx..last_idx].iter().enumerate() {
                let pi = start_idx + off;
                self.position = pi;
                self.forward_token(token, pi)?;
            }
        }
        self.position = last_idx;
        self.forward_token(tokens[last_idx], last_idx)
    }

    fn prefill_prepared_multimodal(
        &mut self,
        prepared: &PreparedMultimodalPrompt,
    ) -> crate::Result<bool> {
        if prepared.tokens.is_empty() {
            return Err(Error::Generation("empty multimodal prompt".into()));
        }
        if prepared.ple_tokens.len() != prepared.tokens.len() {
            return Err(Error::InvalidArgument(format!(
                "PLE token count {} does not match token count {}",
                prepared.ple_tokens.len(),
                prepared.tokens.len()
            )));
        }
        let last_idx = prepared.tokens.len() - 1;
        let batch_prefill = std::env::var("LOCAL_AI_PREFILL_BATCH").ok().as_deref() != Some("0");
        let chunk_len = std::env::var("LOCAL_AI_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(self.batch_scratch.max_batch)
            .min(self.batch_scratch.max_batch);
        if batch_prefill && last_idx > 0 {
            let mut pos = 0usize;
            while pos < last_idx {
                let end = (pos + chunk_len).min(last_idx);
                let overrides: Vec<SoftTokenOverride> = prepared
                    .soft_tokens
                    .iter()
                    .filter(|soft| (pos..end).contains(&soft.position))
                    .cloned()
                    .collect();
                self.forward_tokens_batch_with_embeddings(
                    &prepared.tokens[pos..end],
                    &prepared.ple_tokens[pos..end],
                    &overrides,
                    pos,
                )?;
                pos = end;
            }
        } else {
            for pos in 0..last_idx {
                let embedding = prepared
                    .soft_tokens
                    .iter()
                    .find(|soft| soft.position == pos)
                    .map(|soft| soft.embedding.as_slice());
                self.position = pos;
                self.forward_token_with_embedding(
                    prepared.tokens[pos],
                    prepared.ple_tokens[pos],
                    embedding,
                    pos,
                )?;
            }
        }
        self.position = last_idx;
        let embedding = prepared
            .soft_tokens
            .iter()
            .find(|soft| soft.position == last_idx)
            .map(|soft| soft.embedding.as_slice());
        self.forward_token_with_embedding(
            prepared.tokens[last_idx],
            prepared.ple_tokens[last_idx],
            embedding,
            last_idx,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn prepare_multimodal_prompt_tokens(
        &self,
        prompt: &MultimodalPrompt,
    ) -> crate::Result<PreparedMultimodalPrompt> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Tokenizer("no tokenizer.json found in model directory".into()))?;
        let wrapped = tokenizer
            .chat_prompt(&prompt.text)
            .unwrap_or_else(|| prompt.text.clone());
        let (prefix, suffix) = wrapped
            .find(&prompt.text)
            .map_or(("", wrapped.as_str()), |idx| wrapped.split_at(idx));
        let mut tokens = tokenizer.encode(prefix, true)?;
        let mut ple_tokens = tokens.clone();
        // Gemma 4 HF uses the text PAD token identity for media placeholder
        // slots when computing PLE. The released configs use pad id 0.
        let pad_token = 0;
        let image_span = image_placeholder_tokens(&self.config.multimodal).ok_or_else(|| {
            Error::InvalidFormat("Gemma 4 image token IDs are missing from config".into())
        })?;
        for media in &prompt.media {
            match media {
                MediaInput::Image { .. } | MediaInput::DecodedImage { .. } => {
                    append_media_placeholder(
                        &mut tokens,
                        &mut ple_tokens,
                        &image_span,
                        self.config.multimodal.image_token_id,
                        pad_token,
                    );
                }
                MediaInput::Audio { path } => {
                    let samples = gemma4_audio::wav_sample_count_16khz(path)?;
                    let soft_count = audio_soft_token_count_for_samples(samples);
                    let audio_span = audio_placeholder_tokens(&self.config.multimodal, soft_count)
                        .ok_or_else(|| {
                            Error::InvalidFormat(
                                "Gemma 4 audio token IDs are missing from config".into(),
                            )
                        })?;
                    append_media_placeholder(
                        &mut tokens,
                        &mut ple_tokens,
                        &audio_span,
                        self.config.multimodal.audio_token_id,
                        pad_token,
                    );
                }
                MediaInput::PcmAudio { audio } => {
                    let samples = gemma4_audio::pcm_sample_count_16khz(audio)?;
                    let soft_count = audio_soft_token_count_for_samples(samples);
                    let audio_span = audio_placeholder_tokens(&self.config.multimodal, soft_count)
                        .ok_or_else(|| {
                            Error::InvalidFormat(
                                "Gemma 4 audio token IDs are missing from config".into(),
                            )
                        })?;
                    append_media_placeholder(
                        &mut tokens,
                        &mut ple_tokens,
                        &audio_span,
                        self.config.multimodal.audio_token_id,
                        pad_token,
                    );
                }
                MediaInput::Video { path } => {
                    let video_span = video_frame_placeholder_tokens(&self.config.multimodal)
                        .ok_or_else(|| {
                            Error::InvalidFormat(
                                "Gemma 4 video token IDs are missing from config".into(),
                            )
                        })?;
                    let frames =
                        sampled_video_frames(path, video_frame_count(&self.config.multimodal))?;
                    append_video_frame_placeholders(
                        tokenizer,
                        &mut tokens,
                        &mut ple_tokens,
                        &video_span,
                        frames.iter().map(|frame| frame.timestamp_seconds),
                        self.config.multimodal.video_token_id,
                        pad_token,
                    )?;
                }
                MediaInput::DecodedVideo { frames } => {
                    let video_span = video_frame_placeholder_tokens(&self.config.multimodal)
                        .ok_or_else(|| {
                            Error::InvalidFormat(
                                "Gemma 4 video token IDs are missing from config".into(),
                            )
                        })?;
                    if frames.is_empty() {
                        return Err(Error::InvalidArgument(
                            "decoded video contains no frames".into(),
                        ));
                    }
                    append_video_frame_placeholders(
                        tokenizer,
                        &mut tokens,
                        &mut ple_tokens,
                        &video_span,
                        frames
                            .iter()
                            .take(video_frame_count(&self.config.multimodal))
                            .map(|frame| frame.timestamp_seconds),
                        self.config.multimodal.video_token_id,
                        pad_token,
                    )?;
                }
            }
        }
        let rest = tokenizer.encode(suffix, false)?;
        tokens.extend_from_slice(&rest);
        ple_tokens.extend_from_slice(&rest);
        Ok(PreparedMultimodalPrompt {
            tokens,
            ple_tokens,
            soft_tokens: Vec::new(),
        })
    }

    /// Generate a chat reply: wraps `prompt` in the model's turn template
    /// (detected from the vocabulary) so the instruction-tuned model sees a
    /// proper `user` turn and answers as `model`. Falls back to raw
    /// continuation when the vocabulary has no turn tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if no tokenizer is available, or Metal/sampling fails.
    pub fn generate_chat(
        &mut self,
        prompt: &str,
        params: &crate::GenerateParams,
    ) -> crate::Result<String> {
        let wrapped = self.tokenizer.as_ref().and_then(|t| t.chat_prompt(prompt));
        match wrapped {
            Some(w) => self.generate(&w, params),
            None => self.generate(prompt, params),
        }
    }

    fn project_audio_frontend_embeddings(
        &self,
        mut frontend: gemma4_audio::AudioFrontendOutput,
    ) -> crate::Result<Vec<Vec<half::f16>>> {
        gemma4_audio::run_audio_first_ffn_stage(
            &self.ctx,
            &self.kernels,
            &self.modality_weights,
            &mut frontend,
        )?;
        gemma4_audio::project_audio_soft_tokens(
            &self.ctx,
            &self.kernels,
            &self.modality_weights,
            &frontend,
        )
    }

    fn encode_image_path_embeddings(
        &mut self,
        path: &Path,
        soft_token_budget: usize,
    ) -> crate::Result<Vec<Vec<half::f16>>> {
        let key = Self::path_media_key("image_path", path, soft_token_budget);
        if let Some(cached) = self.get_cached_media_embeddings(key) {
            return Ok(cached);
        }
        let embeddings = gemma4_vision::encode_image_soft_tokens(
            &self.ctx,
            &self.kernels,
            &self.modality_weights,
            path,
            soft_token_budget,
        )?;
        self.remember_media_embeddings(key, &embeddings);
        Ok(embeddings)
    }

    fn encode_decoded_image_embeddings(
        &mut self,
        image: &crate::DecodedRgbImage,
        soft_token_budget: usize,
    ) -> crate::Result<Vec<Vec<half::f16>>> {
        let key = Self::decoded_image_media_key("decoded_image", image, soft_token_budget);
        if let Some(cached) = self.get_cached_media_embeddings(key) {
            return Ok(cached);
        }
        let embeddings = gemma4_vision::encode_decoded_rgb_soft_tokens(
            &self.ctx,
            &self.kernels,
            &self.modality_weights,
            image,
            soft_token_budget,
        )?;
        self.remember_media_embeddings(key, &embeddings);
        Ok(embeddings)
    }

    fn encode_audio_path_embeddings(&mut self, path: &Path) -> crate::Result<Vec<Vec<half::f16>>> {
        let key = Self::path_media_key("audio_path", path, 0);
        if let Some(cached) = self.get_cached_media_embeddings(key) {
            return Ok(cached);
        }
        let frontend = gemma4_audio::run_audio_frontend(
            &self.ctx,
            &self.kernels,
            &self.modality_weights,
            path,
        )?;
        let embeddings = self.project_audio_frontend_embeddings(frontend)?;
        self.remember_media_embeddings(key, &embeddings);
        Ok(embeddings)
    }

    fn encode_pcm_audio_embeddings(
        &mut self,
        audio: &PcmAudio,
    ) -> crate::Result<Vec<Vec<half::f16>>> {
        let key = Self::pcm_audio_media_key(audio);
        if let Some(cached) = self.get_cached_media_embeddings(key) {
            return Ok(cached);
        }
        let frontend = gemma4_audio::run_pcm_audio_frontend(
            &self.ctx,
            &self.kernels,
            &self.modality_weights,
            audio,
        )?;
        let embeddings = self.project_audio_frontend_embeddings(frontend)?;
        self.remember_media_embeddings(key, &embeddings);
        Ok(embeddings)
    }

    fn encode_video_path_embeddings(&mut self, path: &Path) -> crate::Result<Vec<Vec<half::f16>>> {
        let key = Self::path_media_key(
            "video_path",
            path,
            video_soft_token_count(&self.config.multimodal),
        );
        if let Some(cached) = self.get_cached_media_embeddings(key) {
            return Ok(cached);
        }
        let frames = sampled_video_frames(path, video_frame_count(&self.config.multimodal))?;
        let soft_tokens_per_frame = video_soft_token_count(&self.config.multimodal);
        let mut embeddings = Vec::with_capacity(frames.len() * soft_tokens_per_frame);
        for frame in frames {
            embeddings
                .extend(self.encode_image_path_embeddings(&frame.path, soft_tokens_per_frame)?);
        }
        self.remember_media_embeddings(key, &embeddings);
        Ok(embeddings)
    }

    fn encode_decoded_video_embeddings(
        &mut self,
        frames: &[DecodedVideoFrame],
    ) -> crate::Result<Vec<Vec<half::f16>>> {
        if frames.is_empty() {
            return Err(Error::InvalidArgument(
                "decoded video contains no frames".into(),
            ));
        }
        let frame_count = frames.len().min(video_frame_count(&self.config.multimodal));
        let soft_tokens_per_frame = video_soft_token_count(&self.config.multimodal);
        let mut embeddings = Vec::with_capacity(frame_count * soft_tokens_per_frame);
        for frame in frames.iter().take(frame_count) {
            embeddings
                .extend(self.encode_decoded_image_embeddings(&frame.image, soft_tokens_per_frame)?);
        }
        Ok(embeddings)
    }

    /// Build a fully-prepared multimodal prompt: validate modality support,
    /// expand placeholder tokens, GPU-encode each media item into soft
    /// embeddings, validate the slot/embedding counts, and attach the sorted
    /// [`SoftTokenOverride`]s. Shared by the single-stream
    /// [`Self::generate_multimodal_chat`] and the batched server admission path
    /// (`Pipeline::prefill_lane_prepared`).
    ///
    /// Callers are responsible for context-window enforcement, which differs by
    /// path (single-stream uses `max_position_embeddings`; the batched path uses
    /// the lane capacity).
    ///
    /// # Errors
    ///
    /// Returns an error when a requested modality's tensors are absent from the
    /// bundle, when media decoding/encoding fails, or when the produced soft
    /// embeddings do not match the reserved placeholder slots.
    #[allow(clippy::too_many_lines)]
    pub fn prepare_multimodal_prompt(
        &mut self,
        prompt: &MultimodalPrompt,
    ) -> crate::Result<PreparedMultimodalPrompt> {
        let wants_images = prompt.has_images();
        let wants_audio = prompt.has_audio();
        let wants_video = prompt.has_video();
        let missing_images = wants_images && !self.multimodal.supports_images();
        let missing_audio = wants_audio && !self.multimodal.supports_audio();
        let missing_video = wants_video && !self.multimodal.supports_video();
        if missing_images || missing_audio || missing_video {
            let mut missing = Vec::new();
            if missing_images {
                missing.push("image");
            }
            if missing_audio {
                missing.push("audio");
            }
            if missing_video {
                missing.push("video");
            }
            return Err(Error::InvalidFormat(format!(
                "Gemma 4 E2B multimodal {} input requested, but this installed model bundle does not contain the required modality tensors. Install the full optimized Unsloth Gemma 4 E2B multimodal assets (downloaded via the Kaggle workaround when Hugging Face is blocked) and rebuild the .lma; the current bundle can still run text-only generation.",
                missing.join("+")
            )));
        }

        let mut prepared = self.prepare_multimodal_prompt_tokens(prompt)?;
        let image_slots = prepared
            .tokens
            .iter()
            .filter(|&&tok| Some(tok) == self.config.multimodal.image_token_id)
            .count();
        let audio_slots = prepared
            .tokens
            .iter()
            .filter(|&&tok| Some(tok) == self.config.multimodal.audio_token_id)
            .count();
        let video_slots = prepared
            .tokens
            .iter()
            .filter(|&&tok| Some(tok) == self.config.multimodal.video_token_id)
            .count();

        let mut image_embeddings = Vec::with_capacity(image_slots);
        let mut audio_embeddings = Vec::with_capacity(audio_slots);
        let mut video_embeddings = Vec::with_capacity(video_slots);
        for media in &prompt.media {
            match media {
                MediaInput::Image { path } => {
                    image_embeddings.extend(self.encode_image_path_embeddings(
                        path,
                        image_soft_token_count(&self.config.multimodal),
                    )?);
                }
                MediaInput::DecodedImage { image } => {
                    image_embeddings.extend(self.encode_decoded_image_embeddings(
                        image,
                        image_soft_token_count(&self.config.multimodal),
                    )?);
                }
                MediaInput::Audio { path } => {
                    audio_embeddings.extend(self.encode_audio_path_embeddings(path)?);
                }
                MediaInput::PcmAudio { audio } => {
                    audio_embeddings.extend(self.encode_pcm_audio_embeddings(audio)?);
                }
                MediaInput::Video { path } => {
                    video_embeddings.extend(self.encode_video_path_embeddings(path)?);
                }
                MediaInput::DecodedVideo { frames } => {
                    video_embeddings.extend(self.encode_decoded_video_embeddings(frames)?);
                }
            }
        }
        if image_embeddings.len() != image_slots {
            return Err(Error::InvalidState(format!(
                "image encoder produced {} soft embeddings, but the prompt reserved {image_slots} image slots",
                image_embeddings.len()
            )));
        }
        if audio_embeddings.len() != audio_slots {
            return Err(Error::InvalidState(format!(
                "audio encoder produced {} soft embeddings, but the prompt reserved {audio_slots} audio slots",
                audio_embeddings.len()
            )));
        }
        if video_embeddings.len() != video_slots {
            return Err(Error::InvalidState(format!(
                "video encoder produced {} soft embeddings, but the prompt reserved {video_slots} video slots",
                video_embeddings.len()
            )));
        }
        let mut soft_tokens: Vec<SoftTokenOverride> = prepared
            .tokens
            .iter()
            .enumerate()
            .filter_map(|(position, &tok)| {
                (Some(tok) == self.config.multimodal.image_token_id).then_some(position)
            })
            .zip(image_embeddings)
            .map(|(position, embedding)| SoftTokenOverride {
                position,
                embedding,
            })
            .collect();
        soft_tokens.extend(
            prepared
                .tokens
                .iter()
                .enumerate()
                .filter_map(|(position, &tok)| {
                    (Some(tok) == self.config.multimodal.audio_token_id).then_some(position)
                })
                .zip(audio_embeddings)
                .map(|(position, embedding)| SoftTokenOverride {
                    position,
                    embedding,
                }),
        );
        soft_tokens.extend(
            prepared
                .tokens
                .iter()
                .enumerate()
                .filter_map(|(position, &tok)| {
                    (Some(tok) == self.config.multimodal.video_token_id).then_some(position)
                })
                .zip(video_embeddings)
                .map(|(position, embedding)| SoftTokenOverride {
                    position,
                    embedding,
                }),
        );
        soft_tokens.sort_by_key(|override_| override_.position);
        prepared.soft_tokens = soft_tokens;
        Ok(prepared)
    }

    /// Generate from a structured Gemma 4 prompt that may include image/audio
    /// inputs.
    ///
    /// # Errors
    ///
    /// Returns an actionable error when the installed bundle declares Gemma 4
    /// multimodal tokens/config but does not include the corresponding
    /// optimized modality tensors. Text-only prompts reuse [`Self::generate_chat`].
    #[allow(clippy::too_many_lines)]
    pub fn generate_multimodal_chat(
        &mut self,
        prompt: &MultimodalPrompt,
        params: &crate::GenerateParams,
    ) -> crate::Result<String> {
        if prompt.media.is_empty() {
            return self.generate_chat(&prompt.text, params);
        }

        let prepared = self.prepare_multimodal_prompt(prompt)?;

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Tokenizer("no tokenizer.json found in model directory".into()))?;
        let sample_params = SamplingParams {
            temperature: params.temperature,
            top_p: params.top_p,
            eos_tokens: tokenizer.eos_ids().to_vec(),
            // Softcapping is applied on-device after the logits matvec
            // (see `write_logits`), so skip the CPU pass.
            logit_softcap: 0.0,
            ..SamplingParams::default()
        };
        let max_pos = self.config.max_position_embeddings.min(self.max_context);
        if prepared.tokens.len() > max_pos {
            return Err(Error::Generation(format!(
                "multimodal prompt is {} tokens but the effective context window is {max_pos}",
                prepared.tokens.len()
            )));
        }

        let output_limit = self.effective_output_limit(prepared.tokens.len(), params.max_tokens);
        if output_limit == 0 {
            return Ok(String::new());
        }

        let mut output_tokens = Vec::with_capacity(output_limit.min(4096));
        let mut context_tokens = prepared.tokens.clone();
        let final_in_a = self.prefill_prepared_multimodal(&prepared)?;
        self.write_logits(final_in_a);
        let next = self.sample_next(&context_tokens, &sample_params)?;
        if next.is_eos {
            return self.detokenize(&output_tokens);
        }
        context_tokens.push(next.token_id);
        output_tokens.push(next.token_id);

        while output_tokens.len() < output_limit {
            let last = *output_tokens
                .last()
                .ok_or_else(|| Error::Generation("no token to continue from".into()))?;
            if self.position + 1 >= max_pos {
                break;
            }
            self.position += 1;
            let timing = timing_enabled();
            let t0 = std::time::Instant::now();
            let final_in_a = self.forward_token(last, self.position)?;
            let t1 = std::time::Instant::now();
            self.write_logits(final_in_a);
            let t2 = std::time::Instant::now();
            let next = self.sample_next(&context_tokens, &sample_params)?;
            let t3 = std::time::Instant::now();
            if timing {
                eprintln!(
                    "[timing] forward={:.2}ms logits={:.2}ms sample={:.2}ms",
                    (t1 - t0).as_secs_f64() * 1e3,
                    (t2 - t1).as_secs_f64() * 1e3,
                    (t3 - t2).as_secs_f64() * 1e3,
                );
            }
            if next.is_eos {
                break;
            }
            context_tokens.push(next.token_id);
            output_tokens.push(next.token_id);
        }

        self.finish_generation(&output_tokens)
    }

    /// Generate text from `prompt`.
    ///
    /// # Errors
    ///
    /// Returns an error if no tokenizer is available, or Metal/sampling fails.
    #[allow(clippy::too_many_lines)]
    pub fn generate(
        &mut self,
        prompt: &str,
        params: &crate::GenerateParams,
    ) -> crate::Result<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Tokenizer("no tokenizer.json found in model directory".into()))?;
        let tokens = tokenizer.encode(prompt, true)?;
        let eos_tokens = tokenizer.eos_ids().to_vec();
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let sample_params = SamplingParams {
            temperature: params.temperature,
            top_p: params.top_p,
            eos_tokens,
            // Softcapping is applied on-device after the logits matvec
            // (see `write_logits`), so skip the CPU pass.
            logit_softcap: 0.0,
            ..SamplingParams::default()
        };
        let max_pos = self.config.max_position_embeddings.min(self.max_context);
        // The KV cache holds `max_pos` rows; positions past it would silently
        // overwrite earlier rows while attention still assumes linear
        // positions. Refuse oversized prompts instead of corrupting state.
        if tokens.len() > max_pos {
            return Err(Error::Generation(format!(
                "prompt is {} tokens but the effective context window is {max_pos}",
                tokens.len()
            )));
        }
        let output_limit = self.effective_output_limit(tokens.len(), params.max_tokens);
        if output_limit == 0 {
            return Ok(String::new());
        }

        let mut output_tokens = Vec::with_capacity(output_limit.min(4096));
        let mut context_tokens = tokens.clone();

        let prefill_t = std::time::Instant::now();
        {
            let final_in_a = self.prefill_prompt(&tokens)?;
            self.remember_prompt(&tokens, final_in_a);
            self.write_logits(final_in_a);
            let next = self.sample_next(&context_tokens, &sample_params)?;
            if next.is_eos {
                return self.detokenize(&output_tokens);
            }
            context_tokens.push(next.token_id);
            output_tokens.push(next.token_id);
        }
        if timing_enabled() {
            eprintln!(
                "[timing] prefill+first-token ({} prompt tokens) = {:.2}ms",
                tokens.len(),
                prefill_t.elapsed().as_secs_f64() * 1e3
            );
        }
        let loop_t = std::time::Instant::now();
        let loop_start_tokens = output_tokens.len();

        while output_tokens.len() < output_limit {
            let last = *output_tokens
                .last()
                .ok_or_else(|| Error::Generation("no token to continue from".into()))?;
            // Stop cleanly at the context capacity rather than wrapping
            // positions into already-occupied KV rows.
            if self.position + 1 >= max_pos {
                break;
            }
            self.position += 1;
            let timing = timing_enabled();
            let t0 = std::time::Instant::now();
            let final_in_a = self.forward_token(last, self.position)?;
            let t1 = std::time::Instant::now();
            self.write_logits(final_in_a);
            let t2 = std::time::Instant::now();
            let next = self.sample_next(&context_tokens, &sample_params)?;
            let t3 = std::time::Instant::now();
            if timing {
                eprintln!(
                    "[timing] forward={:.2}ms logits={:.2}ms sample={:.2}ms",
                    (t1 - t0).as_secs_f64() * 1e3,
                    (t2 - t1).as_secs_f64() * 1e3,
                    (t3 - t2).as_secs_f64() * 1e3,
                );
            }
            if next.is_eos {
                break;
            }
            context_tokens.push(next.token_id);
            output_tokens.push(next.token_id);
        }

        if timing_enabled() {
            let n = output_tokens.len() - loop_start_tokens;
            let w = loop_t.elapsed().as_secs_f64();
            eprintln!(
                "[timing] decode loop: {} tokens in {:.1}ms = {:.2}ms/tok ({:.1} tok/s)",
                n,
                w * 1e3,
                if n > 0 { w * 1e3 / n as f64 } else { 0.0 },
                if w > 0.0 { n as f64 / w } else { 0.0 },
            );
        }
        self.finish_generation(&output_tokens)
    }

    /// Detokenize the generated output.
    fn finish_generation(&self, output_tokens: &[u32]) -> crate::Result<String> {
        self.detokenize(output_tokens)
    }

    fn detokenize(&self, tokens: &[u32]) -> crate::Result<String> {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| Error::Tokenizer("no tokenizer available".into()))?
            .decode(tokens)
    }
}
