# Roadmap

_Single focus: `gemma-4-e2b-qat` (dense, QAT-tuned Gemma 4 E2B). Text, image, sampled-frame video, and WAV audio inputs are implemented; the downloaded Unsloth companion projector is packaged and loaded into Metal modality buffers._

## Done

- ✅ **Real tokenizer** — Hugging Face `tokenizers` integrated (`tokenizer.rs`), loads `tokenizer.json`, BOS/EOS aware, wired into `generate`.
- ✅ **Conversation reset** — `Engine::reset()` clears position + KV cache.
- ✅ **`.lma` archive + compression CLI** — `local-ai compress` produces a self-describing zstd archive (`lma.rs`); GGUF→.lma→load round-trip is tested.
- ✅ **Obtained the latest Unsloth mobile QAT assets** — fetched via a Kaggle internet-notebook relay (HF is Zscaler-blocked locally): QAT backbone GGUF, multimodal companion projector, tokenizer/config/processor metadata.
- ✅ **Correct quant dequant** — fixed Q4_0 (18-byte blocks, `(nibble-8)*d`, split-nibble layout), Q8_0 (34-byte blocks, signed int8), and the GGUF reader (32-byte data-section alignment + relative tensor offsets). These were the root cause of garbage output.
- ✅ **Full Gemma4 E2B forward pass** — per-layer head dims (256/512), NEOX RoPE per-layer base, KV sharing (layers 15–34), per-layer-input (PLE) injection, embedding scale, QK/V norms, logit soft-cap. **Generates correct, coherent text.** Spec: [`gemma4-forward-pass.md`](gemma4-forward-pass.md).
- ✅ **Sliding-window attention masking** — SWA layers enforce the 512-position window in decode and batched attention.
- ✅ **Proportional RoPE** — full-attention layers apply HF proportional RoPE semantics in decode and batch paths.
- ✅ **Batched prefill** — prompt prefill uses the shared multi-row path by default.
- ✅ **TurboQuant KV path** — TurboQuant is the sole live KV cache; all bit-widths (`tq2`/`tq3`/`tq4`) share one GPU encode + fused-attention path, so low-bit caches run at the same speed as 4-bit.
- ✅ **MLX-style prompt-prefix cache** — repeated prompts skip prefill and shared-prefix prompts restore quantized KV rows before running only the suffix by default; the cache is now a bounded MRU instead of a single previous-prompt entry.
- ✅ **MLX-style media embedding cache** — repeated image/audio/video inputs reuse cached soft embeddings within an engine session, avoiding expensive media tower reruns for follow-up chats over the same media.
- ✅ **SWA windowed KV dequant** — sliding-window layers dequantize only the active 512-token KV window.
- ✅ **SWA KV ring cap** — sliding-window layers store KV in a fixed physical ring (`position % ring_capacity`) instead of one row per logical position, so their footprint is constant regardless of context length; wired through single-seq and batched encode/dequant/attention, prefill copy, and prompt-cache snapshot/restore, with forced-wrap tests proving bit-identical in-window results vs a full cache. `LOCAL_AI_SWA_RING=0` disables it.
- ✅ **Multimodal companion packaging + GPU residency** — `compress` auto-bundles `mmproj-BF16.gguf` when present, `.lma` capability detection inspects companion tensors, the loader uploads companion tensors into device-resident modality buffers, and Metal vision patch/position/pool kernels are compiled and unit-tested.
- ✅ **Image path** — `chat --image` runs CPU decode/resize control plus Metal patch embedding, learned 2D position add, the 16-block SigLIP-style vision tower, pooling, RMSNorm, projector, soft-token replacement, and decoder prefill.
- ✅ **Cold vision-path optimization** — Gemma 4 vision QAT projections now use a tiled clamp-preserving Metal kernel (`clipped_linear_f16_nt`) instead of the serial reference clipped-linear shader; local `benchmark --multimodal-suite` improved cold decoded image from ~51 s to ~14 s and decoded-frame video from ~49 s to ~12 s.
- ✅ **Video path** — `chat --video` accepts a decoded frame image or frame directory, samples up to 32 frames, inserts timestamped Gemma4VideoProcessor-style 70-token video spans per frame, reuses the GPU vision/projector path, replaces video soft-token slots, and reaches decoder prefill. Container decode remains app/OS-side.
- ✅ **Audio path** — `chat --audio` accepts PCM16/float32 WAV, downmixes/resamples to Gemma 4's 16 kHz log-mel cadence, runs native log-mel extraction, GPU strided-conv frontend, 12-layer USM/Conformer-style tower, final audio projectors, soft-token replacement, and decoder prefill.
- ✅ **In-memory app handoff APIs** — library callers can pass `DecodedRgbImage`, `DecodedVideoFrame`/`DecodedVideo`, and `PcmAudio` through `MediaInput`, avoiding temporary files after OS-side media decode.
- ✅ **Multimodal prep simplification** — media placeholder insertion now shares one token/PLE helper, WAV and in-memory PCM share the same bounded downmix/resample implementation, and audio/video embedding projection is factored into reusable pipeline helpers.
- ✅ **Remaining decode sync reduction** — the single-token decode path batches projection/FFN/PLE spans while preserving the existing windowed attention boundary.
- ✅ **CLI simplification + benchmark refresh** — `chat`, `benchmark`, and `serve` default to `models/gemma-4-e2b-it`, context is auto-sized, prompt text can be passed naturally as multiple words, the benchmark command measures the same chat quality path by default, `benchmark --suite` provides a repeatable multi-prompt default-path suite, and `benchmark --cache-suite` measures prompt-cache miss/hit/shared-prefix behavior.
- ✅ **Multimodal benchmark suite** — `benchmark --multimodal-suite` generates deterministic in-memory decoded image/audio/video fixtures and times the full Gemma 4 soft-token paths without checking in temporary media artifacts.
- ✅ **Capability-first generation defaults** — `EngineConfig.max_context_length=0` targets the model's max context within memory budget, `GenerateParams::default()` asks for the model/context-max output budget (clamped before allocation), and default sampling is the quality path (`temperature=0.7`, `top_p=0.95`).

## Now (active priorities)

1. **App integration outside this engine crate** — call OS-side media decode for arbitrary containers and surface progress/failure/retry in the app UI.
2. **Harden continuous batching / serve** — the request queue, state manager, and multi-request KV scaffolding exist, but the server remains a demo path.
3. **Keep benchmarking app-like defaults** — refresh plain decode throughput numbers after each runtime change and keep the measured winner as the automatic path.

## Next

- Promote prompt-cache reuse beyond a single engine instance only if app-level session behavior shows it is worth the extra memory/accounting.

## Model scope

- **`gemma-4-e2b-qat`** is the only target: dense, QAT-tuned Gemma 4 E2B generation.
- Native context window: as defined by the model config (`max_position_embeddings`).
- Image, sampled-frame video, and WAV audio inputs have executable GPU paths using the packaged and Metal-loaded Unsloth companion assets. MoE and SSM remain out of scope for this model.
