# Current Architecture Snapshot

_Scope: `gemma-4-e2b-qat` (dense, QAT-tuned Gemma 4 E2B). Text, image input, sampled-frame video input, and WAV audio input are implemented._

## Workspace layout

- `local-core` — model config types (`Gemma4QATConfig`) and shared core utilities
- `local-metal` — Metal context, kernel registry, shader library, command/buffer abstractions, memory planner
- `local-engine` — archive loading, GGUF parsing, QAT dequant, dense decode pipeline, sampling, KV cache, CLI

## Runtime data path

1. **Model init** (`local-engine/src/pipeline.rs` `Pipeline::new_qat`)
   - `load_config_and_weights` resolves the weight source in order: `model.lma` → Unsloth QAT GGUF (`gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf`) + `config.json`
   - The `.lma` archive (`lma.rs`) is a self-describing zstd container built on `archive.rs`: a metadata frame holds the embedded `config.json` plus tensor indexes for backbone and optional multimodal companion tensors, and one frame per tensor holds the raw quantized payload
   - Keeps Q4_0 and TQ2_0 matrices quantized at rest when supported by fused Metal matvec/matmul kernels; remaining tensors decode through the shared `decode_tensor_to_f16` path (`qat_recover.rs`)
   - Builds per-layer Metal weight buffers, a persistent tied token-embedding/output-norm buffer, and device-resident modality buffers for the bundled Unsloth companion projector tensors
   - Allocates scratch buffers and TurboQuant per-layer KV caches (`layer.rs`, `kv_cache/`)

2. **Text decode** (`Pipeline::generate`)
   - Tokenizes the prompt with the Hugging Face `tokenizers` library (`tokenizer.rs`, loads `tokenizer.json`, prepends BOS, stops on `<eos>` / `<end_of_turn>`)
   - Prefills prompt tokens through the batched multi-row path, restoring an in-process MRU prompt-prefix cache first when a recent request shares the same token prefix
   - `Pipeline::reset` clears the token position and KV cache for a fresh sequence
   - Each layer runs a dense decoder step (`layer.rs` `TransformerLayer::forward` / `forward_batch`):
     RMSNorm → Q/K/V matvec → QK-norm → RoPE → KV-cache write → flash attention →
     output projection → residual → per-layer-input (PLE) injection → pre-FFN norm →
     GeGLU FFN → post-FFN norm → residual → optional layer-output scale
   - Final RMSNorm + logits projection, then sampling (`sampler.rs`)

3. **Serving / batching**
   - `chat`, `benchmark`, and `serve` default to `models/gemma-4-e2b-it`; model-max context when memory allows, model/context-max output, quality sampling (`temperature=0.7`, `top_p=0.95`), KV cache, and prompt caching are auto-selected for the normal path. `benchmark --suite`, `--cache-suite`, and `--multimodal-suite` cover text defaults, prompt-cache reuse, and deterministic in-memory image/audio/video media paths.
   - `serve` (`cli/serve.rs`, `cli/serve_h3.rs`) runs a production continuous-batching, OpenAI-compatible streaming server: a single GPU **worker thread** owns the non-`Send` pipeline and runs the lifecycle loop, while an **axum (HTTP/1.1 + HTTP/2)** front-end and a native **HTTP/3 (QUIC)** listener fan concurrent requests into it. Routes: `GET /health`, `POST /v1/chat/completions`, `POST /v1/completions`. Features: SSE token streaming (`chat.completion.chunk` deltas), bearer-token auth (`--api-key` / `LOCAL_AI_API_KEY`), in-flight backpressure (`--max-inflight` → `503`), zstd/brotli/gzip response compression, loopback-bound by default (`--host 0.0.0.0` to expose), and graceful shutdown.
   - The continuous-batching machinery is fully wired: `continuous_batching.rs` + `request_queue.rs` (priority admission, prefix ref-counting), `kv_cache/quant_unified_pool.rs` (by-lane TurboQuant KV pool), `pipeline/batched.rs` (batched decode forward over N lanes + per-lane prefill), and `pipeline/runner.rs` (the scheduler + `serve_batched` loop). Batched decode runs on the same TurboQuant KV backing as single-sequence decode (`flash_attention_tq_batched`); there is no FP16 KV path.

## Metal kernels & shaders

- Compute pipelines are registered in `local-metal/src/kernels.rs` from the compiled metallib (`local-metal/build.rs` compiles `shaders/*.metal`).
- Active shader set covers the dense decode path: `rms_norm`, `gelu`, `silu`, `softmax`, `embedding`, `rope` (+ YaRN/batch variants), `matvec_f16`, `matmul_f16`, `matmul_f16_nt`, `clipped_linear_f16_nt`, `qk_norm`, `elementwise_mul`, `residual_add`, `scale_in_place`, `clipped_linear`, `attention_sliding`, `attention_flash`, `flash_attention_prefill(_masked)`, `flash_decoding`, `flash_attention_tq(_batched)`, `encode_kv_turboquant(_batched)`, `tq2_dequant`, `dequantize_kv`, `bf16_to_fp16`, `attention_gate` (provides `scale_by_sigmoid_scalar_f16`), and Gemma 4 vision helpers (`vision_patch_embed`, `vision_add_position_embedding`, `vision_avg_pool_2d`, `vision_rope`).
- MoE, mamba/SSM, IQ-quantization, and the old Qwen-era vision/audio kernels have been removed. Gemma 4 image, sampled-frame video, and WAV audio support are implemented as Gemma-specific processor + companion projector/encoder paths using the optimized Unsloth assets now packaged in the `.lma` and loaded into Metal buffers. The vision tower uses the tiled `clipped_linear_f16_nt` kernel for QAT clamp-preserving projections instead of the simple serial-reference clipped-linear path.

## Quantization

- **Weights:** Q4_0 and TQ2_0 matrices stay in GGUF block format at rest and run through fused dequant matvec/matmul kernels; unsupported or small tensors are decoded to FP16 at load (`qat_recover.rs`, `layer.rs`, `shaders/matvec_quant.metal`).
- **KV cache:** TurboQuant is the **only** KV backing for both single-sequence and continuous-batching decode — there is no FP16 KV path. Encode uses randomized Hadamard rotation + Lloyd–Max codes + f16 norm (GPU `encode_kv_turboquant` / `encode_kv_turboquant_batched`); the fused attention kernels (`flash_attention_tq` / `flash_attention_tq_batched`) read the packed codes directly (rotate query → attend over codes → inverse-rotate output) with no FP16 expansion at every bit-width. `dequantize_kv_turboquant` remains for windowed/ring restore paths. Bit-width is `LOCAL_AI_KV_QUANT` (`tq2`/`tq3`/`tq4`, **default 2-bit** ≈ 8× KV cut vs FP16); all three widths use the same GPU encode + fused-attention path, so 2-bit runs at the same speed as 4-bit. 2-bit is the default everywhere — its smallest footprint is what lets every device hold the model's full context window, and there is no speed penalty for choosing it.
- **SWA KV ring cap:** sliding-window layers store their KV in a fixed physical ring sized to the window (rounded up to a power of two) plus prefill slack, rather than one row per logical position. Logical position `p` maps to physical slot `p % ring_capacity`; the GPU kernels (`flash_attention_tq`, `dequantize_kv`, `encode_kv_turboquant`, and their batched forms) take a `ring_capacity` scalar (`0` = absolute/full-attention addressing). The ring is wired through single-seq encode/dequant/fused-attention, the by-lane batched pool (`quant_unified_pool.rs`, with the same modulus mirrored so a wrapped prefill copies slot-for-slot), prefill copy, and prompt-cache snapshot/restore (snapshots only cover an unwrapped prefix). Because SWA layers' KV no longer grows with context, the adaptive memory model (`pipeline.rs::kv_memory_model`) treats them as a fixed floor and only the full-attention layers + dequant scratch as the per-position rate. `LOCAL_AI_SWA_RING=0` forces the full absolute allocation.
- **Prompt/media caches:** `Pipeline` keeps a small MRU of recent prompts' token IDs, quantized KV prefix rows, and exact-prompt final hidden state in CPU memory. Repeated prompts skip prefill entirely; shared-prefix prompts restore the common KV rows and only run the suffix. Defaults are bounded (`LOCAL_AI_PROMPT_CACHE_ENTRIES`, default 4; `LOCAL_AI_PROMPT_CACHE_MAX_TOKENS`, default 8192) so the optimization does not hold multiple full-context snapshots by accident. A second bounded MRU (`LOCAL_AI_MEDIA_CACHE_ENTRIES`, default 8) stores already-computed multimodal soft embeddings so follow-up chats over the same image/audio/video avoid rerunning the media towers.

## Execution hardware

- Model execution uses Metal GPU kernels plus CPU control, tokenization, archive IO, and bounded tensor/embedding streaming.
- No separate accelerator offload crate or generated model-artifact path is part of the current workspace.
