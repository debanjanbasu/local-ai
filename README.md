# local-ai

On-device multimodal Gemma 4 E2B QAT inference for Apple hardware, written in Rust.

Runs the Unsloth Gemma 4 E2B QAT model locally with zero cloud dependency. The current implementation supports plain single-token autoregressive text decode, packages the downloaded multimodal companion projector, loads those modality tensors into Metal buffers, and runs image, video, and audio inputs into decoder soft tokens. Image uses the GPU SigLIP-style vision stack; video accepts sampled frame images/directories or in-memory decoded frames, follows Gemma4VideoProcessor's timestamped frame layout, and reuses the GPU vision path with 70 soft tokens per sampled frame; audio accepts PCM16/float32 WAV or in-memory PCM at common sample rates, resamples to Gemma 4's 16 kHz feature cadence internally, runs the 12-layer GPU USM/Conformer-style audio tower, final audio projectors, and decoder soft-token prefill. Execution is Metal GPU plus CPU.

## Key features

- **Gemma 4 E2B QAT multimodal path** with a full dense text decoder, PLE injection, KV sharing, proportional RoPE, 512-token SWA masking, and image/video/audio soft-token merge.
- **Metal GPU acceleration** for quantized-at-rest weights, tiled matvec/matmul/clipped-linear projections, RoPE, flash attention, normalization, elementwise kernels, Gemma 4 image/video encoding, and Gemma 4 audio encoding.
- **Quantized-at-rest weights** for GGUF `Q4_0` and `TQ2_0` matrices, with fused GPU dequant matvec/matmul kernels.
- **`.lma` single-file archives** bundling the quantized backbone, optional multimodal companion projector, tokenizer, and config with zstd frames.
- **TurboQuant KV cache** as the sole live KV path, **2-bit by default**; all bit-widths (`tq2`/`tq3`/`tq4`) share one GPU encode + fused-attention path, so 2-bit runs at the same speed as 4-bit while cutting KV ~8× vs FP16 (full 131072 context in ~0.35 GiB). Override with `LOCAL_AI_KV_QUANT`.
- **SWA KV ring cap**: sliding-window layers store KV in a fixed physical ring (`position % ring_capacity`), so their footprint stays constant as context grows — only the full-attention layers' KV scales with context. This is what lets the model's full context window fit on memory-constrained devices. `LOCAL_AI_SWA_RING=0` disables it.
- **Batched prompt prefill** plus an MLX-style prompt-prefix cache to reuse quantized KV rows across repeated/shared-prefix prompts.
- **Bounded media embedding cache** to reuse already-computed image/audio/video soft embeddings for follow-up chats over the same media.
- **Capability-first defaults**: model-max context when memory allows, model/context-max output, and quality sampling (`temperature=0.7`, `top_p=0.95`) unless a caller explicitly chooses a shorter/greedy run.
- **Continuous batching demo** and request/state management scaffolding.

## Supported models

| Model | Architecture | Context | Modalities |
|-------|-------------|---------|------------|
| `gemma-4-e2b-qat` | Dense (QAT tuned, TQ2_0 / Q4_0 mix) | model max, device-budgeted | Text + image + video frames + WAV audio implemented |

**`gemma-4-e2b-qat`** is the sole target. Gemma 4 E2B is multimodal, and the current Unsloth install includes the `mmproj-BF16.gguf` companion projector in the `.lma` bundle. The loader keeps those modality tensors resident on the Apple GPU. `chat --image <path>` runs the image through GPU patch embedding, learned 2D position add, the 16-block SigLIP-style encoder, pooling, projection, and decoder soft-token prefill. `chat --video <frame-or-dir>` samples up to 32 decoded frame images, inserts timestamped `boi + <|video|> × 70 + eoi` spans per frame, encodes each frame through the same GPU vision/projector path, and replaces the video soft-token slots before decoder prefill; mp4/mov container decode should happen in the app/OS media stack (for example AVFoundation on Apple platforms). `chat --audio <path.wav>` decodes PCM16/float32 WAV, downmixes and resamples to Gemma 4's 16 kHz log-mel cadence, reserves duration-derived audio slots, runs the GPU strided conv frontend, 12-layer USM/Conformer-style tower, final audio projectors, and decoder soft-token prefill. Library callers can bypass temp files with `MediaInput::DecodedImage`, `MediaInput::DecodedVideo`, and `MediaInput::PcmAudio`. Non-WAV containers such as m4a/mp3 should be decoded by the app or OS media stack before handing PCM to the engine.

## Quick start

**Requirements:** macOS with Apple Silicon (M1+), Rust toolchain, Xcode (for Metal shaders).

```bash
# Build
cargo build -p local-engine --release

# Run with the default local model directory (models/gemma-4-e2b-it)
cargo run -p local-engine --release --bin local-ai -- chat "Hello"

# Override the model directory only when needed
cargo run -p local-engine --release --bin local-ai -- chat --model <model-dir> "Hello"

# Image prompt smoke path (GPU SigLIP-style vision tower + decoder soft-token prefill)
cargo run -p local-engine --release --bin local-ai -- chat --image <image.png> "Describe this image"

# Video prompt smoke path (directory of decoded sampled frames)
cargo run -p local-engine --release --bin local-ai -- chat --video <frames-dir> "Describe this video"

# Default-path benchmark suites, including deterministic in-memory multimodal media
cargo run -p local-engine --release --bin local-ai -- benchmark --suite
cargo run -p local-engine --release --bin local-ai -- benchmark --multimodal-suite
```

## CLI reference

| Command | Description |
|---------|-------------|
| `chat` | Generate text from a prompt |
| `serve` | Run the continuous batching demo |
| `compress` | Compress a model for on-device inference |
| `benchmark` | Run the default chat-path inference benchmark; use `--suite` for the built-in multi-prompt suite, `--cache-suite` for prompt-cache hit/miss cases, and `--multimodal-suite` for deterministic in-memory image/audio/video cases |

Run `local-ai <command> --help` for details on any subcommand.

The CLI intentionally keeps the common path small: it uses the model's maximum context window when memory allows, a model/context-max output budget, quality sampling defaults, KV cache, and prompt caching for best default capability/quality/performance. Expert/debug overrides remain available through the Rust API and environment variables, but normal usage should not require tuning them.

## App integration

For iOS/macOS apps, keep container decode in the OS media stack and pass decoded data to the engine:

- Use AVFoundation or equivalent to decode `mp4`/`mov` into sampled RGB frames, then call `generate_multimodal_chat` with `MediaInput::DecodedVideo { frames }`.
- Use AVFoundation or equivalent to decode `m4a`/`mp3`/`aac`/`flac` into f32 PCM, then pass `MediaInput::PcmAudio { audio }`.
- Use `MediaInput::DecodedImage { image }` for already-decoded still images.

## Compression

Pack a model directory into a single self-describing `.lma` archive:

```bash
cargo run -p local-engine --release --bin local-ai -- compress --model <model-dir>
```

This reads the model's GGUF weights, `config.json`, optional `mmproj-BF16.gguf` companion projector, and tokenizer metadata, then writes `<model-dir>/model.lma` — a zstd-compressed container that embeds the config plus every tensor (raw quantized payload) in its own frame. At load time `Pipeline::new_qat` automatically prefers `model.lma` when present and falls back to the GGUF file otherwise.

Options: `--out <path.lma>` to choose the output path, `--level <1-22>` for the zstd level (default 22).

## Workspace structure

| Crate | Purpose |
|-------|---------|
| `local-core` | Model configs, capability types, shared core utilities |
| `local-metal` | Metal context, kernel registry, command and buffer abstractions |
| `local-engine` | Model archive loading, runtime pipeline, sampling, CLI, model execution |

## Documentation

- `docs/README.md` -- docs index
- `docs/STATUS.md` -- code-verified implementation status
- `docs/ROADMAP.md` -- active priorities
- `docs/ARCHITECTURE.md` -- architecture snapshot

## Development

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run with verbose logging
RUST_LOG=debug cargo run -p local-engine --release --bin local-ai -- chat --model <model-dir>
```

---

This repository is under active development.
