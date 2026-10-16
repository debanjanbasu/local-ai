# Native audio embeddings: EmbeddingGemma 2

EmbeddingGemma 2 embeds audio into the same 768-dimensional space as text (see
[EMBEDDINGS.md](EMBEDDINGS.md) for the text path). The audio path runs natively on Metal:
decoding and resampling, the log-mel front end, the 12-layer `gemma4_audio` Conformer,
`embed_audio`, and the shared text encoder.

Audio is available through the **native Rust API only**. OpenAI's `POST /v1/embeddings`
takes text or token ids, and no current OpenAI mechanism carries audio into an
embeddings request, so the server does not invent one. The text endpoint is unchanged.

## Install

The audio encoder is an optional add-on to the text installation. It lives in the same
directory and needs the text artifact:

```sh
python3 tools/convert_embeddinggemma2_audio.py --source cache/embeddinggemma-2
```

The command reads the same pinned Kaggle checkpoint as the text converter and writes two
files into `models/embeddinggemma2-text/`:

- `embeddinggemma2-audio-bf16.bin`: 611,720,448 bytes, SHA-256
  `3b3354a4df46fb2463d5b10b2d6bfa80e616aa07f1f87c658cdd184023dbe3bc`.
- `manifest-audio.json`.

Every source file is size- and hash-checked, and the output must match the pinned size
and hash. It uses the standard library only.

The artifact uses the text artifact's container (magic `EG2AUD\0\1`, format
`embeddinggemma2-audio`). When the runtime opens it, it checks the header, the
architecture, every tensor's bounds, dtype and shape, and the payload's SHA-256. It holds:

- the Linear matrices as the checkpoint's own BF16 values (copied byte for byte);
- the conv kernels, norm weights, per-dimension attention scales and the `output_proj`
  bias, widened to F32 (exact);
- the clamp bounds of all 120 clipped linears, widened to F32 (exact). Each linear stores
  `[input_min, input_max, output_min, output_max]` as one vector `<linear>.clip`.

## API

```rust
use local_engine::embedding::{EmbeddingModel, decode_audio, installed_model};

let dir = installed_model(Path::new("models")).unwrap();
let mut model = EmbeddingModel::open(&dir)?;      // text encoder
model.load_audio(&dir)?;      // optional preload (+ ~610 MB of GPU memory); else loaded on first use
let clip = decode_audio(std::fs::read("dog.flac")?)?;  // 16 kHz mono f32
let audio = model.embed_audio(&[clip.as_slice()], &mut || Ok(()))?;   // [768], L2-normalized
let query = model.embed_texts(&["task: search result | query: a dog barks"], &mut || Ok(()))?;
let score: f32 = audio[0].iter().zip(&query[0]).map(|(a, b)| a * b).sum();
```

- **`decode_audio(bytes)`** decodes with Symphonia, probing the container from the bytes.
  It averages the channels to mono and resamples any other rate to 16 kHz with Rubato's
  synchronous FFT resampler. Decoding stops with an error past 600 s. Supported formats:

  | Container | Codecs | Test result (`tests/fixtures/audio-formats/`) |
  | --- | --- | --- |
  | WAV | PCM | decodes to the exact samples |
  | FLAC | FLAC | decodes to the exact samples |
  | MP4 / M4A | ALAC | decodes to the exact samples |
  | MP4 / M4A | AAC-LC | correlation 0.995; the encoder's priming and padding are kept (+1,216 samples, mostly 1,024 of near-silence at the start), because MP4 edit lists are not applied |
  | MP3 | MPEG-1/2 Layer III | correlation 0.99999 |
  | Ogg | Vorbis | correlation 0.99999 |
  | Matroska / WebM | FLAC, Vorbis | FLAC decodes to the exact samples; Vorbis-in-WebM reaches correlation 0.99999, with the encoder's padding kept (+192 samples) |

  Opus (in Ogg or WebM) and every other codec are refused with `InvalidArgument`. The
  fixtures are 0.5 s of a two-tone signal, encoded once per format by FFmpeg 9.0. The
  test runs them through this decoder and through the transcription decoder, which uses
  the same codec set. Correlation is the best normalized cross-correlation within
  ±2,048 samples.
- **`to_mono_16k(interleaved, channels, rate)`** does the same downmix and resampling for
  PCM you already have.
- **`EmbeddingModel::embed_audio(clips, pace)`** returns one normalized vector per clip,
  in order. It is `embed_multimodal` with one `MediaPart::Audio` part per input, the
  same sequence assembly the image and video path uses; `MediaPart::Audio` can also be
  combined with text or image parts in one `embed_multimodal` input.
  - Each clip is 16 kHz mono F32 samples, from 161 samples up to 30 s (480,000 samples).
  - Longer clips are refused with `Error::ContextOverflow`, not truncated. The reference
    feature extractor would silently keep only the first 30 s. Split long audio and pool
    the pieces yourself.
  - Clips are packed several per GPU pass. The result does not depend on the packing (see
    Verification).
  - `pace` runs between command buffers, exactly as for `embed_ids`. The server's
    `EngineActivity` hook can be passed so audio work yields to Bonsai decode.
- **`audio_soft_tokens(clips, pace)`** returns the encoder's output, `[tokens, 512]` per
  clip: the text encoder's input embeddings for the audio. It is mainly for diagnostics.
- **`audio_features(samples)`, `FeatureExtractor`, `frame_counts`, `audio_tokens`** expose
  the log-mel front end and the token arithmetic: 25 tokens per second, 250 for 10 s,
  750 for 30 s.
- **Matryoshka:** `truncate_normalized` works for audio vectors as for text. Phase 1
  measured 512-d as near-lossless for audio and 256-d at −1.1/−0.8 mean R. 128-d costs 8
  points.

**Prompts.** Audio takes no prompt: the reference embeds media with no text. For
text→audio search, embed the text query with the model card's query prompt
(`task: search result | query: …`, i.e. `EmbeddingPrompt::Query`). For audio→text search,
the document prompt (`title: none | text: …`) is the reference protocol, though raw
captions score slightly better (see Retrieval).

## What runs

1. **Front end (CPU, F64).** This is transformers 5.19.0's `Gemma4AudioFeatureExtractor`
   for a single clip, reproduced step for step:
   - Pad the clip with zeros to a multiple of 128 samples and prepend 160 zeros.
   - Take a 321-sample window every 160 samples and use its first 320 samples.
   - Multiply by a periodic Hann window. The window values and the products are F32, as
     numpy computes them.
   - Run a 512-point real FFT (F64) and take the magnitudes.
   - Apply 128 HTK mel filters over 0–8 kHz (unnormalized triangles, built like
     `mel_filter_bank`).
   - Take `ln(mel + 0.001)` and round to F32.
   - Frames whose window reaches past the real samples are masked to zero.
2. **Conformer (GPU, `shaders/embedding_audio.metal`, F32 activations).**
   - **Stem:** two 3×3 stride-2 convolutions, each followed by LayerNorm and ReLU. The
     input frame mask is applied after the first, as in the reference. Then a
     1024→1024 projection.
   - **12 layers**, each with:
     - two half-step macaron feed-forwards (1024→4096→1024, SiLU);
     - chunked local self-attention: 8 heads × 128. Each frame attends to itself and the
       11 frames before it, which is the reference's mask (`0 ≤ q − k < 12`, chunk 12,
       context 13). Logits are `(q·s)(k·ln(1+e)/ln 2) + (q·s)·r(q−k)`, where
       `s = 128^-½ / ln 2 · softplus(per_dim_scale)` and
       `r = relative_k_proj(sinusoid(distance))` is precomputed per layer at load. The
       logits are soft-capped at 50, then softmaxed;
     - a light-conv block: GLU fused with the causal depthwise conv (k = 5), RMSNorm, SiLU,
       and a residual linear;
     - RMSNorm after each sub-block.
   - **Clipped linears:** every one of them clamps its input and output to the
     checkpoint's bounds inside the GEMM, with SiLU, bias or residual fused into the same
     epilogue.
   - **Dropped clamps:** the reference's `gradient_clipping` clamps (±1e10) are no-ops in
     F32 for finite activations and are not applied.
   - **Output:** `output_proj` (1024→1536 with bias), then `embed_audio`: RMSNorm without
     scale and 1536→512.
   - **Packing:** several clips share a pass. Attention and the causal conv read each
     token's sequence start, so clips never see each other.
   - **Kernel reuse:** the BF16 tiling of the GEMM is the text encoder's, extended with the
     clamp/bias/activation/accumulate epilogue. The RMSNorm kernel is the text encoder's.
3. **Shared encoder.**
   - Build `<bos> <start_of_audio> soft tokens <end_of_audio> <eos>`: ids 2, 256000, the
     soft tokens, 258883, 1. The bracketing tokens come from the token table (×√512); the
     soft tokens are used as given. This is the reference processor's sequence: the
     reference kernel recorded `[2, 256000, 258881 × N, 258883, 1]`.
   - The per-layer embeddings are computed from these input embeddings, as the reference
     does.
   - Then run the same 24 layers as text, mean-pool over all positions, project 512→768
     and L2-normalize. The text path's code is shared: only the input embeddings differ.

**Precision.** The matrices are BF16, which is exact to the checkpoint. Everything else is
F32: activations, accumulation, softmax and norms. Phase 1 measured int8 audio weights as
retrieval-lossless (+0.2/−0.3 mean R). The BF16 path already meets the parity gate, and
int8 was not needed to fit memory, so it is not shipped. (The text path's int8 parity
failure, min cosine 0.9976, came from the shared decoder that audio also uses.)

## Verification

The reference is `eg2-audioref`, a private Kaggle CPU kernel running FP32,
sentence-transformers 6.1.0, transformers 5.19.0 and torch 2.11.0. It used the same
`nickkar30/audiocaps` mirror as Phase 1 and the hash-identical model files. It embedded 54
inputs one at a time, through the same `Sequential` forward as `encode()` (checked: cosine
1.0 against `encode`). The inputs:

- 41 full AudioCaps test clips (every 24th);
- 8 crops (800 to 159,999 samples, including lengths that are not multiples of 128 or 160);
- one 30 s concatenation (750 tokens);
- a 1 kHz tone, Gaussian noise and silence;
- a 44.1 kHz stereo probe.

The kernel recorded the waveform, input ids, features and mask, the `embed_audio` soft
tokens, the final embedding, and per-layer intermediates for two clips.

| Gate | Threshold | Result |
| --- | --- | --- |
| log-mel features, 54 inputs (47,153 frames × 128) | frame counts identical, F32 rounding | **54/54 counts identical; max abs difference 4.8e-7** |
| FLAC decode (Symphonia) vs the reference's soundfile decode, 41 clips | bit-identical | **41/41 identical** |
| soft tokens (`embed_audio` output) | — | per-clip mean cosine 1.000000, worst token 0.999999 |
| embedding cosine, native vs reference, 54 inputs | mean ≥ 0.9995, min ≥ 0.999 | **mean 1.000000, min 0.999999** (all 54; the worst is the tone) |
| packing independence (each clip alone vs packed) | ≥ 0.99999 | **54/54 bitwise identical** |
| committed fixture test (5 inputs, `#[ignore]`d real-model test) | same gate | pass |
| text + audio in one sequence, 4 inputs ([mixed-modality inputs](EMBEDDINGS.md#mixed-modality-inputs)) | same gate | **cosine 1.000000 each**; token layouts identical |

Native resampling is not part of the gate: the reference fed 16 kHz audio. As a probe, a
44.1 kHz stereo version of one clip went through the native downmix and FFT resampler, and
the reference's version went through librosa (soxr_hq). They agree to 0.99903 embedding cosine.

### Retrieval: AudioCaps test

The set is 963 clips and 4,815 captions, decoded natively from the mirror's FLACs, with
exact cosine search at 768-d. The reference protocol embeds captions with the query prompt
for text→audio and with the document prompt for audio→text.

| Direction | Captions | Kaggle FP32 (Phase 1) | Native BF16 | Δ mean R |
| --- | --- | --- | --- | ---: |
| text→audio | query prompt (reference protocol) | 8.0 / 25.1 / 37.3 (mean 23.5) | 8.0 / 25.1 / 37.3 (mean 23.5) | 0.0 |
| audio→text | document prompt (reference protocol) | 7.6 / 22.7 / 35.1 (mean 21.8) | 7.6 / 22.9 / 34.8 (mean 21.8) | 0.0 |
| text→audio | raw | 9.0 / 26.0 / 38.1 (mean 24.3) | 9.0 / 26.0 / 38.1 (mean 24.3) | 0.0 |
| audio→text | raw | 7.8 / 24.4 / 38.5 (mean 23.6) | 7.8 / 24.4 / 38.4 (mean 23.5) | −0.1 |
| text→audio | document prompt | — | 7.6 / 23.6 / 34.9 (mean 22.0) | |
| audio→text | query prompt | — | 8.0 / 22.6 / 36.4 (mean 22.4) | |

The gate is within 0.5 mean R of Phase 1: it passes, and the native path reproduces the
reference ranking. Embedding the 963 clips took 82.9 s, or 11.6 clips/s (115× real time).
Decoding all 963 FLACs took 4.0 s on the CPU.

### The low absolute AudioCaps score

Phase 1 found AudioCaps text→audio R@1 at 8%, against roughly 35–40% for CLAP-style
models. Cheap checks to explain the gap:

- **Not a pipeline mismatch, at least not between this runtime and transformers.** This
  implementation is independent and shares no code with transformers. It decodes the FLACs
  bit-identically to soundfile, matches transformers' log-mel features to 4.8e-7, and
  matches the FP32 embeddings to cosine ≥ 0.999999. Its AudioCaps numbers are the
  reference's to ±0.1.
- **Not the caption prompt.** All nine caption forms fall within 22.0–24.3 mean R for
  text→audio. Raw captions are best (24.3), as Phase 1 found.
- **Not the input level.** Clip loudness is ordinary: median −21.9 dBFS RMS (5th–95th
  percentile −41 to −11.5), peaks 0.12–1.0. With the 0.001 mel floor, level has little
  influence at these amplitudes.
- **It leans to speech.** Captions that mention speech score 25.7 mean R text→audio (2,202
  captions); the other 2,613 score 21.6. This fits a speech-heavy audio training mix (the
  card's audio benchmark is MSEB, which is mostly speech).
- **Audio embeddings are crowded.** The mean pairwise cosine is 0.759 among clips and 0.678
  among query-prompted captions, and the gap between the two modality means has norm 0.43.
  Centering each modality on its own mean is a transductive diagnostic that needs the whole
  gallery, and it is not shipped. It raises mean R to 25.8 (t2m) and 24.3 (m2t): a gain,
  but not the missing 10–15 points.
- **The text side is not the bottleneck.** Retrieving another caption of the same clip from
  a caption gives R@1/5/10 41.6 / 64.2 / 74.7. Even with text only, the five AudioCaps
  captions of a clip are far apart.

**Unresolved:** whether transformers' `Gemma4AudioFeatureExtractor` matches the front end
Google trained with. Checking that needs the original model or a published audio-retrieval
number to compare against, and neither is available. The evidence points to model
behaviour on environmental sound rather than to a preprocessing error on our side.
Relative comparisons, such as int8 against BF16 or prompt choices, are valid.

### Throughput

Measured under the GPU lock with `embedding-audio-eval bench`, best of three, synthetic
noise clips:

| Clips | Wall time | Clips/s | Real-time factor |
| --- | ---: | ---: | ---: |
| 1 × 10 s | 94 ms | 10.6 | 106× |
| 32 × 10 s | 2,779 ms | 11.5 | 115× |
| 1 × 30 s (750 tokens) | 326 ms | 3.1 | 92× |
| 8 × 30 s | 2,550 ms | 3.1 | 94× |

Most of the cost is the Conformer: about 40 M multiply-adds per token per layer, against
5.5 M for a text-encoder layer. A 10 s clip is 250 tower tokens, then 254 tokens through the
shared encoder. The GPU was shared with other work's lock queue but held exclusively for the
run. The Conformer submits one layer per command buffer (≥ 128 tokens) before the pacing
hook runs. This keeps a concurrent decode step's wait to roughly one layer of a pass, a few
milliseconds for a single 10 s clip (the whole clip takes 94 ms end to end). Concurrent-decode impact was not measured for audio.

## Tools

- `tools/convert_embeddinggemma2_audio.py`: the converter (`tools/test_convert_embeddinggemma2_audio.py`
  covers it).
- `embedding-audio-eval features REF_DIR [AUDIOCAPS_DIR]`: front-end and FLAC-decode parity
  (CPU).
- `embedding-audio-eval parity REF_DIR`: soft-token and embedding parity, packing, and the
  resampling probe.
- `embedding-audio-eval retrieval AUDIOCAPS_DIR`: AudioCaps R@1/5/10 for every caption form.
  `AUDIO_EVAL_DUMP=DIR` also writes the raw embeddings.
- `embedding-audio-eval bench [--clips N] [--seconds S]`.
- `local-engine/tests/fixtures/embeddinggemma2-audio-reference.jsonl`: three AudioCaps crops
  (waveform, features, embedding) and the tone and silence embeddings. The log-mel test
  always runs; the embedding test is ignored unless the model is installed.
- `local-engine/tests/fixtures/audio-formats/`: one short two-tone clip per supported
  container and codec, plus an Opus file that must be refused. The decode test always runs.

## Limitations

- Audio is native-API only. There is no HTTP endpoint, because no current OpenAI
  embeddings mechanism carries audio.
- Clips are 30 s at most. The model's soft-token budget is 750 per clip, and longer audio
  must be split by the caller.
- Text and audio in one sequence work through `embed_multimodal` and match the reference
  (see [mixed-modality inputs](EMBEDDINGS.md#mixed-modality-inputs)). The audio
  retrieval protocol above embeds media without text, as the model card does.
- The resampler is Rubato's FFT resampler, not soxr. On non-16 kHz input the embeddings
  can therefore differ slightly from a librosa-fed reference (see the probe above).
- Images and video are documented in [EMBEDDINGS.md](EMBEDDINGS.md#images-and-video).
