# Native text embeddings: EmbeddingGemma 2

The server can embed text natively with Google's **EmbeddingGemma 2**
(`google/embeddinggemma-2`, 744 M parameters with its image, audio and video
encoders; 271 M on the text path), on Metal, beside the Bonsai checkpoint. It
is optional and opt-in by installation: install the artifact below and
`POST /v1/embeddings` appears; without it nothing changes and the route is
`404`. Text and code are served over HTTP; **images and video** are native
through the Rust API once the optional vision artifact is installed (see
[Images and video](#images-and-video); OpenAI's embeddings API has no media
input, so there is no HTTP surface for them). Audio: see
[Multimodal plan](#multimodal-plan).

## Install

The checkpoint comes from Kaggle Models (accept the Gemma terms on Kaggle for
your account first). The conversion is offline, standard library only, checks
every source file against pinned sizes and SHA-256s, and checks its own output
against a pinned SHA-256:

```bash
kaggle models instances versions download \
  google/embeddinggemma-2/transformers/embeddinggemma-2/1 \
  -p cache/embeddinggemma-2 --untar
python3 tools/convert_embeddinggemma2.py --source cache/embeddinggemma-2
```

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `models/embeddinggemma2-text/embeddinggemma2-text-bf16.bin` | 542,996,480 | `dd745d5e7402ba8e3e654a978adb70ab4af97e780f000742d6a8a309773a492c` |
| `models/embeddinggemma2-text/tokenizer.json` | 32,170,510 | `4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4` |

Source pins (Kaggle `google/embeddinggemma-2/transformers/embeddinggemma-2/1`):
`model.safetensors` 1,488,915,288 B `197a3296…5553d79`, `tokenizer.json` as
above, `config.json` 4,455 B `b8f1e993…4bb3`. The server looks for
`embeddinggemma2-text/` in the same `models/` directory as the Bonsai GGUF and
verifies the artifact's payload SHA-256 on every load (about 0.3 s).

## What runs

- **Architecture** (from the pinned transformers 5.19.0 source): 24
  bidirectional encoder layers, hidden 512, gated GELU-tanh MLP 2048; five
  sliding-window layers (window 512, `|i − j| ≤ 512`, 4 query heads × 256, 2 KV
  heads, RoPE θ 1e4) per global layer (4 × 512, 1 KV head, θ 1e6); q/k RMSNorm
  with scale, v RMSNorm without, attention scale 1.0; per-layer embeddings
  (each layer's slice of a 512→12,288 projection of the input embeddings,
  ×1/√512, RMS-normalized, gating a 512→512→512 block); a layer scalar; final
  RMSNorm; mean pooling over every token (`<bos>`, prompt and `<eos>`
  included); 512→768 projection; L2 normalization.
- **Precision:** every matrix keeps the checkpoint's BF16 values (543 MB on
  disk). The 268 MB token table is memory-mapped and read on the CPU (only the
  rows of tokens actually seen become resident); the encoder's 273 MB of
  matrices live in one Metal buffer. Activations, accumulation and softmax are
  F32: the model card rules out F16 activations, which overflow.
- **Kernels** (`shaders/embedding.metal`): a BF16×F32 GEMM on 8×8 simdgroup
  matrices (32-token × 32-row tiles), RMSNorm, fused residual RMSNorm-add with
  the layer scalar, RoPE, gated GELU, and packed bidirectional attention with
  per-token sequence bounds and an optional window. Several inputs share one
  pass (packed up to 1,024 tokens; a longer input takes its own) with no
  padding.
- **Tokenizer:** the HF `tokenizers` crate on the checkpoint's own
  `tokenizer.json` (`<bos>` … `<eos>` from its post-processor).
- **Limits:** at most 2,048 tokens per input (the model's context is 8,192;
  attention here is quadratic and unblocked). Longer inputs are refused, never
  truncated.

### Why BF16 and not int8

Per-row int8 for every matrix (274 MB) was built and run first. It is
lossless for retrieval (Kaggle FP32 fake-quant, `docs` of the research run:
−0.01 to +0.17 nDCG@10 on SciFact, NFCorpus, CosQA and CodeTrans), and on the
1,000-text parity set it reached mean cosine 0.99984 to the FP32 reference —
but its worst input (`def f(x):\n    return x ** 2  # square\n`, 18 tokens)
reached only 0.99762, below the 0.999 gate. The loss is spread over every
component (keeping any one group in BF16 left the worst case at 0.9981–0.9988),
and group-wise int8 (g32–g128) still left a worst case near 0.9993 on a 96-text
subset. BF16 is the checkpoint's own precision, so the native output matches
the reference to F32 rounding; it costs 136 MB more GPU memory than int8.

## API

`POST /v1/embeddings`, per the pinned OpenAI `CreateEmbeddingRequest`:

- `input`: a string, an array of strings, an array of token ids, or an array
  of token-id arrays; at most 2,048 inputs, no empty strings, at most 300,000
  tokens in total. Token ids are EmbeddingGemma 2 ids and are used as given
  (no `<bos>`/`<eos>` added).
- `model`: `embeddinggemma-2` (listed by `GET /v1/models` when installed).
- `encoding_format`: `float` (default) or `base64` (little-endian F32).
- `dimensions`: 768 (default), 512, 256 or 128, the trained Matryoshka widths;
  the vector is truncated, then renormalized. Other values are refused.
- `user`: accepted and ignored.

Text is embedded exactly as sent. Retrieval quality depends on the model
card's prompts, which the client adds: queries as
`task: search result | query: {text}` (other tasks: `question answering`,
`fact checking`, `code retrieval`, `classification`, `clustering`,
`sentence similarity`), documents as `title: {title | none} | text: {text}`.
The Rust API formats both (`local_engine::embedding::EmbeddingPrompt`).

Rust: `local_engine::embedding::EmbeddingModel::open(dir)`, then `tokenize`,
`embed_texts` or `embed_ids` (full-width, normalized), and
`truncate_normalized` for Matryoshka widths.

### Scheduling beside Bonsai decode

Embeddings run on their own Metal queue, one request at a time. Metal has no
queue priorities, so the encoder submits about 1,024 token-layers per command
buffer (whole layers: four layers of a 256-token pass, one layer of a
1,024-token one; about 5 ms of GPU time) and, while Bonsai generations are
running, waits between submissions until the
engine worker finishes a decode step (`local_engine::activity::EngineActivity`,
a condition variable the worker signals; no polling). A decode step is
therefore delayed by at most one short embedding submission; the wait has a
500 ms limit so embeddings are slowed, never starved. At most 8 embedding
requests are admitted (one running); more get `503` with `Retry-After`. A
client that disconnects cancels its request at the next submission boundary.

## Verification

Reference fixtures were generated on Kaggle (CPU, FP32,
sentence-transformers 6.1.0, transformers 5.19.0, the same pinned model files):
token ids and normalized embeddings for 1,000 mixed texts (SciFact, NFCorpus
and CosQA queries and documents, 60 edge cases: empty and whitespace input,
13 scripts, emoji, code in five languages, special-token text, inputs over 512
tokens), and the three retrieval sets with prompts applied. 97 of the 1,000
rows are committed as `local-engine/tests/fixtures/embeddinggemma2-reference.jsonl`
for the ignored real-model test `native_embeddings_match_the_reference_fixture`.

`target/release/embedding-eval` runs the gates (`parity`, `retrieval`,
`bench`). Results on this machine (M-series, under the shared GPU lock):

| Gate | Threshold | Result |
| --- | --- | --- |
| Token ids vs HF reference, 1,000 mixed texts | identical | **1000/1000 identical** |
| Embedding cosine to the FP32 reference (reference ids) | mean ≥ 0.9995, min ≥ 0.999 | **mean 1.000000, min 0.999999** |
| Packing independence (64 inputs alone vs packed) | cosine ≥ 0.99999 | **64/64 bitwise identical** |
| Same gate, per-row int8 artifact (rejected) | | mean 0.999837, **min 0.997625** |

Retrieval, native BF16 against the Kaggle FP32 run of the same model (same prompts, inputs
truncated at 512 tokens like the reference, exact cosine search, nDCG@10):

| Set | Kaggle FP32 | Native | Δ | Recall@100 (Kaggle / native) |
| --- | ---: | ---: | ---: | --- |
| SciFact (300 q, 5,183 docs) | 85.19 | 85.19 | 0.00 | 97.7 / 97.7 |
| NFCorpus (323 q, 3,633 docs) | 36.90 | 36.94 | +0.04 | 36.6 / 36.6 |
| CosQA (500 q, 20,604 docs) | 47.16 | 47.29 ¹ | +0.13 | 99.2 / 99.2 |

¹ CosQA's corpus holds only 6,267 distinct texts among 20,604 documents, and 426 of the 500
relevant documents have exact duplicates, so its nDCG@10 depends on how tied scores are ordered.
Identical texts embed bitwise identically here, so the native number is the expectation under a
random order of exact ties; breaking ties by corpus order instead gives 34.49. The reference's
duplicates differ by batch-padding noise, which orders them effectively at random. All three
sets are within the literature report's int8 gate (mean within 0.3, every task within 0.5).

Throughput (`embedding-eval bench`, best of three, synthetic token ids, GPU otherwise idle):

| Inputs × tokens | Wall time | Tokens/s | Inputs/s |
| --- | ---: | ---: | ---: |
| 64 × 128 | 681 ms | 12,038 | 94.0 |
| 16 × 512 | 1,215 ms | 6,743 | 13.2 |
| 1 × 2,048 | 586 ms | 3,493 | 1.7 |
| 1 × 16 | 9.3 ms | 1,717 | 107 |

Corpus embedding during the retrieval runs: 7,700–8,200 tokens/s on SciFact and NFCorpus
(long documents) and 11,000–12,000 on CosQA (short code). Attention is the cost at length: it
walks keys one at a time per head, quadratic in the input, which is why 2,048-token inputs run at
a third of the short-input rate.

Effect on concurrent Bonsai decode (server, `--no-thinking`, greedy, 300-token answers, a
different prompt per request so suffix lookup cannot copy an earlier answer, conditions
alternated over eight prompts; the load is a loop of 64-document SciFact embedding requests):

| | Bonsai decode | Embeddings |
| --- | ---: | ---: |
| Alone | 36.06 tok/s (34.1–37.4) | 7,604 tok/s through the server |
| Together | 32.68 tok/s (31.0–33.4), **−9.4%** | 1,790 tok/s |

Embedding yields most of the GPU while a generation runs and takes it back when decoding stops.
The machine was shared during this run (load average 10–18 from other work), so treat the
absolute rates as indicative; the alternation shares that load between both conditions.

## Images and video

Image and video embeddings run natively in the same 768-d space, through the
Rust API only (see [Why no HTTP surface](#why-no-http-surface)). They need the
optional **vision artifact** next to the text install:

```bash
python3 tools/convert_embeddinggemma2_vision.py --source cache/embeddinggemma-2
```

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `models/embeddinggemma2-text/embeddinggemma2-vision-bf16.bin` | 335,642,624 | `c3f928d2a65aad2173f7ed6d8acac6e58d0a7336c9a03f0eb022f4a186909968` |

It also writes `manifest-vision.json`. The converter is standard library
only, checks `model.safetensors` and `config.json` against the same pins as
the text converter and its own output against the pinned SHA-256. The engine
verifies the payload hash when it loads the encoder (lazily, on the first
media input; 335 MB more GPU memory). Without the file, media inputs are
refused and nothing else changes.

### What runs

- **Preprocessing** (`local_engine::embedding::vision::preprocess`), matching
  transformers 5.19.0 `Gemma4ImageProcessor` / `EmbeddingGemma2VideoProcessor`
  (torchvision backend) exactly:
  - decode to 8-bit RGB: JPEG with libjpeg-turbo's decoder (the `mozjpeg`
    crate, the decoder Pillow and torchvision use), PNG, GIF, BMP and WebP
    with the `image` crate; alpha dropped, grey replicated, **no EXIF
    rotation** (the reference applies none either);
  - aspect-preserving resize to the largest size with both sides multiples
    of 48 (patch 16 × pooling 3) holding at most 9 × budget patches (budget
    280 soft tokens per image, 140 per video frame; 70/560/1120 accepted),
    small images scaled up; torch's antialiased bicubic `uint8` kernel
    reproduced exactly (Keys a = −0.5, int16 fixed-point weights, horizontal
    then vertical pass, rounded and clamped to `uint8` after each);
  - rescale `f32(u8) × f32(1/255)`; 16×16×3 patches (row, column, channel) in
    raster order with `(x, y)` positions. The number of soft tokens is
    `(h/48)·(w/48)` (≤ 280; e.g. 260–272 for photos).
- **Video:** 1 frame per second of stream time (`floor(i · fps)` for
  `i < floor(duration)`), then at most 32 frames chosen uniformly
  (`numpy.linspace`), 140-token budget per frame. Each frame becomes
  `<|image>` + soft tokens + `<image|>` in the sequence.
  - Frame sequences: `prepare_video(frames, timing)` takes decoded RGB frames
    (all of them, with the stream's frame count, rate and duration; without a
    rate every frame is kept, then the 32-frame cap applies, as the
    reference does for decoded arrays).
  - Files: `vision::video::decode_video_file(path)`. There is no maintained
    pure-Rust H.264/HEVC/VP9 decoder, so this **requires `ffmpeg` and
    `ffprobe` on `PATH`** (an external dependency of this one function). It
    mirrors torchcodec's exact mode: a packet scan for the frame count,
    duration and rate; for 8-bit 4:2:0 streams the raw YUV planes come from
    `ffmpeg` and are converted to RGB in Rust with swscale's fixed-point
    `yuv2rgb` arithmetic and the stream's colour matrix (what the reference's
    `sws_scale` computed); other pixel formats use `ffmpeg`'s RGB conversion.
- **Encoder** (`shaders/embedding_vision.metal`, `local-metal/src/embedding_vision.rs`):
  the 16-layer `gemma4_vision` tower — input projection of `2·(pixel − 0.5)`,
  learned x/y position table, per layer RMSNorm → q/k/v (q/k RMSNorm with
  scale, v without) → axial RoPE θ 100 (the first 32 dims of each 64-dim head
  rotate by x, the last 32 by y) → bidirectional attention within each image
  or frame (scale 1.0) → o → post-norm residual, then the gated GELU-tanh MLP
  3072 with pre/post norms — 3×3 average pooling by patch position in F32,
  × √768, `embed_vision` (RMSNorm without scale, 768→512). BF16 GEMM, RMSNorm,
  residual, RoPE and gated GELU reuse the text kernels; new kernels add the
  position table, the pooling and a head-dim-64 simdgroup-matrix flash
  attention. BF16 weights (the checkpoint's values), F32 activations.
- **Sequence:** `<bos>`, then the parts in order (text tokens; per image or
  frame `<|image>`, one slot per soft token, `<image|>`), `<eos>`; the soft
  tokens replace the placeholder embeddings (they are not scaled by √512, and
  the per-layer embeddings are computed from them, as in the reference); then
  the shared text encoder, mean pooling over every token, 512→768, L2. Up to
  8,192 tokens (a 32-frame video at 130–140 tokens per frame is about 4,500).

### Rust API

```rust
use local_engine::embedding::{EmbeddingModel, MediaPart, prepare_image, prepare_video};
use local_engine::embedding::vision::video::decode_video_file;

let mut model = EmbeddingModel::open(&dir)?;          // dir = models/embeddinggemma2-text
assert!(model.supports_vision());
let image = prepare_image(&std::fs::read("photo.jpg")?)?;
let a = model.embed_image(&image, &mut || Ok(()))?;   // 768-d, normalized
let frames = decode_video_file(Path::new("clip.mp4"))?; // needs ffmpeg/ffprobe
let b = model.embed_video(&frames, &mut || Ok(()))?;
// Interleaved text and media in one embedding (no prompt for media, per the card):
let c = model.embed_multimodal(&[vec![MediaPart::Text("Trail shoes. "), MediaPart::Image(&image)]], &mut || Ok(()))?;
```

`truncate_normalized` gives the Matryoshka widths as for text (the research
run measured 512-d as near-lossless for images and video and 128-d as a
4–8 point loss). The `pace` hook runs after every vision layer (one command
buffer per layer: about 15 ms of GPU time for a 2,520-patch image) and every
text-encoder submission, so a caller can yield to Bonsai decoding or cancel.

### Why no HTTP surface

OpenAI's `POST /v1/embeddings` (`CreateEmbeddingRequest`, current reference)
accepts only text and token arrays, and no other OpenAI endpoint returns
embeddings of images or video. Adding media fields there would be a
non-standard extension clients cannot discover, so the server does not: image
and video embeddings are native-only until a spec mechanism exists. The
placeholder token ids stay refused on `/v1/embeddings`.

### Verification

Reference fixtures were generated on a Kaggle CPU kernel
(`eg2-vis-fixtures`; transformers 5.19.0, sentence-transformers 6.1.0,
torchvision 0.26, torchcodec 0.11, Pillow 12.3, FP32, the pinned model
files), feeding **file paths** to sentence-transformers so the reference
covers decode → embedding: 67 images (56 Flickr30k test photos and 11
synthetic edge cases: 7×5 PNG scaled up, 1600×90 and 70×1300 extremes,
greyscale, RGBA, palette, 3000×2000, EXIF-rotated, already-on-grid 768×768,
progressive, 4:4:4) and 6 videos (5 MSR-VTT test clips, H.264, 12–22 s, and a
40 s MPEG-4 Part 2 clip that triggers the 32-frame cap). For each: the
decoded pixels the processor received, its `pixel_values` and positions, the
token ids and the normalized embedding. `target/release/embedding-vision-eval`
runs the gates; the committed fixture
`local-engine/tests/fixtures/vision/` (two synthetic images, their
pixel-value digests and embeddings) backs a unit test (preprocessing, always
run) and an ignored real-model test.

Results on this machine (M4 Pro, 20-core GPU, under the shared GPU lock):

| Gate | Threshold | Result |
| --- | --- | --- |
| Resize + patches from the reference's decoded pixels, 67 images | identical | **67/67 identical** |
| Decoder (file → RGB) vs the pixels the reference received | identical | **67/67 identical** (JPEG via libjpeg-turbo, PNG via `image`) |
| Whole pipeline file → `pixel_values`, 67 images | identical | **67/67 identical**; token layouts 67/67 identical |
| Video frame indices (1 fps, 32-frame cap), 6 videos | identical | **6/6 identical** (stream facts identical to torchcodec's) |
| Video resize + patches from the reference's frames | identical | **6/6 identical** |
| Video file → RGB frames (`ffmpeg` + Rust `yuv2rgb`) | identical | **5/6 identical** (H.264); the MPEG-4 Part 2 clip: 4 values off by ≤ 2 over 32 frames |
| Image embedding cosine to FP32 reference, 67 images | mean ≥ 0.9995, min ≥ 0.999 | **mean 1.000000, min 0.999999** (reference patches and native pipeline alike) |
| Video embedding cosine, 6 videos (12–32 frames, 1,586–4,226 tokens) | same | **mean 1.000000, min 0.999999** (reference frames and native `ffmpeg` pipeline alike) |

Flickr30k 1K (Karpathy test: 1,000 images, 5,000 captions; captions with
`task: search result | query: ` for text→image and `title: none | text: `
for image→text, as in the research run; exact cosine search):

| Direction | Kaggle FP32 R@1/5/10 (mean) | Native BF16 R@1/5/10 (mean) | Δ mean R |
| --- | --- | --- | ---: |
| text → image | 86.4 / 97.5 / 98.7 (94.2) | 86.4 / 97.5 / 98.7 (94.19) | 0.0 |
| image → text | 94.2 / 99.7 / 100.0 (98.0) | 94.2 / 99.7 / 100.0 (97.97) | 0.0 |

Throughput (GPU otherwise idle):

| Workload | Result |
| --- | --- |
| Images (500×461 photo, 272 soft tokens), 16 per call, best of 3 | 4.19 s: **3.82 images/s** (262 ms each: vision tower ≈ 0.76 TFLOP per image, plus the decoder) |
| Flickr30k corpus, 8 images per call, file decode included | 3.69 images/s |
| Decode + preprocess on the CPU (same photo) | 7.2 ms per image |
| Video, 32 frames × 130 tokens (4,226-token sequence) | 5.0 s (6.4 frames/s); 12 frames: 1.5 s |

### Precision

BF16 weights with F32 activations, like the text path. The research run found
int8 per-row vision weights retrieval-lossless (+0.0 mean R), but the text
path's int8 artifact already failed the per-input parity gate (worst cosine
0.9976) and every image passes through that same decoder, so int8 was not
pursued for the vision tower; ternary vision weights cost 6–9 mean R (§4b of
the research results) and are not used.

### Limits

- EXIF orientation is not applied (as in the reference); rotate images first
  if their pixels are stored sideways.
- Video file decoding needs `ffmpeg`/`ffprobe`; codecs that are not
  bit-exact by specification (e.g. MPEG-4 Part 2) can differ from the
  reference by a level or two in a few pixels; non-4:2:0 or high-bit-depth
  streams use `ffmpeg`'s RGB conversion (a few levels off the reference).
- CMYK JPEGs fall back to the `image` crate's conversion (Pillow's differs).
- One media request at a time per model (the model is not internally
  synchronized), no HTTP route, no batching across callers.

## Multimodal plan

Images and video are native (above). Every modality feeds soft tokens into
the same text encoder (the placeholder ids 258880, 258881 and 258884 stay
refused as token-id input), so adding one means its encoder and its
preprocessing, with no change to the text path:

1. **Audio** (12-layer Conformer, width 1024, 128-bin log-mel at 16 kHz):
   needs a mel front end, chunked local attention with relative positions,
   a depthwise causal convolution and clipped linears; the Kaggle AudioCaps
   baseline itself looked low (t2a R@1 8%) and should be understood first.
