# Native text embeddings: EmbeddingGemma 2

The server can embed text natively with Google's **EmbeddingGemma 2**
(`google/embeddinggemma-2`, 744 M parameters with its image, audio and video
encoders; 271 M on the text path), on Metal, beside the Bonsai checkpoint. It
is optional and opt-in by installation: install the artifact below and
`POST /v1/embeddings` appears; without it nothing changes and the route is
`404`. Only the text path (text and code) is native today; see
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

## Vector stores

When the artifact is installed, the server's vector stores
(`/v1/vector_stores`) embed every chunk as it is indexed and rank
`ranker: auto` searches by the cosine similarity of chunk and query
embeddings; `ranker: none` stays BM25, and without the artifact everything
is BM25 exactly as before. The full contract is the
`local_services::vector_stores` module docs ("Embeddings"); the essentials:

- **Prompts:** chunks as `title: none | text: {chunk}`, queries as
  `task: search result | query: {query}`, 768 dimensions. A chunk longer
  than 2,048 tokens (chunks are counted in words; only large static chunk
  sizes reach it) is embedded from its first 2,047 tokens and `<eos>`.
- **Storage:** one F16 vector per chunk (1.5 KB) in the services database,
  keyed like the chunk, written in the same transaction that completes the
  file and deleted with the garbage outbox entry that removes its index
  documents; each file records the vector space (model, weights digest,
  width, prompts) of its vectors. Stores created before embeddings (schema
  version 1) are migrated in place and **backfilled in the background** by
  the indexer (on start and on every notification, never polling), as are
  files indexed while the model was absent and every file after a model or
  prompt change. While a store is partly embedded, `auto` uses the
  reciprocal rank fusion below so unembedded files are still found by BM25.
- **Priority:** indexing embeds outside the index lock, holds the shared
  model for at most about 4,096 tokens at a time (an API embedding request
  waits for at most one group) and yields to Bonsai decode steps between
  command buffers exactly like `/v1/embeddings`; stopping the server stops
  an embedding batch at its next submission.
- **Search** is brute force over the admitted files' vectors, streamed
  from the database; memory is bounded by the result count, time is linear
  in the store's chunks.

### Choosing the ranking

`target/release/vector-store-eval` measures through the vector store search
path itself: `prepare` stores each fixture corpus as Files in one vector
store with the default chunking (800/400 words; documents as
`title\n\ntext`, exact duplicate texts as one File) and indexes it for BM25
without the model; `measure` loads the encoder, embeds the store through the
backfill, and runs every query (without its prompt; the store adds its own)
through `VectorIndex::search_with`, 50 results reduced to distinct Files,
nDCG@10 against the qrels (a File holding several documents is relevant
with their best grade, so duplicates do not tie). Under the shared GPU lock
on this machine:

| Ranking | SciFact | NFCorpus | CosQA | Mean |
| --- | ---: | ---: | ---: | ---: |
| BM25 (`none`) | 68.78 | 32.31 | 38.76 | 46.62 |
| **Embeddings (`auto`)** | **84.54** | 35.37 | **81.10** | **67.00** |
| RRF k = 60, text 1 : embedding 1 | 77.54 | 37.10 | 57.90 | 57.51 |
| RRF k = 60, 1 : 2 | 80.46 | **37.21** | 62.34 | 60.00 |
| RRF k = 60, 1 : 3 | 81.16 | 37.17 | 66.38 | 61.57 |
| RRF k = 60, 2 : 1 | 75.29 | 36.57 | 52.19 | 54.68 |
| RRF k = 20, 1 : 1 | 78.97 | 37.03 | 61.24 | 59.08 |

SciFact: 300 queries, 5,183 Files (5,192 chunks); NFCorpus: 323 queries,
3,593 Files (3,602 chunks); CosQA: 500 queries, 6,267 Files (6,267 chunks).
CosQA's numbers are not comparable with the retrieval table above: there,
exact duplicates (426 of the 500 relevant documents have them) tie and
count against the ranking; here they are one File. The query prompt is
`search result` for every set (the store cannot know a query is a fact
check or a code search) and titles are part of the document text, so
SciFact lands 0.65 below the run above with the fact-checking prompt and
the title in the prompt.

Embeddings alone win the mean by 5.4 points over the best fusion and by
20.4 over BM25; every fusion loses to them on SciFact (by 3.4 or more) and
CosQA (by 14.7 or more), and the best fusion wins NFCorpus by only 1.8. So
`auto` is embeddings alone, with fusion only as the fallback for partly
embedded stores. Its score is `max(cos, 0)`, a similarity `score_threshold`
can meaningfully cut.

Cost on this machine: embedding the corpora through the backfill took 234 s
for SciFact (5,192 chunks), 184 s for NFCorpus (3,602) and 87 s for CosQA
(6,267), 20–72 chunks/s on a GPU otherwise idle for this run; a 50-result
search took 30–39 ms with BM25 and 30–36 ms with embeddings (query embedding
included), 36–55 ms fused.

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

## Multimodal plan

The image, audio and video encoders are not native yet. Each feeds soft tokens
into the same text encoder implemented here (placeholder ids 258880, 258881,
258884 are refused today), so adding a modality means its encoder and its
preprocessing, with no change to the text path:

1. **Vision** (16 layers, width 768, 12 × 64 heads, MLP 3072, 2-D learned
   positions and axial RoPE θ 100, 3×3 pooling to 280 tokens per image or 140
   per video frame, `embed_vision` RMSNorm → 768→512): BF16 GEMM and RMSNorm
   reuse these kernels; new kernels for patchify, the position-table gather,
   axial RoPE and the pooling. Preprocessing: rescale to [0, 1] with no
   mean/std, 16-px patches, pooling in F32. Gate: Flickr30k 1K within 0.5 mean
   R@1/5/10 of the Kaggle FP32 run (t2i 86.4 / 97.5 / 98.7).
2. **Video:** the vision encoder at 1 frame/s, at most 32 frames.
3. **Audio** (12-layer Conformer, width 1024, 128-bin log-mel at 16 kHz):
   needs a mel front end, chunked local attention with relative positions,
   a depthwise causal convolution and clipped linears; the Kaggle AudioCaps
   baseline itself looked low (t2a R@1 8%) and should be understood first.
