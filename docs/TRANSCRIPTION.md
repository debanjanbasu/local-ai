# Native speech recognition and translation: Whisper large-v3

The server can transcribe and translate speech natively with OpenAI's
**Whisper large-v3** (MIT, 1.55 B parameters), on Metal, beside the Bonsai
checkpoint. It is optional and opt-in by installation: install the artifact
below and `POST /v1/audio/transcriptions` and `POST /v1/audio/translations`
appear and `GET /v1/models` lists `whisper-large-v3`; without it nothing
changes and both routes are `404`.

## Why Whisper large-v3

The two endpoints need one model that transcribes many languages *and*
translates them into English. Candidates, with published numbers (Open ASR
Leaderboard, arXiv 2510.06961 v3, average English WER over its eight sets;
multilingual = FLEURS + CoVoST-2 + MLS average over de/fr/it/es/pt):

| Model | Licence | Params | English avg WER | LS clean / other | Multilingual avg | Speech translation | Port |
| --- | --- | ---: | ---: | --- | ---: | --- | --- |
| **Whisper large-v3** | MIT | 1.55 B | 7.44 | 2.0 / 3.9 ¹ | 4.97–6.59 per language | **99 → English** | transformer enc-dec, GEMMs + attention |
| Whisper large-v3-turbo | MIT | 0.81 B | 7.83 | ~2.1 / 4.2 | slightly worse | **not trained for translation** (model card) | same, 4 decoder layers |
| Distil-Whisper large-v3.5 | MIT | 0.76 B | 7.21 | | English only | none | same |
| Canary-1B-v2 | CC-BY-4.0 | 1 B | 7.15 | | 4.60 | 24 European ↔ English only | FastConformer (rel-pos attention, depthwise conv), NeMo-only reference |
| Parakeet-TDT-0.6B-v3 | CC-BY-4.0 | 0.6 B | 6.32 | | 25 European | none | FastConformer + TDT transducer |
| Moonshine | MIT | 0.06–0.2 B | 6.66 (base) | | English only | none | small enc-dec |

¹ Whisper paper / model card figures for large-v3 on LibriSpeech; the
leaderboard computes its own.

Whisper large-v3 is the only candidate that covers both endpoints for every
language with one permissive model: turbo drops translation, the NVIDIA models
cover 24–25 European languages and need a FastConformer port, the small models
are English-only. Its architecture also maps directly onto kernels this
repository already has (an F16 version of the EmbeddingGemma 2 GEMM; LayerNorm,
attention), so the port is the same kind of work as the embedding model. It is
also the model family behind OpenAI's own `whisper-1`.

## Install

The checkpoint is `openai/whisper-large-v3` (transformers layout) at revision
`06f233fe06e710322aca913c1bc4249a0d71fce1`, plus the original package's
128-band mel filterbank (`whisper/assets/mel_filters.npz`, openai-whisper
20250625). Whisper large-v3 is not on Kaggle Models and this machine does
not download from Hugging Face, so `tools/kaggle_whisper_job.py` stages a
private Kaggle **CPU** kernel that downloads these public MIT files at the
pinned revision (and the pinned openai-whisper sdist from PyPI for the
filterbank), checks every size and SHA-256 against the converter's pins, and
saves them as its output; the conversion is then offline (standard library
only, sources size- and SHA-256-checked, output checked against a pinned
SHA-256):

```bash
python3 tools/kaggle_whisper_job.py --kernel <owner>/whisper-large-v3-fetch --output cache/whisper-job
kaggle kernels push -p cache/whisper-job
kaggle kernels output <owner>/whisper-large-v3-fetch -p cache/whisper-large-v3
python3 tools/convert_whisper.py --source cache/whisper-large-v3
```

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `models/whisper-large-v3/whisper-large-v3-f16.bin` | 3,093,458,432 | `cce7ae83d4a15146f356a9f3bb5169b0fb42a86dc2e6a0c6cf932766e6d1cb53` |
| `models/whisper-large-v3/tokenizer.json` | 2,480,617 | `6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca` |

Source pins: `model.safetensors` 3,087,130,976 B `a8e94b85…40fd95`,
`config.json` 1,272 B `ad0e8d1e…9807b`, `generation_config.json` 3,903 B
`fbdfa701…09d23`, `mel_filters.npz` 4,271 B `7450ae70…f6d4c`. The reference
kernel compared the transformers weights with the original `large-v3.pt` that
openai-whisper downloads (its URL pins SHA-256 `e5b1a55b…1dadb`): **all 1,259
tensors are bit-identical**, so the artifact holds exactly the weights the
original implementation runs. The server verifies the artifact's payload
SHA-256 on every load.

## What runs

- **Front end** (`local_engine::transcription::mel`, CPU): the reference's
  `log_mel_spectrogram` — centered STFT (periodic Hann 400, hop 160, reflect
  padding), power of bins 0–200 with the last frame dropped, the 128-band
  filterbank, `log10` clamped at 1e-10, floor at max − 8, `(x + 4) / 4` —
  over the audio plus 30 s of zeros, as `transcribe()` does.
- **Audio** (`local_engine::transcription::audio`): Symphonia 0.6 decodes FLAC, MP3,
  WAV, AAC/ALAC in MP4/M4A and Vorbis in Ogg/WebM; channels are averaged,
  other rates are resampled to 16 kHz with a Kaiser-windowed sinc (16 zero
  crossings, β 9, pass band 0.97 of Nyquist, like FFmpeg's default), and
  samples are rounded to 16-bit steps because the reference reads FFmpeg's
  `s16le` output. 16 kHz input reaches the model exactly as it reaches the
  reference. **Opus is refused** (no maintained pure-Rust decoder), so Ogg or
  WebM voice notes encoded with Opus get a `400`.
- **Encoder** (Metal): two kernel-3 convolutions as im2col + GEMM with exact
  (erf) GELU, the sinusoidal table, 32 pre-LayerNorm blocks (20 heads × 64,
  MLP 5,120), final LayerNorm; then each of the 32 decoder layers'
  cross-attention keys and values are projected once per window.
- **Decoder** (Metal): token + learned position embeddings (CPU, from the
  mapped F16 table), 32 blocks with a self-attention KV cache of 448
  positions and cross-attention over the 1,500 encoder frames, final
  LayerNorm, logits against the tied token table. Prefill uses the GEMM; each
  new token a GEMV.
- **Precision:** every matrix keeps the checkpoint's F16 values (3.1 GB);
  biases, LayerNorm parameters and the encoder position table are F32.
  Activations, accumulation, LayerNorm and softmax are F32, like the FP32
  reference run.
- **Kernels** (`shaders/whisper.metal`): F16×F32 GEMM on 8×8 simdgroup
  matrices with a bias/GELU/residual epilogue, F16 GEMV, LayerNorm, im2col,
  a thread-per-query encoder attention over shared K/V tiles, and a
  256-thread decoder attention.
- **Decoding** (`local_engine::transcription::decode`, CPU, ported from
  openai-whisper 20250625 `decoding.py`/`transcribe.py`): the reference's
  logit filters (blank and non-speech suppression, timestamp rules with the
  1-second initial limit), greedy decoding at temperature 0 and sampling
  above it, the temperature fallback (0, 0.2 … 1.0 while the compression
  ratio exceeds 2.4 or the average log-probability is below −1, unless the
  window is likely silent), the no-speech skip, and the long-form loop that
  seeks to the last complete timestamp pair, conditioning each window on the
  previous text. The compression ratio uses zlib itself (level 6), as the
  reference does. Beam search, word timestamps and `clip_timestamps` are not
  ported.

## API

`POST /v1/audio/transcriptions` and `POST /v1/audio/translations`
(multipart), per the pinned `CreateTranscriptionRequest` and
`CreateTranslationRequest`:

| Field | Transcriptions | Translations |
| --- | --- | --- |
| `file` | flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm (not Opus); ≤ 25 MiB, ≤ 1 hour | same |
| `model` | `whisper-large-v3` (or `whisper-1`, mapped to it) | same |
| `language` | ISO-639-1 code (or English name) of Whisper's 100 languages; detected from the first 30 s when absent | not in the schema (refused) |
| `prompt` | prepended as previous text (`<\|startofprev\|>`), last 223 tokens kept | same |
| `temperature` | 0–1; 0 (default) = greedy with the fallback ladder | same |
| `response_format` | `json` (text + duration usage), `text`, `srt`, `vtt`, `verbose_json` (task, language name, duration, segments) | same (`language` is `english`) |
| `timestamp_granularities[]` | `segment` (with `verbose_json`); **`word` refused** | not in the schema |
| `include[]` | `logprobs` (with `json` or `stream`): per text token, `token`, `logprob`, `bytes` | not in the schema |
| `stream` | `true`: `transcript.text.delta` per finished 30-second window (whole segments), then `transcript.text.done` with token usage; `json`/`text` only | not in the schema |
| `chunking_strategy`, `known_speaker_*`, `languages`, `keywords`, `diarized_json` | refused (`400`) | — |

Captions follow the reference writers (`srt` always shows hours, `vtt`
only past an hour; empty segments are skipped; `-->` in text becomes `->`).
Streaming deltas are whole segments, not tokens: Whisper rewrites the last,
unfinished segment of every window from the next window, so tokens are
final only once their window is.

Rust: `local_engine::transcription::TranscriptionModel::open(dir)`, then `transcribe(audio,
options, pace, on_segments)` with 16 kHz samples from
`local_engine::transcription::audio::decode`.

### Scheduling beside Bonsai decode

Speech runs on its own Metal queue, one request at a time. Work is split into
command buffers by estimated GPU time: while Bonsai generations run
(`EngineActivity::busy`) about 2 ms per submission (large GEMMs and encoder
attention are split by rows), each followed by
`EngineActivity::yield_to_decode` (condition variable, 500 ms limit), as
`docs/EMBEDDINGS.md` describes; otherwise about 50 ms, pipelined. At most 4
speech requests are admitted (one running); more get `503` with
`Retry-After`. A client that disconnects cancels its request at the next
submission boundary (streams: at the next window).

## Verification

RESULTS
