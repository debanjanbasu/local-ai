# Native speech recognition and translation: Whisper large-v3

The server can transcribe and translate speech natively with OpenAI's
**Whisper large-v3** (MIT, 1.55 B parameters), on Metal, beside the Bonsai
checkpoint. The artifact below is required: `serve` does not start without
it, and it serves `POST /v1/audio/transcriptions` and
`POST /v1/audio/translations` and lists `whisper-large-v3` in
`GET /v1/models`.

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
  WAV, AAC/ALAC in MP4/M4A and Vorbis in Ogg/WebM at 4 to 384 kHz (other
  declared rates are a `400`: the resampler's kernel grows with the rate);
  channels are averaged,
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
  reference does. Beam search and `clip_timestamps` are not ported.
- **Word timestamps** (`local_engine::transcription::timing`, from
  `timing.py`): a forced decode of each window's text captures the
  cross-attention queries of the ten alignment heads (layer/head pairs from
  `generation_config.json`); their scores over the window's frames are
  softmaxed, normalized per frame, median-filtered (width 7), averaged and
  aligned by DTW (F32 costs, the reference's tie order); words split on
  spaces (Unicode boundaries for zh/ja/th/lo/my/yue), punctuation merged,
  and the reference's duration heuristics applied.

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
| `timestamp_granularities[]` | `segment` and `word` (with `verbose_json`; `words` holds `word`, `start`, `end`) | not in the schema |
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
`Retry-After`. An admitted request's form must arrive whole within
`--stall-timeout` (default 30 s), or it is refused with `408` and its slot
freed, so slow uploads cannot hold the four slots. A client that disconnects
cancels its request at the next submission boundary (streams: at the next
window).

## Verification

Reference: the original **openai-whisper 20250625**, FP32 on a Kaggle CPU
(torch 2.11.0), with the same weights (bit-identical, above). The jobs
(`stt-whisper-ref-s0` … `s4`) decoded the audio exactly as `whisper.load_audio`
does and saved it, so the native side runs the very same files.
`target/release/transcription-eval` runs the comparisons; WER uses the
package's own `EnglishTextNormalizer`. Results on this machine (M4 Pro, under
the shared GPU lock):

| Gate | Threshold | Result |
| --- | --- | --- |
| Log-mel window vs reference (10.4 s clip) | ≈ F32 rounding | max abs 1.0e-5, mean 3.0e-8 |
| Encoder output vs reference (1,500 × 1,280) | ≈ F32 rounding | max abs 8.3e-3 (values up to 19.9), mean 6.6e-6 |
| Language probabilities (5 LibriSpeech/silence windows) | same top language | identical to 4–6 digits |
| Greedy decode with timestamps, 5 parity windows (prompted and silent included) | token-identical | **5/5 identical** |
| Teacher-forced logits along the reference tokens, 121 steps | same argmax | **121/121 argmax, 121/121 top-10 sets; max abs logit difference 1.1e-4** |
| LibriSpeech test-clean subset (202 utterances, every 13th id, 4,056 words), `transcribe()` at temperature 0, language en | WER within 0.3 of the reference | **reference 7.372 %, native 7.372 % (Δ 0.000); 202/202 token-identical** ¹ |
| Long-form (two 75 s concatenations of one speaker, with and without a prompt) | same segments | **4/4 token-identical, same seeks** (WER 0.59 % / 0.54 %) |
| FLEURS fr/de/es/ja (3 test clips each; float WAVs), greedy translate windows | token-identical | **12/12 identical; 330/330 teacher-forced argmax, max abs logit difference 7.3e-5** |
| Same 12 clips, `transcribe()` translate and transcribe, language detected | sane English, same as reference | **24/24 token-identical, 12/12 languages detected correctly; translation BLEU vs the FLEURS English references 30.18 (reference 30.18)** |
| Word timestamps (`word_timestamps=True`): 21 LibriSpeech clips, the 75 s long-form file, 4 FLEURS clips (ja split on Unicode boundaries), language given or detected | same words and times | **26/26 token-identical, 653/653 words with identical start and end times** |

¹ One utterance (5105-28241-0001) falls into a "the king of the king…" loop
at temperature 0 in both implementations identically (223 of the 299
errors); excluding it the WER is 1.90 % for both. The API's default
temperature 0 uses the fallback ladder, which re-decodes such windows.

Speed (`transcription-eval bench`, GPU otherwise idle):

| Measure | Result |
| --- | --- |
| Encoder, one 30 s window | 1.07 s (F16 GEMMs at 5.4–5.7 TFLOPS) |
| Decoder | 14.3 ms per token (≈ 70 tokens/s) |
| Long-form 75 s file, `transcribe()` | 7.1 s: **real-time factor 0.095** |
| LibriSpeech subset (202 short files, 1,518 s) | 307 s: RTF 0.20 (each file pays a whole 30 s window) |
| Reference on the Kaggle CPU (4 cores, FP32) | RTF 4.2 |

Effect on concurrent Bonsai decode (server, `--no-thinking`, greedy,
300-token answers, 24 different prompts used once each in A-B-B-A order so
suffix lookup cannot copy an earlier answer; the load is a loop of 75 s
`verbose_json` transcriptions started 1 s before each answer; the machine was
shared, load average about 5):

| | Bonsai decode | Transcription |
| --- | ---: | ---: |
| Alone | 34.35 tok/s (31.1–38.2) | RTF 0.096 |
| Together | 33.79 tok/s (31.0–38.8), **−1.6 %** (median −3.2 %) | RTF 0.21 ¹ |

¹ Includes the tail of the last transcription after the answer finished.

Through the server (`api_smoke.py` in the research notes) every response
format, the `whisper-1` alias, streaming with logprobs, word and segment
timestamps, M4A (AAC) and 48 kHz stereo WAV input, French transcription with
detection and French→English translation return `200`, and unsupported
fields, models, formats, garbage audio and a 25 MiB+ upload return `400`/`413`.
A request whose client disconnects stops within about 2 s (the next request
then waited 4.7 s instead of the ≈ 10 s an uncancelled 75 s transcription
would have cost).

## Limitations

- Word timestamps follow the reference's `word_timestamps=True`, which also
  moves each window's seek to the last word's end, so a request with words
  can segment long audio differently from one without (as in the
  reference). The hallucination-silence heuristics are not ported.
- **Opus** (Ogg/WebM voice notes) is refused: there is no maintained pure-Rust
  decoder. MP4/M4A edit lists are not read, so AAC encoder priming (typically
  2,112 samples, 0.13 s) stays at the start of the audio and shifts
  timestamps by that much (FFmpeg trims it); 16 kHz FLAC/WAV/MP3 input is
  bit-exact with the reference.
- Greedy decoding and temperature sampling only (no beam search, `best_of` 1:
  the Python API's defaults). Sampling uses this server's own random numbers,
  so sampled output is not reproducible against the reference.
- Streaming deltas arrive per finished 30-second window (whole segments).
- Whisper pads every window to 30 s, so a short clip costs a whole encoder
  pass (about 1.1 s on this machine).
- One request runs at a time; under Bonsai load transcription runs at low
  priority (see above).
- The audio-embedding work also decodes with Symphonia 0.6 and computes a
  128-bin log-mel; the two front ends use different parameters (Whisper's
  STFT here, Gemma's feature extractor there) and are kept in separate files,
  but the decoding and resampling could be shared.

