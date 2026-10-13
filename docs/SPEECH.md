# Native text to speech: Kokoro-82M

The server can synthesize speech natively with hexgrad's **Kokoro-82M**
(v1.0, Apache-2.0, 82 M parameters), on Metal, beside the Bonsai checkpoint.
It is optional and opt-in by installation: install the artifact below and
`POST /v1/audio/speech` appears (and `kokoro-82m` is listed by
`GET /v1/models`); without it nothing changes and the route is `404`.

## Why Kokoro-82M

Requirements: permissive weights *and* voices, no GPL in the binary, quality
competitive with much larger open models, real-time on a shared GPU without
slowing Bonsai much, and a port feasible with this repository's kernels.

| Candidate | Licence | Size | Evidence on quality | Blocking issue here |
| --- | --- | --- | --- | --- |
| **Kokoro-82M v1.0** | Apache-2.0 (weights, 54 voices) | 82 M | #1 on the TTS Spaces Arena at release (Jan 2025, ahead of XTTS v2 467 M, MetaVoice 1.2 B, Fish Speech); UTMOS 4.44–4.45 in a 2026 CPU benchmark; best-in-class UTMOS among the systems compared in the T5Gemma-TTS report | needs a G2P (its pipeline falls back to GPL espeak-ng for unknown words) |
| Piper / VITS voices | MIT code, per-voice licences | 15–60 M | lower naturalness than Kokoro in public comparisons | phonemizer is espeak-ng (GPL); voices' licences vary |
| StyleTTS 2 | MIT code; LibriTTS weights carry a disclosure condition | ~150 M | Kokoro is a distilled StyleTTS 2 / iSTFTNet that ranks above it | weights conditions; same G2P problem |
| Orpheus 3B | Apache-2.0 code, Llama 3.2 licence weights | 3 B | best WER (9%) in one 2026 comparison | Llama licence; a 3 B LLM plus SNAC decoding competes with Bonsai for the GPU |
| Chatterbox | MIT | 0.5 B LLM + flow decoder | expressive, voice cloning | 6x the size; a flow-matching decoder plus an LLM per request |
| Dia 1.6B | Apache-2.0 | 1.6 B | dialogue; 16% WER in the same comparison | size; English only; slower than real time on modest GPUs |
| Sesame CSM-1B | Apache-2.0 | 1 B + Mimi codec | conversational | needs the gated Llama tokenizer; size |
| Parler-TTS / Qwen3-TTS (instruction-following) | Apache-2.0 | 0.6–2.3 B | honor style descriptions | 7–28x the size; autoregressive codec decoding |

Kokoro is the only candidate that is small enough to run far faster than real
time beside Bonsai, permissive for weights and voices, near the top of public
preference rankings, and made of operations this repository already runs
(GEMMs, norms, attention) plus a few new kernels (LSTM recurrence, 1-D
convolutions, iSTFT). Its two costs are documented below: no style
instructions, and a G2P to port.

**What Kokoro cannot do.** It has no instruction or description input: its
only controls are the voice (a 256-value style vector per phoneme count) and
speed. The API's `instructions` field is therefore refused when non-empty
rather than silently ignored. Of its eight languages only American and
British English have a non-GPL G2P (`misaki`'s English lexicons); Spanish,
French, Hindi, Italian and Portuguese depend on espeak-ng, and Japanese and
Chinese on other packages, so this build serves English only.

## Install

The model never comes from Hugging Face on this machine. A private Kaggle CPU
job downloads it there and records the upstream metadata:

```bash
python3 tools/kaggle_kokoro_job.py --kernel OWNER/kokoro-ref-tts --output cache/kokoro-job
kaggle kernels push -p cache/kokoro-job            # private CPU kernel
kaggle kernels output OWNER/kokoro-ref-tts -p cache/kokoro
tar -xf cache/kokoro/kokoro-ref.tar -C cache/kokoro   # or the output directory
python3 tools/convert_kokoro.py --source cache/kokoro/kokoro-ref
```

The job (CPU, Python 3.12 virtual environment, `kokoro` 0.9.4, `misaki`
0.9.4, torch 2.8.0+cpu) downloads `hexgrad/Kokoro-82M` at revision
`f3ff3571791e39611d31c381e3a41a3af07b4987`, checks every file against the
repository's LFS SHA-256, and also writes the parity fixtures described
below. The converter is standard library only (a restricted unpickler reads
the `.pth`; anything but tensors and dicts is refused), checks all 34 source
files against pinned sizes and SHA-256s (`kokoro-v1_0.pth` 327,212,226 B
`496dba11…8ad1e4`, `config.json`, the 28 English voices, the four English
`misaki` lexicons), and checks its output:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `models/kokoro-82m/kokoro-82m-f32.bin` | 354,146,560 | `aa3b5c769b9fdcc9eac3d6068352610183e45c20542f1232d629ead82c695928` |

The server looks for `kokoro-82m/` in the same `models/` directory as the
Bonsai GGUF and verifies the payload SHA-256 on every load.

## What runs

- **Network** (`kokoro` 0.9.4 `KModel.forward_with_tokens`): PL-BERT (ALBERT,
  12 shared layers, 768 wide, 12 heads, GELU-tanh FFN 2048) → 768→512
  projection; the prosody predictor (three bidirectional LSTM + AdaLayerNorm
  layers, a duration LSTM and a 50-way duration head, the shared LSTM, the F0
  and energy branches of AdaIN residual blocks); the text encoder (embedding,
  three conv-LayerNorm-LeakyReLU blocks, bidirectional LSTM); the decoder
  (AdaIN residual blocks at 1024 channels); the iSTFTNet generator (harmonic
  plus noise source from F0, two transposed-convolution upsamplings ×10 and
  ×6, Snake-activated AdaIN resblocks with kernels 3/7/11 and dilations
  1/3/5, a 22-channel head) and an inverse STFT (n_fft 20, hop 5) to 24 kHz.
- **Precision**: every weight is the checkpoint's F32 (weight normalization
  folded in by the converter); activations and accumulation are F32.
- **Kernels** (`shaders/kokoro.metal`): one gathered GEMM covers every Conv1d,
  linear map and each output phase of a transposed convolution (8×8 simdgroup
  matrices, 32×32 tiles); a bidirectional LSTM recurrence (one threadgroup per
  direction walks time; input projections are a GEMM first); channel
  LayerNorm, instance norm with style affine and LeakyReLU or Snake,
  softmax attention, depthwise transposed upsampling, gathers for the
  duration alignment, GELU and the inverse STFT. Durations, the harmonic
  source (in the reference's F32 operation order) and its 20-point STFT run on
  the CPU between three GPU phases.
- **G2P** (`local-engine/src/speech/g2p.rs`): a port of `misaki` 0.9.4's
  English G2P reading its own gold and silver lexicons (American with `us_*`,
  British with `gb_*`, chosen by the voice's prefix): lexicon lookup with
  capitalization stress, `-s`/`-ed`/`-ing` stemming, number, ordinal, year
  and currency reading, context forms of *the*, *a*, *to*, *in*, and the
  subtoken merging of hyphenated and compound words. spaCy's tagger is
  replaced by a rule tokenizer and context rules for the tags the lexicon
  needs (determiner *a*, demonstrative *that*, heteronyms after modals,
  determiners, pronouns and auxiliaries, proper nouns, *used to*). Words in
  neither lexicon are segmented into known words of three or more letters and
  letter sounds instead of being sent to espeak-ng.
- **Chunking**: like Kokoro's pipeline: newlines split the text, then chunks
  of at most 510 phonemes are cut at the last sentence end, clause or comma
  that fits. Each chunk is synthesized separately and streamed as soon as it
  is ready. A chunk that would exceed 2,000 acoustic frames (50 s at speed 1;
  the bound keeps the generator's buffers under about 600 MB) is split in half
  at a word boundary.
- **Limits**: input up to 4,096 characters (the API's limit); speed 0.25–4.

## API

`POST /v1/audio/speech`, per the pinned `CreateSpeechRequest`
(additional properties refused):

| Field | Implemented |
| --- | --- |
| `model` | `kokoro-82m` (other names are refused, as on `/v1/embeddings`) |
| `input` | 1–4,096 characters, English |
| `voice` | a Kokoro voice id (`af_heart`, `af_bella`, `am_michael`, `bf_emma`, `bm_george`, … the 28 English voices), a pinned built-in name, or `{"id": …}` naming either |
| `instructions` | **refused** when non-empty (Kokoro cannot honor it) |
| `response_format` | `mp3` (default; MPEG-2 Layer III 64 kbit/s, `rusty_mp3`), `opus` (Ogg Opus 32 kbit/s, `rusty-opus` + `ogg`), `flac` (`flacenc`), `wav`, `pcm` (s16le, 24 kHz, mono); `aac` **refused** (no permissively licensed encoder) |
| `speed` | 0.25–4.0, divides the predicted durations (Kokoro's own speed control) |
| `stream_format` | `audio` (default: the encoded stream, chunked per synthesized text chunk) and `sse` (`speech.audio.delta` with base64 bytes of the same encoded stream, then `speech.audio.done` with `usage`: input tokens = phonemes, output tokens = 24 kHz frames of 600 samples) |

The built-in names map to distinct Kokoro voices. **They are Kokoro's voices,
not OpenAI's**; the names are accepted so that existing clients work:

| Name | Kokoro voice | Name | Kokoro voice |
| --- | --- | --- | --- |
| `alloy` | `af_alloy` | `onyx` | `am_onyx` |
| `ash` | `am_michael` | `nova` | `af_nova` |
| `ballad` | `bm_george` | `sage` | `af_sarah` |
| `coral` | `af_heart` | `shimmer` | `af_bella` |
| `echo` | `am_echo` | `verse` | `am_puck` |
| `fable` | `bm_fable` | `marin` | `bf_emma` |
| | | `cedar` | `am_fenrir` |

Encoders are pure Rust under permissive licences. PCM, WAV (open-ended
length in a streamed header), MP3 (whole frames; no Xing header, which cannot
be prepended to a stream) and Opus (whole Ogg pages) stream progressively;
FLAC needs the whole signal for its header and arrives at the end. Checked
with ffmpeg as an independent decoder on a 9.5 s reference utterance: FLAC
and WAV bit-exact to 16-bit PCM (75.5 dB SNR = quantization), MP3 log-mel
difference 1.69 dB (mel cosine 0.99995), Opus 1.88 dB (0.998).

Rust: `local_engine::speech::SpeechModel::open(dir)`, then `synthesize`
(text, voice, speed, noise, a pacing hook, a sink receiving each chunk's
samples), `chunks` and `synthesize_phonemes` for phoneme-level use, and
`speech::encode` for the formats.

### Scheduling beside Bonsai decode

Speech runs on its own Metal queue, one request at a time. Work is submitted
in command buffers of at most 4·10⁹ multiply-accumulates (convolutions are
split by output columns to stay under it; about 3 ms of GPU time), and while
Bonsai generations run, the synthesizer waits between submissions until the
engine worker finishes a decode step (`EngineActivity::yield_to_decode`,
500 ms limit), as embeddings do. At most 4 speech requests are admitted (one
running); more get `503` with `Retry-After`. A client that disconnects
cancels its request at the next submission boundary.

## Verification

Reference: the original `kokoro` 0.9.4 / `misaki` 0.9.4 on a Kaggle CPU kernel
(Python 3.12, torch 2.8.0+cpu, F32; private kernel version 3 of
`tools/kaggle_kokoro_job.py --mode reference`, whose fixture outputs are all
complete; its final ASR step failed for want of `torchaudio` and was rerun in
the ASR job below, which resamples to 16 kHz itself) on 30 fixed fixtures:
English sentences of 5 to 166 phonemes over 26 voices (American and British),
speeds 0.8, 1.0 and 1.25. For each the job recorded misaki's phonemes, the
token ids, the predicted durations, three waveforms (deterministic: source
noise and random phase zeroed; seeded: as users hear it; and the full
`KPipeline` from text, with espeak-ng as misaki's fallback) and, for three of
them, every intermediate activation. The rows' phonemes, ids and durations
are committed as `local-engine/tests/fixtures/kokoro-reference.json`
(ignored real-model tests `native_g2p_matches_misaki_on_the_fixture_texts`
and `native_durations_match_the_reference_fixture`). Log-mel distances use 80
bins, n_fft 1024, hop 256, cells within 80 dB of the reference peak.

### Gates (fixed before the first native run, commit `cda9cff`)

| Gate | Threshold | Result |
| --- | --- | --- |
| Token ids of the reference phonemes | identical on every fixture | **30/30 identical** |
| Predicted durations | identical on ≥ 90% of fixtures | **30/30 identical** |
| Deterministic mode, log-mel distance | MAD ≤ 1.0 dB and mel cosine ≥ 0.99 on every fixture with identical durations | **MAD mean 0.650 dB (max 0.964), cosine mean 0.99874 (min 0.99605)** |
| Seeded mode (different noise streams), log-mel distance | MAD ≤ 2.0 dB, cosine ≥ 0.98 | **MAD mean 0.572 dB (max 0.757), cosine mean 0.99897 (min 0.99765)** |
| Intelligibility | native text-to-speech WER ≤ reference pipeline WER + 3 points | **native 0.30%, reference 0.30%** (1 of 328 words: "three thirty" heard as "3.30" in both); Whisper's transcripts of native and reference audio are identical on all 30 texts, in both the text path and the seeded phoneme path |
| Native G2P vs `misaki` (lexicon only) | reported | **fixture texts 30/30 identical in both accents; Gutenberg (*Pride and Prejudice*, 400 sentences) words 99.46% (US) / 99.45% (GB) identical** |

The ASR job (`tools/kaggle_kokoro_job.py --mode asr`, private Kaggle CPU
kernel, `openai/whisper-large-v3-turbo` with Whisper's English text
normalizer, audio resampled 24→16 kHz with `resample_poly`) transcribed the
native and reference 16-bit WAVs of the text and seeded paths (120 clips,
uploaded as a private dataset).

Full text path (native G2P and chunking, seeded) against the reference
`KPipeline` output: MAD mean 0.576 dB (max 0.758), cosine mean 0.99897.

Intermediate activations (rows 0–2, deterministic mode) match the reference
to F32 rounding all the way to the generator's input: relative RMS error
4–6·10⁻⁶ for PL-BERT and its projection, 1–4·10⁻⁶ for the duration encoder
and LSTMs, 3–9·10⁻⁷ for the duration logits, F0 and energy curves, 1.3–1.6·10⁻⁶
for the decoder blocks. The generator head differs by 1.1·10⁻² and the
deterministic waveforms by 16 dB SNR (median; min 10.4 dB): with the noise
zeroed, the harmonic source is a constant in unvoiced stretches, so the
phases of its near-empty STFT bins are rounding noise (±π) in both
implementations and feed the generator different values there. The log-mel
gates above measure what remains audible.

### Throughput

`speech-eval bench` (best of three, GPU otherwise idle, M4 Pro; the 151-
character sentence of fixture 26, once and as six newline-separated chunks):

| Voice | Audio | Wall time | RTF | × real time |
| --- | ---: | ---: | ---: | ---: |
| `af_heart` | 9.47 s | 0.225 s | 0.0237 | 42 |
| `bm_george` | 10.53 s | 0.242 s | 0.0230 | 43 |
| `af_heart`, 6 chunks | 56.85 s | 1.348 s | 0.0237 | 42 |
| `bm_george`, 6 chunks | 63.15 s | 1.468 s | 0.0232 | 43 |

Through the server (`pcm`, 1,864 characters, 121.8 s of audio): first bytes
after 0.73 s, all after 2.91 s (RTF 0.024). All 90 fixture syntheses (3 modes
× 30 rows) took 9.4 s. Load, including the payload hash: 0.4–1.4 s.

### Beside Bonsai decode

Server, `--no-thinking`, greedy, 300-token answers, a different prompt per
request, conditions alternated over eight prompts; the load is a loop of
`pcm` speech requests of 44 s of audio each (`.amp/in/decode_bench.py`,
under the shared GPU lock):

| | Bonsai decode | Speech |
| --- | ---: | ---: |
| Alone | 37.60 tok/s (33.0–40.2) | 44 s of audio in 1.04–1.07 s (RTF 0.024) |
| Together | 34.28 tok/s (33.1–35.9), **−8.8%** | 44 s of audio in 7.9–8.5 s while decoding (RTF 0.19) |

Speech yields most of the GPU while a generation runs (it still runs five
times faster than real time) and takes it back when decoding stops — the same
trade as embeddings (−9.4%).

End-to-end checks on the same server (`.amp/in/e2e_speech.py`): every
supported format decodes with the expected magic and content type, the
default is `mp3`, `aac`, non-empty `instructions`, unknown voices, other
models, speed outside 0.25–4, unknown fields, empty and over-long inputs are
`400` with a reason; aliases and `{"id": …}` voices work; speed 0.5 vs 2.0
changes the length 3.7×; `sse` streams `speech.audio.delta` events and one
`speech.audio.done` with usage; a long input streams its first chunk after
25% of the total time; dropping a long request lets the next one start in
0.06 s (cancelled at the next submission); seven concurrent long requests get
four `200` and three `503` with `Retry-After`.

### Samples and how to rerun

```bash
target/release/speech-eval synth cache/kokoro/kokoro-ref out   # GPU
target/release/speech-eval compare cache/kokoro/kokoro-ref out # CPU, writes out/wav/
target/release/speech-eval g2p cache/kokoro/kokoro-ref          # CPU
target/release/speech-eval bench --chunks 6                     # GPU
```
