# Native text to speech: Kokoro-82M

<!-- Gates below were fixed before the first native run; results are filled in after. -->

## Gates (stated before the first native output)

Reference: the original `kokoro` 0.9.4 / `misaki` 0.9.4 on a Kaggle CPU
kernel (torch 2.8.0+cpu, F32), the same pinned model files.

| Gate | Threshold |
| --- | --- |
| Token ids of the reference phonemes | identical on every fixture |
| Predicted durations | identical on at least 90% of fixtures |
| Deterministic mode (source noise and random phase zeroed on both sides), log-mel distance | mean absolute difference at most 1.0 dB and mel cosine at least 0.99 on every fixture with identical durations |
| Seeded mode (different noise streams), log-mel distance | mean absolute difference at most 2.0 dB, mel cosine at least 0.98 |
| Intelligibility | Whisper (large-v3-turbo) WER of native text-to-speech output at most the reference pipeline's WER + 3 points (absolute) on the 30 fixture texts |
| Native G2P vs `misaki` (lexicon only) | reported, no threshold (no tagger, no espeak) |
