# Third-party notices

## Bonsai PTQ1_0 block layout and decoder

The PTQ1_0 block layout (128 signed trits plus one FP16 scale per 28 bytes) in
`local-engine/src/bonsai.rs` and the decoders in `shaders/` are adapted from
[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp/tree/9a9394a895b96003ca842a6041cb28ac49a108f7),
revision `9a9394a895b96003ca842a6041cb28ac49a108f7`:
`ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c` and
`tests/test-ptq1_0-element-map.cpp`.

The Metal matvec/matmul, Hadamard and online-softmax attention kernels in
`shaders/` were originally adapted from the same fork and have since been
rewritten for this runtime (PTQ1 small-batch projection, Hadamard, GDN and
fused decode attention); the layout and decoder attribution above still
applies to them. The blocked online-softmax attention in
`shaders/bonsai_ops.metal` follows the fork's
`ggml/src/ggml-metal/kernels/fa.metal` at revision
[`0781925904391351963d499cb32cd735849b06a5`](https://github.com/PrismML-Eng/llama.cpp/tree/0781925904391351963d499cb32cd735849b06a5),
and the tile-local PTQ1 decoding in `shaders/bonsai.metal` follows its
`mul_mm.metal`; both are under the same license.

This attribution concerns code, not model weights. Downloaded model weights
retain their own license and notices.

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MTP head training scripts

`tools/mtp_train/mtp_head.py`, `data.py` and `train.py` are adapted from
[xkm/qwen3.8-27b-mtp-head-retrained](https://huggingface.co/xkm/qwen3.8-27b-mtp-head-retrained/tree/b0b5836b399d69730373afa6f1346220465394b0)
(`train/`), revision `b0b5836b399d69730373afa6f1346220465394b0`, licensed under
the Apache License, Version 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>). They were modified to read this
runtime's capture shards and frozen tables and to match its draft chain layout.

## llguidance, toktrie and derivre

Schema-constrained generation and the native tool-call grammar (llguidance's
built-in `lark` feature, which adds no crate) in `local-engine` link these
crates from crates.io, unmodified: [`llguidance`](https://github.com/guidance-ai/llguidance)
1.9.1 and its dependencies `toktrie` 1.9.1 (same repository) and
[`derivre`](https://github.com/microsoft/derivre) 0.3.13. Each is licensed
under the MIT License; the `LICENSE` file shipped in each crate's registry
package is identical and reproduced below.

```text
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

## Kokoro-82M and misaki (text to speech)

`local-engine/src/speech/` and `shaders/kokoro.metal` reimplement, in Rust and
Metal, the inference code of [hexgrad/kokoro](https://github.com/hexgrad/kokoro)
0.9.4 (`kokoro/model.py`, `modules.py`, `istftnet.py`; Apache-2.0), itself
adapted from [StyleTTS 2](https://github.com/yl4579/StyleTTS2) (MIT), and port
the English grapheme-to-phoneme rules of [hexgrad/misaki](https://github.com/hexgrad/misaki)
0.9.4 (`misaki/en.py`; Apache-2.0). Number words follow the forms of
[num2words](https://github.com/savoirfairelinux/num2words) (LGPL-2.1; only its
output conventions are reproduced, no code). The misaki lexicons and the
Kokoro-82M weights and voices (Apache-2.0) are not included; the converter
reads them from the user's download. Copyright (c) 2025 hexgrad. Licensed under
the Apache License, Version 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>);
see `LICENSE-APACHE`.

## Whisper decoding and transcription logic

`local-engine/src/transcription/decode.rs` and the log-mel front end in
`local-engine/src/transcription/mel.rs` port the decoding, timestamp-rule,
temperature-fallback and long-form transcription logic and the spectrogram
of [openai/whisper](https://github.com/openai/whisper) 20250625
(`whisper/decoding.py`, `whisper/transcribe.py`, `whisper/audio.py`); the
installed artifact embeds the package's `assets/mel_filters.npz`. The Whisper
large-v3 weights the converter reads are released under the same licence.

```text
MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Symphonia

Audio decoding for the transcription endpoints links
[Symphonia](https://github.com/pdeljanov/Symphonia) 0.6.1 and its codec and
format crates from crates.io, unmodified, under the Mozilla Public License 2.0
(<https://mozilla.org/MPL/2.0/>); their source is available from crates.io and
the repository above.

## LibriSpeech test fixture

`local-engine/tests/fixtures/librispeech-237-134493-0010.flac` is utterance
237-134493-0010 of the LibriSpeech ASR corpus test-clean set (Panayotov, Chen,
Povey and Khudanpur, <https://www.openslr.org/12>), licensed under CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>), unmodified.

