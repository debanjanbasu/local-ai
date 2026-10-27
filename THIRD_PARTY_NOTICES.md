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

## Bonsai DFlash2 drafter weights

The KFlash drafter artifact (`models/bonsai2-27b-dflash2/`, not
included; converted by `local-ai bonsai --export dflash-head=DIR`) is derived
from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b),
revision `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`, licensed under the Apache
License, Version 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>). The
conversion quantizes its dense matrices per row to int8. Its NOTICE reads:

```text
Independent Bonsai DFlash2 adaptation by ProCreations, September 2026. Derived
model weights: z-lab Qwen3.8-27B-DFlash2 and frozen Prism
Ternary-Bonsai-2-27B target; Apache-2.0. Runtime: Prism/llama.cpp with
upstream DFlash2 contribution (PR27342), plus Bonsai rotation integration.
Training: vendored SpecForge with noncausal block-mask correction. See
included upstream model cards, licenses, exact source archives and manifest
for provenance.
```

It derives from [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
(Apache-2.0) and was trained against the frozen
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
target (Apache-2.0). No runtime or training code from it is used here.

## Neural Engine prefill FFN (Kapitsa, Splash)

The private `AppleNeuralEngine.framework` client in `local-metal/src/ane.rs`
(its interface declarations and method-signature table, IOSurface layout, MIL
program caching with purge-and-recompile, and the shared-event handoff with
its watchdog), the pack, weight-staging and join kernels in
`shaders/bonsai_ane.metal`, and the W8A8 FFN program, wavefront schedule and
keepalive in `local-engine/src/bonsai_native/ane_split.rs` are adapted from
[Kapitsa](https://github.com/nace-ai/kapitsa/tree/a6a7bcac98daedfeaea323baa86d3a28a59905d6)
(Copyright 2026 Nace AI), revision
`a6a7bcac98daedfeaea323baa86d3a28a59905d6`: `src/ane/ane.{h,mm}`,
`src/ane/ffn_split.{h,cpp}`, `src/kernels/ane_ffn.metal` and
`src/model/qwen35_ane.cpp`. Kapitsa adapted the interface declarations and
signature table from [Splash](https://github.com/incoai/splash) (Inco). Both
are licensed under the Apache License, Version 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>); these portions remain under
it. They were rewritten in Rust and Metal for this runtime: Bonsai's
activations and weights are already in the checkpoint's Hadamard basis, so
nothing is rotated again; ternary weights become per-row int8 by new block
scales; the whole FFN runs on the Neural Engine.

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

Audio decoding for the transcription endpoints and for audio embeddings
(`embedding::decode_audio`) links
[Symphonia](https://github.com/pdeljanov/Symphonia) 0.6.1 and its codec and
format crates from crates.io, unmodified, under the Mozilla Public License 2.0
(<https://mozilla.org/MPL/2.0/>); their source is available from crates.io and
the repository above.

## libjpeg-turbo (image embeddings)

JPEG decoding for image embeddings statically links libjpeg-turbo, built from
source by the `mozjpeg-sys` crate (2.2) and used through `mozjpeg` (0.10),
unmodified, under the IJG License, the Modified (3-clause) BSD License and the
zlib License (<https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/LICENSE.md>).
This software is based in part on the work of the Independent JPEG Group.

## Media crates

The media features link these crates from crates.io, unmodified, under the
licences their authors chose (full texts in each crate's source):

| Crate | Used for | Licence |
| --- | --- | --- |
| `image` 0.25 | image decoding (non-JPEG) for image embeddings | MIT OR Apache-2.0 |
| `rubato` 5 | resampling to 16 kHz for audio embeddings | MIT OR Apache-2.0 |
| `realfft` 3.5 | log-mel STFT for audio embeddings | MIT |
| `rustfft` 6.4 | log-mel STFT for transcription | MIT OR Apache-2.0 |
| `flate2` 1 with `libz-sys` (zlib) | Whisper's compression-ratio test | MIT OR Apache-2.0; zlib: zlib License |
| `flacenc` 0.5 | FLAC speech output | Apache-2.0 |
| `rusty_mp3` 1.0 | MP3 speech output | Apache-2.0 |
| `rusty-opus` 1.0 | Opus speech output | BSD-3-Clause |
| `ogg` 0.9 | Ogg container for Opus speech output | BSD-3-Clause |

The BSD-3-Clause crates' notices, as their binary distributions require:

`rusty-opus` 1.0.1:

```text
Copyright 2026 restsend.com

Copyright 2001-2023 Xiph.Org, Skype Limited, Octasic,
                    Jean-Marc Valin, Timothy B. Terriberry,
                    CSIRO, Gregory Maxwell, Mark Borgerding,
                    Erik de Castro Lopo, Mozilla, Amazon

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Opus is subject to the royalty-free patent licenses which are
specified at:

Xiph.Org Foundation:
https://datatracker.ietf.org/ipr/1524/

Microsoft Corporation:
https://datatracker.ietf.org/ipr/1914/

Broadcom Corporation:
https://datatracker.ietf.org/ipr/1526/
```

`ogg` 0.9.2:

```text
Copyright (c) 2016-2017 est31 <MTest31@outlook.com> and contributors
Copyright (c) 2002-2015 Xiph.org Foundation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

- Neither the name of the Xiph.org Foundation nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## zstd

The server's zstd response compression (`local-ai`) and web fetch's zstd
decoding (`local-services`) link the
[`zstd`](https://github.com/gyscos/zstd-rs) crate 0.14.3 and its
`zstd-safe` 8.2 and `zstd-sys` 2.1.2 bindings from crates.io, unmodified,
under the BSD 3-Clause License, and through `zstd-sys` statically links the
Zstandard library 1.5.7, unmodified, under its BSD License.

`zstd`, `zstd-safe` and `zstd-sys`:

```text
BSD 3-Clause License

Copyright (c) 2026, Alexandre Bury

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The Zstandard library:

```text
BSD License

For Zstandard software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook, nor Meta, nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## SQLite and the vector store stemmer

The services store (`local-services`) statically links SQLite 3.54.1,
bundled by `libsqlite3-sys` 0.39 (through `rusqlite` 0.41), unmodified.
SQLite is in the public domain (<https://sqlite.org/copyright.html>).

Vector store and repository search use Tantivy 0.27 (MIT), whose English
stemmer is the [`frostem`](https://crates.io/crates/frostem) crate
1.20260821.8, unmodified, a Rust port of the Snowball stemmers, under the
BSD 3-Clause License:

```text
Copyright (c) 2026, frostem contributors
Copyright (c) 2001, Dr Martin Porter
Copyright (c) 2004,2005, Richard Boulton
Copyright (c) 2013, Yoshiki Shibukawa
Copyright (c) 2006-2025, Olly Betts
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
  3. Neither the name of the Snowball project nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Web fetch crates

Web fetch (`local-services` `tools::fetch`) links these crates from
crates.io, unmodified, beyond those the server already uses (`quinn`, `h3`,
`h3-quinn`, `rustls`, `hyper`, `flate2`; zstd above), under the licences
their authors chose (full texts in each crate's source):

| Crate | Used for | Licence |
| --- | --- | --- |
| `html-to-markdown-rs` 3.18 | HTML to Markdown conversion | MIT |
| `astral-tl` 0.8, `html5ever` 0.40 (with `markup5ever`, `tendril`, `string_cache`, `web_atoms`), `html-escape` 0.3, `phf` 0.14 | its HTML parsing | MIT; MIT OR Apache-2.0 |
| `url` 2.5 (with `idna`, `form_urlencoded`) | resolving links against the page URL | MIT OR Apache-2.0 |
| ICU4X crates (`icu_normalizer`, `icu_properties`, `icu_collections`, `icu_locale_core`, `icu_provider`, `zerovec`, `zerotrie`, `zerofrom`, `yoke`, `tinystr`, `litemap`, `writeable`, `potential_utf`) | `idna`'s Unicode normalization | Unicode-3.0 |
| `brotli-decompressor` 6.2 | `br` content decoding | BSD-3-Clause OR MIT, used under MIT |
| `alloc-no-stdlib` 3.0, `alloc-stdlib` 0.3 | `brotli-decompressor`'s allocator | BSD-3-Clause |
| `encoding_rs` 0.8 | charset decoding of fetched text | (Apache-2.0 OR MIT) AND BSD-3-Clause |
| `dom_smoothie` 0.18 | main-content extraction (a port of Mozilla's Readability) | MIT |
| `dom_query` 0.28, `html5ever` 0.39 (with `markup5ever` 0.39, `string_cache` 0.9, `web_atoms` 0.2), `html-escape` 0.2, `phf` 0.13, `servo_arc` 0.4, `dtoa` 1.0, `gjson` 0.8, `flagset` 0.4 | its HTML parsing and DOM | MIT; MIT OR Apache-2.0; Apache-2.0 (`flagset`) |
| `selectors` 0.38, `cssparser` 0.37, `dtoa-short` 0.3 | `dom_query`'s CSS selectors | MPL-2.0 |
| `lopdf` 0.46, `nom` 8, `rangemap` 1.8, `weezl` 0.2 | PDF parsing and its stream filters | MIT; MIT OR Apache-2.0 |
| `aes` 0.9, `cbc` 0.2, `ecb` 0.2, `cipher` 0.5, `inout` 0.2, `block-padding` 0.4, `md-5` 0.11, `cpubits` 0.1 | opening PDFs encrypted with an empty user password | MIT OR Apache-2.0 |
| `stringprep` 0.1, `unicode-bidi` 0.3, `unicode-normalization` 0.1, `unicode-properties` 0.1 | PDF password preparation | MIT OR Apache-2.0; `unicode-normalization` also Unicode-3.0 |
| `brotli-decompressor` 5.0 (with `alloc-no-stdlib` 2.0, `alloc-stdlib` 0.2) | `lopdf`'s Brotli stream filter | BSD-3-Clause OR MIT, used under MIT; BSD-3-Clause |

Build-time macro crates (`cssparser-macros` (MPL-2.0), `derive_more` (MIT),
`phf_macros`, `phf_codegen`, `phf_generator`, `string_cache_codegen`) run
during the build and are not linked into the binary.

The notices those licences require in binary distributions:

`alloc-no-stdlib` 3.0.0 and 2.0.4, and `alloc-stdlib` 0.3.0 and 0.2.4:

```text
Copyright (c) 2016 Dropbox, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The ICU4X crates (version 2.4 and their companions), and the Unicode data
in `unicode-normalization` 0.1.26 (Copyright © 1991-2023 Unicode, Inc.,
under the same licence):

```text
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 2020-2024 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.

SPDX-License-Identifier: Unicode-3.0

Portions of ICU4X may have been adapted from ICU4C and/or ICU4J.
ICU 1.8.1 to ICU 57.1 © 1995-2016 International Business Machines Corporation and others.
```

`selectors` 0.38.0, `cssparser` 0.37.0 and `dtoa-short` 0.3.5 are subject
to the terms of the Mozilla Public License, v. 2.0
(<https://mozilla.org/MPL/2.0/>). They are linked unmodified; their source
code is available from crates.io and from
<https://github.com/servo/stylo>, <https://github.com/servo/rust-cssparser>
and <https://github.com/upsuper/dtoa-short>.

## LibriSpeech test fixture

`local-engine/tests/fixtures/librispeech-237-134493-0010.flac` is utterance
237-134493-0010 of the LibriSpeech ASR corpus test-clean set (Panayotov, Chen,
Povey and Khudanpur, <https://www.openslr.org/12>), licensed under CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>), unmodified.

