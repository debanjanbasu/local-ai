# local-ai

Native Rust and Metal inference for Prism ML's Ternary Bonsai 2 27B PTQ1 GGUF
on Apple Silicon. The engine reads the memory-mapped checkpoint directly,
without Python, MLX, llama.cpp, or a cloud service.

Requirements: Apple Silicon, Xcode's Metal toolchain, and the Rust nightly
pinned by `rust-toolchain.toml`. Measurements use an M4 Pro with 48 GB memory
unless a section says otherwise; [Other hardware](#other-hardware) gives a
fanless MacBook Air M2 reference.

## Quick start

Install the [model files](#model-install), then:

```bash
cargo build -p local-ai --release
target/release/local-ai serve
target/release/local-ai chat 'What is 17 * 23?'
```

The server exposes an OpenAI-compatible API at `127.0.0.1:8080`. A client that
stops reading its socket without closing it is invisible to TCP, so `serve`
takes `--stall-timeout SECONDS` (default 30, accepted 10 to 3600) to bound how
long a generation may go undelivered before it is cancelled and the engine
released; the budgets, measurements and limits are in
[docs/BONSAI.md](docs/BONSAI.md#server-api).

## Library use

`local-engine` computes synchronously and reaches no network, but it does depend
on an async runtime: `tokio` (features `rt` and `sync`) carries event transport
and `futures-core` supplies the `Stream` implementation. `Engine` is `Send` but
not `Sync`; `EngineHandle` provides a cloneable `Send + Sync` worker handle.

```rust,no_run
use local_engine::{ChatMessage, ChatRequest, Engine, Event, Sampling};
use std::ops::ControlFlow;

let mut engine = Engine::open()?;
let request = ChatRequest {
    messages: vec![ChatMessage {
        role: "user".into(), content: "Hello".into(), reasoning_content: None,
    }],
    max_tokens: 128,
    sampling: Sampling::default(),
    thinking: true,
    session: None,
};
engine.chat_with(&request, |event| {
    if let Event::Content(text) = event { print!("{text}"); }
    ControlFlow::Continue(())
})?;
# Ok::<(), local_engine::Error>(())
```

Use `Engine::open()?.chat("prompt")` for a collected response. For sharing,
call `Engine::into_handle`; its bounded queue returns cancellable event streams
that are both a blocking `Iterator` and a `Stream`. `EngineHandle::open` is
`async` and needs a runtime; everything else works without one, so a program on
another executor should reach the model through `Engine::open` plus
`spawn_blocking`. See `local-engine/examples/chat.rs`.

## Automatic policy

Startup prints every decision and reason under `experimental_bonsai`.

| Decision | Rule |
| --- | --- |
| Model | First pinned GGUF in `./models`, beside the executable, or in `~/Library/Caches/local-ai/models`; explicit path wins |
| Context | Largest value up to 262,144 whose fully grown state fits 90% of Metal's recommended working set |
| K/V | F16; Q8 only if F16 cannot provide 32,768 tokens; start at 1,024 tokens and grow on demand |
| Kernels | Metal 4 tensor kernels where supported, SIMD fallbacks elsewhere |
| Speculation | Lossless suffix lookup; gated depth-3 MTP when a head is installed, preferring the int8 artifact over the BF16 source |
| Prompt cache | Purgeable GPU checkpoints and disk snapshots; host snapshots only below the 500 MiB/s storage threshold |
| Disk | `min(512 GiB, 25% of free space)` under `~/Library/Caches/local-ai/prompt-cache` |
| Server | Eight-request queue; HTTP/3 when its certificate and key are discovered |

## Performance

- Decode: 17.5 tok/s plain, about 27 tok/s on arithmetic with speculation,
  and 40–54 tok/s on code copy-edits.
- Prefill: about 96 tok/s at 4K tokens and 56 tok/s at 128K.
- Prompt cache: follow-up turn 2.7 s versus 25 s; disk restore 0.69 s;
  shared 6.7K-token prefix TTFT 1.2 s versus 73 s.
- Memory: about 1.4 GB peak for a short request and 1.1 GB idle after an
  8K-token prompt. File-backed weights remain demand-paged.

### Other hardware

The figures above are from an M4 Pro. The same engine on a fanless MacBook Air
M2, 8-core CPU, 10-core GPU, 16 GB unified memory, about 100 GB/s memory
bandwidth:

| Measurement | Value |
| --- | ---: |
| Decode, speculation on | 5.2 tok/s |
| Decode, `bonsai --no-speculation` | 2.55 tok/s |
| Speculation gain | 2.0x |
| Model load | 0.45 s |
| Peak RSS | 361 MB |
| Peak memory footprint | 0.96 GB |
| Swaps | 0 |
| Automatically selected context | 61,538 tokens |
| MTP head load, cold from the BF16 source | 5.94 s |
| MTP head load, warm int8 artifact | 3.18 s |

The lower tok/s is hardware, not a defect. Decode is memory-bandwidth-bound:
PTQ1 decode moves about 5.65 GB of weights per token, and
[docs/BONSAI.md](docs/BONSAI.md) records 97–98% of decode wall time on the
GPU, so throughput tracks memory bandwidth closely. This machine's roughly
100 GB/s is about a third of the M4 Pro's, and its decode rate is about a third
too. Expect prefill to be slower by a similar margin;
`GenerationStats.prefill` (`local-engine/src/runtime.rs:77`) is a duration
rather than a rate, so no prefill tok/s figure is published here.

Two things adapt without configuration. Context is derived at startup from
Metal's `recommendedMaxWorkingSetSize`, which is why 16 GB of unified memory
still yields a 61,538-token context rather than a hard-coded limit. The M2 also
selects Metal 4 tensor kernels for full-attention prefill
(`attention_kernel: tensor_f32`), not the `simd_f32` fallback.

### Speculation, decomposed

`--no-speculation` is one flag over two independent mechanisms: the MTP head,
a learned draft that speculates ahead, and n-gram suffix lookup
(`ngram_policy`, `min_match: 24`), an exact-match reuse of a previously-seen
24-token suffix that proposes its known continuation. Both are honestly called
speculation and the flag disables both, so the 2.0x above is not in dispute;
what an A/B against that flag cannot show is how the gain divides between them.
Measured today on this M2, greedy, median of three repetitions per arm, every
arm producing byte-identical output: on varied prose and on ordinary repetition
the n-gram path never fired at all, and the gains there — **1.26x** (1.22x on an
independent second run) and **1.62x** — are MTP alone. The larger figure appears
only where n-gram lookup engages, which on a verbatim repetition workload it did
at 260 of 261 proposals accepted for 0.1 ms of total lookup time; both
mechanisms were live in that run, so these figures bound the head's contribution
rather than dividing the gain between the two. Verification, not drafting, is
89.0% of decode time, and a batched verify token costs **0.65x** what a standalone
decode token costs — which is also why a published CUDA measurement of the same
PTQ1 format gained only 1.6%.

See [docs/BONSAI.md](docs/BONSAI.md) for the technical reference.

## Model install

Weights are not included. The runtime validates these pinned files:

| File | Bytes | SHA256 | Needed |
| --- | ---: | --- | --- |
| `models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` | always |
| `models/bonsai2-27b-mtp/model_mtp.safetensors` | 849,400,392 | `7a4a18b2d02116ef184d1b0ee4af46d829825ff2c042f79cf37ef8a03c399218` | once, to build the [head artifact](#mtp-head-artifact) |

The GGUF is required. The BF16 MTP head is not: it is the *rebuild source*
for the [int8 head artifact](#mtp-head-artifact), and an install that ships the
artifact instead never opens it. Discovery prefers the artifact, so a working
install can carry 58.1% less head data than one that stages the source.

`--include` is required on both commands. Without it `hf download` fetches the
whole repository: the GGUF repository holds 68.5 GB across five checkpoints and
the MTP repository 9.7 GB across 86 files, most of it unusable here. One
measured attempt transferred 21 GB+ before being stopped. Only the files listed
above are ever opened, and the rest are not merely unused:

| Skipped file | Bytes | Why |
| --- | ---: | --- |
| `Ternary-Bonsai-2-27B-F16.gguf` | 53,808,408,928 | Unquantized; the runtime reads PTQ1 only |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 | Wrong quantization; the runtime reads PTQ1 only |
| `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` | 931,145,856 | Vision projector; no multimodal path exists |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 629,246,976 | Vision projector; no multimodal path exists |

There is no vision, clip, projector, mmproj, or image code anywhere in
`local-engine/src` or `local-metal/src`, so the 1.45 GiB of projectors cannot be
loaded. `local-engine/src/resources.rs` names the only two paths the runtime
resolves, `MODEL_RELATIVE` and `MTP_RELATIVE`.

Both repositories are pinned. The MTP pin is that repository's current head, and
`model_mtp.safetensors` at it is byte-identical to the SHA256 above. The GGUF
repository's head has moved past `6ed5e12b`, so keep the pin rather than
tracking the branch.

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  --revision 6ed5e12bf84b7a63069882c91dd9e9218647d17b \
  --include 'Ternary-Bonsai-2-27B-PTQ1_0.gguf' \
  --local-dir models/bonsai2-27b-ptq1
hf download ProCreations/Ternary-Bonsai-2-27B-MTP \
  --revision efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51 \
  --include 'model_mtp.safetensors' \
  --local-dir models/bonsai2-27b-mtp
shasum -a 256 \
  models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  models/bonsai2-27b-mtp/model_mtp.safetensors
```

Compare against the table above. The `shasum` arguments are spelled out rather
than globbed so a stray F16 or PQ2_0 file in the directory cannot pass for the
checkpoint.

If Hugging Face is unavailable, stage private CPU-only Kaggle jobs. Staging
does not submit them; review the generated directories first.

```bash
python3 tools/kaggle_bonsai_job.py \
  --kernel USER/bonsai2-ptq1 --output cache/kaggle-bonsai
python3 tools/kaggle_bonsai_mtp_job.py \
  --kernel USER/bonsai2-mtp --output cache/kaggle-bonsai-mtp
```

The tools produce digest-checked, no-clobber archives and retain upstream
licenses. The head tool also supports `--install ARCHIVE` and optional
`--destination PATH`.

### MTP head artifact

The BF16 head is quantized to symmetric int8 at load time and cached, which
makes it a runtime cost rather than a shipping format. `--export mtp-head=DIR`
runs that same transform offline and writes the result as a file an install can
ship, so speculation no longer requires the 849 MB source at runtime.

Prebuilt copies are published for both encodings:

```bash
hf download debanjanbasu/Ternary-Bonsai-2-27B-MTP-int8 \
  --revision 258b56354b0f3c2d8907c5d4cacc04e9a860b785 \
  --include 'mtp-head-int8-v2.bin' \
  --local-dir models/bonsai2-27b-mtp
```

Take the stored form unless download size matters more than memory: it is
mapped and demand-paged into Metal with no copy, while the `.zst` variant must
inflate into one 425,263,104-byte anonymous buffer held for the process
lifetime. `mtp-head-int8-v2.bin.zst` is 16.3% smaller at 412 MiB against
752 MiB peak RSS.

To build it yourself instead:

```bash
target/release/local-ai bonsai \
  --model "$PWD/models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --export mtp-head=models/bonsai2-27b-mtp
```

It writes `models/bonsai2-27b-mtp/mtp-head-int8-v2.bin`. No engine is started
and no checkpoint is opened: the command locates the installed model directory
only to find the head beside it, then transforms the head file alone. `--export`
takes one kind and accepts only `--model`, so it is rejected alongside any
prompt, `--prompt-file`, `--json`, `--tokenize`, or sampling flag. The two head
kinds are `--export mtp-head=DIR` and `--export mtp-head-zstd=DIR`.

The artifact is **the bytes the loader already computes**, not a second
quantization, so there is no quality trade-off to weigh. It is the transform a
cache miss runs; both forms describe the same 25 sections at the same offsets,
and inflating one form rebuilds the other byte for byte. Each load re-derives the
payload digest from the file and compares it with the header, which is what lets
an install ship the artifact and trust it.

| Form | Bytes | Saving against the 849 MB source |
| --- | ---: | ---: |
| `--export mtp-head=DIR`, `"compression":"stored"` | 425,263,104 | 49.9% |
| `--export mtp-head-zstd=DIR`, `"compression":"zstd"` | 355,837,652 | 58.1% |

Both forms report the same identity, which is what makes them interchangeable:

| Field | Value |
| --- | --- |
| `payload_bytes` | 425056256 |
| `payload_sha256` | `a98a24e58fcfb711cd2b466d3a32375e428000b8f2b06b677d5839946b20f93b` |

The digest covers the 25 logical sections concatenated in order, skipping the
16 KiB alignment padding, so it identifies the payload rather than the file
layout. The filename is identical for both forms: the file's own magic records
which encoding its sections are in, so the loader reads either without being
told which to expect.

`--export mtp-head-zstd=DIR` writes the same file with its section region behind
one zstd frame, at **level 19**. Level 19 is a measurement, not a default: on
this payload level 22 measures 358,063,503 bytes and `--long=27` measures
355,688,546,
both worse than level 19's 355,361,865. Widening the window does not help this
data.

The saving costs resident memory, which is why discovery prefers the stored form.
The stored file is mapped and `from_bytes_no_copy` hands Metal pointers into
pages the kernel demand-pages, so it spends no anonymous RAM; an inflated head
must exist in full before any section can be read, so every load through one
holds a 425,263,104-byte anonymous allocation for the life of the process. Peak
RSS on an M2 is 412 MiB stored against 752 MiB zstd.

Installed beside the checkpoint, the artifact is preferred over the BF16 source
and that source is never `open()`ed, so a load drops from 5.94 s cold to 3.18 s
warm on an M2. Measured end to end on the same machine with
`model_mtp.safetensors` **absent**, the engine loads the artifact, reports
`mtp.enabled: true` with `head_cache: "hit"`, and answers correctly. A corrupt,
truncated, or renamed artifact is always a hard error naming the file and the
reason, even when a valid BF16 source sits beside it; there is deliberately no
fallback to the source, because silently decoding without speculation looks like
a healthy install rather than a broken one.

This is specific to the head. The 5,946,648,928-byte GGUF is not compressible in
practice: the measured ceiling is 1.074x and the best measured result 1.29%.

## Workspace

- `local-engine`: GGUF reader, tokenizer, inference, caching, speculation, API.
- `local-ai`: `chat`, `bonsai`, and `serve` commands and HTTP server.
- `local-metal`, `shaders`: Metal dispatch code, kernels, and GPU tests.
- `tools`: Kaggle staging and installation helpers and tests.
- `docs/BONSAI.md`: technical reference and measurements.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked --release
uvx ruff check tools/
python3 -m unittest discover -s tools -p 'test_*.py'
```

Real-model checks are in [docs/BONSAI.md](docs/BONSAI.md#verification).

## Running the real-model integration tests

Fifteen tests across the workspace are `#[ignore]`d behind environment
variables, fourteen of them in `local-engine`. `cargo test --workspace` above
skips all of them, so the default verification run never exercises them. Only
three of the `local-engine` tests actually need an environment variable: they
`expect` `BONSAI_GGUF` or `BONSAI_INDEX` outright and have no fallback. The
other eleven resolve the workspace-relative defaults, so the weights need only
be staged as above. Export the index once, then point the two variables that
have no default at absolute paths; the block spells out `BONSAI_MTP_HEAD` too
so the head is pinned even though its default now resolves:

```bash
mkdir -p cache
./target/release/local-ai bonsai \
  --model "$PWD/models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --export index > cache/bonsai-index.json

BONSAI_GGUF=$PWD/models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
BONSAI_MTP_HEAD=$PWD/models/bonsai2-27b-mtp/model_mtp.safetensors \
BONSAI_INDEX=$PWD/cache/bonsai-index.json \
  cargo test -p local-engine --lib --release -- \
  --include-ignored --test-threads=1 bonsai
```

`--test-threads=1` is load-bearing:

- **Do not run these in parallel.** Cargo's default test threads make them fail
  spuriously. Each test maps and uploads the full 5.5 GB checkpoint, and
  concurrent unified-memory and GPU contention purges volatile prompt-cache
  state, so cache-source assertions fail on correct code.

- **Serial execution is necessary but not sufficient on a 16 GB machine.** The
  same pressure builds up across a whole serial run, because every test in the
  binary still maps and uploads the full checkpoint and then releases it. Two
  assertions were sensitive enough to that cumulative pressure to be rewritten:
  `interleaved_and_restarted_session_snapshots_match_cold_tokens` now accepts
  either resume tier (`Host` or `Disk` — both write a snapshot, and restoring the
  host one copies KV state to the GPU, so it can fail and fall through), and
  `prompt_checkpoints_match_cold_greedy_for_extension_and_mid_prompt_edit` asserts
  the reuse *bound* rather than an exact count. Both still assert token equality
  against a cold reference, which is the property that actually matters. Running
  either test in its own process restores the stricter original expectation.

`BONSAI_MTP_HEAD` no longer has to be absolute. `DEFAULT_BONSAI_MTP_HEAD` at
`local-engine/src/bonsai_mtp/policy.rs:65` and `DEFAULT_BONSAI_GGUF` at
`local-engine/src/bonsai.rs:50` are each a `#[cfg(not(test))]`/`#[cfg(test)]`
pair: a release build gets the workspace-relative
`models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf`, and the test build
gets `concat!(env!("CARGO_MANIFEST_DIR"), "/../models/...")`, which anchors the
identical file to the workspace root. `cargo test -p local-engine` runs with
`local-engine/` as its working directory, so the relative string never resolved
there and the test arm fixes exactly that. Running the `local-engine` ignored
suite with no environment variables set therefore gets as far as the eleven
tests that read those defaults; the rest stop on a bare `expect`.

The three that stop are the ones with no fallback, listed so they are not
re-derived: `bonsai::tests::real_bonsai_profile_is_pinned` `expect`s
"BONSAI_GGUF must name the verified GGUF",
`bonsai_tokenizer::tests::real_gguf_roundtrips_and_renders_prompts` `expect`s
"BONSAI_GGUF is required", and
`bonsai_tokenizer::tests::exported_tokenizer_object_reproduces_prism_token_ids`
`expect`s `BONSAI_INDEX`. The four `bonsai_native` MTP tests additionally read
`DEFAULT_BONSAI_MTP_HEAD`, so they need
`models/bonsai2-27b-mtp/model_mtp.safetensors` staged even with
`BONSAI_MTP_HEAD` unset: they build their settings from that path directly and so
bypass the artifact preference `local-ai` discovery applies. `BONSAI_MTP_HEAD`
itself may name either form, since the loader tells them apart by the file's own
magic.

`prompt_checkpoints_match_cold_greedy_for_extension_and_mid_prompt_edit`
(`local-engine/src/bonsai_model/tests.rs`) used to fail, and no longer does: it
asserted `reused_prompt_tokens > 0` against a 4-slot volatile LRU GPU
checkpoint cache while deliberately holding a second live engine open for its
cold references, whose unified-memory pressure purges those very checkpoints. It
now asserts the bound that survives a purge and keeps the token equality it was
aborting before, which had therefore never actually run. Its sibling
`purged_gpu_and_host_snapshots_fall_back_to_disk_with_identical_tokens` already
covered the purge path.

`deep_cancellation_stops_in_prefill_and_leaves_the_prompt_cache_empty`
(`local-engine/src/bonsai_model/tests.rs`) covers the other new path: it trips a
`CancelToken` from a producer thread during prefill, asserts
`StopReason::Cancelled` rather than an error, and asserts the next request on
the same engine reuses nothing and matches a cold reference token for token.

One further ignored test, `dump_full_attention_kv_caches`
(`local-engine/src/bonsai_native/tests.rs:757`), is a dump utility rather than
an assertion: it also needs `BONSAI_KV_DUMP_DIR` and writes
`layer_<index>_{k,v}.f16`, and returns early when the variable is unset.

GPU-side ignored tests in `local-metal` use the same serial requirement:

```bash
cargo test -p local-metal --release -- --include-ignored --test-threads=1
```

The fifteenth is a server test at `local-ai/src/cli/serve/tests.rs:39`,
`serve_real_bonsai_validates_before_streaming_and_preserves_history`. It needs
only `BONSAI_GGUF` and lives in a different package:

```bash
BONSAI_GGUF=$PWD/models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  cargo test -p local-ai --release -- --include-ignored --test-threads=1 serve
```

## License and sources

The code is dual-licensed under [Apache-2.0](LICENSE-APACHE) or
[MIT](LICENSE-MIT). Model files retain their own licenses. The PTQ1 layout and
kernels derive from [Prism ML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp);
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model sources are
[Prism ML's GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
and the [community MTP head](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP).
