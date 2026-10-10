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

The server listens on `127.0.0.1:8080` and implements a text, function-calling and
JSON-constrained subset of the OpenAI Chat Completions, Completions and Responses HTTP
contracts (not full OpenAI compatibility). A client that stops reading its
socket without closing it is invisible to TCP, so `serve`
takes `--stall-timeout SECONDS` (default 30, accepted 10 to 3600) to bound how
long a generation may go undelivered before it is cancelled and the engine
released; the budgets, measurements and limits are in
[docs/BONSAI.md](docs/BONSAI.md#server-api).

Responses are stateless and nothing is stored by default. `--response-store DIR`
opts in to keeping them (owner-only, unencrypted files) for
`GET`/`DELETE /v1/responses/{id}`, `input_items` pagination and
`previous_response_id`, which carries over conversation items but not
`instructions`, tools or sampling settings. With a store, `background: true`
requests are stored before they are acknowledged: without `stream` the `POST`
returns the `queued` response for polling `GET /v1/responses/{id}`; with
`stream: true` it streams journaled events that
`GET /v1/responses/{id}?stream=true&starting_after=N` resumes after any
sequence number, even across a server restart. A disconnected or stalled
stream does not cancel the job; `POST /v1/responses/{id}/cancel` does.
Background `store: false` (temporary retention) is refused, and a job
interrupted by a restart is marked `failed`, never resumed.

Chat `response_format` and Responses `text.format` accept `json_object` and
`json_schema`; the engine compiles the schema with llguidance before queueing
and masks every answer token to it, whether or not `strict` is set. A format
cannot be combined with tools unless `tool_choice` is `"none"`, and a token
limit leaves the JSON incomplete (`length`/`incomplete`, never a success).

`--reasoning-key FILE` adds
AES-256-GCM `reasoning.encrypted_content` for stateless replay; the raw
reasoning is still returned and, with a store, still written, so it is neither
encryption at rest nor hidden reasoning. `POST /v1/responses/input_tokens`
returns the exact templated prompt-token count without generating. This is a
subset of the public OpenAI contract, not a drop-in Codex provider; the gaps
are listed in [docs/BONSAI.md](docs/BONSAI.md#server-api).

## Library use

`local-engine` computes synchronously and reaches no network, but it does depend
on an async runtime: `tokio` (features `rt` and `sync`) carries event transport
and `futures-core` supplies the `Stream` implementation. `Engine` is `Send` but
not `Sync`; `EngineHandle` provides a cloneable `Send + Sync` worker handle.

```rust,no_run
use local_engine::{ChatMessage, ChatRequest, Engine, Event, ResponseFormat, Sampling};
use std::ops::ControlFlow;

let mut engine = Engine::open()?;
let request = ChatRequest {
    messages: vec![ChatMessage {
        role: "user".into(), content: "Hello".into(), ..ChatMessage::default()
    }],
    max_tokens: 128,
    sampling: Sampling::default(),
    thinking: true,
    session: None,
    tools: vec![],
    response_format: ResponseFormat::Text,
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

`count_chat_tokens` and `count_completion_tokens` on `Engine` and
`EngineHandle` return the exact prompt length generation would prefill, using
the same template and tokenizer. They run on the CPU, queue no job and submit
no GPU work.

## Automatic policy

Startup prints every decision and reason under `experimental_bonsai`.

| Decision | Rule |
| --- | --- |
| Model | First pinned GGUF in `./models`, beside the executable, or in `~/Library/Caches/local-ai/models` |
| Context | Largest value up to 262,144 whose fully grown state fits 90% of Metal's recommended working set |
| K/V | Hadamard-rotated Q8 (8.5 bits/value); start at 1,024 tokens and grow on demand |
| Kernels | Metal 4 tensor kernels where supported, SIMD fallbacks elsewhere |
| Speculation | Lossless suffix lookup; gated depth-3 MTP when the mixed ternary/int8 head artifact is installed |
| Prompt cache | Purgeable GPU checkpoints and disk snapshots; host snapshots only below the 500 MiB/s storage threshold |
| Disk | Smallest of the checkpoint's size, an eighth of free disk and a quarter of physical memory, under `~/Library/Caches/local-ai/prompt-cache` |
| Server | Up to eight requests decoded together in one batched pass per step, verifying lookup and MTP drafts where a cost model finds them worth their rows; a long prompt prefills 24 to 31 tokens per step inside that pass (32 in a pass of its own beside a single stream); eight more queued; HTTP/3 when its certificate and key are discovered |

## Performance

- Decode: 31.2 tok/s plain and 38.6 tok/s geomean with the default gated
  depth-3 speculation (35–43 across prose, explanation, arithmetic, code,
  essay and thinking prompts) in the depth/margin sweep, with byte-identical
  greedy text; later best-of-two six-prompt A/B runs on newer kernels
  measured 30–32 and 38–39 tok/s. Two 600-token copy-edit prompts reached
  about 90 tok/s with suffix lookup. Run-by-run provenance is in
  [docs/BONSAI.md](docs/BONSAI.md#performance).
- Server: concurrent requests decode in one batched pass per step, their MTP
  heads drafting together in one head pass per depth; 300-token chat
  requests reach about 39, 56 and 74 tok/s aggregate at 2, 4 and 8 streams
  against 33 for one. A 9.2K-token prompt arriving beside four streams
  prefills inside their passes: they keep a token every 0.32 s (p50; 0.36 s
  p99) and its first token comes at 106 s. Copy-edit requests verify
  suffix-lookup drafts inside the batch: 37.9 against 34.9 tok/s aggregate
  at four streams for the previous build, byte-identical to each request
  alone.
- Prefill (cold, no speculation): about 125 tok/s from 2K to 8K tokens and
  106 tok/s at 32K on the large-batch kernels. The 56 tok/s at 128K was
  measured on the original build and has not been re-measured.
- Prompt cache: follow-up turn 2.7 s versus 25 s; disk restore 0.69 s;
  shared 6.7K-token prefix TTFT 1.2 s versus 73 s.
- Memory: 0.8–0.9 GB peak for a short request and 2.09 GB peak footprint for
  a 16K-token prompt plus 500 generated tokens (F16 recurrent state, copy-free
  verify rollback, 4K-step K/V growth). File-backed weights remain
  demand-paged.

### Other hardware

The figures above are from an M4 Pro. The same engine on a fanless MacBook Air
M2, 8-core CPU, 10-core GPU, 16 GB unified memory, about 100 GB/s memory
bandwidth:

| Measurement | Value |
| --- | ---: |
| Decode, speculation on | 5.2 tok/s |
| Decode, `bonsai --no-speculation` | 2.55 tok/s |
| Speculation gain | 1.15x–1.96x |
| Model load | 0.45 s |
| Peak RSS | 361 MB |
| Peak memory footprint | 0.96 GB |
| Swaps | 0 |
| Automatically selected context | 61,538 tokens, under the former F16 K/V default |
| MTP head load, retired int8 head: cold from the BF16 source | 5.94 s |
| MTP head load, retired int8 head: warm artifact | 3.18 s |

**Read the two decode rates as one session, not as constants.** Both come from
the same machine with `--no-thinking --greedy`, but they are not comparable to
each other as published: the speculation run used a planets prompt and the
baseline a counting prompt, so the original single "2.0x" was a ratio across two
different workloads. Re-measured on one prompt with
[`tools/benchmark.py`](tools/benchmark.py), which refuses to compare
configurations that emit different text, the gain is **1.15x on varied prose**
and **1.62x on repetitive text**. A separate run with the speculation counters
([below](#speculation-decomposed)) found n-gram lookup idle on both kinds of
workload, so those gains come from the MTP head.

Absolute rates are the least portable number here: they move with prompt length
because prefill amortizes differently (a 160-token generation runs well above
the rate a 1024-token one does on identical hardware), and two measurements of
the same configuration on this box have differed by 1.6x when the machine was
busy. Measure the ratio, not the rate.

The lower tok/s is mostly hardware, not a defect. PTQ1 decode moves about
5.65 GB of weights per token and [docs/BONSAI.md](docs/BONSAI.md) records
97–98% of decode wall time on the GPU. This machine's roughly 100 GB/s is
about a third of the M4 Pro's, and on the build measured here its decode rate
was about a third too. The M4 Pro's current single-row kernels were
ALU-limited before the half-FMA decoder and now reach 82–95% of a streaming
read control; the M2 has not been re-measured on them. Expect prefill to be
slower by a similar margin; `GenerationStats.prefill`
(`local-engine/src/runtime.rs`) is a duration rather than a rate, so no M2
prefill tok/s figure is published here.

Two things adapt without configuration. Context is derived at startup from
Metal's `recommendedMaxWorkingSetSize`, which is why 16 GB of unified memory
still yields a 61,538-token context rather than a hard-coded limit. That figure
predates Q8 K/V, which costs 38 KiB per token (MTP cache included) against
F16's 68 KiB, so the same machine should now select roughly 1.8 times as much;
it has not been re-measured. The M2 also
selects Metal 4 tensor kernels for full-attention prefill
(`attention_kernel: tensor_f32`), not the `simd_f32` fallback.

### Speculation, decomposed

`--no-speculation` disables two independent mechanisms: the MTP head and
n-gram suffix lookup (`min_match: 12`). On this M2, greedy, median of three
repetitions per arm, every arm byte-identical: varied prose gained **1.26x**
(1.22x on a second run) and counting **1.62x** with n-gram lookup never
firing, so those gains are the head alone. The **1.96x** verbatim-repetition
row is the only one where n-gram lookup engaged (260 of 261 proposals
accepted), with both mechanisms live, so the split there is not measured.
Verification, not drafting, is 89.0% of decode time, and a batched verify
token costs **0.65x** a standalone decode token. The full table, time
breakdown and comparison with a published CUDA result are in
[docs/BONSAI.md](docs/BONSAI.md#what-the-speculation-gain-is-made-of).

## Model install

Weights are not included. The runtime validates these pinned files:

| File | Bytes | SHA256 | Needed |
| --- | ---: | --- | --- |
| `models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` | always |
| `models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin` | 166,969,344 | `83cc72279159c12255784f8a0a08ddc916519d466811276e4ff621eb7b88bd1b` | for MTP speculation |

The GGUF is required. The [mixed ternary/int8 MTP head](#mtp-head-artifact) is
not: without it speculation falls back to suffix lookup alone and the startup
policy says why.

The GGUF repository is pinned. Its head has moved past `6ed5e12b`, so keep the
pin rather than tracking the branch.

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  --revision 6ed5e12bf84b7a63069882c91dd9e9218647d17b \
  --include 'Ternary-Bonsai-2-27B-PTQ1_0.gguf' \
  --local-dir models/bonsai2-27b-ptq1
```

The `--include` filter is required. Without it `hf download` fetches the
whole repository: it holds 68.5 GB across five checkpoints, most of it
unusable here. One
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
resolves, `MODEL_RELATIVE` and the head's `MTP_DIRECTORY`/`MTP_HEAD_ARTIFACT`.

If Hugging Face is unavailable, stage a private CPU-only Kaggle job. Staging
does not submit it; review the generated directory first.

```bash
python3 tools/kaggle_bonsai_job.py \
  --kernel USER/bonsai2-ptq1 --output cache/kaggle-bonsai
```

The tool produces digest-checked, no-clobber archives and retains upstream
licenses.

The optional MTP head is published as a release asset,
[`mtp-head-mixed-v1`](https://github.com/debanjanbasu/local-ai/releases/tag/mtp-head-mixed-v1)
(with its model card, license, notice and checksums), mirrored on Hugging Face
as
[`debanjanbasu/Ternary-Bonsai-2-27B-MTP-mixed`](https://huggingface.co/debanjanbasu/Ternary-Bonsai-2-27B-MTP-mixed).
From the repository root:

```bash
python3 tools/fetch_mtp_head.py
```

It streams the pinned asset into `models/bonsai2-27b-mtp/`, checks its size and
SHA256, and never replaces an existing file.

If you installed both files, verify them against the table above:

```bash
shasum -a 256 \
  models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin
```

The `shasum` arguments are spelled out rather than globbed so a stray F16 or
PQ2_0 file in the directory cannot pass for the checkpoint.

### MTP head artifact

The shipped MTP head is a trained **mixed ternary/int8** head, Bonsai-style:
one Qwen3.5 decoder layer whose nine matrices live in the target's
Hadamard-rotated basis. `q`, `gate`, `up` and `down` are exactly ternary with
one F16 scale per 128 columns, stored as the target's own `PTQ1_0` blocks and
multiplied by its kernels; the sensitive `fc`, `k`, `v` and `o` are per-row
int8. It is 166,969,344 bytes on disk and memory-mapped into Metal with no
copy: **258 MB less resident memory** than an all-int8 head at the same decode
speed (29.48 against 29.57 tok/s geomean, 78.6% against 80.4% acceptance).

| Field | Value |
| --- | --- |
| File | `models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin` |
| `payload_bytes` | 166,799,360 |
| `payload_sha256` | `fb05507f87f54432c782b0fcb35e5cf1da1b4997610ab35a3d98502e536351cd` |

The digest covers the 21 logical sections (nine matrices, seven folded
norms) concatenated in order, skipping the 16 KiB alignment padding. Each load
re-derives it from the file and compares it with the header. A corrupt,
truncated, or renamed artifact is always a hard error naming the file and the
reason; there is deliberately no fallback, because silently decoding without
speculation looks like a healthy install rather than a broken one. A missing
head turns MTP off and the startup policy records the reason.

The head was distilled from the
[community BF16 MTP head](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP)
by ProCreations, which is the teacher, not a runtime input. The recipe is ours:
`mtp-capture` records the target's hidden states and logits,
`tools/mtp_train/train_ternary.py` runs ternary QAT with a KL objective to the
teacher's draft chains, and `tools/mtp_train/convert.py to-ternary` writes
`model_mtp_ternary.safetensors` (its `--int8` option, defaulting to the
training checkpoint's own set, selects the int8 matrices). Pack it into the
artifact with:

```bash
target/release/local-ai bonsai --export mtp-head=DIR
cp DIR/mtp-head-ptq1-v1.bin models/bonsai2-27b-mtp/
```

A head may also be mixed-precision: any matrix given as `<name>.int8` (I8 in
[-127, 127]) and `<name>.row_scales` (F32 per row) instead of codes and scales
is stored as int8 in the same artifact and multiplied by the int8 kernels from
the same rotated activations; see [docs/BONSAI.md](docs/BONSAI.md#mixed-precision-heads).

No engine is started and no checkpoint is opened. `--export` takes one kind
and no other options, so it is rejected alongside any prompt, `--prompt-file`,
`--json`, `--tokenize`, or sampling flag.

The teacher can be fetched for training through a private CPU-only Kaggle job;
it installs to `models/bonsai2-27b-mtp-teacher`, never into the runtime head
directory:

```bash
python3 tools/kaggle_bonsai_mtp_job.py \
  --kernel USER/bonsai2-mtp --output cache/kaggle-bonsai-mtp
```

An earlier all-ternary QAT head measured 27.46 tok/s at 74.3% draft acceptance
against the retired int8 head's 28.40 at 80.8% (six prompts, byte-identical
text); the mixed head above superseded it. See
[docs/BONSAI.md](docs/BONSAI.md#tried-and-rejected) for those measurements.

Artifact compression only ever applied to the retired int8 head. The
5,946,648,928-byte GGUF is not compressible in practice: the measured ceiling
is 1.074x and the best measured result 1.29%.

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

The `#[ignore]`d tests exercise the real checkpoint and GPU, so the default
`cargo test` above skips them. They resolve the pinned model and MTP head from
the workspace's `models/` directory, exactly where installation puts them, and
read no environment variables:

```bash
cargo test --workspace --release -- --ignored --test-threads=1
```

`--test-threads=1` is load-bearing. Each test maps and uploads the full 5.5 GB
checkpoint, and concurrent unified-memory and GPU contention purges volatile
prompt-cache state, so cache-source assertions fail on correct code. On a
16 GB machine the same pressure also accumulates across a serial run, which is
why `interleaved_and_restarted_session_snapshots_match_cold_tokens` accepts
either resume tier and
`prompt_checkpoints_match_cold_greedy_for_extension_and_mid_prompt_edit` asserts
a reuse bound rather than an exact count; both still assert token equality
against a cold reference.

## License and sources

The code is dual-licensed under [Apache-2.0](LICENSE-APACHE) or
[MIT](LICENSE-MIT). Model files retain their own licenses. The PTQ1 layout and
kernels derive from [Prism ML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp);
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model sources are
[Prism ML's GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
and the [community MTP head](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP)
by ProCreations, from which the shipped ternary head was distilled.
