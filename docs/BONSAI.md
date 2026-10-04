# Bonsai 2 27B technical reference

`local-ai` runs Prism ML's Ternary Bonsai 2 27B PTQ1 checkpoint directly on
Apple Silicon. This document describes the runtime policy and interfaces. The
top-level [README](../README.md) contains installation instructions.

Measurements use an M4 Pro with 48 GB unified memory and greedy sampling.

## Checkpoint format

The pinned 5,946,648,928-byte GGUF stores ternary linear weights as PTQ1_0:
128 signed trits and one FP16 scale per 28-byte block. Metadata, tokenizer,
embedding, norms, and other tensors remain in their checkpoint types. The GGUF
is memory-mapped and exposed to Metal without copying the complete file.

The runtime checks the architecture, tensor names, shapes, types, byte ranges,
alignment, file size, and SHA256 before inference. `bonsai --export index`
emits the validated metadata and tokenizer index as JSON.

### MTP head

The optional `model_mtp.safetensors` file is a one-layer multi-token prediction
head. Its published BF16 file is 849 MB. At first load, dense matrices are
quantized to symmetric int8 and written to
`~/Library/Caches/local-ai/mtp-head`; later runs validate and memory-map the
425 MB cache. A cached load takes 0.38 s.

Norms and other sensitive vectors retain their required precision. The cache
key binds the source identity and format, so a changed or invalid source is
rebuilt rather than reused.

#### The int8 head artifact

That cache was a runtime cost rather than a shipping format, so the same
transform is also available offline. `local-ai bonsai --export mtp-head=DIR`
writes `DIR/mtp-head-int8-v2.bin`, and `--export mtp-head-zstd=DIR` writes the
same filename with its section region behind one zstd frame. The export starts
no engine and opens no checkpoint: it locates the installed model directory only
to find the head beside it, then transforms the head file alone, so it needs no
GPU, no second copy of the transform, and no memory for the target. The model
path must still be discoverable, since that directory is what says where the
head is. `--export` accepts only `--model`: it is rejected alongside any prompt,
`--prompt-file`, `--json`, `--tokenize`, or sampling flag.

```bash
target/release/local-ai bonsai \
  --model "$PWD/models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --export mtp-head=models/bonsai2-27b-mtp
target/release/local-ai bonsai \
  --model "$PWD/models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --export mtp-head-zstd=models/bonsai2-27b-mtp
```

Discovery finds `models/bonsai2-27b-mtp/mtp-head-int8-v2.bin` beside the pinned
BF16 head, so the directory written above is already the installed one.

The artifact is not a re-quantization. It is the byte-identical product of the
same `transform_sections` a cache miss runs, so both forms describe the same 25
sections with the same names, sizes, and offsets, and inflating one rebuilds the
other byte for byte. Every later load re-derives the payload digest from the
file and compares it against the header, which is what lets an install ship the
artifact and trust it without the 849 MB source. There is therefore no quality
trade-off between the two forms: they hash the same.

| Form | Bytes on disk | `compression` | Install saving |
| --- | ---: | --- | ---: |
| `--export mtp-head=DIR` | 425,263,104 | `stored` | 49.9% |
| `--export mtp-head-zstd=DIR` | 355,837,652 | `zstd` | 58.1% |

Both forms save against the 849,400,392-byte BF16 source; the compressed form is
a further 16.325% below the stored one.

Both report `"payload_bytes":425056256` and the same digest:

```text
payload_sha256 = a98a24e58fcfb711cd2b466d3a32375e428000b8f2b06b677d5839946b20f93b
```

The digest is taken over the 25 logical sections concatenated in artifact order,
skipping the 16 KiB alignment padding, so it identifies the payload rather than
one file layout. The installed filename is the same for both forms, because the
encoding is recorded in the file's own eight-byte magic rather than in its name;
a compressed head sits at the one installed artifact path and is still read as
itself. A section-region zstd frame is decoded before any of it is validated, so
one validator holds both encodings to one standard rather than trusting the
compressed form separately.

The artifact is content-addressed inside the head cache as
`mtp-head-v2-<sha256>.bin`. Discovery prefers it over the BF16 source, which is
then used only as a rebuild source and is not `open()`ed at all when an artifact
is present. Measured on an M2 MacBook Air, a load drops from 5.94 s cold from the
BF16 source to 3.18 s warm from the artifact. End to end on the same machine
with `model_mtp.safetensors` absent, the engine loads the artifact and reports
`mtp.enabled: true` with `head_cache: "hit"`, and answers correctly.

A corrupt, truncated, or renamed artifact is always a hard error naming the file
and the reason, even with a valid BF16 source beside it. There is deliberately
no fallback to the source on this path: a head that cannot be read has no second
source to rebuild from, and decoding without speculation looks like a healthy
install rather than a broken one.

**The cost of the compressed form is resident memory.** A stored head is mapped,
and `from_bytes_no_copy` hands Metal pointers into pages the kernel demand-pages
and reclaims, so it costs no anonymous RAM. An inflated head must exist in full
before a single section can be read, so every load through one holds a
425,263,104-byte anonymous allocation for the life of the process. Peak RSS is
412 MiB stored against 752 MiB zstd, which is why discovery prefers the stored
form and the compressed export stays opt-in.

The compressed export encodes at **zstd level 19**, which is where this payload's
ratio turns over: `--long=27` measures 355,688,546 bytes and level 22 measures
358,063,503, both worse than level 19's 355,361,865. This is a property of the
payload, not of zstd — the HTTP response path deliberately uses level 22, which
is the best ratio on JSON bodies. Widening the window does not help here.

## Commands

```text
local-ai chat [options] <prompt>
  --model PATH       override model discovery
  --max-tokens N     output cap (default 8192)
  --no-thinking      skip the checkpoint's xhigh reasoning (default on)
  --greedy           disable sampling
  --raw              skip the chat template

local-ai serve [options]
  --model PATH       override model discovery
  --host IP          bind address (default 127.0.0.1)
  --port N           TCP and UDP port (default 8080)
  --api-key KEY      require Bearer authentication
  --no-thinking      disable reasoning
  --stall-timeout N  drop a generation whose client stopped reading; seconds,
                     default 30, accepted 10 to 3600

local-ai bonsai [options] <prompt>
  --model PATH       override model discovery
  --max-tokens N     output cap (default 8192)
  --prompt-file PATH read a UTF-8 prompt instead of positional text
  --raw              skip the chat template
  --no-thinking      skip the checkpoint's xhigh reasoning (default on)
  --greedy           disable sampling
  --json             return token IDs and measured timings
  --tokenize         emit prompt token IDs without loading weights
  --export KIND      write an artifact instead of generating; accepts only
                     --model. Kinds: index, mtp-head=DIR, mtp-head-zstd=DIR
  --no-speculation   disable both MTP and suffix lookup for comparison
```

`chat` and `bonsai` stream text to stdout. Startup policy JSON goes to stderr.
`--no-thinking` skips the checkpoint's xhigh reasoning, which is on by default;
there is no flag to turn it back on, since it is already on. `--raw` drops the
chat template that reasoning is asked inside. Use `--` before prompt text that
starts with a hyphen.

`--export` takes one kind, so passing it twice is an error rather than a choice
between two exports. Every kind decodes nothing and takes no prompt. `index`
writes no file and therefore takes no directory; `mtp-head=DIR` and
`mtp-head-zstd=DIR` need only the BF16 head beside the checkpoint, since neither
starts an engine nor opens the GGUF, and each prints the artifact record as JSON.
Both head kinds write `DIR/mtp-head-int8-v2.bin` and overwrite whatever was
there.

## Server API

`serve` provides HTTP/1.1 and HTTP/2 on TCP. If
`~/Library/Application Support/local-ai/tls/cert.pem` and `key.pem` exist, it
also starts HTTP/3 on UDP at the same address and advertises it with `Alt-Svc`.
TLS and HTTP/3 are automatic runtime behavior, not a build option.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | readiness |
| `GET` | `/v1/models` | installed model |
| `POST` | `/v1/chat/completions` | templated chat |
| `POST` | `/v1/completions` | raw completion |

Generation accepts `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, `seed`, and `stream`. `session_id` or
`user` supplies prompt-cache affinity. Chat messages support text and optional
`reasoning_content`; media is rejected. Requests are limited to 2 MiB.

With `stream: true`, responses are `text/event-stream` chunks terminated by
`data: [DONE]`. Chat separates `reasoning_content` from visible `content`.
Non-streamed JSON includes token usage, cache source, timings, and speculation
counters. JSON responses use zstd when `Accept-Encoding` contains `zstd`, at
level 22, chosen for ratio rather than for a cheaper CPU bill. A zstd
dictionary was measured for this path and rejected; see
[Tried and rejected](#tried-and-rejected).

The engine processes one generation at a time through an eight-slot queue.
Submission to a full queue fails immediately. Dropping an `EventStream`, using
its cancellation handle, or disconnecting a streaming client cooperatively
cancels queued or running generation. A reset and a graceful close are both
detected and release the engine promptly — a graceful close was caught after
only 3,544 bytes — so `--stall-timeout` covers the one case neither of them
reports: a client that stops reading while holding the socket open.

`--stall-timeout` bounds how long a response may go without the server being
able to hand it a frame. A client can stop reading its socket without closing
it — a stalled network, a dead consumer, a client that wandered off — and
neither TCP nor axum signals that. Left alone, the send path blocks, the engine
worker blocks inside its emit callback upstream of every cancellation
checkpoint, and the single-flight engine holds its queue slot indefinitely, so
nothing else can be served. When the budget is exceeded the server logs, cancels
that generation, releases the queue slot, and drops the request, leaving the
engine free for the next client. The default is 30 seconds; a value outside
10 to 3600 inclusive, or one that is not a whole number of seconds, is refused
at startup.

One flag governs two different clocks. On a streaming response it is a
consumer-liveness budget, and a small value is safe there because the server
only produces a frame when the engine emits one. On a non-streaming response the
same number is a producer-liveness budget instead, and the engine's own gaps
between events are not small, so the floor comes from a measurement rather than
from taste; see [Tried and rejected](#tried-and-rejected).

The budget starts only after the first event arrives, so it does not bound
time-to-first-token. Nothing is emitted during prefill, the prefill chunk is 128
tokens, and this class of machine prefills at roughly 3.6-4.2 tok/s, so one
chunk is about 35 seconds of silence that is entirely healthy. A budget that
included prefill would abandon ordinary long prompts.

Two limits are worth stating plainly. On streaming, the clock starts late by
construction: a client must first fill the socket buffers before the server feels
backpressure. Measured on an M2, the server absorbed 564,550 bytes (551.3 KiB)
before the stall began, and SSE frames average 194.9 bytes, so roughly 2,900
tokens are generated first. Reclaim time is therefore time to fill the socket
buffers plus the budget, not the budget counted from the request. The buffer
ceiling is autotuned rather than fixed, and 4 MiB is the observed autotune
maximum on that machine. Second, a non-streaming client that vanishes cannot be
detected: the body does not exist until generation finishes, so there is no
disconnect signal to observe, and the budget is producer-liveness only. One such
client was measured burning 706 seconds of engine time before the request
completed.

## Resource policy

All choices and reasons are reported in startup JSON.

- **Model:** explicit path, then the pinned path in the working directory,
  beside the executable, or under `~/Library/Caches/local-ai/models`.
- **Context:** largest value up to 262,144 tokens whose fully grown model state
  fits 90% of Metal's recommended working set.
- **K/V:** F16 unless it cannot reach a useful 32,768-token context, then Q8.
- **Prefill:** fixed 128-token chunks with Metal 4 kernels where supported.
- **Speculation:** suffix lookup is enabled; the discovered MTP head adds
  gated depth-3 drafting. Discovery prefers the int8 head artifact over the BF16
  source, so the 849 MB source is optional and used only to rebuild.
- **Prompt cache:** GPU checkpoint count derives from working-set headroom.
  Disk budget is the smaller of 512 GiB and 25% of free space.
- **Host snapshots:** disabled when the startup write probe measures at least
  500 MiB/s, making the integrity-checked disk tier the first durable tier;
  otherwise capped at one quarter of remaining headroom or 4 GiB.
- **Server:** queue capacity is eight. HTTP/3 is enabled only when its
  certificate and key are discovered.

## Lossless speculation

Speculation changes scheduling, not sampling results. Drafts are checked by
the target model, and only target-approved tokens are committed.

The suffix store finds a matching generated-token suffix and proposes the
known continuation. Draft depth adapts to match length and remaining context.
When the optional head is installed, it drafts up to three tokens. A gate
avoids head work where measured acceptance does not repay its cost.

The target verifies a draft as a row block. Recurrent layers save one initial
state plus compact factors for intermediate rows. On rejection, accepted rows
are committed and recurrent state, target K/V, and MTP state are rolled back to
that exact boundary. Full acceptance needs no restore. This compact rollback
uses 282 MB for 63 drafts rather than 9.9 GB of per-row full checkpoints.

### What the speculation gain is made of

`--no-speculation` disables two independent mechanisms under one flag: the MTP
head, which drafts speculatively ahead, and n-gram suffix lookup
(`ngram_policy`, `min_match: 24`), which proposes the known continuation of a
previously-seen 24-token suffix. Both are honestly called speculation, the flag
disables both by design, and the startup policy reports the two separately, so
an A/B against the flag measures speculation rather than the head alone. What
that A/B cannot show is how the gain divides between them.

Measured on a 16 GB M2 MacBook Air, greedy decoding, `--max-tokens` 160 to 400,
three repetitions per arm taking the median. Every arm produced byte-identical
output, so the comparison is fair.

| Workload | Speculation | `--no-speculation` | Speedup | MTP | n-gram |
| --- | ---: | ---: | ---: | --- | --- |
| Varied prose: planets with one-line descriptions | 3.747 tok/s | 2.983 tok/s | 1.26x, and 1.22x on an independent second run | 75 rounds, 84/106 accepted | 0 rounds |
| Repetitive: counting one to twenty and back | 4.824 tok/s | 2.975 tok/s | 1.62x | 28 rounds, 73 accepted | 0 rounds |
| Verbatim: the same sentence 20 times | 5.935 tok/s | 3.023 tok/s | 1.96x | 17 rounds, 50/50 accepted | 12 rounds, 260/261 accepted |

The n-gram path never engaged on the first two workloads, so their 1.26x and
1.62x are the MTP head on its own; run-to-run spread on the first is about 4%,
which is worth knowing before reading much into a single figure. The 1.96x row
reproduces the headline gain, and it is the only row where n-gram engaged at
all: 260 of 261 proposals accepted for 0.1 ms of total lookup time. Both
mechanisms were live on that row — MTP accepted 50 of 50 there, a higher rate
than on any other workload — so these counters cannot rank the two against each
other, and the split is not measured. What is established is narrower and
still worth stating: the head alone delivers 1.26x on non-repetitive text and
1.62x on lightly repetitive text, and the larger headline figure coincides with
the one workload where suffix reuse switched on.

#### Where the decode time goes

From the 160-token varied-prose run, with speculation on:

| Phase | Seconds | Share |
| --- | ---: | ---: |
| Verification | 39.211 | 89.0% |
| Drafting | 3.390 | 7.7% |
| Commit | 0.549 | 1.2% |
| Other | 0.896 | 2.0% |

Drafting is the speculative phase and it is cheap; checking the draft is 89.0%
of decode. That ordering is what makes the unit costs worth recording.

| Quantity | Value |
| --- | ---: |
| Plain single-token decode | 0.33563 s |
| Verify block | 0.52282 s, covering 2.413 tokens |
| Per batched verify token | 0.21664 s, or 0.6455x a plain decode token |
| Verify block against one plain decode | 1.5577x, for 2.41x the tokens |
| Batching gain on the dominant phase | 1.55x |
| Draft step | 0.04521 s, or 13.5% of a decode step |
| MTP acceptance | 79.2% (84/106) |
| Tokens per round achieved | 2.133 |
| Tokens per round at full depth 3, including the token every round emits anyway | 3.377 |
| Depth-3 opportunity captured | 63% |

A verify block costs 1.5577x a plain decode and returns 2.41x the tokens, so
batching is worth 1.55x on the phase that dominates decode. A draft step costs
13.5% of a decode step, and 2.133 tokens per round against 3.377 at full depth 3
means the gate already collects 63% of what depth 3 could reach. That is also
why a cheaper draft has so little left to win on this path.

Throughput then follows from acceptance and depth rather than from a benchmark
run:

```text
speedup = tokens_per_round x T / (draft + verify)
```

Every input to that formula comes from one run of the varied-prose workload, so
the comparison is like for like: the formula predicts 1.2605x against the
1.2192x that run measured, 3.4% apart. The model is worth more than the single
number it produces — a change to acceptance or to draft depth can be priced
arithmetically instead of benchmarked.

#### Why this engine's speculation pays and a published CUDA result did not

A published community MTP graft of this same 27B ternary trunk, measured on the
1.75 bpw PTQ1 configuration, reported a 1.6% gain, which is to say nothing, and
attributed it to verifying a batch of 3 tokens costing about 3x verifying 1.
That is a per-token ratio near 1.0: no batching gain at all. Here the same ratio
is 0.6455, a 1.55x gain, because a batched verify shares the weight stream across
its positions instead of re-reading it per position. Their own note that the
PTQ1 unpacking was compute-bound is consistent with per-token unpack work
consuming that share.

This reconciles direction and magnitude between two reported numbers on two
different runtimes. The batching explanation is inferred, not measured: nothing
was measured on the other implementation's kernels, so it is the reading that
makes the two sets of numbers compatible rather than a result.

## Prompt cache

Prompt reuse is exact-prefix reuse. Each snapshot binds model identity, K/V
layout, MTP configuration, tokens, recurrent state, and target and head K/V.

1. Purgeable GPU checkpoints provide the fastest restore. The operating system
   may discard them under pressure; residency is checked before use.
2. Host snapshots are purgeable Metal buffers, enabled only when the startup
   storage probe does not select SSD-first. Snapshot staging uses anonymous
   page mappings that return to the OS when dropped.
3. Disk `.bpc` snapshots are versioned and SHA256-checked, written atomically,
   discovered at startup, and trimmed within the automatic disk budget.

The request tail is checkpointed at the penultimate chat token because the
next rendered turn diverges at the assistant opener. One reusable boundary is
also materialized per prefill: a newly observed shared-prefix divergence is
preferred, otherwise the end of the system turn. Boundaries shorter than one
128-token prefill chunk are skipped. Shared boundaries receive eviction
priority over request-tail snapshots and can be reused across sessions.

Measured effects:

| Case | Cached result | Cold result |
| --- | ---: | ---: |
| Follow-up conversation turn | 2.7 s | 25 s |
| Restart and disk restore | 0.69 s | full prefill |
| Shared 6.7K-token prefix | 1.2 s TTFT | 73 s TTFT |

## K/V cache

The target's full-attention layers use F16 keys and values by default. If F16
cannot provide a 32K context within the working-set policy, both use Q8 blocks
with an F16 scale per 32 values. Scores and accumulation remain F32. The MTP
head's one-layer cache always remains F16.

Caches begin at 1,024 tokens, or the selected context when smaller. Capacity
doubles only when a request reaches it, preserving written rows exactly.
Attention workspaces and speculation buffers grow with actual need rather than
the maximum context.

F16 is retained whenever it fits: Q8 prefill is about 1.5–1.9 times slower at
12K–32K and decode is up to 15% slower at 32K.

## Memory

Checkpoint weights remain file-backed and are paged on demand. Startup does
not touch every weight page. A stored MTP int8 head, whether it is the machine
cache or the installed artifact, is memory-mapped and demand-paged too; an
inflated zstd head is the exception, since it must be whole in anonymous memory
before it can be read. K/V, attention workspaces, verify rows, and other
context-dependent buffers are allocated at their minimum useful size and grow
lazily.

The resulting short-request peak footprint is about 1.4 GB. After a prompt of
roughly 8K tokens, idle footprint is about 1.1 GB. These footprint values count
resident runtime memory; the 5.9 GB mapped checkpoint remains part of virtual
address space and its resident pages vary with operating-system pressure.

## Performance

| Workload | Result |
| --- | ---: |
| Plain short-prompt decode | 17.5 tok/s |
| Arithmetic with speculation | about 27 tok/s |
| Code copy-edits with speculation | 40–54 tok/s |
| 4K prefill | about 96 tok/s |
| 128K prefill | 56 tok/s |

Decode spends 97–98% of wall time on the GPU. PTQ1 decode moves 5.65 GB of
weights per token at 116–127 GB/s; trit reconstruction, rather than host gaps,
is the main limit.

## Tried and rejected

- Chunkwise GDN prefill: recurrence is 17.25 ms per 128-row block, only
  1.17–1.29% of block time; perfect removal yields at most 1.013x.
- F16 prefill activations: only 1.04–1.07x projection speedup before conversion,
  below the 1.15x threshold; tile conversion is 1.9–3.5x slower.
- Half-precision PTQ1 decode: exact recurrence is 36–51% slower than FP32.
- DSpark: available Bonsai 2 weights have no usable license; the measured
  perfect-drafter ceiling is 32.8 tok/s and realistic acceptance loses to MTP.
- Eight-row small-batch verify kernel: faster on 17408×5120 but slower on
  5120×17408; whole-model 8/16/24-row verify went from 256/448/643 ms to 275/489/705 ms.
- A lower-precision MTP head as a throughput lever: three independent Apple
  Silicon sources report that keeping the head in BF16 beats a narrower one,
  since dequant overhead dominates latency-bound draft matmuls. That matches
  this engine's cost structure, where drafting is 7.7% of decode time, so the
  ceiling here is arithmetic on that share rather than an experiment: an
  infinitely fast head matmul caps the whole speculation gain at 8.34%, and
  halving it buys 4.00%. Separately, the BF16 and int8 paths produce
  bit-identical GPU weights — the same 425,056,256-byte payload and sha256
  `a98a24e58fcfb711cd2b466d3a32375e428000b8f2b06b677d5839946b20f93b` — so the
  int8 artifact is a load-time and footprint result, 5.94 s to 3.18 s and
  849,400,392 to 355,837,652 bytes, and cannot be a decode-throughput result at
  all.
- Residency sets: 17.23/27.50/48.05 tok/s versus 17.17/27.51/47.98; neutral.
- Untracked hazards: 17.25 versus 17.25 tok/s plain and incorrect speculative tokens.
- Multi-request batching: 25.5/29.5/31.4 aggregate tok/s at 2/3/4 requests,
  versus 31.1/35.4/37.7 for serial requests retaining speculation.
- Weight prewarm: increases resident footprint without improving steady-state inference.
- Q4 K/V: quality and kernel cost did not justify a format below the Q8 fallback.
- Neural Engine: real-size projections execute on GPU and 54 GB FP16 weights do not fit.
- `float4` verify activation loads: 1–5% slower than scalar gathers.
- Tensor Q8 attention: 3.6–7.3 times slower decode and twice as slow prefill.
- Trit lookup tables and 2-bit repacks: none exceed the FP32 digit decoder's 127 GB/s.
- Compressing the int8 head artifact above zstd level 19: level 22 measures
  358,063,503 bytes and `--long=27` measures 355,688,546, both worse than level
  19's 355,361,865 on the same payload. This is a property of the payload, not
  of zstd — the HTTP response path uses level 22 because it is the best ratio
  there — so neither level is presented as universally best.
- A zstd dictionary on the HTTP response path: a 1 MB dictionary at level 9 would
  give +5.6% ratio and roughly 14x less time than the level-22 setting, and it
  was rejected anyway. RFC 8878 section 6 says `application/zstd` payloads
  should not use a dictionary, section 5 records that the specification does not
  define dictionary delivery, and section 7.4 confirms no public registry of
  such dictionaries exists. There is no HTTP mechanism to deliver the dictionary a
  frame's `Dictionary_ID` points at, so a stock client without it — `curl`,
  `httpx`, `requests`, `undici` — fails hard with `Dictionary mismatch` and
  decodes zero bytes. Verified. Dictionaries remain valid where both ends are
  yours, such as internal IPC or a bundled SDK.
- Compressing the 5,946,648,928-byte PTQ1 GGUF: the measured ceiling is 1.074x
  and the best measured result 1.29%, so the checkpoint is not a compression
  target. Only the int8 head is, which is why it ships in two encodings.
- A rounder `--stall-timeout` floor: the same number bounds a consumer-liveness
  budget on a streaming response and a producer-liveness budget on a
  non-streaming one, and the latter is set by the engine's own inter-event gaps.
  A speculative round measured 3.076 s, p95 0.728 s, on a six-token prompt on an
  M2, so a budget below that abandons healthy non-streaming requests. The floor
  is 10 rather than a rounder number because it sits an order of magnitude above
  the worst gap actually observed.

## Verification

Run checks that do not require model files:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked --release
uvx ruff check tools/
python3 -m unittest discover -s tools -p 'test_*.py'
```

Create the checked index used by tokenizer tests:

```bash
target/release/local-ai bonsai \
  --model "$PWD/models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --export index > cache/bonsai-index.json
```

Ignored real-model tests use absolute paths supplied by these variables:

```bash
BONSAI_GGUF=$PWD/models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
BONSAI_MTP_HEAD=$PWD/models/bonsai2-27b-mtp/model_mtp.safetensors \
BONSAI_INDEX=$PWD/cache/bonsai-index.json \
  cargo test -p local-engine --lib --release -- \
  --include-ignored --test-threads 1 bonsai
```

Run GPU checks serially when including ignored tests:

```bash
cargo test -p local-metal --release -- --include-ignored --test-threads 1
```

The engine targets text inference on Apple Silicon. Model quality relative to
the unquantized checkpoint and broad application benchmarks are outside these
runtime measurements.
