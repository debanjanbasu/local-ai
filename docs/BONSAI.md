# Bonsai 2 27B technical reference

`local-ai` runs Prism ML's Ternary Bonsai 2 27B PTQ1 checkpoint directly on
Apple Silicon. This document describes the runtime policy and interfaces. The
top-level [README](../README.md) contains installation instructions.

Measurements use an M4 Pro with 48 GB unified memory and greedy sampling
unless a section names another machine; several server-timeout, compression
and speculation-decomposition figures come from a 16 GB M2 MacBook Air. A
kernel-only timing, an end-to-end throughput and a quality check are not
interchangeable; read each figure in its section's context.

## Checkpoint format

The pinned 5,946,648,928-byte GGUF stores ternary linear weights as PTQ1_0:
128 signed trits and one FP16 scale per 28-byte block. Metadata, tokenizer,
embedding, norms, and other tensors remain in their checkpoint types. The GGUF
is memory-mapped and exposed to Metal without copying the complete file.

The runtime checks the architecture, tensor names, shapes, types, byte ranges,
alignment, file size, and SHA256 before inference. `bonsai --export index`
emits the validated metadata and tokenizer index as JSON.

### MTP head

The required MTP head is a one-layer multi-token prediction head (one dense
Qwen3.5 full-attention decoder layer). The shipped head is **mixed
precision, trained for it**, the way Bonsai keeps a few tensors above ternary.
All nine matrices live in the target's Hadamard-rotated basis; `q_proj`,
`gate_proj`, `up_proj` and `down_proj` are exactly ternary with one F16 scale
per 128 columns, stored as the target's own `PTQ1_0` blocks and multiplied by
its kernels, while the sensitive `fc` (both halves), `k_proj`, `v_proj` and
`o_proj` are per-row int8 (see [Mixed-precision heads](#mixed-precision-heads)).
The installed file is `models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin`,
166,969,344 bytes, mapped into Metal with no copy: 258 MB less resident memory
than an all-int8 head at the same decode speed. Discovery looks for exactly this file beside the pinned checkpoint
directory; when it is absent, startup stops with an error naming it.

```text
payload_bytes  = 166799360
payload_sha256 = fb05507f87f54432c782b0fcb35e5cf1da1b4997610ab35a3d98502e536351cd
```

The digest is taken over the 21 logical sections (nine matrices, each one
`.ptq1` section or `.int8` plus `.row_scales`, then seven folded F32 norms) concatenated in artifact order, skipping the 16 KiB
alignment padding, so it identifies the payload rather than one file layout.
The header page names every section and records its size, because sizes alone
cannot bind them (`k_proj` and `v_proj` are both `[1024, 5120]`). Every load
re-derives the digest from the file and compares it with the header. A corrupt,
truncated, or renamed artifact is always a hard error naming the file and the
reason; there is deliberately no fallback, because decoding without
speculation looks like a healthy install rather than a broken one.

#### Provenance and export

The head was distilled from the community BF16 head
[`ProCreations/Ternary-Bonsai-2-27B-MTP`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP)
(849 MB, `model_mtp.safetensors`), which serves only as the teacher. The
recipe is ours: `mtp-capture` records the target's hidden states and logits;
`tools/mtp_train/train_ternary.py` runs TWN g128 QAT in the rotated basis with
a straight-through estimator and KL to the teacher's draft chains; and
`tools/mtp_train/convert.py to-ternary` writes `model_mtp_ternary.safetensors`:
for each of the nine matrices (`mtp.fc.weight.embedding`,
`mtp.fc.weight.hidden`, then the layer's q/k/v/o and gate/up/down) an I8
`<name>.codes` `[rows, cols]` in {-1, 0, 1} and an F16 `<name>.scales`
`[rows, cols/128]`, meaning `y = Σ codes·scale·(R x)` with `R` the target's
forward signed Hadamard for `cols` (5120, 6144 or 17408), plus the seven BF16
zero-centered norms. `tools/kaggle_bonsai_mtp_job.py` fetches the pinned teacher
into `models/bonsai2-27b-mtp-teacher` for training.

```bash
target/release/local-ai bonsai --export mtp-head=DIR
cp DIR/mtp-head-ptq1-v1.bin models/bonsai2-27b-mtp/
```

The export starts no engine and opens no checkpoint. It packs codes and scales
into `PTQ1_0` blocks bit-identical to Prism's reference encoder, folds the norms
to `1 + w`, and writes `DIR/mtp-head-ptq1-v1.bin`. `--export` takes no other
options: it is rejected alongside any prompt, `--prompt-file`, `--json`,
`--tokenize`, or sampling flag.

#### Mixed-precision heads

Like Bonsai itself, a head may keep a few sensitive matrices at higher
precision. Any of the nine matrices may instead be given in the source as an I8
`<name>.int8` `[rows, cols]` in [-127, 127] and an F32 `<name>.row_scales`
`[rows]`, meaning `y = Σ int8·row_scale·(R x)` in the same rotated basis; the
export detects the format per matrix (holding both pairs for one matrix is an
error) and prints the int8 ones as `int8_matrices`. The artifact is the same
file and container version: a ternary matrix is one `<name>.ptq1` section, an
int8 one is `<name>.int8` then `<name>.row_scales`, and the section names are
what the loader reads each matrix's format from. An all-ternary head is
therefore byte-identical to what earlier builds wrote (the installed head
re-exports bit for bit); a build without int8 support refuses a mixed head by
its section count or names.

Int8 matrices multiply the same rotated activations through their own kernels
(`shaders/bonsai_int8.metal`): one and two activation rows use vector kernels
reducing four weight rows per SIMD group with 16-byte weight loads (236-248
GB/s on an M4 Pro, 360 us for 17408x5120 against 134 us in `PTQ1_0`); 3 to 8
rows use one F32 simdgroup-matrix dispatch whose cost is flat in the row count
(201-216 GB/s, 414 us for 17408x5120); larger blocks, such as K/V-only
ingestion of prefill chunks, take even chunks of at most eight rows. Single-row groups are fused per format: the
`PTQ1_0` members of q/k/v share one concatenated matvec and the int8 members
another, and gate/up use one fused SwiGLU when both share a format; a mixed
gate/up pair runs as two projections and the elementwise SwiGLU.

#### Fine-tuning the shipped head on code and tool calls (not adopted)

Goal: raise draft acceptance on coding-agent output (code, tool-call
arguments, the reasoning around them) by training the installed head further
on the target's own greedy transcripts. Prior attempts above (corpus
fine-tuning, general self-rollouts) left live acceptance flat or lower, and
so did this one: the candidate is not adopted.

Run (2026-10-22/23): 120 rollouts (32 code, 32 context, 32 agent, 24 API
calls; 146K generated tokens) over this repository and the open-source
`dioxus`, `tailwindcss` and `veilid` checkouts, captured as 104 documents
and 321,318 positions (16 agent transcripts ending on an unanswered tool call
were skipped), then two epochs of the recipe below from the installed head
on a private Kaggle T4 x2 job (244 steps, 27 minutes). On the 2% held-out
windows draft accuracy rose at every depth: 0.660 → 0.692, 0.499 → 0.528
and 0.432 → 0.451 (best at step 100: 0.692, 0.535, 0.477). Live, it did not:

| Head | Tasks ok | MTP accepted/proposed (code, tool, prose) | Accepted per round | Decode tok/s, geomean |
| --- | --- | --- | ---: | ---: |
| installed (`83cc7227…`) | 16/20 | 73.6% (71.5%, 82.7%, 81.9%) | 1.07 | 40.98 |
| candidate (`dbadd150…`) | 15/20 | 73.9% (72.1%, 83.3%, 80.9%) | 1.04 | 40.92 |

`tools/mtp_code_bench.py ab --rounds 2` (ABBA), the fixed-build server, M4
Pro; unrelated builds shared the CPU during the run, alike for both heads. Outputs were identical on 17 of 20 tasks; of the three that differ
(verify near-ties), one, `sql-top-earner`, fails its test with the candidate.
The 0.3-point acceptance gain is within run-to-run noise and buys no decode
speed, so the installed head stays. The candidate is kept beside it as
`models/bonsai2-27b-mtp/mtp-head-ptq1-code-cand.bin` (never installed over
it; the dataset and kernel are private on Kaggle). As with the earlier
attempts, held-out accuracy on captured windows did not predict live
acceptance.

What works:

- `convert.py from-head HEAD.bin DIR` unpacks the installed head into
  `DIR/model_mtp_ternary.safetensors`; `--export mtp-head=DIR` re-packs it
  byte for byte (`83cc7227…`, payload `fb05507f…`). `convert.py to-qat`
  turns that into a `train_ternary.py --init` checkpoint whose step-0 export
  is again byte-identical, and `from-ternary` gives a dense copy to use as
  `--teacher`. The community teacher is not needed (Hugging Face is
  unreachable from this network; the Kaggle job fetches it if wanted).
- `tools/mtp_train/rollouts.py` collects greedy transcripts from a running
  `serve` (code writing in eight languages, real source excerpts from
  `--repo` checkouts, multi-turn agents over those repositories with
  read-only tools and in-memory writes, general JSON tool calls), never
  reusing the bench tasks. `mtp-capture capture` now renders transcript
  lines (`tools`, replayed `tool_calls`, `reasoning_content`, `thinking`)
  exactly as the server does; `mtp-capture render` prints them without a
  model.
- `train_ternary.py --min-match 0.9` keeps only windows that follow the
  target's argmax (the generated spans), not prompts or tool output.
- `tools/mtp_code_bench.py ab` A/Bs heads (ABBA rounds, each served from a
  scratch root whose `models/` holds only that head, so the installed file is
  never touched) on 9 code tasks run against tests, 9 agent tool-call turns
  (edits must apply and pass their test) and 2 prose controls; `summary`
  prints acceptance, accepted per round, decode tok/s, first-token and total
  latency, task checks and output identity.

To resume (GPU steps under `lockf -k /tmp/local-ai-gpu.lock`; `W` the
checkout, `R=$W/.scratch/mtp`):

```bash
target/release/mtp-capture signs $R/signs && target/release/mtp-capture tables $R/tables
python tools/mtp_train/convert.py from-head models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin $R/v1-src
python tools/mtp_train/convert.py to-qat $R/v1-src/model_mtp_ternary.safetensors $R/v1-qat.safetensors
python tools/mtp_train/convert.py from-ternary $R/v1-src/model_mtp_ternary.safetensors $R/v1-dense.safetensors --signs $R/signs
# 1. baseline: tools/mtp_code_bench.py ab --head installed=models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin --rounds 1 --out ...
# 2. rollouts: serve, then rollouts.py --out $R/rollouts.jsonl --repo . --repo <open-source checkouts>
# 3. target/release/mtp-capture capture --out $R/capture --max-doc-tokens 8192 $R/rollouts.jsonl
# 4. train_ternary.py --capture $R/capture --tables $R/tables --teacher $R/v1-dense.safetensors \
#      --signs $R/signs --init $R/v1-qat.safetensors --min-match 0.9 --kl-weight 1 --target-weight 1 \
#      --epochs 2 --batch 4 --grad-accum 2 --lr 3e-5   (Kaggle T4 approved by the user; private only)
# 5. convert.py to-ternary qat-final.safetensors CAND --signs $R/signs; local-ai bonsai --export mtp-head=CAND
# 6. mtp_code_bench.py ab --head installed=... --head cand=CAND/mtp-head-ptq1-v1.bin --rounds 2
```

Adopting a winner means a new release asset and pin in
`tools/fetch_mtp_head.py` plus the digests at the top of this section, which
needs approval to publish.

### DFlash2 drafter

The required KFlash drafter (Kapitsa's
[KFlash](https://github.com/nace-ai/kapitsa/blob/main/docs/kflash.md)) is
z-lab's `DFlash2DraftModel`. The shipped weights are ProCreations'
[`Ternary-Bonsai-2-27B-DFlash2`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2)
at revision `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`: z-lab's
[`Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
trained further against the frozen Ternary Bonsai 2 27B target (its PQ2_0
packing rather than this PTQ1 GGUF), with the same 81 tensors and
configuration. Five Qwen3 layers (32 query and 8 K/V heads of 128,
FFN 17408), each attention and MLP wrapped in a grouped dynamic convolution
(groups of 16 channels, kernel 2), draft an 8-row block in one pass: the
anchor token and seven mask tokens (`mask_token_id` 248070) attend each
other bidirectionally and a 2048-token sliding window of context. The
context is the target's residual stream at layers 5, 19, 33, 47 and 61
(standard Qwen basis, no unrotation), projected by `fc` (25600 to 5120) and
`hidden_norm` into each layer's K/V. The seven draft rows go through the
target's own `PTQ1_0` output matrix (forward-rotated first, as the target's
hidden is) to the top 16 candidates per position, and a rank-256 candidate
selector scores each candidate against each candidate of the position
before it (the anchor for the first). Those scores make a first-order
lattice; a best-first tree of up to 8 rows over it, scored with per-position
calibrated path probabilities, is what the target verifies. The drafter has
1.797B dense parameters plus 127M selector-codebook rows, which are only
gathered, and shares the target's embedding and output matrix.

```text
file           = models/bonsai2-27b-dflash2/dflash2-drafter-int8-v1.bin
file_bytes     = 2054275072   (SHA256 f43f846c60fc3f83c18ab772006d55f3cd6287944b4989fe4ac8d774c7a955e8)
payload_bytes  = 2053443584
payload_sha256 = 1b5dc3551406e677ec9f4321f8f5c20f7310eaae0cede95792ceaa2375dfc02a
source         = ProCreations/Ternary-Bonsai-2-27B-DFlash2 @ 4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b
                 model.safetensors, 3848817896 bytes BF16, SHA256 708e141bf4a33b09620c4f86263f1c7edcec7882f5dd39448e83e0c0defdc230
```

It replaced the first pin, `incoai/Qwen3.8-27B-DFlash2` at revision
`015e795` (payload SHA256 `768b9f89…614a`, source SHA256 `67fc76d6…b65c`),
after a selection run over published DFlash2 fine-tunes: regenerating every
lattice of the seven trace prompts and running `sim.py`'s leave-one-out
calibrated 8-row tree, it yields 4.97 tokens per cycle against 4.68
(bootstrap ratio 1.059, 95% CI 1.048–1.069) and 4.65 against 4.33 as a
7-draft chain, at or above the first pin on every prompt and most on prose
and essays, where trees were weakest. The measurements below that name no
drafter were taken with the first pin; the architecture, kernels and costs
are the same for both.

The artifact is the MTP head's container under its own magic (`DFLSH2`) and
a 16 KiB header page: 128 named, page-aligned sections (every dense matrix as
`<name>.int8` and `<name>.row_scales`, norms and convolution base kernels as
`<name>.f32`, the two selector codebooks as `<name>.bf16`) and the payload
digest, which every load re-derives. `bonsai --export dflash-head=DIR`
writes it from `DIR/model.safetensors` and `DIR/config.json` of any
`DFlash2DraftModel` with this configuration (BF16, F16 or F32; the codebooks
may carry a `.weight` suffix), so fine-tuned drafters convert the same way.
Discovery looks for exactly this file; a missing or corrupt file is a hard
error that stops startup.

#### Drafter precision

Per-row int8 (`row_scale = absmax / 127`) through the existing int8 kernels.
Output is lossless at any drafter precision, since the target verifies every
token, so the choice is tokens per second. Acceptance of the formats, each
regenerating every lattice of the seven trace prompts (3,047 anchors) and
running `sim.py`'s leave-one-out calibrated 8-row tree, tokens per cycle,
geomean:

| Drafter weights | Tokens per cycle | Top-1 = fp32 | Top-16 overlap |
| --- | ---: | ---: | ---: |
| fp32 reference (`drafter_lib.py`) | 4.685 | | |
| int8 per row, Metal (shipped) | 4.693 | 98.97% | 98.25% |
| 4-bit affine, groups of 64 (CPU) | 4.665 | 92.5% | 88.6% |
| ternary, TWN per 128 columns (CPU) | 1.436 | 7.1% | 15.7% |

Ternary without training destroys the drafter, so the `PTQ1_0` kernels are
out. 4-bit costs 0.6% of the tokens per cycle and would halve the bytes, but
bytes are not what an 8-row pass is short of: the five layers' int8
projections take 8.33 ms at 8 rows against 6.86 ms at one row (243 GB/s, the
int8 bandwidth floor), and the 8-row simdgroup-matrix kernel is limited by
its F32 multiplies, not by bytes: 26.6 GFLOP at 3.2 TFLOP/s, and on a
cache-resident 2048x5120 matrix, where one row streams at 420 GB/s, it still
tops out at 3.55 TFLOP/s (3.52 on 17408x5120 from memory), the limit
`bonsai_int8.metal` documents. A 4-bit kernel built the same way keeps those
multiplies, so it could save at most 0.7 ms of a 12.6 ms pass (under 1% of a
cycle), against a new kernel and the 0.6% acceptance loss.

#### Drafting pass

Measured on the M4 Pro, holding the GPU lock, over the 3,047 trace anchors
(`drafter_lattices_on_captures`): a drafting pass (embedding gather, five
layers, output head, top 16, lattice and 8-row tree) takes **12.58 ms** of GPU
time (median), and appending one committed position of taps to the context
**0.83 ms**. With synthetic taps, a pass costs 12.16 ms at 60 positions of
context, 12.44 ms at 500 and 13.23 ms at the full window; appending a
2,048-row prompt costs 250 ms once (the int8 kernels take it in 8-row
chunks). Of the pass, the int8 projections are 8.3 ms, the output head over
seven rows 3.3 ms, top 16 0.14 ms, and block attention 0.03 ms per layer at
60 context positions, 0.22 ms at the full window.

Parity against `drafter_lib.py` (fp32, itself within 6.5e-6 of z-lab's class)
on the captured Bonsai hidden states, every anchor of the seven prompts:

| Prompt | Anchors | Top-1 equal | Top-16 overlap | Same top-16 set | Selector chain equal | Logit MAE | Edge MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose | 209 | 98.2% | 98.0% | 69.2% | 87.6% | 0.078 | 0.034 |
| explanation | 511 | 98.9% | 98.4% | 74.6% | 91.8% | 0.112 | 0.038 |
| arithmetic | 511 | 99.2% | 98.3% | 73.8% | 95.9% | 0.090 | 0.036 |
| code | 511 | 99.3% | 98.2% | 72.9% | 96.7% | 0.081 | 0.035 |
| essay | 511 | 97.9% | 98.2% | 72.6% | 86.5% | 0.098 | 0.030 |
| thinking | 283 | 99.8% | 98.7% | 79.1% | 96.5% | 0.091 | 0.044 |
| agent | 511 | 99.6% | 98.0% | 69.3% | 97.7% | 0.110 | 0.041 |

Logit errors are in logits (the top candidate's is about 20); selector edge
errors are against a mean magnitude of 1.2-2.5. The candidates that differ
are the 16th-place near-ties. Every one of the 3,047 GPU trees equals
`sim.py`'s `build_tree` on the same lattice, and the acceptance simulation on
the Metal lattices gives 4.693 tokens per 8-row tree cycle against 4.685 for
the fp32 lattices (report: 4.68):

| Prompt | Metal, calibrated tree | fp32, calibrated tree | Metal, chain of 7 |
| --- | ---: | ---: | ---: |
| prose | 3.542 | 3.542 | 2.986 |
| explanation | 4.024 | 4.088 | 3.476 |
| arithmetic | 5.678 | 5.678 | 5.379 |
| code | 5.742 | 5.678 | 5.323 |
| essay | 2.594 | 2.621 | 2.222 |
| thinking | 6.021 | 6.021 | 6.289 |
| agent | 6.905 | 6.724 | 6.724 |
| geomean | 4.693 | 4.685 | 4.316 |

#### Interface

`local-engine/src/bonsai_dflash.rs` documents the contract the verify side
integrates against. In short: `DflashMode::resolve` finds the drafter or fails;
`DflashDrafter::append_context` takes `[n, 5, 5120]` F16 or F32 taps from a
Metal buffer into a 2,112-slot F16 K/V ring per layer (the 2,047-position
window plus 64 positions of `truncate_context` rollback); `draft` encodes
the pass for an anchor at position `P` (host token or a `u32` the target
wrote on the GPU) onto the caller's `CommandBatch`, leaving the
`DflashLattice` on the GPU; `build_tree(max_rows)` encodes the tree kernel
behind it, writing `DflashTree`: per row its token, parent, depth (position
`P + depth`), lattice slot, ancestor mask, conditional log q and calibrated
path log p, plus `E(n)` for `n = 1..=8`. The first `n` rows are the best
`n`-row tree for every `n`. `Calibrator` turns each verified cycle into
records (every node whose parent was accepted), refits `p = σ(α_d + β_d ·
logit q)` every 16 cycles over the last 2,048 records per position, starting
from `p = q`, and persists the fit to
`~/Library/Caches/local-ai/dflash/<model>-<drafter>-<chip>.json`.
`draft_gpu_time` reports the measured pass time.

## Commands

```text
local-ai chat [options] <prompt>
  --max-tokens N     output cap (default 8192)
  --greedy           disable sampling
  --raw              skip the chat template (and so reasoning)

local-ai serve [options]
  --host IP          bind address (default 127.0.0.1)
  --port N           TCP and UDP port (default 8080)
  --api-key KEY      require Bearer authentication
  --allow-origin ORIGIN  admit browser requests from ORIGIN while
                     authentication is off; repeatable, * for every origin,
                     null for file: pages (see docs/ADMIN.md#browsers)
  --allow-host NAME  answer requests addressed to NAME besides IP addresses,
                     localhost and *.local or *.internal names; repeatable
                     (DNS rebinding protection, see docs/ADMIN.md#browsers)
  --stall-timeout N  drop a generation or download whose client stopped
                     reading, or a request body that stopped arriving;
                     seconds, default 30, accepted 10 to 3600
  --response-store DIR  keep Responses API responses in DIR (owner-only,
                     unencrypted) until deleted; default: the user's data
                     directory, pruned of responses older than 30 days
  --reasoning-key FILE  owner-only AES-256-GCM key for
                     reasoning.encrypted_content and compaction, created if
                     absent; default: the user's data directory

local-ai bonsai [options] <prompt>
  --max-tokens N     output cap (default 8192)
  --prompt-file PATH read a UTF-8 prompt instead of positional text
  --raw              skip the chat template (and so reasoning)
  --greedy           disable sampling
  --json             return token IDs and measured timings
  --tokenize         emit prompt token IDs without loading weights
  --export KIND      write an artifact instead of generating; takes no other
                     options. Kinds: index, mtp-head=DIR, dflash-head=DIR
```

There is no model flag, and no flag turns off or tunes any part of the
[resource policy](#resource-policy): speculation, thinking, the response
store and every model run as they ship. The engine runs one pinned checkpoint and discovers it
under `./models`, beside the executable, or under
`~/Library/Caches/local-ai/models`, and nothing at runtime reads an
environment variable other than `HOME`, plus `TMPDIR` where `serve` places
its private [temporary background store](#temporary-background-responses)
through the system temporary directory.

`chat` and `bonsai` stream text to stdout. Startup policy JSON goes to stderr.
The checkpoint's xhigh reasoning is always on, and no flag turns it off (server
requests choose their own effort). `--raw` drops the chat template that
reasoning is asked inside. Use `--` before prompt text that
starts with a hyphen.

`--export` takes one kind, so passing it twice is an error rather than a choice
between two exports. Every kind decodes nothing and takes no prompt. `index`
writes no file and therefore takes no directory; `mtp-head=DIR` needs only
`DIR/model_mtp_ternary.safetensors`, starts no engine, opens no GGUF, writes
`DIR/mtp-head-ptq1-v1.bin` (overwriting whatever was there), and prints the
artifact record as JSON. `dflash-head=DIR` needs `DIR/model.safetensors` and
`DIR/config.json` of a DFlash2 drafter, likewise starts no engine and opens no
GGUF, and writes `DIR/dflash2-drafter-int8-v1.bin`
([DFlash2 drafter](#dflash2-drafter)).

## Server API

`serve` provides HTTP/1.1 and HTTP/2 on TCP. If
`~/Library/Application Support/local-ai/tls/cert.pem` and `key.pem` exist, it
also starts HTTP/3 on UDP at the same address and advertises it with `Alt-Svc`.
TLS and HTTP/3 are automatic runtime behavior, not a build option. An HTTP/3
request's header block is limited to 64 KiB; a response body that fails
partway (speech whose synthesis fails mid-stream) resets its stream with
`H3_INTERNAL_ERROR` instead of finishing it, so a client never takes a
truncated body for a complete one; and a client whose QUIC connection closes
cancels its request even while nothing is being written to it.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | readiness |
| `GET` | `/v1/models` | the loaded Bonsai model, `ternary-bonsai-2-27b`, with its admitted `context_length`/`max_model_len`, then every other served model (`embeddinggemma-2`, `embeddinggemma-2-moderation-1`, `kokoro-82m`, `whisper-large-v3`) |
| `GET`, `DELETE` | `/v1/models/{model}` | one served model by id or alias ([model ids](#model-ids)); `DELETE` is for fine-tuned models, so a served base model is a 403 and any other name a 404 |
| `POST` | `/v1/chat/completions` | templated text and function calling |
| `POST` | `/v1/responses` | text and function calling; stored by default |
| `POST` | `/v1/responses/input_tokens` | exact templated input-token count, no generation |
| `POST` | `/v1/responses/compact` | compact a window with the loaded model ([compaction](#compaction-and-reasoning-summaries)) |
| `GET`, `DELETE` | `/v1/responses/{id}` | retrieve or delete a stored or temporary background response, or (`GET ?stream=true`) resume a streamed background response |
| `GET` | `/v1/responses/{id}/input_items` | paginated input items of a stored or temporary background response |
| `POST` | `/v1/responses/{id}/cancel` | cancel a background response |
| `POST`, `GET`, `DELETE` | `/v1/conversations[/{id}[/items[/{item_id}]]]` | durable conversations and items ([native services](#native-services)) |
| `POST`, `GET`, `DELETE` | `/v1/files[/{id}[/content]]` | streamed multipart files |
| `POST` | `/v1/uploads[/{id}/parts\|complete\|cancel]` | multipart uploads with optional MD5 |
| `POST`, `GET` | `/v1/batches[/{id}[/cancel]]` | batches of Chat Completions or Responses requests on the loaded model |
| `POST`, `GET`, `DELETE` | `/v1/vector_stores[/{id}[/files\|file_batches\|search…]]` | vector stores, BM25 or (with the embedding model) embedding-ranked ([native services](#native-services)) |
| `POST` | `/v1/embeddings` | EmbeddingGemma 2 text embeddings (installed model only; [EMBEDDINGS.md](EMBEDDINGS.md)) |
| `POST` | `/v1/moderations` | calibrated moderation heads on EmbeddingGemma 2 (installed heads only; [MODERATION.md](MODERATION.md)) |
| `POST` | `/v1/audio/speech` | Kokoro-82M text to speech (installed model only; [SPEECH.md](SPEECH.md)) |
| `POST` | `/v1/audio/transcriptions`, `/v1/audio/translations` | Whisper large-v3 speech recognition and translation to English (installed model only; [TRANSCRIPTION.md](TRANSCRIPTION.md)) |
| `GET`, `POST`, `DELETE` | `/v1/organization/*` | Administration API: admin keys, projects, service accounts, audit logs, usage (admin key only; [ADMIN.md](ADMIN.md)) |
| `POST` | `/v1/decisions` | calibrated, scope-gated decisions (from the required `models/decisions-v2`) |

### Model ids

Each served model has one stable id, and every route reads `model` from the
same table:

| Model | Id | Also accepted | Routes |
| --- | --- | --- | --- |
| Ternary Bonsai 2 27B | `ternary-bonsai-2-27b` | none | Chat Completions, Responses, `input_tokens`, `compact`, decisions, batches |
| EmbeddingGemma 2 | `embeddinggemma-2` | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` | `/v1/embeddings` |
| moderation heads | `embeddinggemma-2-moderation-1` | `omni-moderation-latest`, `omni-moderation-2024-09-26`, `text-moderation-latest`, `text-moderation-stable` | `/v1/moderations` (`model` may be omitted) |
| Kokoro-82M | `kokoro-82m` | `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` | `/v1/audio/speech` |
| Whisper large-v3 | `whisper-large-v3` | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` | `/v1/audio/transcriptions`, `/v1/audio/translations` |

The generation routes accept an omitted `model` or any name that is not
another local model's, since many clients send a configured name the server
cannot know; the response always names `ternary-bonsai-2-27b`. Naming another
local model there (for example `embeddinggemma-2` on Chat Completions) is a
400 that names the right route. Batch lines follow the same rule (each line
must name a model, all lines the same one, and the batch object reports it as
given); a stored line naming another local model fails without generating.
`GET /v1/models/{model}` answers an installed model by id or alias (an alias
answers with the model's own id) and 404 otherwise.

### Supported OpenAI APIs

The server implements only current OpenAI APIs, as the pinned OpenAPI
specification defines them: Chat Completions, Responses (with
`input_tokens`, `compact`, retrieval, cancellation and input items), Models,
Conversations, Files, Uploads, Batches, Vector Stores, Decisions, the
Administration API, Embeddings, Moderations, Audio speech, and Audio
transcriptions and translations.
`/health` is this server's own.
Only the current specification is served:

- The legacy Completions API, `POST /v1/completions`, is a 404 like any
  unknown route, and a batch cannot target it.
- Of the request members the specification marks deprecated, only Chat
  `max_tokens` is honoured, because opencode's AI SDK sends it on every
  request: it is `max_completion_tokens`, which wins when both are set. The
  rest (Chat `functions`, `function_call` and the `function` history role,
  `user`, `reasoning.generate_summary`, `prompt_cache_retention`, this
  server's former `session_id` alias) are not supported: as members they are
  ignored like any unknown member (use `tools`, `tool_choice`,
  `reasoning.summary` and `prompt_cache_key`), and a `function` message is
  an unsupported role. Codex, Oh My Pi and opencode send none of them.
- Responses `truncation` is current: `auto` or `disabled`, on
  `input_tokens` too; the engine never drops input, so an over-long request
  fails either way, and a Response reports `disabled`. `seed` (any JSON
  integer on Chat; a negative one keeps its 64 bits) is a sampling
  extension, like `top_k`.

### Generation requests

Generation accepts `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, `seed` and `stream` on both APIs
(`top_k`, `min_p`, the penalties and the Responses `seed` are extensions),
and vLLM's multiplicative `repetition_penalty`, which replaces the one
`frequency_penalty` maps to. Chat also takes vLLM's and llama.cpp's
`chat_template_kwargs`, of which `enable_thinking: false` turns reasoning off
(other keys are ignored). Both APIs ignore members they do not know, at the
top level and inside the objects they read (`reasoning`, `text`,
`stream_options`), so a vendor field such as `extra_body` or a newer OpenAI
member does not fail the request; a malformed value of a member they do read
is still a 400, and so is an option they know but cannot honour. Chat limits
output with `max_completion_tokens` (or `max_tokens`), Responses with
`max_output_tokens`. Cache affinity is `prompt_cache_key`. Chat Completions
and Responses bodies are limited to 64 MiB, since they may carry base64
images ([image input](#image-input)); other JSON bodies to 2 MiB (multipart
file and upload-part bodies have their own streamed limits). Audio input is
rejected.

Chat `stop` (a string or up to four strings) ends the visible answer at the
first match: the content stops just before the sequence and `finish_reason`
is `stop` (`tool_calls` when calls were made before it). As on OpenAI, only
answer text is matched. Reasoning is never matched, so a sequence the model
thinks about does not end its turn, and neither are tool-call markup and
arguments, so a sequence inside a call's JSON cannot corrupt it; text before,
between and after calls is. A call made before the match stands, and a match
before a call means it is never made. Streaming holds back text that could
still be the start of a sequence until it cannot, so a partial match is never
sent. A JSON answer under `response_format` is cut short like any other text.
Responses has no `stop`; the engine supports it for any request, so accepting
`stop` there as a vendor extension would be a few lines of parsing, but it is
not offered.

Both APIs report token log-probabilities in OpenAI's shape (`token`,
`logprob`, `bytes`, `top_logprobs`), for harnesses that verify or route on
confidence. Chat takes `logprobs: true` with `top_logprobs` (0 to 20; it
needs `logprobs`) and reports `choices[].logprobs.content`, and on each
streamed content chunk the tokens behind it. Responses takes `include:
["message.output_text.logprobs"]` with `top_logprobs` (0 to 20; a
`top_logprobs` above 0 alone also asks) and reports `output_text.logprobs`
and each `response.output_text.delta`'s tokens. The probabilities are the
model's raw distribution, its logits before temperature, penalties,
`top_k`/`top_p`/`min_p` and any response-format or tool grammar mask (vLLM's
`raw_logprobs`), so they compare across sampling settings, and the token
sampling chose need not be the most likely one. Only answer tokens are
reported: reasoning, `</think>`, tool-call markup and arguments, the
whitespace the answer drops around them and text from a stop sequence on
are not. Speculation never shows: every token is reported from the target
model's own row that selected it (plain decode, MTP or lookup verification,
batched decode), never from a draft. A fused Metal kernel computes each such
row's log-sum-exp and best logits in the block's own command buffer, only
for requests that ask; see [Performance](#log-probabilities) for its cost.
A Response finished after a server restart (from its journal) reports empty
`logprobs`.

`verbosity` (Chat) and `text.verbosity` (Responses) accept `low`, `medium`
(the default) and `high`. The model has no verbosity control of its own, so
`low` and `high` add a short instruction at the end of the system turn, after
the tools and the client's instructions (so the cached prefix before it is
unchanged): `low` asks for a few sentences or a short list without preamble
or optional detail (code stays complete), `high` for a long answer in several
sections with explanations, caveats and worked examples. `medium` adds
nothing. Responses echoes the level in `text.verbosity`. A `high` answer can
run to thousands of tokens, so a small output limit may leave it incomplete.
Measured answer length in words on three prompts, at the default `xhigh`
reasoning and with reasoning off, seed 7 (low / medium / high):

| Prompt | xhigh | reasoning off |
|---|---|---|
| What is a hash table? | 42 / 256 / 742 | 43 / 563 / 738 |
| Why is the sky blue? | 77 / 135 / 195 | 64 / 241 / 471 |
| How do I reverse a list in Python? | 64 / 72 / 258 | 36 / 239 / 315 |

With seed 11 at `xhigh` the same instructions gave 34 / 218 / 1989,
41 / 126 / 346 and 56 / 113 / 181. Earlier wordings ("Be concise" alone, or
"Verbosity is set to ..." without the concrete shape) moved `high` little at
`xhigh` and sometimes left it below `medium`. Evidence:
`.scratch/artifacts/request-surface/verbosity-*.json`.

Chat supports function `tools`, assistant `tool_calls`, and tool-result messages
with `tool_call_id`. Native rendering and validation belong to `local-engine`
([contract](../local-engine/README.md#native-tool-calling)); HTTP never executes
a tool. Calls carry stable IDs and JSON argument strings in the order the model
wrote the arguments. Arguments stream while the call is generated: the first
`tool_calls` chunk carries the ID and name as soon as the call names a
declared tool, a string parameter whose schema accepts any string streams as
it is written, any other value arrives whole when it ends, and the validated
call sends whatever is left (usually `}`), so the fragments concatenate to the
final arguments. Responses streams the same `response.function_call_arguments.delta`
events, and a `custom` tool's raw input as `response.custom_tool_call_input.delta`;
shell, `local_shell` and `apply_patch` calls are still reported whole. A call
is still validated when it ends: one that fails ends the stream with an error
after its fragments, and one cut short by the token limit stays `incomplete`
with what streamed. Chat finishes with `tool_calls` unless a token
limit takes precedence. Ordinary answer and reasoning text still stream.

Both APIs accept `tool_choice`, `parallel_tool_calls` and a function tool's
`strict`, and map them one to one onto the engine's native settings, which a
grammar enforces on every sampled token
([contract](../local-engine/README.md#constrained-tool-calling)).
`tool_choice` is `auto` (the default), `none`, `required`, or one named
function: `{"type":"function","function":{"name":"NAME"}}` in Chat and
`{"type":"function","name":"NAME"}` in Responses, or `allowed_tools`
(`{"type":"allowed_tools","allowed_tools":{"mode","tools"}}` in Chat,
`{"type":"allowed_tools","mode","tools"}` in Responses, whose entries name
functions, custom tools or built-in types as a `tool_choice` would): every
tool stays in the prompt, so the cached prefix is shared, while the grammar
admits calls only to the listed ones, mandatory with `mode: "required"`. The
other API's shape, unknown members and other tool kinds are refused. `required`
with no tools, or a name not in `tools`, is a 400. `none` no longer withholds
the tools from the prompt: they are rendered as for every choice, and the
grammar keeps the answer from containing a call. `parallel_tool_calls: false`
allows at most one call. `strict` (absent or `null` is `false`, unlike
OpenAI Responses, which attempts strict-schema normalization when omitted)
enforces that tool's arguments and needs a closed
schema the engine can enforce exactly; otherwise the request fails with 400
before generation, with the reason, rather than being enforced partially.
Responses echoes the accepted `tool_choice`, `parallel_tool_calls` and each
tool's `strict`. With all defaults, generation is unconstrained and calls are
validated after they are generated, as before. The grammar
constrains the call layout and strict arguments, not which tool is
appropriate or whether its values are sensible.

Responses accepts explicit message, reasoning, `function_call` and
`function_call_output` history in `input`, plus `instructions`. SSE uses typed
`response.*` events with sequence numbers and stable response/item IDs, ending
with completed, incomplete (output limit), or failed. It does not use Chat's
`[DONE]` sentinel. Non-streaming Responses sends whitespace heartbeats before
its final JSON.

Chat `response_format` and Responses `text.format` constrain the answer to
JSON. Each accepts `text` (the default), `json_object` (any JSON object) and
`json_schema`. Chat nests the schema in a `json_schema` object; Responses puts
`name`, `schema` and the optional `description` and `strict` directly in
`format`. `name` (1 to 64 of `a-z`, `A-Z`, `0-9`, `_`, `-`) and an object
`schema` are required, and unknown members are refused.

```bash
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Capital of France?"}],
  "response_format":{"type":"json_schema","json_schema":{"name":"capital",
    "schema":{"type":"object","properties":{"country":{"type":"string"},
      "capital":{"type":"string"}},"required":["country","capital"],
      "additionalProperties":false}}}}'
curl -s localhost:8080/v1/responses -H 'content-type: application/json' -d '{
  "input":"Capital of France?",
  "text":{"format":{"type":"json_schema","name":"capital","strict":true,
    "schema":{"type":"object","properties":{"country":{"type":"string"},
      "capital":{"type":"string"}},"required":["country","capital"],
      "additionalProperties":false}}}}'
```

The adapter maps either form onto the engine's native `ResponseFormat`
([contract](../local-engine/README.md#structured-output)). The engine compiles
the schema with llguidance 1.9.1 before the request is queued, so an invalid
or unsupported schema is a 400 before any generation, and then masks every
target token selection of the answer (first token, plain decode, lookup and
MTP verification, batched rows) to what the grammar allows; drafts remain
unconstrained proposals that the masked target verifies. The schema is
enforced whether `strict` is `true`, `false` or absent; Responses echoes
`strict` as sent (`false` when omitted). Reasoning, when enabled, is
unconstrained, and the format starts after `</think>`. Properties are generated
in the schema's own key order.

Compilation is strict rather than lenient. `uniqueItems`, `contains`, `not`,
unknown `format` values, `x-guidance` and `$ref`s outside the document are
refused, and nothing is fetched. `oneOf` compiles only when llguidance can
prove its branches disjoint (different JSON types, or objects whose shared
required property has different `const` values); overlapping branches such as
`integer`/`number` and unprovable ones such as `$ref` branches are refused.

A structured format may accompany tools. The native tool grammar then carries
the format as the final-answer branch: with `auto` the answer is either calls
or one document in the format, with `none` only the document, and with
`required` or a named function only calls. A token limit leaves incomplete JSON reported as
`finish_reason: "length"` (Chat) or `incomplete` with `max_output_tokens`
(Responses), never as a successful document. If the model ends its turn while
still reasoning, before any answer, Chat reports an error and Responses
`failed`, because `stop`/`completed` would claim a valid document. The engine's
`response_format_complete` applies only to non-text formats;
`tool_constraints_complete` separately reports tool-policy completion.
Optional tools with a text answer may end during reasoning without a call;
required or named calls mask end-of-sequence until the call is complete.

Responses are **stored by default**, as on OpenAI, in
`~/Library/Application Support/local-ai/responses`; a server using that
default store removes, at startup and then every hour, settled responses
created and last written more than 30 days earlier (pending background ones
are kept). The hourly pass is a timer because nothing signals a record turning
30 days old; each pass reads only records whose file is that old, judged by
the file's modification time, so a record written again later (a background
response settled late, or a file copied without its times) is kept 30 days
from that write. `--response-store DIR` uses DIR instead and prunes nothing
(conversations, files and batches use the always-on
[native services](#native-services) store either way). There is no way to
run without the store. `store` takes OpenAI's default of `true`
(an explicit `store: false` is honoured), and completed and incomplete
responses are written durably before their terminal event is sent; a failed
foreground generation is not stored, and a response whose write fails is
reported as failed rather than as stored (background responses differ; see
below). Each record is one JSON file holding the final response and its
resolved input (the earlier response's input and output, then
this request's items), so a follow-up never walks a chain and deleting an
earlier response does not break a later one. Large values are stored once
rather than once per turn: every string of 16 KiB or more (an inline
`input_image` or `input_file` data URL, a long text or answer) goes to an
owner-only content-addressed blob, `.{sha256}.blob`, and the record keeps
the digest in its place, lists the places (`blobs`, JSON pointer to digest)
and names each blob with a hard link of its own, `.{id}.{sha256}.ref`,
through which it is loaded, so a record is its file and its links and loads
without any other record. Below 16 KiB a blob's extra file, flush and open
cost more than repeating the string (at most 320 KiB over 20 turns). A
record with blobs is version 2, which binaries from before blobs refuse as
"not a version 1 record" (a 500) rather than misread; a record without large
strings is still version 1, and version 1 records load unchanged. Blobs and
links are flushed, and the directory synced, before the record naming them
is renamed into place, all under a shared lock on `.blobs.lock`; collection
takes that lock exclusively, lists the directory and unlinks links whose
record is gone and blobs no link names, on `DELETE` and in every sweep
(store open and each pruning pass). So no blob is collected between being
written for a record and the record appearing, a crash in between leaves an
orphan the next sweep removes, and finding references opens no record. A
20-turn conversation with a 5 MB image in its first turn takes 5.6 MB
instead of 105 MB, and a record of it loads in 0.44 to 0.77 ms instead of
0.89 to 1.6 ms. A save no longer rewrites the image but hashes it (SHA-256,
1.8 ms per 5 MB) and syncs the directory once more for its link, which
comes out about even: 14.5 to 17.2 ms a turn against 15.3 to 19.7 ms (six
runs each, on a machine busy with other builds; fsync dominates). A
`DELETE` costs a listing and a directory sync more (7.7 to 9.8 ms against
5.2 to 6.5 ms). Opening a store of 2,000 records whose 100 conversations
each carry an image takes 7.0 to 9.7 ms instead of 1.3 to 2.9 ms, two
listings of a directory with a link per record and no record opened
(reading each record's references instead took 0.3 to 1.4 s here, where
opening a file costs about 0.1 ms); a store without blobs opens as before,
in 1.2 to 1.6 ms. Records
contain prompts, tool results and the model's raw reasoning in plain text. The
directory is created `0700` and records and blobs `0600`; that is owner-only local data,
not encryption at rest. Only `resp_` IDs of ASCII letters and digits ever reach the filesystem.

`previous_response_id` carries over conversation items only: the earlier
request's `instructions`, tools, reasoning effort and sampling controls are not
inherited and must be sent again. `GET /v1/responses/{id}` returns the stored
object. `stream=true` (with optional `starting_after`) replays events only for
a background response created with `stream: true`, described below; other
records are stored as documents, not event streams, and refuse it with 400.
`starting_after` without `stream=true`, `include_obfuscation=true` and a
non-empty `include` are refused, as there are no extra fields to add.
`GET .../input_items` pages the resolved input with `after`, `limit` (1 to
100, default 20) and `order` (`desc` by default), reporting `first_id`,
`last_id` and `has_more`; an unknown `after` is 404. `DELETE` returns `response.deleted`.
Unknown, malformed, deleted and never-stored IDs are the same 404, and all of
these routes sit behind `--api-key` when it is set. With project keys, a
Response belongs to the project that created it, and another project's is
the same 404 ([ADMIN.md](ADMIN.md#limits)). `item_reference` inputs
remain unsupported; Responses `conversation` uses the separate
[native services](#native-services) store, not `--response-store`.

Background Responses are kept either durably or temporarily. With
`--response-store` and `store: true` (the default with a store) they are
stored durably as described here. With `store: false`, or on a server without
a store, they are kept temporarily (see
[Temporary background responses](#temporary-background-responses)); the
lifecycle below is the same apart from where and how long they are kept. Once
the engine has admitted the job, the `queued` response is already written to
its store. Without `stream`, `POST /v1/responses` returns it with 200; with
`stream: true` it answers with an SSE stream instead (see below). Either way a
detached worker persists
`in_progress` and one terminal state, `completed`, `incomplete`, `failed` or
`cancelled`. A failure after acceptance, such as a generation error, therefore
appears as a stored `failed` record rather than as an HTTP 400 on the
`POST`; requests that fail validation or engine admission are still rejected
in-request. Every state keeps the resolved `input_items` and reports
`background: true`, and `store: true` for a durable response. For a durable
response `GET /v1/responses/{id}` reads the latest state from the store, so
polling works from any server sharing the directory, and
`previous_response_id` refuses a response that is still `queued` or
`in_progress`.

`POST /v1/responses/{id}/cancel` settles a pending background response as
`cancelled`, keeping any output produced so far, and stops its generation.
It is idempotent: a response already terminal is returned unchanged. A
foreground (non-background) response is refused with 400. `DELETE` of a
background response this server is running deletes the record and cancels the
work, and the worker never writes it back. Each job holds an advisory lease on
an owner-only `.{id}.lock` sidecar in the store; cancel or delete of a
response still pending under another live server's lease is refused with 409.
A sidecar is unlinked, under its lease, when its response is deleted or by the
next sweep (store open or pruning pass) after it settles; since every pending
record has one, a sweep reads only records with a sidecar, and an ordinary
stored response is never opened at startup. Temporary files a crash left
mid-write, and sidecars with no record, are removed once an hour old.
A pending background record whose lease is free lost its writer to a crash or
`kill -9`. While pending, a durable record keeps the request it was created
from, and the server resumes it at startup (before serving), when a client
retrieves it after another server sharing the store died, and, for jobs a full
engine queue deferred, whenever a queue slot is released. Generation state is
not checkpointed, so the job restarts from its original request with its
original ID; a streamed job first continues its journal with a restart
boundary that finishes every output item the lost generation opened (as
`incomplete`, with the text it had), and the regenerated items follow at the
next output index, so no journaled event is repeated or contradicted and the
final Response holds every item its stream announced. A reasoning item
waiting for its summary is finished there too: a summary part the journal
opened is closed with the text it had and the item is `incomplete`; one whose
summary never started is `incomplete` with an empty summary. The regenerated
output gets its own summary. A compaction the record already reports
(`context_management`, run before the job was admitted) is reused, not run
again, and is announced exactly once. Only one server can
resume a job (the lease), a cancel or delete before the restart wins, and a
job continuing a conversation, one that declares hosted tools and had started
executing a hosted call (regenerating could run a shell command or file write
twice; see [docs/TOOLS.md](TOOLS.md)), or one without its stored request, is
marked `failed` instead, saying why. After a restart, `usage` counts only the
resuming server's generations (the regenerated answer and its summary), not
the lost generation or a compaction run before the restart.
Graceful shutdown refuses new background requests (503), marks every
unfinished job `failed`, cancels it and waits for the workers to release the
engine before exiting; a resumption still waiting for a queue slot stays
pending for the next server. A job has no client to stall, so `--stall-timeout`
does not apply to it. Stored records still contain the raw reasoning in plain
text; there is no encryption at rest.

A streamed background response (`background: true, stream: true`) also keeps
an append-only, owner-only event journal, `.{id}.events`, beside its record:
`response.created` and `response.queued`, `response.in_progress`, the output
events and the terminal events, each line numbered by `sequence_number` from
0. Each batch is flushed before any subscriber sees it, and terminal events
are appended only after the terminal record is durable, so a stream never
announces an end the store does not hold. The `POST` stream and
`GET /v1/responses/{id}?stream=true&starting_after=N` both read that journal
and send every event with a sequence number strictly greater than `N` (all of
them without `starting_after`), byte-for-byte as first written, from this
server or another sharing the store, and also after a restart. A cursor at or
past the end of a finished stream gives an empty stream. While this server is
generating the response, a subscriber is woken by the job's own notifications
rather than by polling, and holds no thread while it waits. At most 64 such
streams are open at once (each holds an open journal file); another, `POST`
or `GET`, is a 503 with `Retry-After: 1` before any job starts. A response still being generated by another live
server is refused with 409 rather than followed by polling a file this
process is not told about; after it finishes it can be replayed anywhere.

The stream is only a reader: a subscriber that disconnects, or stalls past
`--stall-timeout`, is dropped and the job runs on; reconnect with the last
`sequence_number` received. Only `POST /v1/responses/{id}/cancel` (or
`DELETE`) stops it. The published Responses stream has no cancellation event,
and a cancelled response is neither completed, incomplete nor failed, so as a
local convention a cancelled stream ends with a valid `error` event whose
`code` is `response_cancelled`; this is not a claim about `OpenAI`'s exact
cancellation wire behaviour. A deleted response's stream stops early. If a
journal write fails and cannot be cut back off, live streams close without a
terminal event rather than guess the next sequence number; once the writer is
gone, a replay numbers the end after the lines actually on disk, so every
sequence number a client has seen still names the same event. A torn final
line is ignored by readers and truncated by recovery. Only the first and
terminal batches pay for `F_FULLFSYNC`, so a power cut may lose deltas a
client saw but never the terminal record.

```bash
curl -s localhost:8080/v1/responses -H 'content-type: application/json' \
  -d '{"input":"Summarise RFC 9110 in one line.","background":true}'
# {"id":"resp_…","status":"queued",…}
curl -s localhost:8080/v1/responses/resp_…          # poll until terminal
curl -s -X POST localhost:8080/v1/responses/resp_…/cancel

curl -sN localhost:8080/v1/responses -H 'content-type: application/json' \
  -d '{"input":"Summarise RFC 9110 in one line.","background":true,"stream":true}'
# event: response.created … "sequence_number":0 …  (disconnect at any point)
curl -sN 'localhost:8080/v1/responses/resp_…?stream=true&starting_after=7'
```

`reasoning.encrypted_content` is always available: the key is
`~/Library/Application Support/local-ai/reasoning.key` unless
`--reasoning-key FILE` names another. The key is 32
random bytes in an owner-only file, created on first start without
overwriting anything and reused across restarts; a symlink, a non-regular
file, a file owned by another user, a file with group or other access, or a
malformed key is refused at startup. Reasoning output items of a request whose
`include` lists `reasoning.encrypted_content` then carry an
AES-256-GCM envelope (fresh random nonce per item) whose associated data binds
the envelope version, the model ID and the reasoning item's ID; the envelope
also carries that ID in the clear. A client may send the item back in a later
stateless request, with its original `id` or without one (the AI SDK omits it
with `store: false`, since `OpenAI`'s envelopes need none), when the ID the
envelope carries is used; the authenticated text replaces any plaintext
beside it, and an altered envelope, one from another key, model or item, or an
unknown version fails the request. Compaction items are sealed the same way.
`include: ["reasoning.encrypted_content"]` is accepted only with a key and is
the only `include` value implemented; without a key it, and any
`encrypted_content` input, are refused. A request that does not ask for it gets
no `encrypted_content` even when the server has a key (as with `OpenAI`); stored
responses, conversations and `previous_response_id` replay the plaintext they
keep, so they never need it. Replay after a server restart and
tamper rejection are covered by tests.

The key option is replay authentication, **not secrecy**: the raw reasoning is
still sent in the same response as `reasoning_text` content (streaming and
buffered), and with a store configured it is still written in plain text. It is
not encryption at rest and does not hide chain-of-thought from the client or
from anyone who can read the store.

`POST /v1/responses/input_tokens` accepts the Responses input contract
(`input`, `instructions`, `tools`, `reasoning`, `previous_response_id`, `text`,
`tool_choice`, `parallel_tool_calls`, `model`) and returns
`{"object":"response.input_tokens","input_tokens":N}`. `N` is exact: the
request is validated, rendered by the checkpoint template (tools, replayed
calls and the reasoning mode included) and tokenized by the same code that
generation uses, so it equals the `input_tokens` generation would report,
cached prefix included. Counting is CPU-only, never queues behind or waits for
generation, stores nothing and is not checked against the context window. Other
fields are refused rather than ignored. `tool_choice`, `parallel_tool_calls`
and tool `strict` do not change the prompt, so the count is the same for every
setting; no tool grammar is compiled, so a strict schema the grammar cannot
enforce is refused only when generation is requested.

Reasoning controls are `reasoning_effort` (Chat) and `reasoning.effort`
(Responses). The checkpoint template has three reasoning levels, `low` (a
"keep your thinking brief" instruction), `medium` (no instruction) and
`xhigh` (the default), or none: `none`, `low`, `medium` and `xhigh` run as
named, `minimal` runs at `low` and `high` at `xhigh`, and a Response's
`reasoning.effort` reports the level that ran. Usage counts reasoning tokens
from generated IDs through the first `</think>` token, inclusive; a response
truncated before that token counts all generated tokens as reasoning.
Prior reasoning is retained in replayed history by default, as the template's
`preserve_thinking` default does. Responses `reasoning.context:
"current_turn"` renders the template's `preserve_thinking: false`: assistant
turns up to the last user message lose their thought block, while the current
tool loop keeps its reasoning. It shortens long multi-turn prompts; the cached
prefix stays valid up to the previous user turn. `auto` (the default) and
`all_turns` keep every turn's reasoning, and the Response reports the
effective mode.
Responses reports measured prefix reuse as `input_tokens_details.cached_tokens`.
`cache_write_tokens` is zero: local cache creation has no separately accounted
or charged cache-write tier.

This is **not the full OpenAI platform contract**. Schema keywords
llguidance cannot enforce exactly, strict tool schemas the native grammar
cannot enforce exactly (including any without `"additionalProperties": false`),
built-in tools other than those of
[docs/TOOLS.md](TOOLS.md) (MCP, code interpreter and the like; `web_search`
is accepted but never offered to the model, which cannot search here, while
`file_search` runs natively over the vector stores),
`include` values other than `file_search_call.results` and
`reasoning.encrypted_content` (and that one without `--reasoning-key`),
log-probabilities and audio are rejected.
`tool_choice` `auto`, `none`, `required` and a named function,
`parallel_tool_calls` and tool `strict` are supported and enforced during
decoding, as is a structured format beside tools; a grammar guarantees the
form of what is generated, not that the model chooses well. Decisions are
answered only within the one evaluated scope of the installed artifact and
refused otherwise ([decisions](#decisions)). The MTP head and
next-token softmax are not calibrated decision probabilities either.

The compatibility target is the public OpenAI contract, not a particular
harness. The current schema audit is pinned to
[`openai-openapi` at e95c0fe](https://github.com/openai/openai-openapi/tree/e95c0fe615f45a19e8878af923a729948445a6bd)
(OpenAPI 3.1, 230 paths). SDKs, Codex and Oh My Pi are independent clients of
the same HTTP adapters; native Rust callers use the engine's request/event
types directly. No client-name branches belong in model execution.

Missing endpoint families are not all model limitations. Opt-in local response
storage (retrieve, delete, `input_items`), polled and streamed background
responses with cancellation and journal-based stream resumption, temporary
(`store: false`) background retention, Responses input-token counting,
JSON-object/JSON-schema output, tool calls constrained during decoding, and
durable conversations, files, uploads and text-endpoint batches
([native services](#native-services)) are now implemented as described;
vector stores (BM25, or embedding-ranked with the embedding model) are implemented too ([native services](#native-services));
compaction and reasoning summaries are implemented with the loaded model
([below](#compaction-and-reasoning-summaries)); decisions are served within one
evaluated scope ([below](#decisions)); embeddings,
moderations, text to speech and transcription run on required native models
(EmbeddingGemma 2, its moderation heads, Kokoro-82M, Whisper large-v3; see the
endpoint table); image input to generation runs on the checkpoint's own
vision tower ([image input](#image-input)), while audio input still needs a
multimodal generator.
Client-executed `custom`/`local_shell`/`apply_patch` tools, `file_search` over
the vector stores and the always-on sandboxed hosted tools are described in
[docs/TOOLS.md](TOOLS.md); other
built-in tools and fine-tuning still need their own services; the
Administration API is described in [docs/ADMIN.md](ADMIN.md). None is implemented by merely accepting its request fields.

[Codex at 4aa94dc](https://github.com/openai/codex/tree/4aa94dce270de668eff6e2fa8585c82385e84455)
uses stateless Responses, requests `reasoning.encrypted_content`, and sends
`client_metadata` (an extension absent from the pinned public spec; see
[Codex's request type](https://github.com/openai/codex/blob/4aa94dce270de668eff6e2fa8585c82385e84455/codex-rs/codex-api/src/common.rs#L278-L304)).
Responses creation ignores it like any member it does not know.
It is opaque tracking data: discarded, not echoed or persisted, and never
used for prompts, sampling or cache-session selection. It is distinct from
the public `metadata` field, which is retained on the response. With
`--reasoning-key` the encrypted-content request is honoured, and so are
reasoning summaries and `/responses/compact`, each at the cost of an extra
generation ([below](#compaction-and-reasoning-summaries)); neither has been
exercised with Codex itself. Codex's `custom` `apply_patch` (its Lark
grammar enforced during decoding), `local_shell` and `namespace` (sub-agent)
tools are round-tripped for the client to run, and several leading developer
messages are joined into the template's one system turn; `web_search` is
accepted but never offered to the model (no search provider), other hosted
OpenAI tools are refused, and strict tools are accepted only when their
schemas can be enforced exactly. Codex's default configuration is accepted
(see the client results below), but with no web search and only this one
local model it is not equivalent to Codex on OpenAI. Codex handles raw `reasoning_text` and
summary events separately; raw reasoning is not relabeled as a summary.
The Oh My Pi example in the engine README configures that client to use only
implemented capabilities.

Chat rejects `store:true`.
Error envelopes include nullable `code` and `param`; choices include nullable
`logprobs`; Responses includes nullable `access_programs`. The model's `created`
timestamp is its registration time at server startup, not its training date.

With `stream: true`, Chat Completions responses are `text/event-stream` chunks terminated by
`data: [DONE]`. Chat separates `reasoning_content` from visible `content`.
Non-streamed Chat/completion JSON includes token usage, cache source, timings, and speculation
counters. JSON responses use zstd when `Accept-Encoding` contains `zstd`, at
level 22, chosen for ratio rather than for a cheaper CPU bill. A zstd
dictionary was measured for this path and rejected; see
[Tried and rejected](#tried-and-rejected).

That ratio is set by how much of the body is flushed at once rather than by the
level. A completion response is written in pieces as tokens arrive, and zstd
cannot compress across a flush boundary, so the same text split into many
pieces gives each frame less to compress than one buffered write does. Measured
on an M2 over 1139 tokens of real generated text at level 22, one zstd flush
per frame, with the 18 bytes of HTTP chunk overhead per frame counted in:

| Body written every | Plain bytes | zstd bytes | zstd ratio |
| --- | ---: | ---: | ---: |
| 1 piece | 24,957 | 28,332 | 0.16x |
| 16 pieces | 5,733 | 4,384 | 1.02x |
| 64 pieces | 4,761 | 2,743 | 1.62x |
| the whole body | 4,455 | 1,903 | 2.34x |

A chunked body compresses worse than a single buffered one, and a frame per
piece makes compression a net loss rather than a saving: 28,332 bytes on the
wire against 24,957 plain. That is why the flush carries two bounds and not
one. The body is written when `BODY_FLUSH_BYTES` (8 KiB) has accumulated or
`BODY_FLUSH_INTERVAL` (1 s) has elapsed, whichever comes first, and at the
2.6-17 tok/s measured here 8 KiB is about 150 s of output at the slow end, so
the interval is what binds. The frame count then stays near one per second, and
a streaming completion body lands between the 16- and 64-piece rows — nearer
1-1.6x — so there is no single ratio to quote for one and the whole-body 2.34x
is the ceiling rather than a figure a completion response reaches. Level 22 is
deliberately unchanged: the ratio is a property of the flush policy, not of
the level.

The engine decodes up to eight requests together (see
[Concurrent requests](#concurrent-requests)) and queues eight more behind them;
submission to a full queue fails immediately. Dropping an `EventStream`, using
its cancellation handle, or disconnecting a client cooperatively cancels
queued or running generation, also while it is still queued or prefilling and
nothing has been sent. A streaming (SSE) response sends its headers at the
first prefill boundary rather than the first token, and an SSE comment
(`: keep-alive`) at every later boundary, as the non-streaming body sends
whitespace, so a client that leaves during a long prefill fails a write and
cancels it. A reset and a graceful close are both
detected and release the engine promptly — a graceful close was caught after
only 3,544 bytes — so `--stall-timeout` covers the one case neither of them
reports: a client that stops reading while holding the socket open.

`--stall-timeout` bounds how long a response may go without the server being
able to hand it a frame. A client can stop reading its socket without closing
it — a stalled network, a dead consumer, a client that wandered off — and
neither TCP nor axum signals that. Left alone, the send path blocks, the engine
worker blocks inside its emit callback upstream of every cancellation
checkpoint, and the generation holds its queue slot indefinitely. (Beside other
requests the worker no longer waits on a slow reader at all: events queue per
request and reach the channel without blocking, so only the stalled request
is held. A finished request's undelivered last events never hold up the
next job either: the idle worker waits for the next job and for room in
those channels together, so a stalled reader's leftovers wait for its stall
timer while later requests run.) When the budget is exceeded the server logs, cancels
that generation, releases the queue slot, and drops the request, leaving the
engine free for the next client. The default is 30 seconds; a value outside
10 to 3600 inclusive, or one that is not a whole number of seconds, is refused
at startup.

What a client sends is bounded too, on HTTP/1.1, HTTP/2 and (for bodies)
HTTP/3. An HTTP/1 request's headers must arrive within 30 s of the server
starting to read them, which also closes an idle keep-alive connection, and
a TCP connection that has not delivered a whole request within 30 s of being
accepted (one that sends nothing, or half an HTTP/2 preface) is closed. A
request body must deliver at least 64 KiB, or end, within every
`--stall-timeout` the handler spends waiting for it; otherwise the read fails
(`400`, "the request body stalled") and the connection closes. A slow but
moving upload is never cut off, so a 512 MiB file is fine at a few KiB/s,
but a trickle cannot hold one of the few upload, transcription or generation
slots indefinitely. A transcription form, which is held in memory, must
arrive whole within `--stall-timeout` once it has its slot (`408`
otherwise), and a file download whose client stops reading for
`--stall-timeout` is dropped and gives its slot back.

Both HTTP body pumps share an event-driven backpressure wait. A full channel
parks the pump thread until capacity becomes available, the receiver closes,
or the single stall deadline expires. It does not retry every few milliseconds.
The engine's worker likewise blocks on its request channel when idle (and on
room in finished requests' channels while they hold events); its
nonblocking admission checks between GPU steps are active scheduling, not
idle polling.

One number bounds two clocks, but not to the same depth on the two paths. On a
streaming response it is a consumer-liveness budget: the server only produces a
frame when the engine emits one, so frames arrive as fast as the engine makes
them, and what is left to measure is how long a client goes without taking one.
On a non-streaming response it bounds that too, because the body is written
while generation runs, and it bounds the engine's own gaps between events as
well, since the wait for the next event carries the same budget as the write.
The producer half is the binding one for a healthy request, so the floor still
has to clear it, and the engine's inter-event gaps are not small: the floor
comes from a measurement rather than from taste; see
[Tried and rejected](#tried-and-rejected).

The budget starts only after the first event arrives, so it does not bound
time-to-first-token. Prefill now reports a boundary at every 512-token block (or
every chunk of 24 to 32 tokens while other requests decode; see
[Concurrent requests](#concurrent-requests)) and the boundary is written to the
client, but it deliberately does not start a clock. On the M2 these
timeout measurements used, an early build prefilled at roughly 3.6-4.2 tok/s,
about 35 seconds per 128-token block of entirely healthy work, and a budget
armed on a boundary would abandon ordinary long prompts. A later cold
6,438-token prompt on that M2 measured 50 boundaries and 349.9 seconds of
prefill (about 7 seconds per boundary) at the 10-second floor with no false
cancellation.

Two limits are worth stating plainly. On streaming, the clock starts late by
construction: a client must first fill the socket buffers before the server feels
backpressure. Measured on an M2, the server absorbed 564,550 bytes (551.3 KiB)
before the stall began, and SSE frames average 194.9 bytes, so roughly 2,900
tokens are generated first. Reclaim time is therefore time to fill the socket
buffers plus the budget, not the budget counted from the request. The buffer
ceiling is autotuned rather than fixed, and 4 MiB is the observed autotune
maximum on that machine. Second, a non-streaming body is written while
generation runs, so a vanished client's write can fail and the generation is
cancelled with it — but only once the socket buffers have filled, and a
completion body is about 4 KiB at 900 tokens, some 140x under the 551.3 KiB
measured above. At that size the write does not fail, the disconnect itself is
not observed, and the generation runs to its end; a failing write needs a body
past that 551.3 KiB, roughly 125,000 tokens. The budget is not a substitute for
that detection. What it does now bound on this path is a request going
undeliverable for longer than the budget, which a body that exists on the wire
makes observable and a buffered body could not be at all. That buffered body is
what one measured vanished client cost: 706 seconds of engine time.

### Image input

With Prism's BF16 vision projector beside the GGUF
(`Ternary-Bonsai-2-27B-mmproj-BF16.gguf`, see the README's model install;
size and SHA-256 are checked when it loads, in about 0.6 s), Chat Completions
and Responses take images. The projector is required: without it startup
stops with an error naming the missing file, and the startup policy's
`vision_reason` names the one loaded. There is no flag: the projector is
discovered like the MTP head. The lossy Q8_0
projector is not read.

| API | Image parts |
| --- | --- |
| Chat Completions | `{"type":"image_url","image_url":{"url":"data:image/…;base64,…" or "https://…","detail":…}}` (or `image_url` as a bare string); `{"type":"file","file":{"file_id":…}}` or `{"file_data":"<base64>"}` |
| Responses | `{"type":"input_image","image_url":"data:…" or "https://…","detail":…}` or `{"type":"input_image","file_id":…,"detail":…}`, in user messages and `function_call_output` |

Images may sit anywhere among the text parts of user and tool messages and
keep their place; system, developer and assistant messages cannot hold them,
as in the checkpoint's chat template. `file_id` names an upload of the
[Files API](#native-services) (at most 64 MiB per image or video). JPEG
(decoded with libjpeg-turbo), PNG, WebP, GIF (first frame) and BMP are read,
at most 16,384 pixels a side and 67,108,864 (64 MP) in all, checked from the
header before any pixel is decoded.
`http(s)` URLs are fetched by the server ([remote media
URLs](#remote-media-urls)). Conversations store `input_image` parts too
(within the 4 MiB item limit, so larger images go by `file_id` or URL).

The template renders each image as
`<|vision_start|><|image_pad|><|vision_end|>` exactly where the checkpoint's
template does; the engine then replaces the pad with one token per merged
2x2 patch group. `detail` sets the pixel budget the image is resized into
(the processor's `smart_resize`: sides rounded to multiples of 32, at least
65,536 pixels):

| `detail` | Budget | Image tokens at most | Tower time, M4 Pro |
| --- | ---: | ---: | --- |
| `low` | 262,144 px | 256 | ≤ 0.2 s |
| `auto` (default) | 4,194,304 px | 4,096 | 9 s at 4,015 tokens (a 2400x1800 photo) |
| `high` | 16,777,216 px | 16,384 | attention grows with the square of the patches |

`auto` is llama.cpp's mtmd limit for this architecture, the runtime Prism
publishes the checkpoint for; `high` is the HF processor's own maximum.

**Limits per request.** At most 64 image and video parts, 256 MiB of
encoded media together, 536,870,912 pixels (512 MP) of decoded images
together, and 72,000 video frames together (two videos at the
[per-video limit](#video-input), checked as each is read); more is a 400.
Images are sized from their headers first, so `input_tokens` counts them
without decoding a pixel, and a request whose prompt plus `max_tokens`
exceeds the context fails with the context error before any image is
decoded or encoded. Images are then decoded, resized and encoded one at a
time. A file prepared before (an image kept in a conversation's history) is
remembered by its SHA-256 and `detail` (the last 4,096), so while its encoded
rows are cached it is not decoded again. A library caller that submits with
`EngineHandle::chat_with_cancel` can stop video decoding between samples
and the tower between submissions.

**Implementation.** The tower (`local-engine/src/bonsai_vision/`,
`shaders/bonsai_vision.metal`) is the Qwen3.5 ViT: a 16x16 patch embedding
over 2 temporal frames, bilinearly interpolated learned positions, 27
pre-norm blocks of width 1152 (16 heads of 72, 2-D rotary positions,
tanh-GELU MLP of 4304) and the 2x2 merger (LayerNorm, then an exact-erf
GELU MLP to the decoder's width of 5120), all in F32 with BF16 weights widened exactly. Attention is
bidirectional over the whole image, a flash-attention kernel with its online
softmax on the score matrices' lanes, split into submissions of 2^24
query-key pairs that yield to running decode steps, so a large image does
not stall other streams. Encoded images are cached by content. Their rows
enter the decoder in its signed-Hadamard input basis in place of token
embeddings, and the decoder's interleaved M-RoPE takes `(t, h, w)` positions
for image tokens; text after an image continues from the largest position
plus one, as the HF model's `get_rope_index`.

**Prompt caching.** Each image token has a virtual id above the vocabulary,
derived from the projector digest, the digest of the resized pixels and its
index, so every prompt-cache tier (GPU, host and disk snapshots) reuses a
prefix only when the same image at the same resolution sits at the same
place; a different image, or the same one at another `detail` that changes
its size, misses from its first token. The n-gram drafter never proposes
image tokens.

**Parity.** References came from `bonsai-vision-ref`, a private Kaggle CPU
job: HF transformers 5.20.1 `Qwen3_5VisionModel` in FP32 with the mmproj's
weights, and llama.cpp mtmd from PrismML's fork on the pinned PTQ1 GGUF.
`cargo run --release -p local-ai --bin bonsai-vision-eval -- tower|llama DIR`
compares a download of its output (`bench IMAGE [detail]` times the tower):

- Preprocessing against `Qwen2VLImageProcessorFast`: all pixels identical for
  10 cases (a text card, four photos, one also at `low`, a 2400x1800 photo
  at `auto`, 1600x96 and 96x1600 strips, a 7x5 image raised to the minimum).
- Tower output against HF FP32: cosine 1.0000000 for every case (minimum
  per-token cosine 1.000000), including the 4,015-token photo; per-block
  outputs on the text card agree to cosine 1.0000000.
- End to end against llama.cpp (five prompts: text only, text card, photo,
  two images, multi-turn), next-token distributions over the first five
  answer positions: KL ≤ 4.2e-4 nats (mostly ≤ 1e-4), top-1 identical at
  every position, and greedy 24-token continuations identical in four of
  five; the photo one diverges at token 18 between two wordings. With
  llama.cpp's own image embeddings substituted for the native tower, all five
  are identical, so the remaining difference is llama.cpp's tower: its
  embeddings are 0.9991–0.99999 cosine from the native ones, which match
  HF's exactly. The card's text (`BONSAI 27B`, `Vision test 2026`,
  `Invoice #4821`, …) is read back verbatim.

### Remote media URLs

An `http(s)` URL in `image_url`, `input_image`, `video_url` or `input_video`
is fetched by the server, through the hosted fetch tool's transport
(`local_services::tools::fetch::download`, [docs/TOOLS.md](TOOLS.md)):

- Any public host, by name or public IP literal; no allowlist.
- **Loopback, private (RFC 1918), link-local (including the
  169.254.169.254 metadata address), CGNAT, unique-local, documentation and
  other non-public addresses are refused, by default and without a flag.**
  Every address a name resolves to must be public, and the connection goes
  to a checked address, so DNS rebinding cannot reach the LAN. A request
  names a URL on behalf of whatever produced its content (a web page a tool
  read, a document an agent pasted), so a server reachable by others must
  not become a way to probe or `GET` its own network. Media on the local
  machine or LAN goes as a data URL or through `/v1/files`.
- At most five redirects, each target checked anew, never from `https` to
  `http`; `https` verified by the macOS trust store.
- At most 64 MiB (the generation request body limit), refused (not
  truncated) by `Content-Length` or while reading; one 60-second deadline
  for the whole exchange, and 120 seconds for all of one request's fetches
  together.
- The response's `Content-Type` must be `image/*` for images and `video/*`
  (or `application/mp4`) for videos; none, `application/octet-stream` and
  `binary/octet-stream` pass and the decoder decides. Anything else (an HTML
  error page, say) is a 400 naming the type, refused from the headers before
  any of the body is read.
- Fetched bytes are kept by URL for ten minutes (256 MiB in all), so a
  conversation replaying the URL in its history every turn fetches it once.
- One request's parts, whatever their source (data URL, `file_id` or URL),
  hold at most 64 parts and 256 MiB together, checked as they are read (a
  `file_id` before its bytes are); a `file_id` or URL named several times is
  read once.

Every failure is a 400 naming the URL and the reason.

### Video input

With the vision projector installed, videos go where images go:

| API | Video parts |
| --- | --- |
| Chat Completions | `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,…" or "https://…"}}` (vLLM's and llama.cpp's extension; `video_url` may be a bare string); llama.cpp's `{"type":"input_video","input_video":{"url"|"data":…}}`; `{"type":"file","file":{"file_id"|"file_data":…}}` holding a video |
| Responses | `{"type":"input_video","video_url":"…"}` (or `{"url":…}`) or `{"type":"input_video","file_id":…}` — this server's extension, as OpenAI's Responses API has no video part |

MP4, MOV and M4V files are read, with any codec macOS decodes (H.264 and
HEVC on every Apple Silicon Mac, ProRes, AV1 where the hardware has it);
Matroska and WebM are refused with a 400, since AVFoundation does not open
them. A `file` part whose bytes are an ISO media file is taken as a video,
otherwise as an image. A video may last at most 600 s and hold at most 36,000
frames (600 s at 60 fps): the counting pass refuses it with a 400 as soon as
it passes either, since both passes read every frame up to the last. Frames
are held to the image limits (16,384 pixels a side, 64 MP), checked on the
track header and again on every decoded frame. The file is opened with
`AVURLAssetReferenceRestrictionsKey` set to forbid all references, so a movie
naming other files or URLs as its media data is not followed outside itself.

The layout is the checkpoint's own processor (`Qwen3VLVideoProcessor`,
which Qwen3.5 uses, with the base model's `video_preprocessor_config.json`):

1. **Sampling**: 2 frames per second, `int(frames / fps * 2)` frames, at
   least 4 and at most 768 (and at most the video's own), at
   `round(linspace(0, frames - 1, n))`.
2. **Size**: every frame resized (antialiased bicubic) to one size in
   multiples of 32 within a clip budget of 25,165,824 pixels (12,288 tokens)
   and, as the reference qwen-vl-utils does (`cap_pixels_per_frame`, the
   transformers default from v5.22), at most 786,432 pixels (768 tokens) a
   frame; at least 4,096 pixels a clip.
3. **Frame groups**: each pair of frames (the last repeated when the count
   is odd) is one temporal patch group, its 16x16x2 patches embedded with
   both temporal kernel slices, and attended to within the group only.
4. **Prompt**: the template's `<|vision_start|><|video_pad|><|vision_end|>`,
   with the pad expanded as the processor expands it: for each group
   `<{t:.1f} seconds>` (the mean time of its two frames),
   `<|vision_start|>`, the group's tokens, `<|vision_end|>`. Each group takes
   image M-RoPE positions; the timestamps carry the time
   (`get_rope_index` splits video grids into `t = 1` grids).

Video tokens count toward the prompt like image tokens (a 10-second 640x360
clip at 30 fps is 10 groups of 220 tokens; a long or large one stops at the
12,288-token clip budget). Memory stays bounded by that budget rather than
the video's length: frames are resized as soon as they are decoded (one
source frame in memory at a time), a clip's resized frames hold at most 72
MiB, the tower runs one frame group at a time (at most 3,072 patches, about
150 MB of activations), and the encoded rows (20 KB a token, at most 240 MiB
a clip) are cached by content with images. Prepared videos are cached by
file digest (256 MiB), so a video kept in a conversation's history is
decoded once. `detail` does not apply to videos.

**Decoding.** AVFoundation through `objc2` (no new C dependency; the repo
already binds Metal through it), from a temporary file: one pass over the
compressed samples counts the frames and their duration (the reference
decoders' frame count and `frames / duration` rate) and reads the stream's
declared colour matrix; a second pass decodes in presentation order and
keeps only the sampled frames. VideoToolbox returns its native 8-bit 4:2:0
planes, converted to RGB with the swscale fixed-point arithmetic the
EmbeddingGemma video path already reproduces (BT.601 when the stream
declares no matrix, as FFmpeg reads it, where VideoToolbox's own conversion
would guess BT.709 for HD). 10-bit streams are reduced to 8 bits by
VideoToolbox, which can differ from FFmpeg's conversion by a few levels.

**Parity.** `bonsai-vision-eval video OUT_DIR CLIP...` dumps the native
probe, sampled frames, patch rows, tower output and prompt layout of each
clip; `.scratch/video/ref.py` (run with `uv run --with torch --with
torchvision --with transformers==5.20.1 --with av --with gguf --with numpy`)
compares them with transformers 5.20.1 on CPU: `sample_frames` on PyAV's
metadata, `Qwen3VLVideoProcessor` on the native frames, `replace_video_token`
inside the template's placeholder, `Qwen3_5Model.get_rope_index`, and the
FP32 `Qwen3_5VisionModel` with the mmproj's weights. Five synthetic clips
(H.264 640x360 at 30 fps with B-frames, HEVC 272x480 at 25 fps in MOV,
H.264 64x48 with 3 frames, H.264 1280x720 at 24 fps for 20 s, BT.709-tagged
H.264 480x270 at 29.97 fps):

- Frame counts, rates and sampled indices equal PyAV's for all five.
- Decoded frames: bit-identical, for all five, to the EmbeddingGemma video
  path's (`ffmpeg`'s decoded planes through the same swscale conversion,
  which matched torchcodec's frames for the reference): VideoToolbox's
  planes equal FFmpeg's and the colour matrix is chosen alike, including
  BT.601 for the untagged 1280x720 clip. PyAV on this Mac (arm64 swscale)
  is within 2 levels (50–52 dB PSNR).
- Grids identical, and every pixel value of `Qwen3VLVideoProcessor`'s output
  identical (0 of 81,518,592 differ; the 20-second clip at the per-frame
  cap included).
- The expanded prompt text and every M-RoPE position identical.
- Tower output against HF FP32: cosine 1.0000000 for every clip (minimum
  per-token cosine 0.99999997), up to 2,376 tokens (4 groups) per clip.

No llama.cpp reference was made: upstream mtmd's video support lays videos
out differently from the checkpoint's processor (FFmpeg's `fps` filter at 4
fps, a `Video:` prefix and `[0m5.00s]` timestamps every 5 s, no per-group
timestamps), so it would not check this layout. The decoder side is the
image path's, whose end-to-end parity is above.

### Coding client configuration

Codex, opencode and Oh My Pi drive `local-ai serve` end to end with the
configurations below, each recorded against the real model in
[client runs](#coding-clients-and-long-context-coding-quality). Each sets a
65,536-token context window and compacts at 32,000 prompt tokens
(chosen by the [threshold sweep](#compaction-threshold)). `GET /v1/models` reports
the admitted `context_length` (262,144 on a 48 GB Mac), but a client that
compacts only near that limit runs long prompts whose prefill is about half as
fast and whose exact lookups fail. A window larger than the admitted
`context_length` gets a context-overflow 400 instead of a compaction.

**Codex** (`~/.codex/config.toml`; Codex 0.163 accepts only
`wire_api = "responses"` for a custom provider):

```toml
model = "ternary-bonsai-2-27b"
model_provider = "local-ai"
model_reasoning_effort = "xhigh"   # low, medium, xhigh (high also runs at xhigh), or none
model_reasoning_summary = "none"   # "auto" adds a summary generation after every reasoning answer
model_context_window = 65536       # Codex assumes 272,000 for a model it does not know
model_auto_compact_token_limit = 32000  # default 90% of the window; capped at 90% of it
# model_catalog_json = "/Users/you/.codex/bonsai.json"  # optional, see below

[model_providers.local-ai]
name = "local-ai"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
# with authentication on: env_key = "LOCAL_AI_API_KEY"

[model_providers.local-ai.capabilities]
remote_compaction = "v2"   # compaction_trigger turns, for automatic compaction and /compact
```

Codex sends `store: false` with the whole conversation in `input` and asks for
`reasoning.encrypted_content` (sealed with the reasoning key created on first
start). With `remote_compaction = "v2"` it compacts through this server: when
a turn's total tokens reach `model_auto_compact_token_limit` (checked before
each user turn and after each sampling round that continues with tool
results), or on `/compact`, it sends a `compaction_trigger` turn and
replays the sealed `compaction` item it gets back
([compaction](#compaction-and-reasoning-summaries)). The summary is written
with the server's own compaction prompt, and the item keeps the newest tool
calls and results verbatim, where Codex's local compaction (what it does
without the capability: its summarization prompt as an ordinary Response)
keeps only user messages and its summary. Codex 0.163 has no `"v1"` value, so it never calls
`/v1/responses/compact`. Codex declares its freeform `apply_patch` tool (a `custom` tool
with a Lark grammar, enforced while the patch is generated) only for models
its catalog marks freeform; for any other model it edits through
`exec_command`. Its fallback instructions still mention `apply_patch`, so the
model sometimes calls it anyway: the server returns that call like any other
(a call to an undeclared tool is not an error), Codex answers it `unsupported
call: apply_patch`, and the model edits with `exec_command` instead. To give
it the tool, copy its own catalog entry for a
freeform model, whose base instructions are written for that tool, and
point `model_catalog_json` at the result:

```bash
codex debug models | jq '{models: [.models[] | select(.slug == "gpt-5.5")
  | .slug = "ternary-bonsai-2-27b" | .display_name = "Ternary Bonsai 2 27B (local-ai)"
  | .supported_reasoning_levels = (["low", "medium", "xhigh"] | map({effort: ., description: .}))
  | .default_reasoning_level = "xhigh" | .default_reasoning_summary = "none"
  | .support_verbosity = false | .default_verbosity = null | .model_messages = null
  | .upgrade = null | .service_tiers = [] | .additional_speed_tiers = []
  | .web_search_tool_type = "text" | .supports_search_tool = false
  | .input_modalities = ["text"] | .supports_image_detail_original = false
  | .context_window = 65536 | .max_context_window = 65536]}' > ~/.codex/bonsai.json
```

The copied instructions introduce the assistant as Codex based on GPT-5;
nothing else in them is specific to OpenAI's models.

**opencode** (`~/.config/opencode/opencode.json`, or `opencode.json` in a
project). Its custom providers are AI SDK packages:
`@ai-sdk/openai-compatible` speaks Chat Completions and `@ai-sdk/openai`
speaks Responses; both work.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "local-ai/ternary-bonsai-2-27b",
  "small_model": "local-ai/ternary-bonsai-2-27b",
  "provider": {
    "local-ai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local-ai",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "ternary-bonsai-2-27b": {
          "name": "Ternary Bonsai 2 27B (local-ai)",
          "reasoning": true,
          "tool_call": true,
          "limit": { "context": 65536, "input": 40192, "output": 8192 },
          "options": { "reasoningEffort": "xhigh" }
        }
      }
    }
  },
  "compaction": { "reserved": 8192 }
}
```

opencode compacts when a turn's tokens reach `limit.input` less
`compaction.reserved` (40,192 − 8,192 = 32,000 here); without `limit.input`
it compacts at `limit.context` less `limit.output` (57,344 for this window).

Through Chat Completions opencode sends `max_tokens` (the model's
`limit.output`), `stream_options.include_usage`, its tools with `tool_choice:
"auto"`, and `reasoning_effort` from the model's `reasoningEffort` option
(`none`, `low`, `medium`, `high` or `xhigh`; without one the server reasons at
`xhigh`); it reads `reasoning_content` deltas as reasoning and replays the
history as Chat messages. For Responses, use `"npm": "@ai-sdk/openai"` and
add `"apiKey"` to `options` (the package refuses to start without one; the
server checks it only when authentication is on); it sends `store: false`,
asks for `reasoning.encrypted_content` and, on its title request,
`reasoning.summary: "auto"`.

**Oh My Pi**: the `models.yml` provider in the
[engine README](../local-engine/README.md#oh-my-pi-custom-provider) (Responses, efforts
`low`, `medium` and `xhigh`), with `omp config set compaction.thresholdTokens 32000`.

### Compaction and reasoning summaries

Both run on the one loaded checkpoint as ordinary, queued generations with
thinking off; no second model exists. The policy (when to compact, what is
kept) is model-free in `local_services::compaction`; the summary prompts and
the rendering of a summary belong to `local_engine::compaction`; the server
only adapts them ([engine contract](../local-engine/README.md#compaction-and-reasoning-summaries)).

`POST /v1/responses/compact` takes `model`, `input`, `previous_response_id`,
`instructions` and `prompt_cache_key` (`prompt_cache_options` is refused,
`service_tier` as elsewhere, and unknown members are ignored) and returns
a `response.compaction` object. It counts the resolved window, keeps the
newest items up to a quarter of that count verbatim, and replaces the rest by
a summary (written from the whole window, kept items included).
`output` is the kept items, then one `compaction` item whose
`encrypted_content` is the summary *and* the kept items, sealed with the
`--reasoning-key` cipher under its own domain and bound to the model and item
ID. `usage` is the summary generation's. When nothing can be summarized the
window comes back unchanged with zero usage; when the summary would not
shorten the prompt (a short history can summarize longer than itself), the
window also comes back unchanged, with the summary's usage, since it ran. A cut never falls between a
`function_call` and its `function_call_output` or inside an assistant turn, so
an unanswered call and everything after it are always kept; a leading system
message and the latest user message are kept verbatim too.

A compaction item is accepted as input anywhere (Responses, input-token
counting, conversations, stored history). A prompt is rendered from the latest
compaction item: the kept leading system message, then its summary as one
`user` message under a `[Context compacted]` header, then its kept items, then
everything after it. Items before it are skipped, so the client, the response
store and the conversation keep every original item while the prompt shrinks.

`context_management: [{"type":"compaction","compact_threshold":N}]` on
`POST /v1/responses` (one entry, `N` at least 1000, or null or absent for the
server's default of 32,000, chosen by the
[threshold sweep](#compaction-threshold)) compacts before
generation when the exact prompt count reaches `N`, or when the prompt leaves
less than `max_output_tokens` (default 8,192) of the admitted context, which is
the output space the harness reserves. The kept budget is a quarter of `N` (or
of the context less that reservation). A compaction that would not shorten the
prompt is discarded and the request is generated uncompacted. The compaction item is the Response's
first output item (`output_item.added`/`done` at index 0, right after
`response.in_progress`). With
`conversation` or `previous_response_id` the item is stored with the output, so
the compacted window persists across restarts. With `background`, a due
compaction runs before the job is admitted, so the `POST` returns `queued`
only after it, with the compaction item already the first output item; a
streamed job announces it right after `response.in_progress`.

An input that ends with a `{"type":"compaction_trigger"}` item is a
compaction turn, the protocol of Codex's remote compaction (provider
capability `remote_compaction = "v2"`): Codex sends its whole history, its
tools and the trigger as an ordinary streamed `POST /v1/responses` and
expects exactly one `compaction` output item, delivered in
`response.output_item.done` before `response.completed`. The history is
compacted whatever its length, keeping the newest quarter of its prompt count
verbatim as `/responses/compact` does, and the Response's only output item is
the compaction item; nothing else is generated, `usage` is the summary
generation's, and the Response is `completed` even when the summary used its
whole 2,048-token budget. When nothing can be summarized, the summary is
empty, or it would not shorten the prompt, the item seals the history
unchanged instead (it renders exactly as the input did), since the client
asked for one. Codex then replaces its history with the user messages it
retains, its re-injected developer context, and the item, last; the next
request renders from the item, so those retained copies before it are not
rendered twice (the item kept the leading instructions and the latest user
message itself). The trigger is never history: it is not stored, appended to
a conversation, or rendered, and one that is not the last input item is
dropped. A trigger needs the reasoning key and is refused with `background`.

Without a key (only in tests now, since a server always has one) both are refused rather than returning the summary
in a field named `encrypted_content`: the item carries the conversation's
content, the contract calls it encrypted, and plaintext under that name would
mislead.

`reasoning.summary` (`auto`, `concise` or `detailed`) runs after the answer's generation ends: the raw
reasoning is quoted to the model, which writes the summary as visible content.
It streams as `response.reasoning_summary_part.added`,
`response.reasoning_summary_text.delta`, `.done` and
`response.reasoning_summary_part.done`, then the reasoning item's
`output_item.done` (withheld until then, so the answer's items finish first)
with one `summary_text` part. The raw reasoning stays the item's
`reasoning_text` content (and `encrypted_content`, when requested) and is never relabeled.
`reasoning.summary` is echoed. A summary that cannot be generated fails the
Response (a part already streamed is closed with the text it had, and its
item is `incomplete`); a cancelled one gets none; with effort `none` nothing
runs. The summary runs on the stream's own pump and its deltas go out as they
are generated, under the same stall budget; a client that disconnects cancels
it. With `background` it runs on the job's pump: each piece is journaled
before the next is generated, so a subscriber or a replay sees it at once,
and a cancel, delete or shutdown during it stops it at its next piece
(`cancelled`, or `failed` saying why). Its output budget is 160, 320 or 900 tokens by detail.

Cost: the summary request is the conversation's own request (its messages,
tools, session and reasoning effort) with the compaction prompt appended as
one more user turn, so its prompt extends the one the conversation's last
turn prefilled, and the prompt cache serves everything up to that turn's
snapshot. The cache keeps exact prefixes only at snapshot boundaries (the
recurrent layers' state cannot be rewound), so a summary of a shorter prefix,
or one rendered under another system turn (thinking off for a conversation
that reasons, whose effort instruction leads the system turn), would prefill
the whole history again: before this change a 32K-token compaction cost about
five minutes of prefill. The summary never reasons: for a conversation that
reasons, the request keeps the effort's instruction in the system turn and
closes the reasoning block in the prompt (`<think>\n\n</think>\n\n`, the
template's thinking-off opener), so it shares the conversation's prefix and
answers at once, in up to 2,048 tokens. Summaries used to reason at the
conversation's effort with 4,096 more output tokens, and at `xhigh` that
reasoning could take the whole 6,144-token budget: the summary came back
empty, the compaction was discarded, and the client asked again at its next
turn (see [Compaction threshold](#compaction-threshold)). The prompt after a compaction (summary
and kept quarter) is new and is prefilled cold. A
reasoning summary prefills the reasoning plus about 70 tokens and decodes up
to its budget. Both run for the Response, so both are in its `usage`: the
pinned `ResponseUsage` has no field of its own for them, so their prompt
tokens are added to `input_tokens` (their reused prefix to `cached_tokens`),
their generated tokens to `output_tokens`, and `total_tokens` stays
`input_tokens + output_tokens`; `reasoning_tokens` is the answer's own (a
`compaction_trigger` turn's usage is the summary generation's). A compaction discarded for not shortening the prompt still
counts. Risk: a summary is
lossy and is believed by every later turn; an exact string the model left out
is gone from the prompt, and a mistaken summary is not detectable from it.
Only kept items are exact.

Live check (2026-10-11, Apple M4 Pro, the one Bonsai 27B checkpoint, MTP on,
`--reasoning-key`, one bounded run). A 12-item agent history (developer
instructions, two user turns, three tool call/result pairs, assistant notes)
was asked, with tools disabled and thinking off, for six facts (failing test,
file and line, root cause, release tag, currency and config owner, open
work); the answer was checked for seven exact strings.

| Window | Prompt tokens | Facts kept | Answer time |
| --- | ---: | ---: | ---: |
| Short history, uncompacted | 789 | 7/7 | 7.0 s |
| Short history after `/responses/compact` | 789 (returned unchanged: the 313-token summary did not shorten it) | 7/7 | 0.8 s (cached) |
| Long history (160-line tool log), uncompacted | 2,869 | 7/7 | 20.8 s |
| Long history after `/responses/compact` (23.2 s, 2,376 in / 183 out) | 891 | 7/7 | 7.1 s |
| Long history, `context_management` threshold 1,000 (compaction then answer) | 1,014 | 7/7 | 45.8 s total |

Every compacted window kept the call/result pairs it kept whole, and the
compaction item came first in the `context_management` output. Limits of
this evidence: one task, one sample per row, facts the summary prompt asks
for by kind, and a test log of filler that summarizes well; it shows the
mechanism and its cost, not that information is preserved in general. One
`context_management` answer listed an invented open item ("the short list had
not been delivered"), an example of summary noise later turns believe.

A streamed `reasoning: {effort: "xhigh", summary: "concise"}` request ("Is 391
prime?") produced 280 reasoning tokens as `reasoning_text` deltas (0.68 s to
7.84 s), the answer by 8.38 s, then the `reasoning_summary_part.added`, text
deltas and `.done` events from 8.38 s to 9.54 s: the summary added 1.16 s to a
9.54 s response. The item's `summary` held one `summary_text` part ("I checked
divisibility by primes up to the square root of 391 and found it is divisible
by 17. ...") and its content stayed the raw `reasoning_text`.

### Compaction threshold

The server's default is 32,000 prompt tokens (`DEFAULT_COMPACT_THRESHOLD`
in `local-services/src/compaction.rs`), and the client settings above use
the same number. In the sweep below every threshold kept full quality at
`xhigh`; 32K was as fast as any compacting threshold at 64K and the fastest
at 128K, while no compaction slows every step of a long session (see the
128K rows).

Why compact at all: the [long-context eval](#coding-clients-and-long-context-coding-quality)
keeps recall at every size, but cold prefill falls from about 106 tok/s to
32K to about 54 tok/s marginal between 72K and 128K, so each step of a long
session costs more. Lookups at effort `none` also fall from 7/9 at 32K and
64K to 2/9 at 128K and 0/6 at about 250K, but that is a reasoning effect,
not a length one (see [why](#long-context-coding-quality)).

Method: `tools/context_eval.py --compact-at T` replays the eval repository as
an agent session, `read_files` call/output pairs of about 3K tokens, each one
`store: false` Responses request with a 1-token output. Before the next
request, when the previous one's `total_tokens` reached `T`, it compacts the
Codex way (a `compaction_trigger` turn), and once more before the questions.
Then it asks the nine questions with a `read_files` tool and up to four tool
rounds. The report adds the median prefilled tokens per read step, the stale
tokens re-prefilled (read steps that restored less than the previous prompt)
and the total wall time less those tokens at the session's prefill rate. Run
one size, seed and effort per GPU-lock hold
(`lockf -k /tmp/local-ai-gpu.lock`), with a fresh `HOME` per group so no group
reuses another's prompt cache, for example
`HOME=/tmp/ct/64k-32k python3 tools/context_eval.py --out OUT --server-bin
target/release/local-ai --sizes 64k --seeds 0 --efforts xhigh --compact-at
32k`, and `--report --out OUT` for the table. At effort `none` the model often
re-reads the same files until it runs out of tool rounds, so lookups there are
noisy; `xhigh` is the informative effort.

Results (2026-10-22/23, Apple M4 Pro 48 GB, seed 0, `xhigh`, each run alone
under the GPU lock). "Before" is main at `a4cf247`; "summary fix" adds
non-reasoning summaries; "both fixes" also skips the prompt-end GPU
checkpoint (see [Prompt cache](#prompt-cache)):

| Size | Compact at | Server | Session: requests, compactions, wall s | TTFT s, median / max | Compaction s, median | Final prompt | Recall | Lookup | Fix | Reads per question | Question wall s, median | Total wall s | Prefilled per read, median (stale) |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 64k | 32k | before | 23, 4, 1,347 | 29.2 / 252.0 | 198 | 39,600 | 3/3 | 3/3 | 3/3 | 0.3 | 18.0 | 2,232 | 3,119 (0) |
| 64k | 32k | summary fix | 23, 2, 910 | 29.5 / 115.7 | 59 | 9,093 | 3/3 | 3/3 | 3/3 | 0.7 | 19.6 | 1,129 | 3,134 (13,195) |
| 64k | off | summary fix | 23, 0, 770 | 33.8 / 47.4 | - | 69,997 | 3/3 | 3/3 | 3/3 | 0.0 | 24.0 | 989 | 3,107 (0) |
| 64k | 16k | both fixes | 23, 4, 885 | 27.0 / 117.5 | 56 | 6,832 | 3/3 | 3/3 | 3/3 | 1.3 | 20.2 | 1,060 | 3,134 (0) |
| 64k | 32k | both fixes | 23, 2, 871 | 30.2 / 122.1 | 65 | 9,093 | 3/3 | 3/3 | 3/3 | 0.7 | 18.4 | 1,035 | 3,131 (0) |
| 64k | 48k | both fixes | 23, 1, 807 | 29.6 / 169.4 | 65 | 30,436 | 3/3 | 3/3 | 3/3 | 0.7 | 22.8 | 1,032 | 3,119 (0) |
| 128k | 32k | both fixes | 44, 4, 1,646 | 28.7 / 140.5 | 73 | 24,456 | 3/3 | 3/3 | 3/3 | 0.8 | 21.5 | 1,860 | 3,180 (0) |
| 128k | 48k | both fixes | 44, 3, 1,700 | 31.5 / 153.2 | 46 | 16,547 | 3/3 | 3/3 | 3/3 | 0.8 | 23.8 | 1,897 | 3,177 (0) |
| 128k | off | both fixes | 44, 0, 1,905 | 44.1 / 67.3 | - | 139,107 | 3/3 | 3/3 | 3/3 | 0.0 | 29.9 | 2,206 | 3,174 (0) |

Findings:

- Before, at `xhigh`, a summary reasoned with up to 4,096 tokens of its own
  and could spend its whole 6,144-token budget: three of six compactions in
  the 64K run did, at 90–198 s each, returned no summary and were discarded
  (the history went on unchanged and the client asked again). Summaries now
  never reason (see [Compaction and reasoning summaries](#compaction-and-reasoning-summaries)):
  the two compactions took 59 s each (781 and 968 tokens) and the run fell
  from 2,232 s to 1,129 s with the same 9/9. Summaries that reach the
  2,048-token cap (twice at 16K, twice at 128K/32K) are cut off but still
  shorten the prompt, and the questions after them still passed.
- That run still re-prefilled 13,195 stale tokens: three read steps after
  the first compaction each restored an older checkpoint. The longest-prefix
  restore from round 1 did remove the round-1 stale reuse (the unfixed run
  had none); this was a second cause, now fixed (see
  [Prompt cache](#prompt-cache)), and the stale column is 0 in every later
  run.
- At 64K every threshold keeps 9/9, and total time differs by under 7%: no
  compaction is fastest (989 s), each compaction costs about a minute of
  summary plus a cold prefill of the summary and kept quarter, and facts
  compacted away are re-read when a question needs them (0.7 reads per
  question at 32K and 48K, 1.3 at 16K). 16K compacts four times and gains
  nothing.
- At 128K the trade-off turns: still 9/9 everywhere, but without compaction
  the median step waits 44 s (up to 67 s near 139K tokens) against 29 s at
  32K, and the run takes 2,206 s against 1,860 s at 32K (-16%) and 1,897 s at
  48K. 32K is as fast as 48K at 64K and faster at 128K, so it stays the
  default.
- Codex through `tools/agent_bench.py --clients codex-remote-compact`
  (Codex's own limit at 10,000 tokens, so most tasks compact; `xhigh`, one
  run each): main `a4cf247` solved 7/7 in 1,130 s with six compactions of
  22–102 s (one used the whole 6,144-token budget); this build 7/7 in 768 s
  with three of 12–17 s, 97% of each summary prompt served from the cache.
  At effort `none` the summary request is the same on both builds (it never
  reasoned there); two single runs (5/7 in 2,437 s and 7/7 in 900 s)
  differed the way two runs of one build do (the baseline's: 6/7 and 7/7).
  Codex's 7,000-token limit from `client_compat`, below a whole task's
  needs, made it compact after every file read and fail.

### Temporary background responses

A background Response with `store: false`, or with `store` omitted on a
server without `--response-store`, is kept only temporarily, approximating
OpenAI's documented retention of roughly 10 minutes. (`store: true` without a
store is still refused, and a foreground `store: false` response is still not
kept at all.) At startup the server creates a fresh private store for these
in a randomly named directory under the system temporary directory, created
`0700` and checked to be a real directory owned by the server's user; it is
never shared with another server or reused. Its records and journals are the
ordinary response-store files, in plain text with the raw reasoning, so this
is short-lived owner-only local data on disk, not memory-only or encrypted
storage. The configured `--response-store` is never touched: an ID is
answered from the temporary store exactly while it holds it.

Everything else follows the background lifecycle above: the `queued`
response, `in_progress` and terminal states, `GET`, `input_items`, cancel,
`DELETE`, and journaled `stream: true` events with
`?stream=true&starting_after=N` resumption, all reporting `store: false`. A
running job never expires. Its 10-minute retention starts once it reaches a
terminal state; afterwards it is deleted and is the same 404 as an unknown ID.
An expiry thread sleeps until the nearest deadline (or until a new, earlier
one arrives) rather than scanning, and any access past a deadline also expires
the response on the spot; if that thread cannot be started, the server warns
and expiry happens only on access. `previous_response_id` naming a temporary
response is refused with 400 instead of being looked up in the durable store.
Graceful shutdown cancels unfinished jobs like durable ones and then removes
the whole directory, so nothing temporary survives a restart. A crash or
`SIGKILL` skips that removal and can leave the directory behind in the system
temporary directory; the next start creates a new one and never reads it. If
the private store cannot be created, the server warns at startup and refuses
these requests.

### Native services

`serve` always opens one durable store at
`~/Library/Application Support/local-ai/services` (an owner-only SQLite
database in WAL mode plus file blobs, all plaintext) and refuses to start if
it cannot. There is no flag and no alternative backend. The store is separate
from the Responses store. Storage, limits, crash safety and
the native API, including the workspace search that has no HTTP route, are in
[local-services/README.md](../local-services/README.md). Routes sit behind
`--api-key` and are also served over HTTP/3. Each object belongs to the
project whose key created it, and other projects' are the same 404 as a
missing ID ([ADMIN.md](ADMIN.md#limits)). Service CRUD uses a bounded
blocking pool; conversation reads and appends run on the existing Response
preparation and generation blocking tasks, never on runtime workers.

- **Conversations.** Create, retrieve, update metadata, delete, and list, add,
  retrieve and delete items. JSON bodies and query strings are strict:
  unknown or repeated fields are a 400. Items cover text messages, function
  calls and outputs, and reasoning. Any other item type is refused.
  `include` accepts only `reasoning.encrypted_content`, which is a no-op.
  Deleted conversations become unreachable, but their items stay on disk.
- **Responses `conversation`.** A Response may name a conversation by ID or
  `{"id"}`, but not together with `previous_response_id`. The conversation's
  items, read as one snapshot, are placed before the new input, and the
  Response echoes `conversation`. When the Response completes or ends
  `incomplete`, its new input and output are appended once, keyed by the
  response ID. The append happens only if the conversation is still at the
  version the prompt was built from. Otherwise, or if the append fails, the
  Response fails and nothing is appended. Failed and cancelled Responses
  append nothing. Background Responses append once, at their end.
  `input_tokens` counting reads the conversation without changing it.
- **Files and Uploads.** Multipart bodies stream into an owner-only spool and
  are never held in memory. A file is at most 512 MiB (200 MiB for `batch`)
  and a part at most 64 MiB; anything larger is a 413. Completion verifies an
  optional `md5`. At most 4 multipart transfers and 16 downloads run at once;
  past that the answer is a 503 with `Retry-After: 1`. An upload that stalls
  or a download whose client stops reading gives its slot back (see
  [`--stall-timeout`](#generation-requests)). Expired objects are a
  404, and changes to an expired or cancelled upload are a 410. Expiry is
  enforced on access. Bytes are purged physically at startup, every 15
  minutes by the vector store indexer, or through the native
  `purge_file_storage`; vector store searches leave an expired File out at
  once. A `purpose` is stored and echoed only; it
  enables no processing.
- **Batches.** These run `/v1/responses` and `/v1/chat/completions`, with
  `completion_window: "24h"`, on the loaded model, whatever generation model
  name the lines share; the legacy
  `/v1/completions` is refused. The whole input file is validated before anything is stored.
  `output_expires_after` is refused, like any unknown field. One worker thread
  runs one line at a time through the same preparation and collector as the
  HTTP endpoints. Lines are stateless: `store: true`, `previous_response_id`
  and `conversation` fail their line. Batch lines share the engine queue with
  interactive requests. A full queue is retried with a 50 ms to 2 s capped
  backoff. Other than that backoff, the worker does no polling: it scans for
  runnable batches at startup and after each create or cancel. A per-batch OS
  lock gives one server ownership of a batch. A restart reclaims batches with
  free leases and resumes their unsettled lines. A line interrupted before
  settling is generated again. Result files have stable `batch_output` IDs.
- **Vector Stores.** Stores, store files, file batches (create, retrieve,
  cancel, list files), parsed file content and `search`, with the contract's
  pagination, expiry, file status lifecycle, `auto`/`static` chunking,
  attributes and filters. Ranking is **lexical BM25** over one persistent
  Tantivy index derived from the SQLite rows (scores `s / (1 + s)` of the
  BM25 score) or, with the server's EmbeddingGemma 2 encoder,
  **semantic** for `ranker: auto`: cosine similarity of chunk and query
  embeddings, which beat BM25 and its fusion with embeddings in the
  measurements (see [EMBEDDINGS.md](EMBEDDINGS.md#vector-stores));
  `ranker: none` stays BM25.
  `rewrite_query: true` and the `default-2024-11-15` ranker
  are a 400. Attaching answers `in_progress` and wakes one indexer thread;
  it never polls, and a restart resumes files left `in_progress`. A
  garbage outbox keeps the index consistent with the database across
  crashes and deletions; startup rebuilds an index that disagrees. One server
  owns a data directory's index; a second server on the same directory
  serves everything except vector stores. Details are in
  [local-services/README.md](../local-services/README.md#vector-stores).
  Vectors are stored in SQLite under the same outbox protocol and backfilled
  in the background for stores indexed before the model was installed.

Resuming background Responses after a crash is the server's job (see
[Server API](#server-api)), not these services'; batch
resumption is a separate mechanism. Compaction's policy (when to compact,
what is kept) lives here; its summary generation is the server's. They
provide no training or change to the model.

### Decisions

`POST /v1/decisions` is served from the required artifact in
`models/decisions-v2` (`head.safetensors`, `gate.safetensors`,
`calibration.json`); without it the server does not start. At startup the server checks the
artifact's canonical-JSON content hash, the head and gate file hashes it
pins, that its pre-registered acceptance passed and that its declared scope
is the one the server implements; any mismatch stops it before the model
loads. A `predicate` question is answered only when it is one of the declared
questions (CodeReviewer "Does this change need a reviewer comment?",
CommitPackFT "Does this commit message describe this diff? Message: "…"")
over a text input starting `Diff:` or `Context before hunk:`; it then runs
one [native decision](../local-engine/README.md#native-decisions), and is
answered with the calibrated `probability` of Yes unless the prompt exceeds
2048 tokens, the gate (PCA Mahalanobis distance of the `Decision:` hidden
state, fitted on training rows) puts it out of distribution, its family has
no threshold (CodeReviewer: always refused) or its top-label confidence is
below the family's threshold. Every other question, and every question about
a message or image input, is answered `{"type":"refusal"}` without GPU work.
Results and limits are in [DECISIONS_EVALUATION.md](DECISIONS_EVALUATION.md)
part 3; in short only commit-message match is answered, which says nothing
about code review or coding quality. This is not a production judgment
service.

Questions run one after another over the same input, each taking one slot of
the engine's bounded FIFO queue and running only once no generation is
active, so other requests interleave between questions; a long input delays
generations queued behind a running capture. At most eight decision requests
are admitted at once; more get 503 with `Retry-After: 1`, and a full engine
queue gets the usual queue-full 503. A client that disconnects drops its
pending decision, which cancels it, queued or running; shutdown refuses new
requests and ends waiting ones with 503, cancelling their decisions the same
way. These waits are futures woken by the engine or by shutdown, not polling.
Real-model native and HTTP checks reproduced stored-feature probabilities
bit-for-bit on four rows from both training tasks and both option orders,
with and without MTP; queue ordering, cancellation, and generation/cache
parity after a decision also passed. Capture measured about 115–130 tokens/s
on the M4 Pro, and later generations wait behind it. These checks establish
implementation parity, not general judgment quality or calibration.

### Concurrent requests

Decode is bound by trit-decode ALU work per weight byte, so one projection
pass over several activation rows costs little more than over one. With two or
more requests running, each engine step decodes every request in a single
pass: every projection (QKV, gates, FFN, output head) reads its weights once
for all rows, through the same multi-row kernels speculative verification
uses. Only what reads a request's own state runs per request, against that
request's buffers: the convolution and gated-delta recurrence of the 48
recurrent layers and the K/V append and attention of the 16 full-attention
layers. Each request owns its recurrent state, convolution history, K/V caches,
position and MTP-head caches; one set is resident in the model and the others
are parked, and a swap exchanges buffer handles, never bytes. Requests join
between steps and leave when they finish or are cancelled. A failure that
belongs to one request ends that request alone: every sequence's caches grow
for its rows before the step's GPU work, so running out of memory fails that
sequence; a sequence whose drafting or selection fails leaves the pass or
commits only its seed row; a failed batched pass ends only the requests whose
state it wrote, and a request still waiting to prefill keeps its place.

A request alone runs exactly the single-sequence rounds it always ran, with
speculation; its text and speed are unchanged. Taking speculative rounds in
turn instead of batching measured 32.7 tok/s aggregate at two streams against
36.8 batched, so batching starts at two.

#### Prompts prefill inside the decode steps

Admission only restores a request's reusable prompt prefix. Its prefill then
runs inside the engine's steps, one prompt per step: the one with the fewest
tokens left. Its first chunk always runs in a pass of its own, which first
asks the prompt cache again for a longer prefix than admission restored and
restores that one when it gains at least one token in 64 of it: requests
admitted together that share a long uncached prefix (sub-agents fanned out
at once) prefill one after another, and each later one starts from the
snapshot the first wrote at the end of that prefix instead of prefilling it
again. While that write is still in flight the later one skips its prefill
each step another request decodes, so their tokens keep coming, and waits for
the write only when none does; admission never waits for it (see
[Concurrency measurements](#concurrency-measurements)). A prompt whose disk
snapshot is still being read back (see [Prompt cache](#prompt-cache)) is
held out the same way: the step's prefill goes to another prompt, or is sat
out while others decode. A prompt of at most 128 tokens to prefill (a typical
chat turn) goes whole, in a pass of its own after the step's decode round. A
longer one, beside two or more decoding requests, prefills inside their
batched pass: its chunk's rows are stacked after theirs, run every layer with
them, advance its own recurrent state, convolution history and K/V caches in
place as a prefill block does (the same row-offset kernels a verify block
uses, here reading and writing the sequence's own buffers), and are not
projected to logits; its MTP head ingests them, K/V rows and committed
hidden, in the same submission, as a prefill block's head ingestion does. The
chunk fills the pass's rows to a multiple of eight from at least 24 prompt
tokens (28 beside four streams, 24 beside eight), since a pass costs about
the same for every row count of an eight-row group. A chunk stops short of
each prompt-cache milestone (the reusable boundary checkpoint and snapshot,
the penultimate-token checkpoint and pinned snapshot) and of the prompt's
last token. Those, and every chunk beside a single request taking its solo
rounds, prefill 32 tokens in a pass of their own after the step's decode
round, which handles the milestone (the GPU tier is claimed for the
prefilling sequence's buffer set, so a checkpoint never describes another
request's state) and takes the first sample. With nothing decoding, a step
prefills one block (512 tokens, about four seconds), so a request arriving
behind a long prompt is admitted, and can start its own shorter prefill,
within a block; each block is its own GPU round trip either way, so this
costs the prompt nothing (measured with 128-token blocks, before blocks grew
to 512: 6.6K tokens alone, 51.9 s at one block a step against 52.1 s at
four). A cancellation between two chunks ends the request with no tokens.
Its sequence's K/V rows still hold the prompt up to where it stopped, so
the GPU checkpoints of prefixes of that stay and only those past it go: a
retry restores the checkpoint the cancelled request restored. A cancelled
request used to clear the whole GPU tier, while the same client disconnect
seen by the stream instead of the token (a stop) kept it.

The chunk size trades the running requests' inter-token gap against the
prompt's own time to first token. A prefill block costs about 9.2 ms a row at
every size from 8 to 128 rows (75, 146, 289, 441, 589 and 1,181 ms at 8, 16,
32, 48, 64 and 128 rows), so chunking costs no prefill efficiency; only the
decode rows in between lengthen the prompt's prefill. Folding the chunk into
the decode pass removes most of what those rows cost, because a pass of 24
rows or more costs about 9.4 ms a row however many of them are decoding
(M4 Pro, `prefill_fold_timings`):

| Decoding rows + chunk | One pass | Decode pass, then the chunk alone |
| --- | ---: | ---: |
| 2 + 38 | 369 ms | 422 ms |
| 4 + 36 | 377 ms | 449 ms |
| 4 + 28 | 301 ms | 370 ms |
| 4 + 20 | 237 ms | 318 ms |
| 8 + 32 | 387 ms | 395 ms |
| 8 + 24 | 314 ms | 325 ms |

Four streams' rows ride almost free (72 ms of 449 saved at a 36-token
chunk); eight streams' rows fill a group of their own, so folding saves
little there. Of the two chunks that fill a group beside four streams, 36
and 28 tokens prefill at the same rate (95.5 and 93 tokens/s), and 28 keeps
the streams' tokens coming 77 ms sooner, so the chunk fills the nearest group
from 24. Four 600-token streams with a 9.2K-token prompt arriving 4 s in, M4
Pro, greedy, cold prompt cache, MTP head loaded:

| Build | Long prompt TTFT | Streams' ITL p50 / p99 during its prefill | Streams' tok/s during it | Aggregate tok/s |
| --- | ---: | ---: | ---: | ---: |
| Chunk 32 as its own pass (`f088159`) | 115.2 s | 0.40 / 0.44 s | 10.0 | 17.0 |
| Folded, 36-token chunk (40 rows) | 105.0 s | 0.41 / 0.46 s | 9.8 | 18.0 |
| **Folded, 28-token chunk (32 rows)** | 105.8 s | 0.32 / 0.36 s | 12.5 | 19.0 |

Earlier measurements of the separate chunk, on the same scenario without
the folded pass:

| Build | Long prompt TTFT | Streams' ITL p50 / p99 during its prefill | Streams' tok/s during it |
| --- | ---: | ---: | ---: |
| Prefill as one pass | 88.5 s | 88.5 s stall | 0.09 |
| Chunk 48 | 103.6 s | 0.54 / 0.59 s | 7.3 |
| Chunk 32 | 111.4 s | 0.39 / 0.41 s | 10.3 |
| Chunk 16 | 136.3 s | 0.24 / 0.29 s | 16.2 |
| Earlier kernels, one pass | 96.9 s | 97 s stall | 0.04 |
| Earlier kernels, chunk 48 | 123.0 s | 0.64 / 0.68 s | 6.2 |
| Earlier kernels, chunk 128 | 103.3 s | 1.43 / 1.51 s | 2.9 |

#### Concurrency measurements

`tools/serve_concurrency.py` starts the server with a scratch HOME (cold
prompt cache) and streams greedy Chat Completions with reasoning off. A gap
is the wait between two engine steps that streamed a request text (chunks
within 5 ms of each other count once). M4 Pro, 2026-10-22, `b09e22f` against
main `a4cf247` back to back, one run each, with no other GPU job running
(every job held `lockf -k /tmp/local-ai-gpu.lock`). Cells read `b09e22f` /
main.

| Streams, 300-token answers | Aggregate tok/s | Per stream | TTFT p50 | Gap p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 34.8 / 34.9 | 36.5 / 36.5 | 0.39 / 0.39 s | 49 / 78 ms, 49 / 79 ms |
| 2 | 50.1 / 50.0 | 53.1 / 53.0 | 0.59 / 0.59 s | 49 / 302 ms, 49 / 306 ms |
| 4 | 59.2 / 59.2 | 33.5 / 33.6 | 0.99 / 0.99 s | 78 / 461 ms, 78 / 464 ms |
| 8 | 74.2 / 74.0 | 14.6 / 14.5 | 1.62 / 1.63 s | 94 / 467 ms, 95 / 472 ms |

The scheduling changes do not touch these steps, and the two builds agree
within 0.5%; 43 of 44 outputs were byte-identical across builds (the other,
one of four streams beside a long prompt, moved at a near-tie, as below).
The earlier, possibly contended run of the same table read 32.0–35.4,
47.3–50.6, 55.5–59.8 and 73.8–74.8 tok/s aggregate. With N-1 streams decoding
512-token answers, a 6,736-token prompt arrives 3 s in:

| Streams | Long prompt TTFT | Its prefill | Decoding streams before / during it | Gap p50 / p99 during it | Longest stall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (alone) | 52.9 / 51.0 s | 127 / 132 tok/s | | | |
| 2 | 75.6 / 75.9 s | 89 / 89 tok/s | 42.1 / 5.3 tok/s (both builds) | 356 / 392 ms, 359 / 399 ms | 0.44 / 0.44 s |
| 4 | 72.0 / 72.1 s | 94 / 93 tok/s | 46.6 / 10.1, 46.3 / 10.0 tok/s | 309 / 327 ms, 309 / 327 ms | 0.33 / 0.38 s |
| 8 | 86.0 / 86.2 s | 78 / 78 tok/s | 18.8 / 22.7, 18.8 / 22.6 tok/s | 316 / 337 ms, 318 / 336 ms | 0.56 / 0.55 s |

A one-token request arriving while that prompt prefills alone waits 1.27
/ 1.19 s (p50) beside one other request and 3.3 / 3.1 s at most beside seven.
The long prompt alone prefills 4% faster on main (512-row blocks); beside
decoding streams it prefills in chunks folded into their passes, which the
block size does not affect.

No stream waits more than about half a second for its next token while a
long prompt prefills; without chunking the same prompt stalled every stream
for its whole prefill (the first table above). What fairness costs is
throughput, on both sides: beside a lone stream the prompt takes a 32-token
pass after each of that stream's speculative rounds (about 300 ms against
about 40), so the stream keeps an eighth of its speed and the prompt 70% of
its own. Folded passes beside four or eight streams sit close to what time
slicing at full block efficiency would give both.

Sub-agents fanned out at once usually share a long prefix that no earlier
request cached (their instructions and tool schemas). Requests 0.3 s apart
sharing a 6.8K-token system prompt, each with its own question
(`--phases shared --streams 2,4`, same conditions):

| Build | Two: TTFT first / mean / last, all done | Four: TTFT first / mean / last, all done |
| --- | ---: | ---: |
| `b09e22f` | 53.1 / 80.8 / 108.4 s, 109.9 s | 53.1 / 133.4 / 213.9 s, 215.1 s |
| main (prefix refresh) | 51.4 / 51.6 / 51.8 s, 54.1 s | 51.5 / 52.5 / 53.5 s, 55.6 s |

Each request restored what the cache held at admission, nothing, and
prefilled the shared prefix again, one after another. Asking the cache again
at a prompt's first prefill step finds the snapshot the first request wrote
at the end of the system turn, so the others prefill only their question.
Their text is byte-identical. Running a prompt's first chunk in a pass of its
own, so it can restore first, costs one unfolded step per long prompt (the
longest stall beside three streams 0.38 s against 0.33 s, the prompt's TTFT
unchanged at 72.1 s); the different pass shapes moved one of those three
streams' 512 greedy tokens to another continuation at a near-tie.

Not measured yet: whether Codex 0.163 (the `multi_agent_v1` tools, up to six
sub-agents, spawned one call at a time but running concurrently) and opencode
1.19 (several `task` calls in one turn run concurrently, no cap) fan out
against this server (`tools/client_compat.py --scenarios
codex-fanout,opencode-fanout`), and a lone decoding stream beside a long
prefill with a decode-only round between its prompt chunks (modelled: that
stream 5.3 to about 9 tok/s, the prompt 89 to about 86 tok/s), the one case
where chunked prefill sits well below what time slicing would give both.

#### Speculation in a batch

A batched step verifies drafts too. Each sequence contributes its seed row and
any drafts as consecutive rows of one stacked block, so a pass of four
sequences with three drafts each is one 16-row pass. A sequence's verify rows
run its convolution and recurrence with the multi-row kernels at a row offset
(the same pipelines bound with byte offsets, so its rows compute what a block
of only those rows computes), reading its state and history and writing the
block's final ones to shared spares that nobody reads; the K/V rows are
appended at its position and attended with the block attention kernel at the
same offset. Every row is projected to logits and selected on the GPU; each
sequence then verifies its drafts exactly as a single sequence does. A
sequence's verify rows go on its own GDN tape, and its kept rows are folded in
by its next step, batched or alone, at the start of that step's kernels; only
a verify longer than eight rows (a long lookup draft) is replayed from the
start of the round, in one submission for all of them, as every partial commit
was before the tape (2.5-12.7 ms for 2 to 8 sequences). Positions advance by
the kept rows, and the kept rows' hidden reaches the MTP head through the same
per-request lag a batched plain step uses.

Drafts come from each sequence's suffix lookup, or from its own MTP head.
Head drafts are made for every drafting sequence together. Each sequence's
head caches and committed hidden stay where they are (the resident
sequence's in the model's head, the others' in their parked buffer sets), and
a draft round runs one stacked head pass per draft depth, one row per
drafting sequence: the head's matrices read their weights once for all the
rows, and each row's K/V append and attention run at its row offset against
its own sequence's head caches (the target's row-offset wrappers on F16
caches). Each row's next-token embedding gather and its exact top two run on
the GPU (`draft_embed_inverse_rows`, `draft_top2_rows_*`), each writing its
chain's next token where that chain's next row reads it; the host reads the
top twos after each depth and applies each chain's EOS and margin gate, and
rows whose chain stopped leave the next pass. The first pass also feeds each
head the rows it missed in batched steps, as K/V-only rows beside the
drafting rows (a lag too long for one pass is fed first, in K/V-only passes),
so no sequence is swapped in to draft. A sequence that does not draft for a
long time (eight seeds fill a pass, so eight streams seldom draft) has its
lag fed the same way before it would pass 512 rows. A lag used to grow until
4,096 rows and then give the head up for the rest of the request, so a long
reasoning request lost MTP for good after batching beside others, even once
it ran alone again (about 38 against 31 tok/s), and its 80 MB of owed hidden
rows stayed allocated; now the lag buffer stays at 10 MB, and after 552
batched rows a head caught up this way drafts exactly what a head that
ingested every row in prefill blocks drafts. A pass costs about 2.1 ms plus 0.5 ms
per drafting sequence (2.5 ms for one, 4.4 for three, 6.1 for eight, against
3.0, 8.9 and 23.5 ms for the same chains drafted one sequence at a time,
`batched_head_draft_timings`), so a full depth-3 round for four sequences
costs about 12 ms where it cost 60. Each chain is the one the sequence's solo
round would draft: over 24 chains of three sequences owing their heads 0 to
20 rows, one of them resident, every chain matched the solo catch-up and
draft token for token.

Whether a sequence's drafts are worth their rows is decided per step by a cost
model: measured pass time by stacked rows (32, 48, 64, 75, 78, 80, 82 and 84
ms for 1 to 8 rows, 154 at 16, 303 at 32, 602 at 64; about 75 ms per eight
rows past eight), plus 1.5 ms of commit per verifying sequence and, for head
drafts, 0.5 ms per draft and 2.1 ms for every depth the deepest head chain
reaches, against the tokens the drafts are expected to add (a geometric chain
from each sequence's running acceptance estimate). Pass time is not linear in
rows, so one sequence's drafts may pay only beside another's (at two streams
the third and fourth rows cost 16 and 11 ms, the fifth to eighth 2 to 3 ms):
candidates are added in order of expected tokens per row, each at its best
depth whether or not it pays yet, the best prefix is kept, and then each
choice is revised given the others. A chain is one to three drafts, so one
rejection moves an estimate far; a head that did not draft in a step moves
its estimate an eighth of the way back to the 0.6 prior, so a run of
rejections does not stop it drafting for good. Beside a prompt chunk folded
into the pass, the decoding rows and drafts are charged only what they add to
the chunk's own pass, where every row costs its full 9.4 ms; drafts seldom
pay there.

Rows five to eight cost little more than four, so confident lookup drafts
ride almost free beside two to seven sequences, and so do head drafts now
that a round costs one head pass per depth for all of them. Prose requests
arriving together, 160 greedy tokens each, decode-only throughput in the
engine, two runs (`batched_prose_throughput`; head drafts were accepted
71-75% of the time when forced every step):

| Streams | No head drafts | Cost model | Head drafts every step |
| ---: | ---: | ---: | ---: |
| 2 | 39.0, 40.2 tok/s | 42.1, 42.1 tok/s | 44.2, 46.5 tok/s |
| 4 | 48.6, 50.3 tok/s | 60.8, 61.9 tok/s | 60.5, 62.2 tok/s |
| 8 | 81.4, 85.0 tok/s | 82.6, 84.7 tok/s | 67.9, 68.9 tok/s |

At eight streams the seeds fill an eight-row group and any draft starts
another, so the model drafts only as streams finish. At two, drafting every
step does better than the model: one rejected chain drops a stream below the
acceptance at which two streams' drafts pay together, until its estimate
recovers; over HTTP, where prefill and the first stream's solo rounds count
too, neither gain at two streams stands out of the runs' spread (see below).
Before batched drafting, head drafts cost 15 ms of sequential drafting per
chain and rarely paid (at four streams, one chain in
the free rows: 54.5 against 52.9 tok/s); on the earlier kernels (8 rows 92
ms, 16 rows 172, 32 rows 337) they never did.

On copy-edit prompts (rename a variable in a 17-line function, edit a 30-line
config; about 300 prompt tokens, 400-token limit) arriving together, aggregate
throughput including prefill rose from 31.7 to 37.3 tok/s at four streams and
from 28.5 to 34.0 at two, against the same kernels without batched
speculation or chunked prefill. With batched head drafting and folded prefill
it measured 34.2, 34.7, 37.9 and 41.7 tok/s at one, two, four and eight
streams, against 32.8, 32.9, 34.9 and 38.0 for `f088159`.

Greedy output per request matches a request run alone up to the engine's own
near-ties. Rows of one batch are independent of each other, and against
single-row decode, teacher-forced over 16 steps of four sequences, mean
next-token KL is 4.5e-7 to 5.1e-7 (worst 2.8e-6) with top-1 agreement on every
step. A batched verify of three sequences (four, one and three rows) tracks
each sequence's own verify block at KL at most 2.1e-7 with every argmax equal,
and after a partial, a plain and a full commit the next step tracks each
sequence decoded alone from the same kept rows. Over eight 300-token chat
prompts (four prose, four copy-edits), every 2-, 4- and 8-stream output was
byte-identical to that prompt run alone, on both kernel generations; the
previous build split one 8-stream output at a near-tie (character 1,206).
With batched head drafting and folded prefill the eight prompts, run alone
and in groups of two, four and eight arriving together (so every group size
covers prose and copy-edits), were again byte-identical to each prompt run
alone, and the prompts run alone to `f088159`. A prompt chunk prefilled
inside a batched pass leaves its sequence where prefill blocks leave it: the
next token's logits are bitwise those of the reference (KL 0), with the
sequence parked or resident, and its head drafts the same chain; the
decoding rows beside the chunk track the same step without it. A long prompt
prefilled beside two decoding requests with the head loaded generated its
own tokens, and its head then proposed and had accepted exactly as many
drafts (6 and 5 in 4 rounds) as after prefilling alone.

Measured on an M4 Pro, chat requests with distinct prose prompts arriving
together, greedy, 300 output tokens each, thinking off, cold prompt cache,
after two warm-up requests, streamed over HTTP (TTFT includes this harness's
0.4-0.5 s for a lone request); the previous build is `f088159`, measured the
same way, interleaved; aggregate throughput is two runs each:

| Streams | Aggregate tok/s | TTFT mean / worst | ITL p50 / p99 / max | Footprint | Previous: aggregate tok/s, ITL p50 / p99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.4, 33.0 | 0.41 / 0.41 s | 45 / 82 / 87 ms | 0.78 GB | 32.5, 33.7; 46 / 80 ms |
| 2 | 37.1, 40.4 | 0.69 / 0.95 s | 53 / 101 / 529 ms | 0.85 GB | 37.7, 39.2; 50 / 67 ms |
| 4 | 56.4, 56.2 | 1.18 / 1.93 s | 85 / 141 / 503 ms | 1.40 GB | 45.8, 47.4; 81 / 134 ms |
| 8 | 73.5, 75.2 | 2.15 / 3.88 s | 93 / 486 / 497 ms | 2.18 GB | 73.1, 74.8; 95 / 503 ms |

The gain is batched head drafting at four streams (+21%; the same build
without head drafts in a batch measured 46.9 and 47.0, and with head drafts
forced every step 54.2). At two streams it is within the runs' spread on the
server (forced drafts: 38.4), and at eight the cost model rightly drafts only
as streams finish (forced drafts: 54.7). The earlier build, `18c274d`,
measured 34.4, 39.0, 46.9 and 69.0 tok/s at one, two, four and eight
streams.

The engine-only step time is 33 ms for one sequence and 49, 64, 78, 80, 82, 85
and 89 ms for two to eight (89 tok/s at eight); the server figures above add
prefill, sampling and event delivery. Against `18c274d`, most of the gain at
eight streams was the 5- to 128-row kernels; on them alone, prompts admitted
as one pass each measured 32.8, 37.8, 45.9 and 75.0 tok/s with TTFT mean 0.41,
0.62, 1.02 and 1.85 s (worst 0.41, 0.83, 1.64 and 3.34 s). Requests arriving
together now start decoding one by one as each prompt is prefilled, with a
decode step between prefills: the first requests stream while the later
prompts prefill (the ITL p99 is those prefills, against a 3.0-3.6 s stall
before), and the mean TTFT of eight simultaneous short prompts is about 0.4 s
later than admitting them back to back on the same kernels.

Batched head drafting keeps one row of draft logits per sequence (8 MB at
eight) and its own small token and top-two buffers; a prompt chunk folded
into a pass adds its rows to the pass's activations but no logits rows.
During the long-prompt scenario above the footprint peaked at 1.91 GB
against 2.30 GB for `f088159`.

Each extra stream costs its own state: 81 MB of F16 recurrent state and
convolution history, 34 KiB of Q8 K/V per token (initially 1,024 tokens,
36 MB) plus 4 KiB of head K/V per token, and 20 KiB per batched token of
hidden rows owed to its MTP head (at most 512 rows, 10 MB). A verifying step keeps each recurrent
layer's raw QKV rows, decay and beta for its rows (about 2 MB per stacked row,
allocated on first use at the next power of two of the rows, up to 126 MB at
64 rows) plus one spare state, and batched logits grow to the stacked rows
(1 MB each). The measured footprint grew by about 200 MB per stream at these
lengths. Admission shares the context the memory policy sized for one
sequence: a request joins only while the running requests'
prompt-plus-`max_tokens` reservations, plus each extra sequence's fixed state
expressed in K/V tokens (about 2,340), fit in it; otherwise it waits for a
running request to finish. A request whose own prompt and `max_tokens`
exceed the context fails as soon as it reaches the head of the queue,
rather than holding the requests behind it until the engine is idle. At most
eight decode together, in at most 64 stacked rows.

KFlash leaves batched passes as they are: they write no taps and verify only
head and lookup drafts (see [KFlash cycles](#kflash-cycles)). With the
drafter installed, `tools/serve_concurrency.py --phases decode` (300-token
greedy essays, back to back against `d25b277`) measured 34.9, 49.4, 60.2
and 80.4 tok/s aggregate at 1, 2, 4 and 8 streams against 35.4, 50.5, 60.2
and 76.5; an earlier pair gave 38.6, 49.7, 60.8 and 77.1 against 35.6,
50.6, 60.4 and 77.1. At two streams both requests finish their batched
passes together with main's (4.2 s for the first), and the second's solo
essay tail, where the head's chain leads and the tree's trial cycles cost
a little, takes 12.15 against 11.87 s. Every text matched main's but one
per pair, the earlier pair's one-stream essay and the later pair's eighth
stream at eight, each a near-tie resolved the other way: the batched
passes are main's, so solo cycles (trees in that essay's case) took it.

## Resource policy

All choices and reasons are reported in startup JSON.

- **Model:** a library caller's explicit `Engine::open_model` path; otherwise
  the pinned path in the working directory, beside the executable, or under
  `~/Library/Caches/local-ai/models`. The commands have no model flag.
- **Context:** largest value up to 262,144 tokens whose fully grown model state
  fits 90% of Metal's recommended working set.
- **K/V:** Hadamard-rotated Q8 for the target, always (see
  [K/V cache](#kv-cache)); the MTP head's one-layer cache is F16.
- **Prefill:** on Apple M4 and later, blocks of 192 to 1,024 tokens run as
  two halves of up to 512 rows, the Neural Engine computing every layer's
  FFN and the leading rows of each mixer output projection while the GPU
  runs the rest of the mixers ([Neural Engine prefill
  FFN](#neural-engine-prefill-ffn)); elsewhere, and from the first block
  whose Neural Engine work fails (that block reruns on the GPU), fixed
  512-token chunks on the GPU. Metal 4 kernels where supported. Beside two or
  more decoding requests, a long prompt prefills 24 to 31 tokens per step
  inside their batched pass, and 32 per step in a pass of its own beside
  one, on the GPU. The policy JSON reports `neural_engine_prefill`.
- **Speculation:** suffix lookup, gated depth-3 drafting with the mixed
  ternary/int8 MTP head at `models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin`, and
  KFlash trees with the DFlash2 drafter; both artifacts are required.
- **Vision:** image input with the required BF16 projector beside the GGUF
  ([image input](#image-input)); `vision_reason` names it.
- **Reasoning:** the checkpoint's xhigh thinking for every request that asks
  for no other effort (`reasoning_effort`, `reasoning.effort`,
  `chat_template_kwargs.enable_thinking`); no flag turns it off.
- **Prompt cache:** GPU checkpoint count (at most four) derives from
  working-set headroom. The disk budget is the smallest of the checkpoint's
  own size, an eighth of free disk and a quarter of physical memory; the
  startup reason prints every term.
- **Host snapshots:** disabled when the startup write probe measures at least
  500 MiB/s, making the integrity-checked disk tier the first durable tier;
  otherwise capped at one quarter of remaining headroom or 4 GiB.
- **Server:** queue capacity is eight. HTTP/3 is enabled only when its
  certificate and key are discovered.

## Lossless speculation

Speculation changes scheduling, not the model: drafts are checked by the
target model, and only target-approved tokens are committed. It is lossless up
to floating-point rounding, as in other engines: a verify block runs the
target over several rows with multi-row kernels whose summation order differs
from single-row decode (logit differences around 1e-6 relative), so a greedy
near-tie can resolve differently with and without speculation. Measured: the
default settings matched plain decoding token for token on the six
benchmark prompts; one server prompt split at character 624 ("of entries"
against "of all entries"), deterministically, the same way on every run.

The suffix store finds a matching generated-token suffix and proposes the
known continuation. Draft depth adapts to match length and remaining context.
Each of the last eight requests (at most 131,072 tokens in all) is indexed
once, every 12-token anchor by a rolling 64-bit hash whose matches are
confirmed token by token, and shared with every later request, so admitting a
request indexes only its own prompt. Rebuilding one map of every anchor's
tokens per admission (twice, counting an empty placeholder) cost 10-19 ms and
a 2-3 ms drop on the engine thread for 128K remembered tokens and a 30K-token
prompt, M4 Pro; the shared index costs 0.2 ms for the prompt (1.2 ms the
first time it indexes the remembered requests) and, from the structures'
sizes, about 28 bytes per anchor against about 160. 72,000 lookups over
random token streams with repeats (anchors of 1, 3 and 12) drafted exactly
what the rebuilt map drafted, and on a real edit (rename a field in 35 quoted
lines of this repository's Rust, 265 greedy tokens) both builds ran 18 lookup
rounds proposing 218 drafts and accepting 147 (67.4%), with identical text,
at 57.06 and 57.31 tok/s best of two (lookup 0.28 and 0.23 µs per token).
The head drafts up to three tokens. A gate
avoids head work where measured acceptance does not repay its cost.

Both were re-swept on the current kernels (M4 Pro, six prompts including one
thinking, 300 greedy tokens, best of two, interleaved), drafting while the
head's top logit leads the runner-up by at least the margin:

| Depth | Margin 0 | 2 | 3 | 4 | 5 | 6 | 8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.2 | | | | | | |
| 2 | 36.4 | 38.0 | 38.0 | 38.1 | 38.0 | 38.1 | 37.6 |
| 3 | 33.5 | 37.8 | 38.1 | **38.6** | 38.4 | 38.3 | 37.5 |
| 4 | 30.6 | 37.6 | 38.2 | 38.4 | 38.3 | 38.3 | 37.6 |

tok/s geomean against 31.2 without speculation (depth 1 has no chained draft
for the margin to gate). Depth 3 at margin 4 (79%
acceptance, 1.57 drafts per round) stays the default; the optimum is flat
across depths 2-4 and margins 3-6, and stricter margins for later drafts (3/5,
4/6 at depth 3; 2/4/6, 3/4/6, 4/5/6 at depth 4) measured 38.1-38.5. The default
emitted the plain-decoding tokens on all six prompts. Some non-default
settings did not, each on one prompt: depth 2 or 4 ungated and depth 4 at
margin 2 or 4 resolved a near-tie the other way (for example `So,\n` against
`So, the` at token 178 of the arithmetic prompt), so verify blocks of different
row counts are not bit-identical to single-row decode in every case.

Verifying the head's runner-up as a second branch was measured and rejected.
The idea: rows five to eight cost little more than four, so a round could
also verify a forked chain (the seed, the accepted prefix, the head's
second-best token where the chain is weakest, and its continuation) as a
virtual sequence from the same committed state, and keep that branch when the
target picks the runner-up. On the six prompts (721 rounds, 233 rejected
drafts), the target's token at a rejected position was the head's second
choice 42.9% of the time (71 of 147 at the first draft; 31-70% per prompt).
A hit is worth the branch's bonus token plus its accepted continuation, 1.73
tokens on average gated at margin 4, so verifying every hit's branch at no
cost would add 0.24 tokens to the 2.24 a round now yields: +10.7% at most.
Nothing can be verified for free, though. 97% of rejections are of the
chain's last, low-margin draft, and 59% of rounds are two-row blocks, where
rows cost about 13 ms each up to four (45.7 / 59.6 / 72.4 ms measured at two
/ three / four rows); the branch must repeat the seed and accepted prefix,
since recurrent state cannot fork mid-block, and even at a near-tie (margin
under 0.5) the runner-up is the target's token in only 22% of rounds.
Priced on the trace with the measured block costs, branching at the gated
draft lost 20.6% (no continuation) to 29.3% (two continuation drafts),
branching only four-row rounds into eight rows lost 0.9%, and every other
placement lost 13-28%. In batched serving the flat five-to-eight band is
sometimes left over beside other sequences: with the scheduler's pass
costs and a margin-conditioned hit estimate choosing branches only when they
raise expected tokens per second, two sequences gain at most 0.9%, one or
three sequences 0.1% or less, and four leave no room, before the extra K/V
ordering and stash copies a shared-prefix branch needs in every attention
layer (both branches write the same cache slots). n-gram and head
drafts as two branches would add nothing on these prompts, where suffix
lookup never fires. No branch path ships.

Suffix lookup's 12-token anchor was re-checked the same way on two edits that
rename an identifier in about 50 quoted lines of Rust (600 tokens): anchors of
8, 10, 12 and 16 gave 60.3, 58.9, 60.1 and 58.5 tok/s geomean against 47.5 with
lookup off, and none fired on two novel prompts. 8 tied 12 only as the two
edits disagreed by about 10% each way, so 12 stays.

The target verifies a draft as a row block. Its recurrent layers read their
state and convolution history but write the block's final ones, and each row's
compact recurrence inputs, to the verifier, so the layer keeps the round's start
with no copy. Full acceptance swaps the verifier's buffers in; on rejection the
accepted rows are replayed from the untouched start, and target K/V and MTP
state are rolled back to that exact boundary. This uses 208 MB for 63 drafts
rather than 9.9 GB of per-row full checkpoints, and no state copy at all: the
former scheme copied all 48 states into a snapshot before every verify block
and back on every rejection.

A verify of at most eight rows (every tree, every head or lookup chain, every
sequence's drafts in a batched pass) no longer replays at all; this follows
the GDN tape of [Splash](https://github.com/incoai/splash). It records each
row's raw QKV projection, decay and beta on the sequence's tape (two halves
of 48 layers × 8 rows, 31.8 MB per sequence) and leaves the state and the
convolution history at the step's start. The commit only marks the kept rows
pending, with no GPU work. The sequence's next step (a decode step, a verify,
a tree or a batched pass) folds them in at the start of its own convolution
and recurrence kernels (`bo_conv_l2_decay_fold`, `bo_gdn_fold`,
`bo_gdn_rows_4_fold` and their tree forms): one threadgroup per head first
runs the replay's arithmetic over the pending rows on the state it is about
to read, then the step's own body. A step records into the half the pending
rows are not in. A long prefill block, a prompt checkpoint and a snapshot
replay the pending rows first (`encode_settle`), so everything that saves or
restores state sees what the replay left. The folded state is bitwise the
replayed one: `tape_folds_are_bitwise_the_replay` compares every logit row
and every recurrent state, folded against replayed, after a partial chain, a
full chain (whose buffers are swapped in, leaving nothing pending) and two
tree paths, each folded into a decode step, a chain verify and a tree
verify.

Measured against the `kflash` base `adc358f`, interleaved under the GPU lock
while other agents loaded the machine, so absolute rates sit below the KFlash
table further down. Alone, the tape moves work rather than removing much. The
commit's wall time halves (0.79 to 0.38, 2.33 to 1.21 and 0.56 to 0.27 ms per
cycle over three runs of the six prompts), but the next verify now writes the
state it folds, and its time per tree cycle rose by 0.7, 0.0 and 4.8 ms (the
last run also verified 0.15 more rows per tree). Seven interleaved rounds of
the six prompts at 300 tokens ended from 7.9% slower to 5.8% faster, 0.1%
slower on average: 38.3, 44.4, 40.2, 34.7, 39.8, 38.7 and 41.4 tok/s against
39.3, 43.5, 39.7, 32.8, 39.2, 42.0 and 41.7. The trace prompts through `serve`
gave 45.1 against 43.8 (one run), and the copy-edit prompts 79.2 against 77.5
in one run and 81.2 / 93.7 for both builds in another. Every text matched,
except the essay's known near-tie at token 123 in one round.

Batched steps gain more. There the replay was a submission of its own after
every pass in which a sequence kept part of its drafts. Paired runs, base then
tape, back to back in one hold of the GPU lock (`tools/serve_concurrency.py`,
300-token answers, aggregate tok/s at 1, 2, 4 and 8 streams), gave 31.8 / 33.8
/ 46.6 / 65.1 against 36.3 / 36.6 / 46.6 / 64.5, then 24.4 / 30.8 / 44.6 /
66.5 against 37.7 / 35.3 / 49.4 / 67.4. The single stream's spread is the
source choice, not the tape. Two unpaired rounds earlier read 38.7 and 40.2
against 50.0 and 45.6 at two streams. At eight streams the seeds fill the pass
and few sequences draft, so there was little replay to save. Within each pair
all 15 outputs matched. The single stream's text differed between the pairs
for both builds alike, because the base's own runs resolve that near-tie with
whichever source the timing picked.

With a greedy sampler and no penalty, the verify block (and a plain decode
step) also selects each row's argmax in its own command buffer, with the same
`total_cmp` order and ties to the lower id as the GPU top-k, and flags any
non-finite logit; the host reads one id per row instead of submitting a top-k
per row and scanning every logit. Every other sampler gets the same
non-finite flag, and one that draws from the plain top k (no penalty, no
masking grammar) gets each row's top k from the same command buffer too (see
[Sampled decoding](#sampled-decoding)); penalties and grammars keep the host
path.

Drafting stays on the GPU. Each draft step is one submission: the token's
`token_embd` row is decoded and inverse-rotated on the device
(`draft_embed_inverse`, reading the GGUF mapping without a copy), the head
layer runs, and a two-pass top-two (`draft_top2_partial`/`draft_top2_final`,
`total_cmp` order, ties to the lower id) leaves 16 bytes for the host, which
applies the EOS check and the margin gate exactly as the host sampler did; the
next step reads its token from the device buffer. Rollback and the head commit
share one submission, and a verify block submits after its first layer so the
GPU starts while the rest encodes. Against the per-step host loop: 3.24 to 3.05
ms per draft and 27.96 to 28.39 tok/s geomean on six prompts, byte-identical
tokens. Running all steps in one command buffer and truncating at the gate
afterwards lost (24.9 to 23.3 tok/s): only 211 of 477 possible steps pass the
gate, so the ungated steps cost more than the saved round trips. Requests with
presence or repetition penalties keep the host draft loop, since the device
top-two applies none.

#### Tree verify

A block drafter (`KFlash`, after Kapitsa) proposes a best-first tree of up to
eight rows rather than a chain; `bonsai_native/tree.rs` verifies it in one
pass. Row 0 is the anchor; every row names its parent and computes what plain
decoding of its own path would. Its K/V row sits at cache row `start + r`
with the rotary position `start + depth`. Attention reads the history below
`start` through the decode split kernel of that prefix, or from three rows on
a Q8 cache through the verify block's row kernel with every row's prefix
pinned at `start`. It reads the tree zone, its ancestors and itself, key by
key in path order in one more partial record (`bo_attn_tree_tail`), and
`bo_attn_reduce` joins the records. The recurrent layers break the runner-up
section's constraint: the state forks mid-block. Each row's short
convolution reads its ancestors' inputs (`bo_conv_l2_decay_tree`). Its
recurrence starts from its parent's state, which `bo_gdn_tree_rows_4*`
rebuilds from the committed state by replaying the path's updates with the
chain kernel's own operations, so its recurrence matches the chain kernel's
on the same path bit for bit. Nothing writes the committed state. The commit gathers the
accepted path's K/V, recurrence inputs and normalized rows into consecutive
rows in place (`bo_gather_rows`) and leaves the recurrence inputs on the
sequence's tape for the next step to fold in (see
[Lossless speculation](#lossless-speculation)). A chain goes to the chain
verify and one row to a plain decode step, so both keep their numbers.

A row's logits depend on `start`, its path and the projection class of the
pass (one row, two to four, five to eight), never on the rest of the tree or
on where its ancestors sit. The same path under two trees of a class is
bitwise identical (19 pairs at 18 and 1100 tokens of history). Across classes
it is within 0.008. Every row's GPU argmax equals plain decoding of its path
over 144 random rows, and logits agree within 0.019 of the 3e-2 verify
tolerance. GPU ms (median of 9) against the chain verify of the same row
count; n = 2 trees are chains:

| history | 3 rows | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| 18, tree | 58.4 | 70.9 | 71.7 | 72.4 | 74.0 | 77.0 |
| 18, chain | 57.3 | 69.6 | 71.8 | 71.2 | 72.6 | 75.4 |
| 8192, tree | 61.4 | 74.0 | 78.0 | 78.8 | 81.5 | 84.7 |
| 8192, chain | 61.0 | 73.4 | 77.3 | 77.8 | 80.4 | 83.2 |

That is -0.2 to +2.1%, against 1-6% in Kapitsa. `BonsaiModel::set_taps` adds the
drafter's residual taps, the residual after layers 5, 19, 33, 47 and 61 of
every committed position, in a 2048-position ring (210 MB). The taps cost
nothing while off and stay within run-to-run noise while on.

#### KFlash cycles

With the drafter installed, startup turns on the residual taps and tree
verify beside the MTP head, loads the drafter, measures on this machine the
verify cost of 1 to 8 rows (the faster of a warm-up pass and one timed pass of
each size) and the drafting pass, and reports it all under `policy.kflash`
(`resources.dflash_reason` names the drafter it loaded). The context fit counts
the drafter's 2.05 GB of mapped weights, its 67 MB of context ring and
scratch, the 217 MB tap ring with its verify staging rows and the 105 MB tree
verifier.

Each solo round then picks one source for its cycle, before any drafting, by
measured tokens per millisecond: the DFlash2 tree, the head's gated depth-3
chain, or suffix lookup when it has a 12-token match. Each source's tokens
per cycle and wall milliseconds per cycle are averaged over the cycles it ran
(evenly for the first ten, then decaying by a tenth), and the cycle goes to
the leader. A source that has not run in the request runs first; a trailing
one runs again after `base · (leader rate / its rate)³` cycles (base 16 for
the tree, 32 for the head chain, 8 for lookup, at most 512), so a change in
the text is noticed without paying for it every cycle. Because the choice
comes first, no drafting pass is drafted only to be compared and thrown away.

The bases, the cap and the cube were swept on the six prompts (300 tokens, two
rounds, every variant interleaved under the GPU lock while other agents
worked; geomean of the cleaner round): 16/32/8 with cap 512 gave 42.0 tok/s,
half the bases with cap 256 42.3, double the bases with cap 1,024 43.7, cap
128 43.3, a square in place of the cube 40.9, and tree and lookup bases halved
43.1. Run to run a single prompt varied by up to 20%. Doubling the bases
against the shipped ones on a quiet machine (three rounds, best of three) gave
50.7 against 50.4. The variants move only the few cycles a source sits out. On
the essay, where the head leads, the tree ran 4 to 24 of about 160 cycles with
the shipped bases and 2 to 3 with doubled ones: 35.3 against 33.1 tok/s at
best, and a tie on the other five prompts. That is inside the run-to-run
spread, and doubling the bases also doubles how long a source that has fallen
behind waits to be tried again, so none was kept. The 64-position restart wait
was swept the same way through `serve`: a 48-token request beside a 512-token
one, which then decodes alone, over three prompt pairs. Waits of 16, 32, 64
and 128 gave 43, 32, 44 and 28 tok/s geomean. The 32 and 128 runs were
contended. The source choice after the restart decided more than the wait did:
the hash-map and arithmetic answers each ran either 0 to 3 trees or about 60,
at no consistent wait. So the wait stays at 64.

A tree cycle is three submissions:

```text
draft:   append the positions committed since the drafter last drafted (from
         the tap ring, in 64-row chunks), drafting pass at the anchor, 8-row
         best-first tree                                     one command buffer
choose:  n = argmax E(n) - r * cost(n) over the first n rows, depth capped by
         the request's remaining tokens and the context     host, ~µs
verify:  tree_verify (one row: decode step; a chain: chain verify block;
         else the tree kernels), accept the longest agreeing path with the
         sampler's own draw per position                     early-submitted
commit:  gather the path, its taps to the ring, its rows into the MTP head;
         its recurrence updates stay on the tape for the next
         step's kernels to fold in                           one command buffer
learn:   record the tested nodes, refit every 16 cycles, upload; fold the
         verify's GPU time into the per-row cost drift       host
```

`E(n)` is 1 plus the calibrated path probabilities of rows 1 to n − 1, read
back with the tree; `cost(n)` is the load-time verify cost of n rows plus n
times a per-row drift that every verify's GPU time updates, so it follows the
history the rows attend as the context grows; `r` is the request's recent
tokens per millisecond over all sources (before four cycles,
`E(n) / (draft + cost(n))`). A tree whose best size is the anchor alone is
handed to the head's chain instead of a bare decode step, which costs at most
one head step when the head is no surer than the drafter.

Splash encodes a whole decode cycle as one command buffer (`decodeAsync`):
draft, a fixed verify of every drafted row, acceptance and the row choice on
the GPU, with the host reading results only at the end. That was not built,
because the cycle's two host points cost almost nothing. Choosing the size
reads back the tree's eight rows with their calibrated path probabilities and
takes microseconds; the commit, now that the tape removed its replay, is a
gather and the taps, 0.33 ms of wall time per cycle on the six prompts. A
fixed graph has to verify all eight rows, and a size can only be picked once
the tree is known. Over 481 tree cycles of the six prompts, 12% chose two
rows, 2% three to six and 86% seven or eight. Verifying eight rows every time
would cost 4.8 ms more per tree cycle (77.2 against a mean of 72.4 ms). The
source choice before drafting also stays on the host: the head and lookup
draft nothing on the GPU that a tree cycle could share. Folding the commit
into the next cycle's drafting buffer would save at most that 0.33 ms (under
0.5% of a cycle), and an earlier profile put all GPU idle time in decode at
1.5-2.5% of wall time.

The drafter's context follows the tap ring lazily. Cycles that verify head
or lookup drafts append nothing; the next drafting pass appends every
position committed since in a few 8-row chunks (one row costs 0.83 ms, a
64-row chunk about 8 ms), and a backlog over 64 rows (a prompt, or a long run
of head cycles) is appended in its own command buffer first so it does not
count against the cycle. The ring records which consecutive positions of
which buffer set it holds. When that span breaks (rows decoded in a batched
pass, another sequence's prefill, a restored prefix) the drafter restarts at
the first position tapped since, at most a window back, and attends only
what it has from there, as a sequence that started there would. Unless that
is position 0 it first waits for 64 tapped positions, decoding on the old
rule (lookup, else the head) meanwhile, since trees drafted from a handful
of positions would be worth little and would rank the tree behind for many
cycles (the 64 is a choice, not a measured optimum). Once it can draft
again the tree counts as untried. A request that restores a prompt-cache
prefix of the previous drafting request's tokens (the next turn of a
conversation) continues that request's context from the prefix instead, at
once, provided the drafter's ring still holds it. A disk snapshot restored
in a new process has no drafter context: those requests draft once they
have tapped 64 positions past it.

Batched serving keeps its passes as they were: no taps, no trees, head and
lookup drafts priced by the cost model. A sequence that becomes the only one
decoding takes its solo rounds on the old rule (lookup, else the head),
which tap its committed rows, and after 64 of them drafts trees from a
restarted context.

Per-stream DFlash2 drafts in batched passes were built and measured, then
reverted. The drafter's context followed one sequence: the one it already
followed, else the participant with the most tokens left, the likeliest to
finish alone. That sequence's rows wrote the residual taps at its row offset
in every batched pass, and its commit recorded how many it kept. Its context
never restarted, so once alone it drafted trees at once. Its DFlash2 chain
competed in the cost model with every sequence's head and lookup chains. The
batched kernels verify chains, not branching trees, so the candidate was the
tree's most likely path, up to seven drafts. It was priced first at the
sequence's running chain acceptance (0.75 to start) plus a drafting pass,
then, once drafted, at its calibrated path probabilities. Batched output
matched each sequence decoded alone (`batched_kflash_matches_each_alone`,
three prompts, forced drafts: 41 chains, 130 of 211 drafts accepted, then
solo trees from the same context).

The cost model seldom picked a chain: two to five per 300-token request at
two to four streams, on essays and on mixed prompts alike. A single path
gives up the tree's branching, which is where the solo tree's lead over the
head comes from. Its 12.6 ms drafting pass, for one sequence, costs about
what a depth-3 head round costs for four sequences together. From five
sequences a chain's rows also cross the eight-row step of the pass cost.
Paired runs against the tape alone, back to back in one hold of the GPU
lock (`tools/serve_concurrency.py`, aggregate tok/s at 1, 2, 4 and 8
streams):

| Run | With chains | Tape alone |
| --- | --- | --- |
| 1 | 28.8 / 28.3 / 40.6 / 60.4 | 32.2 / 38.8 / 49.1 / 66.7 |
| 2 | 36.4 / 43.2 / 52.6 / 71.0 | 35.7 / 42.2 / 52.6 / 71.4 |

The first run overlapped another agent's build. Mixed code, arithmetic and
prose requests at 2, 3 and 4 streams gave 38.2 / 51.9 / 55.7 against
38.1 / 52.3 / 56.4. Through `serve`, a 512-token request that runs alone
after a 48-token one gave 56 / 37 / 43 tok/s on three prompt pairs with the
continuous context, against 52 / 33 / 48 with the restart. The source choice
after the hand-over varied from run to run more than that: one bad first
tree measurement could keep the tree out for the rest of the request. A
drafter context for every sequence (about 260 MB each at a 2,048-position
window, the drafting pass still one sequence at a time) was not built,
for the same reason.

Measured on the M4 Pro with the shipped drafter (`bonsai --greedy --json`,
300 tokens, six prompts of the depth/margin sweep, cold prompt cache, best
of two, builds interleaved; main is `d25b277`). Startup measured 30.5,
44.0, 57.9, 69.9, 72.1, 72.7, 73.6 and 77.2 ms to verify 1 to 8 rows and
13.1 ms to draft. Cycles are tree cycles, trees declined to the head's
chain, head cycles and lookup cycles; tokens per cycle are the tree cycles'
emitted tokens against the calibrated `E(n)` they were chosen for:

| Prompt | KFlash | Main | Cycles (tree / declined / head / lookup) | Tokens per tree cycle (expected) |
| --- | ---: | ---: | --- | ---: |
| prose | 40.8 | 37.5 | 54 / 2 / 1 / 0 | 3.80 (3.95) |
| explanation | 46.5 | 39.3 | 64 / 6 / 1 / 2 | 4.28 (4.33) |
| arithmetic | 63.3 | 41.6 | 48 / 9 / 1 / 0 | 5.90 (5.36) |
| code | 58.5 | 42.3 | 56 / 3 / 1 / 0 | 5.21 (5.01) |
| essay | 34.9 | 35.5 | 5 / 0 / 154 / 0 | 2.20 (2.76) |
| thinking | 66.5 | 43.7 | 41 / 6 / 1 / 0 | 6.59 (6.25) |
| geomean | **50.4** | 39.9 | | 5.01 over all tree cycles |

An earlier run of the same code measured 50.9 against 40.1 and 32.8
without speculation, every KFlash text byte-identical to plain decoding.
Main's essay was not: its head chain resolves a near-tie at token 123
differently from a decode step. The essay runs on the head's chain under
KFlash too, and in the later run above took main's text. The six trace
prompts at 512 tokens (one round) gave 49.2 against 39.9 (32.2 without
speculation): prose 39.6, explanation 41.0, arithmetic 63.0, code 59.6,
essay 34.8 and thinking 67.2, against 37.4, 38.3, 42.3, 43.2, 35.2 and
43.5, at 4.98 tokens per tree cycle. Five match plain decoding; the
essay (269 head cycles, 4 trees) differs at the same token as main's. The
two 600-token copy-edit prompts reached 79.7 and 93.1 against 71.9 and
93.9 tok/s, both identical to plain decoding: the rename edit split
its cycles between 18 lookups and 14 trees (7.93 tokens each), the other
stayed on lookup. Through `serve`, a 600-token coding answer decoded at
53.1 against 41.6 tok/s and an agent turn with tools at 61.9 against 55.2,
with the same text as main; a 53-token pair of tool calls, 15 cycles
mostly on the head, at 40.2 against 43.9. Two 300-token essays through
`serve` gave the same text as a server with neither the head nor the
drafter installed. The `kflash_generation_matches_plain_decoding` test
(ignored; GGUF, head and drafter) compares a code and a prose answer,
greedy, and a seeded sampled one with plain decoding, then a greedy answer
decoded first in a batch and then solo (14 batched tokens, 27 head cycles
while the restarted context rebuilt, 38 tree cycles). Greedy answers may
first differ only where plain decoding's two best logits are within 0.03.
One earlier run, which still required exact equality, saw the code answer
differ at token 132 (its margin was not recorded) under a calibration other
runs had moved; the two runs since matched throughout.

A long prompt pays for the drafter's context once, after its first token.
Summarizing a 2,573-token section of this file (128 tokens, two runs) took
the same prefill (19.2-19.3 s against 19.5-20.3 s to the first token), then
appended its last 2,047 positions to the drafter in 0.26 s, drafted one
tree (2 tokens against an expected 2.94) and gave the other 67 cycles to
the head: 31.7-31.8 tok/s against 33.9-34.5, the difference being that
append. Short prompts reach their first token sooner (0.46 s against
0.88-0.92 s for 45 tokens), likely because the cost measurement at load
leaves the GPU warm.

Simulated over the trace lattices the shipped drafter yields 4.97 tokens
per 8-row cycle against the first pin's 4.68 (×1.059). Through this engine,
same six prompts and run (the tree cycles stay the target's own path, so
both drafters emit the same text):

| Prompt | Shipped | First pin | Shipped tokens per tree cycle | First pin |
| --- | ---: | ---: | ---: | ---: |
| prose | 39.3 | 38.3 | 3.62 | 3.91 |
| explanation | 46.4 | 42.4 | 4.35 | 4.25 |
| arithmetic | 64.6 | 59.3 | 6.36 | 5.82 |
| code | 58.2 | 53.9 | 5.35 | 5.03 |
| essay | 35.1 | 35.0 | 2.20 | 2.50 |
| thinking | 67.4 | 65.4 | 6.90 | 6.28 |
| geomean | **50.3** | 47.8 | 5.10 over all | 4.92 over all |

×1.052 end to end and ×1.037 per tree cycle: the gain trained on PQ2_0
packing carries over to `PTQ1_0`. The first pin also declined more of its
trees to the head (446 against 389 head or declined cycles over both
rounds).

### What the speculation gain is made of

Speculation combines two independent mechanisms: the MTP head, which drafts
speculatively ahead, and n-gram suffix lookup (`ngram_policy`,
`min_match: 12`), which proposes the known continuation of a previously-seen
12-token suffix. Both are honestly called speculation, the opt-out flag these
measurements used (since removed) disabled both, and the startup policy
reports the two separately, so an A/B against plain decoding measures
speculation rather than the head alone. What
that A/B cannot show is how the gain divides between them.

Measured on a 16 GB M2 MacBook Air, greedy decoding, `--max-tokens` 160 to 400,
three repetitions per arm taking the median. Every arm produced byte-identical
output, so the comparison is fair.

| Workload | Speculation | Plain decoding | Speedup | MTP | n-gram |
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

These figures were taken with the retired int8 head, whose matrix-vector
product read four output rows per SIMD group with eight-byte weight loads; that
took drafting on the planets prompt from 0.588 s to 0.503 s on an M4 Pro,
leaving it about 8% of decode time. The shipped mixed head runs its ternary
matrices on the target's own PTQ1 kernels and its int8 matrices on the int8
kernels; this breakdown has not been re-measured with it.

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
   discovered at startup, and trimmed within the automatic disk budget (at
   startup too, so a store a previous process left over budget is trimmed
   before the first request).

A request restores the longest prefix any tier holds, the cheaper tier on a
tie, and a failed restore falls back to the next longest. The tiers used to
be asked in order, so any matching GPU checkpoint won over a longer disk
snapshot. In a compacted 64K Codex-style session (one `xhigh` run on
`b09e22f`'s engine) that pinned four turns of 19K–31K-token prompts to a
12,980-token checkpoint while disk held their predecessors' 16K–25K
snapshots, and steps grew from 30 s to 179 s. The cause: a thinking request's
prompt ends with the generation opener `<think>\n`, which the next turn's
history re-renders without, so every turn diverges from the last prompt just
before its end and checkpoints a divergence boundary that barely extends the
previous turn's. Those boundaries filled the four GPU slots; from then on
each new checkpoint, boundary or request tail, was the eviction victim the
moment it was saved, and the GPU tier kept restoring the newest boundary it
still held, a few turns old. `agent_turns_keep_restoring_the_previous_prompt`
reproduces it in ten 300–600-token turns: on `b09e22f` the sixth turn
restores 357 tokens from the GPU tier where the previous boundary is 395.
Now the two newest GPU checkpoints (the last request's boundary or
penultimate token, and its end) are never the victim, eviction picks among
the others by the disk tier's rule below, and the longest prefix wins across
tiers: in the same test every turn restores the previous boundary, two turns
in three from the GPU tier and every third from disk, where the boundary the
GPU tier evicted (the newest one that barely extends another) was persisted.

A request used to checkpoint its prompt's end beside its penultimate token
and its output end. The end serves only a next prompt that extends the whole
prompt, and the penultimate checkpoint serves that one for one more token of
prefill, so the end is no longer checkpointed when the penultimate token was.
With it, the two newest checkpoints were the end and the output end, and once
the tier filled the penultimate checkpoint was the oldest request tail left
to evict. The next agent turn, which diverges at the last prompt token, then
restored an older boundary, and the disk tier had no penultimate snapshot to
fall back on (a request that persisted a boundary does not persist one). After
a compaction in a 64K `xhigh` session (a cold prompt restoring the system-turn
boundary from the GPU tier and checkpointing its divergence) three read steps
in a row each prefilled one turn too many: 367, 7,474 and 10,456 tokens
cached instead of 7,474, 10,456 and 13,562, 78, 50 and 53 s instead of 26 to
29 s, 13,195 stale tokens in all. With the change the same session has none.
`a_turn_after_a_new_boundary_restores_the_penultimate_checkpoint` reproduces
it in three requests: without the change the third restores 385 tokens from
the GPU tier where the penultimate checkpoint is at 396.

A request that persisted a boundary leaves its penultimate snapshot to its
GPU checkpoint, which its session's next turn restores, and reads it back
(about 38 KiB a prompt token) only for a tier that keeps it. It used to read
it back at every request's end and, with no host tier (fast storage) and the
boundary already written, drop it. On a 13,223-token chat prompt (`local-ai
bonsai --json --greedy --max-tokens 1` with reasoning off, cold, host tier off,
MTP head; two interleaved runs per arm with other jobs loading the CPU) the
time from the first token to the end of the request fell from 80 and 85 ms
to 38 and 39 ms and the peak footprint from 2,620 and 2,629 MB to 2,059 and
2,064 MB, the 0.56 GB copy (3.8 GB at 100K tokens). The GPU tier describes
one buffer set, though, so that checkpoint was the only copy of the tail and
was lost whenever another session's request took the tier: a request of
another session running at the same time claims it at its milestones and end, and
one that runs next prefills over its rows. The tail now goes to disk before
its GPU copy does: at the request's end when the copy is already gone or
another request is running, else when a request that does not extend it
takes the set. `concurrent_sessions_each_restore_their_previous_tail`
reproduces it with two sessions admitted together: on the old code the
session whose turn ended first restored 245 tokens (its system turn) where
its tail is at 286. Keeping per-set checkpoint lists would have avoided the
readback but needs the scheduler's pooled and parked sets to keep their
rows for it, and up to four 81 MB checkpoints per set.

Disk snapshots are written by one background thread the engine owns. The
engine thread pays only the GPU readback; hashing, the write, `fsync`, rename
and trim run behind the request. A snapshot is indexed for reuse only after
its rename is reported, so a file still being written is never loaded. A
request whose best disk match is still in flight is not held up at
admission; its first prefill step asks again and skips steps while other
requests decode, or waits for that one write when none does (see
[Prompts prefill inside the decode steps](#prompts-prefill-inside-the-decode-steps)).
Reading a snapshot back is off the engine thread too. A request whose
longest match is a disk snapshot starts a load, which reads the file and
checks its SHA-256 on a thread of its own, and restores at most a GPU
checkpoint meanwhile, as a stand-in it does not prefill from. It stays out of
prefill until the load ends: the step's prefill goes to another prompt, or is
sat out while others decode, and the engine waits for the load only when
nothing else can run. Its first prefill step then copies the snapshot into
its buffers, the one part left on the engine thread; a load that failed is a
failed restore like any other, so its entry leaves the index and the tiers
are asked again. A load dropped unfinished (its request cancelled) stops
within a megabyte of reading.
At most one job is in flight and one queued: a newer job replaces the queued
one, except that a request tail never displaces a shared boundary. Dropping
the engine finishes pending writes; the server, whose engine is never
dropped, closes the writer on SIGINT or SIGTERM instead: accepted writes get
two minutes to commit, past which the one in flight is aborted and its
temporary removed. Before that, shutdown stops accepting connections and
gives foreground requests 10 s to finish; generations still running then are
cancelled, so their responses end at the next engine step (connections not
closed 5 s later are dropped), and a long generation cannot delay shutdown
until a supervisor's SIGKILL interrupts these writes. A temporary is live only while its writer holds an
exclusive lock on it (taken in `open`), so the sweep after every store and
at startup removes any other at once, whatever its age. A snapshot the trim
after it would delete again is not written, nor read back from the GPU: the
engine sizes it from its token count and asks its own index the writer's
question first. A snapshot larger than the whole budget is still written,
and kept while it is the only one, evicting every other: what it evicts fits
in the budget and it does not, so it costs more prefill to rebuild than all
of them together (131K tokens prefill in about half an hour), and the
staged long-context runs rely on it (see Verification). A failed write is a cache miss reported
on stderr, not a request error. Trim keeps an in-memory index of the
directory instead of re-reading every snapshot header after each store.

Reading and checking a snapshot used to run inside the restoring request's
admission, where every decoding stream waited for it. Synthetic snapshots
without the head (34 KiB a token), file in the page cache, M4 Pro, best of
two:

| Tokens | Snapshot | Read and SHA-256 (now off the engine thread) | Copy into the model's buffers |
| ---: | ---: | ---: | ---: |
| 8,192 | 366 MB | 211 ms | 34 ms |
| 32,768 | 1,222 MB | 656 ms | 58 ms |
| 102,400 | 3,646 MB | 1,924 ms | 336 ms (623 ms growing the caches) |

With a plain greedy stream decoding beside it, a request restoring a
6,978-token, 324 MB disk snapshot (read and checked in 258–275 ms;
`a_stream_keeps_decoding_while_another_request_loads_a_disk_snapshot`, one
run per build, other builds loading the CPU) was admitted in 301 ms before
and 8.7 ms after. The stream's longest gap until that request's first token
fell from 419 ms (one token in that window) to 167 ms (six tokens emitted
while the snapshot loaded, the longest gap being the copy and the request's
own last-token prefill).

Measured on an M4 Pro (1.7 GB/s write probe, host tier off), six interleaved
cold runs per arm of `local-ai bonsai --raw --json --greedy --max-tokens 16`,
medians, ~204 MB snapshots:

| Case | Synchronous write | Background write |
| --- | ---: | ---: |
| 696-token prompt, 678-token system boundary: TTFT | 8.611 s | 8.469 s |
| Same, request time | 9.300 s | 9.153 s |
| 17-token prompt, tail snapshot: request time | 1.931 s | 1.847 s |

The request tail is checkpointed at the penultimate chat token because the
next rendered turn diverges at the assistant opener. One reusable boundary is
also materialized per prefill: a newly observed shared-prefix divergence is
preferred, otherwise the end of the system turn. Boundaries shorter than 128
tokens are skipped. Shared boundaries receive eviction
priority over request-tail snapshots and can be reused across sessions. When
the budget cannot hold both, a boundary that extends a kept boundary by
fewer tokens than that boundary holds is the one evicted (and is then not
written or read back at all): every request it serves, its ancestor serves
too, so keeping the ancestor costs those requests only the short extension,
where keeping it would cost every other request the whole prefix. A
divergence a few tokens into one kind of question therefore never replaces
the system-turn boundary every question of a 128K-token prompt restores.

Measured effects:

| Case | Cached result | Cold result |
| --- | ---: | ---: |
| Follow-up conversation turn | 2.7 s | 25 s |
| Restart and disk restore | 0.69 s | full prefill |
| Shared 6.7K-token prefix | 1.2 s TTFT | 73 s TTFT |

## K/V cache

The target's full-attention layers store keys and values as Q8: int8 values
with an F16 scale per 32, 8.5 bits per value and 34 KiB per token across the
sixteen layers, against F16's 64 KiB. There is no other target format and no
fallback: Q8's 34 KiB per token is what the context policy budgets. Scores and
accumulation remain F32. The MTP head's one-layer cache always remains F16.
A block's scale saturates at F16's largest value, 65,504, so values beyond
127 times that (8.3e6) clip instead of an infinite scale reading the whole
block back as NaN (0 × inf).

Quantized rows are stored in a rotated basis: each head's 256-value key and
value row is multiplied by the orthonormal Walsh-Hadamard matrix H/16 before
quantization, and the query by the same matrix, so every score (Hq)·(Hk) equals
q·k exactly. Rotation spreads an outlier channel's energy across the whole row,
which shrinks the per-block scale for everything else in it. Values come back
through `bo_attn_unrotate`, which rotates each head's output and then applies
the output gate, since gating does not commute with the rotation. It took Q8's
error from 1.4e-5 to 8.9e-6 and costs nothing measurable: plain decode at a
512-token prompt measured 21.9 tok/s with F16 and Q8 alike.

Caches begin at 1,024 tokens, or the selected context when smaller. Capacity
grows only when a request reaches it, preserving written rows exactly: it
doubles up to 4,096 tokens, then grows 4,096 tokens at a time, because Metal
keeps a whole buffer resident once the GPU uses it (see Memory).
Attention workspaces and speculation buffers grow with actual need rather than
the maximum context.

Q8 is the default because it is effectively lossless and faster. Against F16 on
the real model, teacher-forced along 64 greedy tokens after a prompt of this
repository's own text (`quantized_kv_tracks_f16_next_token_distribution`):

| Layout | KiB/token | Mean next-token KL | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Q8, 4,096-token prompt | 34 | 8.9e-6 | 64/64 |
| Q8 unrotated, 4,096-token prompt | 34 | 1.4e-5 | 64/64 |
| Q8 unrotated, 16,384-token prompt | 34 | 1.4e-5 | 64/64 |

Decode attention reads the whole cache for every token and is bandwidth-bound,
so fewer bytes are faster. Each quantized K or V tile is dequantized to half in
threadgroup memory and fed to the same Metal 4 tensor matmuls the F16 kernel
uses; the P·V accumulator stays in registers for the whole split and is
rescaled in place, instead of a per-tile round trip through threadgroup memory.
One layer on an M4 Pro, GPU time:

| Context | F16 | Q8 |
| ---: | ---: | ---: |
| 1K | 33.8 µs | 33.3 µs |
| 4K | 86.7 µs | 69.6 µs |
| 16K | 301 µs | 244 µs |
| 64K | 1,192 µs | 903 µs |
| 128K | 2,476 µs | 1,844 µs |

Below Q8 the unpacking, not the bytes, sets the pace, which is why smaller
formats were not worth their accuracy (see Tried and rejected).

Each decode split costs a Q load, a 258-float partial record per query head
and a step of `bo_attn_reduce`'s serial walk, so splits grow with the prefix:
64 tokens below 4,096, then the largest power of two under prefix/64,
clamped to 256..1,024 (`tensor_split_tokens`), where a fixed 256 had paid
512 splits per KV head at 128K. Quantized splits also run KV head fastest:
a token's four head rows share the cache lines of its 32 scales, which the
split-major order fetched once per head. Plain decode attention, one layer
on an M4 Pro, best of five and of two interleaved runs, before and after:

| Context | Q8 | F16 |
| ---: | ---: | ---: |
| 8K | 137 → 129 µs | 161 → 159 µs |
| 16K | 254 → 239 µs | 313 → 310 µs |
| 32K | 468 → 442 µs | 602 → 586 µs |
| 64K | 914 → 846 µs | 1,205 → 1,120 µs |
| 128K | 1,874 → 1,579 µs | 2,497 → 2,243 µs |

Whole model, plain decode after a synthetic prefix (best of three 8-token
runs, two rounds): 30.1 → 30.0 tok/s at 8K, 25.5 → 25.9 at 32K, and 16.2 →
17.5 at 128K, where attention was half of a token (16 layers x 1.87 ms of
62 ms).

Below a 1,024-token prefix, decode attention uses the SIMD `bo_attn_split`
kernel on every build, in 32-token splits (see Performance).

Causal prefill uses a tensor kernel of the same shape. Whole model, plain
decode without speculation, M4 Pro:

| Prompt | Prefill F16 | Prefill Q8 | Decode F16 | Decode Q8 |
| ---: | ---: | ---: | ---: | ---: |
| 512 tokens | 88.9 tok/s | 90.7 tok/s | 21.9 tok/s | 21.9 tok/s |
| 24,576 tokens | 87.7 tok/s | 85.6 tok/s | 18.73 tok/s | 19.34 tok/s |

The decode gap widens with context, since attention's share of each token does.
Every attention path reads quantized values dequantized and rounded once to
half, so the paths agree on their operands as F16 caches always have. A
quantized cache still amplifies kernels' last-bit differences in the K/V rows
they write (a value near a rounding boundary moves a whole step), which the
verify-block test bounds separately per format; greedy output with speculation
remained byte-identical to output without it over 300 tokens on three prompts.

## Memory

Checkpoint weights remain file-backed and are paged on demand. Startup does
not touch every weight page. The 167 MB mixed ternary/int8 MTP head artifact is
memory-mapped and demand-paged too. K/V, attention workspaces, verify rows, and other
context-dependent buffers are allocated at their minimum useful size and grow
lazily.

The resulting short-request peak footprint is 0.8-0.9 GB (`/usr/bin/time -l`,
a 14-token prompt; 1.18 GB before the changes below). These footprint
values count resident runtime memory; the 5.9 GB mapped checkpoint and the
head remain file-backed, outside the footprint, and their resident pages vary
with operating-system pressure.

The vision projector's BF16 tensors are mapped the same way (931 MB file);
only the summed F32 patch weights and the position table (14 MB) are copied.
Encoding an image allocates about 48 KB of F32 activations per patch while it
runs (0.8 GB for a 4,015-token image at `auto`, about 3 GB at the `high`
maximum) and keeps recent embeddings (20 KB per image token) in a 512 MiB
cache. The context fit does not count these transient buffers.

Audit of one `bonsai --greedy --json --max-tokens 500` request on a
15,920-token prompt (MTP and n-gram on, cold prompt cache), `footprint`
sampled every 2 s, M4 Pro:

| Component | Before | After |
| --- | ---: | ---: |
| GDN state, 48 layers (F32 to F16) | 151 MB | 75 MB |
| Verify rollback, depth 3 / 63 drafts | 163 / 282 MB | 89 / 208 MB |
| Verify logits, 4 / 64 rows | 4 / 64 MB | unchanged |
| Target Q8 + head F16 K/V at 16,420 tokens | 1,275 MB (32,768 allocated) | 797 MB (20,480) |
| Prompt GPU checkpoint, each of up to 4 (purgeable) | 157 MB | 81 MB |
| Prompt snapshot held in host memory during decode | 777 MB | none |
| Footprint after load | 491 MB | 416 MB |
| Footprint during decode | 2,118 MB | 1,254 MB |
| Peak footprint | 2,769 MB | 2,090 MB |

Three changes made the difference. The recurrent state is stored F16 (see
Performance). K/V caches stop doubling past 4,096 tokens: a Metal buffer
becomes resident as a whole once the GPU uses it (each doubling during prefill
raised the footprint by the full new allocation less the old one), so doubling
kept up to half of a long request's K/V resident and unused. The prompt's
cache snapshot was read back to host memory right after prefill and held
through the whole decode until the request ended; the request now pins an
81 MB recurrent checkpoint instead and reads the snapshot at the end, from that
checkpoint and the live K/V prefix, which decode never rewrites. The peak is
that end-of-request readback (673 MB) on its way to the disk writer; a
request that already wrote a boundary skips it when no host tier keeps the
snapshot (see Prompt cache).

With the DFlash2 drafter installed, its 2.05 GB of int8 weights are mapped
and file-backed like the head's, and KFlash adds about 380 MB that stays
resident: the drafter's context ring and scratch (67 MB), the tap ring
(210 MB) and the tree verifier (105 MB). The context fit counts all of it
(`state_bytes` 389 MB higher, estimated working set 20.47 against 18.03 GB)
and still selects 262,144 tokens. Peak footprint (`/usr/bin/time -l`, cold
prompt cache, two runs each against `d25b277`): 1.44-1.47 against
1.08-1.11 GB for a 45-token prompt and 300 tokens, 1.71 against 1.30-1.33
GB for a 2,573-token prompt and 128 tokens. Maximum resident set size,
which counts the drafter's mapped pages, is 2.69 and 2.84 against
0.64-0.67 and 0.77-0.80 GB. Loading takes 3.2-3.3 s against 0.3 s: the
drafter's payload digest and the verify and drafting cost measurement.

The GDN tape (see [Lossless speculation](#lossless-speculation)) keeps
31.8 MB per sequence, for the resident sequence and for each parked batched
one, plus a 2.9 MB block its folds' convolutions write to. The context fit
and the batched admission count both (a parked sequence's state is now
113 MB rather than 81 MB).

[Neural Engine prefill](#neural-engine-prefill-ffn) allocates, on its first
block of 192 rows or more, two weight sets of int8 FFN weights and output
projection rows (603 MB of IOSurfaces), two lanes of input and output
surfaces (22 MB), F32 rows for the second half's residual stream and both
halves' joined FFN outputs (52 MB) and an 81 MB recurrent checkpoint for a
GPU rerun: about 0.76 GB, counted in the admission estimate whenever the
Neural Engine may be used.
Peak footprint rose by 0.73-0.75 GB in the A/B there.

## Performance

End-to-end summary (M4 Pro, greedy). Each row names where it was measured;
the sections below give the A/B detail. Rates move by a few percent between
interleaved runs, so compare builds within one run, not across rows.

`tools/benchmark.py` records macOS power source at the start and around each
measured sample. Battery power, an unknown source, and source changes do not
block a run; mixed power states are reported as an uncontrolled comparison.
Sleep and competing builds remain timing-validity guards. `--allow-busy`
bypasses the load/build guards and downgrades sleep failures to warnings with
an `UNTRUSTED` result. There is no background power-polling loop.

Release builds use fat LTO, one codegen unit and a stripped symbol table
(`[profile.release]` in `Cargo.toml`). Against the previous thin-local,
16-unit, unstripped profile the `local-ai` binary shrinks from 38.5 MB to
23.5 MB, and decode is unchanged: 37.38 and 37.30 against 37.41 and 37.26
tok/s best (`tools/benchmark.py --rounds 2 --max-tokens 300`, binaries
interleaved), as expected for an engine that spends 97-98% of decode on the
GPU. The cost is build time: about 3 minutes for a full `--release` build or
test run, and every relink repeats whole-program optimization.

| Workload | Result | Source |
| --- | ---: | --- |
| Plain short-prompt decode, six-prompt geomean | 30.4–32.3 tok/s | best-of-two A/B runs below, across recent kernel changes |
| Default speculative decode, six-prompt geomean | 38.0–39.2 tok/s | same runs; 38.6 in the depth/margin sweep |
| KFlash with the DFlash2 drafter installed, six-prompt geomean | 50.3–50.9 tok/s (main 39.9–40.1 in two of the runs) | best of two, cold prompt cache, three runs ([KFlash cycles](#kflash-cycles)) |
| Two 600-token copy-edit prompts, KFlash | 79.7 and 93.1 tok/s (main 71.9 and 93.9) | best of two, cold prompt cache, one run |
| Two 600-token copy-edit prompts, suffix lookup | 89.8 and 90.5 tok/s | 5- to 128-row kernel change |
| Cold prefill, 2K–8K tokens | 133.4 → 364.6 and 130.2 → 346.5 tok/s | FFN and up to 3,072 mixer output rows on the Neural Engine against main `d25b277`, best of two interleaved rounds (2026-10-27, [below](#neural-engine-prefill-ffn)); the FFN alone 357.6 and 333.6 |
| Cold prefill, 32,726 tokens | 112.8 → 252.3 tok/s | same, one round; the FFN alone 244.2 |
| Cold prefill, 510 tokens | 128.6 → 288.4 tok/s | same; blocks of 192-511 rows ran on the GPU before |
| Cold prefill, 131,146 tokens | 69.7 → 74.8 tok/s | GPU only, 512-row blocks against 128-row blocks, one round (2026-10-22; 56 tok/s on the [original build](https://github.com/debanjanbasu/local-ai/commit/b389329)); not re-measured with the Neural Engine |

Shaders compile with `-fmetal-math-mode=relaxed`. Metal's default fast mode
also assumes no infinities or NaNs, yet online softmax starts each running
maximum at `-INFINITY` and relies on `exp(-inf) = 0`, and sampling and the
greedy selectors must see non-finite logits. Relaxed keeps the other
fast-math freedoms (reassociation, contraction, approximate functions); the
generated AIR differs only in the dropped no-NaN/no-infinity flags. Against
the fast-mode build it measured the same: plain decode 32.07 against 32.07
tok/s, best of four interleaved rounds on the default prompt with a 300-token
cap (while other builds loaded the CPU), and greedy text identical on six
prompts at 300 tokens with and without speculation.

Decode spends 97–98% of wall time on the GPU. PTQ1 decode moves 5.65 GB of
weights per token. The packed projections were ALU-bound, not bandwidth-bound:
with weights pinned in cache the floor-based matvec took the same 135 us on
17408x5120 as streaming them, while loads alone took 89 us. Apple GPUs run
`floor` and integer-to-float conversion at quarter rate, so the kernels now
derive each base-3 prefix `floor(q * 3^p / 256)` with one half FMA that rounds
onto [1024, 2048), where halves are integers (the byte enters as the half
`1024 + q` by OR-ing it into the mantissa; offsets make every byte, including
the q = 0 tie, round to the floor). The values are the integers the floor gave,
so decoded trits are unchanged and the small-batch kernels stay bit-identical.
The single-row matvec also moved to four lanes per block (one uchar4, two
uchar2 and the scale per row and block, float4/float2 activations) and eight
rows per SIMD group; fused SwiGLU now reduces four rows of each matrix.
Kernel GPU time on an M4 Pro, best of two interleaved runs, weights rotating
through 512 MB:

| Projection | Before | After |
| --- | ---: | ---: |
| 17408x5120, 1 row | 135.4 us, 144 GB/s | 89.7 us, 217 GB/s |
| 5120x17408, 1 row | 140.0 us, 139 GB/s | 93.5 us, 209 GB/s |
| 12288 / 10240 / 6144 x5120, 1 row | 97.7 / 81.6 / 50.8 us | 65.8 / 54.8 / 35.1 us |
| 5120x6144, 1 row | 50.7 us | 34.6 us |
| 248320x5120 output head | 1,895 us, 147 GB/s | 1,245 us, 223 GB/s |
| Fused gate/up SwiGLU | 267.2 us | 172.3 us |
| Fused GDN 10240+6144 / attention 12288+1024+1024 | 128.0 / 113.2 us | 86.1 / 75.9 us |
| 17408x5120, 2 / 3 / 4 rows | 191.9 / 234.8 / 281.8 us | 155.2 / 197.9 / 234.8 us |
| 17408x5120, 5 / 8 rows (wide) | 341.2 / 343.4 us | 280.4 / 282.2 us |
| 17408x5120, 64 / 128 rows (tensor tile) | 2,630 / 4,159 us | unchanged |

The int8 matvec on the same machine reaches 234–252 GB/s, and the new
matvec sits within a few percent of its own load-only time, so little remains
in the single-row path short of fewer bytes. Matvec sums keep the telescoped
coefficient form and its rounding (worst error against F64 1.7e-6 to 4.3e-6
before, 2.3e-6 to 3.7e-6 after on 5,120-17,408-wide rows); exact trits dotted
with the activations are 18 times more accurate but measured 112 against
99 us. Six prompts, 300 greedy tokens, best of two interleaved runs of both
builds: plain decode 22.52 to 31.14 tok/s geomean (1.38x, every prompt
1.37-1.39x), default speculative decode 30.73 to 36.53 (1.19x); greedy text is
byte-identical to the previous build on every prompt, with and without
speculation. Teacher-forced along 128 tokens of this file after a 1,024-token
prompt, next-token KL of the new build against the old is 7.3e-7 mean and
6.3e-6 at worst, top-1 128/128 (rotated Q8 K/V against F16 is 8.9e-6).
Prefill of a 2,484-token prompt runs on the tensor tile and is unchanged
(97.40 to 97.84 tok/s).

A fresh bandwidth control on the M4 Pro reads 512 MiB per pass, sixteen
passes per sample, with a checksum reduction that validates every output.
After warm-up, five samples at 128/256/512 threads per group reach 259-261
GB/s median (read bytes only), with AC power and sleep guards. Against that
streaming-read control, current single-row projections reach 215-225 GB/s
for the large internal shapes, 237 GB/s for fused gate/up, and 248 GB/s for
the output head: roughly 82-86%, 91% and 95%. This is a comparison of useful
packed-weight bytes to streaming reads, not a hardware-counter measurement
of DRAM utilization. The gap is not automatically recoverable: access
patterns and arithmetic differ. Three- and four-row verify blocks reuse
weights and spend more time computing, so their lower weight GB/s does not
by itself identify a memory bottleneck. Optimize verified tokens per second,
not bytes moved for their own sake; retain the mixed-precision head.

Short-prefix decode attention runs in 32-token splits. Below 1,024 tokens
every build takes the SIMD `bo_attn_split` kernel (one SIMD group per KV
head, pair of query heads and split), where 128-token splits gave the GPU only
twelve SIMD groups per 128 tokens of prefix; from 1,024 tokens on nothing
changes. `BonsaiOps::with_short_split_tokens` selects 32 to 128 tokens for
comparisons. One layer on the M4 Pro, median GPU time per call over nine
trials of 64 calls (64-token splits, measured in an earlier round, were slower
than 32 at every short prefix but a tie at 33):

| Layout, prefix | 128-token splits | 32-token splits |
| --- | ---: | ---: |
| F16, 64 | 28.8 µs | 11.9 µs |
| F16, 128 | 34.6 µs | 12.2 µs |
| F16, 512 | 35.3 µs | 15.0 µs |
| F16, 1,023 | 46.1 µs | 21.7 µs |
| Q8, 33 | 12.7 µs | 12.7 µs |
| Q8, 128 | 37.1 µs | 13.1 µs |
| Q8, 512 | 38.8 µs | 19.4 µs |
| Q8, 1,023 | 42.1 µs | 27.5 µs |
| F16 / Q8, 1,024 (control) | 20.9 / 20.3 µs | 20.9 / 21.3 µs |
| Q8, 2,048 (control) | 34.8 µs | 35.3 µs |

Only the summation order changes. Both sizes match the F64 reference within
2e-5 (`split_gqa_matches_f64_across_boundaries_and_shorter_workspace_reuse`).
Teacher-forced on the real model along 64 greedy tokens after three prompts
of this repository's text, against 128-token splits
(`short_attention_splits_track_128_token_splits_next_token_distribution`):

| Cache, prompt | Mean next-token KL | Worst step | Top-1 |
| --- | ---: | ---: | ---: |
| F16, 128 tokens | 1.5e-8 | 4.8e-7 | 64/64 |
| F16, 512 tokens | 9.7e-9 | 1.5e-7 | 64/64 |
| F16, 896 tokens | 7.6e-9 | 3.9e-8 | 64/64 |
| Q8, 128 tokens | 8.8e-7 | 1.1e-5 | 64/64 |
| Q8, 512 tokens | 5.6e-7 | 6.8e-6 | 64/64 |
| Q8, 896 tokens | 5.3e-7 | 6.4e-6 | 64/64 |

128-token splits on both sides give exactly zero, so this is the split change
alone. Q8 presumably shows more of it because a perturbed key or value row
that crosses an int8 rounding boundary moves by a whole quantization step; it
is still a tenth of the bar the test asserts (the prefill acceptance bar,
mean at most 1e-5 and at least 63/64 top-1) and of rotated Q8 against F16
(8.9e-6). Whole model, six prompts, 300 greedy tokens, best of two
interleaved runs of both builds while other agents' builds loaded the CPU:
plain decode 29.47 to 30.26 tok/s geomean (+2.7%, every prompt +1.8% to
+5.0%), default speculative decode 37.45 to 37.46. Speculative rounds verify
their two to four tokens with the block kernels, so there only drafting and
undrafted single-token steps take the split kernel. Greedy text on these six
prompts is byte-identical across the builds in both modes. An earlier
evaluation rejected the change because greedy text then differed from the
previous build on two of six prompts, but byte identity counts near-tie
flips, not error, so it is judged by the teacher-forced bar like the other
numerical changes in this document.

Q8 attention now shares each dequantized K/V tile across two to four query
tokens. Their six GQA heads are packed into eight-query tensor tiles: four
tokens fill three tiles instead of leaving two lanes idle in each of four.
Each query retains its own causal prefix. Blocks of five to eight tokens
use multiple groups; crossing a split-size boundary falls back to per-row
attention. Reduction shares split metadata and reads numerator values ahead
without changing the FMA order, then applies the inverse Hadamard transform
and gate in the same dispatch. Workspace adds 96 KiB for gathered queries;
partial storage covers four tensor rows without quadrupling the existing
full-context SIMD allocation.

Full-model timings on M4 Pro, synthetic nonzero Q8 caches, best of two
interleaved runs (six timed samples after two warm-ups per case), milliseconds
per target block:

| Prefix | 1 row, before → after | 2 rows | 3 rows | 4 rows |
| --- | ---: | ---: | ---: | ---: |
| 8,192 | 31.78 → 31.86 | 48.54 → 47.84 | 64.37 → 62.53 | 78.67 → 75.21 |
| 32,768 | 36.91 → 37.14 | 58.83 → 54.92 | 79.33 → 72.91 | 99.27 → 85.31 |
| 131,072 | 55.64 → 54.61 | 95.15 → 83.10 | 134.23 → 112.13 | 171.86 → 121.33 |

All twelve configurations produce identical baseline/candidate logit
fingerprints. The four-row reduction is 4.4%, 14.1% and 29.4% respectively;
single-row differences below 128K are within noise. These are synthetic-cache
target execution costs, not real long-prompt generation rates, acceptance
measurements or context-quality results. Numerical tests compare causal
blocks of two through eight rows against F64 attention, and check four-row
attention at the 262,144-token limit with an asymmetric final-token probe.
Six real short prompts, 300-token cap, plain and default speculative modes,
two interleaved runs: all 24 baseline/candidate pairs have identical text and
token counts. Best-of-two geomeans are 32.10 → 32.32 tok/s plain and
39.17 → 39.05 speculative, effectively unchanged at short context. The
installed mixed-precision MTP head is unchanged.

The two-/three-token split kernels subsequently combine all 12/18 packed query
rows into one 16-/24-row matrix operation per K/V tile, rather than issuing one
operation per eight rows. This keeps each SIMD group's tile slice reusable
across every packed row. Interleaved kernel-only GPU timings, microseconds:

| Prefix | 2 rows, before → after | 3 rows, before → after |
| --- | ---: | ---: |
| 1,024 | 35.1 → 33.5 | 49.1 → 47.6 |
| 8,190 | 189.3 → 168.8 | 271.0 → 244.9 |
| 131,072 | 2,378 → 2,129 | 3,358 → 3,138 |
| 262,140 | 4,700 → 4,211 | 6,577 → 6,260 |

Every partial output word matches the prior kernel in twelve configurations;
reversed-order runs confirm the gain. These are additional **kernel-only**
measurements, not updated full-model numbers for the table above. Four rows
retain the previous kernel: combining them regressed 1–2.5% at 128K–262K.
Halving tile storage to gain occupancy cost 13–25%; extra SIMD groups,
transposing K and alternative split sizes also lost. Single-row attention
retained about 74% of its latency with K/V loads removed in an ablation, so
unused memory bandwidth alone does not establish recoverable decode speed.

Causal Q8 prefill blocks (more than eight tokens) ending past 1,024 positions
pack the same way: `bo_attn_tensor_rows_q8` gives one threadgroup four
consecutive tokens of one KV head, so their 24 query rows share each
dequantized K/V tile in one 24-row product, where `bo_attn_tensor_q8`
dequantized every tile once per query head for an 8-row product. Per row the
operands, causal limit, online softmax and division are unchanged, and all
128 x 6,144 outputs are bitwise equal to the previous kernel at 0, 1K, 4K, 8K,
32K and 128K prefixes. The workspace's gathered-query buffer grows from
96 KiB to 3 MiB once its context exceeds 1,024. M4 Pro, 128-token block,
GPU time per layer (median of five, two interleaved runs) and whole-model
block time (best of three, two interleaved runs, synthetic cache):

| Prefix | Attention, one layer | Whole 128-token block |
| ---: | ---: | ---: |
| 1,024 | 1,007 → 975 µs | not measured |
| 8,192 | 7,438 → 6,948 µs | 1,073 → 1,064 ms |
| 32,768 | 29,730 → 27,595 µs | 1,434 → 1,399 ms |
| 131,072 | 125,681 → 112,402 µs | 2,982 → 2,785 ms |

Short-prefix blocks (0 positions: 955 ms either way) keep the per-head kernel.
A greedy 13,239-token prompt (96 tokens, with and without MTP) produced
identical token IDs on both kernels; its prefill rate moved only within noise
(120.5-121.3 tok/s), since attention is a small share of blocks below 16K.
Remaining headroom, from the same runs: the GPU was busy for 99.1-99.5% of
wall time, so host encoding and synchronization cost under 1%; a cold 128-row
block (955 ms, about 3.3 T multiply-adds per second over the 25.8 G ternary
weights) runs near the 8x8 multiplies' measured 3.9 T peak; long-prefix
prefill attention does 2.1e11 multiply-adds per layer at 128K in 112 ms
(1.8 T/s), half that peak; per-tile dequantization and softmax are the likely
remainder (not profiled per stage).

Profiled stage by stage afterwards (temporary kernel variants that each drop
one stage, 128-token block, kernel GPU time, M4 Pro), dequantization and
softmax turned out to be the smaller part. At 8K / 32K / 128K the kernel took
6.9 / 27.7 / 112 ms; dropping the QK product saved 47%, the P.V product with
its rescale 36%, per-tile dequantization 10-13% (of which only 2-5% came back
when every tile was re-read from one cached position, so it is ALU and
threadgroup stores rather than bandwidth: 17 KB of Q8 K/V per tile is about
20 GB/s per core), and the softmax 6-7%. P.V alone runs at 2.8 T
multiply-adds per second and QK alone at 1.8 T: each SIMD group owns a 24x8
score tile over a 256-long reduction, three dependent accumulator chains,
against 24x64 over 32 for P.V. Reading every Q row from one shared address,
or Q as half, saved only 2-4%, so the Q operand's device reads are not the
limit. Occupancy is set by threadgroup memory: 19.7 KB (a 16 KB half K/V tile
plus 24 float score rows) admits four threadgroups per core (padding to
23.8 KB leaves three, to 28 KB two), 80 on 20 cores, so a 128-token block's
128 threadgroups run in 1.6 waves; sweeping the token count at 32K gave flat
steps (16 groups 9.1 ms, 80 groups 15.9 ms, 128 groups 28.3 ms), about 10%
lost to the partial second wave and a per-core throughput that grows only 9%
from three resident groups to four.

Two bitwise-preserving changes followed:

- Q8 tiles dequantize with exact half arithmetic instead of quarter-rate
  integer-to-float conversion (`bonsai_q8_dequant4`): the biased byte is
  OR-ed into a half mantissa (1024 + code + 128), 1152 is subtracted exactly,
  and one half multiply by the scale rounds the exact product once, as the
  float product's conversion did. All 256 codes against all 65,536 scale bit
  patterns give identical bits. This applies to every Q8 tensor kernel.
- The head-packed prefill kernel skips the per-element accumulator rescale
  when every row's correction in the tile is exactly 1 (the running maxima
  did not move), since multiplying by 1 is the identity; a threadgroup flag
  set by any row whose correction differs decides it. At long prefixes almost
  every tile skips 48 threadgroup loads and multiplies per thread.

All 128 x 6,144 outputs stay bitwise equal to the previous kernel's at 0, 1K,
4K, 8K, 32K and 128K prefixes. Whole-model 128-token block, M4 Pro, synthetic
Q8 cache, recurrent state reset before each block, kernels selected in one
build, best of three interleaved samples:

| Prefix | Before | Exact dequantization | Plus rescale skip |
| ---: | ---: | ---: | ---: |
| 0 | 957.7 ms | 954.9 ms | 955.7 ms |
| 8,192 | 1,065.1 ms | 1,063.5 ms | 1,060.7 ms |
| 32,768 | 1,398.5 ms | 1,392.7 ms | 1,369.3 ms (-2.1%) |
| 131,072 | 2,762.4 ms | 2,737.8 ms | 2,632.1 ms (-4.7%) |

Kernel time alone moved by -1 to -2% for the dequantization and a further
-4 to -5% for the skip at 8K-128K (interleaved medians, about 3% run-to-run
noise at 128K). The skip depends on finite scores: a block whose queries are
non-finite rescales every tile, as before. A greedy 13,252-token prompt
(96 tokens, cold isolated prompt caches) produced identical token IDs on the
previous build and this one, with MTP (same acceptance, 46 drafts in 44
rounds) and without speculation, and the two modes agree with each other;
prefill moved from 120.8-121.1 to 121.9-122.6 tok/s, attention being a small
share of blocks at 13K.

Rejected, kernel time against the kernel of the time: issuing each tile's
K/V loads one phase early (V during QK, the next K during P.V), +15-16%
from the registers held across the products; one 8-row product per SIMD
group with Q in a cooperative register tensor (three, four or six SIMD
groups), 14-42% slower and not bitwise (the products' summation follows
the operation's shape); a K·Qᵀ product, 1-2% faster but not bitwise; a
transposed K tile, +50%; half-dimension tiles (QK accumulated over two
128-wide halves and P.V into two 128-column accumulators, bitwise, 11.4 KB,
seven groups per core so a block runs in one wave), +1-4% from twice the
barriers and half-size products; P.V as two 16-key products, +30%; QK as
two 16-key products, unchanged; softmax scores kept in registers, -1%
alone but slower combined with the rescale skip. Split-K (flash-decoding)
over the prefix would fill the partial wave but changes the summation order
for at most the ~10% wave loss; it was not pursued.

Two- to four-row projections (speculative verify blocks, short prefill
chunks, batched decode) then got cheaper with bit-identical outputs. Trits
come from balanced prefixes, prefix(p) - (3^p - 1) / 2, whose offsets fold
into the rounding FMA's subtraction, so trit n is one FMA of two of them
with no separate "- 1". Two-row blocks also hold both input rows in
registers and decode each weight row just before use (bonsai_small_batch.h
explains why three and four rows keep the other loop order). Kernel GPU
time on an M4 Pro from `local-metal/tests/bonsai_throughput.rs`, µs, before
and after (interleaved A/B runs of both kernel sets agree within 1%):

| Projection | 2 rows | 3 rows | 4 rows |
| --- | ---: | ---: | ---: |
| 17408x5120 | 156.3 → 139.7 | 193.3 → 187.1 | 231.7 → 228.3 |
| 5120x17408 | 158.1 → 143.0 | 196.5 → 188.6 | 238.1 → 233.6 |
| 12288x5120 | 111.5 → 100.4 | 138.6 → 135.5 | 165.2 → 163.5 |
| 10240x5120 | 94.1 → 85.5 | 118.2 → 113.1 | 139.3 → 137.1 |

Single rows and the five- to eight-row matrix kernel are unchanged. Outputs
of 1 to 24 rows on four shapes, with 2- and 4-byte-aligned matrices, are
bitwise those of the previous kernels, so greedy text and logits are too.
Five prompts, 300 greedy tokens, best of two interleaved runs of both
builds: default speculative decode 36.05 to 38.06 tok/s geomean (+5.6%,
every prompt +3.2% to +6.9%), plain decode 31.34 against 31.25 (unchanged
kernels); text byte-identical across builds and modes.

Verification itself got cheaper in two steps, measured on an M4 Pro. The scalar
small-batch kernel now holds its decoded trits in half registers, exact for
-1/0/+1 and widened into the same F32 FMAs, so outputs stay bit-identical:
2/3/4 rows went from 200/256/324 to 191/235/281 us on 17408x5120 and from
197/259/344 to 194/237/286 us on 5120x17408. Blocks of five or more rows use a
simdgroup-matrix kernel that covers up to eight rows in one pass for 340-348 us
whatever the row count, where the former code issued two re-streaming
dispatches costing 452-720 us; its summation order differs, with gaps of
1.5e-7 to 2.2e-7 of the largest output. A whole-model verify block at a
1,024-token prefix went from 108.4 / 257.4 / 452.8 / 648.2 ms to 95.6 / 166.1 /
267.6 / 372.2 ms at 4 / 8 / 16 / 24 rows, and the small-batch range now reaches
60 rows (860.5 ms against the 64-token tile's 876.6 ms), so n-gram drafts are
padded to fill a tile only past 60 rows. Greedy output is
byte-identical to before and to plain decoding. With `tools/benchmark.py`,
best of three, the planets prompt went from 23.58 to 24.74 tok/s at
depth 3 (19.02 to 19.18 without speculation), and a prompt asking for a
short function three times verbatim from 25.26 to 31.95 tok/s.

Blocks of 5 to 128 rows then got cheaper in three steps. The 48x5120 BF16
alpha/beta projections switched at eight rows to a token-tiled GEMM that gave
each matrix two threadgroups: 0.6 ms per dispatch whatever the row count, 56
ms of an eight-row verify block once concurrency stopped hiding it. They now
run one dispatch per layer for both matrices, four rows per SIMD group sharing
each weight load, every row bitwise the single-row matvec's: 11.5 us for 2 to
8 rows and 50 us at 64, against 1.18-1.37 ms tiled and 26-222 us per row. The
wide PTQ1 kernel loads each lane's two activations as one `float2` and no
longer zeroes rows past the block (row m of X reaches only row m of the
product), bit-identical: 275 to 237 us on 17408x5120 and 293 to 281 us on
5120x17408 at eight rows. That made eight-row passes cheaper than the tensor
tile up to 128 rows, so every verify block and prefill chunk now runs on the
small-batch kernels. Whole-model block at a 1,024-token prefix (Q8 K/V, device
argmax), best of five, ms:

| Rows | 2 | 4 | 5 | 8 | 12 | 16 | 20 | 32 | 48 | 64 | 65 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 48.2 | 73.8 | 85.6 | 115.8 | 183.0 | 199.5 | 266.3 | 369.7 | 526.1 | 786.3 | ≥1,224 |
| After | 45.5 | 72.2 | 73.7 | 79.6 | 145.4 | 150.8 | 218.0 | 298.1 | 445.2 | 593.5 | 631.8 |

A 128-row block without logits went from 1,221 to 1,155 ms. Profiled with
each dispatch alone, PTQ1 projections are now 83-90% of a 5- to 65-row block
(the output head 4-5%, attention 1.5-5.6%, recurrence 1.3-1.7%, alpha/beta
0.5-0.8%); an eight-row block costs 10% more than a four-row one and the
block cost is close to ceil(rows / 8) times 74 ms. The wide kernel is near
its limit: eight rows of 17408x5120 in 237 us is 3.0 T multiply-adds per
second plus the trit decode. Prefill and verify bits change
(alpha/beta order now the matvec's, projections of 61-128 rows no longer on
the tile). Teacher-forced along 128 tokens of this file after a 1,024-token
prompt, next-token KL against the old build is 5.1e-6 (decode after
prefill), 5.9e-6 (8-row verify blocks) and 5.7e-6 (20-row), top-1 128/128;
against a reference that runs the prompt one token at a time, the new build's
KL is 1.37e-5 / 1.31e-5 / 1.35e-5 where the old build's was 1.52e-5 / 1.50e-5
/ 1.50e-5, so both sit at the block-versus-token gap and the new one is
slightly closer. Five prompts, 300
greedy tokens, best of two interleaved runs of both builds: default
speculative decode 37.80 to 37.99 tok/s geomean (+0.5%; its verify blocks
are two to four rows), plain decode 31.00 against 31.01. Two copy-edit
prompts (about 50 lines of this repository's Rust, rename one identifier,
600 tokens), where n-gram lookup verifies long drafts: 72.8 to 90.5 and 74.0
to 89.8 tok/s (+24%, +21%). A 2,613-token prompt prefills at 106.9 against
96.8 tok/s. Run cold (no prompt cache), greedy text is byte-identical across
builds and with and without speculation on six of the seven; on the train
arithmetic prompt the new build without speculation words one line
differently at character 544 ("So, the time is:" for "So,"), the same
answer, where the old build in both modes and the new one with speculation
agree.

Blocks of 56 rows or more (every prefill chunk but a short last one) then
moved to a large-batch kernel. The wide kernel decodes each weight once per
eight rows and loads an activation tile for every 8x8 multiply. The new one
gives a threadgroup 64 tokens and 64 output rows: it decodes each packed
block once into threadgroup memory as trit times scale in half (exact), and
each of four SIMD groups multiplies a 32-by-32 quarter with sixteen
accumulators, so each tile load feeds four multiplies. Activations stay F32
(the F32-by-half multiply keeps the F32 operand, which the 1000.125-input
test checks) and so does the accumulation; only the summation order changes.
On this M4 Pro every 8x8 simdgroup multiply peaks near 3.9 T multiply-adds
per second, whatever its operand types (F32, F16, BF16, half into F32). The
wide kernel ran at 3.0 T, the large-batch kernel runs at 3.4 T, and with its
decode skipped after the first block it measured 3.5 T, so the multiplies
set its cost. Kernel GPU time, µs, small-batch routing against the new
kernel (same run, weights rotating through 512 MB):

| Projection | 64 rows | 128 rows | 256 rows | 512 rows |
| --- | ---: | ---: | ---: | ---: |
| 17408x5120 | 1,916 → 1,705 | 3,842 → 3,348 | 7,977 → 6,555 | 15,704 → 13,015 |
| 5120x17408 | 2,296 → 1,754 | 4,583 → 3,362 | 8,275 → 6,617 | 16,635 → 13,162 |
| 12288x5120 | 1,362 → 1,229 | 2,726 → 2,402 | 5,661 → 4,656 | 11,095 → 9,154 |
| 10240x5120 | 1,134 → 1,001 | 2,277 → 1,956 | 4,686 → 3,853 | 9,225 → 7,781 |
| 6144x5120 | 710 → 650 | 1,409 → 1,235 | 2,942 → 2,418 | 5,637 → 4,696 |
| 5120x6144 | 692 → 625 | 1,381 → 1,197 | 2,857 → 2,353 | 5,569 → 4,667 |
| 1024x5120 | 155 → 197 | 301 → 292 | 677 → 516 | 1,126 → 876 |

A tile that is mostly empty costs what a full one does (17408x5120: 1,698
at 56 rows; 32 rows would take 1,716 against the wide kernel's 957), so
blocks below 56 rows stay on the small-batch kernels and so does a remainder
of fewer than 56 rows past the last whole tile. 1024x5120 has only 16
threadgroups per 64 tokens and loses at 64 rows; it is the K/V projection of
the 16 attention layers, about 0.3% of a 64-row block, and kept on the same
routing. Whole-model blocks (`prefill_block_timings`, no logits, mean of
five after a warm-up), ms:

| Rows | 32 | 48 | 64 | 96 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 291.7 | 437.5 | 582.0 | 868.0 | 1,153.5 |
| After | 289.7 | 433.7 | 504.1 | 797.9 | 984.2 |

Cold prefill (prompt cache deleted, no speculation, this repository's
text), tok/s, interleaved runs of both builds, best of two: 512 tokens 100.7
to 116.9, 2,048 tokens 107.5 to 126.3, 4,096 tokens 106.2 to 124.8 (one
run), 8,192 tokens 104.8 to 123.3, 32,767 tokens 92.1 to 105.9 (one run;
attention takes a growing share). The 16 greedy tokens after each prompt up
to 8,192 tokens are the same in both builds. Teacher-forced
along 64 greedy tokens after 1,024- and 8,192-token prompts
(`large_batch_prefill_tracks_small_batch_next_token_distribution`), next-token
KL against the small-batch kernels is 5.5e-6 and 2.1e-6 mean (2.1e-5 and
1.6e-5 at worst), top-1 64/64 both. Decode and verify blocks below 56 rows
run the same kernels as before: six prompts, 300 greedy tokens, plain decode
30.52 against 30.39 tok/s geomean and speculative 38.02 against 38.12, text
byte-identical in both modes (the six chat prompts are 27-45 tokens).

If the SDK lacks the tensor kernels or the device rejects their attention
pipelines, startup prints the reason to stderr (`metal: tensor attention
kernels unavailable, ...`) and continues on the slower SIMD kernels, which
the policy's `attention_kernel: simd_f32` also shows.

With the default thresholds no block reaches the prefill tile (that takes
more than 128 rows and fewer than 56), so `BonsaiKernels::new` compiles its
three tensor pipelines on first use, under overridden thresholds, instead of
at every startup: 2.02 → 1.92 ms median over twenty interleaved launches
with a warm shader cache (a cold-cache launch was not measured). An SDK
without the tensor tile prints the reason and uses the SIMD tile.

Prefill blocks then grew from 128 to 512 rows (`PREFILL_CHUNK`,
`MAX_PREFILL_TOKENS`). Rows are independent in every projection and attention
kernel (attention walks absolute 32-key tiles and groups rows from the block's
start), but the recurrence keeps its state in F32 registers for a whole block
and rounds it to F16 when it stores it, so a 512-row block rounded a quarter as
often. That moved a 2,051-token prompt's first greedy token in an A/B run, so
`bonsai_gdn_rows` now rounds its registers every 128 rows from the block's
first (`bonsai_gdn_round_rows`), which reproduces four 128-row blocks;
`prefill_block_size_leaves_identical_logits` checks the logits bitwise after
2,600 tokens. Verify blocks and folded prefill chunks are shorter than 128 rows
and unchanged. Shared-prefix checkpoints keep their 128-token minimum
(`SHARED_PREFIX_MIN_TOKENS`). Larger blocks fill more of the GPU per dispatch:
the head-packed attention kernel runs four threadgroups per 4 tokens and KV
head, so 128 rows made 1.6 waves on 80 resident threadgroups and 512 rows make
6.4, and projection shapes such as 6144x5120 leave fewer partial waves.

Cold prefill, `tools/benchmark.py --prefill`, raw repository text,
no speculation, `b09e22f` (128-row blocks) against main `a4cf247`
(512-row blocks with the 128-row rounding), interleaved, best of two rounds
(128K: one round), M4 Pro, 2026-10-22, with no other GPU job running (every
job held `lockf -k /tmp/local-ai-gpu.lock`). Every sample of a size, both
builds, cold and restored, generated the same 16 greedy tokens:

| Prompt | 128-row blocks | 512-row blocks | Warm restart (disk restore), TTFT |
| ---: | ---: | ---: | ---: |
| 2,047 tokens | 128.3 tok/s (16.0 s) | 133.7 tok/s (15.3 s) | 0.54 s |
| 8,204 tokens | 125.7 tok/s (65.3 s) | 131.4 tok/s (62.4 s) | 0.66–0.68 s |
| 32,796 tokens | 110.0 tok/s (298.1 s) | 115.6 tok/s (283.6 s) | 1.15 s |
| 131,146 tokens | 69.7 tok/s (1,882 s) | 74.8 tok/s (1,754 s) | 3.8–4.0 s |

So 512-row blocks prefill 4–5% faster to 32K and 7% faster at 128K (one
round); the earlier numbers
(8,201 tokens 122.4 → 130.3, 32,706 tokens 109.2 → 115.2, 131,231 tokens 70.9
→ 76.4, taken while another GPU job could overlap) were within a few percent
of these. Round to round, one build varied by up to 6% at 8K (118.1 and 125.7
tok/s for `b09e22f`), so read single-round differences below that as noise.
An earlier block-size sweep at 8,201 tokens (possibly contended) measured
128.7 tok/s at 256 rows, 130.3 at 512 and 129.4 at 640. Peak footprint for a
cold 8,201-token prompt rose from 1.56 to 1.90 GiB (`/usr/bin/time -l`): row
scratch is sized for 512 rows.

End to end through `serve`, `tools/benchmark.py --ttft` (a ~5.9K-token system
prompt with six tool schemas, then a ~2K-token tool result, `reasoning_effort`
`none`, time to first streamed delta), 128- against 512-row blocks, one run
each, same conditions:

| Step | Prompt / reused tokens | 128 rows | 512 rows |
| --- | ---: | ---: | ---: |
| Cold first request | 5,881 / 0 | 46.9 s | 44.9 s |
| New session, same system prompt and tools | 5,883 / 5,862 (GPU) | 0.48 s | 0.47 s |
| Follow-up with a 2K tool result | 7,799 / 5,866 (GPU) | 16.2 s | 15.5 s |
| After a restart: new session | 5,881 / 5,866 (disk) | 1.02 s | 1.01 s |
| After a restart: the same follow-up | 7,799 / 7,798 (disk) | 0.53 s | 0.53 s |

The disk tier already persists the system-prompt-and-tools boundary across
restarts, so a restarted server's new session starts in about a second rather
than 45. What an agent still pays is new text: a tool result prefills at the
short-prompt rate (about 125-130 tok/s), and at 2K-8K tokens prefill is set by
the ternary projections at about 85% of the 8x8 multiplies' measured peak.

### Neural Engine prefill FFN

On Apple M4 and later, prefill runs every layer's FFN (gate and up
projections, `SiLU`, the checkpoint's 17,408-wide rotation, down
projection) and the leading rows of each mixer's output projection on the
Neural Engine while the GPU runs the rest of the mixers (Gated `DeltaNet`
and full attention), as Kapitsa does
(`local-engine/src/bonsai_native/ane_split.rs`, `local-metal/src/ane.rs`,
`shaders/bonsai_ane.metal`; adapted from Kapitsa `a6a7bca`, Apache-2.0, see
`THIRD_PARTY_NOTICES.md`). It is on by default with no flag; correctness
never depends on it.

- **Program.** One MIL program with FFN functions for 128, 256, 384 and
  512 rows and a second with each mixer output projection's leading 512 to
  3,072 rows (in steps of 512) for those row counts, compiled on a
  background thread from model load through the private
  `AppleNeuralEngine.framework` (every selector's type encoding checked
  before its first call) and cached by the daemon under
  `$(DARWIN_USER_CACHE_DIR)/local-ai-bonsai-ffn/<hash>`; a held compilation
  that fails to load is purged and compiled again. The rotation is the
  program's only constant (a grouped convolution, 36 MB); `SiLU` is
  `g/2 (1 + tanh(g/2))`.
- **Precision: int8 weights x int8 activations.** Weights are surfaces with
  one fp16 scale per row; the GPU converts each layer's ternary blocks into
  one of two weight sets (0.54 GB together) while the ANE runs the previous
  layer from the other, so there is no resident int8 copy. A ternary row is
  its trits times block scales, so its codes are the trits times
  `rint(127 s_b / max |s|)`. Activations are packed int8 per token; the
  rotated intermediate is re-quantized per token on the ANE, which returns
  the down projection before its scales plus a scale row that the GPU joins
  in F32.
- **Wavefront.** A prefill block of up to 1,024 rows runs as two halves of
  at most 512 (the GPU scratch rows; the first half is half the block
  rounded up to 128). Per layer the GPU runs half A's mixer and hands its
  FFN to the ANE, joins half B's previous FFN, runs half B's mixer and hands
  that over, converts the next layer's weights, then joins half A: the ANE
  works on one half while the GPU runs the other's mixer. Recurrent state
  and K/V flow from half A to half B in the GPU's order. Blocks under 192
  rows (a short prompt or a short last block) and everything that is not
  prefill (decode, verification, folded chunks, capture) stay on the GPU.
  The MTP head ingests each half after the last layer.
- **Balance.** The ANE's FFN (about 19 ms per 512-row half) finishes before
  the GPU's mixer and weight conversion (21-27 ms, growing with context), so
  each half's mixer output projection (`ssm_out` or `attn_output`, 5,120 x
  6,144) is split by output row: the ANE computes the leading rows from int8
  weight-set surfaces between FFNs while the GPU computes the rest, and the
  GPU joins both in F32. The ANE takes 0.95 ms per 1,024 rows of 512 tokens,
  about the GPU's rate, so the gain is the overlap. The split (0 to 3,072
  rows in steps of 512, the same for every layer of a block) is chosen per
  half-block size from measured block times as prefill runs: contiguous
  blocks alternate between the split and a probe one step up or down, and a
  probe that ran faster per row than the mean of the equal-sized blocks on
  either side, by more than 0.2%, becomes the split. Averaging the
  neighbours cancels the growth of block time with context, so the split
  follows a long prompt as attention's GPU work grows. A block whose GPU
  waited over 0.5 ms per layer on the ANE's FFNs takes a step away. Blocks
  starting in a prompt's first 1,024 positions take no output rows and
  teach nothing: there attention is cheapest, the GPU's half-block mixer
  about as short as the ANE's FFN, and output rows made the ANE the
  bottleneck (a fresh 2,049-token prompt's first block took 3.02 s with
  3,072 rows against 2.74 s with none, its second 2.71 s with 2,560 rows
  against 2.79 s with 512; the ANE's jobs ran their usual time in both). It
  starts at 3,072 rows and the learned splits persist in `out-split.json`
  under the program cache. With every layer fixed at one split, 4,096-token
  prompts measured (`ane_out_split_timings`, best of two):

  | ANE rows | 0 | 512 | 1,024 | 1,536 | 2,048 | 2,560 | 3,072 | 3,584 | 4,096 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | tok/s | 357.6 | 356.4 | 359.9 | 365.5 | 368.0 | 370.4 | 373.7 | 373.8 | 371.0 |

  More int8 rows raise the KL (below), and 3,072 rows ran as fast as
  3,584, so the steps stop there. At 16K tokens 3,584 rows measured 306.4
  against 298.9 tok/s with none, but other agents' load moved the points in
  between by 5%, so that sweep only bounds the gain.
- **Ordering.** Two `MTLSharedEvent`s carry GPU-to-ANE and ANE-to-GPU order
  (set on each `_ANERequest`), so no host thread waits mid-block. A
  watchdog retires the handoff when an evaluation runs 2 s past its release
  (one takes about 19 ms) or a block runs 300 s, raising both events so
  every GPU wait passes.
- **Fallback.** No M4-class chip, no framework, a failed compile or a
  start-up check (layer 0's FFN, and its output projection with the largest
  split, on 256 pseudo-random rows against the GPU's, at most 10% relative
  difference; int8 x int8 measured 0.42% for the FFN) leaves
  prefill on the GPU with
  the reason in `neural_engine_prefill`. A block whose ANE work fails,
  times out or returns non-finite rows (the join flags them), or whose
  logits are non-finite, is restored from an 81 MB recurrent checkpoint
  refreshed before every block and rerun on the GPU, and the model keeps
  prefill on the GPU from then on (stderr says why).
  `ane_fault_reruns_the_chunk_on_the_gpu` skips one evaluation and checks
  the rerun's logits bitwise against GPU prefill.
- **Keepalive.** The ANE sleeps after about a second idle and its first
  evaluation then takes 70-120 ms longer. Within 60 s of the last block a
  side thread runs one 128-row FFN evaluation (the smallest) every 0.8 s;
  the ANE offers no idle or wake notification to wait on instead.

Cold prefill, `tools/benchmark.py --prefill --no-speculation`, raw
repository text, main `d25b277` against this build, interleaved, best of two
rounds (32K: one round), M4 Pro, 2026-10-27, every job holding
`lockf -k /tmp/local-ai-gpu.lock`. Other agents' builds and downloads were
running (`--allow-busy`, load average 7-19), so read small differences as
noise. Every sample of a size, both builds, cold and restored, generated the
same 16 greedy tokens:

| Prompt | GPU only (main) | FFN on the Neural Engine | Cold TTFT |
| ---: | ---: | ---: | ---: |
| 2,049 tokens | 134.4 tok/s | 361.7 tok/s | 15.2 → 5.7 s |
| 8,200 tokens | 131.6 tok/s | 337.1 tok/s | 62.3 → 24.3 s |
| 32,726 tokens | 113.1 tok/s | 225.6 tok/s | 289.5 → 145.1 s |

The gain shrinks with context because only the FFN moves: full attention
grows with the prefix and stays on the GPU, which sets the pace once its
mixers outlast the ANE's 19 ms per 512-row FFN. Round 1 at 2K was slower
for both builds (110.9 and 215.9 tok/s) under the other agents' load. A
process loads the daemon's held compilation in about 30 ms; the first
wavefront block waited 135 ms for the background compile thread (which also
generates the 36 MB rotation) and the start-up check. Restored (warm) TTFT
is unchanged (0.58-1.08 s at 2K-8K, 2.9-3.7 s at 32K).

Through `serve` (`tools/benchmark.py --ttft`, default settings, one run
each, same conditions), time to the first streamed delta:

| Step | Prompt / reused tokens | GPU only (main) | FFN on the Neural Engine |
| --- | ---: | ---: | ---: |
| Cold first request | 5,846 / 0 | 46.38 s | 17.82 s |
| New session, same system prompt and tools | 5,848 / 5,827 (GPU) | 0.46 s | 0.45 s |
| Follow-up with a 2K tool result | 7,805 / 5,845 (disk) | 15.92 s | 6.88 s |
| After a restart: new session | 5,846 / 5,831 (disk) | 2.45 s | 1.02 s |
| After a restart: the same follow-up | 7,805 / 7,804 (disk) | 0.43 s | 0.43 s |

Restores do not prefill, so only the cold request and the tool result move
(the 2.45 s restart is main's one run; it measured 1.01-1.02 s before).

Decode is unchanged: 36.20 against 36.14 tok/s best of two interleaved
rounds with the default speculation (`tools/benchmark.py --rounds 2`, same
conditions). Peak footprint (`/usr/bin/time -l`, cold scratch `HOME`) rises
by the weight sets, the second half's rows and the rerun checkpoint: 1.51 →
2.25 GB for an 8,200-token prompt (`--no-speculation`, 16 tokens) and
2.68 → 3.41 GB for a 16,361-token prompt plus 500 tokens with speculation
(prefill there 124.1 → 296.5 tok/s with head ingestion; decode 44.1 and
43.2 tok/s, one run each).

The output projection split and the 192-row minimum, against the FFN-only
build before them (`a3ea7a0`), same tool and conditions (best of two
interleaved rounds; 32K one round; 2026-10-27; main `d25b277` from an
earlier round of the same day):

| Prompt | GPU only (main) | FFN on the ANE | FFN and output rows | Cold TTFT (FFN only → split) |
| ---: | ---: | ---: | ---: | ---: |
| 303 tokens | 114.3 tok/s | 113.7 tok/s | 171.1 tok/s | 2.67 → 1.77 s |
| 510 tokens | 128.6 tok/s | 126.9 tok/s | 288.4 tok/s | 4.02 → 1.77 s |
| 2,049 tokens | 133.4 tok/s | 357.6 tok/s | 364.6 tok/s | 5.73 → 5.62 s |
| 8,200 tokens | 130.2 tok/s | 333.6 tok/s | 346.5 tok/s | 24.58 → 23.67 s |
| 32,726 tokens | 112.8 tok/s | 244.2 tok/s | 252.3 tok/s | 134.0 → 129.7 s |

Blocks of 192-511 rows gain the most: they ran on the GPU before (and take
no output rows, being in a prompt's first 1,024 positions). Longer prompts
gain 2-4% from the split. Before the first-position rule, output rows in
the first block cost a fresh 2,049-token prompt 0.3 s (330.2 against
349.3 tok/s).

Through `serve`, a follow-up adding a 408-token tool result to a
5,747-token session (`--ttft` with a 350-token tool result) reached its
first token in 2.30 s against 3.65 s for the FFN-only build (a second run:
2.39 against 3.72 s; main 3.65 s). With the standard 2K tool result the
follow-up took 6.78 against 6.90 s and the cold 5,846-token first request
17.59 against 17.76 s; restart steps are unchanged. Decode is unchanged
(37.55 against 37.58 tok/s, best of two with the default speculation,
while another agent's `rustc` ran). Peak
footprint: 2.23 → 2.32 GB at 8,200 tokens (the output projection rows and
their inputs), and 0.84 → 1.65 GB for a 279-token prompt, which now uses
the Neural Engine.

Next-token distribution (`ane_prefill_tracks_gpu_prefill_next_token_distribution`,
64 greedy tokens teacher-forced after prompts of this repository's text, the
same model with and without the Neural Engine, decode on the GPU in both):
with the FFN alone on the ANE, mean KL 6.3e-5 (max 4.4e-4) after 1,000
tokens and 5.7e-5 (max 2.1e-4) after 8,192; with the output projection
split as shipped, 6.3e-5 (max 4.4e-4; that prompt is one block in the
first 1,024 positions, so no output rows) and 7.4e-5 (max 3.9e-4); top-1
64/64 in every run. Output rows in the first block measured 7.0e-5 after
1,000 tokens at up to 3,072 rows and 1.44e-4 (max 1.06e-3) at up to
3,584, which is why the steps stop at 3,072. This is above the 1e-5 bar earlier prefill changes were held to:
int8 x int8 was chosen for speed. Kapitsa gates its own at 3.84e-4. The
test asserts mean at most 2e-4, no row over 2e-3 and at least 60/64 top-1.
Because the split follows measured timing, the last bits of a prompt's
prefill can differ between runs (within those bounds); decode, and every
path that is not a prefill wavefront, are unchanged. `ane_prefill_feeds_the_head_like_gpu_prefill` checks the
MTP head's keys for each half and the following GPU block within 5%
relative of GPU prefill (measured 0.8%, 1.7% and 1.2% with the FFN alone
on the ANE, 0.9%, 1.2% and 2.2% with the output projection split; leaving half A's
embeddings staged for half B measured 47%) and the same next draft.
`prefill_block_size_leaves_identical_logits` now pins the GPU path, where
block size still leaves the logits bitwise identical.

Single-token decode encodes 835 dispatches per token, down from 1,364, and
every fused kernel is bitwise identical to the dispatches it replaces
(`local-metal/tests/bonsai_fusion.rs`):

- RMSNorm and the input rotation: each 1024-wide rotation threadgroup repeats
  its row's square sum in the same order. 13.4 to 8.0 us.
- FFN gate and up: one SIMD group reduces four rows of each (two before the
  half-prefix decoder), sharing activation coefficients across the eight as
  the plain matvec does, and writes `silu(gate) * up`. 269.1 to 262.1 us.
- A recurrent layer's QKV and gate projections and its two 48-row BF16
  alpha/beta projections: one dispatch, BF16 rows first so their latency
  overlaps the packed rows. 148.4 to 130.3 us. A full-attention layer's Q, K
  and V: 120.7 to 111.3 us.
- Convolution, Q/K L2 normalization and decay/beta: one dispatch for blocks of
  up to 32 rows.

Multi-row FFNs also fuse SwiGLU with the forward Hadamard rotation, removing
one dispatch per layer and the intermediate product write/read; single-row
decode is unchanged. Separate-output and in-place kernels match bitwise for
2, 4, 8 and 64 rows. On the M4 Pro, best warm stage times were 9.6 to 5.9 us
at 2 rows, 63.9 to 37.6 us at 64 rows, and 179.5 to 87.5 us at 128 rows.
A real-model 16-row feature capture in separate/fused/fused/separate order
took 33.438/33.379/33.361/33.506 seconds, with every captured FP16 feature
byte identical. The mean whole-capture difference is only 0.3%; these stage
timings do not establish a broad decode speedup or bandwidth saturation.

The BF16 matvec was bound by one load latency per loop iteration, 45 us for
each 48x5120 alpha or beta projection; it now issues eight iterations' loads
before their in-order fused multiply-adds, also bitwise identical. Kernel
times above are best of three over 200 dispatches with weights rotating
through 600 MB.

A single-row token was then profiled two ways on an M4 Pro (planets prompt,
96 tokens after 16 of warm-up): the real command buffers with their GPU and
host timestamps, and every dispatch in a command buffer of its own (a
one-dispatch buffer costs 2.0 us, subtracted below; the per-kernel sum lands
within 2.5% of the in-pipeline GPU time). PTQ1 projections took 27.5 ms of
32.3; most of the rest was latency, not bandwidth. Three changes, all bitwise
identical (`fused_residual_and_gdn_output_chains_match_separate_dispatches_bitwise`,
`bonsai_recurrent.rs`; logits and hidden rows of a 300-token prefill and 48
decode steps match the old encoding bit for bit):

- Every residual add rides in the next RMSNorm + rotation (`bo_add_rms_fwht`).
  Each threadgroup recomputes its row's sums, so the residual stream alternates
  between `hidden` and `hidden_alt` instead of updating in place, and each
  thread loads its whole 5120-wide row before the in-order square-sum chain
  (the old kernel waited on one load per iteration): 9.1 us for the add and
  norm pair, 4.1 us fused.
- A recurrent layer's output norm, gate, head regrouping and rotation are one
  dispatch (`bo_gdn_post_fwht`, 3.6 to 3.3 us).
- The single-token recurrence runs four SIMD groups (one value row each) per
  threadgroup: 6,144 one-SIMD threadgroups took 21.2 us per layer streaming
  F16 state, 1,536 four-SIMD ones 15.8 us (eight SIMD groups, or two rows per
  SIMD group, were no better).

The block's first command buffer now holds only the input rotation, and
layers 1-3 go in a second: when the first held layer 0 alone, encoding the
other 63 layers (0.25-0.7 ms on a cold core) often outlasted it and the GPU
waited 0.18 ms per token between the two buffers. Dispatches per token fell
from 853 to 677. Per token, ms (kernel time without the 2.0 us per buffer):

| | Before | After |
| --- | ---: | ---: |
| PTQ1 projections (incl. 1.18 output head) | 27.48 | 27.45 |
| Residual add + RMSNorm + rotation | 2.13 | 1.18 |
| GDN recurrence | 1.01 | 0.48 |
| Full attention + K/V preparation | 1.06 | 1.05 |
| FFN and attention-output rotations, GDN post | 0.66 | 0.74 |
| GDN conv / L2 / decay | 0.16 | 0.13 |
| Input rotation, argmax | 0.03 | 0.03 |
| GPU busy (sum of command buffers) | 31.77 | 30.83 |
| GPU idle between a token's buffers | 0.18 | 0.02 |
| GPU idle between tokens | 0.33 | 0.32 |
| Wall time per token | 32.28 | 31.17 |

Between tokens the GPU idles 0.32 ms: 0.13-0.15 ms from its last command to
the host waking, 0.05-0.08 ms of host work (embedding decode and encoding up
to the first commit) and 0.08-0.14 ms from commit to the GPU starting.
Spinning on the command buffer's status instead of blocking cut the first to
0.10 ms and kept the host core fast (0.24 instead of 0.5 ms to encode a
token), 0.1-0.4 ms per token in all, but burns a core for the whole token, so
it was not adopted. Six prompts, 300 greedy tokens, best of two interleaved
runs of both builds: plain decode 30.92 to 32.17 tok/s geomean (+4.0%, every
prompt +3.5% to +5.2%), default speculative decode 38.42 to 39.10 (+1.8%);
greedy text is byte-identical across builds in both modes.

Recurrent state is stored F16 and computed in F32 registers, rounded once
per block. Each decode step read and wrote 302 MB of F32 state, and every
verify block copied it into a snapshot (and back on rejection). Against F32
state, teacher-forced along 512 greedy tokens after an 8,192-token prompt
(`reduced_state_tracks_f32_state_next_token_distribution`): mean next-token KL
7.5e-6 over 64 positions, with no growth along the generation (quarters
1.6e-5, 4.9e-6, 6.4e-6, 3.4e-6), top-1 512/512; on a later build of the
projection kernels 7.1e-6 and 511/512. The largest F32 state value there is
46.2, far inside F16's range, which saturates rather than overflows. Greedy
output on the six benchmark prompts is byte-identical with and without
speculation. Free-running past a 15,920-token prompt it first differed from F32
state at the 188th of 500 generated tokens (two list items swapped order).

Together with the copy-free verify rollback and the in-block greedy argmax
(see Lossless speculation), six prompts, greedy, 300 tokens, best of two,
interleaved, identical tokens in every arm, one build with each change
switchable: speculative decode 34.68 to 36.56 tok/s geomean (+5.4%); removing
one change at a time cost 2.4% (rollback copies), 1.6% (host greedy
selection, 0.68-1.02 ms per round of top-k submissions) and 0.7% (F32 state).
Plain decode, where only the state width and argmax apply, moved 22.18 to
22.36 tok/s, within noise.

Verification blocks use a concurrent compute encoder: every dispatch is
followed by a buffer barrier except within a layer's group of independent
projections. That measured 24.99 to 25.40 tok/s on speculative decode.

Together, on the default benchmark prompt with identical greedy text: plain
decode 18.84 to 21.64 tok/s and speculative decode (depth 3) 23.58 to
25.44 tok/s, interleaved runs of both builds. Removing one fusion at a time
cost: RMSNorm with rotation 3.4%, QKV/gate concatenation 4.0%, BF16 folding
2.3%, SwiGLU 1.3%, convolution 0.8%. The BF16 load change alone took plain
decode from 19.23 to 20.39 tok/s.

### Log-probabilities

A request that does not ask for log-probabilities encodes no extra kernel
and copies no row, so Codex, Oh My Pi and opencode, which never ask, see no
change. Uncontended (2026-10-22, every GPU job under `lockf -k`),
`tools/benchmark.py --rounds 3 --max-tokens 256` decoded its default prompt
(209 greedy tokens to end of turn, default speculation) at 36.93 tok/s best
on `b09e22f`, before log-probabilities support, and 36.89 on main `a4cf247`
(medians 36.90 and 36.85, every run 36.75–36.93), with identical text. An
earlier run of two interleaved rounds of the server built without and with
log-probabilities support (default speculation and sampling, 256-token
answers, seed 11, `reasoning_effort: "none"`, possibly overlapping another
GPU job): decode median 105.69 and
105.85 tok/s (8 runs each, every run 105.2-106.0), cold prefill of
about 2.3K-token prompts median 129.59 and 129.58 tok/s (6 runs each,
129.5-129.7), with byte-identical answers. Asking costs one two-pass
reduction over the 248,320 logits of each selecting row: the same build
decoded at 105.60 tok/s median with `top_logprobs: 0` and 105.51 with 20
(−0.2% and −0.3%, within run-to-run spread), again with the same answer.
Against an f64 reference the kernel's log-sum-exp is within 1e-5 relative
and its top 20 are bit-exact, ties going to the lower id like argmax.

### Sampled decoding

A sampler that is not greedy (the server's defaults are temperature 1.0,
top-p 0.95 and top-k 20) selects from the GPU's top k of each row. The top-k
kernels have the log-probability kernels' shape: each group of 4,096 logits
selects its best k in one dispatch and each row merges its groups' candidates
in a second, where the former kernels sorted 970 chunks of 256 and merged
them in ten more dispatches. GPU time per 248,320-logit row, M4 Pro, 50
selections per command buffer: 170, 177 and 183 µs at k = 1, 20 and 64
before, 11, 43 and 127 µs after. The candidates are bit-exact against a
`total_cmp` reference, ties to the lower id, special bit patterns included.

Every block and batched pass that produces logits now selects each row's top
k in its own command buffer while a sampler draws from the plain top k (no
presence or repetition penalty, no masking grammar; a batch selects the
largest k any of its requests uses, and each takes its prefix), and every
such pass flags non-finite rows with the argmax kernel. Before, each sampled
row submitted a top-k command buffer of its own and waited for it, and every
non-greedy pass scanned its logits on the host for non-finite values (1 MB a
row); head drafting also cloned the request's sampler, about 250 KB of
seen-token flags (2 MB of candidates at `top_k: 0`), only to ask whether it
applies penalties. Penalties and grammars keep the submitted top k, whose
candidates they must see. `tools/benchmark.py --sampled` and
`tools/serve_concurrency.py --sampled` measure the default settings with a
fixed seed. M4 Pro, 2026-10-26, back to back under the GPU lock but with
other builds loading the CPU (load average about 9 of 14), one build before
and after this change, every sampled output byte-identical between them:

| Default sampling, default speculation | Before | After |
| --- | ---: | ---: |
| `bonsai`, planets prompt, 203 tokens, best / median of 3 | 34.89 / 33.83 tok/s | 35.10 / 35.07 tok/s |
| `serve`, 1 stream, 300 tokens | 32.61 tok/s | 33.05 tok/s |
| `serve`, 8 streams, 300 tokens each, aggregate | 74.36 tok/s | 76.25 tok/s |

Greedy decoding, which already selected on the device, measured 37.08 and
37.28 tok/s (`bonsai`), and 35.38 / 74.80 and 35.16 / 73.92 tok/s at one and
eight streams (`serve`, one run each, inside its run-to-run spread), with
identical text.

## Tried and rejected

Entries are measured on the build current when they were tried. Decode rates
of roughly 17–25 tok/s come from earlier kernels (17.5 tok/s plain decode at
[the original build](https://github.com/debanjanbasu/local-ai/commit/b389329),
about 22 before the half-prefix decoder) and compare only within
their own entry, not with current rates.

- Other ways to move mixer work onto the Neural Engine (October 2026,
  with the output projection split in
  [Neural Engine prefill FFN](#neural-engine-prefill-ffn)). One program of
  the four FFN functions and the 32 output projection functions fails
  `ANECCompile()` (it compiles with up to 26 of them, and the 32 alone
  compile in 1.4 s), so they are two programs. Choosing the split per layer,
  from each layer's interval between the halves' FFN handoffs with every
  other layer of a type probing a step, settled recurrent layers at 1,024
  rows while a uniform sweep put the best split at 3,072-3,584: a layer's
  extra ANE rows delay the FFN the next layer joins, so its cost lands on
  its neighbour. The input projections (Gated `DeltaNet`'s QKV and gate,
  attention's Q, K and V) were not moved: they would queue behind the other
  half's FFN on the ANE while the GPU's recurrence or attention waits for
  them, where the output projection's rows land in the ANE's idle time
  after that FFN (not measured).

- 4-bit Neural Engine weights (October 2026, after the
  [Neural Engine prefill FFN](#neural-engine-prefill-ffn)). Ternary rows are
  exactly trits times one fp16 scale per 128 columns, so blockwise int4 (or a
  4-bit palette) with group-128 scales would remove the weight half of the
  int8 path's error. The private-API path offers no runtime 4-bit weight:
  int4 and uint4 `tensor_buffer` inputs are rejected (`InvalidMILProgram`)
  through iOS 17 `dequantize`, `cast` and iOS 19 `dequantize` with scalar or
  blockwise runtime scales; a runtime `constexpr_blockwise_shift_scale`
  crashed `ANECompilerService`; and the `_ANERequest` weights buffer is
  ignored (negating it left the output unchanged). What remains is exact
  int8 trits times runtime fp16 block scales, or 4-bit weights compiled into
  the program as constants. One gate projection segment (17,408 x 2,560),
  ms on the ANE, relative error against an F64 product (3e-4 is the fp16
  arithmetic itself):

  | Weights | 128 rows | 256 rows | 384 rows | 512 rows | Error |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | int8 surface, one scale per row (shipped) | 1.79 | 2.48 | 3.09 | 5.26 | 3.4e-3 |
  | int8 trits x runtime block scales (exact) | 12.33 | 12.84 | 13.38 | 17.71 | 3.1e-4 |
  | Constant blockwise int4 | 0.61 | 1.09 | 1.61 | 3.34 | 3.1e-4 |
  | Constant 4-bit palette | 0.63 | 1.09 | 1.63 | 3.37 | 3.1e-4 |
  | Constant fp16 | 0.63 | 1.13 | 1.60 | 3.36 | 3.1e-4 |

  The exact runtime format is 3.4-6.9x slower, which would leave the ANE far
  behind the GPU's mixers. Constant weights are faster at every size (fp16
  constants as fast as 4-bit ones: the ANE holds constants in its own
  format), but each 2,560-column segment compiles in 3-7 s and loads in
  2.9-6.1 s, the model has 384 of them, and they stay resident (at least
  8.6 GB at 4 bits) instead of the two 0.3 GB weight sets converted per
  layer. int8 with one scale per row stays: int8 weights with fp16
  activations measured a mean KL of about 2e-5 in the spike, so exact
  weights could remove at most about a third of the 7e-5, and smaller
  blocks, the other thing halved weight bytes could have bought, now use the
  int8 wavefront from 192 rows.

- Other Neural Engine FFN configurations (October 2026, a spike before the
  [Neural Engine prefill FFN](#neural-engine-prefill-ffn)): fp16
  activations with int8 weights took 32.7 ms per 512-row FFN on the ANE
  against 18.5 for int8 x int8, longer than the 20-25 ms mixer it has to
  hide behind, for a mean KL of about 2e-5 instead of 6e-5. A 512-row block
  as two 256-row halves left the ANE bound by fetching weights (11.9 ms for
  256 rows); 1,024-row blocks of 512-row halves projected fastest. Splitting
  the FFN's channels between the GPU and the ANE was not needed once the
  whole FFN fit behind the other half's mixer.

- Pooled Bonsai hidden states as retrieval embeddings (October 2026):
  BEIR SciFact, all 300 test queries against a 1,000-document subset (the
  283 relevant abstracts plus 717 random others, title and abstract,
  truncated to 256 tokens). Each text was wrapped in a PromptEOL template
  (`This passage : "…" means in one word:"`) and run through the
  hidden-state capture path the judgment head uses (`capture_features`,
  output-normalized 5120-wide rows). Cosine ranking of the mean over the
  text's rows, mean-centered, scored nDCG@10 0.643 and Recall@100 0.951
  (uncentered 0.649 and 0.944; the final-token row 0.574 to 0.594 and 0.907
  to 0.934). The shipped Tantivy BM25 scored nDCG@10 0.811 and Recall@100
  0.960 on the same documents and queries (0.692 and 0.925 on the full
  5,183-document corpus). Reciprocal-rank fusion of the two (k = 60) lowered
  nDCG@10 to 0.766 for Recall@100 0.987. Embedding also ran at 117–124
  tokens/s on the GPU (about 0.5 documents/s, 34 minutes for the 1,000
  documents), while BM25 indexed the same documents in 4.9 s and answered a
  query in 7 ms. Neither quality nor throughput is competitive, so pooled
  Bonsai states are not used as embeddings. A dedicated embedding model was
  the answer: vector stores now rank `ranker: auto` by EmbeddingGemma 2
  embeddings (see [EMBEDDINGS.md](EMBEDDINGS.md#vector-stores)).
- PTQ1 pipeline descriptor hints (32-thread maximum and the multiple-of-SIMD
  execution-width flag): outputs stayed bitwise identical, but timing shifts
  were within about 2% and moved with the repeated default control. No
  repeatable gain, so the default descriptors remain.
- BF16 recurrent state: mean KL 4.7e-5 against F32 state (511/512 top-1) in
  the test above, six times F16's 7.5e-6 for the same bytes.
- Chunkwise GDN prefill: recurrence is 17.25 ms per 128-row block, only
  1.17–1.29% of block time; perfect removal yields at most 1.013x.
- F16 prefill activations: only 1.04–1.07x projection speedup before conversion,
  below the 1.15x threshold; tile conversion is 1.9–3.5x slower.
- Half-precision PTQ1 decode: exact recurrence is 36–51% slower than FP32.
  (The shipped half decode is different: one rounding FMA per prefix.)
- Other PTQ1 projection decoders, 17408x5120 single row against the 135 us
  floor decoder: a threadgroup lookup table of byte to five half trits, 151 us
  (random threadgroup loads are slower than the arithmetic); the same rounding
  trick in F32, 119 us, and 149-173 us once its constants came from arrays the
  compiler did not fold or the qh trit used it too; half prefixes for the one
  qh trit per lane, 2-3% slower than its floor; 16 rows per SIMD group, 250 us
  (spills). A load-time repack into 2-bit codes was not pursued: the matvec is
  now near its load-only time, and 2 bits per weight would move 14% more bytes
  and hold 1.2 GB more.
- Small-batch (2-4 row) variants of the new layout: four lanes per block with
  trits streamed per byte group, 204-349 us against 155-244 us, and float2
  activation loads in the eight-lane kernel, 160 against 152 us.
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
- Folding the residual addition into the down and attention-output matvec
  stores: 21.58 tok/s fused versus 21.69 separate, plain decode.
- Fusing the GDN output normalization with the following rotation: 11.1 to
  5.4 us in isolation, but 21.58 versus 21.62 tok/s plain and 25.31 versus
  25.40 speculative with it; neutral.
- A concurrent encoder for single-token decode: 21.66 versus 21.95 tok/s once
  its independent projections were single dispatches. Before the BF16 change it
  measured 20.30 versus 19.33, by hiding the alpha/beta latency.
- SwiGLU with four rows of each matrix per SIMD group: 264.5 versus 262.1 us
  for two.
- Multi-request batching on the original kernels: 25.5/29.5/31.4 aggregate
  tok/s at 2/3/4 requests, versus 31.1/35.4/37.7 for serial requests
  retaining speculation. Superseded:
  batched serving with in-batch speculation now ships (see
  [Concurrent requests](#concurrent-requests)).
- Weight prewarm: increases resident footprint without improving steady-state inference.
- Q6 K/V (six-bit codes, F16 scale per 32, 26 KiB per token) as a fallback
  for machines where Q8 could not reach 32K tokens. Rotated: mean KL 1.3e-4
  (64/64 top-1), 15 times Q8's; unrotated 2.5e-4. Decode attention measured
  within 2% of Q8 at every length (1,809 against 1,844 µs at 128K) and prefill
  attention a few percent slower, so it bought 24% less memory and nothing
  else. Q8 alone is now budgeted.
- 4-bit K/V. Measured against F16 like the formats above (4,096-token prompt),
  unrotated: Q4 mean KL 3.8e-3 (61/64 top-1), FP4 E2M1 3.5e-3 (62/64), Q8 keys
  with Q4 values 1.6e-3 (62/64), Q8 keys with FP4 values 1.1e-3 (63/64).
  Hadamard rotation, which cut Q8's error by a third, barely moved them: Q4
  3.3e-3 (60/64), and FP4 got worse at 4.3e-3, since rotation removes the heavy
  tails its non-uniform levels are shaped for. Their error is the resolution of
  15 levels per 32 values, not outliers. That is about 0.3% in perplexity terms,
  the band the literature calls near-lossless, but 370 times rotated Q8's
  divergence for 47% less memory. Tensor decode attention was 7% faster than
  Q8's at 128K for Q4 and slower for FP4, whose level lookup costs more than
  the bytes it saves; whole-model decode at a 24,576-token prompt measured
  19.47 tok/s for Q4 against Q8's 19.34, 0.7%.
- Asymmetric K/V: Q8 keys with 4- or 3-bit values, the TurboQuant-style
  shortlist (keys carry the sensitivity, since each key error reaches six
  query heads). Rotated, against F16 at a 4,096-token prompt of repository
  text where rotated Q8 measured 1.2e-5 (64/64): Q4 keys with Q8 values 2.5e-3
  (61/64), Q8 keys with Q4 values 1.1e-3 (62/64). Values were then coded with
  a 4-bit Lloyd-Max codebook for a unit Gaussian on the Hadamard basis, scaled
  per 32 by a least-squares gain (two Lloyd steps on write): 9.4e-4 (62/64),
  only 17% under uniform Q4, and 3-bit 4.6e-3 (62/64); against Q8 the
  divergences are the same. The 4-bit codebook stays 80 times above the
  near-1e-5 bar (3-bit 380 times), every miss on a near-tie (top-2 margin
  under 0.2 nats): values are the cheaper half, not a cheap one. The 4-bit value cache is
  24% less K/V memory (3.5 against 4.6 GB at 128K tokens, 7.0 against 9.1 GB
  at 262K) but bought no speed: tensor decode attention at 128K measured
  6% faster than Q8 for uniform values and 36% slower for codebook ones, whose
  per-value level lookup costs more than the bytes it saves (3-bit: 66%
  slower).
- Software-pipelined Q8 tensor attention: each tile's V codes fetched into
  registers before its Q·K and the next tile's K codes before its P·V, so
  device loads overlap the tensor work. Decode attention got slower, 1,912 to
  2,100 µs per layer at 128K and 136 to 147 µs at 8K, presumably from the
  registers the in-flight codes hold; prefetching V alone was a wash (1,919
  against 1,906 µs).
- A ternary MTP head. Every head matrix rotated by the target's own signed
  Hadamard transform for its input width (which leaves w·x unchanged) and then
  replaced, per 128-value block, by the least-squares ternary vector and F16
  scale, packed as PTQ1 and run through the target's matvec: 4.5x smaller and
  drafting 30% cheaper (0.503 s to 0.352 s on the planets prompt), but draft
  acceptance fell from 79% to 68% (planets) and 95% to 80% (repetitive), so
  verification rounds rose 19-53% and throughput fell 9-10% (26.53 to 24.14 and
  39.43 to 35.29 tok/s, byte-identical text). The head was trained in BF16 and
  never calibrated for ternary; drafting is about 8% of decode time, which caps
  what any cheaper head can return. Quantization-aware training then closed
  two thirds of the gap but not all of it: `tools/mtp_train/train_ternary.py`
  (TWN g128 in the rotated basis, straight-through estimator, KL to the int8
  head on its own draft chains plus 0.1 of the target labels, 485K captured
  positions) for 10 epochs on a Kaggle T4x2 (7.6 GPU-hours) gave 27.46 tok/s
  at 74.3% acceptance against int8's 28.40 at 80.8%, six prompts, byte-identical
  text (untrained TWN: 25.15 at 67.5%; 4 epochs: 27.27 at 73.4%). The 93 MB
  head saves 332 MB resident, not decode time.
- An all-ternary head, superseded by the shipped mixed head. A per-matrix
  sensitivity study (each group ternarized alone in an otherwise int8 head,
  held-out KL to the int8 head per MB saved) ranked k/v, fc and o as costly and
  q, gate/up and down as cheap; heads keeping the costly groups int8 were then
  trained the same way on Kaggle (3 epochs from the ternary student). Six
  prompts, greedy, byte-identical text, with the GPU-resident draft loop: all
  ternary 93 MB 28.57 tok/s at 74.3% acceptance; fc+k+v+o int8 167 MB 29.48 at
  78.6% (shipped); plus `down` int8 237 MB 29.41 at 78.9%; all int8 425 MB 29.57
  at 80.4%.
- Keeping the int8 MTP head (retired). The community BF16 head quantized to
  symmetric per-row int8 (425,263,104-byte artifact, or 355,837,652 behind zstd
  level 19 at the cost of a 425 MB anonymous buffer) measured 28.40 tok/s at
  80.8% acceptance against the trained ternary head's 27.46 at 74.3%. It was
  removed with its BF16 loader, machine cache and zstd codec; int8 returned
  only as a per-matrix format inside the mixed head.
- Swapping in a head trained against the BF16 Qwen3.8-27B target. Same 15-tensor
  layout, int8 through the same export, greedy, six prompts (prose, explanation,
  arithmetic, code, essay, thinking), 300 tokens, best of two, byte-identical
  text in every arm. The community head (distilled from this ternary trunk's
  hidden states, run as int8 then) measured 28.18 tok/s geomean at 80.9% mean acceptance;
  `xkm/qwen3.8-27b-mtp-head-retrained` (multi-step, 17.5M positions, +2-5
  points over stock on BF16) 26.81 at 76.1%; the stock head
  (`EigenLabs/Qwen3.8-27B-MTP-bf16`) 26.68 at 74.5%. Alignment with the exact
  target outweighs a stronger general-purpose recipe. Block drafters (DFlash2,
  DSpark) were not ported: the published Qwen3.8-27B ones read 1.1-2 GB per
  draft pass, the Bonsai DFlash2 port reports 50% acceptance at three drafts,
  and the SpecForge-trained Qwen3.8 DFlash measures 1.81 accepted per step.
- Fine-tuning the community head on this trunk's own outputs (`mtp-capture` +
  `tools/mtp_train`, xkm's chain-faithful objective, depth 3): 485K captured
  positions (UltraChat, FineWeb-Edu, code, OpenWebMath), 820 steps. Held-out
  accuracy per draft step rose from 0.682/0.568/0.514 to 0.721/0.639/0.603,
  but live acceptance on the six-prompt A/B fell from 80.8% to 78.7% and
  throughput was flat (28.54 against 28.66 tok/s geomean, byte-identical
  text): teacher-forced corpus accuracy did not transfer to the model's own
  generations.
- Further mixed-head fine-tuning on self-generated rollouts: two probes keep
  `kv,fc,o` int8 and the remaining projections ternary, with target-loss weights
  0.1 and 0.5, trained for two epochs on 273 captured shards. On the recovered
  optimized runtime, six prompts, 300-token cap, reversed-order best of two:
  installed 38.88 tok/s at 79.0% pooled acceptance; probe A 38.67 at 77.6%;
  probe B 38.82 at 77.8%. A separate single-round coding comparison (two
  repairs and an 8,134-token repository prompt) measured 41.33 / 40.07 / 40.37
  tok/s geomean. Every candidate's output token sequence matched the installed
  head in both comparisons. Neither candidate replaces it. These are speed
  and agreement checks, not evidence of coding-agent quality: the generated
  Python repair passed its two supplied assertions, but the Rust repair kept
  a wrong tie-breaker and failed its supplied ordering test despite passing
  a superficial output-pattern check.
- Grouping independent single-row PTQ1 projection SIMD groups into larger
  threadgroups (2, 4 or 8 groups rather than 1). Arithmetic and outputs were
  bitwise unchanged on four representative shapes. Reversed-order three-round
  timings for 17408x5120 were 86.9 us baseline versus 87.9 / 87.5 / 88.7 us;
  5120x17408 stayed around 89 us. The output head's best four-group result
  was 1,121 us versus 1,123 us, only 0.2%; the tiny K/V projection saved
  about 0.3 us. No meaningful general gain, so this change was not adopted.
- Neural Engine: real-size projections execute on GPU and 54 GB FP16 weights do not fit.
- `float4` verify activation loads: 1–5% slower than scalar gathers.
- Further small-batch variants, M4 Pro, µs on 17408×5120 / 5120×17408. The
  simdgroup-matrix kernel for 2–4 rows: 340–348 at any row count, slower than
  the scalar 191–286. A 16-row matrix variant decoding trits once for two token
  tiles: 721–735, no better than two 8-row passes, because the F32 8×8
  multiplies alone cost about 270 per eight rows. One to four SIMD groups per
  threadgroup splitting the blocks: 379/407, 371/383, 367/369, the last kept;
  two or four independent accumulator chains: within 1%. Prefetching the next
  block's packed bytes in the scalar kernel: 197/262/321 against 197/256/325 at
  2/3/4 rows. Half trits with eight rows per SIMD group: 494/515 at four rows,
  and with two rows 319/351, both worse than four rows' 281/286. The half-trit
  scalar kernel at five and six rows: 337/351 and 394/409, against the matrix
  kernel's 340/345.
- Small-batch layouts after the half-prefix decoder, M4 Pro, µs on 17408×5120
  at 2/3/4 rows against 156/194/231 for the kernel they would replace. The
  matvec's layout (four lanes per block, float4/float2 inputs, balanced trits
  decoded per digit level) with 2/4/8 rows per SIMD group: 187–267 / 209–466 /
  342–696. One byte of five trits per phase for 4–16 rows with the scale folded
  into exact half factors (trit × d): 158–529 / 204–657 / 269–1036. Every input
  row in registers with rows decoded one at a time, shipped for two rows: 236 /
  360 at three and four. Each falls off a register cliff once a lane holds
  much more than about 60 live values, and nothing that stayed under it cut
  the per-row cost. The kernels issue close to one instruction per lane-cycle
  (the matrix kernel's 276 at eight rows is what one lane-cycle per 8×8×8
  multiply-add plus its decode predicts), so 2–4 rows pay the exact decode,
  about three half operations per weight, plus one F32 FMA and a share of an
  input load per weight and row; 1.3–1.6 times the one-row time is out of
  reach while trits stay exact. The telescoped decode the matvec uses (two
  operations per weight) is not an option: it loses about 73 on the lone-trit
  row that `matmul_keeps_f32_operand_bits_and_accumulates_beyond_f16_range`
  bounds at 4.
  Half-prefix qh trits from per-lane constants: 1–3% slower at four rows, and
  272 to 295 at five in the matrix kernel together with balanced prefixes.
  float2 inputs in the eight-lane kernel: 240 against 234 at four rows. Half
  simdgroup matrices: one 8×8 half multiply costs what an F32 one does (272
  against 272 at five rows, inputs rounded to half), and exact inputs split
  into high and low halves need two, 499.
- Wide-kernel layouts after the `float2` activation loads, M4 Pro, µs on
  17408×5120 / 5120×17408 against 237 / 281 at eight rows. Decoding the W
  tile once for two, three or four eight-row token tiles (bit-identical to
  separate passes): 16 rows 558 / 563 and 32 rows 1,178 / 1,225, against two
  and four passes' 474 / 562 and 948 / 1,124; the multiply-accumulates, not
  the decode, set the cost. Two or four eight-row tiles per threadgroup,
  with the four SIMD groups' block split kept (8 or 16 SIMD groups) or
  traded for the rows (one or two per tile), all bit-identical: 239-251 /
  266-281 at eight rows; the 266, one SIMD group per tile walking every
  block, costs 262 at five rows against 236. 5120×17408 is the one shape
  whose cost grows from five rows (236) to eight (281).
- Narrower prefill operands (half or BF16 activations, int8 activations
  against the trits), as a way past the large-batch kernel. They cannot be
  faster on this GPU. Chains of 8x8 simdgroup multiplies with no loads
  measured 3.85 T multiply-adds per second in F32, 3.95 T in F16, 3.85 T for BF16 or half
  into F32, and 3.93 T with F32 and F16 chains in alternate SIMD groups, so
  the half and F32 pipes do not run matrix work in parallel. Scalar FMAs
  were 3.48 T (F32), 3.66 T (F16) and 3.93 T mixed. Metal has no integer
  simdgroup matrix and the M4 Pro has no neural accelerators for
  `matmul2d` to use, so an int8 path would run on the same units. The
  large-batch kernel already reaches 3.4 T with exact F32 activations, so
  even a free lossy operand could gain at most about 15%, and narrower
  types make the multiplies no faster at all. Large-batch layouts, M4 Pro, µs on 17408x5120 / 5120x17408
  against 1,687 / 1,734 at 64 tokens and 3,317 / 3,333 at 128: a padded
  threadgroup stride of 72 halves, 1,724 / 1,765 and 3,374 / 3,397; one tile
  of 128 tokens with eight SIMD groups, 3,389-3,410 at any count up to 128;
  32 tokens by 64 rows per SIMD group (32 accumulators), 2,469 / 2,613 at
  64 tokens from register pressure; a 32-token
  tile of two SIMD groups for 32-55 rows, 950 / 1,027 at 32 against the
  wide kernel's 958 / 1,146 and 1,798 / 1,918 at 48 against 1,420 / 1,701.
  With the decode skipped after the first block (results invalid) 64 tokens
  took 1,634, without the barrier between decode and multiply 1,695, so
  double-buffering the decode is worth at most about 6%.
- BF16 alpha/beta with eight rows per SIMD group: 14-60 µs for both matrices
  at 2-64 rows, against four rows' 11.3-50.
- Converting the 96 BF16 alpha/beta matrices to int8 at load to drop the BF16
  kernels (absmax/127 scales; kernels bitwise across block sizes and fused into
  the decode concat). Teacher-forced against BF16 over 640 greedy steps after a
  2,048-token prompt: one scale per row 1.46e-5 mean KL, per 128 columns
  9.5e-6, per 32 columns 7.5e-6, each 638/640 top-1 and 64/64 over the last 64,
  KL falling rather than growing along the generation (per-128-step windows
  2.0e-5 to 9.1e-6 per row; 9.9e-6 to 4.4e-6 per 32). Greedy output matched on
  11 of the 12 runs (six prompts, plain and speculative) for every variant;
  all three changed the train prompt's plain decode at token 178, a 0.005-logit
  tie (18.241 against 18.236) that BF16's own speculative path resolves the
  same way the int8 variants do. Rejected on that byte-identity bar (per row
  also on the 1e-5 mean-KL bar), and the gain is small: plain decode
  30.6-30.9 tok/s for all four, the fused decode concat 91-100 µs against
  BF16's 98, the pair alone 8.6-10.3 µs against 10.9-11.7 at 1-8 rows but
  89-95 µs against 50 at 64, and 23.6 MB of codes (plus 0.8 or 3.1 MB of
  scales) in place of 47 MB.
- Depth 4 once five-row verify blocks cost 73.7 against four rows'
  72.2 ms: 36.4 against 36.2-36.5 tok/s on the planets prompt and 87.1
  against 89.5 on a copy-edit prompt.
- Tensor Q8 attention that round-trips the P·V result through threadgroup
  memory every tile: 3.6–7.3 times slower decode and twice as slow prefill in a
  first attempt; with 32- or 16-token dequantized tiles, 75 GB/s at 128K
  against F16's 200. Keeping the accumulator in registers is what made it pay.
- Trit lookup tables and 2-bit repacks: none exceed the FP32 digit decoder's 127 GB/s.
- Compressing the retired int8 head artifact above zstd level 19: level 22 measures
  358,063,503 bytes and `--long=27` measures 355,688,546, both worse than level
  19's 355,361,865 on the same payload. This is a property of the payload, not
  of zstd — the HTTP response path uses level 22 because it measured best on
  JSON bodies — so neither level is presented as universally best.
- A zstd dictionary on the HTTP response path: a 1 MB dictionary at level 9 was
  measured at +5.6% ratio and roughly 14x less time than the level-22 setting,
  on one buffered body, which is not what this path writes any more. A frame per
  piece measures 0.16x there, a net loss, so that comparison does not carry
  over. It was rejected anyway. RFC 8878 section 6 says `application/zstd`
  payloads should not use a dictionary, section 5 records that the specification
  does not define dictionary delivery, and section 7.4 confirms no public
  registry of such dictionaries exists. There is no HTTP mechanism to deliver the
  dictionary a frame's `Dictionary_ID` points at, so a stock client without it —
  `curl`, `httpx`, `requests`, `undici` — fails hard with `Dictionary mismatch`
  and decodes zero bytes. Verified. Dictionaries remain valid where both ends
  are yours, such as internal IPC or a bundled SDK.
- Compressing the 5,946,648,928-byte PTQ1 GGUF: the measured ceiling is 1.074x
  and the best measured result 1.29%, so the checkpoint is not a compression
  target. Only the retired int8 head was, which is why it shipped in two
  encodings.
- A rounder `--stall-timeout` floor: on a streaming response the number bounds
  only how long a client goes without taking a frame; on a non-streaming one it
  bounds that and the engine's own inter-event gaps, the producer half being the
  binding one. A speculative round measured 3.076 s, p95 0.728 s, on a six-token
  prompt on an M2, so a budget below that abandons healthy non-streaming
  requests. The floor is 10 rather than a rounder number because it sits an
  order of magnitude above the worst gap actually observed.
- Pipelining greedy plain decode (October 2026, prototype `408924e`,
  reverted in the next commit): a lone greedy plain decode committed the pass
  of token N+1 before waiting for token N. That pass gathered its embedding
  on the GPU from token N's argmax (`draft_embed_inverse` over a second
  `GreedyRows` selector, so the host read one while the pass in flight wrote
  the other) and wrote the recurrent state and convolution history out of
  place into a spare set (157 MB), which then swapped with the layers' own.
  The state before the pass in flight survived until the next pass was
  committed, so a pass that guessed wrong (an end of sequence, a client
  stop, an admission, a lookup round) rolled back by swapping again and
  stepping the position back, and the request's closing prompt checkpoint
  was exactly the one-pass-at-a-time one; every other use of the model first
  waited for and rolled back the pass in flight. Greedy text was
  byte-identical to the unpipelined build on six prompts at 300 tokens, and
  a real-model test matched tokens, the closing checkpoint and the next turn
  after end-of-sequence, token-limit and client-stop endings. GPU timestamps
  over 256 tokens (M4 Pro, five interleaved pairs in one process): idle
  between tokens 0.25–0.78 ms per token unpipelined (0.25–0.31 with the CPU
  quiet), 0.013–0.031 ms pipelined; wall time per token moved by 0.03–1.1 ms
  (best 33.17 to 32.84 ms) while GPU busy time moved 1.7 ms between rounds.
  `tools/benchmark.py` without speculation (an option since removed),
  300 tokens, the two builds
  interleaved, best of four (load average 4–8 of 14 from other agents' jobs):

  | Prompt | One pass at a time | Pipelined |
  | --- | ---: | ---: |
  | Planets, one per line (209 tokens, end of sequence) | 30.61 tok/s | 30.71 tok/s |
  | Essay on lighthouses | 29.99 | 30.13 |
  | Hash map collisions | 30.36 | 30.81 |
  | 1234 × 5678 step by step | 30.92 | 31.45 |
  | Rust linked-list reversal | 32.08 | 32.32 |
  | Causes of the French Revolution (293 tokens, end of sequence) | 32.02 | 32.32 |

  Every prompt gained, 0.3–1.7% (geomean 0.95%, about the idle time it
  removed), but each gain sits inside that run's own spread (1–8% between a
  build's best and worst sample). It applies only where no MTP head drafts
  and no lookup matches, which the default configuration never is, and it
  is not free: the spare state, a wasted pass (and about 33 ms longer
  request) at every end of sequence, a wasted pass at every stop, admission
  or lookup round that follows a pipelined one, and a settle before every
  other use of the model. Rejected.

## Verification

Run checks that do not require model files:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked --release
uvx ruff check tools/
python3 -m unittest discover -s tools -p 'test_*.py'
```

Ignored real-model tests resolve the pinned model and head from the
workspace's `models/` directory, so they need no environment variables. Run
them serially:

```bash
cargo test --workspace --release -- --ignored --test-threads 1
```

Metal's API validation (`MTL_DEBUG_LAYER=1`) passes every `local-metal`
test, ignored ones included, and a real-model generation (a 2,755-token
prompt, then 48 greedy tokens with MTP and suffix-lookup speculation) without
an error. Its shader validation (`MTL_SHADER_VALIDATION=1`) counts a kernel's
static threadgroup memory twice against the 32 KiB limit, so a kernel
declaring more than 16 KiB statically aborts at its first dispatch (`total
used threadgroupMemoryLength(33280) must be <= 32768` for 16,640 bytes);
dynamic threadgroup memory set with `setThreadgroupMemoryLength` counts once.
Nine kernels declared more: the Q8 tensor attention kernels (decode split,
prefill, two- to four-row splits and packed prefill rows, 17.1-19.3 KiB),
the SIMD prefill tile `bonsai_ptq1_matmul_bytewise_32` (16.25 KiB) and the
two vision attention kernels (19.9 and 22.6 KiB). Each now takes its largest
tile as dynamic threadgroup memory and is otherwise unchanged: outputs are
bitwise those of the static build in all twelve benchmarked cases (Q8 decode
at 1K-16K, two- to four-row blocks, 128- and 512-row prefill blocks, the
prefill tile, both vision kernels), greedy generation after the 2,755-token
prompt is token-identical, and GPU time is the same or lower in every case
(M4 Pro, median of nine trials, two interleaved rounds: Q8 decode at 16K
224-232 to 215-223 µs, the four-row split at 2K 89-90 to 84 µs, the 128-row
prefill block at 512 529 to 501-503 µs; the prefill tile and both vision
kernels within 1%).
Keep new kernels at or under 16 KiB of static threadgroup memory. With the
fix, every `local-metal` test passes under shader validation except
`fused_residual_and_gdn_output_chains_match_separate_dispatches_bitwise`:
the instrumented recompile changes the fused GDN-output kernel's
floating-point code generation, leaving it within 9e-8 (largest value 0.63)
of the separate dispatches instead of bitwise; it also fails on the build
before the fix, and uninstrumented builds stay bitwise.

The engine targets text inference on Apple Silicon. Model quality relative to
the unquantized checkpoint and broad application benchmarks are outside these
runtime measurements.

A real-model repository-retrieval probe on 2026-10-09 processed 69,097 input
tokens and generated six output tokens in 798.7 seconds on AC power without
sleep. With greedy sampling and reasoning disabled, it returned `56, 17`
instead of `38, 17`: the fixture placed the actual fee and factor near the
beginning, the helper in the middle, and 1,800 similarly named distractor
modules throughout. This is a failed quality check, not a passing long-context
coding benchmark. The admitted 262,144-token context and maximum-length kernel
tests do not establish reliable repository reasoning at that length. A short
prompt with the same formula and arguments returned the correct `38, 17`;
that control does not isolate the cause of the long-context failure.

Repeating the same 1,800-module fixture with the checkpoint's default `xhigh`
reasoning returned the correct `38,17`: 69,137 input tokens, 208 output tokens
(202 reasoning), 807.4 seconds on AC without sleep. Both runs were greedy;
the reasoning run allowed 1,024 output tokens instead of 32 and included the
template's reasoning instruction. It is a passing paired probe, not evidence
of reliable coding at 262,144 tokens or a causal isolation of every difference.

On 2026-10-10, a near-maximum-context probe **passed** on the 48 GB M4 Pro:
261,119 input tokens plus a 1,024-token output budget (262,143 tokens in
all, within the admitted 262,144). The synthetic repository contained 4,218
files, with production constants near token 91, the helper near token
129,531 and the question near token 261,030. Greedy generation with default
`xhigh` reasoning returned the expected `255,90` in 384 generated tokens,
closed its thinking section and stopped at EOS. The 3,163-token short
control also answered correctly, in 345 generated tokens. Native chat and
completion counts both matched the long fixture's exact 261,119 tokens.

This was staged prefill, not a cold single-request latency benchmark. Four
GPU checkpoints retained token-aligned prefixes across 14 bounded warmup
windows, with no host or disk snapshots. The final request reused 261,030
tokens, prefilled the remaining 89 in 3.67 seconds and decoded at 16.0 tok/s;
its 27.65-second duration excludes building that prefix. The complete run,
including the short control and 20-second cooldowns, took 6,516.5 seconds
(108.6 minutes); cumulative long-prefix prefill took 6,153.5 seconds.
No resource guard fired, the longest window took 588.4 seconds, sampled
free memory stayed at or above 59%, and sampled swap usage stayed at zero.
The local evidence is `context-max/run-20261010T135129/{progress.json,run.log,provenance.txt}`
under `.scratch/`, including prompt/token hashes and build provenance.
It is one passing long-context repository-retrieval/reasoning probe, not
evidence of general coding reliability at 262K or parity with the unquantized
model. Validation serializes GPU work and records the power source;
battery power is not a correctness-test blocker.

Real-model Chat and Responses each completed a three-turn read/edit/result/final
loop whose edit passed two executed assertions. Seven unmodified live response
objects and 57 Responses SSE events, including reasoning and function calls,
validated against OpenAI Python SDK 3.27.0. These checks cover the supported
text/function subset, not full platform compatibility or coding benchmarks.

A separate live audit validated 25 unmodified JSON objects and SSE events
against the pinned public OpenAPI schemas: Models, buffered and streaming
Chat/Completions/Responses, and rejected requests. Negative controls removing
required `created` and `param` fields failed validation as expected. This is
schema coverage of those sampled exchanges, not proof of full API conformance.

The opt-in lifecycle extension passed 208 real-model HTTP checks against that
same pinned schema: completed/incomplete persistence, `previous_response_id`,
input-item pagination, deletion, authentication, and exact input-token counts.
Encrypted replay matched its plaintext equivalent before and after a server
restart; altered envelopes and mismatched model/item IDs were rejected.
Continuation preserves the encrypted envelope in returned input items.
Stateless requests left no response records. Native chat and completion token
counts also matched real generation, including tool history and both reasoning
modes. These checks cover this text/function subset, not all OpenAI endpoints.

After background Responses were added, the same 208 checks passed again,
along with 180 additional real-model HTTP checks against the pinned schemas:
queued and terminal responses, output/usage, cancellation and idempotent
retry, deletion without resurrection, authentication, pending-history
rejection, persistence across restart, graceful shutdown, and recovery after
SIGKILL of the test server. These checks passed without skips. They did not
cover background streaming and resumption (checked separately below) or
temporary retention, which was then unsupported; cross-server lease conflicts
and write failures have CPU tests.

With structured output and streamed background responses added, the existing
208 HTTP checks passed again on the real model. Natively, 57 real-model checks
passed through the engine API, including schema-constrained raw-prompt
generations (an engine API since removed).
Four requests with different schemas decoded concurrently in batched rows
produced the same token IDs as each run alone, and constrained output matched
with speculation on and off. A streamed background response produced 68
events with contiguous sequence numbers, each valid against the pinned
schema; a resumed stream and a replay after server restart were
byte-identical to the original, a client disconnect did not cancel the job,
and cancel and delete stopped it.
The final structured-output/background-streaming HTTP run passed 212 checks,
and a focused schema-validation follow-up passed 33, with no skips. They cover
Chat nested `response_format`, Responses flat `text.format`, JSON-object mode,
strict true/false/absent, Unicode and reasoning, token-limit truncation,
concurrent schemas, and replay/disconnect/cancel/delete behavior. Disjoint
`oneOf` is accepted; overlapping branches and unsupported keywords return 400
JSON before streaming starts, name the correct API field, and store nothing.
Ending during reasoning before a structured answer is covered by CPU event
tests, not a forced live-model failure.
These checks cover the implemented JSON-constrained text subset and the
journaled background stream, not tool-call constraints, temporary retention
or full OpenAI conformance; cross-server 409 and journal write faults are
covered by CPU tests.

Native constrained tool calling (`tool_choice`, `parallel_tool_calls`, tool
`strict`, a format beside tools), temporary `store: false` background
retention and native decisions were added after those live
runs. CPU unit and HTTP-adapter tests cover grammar masks over synthetic
vocabularies, an ignored CPU-only test against the real tokenizer's call tags,
request parsing and echo, temporary-store expiry with an injected clock,
decision validation, queueing, cancellation and shutdown with a synthetic
head. Real-model runs with MTP enabled and disabled cover all three features:
forced and strict calls, mixed format/tool branches, temporary retrieval,
cancellation, deletion, SSE cursor replay and shutdown cleanup, plus native
decision feature parity and cache/FIFO/cancellation checks. The earlier 208
HTTP regression checks passed again. The ten-minute expiry is tested with an
injected clock, not a ten-minute live wait. No full OpenAI conformance or
general coding-quality claim follows from these contract checks.

### Coding clients and long-context coding quality

`tools/client_compat.py` runs Codex CLI, Oh My Pi and opencode against
`local-ai serve` through a recording proxy, in disposable scratch repositories
with scratch `HOME`/`CODEX_HOME`/XDG directories (Oh My Pi and opencode inside
a `sandbox-exec` write jail, Codex in its own `workspace-write` sandbox, no
network). Each scenario records pass/fail,
the first failing request/response, the server-log slice and the owning
workstream. `tools/context_eval.py` measures seeded coding questions inside
synthetic repositories of increasing size. Both are reruns, not fixtures:
run them again after a merge and compare.

```bash
# client scenarios (≈10 GPU-minutes per group); Codex installed under .scratch/codex
# (npm install @openai/codex), omp and opencode on PATH
CODEX=$PWD/.scratch/codex/node_modules/.bin/codex
lockf -k /tmp/local-ai-gpu.lock python3 tools/client_compat.py --codex-bin "$CODEX" \
    --timeout 420 --out .scratch/artifacts/client-compat/<run>
lockf -k /tmp/local-ai-gpu.lock python3 tools/client_compat.py --no-reasoning-key \
    --scenarios omp-fix-xhigh --out .scratch/artifacts/client-compat/<run>-nokey
# long-context matrix; finished items are skipped, so split it into bounded runs
lockf -k /tmp/local-ai-gpu.lock python3 tools/context_eval.py --sizes short,8k,32k
# a long group in pieces: stage a token-exact prefix (a raw `local-ai bonsai
# --raw` prefill, no server, whose disk snapshot the next hold restores), then
# ask; one HOME per group keeps the over-budget snapshot from being evicted
export HOME=$PWD/.scratch/home/250k-s0-xhigh
for n in 100000 165000 215000; do   # one lock hold each
  lockf -k /tmp/local-ai-gpu.lock python3 tools/context_eval.py --sizes 250k --seeds 0 --efforts xhigh --stage-tokens $n
done
lockf -k /tmp/local-ai-gpu.lock python3 tools/context_eval.py --sizes 250k --seeds 0 --efforts xhigh
python3 tools/context_eval.py --report
```

#### Client baseline (2026-10-11, server at `d2cad1b`)

Clients: Codex CLI 0.162.1 (npm `@openai/codex`, `wire_api = "responses"`,
function tools only: web search, sub-agents, goals, apps, plugins, skills and
the permissions/apps/collaboration developer text turned off) and Oh My Pi
18.8.9 (the engine README's `models.yml`, tools `read,bash,edit,write,grep,glob`,
`--thinking off|xhigh`). The server ran with `--reasoning-key` unless noted.

| Scenario | Result | Evidence |
| --- | --- | --- |
| omp-fix (run tests, edit, rerun, summarize) | PASS | 5 requests, 56 s, tests pass, test file untouched |
| omp-read + resumed second turn | PASS | `42`, then `7` after `--continue` |
| omp-recover (missing file, then fallback) | PASS | `PERIWINKLE-42` after the failed read's tool result |
| omp-fix-xhigh | **FAIL** | turn 2: 400 `encrypted reasoning requires its original unique item id`; without the key: 11× `response.failed` `invalid tool call: read: arguments: "i" is a required property` |
| omp-cancel (SIGINT mid-stream) | PASS | client exited, upstream closed, follow-up 200 in 0.3 s |
| codex-fix | PASS | 4 requests, 65 s |
| codex-read + `exec resume --last` | PASS | `42`, then `7` |
| codex-recover | PASS | `cat missing \|\| cat notes` in one call; the error reached the model |
| codex-fix-xhigh | PASS | 6 requests, 81 s, 7 tool results, encrypted reasoning replayed |
| codex-cancel | PASS | exited 0.1 s after SIGINT, follow-up 200 |
| codex-default-config | **FAIL** | 400 `tool type "namespace" is not supported` (`multi_agent_v1`; `web_search` also sent) |
| Codex with its default developer messages | **FAIL** | every request 400 `developer message must be first` |
| http-reasoning-summary-auto | **FAIL** | 400 `reasoning.summary is not supported` |
| http-custom-tool-apply-patch / local-shell / web-search / namespace | **FAIL** | 400 `tool type … is not supported` |
| http-compact | **FAIL** | 404 on `/v1/responses/compact` |
| http-multiple-developer-messages | **FAIL** | 400 `developer message must be first` |
| http-encrypted-reasoning-unrequested | **FAIL** | `encrypted_content` returned without `include` |
| http-encrypted-reasoning-replay, stream-disconnect, background-cancel, invalid-then-valid | PASS | replay answers `2.7.1`; follow-ups 200 |

Gaps by owner:

- **Protocol.** (1) Two leading `developer` messages (Codex sends base
  instructions, then permissions/skills text) are refused; `tools.rs` allows
  one leading instruction turn instead of merging consecutive ones.
  (2) With `--reasoning-key`, every reasoning item carries `encrypted_content`
  even when `include` does not ask for it; Oh My Pi (`includeEncryptedReasoning:
  false`) replays such items without their `id`, and replay requires the id.
  (3) A generated call to a non-strict function tool whose arguments miss a
  required property fails the whole response (`response.failed`,
  `server_error`) rather than returning the call for the client to reject or
  constraining it; Oh My Pi retried the identical request ten times.
- **Hosted/custom tools.** `custom` (freeform `apply_patch`), `local_shell`,
  `web_search` and `namespace` tools; Codex's default configuration needs
  `namespace` and `web_search`.
- **Reasoning summaries / compaction.** `reasoning.summary` and
  `/v1/responses/compact` are not implemented; the clients ran with summaries
  off and never reached their compaction thresholds in these short tasks.
- **Background recovery.** Cancellation of a running background response
  passes; recovery after a restart is not exercised by this harness.

Raw evidence: `.scratch/artifacts/client-compat/run5` (Oh My Pi, HTTP),
`run6` (Codex defaults for developer text), `run7` (Codex compat, new probes),
`run7-nokey`; each has `results.json`, `summary.md`, `exchanges.jsonl` and
`server.log`. The recovery scores above use the current rule (the failed read
must reach the model as a tool result); `run5`/`run7` were scored with an
earlier two-call rule.

**Re-run after merging the parallel work and the three protocol fixes**
(main at 6ac7dec, 2026-10-12; Codex CLI 0.162.1, Oh My Pi 18.9.1; one bounded
run of the previously failing and newly implemented scenarios, about 10
GPU-minutes). The `codex-*` configuration now keeps Codex's default developer
messages (only web search, sub-agents, goals, apps, plugins and the like stay
off), so those scenarios exercise the multi-developer-message path.

| Scenario | Result | Evidence |
| --- | --- | --- |
| omp-read + resumed second turn | PASS | `42`, then `7` |
| omp-fix-xhigh (with `--reasoning-key`) | PASS | 7 requests, 80 s, tests pass, test file untouched |
| omp-fix-xhigh (no key) | PASS | tests pass, 6 tool results |
| codex-fix (default developer messages) | PASS | 8 requests, 101 s |
| codex-read + `exec resume --last` (default developer messages) | PASS | `42`, then `7` |
| codex-fix-xhigh (default developer messages) | PASS | 6 requests, 87 s |
| codex-default-config | **FAIL** | 400 `tool type "namespace" is not supported` (sub-agents; still a hosted OpenAI tool) |
| http-multiple-developer-messages | PASS | 200 |
| http-encrypted-reasoning-unrequested | PASS | no `encrypted_content` without `include` |
| http-encrypted-reasoning-replay | PASS | replay answers `2.7.1` |
| http-reasoning-summary-auto | PASS | 200 |
| http-local-shell-tool | PASS | 200 |
| http-compact | PASS | 200 `response.compaction` |
| http-custom-tool-apply-patch | **FAIL** | 400 `tool name "apply_patch" is reserved`; the probe sends Codex's freeform `apply_patch` as a `custom` tool with a `lark` grammar, and `lark` grammars are refused as well ([docs/TOOLS.md](TOOLS.md)) |

Protocol gaps (1)–(3) above are fixed: consecutive leading instruction
messages are joined into the system turn, `encrypted_content` is emitted only
when `include` asks for it, and a non-strict function call that misses its
schema is returned as generated for the client to reject (covered by unit
tests; no such call occurred in this re-run). Codex's default configuration, its `namespace` and `web_search` tools and
its `lark` freeform `apply_patch` declaration are handled since (below). Evidence:
`.scratch/artifacts/client-compat/merged-main` and `merged-main-nokey`.

**Codex's default configuration** (branch `par-codextools` at c668412,
2026-10-12; Codex CLI 0.162.1; `tools/client_compat.py --codex-default-config`,
so every `codex-*` scenario sent Codex's default tools: `exec_command` and
friends, the `multi_agent_v1` sub-agent `namespace`, goals and `web_search`).
`namespace` tools are now offered as `NAMESPACE.NAME` functions and reported
with their `namespace`; `web_search` is accepted but never offered (no search
provider, so the model is never told it can search and no `web_search_call`
is reported); `custom` tools take Lark grammars with a regular language,
which covers Codex's freeform `apply_patch` grammar verbatim
([docs/TOOLS.md](TOOLS.md)).

| Scenario | Result | Evidence |
| --- | --- | --- |
| codex-fix (default config) | PASS | 5 requests, 113 s, tests pass, test file untouched |
| codex-read + `exec resume --last` (default config) | PASS | `42`, then `7` |
| codex-fix-xhigh (default config) | PASS | 8 requests, 146 s, 8 tool results |
| codex-default-config | PASS | `42`, then `7` |
| http-custom-tool-apply-patch | PASS | Codex's grammar, call forced: `*** Begin Patch\n*** Add File: "hello.txt"\n+hi\n*** End Patch` (the grammar admits the quoted file name; it constrains form, not judgment) |
| http-namespace-tool / http-web-search-tool | PASS | 200 |

The model never chose a sub-agent tool in these tasks, so a namespaced call
has been exercised by unit tests only. Codex declares its freeform
`apply_patch` only for catalog models marked freeform, so the Codex runs above
edited files through `exec_command`; the probe sends Codex's declaration
itself. Evidence: `.scratch/artifacts/codextools/client-compat/codextools`.

**Codex, opencode and Oh My Pi end to end** (branch `harness-compat`,
2026-10-19; Codex CLI 0.163.0, opencode 1.19.3, Oh My Pi 18.12.2, each with
the configuration in [Coding client configuration](#coding-client-configuration);
server with `--reasoning-key`). opencode runs both of its providers:
`opencode-*` through `@ai-sdk/openai-compatible` (Chat Completions),
`opencode-responses-*` through `@ai-sdk/openai` (Responses, `store: false`).
`codex-apply-patch` uses a catalog copied from `codex debug models`, so Codex
declares its freeform `apply_patch`; `codex-compact` sets
`model_auto_compact_token_limit = 7000`, so Codex compacts mid-task.

| Scenario | Result | Evidence |
| --- | --- | --- |
| opencode-fix / opencode-fix-xhigh (Chat) | PASS | 6 requests, 68 s / 95 s; tests pass, test file untouched; xhigh replays `reasoning_content` |
| opencode-read + `--continue` second turn (Chat) | PASS | `42`, then `7` |
| opencode-recover (Chat) | PASS | `PERIWINKLE-42` after the failed read's tool result |
| opencode-responses-fix / -read | PASS | 5 requests, 65 s; `42`, then `7` |
| opencode-responses-fix-xhigh | PASS | 5 requests, 90 s; 3 encrypted reasoning items replayed without `id` |
| opencode-cancel | PASS | SIGTERM (its `run` ignores SIGINT while the title request streams); upstream closed, follow-up 200 in 0.4 s |
| codex-fix | PASS | 6 requests, 70 s; one `apply_patch` call to an undeclared tool, answered `unsupported call: apply_patch`, then `sed` |
| codex-read + `exec resume --last`, codex-recover, codex-default-config | PASS | `42`/`7`; `PERIWINKLE-42` |
| codex-fix-xhigh | PASS | 6 requests, 102 s |
| codex-apply-patch | PASS | 5 requests, 79 s; 2 freeform `apply_patch` results |
| codex-compact | PASS | 14 requests, 236 s; 3 local compactions, tests pass |
| codex-cancel, omp-cancel | PASS | exited 0.0 s after SIGINT, follow-up 200 |
| omp-fix / omp-read / omp-recover / omp-fix-xhigh | PASS | 5 requests, 53 s; `42`/`7`; `PERIWINKLE-42`; 8 requests, 89 s |
| http-legacy-members | PASS | `/v1/completions` 404; Chat with `max_tokens`, `seed`, `user` 200; legacy `functions` answered as `function_call`; Responses `user` 200 |
| every other `http-*` probe | PASS | as above |

Gaps these runs found, all fixed on the branch:

- **Deprecated request members.** opencode's Chat provider sends
  `max_tokens` on every request, which was a 400, so no opencode Chat
  request succeeded; Aider, LiteLLM and Continue send `max_tokens`, `seed`
  and `user` too. They are now honoured or ignored (see
  [Supported OpenAI APIs](#supported-openai-apis)).
- **Id-less encrypted reasoning.** The AI SDK replays reasoning items
  without their `id` under `store: false`, so opencode's Responses provider
  failed every xhigh turn after the first with 400 `encrypted reasoning
  requires its original unique item id`. Envelopes now carry the id they
  are bound to.
- **Calls to undeclared tools.** Codex's fallback instructions name
  `apply_patch`, which its default tool set lacks; the model's call failed
  the Response (`invalid tool call: unknown tool "apply_patch"`) and Codex
  retried. Such a call is now returned to the client, which answers it.
- **Harness.** The cancel scenario waits for a streamed agent turn (opencode's
  title request streams first) and stops opencode with SIGTERM; the compaction
  limit sits between Codex's first request (about 6,000 tokens) and the end
  of the task (about 7,700).

Not implemented: Codex's `remote_compaction = "v2"` (`compaction_trigger`
input item). Chat `stop` sequences and `verbosity` `low`/`high` were added
afterwards (see [Generation requests](#generation-requests)).
Evidence: `.scratch/artifacts/client-compat/final-a` (Codex, opencode
Responses; its `codex-fix`, `codex-compact` and `opencode-cancel` predate the
fixes above), `final-b` (Oh My Pi, Codex reruns, HTTP probes), `final-c`
(opencode cancel, invalid-then-valid), `final-d` (opencode Chat), `final-e`
(one read scenario per client and two probes, rerun after rebasing onto
image input).

**Codex remote compaction** (branch `codex-compaction`, 2026-10-19; Codex
CLI 0.163.0; server with `--reasoning-key`). Codex 0.163 sends remote
compaction only as `compaction_trigger` turns (its provider capability has no
`"v1"`), so that gap above is closed by implementing the trigger; Codex never
calls `/v1/responses/compact`, which `http-compact` still covers.
`codex-remote-compact` is `codex-compact` with
`remote_compaction = "v2"` (automatic compaction at the 7,000-token limit);
`codex-remote-compact-manual` drives `codex app-server`: a read turn,
`thread/compact/start` (the TUI's `/compact`, which `codex exec` cannot
send), then a follow-up that needs the compacted context.

| Scenario | Result | Evidence |
| --- | --- | --- |
| codex-remote-compact | PASS | 12 requests, 186 s; three `compaction_trigger` turns (7,342 to 7,679 prompt tokens, 6,715 to 7,101 of them from the prompt cache, 475 to 561-token summaries, 15 to 17 s each), each answered with one `compaction` item; the second and third fold the earlier item in; the 7 other requests after the first replay the latest item, and the next prompt after each falls to about 6,050 tokens, near Codex's first request of 6,007; tests pass |
| codex-remote-compact-manual | PASS | 5 requests, 23 s; `42`; the manual compaction turn (6,264 prompt tokens, 90-token summary, 4 s); then "that same file" resolved to `settings/limits.py`, which the prompt names only through the compaction item, re-read, `7`; both later requests replay the item |
| codex-compact (local compaction, unchanged) | PASS | 26 requests, 509 s; local summarization turns, tests pass |
| http-compact | PASS | 200 `response.compaction` |

Evidence: `.scratch/artifacts/client-compat/remote-b` (after rebasing; an
earlier run, `remote-a`, passed the same four with one automatic compaction).

**Current-spec request surface** (branch `request-surface`, 2026-10-19;
Codex CLI 0.163.0, opencode 1.19.3, Oh My Pi 18.12.2; server with
`--reasoning-key`), after Chat `stop` and `verbosity`, ignored unknown
Responses members, and the removal of the deprecated-member adapters
(`functions`/`function_call`, `user`, `reasoning.generate_summary`,
`session_id`, `prompt_cache_retention`) and of `http-legacy-members`. Every
Codex scenario now runs with `remote_compaction = "v2"`, and its requests
carry `x-codex-beta-features: remote_compaction_v2`. The recorded requests
use only current members, plus `max_tokens` (opencode Chat) and
`client_metadata` (Codex).

| Scenario | Result | Evidence |
| --- | --- | --- |
| omp-read | PASS | 3 requests, 46 s; `42`, then `7` |
| opencode-read (Chat) | PASS | 4 requests, 66 s; `42`, then `7` |
| opencode-responses-read | PASS | 4 requests, 62 s; `42`, then `7` |
| codex-read (v2) | PASS | 4 requests, 55 s; `42`, then `7` |
| codex-remote-compact | PASS | 23 requests, 368 s; seven `compaction_trigger` turns, each answered with one `compaction` item, 13 later requests replaying one, no local compaction; tests pass |
| codex-remote-compact-manual | PASS | 5 requests, 23 s; `42`, the manual compaction turn, `7` |

Evidence: `.scratch/artifacts/client-compat/request-surface` and
`request-surface-codex`, and `request-surface-rebased` (the four read
scenarios again, all passing, after rebasing onto video input); live `stop`, `verbosity` and unknown-member probes
in `.scratch/artifacts/request-surface/live.json`.

#### Tool-call reliability (2026-10-20, branch `opt-tool-calls`, base `b09e22f`)

Work in progress; tooling and raw data are in `tools/tool_calls/` (scratch
scripts, kept so the work can resume: `suite.py` runs real Codex 0.163.0 and
opencode 1.19.3 Chat/Responses sessions through `client_compat`'s proxy,
`audit.py` classifies every call, `replay.py` resends recorded requests under
several seeds and sampling settings, `classify.py`/`error_points.py` tally
known-bad calls; `data/` holds the second baseline's exchanges and the replay
results). The agent benchmark (`opt-agent-bench`, `tools/agent_bench.py`)
supersedes `suite.py` for whole-session A/B runs once it lands.

Method: four small tasks (fix a bug, rename across files, change a config
value, implement from a docstring) × Codex, opencode Chat, opencode
Responses, at reasoning effort `none`; then *decision-point replays*: a
recorded request resent with seeds 0–5 (or 0–7) per setting, so a failure
seen once in a session becomes a rate. Seeds pair the settings (common random
numbers). These are correctness counts, not timings; some replays may have
overlapped another GPU job (the shared lock was unreliable before about
12:30), which cannot change sampled outputs.

What the requests are: Codex sends `tool_choice: auto`,
`parallel_tool_calls: true` and non-strict function tools, opencode sends
`auto` and non-strict tools, and none sends sampling parameters. So real
client traffic is **not** grammar constrained (`needs_grammar` is false) and
runs at the server defaults (temperature 1.0, top-p 0.95, top-k 20, fixed
default seed), also at effort `none`.

Findings:

- **Syntax is not the problem.** Zero leaked call markup, zero `invalid tool
  call` failures, zero non-JSON or schema-violating arguments in about 320
  session calls (two baselines) and 487 replayed calls.
- **Bad arguments are semantic, mostly Codex assuming a GNU/Linux shell** on
  macOS: `cat -A`, `sed -i 's/…/'` without `''`, `python` (only `python3`
  exists), `rg --limit`, and once a hallucinated `sed` that inserted unrelated
  code. 2 of 4 Codex sessions in each baseline had at least one. Replayed at
  the turn where it happened, the GNU `sed -i` recurs in 1/8 seeds at the
  default settings, 1/8 at T 0.7/top-p 0.8, 2/8 at T 0.6, and 1/8 with a
  "this machine runs macOS (BSD userland, `sed -i ''`, `python3`)" note
  appended to the instructions: rare per turn, frequent per session, and
  neither sampling nor that note removed it. Codex's `environment_context`
  says the shell is zsh but not the OS; opencode's prompt says darwin and
  showed none of these.
- **Wrong tool**: Codex called the undeclared `apply_patch` once (8 Codex
  sessions), in the `{"command": [...]}` form its fallback instructions
  describe; it got `unsupported call` and recovered. The `model_catalog_json`
  recipe in [Coding client configuration](#coding-client-configuration)
  declares the tool.
- **Invented parameters**: opencode `grep` with `i`/`glob` (Claude Code's
  names) in both opencode config sessions, because both ran the same default
  seed; replayed, 1/10 calls at default sampling, 0 at T 0.7.
- **Loops**: in the first baseline (raw data lost to a directory deletion),
  2 of 4 Codex sessions finished the task and then kept "verifying" for 100+
  turns until the 600 s timeout; the second baseline had none, opencode none.
  Lower temperature makes this worse, not better: at turns right after the
  tests passed, Codex stopped in 3/6 seeds at T 1.0, 1/6 at T 0.7/top-p 0.8
  and 0/6 greedy (rename task); 2/6, 1/6, 0/6 (config task). So the model
  card's instruct settings (T 0.7, top-p 0.8) were **not** adopted for
  effort `none`.
- **Prompt rendering**: the renderer sorted keys in tool schemas and replayed
  arguments, while the GGUF's template rendered with `transformers`' `tojson`
  (and vLLM, llama.cpp) keeps the client's order; fixed on this branch
  (fixtures re-verified against the template). Replays: 48 of 138 outputs
  changed, no category changed; neutral so far.
- **Harness**: `client_compat.run_client` left Codex's native agent running
  after a timeout, and it kept sending requests during later scenarios
  (also affects `agent_bench.py`); fixed on this branch.
- Not a client issue today, but noted: the sampler counts prompt tokens for
  `presence_penalty` (vLLM and OpenAI count only output tokens), so the model
  card's instruct `presence_penalty: 1.5` would penalize every token of a
  coding prompt.

Reasoning effort (2026-10-23, `tools/agent_bench.py` on `a4cf247`, seed 1,
one run per effort, each alone under the GPU lock; `none` is the two
[baseline](AGENT_BENCH.md#baseline) runs). Codex columns; "tail" is the turns
after the last edit, summed over tasks:

| Effort | Codex success | Turns, mean | wrong_tool | Loops | Tail | `sed -i` without `''` | `cat -A` | Wall s | opencode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| none (two runs) | 6/7, 7/7 | 18.9, 14.9 | 4, 3 | 2, 1 | 72, 40 | 3, 5 | 2, 1 | 1,197, 862 | 7/7, 7/7 |
| medium | 6/7 | 10.3 | 2 | 0 | 22 | 4 | 2 | 1,641 | 7/7 |
| xhigh | 5/7 | 8.6 | 0 | 0 | 16 | 3 | 2 | 1,481 | 7/7 |
| xhigh, here-document fix | 7/7 | 9.6 | 2 | 0 | 13 | 5 | 0 | 769 | - |

- Reasoning removes the verification loops: no loop at `medium` or `xhigh`,
  and the turns after the last edit fall from 40–72 to 13–22. It does not
  cost wall time once the harness lets here-documents work: `xhigh` took
  769 s for 7/7, fewer turns making up for the reasoning, against 862–1,197 s
  at `none` (1,481 s before the fix, see below).
- It does not remove the GNU-isms: `sed -i` without `''` appears as often
  at every effort (3–5 per run), `cat -A` 0–2 times. Each costs a failed
  command and a retry, and no task failed on one. Nothing server-side can
  fix them (the server does not know the client's OS, and round 1's macOS
  note in the instructions did not reduce them at `none`); on macOS the
  remedy is client-side, for example GNU `sed` and `coreutils` first on the
  `PATH` Codex inherits.
- The three Codex failures at `medium` and `xhigh` were one chain, not
  reasoning quality: the harness's Codex sandbox excluded `TMPDIR` and
  `/tmp` from its writable roots, so `cat > file << 'EOF'` failed with "can't
  create temp file for here document" (default Codex allows both), the model
  then reasoned about quoting until the server's default 8,192-token output
  budget ran out (Codex sends no `max_output_tokens`), and Codex ended the
  turn on the incomplete response. The same here-document error appeared at
  `none` too (in 2 and 4 log lines), where the model, not reasoning, simply
  tried another way. `client_compat` now keeps the run's own scratch
  `TMPDIR` writable and points zsh's `TMPPREFIX` into it. `medium`'s
  `rename-function` was rerun alone (with that fix) because Homebrew replaced
  the running Python mid-run (`No module named 'glob'` in the harness).
- opencode solved every task at every effort; its one bare `python` (per
  `none` run) did not recur at `medium` or `xhigh`.

So the documented Codex setting, `model_reasoning_effort = "xhigh"` (or
`medium`), is the configuration fix for the loops, at no cost in wall time
on these tasks; one run per effort, so read single-task differences as
noise.

#### Long-context coding quality

Each seed plants nine questions in a synthetic Python repository: at the
start, middle and end (4%, 50%, 94% of the prompt) a `recall` (state a
production fee constant), a `lookup` (evaluate a pricing function whose
constants live in a config module, with similarly named legacy decoys
elsewhere) and a `fix` (return the one corrected line of a function that
contradicts its docstring; the line is substituted and run against hidden
tests in an isolated interpreter). The `short` control holds the same nine
needles in about 1.7K tokens. The repository is the system message, each
question its own user turn, greedy, non-streamed Chat Completions; the
first question of each (size, seed, effort) group pays the cold prefill and
the rest reuse it from the prompt cache (three in parallel). `none` allows 384
output tokens, `xhigh` 4,096; no answer hit its limit. Groups at 128K and
above were prefilled in pieces, one GPU-lock hold each: a raw, template-free
prefill of a token-exact prefix of the rendered prompt (72K tokens at 128K;
then through the server, now `local-ai bonsai --raw`) whose disk snapshot the
next hold restored, so the cold-prefill column adds the
stages and the first question's own prefill. Intervals are 95% Wilson
intervals over questions; the nine questions of a seed are not independent,
so they are optimistic.

Third run 2026-10-13/14, server built at `0c9ec76`, M4 Pro 48 GB (decode
is the warm median per answer; ~250K and some 128K follow-ups ran one at a
time, the rest three at a time):

| Size | Effort | Seeds | Prompt tokens | Recall | Lookup | Fix | Start | Middle | End | All (95% CI) | Cold prefill | Warm decode | Warm wall, median |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 64K | none | 0,1,2 | 64,296–64,415 | 9/9 | 7/9 | 6/9 | 7/9 | 9/9 | 6/9 | 22/27 (63–92%) | 87–93 tok/s (691–736 s) | 3.9 tok/s | 7.2 s |
| 64K | xhigh | 0,1,2 | 64,332–64,451 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | **27/27** (88–100%) | 92–93 tok/s (693–702 s) | 9.6 tok/s | 22.9 s |
| 128K | none | 0 | 127,931–127,943 | 3/3 | 0/3 | 2/3 | 1/3 | 2/3 | 2/3 | 5/9 (27–81%) | 71 tok/s (1,809 s) | 2.1 tok/s | 24.5 s |
| 128K | xhigh | 0 | 127,967–127,979 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | **9/9** (70–100%) | 71 tok/s (1,790 s) | 18.4 tok/s | 16.3 s |
| ~250K | xhigh | 1 | 249,944–249,956 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | **9/9** (70–100%) | 48 tok/s (5,242 s) | 10.9 tok/s | 31.4 s |
| ~250K | none | 0,1 | 249,399–249,920 | 6/6 | 0/6 | 4/6 | 2/6 | 4/6 | 4/6 | 10/18 (34–75%) | 44–50 tok/s (5,040–5,699 s) | 2.2 tok/s | 11.3 s |

~250K `none` (2026-10-16, same build; seed 0 resumed from its 215K stage,
whose last 34K took 1,108 s; seed 1 staged at 100K/165K/215K in 1,251, 1,369
and 1,350 s, then 1,070 s for the rest) scored 5/9 on each seed, the same
pattern as 128K seed 0: recall 6/6 (61–100%), lookup 0/6 (0–39%; answers
101–137 for 182–552), fix 4/6 (30–90%; both misses are the `sum_even`
needle, at the start in both seeds). By position start 2/6 (10–70%), middle
4/6 and end 4/6 (30–90% each). Every follow-up was asked alone after
restoring a clone of the system-turn snapshot (10–17 s per answer). Seed 0's
lookup-start was asked again and repeated its earlier 133.

The scores repeat the earlier runs exactly where they overlap: 128K seed 0
`none` 5/9 with every lookup wrong, `xhigh` 9/9. Pooled with the earlier
seeds, `xhigh` is 27/27 at 64K, 27/27 at 128K (88–100%) and 18/18 at ~250K
(82–100%, seeds 0 and 1). `none` fails the same `sum_even` fix needle at
every size (the changed line is not the buggy one, and its indentation no
longer parses) and most lookups, while recall stays 100%. Single-stream
`xhigh` decode was 17–22 tok/s at 128K and 9–13 tok/s at ~250K after a
12 s disk restore of the 9.8 GB system-turn snapshot; cold prefill was 92
tok/s to 64K, 88 tok/s to 72K and 48 tok/s averaged to ~250K (stages to 100K,
165K, 215K: 1,259, 1,409 and 1,425 s, then 1,148 s for the last 34K).

**Prompt-cache trimming at ≥128K.** The automatic disk budget is 5.9 GB
(bounded by the 5 GiB model), and a 128K snapshot is 5.06 GB, so one snapshot
fills it; each newer one evicts the older, always keeping one. Staging
relies on that: each ~250K stage restored the previous one and its commit
deleted it, except that the 100K stage of seed 0 `none` survived beside the
165K one (10.5 GB on disk) until the 215K commit trimmed both, so a
restarted server can briefly hold more than the budget. With three
follow-ups in parallel, a later question that shared a deeper prefix with an
earlier one of its kind (recall vs recall at `none`, 127,895 tokens; fix vs
fix at `xhigh`, 127,926) wrote a divergence boundary that evicted the
system-turn boundary (127,882 / 127,920) and the staged prefix, so every
question of another kind fell back to a full cold prefill (a stack sample
showed the engine in `prefill_segment` at 99% GPU six minutes into the last
two `none` requests). Those holds were stopped, the snapshot deleted, the 72K
stage rebuilt, and the remaining questions asked one per server start after
restoring an APFS clone of the system-turn snapshot (a scratch wrapper
around `tools/context_eval.py`; prompts and scoring unchanged). Each answer
then took 11–50 s plus a few seconds of server start. Under the previous
build every follow-up of five 128K/~250K groups reused the system-turn
boundary or one 4 tokens past it and no deeper one appeared, so this looks like a
regression in long-context multi-question reuse (not bisected). A SIGINT
stop does not always finish a queued write either: signalled right after a
3-token ~250K `none` answer, the server exited within about a second, leaving
its 249,350-token system-turn snapshot as a 4.66 GB `.tmp` (of ~9.8 GB), so
the next question could only restore the 215K stage. The `xhigh` answers
presumably took long enough for the write to commit first.

Both are fixed after `0c9ec76`. The first was not a code regression: the
prompt-cache code is identical at `d2cad1b`, `6ac7dec` and `0c9ec76`. A
follow-up's divergence boundary is its common prefix with whichever prompt
the engine cached last, so a question that follows one of its own kind
yields a boundary a few tokens past the system turn, and both the disk trim
and the four GPU checkpoints evicted the oldest boundary first; whether
that happened depended on arrival order. A boundary that only barely
extends a kept one is now the one dropped (see [Prompt cache](#prompt-cache)).
The second: the server's engine lives on a worker thread that exit never
joined, so SIGINT killed the background write; shutdown (SIGINT or SIGTERM)
now closes the writer with a two-minute deadline, and startup removes
orphaned temporaries and trims to the budget. Rerun of 128K seed 0 `none`
(72K stage, then nine questions, eight three at a time, server at `4004c39`):
every follow-up restored the 127,882-token system-turn snapshot (seven from
disk, one from a GPU checkpoint 4 tokens past it), 4–11 s to first token and
8.5–27 s per answer, scoring the same 5/9; afterwards the disk held only that
snapshot, and the SIGINT stop left no `.tmp`.

Rerun 2026-10-12/13, server built at `6ac7dec` (`9228478` adds only docs), M4 Pro 48 GB:

| Size | Effort | Seeds | Prompt tokens | Recall | Lookup | Fix | Start | Middle | End | All (95% CI) | Cold prefill | Warm wall, median |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| short | none | 0,1,2 | 1,593–1,808 | 9/9 | 3/9 | 8/9 | 7/9 | 7/9 | 6/9 | 20/27 (55–87%) | 126–130 tok/s | 1.6 s |
| short | xhigh | 0,1,2 | 1,629–1,844 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | **27/27** (88–100%) | 129 tok/s | 7.9 s |
| 128K | none | 1,2 | 127,510–127,668 | 6/6 | 2/6 | 4/6 | 3/6 | 5/6 | 4/6 | 12/18 (44–84%) | 67–69 tok/s (1,861–1,894 s) | 21.2 s |
| 128K | xhigh | 1,2 | 127,546–127,704 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | **18/18** (82–100%) | 68–69 tok/s (1,862–1,872 s) | 41.5 s |
| ~250K | xhigh | 0 | 249,435–249,447 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | **9/9** (70–100%) | 47 tok/s (5,280 s) | 94.5 s |

Earlier run (2026-10-11, server at `d2cad1b`; raw rows lost, not rerun):

| Size | Effort | Seeds | Prompt tokens | Recall | Lookup | Fix | Start | Middle | End | All | Cold prefill | Warm wall, median |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 8K | none | 0,1,2 | 8,636–8,758 | 9/9 | 4/9 | 6/9 | 4/9 | 9/9 | 6/9 | 19/27 | 111–124 tok/s (≈76 s) | 2.7 s |
| 8K | xhigh | 0,1,2 | 8,672–8,794 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | **27/27** | 104–125 tok/s (≈80 s) | 12.4 s |
| 32K | none | 0,1,2 | 32,380–32,759 | 9/9 | 7/9 | 6/9 | 7/9 | 9/9 | 6/9 | 22/27 | 106–107 tok/s (≈305 s) | 3.7 s |
| 32K | xhigh | 0,1,2 | 32,416–32,795 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | **27/27** | 107 tok/s (≈305 s) | 16.2 s |
| 128K | none | 0 | 127,931–127,943 | 3/3 | 0/3 | 2/3 | 1/3 | 2/3 | 2/3 | 5/9 | 67 tok/s (1,911 s) | 12.1 s |
| 128K | xhigh | 0 | 127,967–127,979 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | **9/9** | 66 tok/s (1,923 s) | 36.7 s |

The short controls matched the earlier run's scores (20/27 and 27/27). With the earlier seed 0, 128K now has three seeds per
effort: `xhigh` 27/27 (88–100%), `none` 17/27 (44–78%). `xhigh` answered
every question at every size measured, with 83–408 reasoning tokens and no
truncation; its 128K follow-ups decoded at 18–23 tok/s three at a time, with
a 4–7 s time to first token. `none` retrieved every constant (recall 100% at
every size) but missed lookups, which need a two-step calculation (2/6 at
128K, 3/9 in the short control), and failed the same fix needle wherever it
appeared: for `sum_even` (`range(1, n)` should be `range(1, n + 1)`) it
returns `total += 2 * (i + 1)`, a change to the wrong line that is also
incorrect. Its failures appear in the short control too, so they are a
reasoning-mode effect, not a context-length one; no position failed under
`xhigh`. Latency is dominated by cold prefill: about 127 tok/s at 4K, 87 tok/s
over the first 72K and 54 tok/s from 72K to 128K, so a fresh 128K repository
costs 31 minutes before its first token. With the prefix cached, a 128K
follow-up took 15–83 s (median 21 s) for a two-token `none` answer and 9–49 s
for an `xhigh` one.

At ~250K (seed 0, `xhigh`; 249,445 prompt tokens plus 4,096 of output room
within the 262,144 limit) every question passed, with 98–408 reasoning
tokens. The cold prefill took four holds: to 100K in 1,310 s (76 tok/s), to
165K in 1,429 s (46), to 215K in 1,419 s (35) and the last 34K in 1,122 s
(30), each including the restore of the previous stage from disk: 88 minutes
in all. Follow-ups at 128K and ~250K were served from the disk tier, not a
GPU checkpoint (every warm row reports `cache_source: disk`): restoring the
9.8 GB system-turn snapshot put the ~250K time to first token at 10–15 s,
and answers then decoded at 12–14 tok/s three at a time, 26–110 s per
answer. One ~250K seed is evidence that the length works, not a rate.

Not measured: 8K/32K on a current server, ~250K seed 2, and 128K seeds 1–2
on `0c9ec76`. The raw rows of the second and third runs were deleted; those
of the ~250K `none` run (answers, token counts, timings, cache hits, seed 1's
stages, prompts, server log and `holds.log` with the snapshot files after
each item) are in `.scratch/artifacts/context-eval/`.

Why lookups fail at effort `none` (2026-10-23): the model, not the engine.
A lookup asks for `min(units * FACTOR + FEE * weight, CAP)` with the three
constants in a config module and the formula in a service file, as one
integer. A decomposition probe (seed 0, effort `none`, greedy, Q8 K/V, the
eval's own repository text at about 1.7K, 64K and 128K tokens, each lookup
split into five questions) gives the same result at every length:

| Question (three per length) | ~1.7K | 64K | 128K |
| --- | --- | --- | --- |
| the original lookup, integer only | 0/3 | 0/3 | 0/3 |
| the three constants | 3/3 | 3/3 | 3/3 |
| quote the function body | 3/3 | 3/3 | 3/3 |
| the lookup, writing constants and arithmetic first | 3/3 | 3/3 | 3/3 |
| the recall needle | 3/3 | 3/3 | 3/3 |

Retrieval is exact at 128K (every constant, every formula), and the model
computes the answer correctly whenever it may write the steps. Asked for the
bare integer, it answers without them and is wrong at every length,
including the 1.7K control: at 128K 130, 137 and 122 for 182, 239 and 438
(130 is the first term, 26 × 5, alone), at 1.7K 130 and 100 (answers were
capped at 8 tokens, so the third, and two at 64K that began explaining,
were cut off before an integer). The same
arithmetic with no repository at all (`Compute min(26 * 5 + 4 * 13, 676).
Reply with only the integer.`) starts writing the multiplication instead of
an integer. This matches the eval: `none` scored 3/9 lookups in the short
control too, and `xhigh`, which reasons first, answered every lookup at
every size measured, up to ~250K. So the remedy is reasoning (`xhigh`
measured) or letting the answer show its work; nothing in the engine
changes it. An F16 K/V run of
the 128K probe was stopped as redundant: the failure does not depend on
length, so it cannot come from K/V quantization or position handling at long
context.
