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

The optional MTP head is a one-layer multi-token prediction head (one dense
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
directory; when it is absent, MTP is off and the policy records why.

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

## Commands

```text
local-ai chat [options] <prompt>
  --max-tokens N     output cap (default 8192)
  --no-thinking      skip the checkpoint's xhigh reasoning (default on)
  --greedy           disable sampling
  --raw              skip the chat template

local-ai serve [options]
  --host IP          bind address (default 127.0.0.1)
  --port N           TCP and UDP port (default 8080)
  --api-key KEY      require Bearer authentication
  --no-thinking      disable reasoning
  --stall-timeout N  drop a generation whose client stopped reading; seconds,
                     default 30, accepted 10 to 3600
  --response-store DIR  keep Responses API responses in DIR (owner-only,
                     unencrypted); default: nothing is kept durably
  --reasoning-key FILE  enable reasoning.encrypted_content replay with an
                     owner-only persistent AES-256-GCM key in FILE
  --experimental-decision-head FILE  EXPERIMENTAL: serve
                     POST /v1/experimental/decisions with this judgment head
                     (probabilities only; default: off)

local-ai bonsai [options] <prompt>
  --max-tokens N     output cap (default 8192)
  --prompt-file PATH read a UTF-8 prompt instead of positional text
  --raw              skip the chat template
  --no-thinking      skip the checkpoint's xhigh reasoning (default on)
  --greedy           disable sampling
  --json             return token IDs and measured timings
  --tokenize         emit prompt token IDs without loading weights
  --export KIND      write an artifact instead of generating; takes no other
                     options. Kinds: index, mtp-head=DIR
  --no-speculation   disable both MTP and suffix lookup for comparison
  --mtp-depth N      drafts per speculative round, 1 to 4 (default 3)
```

There is no model flag. The engine runs one pinned checkpoint and discovers it
under `./models`, beside the executable, or under
`~/Library/Caches/local-ai/models`, and nothing at runtime reads an
environment variable other than `HOME`, plus `TMPDIR` where `serve` places
its private [temporary background store](#temporary-background-responses)
through the system temporary directory.

`chat` and `bonsai` stream text to stdout. Startup policy JSON goes to stderr.
`--no-thinking` skips the checkpoint's xhigh reasoning, which is on by default;
there is no flag to turn it back on, since it is already on. `--raw` drops the
chat template that reasoning is asked inside. Use `--` before prompt text that
starts with a hyphen.

`--export` takes one kind, so passing it twice is an error rather than a choice
between two exports. Every kind decodes nothing and takes no prompt. `index`
writes no file and therefore takes no directory; `mtp-head=DIR` needs only
`DIR/model_mtp_ternary.safetensors`, starts no engine, opens no GGUF, writes
`DIR/mtp-head-ptq1-v1.bin` (overwriting whatever was there), and prints the
artifact record as JSON.

## Server API

`serve` provides HTTP/1.1 and HTTP/2 on TCP. If
`~/Library/Application Support/local-ai/tls/cert.pem` and `key.pem` exist, it
also starts HTTP/3 on UDP at the same address and advertises it with `Alt-Svc`.
TLS and HTTP/3 are automatic runtime behavior, not a build option.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | readiness |
| `GET` | `/v1/models` | installed model and admitted `context_length`/`max_model_len` |
| `POST` | `/v1/chat/completions` | templated text and function calling |
| `POST` | `/v1/completions` | raw completion |
| `POST` | `/v1/responses` | text and function calling; stateless unless `--response-store` is given |
| `POST` | `/v1/responses/input_tokens` | exact templated input-token count, no generation |
| `GET`, `DELETE` | `/v1/responses/{id}` | retrieve or delete a stored or temporary background response, or (`GET ?stream=true`) resume a streamed background response |
| `GET` | `/v1/responses/{id}/input_items` | paginated input items of a stored or temporary background response |
| `POST` | `/v1/responses/{id}/cancel` | cancel a background response |
| `POST`, `GET`, `DELETE` | `/v1/conversations[/{id}[/items[/{item_id}]]]` | durable conversations and items ([native services](#native-services)) |
| `POST`, `GET`, `DELETE` | `/v1/files[/{id}[/content]]` | streamed multipart files |
| `POST` | `/v1/uploads[/{id}/parts\|complete\|cancel]` | multipart uploads with optional MD5 |
| `POST`, `GET` | `/v1/batches[/{id}[/cancel]]` | batches of the three text endpoints on the loaded model |
| `POST` | `/v1/experimental/decisions` | EXPERIMENTAL judgment-head probabilities (`--experimental-decision-head` only) |
| `POST` | `/v1/decisions` | always 404, explaining why OpenAI Decisions is not implemented |

Generation accepts `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, `seed`, and `stream`. Chat's
`max_completion_tokens` takes precedence over `max_tokens`; Responses uses
`max_output_tokens`. Cache affinity prefers `prompt_cache_key`, then
`session_id`, then `user`. Requests are limited to 2 MiB (multipart file and
upload-part bodies have their own streamed limits); media is rejected.

Chat supports function `tools`, assistant `tool_calls`, and tool-result messages
with `tool_call_id`. Native rendering and validation belong to `local-engine`
([contract](../local-engine/README.md#native-tool-calling)); HTTP never executes
a tool. Calls carry stable IDs and canonical JSON argument strings. A complete
validated call is delivered as one argument delta: incremental partial argument
streaming is not implemented. Chat finishes with `tool_calls` unless a token
limit takes precedence. Ordinary answer and reasoning text still stream.

Both APIs accept `tool_choice`, `parallel_tool_calls` and a function tool's
`strict`, and map them one to one onto the engine's native settings, which a
grammar enforces on every sampled token
([contract](../local-engine/README.md#constrained-tool-calling)).
`tool_choice` is `auto` (the default), `none`, `required`, or one named
function: `{"type":"function","function":{"name":"NAME"}}` in Chat and
`{"type":"function","name":"NAME"}` in Responses; the other API's shape,
unknown members, `allowed_tools` and other tool kinds are refused. `required`
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
validated after they are generated, as before. Legacy Completions refuses
`tools`, `tool_choice` and `parallel_tool_calls: false`. The grammar
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
`schema` are required, and unknown members are refused. Legacy Completions
accepts only `{"type":"text"}`.

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

Response storage is opt-in and **no Response is stored durably by default**
(conversations, files and batches use the always-on
[native services](#native-services) store); only
background responses are kept, briefly, without it. Without
`--response-store DIR` the endpoint is stateless: responses report
`store: false`, `store: true` and `previous_response_id` are refused, and
`GET`/`DELETE /v1/responses/{id}` answer 404 for everything except a
temporary background response (below); resend history in `input` instead.
With a store configured, `store` takes OpenAI's default of `true`
(an explicit `store: false` is honoured), and completed and incomplete
responses are written durably before their terminal event is sent; a failed
foreground generation is not stored, and a response whose write fails is
reported as failed rather than as stored (background responses differ; see
below). Each record is one JSON file holding the final response and its
resolved input (the earlier response's input and output, then
this request's items), so a follow-up never walks a chain and deleting an
earlier response does not break a later one. Records contain prompts, tool
results and the model's raw reasoning in plain text. The directory is created
`0700` and records `0600`; that is owner-only local data, not encryption at
rest. Only `resp_` IDs of ASCII letters and digits ever reach the filesystem.

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
these routes sit behind `--api-key` when it is set. `item_reference` inputs
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
At startup, a pending background record whose lease is free lost its writer to
a crash or restart and is marked `failed`; interrupted work is never resumed.
Graceful shutdown refuses new background requests (503), marks every
unfinished job `failed`, cancels it and waits for the workers to release the
engine before exiting. A job has no client to stall, so `--stall-timeout`
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
rather than by polling. A response still being generated by another live
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

`--reasoning-key FILE` enables `reasoning.encrypted_content`. The key is 32
random bytes in an owner-only file, created on first start without
overwriting anything and reused across restarts; a symlink, a non-regular
file, a file owned by another user, a file with group or other access, or a
malformed key is refused at startup. Reasoning output items then carry an
AES-256-GCM envelope (fresh random nonce per item) whose associated data binds
the envelope version, the model ID and the reasoning item's ID. A client may
send the item back, with its original `id`, in a later stateless request; the
authenticated text replaces any plaintext beside it, and an altered envelope,
one from another key, model or item, or an unknown version fails the request.
`include: ["reasoning.encrypted_content"]` is accepted only with a key and is
the only `include` value implemented; without a key it, and any
`encrypted_content` input, are refused. Replay after a server restart and
tamper rejection are covered by tests.

The key option is replay authentication, **not secrecy**: the raw reasoning is
still sent in the same response as `reasoning_text` content (streaming and
buffered), and with a store configured it is still written in plain text. It is
not encryption at rest and does not hide chain-of-thought from the client or
from anyone who can read the store.

`POST /v1/responses/input_tokens` accepts the Responses input contract
(`input`, `instructions`, `tools`, `reasoning`, `previous_response_id`, `text`,
`tool_choice`, `parallel_tool_calls`, `truncation`, `model`) and returns
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
(Responses): `none` or `xhigh`. A server started with `--no-thinking` rejects
`xhigh`. Usage counts reasoning tokens from generated IDs through the first
`</think>` token, inclusive; a response truncated before that token counts all
generated tokens as reasoning. Prior reasoning is retained in replayed history.
Responses accepts `reasoning.context: "all_turns"` (also the effective mode
reported for the default `auto`); `current_turn` filtering is not implemented.
Responses reports measured prefix reuse as `input_tokens_details.cached_tokens`.
`cache_write_tokens` is zero: local cache creation has no separately accounted
or charged cache-write tier.

This is **not the full OpenAI platform contract**. Schema keywords
llguidance cannot enforce exactly, strict tool schemas the native grammar
cannot enforce exactly (including any without `"additionalProperties": false`),
`allowed_tools` and non-function tool choices, built-in/hosted
tools, reasoning summaries, `include` values other than
`reasoning.encrypted_content` (and that one without `--reasoning-key`),
`truncation:auto`, log-probabilities, images and audio are rejected.
`tool_choice` `auto`, `none`, `required` and a named function,
`parallel_tool_calls` and tool `strict` are supported and enforced during
decoding, as is a structured format beside tools; a grammar guarantees the
form of what is generated, not that the model chooses well. OpenAI Decisions
is not implemented: the opt-in
[experimental decisions](#experimental-decisions) route returns a head's
probabilities without the confidence and refusal Decisions requires, and any
calibration holds only within the loaded artifact's own evaluated scope (for
example, the legacy devtools-v1 head's two-option Yes/No code-diff questions,
or the development-only
[hard-v1 results](../local-engine/README.md#kev-hard-v1-preparation-and-development-only-training)).
The MTP head and
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
compaction, resuming a background Response interrupted by a restart and
vector stores still need server implementations; OpenAI-compatible Decisions needs
a judgment head with evaluated confidence and refusal semantics; media,
embedding, audio and moderation capabilities need suitable models or heads. Hosted tools,
evals, fine-tuning and administrative APIs also need their own services. None
is implemented by merely accepting its request fields.

[Codex at 4aa94dc](https://github.com/openai/codex/tree/4aa94dce270de668eff6e2fa8585c82385e84455)
uses stateless Responses, requests `reasoning.encrypted_content`, and sends
`client_metadata` (an extension absent from the pinned public spec). Responses
creation accepts that field as an optional string-to-string map, matching
[Codex's request type](https://github.com/openai/codex/blob/4aa94dce270de668eff6e2fa8585c82385e84455/codex-rs/codex-api/src/common.rs#L278-L304).
It is opaque tracking data: discarded, not echoed or persisted, and never
used for prompts, sampling or cache-session selection. It is distinct from
the public `metadata` field, which is retained on the response. With
`--reasoning-key` the encrypted-content request is honoured. Reasoning
summaries and hosted tools remain unsupported, strict tools are accepted only
when their schemas can be enforced exactly, and the constrained tool settings
have not been exercised with Codex, so this server is
**not a drop-in Codex provider**. Codex handles raw `reasoning_text` and
summary events separately; raw reasoning is not relabeled as a summary.
The Oh My Pi example in the engine README configures that client to use only
implemented capabilities.

Legacy Completions now rejects unsupported `stop`, `n`, `logprobs`, `logit_bias`,
`best_of`, `echo`, `suffix` and streaming usage options rather than silently
ignoring them. Chat also rejects `store:true` and non-default `verbosity`.
Error envelopes include nullable `code` and `param`; choices include nullable
`logprobs`; Responses includes nullable `access_programs`. The model's `created`
timestamp is its registration time at server startup, not its training date.

With `stream: true`, Chat/completions are `text/event-stream` chunks terminated by
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
checkpoint, and the generation holds its queue slot indefinitely. (Beside other
requests the worker no longer waits on a slow reader at all: events queue per
request and reach the channel without blocking, so only the stalled request
is held.) When the budget is exceeded the server logs, cancels
that generation, releases the queue slot, and drops the request, leaving the
engine free for the next client. The default is 30 seconds; a value outside
10 to 3600 inclusive, or one that is not a whole number of seconds, is refused
at startup.

Both HTTP body pumps share an event-driven backpressure wait. A full channel
parks the pump thread until capacity becomes available, the receiver closes,
or the single stall deadline expires. It does not retry every few milliseconds.
The engine's worker likewise blocks on its request channel when idle; its
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
time-to-first-token. Prefill now reports a boundary at every 128-token block (or
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
from `--response-store`, which stays opt-in. Storage, limits, crash safety and
the native API, including the workspace search that has no HTTP route, are in
[local-services/README.md](../local-services/README.md). Routes sit behind
`--api-key` and are also served over HTTP/3. Service CRUD uses a bounded
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
  past that the answer is a 503 with `Retry-After: 1`. Expired objects are a
  404, and changes to an expired or cancelled upload are a 410. Expiry is
  enforced on access. Bytes are purged physically only at startup or through
  the native `purge_file_storage`. A `purpose` is stored and echoed only; it
  enables no processing.
- **Batches.** These run `/v1/responses`, `/v1/chat/completions` and
  `/v1/completions`, with `completion_window: "24h"`, on the loaded model
  only. The whole input file is validated before anything is stored.
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

These services do not resume background Responses after a crash or restart.
Batch resumption is a separate mechanism. They also provide no compaction,
training or change to the model.

### Experimental decisions

`--experimental-decision-head FILE` enables `POST /v1/experimental/decisions`.
Without the flag the route is a 404 and no head is loaded. The head is opened
and its width checked before the model loads, so a wrong file fails startup,
as does a head whose `renderer` metadata is outside the native engine's
supported devtools-v1 and hard-v1 formats. The server announces the route as
experimental together with what the loaded file declares: its renderer
(declared, or assumed for a legacy head that declares none) and its
`calibration_scope` metadata, or an explicit "unknown" when it has none. The
route bridges to the engine's
[native decisions](../local-engine/README.md#experimental-native-decisions)
and sits behind `--api-key` like every other route.

The body borrows the Decisions request vocabulary but accepts only what the
native renderer reproduces exactly: `input` must be a non-empty text string
(rendered as the `State:`); `questions` holds 1 to 200 questions of `type`
`predicate`, `choice` (2 to 26 string or boolean `choices[].value`) or
`score` (2 to 10 `levels[].label`), each with `instructions` and an optional
`name`; `model` and `safety_identifier` are accepted and ignored. Message
arrays, images, non-empty option or level descriptions and unknown fields are
refused with 400 rather than approximated. A 200 response has
`"object":"experimental.decision"` and one answer per question: a predicate's
`probability` of Yes; a choice's argmax `choice` and per-value
`probabilities`; a score's probability-weighted mean level index as `score`
and per-level `probabilities`. `usage.input_tokens` sums the rendered prompt
tokens (`output_tokens` is 0), and an `experimental` object repeats the
`renderer`, `renderer_source` (`"artifact"` when the head declared it,
`"assumed"` for a legacy head), the loaded head's `calibration_scope` (or the
unknown-provenance text) and the limitations.

Hard-v1 and devtools-v1 use identical prompt framing; compatibility does not
prepare a caller's state or add training descriptions. To reproduce a described
hard-v1 option, pass its full rendered label as a string choice (for example,
`"Yes: Meets all constraints"`), not a bare predicate or a description field.
Preserve training option order. This preserves prompt bytes, not calibration
outside the artifact's recorded scope.

Questions run one after another over the same input, each taking one slot of
the engine's bounded FIFO queue and running only once no generation is
active, so other requests interleave between questions; a long input delays
generations queued behind a running capture. At most eight decision requests
are admitted at once; more get 503 with `Retry-After: 1`, and a full engine
queue gets the usual queue-full 503. A client that disconnects drops its
pending decision, which cancels it, queued or running; shutdown refuses new
requests and ends waiting ones with 503, cancelling their decisions the same
way. These waits are futures woken by the engine or by shutdown, not polling.

`POST /v1/decisions` is always a 404 with an explanation. OpenAI's Decisions
responses require a `confidence` on every choice and score answer and may
answer with a `refusal`; the head produces neither, and a confidence could
only be invented. The existing devtools-v1 head declares no renderer or
calibration scope, so the route reports `renderer_source: "assumed"` and an
unknown calibration scope for it; as measured when it was evaluated, its
probabilities are calibrated only for the two-option Yes/No code-diff
questions it was trained on, and are the head's softmax output for anything
else. This is not a production judgment service. Real-model
native and HTTP checks reproduced stored-feature probabilities bit-for-bit
on four rows from both training tasks and both option orders, with and without
MTP. Queue ordering, cancellation, and generation/cache parity after a decision
also passed. Capture measured about 115–130 tokens/s on the M4 Pro, and later
generations wait behind it. These checks establish implementation parity,
not general judgment quality or calibration.

The three development-only hard-v1 heads subsequently completed private CPU
training, reaching 58.10–58.21% accuracy against a 30.12% majority baseline on
the 883-row checkpoint-selection split. They remain uninstalled experimental
artifacts, not production heads. Real-model checks passed for all three heads:
36 native cases with MTP off/on and 18 HTTP cases with MTP on matched stored
features exactly across choices, both boolean orders, predicates and scores.
See the [hard-v1 results and limitations](../local-engine/README.md#kev-hard-v1-preparation-and-development-only-training)
for calibration, provenance and weak task families. No locked test was scored;
these synthetic results do not establish coding quality or Decisions confidence
and refusal semantics.

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
between steps and leave when they finish or are cancelled.

A request alone runs exactly the single-sequence rounds it always ran, with
speculation; its text and speed are unchanged. Taking speculative rounds in
turn instead of batching measured 32.7 tok/s aggregate at two streams against
36.8 batched, so batching starts at two.

#### Prompts prefill inside the decode steps

Admission only restores a request's reusable prompt prefix. Its prefill then
runs inside the engine's steps, one prompt per step: the one with the fewest
tokens left. A prompt of at most 128 tokens to prefill (one block; a typical
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
prefills up to 512 tokens, which only bounds how long a newly arrived request
waits to be admitted. A cancellation between two chunks ends the request with
no tokens and clears the prompt cache, as a cancelled single-pass prefill did.

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

#### Speculation in a batch

A batched step verifies drafts too. Each sequence contributes its seed row
and any drafts as consecutive rows of one stacked block, so a pass of four
sequences with three drafts each is one 16-row pass. A sequence's verify rows
run its convolution and recurrence with the multi-row kernels at a row offset
(the same pipelines bound with byte offsets, so its rows compute what a block
of only those rows computes), reading its state and history and writing the
block's final ones to shared spares that nobody reads; the K/V rows are
appended at its position and attended with the block attention kernel at the
same offset. Every row is projected to logits and selected on the GPU; each
sequence then verifies its drafts exactly as a single sequence does, and one
submission replays every verifying sequence's kept rows from the start of the
round (its raw QKV rows, decay and beta were kept per layer), as a partial
commit always has, 2.5-12.7 ms for 2 to 8 sequences. Positions advance by the
kept rows, and the kept rows' hidden reaches the MTP head through the same
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
so no sequence is swapped in to draft. A pass costs about 2.1 ms plus 0.5 ms
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
hidden rows owed to its MTP head. A verifying step keeps each recurrent
layer's raw QKV rows, decay and beta for its rows (about 2 MB per stacked row,
allocated on first use at the next power of two of the rows, up to 126 MB at
64 rows) plus one spare state, and batched logits grow to the stacked rows
(1 MB each). The measured footprint grew by about 200 MB per stream at these
lengths. Admission shares the context the memory policy sized for one
sequence: a request joins only while the running requests'
prompt-plus-`max_tokens` reservations, plus each extra sequence's fixed state
expressed in K/V tokens (about 2,340), fit in it; otherwise it waits for a
running request to finish. At most eight decode together, in at most 64
stacked rows.

## Resource policy

All choices and reasons are reported in startup JSON.

- **Model:** a library caller's explicit `Engine::open_model` path; otherwise
  the pinned path in the working directory, beside the executable, or under
  `~/Library/Caches/local-ai/models`. The commands have no model flag.
- **Context:** largest value up to 262,144 tokens whose fully grown model state
  fits 90% of Metal's recommended working set.
- **K/V:** Hadamard-rotated Q8 for the target, always (see
  [K/V cache](#kv-cache)); the MTP head's one-layer cache is F16.
- **Prefill:** fixed 128-token chunks with Metal 4 kernels where supported;
  beside two or more decoding requests, a long prompt prefills 24 to 31
  tokens per step inside their batched pass, and 32 per step in a pass of its
  own beside one.
- **Speculation:** suffix lookup is enabled; the mixed ternary/int8 MTP head
  artifact, when installed at `models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin`,
  adds gated depth-3 drafting. Without it MTP is off and the policy says why.
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
default settings matched `--no-speculation` token for token on the six
benchmark prompts; one server prompt split at character 624 ("of entries"
against "of all entries"), deterministically, the same way on every run.

The suffix store finds a matching generated-token suffix and proposes the
known continuation. Draft depth adapts to match length and remaining context.
When the optional head is installed, it drafts up to three tokens. A gate
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
emitted the `--no-speculation` tokens on all six prompts. Some non-default
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

With a greedy sampler and no penalty, the verify block (and a plain decode
step) also selects each row's argmax in its own command buffer, with the same
`total_cmp` order and ties to the lower id as the GPU top-k, and flags any
non-finite logit; the host reads one id per row instead of submitting a top-k
per row and scanning every logit. Other samplers keep the host path.

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

### What the speculation gain is made of

`--no-speculation` disables two independent mechanisms under one flag: the MTP
head, which drafts speculatively ahead, and n-gram suffix lookup
(`ngram_policy`, `min_match: 12`), which proposes the known continuation of a
previously-seen 12-token suffix. Both are honestly called speculation, the flag
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
   discovered at startup, and trimmed within the automatic disk budget.

Disk snapshots are written by one background thread the engine owns. The
engine thread pays only the GPU readback; hashing, the write, `fsync`, rename
and trim run behind the request. A snapshot is indexed for reuse only after
its rename is reported, so a file still being written is never loaded; a
request whose best disk match is still in flight waits for that one write.
At most one job is in flight and one queued: a newer job replaces the queued
one, except that a request tail never displaces a shared boundary. Dropping
the engine finishes pending writes. A failed write is a cache miss reported
on stderr, not a request error. Trim keeps an in-memory index of the
directory instead of re-reading every snapshot header after each store.

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

The target's full-attention layers store keys and values as Q8: int8 values
with an F16 scale per 32, 8.5 bits per value and 34 KiB per token across the
sixteen layers, against F16's 64 KiB. There is no other target format and no
fallback: Q8's 34 KiB per token is what the context policy budgets. Scores and
accumulation remain F32. The MTP head's one-layer cache always remains F16.

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
kernel on every build. A **rejected candidate** reduced its 128-token splits
to 32 below that threshold to expose more parallel work. A one-layer
microbenchmark on the M4 Pro (median GPU time per call over nine
trials of 64 calls; 128-token baseline given as the range of two bracketing
runs; 64-token splits were also measured and were slower than 32 at every
short prefix except a tie at 33):

| Layout, prefix | 128-token splits | 32-token splits |
| --- | ---: | ---: |
| F16, 64 | 19.1–23.9 µs | 11.9 µs |
| F16, 128 | 34.1–37.4 µs | 12.3 µs |
| F16, 512 | 34.6–38.1 µs | 14.4 µs |
| F16, 1,023 | 44.1–47.5 µs | 21.1 µs |
| Q8, 33 | 13.1–13.7 µs | 12.8 µs |
| Q8, 128 | 36.6–39.7 µs | 13.2 µs |
| Q8, 512 | 37.5–40.8 µs | 16.8 µs |
| Q8, 1,023 | 40.3–42.7 µs | 26.7 µs |
| F16 / Q8, 1,024 (control) | 19.5–23.4 / 20.2–20.6 µs | 19.7 / 20.6 µs |
| F16 / Q8, 2,048 (control) | 29.6–32.6 / 34.7–34.9 µs | 29.4 / 35.0 µs |

Short prefixes match the F64 reference, and prefixes at or above 1,024 stay
bitwise identical to the 128-token build. However, an interleaved six-prompt
full-model comparison (300-token cap, AC power, no sleep) changed greedy text
on two of six plain-decode prompts: arithmetic and essay. Plain-decode
geomean was 31.95 → 32.50 tok/s (1.7%); default speculative decode was
38.92 → 38.97 tok/s (0.1%, within noise), with identical text in all six
speculative pairs. These are one-round observations, not stable throughput
estimates. The candidate failed the byte-identity acceptance bar and was
removed; the runtime retains 128-token SIMD splits.

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
that end-of-request readback (673 MB) on its way to the disk writer.

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

| Workload | Result | Source |
| --- | ---: | --- |
| Plain short-prompt decode, six-prompt geomean | 30.4–32.3 tok/s | best-of-two A/B runs below, across recent kernel changes |
| Default speculative decode, six-prompt geomean | 38.0–39.2 tok/s | same runs; 38.6 in the depth/margin sweep |
| Two 600-token copy-edit prompts, suffix lookup | 89.8 and 90.5 tok/s | 5- to 128-row kernel change |
| Cold prefill, 2K–8K tokens | 123–126 tok/s | large-batch kernel change |
| Cold prefill, 32,767 tokens | 105.9 tok/s | same, one run |
| 128K prefill | 56 tok/s | [original build](https://github.com/debanjanbasu/local-ai/commit/b389329); not re-measured |

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
byte-identical to before and to `--no-speculation`. With `tools/benchmark.py`,
best of three, the planets prompt went from 23.58 to 24.74 tok/s at
`--mtp-depth 3` (19.02 to 19.18 without speculation), and a prompt asking for a
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

Cold prefill (prompt cache deleted, `--no-speculation`, this repository's
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
decode 18.84 to 21.64 tok/s and speculative decode (`--mtp-depth 3`) 23.58 to
25.44 tok/s, interleaved runs of both builds. Removing one fusion at a time
cost: RMSNorm with rotation 3.4%, QKV/gate concatenation 4.0%, BF16 folding
2.3%, SwiGLU 1.3%, convolution 0.8%. The BF16 load change alone took plain
decode from 19.23 to 20.39 tok/s.

## Tried and rejected

Entries are measured on the build current when they were tried. Decode rates
of roughly 17–25 tok/s come from earlier kernels (17.5 tok/s plain decode at
[the original build](https://github.com/debanjanbasu/local-ai/commit/b389329),
about 22 before the half-prefix decoder) and compare only within
their own entry, not with current rates.

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
- `--mtp-depth 4` once five-row verify blocks cost 73.7 against four rows'
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
under `.amp/in/`, including prompt/token hashes and build provenance.
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
passed through the engine API, including schema-constrained raw completions.
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
retention and the experimental decisions route were added after those live
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
