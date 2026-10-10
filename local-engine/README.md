# local-engine

Synchronous, network-free Rust library for native Ternary Bonsai 2 27B
inference on Apple Silicon. It provides model discovery, Metal execution,
prompt caching, lossless speculation, tokenization, and sampling.

```rust,no_run
let mut engine = local_engine::Engine::open()?;
let answer = engine.chat("What is 17 * 23?")?;
println!("{}", answer.content);
# Ok::<(), local_engine::Error>(())
```

`Engine` is `Send` but not `Sync`. Use `Engine::into_handle()` for a cloneable
`EngineHandle`; requests return cancellable `EventStream`s that are both a
blocking `Iterator` and a `Stream`. The crate depends on `tokio` (features `rt`
and `sync`) for its event transport and on `futures-core` for the `Stream`
implementation, and has no HTTP server dependency. See the repository
[README](../README.md#library-use) and [technical reference](../docs/BONSAI.md).

## Native agent integration

The engine is the reusable Rust boundary. A future coding harness should call
`Engine` or `EngineHandle` directly; it should not need a loopback HTTP server,
`OpenAI` request objects, or an SSE parser. The `local-ai` server is a protocol
adapter, not the owner of inference behavior.

[Oh My Pi](https://github.com/can1357/oh-my-pi/tree/dde3fc44ed16d3bbec292893c7a902e92d6ce00e)
is a reference for agent requirements, not a dependency. Its canonical
[agent messages and streaming loop](https://github.com/can1357/oh-my-pi/blob/dde3fc44ed16d3bbec292893c7a902e92d6ce00e/packages/agent/src/agent-loop.ts)
and [tool-batch recovery rules](https://github.com/can1357/oh-my-pi/blob/dde3fc44ed16d3bbec292893c7a902e92d6ce00e/packages/coding-agent/src/session/turn-recovery.ts)
inform this division:

| Engine owns | Harness owns |
| --- | --- |
| Checkpoint-specific prompt rendering and tokenization | Conversation persistence, branching, compaction and steering |
| Model limits, cache occupancy and generation measurements | Reserving output space and choosing when to compact |
| Metal scheduling, hybrid recurrent/KV state and prefix reuse | Tool registry, permissions, sandboxing and side effects |
| Cancellable generation and prefill progress | Deadlines, retry policy and tool execution concurrency |
| Model output decoding | Whether a completed tool call may execute or be replayed |

### Counting prompt tokens

`Engine::count_chat_tokens(&ChatRequest)` and
`Engine::count_completion_tokens(&CompletionRequest)` return the exact number
of prompt tokens generation would prefill. A chat request is validated,
rendered (messages, replayed tool calls and results, tool definitions and
`thinking`) and tokenized, special tokens included, by the same code as
`chat_with`, so the count equals `prompt_tokens` in the generation statistics,
cached prefix included, and invalid prompt content fails with the error
generation would return. A completion is tokenized as `complete` tokenizes it,
without a template; an empty prompt counts zero although generation rejects it.
`max_tokens`, `sampling`, `session` and `response_format` are ignored; no schema
is compiled and the count is not checked against the context window.
Counting is CPU-only and submits no GPU work.
`EngineHandle` offers the same two methods against the worker's shared
tokenizer: they run on the calling thread, queue no job and never wait for
running generations, so they work while the queue is full. Async callers should
run long prompts on a blocking pool.

## Structured output

`ChatRequest::response_format` and `CompletionRequest::response_format` take a
`ResponseFormat`: `Text` (default), `JsonObject` (`OpenAI` `json_object`) or
`JsonSchema(schema)` (the `schema` of `OpenAI` `json_schema`; `name`,
`description` and `strict` are not needed by the engine, and the schema is
always enforced). The format is compiled with
[llguidance](https://github.com/guidance-ai/llguidance) before the request is
queued (`Engine::chat_with`/`complete`, `EngineHandle::chat`/`complete`), so
invalid or unsupported schemas fail with `Error::InvalidArgument` before any
generation. Every target token selection (first token, plain decode, n-gram
and MTP verification, batched rows) is masked to what the grammar allows;
drafts stay unconstrained proposals that the masked target verifies.

- With `thinking`, reasoning is unconstrained and the format applies from the
  token after `</think>`; without it, and for raw completions, from the first
  generated token, with no whitespace before the document.
- End-of-sequence is only selectable once the document is complete.
  `GenerationStats::response_format_complete` is `Some(true)` only then;
  a token limit or cancellation yields `Some(false)` and incomplete JSON.
  With `thinking`, the model may also end its turn before `</think>`, which
  is unconstrained; that `Finished` carries `Some(false)` and no answer at
  all, so check the flag rather than trusting a stop at end-of-sequence
  (`local-ai` reports it as a failure).
- Compilation (llguidance 1.9.1) is strict: unsupported keywords (for example
  `uniqueItems`, `contains`, `not`), unknown `format`s, `x-guidance`, and
  `$ref`s outside the document are rejected rather than ignored. `oneOf` is
  accepted only when its branches are provably disjoint (for example
  different JSON types, or objects sharing a required property with different
  `const` values); overlapping branches, such as `integer` and `number`, or
  ones whose exclusivity cannot be proven, such as `$ref` branches, are
  rejected. Nothing is fetched. Printable characters
  are generated literally rather than as `\uXXXX` escapes.
- Object properties are generated in the schema's key order, which is
  preserved as sent. (Tool schemas are different; see below.)
- A format cannot be combined with `tools`: it constrains the whole answer,
  leaving no room for a call.
- The first constrained request builds the tokenizer's grammar tables once
  (about 0.5 s on CPU); later schemas compile in well under a millisecond.

## Native tool calling

`ChatRequest::tools` takes `ToolDefinition { name, description, parameters }`,
where `parameters` is a JSON Schema for the arguments object. With tools
configured the engine renders them exactly as the pinned checkpoint template
([`chat_template.jinja` at `3f926b4`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/blob/3f926b415992eaa2ae9dd7b573706494d6bbf787/chat_template.jinja))
does, and parses the model's own call format from the visible answer:

```text
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

String arguments are raw (possibly multiline) text; other values are JSON. This
is not the generic Qwen JSON `<tool_call>` format.

```rust,no_run
use local_engine::{ChatMessage, ChatRequest, Engine, Event, Sampling, ToolDefinition};
use std::ops::ControlFlow;

let mut engine = Engine::open()?;
let request = ChatRequest {
    messages: vec![ChatMessage { role: "user".into(), content: "Weather in Paris?".into(), ..ChatMessage::default() }],
    max_tokens: 512,
    sampling: Sampling::default(),
    thinking: true,
    session: None,
    tools: vec![ToolDefinition {
        name: "get_weather".into(),
        description: Some("Current weather for a city.".into()),
        parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    }],
    response_format: local_engine::ResponseFormat::Text,
};
engine.chat_with(&request, |event| {
    if let Event::ToolCall(call) = event {
        println!("{} {} {}", call.id, call.name, call.arguments);
    }
    ControlFlow::Continue(())
})?;
# Ok::<(), local_engine::Error>(())
```

Contract:

- Reasoning is split off first; tool markup is recognised only in the answer,
  and only when `tools` is non-empty. Otherwise it is ordinary content.
- `Event::ToolCall(ToolCall { id, name, arguments })` is emitted only once a
  whole `<tool_call>` block has been generated, parsed and validated against
  the tool's schema. Ordinary text still streams; only a possible `<tool_call>`
  prefix and trailing whitespace are held back. Partial calls are never emitted.
- A malformed, unknown, schema-violating or truncated call ends the generation:
  `EventStream` yields `Event::Error` and no `Event::Finished`; `Engine::chat_with`
  returns `Err`. Earlier valid calls in the same answer have already been emitted.
- IDs (`call_…`) are generated by the engine, identical in the stream and in
  `ChatOutput::tool_calls`, and unique across generations. They are not shown to
  the model.
- Replay a turn as an `assistant` message with `tool_calls` (and its
  `reasoning_content`, which the template keeps in history), followed by one
  `tool` message per call with `tool_call_id`. Consecutive results render as one
  user turn of `<tool_response>` blocks, reordered by ID into call order because
  Bonsai's template has no model-visible IDs. Duplicate IDs, unanswered calls,
  unmatched or repeated results, unknown tools, invalid arguments and invalid
  schemas are rejected with `Error::InvalidArgument` before generation.
- `developer` is accepted as the first message and fills the template's single
  system slot, like `system`.
- There is no forced `tool_choice`: the model decides whether to call.
- The engine never executes a tool. Execution, permissions and side effects
  belong to the harness.

Schemas are compiled once per request with the `jsonschema` crate, defaulting
to draft 2020-12. Assertions such as `pattern`, `uniqueItems` and
`dependentRequired` are enforced, along with known formats. Document-local
references work; network/file retrieval and unknown dialects are rejected.
Annotations such as `title`, `description` and `default` do not constrain values.
Ambiguous raw parameter text remains a string when its declared schema accepts
strings, including nullable strings; the harness should avoid ambiguous unions.
Object keys in rendered schemas and replayed arguments are in sorted order
rather than client order, whether or not `serde_json`'s `preserve_order` is
enabled.

`local-ai` exposes these native calls through Chat Completions and a
text/function subset of Responses, stateless unless the server is started with
`--response-store`, which also enables cancellable background Responses,
polled or streamed with resumable events. It maps Chat `response_format` and
Responses `text.format` onto `ResponseFormat`. See
[server API](../docs/BONSAI.md#server-api) for storage, background streams,
structured output, encrypted reasoning replay, token counting and
compatibility limits.
`OpenAI`'s Decisions API is **not implemented**. Keep tool execution out of the engine;
in particular, a retry must not repeat a partially successful batch's side
effects. Do not
change the pinned checkpoint template merely to match another Qwen model's
conventions.

### Oh My Pi custom provider

Oh My Pi can drive `local-ai serve` through its `openai-responses` provider.
Add this to `~/.omp/agent/models.yml`:

```yaml
providers:
  local-ai:
    baseUrl: http://127.0.0.1:8080/v1
    auth: none # with serve --api-key, use apiKey: LOCAL_AI_API_KEY (an env var name)
    api: openai-responses
    compat:
      supportsStrictMode: false
      supportsReasoningSummary: false
      includeEncryptedReasoning: false
      reasoningDisableMode: none-effort
      supportsForcedToolChoice: false
      reasoningEffortMap: {minimal: xhigh, low: xhigh, medium: xhigh, high: xhigh, max: xhigh}
    models:
      - id: ternary-bonsai-2-27b # not checked by the server
        name: Ternary Bonsai 2 27B (local-ai)
        reasoning: true
        thinking: {mode: effort, efforts: [xhigh], defaultLevel: xhigh, requiresEffort: false}
        input: [text]
        contextWindow: 65536 # use context_length from GET /v1/models
        maxTokens: 8192
        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
```

The flags replace Oh My Pi defaults for an unrecognized host that this server
refuses or would misread. The keys come from the
[`models.yml` schema](https://github.com/can1357/oh-my-pi/blob/703a261df9533e74f16aa8d38e64eb41977ccf89/packages/coding-agent/src/config/models-config-schema-bundle.ts#L40-L88)
and, for `includeEncryptedReasoning` and `reasoningDisableMode`, the runtime
[compatibility axes](https://github.com/can1357/oh-my-pi/blob/703a261df9533e74f16aa8d38e64eb41977ccf89/packages/catalog/src/compat/axes.ts#L131-L161),
which `models.yml` accepts without a warning
([docs](https://github.com/can1357/oh-my-pi/blob/703a261df9533e74f16aa8d38e64eb41977ccf89/docs/models.md#unknown-compatibility-keys)).
The [Responses defaults](https://github.com/can1357/oh-my-pi/blob/703a261df9533e74f16aa8d38e64eb41977ccf89/packages/catalog/src/compat/resolve.ts#L697-L848),
[request builder](https://github.com/can1357/oh-my-pi/blob/703a261df9533e74f16aa8d38e64eb41977ccf89/packages/ai/src/providers/openai-responses.ts#L1600-L1702),
[reasoning policy](https://github.com/can1357/oh-my-pi/blob/703a261df9533e74f16aa8d38e64eb41977ccf89/packages/ai/src/providers/openai-shared.ts#L970-L1084),
[reasoning fields](https://github.com/can1357/oh-my-pi/blob/703a261df9533e74f16aa8d38e64eb41977ccf89/packages/ai/src/providers/openai-shared.ts#L4019-L4096)
and [effort resolution](https://github.com/can1357/oh-my-pi/blob/703a261df9533e74f16aa8d38e64eb41977ccf89/packages/ai/src/stream.ts#L1707-L1737)
show what each flag changes:

- `includeEncryptedReasoning: false` stops `include: ["reasoning.encrypted_content"]`,
  sent on every reasoning request by default. The server refuses that `include`
  unless it was started with `--reasoning-key`; with a key it is honoured, but
  this configuration was not re-checked against Oh My Pi with it enabled.
- `supportsReasoningSummary: false` stops `reasoning.summary: "auto"`.
- `reasoningDisableMode: none-effort` makes `--thinking off` send
  `reasoning.effort: "none"`. The default sends the lowest listed effort,
  `xhigh`, so turning thinking off would silently keep it on.
- `efforts: [xhigh]` and `reasoningEffortMap` reflect the checkpoint's only
  reasoning mode: every level other than off becomes `xhigh`, and a fixed
  internal effort maps to it instead of failing as unsupported.
  `requiresEffort: false` declares that off is accepted, so it is never
  clamped up to `xhigh`; auto-detection gave the same result in the check below.
- `supportsStrictMode: false` keeps `strict` out of tool definitions. It is
  already off for hosts other than `OpenAI` and a few known providers; set it
  explicitly because `strict: true` is refused.
- `supportsForcedToolChoice: false` turns Oh My Pi's forced tool choices into
  `auto`, so the model may answer without calling the tool. Without it those
  requests fail.

Oh My Pi already sends `store: false`, resends earlier turns in `input`, puts
the system prompt in `instructions`, and leaves out the `developer` role and
`stream_options` for non-OpenAI hosts. With `input: [text]`, images in tool
results become text placeholders. Presence or repetition penalties configured
in Oh My Pi are unknown fields to this Responses endpoint and fail the request.
For a server started with `--no-thinking`, set `reasoning: false` and remove
`thinking`; Oh My Pi then sends no reasoning field.

This configuration was checked by reading Oh My Pi at
[`703a261`](https://github.com/can1357/oh-my-pi/tree/703a261df9533e74f16aa8d38e64eb41977ccf89)
and by running `omp` 18.8.6 against a recording stand-in for `/v1/responses`,
not against the model. With the flags, a read, tool-result and reply turn at
`--thinking high` and `off` sent `reasoning.effort` `xhigh` and `none` and no
`include`, summary or `strict`. Without the flags, both levels sent `include`
and `xhigh`, and `high` also sent `summary: "auto"`. The recorded turn is the
`responses_accept_an_oh_my_pi_read_and_tool_result_turn` test in
`local-ai/src/cli/serve/protocol_tests.rs`.

`OpenAI`'s [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
uses `input` and named question/answer arrays. At the pinned revision, Oh My Pi's
[`openrouter-decisions` adapter](https://github.com/can1357/oh-my-pi/blob/dde3fc44ed16d3bbec292893c7a902e92d6ce00e/packages/ai/src/judgment/typesafe.ts)
uses the older System One `state`, keyed questions/answers and `noul` predicate
format. They require distinct adapters to any future native judgment interface.
A judgment head is separate from the speculative MTP head: MTP accelerates
generation and does not train classification, tool use or confidence calibration.

### Experimental judgment-head preparation

`mtp-capture features` captures selected output-normalized hidden states without
changing the frozen model. Input JSONL rows contain a non-empty `id`, raw
`text` (no chat template), optional `metadata`, and exactly one of:

- `positions`: sorted zero-based token indices (the final token is allowed);
  this is not the previous-token layout of MTP training shards;
- `token_end_offsets`: sorted exclusive UTF-8 byte endpoints into `text`. One
  native Bonsai encode of the whole text resolves each endpoint to the token
  ending exactly there. Endpoints at zero, past the text, inside a UTF-8
  character or inside a merged token are rejected, never rounded.

Output `features.jsonl` records tokens, resolved positions and the matching
little-endian FP16 binary file. Existing output is never overwritten.
`--validate-only` tokenizes and checks every row and prints row, vector, token
and per-split counts; it does not open the Metal engine and writes nothing.

Capture uses the hidden-output path, omitting vocabulary projection and verify
rollback work. An M4 Pro A/B/B/A run on 16 real rows (60-token blocks) took
34.860/33.472/33.467/34.859 seconds for verify/hidden/hidden/verify, with identical
exported FP16 bytes: about 4% less capture time, not a general decode speedup.

The Kev pilot pipeline is:

```sh
# 1. Fetch the pinned public files yourself (nothing is uploaded):
#    https://raw.githubusercontent.com/jaredpalmer/kev/5e42a7a03f28134853dd3ff77461457e921e5ec1/evals/devtools-v1/{manifest,train,development,test}.json[l]
python3 tools/judgment_prepare.py --kev-dir KEV --out PREP --pilot-rows 32
cargo build -p local-ai --release --bin mtp-capture
target/release/mtp-capture features --validate-only PREP/rows.jsonl
target/release/mtp-capture features --out FEATURES --max-tokens 2048 PREP/rows.jsonl
python3 tools/judgment_train.py validate FEATURES
python3 tools/kaggle_judgment_job.py --kernel OWNER/judgment-head \
  --dataset OWNER/FEATURES_DATASET --output STAGING
```

`tools/judgment_prepare.py` reads Kev's pinned
[`devtools-v1` manifest](https://github.com/jaredpalmer/kev/blob/5e42a7a03f28134853dd3ff77461457e921e5ec1/evals/devtools-v1/manifest.json)
(its sha256 is pinned in the script) and checks each partition's sha256, size,
record and per-source counts and every record's schema. It converts only the
audited coding questions with native labels, without LLM relabeling:
`CodeReviewer` `needs_comment` (human label) and `CommitPackFT` `message_match`
(label by construction). The later
[Kev-27B label audit](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md)
excludes `FlakeFlagger` and commit-change-type because the supplied state does
not determine their labels; Aegis is outside the coding focus; `When2Call` and
prompt-injection are evaluation-only. Each excluded row is counted by reason in
`PREP/report.json`; unsupported schemas and malformed records are errors.

Kev development becomes `validation` and Kev test stays the locked `test`;
`calibration` is whole `group_id` groups carved from Kev train in a seeded hash
order. Record, group, state-text (exact and normalized) and rendered-text overlap
across splits is an error, as are too few groups or a single label class in any
split and task. Kev's `CodeReviewer` ids repeat, so `row_sha256` identifies
records. Each prompt is `State:`, the state, `Question:` with Kev's instruction,
`Options:` lines `A) Yes` / `B) No` in a per-row deterministic order, and a final
`Decision:` cue. Option endpoints include each line's newline; `metadata.target`
indexes the gold option, and `metadata` also keeps the option mapping, native
label, Kev id, `group_id` and row hash. The report records hashes, attribution,
source and per-repository licences, counts and caveats. `--pilot-rows N` keeps
whole groups up to N rows per split and task and reports the rest; a pilot only
exercises the pipeline and says nothing about quality.

The Kaggle command only stages a private offline job; it does not upload data
or submit training. Capture is local inference, not local training. The
trainer selects weights on validation, fits temperature only on calibration,
and reports held-out accuracy, NLL, Brier score and calibration error. Export is
an experimental F32 pointer head, not a `LoRA` or an installed runtime artifact.

A private offline CPU Kaggle pilot completed on 252 captured rows (64 train,
64 validation, 60 calibration, 64 test). Validation selected epoch 1 of 20;
validation accuracy was 48.44%. Calibrated test accuracy was 40/64 (62.5%), NLL
0.605155, Brier 0.426804 and ECE 0.106413, with temperature 14.01774. Reading the
exported head without `PyTorch` reproduced these metrics within 0.00001.
`CodeReviewer` scored 19/32 versus a train-majority baseline of 16/32;
`CommitPackFT` scored 21/32 versus 17/32. The code-review test has only four
groups, and there is no zero-shot Bonsai comparison. These observed pilot test
rows are no longer an untouched evaluation set for future model selection.

A full evaluation then ran once, as a private offline CPU job, on a single
frozen split of the same two audited tasks: 2,694 train, 299 validation, 306
calibration and 234 test rows (3,533 captured rows). The 64 observed pilot test
rows were excluded, and staging checked that no remaining test row shares a
group with them. The configuration was fixed before the run (40 epochs,
learning rate 3e-5, weight decay 0.01, batch 64, seed 0, head dimension 256
over the 5,120-wide frozen features), informed only by the pilot's diverging
train/validation history at learning rate 1e-3, never by test. Validation NLL
selected epoch 26, and temperature 1.72146 was fitted on calibration only (not
at its bound). The test split was scored once by the job and not used for any
selection.

| Split | Rows | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2,694 | 92.95% | 0.1971 | 0.1186 | 0.0572 |
| Validation | 299 | 83.61% | 0.3168 | 0.2165 | 0.0365 |
| Calibration, after temperature | 306 | 82.35% | 0.3554 | 0.2399 | 0.0264 |
| Test, uncalibrated | 234 | 79.91% | 0.4157 | 0.2818 | 0.0886 |
| Test, calibrated | 234 | 187/234 = 79.91% | 0.3750 | 0.2608 | 0.0523 |

By task on test, `CodeReviewer` `needs_comment` scored 75/118 (63.6%) against a
train-majority baseline of 59/118 (50.0%), and `CommitPackFT` `message_match`
112/116 (96.6%) against 59/116 (50.9%). The pilot head scored 131/234 (56.0%)
on the same test rows. Most of the margin comes from `CommitPackFT`, whose
labels hold by construction; code-review judgment is only modestly above the
baseline. An independent local re-scoring, recomputing logits in pure Python
from the exported head and refitting the temperature, reproduced every metric
within 2e-8 and checked the head, trainer-source and per-sample feature
hashes, the frozen configuration, the sample counts and the temperature refit.
Provenance hashes (SHA256):

| Item | SHA256 |
| --- | --- |
| Exported head, `head.safetensors` (F32 pointer head) | `b3227ce477a2b010fc1c14a335653562f16ac27a90e04fd13d65893801ae7539` |
| Captured `features.jsonl` | `5084534aa23705d5001b9f4467d69dfc8a0402f0e05022eba9d78c21c82eb0db` |
| Trainer source | `e2bfdacf6a5b4da2764eac898b70f91c09999d9dc2ca490097bdd160f8eaf973` |

The recipe is the pipeline above without `--pilot-rows`; the 64 pilot test
rows were then dropped at staging (a one-off filter, not an option of
`tools/judgment_prepare.py`), and `tools/judgment_train.py` at the hash above
trained on CPU (Python 3.13, `PyTorch` 2.11 CPU) after its 13 self-tests passed.
It is one seed and one in-domain dataset with balanced, sampled labels, and
there is still no zero-shot Bonsai comparison. The scores describe
this dataset's held-out split only.

This head is **not installed**, is not loaded by the engine or server, and is
neither a Decisions implementation nor an MTP upgrade: it does not touch
speculative decoding. No production judgment head is supplied. Frozen Bonsai
features may not match Kev's adapter-trained representations; Kev's labels are
balanced by sampling, not natural rates. Larger held-out and out-of-domain
results, prompt/capture orchestration and refusal/confidence semantics remain
prerequisites for Decisions. The existing MTP rollout data has no judgment
labels and is not silently reused for this task.

`judgment::JudgmentHead::open(path)` explicitly loads the experimental F32
artifact; nothing discovers or activates it automatically. Its CPU-only
`score(&features)` returns temperature-scaled logits and softmax probabilities
in option order. Supply output-normalized, unrotated hidden states as flat
position-major rows: at least two option rows, then the decision row. Check
`width()` against the capture model, and round captured values through FP16
to match the training representation. The loader checks format metadata,
dimensions, tensor byte ranges and finite values. Native scoring matched an
independent Python reference on two real captured examples within 1e-12.
This validates the scoring formula, not judgment quality on new tasks or
the calibration of returned probabilities outside the evaluation dataset.

Acceptance should exercise native calls as well as HTTP: a complete read/edit/
tool-result turn, malformed and interrupted arguments that cannot execute,
cancellation during prefill and decode, recovery without duplicate side effects,
and context/output overflow without silent truncation. The checkpoint's 262,144
token limit, the memory policy's admitted context, and measured long-context
coding quality are separate quantities. A maximum-length attention-kernel test
does not validate end-to-end repository reasoning at that length.
