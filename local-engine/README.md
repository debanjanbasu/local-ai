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

Generation is chat-shaped (`ChatRequest`): there is no raw-prompt completion
API, as the server built on it serves only current `OpenAI` APIs (Chat
Completions and Responses, no legacy Completions; see
[supported APIs](../docs/BONSAI.md#supported-openai-apis)). The
`local-ai bonsai --raw` command prefills raw text for measurements.

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

The engine keeps no conversation or file records (only its prompt caches). A
harness that wants durable conversations, files, or bounded read-only
repository search (ranked, exact or regex) can call the model-free [`local-services`](../local-services/README.md) crate, which the
server also uses. Deciding when and what to compact remains the harness's job;
the engine only phrases and renders the summary generation (see
[Compaction and reasoning summaries](#compaction-and-reasoning-summaries)).

### Compaction and reasoning summaries

Both are ordinary generations of the one loaded checkpoint, and the summary
is the generation's visible content, never its raw reasoning.
`local_engine::compaction` owns only the model-facing half:

- `compaction_request(&conversation)` is the conversation's own request (its
  messages, tools, session and reasoning) followed by a fixed checkpoint
  instruction asking for goals, decisions, exact identifiers and results,
  tool outcomes and every unresolved item. Its prompt therefore extends the
  one the conversation's last turn prefilled, and the prompt cache serves it
  from that turn's snapshot; a summary of a shorter prefix, or rendered under
  another system turn, would prefill the whole history again, because exact
  prefixes are kept only at snapshot boundaries. Tools are rendered with
  `ToolChoice::None`, so no call can be made. The summary never reasons: for
  a conversation that reasons it keeps the effort's instruction in the system
  turn (`ChatRequest::reasoning_instruction`) and closes the reasoning block
  in the prompt, so it still shares the conversation's prefix and answers at
  once, in up to 2,048 tokens.
- `summary_message(summary)` is how a compacted summary is rendered back: one
  `user` message under a fixed `[Context compacted]` header, in place of the
  turns it replaced.
- `reasoning_summary_request(reasoning, SummaryDetail)` quotes a generation's
  raw reasoning (thinking off) and asks for an `auto` (one paragraph, at most 320 tokens),
  `concise` (one to three sentences, 160) or `detailed` (900) summary.

The session half is model-free, in `local_services::compaction`: `plan`
chooses which items to summarize and which to keep verbatim, `should_compact`
decides from a token count, an optional threshold, the output reservation and
the context, and a prompt is rendered from the latest `compaction` item, whose
`Compacted` payload carries the summary and the kept items. A cut never falls
between a `function_call` and its `function_call_output` or inside an
assistant turn, so pending calls and everything after them are kept; the
leading system/developer messages and the latest user message are always kept verbatim;
the newest items are kept up to a token budget measured with
`count_chat_tokens`. A harness counts, plans, runs `compaction_request` on the
engine, collects `Event::Content`, and replaces the summarized items.

Cost and risk: a compaction pass prefills only what follows the
conversation's last turn when that turn left its prompt in the cache (the
whole history otherwise), decodes up to 2,048 summary tokens (plus its
reasoning, for a conversation that reasons), and the compacted prompt that
follows is prefilled cold; a reasoning summary
prefills the reasoning once more and decodes up to its budget, after the
answer. Both are queued like any request. A summary is lossy: details the model
leaves out (an exact error string, a file it read) are gone from the prompt,
and a wrong summary is believed by later turns. Only the kept items are exact,
so keep the budget generous for work that depends on verbatim history, and
keep the original items (the server's stores do) when they may be needed again.

### Counting prompt tokens

`Engine::count_chat_tokens(&ChatRequest)` returns the exact number of prompt
tokens generation would prefill. The request is validated,
rendered (messages, replayed tool calls and results, tool definitions and
`thinking`) and tokenized, special tokens included, by the same code as
`chat_with`, so the count equals `prompt_tokens` in the generation statistics,
cached prefix included, and invalid prompt content fails with the error
generation would return. `max_tokens`, `sampling`, `session`, `response_format`, `stop`, `tool_choice`,
`parallel_tool_calls` and tool `strict` are ignored (none of them changes the
prompt); no schema or tool grammar is compiled and the count is not checked
against the context window.
Counting is CPU-only and submits no GPU work.
`EngineHandle` offers the same method against the worker's shared
tokenizer: it runs on the calling thread, queues no job and never waits for
running generations, so it works while the queue is full. Async callers should
run long prompts on a blocking pool.

## Text embeddings

`embedding::EmbeddingModel` runs the optional `EmbeddingGemma` 2 text encoder
natively on Metal (BF16 weights, F32 activations; install and measurements in
[docs/EMBEDDINGS.md](../docs/EMBEDDINGS.md)). Inputs are embedded as given;
`EmbeddingPrompt` formats the model card's query and document prompts, and
`truncate_normalized` gives the Matryoshka widths 512, 256 and 128.

```rust,no_run
use std::path::Path;
use local_engine::embedding::{EmbeddingModel, EmbeddingPrompt, installed_model};

let dir = installed_model(Path::new("models")).expect("model not installed");
let mut model = EmbeddingModel::open(&dir)?;
let query = EmbeddingPrompt::Query { task: "search result" }.format("how do hash maps resize?");
let document = EmbeddingPrompt::Document { title: None }.format("Open addressing doubles the table…");
let vectors = model.embed_texts(&[&query, &document], &mut || Ok(()))?;
let similarity: f32 = vectors[0].iter().zip(&vectors[1]).map(|(a, b)| a * b).sum();
# let _ = similarity;
# Ok::<(), local_engine::Error>(())
```

The last argument is called between short GPU submissions: a server passes
`EngineHandle::activity()`'s `yield_to_decode` there so embedding work waits
for running Bonsai decode steps instead of delaying them.

With the optional vision artifact installed beside it, the same model embeds
images and video into the same space (preprocessing identical to the
reference processor; video files need `ffmpeg`/`ffprobe`):

```rust,no_run
# use std::path::Path;
# use local_engine::embedding::{EmbeddingModel, installed_model};
use local_engine::embedding::{MediaPart, prepare_image};
use local_engine::embedding::vision::video::decode_video_file;

# let dir = installed_model(Path::new("models")).expect("model not installed");
# let mut model = EmbeddingModel::open(&dir)?;
let image = prepare_image(&std::fs::read("photo.jpg")?)?;
let photo = model.embed_image(&image, &mut || Ok(()))?;
let clip = model.embed_video(&decode_video_file(Path::new("clip.mp4"))?, &mut || Ok(()))?;
let mixed = model.embed_multimodal(&[vec![MediaPart::Text("Trail shoes. "), MediaPart::Image(&image)]], &mut || Ok(()))?;
# let _ = (photo, clip, mixed);
# Ok::<(), local_engine::Error>(())
```

## Stop sequences

`ChatRequest::stop` lists sequences that end the visible answer, as `OpenAI`'s
Chat `stop` does. Only answer text is matched: reasoning (up to `</think>`)
and tool-call markup and arguments never are, but the text before, between
and after calls is. At the first match generation stops, the content ends
just before the sequence, and `Stats::stop_reason` is `StopReason::Stopped`.
Content that could still be the start of a sequence is held back until it
cannot, so no part of a sequence is ever emitted, even when it spans
several tokens; held text is released before a tool call and at the end.
A call made before the match stands; a match before a call means the call
is never made. A JSON answer under a `response_format` is cut short like any
other text (`response_format_complete` is `Some(false)`). Empty sequences
never match. The engine stops at a token boundary with its state intact, as
at a token limit, so the prompt cache keeps the turn.

## Structured output

`ChatRequest::response_format` takes a `ResponseFormat`: `Text` (default), `JsonObject` (`OpenAI` `json_object`) or
`JsonSchema(schema)` (the `schema` of `OpenAI` `json_schema`; `name`,
`description` and `strict` are not needed by the engine, and the schema is
always enforced). The format is compiled with
[llguidance](https://github.com/guidance-ai/llguidance) before the request is
queued (`Engine::chat_with`, `EngineHandle::chat`), so
invalid or unsupported schemas fail with `Error::InvalidArgument` before any
generation. Every target token selection (first token, plain decode, n-gram
and MTP verification, batched rows) is masked to what the grammar allows;
drafts stay unconstrained proposals that the masked target verifies.

- With `thinking`, reasoning is unconstrained and the format applies from the
  token after `</think>`; without it, from the first generated token, with no whitespace before the document.
- End-of-sequence is only selectable once the document is complete.
  `GenerationStats::response_format_complete` is `Some(true)` only then;
  a token limit or cancellation yields `Some(false)` and incomplete JSON.
  With `thinking`, the model may also end its turn before `</think>`, which
  is unconstrained; that `Finished` carries `Some(false)` and no answer at
  all, so check the flag rather than trusting a stop at end-of-sequence
  (`local-ai` reports it as a failure). This flag is `None` for text formats.
  `tool_constraints_complete` separately reports tool-policy completion;
  optional calls may end during reasoning without a call. With tools and a
  format, a completed call also satisfies the combined answer contract.
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
- A format can be combined with `tools`: the native tool grammar (see
  [constrained tool calling](#constrained-tool-calling)) then makes the
  answer either tool calls or, unless a call is required, one document in
  the format, both enforced by one grammar.
- The first constrained request builds the tokenizer's grammar tables once
  (about 0.5 s on CPU); later schemas compile in well under a millisecond.
- The mask is applied to the logits a 32-id mask word at a time, skipping
  fully allowed words and filling fully forbidden ones. An isolated CPU
  microbenchmark of that step measured roughly 2.5-3x faster than the
  previous per-id loop; its effect on end-to-end constrained decoding on the
  GPU has not been measured.

## Native tool calling

`ChatRequest::tools` takes `ToolDefinition { name, description, parameters,
strict }`, where `parameters` is a JSON Schema for the arguments object and
`strict` opts into enforcing it during generation (below). With tools
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
use local_engine::{ChatMessage, ChatRequest, Engine, Event, Sampling, ToolChoice, ToolDefinition};
use std::ops::ControlFlow;

let mut engine = Engine::open()?;
let request = ChatRequest {
    messages: vec![ChatMessage { role: "user".into(), content: "Weather in Paris?".into(), ..ChatMessage::default() }],
    max_tokens: 512,
    sampling: Sampling::default(),
    thinking: true,
    effort: local_engine::ReasoningEffort::default(),
    reasoning_instruction: false,
    preserve_thinking: true,
    session: None,
    tools: vec![ToolDefinition {
        name: "get_weather".into(),
        description: Some("Current weather for a city.".into()),
        parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        strict: false,
    }],
    tool_choice: ToolChoice::Auto,
    parallel_tool_calls: true,
    response_format: local_engine::ResponseFormat::Text,
    stop: Vec::new(),
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
  whole `<tool_call>` block has been generated and parsed, and, for a strict
  tool, validated against its schema. A non-strict tool's schema guides the
  model but is not enforced: its call is emitted with the arguments as
  generated, for the harness or client to validate and answer with an error
  result, and so is a call to a tool the request does not declare (the
  harness answers it as unknown; Codex's instructions name `apply_patch`
  even when its tool set has none). Ordinary text still streams; only a possible `<tool_call>`
  prefix and trailing whitespace are held back.
- While a call is generated, `Event::ToolCallDelta` reports the part of its
  arguments that can no longer change, once the call names a configured tool:
  in order, a call's `arguments` pieces spell a prefix of the final
  `ToolCall::arguments` serialized as compact JSON (keys in the order the model
  wrote them), under the same ID. A string parameter whose schema accepts any
  string (and a strict tool's raw string) streams as it is written, with its
  raw text in `text`; any other value arrives whole when it ends. Deltas are
  progress only: the call is still validated at `</tool_call>`, and a call
  that fails or is cut short has no `Event::ToolCall`.
- A malformed, strict-schema-violating or EOS-truncated call ends the generation:
  `EventStream` yields `Event::Error` and no `Event::Finished`; `Engine::chat_with`
  returns `Err`. Earlier valid calls in the same answer have already been emitted.
  Deltas of the failed call may have been emitted too.
- A token limit or cancellation instead discards an unfinished call (beyond the
  deltas already emitted) and preserves the original stop reason and
  statistics. Previously completed calls remain; no partial call can execute.
- IDs (`call_…`) are generated by the engine, identical in the stream and in
  `ChatOutput::tool_calls`, and unique across generations. They are not shown to
  the model.
- Replay a turn as an `assistant` message with `tool_calls` (and its
  `reasoning_content`, which the template keeps in history), followed by one
  `tool` message per call with `tool_call_id`. Consecutive results render as one
  user turn of `<tool_response>` blocks, reordered by ID into call order because
  Bonsai's template has no model-visible IDs. Duplicate IDs, unanswered calls,
  unmatched or repeated results, arguments that violate a strict tool's schema
  and invalid schemas are rejected with `Error::InvalidArgument` before
  generation; a non-strict call replays as it was generated, and so does a
  call to a tool the request no longer declares.
- `developer` is accepted like `system` and fills the template's single
  system slot. Consecutive leading `system`/`developer` messages are joined
  there in order, separated by a blank line; one renders exactly as before,
  and an instruction message after the first non-instruction one is refused.
- With the defaults (`ToolChoice::Auto`, `parallel_tool_calls: true`, no
  strict tool, `ResponseFormat::Text`) generation is unconstrained and the
  model decides whether to call. Any other setting is enforced by the
  native tool grammar described below.
- The engine never executes a tool. Execution, permissions and side effects
  belong to the harness.

Schemas are compiled once per request with the `jsonschema` crate, defaulting
to draft 2020-12. Assertions such as `pattern`, `uniqueItems` and
`dependentRequired` are enforced, along with known formats. Document-local
references work; network/file retrieval and unknown dialects are rejected.
Annotations such as `title`, `description` and `default` do not constrain values.
Ambiguous raw parameter text remains a string when its declared schema accepts
strings, including nullable strings; the harness should avoid ambiguous unions.
Object keys in rendered schemas and replayed arguments keep the client's
order, as the template's `tojson` prints them under `transformers`, vLLM and
llama.cpp (the engine enables `serde_json`'s `preserve_order`).

### Constrained tool calling

`ChatRequest::tool_choice` (`ToolChoice::{Auto, None, Required,
Function(name)}`), `ChatRequest::parallel_tool_calls` and
`ToolDefinition::strict` are enforced by the sampler, not requested in the
prompt: the prompt, and therefore the prompt-token count, is identical for
every choice, and the tools are rendered even for `None`. When any of them
differs from the defaults, or a non-text `response_format` accompanies tools,
the request compiles one llguidance grammar (a Lark root over the
checkpoint's own call layout, plus JSON-schema subgrammars) before it is
queued, and every target token selection is masked by it exactly as for
[structured output](#structured-output), including batched rows and lookup
and MTP verification; drafts stay unconstrained proposals. With `thinking`
the grammar starts after `</think>`.

| Setting | What the grammar allows in the answer |
| --- | --- |
| `Auto` | free text without `<tool_call>`, or text followed by calls |
| `None` | free text that can never contain `<tool_call>` |
| `Required` | up to four newlines, then at least one call; end-of-sequence is not selectable before a complete call, and with `thinking` the reasoning phase cannot end the request either |
| `Function(name)` | as `Required`, but only calls to `name` |
| `parallel_tool_calls: false` | at most one call, followed only by whitespace |
| non-text `response_format` | `Auto`: one document in the format, or calls; `None`: only the document; `Required`/`Function`: only calls |

After a call only whitespace and (with parallel calls) further calls are
allowed. `<tool_call>` and `</tool_call>` are single special tokens in the
pinned checkpoint; the grammar accepts either that token or the literal text,
and no other special token. `Required` with no tools, or `Function` naming an
undeclared tool, is `Error::InvalidArgument` before queueing.

A non-strict tool inside a grammar only has its call framing enforced; its
arguments are still parsed and validated once the call is complete, as
without a grammar. A `strict: true` tool has its arguments enforced token by
token in the template's layout: declared parameters in declared order, every
required one exactly once, optional ones at most once. Its schema must be a
closed object (`"additionalProperties": false`) that the engine can enforce
exactly:

- A string parameter (`"type": "string"`, optionally nullable, or a string
  `enum`/`const`) is raw text constrained by `enum`/`const`, `pattern`
  (ECMA-262 translated where exact; lookarounds, backreferences, word
  boundaries, inline flags and inner anchors are refused), `minLength` and
  `maxLength` (up to 16,384). No raw value may contain `</parameter>` or
  `</tool_call>`.
- Any other parameter is compact JSON (`", "` and `": "`) from its own schema,
  compiled by llguidance with the same strictness as a response format.
- Document-local `$ref`s into `$defs`/`definitions` are followed. Anything
  else, such as other top-level keywords, a `minProperties`/`maxProperties`
  the fixed layout does not provably satisfy, mixed string/non-string type
  lists or a parameter schema llguidance refuses, fails the request before
  generation with the reason; it is never enforced partially.

A strict call is read back by its exact layout (it ends at
`\n</function>\n</tool_call>`, so a JSON string holding `</tool_call>` does
not cut it short) and then validated against the whole schema like every call.
The grammar guarantees the shape of what is generated, not that the model
calls the right tool with sensible values. A token limit still applies: a
call it cuts off is discarded, and a limit reached before a required call
starts (for example during reasoning) finishes without one. Check the stop
reason and `tool_constraints_complete` before assuming a required call exists.

Unit tests cover the grammar against synthetic vocabularies and, CPU-only,
the real tokenizer's special tags (an ignored test). Real-model HTTP checks
with MTP enabled and disabled cover forced/named/none choices, strict UTF-8,
numeric and nested JSON arguments, parallel limits and format/tool branches.

`local-ai` exposes these native calls through Chat Completions and a
text/function subset of Responses, including `tool_choice`,
`parallel_tool_calls` and tool `strict`. Responses are stored by default
(stateless with `--no-response-store`); background Responses work either
way (with `store: false` they are kept only temporarily). The server maps Chat
`response_format` and Responses `text.format` onto `ResponseFormat`. See
[server API](../docs/BONSAI.md#server-api) for storage, background streams,
structured output, encrypted reasoning replay, token counting, the
decisions route and compatibility limits. Keep tool execution out
of the engine; in particular, a retry must not repeat a partially successful
batch's side effects. Do not change the pinned checkpoint template merely to
match another Qwen model's conventions.

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
      reasoningEffortMap: {minimal: low, low: low, medium: medium, high: xhigh, max: xhigh}
    models:
      - id: ternary-bonsai-2-27b # the server's id (GET /v1/models)
        name: Ternary Bonsai 2 27B (local-ai)
        reasoning: true
        thinking: {mode: effort, efforts: [low, medium, xhigh], defaultLevel: xhigh, requiresEffort: false}
        input: [text]
        contextWindow: 65536 # at most context_length from GET /v1/models
        maxTokens: 8192
        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
```

Oh My Pi compacts when a prompt exceeds `compaction.thresholdTokens`; set it
to 32,000 (provisional until the
[threshold sweep](../docs/BONSAI.md#compaction-threshold) finishes) with `omp config set compaction.thresholdTokens 32000`.

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
- `supportsReasoningSummary: false` stops `reasoning.summary: "auto"`. The
  server now honours it, but by a second generation after every reasoning
  answer (see [Compaction and reasoning summaries](#compaction-and-reasoning-summaries)),
  so keep it off unless the summaries are wanted.
- `reasoningDisableMode: none-effort` makes `--thinking off` send
  `reasoning.effort: "none"`. The default sends the lowest listed effort,
  `low`, so turning thinking off would silently keep it on.
- `efforts: [low, medium, xhigh]` and `reasoningEffortMap` reflect the
  checkpoint template's three reasoning levels: each Oh My Pi level becomes
  the nearest one, so a fixed internal effort such as `max` never reaches the
  server as an unsupported value. (Earlier versions served only `xhigh`, and
  the live check below ran with every level mapped to it.)
  `requiresEffort: false` declares that off is accepted, so it is never
  clamped up to `xhigh`; auto-detection gave the same result in the check below.
- `supportsStrictMode: false` keeps `strict` out of tool definitions. It is
  already off for hosts other than `OpenAI` and a few known providers. The
  server now accepts `strict: true`, but then requires every such tool's
  schema to be a closed object it can enforce exactly and refuses the whole
  request otherwise; Oh My Pi's tool schemas have not been audited for that,
  so keep it off.
- `supportsForcedToolChoice: false` turns Oh My Pi's forced tool choices into
  `auto`, so the model may answer without calling the tool. The server now
  enforces `required` and named tool choices, but this configuration was
  recorded when they were refused; enabling forced choices has not been
  re-checked against Oh My Pi.

Oh My Pi already sends `store: false`, resends earlier turns in `input`, puts
the system prompt in `instructions`, and leaves out the `developer` role and
`stream_options` for non-OpenAI hosts. With `input: [text]`, images in tool
results become text placeholders. Presence and repetition penalties, `top_k`
and `min_p` configured in Oh My Pi are honoured (it sends no frequency
penalty to a Responses endpoint).
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
format. They require distinct adapters to the native judgment interface
described [below](#native-decisions), which is not either of
them. A judgment head is separate from the speculative MTP head: MTP accelerates
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

This head is **not installed** or discovered: the engine loads it only when
explicitly given its path (see [native decisions](#native-decisions)). It is
not an
implementation of `OpenAI` Decisions and not an MTP upgrade: it does not
touch speculative decoding. No production judgment head is supplied. Frozen Bonsai
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
dimensions, tensor byte ranges and finite values. It also keeps the
artifact's optional `renderer` metadata (`renderer()`; `None` when absent).
`--development` exports record it when every feature row agrees; the
existing devtools-v1 head, the `models/decisions-v2` head and default-mode
exports do not. The CPU scorer accepts any declared renderer. Native scoring matched an
independent Python reference on two real captured examples within 1e-12.
This validates the scoring formula, not judgment quality on new tasks or
the calibration of returned probabilities outside the evaluation dataset.

#### Kev hard-v1 preparation and development-only training

`tools/judgment_prepare.py --dataset hard-v1` adapts Kev's synthetic
[`hard-v1`](https://github.com/jaredpalmer/kev/tree/62c91838b9a6adc5b386cbeae8ed73daa36ce220/evals/hard-v1)
suite at a pinned commit and manifest digest. Kev keeps `train.jsonl` out of
git, so it must be regenerated byte-identically by Kev's own generator, which
needs the pinned Qwen3.5-4B-Base tokenizer. Only train and development are
read; the test partition (template 5) is never read, rendered or scored.
Splits follow the surface template, so every evaluation split measures
transfer to an unseen phrasing: train templates 0-2 train, template 3
calibrates, development (template 4) validates. Kev's programmatic labels are
kept unchanged: predicate questions become Yes/No, choice questions three to
six options in Kev's order, score questions six ordered levels (2-6 options in
all). `long_policy` is excluded in this round for capture cost
(`--include-long-policy` keeps it), and `--no-option-descriptions` renders
named options without descriptions. The default devtools-v1 output is
unchanged byte for byte.

```sh
python3 tools/kaggle_judgment_job.py --regenerate-hard-v1 \
  --kernel OWNER/kev-hard-v1 --output STAGING [--hf-secret HF_TOKEN]
```

stages, but does not submit, a private CPU Kaggle job. Unlike the training
job it needs internet: it downloads Kev at the pinned commit, checks the
generator, library, lockfile and licence digests, builds Kev's locked
environment with CPU `torch`, regenerates the partitions, compares every one
with the manifest and writes train and development only. The generator also
builds test in memory, because train is deduplicated against it; only its
digest comparison is kept and the job fails if `test.jsonl` is ever written.
A Hugging Face token is read from the named Kaggle secret if one is attached
and is never printed or saved. If every digest matches, the job runs
`judgment_prepare.py --dataset hard-v1` on the result and writes
`hard-v1-kev/`, `hard-v1-rows/` and a `hard-v1-job.json` summary; on any
mismatch it records the errors and prepares nothing. The private Kaggle
regeneration completed on 2026-10-10: all partition hashes matched the pinned
manifest, and local preparation reproduced the same 8,479 rows (5,716 train,
1,880 calibration, 883 validation). No locked test was written or scored.
Its labels are produced by family solvers over generated facts, so scores
are synthetic-task evidence, not natural code-review quality.

Both `judgment_train.py` and `kaggle_judgment_job.py` accept `--development`
for this three-split workflow. It refuses test rows, selects on validation,
calibrates on calibration, and reports validation metrics beside train-only
uniform and majority baselines, with family/type/option-count breakdowns.
Validation selected the checkpoint: those numbers are not held-out test
results. The default four-split devtools workflow is unchanged. Train on
Kaggle, after capturing features with the native model; staging submits nothing.

Three private offline CPU Kaggle runs completed on 2026-10-10, one version
each, with seeds 0/1/2. All used the recipe frozen before hard-v1 feature
capture: 40 epochs, learning rate 3e-5, weight decay 0.01, batch 64, feature
width 5,120 and pointer-head dimension 256. Only the head was trained; the
Bonsai trunk and speculative MTP head were unchanged. Checkpoints were selected
by validation NLL and temperature was fitted on the separate calibration split.

| Seed | Selected epoch | Temperature | Validation accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 1.514628 | 58.21% | 0.940797 | 0.526739 | 0.035773 |
| 1 | 4 | 1.446252 | 58.10% | 0.960556 | 0.527300 | 0.029395 |
| 2 | 4 | 1.263283 | 58.21% | 0.967057 | 0.534123 | 0.039192 |

All metrics above use the 883-row **checkpoint-selection split**, not a
held-out test. The train-position majority baseline scores 30.12% accuracy,
NLL 1.280109 and Brier 0.699157; uniform expected accuracy is 30.09%.
Calibration reduced validation NLL from 0.977–0.982 to the values above;
none of the temperatures hit the search bounds. Small aggregate ECE does not
establish useful confidence: even the majority baseline has ECE 0.011132.
The seeds agree on only about 77–80% of predictions. Temporal/numeric accuracy
is about 44%, and score-question accuracy varies from 37–51% on just 35 rows.
Training overfits after the selected checkpoints (epoch-40 validation NLL
1.51–1.54). `long_policy` remains excluded. This is evidence that frozen-trunk
features support learning these synthetic tasks, not evidence of general
coding quality, reliable refusal or production calibration. No head was
installed, published or promoted, and no locked test was scored.

The captured feature manifest SHA-256 is
`dabe32c4a14c40a1041599b113609b8a2e8bc8cccb1db05695792a209f115d63`;
all 8,479 feature hashes, split IDs and frozen training settings were verified
against the downloaded reports. Independent CPU recomputation of validation
metrics agreed within 7e-8. Exported head SHA-256 values, in seed order:

```text
24971f5fadb02fb0770f9beb372e5f0ef1bb650e4215064cafb5b901203ac39e
11bf6f7459a25e035c16a0d4dce2afa412286e3061bddf98679d513160731465
657cde86d4a8fa45fffbe9ffc4f18b087600cb4bc79cc5d82f7fcaf606079e2d
```

### Native decisions

`Engine::decide(&head, &DecisionRequest)` and
`EngineHandle::decide(Arc<JudgmentHead>, &DecisionRequest)` answer a typed
text question with an explicitly loaded head; `POST /v1/decisions` uses them
through `judgment::CalibratedHead` (see the
[server API](../docs/BONSAI.md#decisions)). `DecisionRequest { state,
question, kind }` is rendered exactly as `tools/judgment_prepare.py` renders
training rows (`State:`, `Question:`, lettered `Options:`, `Decision:`), the
renderer `judgment::DECISION_RENDERER` (`kev-devtools-v1-judgment-render.v1`).
`DecisionKind::Predicate` is `A) Yes` / `B) No`; `Choice` takes 2 to 26
distinct single-line values (strings or booleans) in the caller's order;
`Score` takes 1 to 26 ordered labels. The option and decision rows are
captured on the already loaded model (no second model) in 60-row blocks, the
`mtp-capture` default the training features used, rounded through FP16 and
scored on the CPU. `Engine::capture_decision` (the `judgment-capture` binary)
returns the same rendering and features before any head. All 8,479 prepared
hard-v1 development rows reproduced their training prompt bytes and
option/decision endpoints through this renderer and the real tokenizer, with
no embedded special-token ambiguity.

A `Decision` carries the head's probability per option, the argmax and its
value, for scores the probability-weighted mean level index, the prompt token
count and the scored `Decision:` row. It has no confidence, threshold or
refusal; `CalibratedHead::judge` adds those for its declared scope. A head
declaring a renderer other than `DECISION_RENDERER` is refused; one declaring
none, such as the `models/decisions-v2` head, is rendered with it.

Invalid requests, text that spells a special token, and a head whose width is
not the model's or whose declared renderer is unsupported fail before
any GPU work; a single-level score is answered
without capture. On a handle, preparation and tokenization run on the calling
thread and the decision takes one slot of the bounded FIFO queue (a full queue
is `Error::QueueFull`). It runs synchronously on the worker only once no
generation is active, and while it waits nothing queued behind it is admitted,
so a long capture delays later requests. The returned `PendingDecision` is a
`Future` (or blocks with `wait`); dropping or cancelling it skips a queued
decision or stops a running capture before its next block, resolving to
`Ok(None)`. Capture overwrites one sequence buffer set: the GPU prompt-cache
tier is kept when a free pooled set exists and cleared otherwise; host and
disk snapshots are unaffected.

Unit tests cover rendering, token endpoints, validation, queueing,
cancellation and score mapping with a synthetic head. Real-model native and
HTTP checks reproduce stored-feature probabilities bit-for-bit for four rows
covering both devtools tasks and both option orders, with and without MTP.
Capture/cancellation preserve later generation token IDs and cache
correctness. Capture measured about 115–130 tokens/s on the M4 Pro; later
generations wait behind it.

Acceptance should exercise native calls as well as HTTP: a complete read/edit/
tool-result turn, malformed and interrupted arguments that cannot execute,
cancellation during prefill and decode, recovery without duplicate side effects,
and context/output overflow without silent truncation. The checkpoint's 262,144
token limit, the memory policy's admitted context, and measured long-context
coding quality are separate quantities. A maximum-length attention-kernel test
does not validate end-to-end repository reasoning at that length.
