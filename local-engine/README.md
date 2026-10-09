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

Today `ChatMessage` supports system/user/assistant text and prior reasoning;
`Event` exposes text, reasoning, token IDs and completion/error events. Structured
tool definitions, tool-call events, call/result replay, Responses and Decisions
are **not implemented**. Text streaming alone is not coding-agent compatibility.

The next extension belongs in these native request/event contracts first, with
HTTP adapters translating them. Tool calls need stable IDs, names and raw JSON
argument deltas, followed by authoritative final parsing. Partial JSON is not an
executable call. Keep tool execution out of the engine; in particular, a retry
must not repeat a partially successful batch's side effects. Do not change the
pinned checkpoint template merely to match another Qwen model's conventions.

`OpenAI`'s [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
uses `input` and named question/answer arrays. At the pinned revision, Oh My Pi's
[`openrouter-decisions` adapter](https://github.com/can1357/oh-my-pi/blob/dde3fc44ed16d3bbec292893c7a902e92d6ce00e/packages/ai/src/judgment/typesafe.ts)
uses the older System One `state`, keyed questions/answers and `noul` predicate
format. They require distinct adapters to any future native judgment interface.
A judgment head is separate from the speculative MTP head: MTP accelerates
generation and does not train classification, tool use or confidence calibration.

Acceptance should exercise native calls as well as HTTP: a complete read/edit/
tool-result turn, malformed and interrupted arguments that cannot execute,
cancellation during prefill and decode, recovery without duplicate side effects,
and context/output overflow without silent truncation. The checkpoint's 262,144
token limit, the memory policy's admitted context, and measured long-context
coding quality are separate quantities. A maximum-length attention-kernel test
does not validate end-to-end repository reasoning at that length.
