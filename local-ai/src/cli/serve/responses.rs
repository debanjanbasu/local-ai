//! `POST /v1/responses`: the text and function-calling subset of the `OpenAI`
//! Responses API.
//!
//! Schema source: <https://developers.openai.com/api/reference/resources/responses/methods/create>
//! and its streaming-events page. What is implemented is implemented to that
//! shape. A known option this server cannot honour is refused with a 400
//! naming it, because a request answered as if an option were honoured when
//! it was not is worse than one that fails, and so is a malformed value of a
//! member it reads. Members it does not know are ignored, as on Chat
//! Completions, so vendor extensions and newer fields do not break clients:
//!
//! - Storage is opt-in. Without `--response-store` nothing is kept: `store`
//!   is reported as `false`, and `store: true` and `previous_response_id`
//!   fail. With it, `store` takes `OpenAI`'s default of `true`, completed and
//!   incomplete responses are persisted (see [`super::store`]) before their
//!   terminal event is sent, and `previous_response_id` continues one. Only
//!   conversation items carry over; `instructions`, tools and sampling
//!   controls are this request's alone. `background: true` is stored in the
//!   response store when `store` is `true` (the default once a store exists);
//!   with `store: false`, or with `store` omitted on a server without one, it
//!   is kept only temporarily, in a private store of this process, until
//!   [`super::background::TEMPORARY_RETENTION`] after it ends, and reports
//!   `store: false` (see [`super::background::Temporary`]). With
//!   `stream: true` its events are journaled so a stream can be resumed (see
//!   [`super::background`] and [`super::resume`]).
//!   `item_reference` inputs still fail. Clients may instead carry history
//!   explicitly in `input`, which is supported in full for messages,
//!   reasoning, `function_call` and `function_call_output` items.
//! - `conversation` (an ID or `{"id"}`, never with `previous_response_id`)
//!   needs the durable conversation store. The conversation's items, read as
//!   one snapshot, precede this request's input; the Response echoes
//!   `conversation: {"id"}`. A completed or incomplete Response appends this
//!   request's new input and its output to the conversation exactly once,
//!   keyed by the response ID and only if the conversation is still at the
//!   version the prompt was built from, before its terminal state is
//!   reported; if that append fails the Response fails instead. Failed and
//!   cancelled Responses append nothing.
//! - Function, `custom`, `local_shell` and `apply_patch` tools are returned
//!   for the client to run; with server permission, `shell` and the
//!   `local_fs_*`/`local_web_fetch` functions run here, in the loop of
//!   [`super::tool_loop`] (see `docs/TOOLS.md`), and so does `file_search`
//!   over this server's vector stores, with no permission needed (see
//!   [`super::file_search`]; `include: ["file_search_call.results"]` lists
//!   its results). Other built-in and MCP tools need hosted services this
//!   server lacks. `tool_choice` (`auto`, `none`, `required` or
//!   `{"type":"function","name"}`), `parallel_tool_calls` and a tool's
//!   `strict` map onto the engine's native tool grammar, which enforces them
//!   on every sampled token, and are echoed back as accepted (`strict` as
//!   `false` when omitted: this server enforces arguments only on request).
//! - `text.format` is `text`, `json_object`, or the flat `json_schema`
//!   (`name`, `schema`, optional `description` and `strict`), parsed strictly
//!   and echoed back as accepted. The engine compiles the schema before
//!   queueing (an invalid or unsupported one is a 400) and enforces it on
//!   every answer token whatever `strict` says; reasoning stays unconstrained.
//!   With tools, the answer is either a call or (unless a call is
//!   required) the final answer in that format, enforced by one grammar.
//!   An answer the format did not complete is never `completed`: a token
//!   limit is `incomplete` (`max_output_tokens`), and an end of turn during
//!   reasoning, before any answer, is `failed`.
//! - `text.verbosity` `low` and `high` add an instruction at the end of the
//!   system turn (see [`super::request::add_verbosity`]), which the model has
//!   no other control for; the Response echoes the level.
//! - Text only: image, file and audio parts fail.
//! - `reasoning.effort` runs at the checkpoint template's `low`, `medium` or
//!   `xhigh` (`minimal` at `low`, `high` at `xhigh`; the Response reports the
//!   level that ran) or not at all (`none`). `reasoning.context`
//!   `current_turn` renders earlier turns without their reasoning, as the
//!   template's `preserve_thinking: false` does; `auto` and `all_turns` keep
//!   it (the template's default). Raw reasoning is returned as `reasoning_text` content, never as a
//!   summary. `reasoning.summary` (`auto`, `concise` or `detailed`) runs a
//!   second, short generation of the same model after the first ends, which
//!   summarizes the raw reasoning into the reasoning item's `summary` with
//!   the `response.reasoning_summary_*` events; the item's `output_item.done`
//!   waits for it. A background job journals each piece as it is generated.
//! - Compaction (`POST /v1/responses/compact`, and `context_management`
//!   compaction here) summarizes older items with the same model and keeps
//!   the rest verbatim (policy: [`local_services::compaction`]; prompts:
//!   [`local_engine::compaction`]). The `compaction` item is sealed with the
//!   `--reasoning-key` cipher, so both need one. A prompt is rendered from the
//!   latest compaction item onward: the summary, then the items it kept, then
//!   the items after it; anything before it is skipped. With
//!   `context_management`, a prompt that reaches `compact_threshold` (null or
//!   absent: the server's default, 32,000 tokens), or that leaves less than
//!   `max_output_tokens` of the context, is compacted before
//!   generation and the compaction item is the Response's first output item;
//!   for `background`, before the job is admitted, and a resumed job reuses
//!   the item its record holds. Both extra generations' tokens are added to
//!   `usage` (see [`Spent`](parse::Spent)). An input ending with a `compaction_trigger`
//!   item (Codex's remote compaction) is a compaction turn: the history is
//!   compacted whatever its length and the Response's only output item is
//!   the compaction item, with the summary generation's `usage`; nothing
//!   else is generated, and the trigger itself is never kept as history.

mod history;
mod parse;
mod state;

pub(super) use self::history::{History, compaction_resource, prepare_compact};
pub(super) use self::parse::{
    Echo, Generate, PreparedResponses, Summarizer, prepare_input_tokens_conversing,
    prepare_responses_conversing, prepare_resumed,
};
#[cfg(test)]
pub(super) use self::parse::{
    prepare_input_tokens, prepare_responses, prepare_responses_retaining, prepare_responses_with,
};
pub(super) use self::state::{ResponsesState, sse_frame};

#[cfg(test)]
#[allow(clippy::expect_used)]
#[path = "conversation_response_tests.rs"]
mod conversation_response_tests;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
#[path = "compaction_tests.rs"]
pub(super) mod compaction_tests;
