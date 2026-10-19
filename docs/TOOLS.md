# Tools on the Responses API

The model always chooses tools; Rust executes them, outside model execution.
The client runs some; this server runs `file_search` (no permission needed)
and, with explicit permission, the hosted tools. Tools are declared as
Responses `tools`, or Chat Completions `tools` with `tool_choice`; Chat's
deprecated `functions`/`function_call` are function tools too, answered in
their own `function_call` shape
([supported OpenAI APIs](BONSAI.md#supported-openai-apis)).

## Client-executed tools (no permission needed)

These are returned to the client, which runs them and sends the result back
in the next request (or with `previous_response_id`):

| request tool | output item | client answers with |
|---|---|---|
| `function` | `function_call` | `function_call_output` |
| `custom` (`format` text, or `grammar` with `syntax: "regex"` or `"lark"`) | `custom_tool_call` | `custom_tool_call_output` |
| `local_shell` | `local_shell_call` | `local_shell_call_output` |
| `apply_patch` | `apply_patch_call` | `apply_patch_call_output` |
| `namespace` of `function`/`custom` tools | `function_call` / `custom_tool_call` with `namespace` | `function_call_output` / `custom_tool_call_output` |

Each becomes a strict native function, so arguments are constrained while
they are generated; a `regex` grammar constrains the custom tool's `input`
the same way. A `lark` grammar is enforced exactly when its language is
regular: its rules and terminals are inlined into one regular expression (a
recursive rule is refused), with `|`, groups, `[ ]`/`?`/`*`/`+`/`~ n..m`,
string literals and `"a".."z"` ranges, `/regex/` terminals (flags `i`, `s`)
and `%import common.NAME` for the common terminals (`LF`, `CR`, `NEWLINE`,
`WS`, `WS_INLINE`, `DIGIT`, `HEXDIGIT`, `INT`, `SIGNED_INT`, `LETTER`,
`LCASE_LETTER`, `UCASE_LETTER`, `WORD`, `CNAME`). Each terminal may match any
string of its language (no longest-match lexing), aliases and priorities
only shape a parse tree and are ignored, and everything else (`%ignore`,
`%declare`, templates, other imports, regex anchors, lookarounds,
backreferences) is refused with the reason (400). Codex's freeform
`apply_patch` grammar is supported verbatim. A custom tool may be named like
a built-in tool type (Codex's `apply_patch`); declaring one name twice is
refused.

A `namespace` groups client `function` and `custom` tools: each member is
offered to the model as `NAMESPACE.NAME`, with the namespace description
before its own, and its calls come back with `"namespace"` and the member's
own `name` (Codex's sub-agent tools, `multi_agent_v1`, are such a namespace;
Codex runs them, not this server). Calls replayed with a `namespace` are read
back the same way. Namespaces cannot be combined with `conversation`.

`web_search` and `web_search_preview` are accepted and echoed but never
offered to the model: this server has no search provider, so the model is
not told it can search and no `web_search_call` is ever reported; the
declaration has no effect. The spec leaves calling a declared tool to the
model, so this is a model that never searches, not a fake search. Forcing a
search (`tool_choice` of a web search type, or `"required"` with no other
tool) and `include: ["web_search_call.action.sources"]` are refused (400).

`tool_choice` accepts
`{"type":"custom","name":...}`, `{"type":"shell"}` and
`{"type":"apply_patch"}` as well as the function forms.

## File search (no permission needed)

`{"type":"file_search","vector_store_ids":[...]}` runs here, natively, over
this server's [vector stores](BONSAI.md#native-services), in the same
server-side loop as the hosted tools below. It needs no flag: it only reads
the vector stores of the server's own data directory, which the same callers
can already search with `POST /v1/vector_stores/{id}/search`.

| member | accepted |
|---|---|
| `vector_store_ids` | 1 to 16 store IDs; each must exist and not be expired, or the request is refused (400) before generation |
| `max_num_results` | 1 to 50; **default 5** (results are chunks the model must read; see the cost below) |
| `ranking_options.ranker` | `auto` (default: `EmbeddingGemma` 2 cosine ranking when the embedding model is installed, BM25 otherwise) or `none` (BM25 only, no GPU work); `default-2024-11-15` is refused |
| `ranking_options.score_threshold` | 0 to 1, on the reported score |
| `ranking_options.hybrid_search` | `embedding_weight` and `text_weight`: weighted reciprocal rank fusion of the two rankings (`k` = 60); needs the embedding model and ranker `auto` |
| `filters` | the vector store attribute filters (`eq` ... `nin`, `and`/`or`) |

The model sees one strict function, `file_search(queries)`, taking 1 to 16
query strings. A call ranks the chunks of every declared store together
(`VectorIndex::search_stores`: each query embedded once with the loaded
model, one database snapshot, chunk text read from the database rather than
the source files) and the model reads the results as numbered passages,
`【n】 filename` and the chunk text, numbered across the Response's searches.
`tool_choice: {"type":"file_search"}` forces a search.

Each call is one `file_search_call` item: `queries`, `status` (`completed`,
or `failed` with the error the model read, or `incomplete` if the Response
ended first) and `results` (`file_id`, `filename`, `score`, `text`,
`attributes`), which are listed only with
`include: ["file_search_call.results"]` and are `null` otherwise. Streaming
emits `response.output_item.added` (`in_progress`),
`response.file_search_call.in_progress` and `.searching` when the model has
made the call, and `.completed` and `response.output_item.done` once it ran.

**Citations.** The model is asked to cite a passage by writing its marker
after the text it supports. A marker naming one of this Response's results
(`【1】`, `【1, 2】`, `【1†notes.md】`) is removed from the answer, with the
whitespace before it, and becomes a `file_citation` annotation (`file_id`,
`filename`, `index`) at that point of the final text, `index` counted in
Unicode scalar values, announced by `response.output_text.annotation.added`.
Anything else in lenticular brackets stays as written. Citations depend on
the model writing markers; a model that does not still answers from the
results, without annotations.

**Replay.** `file_search_call` items round-trip through `input`,
`previous_response_id` and conversations: the model reads each as its call
and the passages it returned, without markers (they are not the new
Response's to cite), or, when the item carries no `results`, a note that the
results are not part of the conversation, so it can search again.

**Cost.** A search itself is fast (measured below); reading the results is
not free: each result is a chunk, about 800 words with the default chunking,
roughly 1,000 tokens of prefill (about 8 s at 125 tok/s), which is why the
default is 5 results. A smaller `chunking_strategy` or `max_num_results`
answers sooner.

## Hosted tools (server permission required; default: none)

The server runs these itself, in the same loop: generation stops on the call,
the tool runs, its result is appended and generation continues, all within
one Response. A request declares them; the server must also have been
started with the matching permission, or the request is refused with a 400
naming the flag.

| request tool | needs | output items |
|---|---|---|
| `{"type":"shell"}` (environment omitted or `{"type":"local"}`) | `--tools-root DIR --tools-shell` | `shell_call`, `shell_call_output` |
| `{"type":"function","name":"local_fs_list"}` / `local_fs_read` / `local_fs_search` | `--tools-root DIR` | `function_call`, `function_call_output` |
| `{"type":"function","name":"local_fs_write"}` / `local_fs_edit` | `--tools-root DIR --tools-allow-write` | `function_call`, `function_call_output` |
| `{"type":"function","name":"local_web_fetch"}` | `--tools-fetch-domain DOMAIN` (repeatable) | `function_call`, `function_call_output` |

The `local_*` names are reserved: declare them without `parameters` (the
server supplies the schema). The OpenAI spec defines no hosted filesystem or
fetch tool (`file_search` is vector-store retrieval, `web_search` is search),
so these are reported as what they are, a function call plus the output the
server produced, not as a built-in tool they are not.

Bounds (flags, which also bound `file_search`): `--tools-max-iterations`
rounds of hosted calls per Response (default 8), `--tools-call-timeout`
seconds per call (default 20),
`--tools-budget` seconds of tool time per Response (default 120), and the
request's `max_tool_calls`. A call past a bound is not run, its output says
so, and the model is asked once more with `tool_choice: none` for an answer.
`max_output_tokens` covers all rounds together and `usage` is their sum.

Batch rules: every hosted call in a turn runs, independently; a refusal,
malformed argument, timeout or error becomes that call's output (status
`incomplete`, text starting `error:`) for the model to read, never a failed
Response. A turn that also calls a client tool runs its hosted calls and then
ends the Response for the client. A turn cut short (token limit,
cancellation) runs nothing. Cancelling the Response (disconnect,
`POST /v1/responses/{id}/cancel`, shutdown) stops the running tool.

Hosted tools are only for the Responses API; Chat Completions is unchanged.
A background Response that declares hosted tools runs the same loop. Before
its first hosted call starts executing, a durable one writes and flushes an
owner-only `.{id}.tools` marker beside its record (a write that fails runs no
call and fails the Response). If a crash or restart interrupts it, a record
without the marker provably never ran a hosted call and is regenerated like
any other; one with it is marked `failed`, saying that a hosted call had
started executing, rather than regenerated: regenerating could run a shell
command or file write a second time. The marker is conservative: it is
written once per Response, before the first call even if that call is then
refused or not run.

A plain (non-strict) `function` tool's schema is not enforced: a call whose
arguments miss it (for example, without a required property) is returned to
the client as generated, for the client to reject in its
`function_call_output`, rather than failing the Response. So is a call to a
tool the request does not declare (Codex's instructions name `apply_patch`
even when its default tool set has none, and the model sometimes calls it):
it is reported as a `function_call` for the client to answer as unknown, and
history that replays a call to a tool no longer declared is accepted, since
harnesses change their tool sets between turns. Only a malformed call layout
still fails the Response.

## Permission model and its limits

All grants are explicit server flags; nothing is reachable by default.

**Filesystem** (`local-services` `tools::fs`, reusing `workspace.rs`): every
path is relative to the root. `..`, absolute paths, any `.git` component and
symlinks (at any component) are refused; files are opened through a
capability handle on the root (`cap-std`), which cannot resolve outside it
even if a component is swapped between check and open. Reads, listings and
searches are bounded by the workspace limits and marked when truncated.
Writes and edits need `--tools-allow-write`, create missing directories and
are capped at 4 MiB; edits replace exactly one occurrence. A file with more
than one hard link is never written (its other name may be outside the
root), and the shell sandbox refuses to create a hard link to a file outside
the root.

**Shell** (`tools::shell`): `/bin/sh -c` under `sandbox-exec` with a
deny-by-default profile: no network (`connect` fails with EPERM), no Mach
lookups (so it cannot ask launchd to start anything outside the sandbox),
reads limited to system directories, the root and a per-call scratch
directory (metadata is readable everywhere), writes limited to the scratch
directory, `/dev/null` and, only with `--tools-allow-write`, the root minus
`.git`. In the child before `exec`: `RLIMIT_CPU` 10 s, `RLIMIT_FSIZE` 16 MiB,
`RLIMIT_NOFILE` 256, `RLIMIT_CORE` 0 and `RLIMIT_NPROC` set to the user's
current process count plus 64 (the limit is per user on macOS). The child
leads a new process group; at the deadline, when the command exits, or when
the call is dropped, that group (and only it) receives `SIGKILL`, always
before the leader is reaped so the group ID cannot have been reused. Each
output stream keeps 32 KiB and reports the rest as a truncation marker. The
environment is cleared (`PATH`, `HOME`/`TMPDIR` = scratch, `LANG`), stdin is
`/dev/null`.

**Web fetch** (`tools::fetch`): `http`/`https` only, no user info, DNS names
only (IP literals refused), host must equal an allowlisted domain or be a
subdomain. Every resolved address must be public (loopback, RFC 1918,
link-local, CGNAT, documentation, multicast, unique-local and IPv4-mapped
forms are refused), and the connection goes to a checked address, so the
name is not re-resolved. Redirects are reported, not followed. TLS is
verified by the macOS trust store. The body is capped at 256 KiB.

Remote image and video URLs in chat requests use the same transport
(`tools::fetch::download`) with the same address checks but no allowlist,
public IP literals allowed, up to five redirects each checked anew, and a
64 MiB cap ([docs/BONSAI.md](BONSAI.md#remote-media-urls)).

**Known gaps** (stated, not papered over):

- macOS rejects `RLIMIT_AS` and does not enforce `RLIMIT_DATA`/`RLIMIT_RSS`,
  so a command's memory is bounded only indirectly by CPU time and the
  deadline.
- A process that calls `setsid` leaves the process group and survives the
  deadline kill; it stays inside the sandbox, its CPU limit and the process
  limit.
- `RLIMIT_NPROC` counts all of the user's processes, so the margin is shared
  with anything else the user starts meanwhile.
- The shell can read system directories (`/usr`, `/System`, `/private/etc`,
  ...), which any local user can read; it cannot read the home directory
  outside the root.
- Reads do not check hard links: a hard link to an outside file that already
  exists in the root (made by its owner, not by the tools) is readable.
- Filesystem work is synchronous and bounded by size limits rather than
  interrupted by cancellation.
- `sandbox-exec` is deprecated by Apple but still shipped and enforced; if it
  disappears, the shell fails closed (it cannot start).

## Live check (2026-10-11, Apple M4 Pro, PTQ1 checkpoint)

Server: `local-ai serve --tools-root SCRATCH --tools-shell` (no writes, no
fetch), scratch root holding `notes.txt` (3 lines) and `docs/readme.md`.

- Tool loop, one non-streaming Response (HTTP 200, 10.2 s): the model called
  `local_fs_list`, `local_fs_read {"path":"notes.txt"}` and `shell` with
  `["ls", "wc -l notes.txt docs/readme.md"]`; each ran (`shell_call_output`
  with `exit_code` 0 and the real `ls`/`wc` output) and the final message gave
  the correct line counts.
- Streaming (HTTP 200): `response.shell_call_command.added/delta/done`,
  `response.shell_call_output_content.done` and the output item events, in
  order, for a `shell` call running `ls`.
- Declaring `local_fs_write` or `local_web_fetch` on this server: 400 naming
  the missing flag, before any generation.
- Forced `local_fs_read {"path":"../outside/secret.txt"}`: the call's output
  is `error: invalid arguments: ... must be a relative path that stays inside
  the workspace` (status `incomplete`); the outside file's canary text
  appears nowhere in the Response, and the model reported the refusal.

## Live check: `file_search` (2026-10-19, Apple Silicon, PTQ1 checkpoint)

Server: `local-ai serve` with no `--tools-*` flag and the `EmbeddingGemma` 2
text model installed; one vector store of three small Markdown files
(a fictional station handbook, a travel policy and bread notes; facts the
model cannot know).

- Non-streaming, `include: ["file_search_call.results"]`, reasoning `low`
  (HTTP 200, 20.5 s, 1,202 input and 319 output tokens): the model made one
  call with three queries, the `file_search_call` item listed the three
  files' results with scores, and the answer gave the right figures (41
  hours, Ilse Varga, 37 euros) with three `file_citation` annotations, each
  at the end of the sentence it supports and naming the right file.
- Streaming (9.6 s): `output_item.added` (`in_progress`),
  `file_search_call.in_progress`, `.searching`, `.completed`,
  `output_item.done` (`completed`), then the answer (docking port C, Priya
  Anand) with two `output_text.annotation.added` events; the concatenated
  deltas equal the final text, markers removed.
- A follow-up with `previous_response_id` answered from the replayed results
  ("quillmark.md") without searching again; a Response in a conversation
  appended its `file_search_call` item; an unknown store ID was a 400 naming
  it, before generation.

Measured search time inside the loop (call start to results, warm): 13 to
15 ms with the embedding ranking (one or two queries, including embedding
them; 46 ms for the first, three-query call), 2 ms with `ranker: none`
(BM25). Through `POST /v1/vector_stores/{id}/search` on the same store:
about 55 ms per HTTP request with embeddings and 1 ms with BM25. Whole
Responses with a one-number answer and no reasoning took 2 to 6 s; what
dominates is generation and the prefill of the returned chunks, not the
search.
