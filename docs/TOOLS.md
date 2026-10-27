# Tools on the Responses API

The model always chooses tools; Rust executes them, outside model execution.
The client runs some; this server runs `file_search` and the hosted tools,
all of them always on. Tools are declared as Responses `tools`, or Chat
Completions `tools` with `tool_choice`.

## Client-executed tools

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

## File search

`{"type":"file_search","vector_store_ids":[...]}` runs here, natively, over
this server's [vector stores](BONSAI.md#native-services), in the same
server-side loop as the hosted tools below. It only reads the vector stores
of the server's own data directory, which the same callers can already
search with `POST /v1/vector_stores/{id}/search`.

| member | accepted |
|---|---|
| `vector_store_ids` | 1 to 16 store IDs; each must exist and not be expired, or the request is refused (400) before generation |
| `max_num_results` | 1 to 50; **default 5** (results are chunks the model must read; see the cost below) |
| `ranking_options.ranker` | `auto` (default: `EmbeddingGemma` 2 cosine ranking) or `none` (BM25 only, no GPU work); `default-2024-11-15` is refused |
| `ranking_options.score_threshold` | 0 to 1, on the reported score |
| `ranking_options.hybrid_search` | `embedding_weight` and `text_weight`: weighted reciprocal rank fusion of the two rankings (`k` = 60); needs ranker `auto` |
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

## Hosted tools (always on)

The server runs these itself, in the same loop: generation stops on the call,
the tool runs, its result is appended and generation continues, all within
one Response. A request declares them, and every one is always available:
there is no flag to grant or withhold any of them. The filesystem tools and
the shell are rooted at the directory `serve` was started in, canonicalized
at startup and reported in the startup policy (`hosted_tools.root`).

| request tool | does | output items |
|---|---|---|
| `{"type":"shell"}` (environment omitted or `{"type":"local"}`) | sandboxed commands in the root, writes allowed | `shell_call`, `shell_call_output` |
| `{"type":"function","name":"local_fs_list"}` / `local_fs_read` / `local_fs_search` | list, read and search under the root | `function_call`, `function_call_output` |
| `{"type":"function","name":"local_fs_write"}` / `local_fs_edit` | write and edit files under the root | `function_call`, `function_call_output` |
| `{"type":"function","name":"local_web_fetch"}` | one `GET` to any public host | `function_call`, `function_call_output` |

Every caller can therefore run shell commands and change files in the root,
so start `serve` in a directory you are willing to expose to its callers.
`serve` refuses a non-loopback `--host` without `--api-key`; on loopback
without a key, any local process may use the tools, and web pages are kept
out by the [DNS-rebinding and browser-origin protections](ADMIN.md#browsers).

The `local_*` names are reserved: declare them without `parameters` (the
server supplies the schema). The OpenAI spec defines no hosted filesystem or
fetch tool (`file_search` is vector-store retrieval, `web_search` is search),
so these are reported as what they are, a function call plus the output the
server produced, not as a built-in tool they are not.

Bounds (fixed, and they also bound `file_search`): 8 rounds of hosted calls
per Response, 20 s per call, 120 s of tool time per Response, and the
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

## Protections and their limits

Nothing is granted or withheld per server: every tool is on, and these
protections hold each one in.

**Filesystem** (`local-services` `tools::fs`, reusing `workspace.rs`): every
path is relative to the root. `..`, absolute paths, any `.git` component and
symlinks (at any component) are refused; files are opened through a
capability handle on the root (`cap-std`), which cannot resolve outside it
even if a component is swapped between check and open. Reads, listings and
searches are bounded by the workspace limits and marked when truncated.
Writes and edits create missing directories and are capped at 4 MiB; edits replace exactly one occurrence. The file itself is
opened with `O_NOFOLLOW` and checked through that handle (a regular file, the
inode that was checked) before it is truncated, read and written through the
same handle, so a symlink swapped in after the check is refused, not
followed. A file with more than one hard link is never written (its other
name may be outside the root), and the shell sandbox refuses to create a hard link to a file outside
the root.

**Shell** (`tools::shell`): `/bin/sh -c` under `sandbox-exec` with a
deny-by-default profile: no network (`connect` fails with EPERM), no Mach
lookups (so it cannot ask launchd to start anything outside the sandbox),
reads limited to system directories, the root and a per-call scratch
directory (metadata is readable everywhere), writes limited to the scratch
directory, `/dev/null` and the root minus every `.git` directory (at any depth, matched case-insensitively, as the file
tools refuse them). In the child before `exec`: `RLIMIT_CPU` 10 s, `RLIMIT_FSIZE` 16 MiB,
`RLIMIT_NOFILE` 256, `RLIMIT_CORE` 0 and `RLIMIT_NPROC` set to the user's
current process count plus 64 (the limit is per user on macOS). The child
leads a new process group; at the deadline, when the command exits, or when
the call is dropped, that group (and only it) receives `SIGKILL`, always
before the leader is reaped so the group ID cannot have been reused. At the
deadline, and when the call is dropped, the leader is also killed by PID (it
is still unreaped), so a leader that moved itself into another group cannot
outlive the call; a call still unreaped 2 s after its deadline is abandoned
(and killed again). Descriptors the server holds are marked close-on-exec in
the child, so the command inherits only its three standard streams. The
scratch directory is measured every 250 ms (polling, because macOS has no
per-directory quota and no event that reports bytes written) and a command
whose scratch tree passes 1 GiB allocated or 20,000 entries is killed
(reported as `SIGKILL`, with a marker on stderr). Each
output stream keeps 32 KiB and reports the rest as a truncation marker. The
environment is cleared (`PATH`, `HOME`/`TMPDIR` = scratch, `LANG`), stdin is
`/dev/null`.

**Web fetch** (`tools::fetch`): `http`/`https` only, no user info, DNS names
only (IP literals refused), any host. Every resolved address must be public (loopback, RFC 1918,
link-local, CGNAT, documentation, multicast, unique-local and IPv4-mapped
forms are refused), and the connection goes to a checked address, so the
name is not re-resolved. Redirects are reported, not followed. TLS is
verified by the macOS trust store. The body is capped at 256 KiB, and the
call's deadline covers the whole exchange, decoding and conversion included
(a call that runs out abandons its connection or stream).

*Transport.* `https` origins known to serve HTTP/3 use QUIC (`quinn`, `h3`),
raced against TCP: QUIC starts first and TCP 300 ms later, and whichever
connects first carries the request. Support is learnt from `Alt-Svc`
response headers (cached per origin for their `ma`, at most 7 days; `clear`
forgets them) and from DNS HTTPS records (RFC 9460, asked of the system DNS
service alongside the address lookup, so a first request can already use
HTTP/3; the first fetch waits at most 30 ms for that answer). Only an
alternative on the same host is used, at a port of its choosing (an origin
on an unprivileged port cannot name a privileged one); one naming another
host, an HTTPS record in alias mode or with another target, and its address
hints are ignored, so an alternative never adds a host or an address. A
QUIC failure (handshake over 3 s, blocked UDP, error) falls back to TCP
without failing the call and keeps the origin off QUIC for 5 minutes,
doubling per failure up to 48 hours. Other origins use TCP with ALPN `h2`
then `http/1.1`. A host's addresses are tried in turn, 250 ms apart, IPv6
and IPv4 alternating (RFC 8305).

*Connections and TLS.* One rustls configuration (ring, platform verifier)
is built once and shared by TCP and QUIC, with ALPN and a session cache per
transport, so a returning origin resumes its TLS session. QUIC sends the
request as 0-RTT early data when it holds a ticket: the tool only sends
bodiless `GET`s, which are idempotent, so a replayed request changes
nothing; a request lost to rejected early data is retried once on a full
handshake. Connections are pooled on a small runtime of their own: per
origin one multiplexed HTTP/2 or HTTP/3 connection and up to four idle
HTTP/1.1 connections (keep-alive), for at most 32 origins, each closed
after 30 s unused. An HTTP/1.1 connection returns to the pool only after
its body was read to the end; a body cut at the limit or a cancelled call
closes it, and a cancelled HTTP/2 or HTTP/3 stream is reset. A request that
fails on a pooled connection before any response is retried once on a new
one. Checked address lists are cached for 30 s (`getaddrinfo` reports no
TTL); only a list whose every address passed the check is cached, and
every TCP or UDP connection goes to an address from such a list, checked
again just before connecting.

*Compression.* Requests send `Accept-Encoding: zstd, br, gzip, deflate`;
the body is decoded as it streams (raw deflate is accepted for `deflate`).
Two limits bound it: the decoded bytes kept, and the encoded bytes read
(the decoded limit plus 64 KiB), so a decompression bomb stops at the limit
and a stream that decodes to nothing stops at the encoded one. The zstd
window is capped at 8 MiB (RFC 9659). A corrupt stream, or an unknown or
layered coding, fails the call.

*HTML to Markdown.* `text/html` and `application/xhtml+xml` bodies are
converted to Markdown (`html-to-markdown-rs`, html5ever-based; chosen over
`htmd`, `fast_html2md` and `mdka` on eight real pages for its output:
tables, code blocks and absolute links intact, where `htmd` lost table
cells and overflowed the stack on deeply nested markup and `mdka` took
10 s on it). Scripts, styles, comments, images, forms and navigation chrome
are dropped; headings, paragraphs, lists, links (made absolute against the
page URL and any `<base>`), code blocks and tables are kept. Up to four
times the limit (1 MiB) of decoded HTML is read, the Markdown is capped at
the limit (256 KiB), and each cut has its own marker (`[truncated: the page
was cut at ... bytes of HTML before conversion]`, `[truncated: the Markdown
was cut at ... bytes]`). The charset is the header's, else a `<meta>`
declaration, else UTF-8. Conversion runs on its own thread with a 32 MiB
stack and a depth limit of 512, taking milliseconds per page (about 70 ms
for a 1 MB rustdoc page; at worst about 0.3 s per MiB of hostile markup).
The request asks for `text/markdown` first, so a server that offers
Markdown returns it as is. JSON and other text are decoded from their
declared charset and passed through; other content is shown as UTF-8, as
before.

Remote image and video URLs in chat requests use the same transport
(`tools::fetch::download`) with the same address checks,
public IP literals allowed, up to five redirects each checked anew (never
from `https` to `http`), the media type checked before the body is read, and
a 64 MiB cap on the decoded body (a larger one fails rather than being cut;
[docs/BONSAI.md](BONSAI.md#remote-media-urls)). They share the pool, the
HTTP/3 race and the content codings; nothing is converted.

**Known gaps** (stated, not papered over):

- macOS rejects `RLIMIT_AS` and does not enforce `RLIMIT_DATA`/`RLIMIT_RSS`,
  so a command's memory is bounded only indirectly by CPU time and the
  deadline.
- A process other than the leader that calls `setsid` or `setpgid` leaves
  the process group and survives the deadline kill; it stays inside the
  sandbox, its CPU limit and the process limit.
- Disk use is bounded only in the scratch directory, and only to within what
  a command writes in one 250 ms check interval. Writes to the root are
  bounded per file
  (`RLIMIT_FSIZE`, 16 MiB) and by the deadline, CPU and process limits, not
  in total; and a process that escaped the group can keep writing after
  the call ends (to the root, or to scratch files it still holds open).
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

Server: `local-ai serve` with the shell and read-only file tools granted to
a scratch root (the grant flags of the time, since removed: every tool is now
always on), the root holding `notes.txt` (3 lines) and `docs/readme.md`.

- Tool loop, one non-streaming Response (HTTP 200, 10.2 s): the model called
  `local_fs_list`, `local_fs_read {"path":"notes.txt"}` and `shell` with
  `["ls", "wc -l notes.txt docs/readme.md"]`; each ran (`shell_call_output`
  with `exit_code` 0 and the real `ls`/`wc` output) and the final message gave
  the correct line counts.
- Streaming (HTTP 200): `response.shell_call_command.added/delta/done`,
  `response.shell_call_output_content.done` and the output item events, in
  order, for a `shell` call running `ls`.
- Forced `local_fs_read {"path":"../outside/secret.txt"}`: the call's output
  is `error: invalid arguments: ... must be a relative path that stays inside
  the workspace` (status `incomplete`); the outside file's canary text
  appears nowhere in the Response, and the model reported the refusal.

## Live check: `file_search` (2026-10-19, Apple Silicon, PTQ1 checkpoint)

Server: `local-ai serve` with the `EmbeddingGemma` 2 text model; one vector store of three small Markdown files
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

## Live check: web fetch transport (2026-10-27, Apple Silicon, release build)

One round per site from a corporate network whose proxy (Zscaler)
intercepts TLS over TCP (certificates issued by a Zscaler CA, trusted by
the macOS store) but let QUIC on UDP 443 through untouched (the QUIC
certificate chains were the origins' own). Before: the previous transport
(`main` at `d25b277`: HTTP/1.1, `identity`, a new connection per call, raw
HTML up to 256 KiB). After: this transport, a fresh client per site; "warm"
is the next request on the pooled connection, "new conn" a new connection
after the pool was cleared with HTTP/3 support already learnt. Times are
TTFB / total in ms (total includes decoding and Markdown conversion); wire
is body bytes received (encoded; headers and TLS excluded); tokens are the
Bonsai tokenizer's count (`local-ai bonsai --tokenize --raw`) of the text
the model would read.

| site | before cold | before warm | after cold | after warm | after new conn | wire before → after | tokens before → after |
|---|---|---|---|---|---|---|---|
| www.cloudflare.com | 435 / 492 | 399 / 453 | 158 / 159 h2 | 37 / 38 h2 | 69 / 69 h3 | 262,144 → 1,331 | 105,679 → 754 |
| www.google.com | 394 / 420 | 424 / 458 | 374 / 425 h3 | 192 / 218 h3 | 304 / 341 h3 | 86,390 → 29,199 | 34,952 → 220 |
| doc.rust-lang.org `Vec` (955 KB) | 270 / 360 | 129 / 186 | 372 / 543 h2 | 19 / 106 h2 | 160 / 379 h3 | 262,144 → 108,570 | 87,447 → 90,963 |
| developer.mozilla.org `Alt-Svc` | 1629 / 1754 | 348 / 388 | 710 / 739 h2 | 97 / 110 h2 | 94 / 115 h2 | 209,790 → 21,305 | 61,350 → 1,590 |
| www.gnu.org (HTTP/1.1 only) | 1228 / 1238 | 1299 / 1308 | 879 / 884 h1 | 286 / 289 h1 | 335 / 340 h1 | 31,743 → 9,886 | 9,977 → 3,089 |

- `www.google.com` used HTTP/3 on its very first request, learnt from its
  DNS HTTPS record; `www.cloudflare.com`'s record arrived after the 30 ms
  grace, so its first request used HTTP/2 and later connections HTTP/3, as
  did rustdoc's after its `Alt-Svc`. MDN and gnu.org advertise no HTTP/3.
- `www.cloudflare.com` answered the `text/markdown` preference with its own
  Markdown (1.3 KB on the wire instead of a 1.3 MB page); the other pages
  were converted here.
- rustdoc's `Vec` page is larger than either limit: before, the model got
  the first 256 KiB of raw HTML (mostly the sidebar); after, the whole page
  (108 KB of Brotli) is converted and the Markdown is cut at 256 KiB, so the
  token count stays at the cap while the content is the page's text.
- A second round minutes later, on a congested link, was 2 to 4 times slower
  across the board (gnu.org timed out for both transports) with the same
  protocols and byte counts.
