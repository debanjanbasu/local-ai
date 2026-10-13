# Tools on the Responses API

The model always chooses tools; Rust executes them, outside model execution.
Two families exist.

## Client-executed tools (no permission needed)

These are returned to the client, which runs them and sends the result back
in the next request (or with `previous_response_id`):

| request tool | output item | client answers with |
|---|---|---|
| `function` | `function_call` | `function_call_output` |
| `custom` (`format` text, or `grammar` with `syntax: "regex"`) | `custom_tool_call` | `custom_tool_call_output` |
| `local_shell` | `local_shell_call` | `local_shell_call_output` |
| `apply_patch` | `apply_patch_call` | `apply_patch_call_output` |

Each becomes a strict native function, so arguments are constrained while
they are generated; a `regex` grammar constrains the custom tool's `input`
the same way. `lark` grammars are refused (400). `tool_choice` accepts
`{"type":"custom","name":...}`, `{"type":"shell"}` and
`{"type":"apply_patch"}` as well as the function forms.

## Hosted tools (server permission required; default: none)

The server runs these itself, in a loop: generation stops on the call, the
tool runs, its result is appended and generation continues, all within one
Response. A request declares them; the server must also have been started
with the matching permission, or the request is refused with a 400 naming
the flag.

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

Bounds (flags): `--tools-max-iterations` rounds of hosted calls per Response
(default 8), `--tools-call-timeout` seconds per call (default 20),
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
`function_call_output`, rather than failing the Response.

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
