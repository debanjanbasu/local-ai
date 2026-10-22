# Agent-quality benchmark

`tools/agent_bench.py` runs a fixed set of seven coding tasks through the real
Codex and opencode CLIs against `local-ai serve`, and records how well the
model performs as their agent. Use it to compare a baseline build with a
candidate build after any engine, kernel, sampling, caching, compaction or
tool-calling change. `tools/client_compat.py` remains the pass/fail
compatibility suite. This benchmark measures quality and latency.

## Running it

The run loads the model, so hold the shared GPU lock for the whole run. The
script starts `serve` itself on port 8080, runs every task, then stops the
server.

```bash
# once: Codex 0.163.0 (opencode 1.19.3 is on PATH via Homebrew)
npm install --prefix .scratch/codex @openai/codex@0.163.0

# baseline (about 25 minutes at effort none)
lockf -k /tmp/local-ai-gpu.lock python3 tools/agent_bench.py \
    --codex-bin "$PWD/.scratch/codex/node_modules/.bin/codex" \
    --label baseline-b09e22f --out "$PWD/.scratch/artifacts/agent-bench/baseline"

# candidate: the same command with another server binary and label
lockf -k /tmp/local-ai-gpu.lock python3 tools/agent_bench.py \
    --codex-bin "$PWD/.scratch/codex/node_modules/.bin/codex" \
    --server-bin /path/to/candidate/target/release/local-ai \
    --label candidate-xyz --out "$PWD/.scratch/artifacts/agent-bench/candidate"

python3 tools/agent_bench.py --compare \
    .scratch/artifacts/agent-bench/baseline/results.json \
    .scratch/artifacts/agent-bench/candidate/results.json
```

Options: `--clients codex,opencode` (also `codex-apply-patch`, Codex with its
freeform `apply_patch` grammar tool, `codex-remote-compact`, Codex compacting
through the server whenever a turn reaches 7,000 tokens, and
`opencode-responses`), `--tasks`
(a comma-separated subset), `--effort none|low|medium|xhigh` (default
`none`), `--seeds 1,2,3` (each seed runs the whole matrix), `--timeout`
(per run, default 600 s), and `--no-serve` (use a server that is already
listening on `--port`).

## Tasks

| Task | What it exercises | Check (hidden from the agent) |
| --- | --- | --- |
| fix-pagination | Trace a failing test to the helper `ceil_div` in another module | Visible tests pass, tests/ untouched, and a hidden test of `ceil_div` itself, so a fix that only patches the caller fails |
| implement-slugify | Implement a function exactly from its docstring | Nine hidden cases (Unicode, truncation, empty input) |
| rename-function | Rename `calc_total` across modules, tests and README | No `calc_total` anywhere, no alias, visible and hidden tests pass |
| locate-answer | Answer a question that needs three files, with a decoy legacy file | Answer has `status=409` and `limit=1000`, and the repository is unchanged |
| add-cli-flag | Add `--ignore-blank` to a CLI and document it | Hidden subprocess tests with and without the flag, README mentions it, test file untouched |
| fix-large-file | Find a mistyped constant in a 600-line module | Hidden exact-constant test, tests/ untouched |
| make-check-pass | Make `sh scripts/check.sh` pass: a lint failure, then a test failure | The check passes and scripts/, tools/ and tests/ are untouched; hidden behaviour test |

`tools/test_agent_bench.py` proves that every check fails on the untouched
fixture, passes on a reference solution, and rejects the likely shortcuts:
patching the symptom, editing the test, leaving an alias, giving the decoy's
answer, and leaving the option undocumented.

## Metrics

Each run (task × client × seed) records:

- **success**: the hidden check passed and the client exited 0 within the
  timeout.
- **turns**: the agent's model requests, meaning requests that declare
  tools. Compaction turns are counted separately.
- **tool-call failures**, in the three categories the tool-reliability work
  uses:
  - `wrong_tool`: a call to a name the request did not declare, or one the
    client refused as unavailable (`unsupported call`, `unavailable tool`).
  - `bad_args`: arguments that are not JSON or not an object, or that
    violate the declared JSON schema (missing required properties, wrong
    types, properties outside an `additionalProperties: false` schema,
    values outside an enum); a client rejection of the arguments (`failed to
    parse function arguments`, `invalid arguments`, `apply_patch
    verification failed`); tool-call markup leaked into message text; or a
    server failure that mentions a tool call.
  - `loop`: an identical call (same name and canonical arguments) used for
    the third or later time in a run. Running the tests before and after a
    fix is one allowed repeat. `write_stdin` polling is exempt.
- **wall_seconds**: wall-clock time of the client process.
- **TTFT** per turn: the time from the proxy forwarding the request to the
  first streamed chunk with generated output, either a Responses `*.delta`
  event or a Chat chunk with content, reasoning or a tool call. Headers and
  `response.created` do not count. Reported as first turn, median and max.
- Per turn: prompt tokens, cached prompt tokens, output tokens and request
  duration. Per run: total and uncached prompt tokens and the largest prompt.

Each call record keeps its category and evidence, so a failure can be traced
to the exact exchange in `exchanges.jsonl`.

Output under `--out`: `results.json` (runs, per-turn and per-call records,
per-client aggregates, server binary hash and commit, client versions, and
the command line), `summary.md`, `exchanges.jsonl`, `server.log`, and
`runs/<client>-<task>-seed<N>/` with the client log and the agent's diff.

## Determinism

- Task files, prompts and task order are fixed. The 600-line module is
  generated from a fixed table.
- Each client works in one fixed repository path, HOME and TMPDIR under
  `--work` (default `/tmp/local-ai-agent-bench`), which are wiped and rebuilt
  for every task. The clients' system prompts quote these paths, so the
  prompts are identical between tasks and between runs. A client's first
  task prefills its roughly 6,000-token prompt cold, and later tasks reuse
  the shared prefix. Compare full runs, or the same `--tasks` subset.
- The proxy pins `seed` on every generation request (`--seeds`; the server's
  default sampling seed is fixed too).
- The server runs with a fresh `HOME`, so its disk prompt cache, response
  store and reasoning key start empty.
- opencode runs with `--title`, so no title-generation request competes with
  its first turn.

Still nondeterministic: the date in the clients' system prompts, and
floating-point differences that a candidate's kernel changes legitimately
introduce, which can change sampled tokens and therefore the agent's path.
The same-build rerun below measures the remaining noise.

## Baseline (2026-10-22, main `a4cf247`)

Effort `none`, seed 1, Codex 0.163.0 and opencode 1.19.3, M4 Pro 48 GB, no
other GPU job (the run held `lockf -k /tmp/local-ai-gpu.lock` throughout),
about 25 minutes per run. The same command twice on the same build:

```bash
lockf -k /tmp/local-ai-gpu.lock python3 tools/agent_bench.py \
    --codex-bin "$PWD/.scratch/codex/node_modules/.bin/codex" \
    --server-bin "$PWD/.scratch/bins/a4cf247/local-ai" \
    --label main-a4cf247 --out "$PWD/.scratch/artifacts/agent-bench/baseline-a4cf247"
```

| Client | Success | Turns (mean) | Tool calls | wrong_tool | bad_args | loop | Timeouts | Wall s (total) | TTFT first / all (median) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| codex, run 1 | 6/7 | 18.9 | 130 | 4 | 0 | 2 | 1 | 1,197 | 1.3 / 3.4 s |
| codex, run 2 | 7/7 | 14.9 | 99 | 3 | 2 | 1 | 0 | 862 | 1.2 / 2.6 s |
| opencode, run 1 | 7/7 | 6.4 | 60 | 0 | 0 | 1 | 0 | 309 | 1.1 / 1.1 s |
| opencode, run 2 | 7/7 | 6.4 | 60 | 0 | 0 | 1 | 0 | 309 | 1.1 / 1.1 s |

| Task | Codex run 1: pass, turns, wall s | Codex run 2 | opencode (both runs) |
| --- | --- | --- | --- |
| fix-pagination | FAIL, 50, 600 (timeout) | PASS, 17, 234 | PASS, 6, 72 |
| implement-slugify | PASS, 12, 94 | PASS, 14, 116 | PASS, 7, 35 |
| rename-function | PASS, 22, 149 | PASS, 14, 74 | PASS, 5, 37 |
| locate-answer | PASS, 4, 10 | PASS, 4, 10 | PASS, 4, 15 |
| add-cli-flag | PASS, 22, 187 | PASS, 24, 208 | PASS, 8, 37 |
| fix-large-file | PASS, 9, 72 | PASS, 15, 104 | PASS, 6, 87 |
| make-check-pass | PASS, 13, 85 | PASS, 16, 117 | PASS, 9, 25 |

What the baseline shows:

- **opencode is deterministic, Codex is not.** Both opencode runs made the
  same calls in the same order, and every wall time agreed within a second.
  Codex diverged at its fifth call of the first task (a different
  `apply_patch` hunk) and took a different path in five of seven tasks,
  although the proxy pins the seed. So one Codex run is a sample: run 1
  failed `fix-pagination` after 50 turns spent debugging Python's import
  cache when its edit had not landed, and run 2 passed it in 17. Compare
  Codex with several seeds (`--seeds 1,2,3`) or repeated runs, and read a
  single-run Codex difference of one task or a few hundred seconds as noise.
- **Tool-call failures are rare and the same kind every time.** Every
  `wrong_tool` is Codex calling the undeclared `apply_patch` (it gets
  `unsupported call` and edits with `exec_command`); the two `bad_args` in
  run 2 are `write_stdin` calls without the required `session_id`. No call
  had malformed JSON or leaked markup.
- **GNU-userland commands on macOS** (not a bench metric; counted by a
  scratch script over the call signatures and the tool outputs): Codex used
  `sed -i 's/…/'` without the `''` backup suffix 3 and 5 times, and `cat -A`
  twice and once; opencode ran a bare `python` once in each run. Each one
  failed and the model retried with a working command.
- **TTFT**: a client's first task prefills its roughly 6,000-token prompt
  cold (45 to 48 s); every later first turn restores it from the cache
  (0.8 to 3.5 s), and the median turn starts streaming in 1.1 to 3.4 s.
