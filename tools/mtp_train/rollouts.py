"""Self-generated coding and tool-call transcripts for MTP head training.

The head drafts the target's own continuations, and corpus text the target
did not write did not transfer to live acceptance (docs/BONSAI.md), so this
collects the target's greedy outputs on coding-agent work through a running
`local-ai serve`, as `mtp-capture capture` transcripts (one JSON object per
line: `messages`, `tools`, `thinking`, plus `id` and `kind`):

  code     write a function, class or script in one of eight languages
  context  explain, test, refactor or review a real source excerpt
  agent    a multi-turn coding agent over a real repository: read-only tools
           run for real (reads, listings, ripgrep), writes land in an
           in-memory overlay, and commands outside a read-only allowlist are
           not executed
  api      single calls to general JSON tools (structured arguments)

Prompts are generated from a fixed seed and never reuse the tasks of
`tools/mtp_code_bench.py`. Output is appended and existing ids are skipped,
so a run can stop and resume (lock the GPU around it, a chunk at a time).

  lockf -k /tmp/local-ai-gpu.lock sh -c 'local-ai serve --no-response-store & ...;
      python3 tools/mtp_train/rollouts.py --out rollouts.jsonl --repo . --limit 40'
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import shlex
import subprocess
import threading
import urllib.request
from pathlib import Path

LANGUAGES = ["Python", "Rust", "TypeScript", "Go", "Java", "C++", "JavaScript", "Bash"]
CODE_TASKS = [
    "a trie with insert, search and prefix search",
    "Dijkstra's shortest paths over an adjacency list",
    "a topological sort that reports cycles",
    "a token-bucket rate limiter with a refill rate and burst size",
    "a function returning a matrix's elements in spiral order",
    "run-length encoding and decoding of strings",
    "a bounded blocking queue for producer/consumer threads",
    "retry with exponential backoff and jitter around a fallible call",
    "a parser for durations like '1h30m15s' into seconds",
    "a min-heap priority queue without the standard library heap",
    "an interval tree supporting overlap queries",
    "a function that validates and normalizes semantic version strings",
    "a union-find structure with path compression and union by rank",
    "a word-frequency counter reading a file line by line",
    "a function that deep-merges two nested maps",
    "a sliding-window maximum over an array",
    "a simple tokenizer for arithmetic expressions and an evaluator using shunting-yard",
    "a ring buffer with push, pop and iteration",
    "a function that finds all anagram groups in a list of words",
    "a debounced file-change watcher callback wrapper",
    "an LFU cache",
    "a function computing the Levenshtein distance between two strings",
    "a bloom filter with configurable false-positive rate",
    "a markdown table renderer from rows of records",
    "a function converting Roman numerals to integers and back",
    "a thread pool that runs submitted jobs and joins on shutdown",
    "a command-line tool that counts lines of code per file extension in a directory",
    "a function that flattens arbitrarily nested lists",
    "binary tree serialization to a string and back",
    "a function that schedules meetings into the minimum number of rooms",
    "a CSV-to-JSON converter that infers numeric columns",
    "an HTTP request router matching paths with :params",
    "a function that checks whether a Sudoku board is valid",
    "a k-means clustering routine for 2D points",
    "an event emitter with on, off and once",
    "a function that pretty-prints elapsed time in human units",
    "a memoizing decorator or wrapper with a maximum size",
    "a big-integer addition and multiplication on decimal strings",
    "a function that wraps text to a given width without breaking words",
    "a simple key-value store persisted to a JSON file with atomic writes",
]
CODE_ASKS = [
    "Write {task} in {lang}. Include brief comments and a small usage example.",
    "Implement {task} in {lang}, with unit tests.",
    "In {lang}, write {task}. Keep it idiomatic and explain any trade-offs briefly.",
    "I need {task} in {lang}. Show the code first, then explain its complexity.",
]
CONTEXT_ASKS = [
    "Explain what this code does and point out anything surprising.",
    "Write unit tests for this code.",
    "Refactor this code for readability without changing its behaviour. Show the full result.",
    "Review this code for bugs, edge cases and performance problems.",
    "Add documentation comments to this code and return it.",
    "Suggest how to make this code more robust to bad input, with the changed code.",
]
SOURCE_SUFFIXES = {".rs": "rust", ".py": "python", ".ts": "typescript", ".tsx": "tsx",
                   ".js": "javascript", ".go": "go"}
DEFINITION = re.compile(r"^\s*(pub(\(crate\))? )?(async )?(fn|def|function|class|impl|struct|"
                        r"export (async )?function|export class|func) ")
AGENT_ASKS = [
    "Find where `{symbol}` is defined in this repository and explain how it is used.",
    "Add a doc comment to `{symbol}` describing what it does, its inputs and outputs.",
    "Write a unit test for `{symbol}` next to the existing tests.",
    "Is `{symbol}` covered by any tests? Find out and summarize.",
    "Look at `{path}` and tell me what its main responsibilities are.",
    "Rename the local variables in `{symbol}` to clearer names where they are unclear.",
    "Find every caller of `{symbol}` and list them with one line each about why they call it.",
    "There may be an unhandled error path in `{symbol}`. Check and fix it if so.",
]
SYSTEMS = [
    ("You are a coding agent working in a Git repository. Use the tools to inspect and change "
    "files; never guess file contents you have not read. Make the smallest correct change, "
    "verify it, and then reply with a short summary."),
    ("You are an expert software engineer operating inside the user's repository through tools. "
    "Investigate before you edit, keep changes minimal, and finish with a concise report."),
]


def _fn(name, description, properties, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}


S = {"type": "string"}
I = {"type": "integer"}
# Two tool vocabularies, so the head does not learn one client's spelling only.
TOOLSETS = {
    "files": [
        _fn("read_file", "Read a file, optionally a line range.",
            {"path": S, "start_line": I, "end_line": I}, ["path"]),
        _fn("list_directory", "List the entries of a directory.", {"path": S}, ["path"]),
        _fn("search", "Search the repository with a regular expression (ripgrep).",
            {"query": S, "glob": S}, ["query"]),
        _fn("replace_in_file", "Replace one exact occurrence of `find` with `replace`.",
            {"path": S, "find": S, "replace": S}, ["path", "find", "replace"]),
        _fn("create_file", "Create or overwrite a file.", {"path": S, "content": S},
            ["path", "content"]),
        _fn("shell", "Run a shell command in the repository root.", {"command": S}, ["command"]),
    ],
    "compact": [
        _fn("read", "Read a file.", {"filePath": S, "offset": I, "limit": I}, ["filePath"]),
        _fn("glob", "Find files by glob pattern.", {"pattern": S}, ["pattern"]),
        _fn("grep", "Search file contents by regex.", {"pattern": S, "include": S}, ["pattern"]),
        _fn("edit", "Edit a file by exact string replacement.",
            {"filePath": S, "oldString": S, "newString": S}, ["filePath", "oldString", "newString"]),
        _fn("write", "Write a file.", {"filePath": S, "content": S}, ["filePath", "content"]),
        _fn("bash", "Execute a bash command.", {"command": S, "description": S}, ["command"]),
    ],
}
API_TOOLS = [
    _fn("get_weather", "Current weather or forecast for a location.",
        {"location": S, "units": {"type": "string", "enum": ["metric", "imperial"]},
         "days": {"type": "integer", "minimum": 0, "maximum": 7}}, ["location"]),
    _fn("schedule_meeting", "Create a calendar event.",
        {"title": S, "start": {"type": "string", "description": "ISO 8601"},
         "duration_minutes": I, "attendees": {"type": "array", "items": S}, "location": S},
        ["title", "start", "duration_minutes"]),
    _fn("send_email", "Send an email.", {"to": {"type": "array", "items": S}, "subject": S,
                                         "body": S, "cc": {"type": "array", "items": S}},
        ["to", "subject", "body"]),
    _fn("query_database", "Run a read-only SQL query against the analytics warehouse.",
        {"sql": S, "limit": I}, ["sql"]),
    _fn("create_ticket", "Open a ticket in the tracker.",
        {"title": S, "description": S, "severity": {"type": "string",
                                                    "enum": ["low", "medium", "high", "critical"]},
         "component": S, "assignee": S}, ["title", "description", "severity"]),
    _fn("search_docs", "Search internal documentation.",
        {"query": S, "product": S, "max_results": I}, ["query"]),
    _fn("convert_currency", "Convert an amount between currencies.",
        {"amount": {"type": "number"}, "from": S, "to": S}, ["amount", "from", "to"]),
    _fn("set_reminder", "Set a reminder.", {"text": S, "at": S, "repeat": {
        "type": "string", "enum": ["none", "daily", "weekly", "monthly"]}}, ["text", "at"]),
    _fn("deploy_service", "Deploy a service version to an environment.",
        {"service": S, "version": S, "environment": {"type": "string",
                                                     "enum": ["dev", "staging", "prod"]},
         "canary_percent": {"type": "integer", "minimum": 0, "maximum": 100}},
        ["service", "version", "environment"]),
    _fn("get_stock_quote", "Latest quote for ticker symbols.",
        {"symbols": {"type": "array", "items": S}}, ["symbols"]),
    _fn("book_flight", "Book a flight.",
        {"origin": S, "destination": S, "date": S, "passengers": I,
         "cabin": {"type": "string", "enum": ["economy", "premium", "business"]}},
        ["origin", "destination", "date"]),
    _fn("update_record", "Update fields of a CRM record.",
        {"record_id": S, "fields": {"type": "object"}}, ["record_id", "fields"]),
]
API_REQUESTS = [
    "What's the weather going to be in Melbourne for the next three days, in Celsius?",
    ("Set up a 45 minute design review called 'Checkout redesign' next Tuesday at 2pm with "
    "priya@example.com and tom@example.com in room Kestrel."),
    ("Email the team (dev-team@example.com) that the release is postponed to Friday because of "
    "a failing migration, and cc ops@example.com."),
    "How many orders did we get per day last week? The table is orders(created_at, total).",
    ("Open a critical ticket: payments API returns 502 for 30% of requests since 10:05, "
    "component payments-gateway, assign it to on-call."),
    "Find our docs about rotating API keys for the Billing product, top 5.",
    "How much is 1,250 US dollars in Australian dollars?",
    "Remind me every Monday at 9am to review the on-call handover notes.",
    "Ship version 2.14.1 of the search-indexer to production as a 10% canary.",
    "Get me quotes for AAPL, MSFT and NVDA.",
    "Book two economy seats from Sydney to Auckland on 2026-12-03.",
    "Update record CUST-88213: set status to churn-risk and owner to Maria Chen.",
    "Is it going to rain in Seattle tomorrow? Use Fahrenheit.",
    "Schedule a 30 minute 1:1 with sam@example.com tomorrow at 10:30.",
    ("Which 10 customers had the highest total spend this year? customers(id, name), "
    "orders(customer_id, total, created_at)."),
    "Deploy auth-service 5.0.0 to staging.",
    "Convert 99.95 euros to Japanese yen.",
    "Create a low-severity ticket for the typo on the pricing page footer, component website.",
    "Send a reminder email to finance@example.com that expense reports are due Thursday.",
    "Set a one-off reminder for 2026-11-02T08:00 to renew the TLS certificates.",
    "Look up the docs on configuring SSO with Okta for the Admin Console product.",
    "What's the average order value by month for 2025? orders(created_at, total).",
    "Book a business class flight for one from London to New York on 2027-01-15.",
    "Mark record LEAD-1029 as qualified with an estimated deal size of 42000.",
]


class Overlay:
    """A repository seen read-only, with writes kept in memory."""

    ALLOWED = frozenset({"ls", "cat", "head", "tail", "wc", "rg", "grep", "find", "git"})
    GIT_ALLOWED = frozenset({"log", "status", "diff", "show", "ls-files", "grep", "blame"})

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}

    def path(self, value):
        target = (self.root / str(value).lstrip("/")).resolve()
        if self.root not in (target, *target.parents):
            raise ValueError(f"{value}: outside the repository")
        return target

    def read(self, value, start=None, end=None):
        target = self.path(value)
        key = str(target)
        if key in self.files:
            text = self.files[key]
        elif target.is_file():
            text = target.read_text(errors="replace")
        else:
            return f"Error: {value} does not exist."
        lines = text.splitlines()
        start = max(1, int(start or 1))
        end = min(len(lines), int(end or start + 149))
        body = "\n".join(f"{n:>5}\t{lines[n - 1]}" for n in range(start, end + 1))
        more = f"\n... ({len(lines) - end} more lines)" if end < len(lines) else ""
        return body + more

    def listing(self, value):
        target = self.path(value or ".")
        if not target.is_dir():
            return f"Error: {value} is not a directory."
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir()
                       if not p.name.startswith("."))
        return "\n".join(names[:200])

    def run(self, argv):
        try:
            done = subprocess.run(argv, cwd=self.root, capture_output=True, text=True,
                                  timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"Error: {error}"
        out = (done.stdout + done.stderr)[:6000]
        return out if out.strip() else f"(no output, exit status {done.returncode})"

    def search(self, pattern, glob=None):
        argv = ["rg", "-n", "--max-count", "5", "--max-columns", "200", "-e", pattern]
        if glob:
            argv += ["-g", glob]
        lines = self.run(argv).splitlines()
        return "\n".join(lines[:40]) or "No matches."

    def replace(self, value, find, replace):
        target = self.path(value)
        current = self.files.get(str(target))
        if current is None:
            if not target.is_file():
                return f"Error: {value} does not exist."
            current = target.read_text(errors="replace")
        if current.count(find) != 1:
            return f"Error: found {current.count(find)} occurrences of the text in {value}."
        self.files[str(target)] = current.replace(find, replace, 1)
        return f"Edited {value}."

    def write(self, value, content):
        self.files[str(self.path(value))] = content
        return f"Wrote {len(content.splitlines())} lines to {value}."

    def shell(self, command):
        try:
            argv = shlex.split(command)
        except ValueError as error:
            return f"Error: {error}"
        if not argv or any(token in {"|", ";", "&&", "||", ">", ">>", "<"} for token in argv):
            return "Error: pipelines and redirection are not available here; run one command."
        if argv[0] not in self.ALLOWED or (argv[0] == "git" and (
                len(argv) < 2 or argv[1] not in self.GIT_ALLOWED)):
            return f"Error: `{argv[0]}` is not available in this read-only sandbox."
        if argv[0] == "find" and any(a in {"-delete", "-exec", "-execdir", "-ok"} for a in argv):
            return "Error: find actions are not available."
        return self.run(argv)

    def call(self, name, args):
        a = args if isinstance(args, dict) else {}
        try:
            if name in {"read_file", "read"}:
                start = a.get("start_line") or a.get("offset")
                end = a.get("end_line")
                if end is None and a.get("limit"):
                    end = int(start or 1) + int(a["limit"]) - 1
                return self.read(a.get("path") or a.get("filePath"), start, end)
            if name == "list_directory":
                return self.listing(a.get("path"))
            if name == "glob":
                return self.run(["rg", "--files", "-g", a.get("pattern", "*")])[:4000]
            if name in {"search", "grep"}:
                return self.search(a.get("query") or a.get("pattern", ""),
                                   a.get("glob") or a.get("include"))
            if name in {"replace_in_file", "edit"}:
                return self.replace(a.get("path") or a.get("filePath"),
                                    a.get("find", a.get("oldString", "")),
                                    a.get("replace", a.get("newString", "")))
            if name in {"create_file", "write"}:
                return self.write(a.get("path") or a.get("filePath"), a.get("content", ""))
            if name in {"shell", "bash"}:
                return self.shell(a.get("command", ""))
        except (ValueError, TypeError, KeyError) as error:
            return f"Error: {error}"
        return f"Error: unknown tool {name}."


def source_files(root):
    listed = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True,
                            check=True).stdout.split()
    return [p for p in listed if Path(p).suffix in SOURCE_SUFFIXES and "test" not in p.lower()
            and "vendor" not in p and "/generated" not in p]


def excerpt(rng, root, files):
    """A 30-120 line excerpt starting at a definition, or None."""
    for _ in range(20):
        rel = rng.choice(files)
        try:
            lines = Path(root, rel).read_text(errors="replace").splitlines()
        except OSError:
            continue
        starts = [i for i, line in enumerate(lines) if DEFINITION.match(line)]
        if not starts:
            continue
        start = rng.choice(starts)
        chunk = lines[start:start + rng.randint(30, 120)]
        if len(chunk) >= 20 and max(map(len, chunk)) < 300:
            return rel, "\n".join(chunk)
    return None


def symbols(rng, root, files, count):
    """(symbol, path) pairs: names defined in the repository."""
    pattern = re.compile(r"\b(?:fn|def|function|class|struct|func)\s+([A-Za-z_][A-Za-z0-9_]{3,})")
    found = []
    for rel in rng.sample(files, min(len(files), 4 * count)):
        try:
            names = pattern.findall(Path(root, rel).read_text(errors="replace"))
        except OSError:
            continue
        if names:
            found.append((rng.choice(names), rel))
        if len(found) >= count:
            break
    return found


def build_jobs(seed, repos, counts):
    rng = random.Random(seed)
    jobs = []
    combos = [(t, l) for t in CODE_TASKS for l in LANGUAGES]
    rng.shuffle(combos)
    for i, (task, lang) in enumerate(combos[:counts["code"]]):
        prompt = rng.choice(CODE_ASKS).format(task=task, lang=lang)
        jobs.append({"id": f"code-{i:04d}", "kind": "code",
                     "messages": [{"role": "user", "content": prompt}]})
    repo_files = {root: source_files(root) for root in repos}
    for i in range(counts["context"]):
        root = rng.choice(repos)
        found = excerpt(rng, root, repo_files[root])
        if not found:
            continue
        rel, text = found
        lang = SOURCE_SUFFIXES[Path(rel).suffix]
        prompt = f"{rng.choice(CONTEXT_ASKS)}\n\n`{rel}`:\n```{lang}\n{text}\n```"
        jobs.append({"id": f"context-{i:04d}", "kind": "context",
                     "messages": [{"role": "user", "content": prompt}]})
    per_repo = max(1, counts["agent"] // len(repos))
    i = 0
    for root in repos:
        for symbol, rel in symbols(rng, root, repo_files[root], per_repo):
            ask = rng.choice(AGENT_ASKS).format(symbol=symbol, path=rel)
            toolset = rng.choice(sorted(TOOLSETS))
            jobs.append({"id": f"agent-{i:04d}", "kind": "agent", "root": str(root),
                         "toolset": toolset,
                         "messages": [{"role": "system", "content": rng.choice(SYSTEMS)},
                                      {"role": "user", "content": ask}]})
            i += 1
    for i, request in enumerate(API_REQUESTS[:counts["api"]]):
        tools = rng.sample(API_TOOLS, 6)
        needed = [t for t in API_TOOLS if t not in tools]
        rng.shuffle(needed)
        jobs.append({"id": f"api-{i:04d}", "kind": "api", "tools": tools + needed[:2],
                     "messages": [{"role": "user", "content": request}]})
    return jobs


def chat(url, messages, tools, max_tokens):
    body = {"model": "ternary-bonsai-2-27b", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "seed": 0}
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def wire(messages):
    """Engine-format history (tool_calls {id, name, arguments}) as OpenAI wire messages."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"],
                                             "arguments": json.dumps(c["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def run_job(url, job, max_tokens, max_turns):
    tools = job.get("tools") or TOOLSETS.get(job.get("toolset"), [])
    messages = list(job["messages"])
    overlay = Overlay(job["root"]) if job["kind"] == "agent" else None
    tokens = 0
    for _ in range(max_turns if overlay else 1):
        reply = chat(url, wire(messages), tools, max_tokens)
        message = reply["choices"][0]["message"]
        tokens += reply.get("usage", {}).get("completion_tokens", 0)
        calls = []
        for call in message.get("tool_calls") or []:
            try:
                args = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": call["id"], "name": call["function"]["name"], "arguments": args})
        turn = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("reasoning_content"):
            turn["reasoning_content"] = message["reasoning_content"]
        if calls:
            turn["tool_calls"] = calls
        messages.append(turn)
        if not overlay or not calls or reply["choices"][0].get("finish_reason") == "length":
            break
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": overlay.call(call["name"], call["arguments"])})
    return {"id": job["id"], "kind": job["kind"], "thinking": True, "tools": tools,
            "messages": messages, "completion_tokens": tokens}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", action="append", required=True,
                    help="git checkout for excerpts and agent tasks (repeatable)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--code", type=int, default=100)
    ap.add_argument("--context", type=int, default=60)
    ap.add_argument("--agent", type=int, default=40)
    ap.add_argument("--api", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=1536, help="per assistant turn")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new jobs")
    ap.add_argument("--list", action="store_true", help="print the jobs and exit")
    args = ap.parse_args()
    jobs = build_jobs(args.seed, [str(Path(r).resolve()) for r in args.repo],
                      {"code": args.code, "context": args.context, "agent": args.agent,
                       "api": args.api})
    if args.list:
        for job in jobs:
            print(json.dumps({"id": job["id"], "prompt": job["messages"][-1]["content"][:160]}))
        return
    out = Path(args.out)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    pending = [job for job in jobs if job["id"] not in done]
    # Interleave kinds so a partial run still covers every kind.
    pending.sort(key=lambda job: (int(job["id"].rsplit("-", 1)[1]), job["kind"]))
    if args.limit:
        pending = pending[: args.limit]
    lock = threading.Lock()
    total = 0
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(run_job, args.url, job, args.max_tokens, args.max_turns): job
                   for job in pending}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 -- one failed job must not lose the others
                print(f"{job['id']}: {error}", flush=True)
                continue
            with lock, open(out, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            total += record["completion_tokens"]
            print(f"{record['id']}: {record['completion_tokens']} tokens, "
                  f"{len(record['messages'])} messages (total {total})", flush=True)


if __name__ == "__main__":
    main()
