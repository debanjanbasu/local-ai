#!/usr/bin/env python3
"""A/B MTP heads on coding and tool-call requests: speed and task success.

Speculation never changes what greedy decoding emits on purpose, but a head
does change how many rows each verify step has, and near-ties can resolve the
other way, so a head comparison checks both sides: speed (MTP acceptance,
accepted drafts per round, decode tok/s, first-token and total latency) and
quality (each task's own check, plus whether the two heads' outputs match).

Tasks are fixed, greedy, and never part of the training rollouts
(`tools/mtp_train/rollouts.py`). Code tasks ask for one function or file and
run the supplied tests against it (python3, rustc, node, go, sqlite3); tool
tasks are coding-agent turns with tool definitions, checked by applying the
call (an edit must apply and the edited code must pass its test) or by its
arguments; two prose prompts are controls.

    lockf /tmp/local-ai-gpu.lock tools/mtp_code_bench.py ab \\
        --head installed=models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin \\
        --head candidate=DIR/mtp-head-ptq1-v1.bin --rounds 2 --out RUN_DIR
    tools/mtp_code_bench.py summary RUN_DIR/*.jsonl

`ab` serves each head in turn (rounds alternate the order, ABBA), from a
scratch root whose `models/` holds the pinned checkpoint and only that head,
so discovery finds exactly the file under test; it never touches the
installed head. Model-written code runs locally with a timeout: run it only
on a machine where that is acceptable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "target" / "release" / "local-ai"
CHECKPOINT_DIR = "bonsai2-27b-ptq1"

# --------------------------------------------------------------------------
# Code tasks: (id, language, prompt, test appended to the extracted code).
# --------------------------------------------------------------------------

ONLY_BLOCK = "Reply with only one ```{lang} code block{extra}."

CODE_TASKS = [
    {
        "id": "py-merge-intervals",
        "lang": "python",
        "prompt": "Write a Python function `merge_intervals(intervals: list[list[int]]) -> "
        "list[list[int]]` that merges overlapping or touching closed intervals and returns "
        "them sorted by start. The input may be unsorted and may be empty.",
        "test": """
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
assert merge_intervals([]) == []
assert merge_intervals([[5, 7], [1, 2]]) == [[1, 2], [5, 7]]
assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
""",
    },
    {
        "id": "py-lru-cache",
        "lang": "python",
        "prompt": "Implement a Python class `LRUCache` with `__init__(self, capacity: int)`, "
        "`get(self, key: int) -> int` returning -1 when absent, and `put(self, key: int, "
        "value: int) -> None`, evicting the least recently used key when over capacity. "
        "Both operations must be O(1); do not use functools.",
        "test": """
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
c.put(3, 30)
assert c.get(3) == 30
""",
    },
    {
        "id": "py-fix-binary-search",
        "lang": "python",
        "prompt": "This function should return the index of `target` in the sorted list "
        "`items`, or -1, but it has bugs. Fix it.\n\n```python\n"
        "def binary_search(items, target):\n"
        "    lo, hi = 0, len(items)\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if items[mid] == target:\n"
        "            return mid\n"
        "        if items[mid] < target:\n"
        "            lo = mid\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n```",
        "extra": " containing the corrected function",
        "test": """
data = [1, 3, 5, 7, 9, 11]
for i, v in enumerate(data):
    assert binary_search(data, v) == i, v
for v in (0, 2, 4, 12):
    assert binary_search(data, v) == -1, v
assert binary_search([], 1) == -1
assert binary_search([4], 4) == 0
""",
    },
    {
        "id": "py-annotate-module",
        "lang": "python",
        "prompt": "Add type hints and a concise docstring to every function in this module "
        "without changing its behaviour, and return the whole module.\n\n```python\n"
        "import re\n\n\n"
        "def tokenize(text):\n"
        "    return re.findall(r\"[a-z0-9']+\", text.lower())\n\n\n"
        "def word_counts(text):\n"
        "    counts = {}\n"
        "    for token in tokenize(text):\n"
        "        counts[token] = counts.get(token, 0) + 1\n"
        "    return counts\n\n\n"
        "def top_words(text, n=3):\n"
        "    counts = word_counts(text)\n"
        "    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))\n"
        "    return [word for word, _ in ranked[:n]]\n\n\n"
        "def summary(text, n=3):\n"
        "    words = top_words(text, n)\n"
        "    total = sum(word_counts(text).values())\n"
        "    return f\"{total} words; top: {', '.join(words)}\"\n```",
        "extra": " containing the whole module",
        "test": """
import inspect
assert tokenize("Don't stop, DON'T!") == ["don't", "stop", "don't"]
assert word_counts("a b a") == {"a": 2, "b": 1}
assert top_words("b a c a b d", 3) == ["a", "b", "c"]
assert summary("x y x") == "3 words; top: x, y"
for fn in (tokenize, word_counts, top_words, summary):
    assert fn.__doc__, fn.__name__
    assert inspect.signature(fn).return_annotation is not inspect.Signature.empty, fn.__name__
""",
    },
    {
        "id": "rs-parse-kv",
        "lang": "rust",
        "prompt": "Write a Rust function `pub fn parse_kv(input: &str) -> "
        "std::collections::HashMap<String, String>` that parses `key=value` pairs separated "
        "by `;`, trims whitespace around keys and values, skips empty segments and segments "
        "without `=`, splits only on the first `=`, and lets later keys overwrite earlier ones.",
        "extra": " containing the function and any `use` lines, without a main function or tests",
        "test": """
#[cfg(test)]
mod bench_tests {
    use super::*;
    #[test]
    fn parses() {
        let m = parse_kv(" a = 1 ;b=2;; c ; d=x=y ; a=3 ");
        assert_eq!(m.len(), 3);
        assert_eq!(m["a"], "3");
        assert_eq!(m["b"], "2");
        assert_eq!(m["d"], "x=y");
        assert!(parse_kv("").is_empty());
    }
}
""",
    },
    {
        "id": "rs-top-k-frequent",
        "lang": "rust",
        "prompt": "Write a Rust function `pub fn top_k_frequent(words: &[&str], k: usize) -> "
        "Vec<String>` returning the k most frequent words, ordered by descending frequency "
        "and, for equal frequency, ascending lexicographic order. If there are fewer than k "
        "distinct words, return all of them.",
        "extra": " containing the function and any `use` lines, without a main function or tests",
        "test": """
#[cfg(test)]
mod bench_tests {
    use super::*;
    #[test]
    fn ranks() {
        assert_eq!(top_k_frequent(&["b", "a", "c", "a", "b", "d"], 3), vec!["a", "b", "c"]);
        assert_eq!(top_k_frequent(&["x", "y", "x"], 5), vec!["x", "y"]);
        assert_eq!(top_k_frequent(&[], 2), Vec::<String>::new());
    }
}
""",
    },
    {
        "id": "js-parse-csv-line",
        "lang": "javascript",
        "prompt": "Write a Node.js function `parseCsvLine(line)` that splits one CSV line into "
        "an array of field strings. Fields may be quoted with double quotes; quoted fields "
        "may contain commas, and a doubled quote inside a quoted field is a literal quote. "
        "Export it with `module.exports = { parseCsvLine };`.",
        "test": r'''
const assert = require('assert');
assert.deepStrictEqual(parseCsvLine('a,b,c'), ['a', 'b', 'c']);
assert.deepStrictEqual(parseCsvLine('"a,1",b'), ['a,1', 'b']);
assert.deepStrictEqual(parseCsvLine('"say ""hi""",x'), ['say "hi"', 'x']);
assert.deepStrictEqual(parseCsvLine('a,,c'), ['a', '', 'c']);
assert.deepStrictEqual(parseCsvLine(''), ['']);
''',
    },
    {
        "id": "go-reverse-words",
        "lang": "go",
        "prompt": "Write a Go function `func ReverseWords(s string) string` that reverses the "
        "order of whitespace-separated words, collapsing runs of whitespace to one space and "
        "trimming the ends, and also reverses the characters of each word, handling Unicode "
        "correctly (reverse runes, not bytes).",
        "extra": " containing `package main`, the function and its imports, without a main "
        "function",
        "test": """
func main() {
	cases := map[string]string{
		"hello world":     "dlrow olleh",
		"  a  bc   ":      "cb a",
		"héllo wörld":     "dlröw olléh",
		"":                "",
	}
	for in, want := range cases {
		if got := ReverseWords(in); got != want {
			panic(fmt.Sprintf("%q: got %q want %q", in, got, want))
		}
	}
}
""",
    },
    {
        "id": "sql-top-earner",
        "lang": "sql",
        "prompt": "Table `employees(id INTEGER, name TEXT, dept TEXT, salary INTEGER)`. Write "
        "one SQLite query returning `dept` and `name` of the highest-paid employee in each "
        "department (on a salary tie, the alphabetically first name), ordered by `dept`.",
        "extra": " containing the query",
        "test": None,
    },
]

# --------------------------------------------------------------------------
# Tool tasks: a coding agent with file and shell tools.
# --------------------------------------------------------------------------


def _fn(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False},
        },
    }


S = {"type": "string"}
AGENT_TOOLS = [
    _fn("read_file", "Read a UTF-8 file in the repository.", {"path": S}, ["path"]),
    _fn("list_dir", "List a directory in the repository.", {"path": S}, ["path"]),
    _fn("grep", "Search file contents with a regular expression; prints path:line:text.",
        {"pattern": S, "path": {"type": "string", "description": "directory or file; "
                                "default the repository root"}}, ["pattern"]),
    _fn("edit_file", "Replace exactly one occurrence of old_string with new_string in a file.",
        {"path": S, "old_string": S, "new_string": S}, ["path", "old_string", "new_string"]),
    _fn("write_file", "Create or overwrite a file with the given content.",
        {"path": S, "content": S}, ["path", "content"]),
    _fn("run_command", "Run a shell command in the repository root and return its output.",
        {"command": S}, ["command"]),
]
AGENT_SYSTEM = (
    "You are a coding agent working in a Git repository. Use the tools to inspect and change "
    "files; never guess file contents you have not read. Make the smallest correct change, "
    "verify it, and then reply with a short summary."
)

CONFIG_PY = '''import json


DEFAULTS = {"timeout": 30, "retries": 3, "verbose": False}


def load_settings(path):
    """Load settings from a JSON file, falling back to DEFAULTS."""
    with open(path) as handle:
        raw = json.load(handle)
    settings = {}
    for key in DEFAULTS:
        settings[key] = raw[key]
    return settings
'''
CONFIG_TEST = """
import json, os, tempfile
_d = tempfile.mkdtemp()
_p = os.path.join(_d, "s.json")
with open(_p, "w") as _f:
    json.dump({"timeout": 5}, _f)
assert load_settings(_p) == {"timeout": 5, "retries": 3, "verbose": False}, load_settings(_p)
with open(_p, "w") as _f:
    json.dump({"timeout": 1, "retries": 0, "verbose": True}, _f)
assert load_settings(_p) == {"timeout": 1, "retries": 0, "verbose": True}
"""
RANK_RS = """use std::collections::HashMap;

pub fn top_k_frequent(words: &[&str], k: usize) -> Vec<String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &word in words {
        *counts.entry(word).or_insert(0) += 1;
    }
    let mut pairs: Vec<(&str, usize)> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs.into_iter().take(k).map(|(word, _)| word.to_string()).collect()
}
"""
RANK_TEST = """
#[cfg(test)]
mod bench_tests {
    use super::*;
    #[test]
    fn ties_alphabetical() {
        for _ in 0..20 {
            assert_eq!(top_k_frequent(&["b", "a", "c", "a", "b", "d"], 3), vec!["a", "b", "c"]);
            assert_eq!(top_k_frequent(&["q", "p", "r"], 2), vec!["p", "q"]);
        }
    }
}
"""
SLUGIFY_TEST = """
assert slugify("Hello, World!") == "hello-world"
assert slugify("  --Rust & Go--  ") == "rust-go"
assert slugify("a__b") == "a-b"
assert slugify("Already-slugged") == "already-slugged"
"""


def _call(id_, name, args):
    return {"id": id_, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


FIX_USER = ("`load_settings` crashes with a KeyError when the settings file leaves out a key. "
            "Missing keys should fall back to DEFAULTS. Please fix it.")
FIX_1 = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": FIX_USER}]
FIX_2 = FIX_1 + [
    {"role": "assistant", "content": "",
     "tool_calls": [_call("call_1", "grep", {"pattern": "def load_settings"})]},
    {"role": "tool", "tool_call_id": "call_1", "content": "app/config.py:7:def load_settings(path):"},
]
FIX_3 = FIX_2 + [
    {"role": "assistant", "content": "",
     "tool_calls": [_call("call_2", "read_file", {"path": "app/config.py"})]},
    {"role": "tool", "tool_call_id": "call_2", "content": CONFIG_PY},
]
FIX_4 = FIX_3 + [
    {"role": "assistant", "content": "",
     "tool_calls": [_call("call_3", "edit_file", {"path": "app/config.py",
                                                  "old_string": "        settings[key] = raw[key]",
                                                  "new_string": "        settings[key] = raw.get(key, DEFAULTS[key])"})]},
    {"role": "tool", "tool_call_id": "call_3", "content": "Edited app/config.py (1 replacement)."},
]
FIX_5 = FIX_4 + [
    {"role": "assistant", "content": "",
     "tool_calls": [_call("call_4", "run_command", {"command": "python -m pytest -q tests/test_config.py"})]},
    {"role": "tool", "tool_call_id": "call_4", "content": "....                                    [100%]\n4 passed in 0.03s"},
]

RANK_MESSAGES = [
    {"role": "system", "content": AGENT_SYSTEM},
    {"role": "user", "content": "`cargo test` fails:\n\n```\n---- rank::tests::ties_alphabetical stdout ----\n"
     "thread 'rank::tests::ties_alphabetical' panicked at src/rank.rs:24:9:\n"
     "assertion `left == right` failed\n  left: [\"b\", \"a\", \"c\"]\n right: [\"a\", \"b\", \"c\"]\n```\n\n"
     "Words with equal counts must come out in alphabetical order. Fix it."},
    {"role": "assistant", "content": "",
     "tool_calls": [_call("call_1", "read_file", {"path": "src/rank.rs"})]},
    {"role": "tool", "tool_call_id": "call_1", "content": RANK_RS},
]

ISSUE_TOOLS = [_fn(
    "create_issue", "File an issue in the project tracker.",
    {"title": S, "body": S,
     "labels": {"type": "array", "items": {"type": "string", "enum": [
         "bug", "enhancement", "docs", "performance", "security"]}},
     "priority": {"type": "integer", "minimum": 1, "maximum": 4,
                  "description": "1 is the most urgent"}},
    ["title", "body", "labels", "priority"])]

TOOL_TASKS = [
    {"id": "tool-find-definition", "messages": FIX_1, "check": "find_definition"},
    {"id": "tool-read-after-grep", "messages": FIX_2, "check": "read_config"},
    {"id": "tool-edit-python", "messages": FIX_3, "check": "edit_config"},
    {"id": "tool-run-tests", "messages": FIX_4, "check": "run_tests"},
    {"id": "tool-final-summary", "messages": FIX_5, "check": "final_summary"},
    {"id": "tool-edit-rust", "messages": RANK_MESSAGES, "check": "edit_rank"},
    {"id": "tool-write-module", "check": "write_slugify", "messages": [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": "Create `app/slugify.py` with a function `slugify(text: str) -> "
         "str` that lowercases the text, replaces every run of characters other than ASCII "
         "letters and digits with a single hyphen, and strips leading and trailing hyphens. "
         "Write the file directly; it does not exist yet."}]},
    {"id": "tool-parallel-reads", "check": "parallel_reads", "messages": [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": "Show me what's in README.md and pyproject.toml."}]},
    {"id": "tool-create-issue", "check": "create_issue", "tools": ISSUE_TOOLS, "messages": [
        {"role": "user", "content": "File an issue: the server leaks memory when a client "
         "disconnects in the middle of a streamed response. It's a performance bug and the "
         "most urgent thing on our list."}]},
]

# Prose controls: a head tuned for code must not slow ordinary answers down.
PROSE_TASKS = [
    {"id": "prose-planets", "prompt": "List the planets of the solar system in order, one per "
     "line, with a one sentence description of each."},
    {"id": "prose-explain-tcp", "prompt": "Explain how TCP congestion control works to a new "
     "engineer, in about 250 words."},
]


# --------------------------------------------------------------------------
# Checks.
# --------------------------------------------------------------------------


def code_blocks(text, lang):
    aliases = {"python": ("python", "py"), "rust": ("rust", "rs"), "javascript": ("javascript", "js"),
               "go": ("go", "golang"), "sql": ("sql", "sqlite")}[lang]
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", text or "", re.S)
    tagged = [body for tag, body in blocks if tag.lower() in aliases]
    return tagged or [body for _, body in blocks]


def run(argv, cwd, timeout=120):
    try:
        done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return done.returncode == 0, (done.stdout + done.stderr)[-600:]


def run_python(source):
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "t.py").write_text(source)
        return run([sys.executable, "t.py"], tmp, timeout=30)


def run_rust(source):
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "t.rs").write_text(source)
        ok, out = run(["rustc", "--edition", "2021", "--test", "-o", "t", "t.rs"], tmp)
        return run(["./t"], tmp, timeout=30) if ok else (False, out)


def check_code(task, content):
    blocks = code_blocks(content, task["lang"])
    if not blocks:
        return False, "no code block"
    code = max(blocks, key=len)
    lang = task["lang"]
    if lang == "python":
        return run_python(code + "\n" + task["test"])
    if lang == "rust":
        return run_rust(code + "\n" + task["test"])
    if lang == "javascript":
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "t.js").write_text(code + "\n" + task["test"])
            return run(["node", "t.js"], tmp, timeout=30)
    if lang == "go":
        body = code if "package main" in code else "package main\n" + code
        if '"fmt"' not in body:
            body = body.replace("package main", 'package main\n\nimport "fmt"', 1)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "t.go").write_text(body + "\n" + task["test"])
            return run(["go", "run", "t.go"], tmp)
    if lang == "sql":
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE employees(id INTEGER, name TEXT, dept TEXT, salary INTEGER)")
        db.executemany("INSERT INTO employees VALUES (?,?,?,?)", [
            (1, "Zoe", "eng", 200), (2, "Adam", "eng", 200), (3, "Bob", "eng", 150),
            (4, "Cara", "ops", 90), (5, "Dan", "ops", 120), (6, "Eve", "art", 70)])
        try:
            rows = db.execute(code).fetchall()
        except sqlite3.Error as error:
            return False, str(error)
        want = [("art", "Eve"), ("eng", "Adam"), ("ops", "Dan")]
        return rows == want, repr(rows)
    raise ValueError(lang)


def calls_of(message):
    calls = []
    for call in message.get("tool_calls") or []:
        try:
            args = json.loads(call["function"]["arguments"])
        except (KeyError, json.JSONDecodeError):
            args = None
        calls.append((call.get("function", {}).get("name"), args))
    return calls


def apply_edit(original, args):
    old = args.get("old_string") if isinstance(args, dict) else None
    if not old or original.count(old) != 1:
        return None
    return original.replace(old, args.get("new_string", ""), 1)


def check_tool(task, message):
    calls = calls_of(message)
    kind = task["check"]
    if any(args is None for _, args in calls):
        return False, "malformed arguments"
    name, args = calls[0] if calls else (None, {})
    if kind == "find_definition":
        ok = (name == "grep" and "load_settings" in args.get("pattern", "")) or (
            name == "run_command" and "load_settings" in args.get("command", ""))
        return ok, repr(calls)
    if kind == "read_config":
        return name == "read_file" and args.get("path", "").endswith("app/config.py"), repr(calls)
    if kind == "edit_config":
        if name == "write_file" and args.get("path", "").endswith("app/config.py"):
            edited = args.get("content")
        elif name == "edit_file" and args.get("path", "").endswith("app/config.py"):
            edited = apply_edit(CONFIG_PY, args)
        else:
            return False, repr(calls)
        if edited is None:
            return False, "edit does not apply: " + repr(args)
        return run_python(edited + "\n" + CONFIG_TEST)
    if kind == "run_tests":
        command = args.get("command", "") if name == "run_command" else ""
        return "pytest" in command or "unittest" in command, repr(calls)
    if kind == "final_summary":
        text = message.get("content") or ""
        return not calls and len(text.strip()) > 20, text[:200]
    if kind == "edit_rank":
        if name == "write_file" and args.get("path", "").endswith("src/rank.rs"):
            edited = args.get("content")
        elif name == "edit_file" and args.get("path", "").endswith("src/rank.rs"):
            edited = apply_edit(RANK_RS, args)
        else:
            return False, repr(calls)
        if edited is None:
            return False, "edit does not apply: " + repr(args)
        return run_rust(edited + "\n" + RANK_TEST)
    if kind == "write_slugify":
        if name != "write_file" or not args.get("path", "").endswith("app/slugify.py"):
            return False, repr(calls)
        return run_python(args.get("content", "") + "\n" + SLUGIFY_TEST)
    if kind == "parallel_reads":
        paths = sorted(a.get("path", "") for n, a in calls if n == "read_file")
        ok = len(paths) == 2 and paths[0].endswith("README.md") and paths[1].endswith("pyproject.toml")
        return ok, repr(calls)
    if kind == "create_issue":
        if name != "create_issue":
            return False, repr(calls)
        labels = args.get("labels")
        ok = (isinstance(args.get("title"), str) and args["title"].strip()
              and isinstance(labels, list) and labels
              and set(labels) <= {"bug", "enhancement", "docs", "performance", "security"}
              and ({"bug", "performance"} & set(labels))
              and args.get("priority") == 1)
        return bool(ok), repr(args)
    raise ValueError(kind)


def all_tasks():
    tasks = [{**task, "kind": "prose", "messages": [{"role": "user", "content": task["prompt"]}]}
             for task in PROSE_TASKS]
    for task in CODE_TASKS:
        prompt = task["prompt"] + "\n\n" + ONLY_BLOCK.format(lang=task["lang"],
                                                            extra=task.get("extra", ""))
        tasks.append({**task, "kind": "code", "messages": [{"role": "user", "content": prompt}]})
    for task in TOOL_TASKS:
        tasks.append({**task, "kind": "tool", "tools": task.get("tools", AGENT_TOOLS)})
    return tasks


# --------------------------------------------------------------------------
# Running.
# --------------------------------------------------------------------------


def post(url, body, timeout=1800):
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_tasks(url, label, out, max_tokens, only=None):
    records = []
    for task in all_tasks():
        if only and not any(re.search(pattern, task["id"]) for pattern in only):
            continue
        body = {"model": "ternary-bonsai-2-27b", "messages": task["messages"],
                "max_tokens": max_tokens, "temperature": 0, "seed": 0}
        if task["kind"] == "tool":
            body["tools"] = task["tools"]
        started = time.monotonic()
        reply = post(url, body)
        wall = time.monotonic() - started
        choice = reply["choices"][0]
        message = choice["message"]
        if task["kind"] == "code":
            ok, detail = check_code(task, message.get("content"))
        elif task["kind"] == "prose":
            text = message.get("content") or ""
            ok, detail = len(text) > 200, text[:200]
        else:
            ok, detail = check_tool(task, message)
        if choice.get("finish_reason") == "length":
            ok, detail = False, "truncated: " + str(detail)
        timings = reply.get("timings", {})
        usage = reply.get("usage", {})
        mtp = reply.get("speculation", {}).get("mtp", {})
        lookup = reply.get("speculation", {}).get("lookup", {})
        completion = usage.get("completion_tokens", 0)
        first = timings.get("first_token_seconds")
        elapsed = timings.get("elapsed_seconds")
        decode = ((completion - 1) / (elapsed - first)
                  if first is not None and elapsed and elapsed > first and completion > 1 else None)
        record = {
            "label": label, "id": task["id"], "kind": task["kind"], "ok": bool(ok),
            "detail": str(detail)[-400:], "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
            "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            "first_token_seconds": first, "elapsed_seconds": elapsed, "wall_seconds": wall,
            "prefill_seconds": timings.get("prefill_seconds"), "decode_tokens_per_second": decode,
            "mtp": mtp, "lookup": lookup,
            "output": {"content": message.get("content"),
                       "reasoning": message.get("reasoning_content"),
                       "tool_calls": [(c["function"]["name"], c["function"]["arguments"])
                                      for c in message.get("tool_calls") or []]},
        }
        records.append(record)
        with open(out, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"{label:>10} {task['id']:<24} ok={int(ok)} tokens={completion:>5} "
              f"decode={decode or 0:6.2f} tok/s "
              f"mtp={mtp.get('accepted_tokens', 0)}/{mtp.get('proposed_tokens', 0)} "
              f"elapsed={elapsed or 0:6.1f}s", flush=True)
    return records


def scratch_root(base, label, head):
    """A root whose models/ holds the pinned checkpoint and only `head`."""
    root = Path(base, "roots", label)
    models = root / "models"
    (models / "bonsai2-27b-mtp").mkdir(parents=True, exist_ok=True)
    checkpoint = models / CHECKPOINT_DIR
    if not checkpoint.exists():
        checkpoint.symlink_to((REPO / "models" / CHECKPOINT_DIR).resolve())
    link = models / "bonsai2-27b-mtp" / "mtp-head-ptq1-v1.bin"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(head).resolve())
    (root / "home").mkdir(exist_ok=True)
    return root


def serve(root, port, log):
    env = dict(os.environ, HOME=str(root / "home"))
    process = subprocess.Popen(
        [str(BINARY), "serve", "--port", str(port), "--no-response-store"],
        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited with {process.returncode}; see {log.name}")
        try:
            # Startup offers no readiness callback; poll /v1/models once a second.
            urllib.request.urlopen(url + "/v1/models", timeout=5).read()
            return process, url
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(1)
    process.terminate()
    raise SystemExit("server did not become ready in 300 s")


def ab(args):
    heads = [spec.split("=", 1) for spec in args.head]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for round_index in range(args.rounds):
        order = heads if round_index % 2 == 0 else heads[::-1]
        for label, head in order:
            root = scratch_root(out, label, head)
            with open(out / f"serve-{label}-r{round_index}.log", "w") as log:
                process, url = serve(root, args.port, log)
                try:
                    run_tasks(url, label, out / f"{label}.jsonl", args.max_tokens, args.only)
                finally:
                    process.terminate()
                    process.wait(timeout=60)


# --------------------------------------------------------------------------
# Summary.
# --------------------------------------------------------------------------


def geomean(values):
    values = [v for v in values if v]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def summary(args):
    records = [json.loads(line) for path in args.files for line in Path(path).read_text().splitlines()]
    labels = list(dict.fromkeys(r["label"] for r in records))
    print("| head | kind | tasks ok | MTP accepted/proposed | accepted per round | decode tok/s "
          "(geomean of per-task best) | median first token s | median elapsed s |")
    print("|---|---|---|---|---|---|---|---|")
    best = {}
    for r in records:
        key = (r["label"], r["id"])
        if key not in best or (r["decode_tokens_per_second"] or 0) > (best[key]["decode_tokens_per_second"] or 0):
            best[key] = r
    for label in labels:
        for kind in ("code", "tool", "prose", "all"):
            rows = [r for r in records if r["label"] == label and (kind == "all" or r["kind"] == kind)]
            if not rows:
                continue
            ids = sorted({r["id"] for r in rows})
            per_task = [best[(label, i)] for i in ids]
            accepted = sum(r["mtp"].get("accepted_tokens", 0) for r in rows)
            proposed = sum(r["mtp"].get("proposed_tokens", 0) for r in rows)
            rounds = sum(r["mtp"].get("rounds", 0) for r in rows)
            ok = sum(r["ok"] for r in per_task)
            print(f"| {label} | {kind} | {ok}/{len(per_task)} | {accepted}/{proposed} = "
                  f"{accepted / max(1, proposed):.1%} | {accepted / max(1, rounds):.2f} | "
                  f"{geomean(r['decode_tokens_per_second'] for r in per_task):.2f} | "
                  f"{median(r['first_token_seconds'] or 0 for r in per_task):.2f} | "
                  f"{median(r['elapsed_seconds'] or 0 for r in per_task):.1f} |")
    if len(labels) >= 2:
        a, b = labels[:2]
        same = differ = 0
        for i in sorted({r["id"] for r in records}):
            if (a, i) in best and (b, i) in best:
                if best[(a, i)]["output"] == best[(b, i)]["output"]:
                    same += 1
                else:
                    differ += 1
        print(f"\noutputs identical {a} vs {b}: {same}/{same + differ}")
        for i in sorted({r["id"] for r in records}):
            pair = [best.get((label, i)) for label in (a, b)]
            if all(pair):
                da, db = (p["decode_tokens_per_second"] or 0 for p in pair)
                ra, rb = (p["mtp"].get("accepted_tokens", 0) / max(1, p["mtp"].get("proposed_tokens", 0))
                          for p in pair)
                same_text = "same" if pair[0]["output"] == pair[1]["output"] else "DIFFERS"
                print(f"  {i:<24} {a} {da:6.2f} tok/s {ra:.1%} ok={int(pair[0]['ok'])} | "
                      f"{b} {db:6.2f} tok/s {rb:.1%} ok={int(pair[1]['ok'])} | {same_text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("ab", help="serve each head in turn and run every task")
    a.add_argument("--head", action="append", required=True, help="LABEL=PATH to a head file")
    a.add_argument("--rounds", type=int, default=2)
    a.add_argument("--out", required=True)
    a.add_argument("--port", type=int, default=8080)
    a.add_argument("--max-tokens", type=int, default=4096)
    a.add_argument("--only", action="append", help="regex over task ids")
    r = sub.add_parser("run", help="run every task against a running server")
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--only", action="append")
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    args = parser.parse_args()
    if args.command == "ab":
        ab(args)
    elif args.command == "run":
        run_tasks(args.url, args.label, args.out, args.max_tokens, args.only)
    else:
        summary(args)


if __name__ == "__main__":
    main()
