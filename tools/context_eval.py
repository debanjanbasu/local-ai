#!/usr/bin/env python3
"""Bounded long-context coding evaluation for `local-ai serve`.

One seed fixes nine coding questions ("needles"): at each of three positions
in a synthetic Python repository (start, middle, end) it plants

- `lookup`: a pricing function whose constants come from a config module;
  the question asks for its value at given arguments, while similarly named
  legacy decoys with other constants are scattered through the repository;
- `recall`: the same production fee constant asked for directly, which
  separates finding the code from the arithmetic `lookup` also needs;
- `fix`: a function whose implementation contradicts its docstring; the
  answer must be the corrected line, which is substituted into the function
  and executed against hidden test cases.

The context size only changes how much seeded filler surrounds the same
needles, so a seed's short-context control (the same needles plus six filler
files, about 3K tokens) asks exactly the same questions. The repository goes in the system message
and each question in its own user turn, so every question after the first
reuses the repository prefix from the server's prompt cache; the first
request of each (size, seed, effort) group pays the cold prefill. The
template puts the xhigh reasoning instruction ahead of the system message, so
the two efforts do not share a prefix and each pays its own prefill.

Requests go through non-streamed Chat Completions with temperature 0, which
reports usage, cached tokens and server-side timings. Results are appended to
a JSONL file and finished items are skipped on rerun, so a long matrix can be
split over several bounded runs of the shared GPU lock:

    lockf -k /tmp/local-ai-gpu.lock tools/context_eval.py --sizes 8k --seeds 0,1,2
    lockf -k /tmp/local-ai-gpu.lock tools/context_eval.py --sizes 32k --seeds 0,1
    tools/context_eval.py --report            # tables from the JSONL, no server
    tools/context_eval.py --dry-run --sizes 8k  # write prompts, no server

Sizes are nominal; the measured prompt tokens are recorded per request.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from client_compat import ROOT, LocalServer

POSITIONS = {"start": 0.04, "middle": 0.5, "end": 0.94}
SIZES = {"short": 0, "8k": 8_000, "32k": 32_000, "64k": 64_000, "128k": 128_000, "250k": 250_000}
CHARS_PER_TOKEN = 3.84  # measured with `local-ai bonsai --raw --tokenize`; prompt_tokens is recorded
MAX_TOKENS = {"none": 384, "xhigh": 4096}
WORDS = [
    "alpha", "beta", "gamma", "delta", "ledger", "audit", "cache", "queue", "route", "shard",
    "token", "parser", "index", "batch", "vector", "window", "budget", "quota", "tenant", "region",
    "invoice", "refund", "rebate", "tariff", "carrier", "parcel", "depot", "fleet", "sensor",
    "metric", "signal", "sample", "buffer", "stream", "cursor", "bucket", "schema", "record",
    "report", "export", "import",
]
TOPICS = ["parcel", "tenant", "invoice", "carrier", "sensor", "depot", "quota", "refund", "ledger"]


@dataclass(frozen=True)
class Needle:
    task: str  # lookup | fix
    position: str
    files: dict[str, str]
    question: str
    expected: str
    function: str = ""
    buggy_line: str = ""
    tests: tuple = ()


# Each bug: (stem, docstring, signature args, correct lines, index of buggy line, buggy line, tests)
BUGS = [
    ("sum_even", "Return the sum of the first n positive even numbers (2 + 4 + ... + 2n).", "n",
     ["total = 0", "for i in range(1, n + 1):", "    total += 2 * i", "return total"], 1,
     "for i in range(1, n):", [((1,), 2), ((3,), 12), ((5,), 30)]),
    ("count_at_least", "Return how many readings are at least threshold (greater than or equal).", "readings, threshold",
     ["return sum(1 for value in readings if value >= threshold)"], 0,
     "return sum(1 for value in readings if value > threshold)", [(([1, 5, 5, 9], 5), 3), (([2, 3], 3), 1)]),
    ("tail_items", "Return the last n items of the list, or an empty list when n is 0.", "items, n",
     ["if n == 0:", "    return []", "return items[-n:]"], 2,
     "return items[n:]", [(([1, 2, 3, 4], 1), [4]), (([1, 2, 3, 4], 3), [2, 3, 4])]),
    ("share_percent", "Return part as a percentage of whole, rounded to two decimals.", "part, whole",
     ["return round(part * 100 / whole, 2)"], 0,
     "return round(whole * 100 / part, 2)", [((1, 4), 25.0), ((3, 8), 37.5)]),
    ("is_leap", "Return True for Gregorian leap years: divisible by 4, except centuries not divisible by 400.", "year",
     ["return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)"], 0,
     "return year % 4 == 0 and (year % 100 == 0 or year % 400 == 0)", [((2024,), True), ((1900,), False), ((2000,), True), ((2023,), False)]),
    ("word_total", "Return the number of whitespace-separated words, ignoring repeated spaces.", "text",
     ["return len(text.split())"], 0,
     'return len(text.split(" "))', [(("a  b c",), 3), (("one",), 1), ((" x y ",), 2)]),
]


def suffix(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghjkmnpqrstuvwxyz") for _ in range(3)) + str(rng.randrange(10, 99))


def make_needles(seed: int) -> list[Needle]:
    """The six seeded tasks; independent of context size."""
    rng = random.Random(f"needles-{seed}")
    needles: list[Needle] = []
    bugs = rng.sample(BUGS, 3)
    topics = rng.sample(TOPICS, 3)
    for (position, _), topic, bug in zip(POSITIONS.items(), topics, bugs, strict=True):
        tag = suffix(rng)
        fee, factor, cap = rng.randrange(11, 97), rng.randrange(2, 9), rng.randrange(400, 900)
        units, weight = rng.randrange(5, 40), rng.randrange(2, 7)
        expected = min(units * factor + fee * weight, cap)
        config = f"config/{topic}_rates_{tag}.py"
        service = f"services/{topic}_pricing_{tag}.py"
        module = config[:-3].replace("/", ".")
        files = {
            config: f'"""Production {topic} rates."""\n\n{topic.upper()}_FEE = {fee}\n{topic.upper()}_FACTOR = {factor}\n{topic.upper()}_CAP = {cap}\n',
            service: (
                f"from {module} import {topic.upper()}_FEE, {topic.upper()}_FACTOR, {topic.upper()}_CAP\n\n\n"
                f"def {topic}_price(units, weight):\n"
                f'    """Price a {topic} order: units times factor plus fee times weight, capped."""\n'
                f"    raw = units * {topic.upper()}_FACTOR + {topic.upper()}_FEE * weight\n"
                f"    return min(raw, {topic.upper()}_CAP)\n"
            ),
        }
        needles.append(Needle(
            "lookup", position, files,
            f"Using the production implementation in {service} and the constants it actually imports "
            f"(not the legacy modules with similar names), compute {topic}_price({units}, {weight}). "
            "Reply with only the integer.",
            str(expected),
        ))
        # Pure retrieval of the same planted constant (no arithmetic): it
        # separates finding the code from computing with it. No files of its
        # own; the lookup needle plants them.
        needles.append(Needle(
            "recall", position, {},
            f"What value does the configuration module imported by {service} assign to "
            f"{topic.upper()}_FEE? Ignore the legacy modules. Reply with only the integer.",
            str(fee),
        ))
        stem, doc, args, lines, index, buggy, tests = bug
        name = f"{stem}_{tag}"
        body = list(lines)
        body[index] = buggy
        path = f"lib/{topic}_utils_{tag}.py"
        source = f"def {name}({args}):\n    \"\"\"{doc}\"\"\"\n" + "".join(f"    {line}\n" for line in body)
        needles.append(Needle(
            "fix", position, {path: source},
            f"The function {name} in {path} does not do what its docstring says because exactly one line "
            "is wrong. Reply with only that one corrected line of Python code, nothing else.",
            lines[index].strip(), name, buggy.strip(), tuple(tests),
        ))
    return needles


def filler_file(rng: random.Random, index: int, decoy_topics: list[str]) -> tuple[str, str]:
    """A plausible module; some are legacy decoys of the lookup needles."""
    if decoy_topics and rng.random() < 0.18:
        topic = rng.choice(decoy_topics)
        version = rng.randrange(1, 40)
        fee, factor, cap = rng.randrange(1, 99), rng.randrange(2, 12), rng.randrange(100, 999)
        name = f"legacy/{topic}_pricing_v{version}_{index}.py"
        return name, (
            f'"""Deprecated {topic} pricing kept for old invoices."""\n\n'
            f"{topic.upper()}_FEE = {fee}\n{topic.upper()}_FACTOR = {factor}\n{topic.upper()}_CAP = {cap}\n\n\n"
            f"def {topic}_price(units, weight):\n"
            f"    raw = units * {topic.upper()}_FACTOR - {topic.upper()}_FEE + weight\n"
            f"    return min(raw, {topic.upper()}_CAP)\n"
        )
    a, b, c = rng.sample(WORDS, 3)
    name = f"pkg/{a}/{b}_{c}_{index}.py"
    lines = [f'"""Helpers for {a} {b} handling."""', "", f"{b.upper()}_LIMIT = {rng.randrange(1, 500)}", ""]
    for _ in range(rng.randrange(2, 4)):
        fn = f"{rng.choice(WORDS)}_{rng.choice(WORDS)}_{rng.randrange(100)}"
        op = rng.choice(["+", "-", "*"])
        lines += [
            "", f"def {fn}(values, scale={rng.randrange(1, 9)}):",
            f'    """Combine {rng.choice(WORDS)} values for the {rng.choice(WORDS)} report."""',
            "    result = []", "    for value in values:",
            f"        if value > {b.upper()}_LIMIT:", "            continue",
            f"        result.append(value {op} scale)", "    return result",
        ]
    if rng.random() < 0.5:
        cls = f"{a.title()}{c.title()}{index}"
        lines += ["", "", f"class {cls}:", f'    """Track {c} state for {a}."""', "",
                  "    def __init__(self):", "        self.items = {}", "",
                  "    def add(self, key, value):", "        self.items[key] = self.items.get(key, 0) + value",
                  "        return self.items[key]"]
    return name, "\n".join(lines) + "\n"


def build_context(seed: int, size: str, needles: list[Needle]) -> tuple[str, dict]:
    """The repository text and where each needle landed (char fraction)."""
    target_chars = int(SIZES[size] * CHARS_PER_TOKEN)
    rng = random.Random(f"filler-{seed}-{size}")
    topics = sorted({re.match(r"config/(\w+?)_rates", p).group(1) for n in needles for p in n.files if p.startswith("config/")})
    filler: list[tuple[str, str]] = []
    total = 0
    minimum = 6 if size == "short" else 0
    while total < target_chars or len(filler) < minimum:
        name, text = filler_file(rng, len(filler), topics)
        filler.append((name, text))
        total += len(text) + len(name) + 12
    by_position = {p: [n for n in needles if n.position == p] for p in POSITIONS}
    insert_at = {p: min(len(filler), round(f * len(filler))) for p, f in POSITIONS.items()}
    parts: list[str] = []
    offsets: dict[str, float] = {}
    for index in range(len(filler) + 1):
        for position, at in insert_at.items():
            if at == index:
                offsets[position] = sum(len(x) for x in parts)
                for needle in by_position[position]:
                    parts.extend(f"FILE {path}\n{text}\n" for path, text in needle.files.items())
        if index < len(filler):
            parts.append(f"FILE {filler[index][0]}\n{filler[index][1]}\n")
    header = (
        "You are answering questions about the following Python repository. Each file starts with a "
        "line 'FILE <path>'. Files under legacy/ are deprecated and not used in production.\n\n"
    )
    text = header + "".join(parts)
    length = len(text)
    return text, {p: round((o + len(header)) / length, 4) for p, o in offsets.items()}


# ---------------------------------------------------------------- scoring


def extract_code(answer: str) -> str:
    fenced = re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
    return (fenced[0] if fenced else answer).strip("\n")


def score_fix(needle: Needle, answer: str) -> tuple[bool, str]:
    """Substitute the answer for the buggy line and run the hidden tests."""
    code = extract_code(answer)
    if f"def {needle.function}" in code:
        source = code[code.index(f"def {needle.function}"):]
    else:
        line = next((x.strip() for x in code.splitlines() if x.strip() and not x.strip().startswith("#")), "")
        line = line.strip("`")
        if not line:
            return False, "empty answer"
        original = next(text for text in needle.files.values())
        source = None
        for raw in original.splitlines():
            if raw.strip() == needle.buggy_line:
                indent = raw[: len(raw) - len(raw.lstrip())]
                source = original.replace(raw, indent + line, 1)
                break
        assert source is not None
    unsafe = unsafe_construct(source)
    if unsafe:
        return False, unsafe if unsafe.startswith("SyntaxError") else f"rejected unsafe answer: {unsafe}"
    return run_isolated(needle, source)


# Model output is code; it is only ever run after an AST allow-check, in a
# separate isolated interpreter with a CPU limit and a wall-clock timeout.
UNSAFE_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview", "type", "object",
}
UNSAFE_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.While, ast.Try, ast.With,
                ast.AsyncFunctionDef, ast.ClassDef, ast.Delete, ast.Await, ast.Yield, ast.YieldFrom)
SCORER = """
import json, sys
spec = json.load(sys.stdin)
namespace = {}
exec(compile(spec["source"], "<answer>", "exec"), namespace)
function = namespace[spec["function"]]
for args, expected in spec["tests"]:
    got = function(*args)
    if got != expected:
        print(json.dumps([False, f"{spec['function']}{tuple(args)} -> {got!r}, expected {expected!r}"]))
        sys.exit()
print(json.dumps([True, "tests pass"]))
"""


def unsafe_construct(source: str) -> str | None:
    """Why `source` may not be executed, or None when it uses only plain code."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return f"SyntaxError: {error.msg}"
    for node in ast.walk(tree):
        if isinstance(node, UNSAFE_NODES):
            return type(node).__name__
        if isinstance(node, ast.Name) and node.id in UNSAFE_CALLS:
            return node.id
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return node.attr
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and abs(node.value) > 10**7:
            return "huge constant"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            return "power"
    return None


def _limit_cpu() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0)) if sys.platform != "darwin" else None


def run_isolated(needle: Needle, source: str) -> tuple[bool, str]:
    spec = json.dumps({"source": source, "function": needle.function, "tests": [[list(a), e] for a, e in needle.tests]})
    with tempfile.TemporaryDirectory() as scratch:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", SCORER], input=spec, capture_output=True, text=True,
                timeout=15, cwd=scratch, env={"PATH": "/usr/bin:/bin"}, preexec_fn=_limit_cpu, check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    try:
        ok, why = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        error = (completed.stderr.strip().splitlines() or ["no output"])[-1]
        return False, error[:200]
    return bool(ok), why


def score_lookup(needle: Needle, answer: str) -> tuple[bool, str]:
    numbers = re.findall(r"-?\d+", answer.replace(",", ""))
    if not numbers:
        return False, "no integer"
    return numbers[-1] == needle.expected, f"last integer {numbers[-1]}, expected {needle.expected}"


def score(needle: Needle, answer: str) -> tuple[bool, str]:
    return score_fix(needle, answer) if needle.task == "fix" else score_lookup(needle, answer)  # lookup, recall


# ---------------------------------------------------------------- running


def chat(port: int, system: str, question: str, effort: str, seed: int) -> dict:
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "reasoning_effort": effort, "temperature": 0, "seed": seed, "max_tokens": MAX_TOKENS[effort],
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=14_400) as response:
            payload = json.loads(response.read())
            status = response.status
    except urllib.error.HTTPError as error:
        payload, status = json.loads(error.read() or b"{}"), error.code
    payload["_wall_seconds"] = time.monotonic() - started
    payload["_status"] = status
    return payload


def record(size: str, seed: int, effort: str, needle: Needle, payload: dict, concurrency: int, where: dict) -> dict:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    answer = message.get("content") or ""
    ok, why = score(needle, answer) if payload.get("_status") == 200 else (False, f"HTTP {payload.get('_status')}: {payload.get('error')}")
    usage = payload.get("usage") or {}
    timings = payload.get("timings") or {}
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    prefill = timings.get("prefill_seconds") or 0
    first = timings.get("first_token_seconds") or 0
    elapsed = timings.get("elapsed_seconds") or 0
    return {
        "size": size, "seed": seed, "effort": effort, "task": needle.task, "position": needle.position,
        "needle_fraction": where.get(needle.position), "passed": ok, "why": why, "expected": needle.expected,
        "answer": answer[-600:], "finish_reason": choice.get("finish_reason"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "prompt_tokens": prompt, "cached_tokens": cached,
        "cache_source": (usage.get("prompt_tokens_details") or {}).get("cache_source"),
        "completion_tokens": completion, "prefill_seconds": prefill, "first_token_seconds": first,
        "elapsed_seconds": elapsed, "wall_seconds": payload.get("_wall_seconds"),
        "prefill_tok_s": round((prompt - cached) / prefill, 2) if prefill else None,
        "decode_tok_s": round(completion / (elapsed - first), 2) if elapsed > first and completion > 1 else None,
        "concurrency": concurrency, "finished_at": time.time(),
    }


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        keys.add((row["size"], row["seed"], row["effort"], row["task"], row["position"]))
    return keys


def run_group(port: int, out: Path, size: str, seed: int, effort: str, concurrency: int) -> None:
    needles = make_needles(seed)
    results = out / "results.jsonl"
    finished = done_keys(results)
    pending = [n for n in needles if (size, seed, effort, n.task, n.position) not in finished]
    if not pending:
        return
    # The short control holds the same six needles with only a few filler files.
    system, where = build_context(seed, size, needles)
    (out / "prompts").mkdir(exist_ok=True)
    (out / "prompts" / f"{size}-seed{seed}.txt").write_text(system)
    jobs = [(n, system, where) for n in pending]

    def run(job, width):
        needle, system, where = job
        row = record(size, seed, effort, needle, chat(port, system, needle.question, effort, seed), width, where)
        with results.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"{size:>5} seed{seed} {effort:5} {needle.task:6} {needle.position:6} "
              f"{'PASS' if row['passed'] else 'FAIL'} prompt={row['prompt_tokens']} cached={row['cached_tokens']} "
              f"wall={row['wall_seconds']:.1f}s out={row['completion_tokens']} {row['why'][:80]}", flush=True)

    # The first request pays the cold prefill alone; the rest reuse its prefix.
    run(jobs[0], 1)
    width = max(1, concurrency)
    with concurrent.futures.ThreadPoolExecutor(width) as pool:
        list(pool.map(lambda job: run(job, width), jobs[1:]))


def report(out: Path) -> str:
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()] if (out / "results.jsonl").exists() else []
    order = list(SIZES)
    lines = ["| Size | Effort | Seeds | Prompt tokens | Recall | Lookup | Fix | Start | Middle | End | All | Cold prefill tok/s | Decode tok/s (median) | Median wall s |",
             "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["size"], row["effort"]), []).append(row)

    def frac(items):
        return f"{sum(r['passed'] for r in items)}/{len(items)}" if items else "-"

    def median(values):
        values = sorted(v for v in values if v is not None)
        return values[len(values) // 2] if values else None

    for (size, effort), items in sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        seeds = ",".join(str(s) for s in sorted({r["seed"] for r in items}))
        prompts = [r["prompt_tokens"] for r in items]
        cold = [r["prefill_tok_s"] for r in items if r["prompt_tokens"] - r["cached_tokens"] > 1000]
        decode = median([r["decode_tok_s"] for r in items])
        wall = median([r["wall_seconds"] for r in items])
        lines.append(
            f"| {size} | {effort} | {seeds} | {min(prompts):,}–{max(prompts):,} | "
            + " | ".join(frac([r for r in items if r["task"] == t]) for t in ("recall", "lookup", "fix")) + " | "
            + " | ".join(frac([r for r in items if r["position"] == p]) for p in POSITIONS)
            + f" | {frac(items)} | {', '.join(f'{c:.0f}' for c in cold) or '-'} | "
            f"{decode if decode is not None else '-'} | {wall:.0f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server-bin", type=Path, default=ROOT / "target/release/local-ai")
    parser.add_argument("--port", type=int, default=8107)
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--sizes", default="short,8k")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--efforts", default="none,xhigh")
    parser.add_argument("--concurrency", type=int, default=3, help="parallel follow-up questions per group")
    parser.add_argument("--out", type=Path, default=ROOT / ".scratch/artifacts/context-eval")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report:
        print(report(args.out), end="")
        return 0
    sizes = [s for s in args.sizes.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    efforts = [e for e in args.efforts.split(",") if e]
    for size in sizes:
        if size not in SIZES:
            parser.error(f"unknown size {size}")
    if args.dry_run:
        for size in sizes:
            for seed in seeds:
                text, where = build_context(seed, size, make_needles(seed))
                print(f"{size} seed{seed}: {len(text):,} chars, needles at {where}")
        return 0
    server = None
    try:
        if not args.no_serve:
            server = LocalServer(args.server_bin, args.port, args.out / "server.log", [])
            server.start()
        for size in sizes:
            for seed in seeds:
                for effort in efforts:
                    run_group(args.port, args.out, size, seed, effort, args.concurrency)
    finally:
        if server:
            server.stop()
    print(report(args.out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
