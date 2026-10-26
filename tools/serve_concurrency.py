#!/usr/bin/env python3
"""Measure `local-ai serve` under concurrent streams, with and without a long prefill.

Starts the server itself (scratch HOME, so the prompt cache starts cold and
the user's cache is untouched), runs the phases below over streaming Chat
Completions, stops it, and writes `results.json` and `summary.md`. The model
is loaded by `serve`, so hold the shared GPU lock for the whole run:

    lockf -k /tmp/local-ai-gpu.lock tools/serve_concurrency.py --out DIR

Phases, for each stream count N (default 1, 2, 4, 8):

- `decode`: N short prompts at once, each asking for a long answer; aggregate
  and per-stream tok/s, TTFT and the gap between a stream's engine steps.
- `mixed`: N-1 short prompts decoding, then a long prompt (a prefix of
  docs/BONSAI.md, about 6.6K tokens) arriving `--long-delay` seconds later.
  Reports its TTFT, and for the decoding streams the gaps while it prefilled
  (p50, p99 and the longest stall) and their tok/s during the prefill against
  before it. With N = 1 the long prompt runs alone.
- `late`: the long prompt first, alone, and N-1 one-token requests
  arriving 2.3 s apart while it prefills; reports their TTFT (N > 1).
- `shared` (not run by default): N requests 0.3 s apart sharing that long
  prompt as their system message, none cached, each with its own question,
  as sub-agents fanned out at once; reports their TTFTs.

Greedy sampling with reasoning off, so outputs are comparable across builds:
`outputs` in `results.json` maps each request to a hash of its text.
`--sampled` uses the server's default sampling (temperature 1.0, top-p 0.95,
top-k 20) instead, with a fixed seed per request, so outputs stay comparable. A gap is
the wait between engine steps that streamed a stream text: chunks within 5 ms
of each other (the tokens of one step) count once. `--report DIR` recomputes
the summary from a run's saved timings.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BURST = 0.005
TOPICS = [
    "how a CPU cache hierarchy works",
    "the history of the printing press",
    "how vaccines train the immune system",
    "why the sky is blue and sunsets are red",
    "how public-key cryptography works",
    "the water cycle and its effect on climate",
    "how a compiler turns source code into machine code",
    "the causes of the French Revolution",
]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


class Server:
    def __init__(self, binary: Path, port: int, log: Path, home: Path):
        self.binary, self.port, self.log, self.home = binary, port, log, home
        self.process: subprocess.Popen | None = None
        self.ready = threading.Event()

    def start(self) -> None:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"port {self.port} is already in use")
        env = {"HOME": str(self.home), "PATH": "/usr/bin:/bin"}
        self.process = subprocess.Popen(
            [str(self.binary), "serve", "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        threading.Thread(target=self._pump, daemon=True).start()
        # The listening line is the server's only readiness signal.
        if not self.ready.wait(900) or self.process.poll() is not None:
            self.stop()
            raise RuntimeError(f"server did not start; see {self.log}")

    def _pump(self) -> None:
        assert self.process and self.process.stderr
        with self.log.open("a") as log:
            for line in self.process.stderr:
                log.write(line)
                log.flush()
                if "server listening on http://" in line:
                    self.ready.set()
        self.ready.set()

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


# Request fields that choose sampling; `--sampled` replaces them.
SAMPLING: dict = {"temperature": 0}


def stream(port: int, name: str, prompt: str | list, max_tokens: int, start: float,
           out: dict) -> None:
    """One streaming request (a user message, or the messages themselves);
    records chunk arrival times relative to `start`."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    body = json.dumps({
        "model": "ternary-bonsai-2-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        **SAMPLING,
        "reasoning_effort": "none",
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", body,
        {"Content-Type": "application/json"},
    )
    sent = time.monotonic() - start
    chunks: list[float] = []
    sizes: list[int] = []
    text: list[str] = []
    usage = None
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                piece = choice.get("delta", {}).get("content") or ""
                if piece:
                    chunks.append(time.monotonic() - start)
                    sizes.append(len(piece))
                    text.append(piece)
    out[name] = {
        "sent": sent,
        "chunks": chunks,
        "sizes": sizes,
        "done": time.monotonic() - start,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "text_sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16],
        "text_head": "".join(text)[:80],
    }


def run_phase(port: int, requests: list[tuple[str, str | list, int, float]]) -> dict:
    """Start each (name, prompt, max_tokens, delay) request after its delay."""
    out: dict = {}
    start = time.monotonic()
    threads = []
    for name, prompt, max_tokens, delay in requests:
        def worker(name=name, prompt=prompt, max_tokens=max_tokens, delay=delay):
            time.sleep(delay)
            stream(port, name, prompt, max_tokens, start, out)
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return out


def document(chars: int) -> str:
    return (ROOT / "docs" / "BONSAI.md").read_text()[:chars]


def short_prompt(tag: str, index: int) -> str:
    topic = TOPICS[index % len(TOPICS)]
    return f"[{tag}-{index}] Write a detailed, 800-word essay explaining {topic}."


def long_prompt(tag: str, chars: int) -> str:
    return (f"[{tag}] Here is a design document.\n\n{document(chars)}\n\n"
            "In one sentence, what is this document about?")


def shared_prompt(tag: str, index: int, chars: int) -> list:
    return [{"role": "system", "content": f"[{tag}] You answer questions about this design "
                                          f"document.\n\n{document(chars)}"},
            {"role": "user", "content": f"Question {index}: in one sentence, what does the "
                                        f"document say about topic number {index + 1}?"}]


def arrivals(times: list[float]) -> list[float]:
    """Chunk times with each burst kept once: the tokens one engine step
    commits (accepted drafts) stream as separate chunks within a millisecond."""
    return [t for i, t in enumerate(times) if i == 0 or t - times[i - 1] > BURST]


def summarize_decode(n: int, out: dict) -> dict:
    tokens = sum(r["completion_tokens"] or 0 for r in out.values())
    begin = min(r["sent"] for r in out.values())
    end = max(r["done"] for r in out.values())
    gaps = [b - a for r in out.values()
            for a, b in zip(arrivals(r["chunks"]), arrivals(r["chunks"])[1:])]
    rates = [(r["completion_tokens"] or 0) / (r["chunks"][-1] - r["chunks"][0])
             for r in out.values() if len(r["chunks"]) > 1]
    return {
        "streams": n,
        "aggregate_tok_s": tokens / (end - begin),
        "per_stream_tok_s": statistics.mean(rates) if rates else float("nan"),
        "ttft_p50_s": statistics.median(r["chunks"][0] - r["sent"] for r in out.values()),
        "gap_p50_ms": 1000 * percentile(gaps, 0.5),
        "gap_p99_ms": 1000 * percentile(gaps, 0.99),
    }


def summarize_mixed(n: int, out: dict) -> dict:
    long = out["long"]
    arrive, first = long["sent"], long["chunks"][0]
    decoders = [r for name, r in out.items() if name != "long"]
    result = {
        "streams": n,
        "long_prompt_tokens": long["prompt_tokens"],
        "long_ttft_s": first - arrive,
        "long_prefill_tok_s": (long["prompt_tokens"] or 0) / (first - arrive),
    }
    if not decoders:
        return result
    during, stalls, before_tokens, during_tokens = [], [], 0.0, 0.0
    for r in decoders:
        times = r["chunks"]
        # Tokens are apportioned to chunks by their share of the text, since a
        # speculative step can stream several tokens in one chunk.
        per_char = (r["completion_tokens"] or 0) / max(sum(r["sizes"]), 1)
        before_tokens += per_char * sum(s for t, s in zip(times, r["sizes"]) if t < arrive)
        during_tokens += per_char * sum(s for t, s in zip(times, r["sizes"])
                                        if arrive <= t < first)
        # Every wait for a next step that overlaps the prefill; the stall is
        # the longest of them.
        steps = arrivals(times)
        gaps = [b - a for a, b in itertools.pairwise(steps) if b > arrive and a < first]
        during += gaps
        if gaps:
            stalls.append(max(gaps))
    first_decode = min(r["chunks"][0] for r in decoders if r["chunks"])
    total = sum(r["completion_tokens"] or 0 for r in out.values())
    begin = min(r["sent"] for r in out.values())
    end = max(r["done"] for r in out.values())
    result.update({
        "decoders_ttft_p50_s": statistics.median(r["chunks"][0] - r["sent"] for r in decoders),
        "decode_tok_s_before": before_tokens / max(arrive - first_decode, 1e-9),
        "decode_tok_s_during": during_tokens / (first - arrive),
        "gap_during_p50_ms": 1000 * percentile(during, 0.5),
        "gap_during_p99_ms": 1000 * percentile(during, 0.99),
        "max_stall_ms": 1000 * max(stalls) if stalls else float("nan"),
        "aggregate_tok_s": total / (end - begin),
    })
    return result


def summarize_late(n: int, out: dict) -> dict:
    long = out["long"]
    late = [r["chunks"][0] - r["sent"] for name, r in out.items() if name != "long"]
    return {
        "streams": n,
        "long_ttft_s": long["chunks"][0] - long["sent"],
        "late_ttft_p50_s": statistics.median(late),
        "late_ttft_max_s": max(late),
    }


def summarize_shared(n: int, out: dict) -> dict:
    ttft = sorted(r["chunks"][0] - r["sent"] for r in out.values())
    return {
        "streams": n,
        "prompt_tokens": max(r["prompt_tokens"] or 0 for r in out.values()),
        "ttft_first_s": ttft[0],
        "ttft_mean_s": statistics.mean(ttft),
        "ttft_last_s": ttft[-1],
        "all_done_s": max(r["done"] for r in out.values()) - min(r["sent"] for r in out.values()),
    }


SUMMARIES = {"decode": summarize_decode, "mixed": summarize_mixed, "late": summarize_late,
             "shared": summarize_shared}


def report(out_dir: Path) -> None:
    """Summarize every phase file a run saved in `out_dir`."""
    results: dict = {phase: [] for phase in SUMMARIES}
    results["outputs"] = {}
    for phase, summarize in SUMMARIES.items():
        files = sorted(out_dir.glob(f"{phase}[0-9]*.json"), key=lambda f: int(f.stem[len(phase):]))
        for path in files:
            out = json.loads(path.read_text())
            results[phase].append(summarize(int(path.stem[len(phase):]), out))
            for name, r in out.items():
                results["outputs"][f"{path.stem}/{name}"] = r["text_sha"]
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    lines = []
    for phase in SUMMARIES:
        rows = results[phase]
        if not rows:
            continue
        keys = list(max(rows, key=len))
        lines += [f"## {phase}", "", "| " + " | ".join(keys) + " |", "|" + " ---: |" * len(keys)]
        for r in rows:
            lines.append("| " + " | ".join(
                f"{r[k]:.2f}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                for k in keys) + " |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def requests_for(phase: str, n: int, args: argparse.Namespace) -> list:
    tag = f"{phase}{n}"
    if phase == "decode":
        return [(f"s{i}", short_prompt(tag, i), args.decode_tokens, 0.1 * i) for i in range(n)]
    if phase == "mixed":
        requests = [(f"s{i}", short_prompt(tag, i), args.mixed_tokens, 0.1 * i)
                    for i in range(n - 1)]
        return [*requests, ("long", long_prompt(tag, args.long_chars), 64,
                            args.long_delay if n > 1 else 0.0)]
    if phase == "shared":
        return [(f"s{i}", shared_prompt(tag, i, args.long_chars), 64, 0.3 * i) for i in range(n)]
    if n == 1:
        return []
    # `late`: short requests arriving 2.3 s apart while the long one prefills
    # with nothing else running; each ends at its first token, so the next
    # finds the long one alone again.
    return [("long", long_prompt(tag, args.long_chars), 64, 0.0),
            *((f"s{i}", short_prompt(tag, i), 1, args.long_delay + 2.3 * i)
              for i in range(n - 1))]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path, help="summarize a finished run's directory")
    parser.add_argument("--binary", type=Path, default=ROOT / "target" / "release" / "local-ai")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--streams", default="1,2,4,8")
    parser.add_argument("--phases", default="decode,mixed,late")
    parser.add_argument("--decode-tokens", type=int, default=300)
    parser.add_argument("--mixed-tokens", type=int, default=512)
    parser.add_argument("--long-chars", type=int, default=24000)
    parser.add_argument("--long-delay", type=float, default=3.0)
    parser.add_argument("--sampled", action="store_true",
                        help="server default sampling, seeded, instead of greedy")
    args = parser.parse_args(argv)
    if args.sampled:
        SAMPLING.clear()
        SAMPLING["seed"] = 11
    if args.report:
        report(args.report)
        return 0
    if not args.out:
        parser.error("--out or --report is required")
    args.out.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="serve-concurrency-"))
    server = Server(args.binary, args.port, args.out / "server.log", home)
    try:
        server.start()
        # Warm the GPU pipelines so the first measured phase does not pay for them.
        run_phase(args.port, [("warm", short_prompt("warm", 0), 16, 0.0)])
        for phase in args.phases.split(","):
            for n in (int(n) for n in args.streams.split(",")):
                requests = requests_for(phase, n, args)
                if requests:
                    out = run_phase(args.port, requests)
                    (args.out / f"{phase}{n}.json").write_text(json.dumps(out, indent=1))
                    print(f"{phase}{n} done", flush=True)
    finally:
        server.stop()
        shutil.rmtree(home, ignore_errors=True)
    report(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
