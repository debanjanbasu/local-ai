#!/usr/bin/env python3
"""Measure Bonsai decode throughput without fooling yourself.

Timing a memory-bandwidth-bound engine on a shared machine is easy to get
wrong. Two measurements in this project's history were invalidated by an
unrelated fat-LTO Rust build saturating four cores: one reported a 1.5x win
for a configuration change that a later run reversed, and one reported a
working streaming change as broken. Both produced byte-identical model output,
so nothing about the work differed -- only what else was competing for memory
bandwidth.

Decode spends 97-98% of wall time on the GPU and reads about 5.65 GB of weights
per token; individual kernels can be arithmetic- or latency-limited as well.
On a unified-memory machine, CPU workloads compete for memory bandwidth too,
so the idle check guards against interference without assuming every kernel
is bandwidth-bound.

The estimator is the **maximum** observed rate, not the median or mean.
Contention can only ever slow a run down, so the best sample is the one least
disturbed by something outside the measurement. Configurations are interleaved
round-robin rather than run to completion in sequence, so drift in machine state
is shared between them instead of being attributed to whichever happened to run
during the bad patch.

What the guard rejects, and why load average is not the whole test: the failure
that invalidated an earlier sweep was a build that *started and finished*
mid-measurement, so some samples paid a tax the others did not. Interleaving and
a best-case estimator are designed to absorb a *constant* co-tenant -- every
configuration pays it equally -- so the hard gate is on build tools appearing at
all, since those are the ones that come and go. A constant background load is
reported, not refused.

    tools/benchmark.py --rounds 3
    tools/benchmark.py --rounds 3 --binary A --binary B
    tools/benchmark.py --rounds 3 --sampled
    tools/benchmark.py --check-only
    tools/benchmark.py --prefill 2048,8192 --binary A --binary B
    tools/benchmark.py --ttft

Every configuration is a build (`--binary`, default target/release/local-ai)
running the shipped automatic policy: the engine has no opt-out or tuning
flags to compare, so speculation, thinking and every other policy run as they
ship. The losslessness of speculation against plain decoding is proved by the
engine's real-model tests (`kflash_generation_matches_plain_decoding`,
`batched_speculation_matches_alone` and their neighbours), not here.

`--sampled` measures the engine's default sampling (temperature 1.0, top-p
0.95, top-k 20, the CLI's fixed seed) instead of `--greedy`; the seed keeps
every configuration's output comparable, so the identical-output check still
applies.

`--prefill` times cold prefill of raw prompts of repository text and then a
second process restoring the first one's disk snapshot (a restart), with the
engine's disk prompt cache in a scratch `HOME`; every sample of a size must
generate the same greedy tokens. `--ttft` times the first streamed delta
through `local-ai serve` for an agent-shaped session (about 6K tokens of
system prompt and tools, a ~2K-token tool result) across a server restart.
Both load the model, so hold the shared GPU lock around them.

Sleep is part of the same guard. A 7 tok/s run in this project's history was
taken on battery with the lid closing, and the engine's own timing gave no hint
that anything was wrong. Every measured sample compares the wall clock with the
monotonic clock (which stops while the machine sleeps) and is refused if the
machine slept, and on macOS each generation runs under `caffeinate` when it is
available. `--allow-busy` downgrades load, build and sleep refusals; a waived
sleep labels the report untrusted rather than passing it off as clean.

Power source never blocks a run. Battery and AC are both valid ways to work, so
on macOS the source is *observed* -- once at the start and immediately before
and after each measured sample, never by a background poller -- and reported
rather than refused. Battery, an unreadable source (`unknown`) or a change of
source mid-run does not stop the run and does not by itself mark it untrusted.
What it does do is show up: the table carries a per-config `power` column and
the run ends with the start and end source plus a note when samples span more
than one source (or an unverified one), so timings taken under different power
states are never silently presented as a controlled comparison.

Exit status is 0 on success, 2 if the machine is too busy (or slept) to
measure, 1 on a usage or runtime error. Power source never affects it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "target" / "release" / "local-ai"
MODEL = REPO / "models" / "bonsai2-27b-ptq1" / "Ternary-Bonsai-2-27B-PTQ1_0.gguf"

# Build tools are the hard gate, because they arrive and leave mid-run; see
# `require_idle`. Load average cannot do that job: it cannot tell a constant
# co-tenant from a varying one, and on this machine the floor is constant --
# `AMPDevicesAgent` alone sits at 100% -- while the reported number still
# moves by about two as *my own* tooling does work. So load is bounded by
# something derived from the machine rather than a chosen constant, and only
# warns until there are more runnable tasks than cores.
CPU_COUNT = os.cpu_count() or 1
ADVISORY_LOAD = CPU_COUNT * 0.75
HARD_LOAD = float(CPU_COUNT)

DEFAULT_PROMPT = (
    "List the planets of the solar system in order, one per line, with a one "
    "sentence description of each."
)

BUILD_TOOLS = re.compile(r"\b(rustc|cargo|swift-frontend|clang)\b")

# `pmset -g batt` prints e.g. "Now drawing from 'AC Power'" or "'Battery Power'"
# (desktops report AC as well).
POWER_SOURCE = re.compile(r"Now drawing from '([^']+)'")

# Wall time keeps running while the machine sleeps; the monotonic clock does
# not. More divergence than this over one sample means the sample spanned a
# sleep. The margin absorbs ordinary NTP slews, which are far smaller.
SLEEP_TOLERANCE_SECONDS = 2.0


class Busy(RuntimeError):
    """The machine is too loaded to produce a trustworthy measurement."""


def power_source() -> str | None:
    """Return macOS's current power source, or None where it is not checked.

    Only macOS is checked; elsewhere this returns None and nothing is recorded.
    On macOS a missing or unparseable `pmset` returns "unknown", which is
    reported as unverified provenance rather than as AC. Neither value blocks.
    """
    if sys.platform != "darwin":
        return None
    try:
        done = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
        )
    except OSError:
        return "unknown"
    match = POWER_SOURCE.search(done.stdout)
    return match.group(1) if match else "unknown"


def keep_awake(argv: list[str]) -> list[str]:
    """Wrap a command in `caffeinate` on macOS so it cannot idle- or system-sleep.

    `-i` blocks idle sleep and `-s` blocks system sleep while on AC. A closing
    lid can still sleep the machine, which is what the clock check catches.
    """
    if sys.platform != "darwin":
        return argv
    caffeinate = shutil.which("caffeinate")
    return [caffeinate, "-i", "-s", *argv] if caffeinate else argv


@dataclass
class PowerGuard:
    """Refuse (or, with --allow-busy, waive) sleep; record power provenance.

    Power-source readings are observations only: they are appended to
    `observations` as (stage, source) pairs and never raise or waive anything.
    """

    allow_busy: bool = False
    waived: list[str] = field(default_factory=list)
    observations: list[tuple[str, str]] = field(default_factory=list)

    def reject(self, problem: str) -> None:
        if not self.allow_busy:
            raise Busy(problem)
        self.waived.append(problem)
        print(f"warning (--allow-busy): {problem}", file=sys.stderr, flush=True)

    def observe_power(self, stage: str, source: str | None = None) -> str | None:
        """Record the power source at `stage`; never refuses.

        Pass `source` to record a reading already taken, so the start of the
        run does not query `pmset` twice. Returns the recorded source, or None
        where power is not observed (non-macOS).
        """
        if source is None:
            source = power_source()
        if source is not None:
            self.observations.append((stage, source))
        return source

    def power_summary(self) -> list[str]:
        """Lines describing power provenance for the end of the report."""
        if not self.observations:
            return []
        start, end = self.observations[0][1], self.observations[-1][1]
        sources = list(dict.fromkeys(source for _, source in self.observations))
        lines = [
            (
                f"power provenance: started on {start}, ended on {end}; "
                f"observed {', '.join(sources)} "
                f"across {len(self.observations)} reading(s)."
            )
        ]
        if len(sources) > 1:
            changes = [
                f"{before} -> {after} {stage}"
                for (_, before), (stage, after) in zip(
                    self.observations, self.observations[1:]
                )
                if before != after
            ]
            lines.append(
                "note: the power source changed during the run ("
                + "; ".join(changes)
                + "). Samples span power states, so this is not a controlled "
                "comparison; check the power column before comparing rates."
            )
        elif "unknown" in sources:
            lines.append(
                "note: the power source could not be read (`pmset -g batt`), so "
                "these rates have unverified power provenance."
            )
        return lines

    def check_awake(self, stage: str, wall: float, steady: float) -> None:
        slept = wall - steady
        if slept > SLEEP_TOLERANCE_SECONDS:
            self.reject(
                f"the machine appears to have slept for about {slept:.1f}s {stage} "
                "(wall clock ran ahead of the monotonic clock). Keep the lid open "
                "and rerun."
            )


@dataclass(frozen=True)
class Config:
    """One build to measure."""

    name: str
    # None is the default build, read when the command is made so tests can
    # patch `BINARY`; `--binary` names others to compare builds.
    binary: Path | None = None
    # `--sampled` drops `--greedy`: the engine's default sampling (temperature
    # 1.0, top-p 0.95, top-k 20) with its fixed seed, still deterministic.
    greedy: bool = True

    def executable(self) -> str:
        return str(self.binary or BINARY)

    def argv(self, prompt: str, max_tokens: int) -> list[str]:
        return [
            self.executable(),
            "bonsai",
            *(["--greedy"] if self.greedy else []),
            "--json",
            "--max-tokens",
            str(max_tokens),
            prompt,
        ]


@dataclass
class Sample:
    tokens_per_second: float
    generated: int
    stop: str
    text: str
    # Distinct power sources observed before/after this sample, in order.
    power: tuple[str, ...] = ()


@dataclass
class Report:
    """Per-config results. `best` is the estimator; the rest is context."""

    samples: dict[str, list[Sample]] = field(default_factory=dict)

    def add(self, name: str, sample: Sample) -> None:
        self.samples.setdefault(name, []).append(sample)

    def render(self) -> str:
        width = max(len(name) for name in self.samples)
        show_power = any(
            sample.power for samples in self.samples.values() for sample in samples
        )
        header = (
            f"{'config'.ljust(width)}  {'best tok/s':>10}  {'median':>8}  "
            f"{'worst':>8}  {'spread':>7}  tokens  stop"
        )
        lines = [header + ("  power" if show_power else "")]
        for name, samples in self.samples.items():
            rates = [sample.tokens_per_second for sample in samples]
            spread = max(rates) / min(rates) if min(rates) > 0 else float("inf")
            line = (
                f"{name.ljust(width)}  {max(rates):10.2f}  {median(rates):8.2f}  "
                f"{min(rates):8.2f}  {spread:6.2f}x  {samples[0].generated:6d}  "
                f"{samples[0].stop}"
            )
            if show_power:
                sources = dict.fromkeys(
                    source for sample in samples for source in sample.power
                )
                line += "  " + " + ".join(sources)
            lines.append(line)
        return "\n".join(lines)


def machine_state() -> tuple[float, list[str]]:
    """Return the 1-minute load average and any competing builds."""
    load = os.getloadavg()[0]
    competing: list[str] = []
    listing = subprocess.run(
        ["ps", "-Ao", "pcpu,args"], capture_output=True, text=True, check=False
    )
    for line in listing.stdout.splitlines()[1:]:
        percent, _, args = line.strip().partition(" ")
        try:
            busy = float(percent) >= 50.0
        except ValueError:
            continue
        is_build = busy and BUILD_TOOLS.search(args)
        if is_build and str(REPO) not in args:
            competing.append(args.split()[0].rsplit("/", 1)[-1])
    return load, competing


def require_idle() -> None:
    """Refuse to measure on a busy machine."""
    load, competing = machine_state()
    if competing:
        raise Busy(
            "another build is saturating the CPU: "
            + ", ".join(sorted(set(competing)))
            + ". Decode here is memory-bandwidth-bound, so that competes for the "
            "same resource. Wait for it to finish."
        )
    if load > HARD_LOAD:
        raise Busy(
            f"load average {load:.2f} exceeds {HARD_LOAD:.2f}, the core count. More "
            "tasks are runnable than there are cores, so a best-case sample can "
            "still be paying for a co-tenant that is not there for every round. "
            f"Pass --allow-busy to measure anyway; the interleaving and estimator "
            f"then carry the comparison. Above {ADVISORY_LOAD:.2f} is already "
            "suspect; above the core count the contention is not noise."
        )
    if load > ADVISORY_LOAD:
        print(
            f"warning: load average {load:.2f} is above {ADVISORY_LOAD:.2f}. Constant "
            "co-tenants are absorbed by the interleaving and the best-case "
            "estimator, so this is a heads-up rather than a refusal.",
            file=sys.stderr,
        )


def run_once(
    config: Config, prompt: str, max_tokens: int, guard: PowerGuard | None = None
) -> Sample:
    """Run one generation and parse the engine's own measurement of it.

    With a guard, the sample is rejected if the machine slept during it, and
    the power source is observed (not enforced) immediately before and after
    it; without one (warm-up, output comparison) the timing is not used, so it
    is not policed. These are two `pmset` reads per sample, not a poller.
    """
    # The engine reports its own decode timing; the clocks here only detect
    # sleep. From the repository root, where the engine discovers the pinned
    # model.
    power: list[str] = []
    if guard is not None:
        before = guard.observe_power(f"before a {config.name} sample")
        power += [before] if before is not None else []
    wall, steady = time.time(), time.monotonic()
    done = subprocess.run(
        keep_awake(config.argv(prompt, max_tokens)),
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    if guard is not None:
        stage = f"during a {config.name} sample"
        guard.check_awake(stage, time.time() - wall, time.monotonic() - steady)
        after = guard.observe_power(f"after a {config.name} sample")
        power += [after] if after is not None else []
    if done.returncode != 0:
        raise RuntimeError(
            f"{config.name} exited {done.returncode}: {done.stderr.strip()[:400]}"
        )
    try:
        stats = json.loads(done.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{config.name} did not produce JSON: {error}. stdout was "
            f"{done.stdout[:200]!r}"
        ) from error
    return Sample(
        tokens_per_second=float(stats["decode_tokens_per_second"]),
        generated=int(stats["generated_tokens"]),
        stop=str(stats["stop_reason"]),
        # The whole generation: thinking is always on, so a capped run may
        # end inside its reasoning, before any final answer.
        text=stats["text"],
        power=tuple(dict.fromkeys(power)),
    )


def check_identical(configs: list[Config], prompt: str, max_tokens: int) -> None:
    """Refuse to compare configurations that did not do identical work.

    `--greedy` should make every configuration emit the same tokens. When it
    does not, the runs are not comparable, and the likelier explanation is the
    model changing its mind rather than the configuration being faster.
    """
    texts = {
        config.name: run_once(config, prompt, max_tokens).text for config in configs
    }
    if len(set(texts.values())) > 1:
        detail = ", ".join(f"{name}: {len(text)} chars" for name, text in texts.items())
        raise RuntimeError(
            "configurations produced different output, so their throughput is not "
            f"comparable ({detail}). Greedy decoding should make them identical."
        )


def measure(
    configs: list[Config],
    prompt: str,
    max_tokens: int,
    rounds: int,
    warmup: bool,
    guard: PowerGuard | None = None,
) -> Report:
    """Interleave configurations round-robin so drift is shared, not attributed."""
    report = Report()
    if warmup:
        # Page the checkpoint in before measuring, so the first sample is not
        # paying for cold demand-paged reads.
        run_once(configs[0], prompt, min(max_tokens, 8))
    for round_index in range(rounds):
        for config in configs:
            report.add(config.name, run_once(config, prompt, max_tokens, guard))
        print(f"  round {round_index + 1}/{rounds} done", file=sys.stderr, flush=True)
    return report


def builds(binaries: list[Path]) -> list[Config]:
    """One configuration per `--binary`, labelled by its build (the default
    build when none is named)."""
    if not binaries:
        return [Config("default")]
    return [
        Config(binary.name if len(binaries) > 1 else "default", binary)
        for binary in binaries
    ]


# ---------------------------------------------------------------- prefill

PREFILL_SIZES = "2048,8192,32768,131072"
PREFILL_OUTPUT_TOKENS = 16
# Prose first, then code. The order fixes the text each size reads, so builds
# compared in one run prefill byte-identical prompts.
CORPUS = (
    "docs/BONSAI.md",
    "README.md",
    "docs/*.md",
    "local-engine/src/**/*.rs",
    "local-metal/src/**/*.rs",
    "local-ai/src/**/*.rs",
)


def corpus(chars: int) -> str:
    """The first `chars` characters of the repository text in `CORPUS` order."""
    pieces: list[str] = []
    seen: set[Path] = set()
    total = 0
    while total < chars:
        before = total
        for pattern in CORPUS:
            for path in sorted(REPO.glob(pattern)):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                pieces.append(path.read_text(errors="replace"))
                total += len(pieces[-1]) + 1
                if total >= chars:
                    return "\n".join(pieces)[:chars]
        if total == before:
            # Everything was read once: repeat it.
            seen.clear()
            if not pieces:
                raise RuntimeError("no corpus text found")
    return "\n".join(pieces)[:chars]


def count_tokens(binary: str, path: Path) -> int:
    done = subprocess.run(
        [binary, "bonsai", "--tokenize", "--raw", "--prompt-file", str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    if done.returncode != 0:
        raise RuntimeError(f"tokenize failed: {done.stderr.strip()[:400]}")
    return int(json.loads(done.stdout)["prompt_tokens"])


def prefill_prompt(binary: str, tokens: int, directory: Path) -> tuple[Path, int]:
    """Write a raw prompt of about `tokens` tokens (within 0.2%) and count it."""
    path = directory / f"prompt-{tokens}.txt"
    chars = tokens * 4
    count = 0
    for _ in range(8):
        path.write_text(corpus(chars))
        count = count_tokens(binary, path)
        if abs(count - tokens) <= max(4, tokens // 500):
            break
        chars = max(1, round(chars * tokens / count))
    return path, count


@dataclass
class PrefillSample:
    prompt_tokens: int
    cached_tokens: int
    prefill_tokens_per_second: float | None
    first_token_seconds: float
    load_seconds: float | None
    token_ids: tuple[int, ...]


def run_prefill(
    config: Config, prompt: Path, home: Path, guard: PowerGuard | None
) -> PrefillSample:
    """One process: load, prefill `prompt` raw, decode a few greedy tokens.

    `HOME` points the engine's disk prompt cache at `home`, so a fresh `home`
    is a cold start and a second process on the same one restores what the
    first persisted: a restart.
    """
    argv = [
        config.executable(),
        "bonsai",
        "--raw",
        "--greedy",
        "--json",
        "--max-tokens",
        str(PREFILL_OUTPUT_TOKENS),
        "--prompt-file",
        str(prompt),
    ]
    if guard is not None:
        guard.observe_power(f"before a {config.name} prefill sample")
    wall, steady = time.time(), time.monotonic()
    done = subprocess.run(
        keep_awake(argv),
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
        env={**os.environ, "HOME": str(home)},
    )
    if guard is not None:
        stage = f"during a {config.name} prefill sample"
        guard.check_awake(stage, time.time() - wall, time.monotonic() - steady)
        guard.observe_power(f"after a {config.name} prefill sample")
    if done.returncode != 0:
        raise RuntimeError(
            f"{config.name} exited {done.returncode}: {done.stderr.strip()[-400:]}"
        )
    stats = json.loads(done.stdout)
    load = None
    for line in done.stderr.splitlines():
        if '"load_seconds"' in line:
            with contextlib.suppress(json.JSONDecodeError):
                load = float(json.loads(line)["load_seconds"])
    return PrefillSample(
        prompt_tokens=int(stats["prompt_tokens"]),
        cached_tokens=int(stats["cached_tokens"]),
        prefill_tokens_per_second=stats["prefill_tokens_per_second"],
        first_token_seconds=float(stats["first_token_seconds"]),
        load_seconds=load,
        token_ids=tuple(stats["token_ids"]),
    )


def measure_prefill(
    configs: list[Config],
    sizes: list[int],
    rounds: int,
    warmup: bool,
    guard: PowerGuard,
    directory: Path,
) -> list[str]:
    """Cold then warm (restarted) prefill per size, configurations interleaved.

    Every sample of a size must generate the same greedy tokens, cold or warm
    and in every configuration; the warm one must reuse the cold one's disk
    snapshot.
    """
    rows: dict[tuple[int, str], list[tuple[PrefillSample, PrefillSample]]] = {}
    if warmup:
        prompt, _ = prefill_prompt(configs[0].executable(), 256, directory)
        home = Path(tempfile.mkdtemp(dir=directory))
        run_prefill(configs[0], prompt, home, None)
        shutil.rmtree(home)
    for size in sizes:
        prompt, count = prefill_prompt(configs[0].executable(), size, directory)
        print(f"  {size}: {count} prompt tokens", file=sys.stderr, flush=True)
        reference: tuple[int, ...] | None = None
        for round_index in range(rounds):
            for config in configs:
                home = Path(tempfile.mkdtemp(dir=directory))
                try:
                    cold = run_prefill(config, prompt, home, guard)
                    warm = run_prefill(config, prompt, home, guard)
                finally:
                    shutil.rmtree(home, ignore_errors=True)
                if cold.cached_tokens != 0:
                    raise RuntimeError(
                        f"{config.name}: a cold run reused {cold.cached_tokens} tokens"
                    )
                reference = reference or cold.token_ids
                for label, sample in (("cold", cold), ("warm", warm)):
                    if sample.token_ids != reference:
                        raise RuntimeError(
                            f"{config.name} {label} at {size} tokens generated "
                            f"{list(sample.token_ids)}, not {list(reference)}: "
                            "greedy output must not depend on the build or cache"
                        )
                rows.setdefault((size, config.name), []).append((cold, warm))
                print(
                    f"  {size} {config.name} round {round_index + 1}/{rounds}: "
                    f"cold {cold.prefill_tokens_per_second or 0:.1f} tok/s, "
                    f"TTFT {cold.first_token_seconds:.2f}s; warm reused "
                    f"{warm.cached_tokens}, TTFT {warm.first_token_seconds:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )
    width = max(len(name) for _, name in rows)
    lines = [
        (f"{'tokens':>7}  {'config'.ljust(width)}  {'cold tok/s':>10}  {'median':>7}  "
        f"{'cold TTFT':>9}  {'warm reused':>11}  {'warm TTFT':>9}  {'load':>5}")
    ]
    for (size, name), samples in rows.items():
        rates = [cold.prefill_tokens_per_second or 0.0 for cold, _ in samples]
        lines.append(
            f"{samples[0][0].prompt_tokens:7d}  {name.ljust(width)}  {max(rates):10.1f}  "
            f"{median(rates):7.1f}  "
            f"{min(cold.first_token_seconds for cold, _ in samples):8.2f}s  "
            f"{min(warm.cached_tokens for _, warm in samples):11d}  "
            f"{min(warm.first_token_seconds for _, warm in samples):8.2f}s  "
            f"{min(cold.load_seconds or 0.0 for cold, _ in samples):4.1f}s"
        )
    return lines


# ------------------------------------------------------------------ TTFT

TTFT_SYSTEM_TOKENS = 5000
TTFT_TOOL_RESULT_TOKENS = 2000


def ttft_prompts(binary: str, directory: Path) -> tuple[str, list[dict], str]:
    """A coding-agent-shaped system prompt, tool schemas and one tool result.

    About 6K prompt tokens with the tools rendered, as a Codex session's first
    request; the tool result is about 2K tokens of source.
    """
    path, _ = prefill_prompt(binary, TTFT_SYSTEM_TOKENS + TTFT_TOOL_RESULT_TOKENS, directory)
    text = path.read_text()
    split = len(text) * TTFT_SYSTEM_TOKENS // (TTFT_SYSTEM_TOKENS + TTFT_TOOL_RESULT_TOKENS)
    system = "You are a coding agent. Follow the project notes below.\n\n" + text[:split]
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        key: {"type": "string", "description": f"The {key} to use."}
                        for key in keys
                    },
                    "required": list(keys[:1]),
                },
            },
        }
        for name, description, keys in (
            ("shell", "Run a shell command in the workspace and return its output.", ("command", "workdir")),
            ("read_file", "Read a file, optionally a line range.", ("path", "range")),
            ("apply_patch", "Apply a unified diff to the workspace.", ("patch",)),
            ("search", "Search the repository with a regular expression.", ("pattern", "path", "glob")),
            ("list_files", "List files under a directory.", ("path", "depth")),
            ("web_fetch", "Fetch a URL and return it as Markdown.", ("url", "objective")),
        )
    ]
    return system, tools, text[split:]


def stream_ttft(port: int, body: dict) -> tuple[float, dict]:
    """POST a streaming chat completion; return time to first delta and usage."""
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    started = time.monotonic()
    connection.request(
        "POST",
        "/v1/chat/completions",
        json.dumps({**body, "stream": True, "stream_options": {"include_usage": True}}),
        {"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:400]!r}")
    first: float | None = None
    usage: dict = {}
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if first is None and any(
                delta.get(key) for key in ("content", "reasoning_content", "reasoning", "tool_calls")
            ):
                first = time.monotonic() - started
    connection.close()
    if first is None:
        raise RuntimeError("the response streamed no delta")
    return first, usage


def measure_ttft(config: Config, directory: Path) -> list[str]:
    """End-to-end TTFT over HTTP for an agent session, across a restart.

    Steps: a cold first request; a second session sharing the system prompt
    and tools; a follow-up that adds one tool call and a ~2K-token result;
    then, after restarting the server on the same prompt cache, a new session
    and the follow-up again.
    """
    import socket

    system, tools, result = ttft_prompts(config.executable(), directory)
    home = Path(tempfile.mkdtemp(dir=directory))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def request(question: str, follow_up: bool = False) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        if follow_up:
            messages += [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "local-engine/src/lib.rs"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": result},
            ]
        return {
            "model": "bonsai",
            "messages": messages,
            "tools": tools,
            "max_completion_tokens": 16,
            "temperature": 0,
            "reasoning_effort": "none",
        }

    def serve() -> subprocess.Popen:
        process = subprocess.Popen(
            keep_awake([config.executable(), "serve", "--port", str(port)]),
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "HOME": str(home)},
        )
        assert process.stderr is not None
        # The listening line is the server's readiness signal.
        for line in process.stderr:
            if "server listening on http://" in line:
                break
        else:
            raise RuntimeError("server exited before listening")
        threading.Thread(target=process.stderr.read, daemon=True).start()
        return process

    def stop(process: subprocess.Popen) -> None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(300)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    steps = [
        ("cold first request", "Where is prefill implemented?", False, False),
        ("new session, shared system+tools", "How does the prompt cache evict?", False, False),
        ("tool result follow-up (+2K)", "Where is prefill implemented?", True, False),
        ("after restart: new session", "What does the scheduler batch?", False, True),
        ("after restart: tool result follow-up", "Where is prefill implemented?", True, False),
    ]
    lines = [f"{'step':38}  {'TTFT':>7}  {'prompt':>6}  {'reused':>6}  source"]
    process = serve()
    try:
        for name, question, follow_up, restart in steps:
            if restart:
                stop(process)
                process = serve()
            seconds, usage = stream_ttft(port, request(question, follow_up))
            details = usage.get("prompt_tokens_details") or {}
            lines.append(
                f"{name:38}  {seconds:6.2f}s  {usage.get('prompt_tokens', 0):6d}  "
                f"{details.get('cached_tokens', 0):6d}  {details.get('cache_source', '')}"
            )
            print(f"  {lines[-1]}", file=sys.stderr, flush=True)
    finally:
        stop(process)
        shutil.rmtree(home, ignore_errors=True)
    return lines


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rounds", type=int, default=3, help="repetitions per config")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="decode with the default sampling settings (seeded) instead of --greedy",
    )
    parser.add_argument(
        "--prefill",
        nargs="?",
        const=PREFILL_SIZES,
        metavar="SIZES",
        help="measure cold and warm (restarted) prefill of raw prompts of these "
        f"comma-separated token counts (default {PREFILL_SIZES}) instead of decode",
    )
    parser.add_argument(
        "--ttft",
        action="store_true",
        help="measure end-to-end TTFT through `local-ai serve` for an "
        "agent-shaped session across a restart, instead of decode",
    )
    parser.add_argument(
        "--binary",
        action="append",
        type=Path,
        default=[],
        help="build to measure (repeat to compare builds); default "
        "target/release/local-ai",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report machine state (load, builds, power source) and exit "
        "without measuring",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the warm-up generation that pages the checkpoint in",
    )
    parser.add_argument(
        "--allow-busy",
        action="store_true",
        help="measure anyway: skip the load and competing-build refusals and "
        "downgrade sleep refusals to warnings (a waived sleep labels the "
        "results untrusted). Power source never blocks, with or without this",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    load, competing = machine_state()
    print(
        f"load average {load:.2f} of {HARD_LOAD:.2f} cores"
        + (f" (advisory floor {ADVISORY_LOAD:.2f})" if load > ADVISORY_LOAD else ""),
        file=sys.stderr,
    )
    if competing:
        print(f"competing builds: {', '.join(sorted(set(competing)))}", file=sys.stderr)
    source = power_source()
    if source is not None:
        print(f"power source: {source}", file=sys.stderr)
    if args.check_only:
        return 0

    missing = [b for b in (args.binary or [BINARY]) if not b.exists()]
    if missing:
        print(
            f"error: {missing[0]} not found; run `cargo build --release -p local-ai` first",
            file=sys.stderr,
        )
        return 1
    if not MODEL.exists():
        print(f"error: {MODEL} not found", file=sys.stderr)
        return 1

    guard = PowerGuard(allow_busy=args.allow_busy)
    # Reuse the status-line reading as the start of the provenance record.
    guard.observe_power("at start", source)
    try:
        if not args.allow_busy:
            require_idle()

        configs = builds(args.binary)
        if args.sampled:
            configs = [replace(config, greedy=False) for config in configs]
        print(
            f"comparing {', '.join(config.name for config in configs)} "
            f"for {args.rounds} rounds",
            file=sys.stderr,
        )
        if args.prefill or args.ttft:
            with tempfile.TemporaryDirectory(prefix="local-ai-prefill-") as directory:
                if args.prefill:
                    sizes = [int(size) for size in args.prefill.split(",")]
                    lines = measure_prefill(
                        configs, sizes, args.rounds, not args.no_warmup, guard, Path(directory)
                    )
                else:
                    lines = [
                        line
                        for config in configs
                        for line in [f"[{config.name}]", *measure_ttft(config, Path(directory))]
                    ]
            print()
            print("\n".join(lines))
            print()
            if args.prefill:
                print(
                    "cold tok/s is prefilled tokens over prefill time (best of rounds); "
                    "TTFT is the engine's time to first token after load; warm runs "
                    "are new processes restoring the cold run's disk snapshot."
                )
            for line in guard.power_summary():
                print(line)
            return 0
        # Prove the configurations do identical work before trusting any rate.
        check_identical(configs, args.prompt, min(args.max_tokens, 8))
        report = measure(
            configs,
            args.prompt,
            args.max_tokens,
            args.rounds,
            not args.no_warmup,
            guard,
        )
    except Busy as error:
        print(f"refusing to measure: {error}", file=sys.stderr)
        return 2
    print()
    print(report.render())
    print()
    print("best tok/s is the estimator: contention only ever slows a run down.")
    if guard.waived:
        print(
            f"UNTRUSTED: {len(guard.waived)} sleep check(s) failed and were "
            "waived by --allow-busy; see the warnings above. Do not compare these "
            "rates with clean runs."
        )
    for line in guard.power_summary():
        print(line)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Busy as failure:
        print(f"refusing to measure: {failure}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as failure:
        print(f"error: {failure}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
