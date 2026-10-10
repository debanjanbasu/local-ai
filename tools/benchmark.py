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
    tools/benchmark.py --sweep-depth
    tools/benchmark.py --rounds 3 -- '--no-thinking' '--greedy'
    tools/benchmark.py --check-only

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
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
    """One set of engine flags to measure."""

    name: str
    flags: tuple[str, ...] = ()

    def argv(self, prompt: str, max_tokens: int) -> list[str]:
        return [
            str(BINARY),
            "bonsai",
            "--greedy",
            "--no-thinking",
            "--json",
            "--max-tokens",
            str(max_tokens),
            *self.flags,
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
        text=stats["final_text"],
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


def depth_sweep() -> list[Config]:
    """Every depth the engine accepts, plus a no-speculation reference."""
    return [
        Config("no-speculation", ("--no-speculation",)),
        # Space-separated, not `--mtp-depth=N`: every value-taking flag in this
        # CLI takes its value as a separate argument. `--export` is the only
        # exception, because its own value grammar contains `=`.
        *[
            Config(f"depth {depth}", ("--mtp-depth", str(depth)))
            for depth in (1, 2, 3, 4)
        ],
    ]


def parse_configs(groups: list[str]) -> list[Config]:
    """Build configurations from quoted flag groups on the command line.

    A bare ``--`` separates one configuration from the next, so a comparison
    reads as two groups either side of it::

        tools/benchmark.py -- '--no-thinking' -- '--no-speculation'

    An empty group is rejected rather than silently measured as the default:
    a configuration that is not the one you asked for produces numbers that
    look real and answer a different question, which is the failure this whole
    tool exists to prevent.
    """
    if not groups or groups == ["--"]:
        return [Config("default")]

    configs: list[Config] = []
    current: list[str] = []
    for group in [*groups, "--"]:
        if group != "--":
            current.append(group)
            continue
        if not current:
            raise RuntimeError(
                "empty configuration: `--` must separate two flag groups, "
                "not stand alone"
            )
        flags = tuple(shlex.split(" ".join(current)))
        configs.append(Config(" ".join(flags), flags))
        current = []
    return configs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rounds", type=int, default=3, help="repetitions per config")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--sweep-depth",
        action="store_true",
        help="compare every --mtp-depth the engine accepts",
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
    parser.add_argument(
        "configs",
        nargs="*",
        help="quoted flag groups to compare, one per configuration, separated by `--`",
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

    if not BINARY.exists():
        print(
            f"error: {BINARY} not found; run `cargo build --release -p local-ai` first",
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

        configs = depth_sweep() if args.sweep_depth else parse_configs(args.configs)
        print(
            f"comparing {', '.join(config.name for config in configs)} "
            f"for {args.rounds} rounds",
            file=sys.stderr,
        )
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
