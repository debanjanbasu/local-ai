import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark

CAFFEINATE = "/usr/bin/caffeinate"
AC = "Now drawing from 'AC Power'\n -InternalBattery-0\t99%; AC attached;\n"
BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0\t80%; discharging;\n"


class FakeMachine:
    """Stands in for ps, pmset, the engine and the clocks; nothing real runs."""

    def __init__(self, power, sleep_on_sample=None):
        # One pmset reading per call; the last one repeats.
        self.power = list(power)
        self.sleep_on_sample = sleep_on_sample
        self.engine_calls = []
        self.wall = 1_000.0
        self.steady = 50.0

    def run(self, argv, **kwargs):
        if argv[0] == "ps":
            return subprocess.CompletedProcess(argv, 0, "%CPU ARGS\n", "")
        if argv[0] == "pmset":
            reading = self.power.pop(0) if len(self.power) > 1 else self.power[0]
            return subprocess.CompletedProcess(argv, 0, reading, "")
        self.engine_calls.append(argv)
        self.wall += 1.0
        self.steady += 1.0
        if self.sleep_on_sample == len(self.engine_calls):
            self.wall += 30.0  # wall clock keeps running while asleep
        stats = {
            "decode_tokens_per_second": 20.0,
            "generated_tokens": 8,
            "stop_reason": "max_tokens",
            "final_text": "Mercury",
        }
        return subprocess.CompletedProcess(argv, 0, json.dumps(stats), "")


class MainTests(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        binary = Path(root.name) / "local-ai"
        model = Path(root.name) / "model.gguf"
        binary.touch()
        model.touch()
        for patch in (
            mock.patch.object(benchmark, "BINARY", binary),
            mock.patch.object(benchmark, "MODEL", model),
            mock.patch.object(benchmark.os, "getloadavg", return_value=(0.1, 0, 0)),
            mock.patch.object(benchmark.shutil, "which", return_value=CAFFEINATE),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def main(self, machine, *args, platform="darwin"):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(benchmark.sys, "platform", platform),
            mock.patch.object(benchmark.subprocess, "run", side_effect=machine.run),
            mock.patch.object(benchmark.time, "time", lambda: machine.wall),
            mock.patch.object(benchmark.time, "monotonic", lambda: machine.steady),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = benchmark.main(["--rounds", "2", *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_on_ac_measures_under_caffeinate(self):
        machine = FakeMachine([AC])
        code, out, err = self.main(machine)
        self.assertEqual(code, 0, err)
        self.assertIn("20.00", out)
        self.assertNotIn("UNTRUSTED", out)
        self.assertIn("power source: AC Power", err)
        # identity check + warm-up + two rounds of one config
        self.assertEqual(len(machine.engine_calls), 4)
        for argv in machine.engine_calls:
            self.assertEqual(argv[:3], [CAFFEINATE, "-i", "-s"])
            self.assertEqual(argv[3], str(benchmark.BINARY))

    def test_battery_measures_and_reports_provenance_without_refusing(self):
        machine = FakeMachine([BATTERY])
        code, out, err = self.main(machine)
        self.assertEqual(code, 0, err)
        self.assertNotIn("refusing", err)
        self.assertNotIn("UNTRUSTED", out)
        self.assertIn("power source: Battery Power", err)
        self.assertIn("  power", out)
        self.assertIn("20.00", out)
        self.assertIn(
            "power provenance: started on Battery Power, ended on Battery Power", out
        )
        self.assertNotIn("note:", out)
        # identity check + warm-up + two rounds, all still under caffeinate
        self.assertEqual(len(machine.engine_calls), 4)
        for argv in machine.engine_calls:
            self.assertEqual(argv[:3], [CAFFEINATE, "-i", "-s"])

    def test_switch_to_battery_mid_run_is_reported_not_refused(self):
        # Readings: start, before sample 1, then the charger is pulled while
        # sample 1 runs.
        machine = FakeMachine([AC, AC, BATTERY])
        code, out, err = self.main(machine)
        self.assertEqual(code, 0, err)
        self.assertNotIn("refusing", err)
        self.assertNotIn("UNTRUSTED", out)
        self.assertEqual(len(machine.engine_calls), 4)
        self.assertIn("AC Power + Battery Power", out)
        self.assertIn("started on AC Power, ended on Battery Power", out)
        self.assertIn("AC Power -> Battery Power after a default sample", out)
        self.assertIn("not a controlled comparison", out)

    def test_allow_busy_does_not_change_power_handling(self):
        machine = FakeMachine([AC, AC, BATTERY])
        code, out, err = self.main(machine, "--allow-busy")
        self.assertEqual(code, 0, err)
        self.assertNotIn("warning (--allow-busy)", err)
        self.assertNotIn("UNTRUSTED", out)
        self.assertIn("not a controlled comparison", out)
        self.assertEqual(len(machine.engine_calls), 4)

    def test_sleep_during_a_sample_is_rejected(self):
        # The third engine call is the first measured sample.
        machine = FakeMachine([AC], sleep_on_sample=3)
        code, out, err = self.main(machine)
        self.assertEqual(code, 2)
        self.assertIn("slept for about 30.0s", err)
        self.assertNotIn("best tok/s", out)

    def test_sleep_on_battery_is_still_rejected(self):
        machine = FakeMachine([BATTERY], sleep_on_sample=3)
        code, _, err = self.main(machine)
        self.assertEqual(code, 2)
        self.assertIn("slept for about 30.0s", err)
        self.assertNotIn("Battery Power", err.split("refusing to measure", 1)[1])

    def test_allow_busy_waives_sleep_and_labels_it_untrusted(self):
        machine = FakeMachine([AC], sleep_on_sample=3)
        code, out, err = self.main(machine, "--allow-busy")
        self.assertEqual(code, 0, err)
        self.assertIn("warning (--allow-busy)", err)
        self.assertIn("UNTRUSTED: 1 sleep check(s)", out)

    def test_unreadable_power_source_is_reported_as_unverified(self):
        machine = FakeMachine(["garbage\n"])
        code, out, err = self.main(machine)
        self.assertEqual(code, 0, err)
        self.assertNotIn("UNTRUSTED", out)
        self.assertIn("power source: unknown", err)
        self.assertIn("unverified power provenance", out)

    def test_competing_build_still_refuses(self):
        machine = FakeMachine([BATTERY])
        build = "%CPU ARGS\n 99.0 /usr/bin/rustc --crate-name other\n"
        run = machine.run

        def busy_ps(argv, **kwargs):
            if argv[0] == "ps":
                return subprocess.CompletedProcess(argv, 0, build, "")
            return run(argv, **kwargs)

        machine.run = busy_ps
        code, _, err = self.main(machine)
        self.assertEqual(code, 2)
        self.assertIn("another build is saturating the CPU: rustc", err)
        self.assertEqual(machine.engine_calls, [])

    def test_other_platforms_skip_power_checks_and_caffeinate(self):
        machine = FakeMachine([BATTERY])
        code, out, err = self.main(machine, platform="linux")
        self.assertEqual(code, 0, err)
        self.assertNotIn("power source", err)
        self.assertNotIn("power provenance", out)
        self.assertNotIn("UNTRUSTED", out)
        for argv in machine.engine_calls:
            self.assertEqual(argv[0], str(benchmark.BINARY))

    def test_check_only_reports_power_without_measuring(self):
        machine = FakeMachine([BATTERY])
        code, _, err = self.main(machine, "--check-only")
        self.assertEqual(code, 0)
        self.assertIn("power source: Battery Power", err)
        self.assertNotIn("refusing", err)
        self.assertEqual(machine.engine_calls, [])


if __name__ == "__main__":
    unittest.main()
