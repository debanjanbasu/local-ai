import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import context_eval as ce


class NeedleTest(unittest.TestCase):
    def test_needles_are_seeded_and_cover_every_position(self):
        first, again, other = ce.make_needles(0), ce.make_needles(0), ce.make_needles(1)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(sorted((n.task, n.position) for n in first),
                         sorted((t, p) for t in ("recall", "lookup", "fix") for p in ce.POSITIONS))

    def test_every_bug_template_is_detected_by_its_tests(self):
        for seed in range(6):
            for needle in ce.make_needles(seed):
                if needle.task != "fix":
                    continue
                self.assertEqual(ce.score_fix(needle, needle.expected), (True, "tests pass"), needle.function)
                self.assertFalse(ce.score_fix(needle, needle.buggy_line)[0], needle.function)

    def test_fix_answers_in_other_shapes(self):
        needle = next(n for n in ce.make_needles(0) if n.task == "fix")
        self.assertTrue(ce.score_fix(needle, f"```python\n{needle.expected}\n```")[0])
        whole = next(iter(needle.files.values())).replace(needle.buggy_line, needle.expected)
        self.assertTrue(ce.score_fix(needle, f"Here:\n```python\n{whole}```")[0])
        self.assertFalse(ce.score_fix(needle, "")[0])
        self.assertFalse(ce.score_fix(needle, "return (")[0])

    def test_unsafe_or_runaway_answers_are_never_executed(self):
        needle = next(n for n in ce.make_needles(0) if n.task == "fix")
        for answer in ("import os", "open('x', 'w')", "while True: pass", "().__class__", "return 9 ** 9 ** 9"):
            ok, why = ce.score_fix(needle, answer)
            self.assertFalse(ok, answer)
            self.assertTrue(why.startswith(("rejected unsafe answer", "SyntaxError")) or "rejected" in why, why)

    def test_lookup_scoring_uses_last_integer(self):
        needle = next(n for n in ce.make_needles(0) if n.task == "lookup")
        self.assertTrue(ce.score_lookup(needle, f"The fee is 3, so {needle.expected}")[0])
        self.assertFalse(ce.score_lookup(needle, "unknown")[0])

    def test_lookup_expected_value_matches_the_planted_code(self):
        for needle in ce.make_needles(3):
            if needle.task != "lookup":
                continue
            namespace = {}
            config, service = sorted(needle.files)
            exec(needle.files[config], namespace)  # noqa: S102 -- fixture code
            body = needle.files[service].split("\n", 1)[1]
            exec(body, namespace)  # noqa: S102 -- fixture code
            call = needle.question.split("compute ")[1].split(".")[0]
            self.assertEqual(str(eval(call, namespace)), needle.expected)


class ContextTest(unittest.TestCase):
    def test_context_contains_needles_in_position_order(self):
        needles = ce.make_needles(2)
        text, where = ce.build_context(2, "8k", needles)
        self.assertEqual(text, ce.build_context(2, "8k", needles)[0])
        offsets = {}
        for needle in needles:
            for path, body in needle.files.items():
                self.assertIn(f"FILE {path}\n{body}", text)
                offsets.setdefault(needle.position, text.index(f"FILE {path}"))
        self.assertLess(offsets["start"], offsets["middle"])
        self.assertLess(offsets["middle"], offsets["end"])
        self.assertLess(where["start"], 0.1)
        self.assertGreater(where["end"], 0.85)
        self.assertGreater(len(text), 8_000 * ce.CHARS_PER_TOKEN)
        self.assertIn("legacy/", text)

    def test_short_control_holds_every_needle_in_little_text(self):
        needles = ce.make_needles(0)
        text, _ = ce.build_context(0, "short", needles)
        self.assertLess(len(text), 16_000)
        for needle in needles:
            for path in needle.files:
                self.assertIn(path, text)


class RecordTest(unittest.TestCase):
    def test_record_report_and_resume(self):
        needle = next(n for n in ce.make_needles(0) if n.task == "lookup")
        payload = {
            "_status": 200, "_wall_seconds": 12.0,
            "choices": [{"message": {"content": needle.expected}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8000, "completion_tokens": 11,
                      "prompt_tokens_details": {"cached_tokens": 0, "cache_source": "none"},
                      "completion_tokens_details": {"reasoning_tokens": 0}},
            "timings": {"prefill_seconds": 80.0, "first_token_seconds": 80.5, "elapsed_seconds": 81.5},
        }
        row = ce.record("8k", 0, "none", needle, payload, 1, {"start": 0.03})
        self.assertTrue(row["passed"])
        self.assertEqual((row["prefill_tok_s"], row["decode_tok_s"]), (100.0, 11.0))
        failed = ce.record("8k", 0, "none", needle, {"_status": 400, "error": {"message": "x"}}, 1, {})
        self.assertFalse(failed["passed"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "results.jsonl").write_text(json.dumps(row) + "\n")
            self.assertEqual(ce.done_keys(out / "results.jsonl"), {("8k", 0, "none", "lookup", needle.position)})
            table = ce.report(out)
        self.assertIn("| 8k | none | 0 | 8,000–8,000 | - | 1/1 | - |", table)

    def test_http_failures_are_retried_and_not_scored(self):
        needle = next(n for n in ce.make_needles(0) if n.task == "recall")
        ok = {"_status": 200, "_wall_seconds": 1.0, "choices": [{"message": {"content": needle.expected}}],
              "usage": {"prompt_tokens": 100}}
        failed = ce.record("short", 0, "none", needle, {"_status": 503, "error": {"message": "busy"}}, 1, {})
        passed = ce.record("short", 0, "none", needle, ok, 1, {})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(json.dumps(failed) + "\n")
            self.assertEqual(ce.done_keys(path), set())
            self.assertNotIn("| short |", ce.report(Path(tmp)))
            path.write_text(json.dumps(failed) + "\n" + json.dumps(passed) + "\n" + json.dumps(failed) + "\n")
            self.assertEqual([r["passed"] for r in ce.load_rows(path)], [True])
            self.assertIn("| short | none | 0 | 100–100 | 1/1 |", ce.report(Path(tmp)))

    def test_wilson_interval(self):
        low, high = ce.wilson(27, 27)
        self.assertAlmostEqual(low, 0.875, places=3)
        self.assertEqual(high, 1.0)
        low, high = ce.wilson(5, 9)
        self.assertLess(low, 5 / 9)
        self.assertGreater(high, 5 / 9)

    def test_staged_prefill_adds_up_every_stage(self):
        first = {"prompt_tokens": 250_000, "cached_tokens": 195_000, "prefill_seconds": 1_000.0}
        warm = {"prompt_tokens": 250_010, "cached_tokens": 249_990, "prefill_seconds": 2.0}
        stages = [{"status": 200, "prompt_tokens": 120_000, "prefill_seconds": 1_500.0},
                  {"status": 200, "prompt_tokens": 195_001, "prefill_seconds": 1_700.0}]
        self.assertEqual(ce.cold_prefill([first, warm], stages), (250_000, 4_200.0))
        self.assertIsNone(ce.cold_prefill([first, warm], stages[:1]))  # a prefix nobody measured
        self.assertIsNone(ce.cold_prefill([warm], []))


class StopTest(unittest.TestCase):
    def test_server_is_interrupted_before_it_is_terminated(self):
        # The server finishes its prompt-cache writes only on SIGINT.
        child = (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGINT, lambda *_: (print('interrupted', flush=True), sys.exit(0)))\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)\n"
        )
        server = ce.LocalServer(Path(sys.executable), 0, Path(os.devnull), [])
        server.process = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True)
        self.assertEqual(server.process.stdout.readline().strip(), "ready")
        ce.stop_server(server, 10)
        self.assertEqual(server.process.stdout.read().strip(), "interrupted")
        self.assertEqual(server.process.returncode, 0)
        server.process.stdout.close()


class StageTest(unittest.TestCase):
    def test_reasoning_instruction_matches_the_engine_template(self):
        source = (ce.ROOT / "local-engine/src/bonsai_tokenizer.rs").read_text()
        self.assertIn(f'DEFAULT_REASONING_INSTRUCTION: &str = "{ce.REASONING_INSTRUCTION}";', source)

    def test_stage_prefix_ends_at_a_file_boundary_of_the_rendered_prompt(self):
        system, _ = ce.build_context(1, "8k", ce.make_needles(1))
        for effort in ("none", "xhigh"):
            rendered = ce.rendered_system(system, effort)
            prefix = ce.stage_prefix(system, effort, 4_000)
            self.assertTrue(rendered.startswith(prefix))
            self.assertTrue(rendered[len(prefix):].startswith("FILE "))
            self.assertGreaterEqual(len(prefix), 4_000 * ce.CHARS_PER_TOKEN)
        self.assertTrue(ce.rendered_system(system, "xhigh").startswith(
            f"<|im_start|>system\n{ce.REASONING_INSTRUCTION}\n\nYou are answering"))
        with self.assertRaises(ValueError):
            ce.stage_prefix(system, "none", 100_000)


if __name__ == "__main__":
    unittest.main()
