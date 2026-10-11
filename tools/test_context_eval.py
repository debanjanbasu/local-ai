import json
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
                         sorted((t, p) for t in ("lookup", "fix") for p in ce.POSITIONS))

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
        self.assertIn("| 8k | none | 0 | 8,000–8,000 | 1/1 | - |", table)


if __name__ == "__main__":
    unittest.main()
