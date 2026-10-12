"""Tests for tools/judgment_calibrate.py (pure Python; no NumPy, model or GPU)."""

import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judgment_calibrate as jc


def row(raw, target, yes=0, group="g", family="f", query=None, **extra):
    out = {"id": extra.pop("id", f"r{random.random()}"), "set": "s", "split": "calibration",
           "raw": raw, "target": target, "yes_index": yes, "group": group, "family": family,
           "query": query or [0.0, 0.0], "options": ["Yes", "No"], "prompt_tokens": 10,
           "question_text": jc.CR_QUESTION, "state_prefix": "Diff:\n@@"}
    out.update(extra)
    return out


def synthetic(n, seed, scale=3.0):
    """Predicate rows whose true P(yes) is sigmoid(margin / scale): overconfident raw."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        margin = rng.uniform(-8, 8)
        label = rng.random() < jc.sigmoid(margin / scale)
        yes = i % 2
        raw = [0.0, 0.0]
        raw[yes], raw[1 - yes] = margin / 2, -margin / 2
        rows.append(row(raw, yes if label else 1 - yes, yes, group=f"g{i % 37}",
                        id=f"r{seed}-{i}"))
    return rows


class CalibrationMathTest(unittest.TestCase):
    def test_platt_recovers_a_known_slope_and_bias(self):
        rng = random.Random(0)
        margins = [rng.uniform(-6, 6) for _ in range(3000)]
        labels = [rng.random() < jc.sigmoid(0.5 * x - 0.7) for x in margins]
        a, b = jc.fit_platt(margins, labels)
        self.assertAlmostEqual(a, 0.5, delta=0.06)
        self.assertAlmostEqual(b, -0.7, delta=0.12)

    def test_platt_probabilities_follow_the_yes_option_in_either_order(self):
        calibrator = jc.Calibrator("platt", slope=1.0, bias=0.0)
        self.assertEqual(calibrator.probabilities([2.0, 0.0], 0),
                         calibrator.probabilities([0.0, 2.0], 1)[::-1])
        self.assertAlmostEqual(calibrator.probabilities([2.0, 0.0], 0)[0], jc.sigmoid(2.0))
        # Out of a predicate scope it degrades to an inverse temperature, for diagnostics.
        self.assertAlmostEqual(sum(calibrator.probabilities([1.0, 2.0, 3.0])), 1.0)

    def test_calibration_fixes_overconfidence_on_held_out_rows(self):
        cal, test = synthetic(800, 1), synthetic(800, 2)
        raw = jc.metrics(jc.predictions(jc.Calibrator("temperature"), test))
        for method in ("temperature", "platt"):
            fitted = jc.fit_calibrator(method, cal)
            calibrated = jc.metrics(jc.predictions(fitted, test))
            self.assertLess(calibrated["nll"], raw["nll"])
            self.assertLess(calibrated["ece"], raw["ece"])
            self.assertLess(calibrated["ece"], 0.06)

    def test_grouped_cross_validation_keeps_groups_in_one_fold(self):
        rows = synthetic(400, 3)
        folds = {}
        for r in rows:
            folds.setdefault(r["group"], set()).add(jc.fold_of(r["group"]))
        self.assertTrue(all(len(f) == 1 for f in folds.values()))
        self.assertTrue(math.isfinite(jc.cross_validated_nll("platt", rows)))
        self.assertEqual(jc.candidate_methods(rows), ["temperature", "platt"])
        self.assertEqual(jc.candidate_methods(rows + [row([0, 1, 2], 0, None)]), ["temperature"])

    def test_wilson_interval(self):
        low, high = jc.wilson(80, 100)
        self.assertAlmostEqual(low, 0.7112, places=3)
        self.assertAlmostEqual(high, 0.8666, places=3)
        self.assertEqual(jc.wilson(0, 0), (0.0, 1.0))

    def test_metrics_on_a_hand_checked_example(self):
        preds = jc.predictions(jc.Calibrator("temperature"),
                               [row([math.log(3), 0.0], 0), row([math.log(3), 0.0], 1)])
        m = jc.metrics(preds)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["nll"], (-math.log(0.75) - math.log(0.25)) / 2)
        self.assertAlmostEqual(m["brier_binary"], (0.25 ** 2 + 0.75 ** 2) / 2)
        self.assertAlmostEqual(m["ece"], 0.25)  # one bin: confidence 0.75, accuracy 0.5

    def test_threshold_meets_the_target_on_its_own_split_or_refuses_all(self):
        preds = [{"confidence": 0.5 + i / 2000, "correct": i > 300} for i in range(1000)]
        threshold = jc.choose_threshold(preds, 0.10, 50)
        answered = [p for p in preds if p["confidence"] >= threshold]
        wrong = sum(not p["correct"] for p in answered)
        self.assertLessEqual(jc.wilson(wrong, len(answered))[1], 0.10)
        self.assertGreaterEqual(len(answered), 50)
        lower = max(p["confidence"] for p in preds if p["confidence"] < threshold)
        smaller = [p for p in preds if p["confidence"] >= lower]
        self.assertGreater(jc.wilson(sum(not p["correct"] for p in smaller),
                                     len(smaller))[1], 0.10)
        hopeless = [{"confidence": 0.9, "correct": i % 2 == 0} for i in range(200)]
        self.assertIsNone(jc.choose_threshold(hopeless, 0.10, 50))

    def test_selective_curve_and_point(self):
        preds = [{"confidence": c, "correct": ok}
                 for c, ok in [(0.9, True), (0.8, True), (0.7, False), (0.6, False)]]
        sel = jc.selective(preds, 0.75)
        self.assertEqual(sel["at_threshold"]["answered"], 2)
        self.assertEqual(sel["at_threshold"]["risk"], 0.0)
        self.assertEqual(sel["curve"]["1.0"]["risk"], 0.5)
        self.assertAlmostEqual(sel["aurc"], (0 + 0 + 1 / 3 + 1 / 2) / 4)

    def test_gate_refuses_far_queries_and_keeps_its_percentile(self):
        rng = random.Random(4)
        rows = [row([0, 0], 0, query=[rng.gauss(0, 1), rng.gauss(5, 2)]) for _ in range(500)]
        gate = jc.fit_gate(rows, 0.99)
        refused = sum(jc.gate_distance(gate, r["query"]) > gate["threshold"] for r in rows)
        self.assertLessEqual(refused, 5)
        self.assertGreater(jc.gate_distance(gate, [10.0, 5.0]), gate["threshold"])

    def test_declared_scope(self):
        scope = jc.DEVTOOLS_SCOPE
        self.assertTrue(jc.in_declared_scope(row([0, 0], 0), scope))
        commit = row([0, 0], 0, question_text='Does this commit message describe this diff? '
                                              'Message: "Fix bug"')
        self.assertTrue(jc.in_declared_scope(commit, scope))
        for change in ({"question_text": "Is this request unsafe?"},
                       {"state_prefix": "How do I do lsd?"},
                       {"options": ["A", "B"]},
                       {"prompt_tokens": 4096}):
            self.assertFalse(jc.in_declared_scope(dict(row([0, 0], 0), **change), scope))

    def test_half_split_is_by_group_and_stable(self):
        self.assertEqual(jc.half_of("hard-v1/x/1"), jc.half_of("hard-v1/x/1"))
        halves = {jc.half_of(f"g{i}") for i in range(50)}
        self.assertEqual(halves, {"a", "b"})


class EvaluateTest(unittest.TestCase):
    def records(self):
        rows = []
        for split, seed in (("train", 5), ("calibration", 6), ("test", 7)):
            for r in synthetic(200, seed):
                r = dict(r, split=split, group=f"{split}-{r['group']}",
                         family="a" if seed % 2 else "b")
                rows.append(r)
        for r in synthetic(100, 8, scale=1000.0):  # labels unrelated to the logits
            rows.append(dict(r, set="ood", split="test", group=f"ood-{r['group']}",
                             question_text="Is this request unsafe?", query=[9.0, 9.0]))
        return rows

    def test_evaluate_separates_roles_and_reports_every_split(self):
        report = jc.evaluate(self.records(), ["s:calibration"], ["s:test"], ["ood:test"],
                             ["s:train"], jc.DEVTOOLS_SCOPE)
        self.assertEqual(set(report["splits"]), {"cal", "test", "ood"})
        self.assertEqual(report["splits"]["ood"]["scope_refused"], 1.0)
        self.assertEqual(report["splits"]["test"]["scope_refused"], 0.0)
        test = report["splits"]["test"]
        self.assertLess(test["calibrated"]["nll"], test["raw"]["nll"])
        acceptance = jc.acceptance(report, ["a", "b"], multiclass=False)
        self.assertEqual(set(acceptance), {"A1", "A2", "A3", "A4", "A5", "passed"})
        self.assertIn("| test |", jc.markdown(dict(report, acceptance=acceptance), "t"))

    def test_evaluate_refuses_shared_rows_or_groups(self):
        records = self.records()
        with self.assertRaises(jc.CalibrateError):
            jc.evaluate(records, ["s:calibration"], ["s:calibration"], [], ["s:train"])
        leaked = [dict(r, group="shared") if r["split"] in ("calibration", "test") else r
                  for r in records]
        with self.assertRaises(jc.CalibrateError):
            jc.evaluate(leaked, ["s:calibration"], ["s:test"], [], ["s:train"])
        with self.assertRaises(jc.CalibrateError):
            jc.select_rows(records, "s")

    def test_role_specs_filter_by_half_and_family(self):
        records = self.records()
        self.assertEqual(jc.select_rows(records, "s:test@a"), jc.select_rows(records, "s:test"))
        with self.assertRaises(jc.CalibrateError):  # the family must match in full
            jc.select_rows(records, "s:calibration@a")
        half = jc.select_rows(records, "s:test:a@a|b")
        self.assertTrue(half and all(jc.half_of(r["group"]) == "a" for r in half))
        self.assertLess(len(half), len(jc.select_rows(records, "s:test")))

    def test_exact_scope_admits_seen_questions_and_option_counts_only(self):
        seen = [row([0, 0], 0, question_text="Which?", options=["x", "y"])]
        scope = jc.exact_scope(seen)
        self.assertTrue(jc.in_declared_scope(seen[0], scope))
        self.assertFalse(jc.in_declared_scope(dict(seen[0], question_text="Other?"), scope))
        self.assertFalse(jc.in_declared_scope(dict(seen[0], options=["x", "y", "z"]), scope))
        self.assertFalse(jc.in_declared_scope(dict(seen[0], prompt_tokens=5000), scope))

    def test_calibration_artifact_pins_head_and_content(self):
        report = jc.evaluate(self.records(), ["s:calibration"], ["s:test"], ["ood:test"],
                             ["s:train"], jc.DEVTOOLS_SCOPE)
        report.update(acceptance=jc.acceptance(report, ["a", "b"], False),
                      scores_sha256="0" * 64, tool_sha256="1" * 64)
        artifact = jc.calibration_artifact(report, "ab" * 32, jc.DEVTOOLS_SCOPE)
        jc.verify_calibration_artifact(json.loads(json.dumps(artifact)), "ab" * 32)
        with self.assertRaises(jc.CalibrateError):
            jc.verify_calibration_artifact(artifact, "cd" * 32)
        tampered = dict(artifact, refusal=dict(artifact["refusal"], threshold=0.0))
        with self.assertRaises(jc.CalibrateError):
            jc.verify_calibration_artifact(tampered, "ab" * 32)

    def test_scores_are_checked_against_their_head_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            head, other, scores = tmp / "head", tmp / "other", tmp / "scores.jsonl"
            head.write_bytes(b"head")
            other.write_bytes(b"other")
            scores.write_text("{}\n")
            jc.score_summary_path(scores).write_text(json.dumps(
                {"head_sha256": jc.sha256_file(head), "scores_sha256": jc.sha256_file(scores)}))
            jc.check_scores(scores, head)
            with self.assertRaises(jc.CalibrateError):
                jc.check_scores(scores, other)
            scores.write_text("{}\n{}\n")
            with self.assertRaises(jc.CalibrateError):
                jc.check_scores(scores, head)

    def test_majority_baseline_uses_train_answers(self):
        train = [row([0, 0], 0, 0)] * 3 + [row([0, 0], 1, 0)]
        rows = [row([0, 0], 1, 1), row([0, 0], 1, 0)]  # Yes, No
        self.assertEqual(jc.majority_baseline(train, rows)["accuracy"], 0.5)


class OrchestrationTest(unittest.TestCase):
    def test_capture_runs_bounded_locked_chunks_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            requests = tmp / "requests.jsonl"
            requests.write_text("".join(json.dumps({"id": f"r{i}"}) + "\n" for i in range(5)))
            calls = []

            def runner(command, **kwargs):
                calls.append(command)
                out = Path(command[command.index("--out") + 1])
                out.mkdir()
                rows = len(Path(command[-1]).read_text().splitlines())
                manifest = []
                for i in range(rows):
                    (out / f"feature-{i:06d}.bin").write_bytes(bytes([i]))
                    manifest.append(json.dumps({"id": f"{out.name}-{i}",
                                                "feature_file": f"feature-{i:06d}.bin"}))
                (out / "features.jsonl").write_text("\n".join(manifest) + "\n")
                return subprocess.CompletedProcess(command, 0, json.dumps({"rows": rows}), "")

            work = tmp / "work"
            first = jc.capture(requests, work, Path("bin"), 2, max_chunks=2, runner=runner)
            self.assertEqual(first, {"chunks": 3, "ran": 2, "remaining": 1})
            self.assertEqual(calls[0][:5], ["timeout", "1800", "lockf", "-k", jc.LOCK])
            second = jc.capture(requests, work, Path("bin"), 2, runner=runner, lock_wait=900)
            self.assertEqual(second, {"chunks": 3, "ran": 1, "remaining": 0})
            self.assertEqual(calls[2][:7],
                             ["timeout", "1800", "lockf", "-k", "-t", "900", jc.LOCK])
            self.assertEqual(len(calls), 3)
            merged = jc.merge(work, tmp / "merged")
            self.assertEqual(merged["rows"], 5)
            names = [json.loads(line)["feature_file"]
                     for line in (tmp / "merged" / "features.jsonl").read_text().splitlines()]
            self.assertEqual(names, [f"feature-{i:06d}.bin" for i in range(5)])
            with self.assertRaises(jc.CalibrateError):
                jc.merge(work, tmp / "merged")

    def test_parity_compares_features_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = []
            for name, payload in (("a", b"\x01\x02"), ("b", b"\x01\x03")):
                d = Path(tmp) / name
                d.mkdir()
                (d / "feature-000000.bin").write_bytes(b"\x00\x00")
                (d / "feature-000001.bin").write_bytes(payload)
                rows = [{"id": i, "token_ids": [1], "positions": [0],
                         "feature_file": f"feature-00000{n}.bin"}
                        for n, i in enumerate(("same", "diff"))]
                (d / "features.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
                dirs.append(d)
            result = jc.parity(*dirs)
            self.assertEqual(result, {"shared": 2, "identical": 1, "mismatched": ["diff"]})

    def test_requests_use_the_native_predicate_rendering(self):
        request = jc.request_row("x", "Diff:\n@@ -1 +1 @@", jc.CR_QUESTION, False,
                                 {"split": "test"})
        self.assertEqual(request["kind"], "predicate")
        self.assertEqual(request["metadata"]["target"], 1)
        self.assertEqual(request["metadata"]["options"], ["Yes", "No"])
        with tempfile.TemporaryDirectory() as tmp:
            report = jc.write_requests(Path(tmp) / "out", [request], {"set": "t"})
            self.assertEqual(report["rows"], 1)
            with self.assertRaises(jc.CalibrateError):
                jc.write_requests(Path(tmp) / "out", [request], {})

    def test_codereviewer_state_matches_kev(self):
        oldf = "\n".join(f"line {i}" for i in range(1, 30))
        state = jc.codereviewer_state("@@ -15,2 +15,3 @@\n-a\n+b", oldf)
        self.assertEqual(state["lines_before_hunk"].split("\n"),
                         [f"line {i}" for i in range(4, 14 + 1)][-10:])
        self.assertEqual(jc.codereviewer_state("no hunk", oldf), {"diff": "no hunk"})


if __name__ == "__main__":
    unittest.main()
