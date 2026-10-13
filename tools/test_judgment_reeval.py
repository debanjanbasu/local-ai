"""Tests for tools/judgment_reeval.py (pure Python; the gate test needs NumPy)."""

import importlib.util
import json
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judgment_calibrate as jc
import judgment_reeval as jr

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def scored(i, family, margin, label, group, split="calibration", gate=0.5, set_name="s"):
    yes = i % 2
    raw = [0.0, 0.0]
    raw[yes], raw[1 - yes] = margin / 2, -margin / 2
    question = (jc.CR_QUESTION if family == jc.CR_TASK
                else 'Does this commit message describe this diff? Message: "x y"')
    return {"id": f"{set_name}-{split}-{family}-{i}", "set": set_name, "split": split, "raw": raw,
            "target": yes if label else 1 - yes, "yes_index": yes, "group": group,
            "family": family, "query": [0.0], "options": ["Yes", "No"],
            "prompt_tokens": 50, "question_text": question, "state_prefix": "Diff:\n@@",
            "gate_distance": gate}


def family_rows(split, n, seed, set_name="s", noisy_scale=4.0, clean_scale=0.5):
    """CodeReviewer-like noisy rows and CommitPackFT-like clean rows."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        margin = rng.uniform(-6, 6)
        label = rng.random() < jc.sigmoid(margin / noisy_scale)
        rows.append(scored(i, jc.CR_TASK, margin, label, f"{split}-cr{i % 9}", split,
                           set_name=set_name))
        margin = rng.uniform(-6, 6)
        label = rng.random() < jc.sigmoid(margin / clean_scale)
        rows.append(scored(i, jc.CM_TASK, margin, label, f"{split}-cm{i}", split,
                           set_name=set_name))
    return rows


class RolesTest(unittest.TestCase):
    def test_codereviewer_roles_are_deterministic_and_project_disjoint(self):
        kev = {f"p{i}": "train" for i in range(60)}
        kev.update({"d1": "development", "t1": "test"})
        permissive = set(kev) | {"never-used"}
        roles = jr.cr_project_roles(kev, permissive)
        self.assertEqual(roles, jr.cr_project_roles(dict(reversed(list(kev.items()))),
                                                    permissive))
        self.assertEqual(roles["d1"], "validation")
        self.assertEqual(roles["t1"], "train")
        values = list(roles.values())
        self.assertEqual(values.count("cal"), jr.CAL_CR_PROJECTS)
        self.assertEqual(values.count("test"), jr.TEST_CR_PROJECTS)
        self.assertNotIn("x", roles)  # non-permissive projects get no role

    def test_commitpackft_roles_follow_the_buckets(self):
        counts = {}
        for i in range(4000):
            role = jr.cm_role(f"repo{i}")
            counts[role] = counts.get(role, 0) + 1
        for role, share in jr.CM_ROLE_BUCKETS.items():
            self.assertAlmostEqual(counts[role] / 4000, share / 100, delta=0.03)
        self.assertEqual(jr.cm_role("repo7"), jr.cm_role("repo7"))


class EvaluateTest(unittest.TestCase):
    def records(self):
        rows = family_rows("calibration", 600, 1) + family_rows("test", 600, 2)
        rows += [dict(r, split="train") for r in family_rows("train", 50, 3)]
        ood = [scored(i, "aegis_unsafe", 0.0, True, f"o{i}", "ood", gate=50.0)
               for i in range(40)]
        for r in ood:
            r["question_text"] = "Is this request unsafe?"
        return rows + ood

    def test_per_family_thresholds_hold_a_noisy_family_back(self):
        report = jr.evaluate(self.records(), ["s:calibration"], ["s:test"], ["s:ood"],
                             ["s:train"], jc.DEVTOOLS_SCOPE)
        self.assertEqual(set(report["thresholds"]), set(jr.FAMILIES))
        test = report["splits"]["test"]
        cm = test["by_family"][jc.CM_TASK]["selective"]
        cr = test["by_family"][jc.CR_TASK]["selective"]
        self.assertGreater(cm["coverage"], 0.5)
        if report["thresholds"][jc.CR_TASK] is None:
            self.assertEqual(cr["coverage"], 0.0)
        else:
            self.assertGreater(report["thresholds"][jc.CR_TASK],
                               report["thresholds"][jc.CM_TASK])
        self.assertEqual(report["splits"]["ood"]["scope_refused"], 1.0)
        self.assertEqual(report["splits"]["ood"]["gate_refused"], 1.0)
        self.assertLessEqual(test["gate_refused"], 0.05)
        accepted = jc.acceptance(report, list(jr.FAMILIES), False)
        self.assertEqual(set(accepted), {"A1", "A2", "A3", "A4", "A5", "passed"})
        self.assertTrue(accepted["A4"])

    def test_selective_at_uses_each_rows_threshold(self):
        preds = [{"confidence": 0.9, "correct": True}, {"confidence": 0.9, "correct": False},
                 {"confidence": 0.6, "correct": True}]
        point = jr.selective_at(preds, [0.8, None, 0.5])
        self.assertEqual(point["answered"], 2)
        self.assertEqual(point["risk"], 0.0)
        self.assertEqual(jr.selective_at(preds, [None] * 3)["risk"], None)

    def test_shared_groups_between_roles_are_refused(self):
        rows = self.records()
        for r in rows:
            if r["split"] == "test":
                r["group"] = "calibration-cr1"
        with self.assertRaises(jr.ReevalError):
            jr.evaluate(rows, ["s:calibration"], ["s:test"], [], ["s:train"],
                        jc.DEVTOOLS_SCOPE)

    def test_artifact_pins_head_gate_and_content(self):
        report = jr.evaluate(self.records(), ["s:calibration"], ["s:test"], ["s:ood"],
                             ["s:train"], jc.DEVTOOLS_SCOPE)
        report.update(acceptance={"passed": False}, scores_sha256="s", tool_sha256={})
        artifact = jr.calibration_artifact(report, "h" * 64, "g" * 64)
        jr.verify_calibration_artifact(artifact, "h" * 64, "g" * 64)
        with self.assertRaises(jc.CalibrateError):
            jr.verify_calibration_artifact(artifact, "x" * 64, "g" * 64)
        with self.assertRaises(jr.ReevalError):
            jr.verify_calibration_artifact(artifact, "h" * 64, "x" * 64)
        tampered = json.loads(json.dumps(artifact))
        tampered["families"][jc.CR_TASK]["threshold"] = 0.0
        with self.assertRaises(jc.CalibrateError):
            jr.verify_calibration_artifact(tampered, "h" * 64, "g" * 64)


class OfflineAndHttpParityTest(unittest.TestCase):
    ARTIFACT: ClassVar[dict] = {
        "families": {
            jc.CR_TASK: {"calibration": {"method": "platt", "slope": 1.0, "bias": 0.0},
                         "threshold": None},
            jc.CM_TASK: {"calibration": {"method": "temperature", "temperature": 1.0},
                         "threshold": 0.8}},
        "gate": {"threshold": 1.0},
        "scope": jc.DEVTOOLS_SCOPE,
    }

    def test_offline_decisions_apply_scope_gate_and_family_thresholds_in_order(self):
        rows = [scored(0, jc.CM_TASK, 4.0, True, "a", gate=0.5),
                scored(1, jc.CM_TASK, 0.2, True, "b", gate=0.5),
                scored(2, jc.CM_TASK, 4.0, True, "c", gate=3.0),
                scored(3, jc.CR_TASK, 4.0, True, "d", gate=0.5),
                dict(scored(4, jc.CM_TASK, 4.0, True, "e"), prompt_tokens=4096),
                dict(scored(5, "aegis_unsafe", 4.0, True, "f"), question_text="Unsafe?")]
        out = jr.offline_decisions(rows, self.ARTIFACT)
        self.assertEqual([r.get("refusal", "out") for r in out],
                         [None, "low_confidence", "gate", "family_refused", "prompt_too_long",
                          "out"])
        self.assertAlmostEqual(out[0]["p_yes"], jc.sigmoid(4.0))
        self.assertFalse(out[5]["in_scope"])

    def test_http_parity_compares_answers_with_offline_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            requests = [{"id": "a", "state": "Diff:\n@@", "question": "q", "kind": "predicate"},
                        {"id": "b", "state": "Diff:\n@@", "question": "q", "kind": "choice",
                         "options": ["x", "y"]}]
            expected = [{"id": "a", "in_scope": True, "refusal": None, "p_yes": 0.9},
                        {"id": "b", "in_scope": False}]
            (tmp / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
            (tmp / "e.jsonl").write_text("".join(json.dumps(r) + "\n" for r in expected))
            sent = []

            class Reply:
                def __init__(self, body):
                    self.body = body

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def read(self):
                    return self.body

            def opener(request, timeout):
                body = json.loads(request.data)
                sent.append(body)
                question = body["questions"][0]
                answer = ({"type": "predicate", "name": question["name"], "probability": 0.9}
                          if question["type"] == "predicate"
                          else {"type": "refusal", "name": question["name"]})
                return Reply(json.dumps({"answers": [answer]}).encode())

            report = jr.http_parity("http://x/v1/decisions", tmp / "r.jsonl", tmp / "e.jsonl",
                                    ["a", "b"], opener=opener)
            self.assertEqual((report["agree"], report["answered"]), (2, 1))
            self.assertEqual(sent[1]["questions"][0]["choices"], [{"value": "x"},
                                                                  {"value": "y"}])
            expected[0]["p_yes"] = 0.5
            (tmp / "e.jsonl").write_text("".join(json.dumps(r) + "\n" for r in expected))
            report = jr.http_parity("http://x/v1/decisions", tmp / "r.jsonl", tmp / "e.jsonl",
                                    ["a"], opener=opener)
            self.assertEqual(report["mismatched"], ["a"])


def write_features(directory, rows, width=jr.jt.WIDTH, value=None):
    directory.mkdir(parents=True)
    lines = []
    for i, (split, group, task) in enumerate(rows):
        name = f"feature-{i:06d}.bin"
        rng = random.Random(i)
        values = [value if value is not None else rng.gauss(0, 1) for _ in range(3 * width)]
        (directory / name).write_bytes(struct.pack(f"<{3 * width}e", *values))
        lines.append(json.dumps({"id": f"{directory.name}-{i}", "token_ids": [1, 2, 3, 4],
                                 "positions": [1, 2, 3], "width": width, "feature_file": name,
                                 "metadata": {"split": split, "target": 0, "group_id": group,
                                              "task": task}}))
    (directory / "features.jsonl").write_text("\n".join(lines) + "\n")


class AssembleAndGateTest(unittest.TestCase):
    def test_assemble_maps_splits_and_drops_held_out_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_features(tmp / "a", [("train", "g1", "t"), ("train", "held", "t"),
                                       ("test", "g2", "t"), ("calibration", "g3", "u")],
                           width=4)
            result = jr.assemble([("a", tmp / "a")], {"a": {
                "split_map": {"train": "train", "test": "validation", "calibration": "train"},
                "drop_groups": ["held"], "keep_tasks": ["t"]}}, tmp / "out")
            self.assertEqual(result["rows"], 2)
            rows = [json.loads(line) for line in
                    (tmp / "out" / "features.jsonl").read_text().split("\n") if line]
            self.assertEqual([r["metadata"]["split"] for r in rows], ["train", "validation"])
            self.assertEqual(rows[1]["metadata"]["original_split"], "test")
            self.assertTrue((tmp / "out" / "feature-000001.bin").samefile(
                tmp / "a" / "feature-000002.bin"))
            with self.assertRaises(jr.ReevalError):
                jr.assemble([("a", tmp / "a")], {"a": {"split_map": {}}}, tmp / "out")

    @unittest.skipUnless(HAS_NUMPY, "the gate needs NumPy")
    def test_gate_is_fitted_on_train_rows_and_scores_far_rows_higher(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_features(tmp / "f", [("train", f"g{i}", "t") for i in range(40)]
                           + [("validation", "v", "t")], width=jr.jt.WIDTH)
            summary = jr.fit_gate(tmp / "f", tmp / "gate.safetensors", components=8)
            self.assertEqual(summary["rows"], 40)
            gate = jr.load_gate(tmp / "gate.safetensors")
            near = np.random.default_rng(0).normal(0, 1, jr.jt.WIDTH)
            far = near + 25.0
            self.assertGreater(jr.gate_distance(gate, far), 10 * jr.gate_distance(gate, near))


if __name__ == "__main__":
    unittest.main()
