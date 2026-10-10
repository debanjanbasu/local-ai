import ast
import contextlib
import importlib.util
import io
import json
import math
import random
import struct
import tempfile
import unittest
from pathlib import Path

import judgment_train as jt

HAS_TORCH = importlib.util.find_spec("torch") is not None
WIDTH = 4


def row(index, split, target, options=2, width=WIDTH, **overrides):
    positions = list(range(1, options + 2))
    value = {
        "id": f"sample-{index}",
        "token_ids": [index + 100] + list(range(options + 1)),
        "positions": positions,
        "width": width,
        "feature_file": f"feature-{index:06d}.bin",
        "metadata": {"split": split, "target": target, "source": "synthetic"},
    }
    value.update(overrides)
    return value


def fp16(values):
    return struct.pack(f"<{len(values)}e", *values)


def write_dataset(root, rows, payloads=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payloads = payloads or {}
    for index, item in enumerate(rows):
        name = item["feature_file"]
        if name not in payloads and isinstance(name, str) and "/" not in name:
            count = len(item["positions"]) * WIDTH
            payloads[name] = fp16([(index + 1) * 0.125 + i * 0.01 for i in range(count)])
    for name, data in payloads.items():
        (root / name).write_bytes(data)
    (root / "features.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def dev_row(index, split, target, options, family="probability", qtype="choice",
            width=WIDTH, **meta):
    item = row(index, split, target, options=options, width=width)
    item["metadata"].update({"dataset": "hard-v1", "renderer": "render.v1",
                             "family": family, "question_type": qtype,
                             "option_count": options, "group_id": f"g-{index}",
                             "kev_partition": "development" if split == "validation"
                             else "train", **meta})
    return item


def dev_rows():
    """train/validation/calibration only, option counts 2..6, no test."""
    splits = ["train"] * 5 + ["validation"] * 3 + ["calibration"] * 2
    return [dev_row(i, split, i % (2 + i % 5), 2 + i % 5,
                    family=("judge", "ambiguous")[i % 2], qtype=("choice", "noul")[i % 2])
            for i, split in enumerate(splits)]


def valid_rows():
    return [
        row(0, "train", 1, options=3),
        row(1, "validation", 0),
        row(2, "calibration", 1),
        row(3, "test", 0, options=4),
    ]


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "features"

    def tearDown(self):
        self.tmp.cleanup()

    def refuses(self, rows, message, payloads=None):
        write_dataset(self.root, rows, payloads)
        with self.assertRaisesRegex(jt.DatasetError, message):
            jt.load_dataset(self.root, WIDTH)

    def test_accepts_exact_format_with_variable_option_counts(self):
        samples = jt.load_dataset(write_dataset(self.root, valid_rows()), WIDTH)
        self.assertEqual(jt.split_counts(samples),
                         {"train": 1, "validation": 1, "calibration": 1, "test": 1})
        self.assertEqual([s.options for s in samples], [3, 2, 2, 4])
        self.assertEqual(len(samples[3].features), 5 * WIDTH * 2)

    def test_refuses_path_traversal_and_symlinks(self):
        rows = valid_rows()
        rows[0]["feature_file"] = "../feature-000000.bin"
        (Path(self.tmp.name) / "feature-000000.bin").write_bytes(fp16([0.0] * 4 * WIDTH))
        self.refuses(rows, "plain feature")
        rows = valid_rows()
        outside = Path(self.tmp.name) / "outside.bin"
        outside.write_bytes(fp16([0.0] * 4 * WIDTH))
        write_dataset(self.root, rows)
        (self.root / "feature-000000.bin").unlink()
        (self.root / "feature-000000.bin").symlink_to(outside)
        with self.assertRaisesRegex(jt.DatasetError, "regular file"):
            jt.load_dataset(self.root, WIDTH)

    def test_refuses_missing_malformed_and_nonfinite_features(self):
        rows = valid_rows()
        write_dataset(self.root, rows)
        (self.root / "feature-000002.bin").unlink()
        with self.assertRaisesRegex(jt.DatasetError, "missing"):
            jt.load_dataset(self.root, WIDTH)
        self.refuses(valid_rows(), "bytes, expected",
                     {"feature-000001.bin": fp16([0.5] * (3 * WIDTH))[:-3]})
        for bad in (math.inf, -math.inf, math.nan):
            values = [0.25] * (3 * WIDTH)
            values[5] = bad
            self.refuses(valid_rows(), "non-finite", {"feature-000001.bin": fp16(values)})

    def test_refuses_duplicates_across_splits(self):
        rows = valid_rows()
        rows[3]["id"] = rows[0]["id"]  # train/test duplicate id
        self.refuses(rows, "duplicate id .*train.*test")
        rows = valid_rows()
        rows[2]["token_ids"] = rows[1]["token_ids"]
        self.refuses(rows, "duplicate prompt")
        shared = fp16([0.5] * (3 * WIDTH))
        self.refuses(valid_rows(), "duplicate features",
                     {"feature-000001.bin": shared, "feature-000002.bin": shared})

    def test_refuses_bad_targets_splits_widths_and_shapes(self):
        cases = [
            ({"metadata": {"split": "train", "target": 3}}, "target"),
            ({"metadata": {"split": "train", "target": -1}}, "target"),
            ({"metadata": {"split": "train", "target": 1.0}}, "target"),
            ({"metadata": {"split": "train", "target": True}}, "target"),
            ({"metadata": {"split": "train"}}, "target"),
            ({"metadata": {"split": "dev", "target": 0}}, "unsupported split"),
            ({"width": 8}, "width"),
            ({"positions": [1, 1, 2, 3]}, "sorted"),
            ({"positions": [3, 2, 1, 0]}, "sorted"),
            ({"positions": [1, 2]}, "two option"),
            ({"positions": [1, 2, 9, 10]}, "past token_ids"),
            ({"id": ""}, "id"),
            ({"token_ids": [1, "2"]}, "token_ids"),
        ]
        for override, message in cases:
            with self.subTest(override=override):
                rows = valid_rows()
                rows[0].update(override)
                self.refuses(rows, message)
        rows = valid_rows()
        rows[0]["extra"] = 1
        self.refuses(rows, "exactly the keys")

    def test_refuses_invalid_json_and_empty_manifest(self):
        write_dataset(self.root, valid_rows())
        with open(self.root / "features.jsonl", "a") as handle:
            handle.write("{not json\n")
        with self.assertRaisesRegex(jt.DatasetError, "invalid JSON"):
            jt.load_dataset(self.root, WIDTH)
        (self.root / "features.jsonl").write_text("")
        with self.assertRaisesRegex(jt.DatasetError, "no rows"):
            jt.load_dataset(self.root, WIDTH)

    def test_cli_validate_and_train_require_every_split(self):
        write_dataset(self.root, valid_rows())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            jt.main(["validate", str(self.root), "--width", str(WIDTH)])
        self.assertEqual(json.loads(out.getvalue())["option_counts"], [2, 3, 4])
        rows = [r for r in valid_rows() if r["metadata"]["split"] != "calibration"]
        root = write_dataset(Path(self.tmp.name) / "partial", rows)
        with self.assertRaisesRegex(SystemExit, "empty: \\['calibration'\\]"):
            jt.main(["train", str(root), "--output", str(Path(self.tmp.name) / "out"),
                     "--width", str(WIDTH)])
        self.assertFalse((Path(self.tmp.name) / "out").exists())


    def test_development_validate_accepts_mixed_two_to_six_options_without_test(self):
        write_dataset(self.root, dev_rows())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            jt.main(["validate", str(self.root), "--width", str(WIDTH), "--development"])
        summary = json.loads(out.getvalue())
        self.assertEqual(summary["mode"], "development")
        self.assertEqual(summary["option_counts"], [2, 3, 4, 5, 6])
        self.assertEqual(summary["sample_counts"],
                         {"train": 5, "validation": 3, "calibration": 2, "test": 0})
        self.assertEqual(summary["option_count_histogram"]["train"],
                         {"2": 1, "3": 1, "4": 1, "5": 1, "6": 1})
        self.assertEqual(summary["feature_scope"],
                         {"dataset": "hard-v1", "renderer": "render.v1"})
        # The default four-split rule still refuses the same directory.
        with self.assertRaisesRegex(jt.DatasetError, "empty: \\['test'\\]"):
            jt.check_mode(jt.load_dataset(self.root, WIDTH))

    def test_development_mode_refuses_test_leaks_and_inconsistencies(self):
        cases = [
            (lambda r: r.append(dev_row(10, "test", 0, 2)), "refuses test rows"),
            (lambda r: r[0]["metadata"].update(kev_partition="test"), "locked-partition"),
            (lambda r: r[1]["metadata"].update(option_count=5), "option_count"),
            (lambda r: r[5]["metadata"].update(group_id="g-0"), "spans train"),
            (lambda r: r.__setitem__(slice(8, 10), []), "empty: \\['calibration'\\]"),
        ]
        for edit, message in cases:
            with self.subTest(message=message):
                rows = dev_rows()
                edit(rows)
                root = write_dataset(Path(self.tmp.name) / message[:6], rows)
                samples = jt.load_dataset(root, WIDTH)
                with self.assertRaisesRegex(jt.DatasetError, message):
                    jt.check_mode(samples, "development")
        with self.assertRaises(ValueError):
            jt.check_mode(samples, "lockedtest")

    def test_cli_development_train_refuses_test_rows_before_torch(self):
        rows = dev_rows() + [dev_row(10, "test", 0, 2)]
        write_dataset(self.root, rows)
        out = Path(self.tmp.name) / "out"
        with self.assertRaisesRegex(SystemExit, "development mode refuses test rows"):
            jt.main(["train", str(self.root), "--output", str(out), "--width", str(WIDTH),
                     "--development"])
        self.assertFalse(out.exists())

    def test_feature_scope_omits_mixed_or_partial_metadata(self):
        rows = dev_rows()
        rows[0]["metadata"]["renderer"] = "render.v2"
        del rows[1]["metadata"]["dataset"]
        samples = jt.load_dataset(write_dataset(self.root, rows), WIDTH)
        self.assertEqual(jt.feature_scope(samples), {
            "dataset": None, "dataset_values": ["hard-v1"],
            "renderer": None, "renderer_values": ["render.v1", "render.v2"]})


class MathTests(unittest.TestCase):
    def test_baselines_fit_on_train_and_evaluate_elsewhere(self):
        sample = lambda target, options, **meta: jt.Sample(
            "x", "train", target, options, 1, b"", "", (0,), meta)
        train = [sample(1, 2), sample(1, 2), sample(0, 2), sample(2, 3)]
        evaluated = [sample(1, 2, family="a"), sample(0, 3, family="b"),
                     sample(3, 4, family="b")]
        result = jt.baselines(train, evaluated)
        self.assertEqual(result["majority_table"], {"2": [1, 2], "3": [0, 0, 1]})
        # Smoothed priors: 2 options [2/5, 3/5]; 3 options [1/4, 1/4, 2/4]; 4 unseen: uniform.
        self.assertAlmostEqual(result["majority"]["nll"],
                               -(math.log(3 / 5) + math.log(1 / 4) + math.log(1 / 4)) / 3)
        self.assertAlmostEqual(result["majority"]["accuracy"], 1 / 3)
        self.assertAlmostEqual(result["uniform"]["nll"],
                               (math.log(2) + math.log(3) + math.log(4)) / 3)
        self.assertAlmostEqual(result["uniform"]["expected_accuracy"], (1/2 + 1/3 + 1/4) / 3)
        groups = jt.group_metrics(evaluated, [[0.0, 1.0], [1.0, 0.0, 0.0], [0, 0, 0, 1.0]],
                                  1.0, jt.majority_table(train))
        self.assertEqual(set(groups), {"option_count", "family"})  # question_type absent
        self.assertEqual(groups["family"]["b"]["model"]["accuracy"], 1.0)
        self.assertEqual(groups["family"]["b"]["majority"]["count"], 2)
        self.assertEqual(groups["option_count"]["2"]["majority"]["accuracy"], 1.0)


    def test_evaluate_matches_hand_computed_values(self):
        # A: p = [0.75, 0.25], target 1 (wrong). B: p = [0.5, 0.25, 0.25], target 0.
        logits = [[math.log(3) + 1.5, 1.5], [math.log(2) - 0.7, -0.7, -0.7]]
        result = jt.evaluate(logits, [1, 0])
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["nll"], 1.5 * math.log(2))
        self.assertAlmostEqual(result["brier"], (1.125 + 0.375) / 2)
        self.assertAlmostEqual(result["ece"], (0.75 + 0.5) / 2)
        scaled = jt.evaluate(logits[:1], [1], temperature=2.0)
        self.assertAlmostEqual(scaled["nll"], math.log(1 + math.sqrt(3)))

    def test_temperature_matches_closed_form_two_options(self):
        # Every gap is 2 in favour of option 1, right 3 times in 4:
        # sigmoid(2 / T) = 3/4  =>  T = 2 / ln 3.
        logits = [[1.0, 3.0], [-2.0, 0.0], [0.5, 2.5], [4.0, 6.0]]
        temperature = jt.fit_temperature(logits, [1, 1, 0, 1])
        self.assertAlmostEqual(temperature, 2 / math.log(3), places=9)
        self.assertLess(jt.evaluate(logits, [1, 1, 0, 1], temperature)["nll"],
                        jt.evaluate(logits, [1, 1, 0, 1])["nll"])

    def test_temperature_matches_closed_form_three_options(self):
        # Top logit leads both others by 3 and is right half the time:
        # e^(3/T) / (e^(3/T) + 2) = 1/2  =>  T = 3 / ln 2.
        logits = [[0.0, 3.0, 0.0], [1.0, 1.0, 4.0], [-1.0, 2.0, -1.0], [5.0, 2.0, 2.0]]
        temperature = jt.fit_temperature(logits, [1, 0, 0, 0])
        self.assertAlmostEqual(temperature, 3 / math.log(2), places=9)

    def test_temperature_bounds_and_degenerate_inputs(self):
        self.assertEqual(jt.fit_temperature([[0.0, 1.0], [1.0, 0.0]], [1, 0]),
                         jt.TEMPERATURE_BOUNDS[0])
        self.assertEqual(jt.fit_temperature([[2.0, 2.0]], [0]), 1.0)
        with self.assertRaises(ValueError):
            jt.fit_temperature([[0.0, math.nan]], [0])
        with self.assertRaises(ValueError):
            jt.evaluate([[0.0, 1.0]], [2])
        with self.assertRaises(ValueError):
            jt.evaluate([[0.0, 1.0]], [0], temperature=0.0)

    def test_export_round_trip_and_documented_layout(self):
        tensors = {
            "q.weight": ((1, 2), [1.0, 2.0]),
            "q.bias": ((1,), [0.5]),
            "k.weight": ((1, 2), [3.0, -1.0]),
            "k.bias": ((1,), [0.0]),
            "temperature": ((1,), [2.0]),
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "head.safetensors"
            jt.write_safetensors(path, tensors, {"width": 2})
            data = path.read_bytes()
            (size,) = struct.unpack_from("<Q", data)
            self.assertEqual((8 + size) % 8, 0)
            loaded, metadata = jt.read_safetensors(path)
        self.assertEqual(metadata, {"width": "2"})
        self.assertEqual(loaded, tensors)
        # q = 1 + 2 + 0.5 = 3.5; k = [3, -1]; / sqrt(1) / 2.
        self.assertEqual(jt.reference_logits(loaded, [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]),
                         [5.25, -1.75])


class StagingTests(unittest.TestCase):
    def test_stages_private_offline_job_bound_to_dataset(self):
        import hashlib

        import kaggle_judgment_job

        with tempfile.TemporaryDirectory() as root:
            staged = kaggle_judgment_job.stage_job(
                "me/judgment-head", Path(root) / "stage", "me/bonsai-features",
                "capture", {"epochs": 2})
            metadata = json.loads((staged / "kernel-metadata.json").read_text())
            script = (staged / "convert.py").read_text()
            namespace = {
                node.targets[0].id: ast.literal_eval(node.value)
                for node in ast.parse(script).body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.isupper()
            }
            with self.assertRaises(FileExistsError):
                kaggle_judgment_job.stage_job("me/judgment-head", staged, "me/bonsai-features")
            with self.assertRaisesRegex(ValueError, "relative path"):
                kaggle_judgment_job.stage_job("me/x", Path(root) / "y", "me/d", "../up")
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(metadata["dataset_sources"], ["me/bonsai-features"])
        self.assertEqual(namespace["CONFIG"]["epochs"], 2)
        self.assertEqual(namespace["FEATURES_SUBDIR"], "capture")
        source = Path(jt.__file__).read_text(encoding="utf-8")
        self.assertEqual(namespace["SOURCES"]["judgment_train.py"], source)
        self.assertEqual(namespace["SOURCE_SHA256"]["judgment_train.py"],
                         hashlib.sha256(source.encode()).hexdigest())
        self.assertNotIn("TOKEN", kaggle_judgment_job.JOB)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TorchTrainingTests(unittest.TestCase):
    def test_tiny_synthetic_training_exports_consistent_head(self):
        width = 8
        rng = random.Random(7)
        rows, payloads = [], {}
        splits = ["train"] * 24 + ["validation"] * 6 + ["calibration"] * 6 + ["test"] * 6
        for index, split in enumerate(splits):
            options = 2 + index % 3
            target = rng.randrange(options)
            decide = [rng.uniform(-1, 1) for _ in range(width)]
            matrix = [[rng.uniform(-1, 1) for _ in range(width)] for _ in range(options)]
            matrix[target] = [v + rng.uniform(-0.1, 0.1) for v in decide]
            item = row(index, split, target, options=options, width=width)
            rows.append(item)
            payloads[item["feature_file"]] = fp16([v for r in matrix + [decide] for v in r])
        with tempfile.TemporaryDirectory() as root:
            data = write_dataset(Path(root) / "features", rows, payloads)
            output = Path(root) / "head"
            config = {"epochs": 3, "width": width, "head_dim": 4, "batch_size": 8}
            report = jt.train_and_export(data, output, config)
            with self.assertRaises(FileExistsError):
                jt.train_and_export(data, output, config)
            tensors, metadata = jt.read_safetensors(output / "head.safetensors")
            saved = json.loads((output / "report.json").read_text())
            samples = [s for s in jt.load_dataset(data, width) if s.split == "test"]
        self.assertEqual(saved["sample_counts"],
                         {"train": 24, "validation": 6, "calibration": 6, "test": 6})
        self.assertEqual(len(saved["provenance"]["feature_sha256"]), len(rows))
        self.assertEqual(metadata["experimental"], "true")
        self.assertEqual(tensors["q.weight"][0], (4, width))
        self.assertGreater(report["temperature"], 0)
        self.assertEqual(tensors["temperature"][1][0], report["temperature"])
        logits = []
        for sample in samples:
            values = struct.unpack(f"<{(sample.options + 1) * width}e", sample.features)
            matrix = [values[i * width : (i + 1) * width] for i in range(sample.options + 1)]
            logits.append(jt.reference_logits(tensors, matrix[:-1], matrix[-1]))
        expected = jt.evaluate(logits, [s.target for s in samples])
        for key in ("accuracy", "nll", "brier", "ece"):
            self.assertAlmostEqual(expected[key], report["metrics"]["test"][key], places=4)
        self.assertNotIn("mode", saved)
        self.assertNotIn("mode", metadata)

    def test_tiny_development_training_reports_validation_without_test(self):
        width = 8
        rng = random.Random(11)
        rows, payloads = [], {}
        splits = ["train"] * 30 + ["validation"] * 10 + ["calibration"] * 10
        for index, split in enumerate(splits):
            options = 2 + index % 5
            target = rng.randrange(options)
            decide = [rng.uniform(-1, 1) for _ in range(width)]
            matrix = [[rng.uniform(-1, 1) for _ in range(width)] for _ in range(options)]
            matrix[target] = [v + rng.uniform(-0.1, 0.1) for v in decide]
            item = dev_row(index, split, target, options, width=width,
                           family=("judge", "tradeoff")[index % 2])
            rows.append(item)
            payloads[item["feature_file"]] = fp16([v for r in matrix + [decide] for v in r])
        with tempfile.TemporaryDirectory() as root:
            data = write_dataset(Path(root) / "features", rows, payloads)
            config = {"epochs": 2, "width": width, "head_dim": 4, "batch_size": 8}
            with self.assertRaisesRegex(jt.DatasetError, "empty: \\['test'\\]"):
                jt.train_and_export(data, Path(root) / "final", config)
            output = Path(root) / "head"
            report = jt.train_and_export(data, output, config, mode="development")
            tensors, metadata = jt.read_safetensors(output / "head.safetensors")
            saved = json.loads((output / "report.json").read_text())
            summary = jt.summary_of(saved)
        self.assertEqual(saved, json.loads(json.dumps(report)))
        self.assertEqual(saved["mode"], "development")
        self.assertFalse(saved["evaluation"]["held_out_final_test"])
        self.assertFalse(any("test" in key for key in saved["metrics"]))
        self.assertEqual(saved["sample_counts"]["test"], 0)
        self.assertEqual(saved["baselines"]["evaluated_split"], "validation")
        self.assertEqual(saved["baselines"]["majority"]["count"], 10)
        self.assertEqual(set(saved["groups"]["family"]), {"judge", "tradeoff"})
        self.assertEqual(sorted(saved["groups"]["option_count"]), ["2", "3", "4", "5", "6"])
        self.assertEqual(metadata["mode"], "development")
        self.assertEqual(metadata["renderer"], "render.v1")
        self.assertEqual(metadata["dataset"], "hard-v1")
        self.assertEqual(json.loads(metadata["split_counts"]),
                         {"train": 30, "validation": 10, "calibration": 10, "test": 0})
        self.assertIn("calibration split", metadata["calibration_scope"])
        self.assertFalse(any("model" in key for key in metadata))
        self.assertAlmostEqual(tensors["temperature"][1][0], report["temperature"], places=6)
        self.assertNotIn("test", summary)
        self.assertEqual(summary["validation_not_held_out_test"],
                         saved["metrics"]["validation_calibrated"])


if __name__ == "__main__":
    unittest.main()
