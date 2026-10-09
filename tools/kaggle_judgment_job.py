"""Stage (never submit) a private Kaggle job for EXPERIMENTAL judgment head training.

The job trains only the pointer head in `judgment_train.py` on features already
captured by the native worker and uploaded by the user as a private Kaggle
dataset. It downloads nothing, needs no Hugging Face token and runs without
internet. Staging embeds `judgment_train.py` and its tests verbatim with SHA256
digests (via `kaggle_staging.stage_script`), then attaches the named dataset.
Kaggle writes `/kaggle/working/judgment-head/{head.safetensors,report.json}`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SOURCE_FILES = ("judgment_train.py", "test_judgment_train.py")
OUTPUT_NAME = "judgment-head"

JOB = r"""
import hashlib
import sys
import types
import unittest
from pathlib import Path

for name, source in SOURCES.items():
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256[name]:
        raise RuntimeError("Frozen source hash mismatch: " + name)


def frozen(name):
    module = types.ModuleType(name)
    module.__file__ = "<frozen:" + name + ".py>"
    sys.modules[name] = module
    exec(compile(SOURCES[name + ".py"], module.__file__, "exec"), module.__dict__)
    return module


trainer = frozen("judgment_train")
tests = frozen("test_judgment_train")
# Staging tests need the local tool tree; the cloud runs the dataset/math/torch tests.
suite = unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromTestCase(getattr(tests, name))
    for name in ("ValidationTests", "MathTests", "TorchTrainingTests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped:
    raise RuntimeError("judgment_train self-tests failed or were skipped in the cloud")

owner, slug = DATASET.split("/")
candidates = [Path("/kaggle/input") / slug / FEATURES_SUBDIR,
              Path("/kaggle/input/datasets") / owner / slug / FEATURES_SUBDIR]
found = [path for path in candidates if (path / "features.jsonl").is_file()]
if len(found) != 1:
    raise RuntimeError("expected exactly one features.jsonl at " + repr(candidates))
report = trainer.train_and_export(found[0], Path("/kaggle/working") / OUTPUT_NAME, CONFIG,
                                  source_sha256=SOURCE_SHA256["judgment_train.py"])
print(json.dumps({"sample_counts": report["sample_counts"],
                  "temperature": report["temperature"],
                  "test": report["metrics"]["test"]}, indent=2), flush=True)
"""


def stage_job(kernel: str, output: Path, dataset: str, features_subdir: str = "",
              config: dict | None = None, *, gpu: bool = True) -> Path:
    """Write a private, offline kernel bound to one user-provided features dataset."""
    from judgment_train import _validate_config
    from kaggle_staging import stage_script

    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", dataset):
        raise ValueError("dataset must be owner/slug")
    if features_subdir and not re.fullmatch(r"[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*",
                                            features_subdir):
        raise ValueError("features subdir must be a relative path of plain names")
    bindings = {
        "DATASET": dataset,
        "FEATURES_SUBDIR": features_subdir,
        "CONFIG": _validate_config(config or {}),
        "OUTPUT_NAME": OUTPUT_NAME,
    }
    staged = stage_script(kernel, output, bindings, JOB, SOURCE_FILES, gpu=gpu)
    path = staged / "kernel-metadata.json"
    metadata = json.loads(path.read_text())
    metadata["dataset_sources"] = [dataset]
    metadata["enable_internet"] = False
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return staged


def main(argv=None) -> None:
    from judgment_train import DEFAULT_CONFIG

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kernel", required=True, help="new private owner/slug")
    parser.add_argument("--output", type=Path, required=True, help="new staging directory")
    parser.add_argument("--dataset", required=True,
                        help="private Kaggle dataset owner/slug holding captured features")
    parser.add_argument("--features-subdir", default="",
                        help="directory inside the dataset containing features.jsonl")
    parser.add_argument("--cpu", action="store_true", help="stage without a GPU")
    for key, value in DEFAULT_CONFIG.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    config = {key: getattr(args, key) for key in DEFAULT_CONFIG}
    print(stage_job(args.kernel, args.output, args.dataset, args.features_subdir, config,
                    gpu=not args.cpu))


if __name__ == "__main__":
    main()
