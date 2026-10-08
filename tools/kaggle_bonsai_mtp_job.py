"""Stage a private CPU-only Kaggle job for the pinned Bonsai 2 MTP teacher head.

The community BF16 head (`ProCreations/Ternary-Bonsai-2-27B-MTP`,
`model_mtp.safetensors`) is the distillation teacher for the ternary head the
engine ships (`tools/mtp_train/train_ternary.py`); the runtime never reads it.
It travels the same bounded, digest-checked path as the target checkpoint: this
module rebinds the pinned identity of `kaggle_bonsai_job` to the head repository
and reuses its downloader, archive format and no-clobber installer unchanged.
Staging never submits a job; installation never replaces an existing directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PIN = {
    "MODEL_ID": "ProCreations/Ternary-Bonsai-2-27B-MTP",
    "REVISION": "efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51",
    "MODEL_FILE": "model_mtp.safetensors",
    "MODEL_SIZE": 849_400_392,
    "MODEL_SHA256": "7a4a18b2d02116ef184d1b0ee4af46d829825ff2c042f79cf37ef8a03c399218",
    # Small text files kept beside the head for attribution and provenance.
    "SUPPORT_FILES": ("README.md", "LICENSE", "NOTICE", "TRAINING.md", "mtp_config.json"),
}
# Where the teacher is installed for training: beside, never inside, the runtime
# head directory the engine's `DEFAULT_BONSAI_MTP_ARTIFACT` names, which the
# no-clobber installer would refuse anyway.
DEFAULT_DESTINATION = Path("models/bonsai2-27b-mtp-teacher")
SOURCE_FILES = (
    "kaggle_bonsai_job.py",
    "kaggle_bonsai_mtp_job.py",
    "test_kaggle_bonsai_job.py",
)


def bind(module, pin: dict | None = None):
    """Rebind a `kaggle_bonsai_job` module instance to the head's pinned identity."""
    pin = PIN if pin is None else pin
    if set(pin) != set(PIN):
        raise ValueError("pin must define exactly the pinned identity keys")
    for name in PIN:
        if not hasattr(module, name):
            raise ValueError(f"downloader lacks pinned constant {name}")
    if type(pin["MODEL_SIZE"]) is not int or pin["MODEL_SIZE"] <= 0:
        raise ValueError("pinned head size must be a positive integer")
    if len(pin["MODEL_SHA256"]) != 64 or set(pin["MODEL_SHA256"]) - set("0123456789abcdef"):
        raise ValueError("pinned head digest must be 64 lowercase hex digits")
    if len(pin["REVISION"]) != 40 or set(pin["REVISION"]) - set("0123456789abcdef"):
        raise ValueError("pinned revision must be a 40-hex commit")
    for name in (pin["MODEL_FILE"], *pin["SUPPORT_FILES"]):
        module.safe_name(name)
    if len(set(pin["SUPPORT_FILES"])) != len(pin["SUPPORT_FILES"]) or pin[
        "MODEL_FILE"
    ] in pin["SUPPORT_FILES"]:
        raise ValueError("pinned file names must be distinct")
    for name, value in pin.items():
        setattr(module, name, value)
    return module


def pinned_downloader(pin: dict | None = None):
    """A fresh `kaggle_bonsai_job` instance bound to the head; the shared module is untouched."""
    path = Path(__file__).with_name("kaggle_bonsai_job.py")
    spec = importlib.util.spec_from_file_location("kaggle_bonsai_job_mtp_head", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("kaggle_bonsai_job.py is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return bind(module, pin)


def install_head(archive: Path, destination: Path = DEFAULT_DESTINATION) -> dict:
    """Verify a recovered teacher archive against the pin and publish a new directory."""
    return pinned_downloader().install_archive(Path(archive), Path(destination))


JOB = r"""
import hashlib
import sys
import types
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


module = frozen("kaggle_bonsai_job")
head = frozen("kaggle_bonsai_mtp_job")
head.bind(module)
tests = frozen("test_kaggle_bonsai_job")
checks = tests.run_cloud_checks()
output = module.run_job(archive_name=ARCHIVE_NAME, source_sha256=SOURCE_SHA256,
                        checks_run=checks)
print("Verified pinned Bonsai MTP teacher head archive:", output.name, module.MODEL_SIZE,
      module.MODEL_SHA256, flush=True)
"""


def stage_job(kernel: str, output: Path) -> Path:
    from kaggle_staging import stage_script

    return stage_script(kernel, output, {}, JOB, SOURCE_FILES, gpu=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--kernel", help="stage for this new private owner/slug")
    modes.add_argument("--install", type=Path, help="recovered teacher archive to install")
    parser.add_argument("--output", type=Path, help="staging directory (with --kernel)")
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"new teacher directory (with --install; default: {DEFAULT_DESTINATION})",
    )
    args = parser.parse_args()
    if args.kernel:
        if args.output is None:
            parser.error("--kernel requires --output")
        print(stage_job(args.kernel, args.output))
    else:
        manifest = install_head(args.install, args.destination)
        json.dump(manifest["upstream"], sys.stdout, indent=2, sort_keys=True)
        print()


if __name__ == "__main__":
    main()
