"""Stage (never submit) a private Kaggle job for EXPERIMENTAL judgment head training.

The job trains only the pointer head in `judgment_train.py` on features already
captured by the native worker and uploaded by the user as a private Kaggle
dataset. It downloads nothing, needs no Hugging Face token and runs without
internet. Staging embeds `judgment_train.py` and its tests verbatim with SHA256
digests (via `kaggle_staging.stage_script`), then attaches the named dataset.
Kaggle writes `/kaggle/working/judgment-head/{head.safetensors,report.json}`.
`--development` binds `judgment_train`'s development mode (no test rows; the
printed summary reports validation, which is not a held-out test).

`--regenerate-hard-v1` instead stages a private CPU job that regenerates Kev's
hard-v1 train partition with Kev's own deterministic generator (pinned commit,
pinned lockfile, pinned Qwen3.5-4B-Base tokenizer, CPU torch) and prepares it
with `judgment_prepare.py --dataset hard-v1`. It uses internet and, if the
named Kaggle secret exists, that Hugging Face token (never printed). The
locked hard-v1 test partition is never extracted, written or scored: the
generator builds it in memory only because train deduplicates against it, and
the job keeps nothing of it but its sha256 match. Kaggle writes
`/kaggle/working/{hard-v1-kev,hard-v1-rows,hard-v1-job.json}`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
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
                                  source_sha256=SOURCE_SHA256["judgment_train.py"], mode=MODE)
print(json.dumps(trainer.summary_of(report), indent=2), flush=True)
"""


def stage_job(kernel: str, output: Path, dataset: str, features_subdir: str = "",
              config: dict | None = None, *, gpu: bool = True, mode: str = "final") -> Path:
    """Write a private, offline kernel bound to one user-provided features dataset.

    `mode="development"` trains on train/validation/calibration only and refuses
    any test row (see `judgment_train.check_mode`)."""
    from judgment_train import MODES, _validate_config
    from kaggle_staging import stage_script

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", dataset):
        raise ValueError("dataset must be owner/slug")
    if features_subdir and not re.fullmatch(r"[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*",
                                            features_subdir):
        raise ValueError("features subdir must be a relative path of plain names")
    bindings = {
        "DATASET": dataset,
        "FEATURES_SUBDIR": features_subdir,
        "CONFIG": _validate_config(config or {}),
        "MODE": mode,
        "OUTPUT_NAME": OUTPUT_NAME,
    }
    staged = stage_script(kernel, output, bindings, JOB, SOURCE_FILES, gpu=gpu)
    path = staged / "kernel-metadata.json"
    metadata = json.loads(path.read_text())
    metadata["dataset_sources"] = [dataset]
    metadata["enable_internet"] = False
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return staged


# ---------------------------------------------------------------------------
# hard-v1 train regeneration (CPU, internet, optional Hugging Face secret)
# ---------------------------------------------------------------------------

HARD_SOURCE_FILES = (
    "kaggle_judgment_job.py",
    "judgment_prepare.py",
    "test_judgment_prepare.py",
    "judgment_train.py",
)
KEV_REVISION = "62c91838b9a6adc5b386cbeae8ed73daa36ce220"
KEV_ARCHIVE_URL = f"https://codeload.github.com/jaredpalmer/kev/tar.gz/{KEV_REVISION}"
# Files at KEV_REVISION (codeload archives are not byte-stable; their files are).
KEV_CODE_SHA256 = {
    "scripts/build_hard_v1.py": "8767ead9a80ee27d7b45d64c10431d7b04e1cf250dba8411a92cf17f0e02ef8b",
    "scripts/hard_v1_common.py": "0f5f3577918aef14a8b4c4efd8e04a2b6c4197b898ea0cea8f20a1dacceba4d6",
    "scripts/hard_v1_policy.py": "3097bc661bfaf9f9090a88c1e2dc75169ecf5eff92632e3662919cbf0ada79da",
    "scripts/hard_v1_families.py": "95468c125a5c73d682d88e6449c65aa8af7140c85a70ba8054f76ffb74897392",
    "scripts/hard_v1_numeric.py": "40342d4f4d3ae78a0a54cadd089b47306fad1f6bf4fa198b21e3c2bc269746ab",
    "kev/api.py": "7bffacfb762c626b8bc2f670f350295af5ccb0883dbe90239c7d2f8e5ef56582",
    "kev/data.py": "e77a2fffeb8ee7e05b118893be8e38b8358aef97c3ca2e143cdce208201b92f9",
    "kev/model.py": "43426d28084e52b65613286efb74e4f6a91703e42aab88920735ac2f3e0a7c64",
    "kev/suite.py": "b6a731aaa9370ab180dc60b990339089be8d3e6fb125150f5ba66305e072a91e",
    "uv.lock": "a9922dbb89acdef78299fd2b4a8c3f7f0fa1b2bc08b55595b6926fa785a9c466",
    "LICENSE": "6b08bb37982c233aa12bcbdf19106da12f3f4fcf800773ea42e28ebeddd34fb8",
}
KEV_EXTRACT = (
    "kev/",
    "scripts/",
    "evals/hard-v1/manifest.json",
    "evals/hard-v1/development.jsonl",
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
)
LOCKED_TEST = "evals/hard-v1/test.jsonl"
UV_VERSION = "0.12.23"
VENV_PYTHON = "3.12"
TORCH_CPU = ("torch==2.8.0+cpu", "https://download.pytorch.org/whl/cpu")
CUDA_ONLY = re.compile(r"(torch|triton|nvidia-[a-z0-9-]+)==")
HARD_OUTPUT_KEV = "hard-v1-kev"
HARD_OUTPUT_ROWS = "hard-v1-rows"

# Runs inside Kev's checkout with its pinned environment. Writes train and
# development; test is built in memory (train deduplicates against it), only
# its digest is compared, and it is discarded unwritten.
REGEN_SCRIPT = r"""
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from scripts import build_hard_v1 as builder

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
manifest = json.loads(Path("evals/hard-v1/manifest.json").read_text(encoding="utf-8"))
start = time.time()
checker = builder.Checker()
context = {**checker.train_ctx, "truncate": False}
parts, report = builder.build(builder.SIZES, checker)
result = {"sizes": builder.SIZES, "seed": builder.SEED, "tokenizer": list(builder.TOKENIZER),
          "context_matches_manifest": context == manifest["context"],
          "generation_report_matches_manifest": report == manifest["generation_report"], "files": {}}
for split in ("train", "development", "test"):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in parts.pop(split)).encode("utf-8")
    name = split + ".jsonl"
    entry = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "records": data.count(b"\n")}
    pinned = {k: manifest["files"][name][k] for k in entry}
    written = split != "test"
    if written:
        (out / name).write_bytes(data)
    result["files"][name] = {**(entry if written else {}), "matches_manifest": entry == pinned, "written": written}
    del data
result["seconds"] = round(time.time() - start, 1)
(out / "regeneration.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
"""

HARD_JOB = r"""
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


frozen("judgment_train")
prepare = frozen("judgment_prepare")
tests = frozen("test_judgment_prepare")
job = frozen("kaggle_judgment_job")
# Synthetic-fixture self-tests of both adapters; no corpus is read.
suite = unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromTestCase(getattr(tests, name))
    for name in ("PrepareTest", "HardV1Test"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped:
    raise RuntimeError("judgment_prepare self-tests failed or were skipped in the cloud")
summary = job.run_hard_v1_job(Path("/kaggle/working"), prepare, HF_SECRET, PREPARE_OPTIONS,
                              SOURCE_SHA256)
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
if summary["errors"]:
    raise RuntimeError("hard-v1 regeneration did not reproduce the pinned data: "
                       + "; ".join(summary["errors"]))
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_kev(archive: Path, dest: Path) -> tuple[Path, int]:
    """Extract only KEV_EXTRACT from a codeload archive; never the locked test.

    Returns (checkout root, number of locked-test members skipped)."""
    members, root, skipped = [], None, 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            top, _, rel = member.name.partition("/")
            root = root or top
            if top != root or member.name.startswith("/") or ".." in member.name.split("/"):
                raise ValueError(f"unexpected archive member {member.name!r}")
            if rel == LOCKED_TEST:
                skipped += 1
                continue
            if not any(rel == p or (p.endswith("/") and rel.startswith(p)) for p in KEV_EXTRACT):
                continue
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"refusing non-regular archive member {member.name!r}")
            members.append(member)
        if root is None:
            raise ValueError("empty archive")
        tar.extractall(dest, members=members, filter="data")
    checkout = Path(dest) / root
    if (checkout / LOCKED_TEST).exists():
        raise RuntimeError("locked hard-v1 test partition was extracted")
    return checkout, skipped


def verify_kev(checkout: Path, manifest_sha256: str) -> dict:
    """Check pinned code, manifest and development digests; report the
    manifest's own code_sha256 entries that differ from the pinned commit."""
    observed = {rel: sha256_file(checkout / rel) for rel in KEV_CODE_SHA256}
    wrong = sorted(rel for rel, digest in observed.items() if digest != KEV_CODE_SHA256[rel])
    if wrong:
        raise RuntimeError(f"Kev files differ from {KEV_REVISION}: {wrong}")
    manifest_path = checkout / "evals/hard-v1/manifest.json"
    digest = sha256_file(manifest_path)
    if digest != manifest_sha256:
        raise RuntimeError(f"manifest sha256 {digest} != pinned {manifest_sha256}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dev = sha256_file(checkout / "evals/hard-v1/development.jsonl")
    if dev != manifest["files"]["development.jsonl"]["sha256"]:
        raise RuntimeError("committed development.jsonl does not match the manifest")
    return {
        "code_sha256": observed,
        "manifest_sha256": digest,
        "development_sha256": dev,
        "manifest_code_sha256_differs": sorted(
            rel
            for rel, pinned in manifest.get("code_sha256", {}).items()
            if (checkout / rel).is_file() and sha256_file(checkout / rel) != pinned
        ),
    }


def cpu_requirements(exported: str) -> str:
    """Kev's locked requirements minus the CUDA torch stack (CPU torch is pinned
    separately; the generator only imports torch)."""
    kept = [line for line in exported.splitlines() if not CUDA_ONLY.match(line.strip())]
    return "\n".join(kept) + "\n"


def hf_token(secret: str):
    """The named Kaggle secret, or None. The value is never printed or saved."""
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(secret) or None
    except Exception:  # noqa: BLE001 - no secret attached, or not on Kaggle
        return None


def run_hard_v1_job(working: Path, prepare, secret: str, options: dict,
                    source_sha256: dict, *, run=subprocess.run, fetch=None) -> dict:
    """Download Kev, build its pinned CPU environment, regenerate hard-v1 and
    prepare rows. Always writes `hard-v1-job.json`; returns that summary,
    whose `errors` list is empty only if every pinned digest was reproduced."""
    import os
    import tempfile

    if fetch is None:
        from urllib.request import urlretrieve as fetch
    working = Path(working)
    scratch = Path(tempfile.mkdtemp(prefix="kev-hard-v1-"))
    summary = {
        "kev_revision": KEV_REVISION,
        "job_source_sha256": source_sha256,
        "prepare_options": options,
        "errors": [],
    }
    try:
        archive = scratch / "kev.tar.gz"
        fetch(KEV_ARCHIVE_URL, archive)
        summary["archive_sha256"] = sha256_file(archive)
        checkout, summary["locked_test_members_skipped"] = extract_kev(archive, scratch)
        summary.update(verify_kev(checkout, prepare.HARD_MANIFEST_SHA256))

        def call(*args, **kwargs):
            return run(list(map(str, args)), check=True, cwd=checkout, **kwargs)

        uv = (sys.executable, "-m", "uv")
        call(sys.executable, "-m", "pip", "install", "--quiet", f"uv=={UV_VERSION}")
        exported = call(*uv, "export", "--frozen", "--no-dev", "--no-emit-project",
                        "--no-hashes", "--no-header", "--no-annotate", "--format",
                        "requirements-txt", capture_output=True, text=True).stdout
        (checkout / "requirements-cpu.txt").write_text(cpu_requirements(exported))
        venv = checkout / ".venv"
        python = venv / "bin" / "python"
        call(*uv, "venv", "--python", VENV_PYTHON, venv)
        call(*uv, "pip", "install", "--python", python, "--no-deps", "--index-url",
             TORCH_CPU[1], TORCH_CPU[0])
        call(*uv, "pip", "install", "--python", python, "--no-deps", "-r",
             "requirements-cpu.txt")
        summary["environment"] = call(*uv, "pip", "freeze", "--python", python,
                                      capture_output=True, text=True).stdout.splitlines()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "HUGGING"))}
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        token = hf_token(secret)
        summary["hf_token"] = "present" if token else "absent (anonymous download)"
        if token:
            env["HF_TOKEN"] = token
        kev_out = working / HARD_OUTPUT_KEV
        try:
            call(python, "-c", REGEN_SCRIPT, kev_out, env=env)
        finally:
            del env, token
        shutil.copy2(checkout / "evals/hard-v1/manifest.json", kev_out / "manifest.json")
        shutil.copy2(checkout / "LICENSE", kev_out / "LICENSE")
        regen = json.loads((kev_out / "regeneration.json").read_text())
        summary["regeneration"] = regen
        for name, entry in sorted(regen["files"].items()):
            if not entry["matches_manifest"]:
                summary["errors"].append(f"regenerated {name} does not match the manifest")
        if (kev_out / "test.jsonl").exists():
            summary["errors"].append("test.jsonl was written")
        if not summary["errors"]:
            report = prepare.prepare_hard_v1(kev_out, working / HARD_OUTPUT_ROWS, **options)
            summary["rows"] = {
                "output": report["output"],
                "excluded_rows": report["excluded_rows"],
                "renderer": report["renderer"],
            }
    except subprocess.CalledProcessError as error:
        summary["errors"].append(f"command failed ({error.returncode}): {error.cmd[:4]}")
    except Exception as error:  # noqa: BLE001 - keep what was regenerated and say why
        summary["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    working.mkdir(parents=True, exist_ok=True)
    (working / "hard-v1-job.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def stage_hard_v1_job(kernel: str, output: Path, *, hf_secret: str = "HF_TOKEN",
                      option_descriptions: bool = True,
                      include_long_policy: bool = False) -> Path:
    """Write a private CPU kernel that regenerates and prepares hard-v1 train."""
    from kaggle_staging import stage_script

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", hf_secret):
        raise ValueError("hf secret must be a plain Kaggle secret label")
    options = {
        "option_descriptions": bool(option_descriptions),
        "include_families": ["long_policy"] if include_long_policy else [],
    }
    bindings = {"HF_SECRET": hf_secret, "PREPARE_OPTIONS": options}
    staged = stage_script(kernel, output, bindings, HARD_JOB, HARD_SOURCE_FILES, gpu=False)
    path = staged / "kernel-metadata.json"
    metadata = json.loads(path.read_text())
    metadata["enable_internet"] = True
    metadata["dataset_sources"] = []
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return staged


def main(argv=None) -> None:
    from judgment_train import DEFAULT_CONFIG

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kernel", required=True, help="new private owner/slug")
    parser.add_argument("--output", type=Path, required=True, help="new staging directory")
    parser.add_argument("--dataset",
                        help="private Kaggle dataset owner/slug holding captured features")
    parser.add_argument("--features-subdir", default="",
                        help="directory inside the dataset containing features.jsonl")
    parser.add_argument("--cpu", action="store_true", help="stage without a GPU")
    parser.add_argument("--development", action="store_true",
                        help="development-only training: train/validation/calibration, "
                             "refuse test rows, report validation (not a held-out test)")
    parser.add_argument("--regenerate-hard-v1", action="store_true",
                        help="stage the CPU hard-v1 train regeneration job instead")
    parser.add_argument("--hf-secret", default="HF_TOKEN",
                        help="Kaggle secret label holding a Hugging Face token (optional)")
    parser.add_argument("--no-option-descriptions", action="store_true")
    parser.add_argument("--include-long-policy", action="store_true")
    for key, value in DEFAULT_CONFIG.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    if args.regenerate_hard_v1:
        if args.dataset or args.features_subdir:
            parser.error("--regenerate-hard-v1 takes no features dataset")
        if args.development:
            parser.error("--development applies to head training, not --regenerate-hard-v1")
        print(stage_hard_v1_job(args.kernel, args.output, hf_secret=args.hf_secret,
                                option_descriptions=not args.no_option_descriptions,
                                include_long_policy=args.include_long_policy))
        return
    if not args.dataset:
        parser.error("--dataset is required")
    config = {key: getattr(args, key) for key in DEFAULT_CONFIG}
    print(stage_job(args.kernel, args.output, args.dataset, args.features_subdir, config,
                    gpu=not args.cpu, mode="development" if args.development else "final"))


if __name__ == "__main__":
    main()
