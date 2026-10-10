"""Staging writes a private kernel that embeds hashed sources; bad input is refused."""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judgment_prepare as jp
import kaggle_judgment_job as kj
import kaggle_staging
from test_judgment_prepare import hard_partitions, write_hard


class Tests(unittest.TestCase):
    def test_rejects_malformed_kernel_and_non_bool_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for kernel in ("owner", "owner/slug/extra", "owner/sl ug", ""):
                with self.assertRaises(ValueError):
                    kaggle_staging.stage_script(kernel, Path(tmp) / "x", {}, "", ())
            with self.assertRaises(ValueError):
                kaggle_staging.stage_script("o/s", Path(tmp) / "y", {}, "", (), gpu=1)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_cpu_stage_embeds_sources_hashes_and_private_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = kaggle_staging.stage_script(
                "owner/bonsai-slug",
                Path(tmp) / "stage",
                {"PIN": 7},
                "print(PIN, ARCHIVE_NAME, sorted(SOURCE_SHA256))\n",
                ("kaggle_staging.py",),
                gpu=False,
            )
            script = (out / "convert.py").read_text(encoding="utf-8")
            metadata = json.loads((out / "kernel-metadata.json").read_text())
            source = Path(kaggle_staging.__file__).read_text(encoding="utf-8")
            digest = hashlib.sha256(source.encode()).hexdigest()
            self.assertIn("PIN = 7\n", script)
            self.assertIn("ARCHIVE_NAME = 'bonsai-slug.tar'\n", script)
            self.assertIn(f"'kaggle_staging.py': '{digest}'", script)
            self.assertIn(repr(source), script)
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_gpu"])
            self.assertNotIn("machine_shape", metadata)
            self.assertEqual(metadata["id"], "owner/bonsai-slug")
            self.assertEqual(metadata["title"], "bonsai-slug")
            with self.assertRaises(FileExistsError):
                kaggle_staging.stage_script("owner/bonsai-slug", out, {}, "", ())

    def test_gpu_stage_pins_t4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = kaggle_staging.stage_script("o/g", Path(tmp) / "gpu", {}, "", ())
            metadata = json.loads((out / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["enable_gpu"])
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")


def bindings(script: str) -> dict:
    """The upper-case literal assignments a staged convert.py starts with."""
    import ast

    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(script).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()}


class TrainingJobModeTests(unittest.TestCase):
    def test_mode_is_bound_validated_and_summarised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            final = kj.stage_job("o/final", Path(tmp) / "final", "o/features")
            dev = kj.stage_job("o/dev", Path(tmp) / "dev", "o/features", "capture",
                               {"seed": 2}, gpu=False, mode="development")
            final_bound = bindings((final / "convert.py").read_text(encoding="utf-8"))
            dev_bound = bindings((dev / "convert.py").read_text(encoding="utf-8"))
            for bad in ("Development", "test", ""):
                with self.assertRaisesRegex(ValueError, "mode must be one of"):
                    kj.stage_job("o/bad", Path(tmp) / "bad", "o/features", mode=bad)
            self.assertFalse((Path(tmp) / "bad").exists())
        self.assertEqual(final_bound["MODE"], "final")
        self.assertEqual(dev_bound["MODE"], "development")
        self.assertEqual(dev_bound["CONFIG"]["seed"], 2)
        # The cloud job forwards MODE and prints the trainer's mode-aware summary
        # (a development report has no test metrics to print).
        self.assertIn("mode=MODE)", kj.JOB)
        self.assertIn("trainer.summary_of(report)", kj.JOB)
        self.assertNotIn('["test"]', kj.JOB)

    def test_cli_development_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("sys.stdout"):
                kj.main(["--kernel", "o/dev", "--output", str(Path(tmp) / "dev"),
                         "--dataset", "o/features", "--cpu", "--development", "--seed", "1"])
                kj.main(["--kernel", "o/fin", "--output", str(Path(tmp) / "fin"),
                         "--dataset", "o/features", "--cpu"])
            dev = bindings((Path(tmp) / "dev" / "convert.py").read_text(encoding="utf-8"))
            fin = bindings((Path(tmp) / "fin" / "convert.py").read_text(encoding="utf-8"))
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                kj.main(["--kernel", "o/regen", "--output", str(Path(tmp) / "regen"),
                         "--regenerate-hard-v1", "--development"])
            self.assertFalse((Path(tmp) / "regen").exists())
        self.assertEqual((dev["MODE"], dev["CONFIG"]["seed"]), ("development", 1))
        self.assertEqual(fin["MODE"], "final")


FAKE_BUILDER = '''
import json
from pathlib import Path
SIZES = {"train": 1, "development": 1, "test": 1}
SEED = "fixture"
TOKENIZER = ("Qwen/Qwen3.5-4B-Base", "rev")
class Checker:
    train_ctx = {"max_state": 1}
def build(sizes, checker):
    parts = json.loads(Path("scripts/fixture_parts.json").read_text(encoding="utf-8"))
    return parts, {"train": {}}
'''
SECRET = "hf_s3cret_value_never_logged"


def jsonl(rows) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


class HardV1JobTests(unittest.TestCase):
    def test_stage_is_private_cpu_with_frozen_sources_and_no_secret_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = kj.stage_hard_v1_job("owner/hard-v1-regen", Path(tmp) / "stage")
            script = (out / "convert.py").read_text(encoding="utf-8")
            metadata = json.loads((out / "kernel-metadata.json").read_text())
            compile(script, "convert.py", "exec")
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_gpu"])
            self.assertTrue(metadata["enable_internet"])
            self.assertEqual(metadata["dataset_sources"], [])
            self.assertIn("HF_SECRET = 'HF_TOKEN'\n", script)
            self.assertIn("'include_families': []", script)
            for name in kj.HARD_SOURCE_FILES:
                source = Path(kj.__file__).with_name(name).read_text(encoding="utf-8")
                self.assertIn(f"'{name}': '{hashlib.sha256(source.encode()).hexdigest()}'", script)
            for bad in ("HF TOKEN", "", "a;b"):
                with self.assertRaises(ValueError):
                    kj.stage_hard_v1_job("o/s", Path(tmp) / "bad", hf_secret=bad)
            self.assertFalse((Path(tmp) / "bad").exists())

    def test_cpu_requirements_drop_only_the_cuda_stack(self) -> None:
        exported = "numpy==2.5.3\ntorch==2.8.0 ; sys_platform == 'linux'\ntriton==3.4.0\n" \
                   "nvidia-cublas-cu12==12.8.4.1\ntorchvision-extra==1\n"
        self.assertEqual(kj.cpu_requirements(exported), "numpy==2.5.3\ntorchvision-extra==1\n")

    def _archive(self, tmp: Path, parts: dict, link: bool = False) -> tuple[Path, str, dict]:
        """A codeload-shaped archive with a fake generator and real hard-v1 fixtures."""
        root = tmp / "src" / f"kev-{kj.KEV_REVISION}"
        evals = root / "evals" / "hard-v1"
        test_bytes = jsonl(parts["test"])

        def mutate(manifest):
            manifest["context"] = {"max_state": 1, "truncate": False}
            manifest["generation_report"] = {"train": {}}
            manifest["files"]["test.jsonl"] = {
                "sha256": hashlib.sha256(test_bytes).hexdigest(), "bytes": len(test_bytes),
                "records": len(parts["test"]), "questions": len(parts["test"])}

        digest = write_hard(evals, {"train": parts["train"], "development": parts["development"]},
                            mutate)
        (evals / "train.jsonl").unlink()
        (evals / "test.jsonl").write_bytes(test_bytes)
        (root / "scripts").mkdir()
        (root / "scripts" / "build_hard_v1.py").write_text(FAKE_BUILDER)
        (root / "scripts" / "fixture_parts.json").write_text(json.dumps(parts, ensure_ascii=False))
        (root / "LICENSE").write_text("Apache-2.0 fixture\n")
        (root / "README.md").write_text("not extracted\n")
        if link:
            (root / "scripts" / "evil").symlink_to("/etc/passwd")
        archive = tmp / "kev.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root, arcname=root.name)
        pins = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
                for rel in ("scripts/build_hard_v1.py", "LICENSE")}
        return archive, digest, pins

    def test_extract_skips_locked_test_and_refuses_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive, _, _ = self._archive(tmp, {**hard_partitions(), "test": [{"x": 1}]})
            checkout, skipped = kj.extract_kev(archive, tmp / "x")
            self.assertEqual(skipped, 1)
            self.assertFalse((checkout / kj.LOCKED_TEST).exists())
            self.assertFalse((checkout / "README.md").exists())
            self.assertTrue((checkout / "evals/hard-v1/development.jsonl").is_file())
            linked, _, _ = self._archive(tmp / "l", {**hard_partitions(), "test": []}, link=True)
            with self.assertRaisesRegex(ValueError, "non-regular"):
                kj.extract_kev(linked, tmp / "y")

    def test_job_regenerates_verifies_and_prepares_without_keeping_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parts = {**hard_partitions(), "test": [{"locked": "row"}]}
            archive, digest, pins = self._archive(tmp, parts)
            calls = []

            def run(cmd, check, cwd, env=None, **kwargs):
                calls.append((cmd, env))
                if "-c" in cmd:
                    at = cmd.index("-c")
                    return subprocess.run([sys.executable, *cmd[at:]], check=True, cwd=cwd, env=env)
                out = "torch==2.8.0\npydantic==2.13.5\n" if "export" in cmd else "pydantic==2.13.5\n"
                return subprocess.CompletedProcess(cmd, 0, stdout=out)

            def fetch(url, path):
                self.assertEqual(url, kj.KEV_ARCHIVE_URL)
                shutil.copy(archive, path)

            secrets = types.ModuleType("kaggle_secrets")
            secrets.UserSecretsClient = type(
                "Client", (), {"get_secret": lambda self, label: SECRET if label == "HF_TOKEN" else None})
            prepare = types.SimpleNamespace(
                HARD_MANIFEST_SHA256=digest,
                prepare_hard_v1=functools.partial(jp.prepare_hard_v1, manifest_sha256=digest))
            working = tmp / "working"
            with mock.patch.object(kj, "KEV_CODE_SHA256", pins), \
                    mock.patch.dict(sys.modules, {"kaggle_secrets": secrets}):
                summary = kj.run_hard_v1_job(working, prepare, "HF_TOKEN", {}, {"a": "b"},
                                             run=run, fetch=fetch)
            self.assertEqual(summary["errors"], [])
            self.assertEqual(summary["locked_test_members_skipped"], 1)
            self.assertEqual(summary["hf_token"], "present")
            regen = summary["regeneration"]
            self.assertTrue(all(f["matches_manifest"] for f in regen["files"].values()))
            self.assertFalse(regen["files"]["test.jsonl"]["written"])
            self.assertNotIn("sha256", regen["files"]["test.jsonl"])
            kev_out = working / kj.HARD_OUTPUT_KEV
            self.assertEqual(sorted(p.name for p in kev_out.iterdir()),
                             ["LICENSE", "development.jsonl", "manifest.json",
                              "regeneration.json", "train.jsonl"])
            self.assertEqual((kev_out / "train.jsonl").read_bytes(), jsonl(parts["train"]))
            self.assertTrue((working / kj.HARD_OUTPUT_ROWS / "rows.jsonl").is_file())
            self.assertEqual(summary["rows"]["output"]["row_count"],
                             len((working / kj.HARD_OUTPUT_ROWS / "rows.jsonl").read_bytes()
                                 .splitlines()))
            regen_env = next(env for cmd, env in calls if "-c" in cmd)
            self.assertEqual(regen_env["HF_TOKEN"], SECRET)
            self.assertTrue(all(env is None for cmd, env in calls if "-c" not in cmd))
            for path in working.rglob("*"):
                if path.is_file():
                    self.assertNotIn(SECRET.encode(), path.read_bytes())
            torch_installs = [cmd for cmd, _ in calls if kj.TORCH_CPU[0] in cmd]
            self.assertEqual(len(torch_installs), 1)
            self.assertIn(kj.TORCH_CPU[1], torch_installs[0])

    def test_job_reports_a_digest_mismatch_and_prepares_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parts = {**hard_partitions(), "test": [{"locked": "row"}]}
            _, digest, pins = self._archive(tmp, parts)
            parts["train"] = parts["train"][1:]   # the "generator" now differs from the manifest
            fixture = tmp / "src" / f"kev-{kj.KEV_REVISION}" / "scripts" / "fixture_parts.json"
            fixture.write_text(json.dumps(parts))
            archive2 = tmp / "kev2.tar.gz"
            with tarfile.open(archive2, "w:gz") as tar:
                tar.add(fixture.parents[1], arcname=fixture.parents[1].name)

            def run(cmd, check, cwd, env=None, **kwargs):
                if "-c" in cmd:
                    at = cmd.index("-c")
                    return subprocess.run([sys.executable, *cmd[at:]], check=True, cwd=cwd, env=env)
                return subprocess.CompletedProcess(cmd, 0, stdout="")

            prepare = types.SimpleNamespace(HARD_MANIFEST_SHA256=digest)
            with mock.patch.object(kj, "KEV_CODE_SHA256", pins):
                summary = kj.run_hard_v1_job(
                    tmp / "w", prepare, "HF_TOKEN", {}, {}, run=run,
                    fetch=lambda url, path: shutil.copy(archive2, path))
            self.assertEqual(summary["errors"],
                             ["regenerated train.jsonl does not match the manifest"])
            self.assertEqual(summary["hf_token"], "absent (anonymous download)")
            self.assertFalse((tmp / "w" / kj.HARD_OUTPUT_ROWS).exists())
            self.assertTrue((tmp / "w" / "hard-v1-job.json").is_file())


if __name__ == "__main__":
    unittest.main()
