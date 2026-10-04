"""Staging writes a private kernel that embeds hashed sources; bad input is refused."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_staging


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


if __name__ == "__main__":
    unittest.main()
