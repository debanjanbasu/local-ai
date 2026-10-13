import ast
import json
import tempfile
import unittest
from pathlib import Path

import convert_whisper
import kaggle_whisper_job as job


class StagingTests(unittest.TestCase):
    def test_stages_a_private_cpu_kernel_with_the_converter_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = job.stage("someone/whisper-fetch", Path(tmp, "job"))
            meta = json.loads((out / "kernel-metadata.json").read_text())
            self.assertTrue(meta["is_private"])
            self.assertFalse(meta["enable_gpu"])
            self.assertNotIn("machine_shape", meta)
            script = (out / "convert.py").read_text()
            compile(script, "convert.py", "exec")
            namespace = {node.targets[0].id: ast.literal_eval(node.value)
                         for node in ast.parse(script).body
                         if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                         and node.targets[0].id in ("PINS", "REVISION", "SDIST_SHA256")}
            self.assertEqual(namespace["PINS"], {k: list(v) for k, v in
                                                 convert_whisper.SOURCE_FILES.items()})
            self.assertEqual(namespace["REVISION"], job.REVISION)
            self.assertEqual(len(namespace["SDIST_SHA256"]), 64)

    def test_refuses_an_existing_directory_and_bad_slugs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                job.stage("someone/whisper-fetch", Path(tmp))
            with self.assertRaises(ValueError):
                job.stage("not a slug", Path(tmp, "x"))


if __name__ == "__main__":
    unittest.main()
