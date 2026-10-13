"""Offline checks for the Kokoro Kaggle job staging (nothing is submitted)."""

import json
import tempfile
import unittest
from pathlib import Path

import kaggle_kokoro_job as job


class StagingTests(unittest.TestCase):
    def test_reference_job_is_a_private_cpu_kernel(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = job.stage_job("owner/kokoro-ref", Path(tmp) / "k")
            metadata = json.loads((staged / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_gpu"])
            self.assertNotIn("machine_shape", metadata)
            script = (staged / "convert.py").read_text()
            compile(script, "convert.py", "exec")
            self.assertIn(f"KOKORO_VERSION = {job.KOKORO_VERSION!r}", script)
            self.assertIn("FIXTURES = ", script)

    def test_asr_job_needs_a_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                job.stage_job("owner/asr", Path(tmp) / "a", mode="asr")
            with self.assertRaises(ValueError):
                job.stage_job("owner/asr", Path(tmp) / "b", mode="asr", dataset="bad slug")
            staged = job.stage_job("owner/asr", Path(tmp) / "c", mode="asr",
                                   dataset="owner/speech-wavs")
            metadata = json.loads((staged / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["dataset_sources"], ["owner/speech-wavs"])
            self.assertFalse(metadata["enable_gpu"])
            compile((staged / "convert.py").read_text(), "convert.py", "exec")

    def test_fixtures_are_english_voices_within_kokoro_limits(self):
        for text, voice, speed in job.FIXTURES:
            self.assertIn(voice, job.ENGLISH_VOICES)
            self.assertTrue(0.25 <= speed <= 4.0)
            self.assertTrue(0 < len(text) < 400)
        self.assertEqual(len(job.ENGLISH_VOICES), len(set(job.ENGLISH_VOICES)))

    def test_unknown_mode_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            job.stage_job("owner/x", Path(tmp) / "x", mode="train")


if __name__ == "__main__":
    unittest.main()
