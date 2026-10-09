"""Capture windows cover full documents without discarding their tails."""

import importlib.util
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("numpy", "torch"))


def write_shard(path, count, base=0, hdim=4, ktop=2):
    import numpy as np

    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", 0x4D545044, count, hdim, ktop))
        for p in range(count):
            f.write(struct.pack("<i", base + p))
            f.write(np.zeros(ktop, np.int32).tobytes() + np.zeros(ktop, np.float32).tobytes())
            f.write(np.zeros(hdim, np.float16).tobytes())


@unittest.skipUnless(HAVE_TORCH, "numpy/torch not installed")
class Tests(unittest.TestCase):
    def test_window_starts_cover_tail(self):
        from data import window_starts

        for count in (64, 255, 256, 257, 450, 512, 513, 1000):
            covered = set()
            for start in window_starts(count, 256):
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(start + min(256, count), count)
                covered.update(range(start, min(start + 256, count)))
            self.assertEqual(covered, set(range(count)), count)

    def test_batches_cover_every_position_of_long_documents(self):
        import numpy as np
        from data import DocBatcher

        counts = [70, 100, 200, 256, 300, 450, 700, 90, 130, 180]
        with tempfile.TemporaryDirectory() as tmp:
            for doc, count in enumerate(counts):
                write_shard(os.path.join(tmp, f"shard-{doc:06d}.bin"), count, base=doc * 10000)
            batcher = DocBatcher(tmp, window=256, batch=2, seed=1, val_fraction=0.0)
            seen = set()
            for tokens, _, _, _, positions in batcher.train_batches():
                self.assertLessEqual(tokens.shape[0], 2)
                self.assertTrue(np.array_equal(tokens.numpy() % 10000, positions.numpy()))
                seen.update(tokens.numpy().ravel().tolist())
            docs = {token // 10000 for token in seen}
            held_out = {int(os.path.basename(p)[6:12]) for p in batcher.val_paths}
            self.assertEqual(docs | held_out, set(range(len(counts))))
            for doc, count in enumerate(counts):
                if count >= 256 and doc not in held_out:
                    self.assertTrue({doc * 10000 + p for p in range(count)} <= seen, doc)

if __name__ == "__main__":
    unittest.main()
