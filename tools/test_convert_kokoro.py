"""Offline checks for the Kokoro-82M converter (no torch, no model files)."""

import array
import json
import math
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import convert_kokoro as ck


def tensor(values, shape):
    return ck.Tensor(array.array("f", values), shape)


def torch_zip(path: Path, shape, values):
    """A minimal `torch.save` archive holding {"w": tensor} (hand-written pickle)."""
    def text(s):
        b = s.encode()
        return b"X" + struct.pack("<I", len(b)) + b

    def ints(seq):
        return b"(" + b"".join(b"K" + bytes([n]) for n in seq) + b"t"

    stride, step = [], 1
    for dim in reversed(shape):
        stride.insert(0, step)
        step *= dim
    pkl = (
        b"\x80\x02}" + text("w")
        + b"ctorch._utils\n_rebuild_tensor_v2\n("
        + b"(" + text("storage") + b"ctorch\nFloatStorage\n" + text("0") + text("cpu")
        + b"K" + bytes([len(values)]) + b"tQ"
        + b"K\x00" + ints(shape) + ints(stride) + b"\x89"
        + b"ccollections\nOrderedDict\n)R" + b"tR" + b"s."
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pkl)
        archive.writestr("archive/data/0", array.array("f", values).tobytes())


class LayoutTests(unittest.TestCase):
    def test_conv_layout_moves_taps_before_inputs(self):
        # [out=1, in=2, k=3]
        t = ck.conv_layout(tensor([1, 2, 3, 4, 5, 6], (1, 2, 3)))
        self.assertEqual(t.shape, (1, 3, 2))
        self.assertEqual(list(t.values), [1, 4, 2, 5, 3, 6])

    def test_transposed_layout_splits_output_phases(self):
        # ConvTranspose1d [in=1, out=1, K=4], stride 2: phase r holds taps r, r + 2.
        t = ck.transposed_layout(tensor([10, 11, 12, 13], (1, 1, 4)), 2)
        self.assertEqual(t.shape, (2, 1, 2, 1))
        self.assertEqual(list(t.values), [10, 12, 11, 13])

    def test_weight_norm_folds_over_all_but_the_first_dimension(self):
        state = {
            "c.weight_g": tensor([2.0, 3.0], (2, 1, 1)),
            "c.weight_v": tensor([3.0, 4.0, 0.0, 5.0], (2, 1, 2)),
        }
        folded = ck.fold_weight_norm(state)
        self.assertEqual(set(folded), {"c.weight"})
        self.assertTrue(all(math.isclose(a, b, rel_tol=1e-6) for a, b in
                            zip(folded["c.weight"].values, [1.2, 1.6, 0.0, 3.0])))

    def test_lstm_layout_stacks_directions_and_sums_biases(self):
        h = ck.LSTM_HIDDEN
        state = {}
        for suffix, base in (("", 0.0), ("_reverse", 1.0)):
            state[f"l.weight_ih_l0{suffix}"] = tensor([base] * (4 * h * 3), (4 * h, 3))
            state[f"l.weight_hh_l0{suffix}"] = tensor(
                [float(r * h + j) for r in range(4 * h) for j in range(h)], (4 * h, h))
            state[f"l.bias_ih_l0{suffix}"] = tensor([1.0] * (4 * h), (4 * h,))
            state[f"l.bias_hh_l0{suffix}"] = tensor([base + 0.5] * (4 * h), (4 * h,))
        out = ck.lstm_layout(state, "l")
        self.assertEqual(out["l.ih.weight"].shape, (8 * h, 3))
        self.assertEqual(out["l.ih.bias"].values[0], 1.5)
        self.assertEqual(out["l.ih.bias"].values[4 * h], 2.5)
        whh = out["l.whh_t"]
        self.assertEqual(whh.shape, (2, h, 4 * h))
        # whh_t[0][j][r] = W_hh[r][j]
        self.assertEqual(whh.values[1 * 4 * h + 2], float(2 * h + 1))
        self.assertEqual(state, {})


class LoaderTests(unittest.TestCase):
    def test_restricted_loader_reads_a_torch_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.pt"
            torch_zip(path, (2, 2), [1.0, 2.0, 3.0, 4.0])
            loaded = ck.load_torch(path)
            self.assertEqual(loaded["w"].shape, (2, 2))
            self.assertEqual(list(loaded["w"].values), [1.0, 2.0, 3.0, 4.0])

    def test_loader_refuses_arbitrary_globals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evil.pt"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/data.pkl", b"\x80\x02cos\nsystem\n.")
            with self.assertRaises(__import__("pickle").UnpicklingError):
                ck.load_torch(path)

    def test_sources_are_checked_against_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a").write_bytes(b"abc")
            ck.check_sources(Path(tmp), {"a": (3, ck.sha256_file(Path(tmp) / "a"))})
            with self.assertRaises(SystemExit):
                ck.check_sources(Path(tmp), {"a": (3, "0" * 64)})
            with self.assertRaises(SystemExit):
                ck.check_sources(Path(tmp), {"missing": (1, "0" * 64)})

    def test_pins_cover_the_model_voices_and_lexicons(self):
        names = set(ck.SOURCE_FILES)
        self.assertIn("model/kokoro-v1_0.pth", names)
        self.assertIn("model/config.json", names)
        for voice in ck.VOICES:
            self.assertIn(f"model/voices/{voice}.pt", names)
        for lexicon in ck.LEXICONS:
            self.assertIn(f"src/misaki/data/{lexicon}.json", names)
        self.assertEqual(len(ck.OUTPUT_SHA256), 64)


class ArtifactTests(unittest.TestCase):
    def test_header_layout_is_aligned_and_hashed(self):
        state = {"g": {"x.weight": tensor([1.0, 2.0], (2,))}}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            (src / "model" / "voices").mkdir(parents=True)
            (src / "src" / "misaki" / "data").mkdir(parents=True)
            (src / "model" / "config.json").write_text(json.dumps({"vocab": {"a": 1}}))
            for name in ck.LEXICONS:
                (src / "src" / "misaki" / "data" / f"{name}.json").write_text("{}")
            original = ck.load_torch
            voice = ck.Tensor(array.array("f", [0.5] * (510 * 256)), (510, 1, 256))
            ck.load_torch = lambda path: state if path.name.endswith(".pth") else voice
            try:
                target = ck.build(src, Path(tmp) / "out", check=False)
            finally:
                ck.load_torch = original
            data = target.read_bytes()
            self.assertEqual(data[:8], ck.MAGIC)
            length = struct.unpack("<Q", data[8:16])[0]
            self.assertEqual((16 + length) % ck.HEADER_ALIGN, 0)
            header = json.loads(data[16:16 + length])
            payload = data[16 + length:]
            self.assertEqual(len(payload), header["payload_length"])
            import hashlib
            self.assertEqual(hashlib.sha256(payload).hexdigest(), header["payload_sha256"])
            self.assertEqual(header["tensors"]["g.x.weight"]["shape"], [2])
            self.assertEqual(set(header["voices"]), set(ck.VOICES))
            self.assertTrue(all(v % ck.TENSOR_ALIGN == 0 for v in header["voices"].values()))
            self.assertEqual(header["blobs"]["us_gold"]["length"], 2)


if __name__ == "__main__":
    unittest.main()
