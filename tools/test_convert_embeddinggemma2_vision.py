import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import convert_embeddinggemma2_vision as conv
from test_convert_embeddinggemma2 import bf16, write_safetensors


def pins_for(directory: Path) -> dict:
    return {name: ((directory / name).stat().st_size,
                   hashlib.sha256((directory / name).read_bytes()).hexdigest())
            for name in ("model.safetensors", "config.json")}


def tiny_checkpoint(source: Path) -> dict:
    source.mkdir()
    tensors = {}
    for _, src, kind in conv.plan(num_layers=1):
        if src.endswith("position_embedding_table"):
            shape = (2, 3, 64)
        elif kind == "bf16":
            shape = (4, 64)
        else:
            shape = (64,)
        count = 1
        for dim in shape:
            count *= dim
        tensors[src] = (shape, bf16([(i % 5 - 2) / 8 for i in range(count)]))
    write_safetensors(source / "model.safetensors", tensors)
    (source / "config.json").write_text("{}")
    return tensors


class VisionConversionTests(unittest.TestCase):
    def test_plan_covers_every_vision_tensor_once(self):
        plan = conv.plan()
        names = [runtime for runtime, _, _ in plan]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(plan), 2 + 16 * 13 + 1)
        self.assertIn(("embed_vision", "embed_vision.embedding_projection.weight", "bf16"), plan)
        self.assertIn(("layers.15.self_attn.k_norm",
                       "vision_tower.encoder.layers.15.self_attn.k_norm.weight", "f32"), plan)

    def test_tiny_checkpoint_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp, "src"), Path(tmp, "dst")
            tensors = tiny_checkpoint(source)
            manifest = conv.convert(source, dest, pins=pins_for(source), expected=(None, None),
                                    num_layers=1)
            raw = (dest / conv.ARTIFACT_NAME).read_bytes()
            self.assertEqual(raw[:8], conv.MAGIC)
            (length,) = struct.unpack("<Q", raw[8:16])
            header = json.loads(raw[16:16 + length])
            payload = raw[16 + length:]
            self.assertEqual(header["format"], conv.FORMAT)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), header["payload_sha256"])
            self.assertEqual(manifest["files"][conv.ARTIFACT_NAME]["sha256"],
                             hashlib.sha256(raw).hexdigest())
            table = header["tensors"]["patch_embedder.position_embedding_table"]
            self.assertEqual((table["dtype"], table["shape"]), ("bf16", [6, 64]))
            src = tensors["vision_tower.patch_embedder.position_embedding_table"][1]
            self.assertEqual(payload[table["offset"]:table["offset"] + table["length"]], src)
            norm = header["tensors"]["layers.0.self_attn.q_norm"]
            self.assertEqual((norm["dtype"], norm["shape"]), ("f32", [64]))
            first = struct.unpack("<f", payload[norm["offset"]:norm["offset"] + 4])[0]
            self.assertEqual(first, -0.25)
            self.assertTrue((dest / conv.MANIFEST_NAME).is_file())
            with self.assertRaises(SystemExit):  # no clobbering
                conv.convert(source, dest, pins=pins_for(source), expected=(None, None),
                             num_layers=1)

    def test_a_changed_conversion_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp, "src"), Path(tmp, "dst")
            tiny_checkpoint(source)
            with self.assertRaises(SystemExit):
                conv.convert(source, dest, pins=pins_for(source), expected=(1, "0" * 64),
                             num_layers=1)
            self.assertFalse((dest / conv.ARTIFACT_NAME).exists())


if __name__ == "__main__":
    unittest.main()
