import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import convert_embeddinggemma2_audio as conv
from test_convert_embeddinggemma2 import bf16, write_safetensors


def pins_for(directory: Path) -> dict:
    return {name: ((directory / name).stat().st_size,
                   hashlib.sha256((directory / name).read_bytes()).hexdigest())
            for name in ("model.safetensors", "config.json")}


def tiny_checkpoint(source: Path) -> dict:
    """One-layer checkpoint with every planned tensor (matrices 4x64)."""
    tensors = {}
    for _, src, kind in conv.plan(num_layers=1):
        if kind == "clip":
            for name, value in zip(src, (-2.0, 2.0, -8.0, 8.0)):
                tensors[name] = ((), bf16([value]))
        else:
            shape = (4, 64) if kind == "bf16" else (64,)
            count = shape[0] * (shape[1] if len(shape) > 1 else 1)
            tensors[src] = (shape, bf16([(i % 5 - 2) / 4 for i in range(count)]))
    write_safetensors(source / "model.safetensors", tensors)
    (source / "config.json").write_text("{}")
    return tensors


class AudioConverterTests(unittest.TestCase):
    def test_plan_covers_the_audio_tower_once(self):
        plan = conv.plan()
        names = [runtime for runtime, _, _ in plan]
        self.assertEqual(len(names), len(set(names)))
        # 10 clipped linears + relative_k_proj per layer, stem projection,
        # output_proj and embed_audio.
        self.assertEqual(sum(kind == "bf16" for _, _, kind in plan), 12 * 11 + 3)
        self.assertEqual(sum(kind == "clip" for _, _, kind in plan), 12 * 10)
        self.assertIn(("embed_audio", "embed_audio.embedding_projection.weight", "bf16"), plan)
        self.assertIn(("layers.11.lconv1d.depthwise_conv1d",
                       "audio_tower.layers.11.lconv1d.depthwise_conv1d.weight", "f32"), plan)

    def test_tiny_checkpoint_round_trips_with_clip_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp, "src"), Path(tmp, "dst")
            source.mkdir()
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
            clip = header["tensors"]["layers.0.self_attn.q_proj.clip"]
            self.assertEqual((clip["dtype"], clip["shape"]), ("f32", [4]))
            bounds = struct.unpack("<4f", payload[clip["offset"]:clip["offset"] + 16])
            self.assertEqual(bounds, (-2.0, 2.0, -8.0, 8.0))
            q = header["tensors"]["layers.0.self_attn.q_proj"]
            self.assertEqual(payload[q["offset"]:q["offset"] + q["length"]],
                             tensors["audio_tower.layers.0.self_attn.q_proj.linear.weight"][1])
            with self.assertRaises(SystemExit):  # no clobbering
                conv.convert(source, dest, pins=pins_for(source), expected=(None, None),
                             num_layers=1)

    def test_a_changed_conversion_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp, "src"), Path(tmp, "dst")
            source.mkdir()
            tiny_checkpoint(source)
            with self.assertRaises(SystemExit):
                conv.convert(source, dest, pins=pins_for(source), expected=(1, "0" * 64),
                             num_layers=1)
            self.assertFalse((dest / conv.ARTIFACT_NAME).exists())


if __name__ == "__main__":
    unittest.main()
