import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import convert_embeddinggemma2 as conv


def write_safetensors(path: Path, tensors: dict) -> None:
    """tensors: name -> (shape, raw BF16 bytes)."""
    header, offset, blobs = {}, 0, []
    for name, (shape, raw) in tensors.items():
        header[name] = {"dtype": "BF16", "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def bf16(values) -> bytes:
    out = bytearray()
    for v in values:
        out += struct.pack("<I", struct.unpack("<I", struct.pack("<f", v))[0])[2:]
    return bytes(out)


def pins_for(directory: Path) -> dict:
    return {name: ((directory / name).stat().st_size,
                   hashlib.sha256((directory / name).read_bytes()).hexdigest())
            for name in ("model.safetensors", "tokenizer.json", "config.json")}


class FormatTests(unittest.TestCase):
    def test_plan_covers_every_text_tensor_once(self):
        plan = conv.plan()
        names = [runtime for runtime, _, _ in plan]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(kind == "bf16" for _, _, kind in plan), 2 + 24 * 9)
        self.assertIn(("embedding_projection", "embedding_projection.weight", "f32"), plan)
        self.assertIn(("layers.23.layer_scalar", "layers.23.layer_scalar", "f32"), plan)

    def test_header_pads_the_payload_to_a_page(self):
        head = conv.build_header({}, 0, "0" * 64, {})
        self.assertEqual((16 + len(head)) % conv.HEADER_ALIGN, 0)
        self.assertEqual(json.loads(head)["format"], conv.FORMAT)

    def test_sources_are_size_and_digest_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").write_bytes(b"abc")
            good = {"a": (3, hashlib.sha256(b"abc").hexdigest())}
            self.assertEqual(conv.verify_sources(root, good)["a"]["size"], 3)
            with self.assertRaises(SystemExit):
                conv.verify_sources(root, {"a": (3, "0" * 64)})
            with self.assertRaises(SystemExit):
                conv.verify_sources(root, {"a": (4, good["a"][1])})
            with self.assertRaises(SystemExit):
                conv.verify_sources(root, {"missing": good["a"]})


class ConversionTests(unittest.TestCase):
    def test_bf16_widens_exactly(self):
        raw = bf16([1.0, -2.5, 0.0, 3.140625])
        widened = struct.unpack("<4f", conv.bf16_to_f32(raw))
        self.assertEqual(widened, (1.0, -2.5, 0.0, 3.140625))

    def test_tiny_checkpoint_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp, "src"), Path(tmp, "dst")
            source.mkdir()
            tensors = {}
            for _, src, kind in conv.plan(num_layers=1):
                shape = (4, 64) if kind == "bf16" else ((1,) if src.endswith("scalar") else (64,))
                count = shape[0] * (shape[1] if len(shape) > 1 else 1)
                tensors[conv.PREFIX + src] = (shape, bf16([(i % 7 - 3) / 4 for i in range(count)]))
            write_safetensors(source / "model.safetensors", tensors)
            (source / "tokenizer.json").write_text("{}")
            (source / "config.json").write_text("{}")
            manifest = conv.convert(source, dest, pins=pins_for(source), expected=(None, None),
                                    num_layers=1)
            raw = (dest / conv.ARTIFACT_NAME).read_bytes()
            self.assertEqual(raw[:8], conv.MAGIC)
            (length,) = struct.unpack("<Q", raw[8:16])
            self.assertEqual((16 + length) % conv.HEADER_ALIGN, 0)
            header = json.loads(raw[16:16 + length])
            payload = raw[16 + length:]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), header["payload_sha256"])
            self.assertEqual(manifest["files"][conv.ARTIFACT_NAME]["sha256"],
                             hashlib.sha256(raw).hexdigest())
            info = header["tensors"]["layers.0.mlp.up_proj"]
            self.assertEqual((info["dtype"], info["shape"]), ("bf16", [4, 64]))
            self.assertEqual(info["offset"] % conv.TENSOR_ALIGN, 0)
            start = info["offset"]
            self.assertEqual(payload[start:start + info["length"]],
                             tensors[conv.PREFIX + "layers.0.mlp.up_proj.weight"][1])
            norm = header["tensors"]["norm"]
            self.assertEqual((norm["dtype"], norm["shape"]), ("f32", [64]))
            first = struct.unpack("<f", payload[norm["offset"]:norm["offset"] + 4])[0]
            self.assertEqual(first, -0.75)
            self.assertEqual((dest / "tokenizer.json").read_text(), "{}")
            with self.assertRaises(SystemExit):  # no clobbering
                conv.convert(source, dest, pins=pins_for(source), expected=(None, None),
                             num_layers=1)

    def test_a_changed_conversion_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp, "src"), Path(tmp, "dst")
            source.mkdir()
            tensors = {conv.PREFIX + src: ((4, 64) if kind == "bf16" else (64,),
                                           bf16([0.5] * (256 if kind == "bf16" else 64)))
                       for _, src, kind in conv.plan(num_layers=1)}
            write_safetensors(source / "model.safetensors", tensors)
            (source / "tokenizer.json").write_text("{}")
            (source / "config.json").write_text("{}")
            with self.assertRaises(SystemExit):
                conv.convert(source, dest, pins=pins_for(source), expected=(1, "0" * 64),
                             num_layers=1)
            self.assertFalse((dest / conv.ARTIFACT_NAME).exists())


if __name__ == "__main__":
    unittest.main()
