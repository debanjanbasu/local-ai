import hashlib
import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import convert_whisper as conv


def f16(values) -> bytes:
    return struct.pack(f"<{len(values)}e", *values)


def write_safetensors(path: Path, tensors: dict) -> None:
    """tensors: name -> (shape, raw F16 bytes)."""
    header, offset, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for name, (shape, raw) in tensors.items():
        header[name] = {"dtype": "F16", "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def npy(shape, values) -> bytes:
    meta = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape[0]}, {shape[1]}), }}"
    meta = meta.ljust(118) + "\n"
    return (b"\x93NUMPY\x01\x00" + struct.pack("<H", len(meta)) + meta.encode()
            + struct.pack(f"<{len(values)}f", *values))


def tiny_source(root: Path, extra_tensor: bool = False) -> dict:
    """A one-layer checkpoint with every file the converter reads; returns
    the tensors written."""
    (root / "model").mkdir(parents=True)
    tensors = {}
    for name, kind in conv.plan(1, 1):
        if kind == "conv":
            shape = (4, 2, 3)
        elif kind == "f16":
            shape = (4, 8)
        elif name.endswith("embed_positions.weight"):
            shape = (3, 4)
        else:
            shape = (4,)
        count = 1
        for dim in shape:
            count *= dim
        tensors[conv.PREFIX + name] = (shape, f16([(i % 9 - 4) / 8 for i in range(count)]))
    if extra_tensor:
        tensors["proj_out.weight"] = ((4, 8), f16([0.0] * 32))
    write_safetensors(root / "model/model.safetensors", tensors)
    (root / "model/config.json").write_text(json.dumps({"d_model": 4}))
    languages = {f"<|{code}|>": 50259 + i for i, code in enumerate(["en", "zh"])}
    (root / "model/generation_config.json").write_text(json.dumps({
        "decoder_start_token_id": 50258, "eos_token_id": 50257, "prev_sot_token_id": 50362,
        "no_timestamps_token_id": 50364, "task_to_id": {"transcribe": 50360, "translate": 50359},
        "lang_to_id": languages, "suppress_tokens": [1, 2], "begin_suppress_tokens": [220, 50257],
        "alignment_heads": [[7, 0], [10, 17]], "max_initial_timestamp_index": 50,
    }))
    added = [{"id": 50257, "content": "<|endoftext|>"},
             {"id": 50258, "content": "<|startoftranscript|>"},
             {"id": 50361, "content": "<|startoflm|>"}, {"id": 50363, "content": "<|nospeech|>"},
             {"id": 50365, "content": "<|0.00|>"}]
    (root / "model/tokenizer.json").write_text(json.dumps({"added_tokens": added}))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mel_128.npy", npy((128, 201), [0.25] * (128 * 201)))
        archive.writestr("mel_80.npy", npy((80, 201), [0.5] * (80 * 201)))
    (root / "mel_filters.npz").write_bytes(buffer.getvalue())
    return tensors


def pins_for(root: Path) -> dict:
    return {name: ((root / name).stat().st_size,
                   hashlib.sha256((root / name).read_bytes()).hexdigest())
            for name in conv.SOURCE_FILES}


class FormatTests(unittest.TestCase):
    def test_plan_covers_every_tensor_once(self):
        plan = conv.plan()
        names = [name for name, _ in plan]
        self.assertEqual(len(names), len(set(names)))
        # 32 encoder layers x 15 tensors, 32 decoder layers x 24, 11 others.
        self.assertEqual(len(plan), 32 * 15 + 32 * 24 + 11)
        self.assertIn(("decoder.layers.31.encoder_attn.k_proj.weight", "f16"), plan)
        self.assertNotIn(("decoder.layers.31.encoder_attn.k_proj.bias", "f32"), plan)

    def test_header_pads_the_payload_to_a_page(self):
        head = conv.build_header({}, {}, 0, "0" * 64, {})
        self.assertEqual((16 + len(head)) % conv.HEADER_ALIGN, 0)
        self.assertEqual(json.loads(head)["format"], conv.FORMAT)

    def test_f16_widens_exactly(self):
        self.assertEqual(struct.unpack("<3f", conv.f16_to_f32(f16([1.0, -2.5, 0.000061035]))),
                         (1.0, -2.5, struct.unpack("<e", f16([0.000061035]))[0]))

    def test_sources_are_size_and_digest_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").write_bytes(b"abc")
            good = {"a": (3, hashlib.sha256(b"abc").hexdigest())}
            self.assertEqual(conv.verify_sources(root, good)["a"]["size"], 3)
            for bad in ({"a": (3, "0" * 64)}, {"a": (4, good["a"][1])}, {"b": good["a"]}):
                with self.assertRaises(SystemExit):
                    conv.verify_sources(root, bad)

    def test_the_real_config_is_checked(self):
        conv.check_config(dict(conv.HF_CONFIG))
        with self.assertRaises(SystemExit):
            conv.check_config({**conv.HF_CONFIG, "decoder_layers": 4})


class ConversionTests(unittest.TestCase):
    def test_tiny_checkpoint_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp, "src"), Path(tmp, "dst")
            tensors = tiny_source(source)
            manifest = conv.convert(source, dest, pins=pins_for(source), expected=(None, None),
                                    layers=(1, 1), check_hf_config=False)
            raw = (dest / conv.ARTIFACT_NAME).read_bytes()
            self.assertEqual(raw[:8], conv.MAGIC)
            (length,) = struct.unpack("<Q", raw[8:16])
            self.assertEqual((16 + length) % conv.HEADER_ALIGN, 0)
            header = json.loads(raw[16:16 + length])
            payload = raw[16 + length:]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), header["payload_sha256"])
            self.assertEqual(manifest["files"][conv.ARTIFACT_NAME]["sha256"],
                             hashlib.sha256(raw).hexdigest())
            conv1 = header["tensors"]["encoder.conv1.weight"]
            self.assertEqual((conv1["dtype"], conv1["shape"]), ("f16", [4, 6]))
            self.assertEqual(payload[conv1["offset"]:conv1["offset"] + conv1["length"]],
                             tensors["model.encoder.conv1.weight"][1])
            fc1 = header["tensors"]["decoder.layers.0.fc1.weight"]
            self.assertEqual(fc1["offset"] % conv.TENSOR_ALIGN, 0)
            bias = header["tensors"]["decoder.layers.0.fc1.bias"]
            self.assertEqual((bias["dtype"], bias["shape"]), ("f32", [4]))
            self.assertEqual(struct.unpack("<f", payload[bias["offset"]:bias["offset"] + 4])[0],
                             -0.5)
            mel = header["tensors"]["mel_filters"]
            self.assertEqual((mel["dtype"], mel["shape"]), ("f32", [128, 201]))
            tokens = header["tokens"]
            self.assertEqual((tokens["sot"], tokens["no_speech"], tokens["timestamp_begin"]),
                             (50258, 50363, 50365))
            self.assertEqual(tokens["languages"], {"en": 50259, "zh": 50260})
            self.assertEqual((dest / "tokenizer.json").read_bytes(),
                             (source / "model/tokenizer.json").read_bytes())
            with self.assertRaises(SystemExit):  # no clobbering
                conv.convert(source, dest, pins=pins_for(source), expected=(None, None),
                             layers=(1, 1), check_hf_config=False)

    def test_unread_checkpoint_tensors_and_changed_output_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp, "src"), Path(tmp, "dst")
            tiny_source(source, extra_tensor=True)
            with self.assertRaises(SystemExit):
                conv.convert(source, dest, pins=pins_for(source), expected=(None, None),
                             layers=(1, 1), check_hf_config=False)
            self.assertFalse((dest / conv.ARTIFACT_NAME).exists())
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp, "src"), Path(tmp, "dst")
            tiny_source(source)
            with self.assertRaises(SystemExit):
                conv.convert(source, dest, pins=pins_for(source), expected=(1, "0" * 64),
                             layers=(1, 1), check_hf_config=False)
            self.assertFalse((dest / conv.ARTIFACT_NAME).exists())


if __name__ == "__main__":
    unittest.main()
