"""Convert the Kaggle EmbeddingGemma 2 checkpoint's vision encoder into the runtime's vision artifact.

The vision artifact is an optional add-on to the text install written by
tools/convert_embeddinggemma2.py: image and video soft tokens run through the
same text encoder, so the text artifact must be installed too. With the pinned
Kaggle checkpoint downloaded (see convert_embeddinggemma2.py):

  python3 tools/convert_embeddinggemma2_vision.py --source cache/embeddinggemma-2

writes models/embeddinggemma2-text/{embeddinggemma2-vision-bf16.bin,
manifest-vision.json} (standard library only).

Converted: the `gemma4_vision` encoder (`vision_tower.*`: the 768->768 patch
input projection, the learned 2 x 10,240 x 768 position table, 16 encoder
layers) and `embed_vision` (768->512). Every source file the conversion reads
is size- and SHA256-checked, and the written artifact must hash to
OUTPUT_SHA256, so a changed checkpoint or conversion fails instead of
installing silently.

Precision (docs/EMBEDDINGS.md): matrices and the position table keep the
checkpoint's own BF16 values (copied byte for byte; the table is stored as a
[20480, 768] matrix, x rows then y rows); norms are widened to F32 (exact).

Layout: as the text artifact (convert_embeddinggemma2.py) with magic
b"EG2VIS\\x00\\x01" and format "embeddinggemma2-vision".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import convert_embeddinggemma2 as text

MAGIC = b"EG2VIS\x00\x01"
FORMAT = "embeddinggemma2-vision"
VERSION = 1
SOURCE_FILES = {name: text.SOURCE_FILES[name] for name in ("model.safetensors", "config.json")}
ARTIFACT_NAME = "embeddinggemma2-vision-bf16.bin"
MANIFEST_NAME = "manifest-vision.json"
OUTPUT_SIZE = 335_642_624
OUTPUT_SHA256 = "c3f928d2a65aad2173f7ed6d8acac6e58d0a7336c9a03f0eb022f4a186909968"
DEFAULT_DESTINATION = text.DEFAULT_DESTINATION

CONFIG = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_layers": 16,
    "num_attention_heads": 12,
    "head_dim": 64,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 10240,
    "rope_theta": 100.0,
    "rms_norm_eps": 1e-6,
    "text_hidden_size": 512,
    "image_soft_tokens": 280,
    "video_soft_tokens": 140,
    "video_fps": 1,
    "video_max_frames": 32,
    "image_token_id": 258880,
    "video_token_id": 258884,
    "boi_token_id": 255999,
    "eoi_token_id": 258882,
}

LAYER_LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
LAYER_F32 = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)


def plan(num_layers: int = CONFIG["num_layers"]):
    """(runtime name, source tensor name, kind) for every artifact tensor; kind is
    "bf16" (matrix, copied; higher-rank tensors are flattened to [rows, last dim])
    or "f32" (widened, flattened)."""
    out = [
        ("patch_embedder.input_proj", "vision_tower.patch_embedder.input_proj.weight", "bf16"),
        ("patch_embedder.position_embedding_table",
         "vision_tower.patch_embedder.position_embedding_table", "bf16"),
    ]
    for i in range(num_layers):
        p = f"vision_tower.encoder.layers.{i}."
        for name in LAYER_LINEARS:
            out.append((f"layers.{i}.{name}", p + name + ".linear.weight", "bf16"))
        for name in LAYER_F32:
            out.append((f"layers.{i}.{name}", p + name + ".weight", "f32"))
    out.append(("embed_vision", "embed_vision.embedding_projection.weight", "bf16"))
    return out


def convert(source: Path, destination: Path, pins=SOURCE_FILES,
            expected=(OUTPUT_SIZE, OUTPUT_SHA256), num_layers=CONFIG["num_layers"]) -> dict:
    """Write `destination/ARTIFACT_NAME` and `MANIFEST_NAME` (no clobbering)."""
    pinned = text.verify_sources(source, pins)
    model = source / "model.safetensors"
    header, data_start = text.read_safetensors_header(model)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / ARTIFACT_NAME
    manifest_path = destination / MANIFEST_NAME
    for path in (artifact, manifest_path):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"{path}: exists; remove the old install first")
    partial = artifact.with_name(artifact.name + ".partial")
    payload = destination / (ARTIFACT_NAME + ".payload.partial")
    tensors, offset = {}, 0
    digest = hashlib.sha256()
    with model.open("rb") as handle, payload.open("xb") as out:

        def emit(name, raw, dtype, shape):
            nonlocal offset
            pad = text.align(offset, text.TENSOR_ALIGN) - offset
            if pad:
                out.write(b"\0" * pad)
                digest.update(b"\0" * pad)
                offset += pad
            out.write(raw)
            digest.update(raw)
            tensors[name] = {"dtype": dtype, "shape": shape, "offset": offset, "length": len(raw)}
            offset += len(raw)

        for runtime, src, kind in plan(num_layers):
            info = header.get(src)
            if info is None or info["dtype"] != "BF16":
                raise SystemExit(f"source tensor {src} missing or not BF16")
            begin, end = info["data_offsets"]
            shape = list(info["shape"])
            count = 1
            for dim in shape:
                count *= dim
            if end - begin != 2 * count:
                raise SystemExit(f"source tensor {src} has inconsistent bounds")
            handle.seek(data_start + begin)
            raw = handle.read(end - begin)
            if kind == "bf16":
                if len(shape) < 2:
                    raise SystemExit(f"source tensor {src} is not a matrix")
                emit(runtime, raw, "bf16", [count // shape[-1], shape[-1]])
            else:
                emit(runtime, text.bf16_to_f32(raw), "f32", [count])
    payload_sha256 = digest.hexdigest()
    head = json.dumps({"format": FORMAT, "version": VERSION, "config": CONFIG, "source": pinned,
                       "tensors": tensors, "payload_length": offset, "payload_sha256": payload_sha256},
                      sort_keys=True, separators=(",", ":")).encode()
    head += b" " * (text.align(16 + len(head), text.HEADER_ALIGN) - 16 - len(head))
    try:
        with partial.open("xb") as out, payload.open("rb") as body:
            out.write(MAGIC)
            out.write(struct.pack("<Q", len(head)))
            out.write(head)
            while chunk := body.read(1 << 22):
                out.write(chunk)
        payload.unlink()
        size, sha256 = partial.stat().st_size, text.sha256_file(partial)
        want_size, want_sha = expected
        if want_sha is not None and (size, sha256) != (want_size, want_sha):
            raise SystemExit(f"artifact is {size} bytes / {sha256}, expected "
                             f"{want_size} / {want_sha}: the conversion changed")
        os.link(partial, artifact)
        partial.unlink()
    finally:
        payload.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    manifest = {
        "model": "google/embeddinggemma-2 (Kaggle google/embeddinggemma-2/transformers/"
                 "embeddinggemma-2/1), vision encoder + embed_vision, BF16 matrices, F32 vectors",
        "files": {ARTIFACT_NAME: {"size": size, "sha256": sha256}},
        "source": pinned,
        "payload_sha256": payload_sha256,
        "converter": "tools/convert_embeddinggemma2_vision.py",
    }
    with manifest_path.open("x") as out:
        json.dump(manifest, out, indent=2, sort_keys=True)
        out.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="directory of the downloaded Kaggle model files")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION,
                        help=f"install directory (default: {DEFAULT_DESTINATION})")
    args = parser.parse_args()
    manifest = convert(args.source, args.destination)
    for name, record in manifest["files"].items():
        print(args.destination / name, record["size"], record["sha256"])


if __name__ == "__main__":
    main()
