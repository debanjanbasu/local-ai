"""Convert the Kaggle EmbeddingGemma 2 checkpoint into the runtime's text artifact.

Download the pinned checkpoint from Kaggle Models first (the Gemma terms must be
accepted on Kaggle for the account):

  kaggle models instances versions download \\
    google/embeddinggemma-2/transformers/embeddinggemma-2/1 \\
    -p cache/embeddinggemma-2 --untar

then convert (standard library only):

  python3 tools/convert_embeddinggemma2.py --source cache/embeddinggemma-2

Only the text path is converted: the token table, the 24 encoder layers, the
per-layer-embedding (PLE) projection and the 512->768 embedding projection.
Every source file the conversion reads is size- and SHA256-checked against the
pins below, and the written artifact must hash to OUTPUT_SHA256, so a changed
checkpoint or a changed conversion fails instead of installing silently.

Precision (docs/EMBEDDINGS.md): every matrix of the text path, the token table
included, keeps the checkpoint's own BF16 values (copied byte for byte); norms,
layer scalars and the 768x512 embedding projection are widened to F32 (exact).
Per-row int8 was measured too: it is lossless for retrieval, but its worst
input in the 1,000-text parity set reached only cosine 0.9976 to the FP32
reference, below the 0.999 gate, so the artifact stays BF16.

Artifact layout (little endian):

  0   8 bytes  magic b"EG2TXT\\x00\\x01"
  8   u64      header length H (JSON, space padded so the payload starts at a
               multiple of 4096)
  16  H bytes  header: {"format", "version", "config", "source",
               "tensors": {name: {"dtype": "i8"|"f32", "shape", "offset",
               "length"}}, "payload_length", "payload_sha256"}
  16+H         payload; tensor offsets are relative to it, 256-byte aligned;
               dtype "bf16" (matrices, [rows, cols]) or "f32" (vectors and
               the embedding projection, flattened)
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

MAGIC = b"EG2TXT\x00\x01"
FORMAT = "embeddinggemma2-text"
VERSION = 1
HEADER_ALIGN = 4096
TENSOR_ALIGN = 256

# Kaggle google/embeddinggemma-2/transformers/embeddinggemma-2/1 (identical to
# the files the Kaggle reference kernels mounted).
SOURCE_FILES = {
    "model.safetensors": (
        1_488_915_288,
        "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
    ),
    "tokenizer.json": (
        32_170_510,
        "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    ),
    "config.json": (
        4_455,
        "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
    ),
}
ARTIFACT_NAME = "embeddinggemma2-text-bf16.bin"
OUTPUT_SIZE = 542_996_480
OUTPUT_SHA256 = "dd745d5e7402ba8e3e654a978adb70ab4af97e780f000742d6a8a309773a492c"
DEFAULT_DESTINATION = Path("models/embeddinggemma2-text")

CONFIG = {
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_layers": 24,
    "num_attention_heads": 4,
    "vocab_size": 262_144,
    "embedding_dim": 768,
    "rms_norm_eps": 1e-6,
    "sliding_window": 512,
    "max_position_embeddings": 8192,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "pad_token_id": 0,
    # Per layer type: head dim, KV heads, RoPE theta.
    "sliding_attention": {"head_dim": 256, "num_key_value_heads": 2, "rope_theta": 10_000.0},
    "full_attention": {"head_dim": 512, "num_key_value_heads": 1, "rope_theta": 1_000_000.0},
    "layer_types": [
        "full_attention" if i % 6 == 5 else "sliding_attention" for i in range(24)
    ],
}

PREFIX = "language_model."
LAYER_LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "ple_block.per_layer_input_gate",
    "ple_block.per_layer_projection",
)
LAYER_F32 = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "ple_block.post_per_layer_input_norm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "layer_scalar",
)


def plan(num_layers: int = CONFIG["num_layers"]):
    """(runtime name, source tensor name, kind) for every artifact tensor;
    kind is "bf16" (matrix, copied) or "f32" (widened)."""
    out = [("embed_tokens", "embed_tokens.weight", "bf16")]
    out.append(("ple.per_layer_model_projection", "ple.per_layer_model_projection.weight", "bf16"))
    out.append(("ple.per_layer_projection_norm", "ple.per_layer_projection_norm.weight", "f32"))
    for i in range(num_layers):
        for name in LAYER_LINEARS:
            out.append((f"layers.{i}.{name}", f"layers.{i}.{name}.weight", "bf16"))
        for name in LAYER_F32:
            out.append((f"layers.{i}.{name.removesuffix('.weight')}", f"layers.{i}.{name}", "f32"))
    out.append(("norm", "norm.weight", "f32"))
    out.append(("embedding_projection", "embedding_projection.weight", "f32"))
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(source: Path, pins=SOURCE_FILES) -> dict:
    """Size- and SHA256-check every pinned source file; returns the pins."""
    record = {}
    for name, (size, sha256) in pins.items():
        path = source / name
        if not path.is_file():
            raise SystemExit(f"{path}: missing (download the pinned Kaggle model first)")
        if path.stat().st_size != size:
            raise SystemExit(f"{path}: {path.stat().st_size} bytes, expected {size}")
        actual = sha256_file(path)
        if actual != sha256:
            raise SystemExit(f"{path}: sha256 {actual}, expected {sha256}")
        record[name] = {"size": size, "sha256": sha256}
    return record


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if not 0 < length <= 64 << 20:
            raise SystemExit(f"{path}: implausible safetensors header length {length}")
        header = json.loads(handle.read(length))
    return header, 8 + length


def bf16_to_f32(raw: bytes) -> bytes:
    """BF16 little-endian bytes -> F32 little-endian bytes (exact)."""
    halves = array.array("H")
    halves.frombytes(raw)
    if sys.byteorder != "little":
        halves.byteswap()
    words = array.array("I", (h << 16 for h in halves))
    if sys.byteorder != "little":
        words.byteswap()
    return words.tobytes()


def align(value: int, to: int) -> int:
    return (value + to - 1) // to * to


def build_header(tensors: dict, payload_length: int, payload_sha256: str, source: dict) -> bytes:
    header = {
        "format": FORMAT,
        "version": VERSION,
        "config": CONFIG,
        "source": source,
        "tensors": tensors,
        "payload_length": payload_length,
        "payload_sha256": payload_sha256,
    }
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    padded = align(16 + len(raw), HEADER_ALIGN) - 16
    return raw + b" " * (padded - len(raw))


def convert(source: Path, destination: Path, pins=SOURCE_FILES,
            expected=(OUTPUT_SIZE, OUTPUT_SHA256), num_layers=CONFIG["num_layers"]) -> dict:
    """Write `destination/ARTIFACT_NAME`, `tokenizer.json` and `manifest.json`
    (no clobbering: the destination files must not exist)."""
    pinned = verify_sources(source, pins)
    model = source / "model.safetensors"
    header, data_start = read_safetensors_header(model)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / ARTIFACT_NAME
    for path in (artifact, destination / "tokenizer.json", destination / "manifest.json"):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"{path}: exists; remove the old install first")
    partial = artifact.with_name(artifact.name + ".partial")
    payload = destination / (ARTIFACT_NAME + ".payload.partial")
    tensors, offset = {}, 0
    digest = hashlib.sha256()
    with model.open("rb") as handle, payload.open("xb") as out:

        def emit(name, raw, dtype, shape):
            nonlocal offset
            pad = align(offset, TENSOR_ALIGN) - offset
            if pad:
                out.write(b"\0" * pad)
                digest.update(b"\0" * pad)
                offset += pad
            out.write(raw)
            digest.update(raw)
            tensors[name] = {"dtype": dtype, "shape": shape, "offset": offset, "length": len(raw)}
            offset += len(raw)

        for runtime, src, kind in plan(num_layers):
            info = header.get(PREFIX + src)
            if info is None or info["dtype"] != "BF16":
                raise SystemExit(f"source tensor {PREFIX + src} missing or not BF16")
            begin, end = info["data_offsets"]
            shape = list(info["shape"])
            count = 1
            for dim in shape:
                count *= dim
            if end - begin != 2 * count:
                raise SystemExit(f"source tensor {PREFIX + src} has inconsistent bounds")
            handle.seek(data_start + begin)
            raw = handle.read(end - begin)
            if kind == "bf16":
                if len(shape) != 2:
                    raise SystemExit(f"source tensor {PREFIX + src} is not a matrix")
                emit(runtime, raw, "bf16", shape)
            else:
                emit(runtime, bf16_to_f32(raw), "f32", [count])
    payload_sha256 = digest.hexdigest()
    head = build_header(tensors, offset, payload_sha256, pinned)
    try:
        with partial.open("xb") as out, payload.open("rb") as body:
            out.write(MAGIC)
            out.write(struct.pack("<Q", len(head)))
            out.write(head)
            while chunk := body.read(1 << 22):
                out.write(chunk)
        payload.unlink()
        size, sha256 = partial.stat().st_size, sha256_file(partial)
        want_size, want_sha = expected
        if want_sha is not None and (size, sha256) != (want_size, want_sha):
            raise SystemExit(f"artifact is {size} bytes / {sha256}, expected "
                             f"{want_size} / {want_sha}: the conversion changed")
        os.link(partial, artifact)
        partial.unlink()
    finally:
        payload.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    tokenizer = destination / "tokenizer.json"
    with (source / "tokenizer.json").open("rb") as src_handle, tokenizer.open("xb") as out:
        while chunk := src_handle.read(1 << 22):
            out.write(chunk)
    files = {
        ARTIFACT_NAME: {"size": size, "sha256": sha256},
        "tokenizer.json": dict(pinned["tokenizer.json"]),
    }
    manifest = {
        "model": "google/embeddinggemma-2 (Kaggle google/embeddinggemma-2/transformers/"
                 "embeddinggemma-2/1), text path, BF16 matrices, F32 vectors",
        "files": files,
        "source": pinned,
        "payload_sha256": payload_sha256,
        "converter": "tools/convert_embeddinggemma2.py",
    }
    with (destination / "manifest.json").open("x") as out:
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
