"""Convert the Kaggle EmbeddingGemma 2 checkpoint's audio encoder into the runtime's audio artifact.

The audio artifact is an optional add-on to the text install written by
tools/convert_embeddinggemma2.py: audio soft tokens run through the same text
encoder, so the text artifact must be installed too. With the pinned Kaggle
checkpoint downloaded (see convert_embeddinggemma2.py):

  python3 tools/convert_embeddinggemma2_audio.py --source cache/embeddinggemma-2

writes models/embeddinggemma2-text/{embeddinggemma2-audio-bf16.bin,
manifest-audio.json} (standard library only).

Converted: the `gemma4_audio` Conformer (`audio_tower.*`: subsampling conv stem,
12 layers, output_proj 1024->1536 with bias) and `embed_audio` (1536->512).
Every source file the conversion reads is size- and SHA256-checked, and the
written artifact must hash to OUTPUT_SHA256.

Precision (docs/EMBEDDINGS.md): the Linear matrices keep the checkpoint's own
BF16 values (copied byte for byte); conv kernels, norms, per-dim scales, the
output bias and the clipped-linear bounds (input_min/max, output_min/max of
every Gemma4ClippableLinear) are widened to F32 (exact).

Layout: as the text artifact (convert_embeddinggemma2.py) with magic
b"EG2AUD\\x00\\x01" and format "embeddinggemma2-audio"; clamp bounds are
stored as one F32 vector [input_min, input_max, output_min, output_max] per
clipped linear under "<linear>.clip".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import convert_embeddinggemma2 as text

MAGIC = b"EG2AUD\x00\x01"
FORMAT = "embeddinggemma2-audio"
VERSION = 1
SOURCE_FILES = {name: text.SOURCE_FILES[name] for name in ("model.safetensors", "config.json")}
ARTIFACT_NAME = "embeddinggemma2-audio-bf16.bin"
MANIFEST_NAME = "manifest-audio.json"
OUTPUT_SIZE = 611_720_448
OUTPUT_SHA256 = "3b3354a4df46fb2463d5b10b2d6bfa80e616aa07f1f87c658cdd184023dbe3bc"
DEFAULT_DESTINATION = text.DEFAULT_DESTINATION

CONFIG = {
    "hidden_size": 1024,
    "num_layers": 12,
    "num_attention_heads": 8,
    "intermediate_size": 4096,
    "attention_chunk_size": 12,
    "attention_context_left": 13,
    "attention_context_right": 0,
    "attention_logit_cap": 50.0,
    "attention_invalid_logits_value": -1e9,
    "conv_kernel_size": 5,
    "residual_weight": 0.5,
    "gradient_clipping": 1e10,
    "rms_norm_eps": 1e-6,
    "output_proj_dims": 1536,
    "subsampling_conv_channels": [128, 32],
    "text_hidden_size": 512,
    "use_clipped_linears": True,
    "audio_token_id": 258881,
    "boa_token_id": 256000,
    "eoa_token_id": 258883,
    "feature_extractor": {
        "sampling_rate": 16000, "feature_size": 128, "frame_length": 320, "hop_length": 160,
        "fft_length": 512, "min_frequency": 0.0, "max_frequency": 8000.0, "mel_floor": 0.001,
        "max_samples": 480000, "pad_to_multiple_of": 128, "mel_scale": "htk",
    },
}

CLIPPED = (
    "feed_forward1.ffw_layer_1", "feed_forward1.ffw_layer_2",
    "feed_forward2.ffw_layer_1", "feed_forward2.ffw_layer_2",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.post",
    "lconv1d.linear_start", "lconv1d.linear_end",
)
LAYER_F32 = (
    "feed_forward1.pre_layer_norm.weight", "feed_forward1.post_layer_norm.weight",
    "feed_forward2.pre_layer_norm.weight", "feed_forward2.post_layer_norm.weight",
    "norm_pre_attn.weight", "norm_post_attn.weight", "norm_out.weight",
    "self_attn.per_dim_scale",
    "lconv1d.pre_layer_norm.weight", "lconv1d.conv_norm.weight", "lconv1d.depthwise_conv1d.weight",
)
CLIP_BOUNDS = ("input_min", "input_max", "output_min", "output_max")


def plan(num_layers: int = CONFIG["num_layers"]):
    """(runtime name, source tensor name(s), kind) for every artifact tensor; kind is
    "bf16" (matrix, copied), "f32" (widened, flattened) or "clip" (4 scalars -> one vector)."""
    stem = "audio_tower.subsample_conv_projection."
    out = [
        ("stem.layer0.conv", stem + "layer0.conv.weight", "f32"),
        ("stem.layer0.norm", stem + "layer0.norm.weight", "f32"),
        ("stem.layer1.conv", stem + "layer1.conv.weight", "f32"),
        ("stem.layer1.norm", stem + "layer1.norm.weight", "f32"),
        ("stem.input_proj", stem + "input_proj_linear.weight", "bf16"),
    ]
    for i in range(num_layers):
        p = f"audio_tower.layers.{i}."
        for name in CLIPPED:
            out.append((f"layers.{i}.{name}", p + name + ".linear.weight", "bf16"))
            out.append((f"layers.{i}.{name}.clip", tuple(p + name + "." + b for b in CLIP_BOUNDS), "clip"))
        out.append((f"layers.{i}.self_attn.relative_k_proj", p + "self_attn.relative_k_proj.weight", "bf16"))
        for name in LAYER_F32:
            out.append((f"layers.{i}.{name.removesuffix('.weight')}", p + name, "f32"))
    out.append(("output_proj", "audio_tower.output_proj.weight", "bf16"))
    out.append(("output_proj.bias", "audio_tower.output_proj.bias", "f32"))
    out.append(("embed_audio", "embed_audio.embedding_projection.weight", "bf16"))
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

    def read(handle, name):
        info = header.get(name)
        if info is None or info["dtype"] != "BF16":
            raise SystemExit(f"source tensor {name} missing or not BF16")
        begin, end = info["data_offsets"]
        count = 1
        for dim in info["shape"]:
            count *= dim
        if end - begin != 2 * count:
            raise SystemExit(f"source tensor {name} has inconsistent bounds")
        handle.seek(data_start + begin)
        return handle.read(end - begin), list(info["shape"]), count

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
            if kind == "clip":
                raw = b"".join(read(handle, name)[0] for name in src)
                if len(raw) != 8:
                    raise SystemExit(f"clip bounds of {runtime} are not 4 scalars")
                emit(runtime, text.bf16_to_f32(raw), "f32", [4])
                continue
            raw, shape, count = read(handle, src)
            if kind == "bf16":
                if len(shape) != 2:
                    raise SystemExit(f"source tensor {src} is not a matrix")
                emit(runtime, raw, "bf16", shape)
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
                 "embeddinggemma-2/1), audio encoder + embed_audio, BF16 matrices, F32 vectors",
        "files": {ARTIFACT_NAME: {"size": size, "sha256": sha256}},
        "source": pinned,
        "payload_sha256": payload_sha256,
        "converter": "tools/convert_embeddinggemma2_audio.py",
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
