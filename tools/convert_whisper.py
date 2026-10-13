"""Convert OpenAI's Whisper large-v3 checkpoint into the runtime's speech artifact.

The source is `openai/whisper-large-v3` (MIT) at revision
06f233fe06e710322aca913c1bc4249a0d71fce1 in the transformers layout, plus the
original package's 128-bin mel filterbank (`whisper/assets/mel_filters.npz`,
openai-whisper 20250625). They reach this machine through a private Kaggle CPU
kernel that downloads them and saves them as its output (docs/TRANSCRIPTION.md):

  kaggle kernels output <owner>/stt-whisper-ref-s0 -p cache/whisper-large-v3

then convert (standard library only):

  python3 tools/convert_whisper.py --source cache/whisper-large-v3

Every source file the conversion reads is size- and SHA256-checked against the
pins below, and the written artifact must hash to OUTPUT_SHA256, so a changed
checkpoint or a changed conversion fails instead of installing silently. The
Kaggle reference job checked the transformers weights against the original
openai-whisper `large-v3.pt`: all 1,259 tensors are bit-identical.

Precision: every matrix keeps the checkpoint's own F16 values (copied byte for
byte; the conv kernels [out, in, 3] are flattened to [out, in * 3], which is
their memory order already). Biases, LayerNorm parameters and the encoder's
positional table are widened to F32 (exact); the decoder's positional table
stays F16 (it is read on the CPU).

Artifact layout (little endian):

  0   8 bytes  magic b"WHSPRV3\\x01"
  8   u64      header length H (JSON, space padded so the payload starts at a
               multiple of 4096)
  16  H bytes  header: {"format", "version", "config", "tokens", "source",
               "tensors": {name: {"dtype": "f16"|"f32", "shape", "offset",
               "length"}}, "payload_length", "payload_sha256"}
  16+H         payload; tensor offsets are relative to it, 256-byte aligned
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import struct
import zipfile
from pathlib import Path

MAGIC = b"WHSPRV3\x01"
FORMAT = "whisper-large-v3"
VERSION = 1
HEADER_ALIGN = 4096
TENSOR_ALIGN = 256
PREFIX = "model."

# openai/whisper-large-v3 @ 06f233fe06e710322aca913c1bc4249a0d71fce1 and
# openai-whisper 20250625's assets/mel_filters.npz, as the Kaggle kernel saved
# them.
SOURCE_FILES = {
    "model/model.safetensors": (
        3_087_130_976,
        "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
    ),
    "model/config.json": (
        1_272,
        "ad0e8d1e46f4d01f7861a21509e5d0f977d6cc1f367a370603c92541d819807b",
    ),
    "model/generation_config.json": (
        3_903,
        "fbdfa70135de9b1d31553393f14e80aaeb1936ea36576b2ba864055943c09d23",
    ),
    "model/tokenizer.json": (
        2_480_617,
        "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca",
    ),
    "mel_filters.npz": (
        4_271,
        "7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c",
    ),
}
ARTIFACT_NAME = "whisper-large-v3-f16.bin"
OUTPUT_SIZE = 3_093_458_432
OUTPUT_SHA256 = "cce7ae83d4a15146f356a9f3bb5169b0fb42a86dc2e6a0c6cf932766e6d1cb53"
DEFAULT_DESTINATION = Path("models/whisper-large-v3")

CONFIG = {
    "n_mels": 128,
    "n_audio_ctx": 1500,
    "n_audio_state": 1280,
    "n_audio_head": 20,
    "n_audio_layer": 32,
    "n_vocab": 51866,
    "n_text_ctx": 448,
    "n_text_state": 1280,
    "n_text_head": 20,
    "n_text_layer": 32,
    "n_ffn": 5120,
    "n_fft_bins": 201,
}
# transformers config.json keys that must hold these values.
HF_CONFIG = {
    "num_mel_bins": 128,
    "max_source_positions": 1500,
    "d_model": 1280,
    "encoder_attention_heads": 20,
    "encoder_layers": 32,
    "vocab_size": 51866,
    "max_target_positions": 448,
    "decoder_attention_heads": 20,
    "decoder_layers": 32,
    "encoder_ffn_dim": 5120,
    "decoder_ffn_dim": 5120,
    "activation_function": "gelu",
    "scale_embedding": False,
}

ATTENTION = ("q_proj", "k_proj", "v_proj", "out_proj")


def plan(encoder_layers: int = CONFIG["n_audio_layer"],
         decoder_layers: int = CONFIG["n_text_layer"]):
    """(name, kind) for every checkpoint tensor the runtime reads, in payload
    order; kind is "f16" (matrix, copied), "f32" (widened) or "conv" (F16
    [out, in, 3] copied as [out, in * 3])."""
    out = [("encoder.conv1.weight", "conv"), ("encoder.conv1.bias", "f32"),
           ("encoder.conv2.weight", "conv"), ("encoder.conv2.bias", "f32"),
           ("encoder.embed_positions.weight", "f32")]

    def block(prefix: str, attentions: tuple[str, ...]):
        for attention in attentions:
            for proj in ATTENTION:
                out.append((f"{prefix}.{attention}.{proj}.weight", "f16"))
                if proj != "k_proj":
                    out.append((f"{prefix}.{attention}.{proj}.bias", "f32"))
            out.append((f"{prefix}.{attention}_layer_norm.weight", "f32"))
            out.append((f"{prefix}.{attention}_layer_norm.bias", "f32"))
        for fc in ("fc1", "fc2"):
            out.append((f"{prefix}.{fc}.weight", "f16"))
            out.append((f"{prefix}.{fc}.bias", "f32"))
        out.append((f"{prefix}.final_layer_norm.weight", "f32"))
        out.append((f"{prefix}.final_layer_norm.bias", "f32"))

    for i in range(encoder_layers):
        block(f"encoder.layers.{i}", ("self_attn",))
    out += [("encoder.layer_norm.weight", "f32"), ("encoder.layer_norm.bias", "f32"),
            ("decoder.embed_tokens.weight", "f16"), ("decoder.embed_positions.weight", "f16")]
    for i in range(decoder_layers):
        block(f"decoder.layers.{i}", ("self_attn", "encoder_attn"))
    out += [("decoder.layer_norm.weight", "f32"), ("decoder.layer_norm.bias", "f32")]
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
            raise SystemExit(f"{path}: missing (download the pinned files first)")
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
    header.pop("__metadata__", None)
    return header, 8 + length


def f16_to_f32(raw: bytes) -> bytes:
    """F16 little-endian bytes -> F32 little-endian bytes (exact)."""
    count = len(raw) // 2
    values = struct.unpack(f"<{count}e", raw)
    return struct.pack(f"<{count}f", *values)


def read_npy_f32(raw: bytes) -> tuple[list[int], bytes]:
    """Shape and little-endian F32 bytes of a C-order '<f4' .npy file."""
    if raw[:6] != b"\x93NUMPY":
        raise SystemExit("mel filters: not an .npy file")
    major = raw[6]
    if major == 1:
        (length,) = struct.unpack("<H", raw[8:10])
        start = 10
    else:
        (length,) = struct.unpack("<I", raw[8:12])
        start = 12
    meta = ast.literal_eval(raw[start:start + length].decode("latin1"))
    if meta.get("descr") != "<f4" or meta.get("fortran_order"):
        raise SystemExit(f"mel filters: unsupported array {meta}")
    shape = list(meta["shape"])
    data = raw[start + length:]
    if len(data) != 4 * shape[0] * shape[1]:
        raise SystemExit("mel filters: size does not match the shape")
    return shape, data


def mel_filters(path: Path, n_mels: int) -> bytes:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(f"mel_{n_mels}.npy")
    shape, data = read_npy_f32(raw)
    if shape != [n_mels, CONFIG["n_fft_bins"]]:
        raise SystemExit(f"mel filters: shape {shape}")
    return data


def token_config(generation: dict, tokenizer: dict) -> dict:
    """Special-token ids the runtime needs, from the pinned configs."""
    added = {item["content"]: item["id"] for item in tokenizer.get("added_tokens", [])}

    def need(content: str) -> int:
        if content not in added:
            raise SystemExit(f"tokenizer.json lacks {content}")
        return added[content]

    languages = {}
    for token, ident in generation["lang_to_id"].items():
        if not (token.startswith("<|") and token.endswith("|>")):
            raise SystemExit(f"unexpected language token {token}")
        languages[token[2:-2]] = ident
    tokens = {
        "sot": generation["decoder_start_token_id"],
        "eot": generation["eos_token_id"],
        "sot_prev": generation["prev_sot_token_id"],
        "no_timestamps": generation["no_timestamps_token_id"],
        "transcribe": generation["task_to_id"]["transcribe"],
        "translate": generation["task_to_id"]["translate"],
        "sot_lm": need("<|startoflm|>"),
        "no_speech": need("<|nospeech|>"),
        "timestamp_begin": need("<|0.00|>"),
        "languages": dict(sorted(languages.items(), key=lambda kv: kv[1])),
        "suppress_tokens": list(generation["suppress_tokens"]),
        "begin_suppress_tokens": list(generation["begin_suppress_tokens"]),
        "alignment_heads": [list(pair) for pair in generation["alignment_heads"]],
        "max_initial_timestamp_index": generation["max_initial_timestamp_index"],
    }
    if need("<|endoftext|>") != tokens["eot"] or need("<|startoftranscript|>") != tokens["sot"]:
        raise SystemExit("generation_config.json and tokenizer.json disagree")
    return tokens


def check_config(hf: dict) -> None:
    for key, want in HF_CONFIG.items():
        if hf.get(key) != want:
            raise SystemExit(f"config.json {key}={hf.get(key)!r}, expected {want!r}")


def align(value: int, to: int) -> int:
    return (value + to - 1) // to * to


def build_header(tensors: dict, tokens: dict, payload_length: int, payload_sha256: str,
                 source: dict) -> bytes:
    header = {
        "format": FORMAT,
        "version": VERSION,
        "config": CONFIG,
        "tokens": tokens,
        "source": source,
        "tensors": tensors,
        "payload_length": payload_length,
        "payload_sha256": payload_sha256,
    }
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    padded = align(16 + len(raw), HEADER_ALIGN) - 16
    return raw + b" " * (padded - len(raw))


def convert(source: Path, destination: Path, pins=SOURCE_FILES,
            expected=(OUTPUT_SIZE, OUTPUT_SHA256), layers=(CONFIG["n_audio_layer"],
                                                         CONFIG["n_text_layer"]),
            check_hf_config=True) -> dict:
    """Write `destination/ARTIFACT_NAME`, `tokenizer.json` and `manifest.json`
    (no clobbering: the destination files must not exist)."""
    pinned = verify_sources(source, pins)
    hf_config = json.loads((source / "model/config.json").read_text())
    if check_hf_config:
        check_config(hf_config)
    generation = json.loads((source / "model/generation_config.json").read_text())
    tokenizer_json = json.loads((source / "model/tokenizer.json").read_text())
    tokens = token_config(generation, tokenizer_json)
    model = source / "model/model.safetensors"
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
    used = set()
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

        for name, kind in plan(*layers):
            info = header.get(PREFIX + name)
            if info is None or info["dtype"] != "F16":
                raise SystemExit(f"source tensor {PREFIX + name} missing or not F16")
            used.add(PREFIX + name)
            begin, end = info["data_offsets"]
            shape = list(info["shape"])
            count = 1
            for dim in shape:
                count *= dim
            if end - begin != 2 * count:
                raise SystemExit(f"source tensor {PREFIX + name} has inconsistent bounds")
            handle.seek(data_start + begin)
            raw = handle.read(end - begin)
            if kind == "conv":
                if len(shape) != 3 or shape[2] != 3:
                    raise SystemExit(f"source tensor {PREFIX + name} is not a kernel-3 conv")
                emit(name, raw, "f16", [shape[0], shape[1] * 3])
            elif kind == "f16":
                if len(shape) != 2:
                    raise SystemExit(f"source tensor {PREFIX + name} is not a matrix")
                emit(name, raw, "f16", shape)
            else:
                emit(name, f16_to_f32(raw), "f32", shape)
        emit("mel_filters", mel_filters(source / "mel_filters.npz", CONFIG["n_mels"]), "f32",
             [CONFIG["n_mels"], CONFIG["n_fft_bins"]])
    unused = sorted(set(header) - used)
    if unused:
        payload.unlink(missing_ok=True)
        raise SystemExit(f"checkpoint tensors the runtime does not read: {unused[:5]}")
    payload_sha256 = digest.hexdigest()
    head = build_header(tensors, tokens, offset, payload_sha256, pinned)
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
    with (source / "model/tokenizer.json").open("rb") as src_handle, tokenizer.open("xb") as out:
        while chunk := src_handle.read(1 << 22):
            out.write(chunk)
    files = {
        ARTIFACT_NAME: {"size": size, "sha256": sha256},
        "tokenizer.json": dict(pinned["model/tokenizer.json"]),
    }
    manifest = {
        "model": "openai/whisper-large-v3 (MIT) @ 06f233fe06e710322aca913c1bc4249a0d71fce1, "
                 "F16 matrices, F32 vectors; mel filters from openai-whisper 20250625",
        "files": files,
        "source": pinned,
        "payload_sha256": payload_sha256,
        "converter": "tools/convert_whisper.py",
    }
    with (destination / "manifest.json").open("x") as out:
        json.dump(manifest, out, indent=2, sort_keys=True)
        out.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="directory of the Kaggle kernel output (model/, mel_filters.npz)")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION,
                        help=f"install directory (default: {DEFAULT_DESTINATION})")
    args = parser.parse_args()
    manifest = convert(args.source, args.destination)
    for name, record in manifest["files"].items():
        print(args.destination / name, record["size"], record["sha256"])


if __name__ == "__main__":
    main()
