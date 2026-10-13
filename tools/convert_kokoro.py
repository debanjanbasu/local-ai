"""Convert Kokoro-82M (hexgrad, Apache-2.0) into the runtime's speech artifact.

The source is the archive written by the private Kaggle CPU job staged with
`tools/kaggle_kokoro_job.py --mode reference` (it downloads
`hexgrad/Kokoro-82M` and the pinned `misaki` package on Kaggle; nothing is
fetched from Hugging Face on this machine):

  kaggle kernels output OWNER/kokoro-ref-tts -p cache/kokoro
  tar -xf cache/kokoro/kokoro-ref.tar -C cache/kokoro
  python3 tools/convert_kokoro.py --source cache/kokoro/kokoro-ref

Standard library only. Every source file read is size- and SHA256-checked
against the pins below and the written artifact must hash to OUTPUT_SHA256.

What is converted:

- every weight of `kokoro-v1_0.pth` as F32 (the checkpoint's precision), with
  weight normalization folded (w = g * v / ||v||, norm over all but dim 0)
  and layouts the Metal kernels read directly: Conv1d weights
  [out][tap][in]; generator ConvTranspose1d weights split into their `stride`
  output phases, [phase][out][tap][in] with tap kk = kernel index
  phase + kk * stride; LSTM input weights of both directions stacked
  [2 * 4H][in] with b_ih + b_hh summed, recurrent weights transposed
  [2][H][4H]; Snake alphas [C];
- the 28 English voice packs ([510][256] F32 each);
- the four English `misaki` lexicons (`us_gold`, `us_silver`, `gb_gold`,
  `gb_silver`, JSON verbatim) for the native G2P.

Artifact layout (little endian):

  0   8 bytes  magic b"KOKORO\\x00\\x01"
  8   u64      header length H (JSON, space padded so the payload starts at a
               multiple of 4096)
  16  H bytes  header: {"format", "version", "config", "tensors": {name:
               {"shape", "offset"}}, "voices": {name: offset}, "blobs":
               {name: {"offset", "length"}}, "payload_length",
               "payload_sha256"}
  16+H         payload; offsets are relative to it, 256-byte aligned
"""

from __future__ import annotations

import argparse
import array
import collections
import hashlib
import json
import math
import os
import pickle
import struct
import sys
import zipfile
from pathlib import Path

MAGIC = b"KOKORO\x00\x01"
FORMAT = "kokoro-82m"
VERSION = 1
HEADER_ALIGN = 4096
TENSOR_ALIGN = 256
ARTIFACT_NAME = "kokoro-82m-f32.bin"
DEFAULT_DESTINATION = Path("models/kokoro-82m")

VOICES = (
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam",
    "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)
LEXICONS = ("us_gold", "us_silver", "gb_gold", "gb_silver")

# Paths relative to the extracted `kokoro-ref/` directory. Filled from the
# Kaggle job's `upstream.json`, whose model-file digests it checked against
# the Hugging Face LFS metadata of hexgrad/Kokoro-82M.
SOURCE_FILES: dict[str, tuple[int, str]] = {
    "model/config.json": (2_351, "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"),
    "model/kokoro-v1_0.pth": (327_212_226, "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"),
    "model/voices/af_alloy.pt": (523_425, "6d877149dd8b348fbad12e5845b7e43d975390e9f3b68a811d1d86168bef5aa3"),
    "model/voices/af_aoede.pt": (523_425, "c03bd1a4c3716c2d8eaa3d50022f62d5c31cfbd6e15933a00b17fefe13841cc4"),
    "model/voices/af_bella.pt": (523_425, "8cb64e02fcc8de0327a8e13817e49c76c945ecf0052ceac97d3081480e8e48d6"),
    "model/voices/af_heart.pt": (523_425, "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff"),
    "model/voices/af_jessica.pt": (523_435, "cdfdccb8cc975aa34ee6b89642963b0064237675de0e41a30ae64cc958dd4e87"),
    "model/voices/af_kore.pt": (523_420, "8bfbc512321c3db49dff984ac675fa5ac7eaed5a96cc31104d3a9080e179d69d"),
    "model/voices/af_nicole.pt": (523_430, "c5561808bcf5250fe8c5f5de32caf2d94f27e57e95befdb098c5c85991d4c5da"),
    "model/voices/af_nova.pt": (523_420, "e0233676ddc21908c37a1f102f6b88a59e4e5c1bd764983616eb9eda629dbcd2"),
    "model/voices/af_river.pt": (523_425, "e149459bd9c084416b74756b9bd3418256a8b839088abb07d463730c369dab8f"),
    "model/voices/af_sarah.pt": (523_425, "49bd364ea3be9eb3e9685e8f9a15448c4883112a7c0ff7ab139fa4088b08cef9"),
    "model/voices/af_sky.pt": (523_351, "c799548aed06e0cb0d655a85a01b48e7f10484d71663f9a3045a5b9362e8512c"),
    "model/voices/am_adam.pt": (523_420, "ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34"),
    "model/voices/am_echo.pt": (523_420, "8bcfdc852bc985fb45c396c561e571ffb9183930071f962f1b50df5c97b161e8"),
    "model/voices/am_eric.pt": (523_420, "ada66f0eefff34ec921b1d7474d7ac8bec00cd863c170f1c534916e9b8212aae"),
    "model/voices/am_fenrir.pt": (523_430, "98e507eca1db08230ae3b6232d59c10aec9630022d19accac4f5d12fcec3c37a"),
    "model/voices/am_liam.pt": (523_420, "c82550757ddb31308b97f30040dda8c2d609a9e2de6135848d0a948368138518"),
    "model/voices/am_michael.pt": (523_435, "9a443b79a4b22489a5b0ab7c651a0bcd1a30bef675c28333f06971abbd47bd37"),
    "model/voices/am_onyx.pt": (523_420, "e8452be16cd0f6da7b4579eaf7b1e4506e92524882053d86d72b96b9a7fed584"),
    "model/voices/am_puck.pt": (523_420, "dd1d8973f4ce4b7d8ae407c77a435f485dabc052081b80ea75c4f30b84f36223"),
    "model/voices/am_santa.pt": (523_425, "7f2f7582fa2b1f160e90aafe6d0b442a685e773608b6667e545d743b073e97a7"),
    "model/voices/bf_alice.pt": (523_425, "d292651b6af6c0d81705c2580dcb4463fccc0ff7b8d618a471dbb4e45655b3f3"),
    "model/voices/bf_emma.pt": (523_420, "d0a423deabf4a52b4f49318c51742c54e21bb89bbbe9a12141e7758ddb5da701"),
    "model/voices/bf_isabella.pt": (523_440, "cdd4c37003805104d1d08fb1e05855c8fb2c68de24ca6e71f264a30aaa59eefd"),
    "model/voices/bf_lily.pt": (523_420, "6e09c2e481e2d53004d7e5ae7d3a325369e130a6f45c35a6002de75084be9285"),
    "model/voices/bm_daniel.pt": (523_430, "fc3fce4e9c12ed4dbc8fa9680cfe51ee190a96444ce7c3ad647549a30823fc5d"),
    "model/voices/bm_fable.pt": (523_425, "d44935f3135257a9064df99f007fc1342ff1aa767552b4a4fa4c3b2e6e59079c"),
    "model/voices/bm_george.pt": (523_430, "f1bc812213dc59774769e5c80004b13eeb79bd78130b11b2d7f934542dab811b"),
    "model/voices/bm_lewis.pt": (523_425, "b5204750dcba01029d2ac9cec17aec3b20a6d64073c579d694a23cb40effbd0e"),
    "src/misaki/data/gb_gold.json": (2_838_552, "29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71"),
    "src/misaki/data/gb_silver.json": (3_663_898, "48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce"),
    "src/misaki/data/us_gold.json": (3_000_469, "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f"),
    "src/misaki/data/us_silver.json": (3_099_517, "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475"),
}
OUTPUT_SIZE = 354_146_560
OUTPUT_SHA256 = "aa3b5c769b9fdcc9eac3d6068352610183e45c20542f1232d629ead82c695928"

# Generator ConvTranspose1d layers: name -> (stride, padding).
TRANSPOSED = {"decoder.generator.ups.0": (10, 5), "decoder.generator.ups.1": (6, 3)}
LSTM_HIDDEN = 256


class Tensor:
    """A loaded F32 tensor: flat values plus shape."""

    def __init__(self, values: array.array, shape: tuple[int, ...]):
        if math.prod(shape) != len(values):
            raise ValueError("tensor size does not match its shape")
        self.values = values
        self.shape = tuple(shape)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sources(source: Path, pins: dict[str, tuple[int, str]]) -> None:
    if not pins:
        raise SystemExit("no source pins: this converter build is incomplete")
    for name, (size, digest) in sorted(pins.items()):
        path = source / name
        if not path.is_file():
            raise SystemExit(f"missing source file {path}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise SystemExit(f"{path} does not match its pinned size and SHA256")


_DTYPES = {"FloatStorage": ("f", 4)}


class _Storage:
    def __init__(self, typecode: str, key: str):
        self.typecode = typecode
        self.key = key


def _rebuild_tensor(storage, offset, size, stride, *_rest):
    expected, step = [], 1
    for dim in reversed(size):
        expected.insert(0, step)
        step *= dim
    if tuple(stride) != tuple(expected) and math.prod(size) > 1:
        raise ValueError("non-contiguous tensor in checkpoint")
    return ("tensor", storage, offset, tuple(size))


def _rebuild_parameter(data, *_rest):
    return data


class _Unpickler(pickle.Unpickler):
    """Restricted unpickler for `torch.save` zip archives: tensors and dicts only."""

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if (module, name) == ("torch._utils", "_rebuild_parameter"):
            return _rebuild_parameter
        if module == "torch" and name in _DTYPES:
            return name
        raise pickle.UnpicklingError(f"refusing {module}.{name} in a checkpoint")

    def persistent_load(self, pid):
        kind, storage_type, key, _location, _numel = pid
        if kind != "storage" or storage_type not in _DTYPES:
            raise pickle.UnpicklingError(f"unsupported storage {storage_type!r}")
        return _Storage(_DTYPES[storage_type][0], key)


def load_torch(path: Path):
    """Load a `torch.save` archive of (nested dicts of) F32 tensors."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickles = [n for n in names if n.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise ValueError(f"{path}: expected one data.pkl")
        root = pickles[0][: -len("data.pkl")]
        tree = _Unpickler(archive.open(pickles[0])).load()
        cache: dict[str, array.array] = {}

        def materialize(node):
            if isinstance(node, dict):
                return {key: materialize(value) for key, value in node.items()}
            if isinstance(node, tuple) and node and node[0] == "tensor":
                _, storage, offset, size = node
                if storage.key not in cache:
                    values = array.array(storage.typecode)
                    values.frombytes(archive.read(f"{root}data/{storage.key}"))
                    if sys.byteorder != "little":
                        values.byteswap()
                    cache[storage.key] = values
                count = math.prod(size)
                return Tensor(cache[storage.key][offset : offset + count], size)
            raise ValueError(f"{path}: unexpected object {type(node).__name__}")

        return materialize(tree)


def fold_weight_norm(state: dict[str, Tensor]) -> dict[str, Tensor]:
    out = {}
    for name, tensor in state.items():
        if name.endswith(".weight_v"):
            continue
        if name.endswith(".weight_g"):
            stem = name[: -len("_g")]
            v = state[stem + "_v"]
            g = tensor
            rows = v.shape[0]
            if len(g.values) != rows:
                raise ValueError(f"{name}: weight norm is not over dim 0")
            per = len(v.values) // rows
            values = array.array("f", bytes(4 * len(v.values)))
            for r in range(rows):
                chunk = v.values[r * per : (r + 1) * per]
                norm = math.sqrt(sum(float(x) * float(x) for x in chunk))
                factor = g.values[r] / norm
                values[r * per : (r + 1) * per] = array.array("f", (x * factor for x in chunk))
            out[stem] = Tensor(values, v.shape)
        else:
            out[name] = tensor
    return out


def conv_layout(t: Tensor) -> Tensor:
    """[out, in, k] -> [out, k, in]."""
    co, ci, k = t.shape
    v = t.values
    values = array.array("f")
    for o in range(co):
        block = v[o * ci * k : (o + 1) * ci * k]
        for tap in range(k):
            values.extend(block[tap::k])
    return Tensor(values, (co, k, ci))


def transposed_layout(t: Tensor, stride: int) -> Tensor:
    """ConvTranspose1d [in, out, K] -> [stride][out][K / stride][in] (tap kk = phase + kk * stride)."""
    ci, co, k = t.shape
    if k % stride:
        raise ValueError("transposed kernel not divisible by its stride")
    v = t.values
    values = array.array("f")
    for r in range(stride):
        for o in range(co):
            for kk in range(k // stride):
                start = o * k + r + kk * stride
                values.extend(v[start : start + ci * co * k : co * k])
    return Tensor(values, (stride, co, k // stride, ci))


def lstm_layout(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    h = LSTM_HIDDEN
    w_ih, bias, whh = array.array("f"), array.array("f"), array.array("f")
    inputs = None
    for suffix in ("", "_reverse"):
        wi = state.pop(f"{prefix}.weight_ih_l0{suffix}")
        wh = state.pop(f"{prefix}.weight_hh_l0{suffix}")
        bi = state.pop(f"{prefix}.bias_ih_l0{suffix}")
        bh = state.pop(f"{prefix}.bias_hh_l0{suffix}")
        if wi.shape[0] != 4 * h or wh.shape != (4 * h, h):
            raise ValueError(f"{prefix}: unexpected LSTM shape")
        inputs = wi.shape[1]
        w_ih.extend(wi.values)
        bias.extend(array.array("f", (a + b for a, b in zip(bi.values, bh.values))))
        for j in range(h):
            whh.extend(wh.values[j::h])
    return {
        f"{prefix}.ih.weight": Tensor(w_ih, (8 * h, inputs)),
        f"{prefix}.ih.bias": Tensor(bias, (8 * h,)),
        f"{prefix}.whh_t": Tensor(whh, (2, h, 4 * h)),
    }


def is_lstm(state: dict[str, Tensor], name: str) -> bool:
    return name.endswith(".weight_ih_l0")


def convert_state(checkpoint: dict) -> dict[str, Tensor]:
    flat: dict[str, Tensor] = {}
    for group, state in checkpoint.items():
        for key, tensor in state.items():
            key = key.removeprefix("module.")
            flat[f"{group}.{key}"] = tensor
    flat = fold_weight_norm(flat)
    out: dict[str, Tensor] = {}
    for prefix in sorted({n[: -len(".weight_ih_l0")] for n in flat if is_lstm(flat, n)}):
        out.update(lstm_layout(flat, prefix))
    for name, tensor in flat.items():
        stem = name[: -len(".weight")] if name.endswith(".weight") else None
        if stem in TRANSPOSED:
            out[name] = transposed_layout(tensor, TRANSPOSED[stem][0])
        elif stem is not None and len(tensor.shape) == 3 and ".pool" not in stem:
            out[name] = conv_layout(tensor)
        elif len(tensor.shape) == 3 and tensor.shape[0] == 1 and tensor.shape[2] == 1:
            out[name] = Tensor(tensor.values, (tensor.shape[1],))  # Snake alpha [1, C, 1]
        else:
            out[name] = tensor
    return out


def build(source: Path, destination: Path, *, check: bool = True) -> Path:
    if check:
        check_sources(source, SOURCE_FILES)
    config = json.loads((source / "model/config.json").read_text(encoding="utf-8"))
    tensors = convert_state(load_torch(source / "model/kokoro-v1_0.pth"))
    voices = {}
    for voice in VOICES:
        pack = load_torch(source / f"model/voices/{voice}.pt")
        if pack.shape != (510, 1, 256):
            raise ValueError(f"{voice}: unexpected voice shape {pack.shape}")
        voices[voice] = pack.values
    blobs = {name: (source / f"src/misaki/data/{name}.json").read_bytes() for name in LEXICONS}

    payload = bytearray()

    def place(data: bytes) -> int:
        payload.extend(bytes(-len(payload) % TENSOR_ALIGN))
        offset = len(payload)
        payload.extend(data)
        return offset

    def f32_bytes(values: array.array) -> bytes:
        values = array.array("f", values)
        if sys.byteorder != "little":
            values.byteswap()
        return values.tobytes()

    header = {
        "format": FORMAT,
        "version": VERSION,
        "config": config,
        "tensors": {
            name: {"shape": list(t.shape), "offset": place(f32_bytes(t.values))}
            for name, t in sorted(tensors.items())
        },
        "voices": {name: place(f32_bytes(values)) for name, values in voices.items()},
        "blobs": {
            name: {"offset": place(data), "length": len(data)} for name, data in blobs.items()
        },
    }
    payload.extend(bytes(-len(payload) % TENSOR_ALIGN))
    header["payload_length"] = len(payload)
    header["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    text = json.dumps(header, sort_keys=True, ensure_ascii=False).encode()
    length = (len(text) + 16 + HEADER_ALIGN - 1) // HEADER_ALIGN * HEADER_ALIGN - 16
    text += b" " * (length - len(text))
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / ARTIFACT_NAME
    partial = target.with_suffix(".partial")
    with partial.open("wb") as handle:
        handle.write(MAGIC + struct.pack("<Q", length) + text)
        handle.write(payload)
    if check and (partial.stat().st_size != OUTPUT_SIZE or sha256_file(partial) != OUTPUT_SHA256):
        actual = (partial.stat().st_size, sha256_file(partial))
        partial.unlink()
        raise SystemExit(f"converted artifact does not match its pin: {actual}")
    os.replace(partial, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="extracted kokoro-ref/ directory")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    target = build(args.source, args.destination)
    print(f"{target} {OUTPUT_SIZE} {OUTPUT_SHA256}")


if __name__ == "__main__":
    main()
