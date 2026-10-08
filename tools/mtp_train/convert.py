"""Convert MTP heads between the runtime's formats and the trainer's.

  from-artifact ARTIFACT OUT.safetensors
      dequantize a retired int8 runtime head (`mtp-head-int8-v2.bin`, stored
      codec; the engine no longer reads int8 heads) into the trainer convention: no `mtp.` prefix, direct-multiply norms (the artifact
      already holds the folded 1 + w), matrices as F32 (or --dtype bfloat16).
  to-hf IN.safetensors OUT_DIR
      write OUT_DIR/model_mtp.safetensors in the HF convention of the upstream
      BF16 teacher head: 15 `mtp.*` BF16 tensors, norms zero-centered.
      IN may use either convention; it is detected from the norm means.
  to-ternary IN OUT_DIR --signs DIR [--quantizer twn]
      write OUT_DIR/model_mtp_ternary.safetensors, the runtime's ternary g128
      head (`local-ai bonsai --export mtp-head=OUT_DIR` packs it): per matrix `<name>.codes` I8 [rows, cols] in {-1,0,1} and
      `<name>.scales` F16 [rows, cols/128] in the target's rotated basis
      (ternary.py; fc split into `mtp.fc.weight.embedding` / `.hidden`), and
      the 7 norms BF16 zero-centered. IN is a QAT checkpoint
      (train_ternary.py) or any dense head (rotated and ternarized: the PTQ
      baseline). Then checks the file: names, shapes, dtypes, codes, and
      codes * scales applied to R x in float64 against the trainer's forward.
      Mixed precision (--int8 SPEC, default the checkpoint's int8 set; see
      ternary.parse_int8): those matrices are instead `<name>.int8` I8
      [rows, cols] in [-127, 127] and `<name>.row_scales` F32 [rows], same
      rotated basis: y = sum_c int8[r, c] * row_scales[r] * (R x)[c].
  from-ternary TERNARY OUT.safetensors --signs DIR
      dense trainer-convention F32 head reconstructed from a ternary or mixed
      file (W = (codes * scales) R or (int8 * row_scales) R, norms + 1): an independent reference that
      train.py --eval-only / parity.py can run.
  inspect FILE
      print tensor names, the detected convention and the norm means.

Run with: uv run --no-project --with torch --with safetensors --with numpy python ...
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from mtp_head import NORMS, MTPHead, to_trainer_convention
from safetensors.torch import load_file, save_file

HF_SHAPES = {
    "fc.weight": (5120, 10240),
    "pre_fc_norm_embedding.weight": (5120,),
    "pre_fc_norm_hidden.weight": (5120,),
    "layers.0.input_layernorm.weight": (5120,),
    "layers.0.post_attention_layernorm.weight": (5120,),
    "layers.0.self_attn.q_proj.weight": (12288, 5120),
    "layers.0.self_attn.k_proj.weight": (1024, 5120),
    "layers.0.self_attn.v_proj.weight": (1024, 5120),
    "layers.0.self_attn.o_proj.weight": (5120, 6144),
    "layers.0.self_attn.q_norm.weight": (256,),
    "layers.0.self_attn.k_norm.weight": (256,),
    "layers.0.mlp.gate_proj.weight": (17408, 5120),
    "layers.0.mlp.up_proj.weight": (17408, 5120),
    "layers.0.mlp.down_proj.weight": (5120, 17408),
    "norm.weight": (5120,),
}

ARTIFACT_MAGIC = b"MTPQ8\0\0\0"


def read_artifact(path):
    """Sections of a stored int8 head by name (the retired int8 head format)."""
    raw = Path(path).read_bytes()
    if raw[:8] != ARTIFACT_MAGIC:
        raise SystemExit(f"{path}: not a stored int8 MTP artifact ")
    version, count = struct.unpack_from("<II", raw, 8)
    if version != 2:
        raise SystemExit(f"{path}: unsupported artifact version {version}")
    at = 8 + 8 + 16 + 64
    (names_len,) = struct.unpack_from("<I", raw, at)
    at += 4
    names = []
    end = at + names_len
    while at < end:
        length = raw[at]
        names.append(raw[at + 1 : at + 1 + length].decode())
        at += 1 + length
    sections = {}
    for name in names:
        offset, length = struct.unpack_from("<QQ", raw, at)
        at += 16
        sections[name] = raw[offset : offset + length]
    if len(sections) != count:
        raise SystemExit(f"{path}: {len(sections)} sections, header says {count}")
    return sections


def dequantize_artifact(path):
    """Trainer-convention F32 state from an int8 artifact."""
    sections = read_artifact(path)

    def matrix(section, rows):
        q = np.frombuffer(sections[f"{section}.i8"], dtype=np.int8).reshape(rows, -1)
        scales = np.frombuffer(sections[f"{section}.scales"], dtype=np.float32)
        return torch.from_numpy(q.astype(np.float32) * scales[:, None])

    state = {}
    fc_e = matrix("mtp.fc.weight.embedding", 5120)
    fc_h = matrix("mtp.fc.weight.hidden", 5120)
    state["fc.weight"] = torch.cat([fc_e, fc_h], dim=1)
    for name, shape in HF_SHAPES.items():
        if name == "fc.weight":
            continue
        if name in NORMS:
            state[name] = torch.from_numpy(
                np.frombuffer(sections[f"mtp.{name}"], dtype=np.float32).copy())
        else:
            state[name] = matrix(f"mtp.{name}", shape[0])
    for name, shape in HF_SHAPES.items():
        assert tuple(state[name].shape) == shape, (name, state[name].shape)
    return state


def to_hf(state):
    """HF-convention BF16 tensors from a trainer-convention state."""
    out = {}
    for name, shape in HF_SHAPES.items():
        tensor = state[name].float()
        if tuple(tensor.shape) != shape:
            raise SystemExit(f"{name}: shape {tuple(tensor.shape)}, expected {shape}")
        if name in NORMS:
            tensor = tensor - 1.0
        if not torch.isfinite(tensor).all():
            raise SystemExit(f"{name}: non-finite values")
        out[f"mtp.{name}"] = tensor.to(torch.bfloat16).contiguous()
    return out


def ternary_names():
    """Runtime matrix name -> (trainer matrix name, column slice)."""
    from ternary import MATRICES, RUNTIME

    names = {}
    for runtime, (name, part) in RUNTIME.items():
        start, end, _ = MATRICES[name][part]
        names[runtime] = (name, slice(start, end))
    return names


def to_ternary(args):
    from safetensors import safe_open
    from ternary import MODULES, QUANTIZERS, RUNTIME, head_bytes, load_signs, parse_int8
    from train_ternary import QAT_FORMAT, build_student

    with safe_open(args.input, framework="pt") as f:
        meta = f.metadata() or {}
    qat = meta.get("format") == QAT_FORMAT
    quantizer = args.quantizer or meta.get("quantizer", "twn")
    if quantizer not in QUANTIZERS:
        raise SystemExit(f"quantizer {quantizer!r} is not exportable")
    signs = load_signs(args.signs)
    int8 = parse_int8(args.int8) if args.int8 is not None else None
    if qat and int8 is not None and set(int8) != set(parse_int8(meta.get("int8", ""))):
        raise SystemExit(f"--int8 differs from {args.input}'s int8 set "
                         f"{meta.get('int8', '')!r}: export what was trained")
    # eps / RoPE base do not affect the matrices or norms being exported.
    head = build_student(None if qat else args.input, signs, quantizer, 1e-6, 1e7,
                         init=args.input if qat else None, device=args.device, int8=int8)
    head.eval()
    int8 = head.int8_matrices
    print(f"{args.input}: {'QAT checkpoint' if qat else 'dense head (PTQ)'}, "
          f"quantizer {quantizer}, int8 [{','.join(int8)}], device {args.device}")
    out = {}
    exported = {}
    with torch.no_grad():
        for runtime, (name, part) in RUNTIME.items():
            module = getattr(head, MODULES[name])
            if name not in exported:
                exported[name] = module.export()
            for suffix, tensor in exported[name][part].items():
                out[f"{runtime}.{suffix}"] = tensor.cpu().contiguous()
        key_map = MTPHead.key_map()
        for name in NORMS:
            tensor = getattr(head, key_map[name]).detach().float().cpu() - 1.0
            out[f"mtp.{name}"] = tensor.to(torch.bfloat16).contiguous()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model_mtp_ternary.safetensors"
    metadata = {
        "format": "mtp-ternary-g128-v1" if not int8 else "mtp-mixed-ternary-g128-int8-v1",
        "quantizer": quantizer,
        "basis": "rotated: y = sum_c codes*scales[c//128]*(R x)[c], R = blockwise H_1024 diag(s)/32",
        "source": str(args.input)}
    if int8:
        metadata["int8"] = ",".join(int8)
        metadata["int8_basis"] = "rotated: y = sum_c int8[r,c]*row_scales[r]*(R x)[c]"
    save_file(out, str(path), metadata=metadata)
    print(f"wrote {path} ({len(out)} tensors, matrices {head_bytes(int8) / 1e6:.1f} MB packed)")
    check_ternary(path, head, signs, args.device, int8)


def check_ternary(path, head, signs, device, int8=()):
    """The written file honours the contract and reproduces the trainer."""
    from ternary import MODULES, hadamard

    weights = load_file(str(path))
    h64 = hadamard(dtype=torch.float64)

    def rotate64(x, width):
        s = signs[width].double()
        return ((x * s).reshape(x.shape[0], -1, 1024) @ h64 / 32.0).reshape(x.shape)

    expected = {f"mtp.{n}" for n in NORMS}
    for runtime in ternary_names():
        if runtime in int8:
            expected |= {f"{runtime}.int8", f"{runtime}.row_scales"}
        else:
            expected |= {f"{runtime}.codes", f"{runtime}.scales"}
    if set(weights) != expected:
        raise SystemExit(f"tensor names differ: {sorted(set(weights) ^ expected)}")
    for name in NORMS:
        t = weights[f"mtp.{name}"]
        assert t.dtype == torch.bfloat16 and tuple(t.shape) == HF_SHAPES[name], name
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for name, module_name in MODULES.items():
        module = getattr(head, module_name)
        rows, cols = HF_SHAPES[name]
        x = torch.randn(4, cols, generator=gen, dtype=torch.float64)
        with torch.no_grad():
            y_train = module(x.float().to(device)).cpu().double()
        y_ref = torch.zeros(4, rows, dtype=torch.float64)
        for runtime, (src, part) in ternary_names().items():
            if src != name:
                continue
            width = part.stop - part.start
            if runtime in int8:
                q = weights[f"{runtime}.int8"]
                row_scales = weights[f"{runtime}.row_scales"]
                assert q.dtype == torch.int8 and row_scales.dtype == torch.float32, runtime
                assert tuple(q.shape) == (rows, width), runtime
                assert tuple(row_scales.shape) == (rows,), runtime
                assert int(q.abs().max()) <= 127, runtime
                assert torch.isfinite(row_scales).all() and (row_scales >= 0).all(), runtime
                w = q.double() * row_scales.double()[:, None]
            else:
                codes = weights[f"{runtime}.codes"]
                scales = weights[f"{runtime}.scales"]
                assert codes.dtype == torch.int8 and scales.dtype == torch.float16
                assert tuple(codes.shape) == (rows, width), runtime
                assert tuple(scales.shape) == (rows, width // 128), runtime
                assert set(codes.unique().tolist()) <= {-1, 0, 1}, runtime
                assert torch.isfinite(scales.float()).all(), runtime
                w = (codes.double().reshape(rows, -1, 128) * scales.double()[..., None]
                     ).reshape(rows, width)
            y_ref += rotate64(x[:, part], width) @ w.T
        err = float((y_train - y_ref).abs().max() / y_ref.abs().max())
        worst = max(worst, err)
        print(f"  {name}: max |trainer - float64 codes*scales*R x| / max|y| = {err:.2e}")
    if worst > 1e-5:
        raise SystemExit(f"export check FAILED: {worst:.2e}")
    print(f"export check OK (worst relative error {worst:.2e})")


def from_ternary(args):
    from ternary import Rotation, dequantize, load_signs

    signs = load_signs(args.signs)
    weights = load_file(args.input)
    names = ternary_names()
    state = {}
    for runtime, (name, _) in names.items():
        if f"{runtime}.int8" in weights:  # mixed head: per-row int8
            w_r = weights[f"{runtime}.int8"].float() * weights[f"{runtime}.row_scales"][:, None]
        else:
            w_r = dequantize(weights[f"{runtime}.codes"].float(),
                             weights[f"{runtime}.scales"].float())
        # W = W_r R: every row goes through Rᵀ.
        dense = Rotation(signs[w_r.shape[1]]).double().inverse(w_r.double()).float()
        state.setdefault(name, []).append(dense)
    out = {name: torch.cat(parts, dim=1).contiguous() for name, parts in state.items()}
    for name in NORMS:
        out[name] = (weights[f"mtp.{name}"].float() + 1.0).contiguous()
    for name, shape in HF_SHAPES.items():
        assert tuple(out[name].shape) == shape, name
    save_file(out, args.out)
    print(f"wrote {args.out} (dense trainer convention)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("from-artifact")
    a.add_argument("artifact")
    a.add_argument("out")
    a.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    b = sub.add_parser("to-hf")
    b.add_argument("input")
    b.add_argument("out_dir")
    b.add_argument("--convention", default="auto", choices=["auto", "hf", "mlx"])
    t = sub.add_parser("to-ternary")
    t.add_argument("input")
    t.add_argument("out_dir")
    t.add_argument("--signs", required=True, help="`mtp-capture signs` dir")
    t.add_argument("--quantizer", help="default: the checkpoint's, else twn")
    t.add_argument("--int8", help="int8 matrices (ternary.parse_int8 spec or config JSON); "
                   "default: the checkpoint's int8 set, else none")
    t.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                   help="quantize where training ran: MPS and CPU reductions differ by ~1e-8 codes")
    f = sub.add_parser("from-ternary")
    f.add_argument("input")
    f.add_argument("out")
    f.add_argument("--signs", required=True)
    c = sub.add_parser("inspect")
    c.add_argument("input")
    args = ap.parse_args()

    if args.command == "from-artifact":
        state = dequantize_artifact(args.artifact)
        dtype = getattr(torch, args.dtype)
        save_file({k: (v if k in NORMS else v.to(dtype)).contiguous()
                   for k, v in state.items()}, args.out)
        print(f"wrote {args.out}; norm means "
              f"{ {k: round(float(state[k].mean()), 4) for k in NORMS} }")
    elif args.command == "to-hf":
        state, detected = to_trainer_convention(load_file(args.input), args.convention)
        print(f"{args.input}: {detected}")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = to_hf(state)
        save_file(out, str(out_dir / "model_mtp.safetensors"))
        print(f"wrote {out_dir / 'model_mtp.safetensors'} ({len(out)} tensors)")
    elif args.command == "to-ternary":
        to_ternary(args)
    elif args.command == "from-ternary":
        from_ternary(args)
    else:
        weights = load_file(args.input)
        print(sorted(weights)[:3], "...", len(weights), "tensors")
        _, detected = to_trainer_convention(weights)
        print(detected)
        assert set(MTPHead.key_map()) == set(HF_SHAPES)


if __name__ == "__main__":
    main()
