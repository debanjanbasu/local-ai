"""Ternary g128 matrices in the checkpoint's Hadamard-rotated basis.

The runtime's ternary MTP head stores every matrix like the Ternary Bonsai 2
target: codes in {-1, 0, +1} and one FP16 scale per 128 input columns, applied
to the rotated activation:

  y[r] = sum_c codes[r, c] * scales[r, c // 128] * (R x)[c]

R is the target's activation rotation for the input width n (n % 1024 == 0):
per 1024-block b, (R x)_b = H_1024 (s_b * x_b) / 32 with H the natural-order
(Sylvester) Walsh-Hadamard matrix, H[i, j] = (-1)^popcount(i & j), and s the
package's `hadamard().signs(n)` (`mtp-capture signs DIR` writes them). This is
`shaders/bonsai_projection.h::bonsai_fwht_values` (non-inverse); R is
orthonormal, and Rᵀ is `capture.rs::inverse_rotate`.

A dense W maps to the rotated basis as W_r = W Rᵀ, i.e. every row is rotated
like an activation, so y = W x = W_r (R x). `fc` is two matrices to the
runtime (embedding and hidden halves, each rotated with the 5120 signs).

Mixed precision: any runtime matrix may instead be int8 per row in the same
rotated basis, y[r] = sum_c int8[r, c] * row_scales[r] * (R x)[c], with
symmetric absmax scales (row_scales = max_c |w_r[r, c]| / 127, F32) and the
same straight-through fake-quant in training. `parse_int8` reads which ones.

Run the self-checks: uv run --no-project --with torch --with safetensors \\
  --with numpy python tools/mtp_train/ternary.py --signs .amp/in/mtp/signs
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BLOCK = 1024
GROUP = 128
QUANTIZERS = ("twn", "absmean", "opt")
# "none" keeps the rotated latent dense: a diagnostic that the rotated student
# reproduces the teacher (not exportable).
DIAGNOSTIC_QUANTIZERS = (*QUANTIZERS, "none")

# Ternary matrices: trainer name -> (runtime name(s), [(column start, end, sign width)]).
MATRICES = {
    "fc.weight": [(0, 5120, 5120), (5120, 10240, 5120)],
    "layers.0.self_attn.q_proj.weight": [(0, 5120, 5120)],
    "layers.0.self_attn.k_proj.weight": [(0, 5120, 5120)],
    "layers.0.self_attn.v_proj.weight": [(0, 5120, 5120)],
    "layers.0.self_attn.o_proj.weight": [(0, 6144, 6144)],
    "layers.0.mlp.gate_proj.weight": [(0, 5120, 5120)],
    "layers.0.mlp.up_proj.weight": [(0, 5120, 5120)],
    "layers.0.mlp.down_proj.weight": [(0, 17408, 17408)],
}
# Trainer names of fc's halves as the runtime stores them.
FC_PARTS = {"fc.weight.embedding": (0, 5120), "fc.weight.hidden": (5120, 10240)}
# Module attribute of each trainer matrix name (mtp_head.MTPHead.key_map).
MODULES = {
    "fc.weight": "fc",
    "layers.0.self_attn.q_proj.weight": "q_proj",
    "layers.0.self_attn.k_proj.weight": "k_proj",
    "layers.0.self_attn.v_proj.weight": "v_proj",
    "layers.0.self_attn.o_proj.weight": "o_proj",
    "layers.0.mlp.gate_proj.weight": "gate_proj",
    "layers.0.mlp.up_proj.weight": "up_proj",
    "layers.0.mlp.down_proj.weight": "down_proj",
}

# Runtime matrix name -> (trainer matrix name, part index into MATRICES[name]).
RUNTIME = {
    "mtp.fc.weight.embedding": ("fc.weight", 0),
    "mtp.fc.weight.hidden": ("fc.weight", 1),
    **{f"mtp.{name}": (name, 0) for name in MATRICES if name != "fc.weight"},
}
# Runtime matrix shapes [rows, cols].
RUNTIME_SHAPES = {
    "mtp.fc.weight.embedding": (5120, 5120),
    "mtp.fc.weight.hidden": (5120, 5120),
    "mtp.layers.0.self_attn.q_proj.weight": (12288, 5120),
    "mtp.layers.0.self_attn.k_proj.weight": (1024, 5120),
    "mtp.layers.0.self_attn.v_proj.weight": (1024, 5120),
    "mtp.layers.0.self_attn.o_proj.weight": (5120, 6144),
    "mtp.layers.0.mlp.gate_proj.weight": (17408, 5120),
    "mtp.layers.0.mlp.up_proj.weight": (17408, 5120),
    "mtp.layers.0.mlp.down_proj.weight": (5120, 17408),
}
# Short names (and groups) accepted by parse_int8.
ALIASES = {
    "fc_e": ("mtp.fc.weight.embedding",),
    "fc_h": ("mtp.fc.weight.hidden",),
    "fc": ("mtp.fc.weight.embedding", "mtp.fc.weight.hidden"),
    "q": ("mtp.layers.0.self_attn.q_proj.weight",),
    "k": ("mtp.layers.0.self_attn.k_proj.weight",),
    "v": ("mtp.layers.0.self_attn.v_proj.weight",),
    "kv": ("mtp.layers.0.self_attn.k_proj.weight", "mtp.layers.0.self_attn.v_proj.weight"),
    "o": ("mtp.layers.0.self_attn.o_proj.weight",),
    "gate": ("mtp.layers.0.mlp.gate_proj.weight",),
    "up": ("mtp.layers.0.mlp.up_proj.weight",),
    "gateup": ("mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight"),
    "down": ("mtp.layers.0.mlp.down_proj.weight",),
    "all": tuple(RUNTIME),
}
INT8_MAX = 127
# Packed runtime bytes per weight: PTQ1 ternary (28 B per 128) vs int8 (+ F32 row scale).
TERNARY_BYTES_PER_GROUP = 28


def parse_int8(spec):
    """Sorted runtime names of the int8 matrices from a spec: comma-separated
    aliases (ALIASES) or runtime names, `none`/empty, `all`, `~X` (everything
    but X), or a JSON file {runtime name or alias: "int8" | "ternary"}."""
    if spec is None:
        return ()
    if isinstance(spec, (list, tuple)):
        spec = ",".join(spec)
    spec = str(spec).strip()
    if spec.endswith(".json"):
        import json
        config = json.loads(Path(spec).read_text())
        return parse_int8(",".join(k for k, v in config.items() if v == "int8"))
    if spec.startswith("~"):
        drop = set(parse_int8(spec[1:]))
        return tuple(sorted(n for n in RUNTIME if n not in drop))
    names = set()
    for item in filter(None, (x.strip() for x in spec.split(","))):
        if item == "none":
            continue
        if item in ALIASES:
            names.update(ALIASES[item])
        elif item in RUNTIME:
            names.add(item)
        else:
            raise SystemExit(f"unknown matrix {item!r} (aliases: {', '.join(ALIASES)})")
    return tuple(sorted(names))


def matrix_bytes(runtime, int8):
    rows, cols = RUNTIME_SHAPES[runtime]
    if int8:
        return rows * cols + 4 * rows
    return rows * cols // GROUP * TERNARY_BYTES_PER_GROUP


def head_bytes(int8_names=()):
    """Runtime matrix bytes of a mixed head (norms excluded, ~52 KB)."""
    return sum(matrix_bytes(n, n in int8_names) for n in RUNTIME)


def hadamard(n=BLOCK, dtype=torch.float32):
    """Natural-order (Sylvester) Walsh-Hadamard matrix, entries ±1."""
    i = torch.arange(n)
    bits = i[:, None] & i[None, :]
    parity = torch.zeros_like(bits)
    while bool(bits.any()):
        parity ^= bits & 1
        bits = bits >> 1
    return (1 - 2 * parity).to(dtype)


def load_signs(directory):
    """Width -> float ±1 tensor from `mtp-capture signs` output."""
    signs = {}
    for width in sorted({w for parts in MATRICES.values() for _, _, w in parts}):
        path = Path(directory, f"signs-{width}.bin")
        raw = np.fromfile(path, dtype=np.int8)
        if raw.shape != (width,) or not np.all(np.abs(raw) == 1):
            raise SystemExit(f"{path}: expected {width} ±1 bytes")
        signs[width] = torch.from_numpy(raw.astype(np.float32))
    return signs


class Rotation(nn.Module):
    """x -> R x (and Rᵀ) for one width; blockwise matmul with H / 32, which
    is exact in F32 (entries ±2^-5)."""

    def __init__(self, signs):
        super().__init__()
        assert signs.numel() % BLOCK == 0
        self.register_buffer("signs", signs.float().clone(), persistent=False)
        self.register_buffer("h", hadamard() / 32.0, persistent=False)

    def forward(self, x):
        shape = x.shape
        blocks = (x * self.signs.to(x.dtype)).reshape(*shape[:-1], -1, BLOCK)
        return (blocks @ self.h.to(x.dtype)).reshape(shape)

    def inverse(self, y):
        shape = y.shape
        blocks = y.reshape(*shape[:-1], -1, BLOCK) @ self.h.to(y.dtype)
        return blocks.reshape(shape) * self.signs.to(y.dtype)


def quantize(w, mode="twn"):
    """Per-row, per-128-column ternary codes and FP16-representable scales.

    Returns (codes as w.dtype in {-1,0,1}, scales [rows, cols/128] already
    rounded to FP16 and back), so codes * scales is the exported matrix.
      twn:     threshold 0.7 mean|w|, scale = mean |w| above it (TWN).
      absmean: scale = mean|w|, codes = round(clip(w / scale, -1, 1)) (BitNet b1.58).
      opt:     the L2-optimal ternary per group: keep the k largest |w| with
               k maximizing (sum of them)^2 / k, scale = their mean.
    """
    rows, cols = w.shape
    if mode == "none":
        return w.detach(), torch.ones(rows, cols // GROUP, device=w.device, dtype=w.dtype)
    g = w.detach().reshape(rows, cols // GROUP, GROUP)
    a = g.abs()
    if mode == "twn":
        delta = 0.7 * a.mean(-1, keepdim=True)
        mask = (a > delta).to(g.dtype)
        scale = (a * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    elif mode == "absmean":
        scale = a.mean(-1).clamp_min(1e-12)
        mask = (a / scale[..., None] >= 0.5).to(g.dtype)  # round(clip(w/s))
    elif mode == "opt":
        sorted_a, _ = a.sort(-1, descending=True)
        csum = sorted_a.cumsum(-1)
        k = torch.arange(1, GROUP + 1, device=w.device, dtype=g.dtype)
        best = (csum * csum / k).argmax(-1, keepdim=True)
        thresh = sorted_a.gather(-1, best)
        mask = (a >= thresh).to(g.dtype)
        scale = (a * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    else:
        raise ValueError(mode)
    scale = scale.to(torch.float16).to(g.dtype)
    codes = torch.sign(g) * mask * (scale[..., None] != 0)
    return codes.reshape(rows, cols), scale


def dequantize(codes, scales):
    rows, cols = codes.shape
    return (codes.reshape(rows, cols // GROUP, GROUP) * scales[..., None]).reshape(rows, cols)


def quantize_int8(w):
    """Per-row symmetric absmax int8: (codes as w.dtype in [-127, 127], F32
    row scales max|w_row| / 127); codes * scales[:, None] is the matrix."""
    w = w.detach()
    scale = (w.float().abs().amax(-1) / INT8_MAX).to(w.dtype)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(w / safe[:, None]).clamp(-INT8_MAX, INT8_MAX)
    return codes * (scale[:, None] > 0), scale


def dequantize_int8(codes, scales):
    return codes * scales[:, None]


class _TernarySTE(torch.autograd.Function):
    """F.linear(x, ternary(w)) with the straight-through gradient, recomputing
    the ternary matrix in backward instead of keeping a weight-sized copy per
    call alive for it (the memory-saving path of TernaryLinear.recompute)."""

    @staticmethod
    def forward(ctx, x, w, module):
        with torch.no_grad():
            q = module.quantize_weight(w).to(w.dtype)
        ctx.save_for_backward(x, w)
        ctx.module = module
        return F.linear(x, q)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                q = ctx.module.quantize_weight(w).to(w.dtype)
            grad_x = grad @ q
            del q
        if ctx.needs_input_grad[1]:
            grad_w = grad.reshape(-1, grad.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        return grad_x, grad_w, None


class TernaryLinear(nn.Module):
    """A Linear whose latent weight lives in the rotated basis and is ternary
    g128 (or, per column part, per-row int8) in the forward (straight-through
    gradient to the latent)."""

    # Same values and gradients; True trades a re-quantization in backward for
    # not saving the ternary matrix (set by train_ternary --recompute-ternary).
    recompute = False

    def __init__(self, latent, parts, rotations, quantizer="twn"):
        super().__init__()
        self.weight = nn.Parameter(latent)
        self.parts = parts  # [(start, end, width)]
        self.rotations = rotations  # width -> Rotation (shared, not owned)
        self.quantizer = quantizer
        self.int8 = (False,) * len(parts)  # per part: int8 instead of ternary

    def quantize_weight(self, w):
        """The forward matrix (ternary / int8 per part, dequantized)."""
        if not any(self.int8):
            return dequantize(*quantize(w, self.quantizer))
        return torch.cat([
            dequantize_int8(*quantize_int8(w[:, s:e])) if is8
            else dequantize(*quantize(w[:, s:e], self.quantizer))
            for (s, e, _), is8 in zip(self.parts, self.int8)], dim=1)

    def rotate_input(self, x):
        if len(self.parts) == 1:
            return self.rotations[str(self.parts[0][2])](x)
        return torch.cat([self.rotations[str(w)](x[..., s:e]) for s, e, w in self.parts], -1)

    def quantized(self):
        return self.quantize_weight(self.weight)

    def forward(self, x):
        w = self.weight
        if self.recompute and torch.is_grad_enabled() and w.requires_grad:
            return _TernarySTE.apply(self.rotate_input(x), w, self)
        q = self.quantized().to(w.dtype)
        if torch.is_grad_enabled() and w.requires_grad:
            q = w + (q - w).detach()  # value = ternary (to an ulp), d/dlatent = 1 (STE)
        return F.linear(self.rotate_input(x), q)

    def export(self):
        """Per part (runtime matrix): {tensor suffix: tensor}, either
        `codes` I8 + `scales` F16 (ternary) or `int8` I8 + `row_scales` F32."""
        out = []
        for (s, e, _), is8 in zip(self.parts, self.int8):
            w = self.weight[:, s:e]
            if is8:
                codes, scales = quantize_int8(w)
                out.append({"int8": codes.to(torch.int8), "row_scales": scales.float()})
            else:
                codes, scales = quantize(w, self.quantizer)
                out.append({"codes": codes.to(torch.int8), "scales": scales.to(torch.float16)})
        return out


def rotate_rows(w, parts, rotations):
    """W -> W Rᵀ: rotate every row like an activation, per column part."""
    return torch.cat(
        [rotations[str(width)](w[:, s:e]) for s, e, width in parts], dim=1)


def set_int8(head, int8=()):
    """Mark the runtime matrices `int8` (names) int8 and all others ternary."""
    int8 = set(int8)
    for runtime in int8:
        assert runtime in RUNTIME, runtime
    for name in MATRICES:
        module = getattr(head, MODULES[name])
        module.int8 = tuple(r in int8 for r, (n, _) in RUNTIME.items() if n == name)
    head.int8_matrices = tuple(sorted(int8))


def make_ternary(head, signs, quantizer="twn", latent=None, int8=(), from_dense=None):
    """Replace the head's 8 Linear modules by TernaryLinear in place. `latent`
    maps trainer names to rotated-basis weights (a QAT checkpoint); otherwise
    the dense weights are rotated. `from_dense` (runtime names) take the rotated
    dense weights even with `latent` (e.g. new int8 matrices from the teacher).
    `int8` (runtime names) are int8, the rest ternary. Returns the rotations."""
    device = head.fc.weight.device
    rotations = nn.ModuleDict({str(w): Rotation(s) for w, s in signs.items()}).to(device)
    from_dense = set(from_dense or ())
    for name, parts in MATRICES.items():
        module = getattr(head, MODULES[name])
        dense_parts = [r in from_dense for r, (n, _) in RUNTIME.items() if n == name]
        if latent is not None:
            weight = latent[name].to(device=device, dtype=torch.float32)
            if any(dense_parts):
                with torch.no_grad():
                    rotated = rotate_rows(module.weight.detach().float().to(device), parts,
                                          rotations)
                    weight = weight.clone()
                    for (s, e, _), fresh in zip(parts, dense_parts):
                        if fresh:
                            weight[:, s:e] = rotated[:, s:e]
                    del rotated
        else:
            with torch.no_grad():
                weight = rotate_rows(module.weight.detach().float(), parts, rotations)
        setattr(head, MODULES[name], TernaryLinear(weight.contiguous(), parts, rotations, quantizer))
    head.rotations = rotations
    set_int8(head, int8)
    return rotations


def check_rotation(signs_dir, device="cpu"):
    """R matches capture.rs's inverse_rotate (Rᵀ) on its test vectors, is
    orthonormal, and the blocked matmul is the explicit Sylvester H."""
    signs = load_signs(signs_dir)
    h = hadamard(dtype=torch.float64)
    assert torch.equal(h @ h, BLOCK * torch.eye(BLOCK, dtype=torch.float64))
    assert h[3, 5] == -1 and h[1, 2] == 1 and h[7, 7] == -1  # popcount(i & j)
    worst = {}
    for width, s in signs.items():
        rot = Rotation(s).to(device)
        raw = np.fromfile(Path(signs_dir, f"rotate-check-{width}.bin"), dtype=np.float32)
        x = torch.from_numpy(raw[:width].copy()).to(device)
        rust_inv = torch.from_numpy(raw[width:].copy()).to(device)
        ours_inv = rot.inverse(x)
        err_inv = float((ours_inv - rust_inv).abs().max())
        # Explicit float64 R = blockdiag(H) diag(s) / 32.
        x64 = x.cpu().double()
        ref = ((x64 * s.double()).reshape(-1, BLOCK) @ h.T / 32.0).reshape(-1)
        err_fwd = float((rot(x).cpu().double() - ref).abs().max())
        err_round = float((rot.inverse(rot(x)) - x).abs().max())
        worst[width] = (err_inv, err_fwd, err_round)
        assert err_inv < 1e-5 and err_fwd < 1e-5 and err_round < 1e-5, (width, worst[width])
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    for width, (inv, fwd, rt) in check_rotation(args.signs, args.device).items():
        print(f"width {width}: |Rᵀx - rust inverse_rotate| {inv:.2e}, "
              f"|Rx - float64 Sylvester| {fwd:.2e}, |RᵀRx - x| {rt:.2e}")
    # Quantizer sanity: codes ternary, scales FP16-exact, opt minimizes L2.
    torch.manual_seed(0)
    w = torch.randn(64, 512) * 0.02
    errs = {}
    for mode in QUANTIZERS:
        codes, scales = quantize(w, mode)
        assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
        assert torch.equal(scales, scales.half().float())
        errs[mode] = float((dequantize(codes, scales) - w).pow(2).sum() / w.pow(2).sum())
    assert errs["opt"] <= min(errs.values()) + 1e-6, errs
    codes, scales = quantize_int8(w)
    assert codes.abs().max() == INT8_MAX and torch.equal(codes, codes.round())
    errs["int8"] = float((dequantize_int8(codes, scales) - w).pow(2).sum() / w.pow(2).sum())
    assert errs["int8"] < 1e-3, errs
    assert parse_int8("fc,q") == tuple(sorted(ALIASES["fc"] + ALIASES["q"]))
    assert len(parse_int8("~down")) == len(RUNTIME) - 1
    assert head_bytes() == sum(r * c for r, c in RUNTIME_SHAPES.values()) * 28 // 128
    print("quantizer relative L2 error on N(0,1):",
          {k: round(v, 6) for k, v in errs.items()})
    print(f"all-ternary {head_bytes() / 1e6:.1f} MB, all-int8 "
          f"{head_bytes(parse_int8('all')) / 1e6:.1f} MB")
    print("OK")


if __name__ == "__main__":
    main()
