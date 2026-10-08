"""PyTorch port of the Qwen3.5/3.8 one-layer MTP ("nextn") draft head.

Adapted from xkm/qwen3.8-27b-mtp-head-retrained (train/mtp_head.py). The math
is the runtime's (local-engine/src/bonsai_mtp/head/step.rs):

  fuse:  fc(concat(RMSNorm_e(embed(tok_p)), RMSNorm_h(hidden_{p-1}))) -> 5120
  layer: input_ln -> gated full attention -> +res -> post_ln -> SiLU MLP -> +res
  out:   norm -> lm_head (frozen)

Basis: `embed` and `lm_head` are the tables `mtp-capture tables` exports, i.e.
the PTQ1 rows with the checkpoint's Hadamard rotation undone, and `hidden` is
the target's output-normalized final hidden, which the runtime keeps unrotated.
So the whole head runs in the plain Qwen basis, as the retired int8 runtime head
did.

Attention:
  q_proj [12288,5120]: per-head INTERLEAVED [q(256) | gate(256)] x 24 heads
  k/v_proj [1024,5120]: 4 KV heads x 256
  q_norm/k_norm: RMSNorm(256) per head BEFORE RoPE
  RoPE: first 64 dims, rotate-half pairing (i, i+32), base from the GGUF (1e7)
  SDPA scale 256^-0.5, GQA 24/4, causal; out = o_proj(attn * sigmoid(gate))

Weights are held in the trainer ("MLX") convention: tensor names without the
`mtp.` prefix and RMSNorm multiplying by the stored weight. The HF checkpoint
convention (`mtp.` prefix, zero-centered norms, loader adds 1) is converted on
load and by convert.py.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

HIDDEN = 5120
N_HEADS = 24
N_KV = 4
HEAD_DIM = 256
ROPE_DIMS = 64
INTERMEDIATE = 17408
ROPE_BASE = 10_000_000.0  # overridden from tables meta.json
EPS = 1e-6  # overridden from tables meta.json

NORMS = (
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "norm.weight",
)


def rms_norm(x, weight, eps):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


def rope_rotate(x, positions, base):
    """Partial RoPE on the first ROPE_DIMS dims. x: [B, H, L, HEAD_DIM],
    positions: [L] or [B, L] absolute positions; rotate-half pairing."""
    half = ROPE_DIMS // 2
    freqs = torch.exp(
        -math.log(base) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )
    if positions.dim() == 1:
        positions = positions[None]
    angles = positions.float()[:, :, None] * freqs[None, None, :]
    cos = angles.cos()[:, None]
    sin = angles.sin()[:, None]
    x1 = x[..., :half].float()
    x2 = x[..., half:ROPE_DIMS].float()
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)
    return torch.cat([rotated, x[..., ROPE_DIMS:]], dim=-1)


class MTPHead(nn.Module):
    def __init__(self, eps=EPS, rope_base=ROPE_BASE):
        super().__init__()
        self.eps = eps
        self.rope_base = rope_base
        self.fc = nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.pre_fc_norm_hidden = nn.Parameter(torch.ones(HIDDEN))
        self.pre_fc_norm_embedding = nn.Parameter(torch.ones(HIDDEN))
        self.input_layernorm = nn.Parameter(torch.ones(HIDDEN))
        self.post_attention_layernorm = nn.Parameter(torch.ones(HIDDEN))
        self.norm = nn.Parameter(torch.ones(HIDDEN))
        self.q_norm = nn.Parameter(torch.ones(HEAD_DIM))
        self.k_norm = nn.Parameter(torch.ones(HEAD_DIM))
        self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM * 2, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_KV * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, N_KV * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    @staticmethod
    def key_map():
        """checkpoint name (trainer convention) -> module parameter name"""
        return {
            "fc.weight": "fc.weight",
            "pre_fc_norm_hidden.weight": "pre_fc_norm_hidden",
            "pre_fc_norm_embedding.weight": "pre_fc_norm_embedding",
            "norm.weight": "norm",
            "layers.0.input_layernorm.weight": "input_layernorm",
            "layers.0.post_attention_layernorm.weight": "post_attention_layernorm",
            "layers.0.self_attn.q_norm.weight": "q_norm",
            "layers.0.self_attn.k_norm.weight": "k_norm",
            "layers.0.self_attn.q_proj.weight": "q_proj.weight",
            "layers.0.self_attn.k_proj.weight": "k_proj.weight",
            "layers.0.self_attn.v_proj.weight": "v_proj.weight",
            "layers.0.self_attn.o_proj.weight": "o_proj.weight",
            "layers.0.mlp.gate_proj.weight": "gate_proj.weight",
            "layers.0.mlp.up_proj.weight": "up_proj.weight",
            "layers.0.mlp.down_proj.weight": "down_proj.weight",
        }

    def fuse(self, embeds, hidden):
        e = rms_norm(embeds, self.pre_fc_norm_embedding, self.eps)
        h = rms_norm(hidden, self.pre_fc_norm_hidden, self.eps)
        return self.fc(torch.cat([e, h], dim=-1))


def head_qkv(head, x, positions):
    """Project rows to (q, k, v, gate) with per-head norms and RoPE applied."""
    B, L, _ = x.shape
    qp = head.q_proj(x).view(B, L, N_HEADS, 2 * HEAD_DIM)
    q, gate = qp.split(HEAD_DIM, dim=-1)
    gate = gate.reshape(B, L, N_HEADS * HEAD_DIM)
    k = head.k_proj(x).view(B, L, N_KV, HEAD_DIM)
    v = head.v_proj(x).view(B, L, N_KV, HEAD_DIM).transpose(1, 2)
    q = rms_norm(q, head.q_norm, head.eps).transpose(1, 2)
    k = rms_norm(k, head.k_norm, head.eps).transpose(1, 2)
    q = rope_rotate(q, positions, head.rope_base)
    k = rope_rotate(k, positions, head.rope_base)
    return q, k, v, gate


def repeat_kv(x):
    return x.repeat_interleave(N_HEADS // N_KV, dim=1)


def run_layer(head, fused, positions, kv_extra=None, attn_mask=None):
    """One MTP decoder layer over `fused` rows.

    kv_extra: optional (k_hist, v_hist) each [B, n_kv, Lh, hd] prepended to
    this pass's own K/V. attn_mask: [Lq, Lh + Lq] boolean (True = attend) or
    None for plain causal within the pass.
    Returns (normalized output, k_own, v_own); the output is what the runtime
    calls `predicted`, the hidden a deeper draft step chains.
    """
    x = rms_norm(fused, head.input_layernorm, head.eps)
    q, k_own, v_own, gate = head_qkv(head, x, positions)
    if kv_extra is not None:
        k = torch.cat([kv_extra[0], k_own], dim=2)
        v = torch.cat([kv_extra[1], v_own], dim=2)
    else:
        k, v = k_own, v_own
    # GQA is expanded explicitly: MPS SDPA has no `enable_gqa` fast path.
    out = F.scaled_dot_product_attention(
        q,
        repeat_kv(k),
        repeat_kv(v),
        attn_mask=attn_mask,
        is_causal=attn_mask is None,
        scale=1.0 / math.sqrt(HEAD_DIM),
    )
    out = out.transpose(1, 2).reshape(fused.shape[0], fused.shape[1], -1)
    attn = head.o_proj(out * torch.sigmoid(gate))
    h = fused + attn
    m = rms_norm(h, head.post_attention_layernorm, head.eps)
    layer_out = h + head.down_proj(F.silu(head.gate_proj(m)) * head.up_proj(m))
    return rms_norm(layer_out, head.norm, head.eps), k_own, v_own


def chain_mask(L, step, device):
    """Attention mask for chain step `step` >= 2 (1-based, as in xkm).

    Query at window position i (draft base i, absolute position i + step - 1)
    attends step-1 rows 0..i — the committed history the runtime keeps in the
    head KV, including the base row the first draft wrote — plus its own chain's
    rows for steps 2..step-1 and itself. That is exactly the KV a runtime draft
    step sees (speculation.rs: committed rows, then one row per earlier draft).
    KV layout: [step1 rows (L)] + [step2 rows (L)] + ... + [step rows (L)].
    """
    total = step * L
    mask = torch.zeros(L, total, dtype=torch.bool, device=device)
    base = torch.arange(L, device=device)
    mask[:, :L] = base[None, :] <= base[:, None]
    eye = torch.eye(L, dtype=torch.bool, device=device)
    for s in range(1, step):
        mask[:, s * L : (s + 1) * L] = eye
    return mask


def norm_means(state):
    """Mean of each WIDTH-sized norm: ~1 for direct-multiply weights, ~0 for
    zero-centered (HF Qwen3.5) weights."""
    return {
        name: float(state[name].float().mean())
        for name in NORMS
        if state[name].numel() == HIDDEN
    }


def to_trainer_convention(weights, convention="auto"):
    """Return (state, detected) in the trainer convention from either an HF
    (`mtp.`-prefixed, zero-centered norms) or an MLX-style checkpoint.

    `auto` decides from the numbers, not the names: the mean of the 5120-wide
    norms is ~0 for zero-centered weights and ~1 for direct multipliers. A name
    prefix that disagrees with the numbers is reported and the numbers win."""
    prefixed = any(name.startswith("mtp.") for name in weights)
    state = {
        (name.removeprefix("mtp.")): tensor
        for name, tensor in weights.items()
    }
    missing = set(MTPHead.key_map()) - set(state)
    if missing:
        raise ValueError(f"head is missing tensors: {sorted(missing)}")
    means = norm_means(state)
    average = sum(means.values()) / len(means)
    numeric = "hf" if average < 0.5 else "mlx"
    if convention == "auto":
        convention = numeric
    if (convention == "hf") != prefixed or convention != numeric:
        print(
            f"note: head names {'have' if prefixed else 'lack'} the mtp. prefix, "
            f"norm mean {average:.3f} suggests {numeric}; using {convention}",
            flush=True,
        )
    if convention == "hf":
        state = {
            name: (tensor.float() + 1.0 if name in NORMS else tensor)
            for name, tensor in state.items()
        }
    return {name: state[name] for name in MTPHead.key_map()}, {
        "convention": convention,
        "prefixed": prefixed,
        "norm_means": means,
    }


def load_head(path, eps=EPS, rope_base=ROPE_BASE, convention="auto"):
    from safetensors.torch import load_file

    state, detected = to_trainer_convention(load_file(path), convention)
    head = MTPHead(eps=eps, rope_base=rope_base)
    head.load_state_dict(
        {dst: state[src].float() for src, dst in MTPHead.key_map().items()}, strict=True
    )
    return head, detected
