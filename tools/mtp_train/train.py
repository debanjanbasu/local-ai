"""Distill/fine-tune the Bonsai 2 MTP draft head on the PTQ1 target's own outputs.

Adapted from xkm/qwen3.8-27b-mtp-head-retrained (train/train.py): same
chain-faithful multi-step objective, retargeted at this runtime.

  step 1: rows f1_p = fuse(embed(tok_p), target_hidden_{p-1}) through the head
          layer with plain causal attention: the committed-history rows the
          runtime keeps in the head KV. Label: target argmax of row p, plus
          soft top-k KD.
  step j>=2 at base i (absolute position i + j - 1):
          query = fuse(embed(target argmax of row i + j - 2), head output of
          step j-1 at base i). It attends step-1 rows 0..i plus its own chain
          rows and itself (mtp_head.chain_mask) — the KV a runtime draft step
          sees (local-engine/src/bonsai_native/speculation.rs). Supervised only
          where the document followed the target argmax along the whole chain.

Differences from xkm: runs on MPS (fp32 head, bf16 frozen tables); captured
top-k values are logits, so KD targets are softmax(top-k logits) by default
(`--kd-target clamp` restores xkm's clamp-and-renormalize); optional int8
per-row fake quantization (`--qat-int8`) matching the retired int8 runtime head
(fc halves quantized separately); MLX 4-bit QAT dropped; the step-j lm_head is
evaluated only on supervised rows.

Run: uv run --no-project --with torch --with safetensors --with numpy \\
       python tools/mtp_train/train.py --capture DIR --tables DIR --head H --out DIR
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from data import DocBatcher
from mtp_head import NORMS, MTPHead, chain_mask, load_head, run_layer
from safetensors.torch import load_file, save_file


def int8_fq(w, split=None):
    """Symmetric per-row int8 fake quantization, as the retired int8 runtime
    head's quantize_rows did (from the BF16 the converter writes). `split` quantizes column halves
    independently, as the runtime splits fc."""
    if split:
        return torch.cat([int8_fq(part) for part in w.split(split, dim=1)], dim=1)
    w = w.to(torch.bfloat16).float()
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.clamp(torch.round(w / scale), -127, 127) * scale


class Int8Param(torch.nn.Module):
    def __init__(self, split=None):
        super().__init__()
        self.split = split

    def forward(self, w):
        with torch.no_grad():
            fq = int8_fq(w.float(), self.split).to(w.dtype)
        return w + (fq - w.detach())  # value = fq, d/dw = 1 (STE)


def enable_qat(head):
    import torch.nn.utils.parametrize as P

    for name, mod in head.named_modules():
        if isinstance(mod, torch.nn.Linear):
            split = mod.weight.shape[1] // 2 if name == "fc" else None
            P.register_parametrization(mod, "weight", Int8Param(split))


def plain_state(head):
    """Module parameters by module name, with parametrized (fake-quant) weights
    resolved to their quantized values."""
    state = {k: v for k, v in head.state_dict().items() if "parametrizations" not in k}
    for name, mod in head.named_modules():
        if isinstance(mod, torch.nn.Linear):
            state[f"{name}.weight"] = mod.weight
    return state


def save_head(head, path):
    """Trainer convention (no `mtp.` prefix, direct-multiply norms): matrices
    BF16, norms F32 so their 1 + w values keep full precision."""
    state = plain_state(head)
    out = {}
    for src, dst in MTPHead.key_map().items():
        tensor = state[dst].detach().float().cpu()
        out[src] = (tensor if src in NORMS else tensor.to(torch.bfloat16)).contiguous()
    save_file(out, path)
    print(f"saved {path}", flush=True)


def topk_target(top_logits, mode):
    if mode == "softmax":
        return torch.softmax(top_logits.float(), dim=-1)
    p = top_logits.float().clamp_min(0)
    return p / p.sum(-1, keepdim=True).clamp_min(1e-9)


def soft_kd_loss(logq, top_ids, target):
    """CE of the head's log-distribution against the captured top-k target."""
    return -(target * torch.gather(logq, -1, top_ids)).sum(-1)


def memory_gb(device):
    if device == "mps":
        return (f"{torch.mps.current_allocated_memory() / 2**30:.1f}GB allocated, "
                f"{torch.mps.driver_allocated_memory() / 2**30:.1f}GB driver")
    if device == "cuda":
        return f"{torch.cuda.max_memory_allocated() / 2**30:.1f}GB peak"
    return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="dir of shard-*.bin")
    ap.add_argument("--tables", required=True, help="`mtp-capture tables` dir")
    ap.add_argument("--head", required=True, help="initial head safetensors (HF or MLX names)")
    ap.add_argument("--head-convention", default="auto", choices=["auto", "hf", "mlx"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=3, help="runtime DEFAULT_MTP_DEPTH is 3")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=0, help="cosine horizon (0 = constant)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-target", default="softmax", choices=["softmax", "clamp"])
    ap.add_argument("--depth-weights", default="1,1.5,2,2")
    ap.add_argument("--band-weight", type=float, default=0.0,
                    help="extra weight on labels whose target logit margin is in (0.5, 2.0)")
    ap.add_argument("--hinge-weight", type=float, default=0.0)
    ap.add_argument("--hinge-gamma", type=float, default=1.0)
    ap.add_argument("--qat-int8", action="store_true")
    ap.add_argument("--table-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=30)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    meta = json.loads(Path(args.tables, "meta.json").read_text())
    head, detected = load_head(
        args.head, eps=meta["rms_norm_eps"], rope_base=meta["rope_theta"],
        convention=args.head_convention)
    print(f"head {args.head}: {detected}", flush=True)
    head = head.to(device)
    if args.qat_int8:
        enable_qat(head)

    table_dtype = getattr(torch, args.table_dtype)
    embed_w = load_file(os.path.join(args.tables, "embed_tokens.safetensors"))["weight"]
    embed_w = embed_w.to(device=device, dtype=table_dtype)
    lmhead_w = load_file(os.path.join(args.tables, "lm_head.safetensors"))["weight"]
    lmhead_w = lmhead_w.to(device=device, dtype=table_dtype)

    def embed(ids):
        return F.embedding(ids, embed_w).float()

    def lm(h):
        return F.linear(h.to(table_dtype), lmhead_w)

    depth_w = [float(x) for x in args.depth_weights.split(",")]
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)

    def lr_at(step):
        if args.warmup and step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        if args.total_steps <= 0:
            return args.lr
        t = min(1.0, (step - args.warmup) / max(1, args.total_steps - args.warmup))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    batcher = DocBatcher(args.capture, window=args.window, batch=args.batch, seed=args.seed)

    def forward_chain(tokens, top_ids, top_logits, hidden, positions, want_stats=False):
        B, L = tokens.shape
        argmax = top_ids[:, :, 0]
        # match[p]: the document's token p + 1 is the target's argmax of row p
        match = torch.zeros(B, L, dtype=torch.bool, device=tokens.device)
        match[:, :-1] = tokens[:, 1:] == argmax[:, :-1]
        target = topk_target(top_logits, args.kd_target)
        margin = (top_logits[:, :, 0] - top_logits[:, :, 1]).float()

        losses = []
        stats = {}
        fused1 = head.fuse(embed(tokens), hidden)
        out1, k1, v1 = run_layer(head, fused1, positions)
        logits1 = lm(out1)
        # One F32 log-softmax serves both the hard CE and the KD term.
        logq1 = torch.log_softmax(logits1.float(), dim=-1)
        hard1 = -torch.gather(logq1, -1, argmax.unsqueeze(-1)).squeeze(-1)
        kd1 = soft_kd_loss(logq1, top_ids, target)
        del logq1
        band = ((margin > 0.5) & (margin < 2.0)).float()
        w1 = 1.0 + args.band_weight * band
        losses.append(depth_w[0] * (w1 * ((1 - args.kd_weight) * hard1
                                          + args.kd_weight * kd1)).mean())
        if args.hinge_weight > 0:
            zf = logits1.float()
            zt = zf.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)
            zother = zf.scatter(-1, argmax.unsqueeze(-1), float("-inf")).max(-1).values
            losses.append(args.hinge_weight
                          * (w1 * F.relu(args.hinge_gamma - (zt - zother))).mean())
        if want_stats:
            stats["acc1"] = (logits1.argmax(-1) == argmax).float().mean().item()
        del logits1

        prev_out = out1
        prev_kv = [(k1, v1)]
        valid = match.clone()
        for step in range(2, args.depth + 1):
            offset = step - 1
            token_in = torch.zeros_like(tokens)
            token_in[:, : L - offset + 1] = argmax[:, offset - 1:]
            label = torch.full_like(tokens, -100)
            label[:, : L - offset] = argmax[:, offset:]
            if step > 2:
                valid = valid & torch.roll(match, shifts=-(step - 2), dims=1)
                valid[:, L - (step - 2):] = False
            label = torch.where(valid, label, torch.full_like(label, -100))
            label[:, L - offset:] = -100

            fused_j = head.fuse(embed(token_in), prev_out)
            k_hist = torch.cat([kv[0] for kv in prev_kv], dim=2)
            v_hist = torch.cat([kv[1] for kv in prev_kv], dim=2)
            out_j, kj, vj = run_layer(
                head, fused_j, positions + offset, kv_extra=(k_hist, v_hist),
                attn_mask=chain_mask(L, step, tokens.device))
            keep = label != -100
            n_valid = keep.sum().clamp_min(1)
            # Only supervised rows reach the vocabulary projection.
            logits_j = lm(out_j[keep])
            hard = F.cross_entropy(logits_j.float(), label[keep], reduction="none")
            wj = torch.ones_like(hard)
            if args.band_weight > 0:
                mj = torch.zeros_like(margin)
                mj[:, : L - offset] = margin[:, offset:]
                wj = 1.0 + args.band_weight * ((mj > 0.5) & (mj < 2.0)).float()[keep]
            losses.append(depth_w[min(step - 1, len(depth_w) - 1)]
                          * (wj * hard).sum() / n_valid)
            if want_stats:
                ok = (logits_j.argmax(-1) == label[keep]).sum()
                stats[f"acc{step}"] = (ok / n_valid).item()
                stats[f"n{step}"] = int(keep.sum().item())
            prev_out = out_j
            prev_kv.append((kj, vj))
        return sum(losses), stats

    def evaluate(max_batches):
        head.eval()
        agg = {}
        with torch.no_grad():
            for bi, batch in enumerate(batcher.val_batches()):
                if bi >= max_batches:
                    break
                tokens, top_ids, top_logits, hidden, positions = (
                    t.to(device) for t in batch)
                _, stats = forward_chain(tokens, top_ids, top_logits, hidden.float(),
                                         positions, want_stats=True)
                for key, value in stats.items():
                    agg.setdefault(key, []).append(value)
        line = " ".join(f"{k}={sum(v) / len(v):.4f}" for k, v in sorted(agg.items())
                        if k.startswith("acc"))
        print(f"EVAL {line}", flush=True)
        head.train()

    if args.eval_only:
        evaluate(args.eval_batches)
        return

    os.makedirs(args.out, exist_ok=True)
    step_count = 0
    t0 = time.time()
    last = t0
    evaluate(args.eval_batches)
    done = False
    for _ in range(args.epochs):
        for batch in batcher.train_batches():
            tokens, top_ids, top_logits, hidden, positions = (t.to(device) for t in batch)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step_count)
            loss, _ = forward_chain(tokens, top_ids, top_logits, hidden.float(), positions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            if device == "mps":
                # The supervised-row count varies per step, so cached blocks
                # never get reused and the driver pool grows past RAM.
                torch.mps.empty_cache()
            step_count += 1
            if step_count % args.log_every == 0:
                if device == "mps":
                    torch.mps.synchronize()
                now = time.time()
                print(f"step {step_count} loss {loss.item():.4f} "
                      f"{(now - last) / args.log_every:.2f}s/step "
                      f"lr {lr_at(step_count):.2e} mem {memory_gb(device)}", flush=True)
                last = now
            if step_count % args.eval_every == 0:
                evaluate(args.eval_batches)
            if step_count % args.save_every == 0:
                save_head(head, f"{args.out}/head-step{step_count}.safetensors")
            if args.max_steps and step_count >= args.max_steps:
                done = True
                break
        if done:
            break
    evaluate(args.eval_batches)
    save_head(head, f"{args.out}/head-final.safetensors")
    print(f"{step_count} steps in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
