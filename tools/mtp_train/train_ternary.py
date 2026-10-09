"""Quantization-aware training of a ternary g128 MTP head, distilled from a
dense teacher head (the community BF16 head; the engine's shipped head is the
result, `mtp-head-ptq1-v1.bin`).

The shipped head's live behaviour is what we want to keep: tuning it towards
the target's corpus labels raised held-out accuracy but lowered live acceptance,
and a naive ternarization lost ~11 points of acceptance. So the student starts
as the teacher's weights in the target's rotated basis (ternary.py), runs
ternary g128 in the forward with a straight-through gradient, and is trained to
match the frozen teacher's draft distributions at every chain step:

  loss = sum_j depth_w[j] * (kl_weight * KL(teacher_j || student_j)
                             + target_weight * target_j)

target_j is train.py's term (step 1: (1 - kd) CE(target argmax) + kd soft top-k;
steps >= 2: CE of the target argmax).

Chains (train.py, the runtime's speculation): step 1 runs over the committed
rows; step j >= 2 fuses the embedding of the target argmax of row i + j - 2
with the previous step's head output. Teacher and student each chain their
OWN hidden output, as each would at runtime, and both are fed the same tokens:
the target argmax, i.e. the draft path that was accepted so far — step j only
matters live when every earlier draft was accepted, so that is the input it
sees whenever it counts. KL is applied where the chain is reachable (step 1:
all rows; j >= 2: rows where the document followed the target argmax along the
chain, the same rows the target CE uses), which also keeps the 248K-wide
softmaxes small.

Metrics on held-out shards: per-step top-1 agreement with the teacher (agreeN),
target accuracy (accN) and the teacher's own target accuracy (taccN), KL.

Checkpoints (`qat-*.safetensors`) hold the latent rotated-basis matrices (F32)
and the trainer-convention norms; `convert.py to-ternary` exports them.

Mixed precision (--int8 q,down,...; ternary.parse_int8): those runtime matrices
are per-row int8 (absmax, STE fake-quant) in the same rotated basis. Matrices
that are int8 now but were not in the --init checkpoint start from the rotated
teacher; the checkpoint metadata records the int8 set. --sweep CONFIGS.json
(with --eval-only) evaluates a list of [{"label": ..., "int8": spec}] configs,
int8 matrices taken from the teacher and ternary ones from --init, without
reloading anything.

Run: uv run --no-project --with torch --with safetensors --with numpy \\
       python tools/mtp_train/train_ternary.py --capture DIR --tables DIR \\
       --teacher HEAD --signs DIR --out DIR [--init qat-ckpt] [--eval-only]
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from data import DocBatcher
from mtp_head import NORMS, MTPHead, chain_mask, load_head, run_layer
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from ternary import (
    DIAGNOSTIC_QUANTIZERS,
    MATRICES,
    MODULES,
    RUNTIME,
    TernaryLinear,
    head_bytes,
    load_signs,
    make_ternary,
    parse_int8,
    rotate_rows,
    set_int8,
)

QAT_FORMAT = "mtp-qat-rotated-v1"


def load_qat(path):
    """(latent matrices, norms, metadata) of a QAT checkpoint."""
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    if meta.get("format") != QAT_FORMAT:
        raise SystemExit(f"{path}: not a {QAT_FORMAT} checkpoint")
    state = load_file(path)
    return ({k: state[k] for k in MATRICES}, {k: state[k] for k in NORMS}, meta)


def build_student(teacher_path, signs, quantizer, eps, rope_base, init=None, device="cpu",
                  int8=None):
    """Ternary / mixed student: the teacher's weights rotated, or a QAT
    checkpoint (`init`, which then supplies every weight; `teacher_path` may be
    None). `int8`: runtime names that are int8 (default: the checkpoint's);
    int8 matrices the checkpoint holds as ternary start from the teacher."""
    init_int8 = ()
    if init:
        latent, norms, meta = load_qat(init)
        init_int8 = parse_int8(meta.get("int8", ""))
    int8 = init_int8 if int8 is None else tuple(int8)
    from_dense = sorted(set(int8) - set(init_int8)) if init else []
    if init and not from_dense:
        with torch.device("meta"):
            head = MTPHead(eps=eps, rope_base=rope_base)
        head = head.to_empty(device=device)
    else:
        if teacher_path is None:
            raise SystemExit(f"int8 {from_dense} not int8 in {init}: needs the teacher")
        head, _ = load_head(teacher_path, eps=eps, rope_base=rope_base)
        head = head.to(device)
    if not init:
        latent = None
    else:
        if meta.get("quantizer", quantizer) != quantizer:
            print(f"note: {init} was trained with {meta['quantizer']}, using {quantizer}")
        key_map = MTPHead.key_map()
        with torch.no_grad():
            for name, tensor in norms.items():
                getattr(head, key_map[name]).copy_(tensor.float())
        if from_dense:
            print(f"int8 from the rotated teacher: {', '.join(from_dense)}", flush=True)
    make_ternary(head, signs, quantizer, latent, int8=int8, from_dense=from_dense)
    return head


def save_qat(head, path, quantizer, extra=None):
    key_map = MTPHead.key_map()
    out = {}
    for name in MATRICES:
        out[name] = getattr(head, MODULES[name]).weight.detach().float().cpu().contiguous()
    for name in NORMS:
        out[name] = getattr(head, key_map[name]).detach().float().cpu().contiguous()
    meta = {"format": QAT_FORMAT, "quantizer": quantizer,
            "int8": ",".join(getattr(head, "int8_matrices", ())), **(extra or {})}
    save_file(out, path, metadata={k: str(v) for k, v in meta.items()})
    print(f"saved {path}", flush=True)


def memory_gb(device, aux_device=None):
    if device == "mps":
        return (f"{torch.mps.current_allocated_memory() / 2**30:.1f}GB allocated, "
                f"{torch.mps.driver_allocated_memory() / 2**30:.1f}GB driver")
    if str(device).startswith("cuda"):
        devices = dict.fromkeys([torch.device(device), torch.device(aux_device or device)])
        return ", ".join(f"{d} {torch.cuda.max_memory_allocated(d) / 2**30:.1f}GB peak"
                         for d in devices)
    return "n/a"


class _GradScale(torch.autograd.Function):
    """Identity whose backward multiplies the gradient by `scale`."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def sweep(args, student, teacher, evaluate):
    """Evaluate mixed configs: int8 matrices from the rotated teacher, ternary
    ones from the student's own latents (--init)."""
    configs = json.loads(Path(args.sweep).read_text())
    own = {}
    rotated = {}
    with torch.no_grad():
        for name, parts in MATRICES.items():
            module = getattr(student, MODULES[name])
            own[name] = module.weight.detach().clone()
            dense = getattr(teacher, MODULES[name]).weight.detach().float().to(own[name].device)
            rotated[name] = rotate_rows(dense, parts, student.rotations)
            del dense
    for config in configs:
        names = parse_int8(config["int8"])
        with torch.no_grad():
            for name, parts in MATRICES.items():
                module = getattr(student, MODULES[name])
                for (s, e, _), runtime in zip(parts, [r for r, (n, _) in RUNTIME.items()
                                                      if n == name]):
                    src = rotated if runtime in names else own
                    module.weight[:, s:e].copy_(src[name][:, s:e])
        set_int8(student, names)
        print(f"SWEEP {config['label']} MB={head_bytes(names) / 1e6:.1f} "
              f"int8=[{','.join(n.removeprefix('mtp.') for n in names)}]", flush=True)
        evaluate(args.eval_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="dir of shard-*.bin")
    ap.add_argument("--tables", required=True, help="`mtp-capture tables` dir")
    ap.add_argument("--teacher", required=True, help="frozen teacher head (any convention)")
    ap.add_argument("--signs", required=True, help="`mtp-capture signs` dir")
    ap.add_argument("--init", help="QAT checkpoint to start from (default: rotated teacher)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantizer", default="twn", choices=DIAGNOSTIC_QUANTIZERS)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--norm-lr", type=float, default=0.0, help="0 = --lr")
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--total-steps", type=int, default=0, help="cosine horizon (0 = constant)")
    ap.add_argument("--start-step", type=int, default=0, help="schedule offset when resuming")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--target-weight", type=float, default=0.1)
    ap.add_argument("--kd-weight", type=float, default=0.5, help="soft top-k share of step-1 target")
    ap.add_argument("--depth-weights", default="1,1,1")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=30)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-only", action="store_true", help="PTQ / checkpoint metrics only")
    ap.add_argument("--table-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--aux-device", help="teacher + embedding/output tables (default: --device)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="split each training batch into this many micro-batches")
    ap.add_argument("--loss-scale", type=float, default=1.0,
                    help="gradient scale across the output-table matmul (for float16 tables)")
    ap.add_argument("--recompute-ternary", action="store_true",
                    help="re-quantize in backward instead of saving ternary weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.02,
                    help="share of capture shards held out for EVAL")
    ap.add_argument("--int8", help="int8 runtime matrices (ternary.parse_int8 spec; "
                    "default: the --init checkpoint's, else none)")
    ap.add_argument("--sweep", help="eval-only: JSON list of {label, int8} configs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    aux = args.aux_device or device
    TernaryLinear.recompute = args.recompute_ternary
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    meta = json.loads(Path(args.tables, "meta.json").read_text())
    eps, rope_base = meta["rms_norm_eps"], meta["rope_theta"]
    teacher, detected = load_head(args.teacher, eps=eps, rope_base=rope_base)
    print(f"teacher {args.teacher}: {detected['convention']}", flush=True)
    teacher = teacher.to(aux).eval().requires_grad_(False)
    signs = load_signs(args.signs)
    int8 = parse_int8(args.int8) if args.int8 is not None else None
    student = build_student(args.teacher, signs, args.quantizer, eps, rope_base,
                            init=args.init, device=device, int8=int8)
    print(f"student: ternary g128 {args.quantizer}, int8 "
          f"[{','.join(student.int8_matrices)}] ({head_bytes(student.int8_matrices) / 1e6:.1f} MB), "
          f"init {args.init or 'rotated teacher (PTQ)'}", flush=True)

    table_dtype = getattr(torch, args.table_dtype)
    embed_w = load_file(os.path.join(args.tables, "embed_tokens.safetensors"))["weight"]
    embed_w = embed_w.to(device=aux, dtype=table_dtype)
    lmhead_w = load_file(os.path.join(args.tables, "lm_head.safetensors"))["weight"]
    lmhead_w = lmhead_w.to(device=aux, dtype=table_dtype)

    def embed(ids):
        return F.embedding(ids.to(aux), embed_w).float().to(ids.device)

    def lm(h):
        """Logits on the aux device. With --loss-scale, gradients cross the
        low-precision matmul scaled up (no underflow) and are unscaled in F32."""
        if args.loss_scale == 1.0 or not torch.is_grad_enabled():
            return F.linear(h.to(aux, table_dtype), lmhead_w)
        h = _GradScale.apply(h, 1.0 / args.loss_scale)
        logits = F.linear(h.to(aux, table_dtype), lmhead_w).float()
        return _GradScale.apply(logits, args.loss_scale)

    depth_w = [float(x) for x in args.depth_weights.split(",")]
    matrices = [p for p in student.parameters() if p.dim() == 2]
    norms = [p for p in student.parameters() if p.dim() == 1]
    optimizer = torch.optim.AdamW(
        [{"params": matrices, "lr_mult": 1.0},
         {"params": norms, "lr_mult": (args.norm_lr / args.lr) if args.norm_lr else 1.0}],
        lr=args.lr, weight_decay=0.0)

    def lr_at(step):
        if args.warmup and step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        if args.total_steps <= 0:
            return args.lr
        t = min(1.0, (step - args.warmup) / max(1, args.total_steps - args.warmup))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    batcher = DocBatcher(args.capture, window=args.window, batch=args.batch, seed=args.seed,
                         val_fraction=args.val_fraction)
    if args.start_step:
        # Same held-out split (fixed in the constructor), fresh training order.
        batcher.rng = np.random.default_rng((args.seed, args.start_step))

    def chain_inputs(tokens, top_ids):
        """Per step: (input tokens, labels, keep mask) as train.py builds them."""
        B, L = tokens.shape
        argmax = top_ids[:, :, 0]
        match = torch.zeros(B, L, dtype=torch.bool, device=tokens.device)
        match[:, :-1] = tokens[:, 1:] == argmax[:, :-1]
        steps = [(tokens, argmax, torch.ones_like(match))]
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
            keep = valid.clone()
            keep[:, L - offset:] = False
            steps.append((token_in, label, keep))
        return steps

    def run_chain(head, steps, hidden, positions):
        """Yield (step, head output rows at keep) along the head's own chain."""
        L = hidden.shape[1]
        prev_out, prev_kv = None, []
        for j, (token_in, _, keep) in enumerate(steps, start=1):
            if j == 1:
                fused = head.fuse(embed(token_in), hidden)
                out, k, v = run_layer(head, fused, positions)
            else:
                fused = head.fuse(embed(token_in), prev_out)
                k_hist = torch.cat([kv[0] for kv in prev_kv], dim=2)
                v_hist = torch.cat([kv[1] for kv in prev_kv], dim=2)
                out, k, v = run_layer(head, fused, positions + j - 1,
                                      kv_extra=(k_hist, v_hist),
                                      attn_mask=chain_mask(L, j, hidden.device))
            prev_out = out
            prev_kv.append((k, v))
            yield j, out[keep]

    def forward(tokens, top_ids, top_logits, hidden, positions, want_stats=False):
        steps = chain_inputs(tokens, top_ids)
        with torch.no_grad():
            if aux != device:
                aux_steps = [tuple(t.to(aux) for t in step) for step in steps]
                teacher_rows = [rows for _, rows in run_chain(
                    teacher, aux_steps, hidden.to(aux), positions.to(aux))]
            else:
                teacher_rows = [rows for _, rows in run_chain(teacher, steps, hidden, positions)]
        if aux != device:  # the loss terms live with the output table
            top_ids, top_logits = top_ids.to(aux), top_logits.to(aux)
        losses, stats = [], {}
        for j, rows in run_chain(student, steps, hidden, positions):
            _, label, keep = steps[j - 1]
            label, keep = label.to(aux), keep.to(aux)
            n = max(1, int(keep.sum()))
            with torch.no_grad():
                t_logits = lm(teacher_rows[j - 1]).float()
                t_logp = torch.log_softmax(t_logits, dim=-1)
                t_arg = t_logits.argmax(-1)
                del t_logits
            s_logits = lm(rows).float()
            logq = torch.log_softmax(s_logits, dim=-1)
            kl = (t_logp.exp() * (t_logp - logq)).sum(-1)
            labels = label[keep]
            hard = -torch.gather(logq, -1, labels.unsqueeze(-1)).squeeze(-1)
            if j == 1:
                target = torch.softmax(top_logits[keep].float(), dim=-1)
                soft = -(target * torch.gather(logq, -1, top_ids[keep])).sum(-1)
                tgt = (1 - args.kd_weight) * hard + args.kd_weight * soft
            else:
                tgt = hard
            dw = depth_w[min(j - 1, len(depth_w) - 1)]
            losses.append(dw * (args.kl_weight * kl.sum() + args.target_weight * tgt.sum()) / n)
            if want_stats:
                s_arg = s_logits.argmax(-1)
                stats[f"agree{j}"] = ((s_arg == t_arg).sum() / n).item()
                stats[f"acc{j}"] = ((s_arg == labels).sum() / n).item()
                stats[f"tacc{j}"] = ((t_arg == labels).sum() / n).item()
                stats[f"kl{j}"] = (kl.sum() / n).item()
            del s_logits, logq, t_logp
        return sum(losses), stats

    def evaluate(max_batches):
        student.eval()
        agg = {}
        with torch.no_grad():
            for bi, batch in enumerate(batcher.val_batches()):
                if bi >= max_batches:
                    break
                tokens, top_ids, top_logits, hidden, positions = (t.to(device) for t in batch)
                _, stats = forward(tokens, top_ids, top_logits, hidden.float(), positions,
                                   want_stats=True)
                for key, value in stats.items():
                    agg.setdefault(key, []).append(value)
                if device == "mps":
                    torch.mps.empty_cache()
        means = {k: sum(v) / len(v) for k, v in agg.items()}
        line = " ".join(f"{k}={means[k]:.4f}" for k in sorted(means))
        print(f"EVAL {line}", flush=True)
        student.train()
        return means

    if args.sweep:
        sweep(args, student, teacher, evaluate)
        return
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
            lr = lr_at(args.start_step + step_count)
            for group in optimizer.param_groups:
                group["lr"] = lr * group["lr_mult"]
            optimizer.zero_grad(set_to_none=True)
            if args.grad_accum > 1:
                # Weight unequal or fewer chunks by their share of the batch.
                micro = zip(*(t.chunk(args.grad_accum) for t in batch))
                loss = 0.0
                for part in micro:
                    tokens, top_ids, top_logits, hidden, positions = (t.to(device) for t in part)
                    part_loss, _ = forward(tokens, top_ids, top_logits, hidden.float(),
                                           positions)
                    weight = tokens.shape[0] / batch[0].shape[0]
                    (part_loss * weight).backward()
                    loss += part_loss.detach() * weight
                    del part_loss
            else:
                tokens, top_ids, top_logits, hidden, positions = (t.to(device) for t in batch)
                loss, _ = forward(tokens, top_ids, top_logits, hidden.float(), positions)
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            if device == "mps":
                torch.mps.empty_cache()
            step_count += 1
            if step_count % args.log_every == 0:
                if device == "mps":
                    torch.mps.synchronize()
                now = time.time()
                print(f"step {step_count} loss {loss.item():.4f} gnorm {float(grad_norm):.3f} "
                      f"{(now - last) / args.log_every:.2f}s/step lr {lr:.2e} "
                      f"mem {memory_gb(device, aux)}", flush=True)
                last = now
            if step_count % args.eval_every == 0:
                evaluate(args.eval_batches)
            if step_count % args.save_every == 0:
                save_qat(student, f"{args.out}/qat-step{args.start_step + step_count}.safetensors",
                         args.quantizer, {"step": args.start_step + step_count})
            if args.max_steps and step_count >= args.max_steps:
                done = True
                break
        if done:
            break
    if step_count % args.eval_every:
        evaluate(args.eval_batches)
    save_qat(student, f"{args.out}/qat-final.safetensors", args.quantizer,
             {"step": args.start_step + step_count})
    print(f"{step_count} steps in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
