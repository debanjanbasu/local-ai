"""Check that the PyTorch head reproduces the runtime head's drafts.

`mtp-capture parity --out DIR` writes DIR/shard-000000.bin (the capture of one
document) and DIR/parity.json (the runtime head's draft chains at a few
positions: top-k ids and logits per step). This replays every chain in PyTorch
from the shard, with the exported tables and a head in the trainer convention
(e.g. `convert.py from-ternary` of the same ternary head the runtime loaded), and
compares argmaxes and logits.

Run: uv run --no-project --with torch --with safetensors --with numpy \\
       python tools/mtp_train/parity.py --parity DIR --tables DIR --head HEAD
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from data import read_shard
from mtp_head import load_head, run_layer
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    meta = json.loads(Path(args.tables, "meta.json").read_text())
    head, detected = load_head(args.head, eps=meta["rms_norm_eps"], rope_base=meta["rope_theta"])
    print(f"head: {detected}")
    head = head.to(device).eval()
    embed_w = load_file(os.path.join(args.tables, "embed_tokens.safetensors"))["weight"]
    lm_w = load_file(os.path.join(args.tables, "lm_head.safetensors"))["weight"].to(device)
    report = json.loads(Path(args.parity, "parity.json").read_text())
    tokens, top_ids, _, hidden = read_shard(os.path.join(args.parity, "shard-000000.bin"))
    assert list(tokens) == report["tokens"], "shard and parity.json disagree on tokens"

    def embed(ids):
        return F.embedding(torch.tensor(ids), embed_w).float().to(device)[None]

    def logits_of(row):
        # F16 table, F32 accumulation, in vocabulary chunks.
        return torch.cat([row @ chunk.float().T for chunk in lm_w.split(32768)], dim=-1)

    agree = total = 0
    worst = 0.0
    with torch.no_grad():
        for chain in report["chains"]:
            p = chain[0]["position"]
            rows = slice(0, p + 1)
            hid = torch.from_numpy(hidden[rows]).float().to(device)[None]
            positions = torch.arange(p + 1, device=device)
            fused = head.fuse(embed(list(tokens[rows])), hid)
            out, k_hist, v_hist = run_layer(head, fused, positions)
            prev = out[:, -1:]
            for depth, step in enumerate(chain):
                if depth > 0:
                    fused = head.fuse(embed([step["input_token"]]), prev)
                    prev, k_own, v_own = run_layer(
                        head, fused, positions[-1:] + depth, kv_extra=(k_hist, v_hist),
                        attn_mask=torch.ones(1, k_hist.shape[2] + 1, dtype=torch.bool,
                                             device=device))
                    k_hist = torch.cat([k_hist, k_own], dim=2)
                    v_hist = torch.cat([v_hist, v_own], dim=2)
                logits = logits_of(prev[0, -1])
                values, ids = logits.topk(len(step["top_ids"]))
                runtime_ids = step["top_ids"]
                runtime_values = torch.tensor(step["top_logits"], device=device)
                same = int(ids[0]) == runtime_ids[0]
                agree += same
                total += 1
                diff = float((logits[torch.tensor(runtime_ids, device=device)]
                              - runtime_values).abs().max())
                worst = max(worst, diff)
                overlap = len(set(ids.tolist()) & set(runtime_ids))
                target_argmax = int(top_ids[step["position"], 0])
                print(f"pos {step['position']:5d} step {depth}: torch {int(ids[0]):6d} "
                      f"({float(values[0]):7.3f}) runtime {runtime_ids[0]:6d} "
                      f"({step['top_logits'][0]:7.3f}) {'OK ' if same else 'DIFF'} "
                      f"top{len(runtime_ids)} overlap {overlap} max|dlogit| {diff:.4f} "
                      f"target argmax {target_argmax}")
    print(f"PARITY argmax agreement {agree}/{total}, worst top-k logit diff {worst:.4f}")


if __name__ == "__main__":
    main()
