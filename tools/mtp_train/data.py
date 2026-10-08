"""Shard reader for the capture format (from xkm/qwen3.8-27b-mtp-head-retrained).

Shard: [u32 magic 0x4D545044][u32 count][u32 hiddenDim][u32 kTop] then
count x { i32 token; kTop x i32 topIds; kTop x f32 topLogits; hiddenDim x f16 hidden }
One shard = one document from an empty cache (`mtp-capture capture`).

Record p is the head row the runtime encodes at absolute position p: token p,
the target's top-k logits of row p, and the target's normalized hidden of row
p - 1 (zeros for p = 0). In xkm's notation t = p - 1: hidden_t, token_{t+1},
top-k of target row t+1.
"""

import glob
import struct

import numpy as np
import torch

MAGIC = 0x4D545044


def read_shard(path):
    with open(path, "rb") as f:
        raw = f.read()
    magic, count, hdim, ktop = struct.unpack("<IIII", raw[:16])
    assert magic == MAGIC, path
    rec = 4 + ktop * 8 + hdim * 2
    body = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, rec)
    tokens = body[:, :4].copy().view(np.int32).reshape(count)
    top_ids = body[:, 4 : 4 + ktop * 4].copy().view(np.int32).reshape(count, ktop)
    top_logits = body[:, 4 + ktop * 4 : 4 + ktop * 8].copy().view(np.float32).reshape(count, ktop)
    hidden = body[:, 4 + ktop * 8 :].copy().view(np.float16).reshape(count, hdim)
    return tokens, top_ids, top_logits, hidden


class DocBatcher:
    """Yields fixed-length contiguous windows from shards, batched.

    Each window gives, for chain training at depth D:
      hidden[w]               target normalized hiddens (chain step 1 input)
      tokens[w]               committed tokens (embedding input, teacher forced)
      top_ids/top_logits[w]   target top-k at each position ([., 0] = argmax)
      positions[w]            absolute positions (RoPE)
    """

    def __init__(self, capture_dir, window=256, batch=8, seed=0, val_fraction=0.02):
        self.paths = sorted(glob.glob(capture_dir + "/shard-*.bin"))
        if not self.paths:
            raise SystemExit(f"no shard-*.bin in {capture_dir}")
        rng = np.random.default_rng(seed)
        rng.shuffle(self.paths)
        n_val = max(1, int(len(self.paths) * val_fraction))
        self.val_paths = self.paths[:n_val]
        self.train_paths = self.paths[n_val:] or self.paths
        self.window = window
        self.batch = batch
        self.rng = rng

    def _windows(self, paths, shuffle=True):
        order = list(paths)
        if shuffle:
            self.rng.shuffle(order)
        buffer = []
        for path in order:
            tokens, top_ids, top_logits, hidden = read_shard(path)
            count = len(tokens)
            starts = list(range(0, max(count - self.window, 1), self.window))
            if shuffle:
                self.rng.shuffle(starts)
            for start in starts:
                end = min(start + self.window, count)
                if end - start < 64:
                    continue
                buffer.append(
                    (tokens[start:end], top_ids[start:end], top_logits[start:end],
                     hidden[start:end], start)
                )
                if len(buffer) == self.batch:
                    yield self._collate(buffer)
                    buffer = []
        if buffer:
            yield self._collate(buffer)

    def _collate(self, items):
        window = min(x[0].shape[0] for x in items)
        tokens = torch.from_numpy(np.stack([x[0][:window] for x in items])).long()
        top_ids = torch.from_numpy(np.stack([x[1][:window] for x in items])).long()
        top_logits = torch.from_numpy(np.stack([x[2][:window] for x in items]))
        hidden = torch.from_numpy(np.stack([x[3][:window] for x in items]))
        positions = torch.from_numpy(
            np.stack([np.arange(x[4], x[4] + window) for x in items])
        ).long()
        return tokens, top_ids, top_logits, hidden, positions

    def train_batches(self):
        return self._windows(self.train_paths, shuffle=True)

    def val_batches(self):
        return self._windows(self.val_paths, shuffle=False)
