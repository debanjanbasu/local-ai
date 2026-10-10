"""EXPERIMENTAL head-only judgment pointer training and evaluation (not production).

Research tool for the frozen Bonsai PTQ1 trunk: it never touches the trunk, never
transplants LoRA weights and does not establish production quality. See the
local-engine README for the small Kev pilot and its limitations. It reads
features captured by the native worker and trains only the pointer head:

    q = Q h_decide + b_q        Q: [HEAD_DIM, width], b_q: [HEAD_DIM]
    k_i = K h_option_i + b_k    K: [HEAD_DIM, width], b_k: [HEAD_DIM]
    logit_i = dot(k_i, q) / sqrt(HEAD_DIM) / temperature
    p = softmax(logit) over the sample's options, in feature-row order

Input directory (exact format, validated before anything else happens):

    features.jsonl   one JSON object per line:
        {"id": str, "token_ids": [int], "positions": [sorted unique zero-based int],
         "width": 5120, "feature_file": "feature-000000.bin",
         "metadata": {"split": "train"|"calibration"|"validation"|"test",
                      "target": int, ...}}
    feature-NNNNNN.bin   len(positions) * width little-endian fp16 values: the
        output-norm hidden state at each requested position; option rows first
        (N - 1 of them), the decide row last. `target` indexes the options.

Splits have fixed roles: weights train on `train`, the checkpoint is chosen by
`validation` NLL, a scalar positive temperature is fit only on `calibration`, and
`test` is touched once for the final report. Temperature scaling is measured on
this dataset's calibration split only; nothing here makes probabilities
calibrated beyond the measured data.

Development mode (`--development`, `mode="development"`) is for datasets whose
final test is locked away (e.g. hard-v1): it needs train/validation/calibration,
refuses any `test` row (or `metadata.kev_partition == "test"`), and reports the
validation split -- which also chose the checkpoint, so it is NOT a held-out
final test -- next to train-only uniform/majority baselines and per
family/question_type/option-count metrics when that metadata is present.

Export (`head.safetensors`, all F32, row-major, little-endian):
    q.weight [HEAD_DIM, width]   q.bias [HEAD_DIM]
    k.weight [HEAD_DIM, width]   k.bias [HEAD_DIM]
    temperature [1]              (> 0, divides the scaled logits)
`report.json` beside it records config, provenance, feature hashes, sample counts
and metrics. Validation and the metric/calibration math are pure Python; PyTorch
is imported lazily and only by `train`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

WIDTH = 5120
HEAD_DIM = 256
SPLITS = ("train", "validation", "calibration", "test")
ROW_KEYS = {"id", "token_ids", "positions", "width", "feature_file", "metadata"}
FEATURE_NAME = re.compile(r"feature-[0-9]{6}\.bin")
TEMPERATURE_BOUNDS = (1e-3, 1e3)
ECE_BINS = 15
FORMAT = "local-ai.judgment-pointer-head.v0-experimental"
MODES = ("final", "development")
DEVELOPMENT_SPLITS = ("train", "validation", "calibration")
LOCKED_PARTITIONS = frozenset({"test"})  # metadata.kev_partition values never trained on
GROUP_KEYS = ("family", "question_type")


class DatasetError(ValueError):
    """The feature directory does not match the exact capture format."""


@dataclass(frozen=True)
class Sample:
    id: str
    split: str
    target: int
    options: int  # N - 1 option rows; the decide row follows them
    width: int
    features: bytes  # raw little-endian fp16, (options + 1) * width values
    sha256: str
    token_ids: tuple[int, ...]
    metadata: dict = field(default_factory=dict, compare=False, repr=False)


def _is_int(value) -> bool:
    return type(value) is int


def _parse_row(row, line: int, directory: Path, width: int) -> Sample:
    where = f"features.jsonl line {line}"
    if not isinstance(row, dict) or set(row) != ROW_KEYS:
        raise DatasetError(f"{where}: row must have exactly the keys {sorted(ROW_KEYS)}")
    sample_id = row["id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise DatasetError(f"{where}: id must be a non-empty string")
    tokens = row["token_ids"]
    if not isinstance(tokens, list) or not tokens or not all(
        _is_int(t) and t >= 0 for t in tokens
    ):
        raise DatasetError(f"{where}: token_ids must be non-empty non-negative integers")
    positions = row["positions"]
    if not isinstance(positions, list) or not all(_is_int(p) for p in positions):
        raise DatasetError(f"{where}: positions must be integers")
    if len(positions) < 3:
        raise DatasetError(f"{where}: need at least two option positions plus decide")
    if positions[0] < 0 or any(a >= b for a, b in itertools.pairwise(positions)):
        raise DatasetError(f"{where}: positions must be sorted, unique and zero-based")
    if positions[-1] >= len(tokens):
        raise DatasetError(f"{where}: position {positions[-1]} is past token_ids")
    if not _is_int(row["width"]) or row["width"] != width:
        raise DatasetError(f"{where}: width {row['width']!r} is not the expected {width}")
    name = row["feature_file"]
    if not isinstance(name, str) or not FEATURE_NAME.fullmatch(name):
        raise DatasetError(f"{where}: feature_file must be a plain feature-NNNNNN.bin name")
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise DatasetError(f"{where}: {name} is missing or not a regular file in the directory")
    metadata = row["metadata"]
    if not isinstance(metadata, dict):
        raise DatasetError(f"{where}: metadata must be an object")
    split = metadata.get("split")
    if split not in SPLITS:
        raise DatasetError(f"{where}: unsupported split {split!r}")
    options = len(positions) - 1
    target = metadata.get("target")
    if not _is_int(target) or not 0 <= target < options:
        raise DatasetError(f"{where}: target must be an integer in [0, {options})")
    data = path.read_bytes()
    count = len(positions) * width
    if len(data) != 2 * count:
        raise DatasetError(f"{where}: {name} has {len(data)} bytes, expected {2 * count}")
    if not all(map(math.isfinite, struct.unpack(f"<{count}e", data))):
        raise DatasetError(f"{where}: {name} contains non-finite values")
    return Sample(
        sample_id, split, target, options, width, data,
        hashlib.sha256(data).hexdigest(), tuple(tokens), metadata,
    )


def load_dataset(directory: Path, width: int = WIDTH) -> list[Sample]:
    """Validate the whole directory and return its samples, or raise DatasetError."""
    directory = Path(directory)
    manifest = directory / "features.jsonl"
    if manifest.is_symlink() or not manifest.is_file():
        raise DatasetError(f"{manifest} is missing")
    text = manifest.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    samples = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetError(f"features.jsonl line {number}: invalid JSON") from error
        samples.append(_parse_row(row, number, directory, width))
    if not samples:
        raise DatasetError("features.jsonl has no rows")
    # Leakage guards: an id, feature file, feature payload or prompt may appear once.
    seen: dict[str, dict] = {"id": {}, "features": {}, "prompt": {}}
    for sample in samples:
        for kind, key in (
            ("id", sample.id),
            ("features", sample.sha256),
            ("prompt", sample.token_ids),
        ):
            if key in seen[kind]:
                other = seen[kind][key]
                raise DatasetError(
                    f"duplicate {kind} between {other.id!r} ({other.split}) "
                    f"and {sample.id!r} ({sample.split})"
                )
            seen[kind][key] = sample
    return samples


def split_counts(samples) -> dict[str, int]:
    return {split: sum(s.split == split for s in samples) for split in SPLITS}


def check_mode(samples, mode: str = "final") -> dict[str, int]:
    """Enforce a mode's split rules on validated samples; return split counts.

    `final` needs every split. `development` needs train/validation/calibration,
    refuses test rows, locked-partition rows, `metadata.option_count` that
    disagrees with the captured option rows, and `metadata.group_id` spanning splits.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    counts = split_counts(samples)
    if mode == "final":
        missing = [split for split, n in counts.items() if n == 0]
        if missing:
            raise DatasetError(f"every split needs samples; empty: {missing}")
        return counts
    if counts["test"]:
        first = next(s.id for s in samples if s.split == "test")
        raise DatasetError(f"development mode refuses test rows; found {counts['test']} "
                           f"(first {first!r})")
    for sample in samples:
        if sample.metadata.get("kev_partition") in LOCKED_PARTITIONS:
            raise DatasetError(f"development mode refuses locked-partition row {sample.id!r}")
        declared = sample.metadata.get("option_count")
        if declared is not None and declared != sample.options:
            raise DatasetError(f"{sample.id!r}: metadata.option_count {declared!r} != "
                               f"{sample.options} captured option rows")
    missing = [split for split in DEVELOPMENT_SPLITS if counts[split] == 0]
    if missing:
        raise DatasetError(f"development mode needs train, validation and calibration; "
                           f"empty: {missing}")
    groups: dict = {}
    for sample in samples:
        group = sample.metadata.get("group_id")
        if isinstance(group, str):
            other = groups.setdefault(group, sample)
            if other.split != sample.split:
                raise DatasetError(f"group {group!r} spans {other.split} ({other.id!r}) and "
                                   f"{sample.split} ({sample.id!r})")
    return counts


def option_histogram(samples) -> dict[str, dict[str, int]]:
    """{split: {option count: rows}} over the splits present."""
    out: dict[str, dict[str, int]] = {}
    for sample in samples:
        bucket = out.setdefault(sample.split, {})
        bucket[str(sample.options)] = bucket.get(str(sample.options), 0) + 1
    return {split: dict(sorted(b.items(), key=lambda kv: int(kv[0])))
            for split, b in sorted(out.items())}


def consistent_value(samples, key: str):
    """The single string `metadata[key]` shared by every sample, else None."""
    values = {sample.metadata.get(key) for sample in samples
              if isinstance(sample.metadata.get(key), str)}
    if len(values) == 1 and all(isinstance(s.metadata.get(key), str) for s in samples):
        return values.pop()
    return None


def feature_scope(samples) -> dict:
    """Dataset/renderer recorded in the feature metadata, only when every row agrees."""
    scope = {}
    for key in ("dataset", "renderer"):
        value = consistent_value(samples, key)
        scope[key] = value
        if value is None:
            seen = sorted({str(s.metadata[key]) for s in samples if key in s.metadata})
            if seen:
                scope[key + "_values"] = seen  # mixed or partial: not exported
    return scope


# ---- pure-Python metrics and temperature fit -------------------------------


def _log_softmax(logits, scale=1.0):
    scaled = [scale * z for z in logits]
    peak = max(scaled)
    total = peak + math.log(sum(math.exp(z - peak) for z in scaled))
    return [z - total for z in scaled]


def _check_logits(logits, targets):
    if not logits or len(logits) != len(targets):
        raise ValueError("need one target per non-empty logit list")
    for row, target in zip(logits, targets):
        if len(row) < 2 or not all(math.isfinite(z) for z in row):
            raise ValueError("each sample needs at least two finite logits")
        if not _is_int(target) or not 0 <= target < len(row):
            raise ValueError("target out of range")


def evaluate(logits, targets, temperature=1.0, bins=ECE_BINS) -> dict:
    """Accuracy, mean NLL, mean multi-class Brier and top-label ECE.

    Argmax ties resolve to the first option. ECE uses `bins` equal-width
    confidence bins on [0, 1]; Brier sums squared error over each sample's options.
    """
    _check_logits(logits, targets)
    if not temperature > 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    n = len(logits)
    correct = nll = brier = 0.0
    bin_count = [0] * bins
    bin_conf = [0.0] * bins
    bin_hits = [0.0] * bins
    for row, target in zip(logits, targets):
        logp = _log_softmax(row, 1.0 / temperature)
        probs = [math.exp(v) for v in logp]
        top = max(range(len(probs)), key=probs.__getitem__)
        hit = float(top == target)
        correct += hit
        nll -= logp[target]
        brier += sum((p - (i == target)) ** 2 for i, p in enumerate(probs))
        b = min(int(probs[top] * bins), bins - 1)
        bin_count[b] += 1
        bin_conf[b] += probs[top]
        bin_hits[b] += hit
    ece = sum(abs(bin_hits[b] - bin_conf[b]) for b in range(bins) if bin_count[b]) / n
    return {
        "count": n,
        "accuracy": correct / n,
        "nll": nll / n,
        "brier": brier / n,
        "ece": ece,
        "temperature": temperature,
    }


def fit_temperature(logits, targets, bounds=TEMPERATURE_BOUNDS, steps=200) -> float:
    """Scalar T > 0 minimising mean NLL of softmax(logits / T).

    Mean NLL is convex in beta = 1/T with derivative mean(E_p[z] - z_target),
    which is monotone, so bisection on log(beta) finds the optimum; it is
    clamped to `bounds` (e.g. perfectly separable data) and is 1.0 when every
    sample's logits are constant (any T is optimal).
    """
    _check_logits(logits, targets)
    if all(max(row) == min(row) for row in logits):
        return 1.0

    def slope(beta):
        total = 0.0
        for row, target in zip(logits, targets):
            probs = [math.exp(v) for v in _log_softmax(row, beta)]
            total += sum(p * z for p, z in zip(probs, row)) - row[target]
        return total / len(logits)

    low, high = math.log(1.0 / bounds[1]), math.log(1.0 / bounds[0])
    if slope(math.exp(low)) >= 0:
        return bounds[1]
    if slope(math.exp(high)) <= 0:
        return bounds[0]
    for _ in range(steps):
        mid = (low + high) / 2
        if slope(math.exp(mid)) < 0:
            low = mid
        else:
            high = mid
    return 1.0 / math.exp((low + high) / 2)


def majority_table(samples) -> dict[int, list[int]]:
    """Target-position counts per option count (fit on the samples given: train only)."""
    table: dict[int, list[int]] = {}
    for sample in samples:
        table.setdefault(sample.options, [0] * sample.options)[sample.target] += 1
    return table


def majority_logits(table, options: int) -> list[float]:
    """Log add-one-smoothed position frequencies; argmax is the majority position
    (ties to the first). Unseen option counts fall back to uniform."""
    return [math.log(c + 1) for c in table.get(options, [0] * options)]


def baselines(train, evaluated) -> dict:
    """Uniform and train-only majority baselines evaluated on `evaluated`."""
    table = majority_table(train)
    targets = [s.target for s in evaluated]
    uniform = evaluate([[0.0] * s.options for s in evaluated], targets)
    uniform["expected_accuracy"] = sum(1 / s.options for s in evaluated) / len(evaluated)
    return {
        "fit_split": "train",
        "uniform": uniform,
        "majority": evaluate([majority_logits(table, s.options) for s in evaluated], targets),
        "majority_table": {str(k): v for k, v in sorted(table.items())},
        "note": ("uniform accuracy uses the argmax tie rule (first option); "
                 "expected_accuracy is mean 1/options. majority predicts the most "
                 "frequent train target position for the row's option count with "
                 "add-one-smoothed train frequencies as probabilities."),
    }


def group_metrics(samples, logits, temperature, table) -> dict:
    """Per-group model (temperature applied) and majority metrics.

    Groups by option count always, and by each of GROUP_KEYS (plus
    family/question_type jointly) only when every sample carries it as a string.
    """
    keys = {"option_count": [str(s.options) for s in samples]}
    present = [k for k in GROUP_KEYS
               if all(isinstance(s.metadata.get(k), str) for s in samples)]
    for key in present:
        keys[key] = [s.metadata[key] for s in samples]
    if len(present) == len(GROUP_KEYS):
        keys["/".join(GROUP_KEYS)] = ["/".join(s.metadata[k] for k in GROUP_KEYS)
                                      for s in samples]
    out = {}
    for key, labels in keys.items():
        out[key] = {}
        for label in sorted(set(labels)):
            index = [i for i, value in enumerate(labels) if value == label]
            members = [samples[i] for i in index]
            targets = [s.target for s in members]
            out[key][label] = {
                "model": evaluate([logits[i] for i in index], targets, temperature),
                "majority": evaluate([majority_logits(table, s.options) for s in members],
                                     targets),
            }
    return out


# ---- export format ----------------------------------------------------------


def write_safetensors(path: Path, tensors: dict, metadata: dict) -> None:
    """Write F32 tensors {name: (shape, flat values)} as a safetensors file."""
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        shape, values = tensors[name]
        if math.prod(shape) != len(values):
            raise ValueError(f"{name}: shape {shape} does not match {len(values)} values")
        blob = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": list(shape),
                        "data_offsets": [offset, offset + len(blob)]}
        chunks.append(blob)
        offset += len(blob)
    header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def read_safetensors(path: Path) -> tuple[dict, dict]:
    """Read an F32-only safetensors file into ({name: (shape, values)}, metadata)."""
    data = Path(path).read_bytes()
    (size,) = struct.unpack_from("<Q", data)
    header = json.loads(data[8 : 8 + size])
    metadata = header.pop("__metadata__", {})
    body = data[8 + size :]
    tensors = {}
    for name, info in header.items():
        if info["dtype"] != "F32":
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        start, end = info["data_offsets"]
        tensors[name] = (tuple(info["shape"]), list(struct.unpack(f"<{(end - start) // 4}f",
                                                                   body[start:end])))
    return tensors, metadata


def reference_logits(tensors: dict, option_rows, decide_row) -> list[float]:
    """Pure-Python reading of the documented export layout (parity reference)."""
    (dim, width), q_w = tensors["q.weight"]
    k_w, q_b, k_b = tensors["k.weight"][1], tensors["q.bias"][1], tensors["k.bias"][1]
    (temperature,) = tensors["temperature"][1]

    def project(weight, bias, h):
        return [sum(weight[r * width + c] * h[c] for c in range(width)) + bias[r]
                for r in range(dim)]

    q = project(q_w, q_b, decide_row)
    scale = math.sqrt(dim) * temperature
    return [sum(a * b for a, b in zip(project(k_w, k_b, h), q)) / scale for h in option_rows]


# ---- training (PyTorch, lazy) ----------------------------------------------

DEFAULT_CONFIG = {
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 64,
    "seed": 0,
    "width": WIDTH,
    "head_dim": HEAD_DIM,
}


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise SystemExit("training needs PyTorch (run it on Kaggle, which ships torch)") from error
    return torch


def _logits(torch, head, samples, head_dim, batch_size=256):
    """Unscaled-by-temperature pointer logits per sample, padding masked to -inf."""
    out = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            scores = _scores(torch, head, batch, head_dim)
            out += [row[: s.options].tolist() for row, s in zip(scores, batch)]
    return out


def _features(torch, sample):
    if sys.byteorder != "little":
        raise RuntimeError("fp16 feature decoding assumes a little-endian host")
    return torch.frombuffer(bytearray(sample.features), dtype=torch.float16).view(
        sample.options + 1, sample.width
    )


def _scores(torch, head, batch, head_dim):
    longest = max(s.options for s in batch)
    width = batch[0].width
    options = torch.zeros(len(batch), longest, width)
    decide = torch.zeros(len(batch), width)
    mask = torch.zeros(len(batch), longest, dtype=torch.bool)
    for i, sample in enumerate(batch):
        rows = _features(torch, sample).float()
        options[i, : sample.options] = rows[:-1]
        decide[i] = rows[-1]
        mask[i, : sample.options] = True
    device = next(head.parameters()).device
    options, decide, mask = options.to(device), decide.to(device), mask.to(device)
    q = head["q"](decide)
    k = head["k"](options)
    scores = torch.einsum("bmd,bd->bm", k, q) / math.sqrt(head_dim)
    # Padded slots get -inf: zero probability and zero gradient, never learned.
    return scores.masked_fill(~mask, float("-inf"))


def _validate_config(config: dict) -> dict:
    config = {**DEFAULT_CONFIG, **config}
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError(f"unknown config keys: {sorted(set(config) - set(DEFAULT_CONFIG))}")
    for key in ("epochs", "batch_size", "width", "head_dim"):
        if not _is_int(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not _is_int(config["seed"]):
        raise ValueError("seed must be an integer")
    for key in ("lr", "weight_decay"):
        if not isinstance(config[key], (int, float)) or not 0 <= config[key] < math.inf:
            raise ValueError(f"{key} must be a finite non-negative number")
    return config


def train_and_export(directory: Path, output: Path, config: dict | None = None,
                     source_sha256: str | None = None, *, mode: str = "final") -> dict:
    """Validate, train on train, select on validation, calibrate, then either test
    once (`final`) or report validation without any test rows (`development`)."""
    config = _validate_config(config or {})
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; refusing to overwrite")
    samples = load_dataset(directory, config["width"])
    counts = check_mode(samples, mode)
    by_split = {split: [s for s in samples if s.split == split] for split in SPLITS}

    torch = _require_torch()
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = config["head_dim"]
    head = torch.nn.ModuleDict({
        "q": torch.nn.Linear(config["width"], dim),
        "k": torch.nn.Linear(config["width"], dim),
    }).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    train, validation = by_split["train"], by_split["validation"]
    validation_targets = [s.target for s in validation]
    best, history = None, []
    for epoch in range(config["epochs"]):
        head.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        total = 0.0
        for start in range(0, len(order), config["batch_size"]):
            batch = [train[i] for i in order[start : start + config["batch_size"]]]
            targets = torch.tensor([s.target for s in batch], device=device)
            loss = torch.nn.functional.cross_entropy(_scores(torch, head, batch, dim), targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        nll = evaluate(_logits(torch, head, validation, dim), validation_targets)["nll"]
        history.append({"epoch": epoch + 1, "train_nll": total / len(train),
                        "validation_nll": nll})
        if best is None or nll < best[1]:
            best = (epoch + 1, nll, {k: v.detach().cpu().clone()
                                     for k, v in head.state_dict().items()})
    head.load_state_dict(best[2])

    used = SPLITS if mode == "final" else DEVELOPMENT_SPLITS
    logits = {split: _logits(torch, head, by_split[split], dim) for split in used}
    targets = {split: [s.target for s in by_split[split]] for split in used}
    temperature = fit_temperature(logits["calibration"], targets["calibration"])
    if mode == "development":
        metrics = {
            "train": evaluate(logits["train"], targets["train"]),
            "validation": evaluate(logits["validation"], targets["validation"]),
            "validation_calibrated": evaluate(logits["validation"], targets["validation"],
                                              temperature),
            "calibration_before": evaluate(logits["calibration"], targets["calibration"]),
            "calibration_after": evaluate(logits["calibration"], targets["calibration"],
                                          temperature),
        }
    else:
        metrics = {
            "train": evaluate(logits["train"], targets["train"]),
            "validation": evaluate(logits["validation"], targets["validation"]),
            "calibration_before": evaluate(logits["calibration"], targets["calibration"]),
            "calibration_after": evaluate(logits["calibration"], targets["calibration"],
                                          temperature),
            "test_uncalibrated": evaluate(logits["test"], targets["test"]),
            "test": evaluate(logits["test"], targets["test"], temperature),
        }

    if source_sha256 is None:
        source = Path(__file__)  # frozen Kaggle copies pass their embedded digest instead
        source_sha256 = (hashlib.sha256(source.read_bytes()).hexdigest()
                         if source.is_file() else "unknown")
    report = {
        "format": FORMAT,
        "experimental": True,
        "note": ("Head-only research artifact over a frozen trunk. Temperature was fit on "
                 "this dataset's calibration split only; test metrics describe this "
                 "dataset's held-out test split only and are not a claim about other data."),
        "config": config,
        "sample_counts": counts,
        "option_counts": sorted({s.options for s in samples}),
        "selected_epoch": best[0],
        "history": history,
        "temperature": temperature,
        "temperature_at_bound": temperature in TEMPERATURE_BOUNDS,
        "metrics": metrics,
        "provenance": {
            "tool_sha256": source_sha256,
            "features_jsonl_sha256": hashlib.sha256(
                (Path(directory) / "features.jsonl").read_bytes()).hexdigest(),
            "feature_sha256": {s.id: s.sha256 for s in samples},
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "device": device,
        },
        "export": {
            "file": "head.safetensors",
            "layout": "F32 row-major: q.weight/k.weight [head_dim, width], q.bias/k.bias "
                      "[head_dim], temperature [1]; logit_i = (K h_i + b_k).(Q h_d + b_q) "
                      "/ sqrt(head_dim) / temperature, softmax over options in row order",
        },
    }
    head_metadata = {
        "format": FORMAT, "width": config["width"], "head_dim": dim,
        "tool_sha256": source_sha256, "experimental": "true",
    }
    if mode == "development":
        scope = feature_scope(samples)
        table = majority_table(train)
        report["mode"] = "development"
        report["note"] = (
            "DEVELOPMENT ONLY. Head-only research artifact over a frozen trunk. No test "
            "rows were present or scored. Validation chose the checkpoint, so validation "
            "metrics are selection-biased development numbers, not a held-out final "
            "test. Temperature was fit on this dataset's calibration split only.")
        report["evaluation"] = {"split": "validation", "held_out_final_test": False,
                                "selection_split": "validation",
                                "calibration_split": "calibration"}
        report["feature_scope"] = scope
        report["option_count_histogram"] = option_histogram(samples)
        report["baselines"] = {"evaluated_split": "validation",
                               **baselines(train, validation)}
        report["groups"] = {"split": "validation", "temperature": temperature,
                            **group_metrics(validation, logits["validation"], temperature,
                                            table)}
        head_metadata.update({
            "mode": "development",
            "evaluation": "validation (selection split; no held-out test)",
            "calibration_scope": "calibration split of the training feature directory only",
            "split_counts": json.dumps(counts, sort_keys=True, separators=(",", ":")),
        })
        head_metadata.update({k: v for k, v in scope.items() if k in ("dataset", "renderer")
                              and v is not None})
    state = best[2]
    tensors = {name: (tuple(state[name].shape), state[name].float().flatten().tolist())
               for name in ("q.weight", "q.bias", "k.weight", "k.bias")}
    tensors["temperature"] = ((1,), [temperature])
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    write_safetensors(partial / "head.safetensors", tensors, head_metadata)
    report["export"]["sha256"] = hashlib.sha256(
        (partial / "head.safetensors").read_bytes()).hexdigest()
    (partial / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    partial.rename(output)
    return report


def summary_of(report: dict) -> dict:
    """The short printed summary: test for final runs, validation for development."""
    summary = {"sample_counts": report["sample_counts"], "temperature": report["temperature"]}
    if report.get("mode") == "development":
        base = report["baselines"]
        summary.update({
            "mode": "development",
            "validation_not_held_out_test": report["metrics"]["validation_calibrated"],
            "baselines": {name: {k: base[name][k] for k in ("accuracy", "nll")}
                          for name in ("uniform", "majority")},
        })
    else:
        summary["test"] = report["metrics"]["test"]
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="validate a feature directory (no torch)")
    check.add_argument("features", type=Path, help="directory containing features.jsonl")
    check.add_argument("--width", type=int, default=WIDTH)
    check.add_argument("--development", action="store_true",
                       help="also enforce development-mode split rules (no test rows)")
    fit = sub.add_parser("train", help="train/select/calibrate/test and export (needs torch)")
    fit.add_argument("features", type=Path,
                     help="directory with features.jsonl; rows must cover all of "
                          "train, validation, calibration and test via metadata.split "
                          "(--development: all but test, and no test rows)")
    fit.add_argument("--output", type=Path, required=True, help="new output directory")
    fit.add_argument("--development", action="store_true",
                     help="development-only: train/validation/calibration, refuse test "
                          "rows, report validation (not a held-out test)")
    for key, value in DEFAULT_CONFIG.items():
        fit.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            samples = load_dataset(args.features, args.width)
            summary = {"sample_counts": split_counts(samples),
                       "option_counts": sorted({s.options for s in samples})}
            if args.development:
                check_mode(samples, "development")
                summary.update({"mode": "development",
                                "option_count_histogram": option_histogram(samples),
                                "feature_scope": feature_scope(samples)})
            print(json.dumps(summary))
        else:
            config = {key: getattr(args, key) for key in DEFAULT_CONFIG}
            mode = "development" if args.development else "final"
            report = train_and_export(args.features, args.output, config, mode=mode)
            print(json.dumps(summary_of(report), indent=2))
    except DatasetError as error:
        raise SystemExit(f"invalid feature directory: {error}") from error


if __name__ == "__main__":
    main()
