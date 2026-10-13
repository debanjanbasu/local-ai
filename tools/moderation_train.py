"""Train, calibrate and evaluate the moderation heads (see docs/MODERATION.md).

Inputs: a `tools/moderation_data.py` output directory (`dataset.jsonl`,
`dataset-manifest.json`) and a directory of `moderation-embed features`
chunks (`*.f32` + `*.json`: first-window EmbeddingGemma 2 embeddings of the
dataset rows, keyed by row id).

    fit       train one L2-regularized logistic regression per category on
              `train` (C chosen by calibration log-loss), fit Platt scaling and
              choose the `flagged` threshold on `calibration`; writes
              `candidate.json`. Never reads `test` or OOD rows.
    evaluate  score `test`, `ood_openai` and `ood_toxicchat` ONCE (refuses if
              `evaluation.json` exists), apply the pre-registered criteria
              (`CRITERIA`, committed in docs/MODERATION.md before the first
              evaluation), and write `evaluation.json` and the final
              `moderation-heads.json` in which every failing category is
              marked unsupported with its reason.

Metric and calibration math is pure Python (unit-tested); numpy and
scikit-learn are imported only by `fit`/`evaluate`:

    uv run --with numpy --with scikit-learn python tools/moderation_train.py fit ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CATEGORIES = (
    "hate",
    "hate/threatening",
    "harassment",
    "harassment/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
)
DIMENSIONS = 768
PROMPT = "task: classification | query: "
WINDOW_TOKENS = 512
MODEL_ID = "embeddinggemma-2-moderation-1"
C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)

# Pre-registered (docs/MODERATION.md, "Acceptance criteria").
CRITERIA = {
    "min_train_positives": 100,
    "min_calibration_positives": 30,
    "calibration_precision_target": 0.80,
    "calibration_min_recall": 0.30,
    "min_test_positives": 30,
    "test_min_auc": 0.90,
    "test_min_precision": 0.70,
    "test_min_recall": 0.50,
    "test_max_ece": 0.05,
    "ece_bins": 10,
    "ood_min_positives": 20,
    "ood_min_auc": 0.80,
}
NO_DATA = {
    "self-harm/intent": "no licensed training data separates self-harm intent from other self-harm content",
    "self-harm/instructions": "no licensed training data separates self-harm instructions from other self-harm content",
    "violence/graphic": "no licensed training data labels graphic violence",
}


# ---------------------------------------------------------------- pure math


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney AUC with average ranks for ties; None without both classes."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def ece(probs: list[float], labels: list[int], bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins."""
    totals = [[0, 0.0, 0] for _ in range(bins)]
    for p, y in zip(probs, labels, strict=True):
        b = min(int(p * bins), bins - 1)
        totals[b][0] += 1
        totals[b][1] += p
        totals[b][2] += y
    n = len(probs)
    return sum(abs(s - k) for c, s, k in totals if c) / n if n else 0.0


def confusion(
    probs: list[float], labels: list[int], threshold: float
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for p, y in zip(probs, labels, strict=True):
        if p >= threshold:
            tp += y
            fp += 1 - y
        else:
            fn += y
    return tp, fp, fn


def precision_recall(probs, labels, threshold) -> tuple[float, float]:
    tp, fp, fn = confusion(probs, labels, threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def choose_threshold(probs, labels, precision_target, min_recall) -> float | None:
    """Threshold with the highest recall whose precision >= target (ties: higher threshold).

    None when no threshold reaches the target with at least `min_recall`.
    """
    pairs = sorted(zip(probs, labels, strict=True), key=lambda t: -t[0])
    total_pos = sum(labels)
    if total_pos == 0:
        return None
    best = None
    tp = fp = 0
    i = 0
    while i < len(pairs):
        p = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == p:
            tp += pairs[i][1]
            fp += 1 - pairs[i][1]
            i += 1
        precision, recall = tp / (tp + fp), tp / total_pos
        if (
            precision >= precision_target
            and recall >= min_recall
            and (best is None or recall > best[1])
        ):
            best = (p, recall)
    if best is None:
        return None
    return min(max(best[0], 1e-6), 1 - 1e-6)


def fit_platt(
    logits: list[float], labels: list[int], iterations: int = 100
) -> tuple[float, float]:
    """Fit p = sigmoid(a * z + c) by damped Newton steps on the log-loss with
    Platt's smoothed targets. Returns (a, c) with a > 0."""
    pos = sum(labels)
    neg = len(labels) - pos
    hi, lo = (pos + 1) / (pos + 2), 1 / (neg + 2)
    targets = [hi if y else lo for y in labels]

    def loss(a: float, c: float) -> float:
        total = 0.0
        for z, t in zip(logits, targets, strict=True):
            u = a * z + c
            # log(1 + e^u) - t u, computed stably.
            total += (u if u > 0 else 0.0) + math.log1p(math.exp(-abs(u))) - t * u
        return total

    a, c = 1.0, 0.0
    current = loss(a, c)
    for _ in range(iterations):
        ga = gc = haa = hac = hcc = 0.0
        for z, t in zip(logits, targets, strict=True):
            p = sigmoid(a * z + c)
            r = p - t
            w = max(p * (1 - p), 1e-12)
            ga += r * z
            gc += r
            haa += w * z * z
            hac += w * z
            hcc += w
        det = haa * hcc - hac * hac
        if abs(det) < 1e-18:
            break
        da = (hcc * ga - hac * gc) / det
        dc = (haa * gc - hac * ga) / det
        step = 1.0
        while step > 1e-10:
            trial = loss(a - step * da, c - step * dc)
            if trial <= current:
                break
            step /= 2
        else:
            break
        a, c = a - step * da, c - step * dc
        improvement = current - trial
        current = trial
        if improvement < 1e-12 * max(1.0, abs(current)):
            break
    if not a > 0:
        raise ValueError(
            "Platt scaling found a non-positive slope; the head is not usable"
        )
    return a, c


def log_loss(probs, labels) -> float:
    eps = 1e-12
    return -sum(
        math.log(max(p, eps)) if y else math.log(max(1 - p, eps))
        for p, y in zip(probs, labels, strict=True)
    ) / len(labels)


def judge(metrics: dict, criteria: dict = CRITERIA) -> list[str]:
    """Failed pre-registered test/OOD criteria of one category (empty = ships)."""
    failed = []
    t = metrics["test"]
    if t["positives"] < criteria["min_test_positives"]:
        failed.append(
            f"test positives {t['positives']} < {criteria['min_test_positives']}"
        )
        return failed
    if t["auc"] < criteria["test_min_auc"]:
        failed.append(f"test AUC {t['auc']:.3f} < {criteria['test_min_auc']}")
    if t["precision"] < criteria["test_min_precision"]:
        failed.append(
            f"test precision {t['precision']:.3f} < {criteria['test_min_precision']}"
        )
    if t["recall"] < criteria["test_min_recall"]:
        failed.append(f"test recall {t['recall']:.3f} < {criteria['test_min_recall']}")
    if t["ece"] > criteria["test_max_ece"]:
        failed.append(f"test ECE {t['ece']:.3f} > {criteria['test_max_ece']}")
    ood = metrics.get("ood_openai")
    if (
        ood
        and ood["positives"] >= criteria["ood_min_positives"]
        and ood["auc"] < criteria["ood_min_auc"]
    ):
        failed.append(f"OpenAI-eval AUC {ood['auc']:.3f} < {criteria['ood_min_auc']}")
    return failed


def split_metrics(probs, labels, threshold) -> dict:
    tp, fp, fn = confusion(probs, labels, threshold)
    precision, recall = precision_recall(probs, labels, threshold)
    return {
        "rows": len(labels),
        "positives": sum(labels),
        "auc": roc_auc(probs, labels),
        "precision": precision,
        "precision_ci95": wilson(tp, tp + fp),
        "recall": recall,
        "recall_ci95": wilson(tp, tp + fn),
        "ece": ece(probs, labels, CRITERIA["ece_bins"]),
        "flagged": tp + fp,
    }


# ---------------------------------------------------------------- data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(data: Path, features: Path):
    import numpy as np

    rows = [
        json.loads(line) for line in (data / "dataset.jsonl").open(encoding="utf-8")
    ]
    vectors: dict[str, object] = {}
    payload = set()
    chunk_hashes = {}
    for meta_path in sorted(features.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        bin_path = meta_path.with_suffix(".f32")
        chunk_hashes[bin_path.name] = sha256_file(bin_path)
        payload.add(meta["embedding_payload_sha256"])
        if meta["prompt"] != PROMPT or meta["window_tokens"] != WINDOW_TOKENS:
            raise ValueError(f"{meta_path}: features use another prompt or window")
        matrix = np.fromfile(bin_path, dtype="<f4").reshape(-1, DIMENSIONS)
        if len(matrix) != len(meta["ids"]):
            raise ValueError(
                f"{bin_path}: {len(matrix)} vectors for {len(meta['ids'])} ids"
            )
        for rid, vector in zip(meta["ids"], matrix, strict=True):
            vectors[rid] = vector
    if len(payload) != 1:
        raise ValueError(f"features come from {len(payload)} encoder payloads")
    missing = [r["id"] for r in rows if r["id"] not in vectors]
    if missing:
        raise ValueError(
            f"{len(missing)} dataset rows have no features (first {missing[0]})"
        )
    return rows, vectors, payload.pop(), chunk_hashes


def matrix_for(rows, vectors, split, category):
    import numpy as np

    picked = [
        r for r in rows if r["split"] == split and r["labels"].get(category) is not None
    ]
    x = (
        np.stack([vectors[r["id"]] for r in picked])
        if picked
        else np.zeros((0, DIMENSIONS), "f4")
    )
    y = np.array([r["labels"][category] for r in picked], dtype=int)
    return x, y


def fit(args) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rows, vectors, payload, chunk_hashes = load(args.data, args.features)
    out: dict = {"categories": {}, "calibration": {}}
    for category in CATEGORIES:
        if category in NO_DATA:
            out["categories"][category] = {"unsupported": NO_DATA[category]}
            continue
        xt, yt = matrix_for(rows, vectors, "train", category)
        xc, yc = matrix_for(rows, vectors, "calibration", category)
        if (
            yt.sum() < CRITERIA["min_train_positives"]
            or yc.sum() < CRITERIA["min_calibration_positives"]
        ):
            reason = (
                f"too little data: {int(yt.sum())} train / {int(yc.sum())} calibration positives "
                f"(need {CRITERIA['min_train_positives']} / {CRITERIA['min_calibration_positives']})"
            )
            out["categories"][category] = {"unsupported": reason}
            continue
        best = None
        for c in C_GRID:
            model = LogisticRegression(C=c, max_iter=2000)
            model.fit(xt, yt)
            loss = log_loss(model.predict_proba(xc)[:, 1].tolist(), yc.tolist())
            if best is None or loss < best[0]:
                best = (loss, c, model)
        loss, c, model = best
        weight = model.coef_[0].astype(np.float32)
        bias = float(model.intercept_[0])
        logits = (xc.astype(np.float64) @ weight.astype(np.float64) + bias).tolist()
        a, b = fit_platt(logits, yc.tolist())
        probs = [sigmoid(a * z + b) for z in logits]
        threshold = choose_threshold(
            probs,
            yc.tolist(),
            CRITERIA["calibration_precision_target"],
            CRITERIA["calibration_min_recall"],
        )
        info = {
            "C": c,
            "train_rows": len(yt),
            "train_positives": int(yt.sum()),
            "calibration_log_loss_raw": loss,
            "platt": [a, b],
            "threshold": threshold,
        }
        if threshold is None:
            out["categories"][category] = {
                "unsupported": "no calibration threshold reaches precision "
                f"{CRITERIA['calibration_precision_target']} with recall >= {CRITERIA['calibration_min_recall']}"
            }
        else:
            info["calibration"] = split_metrics(probs, yc.tolist(), threshold)
            out["categories"][category] = {
                "weight": [float(w) for w in weight],
                "bias": bias,
                "platt_scale": a,
                "platt_offset": b,
                "threshold": threshold,
            }
        out["calibration"][category] = info
        print(
            category,
            json.dumps({k: v for k, v in info.items() if k != "calibration"}),
            flush=True,
        )
    out["embedding_payload_sha256"] = payload
    out["feature_chunks_sha256"] = chunk_hashes
    out["dataset_sha256"] = sha256_file(args.data / "dataset.jsonl")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidate.json").write_text(json.dumps(out) + "\n")
    return 0


def scores_for(head, x):
    import numpy as np

    w = np.array(head["weight"], dtype=np.float64)
    z = x.astype(np.float64) @ w + head["bias"]
    return [sigmoid(head["platt_scale"] * v + head["platt_offset"]) for v in z.tolist()]


def evaluate(args) -> int:
    import numpy as np

    report_path = args.out / "evaluation.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} exists: the test split is scored once")
    candidate = json.loads((args.out / "candidate.json").read_text())
    rows, vectors, payload, _ = load(args.data, args.features)
    if payload != candidate["embedding_payload_sha256"]:
        raise SystemExit("features and candidate come from different encoders")
    evaluation: dict = {"criteria": CRITERIA, "categories": {}}
    final = {}
    for category in CATEGORIES:
        entry = candidate["categories"][category]
        if "unsupported" in entry:
            final[category] = entry
            evaluation["categories"][category] = {
                "ships": False,
                "reason": entry["unsupported"],
            }
            continue
        metrics = {"calibration": candidate["calibration"][category]["calibration"]}
        for split in ("test", "ood_openai"):
            x, y = matrix_for(rows, vectors, split, category)
            if len(y):
                metrics[split] = split_metrics(
                    scores_for(entry, x), y.tolist(), entry["threshold"]
                )
        failed = judge(metrics)
        metrics["ships"] = not failed
        metrics["failed"] = failed
        evaluation["categories"][category] = metrics
        final[category] = (
            entry
            if not failed
            else {"unsupported": "failed evaluation: " + "; ".join(failed)}
        )
        print(category, "SHIPS" if not failed else f"FAILS {failed}", flush=True)
    # Report-only: OpenAI-eval categories with no head, and ToxicChat any-flag.
    shipped = [c for c in CATEGORIES if "weight" in final[c]]
    tc = [r for r in rows if r["split"] == "ood_toxicchat"]
    if tc and shipped:
        x = np.stack([vectors[r["id"]] for r in tc])
        per = [scores_for(final[c], x) for c in shipped]
        top = [max(col) for col in zip(*per, strict=True)]
        flagged = [
            float(
                any(
                    p[i] >= final[c]["threshold"]
                    for p, c in zip(per, shipped, strict=True)
                )
            )
            for i in range(len(tc))
        ]
        y = [r["labels"]["any"] for r in tc]
        evaluation["ood_toxicchat_any"] = {
            "rows": len(y),
            "positives": sum(y),
            "auc_max_score": roc_auc(top, y),
            "precision": precision_recall(flagged, y, 0.5)[0],
            "recall": precision_recall(flagged, y, 0.5)[1],
        }
    artifact = {
        "format": "local-ai-moderation-heads",
        "version": 1,
        "model": MODEL_ID,
        "embedding_model": "embeddinggemma-2",
        "embedding_payload_sha256": payload,
        "prompt": PROMPT,
        "window_tokens": WINDOW_TOKENS,
        "dimensions": DIMENSIONS,
        "categories": final,
        "provenance": {
            "dataset_sha256": candidate["dataset_sha256"],
            "dataset_manifest_sha256": sha256_file(args.data / "dataset-manifest.json"),
            "feature_chunks_sha256": candidate["feature_chunks_sha256"],
            "training": {
                c: {k: v for k, v in i.items() if k != "calibration"}
                for c, i in candidate["calibration"].items()
            },
            "sources": json.loads((args.data / "dataset-manifest.json").read_text())[
                "inputs"
            ]["sources"],
        },
        "evaluation": evaluation,
    }
    for source in artifact["provenance"]["sources"].values():
        source.pop("files_in_repo", None)
    report_path.write_text(json.dumps(evaluation, indent=1) + "\n")
    path = args.out / "moderation-heads.json"
    path.write_text(json.dumps(artifact, separators=(",", ":"), sort_keys=True) + "\n")
    print("artifact", path, sha256_file(path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fit", "evaluate"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return fit(args) if args.command == "fit" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
