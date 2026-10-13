"""Build the moderation training/evaluation set from pinned public datasets.

Input: the output directory of the private Kaggle CPU kernel that downloads the
public sources at pinned revisions (`manifest.json` plus the `*.jsonl.gz`
files it lists with SHA-256s; see docs/MODERATION.md for the sources). Every
file is checked against the manifest before anything is read.

Output (`--out DIR`):

    dataset.jsonl   {"id", "source", "split", "text", "labels": {category: 1 | 0 | null}}
                    one row per distinct (text, split); `null` = not annotated for that
                    category by the row's source (excluded from that category's training
                    and scoring).
    dataset-manifest.json   input hashes, mapping version, row and label counts, and the
                    SHA-256 of dataset.jsonl.

Splits: `train`, `calibration`, `test` (in-distribution: Aegis 2.0 test, validation -> calibration,
its train file 80/10/10 by id hash; Civil Comments 70/15/15 by row hash; Aegis 2.0 + Civil
Comments) and `ood_openai` (OpenAI's 2022 moderation evaluation set) and
`ood_toxicchat` (ToxicChat 0124 test, binary toxicity only, CC-BY-NC-4.0:
evaluation only, never trained on). A text is kept only in the most held-out
split it occurs in (OOD, then `test`, then `calibration`, then `train`), so no
evaluation or calibration text is fitted on.

The category mapping is explicit (`AEGIS_MAP`, `CIVIL_MAP`, `OPENAI_MAP`).
Standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
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
MAPPING_VERSION = 1

# Aegis 2.0 violated-category names -> rule per spec category. Each rule is a
# list of alternatives; an alternative is a set of Aegis categories that must
# all be present.
AEGIS_MAP: dict[str, list[set[str]]] = {
    "hate": [{"Hate/Identity Hate"}],
    "hate/threatening": [
        {"Hate/Identity Hate", "Threat"},
        {"Hate/Identity Hate", "Violence"},
    ],
    "harassment": [{"Harassment"}, {"Threat"}],
    "harassment/threatening": [{"Threat"}],
    "illicit": [
        {"Criminal Planning/Confessions"},
        {"Illegal Activity"},
        {"Controlled/Regulated Substances"},
        {"Fraud/Deception"},
        {"Malware"},
        {"Guns and Illegal Weapons"},
    ],
    "illicit/violent": [
        {"Guns and Illegal Weapons"},
        {"Criminal Planning/Confessions", "Violence"},
    ],
    "self-harm": [{"Suicide and Self Harm"}],
    "sexual": [{"Sexual"}, {"Sexual (minor)"}],
    "sexual/minors": [{"Sexual (minor)"}],
    "violence": [{"Violence"}, {"Threat"}],
}
# Rows whose only categories are these carry no category information.
AEGIS_UNINFORMATIVE = {"Needs Caution", "Other"}

# Civil Comments annotator fractions: positive at >= POS, negative below NEG,
# unannotated (null) in between.
POS, NEG = 0.5, 0.1
CIVIL_MAP: dict[str, list[str]] = {
    "hate": ["identity_attack"],
    "harassment": ["insult", "threat"],
    "harassment/threatening": ["threat"],
    "sexual": ["sexual_explicit"],
}
# Civil Comments has no annotation for these; news comments are assumed not to
# be in them (documented). sexual/minors is assumed negative only when the
# comment is clearly not sexual.
CIVIL_ASSUMED_NEGATIVE = ("illicit", "illicit/violent", "self-harm")

# OpenAI 2022 evaluation set keys (missing = not annotated).
OPENAI_MAP = {
    "hate": "H",
    "hate/threatening": "H2",
    "harassment": "HR",
    "self-harm": "SH",
    "sexual": "S",
    "sexual/minors": "S3",
    "violence": "V",
    "violence/graphic": "V2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bucket(key: str) -> int:
    """Deterministic 0..99 bucket of a string."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100


def empty_labels() -> dict[str, int | None]:
    return dict.fromkeys(CATEGORIES)


def aegis_labels(categories: set[str]) -> dict[str, int | None]:
    labels = empty_labels()
    for name, rule in AEGIS_MAP.items():
        labels[name] = int(any(alt <= categories for alt in rule))
    return labels


def aegis_rows(rows: list[dict], split_of) -> list[dict]:
    """Rows of one Aegis 2.0 file as (text, labels) items.

    Prompt-only rows: the prompt with the violated categories (unsafe) or all
    negative (safe). Rows with a safe response: the response is all negative,
    and the prompt carries the categories (the response being safe, they
    describe the prompt). Rows with both unsafe: prompt and response together,
    since the categories may come from either. Rows whose categories are only
    uninformative (`Needs Caution`, `Other`) are skipped.
    """
    out = []
    for row in rows:
        cats = {
            c.strip()
            for c in (row.get("violated_categories") or "").split(",")
            if c.strip()
        }
        prompt = row.get("prompt") or ""
        response = row.get("response") or ""
        split = split_of(row)
        unsafe_prompt = row.get("prompt_label") == "unsafe"
        unsafe_response = row.get("response_label") == "unsafe"
        informative = bool(cats - AEGIS_UNINFORMATIVE)
        if response and not unsafe_response:
            out.append((response, empty_negative(), split, row["id"] + ":response"))
        if unsafe_prompt or unsafe_response:
            if not informative:
                continue
            text = (
                f"{prompt}\n\n{response}" if (response and unsafe_response) else prompt
            )
            out.append((text, aegis_labels(cats), split, row["id"]))
        else:
            out.append((prompt, empty_negative(), split, row["id"]))
    return out


def empty_negative() -> dict[str, int | None]:
    labels = empty_labels()
    for name in AEGIS_MAP:
        labels[name] = 0
    return labels


def civil_labels(row: dict) -> dict[str, int | None]:
    labels = empty_labels()
    for name, columns in CIVIL_MAP.items():
        top = max(float(row[c]) for c in columns)
        labels[name] = 1 if top >= POS else (0 if top < NEG else None)
    ia, th = float(row["identity_attack"]), float(row["threat"])
    if ia >= POS and th >= POS:
        labels["hate/threatening"] = 1
    elif ia < NEG or th < NEG:
        labels["hate/threatening"] = 0
    for name in CIVIL_ASSUMED_NEGATIVE:
        labels[name] = 0
    labels["sexual/minors"] = 0 if float(row["sexual_explicit"]) < NEG else None
    return labels


def openai_labels(row: dict) -> dict[str, int | None]:
    labels = empty_labels()
    for name, key in OPENAI_MAP.items():
        value = row.get(key)
        labels[name] = None if value is None else int(value)
    return labels


def verify_inputs(raw: Path) -> dict:
    manifest = json.loads((raw / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        actual = sha256_file(raw / name)
        if actual != digest:
            raise ValueError(f"{name}: sha256 {actual} != manifest {digest}")
    return manifest


def build(raw: Path) -> tuple[list[dict], dict]:
    manifest = verify_inputs(raw)
    items: list[tuple[str, dict, str, str, str]] = []

    def aegis_split(name):
        if name == "test":
            return lambda row: "test"
        if name == "validation":
            return lambda row: "calibration"
        return lambda row: ("calibration", "test", *["train"] * 8)[
            bucket("aegis:" + row["id"]) // 10
        ]

    for name in ("train", "validation", "test"):
        rows = read_gz(raw / f"aegis2_{name}.jsonl.gz")
        for text, labels, split, rid in aegis_rows(rows, aegis_split(name)):
            items.append((text, labels, split, "aegis2", f"aegis2:{name}:{rid}"))
    for row in read_gz(raw / "civil_comments_sample.jsonl.gz"):
        b = bucket(f"civil:{row['row']}")
        split = "train" if b < 70 else ("calibration" if b < 85 else "test")
        items.append(
            (
                row["text"],
                civil_labels(row),
                split,
                "civil_comments",
                f"civil:{row['row']}",
            )
        )
    for index, row in enumerate(read_gz(raw / "openai_eval.jsonl.gz")):
        items.append(
            (
                row["prompt"],
                openai_labels(row),
                "ood_openai",
                "openai_eval",
                f"openai:{index}",
            )
        )
    for row in read_gz(raw / "toxicchat_toxic-chat_annotation_test.jsonl.gz"):
        labels = empty_labels()
        labels["any"] = int(row["toxicity"])
        items.append(
            (
                row["user_input"],
                labels,
                "ood_toxicchat",
                "toxicchat",
                f"toxicchat:{row['conv_id']}",
            )
        )

    # Leakage control: an evaluation text never appears in a fitting split.
    rank = {
        "ood_openai": 0,
        "ood_toxicchat": 0,
        "test": 1,
        "calibration": 2,
        "train": 3,
    }
    best: dict[str, int] = {}
    for text, _, split, _, _ in items:
        key = text.strip()
        best[key] = min(best.get(key, 9), rank[split])
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    dropped = 0
    for text, labels, split, source, rid in items:
        key = text.strip()
        if not key:
            dropped += 1
            continue
        if rank[split] != best[key]:
            dropped += 1
            continue
        if (key, split) in seen:
            dropped += 1
            continue
        seen.add((key, split))
        rows.append(
            {
                "id": rid,
                "source": source,
                "split": split,
                "text": text,
                "labels": labels,
            }
        )
    counts: dict = {}
    for row in rows:
        per = counts.setdefault(row["split"], {"rows": 0})
        per["rows"] += 1
        for name, value in row["labels"].items():
            if value is not None:
                c = per.setdefault(name, [0, 0])
                c[value] += 1
    meta = {
        "mapping_version": MAPPING_VERSION,
        "inputs": manifest,
        "dropped_duplicates_or_leaks": dropped,
        "counts_neg_pos": counts,
    }
    return rows, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, meta = build(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    meta["dataset_sha256"] = sha256_file(path)
    (args.out / "dataset-manifest.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n"
    )
    json.dump(meta["counts_neg_pos"], sys.stdout, indent=1, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
