"""Prepare Kev devtools-v1 judgment rows for `mtp-capture features` (no model, no GPU).

Reads locally supplied, pinned Kev `devtools-v1` files (never downloads):

    KEV_DIR/manifest.json       sha256 must equal PINNED_MANIFEST_SHA256
    KEV_DIR/train.jsonl         Kev partition train       -> train + calibration
    KEV_DIR/development.jsonl   Kev partition development -> validation
    KEV_DIR/test.jsonl          Kev partition test        -> test (locked)

Every file's sha256, byte size, record count and per-source count must match the
manifest; every record is schema-checked, including records that are excluded.

Only the audited coding questions with native labels are converted (no LLM
relabeling): CodeReviewer `needs_comment` and CommitPackFT `message_match`.
FlakeFlagger and commit-change-type are excluded (Kev's later label audit: the
input does not determine the label), Aegis is outside the coding focus, and
When2Call / prompt-injection are evaluation-only. Every excluded question row is
counted in the report by reason; nothing is dropped silently.

Calibration is carved from Kev's train partition by whole `group_id`, in a
seeded hash order, per source. Development becomes validation and test stays
the locked test split. Record (`row_sha256`), group, state text (exact and
whitespace/case-normalised) and rendered-text overlap across splits is an error.

Each output row is the capture input contract of `mtp-capture features`:

    {"id": "kev-devtools-v1/<task>/<row_sha256>", "text": <rendered prompt>,
     "token_end_offsets": [<option 1 end>, ..., <decision end>],
     "metadata": {"split", "target", "options", "option_labels", "native_label",
                  "task", "source", "question", "kev_id", "group_id",
                  "kev_partition", "row_sha256", "text_sha256", "repo_licence",
                  "renderer"}}

Rendered text (RENDERER):

    State:\\n<state>\\n\\nQuestion: <instructions>\\nOptions:\\n
    A) <option>\\n B) <option>\\n ... Decision:

Option order is a per-row deterministic permutation (sha256 of seed, record and
option); `target` indexes the gold option in that order. Offsets are exclusive
UTF-8 byte ends: each option endpoint includes its line's "\\n", the decision
endpoint is the end of "Decision:". The native Bonsai tokenizer in
`mtp-capture` resolves them to exact token boundaries, or rejects the row.

`--pilot-rows N` keeps, per split and task, whole groups in a seeded hash order
while they fit in N rows; the rest are reported as `pilot_cap`. A pilot only
exercises the pipeline; it is not evidence of quality.

Output: a new directory (never overwritten) with `rows.jsonl` and `report.json`
(provenance, hashes, licences/attribution, counts, exclusions, caveats).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

KEV_REPOSITORY = "https://github.com/jaredpalmer/kev"
KEV_REVISION = "5e42a7a03f28134853dd3ff77461457e921e5ec1"
KEV_BASE_URL = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_REVISION}/evals/devtools-v1"
PINNED_MANIFEST_SHA256 = (
    "d0f7cf40cec3284684f8b2fe8f440a5f0cc40fe4a78db108eca47a5fbd141b0f"
)
KEV_VERSION = "devtools-v1"
FORMAT = "local-ai.judgment-capture-rows.v1"
RENDERER = "kev-devtools-v1-judgment-render.v1"
DEFAULT_SEED = "local-ai-judgment-v1"

# Kev partition file -> output split. Calibration is carved from train.
PARTITIONS = {
    "train": ("train.jsonl", "train"),
    "development": ("development.jsonl", "validation"),
    "test": ("test.jsonl", "test"),
}
SPLIT_ORDER = ("train", "calibration", "validation", "test")

# Converted questions, keyed by Kev question `src`.
TASKS = {
    "codereviewer_needs_comment": {
        "source": "codereviewer",
        "question": "needs_comment",
    },
    "commitpackft_message": {"source": "commitpackft", "question": "message_match"},
}
# Every other known question `src`, with the reason it is not converted.
EXCLUDED = {
    "commitpackft_type": "label_audit_input_does_not_determine_label",
    "flakeflagger_flaky": "label_audit_input_does_not_determine_label",
    "aegis_unsafe": "outside_coding_focus",
    "aegis_category": "outside_coding_focus",
    "when2call_action": "evaluation_only",
    "prompt_injection": "evaluation_only",
}
KNOWN_SOURCES = {
    "codereviewer",
    "commitpackft",
    "flakeflagger",
    "aegis",
    "when2call",
    "prompt_injection",
}
NOUL_OPTIONS = (("Yes", True), ("No", False))
RECORD_KEYS = {"state", "questions", "_meta"}
META_REQUIRED = {
    "id",
    "source",
    "group_id",
    "row_sha256",
    "text_sha256",
    "split",
    "variant",
}
NOUL_KEYS = {"type", "instructions", "label", "src"}
HEX64 = re.compile(r"[0-9a-f]{64}")
CAVEATS = [
    "Preparation does not run training; these rows alone do not establish quality.",
    (
        "Labels are Kev's native labels: CodeReviewer is human; CommitPackFT message "
        "match is by construction (own subject vs another commit's subject)."
    ),
    "CommitPackFT negative-subject reuse across splits has not been audited.",
    "Kev balanced labels by sampling; rates are not natural rates.",
    "Kev's CodeReviewer `id` is not unique; `row_sha256` identifies records.",
    "Frozen Bonsai features may not match Kev's adapter-trained representations.",
    "A small pilot only exercises the pipeline; its metrics prove nothing.",
    "Repository licences were recorded by Kev at lookup time, not collection time.",
]


class PrepareError(ValueError):
    """The supplied Kev files or options cannot produce a valid dataset."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_key(*parts: str) -> str:
    return sha256_hex("\0".join(parts).encode("utf-8"))


def _strict_object(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise PrepareError(f"duplicate JSON key in {keys}")
    return dict(pairs)


def load_manifest(kev_dir: Path, expected_sha256: str) -> tuple[dict, str]:
    path = kev_dir / "manifest.json"
    data = path.read_bytes()
    digest = sha256_hex(data)
    if digest != expected_sha256:
        raise PrepareError(
            f"manifest.json sha256 {digest} does not match pinned {expected_sha256}"
        )
    manifest = json.loads(data, object_pairs_hook=_strict_object)
    if manifest.get("version") != KEV_VERSION:
        raise PrepareError(f"unsupported manifest version {manifest.get('version')!r}")
    if manifest.get("partitions") != list(PARTITIONS):
        raise PrepareError(f"unsupported partitions {manifest.get('partitions')!r}")
    if "test" not in manifest.get("locked", []):
        raise PrepareError("manifest does not lock the test partition")
    eval_only = set(manifest.get("eval_only_sources", []))
    for task, spec in TASKS.items():
        if spec["source"] in eval_only:
            raise PrepareError(f"{task}: source {spec['source']} is evaluation-only")
        source = manifest.get("sources", {}).get(spec["source"])
        if not isinstance(source, dict) or not source.get("trainable"):
            raise PrepareError(f"manifest has no trainable source {spec['source']!r}")
        for key in ("licence", "licence_url", "attribution", "label_provenance"):
            if not isinstance(source.get(key), str) or not source[key]:
                raise PrepareError(f"manifest source {spec['source']} lacks {key}")
    return manifest, digest


def load_partition(kev_dir: Path, partition: str, manifest: dict) -> list[dict]:
    name = PARTITIONS[partition][0]
    entry = manifest.get("files", {}).get(name)
    if not isinstance(entry, dict):
        raise PrepareError(f"manifest has no files entry for {name}")
    data = (kev_dir / name).read_bytes()
    digest = sha256_hex(data)
    if digest != entry.get("sha256"):
        raise PrepareError(
            f"{name}: sha256 {digest} does not match manifest {entry.get('sha256')}"
        )
    if len(data) != entry.get("bytes"):
        raise PrepareError(
            f"{name}: {len(data)} bytes, manifest says {entry.get('bytes')}"
        )
    lines = data.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    records = []
    for number, line in enumerate(lines, 1):
        where = f"{name}:{number}"
        try:
            record = json.loads(line, object_pairs_hook=_strict_object)
        except (json.JSONDecodeError, PrepareError) as error:
            raise PrepareError(f"{where}: invalid JSON record: {error}") from error
        check_record(record, partition, where)
        records.append(record)
    if len(records) != entry.get("records"):
        raise PrepareError(
            f"{name}: {len(records)} records, manifest says {entry.get('records')}"
        )
    by_source = dict(Counter(r["_meta"]["source"] for r in records))
    if by_source != entry.get("by_source"):
        raise PrepareError(
            f"{name}: per-source counts {by_source} != manifest {entry.get('by_source')}"
        )
    return records


def check_record(record, partition: str, where: str) -> None:
    """Validate the common schema of every record and the full schema of converted ones."""
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise PrepareError(
            f"{where}: record must have exactly the keys {sorted(RECORD_KEYS)}"
        )
    meta = record["_meta"]
    if not isinstance(meta, dict) or not META_REQUIRED <= set(meta):
        raise PrepareError(f"{where}: _meta must include {sorted(META_REQUIRED)}")
    for key in ("id", "source", "group_id"):
        if not isinstance(meta[key], str) or not meta[key]:
            raise PrepareError(f"{where}: _meta.{key} must be a non-empty string")
    for key in ("row_sha256", "text_sha256"):
        if not isinstance(meta[key], str) or not HEX64.fullmatch(meta[key]):
            raise PrepareError(f"{where}: _meta.{key} must be a sha256 hex digest")
    if meta["source"] not in KNOWN_SOURCES:
        raise PrepareError(f"{where}: unsupported source {meta['source']!r}")
    if meta["split"] != partition:
        raise PrepareError(
            f"{where}: _meta.split {meta['split']!r} in partition {partition}"
        )
    if meta["variant"] != "clean":
        raise PrepareError(f"{where}: unsupported variant {meta['variant']!r}")
    questions = record["questions"]
    if not isinstance(questions, dict) or not questions:
        raise PrepareError(f"{where}: questions must be a non-empty object")
    for name, question in questions.items():
        src = question.get("src") if isinstance(question, dict) else None
        if src in TASKS:
            spec = TASKS[src]
            if (meta["source"], name) != (spec["source"], spec["question"]):
                raise PrepareError(
                    f"{where}: question {name!r} with src {src!r} "
                    f"in source {meta['source']!r}"
                )
            if set(question) != NOUL_KEYS or question["type"] != "noul":
                raise PrepareError(
                    f"{where}: {src} must be a noul question with keys "
                    f"{sorted(NOUL_KEYS)}"
                )
            if (
                not isinstance(question["instructions"], str)
                or not question["instructions"]
            ):
                raise PrepareError(
                    f"{where}: {src} instructions must be non-empty text"
                )
            if type(question["label"]) is not bool:
                raise PrepareError(f"{where}: {src} label must be a boolean")
            render_state(meta["source"], record["state"], where)
        elif src not in EXCLUDED:
            raise PrepareError(f"{where}: unsupported question {name!r} (src {src!r})")


def render_state(source: str, state, where: str) -> str:
    if source == "codereviewer":
        if (
            not isinstance(state, dict)
            or "diff" not in state
            or not set(state) <= {"diff", "lines_before_hunk"}
            or not all(isinstance(v, str) for v in state.values())
        ):
            raise PrepareError(f"{where}: unsupported codereviewer state schema")
        parts = []
        if "lines_before_hunk" in state:
            parts.append("Context before hunk:\n" + state["lines_before_hunk"] + "\n\n")
        parts.append("Diff:\n" + state["diff"])
        return "".join(parts)
    if source == "commitpackft":
        if not isinstance(state, str) or not state.strip():
            raise PrepareError(f"{where}: unsupported commitpackft state schema")
        return "Diff:\n" + state
    raise PrepareError(f"{where}: no renderer for source {source!r}")


def permute_options(options, seed: str, record_key: str):
    """Deterministic per-row order: options sorted by sha256(seed, record, option)."""
    return sorted(options, key=lambda option: hash_key(seed, record_key, option[0]))


def render(state_text: str, instructions: str, options) -> tuple[str, list[int]]:
    """Return (text, exclusive UTF-8 byte ends of each option line and of the cue)."""
    pieces = ["State:\n", state_text, "\n\nQuestion: ", instructions, "\nOptions:\n"]
    size = sum(len(piece.encode("utf-8")) for piece in pieces)
    ends = []
    for index, (label, _) in enumerate(options):
        line = f"{chr(ord('A') + index)}) {label}\n"
        pieces.append(line)
        size += len(line.encode("utf-8"))
        ends.append(size)
    pieces.append("Decision:")
    text = "".join(pieces)
    ends.append(len(text.encode("utf-8")))
    return text, ends


def normalized_text_hash(text: str) -> str:
    return sha256_hex(" ".join(text.casefold().split()).encode("utf-8"))


def candidates(records_by_partition, seed: str, excluded):
    """Yield one candidate row per converted question; count every other question."""
    seen_rows = {}
    for partition, records in records_by_partition.items():
        for record in records:
            meta = record["_meta"]
            if meta["row_sha256"] in seen_rows:
                raise PrepareError(
                    f"row_sha256 {meta['row_sha256']} appears in "
                    f"{seen_rows[meta['row_sha256']]} and {partition}"
                )
            seen_rows[meta["row_sha256"]] = partition
            converted = 0
            for question in record["questions"].values():
                src = question["src"]
                if src not in TASKS:
                    excluded[EXCLUDED[src]][partition][src] += 1
                    continue
                converted += 1
                state_text = render_state(meta["source"], record["state"], meta["id"])
                key = f"{meta['row_sha256']}/{src}"
                options = permute_options(NOUL_OPTIONS, seed, key)
                text, ends = render(state_text, question["instructions"], options)
                labels = [value for _, value in options]
                yield {
                    "id": f"kev-{KEV_VERSION}/{src}/{meta['row_sha256']}",
                    "text": text,
                    "token_end_offsets": ends,
                    "metadata": {
                        "split": PARTITIONS[partition][1],
                        "target": labels.index(question["label"]),
                        "options": [label for label, _ in options],
                        "option_labels": labels,
                        "native_label": question["label"],
                        "task": src,
                        "source": meta["source"],
                        "question": TASKS[src]["question"],
                        "kev_id": meta["id"],
                        "group_id": meta["group_id"],
                        "kev_partition": partition,
                        "row_sha256": meta["row_sha256"],
                        "text_sha256": meta["text_sha256"],
                        "repo_licence": meta.get("repo_licence"),
                        "renderer": RENDERER,
                    },
                    "_state": json.dumps(
                        record["state"], sort_keys=True, ensure_ascii=False
                    ),
                    "_state_text": state_text,
                }
            if converted == 0 and meta["source"] in {
                t["source"] for t in TASKS.values()
            }:
                excluded["no_supported_question"][partition][meta["source"]] += 1


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["metadata"]["group_id"]].append(row)
    return groups


def carve_calibration(rows, seed: str, fraction: float, min_groups: int) -> None:
    """Move whole train groups to calibration, per task, in seeded hash order."""
    by_task = defaultdict(list)
    for row in rows:
        if row["metadata"]["split"] == "train":
            by_task[row["metadata"]["task"]].append(row)
    for task, task_rows in sorted(by_task.items()):
        groups = _groups(task_rows)
        if len(groups) < 2 * min_groups:
            raise PrepareError(
                f"{task}: {len(groups)} train groups; need {2 * min_groups} "
                "to carve group-disjoint calibration"
            )
        wanted = math.ceil(fraction * len(task_rows))
        taken = 0
        order = sorted(groups, key=lambda g: hash_key(seed, "calibration", g))
        for chosen, group in enumerate(order):
            if taken >= wanted and chosen >= min_groups:
                break
            if len(groups) - chosen <= min_groups:
                raise PrepareError(
                    f"{task}: calibration would leave fewer than "
                    f"{min_groups} train groups"
                )
            for row in groups[group]:
                row["metadata"]["split"] = "calibration"
            taken += len(groups[group])


def apply_pilot_cap(rows, seed: str, cap: int, excluded) -> list[dict]:
    """Keep whole groups per (split, task) in seeded hash order while they fit in `cap`."""
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[(row["metadata"]["split"], row["metadata"]["task"])].append(row)
    kept = []
    for (split, task), cell in sorted(by_cell.items()):
        groups = _groups(cell)
        used = 0
        for group in sorted(groups, key=lambda g: hash_key(seed, "pilot", split, g)):
            members = groups[group]
            if used + len(members) <= cap:
                kept.extend(members)
                used += len(members)
            else:
                excluded["pilot_cap"][split][task] += len(members)
    return kept


def check_splits(rows, min_groups: int) -> dict:
    """Leakage, uniqueness, group-count and label-diversity checks; return counts."""
    owners = {
        kind: {}
        for kind in (
            "record",
            "group",
            "state",
            "normalized_state",
            "kev_text_sha256",
            "text",
            "id",
        )
    }
    for row in rows:
        meta = row["metadata"]
        keys = {
            "record": meta["row_sha256"] + "/" + meta["task"],
            "group": meta["group_id"],
            "state": sha256_hex(row["_state"].encode("utf-8")),
            "normalized_state": normalized_text_hash(row["_state_text"]),
            "kev_text_sha256": meta["text_sha256"],
            "text": sha256_hex(row["text"].encode("utf-8")),
            "id": row["id"],
        }
        for kind, key in keys.items():
            other = owners[kind].setdefault(key, row)
            if other is row:
                continue
            same_split = other["metadata"]["split"] == meta["split"]
            if kind == "group" and same_split:
                continue
            if (
                kind in ("state", "normalized_state", "kev_text_sha256")
                and same_split
                and (other["metadata"]["row_sha256"] == meta["row_sha256"])
            ):
                continue  # several questions of one record
            raise PrepareError(
                f"{kind} overlap between {other['id']} "
                f"({other['metadata']['split']}) and {row['id']} ({meta['split']})"
            )
    counts = {}
    for split in SPLIT_ORDER:
        counts[split] = {}
        for task in TASKS:
            cell = [
                r
                for r in rows
                if r["metadata"]["split"] == split and r["metadata"]["task"] == task
            ]
            groups = {r["metadata"]["group_id"] for r in cell}
            labels = Counter(str(r["metadata"]["native_label"]).lower() for r in cell)
            if len(groups) < min_groups:
                raise PrepareError(
                    f"{split}/{task}: {len(groups)} groups, need {min_groups}"
                )
            if len(labels) < 2:
                raise PrepareError(
                    f"{split}/{task}: only labels {dict(labels)}; need both"
                )
            counts[split][task] = {
                "rows": len(cell),
                "groups": len(groups),
                "native_labels": dict(sorted(labels.items())),
                "targets": dict(
                    sorted(Counter(str(r["metadata"]["target"]) for r in cell).items())
                ),
            }
    return counts


def prepare(
    kev_dir: Path,
    out: Path,
    *,
    seed: str = DEFAULT_SEED,
    calibration_fraction: float = 0.1,
    pilot_rows: int | None = None,
    min_groups: int = 2,
    manifest_sha256: str = PINNED_MANIFEST_SHA256,
) -> dict:
    """Validate, convert and write `out/rows.jsonl` and `out/report.json`."""
    kev_dir, out = Path(kev_dir), Path(out)
    if out.exists() or out.is_symlink():
        raise PrepareError(f"{out} already exists; refusing to overwrite")
    if not 0 < calibration_fraction < 1:
        raise PrepareError("calibration fraction must be in (0, 1)")
    if min_groups < 1 or (pilot_rows is not None and pilot_rows < 1):
        raise PrepareError("min groups and pilot rows must be positive")
    manifest, manifest_digest = load_manifest(kev_dir, manifest_sha256)
    records = {p: load_partition(kev_dir, p, manifest) for p in PARTITIONS}
    excluded = defaultdict(lambda: defaultdict(Counter))
    rows = list(candidates(records, seed, excluded))
    carve_calibration(rows, seed, calibration_fraction, min_groups)
    if pilot_rows is not None:
        rows = apply_pilot_cap(rows, seed, pilot_rows, excluded)
    counts = check_splits(rows, min_groups)
    rows.sort(
        key=lambda r: (
            SPLIT_ORDER.index(r["metadata"]["split"]),
            hash_key(seed, "order", r["id"]),
        )
    )
    payload = "".join(
        json.dumps(
            {k: r[k] for k in ("id", "text", "token_end_offsets", "metadata")},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for r in rows
    ).encode("utf-8")
    kev_ids = Counter(r["metadata"]["kev_id"] for r in rows)
    report = {
        "format": FORMAT,
        "renderer": RENDERER,
        "options": {
            "seed": seed,
            "calibration_fraction": calibration_fraction,
            "pilot_rows": pilot_rows,
            "min_groups": min_groups,
        },
        "kev": {
            "repository": KEV_REPOSITORY,
            "revision": KEV_REVISION,
            "manifest_url": f"{KEV_BASE_URL}/manifest.json",
            "manifest_sha256": manifest_digest,
            "manifest_pinned": manifest_digest == PINNED_MANIFEST_SHA256,
            "version": manifest["version"],
            "code_licence": "Apache-2.0 (Kev repository); data per source below",
            "files": {
                name: {
                    k: manifest["files"][name][k]
                    for k in ("sha256", "bytes", "records")
                }
                for name, _ in PARTITIONS.values()
            },
            "label_protocol": manifest.get("label_protocol"),
        },
        "split_map": {
            "train": "train minus calibration groups",
            "calibration": "whole train groups, seeded hash order",
            "validation": "development",
            "test": "test (locked)",
        },
        "tasks": {
            task: {
                **spec,
                **{
                    k: manifest["sources"][spec["source"]].get(k)
                    for k in (
                        "licence",
                        "licence_url",
                        "attribution",
                        "label_provenance",
                        "licence_check",
                    )
                },
                "repo_licences": dict(
                    sorted(
                        Counter(
                            str(r["metadata"]["repo_licence"])
                            for r in rows
                            if r["metadata"]["task"] == task
                        ).items()
                    )
                ),
            }
            for task, spec in TASKS.items()
        },
        "excluded_questions": EXCLUDED,
        "counts": counts,
        "excluded_rows": {
            reason: {
                part: dict(sorted(c.items())) for part, c in sorted(by_part.items())
            }
            for reason, by_part in sorted(excluded.items())
        },
        "repeated_kev_ids": sum(n > 1 for n in kev_ids.values()),
        "output": {
            "rows": "rows.jsonl",
            "row_count": len(rows),
            "sha256": sha256_hex(payload),
        },
        "caveats": CAVEATS,
    }
    out.mkdir(parents=True)
    with open(out / "rows.jsonl", "xb") as handle:
        handle.write(payload)
    with open(out / "report.json", "x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--kev-dir",
        type=Path,
        required=True,
        help="directory with manifest.json, train/development/test.jsonl",
    )
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument(
        "--pilot-rows",
        type=int,
        default=None,
        help="per split and task, keep whole groups up to N rows",
    )
    parser.add_argument(
        "--min-groups", type=int, default=2, help="minimum groups per split and task"
    )
    parser.add_argument(
        "--manifest-sha256",
        default=PINNED_MANIFEST_SHA256,
        help="override only for test fixtures; defaults to the pin",
    )
    args = parser.parse_args(argv)
    try:
        report = prepare(
            args.kev_dir,
            args.out,
            seed=args.seed,
            calibration_fraction=args.calibration_fraction,
            pilot_rows=args.pilot_rows,
            min_groups=args.min_groups,
            manifest_sha256=args.manifest_sha256,
        )
    except (PrepareError, OSError, UnicodeDecodeError) as error:
        print(f"judgment_prepare: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps({"counts": report["counts"], "output": report["output"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
