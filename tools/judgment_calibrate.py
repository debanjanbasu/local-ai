"""EXPERIMENTAL judgment-head evaluation, calibration and refusal (no training).

Builds evaluation/calibration requests from pinned public data, captures them
through the serving path (`judgment-capture`, i.e. `Engine::capture_decision`),
scores features with an exported pointer head, fits calibration on a
calibration split only, chooses a refusal threshold for a stated target risk,
fits an out-of-scope feature gate, and reports held-out and out-of-domain
metrics. Nothing here trains a head or claims production quality, MTP gains or
coding quality. See docs/DECISIONS_EVALUATION.md for the pre-registered
protocol this implements.

Subcommands:

    build-codereviewer  new CodeReviewer `needs_comment` requests from Zenodo
                        6900648 cls-valid/cls-test rows of the projects Kev
                        devtools-v1 put in development (-> calibration) and
                        test (-> test), excluding every hunk Kev used
    build-ood           Kev devtools-v1 development/test rows outside the
                        devtools head's scope (prompt_injection, aegis_unsafe,
                        flakeflagger_flaky) as native predicate requests
    rerender            native requests for prepared text rows (byte-checked)
    capture             run `judgment-capture` over requests in bounded chunks,
                        each under `lockf -k /tmp/local-ai-gpu.lock`, resumable
                        per chunk; merges chunks into one feature directory
    score               raw logits (logit x exported temperature) and the
                        head's query projection for every feature row (NumPy,
                        imported lazily: `uv run --with numpy`)
    evaluate            calibration fit on CAL, threshold, feature gate,
                        metrics on CAL/TEST/OOD; writes report.json/report.md
    parity              compare two feature directories bit for bit by id

Requests are `judgment-capture` input rows; features are the
`mtp-capture features` format `judgment_train.py` reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judgment_prepare as prep
import judgment_train as jt

SEED = "local-ai-decisions-v1"
TARGET_RISK = 0.10
MIN_ANSWERED = 50
GATE_PERCENTILE = 0.99
CV_FOLDS = 5
ECE_BINS = 15
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
LOCK = "/tmp/local-ai-gpu.lock"

CR_QUESTION = "Does this change need a reviewer comment?"
CR_TASK = "codereviewer_needs_comment"
CM_TASK = "commitpackft_message"
CM_QUESTION = re.compile(r'Does this commit message describe this diff\? Message: "[^\n]+"')
ZENODO_RECORD = 6900648
ZENODO_FILE = "Diff_Quality_Estimation.zip"
ZENODO_SHA256 = "86d054de47741cb358c8a17ab55b6191356fc63e44010c864dd790481f41fff5"
ZENODO_MEMBERS = {  # sha256 of each extracted member (CRC32 also checked on extraction)
    "cls-valid.jsonl": "ca2725be072cb37c92979e83835d7b0b73f3631a102fc644b94d1ca84f87ad33",
    "cls-test.jsonl": "de62a2cea9f6d746b658266af5fa45debaa0f8e442c35f7a2849cf1c547d5d9c",
}
KEV_LICENCES_URL = ("https://raw.githubusercontent.com/jaredpalmer/kev/"
                    "62c91838b9a6adc5b386cbeae8ed73daa36ce220/scripts/devtools_v1_licences.json")
KEV_LICENCES_SHA256 = "75b45f29fe572f8cac0bcc97901b0a516797b9c4e32a281d867130b103abe3d9"
PERMISSIVE = ("mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense")
CONTEXT_LINES, CONTEXT_CHARS, MAX_PATCH_CHARS = 10, 1200, 6000
MAX_STATE_CHARS = 4200  # Kev devtools-v1's longest CodeReviewer state (JSON-escaped)
HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
OOD_SOURCES = {  # Kev source -> (question key, task)
    "prompt_injection": ("injection", "prompt_injection"),
    "aegis": ("unsafe", "aegis_unsafe"),
    "flakeflagger": ("flaky", "flakeflagger_flaky"),
}


class CalibrateError(ValueError):
    """Inputs do not match what this tool requires."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def text_digest(text: str) -> str:
    """Kev's `text_sha256`: sha256 of casefolded, whitespace-collapsed text."""
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def order_key(*parts: str) -> str:
    return hashlib.sha256(":".join((SEED,) + parts).encode()).hexdigest()


def half_of(group: str) -> str:
    """Deterministic group-level half ('a' or 'b') for splitting one split in two."""
    return "a" if int(order_key("half", group)[:8], 16) % 2 == 0 else "b"


def fold_of(group: str, folds: int = CV_FOLDS) -> int:
    return int(order_key("fold", group)[:8], 16) % folds


# ---- datasets ---------------------------------------------------------------


def codereviewer_state(patch: str, oldf: str | None) -> dict:
    """Kev's `codereviewer_state`: the hunk plus up to 10 old-file lines before it."""
    state = {}
    match = HUNK.match(patch)
    if match and oldf:
        start = int(match.group(1)) - 1
        before = oldf.split("\n")[max(0, start - CONTEXT_LINES):max(0, start)]
        while before and len("\n".join(before)) > CONTEXT_CHARS:
            before = before[1:]
        if any(line.strip() for line in before):
            state["lines_before_hunk"] = "\n".join(before)
    state["diff"] = patch
    return state


def request_row(row_id, state_text, question, label, metadata, kind="predicate"):
    """A `judgment-capture` predicate request (rendered `A) Yes`, `B) No`)."""
    if kind != "predicate":
        raise CalibrateError("only predicate requests are built")
    meta = dict(metadata)
    meta.update(target=0 if label else 1, native_label=bool(label), options=["Yes", "No"],
                option_labels=[True, False], question_text=question,
                state_prefix=state_text[:200])
    return {"id": row_id, "state": state_text, "question": question, "kind": "predicate",
            "metadata": meta}


def kev_records(kev_dir: Path) -> dict[str, list[dict]]:
    manifest, _ = prep.load_manifest(kev_dir, prep.PINNED_MANIFEST_SHA256)
    return {part: prep.load_partition(kev_dir, part, manifest)
            for part in ("train", "development", "test")}


def select_pairs(rows, limit: int, seed_label: str) -> list[dict]:
    """Up to `limit` rows, label-balanced in pairs within project, projects round-robin
    in a seeded order (Kev's `balanced_pairs` scheme)."""
    strata = defaultdict(lambda: ([], []))
    for row in sorted(rows, key=lambda r: order_key(seed_label, r["id"])):
        strata[row["project"]][0 if row["label"] else 1].append(row)
    keys = sorted(strata, key=lambda k: order_key(seed_label, "project", k))
    cursors = {k: 0 for k in keys}
    out, live = [], list(keys)
    while live and len(out) + 2 <= limit:
        for key in list(live):
            if len(out) + 2 > limit:
                break
            trues, falses = strata[key]
            i = cursors[key]
            if i >= len(trues) or i >= len(falses):
                live.remove(key)
                continue
            out += [trues[i], falses[i]]
            cursors[key] = i + 1
    return out


def build_codereviewer(zenodo_dir: Path, kev_dir: Path, licences: Path, out: Path,
                       calibration_rows: int, test_rows: int) -> dict:
    """New CodeReviewer requests from Kev development (calibration) and test (test)
    projects, excluding every hunk any Kev devtools-v1 partition used."""
    if sha256_file(licences) != KEV_LICENCES_SHA256:
        raise CalibrateError(f"{licences}: sha256 is not the pinned Kev licence table")
    members = {}
    for name, expected in ZENODO_MEMBERS.items():
        path = zenodo_dir / name
        digest = sha256_file(path)
        if digest != expected:
            raise CalibrateError(f"{path}: sha256 {digest} != pinned {expected}")
        members[name] = path
    table = json.loads(licences.read_text())["codereviewer"]
    allowed = {p for p, v in table.items() if v and (v.get("spdx") or "").lower() in PERMISSIVE}
    records = kev_records(kev_dir)
    project_split, used = {}, set()
    for part, recs in records.items():
        for record in recs:
            if record["_meta"]["source"] != "codereviewer":
                continue
            project = record["_meta"]["group_id"].split("/", 1)[1]
            if project_split.setdefault(project, part) != part:
                raise CalibrateError(f"Kev project {project} spans partitions")
            used.add(text_digest(record["state"]["diff"]))
    role = {"development": "calibration", "test": "test"}
    pools, counts, seen = {"calibration": [], "test": []}, Counter(), set()
    for name in sorted(members):
        with open(members[name], encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                counts["rows"] += 1
                split = role.get(project_split.get(row["proj"], ""))
                if split is None:
                    continue
                if row["proj"] not in allowed:
                    counts["licence_dropped"] += 1
                    continue
                patch = row["patch"]
                if not patch.strip() or len(patch) > MAX_PATCH_CHARS:
                    counts["patch_size_dropped"] += 1
                    continue
                if text_digest(patch) in used:
                    counts["used_by_kev"] += 1
                    continue
                state = codereviewer_state(patch, row.get("oldf"))
                text = prep.render_state("codereviewer", state, name)
                if len(json.dumps(state)) > MAX_STATE_CHARS:
                    counts["state_size_dropped"] += 1
                    continue
                if "\r" in text or "\u2028" in text or "\u2029" in text:
                    counts["line_separator_dropped"] += 1
                    continue
                key = text_digest(text)
                if key in seen:
                    counts["duplicate_state"] += 1
                    continue
                seen.add(key)
                pools[split].append({
                    "id": f"cr-new/{name.split('.')[0]}/L{line_number}", "project": row["proj"],
                    "label": row["y"] == 1, "text": text, "member": name, "line": line_number,
                })
    limits = {"calibration": calibration_rows, "test": test_rows}
    rows, summary = [], {}
    for split in ("calibration", "test"):
        chosen = select_pairs(pools[split], limits[split], split)
        summary[split] = {"rows": len(chosen), "pool": len(pools[split]),
                          "projects": len({r["project"] for r in chosen}),
                          "labels": dict(Counter(str(r["label"]) for r in chosen))}
        for r in chosen:
            rows.append(request_row(r["id"], r["text"], CR_QUESTION, r["label"], {
                "split": split, "set": "cr-new", "task": CR_TASK, "source": "codereviewer",
                "group_id": f"codereviewer/{r['project']}",
                "project": table[r["project"]]["repo"],
                "repo_licence": table[r["project"]]["spdx"],
                "provenance": {"via": f"zenodo:{ZENODO_RECORD}/{ZENODO_FILE}:{r['member']}",
                               "line": r["line"]},
            }))
    report = {
        "format": "local-ai.judgment-eval-requests.v1",
        "set": "cr-new",
        "sources": {
            "zenodo": {"record": ZENODO_RECORD, "doi": f"10.5281/zenodo.{ZENODO_RECORD}",
                       "file": ZENODO_FILE, "sha256": ZENODO_SHA256,
                       "members": ZENODO_MEMBERS, "licence": "CC-BY-4.0",
                       "attribution": "Li et al., Automating Code Review Activities by "
                                      "Large-Scale Pre-Training (ESEC/FSE 2022)"},
            "kev_manifest_sha256": prep.PINNED_MANIFEST_SHA256,
            "kev_licences": {"url": KEV_LICENCES_URL, "sha256": KEV_LICENCES_SHA256},
        },
        "procedure": ("Kev devtools-v1 project partitions: development projects -> calibration, "
                      "test projects -> test; permissive project licences only; Kev's state "
                      "(hunk plus up to 10 old-file lines); hunks Kev used in any partition "
                      "excluded by normalised-text digest; state <= 4200 JSON chars; "
                      "label-balanced pairs within project, projects round-robin, "
                      f"seeded '{SEED}'"),
        "counts": dict(counts),
        "splits": summary,
    }
    return write_requests(out, rows, report)


def build_ood(kev_dir: Path, out: Path, per_source: int) -> dict:
    """Kev devtools-v1 development/test rows outside the devtools head's scope."""
    records = kev_records(kev_dir)
    rows, summary = [], {}
    for source, (key, task) in OOD_SOURCES.items():
        pool = [r for part in ("development", "test") for r in records[part]
                if r["_meta"]["source"] == source and key in r["questions"]
                and len(json.dumps(r["state"])) <= MAX_STATE_CHARS]
        trues = sorted((r for r in pool if r["questions"][key]["label"]),
                       key=lambda r: order_key("ood", r["_meta"]["row_sha256"]))
        falses = sorted((r for r in pool if not r["questions"][key]["label"]),
                        key=lambda r: order_key("ood", r["_meta"]["row_sha256"]))
        chosen = [r for pair in zip(trues, falses) for r in pair][:per_source]
        summary[task] = {"rows": len(chosen), "pool": len(pool)}
        for record in chosen:
            state = record["state"]
            text = state if isinstance(state, str) else prep.kev_render(state)
            question = record["questions"][key]
            rows.append(request_row(
                f"kev-ood/{task}/{record['_meta']['row_sha256']}", text,
                question["instructions"], question["label"],
                {"split": "test", "set": "kev-ood", "task": task, "source": source,
                 "group_id": record["_meta"]["group_id"], "kev_id": record["_meta"]["id"],
                 "kev_partition": record["_meta"]["split"],
                 "row_sha256": record["_meta"]["row_sha256"]}))
    report = {
        "format": "local-ai.judgment-eval-requests.v1",
        "set": "kev-ood",
        "sources": {"kev_manifest_sha256": prep.PINNED_MANIFEST_SHA256,
                    "kev_revision": prep.KEV_REVISION},
        "procedure": ("Kev devtools-v1 development+test predicates the devtools head was not "
                      "trained on; label-balanced, seeded order; Kev's own question text; "
                      "dict states rendered with kev_render; split 'test' means OOD-only"),
        "splits": summary,
    }
    return write_requests(out, rows, report)


def rerender(rows_path: Path, out: Path, sample: int, set_name: str) -> dict:
    """Native requests for prepared text rows (`judgment_prepare.py` output), so the
    serving path can recapture them: `A) Yes`/`B) No` rows become predicates, any
    other option order a choice over the rendered values (booleans for Yes/No)."""
    rows = []
    for line in rows_path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        row = json.loads(line)
        meta = row["metadata"]
        parsed = _texts_from_rendering(row)
        labels = meta.get("option_labels")
        if meta.get("options") == ["Yes", "No"] and labels == [True, False]:
            kind, options = "predicate", None
        elif isinstance(labels, list) and all(isinstance(v, bool) for v in labels):
            kind, options = "choice", labels
        else:
            kind, options = "choice", meta["options"]
        request = {"id": row["id"], "state": parsed["state"], "question": parsed["question"],
                   "kind": kind, "metadata": dict(meta, set=set_name)}
        if options is not None:
            request["options"] = options
        text, ends = prep.render(parsed["state"], parsed["question"],
                                 [(o, None) for o in meta["options"]])
        if text != row["text"] or ends != row["token_end_offsets"]:
            raise CalibrateError(f"{row['id']}: re-rendering changed the prompt")
        rows.append(request)
    rows.sort(key=lambda r: order_key("rerender", r["id"]))
    rows = rows[:sample] if sample else rows
    report = {"format": "local-ai.judgment-eval-requests.v1", "set": set_name,
              "sources": {"rows": str(rows_path), "rows_sha256": sha256_file(rows_path)},
              "procedure": "prepared rows re-expressed as native requests; rendering checked "
                           "byte-identical (text and option/decision byte ends)"}
    return write_requests(out, rows, report)


def write_requests(out: Path, rows: list[dict], report: dict) -> dict:
    if out.exists():
        raise CalibrateError(f"{out} exists; requests are never overwritten")
    out.mkdir(parents=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    (out / "requests.jsonl").write_text(payload, encoding="utf-8")
    report = dict(report, rows=len(rows),
                  requests_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                  tool_sha256=sha256_file(Path(__file__)))
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


# ---- capture orchestration -------------------------------------------------


def chunk_requests(requests: Path, chunk_rows: int) -> list[list[str]]:
    lines = [line for line in requests.read_text(encoding="utf-8").split("\n") if line]
    return [lines[i:i + chunk_rows] for i in range(0, len(lines), chunk_rows)]


def _run_chunk(runner, command, log_path: Path, attempts: int = 30, wait: float = 10.0):
    """Start one chunk. Only a failure to *start* it (the host's process table full:
    EAGAIN here, or `lockf` exiting EX_OSERR 71 because it could not fork the
    capture) is retried, at a fixed bounded cadence: there is no event to wait on for
    a free process slot. The capture binary itself exits 0 or 1, never 71."""
    for attempt in range(attempts):
        try:
            with open(log_path, "w") as log:
                result = runner(command, stdout=subprocess.PIPE, stderr=log, text=True)
            if result.returncode != EX_OSERR or attempt + 1 == attempts:
                return result
        except BlockingIOError:
            if attempt + 1 == attempts:
                raise
        time.sleep(wait)
    raise AssertionError("unreachable")


EX_OSERR = 71


def capture(requests: Path, work: Path, binary: Path, chunk_rows: int,
            max_chunks: int | None = None, lock: str | None = LOCK, runner=subprocess.run) -> dict:
    """Capture `requests` in chunks with `judgment-capture`, one bounded GPU-lock hold
    per chunk; a chunk is skipped once its `.done.json` exists, so runs resume."""
    work.mkdir(parents=True, exist_ok=True)
    chunks = chunk_requests(requests, chunk_rows)
    binary_sha256 = sha256_file(binary) if binary.is_file() else None
    ran = 0
    for index, lines in enumerate(chunks):
        name = f"chunk-{index:04d}"
        done = work / f"{name}.done.json"
        if done.exists():
            continue
        if max_chunks is not None and ran >= max_chunks:
            break
        source = work / f"{name}.jsonl"
        source.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        target = work / name
        if target.exists():  # a crashed chunk: start it again from nothing
            for child in target.iterdir():
                child.unlink()
            target.rmdir()
        command = [str(binary), "--out", str(target), str(source)]
        if lock:
            command = ["lockf", "-k", lock] + command
        result = _run_chunk(runner, command, work / f"{name}.log")
        if result.returncode != 0:
            raise CalibrateError(f"{name} failed ({result.returncode}); see {name}.log")
        summary = json.loads(result.stdout.strip().split("\n")[-1])
        if summary.get("rows") != len(lines):
            raise CalibrateError(f"{name}: captured {summary.get('rows')} of {len(lines)} rows")
        done.write_text(json.dumps(dict(summary, source_sha256=sha256_file(source),
                                        binary_sha256=binary_sha256)) + "\n")
        ran += 1
    remaining = sum(not (work / f"chunk-{i:04d}.done.json").exists() for i in range(len(chunks)))
    return {"chunks": len(chunks), "ran": ran, "remaining": remaining}


def merge(work: Path, out: Path) -> dict:
    """Hard-link every finished chunk into one feature directory with one manifest."""
    if out.exists():
        raise CalibrateError(f"{out} exists")
    chunks = sorted(p for p in work.glob("chunk-[0-9][0-9][0-9][0-9]") if p.is_dir())
    if not chunks or any(not (work / f"{c.name}.done.json").exists() for c in chunks):
        raise CalibrateError("every chunk must be finished before merging")
    out.mkdir(parents=True)
    lines, index = [], 0
    for chunk in chunks:
        for line in (chunk / "features.jsonl").read_text().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            name = f"feature-{index:06d}.bin"
            os.link(chunk / row["feature_file"], out / name)
            row["feature_file"] = name
            lines.append(json.dumps(row, sort_keys=True))
            index += 1
    payload = "\n".join(lines) + "\n"
    (out / "features.jsonl").write_text(payload)
    return {"rows": index, "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def parity(left: Path, right: Path) -> dict:
    """Bitwise feature comparison of the ids two feature directories share."""
    def index(directory):
        rows = {}
        for line in (directory / "features.jsonl").read_text().split("\n"):
            if line:
                row = json.loads(line)
                rows[row["id"]] = row
        return rows
    a, b = index(left), index(right)
    shared = sorted(set(a) & set(b))
    mismatched = [i for i in shared
                  if a[i]["token_ids"] != b[i]["token_ids"]
                  or a[i]["positions"] != b[i]["positions"]
                  or (left / a[i]["feature_file"]).read_bytes()
                  != (right / b[i]["feature_file"]).read_bytes()]
    return {"shared": len(shared), "identical": len(shared) - len(mismatched),
            "mismatched": mismatched}


# ---- scoring (NumPy, lazy) --------------------------------------------------


def _texts(path: Path | None) -> dict[str, dict]:
    """id -> {state, question, options} from requests or prepared rows (text)."""
    if path is None:
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if "state" in row:
            out[row["id"]] = {"state": row["state"], "question": row["question"]}
        else:
            out[row["id"]] = _texts_from_rendering(row)
    return out


def _texts_from_rendering(row: dict) -> dict:
    head, _, rest = row["text"].partition("\n\nQuestion: ")
    if not head.startswith("State:\n") or "\nOptions:\n" not in rest:
        raise CalibrateError(f"{row['id']}: text is not a judgment rendering")
    return {"state": head[len("State:\n"):], "question": rest.split("\nOptions:\n", 1)[0]}


def family_of(metadata: dict) -> str:
    if metadata.get("dataset") == "hard-v1":
        return f"{metadata['family']}/{metadata['question_type']}"
    return str(metadata.get("task", "unknown"))


def score(head_path: Path, sources: list[tuple[str, Path, Path | None]], out: Path) -> dict:
    """Raw logits and query projections for every row of every feature directory."""
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("score needs NumPy: uv run --with numpy python3 "
                         "tools/judgment_calibrate.py score ...") from error
    tensors, metadata = jt.read_safetensors(head_path)
    (dim, width), q_w = tensors["q.weight"]
    q_w = np.asarray(q_w, dtype=np.float64).reshape(dim, width)
    k_w = np.asarray(tensors["k.weight"][1], dtype=np.float64).reshape(dim, width)
    q_b = np.asarray(tensors["q.bias"][1], dtype=np.float64)
    k_b = np.asarray(tensors["k.bias"][1], dtype=np.float64)
    temperature = float(tensors["temperature"][1][0])
    records = []
    for name, directory, texts_path in sources:
        texts = _texts(texts_path)
        for line in (directory / "features.jsonl").read_text().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            meta = row["metadata"]
            rows = np.frombuffer((directory / row["feature_file"]).read_bytes(),
                                 dtype="<f2").astype(np.float64).reshape(-1, width)
            q = q_w @ rows[-1] + q_b
            k = rows[:-1] @ k_w.T + k_b
            raw = (k @ q) / math.sqrt(dim)  # logit x exported temperature
            options = meta.get("options")
            labels = meta.get("option_labels")
            yes = (labels.index(True) if isinstance(labels, list) and len(labels) == 2
                   and set(map(str, labels)) == {"True", "False"} else None)
            text = texts.get(row["id"], {})
            records.append({
                "id": row["id"], "set": name, "split": meta["split"],
                "target": meta["target"], "group": meta.get("group_id", row["id"]),
                "family": family_of(meta), "options": options, "yes_index": yes,
                "raw": [float(v) for v in raw], "query": [round(float(v), 6) for v in q],
                "prompt_tokens": len(row["token_ids"]),
                "question_text": text.get("question", meta.get("question_text")),
                "state_prefix": (text.get("state") or meta.get("state_prefix") or "")[:200],
                "feature_sha256": hashlib.sha256(
                    (directory / row["feature_file"]).read_bytes()).hexdigest(),
            })
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    out.write_text(payload)
    summary = {"rows": len(records), "head_sha256": sha256_file(head_path),
               "exported_temperature": temperature, "renderer": metadata.get("renderer"),
               "sources": {name: {"features_sha256": sha256_file(directory / "features.jsonl"),
                                  "texts_sha256": sha256_file(texts) if texts else None}
                           for name, directory, texts in sources},
               "scores_sha256": hashlib.sha256(payload.encode()).hexdigest()}
    score_summary_path(out).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def score_summary_path(scores: Path) -> Path:
    return scores.with_name(scores.name + ".summary.json")


def check_scores(scores: Path, head: Path) -> dict:
    """The summary `score` wrote for `scores`, refusing scores of another head or
    scores changed since."""
    summary = json.loads(score_summary_path(scores).read_text())
    if summary["head_sha256"] != sha256_file(head):
        raise CalibrateError(f"{scores} was scored with a different head")
    if summary["scores_sha256"] != sha256_file(scores):
        raise CalibrateError(f"{scores} changed after scoring")
    return summary


# ---- calibration math (pure Python) ----------------------------------------


def softmax(logits):
    return [math.exp(v) for v in jt._log_softmax(logits)]


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def platt_features(raw, yes_index):
    """Yes-minus-No raw logit difference and the Yes label for a 2-option row."""
    return raw[yes_index] - raw[1 - yes_index]


def _platt_loss(a, b, margins, labels, ridge):
    loss = 0.0
    for x, y in zip(margins, labels):
        z = a * x + b
        # -log sigmoid(z) for y, -log sigmoid(-z) otherwise, computed stably
        s = -z if y else z
        loss += max(s, 0.0) + math.log1p(math.exp(-abs(s)))
    return loss + ridge * (a * a + b * b) / 2


def fit_platt(margins, labels, steps: int = 100, ridge: float = 1e-6):
    """(a, b) maximising the likelihood of P(yes) = sigmoid(a*margin + b): damped
    Newton with backtracking, so every accepted step lowers the loss."""
    if not margins or len(margins) != len(labels):
        raise ValueError("need one label per margin")
    a, b = 1.0, 0.0
    loss = _platt_loss(a, b, margins, labels, ridge)
    for _ in range(steps):
        ga, gb, haa, hab, hbb = ridge * a, ridge * b, ridge, 0.0, ridge
        for x, y in zip(margins, labels):
            p = sigmoid(a * x + b)
            r = p - float(y)
            w = p * (1 - p)
            ga += r * x
            gb += r
            haa += w * x * x
            hab += w * x
            hbb += w
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = 1.0
        while step > 1e-10:
            trial = _platt_loss(a - step * da, b - step * db, margins, labels, ridge)
            if trial <= loss:
                break
            step /= 2
        else:
            break
        a, b = a - step * da, b - step * db
        improved, loss = loss - trial, trial
        if improved < 1e-12 * max(1.0, loss):
            break
    return a, b


class Calibrator:
    """A fitted map from raw logits to calibrated option probabilities."""

    def __init__(self, method: str, temperature: float = 1.0, slope: float = 1.0,
                 bias: float = 0.0):
        if method not in ("temperature", "platt"):
            raise ValueError(f"unknown method {method}")
        self.method, self.temperature, self.slope, self.bias = method, temperature, slope, bias

    def probabilities(self, raw, yes_index=None):
        if self.method == "platt":
            if yes_index is None or len(raw) != 2:
                # Out of a predicate head's scope; diagnostic only: the slope as
                # an inverse temperature, no bias.
                return softmax([z * self.slope for z in raw])
            p_yes = sigmoid(self.slope * platt_features(raw, yes_index) + self.bias)
            out = [0.0, 0.0]
            out[yes_index], out[1 - yes_index] = p_yes, 1 - p_yes
            return out
        return softmax([z / self.temperature for z in raw])

    def to_json(self) -> dict:
        if self.method == "platt":
            return {"method": "platt", "slope": self.slope, "bias": self.bias,
                    "input": "raw logit(Yes) - raw logit(No)"}
        return {"method": "temperature", "temperature": self.temperature,
                "input": "raw logits"}


def fit_calibrator(method: str, rows) -> Calibrator:
    if method == "platt":
        margins = [platt_features(r["raw"], r["yes_index"]) for r in rows]
        labels = [r["target"] == r["yes_index"] for r in rows]
        slope, bias = fit_platt(margins, labels)
        return Calibrator("platt", slope=slope, bias=bias)
    return Calibrator("temperature",
                      temperature=jt.fit_temperature([r["raw"] for r in rows],
                                                     [r["target"] for r in rows]))


def candidate_methods(rows) -> list[str]:
    predicates = all(len(r["raw"]) == 2 and r["yes_index"] is not None for r in rows)
    return ["temperature", "platt"] if predicates else ["temperature"]


def cross_validated_nll(method: str, rows, folds: int = CV_FOLDS) -> float:
    """Mean held-out NLL over group folds (every group in exactly one fold)."""
    total, count = 0.0, 0
    for fold in range(folds):
        held = [r for r in rows if fold_of(r["group"], folds) == fold]
        kept = [r for r in rows if fold_of(r["group"], folds) != fold]
        if not held or not kept:
            continue
        calibrator = fit_calibrator(method, kept)
        for r in held:
            p = calibrator.probabilities(r["raw"], r["yes_index"])[r["target"]]
            total -= math.log(max(p, 1e-300))
            count += 1
    if not count:
        raise CalibrateError("cross-validation needs at least two groups")
    return total / count


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def predictions(calibrator: Calibrator, rows) -> list[dict]:
    out = []
    for r in rows:
        probs = calibrator.probabilities(r["raw"], r["yes_index"])
        top = max(range(len(probs)), key=probs.__getitem__)
        out.append({"probs": probs, "top": top, "confidence": probs[top],
                    "correct": top == r["target"], "target": r["target"],
                    "yes_index": r["yes_index"], "family": r["family"]})
    return out


def metrics(preds, bins: int = ECE_BINS) -> dict:
    n = len(preds)
    if not n:
        return {"count": 0}
    correct = sum(p["correct"] for p in preds)
    nll = -sum(math.log(max(p["probs"][p["target"]], 1e-300)) for p in preds) / n
    brier = sum(sum((q - (i == p["target"])) ** 2 for i, q in enumerate(p["probs"]))
                for p in preds) / n
    counts, conf, hits = [0] * bins, [0.0] * bins, [0.0] * bins
    for p in preds:
        b = min(int(p["confidence"] * bins), bins - 1)
        counts[b] += 1
        conf[b] += p["confidence"]
        hits[b] += p["correct"]
    ece = sum(abs(hits[b] - conf[b]) for b in range(bins) if counts[b]) / n
    uniform_nll = sum(math.log(len(p["probs"])) for p in preds) / n
    low, high = wilson(correct, n)
    out = {"count": n, "accuracy": correct / n, "accuracy_wilson95": [low, high],
           "nll": nll, "uniform_nll": uniform_nll, "brier": brier, "ece": ece}
    if all(len(p["probs"]) == 2 for p in preds):
        out["brier_binary"] = brier / 2
    return out


def selective(preds, threshold: float | None = None) -> dict:
    """Risk/coverage curve (sorted by confidence), AURC and the point at `threshold`."""
    n = len(preds)
    ranked = sorted(preds, key=lambda p: -p["confidence"])
    errors, curve, aurc = 0, [], 0.0
    for k, p in enumerate(ranked, 1):
        errors += not p["correct"]
        aurc += errors / k
        curve.append(errors / k)
    points = {}
    for coverage in COVERAGES:
        k = max(1, round(coverage * n))
        points[f"{coverage:.1f}"] = {"answered": k, "risk": curve[k - 1],
                                     "min_confidence": ranked[k - 1]["confidence"]}
    out = {"aurc": aurc / n if n else None, "curve": points}
    if threshold is not None:
        answered = [p for p in preds if p["confidence"] >= threshold]
        wrong = sum(not p["correct"] for p in answered)
        out["at_threshold"] = {
            "threshold": threshold, "answered": len(answered),
            "coverage": len(answered) / n if n else 0.0,
            "risk": wrong / len(answered) if answered else None,
            "risk_wilson95": list(wilson(wrong, len(answered))) if answered else None,
        }
    return out


def choose_threshold(preds, target_risk: float = TARGET_RISK,
                     min_answered: int = MIN_ANSWERED) -> float | None:
    """Smallest CAL confidence whose answered set has >= min_answered rows and a Wilson
    95% upper risk bound <= target_risk; None (refuse everything) if none does."""
    for threshold in sorted({p["confidence"] for p in preds}):
        answered = [p for p in preds if p["confidence"] >= threshold]
        if len(answered) < min_answered:
            return None
        wrong = sum(not p["correct"] for p in answered)
        if wilson(wrong, len(answered))[1] <= target_risk:
            return threshold
    return None


def fit_gate(rows, percentile: float = GATE_PERCENTILE) -> dict:
    """Diagonal Mahalanobis gate on the head's query projection of the decision row."""
    dim = len(rows[0]["query"])
    mean = [sum(r["query"][i] for r in rows) / len(rows) for i in range(dim)]
    std = [max(math.sqrt(sum((r["query"][i] - mean[i]) ** 2 for r in rows) / len(rows)), 1e-6)
           for i in range(dim)]
    gate = {"mean": mean, "std": std}
    distances = sorted(gate_distance(gate, r["query"]) for r in rows)
    gate["threshold"] = distances[min(len(distances) - 1,
                                      math.ceil(percentile * len(distances)) - 1)]
    gate["percentile"] = percentile
    return gate


def gate_distance(gate: dict, query) -> float:
    return sum(((q - m) / s) ** 2 for q, m, s in zip(query, gate["mean"], gate["std"])) / len(query)


def in_declared_scope(row: dict, scope: dict) -> bool:
    """Reference declarative scope check (mirrors what a serving artifact declares)."""
    if row["prompt_tokens"] > scope["max_prompt_tokens"]:
        return False
    options = row.get("options")
    if "questions" in scope:  # a closed set of exact question texts (synthetic templates)
        return (row.get("question_text") in scope["questions"]
                and len(options or []) in scope["option_counts"])
    if sorted(map(str, options or [])) not in [sorted(s) for s in scope["option_sets"]]:
        return False
    question = row.get("question_text") or ""
    if not any(re.fullmatch(pattern, question) for pattern in scope["question_patterns"]):
        return False
    state = row.get("state_prefix") or ""
    return any(state.startswith(prefix) for prefix in scope["state_prefixes"])


DEVTOOLS_SCOPE = {
    "kinds": ["predicate", "choice over the booleans true/false"],
    "option_sets": [["No", "Yes"]],
    "question_patterns": [re.escape(CR_QUESTION), CM_QUESTION.pattern],
    "state_prefixes": ["Diff:\n", "Context before hunk:\n"],
    "max_prompt_tokens": 2048,
}


def exact_scope(rows, max_prompt_tokens: int = 2048) -> dict:
    """A closed scope: the exact question texts and option counts of `rows` (the
    head's training rows), for heads trained on synthetic templates."""
    return {"questions": sorted({r["question_text"] for r in rows}),
            "option_counts": sorted({len(r["options"]) for r in rows}),
            "max_prompt_tokens": max_prompt_tokens}


def select_rows(records, spec: str) -> list[dict]:
    """`set:split[:a|b][@FAMILY_REGEX]`; the optional half splits by group hash, the
    optional regex keeps rows whose family it matches in full."""
    spec_rows, _, family = spec.partition("@")
    parts = spec_rows.split(":")
    if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] not in ("a", "b")):
        raise CalibrateError(f"bad role spec {spec!r} (want set:split[:a|b][@family])")
    rows = [r for r in records if r["set"] == parts[0] and r["split"] == parts[1]]
    if len(parts) == 3:
        rows = [r for r in rows if half_of(r["group"]) == parts[2]]
    if family:
        rows = [r for r in rows if re.fullmatch(family, r["family"])]
    if not rows:
        raise CalibrateError(f"{spec}: no rows")
    return rows


def majority_baseline(train, rows) -> dict:
    """Train-only majority: the majority answer value for predicates, the majority
    target position per option count otherwise; accuracy on `rows`."""
    yes = Counter(r["target"] == r["yes_index"] for r in train if r["yes_index"] is not None)
    positions = defaultdict(Counter)
    for r in train:
        positions[len(r["raw"])][r["target"]] += 1
    hits = 0
    for r in rows:
        if r["yes_index"] is not None and yes:
            guess_yes = yes[True] >= yes[False]
            hits += (r["target"] == r["yes_index"]) == guess_yes
        elif positions[len(r["raw"])]:
            hits += r["target"] == positions[len(r["raw"])].most_common(1)[0][0]
        else:
            hits += r["target"] == 0
    return {"count": len(rows), "accuracy": hits / len(rows)}


def by_family(rows, preds, train, threshold):
    out = {}
    for family in sorted({r["family"] for r in rows}):
        index = [i for i, r in enumerate(rows) if r["family"] == family]
        fp = [preds[i] for i in index]
        out[family] = {"model": metrics(fp),
                       "majority": majority_baseline(train, [rows[i] for i in index]),
                       "selective": selective(fp, threshold)["at_threshold"]
                       if threshold is not None else None}
    return out


def evaluate(records, cal_specs, test_specs, ood_specs, train_specs, scope=None,
             target_risk: float = TARGET_RISK) -> dict:
    cal = [r for s in cal_specs for r in select_rows(records, s)]
    test = [r for s in test_specs for r in select_rows(records, s)]
    ood = [r for s in ood_specs for r in select_rows(records, s)]
    train = [r for s in train_specs for r in select_rows(records, s)]
    ids = [r["id"] for r in cal + test]
    if len(ids) != len(set(ids)):
        raise CalibrateError("CAL and TEST share rows")
    if {r["group"] for r in cal} & {r["group"] for r in test} - {None}:
        shared = {r["group"] for r in cal} & {r["group"] for r in test}
        raise CalibrateError(f"CAL and TEST share groups: {sorted(shared)[:5]}")
    cv = {m: cross_validated_nll(m, cal) for m in candidate_methods(cal)}
    method = min(cv, key=cv.get)
    calibrator = fit_calibrator(method, cal)
    raw = Calibrator("temperature", 1.0)
    cal_preds = predictions(calibrator, cal)
    threshold = choose_threshold(cal_preds, target_risk)
    gate = fit_gate(cal)
    report = {
        "protocol": "docs/DECISIONS_EVALUATION.md part 1",
        "roles": {"cal": cal_specs, "test": test_specs, "ood": ood_specs, "train": train_specs},
        "cross_validated_nll": cv, "calibration": calibrator.to_json(),
        "target_risk": target_risk, "threshold": threshold,
        "gate": {"threshold": gate["threshold"], "percentile": gate["percentile"]},
        "scope": None if scope is None else {
            k: (len(v) if k == "questions" else v) for k, v in scope.items()},
        "splits": {},
    }
    for name, rows in (("cal", cal), ("test", test), ("ood", ood)):
        if not rows:
            continue
        preds = predictions(calibrator, rows)
        entry = {
            "raw": metrics(predictions(raw, rows)),
            "calibrated": metrics(preds),
            "selective": selective(preds, threshold),
            "by_family": by_family(rows, preds, train, threshold),
            "majority": majority_baseline(train, rows),
            "gate_refused": sum(gate_distance(gate, r["query"]) > gate["threshold"]
                                for r in rows) / len(rows),
        }
        if scope is not None:
            entry["scope_refused"] = sum(not in_declared_scope(r, scope)
                                         for r in rows) / len(rows)
        report["splits"][name] = entry
    report["gate_params"] = gate
    return report


def acceptance(report: dict, families: list[str], multiclass: bool) -> dict:
    """The pre-registered criteria A1-A5 (A6 is checked by tests and live runs)."""
    test, ood = report["splits"]["test"], report["splits"].get("ood")
    cal_test = test["calibrated"]
    a1_overall = cal_test["accuracy_wilson95"][0] >= test["majority"]["accuracy"] + 0.10
    a1_families = all(
        test["by_family"][f]["model"]["accuracy_wilson95"][0]
        > test["by_family"][f]["majority"]["accuracy"]
        for f in families if f in test["by_family"])
    brier = cal_test["brier"] if multiclass else cal_test.get("brier_binary", 1.0)
    a2 = (cal_test["ece"] <= 0.05 and cal_test["nll"] < cal_test["uniform_nll"]
          and brier <= (0.40 if multiclass else 0.20))
    point = test["selective"].get("at_threshold")
    a3 = bool(point and point["answered"] and point["risk_wilson95"][1] <= 0.15
              and point["coverage"] >= 0.25)
    a4 = all(
        (fam["selective"] is not None)
        and (fam["selective"]["coverage"] <= 0.05
             or (fam["selective"]["risk"] is not None and fam["selective"]["risk"] <= 0.15))
        for f, fam in test["by_family"].items())
    a5 = bool(ood and ood.get("scope_refused") == 1.0 and ood["gate_refused"] >= 0.80
              and test["gate_refused"] <= 0.05)
    result = {"A1": a1_overall and a1_families, "A2": a2, "A3": a3, "A4": a4, "A5": a5}
    result["passed"] = all(result.values())
    return result


def markdown(report: dict, title: str) -> str:
    lines = [f"## {title}", "",
             (f"Calibration: `{json.dumps(report['calibration'])}`; CV NLL "
              f"{json.dumps({k: round(v, 4) for k, v in report['cross_validated_nll'].items()})}; "
              f"threshold τ = {report['threshold']}; gate threshold "
              f"{report['gate']['threshold']:.3f}"), "",
             ("| Split | Rows | Majority acc | Acc (95% CI) | Raw NLL | Cal NLL | Raw ECE | "
              "Cal ECE | Cal Brier | AURC | Coverage@τ | Risk@τ | Gate refused | Scope refused |"),
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entry in report["splits"].items():
        c, r = entry["calibrated"], entry["raw"]
        at = entry["selective"].get("at_threshold") or {}
        low, high = c["accuracy_wilson95"]
        risk = at.get("risk")
        lines.append(
            f"| {name} | {c['count']} | {entry['majority']['accuracy']:.3f} | "
            f"{c['accuracy']:.3f} ({low:.3f}-{high:.3f}) | {r['nll']:.4f} | {c['nll']:.4f} | "
            f"{r['ece']:.4f} | {c['ece']:.4f} | {c.get('brier_binary', c['brier']):.4f} | "
            f"{entry['selective']['aurc']:.4f} | {at.get('coverage', 0):.3f} | "
            f"{'-' if risk is None else f'{risk:.3f}'} | {entry['gate_refused']:.3f} | "
            f"{entry.get('scope_refused', float('nan')):.3f} |")
    lines += ["", "| Split | Family | Rows | Majority | Accuracy | Cal ECE | Coverage@τ | Risk@τ |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, entry in report["splits"].items():
        for family, fam in entry["by_family"].items():
            sel = fam["selective"] or {}
            risk = sel.get("risk")
            lines.append(
                f"| {name} | {family} | {fam['model']['count']} | "
                f"{fam['majority']['accuracy']:.3f} | {fam['model']['accuracy']:.3f} | "
                f"{fam['model']['ece']:.4f} | {sel.get('coverage', 0):.3f} | "
                f"{'-' if risk is None else f'{risk:.3f}'} |")
    lines += ["", "Risk/coverage (TEST, calibrated confidence ranking):", "",
              "| Coverage | " + " | ".join(COVERAGES_LABELS) + " |",
              "|---|" + "---:|" * len(COVERAGES)]
    for name, entry in report["splits"].items():
        curve = entry["selective"]["curve"]
        lines.append(f"| {name} risk | " + " | ".join(
            f"{curve[f'{c:.1f}']['risk']:.3f}" for c in COVERAGES) + " |")
    if "acceptance" in report:
        lines += ["", f"Acceptance: `{json.dumps(report['acceptance'])}`"]
    return "\n".join(lines) + "\n"


def canonical_sha256(value) -> str:
    """sha256 of compact, key-sorted JSON (how an artifact's content hash is taken)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def calibration_artifact(report: dict, head_sha256: str, scope: dict | None) -> dict:
    """The calibration, refusal threshold, feature gate and scope fitted on CAL, pinned
    to the head they were fitted for, with the evaluation's provenance and acceptance
    and a content hash over everything else. A server may only use it after checking
    both hashes; `acceptance.passed` false means it must not be served."""
    gate = report["gate_params"]
    body = {
        "format": "local-ai.judgment-calibration.v1",
        "head_sha256": head_sha256,
        "calibration": report["calibration"],
        "refusal": {"target_risk": report["target_risk"], "min_answered": MIN_ANSWERED,
                    "confidence": "calibrated probability of the selected option",
                    "threshold": report["threshold"]},
        "gate": {"input": "head query projection of the Decision: row",
                 "distance": "mean squared z-score (diagonal Mahalanobis / dim)",
                 "mean": gate["mean"], "std": gate["std"], "threshold": gate["threshold"],
                 "percentile": gate["percentile"]},
        "scope": scope,
        "provenance": {"roles": report["roles"], "scores_sha256": report["scores_sha256"],
                       "sources": report.get("score_summary", {}).get("sources"),
                       "tool_sha256": report["tool_sha256"],
                       "protocol": report["protocol"]},
        "acceptance": report["acceptance"],
    }
    return dict(body, content_sha256=canonical_sha256(body))


def verify_calibration_artifact(artifact: dict, head_sha256: str) -> None:
    """Refuse an artifact whose content hash or head hash does not match."""
    body = {k: v for k, v in artifact.items() if k != "content_sha256"}
    if artifact.get("content_sha256") != canonical_sha256(body):
        raise CalibrateError("calibration artifact content hash does not match")
    if artifact.get("head_sha256") != head_sha256:
        raise CalibrateError("calibration artifact was fitted for a different head")


COVERAGES_LABELS = [f"{int(c * 100)}%" for c in COVERAGES]


def load_scores(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line]


# ---- CLI --------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-codereviewer")
    p.add_argument("--zenodo-dir", type=Path, required=True)
    p.add_argument("--kev-dir", type=Path, required=True)
    p.add_argument("--licences", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calibration-rows", type=int, default=480)
    p.add_argument("--test-rows", type=int, default=800)
    p = sub.add_parser("build-ood")
    p.add_argument("--kev-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-source", type=int, default=150)
    p = sub.add_parser("rerender")
    p.add_argument("rows", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--set", default="rerender")
    p = sub.add_parser("capture")
    p.add_argument("requests", type=Path)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=Path("target/release/judgment-capture"))
    p.add_argument("--chunk-rows", type=int, default=32)
    p.add_argument("--max-chunks", type=int)
    p.add_argument("--merge", type=Path, help="merge finished chunks into this directory")
    p = sub.add_parser("score")
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--source", action="append", required=True,
                   help="NAME=FEATURE_DIR[,TEXT_ROWS.jsonl]")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True,
                   help="the scored head, whose sha256 the calibration artifact pins")
    p.add_argument("--cal", action="append", required=True)
    p.add_argument("--test", action="append", required=True)
    p.add_argument("--ood", action="append", default=[])
    p.add_argument("--train", action="append", required=True)
    p.add_argument("--scope", choices=["devtools", "seen-questions", "none"], default="none",
                   help="devtools: declared code-diff predicates; seen-questions: the exact "
                        "question texts and option counts of the --train and --cal rows")
    p.add_argument("--families", nargs="*", default=[])
    p.add_argument("--multiclass", action="store_true")
    p.add_argument("--title", default="Evaluation")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("parity")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build-codereviewer":
        result = build_codereviewer(args.zenodo_dir, args.kev_dir, args.licences, args.out,
                                    args.calibration_rows, args.test_rows)
    elif args.command == "build-ood":
        result = build_ood(args.kev_dir, args.out, args.per_source)
    elif args.command == "rerender":
        result = rerender(args.rows, args.out, args.sample, args.set)
    elif args.command == "capture":
        result = capture(args.requests, args.work, args.binary, args.chunk_rows, args.max_chunks)
        if args.merge and not result["remaining"]:
            result["merged"] = merge(args.work, args.merge)
    elif args.command == "score":
        sources = []
        for spec in args.source:
            name, _, rest = spec.partition("=")
            directory, _, texts = rest.partition(",")
            sources.append((name, Path(directory), Path(texts) if texts else None))
        result = score(args.head, sources, args.out)
    elif args.command == "evaluate":
        score_summary = check_scores(args.scores, args.head)
        records = load_scores(args.scores)
        scope = None
        if args.scope == "devtools":
            scope = DEVTOOLS_SCOPE
        elif args.scope == "seen-questions":
            scope = exact_scope([r for s in args.train + args.cal
                                 for r in select_rows(records, s)])
        report = evaluate(records, args.cal, args.test, args.ood, args.train, scope)
        report["acceptance"] = acceptance(report, args.families, args.multiclass)
        report["scores_sha256"] = sha256_file(args.scores)
        report["score_summary"] = score_summary
        report["tool_sha256"] = sha256_file(Path(__file__))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        (args.out / "report.md").write_text(markdown(report, args.title))
        artifact = calibration_artifact(report, sha256_file(args.head), scope)
        (args.out / "calibration.json").write_text(json.dumps(artifact, indent=1) + "\n")
        result = {"acceptance": report["acceptance"], "out": str(args.out),
                  "calibration_content_sha256": artifact["content_sha256"]}
    else:
        result = parity(args.left, args.right)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
