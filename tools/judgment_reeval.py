"""EXPERIMENTAL judgment-head re-evaluation (protocol v2, docs/DECISIONS_EVALUATION.md part 3).

Builds fresh, project-disjoint request sets from pinned public data, assembles a
training feature directory for `judgment_train.py`, fits a training-only
out-of-scope feature gate on the raw `Decision:` hidden state, scores features
and evaluates with per-family calibration and per-family refusal thresholds.
Request building and evaluation reuse `judgment_calibrate.py`; capture runs
through `judgment_calibrate.py capture` (one bounded GPU-lock hold per chunk).
Nothing here claims production quality, MTP gains or coding quality.

Subcommands:

    build-codereviewer  CodeReviewer `needs_comment` requests by v2 project role
                        (train augmentation, CAL, TEST) from Zenodo 6900648
                        cls-valid/cls-test, excluding every hunk Kev or `cr-new` used
    build-commitpackft  CommitPackFT message-match requests from repositories Kev
                        never sampled, by v2 repository-component role
    build-ood           fresh out-of-scope requests (rows no earlier evaluation scored)
    build-dev-ood       out-of-scope requests used only to develop the gate
    assemble            hard-link feature directories into one training directory
                        with v2 roles as `metadata.split` (judgment_train format)
    fit-gate            PCA + residual Mahalanobis gate on training decide rows
    score               raw logits, query and gate distance per feature row
    evaluate            per-family calibration and thresholds on CAL; CAL/TEST/OOD
                        metrics, acceptance A1-A5 and the calibration artifact
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judgment_calibrate as jc
import judgment_prepare as prep
import judgment_train as jt

SEED = "local-ai-decisions-v2"
CAL_CR_PROJECTS, TEST_CR_PROJECTS = 12, 20
CM_ROLE_BUCKETS = {"train": 70, "validation": 10, "cal": 10, "test": 10}  # % of components
MAX_CM_DIFF_CHARS = 4000
CM_SUBJECT_CHARS = (10, 120)
LINE_BREAKS = re.compile("[\r\u0085\u2028\u2029]")
GATE_COMPONENTS = 64
GATE_PERCENTILE = 0.99
FAMILIES = (jc.CR_TASK, jc.CM_TASK)


class ReevalError(ValueError):
    """Inputs do not match what this tool requires."""


def key(*parts: str) -> str:
    return hashlib.sha256(":".join((SEED,) + parts).encode()).hexdigest()


# ---- CodeReviewer -----------------------------------------------------------


def cr_project_roles(kev_projects: dict[str, str], permissive: set[str]) -> dict[str, str]:
    """v2 role per CodeReviewer project. Kev development projects select the epoch
    (`validation`), Kev test projects (the earlier TEST) become `train`; Kev train
    projects plus permissive projects Kev never used are hash-ordered: the first 12
    are `cal`, the next 20 `test`, the rest `train`."""
    roles = {}
    pool = []
    for project in sorted(permissive):
        part = kev_projects.get(project)
        if part == "development":
            roles[project] = "validation"
        elif part == "test":
            roles[project] = "train"
        else:
            pool.append(project)
    pool.sort(key=lambda p: key("cr-project", p))
    for index, project in enumerate(pool):
        roles[project] = ("cal" if index < CAL_CR_PROJECTS else
                          "test" if index < CAL_CR_PROJECTS + TEST_CR_PROJECTS else "train")
    return roles


def _cr_kev(kev_dir: Path):
    projects, used = {}, set()
    for part, records in jc.kev_records(kev_dir).items():
        for record in records:
            if record["_meta"]["source"] != "codereviewer":
                continue
            project = record["_meta"]["group_id"].split("/", 1)[1]
            projects[project] = part
            used.add(jc.text_digest(record["state"]["diff"]))
    return projects, used


def build_codereviewer(zenodo_dir: Path, kev_dir: Path, licences: Path, cr_new: Path,
                       out: Path, train_rows: int, cal_rows: int, test_rows: int) -> dict:
    if jc.sha256_file(licences) != jc.KEV_LICENCES_SHA256:
        raise ReevalError(f"{licences}: not the pinned Kev licence table")
    for name, expected in jc.ZENODO_MEMBERS.items():
        if jc.sha256_file(zenodo_dir / name) != expected:
            raise ReevalError(f"{zenodo_dir / name}: sha256 is not the pinned member")
    table = json.loads(licences.read_text())["codereviewer"]
    permissive = {p for p, v in table.items()
                  if v and (v.get("spdx") or "").lower() in jc.PERMISSIVE}
    kev_projects, used = _cr_kev(kev_dir)
    cr_new_ids = set()
    for line in (cr_new / "requests.jsonl").read_text(encoding="utf-8").split("\n"):
        if line:
            row = json.loads(line)
            cr_new_ids.add(row["id"])
            used.add(jc.text_digest(row["state"]))
    roles = cr_project_roles(kev_projects, permissive)
    pools, counts, seen = defaultdict(list), Counter(), set()
    for name in sorted(jc.ZENODO_MEMBERS):
        with open(zenodo_dir / name, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                counts["rows"] += 1
                role = roles.get(row["proj"])
                if role not in ("train", "cal", "test"):
                    counts["not_a_v2_project"] += 1
                    continue
                if f"cr-new/{name.split('.')[0]}/L{line_number}" in cr_new_ids:
                    counts["used_by_cr_new"] += 1
                    continue
                patch = row["patch"]
                if not patch.strip() or len(patch) > jc.MAX_PATCH_CHARS:
                    counts["patch_size_dropped"] += 1
                    continue
                if jc.text_digest(patch) in used:
                    counts["used_by_kev_or_cr_new"] += 1
                    continue
                state = jc.codereviewer_state(patch, row.get("oldf"))
                if len(json.dumps(state)) > jc.MAX_STATE_CHARS:
                    counts["state_size_dropped"] += 1
                    continue
                text = prep.render_state("codereviewer", state, name)
                if LINE_BREAKS.search(text):
                    counts["line_separator_dropped"] += 1
                    continue
                digest = jc.text_digest(text)
                if digest in seen or digest in used:
                    counts["duplicate_state"] += 1
                    continue
                seen.add(digest)
                pools[role].append({
                    "id": f"cr-v2/{name.split('.')[0]}/L{line_number}", "project": row["proj"],
                    "label": row["y"] == 1, "text": text, "member": name, "line": line_number})
    limits = {"train": train_rows, "cal": cal_rows, "test": test_rows}
    rows, summary = [], {}
    for role in ("train", "cal", "test"):
        chosen = jc.select_pairs(pools[role], limits[role], f"v2-{role}")
        summary[role] = {"rows": len(chosen), "pool": len(pools[role]),
                         "projects": sorted({r["project"] for r in chosen}),
                         "labels": dict(Counter(str(r["label"]) for r in chosen))}
        for r in chosen:
            rows.append(jc.request_row(r["id"], r["text"], jc.CR_QUESTION, r["label"], {
                "split": role, "set": "cr-v2", "task": jc.CR_TASK, "source": "codereviewer",
                "group_id": f"codereviewer/{r['project']}",
                "project": table[r["project"]]["repo"],
                "repo_licence": table[r["project"]]["spdx"],
                "provenance": {"via": f"zenodo:{jc.ZENODO_RECORD}/{jc.ZENODO_FILE}:{r['member']}",
                               "line": r["line"]}}))
    report = {
        "format": "local-ai.judgment-eval-requests.v2", "set": "cr-v2",
        "project_roles": {role: sorted(p for p, v in roles.items() if v == role)
                          for role in ("train", "validation", "cal", "test")},
        "sources": {"zenodo": {"record": jc.ZENODO_RECORD, "members": jc.ZENODO_MEMBERS,
                               "licence": "CC-BY-4.0"},
                    "kev_manifest_sha256": prep.PINNED_MANIFEST_SHA256,
                    "kev_licences_sha256": jc.KEV_LICENCES_SHA256,
                    "cr_new_requests_sha256": jc.sha256_file(cr_new / "requests.jsonl")},
        "procedure": ("permissive CodeReviewer projects; v2 roles by seeded project hash "
                      f"({CAL_CR_PROJECTS} CAL, {TEST_CR_PROJECTS} TEST projects drawn from Kev "
                      "train projects and permissive projects Kev never used); every hunk Kev "
                      "or cr-new used excluded; Kev's state and size limits; label-balanced "
                      f"pairs within project, projects round-robin, seeded '{SEED}'"),
        "counts": dict(counts), "splits": summary,
    }
    return jc.write_requests(out, rows, report)


# ---- CommitPackFT -----------------------------------------------------------


class _Components:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, items):
        roots = [self.find(i) for i in items]
        for r in roots[1:]:
            a, b = sorted((roots[0], r))
            self.parent[self.find(b)] = self.find(a)


def cm_role(root: str) -> str:
    bucket = int(key("cm-group", root)[:8], 16) % 100
    edge = 0
    for role, share in CM_ROLE_BUCKETS.items():
        edge += share
        if bucket < edge:
            return role
    raise AssertionError("buckets sum to 100")


def _kev_commitpackft(kev_dir: Path):
    ids, digests, repos = set(), set(), set()
    for records in jc.kev_records(kev_dir).values():
        for record in records:
            if record["_meta"]["source"] != "commitpackft":
                continue
            meta = record["_meta"]
            ids.add(meta["id"])
            digests.add(jc.text_digest(record["state"]))
            repos.add(meta["repo"])
            repos.add(meta["group_id"].split("/", 1)[1])
    return ids, digests, repos


def build_commitpackft(public_dir: Path, kev_dir: Path, out: Path, limits: dict) -> dict:
    """Message-match requests from CommitPackFT repository components Kev never touched:
    at most one commit per component; the shown message is the commit's own subject
    (match) or the subject of a commit in the same language, role and another
    component (no match); labels alternate in seeded order."""
    summary_file = json.loads((public_dir / "public-data.json").read_text())
    kev_ids, kev_digests, kev_repos = _kev_commitpackft(kev_dir)
    comps, candidates, counts = _Components(), [], Counter()
    for name, digest in sorted(summary_file["outputs"].items()):
        if not name.startswith("commitpackft-"):
            continue
        path = public_dir / name
        if jc.sha256_file(path) != digest:
            raise ReevalError(f"{path}: sha256 does not match the job summary")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                counts["rows"] += 1
                rid = f"commitpackft/{r['lang']}/{r['row']}"
                subject = r["subject"].strip()
                repos = [x.strip() for x in r["repos"].split(",") if x.strip()]
                if rid in kev_ids or jc.text_digest(r["diff"]) in kev_digests:
                    counts["used_by_kev"] += 1
                    continue
                if len(r["diff"]) > MAX_CM_DIFF_CHARS:
                    counts["diff_size_dropped"] += 1
                    continue
                if not CM_SUBJECT_CHARS[0] <= len(subject) <= CM_SUBJECT_CHARS[1] or \
                        '"' in subject or LINE_BREAKS.search(subject + r["diff"]) or \
                        "\n" in subject:
                    counts["subject_or_separator_dropped"] += 1
                    continue
                comps.union(repos)
                candidates.append((rid, r, subject, repos))
    kev_roots = {comps.find(x) for x in kev_repos if x in comps.parent}
    by_role, seen = defaultdict(lambda: defaultdict(list)), set()
    for rid, r, subject, repos in candidates:
        root = comps.find(repos[0])
        if root in kev_roots:
            counts["component_shares_a_kev_repository"] += 1
            continue
        digest = jc.text_digest(r["diff"])
        if digest in seen:
            counts["duplicate_diff"] += 1
            continue
        seen.add(digest)
        by_role[cm_role(root)][root].append((rid, r, subject))
    rows, summary = [], {}
    for role in ("train", "validation", "cal", "test"):
        groups = by_role[role]
        # one commit per component, components in seeded order, languages round-robin
        firsts = defaultdict(list)
        for root in sorted(groups, key=lambda g: key("cm-order", role, g)):
            rid, r, subject = min(groups[root], key=lambda item: key("cm-pick", item[0]))
            firsts[r["lang"]].append((root, rid, r, subject))
        chosen, cursors = [], {lang: 0 for lang in firsts}
        langs = sorted(firsts, key=lambda lang: key("cm-lang", role, lang))
        while len(chosen) < limits[role] and any(cursors[k] < len(firsts[k]) for k in langs):
            for lang in langs:
                if len(chosen) >= limits[role]:
                    break
                if cursors[lang] < len(firsts[lang]):
                    chosen.append(firsts[lang][cursors[lang]])
                    cursors[lang] += 1
        by_lang = defaultdict(list)
        for item in chosen:
            by_lang[item[2]["lang"]].append(item)
        made = Counter()
        for index, (root, rid, r, subject) in enumerate(
                sorted(chosen, key=lambda item: key("cm-match", item[1]))):
            match = index % 2 == 0
            shown = subject
            if not match:
                pool = [o for o in by_lang[r["lang"]] if o[0] != root and
                        " ".join(o[3].casefold().split()) != " ".join(subject.casefold().split())]
                if not pool:
                    counts["no_negative_partner"] += 1
                    continue
                shown = min(pool, key=lambda o: key("cm-negative", rid, o[1]))[3]
            question = f'Does this commit message describe this diff? Message: "{shown}"'
            if not jc.CM_QUESTION.fullmatch(question):
                raise ReevalError(f"{rid}: question is not the declared CommitPackFT form")
            made[str(match)] += 1
            rows.append(jc.request_row(f"cm-v2/{rid}", "Diff:\n" + r["diff"], question, match, {
                "split": role, "set": "cm-v2", "task": jc.CM_TASK, "source": "commitpackft",
                "group_id": f"commitpackft/{root}", "repo": r["repos"].split(",")[0].strip(),
                "repo_licence": r["license"], "language": r["lang"], "commit": r["commit"],
                "provenance": {"via": f"hf:{summary_file.get('commitpackft_repo', 'bigcode/commitpackft')}"
                                      f":data/{r['lang']}/data.jsonl", "row": r["row"]}}))
        summary[role] = {"rows": sum(made.values()), "components": len(groups),
                         "labels": dict(made),
                         "languages": dict(Counter(item[2]["lang"] for item in chosen))}
    report = {
        "format": "local-ai.judgment-eval-requests.v2", "set": "cm-v2",
        "sources": {"public_data": {k: summary_file[k] for k in ("outputs", "job_source_sha256")},
                    "commitpackft": summary_file["commitpackft"],
                    "kev_manifest_sha256": prep.PINNED_MANIFEST_SHA256},
        "procedure": ("CommitPackFT (bigcode/commitpackft@fc56fe33, MIT; permissive per-row "
                      "repository licences) rows of repository components sharing no "
                      "repository with any Kev devtools-v1 row; v2 role by seeded component "
                      f"hash {CM_ROLE_BUCKETS}; diff <= {MAX_CM_DIFF_CHARS} chars; one commit per "
                      "component; Kev's message-match construction (negative = another "
                      "component's subject in the same language and role)"),
        "counts": dict(counts), "splits": summary,
    }
    return jc.write_requests(out, rows, report)


# ---- out-of-scope sets ------------------------------------------------------


def _choice_row(row_id, text, question, options, target, metadata):
    meta = dict(metadata, target=target, options=list(options), question_text=question,
                state_prefix=text[:200])
    return {"id": row_id, "state": text, "question": question, "kind": "choice",
            "options": list(options), "metadata": meta}


def _kev_state_text(record):
    state = record["state"]
    return state if isinstance(state, str) else prep.kev_render(state)


def _choice_from(record, qkey, set_name, split):
    question = record["questions"][qkey]
    labels = list(question["criteria"])
    options = [question["criteria"][k] for k in labels]
    return _choice_row(
        f"{set_name}/{question['src']}/{record['_meta']['row_sha256']}",
        _kev_state_text(record), question["instructions"], options,
        labels.index(question["label"]),
        {"split": split, "set": set_name, "task": question["src"],
         "source": record["_meta"]["source"], "group_id": record["_meta"]["group_id"],
         "kev_partition": record["_meta"]["split"], "row_sha256": record["_meta"]["row_sha256"]})


def build_ood(kev_dir: Path, kev_ood: Path, out: Path, per_source: int) -> dict:
    """Fresh OOD: Kev development/test predicate rows of the earlier OOD sources that
    the earlier `kev-ood` set did not select, plus when2call actions and CommitPackFT
    change types (choices) from Kev development/test, none scored before."""
    records = jc.kev_records(kev_dir)
    earlier = set()
    for line in (kev_ood / "requests.jsonl").read_text(encoding="utf-8").split("\n"):
        if line:
            earlier.add(json.loads(line)["metadata"]["row_sha256"])
    rows, summary = [], {}
    for source, (qkey, task) in jc.OOD_SOURCES.items():
        pool = [r for part in ("development", "test") for r in records[part]
                if r["_meta"]["source"] == source and qkey in r["questions"]
                and len(json.dumps(r["state"])) <= jc.MAX_STATE_CHARS
                and r["_meta"]["row_sha256"] not in earlier]
        pool.sort(key=lambda r: key("ood", r["_meta"]["row_sha256"]))
        chosen = pool[:per_source]
        summary[task] = {"rows": len(chosen), "pool": len(pool)}
        for record in chosen:
            question = record["questions"][qkey]
            rows.append(jc.request_row(
                f"ood-v2/{task}/{record['_meta']['row_sha256']}", _kev_state_text(record),
                question["instructions"], question["label"],
                {"split": "ood", "set": "ood-v2", "task": task, "source": source,
                 "group_id": record["_meta"]["group_id"],
                 "kev_partition": record["_meta"]["split"],
                 "row_sha256": record["_meta"]["row_sha256"]}))
    for source, qkey in (("when2call", "action"), ("commitpackft", "change_type")):
        pool = [r for part in ("development", "test") for r in records[part]
                if r["_meta"]["source"] == source and qkey in r["questions"]
                and len(json.dumps(r["state"])) <= jc.MAX_STATE_CHARS]
        pool.sort(key=lambda r: key("ood", r["_meta"]["row_sha256"]))
        chosen = pool[:per_source]
        summary[f"{source}_{qkey}"] = {"rows": len(chosen), "pool": len(pool)}
        rows += [_choice_from(r, qkey, "ood-v2", "ood") for r in chosen]
    report = {"format": "local-ai.judgment-eval-requests.v2", "set": "ood-v2",
              "sources": {"kev_manifest_sha256": prep.PINNED_MANIFEST_SHA256,
                          "kev_ood_requests_sha256": jc.sha256_file(kev_ood / "requests.jsonl")},
              "procedure": ("Kev devtools-v1 development+test rows never scored by an earlier "
                            "evaluation: prompt_injection/aegis_unsafe/flakeflagger_flaky rows "
                            "the earlier kev-ood set did not select, when2call actions and "
                            "CommitPackFT change types (choices over Kev's criteria texts); "
                            "seeded order"),
              "splits": summary}
    return jc.write_requests(out, rows, report)


def build_dev_ood(kev_dir: Path, out: Path, rows_wanted: int) -> dict:
    """Gate-development OOD only (never part of a pre-registered split): Kev *train*
    CommitPackFT change-type choices, the nearest out-of-scope questions."""
    records = jc.kev_records(kev_dir)
    pool = [r for r in records["train"] if r["_meta"]["source"] == "commitpackft"
            and "change_type" in r["questions"]
            and len(json.dumps(r["state"])) <= jc.MAX_STATE_CHARS]
    pool.sort(key=lambda r: key("dev-ood", r["_meta"]["row_sha256"]))
    rows = [_choice_from(r, "change_type", "dev-ood", "ood") for r in pool[:rows_wanted]]
    report = {"format": "local-ai.judgment-eval-requests.v2", "set": "dev-ood",
              "procedure": "Kev train CommitPackFT change-type choices; gate development only",
              "sources": {"kev_manifest_sha256": prep.PINNED_MANIFEST_SHA256}}
    return jc.write_requests(out, rows, report)


# ---- training directory -----------------------------------------------------


def assemble(sources: list[tuple[str, Path]], roles: dict, out: Path) -> dict:
    """Hard-link rows of feature directories into one `judgment_train` directory.

    `roles` maps a source name to a rule: {"split_map": {"old split/task" or "old
    split": new split},
    "drop_groups": [group ids], "keep_tasks": [tasks]}; rows mapping to no split are
    left out. Every kept row's `metadata.split` becomes its v2 split."""
    if out.exists():
        raise ReevalError(f"{out} exists")
    out.mkdir(parents=True)
    lines, index, counts = [], 0, Counter()
    for name, directory in sources:
        rule = roles[name]
        drop = set(rule.get("drop_groups", ()))
        tasks = set(rule.get("keep_tasks", ()))
        for line in (directory / "features.jsonl").read_text().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            meta = row["metadata"]
            split = rule["split_map"].get(f"{meta['split']}/{meta.get('task')}",
                                          rule["split_map"].get(meta["split"]))
            if split is None or meta.get("group_id") in drop or (
                    tasks and meta.get("task") not in tasks):
                counts[f"{name}/dropped"] += 1
                continue
            target = f"feature-{index:06d}.bin"
            os.link(directory / row["feature_file"], out / target)
            meta = dict(meta, split=split, v2_source=name, original_split=meta["split"])
            row = dict(row, feature_file=target, metadata=meta)
            lines.append(json.dumps(row, sort_keys=True))
            counts[f"{split}/{meta.get('task')}"] += 1
            index += 1
    payload = "\n".join(lines) + "\n"
    (out / "features.jsonl").write_text(payload)
    return {"rows": index, "counts": dict(sorted(counts.items())),
            "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest()}


# ---- feature gate (NumPy, lazy) ---------------------------------------------


def _numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("needs NumPy: uv run --with numpy python3 "
                         "tools/judgment_reeval.py ...") from error
    return np


def _decide_rows(directory: Path, split: str | None):
    np = _numpy()
    rows = []
    for line in (directory / "features.jsonl").read_text().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if split is not None and row["metadata"]["split"] != split:
            continue
        data = (directory / row["feature_file"]).read_bytes()
        rows.append(np.frombuffer(data[-2 * jt.WIDTH:], dtype="<f2").astype(np.float64))
    return np.stack(rows)


def fit_gate(directory: Path, out: Path, components: int = GATE_COMPONENTS,
             split: str = "train") -> dict:
    """PCA of the training `Decision:` rows: Mahalanobis distance in the top
    `components` directions plus the residual energy scaled by the mean residual
    variance, divided by the width. Fitted on `split` rows only."""
    np = _numpy()
    x = _decide_rows(directory, split)
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    var = s ** 2 / (len(x) - 1)
    residual = float(var[components:].sum() / (x.shape[1] - components))
    tensors = {"mean": ((x.shape[1],), mean.tolist()),
               "components": ((components, x.shape[1]), vt[:components].ravel().tolist()),
               "variance": ((components,), var[:components].tolist()),
               "residual_variance": ((1,), [residual])}
    metadata = {"format": "local-ai.judgment-gate.v1", "input": "Decision: row hidden state",
                "rows": len(x), "split": split, "components": components,
                "features_jsonl_sha256": jc.sha256_file(directory / "features.jsonl")}
    jt.write_safetensors(out, tensors, metadata)
    return dict(metadata, sha256=jc.sha256_file(out))


def load_gate(path: Path) -> dict:
    np = _numpy()
    tensors, metadata = jt.read_safetensors(path)
    (k, width), comps = tensors["components"]
    return {"mean": np.asarray(tensors["mean"][1]),
            "components": np.asarray(comps).reshape(k, width),
            "variance": np.asarray(tensors["variance"][1]),
            "residual_variance": float(tensors["residual_variance"][1][0]),
            "metadata": metadata}


def gate_distance(gate: dict, decide) -> float:
    y = decide - gate["mean"]
    z = gate["components"] @ y
    residual = y - gate["components"].T @ z
    total = float((z * z / gate["variance"]).sum() + residual @ residual
                  / gate["residual_variance"])
    return total / len(y)


def score(head_path: Path, gate_path: Path, sources, out: Path) -> dict:
    """jc.score's rows plus `gate_distance` per row."""
    np = _numpy()
    base = out.with_name(out.name + ".base")
    jc.score(head_path, sources, base)
    gate = load_gate(gate_path)
    distances = {}
    for name, directory, _ in sources:
        for line in (directory / "features.jsonl").read_text().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            data = (directory / row["feature_file"]).read_bytes()
            decide = np.frombuffer(data[-2 * jt.WIDTH:], dtype="<f2").astype(np.float64)
            distances[(name, row["id"])] = gate_distance(gate, decide)
    records = jc.load_scores(base)
    for r in records:
        r["gate_distance"] = distances[(r["set"], r["id"])]
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    out.write_text(payload)
    summary = json.loads(jc.score_summary_path(base).read_text())
    summary.update(gate_sha256=jc.sha256_file(gate_path),
                   scores_sha256=hashlib.sha256(payload.encode()).hexdigest())
    jc.score_summary_path(out).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    base.unlink()
    jc.score_summary_path(base).unlink()
    return summary


# ---- evaluation (pure Python) -----------------------------------------------


def selective_at(preds, thresholds) -> dict:
    """Answered = confidence >= the row's own threshold (None refuses the row)."""
    answered = [p for p, t in zip(preds, thresholds) if t is not None and p["confidence"] >= t]
    wrong = sum(not p["correct"] for p in answered)
    n = len(preds)
    return {"answered": len(answered), "coverage": len(answered) / n if n else 0.0,
            "risk": wrong / len(answered) if answered else None,
            "risk_wilson95": list(jc.wilson(wrong, len(answered))) if answered else None}


def fit_family(rows) -> tuple[jc.Calibrator, dict]:
    cv = {m: jc.cross_validated_nll(m, rows) for m in jc.candidate_methods(rows)}
    method = min(cv, key=cv.get)
    return jc.fit_calibrator(method, rows), cv


def evaluate(records, cal_specs, test_specs, ood_specs, train_specs, scope,
             families=FAMILIES, target_risk: float = jc.TARGET_RISK) -> dict:
    """Per-family calibration (method by grouped CV NLL) and per-family threshold on
    CAL; gate threshold at the CAL 99th percentile of the training-fitted gate
    distance; metrics on CAL/TEST/OOD in judgment_calibrate's report shape."""
    def pick(specs):
        return [r for s in specs for r in jc.select_rows(records, s)]
    cal, test, ood, train = pick(cal_specs), pick(test_specs), pick(ood_specs), pick(train_specs)
    ids = [r["id"] for r in cal + test]
    if len(ids) != len(set(ids)):
        raise ReevalError("CAL and TEST share rows")
    for a, b, name in ((cal, test, "CAL and TEST"), (train, cal + test, "train and CAL/TEST")):
        shared = {r["group"] for r in a} & {r["group"] for r in b}
        if shared:
            raise ReevalError(f"{name} share groups: {sorted(shared)[:5]}")
    calibrators, thresholds, cv = {}, {}, {}
    for family in families:
        rows = [r for r in cal if r["family"] == family]
        if not rows:
            raise ReevalError(f"CAL has no {family} rows")
        calibrators[family], cv[family] = fit_family(rows)
        thresholds[family] = jc.choose_threshold(jc.predictions(calibrators[family], rows),
                                                 target_risk)
    pooled = jc.fit_calibrator("temperature", cal)  # out-of-family rows (diagnostic only)
    finite = [t for t in thresholds.values() if t is not None]
    strictest = max(finite) if finite else None
    gate_threshold = sorted(r["gate_distance"] for r in cal)[
        min(len(cal) - 1, math.ceil(GATE_PERCENTILE * len(cal)) - 1)]

    def predict(rows):
        preds, taus = [], []
        for r in rows:
            calibrator = calibrators.get(r["family"], pooled)
            preds += jc.predictions(calibrator, [r])
            taus.append(thresholds[r["family"]] if r["family"] in calibrators else strictest)
        return preds, taus

    report = {"protocol": "docs/DECISIONS_EVALUATION.md part 3",
              "roles": {"cal": cal_specs, "test": test_specs, "ood": ood_specs,
                        "train": train_specs},
              "cross_validated_nll": cv,
              "calibration": {f: c.to_json() for f, c in calibrators.items()},
              "target_risk": target_risk, "thresholds": thresholds,
              "gate": {"threshold": gate_threshold, "percentile": GATE_PERCENTILE},
              "scope": scope, "splits": {}}
    raw = jc.Calibrator("temperature", 1.0)
    for name, rows in (("cal", cal), ("test", test), ("ood", ood)):
        if not rows:
            continue
        preds, taus = predict(rows)
        by_family = {}
        for family in sorted({r["family"] for r in rows}):
            index = [i for i, r in enumerate(rows) if r["family"] == family]
            fp = [preds[i] for i in index]
            by_family[family] = {
                "model": jc.metrics(fp),
                "majority": jc.majority_baseline(train, [rows[i] for i in index]),
                "selective": selective_at(fp, [taus[i] for i in index]),
                "gate_refused": sum(rows[i]["gate_distance"] > gate_threshold
                                    for i in index) / len(index)}
        entry = {"raw": jc.metrics(jc.predictions(raw, rows)),
                 "calibrated": jc.metrics(preds),
                 "selective": dict(jc.selective(preds), at_threshold=selective_at(preds, taus)),
                 "by_family": by_family,
                 "majority": jc.majority_baseline(train, rows),
                 "gate_refused": sum(r["gate_distance"] > gate_threshold
                                     for r in rows) / len(rows),
                 "scope_refused": sum(not jc.in_declared_scope(r, scope)
                                      for r in rows) / len(rows)}
        by_project = defaultdict(list)
        for r, p, t in zip(rows, preds, taus):
            by_project[r["group"]].append((p, t))
        entry["by_group"] = {g: dict(selective_at([p for p, _ in v], [t for _, t in v]),
                                     count=len(v),
                                     accuracy=sum(p["correct"] for p, _ in v) / len(v))
                             for g, v in sorted(by_project.items())
                             if not g.startswith("commitpackft/")}
        report["splits"][name] = entry
    return report


def calibration_artifact(report: dict, head_sha256: str, gate_sha256: str) -> dict:
    """Everything fitted on CAL plus the training-only gate's hash, pinned to the head,
    with provenance, acceptance and a content hash over the rest."""
    body = {
        "format": "local-ai.judgment-calibration.v2",
        "head_sha256": head_sha256,
        "gate_sha256": gate_sha256,
        "families": {f: {"calibration": report["calibration"][f],
                         "threshold": report["thresholds"][f]} for f in report["calibration"]},
        "refusal": {"target_risk": report["target_risk"], "min_answered": jc.MIN_ANSWERED,
                    "confidence": "calibrated probability of the selected option",
                    "rule": "per declared family; a null threshold refuses the family"},
        "gate": {"input": "Decision: row hidden state", "file_sha256": gate_sha256,
                 "distance": "PCA Mahalanobis + scaled residual energy, divided by width",
                 "threshold": report["gate"]["threshold"],
                 "percentile": report["gate"]["percentile"]},
        "scope": report["scope"],
        "provenance": {"roles": report["roles"], "scores_sha256": report["scores_sha256"],
                       "sources": report.get("score_summary", {}).get("sources"),
                       "tool_sha256": report["tool_sha256"], "protocol": report["protocol"]},
        "acceptance": report["acceptance"],
    }
    return dict(body, content_sha256=jc.canonical_sha256(body))


def verify_calibration_artifact(artifact: dict, head_sha256: str, gate_sha256: str) -> None:
    jc.verify_calibration_artifact(artifact, head_sha256)
    if artifact.get("gate_sha256") != gate_sha256:
        raise ReevalError("calibration artifact was fitted with a different gate")


def markdown(report: dict, title: str) -> str:
    flat_cv = {f"{family}/{method}": value
               for family, methods in report["cross_validated_nll"].items()
               for method, value in methods.items()}
    text = jc.markdown(dict(report, cross_validated_nll=flat_cv,
                            threshold=report["thresholds"]), title)
    lines = ["", "| Split | Group | Rows | Accuracy | Coverage@τ | Risk@τ |",
             "|---|---|---:|---:|---:|---:|"]
    for name, entry in report["splits"].items():
        for group, g in entry["by_group"].items():
            risk = "-" if g["risk"] is None else f"{g['risk']:.3f}"
            lines.append(f"| {name} | {group} | {g['count']} | {g['accuracy']:.3f} | "
                         f"{g['coverage']:.3f} | {risk} |")
    return text + "\n".join(lines) + "\n"


# ---- offline decisions and HTTP parity --------------------------------------


def calibrator_from_json(spec: dict) -> jc.Calibrator:
    if spec["method"] == "platt":
        return jc.Calibrator("platt", slope=spec["slope"], bias=spec["bias"])
    return jc.Calibrator("temperature", temperature=spec["temperature"])


def offline_decisions(records, artifact: dict) -> list[dict]:
    """What a server holding `artifact` answers for each scored row: the declared
    scope check, then (in scope) prompt length, gate, family threshold and confidence,
    in that order, with the family's calibrated P(Yes)."""
    families, scope = artifact["families"], artifact["scope"]
    gate_threshold = artifact["gate"]["threshold"]
    # The token limit refuses an in-scope question after rendering, as the server does.
    unlimited = dict(scope, max_prompt_tokens=math.inf)
    out = []
    for r in records:
        in_scope = (r["yes_index"] is not None and r["family"] in families
                    and jc.in_declared_scope(r, unlimited))
        entry = {"id": r["id"], "set": r["set"], "family": r["family"], "in_scope": in_scope}
        if in_scope:
            family = families[r["family"]]
            probs = calibrator_from_json(family["calibration"]).probabilities(
                r["raw"], r["yes_index"])
            confidence = max(probs)
            if r["prompt_tokens"] > scope["max_prompt_tokens"]:
                refusal = "prompt_too_long"
            elif not r["gate_distance"] <= gate_threshold:
                refusal = "gate"
            elif family["threshold"] is None:
                refusal = "family_refused"
            elif confidence < family["threshold"]:
                refusal = "low_confidence"
            else:
                refusal = None
            entry.update(p_yes=probs[r["yes_index"]], confidence=confidence,
                         gate_distance=r["gate_distance"], refusal=refusal)
        out.append(entry)
    return out


def http_parity(url: str, requests_path: Path, expected_path: Path, ids: list[str],
                model: str = "bonsai", opener=None) -> dict:
    """POST each request (one predicate) to a running server's /v1/decisions and
    compare with the offline decision of the same row."""
    import urllib.request
    opener = opener or urllib.request.urlopen
    requests = {}
    for line in requests_path.read_text(encoding="utf-8").split("\n"):
        if line:
            row = json.loads(line)
            requests[row["id"]] = row
    expected = {}
    for line in expected_path.read_text().split("\n"):
        if line:
            row = json.loads(line)
            expected[row["id"]] = row
    rows, mismatched = [], []
    for row_id in ids:
        request, want = requests[row_id], expected[row_id]
        if request["kind"] == "predicate":
            question = {"type": "predicate", "name": row_id, "instructions": request["question"]}
        else:
            question = {"type": "choice", "name": row_id, "instructions": request["question"],
                        "choices": [{"value": v} for v in request["options"]]}
        body = json.dumps({"model": model, "input": request["state"],
                           "questions": [question]}).encode()
        http = urllib.request.Request(url, data=body,
                                      headers={"Content-Type": "application/json"})
        with opener(http, timeout=600) as response:
            answer = json.loads(response.read())["answers"][0]
        answered = want.get("in_scope") and want.get("refusal") is None
        ok = answer["name"] == row_id and (
            (answered and answer["type"] == "predicate"
             and abs(answer["probability"] - want["p_yes"]) <= 1e-6)
            or (not answered and answer["type"] == "refusal"))
        rows.append({"id": row_id, "answer": answer, "expected": want, "agree": ok})
        if not ok:
            mismatched.append(row_id)
    return {"rows": len(rows), "agree": len(rows) - len(mismatched), "mismatched": mismatched,
            "answered": sum(r["answer"]["type"] == "predicate" for r in rows), "details": rows}


# ---- CLI --------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-codereviewer")
    for name in ("zenodo-dir", "kev-dir", "licences", "cr-new", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--train-rows", type=int, default=1600)
    p.add_argument("--cal-rows", type=int, default=400)
    p.add_argument("--test-rows", type=int, default=600)
    p = sub.add_parser("build-commitpackft")
    for name in ("public-dir", "kev-dir", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--train-rows", type=int, default=1200)
    p.add_argument("--validation-rows", type=int, default=200)
    p.add_argument("--cal-rows", type=int, default=400)
    p.add_argument("--test-rows", type=int, default=600)
    p = sub.add_parser("build-ood")
    for name in ("kev-dir", "kev-ood", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--per-source", type=int, default=150)
    p = sub.add_parser("build-dev-ood")
    p.add_argument("--kev-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=150)
    p = sub.add_parser("assemble")
    p.add_argument("--source", action="append", required=True, help="NAME=FEATURE_DIR")
    p.add_argument("--roles", type=Path, required=True, help="JSON {name: rule}")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("fit-gate")
    p.add_argument("features", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--components", type=int, default=GATE_COMPONENTS)
    p = sub.add_parser("score")
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--source", action="append", required=True,
                   help="NAME=FEATURE_DIR[,TEXT_ROWS.jsonl]")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    for name in ("cal", "test", "train"):
        p.add_argument("--" + name, action="append", required=True)
    p.add_argument("--ood", action="append", default=[])
    p.add_argument("--title", default="Evaluation (protocol v2)")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("offline")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True, help="calibration.json")
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("http-parity")
    p.add_argument("--url", default="http://127.0.0.1:8114/v1/decisions")
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True, help="one row id per line")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-codereviewer":
        result = build_codereviewer(args.zenodo_dir, args.kev_dir, args.licences, args.cr_new,
                                    args.out, args.train_rows, args.cal_rows, args.test_rows)
    elif args.command == "build-commitpackft":
        result = build_commitpackft(args.public_dir, args.kev_dir, args.out, {
            "train": args.train_rows, "validation": args.validation_rows,
            "cal": args.cal_rows, "test": args.test_rows})
    elif args.command == "build-ood":
        result = build_ood(args.kev_dir, args.kev_ood, args.out, args.per_source)
    elif args.command == "build-dev-ood":
        result = build_dev_ood(args.kev_dir, args.out, args.rows)
    elif args.command == "assemble":
        sources = [(s.partition("=")[0], Path(s.partition("=")[2])) for s in args.source]
        result = assemble(sources, json.loads(args.roles.read_text()), args.out)
    elif args.command == "fit-gate":
        result = fit_gate(args.features, args.out, args.components)
    elif args.command == "score":
        sources = []
        for spec in args.source:
            name, _, rest = spec.partition("=")
            directory, _, texts = rest.partition(",")
            sources.append((name, Path(directory), Path(texts) if texts else None))
        result = score(args.head, args.gate, sources, args.out)
    elif args.command == "offline":
        summary = jc.check_scores(args.scores, args.head)
        artifact = json.loads(args.artifact.read_text())
        verify_calibration_artifact(artifact, jc.sha256_file(args.head), jc.sha256_file(args.gate))
        if summary.get("gate_sha256") != artifact["gate_sha256"]:
            raise ReevalError("scores were computed with a different gate")
        rows = offline_decisions(jc.load_scores(args.scores), artifact)
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        args.out.write_text(payload)
        result = {"rows": len(rows), "in_scope": sum(r["in_scope"] for r in rows),
                  "answered": sum(r.get("refusal", "x") is None for r in rows),
                  "sha256": hashlib.sha256(payload.encode()).hexdigest()}
    elif args.command == "http-parity":
        ids = [line for line in args.ids.read_text().split("\n") if line]
        report = http_parity(args.url, args.requests, args.expected, ids)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        result = {k: v for k, v in report.items() if k != "details"}
    else:
        summary = jc.check_scores(args.scores, args.head)
        if summary.get("gate_sha256") != jc.sha256_file(args.gate):
            raise ReevalError("scores were computed with a different gate")
        records = jc.load_scores(args.scores)
        report = evaluate(records, args.cal, args.test, args.ood, args.train, jc.DEVTOOLS_SCOPE)
        report["acceptance"] = jc.acceptance(report, list(FAMILIES), False)
        report["scores_sha256"] = jc.sha256_file(args.scores)
        report["score_summary"] = summary
        report["tool_sha256"] = {"judgment_reeval.py": jc.sha256_file(Path(__file__)),
                                 "judgment_calibrate.py": jc.sha256_file(Path(jc.__file__))}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        (args.out / "report.md").write_text(markdown(report, args.title))
        artifact = calibration_artifact(report, jc.sha256_file(args.head),
                                        jc.sha256_file(args.gate))
        (args.out / "calibration.json").write_text(json.dumps(artifact, indent=1) + "\n")
        result = {"acceptance": report["acceptance"], "out": str(args.out),
                  "calibration_content_sha256": artifact["content_sha256"]}
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
