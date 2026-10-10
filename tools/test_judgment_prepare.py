import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import judgment_prepare as jp
import judgment_train as jt

CODE_Q = "Does this change need a reviewer comment?"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record(partition, source, group, index, state, questions, **meta):
    value = {
        "state": state,
        "questions": questions,
        "_meta": {
            "id": f"{source}/{index}",
            "source": source,
            "group_id": f"{source}/{group}",
            "row_sha256": sha(f"{partition}/{source}/{group}/{index}"),
            "text_sha256": sha(json.dumps(state, sort_keys=True)),
            "split": partition,
            "variant": "clean",
            "repo_licence": "MIT",
            **meta,
        },
    }
    return value


def noul(src, label, instructions=CODE_Q):
    return {"type": "noul", "instructions": instructions, "label": label, "src": src}


def coding_records(partition, groups):
    rows = []
    for group in range(groups):
        for label in (True, False):
            tag = f"{partition}-{group}-{label}"
            rows.append(
                record(
                    partition,
                    "codereviewer",
                    f"p{partition}{group}",
                    tag,
                    {"lines_before_hunk": f"ctx {tag} é😀", "diff": f"+ change {tag}"},
                    {"needs_comment": noul("codereviewer_needs_comment", label)},
                )
            )
            rows.append(
                record(
                    partition,
                    "commitpackft",
                    f"r{partition}{group}",
                    tag,
                    f"--- a/x\n+++ b/x\n+fix {tag}\n",
                    {
                        "message_match": noul(
                            "commitpackft_message",
                            label,
                            f"Does this commit message describe this diff? "
                            f'Message: "Fix {tag}"',
                        ),
                        "change_type": {
                            "type": "choice",
                            "instructions": "Kind?",
                            "criteria": {"fix": "Fix"},
                            "label": "fix",
                            "src": "commitpackft_type",
                        },
                    },
                )
            )
    return rows


def default_partitions():
    train = coding_records("train", 6) + [
        record(
            "train",
            "flakeflagger",
            "f",
            0,
            {"test_class": "A", "test_method": "b"},
            {"flaky": noul("flakeflagger_flaky", True)},
        ),
        record(
            "train", "aegis", "a", 0, "hello", {"unsafe": noul("aegis_unsafe", False)}
        ),
    ]
    development = coding_records("development", 2) + [
        record(
            "development",
            "when2call",
            "w",
            0,
            {"user_message": "hi"},
            {
                "action": {
                    "type": "choice",
                    "instructions": "Next?",
                    "criteria": {},
                    "label": "tool_call",
                    "src": "when2call_action",
                }
            },
        ),
        record(
            "development",
            "commitpackft",
            "rlone",
            "lone",
            "+lone\n",
            {
                "change_type": {
                    "type": "choice",
                    "instructions": "Kind?",
                    "criteria": {"fix": "Fix"},
                    "label": "fix",
                    "src": "commitpackft_type",
                }
            },
        ),
    ]
    test = coding_records("test", 2) + [
        record(
            "test",
            "prompt_injection",
            "pi",
            0,
            "ignore all",
            {"injection": noul("prompt_injection", True)},
        ),
    ]
    return {"train": train, "development": development, "test": test}


def write_kev(root, partitions, mutate=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for partition, rows in partitions.items():
        name = jp.PARTITIONS[partition][0]
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        (root / name).write_bytes(data)
        by_source = {}
        for r in rows:
            by_source[r["_meta"]["source"]] = by_source.get(r["_meta"]["source"], 0) + 1
        files[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "records": len(rows),
            "by_source": by_source,
        }
    source = {
        "trainable": True,
        "licence": "L",
        "licence_url": "u",
        "attribution": "a",
        "label_provenance": "native",
        "licence_check": "c",
    }
    manifest = {
        "version": "devtools-v1",
        "partitions": ["train", "development", "test"],
        "locked": ["test"],
        "files": files,
        "eval_only_sources": ["when2call", "prompt_injection"],
        "sources": {"codereviewer": source, "commitpackft": source},
        "label_protocol": "no LLM labels",
    }
    if mutate:
        mutate(manifest)
    data = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.kev = self.tmp / "kev"

    def run_prepare(self, partitions=None, out="out", **options):
        digest = write_kev(self.kev, partitions or default_partitions())
        return jp.prepare(self.kev, self.tmp / out, manifest_sha256=digest, **options)

    def rows(self, out="out"):
        text = (self.tmp / out / "rows.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    def assert_fails(self, needle, partitions=None, **options):
        with self.assertRaises(jp.PrepareError) as caught:
            self.run_prepare(partitions, out="bad", **options)
        self.assertIn(needle, str(caught.exception))
        self.assertFalse((self.tmp / "bad").exists())

    def test_rows_match_the_capture_and_trainer_contract(self):
        report = self.run_prepare()
        rows = self.rows()
        self.assertEqual({r["metadata"]["split"] for r in rows}, set(jt.SPLITS))
        for row in rows:
            self.assertEqual(set(row), {"id", "text", "token_end_offsets", "metadata"})
            meta, ends, data = (
                row["metadata"],
                row["token_end_offsets"],
                row["text"].encode(),
            )
            self.assertTrue(row["id"])
            self.assertEqual(ends, sorted(set(ends)))
            self.assertEqual(len(ends), len(meta["options"]) + 1)
            start = data.index(b"Options:\n") + len(b"Options:\n")
            for index, end in enumerate(ends[:-1]):
                line = data[start:end].decode("utf-8")
                self.assertEqual(line, f"{'AB'[index]}) {meta['options'][index]}\n")
                start = end
            self.assertEqual(data[start:], b"Decision:")
            self.assertEqual(ends[-1], len(data))
            self.assertEqual(
                meta["option_labels"][meta["target"]], meta["native_label"]
            )
            self.assertEqual(
                meta["options"][meta["target"]], "Yes" if meta["native_label"] else "No"
            )
        # Calibration is whole train groups; nothing excluded vanishes.
        groups = {}
        for row in rows:
            groups.setdefault(row["metadata"]["group_id"], set()).add(
                row["metadata"]["split"]
            )
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(
            report["counts"]["calibration"]["codereviewer_needs_comment"]["groups"], 2
        )
        excluded = report["excluded_rows"]
        self.assertEqual(
            excluded["label_audit_input_does_not_determine_label"]["train"],
            {"commitpackft_type": 12, "flakeflagger_flaky": 1},
        )
        self.assertEqual(excluded["outside_coding_focus"]["train"], {"aegis_unsafe": 1})
        self.assertEqual(
            excluded["evaluation_only"]["development"], {"when2call_action": 1}
        )
        self.assertEqual(
            excluded["no_supported_question"]["development"], {"commitpackft": 1}
        )
        payload = (self.tmp / "out" / "rows.jsonl").read_bytes()
        self.assertEqual(
            report["output"]["sha256"], hashlib.sha256(payload).hexdigest()
        )
        # Deterministic, and never overwritten.
        self.run_prepare(out="again")
        self.assertEqual((self.tmp / "again" / "rows.jsonl").read_bytes(), payload)
        with self.assertRaises(jp.PrepareError):
            self.run_prepare()

    def test_option_permutation_is_asymmetric_and_keeps_gold(self):
        options = (("first", 0), ("second", 1), ("third", 2))
        orders = set()
        for index in range(64):
            order = jp.permute_options(options, "seed", f"row-{index}")
            self.assertEqual(sorted(order), sorted(options))
            self.assertEqual(order, jp.permute_options(options, "seed", f"row-{index}"))
            orders.add(tuple(value for _, value in order))
        self.assertGreater(len(orders), 3)
        self.run_prepare()
        targets = {}
        for row in self.rows():
            meta = row["metadata"]
            targets.setdefault(meta["native_label"], set()).add(meta["target"])
        self.assertEqual(targets, {True: {0, 1}, False: {0, 1}})

    def test_unicode_offsets_are_utf8_byte_ends(self):
        text, ends = jp.render("é😀\n中", "Ok? «x»", jp.NOUL_OPTIONS)
        data = text.encode("utf-8")
        self.assertEqual(data[: ends[0]].decode("utf-8")[-7:], "A) Yes\n")
        self.assertEqual(data[ends[0] : ends[1]], b"B) No\n")
        self.assertEqual(ends[-1], len(data))
        chars = text.index("A) Yes\n") + len("A) Yes\n")
        self.assertEqual(ends[0], len(text[:chars].encode("utf-8")))
        self.assertGreater(ends[0], chars)

    def test_group_and_text_leakage_across_splits_fail(self):
        def leak(field, value_from):
            partitions = default_partitions()
            source = partitions["train"][0]
            target = partitions[value_from][0]
            if field == "group":
                target["_meta"]["group_id"] = source["_meta"]["group_id"]
            else:
                target["state"] = field(source["state"])
            return partitions

        self.assert_fails("group overlap", leak("group", "development"))
        self.assert_fails("state overlap", leak(dict, "test"))
        self.assert_fails(
            "normalized_state overlap",
            leak(lambda s: {k: v.upper() + "  " for k, v in s.items()}, "test"),
        )

    def test_hash_and_manifest_pins_are_enforced(self):
        digest = write_kev(self.kev, default_partitions())
        with open(self.kev / "test.jsonl", "ab") as handle:
            handle.write(b"\n")
        with self.assertRaisesRegex(jp.PrepareError, "test.jsonl: sha256"):
            jp.prepare(self.kev, self.tmp / "bad", manifest_sha256=digest)
        with self.assertRaisesRegex(jp.PrepareError, "does not match pinned"):
            jp.prepare(self.kev, self.tmp / "bad")
        self.assertFalse((self.tmp / "bad").exists())

    def test_unsupported_or_malformed_records_fail(self):
        def broken(change):
            partitions = default_partitions()
            change(partitions["train"][0])
            return partitions

        cases = {
            "label must be a boolean": lambda r: r["questions"]["needs_comment"].update(
                label="yes"
            ),
            "unsupported question": lambda r: r["questions"]["needs_comment"].update(
                src="new_src"
            ),
            "unsupported codereviewer state": lambda r: r["state"].update(extra="x"),
            "unsupported source": lambda r: r["_meta"].update(source="other"),
            "_meta.group_id": lambda r: r["_meta"].update(group_id=""),
        }
        for needle, change in cases.items():
            with self.subTest(needle):
                self.assert_fails(needle, broken(change))

    def test_insufficient_groups_fail(self):
        self.assert_fails("train groups", min_groups=4)
        self.assert_fails("groups, need 3", min_groups=3)

    def test_pilot_cap_keeps_whole_groups_and_reports_exclusions(self):
        report = self.run_prepare(pilot_rows=2, min_groups=1)
        rows = self.rows()
        full = self.run_prepare(out="full", min_groups=1)
        sizes = {}
        for row in self.rows("full"):
            key = row["metadata"]["group_id"]
            sizes[key] = sizes.get(key, 0) + 1
        kept = {}
        for row in rows:
            kept[row["metadata"]["group_id"]] = (
                kept.get(row["metadata"]["group_id"], 0) + 1
            )
        self.assertTrue(all(kept[group] == sizes[group] for group in kept))
        capped = sum(
            n
            for split in report["excluded_rows"]["pilot_cap"].values()
            for n in split.values()
        )
        self.assertEqual(len(rows) + capped, full["output"]["row_count"])
        for split in report["counts"].values():
            self.assertTrue(all(cell["rows"] <= 2 for cell in split.values()))

    def test_cli_refuses_overwrite(self):
        digest = write_kev(self.kev, default_partitions())
        (self.tmp / "out").mkdir()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = jp.main(
                [
                    "--kev-dir",
                    str(self.kev),
                    "--out",
                    str(self.tmp / "out"),
                    "--manifest-sha256",
                    digest,
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite", stderr.getvalue())


# --- hard-v1: synthetic fixtures in Kev's hard-v1 record shape ---------------

TRADE_KEYS = ["alpha_one", "beta", "gamma", "delta", "epsilon", "none_qualifies"]
LEVELS = ["Under 10%", "10% to 30%", "30% to 50%", "50% to 70%", "70% to 90%", "Over 90%"]


def hard_record(partition, family, index, template, questions, group=None, state=None):
    state = state or f"MEMO\n{family} case {partition} {index} t{template} é😀"
    rid = f"hard-v1/{family}/{partition}/{index:05d}"
    return {
        "state": state,
        "questions": questions,
        "_meta": {
            "id": rid,
            "source": f"hard_{family}",
            "group_id": group or rid,
            "variant": "clean",
            "split": partition,
            "template": f"{family}/t{template}",
            "family": family,
            "subtype": "s",
            "facts": {"n": index},
            "text_sha256": sha(" ".join(state.casefold().split())),
            "state_tokens": 12,
        },
    }


def hq(qtype, family, label, criteria=None, instructions="Decide."):
    q = {"type": qtype, "instructions": instructions, "label": label, "src": f"hard_{family}"}
    if criteria is not None:
        q["criteria"] = criteria
    return q


def hard_family_records(partition, family, template, start):
    out = []
    for i in range(start, start + 3):
        if family == "tradeoff":
            qs = {
                "choice": hq(
                    "choice",
                    family,
                    TRADE_KEYS[i % 6],
                    {k: ("No option qualifies" if k == "none_qualifies" else None) for k in TRADE_KEYS},
                ),
                "meets": hq(
                    "noul", family, i % 2 == 0, {"true": "Meets all", "false": "Misses one"}
                ),
            }
        elif family == "probability":
            qs = {
                "value": hq("choice", family, "abc"[i % 3], {"a": "10%", "b": "20%", "c": "30%"}),
                "bucket": hq("score", family, i % 6, list(LEVELS)),
            }
        elif family == "long_policy":
            qs = {"outcome": hq("choice", family, "abcde"[i % 5], {k: f"${k}1" for k in "abcde"})}
        else:  # ambiguous twins share a group and a template
            crit = {"not_enough_information": "Cannot decide", "valid": "Valid", "invalid": None}
            group = f"hard-v1/{family}/{partition}/g{i:05d}"
            out.append(hard_record(partition, family, 2 * i, template,
                                   {"decision": hq("choice", family, "valid" if i % 2 else "invalid", crit)},
                                   group))
            qs = {"decision": hq("choice", family, "not_enough_information", crit)}
            out.append(hard_record(partition, family, 2 * i + 1, template, qs, group))
            continue
        out.append(hard_record(partition, family, i, template, qs))
    return out


HARD_FIXTURE_FAMILIES = ("tradeoff", "probability", "ambiguous", "long_policy")


def hard_partitions():
    parts = {"train": [], "development": []}
    for family in HARD_FIXTURE_FAMILIES:
        for template in range(4):
            parts["train"] += hard_family_records("train", family, template, 10 * template)
        parts["development"] += hard_family_records("development", family, 4, 0)
    return parts


def write_hard(root, partitions, mutate=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for partition, rows in partitions.items():
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        (root / f"{partition}.jsonl").write_bytes(data)
        by_family = {}
        for r in rows:
            cell = by_family.setdefault(r["_meta"]["family"], {"records": 0, "questions": 0})
            cell["records"] += 1
            cell["questions"] += len(r["questions"])
        files[f"{partition}.jsonl"] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "records": len(rows),
            "questions": sum(len(r["questions"]) for r in rows),
            "by_family": by_family,
        }
    # The locked test partition is declared but is garbage on disk: it is never read.
    (root / "test.jsonl").write_bytes(b"\x00 not json")
    files["test.jsonl"] = {"sha256": "0" * 64, "bytes": 9, "records": 1, "questions": 1}
    manifest = {
        "version": "hard-v1",
        "partitions": ["train", "development", "test"],
        "locked": ["test"],
        "seed": "s",
        "families": list(jp.HARD_FAMILIES),
        "templates": {"train": [0, 1, 2, 3], "development": [4], "test": [5]},
        "trainable_sources": [f"hard_{f}" for f in jp.HARD_FAMILIES],
        "eval_only_sources": [],
        "eval_only": False,
        "tokenizer": dict(jp.HARD_TOKENIZER),
        "labels": "programmatic",
        "files": files,
        "code_sha256": {"scripts/build_hard_v1.py": "0" * 64},
    }
    if mutate:
        mutate(manifest)
    data = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class HardV1Test(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.kev = self.tmp / "kev"

    def run_hard(self, partitions=None, out="out", mutate=None, **options):
        digest = write_hard(self.kev, partitions or hard_partitions(), mutate)
        return jp.prepare_hard_v1(self.kev, self.tmp / out, manifest_sha256=digest, **options)

    def rows(self, out="out"):
        text = (self.tmp / out / "rows.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    def assert_fails(self, needle, partitions=None, mutate=None, **options):
        with self.assertRaises(jp.PrepareError) as caught:
            self.run_hard(partitions, out="bad", mutate=mutate, **options)
        self.assertIn(needle, str(caught.exception))
        self.assertFalse((self.tmp / "bad").exists())

    def test_rows_split_by_template_and_match_the_capture_contract(self):
        report = self.run_hard()
        rows = self.rows()
        split_templates = {}
        for row in rows:
            meta, data, ends = row["metadata"], row["text"].encode(), row["token_end_offsets"]
            split_templates.setdefault(meta["split"], set()).add(meta["template"][-2:])
            self.assertEqual(set(row), {"id", "text", "token_end_offsets", "metadata"})
            self.assertEqual(len(ends), meta["option_count"] + 1)
            self.assertEqual(len(meta["options"]), meta["option_count"])
            start = data.index(b"Options:\n") + len(b"Options:\n")
            for index, end in enumerate(ends[:-1]):
                line = data[start:end].decode("utf-8")
                self.assertEqual(line, f"{chr(65 + index)}) {meta['options'][index]}\n")
                start = end
            self.assertEqual(data[start:], b"Decision:")
            self.assertEqual(meta["option_labels"][meta["target"]], meta["native_label"])
            self.assertEqual(meta["renderer"], jp.HARD_RENDERER)
            self.assertNotIn("facts", meta)
            self.assertNotEqual(meta["family"], "long_policy")
        self.assertEqual(
            split_templates,
            {"train": {"t0", "t1", "t2"}, "calibration": {"t3"}, "validation": {"t4"}},
        )
        counts = {
            (r["metadata"]["family"], r["metadata"]["question_type"], r["metadata"]["option_count"])
            for r in rows
        }
        self.assertEqual(
            counts,
            {("tradeoff", "choice", 6), ("tradeoff", "noul", 2), ("probability", "choice", 3),
             ("probability", "score", 6), ("ambiguous", "choice", 3)},
        )
        by_id = {r["id"]: r["metadata"] for r in rows}
        trade = by_id["kev-hard-v1/tradeoff/train/00000/choice"]
        self.assertEqual(trade["options"][1:], ["beta", "gamma", "delta", "epsilon",
                                                "none qualifies: No option qualifies"])
        self.assertEqual(by_id["kev-hard-v1/probability/train/00000/value"]["options"],
                         ["10%", "20%", "30%"])
        self.assertEqual(by_id["kev-hard-v1/probability/train/00000/bucket"]["options"], LEVELS)
        self.assertEqual(sorted(by_id["kev-hard-v1/tradeoff/train/00000/meets"]["options"]),
                         ["No: Misses one", "Yes: Meets all"])
        # long_policy is counted, not silently dropped; test is declared but never read.
        self.assertEqual(
            report["excluded_rows"]["long_policy_excluded_round1_capture_cost"],
            {"development": {"long_policy/questions": 3, "long_policy/records": 3},
             "train": {"long_policy/questions": 12, "long_policy/records": 12}},
        )
        self.assertFalse(report["kev"]["files"]["test.jsonl"]["read"])
        self.assertEqual(report["counts"]["calibration"]["ambiguous"]["groups"], 3)
        self.assertEqual(
            report["counts"]["train"]["probability"]["question_types"]["score"]["option_counts"],
            {"6": 9},
        )
        payload = (self.tmp / "out" / "rows.jsonl").read_bytes()
        self.assertEqual(report["output"]["sha256"], hashlib.sha256(payload).hexdigest())
        self.run_hard(out="again")
        self.assertEqual((self.tmp / "again" / "rows.jsonl").read_bytes(), payload)

    def test_options_long_policy_and_descriptions_are_explicit(self):
        report = self.run_hard(include_families=("long_policy",), option_descriptions=False)
        rows = {r["id"]: r["metadata"] for r in self.rows()}
        self.assertEqual(report["excluded_rows"], {})
        self.assertEqual(rows["kev-hard-v1/long_policy/train/00000/outcome"]["options"],
                         ["$a1", "$b1", "$c1", "$d1", "$e1"])
        self.assertEqual(rows["kev-hard-v1/ambiguous/train/00000/decision"]["options"],
                         ["not enough information", "valid", "invalid"])
        self.assertEqual(rows["kev-hard-v1/tradeoff/train/00000/choice"]["options"][-1],
                         "none qualifies")
        self.assertEqual(report["renderer"], jp.HARD_RENDERER_NO_DESCRIPTIONS)
        with self.assertRaisesRegex(jp.PrepareError, "can be included"):
            jp.prepare_hard_v1(self.kev, self.tmp / "x", include_families=("judge",))

    def test_malformed_records_fail(self):
        def broken(change, partition="train", index=0):
            parts = hard_partitions()
            change(parts[partition][index])
            return parts

        cases = {
            "not a train template": lambda r: r["_meta"].update(template="tradeoff/t4"),
            "text_sha256 does not match": lambda r: r.update(state="other state"),
            "is not an option": lambda r: r["questions"]["choice"].update(label="zeta"),
            "choice needs 3-6": lambda r: r["questions"]["choice"].update(
                criteria={"x": None, "alpha_one": None}, label="x"),
            "question keys": lambda r: r["questions"]["meets"].update(target={"true": 1}),
            "noul label must be a boolean": lambda r: r["questions"]["meets"].update(label=1),
            "unsupported question type": lambda r: r["questions"]["meets"].update(type="rank"),
            "unexpected group_id": lambda r: r["_meta"].update(group_id="x"),
            "_meta must include": lambda r: r["_meta"].pop("facts"),
        }
        for needle, change in cases.items():
            with self.subTest(needle):
                self.assert_fails(needle, broken(change))
        with self.subTest("score levels"):
            self.assert_fails("score needs 6 levels", broken(
                lambda r: r["questions"]["bucket"].update(criteria=LEVELS[:5], label=0),
                index=12))

    def test_pins_and_manifest_provenance_are_enforced(self):
        digest = write_hard(self.kev, hard_partitions())
        with self.assertRaisesRegex(jp.PrepareError, "does not match pinned"):
            jp.prepare_hard_v1(self.kev, self.tmp / "bad")
        with open(self.kev / "train.jsonl", "ab") as handle:
            handle.write(b"\n")
        with self.assertRaisesRegex(jp.PrepareError, "train.jsonl: sha256"):
            jp.prepare_hard_v1(self.kev, self.tmp / "bad", manifest_sha256=digest)

        def tokenizer(m):
            m["tokenizer"]["revision"] = "main"

        def unlocked(m):
            m["locked"] = []

        def counts(m):
            m["files"]["train.jsonl"]["by_family"]["tradeoff"]["questions"] += 1

        self.assert_fails("manifest tokenizer", mutate=tokenizer)
        self.assert_fails("does not lock", mutate=unlocked)
        self.assert_fails("per-family counts", mutate=counts)
        self.assertFalse((self.tmp / "bad").exists())

    def test_overlap_and_coverage_failures(self):
        parts = hard_partitions()
        dev = parts["development"][0]
        train = parts["train"][0]
        dev["state"] = train["state"].upper() + "  "
        dev["_meta"]["text_sha256"] = train["_meta"]["text_sha256"]
        self.assert_fails("overlap", parts)

        parts = hard_partitions()
        calib = next(r for r in parts["train"] if r["_meta"]["template"] == "ambiguous/t3")
        twin = next(r for r in parts["train"] if r["_meta"]["template"] == "ambiguous/t0")
        calib["_meta"]["group_id"] = twin["_meta"]["group_id"]
        self.assert_fails("group overlap", parts)

        parts = hard_partitions()
        for r in parts["development"]:
            if r["_meta"]["family"] == "probability":
                r["questions"]["bucket"]["label"] = 2
        self.assert_fails("validation/probability/score: one class only", parts)

        parts = hard_partitions()
        for r in parts["train"]:
            if r["_meta"]["template"] == "probability/t3":
                del r["questions"]["bucket"]
        self.assert_fails("calibration/probability: no score questions", parts)
        self.assert_fails("groups, need 4", min_groups=4)

    def test_cli_selects_the_dataset_and_rejects_mixed_options(self):
        digest = write_hard(self.kev, hard_partitions())
        stdout, stderr = io.StringIO(), io.StringIO()
        base = ["--kev-dir", str(self.kev), "--manifest-sha256", digest]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            ok = jp.main(base + ["--dataset", "hard-v1", "--out", str(self.tmp / "cli")])
            mixed = jp.main(base + ["--dataset", "hard-v1", "--calibration-fraction", "0.2",
                                    "--out", str(self.tmp / "mixed")])
            devtools = jp.main(base + ["--include-long-policy", "--out", str(self.tmp / "dt")])
        self.assertEqual((ok, mixed, devtools), (0, 2, 2))
        self.assertIn("template 3", stderr.getvalue())
        self.assertIn("hard-v1 options", stderr.getvalue())
        self.assertTrue((self.tmp / "cli" / "rows.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
