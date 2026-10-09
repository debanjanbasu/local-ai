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


if __name__ == "__main__":
    unittest.main()
