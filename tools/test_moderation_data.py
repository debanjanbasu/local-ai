import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import moderation_data as md


def civil(**values):
    row = dict.fromkeys(
        (
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
            "sexual_explicit",
        ),
        0.0,
    )
    row.update(values)
    return row


class MappingTest(unittest.TestCase):
    def test_aegis_categories_map_explicitly(self):
        labels = md.aegis_labels({"Hate/Identity Hate", "Threat"})
        self.assertEqual(labels["hate"], 1)
        self.assertEqual(labels["hate/threatening"], 1)
        self.assertEqual(labels["harassment/threatening"], 1)
        self.assertEqual(labels["violence"], 1)
        self.assertEqual(labels["illicit"], 0)
        for name in ("self-harm/intent", "self-harm/instructions", "violence/graphic"):
            self.assertIsNone(labels[name])
        weapons = md.aegis_labels({"Guns and Illegal Weapons"})
        self.assertEqual((weapons["illicit"], weapons["illicit/violent"]), (1, 1))
        minors = md.aegis_labels({"Sexual (minor)"})
        self.assertEqual((minors["sexual"], minors["sexual/minors"]), (1, 1))

    def test_aegis_rows_use_the_unambiguous_text(self):
        rows = [
            {
                "id": "a",
                "prompt": "p1",
                "response": None,
                "prompt_label": "unsafe",
                "response_label": None,
                "violated_categories": "Violence",
            },
            {
                "id": "b",
                "prompt": "p2",
                "response": "r2",
                "prompt_label": "unsafe",
                "response_label": "safe",
                "violated_categories": "Sexual",
            },
            {
                "id": "c",
                "prompt": "p3",
                "response": "r3",
                "prompt_label": "unsafe",
                "response_label": "unsafe",
                "violated_categories": "Harassment",
            },
            {
                "id": "d",
                "prompt": "p4",
                "response": None,
                "prompt_label": "unsafe",
                "response_label": None,
                "violated_categories": "Needs Caution",
            },
            {
                "id": "e",
                "prompt": "p5",
                "response": None,
                "prompt_label": "safe",
                "response_label": None,
                "violated_categories": "",
            },
        ]
        out = {
            rid: (text, labels)
            for text, labels, _, rid in md.aegis_rows(rows, lambda r: "train")
        }
        self.assertEqual(out["a"][1]["violence"], 1)
        self.assertEqual(out["b"][0], "p2")
        self.assertEqual(out["b"][1]["sexual"], 1)
        self.assertEqual(out["b:response"][1]["sexual"], 0)
        self.assertEqual(out["c"][0], "p3\n\nr3")
        self.assertNotIn("d", out)
        self.assertEqual(out["e"][1]["hate"], 0)

    def test_civil_fractions_have_a_gap(self):
        labels = md.civil_labels(civil(insult=0.7, identity_attack=0.3, threat=0.05))
        self.assertEqual(labels["harassment"], 1)
        self.assertIsNone(labels["hate"])
        self.assertEqual(labels["harassment/threatening"], 0)
        self.assertEqual(labels["hate/threatening"], 0)
        self.assertIsNone(labels["violence"])
        self.assertEqual(labels["self-harm"], 0)
        self.assertIsNone(md.civil_labels(civil(sexual_explicit=0.2))["sexual/minors"])
        both = md.civil_labels(civil(identity_attack=0.6, threat=0.6))
        self.assertEqual(both["hate/threatening"], 1)

    def test_openai_keys_and_missing_values(self):
        labels = md.openai_labels({"H": 1, "S3": 0, "V2": None})
        self.assertEqual((labels["hate"], labels["sexual/minors"]), (1, 0))
        self.assertIsNone(labels["violence/graphic"])
        self.assertIsNone(labels["illicit"])


class BuildTest(unittest.TestCase):
    def test_inputs_are_verified_and_evaluation_texts_never_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)

            def put(name, rows):
                with gzip.open(raw / name, "wt", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

            aegis = {
                "id": "x",
                "prompt": "shared text",
                "response": None,
                "prompt_label": "safe",
                "response_label": None,
                "violated_categories": "",
            }
            put("aegis2_train.jsonl.gz", [aegis])
            put("aegis2_validation.jsonl.gz", [])
            put("aegis2_test.jsonl.gz", [])
            put(
                "civil_comments_sample.jsonl.gz",
                [dict(civil(), text="civil text", row=3)],
            )
            put("openai_eval.jsonl.gz", [{"prompt": "shared text", "H": 0}])
            put(
                "toxicchat_toxic-chat_annotation_test.jsonl.gz",
                [{"conv_id": "c", "user_input": "chat", "toxicity": 1}],
            )
            outputs = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in raw.glob("*.gz")
            }
            (raw / "manifest.json").write_text(
                json.dumps({"sources": {}, "outputs": outputs})
            )
            rows, meta = md.build(raw)
            splits = {(r["text"], r["split"]) for r in rows}
            self.assertIn(("shared text", "ood_openai"), splits)
            self.assertNotIn(("shared text", "train"), splits)
            self.assertIn(("chat", "ood_toxicchat"), splits)
            self.assertEqual(meta["dropped_duplicates_or_leaks"], 1)
            (raw / "openai_eval.jsonl.gz").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                md.build(raw)


if __name__ == "__main__":
    unittest.main()
