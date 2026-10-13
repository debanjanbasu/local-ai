import math
import random
import unittest

import moderation_train as mt


class MetricsTest(unittest.TestCase):
    def test_auc_matches_pair_counting_with_ties(self):
        rng = random.Random(1)
        scores = [rng.choice([0.1, 0.2, 0.3, 0.4]) for _ in range(60)]
        labels = [int(rng.random() < 0.4) for _ in range(60)]
        pairs = [
            (p, n)
            for p, y in zip(scores, labels)
            if y
            for n, z in zip(scores, labels)
            if not z
        ]
        expected = sum(
            1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs
        ) / len(pairs)
        self.assertAlmostEqual(mt.roc_auc(scores, labels), expected)
        self.assertIsNone(mt.roc_auc([0.1, 0.2], [1, 1]))

    def test_ece_is_zero_when_bins_match_their_frequency(self):
        probs = [0.25] * 4 + [0.75] * 4
        labels = [1, 0, 0, 0, 1, 1, 1, 0]
        self.assertAlmostEqual(mt.ece(probs, labels, 10), 0.0)
        self.assertAlmostEqual(mt.ece([0.9, 0.9], [0, 0], 10), 0.9)

    def test_threshold_maximizes_recall_at_the_precision_target(self):
        probs = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        labels = [1, 1, 0, 1, 1, 0, 1]
        # precision at 0.6 is 4/5 = 0.8, recall 4/5; at 0.4: 5/7 < 0.8.
        self.assertEqual(mt.choose_threshold(probs, labels, 0.8, 0.3), 0.6)
        self.assertIsNone(mt.choose_threshold(probs, labels, 0.99, 0.5))
        self.assertIsNone(mt.choose_threshold([0.5], [0], 0.5, 0.1))
        p, r = mt.precision_recall(probs, labels, 0.6)
        self.assertEqual((p, r), (0.8, 0.8))

    def test_platt_recovers_a_known_slope(self):
        rng = random.Random(7)
        logits = [rng.uniform(-6, 6) for _ in range(4000)]
        labels = [int(rng.random() < mt.sigmoid(0.5 * z - 1.0)) for z in logits]
        a, c = mt.fit_platt(logits, labels)
        self.assertAlmostEqual(a, 0.5, delta=0.08)
        self.assertAlmostEqual(c, -1.0, delta=0.2)
        with self.assertRaises(ValueError):
            mt.fit_platt([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 0])

    def test_wilson_interval_contains_the_estimate(self):
        lo, hi = mt.wilson(8, 10)
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)
        self.assertEqual(mt.wilson(0, 0), (0.0, 1.0))

    def test_judge_applies_every_preregistered_criterion(self):
        good = {
            "positives": 100,
            "auc": 0.95,
            "precision": 0.8,
            "recall": 0.6,
            "ece": 0.01,
        }
        self.assertEqual(mt.judge({"test": good}), [])
        self.assertEqual(len(mt.judge({"test": dict(good, positives=10)})), 1)
        bad = dict(good, auc=0.8, precision=0.5, recall=0.4, ece=0.2)
        self.assertEqual(len(mt.judge({"test": bad})), 4)
        ood = {"positives": 50, "auc": 0.7}
        self.assertEqual(len(mt.judge({"test": good, "ood_openai": ood})), 1)
        self.assertEqual(
            mt.judge({"test": good, "ood_openai": dict(ood, positives=5)}), []
        )

    def test_sigmoid_is_stable(self):
        self.assertEqual(mt.sigmoid(-1000.0), 0.0)
        self.assertEqual(mt.sigmoid(1000.0), 1.0)
        self.assertAlmostEqual(mt.sigmoid(math.log(3)), 0.75)


if __name__ == "__main__":
    unittest.main()
