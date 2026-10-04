import unittest

import numpy as np

from scripts.exp014_readout_core import auc, gate, hotspot, source_partition


class ReadoutCoreTests(unittest.TestCase):
    def test_auc_known_answer_and_ties(self):
        self.assertEqual(auc([0, 0, 1, 1], [0, 1, 2, 3]), 1.0)
        self.assertEqual(auc([0, 1], [1, 1]), 0.5)
        self.assertIsNone(auc([1, 1], [0, 1]))

    def test_hotspot_selection_fixed_by_baseline(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:25, :25] = True
        baseline = np.zeros((100, 100))
        baseline[0:5, 0:5] = 2
        baseline[70:95, 70:95] = 2
        candidate = np.zeros_like(baseline)
        candidate[mask] = 3
        tp, fp, scores = hotspot(mask, baseline, {"candidate": candidate})
        self.assertGreater(tp, 0)
        self.assertGreater(fp, 0)
        self.assertEqual(scores["candidate"], 1.0)

    def test_partition_stable(self):
        self.assertEqual(source_partition("abc"), source_partition("abc"))
        self.assertIn(source_partition("abc"), {"direction", "fit", "validation"})

    def test_gate(self):
        rows = {str(i): [{"hotspot_vv_only": 0.5,
                          "hotspot_two_branch": 0.54,
                          "hotspot_fixed_mixture": 0.51}]
                for i in range(6)}
        self.assertTrue(gate(rows)["pass"])
        rows["0"][0]["hotspot_two_branch"] = 0.47
        self.assertFalse(gate(rows)["pass"])


if __name__ == "__main__":
    unittest.main()
