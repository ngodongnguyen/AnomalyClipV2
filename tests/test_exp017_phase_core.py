import unittest

import numpy as np

from scripts.exp017_phase_core import inverse_align, phase_statistics, pilot_gate


class PhaseCoreTests(unittest.TestCase):
    def test_inverse_alignment_of_integer_translation(self):
        original = np.arange(40 * 40, dtype=float).reshape(40, 40)
        for dx, dy in ((7, 0), (-7, 0), (0, 7), (0, -7),
                       (14, 0), (-14, 0), (0, 14), (0, -14)):
            shifted = np.full_like(original, -1)
            x0, x1 = max(0, dx), min(40, 40 + dx)
            y0, y1 = max(0, dy), min(40, 40 + dy)
            shifted[y0:y1, x0:x1] = original[y0-dy:y1-dy, x0-dx:x1-dx]
            aligned = inverse_align(shifted, dx, dy)
            np.testing.assert_array_equal(aligned[14:-14, 14:-14], original[14:-14, 14:-14])

    def test_identical_views_have_zero_rank_disagreement(self):
        rng = np.random.default_rng(111)
        base = rng.random((70, 70))
        mask = np.zeros((70, 70), dtype=bool)
        mask[25:40, 25:40] = True
        maps = {"identity": base, "smooth8": base, "smooth16": base, "smooth32": base}
        for group in ("half", "full"):
            for axis in ("x", "y"):
                for sign in ("pos", "neg"):
                    maps[f"{group}_{axis}{sign}"] = base.copy()
        result = phase_statistics(mask, maps)
        self.assertEqual(result["half"], 0)
        self.assertEqual(result["full"], 0)
        self.assertAlmostEqual(result["auc_ensemble_half"], result["auc_identity"])

    def test_pilot_gate_rejects_no_phase_advantage(self):
        row = {"half": .1, "full": .1, "auc_ensemble_half": .9,
               "auc_ensemble_full": .89, "auc_smooth32": .89}
        rows = {name: [row] * 10 for name in ("ClinicDB", "ColonDB", "Kvasir")}
        self.assertFalse(pilot_gate(rows, repetitions=100)["pass"])


if __name__ == "__main__":
    unittest.main()
