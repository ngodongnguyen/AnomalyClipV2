"""Synthetic EXP-015 invariants; no dataset/checkpoint needed."""

import unittest
import numpy as np

from scripts.exp015_probe_core import (GRID, center_label, intervention,
                                       paired_auc, select_centers)
from scripts.exp015_context_probe import analyze_dataset


class SelectionTest(unittest.TestCase):
    def test_centers_are_deterministic_separated_and_interior(self):
        margins = np.random.default_rng(111).normal(size=(GRID, GRID))
        first = select_centers(margins)
        self.assertEqual(first, select_centers(margins.copy()))
        self.assertEqual(len(first), 6)
        self.assertEqual(sum(kind == "high" for _, _, kind in first), 3)
        self.assertEqual(sum(kind == "middle" for _, _, kind in first), 3)
        for i, (r, c, _) in enumerate(first):
            self.assertTrue(4 <= r < GRID - 4 and 4 <= c < GRID - 4)
            for rr, cc, _ in first[i + 1:]:
                self.assertGreaterEqual(max(abs(r - rr), abs(c - cc)), 7)

    def test_labels_and_auc_known_answers(self):
        mask = np.zeros((518, 518), dtype=bool)
        mask[14 * 10:14 * 11, 14 * 10:14 * 11] = True
        distance = np.full_like(mask, 100, dtype=float)
        self.assertEqual(center_label(mask, 10, 10, distance)[0], "TP")
        self.assertEqual(center_label(mask, 20, 20, distance)[0], "far_FP")
        distance[20 * 14:21 * 14, 20 * 14:21 * 14] = 10
        self.assertEqual(center_label(mask, 20, 20, distance)[0], "other")
        self.assertEqual(paired_auc([2, 3], [1]), 1.0)
        self.assertEqual(paired_auc([1], [1]), .5)
        self.assertIsNone(paired_auc([], [1]))


class InterventionTest(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
    def test_protected_pixels_sham_and_two_active_families(self):
        import torch
        image = torch.linspace(-.8, .8, 518 * 518).reshape(1, 1, 518, 518).repeat(1, 3, 1, 1)
        r, c = 18, 18
        self.assertTrue(torch.equal(intervention(image, r, c, "sham", "example"), image))
        for kind in ("photo", "rearrange", "near_photo"):
            edited = intervention(image, r, c, kind, "example")
            replay = intervention(image, r, c, kind, "example")
            self.assertTrue(torch.equal(edited, replay))
            self.assertTrue(torch.equal(edited[..., (r - 3) * 14:(r + 4) * 14,
                                                (c - 3) * 14:(c + 4) * 14],
                                        image[..., (r - 3) * 14:(r + 4) * 14,
                                                  (c - 3) * 14:(c + 4) * 14]))
            self.assertGreater(float((edited - image).abs().sum()), 0)
            self.assertTrue(bool(torch.isfinite(edited).all()))
            coords = torch.nonzero((edited - image).abs().amax(dim=1)[0] > 0).float()
            center = coords.new_tensor([r * 14 + 6.5, c * 14 + 6.5])
            self.assertGreater(float(torch.linalg.vector_norm(coords-center, dim=1).min()), 42)


class DiagnosticAnalysisTest(unittest.TestCase):
    def test_known_answer_within_image_auc_and_bootstrap(self):
        selected, effects = [], []
        for image in range(30):
            sample_id = str(image)
            labels = ["far_FP", "far_FP", "far_FP", "TP", "TP", "TP"]
            selected.append({"sample_id": sample_id,
                             "centers": [{"label": label} for label in labels]})
            for family in ("photo", "rearrange"):
                for label in labels:
                    effects.append({"sample_id": sample_id, "family": family,
                                    "label": label, "sensitivity": 2.0 if label == "far_FP" else 1.0,
                                    "baseline_margin": 0.0,
                                    "minimum_query_to_edit_distance_px": 49.5,
                                    "sham_abs_margin_drift": 0.0,
                                    "signed_margin_drift": 2.0 if label == "far_FP" else 1.0})
        result = analyze_dataset(effects, selected)
        self.assertEqual(result["eligible_images"], 30)
        for family in ("photo", "rearrange"):
            self.assertEqual(result[family]["within_image_auc_mean"], 1.0)
            self.assertEqual(result[family]["margin_and_seam_adjusted_auc_mean"], 1.0)
            self.assertEqual(result[family]["ci95"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
