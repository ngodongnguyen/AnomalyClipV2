import unittest

from collect_exp039 import auc_ap


class MetricTests(unittest.TestCase):
    def test_perfect_and_reversed(self):
        self.assertEqual(auc_ap([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]),
                         {'auroc': 1.0, 'ap': 1.0})
        reverse = auc_ap([0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2])
        self.assertEqual(reverse['auroc'], 0.0)
        self.assertAlmostEqual(reverse['ap'], (1 / 3 + 2 / 4) / 2)

    def test_tied_scores(self):
        tied = auc_ap([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(tied, {'auroc': 0.5, 'ap': 0.5})

    def test_requires_two_classes(self):
        with self.assertRaises(ValueError):
            auc_ap([0, 0], [0.1, 0.2])


if __name__ == '__main__':
    unittest.main()
