import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from collect_exp040 import collect
from exp040_core import CATEGORIES, sha256_file, stable_normals, summarize, transfer_box


def synthetic_groups(crop_shift, sham_shift):
    groups = {}
    for category_index, category in enumerate(CATEGORIES):
        records = []
        for image_index in range(32):
            original = image_index / 10 + category_index / 20
            records.append({
                'original': {'ctrl': original, 'zo': original + 0.2},
                'crop': {'ctrl': original, 'zo': original + 0.2 + crop_shift},
                'sham': {'ctrl': original, 'zo': original + 0.2 + sham_shift},
            })
        groups[category] = records
    return groups


class Exp040CoreTests(unittest.TestCase):
    def test_planted_crop_specific_response_passes(self):
        result = summarize(synthetic_groups(0.8, 0.1), n_boot=100)
        self.assertEqual(result['verdict'], 'SUPPORT_FOR_FOLLOWUP')
        self.assertEqual(result['positive_crop_categories'], 15)
        self.assertGreater(result['bootstrap_95_ci']['specific'][0], 0)

    def test_sham_explains_response(self):
        result = summarize(synthetic_groups(0.4, 0.5), n_boot=100)
        self.assertEqual(result['verdict'], 'NOT_SUPPORTED')

    def test_box_transfer_and_selection_order(self):
        box, ratio = transfer_box([0.25, 0.2, 0.75, 0.8], (200, 100))
        self.assertEqual(box, (50, 20, 150, 80))
        self.assertAlmostEqual(ratio, (0.5 * 0.6) ** 0.5)
        rows = [{'img_path': f'bottle/train/good/{index:02d}.png'} for index in range(40)]
        self.assertEqual(stable_normals(rows, 32), stable_normals(list(reversed(rows)), 32))

    def test_collector_pairs_full_manifest_and_rejects_missing_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = []
            scores = []
            for category_index, category in enumerate(CATEGORIES):
                for index in range(32):
                    sample_id = f'{category}/train/good/{index:02d}.png'
                    views.append({'category': category, 'sample_id': sample_id,
                                  'sham_side_ratio': 0.5})
                    original = index / 10 + category_index / 20
                    for view, shift in (('original', 0), ('crop', 0.8), ('sham', 0.1)):
                        scores.append({'category': category, 'sample_id': sample_id,
                                       'view': view, 'sham_side_ratio': 0.5,
                                       'ctrl_margin': original,
                                       'zo_margin': original + 0.2 + shift})
            with (root / 'views.jsonl').open('w') as handle:
                for row in views:
                    handle.write(json.dumps(row) + '\n')
            geometry = {
                'experiment': 'EXP-040', 'training_source_split': 'test',
                'probe_normal_split': 'train', 'selected_normal_count': 480,
                'views_sha256': sha256_file(root / 'views.jsonl'),
                'category_counts': {category: {'attempts': 10, 'successful': 3}
                                    for category in CATEGORIES},
                'successful_geometries': {
                    category: [{'side_fraction': 0.5}] * 3 for category in CATEGORIES},
            }
            (root / 'geometry.json').write_text(json.dumps(geometry))
            score_path = root / 'scores.csv'
            with score_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
                writer.writeheader()
                writer.writerows(scores)
            provenance = {
                'experiment': 'EXP-040', 'smoke': False,
                'image_count': 480, 'score_count': 1440,
                'settings': {'image_size': 518, 'features_list': [24],
                             'dpam_layer': 20, 'depth': 9, 'n_ctx': 12,
                             't_n_ctx': 4, 'seed': 111},
                'checkpoint_paths': {
                    'ctrl': '/repo/checkpoints/matched_ctrl_s111/epoch_15.pth',
                    'zo': '/repo/checkpoints/EXP-012-matched/zoom_only/checkpoints/epoch_15.pth'},
                'checkpoint_sha256': {'ctrl': 'a', 'zo': 'b'},
                'geometry_sha256': sha256_file(root / 'geometry.json'),
                'views_sha256': sha256_file(root / 'views.jsonl'),
                'score_sha256': sha256_file(score_path),
            }
            provenance_path = root / 'provenance.json'
            provenance_path.write_text(json.dumps(provenance))
            with patch('collect_exp040.summarize', side_effect=lambda groups: summarize(groups, n_boot=30)):
                self.assertEqual(collect(root)['verdict'], 'SUPPORT_FOR_FOLLOWUP')
                with score_path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
                    writer.writeheader()
                    writer.writerows(scores[:-1])
                provenance['score_sha256'] = sha256_file(score_path)
                provenance_path.write_text(json.dumps(provenance))
                with self.assertRaisesRegex(ValueError, 'unpaired or missing view'):
                    collect(root)


if __name__ == '__main__':
    unittest.main()
