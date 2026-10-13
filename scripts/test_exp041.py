"""Small deterministic checks for EXP-041 pairing and severity summaries."""

import unittest
import csv
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from exp040_core import CATEGORIES, sha256_file
from exp041 import build_rows, collect_labels, export_gallery, read_inputs, summarize


class Exp041Tests(unittest.TestCase):
    def test_fixed_category_quartiles_and_sham_subtraction(self):
        selected, scores = {}, {}
        for category in CATEGORIES:
            for index in range(32):
                key = (category, f'{category}/train/good/{index:03d}.png')
                ratio = (index + 1) / 33
                selected[key] = {'sham_side_ratio': ratio}
                q = index // 8 + 1
                scores[key] = {
                    'original': {'ctrl': 0.0, 'zo': 0.0},
                    'crop': {'ctrl': 0.0, 'zo': q + 0.25},
                    'sham': {'ctrl': 0.0, 'zo': 0.25},
                }
        rows = build_rows(selected, scores, {'ctrl': 1.0, 'zo': 1.0})
        report = summarize(rows, n_boot=20, seed=41)
        self.assertEqual(len(rows), 480)
        for q in range(1, 5):
            part = report[str(q)]
            self.assertEqual(part['image_count'], 120)
            self.assertAlmostEqual(part['macro_mean']['d_specific'], q)
            self.assertEqual(part['positive_categories']['d_specific'], 15)
            self.assertAlmostEqual(part['bootstrap_95_ci']['d_specific'][0], q)
            self.assertAlmostEqual(part['bootstrap_95_ci']['d_specific'][1], q)

    def test_copied_exp040_inputs_if_available(self):
        path = Path('results/EXP-040')
        if not (path / 'provenance.json').is_file():
            self.skipTest('copied EXP-040 artifacts unavailable')
        _, selected, scores, scales = read_inputs(path)
        rows = build_rows(selected, scores, scales)
        crop = sum(row['d_crop'] for row in rows) / 480
        specific = sum(row['d_specific'] for row in rows) / 480
        self.assertAlmostEqual(crop, 1.1573802941136966, places=10)
        self.assertAlmostEqual(specific, 0.7548280148253966, places=10)

    def test_gallery_export_and_complete_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            out = root / 'out'
            data.mkdir()
            out.mkdir()
            original = data / 'original.png'
            Image.new('RGB', (32, 32), color='red').save(original)
            digest = sha256_file(original)
            selected = {}
            effect_rows = []
            for category in CATEGORIES:
                folder = data / category / 'train' / 'good'
                folder.mkdir(parents=True)
                for index in range(32):
                    rel = f'{category}/train/good/{index:03d}.png'
                    os.link(original, data / rel)
                    selected[(category, rel)] = {
                        'category': category, 'sample_id': rel,
                        'image_size': [32, 32], 'image_sha256': digest,
                        'pixel_box': [4, 4, 28, 28], 'sham_side_ratio': 0.75,
                    }
                    effect_rows.append({'category': category, 'sample_id': rel,
                                        'quartile': str(index // 8 + 1), 'd_specific': '1.0'})
            with (out / 'per_image.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(effect_rows[0]))
                writer.writeheader()
                writer.writerows(effect_rows)
            (out / 'analysis.json').write_text(json.dumps({
                'source_hashes': {'views_sha256': 'fixture_views'},
                'per_image_sha256': sha256_file(out / 'per_image.csv'),
            }))
            with patch('exp041.read_inputs', return_value=(
                    {'views_sha256': 'fixture_views'}, selected, {}, {})):
                export = export_gallery(root, out, data)
            self.assertEqual(export['image_count'], 480)
            self.assertEqual(len(list((out / 'gallery').glob('*_sheet_*.jpg'))), 30)
            labels_path = out / 'retention_labels.csv'
            with labels_path.open(newline='') as handle:
                labels = list(csv.DictReader(handle))
            for row in labels:
                row['recognizable'] = 'yes'
            with labels_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
                writer.writeheader()
                writer.writerows(labels)
            report = collect_labels(out)
            self.assertTrue(report['unfiltered_geometry_approved'])
            self.assertEqual(report['label_counts']['object']['yes'], 320)
            self.assertEqual(report['label_counts']['texture']['yes'], 160)


if __name__ == '__main__':
    unittest.main()
