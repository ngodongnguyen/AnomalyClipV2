"""Synthetic checks for EXP-043's staged gate and evaluation provenance."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_exp024 import IMG, PIX
from collect_exp043 import EXPECTED_BASELINE_SHA256, audit_pairs, collect, decide, summarize


def fixture(image_delta=0.8, pixel_delta=0.0):
    pairs = {}
    for kind, sets in (('img', IMG), ('pix', PIX)):
        for name, key in sets:
            arms = {}
            for arm in ('ecp_baseline', 'ecp_split'):
                suffix = ('EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth'
                          if arm == 'ecp_baseline' else 'exp043_ecp_split_s111/epoch_15.pth')
                meta = {
                    'seed': 111, 'sigma': 4, 'split': 'test',
                    'checkpoint_extent_cond': 'extent', 'checkpoint_ec_static_aux': False,
                    'z_img': -0.69, 'z_pix': 1.6,
                    'checkpoint_path': f'/repo/checkpoints/{suffix}',
                    'checkpoint_sha256': (EXPECTED_BASELINE_SHA256 if arm == 'ecp_baseline' else arm),
                    'arguments': {'metrics': 'image-level' if kind == 'img' else 'pixel-level'},
                    'data_path': f'/data/{key}', 'meta_json_sha256': f'meta-{key}',
                    'sample_count': 100, 'positive_count': 50, 'negative_count': 50,
                    'evaluation_source_sha256': {'test.py': 'same'},
                    'pretrained_clip_weights_sha256': 'clip',
                }
                gain = image_delta if kind == 'img' else pixel_delta
                metrics = [90 + (gain if arm == 'ecp_split' else 0),
                           85 + (gain if arm == 'ecp_split' else 0)]
                arms[arm] = {'metadata': meta, 'metrics': metrics}
            pairs[name] = {'kind': kind, 'pair': arms}
    return pairs


class Exp043Test(unittest.TestCase):
    def test_stage_gates(self):
        metrics = summarize(fixture())
        self.assertEqual(decide(metrics, pixel_complete=False), 'CONTINUE_PIXEL')
        self.assertEqual(decide(metrics, pixel_complete=True), 'PILOT_PASS_NEEDS_MATCHED_SHAM')
        self.assertEqual(decide(summarize(fixture(image_delta=0.3)), False), 'STOP_IMAGE')
        self.assertEqual(decide(summarize(fixture(pixel_delta=-1.1)), True), 'STOP_PIXEL')

    def test_provenance_guard(self):
        pairs = fixture()
        audit_pairs(pairs)
        pairs['HeadCT']['pair']['ecp_split']['metadata']['z_img'] = 1.6
        with self.assertRaisesRegex(ValueError, 'wrong ECP evaluation settings'):
            audit_pairs(pairs)

    def test_image_only_artifacts_trigger_pixel_stage(self):
        pairs = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, key in IMG:
                for arm, record in pairs[name]['pair'].items():
                    folder = root / 'results/EXP-043' / arm / key
                    folder.mkdir(parents=True)
                    (folder / 'image_level_metrics.json').write_text(json.dumps({
                        'mean': {'image_auroc': record['metrics'][0] / 100,
                                 'image_ap': record['metrics'][1] / 100}}))
                    (folder / 'evaluation_metadata.json').write_text(json.dumps(record['metadata']))
            report = collect(root)
            self.assertEqual(report['status'], 'CONTINUE_PIXEL')
            self.assertEqual(len(report['missing_pixel']), 6)


if __name__ == '__main__':
    unittest.main()
