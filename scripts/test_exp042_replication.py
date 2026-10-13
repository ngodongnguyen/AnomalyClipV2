"""Known-answer checks for the EXP-042 three-seed decision and provenance guard."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_exp024 import IMG, PIX
from collect_exp042_replication import SEEDS, audit, summarize


def fixture(image_gain=2.0, pixel_loss=0.0):
    records = {}
    for kind, datasets in (('pix', PIX), ('img', IMG)):
        for name, key in datasets:
            seeds = {}
            for seed in SEEDS:
                arms = {}
                for arm in ('split', 'sham'):
                    meta = {
                        'data_path': f'/data/{key}', 'meta_json_sha256': f'meta-{key}',
                        'sample_count': 10, 'positive_count': 5, 'negative_count': 5,
                        'seed': seed, 'sigma': 4, 'split': 'test',
                        'checkpoint_extent_cond': None, 'checkpoint_ec_static_aux': False,
                        'z_img': None, 'z_pix': None,
                        'checkpoint_path': f'/repo/checkpoints/exp042_{arm}_s{seed}/epoch_15.pth',
                        'checkpoint_sha256': f'ckpt-{arm}-{seed}',
                        'evaluation_source_sha256': {'test.py': 'same'},
                        'pretrained_clip_weights_sha256': 'clip-hash',
                        'arguments': {'metrics': 'pixel-level' if kind == 'pix' else 'image-level'},
                    }
                    metric = [82.0, 70.0]
                    if arm == 'split':
                        metric = [82.0 + (image_gain if kind == 'img' else 0.2),
                                  70.0 + (0.5 if kind == 'img' else pixel_loss)]
                    arms[arm] = {'metrics': metric, 'metadata': meta}
                seeds[str(seed)] = arms
            records[name] = {'kind': kind, 'key': key, 'seeds': seeds}
    return records


class ReplicationTest(unittest.TestCase):
    def test_registered_gate_pass_and_pixel_failure(self):
        good = fixture()
        self.assertEqual(summarize(good)[1], 'REPLICATION_GATE_PASS')
        bad = fixture(pixel_loss=-1.1)
        self.assertEqual(summarize(bad)[1], 'REPLICATION_GATE_FAIL')

    def test_provenance_mismatch_rejected(self):
        records = fixture()
        audit(records)
        records['BrainMRI']['seeds']['333']['sham']['metadata']['meta_json_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'mismatched meta_json_sha256'):
            audit(records)


if __name__ == '__main__':
    unittest.main()
