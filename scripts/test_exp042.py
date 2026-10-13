"""Small checks for EXP-042 view routing and the registered pilot gate."""

import importlib
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image


class Exp042Test(unittest.TestCase):
    def test_dataset_original_is_before_zoom_without_extra_rng(self):
        # dataset.py only needs torch's Dataset base for this synthetic image fixture.
        torch = types.ModuleType('torch')
        utils = types.ModuleType('torch.utils')
        data = types.ModuleType('torch.utils.data')
        data.Dataset = object
        utils.data = data
        torch.utils = utils
        old = {key: sys.modules.get(key) for key in ('torch', 'torch.utils', 'torch.utils.data', 'dataset')}
        sys.modules.update({'torch': torch, 'torch.utils': utils, 'torch.utils.data': data})
        sys.modules.pop('dataset', None)
        try:
            dataset_module = importlib.import_module('dataset')
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'tile').mkdir()
                image = Image.new('RGB', (64, 64), 'blue')
                for y in range(24, 40):
                    for x in range(24, 40):
                        image.putpixel((x, y), (255, 0, 0))
                image.save(root / 'tile/img.png')
                mask = Image.new('L', (64, 64), 0)
                for y in range(28, 36):
                    for x in range(28, 36):
                        mask.putpixel((x, y), 255)
                mask.save(root / 'tile/mask.png')
                row = {'img_path': 'tile/img.png', 'mask_path': 'tile/mask.png',
                       'cls_name': 'tile', 'specie_name': 'defect', 'anomaly': 1}
                (root / 'meta.json').write_text(json.dumps({'test': {'tile': [row]}}))
                kwargs = dict(root=str(root), transform=lambda im: im.copy(),
                              target_transform=lambda im: im.copy(), dataset_name='mvtec', zoom_aug_p=1.0)
                random.seed(42)
                legacy = dataset_module.Dataset(**kwargs)[0]
                next_legacy = random.random()
                random.seed(42)
                paired = dataset_module.Dataset(**kwargs, return_original=True)[0]
                next_paired = random.random()
                self.assertEqual(next_legacy, next_paired)
                self.assertEqual(legacy['img'].tobytes(), paired['img'].tobytes())
                self.assertEqual(paired['img_original'].size, (64, 64))
                self.assertLess(paired['img'].width, 64)
                self.assertNotIn('img_original', legacy)
        finally:
            for key, value in old.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

    def test_pilot_gate(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from collect_exp042 import medical
        from collect_exp024 import IMG, PIX
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind, sets in (('pix', PIX), ('img', IMG)):
                for _, key in sets:
                    for arm in ('split', 'sham'):
                        out = root / 'results/EXP-042' / arm / key
                        out.mkdir(parents=True)
                        if kind == 'pix':
                            metric = {'pixel_auroc': .90, 'pixel_aupro': .80}
                            name = 'pixel_level_metrics.json'
                        else:
                            metric = {'image_auroc': .82 if arm == 'split' else .80,
                                      'image_ap': .75}
                            name = 'image_level_metrics.json'
                        (out / name).write_text(json.dumps({'mean': metric}))
            self.assertEqual(medical(root)['pilot_gate'], 'ADVANCE_TO_REPLICATION')
            bad = root / 'results/EXP-042/split/isic/pixel_level_metrics.json'
            bad.write_text(json.dumps({'mean': {'pixel_auroc': .88, 'pixel_aupro': .80}}))
            self.assertEqual(medical(root)['pilot_gate'], 'STOP_PILOT')


if __name__ == '__main__':
    unittest.main()
