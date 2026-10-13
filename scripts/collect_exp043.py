"""Audit direct ECP versus ECP+split seed-111 evaluation and apply staged gates."""

import json
import math
import sys
from pathlib import Path

from collect_exp024 import IMG, PIX
from collect_exp042 import metric_pair

ARMS = ('ecp_baseline', 'ecp_split')
EXPECTED_BASELINE_SHA256 = '7278af957c4087d7c00e7495e4cba3a65c253b5d0c9646b36434b951e3df0e7c'


def read_pair(root, kind, key):
    filename = 'image_level_metrics.json' if kind == 'img' else 'pixel_level_metrics.json'
    pair = {}
    for arm in ARMS:
        folder = root / 'results/EXP-043' / arm / key
        metrics = metric_pair(folder / filename, kind)
        metadata_path = folder / 'evaluation_metadata.json'
        if metrics is None or not metadata_path.is_file():
            return None
        if any(not math.isfinite(value) for value in metrics):
            raise ValueError(f'non-finite {arm}/{key} metric')
        pair[arm] = {'metrics': metrics, 'metadata': json.loads(metadata_path.read_text())}
    return pair


def audit_pairs(pairs):
    source_hash, clip_hash, checkpoint_hashes = None, None, {}
    for name, entry in pairs.items():
        pair = entry['pair']
        baseline, split = (pair[arm]['metadata'] for arm in ARMS)
        for field in ('data_path', 'meta_json_sha256', 'sample_count',
                      'positive_count', 'negative_count'):
            if baseline[field] != split[field]:
                raise ValueError(f'{name}: mismatched {field}')
        for arm in ARMS:
            meta = pair[arm]['metadata']
            checkpoint_suffix = ('/checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth'
                                 if arm == 'ecp_baseline'
                                 else '/checkpoints/exp043_ecp_split_s111/epoch_15.pth')
            if (meta['seed'] != 111 or meta['sigma'] != 4 or meta['split'] != 'test'
                    or meta['checkpoint_extent_cond'] != 'extent'
                    or meta['checkpoint_ec_static_aux']
                    or meta['z_img'] != -0.69 or meta['z_pix'] != 1.6
                    or not meta['checkpoint_path'].endswith(checkpoint_suffix)
                    or meta['arguments']['metrics'] != ('image-level' if entry['kind'] == 'img' else 'pixel-level')):
                raise ValueError(f'{name}/{arm}: wrong ECP evaluation settings')
            if source_hash is None:
                source_hash = meta['evaluation_source_sha256']
                clip_hash = meta['pretrained_clip_weights_sha256']
            if (meta['evaluation_source_sha256'] != source_hash
                    or meta['pretrained_clip_weights_sha256'] != clip_hash):
                raise ValueError(f'{name}/{arm}: scoring source or CLIP weights changed')
            previous = checkpoint_hashes.setdefault(arm, meta['checkpoint_sha256'])
            if previous != meta['checkpoint_sha256']:
                raise ValueError(f'{arm}: checkpoint differs across datasets')
    if checkpoint_hashes['ecp_baseline'] == checkpoint_hashes['ecp_split']:
        raise ValueError('baseline and split checkpoints are identical')
    if checkpoint_hashes['ecp_baseline'] != EXPECTED_BASELINE_SHA256:
        raise ValueError('matched EXP-012 ECP checkpoint hash changed')
    return {'evaluation_source_sha256': source_hash,
            'pretrained_clip_weights_sha256': clip_hash,
            'checkpoint_sha256': checkpoint_hashes}


def summarize(pairs):
    return {name: {
        'kind': entry['kind'],
        'baseline': entry['pair']['ecp_baseline']['metrics'],
        'split': entry['pair']['ecp_split']['metrics'],
        'split_minus_baseline': [a - b for a, b in zip(
            entry['pair']['ecp_split']['metrics'], entry['pair']['ecp_baseline']['metrics'])],
    } for name, entry in pairs.items()}


def decide(metrics, pixel_complete):
    image_deltas = [metrics[name]['split_minus_baseline'] for name, _ in IMG]
    if (sum(delta[0] >= 0.5 for delta in image_deltas) < 2
            or any(delta[0] < -0.5 or delta[1] < -1.0 for delta in image_deltas)):
        return 'STOP_IMAGE'
    if not pixel_complete:
        return 'CONTINUE_PIXEL'
    if any(metrics[name]['split_minus_baseline'][index] < -1.0
           for name, _ in PIX for index in (0, 1)):
        return 'STOP_PIXEL'
    return 'PILOT_PASS_NEEDS_MATCHED_SHAM'


def collect(root):
    pairs, missing_image, missing_pixel = {}, [], []
    for kind, datasets in (('img', IMG), ('pix', PIX)):
        for name, key in datasets:
            pair = read_pair(root, kind, key)
            if pair is None:
                (missing_image if kind == 'img' else missing_pixel).append(name)
            else:
                pairs[name] = {'kind': kind, 'pair': pair}
    if missing_image:
        return {'status': 'INCOMPLETE_IMAGE', 'missing_image': missing_image,
                'missing_pixel': missing_pixel}
    provenance = audit_pairs(pairs)
    metrics = summarize(pairs)
    status = decide(metrics, not missing_pixel)
    return {'status': status, 'missing_image': [], 'missing_pixel': missing_pixel,
            'provenance': provenance, 'datasets': metrics,
            'limits': 'One ECP+split training seed on development sets; matched sham and further seeds conditional.'}


if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    report = collect(root)
    path = root / 'results/EXP-043/report.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('EXP-043:', report['status'])
    for name, entry in report.get('datasets', {}).items():
        print(name, entry['kind'], 'ECP', entry['baseline'],
              'ECP+split', entry['split'], 'delta', entry['split_minus_baseline'])
    print('Report:', path)
