"""Audit EXP-042 paired seed artifacts and apply the registered three-seed rule."""

import json
import math
import statistics
import sys
from pathlib import Path

from collect_exp024 import IMG, PIX
from collect_exp042 import metric_pair

SEEDS = (111, 222, 333)
ARMS = ('split', 'sham')


def evaluation_dir(root, seed, arm, key):
    prefix = root / 'results/EXP-042'
    return prefix / arm / key if seed == 111 else prefix / f's{seed}' / arm / key


def load(root):
    records, missing = {}, []
    for kind, datasets in (('pix', PIX), ('img', IMG)):
        for name, key in datasets:
            records[name] = {'kind': kind, 'key': key, 'seeds': {}}
            metric_name = 'pixel_level_metrics.json' if kind == 'pix' else 'image_level_metrics.json'
            for seed in SEEDS:
                arms = {}
                for arm in ARMS:
                    folder = evaluation_dir(root, seed, arm, key)
                    metrics = metric_pair(folder / metric_name, kind)
                    metadata_path = folder / 'evaluation_metadata.json'
                    if metrics is None or not metadata_path.is_file():
                        missing.append(f'{seed}/{arm}/{key}')
                        continue
                    if not all(math.isfinite(value) for value in metrics):
                        raise ValueError(f'non-finite metrics: {seed}/{arm}/{key}')
                    arms[arm] = {'metrics': metrics, 'metadata': json.loads(metadata_path.read_text())}
                records[name]['seeds'][str(seed)] = arms
    return records, missing


def audit(records):
    reference_source, reference_clip = None, None
    checkpoint_hashes = {}
    for name, entry in records.items():
        dataset_reference = None
        for seed in SEEDS:
            pair = entry['seeds'][str(seed)]
            if set(pair) != set(ARMS):
                raise ValueError(f'missing pair: {name}/{seed}')
            left, right = (pair[arm]['metadata'] for arm in ARMS)
            for field in ('data_path', 'meta_json_sha256', 'sample_count',
                          'positive_count', 'negative_count'):
                if left[field] != right[field]:
                    raise ValueError(f'{name}/{seed}: mismatched {field}')
            dataset_signature = tuple(left[field] for field in
                                      ('data_path', 'meta_json_sha256', 'sample_count',
                                       'positive_count', 'negative_count'))
            if dataset_reference is None:
                dataset_reference = dataset_signature
            elif dataset_reference != dataset_signature:
                raise ValueError(f'{name}/{seed}: dataset differs across seeds')
            for arm in ARMS:
                meta = pair[arm]['metadata']
                if (meta['seed'] != seed or meta['sigma'] != 4 or meta['split'] != 'test'
                        or meta['checkpoint_extent_cond'] not in (None, 'none')
                        or meta['checkpoint_ec_static_aux']
                        or meta['z_img'] is not None or meta['z_pix'] is not None
                        or not meta['checkpoint_path'].endswith(
                            f'/checkpoints/exp042_{arm}_s{seed}/epoch_15.pth')):
                    raise ValueError(f'{name}/{seed}/{arm}: wrong evaluation configuration')
                if meta['arguments']['metrics'] != ('pixel-level' if entry['kind'] == 'pix' else 'image-level'):
                    raise ValueError(f'{name}/{seed}/{arm}: wrong metric mode')
                if reference_source is None:
                    reference_source = meta['evaluation_source_sha256']
                    reference_clip = meta['pretrained_clip_weights_sha256']
                if (meta['evaluation_source_sha256'] != reference_source
                        or meta['pretrained_clip_weights_sha256'] != reference_clip):
                    raise ValueError(f'{name}/{seed}/{arm}: evaluation source or CLIP weights changed')
                key = f'{arm}_s{seed}'
                previous = checkpoint_hashes.setdefault(key, meta['checkpoint_sha256'])
                if previous != meta['checkpoint_sha256']:
                    raise ValueError(f'{key}: checkpoint differs across datasets')
    return {'evaluation_source_sha256': reference_source,
            'pretrained_clip_weights_sha256': reference_clip,
            'checkpoint_sha256': checkpoint_hashes}


def summarize(records):
    summary = {}
    for name, entry in records.items():
        deltas = []
        for seed in SEEDS:
            pair = entry['seeds'][str(seed)]
            deltas.append([a - b for a, b in zip(pair['split']['metrics'], pair['sham']['metrics'])])
        summary[name] = {
            'kind': entry['kind'],
            'delta_by_seed': {str(seed): delta for seed, delta in zip(SEEDS, deltas)},
            'mean_delta': [statistics.mean(values) for values in zip(*deltas)],
            'positive_seed_count': [sum(delta[i] > 0 for delta in deltas) for i in (0, 1)],
        }
    image_wins = sum(summary[name]['mean_delta'][0] >= 1.0 for name, _ in IMG)
    image_signs = sum(summary[name]['positive_seed_count'][0] >= 2 for name, _ in IMG)
    image_safe = all(summary[name]['mean_delta'][0] >= -1.0 for name, _ in IMG)
    pixel_safe = all(summary[name]['mean_delta'][i] >= -1.0
                     for name, _ in PIX for i in (0, 1))
    verdict = ('REPLICATION_GATE_PASS' if image_wins >= 2 and image_signs >= 2
               and image_safe and pixel_safe else 'REPLICATION_GATE_FAIL')
    return summary, verdict


def collect(root):
    records, missing = load(root)
    if missing:
        return {'status': 'INCOMPLETE', 'missing': missing}
    provenance = audit(records)
    summary, verdict = summarize(records)
    return {'status': verdict, 'missing': [], 'provenance': provenance, 'datasets': summary,
            'limits': 'Three seeds on the same development datasets; source crop probe only seed 111.'}


if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    report = collect(root)
    path = root / 'results/EXP-042/replication_report.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('EXP-042 replication:', report['status'])
    if report['status'] == 'INCOMPLETE':
        print('Missing:', ', '.join(report['missing']))
    else:
        for name, data in report['datasets'].items():
            print(name, data['kind'], 'mean B-C', data['mean_delta'],
                  'positive seeds', data['positive_seed_count'])
    print('Report:', path)
