"""EXP-039 source selection and frozen-rule target report (stdlib only)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


CANDIDATES = ('top5_score', 'top1_score', 'top10_score')  # tie priority
TARGETS = ('HeadCT_anomaly_detection', 'BrainMRI', 'br35')
MATCHED_ECP_SHA256 = '7278af957c4087d7c00e7495e4cba3a65c253b5d0c9646b36434b951e3df0e7c'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def auc_ap(labels, scores):
    if len(labels) != len(scores) or not labels or set(labels) != {0, 1}:
        raise ValueError('AUROC/AP require paired scores and both labels')
    if any(not math.isfinite(score) for score in scores):
        raise ValueError('non-finite score')
    positives = sum(labels)
    negatives = len(labels) - positives
    ordered = sorted(zip(scores, labels), reverse=True)
    true_pos = false_pos = 0
    auc_count = average_precision = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        block_pos = sum(label for _, label in ordered[cursor:end])
        block_neg = end - cursor - block_pos
        auc_count += block_pos * (negatives - false_pos - block_neg / 2)
        true_pos += block_pos
        false_pos += block_neg
        average_precision += (block_pos / positives) * (true_pos / (true_pos + false_pos))
        cursor = end
    return {'auroc': auc_count / (positives * negatives), 'ap': average_precision}


def load_run(folder, expected_dataset, expected_mode, reference=None):
    folder = Path(folder)
    with (folder / 'evaluation_metadata.json').open() as handle:
        meta = json.load(handle)
    if (meta['dataset'], meta['dataset_mode'], meta['split']) != (expected_dataset, expected_mode, 'test'):
        raise ValueError(f'wrong dataset/mode/split: {folder}')
    if not meta['arguments'].get('exp039_pool_scores') or not meta['arguments'].get('export_image_scores'):
        raise ValueError(f'EXP-039 exports missing: {folder}')
    if meta['arguments'].get('metrics') != 'image-level' or meta['z_img'] != -0.69 or meta['z_pix'] != 1.6 or meta['sigma'] != 4:
        raise ValueError(f'wrong image evaluation settings: {folder}')
    if meta['seed'] != 111 or meta['checkpoint_extent_cond'] != 'extent':
        raise ValueError(f'wrong seed or checkpoint: {folder}')
    if meta['checkpoint_sha256'] != MATCHED_ECP_SHA256:
        raise ValueError(f'not the matched EXP-012 ECP checkpoint: {folder}')
    settings = meta['arguments']
    required = {
        'features_list': [24], 'feature_map_layer': [0, 1, 2, 3],
        'image_size': 518, 'depth': 9, 'n_ctx': 12, 't_n_ctx': 4,
        'paa_scales': [], 'distractor_suppress': 'none',
        'distractor_rerank': 'none', 'ec_oracle': False,
        'ec_const_z': None,
    }
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError(f'wrong model or map settings: {folder}')
    if reference:
        for key in ('checkpoint_sha256', 'pretrained_clip_weights_sha256', 'evaluation_source_sha256'):
            if meta[key] != reference[key]:
                raise ValueError(f'{key} mismatch: {folder}')
    with (folder / 'exp039_pool_scores.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (folder / 'image_level_predictions.csv').open(newline='') as handle:
        ordinary = list(csv.DictReader(handle))
    if len(rows) != len(ordinary) or len(rows) != meta['sample_count'] or not rows:
        raise ValueError(f'sample count mismatch: {folder}')
    paired = {(row['sample_id'], row['ground_truth_image_label']): row for row in ordinary}
    if len(paired) != len(ordinary):
        raise ValueError(f'duplicate ordinary image ID: {folder}')
    ids = set()
    for row in rows:
        key = (row['sample_id'], row['ground_truth_image_label'])
        if row['dataset'] != expected_dataset or key not in paired or key in ids:
            raise ValueError(f'ID, label, or dataset mismatch: {folder}')
        ids.add(key)
        if abs(float(row['cls_score']) - float(paired[key]['image_anomaly_score'])) > 1e-8:
            raise ValueError(f'CLS export parity failed: {folder}')
    if len(ids) != len(paired):
        raise ValueError(f'unpaired image ID: {folder}')
    groups = {}
    for row in rows:
        label = int(row['ground_truth_image_label'])
        if label not in (0, 1):
            raise ValueError(f'nonbinary label: {folder}')
        groups.setdefault(row['category'], []).append(row)
    metrics = {}
    for category, group in groups.items():
        labels = [int(row['ground_truth_image_label']) for row in group]
        metrics[category] = {key: auc_ap(labels, [float(row[key]) for row in group])
                             for key in ('cls_score', *CANDIDATES)}
    with (folder / 'image_level_metrics.json').open() as handle:
        ordinary_metrics = json.load(handle)
    for category, values in metrics.items():
        for calculated, exported in (('auroc', 'image_auroc'), ('ap', 'image_ap')):
            if abs(values['cls_score'][calculated] - ordinary_metrics[category][exported]) > 1e-8:
                raise ValueError(f'CLS metric parity failed for {category}: {folder}')
    return meta, rows, metrics


def select(root):
    folder = root / 'source_mvtec'
    meta, rows, metrics = load_run(folder, 'mvtec', 'mvtec')
    if len(metrics) != 15:
        raise ValueError(f'expected 15 MVTec categories, got {len(metrics)}')
    macro = {key: sum(metric[key]['auroc'] for metric in metrics.values()) / len(metrics)
             for key in ('cls_score', *CANDIDATES)}
    chosen = max(CANDIDATES, key=lambda key: macro[key])
    record = {
        'experiment': 'EXP-039', 'source_dataset': 'mvtec',
        'source_scores_sha256': sha256(folder / 'exp039_pool_scores.csv'),
        'checkpoint_sha256': meta['checkpoint_sha256'],
        'evaluation_source_sha256': meta['evaluation_source_sha256'],
        'rule': chosen, 'tie_priority': list(CANDIDATES),
        'source_macro_auroc': macro, 'source_category_count': len(metrics),
        'source_sample_count': len(rows),
    }
    destination = root / 'selection.json'
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise ValueError(f'frozen selection differs: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2) + '\n')
    return record


def report(root):
    destination = root / 'selection.json'
    if not destination.is_file():
        raise ValueError('run source and select before target/report')
    frozen = json.loads(destination.read_text())
    if select(root) != frozen:
        raise ValueError('source selection changed')
    source_meta, _, _ = load_run(root / 'source_mvtec', 'mvtec', 'mvtec')
    results = {}
    for name in TARGETS:
        meta, rows, metrics = load_run(root / name, name, 'brain', source_meta)
        if len(metrics) != 1 or 'brain' not in metrics:
            raise ValueError(f'expected one brain category: {name}')
        baseline = metrics['brain']['cls_score']
        pooled = metrics['brain'][frozen['rule']]
        results[name] = {
            'samples': len(rows), 'checkpoint_sha256': meta['checkpoint_sha256'],
            'cls': baseline, 'pool': pooled,
            'delta_auroc_points': 100 * (pooled['auroc'] - baseline['auroc']),
            'delta_ap_points': 100 * (pooled['ap'] - baseline['ap']),
        }
    deltas = [value['delta_auroc_points'] for value in results.values()]
    ap_deltas = [value['delta_ap_points'] for value in results.values()]
    if sum(delta >= 1.0 for delta in deltas) >= 2 and all(delta >= -1.0 for delta in deltas) and all(delta >= -2.0 for delta in ap_deltas):
        verdict = 'ADVANCE'
    elif all(delta < 1.0 for delta in deltas) or sum(delta <= -1.0 for delta in deltas) >= 2:
        verdict = 'FAIL'
    else:
        verdict = 'INCONCLUSIVE'
    record = {'experiment': 'EXP-039', 'selected_rule': frozen['rule'],
              'selection_sha256': sha256(destination), 'targets': results, 'verdict': verdict}
    (root / 'report.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('select', 'report'))
    parser.add_argument('--root', type=Path, default=Path('results/EXP-039'))
    args = parser.parse_args()
    print(json.dumps(select(args.root) if args.phase == 'select' else report(args.root), indent=2))


if __name__ == '__main__':
    main()
