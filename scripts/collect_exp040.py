"""Validate EXP-040 paired scores and apply its registered source-only gate."""

import csv
import json
import math
from pathlib import Path

from exp040_core import CATEGORIES, ARMS, VIEWS, percentile, sha256_file, summarize


EXPECTED_SETTINGS = {'image_size': 518, 'features_list': [24], 'dpam_layer': 20,
                     'depth': 9, 'n_ctx': 12, 't_n_ctx': 4, 'seed': 111}


def collect(root):
    root = Path(root)
    with (root / 'geometry.json').open() as handle:
        geometry = json.load(handle)
    with (root / 'provenance.json').open() as handle:
        provenance = json.load(handle)
    if geometry['experiment'] != 'EXP-040' or provenance['experiment'] != 'EXP-040':
        raise ValueError('wrong experiment')
    if geometry['training_source_split'] != 'test' or geometry['probe_normal_split'] != 'train':
        raise ValueError('wrong MVTec source split')
    if geometry['selected_normal_count'] != 480 or provenance['image_count'] != 480:
        raise ValueError('wrong image count')
    if provenance['smoke'] or provenance['score_count'] != 1440:
        raise ValueError('full probe scores required')
    if provenance['settings'] != EXPECTED_SETTINGS:
        raise ValueError('wrong model settings')
    if provenance['geometry_sha256'] != sha256_file(root / 'geometry.json'):
        raise ValueError('geometry changed after probe')
    if provenance['views_sha256'] != sha256_file(root / 'views.jsonl'):
        raise ValueError('view manifest changed after probe')
    if provenance['score_sha256'] != sha256_file(root / 'scores.csv'):
        raise ValueError('score file changed after probe')
    if geometry['views_sha256'] != provenance['views_sha256']:
        raise ValueError('views hash mismatch')
    paths = provenance['checkpoint_paths']
    if (not paths['ctrl'].endswith('/checkpoints/matched_ctrl_s111/epoch_15.pth')
            or not paths['zo'].endswith('/checkpoints/EXP-012-matched/zoom_only/checkpoints/epoch_15.pth')
            or provenance['checkpoint_sha256']['ctrl'] == provenance['checkpoint_sha256']['zo']):
        raise ValueError('wrong or identical checkpoint paths/hashes')
    if any(len(geometry['successful_geometries'][category]) < 3 for category in CATEGORIES):
        raise ValueError('insufficient successful source crop geometry')
    with (root / 'views.jsonl').open() as handle:
        views = [json.loads(line) for line in handle if line.strip()]
    expected = {(row['category'], row['sample_id']): row for row in views}
    if len(expected) != 480 or any(row['category'] not in CATEGORIES for row in views):
        raise ValueError('duplicate or unknown selected view')
    if any('/train/good/' not in row['sample_id'].replace('\\', '/') for row in views):
        raise ValueError('probe contains an image exposed in the MVTec test split')
    scores = {}
    with (root / 'scores.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['category'], row['sample_id'])
            if key not in expected or row['view'] not in VIEWS:
                raise ValueError('score row not in selected normal views')
            if abs(float(row['sham_side_ratio']) - expected[key]['sham_side_ratio']) > 1e-10:
                raise ValueError('score-view geometry mismatch')
            view_scores = {arm: float(row[f'{arm}_margin']) for arm in ARMS}
            if any(not math.isfinite(value) for value in view_scores.values()):
                raise ValueError('non-finite score')
            record = scores.setdefault(key, {})
            if row['view'] in record:
                raise ValueError('duplicate image/view score')
            record[row['view']] = view_scores
    if set(scores) != set(expected) or any(set(record) != set(VIEWS) for record in scores.values()):
        raise ValueError('unpaired or missing view')
    groups = {category: [] for category in CATEGORIES}
    for (category, _sample_id), record in scores.items():
        groups[category].append(record)
    result = summarize(groups)
    arm_category_changes = {
        category: {
            arm: {
                view: {
                    'raw_mean': sum(record[view][arm] - record['original'][arm]
                                    for record in groups[category]) / 32,
                    'standardized_mean': sum(record[view][arm] - record['original'][arm]
                                             for record in groups[category]) / 32
                    / result['normal_score_sd'][arm],
                }
                for view in ('crop', 'sham')
            } for arm in ARMS
        } for category in CATEGORIES
    }
    raw_changes = {
        arm: {
            view: sum(
                record[view][arm] - record['original'][arm]
                for category in CATEGORIES for record in groups[category]) / 480
            for view in ('crop', 'sham')
        } for arm in ARMS
    }
    source_counts = geometry['category_counts']
    attempts = sum(source_counts[category]['attempts'] for category in CATEGORIES)
    successes = sum(source_counts[category]['successful'] for category in CATEGORIES)
    side_fractions = {
        category: [row['side_fraction'] for row in geometry['successful_geometries'][category]]
        for category in CATEGORIES
    }
    side_summary = {
        category: {'p10': percentile(values, 10), 'median': percentile(values, 50),
                   'p90': percentile(values, 90)}
        for category, values in side_fractions.items()
    }
    report = {
        'experiment': 'EXP-040', 'source_only': True,
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'geometry_sha256': provenance['geometry_sha256'],
        'views_sha256': provenance['views_sha256'],
        'scores_sha256': provenance['score_sha256'],
        'source_audit': {'attempts': attempts, 'successful': successes,
                         'success_fraction_of_attempts': successes / attempts,
                         'by_category': source_counts, 'side_fraction_by_category': side_summary},
        'probe_image_count': 480, 'raw_mean_margin_change': raw_changes,
        'arm_category_changes': arm_category_changes,
        **result,
        'interpretation_limit': 'A positive gate licenses a balanced-zoom follow-up only; crops alter context and scale.',
    }
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/EXP-040'))
    args = parser.parse_args()
    report = collect(args.root)
    concise = {key: report[key] for key in (
        'experiment', 'source_audit', 'probe_image_count', 'normal_score_sd',
        'raw_mean_margin_change', 'macro_effect', 'bootstrap_95_ci',
        'positive_crop_categories', 'verdict')}
    concise['per_category'] = report['per_category']
    print(json.dumps(concise, indent=2))


if __name__ == '__main__':
    main()
