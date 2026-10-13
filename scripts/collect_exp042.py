"""Collect EXP-042 medical metrics and paired source-normal crop response."""

import csv
import json
import math
import sys
from pathlib import Path

from exp040_core import CATEGORIES, VIEWS, sha256_file, summarize
from collect_exp024 import IMG, PIX, REF, EXP012_DIR, json_pixel, last_table_row, path_for


def metric_pair(path, kind):
    if not path.is_file():
        return None
    data = json.loads(path.read_text())['mean']
    names = ('pixel_auroc', 'pixel_aupro') if kind == 'pix' else ('image_auroc', 'image_ap')
    return [100.0 * float(data[name]) for name in names]


def medical(root):
    records, missing = {}, []
    for kind, sets in (('pix', PIX), ('img', IMG)):
        for label, key in sets:
            line = {}
            filename = 'pixel_level_metrics.json' if kind == 'pix' else 'image_level_metrics.json'
            for arm in ('split', 'sham'):
                line[arm] = metric_pair(root / 'results/EXP-042' / arm / key / filename, kind)
                if line[arm] is None:
                    missing.append(f'{arm}/{key}/{filename}')
            if kind == 'pix':
                base = root / REF / EXP012_DIR[label] / 'zoom_only/pixel_level_metrics.json'
                line['historical_zoom'] = json_pixel(str(base))
            else:
                base = root / path_for('ZO', 111, kind, key, label)
                line['historical_zoom'] = last_table_row(str(base))
            deltas = {}
            if line['split'] is not None and line['sham'] is not None:
                deltas['split_minus_sham'] = [a - b for a, b in zip(line['split'], line['sham'])]
            if line['split'] is not None and line['historical_zoom'] is not None:
                deltas['split_minus_historical_zoom'] = [a - b for a, b in
                                                        zip(line['split'], line['historical_zoom'])]
            if line['sham'] is not None and line['historical_zoom'] is not None:
                deltas['sham_minus_historical_zoom'] = [a - b for a, b in
                                                       zip(line['sham'], line['historical_zoom'])]
            records[label] = {'kind': kind, 'metrics': line, 'deltas': deltas}
    verdict = 'INCOMPLETE'
    if not missing:
        image_deltas = [records[label]['metrics']['split'][0] - records[label]['metrics']['sham'][0]
                        for label, _ in IMG]
        pixel_deltas = [records[label]['metrics']['split'][i] - records[label]['metrics']['sham'][i]
                        for label, _ in PIX for i in (0, 1)]
        verdict = ('ADVANCE_TO_REPLICATION' if sum(d >= 1.0 for d in image_deltas) >= 2
                   and min(image_deltas) >= -1.0 and min(pixel_deltas) >= -1.0
                   else 'STOP_PILOT')
    return {'medical': records, 'missing_medical': missing, 'pilot_gate': verdict}


def source_probe(root):
    output = root / 'results/EXP-042/source'
    source = root / 'results/EXP-040'
    if not (output / 'scores.csv').is_file() or not (output / 'provenance.json').is_file():
        return {'status': 'INCOMPLETE'}
    provenance = json.loads((output / 'provenance.json').read_text())
    geometry = json.loads((source / 'geometry.json').read_text())
    if (provenance['experiment'] != 'EXP-042' or provenance['score_count'] != 1440
            or provenance['scores_sha256'] != sha256_file(output / 'scores.csv')
            or provenance['views_sha256'] != sha256_file(source / 'views.jsonl')
            or provenance['geometry_sha256'] != sha256_file(source / 'geometry.json')
            or geometry['views_sha256'] != provenance['views_sha256']):
        raise ValueError('EXP-042 source provenance mismatch')
    checkpoint_hash_verified = {}
    for arm in ('split', 'sham'):
        path = Path(provenance['checkpoint_paths'][arm])
        checkpoint_hash_verified[arm] = path.is_file()
        if path.is_file() and provenance['checkpoint_sha256'][arm] != sha256_file(path):
            raise ValueError(f'{arm} checkpoint changed after source probe')
    views = [json.loads(line) for line in (source / 'views.jsonl').read_text().splitlines() if line.strip()]
    expected = {(row['category'], row['sample_id']): row for row in views}
    if len(expected) != 480 or any(row['category'] not in CATEGORIES for row in views):
        raise ValueError('wrong source view manifest')
    grouped = {category: [] for category in CATEGORIES}
    pairs = {}
    with (output / 'scores.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['category'], row['sample_id']
            if key not in expected or row['view'] not in VIEWS:
                raise ValueError('unexpected source score')
            if abs(float(row['sham_side_ratio']) - expected[key]['sham_side_ratio']) > 1e-10:
                raise ValueError('source score geometry mismatch')
            scores = {'ctrl': float(row['sham_margin']), 'zo': float(row['split_margin'])}
            if not all(math.isfinite(v) for v in scores.values()):
                raise ValueError('non-finite source score')
            by_view = pairs.setdefault(key, {})
            if row['view'] in by_view:
                raise ValueError('duplicate source score')
            by_view[row['view']] = scores
    if set(pairs) != set(expected) or any(set(v) != set(VIEWS) for v in pairs.values()):
        raise ValueError('missing source score')
    for (category, _), record in pairs.items():
        grouped[category].append(record)
    summary = summarize(grouped, seed=42)
    summary.pop('verdict')  # EXP-040 gate belongs to its CTRL-versus-ZO contrast.
    return {'status': 'COMPLETE', 'split_minus_sham': summary,
            'checkpoint_hash_verified_locally': checkpoint_hash_verified,
            'interpretation': 'negative specific effect indicates reduced crop-specific response; not a causal proof'}


def collect(root):
    result = medical(root)
    result['source_probe'] = source_probe(root)
    return result


if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    report = collect(root)
    path = root / 'results/EXP-042/report.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'pilot_gate': report['pilot_gate'],
                      'missing_medical': report['missing_medical'],
                      'source_probe': report['source_probe']['status'],
                      'report_path': str(path)}, indent=2))
    for label, row in report['medical'].items():
        print(f"{label:10s} {row['kind']} B-C={row['deltas'].get('split_minus_sham', 'missing')} "
              f"B-A={row['deltas'].get('split_minus_historical_zoom', 'missing')}")
    if report['source_probe']['status'] == 'COMPLETE':
        source = report['source_probe']['split_minus_sham']
        print('Source B-C crop-specific:', source['macro_effect']['specific'],
              source['bootstrap_95_ci']['specific'])
