"""EXP-041: analyze EXP-040 crop severity and export a blind retention gallery."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from exp040_core import CATEGORIES, percentile, sha256_file


VIEWS = ('original', 'crop', 'sham')
ARMS = ('ctrl', 'zo')
LABELS = ('yes', 'no', 'uncertain')
TEXTURE_CATEGORIES = frozenset(('carpet', 'grid', 'leather', 'tile', 'wood'))


def read_inputs(root):
    root = Path(root)
    with (root / 'provenance.json').open() as handle:
        provenance = json.load(handle)
    with (root / 'report.json').open() as handle:
        prior = json.load(handle)
    with (root / 'geometry.json').open() as handle:
        geometry = json.load(handle)
    if any(item['experiment'] != 'EXP-040' for item in (provenance, prior, geometry)):
        raise ValueError('wrong source experiment')
    if provenance['smoke'] or provenance['image_count'] != 480 or provenance['score_count'] != 1440:
        raise ValueError('full EXP-040 probe required')
    for name, key in (('geometry.json', 'geometry_sha256'), ('views.jsonl', 'views_sha256'),
                      ('scores.csv', 'score_sha256')):
        if sha256_file(root / name) != provenance[key]:
            raise ValueError(f'{name} hash mismatch')
    if geometry['views_sha256'] != provenance['views_sha256']:
        raise ValueError('geometry/view hash mismatch')
    if any(prior[key] != provenance[source_key] for key, source_key in (
            ('geometry_sha256', 'geometry_sha256'), ('views_sha256', 'views_sha256'),
            ('scores_sha256', 'score_sha256'))):
        raise ValueError('EXP-040 report input hashes disagree')
    with (root / 'views.jsonl').open() as handle:
        views = [json.loads(line) for line in handle if line.strip()]
    selected = {}
    counts = defaultdict(int)
    for row in views:
        key = (row['category'], row['sample_id'])
        if key in selected or row['category'] not in CATEGORIES:
            raise ValueError('duplicate or unknown selected image')
        if '/train/good/' not in row['sample_id'].replace('\\', '/'):
            raise ValueError('selected view outside MVTec train/good')
        width, height = row['image_size']
        x0, y0, x1, y1 = row['pixel_box']
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('invalid pixel box')
        ratio = math.sqrt((x1 - x0) / width * (y1 - y0) / height)
        if abs(ratio - row['sham_side_ratio']) > 1e-10:
            raise ValueError('ratio/box mismatch')
        selected[key] = row
        counts[row['category']] += 1
    if len(selected) != 480 or any(counts[c] != 32 for c in CATEGORIES):
        raise ValueError('expected 32 selected normals in each of 15 categories')
    scores = defaultdict(dict)
    with (root / 'scores.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['category'], row['sample_id'])
            if key not in selected or row['view'] not in VIEWS or row['view'] in scores[key]:
                raise ValueError('unknown or duplicate score view')
            if abs(float(row['sham_side_ratio']) - selected[key]['sham_side_ratio']) > 1e-10:
                raise ValueError('score/view ratio mismatch')
            values = {arm: float(row[f'{arm}_margin']) for arm in ARMS}
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError('non-finite score')
            scores[key][row['view']] = values
    if set(scores) != set(selected) or any(set(scores[key]) != set(VIEWS) for key in selected):
        raise ValueError('missing paired view scores')
    scales = {arm: statistics.pstdev(scores[key]['original'][arm] for key in selected)
              for arm in ARMS}
    if any(abs(scales[arm] - prior['normal_score_sd'][arm]) > 1e-10 for arm in ARMS):
        raise ValueError('original score normalization disagrees with EXP-040')
    return provenance, selected, scores, scales


def build_rows(selected, scores, scales):
    """Assign fixed within-category quartiles and paired standardized contrasts."""
    rows = []
    for category in CATEGORIES:
        keys = sorted((key for key in selected if key[0] == category),
                      key=lambda key: (selected[key]['sham_side_ratio'], key[1]))
        if len(keys) != 32:
            raise ValueError('expected 32 images per category')
        for rank, key in enumerate(keys):
            view = selected[key]
            values = scores[key]
            def contrast(name):
                return ((values[name]['zo'] - values['original']['zo']) / scales['zo']
                        - (values[name]['ctrl'] - values['original']['ctrl']) / scales['ctrl'])
            crop = contrast('crop')
            sham = contrast('sham')
            rows.append({'category': category, 'sample_id': key[1],
                         'side_ratio': view['sham_side_ratio'], 'rank': rank + 1,
                         'quartile': rank // 8 + 1, 'd_crop': crop,
                         'd_sham': sham, 'd_specific': crop - sham})
    return rows


def summarize(rows, n_boot=2000, seed=41):
    groups = {(category, q): [] for category in CATEGORIES for q in range(1, 5)}
    for row in rows:
        groups[(row['category'], row['quartile'])].append(row)
    if len(rows) != 480 or any(len(group) != 8 for group in groups.values()):
        raise ValueError('expected eight paired images per category/quartile')
    rng = random.Random(seed)
    result = {}
    for q in range(1, 5):
        category_means = {
            category: {name: statistics.mean(row[name] for row in groups[(category, q)])
                       for name in ('d_crop', 'd_sham', 'd_specific')}
            for category in CATEGORIES
        }
        mean = {name: statistics.mean(category_means[c][name] for c in CATEGORIES)
                for name in ('d_crop', 'd_sham', 'd_specific')}
        draws = {name: [] for name in mean}
        for _ in range(n_boot):
            sampled = {name: 0.0 for name in mean}
            for category in CATEGORIES:
                group = groups[(category, q)]
                for _ in range(8):
                    item = rng.choice(group)
                    for name in sampled:
                        sampled[name] += item[name] / 120
            for name in draws:
                draws[name].append(sampled[name])
        subset = [row for category in CATEGORIES for row in groups[(category, q)]]
        ratios = [row['side_ratio'] for row in subset]
        result[str(q)] = {
            'image_count': len(subset), 'side_ratio_min': min(ratios),
            'side_ratio_median': statistics.median(ratios), 'side_ratio_max': max(ratios),
            'macro_mean': mean,
            'bootstrap_95_ci': {name: [percentile(draws[name], 2.5),
                                       percentile(draws[name], 97.5)] for name in mean},
            'positive_categories': {name: sum(category_means[c][name] > 0 for c in CATEGORIES)
                                    for name in mean},
            'per_category': category_means,
        }
    return result


def write_analysis(root, output):
    output = Path(output)
    if (output / 'analysis.json').exists() or (output / 'per_image.csv').exists():
        raise FileExistsError('EXP-041 analysis already exists')
    provenance, selected, scores, scales = read_inputs(root)
    rows = build_rows(selected, scores, scales)
    quartiles = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_image.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'experiment': 'EXP-041', 'status': 'severity_analysis_only',
              'source_experiment': 'EXP-040',
              'source_hashes': {'views_sha256': provenance['views_sha256'],
                                'scores_sha256': provenance['score_sha256'],
                                'geometry_sha256': provenance['geometry_sha256']},
              'normal_score_sd': scales, 'image_count': len(rows),
              'bootstrap_resamples': 2000, 'bootstrap_seed': 41,
              'quartiles': quartiles, 'per_image_sha256': sha256_file(output / 'per_image.csv'),
              'interpretation_limit': 'Descriptive source-only associations; side ratio also changes visible content.'}
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def export_gallery(root, output, data_root):
    from PIL import Image, ImageDraw

    resample = getattr(Image, 'Resampling', Image).BICUBIC
    output = Path(output)
    gallery = output / 'gallery'
    if gallery.exists() or (output / 'retention_labels.csv').exists():
        raise FileExistsError('EXP-041 gallery or annotation template already exists')
    provenance, selected, scores, scales = read_inputs(root)
    del scores, scales
    if not (output / 'analysis.json').is_file():
        raise FileNotFoundError('run analyze before export')
    analysis = json.loads((output / 'analysis.json').read_text())
    if analysis['source_hashes']['views_sha256'] != provenance['views_sha256']:
        raise ValueError('analysis uses different selected views')
    data_root = Path(data_root)
    rows = []
    # Verify every source input before creating any export files.
    for category in CATEGORIES:
        category_rows = [row for row in selected.values() if row['category'] == category]
        for index, view in enumerate(category_rows):
            path = data_root / view['sample_id']
            if sha256_file(path) != view['image_sha256']:
                raise ValueError(f'source image hash mismatch: {view["sample_id"]}')
            with Image.open(path) as image:
                if list(image.size) != view['image_size']:
                    raise ValueError(f'source image dimensions changed: {view["sample_id"]}')
            rows.append((category, index, view, path))
    gallery.mkdir(parents=True)
    manifest = []
    for category, index, view, path in rows:
        with Image.open(path) as source:
            image = source.convert('RGB')
        paired = Image.new('RGB', (448, 224))
        paired.paste(image.resize((224, 224), resample), (0, 0))
        paired.paste(image.crop(tuple(view['pixel_box'])).resize((224, 224), resample), (224, 0))
        filename = f'{category}_{index:02d}.jpg'
        paired.save(gallery / filename, quality=95)
        manifest.append({'category': category, 'index': index, 'sample_id': view['sample_id'],
                         'pair_file': filename, 'pair_sha256': sha256_file(gallery / filename),
                         'source_sha256': view['image_sha256'],
                         'side_ratio': view['sham_side_ratio'],
                         'pixel_box': json.dumps(view['pixel_box'])})
    for category in CATEGORIES:
        category_manifest = [row for row in manifest if row['category'] == category]
        for sheet_number in range(2):
            canvas = Image.new('RGB', (4 * 448, 4 * 248), 'white')
            draw = ImageDraw.Draw(canvas)
            for position, row in enumerate(category_manifest[sheet_number * 16:(sheet_number + 1) * 16]):
                x = (position % 4) * 448
                y = (position // 4) * 248
                draw.text((x + 4, y + 3), f'{category}_{row["index"]:02d}', fill='black')
                with Image.open(gallery / row['pair_file']) as pair:
                    canvas.paste(pair, (x, y + 24))
            canvas.save(gallery / f'{category}_sheet_{sheet_number + 1}.jpg', quality=92)
    with (output / 'gallery_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (output / 'retention_labels.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['category', 'index', 'sample_id',
                                                    'pair_file', 'recognizable', 'comment'])
        writer.writeheader()
        for row in manifest:
            writer.writerow({key: row[key] for key in ('category', 'index', 'sample_id', 'pair_file')})
    record = {'experiment': 'EXP-041', 'image_count': len(manifest),
              'source_views_sha256': provenance['views_sha256'],
              'gallery_manifest_sha256': sha256_file(output / 'gallery_manifest.csv'),
              'annotation_template_sha256': sha256_file(output / 'retention_labels.csv')}
    (output / 'export_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def collect_labels(output):
    output = Path(output)
    with (output / 'analysis.json').open() as handle:
        analysis = json.load(handle)
    if (output / 'retention_report.json').exists():
        raise FileExistsError('retention report already exists')
    with (output / 'gallery_manifest.csv').open(newline='') as handle:
        manifest = list(csv.DictReader(handle))
    with (output / 'retention_labels.csv').open(newline='') as handle:
        labels = list(csv.DictReader(handle))
    by_key = {(row['category'], row['sample_id']): row for row in manifest}
    if len(manifest) != 480 or len(by_key) != 480 or len(labels) != 480:
        raise ValueError('expected 480 unique gallery pairs and annotations')
    seen = set()
    labeled = []
    for row in labels:
        key = (row['category'], row['sample_id'])
        if key not in by_key or key in seen:
            raise ValueError('unknown or duplicate annotation')
        seen.add(key)
        if any(row[field] != by_key[key][field] for field in ('index', 'pair_file')):
            raise ValueError('annotation/gallery identity mismatch')
        label = row['recognizable'].strip().lower()
        if label not in LABELS:
            raise ValueError(f'missing or invalid annotation: {key}')
        if sha256_file(output / 'gallery' / row['pair_file']) != by_key[key]['pair_sha256']:
            raise ValueError('gallery pair image changed')
        labeled.append((key, label))
    if seen != set(by_key):
        raise ValueError('missing gallery annotation')
    with (output / 'per_image.csv').open(newline='') as handle:
        effects = {(row['category'], row['sample_id']): row for row in csv.DictReader(handle)}
    if set(effects) != seen or sha256_file(output / 'per_image.csv') != analysis['per_image_sha256']:
        raise ValueError('per-image analysis changed or is incomplete')
    counts = {kind: {label: 0 for label in LABELS} for kind in ('object', 'texture')}
    detail = defaultdict(list)
    for key, label in labeled:
        kind = 'texture' if key[0] in TEXTURE_CATEGORIES else 'object'
        counts[kind][label] += 1
        detail[(kind, label, effects[key]['quartile'])].append(float(effects[key]['d_specific']))
    summary = {kind: {label: {str(q): {'n': len(detail[(kind, label, str(q))]),
                                      'mean_d_specific': (statistics.mean(detail[(kind, label, str(q))])
                                                          if detail[(kind, label, str(q))] else None)}
                              for q in range(1, 5)} for label in LABELS}
               for kind in counts}
    report = {'experiment': 'EXP-041', 'status': 'complete', 'image_count': 480,
              'label_counts': counts, 'd_specific_by_label_type_quartile': summary,
              'labels_sha256': sha256_file(output / 'retention_labels.csv'),
              'unfiltered_geometry_approved': all(label == 'yes' for _, label in labeled),
              'interpretation_limit': 'Visual retention labels and score associations are descriptive, not causal.'}
    (output / 'retention_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('analyze', 'export', 'collect'))
    parser.add_argument('--exp040', type=Path, default=Path('results/EXP-040'))
    parser.add_argument('--output', type=Path, default=Path('results/EXP-041'))
    parser.add_argument('--data-root', type=Path,
                        default=Path('/home/ai3/NguyenND/AnomalyClipV2/data/mvtec'))
    args = parser.parse_args()
    if args.phase == 'analyze':
        report = write_analysis(args.exp040, args.output)
        print(json.dumps({'image_count': report['image_count'],
                          'quartiles': {q: values['macro_mean']
                                        for q, values in report['quartiles'].items()}}, indent=2))
    elif args.phase == 'export':
        print(json.dumps(export_gallery(args.exp040, args.output, args.data_root), indent=2))
    else:
        print(json.dumps(collect_labels(args.output), indent=2))


if __name__ == '__main__':
    main()
