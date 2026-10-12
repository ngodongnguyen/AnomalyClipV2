"""Pure helpers for EXP-040 geometry, provenance, and paired statistics."""

import hashlib
import math
import statistics
from pathlib import Path


CATEGORIES = (
    'carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid',
    'pill', 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood',
)
VIEWS = ('original', 'crop', 'sham')
ARMS = ('ctrl', 'zo')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_normals(rows, count=32):
    """Select IDs without depending on JSON order or filenames."""
    if len({row['img_path'] for row in rows}) != len(rows):
        raise ValueError('duplicate normal image path')
    ranked = sorted(rows, key=lambda row: (
        hashlib.sha256(row['img_path'].encode()).hexdigest(), row['img_path']))
    if len(ranked) < count:
        raise ValueError(f'expected at least {count} normal images, got {len(ranked)}')
    return ranked[:count]


def transfer_box(normalized_box, size):
    width, height = size
    if width <= 0 or height <= 0 or len(normalized_box) != 4:
        raise ValueError('invalid image or box')
    x0, y0, x1, y1 = normalized_box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('invalid normalized box')
    box = (round(x0 * width), round(y0 * height),
           round(x1 * width), round(y1 * height))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('transferred box collapsed')
    ratio = math.sqrt(((box[2] - box[0]) / width) * ((box[3] - box[1]) / height))
    return box, ratio


def category_effect(records, scales):
    """Mean standardized ZO-minus-CTRL crop and sham changes for one category."""
    out = {}
    for view in ('crop', 'sham'):
        differences = []
        for scores in records:
            differences.append(
                (scores[view]['zo'] - scores['original']['zo']) / scales['zo']
                - (scores[view]['ctrl'] - scores['original']['ctrl']) / scales['ctrl'])
        out[view] = statistics.mean(differences)
    out['specific'] = out['crop'] - out['sham']
    return out


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('empty percentile')
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def summarize(groups, n_boot=2000, seed=0):
    import random

    if set(groups) != set(CATEGORIES) or any(len(groups[category]) != 32 for category in CATEGORIES):
        raise ValueError('expected 32 paired normals in each of 15 MVTec categories')
    originals = {
        arm: [row['original'][arm] for category in CATEGORIES for row in groups[category]]
        for arm in ARMS
    }
    scales = {arm: statistics.pstdev(values) for arm, values in originals.items()}
    if any(not math.isfinite(scale) or scale < 0.01 for scale in scales.values()):
        raise ValueError('original-view margin SD below 0.01')
    per_category = {category: category_effect(groups[category], scales) for category in CATEGORIES}
    observed = {key: statistics.mean(per_category[category][key] for category in CATEGORIES)
                for key in ('crop', 'sham', 'specific')}
    rng = random.Random(seed)
    boot = {key: [] for key in observed}
    for _ in range(n_boot):
        sampled = {
            category: category_effect(
                [rng.choice(groups[category]) for _ in groups[category]], scales)
            for category in CATEGORIES
        }
        for key in boot:
            boot[key].append(statistics.mean(sampled[category][key] for category in CATEGORIES))
    ci = {key: [percentile(values, 2.5), percentile(values, 97.5)] for key, values in boot.items()}
    positive_categories = sum(per_category[category]['crop'] > 0 for category in CATEGORIES)
    if (observed['crop'] >= 0.20 and ci['crop'][0] > 0
            and observed['specific'] >= 0.15 and ci['specific'][0] > 0
            and positive_categories >= 10):
        verdict = 'SUPPORT_FOR_FOLLOWUP'
    elif observed['crop'] <= 0 or observed['specific'] <= 0:
        verdict = 'NOT_SUPPORTED'
    else:
        verdict = 'INCONCLUSIVE'
    return {
        'normal_score_sd': scales, 'per_category': per_category,
        'macro_effect': observed, 'bootstrap_95_ci': ci,
        'positive_crop_categories': positive_categories,
        'bootstrap_resamples': n_boot, 'bootstrap_seed': seed,
        'verdict': verdict,
    }
