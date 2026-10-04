"""Small, dependency-light statistics for EXP-014's prospective branch test."""

import hashlib

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata


def source_partition(sample_id):
    """Stable 70/15/15 split; independent of loader order."""
    bucket = int(hashlib.sha256(("EXP-014:111:" + sample_id).encode()).hexdigest()[:8], 16) % 100
    return "direction" if bucket < 70 else "fit" if bucket < 85 else "validation"


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) /
                 (positive * negative))


def hotspot(mask, baseline, scores, distance=28):
    """Freeze the top-5% baseline selection before inspecting any candidate."""
    mask = np.asarray(mask, dtype=bool)
    hot = baseline >= np.quantile(baseline, 0.95)
    far = (~mask) & (distance_transform_edt(~mask) > distance)
    selected = hot & (mask | far)
    labels = mask[selected]
    return (int(labels.sum()), int((~labels).sum()),
            {name: auc(labels, value[selected]) for name, value in scores.items()})


def gate(dataset_rows):
    """Apply the preregistered six-set gate without tuning on target labels."""
    improvements = []
    mixtures = []
    for name, rows in dataset_rows.items():
        eligible = [r for r in rows if r["hotspot_vv_only"] is not None]
        if not eligible:
            return {"pass": False, "reason": f"{name} has no eligible images"}
        gain = float(np.mean([r["hotspot_two_branch"] - r["hotspot_vv_only"] for r in eligible]))
        mix = float(np.mean([r["hotspot_two_branch"] - r["hotspot_fixed_mixture"] for r in eligible]))
        improvements.append(gain)
        mixtures.append(mix)
    return {"pass": sum(x >= 0.03 for x in improvements) >= 4 and
            min(improvements) >= -0.02 and sum(x > 0 for x in mixtures) >= 4 and
            float(np.mean(mixtures)) > 0,
            "dataset_order": list(dataset_rows), "two_branch_minus_vv_only": improvements,
            "two_branch_minus_fixed_mixture": mixtures}
