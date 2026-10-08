"""Torch-free helpers for analyze_zens.py (EXP-027 / H018: headroom of the extent coordinate z and of per-image calibration).
numpy/scipy only; unit-tested in test_zens_stats.py.  Oracles (best-of-K, background-aligned) use GT and are DIAGNOSTICS, never methods.
"""
import numpy as np
from headroom_stats import signed_distance

Z_GRID = (-0.69, 0.0, 0.8, 1.6, 2.4)
FIXED_Z = 1.6
UNI3_Z = (0.8, 1.6, 2.4)
# pre-registered thresholds (0..1 AUROC scale), fixed before any run
CLOSE_BELOW, PURSUE_AT = 0.010, 0.020
MIN_DATASETS = 4
GAIN_MARGIN = 0.005
BETWEEN_AT, WITHIN_BELOW = 0.50, 0.25
EPS = 1e-12


def best_of_k(table):
    """table [N, K] per-image AUROC per z -> (best value [N], index of the winning column [N]); ties go to the lowest index."""
    t = np.asarray(table, float)
    idx = np.argmax(t, axis=1)
    return t[np.arange(len(t)), idx], idx


def win_hist(idx, k, groups=None, n_groups=4):
    """counts of winners: [K] overall, or [n_groups, K] when `groups` (0..n_groups-1 per image) is given."""
    idx = np.asarray(idx)
    if groups is None:
        return np.bincount(idx, minlength=k)
    g = np.asarray(groups)
    return np.stack([np.bincount(idx[g == q], minlength=k) for q in range(n_groups)])


def mean_maps(maps):
    return np.mean(np.stack([np.asarray(m, np.float32) for m in maps]), axis=0, dtype=np.float32)


def median_offset(m):
    """label-free per-image level: median over all pixels (MEDALIGN subtracts it). Diagnostic only."""
    return float(np.median(m))


def bg_offset(gt, m, far=28):
    """oracle per-image level: median over pixels farther than `far` px from the lesion (BGALIGN subtracts it).
    If no such pixel exists, falls back to the median over ALL background pixels. Returns (offset, used_fallback)."""
    gt = np.asarray(gt, bool)
    d = signed_distance(gt)
    sel = d > far
    fb = not sel.any()
    if fb:
        sel = d > 0
    return float(np.median(m[sel])), fb


def paired_bootstrap(diff, n_boot=2000, seed=0):
    """mean of paired differences and image-bootstrap 95% CI (percentile) -> (mean, lo, hi)."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = d[rng.randint(0, len(d), len(d))].mean()
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def closure(pooled_edit, pooled_base):
    if not (np.isfinite(pooled_edit) and np.isfinite(pooled_base)) or pooled_base >= 1.0:
        return float("nan")
    return float((pooled_edit - pooled_base) / (1.0 - pooled_base))


def verdict_rz(gaps):
    """gaps {dataset: ORACLE_BEST - FIXED mean per-image AUROC}. CLOSE if gap < 0.010 on >= 4 sets; PURSUE if >= 0.020 on >= 4; else INCONCLUSIVE.
    NaN never counts. The two cannot both hold (4 + 4 > 6)."""
    v = [g for g in gaps.values() if np.isfinite(g)]
    n_close = sum(g < CLOSE_BELOW - EPS for g in v)
    n_pursue = sum(g >= PURSUE_AT - EPS for g in v)
    if n_close >= MIN_DATASETS:
        return "CLOSE"
    if n_pursue >= MIN_DATASETS:
        return "PURSUE"
    return "INCONCLUSIVE"


def gain_loss(mean, lo, hi):
    """R_U per dataset: 'GAIN' if mean >= +0.005 and lo > 0; 'LOSS' if mean <= -0.005 and hi < 0; else 'NONE'."""
    if mean >= GAIN_MARGIN - EPS and lo > 0:
        return "GAIN"
    if mean <= -GAIN_MARGIN + EPS and hi < 0:
        return "LOSS"
    return "NONE"


def verdict_cal(closures_bg):
    """closures_bg {dataset: BGALIGN pooled-gap closure}. BETWEEN if >= 0.50 on >= 4 sets; WITHIN if < 0.25 on >= 4; else MIXED."""
    v = [c for c in closures_bg.values() if np.isfinite(c)]
    n_b = sum(c >= BETWEEN_AT - EPS for c in v)
    n_w = sum(c < WITHIN_BELOW - EPS for c in v)
    if n_b >= MIN_DATASETS:
        return "BETWEEN-IMAGE"
    if n_w >= MIN_DATASETS:
        return "WITHIN-IMAGE"
    return "MIXED"
