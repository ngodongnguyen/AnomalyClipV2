"""EXP-033 statistics: lesion position and polarity versus per-image AUROC. Pure numpy/scipy."""
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.ndimage import binary_dilation, center_of_mass


def centroid_offset(mask, size=None):
    """Distance of the mask centroid from the canvas centre divided by half the side (0 centre, ~1.41 corner)."""
    m = np.asarray(mask, bool)
    cy, cx = center_of_mass(m)
    h, w = m.shape
    return float(np.hypot(cy - (h - 1) / 2, cx - (w - 1) / 2) / ((min(h, w) - 1) / 2))


def ring_contrast(v, mask, width=28):
    """mean V inside the mask minus mean V in the ring of pixels within `width` px outside it (NaN when the ring is empty)."""
    m = np.asarray(mask, bool)
    ring = binary_dilation(m, iterations=width) & ~m
    if not ring.any() or not m.any():
        return float("nan")
    return float(v[m].mean() - v[ring].mean())


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing the linear (rank) dependence of both on z."""
    rx, ry, rz = (rankdata(np.asarray(a, float)) for a in (x, y, z))
    def resid(r):
        A = np.c_[np.ones_like(rz), rz]
        beta, *_ = np.linalg.lstsq(A, r, rcond=None)
        return r - A @ beta
    a, b = resid(rx), resid(ry)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def boot_stat(fn, arrays, n_boot=1000, seed=0):
    """fn(*arrays) with an image-bootstrap 95% percentile interval -> (value, lo, hi)."""
    arrays = [np.asarray(a, float) for a in arrays]
    n = len(arrays[0])
    val = float(fn(*arrays))
    rng = np.random.RandomState(seed)
    r = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        v = fn(*[a[i] for a in arrays])
        if np.isfinite(v):
            r.append(v)
    r = np.asarray(r)
    return val, float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def group_diff(auc, flag):
    """mean AUROC of flagged minus unflagged images (NaN if a group is empty)."""
    auc, flag = np.asarray(auc, float), np.asarray(flag, bool)
    if flag.sum() == 0 or (~flag).sum() == 0:
        return float("nan")
    return float(auc[flag].mean() - auc[~flag].mean())


def verdict_periphery(stats, strong=-0.20, weak=-0.10):
    """stats: list of (rho, lo, hi) on the decision sets. SUPPORTED if rho <= strong with hi < 0 on >= 3/4 (scaled for other n); NOT SUPPORTED if rho <= weak on <= 1/4."""
    n = len(stats)
    need, low = int(np.ceil(0.75 * n)), int(np.floor(0.25 * n))
    ok = sum(np.isfinite(r) and r <= strong and h < 0 for r, l, h in stats)
    wk = sum(np.isfinite(r) and r <= weak for r, l, h in stats)
    return "SUPPORTED" if ok >= need else "NOT SUPPORTED" if wk <= low else "INCONCLUSIVE"


def verdict_dark(stats, n_groups, strong=-0.03, weak=-0.015, min_group=20):
    """stats: list of (diff, lo, hi); n_groups: list of (n_dark, n_nondark). Groups smaller than min_group make the set not count as support."""
    n = len(stats)
    need, low = int(np.ceil(0.75 * n)), int(np.floor(0.25 * n))
    ok = sum(np.isfinite(d) and d <= strong and h < 0 and min(g) >= min_group for (d, l, h), g in zip(stats, n_groups))
    wk = sum(np.isfinite(d) and d <= weak for d, l, h in stats)
    return "SUPPORTED" if ok >= need else "NOT SUPPORTED" if wk <= low else "INCONCLUSIVE"
