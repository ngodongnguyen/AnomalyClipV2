"""EXP-036 statistics: centre-weighted multi-shift fusion (CWMS) and its controls. Pure numpy/scipy (no torch).
Rule, controls and thresholds: research/EXPERIMENTS.md EXP-036 (fixed before any run; do not tune).
Convention (as pos_shift_stats / EXP-017): shifted(y, x) = I(y - dy, x - dx); the original pixel (y, x) sits at (y + dy, x + dx) in the view,
and a score map computed on the view is brought back with pos_shift_stats.inverse_align (NaN where no source pixel exists)."""
import numpy as np
from scipy.ndimage import gaussian_filter

import pos_shift_stats as P

SHIFT_BIG, SHIFT_SMALL, SIGMA_W = 130, 14, 130.0
SIGMA_BASE, SIGMA_TARGET = 4.0, 32.0
SIGMA_EXTRA = float(np.sqrt(SIGMA_TARGET ** 2 - SIGMA_BASE ** 2))     # extra Gaussian so that the effective sigma is 32 (EXP-017)
ARMS = ("FIXED", "CWMS", "UNI5", "SMALL5", "SIGMA32")
CONTROLS = ("UNI5", "SMALL5", "SIGMA32")
SETS = ("ClinicDB", "Kvasir", "ColonDB", "ISIC", "Endo", "TN3K")

# rule constants (EXP-036)
D_PASS, D_LOSS, D_FAIL, B_PASS, P_FLOOR = 0.005, -0.005, 0.003, 0.003, -1.0
N_PASS, N_FAIL_LT, N_FAIL_LOSS = 4, 3, 2


def view_list(shift):
    """Identity plus four axis-aligned translations (dx, dy)."""
    return [(0, 0), (shift, 0), (-shift, 0), (0, shift), (0, -shift)]


CWMS_VIEWS = view_list(SHIFT_BIG)
SMALL_VIEWS = view_list(SHIFT_SMALL)


def valid_region(shape, dx, dy):
    """Original-coordinate pixels that have a source in the translated view (the rest is padding)."""
    return P.valid_mask(shape, dx, dy)


def centre_weight(shape, dx, dy, sigma_w=SIGMA_W):
    """w(p) = 0 on padded pixels, else exp(-d^2 / (2 sigma_w^2)), d = distance in the translated view from p's location (y+dy, x+dx) to the view centre."""
    h, w = shape
    yy = np.arange(h, dtype=np.float64)[:, None] + dy - (h - 1) / 2
    xx = np.arange(w, dtype=np.float64)[None, :] + dx - (w - 1) / 2
    wt = np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma_w ** 2))
    return np.where(valid_region(shape, dx, dy), wt, 0.0)


def uniform_weight(shape, dx, dy):
    return valid_region(shape, dx, dy).astype(np.float64)


def fuse(aligned, weights):
    """sum_k w_k m_k / sum_k w_k, with NaN (padded) entries contributing nothing. The caller guarantees a positive denominator (identity view is valid everywhere)."""
    num, den = 0.0, 0.0
    for m, w in zip(aligned, weights):
        num = num + w * np.nan_to_num(np.asarray(m, np.float64), nan=0.0)
        den = den + w
    if not np.all(den > 0):
        raise ValueError("non-positive fusion denominator")
    return num / den


def fuse_views(shifted_maps, views, weight_fn):
    """shifted_maps[k] = score map computed on view k (in the view's own coordinates). Returns the fused map in original coordinates, float64."""
    shape = np.asarray(shifted_maps[0]).shape
    aligned = [np.asarray(shifted_maps[0], np.float64) if (dx, dy) == (0, 0) else P.inverse_align(m, dx, dy)
               for m, (dx, dy) in zip(shifted_maps, views)]
    return fuse(aligned, [weight_fn(shape, dx, dy) for dx, dy in views])


def cwms_map(shifted_maps, views=CWMS_VIEWS, sigma_w=SIGMA_W):
    return fuse_views(shifted_maps, views, lambda s, dx, dy: centre_weight(s, dx, dy, sigma_w))


def uniform_map(shifted_maps, views):
    return fuse_views(shifted_maps, views, uniform_weight)


def sigma32_map(identity_map):
    """The identity map (already sigma 4) with an extra Gaussian of sqrt(32^2 - 4^2)."""
    return gaussian_filter(np.asarray(identity_map, np.float32), sigma=SIGMA_EXTRA)


# ------------------------------------------------------------------ statistics
def boot_mean(x, n_boot=2000, seed=0):
    """mean and image-bootstrap 95% percentile CI -> (mean, lo, hi)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    m = x[rng.randint(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def set_stats(au, n_boot=2000):
    """au = {arm: per-image AUROC array (same images, no NaN)} for all ARMS. Paired CWMS - arm and the best-control margin B."""
    a = {k: np.asarray(au[k], float) for k in ARMS}
    n = len(a["FIXED"])
    if any(len(v) != n for v in a.values()) or not all(np.isfinite(v).all() for v in a.values()):
        raise ValueError("arms must be paired and finite")
    means = {k: float(v.mean()) for k, v in a.items()}
    best = max(CONTROLS, key=lambda k: means[k])
    return dict(n=n, means=means, best_control=best, b=means["CWMS"] - means[best],
                d=boot_mean(a["CWMS"] - a["FIXED"], n_boot),
                d_vs={k: boot_mean(a["CWMS"] - a[k], n_boot) for k in CONTROLS})


def group_gain(diff, c, area_quartile=None, ids=None):
    """Report-only: mean CWMS - FIXED by lesion-offset tercile (P.terciles, ties by id string order; index order when ids is None) and by area quartile (0..3)."""
    diff = np.asarray(diff, float); c = np.asarray(c, float)
    ip, ic = P.terciles(c, ids if ids is not None else [f"{i:09d}" for i in range(len(c))])
    out = dict(peripheral=float(diff[ip].mean()), central=float(diff[ic].mean()), n_tercile=len(ip))
    if area_quartile is not None:
        q = np.asarray(area_quartile)
        out["quartile"] = [float(diff[q == k].mean()) if (q == k).any() else float("nan") for k in range(4)]
    return out


# ------------------------------------------------------------------ verdict
def _r(x):
    return round(float(x), 10)


def set_pass(d, b, p):
    """d = (mean, lo, hi) of CWMS - FIXED; b = CWMS minus best control (mean); p = AUPRO(CWMS) - AUPRO(FIXED) in points."""
    ok = all(np.isfinite(v) for v in (d[0], d[1], b, p))
    return bool(ok and _r(d[0]) >= D_PASS and _r(d[1]) > 0 and _r(b) >= B_PASS and _r(p) >= P_FLOOR)


def set_loss(d):
    return bool(np.isfinite(d[0]) and _r(d[0]) <= D_LOSS)


def set_below_fail(d):
    return bool(np.isfinite(d[0]) and _r(d[0]) < D_FAIL)


def verdict(sets, names=SETS):
    """sets[name] = dict(d=(mean, lo, hi), b=float, p=float) or None. -> (label, n_pass, n_lt003, n_loss).
    INCOMPLETE if a set is missing or any entry is non-finite; ADVANCE if n_pass >= 4 and n_loss == 0; FAIL if n_lt003 >= 3 or n_loss >= 2; else INCONCLUSIVE."""
    for s in names:
        v = sets.get(s)
        if v is None or not all(np.isfinite(x) for x in (*v["d"], v["b"], v["p"])):
            return "INCOMPLETE", 0, 0, 0
    npass = sum(set_pass(sets[s]["d"], sets[s]["b"], sets[s]["p"]) for s in names)
    nlt = sum(set_below_fail(sets[s]["d"]) for s in names)
    nloss = sum(set_loss(sets[s]["d"]) for s in names)
    if npass >= N_PASS and nloss == 0:
        return "ADVANCE", npass, nlt, nloss
    if nlt >= N_FAIL_LT or nloss >= N_FAIL_LOSS:
        return "FAIL", npass, nlt, nloss
    return "INCONCLUSIVE", npass, nlt, nloss
