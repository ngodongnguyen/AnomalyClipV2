"""EXP-034 statistics: translation intervention on lesion position. Pure numpy/scipy (no torch).
Convention (as EXP-017 translated_reflect): shifted(y, x) = I(y - dy, x - dx); the original pixel (y, x) sits at (y + dy, x + dx) in the shifted image."""
import hashlib
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.ndimage import center_of_mass

MIN_CLASS = 20
D_SUPPORT, D_NULL, D_SUPPORT_N, D_NULL_N = 0.010, 0.003, 3, 3
REC_SUPPORT = 0.005          # EXP-035: recentre-minus-sham gain required in addition to the delta test


def reflect_index(i, n):
    """Reflect (edge not repeated) any integer index into [0, n); same as np.pad/F.pad mode 'reflect' for |pad| < n, periodic beyond."""
    i = np.asarray(i, np.int64)
    if n == 1:
        return np.zeros_like(i)
    period = 2 * (n - 1)
    m = np.mod(i, period)
    return np.where(m < n, m, period - m)


def shift_indices(n, d):
    """Source index for every output index along one axis: out[k] = in[reflect(k - d)]."""
    return reflect_index(np.arange(n) - int(d), n)


def shift_image(img, dx, dy):
    """Translate an (..., H, W) array by (dx, dy) with reflection padding (numpy reference of the torch gather used on the server)."""
    img = np.asarray(img)
    h, w = img.shape[-2:]
    return img[..., shift_indices(h, dy), :][..., :, shift_indices(w, dx)]


def shift_image_fill(img, dx, dy, fill):
    """Translate an (..., H, W) array by (dx, dy) and fill every uncovered pixel with `fill` (a scalar or an array broadcastable to (..., 1, 1)); no mirrored content.
    out[y + dy, x + dx] = in[y, x] on the valid rectangle (EXP-035; numpy reference of the torch slice-copy used on the server)."""
    img = np.asarray(img)
    h, w = img.shape[-2:]
    out = np.broadcast_to(np.asarray(fill, img.dtype), img.shape).copy()
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y1 > y0 and x1 > x0:
        out[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx] = img[..., y0:y1, x0:x1]
    return out


def valid_mask(shape, dx, dy):
    """Original-coordinate pixels whose shifted location (y+dy, x+dx) lies inside the canvas."""
    h, w = shape
    v = np.zeros((h, w), bool)
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y1 > y0 and x1 > x0:
        v[y0:y1, x0:x1] = True
    return v


def common_valid(shape, shifts):
    v = np.ones(shape, bool)
    for dx, dy in shifts:
        v &= valid_mask(shape, dx, dy)
    return v


def inverse_align(shifted_map, dx, dy):
    """Map a score map computed on the shifted image back to original coordinates; pixels without a source are NaN (same as EXP-017)."""
    m = np.asarray(shifted_map)
    h, w = m.shape
    out = np.full((h, w), np.nan, np.float64)
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = m[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def centroid(mask):
    return center_of_mass(np.asarray(mask, bool))      # (cy, cx)


def recentre_displacement(mask):
    """Integer (dx, dy) that moves the mask centroid to the canvas centre ((S-1)/2 on each axis)."""
    cy, cx = centroid(mask)
    h, w = np.asarray(mask).shape
    return int(np.rint((w - 1) / 2 - cx)), int(np.rint((h - 1) / 2 - cy))


def random_displacement(sample_id, dx, dy):
    """Same integer magnitude as (dx, dy), direction angle drawn with RandomState seeded from the id (deterministic)."""
    seed = int(hashlib.sha256(("EXP-034:" + sample_id).encode()).hexdigest()[:8], 16)
    ang = np.random.RandomState(seed).uniform(0, 2 * np.pi)
    mag = float(np.hypot(dx, dy))
    return int(np.rint(mag * np.cos(ang))), int(np.rint(mag * np.sin(ang)))


def region_auroc(labels, scores, region, min_class=MIN_CLASS):
    """Exact rank-based AUROC (ties average) over pixels in `region`; None if either class has fewer than min_class pixels or a score is non-finite."""
    r = np.asarray(region, bool)
    y = np.asarray(labels, bool)[r]
    s = np.asarray(scores, float)[r]
    npos = int(y.sum()); nneg = y.size - npos
    if npos < min_class or nneg < min_class:
        return None
    if not np.all(np.isfinite(s)):
        raise ValueError("non-finite score inside the region")
    rk = rankdata(s, method="average")
    return float((rk[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def region_counts(labels, region):
    r = np.asarray(region, bool)
    y = np.asarray(labels, bool)
    return int((y & r).sum()), int((~y & r).sum())


def hit_rate(labels, scores, region, thr):
    r = np.asarray(region, bool) & np.asarray(labels, bool)
    if not r.any():
        return float("nan")
    return float((np.asarray(scores, float)[r] >= thr).mean())


def terciles(c, ids):
    """(peripheral, central) index arrays of size ceil(n/3): largest / smallest c, ties by id string order."""
    c = np.asarray(c, float)
    n = len(c)
    k = int(np.ceil(n / 3))
    rank_id = np.argsort(np.argsort(np.asarray(ids, dtype=object), kind="stable"), kind="stable")
    order_p = np.lexsort((rank_id, -c))
    order_c = np.lexsort((rank_id, c))
    return order_p[:k], order_c[:k]


def boot_mean(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def set_summary(au_sham, au_rec, au_rand, mag, n_boot=2000):
    """Per-set statistics over KEPT images (arrays of equal length; no NaN)."""
    a0, a1, a2 = (np.asarray(v, float) for v in (au_sham, au_rec, au_rand))
    delta = a1 - a2
    out = dict(n=len(delta), delta=boot_mean(delta, n_boot), rec_minus_sham=boot_mean(a1 - a0, n_boot), rand_minus_sham=boot_mean(a2 - a0, n_boot))
    if len(delta) >= 3 and np.ptp(mag) > 0 and np.ptp(delta) > 0:
        out["rho_mag_delta"] = float(spearmanr(mag, delta)[0])
    else:
        out["rho_mag_delta"] = float("nan")
    return out


def pass_support(mean, lo):
    return bool(np.isfinite(mean) and np.isfinite(lo) and round(mean, 10) >= D_SUPPORT and round(lo, 10) > 0)


def pass_null(mean, hi):
    return bool(np.isfinite(mean) and np.isfinite(hi) and (round(mean, 10) <= D_NULL or round(hi, 10) < D_SUPPORT))


def verdict(stats_by_set, decision=("ClinicDB", "ColonDB", "ISIC", "Endo")):
    """stats_by_set[set] = (mean, lo, hi) of delta, or None / NaN entries when the set has no kept images. -> (label, n_support, n_null)."""
    if any(s not in stats_by_set or stats_by_set[s] is None or not np.isfinite(stats_by_set[s][0]) for s in decision):
        return "INCOMPLETE", 0, 0
    ns = sum(pass_support(m, lo) for m, lo, hi in (stats_by_set[s] for s in decision))
    nn = sum(pass_null(m, hi) for m, lo, hi in (stats_by_set[s] for s in decision))
    if ns >= D_SUPPORT_N:
        return "POSITION-RESPONSE SUPPORTED", ns, nn
    if nn >= D_NULL_N:
        return "NOT SUPPORTED", ns, nn
    return "INCONCLUSIVE", ns, nn


def rec_ok(mean, lo):
    """EXP-035: the recentre gain over SHAM is >= +0.005 with CI lower bound > 0."""
    return bool(np.isfinite(mean) and np.isfinite(lo) and round(mean, 10) >= REC_SUPPORT and round(lo, 10) > 0)


def verdict_artifact_free(sets, decision=("ClinicDB", "ColonDB", "ISIC", "Endo")):
    """EXP-035. sets[set] = dict(delta=(mean, lo, hi), rec_minus_sham=(mean, lo, hi)) (or None when the set has no kept images).
    -> (label, n_delta_support, n_rec_ok, n_null).  ARTIFACT-FREE SUPPORT needs PASS_S on >= 3/4 AND REC_OK on >= 3/4; ARTIFACT SUSPECTED if PASS_N on >= 3/4."""
    for s in decision:
        v = sets.get(s)
        if v is None or not np.isfinite(v["delta"][0]) or not np.isfinite(v["rec_minus_sham"][0]):
            return "INCOMPLETE", 0, 0, 0
    ns = sum(pass_support(sets[s]["delta"][0], sets[s]["delta"][1]) for s in decision)
    nr = sum(rec_ok(sets[s]["rec_minus_sham"][0], sets[s]["rec_minus_sham"][1]) for s in decision)
    nn = sum(pass_null(sets[s]["delta"][0], sets[s]["delta"][2]) for s in decision)
    if ns >= D_SUPPORT_N and nr >= D_SUPPORT_N:
        return "ARTIFACT-FREE SUPPORT", ns, nr, nn
    if nn >= D_NULL_N:
        return "ARTIFACT SUSPECTED", ns, nr, nn
    return "INCONCLUSIVE", ns, nr, nn
