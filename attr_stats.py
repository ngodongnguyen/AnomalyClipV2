"""EXP-047 statistics: lesion attributes (solidity, boundary-contrast magnitude, interior inhomogeneity) and the A/B/C gate. Pure numpy/scipy.
Rules are fixed in research/EXPERIMENTS.md (EXP-047). Comparisons are made after rounding to 1e-10 so that exact boundary values (0.80, 0.20, 4/6) behave as written."""
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion
import zpos_stats as Z

ATTRS = ["solidity", "bcm", "inhom"]
NEED = 4                       # 4 of 6 sets
R = lambda x: round(float(x), 10)


# ------------------------------------------------------------------ attributes
def solidity(mask):
    """mask pixels / area of the convex hull of the mask (all mask pixels, not only the largest component).
    The hull is built on the corners of the boundary pixels (x +- 0.5, y +- 0.5) so that a filled disk gives ~1 rather than > 1.
    NaN if the mask is empty or the hull is degenerate (Qhull error / zero area). Capped at 1."""
    m = np.asarray(mask, bool)
    n = int(m.sum())
    if n == 0:
        return float("nan")
    edge = m & ~binary_erosion(m, border_value=0)
    ys, xs = np.nonzero(edge)
    off = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    pts = (np.c_[ys, xs][:, None, :] + off[None]).reshape(-1, 2)
    try:
        area = ConvexHull(pts).volume          # 2-D: volume is the area
    except Exception:
        return float("nan")
    if not np.isfinite(area) or area <= 0:
        return float("nan")
    return float(min(1.0, n / area))


def boundary_contrast_mag(v, mask, width=28):
    """|mean V inside - mean V in the `width` px outside ring| (NaN for an empty ring)."""
    return abs(Z.ring_contrast(v, mask, width))


def inhomogeneity(v, mask):
    """std of V inside the mask / (std of V over the whole image + 1e-6)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return float("nan")
    return float(np.std(v[m]) / (np.std(v) + 1e-6))


# ------------------------------------------------------------------ partial correlation with several controls
def partial_spearman_multi(x, y, *controls):
    """Spearman of x and y after removing the linear dependence of the ranks of both on the ranks of every control."""
    rx, ry = rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float))
    cols = [np.ones(len(rx))] + [rankdata(np.asarray(c, float)) for c in controls]
    A = np.c_[tuple(cols)]
    def resid(r):
        beta, *_ = np.linalg.lstsq(A, r, rcond=None)
        return r - A @ beta
    a, b = resid(rx), resid(ry)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else float("nan")


def boot_partial(attr, auc, area, c, n_boot=1000, seed=0):
    """partial Spearman(attr, auc | log area, c) with image-bootstrap 95% CI -> (rho, lo, hi)."""
    return Z.boot_stat(lambda a, y, la, cc: partial_spearman_multi(a, y, la, cc), [attr, auc, np.log(area), c], n_boot=n_boot, seed=seed)


# ------------------------------------------------------------------ checks
def support_fraction(mv_vals, set_vals, lo_q=5, hi_q=95):
    """A: fraction of the set's finite values inside the [5th, 95th] percentile range of the MVTec values (inclusive)."""
    mv = np.asarray(mv_vals, float); mv = mv[np.isfinite(mv)]
    s = np.asarray(set_vals, float); s = s[np.isfinite(s)]
    if len(mv) == 0 or len(s) == 0:
        return float("nan")
    lo, hi = np.percentile(mv, [lo_q, hi_q])
    return float(np.mean((s >= lo) & (s <= hi)))


def check_A(fracs, thr=0.80, need=NEED):
    """fracs: per-set support fractions. Pass if >= need sets have fraction >= thr (exactly 0.80 counts)."""
    k = sum(np.isfinite(f) and R(f) >= thr for f in fracs)
    return k >= need, int(k)


def area_corr(attr, area):
    a = np.asarray(attr, float); ar = np.asarray(area, float)
    ok = np.isfinite(a) & np.isfinite(ar) & (ar > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(a[ok], np.log(ar[ok]))[0])


def check_B(rho_mvtec, rho_sets, thr=0.80, need=NEED):
    """B: |Spearman(attr, log area)| <= thr on MVTec AND on >= need medical sets (exactly 0.80 passes)."""
    ok_mv = np.isfinite(rho_mvtec) and R(abs(rho_mvtec)) <= thr
    k = sum(np.isfinite(r) and R(abs(r)) <= thr for r in rho_sets)
    return bool(ok_mv and k >= need), int(k), bool(ok_mv)


def set_pass_C(stat, thr=0.20):
    """stat = (rho, lo, hi): |rho| >= thr and CI excludes 0 -> sign (+1/-1) else 0."""
    rho, lo, hi = stat
    if not (np.isfinite(rho) and np.isfinite(lo) and np.isfinite(hi)):
        return 0
    if R(abs(rho)) >= thr and (R(lo) > 0 or R(hi) < 0):
        return 1 if rho > 0 else -1
    return 0


def check_C(stats, thr=0.20, need=NEED):
    """C: >= need sets pass with the SAME sign. Returns (pass, n_pass_pos, n_pass_neg)."""
    s = [set_pass_C(t, thr) for t in stats]
    npos, nneg = sum(x == 1 for x in s), sum(x == -1 for x in s)
    return (npos >= need or nneg >= need), int(npos), int(nneg)


def verdict_attribute(a_ok, b_ok, c_ok):
    return "CANDIDATE" if (a_ok and b_ok and c_ok) else "DROPPED"


def verdict_direction(per_attr):
    """per_attr: dict name -> 'CANDIDATE'/'DROPPED'. CLOSE if none is a candidate."""
    c = [k for k, v in per_attr.items() if v == "CANDIDATE"]
    return ("CANDIDATES: " + ", ".join(c)) if c else "CLOSE"
