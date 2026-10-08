"""Torch-free helpers for analyze_coldcore.py (EXP-026 / H017: why do large lesions have cold cores in ECP?).
numpy/scipy only; unit-tested in test_core_stats.py.  GT is used ONLY to define patch zones for a diagnostic; no method is implied.

Patch zones on the native token grid (37x37 at 518 input, 14 px patches), from the GT mask average-pooled onto the grid (frac = lesion fraction):
  LESION = frac >= 0.9      BG = frac == 0
  CORE   = lesion patches whose 8 neighbours are ALL lesion patches (erosion by one patch; out-of-grid neighbours count as NOT lesion,
           so a lesion patch on the image border is never CORE)
  RIM_IN = lesion & ~CORE
  BG_NEAR= BG patches within 2 patches (Chebyshev, 28 px) of any patch that contains lesion pixels (frac > 0; conservative, so partial
           patches also push background away); BG_FAR = the other BG patches.
"""
import math
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.stats import rankdata

LESION_FRAC = 0.9
NEAR = 2
MIN_CORE, MIN_RIM, MIN_BGFAR = 4, 4, 20
MIN_IMAGES = 30
MIN_DATASETS = 4
VIS_RHO, READ_RHO = -0.30, -0.15
S3 = np.ones((3, 3), bool)


def patch_fraction(gt, grid):
    """average-pool a binary [H, W] mask onto a grid x grid layout (H, W must be multiples of grid)."""
    g = np.asarray(gt, float)
    h, w = g.shape
    if h % grid or w % grid:
        raise ValueError("mask size must be a multiple of the grid")
    return g.reshape(grid, h // grid, grid, w // grid).mean(axis=(1, 3))


def patch_zones(frac, near=NEAR):
    """dict of boolean [grid, grid] masks: lesion, core, rim_in, bg, bg_near, bg_far."""
    frac = np.asarray(frac, float)
    lesion = frac >= LESION_FRAC
    bg = frac == 0
    core = binary_erosion(lesion, structure=S3, border_value=0)
    rim = lesion & ~core
    touched = frac > 0
    nearmask = binary_dilation(touched, structure=S3, iterations=near, border_value=0)
    return {"lesion": lesion, "core": core, "rim_in": rim, "bg": bg, "bg_near": bg & nearmask, "bg_far": bg & ~nearmask}


def zone_counts(z):
    return {k: int(z[k].sum()) for k in ("core", "rim_in", "bg_far")}


def is_eligible(z):
    c = zone_counts(z)
    return c["core"] >= MIN_CORE and c["rim_in"] >= MIN_RIM and c["bg_far"] >= MIN_BGFAR


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def prototypes(tokens, z):
    """tokens [P, C] (already L2-normalised per token); returns {zone: unit mean token} for core, rim_in, bg_far."""
    tokens = np.asarray(tokens, float)
    flat = {k: v.reshape(-1) for k, v in z.items()}
    return {k: _unit(tokens[flat[k]].mean(0)) for k in ("core", "rim_in", "bg_far")}


def visual_ambiguity(proto):
    """V = cos(core, rim_in) - cos(core, bg_far); also returns both cosines."""
    c_r = float(proto["core"] @ proto["rim_in"])
    c_b = float(proto["core"] @ proto["bg_far"])
    return c_r - c_b, c_r, c_b


def f_bg(tokens, z, proto):
    """fraction of CORE patches whose nearest zone prototype (cosine, among rim_in and bg_far) is bg_far (ties go to rim_in)."""
    t = np.asarray(tokens, float)[z["core"].reshape(-1)]
    return float(np.mean(t @ proto["bg_far"] > t @ proto["rim_in"]))


def coldness(s, z):
    """delta = mean s over RIM_IN minus mean s over CORE (positive = cold core). s is [grid, grid] or flat."""
    s = np.asarray(s, float).reshape(z["core"].shape)
    return float(s[z["rim_in"]].mean() - s[z["core"]].mean())


def zone_means(s, z):
    s = np.asarray(s, float).reshape(z["core"].shape)
    return {k: float(s[z[k]].mean()) for k in ("core", "rim_in", "bg_far")}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def partial_spearman(a, b, x):
    """Spearman of a and b after removing the linear dependence of their ranks on the rank of x (residual-based partial rank correlation)."""
    a, b, x = (rankdata(np.asarray(v, float)) for v in (a, b, x))
    if len(a) < 4 or np.ptp(x) == 0:
        return float("nan")
    X = np.c_[np.ones(len(x)), x]
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    if np.ptp(ra) < 1e-12 or np.ptp(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def top_tercile(delta):
    """indices of the ceil(n/3) largest delta (stable order for ties)."""
    d = np.asarray(delta, float)
    k = max(1, math.ceil(len(d) / 3))
    return np.argsort(-d, kind="stable")[:k]


def perm_null(delta, v, n_perm=1000, seed=0):
    """rho, empirical two-sided p = (1 + #{|rho_perm| >= |rho|}) / (1 + n_perm), null 2.5/97.5% quantiles. delta is shuffled across images."""
    delta, v = np.asarray(delta, float), np.asarray(v, float)
    rho = spearman(delta, v)
    if not np.isfinite(rho):
        return rho, float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    rv = rankdata(v); rd = rankdata(delta)
    nul = np.array([np.corrcoef(rd[rng.permutation(len(rd))], rv)[0, 1] for _ in range(n_perm)])
    p = (1 + int((np.abs(nul) >= abs(rho) - 1e-12).sum())) / (1 + n_perm)
    return rho, float(p), float(np.quantile(nul, 0.025)), float(np.quantile(nul, 0.975))


def dataset_summary(delta, v, ncore, n_perm=1000, seed=0):
    """per-dataset numbers used by the rule (inputs: arrays over the ELIGIBLE Q4 images)."""
    delta, v, ncore = np.asarray(delta, float), np.asarray(v, float), np.asarray(ncore, float)
    n = len(delta)
    out = {"n": n, "eligible": n >= MIN_IMAGES}
    if n < 3:
        out.update(rho=float("nan"), vmed_cold=float("nan"), p=float("nan"), q025=float("nan"), q975=float("nan"), rho_partial=float("nan"))
        return out
    rho, p, lo, hi = perm_null(delta, v, n_perm, seed)
    cold = top_tercile(delta)
    out.update(rho=rho, p=p, q025=lo, q975=hi, vmed_cold=float(np.median(v[cold])),
               rho_partial=partial_spearman(delta, v, np.log(ncore)), delta_mean=float(delta.mean()), v_mean=float(v.mean()))
    return out


def verdict(summaries):
    """summaries: {dataset: dict(eligible, rho, vmed_cold)}.  Rule fixed in EXP-026.
    VISUAL  : >= 4 eligible datasets with rho <= -0.30 AND vmed_cold < 0.
    READOUT : >= 4 eligible datasets with vmed_cold > 0 AND rho > -0.15.
    fewer than 4 eligible datasets -> INCONCLUSIVE; otherwise MIXED/INCONCLUSIVE.  NaN never passes."""
    el = {k: s for k, s in summaries.items() if s.get("eligible")}
    if len(el) < MIN_DATASETS:
        return "INCONCLUSIVE", f"only {len(el)} eligible datasets (< {MIN_DATASETS})"
    vis = [k for k, s in el.items() if s["rho"] <= VIS_RHO and s["vmed_cold"] < 0]
    rea = [k for k, s in el.items() if s["vmed_cold"] > 0 and s["rho"] > READ_RHO]
    if len(vis) >= MIN_DATASETS:
        return "VISUAL", f"{len(vis)} datasets pass: {sorted(vis)}"
    if len(rea) >= MIN_DATASETS:
        return "READOUT", f"{len(rea)} datasets pass: {sorted(rea)}"
    return "MIXED/INCONCLUSIVE", f"visual passes {len(vis)} {sorted(vis)}, readout passes {len(rea)} {sorted(rea)}"
