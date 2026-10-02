"""Torch-free helpers for analyze_attn_locality.py (numpy/scipy only; unit-tested in test_locality_stats.py)."""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata


def auroc(y, s):
    y = np.asarray(y, bool).ravel()
    s = np.asarray(s).ravel()
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gauss_bias(G, b):
    """[(1+G*G), (1+G*G)] additive attention-logit bias: -d^2/(2 b^2) between patch tokens (d in patch units),
    zero for the CLS row/column. b = inf -> all zeros (baseline global attention)."""
    n = 1 + G * G
    out = np.zeros((n, n), dtype=np.float32)
    if not np.isfinite(b):
        return out
    yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    pos = np.stack([yy.ravel(), xx.ravel()], 1).astype(np.float64)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    out[1:, 1:] = (-d2 / (2.0 * b * b)).astype(np.float32)
    return out


def smooth_tokens(tok, G, s):
    """Gaussian smoothing (sigma s patches) of a token grid. tok [G*G, C] -> [G*G, C]. s=0 -> unchanged."""
    if s <= 0:
        return tok
    g = tok.reshape(G, G, -1)
    g = gaussian_filter(g, sigma=(s, s, 0), mode="nearest")
    return g.reshape(G * G, -1)


def softmax(a, axis=-1):
    a = a - a.max(axis, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis, keepdims=True)


def quartile_ids(log_area):
    qs = np.quantile(log_area, [.25, .5, .75])
    return np.digitize(log_area, qs)


def verdict(delta32, delta4_q0, fsm_delta32, loc_arms, pass_min=0.010, tol=-0.003, q0_tol=-0.02,
            falsify_max=0.005, fsm_ratio=0.5):
    """
    Pre-registered decision rule.
      delta32[arm][ds]    mean paired per-image AUROC difference vs baseline, both post-smoothed with sigma=32
      delta4_q0[arm][ds]  same at sigma=4, smallest-lesion quartile only
      fsm_delta32[ds]     best feature-smoothing-control Delta32 per dataset
      loc_arms            names of the in-network locality arms
    arm passes  : Delta32 >= pass_min on >=2/3 datasets, >= tol on the rest, and Delta4(Q0) >= q0_tol on all 3.
    CONFIRM     : >=2 locality arms pass AND on >=2/3 datasets feature smoothing reaches < fsm_ratio * best-locality Delta32.
    FALSIFY     : no locality arm has Delta32 >= falsify_max on >=2/3 datasets.
    AGGREGATION_ONLY : >=2 arms pass but feature smoothing explains them (known AF-CLIP-style aggregation).
    else INCONCLUSIVE.
    """
    dss = list(next(iter(delta32.values())).keys())
    passed = []
    for a in loc_arms:
        d = np.array([delta32[a][k] for k in dss])
        q = np.array([delta4_q0[a][k] for k in dss])
        ok = (d >= pass_min).sum() >= 2 and (d >= tol).all() and (q >= q0_tol).all()
        if ok:
            passed.append(a)
    weak = [a for a in loc_arms if (np.array([delta32[a][k] for k in dss]) >= falsify_max).sum() >= 2]
    if not weak:
        return "FALSIFY", passed
    if len(passed) >= 2:
        best_loc = {k: max(delta32[a][k] for a in loc_arms) for k in dss}
        unexplained = sum(fsm_delta32[k] < fsm_ratio * best_loc[k] for k in dss)
        return ("CONFIRM" if unexplained >= 2 else "AGGREGATION_ONLY"), passed
    return "INCONCLUSIVE", passed
