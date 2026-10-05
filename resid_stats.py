"""Torch-free statistics for analyze_resid.py (EXP-019 / H010, ClearCLIP-style residual removal on the V-V stream).
numpy/scipy only; unit-tested in test_resid_stats.py.  auroc / paired_bootstrap_ci are reused from posart_stats (EXP-018)."""
import numpy as np
from posart_stats import auroc, paired_bootstrap_ci  # noqa: F401  (re-exported)

TEST_ARMS = ("no_res", "no_res_last")   # the two candidate arms (multiplicity = 2)


def swap_pairing(n, seed):
    """partner[i] != i for every i (cyclic shift of a seeded permutation), bijective. Label-independent."""
    if n < 2:
        raise ValueError("need >= 2 images to pair")
    p = np.random.RandomState(seed).permutation(n)
    partner = np.empty(n, dtype=int)
    partner[p] = np.roll(p, -1)
    return partner


def recon_max_abs_diff(x5, attn_list, x_final):
    """max |(x5 + sum_l attn_l) - x_final|, float64 sum. Used for the decomposition assert (pre-ln_post tokens)."""
    s = np.asarray(x5, np.float64).copy()
    for a in attn_list:
        s = s + np.asarray(a, np.float64)
    return float(np.abs(s - np.asarray(x_final, np.float64)).max())


def norm_ratio(x5, attn_sum):
    """mean over tokens of ||x5(u)|| / ||attn_sum(u)||  (tokens [P, C])."""
    n5 = np.linalg.norm(np.asarray(x5, np.float64), axis=-1)
    ns = np.linalg.norm(np.asarray(attn_sum, np.float64), axis=-1)
    return float((n5 / (ns + 1e-12)).mean())


def quartile_means(log_area, delta):
    """Mean delta in the 4 quantile bins of log_area (small -> large lesion). Report-only."""
    la = np.asarray(log_area, float); d = np.asarray(delta, float)
    edges = np.quantile(la, [0.25, 0.5, 0.75])
    q = np.digitize(la, edges)  # 0..3
    return [float(np.nanmean(d[q == k])) if (q == k).any() else float("nan") for k in range(4)]


def arm_passes(d32, ci_lo, gap, gain=0.005, tol=-0.005):
    """d32 / ci_lo / gap: dicts keyed by dataset for ONE arm (gap = arm - swap_res @sigma32).
    pass: >=2/3 of sets good (d>=gain, CI lower>0, gap>=gain) AND no set with d<=tol."""
    ds = list(d32)
    good = [k for k in ds if d32[k] >= gain and ci_lo[k] > 0 and gap[k] >= gain]
    need = int(np.ceil(2.0 * len(ds) / 3.0))
    return len(good) >= need and all(d32[k] > tol for k in ds)


def verdict(d32, ci_lo, gap, kill=0.003):
    """Pre-registered EXP-019 rule. Each argument is {arm: {dataset: value}} for arms in TEST_ARMS.
      d32[arm][ds]   : mean paired per-image AUROC(arm@32) - AUROC(base@32)
      ci_lo[arm][ds] : lower 95% paired-bootstrap bound of that delta
      gap[arm][ds]   : mean paired AUROC(arm@32) - AUROC(swap_res@32)
    CONFIRM : at least one of the two arms passes (arm_passes).
    FALSIFY : neither arm has d32 >= kill on >= 2/3 of the sets.
    else INCONCLUSIVE."""
    passed = [a for a in TEST_ARMS if arm_passes(d32[a], ci_lo[a], gap[a])]
    if passed:
        return ("CONFIRM via " + " + ".join(passed) + " (2 candidate arms: multiplicity; needs held-out + pooled AUROC/PRO "
                "re-confirmation before any claim)")
    n = len(next(iter(d32.values())))
    need = int(np.ceil(2.0 * n / 3.0))
    if all(sum(v >= kill for v in d32[a].values()) < need for a in TEST_ARMS):
        return "FALSIFY (neither arm reaches delta32 >= +0.003 on >= 2/3 sets)"
    return "INCONCLUSIVE"
