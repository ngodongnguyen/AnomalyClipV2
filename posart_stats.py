"""Torch-free statistics for analyze_posart.py (EXP-018, position-artifact diagnostic). numpy/scipy only; unit-tested in test_posart_stats.py."""
import numpy as np
from scipy.stats import rankdata


def auroc(y, s):
    y = np.asarray(y, bool).ravel(); s = np.asarray(s).ravel()
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def estimate_artifact(token_sum, count):
    """Position-dependent field G [P, C] from a running sum of tokens [P, C] over `count` images.
    Mean over images at each position, then CENTERED over positions per channel (so a global content/domain offset
    shared by all positions is NOT removed -- only the position-specific part)."""
    m = token_sum / float(count)
    return m - m.mean(0, keepdims=True)


def split_half_cosine(Ga, Gb):
    """Cosine between two independently estimated centered fields (flattened). ~0 if there is no stable pattern."""
    a = (Ga - Ga.mean(0, keepdims=True)).ravel(); b = (Gb - Gb.mean(0, keepdims=True)).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def energy_ratio(G, tok_var):
    """mean_u ||G(u)||^2 / total token variance (sum over channels of per-channel variance of tokens around grand mean)."""
    return float((G ** 2).sum(1).mean() / tok_var)


def cosine(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def split_halves(n, seed=0):
    """Random disjoint index halves (A, B) of range(n), |A| = ceil(n/2)."""
    p = np.random.RandomState(seed).permutation(n)
    h = (n + 1) // 2
    return np.sort(p[:h]), np.sort(p[h:])


def sample_crop(rng, W, H, scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3)):
    """Random-resized-crop box (x0, y0, w, h), inside the W x H image; area fraction ~ U(scale), log-uniform aspect."""
    for _ in range(20):
        area = W * H * rng.uniform(*scale)
        r = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
        w, h = int(round(np.sqrt(area * r))), int(round(np.sqrt(area / r)))
        if 0 < w <= W and 0 < h <= H:
            return int(rng.randint(0, W - w + 1)), int(rng.randint(0, H - h + 1)), w, h
    return 0, 0, W, H


def permute_positions(G, rng):
    return G[rng.permutation(G.shape[0])]


def subtract_artifact(tok, G):
    """tok [P, C] raw (un-normalised) tokens; returns tok - G."""
    return tok - G


def paired_bootstrap_ci(delta, n_boot=2000, seed=0):
    d = np.asarray(delta, float); d = d[~np.isnan(d)]
    rng = np.random.RandomState(seed)
    means = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(split_half_cos, d32_art, d32_perm_gap, ci_lo32, tol=-0.005, gain=0.005, kill_cos=0.30):
    """
    Pre-registered rule. Inputs are dicts keyed by dataset (3 CVC sets) except split_half_cos (scalar, source vs source).
      split_half_cos : split-half cosine of the source-estimated field G (stage-1 existence gate)
      d32_art[ds]    : mean paired per-image AUROC(art_src @ sigma32) - AUROC(base @ sigma32)
      d32_perm_gap[ds]: mean paired AUROC(art_src @32) - AUROC(perm_src @32)   (specificity control)
      ci_lo32[ds]    : lower 95% paired-bootstrap bound of the d32_art
    CONFIRM : stage-1 passes AND >=2/3 sets with d32_art>=gain and ci_lo>0 AND those sets also perm_gap>=gain AND no set d32_art<=tol
    FALSIFY : split_half_cos<kill_cos, OR fewer than 2 sets with d32_art>=0.003
    else INCONCLUSIVE
    """
    ds = list(d32_art)
    if split_half_cos < kill_cos:
        return "FALSIFY (no stable position pattern)"
    if sum(d32_art[k] >= 0.003 for k in ds) < 2:
        return "FALSIFY (no gain over sigma32 baseline)"
    good = [k for k in ds if d32_art[k] >= gain and ci_lo32[k] > 0 and d32_perm_gap[k] >= gain]
    if len(good) >= 2 and all(d32_art[k] > tol for k in ds):
        return "CONFIRM"
    return "INCONCLUSIVE"
