"""EXP-031 statistics: is the per-image best extent coordinate z related to the lesion area, and how much could ANY area-based z rule gain?
Pure numpy/scipy. Inputs are the per-image tables of EXP-027 (per-image AUROC at each z on the grid, GT area fraction, area quartile)."""
import zlib
import numpy as np
from scipy.stats import spearmanr


def slope_stat(aucs, zs):
    """AUROC at the largest z minus AUROC at the smallest z (per image). aucs: [n, K] ordered like zs."""
    zs = np.asarray(zs, float)
    return aucs[:, int(np.argmax(zs))] - aucs[:, int(np.argmin(zs))]


def rho_boot(x, y, n_boot=1000, seed=0):
    """Spearman rho of x and y with an image-bootstrap 95% percentile interval -> (rho, lo, hi)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y)[0])
    rng = np.random.RandomState(seed)
    r = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.randint(0, len(x), len(x))
        r[b] = spearmanr(x[i], y[i])[0]
    r = r[np.isfinite(r)]
    return rho, float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def fold_of(ids):
    """Deterministic 2-fold assignment from the image id."""
    return np.array([zlib.crc32(str(i).encode()) % 2 for i in ids])


def crossfit_gain(aucs, quartile, fixed_idx, ids, groups=None):
    """Per-image AUROC of a 2-fold cross-fitted rule 'z depends only on the area quartile' minus the fixed-z AUROC.
    The z per quartile is chosen on the other fold as the grid value with the best mean AUROC (ties: lowest index).
    groups overrides `quartile` when given (used for the shuffled-label control). Returns an array of per-image gains."""
    q = np.asarray(quartile if groups is None else groups)
    fold = fold_of(ids)
    gain = np.zeros(len(q))
    for f in (0, 1):
        fit, app = fold != f, fold == f
        for g in np.unique(q):
            sel_fit = fit & (q == g)
            if not sel_fit.any():
                best = fixed_idx
            else:
                best = int(np.argmax(aucs[sel_fit].mean(0)))
            sel_app = app & (q == g)
            gain[sel_app] = aucs[sel_app, best] - aucs[sel_app, fixed_idx]
    return gain


def boot_mean(d, n_boot=2000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.RandomState(seed)
    m = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def association(rho, lo, hi, thr=0.20):
    """'POS' / 'NEG' if |rho| >= thr and the interval excludes 0, else 'NONE'."""
    if not (np.isfinite(rho) and np.isfinite(lo) and np.isfinite(hi)):
        return "NONE"
    if rho >= thr and lo > 0:
        return "POS"
    if rho <= -thr and hi < 0:
        return "NEG"
    return "NONE"


def useful(mean, lo, thr=0.005):
    return bool(np.isfinite(mean) and np.isfinite(lo) and mean >= thr and lo > 0)


def verdict(useful_flags, assoc_flags):
    """Pre-registered gate. WORTH A TRAINING STUDY needs the cross-fitted area-rule to be useful on >= 4/6 sets AND a consistent-sign association on >= 4/6
    (same sign among the sets with an association); CLOSE if useful on <= 2/6; otherwise INCONCLUSIVE."""
    nu = sum(useful_flags)
    pos, neg = sum(a == "POS" for a in assoc_flags), sum(a == "NEG" for a in assoc_flags)
    na = max(pos, neg) if min(pos, neg) == 0 else 0
    if nu >= 4 and na >= 4:
        return "WORTH A TRAINING STUDY"
    if nu <= 2:
        return "CLOSE"
    return "INCONCLUSIVE"
