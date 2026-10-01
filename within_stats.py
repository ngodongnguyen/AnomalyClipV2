"""Pure-numpy core stats for analyze_within_image_concepts.py (unit-testable without torch)."""
import numpy as np
from scipy.stats import rankdata
from scipy.ndimage import label

def auroc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def within_image_auroc(y, m, img, min_each=3):
    """Mean over images of AUROC(y | m) computed INSIDE each image (needs >=min_each of both classes).
    Removes any between-image shift of m. Returns (mean, n_images_used)."""
    y = np.asarray(y, bool); m = np.asarray(m, float); img = np.asarray(img)
    out = []
    for i in np.unique(img):
        k = img == i
        if y[k].sum() >= min_each and (~y[k]).sum() >= min_each:
            out.append(auroc(y[k], m[k]))
    return (float(np.mean(out)) if out else np.nan), len(out)

def pool_components(pool, gt, dout, min_size=2, tp_frac=0.5, fp_frac=0.1, far=2):
    """Connected components (4-conn) of the pool mask on the patch grid.
    TP comp: >=tp_frac of its patches inside gt. FP comp: <=fp_frac inside gt AND every patch farther than `far`
    from gt. Others (ambiguous / boundary) dropped. Returns list of (index_array, is_fp)."""
    lab, n = label(pool)
    res = []
    for c in range(1, n + 1):
        idx = np.flatnonzero((lab == c).ravel())
        if len(idx) < min_size: continue
        fin = gt.ravel()[idx].mean()
        if fin >= tp_frac: res.append((idx, False))
        elif fin <= fp_frac and (dout.ravel()[idx] > far).all(): res.append((idx, True))
    return res

def pair_stats(comp_score, comp_margin, comp_fp):
    """Within ONE image: over all (TP comp, FP comp) pairs.
    margin_ok = P(margin_FP > margin_TP) (ties .5); score_wrong = pairs where score_FP >= score_TP;
    fixed = among score_wrong pairs, fraction margin ranks FP above TP (ties .5). Returns (margin_ok, n_pairs, fixed, n_wrong)."""
    s = np.asarray(comp_score, float); m = np.asarray(comp_margin, float); fp = np.asarray(comp_fp, bool)
    tp_i = np.flatnonzero(~fp); fp_i = np.flatnonzero(fp)
    if len(tp_i) == 0 or len(fp_i) == 0: return None
    ok, fixed = [], []
    for a in tp_i:
        for b in fp_i:
            v = 1.0 if m[b] > m[a] else (0.5 if m[b] == m[a] else 0.0); ok.append(v)
            if s[b] >= s[a]: fixed.append(v)
    return float(np.mean(ok)), len(ok), (float(np.mean(fixed)) if fixed else np.nan), len(fixed)
