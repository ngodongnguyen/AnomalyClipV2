"""Torch-free statistics for analyze_layer_probe.py (numpy/scipy only, unit-testable anywhere)."""
import numpy as np
from scipy.stats import rankdata


def auroc(y, s):
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_split(y, rng, min_per_class=8):
    """Stratified half/half split of token indices, reused for every layer of one image."""
    pos, neg = np.where(y)[0], np.where(~y)[0]
    if len(pos) < min_per_class or len(neg) < min_per_class:
        return None
    pos, neg = rng.permutation(pos), rng.permutation(neg)
    a = np.concatenate([pos[: len(pos) // 2], neg[: len(neg) // 2]])
    b = np.concatenate([pos[len(pos) // 2:], neg[len(neg) // 2:]])
    return a, b


def crossfit_probe_auc(feats, y, split):
    """
    Cross-fitted mean-difference probe: direction = mean(lesion) - mean(background) estimated on one
    half of the image's tokens, AUROC measured on the other half (and vice versa). Uses the GT of the
    image itself, so it is an UPPER BOUND on what a perfect readout could do with these features --
    not something available at test time.
    """
    a, b = split
    aucs = []
    for tr, te in ((a, b), (b, a)):
        ytr = y[tr]
        d = feats[tr][ytr].mean(0) - feats[tr][~ytr].mean(0)
        aucs.append(auroc(y[te], feats[te] @ d))
    return float(np.nanmean(aucs))


def salience_auc(feats, y):
    """Unsupervised: how different is each token from the image's typical (median) token?"""
    ref = np.median(feats, axis=0)
    ref = ref / (np.linalg.norm(ref) + 1e-8)
    return auroc(y, 1.0 - feats @ ref)


def _qmean(arr, qid, k):
    v = arr[qid == k]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def print_layer_report(name, res, readout, area_frac, report_layers):
    """
    res: dict metric_name -> array [n_images, n_layers] (NaN allowed); readout: [n_images] AUROC of the
    trained-prompt readout at the last layer; area_frac: [n_images] lesion / valid area.
    """
    n = len(area_frac)
    qid = np.digitize(area_frac, np.quantile(area_frac, [.25, .5, .75]))
    print(f"\n##### Layer-wise lesion/background separability: {name} (n={n}; Q1 smallest lesion ... Q4 largest) #####")
    for metric, arr in res.items():
        print(f"\n--- {metric} ---")
        print(f"{'layer':>6}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}")
        for L in report_layers:
            print(f"{L:>6}" + "".join(f"{_qmean(arr[:, L - 1], qid, k):>8.3f}" for k in range(4)))
        best = []
        for k in range(4):
            means = [_qmean(arr[:, l], qid, k) for l in range(arr.shape[1])]
            l_best = int(np.nanargmax(means))
            best.append(f"Q{k + 1}: layer {l_best + 1} ({means[l_best]:.3f})")
        print("best layer per quartile:  " + " | ".join(best))
    print("\n--- readout with the trained prompt at the last layer (what the model actually outputs) ---")
    print(f"{'':>6}" + "".join(f"{_qmean(readout, qid, k):>8.3f}" for k in range(4)))
    print("(probe = upper bound using the image's own GT; salience = unsupervised; readout = actual zero-shot AUROC)")
    print("NOTE: salience can fall well BELOW 0.5 when the lesion is a tight, homogeneous cluster inside a diverse "
          "background (the lesion then sits closer to the image's median token than the background does). Read it as "
          "|AUROC - 0.5|; the probe is direction-free and is the primary metric.")
