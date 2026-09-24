"""Torch-free statistics for analyze_homogenization.py (numpy only, unit-testable anywhere)."""
import numpy as np


def _mean_pair_cos(x):
    """Mean pairwise cosine among L2-normalised rows, computed in O(n*C) via ||sum||^2."""
    n = x.shape[0]
    if n < 2:
        return np.nan
    s = x.sum(0)
    return float((s @ s - n) / (n * (n - 1)))


def _normalise(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def homog_metrics(feats, y):
    """
    feats: [N, C] L2-normalised token features of the valid tokens of one image; y: [N] bool (lesion).
    Returns raw and image-mean-centred versions of
      c_all : mean pairwise cosine over all tokens (global homogenisation / rank-collapse indicator)
      c_ll, c_bb : mean pairwise cosine inside lesion / inside background
      c_lb  : mean cosine between a lesion token and a background token
      s     : (c_ll + c_bb)/2 - c_lb   (>0 : classes are tighter inside than across; 0 : indistinguishable)
    """
    out = {}
    for tag, f in (("raw", feats), ("cent", _normalise(feats - feats.mean(0)))):
        fl, fb = f[y], f[~y]
        c_ll, c_bb = _mean_pair_cos(fl), _mean_pair_cos(fb)
        c_lb = float(fl.mean(0) @ fb.mean(0))
        out[f"c_all_{tag}"] = _mean_pair_cos(f)
        out[f"c_ll_{tag}"] = c_ll
        out[f"c_bb_{tag}"] = c_bb
        out[f"c_lb_{tag}"] = c_lb
        out[f"s_{tag}"] = (c_ll + c_bb) / 2 - c_lb
    return out


def _qmean(arr, qid, k, layer):
    v = arr[qid == k, layer]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def print_homog_report(name, res, area_frac, report_layers, ref_layer=9, last_layer=24):
    """res: dict '<metric>_<stream>' -> [n_images, n_layers]."""
    qid = np.digitize(area_frac, np.quantile(area_frac, [.25, .5, .75]))
    print(f"\n##### Token homogenisation vs depth: {name} (n={len(area_frac)}; Q1 smallest lesion ... Q4 largest) #####")
    for stream in ("v", "o"):
        print(f"\n=== stream '{stream}' ({'V-V patch features that are scored' if stream == 'v' else 'standard-CLIP residual stream'}) ===")
        for metric in ("c_all_raw", "s_raw", "s_cent"):
            arr = res[f"{metric}_{stream}"]
            print(f"\n--- {metric} ---")
            print(f"{'layer':>6}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}")
            for L in report_layers:
                print(f"{L:>6}" + "".join(f"{_qmean(arr, qid, k, L - 1):>8.3f}" for k in range(4)))
        print(f"\n--- change from layer {ref_layer} to layer {last_layer} (Q1 = small lesions ... Q4 = large lesions) ---")
        print(f"{'metric':<12}" + "".join(f"{'Q' + str(k + 1):>9}" for k in range(4)) + f"{'Q4/Q1':>9}")
        for metric in ("c_all_raw", "c_ll_raw", "c_bb_raw", "c_lb_raw", "s_raw", "s_cent"):
            arr = res[f"{metric}_{stream}"]
            d = [(_qmean(arr, qid, k, last_layer - 1) - _qmean(arr, qid, k, ref_layer - 1)) for k in range(4)]
            ratio = d[3] / d[0] if abs(d[0]) > 1e-9 else float("nan")
            print(f"{metric:<12}" + "".join(f"{x:>+9.3f}" for x in d) + f"{ratio:>9.2f}")
    print("\n(c_all rising with depth = tokens of an image collapse toward a common direction (over-smoothing); "
          "s falling = lesion/background clusters merge. If Q4 changes are several times larger than Q1, large lesions "
          "are homogenised faster.)")
