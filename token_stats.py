"""Torch-free statistics for analyze_token_norms.py (numpy/scipy only, so they can be unit-tested anywhere)."""
import numpy as np
from scipy.stats import spearmanr


def robust_outlier_mask(norm_map, z_thresh=4.0):
    med = np.median(norm_map)
    mad = np.median(np.abs(norm_map - med)) * 1.4826 + 1e-8
    return (norm_map - med) / mad > z_thresh


def image_metrics(norm_map, lesion, valid, abn, z_thresh=4.0):
    """
    norm_map, lesion, valid, abn: [side, side] arrays (lesion/valid boolean; abn = abnormal score).
    Outliers are flagged by a per-image robust z-score on token norms.
    """
    out = robust_outlier_mask(norm_map, z_thresh)
    outside = valid & ~lesion
    border = ~valid
    m = {
        "has_out": float(out.any()),
        "out_frac": float(out.mean()),
        "rate_in_lesion": float(out[lesion].mean()) if lesion.any() else np.nan,
        "rate_out_lesion": float(out[outside].mean()) if outside.any() else np.nan,
        "share_in_border": float((out & border).sum() / out.sum()) if out.any() else np.nan,
        "norm_ratio": float(np.median(norm_map[lesion]) / (np.median(norm_map[outside]) + 1e-8))
        if lesion.any() and outside.any() else np.nan,
    }
    inl = out & lesion
    non = (~out) & lesion
    m["abn_at_out_in_lesion"] = float(abn[inl].mean()) if inl.any() else np.nan
    m["abn_at_clean_in_lesion"] = float(abn[non].mean()) if non.any() else np.nan
    return m


def _mean(rows, k, sel=None):
    v = np.array([r[k] for r in rows], dtype=float)
    if sel is not None:
        v = v[sel]
    return np.nanmean(v) if np.isfinite(v).any() else np.nan


def print_token_report(name, rows, area_frac, auroc):
    n = len(rows)
    qid = np.digitize(area_frac, np.quantile(area_frac, [.25, .5, .75]))
    print(f"\n=== Token-norm outliers: stream '{name}' (n={n} images; Q1 smallest lesion ... Q4 largest) ===")
    print(f"images with >=1 outlier patch: {100 * _mean(rows, 'has_out'):.1f}% | "
          f"mean outlier rate over all patches: {100 * _mean(rows, 'out_frac'):.2f}% | "
          f"share of outliers sitting in the black border: {100 * _mean(rows, 'share_in_border'):.0f}%")
    print(f"{'group':<8}{'rate inside lesion':>20}{'rate outside (valid)':>22}{'enrichment':>12}{'norm ratio':>12}")
    for label, sel in [("ALL", None)] + [(f"Q{k + 1}", qid == k) for k in range(4)]:
        a, b = _mean(rows, "rate_in_lesion", sel), _mean(rows, "rate_out_lesion", sel)
        enr = a / b if b and b > 1e-9 else float("nan")
        print(f"{label:<8}{100 * a:>19.2f}%{100 * b:>21.2f}%{enr:>12.2f}{_mean(rows, 'norm_ratio', sel):>12.3f}")
    print("(enrichment > 1: outlier tokens are MORE frequent inside the lesion than in normal tissue; "
          "norm ratio = median token norm inside lesion / outside)")

    a_out = _mean(rows, "abn_at_out_in_lesion")
    a_clean = _mean(rows, "abn_at_clean_in_lesion")
    print(f"mean abnormal score inside lesion: at outlier tokens = {a_out:.3f} vs clean tokens = {a_clean:.3f}")

    def corr(key, sel=None, label=""):
        x = np.array([r[key] for r in rows], dtype=float)
        y = np.array(auroc, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if sel is not None:
            ok &= sel
        if ok.sum() < 20 or np.ptp(x[ok]) == 0:
            return "n/a"
        rho, p = spearmanr(x[ok], y[ok])
        return f"{rho:+.3f} (p={p:.2g}, n={int(ok.sum())})"

    print("Spearman corr with per-image AUROC (native patch grid):")
    for key in ("rate_in_lesion", "out_frac", "norm_ratio"):
        print(f"  {key:<16} ALL: {corr(key):<32} Q4 only: {corr(key, qid == 3)}")
