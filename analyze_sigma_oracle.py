"""
Zero-cost re-analysis of the existing diag_suite_colon_*.csv files (no GPU, no new model run):
how much headroom would a PERFECT coverage-adaptive sigma selector give, isolated from every
other variant (scale, layer, fusion, ...)? This is the cheapest possible test of idea C's core
premise ("different lesion sizes want different processing") before building any learned gate.
"""
import csv
import argparse
import numpy as np

SIGMAS = [0, 2, 8, 16, 24, 32, 48]


def main(args):
    rows = list(csv.DictReader(open(args.csv)))
    la = np.array([float(r["log_area"]) for r in rows])
    qid = np.digitize(la, np.quantile(la, [.25, .5, .75]))
    n = len(rows)

    sig = {s: np.array([float(r[f"sigma{s}"]) for r in rows]) for s in SIGMAS}
    best_idx = np.argmax(np.stack([sig[s] for s in SIGMAS]), axis=0)
    oracle = np.stack([sig[s] for s in SIGMAS])[best_idx, np.arange(n)]

    fixed32, fixed48 = sig[32], sig[48]
    print(f"\n=== Sigma-oracle headroom: {args.csv} (n={n}) ===")
    print(f"{'group':<8}{'n':>5}{'fixed-s32':>10}{'fixed-s48':>10}{'oracle':>9}{'s48 closes':>12}{'oracle-s48':>11}{'mode':>6}")
    groups = [("ALL", np.ones(n, bool))] + [(f"Q{k+1}", qid == k) for k in range(4)]
    for name, sel in groups:
        b32, b48, o = fixed32[sel].mean(), fixed48[sel].mean(), oracle[sel].mean()
        gap32 = o - b32
        closes = (b48 - b32) / gap32 if abs(gap32) > 1e-9 else float("nan")
        modal_sigma = SIGMAS[np.bincount([best_idx[i] for i in np.where(sel)[0]]).argmax()]
        print(f"{name:<8}{int(sel.sum()):>5}{b32:>10.3f}{b48:>10.3f}{o:>9.3f}{closes:>11.0%}{o-b48:>+11.3f}{modal_sigma:>6}")
    print("('s48 closes' = how much of the [fixed32 -> oracle] gap is already closed by just switching the FIXED sigma to 48)")

    print("\nDistribution of the per-image best sigma, by quartile:")
    print(f"{'group':<8}" + "".join(f"{s:>8}" for s in SIGMAS))
    for name, sel in groups:
        idxs = best_idx[sel]
        counts = np.bincount(idxs, minlength=len(SIGMAS))
        print(f"{name:<8}" + "".join(f"{c/sel.sum():>8.0%}" for c in counts))

    print("\nCorrelation of log_area with the per-image best sigma (Spearman-ish via rank corr):")
    from scipy.stats import spearmanr
    best_sigma_val = np.array([SIGMAS[i] for i in best_idx])
    rho, p = spearmanr(la, best_sigma_val)
    print(f"  rho={rho:+.3f}  p={p:.2g}  (positive => larger lesions really do prefer larger sigma)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    main(ap.parse_args())
