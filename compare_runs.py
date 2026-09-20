"""
Paired comparison of two diagnostic-suite CSVs (e.g. control vs zoom-augmented prompts) on the same images.

Groups: ALL, area quartiles Q1..Q4, the 'large & smooth' subgroup (top-quartile lesion area AND texture ratio
<= median), the rest, and an area-weighted mean (rough proxy for the pooled pixel-AUROC used in the paper).
Reports mean(ctrl), mean(zoom), difference and a 95% paired-bootstrap CI.
"""
import csv
import argparse
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    return {r["image"]: {k: float(v) for k, v in r.items() if k != "image"} for r in rows}


def boot_ci(d, w=None, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, (n_boot, n))
    if w is None:
        stats = d[idx].mean(1)
    else:
        stats = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return np.percentile(stats, [2.5, 97.5])


def main(args):
    A, B = load(args.ctrl), load(args.zoom)
    names = sorted(set(A) & set(B))
    print(f"images in both runs: {len(names)} (ctrl {len(A)}, zoom {len(B)})")
    col = lambda R, k: np.array([R[n][k] for n in names])
    la, tex = col(A, "log_area"), col(A, "tex_ratio")
    q = np.digitize(la, np.quantile(la, [.25, .5, .75]))
    S = (la >= np.quantile(la, .75)) & (tex <= np.median(tex))
    w = np.exp(la)
    groups = [("ALL", np.ones(len(names), bool))] + [(f"Q{k + 1}", q == k) for k in range(4)] + \
             [("big&smooth", S), ("rest", ~S)]

    for v in args.variants:
        key = v + args.suffix
        a, b = col(A, key), col(B, key)
        d = b - a
        print(f"\n=== {key}   (positive diff = zoom better) ===")
        print(f"{'group':<12}{'n':>6}{'ctrl':>9}{'zoom':>9}{'diff':>9}{'95% CI':>20}")
        for name, sel in groups:
            lo, hi = boot_ci(d[sel])
            print(f"{name:<12}{int(sel.sum()):>6}{a[sel].mean():>9.3f}{b[sel].mean():>9.3f}{d[sel].mean():>+9.3f}"
                  f"{'[' + format(lo, '+.3f') + ', ' + format(hi, '+.3f') + ']':>20}")
        lo, hi = boot_ci(d, w)
        wa, wb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
        print(f"{'area-wtd':<12}{len(names):>6}{wa:>9.3f}{wb:>9.3f}{wb - wa:>+9.3f}"
              f"{'[' + format(lo, '+.3f') + ', ' + format(hi, '+.3f') + ']':>20}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--zoom", required=True)
    ap.add_argument("--variants", nargs="+", default=["scale518", "sigma32"])
    ap.add_argument("--suffix", default="", help="'' for all-pixel AUROC, '@fov' for FOV-restricted")
    main(ap.parse_args())
