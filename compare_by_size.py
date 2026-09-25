"""
Where does model B beat model A? Paired per-image AUROC difference by lesion-size quartile, with a paired
bootstrap 95% CI. Inputs are the per_image.csv files written by test.py (same dataset, same images).

    python compare_by_size.py --a results/zoom_mvtec/CVC-ClinicDB/per_image.csv \
                              --b results/ecp_extent/CVC-ClinicDB/per_image.csv
Several datasets / seeds can be passed as repeated --a / --b pairs of equal length; they are pooled.
"""
import csv
import argparse
import numpy as np


def load(path):
    with open(path) as f:
        return {r["image"]: (float(r["area_frac"]), float(r["auroc"])) for r in csv.DictReader(f)}


def paired(paths_a, paths_b):
    area, a, b = [], [], []
    for pa, pb in zip(paths_a, paths_b):
        da, db = load(pa), load(pb)
        for k in da.keys() & db.keys():
            area.append(da[k][0])
            a.append(da[k][1])
            b.append(db[k][1])
    return np.array(area), np.array(a), np.array(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="baseline per_image.csv file(s)")
    ap.add_argument("--b", nargs="+", required=True, help="method per_image.csv file(s), same order")
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    assert len(args.a) == len(args.b)
    area, a, b = paired(args.a, args.b)
    rng = np.random.default_rng(0)
    qid = np.digitize(area, np.quantile(area, [.25, .5, .75]))
    print(f"n={len(area)} paired images   (Q1 smallest lesion ... Q4 largest)")
    print(f"{'group':<10}{'n':>6}{'mean A':>9}{'mean B':>9}{'B-A':>9}{'95% CI':>20}")
    groups = [(f"Q{k + 1}", qid == k) for k in range(4)] + [("all", np.ones(len(area), bool))]
    for name, m in groups:
        d = (b - a)[m]
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{name:<10}{m.sum():>6}{a[m].mean():>9.4f}{b[m].mean():>9.4f}{d.mean():>+9.4f}   [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
