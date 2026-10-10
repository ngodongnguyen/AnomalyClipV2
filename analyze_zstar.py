"""EXP-031: does ANY area-based choice of z help? CPU only, no model. Reads the per-image tables written by EXP-027
(results/EXP-027/zens_<Set>_images.csv: per-image pixel AUROC at each z of the grid, GT area fraction, area quartile).
  python analyze_zstar.py [dir]        (default results/EXP-027)
For each set: Spearman(slope, log area) with bootstrap CI; cross-fitted 'z depends only on the GT area quartile' rule versus fixed z=1.6 (an ORACLE
upper bound for any estimator of extent, since it knows the true area), plus a shuffled-quartile control. Prints the pre-registered verdict."""
import csv, os, sys
import numpy as np
import zstar_stats as zs_

SETS = ["ClinicDB", "Kvasir", "ColonDB", "ISIC", "Endo", "TN3K"]
FIXED_Z = 1.6


def read(path):
    rows = list(csv.DictReader(open(path)))
    zcols = sorted([c for c in rows[0] if c.startswith("auroc_z") and "ctrl" not in c], key=lambda c: float(c[len("auroc_z"):]))
    zvals = [float(c[len("auroc_z"):]) for c in zcols]
    aucs = np.array([[float(r[c]) for c in zcols] for r in rows])
    return dict(ids=[r["image"] for r in rows], area=np.array([float(r["area_frac"]) for r in rows]),
                q=np.array([int(r["quartile"][1]) - 1 for r in rows]), aucs=aucs, zvals=zvals)


def analyse(d):
    zv = d["zvals"]
    fixed = int(np.argmin(np.abs(np.array(zv) - FIXED_Z)))
    slope = zs_.slope_stat(d["aucs"], zv)
    rho, lo, hi = zs_.rho_boot(slope, np.log(d["area"]))
    gain = zs_.crossfit_gain(d["aucs"], d["q"], fixed, d["ids"])
    gm, gl, gh = zs_.boot_mean(gain)
    rng = np.random.RandomState(0)
    ctrl = [zs_.crossfit_gain(d["aucs"], d["q"], fixed, d["ids"], groups=rng.permutation(d["q"])).mean() for _ in range(50)]
    # also the best single z for the whole set (cross-fitted with one group): is 1.6 itself the best constant?
    one = zs_.crossfit_gain(d["aucs"], np.zeros(len(d["q"]), int), fixed, d["ids"]).mean()
    zstar = np.array(zv)[d["aucs"].argmax(1)]
    return dict(n=len(d["ids"]), rho=rho, lo=lo, hi=hi, assoc=zs_.association(rho, lo, hi), gain=gm, gl=gl, gh=gh, useful=zs_.useful(gm, gl),
                ctrl=float(np.mean(ctrl)), one=float(one), mean_zstar_by_q=[float(zstar[d["q"] == k].mean()) for k in range(4)],
                mean_fixed=float(d["aucs"][:, fixed].mean()))


def main(root):
    out = {}
    for s in SETS:
        p = os.path.join(root, f"zens_{s}_images.csv")
        if not os.path.isfile(p):
            print(f"MISSING {p}")
            continue
        out[s] = analyse(read(p))
    for s, r in out.items():
        print(f"\n== {s} (n={r['n']}), mean per-image AUROC at z=1.6: {r['mean_fixed']:.4f}")
        print(f"   Spearman(AUROC(z max) - AUROC(z min), log area) = {r['rho']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  -> {r['assoc']}")
        print(f"   mean argmax-z by area quartile Q1..Q4: " + "  ".join(f"{x:+.2f}" for x in r["mean_zstar_by_q"]))
        print(f"   cross-fitted area-quartile rule - FIXED: {r['gain']:+.4f} [{r['gl']:+.4f}, {r['gh']:+.4f}]  -> {'USEFUL' if r['useful'] else 'not useful'}"
              f"   (shuffled-quartile control {r['ctrl']:+.4f}; one best constant z for the whole set {r['one']:+.4f})")
    print("\n================ VERDICT (rule fixed in research/EXPERIMENTS.md EXP-031) ================")
    if len(out) < len(SETS):
        print(f"INCOMPLETE: only {len(out)}/{len(SETS)} sets found; the rule is not applied")
        return
    uf = [out[s]["useful"] for s in SETS]
    af = [out[s]["assoc"] for s in SETS]
    print("USEFUL (gain >= +0.005, lower bound > 0): " + " ".join(f"{s}:{'Y' if u else 'n'}" for s, u in zip(SETS, uf)) + f"   ({sum(uf)}/6)")
    print("ASSOCIATION (|rho| >= 0.20, CI excludes 0): " + " ".join(f"{s}:{a}" for s, a in zip(SETS, af)))
    print("VERDICT:", zs_.verdict(uf, af))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/EXP-027")
