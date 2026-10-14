"""EXP-047: CPU gate for supervising more lesion attributes (solidity, boundary-contrast magnitude, interior inhomogeneity). No model.
  python analyze_attr_gate.py compute --name ClinicDB --dataset colon --data_path <root> --out_dir results/EXP-047      # one medical set
  python analyze_attr_gate.py compute --name MVTec --dataset mvtec --data_path <mvtec root> --out_dir results/EXP-047  # MVTec source (anomalous test records with masks)
  python analyze_attr_gate.py report --attr_dir results/EXP-047 --mvtec_dir results/EXP-047 --pos_dir results/EXP-032
`compute` needs the dataset on disk (server); `report` needs the attr CSVs plus results/EXP-032/<set>/badcase_<set>.csv and position_<set>.csv."""
import os, csv, time, argparse
from collections import Counter
import numpy as np
import attr_stats as S

SETS = ["ClinicDB", "ColonDB", "ISIC", "Endo", "Kvasir", "TN3K"]
NAMES = {"solidity": "SOLIDITY", "bcm": "BOUNDARY CONTRAST |in-ring|", "inhom": "INHOMOGENEITY"}


def stage_compute(args):
    import argparse as _ap
    from dataset import Dataset
    from utils import get_transform
    from analyze_badcases import read_mask, load_rgb
    is_mv = args.name == "MVTec"
    out_dir = os.path.join(args.out_dir, args.name)
    out_csv = os.path.join(out_dir, f"attr_{args.name}.csv")
    if os.path.exists(out_csv) and not args.redo:
        print(f"{args.name}: {out_csv} exists, skipped (use --redo)"); return
    os.makedirs(out_dir, exist_ok=True)
    _, target_transform = get_transform(_ap.Namespace(image_size=args.image_size))
    data = Dataset(root=args.data_path, transform=None, target_transform=target_transform, dataset_name=args.dataset, mode="test")
    t0 = time.time(); rows = []; seen = Counter(); kept = Counter(); drop = Counter()
    for i in range(len(data)):
        rec = data.data_all[i]
        if is_mv and rec["anomaly"] != 1:
            continue
        it = data[i]
        cat = it["cls_name"]
        seen[cat] += 1
        gt = read_mask(it) if not isinstance(it["img_mask"], list) else np.zeros((args.image_size,) * 2, bool)
        if gt.sum() < 20 or (~gt).sum() < 20:
            drop[cat] += 1; continue
        v = load_rgb(it["img_path"], args.image_size).max(-1)
        bcm = S.boundary_contrast_mag(v, gt, 28)
        if not np.isfinite(bcm):                                  # empty ring (EXP-033 exclusion)
            drop[cat] += 1; continue
        sol = S.solidity(gt)
        if not np.isfinite(sol):
            drop[cat] += 1; continue
        kept[cat] += 1
        if args.limit and sum(kept.values()) > args.limit:
            break
        rows.append([os.path.relpath(it["img_path"], args.data_path), cat, float(gt.mean()), sol, bcm, S.inhomogeneity(v, gt)])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "category", "area_frac", "solidity", "bcm", "inhom"]); w.writerows(rows)
    print(f"[stage] {args.name}: {len(rows)} images kept, {sum(drop.values())} excluded, wrote {out_csv}, {time.time() - t0:.0f}s")
    if is_mv:
        for c in sorted(seen):
            print(f"   {c:12s} anomalous records {seen[c]:4d}  kept {kept[c]:4d}  excluded {drop[c]:3d}")


def read_attr(path):
    rows = list(csv.DictReader(open(path)))
    out = {"id": [r["id"] for r in rows]}
    for k in ["area_frac"] + S.ATTRS:
        out[k] = np.array([float(r[k]) for r in rows])
    return out


def read_set(attr_dir, pos_dir, s):
    a = read_attr(os.path.join(attr_dir, s, f"attr_{s}.csv"))
    bc = {r["id"]: r for r in csv.DictReader(open(os.path.join(pos_dir, s, f"badcase_{s}.csv")))}
    pos = {r["id"]: r for r in csv.DictReader(open(os.path.join(pos_dir, s, f"position_{s}.csv")))}
    idx = [i for i, k in enumerate(a["id"]) if k in bc and k in pos]
    ids = [a["id"][i] for i in idx]
    d = {k: a[k][idx] for k in S.ATTRS}
    d.update(ids=ids, n_attr=len(a["id"]), n_bc=len(bc),
             auroc=np.array([float(bc[k]["auroc"]) for k in ids]), area=np.array([float(bc[k]["area_frac"]) for k in ids]),
             c=np.array([float(pos[k]["c"]) for k in ids]))
    return d


def q(v):
    v = v[np.isfinite(v)]
    return "n/a" if len(v) == 0 else " ".join(f"{x:.3f}" for x in np.percentile(v, [5, 25, 50, 75, 95]))


def stage_report(args):
    mv_path = os.path.join(args.mvtec_dir, "MVTec", "attr_MVTec.csv")
    if not os.path.isfile(mv_path):
        print(f"INCOMPLETE: missing {mv_path}"); return
    mv = read_attr(mv_path)
    sets = {}
    for s in SETS:
        p = os.path.join(args.attr_dir, s, f"attr_{s}.csv")
        if not (os.path.isfile(p) and os.path.isfile(os.path.join(args.pos_dir, s, f"badcase_{s}.csv")) and os.path.isfile(os.path.join(args.pos_dir, s, f"position_{s}.csv"))):
            print(f"MISSING inputs for {s}"); continue
        sets[s] = read_set(args.attr_dir, args.pos_dir, s)
    print(f"MVTec anomalous images kept: {len(mv['id'])}")
    for s, d in sets.items():
        print(f"{s}: joined {len(d['ids'])} of attr {d['n_attr']} / badcase {d['n_bc']}")
    if len(sets) < len(SETS):
        print("INCOMPLETE: not all six medical sets are available; the verdict is not applied"); return
    print("\nattribute percentiles (p5 p25 p50 p75 p95):")
    for k in S.ATTRS:
        print(f"  {NAMES[k]:28s} MVTec  {q(mv[k])}")
        for s in SETS:
            print(f"  {'':28s} {s:8s}{q(sets[s][k])}")
    rng = np.random.RandomState(0)
    verdicts = {}
    for k in S.ATTRS:
        print(f"\n================ {NAMES[k]} ================")
        fr = [S.support_fraction(mv[k], sets[s][k]) for s in SETS]
        a_ok, a_n = S.check_A(fr)
        print(f"A SUPPORT (fraction inside MVTec p5-p95; needs >= 0.80 on >= 4/6): " + " ".join(f"{s}:{f:.3f}" for s, f in zip(SETS, fr)) + f"  -> {a_n}/6  {'PASS' if a_ok else 'FAIL'}")
        r_mv = S.area_corr(mv[k], mv["area_frac"])
        r_s = [S.area_corr(sets[s][k], sets[s]["area"]) for s in SETS]
        b_ok, b_n, b_mv = S.check_B(r_mv, r_s)
        print(f"B NOT REDUNDANT (|Spearman(attr, log area)| <= 0.80 on MVTec and >= 4/6 sets): MVTec {r_mv:+.3f} ({'ok' if b_mv else 'FAIL'}); " + " ".join(f"{s}:{r:+.3f}" for s, r in zip(SETS, r_s)) + f"  -> {b_n}/6  {'PASS' if b_ok else 'FAIL'}")
        stats, null = [], []
        for s in SETS:
            d = sets[s]; ok = np.isfinite(d[k])
            st = S.boot_partial(d[k][ok], d["auroc"][ok], d["area"][ok], d["c"][ok], n_boot=args.boot)
            stats.append(st)
            perm = rng.permutation(d["auroc"][ok])                       # null: per-image AUROC permuted over images
            null.append(S.boot_partial(d[k][ok], perm, d["area"][ok], d["c"][ok], n_boot=args.boot))
        c_ok, npos, nneg = S.check_C(stats)
        print("C ASSOCIATION (partial Spearman(attr, AUROC | log area, c), CI; pass |rho| >= 0.20 and CI excludes 0, same sign on >= 4/6):")
        for s, st in zip(SETS, stats):
            print(f"   {s:8s} {st[0]:+.3f} [{st[1]:+.3f}, {st[2]:+.3f}]  n={int(np.isfinite(sets[s][k]).sum())}  {'pass' if S.set_pass_C(st) else '-'}")
        n_null = sum(S.set_pass_C(t) != 0 for t in null)
        print(f"   -> positive passes {npos}, negative passes {nneg}  {'PASS' if c_ok else 'FAIL'};  permutation null: set passes {n_null}/6 (" + " ".join(f"{t[0]:+.2f}" for t in null) + ")")
        verdicts[k] = S.verdict_attribute(a_ok, b_ok, c_ok)
        print(f"{k}: {verdicts[k]}   (A {'P' if a_ok else 'F'}, B {'P' if b_ok else 'F'}, C {'P' if c_ok else 'F'})")
    print("\n================ VERDICT (rule fixed in research/EXPERIMENTS.md EXP-047) ================")
    for k in S.ATTRS:
        print(f"{k}: {verdicts[k]}")
    print("DIRECTION:", S.verdict_direction(verdicts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compute", "report"])
    ap.add_argument("--name"); ap.add_argument("--dataset"); ap.add_argument("--data_path")
    ap.add_argument("--out_dir", default="results/EXP-047")
    ap.add_argument("--attr_dir", default="results/EXP-047"); ap.add_argument("--mvtec_dir", default="results/EXP-047"); ap.add_argument("--pos_dir", default="results/EXP-032")
    ap.add_argument("--image_size", type=int, default=518); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true"); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    stage_compute(a) if a.mode == "compute" else stage_report(a)
