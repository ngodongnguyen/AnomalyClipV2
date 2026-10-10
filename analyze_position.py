"""EXP-033: lesion position (distance of the lesion centroid from the image centre) and polarity (lesion minus ring brightness) versus the
per-image AUROC of the matched ECP (EXP-032 tables). No model, CPU only.
  python analyze_position.py compute --name ClinicDB --dataset colon --data_path <root> --out_dir results/EXP-032
  python analyze_position.py report  --report_dir results/EXP-032
`compute` needs the dataset on disk (the server); `report` only needs results/EXP-032/<set>/badcase_<set>.csv and position_<set>.csv."""
import os, csv, time, argparse
import numpy as np
import zpos_stats as Z

SETS = ["ClinicDB", "ColonDB", "ISIC", "Endo", "Kvasir", "TN3K"]
DECISION = ["ClinicDB", "ColonDB", "ISIC", "Endo"]           # galleries not opened before the rule was registered
EXPLORATORY = ["Kvasir", "TN3K"]


def stage_compute(args):
    import argparse as _ap
    from dataset import Dataset
    from utils import get_transform
    from analyze_badcases import read_mask, load_rgb
    out_dir = os.path.join(args.out_dir, args.name)
    out_csv = os.path.join(out_dir, f"position_{args.name}.csv")
    if os.path.exists(out_csv) and not args.redo:
        print(f"{args.name}: {out_csv} exists, skipped (use --redo)"); return
    os.makedirs(out_dir, exist_ok=True)
    preprocess, target_transform = get_transform(_ap.Namespace(image_size=args.image_size))
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    t0 = time.time(); rows = []; kept = 0
    for i in range(len(data)):
        it = data[i]
        gt = read_mask(it)
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        kept += 1
        if args.limit and kept > args.limit:
            break
        v = load_rgb(it["img_path"], args.image_size).max(-1)
        rows.append([os.path.relpath(it["img_path"], args.data_path), Z.centroid_offset(gt), Z.ring_contrast(v, gt, 28)])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "c", "contrast"]); w.writerows(rows)
    print(f"[stage] {args.name}: {len(rows)} images, wrote {out_csv}, {time.time() - t0:.0f}s")


def read_join(d, name):
    bc = {r["id"]: r for r in csv.DictReader(open(os.path.join(d, name, f"badcase_{name}.csv")))}
    pos = {r["id"]: r for r in csv.DictReader(open(os.path.join(d, name, f"position_{name}.csv")))}
    ids = [i for i in bc if i in pos]
    f = lambda src, k, ids_=ids: np.array([float(src[i][k]) for i in ids_])
    out = dict(ids=ids, n_bc=len(bc), n_pos=len(pos), auroc=f(bc, "auroc"), hit=f(bc, "hit_rate"), area=f(bc, "area_frac"), c=f(pos, "c"), contrast=f(pos, "contrast"))
    cy, cx = f(bc, "comp_cy"), f(bc, "comp_cx")
    out["fp_c"] = np.hypot(cy - 259.5, cx - 259.5) / 259.5
    return out


def stage_report(args):
    res = {}
    for s in SETS:
        p = os.path.join(args.report_dir, s, f"position_{s}.csv")
        if not os.path.isfile(p):
            print(f"MISSING {p}"); continue
        d = read_join(args.report_dir, s)
        ok = np.isfinite(d["contrast"])
        n = len(d["ids"])
        rho_s = float(Z.spearmanr(d["c"], d["auroc"])[0]); rho_h = float(Z.spearmanr(d["c"], d["hit"])[0])
        pr = Z.boot_stat(Z.partial_spearman, [d["c"], d["auroc"], np.log(d["area"])], n_boot=args.boot)
        dark = (d["contrast"] < 0)
        dd = Z.boot_stat(lambda a, f: Z.group_diff(a, f > 0.5), [d["auroc"][ok], dark[ok].astype(float)], n_boot=args.boot)
        pc = float(Z.partial_spearman(d["contrast"][ok], d["auroc"][ok], np.log(d["area"][ok])))
        res[s] = dict(n=n, pr=pr, dd=dd, groups=(int(dark[ok].sum()), int((~dark[ok]).sum())), rho_s=rho_s, rho_h=rho_h, pc=pc,
                      med_c=float(np.median(d["c"])), med_fp=float(np.nanmedian(d["fp_c"])), dark_prev=float(dark[ok].mean()), n_ring_empty=int((~ok).sum()),
                      joined=f"{n}/{d['n_bc']}")
    for s, r in res.items():
        tag = "DECISION" if s in DECISION else "exploratory"
        print(f"\n=== {s} ({tag}; n={r['n']}, joined {r['joined']}; empty rings {r['n_ring_empty']}) ===")
        print(f"   partial Spearman(lesion offset c, AUROC | log area) = {r['pr'][0]:+.3f} [{r['pr'][1]:+.3f}, {r['pr'][2]:+.3f}]   simple Spearman(c, AUROC) {r['rho_s']:+.3f}   Spearman(c, hit rate) {r['rho_h']:+.3f}")
        print(f"   DARK lesions (contrast < 0): prevalence {r['dark_prev']:.2f}, groups (dark, not dark) = {r['groups']}, AUROC dark - not dark = {r['dd'][0]:+.4f} [{r['dd'][1]:+.4f}, {r['dd'][2]:+.4f}]   partial rank corr (contrast, AUROC | log area) {r['pc']:+.3f}")
        print(f"   median lesion offset {r['med_c']:.2f}; median offset of the largest far-FP blob {r['med_fp']:.2f}")
    print("\n================ VERDICT (rule fixed in research/EXPERIMENTS.md EXP-033; decision sets ClinicDB, ColonDB, ISIC, Endo) ================")
    miss = [s for s in DECISION if s not in res]
    if miss:
        print(f"INCOMPLETE: decision sets missing: {miss}; the rule is not applied"); return
    print("PERIPHERY:", Z.verdict_periphery([res[s]["pr"] for s in DECISION]), "  partial rho per set:", " ".join(f"{s}:{res[s]['pr'][0]:+.2f}" for s in DECISION))
    print("DARK-LESION:", Z.verdict_dark([res[s]["dd"] for s in DECISION], [res[s]["groups"] for s in DECISION]), "  AUROC diff per set:", " ".join(f"{s}:{res[s]['dd'][0]:+.3f}" for s in DECISION))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compute", "report"])
    ap.add_argument("--name"); ap.add_argument("--dataset"); ap.add_argument("--data_path")
    ap.add_argument("--out_dir", default="results/EXP-032"); ap.add_argument("--report_dir", default="results/EXP-032")
    ap.add_argument("--image_size", type=int, default=518); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true"); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    stage_compute(a) if a.mode == "compute" else stage_report(a)
