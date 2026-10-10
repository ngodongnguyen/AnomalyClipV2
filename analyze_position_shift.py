"""
EXP-034 / H024: does the frozen matched ECP pixel map respond to the absolute position of the same lesion content? (translation intervention; diagnostic, NOT a method)
Rule, arms, region, filters: research/EXPERIMENTS.md EXP-034.  Statistics: pos_shift_stats.py (unit-tested).

  python analyze_position_shift.py pixel --name ClinicDB --dataset colon --data_path $CVC/CVC-ClinicDB --checkpoint_path <ckpt> --reference_csv <EXP-012 csv>
  python analyze_position_shift.py smoke --name ClinicDB ...            # = pixel with --limit 12 random kept images, out_dir results/EXP-034/smoke, recorded-value checks off
  python analyze_position_shift.py report --report_dir results/EXP-034  # no torch
One dataset per command, resumable (an existing output CSV is skipped unless --redo).  Per image: SHAM (d=0), RECENTRE, RANDOM (same |d|) on the
normalised 3x518x518 tensor with reflection padding; the frozen ECP map (z_pix 1.6, sigma 4, layer 24) is built exactly as analyze_badcases.py, inverse-translated,
and scored with per-image AUROC only on the pixels valid under BOTH shifts.  STOP: SHAM full-image AUROC vs the EXP-012 reference > 1e-4.
"""
import os
import csv
import time
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

import pos_shift_stats as P
import zpos_stats as Z

SETS = ["ClinicDB", "ColonDB", "ISIC", "Endo", "Kvasir", "TN3K"]
DECISION = ["ClinicDB", "ColonDB", "ISIC", "Endo"]
COLS = ["id", "group", "c", "dx_rec", "dy_rec", "dx_rand", "dy_rand", "mag", "n_les_reg", "n_bg_reg", "status",
        "auroc_full_sham", "auroc_ref", "auroc_sham", "auroc_rec", "auroc_rand", "hit_sham", "hit_rec", "hit_rand"]


def read_thr(position_dir, name):
    p = os.path.join(position_dir, name, f"summary_{name}.csv")
    if os.path.isfile(p):
        for r in csv.DictReader(open(p)):
            if r["key"] == "thr_tpr0.8":
                return float(r["value"])
    return None


def stage_pixel(args):
    import torch
    import AnomalyCLIP_lib
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features
    from metrics import roc_auc_lowmem
    from analyze_badcases import load_model, read_mask

    out_dir = os.path.join(args.out_dir, args.name)
    out_csv = os.path.join(out_dir, f"shift_{args.name}.csv")
    if os.path.exists(out_csv) and not args.redo:
        print(f"{args.name}: {out_csv} exists, skipped (use --redo)")
        return
    os.makedirs(out_dir, exist_ok=True)
    T0 = time.time()
    S = args.image_size
    device, model, pl, cond = load_model(args)
    if cond is None or cond.mode != "extent":
        raise SystemExit("STOP: this diagnostic needs the 'extent' ECP checkpoint (z override)")
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    ref = None
    if args.reference_csv:
        ref = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(args.reference_csv))}
    else:
        print("WARNING: no --reference_csv, EXP-012 per-image parity NOT checked")
    thr = read_thr(args.position_dir, args.name)
    print(f"[stage] hit-rate threshold (EXP-032 pooled TPR 0.8): {thr if thr is not None else 'not found, hit rates skipped'}")

    # pass 1: kept images, lesion offset c (recomputed from the mask)
    t0 = time.time()
    kept, c_all, ids_all = [], [], []
    for i in range(len(data)):
        it = data[i]
        gt = read_mask(it)
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        kept.append(i); c_all.append(Z.centroid_offset(gt)); ids_all.append(os.path.relpath(it["img_path"], args.data_path))
    if args.limit:
        sel = np.random.RandomState(0).permutation(len(kept))[: args.limit]
        kept = [kept[j] for j in sel]; c_all = [c_all[j] for j in sel]; ids_all = [ids_all[j] for j in sel]
        print(f"--limit: {len(kept)} random kept images (smoke; terciles are taken inside this subset)")
    c_all = np.array(c_all)
    pos_csv = os.path.join(args.position_dir, args.name, f"position_{args.name}.csv")
    if os.path.isfile(pos_csv):
        pc = {r["id"]: float(r["c"]) for r in csv.DictReader(open(pos_csv))}
        dif = [abs(pc[s] - c) for s, c in zip(ids_all, c_all) if s in pc]
        print(f"[stage] c vs EXP-033 position csv: {len(dif)}/{len(ids_all)} joined, max |diff| = {max(dif) if dif else float('nan'):.2e}")
        if dif and max(dif) > 1e-6:
            raise SystemExit("STOP: recomputed lesion offset differs from position_<set>.csv")
    else:
        print("[stage] no EXP-033 position csv; c recomputed from the masks only")
    ip, ic = P.terciles(c_all, ids_all)
    todo = [(int(j), "peripheral") for j in ip] + ([(int(j), "central") for j in ic] if not args.no_central else [])
    print(f"[stage] {args.name}: {len(kept)} kept images; peripheral {len(ip)} (c >= {c_all[ip].min():.3f}), central {len(ic)} (c <= {c_all[ic].max():.3f}); pass 1 {time.time() - t0:.0f}s")

    def score_map(img):
        with torch.no_grad():
            imf, pf = model.encode_image(img.unsqueeze(0), [24], DPAM_layer=20)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            pfeat = pf[0] / pf[0].norm(dim=-1, keepdim=True)
            desc = visual_descriptor(imf, pf)
            zt = torch.full((imf.shape[0],), args.z_pix, device=device)
            _, c_pos, c_neg = cond(desc, z_override=zt)
            tf = conditioned_text_features(model, pl, c_pos, c_neg)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfeat, tf[0])
            sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], S)
            amap = ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu()
        return gaussian_filter(amap.numpy(), sigma=args.sigma).astype(np.float32)

    def shifted(img, dx, dy):
        if dx == 0 and dy == 0:
            return img                                              # identical tensor: SHAM is exactly the EXP-012 forward
        iy = torch.from_numpy(P.shift_indices(S, dy)).to(img.device)
        ix = torch.from_numpy(P.shift_indices(S, dx)).to(img.device)
        return img[:, iy][:, :, ix]                                 # out[y, x] = in[reflect(y - dy), reflect(x - dx)]

    rows, max_ref, max_impl, n_drop = [], 0.0, 0.0, 0
    t0 = time.time()
    for n, (j, group) in enumerate(todo):
        it = data[int(kept[j])]
        gt = read_mask(it)
        sid = ids_all[j]
        drec = P.recentre_displacement(gt)
        drand = P.random_displacement(sid, *drec)
        mag = float(np.hypot(*drec))
        reg = P.common_valid(gt.shape, [drec, drand])
        n_les, n_bg = P.region_counts(gt, reg)
        row = dict(id=sid, group=group, c=c_all[j], dx_rec=drec[0], dy_rec=drec[1], dx_rand=drand[0], dy_rand=drand[1], mag=mag,
                   n_les_reg=n_les, n_bg_reg=n_bg, status="kept")
        for k in COLS[11:]:
            row[k] = float("nan")
        if n_les < P.MIN_CLASS or n_bg < P.MIN_CLASS:
            row["status"] = "dropped_region"; n_drop += 1
            rows.append(row); continue
        img = it["img"].to(device)
        maps = {}
        for arm, (dx, dy) in (("sham", (0, 0)), ("rec", drec), ("rand", drand)):
            m = score_map(shifted(img, dx, dy))
            maps[arm] = m if arm == "sham" else P.inverse_align(m, dx, dy)
        row["auroc_full_sham"] = float(roc_auc_lowmem(gt, maps["sham"]))
        if ref is not None:
            if sid not in ref:
                raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
            row["auroc_ref"] = ref[sid]
            err = abs(row["auroc_full_sham"] - ref[sid]); max_ref = max(max_ref, err)
            if err > 1e-4:
                raise SystemExit(f"STOP: SHAM parity failure on {sid}: {row['auroc_full_sham']:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
        full = np.ones_like(gt, bool)
        max_impl = max(max_impl, abs(P.region_auroc(gt, maps["sham"], full) - row["auroc_full_sham"]))
        for arm in ("sham", "rec", "rand"):
            row["auroc_" + arm] = P.region_auroc(gt, maps[arm], reg)
            if thr is not None:
                row["hit_" + arm] = P.hit_rate(gt, maps[arm], reg, thr)
        rows.append(row)
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(todo)} images ({time.time() - t0:.0f}s, dropped {n_drop})", flush=True)
    print(f"[stage] forward {len(todo)} selected images ({n_drop} dropped by the >= {P.MIN_CLASS}-pixel region filter, not forwarded) {time.time() - t0:.0f}s")
    print(f"[parity] max |SHAM full-image AUROC - EXP-012| = {max_ref:.2e}" + ("" if ref is not None else " (NOT CHECKED)"))
    print(f"[parity] max |rank-formula AUROC - roc_auc_lowmem| on the full canvas = {max_impl:.2e} (stop > 1e-9)")
    if max_impl > 1e-9:
        raise SystemExit("STOP: region AUROC implementation disagrees with roc_auc_lowmem")
    tmp = out_csv + ".partial"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS)
        for r in rows:
            w.writerow([r[k] for k in COLS])
    os.replace(tmp, out_csv)
    print(f"wrote {out_csv}")
    report_set(rows, args.name, args.boot)
    print(f"[wall] {args.name} {time.time() - T0:.0f}s")


# ------------------------------------------------------------------ report (no torch)
def load_rows(path):
    out = []
    for r in csv.DictReader(open(path)):
        d = dict(r)
        for k in COLS:
            if k not in ("id", "group", "status"):
                d[k] = float(r[k])
        out.append(d)
    return out


def group_stats(rows, group, boot):
    g = [r for r in rows if r["group"] == group]
    k = [r for r in g if r["status"] == "kept"]
    st = dict(n_sel=len(g), n_drop=len(g) - len(k))
    if k:
        a = lambda key: np.array([r[key] for r in k])
        st.update(P.set_summary(a("auroc_sham"), a("auroc_rec"), a("auroc_rand"), a("mag"), boot))
        for arm in ("sham", "rec", "rand"):
            st["hit_" + arm] = float(np.nanmean(a("hit_" + arm))) if np.isfinite(a("hit_" + arm)).any() else float("nan")
        st["mean_mag"] = float(a("mag").mean()); st["mean_c"] = float(a("c").mean())
    else:
        st["n"] = 0
    return st


def ci(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def report_set(rows, name, boot, quiet=False):
    res = {g: group_stats(rows, g, boot) for g in ("peripheral", "central")}
    tag = "DECISION" if name in DECISION else "report-only"
    print(f"\n=== {name} ({tag}) ===")
    for g, st in res.items():
        if st["n_sel"] == 0:
            continue
        print(f"  [{g}] selected {st['n_sel']}, dropped by region filter {st['n_drop']}, kept {st['n']}")
        if st["n"]:
            print(f"     delta = (REC - SHAM) - (RAND - SHAM) = {ci(st['delta'])}   REC - SHAM {ci(st['rec_minus_sham'])}   RAND - SHAM {ci(st['rand_minus_sham'])}")
            print(f"     Spearman(|d|, delta) {st['rho_mag_delta']:+.3f}   mean |d| {st['mean_mag']:.0f} px   mean c {st['mean_c']:.2f}   hit rate (sham/rec/rand) {st['hit_sham']:.3f}/{st['hit_rec']:.3f}/{st['hit_rand']:.3f}")
    return res


def stage_report(args):
    res = {}
    for s in SETS:
        p = os.path.join(args.report_dir, s, f"shift_{s}.csv")
        if not os.path.isfile(p):
            print(f"MISSING {p}"); continue
        res[s] = report_set(load_rows(p), s, args.boot)
    print("\n================ VERDICT (rule fixed in research/EXPERIMENTS.md EXP-034; decision sets ClinicDB, ColonDB, ISIC, Endo; peripheral tercile) ================")
    by = {}
    for s in DECISION:
        st = res.get(s, {}).get("peripheral")
        by[s] = st["delta"] if st and st.get("n") else None
        if by[s] is not None:
            m, lo, hi = by[s]
            print(f"{s:9s} mean delta {m:+.4f}  lower {lo:+.4f}  upper {hi:+.4f}  PASS_S (>= +0.010, lower > 0): {P.pass_support(m, lo)}  PASS_N (<= +0.003 or upper < 0.010): {P.pass_null(m, hi)}")
        else:
            print(f"{s:9s} no kept images / file missing")
    label, ns, nn = P.verdict(by)
    print(f"PASS_S on {ns}/4, PASS_N on {nn}/4  ->  {label}")
    print("SUPPORTED authorises only a separately registered position-robustness study (not a method); NOT SUPPORTED: the EXP-033 association is not a position response of the frozen model.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pixel", "smoke", "report"))
    ap.add_argument("--name", default="data", help="ClinicDB|ColonDB|ISIC|Endo|Kvasir|TN3K")
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None, help="EXP-012 pixel_per_image_predictions.csv (SHAM parity, tol 1e-4)")
    ap.add_argument("--position_dir", default="results/EXP-032", help="holds <set>/position_<set>.csv and summary_<set>.csv (optional)")
    ap.add_argument("--out_dir", default="results/EXP-034")
    ap.add_argument("--report_dir", default="results/EXP-034")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_central", action="store_true", help="skip the central-tercile control (halves the forwards)")
    ap.add_argument("--sigma", type=int, default=4)
    ap.add_argument("--z_pix", type=float, default=1.6)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "smoke":
        a.limit = a.limit or 12
    {"pixel": stage_pixel, "smoke": stage_pixel, "report": stage_report}[a.mode](a)
