"""
EXP-048 / H037: is the frozen ECP position response (EXP-034/035) mediated by the absolute positional embedding (PE)?  Diagnostic, NOT a method.
Rule, arms, region, statistics: research/EXPERIMENTS.md EXP-048.  Statistics: pe_shift_stats.py; PE hook: pe_shift_core.py (both unit-tested).

  python analyze_pe_shift.py pixel --name ClinicDB --dataset colon --data_path $CVC/CVC-ClinicDB --checkpoint_path <ckpt> --reference_csv <EXP-012 csv>
  python analyze_pe_shift.py smoke --name ClinicDB ...      # = pixel with --limit 8 random kept images, out_dir results/EXP-048/smoke
  python analyze_pe_shift.py report --report_dir results/EXP-048   # no torch
One dataset per command, resumable (an existing output CSV is skipped unless --redo).
EXP-049 (H038): the same diagnostic for other checkpoints.  --checkpoint PATH, --tag NAME (outputs under results/EXP-049/<tag>/<set>/; no tag = EXP-048 paths),
--sets (subset; pixel: skip a set not listed, report: only these).  A checkpoint WITHOUT a conditioner (extent_cond none: CTRL, ZO) is detected automatically: z flags do not
apply and the text features come from the prompt learner only, exactly as test.py (conditioner None).  With --tag in RECORDED_POOLED_CLINICDB, the ClinicDB run first does a
pooled-AUROC parity pass over ALL kept ClinicDB images (unmodified SHAM map, sigma 4) and STOPs if it differs from the recorded value by more than 0.15 point.
Peripheral tercile (EXP-033 c): SHAM, PE_REC, PE_RAND, IMG_REC, IMG_RAND, PE_FLAT (6 forwards).  Central tercile: SHAM, PE_REC (2 forwards).
Everything is scored with per-image AUROC on the SAME pixel set (common valid region).  STOP: SHAM full-image AUROC vs EXP-012 reference > 1e-4.
"""
import os
import csv
import json
import hashlib
import time
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

import pos_shift_stats as P
import pe_shift_stats as S
import zpos_stats as Z

SETS = ["ClinicDB", "ColonDB", "Endo", "Kvasir", "TN3K", "ISIC"]
# EXP-049 pooled pixel AUROC (%) on ClinicDB at sigma 4, fixed z_pix 1.6 where there is a conditioner (one decimal as recorded; tolerance 0.15 point).
# Sources: ECP s111 EXP-012/EXP-048 (88.3), ECP s222/s333 and ZO s111/s222/s333 EXP-024 (research/EXPERIMENTS.md), CTRL s111/s222/s333 EXP-030 (exp030.txt).
RECORDED_POOLED_CLINICDB = {"ecp_s111": 88.3, "ecp_s222": 87.6, "ecp_s333": 87.2,
                            "zo_s111": 84.7, "zo_s222": 84.9, "zo_s333": 83.9,
                            "ctrl_s111": 82.6, "ctrl_s222": 83.0, "ctrl_s333": 81.8}
ARMS = ["sham", "pe_rec", "pe_rand", "img_rec", "img_rand", "pe_flat"]
COLS = ["id", "group", "c", "ci", "cj", "sy_rec", "sx_rec", "sy_rand", "sx_rand", "n_les_reg", "n_bg_reg", "status",
        "auroc_full_sham", "auroc_ref"] + ["auroc_" + a for a in ARMS]


def stage_pixel(args):
    import torch
    import AnomalyCLIP_lib
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features
    from metrics import roc_auc_lowmem
    from analyze_badcases import load_model, read_mask
    import pe_shift_core as C

    if args.sets and args.name not in args.sets:
        print(f"{args.name}: not in --sets {args.sets}, skipped"); return
    out_dir = os.path.join(args.out_dir, args.name)
    out_csv = os.path.join(out_dir, f"pe_{args.name}.csv")
    if os.path.exists(out_csv) and not args.redo:
        print(f"{args.name}: {out_csv} exists, skipped (use --redo)")
        return
    os.makedirs(out_dir, exist_ok=True)
    T0 = time.time()
    Sz = args.image_size
    if Sz != S.SIDE * S.PATCH:
        raise SystemExit(f"STOP: image size {Sz} != {S.SIDE}*{S.PATCH}")
    device, model, pl, cond = load_model(args)
    if cond is not None and cond.mode != "extent":
        raise SystemExit(f"STOP: conditioner mode '{cond.mode}' is not supported (need 'extent' ECP or no conditioner)")
    print(f"[stage] checkpoint {args.checkpoint_path}: " + ("ECP 'extent' conditioner (z_pix override %.2f)" % args.z_pix if cond is not None
          else "NO conditioner (static prompts from the prompt learner, as test.py); z flags ignored"))
    tf_static = None
    if cond is None:                                              # test.py: text_features_base, computed once (it does not depend on the image)
        with torch.no_grad():
            prompts, tokenized_prompts, compound_prompts_text = pl(cls_id=None)
            tf_static = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            tf_static = torch.stack(torch.chunk(tf_static, dim=0, chunks=2), dim=1)
            tf_static = tf_static / tf_static.norm(dim=-1, keepdim=True)
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    ref = None
    if args.reference_csv:
        ref = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(args.reference_csv))}
    else:
        print("WARNING: no --reference_csv, EXP-012 per-image parity NOT checked")

    # pass 1: kept images, lesion offset c from the mask
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
    print(f"[stage] {args.name}: {len(kept)} kept; peripheral {len(ip)} (c >= {c_all[ip].min():.3f}), central {len(ic)} (c <= {c_all[ic].max():.3f}); pass 1 {time.time() - t0:.0f}s")

    visual = model.visual

    def score_map(img, pe=None):
        """ECP pixel map exactly as analyze_position_shift.py; `pe` = None | ('shift', sy, sx) | ('flat',) modifies only the patch PE of this one forward
        (the per-image visual descriptor / text features are computed from the same modified forward)."""
        with torch.no_grad():
            def run():
                imf, pf = model.encode_image(img.unsqueeze(0), [24], DPAM_layer=20)
                imf = imf / imf.norm(dim=-1, keepdim=True)
                pfeat = pf[0] / pf[0].norm(dim=-1, keepdim=True)
                if cond is None:
                    tf = tf_static
                else:
                    desc = visual_descriptor(imf, pf)
                    zt = torch.full((imf.shape[0],), args.z_pix, device=device)
                    _, c_pos, c_neg = cond(desc, z_override=zt)
                    tf = conditioned_text_features(model, pl, c_pos, c_neg)
                sim, _ = AnomalyCLIP_lib.compute_similarity(pfeat, tf[0])
                sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], Sz)
                return ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu()
            if pe is None:
                amap = run()
            else:
                with C.patched_pe(visual, pe[0], *(pe[1:])):
                    amap = run()
        return gaussian_filter(amap.numpy(), sigma=args.sigma).astype(np.float32)

    # warm-up: first forward interpolates the PE 24x24 -> 37x37 inside the model; the PE must be 37x37 before any patched forward
    first = data[int(kept[0])]["img"].to(device)
    score_map(first)
    C.ensure_grid(visual, S.SIDE)
    pe0 = visual.positional_embedding.data.clone()
    print(f"[stage] PE grid {S.SIDE}x{S.SIDE} ready; shift_pe(0,0) bitwise equal to the stored PE: {bool(torch.equal(C.shift_pe(pe0, 0, 0)[0], pe0))}")
    pad_fill = torch.zeros(3, 1, 1, device=device)                   # CLIP-mean colour = 0 in the normalised tensor (EXP-035 arm MEAN)

    def translated(img, dx, dy):
        out = pad_fill.expand(3, Sz, Sz).clone()
        y0, y1, x0, x1 = max(0, -dy), min(Sz, Sz - dy), max(0, -dx), min(Sz, Sz - dx)
        if y1 > y0 and x1 > x0:
            out[:, y0 + dy:y1 + dy, x0 + dx:x1 + dx] = img[:, y0:y1, x0:x1]
        return out

    # EXP-049 pooled-AUROC parity on ALL kept ClinicDB images (unmodified SHAM map, sigma 4); STOP before any other work when it fails (a --limit run is informational only)
    pooled_info = None
    want_pooled = args.pooled_parity != "off" and args.name == "ClinicDB" and bool(args.tag or args.pooled_parity == "on")
    if want_pooled:
        if args.tag not in RECORDED_POOLED_CLINICDB:
            raise SystemExit(f"STOP: no recorded pooled ClinicDB AUROC for tag '{args.tag}' (known: {sorted(RECORDED_POOLED_CLINICDB)}); use --pooled_parity off to skip")
        t1 = time.time()
        Np = len(kept)
        pm = np.empty((Np, Sz, Sz), np.float32)
        pg = np.empty((Np, Sz, Sz), np.uint8)
        for n, i in enumerate(kept):
            itp = data[int(i)]
            pg[n] = read_mask(itp)
            pm[n] = score_map(itp["img"].to(device))
            if (n + 1) % 100 == 0:
                print(f"[pooled parity] {n + 1}/{Np} ({time.time() - t1:.0f}s)", flush=True)
        pooled = float(roc_auc_lowmem(pg, pm))
        del pm, pg
        rec = RECORDED_POOLED_CLINICDB[args.tag]
        diff, ok = S.pooled_parity(pooled, rec)
        limited = bool(args.limit)
        print(f"[parity] pooled ClinicDB AUROC (unmodified SHAM, sigma {args.sigma}, {Np} images{', LIMITED subset: informational only' if limited else ''}) = {pooled * 100:.3f}; "
              f"recorded {rec} ({args.tag}); |diff| = {diff:.3f} point (tol {S.POOLED_TOL_PTS}); {'OK' if ok else 'EXCEEDED'}  [{time.time() - t1:.0f}s]")
        pooled_info = dict(tag=args.tag, n_images=Np, pooled_auroc_pct=pooled * 100, recorded_pct=rec, abs_diff_pts=diff, tol_pts=S.POOLED_TOL_PTS, limited=limited, ok=ok)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(pooled_info, open(os.path.join(out_dir, "pooled_parity.json"), "w"), indent=1)
        if not ok and not limited:
            raise SystemExit(f"STOP: pooled ClinicDB AUROC {pooled * 100:.3f} vs recorded {rec} (> {S.POOLED_TOL_PTS} point): wrong checkpoint, or the load path differs from test.py")
    elif args.name == "ClinicDB" and args.tag:
        print("[parity] pooled ClinicDB parity NOT run (--pooled_parity off)")

    rows, max_ref, max_impl, n_drop, n_zero = [], 0.0, 0.0, 0, 0
    t0 = time.time()
    for n, (j, group) in enumerate(todo):
        it = data[int(kept[j])]
        gt = read_mask(it)
        sid = ids_all[j]
        ci, cj = S.patch_centroid(gt)
        sy, sx = S.rec_shift(gt)
        ry, rx = S.rand_shift(sid, sy, sx)
        row = dict(id=sid, group=group, c=c_all[j], ci=ci, cj=cj, sy_rec=sy, sx_rec=sx, sy_rand=ry, sx_rand=rx, n_les_reg=0, n_bg_reg=0, status="kept")
        for k in COLS[12:]:
            row[k] = float("nan")
        if group == "peripheral" and (sy, sx) == (0, 0):
            row["status"] = "dropped_zero_shift"; n_zero += 1; rows.append(row); continue   # nothing to move: carries no information on the ratio
        grids = [C.shift_pe(pe0, sy, sx)[1], C.shift_pe(pe0, ry, rx)[1]]
        reg = S.common_region(gt.shape, grids, [S.img_shift_px(sy, sx), S.img_shift_px(ry, rx)])
        n_les, n_bg = P.region_counts(gt, reg)
        row["n_les_reg"], row["n_bg_reg"] = n_les, n_bg
        if n_les < P.MIN_CLASS or n_bg < P.MIN_CLASS:
            row["status"] = "dropped_region"; n_drop += 1; rows.append(row); continue
        img = it["img"].to(device)
        maps = {"sham": score_map(img), "pe_rec": score_map(img, ("shift", sy, sx))}
        if group == "peripheral":
            maps["pe_rand"] = score_map(img, ("shift", ry, rx))
            maps["pe_flat"] = score_map(img, ("flat",))
            for arm, (a, b) in (("img_rec", (sy, sx)), ("img_rand", (ry, rx))):
                dx, dy = S.img_shift_px(a, b)
                maps[arm] = P.inverse_align(score_map(translated(img, dx, dy)), dx, dy)
        row["auroc_full_sham"] = float(roc_auc_lowmem(gt, maps["sham"]))
        if ref is not None:
            if sid not in ref:
                raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
            row["auroc_ref"] = ref[sid]
            err = abs(row["auroc_full_sham"] - ref[sid]); max_ref = max(max_ref, err)
            if err > 1e-4:
                raise SystemExit(f"STOP: SHAM parity failure on {sid}: {row['auroc_full_sham']:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
        max_impl = max(max_impl, abs(P.region_auroc(gt, maps["sham"], np.ones_like(gt, bool)) - row["auroc_full_sham"]))
        for arm, m in maps.items():
            row["auroc_" + arm] = P.region_auroc(gt, m, reg)
        rows.append(row)
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(todo)} images ({time.time() - t0:.0f}s, {(time.time() - t0) / (n + 1):.2f}s/image, dropped {n_drop}, zero-shift {n_zero})", flush=True)
    el = time.time() - t0
    n_fw = sum(1 for r in rows if r["status"] == "kept")
    print(f"[stage] {len(todo)} selected images: forwarded {n_fw}, dropped by the >= {P.MIN_CLASS}-pixel region filter {n_drop}, dropped for zero recentre shift {n_zero}; {el:.0f}s ({el / max(n_fw, 1):.2f} s per forwarded image)")
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
    with open(os.path.join(out_dir, f"meta_{args.name}.json"), "w") as f:
        json.dump(dict(checkpoint=args.checkpoint_path, sha256=hashlib.sha256(open(args.checkpoint_path, "rb").read()).hexdigest(), tag=args.tag,
                       conditioner=(cond.mode if cond is not None else None), limit=args.limit, n_rows=len(rows), n_forwarded=n_fw,
                       reference_csv=args.reference_csv, max_ref_err=max_ref if ref is not None else None, pooled_parity=pooled_info), f, indent=1)
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


def ci(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def set_stats(rows, boot):
    """-> (stats dict for the registered statistics on the peripheral tercile or None, printable extras)."""
    per = [r for r in rows if r["group"] == "peripheral"]
    k = [r for r in per if r["status"] == "kept"]
    info = dict(n_sel=len(per), n_kept=len(k), n_drop=sum(r["status"] == "dropped_region" for r in per), n_zero=sum(r["status"] == "dropped_zero_shift" for r in per))
    st = None
    if len(k) >= 3:
        a = lambda key: np.array([r[key] for r in k])
        st = S.boot_ratio_g(a("auroc_sham"), a("auroc_pe_rec"), a("auroc_pe_rand"), a("auroc_img_rec"), boot)
        info["flat_minus_sham"] = P.boot_mean(a("auroc_pe_flat") - a("auroc_sham"), boot)
        info["img_rand_minus_sham"] = P.boot_mean(a("auroc_img_rand") - a("auroc_sham"), boot)
        info["mean_shift_px"] = float(np.mean([14 * np.hypot(r["sy_rec"], r["sx_rec"]) for r in k]))
    cen = [r for r in rows if r["group"] == "central" and r["status"] == "kept"]
    if cen:
        info["central"] = (len(cen), P.boot_mean(np.array([r["auroc_pe_rec"] - r["auroc_sham"] for r in cen]), boot))
    return st, info


def report_set(rows, name, boot):
    st, info = set_stats(rows, boot)
    tag = "DECISION" if name in S.DECISION else "report-only"
    print(f"\n=== {name} ({tag}) ===")
    print(f"  peripheral: selected {info['n_sel']}, dropped by region filter {info['n_drop']}, dropped for zero shift {info['n_zero']}, kept {info['n_kept']}")
    if st is None:
        print("  too few kept images"); return None
    print(f"  D_IMG = {st['D_IMG']:+.4f}   D_PE = {st['D_PE']:+.4f}   D_PErand = {st['D_PErand']:+.4f}   VALID (D_IMG >= +0.020): {S.is_valid(st['D_IMG'])}   mean recentre shift {info['mean_shift_px']:.0f} px")
    print(f"  R = D_PE / D_IMG = {st['R']:+.3f} [{st['R_ci'][0]:+.3f}, {st['R_ci'][1]:+.3f}]   G = mean(PE_REC - PE_RAND) = {st['G']:+.4f} [{st['G_ci'][0]:+.4f}, {st['G_ci'][1]:+.4f}]")
    print(f"  report-only: PE_FLAT - SHAM {ci(info['flat_minus_sham'])}   IMG_RAND - SHAM {ci(info['img_rand_minus_sham'])}")
    if "central" in info:
        n, b = info["central"]
        print(f"  central tercile (n={n}): PE_REC - SHAM {ci(b)}")
    return st


def stage_report(args):
    res = {}
    for s in (args.sets or SETS):
        p = os.path.join(args.report_dir, s, f"pe_{s}.csv")
        if not os.path.isfile(p):
            print(f"MISSING {p}"); continue
        res[s] = report_set(load_rows(p), s, args.boot)
    print("\n================ VERDICT (rule fixed in research/EXPERIMENTS.md EXP-048; decision sets ClinicDB, ColonDB, Endo; peripheral tercile) ================")
    for s in S.DECISION:
        v = res.get(s)
        if v is None:
            print(f"{s:9s} missing / too few images"); continue
        print(f"{s:9s} D_IMG {v['D_IMG']:+.4f} VALID {S.is_valid(v['D_IMG'])}  R {v['R']:+.3f}  G_lo {v['G_ci'][0]:+.4f}  pass_PE {S.set_pass_pe(v)}  pass_NOT {S.set_pass_not(v)}")
    label, nv, npe, nno = S.verdict(res)
    print(f"VALID decision sets {nv}/3, PE-pass {npe}, NOT-pass {nno}  ->  {label}")
    print("PE-MEDIATED authorises only a separately registered, label-free PE modification study; NOT PE-MEDIATED: the response is carried by content/field-of-view/context; MIXED: report R per set.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pixel", "smoke", "report"))
    ap.add_argument("--name", default="data", help="ClinicDB|ColonDB|Endo|Kvasir|TN3K|ISIC")
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None, help="EXP-012 pixel_per_image_predictions.csv (SHAM parity, tol 1e-4)")
    ap.add_argument("--position_dir", default="results/EXP-032", help="holds <set>/position_<set>.csv (optional cross-check of c)")
    ap.add_argument("--out_dir", default=None, help="default results/EXP-049/<tag> with --tag, else results/EXP-048")
    ap.add_argument("--report_dir", default=None, help="default as --out_dir")
    ap.add_argument("--tag", default=None, help="EXP-049 checkpoint tag (ecp_s222, ctrl_s111, zo_s333, ...): output folder and recorded pooled value")
    ap.add_argument("--sets", nargs="+", default=None, help="subset of ClinicDB ColonDB Endo Kvasir TN3K ISIC (pixel: other sets are skipped; report: only these are read)")
    ap.add_argument("--pooled_parity", choices=("auto", "on", "off"), default="auto", help="ClinicDB pooled-AUROC parity pass; auto = on when --tag is given")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_central", action="store_true", help="skip the central-tercile control")
    ap.add_argument("--sigma", type=int, default=4)
    ap.add_argument("--z_pix", type=float, default=1.6)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    default_dir = f"results/EXP-049/{a.tag}" if a.tag else "results/EXP-048"
    a.out_dir = a.out_dir or default_dir
    a.report_dir = a.report_dir or default_dir
    if a.mode == "smoke":
        a.limit = a.limit or 8
    {"pixel": stage_pixel, "smoke": stage_pixel, "report": stage_report}[a.mode](a)
