"""
EXP-027 / H018: headroom of the extent coordinate z at inference, and is the pooled-AUROC gap between-image calibration?
Frozen model, forward passes only, no training.  GT is used ONLY by the oracle arms (ORACLE_BEST, BGALIGN); no method is implied.

Per kept image (mask >= 20 px and background >= 20 px): ONE encoder forward, then for each z in Z = {-0.69, 0.0, 0.8, 1.6, 2.4} text features
are recomputed with the conditioner's z override and the map is built exactly as test.py (--features_list 24, bilinear 518x518 similarity map,
Gaussian sigma 4, float32).  Arms: FIXED = M_1.6; UNIFORM5 = mean of the five maps; UNIFORM3 = mean over z in {0.8, 1.6, 2.4};
ORACLE_BEST = per image the best of the five per-image AUROC_z (per-image only).  Calibration (FIXED only): pooled AUROC of FIXED,
MEDALIGN (map - own median over all pixels; label-free, diagnostic only), BGALIGN (map - own median over pixels > 28 px from the lesion;
uses GT; falls back to all background pixels if none).  Pre-registered rules R_Z / R_U / R_CAL: research/EXPERIMENTS.md EXP-027.

Memory: per dataset only FIXED, UNIFORM5-accumulator, UNIFORM3-accumulator (float32) and the uint8 masks are kept; the alignment buffer reuses
the UNIFORM5 memory after it is freed (peak ~3 float32 copies of the dataset).

STOP conditions: FIXED per-image AUROC vs --reference_csv (EXP-012) > 1e-4 on any kept image; FIXED pooled AUROC/AUPRO vs the recorded sigma-4
values > 0.1 point (when --name is known, no --limit, no --no_recorded_check); single-map "uniform" != FIXED on the first image.

One command per dataset:
  python analyze_zens.py eval --name ClinicDB --data_path $CVC/CVC-ClinicDB --dataset colon \
     --checkpoint_path checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth --reference_csv <EXP-012 pixel_per_image_predictions.csv> \
     --out_csv zens_ClinicDB.csv
Aggregate:
  python analyze_zens.py --aggregate zens_ClinicDB.csv zens_Kvasir.csv zens_ColonDB.csv zens_ISIC.csv zens_Endo.csv zens_TN3K.csv
"""
import os
import csv
import time
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

from headroom_stats import quartile_index
from zens_stats import (Z_GRID, FIXED_Z, UNI3_Z, best_of_k, win_hist, mean_maps, median_offset, bg_offset, paired_bootstrap, closure,
                        verdict_rz, gain_loss, verdict_cal, CLOSE_BELOW, PURSUE_AT, GAIN_MARGIN, BETWEEN_AT, WITHIN_BELOW)

# recorded sigma-4 pooled values of the matched ECP, seed 111, z_pix 1.6 (AUROC %, AUPRO %)
RECORDED = {"ClinicDB": (88.3, 76.4), "Kvasir": (88.9, 51.8), "ColonDB": (85.0, 72.9), "ISIC": (93.9, 88.7), "Endo": (91.6, 79.6), "TN3K": (82.5, 52.0)}
ZCOLS = [f"auroc_z{z}" for z in Z_GRID]
Z_CTRL = (1.5, 1.55, 1.6, 1.65, 1.7)       # selection-floor control: best-of-5 over near-identical z (no real extent difference)
CCOLS = [f"auroc_ctrl_z{z}" for z in Z_CTRL]
IDX_FIXED = Z_GRID.index(FIXED_Z)
IDX_U3 = [Z_GRID.index(z) for z in UNI3_Z]


def load_model(args):
    import torch
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from extent_prompt import ExtentConditioner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    pl = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    pl.load_state_dict(ckpt["prompt_learner"])
    pl.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    cond = None
    if ckpt.get("extent_cond"):
        cond = ExtentConditioner(mode=ckpt["extent_cond"]).to(device)
        cond.load_state_dict(ckpt["conditioner"])
        cond.eval()
    return device, model, pl, cond


def read_mask(it):
    m = it["img_mask"]
    return (m[0].numpy() if m.ndim == 3 else m.numpy()) > 0.5


def stage_eval(args):
    import torch
    import AnomalyCLIP_lib
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features
    from metrics import roc_auc_lowmem, cal_pro_score

    device, model, pl, cond = load_model(args)
    if cond is None or cond.mode != "extent":
        raise SystemExit("STOP: this diagnostic needs the 'extent' ECP checkpoint (z override)")
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    ref = None
    if args.reference_csv:
        ref = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(args.reference_csv))}
    else:
        print("WARNING: no --reference_csv, EXP-012 parity NOT checked")

    t0 = time.time()
    kept, areas = [], []                      # pass 1: GT areas (no forward) -> within-dataset quartiles, as analyze_headroom.py
    for i in range(len(data)):
        gt = read_mask(data[i])
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        kept.append(i); areas.append(float(gt.mean()))
    if args.limit:
        sel = np.random.RandomState(0).permutation(len(kept))[: args.limit]
        kept = [kept[j] for j in sel]; areas = [areas[j] for j in sel]
        print(f"--limit: {len(kept)} random kept images (smoke; recorded-value checks off)")
    areas = np.array(areas)
    q = quartile_index(areas)
    N, S = len(kept), args.image_size
    print(f"{args.name}: {N} kept images, pass 1 {time.time() - t0:.0f}s")

    base = np.empty((N, S, S), np.float32)    # FIXED = M_1.6
    acc5 = np.zeros((N, S, S), np.float32)
    acc3 = np.zeros((N, S, S), np.float32)
    gt_all = np.empty((N, S, S), np.uint8)
    tab = np.empty((N, len(Z_GRID)), np.float64)
    ctab = np.full((N, len(Z_CTRL)), np.nan)
    a_u5, a_u3, ids = np.empty(N), np.empty(N), []
    max_err = 0.0
    t0 = time.time()
    for n, i in enumerate(kept):
        it = data[int(i)]
        gt = read_mask(it)
        sid = os.path.relpath(it["img_path"], args.data_path)
        img = it["img"].unsqueeze(0).to(device)
        maps, cmaps = [], {}
        with torch.no_grad():
            imf, pf = model.encode_image(img, [24], DPAM_layer=20)            # ONE encoder forward
            imf = imf / imf.norm(dim=-1, keepdim=True)
            pfeat = pf[0] / pf[0].norm(dim=-1, keepdim=True)
            desc = visual_descriptor(imf, pf)
            for zv in (Z_GRID + tuple(z for z in Z_CTRL if z != FIXED_Z) if args.ctrl else Z_GRID):
                zt = torch.full((imf.shape[0],), zv, device=device)
                _, c_pos, c_neg = cond(desc, z_override=zt)
                tf = conditioned_text_features(model, pl, c_pos, c_neg)
                sim, _ = AnomalyCLIP_lib.compute_similarity(pfeat, tf[0])
                sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], S)
                amap = ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu()
                mm = gaussian_filter(amap.numpy(), sigma=args.sigma).astype(np.float32)           # test.py: gaussian on float32
                if zv in Z_GRID:
                    maps.append(mm)
                else:
                    cmaps[zv] = mm
        for k, m in enumerate(maps):
            tab[n, k] = float(roc_auc_lowmem(gt, m))
        m_fix = maps[IDX_FIXED]
        if args.ctrl:
            cmaps[FIXED_Z] = m_fix
            for k, zv in enumerate(Z_CTRL):
                ctab[n, k] = float(roc_auc_lowmem(gt, cmaps[zv]))
            del cmaps
        if n == 0:                                                          # single-z uniform must reproduce FIXED exactly
            if not np.array_equal(mean_maps([m_fix]), m_fix):
                raise SystemExit("STOP: single-map uniform ensemble != FIXED")
        if ref is not None:
            if sid not in ref:
                raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
            err = abs(tab[n, IDX_FIXED] - ref[sid]); max_err = max(max_err, err)
            if err > 1e-4:
                raise SystemExit(f"STOP: parity failure on {sid}: {tab[n, IDX_FIXED]:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
        base[n] = m_fix
        acc5[n] = np.mean(maps, axis=0, dtype=np.float32)
        acc3[n] = np.mean([maps[k] for k in IDX_U3], axis=0, dtype=np.float32)
        gt_all[n] = gt
        a_u5[n] = float(roc_auc_lowmem(gt, acc5[n])); a_u3[n] = float(roc_auc_lowmem(gt, acc3[n]))
        ids.append(sid)
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{N} images ({time.time() - t0:.0f}s)", flush=True)
    t_fwd = time.time() - t0
    print(f"{N} images done in {t_fwd:.0f}s; parity max |FIXED per-image AUROC - EXP-012| = {max_err:.2e}" + ("" if ref else " (NOT CHECKED)"))

    out = {"n_images": N, "parity_max_err": max_err, "t_forward_s": t_fwd}
    check = args.name in RECORDED and not args.no_recorded_check and not args.limit

    def pooled(arm, m):
        t = time.time()
        out[f"pooled_{arm}"] = float(roc_auc_lowmem(gt_all, m))
        line = f"pooled AUROC {arm} = {out[f'pooled_{arm}']:.4f}"
        if arm in ("FIXED", "UNIFORM5", "UNIFORM3") and args.pro:
            tp = time.time(); out[f"aupro_{arm}"] = float(cal_pro_score(gt_all, m))
            line += f"  AUPRO = {out[f'aupro_{arm}']:.4f} (PRO {time.time() - tp:.0f}s)"
        print(line + f"  ({time.time() - t:.0f}s)", flush=True)

    pooled("FIXED", base)
    if check:
        if abs(out["pooled_FIXED"] * 100 - RECORDED[args.name][0]) > 0.1:
            raise SystemExit(f"STOP: FIXED pooled AUROC {out['pooled_FIXED'] * 100:.2f} vs recorded {RECORDED[args.name][0]} (> 0.1 point)")
        if args.pro and abs(out["aupro_FIXED"] * 100 - RECORDED[args.name][1]) > 0.1:
            raise SystemExit(f"STOP: FIXED AUPRO {out['aupro_FIXED'] * 100:.2f} vs recorded {RECORDED[args.name][1]} (> 0.1 point)")
    pooled("UNIFORM3", acc3); del acc3
    pooled("UNIFORM5", acc5)
    buf = acc5                                 # reuse the memory for the alignment arms (built one at a time)
    for j in range(N):
        buf[j] = base[j] - median_offset(base[j])
    pooled("MEDALIGN", buf)
    n_fb = 0
    for j in range(N):
        o, fb = bg_offset(gt_all[j] > 0, base[j], far=args.far)
        n_fb += int(fb); buf[j] = base[j] - o
    pooled("BGALIGN", buf)
    out["n_bg_fallback"] = n_fb
    del buf

    _, win = best_of_k(tab)
    with open(args.out_csv.replace(".csv", "_images.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "image", "area_frac", "quartile"] + ZCOLS + ["auroc_UNIFORM5", "auroc_UNIFORM3"] + CCOLS)
        for j in range(N):
            w.writerow([args.name, ids[j], areas[j], f"Q{q[j] + 1}"] + list(tab[j]) + [a_u5[j], a_u3[j]] + list(ctab[j]))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "key", "value"])
        for k, v in out.items():
            w.writerow([args.name, k, v])
    print(f"wrote {args.out_csv} (+ _images.csv); total {time.time() - t0:.0f}s")
    report_dataset(args.name, read_summary(args.out_csv), read_images(args.out_csv.replace(".csv", "_images.csv")))


# ------------------------------------------------------------------ report
def read_summary(path):
    return {r["key"]: float(r["value"]) for r in csv.DictReader(open(path))}


def read_images(path):
    rows = list(csv.DictReader(open(path)))
    cols = ZCOLS + ["auroc_UNIFORM5", "auroc_UNIFORM3", "area_frac"] + (CCOLS if CCOLS[0] in rows[0] else [])
    d = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    d["quartile"] = np.array([int(r["quartile"][1]) - 1 for r in rows])
    d["image"] = [r["image"] for r in rows]
    return d


def dataset_numbers(im):
    tab = np.stack([im[c] for c in ZCOLS], axis=1)
    fixed = tab[:, IDX_FIXED]
    best, win = best_of_k(tab)
    res = {"fixed": fixed.mean(), "u5": im["auroc_UNIFORM5"].mean(), "u3": im["auroc_UNIFORM3"].mean(), "best": best.mean(),
           "gap": best.mean() - fixed.mean(), "win": win, "tab": tab,
           "d_u5": paired_bootstrap(im["auroc_UNIFORM5"] - fixed), "d_u3": paired_bootstrap(im["auroc_UNIFORM3"] - fixed),
           "d_best": paired_bootstrap(best - fixed)}
    res["per_z"] = tab.mean(axis=0)
    ct = np.stack([im[c] for c in CCOLS], axis=1) if CCOLS[0] in im else None
    res["ctrl_gap"] = float((ct.max(axis=1) - ct[:, Z_CTRL.index(FIXED_Z)]).mean()) if ct is not None and np.isfinite(ct).all() else float("nan")
    return res


def report_dataset(name, s, im):
    r = dataset_numbers(im)
    print(f"\n=== {name} (n={int(s['n_images'])}) ===")
    print("mean per-image AUROC per z: " + "  ".join(f"z={z}: {v:.4f}" for z, v in zip(Z_GRID, r["per_z"])))
    print(f"mean per-image AUROC: FIXED {r['fixed']:.4f}  UNIFORM5 {r['u5']:.4f}  UNIFORM3 {r['u3']:.4f}  ORACLE_BEST {r['best']:.4f}")
    for k, nm in (("d_u5", "UNIFORM5-FIXED"), ("d_u3", "UNIFORM3-FIXED"), ("d_best", "ORACLE_BEST-FIXED")):
        m, lo, hi = r[k]
        print(f"  {nm:18s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (f"   R_U: {gain_loss(m, lo, hi)}" if k != "d_best" else ""))
    print(f"selection-floor control (best-of-5 over z in {Z_CTRL}, no real extent difference): gap {r['ctrl_gap']:.4f}  "
          f"(oracle gap {r['gap']:.4f}; ratio {r['ctrl_gap'] / r['gap'] if r['gap'] > 0 else float('nan'):.2f})")
    print("winning z (count; rows: quartile of GT area, Q1 smallest):")
    print("   " + "  ".join(f"{'z=' + str(z):>7s}" for z in Z_GRID) + "    mean(best-fixed)")
    for qi in range(4):
        sel = im["quartile"] == qi
        h = win_hist(r["win"][sel], len(Z_GRID))
        gq = (r["tab"][sel].max(axis=1) - r["tab"][sel][:, IDX_FIXED]).mean()
        print(f"Q{qi + 1} " + "  ".join(f"{c:7d}" for c in h) + f"    {gq:.4f}  (n={int(sel.sum())})")
    h = win_hist(r["win"], len(Z_GRID))
    print("ALL" + "  ".join(f"{c:7d}" for c in h))
    pa = lambda k: s.get(k, float("nan"))
    print(f"pooled AUROC / AUPRO: FIXED {pa('pooled_FIXED'):.4f} / {pa('aupro_FIXED'):.4f}   UNIFORM5 {pa('pooled_UNIFORM5'):.4f} / {pa('aupro_UNIFORM5'):.4f}"
          f"   UNIFORM3 {pa('pooled_UNIFORM3'):.4f} / {pa('aupro_UNIFORM3'):.4f}")
    cm, cb = closure(pa("pooled_MEDALIGN"), pa("pooled_FIXED")), closure(pa("pooled_BGALIGN"), pa("pooled_FIXED"))
    print(f"calibration (FIXED): pooled {pa('pooled_FIXED'):.4f}  MEDALIGN {pa('pooled_MEDALIGN'):.4f} (closure {cm:+.3f})  "
          f"BGALIGN {pa('pooled_BGALIGN'):.4f} (closure {cb:+.3f}, bg-fallback images {int(pa('n_bg_fallback'))})")
    print(f"mean per-image AUROC - pooled AUROC (FIXED) = {r['fixed'] - pa('pooled_FIXED'):+.4f}")


def aggregate(paths):
    gaps, cgaps, clos, ru = {}, {}, {}, {}
    for p in paths:
        s = read_summary(p); im = read_images(p.replace(".csv", "_images.csv"))
        name = os.path.basename(p)[len("zens_"):-len(".csv")] if os.path.basename(p).startswith("zens_") else p
        report_dataset(name, s, im)
        r = dataset_numbers(im)
        gaps[name] = r["gap"]; cgaps[name] = r["ctrl_gap"]
        clos[name] = closure(s["pooled_BGALIGN"], s["pooled_FIXED"])
        ru[name] = (gain_loss(*r["d_u5"]), gain_loss(*r["d_u3"]))
    print("\n================ VERDICTS (rules fixed in research/EXPERIMENTS.md EXP-027) ================")
    print("R_Z  ORACLE_BEST - FIXED (mean per-image AUROC): " + "  ".join(f"{k} {v:+.4f}" for k, v in gaps.items()))
    print(f"     <{CLOSE_BELOW} on {sum(v < CLOSE_BELOW - 1e-12 for v in gaps.values())}/{len(gaps)}; >={PURSUE_AT} on {sum(v >= PURSUE_AT - 1e-12 for v in gaps.values())}/{len(gaps)}"
          f"  ->  R_Z = {verdict_rz(gaps)}  (CLOSE = z is not a worthwhile axis for inference; PURSUE = headroom exists, still an upper bound)")
    nsel = sum(np.isfinite(cgaps[k]) and gaps[k] > 0 and cgaps[k] >= 0.5 * gaps[k] for k in gaps)
    print("     control gap (best-of-5 over near-identical z): " + "  ".join(f"{k} {v:+.4f}" for k, v in cgaps.items()) +
          f"  -> control >= 50% of the oracle gap on {nsel}/{len(gaps)} sets" + ("  (SELECTION-BIAS WARNING: a PURSUE would be mostly best-of-K selection)" if nsel >= 4 else ""))
    for j, nm in enumerate(("UNIFORM5", "UNIFORM3")):
        g = sum(v[j] == "GAIN" for v in ru.values()); l = sum(v[j] == "LOSS" for v in ru.values())
        print(f"R_U  {nm}: GAINS on {g}/{len(ru)}, LOSES on {l}/{len(ru)}   (" + ", ".join(f"{k}:{v[j]}" for k, v in ru.items()) + ")  [report only, not a module claim]")
    print("R_CAL BGALIGN closure of the pooled gap: " + "  ".join(f"{k} {v:+.3f}" for k, v in clos.items()))
    print(f"      >={BETWEEN_AT} on {sum(v >= BETWEEN_AT - 1e-12 for v in clos.values() if np.isfinite(v))}/{len(clos)}; <{WITHIN_BELOW} on "
          f"{sum(v < WITHIN_BELOW - 1e-12 for v in clos.values() if np.isfinite(v))}/{len(clos)}  ->  R_CAL = {verdict_cal(clos)}")
    print("      compare: EXP-025 CORE_FIX closures 0.80-0.84 (pooled, oracle). MEDALIGN and BGALIGN are diagnostics; cross-image calibration is not a method here.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=("eval",))
    ap.add_argument("--aggregate", nargs="+", default=None)
    ap.add_argument("--name", default="data", help="ClinicDB | Kvasir | ColonDB | ISIC | Endo | TN3K (selects the recorded-value parity check)")
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None, help="EXP-012 pixel_per_image_predictions.csv (parity, tol 1e-4)")
    ap.add_argument("--no_recorded_check", action="store_true")
    ap.add_argument("--no_pro", dest="pro", action="store_false", help="skip AUPRO (default: computed for FIXED, UNIFORM5, UNIFORM3)")
    ap.add_argument("--no_ctrl", dest="ctrl", action="store_false", help="skip the selection-floor control (4 extra text encodings per image)")
    ap.add_argument("--sigma", type=int, default=4)
    ap.add_argument("--far", type=float, default=28)
    ap.add_argument("--out_csv", default="zens.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate)
    elif a.mode == "eval":
        stage_eval(a)
    else:
        ap.error("give eval | --aggregate")
