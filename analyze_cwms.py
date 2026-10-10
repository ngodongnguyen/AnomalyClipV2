"""
EXP-036 / H026: centre-weighted multi-shift fusion (CWMS), a LABEL-FREE test-time rule on the frozen matched ECP. Inference only: no training, no target label at inference.
Rule, controls, thresholds: research/EXPERIMENTS.md EXP-036 (fixed before the run; do not tune).  Statistics: cwms_stats.py (unit-tested).

  python analyze_cwms.py eval --name ClinicDB --dataset colon --data_path $CVC/CVC-ClinicDB --checkpoint_path <ckpt> --reference_csv <EXP-012 csv> --out_dir results/EXP-036
  python analyze_cwms.py smoke --name ClinicDB ...             # = eval with --limit 12 random kept images, recorded-value checks off, out_dir results/EXP-036/smoke
  python analyze_cwms.py report --report_dir results/EXP-036   # no torch; per-set report and the verdict
One dataset per command, resumable (an existing summary CSV is skipped unless --redo).

Per kept image (mask >= 20 px and background >= 20 px): 9 forwards on the normalised 3x518x518 tensor: identity, four 130 px translations (CWMS, UNI5) and four 14 px
translations (SMALL5), translations with CLIP-mean fill (0 in the normalised tensor, as EXP-035 arm MEAN).  Each view's ECP pixel map is built exactly as
analyze_position_shift.py / test.py (z_pix 1.6, sigma 4, layer 24) and inverse-translated.  Arms: FIXED (identity), CWMS (centre-weighted), UNI5 (same 5 views, uniform on valid
pixels), SMALL5 (identity + 4 x 14 px, uniform on valid pixels), SIGMA32 (identity + extra Gaussian sqrt(32^2 - 4^2)).  All arms are scored on the full canvas.
STOP conditions: identity per-image AUROC vs --reference_csv (EXP-012) > 1e-4 on any kept image; pooled FIXED AUROC/AUPRO vs the recorded sigma-4 values > 0.1 point
(known --name, no --limit, no --no_recorded_check); torch fill-shift != numpy reference; single-view fusion != identity map.
Memory: per dataset FIXED, CWMS, UNI5 maps (float32) and uint8 masks; SIGMA32 pooled (--pooled_sigma32) reuses the UNI5 buffer.  SMALL5 is per-image only.
"""
import os
import csv
import time
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

import pos_shift_stats as P
import zpos_stats as Z
import cwms_stats as C
from headroom_stats import quartile_index

# recorded sigma-4 pooled values of the matched ECP, seed 111, z_pix 1.6 (AUROC %, AUPRO %)
RECORDED = {"ClinicDB": (88.3, 76.4), "Kvasir": (88.9, 51.8), "ColonDB": (85.0, 72.9), "ISIC": (93.9, 88.7), "Endo": (91.6, 79.6), "TN3K": (82.5, 52.0)}
IMG_COLS = ["id", "c", "area_frac", "quartile"] + [f"auroc_{a}" for a in C.ARMS]
S_PIX = 518


def stage_eval(args):
    import torch
    import AnomalyCLIP_lib
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features
    from metrics import roc_auc_lowmem, cal_pro_score
    from analyze_badcases import load_model, read_mask

    out_dir = os.path.join(args.out_dir, args.name)
    sum_csv = os.path.join(out_dir, f"cwms_{args.name}.csv")
    img_csv = os.path.join(out_dir, f"cwms_{args.name}_images.csv")
    if os.path.exists(sum_csv) and not args.redo:
        print(f"{args.name}: {sum_csv} exists, skipped (use --redo)")
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

    t0 = time.time()                                   # pass 1: masks only (no forward)
    kept, areas, c_all, ids = [], [], [], []
    for i in range(len(data)):
        it = data[i]
        gt = read_mask(it)
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        kept.append(i); areas.append(float(gt.mean())); c_all.append(Z.centroid_offset(gt)); ids.append(os.path.relpath(it["img_path"], args.data_path))
    if args.limit:
        sel = np.random.RandomState(0).permutation(len(kept))[: args.limit]
        kept = [kept[j] for j in sel]; areas = [areas[j] for j in sel]; c_all = [c_all[j] for j in sel]; ids = [ids[j] for j in sel]
        print(f"--limit: {len(kept)} random kept images (smoke; recorded-value checks off; terciles/quartiles are taken inside this subset)")
    areas, c_all = np.array(areas), np.array(c_all)
    quart = quartile_index(areas)
    N = len(kept)
    pos_csv = os.path.join(args.position_dir, args.name, f"position_{args.name}.csv")
    if os.path.isfile(pos_csv):
        pc = {r["id"]: float(r["c"]) for r in csv.DictReader(open(pos_csv))}
        dif = [abs(pc[s] - c) for s, c in zip(ids, c_all) if s in pc]
        print(f"[stage] c vs EXP-033 position csv: {len(dif)}/{N} joined, max |diff| = {max(dif) if dif else float('nan'):.2e}")
        if dif and max(dif) > 1e-6:
            raise SystemExit("STOP: recomputed lesion offset differs from position_<set>.csv")
    n_fw = 9 * N
    print(f"[stage] {args.name}: {N} kept images, pass 1 {time.time() - t0:.0f}s.  Cost: 9 forwards per image (CWMS/UNI5 use 5, SMALL5 adds 4) = {n_fw} forwards;"
          f" at the EXP-034 rate (about 0.19 s per forward, NOT measured for this script) about {n_fw * 0.19 / 60:.0f} min")

    def score_map(img):                                # copy of analyze_position_shift.score_map
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

    pad_fill = torch.zeros(3, 1, 1, device=device)     # EXP-035 arm MEAN: CLIP-mean colour = 0 in the normalised tensor

    def shifted(img, dx, dy):                          # copy of analyze_position_shift.shifted, pad == 'mean'
        if dx == 0 and dy == 0:
            return img
        out = pad_fill.expand(3, S, S).clone()
        y0, y1, x0, x1 = max(0, -dy), min(S, S - dy), max(0, -dx), min(S, S - dx)
        if y1 > y0 and x1 > x0:
            out[:, y0 + dy:y1 + dy, x0 + dx:x1 + dx] = img[:, y0:y1, x0:x1]
        return out

    base = np.empty((N, S, S), np.float32)             # FIXED
    cw = np.empty((N, S, S), np.float32)               # CWMS
    un = np.empty((N, S, S), np.float32)               # UNI5
    gt_all = np.empty((N, S, S), np.uint8)
    au = {a: np.empty(N) for a in C.ARMS}
    max_err = 0.0
    t0 = time.time()
    for n, i in enumerate(kept):
        it = data[int(i)]
        gt = read_mask(it)
        img = it["img"].to(device)
        if n == 0:                                     # torch fill-shift must equal the tested numpy reference
            ref_t = P.shift_image_fill(img.cpu().numpy(), 130, -130, 0.0)
            if not np.array_equal(shifted(img, 130, -130).cpu().numpy(), ref_t):
                raise SystemExit("STOP: torch fill-shift != pos_shift_stats.shift_image_fill")
        mp = {(0, 0): score_map(img)}
        for dx, dy in C.CWMS_VIEWS[1:] + C.SMALL_VIEWS[1:]:
            mp[(dx, dy)] = score_map(shifted(img, dx, dy))
        m_id = mp[(0, 0)]
        if n == 0 and not np.allclose(C.cwms_map([m_id], [(0, 0)]), m_id, rtol=0, atol=1e-6):
            raise SystemExit("STOP: single-view fusion != identity map")
        m_cw = C.cwms_map([mp[v] for v in C.CWMS_VIEWS], C.CWMS_VIEWS).astype(np.float32)
        m_un = C.uniform_map([mp[v] for v in C.CWMS_VIEWS], C.CWMS_VIEWS).astype(np.float32)
        m_sm = C.uniform_map([mp[v] for v in C.SMALL_VIEWS], C.SMALL_VIEWS).astype(np.float32)
        m_s32 = C.sigma32_map(m_id)
        sid = ids[n]
        au["FIXED"][n] = float(roc_auc_lowmem(gt, m_id))
        if ref is not None:
            if sid not in ref:
                raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
            err = abs(au["FIXED"][n] - ref[sid]); max_err = max(max_err, err)
            if err > 1e-4:
                raise SystemExit(f"STOP: identity parity failure on {sid}: {au['FIXED'][n]:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
        au["CWMS"][n] = float(roc_auc_lowmem(gt, m_cw)); au["UNI5"][n] = float(roc_auc_lowmem(gt, m_un))
        au["SMALL5"][n] = float(roc_auc_lowmem(gt, m_sm)); au["SIGMA32"][n] = float(roc_auc_lowmem(gt, m_s32))
        base[n] = m_id; cw[n] = m_cw; un[n] = m_un; gt_all[n] = gt
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{N} images ({time.time() - t0:.0f}s)", flush=True)
    t_fwd = time.time() - t0
    print(f"[stage] forward+fuse {N} images {t_fwd:.0f}s ({t_fwd / max(N, 1):.2f}s per image, 9 forwards each)")
    print(f"[parity] max |identity per-image AUROC - EXP-012| = {max_err:.2e}" + ("" if ref is not None else " (NOT CHECKED)"))

    out = {"n_images": N, "parity_max_err": max_err, "t_forward_s": t_fwd}
    check = args.name in RECORDED and not args.no_recorded_check and not args.limit

    def pooled(arm, m, with_pro=True):
        t = time.time()
        out[f"pooled_{arm}"] = float(roc_auc_lowmem(gt_all, m))
        line = f"[pooled] {arm:8s} AUROC = {out[f'pooled_{arm}']:.4f} ({time.time() - t:.0f}s)"
        if args.pro and with_pro:
            tp = time.time(); out[f"aupro_{arm}"] = float(cal_pro_score(gt_all, m))
            line += f"  AUPRO = {out[f'aupro_{arm}']:.4f} (PRO {time.time() - tp:.0f}s)"
        print(line, flush=True)

    pooled("FIXED", base)
    if check:
        if abs(out["pooled_FIXED"] * 100 - RECORDED[args.name][0]) > 0.1:
            raise SystemExit(f"STOP: FIXED pooled AUROC {out['pooled_FIXED'] * 100:.2f} vs recorded {RECORDED[args.name][0]} (> 0.1 point)")
        if args.pro and abs(out["aupro_FIXED"] * 100 - RECORDED[args.name][1]) > 0.1:
            raise SystemExit(f"STOP: FIXED AUPRO {out['aupro_FIXED'] * 100:.2f} vs recorded {RECORDED[args.name][1]} (> 0.1 point)")
    pooled("CWMS", cw)
    pooled("UNI5", un)
    if args.pooled_sigma32:
        for j in range(N):                             # reuse the UNI5 buffer
            un[j] = C.sigma32_map(base[j])
        pooled("SIGMA32", un)
    del cw, un

    tmp = img_csv + ".partial"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(IMG_COLS)
        for j in range(N):
            w.writerow([ids[j], c_all[j], areas[j], f"Q{quart[j] + 1}"] + [au[a][j] for a in C.ARMS])
    os.replace(tmp, img_csv)
    tmp = sum_csv + ".partial"                         # the summary marks the set as done
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "key", "value"])
        for k, v in out.items():
            w.writerow([args.name, k, v])
    os.replace(tmp, sum_csv)
    print(f"wrote {sum_csv} (+ _images.csv)")
    report_set(args.name, read_summary(sum_csv), read_images(img_csv), args.boot)
    print(f"[wall] {args.name} {time.time() - T0:.0f}s")


# ------------------------------------------------------------------ report (no torch)
def read_summary(path):
    return {r["key"]: float(r["value"]) for r in csv.DictReader(open(path))}


def read_images(path):
    rows = list(csv.DictReader(open(path)))
    d = {k: np.array([float(r[k]) for r in rows]) for k in IMG_COLS if k not in ("id", "quartile")}
    d["quartile"] = np.array([int(r["quartile"][1]) - 1 for r in rows])
    d["id"] = [r["id"] for r in rows]
    return d


def report_set(name, s, im, boot):
    """Prints the per-set numbers used by the rule; returns dict(d, b, p) for the verdict."""
    st = C.set_stats({a: im[f"auroc_{a}"] for a in C.ARMS}, boot)
    pa = lambda k: s.get(k, float("nan"))
    p_pts = (pa("aupro_CWMS") - pa("aupro_FIXED")) * 100
    d = st["d"]
    print(f"\n=== {name} (n={st['n']}) ===")
    print("mean per-image AUROC: " + "  ".join(f"{a} {st['means'][a]:.4f}" for a in C.ARMS))
    print(f"  CWMS - FIXED    {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
    for k in C.CONTROLS:
        m, lo, hi = st["d_vs"][k]
        print(f"  CWMS - {k:8s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  best control {st['best_control']}  B = CWMS - best control = {st['b']:+.4f}")
    print(f"pooled AUROC / AUPRO: FIXED {pa('pooled_FIXED'):.4f} / {pa('aupro_FIXED'):.4f}   CWMS {pa('pooled_CWMS'):.4f} / {pa('aupro_CWMS'):.4f}   UNI5 {pa('pooled_UNI5'):.4f} / {pa('aupro_UNI5'):.4f}"
          + (f"   SIGMA32 {pa('pooled_SIGMA32'):.4f} / {pa('aupro_SIGMA32'):.4f}" if "pooled_SIGMA32" in s else ""))
    print(f"  AUPRO CWMS - FIXED = {p_pts:+.2f} points   (floor {C.P_FLOOR:+.1f})   pooled AUROC CWMS - FIXED = {(pa('pooled_CWMS') - pa('pooled_FIXED')) * 100:+.2f} points")
    gg = C.group_gain(im["auroc_CWMS"] - im["auroc_FIXED"], im["c"], im["quartile"], im["id"])
    print(f"  [report-only] CWMS - FIXED by lesion-offset tercile: peripheral {gg['peripheral']:+.4f}  central {gg['central']:+.4f} (n={gg['n_tercile']} each);"
          f"  by area quartile Q1..Q4: " + "  ".join(f"{v:+.4f}" for v in gg["quartile"]))
    gu = C.group_gain(im["auroc_UNI5"] - im["auroc_FIXED"], im["c"], None, im["id"])
    print(f"  [report-only] UNI5 - FIXED peripheral {gu['peripheral']:+.4f}  central {gu['central']:+.4f}   (EXP-035 oracle recentre-alone gain, peripheral tercile: ClinicDB +0.057, ColonDB +0.100, Endo +0.047, Kvasir +0.056, TN3K +0.045, ISIC +0.002)")
    print(f"  set flags: PASS {C.set_pass(d, st['b'], p_pts)}   LOSS {C.set_loss(d)}   D<+0.003 {C.set_below_fail(d)}")
    return dict(d=d, b=st["b"], p=p_pts)


def stage_report(args):
    sets = {}
    for name in C.SETS:
        sp = os.path.join(args.report_dir, name, f"cwms_{name}.csv")
        ip = os.path.join(args.report_dir, name, f"cwms_{name}_images.csv")
        if not (os.path.isfile(sp) and os.path.isfile(ip)):
            print(f"MISSING {sp}"); sets[name] = None; continue
        sets[name] = report_set(name, read_summary(sp), read_images(ip), args.boot)
    print("\n================ VERDICT (rule fixed in research/EXPERIMENTS.md EXP-036; all six sets count) ================")
    for name in C.SETS:
        v = sets[name]
        if v is None:
            print(f"{name:9s} missing"); continue
        d = v["d"]
        print(f"{name:9s} D {d[0]:+.4f} [lo {d[1]:+.4f}]  B {v['b']:+.4f}  P {v['p']:+.2f} pt   PASS {C.set_pass(d, v['b'], v['p'])}  LOSS {C.set_loss(d)}")
    label, npass, nlt, nloss = C.verdict(sets)
    print(f"PASS on {npass}/6, D < +0.003 on {nlt}/6, LOSS (D <= -0.005) on {nloss}/6  ->  {label}")
    for name in ("ColonDB", "Kvasir"):
        if sets.get(name):
            print(f"AUPRO CWMS - FIXED on {name}: {sets[name]['p']:+.2f} points")
    print("ADVANCE authorises only a three-seed inference-only check on matched ECP seeds 222/333 and a registered comparison against stronger controls; it is not a method claim."
          " CWMS costs 5 forwards per image (5x).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("eval", "smoke", "report"))
    ap.add_argument("--name", default="data", help="ClinicDB|ColonDB|ISIC|Endo|Kvasir|TN3K (selects the recorded-value parity check)")
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None, help="EXP-012 pixel_per_image_predictions.csv (identity parity, tol 1e-4)")
    ap.add_argument("--position_dir", default="results/EXP-032", help="holds <set>/position_<set>.csv (optional cross-check of c)")
    ap.add_argument("--out_dir", default="results/EXP-036")
    ap.add_argument("--report_dir", default="results/EXP-036")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_recorded_check", action="store_true")
    ap.add_argument("--no_pro", dest="pro", action="store_false", help="skip AUPRO (the verdict is then INCOMPLETE; smoke/debug only)")
    ap.add_argument("--pooled_sigma32", action="store_true", help="also compute pooled AUROC/AUPRO of SIGMA32 (rebuilt from the FIXED maps; extra PRO time)")
    ap.add_argument("--sigma", type=int, default=4)
    ap.add_argument("--z_pix", type=float, default=1.6)
    ap.add_argument("--image_size", type=int, default=S_PIX)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "smoke":
        a.limit = a.limit or 12
    {"eval": stage_eval, "smoke": stage_eval, "report": stage_report}[a.mode](a)
