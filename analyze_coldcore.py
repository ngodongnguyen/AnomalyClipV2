"""
EXP-026 / H017: mechanism of the "cold core" of large lesions in the deployed ECP.  Frozen model, forward passes only, no training.
Is a cold core VISUAL (core tokens look like background) or READOUT/PROMPT (core tokens look like the rim but score low)?
GT is used ONLY to define patch zones for the diagnostic; no method is implied.

Only the LARGEST within-dataset area quartile (Q4) is forwarded.  Quartiles come from all kept images (mask >= 20 px and background >= 20 px),
exactly as analyze_headroom.py (areas are read from masks, no forward pass).  Tokens: final-layer patch tokens (--features_list 24,
V-V stream, ln_post@proj), L2-normalised, as test.py.  Patch score s = (p_anom + 1 - p_normal)/2 on the 37x37 grid, before interpolation.
Zones and the pre-registered rule: see core_stats.py and research/EXPERIMENTS.md EXP-026.

STOP conditions: per-image AUROC of the full-resolution map (get_similarity_map + Gaussian sigma 4, as test.py) vs the EXP-012
reference CSV > 1e-4 on any processed image (--reference_csv).

One command per dataset:
  python analyze_coldcore.py eval --name ClinicDB --data_path $CVC/CVC-ClinicDB --dataset colon \
     --checkpoint_path checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth --reference_csv <EXP-012 pixel_per_image_predictions.csv> \
     --out_csv coldcore_ClinicDB.csv
Aggregate:
  python analyze_coldcore.py --aggregate coldcore_ClinicDB.csv coldcore_Kvasir.csv coldcore_ColonDB.csv coldcore_ISIC.csv coldcore_Endo.csv coldcore_TN3K.csv
"""
import os
import csv
import time
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

from posart_stats import auroc as image_auroc
from headroom_stats import quartile_index
from core_stats import (patch_fraction, patch_zones, zone_counts, is_eligible, prototypes, visual_ambiguity, f_bg, coldness, zone_means,
                        dataset_summary, verdict, MIN_IMAGES)

Z_SWEEP = (-0.69, 0.0, 0.8, 1.6, 2.4)     # report-only text-side extent axis
ZONES = ("core", "rim_in", "bg_far")


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

    device, model, pl, cond = load_model(args)
    if cond is None or cond.mode != "extent":
        raise SystemExit("STOP: this diagnostic needs the 'extent' ECP checkpoint (fixed z override)")
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    ref = None
    if args.reference_csv:
        ref = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(args.reference_csv))}
    else:
        print("WARNING: no --reference_csv, EXP-012 parity NOT checked")

    # pass 1: GT areas of all kept images (no forward pass), within-dataset quartiles as analyze_headroom.py
    t0 = time.time()
    kept, areas = [], []
    for i in range(len(data)):
        gt = read_mask(data[i])
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        kept.append(i); areas.append(float(gt.mean()))
    areas = np.array(areas)
    q = quartile_index(areas)
    q4 = [kept[j] for j in range(len(kept)) if q[j] == 3]
    print(f"{args.name}: {len(kept)} kept images, Q4 = {len(q4)} images (area >= {areas[q == 3].min():.4f}) ({time.time() - t0:.0f}s)")
    if args.limit:
        q4 = q4[: args.limit]
        print(f"--limit: processing only the first {len(q4)} Q4 images (smoke)")

    S = args.image_size
    rows, n_done, max_err = [], 0, 0.0
    t0 = time.time()
    for i in q4:
        it = data[int(i)]
        gt = read_mask(it)
        img = it["img"].unsqueeze(0).to(device)
        sid = os.path.relpath(it["img_path"], args.data_path)
        with torch.no_grad():
            imf, pf = model.encode_image(img, [24], DPAM_layer=20)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            pfeat = pf[0] / pf[0].norm(dim=-1, keepdim=True)                      # [1, 1+P, C], as test.py single layer

            def text_at(zv):
                zt = torch.full((imf.shape[0],), zv, device=device)
                _, c_pos, c_neg = cond(visual_descriptor(imf, pf), z_override=zt)
                return conditioned_text_features(model, pl, c_pos, c_neg)

            def patch_score(tf):
                sim, _ = AnomalyCLIP_lib.compute_similarity(pfeat, tf[0])        # [1, 1+P, 2] softmax probs
                return sim, ((sim[0, 1:, 1] + 1 - sim[0, 1:, 0]) / 2.0).float().cpu().numpy()

            sim, s_prim = patch_score(text_at(args.ec_z_pix))
            sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], S)
            amap = ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu()
        m = gaussian_filter(amap.numpy(), sigma=args.sigma).astype(np.float32)
        a = float(image_auroc(gt, m))
        if ref is not None:
            if sid not in ref:
                raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
            err = abs(a - ref[sid]); max_err = max(max_err, err)
            if err > 1e-4:
                raise SystemExit(f"STOP: parity failure on {sid}: {a:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
        n_done += 1
        grid = int(round(s_prim.shape[0] ** 0.5))
        z = patch_zones(patch_fraction(gt, grid))
        cnt = zone_counts(z)
        row = {"image": sid, "area_frac": float(gt.mean()), "n_core": cnt["core"], "n_rim_in": cnt["rim_in"], "n_bg_far": cnt["bg_far"],
               "eligible": int(is_eligible(z))}
        if row["eligible"]:
            tok = pfeat[0, 1:].float().cpu().numpy()
            proto = prototypes(tok, z)
            V, c_r, c_b = visual_ambiguity(proto)
            row.update(delta=coldness(s_prim, z), V=V, cos_core_rim=c_r, cos_core_bg=c_b, F_bg=f_bg(tok, z, proto))
            for zv in Z_SWEEP:
                with torch.no_grad():
                    _, sz = patch_score(text_at(zv))
                for k, v in zone_means(sz, z).items():
                    row[f"s_{k}_z{zv}"] = v
        rows.append(row)
        if n_done % 25 == 0:
            print(f"{n_done}/{len(q4)} Q4 images ({time.time() - t0:.0f}s)", flush=True)
    print(f"{n_done} Q4 images processed; parity max |per-image AUROC - EXP-012| = {max_err:.2e}" + ("" if ref else " (NOT CHECKED)"))
    el = [r for r in rows if r["eligible"]]
    print(f"{args.name}: eligible Q4 images = {len(el)} / {len(rows)} (need >= {MIN_IMAGES} to enter the decision)")
    for k, nm in (("n_core", "CORE"), ("n_rim_in", "RIM_IN"), ("n_bg_far", "BG_FAR")):
        v = np.array([r[k] for r in rows])
        print(f"   {nm} patches per Q4 image: median {np.median(v):.0f}  min {v.min()}  max {v.max()}")
    cols = ["dataset", "image", "area_frac", "n_core", "n_rim_in", "n_bg_far", "eligible", "delta", "V", "cos_core_rim", "cos_core_bg", "F_bg"] + \
           [f"s_{k}_z{zv}" for zv in Z_SWEEP for k in ZONES] + ["n_q4_processed", "parity_max_err", "limit"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({"dataset": args.name, **r, "n_q4_processed": n_done, "parity_max_err": max_err, "limit": args.limit})
    print(f"wrote {args.out_csv}")
    report_dataset(args.name, load_rows(args.out_csv))


# ------------------------------------------------------------------ report
def load_rows(path):
    out = []
    for r in csv.DictReader(open(path)):
        out.append({k: (float(v) if v not in ("", None) and k not in ("dataset", "image") else v) for k, v in r.items()})
    return out


def report_dataset(name, rows):
    el = [r for r in rows if r["eligible"] == 1]
    sm = None
    print(f"\n=== {name}: eligible Q4 images {len(el)} / {len(rows)} ===")
    if len(el) >= 3:
        d = np.array([r["delta"] for r in el]); v = np.array([r["V"] for r in el]); nc = np.array([r["n_core"] for r in el])
        sm = dataset_summary(d, v, nc)
        print(f"   delta (rim - core score): mean {d.mean():+.4f}, frac>0 {np.mean(d > 0):.2f};  V: mean {v.mean():+.4f}, frac>0 {np.mean(v > 0):.2f};  "
              f"cos(core,rim) mean {np.mean([r['cos_core_rim'] for r in el]):.3f}, cos(core,bg_far) mean {np.mean([r['cos_core_bg'] for r in el]):.3f}, "
              f"F_bg mean {np.mean([r['F_bg'] for r in el]):.3f}")
        print(f"   rho_d = {sm['rho']:+.3f}  (perm p = {sm['p']:.3f}, null 2.5/97.5% = {sm['q025']:+.3f}/{sm['q975']:+.3f});  "
              f"Vmed_cold = {sm['vmed_cold']:+.4f};  report-only partial rho (size-adjusted) = {sm['rho_partial']:+.3f}")
        print("   text-side z sweep (mean over eligible images of zone-mean patch score)  z: core / rim_in / bg_far / (rim-core) / (core-bg)")
        for zv in Z_SWEEP:
            c, r_, b = (np.mean([r[f"s_{k}_z{zv}"] for r in el]) for k in ZONES)
            print(f"      z={zv:+.2f}: {c:.4f} / {r_:.4f} / {b:.4f} / {r_ - c:+.4f} / {c - b:+.4f}")
    return sm


def aggregate(paths):
    sums, table = {}, []
    for p in paths:
        rows = load_rows(p)
        name = rows[0]["dataset"]
        if rows[0]["limit"] > 0:
            print(f"WARNING: {p} was produced with --limit {int(rows[0]['limit'])} (smoke); do not decide on it")
        sm = report_dataset(name, rows)
        if sm is None:
            sm = {"n": 0, "eligible": False, "rho": float("nan"), "vmed_cold": float("nan")}
        sums[name] = sm
    print("\n=== per-dataset decision table (eligible = >= %d eligible Q4 images) ===" % MIN_IMAGES)
    print(f"{'dataset':10s} {'n_elig':>6s} {'eligible':>8s} {'rho_d':>8s} {'perm_p':>7s} {'Vmed_cold':>10s} {'partial':>8s}  visual-pass readout-pass")
    for nm, s in sums.items():
        vp = bool(s["eligible"] and s["rho"] <= -0.30 and s["vmed_cold"] < 0)
        rp = bool(s["eligible"] and s["vmed_cold"] > 0 and s["rho"] > -0.15)
        print(f"{nm:10s} {s['n']:6d} {str(s['eligible']):>8s} {s['rho']:+8.3f} {s.get('p', float('nan')):7.3f} {s['vmed_cold']:+10.4f} "
              f"{s.get('rho_partial', float('nan')):+8.3f}  {str(vp):>11s} {str(rp):>12s}")
    v, why = verdict(sums)
    print(f"\nVERDICT: {v}  ({why})")
    if v == "INCONCLUSIVE":
        print("NOTE: fewer than 4 datasets are eligible, so the verdict is INCONCLUSIVE by construction.")
    print("NOTE: zone definitions are patch-level (14 px); the z sweep and partial rho are report-only; the rule is fixed in EXP-026.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=("eval",))
    ap.add_argument("--aggregate", nargs="+", default=None)
    ap.add_argument("--name", default="data", help="ClinicDB | Kvasir | ColonDB | ISIC | Endo | TN3K")
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None, help="EXP-012 pixel_per_image_predictions.csv (parity, tol 1e-4)")
    ap.add_argument("--ec_z_pix", type=float, default=1.6)
    ap.add_argument("--sigma", type=int, default=4)
    ap.add_argument("--out_csv", default="coldcore.csv")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N Q4 images")
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
