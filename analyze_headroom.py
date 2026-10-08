"""
EXP-025 / H016: oracle error decomposition ("headroom") of the deployed ECP pixel maps.  Frozen model, forward passes only.
GT is used ONLY to build diagnostic oracle edits of the final map; no method is implied.

The map is exactly what test.py produces (--features_list 24, constant z_pix, bilinear 518x518 similarity map, Gaussian sigma 4, float32).
Per image the GT lesion is split by the signed distance d to its boundary: FAR d>28, RIM 0<d<=28, EDGE -7<=d<=0, CORE d<-7.
Oracle edits (each alone on the baseline map; gmin/gmax = min/max of the whole dataset's baseline maps):
  FAR_FIX: FAR -> gmin    RIM_FIX: RIM -> gmin    CORE_FIX: CORE -> gmax    EDGE_FIX (report-only): EDGE -> gmax
  PERFECT: outside -> gmin, inside -> gmax.
Metrics: dataset-level pooled pixel AUROC (metrics.roc_auc_lowmem) and AUPRO (metrics.cal_pro_score), overall; pooled AUROC per
lesion-area quartile (area = GT fraction of the 518x518 canvas, quartiles within the dataset).  gap closure = (edit - base) / (1 - base).
AUPRO of PERFECT is degenerate in cal_pro_score (FPR is 0 at every threshold, so its FPR normalisation is 0/0): it is reported as 1.0
BY CONSTRUCTION (PRO = 1 and FPR = 0 at all thresholds), not measured.

STOP conditions: base per-image AUROC vs EXP-012 reference > 1e-4 (--reference_csv); baseline pooled AUROC/AUPRO vs the recorded
sigma-4 seed-111 values > 0.1 point (when --name is known, no --limit, no --no_recorded_check); PERFECT AUROC < 0.9999.

One command per dataset (see research/EXPERIMENTS.md EXP-025 for the full list):
  python analyze_headroom.py eval --name ClinicDB --data_path $CVC/CVC-ClinicDB --dataset colon \
     --checkpoint_path checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth --reference_csv <EXP-012 pixel_per_image_predictions.csv> \
     --out_csv headroom_ClinicDB.csv
Aggregate:
  python analyze_headroom.py --aggregate headroom_ClinicDB.csv headroom_Kvasir.csv headroom_ColonDB.csv headroom_ISIC.csv headroom_Endo.csv headroom_TN3K.csv
"""
import os
import csv
import time
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

from posart_stats import auroc as image_auroc
from headroom_stats import (zone_map, apply_edit, gap_closure, quartile_index, dominant_mode, decide_targets, edit_values,
                            EDITS, RULE_MODES, ZONE_NAMES, MIN_CLOSURE)

SUBSETS = ("ALL", "Q1", "Q2", "Q3", "Q4")
# recorded sigma-4 pooled values of the matched ECP, seed 111, z_pix 1.6 (AUROC %, AUPRO %), EXP-012 / RESULTS_SUMMARY
RECORDED = {"ClinicDB": (88.3, 76.4), "Kvasir": (88.9, 51.8), "ColonDB": (85.0, 72.9), "ISIC": (93.9, 88.7), "Endo": (91.6, 79.6), "TN3K": (82.5, 52.0)}
PRO_EDITS = ("FAR_FIX", "RIM_FIX", "CORE_FIX")   # AUPRO is computed for these (+ base); PERFECT analytic; EDGE_FIX AUROC only


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
    if ckpt.get("extent_cond"):   # same rule as test.py: none -> no conditioner, 'dual' -> constant head shifts, 'extent' -> z override
        cond = ExtentConditioner(mode=ckpt["extent_cond"]).to(device)
        cond.load_state_dict(ckpt["conditioner"])
        cond.eval()
    return device, model, pl, cond


def stage_eval(args):
    import torch
    import AnomalyCLIP_lib
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features
    from metrics import roc_auc_lowmem, cal_pro_score

    device, model, pl, cond = load_model(args)
    mode = cond.mode if cond is not None else "none"
    print(f"checkpoint {args.checkpoint_path}: extent_cond={mode}")
    with torch.no_grad():
        prompts, tok, comp = pl(cls_id=None)
        tf_base = model.encode_text_learn(prompts, tok, comp).float()
        tf_base = torch.stack(torch.chunk(tf_base, dim=0, chunks=2), dim=1)
        tf_base = tf_base / tf_base.norm(dim=-1, keepdim=True)

    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    idx = np.random.RandomState(0).permutation(len(data))[: args.limit] if args.limit else range(len(data))
    ref = None
    if args.reference_csv:
        ref = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(args.reference_csv))}
    else:
        print("WARNING: no --reference_csv, EXP-012 parity NOT checked")

    S = args.image_size
    N = len(idx)
    base = np.empty((N, S, S), np.float32)     # sigma-4 maps, as test.py (float32)
    gt_all = np.empty((N, S, S), np.uint8)
    zones = np.empty((N, S, S), np.int8)
    ids, areas, aur_base, zcount = [], [], [], np.zeros(4, np.int64)
    zsum = np.zeros(4, np.float64)
    n, max_err = 0, 0.0
    t0 = time.time()
    for i in idx:
        it = data[int(i)]
        gt = it["img_mask"][0].numpy() > 0.5 if it["img_mask"].ndim == 3 else it["img_mask"].numpy() > 0.5
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        img = it["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            imf, pf = model.encode_image(img, [24], DPAM_layer=20)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            if cond is None:
                tf = tf_base
            elif cond.mode == "dual":
                tf = conditioned_text_features(model, pl, *cond.dual_shift("pix", imf.shape[0]))
            else:
                zt = torch.full((imf.shape[0],), args.ec_z_pix, device=device)
                _, c_pos, c_neg = cond(visual_descriptor(imf, pf), z_override=zt)
                tf = conditioned_text_features(model, pl, c_pos, c_neg)
            pfeat = pf[0] / pf[0].norm(dim=-1, keepdim=True)          # literal test.py readout, single layer
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfeat, tf[0])
            sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], S)
            amap = ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu()
        m = gaussian_filter(amap.numpy(), sigma=args.sigma).astype(np.float32)   # test.py: gaussian_filter on the float32 tensor
        z = zone_map(gt, args.far, args.edge)
        sid = os.path.relpath(it["img_path"], args.data_path)
        a = float(image_auroc(gt, m))
        if ref is not None:
            if sid not in ref:
                raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
            err = abs(a - ref[sid]); max_err = max(max_err, err)
            if err > 1e-4:
                raise SystemExit(f"STOP: parity failure on {sid}: {a:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
        base[n], gt_all[n], zones[n] = m, gt, z
        for k in range(4):
            sel = z == k
            zcount[k] += int(sel.sum()); zsum[k] += float(m[sel].sum(dtype=np.float64))
        ids.append(sid); areas.append(float(gt.mean())); aur_base.append(a)
        n += 1
        if n % 50 == 0:
            print(f"{n} images ({time.time() - t0:.0f}s)", flush=True)
    base, gt_all, zones = base[:n], gt_all[:n], zones[:n]
    areas = np.array(areas)
    print(f"{n} images kept; parity max |base per-image AUROC - EXP-012| = {max_err:.2e}" + ("" if ref else " (NOT CHECKED)"))
    gmin, gmax = float(base.min()), float(base.max())
    q = quartile_index(areas)
    sub = {"ALL": None, **{f"Q{k + 1}": np.where(q == k)[0] for k in range(4)}}
    print("quartile image counts / area ranges:", {s: (len(v), f"{areas[v].min():.4f}-{areas[v].max():.4f}") for s, v in sub.items() if v is not None})
    print("zone pixel fractions (FAR,RIM,EDGE,CORE):", np.round(zcount / zcount.sum(), 4).tolist(),
          "mean baseline score:", np.round(zsum / np.maximum(zcount, 1), 4).tolist(), f"gmin={gmin:.4f} gmax={gmax:.4f}")

    def sel(arr, key):
        return arr if sub[key] is None else arr[sub[key]]

    def eval_au(m):
        return {s: float(roc_auc_lowmem(sel(gt_all, s), sel(m, s))) for s in SUBSETS}

    res = {}   # edit -> {"au": {subset: v}, "pro": v}
    t = time.time(); res["base"] = {"au": eval_au(base)}
    print(f"base AUROC ({time.time() - t:.0f}s): " + " ".join(f"{s}={v:.4f}" for s, v in res["base"]["au"].items()), flush=True)
    check = args.name in RECORDED and not args.no_recorded_check and not args.limit
    if check and abs(res["base"]["au"]["ALL"] * 100 - RECORDED[args.name][0]) > 0.1:
        raise SystemExit(f"STOP: baseline pooled AUROC {res['base']['au']['ALL'] * 100:.2f} vs recorded {RECORDED[args.name][0]} (> 0.1 point)")
    pro_on = args.pro != "none"
    if pro_on:
        t = time.time(); res["base"]["pro"] = float(cal_pro_score(gt_all, base)); dt = time.time() - t
        print(f"base AUPRO = {res['base']['pro']:.4f}  (PRO time {dt:.0f}s; projected for the {len(PRO_EDITS)} PRO edits: {dt * len(PRO_EDITS) / 60:.0f} min)", flush=True)
        if check and abs(res["base"]["pro"] * 100 - RECORDED[args.name][1]) > 0.1:
            raise SystemExit(f"STOP: baseline AUPRO {res['base']['pro'] * 100:.2f} vs recorded {RECORDED[args.name][1]} (> 0.1 point)")
    elif check:
        print("note: --pro none, recorded AUPRO parity not checked")

    buf = np.empty_like(base)
    for e in EDITS:
        apply_edit(buf, base, zones, e, gmin, gmax)
        t = time.time(); res[e] = {"au": eval_au(buf)}
        line = f"{e}: AUROC " + " ".join(f"{s}={v:.4f}" for s, v in res[e]["au"].items())
        if e == "PERFECT":
            if res[e]["au"]["ALL"] < 0.9999:
                raise SystemExit(f"STOP: PERFECT AUROC {res[e]['au']['ALL']:.6f} < 0.9999")
            res[e]["pro"] = 1.0 if pro_on else float("nan")   # by construction, see module docstring
        elif pro_on and args.pro == "all" and e in PRO_EDITS:
            res[e]["pro"] = float(cal_pro_score(gt_all, buf)); line += f"  AUPRO={res[e]['pro']:.4f}"
        print(line + f"  ({time.time() - t:.0f}s)", flush=True)
    del buf

    if args.per_image_edits:
        cols = {e: [] for e in RULE_MODES}
        for j in range(n):
            for e in RULE_MODES:
                mm = base[j].copy()
                for zz, v in edit_values(e, gmin, gmax):
                    mm[zones[j] == zz] = v
                cols[e].append(float(image_auroc(gt_all[j] > 0, mm)))
    with open(args.out_csv.replace(".csv", "_images.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "area_frac", "quartile", "per_image_auroc_base"] + [f"per_image_auroc_{e}" for e in (RULE_MODES if args.per_image_edits else [])])
        for j in range(n):
            w.writerow([ids[j], areas[j], f"Q{q[j] + 1}", aur_base[j]] + ([cols[e][j] for e in RULE_MODES] if args.per_image_edits else []))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "subset", "edit", "n_images", "auroc", "aupro", "auroc_closure", "aupro_closure", "gmin", "gmax"])
        for s in SUBSETS:
            ni = n if s == "ALL" else len(sub[s])
            for e in ("base",) + EDITS:
                au = res[e]["au"][s]
                pr = res[e].get("pro", float("nan")) if s == "ALL" else float("nan")
                prb = res["base"].get("pro", float("nan")) if s == "ALL" else float("nan")
                w.writerow([args.name, s, e, ni, au, pr, gap_closure(au, res["base"]["au"][s]), gap_closure(pr, prb), gmin, gmax])
    with open(args.out_csv.replace(".csv", "_zones.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "zone", "pixel_fraction", "mean_base_score"])
        for k in range(4):
            w.writerow([args.name, ZONE_NAMES[k], zcount[k] / zcount.sum(), zsum[k] / max(zcount[k], 1)])
    print(f"wrote {args.out_csv} (+ _images.csv, _zones.csv); total {time.time() - t0:.0f}s")
    print_dataset(args.name, read_rows(args.out_csv))


# ------------------------------------------------------------------ report
def read_rows(path):
    return list(csv.DictReader(open(path)))


def closures(rows, subset, key="auroc_closure"):
    return {r["edit"]: float(r[key]) for r in rows if r["subset"] == subset and r["edit"] != "base"}


def print_dataset(name, rows):
    print(f"\n=== {name}: pooled pixel metrics (gap closure = (edit-base)/(1-base)) ===")
    for s in SUBSETS:
        rs = [r for r in rows if r["subset"] == s]
        print(f"[{s}] n_images={rs[0]['n_images']}")
        for r in rs:
            extra = f"  AUPRO={float(r['aupro']):.4f} closure={float(r['aupro_closure']):+.3f}" if s == "ALL" and np.isfinite(float(r["aupro"])) else ""
            print(f"   {r['edit']:9s} AUROC={float(r['auroc']):.4f} closure={float(r['auroc_closure']):+.3f}{extra}")


def aggregate(paths):
    dom_all, dom_q1 = {}, {}
    for p in paths:
        rows = read_rows(p)
        name = rows[0]["dataset"]
        print_dataset(name, rows)
        dom_all[name] = dominant_mode(closures(rows, "ALL"))
        dom_q1[name] = dominant_mode(closures(rows, "Q1"))
    print("\n=== dominant mode per dataset (AUROC closure >= %.0f%%, among %s) ===" % (MIN_CLOSURE * 100, "/".join(RULE_MODES)))
    for nm in dom_all:
        print(f"{nm:10s} all: {dom_all[nm]}   Q1 (smallest area): {dom_q1[nm]}")
    ta, tq = decide_targets(dom_all, dom_q1)
    targets = sorted(set(ta) | set(tq))
    print(f"\nTARGET (dominant on >=3 datasets overall): {ta or 'none'};  TARGET (dominant in Q1 on >=3 datasets): {tq or 'none'}")
    print("RESULT:", ", ".join(targets) if targets else "no single dominant mode (do not pick one)")
    if "FAR_FIX" in targets:
        print("NOTE: text-based suppression of false-positive hotspots was already tried and falsified (EXP-007); a dominant FAR mode does not by "
              "itself open a new direction. Oracle edits are diagnostic upper bounds, not achievable gains.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=("eval",))
    ap.add_argument("--aggregate", nargs="+", default=None)
    ap.add_argument("--name", default="data", help="ClinicDB | Kvasir | ColonDB | ISIC | Endo | TN3K (selects the recorded-value parity check)")
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None, help="EXP-012 pixel_per_image_predictions.csv (parity, tol 1e-4); only valid for the seed-111 EXP-012 ECP")
    ap.add_argument("--no_recorded_check", action="store_true", help="skip the recorded seed-111 pooled AUROC/AUPRO check (other checkpoints)")
    ap.add_argument("--pro", choices=("all", "base", "none"), default="all", help="AUPRO for base + FAR/RIM/CORE edits, base only, or none")
    ap.add_argument("--per_image_edits", action="store_true", help="also write per-image AUROC after each rule edit (slower)")
    ap.add_argument("--ec_z_pix", type=float, default=1.6)
    ap.add_argument("--sigma", type=int, default=4)
    ap.add_argument("--far", type=float, default=28)
    ap.add_argument("--edge", type=float, default=7)
    ap.add_argument("--out_csv", default="headroom.csv")
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
