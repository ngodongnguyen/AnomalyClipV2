"""
EXP-032 / H022: per-image bad-case characterisation of the deployed matched ECP (frozen, forward-only, exploratory; NOT a method).
Rules (operating point, frozen tags, gate, control, galleries): research/EXPERIMENTS.md EXP-032.  Statistics: badcase_stats.py (unit-tested).

Modes (one dataset per command, resumable: an existing output CSV is skipped unless --redo):
  python analyze_badcases.py pixel  --name ClinicDB --dataset colon --data_path $CVC/CVC-ClinicDB --checkpoint_path <ckpt> --reference_csv <EXP-012 csv>
  python analyze_badcases.py imgset --name HeadCT   --dataset brain --data_path $A/HeadCT_anomaly_detection --checkpoint_path <ckpt>
  python analyze_badcases.py report --report_dir results/EXP-032        # tags, prevalence ratios, gate, permutation control (no torch)
Pixel set: maps exactly as analyze_zens.py (z_pix 1.6, sigma 4, features_list 24, 518).  Image set: score exactly as test.py's image path (z_img -0.69).
STOP conditions: per-image AUROC vs the EXP-012 reference CSV > 1e-4; pooled AUROC/AUPRO vs the recorded sigma-4 values > 0.1 point;
image AUROC vs the recorded matched-ECP value > 0.1 point (the recorded checks are off with --limit / --no_recorded_check).
"""
import os
import csv
import time
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

import badcase_stats as B

RECORDED = {"ClinicDB": (88.3, 76.4), "Kvasir": (88.9, 51.8), "ColonDB": (85.0, 72.9), "ISIC": (93.9, 88.7), "Endo": (91.6, 79.6), "TN3K": (82.5, 52.0)}
RECORDED_IMG = {"HeadCT": 94.2, "BrainMRI": 94.8, "Br35H": 97.1}      # matched ECP seed 111, z_img -0.69, image AUROC % (EXP-022 table)
PIXEL_SETS = ("ClinicDB", "Kvasir", "ColonDB", "ISIC", "Endo", "TN3K")
PANEL = 256


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


def load_rgb(path, size):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)      # same size as the model input (Resize + CenterCrop of a square)
    return np.asarray(im, np.float32) / 255.0


def folder_size_mb(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 1e6


# ------------------------------------------------------------------ drawing (cv2, BGR)
def to_bgr(rgb01):
    import cv2
    return cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def small(img):
    import cv2
    return cv2.resize(img, (PANEL, PANEL), interpolation=cv2.INTER_AREA)


def contour(img, mask, color, thick=2):
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, color, thick)
    return img


def text_panel(lines, size=PANEL):
    import cv2
    p = np.zeros((size, size, 3), np.uint8)
    y = 16
    for ln in lines:
        cv2.putText(p, ln, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        y += 17
    return p


def case_row(rgb, score, gt, thr, gmin, gmax, comps, lab, lines):
    import cv2
    pred = score >= thr
    p1 = contour(to_bgr(rgb), gt, (0, 255, 0))
    nm = np.clip((score - gmin) / max(gmax - gmin, 1e-12), 0, 1)
    p2 = contour(cv2.applyColorMap((nm * 255).astype(np.uint8), cv2.COLORMAP_JET), pred, (255, 255, 255))
    p3 = (to_bgr(rgb).astype(np.float32) * 0.6)
    fp = np.zeros(gt.shape, bool)
    for c in comps:
        fp |= lab == c["label"]
    for m, col in ((fp, (0, 0, 255)), (gt & ~pred, (255, 0, 0))):                 # far-FP components red, missed lesion blue (BGR)
        p3[m] = 0.4 * p3[m] + 0.6 * np.array(col, np.float32)
    p3 = contour(p3.astype(np.uint8), gt, (0, 255, 0), 1)
    return np.hstack([small(p1), small(p2), small(p3), text_panel(lines)])


def write_montage(path, rows):
    import cv2
    if rows:
        cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ------------------------------------------------------------------ pixel sets
def stage_pixel(args):
    import torch
    import AnomalyCLIP_lib
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features
    from metrics import roc_auc_lowmem, cal_pro_score
    from headroom_stats import quartile_index

    out_dir = os.path.join(args.out_dir, args.name)
    out_csv = os.path.join(out_dir, f"badcase_{args.name}.csv")
    if os.path.exists(out_csv) and not args.redo:
        print(f"{args.name}: {out_csv} exists, skipped (use --redo)")
        return
    os.makedirs(out_dir, exist_ok=True)
    T0 = time.time()
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

    t0 = time.time()
    kept, areas = [], []
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
    quart = quartile_index(areas)
    N, S = len(kept), args.image_size
    print(f"[stage] {args.name}: {N} kept images, pass 1 (masks) {time.time() - t0:.0f}s")

    base = np.empty((N, S, S), np.float32)
    gt_all = np.empty((N, S, S), np.uint8)
    auroc, ids, paths = np.empty(N), [], []
    max_err = 0.0
    t0 = time.time()
    for n, i in enumerate(kept):
        it = data[int(i)]
        gt = read_mask(it)
        sid = os.path.relpath(it["img_path"], args.data_path)
        img = it["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            imf, pf = model.encode_image(img, [24], DPAM_layer=20)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            pfeat = pf[0] / pf[0].norm(dim=-1, keepdim=True)
            desc = visual_descriptor(imf, pf)
            zt = torch.full((imf.shape[0],), args.z_pix, device=device)
            _, c_pos, c_neg = cond(desc, z_override=zt)
            tf = conditioned_text_features(model, pl, c_pos, c_neg)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfeat, tf[0])
            sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], S)
            amap = ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu()
        base[n] = gaussian_filter(amap.numpy(), sigma=args.sigma).astype(np.float32)
        gt_all[n] = gt
        auroc[n] = float(roc_auc_lowmem(gt, base[n]))
        if ref is not None:
            if sid not in ref:
                raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
            err = abs(auroc[n] - ref[sid]); max_err = max(max_err, err)
            if err > 1e-4:
                raise SystemExit(f"STOP: parity failure on {sid}: {auroc[n]:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
        ids.append(sid); paths.append(it["img_path"])
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{N} images ({time.time() - t0:.0f}s)", flush=True)
    del model, pl, cond
    print(f"[stage] forward {N} images {time.time() - t0:.0f}s; parity max |per-image AUROC - EXP-012| = {max_err:.2e}" + ("" if ref else " (NOT CHECKED)"))

    summ = {"n_images": N, "parity_max_err": max_err}
    t0 = time.time()
    summ["pooled_auroc"] = float(roc_auc_lowmem(gt_all, base))
    line = f"[stage] pooled AUROC {summ['pooled_auroc']:.4f}"
    check = args.name in RECORDED and not args.no_recorded_check and not args.limit
    if args.pro and not args.limit:
        tp = time.time(); summ["pooled_aupro"] = float(cal_pro_score(gt_all, base)); line += f"  AUPRO {summ['pooled_aupro']:.4f} (PRO {time.time() - tp:.0f}s)"
    print(line)
    if check:
        if abs(summ["pooled_auroc"] * 100 - RECORDED[args.name][0]) > 0.1:
            raise SystemExit(f"STOP: pooled AUROC {summ['pooled_auroc'] * 100:.2f} vs recorded {RECORDED[args.name][0]} (> 0.1 point)")
        if "pooled_aupro" in summ and abs(summ["pooled_aupro"] * 100 - RECORDED[args.name][1]) > 0.1:
            raise SystemExit(f"STOP: AUPRO {summ['pooled_aupro'] * 100:.2f} vs recorded {RECORDED[args.name][1]} (> 0.1 point)")
        print("[stage] recorded-value parity OK")

    thr = B.threshold_at_tpr(base[gt_all > 0], B.TPR_TARGET)
    gmin, gmax = float(base.min()), float(base.max())
    summ["thr_tpr0.8"] = thr
    print(f"[stage] operating point: pooled threshold at TPR {B.TPR_TARGET} = {thr:.5f} (map range {gmin:.4f}..{gmax:.4f}); {time.time() - t0:.0f}s")

    t0 = time.time()
    recs = []
    for n in range(N):
        rgb = load_rgb(paths[n], S)
        rec, _, _, _ = B.image_record(rgb, base[n], gt_all[n] > 0, thr)
        rec.update(id=ids[n], quartile=int(quart[n]), auroc=float(auroc[n]))
        recs.append(rec)
    cols = {k: np.array([r[k] for r in recs]) for k in B.REC_COLS}
    tags = B.assign_tags(cols)
    print(f"[stage] per-image statistics {time.time() - t0:.0f}s; images with no far-FP component: {int((cols['n_comp'] == 0).sum())}/{N}")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + B.REC_COLS + list(B.ALL_TAGS))
        for n in range(N):
            w.writerow([args.name] + [recs[n][k] for k in B.REC_COLS] + [int(tags[t][n]) for t in B.ALL_TAGS])
    with open(os.path.join(out_dir, f"summary_{args.name}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "key", "value"])
        for k, v in summ.items():
            w.writerow([args.name, k, v])
    print(f"wrote {out_csv}")

    # galleries
    t0 = time.time()
    order = np.argsort(auroc, kind="stable")
    mid = N // 2
    groups = {
        "worst_auroc": list(order[:12]),
        "median_auroc": list(order[max(mid - 3, 0): mid + 3]),
        "best_auroc": list(order[::-1][:6]),
        "largest_farfp": list(np.argsort(-cols["fp_frac"], kind="stable")[:12]),
        "lowest_hit": list(np.argsort(cols["hit_rate"], kind="stable")[:12]),
    }
    for gname, idxs in groups.items():
        rows = []
        for n in idxs:
            rgb = load_rgb(paths[n], S)
            gt = gt_all[n] > 0
            _, _, lab, comps = B.image_record(rgb, base[n], gt, thr)
            tg = [t for t in B.ALL_TAGS if tags[t][n]]
            lines = [ids[n][-38:], f"area {recs[n]['area_frac']:.4f} Q{quart[n] + 1}  AUROC {auroc[n]:.3f}",
                     f"hit {recs[n]['hit_rate']:.2f}  FPfrac {recs[n]['fp_frac']:.4f}  comps {recs[n]['n_comp']}",
                     f"lesion pct {recs[n]['lesion_pct']:.2f}"]
            lines += [" ".join(tg[i:i + 3]) for i in range(0, len(tg), 3)] or ["(no tags)"]
            lines += ["green: GT  white: thr  red: far-FP  blue: missed"]
            rows.append(case_row(rgb, base[n], gt, thr, gmin, gmax, comps, lab, lines))
        write_montage(os.path.join(out_dir, f"gallery_{gname}.jpg"), rows)
    print(f"[stage] galleries {time.time() - t0:.0f}s; folder {out_dir} = {folder_size_mb(out_dir):.1f} MB")
    print(f"[stage] total {args.name} {time.time() - T0:.0f}s")


# ------------------------------------------------------------------ image-level sets
def stage_imgset(args):
    import torch
    from sklearn.metrics import roc_auc_score
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features

    out_dir = os.path.join(args.out_dir, args.name)
    out_csv = os.path.join(out_dir, f"imglevel_{args.name}.csv")
    if os.path.exists(out_csv) and not args.redo:
        print(f"{args.name}: {out_csv} exists, skipped (use --redo)")
        return
    os.makedirs(out_dir, exist_ok=True)
    T0 = time.time()
    device, model, pl, cond = load_model(args)
    if cond is None or cond.mode != "extent":
        raise SystemExit("STOP: this diagnostic needs the 'extent' ECP checkpoint (z override)")
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    idx = list(range(len(data)))
    if args.limit:
        idx = [int(j) for j in np.random.RandomState(0).permutation(len(data))[: args.limit]]
        print(f"--limit: {len(idx)} random images (smoke; recorded-value check off)")
    ids, paths, labels, scores = [], [], [], []
    t0 = time.time()
    for n, i in enumerate(idx):
        it = data[int(i)]
        img = it["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            imf, pf = model.encode_image(img, [24], DPAM_layer=20)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            desc = visual_descriptor(imf, pf)
            zt = torch.full((imf.shape[0],), args.z_img, device=device)
            _, c_pos, c_neg = cond(desc, z_override=zt)
            tf = conditioned_text_features(model, pl, c_pos, c_neg)
            probs = (imf @ tf.permute(0, 2, 1) / 0.07).softmax(-1)[:, 0, 1]
        ids.append(os.path.relpath(it["img_path"], args.data_path)); paths.append(it["img_path"])
        labels.append(int(it["anomaly"])); scores.append(float(probs.cpu()[0]))
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(idx)} images ({time.time() - t0:.0f}s)", flush=True)
    labels, scores = np.array(labels), np.array(scores)
    auc = float(roc_auc_score(labels, scores))
    print(f"[stage] forward {len(idx)} images {time.time() - t0:.0f}s; image AUROC {auc * 100:.2f}  (anomalous {int(labels.sum())}, normal {int((1 - labels).sum())})")
    if args.name in RECORDED_IMG and not args.no_recorded_check and not args.limit:
        if abs(auc * 100 - RECORDED_IMG[args.name]) > 0.1:
            raise SystemExit(f"STOP: image AUROC {auc * 100:.2f} vs recorded {RECORDED_IMG[args.name]} (> 0.1 point)")
        print("[stage] recorded-value parity OK")

    from PIL import Image
    mean_i, std_i = np.empty(len(ids)), np.empty(len(ids))
    for n, p in enumerate(paths):
        g = np.asarray(Image.open(p).convert("L").resize((args.image_size, args.image_size), Image.BICUBIC), np.float32)
        mean_i[n], std_i[n] = g.mean(), g.std()
    rank = B.deterministic_rank(ids, scores)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "id", "label", "score", "rank", "mean_intensity", "std_intensity"])
        for n in np.argsort(rank):
            w.writerow([args.name, ids[n], labels[n], f"{scores[n]:.8f}", rank[n], f"{mean_i[n]:.2f}", f"{std_i[n]:.2f}"])
    summ = B.score_summary(labels, scores)
    summ.update(image_auroc=auc, n=len(ids), n_anomalous=int(labels.sum()))
    print(f"[summary] {args.name}: " + ", ".join(f"{k}={v}" for k, v in summ.items()))
    with open(os.path.join(out_dir, f"summary_{args.name}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "key", "value"])
        for k, v in summ.items():
            w.writerow([args.name, k, v])

    import cv2
    def thumbs(sel, fname):
        tiles = []
        for n in sel:
            g = to_bgr(load_rgb(paths[n], args.image_size)); t = small(g)
            lab = np.zeros((34, PANEL, 3), np.uint8)
            cv2.putText(lab, f"s={scores[n]:.3f} rank {rank[n]}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(lab, f"mean {mean_i[n]:.0f} std {std_i[n]:.0f}", (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([t, lab]))
        while tiles and len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        write_montage(os.path.join(out_dir, fname), [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    order = np.argsort(rank)                                           # rank 1 = highest score
    fn = [n for n in order[::-1] if labels[n] == 1][:12]               # lowest-scoring anomalous
    fp = [n for n in order if labels[n] == 0][:12]                     # highest-scoring normal
    thumbs(fn, "gallery_false_negatives.jpg"); thumbs(fp, "gallery_false_positives.jpg")
    print(f"[stage] {args.name} thumbnails; folder {out_dir} = {folder_size_mb(out_dir):.2f} MB; total {time.time() - T0:.0f}s")


# ------------------------------------------------------------------ report (no torch)
def read_cols(path):
    rows = list(csv.DictReader(open(path)))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in B.REC_COLS if k != "id"}
    cols["id"] = [r["id"] for r in rows]
    cols["quartile"] = cols["quartile"].astype(int)
    return cols


def stage_report(args):
    paths = {s: os.path.join(args.report_dir, s, f"badcase_{s}.csv") for s in PIXEL_SETS}
    paths = {s: p for s, p in paths.items() if os.path.exists(p)}
    if not paths:
        raise SystemExit("no badcase_<set>.csv under --report_dir")
    tags = list(B.ALL_TAGS)
    res, res_fp, res_null, prev = {}, {}, {}, {}
    for s, p in paths.items():
        cols = read_cols(p)
        tg = B.assign_tags(cols)
        T = np.stack([tg[t] for t in tags], 1)
        au = cols["auroc"]
        n = len(au)
        res[s] = B.analyse_set(au, T, B=args.boot)
        res_fp[s] = B.analyse_set(cols["fp_frac"], T, largest=True, B=args.boot)
        res_null[s] = B.analyse_set(np.random.RandomState(0).permutation(au), T, B=args.boot)
        prev[s] = T.mean(0)
        print(f"\n=== {s} (n={n}; mean per-image AUROC {au.mean():.4f}; images with no far-FP component {int((cols['n_comp'] == 0).sum())}; worst decile n={res[s]['nw']}) ===")
        print(f"{'tag':11s}{'overall':>8s}{'worst':>9s}{'rest':>9s}{'ratio':>8s}{'95% CI':>18s}   {'ratio(FPfrac)':>14s}{'lo':>7s}   null ratio [lo]")
        for i, t in enumerate(tags):
            r, f_, nu = res[s], res_fp[s], res_null[s]
            mark = "*" if B.passes(r)[i] and t in B.GATED_TAGS else " "     # * = passes the gate on this set (real AUROC), not the null
            print(f"{t:11s}{prev[s][i]:8.3f}{r['a'][i] / r['nw']:9.3f}{r['b'][i] / r['nr']:9.3f}{r['ratio'][i]:8.2f}"
                  f"{'[%.2f, %.2f]' % (r['lo'][i], r['hi'][i]):>18s}{mark}  {f_['ratio'][i]:14.2f}{f_['lo'][i]:7.2f}   {nu['ratio'][i]:.2f} [{nu['lo'][i]:.2f}]")
    print("\n================ GATE (rule fixed in research/EXPERIMENTS.md EXP-032) ================")
    print(f"candidate cause = ratio >= {B.RATIO_MIN} with bootstrap lower bound > {B.LO_MIN} on >= {B.SETS_MIN} of 6 pixel sets ({len(paths)} sets available)")
    cnt, cand = B.candidate_verdict({s: B.passes(r) for s, r in res.items()}, tags)
    cnt_n, cand_n = B.candidate_verdict({s: B.passes(r) for s, r in res_null.items()}, tags)
    cnt_f, _ = B.candidate_verdict({s: B.passes(r) for s, r in res_fp.items()}, tags)
    for t in tags:
        ps = [s for s in paths if B.passes(res[s])[tags.index(t)]]
        note = "  (near-definitional for low AUROC)" if t in B.NEAR_DEFINITIONAL else ("  (descriptive stratum, not gated)" if t in B.DESCRIPTIVE_TAGS else "")
        print(f"{t:11s} sets passing {cnt[t]}/{len(paths)} ({','.join(ps) or '-'})   report-only FPfrac-decile: {cnt_f[t]}   permutation null: {cnt_n[t]}{note}")
    if len(paths) < 6:
        print(f"WARNING: only {len(paths)}/6 sets present; the gate needs the full six for a verdict")
    if cand:
        print(f"CANDIDATE CAUSE(S): {', '.join(cand)}  (descriptive association only; authorises a separately registered targeted test, not a method)")
    else:
        print("NO CANDIDATE CAUSE: the failures are not explained by these tags.")
    print(f"permutation-null passes (should be ~0; multiplicity {len(B.GATED_TAGS)} tags x {len(paths)} sets): {sum(cnt_n[t] for t in B.GATED_TAGS)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pixel", "imgset", "report"))
    ap.add_argument("--name", default="data", help="pixel: ClinicDB|Kvasir|ColonDB|ISIC|Endo|TN3K ; imgset: HeadCT|BrainMRI|Br35H")
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None, help="EXP-012 pixel_per_image_predictions.csv (per-image parity, tol 1e-4)")
    ap.add_argument("--no_recorded_check", action="store_true")
    ap.add_argument("--no_pro", dest="pro", action="store_false", help="skip pooled AUPRO (default: computed unless --limit)")
    ap.add_argument("--out_dir", default="results/EXP-032")
    ap.add_argument("--report_dir", default="results/EXP-032")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sigma", type=int, default=4)
    ap.add_argument("--z_pix", type=float, default=1.6)
    ap.add_argument("--z_img", type=float, default=-0.69)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    {"pixel": stage_pixel, "imgset": stage_imgset, "report": stage_report}[a.mode](a)
