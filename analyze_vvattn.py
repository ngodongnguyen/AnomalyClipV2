"""
EXP-045 (H034 head subset, H035 sparse attention): free read-only gate on the V-V attention stream of the frozen matched ECP.
Forward passes only; labels are used only to SCORE. Rule fixed in research/EXPERIMENTS.md (EXP-045) and coded in vvattn_stats.py.

Arms (text features are computed ONCE per image from the unmodified forward and shared by all arms, as EXP-010; only the visual V-V path changes):
  base           unmodified
  ent_all        entmax-1.5 replaces softmax on the V-V logits, all 19 dual-path layers (6..24)
  ent_last6      same, layers 19..24
  sharp_all      CONTROL: softmax on logits x2, all layers (sharper but dense)
  head_top50/25  V-V head outputs zeroed (before self.proj) except the top 50% / 25% of the 304 (layer, head) pairs by the MVTec source score
  head_rand50/25 CONTROL: random masks of the same size, seeds 0,1,2 (reported separately; the control is their per-image mean)
  head_bot50     report-only: keep the lowest-ranked 50%
Fixed z_pix=1.6, layer 24 only, 518 px, per-image pixel AUROC at sigma 4 (decisive) and 32 (report-only), paired image bootstrap (2000, seed 0).

  python analyze_vvattn.py eval --name ClinicDB --dataset colon --data_path $CVC/CVC-ClinicDB --checkpoint_path <ECP ckpt> \
      --reference_csv <EXP-012 pixel_per_image_predictions.csv> --ranking results/EXP-045/head_ranking_mvtec.json --out_csv results/EXP-045/vvattn_ClinicDB.csv
  python analyze_vvattn.py report --dir results/EXP-045
"""
import os
import csv
import sys
import json
import time
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

import vvattn_stats as S

SIGMAS = (4, 32)
ARMS = ["base", "ent_all", "ent_last6", "sharp_all", "head_top50", "head_top25", "head_bot50"] + \
       [f"head_rand{p}_s{s}" for p in (50, 25) for s in S.RAND_SEEDS]
REPORT_ARMS = ["ent_all", "ent_last6", "sharp_all", "head_top50", "head_top25", "head_rand50", "head_rand25", "head_bot50"]
COLS = [f"{a}@{s}" for a in ARMS for s in SIGMAS]


def arm_spec(ranking, device):
    """arm -> (kind, keep-mask or None). Masks come only from the MVTec ranking and fixed random seeds."""
    spec = {"base": ("base", None), "ent_all": ("ent_all", None), "ent_last6": ("ent_last6", None), "sharp_all": ("sharp_all", None),
            "head_top50": ("head", S.mask_from_ranking(ranking, "top", 152)), "head_top25": ("head", S.mask_from_ranking(ranking, "top", 76)),
            "head_bot50": ("head", S.mask_from_ranking(ranking, "bottom", 152))}
    for s in S.RAND_SEEDS:
        spec[f"head_rand50_s{s}"] = ("head", S.random_mask(152, s))
        spec[f"head_rand25_s{s}"] = ("head", S.random_mask(76, s))
    return spec


def read_mask(it):
    m = it["img_mask"]
    return (m[0].numpy() if m.ndim == 3 else m.numpy()) > 0.5


def run(args):
    import torch
    import AnomalyCLIP_lib
    from AnomalyCLIP_lib.AnomalyCLIP import Attention
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import ExtentConditioner, visual_descriptor, conditioned_text_features
    from metrics import roc_auc_lowmem
    import vvattn_core as C

    if os.path.exists(args.out_csv) and not args.redo:
        print(f"{args.out_csv} exists: skipping {args.name} (use --redo to overwrite)")
        return
    ranking_json = json.load(open(args.ranking))
    ranking = [tuple(x) for x in ranking_json["ranking"]]
    assert len(ranking) == S.N_PAIRS

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
    if ckpt.get("extent_cond") != "extent":
        raise SystemExit("STOP: needs the 'extent' ECP checkpoint (fixed z override)")
    cond = ExtentConditioner(mode=ckpt["extent_cond"]).to(device)
    cond.load_state_dict(ckpt["conditioner"])
    cond.eval()

    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    ref = None
    if args.reference_csv:
        ref = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(args.reference_csv))}
    else:
        print("WARNING: no --reference_csv, EXP-012 parity NOT checked")

    t0 = time.time()
    kept, areas = [], []
    for i in range(len(data)):
        gt = read_mask(data[i])
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        kept.append(i); areas.append(float(gt.mean()))
    if args.limit:
        sel = np.sort(np.random.RandomState(0).permutation(len(kept))[: args.limit])
        kept = [kept[j] for j in sel]; areas = [areas[j] for j in sel]
        print(f"--limit: {len(kept)} random kept images (smoke)")
    print(f"{args.name}: {len(kept)} kept images, pass 1 {time.time() - t0:.0f}s", flush=True)

    spec = arm_spec(ranking, device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(os.path.splitext(args.out_csv)[0] + "_masks.json", "w") as f:        # provenance of every mask
        json.dump({a: (None if k is None else np.argwhere(k).tolist()) for a, (_, k) in spec.items()}, f)
    part = args.out_csv + ".partial"
    done = set()
    if args.redo and os.path.exists(part):
        os.remove(part)
    if os.path.exists(part):
        done = {r["image"] for r in csv.DictReader(open(part))}
        print(f"resuming: {len(done)} images already in {part}")
    fields = ["dataset", "image", "log_area"] + COLS
    fh = open(part, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if not done:
        w.writeheader()

    state = C.new_state()
    S_img = args.image_size

    def encode(img, arm):
        kind, keep = spec[arm]
        C.set_arm(state, kind, keep, device)
        with torch.no_grad():
            return model.encode_image(img, [24], DPAM_layer=20)

    def amap(pf, tf):
        with torch.no_grad():
            pfeat = pf[0] / pf[0].norm(dim=-1, keepdim=True)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfeat, tf[0])
            sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], S_img)
            return ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu().numpy()

    # sanity: patched forward with the default state == the original forward (first image)
    img0 = data[int(kept[0])]["img"].unsqueeze(0).to(device)
    with torch.no_grad():
        _, pf_orig = model.encode_image(img0, [24], DPAM_layer=20)
    with C.patched_attention(model, state):
        _, pf_new = encode(img0, "base")
        diff = (pf_new[0].float() - pf_orig[0].float()).abs().max().item()
        print(f"sanity: max|patched(base) - original| = {diff:.2e} (must be < 1e-3)")
        assert diff < 1e-3
        # sanity: an all-ones head mask changes nothing
        C.set_arm(state, "head", np.ones((S.N_LAYERS, S.N_HEADS), bool), device)
        with torch.no_grad():
            _, pf_ones = model.encode_image(img0, [24], DPAM_layer=20)
        assert (pf_ones[0].float() - pf_new[0].float()).abs().max().item() == 0.0
        print("sanity: all-ones head mask == base (exact)")

        max_err, n_new, t1 = 0.0, 0, time.time()
        for n, i in enumerate(kept):
            it = data[int(i)]
            sid = os.path.relpath(it["img_path"], args.data_path)
            if sid in done:
                continue
            gt = read_mask(it)
            img = it["img"].unsqueeze(0).to(device)
            imf, pf = encode(img, "base")
            with torch.no_grad():
                imf = imf / imf.norm(dim=-1, keepdim=True)
                z = torch.full((1,), args.ec_z_pix, device=device)
                _, c_pos, c_neg = cond(visual_descriptor(imf, pf), z_override=z)
                tf = conditioned_text_features(model, pl, c_pos, c_neg)      # shared by all arms
            row = {"dataset": args.name, "image": sid, "log_area": float(np.log(gt.mean()))}
            for arm in ARMS:
                pf_a = pf if arm == "base" else encode(img, arm)[1]
                m = amap(pf_a, tf)
                row[f"{arm}@4"] = float(roc_auc_lowmem(gt, gaussian_filter(m, sigma=4).astype(np.float32)))
                row[f"{arm}@32"] = float(roc_auc_lowmem(gt, gaussian_filter(m, sigma=32).astype(np.float32)))
            if ref is not None:
                if sid not in ref:
                    raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
                err = abs(row["base@4"] - ref[sid]); max_err = max(max_err, err)
                if err > 1e-4:
                    raise SystemExit(f"STOP: parity failure on {sid}: base {row['base@4']:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
            w.writerow(row); fh.flush()
            n_new += 1
            if n_new == 1 or n_new % 25 == 0:
                print(f"{n + 1}/{len(kept)} images; {(time.time() - t1) / n_new:.2f} s/image; parity max err so far {max_err:.2e}", flush=True)
    fh.close()
    os.replace(part, args.out_csv)
    print(f"{n_new} new images in {time.time() - t1:.0f}s ({(time.time() - t1) / max(n_new, 1):.2f} s/image); parity max |base@4 - EXP-012| = {max_err:.2e}"
          + ("" if ref is not None else " (NOT CHECKED)"))
    print(f"wrote {args.out_csv}")
    report_set(args.name, load(args.out_csv))


def load(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k not in ("dataset", "image")}


def report_set(name, t):
    n = len(t["log_area"])
    print(f"\n=== {name} (n={n}); per-image pixel AUROC; delta = arm - base at the same sigma [95% paired image bootstrap CI] ===")
    print(f"{'arm':14s} {'AUC@4':>7s} {'d4':>8s} {'CI4':>17s} {'d4-ctrl':>8s} | {'AUC@32':>7s} {'d32':>8s}")
    for arm in ["base"] + REPORT_ARMS:
        a4, b4 = S.column(t, arm, 4), S.column(t, "base", 4)
        a32, b32 = S.column(t, arm, 32), S.column(t, "base", 32)
        if arm == "base":
            print(f"{arm:14s} {a4.mean():7.4f} {'':8s} {'':17s} {'':8s} | {a32.mean():7.4f}")
            continue
        m, lo, hi = S.paired_ci(a4 - b4)
        c = S.ARM_CONTROL.get(arm)
        dc = f"{np.mean(a4 - S.column(t, c, 4)):+8.4f}" if c else f"{'':8s}"
        print(f"{arm:14s} {a4.mean():7.4f} {m:+8.4f} [{lo:+.4f},{hi:+.4f}] {dc} | {a32.mean():7.4f} {a32.mean() - b32.mean():+8.4f}")
    for p in (50, 25):
        per = [float(np.mean(t[f'head_rand{p}_s{s}@4'] - t['base@4'])) for s in S.RAND_SEEDS]
        print(f"   head_rand{p} delta4 per seed 0/1/2: " + " ".join(f"{x:+.4f}" for x in per))


def report(args):
    tables = {}
    for ds in S.SETS:
        p = os.path.join(args.dir, f"vvattn_{ds}.csv")
        if os.path.exists(p):
            tables[ds] = load(p)
            report_set(ds, tables[ds])
    v = S.family_verdict(tables)
    print("\n=== PRE-REGISTERED RULE (EXP-045; sigma 4 decisive) ===")
    if v["verdict"] == "INCOMPLETE":
        print(f"INCOMPLETE: missing {v['missing']}")
        return
    for arm, r in v["arms"].items():
        cells = "  ".join(f"{ds}: d4={r['per_set'][ds]['delta']:+.4f} [lo {r['per_set'][ds]['lo']:+.4f}] vs ctrl {r['per_set'][ds]['dctrl']:+.4f}"
                          f"{' OK' if r['counts'][ds] else ''}{' LOSS' if r['loss'][ds] else ''}" for ds in S.SETS)
        print(f"{arm:11s} {'PASS' if r['pass'] else 'fail'}   {cells}")
    for fam, vd in v["verdict"].items():
        print(f"FAMILY {fam}: {vd}")
    print("OVERALL:", ", ".join(f"{k}={x}" for k, x in v["verdict"].items()),
          "(ADVANCE authorises one registered training screen only; CLOSE = not worth the training budget under this protocol)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("eval", "report"))
    ap.add_argument("--name", default="data", help="ClinicDB | Kvasir | ColonDB")
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None)
    ap.add_argument("--ranking", default="results/EXP-045/head_ranking_mvtec.json")
    ap.add_argument("--ec_z_pix", type=float, default=1.6)
    ap.add_argument("--out_csv", default="vvattn.csv")
    ap.add_argument("--dir", default="results/EXP-045")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    run(a) if a.mode == "eval" else report(a)
