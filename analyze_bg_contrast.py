"""
Training-free test of the "background-contrastive scoring" idea, on top of an already-trained
(frozen) checkpoint -- no training, just a cheap post-hoc fusion, to see if it's worth building
into a real module before spending GPU time on it.

For each image: take the native patch-grid features + the existing abnormal-probability map
(same recipe as test.py). Estimate a per-image "background" prototype = mean feature of the
patches the model itself currently scores lowest (bottom `bg_pct`% of abnormal-probability).
Score every patch by its dissimilarity to that background prototype (1 - cosine sim). Fuse this
contrast cue with the baseline map (rank-normalized sum, weight `alpha`) and compare per-image
pixel-AUROC by lesion-size quartile against the baseline alone (both smoothed with sigma=32,
the strongest known post-hoc baseline so far).
"""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform
from diag_utils import gauss, norm01, safe_auc, raw_descriptors


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()

    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                   dataset_name=args.dataset)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(ckpt["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized, compound = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized, compound).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    E = args.eval_size
    sc = E / 518.0
    sg = args.sigma * sc

    bg_pcts = args.bg_pcts
    alphas = args.alphas
    variant_names = ["baseline"] + [f"contrast_p{p}" for p in bg_pcts] + \
                     [f"fused_p{p}_a{a}" for p in bg_pcts for a in alphas]

    rows = []
    for items in tqdm(loader):
        img_path = items["img_path"][0]
        gt = F.interpolate(items["img_mask"].float(), size=(E, E), mode="nearest")[0, 0].numpy() > 0.5
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        rgb = np.array(Image.open(img_path).convert("RGB").resize((E, E), Image.BILINEAR))
        desc = raw_descriptors(rgb, gt)
        if desc is None:
            continue

        with torch.no_grad():
            _, patch_features = model.encode_image(items["img"].to(device), args.features_list, DPAM_layer=20)
            pf = patch_features[-1]
            pf = pf / pf.norm(dim=-1, keepdim=True)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pf, text_features[0])
            n_patch = sim.shape[1] - 1
            side = int(n_patch ** 0.5)
            sim_native = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], (side, side))
            abn_native = ((sim_native[..., 1] + 1 - sim_native[..., 0]) / 2.0)[0]  # [side, side]
            feat_native = pf[0, 1:, :].reshape(side, side, -1)  # already L2-normalized

            row = {"image": os.path.basename(img_path), "log_area": desc["log_area"]}
            base_up = F.interpolate(abn_native[None, None], size=(E, E), mode="bilinear",
                                     align_corners=False)[0, 0].cpu().numpy()
            base_smooth = gauss(base_up, sg)
            row["baseline"] = safe_auc(gt.ravel().astype(np.uint8), base_smooth)

            flat_abn = abn_native.reshape(-1)
            flat_feat = feat_native.reshape(-1, feat_native.shape[-1])
            for p in bg_pcts:
                k = max(1, int(len(flat_abn) * p / 100.0))
                bg_idx = torch.topk(flat_abn, k, largest=False).indices
                centroid = flat_feat[bg_idx].mean(0)
                centroid = centroid / centroid.norm()
                contrast_native = 1.0 - (feat_native @ centroid)  # [side, side]
                contrast_up = F.interpolate(contrast_native[None, None], size=(E, E), mode="bilinear",
                                             align_corners=False)[0, 0].cpu().numpy()
                contrast_smooth = gauss(contrast_up, sg)
                row[f"contrast_p{p}"] = safe_auc(gt.ravel().astype(np.uint8), contrast_smooth)
                for a in alphas:
                    fused = norm01(base_smooth) + a * norm01(contrast_smooth)
                    row[f"fused_p{p}_a{a}"] = safe_auc(gt.ravel().astype(np.uint8), fused)
        rows.append(row)

    name = os.path.basename(args.data_path.rstrip("/"))
    out_csv = args.out_csv or f"bg_contrast_{args.dataset}_{name}.csv"
    cols = ["image", "log_area"] + variant_names
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    la = np.array([r["log_area"] for r in rows])
    qid = np.digitize(la, np.quantile(la, [.25, .5, .75]))
    print(f"\n##### Background-contrast report: {name}  (n={n}, sigma={args.sigma}, eval_size={E}) #####")
    print(f"{'variant':<20}{'ALL':>7}{'Q1':>7}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'dALL':>8}{'dQ4':>8}")
    base = np.array([r["baseline"] for r in rows])
    for v in variant_names:
        a = np.array([r[v] for r in rows])
        cells = [a.mean()] + [a[qid == k].mean() for k in range(4)]
        d_all, d_q4 = a.mean() - base.mean(), a[qid == 3].mean() - base[qid == 3].mean()
        print(f"{v:<20}" + "".join(f"{c:>7.3f}" for c in cells) + f"{d_all:>+8.3f}{d_q4:>+8.3f}")
    cand = [v for v in variant_names if v != "baseline"]
    d_all = {v: np.array([r[v] for r in rows]).mean() - base.mean() for v in cand}
    d_q4 = {v: np.array([r[v] for r in rows])[qid == 3].mean() - base[qid == 3].mean() for v in cand}
    print("\nTop-3 by dALL:", ", ".join(f"{v} ({d_all[v]:+.3f})" for v in sorted(cand, key=lambda v: -d_all[v])[:3]))
    print("Top-3 by dQ4 :", ", ".join(f"{v} ({d_q4[v]:+.3f})" for v in sorted(cand, key=lambda v: -d_q4[v])[:3]))
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Background-contrastive scoring (training-free diagnostic)")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--bg_pcts", type=float, nargs="+", default=[5, 10, 20])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--eval_size", type=int, default=259)
    ap.add_argument("--sigma", type=int, default=32)
    ap.add_argument("--features_list", type=int, nargs="+", default=[24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    args = ap.parse_args()
    print(args)
    run(args)
