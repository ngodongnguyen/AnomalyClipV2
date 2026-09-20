"""
Core-vs-boundary diagnostic.

Tests the mechanism hypothesis: for large/smooth lesions AnomalyCLIP scores the
lesion BOUNDARY high but leaves the lesion CORE cold.

Per image the anomaly map is rank-normalised to percentiles inside the visible
field of view (so images are comparable), then averaged over four regions:
  core      : inside GT, farther than `band` px from the GT boundary
  band_in   : inside GT, within `band` px of the boundary
  band_out  : outside GT, within `band` px of the boundary
  far_out   : outside GT, farther than `band` px from the boundary
Prints the region profile per lesion-area quartile and writes a per-image CSV.
"""
import os
import csv
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy import stats
from scipy.ndimage import gaussian_filter, distance_transform_edt

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform
from analyze_failure_factors import descriptors


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

    rows = []
    for items in tqdm(loader):
        img_path = items["img_path"][0]
        gt = (items["img_mask"][0, 0].numpy() > 0.5)
        if gt.all() or not gt.any():
            continue
        desc = descriptors(img_path, gt.astype(np.float32), args.image_size)
        if desc is None:
            continue

        with torch.no_grad():
            _, patch_features = model.encode_image(items["img"].to(device), args.features_list, DPAM_layer=20)
            maps = []
            for idx, pf in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    pf = pf / pf.norm(dim=-1, keepdim=True)
                    sim, _ = AnomalyCLIP_lib.compute_similarity(pf, text_features[0])
                    sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], args.image_size)
                    maps.append((sm[..., 1] + 1 - sm[..., 0]) / 2.0)
            amap = torch.stack(maps).sum(0)[0].cpu().numpy()
            amap = gaussian_filter(amap, sigma=args.sigma)

        rgb = np.array(Image.open(img_path).convert("L").resize((args.image_size, args.image_size), Image.BILINEAR))
        valid = rgb > 10
        pct = np.zeros_like(amap)
        pct[valid] = stats.rankdata(amap[valid]) / valid.sum()

        d_in = distance_transform_edt(gt)
        d_out = distance_transform_edt(~gt)
        b = args.band
        regions = {
            "core": gt & (d_in > b),
            "band_in": gt & (d_in <= b),
            "band_out": (~gt) & (d_out <= b) & valid,
            "far_out": (~gt) & (d_out > b) & valid,
        }
        if any(m.sum() < 100 for m in regions.values()):
            continue

        row = {"image": os.path.basename(img_path), "log_area": desc["log_area"], "tex_ratio": desc["tex_ratio"]}
        for k, m in regions.items():
            row[k] = float(pct[m].mean())
        row["core_minus_band"] = row["core"] - row["band_in"]
        rows.append(row)

    name = os.path.basename(args.data_path.rstrip("/"))
    out_csv = args.out_csv or f"core_boundary_{args.dataset}_{name}.csv"
    cols = ["image", "log_area", "tex_ratio", "core", "band_in", "band_out", "far_out", "core_minus_band"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    arr = {c: np.array([r[c] for r in rows]) for c in cols[1:]}
    print(f"\n=== Core-vs-boundary report: {name} (n={n}, band={args.band}px, values = mean score percentile) ===")
    print(f"ALL   core={arr['core'].mean():.3f}  band_in={arr['band_in'].mean():.3f}  "
          f"band_out={arr['band_out'].mean():.3f}  far_out={arr['far_out'].mean():.3f}  "
          f"| core<band_in in {100 * (arr['core_minus_band'] < 0).mean():.0f}% of images")

    edges = np.quantile(arr["log_area"], [0, .25, .5, .75, 1])
    print(f"\n{'area quartile':<16}{'n':>5}{'core':>8}{'band_in':>9}{'band_out':>10}{'far_out':>9}{'core-band':>11}")
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        sel = (arr["log_area"] >= lo) & ((arr["log_area"] < hi) if q < 3 else (arr["log_area"] <= hi))
        print(f"Q{q + 1} ({'small' if q == 0 else 'large' if q == 3 else '     '})   {sel.sum():>5}"
              f"{arr['core'][sel].mean():>8.3f}{arr['band_in'][sel].mean():>9.3f}"
              f"{arr['band_out'][sel].mean():>10.3f}{arr['far_out'][sel].mean():>9.3f}"
              f"{arr['core_minus_band'][sel].mean():>+11.3f}")

    print("\nPearson corr of core_minus_band with:")
    for k in ["log_area", "tex_ratio"]:
        print(f"  {k:<10} {stats.pearsonr(arr[k], arr['core_minus_band'])[0]:+.3f}")
    print("Pearson corr of core percentile with:")
    for k in ["log_area", "tex_ratio"]:
        print(f"  {k:<10} {stats.pearsonr(arr[k], arr['core'])[0]:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("AnomalyCLIP core-vs-boundary diagnostic")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--band", type=int, default=12)
    ap.add_argument("--features_list", type=int, nargs="+", default=[24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    ap.add_argument("--feature_map_layer", type=int, nargs="+", default=[0])
    ap.add_argument("--sigma", type=int, default=4)
    args = ap.parse_args()
    print(args)
    run(args)
