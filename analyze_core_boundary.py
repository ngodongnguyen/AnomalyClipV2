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
from scipy.ndimage import gaussian_filter, distance_transform_edt, sobel
from sklearn.metrics import roc_auc_score

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform
from analyze_failure_factors import descriptors
from extent_prompt import ExtentConditioner, visual_descriptor, conditioned_text_features


def save_hot_panels(cases, out_dir):
    """4 panels per case: image | GT | anomaly map | image with far false-positive hotspots red, GT contour green."""
    os.makedirs(out_dir, exist_ok=True)
    for rank, (fp, img_path, gt, amap, hot) in enumerate(cases, start=1):
        h, w = gt.shape
        img = np.array(Image.open(img_path).convert("RGB").resize((w, h), Image.BILINEAR))
        gt_rgb = np.repeat((gt * 255).astype(np.uint8)[..., None], 3, axis=2)
        a = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        heat = np.stack([a * 255, (1 - np.abs(2 * a - 1)) * 255, (1 - a) * 255], axis=2).astype(np.uint8)
        over = img.copy()
        over[hot] = (0.4 * over[hot] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        edge = gt & ~(np.roll(gt, 1, 0) & np.roll(gt, -1, 0) & np.roll(gt, 1, 1) & np.roll(gt, -1, 1))
        over[edge] = (0, 255, 0)
        panel = np.concatenate([img, gt_rgb, heat, over], axis=1)
        base = os.path.splitext(os.path.basename(img_path))[0]
        Image.fromarray(panel).save(os.path.join(out_dir, f"{rank:02d}_fphot{fp:.3f}_{base}.png"))


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

    conditioner = None
    if args.ec_z_pix is not None:
        conditioner = ExtentConditioner(mode=ckpt["extent_cond"]).to(device)
        conditioner.load_state_dict(ckpt["conditioner"])
        conditioner.eval()

    rows = []
    cats = ("specular", "dark", "fov_border", "strong_edge")
    hot_cnt = {c: 0 for c in cats}
    far_cnt = {c: 0 for c in cats}
    hot_tot = far_tot = 0
    hot_by_img = {c: [] for c in cats}
    keep = []  # (fp_hot, img_path, gt, amap, hot) downsampled, for rendering the worst cases
    for items in tqdm(loader):
        img_path = items["img_path"][0]
        gt = (items["img_mask"][0, 0].numpy() > 0.5)
        if gt.all() or not gt.any():
            continue
        desc = descriptors(img_path, gt.astype(np.float32), args.image_size)
        if desc is None:
            continue

        with torch.no_grad():
            image_features, patch_features = model.encode_image(items["img"].to(device), args.features_list, DPAM_layer=20)
            tf = text_features
            if conditioner is not None:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                z = torch.full((1,), args.ec_z_pix, device=device)
                _, c_pos, c_neg = conditioner(visual_descriptor(image_features, patch_features), z_override=z)
                tf = conditioned_text_features(model, prompt_learner, c_pos, c_neg)
            maps = []
            for idx, pf in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    pf = pf / pf.norm(dim=-1, keepdim=True)
                    sim, _ = AnomalyCLIP_lib.compute_similarity(pf, tf[0])
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
        # false-positive hotspots: share of far-from-lesion healthy pixels scoring above the lesion's median score
        row["fp_hot"] = float((amap[regions["far_out"]] > np.median(amap[gt])).mean())

        # what are the hotspots? image properties of hot far pixels vs all far pixels (base rate)
        far = regions["far_out"]
        hot = far & (amap > np.median(amap[gt]))
        g = rgb.astype(np.float64)
        grad = np.hypot(sobel(g, 0), sobel(g, 1))
        masks = {
            "specular": rgb > 230,
            "dark": (rgb < 40) & valid,
            "fov_border": distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1] < 20,
            "strong_edge": grad > np.quantile(grad[valid], 0.9),
        }
        hot_tot += int(hot.sum())
        far_tot += int(far.sum())
        for c in cats:
            hot_cnt[c] += int((hot & masks[c]).sum())
            far_cnt[c] += int((far & masks[c]).sum())
            if hot.sum() > 0:
                hot_by_img[c].append(float((hot & masks[c]).sum() / hot.sum()))
        if args.save_hot:
            keep.append((row["fp_hot"], img_path, gt[::2, ::2], amap[::2, ::2], hot[::2, ::2]))
        row["area_frac"] = float(np.exp(desc["log_area"]))
        for name_, key in (("auc_core_far", "core"), ("auc_band_far", "band_in")):
            pos, neg = amap[regions[key]], amap[regions["far_out"]]
            y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
            row[name_] = float(roc_auc_score(y, np.concatenate([pos, neg])))
        rows.append(row)

    name = os.path.basename(args.data_path.rstrip("/"))
    if args.save_hot:
        save_hot_panels(sorted(keep, key=lambda k: -k[0])[: args.save_hot], os.path.join(args.hot_dir, name))
    out_csv = args.out_csv or f"core_boundary_{args.dataset}_{name}.csv"
    cols = ["image", "log_area", "tex_ratio", "core", "band_in", "band_out", "far_out", "core_minus_band",
            "area_frac", "auc_core_far", "auc_band_far", "fp_hot"]
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
    print(f"false-positive hotspots: {100 * arr['fp_hot'].mean():.1f}% of far healthy pixels outscore the lesion median;"
          f" images with >5% such pixels: {100 * (arr['fp_hot'] > 0.05).mean():.0f}%")
    print(f"\nWhat the hotspots are (pooled over images; enrichment = share among hot / share among all far pixels):")
    print(f"{'category':<13}{'share of hot':>13}{'share of far':>13}{'enrichment':>12}")
    for c in cats:
        sh, sf = hot_cnt[c] / max(hot_tot, 1), far_cnt[c] / max(far_tot, 1)
        print(f"{c:<13}{sh:>13.3f}{sf:>13.3f}{sh / max(sf, 1e-9):>12.2f}")
    print(f"(hot far pixels: {hot_tot}; far pixels: {far_tot}; categories can overlap)")
    print(f"\n{'area quartile':<16}{'n':>5}{'area':>7}{'core':>8}{'band_in':>9}{'band_out':>10}{'far_out':>9}{'core-band':>11}{'AUC core/far':>14}{'AUC band/far':>14}{'fp_hot':>8}")
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        sel = (arr["log_area"] >= lo) & ((arr["log_area"] < hi) if q < 3 else (arr["log_area"] <= hi))
        print(f"Q{q + 1} ({'small' if q == 0 else 'large' if q == 3 else '     '})   {sel.sum():>5}"
              f"{arr['area_frac'][sel].mean():>7.2f}"
              f"{arr['core'][sel].mean():>8.3f}{arr['band_in'][sel].mean():>9.3f}"
              f"{arr['band_out'][sel].mean():>10.3f}{arr['far_out'][sel].mean():>9.3f}"
              f"{arr['core_minus_band'][sel].mean():>+11.3f}"
              f"{arr['auc_core_far'][sel].mean():>14.3f}{arr['auc_band_far'][sel].mean():>14.3f}"
              f"{arr['fp_hot'][sel].mean():>8.3f}")

    print("\nPearson corr of auc_core_far - auc_band_far (core lags rim) with:")
    lag = arr["auc_core_far"] - arr["auc_band_far"]
    for k in ["log_area", "tex_ratio"]:
        print(f"  {k:<10} {stats.pearsonr(arr[k], lag)[0]:+.3f}")
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
    ap.add_argument("--save_hot", type=int, default=0, help="render the N images with the most false-positive hotspots")
    ap.add_argument("--hot_dir", type=str, default="hotspots")
    ap.add_argument("--ec_z_pix", type=float, default=None,
                    help="ECP checkpoint: fixed z for the pixel head (omit for plain checkpoints)")
    args = ap.parse_args()
    print(args)
    run(args)
