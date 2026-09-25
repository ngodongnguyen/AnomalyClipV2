"""
Can lesion extent be estimated at all, without training a predictor? (no training)

Runs a checkpoint WITHOUT extent conditioning (e.g. zoom_mvtec), builds the anomaly map of every image and
computes label-free proxies of "how much of the image is anomalous" from the map alone. Reports the Spearman
correlation of each proxy with the true log lesion area. If no proxy correlates well (>~0.6), a two-pass
(map-based) extent estimator is not viable and the extent signal cannot be used at test time.
"""
import os
import csv
import argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
from tqdm import tqdm

import AnomalyCLIP_lib
from dataset import Dataset
from utils import get_transform
from analyze_transfer_probe import build_model


def otsu_frac(m, bins=64):
    h, edges = np.histogram(m, bins=bins)
    p = h / h.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    t = edges[int(np.nanargmax(s)) + 1]
    return float((m > t).mean())


def proxies(m):
    z = (m - m.mean()) / (m.std() + 1e-8)
    mm = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return {"frac>mean+0.5sd": float((z > 0.5).mean()), "frac>mean+1sd": float((z > 1).mean()),
            "frac>0.5(minmax)": float((mm > 0.5).mean()), "mean(minmax)": float(mm.mean()),
            "frac>otsu": otsu_frac(m), "std": float(m.std())}


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, target_transform = get_transform(args)
    model, text_features, _ = build_model(args, device)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    rows, area = [], []
    for items in tqdm(loader):
        gt = items["img_mask"][0, 0].numpy() > 0.5
        if gt.sum() < 20:
            continue
        with torch.no_grad():
            _, pfs = model.encode_image(items["img"].to(device), args.features_list, DPAM_layer=20)
            total = 0
            for pf in pfs:
                pf = pf / pf.norm(dim=-1, keepdim=True)
                sim, _ = AnomalyCLIP_lib.compute_similarity(pf, text_features[0])
                sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], args.image_size)
                total = total + (sm[..., 1] + 1 - sm[..., 0]) / 2.0
            m = total[0].cpu().numpy()
        m = gaussian_filter(m, args.sigma)[::4, ::4]
        rows.append(proxies(m))
        area.append(gt.mean())
    la = np.log(np.clip(np.array(area), 1e-4, None))
    name = os.path.basename(args.data_path.rstrip("/"))
    print(f"\n##### {name}  n={len(la)}  sigma={args.sigma}  (Spearman with true log lesion area) #####")
    for k in rows[0]:
        print(f"  {k:<20} {spearman(np.array([r[k] for r in rows]), la):+.3f}")
    print(f"  (true log-area: median {np.median(la):.2f}, IQR [{np.quantile(la, .25):.2f}, {np.quantile(la, .75):.2f}])")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sigma", type=float, default=16)
    ap.add_argument("--features_list", type=int, nargs="+", default=[24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    args = ap.parse_args()
    run(args)
