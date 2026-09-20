"""
Failure-factor analysis: which properties of the RAW image (independent of CLIP
features) explain low per-image pixel-AUROC of AnomalyCLIP?

Per image it computes pixel-AUROC exactly like test.py (518x518 anomaly map,
gaussian-smoothed) and descriptors from the raw RGB image + GT mask:
  log_area      : log of lesion area fraction
  lab_contrast  : Lab colour distance between mean lesion colour and mean
                  colour of the rest of the visible field of view
  tex_ratio     : log(gradient energy inside lesion / outside lesion)
                  (< 0 => lesion smoother than its surroundings)
  specular_frac : fraction of visible pixels that are near-white glare
  dark_frac     : fraction of visible pixels that are very dark (lumen)
  log_sharpness : log variance of the Laplacian over visible pixels (focus)

Writes a per-image CSV and prints univariate correlations plus a standardized
multiple regression of AUROC on the descriptors.
"""
import os
import csv
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy import stats
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab
from sklearn.metrics import roc_auc_score

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform

FEATURES = ["log_area", "lab_contrast", "tex_ratio", "specular_frac", "dark_frac", "log_sharpness"]


def descriptors(img_path, gt, size):
    rgb = np.array(Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid = gray > 10
    lesion = (gt > 0.5) & valid
    outside = (~(gt > 0.5)) & valid
    if lesion.sum() < 10 or outside.sum() < 10:
        return None

    lab = rgb2lab(rgb)
    lab_contrast = float(np.linalg.norm(lab[lesion].mean(0) - lab[outside].mean(0)))

    gx = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    tex_ratio = float(np.log((grad[lesion].mean() + 1e-6) / (grad[outside].mean() + 1e-6)))

    n_valid = valid.sum()
    specular_frac = float(((gray > 235) & valid).sum() / n_valid)
    dark_frac = float(((gray < 50) & valid).sum() / n_valid)
    sharp = float(cv2.Laplacian(gray, cv2.CV_32F)[valid].var())

    return dict(
        log_area=float(np.log(lesion.sum() / valid.sum())),
        lab_contrast=lab_contrast,
        tex_ratio=tex_ratio,
        specular_frac=specular_frac,
        dark_frac=dark_frac,
        log_sharpness=float(np.log(sharp + 1e-6)),
    )


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
        gt = items["img_mask"][0, 0].numpy()
        gt = (gt > 0.5).astype(np.float32)
        if gt.min() == gt.max():
            continue
        desc = descriptors(img_path, gt, args.image_size)
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

        desc["auroc"] = float(roc_auc_score(gt.ravel(), amap.ravel()))
        desc["image"] = os.path.basename(img_path)
        rows.append(desc)

    out_csv = args.out_csv or f"failure_factors_{args.dataset}_{os.path.basename(args.data_path.rstrip('/'))}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "auroc"] + FEATURES)
        w.writeheader()
        w.writerows(rows)

    y = np.array([r["auroc"] for r in rows])
    X = np.array([[r[k] for k in FEATURES] for r in rows])
    n, p = X.shape

    print(f"\n=== Failure-factor report: {os.path.basename(args.data_path.rstrip('/'))} (n={n}) ===")
    print(f"mean per-image AUROC = {y.mean():.4f}   (CSV: {out_csv})")
    print(f"\n{'factor':<15}{'pearson':>10}{'spearman':>10}")
    for j, k in enumerate(FEATURES):
        print(f"{k:<15}{stats.pearsonr(X[:, j], y)[0]:>+10.3f}{stats.spearmanr(X[:, j], y)[0]:>+10.3f}")

    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    resid = ys - Xs @ beta
    dof = n - p - 1
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xs.T @ Xs)))
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - resid.var() / ys.var()

    print(f"\nStandardized multiple regression: AUROC ~ factors   (R^2 = {r2:.3f})")
    print(f"{'factor':<15}{'beta':>10}{'p-value':>12}")
    for k, b, pv in zip(FEATURES, beta, pval):
        print(f"{k:<15}{b:>+10.3f}{pv:>12.2g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("AnomalyCLIP failure-factor analysis")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
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
