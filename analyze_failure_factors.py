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
from extent_prompt import ExtentConditioner, visual_descriptor, conditioned_text_features

FEATURES = ["log_area", "lab_contrast", "tex_ratio", "specular_frac", "dark_frac", "log_sharpness",
            "solidity", "compactness"]


def shape_descriptors(gt):
    """Scale-invariant boundary-shape descriptors of the largest lesion component in a binary mask.
    solidity    : area / convex-hull area. 1.0 for any convex shape (circle, oval, even elongated ones);
                  LOWER for a lobulated / budding / concave margin -- the morphology of irregular,
                  laterally-spreading lesions, independent of how big or round-on-average the blob is.
    compactness : 4*pi*area / perimeter^2 (isoperimetric ratio). 1.0 for a perfect circle; lower for an
                  elongated OR a rough/irregular boundary (a jagged outline has extra perimeter for its area).
    n_components: number of separate foreground blobs in the mask (usually 1; >1 flags multi-fragment lesions
                  or annotation artifacts, reported for context, not used as a shape covariate).
    """
    m = (gt > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    if area < 10 or perim <= 0 or hull_area <= 0:
        return None
    return dict(solidity=float(area / hull_area), compactness=float(4 * np.pi * area / (perim ** 2)),
                n_components=len(contours))


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

    shape = shape_descriptors(gt)
    if shape is None:
        return None

    out = dict(
        log_area=float(np.log(lesion.sum() / valid.sum())),
        lab_contrast=lab_contrast,
        tex_ratio=tex_ratio,
        specular_frac=specular_frac,
        dark_frac=dark_frac,
        log_sharpness=float(np.log(sharp + 1e-6)),
    )
    out.update(shape)
    return out


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

        desc["auroc"] = float(roc_auc_score(gt.ravel(), amap.ravel()))
        desc["image"] = os.path.basename(img_path)
        rows.append(desc)

    out_csv = args.out_csv or f"failure_factors_{args.dataset}_{os.path.basename(args.data_path.rstrip('/'))}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "auroc", "n_components"] + FEATURES)
        w.writeheader()
        w.writerows(rows)

    n_multi = sum(1 for r in rows if r["n_components"] > 1)
    print(f"multi-fragment masks (n_components>1): {n_multi}/{len(rows)} ({100 * n_multi / len(rows):.1f}%)")

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

    # size-controlled factor check: within each area tercile, compare AUROC of the bottom vs top tercile of
    # the factor -- disentangles the factor from the already-established size effect, and does not assume the
    # linear regression's functional form or trust individual coefficients when two factors are collinear
    # (solidity/compactness flipped sign against each other in some datasets -- a regression-only read would
    # have been misleading there).
    area = X[:, FEATURES.index("log_area")]

    def tercile_check(label, factor):
        edges = np.quantile(area, [0, 1 / 3, 2 / 3, 1])
        rows_out = []
        for t in range(3):
            lo, hi = edges[t], edges[t + 1]
            sel = (area >= lo) & (area <= hi if t == 2 else area < hi)
            ys, fv = y[sel], factor[sel]
            f_lo, f_hi = np.quantile(fv, 1 / 3), np.quantile(fv, 2 / 3)
            a_lo, a_hi = ys[fv <= f_lo].mean(), ys[fv >= f_hi].mean()
            rows_out.append((sel.sum(), a_lo, a_hi, a_hi - a_lo))
        print(f"\n{label} effect within area tercile (bottom-tercile vs top-tercile of the factor):")
        print(f"{'area tercile':<14}{'n':>5}{'low-factor AUROC':>18}{'high-factor AUROC':>19}{'delta':>9}")
        for t, (n_t, a_lo, a_hi, d) in enumerate(rows_out, start=1):
            print(f"T{t:<13}{n_t:>5}{a_lo:>18.3f}{a_hi:>19.3f}{d:>+9.3f}")
        return [r[3] for r in rows_out]

    tercile_check("solidity (1.0=convex/regular margin, lower=lobulated/irregular)", X[:, FEATURES.index("solidity")])
    tercile_check("compactness (1.0=circle, lower=elongated/irregular boundary)", X[:, FEATURES.index("compactness")])
    tercile_check("tex_ratio (higher=lesion rougher than surroundings, lower=lesion blends in texturally)",
                  X[:, FEATURES.index("tex_ratio")])


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
    ap.add_argument("--ec_z_pix", type=float, default=None,
                    help="ECP checkpoint: fixed z for the pixel head (omit for plain checkpoints)")
    args = ap.parse_args()
    print(args)
    run(args)
