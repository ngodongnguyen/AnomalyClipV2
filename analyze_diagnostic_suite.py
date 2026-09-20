"""
Diagnostic suite: many independent probes of AnomalyCLIP in ONE pass over a lesion dataset.

Model-based variants (all scored with per-image pixel-AUROC vs the GT mask):
  scale{S}        input resized to S x S (patch-scale intervention), S in --scales
  fuse_mean/max   fusion of the per-scale maps
  layer{L}        single ViT layers 6/12/18/24 with the trained prompts;  layers_sum = their sum
  sigma{0,2,8,16} different smoothing strengths (baseline sigma = --sigma)
  highpass / local_norm   local-contrast versions of the baseline map
  prop_a0.5/0.9   training-free feature-affinity propagation of the patch scores
  sigma24/32/48, fuse_mean_s16   stronger smoothing / fusion+smoothing
  sp100/sp300     region-mean of the map inside SLIC superpixels of the RGB image (edge-aware aggregation)
  inpaint_spec    glare pixels (gray>235) inpainted in the INPUT before the model sees it
Every AUROC is reported twice: over all pixels (test.py protocol) and restricted to the field of view (@fov).
Model-free reference maps (free_*): centre prior, redness, smoothness, brightness, saturation.
Plus ORACLE rows (per-image best variant) = headroom of an adaptive selector, and a failure-typing
table (where the top-1% pixels land: lesion / glare / dark lumen / FOV edge).

Baseline = scale518 (layer 24, gaussian sigma) -- same recipe as test.py, evaluated at --eval_size.
"""
import os
import csv
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from skimage.segmentation import slic

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform
from diag_utils import (gauss, norm01, safe_auc, raw_descriptors, model_free_maps, failure_typing,
                        region_mean, print_report)

LAYER_IDS = [6, 12, 18, 24]


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
    orig_pos = model.visual.positional_embedding.data.clone()

    prompts, tokenized, compound = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized, compound).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    E = args.eval_size
    sc = E / 518.0
    sg = args.sigma * sc
    scales = sorted(set(args.scales) | {518})

    def encode(x, flist):
        # the visual tower interpolates its positional embedding in place; restore before every forward
        model.visual.positional_embedding.data = orig_pos.clone()
        return model.encode_image(x, flist, DPAM_layer=20)[1]

    def abn_map(pf, size):
        pf = pf / pf.norm(dim=-1, keepdim=True)
        sim, _ = AnomalyCLIP_lib.compute_similarity(pf, text_features[0])
        sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], size)
        return ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].cpu().numpy(), sim, pf

    def propagate(pf, sim, alphas, k=16, tau=0.05, iters=20):
        f = pf[0, 1:]
        A = f @ f.T
        vals, idx = A.topk(k, dim=1)
        W = torch.zeros_like(A).scatter_(1, idx, torch.softmax(vals / tau, dim=1))
        s0 = (sim[0, 1:, 1] + 1 - sim[0, 1:, 0]) / 2.0
        out = {}
        for a in alphas:
            s = s0.clone()
            for _ in range(iters):
                s = (1 - a) * s0 + a * (W @ s)
            side = int(s.numel() ** 0.5)
            m = F.interpolate(s.reshape(1, 1, side, side), size=(E, E), mode="bilinear", align_corners=False)
            out[a] = m[0, 0].cpu().numpy()
        return out

    rows = []
    model_variants, free_variants = None, None
    for i, items in enumerate(tqdm(loader)):
        if args.limit and i >= args.limit:
            break
        img_path = items["img_path"][0]
        gt = F.interpolate(items["img_mask"].float(), size=(E, E), mode="nearest")[0, 0].numpy() > 0.5
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        rgb = np.array(Image.open(img_path).convert("RGB").resize((E, E), Image.BILINEAR))
        desc = raw_descriptors(rgb, gt)
        if desc is None:
            continue
        free, gray, valid = model_free_maps(rgb, sg)

        img = items["img"].to(device)
        V, raw = {}, {}
        with torch.no_grad():
            for s in scales:
                x = img if s == 518 else F.interpolate(img, size=(s, s), mode="bicubic", align_corners=False)
                pfs = encode(x, LAYER_IDS if s == 518 else [24])
                m, sim24, pf24 = abn_map(pfs[-1], E)
                raw[s] = m
                V[f"scale{s}"] = gauss(m, sg)
                if s == 518:
                    layer_raw = []
                    for li, pf in zip(LAYER_IDS, pfs):
                        lm, _, _ = abn_map(pf, E)
                        layer_raw.append(lm)
                        V[f"layer{li}"] = gauss(lm, sg)
                    V["layers_sum"] = gauss(np.sum(layer_raw, 0), sg)
                    props = propagate(pf24, sim24, (0.5, 0.9))
                    for a, pm in props.items():
                        V[f"prop_a{a}"] = gauss(pm, sg)

        V["fuse_mean"] = np.mean([norm01(V[f"scale{s}"]) for s in scales], 0)
        V["fuse_max"] = np.max([norm01(V[f"scale{s}"]) for s in scales], 0)
        V["fuse_mean_s16"] = gauss(np.mean([norm01(raw[s]) for s in scales], 0), 16 * sc)
        base = raw[518]
        for sgv in (0, 2, 8, 16, 24, 32, 48):
            V[f"sigma{sgv}"] = gauss(base, sgv * sc)
        smooth, mu = gauss(base, sg), gauss(base, 20 * sc)
        for n_seg in (100, 300):
            V[f"sp{n_seg}"] = region_mean(smooth, slic(rgb, n_segments=n_seg, compactness=10, start_label=0))

        full = np.array(Image.open(img_path).convert("RGB").resize((518, 518), Image.BICUBIC))
        spec518 = binary_dilation(cv2.cvtColor(full, cv2.COLOR_RGB2GRAY) > 235, iterations=3)
        if spec518.any():
            inp = cv2.inpaint(full, spec518.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
            with torch.no_grad():
                m_i, _, _ = abn_map(encode(preprocess(Image.fromarray(inp)).unsqueeze(0).to(device), [24])[-1], E)
            V["inpaint_spec"] = gauss(m_i, sg)
        else:
            V["inpaint_spec"] = V["scale518"]

        V["highpass"] = smooth - mu
        V["local_norm"] = (smooth - mu) / (np.sqrt(gauss((base - mu) ** 2, 20 * sc)) + 1e-6)

        if model_variants is None:
            order = [f"scale{s}" for s in scales] + ["fuse_mean", "fuse_max", "fuse_mean_s16"] + \
                    [f"layer{l}" for l in LAYER_IDS] + \
                    ["layers_sum", "sigma0", "sigma2", "sigma8", "sigma16", "sigma24", "sigma32", "sigma48",
                     "highpass", "local_norm", "sp100", "sp300", "prop_a0.5", "prop_a0.9", "inpaint_spec"]
            model_variants = [v for v in order if v in V]
            free_variants = list(free.keys())

        gt_flat = gt.ravel().astype(np.uint8)
        row = {"image": os.path.basename(img_path), **desc}
        gt_v = gt[valid].astype(np.uint8)
        for name, m in {**V, **free}.items():
            row[name] = safe_auc(gt_flat, m)
            row[name + "@fov"] = safe_auc(gt_v, m[valid])
        row.update(failure_typing(V["scale518"], gt, gray, valid))
        rows.append(row)

    name = os.path.basename(args.data_path.rstrip("/"))
    out_csv = args.out_csv or f"diag_suite_{args.dataset}_{name}.csv"
    cols = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n##### Diagnostic suite: {name}  (CSV: {out_csv}) #####")
    print(f"eval_size={E}, sigma={args.sigma} (scaled to {sg:.2f}px at eval size), scales={scales}")
    print_report(rows, model_variants, free_variants, base="scale518")
    print_report(rows, model_variants, free_variants, base="scale518", suffix="@fov", typing=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("AnomalyCLIP diagnostic suite")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--scales", type=int, nargs="+", default=[224, 336, 518, 700])
    ap.add_argument("--eval_size", type=int, default=259, help="resolution used to score AUROC (259 = 518/2, faster)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images (smoke test)")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    ap.add_argument("--sigma", type=int, default=4)
    args = ap.parse_args()
    print(args)
    run(args)
