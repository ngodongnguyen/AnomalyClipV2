"""
Fixed-readout transfer probe (no prompt training): mimic "learn a readout on industrial data, apply it
zero-shot to lesions", but as a plain linear direction per layer, so different layers can be compared.

Pass 1 (MVTec, cached): for each layer l and each stream (v = V-V patch features that are scored,
        o = standard-CLIP residual stream), d_l = mean(defect tokens) - mean(non-defect tokens), unit norm.
Pass 2 (target dataset): score every token with the FIXED d_l (no target labels used) and report the
        per-image AUROC by lesion-size quartile, for every layer, next to the actual trained-prompt readout.

Unlike analyze_layer_probe.py (direction fitted on the image's own GT = per-image upper bound), here the
direction is fixed and comes from another domain, i.e. what a zero-shot readout could actually do.
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
from layer_probe_stats import auroc, print_layer_report

N_LAYERS = 24


def build_model(args, device):
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
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
    captured = {}
    for l, blk in enumerate(model.visual.transformer.resblocks):
        blk.register_forward_hook(lambda m, i, o, l=l: captured.__setitem__(l, o))
    return model, text_features, captured


def layer_tokens(model, captured, image, device):
    """Returns (v_list, o_list, patch_features): per-layer L2-normalised token features, [N, C] each."""
    captured.clear()
    _, patch_features = model.encode_image(image.to(device), list(range(1, N_LAYERS + 1)), DPAM_layer=20)
    v, o = [], []
    for l in range(N_LAYERS):
        v.append(F.normalize(patch_features[l][0, 1:, :].float(), dim=-1))
        out = captured[l]
        x_ori = out[1] if isinstance(out, (list, tuple)) else out
        o.append(F.normalize(F.layer_norm(x_ori[1:, 0, :].float(), (x_ori.shape[-1],)), dim=-1))
    return v, o, patch_features


def learn_directions(args, model, captured, device, preprocess, target_transform):
    if args.dir_cache and os.path.isfile(args.dir_cache):
        print(f"loading cached MVTec directions from {args.dir_cache}")
        return torch.load(args.dir_cache, map_location=device)
    data = Dataset(root=args.mvtec_path, transform=preprocess, target_transform=target_transform, dataset_name="mvtec")
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    sums = {s: {"def": torch.zeros(N_LAYERS, c, device=device), "nor": torch.zeros(N_LAYERS, c, device=device)}
            for s, c in (("v", 768), ("o", 1024))}
    n_def = n_nor = 0
    for i, items in enumerate(tqdm(loader, desc="MVTec directions")):
        if args.mvtec_limit and i >= args.mvtec_limit:
            break
        with torch.no_grad():
            v, o, pf = layer_tokens(model, captured, items["img"], device)
            side = int((pf[0].shape[1] - 1) ** 0.5)
            lesion = (F.adaptive_avg_pool2d(items["img_mask"].float(), (side, side))[0, 0] > 0.25).reshape(-1).to(device)
            n_def += int(lesion.sum())
            n_nor += int((~lesion).sum())
            for l in range(N_LAYERS):
                for s, feats in (("v", v), ("o", o)):
                    sums[s]["def"][l] += feats[l][lesion].sum(0)
                    sums[s]["nor"][l] += feats[l][~lesion].sum(0)
    dirs = {}
    for s in ("v", "o"):
        d = sums[s]["def"] / max(n_def, 1) - sums[s]["nor"] / max(n_nor, 1)
        dirs[s] = F.normalize(d, dim=-1)
    print(f"MVTec directions from {n_def} defect tokens and {n_nor} normal tokens")
    if args.dir_cache:
        torch.save(dirs, args.dir_cache)
    return dirs


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, target_transform = get_transform(args)
    model, text_features, captured = build_model(args, device)
    dirs = learn_directions(args, model, captured, device, preprocess, target_transform)

    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                   dataset_name=args.dataset)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    res = {"transfer_v": [], "transfer_o": []}
    readout, area_frac, names = [], [], []
    for items in tqdm(loader, desc=args.dataset):
        img_path = items["img_path"][0]
        with torch.no_grad():
            v, o, pf = layer_tokens(model, captured, items["img"], device)
            side = int((pf[0].shape[1] - 1) ** 0.5)
            lesion = (F.adaptive_avg_pool2d(items["img_mask"].float(), (side, side))[0, 0].numpy() > 0.5)
            gray = np.array(Image.open(img_path).convert("L").resize((side, side), Image.BILINEAR))
            valid = gray > 10
            y = lesion.ravel()[valid.ravel()]
            if y.sum() < 8 or (~y).sum() < 8:
                continue
            valid_t = torch.from_numpy(valid.ravel()).to(device)

            last = pf[-1] / pf[-1].norm(dim=-1, keepdim=True)
            sim, _ = AnomalyCLIP_lib.compute_similarity(last, text_features[0])
            sim_native = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], (side, side))
            abn = ((sim_native[..., 1] + 1 - sim_native[..., 0]) / 2.0)[0].cpu().numpy().ravel()[valid.ravel()]
            readout.append(auroc(y, abn))

            row_v, row_o = np.full(N_LAYERS, np.nan), np.full(N_LAYERS, np.nan)
            for l in range(N_LAYERS):
                row_v[l] = auroc(y, (v[l][valid_t] @ dirs["v"][l]).cpu().numpy())
                row_o[l] = auroc(y, (o[l][valid_t] @ dirs["o"][l]).cpu().numpy())
        res["transfer_v"].append(row_v)
        res["transfer_o"].append(row_o)
        area_frac.append(lesion.sum() / max(valid.sum(), 1))
        names.append(os.path.basename(img_path))

    name = os.path.basename(args.data_path.rstrip("/"))
    res = {k: np.stack(v) for k, v in res.items()}
    print_layer_report(name + "  [fixed direction learned on MVTec]", res, np.array(readout),
                       np.array(area_frac), args.report_layers)
    out_csv = args.out_csv or f"transfer_probe_{args.dataset}_{name}.csv"
    with open(out_csv, "w", newline="") as f:
        cols = ["image", "area_frac", "readout24"] + [f"{m}:L{l + 1}" for m in res for l in range(N_LAYERS)]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, nm in enumerate(names):
            r = {"image": nm, "area_frac": area_frac[i], "readout24": readout[i]}
            for m in res:
                r.update({f"{m}:L{l + 1}": res[m][i, l] for l in range(N_LAYERS)})
            w.writerow(r)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Fixed-readout transfer probe (MVTec -> target)")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--mvtec_path", type=str, required=True)
    ap.add_argument("--mvtec_limit", type=int, default=0)
    ap.add_argument("--dir_cache", type=str, default="mvtec_layer_directions.pt")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--report_layers", type=int, nargs="+", default=[2, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    args = ap.parse_args()
    print(args)
    run(args)
