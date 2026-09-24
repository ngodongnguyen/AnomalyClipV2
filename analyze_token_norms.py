"""
Cheap, training-free test of the "artifact / high-norm outlier token" hypothesis
(cf. "Vision Transformers Need Registers", Darcet et al. ICLR'24; "Don't Need Trained Registers", NeurIPS'25).

Idea being tested: large ViTs recycle tokens of redundant/homogeneous image regions as global-information
storage (high-norm outlier tokens that lose their local content). A large, smooth lesion is exactly such a
redundant region, which would explain why its interior gets no useful anomaly signal.

For every image this records token norms of
  * feat_final : the patch features actually used for scoring (V-V path, after ln_post @ proj)
  * ori{L}     : the standard-CLIP residual stream (x_ori) at the output of block L (forward hooks),
                 which is where CLIP artifacts normally arise and which feeds the V-V attention,
flags per-image robust outliers, and reports where they sit (lesion / normal tissue / black border)
and whether they go together with poor localisation.
"""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform
from token_stats import image_metrics, print_token_report


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

    captured = {}

    def make_hook(name):
        def hook(module, inputs, output):
            captured[name] = output
        return hook

    blocks = model.visual.transformer.resblocks
    for L in args.hook_layers:
        blocks[L - 1].register_forward_hook(make_hook(f"ori{L}"))
    streams = ["feat_final"] + [f"ori{L}" for L in args.hook_layers]

    rows = {s: [] for s in streams}
    area_frac, auroc, names = [], [], []

    for items in tqdm(loader):
        img_path = items["img_path"][0]
        with torch.no_grad():
            captured.clear()
            _, patch_features = model.encode_image(items["img"].to(device), args.features_list, DPAM_layer=20)
            pf = patch_features[-1]  # [1, N+1, C], raw (not yet L2-normalised)
            n_patch = pf.shape[1] - 1
            side = int(n_patch ** 0.5)

            pfn = pf / pf.norm(dim=-1, keepdim=True)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfn, text_features[0])
            sim_native = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], (side, side))
            abn = ((sim_native[..., 1] + 1 - sim_native[..., 0]) / 2.0)[0].cpu().numpy()

            norm_maps = {"feat_final": pf[0, 1:, :].float().norm(dim=-1).reshape(side, side).cpu().numpy()}
            for L in args.hook_layers:
                out = captured[f"ori{L}"]
                x_ori = out[1] if isinstance(out, (list, tuple)) else out  # [N+1, B, C]
                norm_maps[f"ori{L}"] = x_ori[1:, 0, :].float().norm(dim=-1).reshape(side, side).cpu().numpy()

        gt = items["img_mask"].float()
        lesion = (F.adaptive_avg_pool2d(gt, (side, side))[0, 0].numpy() > 0.5)
        gray = np.array(Image.open(img_path).convert("L").resize((side, side), Image.BILINEAR))
        valid = gray > 10
        if lesion.sum() < 2 or (valid & ~lesion).sum() < 2:
            continue
        auroc.append(roc_auc_score(lesion.ravel().astype(int), abn.ravel()))
        area_frac.append(lesion.sum() / max(valid.sum(), 1))
        names.append(os.path.basename(img_path))
        for s in streams:
            rows[s].append(image_metrics(norm_maps[s], lesion, valid, abn, z_thresh=args.z))

    name = os.path.basename(args.data_path.rstrip("/"))
    print(f"\n##### Token-norm analysis: {name}  (n={len(names)}, robust z > {args.z}) #####")
    for s in streams:
        print_token_report(s, rows[s], np.array(area_frac), np.array(auroc))

    out_csv = args.out_csv or f"token_norms_{args.dataset}_{name}.csv"
    with open(out_csv, "w", newline="") as f:
        cols = ["image", "area_frac", "auroc_native"] + [f"{s}:{k}" for s in streams for k in rows[s][0].keys()]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, nm in enumerate(names):
            r = {"image": nm, "area_frac": area_frac[i], "auroc_native": auroc[i]}
            for s in streams:
                r.update({f"{s}:{k}": v for k, v in rows[s][i].items()})
            w.writerow(r)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Token-norm / artifact-token diagnostic")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--hook_layers", type=int, nargs="+", default=[12, 18, 24])
    ap.add_argument("--z", type=float, default=4.0, help="robust z-score threshold for an outlier token")
    ap.add_argument("--features_list", type=int, nargs="+", default=[24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    args = ap.parse_args()
    print(args)
    run(args)
