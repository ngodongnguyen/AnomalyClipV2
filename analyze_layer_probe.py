"""
Where is the information about a lesion lost? Layer-wise lesion/background separability, no training.

Question: for large lesions, is the lesion already indistinguishable from normal tissue in the backbone
features (=> needs trainable visual adaptation), or is it separable in the features (esp. the final
V-V patch features that are scored) but lost by the fixed zero-shot text-prompt readout (=> a better
readout can fix it)?

For every layer l = 1..24 and two token streams
  v : patch features of the V-V path (exactly what is scored; after ln_post @ proj)
  o : standard-CLIP residual stream x_ori at the output of block l (forward hooks)
it measures, per image, on valid (non-border) native-grid tokens:
  probe    : cross-fitted mean-difference direction (uses the image's own GT -> an UPPER BOUND on
             linear lesion/background separability, not available at test time)
  salience : unsupervised AUROC of "cosine distance to the image's median token"
and, for reference, the AUROC of the actual readout (trained prompt, last layer) on the same tokens.
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
from layer_probe_stats import auroc, make_split, crossfit_probe_auc, salience_auc, print_layer_report

N_LAYERS = 24


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
    blocks = model.visual.transformer.resblocks
    for l in range(N_LAYERS):
        blocks[l].register_forward_hook(lambda m, i, o, l=l: captured.__setitem__(l, o))

    rng = np.random.default_rng(args.seed)
    metrics = ["probe_v", "probe_o", "salience_v", "salience_o"]
    res = {m: [] for m in metrics}
    readout, area_frac, names = [], [], []

    for items in tqdm(loader):
        img_path = items["img_path"][0]
        with torch.no_grad():
            captured.clear()
            _, patch_features = model.encode_image(items["img"].to(device), list(range(1, N_LAYERS + 1)),
                                                   DPAM_layer=20)
            side = int((patch_features[0].shape[1] - 1) ** 0.5)

            gt = items["img_mask"].float()
            lesion = (F.adaptive_avg_pool2d(gt, (side, side))[0, 0].numpy() > 0.5)
            gray = np.array(Image.open(img_path).convert("L").resize((side, side), Image.BILINEAR))
            valid = gray > 10
            y = lesion.ravel()[valid.ravel()]
            split = make_split(y, rng)
            if split is None:
                continue
            valid_t = torch.from_numpy(valid.ravel()).to(device)

            pfn = patch_features[-1] / patch_features[-1].norm(dim=-1, keepdim=True)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfn, text_features[0])
            sim_native = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], (side, side))
            abn = ((sim_native[..., 1] + 1 - sim_native[..., 0]) / 2.0)[0].cpu().numpy().ravel()[valid.ravel()]
            readout.append(auroc(y, abn))

            row = {m: np.full(N_LAYERS, np.nan) for m in metrics}
            for l in range(N_LAYERS):
                fv = F.normalize(patch_features[l][0, 1:, :].float(), dim=-1)[valid_t].cpu().numpy()
                out = captured[l]
                x_ori = out[1] if isinstance(out, (list, tuple)) else out  # [N+1, B, C]
                fo = F.normalize(F.layer_norm(x_ori[1:, 0, :].float(), (x_ori.shape[-1],)), dim=-1)[valid_t]
                fo = fo.cpu().numpy()
                for tag, f in (("v", fv), ("o", fo)):
                    row[f"probe_{tag}"][l] = crossfit_probe_auc(f, y, split)
                    row[f"salience_{tag}"][l] = salience_auc(f, y)
        for m in metrics:
            res[m].append(row[m])
        area_frac.append(lesion.sum() / max(valid.sum(), 1))
        names.append(os.path.basename(img_path))

    name = os.path.basename(args.data_path.rstrip("/"))
    res = {m: np.stack(v) for m, v in res.items()}
    print_layer_report(name, res, np.array(readout), np.array(area_frac), args.report_layers)

    out_csv = args.out_csv or f"layer_probe_{args.dataset}_{name}.csv"
    with open(out_csv, "w", newline="") as f:
        cols = ["image", "area_frac", "readout24"] + [f"{m}:L{l + 1}" for m in metrics for l in range(N_LAYERS)]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, nm in enumerate(names):
            r = {"image": nm, "area_frac": area_frac[i], "readout24": readout[i]}
            for m in metrics:
                r.update({f"{m}:L{l + 1}": res[m][i, l] for l in range(N_LAYERS)})
            w.writerow(r)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Layer-wise lesion/background separability")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_layers", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    args = ap.parse_args()
    print(args)
    run(args)
