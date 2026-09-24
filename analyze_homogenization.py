"""
Does depth homogenise tokens faster inside large lesions? (over-smoothing / rank-collapse diagnostic; no training)

For every layer and both streams (v: V-V patch features that are scored; o: standard-CLIP residual stream) it
measures, on valid native-grid tokens of each image, how similar tokens are to each other overall (c_all), inside
the lesion (c_ll), inside normal tissue (c_bb), across the two (c_lb), and the separation index
s = (c_ll + c_bb)/2 - c_lb, in raw and image-mean-centred form. Results are averaged by lesion-size quartile.
"""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dataset import Dataset
from utils import get_transform
from homog_stats import homog_metrics, print_homog_report
from analyze_transfer_probe import build_model, layer_tokens, N_LAYERS

METRIC_KEYS = [f"{m}_{t}" for t in ("raw", "cent") for m in ("c_all", "c_ll", "c_bb", "c_lb", "s")]


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, target_transform = get_transform(args)
    model, _, captured = build_model(args, device)

    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                   dataset_name=args.dataset)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    res = {f"{k}_{s}": [] for k in METRIC_KEYS for s in ("v", "o")}
    area_frac, names = [], []
    for items in tqdm(loader):
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
            rows = {f"{k}_{s}": np.full(N_LAYERS, np.nan) for k in METRIC_KEYS for s in ("v", "o")}
            for l in range(N_LAYERS):
                for s, feats in (("v", v), ("o", o)):
                    m = homog_metrics(feats[l][valid_t].cpu().numpy().astype(np.float64), y)
                    for k, val in m.items():
                        rows[f"{k}_{s}"][l] = val
        for k, arr in rows.items():
            res[k].append(arr)
        area_frac.append(lesion.sum() / max(valid.sum(), 1))
        names.append(os.path.basename(img_path))

    name = os.path.basename(args.data_path.rstrip("/"))
    res = {k: np.stack(v) for k, v in res.items()}
    print_homog_report(name, res, np.array(area_frac), args.report_layers)

    out_csv = args.out_csv or f"homogenization_{args.dataset}_{name}.csv"
    with open(out_csv, "w", newline="") as f:
        cols = ["image", "area_frac"] + [f"{k}:L{l + 1}" for k in res for l in range(N_LAYERS)]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, nm in enumerate(names):
            r = {"image": nm, "area_frac": area_frac[i]}
            for k in res:
                r.update({f"{k}:L{l + 1}": res[k][i, l] for l in range(N_LAYERS)})
            w.writerow(r)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Token homogenisation vs depth")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--report_layers", type=int, nargs="+", default=[2, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    args = ap.parse_args()
    print(args)
    run(args)
