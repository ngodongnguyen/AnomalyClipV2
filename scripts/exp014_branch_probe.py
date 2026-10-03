#!/usr/bin/env python3
"""Read-only EXP-014 pilot: compare AnomalyCLIP's Q-K and V-V patch evidence.

Uses the same frozen ECP checkpoint, preprocessing, text features, map readout, and
smoothing as test.py. No target-data fitting or model update is performed. The
V-V per-image AUROC must match the saved EXP-012 CSV before outputs are accepted.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys

# Running a file from scripts/ puts that directory, not the repository root, on
# sys.path. The project modules live at the root beside scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["sample_id"]: float(row["per_image_pixel_auroc"]) for row in rows}
    if not rows or len(result) != len(rows):
        raise ValueError(f"empty or duplicate reference sample IDs: {path}")
    return result


def hotspot_scores(mask, vv_map, qk_map, far_pixels=28):
    """Measure branch ranking among baseline hotspots; GT is used only for scoring."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from scipy.stats import rankdata

    hot = vv_map >= float(np.quantile(vv_map, 0.95))
    far_healthy = (~mask) & (distance_transform_edt(~mask) > far_pixels)
    selected = hot & (mask | far_healthy)
    hot_tp = int(np.count_nonzero(selected & mask))
    hot_fp = int(np.count_nonzero(selected & far_healthy))
    if not (hot_tp and hot_fp):
        return hot_tp, hot_fp, "", ""
    labels = mask[selected]
    def auroc(scores):
        n_positive = int(labels.sum())
        n_negative = len(labels) - n_positive
        ranks = rankdata(scores, method="average")
        return float((ranks[labels].sum() - n_positive * (n_positive + 1) / 2)
                     / (n_positive * n_negative))
    return (hot_tp, hot_fp,
            auroc(vv_map[selected]),
            auroc(qk_map[selected]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", required=True, choices=("ISBI", "colon", "thyroid"))
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--sigma", type=float, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--ec_z_pix", type=float, default=1.6)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="first N test images for a smoke run; 0 evaluates the whole split")
    args = parser.parse_args()

    if args.image_size != 518 or args.sigma != 4 or args.seed != 111 or args.ec_z_pix != 1.6:
        raise ValueError("EXP-014 pilot requires image_size=518, sigma=4, seed=111, z_pix=1.6")
    if (args.depth, args.n_ctx, args.t_n_ctx) != (9, 12, 4):
        raise ValueError("EXP-014 pilot requires depth=9, n_ctx=12, t_n_ctx=4")
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    out = Path(args.save_path)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    for path in (args.checkpoint_path, args.reference_csv,
                 os.path.join(args.data_path, "meta.json")):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter
    from sklearn.metrics import roc_auc_score
    import AnomalyCLIP_lib
    from dataset import Dataset
    from extent_prompt import ExtentConditioner, conditioned_text_features, visual_descriptor
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = load_reference(args.reference_csv)
    params = {"Prompt_length": args.n_ctx,
              "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    preprocess, target_transform = get_transform(args)
    dataset = Dataset(root=args.data_path, transform=preprocess,
                      target_transform=target_transform, dataset_name=args.dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if checkpoint.get("extent_cond") != "extent" or "conditioner" not in checkpoint:
        raise ValueError("checkpoint must contain an ECP extent conditioner")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device).eval()
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    conditioner = ExtentConditioner(mode="extent").to(device)
    conditioner.load_state_dict(checkpoint["conditioner"])
    conditioner.eval()

    captured = {}

    def capture_last_block(_module, _inputs, output):
        if not isinstance(output, list) or len(output) != 2:
            raise RuntimeError("last visual block did not return Q-K/V-V dual streams")
        captured["vv"], captured["qk"] = output

    hook = model.visual.transformer.resblocks[-1].register_forward_hook(capture_last_block)

    def score_map(tokens, text_features):
        tokens = F.normalize(tokens, dim=-1)
        similarity, _ = AnomalyCLIP_lib.compute_similarity(tokens, text_features[0])
        similarity_map = AnomalyCLIP_lib.get_similarity_map(
            similarity[:, 1:, :], args.image_size)
        raw = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
        return gaussian_filter(raw[0].detach().cpu().numpy(), sigma=args.sigma)

    rows = []
    max_token_error = 0.0
    max_reference_auroc_error = 0.0
    try:
        for index, item in enumerate(loader):
            if args.limit and index >= args.limit:
                break
            captured.clear()
            image = item["img"].to(device)
            mask = item["img_mask"][0, 0].detach().cpu().numpy() > 0.5
            sample_id = os.path.relpath(item["img_path"][0], args.data_path)
            if sample_id not in reference:
                raise ValueError(f"sample absent from EXP-012 reference: {sample_id}")

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, [24], DPAM_layer=20)
                if "qk" not in captured:
                    raise RuntimeError("visual hook did not capture both streams")
                vv_tokens = model.visual.ln_post(captured["vv"].permute(1, 0, 2)) @ model.visual.proj
                qk_tokens = model.visual.ln_post(captured["qk"].permute(1, 0, 2)) @ model.visual.proj
                token_error = (vv_tokens - patch_features[-1]).abs().max().item()
                max_token_error = max(max_token_error, token_error)
                if token_error > 1e-5:
                    raise RuntimeError(f"V-V hook mismatch on {sample_id}: {token_error}")
                image_features = F.normalize(image_features, dim=-1)
                desc = visual_descriptor(image_features, patch_features)
                _, c_pos, c_neg = conditioner(
                    desc, z_override=torch.full((1,), args.ec_z_pix, device=device))
                text_features = conditioned_text_features(
                    model, prompt_learner, c_pos, c_neg)
                vv_map = score_map(patch_features[-1], text_features)
                qk_map = score_map(qk_tokens, text_features)

            if mask.min() == mask.max():
                raise ValueError(f"EXP-014 pilot requires a binary mask: {sample_id}")
            vv_auroc = float(roc_auc_score(mask.ravel(), vv_map.ravel()))
            qk_auroc = float(roc_auc_score(mask.ravel(), qk_map.ravel()))
            ref_error = abs(vv_auroc - reference[sample_id])
            max_reference_auroc_error = max(max_reference_auroc_error, ref_error)
            if ref_error > 1e-6:
                raise RuntimeError(
                    f"V-V AUROC differs from EXP-012 on {sample_id}: "
                    f"{vv_auroc} vs {reference[sample_id]}")

            # Diagnostic labels only; this does not alter inference or select a method.
            hot_tp, hot_fp, vv_hotspot_auc, qk_hotspot_auc = hotspot_scores(
                mask, vv_map, qk_map)
            rows.append({
                "dataset": os.path.basename(os.path.normpath(args.data_path)),
                "sample_id": sample_id,
                "mask_area_fraction": float(mask.mean()),
                "vv_pixel_auroc": vv_auroc,
                "qk_pixel_auroc_raw_ecp_readout": qk_auroc,
                "vv_hotspot_tp": hot_tp,
                "vv_hotspot_far_fp": hot_fp,
                "vv_hotspot_tp_fp_auroc": vv_hotspot_auc,
                "qk_hotspot_tp_fp_auroc_raw_ecp_readout": qk_hotspot_auc,
            })
            if (index + 1) % 50 == 0:
                print(f"EXP-014 pilot: {index + 1} images", flush=True)
    finally:
        hook.remove()

    if not rows:
        raise ValueError("no images were evaluated")
    if not args.limit and {row["sample_id"] for row in rows} != reference.keys():
        raise ValueError("full run did not cover exactly the EXP-012 reference sample IDs")
    out.mkdir(parents=True)
    with (out / "per_image_branch_probe.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "sample_count": len(rows),
        "max_vv_hook_token_abs_error": max_token_error,
        "max_exp012_vv_auroc_abs_error": max_reference_auroc_error,
        "images_with_both_hotspot_classes": sum(
            row["vv_hotspot_tp"] > 0 and row["vv_hotspot_far_fp"] > 0 for row in rows),
        "mean_per_image_vv_auroc": float(np.mean([row["vv_pixel_auroc"] for row in rows])),
        "mean_per_image_qk_auroc_raw_readout": float(np.mean([
            row["qk_pixel_auroc_raw_ecp_readout"] for row in rows])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metadata = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "checkpoint_sha256": sha256(args.checkpoint_path),
        "clip_weights_sha256": sha256(Path.home() / ".cache/clip/ViT-L-14-336px.pt"),
        "meta_json_sha256": sha256(Path(args.data_path) / "meta.json"),
        "reference_csv_sha256": sha256(args.reference_csv),
        "probe_source_sha256": sha256(__file__),
        "code_revision": revision,
        "arguments": vars(args),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"EXP-014 pilot error: {exc}") from exc
