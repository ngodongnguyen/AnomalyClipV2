"""
Flip/rotation robustness (no training, no labels used by the method -- labels only used to SCORE the result).

Question, genuinely untested so far (all prior diagnostics looked at dataset-level image STATISTICS, not model
STABILITY): does the model's own prediction change a lot when the exact same image is flipped or rotated before
being fed in? Endoscopy and dermoscopy frames have no canonical "up" or "left/right" -- flipping or rotating by
90 degrees is label-preserving (the GT mask is flipped/rotated identically, no new information is added or
removed) -- so any instability across these views is a genuine representation weakness, not a meaningless probe.

For each image: run the model on 6 views (identity, h-flip, v-flip, 180-rot, 90-rot, 270-rot), invert each
view's transform on the resulting anomaly map so all 6 maps are aligned back to the ORIGINAL pixel grid, then
report (a) the spread of per-view AUROC for the same image -- instability -- and (b) whether simply AVERAGING
the 6 aligned maps (test-time augmentation, zero extra parameters, zero training) beats the plain single-pass
(identity) AUROC. If it does by a real margin, that is an immediately actionable, well-motivated module.

    python analyze_tta_robustness.py --dataset colon --data_path <ClinicDB> --checkpoint_path checkpoints/ecp_extent/epoch_15.pth --ec_z_pix 1.6
"""
import os
import argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform
from layer_probe_stats import auroc
from extent_prompt import ExtentConditioner, visual_descriptor, conditioned_text_features

# (forward transform on the input image tensor [1,3,H,W], inverse transform on the resulting 2-D map)
VIEWS = {
    "identity": (lambda x: x, lambda m: m),
    "hflip": (lambda x: torch.flip(x, dims=[-1]), lambda m: np.flip(m, axis=-1)),
    "vflip": (lambda x: torch.flip(x, dims=[-2]), lambda m: np.flip(m, axis=-2)),
    "rot180": (lambda x: torch.flip(x, dims=[-2, -1]), lambda m: np.flip(m, axis=(-2, -1))),
    "rot90": (lambda x: torch.rot90(x, k=1, dims=[-2, -1]), lambda m: np.rot90(m, k=-1)),
    "rot270": (lambda x: torch.rot90(x, k=3, dims=[-2, -1]), lambda m: np.rot90(m, k=-3)),
}


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    idx = np.random.RandomState(0).permutation(len(data))[: args.limit] if args.limit else range(len(data))
    loader = [data[int(i)] for i in idx]

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

    auc_identity, auc_ensemble, auc_std, map_disagree = [], [], [], []
    for items in tqdm(loader):
        gt = items["img_mask"][0].numpy() > 0.5
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        img = items["img"].unsqueeze(0).to(device)
        aligned_maps, aucs = {}, {}
        for name, (fwd, inv) in VIEWS.items():
            with torch.no_grad():
                x = fwd(img)
                image_features, patch_features = model.encode_image(x, args.features_list, DPAM_layer=20)
                tf = text_features
                if conditioner is not None:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    z = torch.full((1,), args.ec_z_pix, device=device)
                    _, c_pos, c_neg = conditioner(visual_descriptor(image_features, patch_features), z_override=z)
                    tf = conditioned_text_features(model, prompt_learner, c_pos, c_neg)
                total = 0
                for pf in patch_features:
                    pf = pf / pf.norm(dim=-1, keepdim=True)
                    sim, _ = AnomalyCLIP_lib.compute_similarity(pf, tf[0])
                    sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], args.image_size)
                    total = total + (sm[..., 1] + 1 - sm[..., 0]) / 2.0
                m = gaussian_filter(total[0].cpu().numpy(), sigma=args.sigma)
            m_aligned = inv(m)  # back to the ORIGINAL image's coordinate frame
            aligned_maps[name] = m_aligned
            aucs[name] = auroc(gt.ravel(), m_aligned.ravel())

        ensemble = np.mean(list(aligned_maps.values()), axis=0)
        auc_identity.append(aucs["identity"])
        auc_ensemble.append(auroc(gt.ravel(), ensemble.ravel()))
        auc_std.append(float(np.std(list(aucs.values()))))
        # disagreement: mean absolute difference between each view and identity, each min-max normalised first
        norm = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-8)
        base = norm(aligned_maps["identity"])
        map_disagree.append(float(np.mean([np.abs(norm(aligned_maps[v]) - base).mean()
                                            for v in VIEWS if v != "identity"])))

    name = os.path.basename(args.data_path.rstrip("/"))
    auc_identity, auc_ensemble, auc_std, map_disagree = map(np.array, (auc_identity, auc_ensemble, auc_std, map_disagree))
    print(f"\n=== Flip/rotation robustness: {args.dataset} {name} (n={len(auc_identity)}) ===")
    print(f"mean per-view AUROC std within an image (instability, 0=perfectly consistent): {auc_std.mean():.4f}")
    print(f"mean normalised map disagreement vs identity view: {map_disagree.mean():.4f}")
    print(f"AUROC   single-pass(identity)={auc_identity.mean():.4f}   6-view-ensemble={auc_ensemble.mean():.4f}"
          f"   delta={auc_ensemble.mean() - auc_identity.mean():+.4f}")
    print(f"images where ensemble beats identity: {100 * (auc_ensemble > auc_identity).mean():.1f}%   "
          f"(worse: {100 * (auc_ensemble < auc_identity).mean():.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = use all images")
    ap.add_argument("--ec_z_pix", type=float, default=None, help="ECP checkpoint: fixed z for the pixel head")
    ap.add_argument("--features_list", type=int, nargs="+", default=[24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    ap.add_argument("--sigma", type=int, default=4)
    args = ap.parse_args()
    run(args)
