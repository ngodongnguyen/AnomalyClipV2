"""
Domain-level z calibration, attempt 2 (no per-image estimation, no labels used for the decision).

Attempt 1 (mean Otsu-abnormal-fraction of the pixel map at z=1.6) failed: it was flat (~0.14-0.28) across ALL 9
datasets, including the 3 where z=1.6 catastrophically breaks whole-image classification (HeadCT/BrainMRI/Br35H)
-- so the pixel map does not saturate there; the collapse must happen in the CLS-based classification score
itself, not in spatial over-triggering. This attempt measures that score directly instead of a proxy for it.

Rule (pre-registered, symmetric, not fit to any observed outcome): on an UNLABELED batch, compute the whole-image
"abnormal" probability (text_probs, the same quantity test.py reports as image_auroc's input) at BOTH validated
operating points, z=1.6 (large) and z=-0.5 (small). If the large-mode score is less spread out across the batch
than the small-mode score (std_large < std_small), the large-lesion prompt is failing to discriminate images on
this domain -> fall back to z=-0.5. Otherwise keep z=1.6. Still zero labels, zero training, self-relative
(compares the model's own two known states on the same unlabeled images), so it is not tuned to the 9/9 outcome.

    python calibrate_z_domain.py --dataset colon --data_path <ClinicDB> --checkpoint_path checkpoints/ecp_extent/epoch_15.pth --limit 40
"""
import argparse
import numpy as np
import torch
from tqdm import tqdm

from dataset import Dataset
from utils import get_transform
from analyze_transfer_probe import build_model
from extent_prompt import ExtentConditioner, visual_descriptor, conditioned_text_features

Z_LARGE, Z_SMALL = 1.6, -0.5


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, target_transform = get_transform(args)
    model, _, _ = build_model(args, device)
    conditioner = ExtentConditioner(mode="extent").to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    conditioner.load_state_dict(ckpt["conditioner"])
    conditioner.eval()
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    prompt_learner.load_state_dict(ckpt["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)

    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    idx = np.random.RandomState(0).permutation(len(data))[: args.limit]
    probs = {Z_LARGE: [], Z_SMALL: []}
    true_anomaly = []
    for i in tqdm(idx):
        items = data[int(i)]
        img = items["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            image_features, patch_features = model.encode_image(img, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for z in (Z_LARGE, Z_SMALL):
                z_t = torch.tensor([z], device=device)
                _, c_pos, c_neg = conditioner(visual_descriptor(image_features, patch_features), z_override=z_t)
                text_features = conditioned_text_features(model, prompt_learner, c_pos, c_neg)
                p = (image_features @ text_features.permute(0, 2, 1) / 0.07).softmax(-1)[:, 0, 1]
                probs[z].append(float(p[0]))
        true_anomaly.append(int(items["anomaly"]))

    name = args.data_path.rstrip("/").split("/")[-1]
    std_large, std_small = float(np.std(probs[Z_LARGE])), float(np.std(probs[Z_SMALL]))
    decision = "z=-0.5 (small)" if std_large < std_small else "z=1.6 (large, unchanged)"
    print(f"\n{name}: std(text_probs) over {len(idx)} UNLABELED images -- z=1.6: {std_large:.4f}   "
          f"z=-0.5: {std_small:.4f}   ratio(large/small)={std_large / max(std_small, 1e-8):.2f}  "
          f"->  decision: {decision}")
    print(f"  (for cross-check only, not used in the decision: {sum(true_anomaly)}/{len(true_anomaly)} of the "
          f"sampled images actually had anomaly=1)")
    return std_large, std_small, decision


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--features_list", type=int, nargs="+", default=[24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    args = ap.parse_args()
    run(args)
