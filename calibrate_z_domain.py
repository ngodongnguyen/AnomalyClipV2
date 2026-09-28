"""
Domain-level z calibration (no per-image estimation, no labels used for the decision).

Question: from a handful of UNLABELED images of a new deployment domain, can we tell -- without ever looking at
ground truth -- whether this domain needs the "large lesion" prompt state (z=1.6, validated on endoscopy /
dermatology) or the "small/moderate lesion" state (z=-0.5, validated on brain CT/MRI classification)?

Rule (pre-registered, not fit to the data): run the model at z=1.6 on the unlabeled batch, threshold each
anomaly map with Otsu, and average the "abnormal" fraction over the batch. If that average exceeds 0.5 (z=1.6
is calling more than half of a typical image abnormal), the large-lesion state is over-triggering on this
domain -> fall back to z=-0.5. Otherwise keep z=1.6. No learned component, no training, no labels.

    python calibrate_z_domain.py --dataset colon --data_path <ClinicDB> --checkpoint_path checkpoints/ecp_extent/epoch_15.pth --limit 40
"""
import argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import AnomalyCLIP_lib
from dataset import Dataset
from utils import get_transform
from analyze_transfer_probe import build_model
from extent_prompt import ExtentConditioner, visual_descriptor, conditioned_text_features
from analyze_extent_proxies import otsu_frac

Z_LARGE, Z_SMALL, THRESHOLD = 1.6, -0.5, 0.5


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, target_transform = get_transform(args)
    model, _, _ = build_model(args, device)
    conditioner = ExtentConditioner(mode="extent").to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    conditioner.load_state_dict(ckpt["conditioner"])
    conditioner.eval()
    prompt_learner_state = ckpt["prompt_learner"]
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    prompt_learner.load_state_dict(prompt_learner_state)
    prompt_learner.to(device)
    model.to(device)

    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    idx = np.random.RandomState(0).permutation(len(data))[: args.limit]
    fracs, true_anomaly = [], []
    for i in tqdm(idx):
        items = data[int(i)]
        img = items["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            image_features, patch_features = model.encode_image(img, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            z = torch.tensor([Z_LARGE], device=device)
            _, c_pos, c_neg = conditioner(visual_descriptor(image_features, patch_features), z_override=z)
            text_features = conditioned_text_features(model, prompt_learner, c_pos, c_neg)
            total = 0
            for pf in patch_features:
                pf = pf / pf.norm(dim=-1, keepdim=True)
                sim, _ = AnomalyCLIP_lib.compute_similarity(pf, text_features[0])
                sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], args.image_size)
                total = total + (sm[..., 1] + 1 - sm[..., 0]) / 2.0
            m = gaussian_filter(total[0].cpu().numpy(), sigma=4)
        fracs.append(otsu_frac(m))
        true_anomaly.append(int(items["anomaly"]))

    name = args.data_path.rstrip("/").split("/")[-1]
    mean_frac = float(np.mean(fracs))
    decision = "z=-0.5 (small/moderate)" if mean_frac > THRESHOLD else "z=1.6 (large, unchanged)"
    print(f"\n{name}: mean Otsu-abnormal-fraction at z=1.6 over {len(fracs)} UNLABELED images = {mean_frac:.3f}"
          f"  ->  decision: {decision}")
    print(f"  (for cross-check only, not used in the decision: {sum(true_anomaly)}/{len(true_anomaly)} of the "
          f"sampled images actually had anomaly=1)")
    return mean_frac, decision


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
