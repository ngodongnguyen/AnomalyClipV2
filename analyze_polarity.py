"""
Step 1 of the "Anomaly Polarity" study.

For each image, computes two native (patch-grid resolution, no upsampling) maps:
  - abnormal_score: AnomalyCLIP's own raw similarity-to-"abnormal"-prompt map
    (same formula used in test.py, just kept at the native patch grid instead
    of upsampled to image_size).
  - homogeneity: how similar each patch's CLIP feature is to its immediate
    neighbors (mean cosine similarity to up/down/left/right neighbors, computed
    directly from patch_features - no extra network, no training).

Then reports, per dataset:
  - domain_intrinsic_polarity  = corr(GT label, homogeneity)
        > 0  => true anomaly region is MORE homogeneous than background
                (medical/mass-like domains, e.g. polyps)
        < 0  => true anomaly region is LESS homogeneous than background
                (industrial/texture-break domains, e.g. MVTec scratches)
  - model_implicit_polarity    = corr(AnomalyCLIP abnormal_score, homogeneity)
        shows which polarity AnomalyCLIP's learned prompt actually looks for,
        regardless of domain.

Both correlations are reported pooled-over-all-patches and as the mean of
per-image correlations (more robust to cross-image confounds).
"""
import os
import AnomalyCLIP_lib
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform


def local_homogeneity(feat_grid):
    # feat_grid: [H, W, C], L2-normalized along C
    H, W, _ = feat_grid.shape
    sim_sum = torch.zeros(H, W, device=feat_grid.device)
    count = torch.zeros(H, W, device=feat_grid.device)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        y0, y1 = max(dy, 0), H + min(dy, 0)
        x0, x1 = max(dx, 0), W + min(dx, 0)
        sy0, sy1 = max(-dy, 0), H + min(-dy, 0)
        sx0, sx1 = max(-dx, 0), W + min(-dx, 0)
        sim = (feat_grid[y0:y1, x0:x1] * feat_grid[sy0:sy1, sx0:sx1]).sum(-1)
        sim_sum[y0:y1, x0:x1] += sim
        count[y0:y1, x0:x1] += 1
    return sim_sum / count


def analyze(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
                              "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                         dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    filter_names = None
    if args.filter_list is not None:
        with open(args.filter_list) as f:
            filter_names = set(line.strip() for line in f if line.strip())

    all_gt, all_homog, all_score = [], [], []
    per_image_corr_gt, per_image_corr_score = [], []
    per_image_gap, per_image_auroc = [], []

    for items in tqdm(test_dataloader):
        img_path = items['img_path'][0]
        if filter_names is not None and os.path.basename(img_path) not in filter_names:
            continue
        image = items['img'].to(device)
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)

            homog_sum = None
            score_sum = None
            side = None
            for idx, patch_feature in enumerate(patch_features):
                if idx < args.feature_map_layer[0]:
                    continue
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                n_patch = similarity.shape[1] - 1
                side = int(n_patch ** 0.5)
                sim_native = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], (side, side))
                score = (sim_native[..., 1] + 1 - sim_native[..., 0]) / 2.0  # [1, side, side]
                score = score[0]

                feat_grid = patch_feature[0, 1:, :].reshape(side, side, -1)
                homog = local_homogeneity(feat_grid)

                homog_sum = homog if homog_sum is None else homog_sum + homog
                score_sum = score if score_sum is None else score_sum + score

            homog_map = (homog_sum / len(patch_features[args.feature_map_layer[0]:])).cpu().numpy()
            score_map = score_sum.cpu().numpy()

            gt_small = F.adaptive_avg_pool2d(gt_mask.float(), (side, side))
            gt_small = (gt_small[0, 0] > 0.5).numpy().astype(np.float32)

            all_gt.append(gt_small.ravel())
            all_homog.append(homog_map.ravel())
            all_score.append(score_map.ravel())

            if gt_small.min() != gt_small.max():
                r_gt, _ = pearsonr(gt_small.ravel(), homog_map.ravel())
                per_image_corr_gt.append(r_gt)

                gt_flat = gt_small.ravel().astype(bool)
                gap = homog_map.ravel()[gt_flat].mean() - homog_map.ravel()[~gt_flat].mean()
                auroc = roc_auc_score(gt_small.ravel(), score_map.ravel())
                per_image_gap.append(gap)
                per_image_auroc.append(auroc)
            r_score, _ = pearsonr(score_map.ravel(), homog_map.ravel())
            per_image_corr_score.append(r_score)

    all_gt = np.concatenate(all_gt)
    all_homog = np.concatenate(all_homog)
    all_score = np.concatenate(all_score)

    pooled_domain_polarity, _ = pearsonr(all_gt, all_homog)
    pooled_model_polarity, _ = pearsonr(all_score, all_homog)

    gap_auroc_corr, _ = pearsonr(per_image_gap, per_image_auroc)

    print("\n=== Anomaly Polarity report:", args.dataset, "===")
    print(f"n_images = {len(per_image_corr_score)}")
    print(f"domain_intrinsic_polarity  (GT vs homogeneity)      pooled = {pooled_domain_polarity:+.4f}   "
          f"mean-per-image = {np.mean(per_image_corr_gt):+.4f}")
    print(f"model_implicit_polarity    (AnomalyCLIP vs homogeneity) pooled = {pooled_model_polarity:+.4f}   "
          f"mean-per-image = {np.mean(per_image_corr_score):+.4f}")
    print("(> 0: anomaly/high-score region MORE homogeneous than surroundings; "
          "< 0: LESS homogeneous, i.e. texture-break style)")
    print(f"\nmean per-image homogeneity gap (inside GT - outside GT) = {np.mean(per_image_gap):+.4f}")
    print(f"corr(per-image gap, per-image pixel-AUROC) = {gap_auroc_corr:+.4f}")
    print("(expected NEGATIVE if the mismatch hypothesis holds: the more "
          "homogeneous/blob-like the true lesion is vs its background, the "
          "worse AnomalyCLIP's own texture-break-seeking score does on that image)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP polarity analysis", add_help=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--features_list", type=int, nargs="+", default=[24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0])
    parser.add_argument("--filter_list", type=str, default=None,
                         help="optional text file of image basenames (one per line) to restrict analysis to")
    args = parser.parse_args()
    print(args)
    analyze(args)
