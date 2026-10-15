import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
from extent_prompt import (ExtentConditioner, TokenAdapter, area_to_z, visual_descriptor, conditioned_text_features,
                           batched_similarity)
from paa import paa_torch
from zshuffle import derangement_among
from token_prompt import (TokenHeads, default_cfg, ot_weights, token_probs, token_margin, hinge_loss, ortho_loss,
                          patch_labels)


def roi_resample(feat_map, boxes_norm, out_size):
    """
    Differentiable crop-and-resize of an axis-aligned box out of a feature map (a hand-rolled
    RoIAlign via affine_grid + grid_sample, so no extra dependency on torchvision.ops).
    feat_map: [B, C, H, W]. boxes_norm: [B, 4] as (x0, y0, x1, y1) in [0, 1] (per-sample box,
    e.g. the lesion's own bounding box within that sample's crop). Returns [B, C, out_size, out_size].
    """
    x0, y0, x1, y1 = boxes_norm.unbind(dim=1)
    xa, xb = 2 * x0 - 1, 2 * x1 - 1
    ya, yb = 2 * y0 - 1, 2 * y1 - 1
    theta = torch.zeros(feat_map.shape[0], 2, 3, device=feat_map.device, dtype=feat_map.dtype)
    theta[:, 0, 0] = (xb - xa) / 2
    theta[:, 0, 2] = (xa + xb) / 2
    theta[:, 1, 1] = (yb - ya) / 2
    theta[:, 1, 2] = (ya + yb) / 2
    grid = F.affine_grid(theta, [feat_map.shape[0], feat_map.shape[1], out_size, out_size], align_corners=False)
    return F.grid_sample(feat_map, grid, align_corners=False)


class ScalePairDataset(torch.utils.data.Dataset):
    """Wraps a Dataset's anomalous samples as (view_a, view_b, lesion_box_a, lesion_box_b) pairs
    for the scale-consistency loss -- see Dataset.sample_scale_pair for what the two views are."""

    def __init__(self, base_dataset, scale_ranges):
        self.base = base_dataset
        self.scale_ranges = scale_ranges
        self.indices = [i for i, d in enumerate(base_dataset.data_all) if d['anomaly'] == 1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        for _ in range(5):
            out = self.base.sample_scale_pair(i, scale_ranges=self.scale_ranges)
            if out is not None:
                return out
            i = self.indices[random.randrange(len(self.indices))]
        raise RuntimeError("ScalePairDataset: could not sample a valid scale pair after 5 tries")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    if args.exp042_cls_view != "legacy" and args.exp043_ecp_cls_view != "legacy":
        raise ValueError("EXP-042 and EXP-043 CLS-view modes are mutually exclusive")
    if args.exp042_cls_view != "legacy":
        if (args.extent_cond != "none" or args.zoom_aug_p != 0.5 or args.translate_aug_p != 0
                or args.consistency_weight != 0 or args.token_adapter != "none"
                or args.paa_scales or args.ec_static_aux):
            raise ValueError("EXP-042 split/sham requires zoom-only prompts and the registered augmentation/loss settings")
    if args.exp043_ecp_cls_view != "legacy":
        if (args.extent_cond != "extent" or args.zoom_aug_p != 0.5 or args.translate_aug_p != 0
                or args.consistency_weight != 0 or args.token_adapter != "none"
                or args.paa_scales or args.ec_static_aux or args.ec_z_shuffle != "none"):
            raise ValueError("EXP-043 split/sham requires the matched ECP recipe")
    if args.zoom_place != "uniform" and (args.zoom_aug_p <= 0 or args.translate_aug_p != 0):
        raise ValueError("EXP-051 --zoom_place beta needs zoom augmentation (--zoom_aug_p > 0) and no translation augmentation")
    cls_view = args.exp043_ecp_cls_view if args.exp043_ecp_cls_view != "legacy" else args.exp042_cls_view
    if args.tok_q == 0 and (args.tok_hinge != 0 or args.tok_ortho != 0):
        raise ValueError("--tok_hinge/--tok_ortho need --tok_q > 0")
    if args.tok_q > 0 and (args.extent_cond != "extent" or args.paa_scales or args.token_adapter != "none"
                           or args.ec_static_aux or args.consistency_weight != 0):
        raise ValueError("EXP-044 token prompts are defined only on top of plain extent-conditioned prompts")

    logger = get_logger(args.save_path)
    z_shuffle_rng = np.random.RandomState(args.seed + 1)   # only used with --ec_z_shuffle batch

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset, zoom_aug_p=args.zoom_aug_p, translate_aug_p=args.translate_aug_p, return_original=cls_view != "legacy",
                         zoom_place=args.zoom_place, zoom_beta=args.zoom_beta)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    consistency_loader, consistency_iter = None, None
    if args.consistency_weight > 0:
        scale_ranges = ((args.consistency_scale_a_min, args.consistency_scale_a_max),
                         (args.consistency_scale_b_min, args.consistency_scale_b_max))
        consistency_dataset = ScalePairDataset(train_data, scale_ranges)
        consistency_loader = torch.utils.data.DataLoader(consistency_dataset,
                                                           batch_size=args.consistency_batch_size or args.batch_size,
                                                           shuffle=True, drop_last=True)
        consistency_iter = iter(consistency_loader)

    def next_consistency_batch():
        nonlocal consistency_iter
        try:
            return next(consistency_iter)
        except StopIteration:
            consistency_iter = iter(consistency_loader)
            return next(consistency_iter)

  ##########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    ##########################################################################################
    vv_record = None   # EXP-046: default-off V-V head mask (installed before the first encode_image, so training sees the masked V-V stream)
    if args.vv_head_mask:
        import vvattn_core, vvattn_stats
        vv_keep, vv_sha = vvattn_stats.load_keep(args.vv_head_mask)
        vv_record = vvattn_stats.mask_record(args.vv_head_mask, vv_keep, vv_sha)
        vvattn_core.install_head_mask(model, vv_keep, device)
        logger.info("V-V head mask %s sha256 %s keeps %d/%d heads" % (args.vv_head_mask, vv_sha, vv_record["n_keep"], vv_keep.size))
    conditioner = None
    if args.extent_cond != "none":
        assert args.consistency_weight == 0, "extent conditioning is not combined with the consistency loss"
        conditioner = ExtentConditioner(mode=args.extent_cond).to(device)
    static_heads = None
    if args.ec_static_aux:
        assert args.extent_cond == "extent", "static-head auxiliary training requires extent conditioning"
        assert args.token_adapter == "none" and not args.paa_scales, "static-head pilot uses the original patch readout"
        static_heads = ExtentConditioner(mode="dual").to(device)
    adapter = None
    if args.token_adapter != "none":
        assert args.extent_cond == "extent", "the token adapter is only defined on top of the extent-conditioned model"
        adapter = TokenAdapter(rank=args.adapter_rank, z_cond=(args.token_adapter == "z")).to(device)
    tok_heads, tok_cfg = None, None
    if args.tok_q > 0:
        tok_cfg = dict(default_cfg(args.tok_q), eps=args.tok_eps, iters=args.tok_iters, topk=args.tok_topk,
                       thr=args.tok_thr, eta=args.tok_hinge_eta, hinge_scale=args.tok_hinge_scale)
        tok_heads = TokenHeads(args.tok_q).to(device)
    params = list(prompt_learner.parameters()) + (list(conditioner.parameters()) if conditioner is not None else []) \
        + (list(static_heads.parameters()) if static_heads is not None else []) \
        + (list(adapter.parameters()) if adapter is not None else []) \
        + (list(tok_heads.parameters()) if tok_heads is not None else [])
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    lam = 4
    
    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        if conditioner is not None:
            conditioner.train()
        if static_heads is not None:
            static_heads.train()
        if adapter is not None:
            adapter.train()
        if tok_heads is not None:
            tok_heads.train()
        tok_hinge_list, tok_ortho_list = [], []
        loss_list = []
        image_loss_list = []
        consistency_loss_list = []
        extent_err_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label =  items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                # Apply DPAM to the layer from 6 to 24
                # DPAM_layer represents the number of layer refined by DPAM from top to bottom
                # DPAM_layer = 1, no DPAM is used
                # DPAM_layer = 20 as default
                zoomed = None
                if args.exp043_ecp_cls_view != "legacy":
                    # Only successful anomaly crops differ from their original view.
                    # Encode those originals first, then release their dense tokens.
                    zoomed = items['zoom_applied'].bool()
                if zoomed is not None and zoomed.any():
                    full_cls_features, original_patches = model.encode_image(
                        items['img_original'][zoomed].to(device), args.features_list, DPAM_layer=20)
                    del original_patches
                    if args.exp043_ecp_cls_view == "split":
                        full_cls_features = F.normalize(full_cls_features, dim=-1)
                    else:
                        del full_cls_features
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                cls_features = image_features
                if args.exp043_ecp_cls_view == "split" and zoomed.any():
                    cls_features = image_features.clone()
                    cls_features[zoomed.to(device)] = full_cls_features
                if args.exp042_cls_view != "legacy":
                    original_features, _ = model.encode_image(items['img_original'].to(device),
                                                               args.features_list, DPAM_layer=20)
                    if cls_view == "split":
                        cls_features = F.normalize(original_features, dim=-1)
                    
           ####################################
            extent_loss = torch.tensor(0.0, device=device)
            z_used = None   # the z the prompts were conditioned on (also fed to the token adapter)
            if conditioner is None:
                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
                text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
                text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
                text_features = text_features/text_features.norm(dim=-1, keepdim=True)
                # Apply DPAM surgery
                text_probs = cls_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
                text_probs = text_probs[:, 0, ...]/0.07
            elif conditioner.mode == "dual":
                # control (EXP-022): head-specific learned constant shifts; image loss uses the image-head prompts, the
                # pixel losses below use the pixel-head prompts (text_features)
                n_b = image.shape[0]
                text_features_img = conditioned_text_features(model, prompt_learner, *conditioner.dual_shift("img", n_b))
                text_features = conditioned_text_features(model, prompt_learner, *conditioner.dual_shift("pix", n_b))
                text_probs = torch.einsum("bc,btc->bt", image_features, text_features_img) / 0.07
            else:
                lab = label.long().to(device)
                area = gt.reshape(image.shape[0], -1).mean(1)
                z_gt = area_to_z(area)
                if args.ec_z_shuffle == "batch":   # EXP-028 control: z labels of anomalous images are rotated among them (default off)
                    src = torch.from_numpy(derangement_among((lab == 1).cpu().numpy(), z_shuffle_rng)).to(z_gt.device)
                    z_gt = z_gt[src]
                desc = visual_descriptor(image_features, patch_features)
                z_teacher = None
                if args.extent_cond == "extent" and args.ec_teacher_p > 0:
                    use_gt = (torch.rand_like(z_gt) < args.ec_teacher_p) & (lab == 1)
                    z_pred_free = conditioner.estimator(desc).squeeze(-1)
                    z_teacher = torch.where(use_gt, z_gt, z_pred_free.detach())
                z_pred, c_pos, c_neg = conditioner(desc, z_override=z_teacher)
                z_used = z_teacher if z_teacher is not None else z_pred
                if z_pred is not None and (lab == 1).any():
                    extent_loss = F.smooth_l1_loss(z_pred[lab == 1], z_gt[lab == 1])
                    extent_err_list.append((z_pred[lab == 1] - z_gt[lab == 1]).abs().mean().item())
                text_features = conditioned_text_features(model, prompt_learner, c_pos, c_neg)
                text_probs = torch.einsum("bc,btc->bt", cls_features, text_features) / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())
            image_loss_list.append(image_loss.item())
            #########################################################################
            similarity_map_list = []
            tok_loss = None   # EXP-044: hinge + orthogonality (stays None, i.e. no change, when --tok_q 0)
            if tok_heads is not None:
                tok_sub = tok_heads(text_features)   # [B, Q, 2, C] subspace embeddings of the ECP-shifted text features
                tok_loss = args.tok_ortho * ortho_loss(tok_sub) if args.tok_ortho != 0 else None
                tok_ortho_list.append(float(ortho_loss(tok_sub).detach()))
                tok_hinge_total = 0
            # similarity_map_list.append(similarity_map)
            # --paa_scales (default empty = original behaviour): one extra similarity map per aggregation scale, each
            # supervised like the original map (MoECLIP-style Patch Average Aggregation, parameter-free)
            for idx, raw_patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    for paa_s in (args.paa_scales if args.paa_scales else [None]):
                        patch_feature = raw_patch_feature if paa_s is None else paa_torch(raw_patch_feature, paa_s)
                        if adapter is not None:
                            patch_feature = adapter(patch_feature, z_used)
                        patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                        if conditioner is None:
                            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                        else:
                            similarity = batched_similarity(patch_feature, text_features)
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                        similarity_map_list.append(similarity_map)
                        if tok_heads is not None:
                            # EXP-044: OT-routed per-patch map, supervised like the base map (focal + Dice, same weight)
                            tok_patch = patch_feature[:, 1:, :].float()
                            tok_w = ot_weights(tok_patch, tok_sub.detach(), tok_cfg)
                            tok_map = AnomalyCLIP_lib.get_similarity_map(token_probs(tok_patch, tok_sub, tok_w), args.image_size).permute(0, 3, 1, 2)
                            similarity_map_list.append(tok_map)
                            if args.tok_hinge != 0:
                                tok_hinge_total = tok_hinge_total + hinge_loss(
                                    token_margin(tok_patch, tok_sub, tok_w, tok_cfg["hinge_scale"]),
                                    patch_labels(gt, tok_patch.shape[1]), tok_cfg["eta"])

            if tok_heads is not None and args.tok_hinge != 0:
                tok_hinge_list.append(float(tok_hinge_total.detach()))
                tok_loss = args.tok_hinge * tok_hinge_total if tok_loss is None else tok_loss + args.tok_hinge * tok_hinge_total

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            if args.paa_scales:
                loss = loss / len(args.paa_scales)   # keep the pixel-vs-image loss ratio of the single-map baseline
            loss = lam * loss

            static_task_loss = None
            if static_heads is not None:
                n_b = image.shape[0]
                static_img = conditioned_text_features(model, prompt_learner, *static_heads.dual_shift("img", n_b))
                static_pix = conditioned_text_features(model, prompt_learner, *static_heads.dual_shift("pix", n_b))
                static_logits = torch.einsum("bc,btc->bt", image_features, static_img) / 0.07
                static_image_loss = F.cross_entropy(static_logits, label.long().to(device))
                static_pixel_loss = 0
                for idx, raw_patch_feature in enumerate(patch_features):
                    if idx >= args.feature_map_layer[0]:
                        patch_feature = F.normalize(raw_patch_feature, dim=-1)
                        similarity = batched_similarity(patch_feature, static_pix)
                        static_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                        static_pixel_loss += loss_focal(static_map, gt)
                        static_pixel_loss += loss_dice(static_map[:, 1, :, :], gt)
                        static_pixel_loss += loss_dice(static_map[:, 0, :, :], 1-gt)
                static_task_loss = lam * static_pixel_loss + static_image_loss

            consistency_loss = torch.tensor(0.0, device=device)
            if consistency_loader is not None:
                img_a, img_b, box_a, box_b = next_consistency_batch()
                img_a, img_b = img_a.to(device), img_b.to(device)
                box_a, box_b = box_a.to(device), box_b.to(device)
                with torch.no_grad():
                    _, patch_features_a = model.encode_image(img_a, args.features_list, DPAM_layer=20)
                    _, patch_features_b = model.encode_image(img_b, args.features_list, DPAM_layer=20)
                canon = []
                for patch_feature, box in ((patch_features_a[-1], box_a), (patch_features_b[-1], box_b)):
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    sim_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    abnormal_map = ((sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0).unsqueeze(1)
                    canon.append(roi_resample(abnormal_map, box, args.consistency_roi_size))
                consistency_loss = F.mse_loss(canon[0], canon[1])
                consistency_loss_list.append(consistency_loss.item())

            optimizer.zero_grad()
            task_loss = loss + image_loss
            if static_task_loss is not None:
                task_loss = 0.5 * (task_loss + static_task_loss)
            total_loss = task_loss + args.consistency_weight * consistency_loss + args.ec_weight * extent_loss
            if tok_loss is not None:
                total_loss = total_loss + tok_loss
            total_loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}, consistency_loss:{:.4f}, extent_abs_err(z):{:.4f}'.format(
                epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list),
                np.mean(consistency_loss_list) if consistency_loss_list else 0.0,
                np.mean(extent_err_list) if extent_err_list else 0.0))
            if tok_heads is not None:
                logger.info('epoch [{}/{}], tok_hinge:{:.4f}, tok_ortho:{:.4f}'.format(
                    epoch + 1, args.epoch, np.mean(tok_hinge_list) if tok_hinge_list else 0.0,
                    np.mean(tok_ortho_list) if tok_ortho_list else 0.0))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            ckpt = {"prompt_learner": prompt_learner.state_dict()}
            if args.exp042_cls_view != "legacy":
                ckpt["exp042_cls_view"] = args.exp042_cls_view
            if args.exp043_ecp_cls_view != "legacy":
                ckpt["exp043_ecp_cls_view"] = args.exp043_ecp_cls_view
            if args.paa_scales:
                ckpt["paa_scales"] = list(args.paa_scales)
            if adapter is not None:
                ckpt["token_adapter"] = adapter.state_dict()
                ckpt["token_adapter_cfg"] = {"rank": args.adapter_rank, "z_cond": args.token_adapter == "z"}
            if conditioner is not None:
                ckpt["conditioner"] = conditioner.state_dict()
                ckpt["extent_cond"] = args.extent_cond
            if tok_heads is not None:
                ckpt["tok_heads"] = tok_heads.state_dict()
                ckpt["tok_cfg"] = dict(tok_cfg)
            if vv_record is not None:
                ckpt["vv_head_mask"] = vv_record
            if static_heads is not None:
                ckpt["static_heads"] = static_heads.state_dict()
                ckpt["ec_static_aux"] = True
            torch.save(ckpt, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--token_adapter", choices=["none", "const", "z"], default="none",
                        help="post-encoder bottleneck adapter on the patch tokens: z = conditioned on the extent z, const = capacity-matched control ignoring z, none = original behaviour")
    parser.add_argument("--adapter_rank", type=int, default=32)
    parser.add_argument("--paa_scales", type=int, nargs="*", default=[], help="odd window sizes of Patch Average Aggregation, e.g. 1 3 5 (empty = off, original behaviour)")
    parser.add_argument("--translate_aug_p", type=float, default=0.0, help="EXP-037: prob. of a label-preserving translation (CLIP-mean fill) of a training image and its mask (0 = original behaviour)")
    parser.add_argument("--zoom_aug_p", type=float, default=0.0, help="prob. of zooming a training image around its anomaly (0 = original behaviour)")
    parser.add_argument("--zoom_place", choices=["uniform", "beta"], default="uniform",
                        help="EXP-051: anomaly placement inside a zoom crop; uniform = original, beta = crop corner ~ Beta(zoom_beta, zoom_beta) over the admissible range")
    parser.add_argument("--zoom_beta", type=float, default=1.0, help="EXP-051: Beta parameter of --zoom_place beta (< 1 pushes the anomaly toward the crop edges)")
    parser.add_argument("--exp042_cls_view", choices=["legacy", "split", "sham"], default="legacy",
                        help="EXP-042: split uses full-frame CLS and zoomed pixels; sham computes both views but uses zoomed CLS")
    parser.add_argument("--exp043_ecp_cls_view", choices=["legacy", "split", "sham"], default="legacy",
                        help="EXP-043 ECP: split uses full-frame CLS; extent descriptor, z and pixel losses retain zoomed view")
    parser.add_argument("--consistency_weight", type=float, default=0.0,
                         help="weight of the scale-consistency loss (0 = off, original behaviour)")
    parser.add_argument("--consistency_batch_size", type=int, default=0,
                         help="batch size for the consistency pair loader (0 = same as --batch_size); "
                              "lower this if you hit CUDA OOM once consistency_weight > 0")
    parser.add_argument("--consistency_roi_size", type=int, default=32,
                         help="side (px) of the canonical grid the lesion's own bbox is resampled to for the consistency loss")
    parser.add_argument("--consistency_scale_a_min", type=float, default=0.05)
    parser.add_argument("--consistency_scale_a_max", type=float, default=0.15)
    parser.add_argument("--consistency_scale_b_min", type=float, default=0.25)
    parser.add_argument("--consistency_scale_b_max", type=float, default=0.45)
    parser.add_argument("--extent_cond", type=str, default="none", choices=["none", "extent", "global", "dual"],
                         help="none = original prompts; extent = extent-conditioned prompts; global = image-conditioned control")
    parser.add_argument("--ec_weight", type=float, default=1.0, help="weight of the extent-estimation loss")
    parser.add_argument("--ec_z_shuffle", type=str, default="none", choices=["none", "batch"],
                         help="EXP-028 control: rotate the extent labels (teacher z and extent-loss target) among the anomalous images of a batch")
    parser.add_argument("--ec_teacher_p", type=float, default=0.5,
                         help="prob. of conditioning on the ground-truth extent (anomalous images) instead of the estimate")
    parser.add_argument("--ec_static_aux", action="store_true",
                        help="train source-only static image/pixel prompts alongside ECP; use static prompts at test")
    parser.add_argument("--tok_q", type=int, default=0,
                        help="EXP-044: number of TokenCLIP-style prompt subspaces (0 = off, original behaviour; 3 = TOK, 1 = TOKQ1 control)")
    parser.add_argument("--tok_hinge", type=float, default=0.0, help="EXP-044: weight of the token hinge loss (0 = off)")
    parser.add_argument("--tok_ortho", type=float, default=0.0, help="EXP-044: weight of the subspace Gram orthogonality penalty (0 = off)")
    parser.add_argument("--tok_eps", type=float, default=0.01, help="EXP-044: Sinkhorn entropic regularisation")
    parser.add_argument("--tok_iters", type=int, default=100, help="EXP-044: Sinkhorn iterations")
    parser.add_argument("--tok_topk", type=int, default=2, help="EXP-044: routing sparsification, entries kept per patch")
    parser.add_argument("--tok_thr", type=float, default=0.2, help="EXP-044: routing sparsification threshold")
    parser.add_argument("--tok_hinge_eta", type=float, default=5.0, help="EXP-044: hinge margin")
    parser.add_argument("--tok_hinge_scale", type=float, default=100.0, help="EXP-044: scale applied to the cosine margin inside the hinge")
    parser.add_argument("--vv_head_mask", type=str, default="",
                        help="EXP-046: .npy bool/0-1 mask [19 layers 6..24, 16 heads] of V-V heads kept (empty = off, original behaviour); stored in the checkpoint")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
