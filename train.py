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
from extent_prompt import (ExtentConditioner, area_to_z, visual_descriptor, conditioned_text_features,
                           batched_similarity)
from zshuffle import derangement_among


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    logger = get_logger(args.save_path)
    z_shuffle_rng = np.random.RandomState(args.seed + 1)   # only used with --ec_z_shuffle batch

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset, zoom_aug_p=args.zoom_aug_p)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

  ##########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    ##########################################################################################
    conditioner = None
    if args.extent_cond != "none":
        conditioner = ExtentConditioner(mode=args.extent_cond).to(device)
    params = list(prompt_learner.parameters()) + (list(conditioner.parameters()) if conditioner is not None else [])
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
        loss_list = []
        image_loss_list = []
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
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                cls_features = image_features

           ####################################
            extent_loss = torch.tensor(0.0, device=device)
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
                if z_pred is not None and (lab == 1).any():
                    extent_loss = F.smooth_l1_loss(z_pred[lab == 1], z_gt[lab == 1])
                    extent_err_list.append((z_pred[lab == 1] - z_gt[lab == 1]).abs().mean().item())
                text_features = conditioned_text_features(model, prompt_learner, c_pos, c_neg)
                text_probs = torch.einsum("bc,btc->bt", cls_features, text_features) / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())
            image_loss_list.append(image_loss.item())
            #########################################################################
            similarity_map_list = []
            for idx, raw_patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = raw_patch_feature / raw_patch_feature.norm(dim = -1, keepdim = True)
                    if conditioner is None:
                        similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    else:
                        similarity = batched_similarity(patch_feature, text_features)
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            loss = lam * loss

            optimizer.zero_grad()
            task_loss = loss + image_loss
            total_loss = task_loss + args.ec_weight * extent_loss
            total_loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}, extent_abs_err(z):{:.4f}'.format(
                epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list),
                np.mean(extent_err_list) if extent_err_list else 0.0))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            ckpt = {"prompt_learner": prompt_learner.state_dict()}
            if conditioner is not None:
                ckpt["conditioner"] = conditioner.state_dict()
                ckpt["extent_cond"] = args.extent_cond
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
    parser.add_argument("--zoom_aug_p", type=float, default=0.0, help="prob. of zooming a training image around its anomaly (0 = original behaviour)")
    parser.add_argument("--extent_cond", type=str, default="none", choices=["none", "extent", "global", "dual"],
                         help="none = original prompts; extent = extent-conditioned prompts; global = image-conditioned control")
    parser.add_argument("--ec_weight", type=float, default=1.0, help="weight of the extent-estimation loss")
    parser.add_argument("--ec_z_shuffle", type=str, default="none", choices=["none", "batch"],
                         help="EXP-028 control: rotate the extent labels (teacher z and extent-loss target) among the anomalous images of a batch")
    parser.add_argument("--ec_teacher_p", type=float, default=0.5,
                         help="prob. of conditioning on the ground-truth extent (anomalous images) instead of the estimate")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
