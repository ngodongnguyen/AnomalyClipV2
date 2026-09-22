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

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset, zoom_aug_p=args.zoom_aug_p)
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
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    lam = 4
    
    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []
        consistency_loss_list = []

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
                    
           ####################################
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            # Apply DPAM surgery
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...]/0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())
            image_loss_list.append(image_loss.item())
            #########################################################################
            similarity_map_list = []
            # similarity_map_list.append(similarity_map)
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            loss = lam * loss

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
            (loss + image_loss + args.consistency_weight * consistency_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}, consistency_loss:{:.4f}'.format(
                epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list),
                np.mean(consistency_loss_list) if consistency_loss_list else 0.0))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

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
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
