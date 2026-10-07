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

import os
import csv
import hashlib
import json
import shlex
import shutil
import random
import subprocess
import sys
import numpy as np
from tabulate import tabulate
from utils import get_transform
from extent_prompt import ExtentConditioner, TokenAdapter, area_to_z, visual_descriptor, conditioned_text_features
from sklearn.metrics import roc_auc_score
from prompt_ensemble import tokenize
from distractor_stats import LESION, DISTRACT, PRUNED, rerank_pool
from paa import paa_torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _git_revision():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def _write_image_level_artifacts(args, rows, metric_values, checkpoint):
    """Export existing scores and metric values without recomputing either."""
    csv_path = os.path.join(args.save_path, 'image_level_predictions.csv')
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'dataset', 'sample_id', 'ground_truth_image_label', 'image_anomaly_score'])
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(args.save_path, 'image_level_metrics.json'), 'w') as handle:
        json.dump(metric_values, handle, indent=2, allow_nan=False)
        handle.write('\n')

    labels = [row['ground_truth_image_label'] for row in rows]
    meta_path = os.path.join(args.data_path, 'meta.json')
    code_root = os.path.dirname(os.path.abspath(__file__))
    source_paths = [
        'test.py', 'metrics.py', 'dataset.py', 'utils.py', 'extent_prompt.py',
        'prompt_ensemble.py', 'logger.py', 'visualization.py', 'distractor_stats.py',
    ]
    source_paths.extend(sorted(
        os.path.join('AnomalyCLIP_lib', name)
        for name in os.listdir(os.path.join(code_root, 'AnomalyCLIP_lib'))
        if name.endswith('.py')))
    source_hashes = {
        path: _sha256_file(os.path.join(code_root, path)) for path in source_paths
    }
    vocab_path = os.path.join(code_root, 'AnomalyCLIP_lib', 'bpe_simple_vocab_16e6.txt.gz')
    source_hashes[os.path.relpath(vocab_path, code_root)] = _sha256_file(vocab_path)
    # model_load.load uses this cache path for the named weights when download_root is unset.
    clip_weights_path = os.path.expanduser('~/.cache/clip/ViT-L-14-336px.pt')
    metadata = {
        'command': shlex.join([sys.executable, *sys.argv]),
        'checkpoint_path': os.path.abspath(args.checkpoint_path),
        'checkpoint_sha256': _sha256_file(args.checkpoint_path),
        'checkpoint_extent_cond': checkpoint.get('extent_cond'),
        'code_revision': _git_revision(),
        'evaluation_source_sha256': source_hashes,
        'pretrained_clip_model': 'ViT-L/14@336px',
        'pretrained_clip_weights_path': clip_weights_path if os.path.isfile(clip_weights_path) else None,
        'pretrained_clip_weights_sha256': (
            _sha256_file(clip_weights_path) if os.path.isfile(clip_weights_path) else None),
        'data_path': os.path.abspath(args.data_path),
        'dataset': os.path.basename(os.path.normpath(args.data_path)),
        'dataset_mode': args.dataset,
        'split': 'test',
        'meta_json_sha256': _sha256_file(meta_path),
        'z_img': args.ec_z_img,
        'z_pix': args.ec_z_pix,
        'sigma': args.sigma,
        'seed': args.seed,
        'sample_count': len(rows),
        'positive_count': sum(label == 1 for label in labels),
        'negative_count': sum(label == 0 for label in labels),
        'arguments': vars(args),
    }
    with open(os.path.join(args.save_path, 'evaluation_metadata.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write('\n')

def _write_pixel_level_artifacts(args, rows, metric_values, checkpoint, results):
    """Export the already-computed pixel metrics and per-image mask statistics."""
    with open(os.path.join(args.save_path, 'pixel_level_metrics.json'), 'w') as handle:
        json.dump(metric_values, handle, indent=2, allow_nan=False)
        handle.write('\n')

    dataset_root = os.path.abspath(args.data_path)
    with open(os.path.join(args.save_path, 'pixel_per_image_predictions.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'dataset', 'sample_id', 'mask_area_fraction', 'per_image_pixel_auroc'])
        writer.writeheader()
        for image_path, area_frac, auroc, _z_pred, _z_gt in rows:
            writer.writerow({
                'dataset': os.path.basename(os.path.normpath(args.data_path)),
                'sample_id': os.path.relpath(image_path, dataset_root),
                'mask_area_fraction': area_frac,
                'per_image_pixel_auroc': auroc,
            })

    labels = []
    sample_count = 0
    pixel_counts = {}
    for obj, result in results.items():
        obj_labels = [int(x) for x in result['gt_sp']]
        labels.extend(obj_labels)
        sample_count += len(obj_labels)
        masks = np.asarray(result['imgs_masks'])
        pixel_counts[obj] = {
            'foreground_pixels': int(np.count_nonzero(masks > 0.5)),
            'background_pixels': int(np.count_nonzero(masks <= 0.5)),
        }

    code_root = os.path.dirname(os.path.abspath(__file__))
    source_paths = [
        'test.py', 'metrics.py', 'dataset.py', 'utils.py', 'extent_prompt.py',
        'prompt_ensemble.py', 'logger.py', 'visualization.py', 'distractor_stats.py',
    ]
    source_paths.extend(sorted(
        os.path.join('AnomalyCLIP_lib', name)
        for name in os.listdir(os.path.join(code_root, 'AnomalyCLIP_lib'))
        if name.endswith('.py')))
    source_hashes = {path: _sha256_file(os.path.join(code_root, path)) for path in source_paths}
    vocab_path = os.path.join(code_root, 'AnomalyCLIP_lib', 'bpe_simple_vocab_16e6.txt.gz')
    source_hashes[os.path.relpath(vocab_path, code_root)] = _sha256_file(vocab_path)
    clip_weights_path = os.path.expanduser('~/.cache/clip/ViT-L-14-336px.pt')
    metadata = {
        'command': shlex.join([sys.executable, *sys.argv]),
        'checkpoint_path': os.path.abspath(args.checkpoint_path),
        'checkpoint_sha256': _sha256_file(args.checkpoint_path),
        'checkpoint_extent_cond': checkpoint.get('extent_cond'),
        'code_revision': _git_revision(),
        'evaluation_source_sha256': source_hashes,
        'pretrained_clip_model': 'ViT-L/14@336px',
        'pretrained_clip_weights_path': clip_weights_path if os.path.isfile(clip_weights_path) else None,
        'pretrained_clip_weights_sha256': (
            _sha256_file(clip_weights_path) if os.path.isfile(clip_weights_path) else None),
        'data_path': dataset_root,
        'dataset': os.path.basename(os.path.normpath(args.data_path)),
        'dataset_mode': args.dataset,
        'split': 'test',
        'meta_json_sha256': _sha256_file(os.path.join(args.data_path, 'meta.json')),
        'z_img': args.ec_z_img,
        'z_pix': args.ec_z_pix,
        'sigma': args.sigma,
        'seed': args.seed,
        'sample_count': sample_count,
        'positive_count': sum(label == 1 for label in labels),
        'negative_count': sum(label == 0 for label in labels),
        'images_with_binary_masks': len(rows),
        'images_excluded_from_per_image_auc': sample_count - len(rows),
        'mask_pixel_counts': pixel_counts,
        'arguments': vars(args),
    }
    with open(os.path.join(args.save_path, 'evaluation_metadata.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write('\n')

from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list


    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    conditioner = None
    if checkpoint.get("extent_cond"):
        conditioner = ExtentConditioner(mode=checkpoint["extent_cond"]).to(device)
        conditioner.load_state_dict(checkpoint["conditioner"])
        conditioner.eval()

    adapter = None
    if checkpoint.get("token_adapter") is not None:
        adapter = TokenAdapter(**checkpoint["token_adapter_cfg"]).to(device)
        adapter.load_state_dict(checkpoint["token_adapter"])
        adapter.eval()

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features_base = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features_base = torch.stack(torch.chunk(text_features_base, dim = 0, chunks = 2), dim = 1)
    text_features_base = text_features_base/text_features_base.norm(dim=-1, keepdim=True)


    # Pixel-head distractor suppression (no training): s' = s * (1 - d), d = softmax mass (T=100) on distractor
    # concepts among LESION+DISTRACT raw-CLIP text prompts, evaluated on each patch feature. Image head untouched.
    dist_T = dist_mask = None
    assert args.distractor_suppress == 'none' or args.distractor_rerank == 'none', 'use one distractor mode at a time'
    assert not (args.paa_scales and (args.distractor_suppress != 'none' or args.distractor_rerank != 'none')), 'PAA is not combined with the distractor branches'
    if not args.paa_scales and checkpoint.get("paa_scales"):
        logger.info("WARNING: checkpoint was trained with --paa_scales %s but test runs without PAA (pass --paa_scales to match)" % checkpoint["paa_scales"])
    dist_mode = args.distractor_suppress if args.distractor_suppress != 'none' else args.distractor_rerank
    if dist_mode != 'none':
        names = LESION + DISTRACT
        dist_names = DISTRACT if dist_mode == 'all9' else PRUNED
        dist_mask = torch.tensor([n in dist_names for n in names], device=device)
        with torch.no_grad():
            tok = tokenize([f"a photo of {n}" for n in names]).to(device)
            # the text transformer takes [x, deep_prompts, counter]; empty deep list = plain frozen CLIP text encoder
            dist_T = model.encode_text_learn(model.token_embedding(tok).type(model.dtype), tok, []).float()
            dist_T = dist_T / dist_T.norm(dim=-1, keepdim=True)

    bad_case_records = []
    per_image_rows = []
    image_level_rows = []
    image_level_metric_values = {}
    pixel_level_metric_values = {}
    # image-level metrics only need one score per image; keeping full-res maps/masks for every image
    # exhausts RAM on large sets (21k COVID images -> ~45 GB, process OOM-killed)
    need_pixel = args.metrics != 'image-level'

    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        if need_pixel:
            results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Per-head operating point on the learned extent axis: the image-level (CLS) head and the pixel
            # (patch) head can each use their own fixed z (--ec_z_img / --ec_z_pix). --ec_const_z applies one z
            # to both heads; with neither, both heads use the extent estimator.
            text_features_img = text_features_pix = text_features_base
            z_pred_val = z_gt_val = float('nan')
            z_adapt = None   # z fed to the optional token adapter (pixel head only)
            if conditioner is not None and conditioner.mode == "dual":
                # control (EXP-022): constant head-specific shifts, the z flags are ignored
                text_features_img = conditioned_text_features(model, prompt_learner, *conditioner.dual_shift("img", image_features.shape[0]))
                text_features_pix = conditioned_text_features(model, prompt_learner, *conditioner.dual_shift("pix", image_features.shape[0]))
            elif conditioner is not None:
                desc = visual_descriptor(image_features, patch_features)
                z_gt = area_to_z(gt_mask.reshape(1, -1).mean(1).to(device))
                z_gt_val = float(z_gt[0])

                def text_at(z_value):
                    z_t = None if z_value is None else (z_gt if z_value == 'oracle' else torch.full_like(z_gt, z_value))
                    z_pred, c_pos, c_neg = conditioner(desc, z_override=z_t)
                    return z_pred, conditioned_text_features(model, prompt_learner, c_pos, c_neg)

                shared = 'oracle' if args.ec_oracle else args.ec_const_z
                z_img = args.ec_z_img if args.ec_z_img is not None else shared
                z_pix = args.ec_z_pix if args.ec_z_pix is not None else shared
                z_pred, text_features_pix = text_at(z_pix)
                z_adapt = z_gt if z_pix == 'oracle' else (torch.full_like(z_gt, z_pix) if z_pix is not None else z_pred)
                text_features_img = text_features_pix if z_img == z_pix else text_at(z_img)[1]
                if z_pred is not None:
                    z_pred_val = float(z_pred[0])

            text_probs = image_features @ text_features_img.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            if args.export_image_scores and args.metrics == 'image-level':
                image_level_rows.append({
                    'dataset': os.path.basename(os.path.normpath(args.data_path)),
                    'sample_id': os.path.relpath(items['img_path'][0], args.data_path),
                    'ground_truth_image_label': int(items['anomaly'][0]),
                    'image_anomaly_score': float(text_probs.detach().cpu()[0]),
                })
            if not need_pixel:
                continue
            anomaly_map_list = []; d_list = []
            for idx, raw_patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    # --paa_scales (default empty = original behaviour): mean over the per-scale anomaly maps of this layer
                    scale_maps = []
                    for paa_s in (args.paa_scales if args.paa_scales else [None]):
                        patch_feature = raw_patch_feature if paa_s is None else paa_torch(raw_patch_feature, paa_s)
                        if adapter is not None:
                            patch_feature = adapter(patch_feature, z_adapt)
                        patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                        similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features_pix[0])
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                        scale_maps.append((similarity_map[...,1] + 1 - similarity_map[...,0])/2.0)
                    anomaly_map = scale_maps[0] if len(scale_maps) == 1 else torch.stack(scale_maps).mean(0)
                    if dist_T is not None:
                        pr = (100.0 * patch_feature[:, 1:, :].float() @ dist_T.T).softmax(-1)
                        d = pr[..., dist_mask].sum(-1, keepdim=True)  # [1, N, 1]
                        d_map = AnomalyCLIP_lib.get_similarity_map(d, args.image_size)[..., 0]
                        if args.distractor_rerank != 'none':
                            d_list.append(d_map)   # module C: used after smoothing, per pool component
                        else:
                            anomaly_map = anomaly_map * (1 - d_map)
                    # The following code is equivalent. 
                    # anomaly_map = similarity_map[...,1] 
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            
            anomaly_map = anomaly_map.sum(dim = 0)
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
            if args.distractor_rerank != 'none':
                d_avg = torch.stack(d_list).mean(0).detach().cpu().numpy()
                anomaly_map = torch.stack([torch.from_numpy(rerank_pool(a_, d_)) for a_, d_ in zip(anomaly_map.numpy(), d_avg)], dim = 0)
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name, gt_mask.detach().cpu().numpy())

            gt_flat = gt_mask[0, 0].detach().cpu().numpy().ravel()
            pred_flat = anomaly_map[0].detach().cpu().numpy().ravel()
            if gt_flat.min() != gt_flat.max():
                score = roc_auc_score(gt_flat, pred_flat)
                img_path = items['img_path'][0]
                cls = img_path.split('/')[-2]
                filename = img_path.split('/')[-1]
                vis_path = os.path.join(args.save_path, 'imgs', cls_name[0], cls, filename)
                bad_case_records.append((score, img_path, vis_path))
                per_image_rows.append((img_path, float(gt_flat.mean()), float(score), z_pred_val, z_gt_val))

    with open(os.path.join(args.save_path, 'per_image.csv'), 'w') as f:
        f.write('image,area_frac,auroc,z_pred,z_gt\n')
        f.writelines(f'{p},{a:.6f},{s:.6f},{zp:.6f},{zg:.6f}\n' for p, a, s, zp, zg in per_image_rows)

    bad_case_records.sort(key=lambda x: x[0])
    bad_cases_dir = os.path.join(args.save_path, 'bad_cases')
    if os.path.isdir(bad_cases_dir):
        shutil.rmtree(bad_cases_dir)
    os.makedirs(bad_cases_dir, exist_ok=True)
    for rank, (score, img_path, vis_path) in enumerate(bad_case_records[:50], start=1):
        if not os.path.isfile(vis_path):
            continue
        ext = os.path.splitext(vis_path)[1]
        dst = os.path.join(bad_cases_dir, f"{rank:02d}_auroc{score:.3f}_{os.path.splitext(os.path.basename(vis_path))[0]}{ext}")
        shutil.copyfile(vis_path, dst)
    logger.info(f"Saved top {min(50, len(bad_case_records))} worst cases to {bad_cases_dir}")

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        if need_pixel:
            results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
            results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            if args.export_image_scores:
                image_level_metric_values[obj] = {
                    'image_auroc': float(image_auroc), 'image_ap': float(image_ap)}
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
        elif args.metrics == 'pixel-auroc':
            # fast mode: skips the (slow) PRO computation
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            if args.export_pixel_metrics:
                pixel_level_metric_values[obj] = {
                    'pixel_auroc': float(pixel_auroc), 'pixel_aupro': float(pixel_aupro)}
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        summary_table = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-auroc':
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1))])
        summary_table = tabulate(table_ls, headers=['objects', 'pixel_auroc'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                       ])
        summary_table = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        summary_table = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
    logger.info("\n%s", summary_table)
    if args.export_image_scores and args.metrics == 'image-level':
        image_level_metric_values['mean'] = {
            'image_auroc': float(np.mean(image_auroc_list)),
            'image_ap': float(np.mean(image_ap_list)),
        }
        _write_image_level_artifacts(args, image_level_rows, image_level_metric_values, checkpoint)
    if args.export_pixel_metrics and args.metrics == 'pixel-level':
        pixel_level_metric_values['mean'] = {
            'pixel_auroc': float(np.mean(pixel_auroc_list)),
            'pixel_aupro': float(np.mean(pixel_aupro_list)),
        }
        _write_pixel_level_artifacts(
            args, per_image_rows, pixel_level_metric_values, checkpoint, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--export_image_scores", action="store_true",
                        help="save paired image-level scores, full-precision metrics, and run metadata")
    parser.add_argument("--export_pixel_metrics", action="store_true",
                        help="save full-precision pixel metrics, paired per-image mask statistics, and run metadata")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--ec_const_z", type=float, default=None,
                        help="condition every image on the same extent z, for BOTH heads (ignores the estimator)")
    parser.add_argument("--ec_z_img", type=float, default=None,
                        help="fixed z for the image-level (CLS) head only; overrides --ec_const_z for that head")
    parser.add_argument("--ec_z_pix", type=float, default=None,
                        help="fixed z for the pixel (patch) head only; overrides --ec_const_z for that head")
    parser.add_argument("--paa_scales", type=int, nargs="*", default=[], help="odd window sizes of Patch Average Aggregation, e.g. 1 3 5 (empty = off, original behaviour)")
    parser.add_argument("--distractor_suppress", choices=['none', 'all9', 'pruned'], default='none',
                        help="pixel-head test-time suppression by raw-CLIP distractor-concept mass (all9 = pre-registered A, pruned = B, held-out only)")
    parser.add_argument("--distractor_rerank", choices=['none', 'all9', 'pruned'], default='none',
                        help="module C: component-gated re-ranking inside the top-10%% pool by distractor mass (pre-registered; all9 only)")
    parser.add_argument("--ec_oracle", action="store_true",
                        help="diagnostic only: condition on the GROUND-TRUTH lesion extent instead of the estimate")
    
    args = parser.parse_args()
    if args.export_pixel_metrics and args.metrics != 'pixel-level':
        parser.error('--export_pixel_metrics requires --metrics pixel-level')
    print(args)
    setup_seed(args.seed)
    test(args)
