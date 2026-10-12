"""EXP-040: audit the source zoom recipe and score paired normal-image views."""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

RESAMPLE_BICUBIC = getattr(Image, 'Resampling', Image).BICUBIC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp040_core import CATEGORIES, ARMS, VIEWS, sha256_file, stable_normals, transfer_box


def image_path(data_root, relative):
    path = Path(relative)
    return path if path.is_absolute() else data_root / path


def audit(args):
    import numpy as np
    from dataset import Dataset

    output = args.output
    geometry_path = output / 'geometry.json'
    views_path = output / 'views.jsonl'
    if geometry_path.exists() or views_path.exists():
        raise FileExistsError('EXP-040 audit outputs already exist; inspect before rerunning')
    output.mkdir(parents=True, exist_ok=True)
    meta_path = args.data_root / 'meta.json'
    with meta_path.open() as handle:
        meta = json.load(handle)
    if set(meta['test']) != set(CATEGORIES) or set(meta['train']) != set(CATEGORIES):
        raise ValueError('MVTec category list mismatch')
    random.seed(111)
    dataset = Dataset(root=str(args.data_root), transform=None, target_transform=None,
                      dataset_name='mvtec', mode='test')
    successful = {category: [] for category in CATEGORIES}
    category_counts = {category: {'anomalous_images': 0, 'attempts': 0, 'successful': 0,
                                  'invalid_mask': 0} for category in CATEGORIES}
    for category in CATEGORIES:
        anomalies = [row for row in meta['test'][category] if int(row['anomaly']) == 1]
        category_counts[category]['anomalous_images'] = len(anomalies)
        for row in anomalies:
            with Image.open(image_path(args.data_root, row['img_path'])) as image:
                width, height = image.size
            with Image.open(image_path(args.data_root, row['mask_path'])) as mask_image:
                if mask_image.size != (width, height):
                    category_counts[category]['invalid_mask'] += 1
                    continue
                # Dataset.__getitem__ thresholds the source mask at >0, then stores 0/255.
                mask = np.asarray(mask_image.convert('L')) > 0
            for epoch in range(15):
                if random.random() >= 0.5:
                    continue
                category_counts[category]['attempts'] += 1
                # Call the original augmentation method, including its random side and location.
                box = dataset._sample_crop_box(SimpleNamespace(size=(width, height)), mask, (0.10, 0.45))
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                successful[category].append({
                    'donor_id': row['img_path'], 'simulated_epoch': epoch,
                    'normalized_box': [x0 / width, y0 / height, x1 / width, y1 / height],
                    'side_fraction': (x1 - x0) / min(width, height),
                    'mask_fraction_in_crop': float(mask[y0:y1, x0:x1].mean()),
                })
                category_counts[category]['successful'] += 1
    if any(len(successful[category]) < 3 for category in CATEGORIES):
        raise ValueError('fewer than three successful crop geometries in a category')

    selection_rng = random.Random(111)
    views = []
    for category in CATEGORIES:
        normals = [row for row in meta['train'][category] if int(row['anomaly']) == 0]
        for row in stable_normals(normals, 32):
            path = image_path(args.data_root, row['img_path'])
            with Image.open(path) as image:
                size = image.size
            donor = selection_rng.choice(successful[category])
            box, ratio = transfer_box(donor['normalized_box'], size)
            views.append({
                'category': category, 'sample_id': row['img_path'],
                'image_sha256': sha256_file(path),
                'donor_id': donor['donor_id'], 'simulated_epoch': donor['simulated_epoch'],
                'normalized_box': donor['normalized_box'], 'pixel_box': box,
                'sham_side_ratio': ratio, 'image_size': size,
            })
    with views_path.open('w') as handle:
        for row in views:
            handle.write(json.dumps(row) + '\n')
    record = {
        'experiment': 'EXP-040', 'data_root': str(args.data_root.resolve()),
        'meta_json_sha256': sha256_file(meta_path),
        'dataset_py_sha256': sha256_file(ROOT / 'dataset.py'),
        'probe_py_sha256': sha256_file(Path(__file__)),
        'simulation_seed': 111, 'simulated_epochs': 15, 'configured_zoom_probability': 0.5,
        'training_source_split': 'test', 'probe_normal_split': 'train',
        'category_counts': category_counts,
        'successful_geometries': successful,
        'selected_normal_count': len(views),
        'views_sha256': sha256_file(views_path),
    }
    geometry_path.write_text(json.dumps(record, indent=2) + '\n')
    make_gallery(args.data_root, output / 'gallery', views)
    print(json.dumps({
        'selected_normal_count': len(views),
        'attempts': sum(x['attempts'] for x in category_counts.values()),
        'successful': sum(x['successful'] for x in category_counts.values()),
        'geometry_path': str(geometry_path), 'gallery_path': str(output / 'gallery'),
    }, indent=2))


def make_gallery(data_root, gallery_dir, views):
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        for index, row in enumerate([row for row in views if row['category'] == category][:5]):
            with Image.open(image_path(data_root, row['sample_id'])) as source:
                image = source.convert('RGB')
            original = image.resize((224, 224), RESAMPLE_BICUBIC)
            cropped = image.crop(tuple(row['pixel_box'])).resize((224, 224), RESAMPLE_BICUBIC)
            paired = Image.new('RGB', (448, 224))
            paired.paste(original, (0, 0))
            paired.paste(cropped, (224, 0))
            paired.save(gallery_dir / f'{category}_{index:02d}.jpg', quality=90)


def load_views(args):
    geometry_path = args.output / 'geometry.json'
    views_path = args.output / 'views.jsonl'
    with geometry_path.open() as handle:
        geometry = json.load(handle)
    if geometry['meta_json_sha256'] != sha256_file(args.data_root / 'meta.json'):
        raise ValueError('source meta.json changed after audit')
    if geometry['data_root'] != str(args.data_root.resolve()):
        raise ValueError('source data root changed after audit')
    if geometry['dataset_py_sha256'] != sha256_file(ROOT / 'dataset.py'):
        raise ValueError('dataset.py changed after audit')
    if geometry['probe_py_sha256'] != sha256_file(Path(__file__)):
        raise ValueError('probe code changed after audit')
    if geometry['views_sha256'] != sha256_file(views_path):
        raise ValueError('selected views changed after audit')
    with views_path.open() as handle:
        views = [json.loads(line) for line in handle if line.strip()]
    if len(views) != 480:
        raise ValueError('expected 480 source-normal images')
    return geometry, views


def score(args, smoke=False):
    import torch
    import torch.nn.functional as F
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    geometry, views = load_views(args)
    if smoke:
        views = [next(row for row in views if row['category'] == category)
                 for category in CATEGORIES[:6]]
    score_path = args.output / ('scores_smoke.csv' if smoke else 'scores.csv')
    provenance_path = args.output / ('provenance_smoke.json' if smoke else 'provenance.json')
    if score_path.exists() or provenance_path.exists():
        raise FileExistsError('score outputs already exist; inspect before rerunning')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(111)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(111)
    params = {'Prompt_length': 12, 'learnabel_text_embedding_depth': 9,
              'learnabel_text_embedding_length': 4}
    model, _ = AnomalyCLIP_lib.load('ViT-L/14@336px', device='cpu', design_details=params)
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model, params)
    checkpoint_paths = {'ctrl': args.ctrl_ckpt, 'zo': args.zo_ckpt}
    text_features = {}
    with torch.no_grad():
        for arm in ARMS:
            checkpoint = torch.load(checkpoint_paths[arm], map_location='cpu')
            if checkpoint.get('extent_cond') not in (None, 'none'):
                raise ValueError(f'{arm} checkpoint has an extent conditioner')
            prompt_learner.load_state_dict(checkpoint['prompt_learner'])
            prompt_learner.eval()
            prompts, tokens, compound = prompt_learner(cls_id=None)
            features = model.encode_text_learn(prompts, tokens, compound).float()
            features = torch.stack(torch.chunk(features, dim=0, chunks=2), dim=1)
            features = F.normalize(features, dim=-1)
            text_features[arm] = features[0].detach().to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    preprocess, _ = get_transform(SimpleNamespace(image_size=518))
    rows = []
    with torch.no_grad():
        for index, row in enumerate(views, 1):
            path = image_path(args.data_root, row['sample_id'])
            if sha256_file(path) != row['image_sha256']:
                raise ValueError(f'input image changed after audit: {row["sample_id"]}')
            with Image.open(path) as source:
                image = source.convert('RGB')
            if list(image.size) != row['image_size']:
                raise ValueError(f'image dimensions changed: {row["sample_id"]}')
            ratio = row['sham_side_ratio']
            reduced_size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
            variants = {
                'original': image,
                'crop': image.crop(tuple(row['pixel_box'])),
                'sham': image.resize(reduced_size, RESAMPLE_BICUBIC).resize(
                    image.size, RESAMPLE_BICUBIC),
            }
            for view in VIEWS:
                tensor = preprocess(variants[view]).unsqueeze(0).to(device)
                image_features, _ = model.encode_image(tensor, [24], DPAM_layer=20)
                image_features = F.normalize(image_features, dim=-1)
                margins = {}
                for arm in ARMS:
                    logits = image_features @ text_features[arm].T / 0.07
                    margins[f'{arm}_margin'] = float((logits[0, 1] - logits[0, 0]).cpu())
                rows.append({'category': row['category'], 'sample_id': row['sample_id'],
                             'view': view, 'sham_side_ratio': ratio, **margins})
            if index % 50 == 0:
                print(f'Scored {index}/{len(views)} source-normal images', flush=True)
    with score_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'category', 'sample_id', 'view', 'sham_side_ratio', 'ctrl_margin', 'zo_margin'])
        writer.writeheader()
        writer.writerows(rows)
    clip_path = Path.home() / '.cache/clip/ViT-L-14-336px.pt'
    provenance = {
        'experiment': 'EXP-040', 'smoke': smoke, 'image_count': len(views),
        'git_revision': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'score_count': len(rows), 'score_sha256': sha256_file(score_path),
        'geometry_sha256': sha256_file(args.output / 'geometry.json'),
        'views_sha256': geometry['views_sha256'],
        'meta_json_sha256': geometry['meta_json_sha256'],
        'checkpoint_paths': {arm: str(path.resolve()) for arm, path in checkpoint_paths.items()},
        'checkpoint_sha256': {arm: sha256_file(path) for arm, path in checkpoint_paths.items()},
        'clip_weights_sha256': sha256_file(clip_path) if clip_path.is_file() else None,
        'source_sha256': {name: sha256_file(ROOT / name) for name in (
            'scripts/exp040_probe.py', 'scripts/exp040_core.py', 'dataset.py',
            'utils.py', 'prompt_ensemble.py', 'AnomalyCLIP_lib/AnomalyCLIP.py',
            'AnomalyCLIP_lib/model_load.py')},
        'settings': {'image_size': 518, 'features_list': [24], 'dpam_layer': 20,
                     'depth': 9, 'n_ctx': 12, 't_n_ctx': 4, 'seed': 111},
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'image_count': len(views), 'score_path': str(score_path),
                      'provenance_path': str(provenance_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('audit', 'smoke', 'probe'))
    parser.add_argument('--data-root', type=Path,
                        default=Path('/home/ai3/NguyenND/AnomalyClipV2/data/mvtec'))
    parser.add_argument('--output', type=Path, default=Path('results/EXP-040'))
    parser.add_argument('--ctrl-ckpt', type=Path,
                        default=Path('checkpoints/matched_ctrl_s111/epoch_15.pth'))
    parser.add_argument('--zo-ckpt', type=Path,
                        default=Path('checkpoints/EXP-012-matched/zoom_only/checkpoints/epoch_15.pth'))
    args = parser.parse_args()
    if args.phase == 'audit':
        audit(args)
    else:
        score(args, smoke=args.phase == 'smoke')


if __name__ == '__main__':
    main()
