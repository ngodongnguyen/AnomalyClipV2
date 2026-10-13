"""Score the fixed EXP-040 MVTec normal views with EXP-042 split and sham prompts."""

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.exp040_core import CATEGORIES, VIEWS, sha256_file


def main():
    import torch
    import torch.nn.functional as F
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    data_root = Path('/home/ai3/NguyenND/AnomalyClipV2/data/mvtec')
    source = ROOT / 'results/EXP-040'
    output = ROOT / 'results/EXP-042/source'
    output.mkdir(parents=True, exist_ok=True)
    scores_path = output / 'scores.csv'
    provenance_path = output / 'provenance.json'
    if scores_path.exists() or provenance_path.exists():
        raise FileExistsError('EXP-042 source probe already exists; inspect before rerunning')
    geometry_path, views_path = source / 'geometry.json', source / 'views.jsonl'
    geometry = json.loads(geometry_path.read_text())
    if geometry['views_sha256'] != sha256_file(views_path):
        raise ValueError('EXP-040 views hash changed')
    if geometry['meta_json_sha256'] != sha256_file(data_root / 'meta.json'):
        raise ValueError('MVTec metadata changed')
    views = [json.loads(line) for line in views_path.read_text().splitlines() if line.strip()]
    if len(views) != 480 or {row['category'] for row in views} != set(CATEGORIES):
        raise ValueError('wrong source-normal view list')
    if any('/train/good/' not in row['sample_id'].replace('\\', '/') for row in views):
        raise ValueError('source probe includes non-training normals')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(111)
    params = {'Prompt_length': 12, 'learnabel_text_embedding_depth': 9,
              'learnabel_text_embedding_length': 4}
    model, _ = AnomalyCLIP_lib.load('ViT-L/14@336px', device='cpu', design_details=params)
    model.eval()
    learner = AnomalyCLIP_PromptLearner(model, params)
    checkpoint_paths = {
        arm: ROOT / f'checkpoints/exp042_{arm}_s111/epoch_15.pth'
        for arm in ('split', 'sham')
    }
    features = {}
    with torch.no_grad():
        for arm, path in checkpoint_paths.items():
            checkpoint = torch.load(path, map_location='cpu')
            if checkpoint.get('exp042_cls_view') != arm or checkpoint.get('extent_cond') not in (None, 'none'):
                raise ValueError(f'wrong {arm} checkpoint')
            learner.load_state_dict(checkpoint['prompt_learner'])
            learner.eval()
            prompts, tokens, compound = learner(cls_id=None)
            text = model.encode_text_learn(prompts, tokens, compound).float()
            features[arm] = F.normalize(torch.stack(torch.chunk(text, 2), dim=1)[0], dim=-1).to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    preprocess, _ = get_transform(SimpleNamespace(image_size=518))
    bicubic = getattr(Image, 'Resampling', Image).BICUBIC
    rows = []
    with torch.no_grad():
        for index, row in enumerate(views, 1):
            path = data_root / row['sample_id']
            if sha256_file(path) != row['image_sha256']:
                raise ValueError(f'source image changed: {path}')
            with Image.open(path) as source_img:
                image = source_img.convert('RGB')
            if list(image.size) != row['image_size']:
                raise ValueError(f'wrong image size: {path}')
            ratio = row['sham_side_ratio']
            reduced = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
            variants = {
                'original': image,
                'crop': image.crop(tuple(row['pixel_box'])),
                'sham': image.resize(reduced, bicubic).resize(image.size, bicubic),
            }
            for view in VIEWS:
                tensor = preprocess(variants[view]).unsqueeze(0).to(device)
                image_features, _ = model.encode_image(tensor, [24], DPAM_layer=20)
                image_features = F.normalize(image_features, dim=-1)
                margins = {}
                for arm in checkpoint_paths:
                    logits = image_features @ features[arm].T / 0.07
                    margins[f'{arm}_margin'] = float((logits[0, 1] - logits[0, 0]).cpu())
                rows.append({'category': row['category'], 'sample_id': row['sample_id'],
                             'view': view, 'sham_side_ratio': ratio, **margins})
            if index % 50 == 0:
                print(f'Scored {index}/480 source normals', flush=True)
    with scores_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['category', 'sample_id', 'view',
                                                    'sham_side_ratio', 'split_margin', 'sham_margin'])
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        'experiment': 'EXP-042', 'score_count': len(rows),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'views_sha256': sha256_file(views_path), 'geometry_sha256': sha256_file(geometry_path),
        'scores_sha256': sha256_file(scores_path),
        'checkpoint_sha256': {arm: sha256_file(path) for arm, path in checkpoint_paths.items()},
        'checkpoint_paths': {arm: str(path) for arm, path in checkpoint_paths.items()},
        'source_sha256': {name: sha256_file(ROOT / name) for name in
                          ('scripts/exp042_source_probe.py', 'dataset.py', 'train.py', 'utils.py')},
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Wrote {scores_path}')


if __name__ == '__main__':
    main()
