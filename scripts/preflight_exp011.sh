#!/usr/bin/env bash
# EXP-011 server preflight. From the repository root, set PYTHON, CHECKPOINT,
# DATA_ROOT, and CLIP_WEIGHTS, then run: bash scripts/preflight_exp011.sh
# CLIP_WEIGHTS must be the file test.py loads from ~/.cache/clip/.
# The private --verify-ready mode is used by run_exp011.sh; it does not infer runs.
set -Eeuo pipefail

PYTHON="${PYTHON:-}"
CHECKPOINT="${CHECKPOINT:-}"
DATA_ROOT="${DATA_ROOT:-}"
CLIP_WEIGHTS="${CLIP_WEIGHTS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$REPO_ROOT/results/EXP-011"
PREFLIGHT_ROOT="$OUT_ROOT/preflight"
READY_FILE="$PREFLIGHT_ROOT/ready.json"
MODE="${1:-preflight}"

fail() { printf 'EXP-011 preflight error: %s\n' "$*" >&2; exit 1; }
[[ "$MODE" == preflight || "$MODE" == --verify-ready ]] || fail "usage: bash scripts/preflight_exp011.sh [--verify-ready]"
for name in PYTHON CHECKPOINT DATA_ROOT CLIP_WEIGHTS; do
  [[ -n "${!name}" ]] || fail "set $name to its server path before running"
done
command -v "$PYTHON" >/dev/null 2>&1 || fail "PYTHON is not executable: $PYTHON"
PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')" || fail "cannot start PYTHON: $PYTHON"
CHECKPOINT="$("$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT")"
DATA_ROOT="$("$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$DATA_ROOT")"
CLIP_WEIGHTS="$("$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CLIP_WEIGHTS")"
[[ -f "$CHECKPOINT" && -r "$CHECKPOINT" ]] || fail "checkpoint missing/unreadable: $CHECKPOINT"
[[ -f "$CLIP_WEIGHTS" && -r "$CLIP_WEIGHTS" ]] || fail "CLIP weights missing/unreadable: $CLIP_WEIGHTS"
[[ -d "$DATA_ROOT" ]] || fail "DATA_ROOT is not a directory: $DATA_ROOT"
[[ -f "$REPO_ROOT/test.py" ]] || fail "test.py is missing from $REPO_ROOT"
mkdir -p "$PREFLIGHT_ROOT" || fail "cannot create $PREFLIGHT_ROOT"
[[ -w "$PREFLIGHT_ROOT" ]] || fail "output directory is not writable: $PREFLIGHT_ROOT"
for dataset in HeadCT BrainMRI Br35H; do
  mkdir -p "$OUT_ROOT/$dataset" || fail "cannot create output parent for $dataset"
  [[ -w "$OUT_ROOT/$dataset" ]] || fail "output parent is not writable for $dataset"
  if [[ "$MODE" == preflight ]]; then
    for arm in zimg_1.6 zimg_-0.69; do
      [[ ! -e "$OUT_ROOT/$dataset/$arm" ]] ||
        fail "full-run output already exists: $OUT_ROOT/$dataset/$arm"
    done
  fi
done
for folder in HeadCT_anomaly_detection BrainMRI br35; do
  [[ -f "$DATA_ROOT/$folder/meta.json" ]] || fail "missing $DATA_ROOT/$folder/meta.json"
done
cd "$REPO_ROOT"

export EXP011_REPO_ROOT="$REPO_ROOT" EXP011_DATA_ROOT="$DATA_ROOT"
export EXP011_CHECKPOINT="$CHECKPOINT" EXP011_CLIP_WEIGHTS="$CLIP_WEIGHTS"
CURRENT_FILE="$(mktemp "$PREFLIGHT_ROOT/current.XXXXXXXX.json")" || fail 'cannot create a writable preflight file'
trap 'rm -f "$CURRENT_FILE"' EXIT
if [[ "$MODE" == preflight ]]; then
  # A failed new preflight must not leave an old pass marker usable.
  rm -f "$READY_FILE"
fi

"$PYTHON" - "$CURRENT_FILE" <<'PY' || fail 'environment, checkpoint, CLIP, or dataset validation failed'
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path(os.environ['EXP011_REPO_ROOT'])
data_root = Path(os.environ['EXP011_DATA_ROOT']).resolve()
checkpoint_path = Path(os.environ['EXP011_CHECKPOINT']).resolve()
clip_path = Path(os.environ['EXP011_CLIP_WEIGHTS']).resolve()
expected_cache = (Path.home() / '.cache/clip/ViT-L-14-336px.pt').resolve()
if clip_path != expected_cache:
    raise SystemExit(f'CLIP_WEIGHTS must resolve to the loader cache file {expected_cache}; got {clip_path}')

for module in ('torch', 'torchvision', 'numpy', 'sklearn', 'scipy', 'skimage', 'PIL', 'tabulate', 'tqdm'):
    importlib.import_module(module)
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in PYTHON; use the project GPU evaluation environment')
importlib.import_module('test')
help_run = subprocess.run([sys.executable, str(repo / 'test.py'), '--help'], cwd=repo,
                          capture_output=True, text=True, check=True)
required_flags = ('--dataset', '--metrics', '--ec_z_img', '--ec_z_pix', '--sigma', '--seed',
                  '--features_list', '--image_size', '--depth', '--n_ctx', '--t_n_ctx',
                  '--checkpoint_path', '--data_path', '--save_path', '--export_image_scores',
                  '--ec_const_z', '--ec_oracle', '--distractor_suppress', '--distractor_rerank')
missing_flags = [flag for flag in required_flags if flag not in help_run.stdout]
if missing_flags:
    raise SystemExit(f'test.py lacks required arguments: {missing_flags}')

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

# model_load.py names and validates this exact CLIP asset before test.py loads it.
clip_hash = sha256(clip_path)
expected_clip_hash = '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'
if clip_hash != expected_clip_hash:
    raise SystemExit(f'CLIP weights SHA256 differs from ViT-L/14@336px: {clip_hash}')
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
except TypeError:  # older torch versions predate weights_only
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
if not isinstance(checkpoint, dict) or checkpoint.get('extent_cond') != 'extent':
    raise SystemExit("checkpoint extent_cond must be 'extent'")
if 'prompt_learner' not in checkpoint or 'conditioner' not in checkpoint:
    raise SystemExit('checkpoint lacks prompt_learner or conditioner weights')

source_names = ['test.py', 'metrics.py', 'dataset.py', 'utils.py', 'extent_prompt.py',
                'prompt_ensemble.py', 'logger.py', 'visualization.py', 'distractor_stats.py',
                'scripts/preflight_exp011.sh', 'scripts/run_exp011.sh']
source_names += [str(path.relative_to(repo)) for path in sorted((repo / 'AnomalyCLIP_lib').glob('*.py'))]
source_names += ['AnomalyCLIP_lib/bpe_simple_vocab_16e6.txt.gz']
sources = {name: sha256(repo / name) for name in source_names}

datasets = {}
for name, folder in (('HeadCT', 'HeadCT_anomaly_detection'),
                     ('BrainMRI', 'BrainMRI'), ('Br35H', 'br35')):
    root = data_root / folder
    meta_path = root / 'meta.json'
    meta = json.loads(meta_path.read_text())
    samples = meta.get('test', {}).get('brain')
    if not isinstance(samples, list) or not samples:
        raise SystemExit(f'{name}: meta.json needs a nonempty test/brain sample list')
    manifest = hashlib.sha256()
    counts = {0: 0, 1: 0}
    for item in samples:
        if item.get('cls_name') != 'brain' or 'specie_name' not in item:
            raise SystemExit(f'{name}: invalid brain sample fields in meta.json')
        label = item.get('anomaly')
        if label not in (0, 1):
            raise SystemExit(f'{name}: image label must be 0 or 1')
        image = root / item['img_path']
        if not image.is_file():
            raise SystemExit(f'{name}: missing image {image}')
        if label == 1:
            mask = root / item['mask_path']
            if not (mask.is_file() or mask.is_dir()):
                raise SystemExit(f'{name}: missing mask path/directory {mask}')
        counts[label] += 1
        stat = image.stat()
        record = (str(image.resolve()), label, stat.st_size, stat.st_mtime_ns)
        manifest.update((json.dumps(record, separators=(',', ':')) + '\n').encode())
    if min(counts.values()) == 0:
        raise SystemExit(f'{name}: AUROC/AP requires both normal and anomalous images')
    datasets[name] = {'path': str(root.resolve()), 'meta_sha256': sha256(meta_path),
                      'image_manifest_sha256': manifest.hexdigest(),
                      'sample_count': len(samples), 'negative_count': counts[0],
                      'positive_count': counts[1]}

fingerprint = {
    'python': str(Path(sys.executable).resolve()), 'checkpoint_path': str(checkpoint_path),
    'checkpoint_sha256': sha256(checkpoint_path), 'checkpoint_extent_cond': 'extent',
    'clip_weights_path': str(clip_path), 'clip_weights_sha256': clip_hash,
    'data_root': str(data_root), 'datasets': datasets, 'source_sha256': sources,
    'fixed_settings': {'dataset': 'brain', 'metrics': 'image-level', 'ec_z_pix': 1.6,
                       'sigma': 4, 'seed': 111, 'features_list': [24], 'image_size': 518,
                       'depth': 9, 'n_ctx': 12, 't_n_ctx': 4,
                       'ec_const_z': None, 'ec_oracle': False,
                       'distractor_suppress': 'none', 'distractor_rerank': 'none'},
}
Path(sys.argv[1]).write_text(json.dumps(fingerprint, sort_keys=True, indent=2) + '\n')
PY

if [[ "$MODE" == --verify-ready ]]; then
  [[ -f "$READY_FILE" ]] || fail "run bash scripts/preflight_exp011.sh successfully first"
  cmp -s "$CURRENT_FILE" "$READY_FILE" || fail 'paths, checkpoint, weights, code, or dataset manifest changed since preflight; rerun preflight'
  exit 0
fi

SMOKE_ROOT="$(mktemp -d "$PREFLIGHT_ROOT/smoke.XXXXXXXX")" || fail 'cannot create smoke directory'
export EXP011_SMOKE_ROOT="$SMOKE_ROOT"
"$PYTHON" - <<'PY' || fail 'could not build a two-image smoke subset'
import json
import os
from pathlib import Path

root = Path(os.environ['EXP011_DATA_ROOT']) / 'HeadCT_anomaly_detection'
smoke = Path(os.environ['EXP011_SMOKE_ROOT']) / 'data'
(smoke / 'images').mkdir(parents=True)
(smoke / 'masks').mkdir()
items = json.loads((root / 'meta.json').read_text())['test']['brain']
selected = {label: next(item for item in items if item['anomaly'] == label) for label in (0, 1)}
rows = []
for label, name in ((0, 'negative'), (1, 'positive')):
    item = selected[label]
    (smoke / 'images' / name).symlink_to((root / item['img_path']).resolve())
    rows.append({'img_path': f'images/{name}', 'mask_path': 'masks',
                 'cls_name': 'brain', 'specie_name': 'brain', 'anomaly': label})
(smoke / 'meta.json').write_text(json.dumps({'test': {'brain': rows}}) + '\n')
PY

FIXED=(--dataset brain --metrics image-level --ec_z_pix 1.6 --sigma 4 --seed 111
       --features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4
       --distractor_suppress none --distractor_rerank none)
smoke_eval() {
  local name="$1" z_img="$2" export_scores="$3" out="$SMOKE_ROOT/$1"
  local cmd=("$PYTHON" "$REPO_ROOT/test.py" --data_path "$SMOKE_ROOT/data"
             --checkpoint_path "$CHECKPOINT" --save_path "$out" "${FIXED[@]}"
             --ec_z_img "$z_img")
  if [[ "$export_scores" == yes ]]; then cmd+=(--export_image_scores); fi
  mkdir -p "$out"
  printf '%q ' "${cmd[@]}" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  (cd "$REPO_ROOT" && "${cmd[@]}") > "$out/stdout_stderr.log" 2>&1 ||
    fail "smoke evaluation $name failed; see $out/stdout_stderr.log"
}

smoke_eval arm_a_export 1.6 yes
smoke_eval arm_a_default 1.6 no
smoke_eval arm_b_export -0.69 yes
smoke_eval arm_b_default -0.69 no
# These Python float arguments differ, forcing the recompute branch, but both
# round to the same float32 z used by ExtentConditioner. Diagnostic only.
smoke_eval arm_a_recompute 1.60000001 yes

"$PYTHON" - "$SMOKE_ROOT" "$CURRENT_FILE" <<'PY' || fail 'smoke artifact/metric/parity validation failed'
import csv
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
import torch
from metrics import image_level_metrics

smoke = Path(sys.argv[1])
fingerprint = json.loads(Path(sys.argv[2]).read_text())
if 1.6 == 1.60000001 or torch.tensor(1.6, dtype=torch.float32) != torch.tensor(1.60000001, dtype=torch.float32):
    raise SystemExit('same-z diagnostic does not select equal float32 values and distinct Python values')

def log_metrics(name):
    lines = (smoke / name / 'log.txt').read_text().splitlines()
    pattern = re.compile(r'\|\s*brain\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|')
    match = next((found for line in lines if (found := pattern.search(line)) is not None), None)
    if match is None:
        raise SystemExit(f'{name}: no brain AUROC/AP row in log')
    return tuple(float(value) for value in match.groups())

def export(name):
    folder = smoke / name
    with (folder / 'image_level_predictions.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    meta = json.loads((folder / 'evaluation_metadata.json').read_text())
    metrics = json.loads((folder / 'image_level_metrics.json').read_text())
    args = meta['arguments']
    expected_z = -0.69 if name.startswith('arm_b') else (1.60000001 if name == 'arm_a_recompute' else 1.6)
    if (any(args.get(key) != value for key, value in fingerprint['fixed_settings'].items()) or
        args.get('ec_z_img') != expected_z or not args.get('export_image_scores') or
        meta['checkpoint_sha256'] != fingerprint['checkpoint_sha256'] or
        meta['checkpoint_extent_cond'] != 'extent' or
        meta['pretrained_clip_weights_sha256'] != fingerprint['clip_weights_sha256']):
        raise SystemExit(f'{name}: fixed settings, checkpoint, or CLIP weights differ')
    if len(rows) != 2 or meta['sample_count'] != len(rows) or (meta['negative_count'], meta['positive_count']) != (1, 1):
        raise SystemExit(f'{name}: row count or class counts differ from evaluated samples')
    ids = [row['sample_id'] for row in rows]
    labels = [int(row['ground_truth_image_label']) for row in rows]
    scores = [float(row['image_anomaly_score']) for row in rows]
    if len(set(ids)) != 2 or set(labels) != {0, 1} or not all(math.isfinite(x) for x in scores):
        raise SystemExit(f'{name}: invalid IDs, labels, or scores')
    if not all(row['dataset'] == 'data' for row in rows):
        raise SystemExit(f'{name}: dataset field missing or incorrect')
    source = {'brain': {'gt_sp': labels, 'pr_sp': scores}}
    for key, metric in (('image_auroc', 'image-auroc'), ('image_ap', 'image-ap')):
        recomputed = image_level_metrics(source, 'brain', metric)
        if not math.isclose(recomputed, metrics['brain'][key], rel_tol=0, abs_tol=1e-12):
            raise SystemExit(f'{name}: full-precision {key} disagrees with exported scores')
    rounded = tuple(float(np.round(metrics['brain'][key] * 100, decimals=1))
                    for key in ('image_auroc', 'image_ap'))
    if rounded != log_metrics(name):
        raise SystemExit(f'{name}: exported metrics differ from the aggregate evaluation log')
    return dict(zip(ids, zip(labels, scores)))

for arm in ('arm_a', 'arm_b'):
    scores = export(f'{arm}_export')
    if log_metrics(f'{arm}_export') != log_metrics(f'{arm}_default'):
        raise SystemExit(f'{arm}: AUROC/AP changed when score export was enabled')
    if set(scores) != {'images/negative', 'images/positive'}:
        raise SystemExit(f'{arm}: sample IDs do not match the smoke subset')
a = export('arm_a_export')
parity = export('arm_a_recompute')
if a.keys() != parity.keys() or any(a[key][0] != parity[key][0] or
                                      not math.isclose(a[key][1], parity[key][1], rel_tol=0, abs_tol=1e-6)
                                      for key in a):
    raise SystemExit('same-z reuse and recompute paths produced different scores')
PY

cp "$CURRENT_FILE" "$READY_FILE" || fail 'could not save preflight readiness marker'
printf 'EXP-011 PREFLIGHT PASSED\n'
