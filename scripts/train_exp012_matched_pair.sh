#!/usr/bin/env bash
# Prepare matched EXP-012 training checkpoints on the GPU server.
# This script performs TWO 15-epoch trainings (~200 minutes total) when run:
#   ECP extent, zoom_aug_p=0.5, seed=111
#   zoom-only,  zoom_aug_p=0.5, seed=111
# Set PYTHON and DATA_ROOT first. Example:
#   PYTHON="$(command -v python)" DATA_ROOT="$PWD/data" bash scripts/train_exp012_matched_pair.sh
# No evaluation is run. Outputs are kept under checkpoints/EXP-012-matched/.
set -Eeuo pipefail

PYTHON="${PYTHON:-}"
DATA_ROOT="${DATA_ROOT:-}"
CLIP_WEIGHTS="${CLIP_WEIGHTS:-$HOME/.cache/clip/ViT-L-14-336px.pt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${EXP012_TRAIN_ROOT:-$REPO_ROOT/checkpoints/EXP-012-matched}"
MVTEC_ROOT=""

fail() { printf 'EXP-012 matched training error: %s\n' "$*" >&2; exit 1; }
[[ -n "$PYTHON" ]] || fail 'set PYTHON to the server project interpreter'
[[ -x "$PYTHON" ]] || fail "PYTHON is not executable: $PYTHON"
[[ -n "$DATA_ROOT" ]] || fail 'set DATA_ROOT to the parent directory containing mvtec/'
[[ -r "$CLIP_WEIGHTS" ]] || fail "CLIP weights missing/unreadable: $CLIP_WEIGHTS"
DATA_ROOT="$("$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$DATA_ROOT")"
CLIP_WEIGHTS="$("$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CLIP_WEIGHTS")"
MVTEC_ROOT="$DATA_ROOT/mvtec"
[[ -f "$MVTEC_ROOT/meta.json" ]] || fail "missing MVTec meta.json: $MVTEC_ROOT/meta.json"
[[ ! -e "$OUT_ROOT" ]] || fail "output already exists: $OUT_ROOT (move it aside; results are never overwritten)"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES EXP012_REPO_ROOT="$REPO_ROOT" EXP012_DATA_ROOT="$DATA_ROOT"
export EXP012_CLIP_WEIGHTS="$CLIP_WEIGHTS"
"$PYTHON" - <<'PY' || fail 'project environment preflight failed'
import importlib
import os
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable to PYTHON')
for name in ('torchvision', 'numpy', 'sklearn', 'scipy', 'PIL', 'tqdm'):
    importlib.import_module(name)
importlib.import_module('train')
print(f"Python: {os.sys.executable}")
print(f"CUDA device count: {torch.cuda.device_count()}")
PY

mkdir -p "$OUT_ROOT"
COMMON=(--dataset mvtec --train_data_path "$MVTEC_ROOT"
        --features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4
        --batch_size 8 --print_freq 1 --epoch 15 --save_freq 1
        --learning_rate 0.001 --seed 111 --zoom_aug_p 0.5
        --consistency_weight 0.0 --ec_weight 1.0 --ec_teacher_p 0.5)

run_arm() {
  local arm="$1" mode="$2"
  local run_dir="$OUT_ROOT/$arm"
  mkdir -p "$run_dir"
  local cmd=("$PYTHON" "$REPO_ROOT/train.py" "${COMMON[@]}"
             --save_path "$run_dir/checkpoints" --extent_cond "$mode")
  printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  printf 'Training EXP-012 %s (log: %s/stdout_stderr.log)\n' "$arm" "$run_dir"
  "${cmd[@]}" > "$run_dir/stdout_stderr.log" 2>&1 ||
    fail "$arm training failed; preserve and inspect $run_dir/stdout_stderr.log"
  [[ -f "$run_dir/checkpoints/epoch_15.pth" ]] ||
    fail "$arm finished without epoch_15.pth"
  EXP012_ARM="$arm" EXP012_MODE="$mode" EXP012_RUN_DIR="$run_dir" \
    EXP012_COMMAND="$(cat "$run_dir/command.txt")" \
    "$PYTHON" - <<'PY' || fail "could not write $arm training manifest"
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

repo = Path(os.environ['EXP012_REPO_ROOT'])
data = Path(os.environ['EXP012_DATA_ROOT']) / 'mvtec'
run_dir = Path(os.environ['EXP012_RUN_DIR'])
checkpoint = run_dir / 'checkpoints/epoch_15.pth'
try:
    checkpoint_state = torch.load(checkpoint, map_location='cpu', weights_only=False)
except TypeError:
    checkpoint_state = torch.load(checkpoint, map_location='cpu')
expected_mode = os.environ['EXP012_MODE']
actual_mode = checkpoint_state.get('extent_cond', 'none')
if actual_mode != expected_mode:
    raise SystemExit(f'checkpoint extent_cond={actual_mode!r}; expected {expected_mode!r}')
if 'prompt_learner' not in checkpoint_state:
    raise SystemExit('checkpoint is missing prompt_learner weights')
if expected_mode == 'extent' and 'conditioner' not in checkpoint_state:
    raise SystemExit('ECP checkpoint is missing conditioner weights')
if expected_mode == 'none' and 'conditioner' in checkpoint_state:
    raise SystemExit('zoom-only checkpoint unexpectedly contains conditioner weights')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
sources = ['train.py', 'dataset.py', 'utils.py', 'prompt_ensemble.py',
           'extent_prompt.py', 'loss.py', 'AnomalyCLIP_lib/bpe_simple_vocab_16e6.txt.gz']
sources += [str(p.relative_to(repo)) for p in sorted((repo / 'AnomalyCLIP_lib').glob('*.py'))]
try:
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                       stderr=subprocess.DEVNULL, text=True).strip()
except (OSError, subprocess.CalledProcessError):
    revision = None
manifest = {
    'experiment': 'EXP-012', 'arm': os.environ['EXP012_ARM'],
    'extent_cond': os.environ['EXP012_MODE'],
    'command': os.environ['EXP012_COMMAND'],
    'checkpoint_path': str(checkpoint.resolve()), 'checkpoint_sha256': digest(checkpoint),
    'code_revision': revision,
    'source_sha256': {p: digest(repo / p) for p in sources},
    'clip_weights_path': os.environ['EXP012_CLIP_WEIGHTS'],
    'clip_weights_sha256': digest(Path(os.environ['EXP012_CLIP_WEIGHTS'])),
    'mvtec_root': str(data.resolve()), 'mvtec_meta_sha256': digest(data / 'meta.json'),
    'seed': 111, 'zoom_aug_p': 0.5, 'dataset': 'mvtec', 'epochs': 15,
    'features_list': [24], 'image_size': 518, 'depth': 9, 'n_ctx': 12,
    't_n_ctx': 4, 'batch_size': 8, 'learning_rate': 0.001,
    'consistency_weight': 0.0, 'ec_weight': 1.0, 'ec_teacher_p': 0.5,
}
(run_dir / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f"Saved training manifest: {run_dir / 'training_manifest.json'}")
PY
}

run_arm ecp_extent extent
run_arm zoom_only none

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
ecp = json.loads((root / 'ecp_extent/training_manifest.json').read_text())
zoom = json.loads((root / 'zoom_only/training_manifest.json').read_text())
for key in ('seed', 'zoom_aug_p', 'dataset', 'epochs', 'features_list', 'image_size',
            'depth', 'n_ctx', 't_n_ctx', 'batch_size', 'learning_rate',
            'consistency_weight', 'ec_weight', 'ec_teacher_p', 'clip_weights_sha256',
            'mvtec_meta_sha256', 'source_sha256'):
    if ecp[key] != zoom[key]:
        raise SystemExit(f"training arms differ on {key}")
print(f"EXP-012 matched training pair complete: {root}")
print(f"ECP checkpoint SHA256: {ecp['checkpoint_sha256']}")
print(f"Zoom-only checkpoint SHA256: {zoom['checkpoint_sha256']}")
PY
