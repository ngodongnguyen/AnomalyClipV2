#!/usr/bin/env bash
# EXP-014 visual-branch pilot, no training or model updates.
# On the GPU server, set PYTHON and DATA_ROOT; CLIP_WEIGHTS defaults to the CLIP cache.
# First: bash scripts/run_exp014_probe.sh --smoke
# Then:  bash scripts/run_exp014_probe.sh --full
# The full run scores six masked datasets; neither mode overwrites prior output.
set -Eeuo pipefail

MODE="${1:-}"
[[ "$MODE" == --smoke || "$MODE" == --full ]] || {
  printf 'Usage: bash scripts/run_exp014_probe.sh --smoke|--full\n' >&2
  exit 2
}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-}"
DATA_ROOT="${DATA_ROOT:-}"
CLIP_WEIGHTS="${CLIP_WEIGHTS:-$HOME/.cache/clip/ViT-L-14-336px.pt}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}"
REFERENCE_ROOT="${EXP014_REFERENCE_ROOT:-$REPO_ROOT/results/EXP-012/matched-retry-20261002}"
INVENTORY="$REPO_ROOT/results/EXP-012/input_inventory.json"
MANIFEST="$REPO_ROOT/checkpoints/EXP-012-matched/ecp_extent/training_manifest.json"
SMOKE_ROOT="$REPO_ROOT/results/EXP-014/branch-pilot-smoke"
if [[ "$MODE" == --smoke ]]; then
  OUT_ROOT="${EXP014_OUT_ROOT:-$SMOKE_ROOT}"
else
  OUT_ROOT="${EXP014_OUT_ROOT:-$REPO_ROOT/results/EXP-014/branch-pilot-full}"
fi
fail() { printf 'EXP-014 pilot error: %s\n' "$*" >&2; exit 1; }
[[ -n "$PYTHON" && -x "$PYTHON" ]] || fail 'set PYTHON to the server project interpreter'
[[ -n "$DATA_ROOT" && -d "$DATA_ROOT" ]] || fail 'set DATA_ROOT to the server data directory'
[[ -f "$CHECKPOINT" && -f "$MANIFEST" ]] || fail 'missing matched ECP checkpoint or training manifest'
[[ -f "$CLIP_WEIGHTS" ]] || fail "missing CLIP weights: $CLIP_WEIGHTS"
[[ -f "$INVENTORY" ]] || fail "missing EXP-012 inventory: $INVENTORY"
[[ -d "$REFERENCE_ROOT" ]] || fail "missing EXP-012 reference: $REFERENCE_ROOT"
[[ ! -e "$OUT_ROOT" ]] || fail "output already exists: $OUT_ROOT"

export REPO_ROOT PYTHON DATA_ROOT CLIP_WEIGHTS CHECKPOINT REFERENCE_ROOT INVENTORY MANIFEST MODE SMOKE_ROOT
cd "$REPO_ROOT"
"$PYTHON" - <<'PY' || fail 'input provenance validation failed'
import hashlib, json, os
import importlib
from pathlib import Path

for module in ('torch', 'torchvision', 'numpy', 'scipy', 'sklearn', 'PIL'):
    importlib.import_module(module)
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable to PYTHON')

repo = Path(os.environ['REPO_ROOT'])
data = Path(os.environ['DATA_ROOT'])
ref = Path(os.environ['REFERENCE_ROOT'])
checkpoint = Path(os.environ['CHECKPOINT'])
clip = Path(os.environ['CLIP_WEIGHTS'])
manifest = json.loads(Path(os.environ['MANIFEST']).read_text())
inventory = json.loads(Path(os.environ['INVENTORY']).read_text())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

expected_cache = Path.home() / '.cache/clip/ViT-L-14-336px.pt'
if clip.resolve() != expected_cache.resolve():
    raise SystemExit('AnomalyCLIP_lib.load reads the home CLIP cache; CLIP_WEIGHTS must name that file')
if sha(checkpoint) != manifest['checkpoint_sha256']:
    raise SystemExit('ECP checkpoint hash differs from training manifest')
if manifest['extent_cond'] != 'extent' or manifest['seed'] != 111:
    raise SystemExit('checkpoint manifest is not the intended ECP seed-111 arm')
if sha(clip) != manifest['clip_weights_sha256']:
    raise SystemExit('CLIP weights hash differs from training manifest')

datasets = {
    'ISIC': 'ISIC', 'ClinicDB': 'CVC/CVC-ClinicDB',
    'ColonDB': 'CVC/CVC-ColonDB', 'Kvasir': 'CVC/Kvasir',
    'Endo': 'EndoTect_2020_Segmentation_Test_Dataset',
    'TN3K': 'TN3K/Thyroid Dataset/tn3k',
}
active = ('ClinicDB',) if os.environ['MODE'] == '--smoke' else tuple(datasets)
for name in active:
    folder = data / datasets[name]
    record = inventory['datasets'][name]
    meta = folder / 'meta.json'
    prior = ref / name / 'ecp_extent'
    if not meta.is_file() or sha(meta) != record['meta_json_sha256']:
        raise SystemExit(f'{name}: dataset split differs from EXP-012 inventory')
    for artifact in ('pixel_per_image_predictions.csv', 'evaluation_metadata.json'):
        if not (prior / artifact).is_file():
            raise SystemExit(f'{name}: missing EXP-012 reference {artifact}')
    old = json.loads((prior / 'evaluation_metadata.json').read_text())
    if old['checkpoint_sha256'] != manifest['checkpoint_sha256'] or old['meta_json_sha256'] != sha(meta):
        raise SystemExit(f'{name}: EXP-012 checkpoint or split differs')
    if old['pretrained_clip_weights_sha256'] != sha(clip):
        raise SystemExit(f'{name}: EXP-012 CLIP weights differ')
    args = old['arguments']
    required = {'metrics': 'pixel-level', 'features_list': [24], 'image_size': 518,
                'depth': 9, 'n_ctx': 12, 't_n_ctx': 4, 'sigma': 4, 'seed': 111,
                'ec_z_img': -0.69, 'ec_z_pix': 1.6,
                'distractor_suppress': 'none', 'distractor_rerank': 'none'}
    if any(args[key] != value for key, value in required.items()):
        raise SystemExit(f'{name}: EXP-012 reference settings differ from this pilot')
    for relative, old_sha in old['evaluation_source_sha256'].items():
        if sha(repo / relative) != old_sha:
            raise SystemExit(f'{name}: evaluation dependency changed: {relative}')

if os.environ['MODE'] == '--full':
    smoke = Path(os.environ['SMOKE_ROOT']) / 'ClinicDB'
    summary = smoke / 'summary.json'
    metadata = smoke / 'metadata.json'
    if not summary.is_file() or not metadata.is_file():
        raise SystemExit('run --smoke successfully before --full')
    if json.loads(metadata.read_text())['probe_source_sha256'] != sha(repo / 'scripts/exp014_branch_probe.py'):
        raise SystemExit('probe changed since smoke; rerun smoke under a new output root')
    if json.loads(summary.read_text())['max_exp012_vv_auroc_abs_error'] > 1e-6:
        raise SystemExit('smoke V-V parity failed')
PY

if [[ "$MODE" == --smoke ]]; then
  DATASETS=(ClinicDB)
  FOLDERS=(CVC/CVC-ClinicDB)
  MODES=(colon)
  LIMIT=(--limit 2)
else
  DATASETS=(ISIC ClinicDB ColonDB Kvasir Endo TN3K)
  FOLDERS=(ISIC CVC/CVC-ClinicDB CVC/CVC-ColonDB CVC/Kvasir EndoTect_2020_Segmentation_Test_Dataset 'TN3K/Thyroid Dataset/tn3k')
  MODES=(ISBI colon colon colon colon thyroid)
  LIMIT=()
fi
mkdir -p "$OUT_ROOT/logs"
for i in "${!DATASETS[@]}"; do
  name="${DATASETS[$i]}"
  output="$OUT_ROOT/$name"
  reference="$REFERENCE_ROOT/$name/ecp_extent/pixel_per_image_predictions.csv"
  cmd=("$PYTHON" "$REPO_ROOT/scripts/exp014_branch_probe.py"
       --data_path "$DATA_ROOT/${FOLDERS[$i]}" --dataset "${MODES[$i]}"
       --checkpoint_path "$CHECKPOINT" --reference_csv "$reference"
       --save_path "$output" --image_size 518 --sigma 4 --seed 111
       --ec_z_pix 1.6 --depth 9 --n_ctx 12 --t_n_ctx 4 "${LIMIT[@]}")
  printf '%q ' "${cmd[@]}" > "$OUT_ROOT/logs/$name.command.txt"
  printf '\n' >> "$OUT_ROOT/logs/$name.command.txt"
  printf 'Evaluating EXP-014 visual branches: %s\n' "$name"
  "${cmd[@]}" > "$OUT_ROOT/logs/$name.stdout_stderr.log" 2>&1 ||
    fail "$name failed; inspect $OUT_ROOT/logs/$name.stdout_stderr.log"
  [[ -s "$output/summary.json" && -s "$output/per_image_branch_probe.csv" && -s "$output/metadata.json" ]] ||
    fail "$name did not export all pilot artifacts"
done
if [[ "$MODE" == --smoke ]]; then
  printf 'EXP-014 BRANCH PILOT SMOKE PASSED\n'
else
  printf 'EXP-014 BRANCH PILOT COMPLETED: %s\n' "$OUT_ROOT"
fi
