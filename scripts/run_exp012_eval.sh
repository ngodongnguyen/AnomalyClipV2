#!/usr/bin/env bash
# Run EXP-012's 12 pixel-level evaluations after matched training.
# Set PYTHON and DATA_ROOT on the GPU server, then run: bash scripts/run_exp012_eval.sh
# Evaluation only; it does not train or overwrite prior results.
set -Eeuo pipefail
PYTHON="${PYTHON:-}"
DATA_ROOT="${DATA_ROOT:-}"
CLIP_WEIGHTS="${CLIP_WEIGHTS:-$HOME/.cache/clip/ViT-L-14-336px.pt}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ROOT="${EXP012_TRAIN_ROOT:-$REPO_ROOT/checkpoints/EXP-012-matched}"
OUT_ROOT="${EXP012_EVAL_ROOT:-$REPO_ROOT/results/EXP-012/matched}"
INVENTORY="$REPO_ROOT/results/EXP-012/input_inventory.json"
ECP_CHECKPOINT="$TRAIN_ROOT/ecp_extent/checkpoints/epoch_15.pth"
ZOOM_CHECKPOINT="$TRAIN_ROOT/zoom_only/checkpoints/epoch_15.pth"
fail() { printf 'EXP-012 evaluation error: %s\n' "$*" >&2; exit 1; }
[[ -n "$PYTHON" && -x "$PYTHON" ]] || fail 'set PYTHON to the server project interpreter'
[[ -n "$DATA_ROOT" && -d "$DATA_ROOT" ]] || fail 'set DATA_ROOT to the server data directory'
[[ -r "$CLIP_WEIGHTS" ]] || fail "missing CLIP weights: $CLIP_WEIGHTS"
[[ -f "$ECP_CHECKPOINT" && -f "$ZOOM_CHECKPOINT" ]] || fail 'matched checkpoints are missing'
[[ -f "$TRAIN_ROOT/ecp_extent/training_manifest.json" && -f "$TRAIN_ROOT/zoom_only/training_manifest.json" ]] || fail 'training manifests are missing'
[[ ! -e "$OUT_ROOT" ]] || fail "output already exists: $OUT_ROOT"
[[ -f "$INVENTORY" ]] || fail "missing dataset inventory: $INVENTORY"
export TRAIN_ROOT ECP_CHECKPOINT ZOOM_CHECKPOINT CLIP_WEIGHTS DATA_ROOT INVENTORY
cd "$REPO_ROOT"
HELP_TEXT="$("$PYTHON" test.py --help)" || fail 'could not inspect test.py CLI'
[[ "$HELP_TEXT" == *--export_pixel_metrics* ]] || fail 'sync the test.py pixel export update first'
"$PYTHON" - <<'PY' || fail 'training manifest/checkpoint validation failed'
import hashlib
import json
import os
from pathlib import Path
root=Path(os.environ['TRAIN_ROOT'])
data=Path(os.environ['DATA_ROOT'])
inventory=json.loads(Path(os.environ['INVENTORY']).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
ecp=json.loads((root/'ecp_extent/training_manifest.json').read_text())
zoom=json.loads((root/'zoom_only/training_manifest.json').read_text())
for arm, manifest in ((os.environ['ECP_CHECKPOINT'],ecp),(os.environ['ZOOM_CHECKPOINT'],zoom)):
    if sha(arm)!=manifest['checkpoint_sha256']: raise SystemExit(f'checkpoint hash mismatch: {arm}')
if sha(os.environ['CLIP_WEIGHTS'])!=ecp['clip_weights_sha256'] or ecp['clip_weights_sha256']!=zoom['clip_weights_sha256']:
    raise SystemExit('CLIP weights differ from the training manifests')
keys=('seed','zoom_aug_p','dataset','epochs','features_list','image_size','depth','n_ctx','t_n_ctx','batch_size','learning_rate','consistency_weight','ec_weight','ec_teacher_p','mvtec_meta_sha256','source_sha256')
if any(ecp[k]!=zoom[k] for k in keys): raise SystemExit('common settings differ between training manifests')
if ecp['extent_cond']!='extent' or zoom['extent_cond']!='none' or ecp['seed']!=111 or ecp['zoom_aug_p']!=0.5:
    raise SystemExit('training manifest configuration does not match EXP-012')
expected={'dataset':'mvtec','epochs':15,'features_list':[24],'image_size':518,'depth':9,'n_ctx':12,'t_n_ctx':4,'batch_size':8}
if any(ecp[k]!=v for k,v in expected.items()): raise SystemExit('training configuration differs from EXP-012')
if sha(data/'mvtec/meta.json')!=ecp['mvtec_meta_sha256']: raise SystemExit('MVTec training split changed')
datasets=(('ISIC','ISIC','skin'),('ClinicDB','CVC/CVC-ClinicDB','colon'),('ColonDB','CVC/CVC-ColonDB','colon'),('Kvasir','CVC/Kvasir','colon'),('Endo','EndoTect_2020_Segmentation_Test_Dataset','colon'),('TN3K','TN3K/Thyroid Dataset/tn3k','thyroid'))
for name,relative,cls in datasets:
    folder=(data/relative).resolve(); meta=folder/'meta.json'; record=inventory['datasets'][name]
    if str(folder)!=record['path'] or not meta.is_file() or sha(meta)!=record['meta_json_sha256']:
        raise SystemExit(f'{name}: dataset path or split differs from inventory')
    if len(json.loads(meta.read_text())['test'][cls])!=record['sample_count']:
        raise SystemExit(f'{name}: test sample count differs from inventory')
PY

FIXED=(--metrics pixel-level --sigma 4 --seed 111 --features_list 24 --image_size 518
       --depth 9 --n_ctx 12 --t_n_ctx 4 --distractor_suppress none
       --distractor_rerank none --export_pixel_metrics)
DATASETS=(ISIC ClinicDB ColonDB Kvasir Endo TN3K)
FOLDERS=(ISIC CVC/CVC-ClinicDB CVC/CVC-ColonDB CVC/Kvasir EndoTect_2020_Segmentation_Test_Dataset 'TN3K/Thyroid Dataset/tn3k')
MODES=(ISBI colon colon colon colon thyroid)
for i in "${!DATASETS[@]}"; do
  [[ -f "$DATA_ROOT/${FOLDERS[$i]}/meta.json" ]] || fail "missing dataset split: $DATA_ROOT/${FOLDERS[$i]}/meta.json"
done
for i in "${!DATASETS[@]}"; do
  for arm in ecp_extent zoom_only; do
    [[ ! -e "$OUT_ROOT/${DATASETS[$i]}/$arm" ]] || fail "output already exists: $OUT_ROOT/${DATASETS[$i]}/$arm"
  done
done
mkdir -p "$OUT_ROOT"
for i in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$i]}"
  data_path="$DATA_ROOT/${FOLDERS[$i]}"
  for arm in ecp_extent zoom_only; do
    out="$OUT_ROOT/$dataset/$arm"
    mkdir -p "$out"
    cp "$TRAIN_ROOT/$arm/training_manifest.json" "$out/"
    if [[ "$arm" == ecp_extent ]]; then
      checkpoint="$ECP_CHECKPOINT"
      arm_flags=(--ec_z_img -0.69 --ec_z_pix 1.6)
    else
      checkpoint="$ZOOM_CHECKPOINT"
      arm_flags=()
    fi
    cmd=("$PYTHON" "$REPO_ROOT/test.py" --data_path "$data_path" --dataset "${MODES[$i]}"
         --checkpoint_path "$checkpoint" --save_path "$out" "${FIXED[@]}" "${arm_flags[@]}")
    printf '%q ' "${cmd[@]}" > "$out/command.txt"
    printf '\n' >> "$out/command.txt"
    printf 'Evaluating EXP-012 %s / %s\n' "$dataset" "$arm"
    "${cmd[@]}" > "$out/stdout_stderr.log" 2>&1 || fail "$dataset $arm failed; preserve output and inspect its log"
    [[ -s "$out/pixel_level_metrics.json" && -s "$out/pixel_per_image_predictions.csv" && -s "$out/evaluation_metadata.json" ]] || fail "$dataset $arm artifacts are incomplete"
  done
done
printf 'EXP-012: all 12 evaluations completed under %s\n' "$OUT_ROOT"
