#!/usr/bin/env bash
# EXP-011 full server evaluation. Set PYTHON, CHECKPOINT, DATA_ROOT, CLIP_WEIGHTS
# to the same paths used for preflight. Run preflight first, then:
# bash scripts/run_exp011.sh
# Exactly six test.py evaluations; no training or alternate prompt settings.
set -Eeuo pipefail

PYTHON="${PYTHON:-}"
CHECKPOINT="${CHECKPOINT:-}"
DATA_ROOT="${DATA_ROOT:-}"
CLIP_WEIGHTS="${CLIP_WEIGHTS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$REPO_ROOT/results/EXP-011"
fail() { printf 'EXP-011 run error: %s\n' "$*" >&2; exit 1; }
[[ $# -eq 0 ]] || fail 'usage: bash scripts/run_exp011.sh'
[[ -n "$PYTHON" ]] || fail 'set PYTHON to the server environment interpreter'
command -v "$PYTHON" >/dev/null 2>&1 || fail "PYTHON is not executable: $PYTHON"
PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')" || fail 'cannot start PYTHON'
[[ -n "$CHECKPOINT" && -n "$DATA_ROOT" && -n "$CLIP_WEIGHTS" ]] ||
  fail 'set CHECKPOINT, DATA_ROOT, and CLIP_WEIGHTS to server paths'
CHECKPOINT="$("$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT")"
DATA_ROOT="$("$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$DATA_ROOT")"
CLIP_WEIGHTS="$("$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CLIP_WEIGHTS")"
export PYTHON CHECKPOINT DATA_ROOT CLIP_WEIGHTS

# This checks environment, checkpoint mode/hash, CLIP weights, data manifests,
# supported flags, source hashes, and the successful smoke-test marker.
bash "$REPO_ROOT/scripts/preflight_exp011.sh" --verify-ready ||
  fail 'preflight is missing or stale; run bash scripts/preflight_exp011.sh first'
cd "$REPO_ROOT"

FIXED=(--dataset brain --metrics image-level --ec_z_pix 1.6 --sigma 4 --seed 111
       --features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4
       --distractor_suppress none --distractor_rerank none --export_image_scores)
DATASETS=(HeadCT BrainMRI Br35H)
FOLDERS=(HeadCT_anomaly_detection BrainMRI br35)
ARMS=(zimg_1.6 zimg_-0.69)
ZS=(1.6 -0.69)

# Never append to or overwrite results from an earlier attempt.
for dataset in "${DATASETS[@]}"; do
  for arm in "${ARMS[@]}"; do
    [[ ! -e "$OUT_ROOT/$dataset/$arm" ]] ||
      fail "output already exists: $OUT_ROOT/$dataset/$arm (move it before rerunning)"
  done
done
[[ -w "$OUT_ROOT" ]] || fail "output root is not writable: $OUT_ROOT"

for i in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$i]}"
  data_path="$DATA_ROOT/${FOLDERS[$i]}"
  for j in "${!ARMS[@]}"; do
    arm="${ARMS[$j]}"
    out="$OUT_ROOT/$dataset/$arm"
    mkdir -p "$out" || fail "cannot create output: $out"
    cmd=("$PYTHON" "$REPO_ROOT/test.py" --data_path "$data_path"
         --checkpoint_path "$CHECKPOINT" --save_path "$out" "${FIXED[@]}"
         --ec_z_img "${ZS[$j]}")
    printf '%q ' "${cmd[@]}" > "$out/command.txt"
    printf '\n' >> "$out/command.txt"
    printf 'Evaluating %s %s (log: %s)\n' "$dataset" "$arm" "$out/stdout_stderr.log"
    (cd "$REPO_ROOT" && "${cmd[@]}") > "$out/stdout_stderr.log" 2>&1 ||
      fail "$dataset $arm failed; see $out/stdout_stderr.log"
  done
done

export EXP011_OUT_ROOT="$OUT_ROOT"
"$PYTHON" - <<'PY' || fail 'post-run artifact validation failed; preserve outputs for inspection'
import csv
import json
import math
import os
from pathlib import Path
from metrics import image_level_metrics

root = Path(os.environ['EXP011_OUT_ROOT'])
ready = json.loads((root / 'preflight/ready.json').read_text())
for dataset in ('HeadCT', 'BrainMRI', 'Br35H'):
    paired = {}
    for arm, z in (('zimg_1.6', 1.6), ('zimg_-0.69', -0.69)):
        folder = root / dataset / arm
        with (folder / 'image_level_predictions.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        metadata = json.loads((folder / 'evaluation_metadata.json').read_text())
        metrics = json.loads((folder / 'image_level_metrics.json').read_text())
        expected = ready['datasets'][dataset]
        if (len(rows) != expected['sample_count'] or metadata['sample_count'] != len(rows) or
            metadata['positive_count'] != expected['positive_count'] or
            metadata['negative_count'] != expected['negative_count']):
            raise SystemExit(f'{dataset} {arm}: sample/class counts do not match preflight')
        if (metadata['checkpoint_sha256'] != ready['checkpoint_sha256'] or
            metadata['checkpoint_extent_cond'] != 'extent' or
            metadata['pretrained_clip_weights_sha256'] != ready['clip_weights_sha256'] or
            metadata['meta_json_sha256'] != expected['meta_sha256'] or
            metadata['evaluation_source_sha256'] != {
                key: value for key, value in ready['source_sha256'].items()
                if key not in ('scripts/preflight_exp011.sh', 'scripts/run_exp011.sh')
            }):
            raise SystemExit(f'{dataset} {arm}: checkpoint, CLIP, code, or split changed')
        args = metadata['arguments']
        fixed = ready['fixed_settings']
        if any(args.get(key) != value for key, value in fixed.items()) or args.get('ec_z_img') != z or not args.get('export_image_scores'):
            raise SystemExit(f'{dataset} {arm}: evaluation arguments differ from EXP-011')
        if Path(metadata['data_path']).resolve() != Path(expected['path']):
            raise SystemExit(f'{dataset} {arm}: dataset path changed')
        sample_map = {}
        labels, scores = [], []
        for row in rows:
            sample_id = row['sample_id']
            label = int(row['ground_truth_image_label'])
            score = float(row['image_anomaly_score'])
            if (sample_id in sample_map or label not in (0, 1) or
                not math.isfinite(score) or row['dataset'] != Path(metadata['data_path']).name):
                raise SystemExit(f'{dataset} {arm}: invalid or duplicated prediction row')
            sample_map[sample_id] = label
            labels.append(label)
            scores.append(score)
        if not all(math.isfinite(value) for value in metrics['brain'].values()):
            raise SystemExit(f'{dataset} {arm}: non-finite image metrics')
        for key, metric in (('image_auroc', 'image-auroc'), ('image_ap', 'image-ap')):
            recalculated = image_level_metrics({'brain': {'gt_sp': labels, 'pr_sp': scores}}, 'brain', metric)
            if not math.isclose(recalculated, metrics['brain'][key], rel_tol=0, abs_tol=1e-12):
                raise SystemExit(f'{dataset} {arm}: {key} does not match exported predictions')
        paired[arm] = sample_map
    if paired['zimg_1.6'] != paired['zimg_-0.69']:
        raise SystemExit(f'{dataset}: arm sample IDs or labels differ')
PY

# Detect input or code changes during the six evaluations.
bash "$REPO_ROOT/scripts/preflight_exp011.sh" --verify-ready ||
  fail 'inputs changed during the run; preserve outputs for inspection'
printf 'EXP-011 six evaluations completed; paired outputs are under %s\n' "$OUT_ROOT"
