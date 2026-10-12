#!/usr/bin/env bash
# EXP-039: MVTec source selection, then frozen-rule medical image evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-/home/ai3/NguyenND/AnomalyClipV2/data}
CKPT=${CKPT:-./checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}
GPU=${GPU:-0}
OUT=./results/EXP-039
COMMON=(--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4)

evaluate() {
  local mode=$1 data=$2 name=$3
  local target="$OUT/$name"
  if [ -f "$target/evaluation_metadata.json" ] && [ -f "$target/exp039_pool_scores.csv" ]; then
    echo "Already evaluated: $target"
    return
  fi
  if [ -e "$target" ]; then
    echo "Incomplete result path exists; inspect it before rerunning: $target" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python test.py --dataset "$mode" --data_path "$data" \
    --save_path "$target" --checkpoint_path "$CKPT" "${COMMON[@]}" \
    --metrics image-level --export_image_scores --exp039_pool_scores \
    --ec_z_img -0.69 --ec_z_pix 1.6 --sigma 4 --seed 111
}

case "${1:-}" in
  source)
    [ -f "$CKPT" ] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
    evaluate mvtec "$DATA_ROOT/mvtec" source_mvtec
    ;;
  select)
    python scripts/collect_exp039.py select --root "$OUT"
    ;;
  target)
    [ -f "$CKPT" ] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
    [ -f "$OUT/selection.json" ] || { echo 'Run source and select first' >&2; exit 1; }
    python scripts/collect_exp039.py select --root "$OUT" >/dev/null
    evaluate brain "$DATA_ROOT/HeadCT_anomaly_detection" HeadCT_anomaly_detection
    evaluate brain "$DATA_ROOT/BrainMRI" BrainMRI
    evaluate brain "$DATA_ROOT/br35" br35
    ;;
  report)
    python scripts/collect_exp039.py report --root "$OUT"
    ;;
  *) echo 'usage: bash scripts/run_exp039.sh source|select|target|report' >&2; exit 2 ;;
esac
