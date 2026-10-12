#!/usr/bin/env bash
# EXP-038 pilot: source-trained static prompts with ECP as an auxiliary task.
# Run one phase at a time on the GPU server: smoke, train, eval.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-/home/ai3/NguyenND/AnomalyClipV2/data}
GPU=${GPU:-0}
SEED=111
NAME=exp038_static_aux_s${SEED}
CKPT=./checkpoints/${NAME}/epoch_15.pth
OUT=./results/EXP-038/${NAME}
COMMON=(--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4)

train_run() {
  local epochs=$1 save_dir=$2
  CUDA_VISIBLE_DEVICES="$GPU" python train.py --dataset mvtec --train_data_path "$DATA_ROOT/mvtec" \
    --save_path "$save_dir" "${COMMON[@]}" --batch_size 8 --print_freq 1 \
    --epoch "$epochs" --save_freq 1 --seed "$SEED" --zoom_aug_p 0.5 \
    --extent_cond extent --ec_static_aux
}

evaluate() {
  local mode=$1 path=$2 tag=$3 metric=$4
  local target="$OUT/$tag"
  if [ -f "$target/evaluation_metadata.json" ]; then
    echo "Already evaluated: $target"
    return
  fi
  if [ -e "$target" ]; then
    echo "Incomplete result path exists; inspect it before rerunning: $target" >&2
    exit 1
  fi
  local export_arg=()
  if [ "$metric" = pixel-level ]; then export_arg=(--export_pixel_metrics); fi
  if [ "$metric" = image-level ]; then export_arg=(--export_image_scores); fi
  CUDA_VISIBLE_DEVICES="$GPU" python test.py --dataset "$mode" --data_path "$path" \
    --save_path "$target" --checkpoint_path "$CKPT" "${COMMON[@]}" \
    --metrics "$metric" --sigma 4 "${export_arg[@]}"
}

case "${1:-}" in
  smoke)
    [ ! -e ./checkpoints/exp038_static_aux_smoke ] || { echo 'Smoke path exists' >&2; exit 1; }
    train_run 1 ./checkpoints/exp038_static_aux_smoke
    ;;
  train)
    [ ! -e "./checkpoints/${NAME}" ] || { echo "Checkpoint path exists: ./checkpoints/${NAME}" >&2; exit 1; }
    train_run 15 "./checkpoints/${NAME}"
    ;;
  eval)
    [ -f "$CKPT" ] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
    evaluate colon "$DATA_ROOT/CVC/CVC-ClinicDB" CVC-ClinicDB pixel-level
    evaluate colon "$DATA_ROOT/CVC/Kvasir" Kvasir pixel-level
    evaluate colon "$DATA_ROOT/CVC/CVC-ColonDB" CVC-ColonDB pixel-level
    evaluate ISBI "$DATA_ROOT/ISIC" isic pixel-level
    evaluate colon "$DATA_ROOT/EndoTect_2020_Segmentation_Test_Dataset" endo pixel-level
    evaluate thyroid "$DATA_ROOT/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level
    evaluate brain "$DATA_ROOT/HeadCT_anomaly_detection" HeadCT_anomaly_detection image-level
    evaluate brain "$DATA_ROOT/BrainMRI" BrainMRI image-level
    evaluate brain "$DATA_ROOT/br35" br35 image-level
    ;;
  *) echo 'usage: bash scripts/run_exp038.sh smoke|train|eval' >&2; exit 2 ;;
esac
