#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/home/ai3/NguyenND/AnomalyClipV2/data
COMMON=(--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4)
mkdir -p logs_exp042 results/EXP-042

train_arm() {
  local arm=$1 epochs=$2 name=$3
  CUDA_VISIBLE_DEVICES=0 python train.py --dataset mvtec --train_data_path "$DATA/mvtec" \
    --save_path "checkpoints/$name" "${COMMON[@]}" --batch_size 8 \
    --epoch "$epochs" --save_freq "$epochs" --seed 111 --zoom_aug_p 0.5 \
    --extent_cond none --exp042_cls_view "$arm" > "logs_exp042/train_${name}.log" 2>&1
}

eval_one() {
  local arm=$1 dataset=$2 path=$3 key=$4 level=$5
  local out="results/EXP-042/$arm/$key"
  local ckpt="checkpoints/exp042_${arm}_s111/epoch_15.pth"
  local metric_file=pixel_level_metrics.json
  if [ "$level" = image-level ]; then metric_file=image_level_metrics.json; fi
  if [ -f "$out/$metric_file" ]; then return; fi
  if [ -d "$out" ]; then mv "$out" "${out}.partial.$(date +%s)"; fi
  local export_flag=--export_pixel_metrics
  if [ "$level" = image-level ]; then export_flag=--export_image_scores; fi
  CUDA_VISIBLE_DEVICES=0 python test.py --dataset "$dataset" --data_path "$path" \
    --save_path "$out" --checkpoint_path "$ckpt" "${COMMON[@]}" \
    --metrics "$level" --sigma 4 "$export_flag" > "logs_exp042/${arm}_${key}.log" 2>&1
}

case "${1:-}" in
  check)
    python -m py_compile dataset.py train.py scripts/exp042_source_probe.py scripts/collect_exp042.py
    bash -n scripts/run_exp042.sh
    python -m unittest scripts.test_exp042 -v ;;
  smoke)
    train_arm split 1 exp042_split_smoke
    test -f checkpoints/exp042_split_smoke/epoch_1.pth
    echo 'EXP-042 split training smoke passed' ;;
  train)
    for arm in split sham; do
      if [ ! -f "checkpoints/exp042_${arm}_s111/epoch_15.pth" ]; then
        train_arm "$arm" 15 "exp042_${arm}_s111"
      fi
      test -f "checkpoints/exp042_${arm}_s111/epoch_15.pth"
    done ;;
  eval)
    for arm in split sham; do
      test -f "checkpoints/exp042_${arm}_s111/epoch_15.pth"
      eval_one "$arm" colon "$DATA/CVC/Kvasir" Kvasir pixel-level
      eval_one "$arm" colon "$DATA/CVC/CVC-ClinicDB" CVC-ClinicDB pixel-level
      eval_one "$arm" colon "$DATA/CVC/CVC-ColonDB" CVC-ColonDB pixel-level
      eval_one "$arm" ISBI "$DATA/ISIC" isic pixel-level
      eval_one "$arm" colon "$DATA/EndoTect_2020_Segmentation_Test_Dataset" endo pixel-level
      eval_one "$arm" thyroid "$DATA/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level
      eval_one "$arm" brain "$DATA/HeadCT_anomaly_detection" HeadCT_anomaly_detection image-level
      eval_one "$arm" brain "$DATA/BrainMRI" BrainMRI image-level
      eval_one "$arm" brain "$DATA/br35" br35 image-level
    done ;;
  source)
    CUDA_VISIBLE_DEVICES=0 python scripts/exp042_source_probe.py ;;
  collect)
    python scripts/collect_exp042.py ;;
  full)
    bash scripts/run_exp042.sh check
    bash scripts/run_exp042.sh smoke
    bash scripts/run_exp042.sh train
    bash scripts/run_exp042.sh eval
    bash scripts/run_exp042.sh source
    bash scripts/run_exp042.sh collect ;;
  *) echo 'usage: bash scripts/run_exp042.sh check|smoke|train|eval|source|collect|full'; exit 2 ;;
esac
