#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/home/ai3/NguyenND/AnomalyClipV2/data
COMMON=(--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4)
mkdir -p logs_exp042 results/EXP-042

train_arm() {
  local arm=$1 epochs=$2 name=$3 seed=$4
  CUDA_VISIBLE_DEVICES=0 python train.py --dataset mvtec --train_data_path "$DATA/mvtec" \
    --save_path "checkpoints/$name" "${COMMON[@]}" --batch_size 8 \
    --epoch "$epochs" --save_freq "$epochs" --seed "$seed" --zoom_aug_p 0.5 \
    --extent_cond none --exp042_cls_view "$arm" > "logs_exp042/train_${name}.log" 2>&1
}

eval_one() {
  local arm=$1 dataset=$2 path=$3 key=$4 level=$5 seed=$6
  local out="results/EXP-042/s${seed}/$arm/$key"
  if [ "$seed" = 111 ]; then out="results/EXP-042/$arm/$key"; fi
  local ckpt="checkpoints/exp042_${arm}_s${seed}/epoch_15.pth"
  local metric_file=pixel_level_metrics.json
  if [ "$level" = image-level ]; then metric_file=image_level_metrics.json; fi
  if [ -f "$out/$metric_file" ]; then return; fi
  if [ -d "$out" ]; then mv "$out" "${out}.partial.$(date +%s)"; fi
  local export_flag=--export_pixel_metrics
  if [ "$level" = image-level ]; then export_flag=--export_image_scores; fi
  CUDA_VISIBLE_DEVICES=0 python test.py --dataset "$dataset" --data_path "$path" \
    --save_path "$out" --checkpoint_path "$ckpt" "${COMMON[@]}" \
    --metrics "$level" --sigma 4 --seed "$seed" "$export_flag" > "logs_exp042/${arm}_s${seed}_${key}.log" 2>&1
}

eval_seed() {
  local seed=$1 arm=$2
  eval_one "$arm" colon "$DATA/CVC/Kvasir" Kvasir pixel-level "$seed"
  eval_one "$arm" colon "$DATA/CVC/CVC-ClinicDB" CVC-ClinicDB pixel-level "$seed"
  eval_one "$arm" colon "$DATA/CVC/CVC-ColonDB" CVC-ColonDB pixel-level "$seed"
  eval_one "$arm" ISBI "$DATA/ISIC" isic pixel-level "$seed"
  eval_one "$arm" colon "$DATA/EndoTect_2020_Segmentation_Test_Dataset" endo pixel-level "$seed"
  eval_one "$arm" thyroid "$DATA/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level "$seed"
  eval_one "$arm" brain "$DATA/HeadCT_anomaly_detection" HeadCT_anomaly_detection image-level "$seed"
  eval_one "$arm" brain "$DATA/BrainMRI" BrainMRI image-level "$seed"
  eval_one "$arm" brain "$DATA/br35" br35 image-level "$seed"
}

case "${1:-}" in
  check)
    python -m py_compile dataset.py train.py scripts/exp042_source_probe.py scripts/collect_exp042.py scripts/collect_exp042_replication.py
    bash -n scripts/run_exp042.sh
    python -m unittest scripts.test_exp042 scripts.test_exp042_replication -v ;;
  smoke)
    train_arm split 1 exp042_split_smoke 111
    test -f checkpoints/exp042_split_smoke/epoch_1.pth
    echo 'EXP-042 split training smoke passed' ;;
  train)
    for arm in split sham; do
      if [ ! -f "checkpoints/exp042_${arm}_s111/epoch_15.pth" ]; then
        train_arm "$arm" 15 "exp042_${arm}_s111" 111
      fi
      test -f "checkpoints/exp042_${arm}_s111/epoch_15.pth"
    done ;;
  eval)
    for arm in split sham; do
      test -f "checkpoints/exp042_${arm}_s111/epoch_15.pth"
      eval_seed 111 "$arm"
    done ;;
  replicate)
    bash scripts/run_exp042.sh check
    python -c 'import json; assert json.load(open("results/EXP-042/report.json"))["pilot_gate"] == "ADVANCE_TO_REPLICATION", "EXP-042 pilot gate did not pass"'
    for seed in 222 333; do
      for arm in split sham; do
        name="exp042_${arm}_s${seed}"
        if [ ! -f "checkpoints/$name/epoch_15.pth" ]; then
          train_arm "$arm" 15 "$name" "$seed"
        fi
        test -f "checkpoints/$name/epoch_15.pth"
        eval_seed "$seed" "$arm"
      done
    done
    python scripts/collect_exp042_replication.py ;;
  collect_replication)
    python scripts/collect_exp042_replication.py ;;
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
  *) echo 'usage: bash scripts/run_exp042.sh check|smoke|train|eval|source|collect|full|replicate|collect_replication'; exit 2 ;;
esac
