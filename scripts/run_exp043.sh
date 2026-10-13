#!/usr/bin/env bash
# Direct ECP+split seed-111 pilot. Stops after image evaluation if it fails the registered gate.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/home/ai3/NguyenND/AnomalyClipV2/data
COMMON=(--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4)
BASE=checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth
SPLIT=checkpoints/exp043_ecp_split_s111/epoch_15.pth
mkdir -p logs_exp043 results/EXP-043

train_split() {
  local epochs=$1 name=$2
  CUDA_VISIBLE_DEVICES=0 python train.py --dataset mvtec --train_data_path "$DATA/mvtec" \
    --save_path "checkpoints/$name" "${COMMON[@]}" --batch_size 8 --epoch "$epochs" \
    --save_freq "$epochs" --seed 111 --learning_rate 0.001 --zoom_aug_p 0.5 \
    --extent_cond extent --ec_teacher_p 0.5 --ec_weight 1.0 --ec_z_shuffle none \
    --exp043_ecp_cls_view split > "logs_exp043/train_${name}.log" 2>&1
}

evaluate() {
  local arm=$1 dataset=$2 path=$3 key=$4 level=$5
  local checkpoint=$BASE
  if [ "$arm" = ecp_split ]; then checkpoint=$SPLIT; fi
  local output="results/EXP-043/$arm/$key"
  local metric_file=pixel_level_metrics.json flag=--export_pixel_metrics
  if [ "$level" = image-level ]; then
    metric_file=image_level_metrics.json
    flag=--export_image_scores
  fi
  if [ -f "$output/$metric_file" ] && [ -f "$output/evaluation_metadata.json" ]; then return; fi
  if [ -d "$output" ]; then mv "$output" "${output}.partial.$(date +%s)"; fi
  CUDA_VISIBLE_DEVICES=0 python test.py --dataset "$dataset" --data_path "$path" \
    --save_path "$output" --checkpoint_path "$checkpoint" "${COMMON[@]}" \
    --metrics "$level" --ec_z_img -0.69 --ec_z_pix 1.6 --sigma 4 --seed 111 "$flag" \
    > "logs_exp043/${arm}_${key}.log" 2>&1
}

image_eval() {
  local arm=$1
  evaluate "$arm" brain "$DATA/HeadCT_anomaly_detection" HeadCT_anomaly_detection image-level
  evaluate "$arm" brain "$DATA/BrainMRI" BrainMRI image-level
  evaluate "$arm" brain "$DATA/br35" br35 image-level
}

pixel_eval() {
  local arm=$1
  evaluate "$arm" colon "$DATA/CVC/Kvasir" Kvasir pixel-level
  evaluate "$arm" colon "$DATA/CVC/CVC-ClinicDB" CVC-ClinicDB pixel-level
  evaluate "$arm" colon "$DATA/CVC/CVC-ColonDB" CVC-ColonDB pixel-level
  evaluate "$arm" ISBI "$DATA/ISIC" isic pixel-level
  evaluate "$arm" colon "$DATA/EndoTect_2020_Segmentation_Test_Dataset" endo pixel-level
  evaluate "$arm" thyroid "$DATA/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level
}

case "${1:-}" in
  check)
    python -m py_compile train.py dataset.py scripts/collect_exp043.py scripts/test_exp043.py
    bash -n scripts/run_exp043.sh
    python -m unittest scripts.test_exp043 -v ;;
  smoke)
    train_split 1 exp043_ecp_split_smoke
    test -f checkpoints/exp043_ecp_split_smoke/epoch_1.pth ;;
  image)
    test -f "$BASE"
    python -c 'import hashlib,sys; p=sys.argv[1]; h=hashlib.sha256(open(p,"rb").read()).hexdigest(); assert h=="7278af957c4087d7c00e7495e4cba3a65c253b5d0c9646b36434b951e3df0e7c", "wrong matched ECP checkpoint hash"' "$BASE"
    image_eval ecp_baseline
    if [ ! -f "$SPLIT" ]; then train_split 15 exp043_ecp_split_s111; fi
    test -f "$SPLIT"
    image_eval ecp_split
    python scripts/collect_exp043.py ;;
  pixel)
    python -c 'import json; assert json.load(open("results/EXP-043/report.json"))["status"] == "CONTINUE_PIXEL", "EXP-043 image gate did not pass"'
    pixel_eval ecp_baseline
    pixel_eval ecp_split
    python scripts/collect_exp043.py ;;
  full)
    bash scripts/run_exp043.sh check
    bash scripts/run_exp043.sh smoke
    bash scripts/run_exp043.sh image
    if python -c 'import json,sys; sys.exit(json.load(open("results/EXP-043/report.json"))["status"] != "CONTINUE_PIXEL")'; then
      bash scripts/run_exp043.sh pixel
    fi ;;
  collect) python scripts/collect_exp043.py ;;
  *) echo 'usage: bash scripts/run_exp043.sh check|smoke|image|pixel|full|collect'; exit 2 ;;
esac
