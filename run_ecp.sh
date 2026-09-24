#!/bin/bash
# Extent-conditioned prompt (ECP) experiments. Run ONE step at a time:  bash run_ecp.sh <step>
#   smoke    : 1-epoch check that training runs (extent_abs_err in the log must be non-zero)
#   train_extent | train_global : 15-epoch training (~100 min each), seed 111, zoom_aug_p 0.5 (same as zoom_mvtec)
#   test_extent | test_global   : pixel-auroc + PRO on ClinicDB / Kvasir / ColonDB
#   test_oracle : diagnostic upper bound, conditions the extent model on the GROUND-TRUTH lesion extent
# Baselines to compare against (already trained): ./checkpoints/zoom_mvtec (zoom-only), ./checkpoints/ctrl_mvtec

DEV=0
ZOOM=0.5
MVTEC=/home/ai3/NguyenND/AnomalyClipV2/data/mvtec
CVC=/home/ai3/NguyenND/AnomalyClipV2/data/CVC
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"

train () {  # $1 = name, $2 = extent_cond, $3 = epochs
  CUDA_VISIBLE_DEVICES=$DEV python train.py --dataset mvtec --train_data_path $MVTEC \
    --save_path ./checkpoints/$1/ $COMMON --batch_size 8 --print_freq 1 \
    --epoch $3 --save_freq 1 --seed 111 --zoom_aug_p $ZOOM --extent_cond $2
}

test_all () {  # $1 = checkpoint name, $2 = results tag, $3 = extra test flags
  for D in CVC-ClinicDB Kvasir CVC-ColonDB; do
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset colon --data_path $CVC/$D \
      --save_path ./results/$2/$D --checkpoint_path ./checkpoints/$1/epoch_15.pth \
      $COMMON --metrics pixel-level $3
  done
}

case "$1" in
  smoke)         train ecp_smoke extent 1 ;;
  train_extent)  train ecp_extent extent 15 ;;
  train_global)  train ecp_global global 15 ;;
  test_extent)   test_all ecp_extent ecp_extent "" ;;
  test_global)   test_all ecp_global ecp_global "" ;;
  test_oracle)   test_all ecp_extent ecp_extent_oracle "--ec_oracle" ;;
  *) echo "usage: bash run_ecp.sh {smoke|train_extent|train_global|test_extent|test_global|test_oracle}" ;;
esac
