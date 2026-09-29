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

train () {  # $1 = name, $2 = extent_cond, $3 = epochs, $4 = seed (default 111)
  CUDA_VISIBLE_DEVICES=$DEV python train.py --dataset mvtec --train_data_path $MVTEC \
    --save_path ./checkpoints/$1/ $COMMON --batch_size 8 --print_freq 1 \
    --epoch $3 --save_freq 1 --seed ${4:-111} --zoom_aug_p $ZOOM --extent_cond $2
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
  # control: plain zoom augmentation on EVERY anomalous image (no module). If it matches ECP with const z=1.2, ECP adds nothing.
  train_zoom1)   ZOOM=1.0; train zoom_p1 none 15 ;;
  train_zoom1_seed) ZOOM=1.0; train zoom_p1_s$2 none 15 $2 ;;   # bash run_ecp.sh train_zoom1_seed 222
  test_extent)   test_all ecp_extent ecp_extent "" ;;
  test_global)   test_all ecp_global ecp_global "" ;;
  test_oracle)   test_all ecp_extent ecp_extent_oracle "--ec_oracle" ;;
  # more seeds for the method:  bash run_ecp.sh train_seed 222   then   bash run_ecp.sh test_seed 222
  train_seed)    train ecp_extent_s$2 extent 15 $2 ;;
  test_seed)     test_all ecp_extent_s$2 ecp_extent_s$2 "" ;;
  # smoothing check (the strong baseline):  bash run_ecp.sh test_sigma <checkpoint_name> 32
  test_sigma)    test_all $2 ${2}_sigma$3 "--sigma $3" ;;
  # fast per-image AUROC dump (no PRO) for the size-stratified comparison:  bash run_ecp.sh per_image <checkpoint_name> [sigma]
  # optional 4th arg = extra test flags (e.g. "--ec_oracle" or "--ec_const_z 1.0"), 5th = results tag suffix
  per_image)
    for D in CVC-ClinicDB Kvasir CVC-ColonDB; do
      CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset colon --data_path $CVC/$D \
        --save_path ./results/${2}${5:-}_pi/$D --checkpoint_path ./checkpoints/$2/epoch_15.pth \
        $COMMON --metrics ${METRICS:-pixel-level} --sigma ${3:-4} ${4:-}
    done ;;   # prints AUROC and PRO; set METRICS=pixel-auroc for the fast (no PRO) mode
  # held-out medical datasets (ISIC skin, Endo polyp, TN3K thyroid ultrasound):  bash run_ecp.sh test_heldout <checkpoint_name> [sigma]
  # optional 4th arg = extra flags (e.g. "--ec_oracle") and 5th = results tag suffix
  test_heldout)
    S=${3:-4}; A=/home/ai3/NguyenND/AnomalyClipV2/data; X=${4:-}; T=${5:-}
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset ISBI --data_path $A/ISIC \
      --save_path ./results/$2$T/isic_s$S --checkpoint_path ./checkpoints/$2/epoch_15.pth $COMMON --metrics pixel-level --sigma $S $X
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset colon --data_path $A/EndoTect_2020_Segmentation_Test_Dataset \
      --save_path ./results/$2$T/endo_s$S --checkpoint_path ./checkpoints/$2/epoch_15.pth $COMMON --metrics pixel-level --sigma $S $X
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset thyroid --data_path "$A/TN3K/Thyroid Dataset/tn3k" \
      --save_path ./results/$2$T/tn3k_s$S --checkpoint_path ./checkpoints/$2/epoch_15.pth $COMMON --metrics pixel-level --sigma $S $X ;;
  # image-level-only classification datasets (HeadCT, BrainMRI, Br35H): bash run_ecp.sh test_cls <checkpoint_name> [extra flags]
  test_cls)
    A=/home/ai3/NguyenND/AnomalyClipV2/data
    for DIR in HeadCT_anomaly_detection BrainMRI br35; do
      CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset brain --data_path $A/$DIR \
        --save_path ./results/$2/$DIR --checkpoint_path ./checkpoints/$2/epoch_15.pth \
        $COMMON --metrics image-level ${3:-}
    done ;;
  # spatial error decomposition (core cold / rim bleed / far false-positive hotspots): bash run_ecp.sh diag_cb <ckpt> [z_pix]
  diag_cb)
    A=/home/ai3/NguyenND/AnomalyClipV2/data; X=${3:+--ec_z_pix $3}; CK=./checkpoints/$2/epoch_15.pth
    for P in "colon $CVC/CVC-ClinicDB" "colon $CVC/Kvasir" "colon $CVC/CVC-ColonDB" "ISBI $A/ISIC" \
             "colon $A/EndoTect_2020_Segmentation_Test_Dataset" "thyroid $A/TN3K/Thyroid Dataset/tn3k"; do
      DS=${P%% *}; DP=${P#* }
      CUDA_VISIBLE_DEVICES=$DEV python analyze_core_boundary.py --dataset $DS --data_path "$DP" \
        --checkpoint_path $CK --out_csv core_boundary_$2_$(basename "$DP").csv $X
    done ;;
  # industrial check (standard protocol: MVTec-trained model tested on VisA): bash run_ecp.sh test_visa <ckpt> [flags]
  test_visa)
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset visa --data_path /home/ai3/NguyenND/AnomalyClipV2/data/visa \
      --save_path ./results/$2/visa --checkpoint_path ./checkpoints/$2/epoch_15.pth \
      $COMMON --metrics image-pixel-level ${3:-} ;;
  # clean held-out check (never used for any design decision): bash run_ecp.sh test_covid <checkpoint_name> [extra flags]
  test_covid)
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset Chest \
      --data_path /home/ai3/NguyenND/AnomalyClipV2/data/Covid \
      --save_path ./results/$2/covid --checkpoint_path ./checkpoints/$2/epoch_15.pth \
      $COMMON --metrics image-level ${3:-} ;;
  # ONE fixed config for every dataset: image head z=$ZI, pixel head z=$ZP.  bash run_ecp.sh fixed [ZI] [ZP]
  fixed)
    ZI=${2:--0.69}; ZP=${3:-1.6}; A=/home/ai3/NguyenND/AnomalyClipV2/data; F="--ec_z_img $ZI --ec_z_pix $ZP"
    CKP=./checkpoints/ecp_extent/epoch_15.pth; OUT=./results/ecp_fixed_i${ZI}_p${ZP}
    for D in CVC-ClinicDB Kvasir CVC-ColonDB; do
      CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset colon --data_path $CVC/$D --save_path $OUT/$D \
        --checkpoint_path $CKP $COMMON --metrics pixel-level $F
    done
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset ISBI --data_path $A/ISIC --save_path $OUT/isic \
      --checkpoint_path $CKP $COMMON --metrics pixel-level $F
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset colon --data_path $A/EndoTect_2020_Segmentation_Test_Dataset \
      --save_path $OUT/endo --checkpoint_path $CKP $COMMON --metrics pixel-level $F
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset thyroid --data_path "$A/TN3K/Thyroid Dataset/tn3k" \
      --save_path $OUT/tn3k --checkpoint_path $CKP $COMMON --metrics pixel-level $F
    for DIR in HeadCT_anomaly_detection BrainMRI br35; do
      CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset brain --data_path $A/$DIR --save_path $OUT/$DIR \
        --checkpoint_path $CKP $COMMON --metrics image-level $F
    done ;;
  # domain-level z calibration on unlabeled images (no labels used for the decision): bash run_ecp.sh calibrate
  calibrate)
    A=/home/ai3/NguyenND/AnomalyClipV2/data; CK=./checkpoints/ecp_extent/epoch_15.pth
    python calibrate_z_domain.py --dataset colon --data_path $A/CVC/CVC-ClinicDB --checkpoint_path $CK
    python calibrate_z_domain.py --dataset colon --data_path $A/CVC/Kvasir --checkpoint_path $CK
    python calibrate_z_domain.py --dataset colon --data_path $A/CVC/CVC-ColonDB --checkpoint_path $CK
    python calibrate_z_domain.py --dataset ISBI --data_path $A/ISIC --checkpoint_path $CK
    python calibrate_z_domain.py --dataset colon --data_path $A/EndoTect_2020_Segmentation_Test_Dataset --checkpoint_path $CK
    python calibrate_z_domain.py --dataset thyroid --data_path "$A/TN3K/Thyroid Dataset/tn3k" --checkpoint_path $CK
    python calibrate_z_domain.py --dataset brain --data_path $A/HeadCT_anomaly_detection --checkpoint_path $CK
    python calibrate_z_domain.py --dataset brain --data_path $A/BrainMRI --checkpoint_path $CK
    python calibrate_z_domain.py --dataset brain --data_path $A/br35 --checkpoint_path $CK ;;
  *) echo "usage: bash run_ecp.sh {smoke|train_extent|train_global|test_extent|test_global|test_oracle|train_seed N|test_seed N|test_sigma CKPT SIGMA|test_heldout CKPT [SIGMA]|test_cls CKPT [FLAGS]|calibrate}" ;;
esac
