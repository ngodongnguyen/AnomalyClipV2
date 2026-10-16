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
VISA=/home/ai3/NguyenND/AnomalyClipV2/data/visa
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"

train () {  # $1 = name, $2 = extent_cond, $3 = epochs, $4 = seed (default 111)
  CUDA_VISIBLE_DEVICES=$DEV python train.py --dataset mvtec --train_data_path $MVTEC \
    --save_path ./checkpoints/$1/ $COMMON --batch_size 8 --print_freq 1 \
    --epoch $3 --save_freq 1 --seed ${4:-111} --zoom_aug_p $ZOOM --extent_cond $2 ${TRAIN_EXTRA:-}
}

train_visa () {  # VisA as the auxiliary source (protocol of CoPS/MRAD/MoECLIP/VisualAD). $1 = name, $2 = extent_cond, $3 = epochs, $4 = seed
  CUDA_VISIBLE_DEVICES=$DEV python train.py --dataset visa --train_data_path $VISA \
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
  # EXP-020: VisA-trained controls and ECP (same seed/epochs/COMMON as the MVTec runs). One at a time, ~100 min each.
  #   bash run_ecp.sh train_visa_ctrl | train_visa_zoom | train_visa_ecp   [seed -> checkpoints/<name>_s<seed>]
  smoke_visa)      ZOOM=0.5; train_visa ecp_visa_smoke extent 1 ;;   # 1-epoch check that VisA training/zoom-aug/extent head run
  train_visa_ctrl) ZOOM=0;   train_visa ctrl_visa${2:+_s$2} none 15 $2 ;;
  train_visa_zoom) ZOOM=0.5; train_visa zoom_visa${2:+_s$2} none 15 $2 ;;
  train_visa_ecp)  ZOOM=0.5; train_visa ecp_visa${2:+_s$2} extent 15 $2 ;;
  # EXP-022: control "learned constant prompt shift per head" (no extent axis), same recipe/seed/zoom as the matched ECP. ~100 min.
  smoke_dual) ZOOM=0.5; train ecp_dual_smoke dual 1 ;;
  train_dual) ZOOM=0.5; train ecp_dual dual 15 ;;
  # EXP-024: matched three-seed arms (same recipe as scripts/train_exp012_matched_pair.sh, new checkpoint names so the older seed-222/333 checkpoints are NOT overwritten).
  #   bash run_ecp.sh train_m_ecp 222 | train_m_zoom 222 | train_m_dual 222   (seed 111 already exists: EXP-012-matched/{ecp_extent,zoom_only}/checkpoints and ecp_dual)
  train_m_ecp)  ZOOM=0.5; train matched_ecp_s$2  extent 15 $2 ;;
  train_m_zoom) ZOOM=0.5; train matched_zoom_s$2 none   15 $2 ;;
  train_m_dual) ZOOM=0.5; train matched_dual_s$2 dual   15 $2 ;;
  # EXP-028 controls (same recipe as train_m_ecp): SHUF = extent labels rotated among the anomalous images of a batch; LATENT = no extent supervision, no teacher forcing
  # EXP-030: matched no-zoom control = original AnomalyCLIP prompts (no zoom augmentation, no ECP), same recipe/seeds as train_m_ecp
  train_m_ctrl)   ZOOM=0; train matched_ctrl_s$2 none 15 $2 ;;
  smoke_ctrl)     ZOOM=0; train ctrl_smoke none 1 ;;
  train_m_shuf)   ZOOM=0.5; TRAIN_EXTRA="--ec_z_shuffle batch" train matched_shuf_s$2 extent 15 $2 ;;
  train_m_latent) ZOOM=0.5; TRAIN_EXTRA="--ec_weight 0 --ec_teacher_p 0" train matched_latent_s$2 extent 15 $2 ;;
  smoke_shuf)     ZOOM=0.5; TRAIN_EXTRA="--ec_z_shuffle batch" train shuf_smoke extent 1 ;;
  # EXP-054: teacher-forcing probability ablation (ECP recipe, only --ec_teacher_p changes).  bash run_ecp.sh train_m_tp SEED P   -> checkpoints/matched_tp<P*100>_s<SEED>
  train_m_tp)     ZOOM=0.5; TRAIN_EXTRA="--ec_teacher_p $3" train matched_tp$(python3 -c "print(int(round($3*100)))")_s$2 extent 15 $2 ;;
  smoke_latent)   ZOOM=0.5; TRAIN_EXTRA="--ec_weight 0 --ec_teacher_p 0" train latent_smoke extent 1 ;;
  # fixed per-head config (image z=ZI, pixel z=ZP) for ANY checkpoint; z flags are ignored by checkpoints without a conditioner.
  #   bash run_ecp.sh fixed_ckpt <checkpoint_name> [ZI] [ZP]      -> ./results/<ckpt>_fixed
  fixed_ckpt)
    ZI=${3:--0.69}; ZP=${4:-1.6}; A=/home/ai3/NguyenND/AnomalyClipV2/data; F="--ec_z_img $ZI --ec_z_pix $ZP ${5:-}"
    CKP=./checkpoints/$2/epoch_15.pth; OUT=./results/${2}${6:-}_fixed
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
    # set SKIP_IMG=1 to skip the three image-level sets: saves ~25 min per run
    for DIR in HeadCT_anomaly_detection BrainMRI br35; do
      [ -n "${SKIP_IMG:-}" ] && continue
      CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset brain --data_path $A/$DIR --save_path $OUT/$DIR \
        --checkpoint_path $CKP $COMMON --metrics image-level $F
    done ;;
  # MVTec-AD with a VisA-trained checkpoint (standard AnomalyCLIP protocol for MVTec): bash run_ecp.sh test_mvtec_visa <ckpt> [flags]
  test_mvtec_visa)
    CUDA_VISIBLE_DEVICES=$DEV python test.py --dataset mvtec --data_path $MVTEC \
      --save_path ./results/$2/mvtec --checkpoint_path ./checkpoints/$2/epoch_15.pth \
      $COMMON --metrics image-pixel-level ${3:-} ;;
  *) echo "usage: bash run_ecp.sh {smoke|train_extent|train_global|test_extent|test_global|test_oracle|train_seed N|test_seed N|test_sigma CKPT SIGMA|test_heldout CKPT [SIGMA]|smoke_dual|train_dual|train_m_ecp SEED|train_m_zoom SEED|train_m_dual SEED|train_m_ctrl SEED|smoke_ctrl|train_m_shuf SEED|train_m_latent SEED|smoke_shuf|smoke_latent|train_m_tp SEED P|smoke_visa|train_visa_ctrl|train_visa_zoom|train_visa_ecp [SEED]|fixed_ckpt CKPT [ZI] [ZP] [FLAGS] [TAG]|test_mvtec_visa CKPT|test_cls CKPT [FLAGS]}" ;;
esac
