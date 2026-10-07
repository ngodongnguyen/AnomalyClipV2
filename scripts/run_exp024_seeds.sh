#!/bin/bash
# EXP-024: matched three-seed confirmation of ECP vs zoom-only vs dual (seed 111 already exists, this trains seeds 222 and 333).
# Pipeline: trains the six models one after another (GPU-bound) and, as soon as a model finishes, evaluates it with two parallel
# streams (PRO is CPU-bound) while the next model trains. Evaluation = fixed z_img=-0.69, z_pix=1.6, sigma 4, six pixel sets + three image sets.
#   nohup bash scripts/run_exp024_seeds.sh > par_seeds.log 2>&1 &
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"
F="--ec_z_img -0.69 --ec_z_pix 1.6 --sigma 4"
mkdir -p logs_seeds

ev () {  # $1 checkpoint name, $2 --dataset, $3 data path, $4 result folder, $5 metrics
  CUDA_VISIBLE_DEVICES=0 python test.py --dataset $2 --data_path "$3" --save_path ./results/$1_fixed/$4 \
    --checkpoint_path ./checkpoints/$1/epoch_15.pth $COMMON --metrics $5 $F > logs_seeds/$1_$4.log 2>&1
}
eval_model () {  # two background streams of about 45 min each
  m=$1
  ( ev $m colon $A/CVC/Kvasir Kvasir pixel-level; ev $m colon $A/CVC/CVC-ClinicDB CVC-ClinicDB pixel-level; \
    ev $m brain $A/HeadCT_anomaly_detection HeadCT_anomaly_detection image-level; ev $m brain $A/BrainMRI BrainMRI image-level ) &
  ( ev $m brain $A/br35 br35 image-level; ev $m thyroid "$A/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level; \
    ev $m ISBI $A/ISIC isic pixel-level; ev $m colon $A/CVC/CVC-ColonDB CVC-ColonDB pixel-level; \
    ev $m colon $A/EndoTect_2020_Segmentation_Test_Dataset endo pixel-level ) &
}
train_one () {  # $1 run_ecp.sh subcommand, $2 seed, $3 checkpoint name
  bash run_ecp.sh $1 $2 > logs_seeds/train_$3.log 2>&1
  if [ ! -f checkpoints/$3/epoch_15.pth ]; then echo "MISSING $3 (see logs_seeds/train_$3.log)"; return 1; fi
  echo "TRAINED $3"
}

for s in 222 333; do
  for pair in "train_m_ecp matched_ecp_s$s" "train_m_zoom matched_zoom_s$s" "train_m_dual matched_dual_s$s"; do
    set -- $pair
    train_one $1 $s $2 && eval_model $2
  done
done
wait
echo SEEDS DONE
