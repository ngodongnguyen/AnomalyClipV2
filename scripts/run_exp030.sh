#!/bin/bash
# EXP-030: matched no-zoom control (original AnomalyCLIP prompts: zoom_aug_p=0, --extent_cond none), seeds 111/222/333, same recipe as the ECP/ZO arms.
#   bash scripts/run_exp030.sh smoke
#   nohup bash scripts/run_exp030.sh full > par_exp030.log 2>&1 &     # 3 x 1h40 training + 3 x ~85 min evaluation, one thing at a time, resumable
#   python scripts/collect_exp030.py
# Evaluation settings are those of EXP-024 (fixed z_img=-0.69, z_pix=1.6 are ignored by a checkpoint without a conditioner; sigma 4).
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"
F="--ec_z_img -0.69 --ec_z_pix 1.6 --sigma 4"
mkdir -p logs_exp030
ev () {  # $1 checkpoint, $2 --dataset, $3 data path, $4 result folder, $5 metrics
  if tail -n 3 ./results/$1_fixed/$4/log.txt 2>/dev/null | grep -q "^| mean"; then return 0; fi
  [ -d ./results/$1_fixed/$4 ] && mv ./results/$1_fixed/$4 ./results/$1_fixed/$4.partial.$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python test.py --dataset $2 --data_path "$3" --save_path ./results/$1_fixed/$4 \
    --checkpoint_path ./checkpoints/$1/epoch_15.pth $COMMON --metrics $5 $F > logs_exp030/$1_$4.log 2>&1
}
eval_model () {
  m=$1
  ev $m colon $A/CVC/Kvasir Kvasir pixel-level; ev $m colon $A/CVC/CVC-ClinicDB CVC-ClinicDB pixel-level
  ev $m colon $A/CVC/CVC-ColonDB CVC-ColonDB pixel-level; ev $m ISBI $A/ISIC isic pixel-level
  ev $m colon $A/EndoTect_2020_Segmentation_Test_Dataset endo pixel-level; ev $m thyroid "$A/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level
  ev $m brain $A/HeadCT_anomaly_detection HeadCT_anomaly_detection image-level; ev $m brain $A/BrainMRI BrainMRI image-level
  ev $m brain $A/br35 br35 image-level
}
case "${1:-}" in
  smoke) bash run_ecp.sh smoke_ctrl > logs_exp030/smoke_ctrl.log 2>&1; tail -n 4 logs_exp030/smoke_ctrl.log ;;
  full)
    for s in ${SEEDS_TO_RUN:-111 222 333}; do
      n=matched_ctrl_s$s
      if [ ! -f checkpoints/$n/epoch_15.pth ]; then bash run_ecp.sh train_m_ctrl $s > logs_exp030/train_$n.log 2>&1; fi
      if [ ! -f checkpoints/$n/epoch_15.pth ]; then echo "MISSING $n (see logs_exp030/train_$n.log)"; continue; fi
      echo "TRAINED $n"; eval_model $n
    done
    echo EXP030 DONE ;;
  *) echo "usage: $0 smoke|full" ;;
esac
