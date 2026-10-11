#!/bin/bash
# EXP-037: position-jittered prompt training (translation augmentation), one seed pilot. Trains matched_tr_s<SEED> (default 111), then evaluates it
# with the fixed z settings of EXP-024 (z_img=-0.69, z_pix=1.6, sigma 4) and exports per-image pixel AUROCs for the position check.
#   bash scripts/run_exp037.sh smoke                       # 1 epoch with the translation augmentation (checks that training runs)
#   nohup bash scripts/run_exp037.sh full > par_exp037.log 2>&1 &    # ~1h40 training + ~85 min evaluation, one thing at a time, resumable
#   python scripts/collect_exp037.py                       # primary rule, mechanism check, guards
# SEEDS_TO_RUN="222 333" later (matched ECP seeds 222/333 exist) only if the pilot passes.
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"
F="--ec_z_img -0.69 --ec_z_pix 1.6 --sigma 4"
mkdir -p logs_exp037
ev () {  # $1 checkpoint, $2 --dataset, $3 data path, $4 result folder, $5 metrics
  if tail -n 3 ./results/$1_fixed/$4/log.txt 2>/dev/null | grep -q "^| mean"; then return 0; fi
  [ -d ./results/$1_fixed/$4 ] && mv ./results/$1_fixed/$4 ./results/$1_fixed/$4.partial.$(date +%s)
  EXP=""; [ "$5" = "pixel-level" ] && EXP="--export_pixel_metrics"
  CUDA_VISIBLE_DEVICES=0 python test.py --dataset $2 --data_path "$3" --save_path ./results/$1_fixed/$4 \
    --checkpoint_path ./checkpoints/$1/epoch_15.pth $COMMON --metrics $5 $F $EXP > logs_exp037/$1_$4.log 2>&1
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
  smoke) bash run_ecp.sh smoke_tr > logs_exp037/smoke_tr.log 2>&1; tail -n 5 logs_exp037/smoke_tr.log ;;
  full)
    for s in ${SEEDS_TO_RUN:-111}; do
      n=matched_tr_s$s
      if [ ! -f checkpoints/$n/epoch_15.pth ]; then bash run_ecp.sh train_m_tr $s > logs_exp037/train_$n.log 2>&1; fi
      if [ ! -f checkpoints/$n/epoch_15.pth ]; then echo "MISSING $n (see logs_exp037/train_$n.log)"; continue; fi
      echo "TRAINED $n"; eval_model $n
    done
    echo EXP037 DONE ;;
  *) echo "usage: $0 smoke|full" ;;
esac
