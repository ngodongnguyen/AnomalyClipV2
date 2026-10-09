#!/bin/bash
# EXP-028: does extent SUPERVISION add anything beyond a conditioner? Trains SHUF (extent labels rotated among the anomalous images of a batch)
# and LATENT (no extent loss, no teacher forcing) for seeds 111, 222, 333 under the matched recipe, and evaluates each, one thing at a time.
#   bash scripts/run_exp028.sh smoke     # 1-epoch trainings of both arms (check they run, then delete nothing: names shuf_smoke / latent_smoke)
#   nohup bash scripts/run_exp028.sh full > par_exp028.log 2>&1 &        # about 6 x 1h40 training + 6 x ~85 min evaluation, resumable
#   python scripts/collect_exp028.py                                       # rules R4-R6 (prints INCOMPLETE until every value exists)
# SHUF is evaluated like ECP (fixed z_img=-0.69, z_pix=1.6, sigma 4). LATENT has no meaningful z axis, so it is evaluated with its own per-image
# estimate (no z flags), sigma 4. Override: SEEDS_TO_RUN="111 222 333", ARMS_TO_RUN="shuf latent".
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"
mkdir -p logs_exp028

ev () {  # $1 checkpoint, $2 --dataset, $3 data path, $4 result folder, $5 metrics, $6 z flags
  if tail -n 3 ./results/$1_fixed/$4/log.txt 2>/dev/null | grep -q "^| mean"; then return 0; fi
  [ -d ./results/$1_fixed/$4 ] && mv ./results/$1_fixed/$4 ./results/$1_fixed/$4.partial.$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python test.py --dataset $2 --data_path "$3" --save_path ./results/$1_fixed/$4 \
    --checkpoint_path ./checkpoints/$1/epoch_15.pth $COMMON --metrics $5 --sigma 4 $6 > logs_exp028/$1_$4.log 2>&1
}
eval_model () {  # $1 checkpoint, $2 z flags
  m=$1; f="$2"
  ev $m colon $A/CVC/Kvasir Kvasir pixel-level "$f"; ev $m colon $A/CVC/CVC-ClinicDB CVC-ClinicDB pixel-level "$f"
  ev $m colon $A/CVC/CVC-ColonDB CVC-ColonDB pixel-level "$f"; ev $m ISBI $A/ISIC isic pixel-level "$f"
  ev $m colon $A/EndoTect_2020_Segmentation_Test_Dataset endo pixel-level "$f"; ev $m thyroid "$A/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level "$f"
  ev $m brain $A/HeadCT_anomaly_detection HeadCT_anomaly_detection image-level "$f"; ev $m brain $A/BrainMRI BrainMRI image-level "$f"
  ev $m brain $A/br35 br35 image-level "$f"
}
train_one () {  # $1 run_ecp.sh subcommand, $2 seed, $3 checkpoint name
  if [ -f checkpoints/$3/epoch_15.pth ]; then echo "SKIP training $3 (checkpoint exists)"; return 0; fi
  bash run_ecp.sh $1 $2 > logs_exp028/train_$3.log 2>&1
  if [ ! -f checkpoints/$3/epoch_15.pth ]; then echo "MISSING $3 (see logs_exp028/train_$3.log)"; return 1; fi
  echo "TRAINED $3"
}

case "${1:-}" in
  smoke) bash run_ecp.sh smoke_shuf > logs_exp028/smoke_shuf.log 2>&1; tail -n 4 logs_exp028/smoke_shuf.log
         bash run_ecp.sh smoke_latent > logs_exp028/smoke_latent.log 2>&1; tail -n 4 logs_exp028/smoke_latent.log ;;
  full)
    for arm in ${ARMS_TO_RUN:-shuf latent}; do
      for s in ${SEEDS_TO_RUN:-111 222 333}; do
        if [ $arm = shuf ]; then flags="--ec_z_img -0.69 --ec_z_pix 1.6"; else flags=""; fi
        train_one train_m_$arm $s matched_${arm}_s$s && eval_model matched_${arm}_s$s "$flags"
      done
    done
    echo EXP028 DONE ;;
  *) echo "usage: $0 smoke|full" ;;
esac
