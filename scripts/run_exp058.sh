#!/bin/bash
# EXP-058: inference strategy of the extent coordinate on the MATCHED ECP checkpoints (seeds 111/222/333), evaluation only.
#   FIX0   : --ec_const_z 0                      (one shared z = 0 for both heads)          all nine sets
#   PRED   : no z flags                          (per-image estimate z-hat, both heads)     all nine sets
#   ORACLE : --ec_oracle                         (z_gt of each test image, DIAGNOSTIC)      the six PIXEL sets only (the image sets have no masks)
#   TASK   : the deployed (-0.69, 1.6) pair; existing numbers are reused, only the parity run below is new.
# Rules: research/EXPERIMENTS.md, EXP-058.  Run from the repo root in the `nguyennd` env; foreground, resumable (a finished log is skipped).
#   bash scripts/run_exp058.sh check      # unit tests, checkpoints present
#   bash scripts/run_exp058.sh parity     # seed 111, ClinicDB with the deployed flags must give 88.33 / 76.43 within 0.1 (STOP otherwise)
#   bash scripts/run_exp058.sh fix0|pred|oracle   [SEEDS_TO_RUN="111 222 333"]   ~85 min per checkpoint (oracle ~60 min)
#   bash scripts/run_exp058.sh report     # python scripts/collect_exp058.py
set -u
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"
OUT=results/EXP-058
mkdir -p $OUT logs_exp058
ck () { case "$1" in 111) echo "checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth" ;; *) echo "checkpoints/matched_ecp_s$1/epoch_15.pth" ;; esac; }
ev () {  # $1 strategy folder, $2 seed, $3 --dataset, $4 data path, $5 result folder, $6 metrics, $7 test.py flags
  local dir=$OUT/$1_s$2/$5
  if tail -n 3 $dir/log.txt 2>/dev/null | grep -q "^| mean"; then echo "skip $dir"; return 0; fi
  [ -d $dir ] && mv $dir $dir.partial.$(date +%s)
  echo "RUN  $1 s$2 $5 :: python test.py ... --metrics $6 --sigma 4 $7" | tee -a logs_exp058/commands.log
  CUDA_VISIBLE_DEVICES=${DEV:-0} python test.py --dataset $3 --data_path "$4" --save_path $dir \
    --checkpoint_path $(ck $2) $COMMON --metrics $6 --sigma 4 $7 > logs_exp058/$1_s$2_$5.log 2>&1
}
pixel_sets () { local s=$1 f=$2 n=$3
  ev $n $s colon $A/CVC/Kvasir Kvasir pixel-level "$f"; ev $n $s colon $A/CVC/CVC-ClinicDB CVC-ClinicDB pixel-level "$f"
  ev $n $s colon $A/CVC/CVC-ColonDB CVC-ColonDB pixel-level "$f"; ev $n $s ISBI $A/ISIC isic pixel-level "$f"
  ev $n $s colon $A/EndoTect_2020_Segmentation_Test_Dataset endo pixel-level "$f"; ev $n $s thyroid "$A/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level "$f"
}
image_sets () { local s=$1 f=$2 n=$3
  ev $n $s brain $A/HeadCT_anomaly_detection HeadCT_anomaly_detection image-level "$f"; ev $n $s brain $A/BrainMRI BrainMRI image-level "$f"
  ev $n $s brain $A/br35 br35 image-level "$f"
}
case "${1:-}" in
  check)
    python scripts/test_collect_exp058.py
    for s in 111 222 333; do p=$(ck $s); [ -f $p ] && echo "  seed $s -> $p OK" || echo "  seed $s -> $p MISSING"; done
    for p in results/EXP-012-matched/ecp_extent/checkpoints/CVC-ClinicDB/log.txt results/matched_ecp_s222_fixed/CVC-ClinicDB/log.txt; do
      [ -f $p ] && echo "  TASK reference $p OK" || echo "  TASK reference $p MISSING (the collector reads the existing deployed-z results)"; done ;;
  parity)
    ev parity 111 colon $A/CVC/CVC-ClinicDB CVC-ClinicDB pixel-level "--ec_z_img -0.69 --ec_z_pix 1.6"
    python - <<'PY'
import sys; sys.path.insert(0, "scripts")
import collect_exp024 as c
v = c.last_table_row("results/EXP-058/parity_s111/CVC-ClinicDB/log.txt")
print("parity ClinicDB seed 111 (deployed flags):", v, "recorded 88.33 / 76.43")
sys.exit(0 if v and abs(v[0] - 88.33) <= 0.1 and abs(v[1] - 76.43) <= 0.1 else 1)
PY
    [ $? -eq 0 ] && echo "EXP-058 PARITY PASSED" || { echo "EXP-058 PARITY FAILED: do not run the arms"; exit 1; } ;;
  fix0)   for s in ${SEEDS_TO_RUN:-111 222 333}; do pixel_sets $s "--ec_const_z 0" fix0; image_sets $s "--ec_const_z 0" fix0; done; echo "EXP058 FIX0 DONE" ;;
  pred)   for s in ${SEEDS_TO_RUN:-111 222 333}; do pixel_sets $s "" pred; image_sets $s "" pred; done; echo "EXP058 PRED DONE" ;;
  oracle) for s in ${SEEDS_TO_RUN:-111 222 333}; do pixel_sets $s "--ec_oracle" oracle; done; echo "EXP058 ORACLE DONE (pixel sets only, diagnostic)" ;;
  report) python scripts/collect_exp058.py ;;
  *) echo "usage: $0 check|parity|fix0|pred|oracle|report" ;;
esac
