#!/usr/bin/env bash
# EXP-051 / H040: position-debiased zoom. The matched ECP recipe with the zoom crop corner drawn from Beta(a, a) instead of uniform (dataset.py --zoom_place beta),
# a chosen ONLY on the MVTec source (sweep, CPU) by the rule in notes/EXP-051_registration.md. One-seed pilot (seed 111) against the matched ECP seed 111.
#   bash scripts/run_exp051.sh check     # unit tests (no GPU)
#   bash scripts/run_exp051.sh sweep     # source-only choice of a (CPU, about 10 min) -> results/EXP-051/sweep/selection.json; STOPs if no a is admissible
#   bash scripts/run_exp051.sh smoke     # 1 training epoch with the selected a
#   bash scripts/run_exp051.sh full      # train matched_eb_s111 (~1h40), evaluate 6 pixel + 3 image sets (~85 min), PE diagnostic on ClinicDB/ColonDB/Endo (~10 min)
#   bash scripts/run_exp051.sh report    # rules and verdicts (no GPU)
# Resumable: an existing checkpoint / evaluation log / pe CSV is skipped.  Needs conda env nguyennd.  No nohup inside.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
MVTEC=${MVTEC:-$A/mvtec}
PY=${PYTHON:-python}
SEED=${SEED:-111}
NAME=matched_eb_s$SEED
SEL=results/EXP-051/sweep/selection.json
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"
F="--ec_z_img -0.69 --ec_z_pix 1.6 --sigma 4"
mkdir -p logs_exp051 results/EXP-051

selected_beta () {
  [ -f "$SEL" ] || { echo "STOP: $SEL missing (run: bash scripts/run_exp051.sh sweep)" >&2; exit 1; }
  local a
  a=$($PY -c "import json; d=json.load(open('$SEL')); assert not d.get('limit'), 'limited sweep'; print(d['selected_beta'])")
  if [ "$a" = "None" ]; then echo "STOP: no admissible a in the registered grid; Step 2 training is not authorised" >&2; exit 1; fi
  echo "$a"
}
ev () {  # $1 --dataset, $2 data path, $3 result folder, $4 metrics
  local out=./results/${NAME}_fixed/$3
  if tail -n 3 "$out/log.txt" 2>/dev/null | grep -q "^| mean"; then return 0; fi
  [ -d "$out" ] && mv "$out" "$out.partial.$(date +%s)"
  local EXP=""; [ "$4" = "pixel-level" ] && EXP="--export_pixel_metrics"
  CUDA_VISIBLE_DEVICES=0 $PY test.py --dataset $1 --data_path "$2" --save_path "$out" \
    --checkpoint_path ./checkpoints/$NAME/epoch_15.pth $COMMON --metrics $4 $F $EXP > logs_exp051/${NAME}_$3.log 2>&1
  echo "[eval] $3 done"
}
pe () {  # $1 set, $2 --dataset, $3 data path
  local out=results/EXP-051/pe
  mkdir -p "$out"
  $PY analyze_pe_shift.py pixel --name "$1" --dataset "$2" --data_path "$3" --checkpoint_path ./checkpoints/$NAME/epoch_15.pth \
     --z_pix 1.6 --out_dir "$out" --pooled_parity off --no_central --position_dir results/EXP-032 2>&1 | tee -a "$out/log_$1.txt" | tail -n 6
}
case "${1:-}" in
  check)
    $PY test_centre_prior_stats.py; $PY scripts/test_collect_exp051.py; $PY test_pe_shift_stats.py
    [ -f "$MVTEC/meta.json" ] && echo "MVTec meta.json OK" || echo "MVTec meta.json MISSING: $MVTEC"
    for f in checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth results/EXP-048/ClinicDB/pe_ClinicDB.csv results/EXP-032/ClinicDB/badcase_ClinicDB.csv; do
      [ -f "$f" ] && echo "  $f OK" || echo "  $f MISSING"
    done ;;
  sweep)
    [ -f "$MVTEC/meta.json" ] || { echo "STOP: $MVTEC/meta.json missing"; exit 1; }
    $PY analyze_centre_prior.py sweep --data_root "$MVTEC" --sweep_dir results/EXP-051/sweep ${REDO:+--redo} 2>&1 | tee results/EXP-051/sweep_log.txt ;;
  smoke)
    a=$(selected_beta); echo "selected zoom_beta $a"
    bash run_ecp.sh smoke_eb "$a" > logs_exp051/smoke_eb.log 2>&1; tail -n 5 logs_exp051/smoke_eb.log ;;
  full)
    a=$(selected_beta); echo "selected zoom_beta $a"
    if [ ! -f checkpoints/$NAME/epoch_15.pth ]; then bash run_ecp.sh train_m_eb $SEED "$a" > logs_exp051/train_$NAME.log 2>&1; fi
    [ -f checkpoints/$NAME/epoch_15.pth ] || { echo "MISSING $NAME (see logs_exp051/train_$NAME.log)"; exit 1; }
    echo "TRAINED $NAME"
    ev colon $CVC/Kvasir Kvasir pixel-level; ev colon $CVC/CVC-ClinicDB CVC-ClinicDB pixel-level
    ev colon $CVC/CVC-ColonDB CVC-ColonDB pixel-level; ev ISBI $A/ISIC isic pixel-level
    ev colon $A/EndoTect_2020_Segmentation_Test_Dataset endo pixel-level; ev thyroid "$A/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level
    ev brain $A/HeadCT_anomaly_detection HeadCT_anomaly_detection image-level; ev brain $A/BrainMRI BrainMRI image-level
    ev brain $A/br35 br35 image-level
    pe ClinicDB colon $CVC/CVC-ClinicDB; pe ColonDB colon $CVC/CVC-ColonDB; pe Endo colon $A/EndoTect_2020_Segmentation_Test_Dataset
    echo "EXP051 DONE" ;;
  report) $PY scripts/collect_exp051.py $SEED | tee results/EXP-051/report_exp051.txt ;;
  *) echo "usage: $0 check|sweep|smoke|full|report" ;;
esac
