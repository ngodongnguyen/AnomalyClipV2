#!/usr/bin/env bash
# EXP-032 / H022: per-image bad-case characterisation of the matched ECP (frozen, forward-only).  Sequential, one dataset at a time, resumable.
#   bash scripts/run_exp032.sh smoke     # 20 random images of ClinicDB + 20 of HeadCT -> results/EXP-032/smoke (no recorded-value checks)
#   bash scripts/run_exp032.sh full      # six pixel sets, three image sets, then the report
#   bash scripts/run_exp032.sh report    # report only (tags, ratios, gate), no GPU
# Needs: conda env nguyennd (conda activate nguyennd).  REDO=1 overwrites finished datasets.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
CKPT=${CKPT:-checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}
REF_ROOT=${REF_ROOT:-results/EXP-012/matched-retry-20261002}      # <ds>/ecp_extent/pixel_per_image_predictions.csv
OUT=${OUT:-results/EXP-032}; mkdir -p "$OUT"
PY=${PYTHON:-python}
REDO_FLAG=""; [ "${REDO:-0}" = "1" ] && REDO_FLAG="--redo"
px () {  # name dataset path
  local t0=$(date +%s)
  $PY analyze_badcases.py pixel --name "$1" --dataset "$2" --data_path "$3" --checkpoint_path "$CKPT" \
     --reference_csv "$REF_ROOT/$1/ecp_extent/pixel_per_image_predictions.csv" --out_dir "$OUT" $REDO_FLAG ${EXTRA:-} 2>&1 | tee -a "$OUT/log_$1.txt"
  echo "[wall] $1 $(( $(date +%s) - t0 ))s"
}
im () {  # name path
  local t0=$(date +%s)
  $PY analyze_badcases.py imgset --name "$1" --dataset brain --data_path "$2" --checkpoint_path "$CKPT" --out_dir "$OUT" $REDO_FLAG ${EXTRA:-} 2>&1 | tee -a "$OUT/log_$1.txt"
  echo "[wall] $1 $(( $(date +%s) - t0 ))s"
}
report () { $PY analyze_badcases.py report --report_dir "$OUT" | tee "$OUT/report.txt"; echo "total folder size:"; du -sh "$OUT"; }
case "${1:-}" in
  smoke)
    OUT=$OUT/smoke; mkdir -p $OUT; EXTRA="--limit 20 --redo"
    px ClinicDB colon $CVC/CVC-ClinicDB
    im HeadCT $A/HeadCT_anomaly_detection
    $PY analyze_badcases.py report --report_dir "$OUT" --boot 200 | tail -25; du -sh "$OUT" ;;
  full)
    px ClinicDB colon $CVC/CVC-ClinicDB
    px ColonDB  colon $CVC/CVC-ColonDB
    px Endo     colon "$A/EndoTect_2020_Segmentation_Test_Dataset"
    px ISIC     ISBI  "$A/ISIC"
    px TN3K     thyroid "$A/TN3K/Thyroid Dataset/tn3k"
    px Kvasir   colon $CVC/Kvasir
    im HeadCT   $A/HeadCT_anomaly_detection
    im BrainMRI $A/BrainMRI
    im Br35H    $A/br35
    report ;;
  report) report ;;
  *) echo "usage: $0 smoke|full|report"; exit 1 ;;
esac
