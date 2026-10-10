#!/usr/bin/env bash
# EXP-034 / H024: translation intervention on lesion position (frozen matched ECP, forward-only). Sequential, one dataset at a time, resumable.
#   bash scripts/run_exp034.sh smoke     # 12 random kept images of ClinicDB and ISIC -> results/EXP-034/smoke (no EXP-033 position csv needed)
#   bash scripts/run_exp034.sh full      # six pixel sets (decision sets first), then the report
#   bash scripts/run_exp034.sh report    # report only, no GPU
# Needs: conda env nguyennd (conda activate nguyennd).  REDO=1 overwrites finished datasets.  NO_CENTRAL=1 skips the central-tercile control.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
CKPT=${CKPT:-checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}
REF_ROOT=${REF_ROOT:-results/EXP-012/matched-retry-20261002}      # <ds>/ecp_extent/pixel_per_image_predictions.csv
POS=${POS:-results/EXP-032}                                       # position_<set>.csv (EXP-033) and summary_<set>.csv (EXP-032), optional
OUT=${OUT:-results/EXP-034}; mkdir -p "$OUT"
PY=${PYTHON:-python}
REDO_FLAG=""; [ "${REDO:-0}" = "1" ] && REDO_FLAG="--redo"
NC_FLAG=""; [ "${NO_CENTRAL:-0}" = "1" ] && NC_FLAG="--no_central"
px () {  # name dataset path
  local t0=$(date +%s)
  $PY analyze_position_shift.py pixel --name "$1" --dataset "$2" --data_path "$3" --checkpoint_path "$CKPT" \
     --reference_csv "$REF_ROOT/$1/ecp_extent/pixel_per_image_predictions.csv" --position_dir "$POS" --out_dir "$OUT" $REDO_FLAG $NC_FLAG ${EXTRA:-} 2>&1 | tee -a "$OUT/log_$1.txt"
  echo "[wall] $1 $(( $(date +%s) - t0 ))s"
}
case "${1:-}" in
  smoke)
    OUT=$OUT/smoke; mkdir -p $OUT; EXTRA="--limit 12 --redo"
    px ClinicDB colon $CVC/CVC-ClinicDB
    px ISIC ISBI "$A/ISIC"
    $PY analyze_position_shift.py report --report_dir "$OUT" --boot 200 | tail -30; du -sh "$OUT" ;;
  full)
    px ClinicDB colon $CVC/CVC-ClinicDB
    px ColonDB  colon $CVC/CVC-ColonDB
    px Endo     colon "$A/EndoTect_2020_Segmentation_Test_Dataset"
    px ISIC     ISBI  "$A/ISIC"
    px Kvasir   colon $CVC/Kvasir
    px TN3K     thyroid "$A/TN3K/Thyroid Dataset/tn3k"
    $PY analyze_position_shift.py report --report_dir "$OUT" | tee "$OUT/report_exp034.txt"; du -sh "$OUT" ;;
  report) $PY analyze_position_shift.py report --report_dir "$OUT" | tee "$OUT/report_exp034.txt" ;;
  *) echo "usage: $0 smoke|full|report" ;;
esac
