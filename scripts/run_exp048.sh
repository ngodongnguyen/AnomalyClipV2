#!/usr/bin/env bash
# EXP-048 / H037: is the frozen ECP position response mediated by the positional embedding?  (PE-shift diagnostic, frozen model, forward only)
#   bash scripts/run_exp048.sh check     # unit tests, no GPU
#   bash scripts/run_exp048.sh smoke     # 8 random kept images of ClinicDB: parity + seconds per image -> results/EXP-048/smoke
#   bash scripts/run_exp048.sh run       # six sets in the order ClinicDB, ColonDB, Endo (decision), Kvasir, TN3K, ISIC (report-only), foreground, resumable
#   bash scripts/run_exp048.sh report    # reports + verdict, no GPU
# Resumable (an existing CSV is skipped; REDO=1 overwrites).  NO_CENTRAL=1 skips the central control.  Needs conda env nguyennd.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
CKPT=${CKPT:-checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}
REF_ROOT=${REF_ROOT:-results/EXP-012/matched-retry-20261002}
POS=${POS:-results/EXP-032}
BASE=${BASE:-results/EXP-048}; mkdir -p "$BASE"
PY=${PYTHON:-python}
REDO_FLAG=""; [ "${REDO:-0}" = "1" ] && REDO_FLAG="--redo"
NC_FLAG=""; [ "${NO_CENTRAL:-0}" = "1" ] && NC_FLAG="--no_central"
px () {  # out_dir name dataset path
  local t0=$(date +%s); local out=$1; shift
  mkdir -p "$out"
  $PY analyze_pe_shift.py pixel --name "$1" --dataset "$2" --data_path "$3" --checkpoint_path "$CKPT" \
     --reference_csv "$REF_ROOT/$1/ecp_extent/pixel_per_image_predictions.csv" --position_dir "$POS" --out_dir "$out" $REDO_FLAG $NC_FLAG ${EXTRA:-} 2>&1 | tee -a "$out/log_$1.txt"
  echo "[wall] $1 $(( $(date +%s) - t0 ))s"
}
case "${1:-}" in
  check)
    $PY test_pe_shift_stats.py; $PY test_pe_shift_core.py ;;
  smoke)
    EXTRA="--limit 8 --redo"
    px $BASE/smoke ClinicDB colon $CVC/CVC-ClinicDB
    $PY analyze_pe_shift.py report --report_dir "$BASE/smoke" --boot 200 | tail -20 ;;
  run)
    px $BASE ClinicDB colon $CVC/CVC-ClinicDB
    px $BASE ColonDB  colon $CVC/CVC-ColonDB
    px $BASE Endo     colon "$A/EndoTect_2020_Segmentation_Test_Dataset"
    px $BASE Kvasir   colon $CVC/Kvasir
    px $BASE TN3K     thyroid "$A/TN3K/Thyroid Dataset/tn3k"
    px $BASE ISIC     ISBI  "$A/ISIC"
    $PY analyze_pe_shift.py report --report_dir "$BASE" | tee "$BASE/report_exp048.txt"; du -sh "$BASE" ;;
  report) $PY analyze_pe_shift.py report --report_dir "$BASE" | tee "$BASE/report_exp048.txt" ;;
  *) echo "usage: $0 check|smoke|run|report" ;;
esac
