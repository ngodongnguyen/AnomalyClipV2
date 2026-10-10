#!/usr/bin/env bash
# EXP-035 / H025: is the EXP-034 position response a reflection-padding artifact?  Same design as EXP-034 with content-free padding.
#   bash scripts/run_exp035.sh smoke      # 12 random kept images of ClinicDB and ISIC, arm MEAN -> results/EXP-035/smoke
#   bash scripts/run_exp035.sh full       # arm MEAN (decisive) on six sets, then arm BLACK (report-only), then both reports
#   bash scripts/run_exp035.sh report     # reports only, no GPU
# Sequential, resumable (an existing CSV is skipped; REDO=1 overwrites).  NO_CENTRAL=1 skips the central control.  Needs conda env nguyennd.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
CKPT=${CKPT:-checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}
REF_ROOT=${REF_ROOT:-results/EXP-012/matched-retry-20261002}
POS=${POS:-results/EXP-032}
BASE=${BASE:-results/EXP-035}; mkdir -p "$BASE"
PY=${PYTHON:-python}
REDO_FLAG=""; [ "${REDO:-0}" = "1" ] && REDO_FLAG="--redo"
NC_FLAG=""; [ "${NO_CENTRAL:-0}" = "1" ] && NC_FLAG="--no_central"
px () {  # pad out_dir name dataset path
  local t0=$(date +%s); local pad=$1 out=$2; shift 2
  mkdir -p "$out"
  $PY analyze_position_shift.py pixel --pad "$pad" --name "$1" --dataset "$2" --data_path "$3" --checkpoint_path "$CKPT" \
     --reference_csv "$REF_ROOT/$1/ecp_extent/pixel_per_image_predictions.csv" --position_dir "$POS" --out_dir "$out" $REDO_FLAG $NC_FLAG ${EXTRA:-} 2>&1 | tee -a "$out/log_$1.txt"
  echo "[wall] $pad $1 $(( $(date +%s) - t0 ))s"
}
arm () {  # pad
  local out=$BASE/$1
  px $1 $out ClinicDB colon $CVC/CVC-ClinicDB
  px $1 $out ColonDB  colon $CVC/CVC-ColonDB
  px $1 $out Endo     colon "$A/EndoTect_2020_Segmentation_Test_Dataset"
  px $1 $out ISIC     ISBI  "$A/ISIC"
  px $1 $out Kvasir   colon $CVC/Kvasir
  px $1 $out TN3K     thyroid "$A/TN3K/Thyroid Dataset/tn3k"
}
rep () { $PY analyze_position_shift.py report --exp035 --report_dir "$BASE/$1" | tee "$BASE/$1/report_exp035_$1.txt"; }
case "${1:-}" in
  smoke)
    EXTRA="--limit 12 --redo"
    px mean $BASE/smoke/mean ClinicDB colon $CVC/CVC-ClinicDB
    px mean $BASE/smoke/mean ISIC ISBI "$A/ISIC"
    $PY analyze_position_shift.py report --exp035 --report_dir "$BASE/smoke/mean" --boot 200 | tail -25 ;;
  full) arm mean; rep mean; arm black; rep black; du -sh "$BASE" ;;
  report) rep mean; [ -d "$BASE/black" ] && rep black || true ;;
  *) echo "usage: $0 smoke|full|report" ;;
esac
