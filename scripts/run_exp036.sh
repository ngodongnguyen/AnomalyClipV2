#!/usr/bin/env bash
# EXP-036 / H026: centre-weighted multi-shift fusion (CWMS), label-free test-time rule on the frozen matched ECP.
#   bash scripts/run_exp036.sh smoke      # 12 random kept images of ClinicDB and ISIC -> results/EXP-036/smoke (no recorded-value checks)
#   bash scripts/run_exp036.sh full       # six sets, one at a time, then the report
#   bash scripts/run_exp036.sh report     # report only, no GPU
# Sequential, resumable (an existing summary CSV is skipped; REDO=1 overwrites).  Needs conda env nguyennd.
# Later seeds (only if ADVANCE): CKPT=checkpoints/matched_ecp_s222/checkpoints/epoch_15.pth BASE=results/EXP-036-s222 bash scripts/run_exp036.sh full
#   (a CKPT other than the default seed-111 checkpoint switches off the EXP-012 parity references and the recorded-value check automatically).
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
DEFAULT_CKPT=checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth
CKPT=${CKPT:-$DEFAULT_CKPT}
REF_ROOT=${REF_ROOT:-results/EXP-012/matched-retry-20261002}
POS=${POS:-results/EXP-032}
BASE=${BASE:-results/EXP-036}; mkdir -p "$BASE"
PY=${PYTHON:-python}
REDO_FLAG=""; [ "${REDO:-0}" = "1" ] && REDO_FLAG="--redo"
SEED_FLAGS=""
if [ "$CKPT" != "$DEFAULT_CKPT" ]; then SEED_FLAGS="--no_recorded_check"; echo "CKPT differs from the seed-111 checkpoint: no EXP-012 parity reference, no recorded-value check"; fi
px () {  # out_dir mode name dataset path
  local t0=$(date +%s); local out=$1 mode=$2; shift 2
  mkdir -p "$out"
  local REF=""
  if [ -z "$SEED_FLAGS" ]; then REF="--reference_csv $REF_ROOT/$1/ecp_extent/pixel_per_image_predictions.csv"; fi
  $PY analyze_cwms.py "$mode" --name "$1" --dataset "$2" --data_path "$3" --checkpoint_path "$CKPT" $REF --position_dir "$POS" --out_dir "$out" \
     $REDO_FLAG $SEED_FLAGS ${EXTRA:-} 2>&1 | tee -a "$out/log_$1.txt"
  echo "[wall] $1 $(( $(date +%s) - t0 ))s"
}
rep () { $PY analyze_cwms.py report --report_dir "$1" ${EXTRA_REP:-} | tee "$1/report_exp036.txt"; }
case "${1:-}" in
  smoke)
    EXTRA="--redo"
    px $BASE/smoke smoke ClinicDB colon $CVC/CVC-ClinicDB
    px $BASE/smoke smoke ISIC ISBI "$A/ISIC"
    echo "(smoke: the six-set verdict is INCOMPLETE by construction; look at the parity and per-set lines)"
    rep $BASE/smoke | tail -30 ;;
  full)
    px $BASE eval ClinicDB colon $CVC/CVC-ClinicDB
    px $BASE eval ColonDB  colon $CVC/CVC-ColonDB
    px $BASE eval Endo     colon "$A/EndoTect_2020_Segmentation_Test_Dataset"
    px $BASE eval ISIC     ISBI  "$A/ISIC"
    px $BASE eval TN3K     thyroid "$A/TN3K/Thyroid Dataset/tn3k"
    px $BASE eval Kvasir   colon $CVC/Kvasir
    rep $BASE; du -sh "$BASE" ;;
  report) rep $BASE ;;
  *) echo "usage: $0 smoke|full|report" ;;
esac
