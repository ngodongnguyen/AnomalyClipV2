#!/usr/bin/env bash
# EXP-050 / H039: does zoom augmentation teach a centre prior (Part A, CPU), and does the frozen ECP PE response shrink when the pixel prompt is
# conditioned on a small extent (Part B, z_pix -0.69 vs the recorded z_pix 1.6 runs of EXP-048/049)?  Diagnostic only, no training.
# Rule: notes/EXP-050_registration.md (copy into research/EXPERIMENTS.md before any run).  Needs conda env nguyennd for Part B.
#   bash scripts/run_exp050.sh check     # unit tests (no GPU) + input listing
#   bash scripts/run_exp050.sh smoke     # Part A on the first 30 anomalous MVTec images + Part B on 8 ClinicDB images (ECP s111) -> results/EXP-050/smoke
#   bash scripts/run_exp050.sh source    # Part A, CPU only (a few minutes); STOPs if the crop stream differs from the EXP-040 audit (9,407 / 8,768)
#   bash scripts/run_exp050.sh zlow      # Part B forwards: ECP s111/s222/s333 x ClinicDB, ColonDB, Endo [+ EXTRA_SETS], peripheral tercile only
#   bash scripts/run_exp050.sh report    # both verdicts, no GPU
# Resumable (an existing pe_<set>.csv / part_a.json is skipped; REDO=1 overwrites).  No nohup.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
MVTEC=${MVTEC:-$A/mvtec}
POS=${POS:-results/EXP-032}
BASE=${BASE:-results/EXP-050}; mkdir -p "$BASE"
PY=${PYTHON:-python}
ZLOW=-0.69
TAGS=${TAGS:-"ecp_s111 ecp_s222 ecp_s333"}
REDO_FLAG=""; [ "${REDO:-0}" = "1" ] && REDO_FLAG="--redo"

ckpt_path () {
  case "$1" in
    ecp_s111) echo "checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth" ;;
    ecp_s222|ecp_s333) echo "checkpoints/matched_ecp_s${1#ecp_s}/epoch_15.pth" ;;
    *) echo "unknown tag $1" >&2; return 1 ;;
  esac
}
hi_dir () { case "$1" in ecp_s111) echo "results/EXP-048" ;; *) echo "results/EXP-049/$1" ;; esac; }
set_args () {
  case "$1" in
    ClinicDB) echo "colon|$CVC/CVC-ClinicDB" ;;
    ColonDB)  echo "colon|$CVC/CVC-ColonDB" ;;
    Endo)     echo "colon|$A/EndoTect_2020_Segmentation_Test_Dataset" ;;
    Kvasir)   echo "colon|$CVC/Kvasir" ;;
    TN3K)     echo "thyroid|$A/TN3K/Thyroid Dataset/tn3k" ;;
    *) echo "unknown set $1" >&2; return 1 ;;
  esac
}
preflight () {
  local miss=0 t s p
  [ -f "$MVTEC/meta.json" ] && echo "  MVTec meta.json OK" || { echo "  MVTec meta.json MISSING: $MVTEC"; miss=1; }
  for t in $TAGS; do
    p=$(ckpt_path "$t"); [ -f "$p" ] && echo "  $t -> $p OK" || { echo "  $t -> $p MISSING"; miss=1; }
    for s in ClinicDB ColonDB Endo ${EXTRA_SETS:-}; do
      p="$(hi_dir "$t")/$s/pe_$s.csv"; [ -f "$p" ] || { echo "  z_pix 1.6 reference MISSING: $p"; miss=1; }
    done
  done
  if [ "$miss" = 1 ]; then echo "STOP: missing inputs (see above)"; exit 1; fi
}
px () {  # tag set out_dir [extra...]
  local tag=$1 set=$2 out=$3; shift 3
  local sa ds path t0
  sa=$(set_args "$set"); ds=${sa%%|*}; path=${sa#*|}
  mkdir -p "$out"; t0=$(date +%s)
  $PY analyze_pe_shift.py pixel --name "$set" --dataset "$ds" --data_path "$path" --checkpoint_path "$(ckpt_path "$tag")" \
     --z_pix $ZLOW --out_dir "$out" --pooled_parity off --no_central --position_dir "$POS" $REDO_FLAG "$@" 2>&1 | tee -a "$out/log_$set.txt"
  echo "[wall] $tag $set $(( $(date +%s) - t0 ))s"
}
case "${1:-}" in
  check)
    $PY test_centre_prior_stats.py; $PY test_pe_shift_stats.py
    echo "inputs:"; preflight ;;
  smoke)
    preflight
    $PY analyze_centre_prior.py source --data_root "$MVTEC" --out_dir "$BASE/smoke/source" --limit 30 --boot 200 --redo
    px ecp_s111 ClinicDB "$BASE/smoke/zlow/ecp_s111" --limit 8 --redo
    echo "smoke done: Part A parity is not compared on a limited run; Part B has no SHAM parity at z_pix $ZLOW (EXP-012 references are z_pix 1.6)." ;;
  source)
    [ -f "$MVTEC/meta.json" ] || { echo "STOP: $MVTEC/meta.json missing"; exit 1; }
    $PY analyze_centre_prior.py source --data_root "$MVTEC" --out_dir "$BASE/source" $REDO_FLAG 2>&1 | tee "$BASE/source_log.txt" ;;
  zlow)
    preflight
    T00=$(date +%s)
    for t in $TAGS; do
      for s in ClinicDB ColonDB Endo ${EXTRA_SETS:-}; do px "$t" "$s" "$BASE/zlow/$t"; done
    done
    echo "[wall] zlow all $(( $(date +%s) - T00 ))s" ;;
  report)
    $PY analyze_centre_prior.py report --out_dir "$BASE/source" --zlow "$BASE/zlow" | tee "$BASE/report_exp050.txt" ;;
  *) echo "usage: $0 check|smoke|source|zlow|report" ;;
esac
