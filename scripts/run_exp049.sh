#!/usr/bin/env bash
# EXP-049 / H038: PE-shift diagnostic (EXP-048 protocol, unchanged) on ECP seeds 222/333 and on the no-zoom control (CTRL) and zoom-only (ZO), seeds 111/222/333.
# Frozen models, forward only, foreground, resumable per checkpoint and set (an existing pe_<set>.csv is skipped; REDO=1 overwrites).  Needs conda env nguyennd.
#   bash scripts/run_exp049.sh check                 # unit tests (no GPU) + checkpoint/data path listing
#   bash scripts/run_exp049.sh smoke                 # ONE checkpoint (SMOKE_TAG, default ctrl_s111), ClinicDB, --limit 8, -> results/EXP-049/smoke/<tag>
#   bash scripts/run_exp049.sh run                   # all checkpoints below; CKPTS="ctrl_s111 zo_s111" runs a subset; EXTRA_SETS="Kvasir TN3K" adds report-only sets for CTRL/ZO
#   bash scripts/run_exp049.sh report                # collect_exp049.py, no GPU
# Order: ecp_s222 ecp_s333 (six sets) then ctrl_s111 ctrl_s222 ctrl_s333 zo_s111 zo_s222 zo_s333 (ClinicDB, ColonDB, Endo [+ EXTRA_SETS]).  ClinicDB comes first for every checkpoint:
# its run begins with the pooled-AUROC parity pass (all kept images, tol 0.15 point) and STOPs the whole script on a failure.  No nohup.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
REF_ROOT=${REF_ROOT:-results/EXP-012/matched-retry-20261002}
POS=${POS:-results/EXP-032}
BASE=${BASE:-results/EXP-049}; mkdir -p "$BASE"
PY=${PYTHON:-python}
ALL_CKPTS="ecp_s222 ecp_s333 ctrl_s111 ctrl_s222 ctrl_s333 zo_s111 zo_s222 zo_s333"
CKPTS=${CKPTS:-$ALL_CKPTS}
REDO_FLAG=""; [ "${REDO:-0}" = "1" ] && REDO_FLAG="--redo"
NC_FLAG=""; [ "${NO_CENTRAL:-0}" = "1" ] && NC_FLAG="--no_central"

ckpt_path () {   # tag -> checkpoint file (names from run_ecp.sh train_m_* / scripts/run_exp024_seeds.sh / scripts/run_exp030.sh / EXP-012 matched pair)
  case "$1" in
    ecp_s222|ecp_s333)   echo "checkpoints/matched_ecp_s${1#ecp_s}/epoch_15.pth" ;;
    ctrl_s111|ctrl_s222|ctrl_s333) echo "checkpoints/matched_ctrl_s${1#ctrl_s}/epoch_15.pth" ;;
    zo_s111)             echo "checkpoints/EXP-012-matched/zoom_only/checkpoints/epoch_15.pth" ;;
    zo_s222|zo_s333)     echo "checkpoints/matched_zoom_s${1#zo_s}/epoch_15.pth" ;;
    *) echo "unknown tag $1" >&2; return 1 ;;
  esac
}
sets_for () {    # tag -> sets in run order
  case "$1" in
    ecp_*) echo "ClinicDB ColonDB Endo Kvasir TN3K ISIC" ;;
    *)     echo "ClinicDB ColonDB Endo ${EXTRA_SETS:-}" ;;
  esac
}
set_args () {    # set -> "<dataset mode>|<data path>"
  case "$1" in
    ClinicDB) echo "colon|$CVC/CVC-ClinicDB" ;;
    ColonDB)  echo "colon|$CVC/CVC-ColonDB" ;;
    Endo)     echo "colon|$A/EndoTect_2020_Segmentation_Test_Dataset" ;;
    Kvasir)   echo "colon|$CVC/Kvasir" ;;
    TN3K)     echo "thyroid|$A/TN3K/Thyroid Dataset/tn3k" ;;
    ISIC)     echo "ISBI|$A/ISIC" ;;
    *) echo "unknown set $1" >&2; return 1 ;;
  esac
}
ref_for () {     # tag set -> reference csv (EXP-012 per-image parity) or empty. Only ZO seed 111 has one (ECP s111 is EXP-048; CTRL and the later seeds were not exported per image).
  case "$1" in zo_s111) echo "$REF_ROOT/$2/zoom_only/pixel_per_image_predictions.csv" ;; *) echo "" ;; esac
}
preflight () {   # fail with a list of every missing file
  local miss=0 t s r
  for t in $CKPTS; do
    p=$(ckpt_path "$t") || exit 1
    if [ -f "$p" ]; then echo "  $t -> $p  OK"; else echo "  $t -> $p  MISSING"; miss=1; fi
    for s in $(sets_for "$t"); do
      r=$(ref_for "$t" "$s")
      if [ -n "$r" ] && [ ! -f "$r" ]; then echo "  reference for $t/$s MISSING: $r"; miss=1; fi
    done
  done
  for s in ClinicDB ColonDB Endo; do
    d=$(set_args $s | cut -d'|' -f2); [ -f "$d/meta.json" ] || { echo "  data $s MISSING meta.json: $d"; miss=1; }
  done
  [ -f "$POS/ClinicDB/position_ClinicDB.csv" ] || echo "  note: $POS/ClinicDB/position_ClinicDB.csv not found (the EXP-033 cross-check of c is skipped)"
  if [ "$miss" = 1 ]; then echo "STOP: missing inputs (see above)"; exit 1; fi
}
px () {  # tag set out_dir [extra args...]
  local tag=$1 set=$2 out=$3; shift 3
  local ds path ref t0 sa
  sa=$(set_args "$set"); ds=${sa%%|*}; path=${sa#*|}
  ref=$(ref_for "$tag" "$set")
  mkdir -p "$out"
  t0=$(date +%s)
  $PY analyze_pe_shift.py pixel --name "$set" --dataset "$ds" --data_path "$path" --checkpoint_path "$(ckpt_path "$tag")" --tag "$tag" --out_dir "$out" \
     ${ref:+--reference_csv "$ref"} --position_dir "$POS" $REDO_FLAG $NC_FLAG "$@" 2>&1 | tee -a "$out/log_$set.txt"
  echo "[wall] $tag $set $(( $(date +%s) - t0 ))s"
}
case "${1:-}" in
  check)
    $PY test_pe_shift_stats.py; $PY test_pe_shift_core.py; $PY test_collect_exp049.py
    echo "checkpoints / inputs:"; preflight ;;
  smoke)
    T=${SMOKE_TAG:-ctrl_s111}; CKPTS=$T
    preflight
    px "$T" ClinicDB "$BASE/smoke/$T" --limit 8 --redo
    $PY analyze_pe_shift.py report --report_dir "$BASE/smoke/$T" --sets ClinicDB --boot 200 | tail -12
    echo "smoke: the pooled parity above was computed on the 8-image subset only (informational); the full pass runs inside 'run'." ;;
  run)
    preflight
    T00=$(date +%s)
    for t in $CKPTS; do
      T0=$(date +%s); echo "=============== EXP-049 checkpoint $t ($(ckpt_path "$t")) ==============="
      for s in $(sets_for "$t"); do px "$t" "$s" "$BASE/$t"; done
      echo "[wall] checkpoint $t total $(( $(date +%s) - T0 ))s"
    done
    echo "[wall] all $(( $(date +%s) - T00 ))s"
    $PY collect_exp049.py --root "$BASE" | tee "$BASE/report_exp049.txt"; du -sh "$BASE" ;;
  report) $PY collect_exp049.py --root "$BASE" | tee "$BASE/report_exp049.txt" ;;
  *) echo "usage: $0 check|smoke|run|report" ;;
esac
