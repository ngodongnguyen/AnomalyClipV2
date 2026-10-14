#!/usr/bin/env bash
# EXP-045 / H034, H035: free read-only gates for sparse (entmax-1.5) V-V attention and a source-selected head subset. Frozen matched ECP, forward only.
# Foreground, output shown in the terminal, resumable (a finished dataset CSV is skipped; a partial one resumes).
#   bash scripts/run_exp045.sh check      # unit tests + compile (no GPU needed for the stats test; the core test needs torch)
#   bash scripts/run_exp045.sh select     # MVTec head scores -> results/EXP-045/head_scores_mvtec.npy, head_ranking_mvtec.json (300 images)
#   bash scripts/run_exp045.sh smoke      # 20 ClinicDB images, all arms, prints parity and s/image -> results/EXP-045/smoke
#   bash scripts/run_exp045.sh run        # ClinicDB, Kvasir, ColonDB sequentially, prints wall time per dataset, then the report
#   bash scripts/run_exp045.sh report     # aggregate + pre-registered verdict (CPU only)
#   bash scripts/run_exp045.sh transfer   # report-only: head scores on the three medical sets and their Spearman with the MVTec scores (never used to select)
# REDO=1 overwrites finished datasets.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
MVTEC=${MVTEC:-$A/mvtec}
CKPT=${CKPT:-checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}
REF_ROOT=${REF_ROOT:-results/EXP-012/matched-retry-20261002}      # <ds>/ecp_extent/pixel_per_image_predictions.csv
OUT=${OUT:-results/EXP-045}; mkdir -p "$OUT"
PY=${PYTHON:-python}
RANK=$OUT/head_ranking_mvtec.json
REDO_FLAG=""; [ "${REDO:-0}" = "1" ] && REDO_FLAG="--redo"
need_rank () { [ -f "$RANK" ] || { echo "missing $RANK: run 'bash scripts/run_exp045.sh select' first"; exit 1; }; }
ev () {  # name dataset path outdir [extra]
  local t0=$(date +%s)
  $PY analyze_vvattn.py eval --name "$1" --dataset "$2" --data_path "$3" --checkpoint_path "$CKPT" \
     --reference_csv "$REF_ROOT/$1/ecp_extent/pixel_per_image_predictions.csv" --ranking "$RANK" \
     --out_csv "$4/vvattn_$1.csv" $REDO_FLAG ${5:-} 2>&1 | tee -a "$4/log_$1.txt"
  echo "[wall] $1 $(( $(date +%s) - t0 ))s"
}
case "${1:-}" in
  check)
    $PY -m py_compile analyze_vvattn.py vvattn_core.py vvattn_stats.py scripts/select_heads_mvtec.py
    $PY test_vvattn_stats.py
    $PY test_vvattn_core.py
    echo "EXP-045 CHECK PASSED" ;;
  select)
    t0=$(date +%s)
    $PY scripts/select_heads_mvtec.py select --data_path "$MVTEC" --out_dir "$OUT" ${EXTRA:-} 2>&1 | tee -a "$OUT/log_select.txt"
    echo "[wall] select $(( $(date +%s) - t0 ))s" ;;
  transfer)
    need_rank
    for spec in "ClinicDB colon $CVC/CVC-ClinicDB" "Kvasir colon $CVC/Kvasir" "ColonDB colon $CVC/CVC-ColonDB"; do
      set -- $spec
      $PY scripts/select_heads_mvtec.py transfer --name "$1" --dataset "$2" --data_path "$3" --out_dir "$OUT" 2>&1 | tee -a "$OUT/log_transfer.txt"
    done ;;
  smoke)
    need_rank
    mkdir -p "$OUT/smoke"; REDO_FLAG="--redo"
    ev ClinicDB colon $CVC/CVC-ClinicDB "$OUT/smoke" "--limit 20" ;;
  run)
    need_rank
    ev ClinicDB colon $CVC/CVC-ClinicDB "$OUT"
    ev Kvasir   colon $CVC/Kvasir "$OUT"
    ev ColonDB  colon $CVC/CVC-ColonDB "$OUT"
    $PY analyze_vvattn.py report --dir "$OUT" | tee "$OUT/report_exp045.txt" ;;
  report) $PY analyze_vvattn.py report --dir "$OUT" | tee "$OUT/report_exp045.txt" ;;
  *) echo "usage: $0 check|select|transfer|smoke|run|report"; exit 1 ;;
esac
