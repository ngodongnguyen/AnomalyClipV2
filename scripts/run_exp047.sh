#!/usr/bin/env bash
# EXP-047: CPU gate for supervising more lesion attributes (solidity, boundary contrast, inhomogeneity). No model, no GPU.
# Needs results/EXP-032/<set>/badcase_<set>.csv and position_<set>.csv (EXP-032/033) on the machine where `report` runs.
#   bash scripts/run_exp047.sh check     # compile + unit tests (synthetic)
#   bash scripts/run_exp047.sh compute   # MVTec + six medical sets, one after another, resumable (skips existing CSVs; EXTRA=--redo to overwrite)
#   bash scripts/run_exp047.sh report    # A/B/C statistics and verdict -> results/EXP-047/report_exp047.txt
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data; CVC=$A/CVC; MVTEC=$A/mvtec; OUT=${OUT:-results/EXP-047}; POS=${POS:-results/EXP-032}; PY=${PYTHON:-python}
c () { $PY analyze_attr_gate.py compute --name "$1" --dataset "$2" --data_path "$3" --out_dir "$OUT" ${EXTRA:-}; }
case "${1:-}" in
  check) $PY -m py_compile attr_stats.py analyze_attr_gate.py test_attr_stats.py && $PY test_attr_stats.py ;;
  compute)
    c MVTec mvtec "$MVTEC"
    c ClinicDB colon $CVC/CVC-ClinicDB; c ColonDB colon $CVC/CVC-ColonDB; c Endo colon "$A/EndoTect_2020_Segmentation_Test_Dataset"
    c ISIC ISBI "$A/ISIC"; c TN3K thyroid "$A/TN3K/Thyroid Dataset/tn3k"; c Kvasir colon $CVC/Kvasir ;;
  report) mkdir -p "$OUT"; $PY analyze_attr_gate.py report --attr_dir "$OUT" --mvtec_dir "$OUT" --pos_dir "$POS" | tee "$OUT/report_exp047.txt" ;;
  *) echo "usage: $0 check|compute|report" ;;
esac
