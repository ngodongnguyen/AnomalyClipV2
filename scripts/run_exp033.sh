#!/usr/bin/env bash
# EXP-033: lesion position / polarity versus per-image AUROC. CPU only (no model). Needs results/EXP-032/<set>/badcase_<set>.csv from EXP-032.
#   bash scripts/run_exp033.sh compute    # six sets, one after another (images and masks only), writes results/EXP-032/<set>/position_<set>.csv
#   bash scripts/run_exp033.sh report     # statistics and the pre-registered verdict (can also run locally on the copied CSVs)
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data; CVC=$A/CVC; OUT=${OUT:-results/EXP-032}; PY=${PYTHON:-python}
c () { $PY analyze_position.py compute --name "$1" --dataset "$2" --data_path "$3" --out_dir "$OUT" ${EXTRA:-}; }
case "${1:-}" in
  compute)
    c ClinicDB colon $CVC/CVC-ClinicDB; c ColonDB colon $CVC/CVC-ColonDB; c Endo colon "$A/EndoTect_2020_Segmentation_Test_Dataset"
    c ISIC ISBI "$A/ISIC"; c TN3K thyroid "$A/TN3K/Thyroid Dataset/tn3k"; c Kvasir colon $CVC/Kvasir ;;
  report) $PY analyze_position.py report --report_dir "$OUT" | tee "$OUT/report_exp033.txt" ;;
  *) echo "usage: $0 compute|report" ;;
esac
