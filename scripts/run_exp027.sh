#!/usr/bin/env bash
# EXP-027 / H018: z-choice headroom + calibration decomposition.  Sequential, one dataset at a time.
#   bash scripts/run_exp027.sh smoke      # 20 random images of ClinicDB, no recorded checks
#   bash scripts/run_exp027.sh full       # all six datasets, then the aggregate
#   bash scripts/run_exp027.sh agg        # aggregate only
set -euo pipefail
A=/home/ai3/NguyenND/AnomalyClipV2/data
CVC=$A/CVC
CKPT=${CKPT:-checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}
REF_ROOT=${REF_ROOT:-results/EXP-012/matched-retry-20261002}      # <ds>/ecp_extent/pixel_per_image_predictions.csv
OUT=${OUT:-results/EXP-027}; mkdir -p "$OUT"
PY=${PYTHON:-python}
run () {  # name flag path
  $PY analyze_zens.py eval --name "$1" --dataset "$2" --data_path "$3" --checkpoint_path "$CKPT" \
     --reference_csv "$REF_ROOT/$1/ecp_extent/pixel_per_image_predictions.csv" --out_csv "$OUT/zens_$1.csv" ${EXTRA:-} 2>&1 | tee "$OUT/zens_$1.log"
}
agg () { $PY analyze_zens.py --aggregate $OUT/zens_ClinicDB.csv $OUT/zens_Kvasir.csv $OUT/zens_ColonDB.csv $OUT/zens_ISIC.csv $OUT/zens_Endo.csv $OUT/zens_TN3K.csv | tee $OUT/zens_aggregate.txt; }
case "${1:-}" in
  smoke) OUT=$OUT/smoke; mkdir -p $OUT; EXTRA="--limit 20" run ClinicDB colon $CVC/CVC-ClinicDB ;;
  full)
    run ClinicDB colon $CVC/CVC-ClinicDB
    run ColonDB  colon $CVC/CVC-ColonDB
    run Endo     colon "$A/EndoTect_2020_Segmentation_Test_Dataset"
    run ISIC     ISBI  "$A/ISIC"
    run TN3K     thyroid "$A/TN3K/Thyroid Dataset/tn3k"
    run Kvasir   colon $CVC/Kvasir
    agg ;;
  agg) agg ;;
  *) echo "usage: $0 smoke|full|agg"; exit 1 ;;
esac
