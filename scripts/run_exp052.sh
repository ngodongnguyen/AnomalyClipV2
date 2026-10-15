#!/usr/bin/env bash
# EXP-052 / H041: source-only operating points of the extent axis.  z_pix = P95 of the training-mixture extent labels of the source (CPU, no model);
# z_img = the grid z with the best macro image AUROC on the OTHER industrial dataset (VisA for MVTec-trained ECP, MVTec for VisA-trained ECP).
# No medical label is used to choose anything.  Rule: notes/EXP-052_registration.md.  Needs conda env nguyennd for the GPU steps.
#   bash scripts/run_exp052.sh check      # unit tests + input listing
#   bash scripts/run_exp052.sh quantiles  # CPU, MVTec (EXP-040 parity) and VisA -> results/EXP-052/quantiles_{mvtec,visa}.json
#   bash scripts/run_exp052.sh parity     # GPU, HeadCT: imgscan scores at z -0.69 must equal test.py's exported image scores (ECP s111)
#   bash scripts/run_exp052.sh imgscan    # GPU, ~10 min per checkpoint: ECP s111/s222/s333 on VisA, ecp_visa on MVTec -> selection.json each
#   bash scripts/run_exp052.sh eval       # GPU, ~85 min per checkpoint: run_ecp.sh fixed_ckpt at the rule (z_img, z_pix) -> results/<ckpt>_srcz_fixed
#   bash scripts/run_exp052.sh report     # rules R1 / R2 / R3 and the VisA image-head check (no GPU)
# Resumable: existing quantile / scan / evaluation outputs are skipped (REDO=1 recomputes quantiles and scans).  No nohup inside.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
MVTEC=${MVTEC:-$A/mvtec}
VISA=${VISA:-$A/visa}
PY=${PYTHON:-python}
TAGS=${TAGS:-"ecp_s111 ecp_s222 ecp_s333 ecp_visa"}
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"
REDO_FLAG=""; [ "${REDO:-0}" = "1" ] && REDO_FLAG="--redo"
mkdir -p results/EXP-052 logs_exp052

ck () { case "$1" in ecp_s111) echo "EXP-012-matched/ecp_extent/checkpoints" ;; ecp_s222|ecp_s333) echo "matched_ecp_s${1#ecp_s}" ;; ecp_visa) echo "ecp_visa" ;; *) echo "?"; return 1 ;; esac; }
other () { case "$1" in ecp_visa) echo "mvtec|$MVTEC" ;; *) echo "visa|$VISA" ;; esac; }
done_eval () {  # all nine fixed_ckpt logs end with a mean row
  local out="results/$(ck "$1")_srcz_fixed" k
  for k in CVC-ClinicDB Kvasir CVC-ColonDB isic endo tn3k HeadCT_anomaly_detection BrainMRI br35; do
    tail -n 3 "$out/$k/log.txt" 2>/dev/null | grep -q "^| mean" || return 1
  done
}
case "${1:-}" in
  check)
    $PY test_zsel_stats.py; $PY test_centre_prior_stats.py
    for p in "$MVTEC/meta.json" "$VISA/meta.json"; do [ -f "$p" ] && echo "  $p OK" || echo "  $p MISSING"; done
    for t in $TAGS; do p="checkpoints/$(ck $t)/epoch_15.pth"; [ -f "$p" ] && echo "  $t -> $p OK" || echo "  $t -> $p MISSING"; done
    for r in ecp_visa zoom_visa ctrl_visa; do [ -f "results/${r}_fixed/HeadCT_anomaly_detection/log.txt" ] && echo "  EXP-020 reference $r OK" || echo "  EXP-020 reference $r MISSING (VisA comparison will print '-')"; done ;;
  quantiles)
    $PY analyze_zsel.py quantiles --dataset mvtec --data_root "$MVTEC" $REDO_FLAG
    $PY analyze_zsel.py quantiles --dataset visa --data_root "$VISA" $REDO_FLAG ;;
  parity)
    out=results/EXP-052/parity/HeadCT
    if [ ! -f "$out/image_level_predictions.csv" ]; then
      CUDA_VISIBLE_DEVICES=0 $PY test.py --dataset brain --data_path $A/HeadCT_anomaly_detection --save_path "$out" \
        --checkpoint_path checkpoints/$(ck ecp_s111)/epoch_15.pth $COMMON --metrics image-level --ec_z_img -0.69 --ec_z_pix 1.6 --export_image_scores > logs_exp052/parity_testpy.log 2>&1
    fi
    $PY analyze_zsel.py imgscan --tag ecp_s111 --suffix parity_headct --checkpoint_path checkpoints/$(ck ecp_s111)/epoch_15.pth --dataset brain \
      --data_path $A/HeadCT_anomaly_detection --parity_csv "$out/image_level_predictions.csv" --redo 2>&1 | { grep -v "^single_para\|^text_layer\|^Initializing\|^embedding_pos\|^tokenized\|^name " || true; } ;;
  imgscan)
    for t in $TAGS; do
      o=$(other $t)
      $PY analyze_zsel.py imgscan --tag $t --checkpoint_path checkpoints/$(ck $t)/epoch_15.pth --dataset ${o%%|*} --data_path "${o#*|}" $REDO_FLAG 2>&1 \
        | { grep -v "^single_para\|^text_layer\|^Initializing\|^embedding_pos\|^tokenized\|^name " || true; } | tee logs_exp052/imgscan_$t.log
    done ;;
  eval)
    for t in $TAGS; do
      read zi zp < <($PY analyze_zsel.py rule --tag $t)
      echo "=== $t ($(ck $t)): z_img $zi  z_pix $zp ==="
      if done_eval $t; then echo "  already evaluated, skipped"; continue; fi
      bash run_ecp.sh fixed_ckpt "$(ck $t)" "$zi" "$zp" "" _srcz > logs_exp052/eval_$t.log 2>&1
      done_eval $t && echo "  done" || echo "  INCOMPLETE (see logs_exp052/eval_$t.log)"
    done ;;
  report) $PY analyze_zsel.py report | tee results/EXP-052/report_exp052.txt ;;
  *) echo "usage: $0 check|quantiles|parity|imgscan|eval|report" ;;
esac
