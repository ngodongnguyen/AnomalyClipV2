#!/bin/bash
# EXP-046 / H034: ECP trained AND tested with the top-50% V-V head mask (HEAD50) versus the EXP-045 random-50% mask of seed 0 (RAND50). One seed (111), foreground, resumable.
#   bash scripts/run_exp046.sh check     # compile + bash -n + CPU unit tests (needs torch for the integration test)
#   bash scripts/run_exp046.sh masks     # results/EXP-045/head_scores_mvtec.npy -> results/EXP-046/{head50,rand50_s0}_mask.npy (+ equality with the EXP-045 masks, sha256)
#   bash scripts/run_exp046.sh smoke     # 1-epoch HEAD50 training + pixel-auroc on Endo of the smoke checkpoint; prints the mask hash read back from the checkpoint
#   bash scripts/run_exp046.sh train     # HEAD50 then RAND50 (~1h40 each), skips finished checkpoints
#   bash scripts/run_exp046.sh eval      # 6 pixel sets + 3 image sets at sigma 4, z_img=-0.69, z_pix=1.6, both checkpoints (~85 min each), skips finished sets
#   bash scripts/run_exp046.sh collect   # the pre-registered rule
# Evaluation reads the mask from the checkpoint (test.py refuses a --vv_head_mask that differs and runs masked even if the flag is absent).
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
PY=${PYTHON:-python}
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"
mkdir -p logs_exp046
ev () {  # $1 checkpoint, $2 --dataset, $3 data path, $4 result folder, $5 metrics
  local out=./results/$1_fixed/$4
  if tail -n 3 $out/log.txt 2>/dev/null | grep -q "^| mean"; then return 0; fi
  [ -d $out ] && mv $out $out.partial.$(date +%s)
  EXP=""; [ "$5" = "pixel-level" ] && EXP="--export_pixel_metrics"; [ "$5" = "image-level" ] && EXP="--export_image_scores"
  echo "[eval] $1 $4"
  CUDA_VISIBLE_DEVICES=0 $PY test.py --dataset $2 --data_path "$3" --save_path $out \
    --checkpoint_path ./checkpoints/$1/epoch_15.pth $COMMON --metrics $5 --ec_z_img -0.69 --ec_z_pix 1.6 --sigma 4 $EXP 2>&1 | tee logs_exp046/$1_$4.log | tail -n 3
}
case "${1:-}" in
  check)
    $PY -m py_compile vvattn_core.py vvattn_stats.py train.py test.py scripts/make_head_masks_exp046.py scripts/collect_exp046.py scripts/test_collect_exp046.py test_vv_head_mask_integration.py
    bash -n scripts/run_exp046.sh; bash -n run_ecp.sh
    $PY test_vvattn_stats.py | tail -n 1
    $PY test_vvattn_core.py | tail -n 1
    $PY scripts/test_make_head_masks_exp046.py
    $PY test_vv_head_mask_integration.py | tail -n 1
    $PY scripts/test_collect_exp046.py
    echo "EXP-046 CHECK PASSED" ;;
  masks)
    $PY scripts/make_head_masks_exp046.py ;;
  smoke)
    for m in head50 rand50_s0; do [ -f results/EXP-046/${m}_mask.npy ] || { echo "missing results/EXP-046/${m}_mask.npy: run 'masks' first"; exit 1; }; done
    if [ ! -f checkpoints/head_smoke/epoch_1.pth ]; then bash run_ecp.sh smoke_head 2>&1 | tee logs_exp046/smoke_head.log | tail -n 5; fi
    $PY - <<'PYEOF'
import torch, hashlib, numpy as np
c = torch.load('checkpoints/head_smoke/epoch_1.pth', map_location='cpu')
r = c['vv_head_mask']
f = hashlib.sha256(open('results/EXP-046/head50_mask.npy', 'rb').read()).hexdigest()
assert r['sha256'] == f and r['n_keep'] == 152 and np.array(r['keep']).shape == (19, 16)
print('checkpoint vv_head_mask: path', r['path'], 'sha256', r['sha256'], 'keeps', r['n_keep'], '/ 304 (file sha256 matches)')
PYEOF
    CUDA_VISIBLE_DEVICES=0 $PY test.py --dataset colon --data_path $A/EndoTect_2020_Segmentation_Test_Dataset --save_path ./results/head_smoke_fixed/endo \
      --checkpoint_path ./checkpoints/head_smoke/epoch_1.pth $COMMON --metrics pixel-auroc --ec_z_img -0.69 --ec_z_pix 1.6 --sigma 4 2>&1 | tee logs_exp046/smoke_test.log | grep -i "head mask\|mean" | tail -n 6 ;;
  train)
    for n in matched_head50_s111:train_m_head50 matched_rand50_s111:train_m_rand50; do
      ck=${n%%:*}; fn=${n##*:}
      if [ ! -f checkpoints/$ck/epoch_15.pth ]; then bash run_ecp.sh $fn 2>&1 | tee logs_exp046/train_$ck.log | tail -n 3; fi
      if [ -f checkpoints/$ck/epoch_15.pth ]; then echo "TRAINED $ck"; else echo "MISSING $ck (see logs_exp046/train_$ck.log)"; fi
    done ;;
  eval)
    for m in matched_head50_s111 matched_rand50_s111; do
      if [ ! -f checkpoints/$m/epoch_15.pth ]; then echo "MISSING checkpoints/$m (run train first)"; continue; fi
      ev $m colon $A/CVC/Kvasir Kvasir pixel-level; ev $m colon $A/CVC/CVC-ClinicDB CVC-ClinicDB pixel-level
      ev $m colon $A/CVC/CVC-ColonDB CVC-ColonDB pixel-level; ev $m ISBI $A/ISIC isic pixel-level
      ev $m colon $A/EndoTect_2020_Segmentation_Test_Dataset endo pixel-level; ev $m thyroid "$A/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level
      ev $m brain $A/HeadCT_anomaly_detection HeadCT_anomaly_detection image-level; ev $m brain $A/BrainMRI BrainMRI image-level; ev $m brain $A/br35 br35 image-level
    done
    echo EXP046 EVAL DONE ;;
  collect) $PY scripts/collect_exp046.py ;;
  *) echo "usage: $0 check|masks|smoke|train|eval|collect" ;;
esac
