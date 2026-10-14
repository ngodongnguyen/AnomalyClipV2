#!/bin/bash
# EXP-044: TokenCLIP-style prompt subspaces with OT routing on top of ECP (TOK, Q=3) and the capacity control TOKQ1 (Q=1). One seed (111), sequential, resumable.
#   bash scripts/run_exp044.sh check                  # local/server static checks and unit tests (no GPU)
#   bash scripts/run_exp044.sh smoke                  # 1-epoch TOK training + a pixel-auroc evaluation of the smoke checkpoint on Endo (exercises the OT test path)
#   nohup bash scripts/run_exp044.sh train > par_exp044_train.log 2>&1 &   # TOK then TOKQ1, ~1h40 each (plus OT overhead), skips finished checkpoints
#   nohup bash scripts/run_exp044.sh eval  > par_exp044_eval.log 2>&1 &    # 9 sets x 2 models at sigma 4 (~85 min per model), skips finished sets
#   bash scripts/run_exp044.sh eval32                 # OPTIONAL, report-only: the six pixel sets of both models at sigma 32 (~85 min per model)
#   python scripts/collect_exp044.py                  # the pre-registered rule + TOK - TOKQ1
# Matched ECP seed 111 numbers are read from the existing result files (same as collect_exp024), with a hard-coded fallback.
cd "$(dirname "$0")/.."
A=/home/ai3/NguyenND/AnomalyClipV2/data
COMMON="--features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4"
mkdir -p logs_exp044
ev () {  # $1 checkpoint, $2 --dataset, $3 data path, $4 result folder, $5 metrics, $6 sigma (default 4), $7 results tag (default _fixed)
  local sig=${6:-4} tag=${7:-_fixed} out=./results/$1${7:-_fixed}/$4
  if tail -n 3 $out/log.txt 2>/dev/null | grep -q "^| mean"; then return 0; fi
  [ -d $out ] && mv $out $out.partial.$(date +%s)
  EXP=""; [ "$5" = "pixel-level" ] && EXP="--export_pixel_metrics"; [ "$5" = "image-level" ] && EXP="--export_image_scores"
  CUDA_VISIBLE_DEVICES=0 python test.py --dataset $2 --data_path "$3" --save_path $out \
    --checkpoint_path ./checkpoints/$1/epoch_15.pth $COMMON --metrics $5 --ec_z_img -0.69 --ec_z_pix 1.6 --sigma $sig $EXP > logs_exp044/$1_$4_s$sig.log 2>&1
}
pixel_sets () {  # $1 checkpoint, $2 sigma, $3 tag
  ev $1 colon $A/CVC/Kvasir Kvasir pixel-level $2 $3; ev $1 colon $A/CVC/CVC-ClinicDB CVC-ClinicDB pixel-level $2 $3
  ev $1 colon $A/CVC/CVC-ColonDB CVC-ColonDB pixel-level $2 $3; ev $1 ISBI $A/ISIC isic pixel-level $2 $3
  ev $1 colon $A/EndoTect_2020_Segmentation_Test_Dataset endo pixel-level $2 $3; ev $1 thyroid "$A/TN3K/Thyroid Dataset/tn3k" tn3k pixel-level $2 $3
}
case "${1:-}" in
  check)
    python -m py_compile token_prompt.py train.py test.py scripts/collect_exp044.py scripts/test_collect_exp044.py
    bash -n scripts/run_exp044.sh; bash -n run_ecp.sh
    python test_token_prompt.py
    python scripts/test_collect_exp044.py ;;
  smoke)
    if [ ! -f checkpoints/tok_smoke/epoch_1.pth ]; then bash run_ecp.sh smoke_tok > logs_exp044/smoke_tok.log 2>&1; fi
    tail -n 4 logs_exp044/smoke_tok.log
    python -c "import torch; c=torch.load('checkpoints/tok_smoke/epoch_1.pth', map_location='cpu'); assert 'tok_heads' in c and c['tok_cfg']['q']==3; print('checkpoint has tok_heads, cfg', c['tok_cfg'])"
    CUDA_VISIBLE_DEVICES=0 python test.py --dataset colon --data_path $A/EndoTect_2020_Segmentation_Test_Dataset --save_path ./results/tok_smoke_fixed/endo \
      --checkpoint_path ./checkpoints/tok_smoke/epoch_1.pth $COMMON --metrics pixel-auroc --ec_z_img -0.69 --ec_z_pix 1.6 --sigma 4 2>&1 | tail -n 8 ;;
  train)
    for n in matched_tok_s111:train_m_tok matched_tokq1_s111:train_m_tokq1; do
      ck=${n%%:*}; fn=${n##*:}
      if [ ! -f checkpoints/$ck/epoch_15.pth ]; then bash run_ecp.sh $fn > logs_exp044/train_$ck.log 2>&1; fi
      if [ -f checkpoints/$ck/epoch_15.pth ]; then echo "TRAINED $ck"; else echo "MISSING $ck (see logs_exp044/train_$ck.log)"; fi
    done ;;
  eval)
    for m in matched_tok_s111 matched_tokq1_s111; do
      if [ ! -f checkpoints/$m/epoch_15.pth ]; then echo "MISSING checkpoints/$m (run train first)"; continue; fi
      pixel_sets $m 4 _fixed
      ev $m brain $A/HeadCT_anomaly_detection HeadCT_anomaly_detection image-level 4 _fixed; ev $m brain $A/BrainMRI BrainMRI image-level 4 _fixed
      ev $m brain $A/br35 br35 image-level 4 _fixed
    done
    echo EXP044 EVAL DONE ;;
  eval32)
    for m in matched_tok_s111 matched_tokq1_s111; do pixel_sets $m 32 _sigma32_fixed; done
    echo EXP044 SIGMA32 DONE ;;
  collect) python scripts/collect_exp044.py ;;
  *) echo "usage: $0 check|smoke|train|eval|eval32|collect" ;;
esac
