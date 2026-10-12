#!/usr/bin/env bash
# EXP-040: source-only class-conditional zoom-cue diagnostic.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-/home/ai3/NguyenND/AnomalyClipV2/data}
GPU=${GPU:-0}
CTRL_CKPT=${CTRL_CKPT:-checkpoints/matched_ctrl_s111/epoch_15.pth}
ZO_CKPT=${ZO_CKPT:-checkpoints/EXP-012-matched/zoom_only/checkpoints/epoch_15.pth}
OUT=results/EXP-040
ARGS=(--data-root "$DATA_ROOT/mvtec" --output "$OUT"
      --ctrl-ckpt "$CTRL_CKPT" --zo-ckpt "$ZO_CKPT")

case "${1:-}" in
  check)
    for path in "$DATA_ROOT/mvtec/meta.json" "$CTRL_CKPT" "$ZO_CKPT"; do
      [ -f "$path" ] || { echo "Missing $path" >&2; exit 1; }
    done
    python -m unittest discover -s scripts -p test_exp040_core.py -v
    ;;
  audit)
    python scripts/exp040_probe.py audit "${ARGS[@]}"
    ;;
  smoke)
    [ -f "$OUT/geometry.json" ] || { echo 'Run audit first' >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/exp040_probe.py smoke "${ARGS[@]}"
    ;;
  probe)
    [ -f "$OUT/geometry.json" ] || { echo 'Run audit first' >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/exp040_probe.py probe "${ARGS[@]}"
    ;;
  report)
    python scripts/collect_exp040.py --root "$OUT"
    ;;
  *) echo 'usage: bash scripts/run_exp040.sh check|audit|smoke|probe|report' >&2; exit 2 ;;
esac
