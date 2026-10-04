#!/usr/bin/env bash
# EXP-014 source-matched visual-branch diagnostic on the GPU server.
# Usage: set PYTHON, DATA_ROOT, CLIP_WEIGHTS; then run fit, smoke, full in order.
# No AnomalyCLIP training or model-code changes. Existing EXP-012 matched
# checkpoint, training manifest, and 12-run reference outputs are required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-}"
case "$STAGE" in fit|smoke|full) ;; *) echo 'Usage: bash scripts/run_exp014_source_readout.sh {fit|smoke|full}' >&2; exit 2;; esac
fail() { echo "EXP-014 source-readout error: $*" >&2; exit 1; }

: "${PYTHON:?set PYTHON to the server Conda environment interpreter}"
: "${DATA_ROOT:?set DATA_ROOT to the server data directory}"
: "${CLIP_WEIGHTS:?set CLIP_WEIGHTS to the CLIP cache weights file}"
CHECKPOINT="${EXP014_CHECKPOINT:-$ROOT/checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}"
MANIFEST="${EXP014_MANIFEST:-$ROOT/checkpoints/EXP-012-matched/ecp_extent/training_manifest.json}"
REFERENCE_ROOT="${EXP014_REFERENCE_ROOT:-$ROOT/results/EXP-012/matched-retry-20261002}"
OUT_ROOT="${EXP014_OUT_ROOT:-$ROOT/results/EXP-014/source-matched}"
FITTED="$OUT_ROOT/source_readout.json"
[[ -x "$PYTHON" ]] || fail "Python interpreter absent or not executable: $PYTHON"
[[ -f "$CHECKPOINT" && -f "$MANIFEST" && -f "$CLIP_WEIGHTS" ]] || fail 'checkpoint, manifest, or CLIP weights missing'
[[ -f "$DATA_ROOT/mvtec/meta.json" ]] || fail 'MVTec meta.json missing'
[[ -d "$REFERENCE_ROOT" ]] || fail 'EXP-012 reference root missing'
[[ -f "$ROOT/scripts/exp014_source_readout.py" && -f "$ROOT/scripts/exp014_readout_core.py" ]] || fail 'probe scripts missing'
if [[ "$STAGE" != fit ]]; then
  NAMES=(ClinicDB)
  FOLDERS=(CVC/CVC-ClinicDB)
  if [[ "$STAGE" == full ]]; then
    NAMES=(ISIC ClinicDB ColonDB Kvasir Endo TN3K)
    FOLDERS=(ISIC CVC/CVC-ClinicDB CVC/CVC-ColonDB CVC/Kvasir EndoTect_2020_Segmentation_Test_Dataset 'TN3K/Thyroid Dataset/tn3k')
  fi
  for i in "${!NAMES[@]}"; do
    [[ -f "$DATA_ROOT/${FOLDERS[$i]}/meta.json" ]] || fail "${NAMES[$i]} meta.json missing"
    BASE="$REFERENCE_ROOT/${NAMES[$i]}/ecp_extent"
    [[ -f "$BASE/evaluation_metadata.json" && -f "$BASE/pixel_per_image_predictions.csv" ]] ||
      fail "${NAMES[$i]} EXP-012 paired reference missing"
  done
fi

if [[ "$STAGE" == fit ]]; then
  [[ ! -e "$FITTED" ]] || fail "source readout already exists: $FITTED"
  OUT="$FITTED"
else
  [[ -s "$FITTED" ]] || fail 'fit the source readout first'
  OUT="$OUT_ROOT/$STAGE"
  [[ ! -e "$OUT" ]] || fail "output already exists: $OUT"
fi
if [[ "$STAGE" == full ]]; then
  [[ -s "$OUT_ROOT/smoke/summary.json" ]] || fail 'run smoke successfully before full'
  "$PYTHON" - "$OUT_ROOT/smoke/summary.json" <<'PY' || fail 'smoke parity check failed'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary['stage'] == 'smoke'
assert summary['max_vv_reference_abs_error'] <= 1e-6
assert summary['counts']['ClinicDB'] == 2
PY
fi

mkdir -p "$OUT_ROOT/logs"
LOG="$OUT_ROOT/logs/$STAGE.stdout_stderr.log"
CMD="$OUT_ROOT/logs/$STAGE.command.txt"
[[ ! -e "$LOG" && ! -e "$CMD" ]] || fail "stage $STAGE already started; inspect existing logs before retrying"
CMD_ARGS=("$PYTHON" "$ROOT/scripts/exp014_source_readout.py" "$STAGE"
  --checkpoint "$CHECKPOINT" --manifest "$MANIFEST"
  --clip-weights "$CLIP_WEIGHTS" --data-root "$DATA_ROOT"
  --reference-root "$REFERENCE_ROOT" --fitted "$FITTED" --out "$OUT")
printf '%q ' "${CMD_ARGS[@]}" > "$CMD"
printf '\n' >> "$CMD"
echo "EXP-014 $STAGE running; log: $LOG"
cd "$ROOT"
"${CMD_ARGS[@]}" > "$LOG" 2>&1 || fail "$STAGE failed; inspect $LOG"
if [[ "$STAGE" == fit ]]; then
  [[ -s "$FITTED" ]] || fail 'fit did not write source readout'
else
  [[ -s "$OUT/summary.json" && -s "$OUT/metadata.json" ]] || fail "$STAGE artifacts missing"
fi
echo "EXP-014 $STAGE completed: $OUT"
