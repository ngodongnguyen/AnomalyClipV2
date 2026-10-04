#!/usr/bin/env bash
# Frozen EXP-017 visual phase diagnostic. On the GPU server set PYTHON,
# DATA_ROOT, CLIP_WEIGHTS, then run: smoke; pilot; full only if pilot gate passes.
# Reuses the verified EXP-012 ECP checkpoint and paired reference artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-}"
case "$STAGE" in smoke|pilot|full) ;; *) echo 'Usage: bash scripts/run_exp017_phase_probe.sh {smoke|pilot|full}' >&2; exit 2;; esac
fail() { echo "EXP-017 error: $*" >&2; exit 1; }

: "${PYTHON:?set PYTHON to the server Conda interpreter}"
: "${DATA_ROOT:?set DATA_ROOT to the server data directory}"
: "${CLIP_WEIGHTS:?set CLIP_WEIGHTS to the CLIP cache weights file}"
CHECKPOINT="${EXP017_CHECKPOINT:-$ROOT/checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}"
MANIFEST="${EXP017_MANIFEST:-$ROOT/checkpoints/EXP-012-matched/ecp_extent/training_manifest.json}"
REFERENCE_ROOT="${EXP017_REFERENCE_ROOT:-$ROOT/results/EXP-012/matched-retry-20261002}"
OUT_ROOT="${EXP017_OUT_ROOT:-$ROOT/results/EXP-017/phase-pilot}"
[[ -x "$PYTHON" ]] || fail "invalid PYTHON: $PYTHON"
for path in "$CHECKPOINT" "$MANIFEST" "$CLIP_WEIGHTS" "$DATA_ROOT/mvtec/meta.json"; do
  [[ -f "$path" ]] || fail "missing required file: $path"
done
[[ -d "$REFERENCE_ROOT" ]] || fail "missing EXP-012 reference: $REFERENCE_ROOT"
for path in "$ROOT/scripts/exp017_phase_probe.py" "$ROOT/scripts/exp017_phase_core.py" "$ROOT/scripts/exp014_source_readout.py" "$ROOT/scripts/exp014_readout_core.py"; do
  [[ -f "$path" ]] || fail "missing script dependency: $path"
done

NAMES=(ClinicDB)
FOLDERS=(CVC/CVC-ClinicDB)
if [[ "$STAGE" != smoke ]]; then
  NAMES=(ClinicDB ColonDB Kvasir)
  FOLDERS=(CVC/CVC-ClinicDB CVC/CVC-ColonDB CVC/Kvasir)
fi
if [[ "$STAGE" == full ]]; then
  NAMES=(ISIC ClinicDB ColonDB Kvasir Endo TN3K)
  FOLDERS=(ISIC CVC/CVC-ClinicDB CVC/CVC-ColonDB CVC/Kvasir EndoTect_2020_Segmentation_Test_Dataset 'TN3K/Thyroid Dataset/tn3k')
fi
for i in "${!NAMES[@]}"; do
  [[ -f "$DATA_ROOT/${FOLDERS[$i]}/meta.json" ]] || fail "${NAMES[$i]} split missing"
  BASE="$REFERENCE_ROOT/${NAMES[$i]}/ecp_extent"
  [[ -f "$BASE/evaluation_metadata.json" && -f "$BASE/pixel_per_image_predictions.csv" ]] ||
    fail "${NAMES[$i]} EXP-012 reference missing"
done

OUT="$OUT_ROOT/$STAGE"
[[ ! -e "$OUT" ]] || fail "output already exists: $OUT"
mkdir -p "$OUT_ROOT/logs"
LOG="$OUT_ROOT/logs/$STAGE.stdout_stderr.log"
COMMAND="$OUT_ROOT/logs/$STAGE.command.txt"
[[ ! -e "$LOG" && ! -e "$COMMAND" ]] || fail "stage $STAGE already started; inspect logs before retry"
CMD=("$PYTHON" "$ROOT/scripts/exp017_phase_probe.py" "$STAGE"
  --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --clip-weights "$CLIP_WEIGHTS"
  --data-root "$DATA_ROOT" --reference-root "$REFERENCE_ROOT" --out "$OUT")
printf '%q ' "${CMD[@]}" > "$COMMAND"
printf '\n' >> "$COMMAND"
echo "EXP-017 $STAGE running; log: $LOG"
cd "$ROOT"
"${CMD[@]}" > "$LOG" 2>&1 || fail "$STAGE failed; inspect $LOG"
[[ -s "$OUT/summary.json" && -s "$OUT/metadata.json" ]] || fail "missing $STAGE artifacts"
echo "EXP-017 $STAGE completed: $OUT"
