#!/usr/bin/env bash
# GPU server, no training. Set PYTHON, DATA_ROOT, CLIP_WEIGHTS.
# Run in order, each on its own line:
#   bash scripts/run_exp015_probe.sh calibrate
#   bash scripts/run_exp015_probe.sh smoke
#   bash scripts/run_exp015_probe.sh pilot
# Pilot uses three CVC datasets, 48 images each, expanding to 96 only if
# baseline/mask coverage is insufficient. It can require several GPU hours.
# Preserve failed outputs; set a fresh EXP015_ROOT and rerun stages to retry.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-}"
PYTHON="${PYTHON:-}"
DATA_ROOT="${DATA_ROOT:-}"
CLIP_WEIGHTS="${CLIP_WEIGHTS:-$HOME/.cache/clip/ViT-L-14-336px.pt}"
CHECKPOINT="${EXP015_CHECKPOINT:-$ROOT/checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}"
MANIFEST="${EXP015_MANIFEST:-$ROOT/checkpoints/EXP-012-matched/ecp_extent/training_manifest.json}"
REFERENCE_ROOT="${EXP015_REFERENCE_ROOT:-$ROOT/results/EXP-012/matched-retry-20261002}"
OUT_ROOT="${EXP015_ROOT:-$ROOT/results/EXP-015}"
fail() { printf 'EXP-015 error: %s\n' "$*" >&2; exit 1; }
[[ "$STAGE" == calibrate || "$STAGE" == smoke || "$STAGE" == pilot ]] ||
  fail 'usage: bash scripts/run_exp015_probe.sh calibrate|smoke|pilot'
[[ -n "$PYTHON" && -x "$PYTHON" ]] || fail 'set PYTHON to the existing project interpreter'
[[ -n "$DATA_ROOT" && -d "$DATA_ROOT" ]] || fail 'set DATA_ROOT to the server data directory'
[[ -f "$CLIP_WEIGHTS" && -f "$CHECKPOINT" && -f "$MANIFEST" ]] ||
  fail 'missing CLIP, matched ECP checkpoint, or training manifest'
[[ -d "$REFERENCE_ROOT" ]] || fail 'missing EXP-012 matched evaluation reference root'
cd "$ROOT"
if [[ "$STAGE" == calibrate ]]; then
  "$PYTHON" -m unittest discover -s tests -p test_exp015_probe_core.py -v ||
    fail 'EXP-015 synthetic tests failed'
fi
mkdir -p "$OUT_ROOT/logs"
LOG="$OUT_ROOT/logs/$STAGE-$(date +%Y%m%d-%H%M%S)-$$.log"
cmd=("$PYTHON" -u scripts/exp015_context_probe.py "$STAGE"
     --data-root "$DATA_ROOT" --checkpoint "$CHECKPOINT"
     --clip-weights "$CLIP_WEIGHTS" --manifest "$MANIFEST"
     --reference-root "$REFERENCE_ROOT" --out-root "$OUT_ROOT")
printf '%q ' "${cmd[@]}" > "${LOG%.log}.command.txt"
printf '\n' >> "${LOG%.log}.command.txt"
"${cmd[@]}" 2>&1 | tee "$LOG"
