#!/usr/bin/env bash
# Server: set PYTHON, DATA_ROOT, CLIP_WEIGHTS in the existing project environment.
# Run these three commands in order, each on its own line:
#   bash scripts/run_exp016.sh smoke
#   bash scripts/run_exp016.sh train
#   bash scripts/run_exp016.sh eval
# Trains ONLY small residuals on frozen ECP, three ablation arms, three epochs each.
# Keep EXP016_ROOT unchanged between stages; choose a fresh root for a failed smoke.
# Existing outputs are never overwritten. No AI tooling is required on the server.
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-}"
DATA_ROOT="${DATA_ROOT:-}"
CLIP_WEIGHTS="${CLIP_WEIGHTS:-$HOME/.cache/clip/ViT-L-14-336px.pt}"
EXP016_BASE_CHECKPOINT="${EXP016_BASE_CHECKPOINT:-$REPO_ROOT/checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth}"
EXP016_ROOT="${EXP016_ROOT:-$REPO_ROOT/results/EXP-016}"
EXP016_TRAIN_ROOT="${EXP016_TRAIN_ROOT:-$REPO_ROOT/checkpoints/EXP-016}"
EXP016_REFERENCE_ROOT="${EXP016_REFERENCE_ROOT:-$REPO_ROOT/results/EXP-012/matched-retry-20261002}"
STAGE="${1:-}"
fail() { printf 'EXP-016 error: %s\n' "$*" >&2; exit 1; }
[[ "$STAGE" == smoke || "$STAGE" == train || "$STAGE" == eval || "$STAGE" == summarize ]] || fail 'usage: bash scripts/run_exp016.sh smoke|train|eval|summarize'
[[ -n "$PYTHON" && -x "$PYTHON" ]] || fail 'set PYTHON to the existing project interpreter'
[[ -n "$DATA_ROOT" && -d "$DATA_ROOT" ]] || fail 'set DATA_ROOT to the server data directory'
[[ -f "$EXP016_BASE_CHECKPOINT" && -f "$CLIP_WEIGHTS" ]] || fail 'missing ECP checkpoint or CLIP weights'
cd "$REPO_ROOT"
mkdir -p "$EXP016_ROOT/logs"
LOG="$EXP016_ROOT/logs/${STAGE}-$(date +%Y%m%d-%H%M%S)-$$.log"
cmd=("$PYTHON" -u scripts/exp016_context_residual.py "$STAGE"
     --data-root "$DATA_ROOT" --checkpoint "$EXP016_BASE_CHECKPOINT"
     --clip-weights "$CLIP_WEIGHTS" --run-root "$EXP016_ROOT"
     --train-root "$EXP016_TRAIN_ROOT" --reference-root "$EXP016_REFERENCE_ROOT")
printf '%q ' "${cmd[@]}" > "${LOG%.log}.command.txt"
printf '\n' >> "${LOG%.log}.command.txt"
"${cmd[@]}" 2>&1 | tee "$LOG"
