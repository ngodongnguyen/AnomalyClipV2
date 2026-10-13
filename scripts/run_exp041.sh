#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:-check}"
PYTHON="${PYTHON:-python}"
EXP040_ROOT="${EXP040_ROOT:-results/EXP-040}"
EXP041_ROOT="${EXP041_ROOT:-results/EXP-041}"
DATA_ROOT="${DATA_ROOT:-/home/ai3/NguyenND/AnomalyClipV2/data/mvtec}"

case "$PHASE" in
  check)
    "$PYTHON" -m unittest discover -s scripts -p 'test_exp041.py' -v
    ;;
  analyze)
    "$PYTHON" scripts/exp041.py analyze --exp040 "$EXP040_ROOT" --output "$EXP041_ROOT"
    ;;
  export)
    "$PYTHON" scripts/exp041.py export --exp040 "$EXP040_ROOT" --output "$EXP041_ROOT" --data-root "$DATA_ROOT"
    ;;
  collect)
    "$PYTHON" scripts/exp041.py collect --output "$EXP041_ROOT"
    ;;
  *)
    echo 'Usage: bash scripts/run_exp041.sh {check|analyze|export|collect}' >&2
    exit 2
    ;;
esac
