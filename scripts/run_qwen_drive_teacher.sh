#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
if [[ "${1:-}" == "setup" ]]; then
  shift
  exec bash scripts/setup_qwen_drive_teacher.sh "$@"
fi
exec "${PYTHON_BIN:-python}" -u \
  scripts/qwen_drive_screen.py "$@"
