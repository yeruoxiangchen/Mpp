#!/usr/bin/env bash
# Explicit user-run network/dependency stage. Existing MPP/ChartQA envs are untouched.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
QWEN_DRIVE_ENV="${QWEN_DRIVE_ENV:-$PROJECT_ROOT/.venvs/qwen-drive}"
QWEN_DRIVE_SOURCE="${QWEN_DRIVE_SOURCE:-$PROJECT_ROOT/third_party/Qwen-Drive-1.0}"
QWEN_DRIVE_REV="28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0"
if [[ ! -d "$QWEN_DRIVE_SOURCE/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/QwenLM/Qwen-Drive-1.0.git "$QWEN_DRIVE_SOURCE"
fi
if [[ "$(git -C "$QWEN_DRIVE_SOURCE" rev-parse HEAD)" != "$QWEN_DRIVE_REV" ]]; then
  git -C "$QWEN_DRIVE_SOURCE" fetch --depth 1 origin "$QWEN_DRIVE_REV"
  git -C "$QWEN_DRIVE_SOURCE" checkout --detach "$QWEN_DRIVE_REV"
fi
git -C "$QWEN_DRIVE_SOURCE" sparse-checkout set --no-cone \
  /src/ /scripts/ /docs/ /README.md /LICENSE /pyproject.toml /requirements.txt
if [[ ! -x "$QWEN_DRIVE_ENV/bin/python" ]]; then
  "${SETUP_PYTHON:-python3}" -m venv "$QWEN_DRIVE_ENV"
fi
"$QWEN_DRIVE_ENV/bin/python" -m pip install --upgrade pip
"$QWEN_DRIVE_ENV/bin/python" -m pip install -r requirements/qwen_drive_vqa.txt
# Import directly from the pinned source directory. VQA uses SDPA; flash-attn,
# the perception CUDA extensions and the planner assets are unnecessary.
PYTHONPATH="$QWEN_DRIVE_SOURCE/src" "$QWEN_DRIVE_ENV/bin/python" -c \
  'import torch, transformers, qwen_drive; print("torch", torch.__version__, "transformers", transformers.__version__, "qwen_drive", qwen_drive.__version__)'
printf '%s\n' "Setup complete. Next: bash scripts/run_qwen_drive_teacher.sh download"
