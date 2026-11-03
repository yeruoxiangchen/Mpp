#!/usr/bin/env bash
# Source this file to select storage and cache locations; it creates no files.
_MPP_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPP_DATA_ROOT="${MPP_DATA_ROOT:-/data/${USER:-user}/mpp}"
export MPP_UPSTREAM_ROOT="${MPP_UPSTREAM_ROOT:-$_MPP_ENV_ROOT/third_party/MPP-LLaVA}"
export MPP_ASSET_ROOT="${MPP_ASSET_ROOT:-$MPP_DATA_ROOT}"
export MPP_PERSISTENT_CHECKPOINT_ROOT="${MPP_PERSISTENT_CHECKPOINT_ROOT:-$MPP_DATA_ROOT/checkpoints}"
export MPP_CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-$MPP_PERSISTENT_CHECKPOINT_ROOT}"
export MPP_VISUAL_CHECKPOINT_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$MPP_PERSISTENT_CHECKPOINT_ROOT}"
export MPP_BRIDGE_INIT="${MPP_BRIDGE_INIT:-$MPP_DATA_ROOT/checkpoints/visual_bridge_init.pt}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export TMPDIR="${TMPDIR:-$MPP_DATA_ROOT/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MPP_DATA_ROOT/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-$MPP_DATA_ROOT/cache/torch}"
export HF_HOME="${HF_HOME:-$MPP_DATA_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$MPP_DATA_ROOT/cache/torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$MPP_DATA_ROOT/cache/matplotlib}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MPP_DATA_ROOT/cache/pip}"
export PYTHONDONTWRITEBYTECODE=1
unset _MPP_ENV_ROOT
