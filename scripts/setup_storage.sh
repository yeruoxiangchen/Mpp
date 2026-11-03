#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$MPP_DATA_ROOT" != /* ]]; then
  echo 'MPP_DATA_ROOT must be an absolute path.' >&2
  exit 2
fi
mkdir -p "$MPP_DATA_ROOT" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
  "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_EXTENSIONS_DIR" \
  "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
for name in datasets checkpoints outputs logs .venvs; do
  target="$MPP_DATA_ROOT/$name"
  if [[ "$name" == .venvs ]]; then target="$MPP_DATA_ROOT/env"; fi
  mkdir -p "$target"
  link="$PROJECT_ROOT/$name"
  if [[ -L "$link" ]]; then
    if [[ "$(readlink -f "$link")" != "$(readlink -f "$target")" ]]; then
      echo "Existing storage link points elsewhere: $link" >&2
      exit 2
    fi
  elif [[ -e "$link" ]]; then
    echo "Existing path requires manual storage configuration: $link" >&2
    exit 2
  else
    ln -s "$target" "$link"
  fi
done
printf 'MPP runtime storage: %s\n' "$MPP_DATA_ROOT"
