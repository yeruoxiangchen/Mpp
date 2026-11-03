#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DIAG_DIR="${DIAG_DIR:-outputs/full_v2_action_diagnosis_v1}"
model="${1:?usage: run_full_sft_diagnosis.sh base|old|new}"
adapter=()
case "$model" in
  base) ;;
  old) adapter=(--adapter outputs/mpp_lingoqa_lora_gc_full_v1/final_adapter) ;;
  new) adapter=(--adapter outputs/mpp_lingoqa_full_v2_lora_1epoch/final_adapter) ;;
  *) exit 2 ;;
esac
export PYTHONPATH=".:third_party/MPP-LLaVA"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
"$PYTHON_BIN" -u scripts/generate_answers.py \
  --data "$DIAG_DIR/diagnostic.jsonl" \
  --cache-dir "${LINGO_CACHE_DIR:-$PROJECT_ROOT/datasets/cache/lingoqa_5f_mpp}" \
  --upstream-root third_party/MPP-LLaVA \
  --checkpoint-root "${MPP_CHECKPOINT_ROOT:-$MPP_DATA_ROOT/checkpoints}" \
  --task-instruction-version legacy_v1 --batch-size 1 --max-new-tokens 96 \
  --output-dir "$DIAG_DIR/$model" "${adapter[@]}"
"$PYTHON_BIN" -u scripts/evaluate_answers.py \
  --predictions "$DIAG_DIR/$model/predictions.jsonl" \
  --output-dir "$DIAG_DIR/$model/scored" --lingo-judge
