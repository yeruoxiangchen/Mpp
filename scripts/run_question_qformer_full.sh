#!/usr/bin/env bash
# Complete original QA, bridge adaptation and joint SFT; four GPUs throughout.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/third_party/MPP-LLaVA}"
PERSISTENT="$MPP_PERSISTENT_CHECKPOINT_ROOT"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-$MPP_DATA_ROOT/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then CHECKPOINT_ROOT="$PERSISTENT"; fi
VISUAL_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
BRIDGE_INIT="${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
CACHE_DIR="${MPP_EVA_CACHE_DIR:-$PROJECT_ROOT/datasets/cache/lingoqa_5f_eva_v1}"
if [[ ! -d "$CACHE_DIR" ]]; then CACHE_DIR="$PROJECT_ROOT/datasets/cache/lingoqa_5f_eva_v1"; fi
DATA_DIR="${QUESTION_FULL_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_qformer_all_v1}"
INITIAL_DIR="${QUESTION_FULL_INITIAL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
RETAINED_DIR="${QUESTION_FULL_RETAINED_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_sft_control_40k_2epoch_v2}"
HISTORY_DIR="${QUESTION_FULL_HISTORY_DIR:-$PROJECT_ROOT/outputs/feedback_opd_eval_v2}"
TRAIN_DIR="${QUESTION_FULL_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_question_qformer_full_1epoch_v2}"
CONTROL_DIR="${QUESTION_FULL_CONTROL_DIR:-$PROJECT_ROOT/outputs/mpp_visual_qformer_full_1epoch_v2}"
EVAL_DIR="${QUESTION_FULL_EVAL_DIR:-$PROJECT_ROOT/outputs/question_qformer_full_eval_v2}"
FREEZE_STEPS="${QUESTION_FULL_FREEZE_STEPS:-1000}"
LANGUAGE_WARMUP="${QUESTION_FULL_LANGUAGE_WARMUP:-200}"
QUESTION_MAX_LEN="${QUESTION_FULL_MAX_LEN:-128}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29735}"
report_args=(--data-dir "$DATA_DIR" --initial-dir "$INITIAL_DIR" --retained-dir "$RETAINED_DIR"
  --history-dir "$HISTORY_DIR" --train-dir "$TRAIN_DIR" --control-dir "$CONTROL_DIR" --eval-dir "$EVAL_DIR"
  --bert-dir "$VISUAL_ROOT/ckpt/bert-base-uncased" --qwen-dir "$CHECKPOINT_ROOT/ckpt/Qwen7B-chat"
  --cache-dir "$CACHE_DIR" --freeze-steps "$FREEZE_STEPS" --language-warmup "$LANGUAGE_WARMUP"
  --question-max-len "$QUESTION_MAX_LEN")
report() { "$PYTHON_BIN" -u scripts/report_question_qformer_full.py "${report_args[@]}" --action "$1" "${@:2}"; }

train() {
  local branch="$1" output text resume
  local extra=() visible=()
  IFS=',' read -r -a visible <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "${#visible[@]}" != 4 ]]; then echo "This full-SFT recipe requires four visible GPUs." >&2; exit 2; fi
  for ((i=0; i<4; i++)); do
    for ((j=0; j<i; j++)); do
      if [[ "${visible[i]}" == "${visible[j]}" ]]; then echo "Repeated GPU ID." >&2; exit 2; fi
    done
  done
  report check
  if [[ "$branch" == question ]]; then
    output="$TRAIN_DIR"; text=question; resume="${QUESTION_FULL_RESUME_FROM:-}"
  else
    output="$CONTROL_DIR"; text=none; resume="${VISUAL_FULL_RESUME_FROM:-}"
  fi
  if [[ -n "$resume" ]]; then extra+=(--resume-from "$resume"); fi
  "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node=4 --master_port="$MASTER_PORT" \
    scripts/train_feedback_opd.py --training-mode sft_control --sft-data-contract original_full \
    --language-lora-scope qkv --qformer-text-input "$text" --qformer-question-max-len "$QUESTION_MAX_LEN" \
    --data "$DATA_DIR/train.jsonl" --output-dir "$output" --initial-dir "$INITIAL_DIR" \
    --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
    --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" \
    --task-instruction-version legacy_v1 --teacher-mode reference_only --seed 44 --cpu-threads 1 \
    --max-txt-len 1536 --max-new-tokens 96 --epochs 1 --max-steps 0 --grad-accumulation 4 \
    --save-every 0 --checkpoint-steps 5000 10000 15000 20000 \
    --lr 2.5e-6 --projector-lr 5e-6 --qformer-lr 1e-5 --qformer-text-lr 2e-5 \
    --language-freeze-steps "$FREEZE_STEPS" --language-warmup-steps "$LANGUAGE_WARMUP" \
    --weight-decay .01 --warmup-ratio .03 --ce-weight 1 "${extra[@]}"
}

evaluate() {
  local key="$1" split="$2" data_path destination bundle label allowed found=false
  local weights=()
  case "$split" in
    action|scenery) data_path="$DATA_DIR/dev_${split}.jsonl" ;;
    eval)
      allowed="$(report selected_all)"
      while IFS= read -r selected; do if [[ "$key" == "$selected" ]]; then found=true; fi; done <<< "$allowed"
      if [[ "$found" != true ]]; then echo "Official model must follow development selection." >&2; exit 2; fi
      data_path="$DATA_DIR/eval.jsonl" ;;
    *) echo "Unknown split: $split" >&2; exit 2 ;;
  esac
  case "$key" in
    parent) weights=(--adapter "$INITIAL_DIR/final_adapter" --projector "$INITIAL_DIR/final_projector.pt"
                     --qformer-adapter "$INITIAL_DIR/final_qformer_adapter.pt") ;;
    retained) weights=(--adapter "$RETAINED_DIR/final/adapter" --projector "$RETAINED_DIR/final/projector.pt"
                       --qformer-adapter "$RETAINED_DIR/final/qformer_adapter.pt") ;;
    visual_*|question_*)
      label="${key#*_}"
      if [[ "$label" != final && ! "$label" =~ ^checkpoint-[0-9]{6}$ ]]; then echo "Unknown checkpoint: $key" >&2; exit 2; fi
      if [[ "$key" == visual_* ]]; then bundle="$CONTROL_DIR/$label"; else bundle="$TRAIN_DIR/$label"; fi
      weights=(--adapter "$bundle/adapter" --projector "$bundle/projector.pt" --qformer-adapter "$bundle/qformer_adapter.pt") ;;
    *) echo "Unknown model: $key" >&2; exit 2 ;;
  esac
  destination="$(report eval_path --model "$key" --split "$split")"
  if [[ -f "$destination/scored/summary.json" ]]; then printf 'Reusing scored answers: %s\n' "$destination"; return; fi
  "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data_path" --output-dir "$destination" \
    --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
    --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" --feature-stage eva \
    --precision-protocol fp32_trainables_v1 --qformer-mode auto --qformer-text-input auto \
    --task-instruction-version legacy_v1 --max-new-tokens 96 --batch-size 1 "${weights[@]}"
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$destination/predictions.jsonl" \
    --output-dir "$destination/scored" --lingo-judge
}

case "${1:-help}" in
  check) report check ;;
  train|train_question) train question ;;
  train_control) train visual ;;
  train_pair)
    if [[ ! -f "$CONTROL_DIR/summary.json" ]]; then train visual; fi
    if [[ ! -f "$TRAIN_DIR/summary.json" ]]; then train question; fi
    report check
    ;;
  smoke)
    "$PYTHON_BIN" -u scripts/smoke_question_qformer_full.py --data "$DATA_DIR/train.jsonl" \
      --output-dir "${QUESTION_FULL_SMOKE_DIR:-$PROJECT_ROOT/outputs/question_qformer_full_smoke_v2}" \
      --initial-dir "$INITIAL_DIR" --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" \
      --checkpoint-root "$CHECKPOINT_ROOT" --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT"
    ;;
  status|eval_status) report status ;;
  eval) evaluate "${2:?model required}" "${3:?split required}" ;;
  eval_dev)
    models="$(report list_models)"
    while IFS= read -r key; do for split in action scenery; do evaluate "$key" "$split"; done; done <<< "$models"
    report report
    ;;
  eval_selected)
    report report
    models="$(report selected_all)"
    while IFS= read -r key; do evaluate "$key" eval; done <<< "$models"
    report report_eval
    ;;
  report) report report ;;
  report_eval) report report_eval ;;
  *) echo "usage: bash scripts/run_question_qformer_full.sh check|smoke|train_question|train_control|train_pair|status|eval_dev|eval_selected|eval MODEL SPLIT|report|report_eval|eval_status" ;;
esac
