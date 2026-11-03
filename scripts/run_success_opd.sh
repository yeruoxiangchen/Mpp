#!/usr/bin/env bash
# Four matched arms; never starts unless explicitly invoked.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${SUCCESS_OPD_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_success_opd_4k_v1}"
SOURCE_DIR="${SUCCESS_OPD_SOURCE_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_feedback_opd_40k_v2}"
BANK_DIR="${SUCCESS_OPD_BANK_DIR:-$PROJECT_ROOT/outputs/success_opd_bank_4k_v1}"
TRAIN_DIR="${SUCCESS_OPD_TRAIN_DIR:-$PROJECT_ROOT/outputs/success_opd_4k_v1}"
EVAL_DIR="${SUCCESS_OPD_EVAL_DIR:-$PROJECT_ROOT/outputs/success_opd_eval_4k_v1}"
INITIAL_DIR="${SUCCESS_OPD_INITIAL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
PROBE_REPORT="${SUCCESS_OPD_PROBE_REPORT:-$PROJECT_ROOT/outputs/feedback_teacher_compare_v2/probe_report.json}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/third_party/MPP-LLaVA}"
PERSISTENT="$MPP_PERSISTENT_CHECKPOINT_ROOT"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-$MPP_DATA_ROOT/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then CHECKPOINT_ROOT="$PERSISTENT"; fi
VISUAL_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
BRIDGE_INIT="${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
CACHE_DIR="${MPP_EVA_CACHE_DIR:-$PROJECT_ROOT/datasets/cache/lingoqa_5f_eva_v1}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export MASTER_PORT="${MASTER_PORT:-29743}"
report_args=(--data-dir "$DATA_DIR" --bank-dir "$BANK_DIR" --train-dir "$TRAIN_DIR"
             --eval-dir "$EVAL_DIR" --initial-dir "$INITIAL_DIR")
runtime_args=(--data "$DATA_DIR/train.jsonl" --initial-dir "$INITIAL_DIR" --cache-dir "$CACHE_DIR"
              --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT"
              --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT"
              --teacher-mode reference_only --max-new-tokens 96 --max-txt-len 1536 --seed 44 --judge-device cpu)
report() { "$PYTHON_BIN" -u scripts/report_success_opd.py "${report_args[@]}" --action "$1" "${@:2}"; }
check_arm() { case "$1" in ce|ce_keep|opd|opd_keep) ;; *) echo "Unknown arm: $1" >&2; exit 2 ;; esac; }
devices() {
  if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo 'Set CUDA_VISIBLE_DEVICES explicitly after checking GPU availability.' >&2; exit 2
  fi
  if [[ "$CUDA_VISIBLE_DEVICES" == ,* || "$CUDA_VISIBLE_DEVICES" == *, ]]; then
    echo 'Empty GPU identifier.' >&2; exit 2
  fi
  IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  local seen=() device previous
  for device in "${VISIBLE_GPUS[@]}"; do
    if [[ -z "$device" || "$device" == *[[:space:]]* ]]; then echo 'Empty/whitespace GPU identifier.' >&2; exit 2; fi
    for previous in "${seen[@]}"; do
      if [[ "$device" == "$previous" ]]; then echo 'Duplicate GPU identifier.' >&2; exit 2; fi
    done
    seen+=("$device")
  done
}
training_layout() {
  devices
  WORLD="${#VISIBLE_GPUS[@]}"
  case "$WORLD" in 2|3|4) ;; *) echo 'Training requires 2, 3 or 4 explicitly visible GPUs.' >&2; exit 2 ;; esac
  if [[ -n "${NPROC_PER_NODE:-}" && "$NPROC_PER_NODE" != "$WORLD" ]]; then
    echo 'NPROC_PER_NODE must match CUDA_VISIBLE_DEVICES.' >&2; exit 2
  fi
  ACCUMULATION="$(( (16 + WORLD - 1) / WORLD ))"
}
prepare() {
  "$PYTHON_BIN" -u scripts/prepare_success_opd.py --source-dir "$SOURCE_DIR" --output-dir "$DATA_DIR" \
    --records 4000 --seed 44 --probe-records 200 --cache-dir "$CACHE_DIR"
}
bank() {
  devices
  local count="${#VISIBLE_GPUS[@]}" index pid failed=0
  local pids=()
  mkdir -p "$BANK_DIR"
  for ((index=0; index<count; index++)); do
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS[index]}" "$PYTHON_BIN" -u scripts/build_success_bank.py generate \
      "${runtime_args[@]}" --output-dir "$BANK_DIR" --shard-index "$index" --num-shards "$count" \
      > "$BANK_DIR/shard-${index}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  if [[ "$failed" != 0 ]]; then echo "Bank shard failed; inspect $BANK_DIR/shard-*.log. Reinvoke bank with the same GPU count to resume." >&2; return 1; fi
  "$PYTHON_BIN" -u scripts/build_success_bank.py merge "${runtime_args[@]}" \
    --output-dir "$BANK_DIR" --num-shards "$count"
}
train() {
  local arm="$1" resume="${2:-0}" state mode keep=0
  local extra=()
  check_arm "$arm"
  training_layout
  if [[ "$resume" == 1 ]]; then
    state="$(report resume_path --arm "$arm" --launch-world-size "$WORLD")"
    if [[ "$state" != complete ]]; then extra+=(--resume-from "$state"); fi
  else
    state="$(report launch --arm "$arm" --launch-world-size "$WORLD")"
  fi
  if [[ "$state" == complete ]]; then printf 'Verified completed arm: %s\n' "$arm"; return; fi
  mode=sft_control
  if [[ "$arm" == opd* ]]; then mode=feedback_opd; fi
  if [[ "$arm" == *_keep ]]; then keep=1; fi
  "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node "$WORLD" --master_port "$MASTER_PORT" \
    scripts/train_feedback_opd.py "${runtime_args[@]}" --output-dir "$TRAIN_DIR/$arm" \
    --training-mode "$mode" --success-bank "$BANK_DIR" --keep-weight "$keep" \
    --probe-report "$PROBE_REPORT" --epochs 2 --global-batch-size 16 --grad-accumulation "$ACCUMULATION" \
    --max-steps 0 --save-every 250 --checkpoint-steps 250 --lr 2.5e-6 --projector-lr 2.5e-6 \
    --qformer-lr 5e-6 --weight-decay .01 --warmup-ratio .03 --ce-weight 1 --opd-weight 1 \
    --answer-only-logits --gradient-bucket-mb 25 --runtime-judge-device cpu --profile-every 20 \
    --runtime-log-flush-every 4 --teacher-gate-cache-dir "$TRAIN_DIR/teacher_gate_cache" "${extra[@]}"
}
train_all() { local arm; for arm in ce ce_keep opd opd_keep; do train "$arm"; done; }
evaluate() {
  local arm="$1" label="$2" split="$3" key destination ready data bundle
  local weights=()
  if [[ "$arm" == parent ]]; then
    key=parent
    weights=(--adapter "$INITIAL_DIR/final_adapter" --projector "$INITIAL_DIR/final_projector.pt"
             --qformer-adapter "$INITIAL_DIR/final_qformer_adapter.pt")
  else
    check_arm "$arm"
    case "$label" in checkpoint-000250|final) ;; *) echo 'Unknown checkpoint.' >&2; exit 2 ;; esac
    key="${arm}__${label}"; bundle="$TRAIN_DIR/$arm/$label"
    weights=(--adapter "$bundle/adapter" --projector "$bundle/projector.pt" --qformer-adapter "$bundle/qformer_adapter.pt")
  fi
  destination="$(report eval_path --model "$key" --split "$split")"
  ready="$(report eval_ready --model "$key" --split "$split")"
  if [[ "$ready" == ready ]]; then printf 'Verified scored answers: %s\n' "$destination"; return; fi
  case "$split" in action|scenery) data="$DATA_DIR/dev_${split}.jsonl" ;; eval) data="$DATA_DIR/eval.jsonl" ;; *) exit 2 ;; esac
  "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data" --output-dir "$destination" \
    --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
    --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" --feature-stage eva \
    --precision-protocol fp32_trainables_v1 --qformer-mode auto --task-instruction-version legacy_v1 \
    --max-new-tokens 96 --max-txt-len 1536 --record-base-identity --batch-size 1 "${weights[@]}"
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$destination/predictions.jsonl" \
    --output-dir "$destination/scored" --lingo-judge
  report eval_ready --model "$key" --split "$split"
}
eval_dev() {
  local split arm label
  for split in action scenery; do evaluate parent final "$split"; done
  for arm in ce ce_keep opd opd_keep; do
    for label in checkpoint-000250 final; do
      for split in action scenery; do evaluate "$arm" "$label" "$split"; done
    done
  done
  report report
}
eval_selected() {
  local arm label
  # The stored dev selection is verified, never recomputed using official answers.
  label="$(report selected --arm ce)"
  evaluate parent final eval
  for arm in ce ce_keep opd opd_keep; do
    label="$(report selected --arm "$arm")"
    evaluate "$arm" "$label" eval
  done
  report report_eval
}
action="${1:-help}"
case "$action" in
  prepare|bank|train|train_all|pipeline|eval_dev|eval_selected|resume|report|report_eval)
    mkdir -p "$TRAIN_DIR"
    exec {SUCCESS_RUN_LOCK}>"$TRAIN_DIR/.runner.lock"
    flock -n "$SUCCESS_RUN_LOCK" || { echo 'Another success-OPD workflow holds this output lock.' >&2; exit 2; }
    ;;
esac
case "$action" in
  prepare) prepare ;;
  bank) bank ;;
  train) train "${2:?arm required}" ;;
  resume) train "${2:?arm required}" 1 ;;
  train_all) train_all ;;
  pipeline) training_layout; prepare; bank; train_all; eval_dev ;;
  eval_dev) devices; eval_dev ;;
  eval_selected) devices; eval_selected ;;
  report|report_eval|status) report "$action" "${@:2}" ;;
  *) echo 'usage: bash scripts/run_success_opd.sh prepare|bank|train ARM|train_all|pipeline|eval_dev|eval_selected|status|resume ARM|report|report_eval' ;;
esac
