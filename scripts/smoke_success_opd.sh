#!/usr/bin/env bash
# Eight real training QA, two updates, then replay the second update from checkpoint.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA="$PROJECT_ROOT/datasets/processed/lingoqa_success_opd_smoke8_20261018"
BANK="$PROJECT_ROOT/outputs/success_opd_smoke8_bank_20261018"
RUN="$PROJECT_ROOT/outputs/success_opd_smoke8_train_20261018"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
case "${1:-check}" in
  bank|train|resume)
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
      echo 'Set CUDA_VISIBLE_DEVICES to one available GPU for this smoke.' >&2; exit 2
    fi
    ;;
esac
common=(--data "$DATA/train.jsonl" --teacher-mode reference_only)
train_args=("${common[@]}" --success-bank "$BANK" --keep-weight 1 --training-mode feedback_opd
  --probe-report "$PROJECT_ROOT/outputs/feedback_teacher_compare_v2/probe_report.json"
  --epochs 1 --global-batch-size 4 --grad-accumulation 4 --save-every 1 --checkpoint-steps 1
  --answer-only-logits --gradient-bucket-mb 25 --profile-every 1 --warmup-ratio 0)
case "${1:-check}" in
  bank)
    "$PYTHON_BIN" -u scripts/prepare_success_opd.py --records 8 --probe-records 8 --output-dir "$DATA"
    "$PYTHON_BIN" -u scripts/build_success_bank.py generate "${common[@]}" --output-dir "$BANK"
    "$PYTHON_BIN" -u scripts/build_success_bank.py merge "${common[@]}" --output-dir "$BANK"
    ;;
  merge)
    "$PYTHON_BIN" -u scripts/build_success_bank.py merge "${common[@]}" --output-dir "$BANK"
    ;;
  train)
    "$PYTHON_BIN" -u scripts/train_feedback_opd.py "${train_args[@]}" --output-dir "$RUN"
    ;;
  resume)
    "$PYTHON_BIN" -u scripts/train_feedback_opd.py "${train_args[@]}" --output-dir "${RUN}_resumed" \
      --resume-from "$RUN/checkpoint-000001"
    ;;
  check)
    "$PYTHON_BIN" -u scripts/check_success_opd_smoke.py --train-dir "$RUN" --bank-dir "$BANK"
    ;;
  *) echo 'Usage: smoke_success_opd.sh {bank|merge|train|resume|check}' >&2; exit 2 ;;
esac
