#!/usr/bin/env python3
"""Full-QFormer on-policy feedback distillation and exactly matched CE control."""
import argparse
from collections import deque
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch.utils.data import DistributedSampler
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256
from mpp_video_qa.opd import SharedFrozenTeacher, completion_logits, distillation_loss, average_gradients
from scripts.feedback_opd_common import (add_runtime_arguments, configure_model, runtime_identity,
                                         training_dataset, build_student, Judge, current_rollout)
from scripts.generate_answers import write_json_atomic
from scripts.train_opd import save
from scripts.train_sft import seed_all, rng_state, restore_rng_state, gpu_memory
from mpp_video_qa.opd_runtime import GradientBuckets, PhaseTimer, TeacherGateCache, validate_resume_identity
from mpp_video_qa.global_batch import ExactGlobalBatchSampler, GLOBAL_BATCH_PROTOCOL

METRICS = ("ce_sum", "jsd_sum", "examples", "rollouts", "eligible", "student_correct",
           "student_truncated", "teacher_attempts", "teacher_finished", "teacher_correct",
           "completion_tokens", "distilled_tokens", "context_skips", "action_examples",
           "scenery_examples", "action_eligible", "scenery_eligible",
           "teacher_cache_hits", "teacher_generations", "judge_cpu_fallbacks",
           "keep_kl_sum", "keep_examples", "keep_tokens", "distill_sum",
           "distilled_correct", "distilled_wrong")


def optimizer_budget(records, world, accumulation, epochs, max_steps=0, global_batch_size=0):
    # DistributedSampler pads to ceil(records/world); each rank sees this many QA.
    steps = (math.ceil(records / global_batch_size) if global_batch_size else
             math.ceil(math.ceil(records / world) / accumulation)) * epochs
    return min(steps, max_steps) if max_steps > 0 else steps


def parse_args():
    p = argparse.ArgumentParser()
    add_runtime_arguments(p)
    p.add_argument("--training-mode", choices=("feedback_opd", "sft_control"), default="feedback_opd")
    p.add_argument("--probe-report", default=str(ROOT / "outputs/feedback_opd_probe_v1/probe_report.json"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=0,
                   help="exact real QA per update; 0 preserves the historical sampler")
    p.add_argument("--max-steps", type=int, default=0, help="0 runs the complete epoch budget")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--checkpoint-steps", type=int, nargs="*", default=[],
                   help="explicit checkpoint steps in addition to --save-every")
    p.add_argument("--lr", type=float, default=2.5e-6)
    p.add_argument("--projector-lr", type=float, default=2.5e-6)
    p.add_argument("--qformer-lr", type=float, default=5e-6)
    p.add_argument("--sft-data-contract", choices=("feedback", "original_full"), default="feedback")
    p.add_argument("--qformer-text-lr", type=float, default=0,
                   help="separate restored-text LR for the complete original-QA SFT recipe")
    p.add_argument("--language-freeze-steps", type=int, default=0)
    p.add_argument("--language-warmup-steps", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup-ratio", type=float, default=.03)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--opd-weight", type=float, default=1.0)
    p.add_argument("--opd-objective", choices=("jsd", "forward_kl"), default="jsd")
    p.add_argument("--opd-trajectory-scope", choices=("errors", "all"), default="errors",
                   help="all includes current correct rollouts; teacher must still pass the same gate")
    p.add_argument("--success-bank", default="", help="validated fixed parent success trajectory bank directory")
    p.add_argument("--keep-weight", type=float, default=0.0,
                   help="forward KL to the unprompted frozen parent on fixed successful token trajectories")
    p.add_argument("--chunk-tokens", type=int, default=8)
    p.add_argument("--resume-from", default="")
    p.add_argument("--answer-only-logits", action="store_true")
    p.add_argument("--gradient-bucket-mb", type=float, default=0)
    p.add_argument("--profile-every", type=int, default=0)
    p.add_argument("--teacher-gate-cache-dir", default="")
    p.add_argument("--runtime-judge-device", default="", help="cpu, local (one Judge per training GPU), or cuda:N")
    p.add_argument("--judge-equivalence-report", default="")
    p.add_argument("--judge-boundary-margin", type=float, default=.05)
    p.add_argument("--runtime-log-flush-every", type=int, default=1)
    args = configure_model(p.parse_args(), train=True)
    if args.language_lora_scope != "qkv" and args.training_mode != "sft_control":
        p.error("expanded language LoRA currently uses the pure CE SFT branch")
    if args.qformer_text_input == "question" and args.training_mode != "sft_control":
        p.error("question-input Q-Former uses the pure CE SFT branch")
    if (min(args.lr, args.projector_lr, args.qformer_lr, args.ce_weight, args.opd_weight) <= 0 or
            args.epochs < 1 or args.grad_accumulation < 1 or args.global_batch_size < 0 or args.chunk_tokens < 1 or
            not 0 <= args.warmup_ratio < 1 or args.gradient_bucket_mb < 0 or
            args.profile_every < 0 or args.runtime_log_flush_every < 1 or args.judge_boundary_margin <= 0 or
            not math.isfinite(args.keep_weight) or args.keep_weight < 0):
        p.error("invalid learning rates, weights, budget or warmup")
    if args.keep_weight and not args.success_bank:
        p.error("positive keep weight requires --success-bank")
    if args.success_bank and (args.sft_data_contract != "feedback" or args.qformer_text_input == "question" or
                              args.language_lora_scope != "qkv" or args.teacher_mode != "reference_only"):
        p.error("success preservation requires the original visual/QKV bridge and reference_only feedback contract")
    if args.teacher_gate_cache_dir and args.teacher_mode != "reference_only":
        p.error("teacher gate cache requires reference_only")
    if args.opd_trajectory_scope == "all" and args.teacher_mode != "reference_only":
        p.error("all-trajectory distillation requires reference_only")
    if args.training_mode == "sft_control" and (args.opd_objective != "jsd" or args.opd_trajectory_scope != "errors"):
        p.error("CE control must use default inactive OPD options")
    if args.sft_data_contract == "original_full":
        if (args.training_mode != "sft_control" or args.qformer_text_lr <= 0 or
                args.language_freeze_steps < 0 or args.language_warmup_steps < 0 or args.gradient_bucket_mb):
            p.error("original_full requires ordinary CE, positive text LR, nonnegative stages and active-only gradient sync")
    elif args.qformer_text_lr or args.language_freeze_steps or args.language_warmup_steps:
        p.error("staged parameter groups require --sft-data-contract original_full")
    return args


def validate_probe(args, identity):
    result = json.loads(Path(args.probe_report).read_text())
    if result.get("protocol") == "feedback_teacher_comparison_v2":
        value = result.get("modes", {}).get(args.teacher_mode, {})
        if (not result.get("completed") or value.get("eligible_draws", 0) <= 0 or
                value.get("token_diagnostic_draws", 0) <= 0):
            raise ValueError("finish the paired teacher comparison for the selected condition first")
        manifest = json.loads((Path(args.data).parent / "manifest.json").read_text())
        probe_identity = result.get("identity_by_mode", {}).get(args.teacher_mode)
        if probe_identity != identity:
            validate_subset_probe(args, identity, probe_identity, manifest)
        if manifest["split_sha256"]["train"] != identity["data_sha256"]:
            raise ValueError("comparison training pool differs")
        return file_sha256(args.probe_report)
    if not result.get("completed") or result.get("eligible_draws", 0) <= 0 or result.get("token_diagnostic_draws", 0) <= 0:
        raise ValueError("finish a probe with eligible draws and token diagnostics before training")
    probe_identity = result["identity"]
    changed = [key for key, value in identity.items()
               if key != "data_sha256" and probe_identity.get(key) != value]
    manifest = json.loads((Path(args.data).parent / "manifest.json").read_text())
    if (changed or manifest["split_sha256"]["train"] != identity["data_sha256"] or
            manifest["split_sha256"]["probe"] != probe_identity["data_sha256"]):
        raise ValueError("probe teacher/data/runtime differs from training: %s" % changed)
    # This verifies provenance and mechanical signals, never claims visual truth.
    return file_sha256(args.probe_report)


def validate_subset_probe(args, identity, probe_identity, manifest):
    """Reuse a teacher diagnostic only for an exact, independently verified train subset."""
    if not getattr(args, "success_bank", "") or not isinstance(probe_identity, dict):
        raise ValueError("paired teacher comparison differs from training data/teacher/runtime")
    changed = [key for key in set(identity) | set(probe_identity)
               if key != "data_sha256" and identity.get(key) != probe_identity.get(key)]
    parent = Path(manifest.get("parent_train_path", ""))
    if (changed or not parent.is_file() or
            manifest.get("parent_train_sha256") != probe_identity.get("data_sha256") or
            file_sha256(parent) != probe_identity.get("data_sha256")):
        raise ValueError("subset teacher probe provenance differs: %s" % changed)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    originals = [json.loads(line) for line in parent.read_text().splitlines() if line.strip()]
    by_id = {row["question_id"]: row for row in originals}
    ids = [row["question_id"] for row in rows]
    if (not rows or len(by_id) != len(originals) or len(ids) != len(set(ids)) or
            any(by_id.get(row["question_id"]) != row for row in rows)):
        raise ValueError("teacher probe reuse requires exact rows from its original training pool")


def tracked_parameters(model):
    named = dict(model.named_parameters())
    patterns = {"language": ".lora_B.default.weight", "projector": "llm_proj.weight",
                "qformer": ".crossattention.self.key.weight"}
    bridge = getattr(getattr(model, "backbone", None), "visual_bridge", None)
    if getattr(bridge, "question_input", False):
        patterns["qformer_text"] = ".encoder.layer.0.intermediate.dense.weight"
    for module in ("attn.c_proj", "mlp.w1", "mlp.w2", "mlp.c_proj"):
        pattern = ".%s.lora_B.default.weight" % module
        if any(pattern in name and p.requires_grad for name, p in named.items()):
            patterns["language_" + module.replace(".", "_")] = pattern
    return {group: next((name, p) for name, p in named.items() if pattern in name and p.requires_grad)
            for group, pattern in patterns.items()}


def teacher_checks(teacher):
    if teacher is None:
        return {}
    chosen = {"qformer": next((name, p) for name, p in teacher.model.named_parameters()
                               if "visual_bridge." in name and ".crossattention.self.key.weight" in name),
              "projector": ("llm_proj.weight", teacher.model.backbone.llm_proj.weight),
              "language": next((name, p) for name, p in teacher.llm.named_parameters()
                                if ".lora_B.opd_teacher.weight" in name)}
    return {group: (name, p, p.detach().float().cpu().clone()) for group, (name, p) in chosen.items()}


def training_config(args, identity, records, world, total_steps, probe_hash, checkpoints):
    config = dict(identity, training_mode=args.training_mode, records=records, world_size=world,
                  epochs=args.epochs, grad_accumulation=args.grad_accumulation, total_steps=total_steps,
                  lr=args.lr, projector_lr=args.projector_lr, qformer_lr=args.qformer_lr,
                  ce_weight=args.ce_weight, opd_weight=args.opd_weight, weight_decay=args.weight_decay,
                  warmup_ratio=args.warmup_ratio, gradient_checkpointing=True,
                  qformer_gradient_checkpointing=True, probe_report_sha256=probe_hash,
                  chunk_tokens=args.chunk_tokens, objective=getattr(args, "opd_objective", "jsd"), feature_stage="eva")
    if getattr(args, "opd_trajectory_scope", "errors") != "errors":
        config["opd_trajectory_scope"] = args.opd_trajectory_scope
    if checkpoints or args.teacher_mode != "reference_draft":
        config.update(teacher_mode=args.teacher_mode, checkpoint_steps=checkpoints)
    if getattr(args, "language_lora_scope", "qkv") != "qkv":
        config.update(objective="assistant_only_ce", new_optimizer_and_schedule=True)
    if "question_qformer" in identity:
        config.update(objective="assistant_only_ce", new_optimizer_and_schedule=True)
    if getattr(args, "global_batch_size", 0):
        config.update(global_batch_size=args.global_batch_size, global_batch_protocol=GLOBAL_BATCH_PROTOCOL)
    if getattr(args, "success_bank", ""):
        config.update(success_bank=args.success_bank_identity, keep_weight=args.keep_weight,
                      keep_objective="forward_kl")
    if getattr(args, "sft_data_contract", "feedback") == "original_full":
        from mpp_video_qa.sft_schedule import recipe
        config.update(objective="assistant_only_ce", new_optimizer_and_schedule=True,
                      sft_recipe=recipe(args, total_steps))
    return config


def backward_example(args, student, teacher, judge, record, device, timer, gate_cache,
                     divisor, stream, epoch, micro):
    """One real QA's CE/error JSD/success KL; all terms use the same QA divisor."""
    with timer.phase("features_to_device"):
        row, features = record["row"], record["features"].unsqueeze(0).to(device)
    values = dict.fromkeys(METRICS, 0)
    values["examples"] = 1
    source = row["source_subset"]
    if source in ("action", "scenery"):
        values[source + "_examples"] = 1
    if teacher is not None and getattr(args, "training_mode", "feedback_opd") == "feedback_opd":
        fallbacks_before = judge.cpu_fallbacks
        rollout_options = {}
        if getattr(args, "opd_trajectory_scope", "errors") != "errors":
            rollout_options["trajectory_scope"] = args.opd_trajectory_scope
        ids, prompt, diagnostic = current_rollout(student, teacher, judge, features, row,
                                                 args.max_new_tokens, args.teacher_mode, gate_cache, timer,
                                                 **rollout_options)
        values.update(rollouts=1, completion_tokens=len(ids),
                      teacher_cache_hits=int(diagnostic["teacher_cache_hit"]),
                      teacher_generations=int(diagnostic["teacher_generation_called"]),
                      judge_cpu_fallbacks=judge.cpu_fallbacks - fallbacks_before,
                      student_correct=int(diagnostic["student_judge"]["correct"]),
                      student_truncated=int(not diagnostic["student_finished"]),
                      teacher_attempts=int(diagnostic["teacher_correction"] is not None),
                      teacher_finished=int(diagnostic["teacher_finished"]),
                      teacher_correct=int(bool(diagnostic["teacher_finished"] and diagnostic["teacher_judge"] and
                                               diagnostic["teacher_judge"]["correct"])))
        if diagnostic["eligible"]:
            try:
                with timer.phase("teacher_probability"):
                    with teacher.active() as frozen:
                        targets = completion_logits(frozen, features, row["question"], ids, prompt,
                                                    answer_only=args.answer_only_logits).detach()
            except ValueError as error:
                if "exceeds max_txt_len" not in str(error):
                    raise
                diagnostic.update(gate_reason="teacher_context_too_long", eligible=False)
            if diagnostic["eligible"]:
                with timer.phase("student_jsd_forward"):
                    logits = completion_logits(student, features, row["question"], ids,
                                               answer_only=args.answer_only_logits)
                    objective = getattr(args, "opd_objective", "jsd")
                    divergence = distillation_loss(logits, targets, objective, args.chunk_tokens)
                if not torch.isfinite(divergence):
                    raise RuntimeError("nonfinite distillation loss")
                loss_value = float(divergence.detach())
                values.update(jsd_sum=loss_value if objective == "jsd" else 0., distill_sum=loss_value,
                              eligible=1, distilled_tokens=len(ids),
                              distilled_correct=int(diagnostic["student_judge"]["correct"]),
                              distilled_wrong=int(not diagnostic["student_judge"]["correct"]))
                if source in ("action", "scenery"):
                    values[source + "_eligible"] = 1
                with timer.phase("student_jsd_backward"):
                    (args.opd_weight * divergence / divisor).backward()
                del divergence, logits, targets
        values["context_skips"] = int(diagnostic["gate_reason"] == "teacher_context_too_long")
        stream.write(json.dumps(dict(diagnostic, epoch=epoch, micro_step=micro,
                                     distillation_objective=getattr(args, "opd_objective", "jsd"),
                                     question_id=row["question_id"], source_subset=row["source_subset"]),
                                ensure_ascii=False) + '\n')
        if (micro + 1) % args.runtime_log_flush_every == 0:
            stream.flush()
    anchor = record.get("success_token_ids")
    if getattr(args, "keep_weight", 0) > 0 and anchor is not None:
        if teacher is None:
            raise ValueError("success preservation requires the frozen original teacher")
        ids = torch.as_tensor(anchor, dtype=torch.long, device=device)
        # This is a fixed off-policy anchor: both models see only the ordinary
        # question/images and the identical original successful prefix, never references.
        with timer.phase("keep_teacher_probability"):
            with teacher.active() as frozen:
                targets = completion_logits(frozen, features, row["question"], ids,
                                            answer_only=args.answer_only_logits).detach()
        with timer.phase("keep_student_forward"):
            logits = completion_logits(student, features, row["question"], ids,
                                       answer_only=args.answer_only_logits)
            divergence = distillation_loss(logits, targets, "forward_kl", args.chunk_tokens)
        if not torch.isfinite(divergence):
            raise RuntimeError("nonfinite success preservation KL")
        values.update(keep_kl_sum=float(divergence.detach()), keep_examples=1, keep_tokens=len(ids))
        with timer.phase("keep_student_backward"):
            (args.keep_weight * divergence / divisor).backward()
        # Complete this graph before switching adapters for any later example.
        del divergence, logits, targets
    with timer.phase("ce_forward"):
        ce = (student.answer_only_ce(features, [row["question"]], [record["answer"]])
              if args.answer_only_logits else student(features, [row["question"]], [record["answer"]]).loss)
    if not torch.isfinite(ce):
        raise RuntimeError("nonfinite reference CE")
    values["ce_sum"] = float(ce.detach())
    with timer.phase("ce_backward"):
        (args.ce_weight * ce / divisor).backward()
    return values


def restore_counters(cumulative, saved):
    if len(saved) > len(METRICS):
        raise ValueError("checkpoint contains unknown future metrics")
    cumulative[:len(saved)] = cumulative.new_tensor(saved)
    if len(saved) <= METRICS.index("distill_sum"):
        cumulative[METRICS.index("distill_sum")] = saved[METRICS.index("jsd_sum")]
        cumulative[METRICS.index("distilled_wrong")] = saved[METRICS.index("eligible")]
    if len(saved) <= METRICS.index("teacher_generations"):
        # Historical runs did not cache corrections: each completed teacher attempt
        # corresponds to one actual generation. Preserve cumulative count semantics.
        cumulative[METRICS.index("teacher_generations")] = saved[METRICS.index("teacher_attempts")]


def training_probe_hash(args, identity):
    """CE needs no teacher probe; retain old resume metadata when present."""
    if args.training_mode == "feedback_opd":
        return validate_probe(args, identity)
    if args.resume_from:
        previous = Path(args.output_dir) / "run_config.json"
        if previous.is_file():
            return json.loads(previous.read_text()).get("probe_report_sha256")
    return None


def archive_failures(output):
    """Preserve old rank errors outside the active status file pattern."""
    for failure in output.glob("failure_rank_*.json"):
        failure.rename(failure.with_suffix(".before_retry_%d" % time.time_ns()))


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feedback OPD")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("completed run exists; use a new output directory")
    if (output / "run_config.json").exists() and not args.resume_from:
        raise ValueError("unfinished run exists; use --resume-from")
    configure_model(args, train=True)
    identity = runtime_identity(args)
    if "question_qformer" in identity and args.training_mode != "sft_control":
        raise ValueError("question-input visual bridge currently uses pure SFT")
    probe_hash = training_probe_hash(args, identity)
    dataset = training_dataset(args)
    success_bank = {}
    if args.success_bank:
        from mpp_video_qa.success_bank import load_success_bank
        success_bank = load_success_bank(args, identity, dataset)
        args.success_bank_identity = success_bank.identity
        if args.keep_weight > 0 and not success_bank:
            raise ValueError("success preservation needs at least one validated eligible parent trajectory")
    if args.global_batch_size:
        sampler = ExactGlobalBatchSampler(len(dataset), args.global_batch_size, world, rank, args.seed)
        if args.grad_accumulation != sampler.accumulation:
            raise ValueError("exact global batch requires grad_accumulation=ceil(global_batch_size/world_size)")
    else:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, shuffle=True)
    total_steps = optimizer_budget(len(dataset), world, args.grad_accumulation, args.epochs,
                                   args.max_steps, args.global_batch_size)
    checkpoints = sorted(set(args.checkpoint_steps))
    if any(step < 1 or step > total_steps for step in checkpoints):
        raise ValueError("checkpoint steps must lie within the actual optimizer budget")
    config = training_config(args, identity, len(dataset), world, total_steps, probe_hash, checkpoints)
    resume = None
    if args.resume_from:
        checkpoint = Path(args.resume_from)
        resume = torch.load(checkpoint / "training_state.pt", map_location="cpu")
        validate_resume_identity(resume["identity"], config)
        args.init_adapter = str(checkpoint / "adapter")
        args.projector = str(checkpoint / "projector.pt")
        args.qformer_adapter = str(checkpoint / "qformer_adapter.pt")
        # The teacher remains the original fixed SFT snapshot on resume.
    # A load-time failure has no run_config/checkpoint. Keep its evidence, but do
    # not let status keep reporting that old failure during a fresh retry.
    if rank == 0:
        archive_failures(output)
    if world > 1:
        torch.distributed.init_process_group("nccl", timeout=timedelta(hours=12))
    seed_all(args.seed + rank)
    for loading_rank in range(world):
        if rank == 0:
            status = {"status": "loading_model", "loading_rank": loading_rank,
                      "loaded_ranks": loading_rank, "world_size": world, "rank0_pid": os.getpid(),
                      "records": len(dataset), "optimizer_steps_total": total_steps,
                      "updated_unix_seconds": time.time()}
            write_json_atomic(output / "progress.json", status)
            print(json.dumps(status), flush=True)
        if rank == loading_rank:
            student = build_student(args, device, train=True)
            bridge = student.backbone.visual_bridge
            if getattr(bridge, "initialization_report", None):
                write_json_atomic(output / ("question_initialization_rank_%d.json" % rank), bridge.initialization_report)
                print(json.dumps(dict(bridge.initialization_report, status="question_branch_initialized", rank=rank)), flush=True)
            expansion = getattr(student.backbone.llm_model, "_lora_expansion_report", None)
            if expansion:
                write_json_atomic(output / ("lora_initialization_rank_%d.json" % rank), expansion)
                print(json.dumps({"status": "language_lora_expanded", "rank": rank,
                                  "target_module_count": expansion["target_module_count"],
                                  "language_trainable_parameters": expansion["language_trainable_parameters"],
                                  "inherited_weights_exact": expansion["inherited_weights_exact"],
                                  "new_B_zero": expansion["new_B_zero"]}), flush=True)
            teacher = SharedFrozenTeacher(student, args) if args.training_mode == "feedback_opd" or args.keep_weight > 0 else None
            print(json.dumps({"status": "model_loaded", "rank": rank, "pid": os.getpid()}), flush=True)
        if world > 1:
            torch.distributed.barrier()
    named = [(name, p) for name, p in student.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    if world > 1:
        for parameter in parameters:
            torch.distributed.broadcast(parameter.data, src=0)
    staged = None
    # Capture the complete planned parameter set before temporarily freezing LoRA.
    tracked = tracked_parameters(student)
    if args.sft_data_contract == "original_full":
        from mpp_video_qa.sft_schedule import StagedSFT
        staged = StagedSFT(named, args, total_steps)
        optimizer, scheduler = staged.optimizer, staged.scheduler
    else:
        groups = []
        for group, marker, lr in (("language", "llm_model.", args.lr), ("projector", "llm_proj.", args.projector_lr),
                                  ("qformer", "visual_bridge.", args.qformer_lr)):
            selected = [p for name, p in named if marker in name]
            if not selected:
                raise ValueError("missing trainable %s branch" % group)
            groups.append({"params": selected, "lr": lr, "group_name": group})
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    judge = Judge(args) if args.training_mode == "feedback_opd" else None
    reducer = GradientBuckets(parameters, world, args.gradient_bucket_mb) if args.gradient_bucket_mb else None
    timer = PhaseTimer(device, args.profile_every)
    runtime = {"protocol": "feedback_opd_execution_v1", "answer_only_logits": args.answer_only_logits,
               "gradient_bucket_mb": args.gradient_bucket_mb, "profile_every": args.profile_every,
               "teacher_gate_cache_dir": args.teacher_gate_cache_dir,
               "judge_device": args.runtime_judge_device or args.judge_device,
               "judge_boundary_margin": args.judge_boundary_margin,
               "judge_equivalence_report_sha256": file_sha256(args.judge_equivalence_report) if args.judge_equivalence_report else None,
               "log_flush_every": args.runtime_log_flush_every, "world_size": world,
               "grad_accumulation": args.grad_accumulation,
               "effective_global_batch_size": args.global_batch_size or world * args.grad_accumulation}
    cache_identity = dict(identity, execution_judge_device=runtime["judge_device"],
                          judge_boundary_margin=args.judge_boundary_margin,
                          judge_equivalence_report_sha256=runtime["judge_equivalence_report_sha256"],
                          judge_scorer_sha256=file_sha256(ROOT / "scripts/evaluate_answers.py"))
    gate_cache = TeacherGateCache(args.teacher_gate_cache_dir, cache_identity) if args.teacher_gate_cache_dir and judge else None
    initial = {key: p.detach().float().cpu().clone() for key, (_, p) in tracked.items()}
    frozen_checks = teacher_checks(teacher)
    step, micro, start_epoch, start_batch = 0, 0, 0, 0
    cumulative = torch.zeros(len(METRICS), device=device, dtype=torch.float64)
    if resume:
        step, micro = resume["optimizer_steps"], resume["micro_steps"]
        if staged:
            staged.restore(resume, step)
        else:
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
        start_epoch, start_batch = resume["next_epoch"], resume["next_batch"]
        saved_metrics = resume["cumulative_metrics"]
        restore_counters(cumulative, saved_metrics)
        restore_rng_state(resume["rng_states"][rank])
    else:
        seed_all(args.seed + rank)
    if staged:
        runtime["sft_parameter_groups"] = staged.describe()
    if rank == 0:
        write_json_atomic(output / "run_config.json", config)
        write_json_atomic(output / "runtime_config.json", runtime)
        with (output / "runtime_history.jsonl").open("a") as history:
            history.write(json.dumps(dict(runtime, resume_from=args.resume_from, start_step=step,
                                          updated_unix_seconds=time.time())) + '\n')
    started = time.time()
    previous_update = started
    recent_seconds = deque(maxlen=20)
    counters = torch.zeros_like(cumulative)
    optimizer.zero_grad(set_to_none=True)
    state = None
    with (output / ("rollouts_rank_%d.jsonl" % rank)).open("a", encoding="utf-8") as stream:
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            student.train()
            for batch_index in range(start_batch if epoch == start_epoch else 0, len(indices)):
                if step >= total_steps:
                    break
                if batch_index % args.grad_accumulation == 0:
                    timer.begin(step)
                    if staged:
                        parameters = staged.apply_phase(step)
                divisor = (sampler.loss_divisor(batch_index) if args.global_batch_size else
                           min(args.grad_accumulation, len(indices) -
                               (batch_index // args.grad_accumulation) * args.grad_accumulation))
                values = dict.fromkeys(METRICS, 0)
                if indices[batch_index] is not None:
                    with timer.phase("data_load", "cpu"):
                        record = dataset[indices[batch_index]]
                        record["success_token_ids"] = success_bank.get(record["row"]["question_id"])
                    values = backward_example(args, student, teacher, judge, record, device, timer,
                                              gate_cache, divisor, stream, epoch, micro)
                micro += 1
                counters += counters.new_tensor([values[key] for key in METRICS])
                boundary = (batch_index + 1) % args.grad_accumulation == 0 or batch_index + 1 == len(indices)
                if boundary:
                    # Normalize by all ranks/examples, including CE-only ranks. Never
                    # skip a collective because the local feedback gate is closed.
                    with timer.phase("gradient_sync"):
                        reducer.reduce() if reducer is not None else average_gradients(parameters, world)
                    group_norms = staged.gradient_norms() if staged else None
                    with timer.phase("optimizer"):
                        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if world > 1:
                        torch.distributed.all_reduce(counters)
                    cumulative += counters
                    stats = dict(zip(METRICS, counters.tolist()))
                    next_epoch, next_batch = epoch, batch_index + 1
                    if next_batch == len(indices):
                        next_epoch, next_batch = epoch + 1, 0
                    state = {"identity": config, "optimizer_steps": step, "micro_steps": micro,
                             "next_epoch": next_epoch, "next_batch": next_batch,
                             "cumulative_metrics": cumulative.tolist(), "scheduler": scheduler.state_dict(),
                             "runtime": runtime}
                    if staged:
                        state["sft_stage"] = staged.stage_state(step)
                    timing = timer.finish()
                    if timing is not None:
                        with (output / ("perf_rank_%d.jsonl" % rank)).open("a") as handle:
                            handle.write(json.dumps(dict(timing, optimizer_step=step, rank=rank,
                                                         updated_unix_seconds=time.time())) + '\n')
                    now = time.time()
                    recent_seconds.append(now - previous_update)
                    previous_update = now
                    if rank == 0:
                        progress = {"status": "running", "rank0_pid": os.getpid(), "training_mode": args.training_mode,
                                    "optimizer_step": step, "optimizer_steps_total": total_steps,
                                    "progress_percent": 100 * step / total_steps,
                                    "ce_mean": stats["ce_sum"] / max(1, stats["examples"]),
                                    "jsd_mean_eligible": stats["jsd_sum"] / max(1, stats["eligible"]),
                                    "distillation_objective": args.opd_objective,
                                    "distillation_mean_eligible": stats["distill_sum"] / max(1, stats["eligible"]),
                                    "keep_kl_mean_anchored": stats["keep_kl_sum"] / max(1, stats["keep_examples"]),
                                    "keep_rate": stats["keep_examples"] / max(1, stats["examples"]),
                                    "total_loss_mean": (args.ce_weight * stats["ce_sum"] +
                                                        args.opd_weight * stats["distill_sum"] +
                                                        args.keep_weight * stats["keep_kl_sum"]) / max(1, stats["examples"]),
                                    "eligible_rate": stats["eligible"] / max(1, stats["rollouts"]),
                                    "step_metrics": stats, "cumulative_metrics": dict(zip(METRICS, cumulative.tolist())),
                                    "gradient_norm": float(norm), "learning_rates": scheduler.get_last_lr(),
                                    "elapsed_seconds_this_run": time.time() - started,
                                    "teacher_mode": args.teacher_mode,
                                    "world_size": world, "grad_accumulation": args.grad_accumulation,
                                    "effective_global_batch_size": args.global_batch_size or world * args.grad_accumulation,
                                    "recent_seconds_per_update": sum(recent_seconds) / len(recent_seconds),
                                    "estimated_remaining_hours": (total_steps - step) *
                                        sum(recent_seconds) / len(recent_seconds) / 3600,
                                    "updated_unix_seconds": time.time(), "gpu_memory": gpu_memory(device)}
                        if staged:
                            progress.update(sft_phase=staged.phase, sft_stage=state["sft_stage"],
                                            parameter_group_gradient_norms=group_norms,
                                            parameter_group_learning_rates={g["group_name"]: g["lr"]
                                                                            for g in optimizer.param_groups},
                                            sft_parameter_groups=staged.describe())
                            if step == args.language_freeze_steps:
                                language = tracked["language"][1]
                                delta = float((language.detach().float().cpu() - initial["language"]).abs().max())
                                if delta != 0 and not args.resume_from:
                                    raise RuntimeError("language LoRA changed during bridge-only adaptation")
                                write_json_atomic(output / "bridge_adaptation_summary.json", {
                                    "completed_steps": step, "language_tracked_delta_this_run": delta,
                                    "next_phase": "joint", "resumed_from": args.resume_from or None})
                        if timing is not None:
                            progress["phase_timing_rank0"] = timing
                        write_json_atomic(output / "progress.json", progress)
                        with (output / "training_log.jsonl").open("a") as handle:
                            handle.write(json.dumps(progress) + '\n')
                        print(json.dumps(progress), flush=True)
                    counters.zero_()
                    if step < total_steps and (step in checkpoints or
                                               (args.save_every and step % args.save_every == 0)):
                        stream.flush()
                        save(output, student, optimizer, state, rank, world)
            start_batch = 0
            if step >= total_steps:
                break
    if state is None:
        raise ValueError("no optimizer updates executed")
    frozen_deltas = {group: float((p.detach().float().cpu() - before).abs().max())
                     for group, (_, p, before) in frozen_checks.items()}
    if any(value != 0 for value in frozen_deltas.values()):
        raise RuntimeError("frozen teacher weights changed")
    save(output, student, optimizer, state, rank, world, final=True)
    if rank == 0:
        summary = {"completed": True, "optimizer_steps": step, "optimizer_steps_total": total_steps,
                   "records": len(dataset), "teacher_mode": args.teacher_mode,
                   "world_size": world, "grad_accumulation": args.grad_accumulation,
                   "effective_global_batch_size": args.global_batch_size or world * args.grad_accumulation,
                   "micro_steps_per_rank": micro, "training_mode": args.training_mode,
                   "final_bundle": str((output / "final").resolve()),
                   "elapsed_seconds_this_run": time.time() - started,
                   "cumulative_metrics": dict(zip(METRICS, cumulative.tolist())),
                   "tracked_parameter_delta_this_run": {group: float((p.detach().float().cpu() - initial[group]).abs().max())
                                                         for group, (_, p) in tracked.items()},
                   "teacher_tracked_parameter_deltas": frozen_deltas, "gpu_memory_rank0": gpu_memory(device)}
        if args.language_lora_scope != "qkv":
            summary.update(language_lora=config["language_lora"],
                           trainable_parameters_by_group={g["group_name"]: sum(p.numel() for p in g["params"])
                                                          for g in optimizer.param_groups})
        if "question_qformer" in config:
            summary.update(question_qformer=config["question_qformer"],
                           trainable_parameters_by_group={g["group_name"]: sum(p.numel() for p in g["params"])
                                                          for g in optimizer.param_groups})
        if staged:
            summary.update(sft_recipe=config["sft_recipe"], original_full_data=config["original_full_data"],
                           sft_stage=staged.stage_state(step),
                           trainable_parameters_by_group={g["group_name"]: sum(p.numel() for p in g["params"])
                                                          for g in optimizer.param_groups})
        totals = summary["cumulative_metrics"]
        summary.update(eligible_rollout_rate=totals["eligible"] / max(1, totals["rollouts"]),
                       teacher_finished_correction_rate=totals["teacher_correct"] / max(1, totals["teacher_finished"]),
                       actual_rollouts=int(totals["rollouts"]), actual_distilled_tokens=int(totals["distilled_tokens"]))
        write_json_atomic(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    cli = parse_args()
    try:
        main(cli)
    except Exception:
        folder = Path(cli.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        write_json_atomic(folder / ("failure_rank_%s.json" % os.environ.get("RANK", "0")),
                          {"status": "failed", "rank": int(os.environ.get("RANK", "0")), "pid": os.getpid(),
                           "updated_unix_seconds": time.time(), "traceback": traceback.format_exc()})
        raise
