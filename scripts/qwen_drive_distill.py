#!/usr/bin/env python3
"""Train-only offline answer transfer; long stages are explicitly user launched."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl, stable_key
from mpp_video_qa.distillation import assert_disjoint, validate_predictions
from mpp_video_qa.driving_teacher import (
    checked_scores, exclusive_lock, immutable_json, model_identity, read, source_identity,
    validate_data, write_json, write_rows,
)
from mpp_video_qa.transfer_pilot import (
    PROTOCOL, audit_cases, build_training_rows, checked_pair, digest, make_plan, validate_audit,
)
from scripts.prepare_visual_adaptation import stratified_subset
from scripts.qwen_drive_parallel import layout, merge_generation, parse_devices, shard_folder, shard_rows
from scripts.qwen_drive_screen import archive_baselines, children, child, event
from scripts.report_answer_opsd import JUDGE_PROTOCOL, line_count, paired_metrics

CODE_FILES = (
    "scripts/qwen_drive_distill.py", "scripts/run_qwen_drive_distill.sh",
    "mpp_video_qa/transfer_pilot.py", "scripts/score_transfer_pilot.py", "scripts/train_sft.py",
    "scripts/generate_answers.py", "scripts/generate_qwen_drive_teacher.py",
    "mpp_video_qa/driving_teacher.py", "scripts/qwen_drive_parallel.py", "scripts/qwen_drive_screen.py",
    "scripts/evaluate_answers.py", "scripts/report_preference.py", "scripts/report_answer_opsd.py",
    "mpp_video_qa/model.py", "mpp_video_qa/data.py", "mpp_video_qa/core.py",
    "mpp_video_qa/visual_bridge.py", "mpp_video_qa/opd.py",
    "mpp_video_qa/model_loading.py",
)
FROZEN = ROOT / "outputs/qwen_drive_teacher_4gpu_20261020"
ARMS = ("ce", "distill")


def utc():
    return datetime.now(timezone.utc).isoformat()


def bundle(args, role):
    if role == "start":
        folder = args.initial_dir / "final"
        return {"adapter": folder / "adapter", "projector": folder / "projector.pt",
                "qformer": folder / "qformer_adapter.pt"}
    folder = args.output_dir / "train" / role
    return {"adapter": folder / "final_adapter", "projector": folder / "final_projector.pt",
            "qformer": folder / "final_qformer_adapter.pt"}


def weight_identity(weights):
    from scripts.generate_answers import directory_fingerprint
    return {"adapter_identity": directory_fingerprint(weights["adapter"]),
            "projector_identity": file_sha256(weights["projector"]),
            "qformer_adapter_sha256": file_sha256(weights["qformer"])}


def configuration(args):
    parent = read(args.initial_dir / "run_config.json")
    summary = read(args.initial_dir / "summary.json")
    if (not summary.get("completed") or summary["optimizer_steps"] != parent["total_steps"] or
            parent["qformer_mode"] != "full" or parent["precision_protocol"] != "fp32_trainables_v1"):
        raise ValueError("initialization must be a completed full-QFormer FP32-trainable CE bundle")
    teacher = read(FROZEN / "official/teacher/run_manifest.json")
    actual_teacher, source = model_identity(args.model), source_identity(args.source)
    if actual_teacher != teacher["model"] or source != teacher["source"]:
        raise ValueError("teacher differs from the frozen official500 evaluated snapshot")
    weights = weight_identity(bundle(args, "start"))
    if weights != read(FROZEN / "official/retained_ce/run_manifest.json")["model_identity"]:
        raise ValueError("student differs from the evaluated retained CE starting point")
    base = args.checkpoints / "mpp-qwen-next-sft/sharded/model.safetensors.index.json"
    base_files = [base] + sorted(base.parent.glob("*.safetensors"))
    if len(base_files) < 2 or file_sha256(base) != parent["base_checkpoint_sha256"]:
        raise ValueError("student base checkpoint missing or differs from retained CE")
    return {"protocol": PROTOCOL, "profile": args.profile, "candidate_size": args.candidate_size,
            "seed": 44, "parallel_workers": len(args.gpus), "steps": args.steps,
            "global_batch": args.global_batch, "initialization": weights,
            "initial_run_config_sha256": file_sha256(args.initial_dir / "run_config.json"),
            "initial_summary_sha256": file_sha256(args.initial_dir / "summary.json"),
            "paths": {k: str(getattr(args, k).resolve()) for k in (
                "initial_dir", "data_dir", "cache_dir", "checkpoints", "visual_checkpoints", "bridge", "upstream")},
            "student_base_index_sha256": file_sha256(base),
            "student_weight_stat_identity": [{"file": str(p), "bytes": p.stat().st_size,
                                               "mtime_ns": p.stat().st_mtime_ns} for p in base_files],
            "bridge_sha256": file_sha256(args.bridge),
            "teacher_model": actual_teacher, "teacher_source": source,
            "teacher_python": str(args.teacher_python),
            "teacher_generation": {k: teacher[k] for k in (
                "max_new_tokens", "max_pixels_per_frame", "task_instruction", "seed", "decoding")},
            "student_loading": {"policy": "serialized_cpu_to_gpu_v1", "lock_scope": "workflow",
                                "generation_and_training": True},
            "training": {"lr": 2.5e-6, "projector_lr": 2.5e-6, "qformer_lr": 5e-6,
                         "weight_decay": .01, "warmup_ratio": .03, "dropout": 0.0,
                         "max_txt_len": 1536, "modules": "language QKV LoRA + projector + full QFormer",
                         "optimizer": "fresh AdamW", "official_used_for_selection": False},
            "candidacy": "exploratory transfer pilot authorized after mixed dev600/official500 evidence; old gate unchanged",
            "code_sha256": {p: file_sha256(ROOT / p) for p in CODE_FILES}}


def data_path(args, split):
    return args.output_dir / "data" / (split + ".jsonl")


def prepare(args):
    identity = configuration(args)
    immutable_json(args.output_dir / "experiment.json", identity)
    train = load_jsonl(args.data_dir / "train.jsonl")
    dev = load_jsonl(args.data_dir / "dev.jsonl")
    official = load_jsonl(args.data_dir / "eval.jsonl")
    assert_disjoint(train, dev, official)
    selected = stratified_subset(train, args.candidate_size, 44, 2)
    # Resolve aliases as well as comparing original frame strings.
    normalized = [[dict(r, frames=[str(Path(p).resolve()) for p in r["frames"]]) for r in split]
                  for split in (selected, dev, official)]
    assert_disjoint(*normalized)
    if args.profile == "formal" and (len(dev), len(official)) != (4320, 500):
        raise ValueError("formal evaluation must use full dev4320 and official500")
    splits = {"bank": selected, "dev": dev, "official": official,
              "smoke_dev": stratified_subset(dev, 8, 44, 2)}
    manifest = {"source_sha256": {str(args.data_dir / (s + ".jsonl")):
                                 file_sha256(args.data_dir / (s + ".jsonl"))
                                 for s in ("train", "dev", "dev_action", "dev_scenery", "eval")},
                "split_isolation": "questions, segments and frame paths; candidate paths also resolved",
                "sampling": "source-proportional seeded video round-robin, max2 questions/video", "splits": {}}
    for split, rows in splits.items():
        validate_data(rows, check_images=True)
        write_rows(data_path(args, split), rows)
        for i in range(len(args.gpus)):
            write_rows(args.output_dir / "data/shards" / ("%s_%03d.jsonl" % (split, i)),
                       shard_rows(rows, i, len(args.gpus)))
        manifest["splits"][split] = {"records": len(rows), "sha256": file_sha256(data_path(args, split)),
                                    "segments": len({r["segment_id"] for r in rows}),
                                    "sources": dict(Counter(r["source_subset"] for r in rows))}
    missing = []
    for row in selected + dev + official:
        key = stable_key(row["segment_id"])
        path = args.cache_dir / key[:2] / (key + ".pt")
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing EVA caches (%d); first: %s" % (len(set(missing)), missing[0]))
    immutable_json(args.output_dir / "data/manifest.json", manifest)
    event(args, "prepared", candidates=len(selected), long_tasks_started=False)
    return identity


def validate_prepared(args):
    if not (args.output_dir / "data/manifest.json").exists():
        return prepare(args)
    identity = configuration(args)
    if read(args.output_dir / "experiment.json") != identity:
        raise ValueError("experiment identity changed; keep worker count/settings/code or use a new output directory")
    manifest = read(args.output_dir / "data/manifest.json")
    for path, sha in manifest["source_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError("source dataset changed: %s" % path)
    for split, value in manifest["splits"].items():
        if file_sha256(data_path(args, split)) != value["sha256"]:
            raise ValueError("prepared split changed: %s" % split)
        rows = load_jsonl(data_path(args, split))
        for i in range(len(args.gpus)):
            if load_jsonl(args.output_dir / "data/shards" / ("%s_%03d.jsonl" % (split, i))) != shard_rows(rows, i, len(args.gpus)):
                raise ValueError("prepared shard changed")
    return identity


def common_student(args):
    return ["--upstream-root", args.upstream, "--checkpoint-root", args.checkpoints,
            "--visual-checkpoint-root", args.visual_checkpoints, "--bridge-init", args.bridge,
            "--cache-dir", args.cache_dir, "--feature-stage", "eva", "--qformer-mode", "full",
            "--precision-protocol", "fp32_trainables_v1", "--dtype", "bf16", "--max-txt-len", "1536",
            "--task-instruction-version", "legacy_v1", "--lora-dropout", "0",
            "--model-load-lock", args.output_dir / ".student_model_load.lock"]


def checked_student_part(args, split, role, index):
    folder = shard_folder(args.output_dir / split / role, index)
    rows = shard_rows(load_jsonl(data_path(args, split)), index, len(args.gpus))
    manifest = read(folder / "run_manifest.json")
    part_data = args.output_dir / "data/shards" / ("%s_%03d.jsonl" % (split, index))
    expected = dict(weight_identity(bundle(args, role)), data_sha256=file_sha256(part_data),
                    max_new_tokens=96, max_txt_len=1536, task_instruction_version="legacy_v1",
                    feature_stage="eva", qformer_mode="full", precision_protocol="fp32_trainables_v1")
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("student generation belongs to different weights/data/settings")
    values = load_jsonl(folder / "predictions.jsonl")
    validate_predictions(rows, values)
    rows_by_id = {r["question_id"]: r for r in rows}
    if any(r.get("source_subset") != rows_by_id[r["question_id"]]["source_subset"] for r in values):
        raise ValueError("student source label mismatch")
    summary = read(folder / "generation_summary.json")
    if summary.get("completed") != len(rows) or summary.get("total") != len(rows):
        raise ValueError("incomplete student generation summary")
    return values


def generate(args, split, role):
    folder = args.output_dir / split / role
    rows = layout(data_path(args, split), folder, len(args.gpus))
    jobs = []
    for i, gpu in enumerate(args.gpus):
        part = shard_folder(folder, i)
        if role == "teacher":
            command = [args.teacher_python, ROOT / "scripts/generate_qwen_drive_teacher.py", "generate",
                       "--data", data_path(args, split), "--model", args.model, "--source", args.source,
                       "--output-dir", part, "--max-new-tokens", "512", "--max-pixels", "921600",
                       "--shard-index", i, "--num-shards", len(args.gpus)]
        else:
            if (part / "generation_summary.json").exists():
                checked_student_part(args, split, role, i)
                continue
            weights = bundle(args, role)
            command = [sys.executable, ROOT / "scripts/generate_answers.py",
                       "--data", args.output_dir / "data/shards" / ("%s_%03d.jsonl" % (split, i)),
                       "--output-dir", part, "--adapter", weights["adapter"],
                       "--projector", weights["projector"], "--qformer-adapter", weights["qformer"],
                       "--record-base-identity", "--max-new-tokens", "96"] + common_student(args)
        jobs.append(dict(label="%s_%s_generate_%03d" % (split, role, i), gpu=gpu, command=command))
    children(args, "%s_%s_generate" % (split, role), jobs)
    if role == "teacher":
        if len(args.gpus) == 1:
            # Backend omits shard metadata for a single worker; preserve that contract.
            part = shard_folder(folder, 0)
            from mpp_video_qa.driving_teacher import validate_partial
            values = load_jsonl(part / "predictions.jsonl")
            if len(validate_partial(rows, values)) != len(rows):
                raise ValueError("incomplete teacher generation")
            immutable_json(folder / "run_manifest.json", read(part / "run_manifest.json"))
            write_rows(folder / "predictions.jsonl", values)
        else:
            merge_generation(data_path(args, split), folder, len(args.gpus))
    else:
        pooled = [r for i in range(len(args.gpus)) for r in checked_student_part(args, split, role, i)]
        by_id = validate_predictions(rows, pooled)
        write_rows(folder / "predictions.jsonl", [by_id[r["question_id"]] for r in rows])
        immutable_json(folder / "run_manifest.json", {
            "data_sha256": file_sha256(data_path(args, split)), "model_identity": weight_identity(bundle(args, role)),
            "shards_sha256": {str(shard_folder(folder, i) / "run_manifest.json"):
                              file_sha256(shard_folder(folder, i) / "run_manifest.json") for i in range(len(args.gpus))}})


def score(args, split, role):
    folder = args.output_dir / split / role
    rows = load_jsonl(data_path(args, split))
    validate_predictions(rows, load_jsonl(folder / "predictions.jsonl"))
    jobs = [dict(label="%s_%s_judge_%03d" % (split, role, i), gpu=gpu,
                 command=[sys.executable, ROOT / "scripts/score_transfer_pilot.py",
                          "--predictions", shard_folder(folder, i) / "predictions.jsonl",
                          "--output-dir", shard_folder(folder, i) / "scored"])
            for i, gpu in enumerate(args.gpus)]
    children(args, "%s_%s_judge" % (split, role), jobs)
    scores = [r for i in range(len(args.gpus))
              for r in checked_scores(shard_folder(folder, i), shard_rows(rows, i, len(args.gpus)))]
    by_id = validate_predictions(rows, scores)
    write_rows(folder / "scored/scored_predictions.jsonl", [by_id[r["question_id"]] for r in rows])
    from scripts.report_preference import summarize
    write_json(folder / "scored/summary.json", dict(summarize(scores), judge_protocol=JUDGE_PROTOCOL))
    checked_scores(folder, rows)


def review(args):
    rows = load_jsonl(data_path(args, "bank"))
    s, t = [checked_scores(args.output_dir / "bank" / role, rows) for role in ("start", "teacher")]
    cases = audit_cases(rows, s, t)
    folder = args.output_dir / "bank/review"
    write_rows(folder / "cases.jsonl", cases)
    identity = {"cases_sha256": file_sha256(folder / "cases.jsonl"),
                "source_sha256": {str(args.output_dir / "bank" / r / "scored/scored_predictions.jsonl"):
                                  file_sha256(args.output_dir / "bank" / r / "scored/scored_predictions.jsonl")
                                  for r in ("start", "teacher")}}
    immutable_json(folder / "manifest.json", identity)
    decisions = folder / "decisions.jsonl"
    if not decisions.exists():
        write_rows(decisions, [dict(question_id=r["question_id"], case_sha256=digest(r), status="pending",
                                   frames_viewed=[], reference_supported=None, teacher_supported=None,
                                   student_supported=None, reviewer="", reviewed_utc="", reason="") for r in cases])
    parts = ['<!doctype html><meta charset="utf-8"><title>训练候选视觉抽查</title>',
             '<style>body{font:16px sans-serif;margin:24px}img{width:19.5%}section{border-bottom:1px solid #999;padding:18px 0}</style>',
             '<h1>训练候选视觉抽查</h1><p>按 Action/Scenery 与四种 Judge 结果抽样；查看全部五帧。此页不是已审核证据。</p>']
    for i, row in enumerate(cases, 1):
        parts.append('<section><h2>%d · %s · %s</h2><p>%s</p>' % (
            i, html.escape(row["question_id"]), html.escape(" / ".join(row["stratum"])), html.escape(row["question"])))
        parts.extend('<a href="%s"><img src="%s" alt="frame %d"></a>' % (
            html.escape(os.path.relpath(p, folder)), html.escape(os.path.relpath(p, folder)), k)
                     for k, p in enumerate(row["frames"]))
        for label, value in (("原始参考", row["answers"]), ("学生", row["student_prediction"]),
                             ("教师", row["teacher_prediction"]), ("筛选未通过原因", row["teacher_target_rejection_reasons"])):
            parts.append('<p><b>%s:</b> %s</p>' % (label, html.escape(str(value))))
        parts.append('</section>')
    (folder / "review.html").write_text("\n".join(parts), encoding="utf-8")
    event(args, "awaiting_visual_audit", cases=len(cases), review=str(folder / "review.html"))


def write_training(args, rows, excluded, selection):
    path = data_path(args, "train")
    write_rows(path, rows)
    write_rows(args.output_dir / "data/excluded.jsonl", excluded)
    plan = make_plan(rows, file_sha256(path), args.steps, args.global_batch)
    immutable_json(args.output_dir / "data/sample_plan.json", plan)
    identity = {str(args.output_dir / "bank" / r / "scored/scored_predictions.jsonl"):
                file_sha256(args.output_dir / "bank" / r / "scored/scored_predictions.jsonl")
                for r in ("start", "teacher")}
    if args.profile == "formal":
        for name in ("cases.jsonl", "decisions.jsonl", "manifest.json"):
            path = args.output_dir / "bank/review" / name
            identity[str(path)] = file_sha256(path)
    immutable_json(args.output_dir / "data/selection.json", dict(selection, profile=args.profile,
                   train_sha256=file_sha256(data_path(args, "train")), source_sha256=identity,
                   plan_sha256=file_sha256(args.output_dir / "data/sample_plan.json")))


def select(args):
    review(args)  # Revalidates the source and fixed case identities; does not overwrite decisions.
    rows = load_jsonl(data_path(args, "bank"))
    s, t = [checked_scores(args.output_dir / "bank" / role, rows) for role in ("start", "teacher")]
    folder = args.output_dir / "bank/review"
    selected, excluded, summary = build_training_rows(rows, s, t, load_jsonl(folder / "cases.jsonl"),
                                                      load_jsonl(folder / "decisions.jsonl"))
    write_training(args, selected, excluded, summary)
    event(args, "selection_ready", **summary)


def validate_selection(args):
    manifest = read(args.output_dir / "data/selection.json")
    if manifest["profile"] != args.profile:
        raise ValueError("smoke-only targets cannot be used in a formal experiment")
    for path, sha in manifest["source_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError("selection evidence changed: %s" % path)
    if (manifest["train_sha256"] != file_sha256(data_path(args, "train")) or
            manifest["plan_sha256"] != file_sha256(args.output_dir / "data/sample_plan.json")):
        raise ValueError("training rows or sampling plan changed")
    if args.profile == "formal":
        folder = args.output_dir / "bank/review"
        validate_audit(load_jsonl(folder / "cases.jsonl"), load_jsonl(folder / "decisions.jsonl"))
    return manifest


def training_command(args, arm, resume=None, stop=0):
    folder = args.output_dir / "train" / arm
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node", len(args.gpus), ROOT / "scripts/train_sft.py",
               "--data", data_path(args, "train"), "--output-dir", folder,
               "--target-mode", "reference" if arm == "ce" else "distillation",
               "--sample-plan", args.output_dir / "data/sample_plan.json", "--disable-dropout",
               "--epochs", "1", "--batch-size", "1", "--grad-accumulation", args.global_batch // len(args.gpus),
               "--max-steps", args.steps, "--save-every", "50" if args.profile == "formal" else "1",
               "--lr", "2.5e-6", "--projector-lr", "2.5e-6", "--qformer-lr", "5e-6",
               "--weight-decay", ".01", "--warmup-ratio", ".03", "--seed", "44",
               "--train-projector", "--train-qformer", "--gradient-checkpointing",
               "--qformer-gradient-checkpointing", "--answer-only-logits"] + common_student(args)
    if resume:
        command.extend(["--resume-from", resume])
    else:
        weights = bundle(args, "start")
        command.extend(["--init-adapter", weights["adapter"], "--projector", weights["projector"],
                        "--qformer-adapter", weights["qformer"]])
    if stop:
        command.extend(["--stop-after-step", stop])
    return [str(v) for v in command]


def verify_training(args, arm):
    folder = args.output_dir / "train" / arm
    summary, config = read(folder / "summary.json"), read(folder / "run_config.json")
    initial = read(args.output_dir / "experiment.json")["initialization"]
    for key, expected in (("initial_adapter_sha256", initial["adapter_identity"]),
                          ("initial_projector_sha256", initial["projector_identity"]),
                          ("initial_qformer_sha256", initial["qformer_adapter_sha256"]),
                          ("data_sha256", file_sha256(data_path(args, "train"))),
                          ("sample_plan_sha256", file_sha256(args.output_dir / "data/sample_plan.json")),
                          ("target_mode", "reference" if arm == "ce" else "distillation"),
                          ("total_steps", args.steps), ("qformer_mode", "full"), ("disable_dropout", True),
                          ("model_load_lock", str(args.output_dir / ".student_model_load.lock"))):
        if config.get(key) != expected:
            raise ValueError("training identity mismatch %s/%s" % (arm, key))
    if (not summary.get("completed") or summary["optimizer_steps"] != args.steps or
            summary.get("sample_presentations") != args.steps * args.global_batch or
            summary.get("max_parameter_checksum_spread", 1) != 0 or
            summary.get("supervised_tokens_presented", 0) <= 0):
        raise ValueError("training completion not verified: %s" % arm)
    gradients = read(folder / "gradients_rank_0.json")
    if set(gradients) != {"lora", "projector", "qformer"} or any(v["nonzero"] == 0 for v in gradients.values()):
        raise ValueError("one of the three parameter groups received no gradient")
    weights = weight_identity(bundle(args, arm))
    if any(weights[k] == initial[k] for k in initial):
        raise ValueError("one of the three weight components did not change")
    return summary, config, weights


def train(args, arm, stop=0):
    validate_selection(args)
    folder = args.output_dir / "train" / arm
    if (folder / "summary.json").exists():
        verify_training(args, arm)
        return
    identity = {"experiment_sha256": file_sha256(args.output_dir / "experiment.json"),
                "selection_sha256": file_sha256(args.output_dir / "data/selection.json"),
                "command_without_resume": training_command(args, arm)}
    immutable_json(folder / "launch_identity.json", identity)
    checkpoints = sorted(p for p in folder.glob("checkpoint-*") if p.name[11:].isdigit()
                         and all((p / n).exists() for n in (
                             "adapter", "projector.pt", "qformer_adapter.pt", "training_state.pt", "metadata.json")))
    resume = checkpoints[-1] if checkpoints else None
    if not resume and (folder / "run_config.json").exists():
        # No committed update checkpoint exists. Preserve the failed attempt before restarting.
        archived = folder.with_name(arm + ".before_restart_%d" % time.time())
        folder.rename(archived)
        immutable_json(folder / "launch_identity.json", identity)
    for incomplete in folder.glob("checkpoint-*.incomplete"):
        incomplete.rename(incomplete.with_name(incomplete.name + ".aborted_%d" % time.time()))
    started = time.time()
    child(args, "train_" + arm, training_command(args, arm, resume, stop), gpu=",".join(args.gpus))
    with (folder / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(completed_utc=utc(), resume_from=str(resume) if resume else None,
                                     elapsed_seconds=time.time() - started, smoke_stop=stop)) + "\n")
    if not stop:
        verify_training(args, arm)


def archive_start(args, split):
    rows = load_jsonl(data_path(args, split))
    values, sources = archive_baselines(args, "official" if split == "official" else "dev", rows)
    if sources["retained_ce"]["model_identity"] != weight_identity(bundle(args, "start")):
        raise ValueError("archived baseline weight identity differs")
    folder = args.output_dir / split / "start"
    immutable_json(folder / "run_manifest.json", sources["retained_ce"])
    write_rows(folder / "predictions.jsonl", values["retained_ce"])
    write_rows(folder / "scored/scored_predictions.jsonl", values["retained_ce"])
    from scripts.report_preference import summarize
    immutable_json(folder / "scored/summary.json", dict(summarize(values["retained_ce"]), judge_protocol=JUDGE_PROTOCOL))
    checked_scores(folder, rows)


def validate_evaluation_identity(args, split, role):
    """A score file alone cannot certify the evaluated weight bundle."""
    folder = args.output_dir / split / role
    manifest = read(folder / "run_manifest.json")
    if manifest.get("model_identity") != weight_identity(bundle(args, role)):
        raise ValueError("evaluated model identity differs from selected weights")
    if role == "start":
        for path, sha in manifest["artifact_sha256"].items():
            if file_sha256(path) != sha:
                raise ValueError("archived start evidence changed")
    else:
        if manifest.get("data_sha256") != file_sha256(data_path(args, split)):
            raise ValueError("evaluation dataset changed")
        pooled = [r for i in range(len(args.gpus)) for r in checked_student_part(args, split, role, i)]
        actual = load_jsonl(folder / "predictions.jsonl")
        if {r["question_id"]: r for r in pooled} != {r["question_id"]: r for r in actual}:
            raise ValueError("merged evaluation differs from shard predictions")
        for i in range(len(args.gpus)):
            part = shard_folder(folder, i)
            stamp = read(part / "scored/input_identity.json")
            if stamp["predictions_sha256"] != file_sha256(part / "predictions.jsonl"):
                raise ValueError("Judge scores were computed on different predictions")
            for path, sha in stamp["code_sha256"].items():
                if file_sha256(ROOT / path) != sha:
                    raise ValueError("Judge implementation changed")


def report(args, split="dev", selected=None):
    from scripts.report_preference import cluster_interval, summarize
    rows = load_jsonl(data_path(args, split))
    roles = ("start", "ce", "distill") if split == "dev" else ("start", selected)
    roles = list(dict.fromkeys(roles))
    for role in roles:
        validate_evaluation_identity(args, split, role)
    values = {role: checked_scores(args.output_dir / split / role, rows) for role in roles}
    result = {"completed_utc": utc(), "protocol": PROTOCOL, "split": split,
              "judge_protocol": JUDGE_PROTOCOL, "records": len(rows), "models": {}, "comparisons": {},
              "interpretation": "Offline text-target transfer, not OPD. Single seed; visual audit is sampled. Official reused; teacher training overlap unknown."}
    if split == "official":
        result.update(selected_by_dev=selected, dev_report_sha256=file_sha256(args.output_dir / "dev_report.json"))
    for role in roles:
        result["models"][role] = {"all": summarize(values[role]), "by_source": {
            source: summarize([r for r in values[role] if r["source_subset"] == source])
            for source in sorted({r["source_subset"] for r in values[role]})}}
    pairs = [("start", r) for r in roles if r != "start"]
    if split == "dev":
        pairs.append(("ce", "distill"))
    for base, candidate in pairs:
        base_correct = sum(r["lingo_judge_correct"] for r in values[base])
        pair = paired_metrics(values[base], values[candidate])
        pair.update(cluster_interval(values[base], values[candidate]))
        pair["base_correct_retention"] = ((base_correct - pair["lingo_judge_correct"]["regressed"]) /
                                          base_correct if base_correct else None)
        pair["by_source"] = {}
        for source in sorted({r["source_subset"] for r in rows}):
            a, b = [[r for r in values[role] if r["source_subset"] == source] for role in (base, candidate)]
            pair["by_source"][source] = dict(paired_metrics(a, b), **cluster_interval(a, b))
        result["comparisons"][candidate + "_vs_" + base] = pair
    if split == "dev":
        # Fixed finals only; ties retain the earlier, simpler model.
        selected = max(roles, key=lambda r: result["models"][r]["all"]["lingo_judge_accuracy"])
        result["selected_by_dev"] = selected
        kd = result["comparisons"]["distill_vs_ce"]
        result["distillation_gain_established"] = (kd["net_change_95pct_interval"][0] > 0 and
            result["models"]["distill"]["all"]["lingo_judge_accuracy"] >
            result["models"]["start"]["all"]["lingo_judge_accuracy"])
        result["training"] = {}
        configs = []
        for arm in ARMS:
            summary, config, weights = verify_training(args, arm)
            result["training"][arm] = dict(summary=summary, weights=weights)
            configs.append(config)
        keys = ("data_sha256", "sample_plan_sha256", "world_size", "seed", "lr", "projector_lr", "qformer_lr",
                "warmup_ratio", "weight_decay", "grad_accumulation", "total_steps", "max_txt_len",
                "trainable_parameters_by_group", "disable_dropout", "qformer_mode")
        if any(configs[0][k] != configs[1][k] for k in keys):
            raise ValueError("control and distillation training conditions differ")
    sources = [args.output_dir / "experiment.json", args.output_dir / "data/selection.json",
               args.output_dir / "data/sample_plan.json", data_path(args, split)]
    if split == "official":
        sources.append(args.output_dir / "dev_report.json")
    for role in roles:
        sources.extend(args.output_dir / split / role / name for name in (
            "run_manifest.json", "predictions.jsonl", "scored/scored_predictions.jsonl", "scored/summary.json"))
    for arm in ARMS:
        sources.extend(args.output_dir / "train" / arm / name for name in (
            "run_config.json", "summary.json", "target_token_budget.json", "launch_identity.json"))
    result["sources_sha256"] = {str(p): file_sha256(p) for p in sources}
    write_json(args.output_dir / (split + "_report.json"), result)
    lines = ["# Qwen-Drive 离线答案蒸馏 · " + split, "", "生成时间（UTC）：" + result["completed_utc"], "",
             "| 模型 | 样本 | Judge | Action | Scenery |", "|---|---:|---:|---:|---:|"]
    for role, metrics in result["models"].items():
        m = metrics["all"]
        source_metrics = ["%.2f%%" % (100*metrics["by_source"][source]["lingo_judge_accuracy"])
                          if source in metrics["by_source"] else "N/A" for source in ("action", "scenery")]
        lines.append("| %s | %d | %.2f%% | %s | %s |" % (
            role, m["records"], 100*m["lingo_judge_accuracy"], *source_metrics))
    for name, pair in result["comparisons"].items():
        m = pair["lingo_judge_correct"]
        ci = pair["net_change_95pct_interval"]
        lines.extend(["", "%s：修复 %d、退化 %d、净变化 %+.2f pp；视频配对95%%区间 [%+.2f, %+.2f] pp；起点正确保留率 %s。" % (
            name, m["fixed"], m["regressed"], 100*m["net_change_mean"], 100*ci[0], 100*ci[1],
            "%.2f%%" % (100*pair["base_correct_retention"]) if pair["base_correct_retention"] is not None else "N/A")])
    if split == "dev":
        lines.extend(["", "dev 选定模型：`%s`；是否建立蒸馏相对匹配CE的收益：`%s`。" % (
            selected, result["distillation_gain_established"]), "", "只比较固定final；未在此阶段运行official。"])
        for arm in ARMS:
            m = result["training"][arm]["summary"]
            lines.append("%s：%d步，%d次有效样本呈现，%d个学生tokenizer监督token。" % (
                arm, m["optimizer_steps"], m["sample_presentations"], m["supervised_tokens_presented"]))
    lines.extend(["", result["interpretation"]])
    (args.output_dir / (split + "_report.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def evaluate(args, split="dev"):
    validate_selection(args)
    for arm in ARMS:
        verify_training(args, arm)
    if split == "dev":
        roles, selected = ARMS, None
    else:
        dev = read(args.output_dir / "dev_report.json")
        for path, sha in dev["sources_sha256"].items():
            if file_sha256(path) != sha:
                raise ValueError("dev selection evidence changed")
        selected = dev["selected_by_dev"]
        roles = [] if selected == "start" else [selected]
    archive_start(args, split)
    for role in roles:
        generate(args, split, role)
        score(args, split, role)
    return report(args, split, selected)


def status(args):
    def maybe(path):
        try:
            return read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    print(json.dumps(maybe(args.output_dir / "workflow.json"), ensure_ascii=False))
    identity = maybe(args.output_dir / "experiment.json")
    workers = identity.get("parallel_workers", len(args.gpus))
    manifest = maybe(args.output_dir / "data/manifest.json")
    print("文件进度；进程存活请结合 systemd/journal。outputs: " + str(args.output_dir))
    progress_roles = [("bank", ("teacher", "start")), ("dev", ("ce", "distill")), ("official", ("ce", "distill"))]
    if identity.get("profile") == "smoke":
        progress_roles = [("bank", ("teacher", "start")), ("smoke_dev", ARMS)]
    smoke_value = maybe(args.output_dir / "smoke/smoke_validation.json")
    if smoke_value:
        print("preflight smoke passed=" + str(smoke_value.get("passed", False)))
    for split, roles in progress_roles:
        total = manifest.get("splits", {}).get(split, {}).get("records", 0)
        for role in roles:
            folder = args.output_dir / split / role
            generated = sum(line_count(shard_folder(folder, i) / "predictions.jsonl") for i in range(workers))
            scored = sum(maybe(shard_folder(folder, i) / "scored/score_progress.json").get("completed", 0)
                         for i in range(workers))
            print("%-9s %-8s generated %5d/%d; Judge %5d/%d" % (split, role, generated, total, scored, total))
    for arm in ARMS:
        folder = args.output_dir / "train" / arm
        p = maybe(folder / "progress.json")
        s = maybe(folder / "summary.json")
        state = "completed_artifacts" if s.get("completed") else p.get("status", "not_started")
        if list(folder.glob("failure_rank_*.json")):
            state = "failure_artifact_present"
        print("%-8s %-28s %d/%d steps" % (arm, state, s.get("optimizer_steps", p.get("optimizer_step", 0)), identity.get("steps", 0)))
    print("selection: " + json.dumps(maybe(args.output_dir / "data/selection.json"), ensure_ascii=False)[:650])
    for split in ("dev", "official"):
        value = maybe(args.output_dir / (split + "_report.json"))
        valid = bool(value) and all(Path(p).is_file() and file_sha256(p) == sha
                                   for p, sha in value.get("sources_sha256", {}).items())
        print(split + " report validated=" + str(valid))


def smoke(args):
    if args.profile != "smoke":
        raise ValueError("smoke requires --profile smoke and a separate output directory")
    for role in ("teacher", "start"):
        generate(args, "bank", role)
        score(args, "bank", role)
    originals = load_jsonl(data_path(args, "bank"))
    teacher = {r["question_id"]: r for r in checked_scores(args.output_dir / "bank/teacher", originals)}
    if not all(r["finished"] for r in teacher.values()):
        raise ValueError("smoke teacher must finish its answers")
    # Deliberately exercises both target types without claiming an audit or a gain.
    rows = [dict(r, distillation_source="train", target_source="teacher" if i % 2 == 0 else "reference_replay",
                 teacher_answer=teacher[r["question_id"]]["prediction"], smoke_only_unfiltered=True)
            for i, r in enumerate(originals)]
    write_training(args, rows, [], {"effective_records": len(rows), "teacher_targets": len(rows)//2,
                                   "reference_replay": len(rows)//2, "quality_selection_tested": False})
    if not (args.output_dir / "train/ce/checkpoint-000001").exists():
        train(args, "ce", stop=1)
    train(args, "ce")
    train(args, "distill")
    for role in ARMS:
        generate(args, "smoke_dev", role)
        score(args, "smoke_dev", role)
    result = {"completed_utc": utc(), "passed": True, "world_size": len(args.gpus),
              "optimizer_steps_each_arm": args.steps, "ce_resume_from_step1": True,
              "heldout_answers_each_arm": 8, "quality_selection_tested": False, "gain_tested": False,
              "teacher_bank_sha256": file_sha256(args.output_dir / "bank/teacher/predictions.jsonl"),
              "training": {arm: verify_training(args, arm)[0] for arm in ARMS}}
    proof_files = [args.output_dir / "experiment.json", args.output_dir / "data/selection.json",
                   args.output_dir / "data/sample_plan.json"]
    for arm in ARMS:
        proof_files.extend(args.output_dir / "train" / arm / n for n in (
            "summary.json", "run_config.json", "target_token_budget.json"))
        proof_files.extend(args.output_dir / "smoke_dev" / arm / n for n in (
            "predictions.jsonl", "scored/scored_predictions.jsonl", "scored/summary.json"))
    result["source_sha256"] = {str(p): file_sha256(p) for p in proof_files}
    write_json(args.output_dir / "smoke_validation.json", result)
    event(args, "smoke_completed", passed=True, gain_tested=False)


def ensure_smoke(args):
    """Run the small GPU integration check before the user-launched formal bank."""
    if args.profile != "formal":
        raise ValueError("bank is a formal action; use smoke for the tiny integration check")
    folder = args.output_dir / "smoke"
    marker = folder / "smoke_validation.json"
    if marker.exists():
        value = read(marker)
        smoke_config = read(folder / "experiment.json")
        formal = read(args.output_dir / "experiment.json")
        if (not value.get("passed") or value.get("world_size") != len(args.gpus) or
                any(smoke_config[k] != formal[k] for k in (
                    "code_sha256", "initialization", "paths", "teacher_model", "teacher_source", "training",
                    "student_loading")) or
                not value.get("source_sha256") or
                any(file_sha256(p) != sha for p, sha in value["source_sha256"].items())):
            raise ValueError("preflight smoke evidence is stale")
        return
    command = [sys.executable, ROOT / "scripts/qwen_drive_distill.py", "smoke", "--profile", "smoke",
               "--gpus", ",".join(args.gpus), "--output-dir", folder]
    for flag in ("data_dir", "initial_dir", "model", "source", "teacher_python", "upstream", "checkpoints",
                 "visual_checkpoints", "cache_dir", "bridge"):
        command.extend(["--" + flag.replace("_", "-"), getattr(args, flag)])
    child(args, "preflight_smoke", command)
    ensure_smoke(args)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "bank", "select", "train", "train-eval", "dev", "report", "official", "status", "smoke"))
    p.add_argument("--profile", choices=("formal", "smoke"), default="formal")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qwen_drive_distill_4k_20261021")
    p.add_argument("--gpus", default=os.environ.get("QWEN_DISTILL_GPUS", "1,2,3,5"))
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_qformer_all_v1")
    p.add_argument("--initial-dir", type=Path, default=ROOT / "outputs/mpp_feedback_sft_control_40k_2epoch_v2")
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints/Qwen-Drive-1.0-4B")
    p.add_argument("--source", type=Path, default=ROOT / "third_party/Qwen-Drive-1.0")
    p.add_argument("--teacher-python", default=str(ROOT / ".venvs/qwen-drive/bin/python"))
    p.add_argument("--upstream", type=Path, default=Path(os.environ.get("MPP_UPSTREAM_ROOT", str(ROOT / "third_party/MPP-LLaVA"))))
    p.add_argument("--checkpoints", type=Path, default=Path(os.environ.get("MPP_CHECKPOINT_ROOT", str(ROOT / "checkpoints"))))
    p.add_argument("--visual-checkpoints", type=Path, default=Path(os.environ.get("MPP_PERSISTENT_CHECKPOINT_ROOT", str(ROOT / "checkpoints"))))
    p.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("MPP_EVA_CACHE_DIR", str(ROOT / "datasets/cache/lingoqa_5f_eva_v1"))))
    p.add_argument("--bridge", type=Path, default=ROOT / "checkpoints/visual_bridge_init.pt")
    args = p.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.gpus = parse_devices(args.gpus)
    args.candidate_size, args.steps, args.global_batch = (4000, 500, 16) if args.profile == "formal" else (8, 2, 4)
    if args.global_batch % len(args.gpus) or len(args.gpus) > 4:
        p.error("this pilot supports 1, 2 or 4 GPUs, with a fixed global batch")
    if args.profile == "smoke" and args.output_dir.name == "qwen_drive_distill_4k_20261021":
        p.error("smoke must specify a separate --output-dir")
    return args


def main():
    args = arguments()
    if args.action == "status":
        status(args)
        return
    with exclusive_lock(args.output_dir / ".workflow.lock"):
        try:
            validate_prepared(args)
            if args.action == "prepare":
                event(args, "prepared", long_tasks_started=False)
            elif args.action == "bank":
                ensure_smoke(args)
                for role in ("teacher", "start"):
                    generate(args, "bank", role)
                    score(args, "bank", role)
                review(args)
            elif args.action == "select":
                select(args)
            elif args.action in ("train", "train-eval"):
                if not (args.output_dir / "data/selection.json").exists():
                    select(args)
                for arm in ARMS:
                    train(args, arm)
                if args.action == "train-eval":
                    evaluate(args)
                event(args, "completed", official_evaluated=False)
            elif args.action in ("dev", "official"):
                evaluate(args, args.action)
                event(args, "completed", evaluated_split=args.action)
            elif args.action == "report":
                validate_selection(args)
                report(args)
            elif args.action == "smoke":
                smoke(args)
        except KeyboardInterrupt:
            event(args, "interrupted", note="No completion claimed; rerun the same action to validate and resume.")
            raise
        except Exception as error:
            event(args, "failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
