#!/usr/bin/env python3
"""Complete original-QA, staged SFT: provenance, progress and dev-selected evaluation."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import SYSTEM_MESSAGE, TASK_INSTRUCTION, build_user_prompt, file_sha256, load_jsonl
from mpp_video_qa.full_sft_data import validate_full_training

PROTOCOL = "question_qformer_full_original_qa_comparison_v2"
BRANCHES = ("visual", "question")
SAVED_STEPS = (5000, 10000, 15000, 20000)


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def line_count(path):
    if not path.is_file():
        return 0
    with path.open() as stream:
        return sum(bool(s.strip()) for s in stream)


def budget(records):
    return math.ceil(math.ceil(records / 4) / 4)


def labels(records):
    total = budget(records)
    return [("checkpoint-%06d" % s, s) for s in SAVED_STEPS if s < total] + [("final", total)]


def data_contract(args):
    if not hasattr(args, "_verified_contract"):
        args._verified_contract = validate_full_training(args.data_dir / "train.jsonl", args.initial_dir)
    return args._verified_contract


def validate_configs(args):
    from scripts.report_feedback_opd import identities
    contract = data_contract(args)
    total = budget(contract["records"])
    parent = identities(args.initial_dir, True)
    initial = {"adapter": parent["adapter_identity"], "projector": parent["projector_identity"],
               "qformer": parent["qformer_adapter_sha256"]}
    expected = {"training_mode": "sft_control", "records": contract["records"], "epochs": 1,
                "world_size": 4, "grad_accumulation": 4, "total_steps": total,
                "lr": 2.5e-6, "projector_lr": 5e-6, "qformer_lr": 1e-5,
                "ce_weight": 1.0, "weight_decay": .01, "warmup_ratio": .03,
                "objective": "assistant_only_ce", "feature_stage": "eva",
                "precision_protocol": "fp32_trainables_v1", "dtype": "bf16",
                "task_instruction_version": "legacy_v1", "max_txt_len": 1536, "seed": 44,
                "new_optimizer_and_schedule": True, "initialization": initial,
                "data_sha256": contract["split_sha256"]["train"], "original_full_data": contract,
                "checkpoint_steps": [s for s in SAVED_STEPS if s < total],
                "sft_recipe": {"protocol": "full_original_qa_two_stage_sft_v1",
                    "language_freeze_steps": args.freeze_steps, "language_warmup_steps": args.language_warmup,
                    "qformer_text_lr": 2e-5,
                    "schedule": "bridge_global_cosine_language_delayed_cosine_v1",
                    "parameter_groups": ["language", "projector", "qformer", "qformer_text"]}}
    configs = {}
    folders = {"visual": args.control_dir, "question": args.train_dir}
    if len({p.resolve() for p in (*folders.values(), args.initial_dir, args.retained_dir)}) != 4:
        raise ValueError("new runs must use separate directories from retained checkpoints")
    for branch, folder in folders.items():
        config = read(folder / "run_config.json")
        if config is None:
            continue
        wrong = [k for k, v in expected.items() if config.get(k) != v]
        question = config.get("question_qformer")
        if (wrong or config.get("language_lora") or
                (branch == "visual" and question is not None) or
                (branch == "question" and (not question or question.get("protocol") != "mpp_question_qformer_v1" or
                    question.get("mode") != "question" or question.get("max_question_tokens") != args.question_max_len))):
            raise ValueError("full SFT configuration differs for %s: %s" % (branch, wrong))
        configs[branch] = config
    return contract, parent, configs


def preflight(args):
    print(json.dumps({"status": "checking_full_original_qa", "data_dir": str(args.data_dir)}), flush=True)
    contract, _, _ = validate_configs(args)
    if not 0 <= args.freeze_steps < budget(contract["records"]):
        raise ValueError("freeze period exceeds the original-QA budget")
    from transformers import AutoTokenizer, BertTokenizer
    from mpp_video_qa.question_qformer import text_identity
    from mpp_video_qa.data import cache_file
    identity = text_identity(args.bert_dir, args.question_max_len)
    rows = []
    for split in ("train", "dev_action", "dev_scenery", "eval"):
        rows.extend(load_jsonl(args.data_dir / (split + ".jsonl")))
    questions = list(dict.fromkeys(r["question"] for r in rows))
    # Original answer[0] is the SFT target. References stay untouched for Judge.
    answers = list(dict.fromkeys(str(r["answers"][0]) for r in rows))
    bert = BertTokenizer.from_pretrained(str(args.bert_dir), do_lower_case=True, local_files_only=True)
    longest_question = 0
    for start in range(0, len(questions), 256):
        ids = bert(questions[start:start + 256], truncation=False).input_ids
        longest_question = max(longest_question, max(map(len, ids)))
    if longest_question > args.question_max_len:
        raise ValueError("original question exceeds the Q-Former budget; no silent truncation")
    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_dir), use_fast=False, trust_remote_code=True,
                                             local_files_only=True)
    newline = tokenizer("\n").input_ids
    system_len = 1 + len(tokenizer("system").input_ids) + len(newline) + len(tokenizer(SYSTEM_MESSAGE).input_ids) + 1 + len(newline)
    fixed_prompt = system_len + 1 + len(tokenizer("user").input_ids) + len(newline) + 1 + len(newline) + 1 + len(tokenizer("assistant").input_ids) + len(newline)
    question_lengths, answer_lengths = {}, {}
    visual_id = tokenizer("<|extra_0|>").input_ids[0]
    for start in range(0, len(questions), 128):
        batch = questions[start:start + 128]
        prompts = [build_user_prompt(q, 5).replace("<ImageHere>", "<|extra_0|>" * 32) for q in batch]
        ids = tokenizer(prompts, truncation=False).input_ids
        if any(item.count(visual_id) != 160 for item in ids):
            raise ValueError("original five-frame placeholders do not produce 160 visual tokens")
        question_lengths.update((q, fixed_prompt + len(item)) for q, item in zip(batch, ids))
    for start in range(0, len(answers), 256):
        batch = answers[start:start + 256]
        ids = tokenizer(batch, truncation=False).input_ids
        answer_lengths.update((a, len(item)) for a, item in zip(batch, ids))
    max_supervised = max(question_lengths[r["question"]] + answer_lengths[str(r["answers"][0])] + 1 + len(newline) for r in rows)
    max_generation = max(question_lengths.values()) + 96
    if max(max_supervised, max_generation) > 1536:
        raise ValueError("full original QA exceeds Qwen max_txt_len=1536; no silent truncation")
    segments = {r["segment_id"] for r in rows}
    missing = [s for s in segments if not cache_file(args.cache_dir, s).is_file()]
    if missing:
        raise ValueError("missing original EVA segments: %d" % len(missing))
    result = {"check_passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "original_full_data": contract, "records_checked": len(rows), "segments": len(segments),
              "max_question_tokens": longest_question, "max_supervised_sequence": max_supervised,
              "max_generation_prompt_plus_96": max_generation, "max_txt_len": 1536,
              "question_qformer": identity, "optimizer_steps_total": budget(contract["records"]),
              "epochs": 1, "world_size": 4, "effective_global_batch_size": 16,
              "new_training_started": (args.train_dir / "run_config.json").is_file()}
    write(args.eval_dir / "preflight.json", result)
    return result


def evaluation_folder(args, key, split):
    if key in ("parent", "retained"):
        prefix = "baseline" if key == "parent" else "sft_control_final"
        historical = args.history_dir / (prefix + "_" + split)
        if (historical / "scored/summary.json").is_file():
            return historical
    return args.eval_dir / (key + "_" + split)


def model_specs(args):
    from scripts.report_feedback_opd_v2 import bundle_identity
    from scripts.report_language_lora_sft import validate_saved_language_adapter
    from mpp_video_qa.question_qformer import QUESTION_FORMAT, question_sidecar
    contract, parent, configs = validate_configs(args)
    models = {"parent": parent, "retained": bundle_identity(args.retained_dir / "final")}
    specs, complete = {}, {}
    for branch in BRANCHES:
        folder = args.control_dir if branch == "visual" else args.train_dir
        config = configs.get(branch)
        if config is None:
            continue
        summary = read(folder / "summary.json") or {}
        complete[branch] = bool(summary.get("completed") and summary.get("optimizer_steps") == budget(contract["records"]))
        for label, step in labels(contract["records"]):
            bundle = folder / label
            metadata = read(bundle / "metadata.json")
            if metadata is None:
                if complete[branch]:
                    raise ValueError("completed full SFT run is missing a planned checkpoint: %s" % bundle)
                continue
            expected_stage = {"protocol": "full_original_qa_two_stage_sft_v1", "completed_steps": step,
                              "phase_next_update": "joint" if step >= args.freeze_steps else "bridge_adaptation",
                              "group_names": ["language", "projector", "qformer"] +
                                             (["qformer_text"] if branch == "question" else [])}
            if (metadata.get("identity") != config or metadata.get("optimizer_steps") != step or
                    metadata.get("sft_stage") != expected_stage):
                raise ValueError("checkpoint metadata differs: %s" % bundle)
            if label == "final" and not complete[branch]:
                raise ValueError("final weights lack a complete training summary")
            validate_saved_language_adapter(bundle / "adapter", "qkv")
            sidecar = read(question_sidecar(bundle / "qformer_adapter.pt"))
            if branch == "question":
                if (not sidecar or sidecar.get("format") != QUESTION_FORMAT or
                        sidecar.get("question_identity") != config["question_qformer"]):
                    raise ValueError("question checkpoint lacks matching complete text metadata")
            elif sidecar is not None:
                raise ValueError("visual-only control unexpectedly contains question weights")
            key = branch + "_" + label
            models[key] = bundle_identity(bundle)
            specs[key] = {"branch": branch, "step": step, "label": label,
                          "bundle": str(bundle.resolve()), "metadata_sha256": file_sha256(bundle / "metadata.json")}
    return models, specs, configs, complete


def select_checkpoints(metrics, specs):
    selected = {}
    for branch in BRANCHES:
        keys = sorted((k for k in metrics if specs.get(k, {}).get("branch") == branch), key=lambda k: specs[k]["step"])
        if keys:
            selected[branch] = max(keys, key=lambda k: metrics[k]["lingo_judge_accuracy"])
    return selected


def build_report(args, official=False):
    from scripts.report_full_qformer_scale import load_result
    from scripts.report_answer_opsd import JUDGE_PROTOCOL, paired_metrics
    from scripts.report_preference import summarize, cluster_interval
    models, specs, configs, complete = model_specs(args)
    dev_hashes = {s: file_sha256(args.data_dir / ("dev_%s.jsonl" % s)) for s in ("action", "scenery")}
    dev = read(args.eval_dir / "report.json") if official else None
    if official:
        if (not dev or dev.get("protocol") != PROTOCOL or not dev.get("selection_finalized") or
                dev.get("models") != models or dev.get("checkpoint_specs") != specs or
                dev.get("split_sha256") != dev_hashes):
            raise ValueError("complete and lock matching development selection before official evaluation")
        selected = select_checkpoints(dev["dev_combined"]["models"], specs)
        if selected != dev.get("selected_checkpoints"):
            raise ValueError("development selection changed")
        keys, splits = ["parent", "retained"] + list(selected.values()), ("eval",)
    else:
        keys, splits = list(models), ("action", "scenery")
        if not any(k.startswith("question_") for k in keys):
            raise ValueError("no question-input checkpoint is available for development comparison")
    result = {"protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
              "judge_protocol": JUDGE_PROTOCOL, "selection_split": "dev", "models": models,
              "checkpoint_specs": specs, "training_configs": configs, "full_budget_complete": complete,
              "matched_full_control_available": bool(complete.get("visual") and complete.get("question")),
              "split_sha256": {}, "splits": {}, "interpretation":
                  "Full original QA and bridge-first/joint ordinary CE; EVA and Qwen base remain frozen. "
                  "Retained 40k CE is a deployment reference, not a matched full-data architecture control. "
                  "The repeatedly reused official 500 is not a fresh blind test."}
    pooled, protocols = {k: [] for k in keys}, set()
    for split in splits:
        data = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        result["split_sha256"][split] = file_sha256(data)
        split_rows = {}
        for key in keys:
            folder = evaluation_folder(args, key, split)
            rows, protocol = load_result(folder, data, models[key])
            run = read(folder / "run_manifest.json")
            question = configs["question"]["question_qformer"] if key.startswith("question_") else None
            if (run.get("question_qformer") != question or run.get("max_new_tokens") != 96 or
                    run.get("batch_size") != 1 or run.get("max_records") != 0):
                raise ValueError("unexpected question bridge or free-generation protocol: %s" % folder)
            protocols.add(protocol)
            split_rows[key] = rows
            pooled[key].extend(rows)
        result["splits"][split] = {"models": {k: summarize(v) for k, v in split_rows.items()}}
    if len(protocols) != 1:
        raise ValueError("free-generation settings differ")
    metrics = {k: summarize(v) for k, v in pooled.items()}

    def pair(a, b):
        return dict(paired_metrics(pooled[a], pooled[b]), **cluster_interval(pooled[a], pooled[b]))

    if not official:
        selected = select_checkpoints(metrics, specs)
        same_step = {}
        for key in metrics:
            if key.startswith("question_"):
                other = "visual_" + key.split("_", 1)[1]
                if other in metrics:
                    same_step[key.split("_", 1)[1]] = pair(other, key)
        result["dev_combined"] = {"models": metrics, "same_step": same_step}
    chosen = ["parent", "retained"] + list(selected.values())
    comparisons = {a + "_vs_" + b: pair(a_key, selected[b])
                   for a, a_key in (("parent", "parent"), ("retained", "retained")) for b in selected}
    if "visual" in selected and "question" in selected:
        comparisons["visual_vs_question"] = pair(selected["visual"], selected["question"])
    result.update(selected_checkpoints=selected,
                  selected_model=dev["selected_model"] if official else max(chosen, key=lambda k: metrics[k]["lingo_judge_accuracy"]),
                  selection_finalized=bool(complete.get("question") and all(complete.values())),
                  selected_comparison={"models": {k: metrics[k] for k in chosen}, "paired": comparisons})
    if official:
        result["dev_report_sha256"] = file_sha256(args.eval_dir / "report.json")
    if "question" in selected:
        retained_ids = {r["question_id"] for r in pooled["retained"] if r["lingo_judge_correct"]}
        result["by_retained_correctness"] = {}
        for name, is_correct in (("retained_correct", True), ("retained_wrong", False)):
            a = [r for r in pooled["retained"] if (r["question_id"] in retained_ids) == is_correct]
            b = [r for r in pooled[selected["question"]] if (r["question_id"] in retained_ids) == is_correct]
            if a:
                result["by_retained_correctness"][name] = dict(paired_metrics(a, b), **cluster_interval(a, b))
    result["selection_note"] = "Each completed arm selects by combined-dev Judge, ties keep earlier steps. Post-selection intervals are exploratory."
    return result


def save_report(args, result, official=False):
    stem = "official_report" if official else "report"
    write(args.eval_dir / (stem + ".json"), result)
    text = ["# Complete original-QA question Q-Former SFT", "", result["interpretation"], "",
            "| Split | Model | N | Judge | Token F1 |", "|---|---|---:|---:|---:|"]
    blocks = list(result["splits"].items())
    if "dev_combined" in result:
        blocks.append(("dev_combined", result["dev_combined"]))
    for split, values in blocks:
        for key, m in values["models"].items():
            text.append("| %s | %s | %d | %.2f%% | %.4f |" % (split, key, m["records"], 100*m["lingo_judge_accuracy"], m["token_f1"]))
    for name, value in result["selected_comparison"]["paired"].items():
        m, ci = value["lingo_judge_correct"], value["net_change_95pct_interval"]
        text.append("\n%s: fixed %d / regressed %d / net %+.2fpp; video-bootstrap 95%% interval [%+.2f, %+.2f]pp." %
                    (name, m["fixed"], m["regressed"], 100*m["net_change_mean"], 100*ci[0], 100*ci[1]))
    text += ["", "Matched complete-data control available: `%s`." % result["matched_full_control_available"],
             "Selection finalized: `%s`." % result["selection_finalized"], result["selection_note"],
             "", "Selected on dev: `%s`." % result["selected_model"]]
    (args.eval_dir / (stem + ".md")).write_text("\n".join(text) + "\n")
    return {"json": str(args.eval_dir / (stem + ".json")), "selected_checkpoints": result["selected_checkpoints"],
            "selection_finalized": result["selection_finalized"]}


def status(args):
    manifest = read(args.data_dir / "manifest.json") or {}
    records = manifest.get("records", 0)
    result = {"records": records, "optimizer_steps_total_per_arm": budget(records), "training": {}, "evaluation": {}}
    totals = {s: line_count(args.data_dir / ("eval.jsonl" if s == "eval" else "dev_%s.jsonl" % s)) for s in ("action", "scenery", "eval")}
    for branch, folder in (("visual", args.control_dir), ("question", args.train_dir)):
        summary, progress = read(folder / "summary.json"), read(folder / "progress.json")
        failures = sorted(folder.glob("failure_rank_*.json"), key=lambda p: p.stat().st_mtime)
        if summary:
            value = dict(summary, status="completed")
        elif failures and (not (folder / "progress.json").is_file() or failures[-1].stat().st_mtime >= (folder / "progress.json").stat().st_mtime):
            value = read(failures[-1])
        else:
            value = progress or {"status": "not_started"}
        if value.get("updated_unix_seconds"):
            value["seconds_since_update"] = round(time.time() - value["updated_unix_seconds"], 1)
        if value.get("rank0_pid") and not summary:
            try:
                os.kill(value["rank0_pid"], 0)
                value["rank0_process_alive"] = True
            except ProcessLookupError:
                value.update(rank0_process_alive=False, status="interrupted")
            except PermissionError:
                value["rank0_process_alive"] = None
        result["training"][branch] = value
        if summary or progress:
            fields = ("status", "records", "optimizer_step", "optimizer_steps", "optimizer_steps_total",
                      "progress_percent", "sft_phase", "sft_stage", "ce_mean", "gradient_norm",
                      "parameter_group_learning_rates", "parameter_group_gradient_norms",
                      "elapsed_seconds_this_run", "recent_seconds_per_update", "estimated_remaining_hours",
                      "seconds_since_update", "rank0_process_alive", "loading_rank", "loaded_ranks", "world_size",
                      "gpu_memory", "gpu_memory_rank0", "final_bundle")
            if value.get("status") not in ("failed", "interrupted"):
                result["training"][branch] = {k: value[k] for k in fields if k in value}
        for label, _ in labels(records):
            for split, total in totals.items():
                key = branch + "_" + label + "_" + split
                target = args.eval_dir / key
                result["evaluation"][key] = {"generated": line_count(target / "predictions.jsonl"), "total": total,
                    "scored": (target / "scored/summary.json").is_file(),
                    "score_progress": read(target / "scored/score_progress.json")}
    result.update(dev_report_ready=(args.eval_dir / "report.json").is_file(),
                  official_report_ready=(args.eval_dir / "official_report.json").is_file())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "initial-dir", "retained-dir", "history-dir", "train-dir", "control-dir", "eval-dir", "bert-dir", "qwen-dir", "cache-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--freeze-steps", type=int, default=1000)
    p.add_argument("--language-warmup", type=int, default=200)
    p.add_argument("--question-max-len", type=int, default=128)
    p.add_argument("--action", choices=("check", "status", "list_models", "eval_path", "selected", "selected_all", "report", "report_eval"), default="status")
    p.add_argument("--model", default="question")
    p.add_argument("--split", choices=("action", "scenery", "eval"), default="action")
    a = p.parse_args()
    if a.action == "eval_path":
        print(evaluation_folder(a, a.model, a.split)); return
    if a.action in ("selected", "selected_all"):
        dev = read(a.eval_dir / "report.json") or {}
        if not dev.get("selection_finalized"):
            raise ValueError("complete full-budget development evaluation before official selection")
        selected = select_checkpoints(dev["dev_combined"]["models"], dev["checkpoint_specs"])
        if selected != dev.get("selected_checkpoints"):
            raise ValueError("development selection changed")
        if a.action == "selected_all":
            print("\n".join(["parent", "retained"] + list(selected.values())))
        else:
            print(selected[a.model])
        return
    if a.action == "list_models":
        print("\n".join(model_specs(a)[0])); return
    value = (preflight(a) if a.action == "check" else status(a) if a.action == "status" else
             save_report(a, build_report(a, a.action == "report_eval"), a.action == "report_eval"))
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
