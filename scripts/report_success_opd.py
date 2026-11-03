#!/usr/bin/env python3
"""Validate and report the four-arm success-retention OPD pilot."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl, TASK_INSTRUCTIONS
from scripts.generate_answers import write_json_atomic
from scripts.report_answer_opsd import JUDGE_PROTOCOL, line_count, paired_metrics, read_json
from scripts.report_feedback_opd import identities
from scripts.report_feedback_opd_v2 import bundle_identity
from scripts.report_full_qformer_scale import load_result
from scripts.report_preference import cluster_interval, summarize

ARMS = ("ce", "ce_keep", "opd", "opd_keep")
LABELS = {"checkpoint-000250": 250, "final": 500}
PAIRS = (("ce", "opd"), ("ce_keep", "opd_keep"), ("opd", "opd_keep"), ("ce", "ce_keep"))
PROTOCOL = "success_retention_opd_4k_factorial_v1"
DIFFERENT_KEYS = {"training_mode", "keep_weight", "probe_report_sha256"}


def arm_mode(arm):
    if arm not in ARMS:
        raise ValueError("unknown arm: %s" % arm)
    return "feedback_opd" if arm.startswith("opd") else "sft_control"


def validate_configs(configs):
    """Only OPD and keep are experimental factors; all other settings match."""
    if not configs:
        return
    required = {"records": 4000, "epochs": 2, "total_steps": 500,
                "checkpoint_steps": [250], "global_batch_size": 16,
                "lr": 2.5e-6, "projector_lr": 2.5e-6, "qformer_lr": 5e-6,
                "ce_weight": 1., "opd_weight": 1., "weight_decay": .01,
                "warmup_ratio": .03, "seed": 44, "teacher_mode": "reference_only",
                "feature_stage": "eva", "precision_protocol": "fp32_trainables_v1",
                "dtype": "bf16", "task_instruction_version": "legacy_v1",
                "max_txt_len": 1536, "max_new_tokens": 96, "objective": "jsd",
                "keep_objective": "forward_kl"}
    reference = next(iter(configs.values()))
    for arm, config in configs.items():
        wrong = [k for k, v in required.items() if config.get(k) != v]
        world = config.get("world_size", 0)
        if (wrong or world not in (2, 3, 4) or config.get("grad_accumulation") != math.ceil(16 / world)
                or config.get("training_mode") != arm_mode(arm)
                or config.get("keep_weight") != float(arm.endswith("_keep"))
                or not config.get("success_bank")):
            raise ValueError("invalid matched pilot recipe for %s: %s" % (arm, wrong))
        changed = [k for k in set(reference) | set(config)
                   if k not in DIFFERENT_KEYS and reference.get(k) != config.get(k)]
        if changed:
            raise ValueError("four-arm settings differ: %s" % sorted(changed))
    probes = {c.get("probe_report_sha256") for a, c in configs.items() if a.startswith("opd")}
    if None in probes or len(probes) > 1:
        raise ValueError("OPD arms require the same validated teacher probe")


def checked_configs(args):
    train = args.data_dir / "train.jsonl"
    rows = load_jsonl(train)
    if len(rows) != 4000 or len({r["question_id"] for r in rows}) != 4000:
        raise ValueError("the pilot requires 4000 unique original QA")
    if any(len(r["frames"]) != 5 or r.get("feedback_training_source") != "train" for r in rows):
        raise ValueError("only original five-frame training QA are allowed")
    parent = identities(args.initial_dir, True)
    initial = {"adapter": parent["adapter_identity"], "projector": parent["projector_identity"],
               "qformer": parent["qformer_adapter_sha256"]}
    parent_config = read_json(args.initial_dir / "run_config.json") or {}
    parent_summary = read_json(args.initial_dir / "summary.json") or {}
    if (not parent_summary.get("completed") or not parent_config.get("total_steps")
            or parent_summary.get("optimizer_steps") != parent_config["total_steps"]):
        raise ValueError("original SFT parent is incomplete")
    from mpp_video_qa.success_bank import bank_contract, read_records, validate_data
    validate_data(train, rows)
    bank = bank_contract(args.bank_dir)
    manifest = read_json(args.bank_dir / "bank_manifest.json")
    if (bank["data_sha256"] != file_sha256(train) or bank["records"] != 4000 or not 0 < bank["eligible"] <= 4000
            or manifest["identity"]["model_identity"]["initialization"] != initial):
        raise ValueError("success bank differs from training data or original parent")
    bank_rows = read_records(args.bank_dir / "records.jsonl", {r["question_id"]: r for r in rows}, manifest["identity"])
    if len(bank_rows) != 4000 or sum(r["eligible"] for r in bank_rows) != bank["eligible"]:
        raise ValueError("success bank does not cover the original QA")
    configs = {a: read_json(args.train_dir / a / "run_config.json") for a in ARMS}
    configs = {a: c for a, c in configs.items() if c is not None}
    validate_configs(configs)
    for arm, config in configs.items():
        if (config.get("data_sha256") != file_sha256(train) or config.get("initialization") != initial
                or config.get("success_bank") != bank):
            raise ValueError("data, original parent or success bank differs: %s" % arm)
    roots = [args.initial_dir, args.bank_dir, args.data_dir, args.eval_dir, args.train_dir]
    if len({p.resolve() for p in roots}) != len(roots):
        raise ValueError("new output directories must be isolated")
    return configs, parent


def checkpoint_valid(folder, config, label):
    bundle = folder / label
    metadata = read_json(bundle / "metadata.json") or {}
    if metadata.get("identity") != config or metadata.get("optimizer_steps") != LABELS[label]:
        raise ValueError("checkpoint identity or update count differs: %s" % bundle)
    required = ("adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                "projector.pt", "qformer_adapter.pt", "training_state.pt")
    if any(not (bundle / f).is_file() or (bundle / f).stat().st_size == 0 for f in required):
        raise ValueError("checkpoint has missing/empty weights or training state: %s" % bundle)
    return metadata


def completed_arm(folder, config):
    summary = read_json(folder / "summary.json")
    if summary is None:
        return False
    totals = summary.get("cumulative_metrics", {})
    if (not summary.get("completed") or summary.get("optimizer_steps") != 500
            or summary.get("optimizer_steps_total") != 500 or summary.get("records") != 4000
            or summary.get("training_mode") != config["training_mode"] or totals.get("examples") != 8000):
        raise ValueError("invalid or incomplete training summary: %s" % folder)
    expected_rollouts = 8000 if config["training_mode"] == "feedback_opd" else 0
    if totals.get("rollouts") != expected_rollouts:
        raise ValueError("rollout budget differs from the arm: %s" % folder)
    if config["training_mode"] == "sft_control" and any(totals.get(k, 0) for k in ("jsd_sum", "teacher_attempts")):
        raise ValueError("CE arm unexpectedly used online OPD: %s" % folder)
    expected_keep = 2 * config["success_bank"].get("eligible", 0) if config["keep_weight"] else 0
    if totals.get("keep_examples", 0) != expected_keep:
        raise ValueError("successful-trajectory replay count differs: %s" % folder)
    if not config["keep_weight"] and any(totals.get(k, 0) for k in ("keep_kl_sum", "keep_tokens")):
        raise ValueError("disabled keep arm unexpectedly used retention")
    if any(not math.isfinite(totals.get(k, 0)) for k in ("ce_sum", "jsd_sum", "keep_kl_sum")):
        raise ValueError("nonfinite accumulated losses")
    if any(summary.get("teacher_tracked_parameter_deltas", {}).values()):
        raise ValueError("frozen teacher weights changed")
    for label in LABELS:
        checkpoint_valid(folder, config, label)
    return True


def model_specs(args):
    configs, parent = checked_configs(args)
    if set(configs) != set(ARMS):
        raise ValueError("all four matched arms must finish before evaluation")
    models, specs = {"parent": parent}, {}
    for arm in ARMS:
        folder = args.train_dir / arm
        if not completed_arm(folder, configs[arm]):
            raise ValueError("training incomplete: %s" % arm)
        for label, step in LABELS.items():
            bundle = folder / label
            key = arm + "__" + label
            models[key] = bundle_identity(bundle)
            specs[key] = {"arm": arm, "label": label, "step": step,
                          "metadata_sha256": file_sha256(bundle / "metadata.json")}
    return models, specs, configs


def data_path(args, split):
    if split not in ("action", "scenery", "eval"):
        raise ValueError("unknown split")
    return args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)


def evaluation_path(args, key, split, identity=None):
    # Historical manifests omit base/context provenance, so this experiment regenerates its parent.
    return args.eval_dir / (key + "_" + split)


def load_evaluation(folder, source, identity, config):
    """One validator is used by reports and by the runner's reuse decision."""
    run = read_json(folder / "run_manifest.json") or {}
    if (any(k.startswith("region_") for k in run) or "question_qformer" in run
            or run.get("max_new_tokens") != 96 or run.get("batch_size") != 1
            or run.get("max_records") != 0 or run.get("max_txt_len") != 1536
            or run.get("qformer_mode") != "auto"
            or run.get("task_instruction") != TASK_INSTRUCTIONS["legacy_v1"]
            or run.get("base_checkpoint_sha256") != config["base_checkpoint_sha256"]
            or run.get("bridge_init_sha256") != config["bridge_init_sha256"]
            or run.get("cache_dir") != config["cache_dir"]
            or run.get("generation_policy") != {"do_sample": False, "num_beams": 1, "top_p": 1.0, "top_k": 50}):
        raise ValueError("unexpected five-frame/base/context/greedy inference protocol: %s" % folder)
    rows, protocol = load_result(folder, source, identity)
    if any("region_observation" in r for r in rows):
        raise ValueError("unexpected region observations in five-frame predictions")
    return rows, protocol


def interaction(rows, draws=1000):
    by_id = {a: {r["question_id"]: r for r in rows[a]} for a in ARMS}
    ids = set(by_id["ce"])
    if not ids or any(set(v) != ids or len(v) != len(rows[a]) for a, v in by_id.items()):
        raise ValueError("interaction requires identical unique QA across all four arms")
    groups = defaultdict(list)
    for qid in sorted(ids):
        segment = by_id["ce"][qid]["segment_id"]
        if any(v[qid]["segment_id"] != segment for v in by_id.values()):
            raise ValueError("interaction video identity differs")
        c = {a: int(by_id[a][qid]["lingo_judge_correct"]) for a in ARMS}
        groups[segment].append(c["opd_keep"] - c["ce_keep"] - c["opd"] + c["ce"])
    values = [(sum(v), len(v)) for v in groups.values()]
    rng, samples = random.Random(42), []
    for _ in range(draws):
        picked = rng.choices(values, k=len(values))
        samples.append(sum(v[0] for v in picked) / sum(v[1] for v in picked))
    samples.sort()
    return {"definition": "(OPD_keep - CE_keep) - (OPD - CE)", "records": len(ids),
            "mean": sum(v[0] for v in values) / len(ids), "unit": "segment_id",
            "segments": len(values), "draws": draws,
            "interval_95pct": [samples[int(.025 * draws)], samples[min(draws - 1, int(.975 * draws))]]}


def comparison(rows, strata=True):
    block = {"models": {a: summarize(rows[a]) for a in ("parent",) + ARMS},
             "paired": {a + "_vs_" + b: dict(paired_metrics(rows[a], rows[b]),
                        **cluster_interval(rows[a], rows[b])) for a, b in PAIRS},
             "interaction": interaction(rows)}
    if strata:
        block["by_parent"] = {}
        for name, correct in (("parent_correct", True), ("parent_wrong", False)):
            selected = {r["question_id"] for r in rows["parent"] if bool(r["lingo_judge_correct"]) == correct}
            sub = {a: [r for r in rows[a] if r["question_id"] in selected] for a in ("parent",) + ARMS}
            block["by_parent"][name] = comparison(sub, False) if selected else {"records": 0}
    return block


def select_checkpoints(metrics, specs):
    return {arm: max(sorted((k for k, v in specs.items() if v["arm"] == arm),
                           key=lambda k: specs[k]["step"]),
                     key=lambda k: metrics[k]["lingo_judge_accuracy"]) for arm in ARMS}


def source_signature(folder):
    return {f: file_sha256(folder / f) for f in ("run_manifest.json", "predictions.jsonl",
            "scored/summary.json", "scored/scored_predictions.jsonl")}


def load_selection(args, models, specs, configs):
    dev = read_json(args.eval_dir / "report.json") or {}
    hashes = {s: file_sha256(data_path(args, s)) for s in ("action", "scenery")}
    if (dev.get("protocol") != PROTOCOL or not dev.get("selection_finalized")
            or dev.get("models") != models or dev.get("checkpoint_specs") != specs
            or dev.get("training_configs") != configs or dev.get("split_sha256") != hashes):
        raise ValueError("complete matching development selection is required")
    expected_paths = {str(evaluation_path(args, key, split).resolve())
                      for key in models for split in ("action", "scenery")}
    if set(dev.get("evaluation_sources", {})) != expected_paths:
        raise ValueError("development evaluation source coverage changed")
    pooled = {key: [] for key in models}
    for split in ("action", "scenery"):
        for key in models:
            folder = evaluation_path(args, key, split)
            if source_signature(folder) != dev["evaluation_sources"][str(folder.resolve())]:
                raise ValueError("development evaluation changed after selection")
            rows, _ = load_evaluation(folder, data_path(args, split), models[key], configs["ce"])
            pooled[key].extend(rows)
    metrics = {key: summarize(rows) for key, rows in pooled.items()}
    if dev.get("dev_combined", {}).get("models") != metrics:
        raise ValueError("development report metrics differ from hashed source predictions")
    if select_checkpoints(metrics, specs) != dev.get("selected_checkpoints"):
        raise ValueError("development selection changed")
    return dev


def build_report(args, official=False):
    models, specs, configs = model_specs(args)
    dev = load_selection(args, models, specs, configs) if official else None
    selected = dev["selected_checkpoints"] if official else None
    config = configs["ce"]
    keys = ["parent"] + list(selected.values()) if official else list(models)
    result = {"protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
              "judge_protocol": JUDGE_PROTOCOL, "models": models, "checkpoint_specs": specs,
              "training_configs": configs, "selection_split": "dev", "selection_finalized": True,
              "split_sha256": {}, "splits": {}, "evaluation_sources": {},
              "interpretation": "Four arms share 4000 original QA, two epochs, 500 updates and five frames. "
                "CE is present in every arm; keep preserves fixed successful parent trajectories without references. "
                "Matched updates do not mean matched GPU time. Development-selected intervals are exploratory; "
                "the repeatedly used official 500 is not a new blind test."}
    pooled, protocols = {k: [] for k in keys}, set()
    for split in (("eval",) if official else ("action", "scenery")):
        path = data_path(args, split)
        result["split_sha256"][split] = file_sha256(path)
        rows = {}
        for key in keys:
            folder = evaluation_path(args, key, split, models[key])
            rows[key], protocol = load_evaluation(folder, path, models[key], config)
            protocols.add(protocol)
            pooled[key].extend(rows[key])
            result["evaluation_sources"][str(folder.resolve())] = source_signature(folder)
        result["splits"][split] = {"models": {k: summarize(v) for k, v in rows.items()}}
    if len(protocols) != 1:
        raise ValueError("generation protocols differ across arms/splits")
    metrics = {k: summarize(v) for k, v in pooled.items()}
    if not official:
        selected = select_checkpoints(metrics, specs)
        result["dev_combined"] = {"models": metrics, "same_step": {}}
        for label in LABELS:
            rows = {"parent": pooled["parent"], **{a: pooled[a + "__" + label] for a in ARMS}}
            result["dev_combined"]["same_step"][label] = comparison(rows)
    result["selected_checkpoints"] = selected
    chosen = {"parent": pooled["parent"], **{a: pooled[selected[a]] for a in ARMS}}
    result["selected_comparison"] = comparison(chosen)
    result["selection_note"] = "Each arm independently selects step 250 or 500 by combined-dev Judge; ties choose 250. Official scores never select models."
    if official:
        result["dev_report_sha256"] = file_sha256(args.eval_dir / "report.json")
    return result


def save_report(args, result, official=False):
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    report_name = "official_report" if official else "report"
    write_json_atomic(args.eval_dir / (report_name + ".json"), result)
    lines = ["# Success-retention OPD: four-arm pilot", "", result["interpretation"], "",
             "| Model | N | Judge |", "|---|---:|---:|"]
    for arm, metric in result["selected_comparison"]["models"].items():
        lines.append("| %s | %d | %.2f%% |" % (arm, metric["records"], 100 * metric["lingo_judge_accuracy"]))
    blocks = [("selected", result["selected_comparison"])]
    blocks.extend(result.get("dev_combined", {}).get("same_step", {}).items())
    for label, block in blocks:
        for name, value in block["paired"].items():
            score, ci = value["lingo_judge_correct"], value["net_change_95pct_interval"]
            lines.append("\n%s/%s: fixed %d, regressed %d, net %+.2f pp; video-bootstrap 95%% CI [%+.2f, %+.2f] pp." %
                         (label, name, score["fixed"], score["regressed"], score["net_change_mean"] * 100, ci[0] * 100, ci[1] * 100))
        value = block["interaction"]
        lines.append("\n%s interaction: %+.2f pp; video-bootstrap 95%% CI [%+.2f, %+.2f] pp." %
                     (label, value["mean"] * 100, value["interval_95pct"][0] * 100, value["interval_95pct"][1] * 100))
        for stratum, sub in block["by_parent"].items():
            if "paired" in sub:
                for name, value in sub["paired"].items():
                    score = value["lingo_judge_correct"]
                    lines.append("\n%s/%s/%s: fixed %d, regressed %d, net %+.2f pp." %
                                 (label, stratum, name, score["fixed"], score["regressed"], score["net_change_mean"] * 100))
    lines += ["", result["selection_note"], "", json.dumps(result["selected_checkpoints"])]
    (args.eval_dir / (report_name + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_state(args, resume=False):
    configs, _ = checked_configs(args)
    arm = args.arm
    arm_mode(arm)
    world = args.launch_world_size
    if world not in (2, 3, 4):
        raise ValueError("training requires 2, 3 or 4 visible GPUs")
    if any(c["world_size"] != world for c in configs.values()):
        raise ValueError("all arms and resumes must use the same GPU count")
    folder, config = args.train_dir / arm, configs.get(arm)
    if config:
        if completed_arm(folder, config):
            return "complete"
        if not resume:
            raise ValueError("unfinished arm; use resume %s explicitly" % arm)
        checkpoint_valid(folder, config, "checkpoint-000250")
        return str((folder / "checkpoint-000250").resolve())
    if resume:
        raise ValueError("no validated saved run exists to resume")
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("nonempty output without run identity; use a new SUCCESS_OPD_TRAIN_DIR")
    return "new"


def status(args):
    result = {"process_liveness_verified": False, "training": {}, "evaluation": {},
              "bank_manifest": read_json(args.bank_dir / "bank_manifest.json"),
              "bank_shards": {p.parent.name: read_json(p) for p in sorted(args.bank_dir.glob("shard-*/progress.json"))}}
    try:
        configs, _ = checked_configs(args)
        result["identity_validated"] = True
    except (OSError, ValueError, KeyError) as error:
        configs = {a: read_json(args.train_dir / a / "run_config.json") for a in ARMS}
        result["identity_validated"] = False
        if (isinstance(error, FileNotFoundError) and error.filename and
                Path(error.filename).resolve() == (args.bank_dir / "bank_manifest.json").resolve() and
                not any((args.train_dir / arm / "run_config.json").exists() for arm in ARMS)):
            result["validation_pending"] = "waiting for bank generation and merge; training starts after bank validation"
        else:
            result["validation_error"] = str(error)
    for arm in ARMS:
        folder = args.train_dir / arm
        config = configs.get(arm)
        value = read_json(folder / "progress.json") or {"status": "not_started"}
        if config:
            try:
                if completed_arm(folder, config) and result["identity_validated"]:
                    value = {"status": "completed_verified", "optimizer_steps": 500, "optimizer_steps_total": 500}
                elif value.get("status") == "running":
                    value["status"] = "progress_recorded_liveness_unverified"
            except (OSError, ValueError, KeyError) as error:
                value = {"status": "invalid_or_incomplete", "error": str(error)}
        if "updated_unix_seconds" in value:
            value["seconds_since_update"] = round(time.time() - value["updated_unix_seconds"], 1)
        value["failures"] = [str(p) for p in sorted(folder.glob("failure_rank_*.json"))]
        value["resume_command"] = "bash scripts/run_success_opd.sh resume " + arm
        result["training"][arm] = value
        for label in LABELS:
            for split in ("action", "scenery", "eval"):
                path = args.eval_dir / (arm + "__" + label + "_" + split)
                result["evaluation"][path.name] = {"generated": line_count(path / "predictions.jsonl"),
                    "expected": line_count(data_path(args, split)),
                    "score_summary_present_unverified": (path / "scored/summary.json").exists()}
    for split in ("action", "scenery", "eval"):
        path = evaluation_path(args, "parent", split)
        result["evaluation"][path.name] = {"generated": line_count(path / "predictions.jsonl"),
            "expected": line_count(data_path(args, split)),
            "score_summary_present_unverified": (path / "scored/summary.json").exists()}
    result["report_present"] = (args.eval_dir / "report.json").is_file()
    result["official_report_present"] = (args.eval_dir / "official_report.json").is_file()
    return result


def print_status(value):
    print("File progress; process liveness is not inferred from progress timestamps.")
    bank = value["bank_manifest"]
    shards = value["bank_shards"]
    if bank:
        print("bank: %s records, %s eligible; artifact validation=%s" %
              (bank.get("records"), bank.get("eligible"), value["identity_validated"]))
    elif shards:
        print("bank: %d/%d generated across %d shards" %
              (sum(x.get("completed", 0) for x in shards.values()),
               sum(x.get("total", 0) for x in shards.values()), len(shards)))
    else:
        print("bank: not started")
    print("arm          state                                    step       ETA(h)   keep KL")
    for arm, info in value["training"].items():
        step = info.get("optimizer_step", info.get("optimizer_steps", 0))
        print("%-12s %-40s %3s/500    %-8s %s" %
              (arm, info["status"], step, info.get("estimated_remaining_hours", "-"),
               info.get("keep_kl_mean_anchored", "-")))
        if info.get("error"):
            print("  " + info["error"])
        if info.get("failures"):
            print("  failures: " + ", ".join(info["failures"]))
    dev = {k: v for k, v in value["evaluation"].items() if not k.endswith("_eval")}
    print("new dev answers: %d/%d; score summaries present (unverified): %d/%d" %
          (sum(x["generated"] for x in dev.values()), sum(x["expected"] for x in dev.values()),
           sum(x["score_summary_present_unverified"] for x in dev.values()), len(dev)))
    active = [(k, x) for k, x in dev.items() if x["generated"] and not x["score_summary_present_unverified"]]
    for name, item in active:
        print("  %s: %d/%d generated, scoring pending" % (name, item["generated"], item["expected"]))
    print("dev report=%s; official report=%s" % (value["report_present"], value["official_report_present"]))
    if value.get("validation_pending"):
        print("validation pending: " + value["validation_pending"])
    if value.get("validation_error"):
        print("validation error: " + value["validation_error"])


def main():
    p = argparse.ArgumentParser()
    defaults = {"data-dir": "datasets/processed/lingoqa_success_opd_4k_v1",
                "bank-dir": "outputs/success_opd_bank_4k_v1", "train-dir": "outputs/success_opd_4k_v1",
                "eval-dir": "outputs/success_opd_eval_4k_v1",
                "initial-dir": "outputs/mpp_lingoqa_qformer_all_1epoch_v1"}
    for name, value in defaults.items():
        p.add_argument("--" + name, type=Path, default=ROOT / value)
    p.add_argument("--action", choices=("status", "launch", "resume_path", "eval_path", "eval_ready", "selected", "report", "report_eval"), default="status")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--model")
    p.add_argument("--split", choices=("action", "scenery", "eval"))
    p.add_argument("--launch-world-size", type=int, default=4)
    p.add_argument("--json", action="store_true", help="machine-readable status")
    args = p.parse_args()
    if args.action == "status":
        value = status(args)
        if args.json:
            print(json.dumps(value, indent=2))
        else:
            print_status(value)
    elif args.action in ("launch", "resume_path"):
        print(launch_state(args, args.action == "resume_path"))
    elif args.action in ("report", "report_eval"):
        value = build_report(args, args.action == "report_eval")
        save_report(args, value, args.action == "report_eval")
        print(json.dumps({"selected_checkpoints": value["selected_checkpoints"], "report_written": True}))
    else:
        models, specs, configs = model_specs(args)
        if args.action == "selected":
            dev = load_selection(args, models, specs, configs)
            print(specs[dev["selected_checkpoints"][args.arm]]["label"])
            return
        if args.model not in models:
            raise ValueError("unknown model")
        if args.split == "eval":
            dev = load_selection(args, models, specs, configs)
            if args.model != "parent" and args.model not in dev["selected_checkpoints"].values():
                raise ValueError("official evaluation requires a dev-selected checkpoint")
        folder = evaluation_path(args, args.model, args.split, models[args.model])
        if args.action == "eval_path":
            print(folder)
        elif (folder / "scored/summary.json").is_file():
            load_evaluation(folder, data_path(args, args.split), models[args.model], configs["ce"])
            print("ready")
        else:
            print("pending")


if __name__ == "__main__":
    main()
