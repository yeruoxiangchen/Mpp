#!/usr/bin/env python3
"""Teacher screening only: prepare -> smoke -> dev600 -> paired report/review.

No training is launched. The reused official500 requires a separate action.
Internal dev originates in public LingoQA train and cannot certify teacher
generalization without an external-training overlap audit.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.distillation import assert_disjoint
from mpp_video_qa.driving_teacher import (
    PROTOCOL, MODEL_ID, MODEL_REVISION, SOURCE_REVISION, INSTRUCTION,
    read, write_json, immutable_json, write_rows, exclusive_lock,
    validate_data, validate_partial, checked_scores, paired_comparison,
)
from scripts.report_answer_opsd import JUDGE_PROTOCOL, line_count
from scripts.prepare_visual_adaptation import stratified_subset
from scripts.qwen_drive_parallel import (
    PARTITION, layout, merge_generation, merge_scores, parse_devices, progress_snapshot,
    score_is_ready, score_stamp, shard_folder, shard_rows,
)

RUN = ROOT / "outputs/qwen_drive_teacher_20261020"
IDENTITY_KEYS = ("adapter_identity", "projector_identity", "qformer_adapter_sha256")
CODE_FILES = (
    "mpp_video_qa/driving_teacher.py", "scripts/generate_qwen_drive_teacher.py",
    "scripts/qwen_drive_screen.py", "scripts/evaluate_answers.py",
    "scripts/run_qwen_drive_teacher.sh", "scripts/compare_teachers.py",
    "scripts/report_preference.py", "scripts/report_answer_opsd.py",
    "scripts/qwen_drive_parallel.py",
)


def summarize(rows):
    # Keep status/argument parsing independent of PyTorch import time.
    from scripts.report_preference import summarize as impl
    return impl(rows)


def configuration(args):
    return {
        "protocol": PROTOCOL, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION, "teacher_model": str(args.model.resolve()),
        "teacher_source": str(args.source.resolve()), "teacher_python": str(args.teacher_python),
        "data_dir": str(args.data_dir.resolve()), "dev_size": args.dev_size, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "max_pixels_per_frame": args.max_pixels,
        "task_instruction": INSTRUCTION, "references_in_teacher_prompt": False,
        "automatic_training": False, "automatic_official_evaluation": False,
        "parallel_workers": len(getattr(args, "gpus", ["0"])), "partition": PARTITION,
        "code_sha256": {p: file_sha256(ROOT / p) for p in CODE_FILES},
    }


def archive_baselines(args, split, selected):
    """Reuse frozen scored predictions, binding them to their reviewed model reports."""
    from scripts.report_full_qformer_scale import load_result
    old_eval = ROOT / "outputs/feedback_opd_eval_v2"
    mechanism = ROOT / "outputs/opd_mechanism_eval_4k_v1"
    old_report = read(old_eval / "report.json")
    mech_report = read(mechanism / "report.json")
    if not mech_report.get("completed"):
        raise ValueError("parent evaluation archive is incomplete")
    source_splits = ("action", "scenery") if split == "dev" else ("eval",)
    result, sources = {}, {}
    for role in ("parent", "retained_ce"):
        if role == "parent" and split == "dev":
            report_path = mechanism / "report.json"
            identity = mech_report["model_identities"]["parent"]
            folders = {s: mechanism / ("parent_" + s) for s in source_splits}
        else:
            report_path = old_eval / "report.json"
            name = "baseline" if role == "parent" else "sft_control_final"
            identity = old_report["models"][name]
            folders = {s: old_eval / (name + "_" + s) for s in source_splits}
        pooled, artifacts = [], {str(report_path): file_sha256(report_path)}
        for source_split, folder in folders.items():
            data = args.data_dir / ("eval.jsonl" if source_split == "eval" else "dev_" + source_split + ".jsonl")
            rows, _ = load_result(folder, data, {k: identity[k] for k in IDENTITY_KEYS})
            # Revalidate duplicates, completion metadata and source labels as well.
            checked_scores(folder, load_jsonl(data))
            pooled.extend(rows)
            for suffix in ("run_manifest.json", "predictions.jsonl",
                           "scored/scored_predictions.jsonl", "scored/summary.json"):
                path = folder / suffix
                artifacts[str(path)] = file_sha256(path)
        keyed = {r["question_id"]: r for r in pooled}
        result[role] = [keyed[r["question_id"]] for r in selected]
        sources[role] = {"model_identity": {k: identity[k] for k in IDENTITY_KEYS},
                         "artifact_sha256": artifacts,
                         "note": "Archived predictions validated against source rows and reviewed report identities; weights not rehashed."}
    return result, sources


def prepare(args):
    immutable_json(args.output_dir / "experiment.json", configuration(args))
    dev = load_jsonl(args.data_dir / "dev.jsonl")
    official = load_jsonl(args.data_dir / "eval.jsonl")
    selected = stratified_subset(dev, args.dev_size, args.seed, 2)
    worker_count = len(getattr(args, "gpus", ["0"]))
    smoke = []
    for source in sorted({r["source_subset"] for r in selected}):
        cases = [r for r in selected if r["source_subset"] == source][:worker_count]
        if len(cases) != worker_count:
            raise ValueError("dev sample needs at least one QA per source for each GPU smoke")
        # Source-major ordering gives each modulo shard one Action and one Scenery.
        smoke.extend(cases)
    assert_disjoint(selected, official, [])
    splits = {"dev": selected, "official": official, "smoke": smoke}
    manifest = {
        "protocol": PROTOCOL, "sampling": "source proportional; seed 44 by default; at most two QA per video",
        "seed": args.seed, "sources": {
            str(args.data_dir / name): file_sha256(args.data_dir / name)
            for name in ("dev.jsonl", "dev_action.jsonl", "dev_scenery.jsonl", "eval.jsonl")},
        "splits": {}, "baseline_copies_sha256": {},
        "external_teacher_training_overlap": {
            "dev": "unknown: our internal dev was split from public LingoQA train",
            "official": "public evaluation split, already reused in this project; external teacher training overlap unverified",
            "unseen_teacher_generalization_verified": False,
            "teacher_qualified_for_distillation": False,
        },
    }
    for split, rows in splits.items():
        validate_data(rows, check_images=True)
        data = args.output_dir / "data" / (split + ".jsonl")
        write_rows(data, rows)
        manifest["splits"][split] = {
            "records": len(rows), "segments": len({r["segment_id"] for r in rows}),
            "source_counts": dict(Counter(r["source_subset"] for r in rows)), "sha256": file_sha256(data)}
        if split == "smoke":
            continue
        baselines, identities = archive_baselines(args, split, rows)
        for role, predictions in baselines.items():
            folder = args.output_dir / split / role
            immutable_json(folder / "run_manifest.json",
                           dict(identities[role], data_sha256=file_sha256(data)))
            write_rows(folder / "predictions.jsonl", predictions)
            write_rows(folder / "scored/scored_predictions.jsonl", predictions)
            immutable_json(folder / "scored/summary.json",
                           dict(summarize(predictions), judge_protocol=JUDGE_PROTOCOL))
            checked_scores(folder, rows)
            for name in ("run_manifest.json", "predictions.jsonl",
                         "scored/scored_predictions.jsonl", "scored/summary.json"):
                path = folder / name
                manifest["baseline_copies_sha256"][str(path)] = file_sha256(path)
    immutable_json(args.output_dir / "data/manifest.json", manifest)
    print(json.dumps({"prepared": {s: m["records"] for s, m in manifest["splits"].items()},
                      "output_dir": str(args.output_dir), "training_started": False}), flush=True)
    return manifest


def validate_prepared(args):
    if not (args.output_dir / "data/manifest.json").exists():
        return prepare(args)
    if read(args.output_dir / "experiment.json") != configuration(args):
        raise ValueError("experiment settings/code changed; use a new --output-dir")
    manifest = read(args.output_dir / "data/manifest.json")
    for path, sha in manifest["sources"].items():
        if file_sha256(path) != sha:
            raise ValueError("source data changed")
    for split, spec in manifest["splits"].items():
        data = args.output_dir / "data" / (split + ".jsonl")
        if file_sha256(data) != spec["sha256"]:
            raise ValueError("prepared split changed")
    for path, sha in manifest["baseline_copies_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError("archived student copy changed")
    return manifest


def event(args, stage, **extra):
    value = dict(stage=stage, action=args.action, pid=os.getpid(),
                 updated_unix_seconds=time.time(), **extra)
    write_json(args.output_dir / "workflow.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def child_env(offline=True, gpu=None):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def child(args, stage, command, offline=True, gpu=None):
    log = args.output_dir / "logs" / (stage + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = child_env(offline, gpu)
    cmd = [str(c) for c in command]
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n" + json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
                                       "command": cmd}, ensure_ascii=False) + "\n")
        stream.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        event(args, stage, child_pid=proc.pid, log=str(log))
        try:
            while True:
                try:
                    code = proc.wait(timeout=30)
                    if code:
                        raise RuntimeError("%s failed (exit %s); see %s" % (stage, code, log))
                    return
                except subprocess.TimeoutExpired:
                    event(args, stage, child_pid=proc.pid, log=str(log), child_alive=True)
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise


def children(args, stage, jobs):
    """One isolated CUDA-visible GPU per worker; stop siblings if any worker fails."""
    if not jobs:
        return
    processes, records = [], []
    with ExitStack() as stack:
        try:
            for job in jobs:
                log = args.output_dir / "logs" / (job["label"] + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context(log.open("a", encoding="utf-8"))
                command = [str(v) for v in job["command"]]
                stream.write("\n" + json.dumps({"command": command, "gpu": job["gpu"],
                                               "started_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                stream.flush()
                proc = subprocess.Popen(command, cwd=ROOT, env=child_env(gpu=job["gpu"]),
                                        stdout=stream, stderr=subprocess.STDOUT)
                processes.append(proc)
                records.append({"label": job["label"], "gpu": job["gpu"], "pid": proc.pid,
                                "log": str(log), "state": "running"})
            event(args, stage, workers=records)
            heartbeat = time.monotonic()
            while True:
                for proc, record in zip(processes, records):
                    code = proc.poll()
                    if code is not None:
                        record.update(exit_code=code, state="completed" if code == 0 else "failed")
                        if code:
                            raise RuntimeError("GPU %s: %s failed (exit %s); see %s" % (
                                record["gpu"], record["label"], code, record["log"]))
                if all(p.returncode is not None for p in processes):
                    event(args, stage, workers=records, workers_completed=True)
                    return
                if time.monotonic() - heartbeat >= 30:
                    event(args, stage, workers=records)
                    heartbeat = time.monotonic()
                # Short waits allow prompt detection of any sibling's failure.
                pending = next(p for p in processes if p.returncode is None)
                try:
                    pending.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            for proc in processes:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def backend_command(args, action):
    if not Path(args.teacher_python).is_file():
        raise FileNotFoundError("teacher environment missing; run: bash scripts/run_qwen_drive_teacher.sh setup")
    return [args.teacher_python, ROOT / "scripts/generate_qwen_drive_teacher.py", action,
            "--model", args.model, "--source", args.source]


def generate(args, split):
    devices = getattr(args, "gpus", ["0"])
    data = args.output_dir / "data" / (split + ".jsonl")
    folder = args.output_dir / split / "teacher"
    if len(devices) > 1:
        layout(data, folder, len(devices))
        jobs = [{"label": "%s_teacher_shard_%03d" % (split, i), "gpu": gpu,
                 "command": backend_command(args, "generate") + [
                     "--data", data, "--output-dir", shard_folder(folder, i),
                     "--max-new-tokens", args.max_new_tokens, "--max-pixels", args.max_pixels,
                     "--shard-index", i, "--num-shards", len(devices)]}
                for i, gpu in enumerate(devices)]
        children(args, split + "_teacher_parallel", jobs)
        merge_generation(data, folder, len(devices))
        return
    child(args, split + "_teacher", backend_command(args, "generate") + [
        "--data", data, "--output-dir", folder,
        "--max-new-tokens", args.max_new_tokens, "--max-pixels", args.max_pixels], gpu=devices[0])


def score(args, split):
    folder = args.output_dir / split / "teacher"
    originals = load_jsonl(args.output_dir / "data" / (split + ".jsonl"))
    predictions = load_jsonl(folder / "predictions.jsonl")
    if len(validate_partial(originals, predictions)) != len(originals):
        raise ValueError("cannot score an incomplete generation")
    devices = getattr(args, "gpus", ["0"])
    if len(devices) > 1:
        jobs = []
        for i, gpu in enumerate(devices):
            part = shard_folder(folder, i)
            selected = shard_rows(originals, i, len(devices))
            if not score_is_ready(part, selected):
                values = load_jsonl(part / "predictions.jsonl")
                if len(validate_partial(selected, values)) != len(selected):
                    raise ValueError("cannot score an incomplete shard")
                jobs.append({"label": "%s_judge_shard_%03d" % (split, i), "gpu": gpu,
                             "command": [sys.executable, ROOT / "scripts/evaluate_answers.py",
                                         "--predictions", part / "predictions.jsonl",
                                         "--output-dir", part / "scored", "--lingo-judge"]})
        children(args, split + "_judge_parallel", jobs)
        for i in range(len(devices)):
            part = shard_folder(folder, i)
            checked_scores(part, shard_rows(originals, i, len(devices)))
            write_json(part / "scored/input_identity.json", score_stamp(part))
        merge_scores(args.output_dir / "data" / (split + ".jsonl"), folder, len(devices))
        return
    stamp = score_stamp(folder)
    stamp_path = folder / "scored/input_identity.json"
    if stamp_path.exists() and read(stamp_path) == stamp:
        checked_scores(folder, originals)
        return
    child(args, split + "_judge", [
        sys.executable, ROOT / "scripts/evaluate_answers.py",
        "--predictions", folder / "predictions.jsonl", "--output-dir", folder / "scored", "--lingo-judge"],
        gpu=devices[0])
    checked_scores(folder, originals)
    write_json(stamp_path, stamp)


def smoke(args):
    generate(args, "smoke")
    rows = load_jsonl(args.output_dir / "smoke/teacher/predictions.jsonl")
    data = load_jsonl(args.output_dir / "data/smoke.jsonl")
    if len(validate_partial(data, rows)) != len(data) or not all(
            r["finished"] and r["prediction"].strip() and r["observation"]["source_frames"] == 5 for r in rows):
        raise ValueError("smoke generation incomplete/empty; inspect smoke/teacher before a long evaluation")
    write_json(args.output_dir / "smoke_validation.json", {
        "completed": True, "records": len(rows), "accuracy_tested": False,
        "predictions_sha256": file_sha256(args.output_dir / "smoke/teacher/predictions.jsonl"),
        "manifest_sha256": file_sha256(args.output_dir / "smoke/teacher/run_manifest.json")})


def audit_rows(originals, student, teacher, limit=32):
    """Balanced disagreement review queue; never claim automatic visual verification."""
    s, t = ({r["question_id"]: r for r in group} for group in (student, teacher))
    pools = {}
    for row in originals:
        key = row["question_id"]
        tc = bool(t[key]["lingo_judge_correct"] and t[key]["finished"] and t[key]["prediction"].strip())
        sc = s[key]["lingo_judge_correct"]
        outcome = ("repair" if tc else "both_wrong") if not sc else ("both_correct" if tc else "regression")
        pools.setdefault((row["source_subset"], outcome), []).append(
            dict(row, diagnostic_group=row["source_subset"] + "/" + outcome))
    rng = random.Random(44)
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    for i in range(max(map(len, pools.values()))):
        for key in sorted(pools):
            if i < len(pools[key]):
                selected.append(pools[key][i])
                if len(selected) == limit:
                    return selected
    return selected


def report(args, split):
    data = args.output_dir / "data" / (split + ".jsonl")
    originals = load_jsonl(data)
    folders = {r: args.output_dir / split / r for r in ("parent", "retained_ce", "teacher")}
    rows = {r: checked_scores(f, originals) for r, f in folders.items()}
    teacher_manifest = read(folders["teacher"] / "run_manifest.json")
    if (teacher_manifest["data_sha256"] != file_sha256(data)
            or teacher_manifest.get("references_in_prompt") is not False
            or teacher_manifest.get("protocol") != PROTOCOL
            or teacher_manifest.get("max_new_tokens") != args.max_new_tokens
            or teacher_manifest.get("max_pixels_per_frame") != args.max_pixels
            or teacher_manifest.get("task_instruction") != INSTRUCTION
            or teacher_manifest.get("model", {}).get("path") != str(args.model.resolve())):
        raise ValueError("teacher manifest differs from this experiment")
    paired = {role: paired_comparison(rows[role], rows["teacher"]) for role in ("parent", "retained_ce")}
    artifacts = {}
    for folder in folders.values():
        for name in ("run_manifest.json", "predictions.jsonl", "scored/scored_predictions.jsonl", "scored/summary.json"):
            path = folder / name
            artifacts[str(path)] = file_sha256(path)
    if teacher_manifest.get("aggregation"):
        aggregate = teacher_manifest["aggregation"]
        artifacts.update(aggregate["manifests_sha256"])
        artifacts.update(aggregate["predictions_sha256"])
        stamp_path = folders["teacher"] / "scored/input_identity.json"
        artifacts[str(stamp_path)] = file_sha256(stamp_path)
        artifacts.update(read(stamp_path)["shards_sha256"])
        if any(file_sha256(p) != sha for p, sha in artifacts.items()):
            raise ValueError("source shard artifacts changed after merging")
    result = {
        "completed": True, "protocol": PROTOCOL, "split": split, "judge_protocol": JUDGE_PROTOCOL,
        "records": len(originals), "data_sha256": file_sha256(data),
        "experiment_sha256": file_sha256(args.output_dir / "experiment.json"),
        "models": {role: summarize(values) for role, values in rows.items()},
        "paired_teacher_vs_student": paired, "groups": {},
        "complete_teacher_answers": sum(r["finished"] is True for r in rows["teacher"]),
        "observed_advantage_over_retained_ce":
            paired["retained_ce"]["usable"]["net_change_95pct_interval"][0] > 0,
        "teacher_qualified_for_distillation": False, "automatic_training": False,
        "external_teacher_training_overlap":
            read(args.output_dir / "data/manifest.json")["external_teacher_training_overlap"],
        "limitations": [
            "Internal dev comes from public LingoQA train: a driving-specialized teacher may have seen these videos.",
            "The public official500 has been repeatedly reused; it is not a fresh blind test.",
            "A positive paired interval is screening evidence, not verified unseen teacher superiority or student distillation gain.",
            "Teacher sees the same five frames with its own processor/resolution; students use their existing EVA features.",
            "Bootstrap resamples videos for fixed models; it does not measure training-seed or model-selection uncertainty.",
        ],
        "artifact_sha256": artifacts,
    }
    for source in sorted({r["source_subset"] for r in originals}):
        sub = {role: [r for r in values if r["source_subset"] == source] for role, values in rows.items()}
        result["groups"][source] = {
            "models": {role: summarize(v) for role, v in sub.items()},
            "paired_teacher_vs_retained_ce": paired_comparison(sub["retained_ce"], sub["teacher"])}
    chosen = audit_rows(originals, rows["retained_ce"], rows["teacher"])
    audit = args.output_dir / split / "review"
    write_rows(audit / "cases.jsonl", chosen)
    write_rows(audit / "audit_template.jsonl", [
        {"question_id": r["question_id"], "group": r["diagnostic_group"],
         "ego_action_supported": None, "cause_or_object_supported": None,
         "direction_supported": None, "temporal_change_supported": None,
         "reference_disputed": None, "usable_teacher_answer": None, "notes": ""}
        for r in chosen])
    ids = {r["question_id"] for r in chosen}
    for role in rows:
        write_rows(audit / (role + ".jsonl"), [r for r in rows[role] if r["question_id"] in ids])
    review_path = audit / "review.html"
    if not review_path.exists():
        child(args, split + "_review", [
            sys.executable, ROOT / "scripts/compare_teachers.py", "report",
            "--data", audit / "cases.jsonl", "--output-dir", audit, "--baseline", "retained_ce",
            "--scored", *[r + "=" + str(audit / (r + ".jsonl")) for r in rows]])
    result["fact_review"] = {"cases": len(chosen), "visually_reviewed": False,
                             "html": str(review_path), "template": str(audit / "audit_template.jsonl")}
    target = args.output_dir / (split + "_report")
    write_json(target.with_suffix(".json"), result)
    lines = ["# Qwen-Drive teacher screening — " + split, "",
             "| Model | N | Lingo-Judge | Token F1 |", "|---|---:|---:|---:|"]
    for role, metric in result["models"].items():
        lines.append("| %s | %d | %.2f%% | %.4f |" % (
            role, metric["records"], metric["lingo_judge_accuracy"] * 100, metric["token_f1"]))
    for role, value in paired.items():
        pair = value["usable"]["lingo_judge_correct"]
        ci = value["usable"]["net_change_95pct_interval"]
        lines.append("\nTeacher vs %s (unfinished answers count as incorrect): repair %d, regression %d; "
                     "net %+.2f pp; video-bootstrap 95%% CI [%+.2f, %+.2f] pp." % (
                         role, pair["fixed"], pair["regressed"], pair["net_change_mean"] * 100,
                         ci[0] * 100, ci[1] * 100))
    lines.extend(["", "## Limits", ""] + ["- " + note for note in result["limitations"]])
    lines.extend(["", "Teacher qualified for distillation: **false / not established**.",
                  "No training was started.", "", "Review: [%d cases](%s/review/review.html)." % (len(chosen), split)])
    target.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(target.with_suffix(".md")),
                      "models": result["models"], "teacher_qualified_for_distillation": False}), flush=True)
    return result


def status(args):
    out = args.output_dir
    workflow = read(out / "workflow.json") if (out / "workflow.json").exists() else {"stage": "not_started"}
    print("File progress only; verify process liveness with systemctl/journalctl.")
    print(json.dumps(workflow, ensure_ascii=False))
    devices = getattr(args, "gpus", ["0"])
    print("Requested GPU mapping: " + ",".join(devices))
    for split in ("smoke", "dev", "official"):
        folder = out / split / "teacher"
        total = line_count(out / "data" / (split + ".jsonl"))
        if len(devices) > 1:
            state = progress_snapshot(folder, total, devices)
            eta = state["estimated_remaining_seconds"]
            print("%-8s generated %4d/%-4d | Judge %4d/%-4d | generation ETA(h)=%s" % (
                split, state["generated"], total, state["scored"], total,
                "%.2f" % (eta / 3600) if eta is not None else "-"))
            for worker in state["shards"]:
                print("  shard %d -> GPU %s | %3d/%-3d answers | %3d scores | %s" % (
                    worker["index"], worker["gpu"], worker["generated"], worker["total"],
                    worker["scored"], worker["stage"]))
            continue
        progress = read(folder / "progress.json") if (folder / "progress.json").exists() else {}
        score_progress = read(folder / "scored/score_progress.json") if (folder / "scored/score_progress.json").exists() else {}
        eta = progress.get("estimated_remaining_seconds")
        print("%-8s generated %4d/%-4d | stage=%-16s | Judge %4d | ETA(h)=%s" % (
            split, line_count(folder / "predictions.jsonl"), total,
            progress.get("stage", "not_started"), score_progress.get("completed", 0),
            "%.2f" % (eta / 3600) if eta is not None else "-"))
    for split in ("dev", "official"):
        path = out / (split + "_report.json")
        valid = False
        error = None
        if path.exists():
            try:
                value = read(path)
                valid = (value.get("completed") is True and
                         value["experiment_sha256"] == file_sha256(out / "experiment.json") and
                         value["data_sha256"] == file_sha256(out / "data" / (split + ".jsonl")) and
                         all(file_sha256(p) == sha for p, sha in value["artifact_sha256"].items()))
            except (OSError, KeyError, ValueError) as exc:
                error = str(exc)
        print("%s report validated=%s%s" % (split, valid, " | " + error if error else ""))
    print("Teacher training-overlap audit: unresolved. Automatic training: disabled.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "download", "preflight", "smoke", "screen",
                                      "official", "report", "status"))
    p.add_argument("--output-dir", type=Path,
                   help="default: original single-GPU run, or an isolated N-gpu run when --gpus has N entries")
    p.add_argument("--gpus", default=os.environ.get("QWEN_DRIVE_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
                   help="physical GPU indices; each worker gets one complete model replica, e.g. 1,2,3,5")
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_qformer_all_v1")
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints/Qwen-Drive-1.0-4B")
    p.add_argument("--source", type=Path, default=ROOT / "third_party/Qwen-Drive-1.0")
    p.add_argument("--teacher-python", default=os.environ.get(
        "QWEN_DRIVE_PYTHON", str(ROOT / ".venvs/qwen-drive/bin/python")))
    p.add_argument("--dev-size", type=int, default=600)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-pixels", type=int, default=921600)
    p.add_argument("--split", choices=("dev", "official"), default="dev")
    args = p.parse_args()
    args.gpus = parse_devices(args.gpus)
    if args.output_dir is None:
        args.output_dir = RUN if len(args.gpus) == 1 else ROOT / (
            "outputs/qwen_drive_teacher_%dgpu_20261020" % len(args.gpus))
    for key in ("output_dir", "data_dir", "model", "source"):
        setattr(args, key, getattr(args, key).resolve())
    if args.action == "status":
        status(args)
        return
    with exclusive_lock(args.output_dir / ".workflow.lock"):
        try:
            event(args, "starting")
            if args.action == "download":
                child(args, "download", backend_command(args, "download"), offline=False)
            elif args.action == "preflight":
                if len(args.gpus) > 1:
                    children(args, "preflight_parallel", [
                        {"label": "preflight_shard_%03d" % i, "gpu": gpu,
                         "command": backend_command(args, "doctor")} for i, gpu in enumerate(args.gpus)])
                else:
                    child(args, "preflight", backend_command(args, "doctor"), gpu=args.gpus[0])
            elif args.action == "prepare":
                prepare(args)
            else:
                validate_prepared(args)
                if args.action in ("smoke", "screen", "official"):
                    smoke(args)
                if args.action in ("screen", "official"):
                    split = "dev" if args.action == "screen" else "official"
                    generate(args, split)
                    score(args, split)
                    report(args, split)
                elif args.action == "report":
                    report(args, args.split)
            event(args, "completed")
        except BaseException as exc:
            event(args, "failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
