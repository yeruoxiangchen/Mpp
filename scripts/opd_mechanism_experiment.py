#!/usr/bin/env python3
"""Bounded four-arm OPD pilot: CE, error JSD, error FKL, all-trajectory JSD.

All arms use 4k original QA, one epoch/250 updates and identical CE/LRs.
Two dev checkpoints (100/250); no automatic official-test tuning or expansion.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
RUN = ROOT / "outputs/opd_mechanism_4k_v1"
EVAL = ROOT / "outputs/opd_mechanism_eval_4k_v1"
AUDIT = ROOT / "outputs/opd_fact_audit_20261019"
DATA = ROOT / "datasets/processed/lingoqa_success_opd_4k_v1"
BANK = ROOT / "outputs/success_opd_bank_4k_v1"
INITIAL = ROOT / "outputs/mpp_lingoqa_qformer_all_1epoch_v1"
PROBE = ROOT / "outputs/feedback_teacher_compare_v2/probe_report.json"
SMOKE = ROOT / "outputs/opd_mechanism_smoke8_20261019/smoke_validation.json"
ARMS = {"ce": ("jsd", "errors"), "jsd_errors": ("jsd", "errors"),
        "kl_errors": ("forward_kl", "errors"), "jsd_all": ("jsd", "all")}
LABELS = {"checkpoint-000100": 100, "final": 250}
CODE = ["scripts/train_feedback_opd.py", "scripts/feedback_opd_common.py",
        "mpp_video_qa/feedback_opd.py", "mpp_video_qa/opd.py",
        "scripts/opd_mechanism_experiment.py", "scripts/run_opd_mechanism.sh",
        "scripts/generate_answers.py", "scripts/evaluate_answers.py",
        "scripts/report_success_opd.py", "scripts/report_full_qformer_scale.py",
        "scripts/report_preference.py", "scripts/report_answer_opsd.py"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def event(stage, **extra):
    value = dict(stage=stage, controller_pid=os.getpid(), updated=time.time(), **extra)
    write(RUN / "workflow.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def command(args, log, **environment):
    env = os.environ.copy(); env.update(environment)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as output:
        output.write("\n" + json.dumps(dict(command=args, started=time.time()), ensure_ascii=False)+"\n")
        output.flush()
        child=subprocess.Popen(args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        while True:
            try:
                code=child.wait(timeout=30)
                if code: raise subprocess.CalledProcessError(code,args)
                break
            except subprocess.TimeoutExpired:
                state=read(RUN/"workflow.json")
                heartbeat=dict(heartbeat=time.time(),child_pid=child.pid,stage=state["stage"],log=str(log))
                if state.get("arm"):
                    p=RUN/state["arm"]/"progress.json"
                    progress=read(p) if p.exists() else {}
                    heartbeat.update(arm=state["arm"],status=progress.get("status","validating_inputs"),
                                     step=progress.get("optimizer_step",0),total=250,
                                     eta_hours=progress.get("estimated_remaining_hours"))
                else:
                    heartbeat.update(model=state.get("model"),split=state.get("split"))
                print(json.dumps(heartbeat,ensure_ascii=False),flush=True)


def runtime_args(data=DATA):
    return ["--data", str(data / "train.jsonl"), "--initial-dir", str(INITIAL),
            "--teacher-mode", "reference_only", "--max-new-tokens", "96", "--max-txt-len", "1536",
            "--seed", "44", "--judge-device", "cpu"]


def prepare():
    from scripts.feedback_opd_common import add_runtime_arguments, configure_model, runtime_identity
    p = argparse.ArgumentParser(); add_runtime_arguments(p)
    args = configure_model(p.parse_args(runtime_args()+["--output-dir", str(RUN)]), False)
    smoke=read(SMOKE)
    if not smoke.get("completed") or any(sha(ROOT/f)!=value for f,value in smoke["code_sha256"].items()):
        raise ValueError("complete current-code real forward/backward/resume validation first")
    audit = read(AUDIT / "report.json")
    if not audit.get("completed") or audit.get("scores_sha256") != sha(AUDIT / "scores.jsonl") or audit.get("annotations_sha256") != sha(AUDIT / "annotations.json"):
        raise ValueError("finish and verify factual audit first")
    decision = read(AUDIT / "decision.json")
    if not decision.get("bounded_pilot_authorized") or decision.get("audit_report_sha256") != sha(AUDIT / "report.json"):
        raise ValueError("review the audit and record the bounded-pilot decision first")
    config = dict(protocol="opd_mechanism_4k_250_v1", arms=ARMS, labels=LABELS,
        runtime_identity=runtime_identity(args),
        fixed=dict(records=4000, epochs=1, total_steps=250, global_batch_size=16,
                   ce_weight=1., opd_weight=1., keep_weight=0., lr=2.5e-6,
                   projector_lr=2.5e-6, qformer_lr=5e-6, warmup_ratio=.03, weight_decay=.01),
        source_hashes={str(p):sha(p) for p in [DATA/"train.jsonl", DATA/"dev_action.jsonl", DATA/"dev_scenery.jsonl",
                                              BANK/"bank_manifest.json", PROBE, SMOKE, AUDIT/"report.json", AUDIT/"decision.json"]},
        code_sha256={f:sha(ROOT/f) for f in CODE})
    # JSON normalizes arm tuples; compare the serialized representation.
    config = json.loads(json.dumps(config))
    path = RUN/"experiment.json"
    if path.exists() and read(path) != config:
        raise ValueError("experiment identity changed; use an isolated experiment")
    write(path, config)
    return config


def verify_contract():
    config = read(RUN/"experiment.json")
    for path, value in config["source_hashes"].items():
        if sha(path) != value: raise ValueError("experiment input changed: "+path)
    for path, value in config["code_sha256"].items():
        if sha(ROOT/path) != value: raise ValueError("experiment code changed: "+path)
    return config


def validate_arm(arm, complete=False):
    exp = read(RUN/"experiment.json")
    c = read(RUN/arm/"run_config.json")
    objective, scope = ARMS[arm]
    expected = dict(exp["runtime_identity"], **exp["fixed"])
    expected.update(training_mode="sft_control" if arm == "ce" else "feedback_opd",
                    objective=objective, checkpoint_steps=[100], feature_stage="eva")
    for k, v in expected.items():
        if c.get(k) != v: raise ValueError("arm config differs %s/%s" % (arm,k))
    if c.get("opd_trajectory_scope", "errors") != scope or c["world_size"] * c["grad_accumulation"] != 16:
        raise ValueError("trajectory scope/global batch mismatch")
    if complete:
        s = read(RUN/arm/"summary.json"); m = s["cumulative_metrics"]
        if not s["completed"] or s["optimizer_steps"] != 250 or m["examples"] != 4000:
            raise ValueError("incomplete training budget")
        if m["rollouts"] != (0 if arm == "ce" else 4000) or m["keep_examples"] != 0:
            raise ValueError("rollout/retention budget differs")
        frozen = s["teacher_tracked_parameter_deltas"]
        if arm != "ce" and set(frozen) != {"language", "projector", "qformer"}:
            raise ValueError("missing frozen-teacher checks")
        if any(frozen.values()): raise ValueError("teacher changed")
        updates = s.get("tracked_parameter_delta_this_run", {})
        if not updates or any(not math.isfinite(v) or v <= 0 for v in updates.values()):
            raise ValueError("missing/nonfinite student parameter changes")
        if not all(math.isfinite(m[k]) for k in ("ce_sum", "distill_sum")): raise ValueError("nonfinite loss")
        if arm != "ce" and m["distilled_correct"]+m["distilled_wrong"] != m["eligible"]:
            raise ValueError("distillation cohort counters differ")
        if scope == "errors" and m["distilled_correct"] != 0: raise ValueError("error-only gate violated")
        if scope == "all" and m["distilled_correct"] <= 0: raise ValueError("correct-trajectory branch never used")
        for label, step in LABELS.items():
            bundle = RUN/arm/label
            md = read(bundle/"metadata.json")
            if md["identity"] != c or md["optimizer_steps"] != step: raise ValueError("checkpoint metadata differs")
            for name in ("adapter/adapter_model.safetensors", "projector.pt", "qformer_adapter.pt", "training_state.pt"):
                if not (bundle/name).is_file() or (bundle/name).stat().st_size == 0: raise ValueError("missing checkpoint "+str(bundle/name))
    return c


def select_checkpoints(metrics):
    # Dict insertion order is the explicit early-checkpoint tie breaker.
    return {a:max([a+"__"+l for l in LABELS], key=lambda k:metrics[k]["lingo_judge_accuracy"])
            for a in ARMS}


def train(arm):
    verify_contract()
    folder = RUN/arm
    if (folder/"summary.json").exists():
        validate_arm(arm, True); return
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 2 or not all(devices) or len(set(devices)) != 2:
        raise ValueError("set two distinct available GPUs in CUDA_VISIBLE_DEVICES")
    resume = []
    if (folder/"run_config.json").exists():
        validate_arm(arm)
        checkpoint = folder/"checkpoint-000100"
        if not (checkpoint/"training_state.pt").exists():
            raise ValueError("interrupted before checkpoint; preserve directory and choose an isolated retry")
        resume = ["--resume-from", str(checkpoint)]
    objective, scope = ARMS[arm]
    args = [PYTHON, "-u", "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", "29749",
            "scripts/train_feedback_opd.py"] + runtime_args() + [
        "--output-dir", str(folder), "--training-mode", "sft_control" if arm=="ce" else "feedback_opd",
        "--success-bank", str(BANK), "--keep-weight", "0", "--probe-report", str(PROBE),
        "--epochs", "1", "--global-batch-size", "16", "--grad-accumulation", "8",
        "--save-every", "0", "--checkpoint-steps", "100", "--lr", "2.5e-6",
        "--projector-lr", "2.5e-6", "--qformer-lr", "5e-6", "--ce-weight", "1", "--opd-weight", "1",
        "--opd-objective", objective, "--opd-trajectory-scope", scope,
        "--answer-only-logits", "--gradient-bucket-mb", "25", "--profile-every", "20",
        "--runtime-log-flush-every", "4", "--teacher-gate-cache-dir", str(RUN/"teacher_gate_cache")]+resume
    event("training", arm=arm)
    command(args, RUN/"logs"/(arm+".log"))
    validate_arm(arm, True)


def weight_args(arm, label):
    if arm=="parent":
        return ["--adapter", str(INITIAL/"final_adapter"), "--projector", str(INITIAL/"final_projector.pt"),
                "--qformer-adapter", str(INITIAL/"final_qformer_adapter.pt")]
    bundle=RUN/arm/label
    return ["--adapter", str(bundle/"adapter"), "--projector", str(bundle/"projector.pt"),
            "--qformer-adapter", str(bundle/"qformer_adapter.pt")]


def identity(arm, label):
    from scripts.generate_answers import directory_fingerprint
    a = weight_args(arm,label)
    return dict(adapter_identity=directory_fingerprint(a[1]), projector_identity=sha(a[3]), qformer_adapter_sha256=sha(a[5]))


def evaluated(arm, label, split):
    from scripts.report_success_opd import load_evaluation
    key = "parent" if arm=="parent" else arm+"__"+label
    return load_evaluation(EVAL/(key+"_"+split), DATA/("dev_"+split+".jsonl"),
                           identity(arm,label), read(RUN/"experiment.json")["runtime_identity"])


def evaluate(arm, label, split):
    verify_contract()
    key = "parent" if arm=="parent" else arm+"__"+label
    folder=EVAL/(key+"_"+split)
    if (folder/"scored/summary.json").exists():
        evaluated(arm,label,split); return
    event("evaluating", model=key, split=split)
    gpu=os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    args=[PYTHON,"-u","scripts/generate_answers.py","--data",str(DATA/("dev_"+split+".jsonl")),
          "--output-dir",str(folder),"--cache-dir",os.environ.get("MPP_EVA_CACHE_DIR", str(ROOT / "datasets/cache/lingoqa_5f_eva_v1")),
          "--upstream-root",str(Path(os.environ.get("MPP_UPSTREAM_ROOT", str(ROOT / "third_party/MPP-LLaVA")))),
          "--visual-checkpoint-root",str(Path(os.environ.get("MPP_PERSISTENT_CHECKPOINT_ROOT", str(ROOT / "checkpoints")))),
          "--bridge-init",str(ROOT/"checkpoints/visual_bridge_init.pt"),
          "--checkpoint-root",os.environ.get("MPP_CHECKPOINT_ROOT", str(ROOT / "checkpoints")),"--feature-stage","eva",
          "--precision-protocol","fp32_trainables_v1","--qformer-mode","auto",
          "--task-instruction-version","legacy_v1","--max-new-tokens","96","--max-txt-len","1536",
          "--record-base-identity","--batch-size","1"]+weight_args(arm,label)
    command(args,RUN/"logs"/(key+"_"+split+".log"),CUDA_VISIBLE_DEVICES=gpu)
    event("scoring",model=key,split=split)
    command([PYTHON,"-u","scripts/evaluate_answers.py","--predictions",str(folder/"predictions.jsonl"),
             "--output-dir",str(folder/"scored"),"--lingo-judge"],RUN/"logs"/(key+"_"+split+".log"),CUDA_VISIBLE_DEVICES=gpu)
    evaluated(arm,label,split)


def report():
    from scripts.report_preference import summarize, cluster_interval
    from scripts.report_answer_opsd import paired_metrics
    verify_contract()
    configs={a:validate_arm(a,True) for a in ARMS}
    versions={(c["world_size"],c["grad_accumulation"]) for c in configs.values()}
    if len(versions)!=1: raise ValueError("execution layouts differ")
    rows={}; sources={}; specs={}
    for arm,label in [("parent","final")]+[(a,l) for a in ARMS for l in LABELS]:
        key="parent" if arm=="parent" else arm+"__"+label
        specs[key]=identity(arm,label); rows[key]=[]
        for split in ("action","scenery"):
            part,_=evaluated(arm,label,split); rows[key].extend(part)
            d=EVAL/(key+"_"+split)
            for f in ("run_manifest.json","predictions.jsonl","scored/summary.json","scored/scored_predictions.jsonl"):
                sources[str(d/f)]=sha(d/f)
    metrics={k:summarize(v) for k,v in rows.items()}
    selected=select_checkpoints(metrics)
    def pair(a,b): return dict(paired_metrics(rows[a],rows[b]), **cluster_interval(rows[a],rows[b]))
    fixed={label:{a+"_vs_"+b:pair(a+"__"+label,b+"__"+label) for a,b in
                 [("ce","jsd_errors"),("ce","kl_errors"),("ce","jsd_all"),("jsd_errors","kl_errors"),("jsd_errors","jsd_all")]} for label in LABELS}
    chosen={"parent_vs_"+a:pair("parent",selected[a]) for a in ARMS}
    chosen.update({"ce_vs_"+a:pair(selected["ce"],selected[a]) for a in ARMS if a!="ce"})
    result=dict(protocol="opd_mechanism_4k_250_v1",completed=True,metrics=metrics,selected=selected,
                fixed_step=fixed,selected_pairs=chosen,model_identities=specs,source_hashes=sources,
                training_configs=configs,experiment_sha256=sha(RUN/"experiment.json"),
                caution="One seed, 4k one epoch. Dev selection intervals exploratory. Matched QA/updates, not compute. No official test selection.")
    write(EVAL/"report.json",result)
    lines=["# OPD mechanism pilot", "", result["caution"], "", "| Model | Dev Judge | Correct / 4320 |", "|---|---:|---:|"]
    for k,m in metrics.items():
        lines.append("| %s | %.2f%% | %d |"%(k,100*m["lingo_judge_accuracy"],sum(bool(r["lingo_judge_correct"]) for r in rows[k])))
    lines.extend(["", "Dev-selected checkpoints: `"+json.dumps(selected)+"`", "", "Detailed paired results and source hashes: report.json."])
    (EVAL/"report.md").write_text("\n".join(lines)+"\n")
    event("completed",report=str(EVAL/"report.json"))


def status():
    if (RUN/"workflow.json").exists(): print(json.dumps(read(RUN/"workflow.json"),ensure_ascii=False))
    print("File progress; check systemd/journal for process liveness.")
    print("arm          state              steps      ETA(h)  distilled correct/wrong")
    for arm in ARMS:
        d=RUN/arm; p=read(d/"progress.json") if (d/"progress.json").exists() else {}
        s=read(d/"summary.json") if (d/"summary.json").exists() else {}
        m=s.get("cumulative_metrics",p.get("cumulative_metrics",{}))
        state="completed_artifacts" if s.get("completed") else p.get("status","not_started")
        if list(d.glob("failure_rank_*.json")): state="error_inspect_log"
        print("%-12s %-18s %3s/250    %-7s %s/%s"%(arm,state,s.get("optimizer_steps",p.get("optimizer_step",0)),
              round(p["estimated_remaining_hours"],2) if "estimated_remaining_hours" in p and not s else "-",
              int(m.get("distilled_correct",0)),int(m.get("distilled_wrong",0))))
    def count(path):
        if not path.exists(): return 0
        with path.open("rb") as f: return sum(1 for line in f if line.endswith(b"\n"))
    answers=sum(count(f) for f in EVAL.glob("*/predictions.jsonl"))
    scored=sum(count(f) for f in EVAL.glob("*/scored/scored_predictions.jsonl"))
    print("dev answers: %d/38880; scored rows: %d/38880; summaries present (unverified): %d/18"%(answers,scored,len(list(EVAL.glob("*/scored/summary.json")))))
    judging=sum(read(f).get("completed",0) for f in EVAL.glob("*/scored/score_progress.json"))
    print("Judge progress (includes in-memory scoring): %d/38880"%judging)
    print("validated final report:", (EVAL/"report.json").exists())
    if (AUDIT/"progress.json").exists(): print("fact audit:",json.dumps(read(AUDIT/"progress.json")))


def pipeline():
    prepare()
    for arm in ARMS: train(arm)
    for arm,label in [("parent","final")]+[(a,l) for a in ARMS for l in LABELS]:
        for split in ("action","scenery"): evaluate(arm,label,split)
    report()


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("prepare","pipeline","status","report","eval_dev"),default="status",nargs="?")
    a=p.parse_args()
    if a.action=="status": status()
    else:
        RUN.mkdir(parents=True,exist_ok=True)
        with (RUN/".runner.lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                if a.action=="prepare": prepare()
                elif a.action=="pipeline": pipeline()
                elif a.action=="report": report()
                else:
                    verify_contract()
                    for arm,label in [("parent","final")]+[(x,l) for x in ARMS for l in LABELS]:
                        for split in ("action","scenery"): evaluate(arm,label,split)
                    report()
            except Exception:
                event("failed",error=traceback.format_exc())
                raise
