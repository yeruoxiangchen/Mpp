#!/usr/bin/env python3
"""Verify real FKL/all-trajectory updates and exact checkpoint-one resume."""
import json
import math
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_success_opd_smoke import tensors
from scripts.generate_answers import write_json_atomic
from mpp_video_qa.core import file_sha256


def main():
    original=ROOT/"outputs/opd_mechanism_smoke8_20261019"
    resumed=Path(str(original)+"_resumed")
    summaries=[]; configs=[]
    for folder in (original,resumed):
        s=json.loads((folder/"summary.json").read_text()); m=s["cumulative_metrics"]
        c=json.loads((folder/"run_config.json").read_text())
        if not s["completed"] or s["optimizer_steps"]!=2 or m["examples"]!=8 or m["rollouts"]!=8:
            raise ValueError("incomplete smoke budget")
        if c["objective"]!="forward_kl" or c["opd_trajectory_scope"]!="all" or c["keep_weight"]!=0:
            raise ValueError("incorrect smoke objective")
        if m["distilled_correct"]<=0 or m["distilled_wrong"]<=0 or m["distill_sum"]<=0 or m["jsd_sum"]!=0:
            raise ValueError("both correct and incorrect gated trajectories must be exercised")
        if m["eligible"]!=m["distilled_correct"]+m["distilled_wrong"]:
            raise ValueError("cohort accounting mismatch")
        if any(s["teacher_tracked_parameter_deltas"].values()) or set(s["teacher_tracked_parameter_deltas"])!={"language","projector","qformer"}:
            raise ValueError("teacher checks missing or weights changed")
        if not all(math.isfinite(v) and v>0 for v in s["tracked_parameter_delta_this_run"].values()):
            raise ValueError("student update missing")
        summaries.append(s); configs.append(c)
    if configs[0]!=configs[1]: raise ValueError("resume identity mismatch")
    differences={}
    for filename in ("adapter/adapter_model.safetensors","projector.pt","qformer_adapter.pt"):
        def read(folder):
            path=folder/"final"/filename
            return dict(tensors(load_file(str(path)) if path.suffix==".safetensors" else torch.load(path,map_location="cpu")))
        a,b=read(original),read(resumed)
        if not a or set(a)!=set(b): raise ValueError("tensor keys differ")
        delta=max(float((a[k].float()-b[k].float()).abs().max()) for k in a)
        if delta>1e-6: raise ValueError("resume mismatch: "+filename)
        differences[filename]=delta
    result=dict(completed=True,validation_only=True,real_qa=8,optimizer_steps=2,
                objective="forward_kl",trajectory_scope="all",metrics=summaries[0]["cumulative_metrics"],
                resumed_tensor_max_abs_differences=differences,gpu_memory=summaries[0]["gpu_memory_rank0"],
                code_sha256={f:file_sha256(ROOT/f) for f in ["scripts/train_feedback_opd.py","scripts/feedback_opd_common.py","mpp_video_qa/feedback_opd.py","mpp_video_qa/opd.py"]},
                interpretation="Forward/backward/checkpoint validation only; no held-out improvement claim.")
    write_json_atomic(original/"smoke_validation.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
