#!/usr/bin/env python3
"""Check real OPD+keep updates and reproducibility after reloading checkpoint one."""
import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_answers import write_json_atomic


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, prefix + "/" + str(key))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    args = parser.parse_args()
    original, resumed = args.train_dir, Path(str(args.train_dir) + "_resumed")
    bank = json.loads((args.bank_dir / "bank_manifest.json").read_text())
    if not bank["completed"] or bank["eligible"] <= 0:
        raise ValueError("smoke needs at least one real completed Judge-positive parent answer")
    summaries = [json.loads((folder / "summary.json").read_text()) for folder in (original, resumed)]
    for summary in summaries:
        metrics = summary["cumulative_metrics"]
        if (not summary["completed"] or summary["optimizer_steps"] != 2 or
                metrics["examples"] != 8 or metrics["rollouts"] != 8 or
                metrics["keep_examples"] != bank["eligible"] or metrics["keep_tokens"] <= 0 or
                metrics["eligible"] <= 0 or metrics["distilled_tokens"] <= 0 or
                any(summary["teacher_tracked_parameter_deltas"].values()) or
                not all(v > 0 for v in summary["tracked_parameter_delta_this_run"].values())):
            raise ValueError("real smoke did not exercise both objectives with an unchanged teacher")
    configs = [json.loads((folder / "run_config.json").read_text()) for folder in (original, resumed)]
    if configs[0] != configs[1]:
        raise ValueError("resumed training identity differs")
    differences = {}
    for filename in ("adapter/adapter_model.safetensors", "projector.pt", "qformer_adapter.pt"):
        def read(folder):
            path = folder / "final" / filename
            return dict(tensors(load_file(str(path)) if path.suffix == ".safetensors" else
                                torch.load(path, map_location="cpu")))
        left, right = read(original), read(resumed)
        if not left or left.keys() != right.keys():
            raise ValueError("resumed model tensor keys differ")
        difference = max(float((left[key].float() - right[key].float()).abs().max()) for key in left)
        differences[filename] = difference
        if difference > 1e-6:
            raise ValueError("resumed final tensor mismatch: %s (%g)" % (filename, difference))
    result = {"completed": True, "validation_only": True, "real_qa": 8, "optimizer_steps": 2,
              "success_bank_eligible": bank["eligible"], "metrics": summaries[0]["cumulative_metrics"],
              "teacher_tracked_deltas": summaries[0]["teacher_tracked_parameter_deltas"],
              "resumed_tensor_max_abs_differences": differences,
              "gpu_memory": summaries[0]["gpu_memory_rank0"],
              "interpretation": "Mechanical forward/backward/resume validation; no held-out improvement measured."}
    write_json_atomic(original / "smoke_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
