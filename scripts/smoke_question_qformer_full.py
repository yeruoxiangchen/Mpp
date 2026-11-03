#!/usr/bin/env python3
"""Short real MPP updates across frozen/unfrozen language stages, never a benchmark."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.feedback_opd_common import add_runtime_arguments, configure_model, build_student, training_dataset
from scripts.train_feedback_opd import tracked_parameters
from scripts.train_sft import gpu_memory, seed_all
from scripts.generate_answers import write_json_atomic
from mpp_video_qa.sft_schedule import StagedSFT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(p)
    a = p.parse_args()
    a.qformer_text_input, a.sft_data_contract = "question", "original_full"
    a.lr, a.projector_lr, a.qformer_lr, a.qformer_text_lr = 2.5e-6, 5e-6, 1e-5, 2e-5
    a.language_freeze_steps, a.language_warmup_steps = 2, 1
    a.weight_decay, a.warmup_ratio = .01, 0
    configure_model(a, True)
    device = torch.device("cuda", 0)
    torch.set_num_threads(1)
    seed_all(a.seed)
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "progress.json", {"status": "loading_staged_mpp_smoke"})
    dataset = training_dataset(a)
    model = build_student(a, device, True)
    tracked = tracked_parameters(model)
    initial = {key: value.detach().float().cpu().clone() for key, (_, value) in tracked.items()}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    stage = StagedSFT(named, a, 4)
    optimizer, scheduler = stage.optimizer, stage.scheduler
    torch.cuda.reset_peak_memory_stats(device)
    progress = []
    for step in range(4):
        active = stage.apply_phase(step)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(4):
            item = dataset[step * 4 + micro]
            loss = model(item["features"].unsqueeze(0).to(device), [item["row"]["question"]], [item["answer"]]).loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite staged smoke loss")
            losses.append(float(loss.detach()))
            (loss / 4).backward()
        norms = stage.gradient_norms()
        torch.nn.utils.clip_grad_norm_(active, 1)
        optimizer.step()
        scheduler.step()
        deltas = {key: float((value.detach().float().cpu() - initial[key]).abs().max())
                  for key, (_, value) in tracked.items()}
        if step < 2 and (deltas["language"] != 0 or norms["language"] != 0):
            raise ValueError("frozen LoRA changed or received gradients")
        progress.append({"optimizer_step": step + 1, "phase": stage.phase, "losses": losses,
                         "gradient_norms": norms, "tracked_deltas": deltas})
        write_json_atomic(output / "progress.json", progress[-1])
    if not all(v > 0 for v in progress[-1]["tracked_deltas"].values()):
        raise ValueError("one of the joint-training groups failed to update")
    item = dataset[0]
    model.eval()
    answer = model.generate(item["features"].unsqueeze(0).to(device), [item["row"]["question"]], 32)[0]
    result = {"completed": True, "optimizer_steps": 4, "gradient_accumulation": 4, "real_qa_visits": 16,
              "single_gpu": True, "stage_boundary_for_smoke": 2, "progress": progress,
              "parameter_groups": stage.describe(), "generation": answer, "gpu_memory": gpu_memory(device),
              "note": "Implementation smoke only; formal full-data training uses a 1000-step boundary."}
    write_json_atomic(output / "summary.json", result)
    print(__import__("json").dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
