"""Parameter groups and resumable bridge-first, then joint, ordinary SFT."""
import math

import torch

from .question_qformer import is_text_parameter


PROTOCOL = "full_original_qa_two_stage_sft_v1"


def recipe(args, total_steps):
    freeze = args.language_freeze_steps
    if not 0 <= freeze < total_steps:
        raise ValueError("language-freeze-steps must be smaller than the full update budget")
    if args.qformer_text_lr <= 0 or args.language_warmup_steps < 0:
        raise ValueError("positive text LR and nonnegative language warmup are required")
    if args.language_warmup_steps >= total_steps - freeze:
        raise ValueError("language warmup must fit the joint-training stage")
    return {"protocol": PROTOCOL, "language_freeze_steps": freeze,
            "language_warmup_steps": args.language_warmup_steps,
            "qformer_text_lr": args.qformer_text_lr,
            "schedule": "bridge_global_cosine_language_delayed_cosine_v1",
            "parameter_groups": ["language", "projector", "qformer", "qformer_text"]}


def parameter_group(name):
    if "visual_bridge." in name:
        local = name.split("visual_bridge.", 1)[1]
        return "qformer_text" if is_text_parameter(local) else "qformer"
    if "llm_proj." in name:
        return "projector"
    if "llm_model." in name and "lora_" in name:
        return "language"
    raise ValueError("unclassified trainable parameter: %s" % name)


def cosine_factor(step, total, warmup):
    if step < 0:
        return 0.0
    if warmup and step < warmup:
        return float(step) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, .5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


class StagedSFT:
    """Stable optimizer groups; frozen parameters have neither gradients nor Adam state."""
    def __init__(self, named_parameters, args, total_steps):
        self.spec = recipe(args, total_steps)
        self.total_steps = total_steps
        self.named = list(named_parameters)
        grouped = {name: [] for name in self.spec["parameter_groups"]}
        seen = set()
        for name, parameter in self.named:
            if id(parameter) in seen:
                raise ValueError("parameter appears twice: %s" % name)
            seen.add(id(parameter))
            grouped[parameter_group(name)].append(parameter)
        if any(not grouped[name] for name in ("language", "projector", "qformer")):
            raise ValueError("ordinary full SFT requires language, projector and visual bridge groups")
        rates = {"language": args.lr, "projector": args.projector_lr,
                 "qformer": args.qformer_lr, "qformer_text": args.qformer_text_lr}
        self.groups = [{"params": grouped[name], "lr": rates[name], "group_name": name}
                       for name in self.spec["parameter_groups"] if grouped[name]]
        self.optimizer = torch.optim.AdamW(self.groups, weight_decay=args.weight_decay)
        warmup = int(total_steps * args.warmup_ratio)
        freeze, language_warmup = args.language_freeze_steps, args.language_warmup_steps

        def bridge_factor(step):
            return cosine_factor(step, total_steps, warmup)

        def language_factor(step):
            return cosine_factor(step - freeze, total_steps - freeze, language_warmup)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, [language_factor if g["group_name"] == "language" else bridge_factor
                             for g in self.groups])
        self.apply_phase(0)

    def apply_phase(self, completed_steps):
        if not 0 <= completed_steps <= self.total_steps:
            raise ValueError("invalid completed-step counter")
        active_language = completed_steps >= self.spec["language_freeze_steps"]
        self.phase = "joint" if active_language else "bridge_adaptation"
        for group in self.groups:
            active = group["group_name"] != "language" or active_language
            for parameter in group["params"]:
                parameter.requires_grad_(active)
                if not active:
                    parameter.grad = None
        return self.active_parameters()

    def active_parameters(self):
        return [p for group in self.groups for p in group["params"] if p.requires_grad]

    def describe(self):
        return [{"name": g["group_name"], "parameters": sum(p.numel() for p in g["params"]),
                 "peak_lr": g["initial_lr"], "active": any(p.requires_grad for p in g["params"])}
                for g in self.optimizer.param_groups]

    def gradient_norms(self):
        values = {}
        for group in self.optimizer.param_groups:
            norms = [p.grad.detach().float().norm().square() for p in group["params"] if p.grad is not None]
            values[group["group_name"]] = float(torch.stack(norms).sum().sqrt()) if norms else 0.0
        return values

    def stage_state(self, completed_steps):
        return {"protocol": PROTOCOL, "completed_steps": completed_steps,
                "phase_next_update": "joint" if completed_steps >= self.spec["language_freeze_steps"]
                                     else "bridge_adaptation",
                "group_names": [g["group_name"] for g in self.optimizer.param_groups]}

    def restore(self, saved, completed_steps):
        if saved.get("sft_stage") != self.stage_state(completed_steps):
            raise ValueError("checkpoint SFT stage or optimizer group order differs")
        self.optimizer.load_state_dict(saved["optimizer"])
        self.scheduler.load_state_dict(saved["scheduler"])
        if self.scheduler.last_epoch != completed_steps:
            raise ValueError("checkpoint scheduler and optimizer steps differ")
        expected_rates = [g["initial_lr"] * f(completed_steps)
                          for g, f in zip(self.optimizer.param_groups, self.scheduler.lr_lambdas)]
        if any(not math.isclose(g["lr"], rate, rel_tol=1e-10, abs_tol=1e-15)
               for g, rate in zip(self.optimizer.param_groups, expected_rates)):
            raise ValueError("checkpoint learning rates differ from the saved stage schedule")
        self.apply_phase(completed_steps)
