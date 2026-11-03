"""Pure data contracts for the Qwen-Drive offline answer-transfer pilot.

This is text-target CE, not token-distribution OPD. Original references are
immutable. Visual spot checks are evidence about sampled cases, not gold
labels for the entire candidate pool.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import random

from .distillation import validate_predictions

PROTOCOL = "qwen_drive_offline_transfer_4k_v1"
OUTCOMES = ("repair", "regression", "both_correct", "both_wrong")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def checked_pair(originals, student, teacher):
    s, t = (validate_predictions(originals, x) for x in (student, teacher))
    for values in (s, t):
        for row in values.values():
            logit = row.get("lingo_judge_logit")
            if (type(row.get("lingo_judge_correct")) is not bool or
                    not isinstance(logit, (int, float)) or not math.isfinite(logit) or
                    row["lingo_judge_correct"] != (logit > 0)):
                raise ValueError("Judge flag/logit is invalid")
    return s, t


def outcome(student, teacher):
    return {(False, True): "repair", (True, False): "regression",
            (True, True): "both_correct", (False, False): "both_wrong"}[
                student["lingo_judge_correct"], teacher["lingo_judge_correct"]]


def teacher_target_reasons(student, teacher):
    reasons = []
    if outcome(student, teacher) != "repair":
        reasons.append("not_judge_repair")
    if (teacher.get("finished") is not True or teacher.get("eos_seen") is not True
            or teacher.get("unclosed_thinking") is not False):
        reasons.append("unfinished_teacher")
    text = teacher["prediction"].strip()
    if not text or len(text.split()) > 96 or "<think>" in text or "</think>" in text:
        reasons.append("empty_long_or_thinking_text")
    # A positive score after truncation is not evidence about the full answer.
    for role, row in (("student", student), ("teacher", teacher)):
        lengths = row.get("judge_input_tokens")
        if (not lengths or len(lengths) != len(row["references"]) or
                any(type(n) is not int or n <= 0 or n > 128 for n in lengths)):
            reasons.append(role + "_judge_truncated_or_unknown")
    return reasons


def audit_cases(originals, student, teacher, per_stratum=4, seed=44):
    s, t = checked_pair(originals, student, teacher)
    groups = defaultdict(list)
    for row in originals:
        key = row["question_id"]
        groups[(row["source_subset"], outcome(s[key], t[key]))].append(row)
    rng = random.Random(seed)
    result = []
    for group in sorted(groups):
        pool = list(groups[group])
        rng.shuffle(pool)
        # Within repair strata, cover usable targets as well as failed filters.
        if group[1] == "repair":
            pool.sort(key=lambda r: bool(teacher_target_reasons(s[r["question_id"]], t[r["question_id"]])))
        for row in pool[:per_stratum]:
            key = row["question_id"]
            result.append(dict(row, stratum=list(group), student_prediction=s[key]["prediction"],
                               teacher_prediction=t[key]["prediction"],
                               teacher_target_rejection_reasons=teacher_target_reasons(s[key], t[key]),
                               student_judge_correct=s[key]["lingo_judge_correct"],
                               teacher_judge_correct=t[key]["lingo_judge_correct"]))
    return result


def validate_audit(cases, decisions):
    expected = {r["question_id"]: r for r in cases}
    actual = {r["question_id"]: r for r in decisions}
    if not expected or set(actual) != set(expected) or len(actual) != len(decisions):
        raise ValueError("visual audit must cover exactly the fixed sampled cases")
    for key, decision in actual.items():
        if decision.get("case_sha256") != digest(expected[key]):
            raise ValueError("visual audit belongs to different images/answers")
        if (decision.get("status") != "reviewed" or decision.get("frames_viewed") != [0, 1, 2, 3, 4]
                or not decision.get("reviewer") or not decision.get("reviewed_utc")
                or not decision.get("reason", "").strip()):
            raise ValueError("visual audit pending: inspect all five frames and record evidence")
        for field in ("reference_supported", "teacher_supported", "student_supported"):
            if field not in decision or (decision[field] is not None and type(decision[field]) is not bool):
                raise ValueError("audit verdict must be true/false/null (uncertain)")
    return actual


def build_training_rows(originals, student, teacher, cases, decisions):
    s, t = checked_pair(originals, student, teacher)
    audit = validate_audit(cases, decisions)
    rows, excluded, reasons = [], [], Counter()
    for original in originals:
        key = original["question_id"]
        decision = audit.get(key)
        if decision and decision["reference_supported"] is not True:
            excluded.append(dict(original, exclusion="reference_uncertain_or_contradicted", audit=decision))
            continue
        rejected = teacher_target_reasons(s[key], t[key])
        if decision and decision["teacher_supported"] is not True:
            rejected.append("visual_audit_teacher_unsupported_or_uncertain")
        reasons.update(rejected)
        rows.append(dict(original, distillation_source="train",
                         target_source="reference_replay" if rejected else "teacher",
                         teacher_answer=t[key]["prediction"].strip(),
                         teacher_filter_reasons=rejected,
                         visual_audit_scope="sampled_case" if decision else "not_visually_reviewed"))
    counts = Counter(r["target_source"] for r in rows)
    if not counts["teacher"] or not counts["reference_replay"]:
        raise ValueError("pilot needs both accepted teacher targets and original-answer replay")
    return rows, excluded, {"candidates": len(originals), "effective_records": len(rows),
                            "teacher_targets": counts["teacher"],
                            "reference_replay": counts["reference_replay"],
                            "teacher_fraction": counts["teacher"] / len(rows),
                            "source_counts": dict(Counter(r["source_subset"] for r in rows)),
                            "teacher_source_counts": dict(Counter(r["source_subset"] for r in rows
                                                                   if r["target_source"] == "teacher")),
                            "excluded_from_both_arms": len(excluded),
                            "filter_reason_counts": dict(reasons), "visually_reviewed": len(cases),
                            "supervision": "Judge-screened text with visual spot checks; not gold or OPD"}


def make_plan(rows, data_sha256, steps=500, global_batch=16, seed=44):
    if not rows or steps <= 0 or global_batch <= 0:
        raise ValueError("nonempty data and a positive update budget are required")
    rng, indices = random.Random(seed), []
    while len(indices) < steps * global_batch:
        cycle = list(range(len(rows)))
        rng.shuffle(cycle)
        indices.extend(cycle)
    indices = indices[:steps * global_batch]
    counts = Counter(indices)
    return {"protocol": PROTOCOL, "data_sha256": data_sha256, "seed": seed,
            "optimizer_steps": steps, "global_batch": global_batch,
            "effective_records": len(rows), "presentations": len(indices),
            "sampling": "seeded shuffled cycles, truncated to exact update budget; no padding",
            "indices": indices, "question_ids": [rows[i]["question_id"] for i in indices],
            "teacher_presentations": sum(rows[i]["target_source"] == "teacher" for i in indices),
            "reference_presentations": sum(rows[i]["target_source"] != "teacher" for i in indices),
            "min_repeats": min(counts.get(i, 0) for i in range(len(rows))),
            "max_repeats": max(counts.values())}


def validate_plan(plan, rows, data_sha256, world, batch_size, accumulation, max_steps, epochs):
    if world < 1 or batch_size != 1 or epochs != 1 or max_steps < 1:
        raise ValueError("fixed pilot plan requires batch1, one scheduled pass, positive steps/world")
    ids = plan.get("indices", [])
    if (plan.get("protocol") != PROTOCOL or plan.get("data_sha256") != data_sha256 or
            plan.get("effective_records") != len(rows) or plan.get("optimizer_steps") != max_steps or
            plan.get("global_batch") != world * accumulation or
            len(ids) != max_steps * world * accumulation or plan.get("presentations") != len(ids) or
            any(type(i) is not int or i < 0 or i >= len(rows) for i in ids) or
            plan.get("question_ids") != [rows[i]["question_id"] for i in ids]):
        raise ValueError("sample plan differs from dataset or exact global update budget")


class PlannedSampler:
    """Global presentation order is shared across arms, independent of RNG state."""
    def __init__(self, indices, world, rank):
        if not 0 <= rank < world or len(indices) % world:
            raise ValueError("invalid plan shard")
        self.indices = indices[rank::world]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        if epoch != 0:
            raise ValueError("plan already contains every scheduled cycle")


def validate_planned_resume(previous, current):
    keys = ("sample_plan_sha256", "disable_dropout", "data_sha256", "world_size", "total_steps",
            "target_mode", "seed", "batch_size", "grad_accumulation", "lr", "projector_lr",
            "qformer_lr", "warmup_ratio", "weight_decay", "max_txt_len", "dtype",
            "task_instruction_version", "gradient_checkpointing", "qformer_gradient_checkpointing")
    changed = [k for k in keys if previous.get(k) != current.get(k)]
    if changed:
        raise ValueError("fixed-plan resume configuration changed: %s" % changed)
