import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mpp_video_qa.core import dump_jsonl, file_sha256
from mpp_video_qa.data import training_answer
from mpp_video_qa.distillation import assert_disjoint
from mpp_video_qa.transfer_pilot import (
    audit_cases, build_training_rows, digest, make_plan, PlannedSampler,
    teacher_target_reasons, validate_audit, validate_plan, validate_planned_resume,
)
from scripts.train_sft import OffsetSampler, retain_initialization_provenance
from scripts.qwen_drive_distill import (
    training_command, report, ensure_smoke, bundle, weight_identity, validate_evaluation_identity,
)
from mpp_video_qa.driving_teacher import write_json
from scripts.report_answer_opsd import JUDGE_PROTOCOL
from scripts.report_preference import summarize


def fixtures():
    originals, student, teacher = [], [], []
    for i in range(16):
        source = "action" if i < 8 else "scenery"
        original = dict(question_id="q%d" % i, segment_id="v%d" % i, source_subset=source,
                        question="What is ahead?", answers=["A car."],
                        frames=["/v%d/%d.jpg" % (i, f) for f in range(5)])
        originals.append(original)
        for values, correct, text in ((student, i % 4 in (1, 2), "A car."),
                                      (teacher, i % 4 in (0, 2), "There is a car ahead.")):
            values.append(dict(question_id=original["question_id"], segment_id=original["segment_id"],
                               question=original["question"], source_subset=source, references=original["answers"],
                               prediction=text, lingo_judge_correct=correct, lingo_judge_logit=1.0 if correct else -1.0,
                               finished=True, eos_seen=True, unclosed_thinking=False, judge_input_tokens=[35]))
    return originals, student, teacher


def decisions(cases):
    return [dict(question_id=r["question_id"], case_sha256=digest(r), status="reviewed",
                 frames_viewed=[0, 1, 2, 3, 4], reference_supported=True, teacher_supported=True,
                 student_supported=None, reviewer="synthetic test only", reviewed_utc="test fixture",
                 reason="Synthetic fixture: no real visual assessment.") for r in cases]


class SelectionTests(unittest.TestCase):
    def test_paired_targets_and_reference_preservation(self):
        original, s, t = fixtures()
        cases = audit_cases(original, s, t)
        rows, excluded, summary = build_training_rows(original, s, t, cases, decisions(cases))
        self.assertFalse(excluded)
        self.assertEqual(summary["teacher_targets"], 4)
        self.assertEqual(summary["reference_replay"], 12)
        for source, row in zip(original, rows):
            self.assertTrue(all(row[k] == value for k, value in source.items()))
            self.assertEqual(training_answer(row, "reference"), source["answers"][0])
            self.assertEqual(training_answer(row, "distillation"),
                             row["teacher_answer"] if row["target_source"] == "teacher" else source["answers"][0])

    def test_truncation_unfinished_and_long_teacher_are_not_targets(self):
        _, s, t = fixtures()
        self.assertEqual(teacher_target_reasons(s[0], t[0]), [])
        for change in ({"finished": False}, {"eos_seen": False}, {"judge_input_tokens": [129]},
                       {"prediction": "word " * 97}, {"unclosed_thinking": True}, {"prediction": ""}):
            self.assertTrue(teacher_target_reasons(s[0], dict(t[0], **change)))
        self.assertTrue(teacher_target_reasons(dict(s[0], judge_input_tokens=[]), t[0]))

    def test_audit_pending_incomplete_frames_and_stale_images_rejected(self):
        cases = audit_cases(*fixtures())
        valid = decisions(cases)
        validate_audit(cases, valid)
        for change in ({"status": "pending"}, {"frames_viewed": [0, 4]}, {"reason": ""},
                       {"case_sha256": "old images"}):
            invalid = copy.deepcopy(valid)
            invalid[0].update(change)
            with self.assertRaises(ValueError):
                validate_audit(cases, invalid)
        with self.assertRaises(ValueError):
            validate_audit(cases, valid[:-1])

    def test_reference_conflicts_excluded_from_both_arms_uncertain_teacher_replays(self):
        original, s, t = fixtures()
        cases = audit_cases(original, s, t)
        reviewed = decisions(cases)
        by_id = {r["question_id"]: r for r in reviewed}
        by_id["q0"]["reference_supported"] = False
        by_id["q1"]["reference_supported"] = None
        by_id["q4"]["teacher_supported"] = None
        rows, excluded, _ = build_training_rows(original, s, t, cases, reviewed)
        self.assertEqual({r["question_id"] for r in excluded}, {"q0", "q1"})
        self.assertEqual(next(r for r in rows if r["question_id"] == "q4")["target_source"], "reference_replay")

    def test_prediction_identity_and_logit_inconsistency_rejected(self):
        for change in ({"references": ["Corrupt ref"]}, {"lingo_judge_logit": float("nan")},
                       {"lingo_judge_correct": False}):
            original, s, t = fixtures()
            t[0].update(change)
            with self.assertRaises(ValueError):
                audit_cases(original, s, t)

    def test_all_available_source_outcome_strata_sampled_deterministically(self):
        a = audit_cases(*fixtures(), per_stratum=1)
        self.assertEqual(len(a), 8)
        self.assertEqual(a, audit_cases(*fixtures(), per_stratum=1))
        self.assertEqual(len({tuple(r["stratum"]) for r in a}), 8)

    def test_heldout_leakage_by_video_and_frame_rejected(self):
        original, _, _ = fixtures()
        for field in ("question_id", "segment_id", "frames"):
            held = dict(original[-1])
            held[field] = original[0][field]
            with self.assertRaisesRegex(ValueError, "overlap"):
                assert_disjoint(original[:1], [held], [])


class PlanTests(unittest.TestCase):
    def setUp(self):
        original, s, t = fixtures()
        cases = audit_cases(original, s, t)
        self.rows = build_training_rows(original, s, t, cases, decisions(cases))[0][:13]
        self.plan = make_plan(self.rows, "data-hash", 500, 16)

    def test_exact_8000_presentations_no_tail_padding_after_filtering(self):
        self.assertEqual(len(self.plan["indices"]), 8000)
        self.assertLessEqual(self.plan["max_repeats"] - self.plan["min_repeats"], 1)
        validate_plan(self.plan, self.rows, "data-hash", 4, 1, 4, 500, 1)
        for world in (1, 2, 4):
            shards = [list(PlannedSampler(self.plan["indices"], world, r)) for r in range(world)]
            interleaved = [item for step in zip(*shards) for item in step]
            self.assertEqual(interleaved, self.plan["indices"])

    def test_checkpoint_resume_keeps_exact_next_global_batch(self):
        for world in (1, 2, 4):
            accumulation = 16 // world
            for checkpoint in (1, 49, 250, 499, 500):
                for rank in range(world):
                    sampler = PlannedSampler(self.plan["indices"], world, rank)
                    resumed = list(OffsetSampler(sampler, checkpoint * accumulation))
                    self.assertEqual(resumed, self.plan["indices"][rank::world][checkpoint * accumulation:])

    def test_dataset_plan_world_or_schedule_changes_rejected(self):
        altered = copy.deepcopy(self.plan)
        altered["question_ids"][0] = "heldout-question"
        for plan, sha, world, accum, steps in (
            (altered, "data-hash", 4, 4, 500), (self.plan, "newdata", 4, 4, 500),
            (self.plan, "data-hash", 3, 5, 500), (self.plan, "data-hash", 4, 4, 501)):
            with self.assertRaises(ValueError):
                validate_plan(plan, self.rows, sha, world, 1, accum, steps, 1)

    def test_resume_rejects_learning_rate_seed_or_order_changes(self):
        previous = dict(lr=2.5e-6, seed=44, sample_plan_sha256="plan", total_steps=500)
        validate_planned_resume(previous, dict(previous))
        for key, value in (("lr", .1), ("seed", 45), ("sample_plan_sha256", "other"), ("total_steps", 501)):
            with self.assertRaises(ValueError):
                validate_planned_resume(previous, dict(previous, **{key: value}))

    def test_resume_preserves_original_three_component_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            original = {k: k + "-initial" for k in (
                "initial_adapter_sha256", "initial_projector_sha256", "initial_qformer_sha256")}
            (folder / "run_config.json").write_text(json.dumps(original))
            current = {k: "checkpoint" for k in original}
            retain_initialization_provenance(current, folder / "checkpoint-000050")
            self.assertTrue(all(current[k] == v for k, v in original.items()))

    def test_two_arm_launch_has_identical_budget_and_initial_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), initial_dir=Path(directory)/"retained",
                                   gpus=["1", "2", "3", "5"], global_batch=16, steps=500, profile="formal",
                                   upstream="upstream", checkpoints="base", visual_checkpoints="visual",
                                   bridge="bridge.pt", cache_dir="cache")
            commands = [training_command(args, arm) for arm in ("ce", "distill")]
            for flag in ("--init-adapter", "--projector", "--qformer-adapter", "--data", "--sample-plan",
                         "--max-steps", "--grad-accumulation", "--qformer-mode", "--lr", "--qformer-lr",
                         "--model-load-lock"):
                self.assertEqual(commands[0][commands[0].index(flag)+1], commands[1][commands[1].index(flag)+1])
            self.assertIn("--disable-dropout", commands[0])
            resumed = training_command(args, "ce", Path(directory)/"checkpoint-000050")
            self.assertNotIn("--init-adapter", resumed)
            self.assertNotIn("--qformer-adapter", resumed)
            self.assertIn("--resume-from", resumed)


class ReportTests(unittest.TestCase):
    def test_evaluation_rejects_changed_final_weights_or_mixed_shard_text(self):
        original, predictions, _ = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=root, gpus=["1", "2"])
            weights = bundle(args, "ce")
            weights["adapter"].mkdir(parents=True)
            (weights["adapter"] / "weights.bin").write_bytes(b"adapter fixture")
            weights["projector"].write_bytes(b"projection fixture")
            weights["qformer"].write_bytes(b"bridge fixture")
            identity = weight_identity(weights)
            dump_jsonl(root / "data/dev.jsonl", original)
            dump_jsonl(root / "dev/ce/predictions.jsonl", predictions)
            write_json(root / "dev/ce/run_manifest.json", dict(
                model_identity=identity, data_sha256=file_sha256(root / "data/dev.jsonl")))
            for i in range(2):
                part_data = root / "data/shards" / ("dev_%03d.jsonl" % i)
                part = root / "dev/ce/shards" / ("shard-%03d" % i)
                dump_jsonl(part_data, original[i::2])
                dump_jsonl(part / "predictions.jsonl", predictions[i::2])
                write_json(part / "run_manifest.json", dict(identity, data_sha256=file_sha256(part_data),
                    max_new_tokens=96, max_txt_len=1536, task_instruction_version="legacy_v1",
                    feature_stage="eva", qformer_mode="full", precision_protocol="fp32_trainables_v1"))
                write_json(part / "generation_summary.json", dict(completed=8, total=8))
                write_json(part / "scored/input_identity.json", dict(
                    predictions_sha256=file_sha256(part / "predictions.jsonl"), code_sha256={}))
            validate_evaluation_identity(args, "dev", "ce")
            weights["qformer"].write_bytes(b"another bridge")
            with self.assertRaisesRegex(ValueError, "model identity"):
                validate_evaluation_identity(args, "dev", "ce")
            weights["qformer"].write_bytes(b"bridge fixture")
            predictions[0]["prediction"] = "foreign answer"
            dump_jsonl(root / "dev/ce/predictions.jsonl", predictions)
            with self.assertRaisesRegex(ValueError, "merged evaluation"):
                validate_evaluation_identity(args, "dev", "ce")

    def test_formal_bank_reuses_only_unchanged_successful_smoke_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(profile="formal", output_dir=root, gpus=["1", "2", "3", "5"])
            config = dict.fromkeys(("code_sha256", "initialization", "paths", "teacher_model", "teacher_source",
                                   "training", "student_loading"), {})
            write_json(root / "experiment.json", config)
            write_json(root / "smoke/experiment.json", config)
            evidence = root / "smoke/training.json"
            write_json(evidence, {"fixture": True})
            proof = dict(passed=True, world_size=4, source_sha256={str(evidence): file_sha256(evidence)})
            write_json(root / "smoke/smoke_validation.json", proof)
            with patch("scripts.qwen_drive_distill.child") as child:
                ensure_smoke(args)
                child.assert_not_called()
            write_json(evidence, {"fixture": "changed"})
            with self.assertRaisesRegex(ValueError, "stale"):
                ensure_smoke(args)

    def test_report_selects_using_dev_and_catches_stale_scores(self):
        original, s, t = fixtures()
        for row in s + t:
            row.update(token_f1=.5, normalized_exact=0.)
        ce = copy.deepcopy(s)
        ce[0].update(lingo_judge_correct=True, lingo_judge_logit=1.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=root)
            dump_jsonl(root / "data/dev.jsonl", original)
            for name in ("experiment.json", "data/selection.json", "data/sample_plan.json"):
                write_json(root / name, {"fixture": True})
            for role, rows in (("start", s), ("ce", ce), ("distill", t)):
                folder = root / "dev" / role
                write_json(folder / "run_manifest.json", {"fixture": True})
                dump_jsonl(folder / "predictions.jsonl", rows)
                dump_jsonl(folder / "scored/scored_predictions.jsonl", rows)
                write_json(folder / "scored/summary.json", dict(summarize(rows), judge_protocol=JUDGE_PROTOCOL))
            for arm in ("ce", "distill"):
                for name in ("run_config.json", "summary.json", "target_token_budget.json", "launch_identity.json"):
                    write_json(root / "train" / arm / name, {"fixture": True})
            keys = ("data_sha256", "sample_plan_sha256", "world_size", "seed", "lr", "projector_lr", "qformer_lr",
                    "warmup_ratio", "weight_decay", "grad_accumulation", "total_steps", "max_txt_len",
                    "trainable_parameters_by_group", "disable_dropout", "qformer_mode")
            config = dict.fromkeys(keys, "matched fixture")
            summary = dict(optimizer_steps=500, sample_presentations=8000, supervised_tokens_presented=10000)
            with patch("scripts.qwen_drive_distill.verify_training", return_value=(summary, config, {})), \
                 patch("scripts.qwen_drive_distill.validate_evaluation_identity"):
                value = report(args)
                self.assertEqual(value["selected_by_dev"], "ce")
                self.assertFalse(value["distillation_gain_established"])
                self.assertEqual(value["comparisons"]["distill_vs_start"]["lingo_judge_correct"]["fixed"], 4)
                self.assertEqual(value["comparisons"]["distill_vs_start"]["lingo_judge_correct"]["regressed"], 4)
                self.assertEqual(value["comparisons"]["distill_vs_start"]["unit"], "segment_id")
                t[0]["prediction"] = "Stale replacement"
                dump_jsonl(root / "dev/distill/scored/scored_predictions.jsonl", t)
                with self.assertRaisesRegex(ValueError, "stale"):
                    report(args)


if __name__ == "__main__":
    unittest.main()
