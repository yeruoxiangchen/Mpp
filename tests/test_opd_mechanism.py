import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mpp_video_qa.feedback_opd import gate_reason
from mpp_video_qa.opd_runtime import validate_resume_identity
from scripts.feedback_opd_common import current_rollout
from scripts.train_feedback_opd import backward_example, training_config
from tests.test_success_keep_training import models, args, record, diagnostic, mock_logits, raw_logits
from scripts import opd_mechanism_experiment as experiment


class MechanismTests(unittest.TestCase):
    def test_forward_kl_gradient_teacher_detached_real_divisor(self):
        config = args(mode="feedback_opd", keep=0., opd_objective="forward_kl")
        student, teacher = models()
        timer = SimpleNamespace(phase=lambda *a, **kw: contextlib.nullcontext())
        ids = torch.tensor([4, 4, 2])
        with patch("scripts.train_feedback_opd.completion_logits", side_effect=mock_logits([])), \
             patch("scripts.train_feedback_opd.current_rollout", return_value=(ids, "privileged-reference", diagnostic())):
            values = backward_example(config, student, teacher, SimpleNamespace(cpu_fallbacks=0), record(False),
                                      torch.device("cpu"), timer, None, 16/3, io.StringIO(), 0, 0)
        expected, _ = models()
        s = raw_logits(expected.llm.default, ids).log_softmax(-1)
        t = raw_logits(expected.llm.opd_teacher.detach(), ids, "privileged-reference").log_softmax(-1)
        ce = torch.nn.functional.cross_entropy(expected.llm.default[:1], torch.tensor([5]))
        loss = (config.ce_weight*ce + config.opd_weight*(t.exp()*(t-s)).sum(-1).mean())/(16/3)
        loss.backward()
        torch.testing.assert_close(student.llm.default.grad, expected.llm.default.grad)
        self.assertIsNone(student.llm.opd_teacher.grad)
        self.assertEqual(values["jsd_sum"], 0)
        self.assertGreater(values["distill_sum"], 0)
        self.assertEqual((values["distilled_correct"], values["distilled_wrong"]), (0, 1))

    def test_all_gate_still_rejects_teacher_errors_and_truncation(self):
        self.assertEqual(gate_reason(True, True, True, True), "student_correct_ce_only")
        self.assertEqual(gate_reason(True, True, True, True, "all"), "eligible")
        self.assertEqual(gate_reason(True, True, True, False, "all"), "teacher_wrong_ce_only")
        self.assertEqual(gate_reason(False, True, True, True, "all"), "student_truncated")

    def test_actual_correct_rollout_gets_teacher_gate_without_replacing_ids(self):
        teacher = SimpleNamespace(active=lambda: contextlib.nullcontext("teacher-view"))
        student = SimpleNamespace(task_instruction="instruction")
        row = dict(question="Where?", frames=list(range(5)), answers=["left"], feedback_training_source="train")
        ids = torch.tensor([3, 4, 9])
        for teacher_ok in (True, False):
            judge = SimpleNamespace(score=lambda r, a: {"logit": 1. if a == "student" or teacher_ok else -1.,
                                                        "correct": a == "student" or teacher_ok})
            with patch("scripts.feedback_opd_common.sample_completions", side_effect=[(ids, True), (torch.tensor([8, 9]), True)]) as sample, \
                 patch("scripts.feedback_opd_common.decode_completion", side_effect=["student", "teacher"]):
                got, prompt, info = current_rollout(student, teacher, judge, None, row, 16,
                                                    "reference_only", trajectory_scope="all")
            self.assertEqual(sample.call_count, 2)
            self.assertTrue(torch.equal(got, ids))
            self.assertEqual(info["eligible"], teacher_ok)
            self.assertIn("reference_answers", prompt)
            self.assertNotIn("previous_student_answer", prompt)

    def test_objective_and_scope_cannot_change_on_resume(self):
        a = training_config(args(mode="feedback_opd"), {}, 4000, 2, 250, "probe", [100])
        b = training_config(args(mode="feedback_opd", opd_objective="forward_kl"), {}, 4000, 2, 250, "probe", [100])
        c = training_config(args(mode="feedback_opd", opd_trajectory_scope="all"), {}, 4000, 2, 250, "probe", [100])
        for other in (b, c):
            with self.assertRaises(ValueError):
                validate_resume_identity(a, other)

    def test_dev_selection_early_ties_and_independent_arms(self):
        scores={a+"__"+l:{"lingo_judge_accuracy":.5} for a in experiment.ARMS for l in experiment.LABELS}
        scores["kl_errors__final"]["lingo_judge_accuracy"] = .51
        chosen=experiment.select_checkpoints(scores)
        self.assertEqual(chosen["kl_errors"],"kl_errors__final")
        self.assertEqual(chosen["jsd_all"],"jsd_all__checkpoint-000100")
        self.assertEqual(chosen["ce"],"ce__checkpoint-000100")

    def test_arm_validation_rejects_unmatched_loss_gate_and_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            exp={"runtime_identity":{"data_sha256":"train","initialization":"parent"},
                 "fixed":{"records":4000,"epochs":1,"total_steps":250,"keep_weight":0}}
            experiment.write(folder/"experiment.json",exp)
            config=dict(exp["runtime_identity"],**exp["fixed"],training_mode="feedback_opd",
                        objective="forward_kl",checkpoint_steps=[100],feature_stage="eva",world_size=2,grad_accumulation=8)
            with patch.object(experiment,"RUN",folder):
                experiment.write(folder/"kl_errors/run_config.json",config)
                experiment.validate_arm("kl_errors")
                for k,v in [("objective","jsd"),("opd_trajectory_scope","all"),("total_steps",500),("data_sha256","different")]:
                    experiment.write(folder/"kl_errors/run_config.json",dict(config,**{k:v}))
                    with self.assertRaises(ValueError): experiment.validate_arm("kl_errors")

    def test_complete_report_pairs_fixed_steps_and_keeps_original_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary); eval_dir=folder/"eval"; run=folder/"train"
            run.mkdir(); (run/"experiment.json").write_text('{}')
            def evaluated(arm,label,split):
                good = arm=="parent" or (arm=="kl_errors" and label=="final")
                return [dict(question_id=split+str(i),segment_id=split,question="q",references=["a"],
                             lingo_judge_correct=bool(i==0 or good),token_f1=.5,normalized_exact=0.) for i in range(2)],()
            for arm,label in [("parent","final")]+[(a,l) for a in experiment.ARMS for l in experiment.LABELS]:
                key="parent" if arm=="parent" else arm+"__"+label
                for split in ("action","scenery"):
                    for f in ("run_manifest.json","predictions.jsonl","scored/summary.json","scored/scored_predictions.jsonl"):
                        p=eval_dir/(key+"_"+split)/f; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')
            with patch.object(experiment,"RUN",run),patch.object(experiment,"EVAL",eval_dir), \
                 patch.object(experiment,"verify_contract"), \
                 patch.object(experiment,"validate_arm",return_value={"world_size":2,"grad_accumulation":8}), \
                 patch.object(experiment,"identity",return_value={"model":"validated"}), \
                 patch.object(experiment,"evaluated",side_effect=evaluated):
                experiment.report()
            report=json.loads((eval_dir/"report.json").read_text())
            self.assertEqual(report["selected"]["kl_errors"],"kl_errors__final")
            self.assertEqual(report["selected"]["jsd_all"],"jsd_all__checkpoint-000100")
            self.assertEqual(report["fixed_step"]["final"]["jsd_errors_vs_kl_errors"]["lingo_judge_correct"]["fixed"],2)
            self.assertEqual(len(report["source_hashes"]),72)
            self.assertTrue((eval_dir/"report.md").exists())


if __name__ == "__main__":
    unittest.main()
