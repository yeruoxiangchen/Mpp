"""CPU/autograd contracts for fixed-success preservation in feedback OPD."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpp_video_qa.core import dump_jsonl, file_sha256
from mpp_video_qa.opd import SharedFrozenTeacher
from mpp_video_qa.opd_runtime import validate_resume_identity
from scripts.train_feedback_opd import (METRICS, backward_example, restore_counters,
                                        training_config, validate_probe)


class TinyAdapters(nn.Module):
    """PEFT-like adapter switches; checkpoint recomputation uses the active one."""
    def __init__(self):
        super().__init__()
        base = torch.arange(36, dtype=torch.float32).reshape(6, 6).sin()
        self.default = nn.Parameter(base.clone())
        self.opd_teacher = nn.Parameter(base.cos(), requires_grad=False)
        self.active_adapter = "default"

    def set_adapter(self, name):
        self.active_adapter = name
        for key, parameter in self.named_parameters():
            parameter.requires_grad_(key == name)


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = TinyAdapters()
        self.ce_calls = []

    def forward(self, features, questions, answers):
        assert self.llm.active_adapter == "default"
        self.ce_calls.append((questions, answers))
        return SimpleNamespace(loss=F.cross_entropy(self.llm.default[:1], torch.tensor([5])))

    def answer_only_ce(self, features, questions, answers):
        return self(features, questions, answers).loss


class TinyTeacherView(nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm


def models():
    student = TinyStudent()
    teacher = SharedFrozenTeacher.__new__(SharedFrozenTeacher)
    teacher.student = student
    teacher.llm = student.llm
    teacher.policy_names = {"default"}
    teacher.model = TinyTeacherView(student.llm)
    return student, teacher


def args(mode="sft_control", keep=0.7, **changes):
    values = dict(training_mode=mode, keep_weight=keep, ce_weight=1.3, opd_weight=2.1,
                  answer_only_logits=False, chunk_tokens=2, max_new_tokens=16,
                  teacher_mode="reference_only", runtime_log_flush_every=1,
                  success_bank="fixed-bank", success_bank_identity={"records_sha256": "bank-1"},
                  epochs=1, grad_accumulation=4, lr=1e-3, projector_lr=1e-3,
                  qformer_lr=1e-3, weight_decay=.01, warmup_ratio=0.)
    values.update(changes)
    return SimpleNamespace(**values)


def record(anchor=True):
    result = {"row": {"question_id": "q1", "question": "Which way?", "source_subset": "action",
                      "answers": ["The reference is left."]},
              "answer": "The reference is left.", "features": torch.zeros(5, 2, 3)}
    if anchor:
        result["success_token_ids"] = [3, 1, 2]
    return result


def diagnostic(eligible=True):
    return dict(teacher_cache_hit=False, teacher_generation_called=eligible,
                student_judge={"correct": not eligible}, student_finished=True,
                teacher_correction="correct" if eligible else None,
                teacher_finished=eligible, teacher_judge={"correct": True} if eligible else None,
                eligible=eligible, gate_reason="eligible" if eligible else "student_correct_ce_only")


def raw_logits(weights, ids, prompt=None):
    indices = torch.cat((ids.new_zeros(1), ids[:-1]))
    result = weights.index_select(0, indices)
    if prompt is not None:
        result = result + result.new_tensor([.6, -.3, .2, 0., -.1, .4])
    return result


def mock_logits(calls):
    def evaluate(model, features, question, ids, prompt=None, answer_only=False):
        is_student = isinstance(model, TinyStudent)
        expected = "default" if is_student else "opd_teacher"
        calls.append((expected, ids.tolist(), question, prompt, torch.is_grad_enabled(), answer_only))
        weights = getattr(model.llm, expected)
        def forward(value):
            assert model.llm.active_adapter == expected, "wrong adapter during checkpoint recomputation"
            return raw_logits(value, ids, prompt)
        return checkpoint(forward, weights, use_reentrant=False) if is_student else forward(weights)
    return evaluate


def direct_loss(student, config, divisor, keep=True, opd=False):
    """Independent full-vocabulary formula, without the production loss helper."""
    p = student.llm.default
    q = student.llm.opd_teacher.detach()
    loss = config.ce_weight * F.cross_entropy(p[:1], torch.tensor([5]))
    if keep and config.keep_weight:
        ids = torch.tensor([3, 1, 2])
        logp = raw_logits(p, ids).log_softmax(-1)
        logq = raw_logits(q, ids).log_softmax(-1)
        loss = loss + config.keep_weight * (logq.exp() * (logq - logp)).sum(-1).mean()
    if opd:
        ids = torch.tensor([4, 4, 2])
        p1 = raw_logits(p, ids).softmax(-1)
        q1 = raw_logits(q, ids, "privileged-reference").softmax(-1)
        mix = (p1 + q1) / 2
        jsd = .5 * ((p1 * (p1.log() - mix.log())).sum(-1) +
                     (q1 * (q1.log() - mix.log())).sum(-1)).mean()
        loss = loss + config.opd_weight * jsd
    return loss / divisor


class SuccessKeepTrainingTests(unittest.TestCase):
    def run_example(self, config, original=None, divisor=4, eligible=True):
        student, teacher = models()
        calls = []
        timer = SimpleNamespace(phase=lambda *a, **kw: contextlib.nullcontext())
        rollout = (torch.tensor([4, 4, 2]), "privileged-reference", diagnostic(eligible))
        with patch("scripts.train_feedback_opd.completion_logits", side_effect=mock_logits(calls)), \
             patch("scripts.train_feedback_opd.current_rollout", return_value=rollout) as generate:
            values = backward_example(config, student, teacher, SimpleNamespace(cpu_fallbacks=0),
                                      original or record(), torch.device("cpu"), timer, None,
                                      divisor, io.StringIO(), 0, 0)
        return student, teacher, calls, generate.call_count, values

    def test_ce_keep_fixed_ids_plain_prompt_no_online_generation_and_real_gradients(self):
        for answer_only in (False, True):
            with self.subTest(answer_only=answer_only):
                config = args(answer_only_logits=answer_only)
                original = record()
                saved = copy.deepcopy(original)
                student, teacher, calls, generations, values = self.run_example(config, original)
                self.assertEqual(generations, 0)
                self.assertEqual(calls, [("opd_teacher", [3, 1, 2], "Which way?", None, False, answer_only),
                                         ("default", [3, 1, 2], "Which way?", None, True, answer_only)])
                self.assertEqual(student.ce_calls, [(["Which way?"], ["The reference is left."])])
                self.assertEqual(original["success_token_ids"], saved["success_token_ids"])
                self.assertIsNone(student.llm.opd_teacher.grad)
                self.assertFalse(student.llm.opd_teacher.requires_grad)
                self.assertTrue(student.llm.default.requires_grad)
                self.assertEqual(student.llm.active_adapter, "default")
                expected, _ = models()
                direct_loss(expected, config, 4).backward()
                torch.testing.assert_close(student.llm.default.grad, expected.llm.default.grad)
                self.assertEqual((values["keep_examples"], values["keep_tokens"], values["rollouts"]), (1, 3, 0))
                self.assertGreater(values["keep_kl_sum"], 0)

    def test_opd_and_keep_both_use_all_qa_divisor_and_finish_before_adapter_switch(self):
        config = args(mode="feedback_opd")
        for divisor in (1., 16 / 3):
            with self.subTest(divisor=divisor):
                student, _, calls, generations, values = self.run_example(config, divisor=divisor)
                self.assertEqual(generations, 1)
                self.assertEqual([call[1] for call in calls], [[4, 4, 2], [4, 4, 2], [3, 1, 2], [3, 1, 2]])
                self.assertEqual([call[3] for call in calls], ["privileged-reference", None, None, None])
                expected, _ = models()
                direct_loss(expected, config, divisor, opd=True).backward()
                torch.testing.assert_close(student.llm.default.grad, expected.llm.default.grad)
                self.assertIsNone(student.llm.opd_teacher.grad)
                self.assertEqual((values["eligible"], values["keep_examples"], values["examples"]), (1, 1, 1))

    def test_keep_zero_exactly_matches_existing_objective_and_skips_teacher_keep_passes(self):
        for mode in ("sft_control", "feedback_opd"):
            with self.subTest(mode=mode):
                config = args(mode=mode, keep=0.)
                student, _, calls, _, values = self.run_example(config)
                expected, _ = models()
                direct_loss(expected, config, 4, keep=False, opd=mode == "feedback_opd").backward()
                torch.testing.assert_close(student.llm.default.grad, expected.llm.default.grad)
                self.assertEqual(len(calls), 2 if mode == "feedback_opd" else 0)
                self.assertEqual(values["keep_examples"], 0)

    def test_no_bank_anchor_is_ce_only_and_closed_error_gate_still_keeps_success(self):
        student, _, calls, _, values = self.run_example(args(), record(anchor=False))
        expected, _ = models()
        direct_loss(expected, args(), 4, keep=False).backward()
        torch.testing.assert_close(student.llm.default.grad, expected.llm.default.grad)
        self.assertEqual(calls, [])
        self.assertEqual(values["keep_tokens"], 0)
        _, _, calls, _, values = self.run_example(args(mode="feedback_opd"), eligible=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(values["keep_examples"], 1)
        self.assertEqual(values["eligible"], 0)

    def test_restore_twenty_metrics_keeps_actual_teacher_generation_count(self):
        saved = list(range(20))
        saved[METRICS.index("teacher_attempts")] = 99
        saved[METRICS.index("teacher_generations")] = 12
        cumulative = torch.zeros(len(METRICS), dtype=torch.float64)
        restore_counters(cumulative, saved)
        self.assertEqual(cumulative[:20].tolist(), saved)
        self.assertEqual(cumulative[20:23].tolist(), [0., 0., 0.])
        self.assertEqual(cumulative[METRICS.index("distill_sum")], saved[METRICS.index("jsd_sum")])
        self.assertEqual(cumulative[METRICS.index("distilled_wrong")], saved[METRICS.index("eligible")])
        legacy = torch.zeros_like(cumulative)
        restore_counters(legacy, saved[:17])
        self.assertEqual(legacy[METRICS.index("teacher_generations")].item(), 99)
        with self.assertRaisesRegex(ValueError, "future metrics"):
            restore_counters(cumulative, list(range(len(METRICS) + 1)))

    def test_bank_artifact_and_keep_weight_are_strict_resume_identity(self):
        original = training_config(args(), {"data_sha256": "rows", "initialization": "parent"},
                                   16, 1, 4, None, [])
        self.assertEqual(original["success_bank"], {"records_sha256": "bank-1"})
        self.assertEqual(original["keep_objective"], "forward_kl")
        validate_resume_identity(original, copy.deepcopy(original))
        for changed in (dict(original, keep_weight=.8),
                        dict(original, success_bank={"records_sha256": "bank-2"}),
                        dict(original, initialization="different-parent")):
            with self.assertRaises(ValueError):
                validate_resume_identity(original, changed)

    def test_subset_probe_accepts_exact_rows_rejects_model_data_and_provenance_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = [{"question_id": "q%d" % i, "question": "question %d" % i,
                          "answers": ["answer %d" % i], "frames": ["frame-%d" % i]}
                         for i in range(3)]
            parent, data, report = root / "parent.jsonl", root / "train.jsonl", root / "probe.json"
            dump_jsonl(parent, originals)
            dump_jsonl(data, [originals[1]])
            identity = {"data_sha256": file_sha256(data), "initialization": {"adapter": "parent"},
                        "teacher_mode": "reference_only", "max_new_tokens": 96}
            probe_identity = dict(identity, data_sha256=file_sha256(parent))
            manifest = {"parent_train_path": str(parent), "parent_train_sha256": file_sha256(parent),
                        "split_sha256": {"train": file_sha256(data)}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            value = {"protocol": "feedback_teacher_comparison_v2", "completed": True,
                     "modes": {"reference_only": {"eligible_draws": 1, "token_diagnostic_draws": 1}},
                     "identity_by_mode": {"reference_only": probe_identity}}
            report.write_text(json.dumps(value))
            config = args(data=str(data), probe_report=str(report))
            self.assertEqual(validate_probe(config, identity), file_sha256(report))
            for changed in (dict(identity, initialization={"adapter": "other"}),
                            dict(identity, max_new_tokens=32)):
                with self.assertRaises(ValueError):
                    validate_probe(config, changed)
            with self.assertRaises(ValueError):
                validate_probe(args(data=str(data), probe_report=str(report), success_bank=""), identity)
            # Rehashing a modified row is insufficient: it must still equal a parent row.
            changed_row = dict(originals[1], answers=["tampered answer"])
            dump_jsonl(data, [changed_row])
            changed_identity = dict(identity, data_sha256=file_sha256(data))
            manifest["split_sha256"]["train"] = file_sha256(data)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "exact rows"):
                validate_probe(config, changed_identity)
            dump_jsonl(data, [originals[1], originals[1]])
            manifest["split_sha256"]["train"] = file_sha256(data)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "exact rows"):
                validate_probe(config, dict(identity, data_sha256=file_sha256(data)))
            dump_jsonl(parent, originals + [{"question_id": "extra"}])
            with self.assertRaisesRegex(ValueError, "provenance"):
                validate_probe(config, identity)


if __name__ == "__main__":
    unittest.main()
