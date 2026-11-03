import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from mpp_video_qa.core import dump_jsonl, file_sha256
from mpp_video_qa.full_sft_data import validate_full_training
from mpp_video_qa.sft_schedule import StagedSFT
from scripts.feedback_opd_common import training_dataset
from scripts.train_feedback_opd import optimizer_budget


def fixture_data(root):
    data, parent = root / "data", root / "parent"
    data.mkdir(); parent.mkdir()
    def row(qid, source="action"):
        return {"question_id": qid, "segment_id": qid + "-video", "source_subset": source,
                "question": "What is ahead?", "answers": ["A car."],
                "frames": ["/%s/frame%d.jpg" % (qid, i) for i in range(5)]}
    rows = [row("train-a"), row("train-s", "scenery")]
    splits = {"train": rows, "dev_action": [row("dev-a")], "dev_scenery": [row("dev-s", "scenery")],
              "eval": [row("official")]}
    splits["dev"] = splits["dev_action"] + splits["dev_scenery"]
    for name, values in splits.items():
        dump_jsonl(data / (name + ".jsonl"), values)
    train_hash = file_sha256(data / "train.jsonl")
    manifest = {"sampling": "all_original_training_rows", "records": 2, "segments": 2,
                "source_counts": {"action": 1, "scenery": 1}, "source_train_sha256": train_hash,
                "split_sha256": {name: file_sha256(data / (name + ".jsonl")) for name in splits}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    (parent / "run_config.json").write_text(json.dumps({"data_sha256": train_hash}))
    return data, parent, rows, manifest


class FullOriginalDataTests(unittest.TestCase):
    def test_unchanged_original_data_needs_no_feedback_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            data, parent, rows, _ = fixture_data(Path(directory))
            proof = validate_full_training(data / "train.jsonl", parent)
            self.assertTrue(proof["original_qa_preserved"])
            a = SimpleNamespace(data=data / "train.jsonl", initial_dir=parent, cache_dir=data / "cache",
                                sft_data_contract="original_full")
            dataset = training_dataset(a)
            self.assertEqual(dataset.rows, rows)
            a.sft_data_contract = "feedback"
            with self.assertRaisesRegex(ValueError, "prepare_feedback_opd"):
                training_dataset(a)

    def test_modified_reference_or_source_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data, parent, rows, _ = fixture_data(Path(directory))
            rows[0]["answers"] = ["A rewritten teacher answer"]
            dump_jsonl(data / "train.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "unchanged complete original"):
                validate_full_training(data / "train.jsonl", parent)

    def test_frame_leakage_rejected_even_with_self_consistent_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            data, parent, rows, manifest = fixture_data(Path(directory))
            held = json.loads((data / "eval.jsonl").read_text().strip())
            rows[0]["frames"][0] = held["frames"][0]
            dump_jsonl(data / "train.jsonl", rows)
            changed = file_sha256(data / "train.jsonl")
            manifest["split_sha256"]["train"] = manifest["source_train_sha256"] = changed
            (data / "manifest.json").write_text(json.dumps(manifest))
            (parent / "run_config.json").write_text(json.dumps({"data_sha256": changed}))
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_full_training(data / "train.jsonl", parent)

    def test_complete_budget_and_real_tail(self):
        self.assertEqual(optimizer_budget(406212, 4, 4, 1), 25389)
        self.assertEqual(406212 - (25389 - 1) * 16, 4)


def toy_model():
    torch.manual_seed(5)
    model = nn.Module()
    model.backbone = nn.Module()
    b = model.backbone
    b.llm_model = nn.Module()
    b.llm_model.lora_weight = nn.Parameter(torch.randn(2, 2))
    b.llm_proj = nn.Linear(2, 2)
    b.visual_bridge = nn.Module()
    bridge = b.visual_bridge
    bridge.query_tokens = nn.Parameter(torch.randn(1, 2, 2))
    bridge.Qformer = nn.Module()
    bridge.Qformer.bert = nn.Module()
    bert = bridge.Qformer.bert
    bert.embeddings = nn.Module()
    bert.embeddings.word_embeddings = nn.Embedding(4, 2)
    bert.encoder = nn.Module()
    layer = nn.Module()
    layer.intermediate = nn.Linear(2, 2)
    layer.attention = nn.Linear(2, 2)
    bert.encoder.layer = nn.ModuleList([layer])
    return model


def stage_args():
    return SimpleNamespace(lr=.01, projector_lr=.02, qformer_lr=.03, qformer_text_lr=.04,
                           language_freeze_steps=2, language_warmup_steps=1,
                           weight_decay=.1, warmup_ratio=0)


def update(model, stage, step):
    active = stage.apply_phase(step)
    stage.optimizer.zero_grad(set_to_none=True)
    loss = sum((p - .2).square().sum() for p in model.parameters())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(active, 1)
    stage.optimizer.step()
    stage.scheduler.step()


class StagedOptimizerTests(unittest.TestCase):
    def test_all_groups_disjoint_and_text_is_separate(self):
        model = toy_model()
        stage = StagedSFT(list(model.named_parameters()), stage_args(), 6)
        self.assertEqual([g["group_name"] for g in stage.groups], ["language", "projector", "qformer", "qformer_text"])
        ids = [id(p) for g in stage.groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        words = model.backbone.visual_bridge.Qformer.bert.embeddings.word_embeddings.weight
        self.assertIn(id(words), {id(p) for p in stage.groups[-1]["params"]})

    def test_freeze_excludes_adam_state_and_weight_decay_then_unfreezes(self):
        model = toy_model()
        stage = StagedSFT(list(model.named_parameters()), stage_args(), 6)
        language = model.backbone.llm_model.lora_weight
        before = language.detach().clone()
        for step in range(2):
            update(model, stage, step)
            self.assertTrue(torch.equal(language, before))
            self.assertIsNone(language.grad)
            self.assertNotIn(language, stage.optimizer.state)
        for step in range(2, 6):
            update(model, stage, step)
        self.assertFalse(torch.equal(language, before))
        self.assertIn(language, stage.optimizer.state)

    def test_resume_before_at_after_phase_boundary_matches_uninterrupted(self):
        reference = toy_model()
        ref_stage = StagedSFT(list(reference.named_parameters()), stage_args(), 6)
        for step in range(6):
            update(reference, ref_stage, step)
        for boundary in (1, 2, 3):
            source = toy_model()
            stage = StagedSFT(list(source.named_parameters()), stage_args(), 6)
            for step in range(boundary):
                update(source, stage, step)
            state = copy.deepcopy({"optimizer": stage.optimizer.state_dict(), "scheduler": stage.scheduler.state_dict(),
                                   "sft_stage": stage.stage_state(boundary)})
            restored = toy_model()
            restored.load_state_dict(source.state_dict())
            resumed = StagedSFT(list(restored.named_parameters()), stage_args(), 6)
            resumed.restore(state, boundary)
            for step in range(boundary, 6):
                update(restored, resumed, step)
            for key, value in reference.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), (boundary, key))

    def test_resume_rejects_changed_group_order_or_schedule_position(self):
        model = toy_model()
        stage = StagedSFT(list(model.named_parameters()), stage_args(), 6)
        update(model, stage, 0)
        state = copy.deepcopy({"optimizer": stage.optimizer.state_dict(), "scheduler": stage.scheduler.state_dict(),
                               "sft_stage": stage.stage_state(1)})
        state["sft_stage"]["group_names"].reverse()
        with self.assertRaisesRegex(ValueError, "group order"):
            stage.restore(state, 1)


class FullSFTCommandTests(unittest.TestCase):
    def test_shell_launches_both_four_gpu_full_budget_arms(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            stub = folder / "capture"
            stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ['CAPTURE_PATH'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(stub), CAPTURE_PATH=str(folder / "calls.jsonl"),
                       CUDA_VISIBLE_DEVICES="4,5,6,7", QUESTION_FULL_TRAIN_DIR=str(folder / "question"),
                       QUESTION_FULL_CONTROL_DIR=str(folder / "visual"))
            subprocess.run(["bash", "scripts/run_question_qformer_full.sh", "train_pair"], cwd=root,
                           env=env, check=True, capture_output=True)
            calls = [json.loads(s) for s in (folder / "calls.jsonl").read_text().splitlines()]
            launches = [a for a in calls if "torch.distributed.run" in a]
            self.assertEqual(len(launches), 2)
            for call, mode in zip(launches, ("none", "question")):
                self.assertIn("--nproc_per_node=4", call)
                for key, value in (("--epochs", "1"), ("--max-steps", "0"), ("--grad-accumulation", "4"),
                                   ("--sft-data-contract", "original_full"), ("--qformer-text-input", mode),
                                   ("--qformer-text-lr", "2e-5"), ("--language-freeze-steps", "1000")):
                    self.assertEqual(call[call.index(key) + 1], value)


if __name__ == "__main__":
    unittest.main()
