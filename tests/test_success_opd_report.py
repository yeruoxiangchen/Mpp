import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from mpp_video_qa.core import dump_jsonl, file_sha256, TASK_INSTRUCTIONS, PROTOCOL
from mpp_video_qa.data import cache_file
from scripts.generate_answers import extended_generation_identity, generation_dataset
from scripts.report_answer_opsd import JUDGE_PROTOCOL
from scripts.report_preference import summarize
from scripts.report_success_opd import (ARMS, LABELS, build_report, completed_arm,
    interaction, save_report, validate_configs, launch_state, load_evaluation)


def configs():
    base = dict(records=4000, epochs=2, total_steps=500, checkpoint_steps=[250],
        global_batch_size=16, world_size=3, grad_accumulation=6, lr=2.5e-6,
        projector_lr=2.5e-6, qformer_lr=5e-6, ce_weight=1., opd_weight=1.,
        weight_decay=.01, warmup_ratio=.03, seed=44, teacher_mode="reference_only",
        feature_stage="eva", precision_protocol="fp32_trainables_v1", dtype="bf16",
        task_instruction_version="legacy_v1", max_txt_len=1536, max_new_tokens=96,
        objective="jsd", keep_objective="forward_kl", success_bank={"manifest_sha256": "same"},
        base_checkpoint_sha256="same-base", bridge_init_sha256="same", cache_dir="same-cache")
    return {a: dict(base, training_mode="feedback_opd" if a.startswith("opd") else "sft_control",
                    keep_weight=float(a.endswith("_keep")),
                    probe_report_sha256="probe" if a.startswith("opd") else None) for a in ARMS}


def fixture(root):
    args = SimpleNamespace(data_dir=root / "data", eval_dir=root / "eval", history_dir=root / "history")
    args.data_dir.mkdir()
    keys = ["parent"] + [a + "__" + label for a in ARMS for label in LABELS]
    models = {k: {"adapter_identity": k, "projector_identity": k, "qformer_adapter_sha256": k} for k in keys}
    specs = {a + "__" + label: {"arm": a, "label": label, "step": step}
             for a in ARMS for label, step in LABELS.items()}
    for split in ("action", "scenery", "eval"):
        rows = [{"question_id": split + str(i), "segment_id": split + str(i // 2),
                 "question": "Original question", "answers": ["Original reference"]} for i in range(4)]
        source = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        dump_jsonl(source, rows)
        for key in keys:
            folder = args.eval_dir / (key + "_" + split)
            (folder / "scored").mkdir(parents=True)
            run = {"protocol": "test-greedy", "data_sha256": file_sha256(source), "cache_dir": "same-cache",
                   "feature_stage": "eva", "task_instruction_version": "legacy_v1", "task_instruction": TASK_INSTRUCTIONS["legacy_v1"],
                   "precision_protocol": "fp32_trainables_v1", "dtype": "bf16", "max_new_tokens": 96,
                   "max_records": 0, "batch_size": 1, "bridge_init_sha256": "same", **models[key],
                   "base_checkpoint_sha256": "same-base", "max_txt_len": 1536, "qformer_mode": "auto",
                   "generation_policy": {"do_sample": False, "num_beams": 1, "top_p": 1.0, "top_k": 50}}
            (folder / "run_manifest.json").write_text(json.dumps(run))
            correct = [True, True, False, False]
            if key.startswith("opd_keep__"):
                correct = [True, True, True, False]
            if key == "opd__final":
                correct = [True, False, True, False]
            if split == "eval" and key == "ce__final":
                correct = [True] * 4  # Must not influence dev selection.
            predictions = [{"question_id": r["question_id"], "segment_id": r["segment_id"],
                            "question": r["question"], "references": r["answers"], "prediction": "An answer",
                            "lingo_judge_correct": v, "token_f1": float(v), "normalized_exact": float(v)}
                           for r, v in zip(rows, correct)]
            dump_jsonl(folder / "predictions.jsonl", predictions)
            dump_jsonl(folder / "scored/scored_predictions.jsonl", predictions)
            (folder / "scored/summary.json").write_text(json.dumps(dict(summarize(predictions), judge_protocol=JUDGE_PROTOCOL)))
    return args, (models, specs, configs())


class SuccessOPDReportTests(unittest.TestCase):
    def test_factorial_configs_reject_unmatched_and_wrong_factors(self):
        original = configs()
        validate_configs(original)
        for key, value in (("keep_weight", 1), ("lr", 1e-5), ("world_size", 2), ("success_bank", {"different": 1})):
            changed = copy.deepcopy(original)
            changed["ce"][key] = value
            with self.assertRaises(ValueError):
                validate_configs(changed)
        changed = copy.deepcopy(original)
        changed["opd_keep"]["probe_report_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "same validated"):
            validate_configs(changed)

    def test_interaction_is_paired_and_bootstraps_videos(self):
        rows = {a: [{"question_id": str(i), "segment_id": "v" + str(i // 2),
                     "lingo_judge_correct": a == "opd_keep"} for i in range(4)] for a in ARMS}
        value = interaction(rows)
        self.assertEqual(value["mean"], 1.)
        self.assertEqual(value["segments"], 2)
        self.assertEqual(value["interval_95pct"], [1., 1.])
        rows["opd"][0]["segment_id"] = "foreign"
        with self.assertRaisesRegex(ValueError, "video identity"):
            interaction(rows)

    def test_two_step_selection_interaction_and_parent_strata(self):
        with tempfile.TemporaryDirectory() as directory:
            args, meta = fixture(Path(directory))
            with patch("scripts.report_success_opd.model_specs", return_value=meta):
                dev = build_report(args)
                self.assertTrue(all(k.endswith("checkpoint-000250") for k in dev["selected_checkpoints"].values()))
                block = dev["dev_combined"]["same_step"]["final"]
                self.assertEqual(block["interaction"]["mean"], .25)
                self.assertEqual(block["by_parent"]["parent_correct"]["paired"]["opd_vs_opd_keep"]["lingo_judge_correct"]["fixed"], 2)
                save_report(args, dev)
                dev_artifacts = {suffix: (args.eval_dir / ("report." + suffix)).read_bytes()
                                 for suffix in ("json", "md")}
                official = build_report(args, True)
                self.assertNotIn("ce__final", official["splits"]["eval"]["models"])
                self.assertIn("ce_keep_vs_opd_keep", official["selected_comparison"]["paired"])
                save_report(args, official, True)
                for suffix in ("json", "md"):
                    self.assertTrue((args.eval_dir / ("official_report." + suffix)).is_file())
                    self.assertEqual((args.eval_dir / ("report." + suffix)).read_bytes(), dev_artifacts[suffix])
                self.assertEqual(json.loads((args.eval_dir / "official_report.json").read_text()), official)
                self.assertIn("| parent | 4 |", (args.eval_dir / "official_report.md").read_text())

    def test_official_rejects_missing_tampered_or_stale_dev_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args, meta = fixture(Path(directory))
            with patch("scripts.report_success_opd.model_specs", return_value=meta):
                with self.assertRaisesRegex(ValueError, "development selection"):
                    build_report(args, True)
                dev = build_report(args)
                dev["selected_checkpoints"]["ce"] = "ce__final"
                save_report(args, dev)
                with self.assertRaisesRegex(ValueError, "selection changed"):
                    build_report(args, True)
                save_report(args, build_report(args))
                path = args.eval_dir / "ce__final_action/scored/scored_predictions.jsonl"
                path.write_text(path.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "evaluation changed"):
                    build_report(args, True)

    def test_corrupt_parent_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, meta = fixture(Path(directory))
            path = args.eval_dir / "parent_action/run_manifest.json"
            run = json.loads(path.read_text()); run["adapter_identity"] = "wrong"
            path.write_text(json.dumps(run))
            with patch("scripts.report_success_opd.model_specs", return_value=meta):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    build_report(args)

    def test_consistently_tampered_report_metrics_and_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, meta = fixture(Path(directory))
            with patch("scripts.report_success_opd.model_specs", return_value=meta):
                dev = build_report(args)
                dev["dev_combined"]["models"]["ce__final"]["lingo_judge_accuracy"] = .99
                dev["selected_checkpoints"]["ce"] = "ce__final"
                save_report(args, dev)
                with self.assertRaisesRegex(ValueError, "metrics differ from hashed"):
                    build_report(args, True)
                dev = build_report(args)
                dev["evaluation_sources"].pop(next(iter(dev["evaluation_sources"])))
                save_report(args, dev)
                with self.assertRaisesRegex(ValueError, "source coverage"):
                    build_report(args, True)

    def test_reuse_validator_rejects_roi_context_base_and_generation_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, meta = fixture(Path(directory))
            folder = args.eval_dir / "opd_keep__final_action"
            path = folder / "run_manifest.json"
            original = json.loads(path.read_text())
            changes = (("region_views", "extra-crops"), ("question_qformer", {}),
                       ("base_checkpoint_sha256", "wrong-base"), ("max_txt_len", 1024),
                       ("max_new_tokens", 12), ("batch_size", 2), ("max_records", 3),
                       ("cache_dir", "other-cache"), ("bridge_init_sha256", "other-bridge"),
                       ("generation_policy", {"do_sample": True}))
            for key, value in changes:
                changed = dict(original); changed[key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "inference protocol"):
                    load_evaluation(folder, args.data_dir / "dev_action.jsonl", meta[0]["opd_keep__final"], meta[2]["ce"])
            path.write_text(json.dumps(original))
            self.assertEqual(len(load_evaluation(folder, args.data_dir / "dev_action.jsonl",
                                                meta[0]["opd_keep__final"], meta[2]["ce"])[0]), 4)

    def test_generation_provenance_is_opt_in_and_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mpp-qwen-next-sft/sharded/model.safetensors.index.json"
            source.parent.mkdir(parents=True)
            source.write_text('{"weight_map": {}}')
            args = SimpleNamespace(checkpoint_root=root, max_txt_len=1536, record_base_identity=False)
            self.assertEqual(extended_generation_identity(args), {})
            args.record_base_identity = True
            result = extended_generation_identity(args)
            self.assertEqual(result["base_checkpoint_sha256"], file_sha256(source))
            self.assertEqual(result["max_txt_len"], 1536)
            self.assertFalse(result["generation_policy"]["do_sample"])

    def test_strict_generation_validates_exact_five_frame_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"question_id": "q", "segment_id": "s", "question": "Question",
                   "answers": ["Answer"], "frames": ["frame%d" % i for i in range(5)]}
            source = root / "data.jsonl"
            dump_jsonl(source, [row])
            args = SimpleNamespace(data=source, cache_dir=root / "cache", feature_stage="eva",
                region_views="", region_cache_dir="", region_condition="roi",
                task_instruction_version="legacy_v1", record_base_identity=True)
            path = cache_file(args.cache_dir, "s")
            path.parent.mkdir(parents=True)
            payload = {"protocol": PROTOCOL, "segment_id": "s", "frame_paths": row["frames"],
                       "view": "full", "feature_stage": "eva", "features": torch.zeros(5, 1, 1408)}
            torch.save(payload, path)
            self.assertEqual(generation_dataset(args)[0]["features"].shape[0], 5)
            for key, value in (("view", "crop"), ("frame_paths", list(reversed(row["frames"]))),
                               ("features", torch.zeros(4, 1, 1408))):
                changed = dict(payload); changed[key] = value
                torch.save(changed, path)
                with self.assertRaisesRegex(RuntimeError, "original full frames"):
                    generation_dataset(args)[0]
            row["frames"] = row["frames"][:4]
            dump_jsonl(source, [row])
            with self.assertRaisesRegex(ValueError, "original five full frames"):
                generation_dataset(args)

    def test_summary_alone_is_not_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = configs()["ce"]
            summary = dict(completed=True, optimizer_steps=500, optimizer_steps_total=500,
                           records=4000, training_mode="sft_control", cumulative_metrics={"examples": 8000, "rollouts": 0})
            (folder / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                completed_arm(folder, config)
            summary["optimizer_steps"] = 250
            (folder / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "incomplete training summary"):
                completed_arm(folder, config)

    def test_complete_arm_checks_keep_budget_and_frozen_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = configs()["opd_keep"]
            config["success_bank"]["eligible"] = 2
            summary = dict(completed=True, optimizer_steps=500, optimizer_steps_total=500,
                           records=4000, training_mode="feedback_opd",
                           cumulative_metrics={"examples": 8000, "rollouts": 8000, "keep_examples": 4},
                           teacher_tracked_parameter_deltas={"language": 0.})
            for label, step in LABELS.items():
                bundle = folder / label
                (bundle / "adapter").mkdir(parents=True)
                for name in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                             "projector.pt", "qformer_adapter.pt", "training_state.pt"):
                    (bundle / name).write_bytes(b"test-fixture")
                (bundle / "metadata.json").write_text(json.dumps({"identity": config, "optimizer_steps": step}))
            path = folder / "summary.json"
            path.write_text(json.dumps(summary))
            self.assertTrue(completed_arm(folder, config))
            summary["cumulative_metrics"]["keep_examples"] = 3
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "replay count"):
                completed_arm(folder, config)
            summary["cumulative_metrics"]["keep_examples"] = 4
            summary["teacher_tracked_parameter_deltas"]["language"] = .01
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "teacher weights changed"):
                completed_arm(folder, config)

    def test_unfinished_run_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(train_dir=Path(directory), arm="opd", launch_world_size=3)
            with patch("scripts.report_success_opd.checked_configs", return_value=({"opd": configs()["opd"]}, {})):
                with self.assertRaisesRegex(ValueError, "use resume"):
                    launch_state(args)
                with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                    launch_state(args, True)
                args.launch_world_size = 2
                with self.assertRaisesRegex(ValueError, "same GPU count"):
                    launch_state(args, True)

    def test_runner_preserves_exact_batch_and_four_factor_commands(self):
        runner = Path(__file__).resolve().parents[1] / "scripts/run_success_opd.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "python-stub"
            stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ['CAPTURE'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\nif 'report_success_opd.py' in ' '.join(sys.argv):print('new')\n")
            stub.chmod(0o755)
            for world, accumulation in ((2, 8), (3, 6), (4, 4)):
                capture = root / ("commands%d" % world)
                env = dict(os.environ, PYTHON_BIN=str(stub), CAPTURE=str(capture),
                           SUCCESS_OPD_TRAIN_DIR=str(root / ("run%d" % world)),
                           CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(world)))
                env.pop("NPROC_PER_NODE", None)
                subprocess.run(["bash", str(runner), "train_all"], env=env, check=True, capture_output=True)
                calls = [json.loads(l) for l in capture.read_text().splitlines()]
                training = [c for c in calls if "torch.distributed.run" in c]
                self.assertEqual(len(training), 4)
                for arm, cmd in zip(ARMS, training):
                    value = lambda flag: cmd[cmd.index(flag) + 1]
                    self.assertEqual(value("--global-batch-size"), "16")
                    self.assertEqual(value("--grad-accumulation"), str(accumulation))
                    self.assertEqual(value("--epochs"), "2")
                    self.assertEqual(value("--checkpoint-steps"), "250")
                    self.assertEqual(value("--keep-weight"), str(int(arm.endswith("_keep"))))
                    self.assertEqual(value("--training-mode"), "feedback_opd" if arm.startswith("opd") else "sft_control")
                    self.assertEqual(value("--opd-weight"), "1")
            for visible in ("0,0", "0,1,", "0,1,2,3,4"):
                env["CUDA_VISIBLE_DEVICES"] = visible
                result = subprocess.run(["bash", str(runner), "train", "ce"], env=env, capture_output=True)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
