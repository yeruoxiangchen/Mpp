import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mpp_video_qa.core import dump_jsonl, file_sha256
from scripts.report_answer_opsd import JUDGE_PROTOCOL
from scripts.report_preference import summarize
from scripts.report_question_qformer_full import build_report, save_report


def fixture(root, control=True):
    a = SimpleNamespace(data_dir=root / "data", eval_dir=root / "eval", history_dir=root / "history")
    a.data_dir.mkdir()
    keys = ["parent", "retained", "question_checkpoint-005000", "question_final"]
    if control:
        keys += ["visual_checkpoint-005000", "visual_final"]
    models = {k: {"adapter_identity": k, "projector_identity": k, "qformer_adapter_sha256": k} for k in keys}
    specs = {k: {"branch": k.split("_", 1)[0], "step": 5000 if "checkpoint" in k else 25389}
             for k in keys if k not in ("parent", "retained")}
    configs = {"question": {"question_qformer": {"protocol": "mpp_question_qformer_v1", "mode": "question"}}}
    complete = {"question": True}
    if control:
        configs["visual"] = {}
        complete["visual"] = True
    for split in ("action", "scenery", "eval"):
        rows = [{"question_id": split + str(i), "segment_id": split + str(i),
                 "question": "Original question", "answers": ["Original answer"]} for i in range(2)]
        data = a.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        dump_jsonl(data, rows)
        for key in keys:
            if key in ("parent", "retained"):
                prefix = "baseline" if key == "parent" else "sft_control_final"
                folder = a.history_dir / (prefix + "_" + split)
            else:
                folder = a.eval_dir / (key + "_" + split)
            (folder / "scored").mkdir(parents=True)
            run = {"protocol": "test-greedy", "data_sha256": file_sha256(data), "cache_dir": "same-cache",
                   "feature_stage": "eva", "task_instruction_version": "legacy_v1", "task_instruction": "same",
                   "precision_protocol": "fp32_trainables_v1", "dtype": "bf16", "max_new_tokens": 96,
                   "max_records": 0, "batch_size": 1, "bridge_init_sha256": "same", **models[key]}
            if key.startswith("question_"):
                run["question_qformer"] = configs["question"]["question_qformer"]
            (folder / "run_manifest.json").write_text(json.dumps(run))
            correct = [True, False]
            if key in ("question_checkpoint-005000", "visual_final"):
                correct = [True, True]
            if split == "eval" and key == "question_final":
                correct = [True, True]
            predictions = [{"question_id": r["question_id"], "segment_id": r["segment_id"], "question": r["question"],
                            "references": r["answers"], "prediction": "An answer", "lingo_judge_correct": v,
                            "token_f1": float(v), "normalized_exact": float(v)} for r, v in zip(rows, correct)]
            dump_jsonl(folder / "predictions.jsonl", predictions)
            dump_jsonl(folder / "scored/scored_predictions.jsonl", predictions)
            (folder / "scored/summary.json").write_text(json.dumps(dict(summarize(predictions), judge_protocol=JUDGE_PROTOCOL)))
    return a, (models, specs, configs, complete)


class FullReportTests(unittest.TestCase):
    def test_matched_full_control_and_official_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args, metadata = fixture(Path(directory))
            with patch("scripts.report_question_qformer_full.model_specs", return_value=metadata):
                dev = build_report(args)
                self.assertTrue(dev["matched_full_control_available"])
                self.assertEqual(dev["selected_checkpoints"]["question"], "question_checkpoint-005000")
                self.assertIn("visual_vs_question", dev["selected_comparison"]["paired"])
                save_report(args, dev)
                official = build_report(args, True)
                self.assertNotIn("question_final", official["splits"]["eval"]["models"])

    def test_question_only_is_reported_without_claiming_matched_control(self):
        with tempfile.TemporaryDirectory() as directory:
            args, metadata = fixture(Path(directory), control=False)
            with patch("scripts.report_question_qformer_full.model_specs", return_value=metadata):
                dev = build_report(args)
                self.assertFalse(dev["matched_full_control_available"])
                self.assertTrue(dev["selection_finalized"])
                self.assertIn("retained_vs_question", dev["selected_comparison"]["paired"])
                save_report(args, dev)
                self.assertFalse(build_report(args, True)["matched_full_control_available"])

    def test_partial_training_cannot_finalize_official_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args, metadata = fixture(Path(directory))
            metadata[-1]["question"] = False
            with patch("scripts.report_question_qformer_full.model_specs", return_value=metadata):
                dev = build_report(args)
                self.assertFalse(dev["selection_finalized"])
                save_report(args, dev)
                with self.assertRaisesRegex(ValueError, "lock matching development"):
                    build_report(args, True)

    def test_changed_question_inference_identity_or_dev_selection_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, metadata = fixture(Path(directory))
            with patch("scripts.report_question_qformer_full.model_specs", return_value=metadata):
                dev = build_report(args)
                dev["selected_checkpoints"]["question"] = "question_final"
                save_report(args, dev)
                with self.assertRaisesRegex(ValueError, "selection changed"):
                    build_report(args, True)
                path = args.eval_dir / "question_final_action/run_manifest.json"
                run = json.loads(path.read_text()); run.pop("question_qformer")
                path.write_text(json.dumps(run))
                with self.assertRaisesRegex(ValueError, "unexpected question bridge"):
                    build_report(args)


if __name__ == "__main__":
    unittest.main()
