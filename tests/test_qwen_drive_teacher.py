import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256
from mpp_video_qa.driving_teacher import (
    PROTOCOL, INSTRUCTION, checked_scores, completion_result, immutable_json,
    paired_comparison, read, validate_data, validate_partial, write_json, write_rows,
)
from scripts.generate_qwen_drive_teacher import build_inputs, generate
from scripts.qwen_drive_screen import audit_rows, report, validate_prepared, configuration
from scripts.report_answer_opsd import JUDGE_PROTOCOL
from scripts.report_preference import summarize


def fixture_rows(count=8):
    return [dict(question_id=str(i), segment_id="seg%d" % i,
                 source_subset="action" if i % 2 else "scenery",
                 frames=["/images/%d/%d.jpg" % (i, j) for j in range(5)],
                 question="Why are you slowing down?", answers=["SECRET_REFERENCE_%d" % i])
            for i in range(count)]


def predictions(rows, correct, unfinished=()):
    return [dict(question_id=r["question_id"], segment_id=r["segment_id"],
                 source_subset=r["source_subset"], question=r["question"], references=r["answers"],
                 prediction="The car is slowing for a pedestrian.", finished=i not in unfinished,
                 lingo_judge_correct=i in correct, token_f1=float(i in correct),
                 normalized_exact=0., observation={"source_frames": 5}) for i, r in enumerate(rows)]


def save_scores(folder, rows):
    write_rows(folder / "predictions.jsonl", rows)
    write_rows(folder / "scored/scored_predictions.jsonl", rows)
    write_json(folder / "scored/summary.json", dict(summarize(rows), judge_protocol=JUDGE_PROTOCOL))


class TeacherTests(unittest.TestCase):
    def test_official_processor_gets_only_five_ordered_images_and_question(self):
        import torch
        seen = {}

        class Processor:
            def encode_vqa(self, frames, question, device):
                seen.update(frames=frames, question=question, device=device)
                return {"input_ids": torch.tensor([[3, 9, 9, 7]])}

        row = fixture_rows(1)[0]
        model = SimpleNamespace(processor=Processor(), vlm=SimpleNamespace(device="cpu"),
                                config=SimpleNamespace(vlm_config=SimpleNamespace(image_token_id=9)))
        value = build_inputs(model, row["frames"], row["question"])
        self.assertEqual(seen["frames"], row["frames"])
        self.assertNotIn(row["answers"][0], seen["question"])
        self.assertIn("earliest to latest", seen["question"])
        self.assertEqual(value["mm_token_type_ids"].tolist(), [[0, 1, 1, 0]])

    def test_eos_thinking_and_truncation_are_not_misclassified(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "<think>analysis</think>Car is slowing.")
        result = completion_result([10, 99], {99}, tokenizer)
        self.assertEqual(result["prediction"], "Car is slowing.")
        self.assertTrue(result["finished"])
        self.assertFalse(completion_result([10, 11], {99}, tokenizer)["finished"])
        tokenizer.decode = lambda ids, **kwargs: "<think>unfinished"
        result = completion_result([10, 99], {99}, tokenizer)
        self.assertFalse(result["finished"])
        self.assertEqual(result["prediction"], "")
        self.assertTrue(result["unclosed_thinking"])

    def test_invalid_data_and_changed_frame_order_fail(self):
        rows = fixture_rows()
        validate_data(rows)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_data(rows + [rows[0]])
        rows[0]["frames"] = rows[0]["frames"][:4]
        with self.assertRaisesRegex(ValueError, "five"):
            validate_data(rows)
        rows = fixture_rows(2)
        rows[1]["segment_id"] = rows[0]["segment_id"]
        rows[1]["frames"] = list(reversed(rows[0]["frames"]))
        with self.assertRaisesRegex(ValueError, "order"):
            validate_data(rows)

    def test_resume_rejects_duplicates_foreign_ids_or_changed_metadata(self):
        rows = fixture_rows()
        pred = predictions(rows, {0, 1})
        self.assertEqual(validate_partial(rows, pred[:2]), {"0", "1"})
        for changed in ([pred[0], pred[0]], [dict(pred[0], question_id="foreign")],
                        [dict(pred[0], references=["wrong"])],
                        [dict(pred[0], source_subset="evaluation")]):
            with self.assertRaises(ValueError):
                validate_partial(rows, changed)

    def test_manifest_prevents_mixing_models_or_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "manifest.json"
            immutable_json(p, {"max_pixels": 921600})
            immutable_json(p, {"max_pixels": 921600})
            with self.assertRaisesRegex(ValueError, "identity changed"):
                immutable_json(p, {"max_pixels": 460800})

    def test_completed_resume_preserves_memory_evidence_without_reloading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rows = fixture_rows(1)
            for i in range(5):
                frame = out / ("%d.jpg" % i)
                frame.write_bytes(b"fixture")
                rows[0]["frames"][i] = str(frame)
            data = out / "data.jsonl"
            write_rows(data, rows)
            args = SimpleNamespace(data=data, output_dir=out / "generation",
                                   model=out / "model", max_new_tokens=512, max_pixels=921600)
            model_class = SimpleNamespace(from_pretrained=lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("stop before model loading")))
            with patch("scripts.generate_qwen_drive_teacher.load_api",
                       return_value=(SimpleNamespace(bfloat16="bf16"), model_class, {"test": True})), \
                    patch("scripts.generate_qwen_drive_teacher.model_identity", return_value={"test": True}), \
                    patch("scripts.generate_qwen_drive_teacher.runtime", return_value={}), patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "stop before model loading"):
                    generate(args)
                write_rows(args.output_dir / "predictions.jsonl", predictions(rows, {0}))
                write_json(args.output_dir / "generation_summary.json", {"peak_allocated_gib": 12.3})
                generate(args)
            summary = read(args.output_dir / "generation_summary.json")
            self.assertEqual(summary["peak_allocated_gib"], 12.3)
            self.assertEqual(summary["complete_answers"], 1)
            self.assertEqual(summary["stage"], "completed")
            self.assertIn("predictions_sha256", summary)

    def test_scores_cannot_belong_to_another_generation(self):
        rows = fixture_rows()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pred = predictions(rows, {0, 1})
            save_scores(folder, pred)
            self.assertEqual(len(checked_scores(folder, rows)), len(rows))
            scored = folder / "scored/scored_predictions.jsonl"
            changed = [dict(r) for r in pred]
            changed[0]["prediction"] = "different answer"
            scored.write_text("\n".join(json.dumps(r) for r in changed) + "\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                checked_scores(folder, rows)

    def test_duplicate_scores_and_wrong_judge_protocol_fail(self):
        rows = fixture_rows()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pred = predictions(rows, {0})
            save_scores(folder, pred)
            write_json(folder / "scored/summary.json", dict(summarize(pred), judge_protocol="other"))
            with self.assertRaisesRegex(ValueError, "official"):
                checked_scores(folder, rows)
            write_json(folder / "scored/summary.json", dict(summarize(pred), judge_protocol=JUDGE_PROTOCOL))
            path = folder / "scored/scored_predictions.jsonl"
            with path.open("a") as stream:
                stream.write(json.dumps(pred[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicated"):
                checked_scores(folder, rows)

    def test_usable_comparison_masks_unfinished_teacher_corrections(self):
        rows = fixture_rows(4)
        student = predictions(rows, {0})
        teacher = predictions(rows, {0, 1, 2}, unfinished={2})
        result = paired_comparison(student, teacher)
        self.assertEqual(result["raw"]["lingo_judge_correct"]["fixed"], 2)
        self.assertEqual(result["usable"]["lingo_judge_correct"]["fixed"], 1)

    def test_audit_queue_preserves_source_metadata_and_has_both_directions(self):
        rows = fixture_rows()
        chosen = audit_rows(rows, predictions(rows, {0, 1, 2, 3}),
                            predictions(rows, {0, 2, 4, 5}), limit=8)
        self.assertEqual(len({r["question_id"] for r in chosen}), 8)
        self.assertTrue(any(r["diagnostic_group"].endswith("/repair") for r in chosen))
        self.assertTrue(any(r["diagnostic_group"].endswith("/regression") for r in chosen))
        self.assertEqual({r["question_id"]: r["answers"] for r in rows},
                         {r["question_id"]: r["answers"] for r in chosen})

    def test_positive_dev_scores_do_not_certify_unseen_teacher_or_trigger_training(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rows = fixture_rows()
            args = SimpleNamespace(output_dir=out, model=out / "model",
                                   max_new_tokens=512, max_pixels=921600, action="report")
            write_rows(out / "data/dev.jsonl", rows)
            write_json(out / "experiment.json", {"fixture": True})
            write_json(out / "data/manifest.json",
                       {"external_teacher_training_overlap": {"unseen_teacher_generalization_verified": False}})
            for role, correct in (("parent", {0}), ("retained_ce", {0, 1}), ("teacher", set(range(8)))):
                folder = out / "dev" / role
                save_scores(folder, predictions(rows, correct))
                write_json(folder / "run_manifest.json",
                           dict(protocol=PROTOCOL, data_sha256=file_sha256(out / "data/dev.jsonl"),
                                references_in_prompt=False, max_new_tokens=512,
                                max_pixels_per_frame=921600, task_instruction=INSTRUCTION,
                                model={"path": str(args.model)}))
            # Only HTML rendering is stubbed; score/report contracts use real files.
            with patch("scripts.qwen_drive_screen.child") as runner, patch("builtins.print"):
                result = report(args, "dev")
            self.assertTrue(result["observed_advantage_over_retained_ce"])
            self.assertFalse(result["teacher_qualified_for_distillation"])
            self.assertFalse(result["automatic_training"])
            self.assertEqual(set(result["groups"]), {"action", "scenery"})
            self.assertEqual(runner.call_args.args[1], "dev_review")

    def test_prepared_student_artifact_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = SimpleNamespace(output_dir=out, model=out / "model", source=out / "source",
                                   teacher_python="/teacher/python", data_dir=out / "data",
                                   dev_size=600, seed=44, max_new_tokens=512, max_pixels=921600)
            write_json(out / "experiment.json", configuration(args))
            artifact = out / "baseline.json"
            write_json(artifact, {"answer": "original"})
            write_json(out / "data/manifest.json", {"sources": {}, "splits": {},
                       "baseline_copies_sha256": {str(artifact): file_sha256(artifact)}})
            validate_prepared(args)
            write_json(artifact, {"answer": "changed"})
            with self.assertRaisesRegex(ValueError, "student copy"):
                validate_prepared(args)

    def test_status_from_arbitrary_directory_does_not_need_teacher_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run(
                [sys.executable, str(ROOT / "scripts/qwen_drive_screen.py"), "status",
                 "--output-dir", directory, "--teacher-python", "/missing/python"],
                cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("not_started", process.stdout)
            self.assertIn("report validated=False", process.stdout)


if __name__ == "__main__":
    unittest.main()
