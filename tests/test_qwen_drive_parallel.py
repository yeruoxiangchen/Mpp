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
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.driving_teacher import checked_scores, read, write_json, write_rows
from scripts.qwen_drive_parallel import (
    PARTITION, layout, merge_generation, merge_scores, parse_devices, progress_snapshot,
    score_stamp, shard_folder, shard_rows,
)
from scripts.qwen_drive_screen import children, generate, score
from tests.test_qwen_drive_teacher import fixture_rows, predictions, save_scores


def completed_fixture(root, count=4, records=16):
    originals = fixture_rows(records)
    data, folder = root / "data.jsonl", root / "teacher"
    write_rows(data, originals)
    layout(data, folder, count)
    for i in range(count):
        selected = shard_rows(originals, i, count)
        part = shard_folder(folder, i)
        values = predictions(selected, {0, 2})
        save_scores(part, values)
        write_json(part / "run_manifest.json", {
            "data_sha256": file_sha256(data), "model": {"fixture": "same_model"},
            "max_new_tokens": 512, "frames": {"fixture_sha": "frame_shard_%d" % i},
            "shard": {"index": i, "count": count, "partition": PARTITION, "records": len(selected)},
        })
        write_json(part / "generation_summary.json", {
            "stage": "completed", "completed": len(selected), "total": len(selected),
            "predictions_sha256": file_sha256(part / "predictions.jsonl"), "peak_allocated_gib": 12.3})
        write_json(part / "scored/input_identity.json", score_stamp(part))
    return data, folder, originals


class ParallelTeacherTests(unittest.TestCase):
    def test_gpu_list_is_explicit_and_duplicates_are_rejected(self):
        self.assertEqual(parse_devices("1, 2,3,5"), ["1", "2", "3", "5"])
        for value in ("", "1,1", "1,,2", "-1", "one,two"):
            with self.assertRaises(ValueError):
                parse_devices(value)

    def test_600_questions_partition_into_four_disjoint_150s(self):
        rows = fixture_rows(600)
        parts = [shard_rows(rows, i, 4) for i in range(4)]
        self.assertEqual([len(p) for p in parts], [150] * 4)
        self.assertEqual(len({r["question_id"] for p in parts for r in p}), 600)
        self.assertEqual([r["question_id"] for r in parts[3][:3]], ["3", "7", "11"])
        with self.assertRaises(ValueError):
            shard_rows(rows[:2], 0, 4)

    def test_generation_and_score_merge_restore_original_order_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            data, folder, originals = completed_fixture(Path(directory))
            merged = merge_generation(data, folder, 4)
            self.assertEqual([r["question_id"] for r in merged],
                             [r["question_id"] for r in originals])
            merge_scores(data, folder, 4)
            self.assertEqual(len(checked_scores(folder, originals)), 16)
            summary = read(folder / "scored/summary.json")
            self.assertEqual(summary["records"], 16)
            self.assertEqual(summary["lingo_judge_accuracy"], .5)
            self.assertEqual(len(read(folder / "run_manifest.json")["aggregation"]["manifests_sha256"]), 4)

    def test_repeated_merge_is_idempotent_and_does_not_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            data, folder, originals = completed_fixture(Path(directory))
            merge_generation(data, folder, 4)
            merge_scores(data, folder, 4)
            paths = [folder / "run_manifest.json", folder / "predictions.jsonl",
                     folder / "scored/scored_predictions.jsonl", folder / "scored/input_identity.json"]
            before = [file_sha256(p) for p in paths]
            merge_generation(data, folder, 4)
            merge_scores(data, folder, 4)
            self.assertEqual(before, [file_sha256(p) for p in paths])
            self.assertEqual(len(load_jsonl(folder / "predictions.jsonl")), len(originals))

    def test_missing_or_duplicate_shard_answers_cannot_be_merged(self):
        for corruption in ("missing", "duplicate"):
            with tempfile.TemporaryDirectory() as directory:
                data, folder, _ = completed_fixture(Path(directory))
                path = shard_folder(folder, 2) / "predictions.jsonl"
                rows = load_jsonl(path)
                changed = rows[:-1] if corruption == "missing" else rows + [rows[0]]
                path.write_text("".join(json.dumps(r) + "\n" for r in changed))
                with self.assertRaises(ValueError):
                    merge_generation(data, folder, 4)
                self.assertFalse((folder / "predictions.jsonl").exists())

    def test_swapped_shard_ids_or_different_generation_budgets_fail(self):
        for corruption in ("assignment", "budget"):
            with tempfile.TemporaryDirectory() as directory:
                data, folder, _ = completed_fixture(Path(directory))
                path = shard_folder(folder, 1) / "run_manifest.json"
                manifest = read(path)
                if corruption == "assignment":
                    manifest["shard"]["index"] = 2
                else:
                    manifest["max_new_tokens"] = 256
                write_json(path, manifest)
                with self.assertRaises(ValueError):
                    merge_generation(data, folder, 4)

    def test_stale_shard_completion_or_scoring_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data, folder, _ = completed_fixture(Path(directory))
            path = shard_folder(folder, 0) / "generation_summary.json"
            summary = read(path)
            summary["predictions_sha256"] = "stale"
            write_json(path, summary)
            with self.assertRaisesRegex(ValueError, "stale"):
                merge_generation(data, folder, 4)
        with tempfile.TemporaryDirectory() as directory:
            data, folder, _ = completed_fixture(Path(directory))
            merge_generation(data, folder, 4)
            path = shard_folder(folder, 0) / "scored/input_identity.json"
            stamp = read(path)
            stamp["predictions_sha256"] = "stale"
            write_json(path, stamp)
            with self.assertRaisesRegex(ValueError, "stale"):
                merge_scores(data, folder, 4)

    def test_parallel_progress_uses_all_shards_and_slowest_remaining_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            data, folder, _ = completed_fixture(Path(directory))
            for i in range(4):
                part = shard_folder(folder, i)
                values = load_jsonl(part / "predictions.jsonl")[:2]
                (part / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
                write_json(part / "progress.json",
                           {"stage": "generating", "estimated_remaining_seconds": 10 * (i + 1)})
                write_json(part / "scored/score_progress.json", {"completed": 1})
            state = progress_snapshot(folder, 16, ["1", "2", "3", "5"])
            self.assertEqual(state["generated"], 8)
            self.assertEqual(state["scored"], 4)
            self.assertEqual(state["estimated_remaining_seconds"], 40)
            self.assertEqual([s["gpu"] for s in state["shards"]], ["1", "2", "3", "5"])

    def test_generation_commands_bind_one_gpu_and_unique_shard_per_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=root, gpus=["1", "2", "3", "5"],
                                   max_new_tokens=512, max_pixels=921600)
            write_rows(root / "data/dev.jsonl", fixture_rows(16))
            with patch("scripts.qwen_drive_screen.backend_command", return_value=["python", "generate"]), \
                    patch("scripts.qwen_drive_screen.children") as run, \
                    patch("scripts.qwen_drive_screen.merge_generation"):
                generate(args, "dev")
            jobs = run.call_args.args[2]
            self.assertEqual([j["gpu"] for j in jobs], args.gpus)
            for i, job in enumerate(jobs):
                command = job["command"]
                self.assertEqual(command[command.index("--shard-index") + 1], i)
                self.assertEqual(command[command.index("--num-shards") + 1], 4)

    def test_completed_scoring_is_reused_without_new_judge_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Fixture paths match the controller's dev split layout.
            data, folder, originals = completed_fixture(root / "dev")
            write_rows(root / "data/dev.jsonl", originals)
            merge_generation(data, folder, 4)
            args = SimpleNamespace(output_dir=root, gpus=["1", "2", "3", "5"])
            with patch("scripts.qwen_drive_screen.children") as run:
                score(args, "dev")
            self.assertEqual(run.call_args.args[2], [])
            self.assertEqual(len(checked_scores(folder, originals)), 16)

    def test_real_cpu_workers_receive_isolated_gpu_environment_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), action="test")
            jobs = [{"label": "worker%d" % i, "gpu": gpu,
                     "command": [sys.executable, "-c",
                                 "import os; print('VISIBLE=' + os.environ['CUDA_VISIBLE_DEVICES'])"]}
                    for i, gpu in enumerate(["1", "2", "3", "5"])]
            with patch("builtins.print"):
                children(args, "cpu_test", jobs)
            for i, gpu in enumerate(["1", "2", "3", "5"]):
                self.assertIn("VISIBLE=" + gpu, (args.output_dir / ("logs/worker%d.log" % i)).read_text())
            state = read(args.output_dir / "workflow.json")
            self.assertTrue(state["workers_completed"])
            self.assertEqual([r["exit_code"] for r in state["workers"]], [0] * 4)

    def test_failed_worker_stops_its_remaining_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), action="test")
            processes = []
            real_popen = subprocess.Popen

            def track(*a, **kw):
                proc = real_popen(*a, **kw)
                processes.append(proc)
                return proc

            jobs = [{"label": "bad", "gpu": "1", "command": [sys.executable, "-c", "raise SystemExit(7)"]},
                    {"label": "waiting", "gpu": "2",
                     "command": [sys.executable, "-c", "import time; time.sleep(30)"]}]
            with patch("scripts.qwen_drive_screen.subprocess.Popen", side_effect=track), patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "exit 7"):
                    children(args, "cpu_failure_test", jobs)
            self.assertEqual(len(processes), 2)
            self.assertTrue(all(p.poll() is not None for p in processes))
            self.assertNotEqual(processes[1].returncode, 0)


if __name__ == "__main__":
    unittest.main()
