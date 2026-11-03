import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl, file_sha256, load_jsonl
from mpp_video_qa.data import cache_file
from mpp_video_qa.success_bank import (PROTOCOL, MODEL_KEYS, HELD_SPLITS, bank_contract,
    bank_identity, digest, load_success_bank, make_record, read_records, validate_data,
    validate_train_rows, write_json, tokenizer_identity_from_args)
from scripts.prepare_success_opd import prepare
from scripts.build_success_bank import generate, merge, shard_dir, shard_config


def qa(name, source="action", segment=None):
    return {"question_id": name, "segment_id": segment or name, "source_subset": source,
        "question": "What is visible?", "answers": ["A vehicle is ahead."],
        "frames": ["/%s/%d.jpg" % (segment or name, i) for i in range(5)],
        "feedback_training_source": "train"}


TOKENIZER = {"files": {"qwen.tiktoken": "vocab", "tokenizer_config.json": "config",
                        "tokenization_qwen.py": "implementation"},
             "vocab_sha256": "vocabulary", "vocab_size": 128,
             "eos_token_id": 2, "pad_token_id": 0, "im_start_id": 1}


class SuccessBankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.cache, self.output = [self.root / name for name in ("source", "cache", "pilot")]
        train = [qa("a%d" % i, segment="av%d" % (i // 2)) for i in range(8)]
        train += [qa("s%d" % i, "scenery", "sv%d" % (i // 2)) for i in range(4)]
        action, scenery, evaluation = [qa("dev_a")], [qa("dev_s", "scenery")], [qa("eval")]
        splits = {"train": train, "dev": action + scenery, "dev_action": action,
                  "dev_scenery": scenery, "eval": evaluation}
        for name, rows in splits.items():
            dump_jsonl(self.source / (name + ".jsonl"), rows)
        write_json(self.source / "manifest.json", {"records": len(train),
            "split_sha256": {name: file_sha256(self.source / (name + ".jsonl")) for name in splits}})
        for row in train:
            file = cache_file(self.cache, row["segment_id"])
            file.parent.mkdir(parents=True, exist_ok=True)
            file.touch()
        self.prepare_args = types.SimpleNamespace(source_dir=self.source, output_dir=self.output,
            records=6, probe_records=2, seed=44, cache_dir=self.cache)
        with patch("builtins.print"):
            prepare(self.prepare_args)
        self.rows = load_jsonl(self.output / "train.jsonl")
        self.args = types.SimpleNamespace(data=str(self.output / "train.jsonl"),
            output_dir=str(self.root / "bank"), success_bank=str(self.root / "bank"),
            checkpoint_root=str(self.root / "checkpoint"), max_new_tokens=8,
            shard_index=0, num_shards=2, seed=44, cpu_threads=1)
        self.runtime = {key: "identity-" + key for key in MODEL_KEYS}
        self.runtime.update(initialization={"adapter": "adapter", "projector": "projector", "qformer": "qformer"},
            qformer_mode="full", dropout=0, max_new_tokens=8, seed=44,
            data_sha256=file_sha256(self.args.data))
        self.token_patch = patch("mpp_video_qa.success_bank.tokenizer_identity_from_args", return_value=TOKENIZER)
        self.token_patch.start()
        self.addCleanup(self.token_patch.stop)
        self.identity = bank_identity(self.args, self.runtime)
        self.dataset = types.SimpleNamespace(rows=self.rows)

    def build_shards(self):
        for i in range(self.args.num_shards):
            assigned = self.rows[i::self.args.num_shards]
            folder = shard_dir(self.args.output_dir, i, self.args.num_shards)
            config = shard_config(self.identity, i, self.args.num_shards, assigned)
            write_json(folder / "run_config.json", config)
            records = [make_record(row, [17, 19, 2] if j % 2 == 0 else [21, 22], j % 2 == 0,
                        "decoded words", {"correct": True, "logit": 1.2}, self.identity)
                       for j, row in enumerate(assigned)]
            dump_jsonl(folder / "records.jsonl", records)
            write_json(folder / "manifest.json", {"protocol": PROTOCOL, "completed": True,
                "config": config, "records": len(records), "eligible": sum(r["eligible"] for r in records),
                "records_sha256": file_sha256(folder / "records.jsonl")})

    def finish_bank(self):
        self.build_shards()
        with patch("builtins.print"):
            return merge(self.args, self.identity, self.rows)

    def test_prepare_is_deterministic_stratified_and_preserves_heldout_bytes(self):
        initial = file_sha256(self.args.data)
        with patch("builtins.print"):
            manifest = prepare(self.prepare_args)
        self.assertEqual(initial, file_sha256(self.args.data))
        self.assertEqual(manifest["sources"], {"action": 4, "scenery": 2})
        self.assertEqual(manifest["segments"], 6)
        originals = {r["question_id"]: r for r in load_jsonl(self.source / "train.jsonl")}
        self.assertTrue(all(originals[r["question_id"]] == r for r in self.rows))
        for name in HELD_SPLITS:
            self.assertEqual((self.source / (name + ".jsonl")).read_bytes(),
                             (self.output / (name + ".jsonl")).read_bytes())
        self.assertEqual(manifest["parent_train_sha256"], file_sha256(self.source / "train.jsonl"))

    def test_qwen_special_eos_outside_get_vocab_uses_model_output_size(self):
        root = Path(self.args.checkpoint_root) / "ckpt/Qwen7B-chat"
        root.mkdir(parents=True)
        for name in ("qwen.tiktoken", "tokenizer_config.json", "tokenization_qwen.py"):
            (root / name).write_text("test")
        write_json(root / "config.json", {"vocab_size": 16})
        tokenizer = types.SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1},
            special_tokens={"end": 9}, im_end_id=9, im_start_id=8, eod_id=7)
        identity = tokenizer_identity_from_args(self.args, tokenizer)
        self.assertEqual(identity["vocab_size"], 16)
        self.assertEqual(identity["eos_token_id"], 9)

    def test_heldout_question_segment_and_frame_overlap_rejected(self):
        for key in ("question_id", "segment_id", "frames"):
            other = qa("heldout")
            other[key] = self.rows[0][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "overlaps held-out"):
                validate_train_rows(self.rows, {"dev": [other]})
        bad = dict(self.rows[0], feedback_training_source="dev")
        with self.assertRaisesRegex(ValueError, "train-only"):
            validate_train_rows([bad])

    def test_loader_returns_original_tokens_including_eos_only_for_eligible(self):
        self.finish_bank()
        bank = load_success_bank(self.args, self.runtime, self.dataset)
        self.assertEqual(len(bank), 4)
        self.assertTrue(all(ids == [17, 19, 2] for ids in bank.values()))
        self.assertEqual(bank.identity, bank_contract(self.args.success_bank))
        self.assertEqual(bank.identity["records"], 6)

    def test_wrong_model_data_and_tokenizer_rejected(self):
        self.finish_bank()
        wrong = copy.deepcopy(self.runtime)
        wrong["initialization"]["adapter"] = "other adapter"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            load_success_bank(self.args, wrong, self.dataset)
        wrong = dict(self.runtime, data_sha256="other data")
        with self.assertRaisesRegex(ValueError, "runtime data differs"):
            load_success_bank(self.args, wrong, self.dataset)
        with patch("mpp_video_qa.success_bank.tokenizer_identity_from_args", return_value=dict(TOKENIZER, eos_token_id=3)):
            with self.assertRaisesRegex(ValueError, "identity differs"):
                load_success_bank(self.args, self.runtime, self.dataset)

    def test_tampered_tokens_and_complete_file_fingerprints_rejected(self):
        self.finish_bank()
        path = Path(self.args.success_bank) / "records.jsonl"
        records = load_jsonl(path)
        records[0]["token_ids"][0] = 42
        dump_jsonl(path, records)
        with self.assertRaisesRegex(ValueError, "records fingerprint"):
            load_success_bank(self.args, self.runtime, self.dataset)
        # Even an updated outer file hash cannot hide a changed per-record payload.
        manifest_path = path.parent / "bank_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["records_sha256"] = file_sha256(path)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "record fingerprint"):
            load_success_bank(self.args, self.runtime, self.dataset)

    def test_incomplete_manifest_and_missing_record_rejected(self):
        self.finish_bank()
        path = Path(self.args.success_bank) / "bank_manifest.json"
        manifest = json.loads(path.read_text())
        write_json(path, dict(manifest, completed=False))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_success_bank(self.args, self.runtime, self.dataset)
        records_path = path.parent / "records.jsonl"
        dump_jsonl(records_path, load_jsonl(records_path)[:-1])
        manifest["records_sha256"] = file_sha256(records_path)
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "cover every"):
            load_success_bank(self.args, self.runtime, self.dataset)

    def test_record_requires_complete_eos_and_judge_positive(self):
        for finished, ids, correct, expected in (
                (True, [17, 2], True, True), (True, [17, 2], False, False),
                (False, [17], True, False)):
            result = make_record(self.rows[0], ids, finished, "vehicle", {"correct": correct,
                "logit": 1.0 if correct else -1.0}, self.identity)
            self.assertEqual(result["eligible"], expected)
        with self.assertRaisesRegex(ValueError, "EOS"):
            make_record(self.rows[0], [17], True, "vehicle", {"correct": True, "logit": 1.}, self.identity)
        with self.assertRaisesRegex(ValueError, "token IDs"):
            make_record(self.rows[0], [True, 2], True, "vehicle", {"correct": True, "logit": 1.}, self.identity)

    def test_resume_rejects_duplicate_and_recovers_only_torn_tail(self):
        record = make_record(self.rows[0], [17, 19, 2], True, "vehicle",
            {"correct": True, "logit": 1.}, self.identity)
        path = self.root / "resume.jsonl"
        dump_jsonl(path, [record])
        with path.open("ab") as stream:
            stream.write(b'{"question_id":')
        originals = {r["question_id"]: r for r in self.rows}
        with self.assertRaisesRegex(ValueError, "tail"):
            read_records(path, originals, self.identity)
        self.assertEqual(read_records(path, originals, self.identity, recover_partial=True), [record])
        dump_jsonl(path, [record, record])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_records(path, originals, self.identity, recover_partial=True)

    def test_merge_requires_every_completed_shard_and_matching_identity(self):
        self.build_shards()
        path = shard_dir(self.args.output_dir, 1, 2) / "manifest.json"
        manifest = json.loads(path.read_text())
        write_json(path, dict(manifest, completed=False))
        with self.assertRaisesRegex(ValueError, "identity"):
            merge(self.args, self.identity, self.rows)
        manifest["config"]["identity"]["generation"]["reference_access"] = True
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "identity"):
            merge(self.args, self.identity, self.rows)

    def test_loaded_dataset_and_parent_qa_tampering_rejected(self):
        self.finish_bank()
        bad = copy.deepcopy(self.rows)
        bad[0]["answers"] = ["Changed answer"]
        with self.assertRaisesRegex(ValueError, "dataset rows differ"):
            load_success_bank(self.args, self.runtime, types.SimpleNamespace(rows=bad))
        dump_jsonl(self.source / "train.jsonl", bad)
        with self.assertRaisesRegex(ValueError, "parent 40k source"):
            validate_data(self.args.data)

    def test_generation_resume_keeps_actual_ids_and_never_reencodes_text(self):
        assigned = self.rows[::2]
        folder = shard_dir(self.args.output_dir, 0, 2)
        first = make_record(assigned[0], [17, 19, 2], True, "decoded words",
            {"correct": True, "logit": 1.}, self.identity)
        dump_jsonl(folder / "records.jsonl", [first])
        class Features:
            def unsqueeze(self, _):
                return self
            def to(self, _):
                return self
        class Dataset:
            rows = self.rows
            def __getitem__(self, index):
                return {"features": Features()}
        class Tokenizer:
            def __call__(self, *_args, **_kwargs):
                raise AssertionError("must never re-encode generated text")
            def decode(self, ids, **kwargs):
                self.ids = ids
                return "decoded words"
        tokenizer = Tokenizer()
        model = types.SimpleNamespace(backbone=types.SimpleNamespace(llm_tokenizer=tokenizer))
        judge = types.SimpleNamespace(score=lambda row, text: {"correct": True, "logit": 1.})
        with patch("scripts.build_success_bank.build_student", return_value=model), \
                patch("scripts.build_success_bank.Judge", return_value=judge), \
                patch("scripts.build_success_bank.training_dataset", return_value=Dataset()), \
                patch("scripts.build_success_bank.tokenizer_identity_from_args", return_value=TOKENIZER), \
                patch("scripts.train_sft.seed_all"), patch("scripts.train_sft.gpu_memory", return_value={}), \
                patch("mpp_video_qa.opd.sample_completions", return_value=(torch.tensor([17, 19, 2]), True)) as sample, \
                patch("builtins.print"):
            generate(self.args, self.identity, self.rows)
            self.assertEqual(sample.call_count, len(assigned) - 1)
            self.assertTrue(all(call.kwargs["do_sample"] is False and "prompt" not in call.kwargs
                                for call in sample.call_args_list))
            generate(self.args, self.identity, self.rows)
            self.assertEqual(sample.call_count, len(assigned) - 1)
        records = load_jsonl(folder / "records.jsonl")
        self.assertEqual(len(records), len(assigned))
        self.assertTrue(all(r["token_ids"] == [17, 19, 2] for r in records))


if __name__ == "__main__":
    unittest.main()
