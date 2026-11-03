#!/usr/bin/env python3
"""Generate disjoint greedy-parent shards, then merge a complete train-only bank."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256
from mpp_video_qa.success_bank import (PROTOCOL, bank_identity, digest, make_record,
    read_records, tokenizer_identity_from_args, validate_data, write_json, bank_contract)
from scripts.feedback_opd_common import (add_runtime_arguments, configure_model,
    runtime_identity, training_dataset, build_student, Judge)


def shard_dir(output, index, count):
    return Path(output) / ("shard-%05d-of-%05d" % (index, count))


def shard_config(identity, index, count, rows):
    return {"protocol": PROTOCOL, "identity": identity, "identity_sha256": digest(identity),
            "shard_index": index, "num_shards": count,
            "question_ids": [row["question_id"] for row in rows]}


def generate(args, identity, rows):
    import torch
    from mpp_video_qa.opd import sample_completions
    from scripts.train_sft import seed_all, gpu_memory
    assigned = rows[args.shard_index::args.num_shards]
    folder = shard_dir(args.output_dir, args.shard_index, args.num_shards)
    folder.mkdir(parents=True, exist_ok=True)
    config = shard_config(identity, args.shard_index, args.num_shards, assigned)
    config_path = folder / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("success-bank shard configuration changed")
    write_json(config_path, config)
    originals = {row["question_id"]: row for row in assigned}
    record_path = folder / "records.jsonl"
    manifest_path = folder / "manifest.json"
    old = read_records(record_path, originals, identity, recover_partial=not manifest_path.exists())
    done = {row["question_id"] for row in old}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("completed") is not True or manifest.get("config") != config or
                manifest.get("records_sha256") != file_sha256(record_path) or
                len(old) != len(assigned) or manifest.get("records") != len(old) or
                manifest.get("eligible") != sum(row["eligible"] for row in old)):
            raise ValueError("completed success-bank shard changed")
        print(json.dumps({"status": "already_complete", "shard": args.shard_index, "records": len(old)}))
        return
    started = time.time()
    device = torch.device("cuda", 0)
    if len(done) < len(assigned):
        torch.set_num_threads(args.cpu_threads)
        seed_all(args.seed)
        write_json(folder / "progress.json", {"status": "loading_model", "completed": len(done),
                   "total": len(assigned), "pid": os.getpid(), "updated_unix_seconds": time.time()})
        model = build_student(args, device, train=False)
        if tokenizer_identity_from_args(args, model.backbone.llm_tokenizer) != identity["tokenizer_identity"]:
            raise ValueError("loaded parent tokenizer differs from bank tokenizer")
        judge, dataset = Judge(args), training_dataset(args)
        indices = {row["question_id"]: i for i, row in enumerate(dataset.rows)}
        with record_path.open("a", encoding="utf-8") as stream:
            for original in assigned:
                qid = original["question_id"]
                if qid in done:
                    continue
                record = dataset[indices[qid]]
                features = record["features"].unsqueeze(0).to(device)
                # No reference/draft prompt: only the original question and five frames.
                ids, finished = sample_completions(model, features, original["question"],
                    max_new_tokens=args.max_new_tokens, do_sample=False)
                exact_ids = ids.detach().cpu().tolist()
                prediction = model.backbone.llm_tokenizer.decode(exact_ids, skip_special_tokens=True).strip()
                score = judge.score(original, prediction)
                result = make_record(original, exact_ids, finished, prediction, score, identity)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                old.append(result)
                done.add(qid)
                progress = {"status": "generating", "completed": len(done), "total": len(assigned),
                    "eligible": sum(row["eligible"] for row in old), "pid": os.getpid(),
                    "elapsed_seconds": time.time() - started, "updated_unix_seconds": time.time()}
                write_json(folder / "progress.json", progress)
                if len(done) == 1 or len(done) % 20 == 0 or len(done) == len(assigned):
                    print(json.dumps(progress), flush=True)
                del features, ids
        write_json(folder / "memory.json", gpu_memory(device))
    elif not record_path.exists():
        record_path.touch()
    # Re-read durable records; all questions, including failures, remain in the bank.
    records = read_records(record_path, originals, identity)
    if len(records) != len(assigned):
        raise ValueError("success-bank shard is incomplete")
    manifest = {"protocol": PROTOCOL, "completed": True, "config": config,
                "records": len(records), "eligible": sum(row["eligible"] for row in records),
                "records_sha256": file_sha256(record_path)}
    write_json(manifest_path, manifest)
    write_json(folder / "progress.json", {"status": "completed", "completed": len(records),
               "total": len(assigned), "eligible": manifest["eligible"], "updated_unix_seconds": time.time()})
    print(json.dumps({"status": "completed", "shard": args.shard_index,
        "records": manifest["records"], "eligible": manifest["eligible"],
        "records_sha256": manifest["records_sha256"]}), flush=True)


def merge(args, identity, rows):
    output = Path(args.output_dir)
    collected, hashes = {}, {}
    for index in range(args.num_shards):
        folder = shard_dir(output, index, args.num_shards)
        assigned = rows[index::args.num_shards]
        expected = shard_config(identity, index, args.num_shards, assigned)
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("success-bank shard is incomplete: %d" % index)
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("protocol") != PROTOCOL or manifest.get("completed") is not True or
                manifest.get("config") != expected or
                json.loads((folder / "run_config.json").read_text()) != expected or
                manifest.get("records_sha256") != file_sha256(folder / "records.jsonl")):
            raise ValueError("success-bank shard identity or record fingerprint differs")
        records = read_records(folder / "records.jsonl", {r["question_id"]: r for r in assigned}, identity)
        if (len(records) != len(assigned) or manifest.get("records") != len(records) or
                manifest.get("eligible") != sum(row["eligible"] for row in records)):
            raise ValueError("success-bank shard does not cover its assigned questions")
        for row in records:
            if row["question_id"] in collected:
                raise ValueError("duplicate question across success-bank shards")
            collected[row["question_id"]] = row
        hashes[str(index)] = file_sha256(manifest_path)
    ordered = [collected[row["question_id"]] for row in rows]
    output.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in ordered)
    records_path = output / "records.jsonl"
    if records_path.exists() and records_path.read_text() != content:
        raise ValueError("existing merged bank differs; use a new output directory")
    if not records_path.exists():
        temporary = records_path.with_suffix(".jsonl.tmp")
        temporary.write_text(content)
        temporary.replace(records_path)
    manifest = {"protocol": PROTOCOL, "completed": True, "identity": identity,
        "identity_sha256": digest(identity), "data_sha256": identity["data_sha256"],
        "records_sha256": file_sha256(records_path), "records": len(ordered),
        "eligible": sum(row["eligible"] for row in ordered), "num_shards": args.num_shards,
        "shard_manifest_sha256": hashes,
        "interpretation": "Judge-positive completed parent responses; not independent visual fact labels."}
    path = output / "bank_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("existing success-bank manifest differs")
    write_json(path, manifest)
    print(json.dumps(bank_contract(output), indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "merge"))
    add_runtime_arguments(parser)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = configure_model(parser.parse_args(), train=False)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards > 0 and 0 <= shard-index < num-shards")
    if args.max_new_tokens < 1 or args.language_lora_scope != "qkv":
        parser.error("success bank requires positive generation budget and the parent QKV adapter")
    rows, _ = validate_data(args.data)
    identity = bank_identity(args, runtime_identity(args))
    if args.command == "generate":
        generate(args, identity, rows)
    else:
        merge(args, identity, rows)


if __name__ == "__main__":
    main()
