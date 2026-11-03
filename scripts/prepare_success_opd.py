#!/usr/bin/env python3
"""Select an immutable source-stratified pilot from the existing 40k training pool."""
import os
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.data import cache_file
from mpp_video_qa.success_bank import DATA_PROTOCOL, HELD_SPLITS, validate_train_rows, write_json
from scripts.prepare_visual_adaptation import stratified_subset, write_rows


def prepare(args):
    source, output = Path(args.source_dir), Path(args.output_dir)
    if source.resolve() == output.resolve():
        raise ValueError("success pilot needs a separate output directory")
    train = load_jsonl(source / "train.jsonl")
    held = {name: load_jsonl(source / (name + ".jsonl")) for name in HELD_SPLITS}
    source_manifest = json.loads((source / "manifest.json").read_text())
    for name in ("train", *HELD_SPLITS):
        if source_manifest.get("split_sha256", {}).get(name) != file_sha256(source / (name + ".jsonl")):
            raise ValueError("original 40k source split fingerprint differs: %s" % name)
    if source_manifest.get("records") != len(train):
        raise ValueError("source training count differs from manifest")
    validate_train_rows(train, held)
    if ({r["question_id"]: r for r in held["dev"]} !=
            {r["question_id"]: r for r in held["dev_action"] + held["dev_scenery"]}):
        raise ValueError("development subsets differ from the combined development split")
    if args.probe_records <= 0:
        raise ValueError("probe-records must be positive")
    cap = max(Counter(row["segment_id"] for row in train).values())
    chosen = stratified_subset(train, args.records, args.seed, cap)
    probe = stratified_subset(chosen, min(args.probe_records, len(chosen)), args.seed + 1, cap)
    missing = [row["segment_id"] for row in chosen
               if not cache_file(args.cache_dir, row["segment_id"]).is_file()]
    if missing:
        raise ValueError("missing original EVA cache: %s" % missing[0])
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "train.jsonl", chosen)
    write_rows(output / "probe.jsonl", probe)
    for name in HELD_SPLITS:
        src, dst = source / (name + ".jsonl"), output / (name + ".jsonl")
        data = src.read_bytes()
        if dst.exists() and dst.read_bytes() != data:
            raise ValueError("held-out output differs; use a new pilot directory")
        if not dst.exists():
            temporary = dst.with_suffix(".jsonl.tmp")
            temporary.write_bytes(data)
            temporary.replace(dst)
    counts = Counter(row["segment_id"] for row in chosen)
    manifest = {"protocol": DATA_PROTOCOL, "records": len(chosen), "probe_records": len(probe),
        "seed": args.seed, "segments": len(counts),
        "sources": dict(Counter(row["source_subset"] for row in chosen)),
        "sampling": "source_stratified_video_round_robin_v2",
        "actual_max_qa_per_segment": max(counts.values()),
        "qa_per_segment_distribution": {str(k): v for k, v in sorted(Counter(counts.values()).items())},
        "source_dir": str(source.resolve()), "source_records": len(train),
        "parent_train_path": str((source / "train.jsonl").resolve()),
        "parent_train_sha256": file_sha256(source / "train.jsonl"),
        "source_train_sha256": file_sha256(source / "train.jsonl"),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "split_sha256": {name: file_sha256(output / (name + ".jsonl"))
                         for name in ("train", "probe", *HELD_SPLITS)},
        "original_qa_preserved": True, "held_out_overlap": 0,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "probe_scope": "subset of train only; no dev/eval labels used",
        "pilot": {"epochs": 2, "effective_global_batch_size": 16,
                  "optimizer_steps": (len(chosen) + 15) // 16 * 2}}
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("existing success pilot manifest differs; use a new directory")
    write_json(path, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(ROOT / "datasets/processed/lingoqa_feedback_opd_40k_v2"))
    parser.add_argument("--output-dir", default=str(ROOT / "datasets/processed/lingoqa_success_opd_4k_v1"))
    parser.add_argument("--cache-dir", default=os.environ.get("MPP_EVA_CACHE_DIR", str(ROOT / "datasets/cache/lingoqa_5f_eva_v1")))
    parser.add_argument("--records", "--train-size", dest="records", type=int, default=4000)
    parser.add_argument("--probe-records", type=int, default=200)
    parser.add_argument("--seed", type=int, default=44)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
