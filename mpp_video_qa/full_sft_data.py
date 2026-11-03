"""Read-only provenance checks for the existing complete original LingoQA split."""
from collections import Counter
import json
from pathlib import Path

from .core import file_sha256, load_jsonl


PROTOCOL = "all_original_lingoqa_train_v1"


def validate_full_training(data_path, initial_dir, rows=None):
    data_path, initial_dir = Path(data_path), Path(initial_dir)
    root = data_path.parent
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parent = json.loads((initial_dir / "run_config.json").read_text())
    actual_hash = file_sha256(data_path)
    if (data_path.name != "train.jsonl" or manifest.get("sampling") != "all_original_training_rows" or
            manifest.get("split_sha256", {}).get("train") != actual_hash or
            manifest.get("source_train_sha256") != actual_hash or parent.get("data_sha256") != actual_hash):
        raise ValueError("full SFT requires the unchanged complete original split used by its SFT parent")
    rows = load_jsonl(data_path) if rows is None else rows
    ids, segments, frames, sources = set(), set(), set(), Counter()
    per_segment = Counter()
    for row in rows:
        qid = row["question_id"]
        if (not qid or qid in ids or len(row["frames"]) != 5 or
                not isinstance(row["question"], str) or not row["question"].strip() or
                not row["answers"] or not str(row["answers"][0]).strip() or
                row["source_subset"] not in ("action", "scenery")):
            raise ValueError("full SFT requires unique original QA, five frames and nonempty reference answers")
        ids.add(qid)
        segments.add(row["segment_id"])
        frames.update(row["frames"])
        sources[row["source_subset"]] += 1
        per_segment[row["segment_id"]] += 1
    if (not rows or len(rows) != manifest.get("records") or len(segments) != manifest.get("segments") or
            dict(sources) != manifest.get("source_counts")):
        raise ValueError("complete original QA counts differ from the manifest")
    held, hashes = {}, {"train": actual_hash}
    for split in ("dev", "dev_action", "dev_scenery", "eval"):
        path = root / (split + ".jsonl")
        hashes[split] = file_sha256(path)
        if hashes[split] != manifest["split_sha256"].get(split):
            raise ValueError("held-out split differs from its original manifest: %s" % split)
        held[split] = load_jsonl(path)
        if (ids & {r["question_id"] for r in held[split]} or
                segments & {r["segment_id"] for r in held[split]} or
                frames & {p for r in held[split] for p in r["frames"]}):
            raise ValueError("original train and held-out question/video/frame overlap")
        held_ids = [r["question_id"] for r in held[split]]
        if not held_ids or len(set(held_ids)) != len(held_ids):
            raise ValueError("empty/duplicate held-out question IDs")
    combined = held["dev_action"] + held["dev_scenery"]
    if (len(combined) != len(held["dev"]) or
            {r["question_id"]: r for r in combined} != {r["question_id"]: r for r in held["dev"]} or
            {r["segment_id"] for r in held["dev"]} & {r["segment_id"] for r in held["eval"]}):
        raise ValueError("development subsets or development/official video separation differ")
    return {"protocol": PROTOCOL, "manifest_sha256": file_sha256(manifest_path),
            "records": len(rows), "segments": len(segments), "source_counts": dict(sources),
            "split_sha256": hashes, "original_qa_preserved": True, "held_out_overlap": 0,
            "qa_per_segment_distribution": {str(k): v for k, v in sorted(Counter(per_segment.values()).items())}}
