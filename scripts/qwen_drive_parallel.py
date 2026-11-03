"""Deterministic QA sharding, strict merging, and file progress for teacher screening."""
from pathlib import Path

from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.driving_teacher import (
    checked_scores, exclusive_lock, immutable_json, read, validate_partial,
    write_json, write_rows,
)

PARTITION = "row_index_mod_shard_count"


def parse_devices(value):
    devices = [v.strip() for v in value.split(",")]
    if (not devices or any(not v or not v.isdigit() for v in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError("--gpus must contain distinct nonnegative GPU indices, e.g. 1,2,3,5")
    return devices


def shard_rows(rows, index, count):
    if count < 1 or not 0 <= index < count or count > len(rows):
        raise ValueError("invalid shard index/count or empty shard")
    return rows[index::count]


def shard_folder(folder, index):
    return Path(folder) / "shards" / ("shard-%03d" % index)


def layout(data, folder, count):
    rows = load_jsonl(data)
    for i in range(count):
        shard_rows(rows, i, count)
    value = {"data_sha256": file_sha256(data), "records": len(rows),
             "shard_count": count, "partition": PARTITION}
    immutable_json(Path(folder) / "sharding.json", value)
    return rows


def score_stamp(folder):
    from scripts.report_answer_opsd import JUDGE_PROTOCOL
    return {"predictions_sha256": file_sha256(Path(folder) / "predictions.jsonl"),
            "judge_protocol": JUDGE_PROTOCOL,
            "scorer_sha256": file_sha256(Path(__file__).parent / "evaluate_answers.py")}


def score_is_ready(folder, originals):
    stamp = Path(folder) / "scored/input_identity.json"
    if stamp.exists() and read(stamp) == score_stamp(folder):
        checked_scores(folder, originals)
        return True
    return False


def merge_generation(data, folder, count):
    """Require every shard, exact per-shard IDs and matching inference settings."""
    folder = Path(folder)
    originals = layout(data, folder, count)
    data_sha = file_sha256(data)
    common = None
    predictions, manifests, sources, frames, peaks = {}, {}, {}, {}, {}
    for i in range(count):
        part = shard_folder(folder, i)
        selected = shard_rows(originals, i, count)
        manifest_path = part / "run_manifest.json"
        manifest = read(manifest_path)
        expected = {"index": i, "count": count, "partition": PARTITION, "records": len(selected)}
        if manifest.get("data_sha256") != data_sha or manifest.get("shard") != expected:
            raise ValueError("wrong shard assignment or source data: %s" % part)
        core = {k: v for k, v in manifest.items() if k not in ("shard", "frames")}
        if common is None:
            common = core
        elif core != common:
            raise ValueError("teacher model/runtime/generation settings differ across shards")
        pred_path = part / "predictions.jsonl"
        values = load_jsonl(pred_path)
        if len(validate_partial(selected, values)) != len(selected):
            raise ValueError("incomplete shard predictions")
        summary = read(part / "generation_summary.json")
        if (summary.get("stage") != "completed" or summary.get("completed") != len(selected)
                or summary.get("total") != len(selected)
                or summary.get("predictions_sha256") != file_sha256(pred_path)):
            raise ValueError("shard generation summary is incomplete/stale")
        for row in values:
            if row["question_id"] in predictions:
                raise ValueError("duplicate question across shards")
            predictions[row["question_id"]] = row
        manifests[str(manifest_path)] = file_sha256(manifest_path)
        sources[str(pred_path)] = file_sha256(pred_path)
        frames[str(i)] = manifest["frames"]
        peaks[str(i)] = summary.get("peak_allocated_gib")
    merged = [predictions[r["question_id"]] for r in originals]
    validate_partial(originals, merged)
    identity = dict(common, frames={"identity_type": "per_shard_content_fingerprints", "shards": frames},
                    aggregation={"partition": PARTITION, "shard_count": count,
                                 "manifests_sha256": manifests, "predictions_sha256": sources})
    with exclusive_lock(folder / ".generation.lock"):
        immutable_json(folder / "run_manifest.json", identity)
        write_rows(folder / "predictions.jsonl", merged)
        value = {"stage": "completed", "completed": len(merged), "total": len(originals),
                 "shard_count": count, "estimated_remaining_seconds": 0,
                 "complete_answers": sum(r["finished"] for r in merged),
                 "predictions_sha256": file_sha256(folder / "predictions.jsonl"),
                 "peak_allocated_gib_by_shard": peaks}
        write_json(folder / "generation_summary.json", value)
        write_json(folder / "progress.json", value)
    return merged


def merge_scores(data, folder, count):
    from scripts.report_answer_opsd import JUDGE_PROTOCOL
    from scripts.report_preference import summarize
    folder = Path(folder)
    originals = layout(data, folder, count)
    results, sources = {}, {}
    for i in range(count):
        part = shard_folder(folder, i)
        selected = shard_rows(originals, i, count)
        if not score_is_ready(part, selected):
            raise ValueError("shard scoring identity is missing/stale")
        rows = checked_scores(part, selected)
        for row in rows:
            if row["question_id"] in results:
                raise ValueError("duplicate scored question across shards")
            results[row["question_id"]] = row
        for name in ("scored/scored_predictions.jsonl", "scored/summary.json", "scored/input_identity.json"):
            path = part / name
            sources[str(path)] = file_sha256(path)
    rows = [results[r["question_id"]] for r in originals]
    # Validate scores against the merged generation before committing a summary.
    generated = {r["question_id"]: r for r in load_jsonl(folder / "predictions.jsonl")}
    for row in rows:
        if any(row.get(k) != generated[row["question_id"]].get(k)
               for k in ("prediction", "references", "segment_id", "question", "source_subset", "finished")):
            raise ValueError("shard scores differ from merged generation")
    write_rows(folder / "scored/scored_predictions.jsonl", rows)
    write_json(folder / "scored/summary.json", dict(summarize(rows), judge_protocol=JUDGE_PROTOCOL))
    write_json(folder / "scored/input_identity.json", dict(score_stamp(folder), shards_sha256=sources))
    write_json(folder / "scored/score_progress.json",
               {"status": "completed", "completed": len(rows), "total": len(rows)})
    checked_scores(folder, originals)
    return sources


def progress_snapshot(folder, total, devices):
    from scripts.report_answer_opsd import line_count
    folder = Path(folder)
    shards = []
    for i, gpu in enumerate(devices):
        part = shard_folder(folder, i)
        progress = read(part / "progress.json") if (part / "progress.json").exists() else {}
        scored = read(part / "scored/score_progress.json") if (part / "scored/score_progress.json").exists() else {}
        expected = len(range(i, total, len(devices)))
        shards.append({"index": i, "gpu": gpu, "total": expected,
                       "generated": line_count(part / "predictions.jsonl"),
                       "scored": scored.get("completed", 0), "stage": progress.get("stage", "not_started"),
                       "estimated_remaining_seconds": progress.get("estimated_remaining_seconds")})
    pending = [s for s in shards if s["generated"] < s["total"]]
    eta = (max(s["estimated_remaining_seconds"] for s in pending)
           if pending and all(s["estimated_remaining_seconds"] is not None for s in pending)
           else (None if pending else 0))
    return {"generated": sum(s["generated"] for s in shards), "total": total,
            "scored": sum(s["scored"] for s in shards),
            "estimated_remaining_seconds": eta, "shards": shards}
