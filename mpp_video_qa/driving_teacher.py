"""Contracts for screening an external driving VQA teacher, without training."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess

from mpp_video_qa.core import TASK_INSTRUCTIONS, file_sha256, load_jsonl

MODEL_ID = "Qwen/Qwen-Drive-1.0-4B"
MODEL_REVISION = "28484089a7cc8c335cf5089fb0745cf7c49b6eaa"
SOURCE_URL = "https://github.com/QwenLM/Qwen-Drive-1.0.git"
SOURCE_REVISION = "28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0"
PROTOCOL = "qwen_drive_five_frame_screen_v1"
INSTRUCTION = TASK_INSTRUCTIONS["driving_perspective_v1"]
ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "chat_template.jinja", "chat_template.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "model.safetensors", "model.safetensors.index.json", "model-*.safetensors",
    "README.md",
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError("identity changed; use a new output directory: %s" % path)
    else:
        write_json(path, value)


def write_rows(path, rows):
    path = Path(path)
    if path.exists():
        if load_jsonl(path) != rows:
            raise ValueError("existing rows differ: %s" % path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another writer owns this experiment: %s" % path)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_data(rows, check_images=False):
    if not rows or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("data must be nonempty with unique question IDs")
    segment_frames = {}
    for row in rows:
        if (not isinstance(row["question"], str) or not row["question"].strip()
                or not row["answers"] or not all(isinstance(x, str) and x.strip() for x in row["answers"])
                or not row["segment_id"] or not row["source_subset"]):
            raise ValueError("invalid question, references or segment")
        frames = row["frames"]
        if len(frames) != 5 or len(set(frames)) != 5:
            raise ValueError("exactly five distinct chronological source frames are required")
        if row["segment_id"] in segment_frames and segment_frames[row["segment_id"]] != frames:
            raise ValueError("frame order differs within one video")
        segment_frames[row["segment_id"]] = frames
    if check_images:
        missing = [p for frames in segment_frames.values() for p in frames if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError("missing image: %s (%d missing)" % (missing[0], len(missing)))


def frame_identity(rows):
    """Content hashes of the actual five-frame observations; no model weights read."""
    paths = sorted({str(Path(p).resolve()) for r in rows for p in r["frames"]})
    files = {p: file_sha256(p) for p in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"unique_files": len(paths), "content_sha256": digest}


def source_identity(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("Qwen-Drive checkout must be pinned to %s" % SOURCE_REVISION)
    files = sorted((source / "src/qwen_drive").glob("*.py"))
    if not files or not (source / "src/qwen_drive/modeling_qwen_drive.py").exists():
        raise ValueError("Qwen-Drive source missing; run setup")
    changed = subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--", "src/qwen_drive"], text=True)
    if changed:
        raise ValueError("upstream Qwen-Drive source has local modifications")
    return {"revision": revision, "path": str(source),
            "sha256": {str(p.relative_to(source)): file_sha256(p) for p in files}}


def model_identity(model):
    model = Path(model).resolve()
    config = read(model / "config.json")
    if config.get("model_type") != "qwen_drive":
        raise ValueError("expected the released qwen_drive checkpoint, not a generic Qwen model")
    weights = sorted(model.glob("*.safetensors"))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError("missing/empty teacher weights; run download")
    index = model / "model.safetensors.index.json"
    if index.exists():
        required = set(read(index)["weight_map"].values())
        if not required <= {p.name for p in weights}:
            raise ValueError("incomplete sharded teacher checkpoint")
    metadata = [p for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".jinja", ".txt")]
    return {"path": str(model), "metadata_sha256": {p.name: file_sha256(p) for p in sorted(metadata)},
            "weight_stat_identity": [{"file": p.name, "bytes": p.stat().st_size,
                                      "mtime_ns": p.stat().st_mtime_ns} for p in weights],
            "weight_identity_note": "file size/mtime, not a fresh content hash of all model weights"}


def teacher_question(question):
    # The function deliberately accepts only the question, never the full row/references.
    return INSTRUCTION + "\nThere are five frames, ordered from earliest to latest.\nQuestion: " + question


def completion_result(token_ids, eos_ids, tokenizer):
    """Expose only the answer after a closed thinking block to the text Judge."""
    finished = any(t in eos_ids for t in token_ids)
    content = []
    for token in token_ids:
        if token in eos_ids:
            break
        content.append(token)
    raw = tokenizer.decode(content, skip_special_tokens=True).strip()
    if "</think>" in raw:
        answer = raw.rsplit("</think>", 1)[1].strip()
        unclosed_thinking = False
    elif "<think>" in raw:
        answer, unclosed_thinking = "", True
    else:
        answer, unclosed_thinking = raw, False
    return {"prediction": answer, "raw_prediction": raw,
            "finished": finished and not unclosed_thinking and bool(answer),
            "eos_seen": finished, "unclosed_thinking": unclosed_thinking,
            "completion_tokens": len(token_ids)}


def validate_partial(rows, predictions):
    expected = {r["question_id"]: r for r in rows}
    done = set()
    for result in predictions:
        key = result["question_id"]
        if key in done or key not in expected:
            raise ValueError("duplicate or foreign resumed prediction")
        original = expected[key]
        for out_key, in_key in (("question", "question"), ("segment_id", "segment_id"),
                                ("references", "answers"), ("source_subset", "source_subset")):
            if result[out_key] != original[in_key]:
                raise ValueError("resumed prediction metadata differs from source")
        if not isinstance(result["finished"], bool) or not isinstance(result["prediction"], str):
            raise ValueError("missing generation completion contract")
        done.add(key)
    return done


def checked_scores(folder, originals):
    """Bind scores to generated text, source metadata, and the official protocol."""
    from mpp_video_qa.distillation import validate_predictions
    from scripts.report_preference import summarize
    from scripts.report_answer_opsd import JUDGE_PROTOCOL
    folder = Path(folder)
    generated = load_jsonl(folder / "predictions.jsonl")
    scores = load_jsonl(folder / "scored/scored_predictions.jsonl")
    by_id = validate_predictions(originals, generated)
    validate_predictions(originals, scores)
    for row in scores:
        if any(row.get(k) != by_id[row["question_id"]].get(k)
               for k in ("prediction", "finished", "source_subset")):
            raise ValueError("scores are stale or metadata differs")
        if type(row.get("lingo_judge_correct")) is not bool:
            raise ValueError("invalid Judge correctness flag")
        if any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k])
               or not 0 <= row[k] <= 1 for k in ("token_f1", "normalized_exact")):
            raise ValueError("invalid exact/F1 score")
    summary = read(folder / "scored/summary.json")
    if summary.get("judge_protocol") != JUDGE_PROTOCOL:
        raise ValueError("official Lingo-Judge scoring is required")
    if any(not isinstance(summary.get(k), (int, float)) or not math.isfinite(summary[k])
           or abs(summary[k] - v) > 1e-9 for k, v in summarize(scores).items()):
        raise ValueError("score summary differs from rows")
    return scores


def paired_comparison(student, teacher):
    from scripts.report_answer_opsd import paired_metrics
    from scripts.report_preference import cluster_interval
    usable = [dict(r, lingo_judge_correct=bool(r["lingo_judge_correct"]
                   and r.get("finished") is True and r["prediction"].strip())) for r in teacher]
    return {"raw": dict(paired_metrics(student, teacher), **cluster_interval(student, teacher)),
            "usable": dict(paired_metrics(student, usable), **cluster_interval(student, usable))}
