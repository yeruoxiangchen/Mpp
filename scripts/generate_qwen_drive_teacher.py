#!/usr/bin/env python3
"""Pinned official Qwen-Drive VQA processor + instrumented token generation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.driving_teacher import (
    MODEL_ID, MODEL_REVISION, MODEL_FILES, PROTOCOL, INSTRUCTION, read, write_json,
    immutable_json, exclusive_lock, validate_data, frame_identity, source_identity,
    model_identity, teacher_question, completion_result, validate_partial,
)
from scripts.qwen_drive_parallel import PARTITION, shard_rows


def runtime():
    versions = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "safetensors",
                 "numpy", "pillow", "flash-linear-attention", "causal-conv1d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "packages": versions}


def load_api(args):
    source = source_identity(args.source)
    sys.path.insert(0, str(args.source / "src"))
    import torch
    from qwen_drive import QwenDriveForPlanning
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback. Run on one available GPU.")
    return torch, QwenDriveForPlanning, source


def download(args):
    from huggingface_hub import snapshot_download
    args.model.mkdir(parents=True, exist_ok=True)
    stamp = {"repo_id": MODEL_ID, "revision": MODEL_REVISION, "allow_patterns": MODEL_FILES}
    with exclusive_lock(args.model / ".download.lock"):
        manifest = args.model / "download_manifest.json"
        if manifest.exists() and read(manifest) != stamp:
            raise ValueError("model directory belongs to another download")
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(args.model),
                          allow_patterns=MODEL_FILES, max_workers=4)
        immutable_json(manifest, stamp)
        print(json.dumps(model_identity(args.model), indent=2), flush=True)


def doctor(args):
    torch, model_class, source = load_api(args)
    identity = model_identity(args.model)
    from transformers import AutoTokenizer
    from qwen_drive import QwenDriveConfig, QwenDriveProcessor
    config = QwenDriveConfig.from_pretrained(str(args.model), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    processor = QwenDriveProcessor(tokenizer, config)
    # No weights or CUDA model are loaded during this preflight.
    result = {"runtime": runtime(), "source_revision": source["revision"],
              "model": identity, "cuda_device": torch.cuda.get_device_name(0),
              "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "vocab_size": config.vlm_config.text_config.vocab_size,
              "image_patch_factor": processor.factor, "weights_loaded": False}
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_inputs(model, frames, question):
    """Use the upstream VQA processor (not the planning scene API)."""
    inputs = model.processor.encode_vqa(frames, teacher_question(question), device=model.vlm.device)
    inputs["mm_token_type_ids"] = (inputs["input_ids"] == model.config.vlm_config.image_token_id).long()
    return inputs


def generate(args):
    torch, model_class, upstream = load_api(args)
    rows = load_jsonl(args.data)
    count, index = getattr(args, "num_shards", 1), getattr(args, "shard_index", 0)
    rows = shard_rows(rows, index, count)
    validate_data(rows, check_images=True)
    if args.max_new_tokens < 1 or args.max_pixels < 4096:
        raise ValueError("invalid generation budget")
    identity = {
        "protocol": PROTOCOL, "backend": "qwen_drive_official_vqa",
        "model": model_identity(args.model), "source": upstream, "runtime": runtime(),
        "data_sha256": file_sha256(args.data), "frames": frame_identity(rows),
        "source_frames": 5, "input_mode": "chronological_multi_image",
        "references_in_prompt": False, "ego_history_in_prompt": False,
        "task_instruction": INSTRUCTION, "prompt_version": "driving_teacher_question_v1",
        "max_new_tokens": args.max_new_tokens, "max_pixels_per_frame": args.max_pixels,
        "decoding": "greedy", "dtype": "bfloat16", "attention": "sdpa", "seed": 3407,
        "implementation_sha256": {
            p: file_sha256(ROOT / p) for p in (
                "scripts/generate_qwen_drive_teacher.py", "mpp_video_qa/driving_teacher.py",
                "scripts/qwen_drive_parallel.py")},
    }
    if count > 1:
        identity["shard"] = {"index": index, "count": count, "partition": PARTITION, "records": len(rows)}
    out = args.output_dir
    with exclusive_lock(out / ".generation.lock"):
        predictions = out / "predictions.jsonl"
        if predictions.exists() and not (out / "run_manifest.json").exists():
            raise ValueError("predictions without an identity manifest")
        immutable_json(out / "run_manifest.json", identity)
        existing = load_jsonl(predictions) if predictions.exists() else []
        done = validate_partial(rows, existing)
        started, latencies = time.monotonic(), []

        def progress(stage, **extra):
            speed = sum(latencies[-20:]) / len(latencies[-20:]) if latencies else None
            value = dict(stage=stage, completed=len(done), total=len(rows), pid=os.getpid(),
                         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                         updated_unix_seconds=time.time(), elapsed_seconds=time.monotonic() - started,
                         recent_seconds_per_question=speed,
                         estimated_remaining_seconds=(len(rows) - len(done)) * speed if speed else None,
                         **extra)
            write_json(out / "progress.json", value)
            print(json.dumps(value), flush=True)
            return value

        try:
            if len(done) == len(rows):
                summary_path = out / "generation_summary.json"
                summary = read(summary_path) if summary_path.exists() else {}
                summary.update(progress("completed"),
                               predictions_sha256=file_sha256(predictions),
                               complete_answers=sum(r["finished"] for r in existing))
                # Re-entering screen rechecks smoke; retain its recorded GPU peak.
                write_json(summary_path, summary)
                return
            progress("loading_teacher")
            # The wrapper initializes an unused planner. Remove it before moving the
            # VLM to CUDA; no planner weights are downloaded or used for this task.
            model, loading = model_class.from_pretrained(
                str(args.model), local_files_only=True, dtype=torch.bfloat16,
                attn_implementation="sdpa", output_loading_info=True)
            missing = [k for k in loading.get("missing_keys", []) if not k.startswith("planning_expert.")]
            if missing or loading.get("unexpected_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise ValueError("incomplete VLM checkpoint load: %s" % loading)
            del model.planning_expert
            model.config.current_image_pixels = args.max_pixels
            model.vlm.to("cuda:0").eval()
            model.vlm.set_attn_implementation("sdpa")
            from transformers import AutoTokenizer
            from qwen_drive import QwenDriveProcessor
            model.processor = QwenDriveProcessor(
                AutoTokenizer.from_pretrained(str(args.model), local_files_only=True), model.config)
            eos = model.vlm.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            eos_ids.update([model.processor.im_end_id, model.config.vlm_config.text_config.eos_token_id])
            eos_ids.discard(None)
            if not eos_ids:
                raise ValueError("teacher EOS IDs missing")
            torch.manual_seed(3407)
            torch.cuda.reset_peak_memory_stats()
            from PIL import Image
            with predictions.open("a", encoding="utf-8") as stream:
                for row in rows:
                    if row["question_id"] in done:
                        continue
                    tick = time.monotonic()
                    images = []
                    try:
                        for path in row["frames"]:
                            with Image.open(path) as original:
                                images.append(original.convert("RGB"))
                        inputs = build_inputs(model, images, row["question"])
                        prompt_length = inputs["input_ids"].shape[1]
                        context_limit = model.config.vlm_config.text_config.max_position_embeddings
                        if prompt_length + args.max_new_tokens > context_limit:
                            raise ValueError("input exceeds context budget; no silent frame/text truncation")
                        with torch.inference_mode():
                            generated = model.vlm.generate(
                                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                num_beams=1, repetition_penalty=1.0, use_cache=True,
                                eos_token_id=sorted(eos_ids), pad_token_id=model.processor.im_end_id)
                        completion = generated[0, prompt_length:].tolist()
                        result = completion_result(completion, eos_ids, model.processor.tokenizer)
                        grid = inputs["image_grid_thw"]
                        result.update(
                            question_id=row["question_id"], segment_id=row["segment_id"],
                            source_subset=row["source_subset"], question=row["question"],
                            references=row["answers"], view="full", teacher_context="view",
                            observation={"source_frames": 5, "grid_thw": grid.tolist(),
                                         "visual_tokens": int(grid.prod(-1).sum()) // model.processor.merge_size**2,
                                         "prompt_tokens": prompt_length, "resampled": False},
                            latency_seconds=time.monotonic() - tick)
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        done.add(row["question_id"])
                        latencies.append(time.monotonic() - tick)
                        progress("generating", question_id=row["question_id"],
                                 peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
                    finally:
                        for picture in images:
                            picture.close()
            summary = progress("completed", peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            summary.update(predictions_sha256=file_sha256(predictions),
                           complete_answers=sum(r["finished"] for r in load_jsonl(predictions)))
            write_json(out / "generation_summary.json", summary)
        except BaseException as error:
            progress("failed", error=repr(error))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("download", "doctor", "generate"))
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints/Qwen-Drive-1.0-4B")
    p.add_argument("--source", type=Path, default=ROOT / "third_party/Qwen-Drive-1.0")
    p.add_argument("--data", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-pixels", type=int, default=921600)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()
    args.model, args.source = args.model.resolve(), args.source.resolve()
    if args.action == "generate" and (args.data is None or args.output_dir is None):
        p.error("generate requires --data and --output-dir")
    {"download": download, "doctor": doctor, "generate": generate}[args.action](args)


if __name__ == "__main__":
    main()
