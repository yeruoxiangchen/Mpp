#!/usr/bin/env python3
"""Resumable official Judge scoring with explicit pre-truncation token lengths."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import best_reference_scores, file_sha256, load_jsonl
from mpp_video_qa.driving_teacher import exclusive_lock, immutable_json, write_json
from scripts.report_answer_opsd import JUDGE_PROTOCOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir
    rows = load_jsonl(args.predictions)
    expected = {r["question_id"]: r for r in rows}
    if not rows or len(expected) != len(rows):
        raise ValueError("nonempty unique prediction IDs are required")
    identity = {"predictions_sha256": file_sha256(args.predictions), "judge_protocol": JUDGE_PROTOCOL,
                "judge_model": "wayveai/Lingo-Judge", "code_sha256": {
                    p: file_sha256(ROOT / p) for p in (
                        "scripts/score_transfer_pilot.py", "scripts/evaluate_answers.py")}}
    with exclusive_lock(folder / ".score.lock"):
        immutable_json(folder / "input_identity.json", identity)
        destination = folder / "scored_predictions.jsonl"
        done = load_jsonl(destination) if destination.exists() else []
        seen = set()
        for row in done:
            key = row["question_id"]
            if key not in expected or key in seen or any(row.get(k) != v for k, v in expected[key].items()):
                raise ValueError("resumed scoring is duplicated/foreign/stale")
            if (type(row.get("lingo_judge_correct")) is not bool or
                    not math.isfinite(row["lingo_judge_logit"]) or
                    row["lingo_judge_correct"] != (row["lingo_judge_logit"] > 0) or
                    len(row.get("judge_input_tokens", [])) != len(row["references"])):
                raise ValueError("invalid resumed scoring")
            seen.add(key)

        def progress(stage):
            write_json(folder / "score_progress.json", dict(status=stage, completed=len(done),
                       total=len(rows), updated_unix_seconds=time.time()))

        if len(done) < len(rows):
            progress("loading_judge")
            import torch
            from scripts.evaluate_answers import load_judge, judge_score
            tokenizer, model = load_judge("wayveai/Lingo-Judge", torch.device("cuda"))
            with destination.open("a", encoding="utf-8") as stream:
                for row in rows:
                    if row["question_id"] in seen:
                        continue
                    logit, correct = judge_score(tokenizer, model, row["question"], row["references"], row["prediction"])
                    texts = ["%s\nQuestion: %s\nAnswer: %s\nStudent: %s" % (
                        tokenizer.cls_token, row["question"], str(ref).lower().strip(),
                        str(row["prediction"]).lower().strip()) for ref in row["references"]]
                    lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
                    exact, f1 = best_reference_scores(row["prediction"], row["references"])
                    value = dict(row, lingo_judge_logit=logit, lingo_judge_correct=correct,
                                 normalized_exact=exact, token_f1=f1, judge_input_tokens=lengths)
                    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    done.append(value)
                    if len(done) % 20 == 0:
                        progress("scoring")
        from scripts.report_preference import summarize
        summary = dict(summarize(done), judge_protocol=JUDGE_PROTOCOL)
        write_json(folder / "summary.json", summary)
        progress("completed")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
