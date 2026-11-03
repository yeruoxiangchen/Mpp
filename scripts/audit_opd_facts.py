#!/usr/bin/env python3
"""Review fixed train-only facts, then score annotated alternatives on exact prefixes.

Annotations are Codex visual review, not independent human gold labels. Selection
does not use teacher probabilities. No optimizer, training or evaluation-label edits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from scripts.generate_answers import write_json_atomic

INDICES = [0, 1, 11, 13, 23, 27, 33, 41, 45, 50, 53, 57, 62, 80, 151,
           152, 166, 180, 193, 197, 201, 203, 208, 212, 225, 250, 258,
           264, 270, 275, 285, 293]
DEFAULT = ROOT / "outputs/opd_fact_audit_20261019"
SOURCE = ROOT / "outputs/feedback_opd_diagnostics_v1"


def read(path):
    return json.loads(Path(path).read_text())


def prepare(folder):
    from PIL import Image, ImageDraw, ImageFont
    from mpp_video_qa.opd_diagnostics import assert_split_isolation
    source = load_jsonl(SOURCE / "cases.jsonl")
    rows = [dict(source[i], source_index=i) for i in INDICES]
    original = {r["question_id"]: r for r in load_jsonl(
        ROOT / "datasets/processed/lingoqa_feedback_opd_40k_v2/train.jsonl")}
    for r in rows:
        if any(r[k] != v for k, v in original[r["question_id"]].items()):
            raise ValueError("original training QA changed")
    data = ROOT / "datasets/processed/lingoqa_feedback_opd_40k_v2"
    isolation = assert_split_isolation(rows, {s: load_jsonl(data / (s + ".jsonl"))
                                            for s in ("dev_action", "dev_scenery", "eval")})
    folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol="reference_only_fact_audit_v1", indices=INDICES,
                    selection="Purposive text-based fact contrasts: 20 historical errors and 12 Judge-positive controls; not a population estimate.",
                    source_sha256=file_sha256(SOURCE / "cases.jsonl"),
                    original_manifest_sha256=file_sha256(SOURCE / "manifest.json"),
                    isolation=isolation,
                    frame_sha256={p: file_sha256(p) for r in rows for p in r["frames"]})
    if (folder / "manifest.json").exists() and read(folder / "manifest.json") != manifest:
        raise ValueError("audit inputs changed")
    write_json_atomic(folder / "manifest.json", manifest)
    write_json_atomic(folder / "cases.json", rows)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    for start in range(0, len(rows), 4):
        canvas = Image.new("RGB", (1600, 4 * 310), "white")
        draw = ImageDraw.Draw(canvas)
        for j, r in enumerate(rows[start:start+4]):
            y = j * 310
            label = "#%d [%s] %s\nREF: %s\nSTUDENT: %s" % (
                r["source_index"], r["cohort"], r["question"], " / ".join(r["answers"]), r["student_answer"])
            lines = []
            for line in label.splitlines():
                lines.extend(textwrap.wrap(line, 165))
            draw.multiline_text((8, y+2), "\n".join(lines), font=font, fill="black", spacing=2)
            for t, path in enumerate(r["frames"]):
                im = Image.open(path).convert("RGB")
                im.thumbnail((320, 180))
                canvas.paste(im, (t*320, y+125))
        canvas.save(folder / ("review_%02d.jpg" % (start//4+1)), quality=95)
    print(json.dumps(dict(prepared=len(rows), folder=str(folder)), ensure_ascii=False))


def run(folder):
    import torch
    from scripts.feedback_opd_common import (add_runtime_arguments, configure_model, runtime_identity,
                                            training_dataset, build_student, Judge)
    from mpp_video_qa.opd import completion_logits, sample_completions
    from mpp_video_qa.feedback_opd import feedback_prompt, decode_completion
    from scripts.train_sft import seed_all
    p = argparse.ArgumentParser()
    add_runtime_arguments(p)
    args = configure_model(p.parse_args(["--data", str(ROOT / "datasets/processed/lingoqa_feedback_opd_40k_v2/train.jsonl"),
                                        "--output-dir", str(folder), "--teacher-mode", "reference_only"]), False)
    identity = runtime_identity(args)
    expected = read(SOURCE / "manifest.json")["runtime_identity"]
    if identity != expected:
        raise ValueError("audit model/runtime differs from original teacher")
    rows = read(folder / "cases.json")
    annotations = read(folder / "annotations.json")
    if set(annotations) != {str(r["source_index"]) for r in rows}:
        raise ValueError("annotate every selected case, including uncertainties")
    manifest = read(folder / "manifest.json")
    if file_sha256(SOURCE / "cases.jsonl") != manifest["source_sha256"]:
        raise ValueError("source trajectories changed")
    for path, sha in manifest["frame_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError("reviewed frame changed")
    config = dict(identity=identity, annotation_sha256=file_sha256(folder / "annotations.json"),
                  manifest_sha256=file_sha256(folder / "manifest.json"))
    config_path = folder / "run_config.json"
    if config_path.exists() and read(config_path) != config:
        raise ValueError("annotations/runtime changed after scoring started")
    write_json_atomic(config_path, config)
    done = load_jsonl(folder / "scores.jsonl") if (folder / "scores.jsonl").exists() else []
    seen = {r["source_index"] for r in done}
    if len(seen) != len(done):
        raise ValueError("duplicate scored cases")
    torch.set_num_threads(1)
    seed_all(44)
    dataset = training_dataset(args)
    indices = {r["question_id"]: i for i, r in enumerate(dataset.rows)}
    model = build_student(args, torch.device("cuda:0"), False)
    tokenizer = model.backbone.llm_tokenizer
    judge = Judge(args)
    with torch.no_grad(), (folder / "scores.jsonl").open("a") as out:
        for r in rows:
            if r["source_index"] in seen:
                continue
            ann = annotations[str(r["source_index"])]
            features = dataset[indices[r["question_id"]]]["features"].unsqueeze(0).cuda()
            ids = torch.tensor(r["student_token_ids"], device="cuda", dtype=torch.long)
            prompt = feedback_prompt(r, r["student_answer"], model.task_instruction, "reference_only")
            plain = completion_logits(model, features, r["question"], ids, answer_only=True).float().log_softmax(-1).cpu()
            feedback = completion_logits(model, features, r["question"], ids, prompt, answer_only=True).float().log_softmax(-1).cpu()
            correction, finished = sample_completions(model, features, r["question"], args.max_new_tokens,
                                                       prompt=prompt, do_sample=False)
            correction_text = decode_completion(model, correction)
            facts = []
            for f in ann["facts"]:
                pos = f["position"]
                token = tokenizer.decode([int(ids[pos])])
                if token != f["observed_token"]:
                    raise ValueError("annotation not aligned to exact token prefix")
                alternatives = {}
                for phrase in f["candidates"]:
                    candidate_ids = tokenizer.encode(phrase, add_special_tokens=False)
                    if not candidate_ids or tokenizer.decode(candidate_ids) != phrase:
                        raise ValueError("candidate tokenization not reversible")
                    # Multi-token alternatives use the same original prefix followed
                    # by each candidate's own continuation, never the old wrong suffix.
                    prefix = ids[:pos]
                    candidate = torch.tensor(candidate_ids, device=ids.device)
                    if len(candidate_ids) == 1:
                        ps, pt = float(plain[pos, candidate_ids[0]]), float(feedback[pos, candidate_ids[0]])
                    else:
                        seq = torch.cat((prefix, candidate))
                        ps = pt = 0.
                        for privileged in (False, True):
                            lp = completion_logits(model, features, r["question"], seq,
                                                   prompt if privileged else None, answer_only=True).float().log_softmax(-1)
                            value = float(lp[pos:pos+len(candidate_ids)].gather(1, candidate[:, None]).sum())
                            if privileged: pt = value
                            else: ps = value
                    alternatives[phrase] = dict(token_ids=candidate_ids, plain_logp=ps, reference_logp=pt,
                                               logp_change=pt-ps, plain_probability=math.exp(ps),
                                               reference_probability=math.exp(pt))
                good, bad = alternatives[f["supported"]], alternatives[f["contrast"]]
                facts.append(dict(f, alternatives=alternatives,
                                  plain_margin=good["plain_logp"]-bad["plain_logp"],
                                  reference_margin=good["reference_logp"]-bad["reference_logp"],
                                  margin_change=good["logp_change"]-bad["logp_change"]))
            value = dict(source_index=r["source_index"], question_id=r["question_id"], cohort=r["cohort"],
                         annotation=ann, facts=facts, teacher_answer=correction_text,
                         teacher_finished=finished, teacher_judge=judge.score(r, correction_text))
            out.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n"); out.flush()
            seen.add(r["source_index"])
            write_json_atomic(folder / "progress.json", dict(stage="scoring", completed=len(seen), total=len(rows)))
            print(json.dumps(dict(completed=len(seen), total=len(rows), source_index=r["source_index"])), flush=True)
    report(folder)


def report(folder):
    rows = read(folder / "cases.json")
    scores = load_jsonl(folder / "scores.jsonl")
    if {r["source_index"] for r in scores} != {r["source_index"] for r in rows} or len(scores) != len(rows):
        raise ValueError("audit incomplete")
    groups = {}
    for status in ("visually_supported", "reference_only_uncertain", "reference_disputed"):
        for cohort in ("cached_error", "cached_correct_control"):
            facts = [f for r in scores if r["annotation"]["status"] == status and r["cohort"] == cohort for f in r["facts"]]
            groups[status+"/"+cohort] = dict(facts=len(facts),
                teacher_prefers_supported=sum(f["reference_margin"] > 0 for f in facts),
                margin_improved=sum(f["margin_change"] > 0 for f in facts),
                mean_margin_change=sum(f["margin_change"] for f in facts)/max(1,len(facts)))
            gated = [f for r in scores if r["annotation"]["status"] == status and r["cohort"] == cohort
                     and r["teacher_finished"] and r["teacher_judge"]["correct"] for f in r["facts"]]
            groups[status+"/"+cohort]["fresh_teacher_gate_passed"] = dict(facts=len(gated),
                teacher_prefers_supported=sum(f["reference_margin"] > 0 for f in gated),
                margin_improved=sum(f["margin_change"] > 0 for f in gated))
    result = dict(completed=True, cases=len(scores), groups=groups,
                  cases_by_review_status={s:sum(r["annotation"]["status"]==s for r in scores)
                      for s in ("visually_supported", "reference_only_uncertain", "reference_disputed")},
                  teacher_fresh_passed=sum(r["teacher_finished"] and r["teacher_judge"]["correct"] for r in scores),
                  annotations_sha256=file_sha256(folder / "annotations.json"),
                  scores_sha256=file_sha256(folder / "scores.jsonl"),
                  caution="Purposive training-side cases reviewed by Codex, not independent human gold or a population error-rate estimate. Phrase probabilities are unnormalized and length-sensitive; no task-accuracy claim.")
    write_json_atomic(folder / "report.json", result)
    write_json_atomic(folder / "progress.json", dict(stage="completed", completed=len(scores), total=len(rows)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "run", "report"))
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    a = p.parse_args()
    {"prepare": prepare, "run": run, "report": report}[a.action](a.output_dir)
