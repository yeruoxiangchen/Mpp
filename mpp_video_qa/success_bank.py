"""Immutable, train-only successful parent trajectories for OPD retention.

Text is decoded only for Judge/review when building the bank. Training receives
the exact generated token IDs, including EOS, without tokenizing that text again.
"""
import hashlib
import json
import math
from pathlib import Path

from .core import file_sha256, load_jsonl


PROTOCOL = "parent_greedy_success_bank_v1"
DATA_PROTOCOL = "original_40k_success_opd_subset_v1"
JUDGE_PROTOCOL = "official_lower_strip_maxref_logit_gt0_maxlength128_v1"
QA_FIELDS = ("question_id", "segment_id", "source_subset", "question", "frames")
MODEL_KEYS = ("initialization", "base_checkpoint_sha256", "bridge_init_sha256",
              "precision_protocol", "qformer_mode", "dtype", "task_instruction_version",
              "cache_dir", "max_txt_len", "max_new_tokens", "judge_model", "judge_device",
              "seed", "dropout")
HELD_SPLITS = ("dev", "dev_action", "dev_scenery", "eval")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def model_identity(identity):
    missing = [key for key in MODEL_KEYS if key not in identity]
    if missing:
        raise ValueError("missing success-bank model identity: %s" % missing)
    if identity["dropout"] != 0 or identity["qformer_mode"] != "full":
        raise ValueError("success bank requires the dropout-free full-QFormer SFT parent")
    return {key: identity[key] for key in MODEL_KEYS}


def tokenizer_identity_from_args(args, tokenizer=None):
    """Hash tokenizer implementation, vocabulary and special IDs; never encode answers."""
    root = Path(args.checkpoint_root) / "ckpt/Qwen7B-chat"
    required = ("qwen.tiktoken", "tokenizer_config.json", "tokenization_qwen.py", "config.json")
    if any(not (root / name).is_file() for name in required):
        raise ValueError("missing Qwen tokenizer files for success-bank validation")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(root), use_fast=False,
            trust_remote_code=True, local_files_only=True, padding_side="right")
        tokenizer.pad_token_id = tokenizer.eod_id
    vocab = sorted((repr(key), int(value)) for key, value in tokenizer.get_vocab().items())
    # Qwen get_vocab() omits special tokens: im_end_id=151645 exceeds the
    # 151643 ordinary-token entries. The LM output vocabulary is padded to
    # 151936. Validate against that actual output domain, not len(get_vocab()).
    special = sorted((str(key), int(value)) for key, value in
                     getattr(tokenizer, "special_tokens", {}).items())
    vocab_size = int(json.loads((root / "config.json").read_text())["vocab_size"])
    largest = max([value for _, value in vocab + special] +
                  [int(tokenizer.im_end_id), int(tokenizer.eod_id), int(tokenizer.im_start_id)])
    if vocab_size <= largest:
        raise ValueError("Qwen tokenizer IDs exceed the model output vocabulary")
    return {"files": {name: file_sha256(root / name) for name in required},
            "vocab_sha256": digest({"ordinary": vocab, "special": special}), "vocab_size": vocab_size,
            "eos_token_id": int(tokenizer.im_end_id), "pad_token_id": int(tokenizer.eod_id),
            "im_start_id": int(tokenizer.im_start_id)}


def validate_train_rows(rows, heldout=None):
    ids = set()
    for row in rows:
        qid = row.get("question_id")
        if (not qid or qid in ids or row.get("feedback_training_source") != "train" or
                len(row.get("frames", [])) != 5 or not row.get("answers") or
                not row.get("question", "").strip() or not row.get("segment_id") or
                any(not isinstance(a, str) or not a.strip() for a in row["answers"])):
            raise ValueError("success bank requires unique original train-only five-frame QA")
        ids.add(qid)
    if not rows:
        raise ValueError("empty success-bank training split")
    for name, other in (heldout or {}).items():
        for key in ("question_id", "segment_id", "frames"):
            values = lambda data: {v for row in data for v in
                                  (row[key] if key == "frames" else [row[key]])}
            if values(rows) & values(other):
                raise ValueError("success-bank training overlaps held-out %s %s" % (name, key))


def validate_data(data_path, rows=None):
    path = Path(data_path)
    if path.name != "train.jsonl":
        raise ValueError("success bank may only use the prepared train.jsonl")
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest.get("protocol") != DATA_PROTOCOL or not manifest.get("original_qa_preserved"):
        raise ValueError("prepare the immutable success-OPD training subset first")
    disk_rows = load_jsonl(path)
    if rows is not None and rows != disk_rows:
        raise ValueError("dataset rows differ from success-bank data file")
    rows = disk_rows
    parent_path = Path(manifest["parent_train_path"])
    if file_sha256(parent_path) != manifest.get("parent_train_sha256"):
        raise ValueError("success-OPD parent 40k source fingerprint differs")
    parent = {row["question_id"]: row for row in load_jsonl(parent_path)}
    if any(parent.get(row["question_id"]) != row for row in rows):
        raise ValueError("success-OPD training row is not an unchanged parent-pool QA")
    names = ("train", *HELD_SPLITS)
    for name in names:
        if file_sha256(path.parent / (name + ".jsonl")) != manifest["split_sha256"].get(name):
            raise ValueError("prepared success-OPD split changed: %s" % name)
    held = {name: load_jsonl(path.parent / (name + ".jsonl")) for name in HELD_SPLITS}
    validate_train_rows(rows, held)
    if len(rows) != manifest.get("records"):
        raise ValueError("prepared success-OPD record count differs")
    return rows, manifest


def bank_identity(args, identity, tokenizer=None):
    _, data = validate_data(args.data)
    if file_sha256(args.data) != identity.get("data_sha256"):
        raise ValueError("runtime data differs from success-bank train split")
    scorer = Path(__file__).resolve().parents[1] / "scripts/evaluate_answers.py"
    return {"protocol": PROTOCOL, "data_sha256": identity["data_sha256"],
            "model_identity": model_identity(identity),
            "tokenizer_identity": tokenizer_identity_from_args(args, tokenizer),
            "split_sha256": {name: data["split_sha256"][name] for name in ("train", *HELD_SPLITS)},
            "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
                           "reference_access": False, "prompt": "original_question_full5"},
            "judge_protocol": JUDGE_PROTOCOL, "judge_scorer_sha256": file_sha256(scorer)}


def make_record(row, token_ids, finished, prediction, judge, identity):
    record = {key: row[key] for key in QA_FIELDS}
    record.update(protocol=PROTOCOL, bank_identity_sha256=digest(identity),
                  references=row["answers"], token_ids=list(token_ids), finished=bool(finished),
                  prediction=str(prediction), judge=dict(judge),
                  eligible=bool(finished and prediction.strip() and judge["correct"] and
                                math.isfinite(judge["logit"])))
    record["record_sha256"] = digest(record)
    validate_record(record, row, identity)
    return record


def validate_record(record, original, identity):
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    if (record.get("protocol") != PROTOCOL or record.get("bank_identity_sha256") != digest(identity) or
            record.get("record_sha256") != digest(payload)):
        raise ValueError("success-bank record fingerprint/protocol differs")
    if any(record.get(key) != original.get(key) for key in QA_FIELDS) or record.get("references") != original["answers"]:
        raise ValueError("success-bank record changed original QA or frames")
    validate_train_rows([original])
    tokenizer = identity["tokenizer_identity"]
    ids = record.get("token_ids")
    if (not isinstance(ids, list) or not ids or
            len(ids) > identity["generation"]["max_new_tokens"] or
            any(type(x) is not int or x < 0 or x >= tokenizer["vocab_size"] for x in ids)):
        raise ValueError("success-bank requires exact valid generated token IDs")
    eos = tokenizer["eos_token_id"]
    if type(record.get("finished")) is not bool or record["finished"] != (ids[-1] == eos) or eos in ids[:-1]:
        raise ValueError("success-bank EOS/finished contract differs")
    judge = record.get("judge", {})
    logit = judge.get("logit")
    if (type(logit) not in (int, float) or not math.isfinite(logit) or
            type(judge.get("correct")) is not bool or judge["correct"] != (logit > 0)):
        raise ValueError("success-bank Judge score is invalid")
    if not isinstance(record.get("prediction"), str):
        raise ValueError("missing decoded prediction for success-bank audit")
    eligible = bool(record["finished"] and record["prediction"].strip() and judge["correct"])
    if type(record.get("eligible")) is not bool or record["eligible"] != eligible:
        raise ValueError("success-bank eligibility differs from completion/Judge gate")


def read_records(path, originals, identity, recover_partial=False):
    """Only a torn final append is recoverable; complete malformed/duplicate rows fail."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        if not recover_partial:
            raise ValueError("incomplete success-bank JSONL tail")
        end = raw.rfind(b"\n") + 1
        with path.open("r+b") as stream:
            stream.truncate(end)
        raw = raw[:end]
    result, seen = [], set()
    for line in raw.splitlines():
        record = json.loads(line)
        qid = record.get("question_id")
        if qid in seen or qid not in originals:
            raise ValueError("duplicate or foreign question in success-bank records")
        validate_record(record, originals[qid], identity)
        result.append(record)
        seen.add(qid)
    return result


class SuccessBank(dict):
    """A qid -> exact token-ID list mapping with an immutable artifact identity."""
    def __init__(self, records, identity):
        super().__init__((row["question_id"], list(row["token_ids"])) for row in records if row["eligible"])
        self.identity = identity


def bank_contract(directory):
    """Validate the finished artifact, without a GPU, for reports and training configs."""
    root = Path(directory)
    path = root / "bank_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("completed") is not True:
        raise ValueError("success bank is incomplete or has the wrong protocol")
    if manifest.get("identity_sha256") != digest(manifest.get("identity")):
        raise ValueError("success-bank manifest identity fingerprint differs")
    records = root / "records.jsonl"
    if manifest.get("records_sha256") != file_sha256(records):
        raise ValueError("success-bank records fingerprint differs")
    return {"protocol": PROTOCOL, "manifest_sha256": file_sha256(path),
            "records_sha256": manifest["records_sha256"], "data_sha256": manifest["data_sha256"],
            "identity_sha256": manifest["identity_sha256"], "records": manifest["records"],
            "eligible": manifest["eligible"]}


def load_success_bank(args, identity, dataset):
    directory = getattr(args, "success_bank", "")
    if not directory:
        return SuccessBank([], {})
    contract = bank_contract(directory)
    manifest = json.loads((Path(directory) / "bank_manifest.json").read_text())
    rows, _ = validate_data(args.data, dataset.rows)
    expected = bank_identity(args, identity)
    if manifest["identity"] != expected or manifest["data_sha256"] != identity["data_sha256"]:
        raise ValueError("success-bank model/data/tokenizer/runtime identity differs")
    originals = {row["question_id"]: row for row in rows}
    records = read_records(Path(directory) / "records.jsonl", originals, expected)
    if (len(records) != len(rows) or {row["question_id"] for row in records} != set(originals) or
            manifest["records"] != len(records) or manifest["eligible"] != sum(row["eligible"] for row in records)):
        raise ValueError("success bank does not cover every training QA exactly once")
    return SuccessBank(records, contract)
