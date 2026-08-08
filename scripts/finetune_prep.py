"""Prepare and upload supervised fine-tuning data, and print the job config.

  - serialize the speaker-disjoint train/val splits to SFT conversational JSONL
  - validate (check_file) and upload (purpose="fine-tune"); poll processing_status
  - print the exact fine_tuning.create(...) config (not executed here)

Uploading is free; training is launched separately by finetune_run.py --launch.
File ids are saved to ft_artifacts.json.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import os
import time

from truthclf import data, llm

SCHEME = "primary"
VARIANT = "full"
BASE_MODEL = "google/gemma-4-31B-it"
N_EPOCHS = 3
OUT_DIR = "ft_data"
ARTIFACTS = "ft_artifacts.json"
TRAIN_PATH = os.path.join(OUT_DIR, "train.jsonl")
VAL_PATH = os.path.join(OUT_DIR, "val.jsonl")


def hr(title):
    print(f"\n{'='*78}\n  {title}\n{'='*78}")


def _check_file(path):
    """Locate the SDK's check_file across versions and run it.

    Returns (report, note). `report` is None only if no known import path
    exposes check_file; `note` then explains which paths were tried and why each
    failed. Callers must surface that — this used to `except Exception: continue`
    and return None, so a validation step that never ran looked identical to one
    that passed.
    """
    attempts = []
    for modname in ("together.lib.utils.files", "together.lib.utils",
                    "together.utils"):
        try:
            mod = __import__(modname, fromlist=["check_file"])
            return mod.check_file(path), None
        except (ImportError, AttributeError) as e:
            attempts.append(f"{modname}: {type(e).__name__}")
    return None, "check_file not found (tried " + "; ".join(attempts) + ")"


def _upload(client, path):
    report, note = _check_file(path)
    if report is None:
        # Not fatal: the server re-validates on upload(check=True). But say so.
        print(f"  !! SDK-side validation SKIPPED for {path} -- {note}", flush=True)
    elif not report.get("is_check_passed", True):
        raise SystemExit(f"check_file failed for {path}: {report}")
    resp = client.files.upload(file=path, purpose="fine-tune", check=True)
    fid = resp.id
    # poll processing status until terminal
    for _ in range(40):
        f = client.files.retrieve(fid)
        st = str(getattr(f, "processing_status", getattr(f, "status", "")) or "").upper()
        if st in ("", "NONE") or "COMPLET" in st:
            break
        if "INVALID" in st or "FAIL" in st:
            raise SystemExit(f"file {fid} processing status {st}")
        time.sleep(3)
    return fid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = data.load("data.csv")
    clean, _ = data.clean_dataset(rows, SCHEME)
    train, val, test = data.speaker_disjoint_3way(clean, val_frac=0.2, test_frac=0.2,
                                                 seed=0, scheme=SCHEME)

    hr("SPLITS (speaker-disjoint, group-disjoint via union-find)")
    for name, rs in [("train", train), ("val", val), ("test (held out)", test)]:
        nt, nf, frac = data.class_balance(rs, SCHEME)
        print(f"  {name:<16} n={len(rs):>5}  True={nt} False={nf} (True {frac:.3f})")

    data.write_sft_jsonl(train, TRAIN_PATH, VARIANT, SCHEME)
    data.write_sft_jsonl(val, VAL_PATH, VARIANT, SCHEME)
    print(f"\n  wrote {TRAIN_PATH} and {VAL_PATH}")

    # reuse already-uploaded files if present
    artifacts = {}
    if os.path.exists(ARTIFACTS):
        artifacts = json.load(open(ARTIFACTS))
    client = llm.TogetherClient(BASE_MODEL)._ensure_client()
    if not artifacts.get("train_file_id") or not artifacts.get("val_file_id"):
        hr("VALIDATE + UPLOAD (free)")
        artifacts["train_file_id"] = _upload(client, TRAIN_PATH)
        artifacts["val_file_id"] = _upload(client, VAL_PATH)
        json.dump(artifacts, open(ARTIFACTS, "w"), indent=2)
        print(f"  uploaded. train_file_id={artifacts['train_file_id']}")
        print(f"            val_file_id={artifacts['val_file_id']}")
    else:
        hr("UPLOAD — reusing existing file ids from ft_artifacts.json")
        print(f"  train_file_id={artifacts['train_file_id']}")
        print(f"  val_file_id={artifacts['val_file_id']}")

    hr("JOB CONFIG (printed, not executed here)")
    cfg = {
        "model": BASE_MODEL,
        "training_file": artifacts["train_file_id"],
        "validation_file": artifacts["val_file_id"],
        "n_evals": 10,
        "lora": True,
        "n_epochs": N_EPOCHS,
        "learning_rate": 1e-5,
        "train_on_inputs": "auto",
        "suffix": "gemma_truth_sft",
    }
    print("  client.fine_tuning.create(")
    for k, v in cfg.items():
        print(f"      {k}={v!r},")
    print("  )   # SFT is the default method; DPO not used")
    json.dump({**artifacts, "job_config": cfg}, open(ARTIFACTS, "w"), indent=2)

    hr("Prep + upload complete; launch with finetune_run.py --launch")


if __name__ == "__main__":
    main()
