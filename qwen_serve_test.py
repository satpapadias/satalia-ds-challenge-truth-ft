"""Tiny throwaway Qwen fine-tune + empirical serverless-serving test (the gate).

Fine-tunes Qwen/Qwen3.5-9B on a small subset (1 epoch, <$1), then EMPIRICALLY
checks whether the resulting adapter serves serverlessly:
  - output_name callable serverlessly (no "non-serverless" error)?
  - does the adapter actually change outputs vs the base? (a fake adapter name
    that returns identical results means serving is silently ignoring it -> FAIL)

Reports PASS/FAIL with evidence. Does NOT run the full fine-tune.
"""

from __future__ import annotations

import json
import os
import time

import truthclf
from truthclf import data, prompts
from together import Together

BASE = "Qwen/Qwen3.5-9B"
N_TINY = 300
N_TEST = 40
ART = "qwen_serve_artifacts.json"
TINY_PATH = "ft_data/qwen_tiny_train.jsonl"
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def hr(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}", flush=True)


def main():
    os.makedirs("ft_data", exist_ok=True)
    art = json.load(open(ART)) if os.path.exists(ART) else {}
    c = Together()
    clean, _ = data.clean_dataset(data.load("data.csv"), "primary")
    train, _, _ = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, "primary")

    if not art.get("job_id"):
        data.write_sft_jsonl(train[:N_TINY], TINY_PATH, "full", "primary")
        fr = c.files.upload(file=TINY_PATH, purpose="fine-tune", check=True)
        resp = c.fine_tuning.create(model=BASE, training_file=fr.id, lora=True,
                                    n_epochs=1, learning_rate=1e-5,
                                    train_on_inputs="auto", suffix="qwen_serve_test")
        jid = getattr(resp, "id", None) or getattr(getattr(resp, "job", None), "id", None)
        art.update(train_file_id=fr.id, job_id=jid)
        json.dump(art, open(ART, "w"), indent=2)
        print(f"launched tiny Qwen FT job {jid} on {N_TINY} rows", flush=True)

    jid = art["job_id"]
    hr("POLLING TINY FT JOB")
    last = None
    while True:
        job = c.fine_tuning.retrieve(jid)
        st = str(getattr(job, "status", "")).upper()
        if st != last:
            print(f"[{time.strftime('%H:%M:%S')}] status={st}", flush=True)
            last = st
        if "COMPLET" in st:
            break
        if any(x in st for x in ("FAIL", "ERROR", "CANCEL")):
            print(f"TERMINAL non-success: {st}", flush=True)
            return
        time.sleep(30)

    job = c.fine_tuning.retrieve(jid)
    out = getattr(job, "model_output_name", None) or getattr(job, "output_name", None)
    art["output_name"] = out
    json.dump(art, open(ART, "w"), indent=2)
    print(f"output_name: {out}", flush=True)

    rows = clean[:N_TEST]

    def ans(model, msgs, extra=None):
        kw = dict(model=model, messages=msgs, max_tokens=1, temperature=0, **NO_THINK)
        if extra:
            kw["extra_body"] = extra
        return c.chat.completions.create(**kw).choices[0].message.content

    hr("SERVING TEST")
    # 1) output_name callable serverlessly?
    direct_ok, direct_err = True, ""
    try:
        ans(out, prompts.build_messages(rows[0], "full", "decision"))
    except Exception as e:
        direct_ok, direct_err = False, str(e)[:160]
    print(f"  [1] output_name callable serverlessly: {direct_ok}  {direct_err}", flush=True)

    # 2) adapter changes outputs vs base?
    diff = 0
    if direct_ok:
        for r in rows:
            m = prompts.build_messages(r, "full", "decision")
            if ans(BASE, m) != ans(out, m):
                diff += 1
        print(f"  [2] adapter vs base differ on {diff}/{len(rows)} rows", flush=True)

    # 3) bogus-adapter control on the base+extra_body path (must NOT silently work)
    bogus_ok = None
    try:
        ans(BASE, prompts.build_messages(rows[0], "full", "decision"),
            extra={"adapters": [{"name": "makisntpap_17e5/does-not-exist-xyz"}]})
        bogus_ok = True
    except Exception:
        bogus_ok = False
    print(f"  [3] base + BOGUS adapter returns normally (silently ignored): {bogus_ok}", flush=True)

    hr("RESULT")
    passed = direct_ok and diff > 0
    print(f"  SERVING TEST: {'PASS' if passed else 'FAIL'}", flush=True)
    print("  Criteria: output_name callable serverlessly AND adapter changes "
          "outputs vs base.", flush=True)
    if not passed:
        print("  -> do NOT proceed to full fine-tune; Qwen serverless LoRA not "
              "confirmed.", flush=True)


if __name__ == "__main__":
    main()
