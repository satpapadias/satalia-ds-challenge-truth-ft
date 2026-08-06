"""Launch the LoRA SFT fine-tuning job and poll it to completion.

Reads ft_artifacts.json (written by finetune_prep.py). Submits the job only when
invoked with --launch; without the flag it prints the config and exits. SFT is the
default method; DPO is not used. The resulting LoRA adapter is served via a
dedicated endpoint for evaluation (see README and evaluate_finetuned.py).

    python3 finetune_run.py            # dry run: print the config, do nothing
    python3 finetune_run.py --launch   # create the job, then poll to completion
"""

from __future__ import annotations

import json
import os
import sys
import time

from truthclf import llm

ARTIFACTS = "ft_artifacts.json"
POLL_INTERVAL = 30


def _poll(client, job_id):
    """Block until the job reaches a terminal state; return the output model name."""
    last = None
    while True:
        job = client.fine_tuning.retrieve(job_id)
        st = str(getattr(job, "status", ""))
        if st != last:
            print(f"  status={st}", flush=True)
            last = st
        su = st.upper()
        if "COMPLET" in su:
            break
        if any(x in su for x in ("FAIL", "ERROR", "CANCEL")):
            raise SystemExit(f"job ended with status {st}")
        time.sleep(POLL_INTERVAL)
    job = client.fine_tuning.retrieve(job_id)
    return getattr(job, "model_output_name", None) or getattr(job, "output_name", None)


def main():
    if not os.path.exists(ARTIFACTS):
        raise SystemExit("ft_artifacts.json not found — run finetune_prep.py first.")
    art = json.load(open(ARTIFACTS))
    cfg = art.get("job_config")
    if not cfg:
        raise SystemExit("no job_config in ft_artifacts.json — run finetune_prep.py first.")

    print("Fine-tuning job config:")
    for k, v in cfg.items():
        print(f"  {k} = {v!r}")

    if "--launch" not in sys.argv:
        print("\nDry run — not launched. Re-run with --launch to create the job.")
        return

    client = llm.TogetherClient(cfg["model"])._ensure_client()
    resp = client.fine_tuning.create(
        model=cfg["model"], training_file=cfg["training_file"],
        validation_file=cfg["validation_file"], n_evals=cfg["n_evals"], lora=cfg["lora"],
        n_epochs=cfg["n_epochs"], learning_rate=cfg["learning_rate"],
        train_on_inputs=cfg["train_on_inputs"], suffix=cfg["suffix"],
    )
    job_id = getattr(resp, "id", None) or getattr(getattr(resp, "job", None), "id", None)
    print(f"\nlaunched fine-tuning job: {job_id}\npolling to completion...")
    art["job_id"] = job_id
    json.dump(art, open(ARTIFACTS, "w"), indent=2)

    output_name = _poll(client, job_id)
    art["output_name"] = output_name
    json.dump(art, open(ARTIFACTS, "w"), indent=2)
    print(f"\ncompleted. output model: {output_name}")


if __name__ == "__main__":
    main()
