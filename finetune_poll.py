"""Poll the fine-tuning job to completion; report status, events (eval-loss
curve), and the output model name. Reads/writes ft_artifacts.json."""

from __future__ import annotations

import json
import time

from truthclf import llm

ARTIFACTS = "ft_artifacts.json"


def main():
    art = json.load(open(ARTIFACTS))
    jid = art["job_id"]
    client = llm.TogetherClient("google/gemma-4-31B-it")._ensure_client()

    last = None
    while True:
        job = client.fine_tuning.retrieve(jid)
        st = str(getattr(job, "status", ""))
        if st != last:
            print(f"[{time.strftime('%H:%M:%S')}] status={st}", flush=True)
            last = st
        su = st.upper()
        if "COMPLET" in su:
            break
        if any(x in su for x in ("FAIL", "ERROR", "CANCEL")):
            print(f"TERMINAL non-success: {st}", flush=True)
            break
        time.sleep(30)

    job = client.fine_tuning.retrieve(jid)
    out = getattr(job, "output_name", None)
    print(f"output_name: {out}", flush=True)
    print("=== events (eval-loss curve) ===", flush=True)
    try:
        for e in client.fine_tuning.list_events(id=jid).data:
            print("  ", getattr(e, "message", e), flush=True)
    except Exception as ex:
        print("  (could not list events:", ex, ")", flush=True)

    art["output_name"] = out
    json.dump(art, open(ARTIFACTS, "w"), indent=2)


if __name__ == "__main__":
    main()
