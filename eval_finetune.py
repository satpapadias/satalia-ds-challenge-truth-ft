"""Fine-tuned vs zero-shot paired evaluation (Gemma), decision+logprob mode.

Serves the Gemma LoRA fine-tune on a SHORT-LIVED dedicated endpoint (serverless
serving is unavailable for this base), runs the held-out test + val through it,
PERSISTS every row to disk immediately, then ALWAYS deletes the endpoint.

Three calibrated numbers on the test split (Platt + threshold fit on val):
  (a) zero-shot Gemma, score-mode       -> secondary 0-100 reference
  (b) zero-shot Gemma, decision+logprob -> the reported zero-shot baseline (serverless)
  (c) fine-tuned Gemma, decision+logprob -> the result (dedicated endpoint)
Paired: (c vs b) = headline fine-tuning effect (matched elicitation);
        (c vs a) = secondary, cross-elicitation note.

Safety: idempotent pre-cleanup of leftover endpoints; list_hardware preflight;
min=max=1 replica; first-call logprob verification; hard max-uptime kill switch;
endpoint deleted in a finally block (covers exceptions + KeyboardInterrupt);
incremental per-row persistence; single endpoint session for val+test.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np

import truthclf
from truthclf import data, llm, calibration, threshold, metrics, prompts, experiments
from truthclf.predictors import ZeroShotPredictor
from truthclf.predictors.zeroshot import prob_from_logprobs
from together import Together

BASE = "google/gemma-4-31B-it"
FT = "makisntpap_17e5/gemma-4-31B-it-gemma_truth_sft-c7afbf0d"
HARDWARE = "2x_nvidia_h100_80gb_sxm"
SCHEME, VARIANT = "primary", "full"
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
FT_CACHE, RESULTS = "ft_eval_cache.json", "ft_eval_results.json"
EP_NAME = "truth-gemma-ft-eval"
PROVISION_DEADLINE = 20 * 60
MAX_UPTIME_MIN = 30                 # hard kill switch: delete + abort past this
WORKERS = 16


def hr(t):
    print(f"\n{'='*78}\n  {t}\n{'='*78}", flush=True)


def _ptrue(choice):
    p = prob_from_logprobs(llm.top_logprobs_from_choice(choice))
    if p is None:
        tok = (choice.message.content or "").strip().lower()
        p = 1.0 if tok.startswith("true") else 0.0
    return p


def cleanup_leftovers(c):
    try:
        eps = c.endpoints.list()
        for e in getattr(eps, "data", eps) or []:
            nm = (getattr(e, "display_name", "") or "")
            mdl = (getattr(e, "model", "") or "")
            if EP_NAME in nm or FT in mdl:
                try:
                    c.endpoints.delete(e.id)
                    print(f"  cleaned leftover endpoint {e.id}", flush=True)
                except Exception as ex:
                    print(f"  could not delete leftover {e.id}: {ex}", flush=True)
    except Exception as ex:
        print(f"  endpoint list/cleanup skipped: {ex}", flush=True)


def base_decision_probs(rows):
    # serverless + cached; Batch API for the ~2k calls (faster/cheaper than serial)
    client = llm.TogetherBatchClient(BASE, max_output_tokens=1,
                                     extra_create_kwargs=NO_THINK, poll_interval=20)
    return ZeroShotPredictor(BASE, VARIANT, client=client, use_logprobs=True).predict(rows).probs


def ft_decision_probs(rows, c):
    """Fine-tuned decision+logprob probs via one short-lived endpoint; per-row
    results persisted by row_id so a rerun never needs the endpoint again."""
    cache = json.load(open(FT_CACHE)) if os.path.exists(FT_CACHE) else {}
    need = [r for r in rows if str(r.row_id) not in cache]
    print(f"  FT rows: {len(rows)} total, {len(need)} need the endpoint", flush=True)
    if not need:
        return [cache[str(r.row_id)] for r in rows]

    # preflight: can this base host a fine-tune? (404 -> stop)
    hr("PREFLIGHT list_hardware")
    hw = c.endpoints.list_hardware(model=FT)
    opts = getattr(hw, "data", hw)
    ids = [getattr(h, "id", None) for h in opts]
    print(f"  hardware options: {ids}", flush=True)
    if not ids:
        raise SystemExit("no hardware options -> base cannot host a fine-tune; stop.")
    hardware = HARDWARE if HARDWARE in ids else ids[0]
    print(f"  using {hardware}", flush=True)

    cleanup_leftovers(c)

    hr("CREATING DEDICATED ENDPOINT (short-lived)")
    created = time.time()
    print(f"  create time: {time.strftime('%H:%M:%S', time.localtime(created))}", flush=True)
    ep = c.endpoints.create(display_name=EP_NAME, model=FT, hardware=hardware,
                            autoscaling={"min_replicas": 1, "max_replicas": 1})
    print(f"  endpoint id={ep.id}", flush=True)

    def uptime():
        return time.time() - created

    def kill_check():
        # hard safety: tear down and abort if the endpoint runs longer than expected
        if uptime() > MAX_UPTIME_MIN * 60:
            raise TimeoutError(f"max uptime {MAX_UPTIME_MIN} min exceeded")

    try:
        # poll to STARTED
        while True:
            st = c.endpoints.retrieve(ep.id).state
            if st == "STARTED":
                break
            if st in ("FAILED", "STOPPED"):
                raise RuntimeError(f"endpoint state {st}")
            if uptime() > PROVISION_DEADLINE:
                raise TimeoutError("provisioning exceeded deadline")
            kill_check()
            time.sleep(20)
        name = c.endpoints.retrieve(ep.id).name
        print(f"  STARTED in {uptime():.0f}s; endpoint.name={name}", flush=True)

        # verify logprobs on the FIRST call before running everything
        hr("VERIFY LOGPROBS ON ENDPOINT (first call)")
        probe = c.chat.completions.create(model=name,
                                          messages=prompts.build_messages(need[0], VARIANT, "decision"),
                                          max_tokens=1, temperature=0, **NO_THINK,
                                          extra_body={"logprobs": True, "top_logprobs": 10})
        top = llm.top_logprobs_from_choice(probe.choices[0])
        if not top or prob_from_logprobs(top) is None:
            raise SystemExit(f"endpoint did not return usable logprobs: {top} -> stop.")
        print(f"  logprobs OK: {dict(list(top.items())[:4])}", flush=True)

        lock = Lock()

        def work(r):
            kill_check()
            resp = c.chat.completions.create(model=name,
                                             messages=prompts.build_messages(r, VARIANT, "decision"),
                                             max_tokens=1, temperature=0, **NO_THINK,
                                             extra_body={"logprobs": True, "top_logprobs": 10})
            return r.row_id, _ptrue(resp.choices[0])

        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for rid, p in ex.map(work, need):
                with lock:
                    cache[str(rid)] = p
                    done += 1
                    if done % 200 == 0:
                        json.dump(cache, open(FT_CACHE, "w"))
                        print(f"    {done}/{len(need)} | uptime {uptime():.0f}s", flush=True)
        json.dump(cache, open(FT_CACHE, "w"))
        print(f"  FT inference complete ({done} rows)", flush=True)
    finally:
        hr("DELETING ENDPOINT (always)")
        deleted = time.time()
        try:
            c.endpoints.delete(ep.id)
            print(f"  deleted endpoint {ep.id} at "
                  f"{time.strftime('%H:%M:%S', time.localtime(deleted))}", flush=True)
        except Exception as ex:
            print(f"  !! delete FAILED: {ex} -- DELETE MANUALLY: {ep.id}", flush=True)
        print(f"  AUDIT: endpoint uptime {(deleted - created)/60.0:.1f} min", flush=True)

    return [cache[str(r.row_id)] for r in rows]


def calibrated_eval(vp, vy, tp, ty):
    cal = calibration.fit_best(vp, vy)
    vpc = list(calibration.apply(vp, cal))
    tpc = list(calibration.apply(tp, cal))
    thr, _ = threshold.tune_threshold(vpc, vy, "balanced_accuracy")
    preds = (np.asarray(tpc) >= thr).astype(int)
    return {"calibrator": cal["method"], "threshold": float(thr),
            "preds": preds.tolist(), "metrics": metrics.metric_bundle(ty, preds, tpc)}


def main():
    c = Together()
    clean, _ = data.clean_dataset(data.load("data.csv"), SCHEME)
    _, val, test = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, SCHEME)
    vy = [r.y(SCHEME) for r in val]
    ty = [r.y(SCHEME) for r in test]
    print(f"val n={len(val)} test n={len(test)}", flush=True)

    hr("(a) zero-shot score-mode + (b) zero-shot decision (serverless, Batch API)")
    a_vp, _ = experiments.run_on_rows(BASE, VARIANT, val, SCHEME, backend="batch")
    a_tp, _ = experiments.run_on_rows(BASE, VARIANT, test, SCHEME, backend="batch")
    b_vp, b_tp = base_decision_probs(val), base_decision_probs(test)

    hr("(c) fine-tuned decision+logprob (one endpoint session for val+test)")
    ft_all = ft_decision_probs(val + test, c)
    c_vp, c_tp = ft_all[:len(val)], ft_all[len(val):]

    hr("CALIBRATED TEST METRICS (Platt + threshold fit on val)")
    A = calibrated_eval(a_vp, vy, a_tp, ty)
    B = calibrated_eval(b_vp, vy, b_tp, ty)
    C = calibrated_eval(c_vp, vy, c_tp, ty)
    for tag, R in [("(a) zero-shot score", A), ("(b) zero-shot decision", B),
                   ("(c) fine-tuned decision", C)]:
        m = R["metrics"]
        print(f"  {tag:<26} acc={m['accuracy']:.3f} bal={m['balanced_accuracy']:.3f} "
              f"f1={m['macro_f1']:.3f} brier={m['brier']:.3f} ece={m['ece']:.3f}", flush=True)

    hr("PAIRED TESTS (test split, n=%d)" % len(test))
    mc_cb = metrics.mcnemar(ty, C["preds"], B["preds"]); d_cb = metrics.paired_accuracy_diff(ty, C["preds"], B["preds"])
    print(f"  (c vs b) CLEAN fine-tuning effect: acc {d_cb[0]:+.3f} "
          f"[{d_cb[1]:+.3f},{d_cb[2]:+.3f}]  McNemar p={mc_cb['p_value']:.4f} "
          f"(b={mc_cb['b']},c={mc_cb['c']})", flush=True)
    mc_ca = metrics.mcnemar(ty, C["preds"], A["preds"]); d_ca = metrics.paired_accuracy_diff(ty, C["preds"], A["preds"])
    print(f"  (c vs a) secondary cross-elicitation: acc {d_ca[0]:+.3f} "
          f"[{d_ca[1]:+.3f},{d_ca[2]:+.3f}]  McNemar p={mc_ca['p_value']:.4f}", flush=True)

    json.dump({"a": A["metrics"], "b": B["metrics"], "c": C["metrics"],
               "c_vs_b": {"acc_diff": d_cb, "mcnemar": mc_cb},
               "c_vs_a": {"acc_diff": d_ca, "mcnemar": mc_ca}},
              open(RESULTS, "w"), indent=2, default=float)
    print(f"\n  results written to {RESULTS}", flush=True)
    hr("DONE")


if __name__ == "__main__":
    main()
