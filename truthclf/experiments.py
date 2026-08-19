"""High-level experiment helpers so notebooks and scripts stay thin.

A single call, run_zeroshot(...), loads the data, selects a split, builds the
right client for the model (from llm.MODEL_CONFIGS), runs the zero-shot
predictor with caching, and returns a ZeroShotRun bundling predictions, labels,
metrics, parse-failure count, and average per-call latency.

The same call works for a fine-tuned model id (it falls back to the default
client config), so a notebook can flip MODEL to a fine-tuned id and rerun
unchanged — the predictor interface is identical.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import data, evaluation, llm, metrics, calibration, threshold, selective
from .predictors import ZeroShotPredictor


@dataclass
class ZeroShotRun:
    model: str
    variant: str
    split: str
    n: int
    labels: list
    probs: list
    preds: list
    metrics: dict
    parse_failures: int
    avg_latency_s: float
    error_failures: int = 0


def _select_rows(split: str, n_rows: int, scheme: str, seed: int, data_path: str):
    rows = data.load(data_path)
    clean, _ = data.clean_dataset(rows, scheme)
    if split == "full":
        return clean
    if split == "dev":
        return data.dev_subset(clean, n=n_rows, seed=seed)
    if split in ("test", "train"):
        train, test = data.speaker_disjoint_split(clean, test_frac=0.2, seed=seed, scheme=scheme)
        chosen = test if split == "test" else train
        return chosen[:n_rows] if n_rows else chosen
    raise ValueError(f"unknown split: {split!r}")


def run_zeroshot(model: str, variant: str = "full", n_rows: int = 200,
                 split: str = "dev", scheme: str = "primary", seed: int = 0,
                 data_path: str = "data.csv", client=None,
                 backend: str = "sync", use_logprobs: bool = False) -> ZeroShotRun:
    # use_logprobs=False here: this helper drives the 0-100 score-mode model
    # comparison; the headline baseline uses the logprob mode (the predictor default).
    rows = _select_rows(split, n_rows, scheme, seed, data_path)
    labels = [r.y(scheme) for r in rows]
    client = client or llm.make_client(model, backend=backend)
    res = ZeroShotPredictor(model=model, variant=variant, client=client, seed=seed,
                            use_logprobs=use_logprobs).predict(rows, labels=labels)
    return ZeroShotRun(
        model=model, variant=variant, split=split, n=res.n, labels=labels,
        probs=res.probs, preds=res.preds, metrics=res.metrics,
        parse_failures=res.parse_failures, avg_latency_s=client.avg_latency,
        error_failures=getattr(client, "error_count", 0),
    )


def zeroshot_predictor(model, variant="full", backend="sync", **kw):
    """Build a ZeroShotPredictor wired to a client."""
    return ZeroShotPredictor(model=model, variant=variant,
                             client=llm.make_client(model, backend=backend), **kw)


def sample_rows(split="test", n=4, seed=0, scheme="primary", data_path="data.csv"):
    """A small deterministic row sample from a split (for explainer demos)."""
    clean, _ = data.clean_dataset(data.load(data_path), scheme)
    if split == "test":
        _, _, sel = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, scheme)
    elif split == "val":
        _, sel, _ = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, scheme)
    else:
        sel = clean
    return data.dev_subset(sel, n=n, seed=seed)


def run_on_rows(model, variant, rows, scheme="primary", backend="sync", client=None,
                use_logprobs=False):
    """Run the zero-shot predictor on an explicit list of rows (cached -> free).
    Defaults to score-mode (use_logprobs=False) so callers that need continuous
    probabilities or the 0-100 reference get them. Returns (probs, labels)."""
    labels = [r.y(scheme) for r in rows]
    client = client or llm.make_client(model, backend=backend)
    res = ZeroShotPredictor(model=model, variant=variant, client=client,
                            use_logprobs=use_logprobs).predict(rows, labels=labels)
    return res.probs, labels


@dataclass
class PhaseAResult:
    model: str
    variant: str
    val_labels: list
    val_probs: list
    test_labels: list
    test_probs: list          # raw (score/100)
    test_probs_cal: list      # calibrated
    thresholds: dict
    calibrator: dict
    threshold_table: object   # pandas DataFrame
    calibration_table: object
    official: dict            # calibrated + val-tuned threshold, test metrics


def _row(name, labels, probs, thr):
    import numpy as np
    preds = (np.asarray(probs) >= thr).astype(int)
    b = metrics.metric_bundle(labels, preds, probs)
    c = metrics.confusion(labels, preds)
    return {"setting": name, "threshold": round(thr, 3),
            "acc": round(b["accuracy"], 3), "bal_acc": round(b["balanced_accuracy"], 3),
            "macro_f1": round(b["macro_f1"], 3),
            "rec_True": round(b["recall_True"], 3), "rec_False": round(b["recall_False"], 3),
            "FN": c["fn"], "FP": c["fp"]}


# In this encoding 1 = True/truthful, so "passing a falsehood as true" is a FALSE
# POSITIVE. That is the costlier error in a misinformation setting, so c_fp > c_fn.
def phase_a_eval(model="gemini-2.5-flash", variant="full", scheme="primary",
                 seed=0, data_path="data.csv", c_fn=1.0, c_fp=2.0):
    """Threshold tuning + calibration for the lead config, tuned on a speaker-
    disjoint validation split and reported on the untouched test split."""
    import pandas as pd
    rows = data.load(data_path)
    clean, _ = data.clean_dataset(rows, scheme)
    _, val, test = data.speaker_disjoint_3way(clean, val_frac=0.2, test_frac=0.2,
                                              seed=seed, scheme=scheme)
    vp, vy = run_on_rows(model, variant, val, scheme, backend="batch")
    tp, ty = run_on_rows(model, variant, test, scheme, backend="batch")

    # --- threshold tuning on RAW probabilities.
    # DELIBERATELY NOT the adopted pipeline: this table exists to show what
    # threshold tuning alone buys, before any calibration. The reported record
    # uses truthclf.evaluation.calibrated_evaluation (calibrate first, then tune
    # on CALIBRATED validation probabilities) — see `official` below.
    thr_bal, _ = threshold.tune_threshold(vp, vy, "balanced_accuracy")
    thr_f1, _ = threshold.tune_threshold(vp, vy, "macro_f1")
    thr_cost, _ = threshold.tune_cost_threshold(vp, vy, c_fn=c_fn, c_fp=c_fp)
    thresholds = {"default": 0.5, "balanced": thr_bal, "macro_f1": thr_f1, "cost": thr_cost}
    threshold_table = pd.DataFrame([
        _row("default 0.5", ty, tp, 0.5),
        _row("balanced-opt (val)", ty, tp, thr_bal),
        _row("macro-F1-opt (val)", ty, tp, thr_f1),
        _row(f"cost FP{c_fp:g}:FN{c_fn:g} (val)", ty, tp, thr_cost),
    ])

    # --- calibration (fit on val, report on test)
    cal = calibration.fit_best(vp, vy)
    tp_cal = list(calibration.apply(tp, cal))
    calibration_table = pd.DataFrame([
        {"stage": "raw score/100", "brier": round(metrics.brier(ty, tp), 3),
         "ece": round(metrics.ece(ty, tp), 3)},
        {"stage": f"calibrated ({cal['method']})", "brier": round(metrics.brier(ty, tp_cal), 3),
         "ece": round(metrics.ece(ty, tp_cal), 3)},
    ])

    # --- official baseline: the adopted pipeline, shared with every other caller
    ev = evaluation.calibrated_evaluation(vp, vy, tp, ty, objective="balanced_accuracy")
    official = {"threshold": ev.threshold, "calibrator": ev.calibrator,
                "metrics": ev.metrics}

    return PhaseAResult(model, variant, vy, vp, ty, tp, tp_cal, thresholds, cal,
                        threshold_table, calibration_table, official)


def ablation_table(test_rows=None, scheme="primary", seed=0, data_path="data.csv"):
    """Metadata ablation on the test split."""
    import pandas as pd
    if test_rows is None:
        rows = data.load(data_path)
        clean, _ = data.clean_dataset(rows, scheme)
        _, _, test_rows = data.speaker_disjoint_3way(clean, 0.2, 0.2, seed, scheme)
    out = []
    for model in ("gemini-2.5-flash",):
        for variant in ("statement_only", "full"):
            probs, labels = run_on_rows(model, variant, test_rows, scheme, backend="batch")
            import numpy as np
            preds = (np.asarray(probs) >= 0.5).astype(int)
            b = metrics.metric_bundle(labels, preds, probs)
            out.append({"model": model.split("/")[-1], "variant": variant,
                        "acc": round(b["accuracy"], 3), "bal_acc": round(b["balanced_accuracy"], 3),
                        "macro_f1": round(b["macro_f1"], 3),
                        "rec_True": round(b["recall_True"], 3),
                        "rec_False": round(b["recall_False"], 3)})
    return pd.DataFrame(out)


def metrics_table(runs):
    """Return a pandas DataFrame summarising one or more ZeroShotRun objects."""
    import pandas as pd
    rows = []
    for r in runs:
        m = r.metrics
        rows.append({
            "model": r.model.split("/")[-1],
            "variant": r.variant,
            "n": r.n,
            "acc": round(m["accuracy"], 3),
            "bal_acc": round(m["balanced_accuracy"], 3),
            "macro_f1": round(m["macro_f1"], 3),
            "parse_fail_%": round(r.parse_failures / r.n * 100, 1) if r.n else 0.0,
            "brier": round(m["brier"], 3),
            "latency_s": round(r.avg_latency_s, 2),
        })
    return pd.DataFrame(rows)
