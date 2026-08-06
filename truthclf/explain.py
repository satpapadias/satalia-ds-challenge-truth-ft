"""Component 3 — the explainer. Model-agnostic: it only CALLS the predictor
(predict + rationale), so it works identically on the zero-shot or fine-tuned
predictor. No gradients/attention.

Two layers, with explicit faithfulness framing:

1. Leave-one-field-out occlusion (PRIMARY, faithful). Re-run the predictor with
   each metadata field blanked in turn (speaker_name, speaker_affiliation,
   subjects, statement_context) plus an all-metadata-removed baseline. Report how
   much each removal shifts the predicted probability and whether it flips the
   label. This is causal, input-level attribution.

2. Model's own rationale (READABLE, not necessarily faithful). One sentence on
   why the model judged the statement True/False. Labelled as plausible only.

Cross-check: does the field occlusion shows actually drove the prediction match
what the rationale cites? Disagreement exposes rationalization. The agreement
rate is reported in aggregate.

Token/word-level attribution is intentionally out of scope (expensive, noisy, and
the wrong granularity for field-structured metadata); see the README for rationale.
"""

from __future__ import annotations

import collections
import dataclasses

from . import metrics

OCCLUSION_FIELDS = ["speaker_name", "speaker_affiliation", "subjects", "statement_context"]
ALL_META = OCCLUSION_FIELDS + ["speaker_job", "speaker_state"]

_DRIVER_KEY = {"speaker_name": "speaker", "speaker_affiliation": "affiliation",
               "subjects": "subjects", "statement_context": "context",
               "statement": "statement", "all_metadata": "statement"}

_AFFIL = {"republican", "democrat", "democratic", "gop", "party", "partisan",
          "political", "liberal", "conservative", "left-wing", "right-wing"}
_CONTEXT = {"interview", "debate", "speech", "ad", "advertisement", "facebook",
            "tweet", "twitter", "post", "email", "chain", "press", "release",
            "rally", "campaign", "radio", "blog", "news", "column"}
_STATEMENT = {"claim", "statement", "figure", "number", "statistic", "statistics",
              "data", "fact", "specific", "amount", "percent", "percentage",
              "quote", "exaggerat", "misleading", "accurate", "evidence"}
_SPEAKER = {"speaker", "said", "spoke", "person", "official"}


def _ablate(row, fields):
    return dataclasses.replace(row, **{f: "" for f in fields})


def _rationale_refs(text, row):
    """Which input fields the rationale text seems to cite (keyword heuristic)."""
    t = (text or "").lower()
    refs = set()
    spk = [w for w in (row.speaker_name or "").lower().replace("-", " ").split() if len(w) > 2]
    if any(w in t for w in spk) or any(w in t for w in _SPEAKER):
        refs.add("speaker")
    if any(w in t for w in _AFFIL):
        refs.add("affiliation")
    subj_words = set()
    for s in (row.subjects or "").split("$"):
        subj_words.update(w for w in s.lower().replace("-", " ").split() if len(w) > 3)
    if any(w in t for w in subj_words):
        refs.add("subjects")
    if any(w in t for w in _CONTEXT):
        refs.add("context")
    if any(w in t for w in _STATEMENT):
        refs.add("statement")
    if not refs:
        refs.add("statement")
    return refs


def explain(model, points, labels=None, with_rationale=True,
            threshold=0.5, driver_eps=0.05):
    """Explain a SET of points with occlusion + rationale + cross-check.
    Returns {"per_point": [...], "metrics": {...} if labels}. Batches all
    occlusion variants into a single predict() call (cache/batch friendly)."""
    rows, tags = [], []
    for i, p in enumerate(points):
        rows.append(p); tags.append((i, "base"))
        for f in OCCLUSION_FIELDS:
            rows.append(_ablate(p, [f])); tags.append((i, f))
        rows.append(_ablate(p, ALL_META)); tags.append((i, "all_metadata"))

    probs = model.predict(rows).probs
    byp = collections.defaultdict(dict)
    for (i, tag), pr in zip(tags, probs):
        byp[i][tag] = pr

    rats = (model.rationale(list(points))
            if with_rationale and hasattr(model, "rationale") else [None] * len(points))

    per, base_preds, base_probs = [], [], []
    for i, p in enumerate(points):
        base = byp[i]["base"]
        bp = int(base >= threshold)
        base_preds.append(bp); base_probs.append(base)
        occ = {}
        for f in OCCLUSION_FIELDS + ["all_metadata"]:
            pr = byp[i][f]
            occ[f] = {"prob": round(pr, 3), "delta": round(base - pr, 3),
                      "flip": int((pr >= threshold) != bool(bp))}
        md = {f: abs(occ[f]["delta"]) for f in OCCLUSION_FIELDS}
        top = max(md, key=md.get)
        driver = top if md[top] > driver_eps else "statement"
        rat = rats[i]
        refs = _rationale_refs(rat, p) if rat is not None else None
        agree = (_DRIVER_KEY[driver] in refs) if refs is not None else None
        per.append({"row_id": p.row_id, "statement": p.statement,
                    "label": (labels[i] if labels is not None else None),
                    "pred": bp, "base_prob": round(base, 3), "occlusion": occ,
                    "driver": driver, "rationale": rat,
                    "rationale_refs": sorted(refs) if refs is not None else None,
                    "agree": agree})

    out = {"per_point": per}
    if labels is not None:
        out["metrics"] = metrics.metric_bundle(labels, base_preds, base_probs)
    return out


def aggregate(result):
    """Aggregate field-flip rates, driver distribution, and the rationale-vs-
    occlusion agreement rate over an explain() result. Returns a dict with a
    pandas field-importance table."""
    import pandas as pd
    per = result["per_point"]
    n = len(per)
    rows = []
    for f in OCCLUSION_FIELDS + ["all_metadata"]:
        flips = sum(pp["occlusion"][f]["flip"] for pp in per)
        mad = sum(abs(pp["occlusion"][f]["delta"]) for pp in per) / n
        rows.append({"removed_field": f, "flip_rate": round(flips / n, 3),
                     "mean_abs_delta": round(mad, 3)})
    drivers = collections.Counter(pp["driver"] for pp in per)
    agrees = [pp["agree"] for pp in per if pp["agree"] is not None]
    return {"field_table": pd.DataFrame(rows),
            "driver_distribution": dict(drivers),
            "rationale_occlusion_agreement_rate": (round(sum(agrees) / len(agrees), 3)
                                                   if agrees else None),
            "n": n}


def examples_frame(result, k=8):
    """Per-point explanation table for display (occlusion Δp + driver + rationale)."""
    import pandas as pd
    rows = []
    for pp in result["per_point"][:k]:
        occ = pp["occlusion"]
        rows.append({"row_id": pp["row_id"], "pred": "True" if pp["pred"] else "False",
                     "p": pp["base_prob"], "driver": pp["driver"],
                     "d_speaker": occ["speaker_name"]["delta"],
                     "d_affil": occ["speaker_affiliation"]["delta"],
                     "d_subj": occ["subjects"]["delta"],
                     "d_ctx": occ["statement_context"]["delta"],
                     "agree": pp["agree"],
                     "rationale": (pp["rationale"] or "")[:70]})
    return pd.DataFrame(rows)
