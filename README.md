# truthclf — Truthfulness Classification

Binary truthfulness classification of short public statements. The package
provides the three required components behind one consistent interface:

1. **Zero-shot predictor** — `predict(points, labels=None)`
2. **Fine-tuned predictor** — `fine_tune(training_dataset)` + `predict(points, labels=None)` (interchangeable with the zero-shot one)
3. **Explainer** — `explain(model, points, labels=None)`

Every component processes a **set** of points and returns evaluation metrics when
labels are supplied. The serving backend is **Together AI** (serverless inference
and LoRA fine-tuning).

---

## Results (held-out, speaker-disjoint test split, n = 1,991)

All numbers are **calibrated** (Platt scaling + decision threshold fit on a
separate validation split, reported on the untouched test split).

| Predictor | Accuracy | Balanced acc | Macro-F1 | Brier | ECE (bins occupied) |
|---|---|---|---|---|---|
| Zero-shot, decision/logprob (baseline) | 0.668 | 0.662 | 0.663 | 0.218 | 0.061 (9/10) |
| **Fine-tuned (LoRA SFT), decision** | **0.699** | **0.694** | **0.694** | **0.203** | 0.052 (10/10) |
| Zero-shot, score-mode (secondary) | 0.700 | 0.699 | 0.699 | 0.227 | 0.108 (7/10) |

**Fine-tuning buys accuracy.** Against the *matched* zero-shot baseline (same
elicitation, same split, same calibration): accuracy **+0.031 [+0.019, +0.043]**,
McNemar exact **p = 6.3e-07**. Modest and unambiguously significant.

**Fine-tuning does not buy calibration.** The ECE difference against the same
baseline is **+0.0097 [−0.0057, +0.0270]** — a paired bootstrap interval that
straddles zero. We do not claim a calibration improvement.

**Post-hoc calibration is what buys confidence quality, and it is load-bearing.**
On the zero-shot baseline, ECE goes **0.316 → 0.061**, a reduction of
**+0.255 [+0.213, +0.279]** with non-overlapping intervals. The raw elicited
probabilities are close to useless as confidences; the calibrated ones are not.

These are two separate contributions with separate evidence: **fine-tuning →
accuracy**, **calibration → confidence quality**. Neither substitutes for the
other. Hosted LLMs are mildly non-deterministic even at temperature 0; a live
refetch of the same prompts reproduced these to within 0.003 on every metric.

**Context:** accuracy on this task is inherently bounded well below 100% —
verifying short factual claims requires external world knowledge, and the human
verdicts have limited inter-annotator agreement. A calibrated result in the
high-0.60s to ~0.70 is strong here, not weak; the value is in honest, leakage-
controlled, well-calibrated evaluation rather than chasing raw accuracy.

### Key findings (one paragraph)

The task is hard and ceiling-bound (~0.62–0.69 in the literature), so the value is
in *honest* evaluation, not chasing accuracy. Using a leakage-controlled split
(speaker-disjoint + repeated-statement dedup) and post-hoc calibration, the
zero-shot decision baseline reaches 0.668 / ECE 0.061. Fine-tuning yields a
CI-backed, significant accuracy gain over the matched zero-shot baseline
(+0.031, p = 6.3e-07) but **no measurable calibration gain**; calibration itself
is what makes the probabilities usable (ECE 0.316 → 0.061). The explainer
(leave-one-field-out occlusion) shows the model leans on a **speaker shortcut**
(speaker identity is the deciding input for ~26% of predictions and its removal
flips ~10%), and crucially that this shortcut **hurts**: predictions driven by
the statement content are right 74.1% of the time versus 56.4% when driven by
the speaker. The model's free-text rationales agree with the occlusion-identified
driver only ~46% of the time, i.e. they are often post-hoc rationalizations
rather than faithful explanations.

---

## Install

Requires **Python ≥ 3.12** (developed and tested on **3.12.13**). Dependencies are
locked in `uv.lock`; [uv](https://docs.astral.sh/uv/) is the supported installer:

```bash
uv venv --python 3.12          # creates .venv with the pinned interpreter
uv sync --extra viz --extra dev   # installs exactly what uv.lock specifies
```

`uv sync` installs the package itself in editable mode, so `import truthclf`
works from the project root. Run everything through the venv (`.venv/bin/python`,
or `source .venv/bin/activate` first).

Plain pip also works if you do not want uv, but it resolves dependencies fresh
rather than using the lockfile:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[viz,dev]"
```

Set your Together API key in a project-root `.env` (loaded automatically):

```
TOGETHER_API_KEY=your_key_here
```

Locked versions (see `uv.lock` for the full graph): Python 3.12.13, numpy 2.5.1,
scikit-learn 1.9.0, scipy 1.18.0, statsmodels 0.14.6, pandas 3.0.5,
together 2.30.0, tiktoken 0.13.0, python-dotenv, matplotlib.

scikit-learn, scipy and statsmodels are **required, not optional**: the metrics,
calibration, threshold-tuning and splitting code defers to their reference
implementations instead of hand-rolling numerical routines.

---

## Reviewer quickstart

**No key needed** — open `notebooks/00_results_walkthrough.ipynb` (Restart & Run
All); it renders the recorded results and makes no API calls.

**To run live** (all commands from the project root):

```bash
# 1. setup (Python 3.12; see Install above)
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra viz --extra dev
printf 'TOGETHER_API_KEY=YOUR_KEY\n' > .env      # your own key
ls data.csv                                       # confirm the dataset is present

# 2. zero-shot logprob baseline on the test split (serverless, ~cents)
python3 scripts/zeroshot_baseline.py              # expect ~0.668 acc / ECE ~0.061

# 3. explainer: occlusion + rationale + cross-check (serverless, ~cents)
python3 scripts/run_explainer.py

# 4. fine-tuned vs zero-shot, paired, same test split
#    (spins up a dedicated Gemma endpoint — COSTS ~$2, ~10 min, needs the key;
#     runs zero-shot-logprob + fine-tuned-logprob, calibrates on val, prints the
#     comparison, and deletes the endpoint in a finally block)
python3 scripts/evaluate_finetuned.py
#    watch for: "logprobs OK" -> per-row progress -> "AUDIT: endpoint uptime ..."
#    -> "deleted endpoint ...". Afterwards confirm the Together endpoints
#    dashboard is empty.
```

**Expected reference numbers** (hosted LLMs are mildly non-deterministic, so small
variation is normal — the trends are what matter):

| Result | Expected |
|---|---|
| zero-shot, logprob (test) | acc ~0.668, ECE ~0.061 (equal-mass, 9 bins) |
| fine-tuned, logprob (test) | acc ~0.699, ECE ~0.052 (equal-mass, 10 bins) |
| paired fine-tuning effect | Δacc ~+0.031, McNemar p ~1e-6 |
| paired calibration effect | ΔECE ~+0.010, CI straddles zero — **no effect** |
| calibration vs raw probabilities | ECE ~0.316 → ~0.061 |
| explainer faithfulness | rationale↔occlusion agreement ~0.46 |
| driver vs correctness | statement-driven (~0.74) > speaker-driven (~0.56) |

---

## Usage / reproduce (run from the project root)

**How things run:** the predictor scripts call Together live and need your own
`TOGETHER_API_KEY` in a `.env` (see Install). The fine-tuning / dedicated-endpoint
scripts additionally cost money and are included for transparency — they are
optional. The **`notebooks/00_results_walkthrough.ipynb`** renders recorded results
from `results/summary.json` and needs **no key and makes no API calls** — start there.

**(a) Zero-shot predict on a set**
```bash
python3 predict.py
```
Or in code:
```python
from truthclf import experiments
pred = experiments.zeroshot_predictor("google/gemma-4-31B-it", variant="full")
rows = experiments.sample_rows("test", n=20)
res = pred.predict(rows, labels=[r.y("primary") for r in rows])
print(res.preds, res.metrics)
```

**(b) Fine-tune (LoRA SFT) + evaluate the fine-tuned model**
```bash
python3 scripts/finetune_prep.py         # build/validate/upload SFT data, print job config
python3 scripts/finetune_run.py --launch # launch the LoRA SFT job and poll to completion
python3 scripts/evaluate_finetuned.py    # serve on a short-lived dedicated endpoint; paired eval
```
In code (`FinetunedPredictor`):
```python
from truthclf.predictors import FinetunedPredictor
ft = FinetunedPredictor(base_model="google/gemma-4-31B-it", variant="full")
ft.fine_tune(training_dataset)                    # speaker-disjoint train/val split made internally
preds = ft.predict(test_rows, labels=test_labels)  # same interface as zero-shot
```

**(c) Explain a set of predictions**
```bash
python3 scripts/run_explainer.py     # occlusion + rationale + cross-check, aggregate over ~300 rows
```
```python
from truthclf import explain
result = explain.explain(predictor, rows, labels=labels)   # per-point + metrics
agg = explain.aggregate(result)                            # field-flip table + agreement rate
```

**(d) Reproduce the evaluation**
```bash
python3 scripts/analyze_dataset.py     # exploratory data analysis (no model calls)
python3 scripts/benchmark_zeroshot.py  # zero-shot model comparison (full set, bootstrap CIs)
python3 scripts/evaluate_zeroshot.py   # calibration + threshold tuning + abstention
python3 scripts/evaluate_finetuned.py  # fine-tuned vs zero-shot paired comparison
```

The `notebooks/00_results_walkthrough.ipynb` reproduces the recorded numbers
offline from `results/summary.json` (no key, no API calls). Running the scripts
live requires your own `TOGETHER_API_KEY` and will call the Together API (the raw
response caches are not bundled). Within a session, repeated calls are cached on
disk so reruns are free; hosted LLMs are mildly non-deterministic even at
temperature 0, so numbers vary slightly run-to-run.

---

## Design decisions

**Task framing & metric.** The six ordinal truthfulness labels are binarized with
a *middle split* (`true, mostly-true, half-true → True`; rest `False`), which is
near-balanced (≈56/44), so accuracy and balanced accuracy are both meaningful. We
report balanced accuracy, macro-F1, ROC/PR-AUC, and **calibration** (Brier, ECE)
with bootstrap CIs, and a sensitivity check that moves `half-true → False`.

**Leakage control.** A handful of speakers dominate the data and non-person
"speakers" (chain-email, viral-image) skew heavily False, so a naive random split
leaks. We use a **speaker-disjoint** split (no speaker in both train and test) plus
**repeated-statement dedup** (statements identical after normalisation never
cross the split — this is exact match after normalisation, not fuzzy matching),
and we **drop self-contradictory repeat groups** (same text, conflicting binary
label). We report the speaker-disjoint result as primary; the
gap to a stratified split estimates speaker memorization.

**Calibration & thresholds.** The raw elicited probability is badly uncalibrated
(ECE 0.316 on the zero-shot baseline), so we fit a post-hoc calibrator and a
decision threshold on validation only, then report on test. This is the largest
single improvement in the whole pipeline: ECE 0.316 → 0.061, +0.255 [+0.213,
+0.279]. Accuracy is essentially unchanged — calibration buys confidence quality,
not correctness. A cost-sensitive threshold is provided because, in a
misinformation setting, labelling a false statement as True (a false positive in
this 1=True encoding) is the costlier error.

*Calibrator selection.* Temperature scaling (1 parameter) and Platt scaling (2)
are both fitted on validation and chosen by **validation NLL**, a proper scoring
rule. Platt is selected only if a paired bootstrap of the NLL difference excludes
zero; otherwise the simpler model is kept on parsimony grounds. **The selection
target and the reported calibration metric are not the same quantity, and we say
so explicitly:** selection optimises NLL, but the resulting calibrators are
*statistically indistinguishable on ECE* — on all three runs the paired CI for
ECE(temperature) − ECE(Platt) includes zero. We do not select on validation ECE,
because a criterion that cannot separate the candidates on test at n≈2000 is
noisier still on validation, and selecting on it would be selecting on noise.

*ECE methodology.* Reported ECE uses **equal-mass (quantile) bins**, and every
ECE ships with the number of bins actually occupied. Equal-width bins are
structurally unusable here: confidence-of-predicted-class lives in [0.5, 1], so
half of a [0,1] grid can never be occupied, and on these runs only 2–3 of 10
equal-width bins carried any mass. That flatters the estimate by hiding
miscalibration inside one large bin — on the score-mode run, equal-width reports
ECE 0.065 from 2 occupied bins where equal-mass reports 0.108 from 7. The
equal-width figure is retained as a diagnostic under `ece_detail.uniform` in the
results files. Ties can still collapse quantile edges, which is why the realised
bin count is always reported alongside: a 2-bin and a 10-bin ECE are not
comparable quantities.

**Fine-tuning.** LoRA supervised fine-tuning on Together (`gpt-4o-mini`/OpenAI was
not viable — deprecated and fine-tuning closed to new users). Serverless serving of
*custom* LoRA adapters was not available for our base, so the fine-tuned model is
served on a **short-lived dedicated endpoint** purely for evaluation (created,
queried, and deleted within minutes). **DPO is deliberately excluded**: a
single-label binary task has no preferred/dispreferred ranking, so DPO degenerates
to what SFT already does.

**Explainer & faithfulness.** Two layers: **leave-one-field-out occlusion** (the
faithful, causal layer — re-run the predictor with each metadata field removed and
measure the probability shift / label flip) and the **model's own one-sentence
rationale** (readable but *not necessarily faithful*). A cross-check compares the
occlusion-identified driver with the field the rationale cites and reports the
agreement rate. **Token/word-level attribution is out of scope** — it is expensive,
noisy, and the wrong granularity for field-structured metadata; field-level
occlusion is the right unit.

---

## Package layout

```
truthclf/
  data.py        load, clean, binarize, contradiction-drop, leakage-aware splits, SFT serialization
  prompts.py     prompt templates (score / decision / rationale), [unknown] for missing fields
  llm.py         Together clients (sync + Batch API), token/cost estimation, on-disk cache
  metrics.py     balanced acc, macro-F1, ROC/PR-AUC, Brier, ECE, bootstrap CIs, McNemar
  calibration.py temperature / Platt scaling
  threshold.py   balanced and cost-sensitive threshold tuning
  selective.py   coverage-vs-accuracy (abstention)
  explain.py     occlusion + rationale + cross-check explainer
  experiments.py high-level helpers (run a predictor, build splits, summary tables)
  viz.py         reliability + coverage plots (matplotlib, optional)
  predictors/
    base.py      Predictor interface + metric bundle
    zeroshot.py  ZeroShotPredictor (0-100 score primary; logprob path optional)
    finetuned.py FinetunedPredictor (fine_tune + predict)
tests/           unit tests (LLM calls mocked — deterministic, free)
predict.py       top-level entry point: zero-shot predict on a set
scripts/         full reproduce flow (EDA, full-test eval, calibration, fine-tune, eval, explain)
notebooks/       00_results_walkthrough.ipynb — guided, import-and-display tour
results/         summary.json — recorded headline numbers (for the walkthrough)
data.csv         the provided challenge dataset
```

## Tests

```bash
python3 -m pytest tests/ -q
```
**All LLM calls are mocked**, so the suite is deterministic and free. Run it
through the venv: `.venv/bin/python -m pytest tests/ -q`.

## Data & confidentiality

`data.csv` is the dataset **provided for the challenge**, included here for
end-to-end reproducibility. Do not push this code or data to any public
repository.
