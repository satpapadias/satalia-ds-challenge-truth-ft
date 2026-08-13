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
(leave-one-field-out occlusion) shows the model leans on a **speaker shortcut** —
speaker identity is the deciding input for **26%** of predictions and removing it
flips **9.7%** — and that leaning on it buys nothing: speaker-driven predictions
sit exactly on their own subset's majority-class rate (**+0.000 [−0.141,
+0.115]**). The clearest signal runs the other way: on the **45.7%** of points
where *no* metadata field moves the prediction at all, accuracy is **+0.204
[+0.088, +0.285]** above that subset's baseline — one of six subsets tested, and
the only interval excluding zero. That is most likely selection rather than
cause: a statement the model judges without reference to metadata is plausibly a
clear-cut one to begin with.

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

```bash
cp .env.example .env      # then fill in TOGETHER_API_KEY
```

`.env` is gitignored. A key is needed only for the scripts that call Together;
the walkthrough notebook, the test suite, `scripts/analyze_dataset.py` and the
offline regeneration path below all run without one.

Locked versions (see `uv.lock` for the full graph): Python 3.12.13, numpy 2.5.1,
scikit-learn 1.9.0, scipy 1.18.0, statsmodels 0.14.6, pandas 3.0.5,
together 2.30.0, tiktoken 0.13.0, python-dotenv, matplotlib.

scikit-learn, scipy and statsmodels are **required, not optional**: the metrics,
calibration, threshold-tuning and splitting code defers to their reference
implementations instead of hand-rolling numerical routines.

---

## The response cache

LLM responses are cached on disk so reruns are free. There are two stores, and
the distinction matters on a fresh clone:

| | tracked? | what it is |
|---|---|---|
| `.llm_cache.json` | **yes** | the schema-1 archive — the record of what was actually spent, and the source the adopted numbers replay from |
| `.llm_cache/` | **no — gitignored** | the schema-2 diskcache the code reads at runtime (~15 MB SQLite) |

**A fresh clone therefore has no `.llm_cache/`.** Rebuild it, or bypass it:

```bash
# Option A — rebuild the runtime cache (recommended)
python3 scripts/migrate_cache.py --apply      # free: re-keys the tracked v1 archive
python3 scripts/refetch_quarantined.py        # ~$0.33, needs TOGETHER_API_KEY

# Option B — no key, no cost: replay the tracked archive directly
python3 scripts/regenerate_results.py --source archive --explainer-source archive
```

Option B reproduces the adopted record in about 13 seconds with no API calls.
The entrypoint **fails with exit code 1** and names these commands if the cache
is missing — it will not silently produce numbers from an incomplete cache.

Why the split: schema 1 keyed on `(model, messages, params)` only, so sync and
Batch-API responses shared a key and whichever ran last won. Schema 2 adds the
serving backend, the call kind and a schema version, stores raw responses, and
never caches failures. `scripts/migrate_cache.py` documents the mapping between
them.

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
| explainer, measurable drivers | ~54% of points; the other ~46% are `undetermined` |
| explainer faithfulness | agreement ~0.356 on those points, vs a ~0.287 permutation null |
| driver vs correctness | speaker-driven sits *on* its own baseline (~+0.00); `undetermined` is ~+0.20 above its own |

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

**(b2) Ship the calibrator with a predictor**

`predict()` returns RAW probabilities thresholded at 0.5 unless you give it a
calibrator. The fitted calibrator and tuned threshold are written as versioned
artifacts by the evaluation entrypoint:

```python
from truthclf.predictors import ZeroShotPredictor
pred = ZeroShotPredictor(
    model="google/gemma-4-31B-it", variant="full", client=client,
    calibrator="results/calibrators/zero_shot_decision_baseline.json")
res = pred.predict(points)      # calibrated probabilities, tuned threshold
```

The artifact records the calibrator parameters, the threshold and its objective,
the split it was fitted on, both candidates' validation NLL, the margin-rule CI,
and the model id. Applying it to a different model raises
`CalibratorModelMismatch` — a calibrator is specific to the probability scale of
the model that produced it, and a silent mismatch still looks like probabilities.

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

*When occlusion measures nothing.* On **137 of 300** sampled points (**45.7%**) no
occluded field moves the probability at all. Those are reported as
`driver: "undetermined"`, not as statement-driven. The distinction is not
cosmetic: `statement` is simultaneously the fallback driver *and* the fallback
rationale reference, so folding these points into it both inflates that category
and scores them as rationale/occlusion agreement — absence on both sides counted
as concordance. Score-mode elicitation emits only ~17 distinct values, so a zero
delta means "below this measurement's resolution", not "the claim's content
decided it". With the undetermined points separated, the driver distribution is
speaker-family 32.0%, other metadata 15.3%, statement 7.0%, undetermined 45.7%.

*Driver versus correctness.* Each driver subset is compared against **its own**
majority-class rate, because the subsets differ in class balance and a single
global baseline would manufacture differences. **Six subsets were tested**, and
only one — `undetermined`, at **+0.204 [+0.088, +0.285]** on n = 137 — has an
interval excluding zero. No multiplicity correction is applied; at six tests a
single nominal 95% interval is not the same as a 95% guarantee, and the effect is
reported as large-and-well-powered rather than as a significance claim.
Speaker-driven predictions sit exactly on their own baseline (+0.000 [−0.141,
+0.115], n = 78). Statement-driven proper (n = 21) cannot be separated from
its baseline in either direction.

**The undetermined result is selection, not causation.** A statement whose
predicted probability does not move when metadata is removed is, by construction,
one the model judges on the claim alone — and such statements are plausibly the
clear-cut ones. Metadata-irrelevance and easiness are most likely the same
property observed twice, rather than one producing the other. Nothing here
supports "removing metadata would make the model more accurate".

*Faithfulness, restated.* Agreement is scored only on the 163 points with a
measurable driver: **0.356 [0.282, 0.429]** against a permutation null of
**0.287 [0.233, 0.344]**, **p = 0.014**. On that subset the rationales carry
*some* detectable information about which field actually drove the prediction —
they are not pure post-hoc narration. Two limits belong with that number: the
restriction to measurable-driver points was decided after inspecting the data
(it followed from a defect fix, not from hypothesis search), and n = 163 with a
single test. It is evidence of a weak signal, not a faithfulness guarantee. The
earlier figure — 0.457 against a 0.436 null, p = 0.19, "at chance" — was computed
over all 300 points including the 137 where there was nothing to agree about;
both the observed rate and the null were inflated by the shared `statement`
default.

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
scripts/         every documented command lives here; run them from the project root
  analyze_dataset.py       exploratory data analysis (no model calls)
  estimate_cost.py         token/cost estimate before any full run (no model calls)
  scale_check.py           200-row model bake-off across candidate bases
  benchmark_zeroshot.py    zero-shot model comparison, full set, bootstrap CIs
  zeroshot_baseline.py     calibrated zero-shot logprob baseline on the test split
  evaluate_zeroshot.py     threshold tuning + calibration + ablation + abstention
  finetune_prep.py         build/validate/upload the SFT data
  finetune_run.py          launch the LoRA SFT job and poll to completion
  evaluate_finetuned.py    fine-tuned vs zero-shot paired comparison
  run_explainer.py         occlusion + rationale + cross-check, aggregate report
  explain_analysis.py      explainer follow-ups from cache (no new API calls)
  migrate_cache.py         one-shot schema-1 -> schema-2 response-cache migration
  refetch_quarantined.py   refetch cache entries the migration quarantined
  regenerate_results.py    recompute the adopted record; writes the results JSON
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

### The tools image carries the dataset

The build produces two images, and they have different handling requirements:

| image | contains | may be pushed to |
|---|---|---|
| `truthclf-tools` | `truthclf`, the data-science stack, **`data.csv`**, the recorded fine-tuned probabilities and their statement-identity map, the fitted calibrators | **a private registry only**, inside the same trust boundary as the data |
| `truthclf-agent` | the four A2A agents, `a2a-sdk`, `mcp`, and their web stack — no dataset, no model artifacts, no `truthclf` | the same private registry; it carries no challenge data, but there is no reason to publish it either |

`data.csv` is baked into the tools image rather than mounted, so a container
behaves identically wherever it runs and there is no volume to forget. The cost
is that **the image is as confidential as the dataset**: it must never reach a
public registry, and pushing it anywhere outside the project that already holds
the data is the same disclosure as publishing the CSV.

### The agent image cannot import `truthclf`

This is checked, not asserted. The agent stage installs only the `agents`
dependency group with `--no-install-project`, copies only `truthclf_agents/`,
and then **fails the build** if `truthclf`, `truthclf_mcp`, or any of
`sklearn`, `scipy`, `pandas`, `statsmodels`, `numpy`, `together`, `tiktoken` or
`diskcache` turns out to be importable. `tests/test_agent_isolation.py` asserts
the same property from the source on every test run, so the mistake is caught
when it is made rather than when an image is next built.

The point is not image size, though it is 3.5× smaller. It is that "the agents
are pure MCP clients" stops being a convention someone has to remember: an agent
that starts calling into `truthclf` directly — turning a tool call into an
in-process function call — stops producing a buildable image.
