# Vertex AI migration plan

Moving both predictors off Together AI and onto Vertex AI. The zero-shot
baseline and the fine-tuning are both re-run on GCP; the agent network then runs
against the re-derived record.

Written 2026-08-14, before any work has started. Nothing here has been executed.

---

## 1. Target model, and a constraint on the evaluation design

**Target: `gemini-2.5-flash`.**

The choice is forced rather than preferred, and the reason belongs in the record
because it constrains how long the evaluation stays reproducible:

| tier | status | logprobs |
|---|---|---|
| Gemini 1.5 | shut down — returns 404 | n/a |
| Gemini 2.0 | shut down — returns 404 | n/a |
| **Gemini 2.5** | **available; retires 2026-10-16** | **supported** |
| Gemini 3.x | available | **returns 400 for logprobs** |

**The 2.5 family is the only tier that both exists and supports logprobs.** It is
therefore the only tier on which our evaluation design — a continuous
probability from first-token logprobs, feeding calibration, ECE, Brier, ROC-AUC
and a tuned decision threshold — can be reproduced at all.

### What this means, stated plainly

**The reported record will become unreproducible on 2026-10-16.** That is a
documented constraint of the evaluation design, not an oversight, and it should
be written into the README rather than discovered by whoever next runs the
regeneration command. Three consequences follow:

1. **The offline replay path becomes the record.** As on Together, the archived
   responses and the calibrator artifacts are what make the numbers reproducible
   after the model is gone. That path already exists
   (`scripts/regenerate_results.py --source archive`) and must be preserved,
   with the cached responses committed, not gitignored.
2. **A live refetch is only possible before the retirement date.** The "a live
   refetch reproduced these to within 0.003" style of cross-check has an expiry.
3. **Any successor model is a new record, not a refresh.** When 2.5 retires,
   either the evaluation moves to a hard-label design without calibration, or it
   waits for a tier that restores logprobs. That is a decision to take then, with
   evidence, not a migration to schedule now.

**Do not treat the retirement date as a reason to rush.** It is a reason to make
the offline replay airtight before it arrives.

---

## 2. Two things to verify empirically before spending anything

The Together phase cost roughly $3 on a Gemma adapter that turned out to be
unservable, then ~$0.05 on a deliberate throwaway Qwen adapter that established
the same limitation for a fraction of the price. **The cheap probe should have
come first.** It does this time.

### Probe A — do logprobs work on a *tuned* `gemini-2.5-flash`?

Base-model logprob support does not imply tuned-model logprob support. This is
the single question the whole evaluation design rests on, and the peer solution
reviewed for GCP conventions does not answer it — it uses no logprobs anywhere.

**Method.** Run a tuning job on **~100 rows** drawn from the existing training
split, minimum epochs. Then, against the resulting endpoint, request logprobs on
one prompt and confirm:

- the call returns 200 rather than 400
- log probabilities are present for **candidate tokens at the first position**
- both `True` and `False` appear among the returned candidates, or enough of the
  top-k to compute a softmax over the two

**Decision rule.** If logprobs are unavailable on the tuned model, **stop and
report before running the full tuning job.** The fallback is score-mode
elicitation, which is measurably worse (it costs accuracy and has ~17 distinct
values, which is what made 45.7% of occlusion measurements degenerate) and would
mean redesigning the zero-shot-versus-fine-tuned comparison. That is a
conversation, not an implementation detail.

### Probe B — does `True` tokenise as a single token?

The SFT target is one token. If the new tokenizer splits `True` or `False`,
first-token logprobs no longer identify the label and the elicitation is wrong
in a way that produces plausible numbers.

**Method.** Tokenise both target strings with the model's own tokenizer, in the
exact form they appear in the training records (including any leading space).
Confirm each is a single token, and that the two are distinct.

**Decision rule.** If either splits, the target must change — a single distinct
token for each class — and the SFT data is regenerated before tuning.

### Cost gate

Both probes ride on one ~100-row tuning run. **Print the estimate and get an
explicit go-ahead before submitting it**, per the working agreement. Report both
results before proposing the full run.

---

## 3. Scope of the code change

**`truthclf/llm.py` gains a Vertex backend. Nothing above it changes.**

### What changes

| file | change |
|---|---|
| `truthclf/llm.py` | a Vertex client alongside `TogetherClient` / `TogetherBatchClient`: `score`, `classify`, `complete` against `google-genai` in Vertex mode. `make_client` gains a backend selector. Cache keys already embed the backend, so Together and Vertex responses cannot collide |
| `truthclf/llm.py` (pricing) | `PRICING` and the token-estimate path gain the Gemini rates |
| `scripts/*` | the hard-coded base-model id, in the places `docs/decisions.md` records it appearing |
| `ft_data/*.jsonl` | regenerated in Vertex SFT format — see §5 |
| `pyproject.toml` | `google-genai` added; `together` retained only while the archive replay is still exercised |

### What does not change

Everything above the client is provider-agnostic and should be touched only if a
test forces it:

- `truthclf/predictors/` — `ZeroShotPredictor`, `FinetunedPredictor`,
  `PredictionResult`, `validate_points`, `measured`
- `truthclf/calibration.py`, `truthclf/threshold.py`, `truthclf/evaluation.py` —
  including the schema-3 `DecisionArtifact` keyed by (model, elicitation), which
  already handles a new model id correctly
- `truthclf/metrics.py`, `truthclf/explain.py`, `truthclf/data.py`
- **all of `truthclf_mcp/`** — the tools call predictors, not providers
- **all of `truthclf_agents/`** — the agents cannot import `truthclf` at all,
  and the build enforces it. **The entire agent and MCP layer is unaffected by
  this migration.** That is the payoff from the boundary.

The one MCP-layer consequence is in `model_tools.py`: the fine-tuned path stops
needing `fine_tuned_source="cached"` as its default. See
`docs/deployment-plan.md` §5.

---

## 4. The record is re-derived, not migrated

**Every number in the adopted record was produced by `gemini-2.5-flash` on
Together. A new base model means a new record.** Nothing carries over:
accuracy, ECE, the +0.031 fine-tuning effect, the calibrators, the driver
distribution, the agreement statistic.

This is a deliberate re-derivation and does not violate the
record-does-not-move-silently rule — but it is the one case where the diff will
be "everything moved", so the handling has to be explicit.

### How to keep both

**Do not overwrite the Together record. Label it and keep it.**

- Preserve `results/`, `ft_eval_results.json`, `explain_results.json`,
  `ft_eval_cache.json`, `ft_eval_identity.json` and `.llm_cache.json` as the
  **Together-era record**, clearly marked with the base model and the date.
- Write the Vertex record alongside it, not over it.
- The README reports the Vertex record as current and the Together record as a
  historical comparison, with the base model named in both cases.

Two reasons this matters beyond tidiness. The Together record is the only
evidence for several findings that cost real effort to establish — the calibrator
selection behaviour, the ECE binning finding, the `pr_auc` order-dependence, the
explainer's `undetermined` share. And a like-for-like comparison across two
providers is itself a result worth having, provided neither is presented as the
other.

### What must be re-established, not assumed

- **The fine-tuning effect.** +0.031 [+0.019, +0.043] was measured for a Gemma
  LoRA against a matched Gemma baseline. A Gemini 2.5 tune may show more, less,
  or nothing. **Report whatever it shows, with its interval and McNemar.**
- **Calibrator selection.** Platt won all three runs on validation NLL by a
  margin. On a new model the parsimony rule may fire. Re-run, do not assume.
- **The ECE occupied-bin counts.** Model-dependent.
- **The explainer's `undetermined` share.** 45.7% is a property of score-mode
  resolution on the old model. It may change substantially, and the agreement
  finding (0.356 vs 0.287 null, p = 0.014) must be recomputed with a matched
  null on whatever subset results.
- **The speaker-shortcut finding.** Re-measure against each subset's own
  baseline.

### What does carry over

The methodology, and it should not be re-litigated: the middle-split
binarisation, the speaker-disjoint three-way split with repeated-statement
grouping and contradiction dropping, calibrator selection on validation NLL with
a parsimony margin, equal-mass ECE reported with occupied-bin counts, paired
bootstrap intervals, McNemar for the paired comparison, and the permutation null
for the agreement statistic. The splits themselves are deterministic from
`data.csv` and are unchanged.

---

## 5. SFT data format

`ft_data/train.jsonl` (5,710 rows) and `val.jsonl` (1,917 rows) are
base-agnostic text and are reusable, but the envelope changes.

**Current — Together conversational SFT:**

```json
{"messages": [{"role": "system",    "content": "<system prompt>"},
              {"role": "user",      "content": "<full-metadata prompt>"},
              {"role": "assistant", "content": "True"}]}
```

**Vertex SFT:**

```json
{"systemInstruction": {"role": "system", "parts": [{"text": "<system prompt>"}]},
 "contents": [{"role": "user",  "parts": [{"text": "<full-metadata prompt>"}]},
              {"role": "model", "parts": [{"text": "True"}]}]}
```

Three points on the conversion:

- `system` moves out of the message list into a top-level `systemInstruction`.
- `assistant` becomes `model`.
- Content becomes `parts: [{"text": ...}]` rather than a bare string.

**Loss masking matches what we already had.** Vertex masks the prompt tokens and
computes loss on the completion tokens only, which is the behaviour
`train_on_inputs="auto"` gave us on Together. **No change in training semantics**
— the comparison across providers is not confounded by the masking.

`data.write_sft_jsonl` / `to_sft_record` are the single place this is generated,
so the conversion is one function and the prompt strings are untouched. **The
prompt must stay byte-identical between training and inference**, as it is
today — the decision-mode prompt is shared by `to_sft_record` and the matched
zero-shot reference, and that identity is what makes the comparison
apples-to-apples.

Staging: the JSONL goes to GCS and the tuning job reads a `gs://` URI.

---

## 6. Order of work

Each step reports before the next begins.

1. **Vertex client in `truthclf/llm.py`**, behind the existing backend selector.
   Verify against the base `gemini-2.5-flash`: logprobs on the base model, cache
   keys distinct from Together's, the record still regenerates from the archive
   unchanged. No new spend beyond a handful of calls.
2. **Probes A and B** on a ~100-row tuning run, estimate printed first. **Report
   both before proposing the full run.**
3. **Zero-shot baseline** on the test split with the new model. Full calibrated
   evaluation, new calibrator artifact, intervals throughout.
4. **Full SFT run**, estimate printed first.
5. **Fine-tuned evaluation**, paired against the zero-shot baseline on the same
   split with matched elicitation. New record written alongside the Together one.
6. **Explainer re-run**, including the `undetermined` share and a matched
   permutation null.
7. **Point the MCP layer at the new artifacts.** Expected to be configuration:
   model ids, calibrator files, and the fine-tuned serving mode.

Steps 1 and 2 are cheap and answer the questions that could invalidate steps
3–7. Nothing after step 2 should start while either probe is unresolved.
