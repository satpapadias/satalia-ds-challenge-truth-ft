# Phase 2 design record

For each component: the challenge-spec requirement it satisfies (quoted), the
library APIs it calls with their import paths, why those rather than the
alternatives, and the things that are easy to get wrong on a rewrite.

Spec quotes are from `Data Science Challenge - Truth.pdf`, pages 1–4.

---

## 0. Architecture

Agents are pure MCP clients; no agent imports `truthclf`. The two MCP servers
are the only processes that do. They are split on the credential boundary:

| server | tools | holds | reaches the network |
|---|---|---|---|
| **model-tools** | `predict`, `explain` | provider credential, response cache, calibrator artifacts | yes |
| **data-tools** | `dataset`, `metrics`, `retrieval` | `data.csv`, TF-IDF index | no |

Both speak streamable HTTP rather than stdio, so the local topology is the same
one that containerises.

`retrieval` is k-NN over the **train split only**. That is a leakage constraint,
not a preference.

**Fine-tuning is deliberately not an MCP tool.** It blocks with no timeout, its
fixed working directory is overwritten by a concurrent call, it spends money,
and the adapter it produces cannot be served. It stays a command-line path.

### Local endpoints

```
data-tools    http://127.0.0.1:8081/mcp
model-tools   http://127.0.0.1:8082/mcp
```

```bash
python -m truthclf_mcp.data_tools  --host 127.0.0.1 --port 8081
python -m truthclf_mcp.model_tools --host 127.0.0.1 --port 8082
```

---

## 1. Prerequisite fix: elicitation is half of a calibrator's identity

Found while specifying the `predict` interface, and fixed before any server code
was written, because every calibrated response depends on it.

`DecisionArtifact` recorded `model` but not the elicitation mode, and
`check_model` verified only `model`. Both zero-shot artifacts carry the model id
`google/gemma-4-31B-it` and differ only by elicitation:

| artifact | elicitation | A | B | threshold |
|---|---|---|---|---|
| `zero_shot_decision_baseline` | logprob | 0.0478 | 0.0922 | 0.5438 |
| `zero_shot_score_secondary` | score | 0.0412 | 0.1901 | 0.5515 |

So either artifact was accepted on either predictor. A logprob predictor handed
the score-mode calibrator applies a mapping fitted on a different probability
scale and decides at a threshold tuned for that scale — no error, no warning,
and output that still looks like probabilities.

**Fix.** `elicitation` added to the artifact and validated at construction;
`check_model(model, elicitation)` verifies both, with `elicitation` positional
and required, since a default would restore the hole; `build_artifact` takes it
positionally; `ZeroShotPredictor` passes its own mode; schema 2 → 3, and
schema-2 files are refused rather than defaulted, because no default can be
correct when the field's purpose is to say which of two scales the file was
fitted on.

**Record re-verified**: 14,530 numeric values across `summary.json`,
`curves.json`, `ft_eval_results.json` and `explain_results.json`, **0 moved**.
The only change under `results/` is the new field plus the schema bump. Test
count 132 → 138.

---

## 2. The JSON ↔ `Row` boundary — `truthclf_mcp/adapter.py`

### Requirement

> "**points** — a set of statements with their attributes (excluding the label);
> optional **labels** — the correct labels for those points."

> "Each component must be able to efficiently process a set of data points (not
> just one) and return its outputs for all of them."

### Problem and resolution

`truthclf` works with `Row` dataclasses and contains no deserialisation layer.
MCP tools receive JSON. `adapter.to_row` is the single bridge; `truthclf` is
unmodified.

Input is checked in two passes. The `Point` pydantic model becomes each tool's
published JSON schema and is enforced by the framework before a handler runs —
structure only. Then `validate_points` checks meaning. **No second validator was
written**: the existing one is called on the constructed rows, and its
exceptions are surfaced verbatim because they are already precise.

The second pass is not redundant. A JSON schema can require that `statement` is
a string but not that `"   "` is unusable, and the failure it prevents is silent
rather than structural.

Missing *metadata* is not an error — every metadata field defaults to `""`,
which is what the CSV loader produces for an absent column and what prompt
assembly renders as `[unknown]`. Missing *statement* is an error.

### APIs called

```python
from truthclf.data import (Row, clean_text, normalized_statement_key,
                           binarize, LABELS_6)
from truthclf.predictors.base import (validate_points, InvalidPointError,
                                      LabelCountMismatch)
```

- `Row(row_id, label, statement, subjects, speaker_name, speaker_job, speaker_state, speaker_affiliation, statement_context, statement_clean="", norm_key="")`
- `clean_text(s: str) -> str`
- `normalized_statement_key(statement: str) -> str`
- `binarize(label: str, scheme: str = "primary") -> int`
- `validate_points(points, labels=None) -> None`

### Why these rather than the alternatives

- **`validate_points` rather than schema-only checking** — as above; the failure
  mode is semantic and silent.
- **`clean_text` / `normalized_statement_key` rather than recomputing** — see
  trap 1.
- **`binarize` rather than a local mapping table** — the six-to-two mapping is a
  deliberate design choice with a documented sensitivity variant. A second copy
  is a second place for it to drift.

### What is easy to get wrong

1. **Leaving `statement_clean` empty.** `Row` defaults it to `""` and prompt
   assembly uses `statement_clean or statement`, so an adapter that skips it
   sends the *raw* statement to the model. Every probability shifts, nothing
   raises, and results stop matching the recorded evaluation. Pinned by
   `tests/test_mcp_adapter.py::test_built_row_matches_data_load_exactly`, which
   asserts full `Row` equality against `data.load` over a spread of real rows,
   plus a companion test restricted to rows that cleaning demonstrably changes —
   without it the first test would pass even for an adapter that skipped
   cleaning entirely.
2. **The `label` placeholder.** `Row.label` is required with no default, but
   prediction inputs arrive without ground truth. The adapter passes `""`, which
   is safe **only because prediction and explanation never call `Row.y()`** —
   `y()` looks the label up in the binarisation table and raises `KeyError` on
   `""`. Ground truth reaches the metrics through the separate labels array.
3. **Assuming labels are one type.** The wire accepts 0/1 or any of the six
   human labels; strings go through `binarize`. A caller sending
   `["mostly-true", ...]` gets the correct mapping rather than a crash or a
   silent miscount.

---

## 3. `predict` — one tool, explicit model choice

### Requirement

> "**ENDPOINT** `predict(points, labels=None)` … Output: a True/False prediction
> for each point. If labels are provided, it additionally returns performance
> metrics for its predictions on the set."

> "The same signature and behavior as Component 1's predict, but backed by the
> fine-tuned model, **so the two predictors are interchangeable**."

### One tool with a `model` enum, not two tools

Interchangeability is the spec's own claim about these two components, so one
tool with a discriminator keeps it enforced rather than letting two response
shapes drift apart. The output shape is identical either way, and tool-list size
is the real cost in a client's context: choosing between two similarly-described
tools by prose is a worse selection problem than filling one enum field.

`model` is **required with no default**, so a caller that meant `fine_tuned`
cannot receive zero-shot results by omission.

Two tools would have allowed the fine-tuned one to be absent when unservable,
which is arguably more honest — rejected because tool availability would then
depend on live provider state, so the client's tool list would change underneath
it. An explicit typed error is more predictable than a disappearing tool.

### Serving the fine-tuned model

The provider cannot serve this adapter without a dedicated endpoint, and none is
running. `fine_tuned_source` defaults to `"cached"`, replaying the per-row
probabilities recorded during the endpoint evaluation. `provenance.served_by` is
a required output field and reads `cached_replay`, so a caller cannot mistake it
for a live call. `"live"` issues the request anyway and maps the provider's
refusal onto `FineTunedModelNotServable`.

**There is no fallback edge in either direction.** A cached miss errors; a live
failure errors. The only way to get zero-shot numbers is to ask for them.

### APIs called

```python
from truthclf import llm, metrics as M, prompts
from truthclf.predictors import ZeroShotPredictor, FinetunedPredictor
from truthclf.predictors import finetuned as FT
from truthclf.evaluation import DecisionArtifact

llm.make_client(model, cache=None, backend="sync", **overrides)
llm.count_tokens(text) -> int
llm.estimate_cost(n_input_tokens, n_output_tokens, model) -> float
prompts.build_messages(row, variant, mode)
ZeroShotPredictor(model, variant, threshold, client, seed, use_logprobs,
                  neutral_score, calibrator).predict(points, labels=None)
FT.load_cached_probs(rows, path) -> list[float]
DecisionArtifact.load(path); .decide(probs) -> (calibrated, preds)
M.metric_bundle(y_true, preds, probs); M.ece_bin_report(y_true, y_prob, ...)
```

### What is easy to get wrong

1. **Treating the stored fine-tuned probabilities as calibrated.** They are raw
   model output — the file contains values like `3.95e-08`. The evaluation
   pipeline calibrates them downstream, and so must this server. Serving them
   directly would report a wildly overconfident calibration error and decide at
   0.5 instead of 0.5152.
2. **Letting a cached miss become a live call, or a live failure become a cached
   replay.** Either makes `served_by` a lie.
3. **Serving stored probabilities under `elicitation="score"`.** They were
   produced by single-token logprob elicitation. Relabelling their scale is
   refused.

### Verified

Paging the full test split through `predict(model="fine_tuned",
fine_tuned_source="cached")` and scoring it with the `metrics` tool reproduces
the recorded evaluation exactly:

| metric | via MCP | record | Δ |
|---|---|---|---|
| accuracy | 0.699146 | 0.699146 | 0 |
| balanced accuracy | 0.693617 | 0.693617 | 0 |
| macro F1 | 0.694089 | 0.694089 | 0 |
| Brier | 0.202724 | 0.202724 | 0 |
| ECE | 0.051621 | 0.051621 | 0 |

n = 1,991, threshold 0.515155, ECE bins occupied 10/10. Accuracy against the
split's own majority-class baseline: **+0.1652 [+0.1371, +0.1924]**.

---

## 4. Calibrators — loaded once at start-up

`_Calibrators` reads every artifact in the calibrator directory at import and
keys them by **(model, elicitation)**.

Start-up rather than per request, because a missing, stale-schema or duplicated
artifact is a deployment fault, and discovering it on whichever request happens
to arrive first is strictly worse than refusing to start. Cost is not the reason
— the files are a few hundred bytes.

The key needs both parts for the reason in section 1: a model id alone does not
identify a probability scale. The artifact's own `check_model` is a second line
of defence, not the selector.

`calibrated` defaults to true and is echoed in the response. Turning it off is a
legitimate diagnostic but silently moves the threshold from 0.5438 to 0.5, so
the response says which was used and warns.

Three artifacts load: `('google/gemma-4-31B-it', 'logprob')`,
`('google/gemma-4-31B-it', 'score')`, and
`('makisntpap_17e5/…-c7afbf0d', 'logprob')`.

---

## 5. `explain`

### Requirement

> "**ENDPOINT** `explain(model, points, labels=None)` … Output: for each point,
> the model's prediction and an explanation of the factors that drive that
> prediction. If labels are provided, it additionally returns performance
> metrics for the underlying model's predictions on the set."

### APIs called

```python
from truthclf import explain as X
X.explain(model, points, labels=None, with_rationale=True, threshold=0.5,
          driver_eps=0.05, max_parse_failure_rate=0.0) -> dict
X.aggregate(result) -> dict          # "field_table" is a pandas DataFrame
```

`aggregate()["field_table"]` is a DataFrame and is not JSON-serialisable; it is
converted with `.to_dict(orient="records")` at the boundary.

### Defaults match the recorded explainer run

Score-mode, uncalibrated, threshold 0.5 — the configuration the reported
explainer numbers came from. Score mode is required in practice: occlusion reads
*graded* probability shifts, and the logprob path is near-saturated, so its
deltas would collapse to 0/1 jumps.

Calibration is available but warns, because this tool's `threshold` is
independent of the calibrator's fitted threshold, so a calibrated run at 0.5
does not make the decisions the calibrated predictor would make.

### Fine-tuned + cached is refused outright

The stored probabilities are keyed on `row_id` alone, and occlusion builds
field-ablated copies that keep the **same `row_id`**. Serving occlusions from
the store would return the base row's probability for every variant: all deltas
0.0, every point attributed to the statement, and a complete, plausible-looking
driver distribution that is an artifact of the join key rather than a
measurement. A response cache cannot answer counterfactual queries, so this
raises `CounterfactualNotAvailable` rather than warning.

Every response carries a fixed interpretation note: the rationale-to-driver
agreement rate is not distinguishable from chance (permutation null 0.436,
95% range [0.393, 0.480], observed 0.457, p = 0.19), and should be read as
evidence of absent signal rather than as a measured faithfulness rate.

### What is easy to get wrong

1. **The sevenfold request amplification.** Each point costs six occlusion calls
   plus one rationale call. Fifty points is 350 provider requests. The ceiling is
   set on the amplified count.
2. **The cached-fine-tuned trap above.**
3. **Hard-coding `max_parse_failure_rate`.** The default of 0.0 fails on any
   neutral fallback. The project's own explainer sample does not pass that
   default — its 1,800 occlusion rows contain a handful of genuine model
   refusals on statements too short to carry a claim. Hard-coding the strict
   value would refuse the very sample the record was built from; hard-coding the
   loose one would weaken the check for every other caller. It is a parameter.

---

## 6. `dataset`, `metrics`, `retrieval`

### Requirements

`metrics` serves the requirement common to both endpoints:

> "for `predict()` and `explain()` endpoints, if the caller also supplies the
> correct labels for those points, the component must return appropriate
> performance metrics for its own predictions on that set."

`dataset` serves the fine-tuning deliverable's data stage —

> "performs the data preparation and train/validation split"

— and the presentation's

> "the held-out data and protocol you used to compare the two predictors fairly."

`retrieval` is **not a spec requirement.** It is supporting infrastructure,
recorded as such rather than back-fitted to one.

### `dataset`

```python
from truthclf.data import (load, clean_dataset, speaker_disjoint_3way,
                           dev_subset, class_balance, speakers_cross, normkeys_cross)
```

The corpus is loaded, cleaned and split once at start-up. Split parameters are
fixed to the reported evaluation's values and echoed in every response, because
changing any of them changes split membership and so every downstream number.
Requesting a different scheme is refused rather than served, since the split
stratifies on each group's majority label under a scheme.

Sampling reuses `dev_subset` rather than a local sampler, so a sample is
reproducible by seed and matches what the analysis scripts draw. Unknown
`row_ids` raise `RowNotInSplit` rather than being dropped.

Leakage counts (`speakers_cross`, `normkeys_cross`, both 0) are computed at
start-up and returned with every response rather than assumed.

### `metrics`

```python
from truthclf import metrics as M
M.metric_bundle(y_true, preds, probs) -> dict
M.bootstrap_bundle(y_true, preds, probs, n_boot=1000, seed=0, alpha=0.05)
M.ece_bin_report(y_true, y_prob, n_bins=10, strategy="quantile")
M.paired_accuracy_diff(y_true, pred_a, pred_b, n_boot=2000, seed=0, alpha=0.05)
M.mcnemar(y_true, pred_a, pred_b) -> dict
```

Delegates entirely; nothing is reimplemented. Three positions worth knowing:

- **Bootstrap intervals default to on.** A tool whose default output is eleven
  bare floats is a bare-number generator. `bootstrap_bundle` shares one resample
  across all metrics per draw, so this is a single pass rather than eleven.
  Turning it off adds a warning to the response.
- **The ECE occupied-bin count is always returned.** A 2-bin and a 10-bin ECE
  are not comparable quantities.
- **The baseline is the set's own majority-class rate**, never a global one.
  Subsets differ in class balance, so a single global figure would manufacture
  differences that are not there.

Warnings raised during the bootstrap — notably the one for dropped degenerate
resamples — are captured and forwarded, because an interval computed over fewer
draws is narrower and otherwise indistinguishable from a confident one.

Undefined metrics are returned as `null`, never `0.0`: `0.0` is indistinguishable
from a genuinely poor score, which is the distinction NaN exists to preserve.

### `retrieval`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                strip_accents="unicode", lowercase=True, min_df=1)
NearestNeighbors(metric="cosine", algorithm="brute")
```

**TF-IDF + cosine rather than sentence embeddings**: scikit-learn is already a
required dependency, it is deterministic and runs offline, and a hosted
embedding model would put an API credential on the server that is otherwise free
of one — collapsing the boundary the two-server split exists to draw.

Index: 5,710 training statements, 63,296 features. Exact brute-force search, as
the split is small enough that an approximate index would trade correctness for
an unneeded gain. The **cleaned** text is indexed, so neighbours are neighbours
in the text the predictor is actually shown, and queries are cleaned the same
way before lookup.

Train-only is enforced twice: the index is built from the training split, and
every result is checked against the validation and test row-id sets, raising
`LeakageAssertionFailed` with no partial results if one ever appears. The split
already guarantees no repeated statement crosses it, so a validation or test row
cannot exactly match a training one; the per-neighbour
`exact_norm_key_match` flag catches a caller-supplied query that is itself a
training statement, which is self-retrieval rather than evidence.

---

## 7. Batch ceilings and spend

Every tool takes a batch. Exceeding a ceiling raises `BatchTooLarge` naming the
limit and the actual count. **Nothing is ever truncated** — metrics over a
silently shortened set answer a different question than the one asked.

| tool | unit | ceiling | provider requests at ceiling |
|---|---|---|---|
| `predict` | points | 500 | 500 |
| `explain` | points | 50 | 350 |
| `dataset` | rows per page | 1,000 | 0 |
| `metrics` | array length | 100,000 | 0 |
| `retrieval` | queries | 500 | 0 |

Each is overridable by environment variable. Enforcement lives in the handlers
rather than as a pydantic list length, so the failure is the named error rather
than a generic validation message; the limit is also published in the relevant
field description.

**These are derived, not measured.** For `predict`: the client runs 16 workers
with one request per row, and recorded per-call latency is about 0.4 s, giving
roughly 40 requests/second and about 13 s for 500 all-miss points. The one
measured anchor is a batch refetch of 6,075 rows in 17.2 minutes at about $0.33,
i.e. roughly $0.000054 per row, putting 500 uncached points near $0.027. They
should be measured through this transport before being treated as fixed.

### The spend gate is weaker than the one it replaces

`predict` carries `estimate_only` (returns the token and cost estimate with no
provider call) and `max_live_calls` (refuses rather than exceeding).

**`max_live_calls` is a budget ceiling, not consent.** The project's operating
rule is that no paid operation runs without a cost estimate and an explicit
go-ahead. Once an agent holds `predict`, there is no human in the loop to give
one: the agent both sets the ceiling and decides to call. A ceiling bounds the
damage from a runaway loop; it does not reconstitute approval, and no parameter
on this schema can. **This is a real weakening that the architecture forces,
not something the schema solves.** Anything stronger has to live outside the
tool — a per-deployment spend cap the agent cannot raise, or an approval step in
the client. Recorded here rather than papered over.

---

## 8. Error handling

Errors crossing the MCP boundary carry their class name, because the class is
the contract. `truthclf_mcp/errors.py` maps a fixed list of caller-facing
exceptions onto `ToolError` with the name preserved and the message verbatim.

Unrecognised exceptions are deliberately **not** caught. A caller should be able
to tell "you sent me a bad point" from "this server has a bug", and a blanket
handler erases that distinction.

Verified live: `InvalidPointError` (names the offending row), `LabelCountMismatch`,
`BatchTooLarge`, `FineTunedRowNotCached`, `CounterfactualNotAvailable`,
`RowNotInSplit`, the empty-query rejection, the live-budget refusal, and the
framework's own rejection of `predict` without a `model`.

---

## 9. Known gaps

- **No published output schemas.** Handlers are annotated `-> dict`, which
  carries no field structure, so results arrive as JSON text rather than
  structured content. Clients and Inspector render them fine, but an agent gets
  no machine-readable shape. Defining pydantic output models is the first
  follow-up.
- Ceilings are derived rather than measured through this transport (section 7).
- The response cache is a local SQLite store: safe across processes on one host,
  not across hosts.
- `explain` runs synchronously; at the ceiling that is 350 provider requests in
  one request handler.

---

## 10. Changelog

- **Schemas agreed**; `fine_tune` excluded from MCP and kept as a command-line
  path; retrieval settled on TF-IDF + cosine; metrics defaults settled
  (bootstrap on, occupied-bin count always returned).
- **Calibrator artifact schema 2 → 3** — elicitation added to the identity;
  14,530 recorded values re-verified unmoved; tests 132 → 138.
- **Adapter built** with the `data.load` equality test; **data-tools** and
  **model-tools** built and verified end to end over streamable HTTP; the
  recorded fine-tuned test-split evaluation reproduced exactly through the
  MCP path.
