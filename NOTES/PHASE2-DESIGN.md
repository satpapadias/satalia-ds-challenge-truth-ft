# Phase 2 design record

For each component: the challenge-spec requirement it satisfies (quoted), the
library APIs it calls with their import paths, why those rather than the
alternatives, and the things that are easy to get wrong on a rewrite.

Part 1 spec quotes are from `Data Science Challenge - Truth.pdf`, pages 1–4.
Part 2 spec quotes are from `internal/Data Science Challenge - Truth - Full.pdf`,
pages 5–8, which is the authoritative brief for both parts.

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

### Live calls go through one builder and one translation site

`_build_predictor(model, elicitation, calibrator=None)` constructs the predictor
for both tools, and `_live_call(model)` is the single context manager that
translates a provider refusal to serve the fine-tuned model into
`FineTunedModelNotServable`.

Two reasons this is not merely tidier:

- **The same failure must report the same way.** `predict` scores a batch and
  `explain` drives occlusion through the explainer, so they reach the model by
  different routes. With the translation attached to one route, a caller sees
  two different errors for one cause depending on which tool they happened to
  use.
- **The fine-tuned choice yields a `FinetunedPredictor`**, not a zero-shot
  predictor aimed at the fine-tuned model id. The two behave identically today,
  since `FinetunedPredictor` delegates to a zero-shot predictor bound to its
  served model — but the serving configuration belongs on the type that owns it.
  When the model is served from somewhere with a live endpoint, that path
  becomes one that should succeed, and it should be built correctly before then
  rather than corrected under a working endpoint.

The translation keys on the fine-tuned model *and* the provider's message text,
so an unrelated failure (a reset connection, a zero-shot error carrying similar
words) is not relabelled as unservable. Pinned by
`tests/test_mcp_model_tools.py`, which asserts both tools name the same error
class and that neither of those two cases is mislabelled.

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

### `model` defaults to `zero_shot`; `predict`'s stays required

The two tools differ deliberately.

`predict` has two working paths — live zero-shot and the stored fine-tuned
replay — so omitting the choice must not silently pick one. `model` stays
required there.

`explain` has one. The fine-tuned model has no live endpoint, and its stored
probabilities cannot answer occlusion queries, so `zero_shot` is the only
predictor currently explainable and is the default.

**That default records what can be served today, not a view about which
predictor is more worth explaining.** Both are equally interesting to explain;
one of them cannot be. `fine_tuned` stays requestable and fails with the
specific reason — `CounterfactualNotAvailable` for the stored path,
`FineTunedModelNotServable` for the live one — rather than being hidden behind a
missing option. When the model is served from a live endpoint, the default
should be revisited on the merits rather than left in place by inertia.

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
- **Live calls unified** behind one builder and one translation site, so
  `predict` and `explain` report a provider refusal identically, and the
  fine-tuned path is built on `FinetunedPredictor`. `explain`'s `model` now
  defaults to `zero_shot` for the serving reason above. Verified against the
  live provider: both tools return `FineTunedModelNotServable` for the same
  refusal.
- **Tool parameters converted to `Annotated[T, Field(...)]`** with real Python
  defaults. Previously `Field(default=...)` left `FieldInfo` objects as the
  defaults, so the handlers were only callable through the MCP framework, which
  resolves them — calling one directly passed a `FieldInfo` where a value was
  expected. Served behaviour was unaffected and the published schemas are
  unchanged (constraints, defaults and required lists all verified identical),
  but the functions are now directly callable and therefore directly testable.
  Found by the first test that called a handler without naming every argument.

---

# Part 2 — the agent network

Spec quotes in this part are from `internal/Data Science Challenge - Truth - Full.pdf`,
pages 5–8. (The `agent_community_spec.html` sitting beside it is a different
document — a brand-campaign brief — and is not the specification for this work.)

Status: **design proposed, awaiting review. Nothing built.**

Library: `a2a-sdk 1.1.2`, no agent framework.

---

## 11. Agent roster and cards

### Requirement

> "Capability discovery: each agent publishes an agent card / capability
> descriptor so others can find out what it does and how to call it."

> "Aim for one deployable unit per agent rather than a single monolith, so the
> A2A boundaries between agents are real and not just in-process function calls."

### APIs called

```python
from a2a.types import (AgentCard, AgentSkill, AgentCapabilities, AgentInterface,
                       AgentProvider, SecurityScheme, HTTPAuthSecurityScheme,
                       SecurityRequirement)
from a2a.utils import AGENT_CARD_WELL_KNOWN_PATH   # "/.well-known/agent-card.json"
from a2a.utils import TransportProtocol            # JSONRPC | GRPC | HTTP+JSON
from a2a.server.routes import (create_agent_card_routes, create_jsonrpc_routes,
                               add_a2a_routes_to_fastapi)
# create_agent_card_routes(agent_card, card_modifier=None,
#                          card_url="/.well-known/agent-card.json") -> list[Route]
```

**These are protobuf message types in 1.1.2, not pydantic models.** They are
constructed with keyword arguments and have no `.model_fields`; introspection
goes through `.DESCRIPTOR`. Code written against a pydantic-style `AgentCard`
from older examples will not run.

### The card has no top-level `url`

In this SDK version the address lives in `supported_interfaces`, a repeated
`AgentInterface{url, protocol_binding, tenant, protocol_version}`. Older
examples show a single `url` field on the card; that field does not exist here.

### How the URL is populated when it is not known until after deploy

Cloud Run assigns the service URL at create time, so it cannot be baked into the
image and cannot be written into a card at build time.

Two different problems, two different answers:

**An agent's own URL** is resolved per request by a `card_modifier` passed to
`create_agent_card_routes`. It fills `supported_interfaces[0].url` from
`PUBLIC_BASE_URL` if set, otherwise from the forwarded host on the incoming
request. Every agent therefore advertises a correct address with no
configuration at all, and the environment variable exists only to override.

**Peer URLs the orchestrator needs** come from Terraform: each agent service's
`uri` attribute is passed to the orchestrator as an environment variable. This
is acyclic — the orchestrator depends on the three agents, and none of them
depends on the orchestrator — so it resolves in a single apply.

Rejected alternatives: baking URLs into images (needs a rebuild per environment);
a second Terraform apply to close a self-reference (two applies, and the first
leaves cards advertising the wrong address); deriving the URL from the documented
`service-projectnumber.region.run.app` pattern (correct today, but it is a
platform convention rather than a contract, and a wrong card is a silent failure
— an agent that advertises an address it does not answer on).

### The four cards

Shared across all four: `provider.organization`, `protocol_version` `"1.0"`,
`protocol_binding` `JSONRPC`, `default_input_modes` and `default_output_modes`
`["application/json"]`, and bearer-token security.

`application/json` rather than `text/plain` because every payload here is a
batch of structured records. Statements, probabilities and per-field occlusion
deltas travel as A2A `DataPart`s; a human-readable summary rides along as a
`TextPart` for display, but is never the machine-readable channel.

| agent | skill id | consumes over MCP | streaming |
|---|---|---|---|
| `truthclf-orchestrator` | `verify_statements` | data-tools `metrics` only | advertised |
| `truthclf-zero-shot-predictor` | `predict_zero_shot` | model-tools `predict` | no |
| `truthclf-fine-tuned-predictor` | `predict_fine_tuned` | model-tools `predict` | no |
| `truthclf-explainer` | `explain_predictions` | model-tools `explain` | yes |

Security on every card: one `HTTPAuthSecurityScheme(scheme="bearer")` named
`bearer`, with a matching entry in `security_requirements`, and the same scheme
repeated at skill level so a reader of one skill sees the requirement without
resolving the whole card. This satisfies:

> "Sensible security. Don't leave the endpoint wide open. A simple API key or
> bearer token (passed in a header) is enough."

The orchestrator's token is the one handed to the reviewer. The three worker
agents carry their own separate token, so a leaked public token cannot drive
them directly.

### What is easy to get wrong

1. **Advertising an address the agent does not answer on.** A card is a
   contract; a stale URL in it fails at the caller with no error on the serving
   side. This is why the URL is resolved per request rather than configured.
2. **Assuming pydantic.** These are protobuf types; keyword construction works,
   attribute-style validation does not.
3. **Putting the batch in a `TextPart`.** It survives one hop and then someone
   parses prose.

---

## 12. Task lifecycle

### Requirement

> "Task delegation: the orchestrator hands work to the predictor and explainer
> agents over A2A rather than calling functions in-process."

### APIs called

```python
from a2a.client import (ClientFactory, ClientConfig, A2ACardResolver,
                        ClientCallInterceptor, create_client)
# A2ACardResolver(httpx_client, base_url, agent_card_path=AGENT_CARD_WELL_KNOWN_PATH)
#   .get_agent_card()
# Client.send_message(request: SendMessageRequest, *, context=None)
#     -> AsyncIterator[StreamResponse]

from a2a.server.agent_execution import AgentExecutor, RequestContext
# AgentExecutor.execute(context: RequestContext, event_queue: EventQueue) -> None
from a2a.server.tasks import TaskUpdater, InMemoryTaskStore, DatabaseTaskStore
# TaskUpdater.submit / start_work / complete / failed / reject / add_artifact
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import TaskState
#   TASK_STATE_SUBMITTED | WORKING | COMPLETED | FAILED | CANCELED
#   | INPUT_REQUIRED | REJECTED | AUTH_REQUIRED
```

### One request, ten statements

Discovery happens once at start-up, not per request: the orchestrator resolves
all three peer cards and holds the clients. A card fetch per request would add a
round trip to every call for information that changes only on redeploy.

| # | hop | method | state after |
|---|---|---|---|
| 0 | client → orchestrator | `POST /verify` (HTTP, bearer) | — |
| 1 | orchestrator internal | assign a stable `row_id` per point; open run context | orchestrator task `SUBMITTED` → `WORKING` |
| 2 | orchestrator → zero-shot | A2A `message/send` | remote `SUBMITTED` |
| 3 | zero-shot → model-tools | MCP `tools/call` `predict` (model=zero_shot, 10 points) | remote `WORKING` |
| 4 | zero-shot → orchestrator | task result, one `DataPart` | remote `COMPLETED` |
| 5 | orchestrator → fine-tuned | A2A `message/send`, issued concurrently with step 2 | remote `SUBMITTED` |
| 6 | fine-tuned → model-tools | MCP `tools/call` `predict` (model=fine_tuned, source=cached) | remote `WORKING` |
| 7 | fine-tuned → orchestrator | task result, or a per-row unavailable marker | remote `COMPLETED` |
| 8 | orchestrator internal | log-odds pooling per statement | orchestrator task still `WORKING` |
| 9 | orchestrator → explainer | A2A `message/stream` | remote `SUBMITTED` → `WORKING` |
| 10 | explainer → model-tools | MCP `tools/call` `explain` (60 occlusion + 10 rationale calls) | remote `WORKING`, incremental updates |
| 11 | explainer → orchestrator | final task result | remote `COMPLETED` |
| 12 | orchestrator → data-tools | MCP `tools/call` `metrics` — only when labels were supplied | — |
| 13 | orchestrator → client | consolidated JSON body | orchestrator task `COMPLETED` |

Steps 2 and 5 run concurrently. Step 9 runs strictly after step 8: the explainer
does not vote, so it cannot be on the critical path to the verdict, and running
it in parallel would mean explaining a verdict that had not been decided.

### Where task state lives, and what a restart costs

Each agent owns its own task store. The default `InMemoryTaskStore` is
per-instance, which has two consequences that must be stated rather than
discovered:

- **`tasks/get` is not reliable under autoscaling.** With more than one Cloud Run
  instance behind a service, a follow-up lookup can land on an instance that
  never saw the task. Initial deployment pins the agents to a single instance;
  `DatabaseTaskStore` is the fix when that ceases to be acceptable.
- **A restart loses task history.** Acceptable for work that completes inside
  one request, not acceptable if resubscription is ever offered.

**If the orchestrator is killed mid-fan-out:** the caller's HTTP connection
drops and the orchestrator's own task record disappears with the instance. The
three worker agents do not notice — they run their tasks to completion and their
results are stranded with no collector. Nothing is corrupted, because no agent
writes shared state; the cost is wasted provider spend on the zero-shot and
explainer calls, and the caller must retry blind.

The response cache absorbs most of a retry's cost: identical prompts are cache
hits on the second attempt, so a retried request is much cheaper than the first
but not free.

This is a real gap, not a solved problem. Closing it needs a durable
orchestrator task store plus an idempotency key from the caller so a retry
rejoins the original run rather than starting a second one. Both are deferred,
and neither should be described as working until it is.

---

## 13. Synchronous or streaming

### The relevant latencies

| agent | 10 statements | at its ceiling | basis |
|---|---|---|---|
| fine-tuned | milliseconds | milliseconds | dictionary lookup plus a two-parameter calibration map; no network |
| zero-shot | ~0.4–0.5 s of inference | ~13 s at 500 points | 16 client workers, one request per point, ~0.4 s recorded per-call latency |
| explainer | 70 provider requests, ~2–4 s | 350 requests at 50 points, tens of seconds | six occlusion calls plus one rationale call per point |

The zero-shot figures are inference only; a cold Cloud Run instance adds
container start to the first request.

### Recommendation

- **fine-tuned: `message/send`.** It answers from a dictionary. An SSE stream
  would add framing overhead and a second round trip to deliver one event.
- **zero-shot: `message/send`.** Ten statements is a single wave of concurrent
  calls; even a full 500-point batch stays an order of magnitude inside Cloud
  Run's default 300 s request timeout. There are no meaningful intermediate
  results to stream — probabilities are useful as a complete set.
- **explainer: `message/stream`.** It is the long pole by an order of magnitude,
  it is the only agent whose work decomposes into natural incremental units
  (one explanation per statement), and it is the only one that can plausibly
  approach a request timeout. Streaming also keeps the connection demonstrably
  alive, which matters when the alternative is a caller unable to distinguish
  slow from hung.
- **orchestrator: advertises `streaming: true`, serves `/verify` synchronously.**
  The spec's own verification is a single curl returning one body, so `/verify`
  must answer that way. Advertising streaming keeps the option open for an A2A
  client that wants per-statement results as they settle.

**One implementation note that changes how this reads.** In `a2a-sdk 1.1.2`,
`Client.send_message` always returns an `AsyncIterator[StreamResponse]`, and
`ClientConfig.streaming` (default `True`) selects the transport underneath. So
the choice above is a per-peer configuration value, not two code paths. There is
no branch to maintain, and revisiting a decision later is a one-line change.

---

## 14. Failure semantics

### Requirement

> "The deployed Orchestrator must accept a set of statements and run the full
> flow: fan out to the predictor agents via A2A, obtain explanations from the
> explainer agent, and return — for each statement — a final True/False verdict
> and an explanation."

> "Aggregation / consensus: the orchestrator combines the agents' outputs into a
> single answer per statement, including a simple reconciliation strategy for
> when the zero-shot and fine-tuned predictors disagree."

### The fine-tuned agent's availability is a normal condition, not an edge case

The fine-tuned agent answers from stored probabilities covering the validation
and test splits — 3,908 rows. **Any statement outside that set returns
`FineTunedRowNotCached`.** For an external caller sending their own statements,
that is the common case, not a rare one.

So the design treats a missing fine-tuned answer as an expected input to
reconciliation rather than as an error, and it is the same code path as a
timeout, a dead agent, or a short batch.

### Per-source status

Every statement carries a status for each predictor:

`ok` · `unavailable` (the row cannot be scored — `FineTunedRowNotCached`) ·
`timeout` · `error` · `not_returned` (the agent answered but omitted this row)

Only `ok` contributes to pooling. Everything else lands in one branch, which is
what makes a live fine-tuned endpoint a configuration change rather than a code
change: the branch simply stops being taken.

### Reconciliation

Log-odds pooling of the two calibrated probabilities with weight `w`, shipping
at `w = 1` (defer to fine-tuned). The fitted value is a constant.

| available | verdict | `reconciliation.applied` |
|---|---|---|
| both `ok` | pooled probability | `true` |
| exactly one `ok` | that predictor's probability, unmodified | `false`, `reason: "single_source"` |
| neither `ok` | `null`, `status: "no_verdict"` | `false`, `reason: "no_source"` |

A single-source result **is a verdict**, not a failure. It is reported with
`reconciliation.applied: false`, the reason, and which sources were used, so it
cannot be mistaken for a two-predictor result. The distinction is carried by a
required field rather than by absence, because a reader who does not look for a
missing key will assume both predictors answered.

`agreement` in the spec's illustrative response is reported per source with its
status, so `{"zero_shot": true, "fine_tuned": null}` is legible as "the
fine-tuned predictor did not answer" rather than as "the fine-tuned predictor
said false".

### Run level

A run returns HTTP 200 whenever the orchestrator itself completed, even if every
statement is single-source. `run.degraded` is true when any source failed, with
a `run.warnings` list naming what and how many. HTTP 5xx is reserved for the
orchestrator failing, which is the only case where the caller has nothing.

Metrics, when labels are supplied, are computed over statements that have a
verdict, and the response states both the number scored and the number excluded.
Reporting an accuracy over a silently reduced subset would be a different
quantity from the one the caller asked for. Metrics are requested from the
data-tools `metrics` tool, so they arrive with bootstrap intervals and the ECE
occupied-bin count rather than as bare floats, and are reported separately for
each source and for the reconciled verdict.

### What is easy to get wrong

1. **Aligning results by position.** Every response must be joined on `row_id`,
   assigned by the orchestrator before fan-out. A short batch joined by index
   silently attaches one statement's probability to another statement's label.
2. **Letting a single-source verdict look like a consensus.** Hence the required
   `applied` flag and per-source statuses.
3. **Treating `FineTunedRowNotCached` as an error.** It would turn the ordinary
   external request into a failed run.

---

## 15. Trace propagation

### Requirement

> "Observability. Emit logs/traces that make it possible to follow a single
> request as it fans out across the agents — for example a request or trace ID
> that ties the Orchestrator call to the predictor and explainer hops."

### Two identifiers, deliberately

- **`traceparent`** — W3C Trace Context, the transport-level trace. Cloud Run
  populates a trace context on ingress, and Cloud Logging correlates log entries
  to traces automatically when an entry carries
  `logging.googleapis.com/trace`. Using the standard header means the platform
  understands the trace without a bespoke correlator.
- **A2A `context_id`** — the logical run. A2A already carries it across tasks
  and messages, so the whole fan-out shares one run identifier at the protocol
  level, independent of whether tracing is enabled or exporting.

Both are logged on every entry. The trace links spans; the context id links the
agents' own records of the same run.

### How each hop carries it

| hop | mechanism |
|---|---|
| client → orchestrator | Cloud Run ingress; an inbound `traceparent` is honoured, otherwise one is generated |
| orchestrator → agent (A2A) | `ClientCallInterceptor.before()` injects `traceparent` into the outbound request |
| agent → MCP server | the MCP client accepts a preconfigured `httpx.AsyncClient`; a request hook on it injects the same header |
| within an agent | `a2a.utils.telemetry` (`trace_class`, `trace_function`) instruments the SDK's own spans when OpenTelemetry is enabled |

The MCP hop is the one that would otherwise be invisible, and it is where the
provider spend happens — a trace that stops at the agent boundary cannot answer
which tool call was slow or expensive.

### Where it is logged

Structured JSON on stdout, which Cloud Run ingests without an agent. Each entry
carries `logging.googleapis.com/trace`, `logging.googleapis.com/spanId`,
`severity`, the A2A `context_id`, the `task_id`, the agent name, and — for tool
calls — the tool name, point count, cache hits and live calls.

Rejected: a hand-rolled `X-Request-ID`. It would work and cost nothing to
implement, but Cloud Logging would treat it as an opaque field, so correlation
would mean text search rather than the console's own trace view.

---

## 16. What is deliberately not A2A

### Requirement

> "In short: A2A is between agents; MCP is between an agent and its tools. A
> strong submission makes that separation clear in both the design and the
> running system."

> "A2A usage: genuine agent-to-agent communication (capability discovery + task
> delegation), not just internal HTTP calls."

Every place a plain HTTP call is tempting, and the ruling:

| # | call | verdict |
|---|---|---|
| 1 | client → orchestrator `POST /verify` | **Legitimate.** The spec's own verification is `curl -X POST .../verify`. The caller is a grader, not an agent. The orchestrator also serves A2A on the same app, so it is a real agent; `/verify` is a thin adapter that builds an A2A message and hands it to the same executor. |
| 2 | `GET /.well-known/agent-card.json` | **Legitimate.** Plain HTTP, but it *is* A2A's discovery mechanism. |
| 3 | any agent → MCP server | **Legitimate and required.** MCP is agent-to-tool by definition. |
| 4 | `GET /healthz` probes | **Legitimate.** Infrastructure, not agent communication. |
| 5 | **orchestrator → model-tools `predict`, skipping the predictor agents** | **Not legitimate.** Faster, simpler, and numerically identical — which is exactly what makes it tempting. It would reduce the predictor agents to decoration and is precisely what the evaluation criteria call out. |
| 6 | explainer → zero-shot agent for the predictions it explains | **Not needed.** The MCP `explain` tool performs its own occlusion predictions; routing through another agent would add a hop and change nothing. |
| 7 | orchestrator → data-tools `metrics` | **Legitimate.** Metric computation is a tool, not an agent capability, and the spec asks for a shared supporting tool consumed over MCP. |

**Item 5 is enforced by configuration, not by discipline: the orchestrator's
container is not given the model-tools URL or credential.** It therefore cannot
take the shortcut even if someone later adds code that tries. Prediction reaches
the orchestrator only through the predictor agents; the only MCP server it can
address is data-tools.

The generic-client requirement —

> "Demonstrate that the tools also work when called from a generic MCP client,
> not just from your agents."

— is already satisfied: both MCP servers were exercised over streamable HTTP
with a stock client session, independently of any agent.

---

## 17. Open questions for this step

- **Durable orchestrator task state and caller idempotency.** Named in section
  12 as a real gap. Not solved, and deferred deliberately rather than described
  as working.
- **Agent instance pinning.** In-memory task stores mean one instance per agent
  service until a database-backed store is introduced.
- **The pooling weight `w`.** Shipping at 1. Fitting it requires a set where
  both predictors answer, which is the validation split; that is a measurement,
  not a code change.

---

# Part 2, built — the agent network

Status: **four agents running and verified locally.** Containers and IaC next.

```
orchestrator            http://127.0.0.1:9100   card /.well-known/agent-card.json   public POST /verify
zero-shot predictor     http://127.0.0.1:9101   card /.well-known/agent-card.json
fine-tuned predictor    http://127.0.0.1:9102   card /.well-known/agent-card.json
explainer               http://127.0.0.1:9103   card /.well-known/agent-card.json
```

## 18. Two design claims, checked rather than assumed

### Transport selection — confirmed, and the card is authoritative

`BaseClient.send_message` branches on
`not self._config.streaming or not self._card.capabilities.streaming`. Streaming
is used only when **both** the client's configuration and the agent's advertised
capability allow it, so the card can veto but never impose.

Verified live with an interceptor recording the method actually invoked, with
the client configured to stream in every case:

| agent | card `streaming` | method used |
|---|---|---|
| zero-shot | false | `send_message` |
| fine-tuned | false | `send_message` |
| explainer | true | `send_message_streaming` |

The consequence is better than the design assumed: the transport is expressed
once, on the card, and no per-peer client configuration is needed. An agent
declares how it wants to be called and every caller honours it.

### The `card_modifier` hook — the design was wrong

`create_agent_card_routes(agent_card, card_modifier, card_url)` calls
`card_modifier(card)`. **It is passed only the card and never the request**, so
it cannot derive the advertised address from the host that was actually reached
— which is exactly what the design relied on.

Corrected: the card is served from the agent's own route, which has the request
in hand, and is still serialised with the SDK's `agent_card_to_dict`, so the
wire format is the protocol's. Verified: with no proxy headers the card
advertises the local address; behind `x-forwarded-host` /
`x-forwarded-proto` it advertises the external one, with no restart and no
configuration.

## 19. What was built

| module | role |
|---|---|
| `truthclf_agents/common.py` | request identity, JSON logging, bearer auth, base-URL resolution |
| `truthclf_agents/cards.py` | the four cards, and the request-aware card route |
| `truthclf_agents/serve.py` | the A2A app: JSON in and out of message parts, task lifecycle |
| `truthclf_agents/mcp_client.py` | the agent-to-tool client |
| `truthclf_agents/peers.py` | discovery and the agent-to-agent client |
| `truthclf_agents/pooling.py` | reconciliation, pure and independently testable |
| `truthclf_agents/{zero_shot,fine_tuned,explainer,orchestrator}.py` | one deployable unit each |

### APIs called

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.client import ClientFactory, ClientConfig, ClientCallInterceptor
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import (AgentCard, AgentSkill, AgentCapabilities, AgentInterface,
                       Message, Part, Role, Task, TaskState, TaskStatus,
                       SendMessageRequest)
from a2a.utils import AGENT_CARD_WELL_KNOWN_PATH, DEFAULT_RPC_URL, TransportProtocol
from google.protobuf import json_format, struct_pb2
```

## 20. Four defects found by building it

Each was found by running the thing, not by reading it.

**1. A status update cannot be the first event of a task.** `TaskUpdater.submit()`
publishes a `TaskStatusUpdateEvent`, and the server rejects one that arrives
before the task exists: *"Agent should enqueue Task before TaskStatusUpdateEvent
event"*. The executor, not the updater, brings a task into being — it must
enqueue the `Task` itself when `context.current_task` is `None`.

**2. Tool errors arrived wrapped and unmatchable.** The MCP transport runs inside
an anyio task group, so anything raised inside the client's context managers
surfaces as a `BaseExceptionGroup`. A caller's `except ToolCallFailed` therefore
never matched, and a recoverable condition — a statement with no recorded
probability — was reported as an unhandled error with no usable detail. Fixed by
raising the result check outside those blocks and flattening any group that
still escapes.

**3. Every number crosses A2A as a double.** Data parts are protobuf `Struct`
values, which have no integer type: a prediction of `1` arrives as `1.0` and a
row id as `233.0`. Anything used as an identifier or a count is coerced
explicitly at the boundary. Left alone, a row id used as a dictionary key
silently fails to match and every result looks absent.

**4. One unseen statement voided the whole fine-tuned batch.** The stored-probability
reader refuses a batch containing any row it has no value for — correct as a
default, since silently dropping rows is worse. But partial coverage is this
predictor's normal condition, so a batch mixing recorded and new statements lost
*all* its fine-tuned answers, and every verdict fell back to single-source. This
contradicted the per-statement design.

Fixed at the tool: `predict` gained `on_missing="error" | "omit"`. The default is
unchanged. Under `omit` the tool scores what it can and returns the rest in
`missing_row_ids` — it still never substitutes another predictor, it just says
which statements it could not answer. Labels are filtered alongside their rows,
or the metrics would be scored against another statement's truth.

## 21. Reconciliation is interpolation, not evidence accumulation

The weights sum to one, so pooling is a weighted average in log-odds space. Two
predictors that agree do **not** produce a more confident pool than either alone.

That is deliberate, and it is the opposite of what a naive-Bayes-style sum would
do. Summing log-odds treats the predictors as independent evidence and drives
agreement towards certainty — wrong here, because both share a base model, a
prompt and a training signal, so their errors are strongly correlated and their
agreement carries little information. A test pins the interpolation property; an
earlier version of that test asserted the accumulating behaviour and was wrong.

## 22. Verified end to end

Every agent standalone: card fetched over plain HTTP, one A2A message sent, the
resulting MCP tool call confirmed in the reply's provenance, and the RPC route
rejecting an unauthenticated request with 401 while the card stays open.

The full fan-out, through `POST /verify` on the orchestrator, over a batch mixing
two statements the fine-tuned predictor has recorded probabilities for with one
it has never seen:

| statement | zero-shot | fine-tuned | verdict | reconciliation |
|---|---|---|---|---|
| recorded | 0.675 | 0.642 | true | pooled, `applied: true` |
| recorded | 0.362 | 0.255 | false | pooled, `applied: true` |
| new | 0.579 | unavailable | true | `applied: false`, `single_source` |

At `w = 1` the pooled probability equals the fine-tuned one exactly, which is
what deferring to it means. The third statement is a verdict, carries
`agreement.fine_tuned: null` rather than `false`, and states in its own
`reconciliation.detail` that pooling did not run and why. The run is marked
`degraded` with a warning naming how many statements were affected.

Aggregate metrics arrive from the data-tools `metrics` tool with bootstrap
intervals and the ECE occupied-bin count, over the statements that received a
verdict, reporting both `n_scored` and `n_excluded`.

## 23. What is easy to get wrong here

1. **Joining agent replies by position.** Every reply is joined on the
   orchestrator-assigned `row_id`. A short or reordered batch joined by index
   attaches one statement's probability to another statement's label, and
   nothing downstream can detect it. This is why `missing_row_ids` exists rather
   than an implicit "shorter list means the tail is missing".
2. **Treating partial coverage as failure.** It is the normal condition for any
   caller sending their own statements, and it shares one code path with
   timeouts and dead agents so that a live endpoint later removes a condition
   rather than changing the logic.
3. **Serving the card from a static configuration.** A card is a contract; an
   address it advertises but does not answer on fails at the caller with nothing
   logged locally.

## 24. Still open

- **Durable orchestrator task state and caller idempotency.** A killed
  orchestrator still strands its peers' completed work.
- **In-memory task stores** mean one instance per agent service until a shared
  store is introduced.
- **The pooling weight `w`** ships at 1. Fitting it needs a set where both
  predictors answer — the validation split — and is a measurement, not a code
  change.
- **No published output schemas on the MCP tools** (carried over from Part 1).
