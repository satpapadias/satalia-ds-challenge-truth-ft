# CLAUDE.md — truthclf, deployment brief

Standing brief. Loaded into every session. Supersedes the Phase 2 brief, which
described the architecture as undecided — it is now built and verified locally.
The work ahead is the migration to Vertex AI and the deployment to Cloud Run.

---

## 0. Working agreement (read first) — unchanged

**Architecture and design decisions are made in conversation, not by you.**
You implement designs we have agreed and report the result. You do not choose
the architecture, the platform, the service boundaries, the storage layer, or
the deployment topology. When a decision is needed, surface the options with
their trade-offs and evidence, recommend one, and **stop**. Do not start
building while the decision is open.

**Stop-gates are hard stops.** Work in phases. At the end of each, summarise what
you found or built and wait for review. When a task says "report before doing X",
that is a hard stop, not a formality.

**Every quantitative claim ships with an interval, a null model, or a baseline.**
No bare numbers in a README, a deck, a commit message, or a report. Four claims
have now died for lacking exactly one of the three:

| claim | what was missing | what it turned out to be |
|---|---|---|
| "fine-tuning more than halved ECE, 0.053 → 0.027" | **an interval** | +0.0097 [−0.0057, +0.0270] — straddles zero |
| "rationales are only ~46% faithful" | **a null model** | matched permutation null; see §1 for the corrected figure |
| "the speaker shortcut hurts, 62.5% vs 75.6%" | **a baseline** | each subset's own majority-class rate; speaker-driven is +0.000 above its own |
| "statement-driven predictions are right 74.1% of the time" | **a denominator that meant one thing** | 158 points of which 137 had no measurable driver at all |

If a number cannot carry one of the three, say so explicitly rather than
reporting it bare.

**Never run a paid operation** — a fine-tuning job, a full-set LLM evaluation, an
endpoint provision — without printing a cost estimate first and getting an
explicit go-ahead.

**The adopted record does not move silently.** Any change that could alter a
value in `results/summary.json`, `ft_eval_results.json`, `explain_results.json`
or `results/curves.json` requires a before/after diff of every value and an
explicit flag on anything that moved by more than 0.001. Verified by re-running
the entrypoint, not by reasoning about it.

**Confidentiality.** Do not push this code or `data.csv` to any public
repository. The `truthclf-tools` image contains `data.csv` and is as confidential
as the dataset — private registry only.

---

## 1. What exists and is verified

Python 3.12.13, uv, `uv.lock` committed. **238 tests, all provider calls mocked.**
Head is `f12de98`, branch `hardening`.

### The offline record

Reproduce the entire adopted record with no key and no network:

```bash
uv sync --all-extras --group agents
uv run python scripts/regenerate_results.py --source archive --explainer-source archive
```

On the speaker-disjoint test split (n = 1,991):

- zero-shot logprob baseline: accuracy **0.668006**, ECE 0.0613 (equal-mass, 9/10 bins)
- fine-tuned (LoRA SFT): accuracy **0.699146**, ECE 0.0516 (10/10 bins)
- fine-tuning effect: **+0.031 [+0.019, +0.043]**, McNemar exact p = 6.3e-07
- calibration effect (raw → calibrated): ECE **0.316 → 0.061**, +0.255 [+0.213, +0.279]
- fine-tuning does **not** improve calibration: +0.0097 [−0.0057, +0.0270]

Explainer, on a 300-row sample:

- **45.7%** of points (137/300) have no measurable occlusion driver and are
  reported as `undetermined`, not as statement-driven
- rationale/occlusion agreement **0.356 [0.282, 0.429]** on the 163 points with a
  driver, against a matched permutation null of **0.287 [0.233, 0.344]**,
  **p = 0.014 — above chance**. Restricted after a defect fix, not by hypothesis
  search; n = 163, one test
- speaker-driven predictions sit on their own subset's baseline: +0.000 [−0.141, +0.115]
- `undetermined` is +0.204 [+0.088, +0.285] above its own baseline — one of six
  subsets tested, no multiplicity correction, and most likely selection rather
  than cause

### The running system

Two MCP servers, four A2A agents, **six containers**, all verified locally.

| service | port | image | holds |
|---|---|---|---|
| `data-tools` (MCP) | 8081 | tools | `data.csv`, TF-IDF index. **No credential, no network** |
| `model-tools` (MCP) | 8082 | tools | provider credential, response cache, calibrators, stored FT probabilities |
| `orchestrator` (A2A) | 9100 | agent | public `POST /verify`; peers by card |
| `zero-shot-predictor` (A2A) | 9101 | agent | |
| `fine-tuned-predictor` (A2A) | 9102 | agent | |
| `explainer` (A2A) | 9103 | agent | |

**Split images.** `truthclf-agent` **418MB**, `truthclf-tools` **1.47GB**. The
agent stage installs only the `agents` dependency group with
`--no-install-project`, copies only `truthclf_agents/`, and **fails the build**
if `truthclf`, `truthclf_mcp`, `sklearn`, `scipy`, `pandas`, `statsmodels`,
`numpy`, `together`, `tiktoken` or `diskcache` is importable. Verified
non-vacuous: the same check run inside the tools image exits 1.
`tests/test_agent_isolation.py` asserts the same property from source.

**All five container checks pass** (`docker compose up -d --wait`, ~27 s to
healthy):

1. six services healthy; both MCP `/healthz` report what they loaded (5,710
   indexed rows and the 5710/1917/1991 split; 3 calibrators, 3,908 stored rows)
2. cards resolve per caller — `http://127.0.0.1:9101` from the host,
   `http://zero-shot-predictor:9101` from a peer, no configuration
3. mixed batch: recorded row pooled from both predictors, novel statement
   single-source with `fine_tuned: unavailable`
4. with `zero-shot-predictor` stopped: HTTP 200, single-source verdict on the
   covered row, `no_verdict` on the novel one, `run.degraded: true`
5. MCP Inspector CLI lists tools on both servers through the container boundary

### Load-bearing properties that are easy to break

- **Stored fine-tuned probabilities are bound to statement identity**, not
  `row_id`. `ft_eval_identity.json` maps `row_id → norm_key`; both are checked,
  and a `row_id` hit with a `norm_key` miss is logged and refused. Without it a
  novel statement assigned `row_id 0` by position was answered with row 0's
  probability, marked `ok`.
- **Calibrator artifacts are keyed by (model, elicitation)**, schema 3. The two
  zero-shot artifacts share a model id and differ only by elicitation.
- **`driver: "undetermined"`** is a distinct category from `statement` and is
  excluded from the rationale cross-check.
- **`measured: list[bool]`** on `PredictionResult` distinguishes a genuine
  neutral score from an absent measurement; `probs` is deliberately unchanged.
- **The orchestrator is not given the model-tools address**, so it cannot bypass
  the predictor agents. Enforcement by configuration, not discipline.

---

## 2. What is deferred, and why

Not gaps discovered late — decisions taken deliberately, with the reason.

**Durable orchestrator task state.** Every agent uses `InMemoryTaskStore`. A
killed orchestrator loses its task record while its peers run to completion and
strand their results; `tasks/get` is unreliable with more than one instance. The
fix is a shared store (`DatabaseTaskStore`) plus one instance per agent until
then. Deferred because every task completes inside one request today, and
resubscription is not offered. **Do not describe this as working.**

**Caller idempotency.** A retried `/verify` starts a second run rather than
rejoining the first. The response cache absorbs most of the cost but not all.
Closing it needs a caller-supplied idempotency key and the durable store above.

**MCP tool output schemas.** Handlers are annotated `-> dict`, which carries no
field structure, so results arrive as JSON text rather than structured content.
Clients and Inspector render them; an agent gets no machine-readable contract.
Pydantic output models are the first follow-up.

**Batch ceilings are derived, not measured** through the transport: predict 500,
explain 50, dataset 1,000/page, metrics 100,000, retrieval 500. Derived from 16
client workers and ~0.4 s recorded per-call latency; the one measured anchor is
6,075 rows in 17.2 min at ~$0.33.

**The spend gate is weaker than §0 requires, and the architecture forces it.**
`max_live_calls` is a budget ceiling, not consent: once an agent holds `predict`
there is no human in the loop. Anything stronger has to live outside the tool.

---

## 3. GCP facts now settled empirically

Source: `docs/deployment-plan.md`, derived from a read-only review of a peer
solution deployed against this organisation (provenance in `docs/decisions.md`,
2026-08-14). These are **observations from a real deployment**, not inference.

- **`allUsers` is blocked** by `iam.allowedPolicyMemberDomains`. The planned
  public orchestrator with `--allow-unauthenticated` plus an in-app bearer does
  not survive the port.
- **`--allow-unauthenticated` does not fail under that policy** — it warns and
  leaves the service private. **A green deploy is not evidence the binding
  exists.** Any deploy step must verify reachability from outside.
- **The sanctioned exposure path is the shared A2A Agents Gateway**, with a
  Secret Manager bearer and `--forward-id-token`.
- **Agent-to-agent auth is per-hop OIDC ID tokens**, not a shared bearer.
- **A tuned Gemini model is served per-token** through
  `generate_content(model=<endpoint>)`. No dedicated endpoint, no machine type,
  no teardown. The standing-cost problem that shaped the current design does not
  exist on Vertex.
- `job.tuned_model.endpoint` is the inference handle; `job.tuned_model.model`
  returns 404.
- Vertex Gemini SFT requires a **regional** location; `global` rejects tuning,
  and the client must be pinned to that region for tuned inference.

**Still open, and the one that matters most:** whether logprobs are returned by a
**tuned** `gemini-2.5-flash`. The peer solution uses no logprobs anywhere, so it
does not answer this. Calibration, ECE, Brier and threshold tuning all depend on
a continuous probability. See `docs/vertex-migration.md` §2 — this is verified by
probe **before** any full tuning run is paid for.

---

## 4. Read these before proposing anything

| file | why |
|---|---|
| `docs/vertex-migration.md` | the migration plan: target model, what must be probed first, what changes and what does not |
| `docs/deployment-plan.md` | the GCP posture: exposure, auth, serving, Terraform |
| `NOTES/PHASE2-DESIGN.md` | every design decision in the MCP and agent layers, with the defects each one prevents |
| `NOTES/WALKTHROUGH.md` §5, §6 | interface contracts, behaviour under stress, shared state, the fine-tuned serving story |
| `docs/decisions.md` | every decision with dates and evidence, newest last |

`docs/decisions.md` (2026-08-13) carries the pattern worth internalising: **a
category whose default absorbs every unmeasured point, then inflates any
statistic computed over it.** It produced two wrong published claims from the
same value in two roles.

---

## 5. Repo conventions

- Commit messages: what changed and **why**, with evidence. **No `Co-Authored-By`
  or "Generated with Claude Code" trailers.**
- One logical change per commit; full suite green after each.
- Do not reimplement what sklearn/scipy/statsmodels provide. Where something is
  genuinely hand-rolled (ECE, the reliability curve, the shared-resample
  bootstrap), it is pinned by a reference test — keep it that way.
- Edge-case behaviour is chosen explicitly at the call site (`zero_division=`,
  `labels=`, tie handling) and commented, never left to a default.
- No silent failure. Errors raise or warn with a count; failures are never
  cached; degraded inputs fail loudly. `except Exception: pass` is a defect.
- Comments explain the code on its own terms. No references to the instruction
  file, to design conversations, or to commit-by-commit process.
- Agents are pure MCP clients. No agent imports `truthclf`; the build enforces it.
- `dist/` is a frozen submission archive. Never write to it.
- Decks and presentation artifacts are historical. Out of scope unless asked.
- Personal working notes (reading sessions, peer reviews) live **outside** this
  repository. It stays clean and standalone as a deliverable.
