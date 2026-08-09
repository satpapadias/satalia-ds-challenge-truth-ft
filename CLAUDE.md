# CLAUDE.md — truthclf, Phase 2

Standing brief. Loaded into every session. Phase 1 is complete and its brief is
retired; this replaces it.

---

## 0. Working agreement (read first)

**Architecture and design decisions are made in conversation, not by you.**
You implement designs we have agreed and report the result. You do not choose
the architecture, the platform, the service boundaries, the storage layer, or
the deployment topology. When a decision is needed, surface the options with
their trade-offs and evidence, recommend one, and **stop**. Do not start
building while the decision is open.

**Stop-gates apply exactly as they did in Phase 1.** Work in phases. At the end
of each, summarise what you found or built and wait for review. Do not skip
ahead. When a task has an explicit "report before doing X", that is a hard stop,
not a formality.

**Every quantitative claim ships with an interval, a null model, or a baseline.**
No bare numbers in a README, a deck, a commit message, or a report. This is not
style — three Phase 1 claims died for lacking exactly one of the three:

| claim | what was missing | what it turned out to be |
|---|---|---|
| "fine-tuning more than halved ECE, 0.053 → 0.027" | **an interval** | +0.0097 [−0.0057, +0.0270] — straddles zero |
| "rationales are only ~46% faithful" | **a null model** | permutation null 0.436, p = 0.19 — at chance |
| "the speaker shortcut hurts, 62.5% vs 75.6%" | **a baseline** | each subset's own majority-class rate; speaker-driven is +0.000 [−0.141, +0.128] above its own |

If a number cannot carry one of the three, say so explicitly rather than
reporting it bare.

**Never run a paid operation** — a fine-tuning job, a full-set LLM evaluation, an
endpoint provision — without printing a cost estimate first and getting an
explicit go-ahead. Endpoints bill until deleted.

**The adopted record does not move silently.** Any change that could alter a
value in `results/summary.json`, `ft_eval_results.json`, `explain_results.json`
or `results/curves.json` requires a before/after diff of every value and an
explicit flag on anything that moved by more than 0.001. Verified by re-running
the entrypoint, not by reasoning about it.

**Confidentiality.** Do not push this code or `data.csv` to any public
repository. Backups are local (`~/backups/satalia/`) plus a personal-email copy.

---

## 1. Where things stand

Phase 1 complete at tag `phase1-complete` (commit `149798e`, branch `hardening`,
23 commits). **132 tests, all LLM calls mocked.** Python 3.12.13, uv, `uv.lock`
committed.

Reproduce the entire adopted record offline, no API key, no network:

```bash
uv sync --all-extras
uv run python scripts/regenerate_results.py --source archive --explainer-source archive
# ~13 s warm, ~47 s cold; prints accuracy 0.668006
```

Adopted headline numbers, on the speaker-disjoint test split (n = 1,991):

- zero-shot logprob baseline: accuracy **0.668006**, ECE 0.0613 (equal-mass, 9/10 bins occupied)
- fine-tuned (LoRA SFT): accuracy **0.699146**, ECE 0.0516 (10/10 bins)
- fine-tuning effect: **+0.031 [+0.019, +0.043]**, McNemar exact p = 6.3e-07
- calibration effect (raw → calibrated, zero-shot): ECE **0.316 → 0.061**, +0.255 [+0.213, +0.279]
- fine-tuning does **not** improve calibration: +0.0097 [−0.0057, +0.0270]

## 2. Read these before proposing anything

| file | why |
|---|---|
| `NOTES/WALKTHROUGH.md` §5 | exact interface contracts, behaviour under empty/malformed/1000-row/timeout/cold-start, every piece of shared state, everything that assumes a single interactive process |
| `NOTES/WALKTHROUGH.md` §6 | the fine-tuned model's serving story |
| `NOTES/WALKTHROUGH.md` §4 | where the project is exposed, ranked |
| `docs/decisions.md` | every decision with dates and evidence, including two failed serverless-LoRA experiments |
| `docs/driver_eps_sensitivity.md` | worked example of the interval/null/baseline rule |

§5 and §6 were written as the input to this phase. Start there rather than
re-deriving from the code.

## 3. What Phase 2 is

The three components — zero-shot predictor, fine-tuned predictor, explainer —
become independently deployable network services. Target topology is one
container per agent, multi-host, likely Cloud Run.

**The architecture is not decided. Do not assume one.** Service boundaries,
state ownership, the cache layer, and the deployment target are all open and
will be settled in conversation.

## 4. Known facts, and open questions — keep them apart

### Established (do not re-derive)

- **Together AI cannot serve our fine-tuned LoRA serverlessly.** Probed
  2026-08-09: `chat.completions.create(model=FT, max_tokens=1)` → HTTP 400,
  `code: model_not_available`, *"Unable to access non-serverless model"*. Two
  earlier experiments agree: the Gemma adapter and a deliberate ~$0.05 throwaway
  Qwen adapter both failed the same way, and the `extra_body={"adapters":[…]}`
  path silently ignores the adapter (a bogus name also "works"). This is an
  account/platform-level limitation **of Together**, established across two
  independent bases. On Together the fine-tuned predictor needs an always-on
  dedicated endpoint (2×H100, `min_replicas=1`, no scale-to-zero) or accepts
  multi-minute provisioning on first request.
- `.llm_cache/` is diskcache/SQLite: safe for concurrent processes **on one
  host**, not across hosts or over NFS/EFS.
- `ft_eval_cache.json` and `results/*.json` use whole-file writes with no
  locking — concurrent writers lose updates.
- `cleanup_leftovers` in `scripts/evaluate_finetuned.py` matches endpoints by
  **display name** and will delete another container's live endpoint.
- The response-cache path is hard-coded to the source tree
  (`llm._PROJECT_ROOT/.llm_cache`) and is not configurable.
- The calibrator is a shippable artifact (`results/calibrators/*.json`, ~600
  bytes, loads in microseconds) and is the one piece already container-ready.
  Without it `predict()` returns raw probabilities at 0.5.

### Open (to be decided or measured — do not assume an answer)

- **Platform is undecided; being settled tomorrow.** GCP Vertex AI is the likely
  target. **GPU quota is unconfirmed. Scale-to-zero support is unconfirmed.**
  Whether Vertex can serve a LoRA adapter without a persistently running
  endpoint is unknown and must not be inferred from the Together result — that
  finding is Together-specific.
- If the platform changes, whether the existing adapter transfers or the SFT
  data must be re-trained on a new base is open. `ft_data/train.jsonl` (5,710
  rows, ~652k tokens) and `val.jsonl` (1,917 rows, ~217k tokens) are
  base-agnostic text and are reusable in principle; the single-token
  `True`/`False` target must tokenise as one token on any new tokenizer.
- Whether the fine-tuned predictor is worth serving at all, given +0.031
  accuracy against a standing GPU cost, versus serving only the zero-shot
  predictor (which **is** serverless and whose calibrator ships identically).

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
- `dist/` is a frozen submission archive. Never write to it.
- Decks and presentation artifacts are historical. Out of scope unless asked.
