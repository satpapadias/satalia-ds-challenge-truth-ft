# Decision log

Short, dated records of non-obvious choices and the evidence behind them.

---

## 2026-06-23 — Platform & base model (CLAUDE.md §3.1)

**Decision:** Use **Together AI** (serverless inference + serverless LoRA fine-tuning).
Carry **two** candidate base models in parallel through the next phase:

- `Qwen/Qwen3.5-9B`
- `google/gemma-4-31B-it`

**Why:**
- **OpenAI is out.** gpt-4o-mini is deprecated/unlisted (2026-06) and OpenAI's
  fine-tuning platform is closed to new users, so the original shared-base plan
  is dead.
- **No-confound requirement.** On Together, Llama models split into a `-Reference`
  (fine-tunable) variant and a `-Turbo` (serverless inference) variant — you
  cannot fine-tune and serve the *same* weights, which confounds the
  zero-shot-vs-fine-tuned comparison. Qwen3.5-9B, gemma-4-31B-it, and the gpt-oss
  models are each fine-tunable **and** serverless-inferenceable under a **single
  id**, so zero-shot and fine-tuned share one base. Verified against the catalog
  and fine-tuning docs.
- **8B Llama is not serverless** on this account ("non-serverless model"), so the
  cheapest-small-Llama idea is not available.
- **Performance (200-row dev subset, zero-shot, score/100, primary labels):**
  - Qwen3.5-9B: acc 0.62 (statement-only) / 0.605 (full), 0% parse-fail, 0.43s
  - gemma-4-31B-it: acc 0.605 / 0.610, bal-acc 0.62, best candidate Brier 0.30, 0.40s
  - Both sit in the literature range (~62–69%) and within ~4 pts of the
    (confounded, non-fine-tunable) Llama-3.3-70B baseline (0.655).
- **Keeping both** for a head-to-head before committing: Qwen is cheapest/fastest;
  gemma is better-calibrated. The full-test run will decide.

**Dropped:** the **gpt-oss family** (20b *and* 120b) — both perform at chance on
this task (acc ~0.48–0.49, bal-acc ~0.55, Brier ~0.40) despite 120b's size, and
are slower. Scale within gpt-oss does not help here.

**Pricing (USD / 1M tokens, verified 2026-06-22):**
Qwen3.5-9B 0.17/0.25 · gemma-4-31B-it 0.39/0.97 · gpt-oss-20b 0.05/0.20 ·
gpt-oss-120b 0.15/0.60 · Llama-3.3-70B-Turbo 1.04/1.04.

---

## 2026-06-23 — Disable reasoning for scoring (`enable_thinking=False`)

**Decision:** For Qwen3.5-9B and gemma-4-31B-it, pass
`chat_template_kwargs={"enable_thinking": False}` and a small output budget
(16 tokens). For the gpt-oss family use `reasoning_effort="low"` + 512 tokens.

**Why:** All of these are reasoning models. With thinking ON they emit a long
chain-of-thought before any answer, which for our terse 0–100 scoring task means:
- Qwen3.5-9B: **no score even at 2048 output tokens** (still reasoning).
- gemma-4-31B-it: emits a score but at **~149 s per call** (≈17 h for the scale
  check).

With thinking OFF both return a clean score in ~3 tokens at ~0.4 s/call, 0%
parse failures. The score itself is what we threshold; the chain-of-thought adds
latency and cost without improving the terse output.

**Thinking-ON vs thinking-OFF check (Qwen3.5-9B, 50-row dev, statement-only):**

<!-- THINKING_CHECK_RESULTS -->

Run: 50-row dev subset, statement-only, Qwen/Qwen3.5-9B (appended 2026-06-23).

| config | acc | Brier | parse-fail | latency/call |
|---|---|---|---|---|
| thinking OFF (enable_thinking=False, 16 tok) | 0.660 | 0.330 | 0/50 (0%) | ~0.43s (scale check) |
| thinking ON (default, 2048 tok) | 0.640 | 0.235 | 47/50 (94%) | 27.4s |

Delta (ON - OFF): acc -0.020, Brier -0.095, parse-fail +47.

**Conclusion:** thinking ON did NOT improve accuracy and added large latency + parse failures (empty content within the 2048-token budget) — disabling reasoning is justified.

Caveat: thinking-ON's lower Brier (0.235) is an artifact, not better calibration —
94% of rows returned empty content and fell back to the neutral-0.5 prediction,
which sits near the all-0.5 baseline Brier.

---

## 2026-06-23 — Batch API over hand-written threaded concurrency

**Decision:** Run full-test (and other large one-off) evaluations through the
**Together Batch API**, not a hand-written threaded/async path. The predictor
keeps two interchangeable backends behind one `predict()` interface: a
synchronous per-row client (cached) for dev/notebook/interactive use, and a
batch client for full-test runs (`backend="sync"|"batch"` at the call site).

**Why:**
- **~50% cheaper** than synchronous inference (full-test ~$0.66 vs ~$1.32).
- **Less code to defend** — no concurrency, rate-limit, or backoff-tuning logic;
  the platform handles parallelism.
- **Latency is irrelevant** for one-off offline eval runs, so the batch
  turnaround cost is acceptable.
- Both backends share the same on-disk cache and cache keys, so dev results and
  full-test results are interchangeable and reruns are free.
- Robustness: a batch can complete with individual failures, so we always read
  `error_file_id` (and any missing `custom_id`s) and route those rows through the
  existing parse-fail / neutral-50 path, logging the count.

**Logprobs:** confirmed available on `Qwen/Qwen3.5-9B` (the `logprobs` parameter
is accepted and returns a logprobs object). Reserved for the post-MVP calibration
experiment; the primary signal remains the 0–100 score.

---

## 2026-06-24 — Lead config locked: gemma-4-31B-it [full]

**Decision:** LEAD = `google/gemma-4-31B-it` with the `full` (metadata) prompt.
SECONDARY (kept) = `Qwen/Qwen3.5-9B` with `statement_only`.

**Evidence (full test set, n=9,618, primary labels, 95% bootstrap CIs):**

| model | variant | acc | bal-acc | macro-F1 | Brier | ECE |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | statement_only | 0.611 | 0.602 | 0.603 | 0.364 | 0.332 |
| Qwen3.5-9B | full | 0.603 | 0.580 | 0.573 | 0.364 | 0.338 |
| gemma-4-31B-it | statement_only | 0.596 | 0.601 | 0.596 | 0.329 | 0.257 |
| **gemma-4-31B-it** | **full** | **0.633** | **0.630** | **0.629** | **0.296** | **0.230** |

Gemma-full's accuracy CI [0.623, 0.642] sits above Qwen-statement-only's
[0.602, 0.622] → the gap is statistically real.

**Methodology points (worth the deck):**
- **The dev preview reversed.** On the 200-row dev subset Qwen led (0.62 vs 0.61);
  on the full set Gemma won. Small samples mislead — the full set + bootstrap CIs
  are what decided it.
- **Metadata is model-specific.** Adding speaker/subject/context metadata *helps*
  Gemma (0.596 → 0.633, ECE 0.257 → 0.230) but *hurts* Qwen, which over-predicts
  True with the richer prompt (Qwen-full recall True 0.778 vs False 0.382, so
  balanced-acc drops to 0.580). So "use all metadata" is not universally correct.
- **Calibration is poor for both** (ECE 0.23–0.34): raw score/100 is uncalibrated.
  Phase (a) addresses this with post-hoc calibration + threshold tuning.

---

## 2026-06-24 — Phase (a): threshold tuning, calibration, abstention

All on cached data (no spend). Tuned on a **speaker-disjoint validation split**,
reported on the **untouched test split** (`data.speaker_disjoint_3way`, ~60/20/20,
test matches the split fine-tuning will reuse).

**Official tuned+calibrated zero-shot baseline (gemma-4-31B-it [full], TEST n=1991):**
accuracy 0.693 · balanced-acc 0.690 · macro-F1 0.690 · Brier 0.231 · ECE 0.070,
using Platt calibration + a val-tuned decision threshold (0.570). **This is the
baseline fine-tuning will be compared against.**

- **Threshold tuning gave little lift** (bal-acc 0.690 → 0.696): the val-optimal
  threshold is ≈0.5, so the model isn't badly mis-thresholded by default.
- **Cost-sensitive threshold** (a false statement labelled True = a FALSE POSITIVE
  in this 1=True encoding, weighted 2×) moves the threshold UP to 0.755 and cuts
  FP 325 → 235, at the expected recall trade-off. (Earlier draft mistakenly
  penalised FN; corrected — the misinformation-costly error here is FP.)
- **Calibration helped reliability, not accuracy** (expected): Platt cut ECE
  0.173 → 0.070 and Brier 0.249 → 0.231 with accuracy unchanged (monotonic map).
  The fitted slope is small (raw scores are overconfident / weakly discriminative,
  ROC-AUC ~0.65), so calibrated probs are compressed.
- **Abstention works**: keeping the most-confident 70% raises accuracy 0.693 →
  0.739 (confidence = |score/100 − 0.5|; raw scores beat the compressed calibrated
  probs as the confidence signal).
- **Split variance, flagged**: this speaker-disjoint test split is ~6 pts easier
  than the full set (0.693 vs 0.633). That's why we report bootstrap CIs and will
  run the fine-tuned comparison on the SAME split (paired), so the comparison is
  unaffected by the split's absolute difficulty.

**Robustness fix:** the response cache is now anchored to the project root
(cwd-independent). A notebook running from `notebooks/` had missed the root cache
and started re-querying the API — fixed so cache hits are location-independent.

---

## 2026-06-24 — Fine-tuning plan: LoRA SFT, serverless only (DPO excluded)

**Serverless LoRA gate — PASSED for `google/gemma-4-31B-it`:** the base is
serverless-callable (verified live), is LoRA-fine-tunable under the same id, the
catalog exposes the serverless multi-LoRA serving id `google/gemma-4-31B-it-lora`,
and Together's serverless LoRA inference is billed at base per-token rates. So a
LoRA fine-tune can be served serverlessly (per-token) — **no dedicated endpoint**.
Gemma logprobs are available (for the eval's p(True)).

**Method = SFT + LoRA only. DPO deliberately excluded.** DPO needs ranked
preferred/dispreferred response pairs. A single-label binary task (True/False)
has no preference ranking — the "preferred" answer is just the gold label, which
is exactly what SFT already trains on. For True/False, DPO degenerates to SFT, so
it adds cost/complexity with no signal. The SFT→DPO combo is for out-of-
distribution preference polishing, not applicable here. Noted as future work,
not built. No dedicated endpoints; no DPO.

**Data:** speaker-disjoint 3-way split (train/val/test). Train and val are
serialized to conversational SFT JSONL (`data.write_sft_jsonl`), system+user =
locked full-metadata prompt (decision mode, `[unknown]` for empty fields),
assistant = single label token. `train_on_inputs="auto"` masks the input so loss
is on the label token only. Val is group-disjoint from train (union-find:
no speaker or near-dup leak); test is the untouched split reused from the
zero-shot baseline so the comparison is paired.

**Shared decision-mode prompt (locked).** The SFT data and the matched
decision-mode zero-shot reference use the IDENTICAL template (both via
`prompts.build_messages(row, "full", "decision")`; verified system and user
strings are equal). Training and the matched reference are therefore apples-to-
apples.

- system: *"You are a careful fact-checking assistant. You judge whether a public
  statement is true or false. Respond with exactly one word: True or False. Do
  not provide any explanation or punctuation."*
- user: the full-metadata block (Speaker / Speaker's job / Speaker's home state /
  Speaker's affiliation / Subjects / Context, each `[unknown]` when empty),
  then `Statement: {cleaned}`, then `Answer (True or False):`.
- assistant (training target): a single token, `True` or `False`.

---

## 2026-06-24 — Serverless-LoRA gate FAILED post-training (gemma-4-31B-it)

The LoRA SFT job **succeeded** (eval loss 0.32 → 0.199 over 3 epochs; adapter
`makisntpap_17e5/gemma-4-31B-it-gemma_truth_sft-c7afbf0d`). But the fine-tune
**cannot be served serverlessly**, which the project requires (serverless only,
no dedicated endpoint):

- Calling the `output_name` directly → `400 "Unable to access non-serverless
  model"` (3 retries).
- Calling the serverless base with `extra_body={"adapters":[{"name": FT}]}`
  returns normally but **silently ignores the adapter**: a *bogus* adapter name
  also returns a normal answer, and base vs "adapter" predictions were identical
  on 20/20 rows. So that path runs the base model, not the LoRA.
- Docs publish no serverless-LoRA base list; web evidence points to Llama-3.1 /
  Qwen as the serverless-LoRA families, not Gemma.

**Lesson (gate was insufficient):** the pre-launch gate trusted the catalog entry
`google/gemma-4-31B-it-lora` as proof of serverless LoRA serving. That was a false
positive — the catalog `-lora` id does NOT guarantee a *custom* adapter can be
served serverlessly. The only reliable verification is actually serving an
adapter, which is only possible post-training. Going forward, verify serverless
serving with a TINY throwaway fine-tune before committing to a full one.

**Status:** training cost (~$3) is sunk; the adapter is saved but unservable
under our constraints. The zero-shot baseline (gemma-4-31B-it [full],
0.693 / ECE 0.070) stands.

**Gemma fine-tune ABANDONED. Fine-tuning track moved to Qwen (verify-first).**
The supported-models doc lists `Qwen/Qwen3.5-9B` as LoRA-fine-tunable but publishes
NO serverless-LoRA serving list, so serving must be tested empirically. Plan:
tiny throwaway Qwen3.5-9B fine-tune (~300 rows, 1 epoch, <$1) → empirically
confirm the adapter (a) is callable serverlessly via `output_name` and (b)
actually changes outputs vs the base (bogus-adapter control). Only if serving
passes do we run the full Qwen SFT. The primary fine-tuning comparison becomes
**Qwen-vs-Qwen** (fine-tuned vs decision-mode zero-shot Qwen, paired); fine-tuned
Qwen vs the Gemma 0.693 zero-shot baseline is a secondary cross-model note.

---

## 2026-06-24 — Serverless custom-LoRA serving appears UNAVAILABLE on this account

The verify-first Qwen test settled it. A tiny `Qwen/Qwen3.5-9B` LoRA fine-tune
(300 rows, 1 epoch, ~$0.05) trained fine but **failed the serving gate, identically
to Gemma**:
- `output_name` → `400 "Unable to access non-serverless model …Qwen3.5-9B-qwen_serve_test…"`
- base + `extra_body={"adapters":[{"name": <bogus>}]}` returns normally → that path
  silently ignores the adapter.

**Two independent bases (gemma-4-31B-it, Qwen3.5-9B) fail the same way**, so this is
an **account/platform-level limitation**, not model choice: serverless serving of
*self-fine-tuned* LoRA adapters is not available here. Together's serverless
Multi-LoRA "Option 1" and catalog `-lora` ids do not apply to our custom adapters.
Switching bases again will not help.

**Implication:** under the strict "serverless only, no dedicated endpoint"
constraint, the fine-tuned-predictor deliverable cannot be served on Together.
Sunk cost so far ~$3 (Gemma) + ~$0.05 (tiny Qwen). The rigorous calibrated
zero-shot baseline (gemma-4-31B-it [full], 0.693 / ECE 0.070) is unaffected.
Decision needed (see below): briefly allow a dedicated endpoint just to run the
FT eval, enable serverless custom-LoRA on the account, or finalize on zero-shot
and document the serving blocker.

---

## 2026-06-24 — FINE-TUNING RESULT (Gemma, dedicated-endpoint eval)

Served the Gemma LoRA fine-tune (`...gemma_truth_sft-c7afbf0d`) on a short-lived
dedicated endpoint (the only serving path; serverless unavailable for this base),
evaluated on the held-out speaker-disjoint **test split (n=1,991)**, then deleted
the endpoint. **Endpoint uptime 10.3 min, ~$2.23, deleted cleanly (no leftovers).**
FT per-row probs persisted in `ft_eval_cache.json`; metrics in `ft_eval_results.json`.

Calibrated (Platt + balanced-threshold, fit on val, reported on test):

| # | config | acc | bal-acc | macro-F1 | Brier | ECE |
|---|---|---|---|---|---|---|
| a | zero-shot, score-mode (orig baseline) | 0.693 | 0.690 | 0.690 | 0.231 | 0.070 |
| b | zero-shot, decision+logprob (matched ref) | 0.669 | 0.663 | 0.663 | 0.217 | 0.056 |
| c | **fine-tuned, decision+logprob** | **0.685** | **0.684** | **0.684** | **0.201** | **0.025** |

**Clean fine-tuning effect (c vs b, same elicitation):** accuracy **+0.017
[+0.003, +0.030]**, McNemar **p = 0.025** (FT wins 119 discordant vs 86) →
**statistically significant**. Calibration improves markedly: ECE 0.056 → 0.025,
Brier 0.217 → 0.201. This meets the success criterion (a modest, CI-backed,
significant gain over matched zero-shot).

**Secondary (c vs a, cross-elicitation):** accuracy −0.008 [−0.021, +0.006],
McNemar p = 0.30 → fine-tuned decision-mode is **statistically tied** with the best
zero-shot config (score-mode), but with **much better calibration** (ECE 0.025 vs
0.070). Labeled secondary because it mixes the fine-tuning effect with the
score→decision elicitation change.

**Takeaways:** (1) fine-tuning genuinely helps on a matched comparison — small but
significant accuracy gain and a large calibration improvement; (2) it does not
beat the strongest zero-shot *elicitation* in raw accuracy but matches it while
being far better calibrated; (3) all numbers sit in the literature ceiling (~0.62–0.69).
Switching elicitation from score→decision costs ~2.4 pts zero-shot (a→b), and
fine-tuning recovers ~1.7 of that.

---

## 2026-06-24 — Explainer (Component 3) design

`explain(model, points, labels=None)` in `truthclf/explain.py`. **Model-agnostic:**
it only calls the predictor's `predict()` and `rationale()`, so it runs identically
on the zero-shot or the fine-tuned predictor (demonstrated on zero-shot serverless;
works on the fine-tuned predictor via the same interface, no endpoint re-run).
Set-level, returns per-point prediction + explanation, and metrics when labels are
passed (reuses `metrics.py`).

**Two layers + faithfulness framing:**
- **Leave-one-field-out occlusion (primary, faithful):** re-run the predictor with
  each metadata field blanked (speaker_name, speaker_affiliation, subjects,
  statement_context) plus an all-metadata baseline; report Δp and label-flip per
  field. Causal input-level attribution. All occlusion variants batched into one
  `predict()` call (cache/batch-friendly; base rows reuse the full-test cache).
- **Model rationale (readable, NOT necessarily faithful):** one-sentence reason,
  explicitly labelled as plausible only.
- **Cross-check:** compare the occlusion-identified driver against the field(s) the
  rationale cites (keyword heuristic); report the agreement rate. Disagreement
  exposes rationalization (e.g. occlusion shows the speaker flips it, rationale
  cites the claim's specificity).

**Out of scope (future work): token/word-level attribution** — expensive, noisy,
and the wrong granularity for field-structured metadata; occlusion at the field
level is the right unit here.

Aggregate (`run_explainer.py`, ~300 test rows) quantifies the speaker/source
shortcuts tracked since EDA. Demonstrated on zero-shot serverless (~$0.13, batched).

### Faithfulness-matching rule (exact criterion)

The cross-check asks: does the rationale "mention" the occlusion-identified
driving field? Definitions:
- **Driver** = the metadata field with the largest |Δp| on removal, if that
  |Δp| > 0.05; otherwise `statement` (no single field materially moved it).
- A rationale **mentions** a field if its text contains any keyword from that
  field's family: speaker = a speaker-name token (len>2) or {speaker, said,
  spoke, person, official}; affiliation = {republican, democrat, gop, party,
  partisan, political, liberal, conservative, …}; subjects = any subject tag
  token (len>3); context = {interview, debate, speech, ad, facebook, tweet,
  post, email, rally, campaign, …}; statement = {claim, statement, figure,
  number, statistic, percent, specific, exaggerat, misleading, …}, and is the
  **default** when no family matches.
- **Agreement** = the driver's family ∈ the rationale's mentioned families.

**This is a LENIENT criterion** (any single family keyword counts, and
`statement` is the permissive default). So the measured agreement is an *upper
bound* on faithfulness — true faithfulness is likely lower. Reported as: under
this lenient keyword field-mention criterion, **rationale↔occlusion agreement =
0.457** (163/300 disagree).

**False-disagreement spot-check (5 cases):** all 5 flagged disagreements are
genuine — the rationale really does not cite the driving field:
- [1267] driver=affiliation (speaker obama); rationale "no factual claim to
  verify" — verdict leaned on party, rationale denies a claim exists. Genuine.
- [7920] driver=speaker_name (claimant *charlie-crist*); rationale discusses
  *Rick Scott* — the claim's SUBJECT, not the claimant. The matcher correctly
  does NOT count "Rick Scott" as a speaker mention. Genuine.
- [4223] driver=speaker_name (rick-perry); rationale about funding sources, no
  claimant mention. Genuine.
- [5357] driver=speaker_name (claimant obama); rationale about *Hillary*'s NAFTA
  stance (the subject), not the claimant. Genuine (claimant≠subject again).
- [5373] driver=subjects (sarah-palin); rationale is claim-content/context, does
  not cite the subject tags. Genuine.
No false disagreements found; notably the rule correctly separates the *claimant*
(speaker) from a *subject person named in the claim*.

### Driver × correctness — does the speaker shortcut help or hurt? (n=300, cached)

| Dominant driver | n | accuracy |
|---|---|---|
| **statement** | 164 | **0.756** |
| speaker_name | 72 | 0.625 |
| subjects | 29 | 0.655 |
| speaker_affiliation | 17 | 0.588 |
| statement_context | 18 | 0.389 |
| _collapsed: statement_ | 164 | 0.756 |
| _collapsed: speaker_name_ | 72 | 0.625 |
| _collapsed: other metadata_ | 64 | 0.562 |

**The speaker shortcut HURTS.** Predictions the model drives from the *statement
content* are right **75.6%** of the time; those it drives from *speaker identity*
only **62.5%**, and other-metadata-driven **56.2%**. So when the model leans on
who-said-it rather than the claim, it is markedly less accurate — the shortcut is
a crutch that degrades performance, not a reliable signal. (Correlational caveat:
the model may fall back on the speaker precisely when the claim is short/
uninformative, i.e. on intrinsically harder items; either reading supports "don't
trust speaker-driven predictions.") This is a headline deck finding.

---

## 2026-08-08 — Post-audit corrections to the record

### Rule: every number on a slide comes from the JSON record

**Decision:** No figure rendered on a slide may be written as prose in
`build_deck.py`. Every number is interpolated from `ft_eval_results.json`,
`results/summary.json` or `results/curves.json`.

**Why:** the deck carried four claims that the JSON had already outgrown —
"+3.3 pts", "halves calibration error", "more than halved", "~46% faithful" —
because they were typed into slide text and speaker notes rather than read from
the record. Numbers inside f-strings self-corrected when the results were
regenerated; numbers in prose did not. Statistics the deck needs are now
persisted deliberately (`ece_difference_zeroshot_vs_finetuned`,
`explainer.driver_vs_baseline`, `explainer.rationale_agreement`,
`results/curves.json`) rather than recomputed or retyped.

Corollary applied at the same time: the slide-8 reliability figure used to
recompute probabilities from the response cache, printing ECE 0.066 on a figure
sitting beside a 0.061 table. It now reads `results/curves.json`, the same
record as the table.

### The 75.6% / 62.5% driver-accuracy figures: STALE, not fabricated

**Correction to an in-flight assumption.** These were flagged as matching no
recorded value. They do: they are in this file, in the earlier explainer
analysis, from a **superseded run** whose subsets were n=164 statement-driven /
n=72 speaker-driven. The current 300-row sample gives n=158 / n=78 and
0.741 / 0.564. So the deck figures were stale — carried forward from an earlier
run after the underlying analysis was re-run — not invented.

The distinction matters for the remediation: a fabricated number means someone
wrote fiction, and the fix is a provenance rule. A stale number means the
pipeline let a superseded value survive a re-run, and the fix is the rule above
(read from the record, never retype). It is the second failure, which is the one
the rule actually prevents.

### Rationale-occlusion agreement is at chance

**Finding:** the explainer's headline faithfulness statistic, 0.457 agreement
between the occlusion-identified driver and the field the model's rationale
cites, is **not distinguishable from chance**. Permutation null (driver labels
shuffled against rationales, 2,000 draws): mean 0.436, 95% range
[0.393, 0.480], **p = 0.19**. Observed 95% CI [0.400, 0.513].

**Why the chance floor is so high:** agreement is scored over 5 categories, but a
rationale cites **1.59 of 5** on average, and `statement` is cited in 183/300
rationales while also being the most common occlusion driver. Assuming a 1/5
chance floor — as "only 46% faithful" implicitly does — is wrong by a factor of
more than two.

**What we now claim:** the rationales carry *no detectable information* about
which field actually drove the prediction. That is evidence of **absent signal**,
not a measured 46% faithfulness rate. Failing to reject at n=300 is not proof of
zero relationship, and the claim is stated that way.

### Speaker-driven predictions sit at the majority-class baseline

**Finding, stronger than the previous "the shortcut hurts":** when speaker
identity is the deciding input, accuracy equals the majority-class rate **of that
subset** — 0.564 vs 0.564, Δ = +0.000 [−0.141, +0.128] (n=78). Statement-driven
predictions are +0.215 [+0.108, +0.285] above *their* subset's baseline (n=158).

**Methodological point:** the baseline must be each subset's own class rate, not
the global one. The subsets have different class balance (0.564 vs 0.525 True),
so comparing both to a single global figure would manufacture a difference.
The point estimate landing exactly on the baseline is coincidence; the claim is
"indistinguishable from the majority-class baseline", and n=78 gives a wide
interval that rules in "no better than the prior" without ruling out a small
effect.

### predict_examples were 100% contaminated by the p=0.5 fallback

**Finding:** all eight rows in `results/summary.json:predict_examples` — the
first table a reviewer sees in `notebooks/00_results_walkthrough.ipynb` — held
`p_true = 0.5` and predicted `True`. Every one came from the silent
logprob-fallback path: when `prob_from_logprobs` found no True/False token it
returned a neutral 0.5, which was recorded as if it were a model output.

**Why it survived:** the fallback was indistinguishable from a real prediction
once written to JSON, and nothing asserted that a demo table should contain
varied probabilities. Regenerated from the schema-2 cache under a known backend:
0/8 at exactly 0.5.

### The Platt calibrator never converged, so it never ran

**Finding:** `fit_platt` was 2,000 fixed steps of gradient descent at lr=0.5 with
no convergence check. Its validation NLL was 0.859 / 1.157 / **3.500** against
temperature scaling's 0.706 / 0.691 / 0.643, and on one run it produced a
**negative** slope (A = −0.0988), i.e. an inverted calibration. So `fit_best`
selected temperature every time and the entire Platt branch was unreachable in
practice while appearing fully implemented and tested.

**After the fix** (logistic regression on the logit, regularisation off via
C=np.inf): Platt wins on all three runs, and the paired bootstrap of the NLL
difference excludes zero each time — so the parsimony margin rule never fires.
This single change moved every non-pr_auc number in the adopted record.

**Lesson:** a code path can be dead because it is *never selected*, not only
because it is never called. Neither coverage nor the test suite would have shown
this; only comparing the two arms' objective values did.

### PR-AUC was order-dependent, not merely imprecise

**Finding:** the hand-rolled `pr_auc` accumulated precision per **sample** rather
than per distinct threshold, crediting operating points the classifier cannot
realise inside a group of tied scores. The result therefore depended on **row
order**. With all scores tied:

| labels | ours | correct |
|---|---|---|
| `[1,0,1,0]` | 0.833 | 0.500 |
| `[1,1,0,0]` | 1.000 | 0.500 |
| `[0,0,1,1]` | 0.417 | 0.500 |

The dataset is squarely in that regime: score-mode elicitation emits ~17 distinct
values across 9.6k rows, and even the logprob path has only 1,134 distinct values
across 1,991 test rows. So "continuous probabilities" was not protection — the
zero-shot logprob run still moved +0.0108.

**Fixed** by delegating to `sklearn.metrics.average_precision_score`, with
regression tests covering the all-tied case and order permutation.

---

## 2026-08-09 — Serverless LoRA serving: settled by direct probe

**Question:** can Together serve `makisntpap_17e5/gemma-4-31B-it-gemma_truth_sft-c7afbf0d`
via serverless inference, with no dedicated endpoint running? The repo asserted
"no" from a 2026-06 observation; NOTES/WALKTHROUGH.md flagged that as unverified
and named the call that would settle it.

**Answer: NO.** One `chat.completions.create(model=FT, max_tokens=1)` with
nothing provisioned returns **HTTP 400**, verbatim:

```json
{"error": {"message": "Unable to access non-serverless model makisntpap_17e5/gemma-4-31B-it-gemma_truth_sft-c7afbf0d. Please visit https://api.together.ai/models/... to create and start a new dedicated endpoint for the model.",
           "type": "invalid_request_error", "param": null, "code": "model_not_available"}}
```

`code: model_not_available` and the phrase "non-serverless model" are explicit:
this is a provider capability statement, not a transient error, not a cold start,
and not a permissions problem. It is deterministic and correctly classified as
non-retryable by `llm.RETRYABLE_ERRORS` (BadRequestError is excluded).

**`endpoints.list_hardware(model=FT)`** returns one option —
`2x_nvidia_h100_80gb_sxm` (gpu_count=2, h100-80gb), `availability: available`.
That confirms a DEDICATED endpoint is possible, which is a different question and
was never in doubt.

**Consequence for the next phase:** the fine-tuned predictor cannot be a
stateless container that calls an API. Serving it requires either a persistently
running dedicated endpoint (always-on cost, no autoscale-to-zero at
min_replicas=1) or accepting multi-minute provisioning on the first request.
Sub-second response with no human present is not achievable with the current
artifact on this provider without an always-on endpoint.

**Cost of settling it:** one 400 response, no tokens billed. Nothing was
provisioned. `endpoints.list()` confirms 0 endpoints for this model.

---

## 2026-08-13 — Stored fine-tuned probabilities are bound to statement identity

**The defect.** `ft_eval_cache.json` maps a `row_id` to a probability and nothing
else. A `row_id` is a position in `data.csv`, not a statement. Any caller
supplying its own text under a `row_id` that happens to exist received a
*different statement's* probability — correctly calibrated, thresholded, and
labelled `status: "ok"`.

Demonstrated end to end: a novel statement sent with no `row_id` was assigned
index `0` by position and came back with `calibrated(cache["0"]) = 0.725283`,
which belongs to a Barack Obama statement in the test split.

**Fix.** `scripts/build_ft_identity.py` writes `ft_eval_identity.json`:
`row_id -> normalised statement key`, one entry per stored probability (3,908 of
3,908 bound). The serving path checks **both** keys and refuses on a mismatch.
`norm_key` is the identity the pipeline already uses for contradiction detection
and split membership, so this introduces no new notion of sameness.

**Both keys, not just the normalised one.** Normalisation is exact match after
lowercasing and punctuation removal, and `_suspicious_normalization_merge` exists
because it is known to over-merge, so equality is strong evidence of identity but
not proof. A `row_id` that matches while the statement does not is logged as a
warning and refused, separately from a plain absence — the two are different
situations and only one of them was ever dangerous.

**Scope.** No recorded number was affected: every figure in the adopted record
comes from real rows carrying their own `row_id`s. The defect only reached
callers supplying their own statements, which is the normal case for the deployed
service and never happens in the offline evaluation.

---

## 2026-08-13 — Occlusion that measures nothing is `undetermined`, not `statement`

**Finding.** On **137 of 300** sampled points (**45.7%**) no occluded field moves
the predicted probability at all. Previously every one of those was attributed to
`statement`, because the driver rule falls back to it when nothing exceeds
`driver_eps`. That fallback conflates two different claims: *"some field moved
it, none by enough"* and *"nothing moved it at all"*.

The second is not a weak version of the first. Score-mode elicitation emits only
~17 distinct values, so a zero delta means the measurement had no resolution
here, not that the claim's content decided the prediction.

**Why it mattered twice.** `statement` is simultaneously the fallback driver and
the fallback rationale reference (`_rationale_refs` returns `{"statement"}` when
no keyword family matches). A point with no signal on either side therefore
scored as *agreement* — absence on both sides counted as concordance, inflating
the faithfulness statistic exactly where there was least to agree about.

**Fix.** A point with no measurable driver reports `driver: "undetermined"` and is
excluded from the rationale cross-check (`agree: None`). The undetermined share is
reported as its own figure. This is separate from the parse-failure gate, which
was working correctly and had nothing to fire on: the model genuinely returned
`"50"` for every variant on the points that prompted the investigation
(`parse_failures = 0`), so these were real measurements of a neutral score, not
absent ones.

**Effect on the record** (300-point sample, regenerated):

| | before | after |
|---|---|---|
| driver `statement` | 158 | 21 |
| driver `undetermined` | — | 137 (45.7%) |
| agreement, observed | 0.457 (n=300) | 0.356 (n=163) |
| permutation null | 0.436 | 0.287 |
| p-value | 0.19 — at chance | **0.014 — above chance** |
| statement-driven vs own baseline | +0.215 [+0.108, +0.285] | +0.238 [−0.048, +0.381] (n=21) |
| `undetermined` vs own baseline | — | **+0.204 [+0.088, +0.285]** |
| speaker-driven vs own baseline | +0.000 [−0.141, +0.128] | +0.000 [−0.141, +0.115] |

**The conclusion reverses, and the reversal is the point.** Restricted to points
where a driver was actually measured, the rationales carry *detectable*
information about which field drove the prediction. The earlier "at chance"
result was an artifact of scoring 137 points that had nothing to agree about,
which inflated the observed rate and the null together.

Two limits stated with it: the restriction to measurable-driver points was
decided after inspecting the data — it followed from this defect fix rather than
from hypothesis search — and it rests on n = 163 and one test.

**The other reversal.** The old "statement-driven predictions are right 74.1% of
the time" was 158 points of which 137 were undetermined. Statement-driven proper
(n = 21) cannot be separated from its own baseline. The category that clearly
beats its baseline is `undetermined`: the model is most accurate precisely where
metadata is irrelevant to it.

**Also fixed.** `explain_results.json` was only ever partly regenerated — the
entrypoint refreshed `metrics` and left `aggregate` from whatever was on disk, so
it had drifted until it disagreed with the same figures in `results/summary.json`.
It is now rewritten in full.
