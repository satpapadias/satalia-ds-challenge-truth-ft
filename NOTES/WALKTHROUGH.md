# Walkthrough

Written for the author. Assumes ML fluency, assumes no memory of why any
particular line is the way it is. Everything here is checkable against the code;
where I am inferring rather than reporting, it says so.

---

# Section 1 — Execution map

## Trace A: one row of `data.csv` → a number in `results/summary.json`

Live path, zero-shot logprob run (`b`), which is the headline baseline.

| # | file:line | what happens to the row |
|---|---|---|
| 1 | [data.py:142](../truthclf/data.py#L142) `load` | CSV line → `Row`. Two derived fields computed here, not later: `statement_clean` ([data.py:75](../truthclf/data.py#L75) `clean_text`) and `norm_key` ([data.py:97](../truthclf/data.py#L97)) |
| 2 | [data.py:228](../truthclf/data.py#L228) `clean_dataset` | Row is dropped entirely if its `norm_key` group carries both binary labels ([data.py:174](../truthclf/data.py#L174) `contradiction_groups`). Survivors keep their original `row_id` |
| 3 | [data.py:337](../truthclf/data.py#L337) `speaker_disjoint_3way` | → [data.py:330](../truthclf/data.py#L330) → [data.py:255](../truthclf/data.py#L255) `_components` (scipy `DisjointSet`, unions on shared `norm_key` and shared `speaker_name`) → [data.py:309](../truthclf/data.py#L309) `_stratified_assign` → [data.py:294](../truthclf/data.py#L294) `_greedy_balanced`. Row lands in train, val or test as a whole component |
| 4 | [data.py:131](../truthclf/data.py#L131) `Row.y` → [data.py:55](../truthclf/data.py#L55) `binarize` | 6-way label → {0,1} |
| 5 | [prompts.py:86](../truthclf/prompts.py#L86) `build_messages` | → [prompts.py:72](../truthclf/prompts.py#L72) `build_user_prompt` → [prompts.py:60](../truthclf/prompts.py#L60) `assemble_metadata`. Empty fields become `[unknown]` ([prompts.py:49](../truthclf/prompts.py#L49)). Note it uses `statement_clean or statement` |
| 6 | [zeroshot.py:127](../truthclf/predictors/zeroshot.py#L127) `predict` → [:115](../truthclf/predictors/zeroshot.py#L115) `_predict_logprobs` | |
| 7 | [llm.py:660](../truthclf/llm.py#L660) `TogetherBatchClient.classify` → [:644](../truthclf/llm.py#L644) `_serve` | Cache key built at [llm.py:570](../truthclf/llm.py#L570) `_key_classify` → [llm.py:155](../truthclf/llm.py#L155) `ResponseCache.key`. Hit → step 9. Miss → step 8 |
| 8 | [llm.py:574](../truthclf/llm.py#L574) `_run` | One JSONL, one upload, one batch job, poll loop, download, reconcile by `custom_id`. Failed rows are **not** cached ([llm.py:628-636](../truthclf/llm.py#L628-L636)) |
| 9 | [llm.py:446](../truthclf/llm.py#L446) `_parse_logprobs_payload` → [llm.py:412](../truthclf/llm.py#L412) `_lp_content_to_top` | Raw stored payload → `{token: logprob}` **on read** |
| 10 | [zeroshot.py:41](../truthclf/predictors/zeroshot.py#L41) `prob_from_logprobs` | softmax over the True/False tokens → p. `None` → 0.5 fallback at [:122](../truthclf/predictors/zeroshot.py#L122) |
| 11 | [evaluation.py:54](../truthclf/evaluation.py#L54) `calibrated_evaluation` | Called from [regenerate_results.py](../scripts/regenerate_results.py) `evaluate`. Fits on **val**, applies to both |
| 12 | [calibration.py:125](../truthclf/calibration.py#L125) `fit_best` | → [:46](../truthclf/calibration.py#L46) `fit_temperature` (scipy bounded Brent), [:62](../truthclf/calibration.py#L62) `fit_platt` (sklearn, `C=np.inf`), [:93](../truthclf/calibration.py#L93) `nll_difference_ci` (paired bootstrap, both refit per resample) |
| 13 | [calibration.py:84](../truthclf/calibration.py#L84) `apply` | row's p → calibrated p |
| 14 | [threshold.py:68](../truthclf/threshold.py#L68) `tune_threshold` → [:32](../truthclf/threshold.py#L32) `candidate_thresholds` | Candidates from `roc_curve(..., drop_intermediate=False)`, `thresholds[0]` (`+inf`) dropped, sorted ascending |
| 15 | [threshold.py:27](../truthclf/threshold.py#L27) `predict_at` | `calibrated >= threshold` → the row's 0/1 |
| 16 | [metrics.py:267](../truthclf/metrics.py#L267) `metric_bundle` | Row's prediction folds into 11 scalars |
| 17 | `scripts/regenerate_results.py` `main` | → `results/summary.json`, `ft_eval_results.json`, `results/curves.json`, `results/calibrators/*.json` |

**Where the row's identity survives to the end:** `row_id` is the join key into
`ft_eval_cache.json` ([finetuned.py:104](../truthclf/predictors/finetuned.py#L104)).
Everywhere else the row becomes a position in a list, and **order is the only
thing keeping probabilities aligned with labels** from step 7 onward.

## Trace B: a fine-tuning run

| # | file:line | what happens |
|---|---|---|
| 1 | [finetuned.py:38](../truthclf/predictors/finetuned.py#L38) `fine_tune(training_dataset)` | |
| 2 | [finetuned.py:47](../truthclf/predictors/finetuned.py#L47) | Internal `speaker_disjoint_split(test_frac=0.2, seed=0)` — **a second, independent split** of whatever you passed in. An explicit `val_rows=` overrides it |
| 3 | [data.py:405](../truthclf/data.py#L405) `write_sft_jsonl` → [data.py:391](../truthclf/data.py#L391) `to_sft_record` | Each row → `{"messages": [system, user, assistant]}` where assistant is the single token `"True"`/`"False"`. Uses `mode="decision"`, so **training prompts and inference prompts are the same strings** |
| 4 | [finetuned.py:56-66](../truthclf/predictors/finetuned.py#L56-L66) | Two uploads (`purpose="fine-tune"`, `check=True`), then `fine_tuning.create` with `lora=True, n_epochs=3, learning_rate=1e-5, train_on_inputs="auto", n_evals=10` |
| 5 | [finetuned.py:71-82](../truthclf/predictors/finetuned.py#L71-L82) | Blocking poll loop, `time.sleep(poll_interval)`, until status contains `COMPLET`. Returns `model_output_name` |
| 6 | `ft_artifacts.json` | The only place the run's identity is recorded: job id `ft-164d480e-cf7f`, output `makisntpap_17e5/gemma-4-31B-it-gemma_truth_sft-c7afbf0d`. **Written by a script, not by `fine_tune`** |
| 7 | [evaluate_finetuned.py:42](../scripts/evaluate_finetuned.py#L42) | That id is hard-coded as `FT` |
| 8 | [evaluate_finetuned.py:113](../scripts/evaluate_finetuned.py#L113) | `endpoints.create(model=FT, hardware="2x_nvidia_h100_80gb_sxm", min=max=1)` |
| 9 | [evaluate_finetuned.py:153-159](../scripts/evaluate_finetuned.py#L153-L159) | `ThreadPoolExecutor(16)`, one request per row, `_ptrue` per response |
| 10 | [evaluate_finetuned.py:170](../scripts/evaluate_finetuned.py#L170) | `json.dump(cache, ...)` every 200 rows and at the end → `ft_eval_cache.json` |
| 11 | [evaluate_finetuned.py:176](../scripts/evaluate_finetuned.py#L176) | `endpoints.delete` in a `finally` |

**The fine-tuned model is never called again after step 10.** Every reported
fine-tuned number is computed from `ft_eval_cache.json`. Run `c` in the record is
a replay of 3,908 stored floats.

## Trace C: the offline reproduction path

`python3 scripts/regenerate_results.py --source archive --explainer-source archive`
— no API key, no network, ~13 s. This is the one a reader runs, and it does not
touch `truthclf.llm`'s clients at all.

1. `ArchiveSource.__init__` loads `.llm_cache.json` (4.7 MB, tracked) into a dict.
2. `ArchiveSource._key` is [llm.py:203](../truthclf/llm.py#L203) `legacy_v1_key` — the **schema-1** hash, no backend/call/schema fields. This is the only surviving consumer of that format.
3. `score` / `logprobs` / `rationale` look up by that key. `logprobs` handles **two** historical value shapes: the bare `{token: logprob}` map that all 3,908 stored classify entries actually use, and the `{text, top_logprobs}` envelope the current sync client writes.
4. Run `c` comes from `ft_eval_cache.json` via `load_cached_probs`, identical in both paths.
5. Steps 11–17 of Trace A follow unchanged.
6. `_assert_complete` runs after the runs, after the explainer sample, and again before writing.

The live path differs only in steps 1–3: `LiveSource` reads `.llm_cache/`
(diskcache) with schema-2 keys, and raises `SystemExit` at construction if the
directory is absent.

---

# Section 2 — The decision log

### Label binarisation: middle split
- **Decided:** `{true, mostly-true, half-true} → True`. [data.py:41-51](../truthclf/data.py#L41-L51) `BINARY_SCHEMES`, enforced at [data.py:55](../truthclf/data.py#L55).
- **Alternative:** `half-true → False` — implemented as the `"sensitivity"` scheme and never run end-to-end.
- **If reversed:** class balance moves ~56/44 → ~35/65. Accuracy stops being a usable headline; the tuned threshold moves substantially; the split's majority-label stratification ([data.py:311-315](../truthclf/data.py#L311-L315)) reassigns components, so **every row's split membership changes** and no number in the record survives.
- **The real reason:** comparability with the published LIAR numbers. Nothing in the code says so.

### Speaker-disjoint split, with repeat-groups unioned in
- **Decided:** components = shared speaker ∪ shared `norm_key`. [data.py:255](../truthclf/data.py#L255).
- **Alternative:** `sklearn.model_selection.GroupShuffleSplit` on speaker alone.
- **If reversed:** repeated statements by *different* speakers would straddle the split. `analyze_dataset.py` counts 19 normalised-repeat groups over 41 rows — small, but they are exactly the rows a memorising model gets free.
- **Subtle:** empty `speaker_name` is given a **unique** placeholder `f"__empty__{row_id}"` ([data.py:279](../truthclf/data.py#L279)), so blank-speaker rows stay singletons rather than merging into one giant component. Without it, every anonymous row collapses into a single group and the split becomes wildly unbalanced.

### Elicitation mode: logprob for the headline, score for the explainer
- **Decided:** `use_logprobs=True` default ([zeroshot.py:64](../truthclf/predictors/zeroshot.py#L64)); the explainer runs score-mode ([run_explainer.py](../scripts/run_explainer.py)).
- **Why the split:** occlusion needs *graded* probability shifts. Logprob mode on this model is near-saturated, so removing a field would produce 0/1 jumps and the driver attribution would be noise.
- **Alternative:** logprob everywhere; the explainer becomes uninformative.
- **Cost of the choice:** the explainer's numbers are not directly comparable to the headline, and score-mode has only ~17 distinct values — the tie regime that broke `pr_auc`.

### Calibrator selection on validation NLL, with a parsimony margin rule
- **Decided:** [calibration.py:125](../truthclf/calibration.py#L125) `fit_best`. Platt wins only if the paired bootstrap CI of (temperature NLL − Platt NLL) excludes zero; otherwise temperature (1 param vs 2).
- **Alternative considered and rejected: selecting on validation ECE.**
- **Why not ECE:** measured, ECE cannot distinguish the two calibrators on *any* of the three runs — all three paired difference CIs include zero (run c: −0.0223 [−0.0322, +0.0006]). A criterion that cannot separate the candidates on 1,991 test rows is noisier still on 1,917 validation rows. NLL is a proper scoring rule and its selections are reliable — all three NLL CIs exclude zero.
- **The honest tension, stated in [README.md:245-252](../README.md#L245-L252):** we select on NLL and report ECE, and the two are not the same objective. On run `c`, Platt is reliably better on val NLL (+0.002044 [+0.000366, +0.006277]) and *worse* on test ECE (0.029 → 0.052). That is not selection noise; it is objective mismatch. The margin rule does not fix it and was never going to.

### Threshold objective: balanced accuracy, tuned on calibrated validation
- **Decided:** [evaluation.py:65](../truthclf/evaluation.py#L65).
- **Alternative:** the cost-sensitive threshold at [threshold.py:85](../truthclf/threshold.py#L85) `tune_cost_threshold`, with `c_fp > c_fn` — which the README's own error-cost argument implies is the *right* one for a misinformation setting.
- **UNDOCUMENTED:** nothing explains why the reported number uses the symmetric objective while the prose argues for the asymmetric one. `tune_cost_threshold` is implemented, tested, and **feeds no reported number**. This is the gap most likely to be probed.

### Explainer: leave-one-field-out occlusion + rationale cross-check
- **Decided:** [explain.py:76](../truthclf/explain.py#L76). Four fields occluded individually plus an all-metadata baseline.
- **Alternative:** token-level attribution (LIME/SHAP) — scoped out in [README.md:272-283](../README.md#L272-L283) as wrong-granularity for field-structured metadata.
- **If reversed:** ~6× more calls per point and no clean mapping back to a field.
- **Two magic constants, both UNDOCUMENTED:** `driver_eps=0.05` ([explain.py:77](../truthclf/explain.py#L77)) decides whether a field "drove" a prediction — it directly sets the 26% speaker-driven share. `threshold=0.5` is a *separate* parameter from the predictor's own threshold and does not know about the calibrator.

### Base model: `google/gemma-4-31B-it`
- **Decided:** hard-coded in seven places, e.g. [evaluate_finetuned.py:41](../scripts/evaluate_finetuned.py#L41), [regenerate_results.py:40](../scripts/regenerate_results.py#L40).
- **Recorded in [docs/decisions.md](../docs/decisions.md):** both serverless-inferenceable and LoRA-fine-tunable under one id, so zero-shot and fine-tuned share a base with no confound. Llama splits `-Reference`/`-Turbo`, which *is* the confound.
- **If reversed:** the whole fine-tuning comparison becomes uninterpretable.

---

### Hardening-pass entries

### The calibrator became a shippable artifact
- **Before:** `T` / `(A,B)` existed only inside a process. `results/curves.json` recorded the method *name* and the transformed probabilities, never the parameters.
- **Consequence:** a container calling `predict()` got raw probabilities at 0.5 — ECE ~0.316 instead of ~0.061, threshold 0.5 instead of 0.544. Reproducing reported behaviour required refitting on the whole validation split at startup.
- **Now:** [evaluation.py:93](../truthclf/evaluation.py#L93) `DecisionArtifact`, written to `results/calibrators/*.json`, loaded via `calibrator=` at [zeroshot.py:64](../truthclf/predictors/zeroshot.py#L64). Opt-in: no artifact → raw path unchanged.
- **Model identity is a hard error** ([evaluation.py:149](../truthclf/evaluation.py#L149)). A calibrator maps one model's probability scale; applied to another it still *looks* like probabilities. That is why it is not a warning.

### ECE: equal-mass bins, and the 2–3 occupied bins finding
- **Decided:** default `strategy="quantile"` ([metrics.py:156](../truthclf/metrics.py#L156)); equal-width retained as a diagnostic.
- **Why:** confidence-of-predicted-class lives in [0.5, 1], so half of a [0,1] equal-width grid is structurally unreachable — and on the real runs only **2–3 of 10** bins carried any mass.
- **Concretely:** run `a` reported ECE 0.065 from **2** occupied bins under equal-width; equal-mass gives **0.108** from 7. The old binning was flattering by hiding miscalibration inside one huge bin.
- **Every ECE now ships with its realised bin count** ([metrics.py:179](../truthclf/metrics.py#L179) `ece_bin_report`). A 2-bin and a 10-bin ECE are not the same quantity.

### Sync path: serial → concurrent
- **Before:** `[self.score_one(m) for m in messages_list]` — one round-trip per point, on the default path used by `predict.py` and the README examples.
- **Now:** [llm.py:314](../truthclf/llm.py#L314) `_map`, `ThreadPoolExecutor(max_workers=16)`, `ex.map` preserving order. `max_workers=1` restores serial.
- **Safe because** only misses reach the network and diskcache is concurrent-safe; the counters are the only shared mutable state.

### `cache or ResponseCache()` — construction that was wrong in a non-obvious way
- **What happened:** `ResponseCache` gained `__len__` when schema 2 landed. That made an **empty** cache falsy. `self.cache = cache or ResponseCache()` therefore silently discarded a caller's explicitly-passed fresh cache and used the project default.
- **Why it is instructive:** the bug was introduced by adding a *correct* dunder. Nothing about `__len__` suggests it changes truthiness of a container that is conceptually always "present". `or`-defaulting is only safe for values whose falsy state genuinely means "absent".
- **Fixed:** `ResponseCache() if cache is None else cache`, both clients. No adopted number was affected — every script uses the default — but a container passing its own cache would have written to the wrong store.

### The p=0.5 fallback reached `predict_examples`
- **What happened:** all eight rows in `results/summary.json:predict_examples` — the first table in the reviewer notebook — held `p_true = 0.5` predicting `True`. Every one came from [zeroshot.py:122](../truthclf/predictors/zeroshot.py#L122), where a missing True/False token becomes a neutral 0.5.
- **Why it survived:** once written to JSON, a fallback is indistinguishable from a real prediction, and nothing asserted that a demo table should contain varied probabilities.
- **Contamination audit of the reported splits:** score-mode 5/1991 test and 2/1917 val (all genuine unparseable prose, not the sentinel); logprob 0/1991; fine-tuned 0/1991.

### `pr_auc` tie handling and order-dependence
- **What happened:** average precision accumulated per **sample** rather than per distinct threshold, crediting operating points the classifier cannot realise inside a tie group. **The result depended on row order.** All scores tied: `[1,0,1,0]` → 0.833, `[1,1,0,0]` → 1.000, `[0,0,1,1]` → 0.417; correct answer 0.5 in all three.
- **Why it mattered here:** score-mode emits ~17 distinct values across 9.6k rows, and even the logprob path has 1,134 distinct values across 1,991 rows. "Continuous probabilities" was not protection — the logprob run still moved +0.0108.
- **Fixed:** [metrics.py:105](../truthclf/metrics.py#L105) delegates to sklearn; regression tests cover all-tied and order-permutation.

### The Platt fit never converged, so the Platt branch was unreachable
- **What happened:** 2,000 fixed gradient steps at lr=0.5, no convergence check. Validation NLL 0.859 / 1.157 / **3.500** against temperature's 0.706 / 0.691 / 0.643, and on run `a` a **negative** slope (A = −0.0988) — an inverted calibration.
- **So `fit_best` selected temperature every single time.** The Platt branch looked like a live design choice, had a code path, and could never win.
- **After the fix** (logistic regression on the logit, `C=np.inf`), Platt wins all three and the margin rule never fires. **This one change moved every non-`pr_auc` number in the record.**
- **The lesson worth keeping:** a path can be dead because it is never *selected*, not only because it is never *called*. Coverage would not have shown this. Only comparing the two arms' objective values did.

### Rationale–occlusion agreement is at chance
- **What happened:** the deck's faithfulness statistic, 0.457, is **not distinguishable from chance**. Permutation null 0.436, 95% range [0.393, 0.480], **p = 0.19**.
- **Why the floor is high:** agreement is scored over 5 categories, but a rationale cites **1.59 of 5** on average, and `statement` appears in 183/300 rationales while also being the most common driver. Assuming a 1/5 floor — as "only 46% faithful" implicitly did — is wrong by more than 2×.
- **What is claimed now:** the rationales carry *no detectable information* about the true driver. Evidence of absent signal, not a measured faithfulness rate. Failing to reject at n=300 is not proof of zero.

### The 75.6% in `build_deck.py`
- **What it was:** the deck asserted "Hurts (62.5% vs 75.6%)". It was flagged to me as fabricated. **It was not.** Both figures are in [docs/decisions.md](../docs/decisions.md), from a superseded explainer run with n=164/72 subsets; the current 300-row sample gives n=158/78 and 0.741/0.564.
- **Why the distinction matters:** fabricated means someone wrote fiction, and the fix is a provenance rule. **Stale means the pipeline let a superseded value survive a re-run** — the fix is "read every slide number from the JSON record", which is now a rule in `docs/decisions.md` and enforced by construction in `build_deck.py`.
- **The stronger replacement finding:** speaker-driven predictions sit at their subset's **majority-class baseline** (0.564 vs 0.564, Δ = +0.000 [−0.141, +0.128]) — no signal beyond the class prior. Statement-driven are +0.215 [+0.108, +0.285] above theirs.

---

# Section 3 — Load-bearing subtleties

**Component ordering in `_components` must be deterministic.**
[data.py:288-291](../truthclf/data.py#L288-L291). Components are keyed in a dict populated while iterating rows in order, so each key is inserted when its *first* row is seen — independent of which member `DisjointSet` picks as root. That ordering feeds the seeded shuffle in `_stratified_assign`. Swapping the union-find implementation was verified to leave all seven split hashes byte-identical. *Remove it and every reported number changes with no error.*

**`drop_intermediate=False` in `candidate_thresholds`.**
[threshold.py:58](../truthclf/threshold.py#L58). sklearn's default prunes thresholds off the ROC convex hull — right for plotting, wrong here: a pruned threshold can still be optimal for balanced accuracy. *Symptom: an existing test started selecting 0.8 instead of the separating 0.7.*

**`thresholds[0]` is dropped.**
[threshold.py:59](../truthclf/threshold.py#L59). `roc_curve` prepends `+inf` so the curve starts at (0,0). That is "predict everything negative", not an operating point. *Symptom: an unreachable threshold could be selected and every prediction would be False.*

**The `>=` convention is load-bearing and stated.**
[threshold.py:10-16](../truthclf/threshold.py#L10-L16). `roc_curve` thresholds are defined for `>=`. With `>`, every candidate is off by one operating point *on tied scores* — the regime this dataset is in.

**Raw-then-parse in the cache.**
Classify stores the API's logprobs structure and flattens on read ([llm.py:446](../truthclf/llm.py#L446)). Schema 1 stored the flattened map, so changing the parser silently reinterpreted stored values behind an unchanged key. *This is why all 3,908 legacy classify entries were quarantined rather than migrated.*

**Failed batch rows are never cached.**
[llm.py:628-636](../truthclf/llm.py#L628-L636). Schema 1 wrote `""`, which was then served as a hit forever, permanently injecting p=0.5. *Symptom if reintroduced: a transient failure becomes a permanent silent wrong answer.*

**`_assert_complete` is called three times, not once.**
[regenerate_results.py](../scripts/regenerate_results.py). The single check ran before the explainer read anything, so a missing cache produced 300 silent p=0.5 defaults and a fabricated table at exit code 0.

**`zero_division` differs by metric, deliberately.**
`macro_f1` uses `0.0` ([metrics.py:77](../truthclf/metrics.py#L77)); `precision_recall` uses `np.nan` ([metrics.py:88](../truthclf/metrics.py#L88)). With `np.nan`, sklearn *drops* the undefined class from a macro average — reporting macro-F1 = 1.0 for a run that never tested one class. Per-class values are reported individually, where undefined should read as undefined.

**`roc_auc` / `pr_auc` return NaN, not 0.0, on degenerate input.**
[metrics.py:95](../truthclf/metrics.py#L95), [:105](../truthclf/metrics.py#L105). sklearn warns and returns 0.0 for no-positives, which is indistinguishable from a terrible ranking. NaN is what `bootstrap_ci` drops-and-counts.

**The explainer batches 6N rows into one `predict()`.**
[explain.py:81-91](../truthclf/explain.py#L81-L91). Pinned by a test asserting exactly one call of N×6.

---

# Section 4 — Where I am exposed

Ranked by how bad it is if someone finds it first.

### 1. `tune_cost_threshold` is argued for and never used — UNDOCUMENTED
The README argues a false-positive is the costlier error in a misinformation setting, yet every reported number uses symmetric balanced accuracy. The cost-sensitive function exists ([threshold.py:85](../truthclf/threshold.py#L85)) and is tested. **There is no written reason for the mismatch.** A reviewer who reads the error-cost paragraph and then greps for `c_fp` finds an implemented, tested, unused function.

### 2. `driver_eps = 0.05` — a magic constant that sets a headline number
[explain.py:77](../truthclf/explain.py#L77). This single undocumented threshold decides the 26% speaker-driven share and therefore the entire "speaker-driven sits at the baseline" finding. No sensitivity analysis exists. *"How does that finding move if driver_eps is 0.03?"* — I cannot currently answer.

### 3. Numbers on slide 6 that no file in the repo backs
[build_deck.py:338-344](../build_deck.py#L338-L344): "Eval loss 0.32 → 0.199 over 3 epochs", "~14 min, ~$2". These come from the Together training log and an endpoint session, neither of which is in the repo. They are prose in the deck builder — the exact pattern the JSON-numbers rule exists to prevent, still present because no JSON records them.

### 4. Reported numbers that cannot be regenerated without paying
Run `c` replays `ft_eval_cache.json`. Regenerating it needs a dedicated endpoint (~$2). The endpoint is gone. **If that file were lost, no fine-tuned number in the record could be reproduced without re-provisioning.** It is tracked, which is the only mitigation.

### 5. `explain()` has no `_assert_complete` equivalent
The predictors' 0.5 fallback is silent by design ([zeroshot.py:109](../truthclf/predictors/zeroshot.py#L109), [:122](../truthclf/predictors/zeroshot.py#L122)). `PredictionResult.parse_failures` counts it, but **`explain()` never inspects it**, and the aggregate reports no parse-failure rate. An explainer run on a degraded cache produces a plausible driver distribution with no warning — the same class of failure as the entrypoint bug, still open on this path.

### 6. Untested paths that touch money or the network
`FinetunedPredictor.fine_tune` ([finetuned.py:38](../truthclf/predictors/finetuned.py#L38)), the whole of `TogetherBatchClient._run`, and `experiments.py` (**0% coverage**, and it is what the README's code examples import). The tenacity retry policy — specifically "do not retry auth errors" — is asserted nowhere; a wrong exception tuple silently restores the old waste.

### 7. Tests that assert shape, not behaviour
- [test_data.py:97](../tests/test_data.py#L97), [:106](../tests/test_data.py#L106): `len(train)+len(test)==len(clean)` proves a partition, not that stratification worked.
- [test_interchangeability.py:82](../tests/test_interchangeability.py#L82): `hasattr(ft, "output_name")` — the *handle* contract is never exercised.
- [test_metrics.py:116](../tests/test_metrics.py#L116): bin count and sum, not bin correctness (mitigated by the calibration_curve pin).

### 8. Errors still swallowed
[build_deck.py:215](../build_deck.py#L215) `except Exception` around figure generation — a broken figure prints `[fig] FAILED` and the deck builds anyway with a missing image. [evaluate_finetuned.py:75-77](../scripts/evaluate_finetuned.py#L75-L77), [:179](../scripts/evaluate_finetuned.py#L179): endpoint cleanup failures are printed, not raised — deliberate (teardown must not mask the original error) but it means a **leaked endpoint is a log line, not a failure**.

### 9. Code/README disagreements
- README [:166](../README.md#L166) says the endpoint script costs "~$2, ~10 min"; the deck says "~14 min". Both are recollections.
- README quotes "a live refetch reproduced these to within 0.003" — true, and reproducible only while `.llm_cache/` exists, which is gitignored.

### 10. `predict([], labels=[])` raises
Verified: `predict([])` returns an empty result fine, but with `labels=[]` sklearn raises `ValueError: Found empty input array`. Untested, undefined, and a plausible edge for a network service.

---

# Section 5 — Interfaces, for downstream reuse

## Signatures

```python
ZeroShotPredictor(model, variant="full", threshold=0.5, client=None, seed=0,
                  use_logprobs=True, neutral_score=50.0, calibrator=None)
    .predict(points, labels=None) -> PredictionResult
    .rationale(rows, max_tokens=64) -> list[str]

FinetunedPredictor(base_model, served_model=None, variant="full", scheme="primary",
                   client=None, threshold=0.5, use_logprobs=True, calibrator=None)
    .fine_tune(training_dataset, val_rows=None, n_epochs=3, learning_rate=1e-5,
               suffix="truthclf_sft", lora=True, train_on_inputs="auto",
               poll=True, workdir="ft_data", poll_interval=30) -> str | None
    .predict(points, labels=None) -> PredictionResult

explain(model, points, labels=None, with_rationale=True,
        threshold=0.5, driver_eps=0.05) -> dict
```

`points`: `list[truthclf.data.Row]` — a dataclass, **not** a dict. `labels`:
`list[int]` in {0,1}. `PredictionResult` ([base.py:16](../truthclf/predictors/base.py#L16)):
`scores` (list or `None` in logprob mode), `probs`, `preds`, `threshold`,
`parse_failures`, `n`, `metrics` (dict or `None`).

`fine_tune` returns `model_output_name` — a string handle — **or `None` if
`poll=False`**, in which case only `self.job_id` is set.

## Behaviour under stress

**Verified by running it** where marked; otherwise stated as untested.

| condition | `predict()` | `explain()` | `fine_tune()` |
|---|---|---|---|
| empty input | **verified:** returns empty `PredictionResult`. With `labels=[]` **raises `ValueError`** from sklearn | **verified:** returns `{"per_point": []}`; with `labels=[]` raises | **untested** |
| batch of 1000 | sync: 1000 calls over 16 threads; batch: one job. **No chunking anywhere** — 1000 messages go into one JSONL / one thread pool | 6000 rows in one `predict()` call | n/a |
| missing field | **verified:** `AttributeError` on the first missing attribute. No validation layer | same | **untested** |
| null `statement` | **verified: silently succeeds.** `statement_clean or statement` yields `None`, f-string renders `"None"` into the prompt. **No error, no warning** | same | **untested** |
| wrong type | **untested** | **untested** | **untested** |
| provider failure mid-batch | sync: tenacity retries transient classes, then raises — **partial results are lost, though successful rows are already cached**. Batch: failed rows → `""`, counted in `error_count`, **not cached**, surface as `parse_failures` | inherits; **`explain()` never inspects `parse_failures`** | poll loop raises `RuntimeError` on FAIL/ERROR/CANCEL |
| provider timeout | sync: `APITimeoutError` is retryable, 4 attempts, exponential + jitter, then raises. Batch: `TimeoutError` after `max_wait=14400s` (4 h) | inherits | **no timeout on the poll loop — it blocks indefinitely** unless the job reaches a terminal state |
| cold start, no cache | every point is a live call. `predict()` works; the *evaluation entrypoint* refuses (`SystemExit`) | same | unaffected |

## Concurrency and shared state

**Can they be called concurrently by several processes?** Mostly yes, with named caveats.

| shared state | where | safe? |
|---|---|---|
| `.llm_cache/` diskcache | [llm.py:118](../truthclf/llm.py#L118) | **Yes.** SQLite/WAL, per-key writes, pinned by a multi-process test. **Same host only** — not safe over NFS/EFS |
| `ft_eval_cache.json` | [evaluate_finetuned.py:170](../scripts/evaluate_finetuned.py#L170) | **No.** Whole-file `json.dump`, non-atomic, no lock. Two processes → lost updates or a truncated file |
| `results/*.json` | `regenerate_results.py` | **No.** Same pattern. Last writer wins |
| `ft_data/train.jsonl` | [finetuned.py:53-64](../truthclf/predictors/finetuned.py#L53-L64) | **No.** Fixed path `workdir="ft_data"`; two concurrent `fine_tune` calls overwrite each other's upload file |
| `NamedTemporaryFile(delete=False)` | [llm.py:580](../truthclf/llm.py#L580) | Unique per call, but **never deleted** — leaks one JSONL per batch |
| provider rate limits | — | **Unmanaged.** No global limiter. 4 containers × 16 workers = 64 concurrent requests |
| `RETRYABLE_ERRORS` | [llm.py:60](../truthclf/llm.py#L60) | Module-level, computed at import, read-only. Safe |
| `_ENC` (tiktoken) | [llm.py:84](../truthclf/llm.py#L84) | Import-time singleton, read-only. Safe |
| `TogetherClient._client` | [llm.py:267](../truthclf/llm.py#L267) | **Lazy, not locked.** Concurrent first-calls can each build a client; benign but wasteful |

## What assumes a single interactive process

- **`load_dotenv()` at import** ([__init__.py:20](../truthclf/__init__.py#L20)). Reads a file next to the CWD at import time. In a container the key normally comes from the environment; this is harmless but means import has a filesystem side effect.
- **Cache path anchored to the source tree** ([llm.py:117-118](../truthclf/llm.py#L117-L118)): `_PROJECT_ROOT/.llm_cache`. **Not configurable** — no env var, no parameter on the default path. A read-only image or a shared volume cannot be pointed at without editing the module.
- **`data_path="data.csv"` relative** throughout `experiments.py` and every script — depends on CWD.
- **`os.chdir(ROOT)` in [build_deck.py:30](../build_deck.py#L30)** — mutates process CWD globally.
- **`workdir="ft_data"` relative** ([finetuned.py:38](../truthclf/predictors/finetuned.py#L38)).
- **Blocking poll loops with `time.sleep`** in `fine_tune` and `_run` — fine for a script, wrong for a request handler.
- **`results/calibrators/*.json` are relative paths** in the README example.

## Runtime requirements per endpoint

**`predict()` (zero-shot):** `TOGETHER_API_KEY`; write access to `.llm_cache/`'s parent; optionally the calibrator artifact. To match the record: `results/calibrators/zero_shot_decision_baseline.json` and `use_logprobs=True`.

**`predict()` (fine-tuned):** the above, plus a **served** model id — either a live dedicated endpoint or a serverless-capable id.

**`explain()`:** whatever the wrapped model needs. Note it defaults to
`with_rationale=True`, so it issues an extra N completion calls unless disabled.

**`fine_tune()`:** `TOGETHER_API_KEY`, a writable `ft_data/`, and a `training_dataset` of `Row` objects.

**The evaluation entrypoint:** `data.csv`, `ft_eval_cache.json`, plus either `.llm_cache/` (live) or `.llm_cache.json` (archive). No key for `--source archive`.

---

# Section 6 — The fine-tuned model's serving story

### The handle
`makisntpap_17e5/gemma-4-31B-it-gemma_truth_sft-c7afbf0d`. Recorded in
`ft_artifacts.json` (`model_output_name`, alongside job id `ft-164d480e-cf7f` and
both upload file ids), and **hard-coded** at
[evaluate_finetuned.py:42](../scripts/evaluate_finetuned.py#L42) and
[regenerate_results.py:41](../scripts/regenerate_results.py#L41). It is also now the
`model` field of `results/calibrators/fine_tuned_decision.json`, which is the
only place it is bound to a calibrator.

### Serverless or dedicated?
**Reporting, not inferring:** the code assumes **dedicated**. The module docstring
of `evaluate_finetuned.py` states serverless serving was unavailable for this
base, `docs/decisions.md:281` records the same, and the script provisions an
endpoint unconditionally when rows are missing.

**Settled by probe on 2026-08-09 — no longer an inference.** One
`chat.completions.create(model=FT, max_tokens=1)` with nothing provisioned returns
**HTTP 400, `code: model_not_available`**: *"Unable to access non-serverless model
... create and start a new dedicated endpoint"*. Serverless is **not** available
for this LoRA. The error is deterministic and correctly non-retryable.
`endpoints.list_hardware(model=FT)` returns `2x_nvidia_h100_80gb_sxm`,
availability `available` — a dedicated endpoint is possible, which is the other
question. See docs/decisions.md, 2026-08-09.

### If dedicated
- **Hardware:** `2x_nvidia_h100_80gb_sxm` ([:43](../scripts/evaluate_finetuned.py#L43)), `min_replicas=max_replicas=1` ([:114](../scripts/evaluate_finetuned.py#L114)).
- **Provisioning:** polled to `STARTED` with a 20-minute deadline ([:48](../scripts/evaluate_finetuned.py#L48)). Observed once, recorded as ~14 min in the deck. **That figure is a recollection, not a logged measurement.**
- **Hourly cost:** **not recorded anywhere in the repo.** The ~$2 figure is a session total, not a rate. I cannot derive $/hour from anything here.
- **Teardown:** [:176](../scripts/evaluate_finetuned.py#L176), in a `finally`, so it covers exceptions and KeyboardInterrupt. Plus a 30-minute kill switch ([:49](../scripts/evaluate_finetuned.py#L49)) and idempotent pre-cleanup of leftovers ([:64-77](../scripts/evaluate_finetuned.py#L64-L77)).
- **If nothing tears it down:** it bills until manually deleted. The `finally` covers process-level failures but **not SIGKILL, not a host crash, and not a container OOM.** The delete failure path prints `!! delete FAILED ... DELETE MANUALLY` and continues — a log line, not an alert.

### What the current code assumes
A **short-lived script run by a human who watches it.** Evidence: blocking poll loops with `time.sleep`; incremental `json.dump` checkpoints every 200 rows for crash recovery; a wall-clock kill switch; an audit line printing endpoint uptime; and the whole design of caching results to a file so the endpoint is needed exactly once. There is no daemon, no health check, no reconnection, and no way to attach to an already-running endpoint — `FinetunedPredictor` takes `served_model=` and assumes something else provisioned it.

### What the calibrator artifact adds to startup
One file, `results/calibrators/fine_tuned_decision.json`, **~600 bytes**. Load is
a `json.load` plus a dataclass construction and a string comparison —
**microseconds**. It replaces what would otherwise be: load validation
probabilities, fit two calibrators, run a 1,000-draw paired bootstrap refitting
both per draw, then tune a threshold over ~1,900 candidates. That is the
difference between negligible and several seconds of CPU per container start.

### Latency and cost
**Reporting what is measured:** the batch refetch of 6,075 rows took 17.2 min
wall-clock at ~$0.33 total (`google/gemma-4-31B-it`, $0.39/$0.97 per 1M tokens,
~150 input tokens per row).

**Inferring, and flagging it as inference:** a single cold request against a
*running* endpoint is one round-trip, `max_tokens=1` — order 0.3–0.5 s based on
the recorded `avg_latency` of the dev runs. A batch of 100 through the sync
client at 16 workers is ~7 waves, so order 3–5 s. **If the endpoint is not
already running, add the full provisioning time — the cold-start cost is
dominated entirely by provisioning, not inference.** None of these are measured
for the fine-tuned path specifically; the only measured fine-tuned timing is the
whole 3,908-row session.

### Four containers, four hosts, same predictor
1. **Four dedicated endpoints, or four failures.** Nothing coordinates provisioning. `cleanup_leftovers` matches on display name `truth-gemma-ft-eval` ([:47](../scripts/evaluate_finetuned.py#L47)) and **deletes any endpoint it finds** — so container B's cleanup can delete container A's live endpoint mid-inference. That is the sharpest failure here.
2. **`ft_eval_cache.json` corruption.** Whole-file `json.dump`, four writers, no locking.
3. **The response cache does not help.** `.llm_cache/` is SQLite on a local path — four hosts means four independent caches, so no sharing and 4× the spend.
4. **Rate limits hit collectively** with no shared limiter; tenacity backs off per-process, which under contention synchronises rather than spreads load (jitter mitigates but does not solve).
5. **The calibrator artifact is the one piece that is genuinely safe** — read-only, tiny, content-addressed by the model id it checks against.
