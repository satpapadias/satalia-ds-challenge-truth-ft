# dist/ — frozen submission artifacts

Everything in this directory is a **point-in-time snapshot of the take-home as
submitted**. It is kept for provenance and is deliberately **not maintained**:

- `truthclf-submission/` — a full copy of the package, tests, scripts, notebook
  and `results/summary.json` exactly as delivered.
- `satalia-serafeim-papadias.pptx` / `.pdf` — the submitted deck.
- `truthclf-submission.zip` — the delivered archive.

## Do not edit these files

They are the record of what was handed over. Fixes go to the live package at the
repository root, never here. In particular `dist/truthclf-submission/results/summary.json`
is **not** regenerated when the root results are.

## Known defect in this snapshot

`truthclf-submission/truthclf/metrics.py::pr_auc` computes average precision by
accumulating precision **per sample** instead of per distinct score threshold.
Inside a group of tied scores it credits operating points the classifier cannot
realise, which makes the result depend on row order — with all scores tied it
returns 0.417 / 0.833 / 1.000 for the same labels in different orderings, where
the correct answer is the positive prevalence in all three.

The dataset is squarely in that regime: score-mode elicitation emits ~17 distinct
values across 9.6k rows, and even the logprob path has only 1,134 distinct values
across the 1,991 test rows.

**Affected numbers in this snapshot** (`truthclf-submission/results/summary.json`,
and the PR-AUC column on slide 7 of the deck):

| entry | published (wrong) | correct |
|---|---|---|
| `zero_shot_score_secondary.pr_auc` | 0.659964 | 0.679616 |
| `zero_shot_decision_baseline.pr_auc` | 0.679441 | 0.690223 |
| `fine_tuned_decision.pr_auc` | 0.716484 | 0.722338 |

Every other metric in the snapshot reproduces to six decimal places and is
unaffected.

## Deck lineage

`build_deck.py` produces exactly one deck. `satalia-serafeim-papadias.pptx` here
is a PowerPoint re-save of that output (65 archive entries and 2 speaker-note
parts, against the generator's 83 and 20 — the re-save dropped most notes). It is
frozen with the rest of this directory and was NOT regenerated; the corrected
decks live at the repository root.

**Fixed at the repository root in commit `1619744`** ("metrics: replace
hand-rolled implementations with scikit-learn / statsmodels"), which delegates to
`sklearn.metrics.average_precision_score` and adds regression tests covering the
all-tied and order-permutation cases.
