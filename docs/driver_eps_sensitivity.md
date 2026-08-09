# `driver_eps` sensitivity

`explain.py` assigns each point a *dominant driver*: the metadata field whose
removal moves the predicted probability most, **provided that move exceeds
`driver_eps`**; otherwise the point is attributed to the statement itself.

`driver_eps = 0.05` was an undocumented constant. It sets the speaker-driven
share and therefore the headline claim that speaker-driven predictions sit at
their subset's majority-class baseline. This is the sweep that was missing.

Method: 300-row speaker-disjoint test sample, score-mode probabilities replayed
from the schema-2 cache, **zero API calls**. Accuracy is compared against the
majority-class rate **of that driver's own subset** (the subsets have different
class balance, so a global baseline would manufacture a difference). CIs are a
10,000-draw paired bootstrap of `accuracy − subset baseline`.

## Sweep

| `driver_eps` | statement n | speaker n | other n | **speaker** acc | base | Δ | 95% CI | **statement** acc | base | Δ | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 137 | 88 | 75 | 0.5795 | 0.5227 | +0.0568 | [−0.0909, +0.1477] | 0.7372 | 0.5328 | +0.2044 | [+0.0949, +0.2847] |
| 0.02 | 137 | 88 | 75 | 0.5795 | 0.5227 | +0.0568 | [−0.0909, +0.1477] | 0.7372 | 0.5328 | +0.2044 | [+0.0949, +0.2847] |
| 0.03 | 137 | 88 | 75 | 0.5795 | 0.5227 | +0.0568 | [−0.0909, +0.1477] | 0.7372 | 0.5328 | +0.2044 | [+0.0876, +0.2774] |
| **0.05** | **158** | **78** | **64** | **0.5641** | **0.5641** | **+0.0000** | **[−0.1410, +0.1154]** | **0.7405** | **0.5253** | **+0.2152** | **[+0.1076, +0.2848]** |
| 0.08 | 158 | 78 | 64 | 0.5641 | 0.5641 | +0.0000 | [−0.1410, +0.1154] | 0.7405 | 0.5253 | +0.2152 | [+0.1076, +0.2848] |
| 0.10 | 208 | 53 | 39 | 0.5283 | 0.5660 | −0.0377 | [−0.2075, +0.0943] | 0.7212 | 0.5192 | +0.2019 | [+0.1058, +0.2644] |
| 0.15 | 233 | 40 | 27 | 0.5000 | 0.5750 | −0.0750 | [−0.2750, +0.1000] | 0.7124 | 0.5107 | +0.2017 | [+0.1073, +0.2532] |

Speaker-driven share: 29.3% (0.01–0.03) → 26.0% (0.05–0.08) → 17.7% (0.10) → 13.3% (0.15).

## Verdict

**The finding holds across the entire swept range, 0.01 to 0.15.** At every
value the speaker-driven Δ-vs-baseline CI **includes zero** — speaker-driven
predictions are never distinguishable from their subset's majority-class
baseline. And at every value the statement-driven CI is **strictly positive** —
statement-driven predictions are always meaningfully above theirs. Neither
conclusion depends on the constant.

**What IS an artifact of 0.05:** the exact-zero coincidence. "0.564 vs 0.564,
Δ = +0.000" holds only for `driver_eps` in [0.05, 0.08]. Elsewhere the point
estimate wanders between +0.057 and −0.075. So the *phrasing* "sits exactly at
the baseline" is a property of one bin boundary; the *claim* "indistinguishable
from the baseline" is not. The README and deck already state it as
"indistinguishable", which the sweep supports — but anyone quoting the
coincidence as if it were meaningful is over-reading it.

**Where it would break:** nowhere inside the swept range. The direction of the
point estimate flips sign between 0.08 and 0.10 (speaker-driven goes from
at-baseline to slightly below), but the interval spans zero on both sides, so
nothing survives as a claim either way. Below 0.01 every point acquires a
metadata driver and the "statement" category empties; above ~0.2 the speaker
subset gets too small to say anything (n=40 at 0.15 already gives a ±0.19
interval).

**Quantisation.** Only four distinct outcomes appear across seven values.
Score-mode elicitation emits ~17 distinct probabilities, so occlusion deltas are
near-multiples of 0.05 and the threshold can only land between them: 0.01/0.02/0.03
are identical, and so are 0.05/0.08. `driver_eps` has far less resolution than its
two decimal places suggest — a point worth knowing before tuning it.

## Recommendation

**Keep 0.05.** It sits in the middle of the widest stable plateau (0.05–0.08),
and no value in range changes either conclusion. Not changed without approval.
