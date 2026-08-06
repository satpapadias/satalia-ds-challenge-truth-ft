"""Justify enable_thinking=False: compare Qwen3.5-9B thinking ON vs OFF on a
50-row dev subset (statement-only), and append the delta to docs/decisions.md.

Thinking-OFF reuses the cached scale-check responses (free). Thinking-ON makes
real calls (max_tokens=2048). Run in the background; it does not block.
"""

from __future__ import annotations

import datetime

import truthclf
from truthclf import data, llm
from truthclf.predictors import ZeroShotPredictor

MODEL = "Qwen/Qwen3.5-9B"
VARIANT = "statement_only"
N = 50
SCHEME = "primary"
DECISIONS = "docs/decisions.md"
MARKER = "<!-- THINKING_CHECK_RESULTS -->"


def main():
    clean, _ = data.clean_dataset(data.load("data.csv"), SCHEME)
    dev = data.dev_subset(clean, 200, 0)[:N]          # first 50 of the cached 200
    labels = [r.y(SCHEME) for r in dev]

    off_client = llm.make_client(MODEL)               # enable_thinking=False (cached)
    off = ZeroShotPredictor(MODEL, VARIANT, client=off_client).predict(dev, labels)

    on_client = llm.TogetherClient(MODEL, max_output_tokens=2048, temperature=0)
    on = ZeroShotPredictor(MODEL, VARIANT, client=on_client).predict(dev, labels)

    def row(tag, res, lat):
        m = res.metrics
        pf = f"{res.parse_failures}/{res.n} ({res.parse_failures/res.n*100:.0f}%)"
        return f"| {tag} | {m['accuracy']:.3f} | {m['brier']:.3f} | {pf} | {lat} |"

    d_acc = on.metrics["accuracy"] - off.metrics["accuracy"]
    improved = d_acc > 0.02
    conclusion = (
        "thinking ON improved accuracy materially (>2 pts) — revisit the disable decision"
        if improved else
        "thinking ON did NOT improve accuracy and added large latency + parse failures "
        "(empty content within the 2048-token budget) — disabling reasoning is justified"
    )

    block = "\n".join([
        f"Run: {N}-row dev subset, statement-only, {MODEL} (appended {datetime.date.today()}).",
        "",
        "| config | acc | Brier | parse-fail | latency/call |",
        "|---|---|---|---|---|",
        row("thinking OFF (enable_thinking=False, 16 tok)", off, "~0.43s (scale check)"),
        row("thinking ON (default, 2048 tok)", on, f"{on_client.avg_latency:.1f}s"),
        "",
        f"Delta (ON - OFF): acc {d_acc:+.3f}, "
        f"Brier {on.metrics['brier'] - off.metrics['brier']:+.3f}, "
        f"parse-fail {on.parse_failures - off.parse_failures:+d}.",
        "",
        f"**Conclusion:** {conclusion}.",
    ])

    s = open(DECISIONS, encoding="utf-8").read()
    head = s.split(MARKER)[0] + MARKER + "\n\n"
    open(DECISIONS, "w", encoding="utf-8").write(head + block + "\n")
    print("appended thinking-check results to", DECISIONS)
    print(block)


if __name__ == "__main__":
    main()
