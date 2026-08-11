"""MCP server exposing dataset access, evaluation metrics and retrieval.

This server owns data.csv and a TF-IDF index over the training split. It holds
no provider credential and makes no network calls, which is why retrieval uses
TF-IDF rather than a hosted embedding model.

Run:  python -m truthclf_mcp.data_tools [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from typing import Annotated, Literal

import numpy as np
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from truthclf import data as D
from truthclf import metrics as M

from .adapter import (MAX_DATASET_ROWS, MAX_METRICS_LENGTH,
                      MAX_RETRIEVAL_QUERIES, LabelValue, Scheme, check_batch,
                      coerce_labels)
from .errors import LeakageAssertionFailed, RowNotInSplit, tool_errors

# Anchored to the project root rather than the working directory, so the server
# finds the dataset regardless of where it was launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get("TRUTHCLF_DATA", os.path.join(PROJECT_ROOT, "data.csv"))

# Split parameters used by the reported evaluation. Fixed here rather than left
# to the caller, because changing any of them changes split membership and so
# every downstream number. They are echoed in each response rather than left
# implicit.
SCHEME = "primary"
VAL_FRAC = 0.2
TEST_FRAC = 0.2
SPLIT_SEED = 0

server = MCPServer(
    name="truthclf-data-tools",
    instructions="Dataset access, evaluation metrics, and train-only k-NN "
                 "retrieval for the truthclf truthfulness classifier. No model "
                 "calls and no provider credential: every tool here is local "
                 "and deterministic.",
)


# ---------------------------------------------------------------------------
# Corpus + index, built once at start-up
# ---------------------------------------------------------------------------
class _Corpus:
    """data.csv, cleaned and split, plus the train-only retrieval index.

    Built at import so a broken data.csv or a failed index build kills the
    server at boot rather than on whichever request happens to arrive first.
    """

    def __init__(self, path: str, scheme: str = SCHEME):
        self.path = path
        self.scheme = scheme
        self.raw = D.load(path)
        self.clean, self.report = D.clean_dataset(self.raw, scheme)
        self.train, self.val, self.test = D.speaker_disjoint_3way(
            self.clean, VAL_FRAC, TEST_FRAC, SPLIT_SEED, scheme)
        self.splits = {"train": self.train, "val": self.val, "test": self.test,
                       "clean_all": self.clean, "raw_all": self.raw}

        # Leakage checks, computed rather than assumed: a non-zero count means
        # a speaker or a repeated statement crosses the train/test boundary and
        # every downstream number is suspect.
        self.speakers_cross = len(D.speakers_cross(self.train, self.test))
        self.normkeys_cross = len(D.normkeys_cross(self.train, self.test))

        self._build_index()

    def _build_index(self):
        """Build a TF-IDF + cosine index over training statements only.

        scikit-learn rather than sentence embeddings: it is already a required
        dependency, it is deterministic and runs offline, and a hosted embedding
        model would require an API credential on a server that is otherwise
        free of one.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        # Index the cleaned text, since that is what the predictor is shown;
        # neighbours are then neighbours in the text the model actually sees.
        corpus = [r.statement_clean or r.statement for r in self.train]
        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            strip_accents="unicode", lowercase=True, min_df=1)
        matrix = self.vectorizer.fit_transform(corpus)
        # Exact brute-force search: the training split is a few thousand rows,
        # so an approximate index would trade correctness for an unneeded gain.
        self.index = NearestNeighbors(metric="cosine", algorithm="brute").fit(matrix)
        self.index_size = matrix.shape[0]
        self.vocab_size = len(self.vectorizer.vocabulary_)

        # Held as sets so every result can be checked against them, rather than
        # relying on the index having been built correctly.
        self.train_ids = {r.row_id for r in self.train}
        self.non_train_ids = ({r.row_id for r in self.val}
                              | {r.row_id for r in self.test})

    @property
    def index_method(self) -> str:
        return (f"tfidf word 1-2gram, sublinear_tf, {self.vocab_size} features "
                f"+ cosine (sklearn NearestNeighbors, brute)")


CORPUS = _Corpus(DATA_PATH)


def _split_info(split: str) -> dict:
    return {
        "split": split,
        "n_train": len(CORPUS.train), "n_val": len(CORPUS.val),
        "n_test": len(CORPUS.test),
        "class_balance": _balance(CORPUS.splits[split]),
        "method": "data.speaker_disjoint_3way",
        "speakers_cross": CORPUS.speakers_cross,
        "normkeys_cross": CORPUS.normkeys_cross,
    }


def _balance(rows) -> dict:
    n_true, n_false, frac = D.class_balance(rows, CORPUS.scheme)
    return {"n_true": n_true, "n_false": n_false, "true_fraction": frac}


def _provenance() -> dict:
    return {"data_path": CORPUS.path, "scheme": CORPUS.scheme,
            "val_frac": VAL_FRAC, "test_frac": TEST_FRAC,
            "split_seed": SPLIT_SEED}


def _row_out(r, scheme: str, include_labels: bool, include_metadata: bool = True) -> dict:
    out = {"row_id": r.row_id, "statement": r.statement}
    if include_metadata:
        out.update({f: getattr(r, f) for f in
                    ("subjects", "speaker_name", "speaker_job", "speaker_state",
                     "speaker_affiliation", "statement_context")})
    if include_labels:
        out["label"] = r.label
        out["y"] = r.y(scheme)
    else:
        out["label"] = None
        out["y"] = None
    return out


def _clean_json(x):
    """Convert NaN to None for JSON output.

    JSON has no NaN. Encoding it as 0.0 would make an undefined metric -- ROC
    AUC with a single class present, for instance -- indistinguishable from a
    genuinely poor score, which is the distinction NaN exists to preserve.
    """
    if isinstance(x, dict):
        return {k: _clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean_json(v) for v in x]
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return None if math.isnan(f) else f
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


# ---------------------------------------------------------------------------
# 1. dataset
# ---------------------------------------------------------------------------
@server.tool(
    title="Dataset access",
    description="Rows from data.csv under the project's own cleaning and "
                "leakage-controlled speaker-disjoint split. Read-only.",
)
@tool_errors
def dataset(
    split: Literal["train", "val", "test", "clean_all", "raw_all"] = "test",
    scheme: Scheme = "primary",
    select: Literal["page", "sample", "by_row_id"] = "page",
    limit: Annotated[int, Field(
        ge=1, description=f"Rows to return. Ceiling {MAX_DATASET_ROWS}.")] = 100,
    offset: Annotated[int, Field(ge=0)] = 0,
    seed: int = 0,
    row_ids: list[int] | None = None,
    include_labels: bool = True,
) -> dict:
    """Return rows from a split, with the split's provenance and integrity counts."""
    check_batch(limit, MAX_DATASET_ROWS, "rows", "dataset")
    if scheme != CORPUS.scheme:
        # The split stratifies on each group's majority label under a scheme,
        # so a different scheme changes split membership, not just the y column.
        raise ValueError(
            f"scheme={scheme!r} but this server was built with {CORPUS.scheme!r}. "
            "The split stratifies on the majority label under a scheme, so "
            "serving another scheme's y values against this split's membership "
            "would silently mix two designs. Restart with TRUTHCLF_SCHEME.")

    rows = CORPUS.splits[split]
    total = len(rows)

    if select == "by_row_id":
        if not row_ids:
            raise ValueError("select='by_row_id' requires a non-empty row_ids")
        check_batch(len(row_ids), MAX_DATASET_ROWS, "row_ids", "dataset")
        by_id = {r.row_id: r for r in rows}
        missing = [i for i in row_ids if i not in by_id]
        if missing:
            raise RowNotInSplit(
                f"{len(missing)} row_ids are not in split {split!r} "
                f"(first few: {missing[:5]}). Not silently dropped: a shortened "
                "set answers a different question than the one asked.")
        chosen = [by_id[i] for i in row_ids]
        nxt = None
    elif select == "sample":
        # Reuse the package sampler: seeded and deterministic, and it draws the
        # same rows as the analysis scripts.
        chosen = D.dev_subset(rows, n=limit, seed=seed)
        nxt = None
    else:
        chosen = rows[offset:offset + limit]
        nxt = offset + limit if offset + limit < total else None

    out = {
        "rows": [_row_out(r, scheme, include_labels) for r in chosen],
        "returned": len(chosen),
        "total_in_split": total,
        "next_offset": nxt,
        "split_info": _split_info(split),
        "clean_report": {"n_in": CORPUS.report.n_in, "n_out": CORPUS.report.n_out,
                         "n_dropped": CORPUS.report.n_dropped,
                         "n_dropped_groups": len(CORPUS.report.dropped_groups)},
        "provenance": _provenance(),
        "warnings": [],
    }
    if split == "raw_all":
        out["warnings"].append(
            f"split='raw_all' returns rows BEFORE clean_dataset: "
            f"{CORPUS.report.n_dropped} rows in "
            f"{len(CORPUS.report.dropped_groups)} self-contradictory repeat "
            "groups are still present. Those rows carry conflicting binary "
            "labels for the same statement text.")
    return _clean_json(out)


# ---------------------------------------------------------------------------
# 2. metrics
# ---------------------------------------------------------------------------
@server.tool(
    title="Evaluation metrics",
    description="Metric bundle with bootstrap CIs, the ECE occupied-bin count, "
                "a majority-class baseline, and an optional paired comparison. "
                "Delegates to truthclf.metrics; reimplements nothing.",
)
@tool_errors
def metrics(
    y_true: list[LabelValue],
    preds: list[Literal[0, 1]],
    probs: list[float] | None = None,
    scheme: Scheme = "primary",
    bootstrap: Annotated[bool, Field(
        description="Default ON. A metric with no interval, null model or "
                    "baseline is not a reportable quantity.")] = True,
    n_boot: Annotated[int, Field(ge=100, le=10000)] = 1000,
    seed: int = 0,
    alpha: Annotated[float, Field(gt=0.0, lt=0.5)] = 0.05,
    compare_to_preds: Annotated[list[Literal[0, 1]] | None, Field(
        description="A second predictor's predictions on the SAME points, for a "
                    "paired comparison (bootstrap on the difference + McNemar).")] = None,
    compare_to_name: str = "other",
    baseline: Literal["majority_class", "none"] = "majority_class",
) -> dict:
    """Score a set of predictions. y_true accepts 0/1 or the six human labels."""
    check_batch(len(y_true), MAX_METRICS_LENGTH, "labels", "metrics")
    yt = coerce_labels(y_true, scheme)
    if len(preds) != len(yt):
        raise ValueError(f"{len(preds)} preds for {len(yt)} labels")
    if probs is not None and len(probs) != len(yt):
        raise ValueError(f"{len(probs)} probs for {len(yt)} labels")
    if not yt:
        raise ValueError("metrics over an empty set is undefined; there is "
                         "nothing to score")

    warns: list[str] = []
    # Ranking and calibration metrics require probabilities. Substituting the
    # binary predictions would make them read as though they had been measured.
    p = probs if probs is not None else None
    if p is None:
        warns.append(
            "probs not supplied: roc_auc, pr_auc, brier and ece are null, not 0. "
            "They are undefined without a probability, and 0.0 would be "
            "indistinguishable from a genuinely terrible score.")
        scoring = preds          # only the threshold metrics are read from this
    else:
        scoring = p

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = M.metric_bundle(yt, preds, scoring)
        if p is None:
            for k in ("roc_auc", "pr_auc", "brier", "ece"):
                bundle[k] = float("nan")
            bins = None
        else:
            bins = M.ece_bin_report(yt, p)
        ci = None
        if bootstrap:
            raw = M.bootstrap_bundle(yt, preds, scoring, n_boot=n_boot,
                                     seed=seed, alpha=alpha)
            ci = {k: list(v) for k, v in raw.items()}
            if p is None:
                for k in ("roc_auc", "pr_auc", "brier", "ece"):
                    ci.pop(k, None)
        # The bootstrap warns when degenerate resamples are dropped. That
        # warning is forwarded to the caller: an interval computed over fewer
        # draws is narrower, and looks indistinguishable from a confident one.
        warns.extend(str(w.message) for w in caught)

    if not bootstrap:
        warns.append(
            "bootstrap=false: the returned metrics carry no interval. Do not "
            "report them as bare numbers.")

    out = {
        "n": len(yt),
        "metrics": {**bundle,
                    "ece_bins": bins or {"strategy": "quantile",
                                         "n_bins_requested": 10,
                                         "n_bins_realised": 0,
                                         "n_bins_occupied": 0,
                                         "count": [], "gap": []},
                    "ci": ci},
        "baseline": None,
        "comparison": None,
        "warnings": warns,
    }

    if baseline == "majority_class":
        # Baseline is this set's own majority-class rate, never a global one.
        # Subsets differ in class balance, so comparing each against a single
        # global figure manufactures differences that are not there.
        arr = np.asarray(yt)
        rate = float(max(arr.mean(), 1.0 - arr.mean()))
        majority = int(arr.mean() >= 0.5)
        const = [majority] * len(yt)
        point, lo, hi = M.paired_accuracy_diff(yt, preds, const, n_boot=n_boot,
                                               seed=seed, alpha=alpha)
        out["baseline"] = {"kind": "majority_class", "rate": rate,
                           "majority_class": majority,
                           "delta_accuracy": [point, lo, hi]}

    if compare_to_preds is not None:
        if len(compare_to_preds) != len(yt):
            raise ValueError(
                f"{len(compare_to_preds)} compare_to_preds for {len(yt)} labels")
        point, lo, hi = M.paired_accuracy_diff(yt, preds, compare_to_preds,
                                               n_boot=n_boot, seed=seed, alpha=alpha)
        out["comparison"] = {
            "name": compare_to_name,
            "delta_accuracy": [point, lo, hi],
            "mcnemar": M.mcnemar(yt, preds, compare_to_preds),
            "note": "delta is accuracy(preds) - accuracy(compare_to_preds), "
                    "paired: one index draw applied to both.",
        }

    return _clean_json(out)


# ---------------------------------------------------------------------------
# 3. retrieval
# ---------------------------------------------------------------------------
@server.tool(
    title="Train-split retrieval",
    description="k nearest TRAIN statements for each query, with their labels. "
                "Train-only is a leakage constraint enforced by construction.",
)
@tool_errors
def retrieval(
    queries: list[str],
    k: Annotated[int, Field(ge=1, le=50)] = 5,
    scheme: Scheme = "primary",
    min_score: Annotated[float | None, Field(
        ge=0.0, le=1.0,
        description="Drop neighbours below this cosine similarity. null returns "
                    "k regardless, with the scores visible.")] = None,
    include_metadata: bool = False,
) -> dict:
    """k-NN over the TRAIN split only. Never returns a val or test row."""
    check_batch(len(queries), MAX_RETRIEVAL_QUERIES, "queries", "retrieval")
    if not queries:
        raise ValueError("queries must be non-empty")
    for i, q in enumerate(queries):
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"queries[{i}] is empty or whitespace-only")

    # Queries must be preprocessed exactly as the index was, or similarity is
    # computed between two different text representations.
    cleaned = [D.clean_text(q) for q in queries]
    vecs = CORPUS.vectorizer.transform(cleaned)
    k_eff = min(k, CORPUS.index_size)
    dist, idx = CORPUS.index.kneighbors(vecs, n_neighbors=k_eff)

    results = []
    for qi, q in enumerate(queries):
        qkey = D.normalized_statement_key(q)
        neighbours = []
        for d, j in zip(dist[qi], idx[qi]):
            score = float(1.0 - d)          # cosine distance -> similarity
            if min_score is not None and score < min_score:
                continue
            r = CORPUS.train[int(j)]
            n = {"row_id": r.row_id, "statement": r.statement,
                 "label": r.label, "y": r.y(scheme), "score": score,
                 # Flags self-retrieval. The split already guarantees no
                 # repeated statement crosses it, so a validation or test row
                 # cannot exactly match a training one; this catches a
                 # caller-supplied query that is itself a training statement.
                 "exact_norm_key_match": r.norm_key == qkey}
            if include_metadata:
                n["metadata"] = {f: getattr(r, f) for f in
                                 ("subjects", "speaker_name", "speaker_job",
                                  "speaker_state", "speaker_affiliation",
                                  "statement_context")}
            neighbours.append(n)
        results.append({"query_id": qi, "query": q, "neighbours": neighbours,
                        "n_returned": len(neighbours)})

    # Verify the train-only guarantee on the actual output rather than
    # assuming the index was built correctly.
    leaked = [n["row_id"] for res in results for n in res["neighbours"]
              if n["row_id"] in CORPUS.non_train_ids]
    if leaked:
        raise LeakageAssertionFailed(
            f"retrieval returned {len(leaked)} row(s) outside the train split "
            f"(first few: {leaked[:5]}). No partial results are returned: "
            "train-only is the constraint this tool exists to honour.")

    return _clean_json({
        "results": results,
        "index_info": {"split": "train", "n_indexed": CORPUS.index_size,
                       "method": CORPUS.index_method, "scheme": scheme,
                       "split_seed": SPLIT_SEED},
        "warnings": ([f"k={k} exceeds the index size; returning {k_eff}"]
                     if k_eff < k else []),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="truthclf data-tools MCP server")
    ap.add_argument("--host", default=os.environ.get("TRUTHCLF_MCP_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("TRUTHCLF_DATA_TOOLS_PORT", 8081)))
    ap.add_argument("--path", default="/mcp")
    args = ap.parse_args()
    print(f"data-tools: {CORPUS.index_size} train rows indexed "
          f"({CORPUS.vocab_size} features); splits "
          f"{len(CORPUS.train)}/{len(CORPUS.val)}/{len(CORPUS.test)}; "
          f"speakers_cross={CORPUS.speakers_cross} "
          f"normkeys_cross={CORPUS.normkeys_cross}", flush=True)
    print(f"data-tools listening on http://{args.host}:{args.port}{args.path}",
          flush=True)
    server.run(transport="streamable-http", host=args.host, port=args.port,
               streamable_http_path=args.path)


if __name__ == "__main__":
    main()
