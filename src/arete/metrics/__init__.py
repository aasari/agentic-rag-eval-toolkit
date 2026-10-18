"""Metrics for ARETE.

Every metric is a callable with the signature::

    metric(trace: Trace, example: EvalExample) -> Score

Metrics are grouped into three families:

* :mod:`arete.metrics.retrieval` — deterministic, label-based retrieval metrics.
* :mod:`arete.metrics.answer` — answer-vs-reference correctness metrics.
* :mod:`arete.metrics.trajectory` — agentic behavior metrics.

LLM-as-judge metrics live in :mod:`arete.judges`.
"""

from __future__ import annotations

from arete.metrics.answer import answer_exact_match, answer_token_f1
from arete.metrics.retrieval import (
    context_precision,
    context_recall,
    hit_rate,
    mrr,
    ndcg,
)
from arete.metrics.trajectory import (
    hop_count,
    redundant_retrieval_rate,
    retrieval_appropriateness,
    tool_error_rate,
)

#: Default metric suite used by the runner when none is supplied.
DEFAULT_METRICS = [
    context_precision,
    context_recall,
    hit_rate,
    mrr,
    ndcg,
    answer_exact_match,
    answer_token_f1,
    retrieval_appropriateness,
    hop_count,
    redundant_retrieval_rate,
    tool_error_rate,
]

__all__ = [
    "context_precision",
    "context_recall",
    "hit_rate",
    "mrr",
    "ndcg",
    "answer_exact_match",
    "answer_token_f1",
    "retrieval_appropriateness",
    "hop_count",
    "redundant_retrieval_rate",
    "tool_error_rate",
    "DEFAULT_METRICS",
]
