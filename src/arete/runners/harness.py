"""Run a system over a dataset and collect scores.

The harness is deliberately small: for each example it asks the system to
produce a trace, applies every metric, and records the scores. Aggregation is
computed on demand. A metric that raises is caught and recorded as a failed
score rather than aborting the whole run.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable

from arete.adapters import System
from arete.metrics import DEFAULT_METRICS
from arete.schema import EvalExample, Score, Trace

Metric = Callable[[Trace, EvalExample], Score]


@dataclass
class QueryResult:
    """Scores (and the trace) for a single example."""

    example: EvalExample
    trace: Trace
    scores: list[Score] = field(default_factory=list)

    def score_map(self) -> dict[str, float]:
        return {s.metric: s.value for s in self.scores}


@dataclass
class EvalResult:
    """The outcome of evaluating a system over a dataset."""

    per_query: list[QueryResult] = field(default_factory=list)

    def metric_names(self) -> list[str]:
        names: list[str] = []
        for qr in self.per_query:
            for s in qr.scores:
                if s.metric not in names:
                    names.append(s.metric)
        return names

    def aggregate(self) -> dict[str, float]:
        """Mean of each metric across all examples (skipped scores excluded)."""
        agg: dict[str, float] = {}
        for name in self.metric_names():
            values = [
                s.value
                for qr in self.per_query
                for s in qr.scores
                if s.metric == name and not s.detail.get("skipped")
            ]
            if values:
                agg[name] = statistics.fmean(values)
        return agg

    def to_dict(self) -> dict:
        """JSON-serializable summary of the run."""
        return {
            "aggregate": self.aggregate(),
            "num_examples": len(self.per_query),
            "per_query": [
                {
                    "id": qr.example.id,
                    "query": qr.example.query,
                    "answer": qr.trace.answer,
                    "num_hops": qr.trace.num_hops,
                    "retrieved_doc_ids": qr.trace.retrieved_doc_ids,
                    "scores": {
                        s.metric: {"value": s.value, "detail": s.detail}
                        for s in qr.scores
                    },
                }
                for qr in self.per_query
            ],
        }


def evaluate(
    system: System,
    dataset: Iterable[EvalExample],
    metrics: list[Metric] | None = None,
) -> EvalResult:
    """Evaluate ``system`` over ``dataset`` using ``metrics``.

    Args:
        system: anything satisfying the :class:`~arete.adapters.System` protocol.
        dataset: iterable of :class:`~arete.schema.EvalExample`.
        metrics: metric callables; defaults to
            :data:`arete.metrics.DEFAULT_METRICS`.

    Returns:
        An :class:`EvalResult` with per-query and aggregate scores.
    """
    metrics = list(DEFAULT_METRICS if metrics is None else metrics)
    result = EvalResult()

    for example in dataset:
        trace = system.run(example.query)
        scores: list[Score] = []
        for metric in metrics:
            name = getattr(metric, "__name__", repr(metric))
            try:
                scores.append(metric(trace, example))
            except Exception as exc:  # noqa: BLE001 - one bad metric shouldn't abort
                scores.append(Score(name, 0.0, {"error": str(exc)}))
        result.per_query.append(QueryResult(example=example, trace=trace, scores=scores))

    return result
