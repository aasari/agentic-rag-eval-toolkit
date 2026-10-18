"""Deterministic, label-based retrieval metrics.

These compare the doc ids the system retrieved against the gold
``relevant_doc_ids`` on the example. They need no LLM and are fully
reproducible, which makes them the easiest to trust and to unit-test.

Ranking metrics (MRR, nDCG) operate on the de-duplicated retrieval order
across all hops, so an agent that surfaces a relevant document earlier scores
better than one that surfaces it on a later hop.
"""

from __future__ import annotations

import math

from arete.schema import EvalExample, Score, Trace


def context_precision(trace: Trace, example: EvalExample) -> Score:
    """Fraction of retrieved docs that are relevant."""
    retrieved = trace.retrieved_doc_ids
    relevant = set(example.relevant_doc_ids)
    if not retrieved:
        value = 1.0 if not relevant else 0.0
        return Score("context_precision", value, {"retrieved": 0})
    hits = sum(1 for d in retrieved if d in relevant)
    return Score(
        "context_precision",
        hits / len(retrieved),
        {"hits": hits, "retrieved": len(retrieved)},
    )


def context_recall(trace: Trace, example: EvalExample) -> Score:
    """Fraction of relevant docs that were retrieved."""
    relevant = set(example.relevant_doc_ids)
    if not relevant:
        return Score("context_recall", 1.0, {"relevant": 0})
    retrieved = set(trace.retrieved_doc_ids)
    hits = len(relevant & retrieved)
    return Score(
        "context_recall",
        hits / len(relevant),
        {"hits": hits, "relevant": len(relevant)},
    )


def hit_rate(trace: Trace, example: EvalExample) -> Score:
    """1.0 if at least one relevant doc was retrieved, else 0.0."""
    relevant = set(example.relevant_doc_ids)
    if not relevant:
        return Score("hit_rate", 1.0, {"relevant": 0})
    hit = any(d in relevant for d in trace.retrieved_doc_ids)
    return Score("hit_rate", 1.0 if hit else 0.0, {})


def mrr(trace: Trace, example: EvalExample) -> Score:
    """Mean reciprocal rank of the first relevant doc in retrieval order."""
    relevant = set(example.relevant_doc_ids)
    if not relevant:
        return Score("mrr", 1.0, {"relevant": 0})
    for rank, doc_id in enumerate(trace.retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return Score("mrr", 1.0 / rank, {"first_relevant_rank": rank})
    return Score("mrr", 0.0, {"first_relevant_rank": None})


def ndcg(trace: Trace, example: EvalExample, k: int | None = None) -> Score:
    """Normalized discounted cumulative gain over retrieval order.

    Binary relevance: a retrieved doc has gain 1 if relevant, else 0.
    """
    relevant = set(example.relevant_doc_ids)
    ranked = trace.retrieved_doc_ids
    if k is not None:
        ranked = ranked[:k]
    if not relevant:
        return Score("ndcg", 1.0, {"relevant": 0})

    dcg = sum(
        (1.0 if doc_id in relevant else 0.0) / math.log2(i + 2)
        for i, doc_id in enumerate(ranked)
    )
    ideal_hits = min(len(relevant), len(ranked)) if ranked else len(relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    value = dcg / idcg if idcg > 0 else 0.0
    return Score("ndcg", value, {"dcg": dcg, "idcg": idcg})
