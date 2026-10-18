from __future__ import annotations

from arete.metrics.retrieval import (
    context_precision,
    context_recall,
    hit_rate,
    mrr,
    ndcg,
)
from arete.schema import EvalExample, RetrievalCall, RetrievedChunk, Trace


def _trace(doc_ids: list[str]) -> Trace:
    return Trace(
        query="q",
        retrieval_calls=[
            RetrievalCall(query="q", chunks=[RetrievedChunk(doc_id=d) for d in doc_ids])
        ],
    )


def test_perfect_retrieval(example, good_trace):
    assert context_precision(good_trace, example).value == 1.0
    assert context_recall(good_trace, example).value == 1.0
    assert hit_rate(good_trace, example).value == 1.0
    assert mrr(good_trace, example).value == 1.0
    assert ndcg(good_trace, example).value == 1.0


def test_precision_with_noise(example):
    trace = _trace(["doc-1", "doc-noise", "doc-noise2"])
    assert context_precision(trace, example).value == 1 / 3
    assert context_recall(trace, example).value == 0.5
    assert hit_rate(trace, example).value == 1.0


def test_mrr_rank_two(example):
    trace = _trace(["doc-noise", "doc-1"])
    assert mrr(trace, example).value == 0.5


def test_no_retrieval_no_relevant():
    ex = EvalExample(id="x", query="hi", relevant_doc_ids=[], requires_retrieval=False)
    trace = Trace(query="hi")
    assert context_precision(trace, ex).value == 1.0
    assert context_recall(trace, ex).value == 1.0
    assert hit_rate(trace, ex).value == 1.0


def test_miss():
    ex = EvalExample(id="x", query="q", relevant_doc_ids=["doc-1"])
    trace = _trace(["doc-9"])
    assert hit_rate(trace, ex).value == 0.0
    assert mrr(trace, ex).value == 0.0
    assert ndcg(trace, ex).value == 0.0


def test_dedup_across_hops():
    ex = EvalExample(id="x", query="q", relevant_doc_ids=["doc-1"])
    trace = Trace(
        query="q",
        retrieval_calls=[
            RetrievalCall(query="q", chunks=[RetrievedChunk(doc_id="doc-1")]),
            RetrievalCall(query="q2", chunks=[RetrievedChunk(doc_id="doc-1")]),
        ],
    )
    # de-duplicated retrieval order has one entry
    assert trace.retrieved_doc_ids == ["doc-1"]
    assert context_precision(trace, ex).value == 1.0
