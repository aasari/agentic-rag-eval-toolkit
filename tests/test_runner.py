from __future__ import annotations

from arete.adapters import CallableSystem
from arete.metrics import DEFAULT_METRICS
from arete.runners import evaluate
from arete.schema import EvalExample, RetrievalCall, RetrievedChunk, Score, Trace


def _system_fn(query: str) -> Trace:
    return Trace(
        query=query,
        answer="Python",
        retrieval_calls=[
            RetrievalCall(query=query, chunks=[RetrievedChunk(doc_id="doc-1")])
        ],
    )


def test_evaluate_end_to_end():
    dataset = [
        EvalExample(id="a", query="q1", reference_answer="Python", relevant_doc_ids=["doc-1"]),
        EvalExample(id="b", query="q2", reference_answer="Python", relevant_doc_ids=["doc-9"]),
    ]
    result = evaluate(CallableSystem(_system_fn), dataset, metrics=DEFAULT_METRICS)
    assert len(result.per_query) == 2
    agg = result.aggregate()
    assert agg["hit_rate"] == 0.5
    assert agg["answer_exact_match"] == 1.0
    assert "context_precision" in agg


def test_failing_metric_is_contained():
    def boom(trace, example) -> Score:
        raise RuntimeError("kaboom")

    boom.__name__ = "boom"
    dataset = [EvalExample(id="a", query="q", reference_answer="x")]
    result = evaluate(CallableSystem(_system_fn), dataset, metrics=[boom])
    scores = result.per_query[0].scores
    assert scores[0].metric == "boom"
    assert scores[0].detail["error"] == "kaboom"


def test_result_serializable():
    dataset = [
        EvalExample(id="a", query="q", reference_answer="Python", relevant_doc_ids=["doc-1"])
    ]
    result = evaluate(CallableSystem(_system_fn), dataset)
    d = result.to_dict()
    assert d["num_examples"] == 1
    assert d["per_query"][0]["id"] == "a"
