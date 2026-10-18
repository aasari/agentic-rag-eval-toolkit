from __future__ import annotations

from arete.metrics.trajectory import (
    hop_count,
    redundant_retrieval_rate,
    retrieval_appropriateness,
    tool_error_rate,
)
from arete.schema import EvalExample, RetrievalCall, RetrievedChunk, ToolCall, Trace


def test_appropriateness_retrieved_when_needed(example, good_trace):
    assert retrieval_appropriateness(good_trace, example).value == 1.0


def test_appropriateness_retrieved_when_not_needed():
    ex = EvalExample(id="x", query="hi", requires_retrieval=False)
    trace = Trace(
        query="hi",
        retrieval_calls=[RetrievalCall(query="hi", chunks=[RetrievedChunk(doc_id="d")])],
    )
    assert retrieval_appropriateness(trace, ex).value == 0.0


def test_appropriateness_abstained_correctly():
    ex = EvalExample(id="x", query="hi", requires_retrieval=False)
    trace = Trace(query="hi", answer="hello")
    assert retrieval_appropriateness(trace, ex).value == 1.0


def test_hop_count(example):
    trace = Trace(
        query="q",
        retrieval_calls=[RetrievalCall(query="q"), RetrievalCall(query="q2")],
    )
    assert hop_count(trace, example).value == 2.0


def test_redundant_retrieval(example):
    trace = Trace(
        query="q",
        retrieval_calls=[
            RetrievalCall(query="q", chunks=[RetrievedChunk(doc_id="d1")]),
            RetrievalCall(query="q2", chunks=[RetrievedChunk(doc_id="d1")]),
        ],
    )
    score = redundant_retrieval_rate(trace, example)
    assert score.value == 0.5


def test_tool_error_rate(example):
    trace = Trace(
        query="q",
        tool_calls=[
            ToolCall(name="calc", error="boom"),
            ToolCall(name="calc", output=42),
        ],
    )
    assert tool_error_rate(trace, example).value == 0.5
    assert tool_error_rate(Trace(query="q"), example).value == 0.0
