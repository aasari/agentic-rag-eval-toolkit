"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from arete.schema import EvalExample, RetrievalCall, RetrievedChunk, Trace


@pytest.fixture
def example() -> EvalExample:
    return EvalExample(
        id="t1",
        query="What is ARETE?",
        reference_answer="ARETE is a toolkit for evaluating agentic RAG systems.",
        relevant_doc_ids=["doc-1", "doc-2"],
        requires_retrieval=True,
    )


@pytest.fixture
def good_trace() -> Trace:
    return Trace(
        query="What is ARETE?",
        answer="ARETE is a toolkit for evaluating agentic RAG systems.",
        retrieval_calls=[
            RetrievalCall(
                query="What is ARETE?",
                chunks=[
                    RetrievedChunk(doc_id="doc-1", text="ARETE overview", score=3.0),
                    RetrievedChunk(doc_id="doc-2", text="metrics", score=2.0),
                ],
                step=1,
            )
        ],
    )
