"""ARETE — Agentic RAG Evaluation Toolkit.

A framework-agnostic, dependency-light toolkit for evaluating agentic RAG
systems across retrieval quality, answer quality, and agent trajectory.
"""

from __future__ import annotations

from arete.schema import (
    EvalExample,
    RetrievalCall,
    RetrievedChunk,
    Score,
    ToolCall,
    Trace,
)

__version__ = "0.1.0"

__all__ = [
    "EvalExample",
    "RetrievalCall",
    "RetrievedChunk",
    "Score",
    "ToolCall",
    "Trace",
    "__version__",
]
