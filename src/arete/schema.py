"""Core data model for ARETE.

Everything in the toolkit reads from the structures defined here. A system
under evaluation reports a :class:`Trace`; the gold labels live on an
:class:`EvalExample`; metrics consume both and emit :class:`Score` objects.

The schema is intentionally built on the standard library only (dataclasses),
so the core package has no third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """A single piece of context returned by a retrieval call."""

    doc_id: str
    text: str = ""
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalCall:
    """One retrieval action taken by the agent.

    Agentic systems may retrieve several times with reformulated queries;
    each such action is captured as its own ``RetrievalCall`` so trajectory
    metrics can reason about hops and redundancy.
    """

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    step: int | None = None

    @property
    def doc_ids(self) -> list[str]:
        return [c.doc_id for c in self.chunks]


@dataclass
class ToolCall:
    """A non-retrieval tool invocation made by the agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    step: int | None = None
    error: str | None = None


@dataclass
class Trace:
    """The full record of how a system produced an answer for one query.

    This is the single contract between a system under evaluation and the
    toolkit. Adapters (see :mod:`arete.adapters`) are responsible for
    populating it.
    """

    query: str
    answer: str = ""
    retrieval_calls: list[RetrievalCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    # Operational signals (optional; populated when the system reports them).
    latency_s: float | None = None
    total_tokens: int | None = None
    llm_calls: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_hops(self) -> int:
        """Number of distinct retrieval actions."""
        return len(self.retrieval_calls)

    def retrieved_chunks(self) -> list[RetrievedChunk]:
        """All chunks retrieved across every hop, in order."""
        chunks: list[RetrievedChunk] = []
        for call in self.retrieval_calls:
            chunks.extend(call.chunks)
        return chunks

    @property
    def retrieved_doc_ids(self) -> list[str]:
        """All retrieved doc ids across every hop, in order, de-duplicated."""
        seen: list[str] = []
        for call in self.retrieval_calls:
            for doc_id in call.doc_ids:
                if doc_id not in seen:
                    seen.append(doc_id)
        return seen

    @property
    def did_retrieve(self) -> bool:
        return self.num_hops > 0


@dataclass
class EvalExample:
    """One row of an evaluation dataset: an input and its gold labels."""

    id: str
    query: str
    reference_answer: str | None = None
    relevant_doc_ids: list[str] = field(default_factory=list)
    # True when a correct agent should NOT need to retrieve (e.g. a greeting
    # or a question answerable from parametric knowledge). Lets trajectory
    # metrics score "appropriate abstention".
    requires_retrieval: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Score:
    """The result of applying one metric to one trace."""

    metric: str
    value: float
    detail: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Score({self.metric}={self.value:.4f})"
