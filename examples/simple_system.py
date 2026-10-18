"""A minimal, framework-free agentic RAG system for demonstration.

This is deliberately tiny and uses no external libraries or LLM. It mimics the
*shape* of an agentic RAG system so the evaluation harness has something real
to run against:

* a toy in-memory corpus,
* a keyword retriever,
* an "agent" that decides whether to retrieve, optionally reformulates and
  retrieves a second time, and then composes an answer.

The important part is the end: it returns a fully-populated ``Trace``. Replace
the internals with your real system and keep the Trace contract.
"""

from __future__ import annotations

import re
import time

from arete.adapters import System
from arete.schema import RetrievalCall, RetrievedChunk, Trace

CORPUS: dict[str, str] = {
    "doc-arete-overview": (
        "ARETE is a framework-agnostic toolkit written in Python for evaluating "
        "agentic RAG systems. It scores retrieval quality, answer quality, and "
        "agent trajectory."
    ),
    "doc-metrics": (
        "Evaluation metrics include context precision and recall, answer "
        "correctness, faithfulness, and trajectory metrics."
    ),
    "doc-faithfulness": (
        "Faithfulness measures whether every claim in the answer is grounded in "
        "the retrieved context."
    ),
    "doc-trajectory": (
        "Trajectory metrics judge whether the agent retrieved when it should "
        "have, how many hops it took, and whether any retrieval was redundant."
    ),
    "doc-noise": "The quick brown fox jumps over the lazy dog.",
}

_STOPWORDS = {"the", "a", "an", "is", "of", "to", "in", "does", "what", "which", "how", "it"}


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS}


def _retrieve(query: str, k: int = 2) -> list[RetrievedChunk]:
    q = _keywords(query)
    scored = []
    for doc_id, text in CORPUS.items():
        overlap = len(q & _keywords(text))
        if overlap:
            scored.append((overlap, doc_id, text))
    scored.sort(reverse=True)
    return [
        RetrievedChunk(doc_id=doc_id, text=text, score=float(score))
        for score, doc_id, text in scored[:k]
    ]


def _looks_like_chitchat(query: str) -> bool:
    return bool(re.fullmatch(r"\s*(hi|hello|hey|thanks|thank you)[!.\s]*", query.lower()))


def run(query: str) -> Trace:
    """Run the toy agent and return its trace."""
    start = time.perf_counter()
    trace = Trace(query=query)

    # Step 1: the agent decides whether retrieval is needed.
    if _looks_like_chitchat(query):
        trace.reasoning_steps.append("Query is chit-chat; answering without retrieval.")
        trace.answer = "Hi! How can I help you?"
        trace.llm_calls = 1
        trace.latency_s = time.perf_counter() - start
        return trace

    trace.reasoning_steps.append("Query needs grounding; retrieving context.")
    chunks = _retrieve(query)
    trace.retrieval_calls.append(RetrievalCall(query=query, chunks=chunks, step=1))

    # Step 2: if the first hop looks weak, reformulate and retrieve once more.
    if not chunks or (chunks[0].score or 0) < 2:
        reformulated = f"{query} ARETE toolkit"
        trace.reasoning_steps.append("First hop weak; reformulating query.")
        more = _retrieve(reformulated)
        trace.retrieval_calls.append(RetrievalCall(query=reformulated, chunks=more, step=2))

    # Step 3: compose an answer grounded in the retrieved context.
    context = " ".join(c.text for c in trace.retrieved_chunks())
    trace.answer = context if context else "I could not find relevant information."
    trace.reasoning_steps.append("Composed answer from retrieved context.")
    trace.llm_calls = 1 + trace.num_hops
    trace.latency_s = time.perf_counter() - start
    return trace


def build_system() -> System:
    """Return the toy system wrapped as an ARETE ``System``."""
    from arete.adapters import CallableSystem

    return CallableSystem(run)
