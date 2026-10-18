"""Judge-based metrics: faithfulness, answer relevance, context relevance.

Each factory takes an :class:`~arete.judges.base.LLMJudge` and returns a metric
callable ``(trace, example) -> Score`` compatible with the runner, so judge
metrics mix freely with the deterministic ones.

The prompts ask the judge to return a strict JSON object::

    {"score": <0..1>, "rationale": "<why>"}

which :func:`~arete.judges.base.parse_verdict` reads back. A verdict that can't
be parsed yields ``value=0.0`` with ``detail={"unparsed": True}``.
"""

from __future__ import annotations

from typing import Callable

from arete.judges.base import LLMJudge, parse_verdict
from arete.schema import EvalExample, Score, Trace

Metric = Callable[[Trace, EvalExample], Score]

_FAITHFULNESS_PROMPT = """You are grading a RAG answer for FAITHFULNESS.
An answer is faithful if every factual claim it makes is supported by the
retrieved context. Penalize hallucinations and unsupported claims.

Question:
{query}

Retrieved context:
{context}

Answer:
{answer}

Return ONLY a JSON object: {{"score": <float 0..1>, "rationale": "<short>"}}
where 1.0 means fully grounded and 0.0 means unsupported."""

_ANSWER_RELEVANCE_PROMPT = """You are grading a RAG answer for RELEVANCE to the
question. Ignore whether it is factually correct; judge only whether it
addresses what was asked.

Question:
{query}

Answer:
{answer}

Return ONLY a JSON object: {{"score": <float 0..1>, "rationale": "<short>"}}."""

_CONTEXT_RELEVANCE_PROMPT = """You are grading RETRIEVAL for CONTEXT RELEVANCE.
Judge how much of the retrieved context is relevant to answering the question.

Question:
{query}

Retrieved context:
{context}

Return ONLY a JSON object: {{"score": <float 0..1>, "rationale": "<short>"}}
where 1.0 means all context is on-topic and 0.0 means none is."""


def _context_text(trace: Trace) -> str:
    chunks = trace.retrieved_chunks()
    if not chunks:
        return "(no context retrieved)"
    return "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks)


def _run_judge(judge: LLMJudge, prompt: str, metric_name: str) -> Score:
    raw = judge.score(prompt)
    value, detail = parse_verdict(raw)
    if value is None:
        return Score(metric_name, 0.0, {"unparsed": True, **detail})
    value = max(0.0, min(1.0, value))
    return Score(metric_name, value, detail)


def faithfulness(judge: LLMJudge) -> Metric:
    """Build a faithfulness metric backed by ``judge``."""

    def metric(trace: Trace, example: EvalExample) -> Score:
        prompt = _FAITHFULNESS_PROMPT.format(
            query=example.query,
            context=_context_text(trace),
            answer=trace.answer,
        )
        return _run_judge(judge, prompt, "faithfulness")

    metric.__name__ = "faithfulness"
    return metric


def answer_relevance(judge: LLMJudge) -> Metric:
    """Build an answer-relevance metric backed by ``judge``."""

    def metric(trace: Trace, example: EvalExample) -> Score:
        prompt = _ANSWER_RELEVANCE_PROMPT.format(query=example.query, answer=trace.answer)
        return _run_judge(judge, prompt, "answer_relevance")

    metric.__name__ = "answer_relevance"
    return metric


def context_relevance(judge: LLMJudge) -> Metric:
    """Build a context-relevance metric backed by ``judge``."""

    def metric(trace: Trace, example: EvalExample) -> Score:
        prompt = _CONTEXT_RELEVANCE_PROMPT.format(
            query=example.query, context=_context_text(trace)
        )
        return _run_judge(judge, prompt, "context_relevance")

    metric.__name__ = "context_relevance"
    return metric


def make_judge_suite(judge: LLMJudge) -> list[Metric]:
    """Return all three judge-based metrics bound to ``judge``."""
    return [faithfulness(judge), answer_relevance(judge), context_relevance(judge)]
