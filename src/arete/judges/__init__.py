"""LLM-as-judge scorers.

The toolkit never hard-codes an LLM provider. Judge-based metrics depend only
on the :class:`LLMJudge` protocol — a single ``score`` method that takes a
prompt and returns text. Plug in OpenAI, Anthropic, a local model, or the
bundled :class:`MockJudge` for tests.
"""

from __future__ import annotations

from arete.judges.base import LLMJudge, MockJudge
from arete.judges.scorers import (
    answer_relevance,
    context_relevance,
    faithfulness,
    make_judge_suite,
)

__all__ = [
    "LLMJudge",
    "MockJudge",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "make_judge_suite",
]
