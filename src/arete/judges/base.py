"""The ``LLMJudge`` protocol and helpers for parsing structured verdicts."""

from __future__ import annotations

import json
import re
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class LLMJudge(Protocol):
    """Minimal interface a judge model must satisfy.

    One method: take a prompt, return the model's text completion. This keeps
    the toolkit free of any provider SDK. A thin wrapper around your client of
    choice is all that is required::

        class OpenAIJudge:
            def __init__(self, client, model="gpt-4o-mini"):
                self.client, self.model = client, model
            def score(self, prompt: str) -> str:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content
    """

    def score(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class MockJudge:
    """A deterministic judge for tests and offline runs.

    Either return a fixed response, or compute one from the prompt via a
    callable. The default parser (:func:`parse_verdict`) understands the JSON
    shape the bundled scorer prompts request.
    """

    def __init__(self, response: str | Callable[[str], str] = '{"score": 1.0}'):
        self._response = response

    def score(self, prompt: str) -> str:
        if callable(self._response):
            return self._response(prompt)
        return self._response


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str) -> tuple[float | None, dict]:
    """Extract a ``score`` float (and any rationale) from a judge response.

    Looks for a JSON object anywhere in the text with a numeric ``score`` key.
    Returns ``(None, {...})`` when nothing parseable is found, so callers can
    decide how to handle an unscoreable verdict rather than crashing a run.
    """
    match = _JSON_RE.search(text or "")
    if not match:
        return None, {"raw": text}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, {"raw": text}
    score = obj.get("score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    return score, obj
