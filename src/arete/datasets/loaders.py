"""Load evaluation datasets from disk.

The on-disk format is JSON Lines: one JSON object per line, each describing an
:class:`~arete.schema.EvalExample`. Minimal required fields are ``id`` and
``query``; everything else is optional.

Example line::

    {"id": "q1", "query": "What is ARETE?", "reference_answer": "...",
     "relevant_doc_ids": ["doc-3"], "requires_retrieval": true}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arete.schema import EvalExample


def parse_example(obj: dict[str, Any]) -> EvalExample:
    """Build an :class:`EvalExample` from a plain dict, validating required keys."""
    if "id" not in obj:
        raise ValueError("dataset row is missing required field 'id'")
    if "query" not in obj:
        raise ValueError(f"dataset row {obj['id']!r} is missing required field 'query'")

    return EvalExample(
        id=str(obj["id"]),
        query=str(obj["query"]),
        reference_answer=obj.get("reference_answer"),
        relevant_doc_ids=list(obj.get("relevant_doc_ids", [])),
        requires_retrieval=bool(obj.get("requires_retrieval", True)),
        metadata=dict(obj.get("metadata", {})),
    )


def load_jsonl(path: str | Path) -> list[EvalExample]:
    """Load a JSONL dataset file into a list of :class:`EvalExample`.

    Blank lines are skipped. Raises ``ValueError`` with the line number if a
    line is not valid JSON or is missing a required field.
    """
    path = Path(path)
    examples: list[EvalExample] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                examples.append(parse_example(obj))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return examples
