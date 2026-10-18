"""Answer-vs-reference correctness metrics.

Deterministic string-based metrics that compare the system's answer against a
gold ``reference_answer``. For semantic correctness beyond surface overlap, use
the LLM-as-judge metrics in :mod:`arete.judges`.

If an example has no reference answer, these metrics return a NaN-free neutral
signal is avoided — instead they are simply skipped by the runner. Here they
return ``value=0.0`` with ``detail={"skipped": True}`` so callers can filter.
"""

from __future__ import annotations

import re
import string

from arete.schema import EvalExample, Score, Trace

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace.

    Mirrors the common SQuAD-style normalization so exact match and token F1
    are not thrown off by trivial formatting differences.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in _WS.split(text) if t and t not in {"a", "an", "the"}]
    return " ".join(tokens)


def _tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def answer_exact_match(trace: Trace, example: EvalExample) -> Score:
    """1.0 if the normalized answer equals the normalized reference."""
    if example.reference_answer is None:
        return Score("answer_exact_match", 0.0, {"skipped": True})
    match = normalize_text(trace.answer) == normalize_text(example.reference_answer)
    return Score("answer_exact_match", 1.0 if match else 0.0, {})


def answer_token_f1(trace: Trace, example: EvalExample) -> Score:
    """Token-level F1 between answer and reference (SQuAD-style)."""
    if example.reference_answer is None:
        return Score("answer_token_f1", 0.0, {"skipped": True})

    pred = _tokens(trace.answer)
    gold = _tokens(example.reference_answer)
    if not pred and not gold:
        return Score("answer_token_f1", 1.0, {})
    if not pred or not gold:
        return Score("answer_token_f1", 0.0, {})

    # Multiset intersection.
    common = 0
    gold_remaining = list(gold)
    for tok in pred:
        if tok in gold_remaining:
            gold_remaining.remove(tok)
            common += 1
    if common == 0:
        return Score("answer_token_f1", 0.0, {})

    precision = common / len(pred)
    recall = common / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return Score("answer_token_f1", f1, {"precision": precision, "recall": recall})
