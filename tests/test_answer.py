from __future__ import annotations

from arete.metrics.answer import answer_exact_match, answer_token_f1, normalize_text
from arete.schema import EvalExample, Trace


def test_normalize():
    assert normalize_text("The, Quick!  Brown.") == "quick brown"


def test_exact_match_ignores_formatting():
    ex = EvalExample(id="x", query="q", reference_answer="The answer is Python.")
    trace = Trace(query="q", answer="answer is python")
    assert answer_exact_match(trace, ex).value == 1.0


def test_exact_match_negative():
    ex = EvalExample(id="x", query="q", reference_answer="Python")
    trace = Trace(query="q", answer="Java")
    assert answer_exact_match(trace, ex).value == 0.0


def test_token_f1_partial():
    ex = EvalExample(id="x", query="q", reference_answer="the cat sat on the mat")
    trace = Trace(query="q", answer="cat sat mat")
    score = answer_token_f1(trace, ex)
    assert 0.0 < score.value < 1.0
    assert score.detail["precision"] == 1.0  # every predicted token is in gold


def test_skipped_without_reference():
    ex = EvalExample(id="x", query="q", reference_answer=None)
    trace = Trace(query="q", answer="anything")
    assert answer_exact_match(trace, ex).detail["skipped"] is True
    assert answer_token_f1(trace, ex).detail["skipped"] is True
