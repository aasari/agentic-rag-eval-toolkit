from __future__ import annotations

from arete.judges import MockJudge, make_judge_suite
from arete.judges.base import parse_verdict
from arete.judges.scorers import faithfulness


def test_parse_verdict_json():
    value, detail = parse_verdict('Here: {"score": 0.8, "rationale": "ok"}')
    assert value == 0.8
    assert detail["rationale"] == "ok"


def test_parse_verdict_unparseable():
    value, detail = parse_verdict("no json here")
    assert value is None
    assert "raw" in detail


def test_faithfulness_with_mock(example, good_trace):
    judge = MockJudge('{"score": 0.75}')
    score = faithfulness(judge)(good_trace, example)
    assert score.metric == "faithfulness"
    assert score.value == 0.75


def test_judge_suite_clamps_and_handles_bad(example, good_trace):
    judge = MockJudge("garbage")
    for metric in make_judge_suite(judge):
        score = metric(good_trace, example)
        assert score.value == 0.0
        assert score.detail.get("unparsed") is True
