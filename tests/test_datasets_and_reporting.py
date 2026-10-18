from __future__ import annotations

import json

import pytest

from arete.adapters import CallableSystem
from arete.datasets import load_jsonl, parse_example
from arete.reporting import write_html_report, write_json_report
from arete.runners import evaluate
from arete.schema import EvalExample, Trace


def test_parse_example_requires_id_and_query():
    with pytest.raises(ValueError):
        parse_example({"query": "no id"})
    with pytest.raises(ValueError):
        parse_example({"id": "x"})


def test_load_jsonl_roundtrip(tmp_path):
    p = tmp_path / "ds.jsonl"
    rows = [
        {"id": "a", "query": "q1", "relevant_doc_ids": ["d1"]},
        {"id": "b", "query": "q2", "requires_retrieval": False},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    examples = load_jsonl(p)
    assert [e.id for e in examples] == ["a", "b"]
    assert examples[1].requires_retrieval is False


def test_load_jsonl_bad_line_reports_lineno(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "a", "query": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_jsonl(p)


def test_reports_written(tmp_path):
    dataset = [EvalExample(id="a", query="q", reference_answer="x")]
    result = evaluate(CallableSystem(lambda q: Trace(query=q, answer="x")), dataset)

    jp = write_json_report(result, tmp_path / "r.json")
    hp = write_html_report(result, tmp_path / "r.html")

    assert jp.exists() and hp.exists()
    loaded = json.loads(jp.read_text())
    assert loaded["num_examples"] == 1
    html = hp.read_text()
    assert "<table" in html and "Aggregate scores" in html
