"""End-to-end example: evaluate the toy system on the sample dataset.

Run from the repo root::

    python examples/run_eval.py

It prints the aggregate scores and writes JSON + HTML reports under ``reports/``.
To include the LLM-as-judge metrics with a real model, swap ``MockJudge`` for a
thin wrapper around your provider (see ``src/arete/judges/base.py``).
"""

from __future__ import annotations

from pathlib import Path

from simple_system import build_system  # noqa: E402  (local example module)

from arete.datasets import load_jsonl
from arete.judges import MockJudge, make_judge_suite
from arete.metrics import DEFAULT_METRICS
from arete.reporting import write_html_report, write_json_report
from arete.runners import evaluate

HERE = Path(__file__).parent


def main() -> None:
    dataset = load_jsonl(HERE / "data" / "sample_dataset.jsonl")
    system = build_system()

    # Deterministic metrics + judge metrics (MockJudge here so it runs offline).
    judge = MockJudge('{"score": 0.9, "rationale": "demo"}')
    metrics = DEFAULT_METRICS + make_judge_suite(judge)

    result = evaluate(system, dataset, metrics=metrics)

    print("Aggregate scores")
    print("-" * 32)
    for name, value in result.aggregate().items():
        print(f"{name:<28} {value:.3f}")

    reports = HERE.parent / "reports"
    json_path = write_json_report(result, reports / "sample.report.json")
    html_path = write_html_report(result, reports / "sample.report.html")
    print(f"\nWrote {json_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
