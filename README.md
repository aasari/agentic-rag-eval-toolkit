# ARETE — Agentic RAG Evaluation Toolkit

A **framework-agnostic**, dependency-light toolkit for evaluating **agentic RAG**
systems. Unlike plain RAG, an agentic system decides *when* to retrieve, may
reformulate its query, retrieve over multiple hops, and call tools before
answering — so ARETE scores both the **final answer** and the **trajectory**
that produced it.

The core package has **no third-party dependencies**: every metric is
implemented from scratch on the standard library. You plug in your own LLM only
for the optional judge-based metrics.

## Why agentic RAG needs its own evaluation

Answer-only scoring misses most of what makes an agent good or bad. ARETE covers
four families:

- **Retrieval quality** — context precision/recall, hit rate, MRR, nDCG against a
  gold set.
- **Answer quality** — exact match and token-F1 vs a reference, plus LLM-judged
  faithfulness and relevance.
- **Agent trajectory** — did it retrieve *when it should have*, how many hops,
  redundant retrieval, tool-call errors.
- **Operational** — latency, tokens, and LLM-call counts captured on the trace.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.9+.

## Quickstart

Any system under evaluation just needs to return a `Trace`:

```python
from arete.adapters import CallableSystem
from arete.schema import Trace, RetrievalCall, RetrievedChunk
from arete.datasets import load_jsonl
from arete.runners import evaluate
from arete.reporting import write_html_report

def my_system(query: str) -> Trace:
    # ... run your real agentic RAG pipeline ...
    return Trace(
        query=query,
        answer="...",
        retrieval_calls=[RetrievalCall(query=query, chunks=[RetrievedChunk(doc_id="doc-1")])],
    )

dataset = load_jsonl("examples/data/sample_dataset.jsonl")
result = evaluate(CallableSystem(my_system), dataset)

print(result.aggregate())
write_html_report(result, "reports/run.html")
```

Run the bundled end-to-end example (toy system, no LLM required):

```bash
python examples/run_eval.py
```

## The trace contract

Because ARETE imports no agent framework, the single contract between your
system and the toolkit is the [`Trace`](src/arete/schema.py). Your adapter
populates it; every metric reads from it. See
[docs/adapters.md](docs/adapters.md) for how to wrap LangGraph, LlamaIndex, or a
custom agent.

## LLM-as-judge metrics

Judge metrics depend only on a tiny `LLMJudge` protocol — one `score(prompt)`
method — so you can back them with any provider:

```python
from arete.judges import make_judge_suite

class OpenAIJudge:
    def __init__(self, client, model="gpt-4o-mini"):
        self.client, self.model = client, model
    def score(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

metrics = make_judge_suite(OpenAIJudge(client))
```

A `MockJudge` is provided for offline runs and tests.

## Documentation

- [Defining a dataset](docs/datasets.md)
- [Writing an adapter](docs/adapters.md)
- [Metrics reference](docs/metrics.md)

## Development

```bash
pip install -e ".[dev]"
pytest          # run the test suite
ruff check .    # lint
```

## License

Apache-2.0. See [LICENSE](LICENSE).
