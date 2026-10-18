# Metrics reference

Every metric is a callable `(trace, example) -> Score`. Unless noted, values are
in `[0, 1]` and higher is better. The runner averages each metric across the
dataset, skipping scores flagged `detail["skipped"]`.

## Retrieval (deterministic, label-based)

Defined in `arete/metrics/retrieval.py`. All compare `trace.retrieved_doc_ids`
(de-duplicated across hops, in order) against `example.relevant_doc_ids`.

| Metric              | Meaning                                                             |
|---------------------|---------------------------------------------------------------------|
| `context_precision` | Fraction of retrieved docs that are relevant.                       |
| `context_recall`    | Fraction of relevant docs that were retrieved.                      |
| `hit_rate`          | 1.0 if at least one relevant doc was retrieved.                     |
| `mrr`               | Reciprocal rank of the first relevant doc.                          |
| `ndcg`              | Normalized DCG over retrieval order (binary relevance).             |

When an example has no relevant docs, these return 1.0 (nothing to find, nothing
wrongly retrieved is the correct behavior when the agent also retrieves nothing).

## Answer quality

Defined in `arete/metrics/answer.py`. Compare `trace.answer` to
`example.reference_answer` using SQuAD-style normalization. Both are **skipped**
when the example has no reference answer.

| Metric               | Meaning                                              |
|----------------------|------------------------------------------------------|
| `answer_exact_match` | 1.0 if normalized answer equals normalized reference.|
| `answer_token_f1`    | Token-level F1 between answer and reference.         |

## Trajectory (agentic behavior)

Defined in `arete/metrics/trajectory.py`.

| Metric                     | Meaning                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| `retrieval_appropriateness`| 1.0 when the agent's decision to retrieve matches `requires_retrieval`. |
| `hop_count`                | Raw number of retrieval hops (diagnostic; mean is reported).            |
| `redundant_retrieval_rate` | Fraction of retrieved chunks that duplicated an earlier one. Lower is better. |
| `tool_error_rate`          | Fraction of non-retrieval tool calls that errored. Lower is better.     |

## LLM-as-judge

Defined in `arete/judges/scorers.py`. Each is a factory taking an `LLMJudge` and
returning a metric. The judge is prompted for strict JSON
(`{"score": float, "rationale": str}`); an unparseable verdict yields `0.0` with
`detail["unparsed"] = True`.

| Metric              | Meaning                                                         |
|---------------------|-----------------------------------------------------------------|
| `faithfulness`      | Are the answer's claims grounded in the retrieved context?      |
| `answer_relevance`  | Does the answer address the question (regardless of truth)?     |
| `context_relevance` | How much of the retrieved context is relevant to the question?  |

Use `make_judge_suite(judge)` to get all three bound to one judge.

## Writing your own

Return a `Score`:

```python
from arete.schema import Score

def my_metric(trace, example) -> Score:
    return Score("my_metric", value=0.0, detail={"note": "..."})
my_metric.__name__ = "my_metric"
```

Pass it to `evaluate(system, dataset, metrics=[...])` or add it to
`DEFAULT_METRICS`.
