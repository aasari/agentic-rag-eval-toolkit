# Defining an evaluation dataset

A dataset is a JSON Lines file: one JSON object per line, each describing an
`EvalExample`.

## Fields

| Field              | Required | Description                                                                 |
|--------------------|----------|-----------------------------------------------------------------------------|
| `id`               | yes      | Unique identifier for the example.                                          |
| `query`            | yes      | The user question passed to the system.                                     |
| `reference_answer` | no       | Gold answer, used by answer-correctness metrics.                            |
| `relevant_doc_ids` | no       | Gold set of relevant document ids, used by retrieval metrics.               |
| `requires_retrieval` | no     | `true` (default) if a correct agent must retrieve; `false` for chit-chat or parametric-knowledge questions. Drives the `retrieval_appropriateness` trajectory metric. |
| `metadata`         | no       | Arbitrary dict for your own bookkeeping (category, difficulty, source).     |

## Example

```json
{"id": "q1", "query": "What does ARETE evaluate?", "reference_answer": "Agentic RAG systems.", "relevant_doc_ids": ["doc-overview"], "requires_retrieval": true}
{"id": "q2", "query": "Hello!", "reference_answer": "Hi there.", "relevant_doc_ids": [], "requires_retrieval": false}
```

The `doc_id` values must match whatever ids your system reports on its
retrieved chunks — that shared id space is how retrieval metrics line gold up
against what was actually retrieved.

## Loading

```python
from arete.datasets import load_jsonl

dataset = load_jsonl("path/to/dataset.jsonl")
```

Blank lines are skipped. A malformed line raises `ValueError` with the file and
line number.
