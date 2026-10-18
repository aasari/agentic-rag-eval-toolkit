# Writing an adapter

ARETE imports no agent framework. To evaluate your system, wrap it in something
that satisfies the `System` protocol — a single `run(query) -> Trace` method —
and populate the [`Trace`](../src/arete/schema.py).

## The minimal contract

```python
from arete.schema import Trace, RetrievalCall, RetrievedChunk, ToolCall

def run(query: str) -> Trace:
    trace = Trace(query=query)
    # record each retrieval hop as the agent performs it
    trace.retrieval_calls.append(
        RetrievalCall(
            query=query,
            chunks=[RetrievedChunk(doc_id="doc-1", text="...", score=0.82)],
            step=1,
        )
    )
    # record non-retrieval tool calls, with error set if one failed
    trace.tool_calls.append(ToolCall(name="calculator", arguments={"x": 2}, output=4))
    # final answer + optional operational signals
    trace.answer = "..."
    trace.latency_s = 1.2
    trace.llm_calls = 3
    return trace
```

Wrap a plain function with `CallableSystem`:

```python
from arete.adapters import CallableSystem
system = CallableSystem(run)
```

## What to capture, and why

- **Every retrieval hop as its own `RetrievalCall`.** Trajectory metrics count
  hops and detect redundant retrieval across them, so don't collapse multiple
  retrievals into one call.
- **`doc_id` that matches your gold labels.** This is the join key for all
  retrieval metrics.
- **`requires_retrieval` handling.** If your agent can legitimately skip
  retrieval, let it — and mark such examples `requires_retrieval: false` in the
  dataset so `retrieval_appropriateness` rewards the decision.
- **Tool errors.** Set `ToolCall.error` on failures so `tool_error_rate` is
  meaningful.

## Sketches for popular frameworks

### LangGraph

Run the graph, then read its state/message history to reconstruct the hops:

```python
def run(query: str) -> Trace:
    final_state = graph.invoke({"question": query})
    trace = Trace(query=query, answer=final_state["answer"])
    for i, step in enumerate(final_state["retrievals"], start=1):
        trace.retrieval_calls.append(
            RetrievalCall(
                query=step["query"],
                chunks=[RetrievedChunk(doc_id=d.id, text=d.text, score=d.score)
                        for d in step["docs"]],
                step=i,
            )
        )
    return trace
```

### LlamaIndex

Use a query engine with source nodes:

```python
def run(query: str) -> Trace:
    resp = query_engine.query(query)
    return Trace(
        query=query,
        answer=str(resp),
        retrieval_calls=[RetrievalCall(
            query=query,
            chunks=[RetrievedChunk(doc_id=n.node.node_id, text=n.node.text, score=n.score)
                    for n in resp.source_nodes],
        )],
    )
```

### A deployed HTTP service

Call your endpoint and map the JSON response onto a `Trace`. The toolkit never
needs to know how the service is built.
