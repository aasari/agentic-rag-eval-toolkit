"""Agentic trajectory metrics.

These are the toolkit's differentiator: they score *how* the agent behaved, not
just what it answered. They read the retrieval calls, tool calls, and the
``requires_retrieval`` label on the example.
"""

from __future__ import annotations

from arete.schema import EvalExample, Score, Trace


def retrieval_appropriateness(trace: Trace, example: EvalExample) -> Score:
    """Did the agent retrieve exactly when it should have?

    1.0 when the agent's decision to retrieve (or not) matches the example's
    ``requires_retrieval`` flag; 0.0 otherwise. This rewards both grounding
    when needed and abstaining from pointless retrieval on chit-chat or
    parametric-knowledge questions.
    """
    should = example.requires_retrieval
    did = trace.did_retrieve
    correct = should == did
    return Score(
        "retrieval_appropriateness",
        1.0 if correct else 0.0,
        {"should_retrieve": should, "did_retrieve": did},
    )


def hop_count(trace: Trace, example: EvalExample) -> Score:
    """Number of retrieval hops the agent took.

    Reported as a raw count (``value`` is the count as a float). Unlike the
    other metrics this is diagnostic rather than "higher is better" — the
    reporter surfaces it as a distribution, and aggregation uses the mean.
    """
    return Score("hop_count", float(trace.num_hops), {"hops": trace.num_hops})


def redundant_retrieval_rate(trace: Trace, example: EvalExample) -> Score:
    """Fraction of retrieved chunks that were duplicates of earlier ones.

    A high rate signals an agent re-fetching the same context across hops
    instead of making progress. 0.0 is best.
    """
    seen: set[str] = set()
    total = 0
    duplicates = 0
    for call in trace.retrieval_calls:
        for doc_id in call.doc_ids:
            total += 1
            if doc_id in seen:
                duplicates += 1
            else:
                seen.add(doc_id)
    rate = duplicates / total if total else 0.0
    return Score(
        "redundant_retrieval_rate",
        rate,
        {"duplicates": duplicates, "total_retrieved": total},
    )


def tool_error_rate(trace: Trace, example: EvalExample) -> Score:
    """Fraction of non-retrieval tool calls that errored. 0.0 is best."""
    if not trace.tool_calls:
        return Score("tool_error_rate", 0.0, {"tool_calls": 0})
    errors = sum(1 for t in trace.tool_calls if t.error)
    return Score(
        "tool_error_rate",
        errors / len(trace.tool_calls),
        {"errors": errors, "tool_calls": len(trace.tool_calls)},
    )
