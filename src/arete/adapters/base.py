"""The adapter contract between a system under evaluation and the toolkit."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from arete.schema import Trace


@runtime_checkable
class System(Protocol):
    """A system under evaluation.

    Implement this however you like — wrap a LangGraph graph, a LlamaIndex
    query engine, an HTTP call to a deployed service, or a plain function.
    The only requirement is that :meth:`run` returns a populated
    :class:`~arete.schema.Trace` for the given query.
    """

    def run(self, query: str) -> Trace:  # pragma: no cover - protocol
        ...


class CallableSystem:
    """Adapt a plain ``query -> Trace`` function into a :class:`System`.

    Handy for quick experiments and tests::

        system = CallableSystem(lambda q: Trace(query=q, answer="..."))
    """

    def __init__(self, fn: Callable[[str], Trace]):
        self._fn = fn

    def run(self, query: str) -> Trace:
        return self._fn(query)
