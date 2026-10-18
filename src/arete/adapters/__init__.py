"""Adapter interface.

ARETE never imports an agent framework. Instead, any system under evaluation
is wrapped in an object that satisfies the :class:`System` protocol: given a
query, return a fully-populated :class:`~arete.schema.Trace`.

Write a thin adapter for LangGraph, LlamaIndex, a custom agent, or anything
else, and the rest of the toolkit works unchanged.
"""

from __future__ import annotations

from arete.adapters.base import CallableSystem, System

__all__ = ["System", "CallableSystem"]
