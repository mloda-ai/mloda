"""Planner stubs shared by the runtime tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class ReiterablePlan:
    """A planner stub that yields the same steps on every pass, like a real ExecutionPlan."""

    def __init__(self, *steps: Any) -> None:
        self._steps = steps

    def __iter__(self) -> Iterator[Any]:
        return iter(self._steps)
