"""Run results that carry the resolved plan of the run that produced them."""

from collections.abc import Generator
from types import TracebackType
from typing import Any
from uuid import UUID

from mloda.core.api.plan_info import PlanStep


class RunResult(list[Any]):
    """The results list of a ``run_all`` call, plus the resolved plan of the run that produced it.

    Slicing and concatenation return a plain ``list`` without ``plan`` (standard list-subclass behavior).
    The list holds one element per compute step that produced requested output, in the relative order
    those steps appear in ``plan`` (a step producing only injected/intermediate output contributes nothing);
    ``frames()`` pairs each frame with its ``PlanStep`` by step identity instead, regardless of order.
    """

    def __init__(
        self,
        results: list[Any],
        plan: list[PlanStep],
        result_items: list[tuple[UUID, Any]] | None = None,
    ) -> None:
        super().__init__(results)
        self._plan = plan
        self._result_items = tuple(result_items or ())

    @property
    def plan(self) -> list[PlanStep]:
        """Resolved execution plan of the run that produced these results."""
        return self._plan

    def frames(self) -> list[tuple[PlanStep, Any]]:
        """Pair each frame with the compute PlanStep that produced it, by ``step_uuid``, regardless of order."""
        steps = {step.step_uuid: step for step in self._plan}
        return [(steps[step_uuid], frame) for step_uuid, frame in self._result_items]


class ResultStream(Generator[Any, None, None]):
    """A ``stream_all`` result stream: iterates like a generator and exposes the resolved plan."""

    def __init__(self, stream: Generator[tuple[UUID, Any], None, None], plan: list[PlanStep]) -> None:
        self._stream = stream
        self._plan = plan

    @property
    def plan(self) -> list[PlanStep]:
        """Resolved execution plan, available before any element is consumed."""
        return self._plan

    def frames(self) -> Generator[tuple[PlanStep, Any], None, None]:
        """Yield frames paired with the compute PlanStep that produced each, by ``step_uuid``; consumes this stream."""
        steps = {step.step_uuid: step for step in self._plan}
        for step_uuid, frame in self._stream:
            yield steps[step_uuid], frame

    def send(self, value: None, /) -> Any:
        return self._stream.send(value)[1]

    def throw(
        self,
        typ: type[BaseException] | BaseException,
        val: BaseException | object = None,
        tb: TracebackType | None = None,
        /,
    ) -> Any:
        if isinstance(typ, BaseException):
            return self._stream.throw(typ)[1]
        if val is None and tb is None:
            return self._stream.throw(typ)[1]
        return self._stream.throw(typ, val, tb)[1]
