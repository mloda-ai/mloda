"""Run results that know the session that produced them (issue #647 follow-up)."""

from collections.abc import Generator
from types import TracebackType
from typing import TYPE_CHECKING, Any

from mloda.core.api.plan_info import PlanStep

if TYPE_CHECKING:
    from mloda.core.api.request import mlodaAPI


class RunResult(list[Any]):
    """The results list of a ``run_all`` call, plus the resolved plan of the run that produced it."""

    def __init__(self, results: list[Any], session: "mlodaAPI") -> None:
        super().__init__(results)
        self._session = session

    @property
    def plan(self) -> list[PlanStep]:
        """Resolved execution plan of the run that produced these results."""
        return self._session.resolved_plan()


class ResultStream(Generator[Any, None, None]):
    """A ``stream_all`` result stream: iterates like a generator and exposes the resolved plan."""

    def __init__(self, stream: Generator[Any, None, None], session: "mlodaAPI") -> None:
        self._stream = stream
        self._session = session

    @property
    def plan(self) -> list[PlanStep]:
        """Resolved execution plan, available before any element is consumed."""
        return self._session.resolved_plan()

    def send(self, value: None, /) -> Any:
        return self._stream.send(value)

    def throw(
        self,
        typ: type[BaseException] | BaseException,
        val: BaseException | object = None,
        tb: TracebackType | None = None,
        /,
    ) -> Any:
        if isinstance(typ, BaseException):
            return self._stream.throw(typ)
        if val is None and tb is None:
            return self._stream.throw(typ)
        return self._stream.throw(typ, val, tb)
