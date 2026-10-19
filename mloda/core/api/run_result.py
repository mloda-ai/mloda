"""Run results that carry the resolved plan of the run that produced them."""

from collections.abc import Generator
from types import TracebackType
from typing import TYPE_CHECKING, Any
from uuid import UUID

from mloda.core.abstract_plugins.components.hashable_dict import _deep_equal
from mloda.core.api.plan_info import PlanStep

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.feature import Feature


class RunResult(list[Any]):
    """The results list of a ``run_all`` call, plus the resolved plan of the run that produced it.

    Slicing and concatenation return a plain ``list`` without ``plan`` (standard list-subclass behavior).
    The list is in plan order, one element per step that produced requested output; ``frames()``
    pairs each frame with its ``PlanStep`` by step identity, regardless of order.
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

    def frame_for(self, feature: "Feature | str") -> Any:
        """Return the frame whose step requested ``feature`` (a Feature also matches group options); else ValueError."""
        name = feature if isinstance(feature, str) else str(feature.name)
        group = None if isinstance(feature, str) else feature.options.group
        pairs = [(step, frame) for step, frame in self.frames() if name in step.requested_feature_names]
        if group is not None:
            pairs = [
                (step, frame)
                for step, frame in pairs
                if step.feature_set_options is not None
                and all(
                    k in step.feature_set_options.group and _deep_equal(step.feature_set_options.group[k], v)
                    for k, v in group.items()
                )
            ]
        if len(pairs) == 1:
            return pairs[0][1]
        if not pairs:
            available = sorted({n for step, _ in self.frames() for n in step.requested_feature_names})
            detail = f" with group options {dict(group)}" if group is not None else ""
            raise ValueError(f"No frame requested feature '{name}'{detail}. Requested features: {available}")
        matches = [
            f"(group options {step.feature_set_options.group if step.feature_set_options else {}}, "
            f"feature group {step.feature_group_name}, "
            f"compute framework {step.compute_framework_name})"
            for step, _ in pairs
        ]
        raise ValueError(
            f"Feature name '{name}' is ambiguous, {len(pairs)} frames match: {'; '.join(matches)}. "
            "Pass a Feature with the group options of the one you want."
        )


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
