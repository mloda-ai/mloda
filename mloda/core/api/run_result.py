"""Run results that carry the resolved plan of the run that produced them."""

from collections.abc import Generator
from types import TracebackType
from typing import TYPE_CHECKING, Any
from uuid import UUID

from mloda.core.abstract_plugins.components.credential_scrub import redact_option_value
from mloda.core.abstract_plugins.components.hashable_dict import _deep_equal
from mloda.core.api.plan_info import PlanStep, snapshot_group

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.feature import Feature


class RunResult(list[Any]):
    """The results list of a ``run_all`` or ``run()`` call, plus the resolved plan of the run that produced it.

    Slicing and concatenation return a plain ``list`` without ``plan`` (standard list-subclass behavior).
    The list is in plan order, one element per step that produced requested output; ``frames()``
    pairs each frame with its ``PlanStep`` by step identity, regardless of order.
    """

    def __init__(
        self,
        results: list[Any],
        plan: list[PlanStep],
        result_items: list[tuple[UUID, Any]] | None = None,
        *,
        run_id: str | None = None,
        plan_id: str | None = None,
        connection_handles: dict[int, str] | None = None,
    ) -> None:
        super().__init__(results)
        self._connection_handles = connection_handles or {}
        self._plan = plan
        self._result_items = tuple(result_items or ())
        self._run_id = run_id
        self._plan_id = plan_id

    @property
    def run_id(self) -> str | None:
        """Id of the run that produced these results, the run's ``HookContext.run_id``."""
        return self._run_id

    @property
    def plan_id(self) -> str | None:
        """Id of the prepared session, ``HookContext.plan_id``."""
        return self._plan_id

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
        group = None if isinstance(feature, str) else snapshot_group(feature.options.group)
        if not isinstance(feature, str) and feature.connection is not None:
            handle = self._connection_handles.get(id(feature.connection))
            group = {**(group or {}), "connection_handle": handle}
        pinned = None if isinstance(feature, str) else feature.compute_frameworks
        frames = self.frames()
        pairs = [(step, frame) for step, frame in frames if name in step.requested_feature_names]
        if group is not None:
            pairs = [
                (step, frame)
                for step, frame in pairs
                if all(k in _group_of(step) and _deep_equal(_group_of(step)[k], v) for k, v in group.items())
                and (not pinned or step.compute_framework in pinned)
            ]
            fewest = min((len(_group_of(step)) for step, _ in pairs), default=0)
            pairs = [(step, frame) for step, frame in pairs if len(_group_of(step)) == fewest]
        if len(pairs) == 1:
            return pairs[0][1]
        if not pairs:
            available = sorted({n for step, _ in frames for n in step.requested_feature_names})
            detail = f" with group options {_redacted(group)}" if group is not None else ""
            hint = ""
            if name not in available:
                computed = sorted({str(step.feature_group_name) for step in self._plan if name in step.feature_names})
                if computed:
                    hint = f"; '{name}' was computed by {', '.join(computed)} but not requested; request it to get its frame"
            raise ValueError(f"No frame requested feature '{name}'{detail}. Requested features: {available}{hint}")
        matches = [
            f"(group options {_redacted(_group_of(step))}, "
            f"feature group {step.feature_group_name}, "
            f"compute framework {step.compute_framework_name})"
            for step, _ in pairs
        ]
        raise ValueError(
            f"Feature name '{name}' is ambiguous, {len(pairs)} frames match: {'; '.join(matches)}. "
            "Pass a Feature with the group options of the one you want."
        )


def _group_of(step: PlanStep) -> Any:
    return {} if step.feature_set_options is None else step.feature_set_options.group


def _redacted(group: Any) -> dict[Any, Any]:
    return {k: redact_option_value(v) for k, v in group.items()}


class ResultStream(Generator[Any, None, None]):
    """A ``stream_all`` or ``stream_run()`` result stream: iterates like a generator and exposes the resolved plan."""

    def __init__(
        self,
        stream: Generator[tuple[UUID, Any], None, None],
        plan: list[PlanStep],
        *,
        run_id: str | None = None,
        plan_id: str | None = None,
    ) -> None:
        self._stream = stream
        self._plan = plan
        self._run_id = run_id
        self._plan_id = plan_id

    @property
    def run_id(self) -> str | None:
        """Id of the run, the run's ``HookContext.run_id``, available before any element is consumed."""
        return self._run_id

    @property
    def plan_id(self) -> str | None:
        """Id of the prepared session, ``HookContext.plan_id``, available before any element is consumed."""
        return self._plan_id

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
