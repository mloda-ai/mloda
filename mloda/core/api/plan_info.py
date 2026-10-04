import heapq
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING
from uuid import UUID

from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.abstract_plugins.components.input_data.base_input_data import (
    _is_fallback_identity,
)
from mloda.core.abstract_plugins.components.options import Options, _safe_deepcopy
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.choose_compute_frameworks import stable_text
from mloda.core.prepare.resolution_failure_renderer import _candidate_sort_key
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework
    from mloda.core.abstract_plugins.feature_group import FeatureGroup
    from mloda.core.prepare.resolved_join import ResolvedJoin, ResolvedJoinPlan


@dataclass(frozen=True)
class PlanStep:
    """One step of a resolved execution plan.

    ``step_kind`` is "compute", "join" or "transform".

    compute: ``feature_names`` are the names computed by ``feature_group`` on ``compute_framework``.
    The names include engine-injected features (link index features, global-filter features):
    ``requested_feature_names`` holds the user-requested names, ``injected_feature_names`` the
    engine-injected/dependency remainder; both are empty for join and transform steps.
    The split is name-based, so a name that is both user-requested and engine-injected within
    one step counts as requested only.
    ``input_feature_names`` holds the sorted, deduplicated names the feature group declares as
    input; it is empty for a root step and for join and transform steps. It is the prepare-time twin
    of the run-time ``HookContext.input_features``, which ``ComputeFramework._build_hook_context``
    fills from the same FeatureSet attribute.
    ``source_*`` and ``join_type`` are None.

    transform: ``feature_group``/``compute_framework`` are the destination, ``source_*`` the origin.

    join: ``feature_group``/``source_feature_group`` are the link's declared left/right sides, and
    ``join_type`` its join type. ``compute_framework`` is the merge destination and
    ``source_compute_framework`` the framework merged in. ``join_destination_side`` is the declared
    side holding the destination, resolved from the declared sides' framework candidates;
    APPEND/UNION report "left", and a right join reports "right" in the common case. When
    declared-side membership doesn't decide, a differing destination/source framework breaks the
    tie by identity against the trekker key (the path RIGHT joins usually take); matching
    frameworks, including same-framework RIGHT joins, fall back to the link's trekker-key flip flag
    instead.
    ``join_inverted`` is a derived property (``join_destination_side == "right"``, or None without
    a side). ``join_token`` is the join
    step's completion token, minted fresh per planning run and therefore excluded from equality.
    All three are None without a resolved join plan. ``declared_left_frameworks``/
    ``declared_right_frameworks`` are the classes each declared side's parent features declared as
    candidates, sorted by class name; ``()`` when no resolved join plan is given, or when the plan
    recorded no candidates for that side. APPEND/UNION sides carry only the index-bearing parent.

    ``feature_set_options`` is a compute step's group-only, deep-copied snapshot of ``FeatureSet.options``,
    and ``step_uuid`` its ``FeatureGroupStep.uuid``, the key ``RunResult.frames()`` pairs frames by; both
    are None for join/transform steps and, like ``join_token``, excluded from equality.
    ``input_feature_edges`` maps each output feature name to its declared inputs (injected features absent);
    it participates in equality but is excluded from hashing.

    ``specialized_from`` lists, for a compute step, the parent classes its feature group replaced for at least one
    feature in the step (subclass preference), sorted by class name; empty otherwise.

    ``compute_framework_reason`` is why the central choice put a compute step on its framework: the distinct
    reasons of its features, sorted and joined with "; ", or None (join/transform steps, or no recorded reason).

    ``result_framework`` is the framework a compute step's requested features come back in: the
    ``output_framework`` option if set, else ``compute_framework``; None without requested features and for join/transform steps.

    ``reader_data_access`` is the (reader class, data access) pair of ``FeatureSet.input_data_match``, excluded from equality.
    ``data_access_identity`` and ``data_access_identity_is_fallback`` mirror the ``HookContext`` fields for that
    pair, computed on access.
    """

    step_kind: Literal["compute", "join", "transform"]
    feature_names: tuple[str, ...]
    feature_group: type["FeatureGroup"] | None
    compute_framework: type["ComputeFramework"] | None
    source_feature_group: type["FeatureGroup"] | None
    source_compute_framework: type["ComputeFramework"] | None
    join_type: str | None = None
    requested_feature_names: tuple[str, ...] = ()
    injected_feature_names: tuple[str, ...] = ()
    input_feature_names: tuple[str, ...] = ()
    join_destination_side: Literal["left", "right"] | None = None
    join_token: UUID | None = field(default=None, compare=False)
    declared_left_frameworks: tuple[type["ComputeFramework"], ...] = ()
    declared_right_frameworks: tuple[type["ComputeFramework"], ...] = ()
    feature_set_options: Options | None = field(default=None, compare=False)
    step_uuid: UUID | None = field(default=None, compare=False)
    input_feature_edges: Mapping[str, tuple[str, ...]] = field(default_factory=dict, hash=False)
    specialized_from: tuple[type["FeatureGroup"], ...] = ()
    reader_data_access: tuple[type["BaseInputData"], Any] | None = field(default=None, compare=False)
    compute_framework_reason: str | None = None
    result_framework: type["ComputeFramework"] | None = None

    @property
    def result_framework_name(self) -> str | None:
        return None if self.result_framework is None else self.result_framework.get_class_name()

    @property
    def feature_group_name(self) -> str | None:
        return None if self.feature_group is None else self.feature_group.get_class_name()

    @property
    def compute_framework_name(self) -> str | None:
        return None if self.compute_framework is None else self.compute_framework.get_class_name()

    @property
    def source_feature_group_name(self) -> str | None:
        return None if self.source_feature_group is None else self.source_feature_group.get_class_name()

    @property
    def source_compute_framework_name(self) -> str | None:
        return None if self.source_compute_framework is None else self.source_compute_framework.get_class_name()

    @property
    def join_inverted(self) -> bool | None:
        return None if self.join_destination_side is None else self.join_destination_side == "right"

    @property
    def declared_left_framework_names(self) -> tuple[str, ...]:
        return tuple(framework.get_class_name() for framework in self.declared_left_frameworks)

    @property
    def declared_right_framework_names(self) -> tuple[str, ...]:
        return tuple(framework.get_class_name() for framework in self.declared_right_frameworks)

    @property
    def data_access_identity(self) -> str | None:
        pair = self.reader_data_access
        return None if pair is None else pair[0].data_access_identity(pair[1])

    @property
    def data_access_identity_is_fallback(self) -> bool | None:
        pair = self.reader_data_access
        return None if pair is None else _is_fallback_identity(pair[1], pair[0].data_access_identity(pair[1]))


def build_plan_steps(
    execution_plan: Iterable[TransformFrameworkStep | JoinStep | FeatureGroupStep],
    resolved_join_plan: "ResolvedJoinPlan | None" = None,
    specialized_from: Mapping[UUID, tuple[type["FeatureGroup"], ...]] | None = None,
    output_framework: type["ComputeFramework"] | None = None,
) -> list[PlanStep]:
    """Map the steps of an ExecutionPlan onto PlanStep records, in dependency order.

    Independent steps are sorted by content, so every process reports the same order.

    Raises ValueError on an unknown step, mirroring ``ExecutionPlan.add_tfs``: a plan that silently
    drops a step it does not understand is a lie. Pass the plan's ``resolved_join_plan`` to fill the
    join orientation fields; without it join steps report none. Pass ``specialized_from`` (feature uuid to
    replaced parents) to fill each compute step's ``specialized_from``.
    """
    replaced_by_uuid = specialized_from or {}
    records: dict[UUID, "ResolvedJoin"] = (
        {} if resolved_join_plan is None else {record.token: record for record in resolved_join_plan.records}
    )

    plan: list[PlanStep] = []
    raw_steps = list(execution_plan)

    for step in raw_steps:
        if isinstance(step, FeatureGroupStep):
            feature_names = tuple(str(name) for name in step.features.get_all_names())
            requested = tuple(sorted(str(name) for name in step.features.get_initial_requested_features()))
            injected = tuple(sorted(set(feature_names) - set(requested)))
            declared = step.features.declared_input_feature_names
            input_feature_names = tuple(sorted(declared)) if declared else ()
            replaced = {
                parent
                for feature_id in step.features.get_all_feature_ids()
                for parent in replaced_by_uuid.get(feature_id, ())
            }
            reasons = {
                feature.chosen_compute_framework_reason
                for feature in step.features.features
                if feature.chosen_compute_framework is step.compute_framework
                and feature.chosen_compute_framework_reason is not None
            }
            plan.append(
                PlanStep(
                    step_kind="compute",
                    feature_names=feature_names,
                    feature_group=step.feature_group,
                    compute_framework=step.compute_framework,
                    source_feature_group=None,
                    source_compute_framework=None,
                    requested_feature_names=requested,
                    injected_feature_names=injected,
                    input_feature_names=input_feature_names,
                    feature_set_options=(
                        Options(
                            group={key: _safe_deepcopy(value, {}) for key, value in step.features.options.group.items()}
                        )
                        if step.features.options is not None
                        else None
                    ),
                    step_uuid=step.uuid,
                    input_feature_edges={
                        name: tuple(sorted(inputs))
                        for name, inputs in (step.features.declared_input_feature_edges or {}).items()
                    },
                    specialized_from=tuple(sorted(replaced, key=_candidate_sort_key)),
                    reader_data_access=_safe_deepcopy(step.features.input_data_match, {}),
                    compute_framework_reason="; ".join(sorted(reasons)) or None,
                    result_framework=(output_framework or step.compute_framework) if requested else None,
                )
            )
        elif isinstance(step, TransformFrameworkStep):
            plan.append(
                PlanStep(
                    step_kind="transform",
                    feature_names=(),
                    feature_group=step.to_feature_group,
                    compute_framework=step.to_framework,
                    source_feature_group=step.from_feature_group,
                    source_compute_framework=step.from_framework,
                )
            )
        elif isinstance(step, JoinStep):
            record = None
            if resolved_join_plan is not None:
                if step.uuid not in records:
                    raise ValueError(
                        internal_invariant_error(
                            "a planned JoinStep has no resolved join record in the given plan.",
                            f"join_step_uuid={step.uuid}, link={step.link}, "
                            f"record_tokens={sorted(str(token) for token in records)}",
                        )
                    )
                record = records[step.uuid]
            plan.append(
                PlanStep(
                    step_kind="join",
                    feature_names=(),
                    feature_group=step.link.left_feature_group,
                    compute_framework=step.destination_framework,
                    source_feature_group=step.link.right_feature_group,
                    source_compute_framework=step.source_framework,
                    join_type=step.link.jointype.value,
                    join_destination_side=None if record is None else record.destination_side.value,
                    join_token=None if record is None else record.token,
                    declared_left_frameworks=()
                    if record is None
                    else tuple(sorted(record.left.declared_frameworks, key=lambda cf: cf.get_class_name())),
                    declared_right_frameworks=()
                    if record is None
                    else tuple(sorted(record.right.declared_frameworks, key=lambda cf: cf.get_class_name())),
                )
            )
        else:
            raise ValueError(f"Element {step} is not a valid element.")

    return _dependency_order(raw_steps, plan)


def _class_path(cls: type | None) -> str:
    return "" if cls is None else f"{cls.__module__}:{cls.__qualname__}"


def _content_key(record: PlanStep) -> tuple[str, ...]:
    return (
        record.step_kind,
        _class_path(record.feature_group),
        _class_path(record.compute_framework),
        _class_path(record.source_feature_group),
        _class_path(record.source_compute_framework),
        ",".join(sorted(record.feature_names)),
        record.join_type or "",
        record.join_destination_side or "",
        stable_text(record.feature_set_options),
        ",".join(record.input_feature_names),
    )


def _dependency_order(raw_steps: list[Any], plan: list[PlanStep]) -> list[PlanStep]:
    """Topological order over step tokens; the smallest content key among ready steps goes first."""
    producer_of = {token: index for index, step in enumerate(raw_steps) for token in step.get_uuids()}
    waits_for = [
        {producer_of[token] for token in step.required_uuids if token in producer_of} - {index}
        for index, step in enumerate(raw_steps)
    ]
    waiters: dict[int, list[int]] = {}
    for index, producers in enumerate(waits_for):
        for producer in producers:
            waiters.setdefault(producer, []).append(index)
    keys = [_content_key(record) for record in plan]
    ready = [(keys[index], index) for index, producers in enumerate(waits_for) if not producers]
    heapq.heapify(ready)
    ordered: list[PlanStep] = []
    while ready:
        _, index = heapq.heappop(ready)
        ordered.append(plan[index])
        for waiter in waiters.get(index, ()):
            waits_for[waiter].discard(index)
            if not waits_for[waiter]:
                heapq.heappush(ready, (keys[waiter], waiter))
    if len(ordered) != len(plan):
        raise ValueError(
            internal_invariant_error(
                "the steps of the plan form a cycle.", f"steps={len(plan) - len(ordered)} unordered"
            )
        )
    return ordered
