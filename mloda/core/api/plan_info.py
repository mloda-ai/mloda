import heapq
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING
from uuid import UUID

from mloda.core.abstract_plugins.components.base_feature_group_version import BaseFeatureGroupVersion
from mloda.core.abstract_plugins.components.credential_scrub import _mask_secret_keys, redact_option_value
from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.abstract_plugins.components.input_data.base_input_data import (
    _is_fallback_identity,
)
from mloda.core.abstract_plugins.components.input_data.format_feature_group import (
    concrete_format_groups,
    implementation_path,
)
from mloda.core.abstract_plugins.components.options import Options, _safe_deepcopy
from mloda.core.abstract_plugins.components.utils import as_str, safe_field
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.choose_compute_frameworks import _refine
from mloda.core.prepare.stable_text import stable_text
from mloda.core.prepare.resolution_failure_renderer import _candidate_sort_key
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework
    from mloda.core.abstract_plugins.feature_group import FeatureGroup
    from mloda.core.abstract_plugins.implementation import Implementation
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

    ``feature_set_options`` is a compute step's group-only, deep-copied snapshot of ``FeatureSet.options``
    (a mapping under a format group's name, a pointer, is redacted to its keys; other mappings have
    credential-shaped keys masked),
    and ``step_uuid`` its ``FeatureGroupStep.uuid`` (equal to ``HookContext.step_uuid`` on that step's hooks), the
    key ``RunResult.frames()`` pairs frames by; both are None for join/transform steps and, like ``join_token``,
    excluded from equality.
    ``input_feature_edges`` maps each output feature name to its declared inputs (injected features absent);
    it participates in equality but is excluded from hashing.

    ``specialized_from`` lists, for a compute step, the parent classes its feature group replaced for at least one
    feature in the step (subclass preference), sorted by class name; empty otherwise.

    ``compute_framework_reason`` is why the central choice put a compute step on its framework: the distinct
    reasons of its features, sorted and joined with "; ", or None (join/transform steps, or no recorded reason).

    ``result_framework`` is the framework whose data type a step's requested features come back in: the
    ``output_framework`` option if set, else the step framework's ``result_framework()``; None otherwise.

    ``data_access_identity`` and ``data_access_identity_is_fallback`` mirror the ``HookContext`` fields for the
    ``FeatureSet.input_data_match`` pair; they are credential-free, computed at plan build and excluded from equality.

    ``join_keys`` is a join step's ``left=right`` index column pairs, as ``HookContext.join_keys``; None for
    APPEND/UNION and non-join steps.

    ``implementation``: the Implementation class serving a compute step, else None.
    ``implementation_path``: its ``module:qualname``, "neutral" for a format group step without one, else None.
    ``implementation_specialized_from``: the winner's concrete Implementation ancestors that serve the step's pair, sorted by path.
    ``implementation_hash`` is the first-party code closure hash, never dependency versions ("unavailable" when
    hashing fails). ``feature_group_hash``: the group's code hash, set when an inherited implementation serves a
    subclass group or a format group loads neutrally. ``implementation_version``, ``implementation_hash``,
    ``feature_group_version``, ``feature_group_hash``: excluded from equality.
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
    compute_framework_reason: str | None = None
    result_framework: type["ComputeFramework"] | None = None
    join_keys: tuple[str, ...] | None = None
    implementation: type["Implementation"] | None = None
    implementation_version: str | None = field(default=None, compare=False)
    implementation_hash: str | None = field(default=None, compare=False)
    implementation_specialized_from: tuple[type["Implementation"], ...] = ()
    feature_group_version: str | None = field(default=None, compare=False)
    feature_group_hash: str | None = field(default=None, compare=False)
    _data_access_identity: str | None = field(default=None, compare=False)
    _data_access_identity_is_fallback: bool | None = field(default=None, compare=False)
    _neutral: bool = field(default=False, compare=False)

    @property
    def implementation_path(self) -> str | None:
        if self.implementation is not None:
            return _class_path(self.implementation)
        return "neutral" if self._neutral else None

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
        return self._data_access_identity

    @property
    def data_access_identity_is_fallback(self) -> bool | None:
        return self._data_access_identity_is_fallback


def _snapshot_option(key: Any, value: Any, format_group_names: Iterable[str]) -> Any:
    if not isinstance(value, Mapping):
        return _safe_deepcopy(value, {})
    if key in format_group_names:
        return redact_option_value(value)
    return _mask_secret_keys(_safe_deepcopy(value, {}))


def snapshot_group(group: Mapping[Any, Any]) -> dict[Any, Any]:
    """Snapshot a group mapping with the plan-step redaction rule."""
    names = concrete_format_groups().keys()
    return {key: _snapshot_option(key, value, names) for key, value in group.items()}


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
            # A feature requested by part (base~part) is not injected under its base name.
            requested_bases = {str(f.name) for f in step.features.features if f.initial_requested_data}
            injected = tuple(sorted(set(feature_names) - set(requested) - requested_bases))
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
            match = step.features.input_data_match
            identity = None if match is None else match[0].data_access_identity(match[1])
            is_fallback = None if match is None or identity is None else _is_fallback_identity(match[1], identity)
            impl = step.features.implementation
            path = implementation_path(step.feature_group, impl, match)
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
                        Options(group=snapshot_group(step.features.options.group))
                        if step.features.options is not None
                        else None
                    ),
                    step_uuid=step.uuid,
                    input_feature_edges={
                        name: tuple(sorted(inputs))
                        for name, inputs in (step.features.declared_input_feature_edges or {}).items()
                    },
                    specialized_from=tuple(sorted(replaced, key=_candidate_sort_key)),
                    compute_framework_reason="; ".join(sorted(reasons)) or None,
                    result_framework=(output_framework or step.compute_framework.result_framework())
                    if requested
                    else None,
                    implementation=impl,
                    implementation_version=None if impl is None else safe_field(lambda: as_str(impl.version()), None),
                    implementation_hash=None
                    if impl is None
                    else safe_field(lambda: BaseFeatureGroupVersion.first_party_hash(impl), "unavailable"),
                    implementation_specialized_from=_replaced_implementations(impl, step),
                    feature_group_version=safe_field(lambda: as_str(step.feature_group.version()), None),
                    feature_group_hash=None
                    if path is None or (impl is not None and step.feature_group is impl.feature_group)
                    else safe_field(
                        lambda: BaseFeatureGroupVersion.first_party_hash(step.feature_group), "unavailable"
                    ),
                    _data_access_identity=identity,
                    _data_access_identity_is_fallback=is_fallback,
                    _neutral=path == "neutral",
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
                    join_keys=step.join_keys(),
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


def _replaced_implementations(
    impl: "type[Implementation] | None", step: FeatureGroupStep
) -> tuple[type["Implementation"], ...]:
    """The concrete Implementation bases of impl that serve the step's group and framework."""
    from mloda.core.abstract_plugins.implementation import Implementation, _is_concrete

    if impl is None:
        return ()
    parents = [
        base
        for base in impl.__mro__[1:]
        if issubclass(base, Implementation)
        and _is_concrete(base)
        and issubclass(step.feature_group, base.feature_group)
        and issubclass(step.compute_framework, base.compute_framework)
    ]
    return tuple(sorted(parents, key=_class_path))


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
        ",".join(_class_path(cf) for cf in record.declared_left_frameworks),
        ",".join(_class_path(cf) for cf in record.declared_right_frameworks),
        ",".join(record.requested_feature_names),
        ",".join(record.injected_feature_names),
        ",".join(_class_path(fg) for fg in record.specialized_from),
        record.compute_framework_reason or "",
        _class_path(record.result_framework),
        ",".join(record.join_keys or ()),
        record.implementation_path or "",
        ",".join(_class_path(impl) for impl in record.implementation_specialized_from),
    )


def _dependency_order(raw_steps: list[Any], plan: list[PlanStep]) -> list[PlanStep]:
    """Topological order over step tokens; ready steps go by content, ties by the content of their neighbours."""
    producer_of = {token: index for index, step in enumerate(raw_steps) for token in step.get_uuids()}
    waits_for = [
        {producer_of[token] for token in step.required_uuids if token in producer_of} - {index}
        for index, step in enumerate(raw_steps)
    ]
    waiters: dict[int, list[int]] = {}
    for index, producers in enumerate(waits_for):
        for producer in producers:
            waiters.setdefault(producer, []).append(index)
    waiter_sets = [set(waiters.get(index, ())) for index in range(len(plan))]
    classes = _refine([_content_key(record) for record in plan], [set(p) for p in waits_for], waiter_sets)
    # raw index only separates steps that refinement cannot tell apart
    ready = [(classes[index], index) for index, producers in enumerate(waits_for) if not producers]
    heapq.heapify(ready)
    ordered: list[PlanStep] = []
    while ready:
        _, index = heapq.heappop(ready)
        ordered.append(plan[index])
        for waiter in waiters.get(index, ()):
            waits_for[waiter].discard(index)
            if not waits_for[waiter]:
                heapq.heappush(ready, (classes[waiter], waiter))
    if len(ordered) != len(plan):
        raise ValueError(
            internal_invariant_error(
                "the steps of the plan form a cycle.", f"steps={len(plan) - len(ordered)} unordered"
            )
        )
    return ordered
