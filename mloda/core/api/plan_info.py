from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, Optional, TYPE_CHECKING
from uuid import UUID

from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep

if TYPE_CHECKING:
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
    """

    step_kind: Literal["compute", "join", "transform"]
    feature_names: tuple[str, ...]
    feature_group: Optional[type["FeatureGroup"]]
    compute_framework: Optional[type["ComputeFramework"]]
    source_feature_group: Optional[type["FeatureGroup"]]
    source_compute_framework: Optional[type["ComputeFramework"]]
    join_type: Optional[str] = None
    requested_feature_names: tuple[str, ...] = ()
    injected_feature_names: tuple[str, ...] = ()
    input_feature_names: tuple[str, ...] = ()
    join_destination_side: Optional[Literal["left", "right"]] = None
    join_token: Optional[UUID] = field(default=None, compare=False)
    declared_left_frameworks: tuple[type["ComputeFramework"], ...] = ()
    declared_right_frameworks: tuple[type["ComputeFramework"], ...] = ()

    @property
    def feature_group_name(self) -> Optional[str]:
        return None if self.feature_group is None else self.feature_group.get_class_name()

    @property
    def compute_framework_name(self) -> Optional[str]:
        return None if self.compute_framework is None else self.compute_framework.get_class_name()

    @property
    def source_feature_group_name(self) -> Optional[str]:
        return None if self.source_feature_group is None else self.source_feature_group.get_class_name()

    @property
    def source_compute_framework_name(self) -> Optional[str]:
        return None if self.source_compute_framework is None else self.source_compute_framework.get_class_name()

    @property
    def join_inverted(self) -> Optional[bool]:
        return None if self.join_destination_side is None else self.join_destination_side == "right"

    @property
    def declared_left_framework_names(self) -> tuple[str, ...]:
        return tuple(framework.get_class_name() for framework in self.declared_left_frameworks)

    @property
    def declared_right_framework_names(self) -> tuple[str, ...]:
        return tuple(framework.get_class_name() for framework in self.declared_right_frameworks)


def build_plan_steps(
    execution_plan: Iterable[TransformFrameworkStep | JoinStep | FeatureGroupStep],
    resolved_join_plan: Optional["ResolvedJoinPlan"] = None,
) -> list[PlanStep]:
    """Map the steps of an ExecutionPlan onto PlanStep records, in execution-plan order.

    Raises ValueError on an unknown step, mirroring ``ExecutionPlan.add_tfs``: a plan that silently
    drops a step it does not understand is a lie. Pass the plan's ``resolved_join_plan`` to fill the
    join orientation fields; without it join steps report none.
    """
    records: dict[UUID, "ResolvedJoin"] = (
        {} if resolved_join_plan is None else {record.token: record for record in resolved_join_plan.records}
    )

    plan: list[PlanStep] = []

    for step in execution_plan:
        if isinstance(step, FeatureGroupStep):
            feature_names = tuple(str(name) for name in step.features.get_all_names())
            requested = tuple(sorted(str(name) for name in step.features.get_initial_requested_features()))
            injected = tuple(sorted(set(feature_names) - set(requested)))
            declared = step.features.declared_input_feature_names
            input_feature_names = tuple(sorted(declared)) if declared else ()
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

    return plan
