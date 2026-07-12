from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


@dataclass(frozen=True)
class PlanStep:
    """One step of a resolved execution plan.

    ``step_kind`` is "compute", "join" or "transform".

    compute: ``feature_names`` are the names computed by ``feature_group`` on ``compute_framework``.
    The names include engine-injected features (link index features, global-filter features):
    ``requested_feature_names`` holds the user-requested names, ``injected_feature_names`` the
    engine-injected/dependency remainder; both are empty for join and transform steps.
    ``source_*`` and ``join_type`` are None.

    transform: ``feature_group``/``compute_framework`` are the destination, ``source_*`` the origin.

    join: ``feature_group``/``source_feature_group`` are the link's declared left/right sides, and
    ``join_type`` its join type. ``compute_framework`` is the merge destination and
    ``source_compute_framework`` the framework merged in. Those two are not necessarily the
    frameworks of the declared left/right sides: ``ExecutionPlan.run_link`` swaps them for RIGHT
    joins (and when no child needs the declared orientation), so for a RIGHT join the destination
    framework belongs to the declared right side.
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


def build_plan_steps(
    execution_plan: Iterable[TransformFrameworkStep | JoinStep | FeatureGroupStep],
) -> list[PlanStep]:
    """Map the steps of an ExecutionPlan onto PlanStep records, in execution-plan order.

    Raises ValueError on an unknown step, mirroring ``ExecutionPlan.add_tfs``: a plan that silently
    drops a step it does not understand is a lie.
    """
    plan: list[PlanStep] = []

    for step in execution_plan:
        if isinstance(step, FeatureGroupStep):
            feature_names = tuple(str(name) for name in step.features.get_all_names())
            requested = tuple(sorted(str(name) for name in step.features.get_initial_requested_features()))
            injected = tuple(sorted(set(feature_names) - set(requested)))
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
            plan.append(
                PlanStep(
                    step_kind="join",
                    feature_names=(),
                    feature_group=step.link.left_feature_group,
                    compute_framework=step.left_framework,
                    source_feature_group=step.link.right_feature_group,
                    source_compute_framework=step.right_framework,
                    join_type=step.link.jointype.value,
                )
            )
        else:
            raise ValueError(f"Element {step} is not a valid element.")

    return plan
