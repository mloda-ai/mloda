from dataclasses import dataclass
from typing import Iterable, Optional, TYPE_CHECKING

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


@dataclass(frozen=True)
class PlanStep:
    """One step of a resolved execution plan.

    ``step_kind`` is "compute", "join" or "transform". For transform steps, ``feature_group`` and
    ``compute_framework`` are the destination and the ``source_*`` fields are the origin.
    """

    step_kind: str
    feature_names: tuple[str, ...]
    feature_group: Optional[type["FeatureGroup"]]
    compute_framework: Optional[type["ComputeFramework"]]
    source_feature_group: Optional[type["FeatureGroup"]]
    source_compute_framework: Optional[type["ComputeFramework"]]

    @property
    def feature_group_name(self) -> Optional[str]:
        return None if self.feature_group is None else self.feature_group.get_class_name()

    @property
    def compute_framework_name(self) -> Optional[str]:
        return None if self.compute_framework is None else self.compute_framework.get_class_name()


def build_plan_steps(
    execution_plan: Iterable[TransformFrameworkStep | JoinStep | FeatureGroupStep],
) -> list[PlanStep]:
    """Map the steps of an ExecutionPlan onto PlanStep records, in execution-plan order."""
    plan: list[PlanStep] = []

    for step in execution_plan:
        if isinstance(step, FeatureGroupStep):
            plan.append(
                PlanStep(
                    step_kind="compute",
                    feature_names=tuple(str(name) for name in step.features.get_all_names()),
                    feature_group=step.feature_group,
                    compute_framework=step.compute_framework,
                    source_feature_group=None,
                    source_compute_framework=None,
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
                    feature_group=None,
                    compute_framework=step.left_framework,
                    source_feature_group=None,
                    source_compute_framework=step.right_framework,
                )
            )

    return plan
