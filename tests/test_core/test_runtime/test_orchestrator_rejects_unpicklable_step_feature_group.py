"""ExecutionOrchestrator.__enter__ must reject a FeatureGroupStep carrying an unpicklable feature
group class before spawning any multiprocessing Manager, mirroring the
ConcatenatedFileContent._create_join_class pattern in read_context_files.py. It must not touch
SYNC mode at all.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)


def _plan_with_unpicklable_feature_group_step() -> ExecutionPlan:
    dynamic_fg = DynamicFeatureGroupCreator.create(properties={}, class_name="OrchestratorProbeDynamicFeatureGroup")
    step = FeatureGroupStep(dynamic_fg, FeatureSet([Feature("f")]), set(), PandasDataFrame)

    plan = ExecutionPlan()
    plan.execution_plan = [step]
    return plan


def test_enter_with_multiprocessing_rejects_an_unpicklable_feature_group_step_before_spawning_a_manager() -> None:
    orchestrator = ExecutionOrchestrator(_plan_with_unpicklable_feature_group_step())

    with pytest.raises(ValueError):
        orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING})

    assert orchestrator.manager is None, "no MyManager/worker process may be created on the rejection path"


def test_enter_with_sync_mode_does_not_reject_the_same_unpicklable_feature_group_step() -> None:
    orchestrator = ExecutionOrchestrator(_plan_with_unpicklable_feature_group_step())

    orchestrator.__enter__({ParallelizationMode.SYNC})

    orchestrator.__exit__(None, None, None)
