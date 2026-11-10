"""ExecutionOrchestrator.__enter__ must reject a FeatureGroupStep carrying an unpicklable feature
group class before spawning any multiprocessing Manager. The step carries a DynamicFeatureGroupCreator
class, which cannot be pickled by module path. It must not touch SYNC mode at all.
"""

from __future__ import annotations

import threading

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)

_ORCHESTRATOR_PROBE_CLASS_NAME = "OrchestratorProbeDynamicFeatureGroup"


def _plan_with_unpicklable_feature_group_step() -> ExecutionPlan:
    dynamic_fg = DynamicFeatureGroupCreator.create(properties={}, class_name=_ORCHESTRATOR_PROBE_CLASS_NAME)
    step = FeatureGroupStep(dynamic_fg, FeatureSet([Feature("f")]), set(), PandasDataFrame)

    plan = ExecutionPlan()
    plan.execution_plan = [step]
    return plan


def test_enter_with_multiprocessing_rejects_an_unpicklable_feature_group_step_before_spawning_a_manager() -> None:
    orchestrator = ExecutionOrchestrator(_plan_with_unpicklable_feature_group_step())

    with pytest.raises(ValueError, match=_ORCHESTRATOR_PROBE_CLASS_NAME):
        orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING})

    assert orchestrator.manager is None, "no MyManager/worker process may be created on the rejection path"


def test_enter_with_sync_mode_does_not_reject_the_same_unpicklable_feature_group_step() -> None:
    orchestrator = ExecutionOrchestrator(_plan_with_unpicklable_feature_group_step())

    orchestrator.__enter__({ParallelizationMode.SYNC})

    orchestrator.__exit__(None, None, None)


_CREDENTIAL_SECRET = "zz-orchestrator-secret-77"


def _plan_and_collection_with_unpicklable_credential() -> tuple[ExecutionPlan, DataAccessCollection]:
    class _SecretHolder:
        def __init__(self) -> None:
            self.lock = threading.Lock()

        def __repr__(self) -> str:
            return _CREDENTIAL_SECRET

    step = FeatureGroupStep(FeatureGroup, FeatureSet([Feature("f")]), set(), PandasDataFrame)
    step.credential_handles = ("h",)
    plan = ExecutionPlan()
    plan.execution_plan = [step]
    return plan, DataAccessCollection(credentials={"h": {"client": _SecretHolder()}})


def test_enter_with_multiprocessing_rejects_an_unpicklable_credential_without_leaking_it() -> None:
    plan, collection = _plan_and_collection_with_unpicklable_credential()
    orchestrator = ExecutionOrchestrator(plan, data_access_collection=collection)

    with pytest.raises(ValueError) as excinfo:
        orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING})

    assert "'h'" in str(excinfo.value)
    assert _CREDENTIAL_SECRET not in str(excinfo.value)
    assert orchestrator.manager is None
    assert orchestrator.worker_manager.standby == []


def test_enter_with_sync_mode_does_not_reject_an_unpicklable_credential() -> None:
    plan, collection = _plan_and_collection_with_unpicklable_credential()
    orchestrator = ExecutionOrchestrator(plan, data_access_collection=collection)

    orchestrator.__enter__({ParallelizationMode.SYNC})

    orchestrator.__exit__(None, None, None)
