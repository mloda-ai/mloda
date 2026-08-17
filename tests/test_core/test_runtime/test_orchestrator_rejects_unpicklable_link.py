"""ExecutionOrchestrator.__enter__ must reject an unpicklable JoinStep link before
spawning any multiprocessing Manager, and must not touch SYNC mode at all. See #1117.
"""

from __future__ import annotations

import pickle  # nosec

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.provider import FeatureGroup
from mloda.user import Index, JoinSpec, Link
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class OrchestratorLinkRight(FeatureGroup):
    """Ordinary module-level feature group: picklable."""


def _make_local_feature_group() -> type[FeatureGroup]:
    """A FeatureGroup subclass whose __qualname__ contains '<locals>': unpicklable."""

    class LocallyDefinedFeatureGroup(FeatureGroup):
        pass

    return LocallyDefinedFeatureGroup


def _plan_with_unpicklable_link() -> ExecutionPlan:
    link = Link.inner(
        JoinSpec(_make_local_feature_group(), Index(("left_key",))),
        JoinSpec(OrchestratorLinkRight, Index(("right_key",))),
    )
    join_step = JoinStep(link, PyArrowTable, PandasDataFrame, set(), set(), set())

    plan = ExecutionPlan()
    plan.execution_plan = [join_step]
    return plan


def test_enter_with_multiprocessing_rejects_an_unpicklable_link_before_spawning_a_manager() -> None:
    orchestrator = ExecutionOrchestrator(_plan_with_unpicklable_link())

    with pytest.raises(ValueError) as excinfo:
        orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING})

    assert not isinstance(excinfo.value, pickle.PicklingError), (
        "the plan-time guard must fire, not the deep multiprocessing pickling failure"
    )
    assert orchestrator.manager is None, "no MyManager/worker process may be created on the rejection path"


def test_enter_with_sync_mode_does_not_reject_the_same_unpicklable_link() -> None:
    orchestrator = ExecutionOrchestrator(_plan_with_unpicklable_link())

    orchestrator.__enter__({ParallelizationMode.SYNC})

    orchestrator.__exit__(None, None, None)
