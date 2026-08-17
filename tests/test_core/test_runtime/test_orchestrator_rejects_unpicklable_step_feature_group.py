"""ExecutionOrchestrator.__enter__ must reject a FeatureGroupStep carrying an unpicklable feature
group class before spawning any multiprocessing Manager, mirroring the
ConcatenatedFileContent._create_join_class pattern in read_context_files.py. It must not touch
SYNC mode at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.user import Options, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.input_data.read_files.text_file_reader import PyFileReader

_ORCHESTRATOR_PROBE_CLASS_NAME = "OrchestratorProbeDynamicFeatureGroup"


@pytest.fixture(autouse=True)
def _cleanup_dynamic_feature_groups() -> Iterator[None]:
    yield
    for name in (_ORCHESTRATOR_PROBE_CLASS_NAME, ConcatenatedFileContent.join_feature_name):
        DynamicFeatureGroupCreator._created_classes.pop(name, None)


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


def test_enter_with_multiprocessing_rejects_a_real_concatenated_file_content_plan(tmp_path: Path) -> None:
    """Drives ConcatenatedFileContent's own planner path, not a hand-built step."""
    (tmp_path / "a.py").write_text("# a")
    (tmp_path / "b.py").write_text("# b")

    ConcatenatedFileContent()._create_join_class(ConcatenatedFileContent.join_feature_name)

    api = mloda(
        [
            Feature(
                "ConcatenatedFileContent",
                options=Options(
                    {"target_folder": [str(tmp_path)], "document_reader_class": PyFileReader.get_class_name()}
                ),
            )
        ],
        compute_frameworks={PandasDataFrame},
    )

    assert api.engine is not None
    with pytest.raises(ValueError, match=ConcatenatedFileContent.join_feature_name):
        api.engine.compute().__enter__({ParallelizationMode.MULTIPROCESSING})
