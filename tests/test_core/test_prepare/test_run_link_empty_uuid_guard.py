"""Guard against run_link constructing a JoinStep with an empty declared side.

`set() <= anything` is always True, so an empty `split.left_uuids`/`right_uuids` can slip
through run_link's subset branch and produce an empty destination/source uuid set. Left
unguarded, that empty set later raises an unguarded StopIteration in compute_framework_executor.
"""

from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec
from mloda.user import Link
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class GuardBaseFG(FeatureGroup):
    """Unmatchable base so a leaked subclass stays invisible to feature resolution."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class GuardLeftFG(GuardBaseFG):
    pass


class GuardRightFG(GuardBaseFG):
    pass


def _link() -> Link:
    return Link.inner(JoinSpec(GuardLeftFG, "id"), JoinSpec(GuardRightFG, "id"))


def test_validate_join_step_uuids_rejects_empty_destination_side() -> None:
    """An empty destination side must raise ValueError before JoinStep construction, not later
    surface as StopIteration in compute_framework_executor."""
    link = _link()
    non_empty: set[UUID] = {uuid4()}

    with pytest.raises(ValueError, match="(?i)empty"):
        ExecutionPlan._validate_join_step_uuids(link, set(), non_empty)


def test_validate_join_step_uuids_rejects_empty_source_side() -> None:
    link = _link()
    non_empty: set[UUID] = {uuid4()}

    with pytest.raises(ValueError, match="(?i)empty"):
        ExecutionPlan._validate_join_step_uuids(link, non_empty, set())


def test_validate_join_step_uuids_message_mentions_the_link() -> None:
    """The raised message must be actionable: it must name the offending link, not just 'empty'."""
    link = _link()

    with pytest.raises(ValueError) as exc_info:
        ExecutionPlan._validate_join_step_uuids(link, set(), {uuid4()})

    assert str(link) in str(exc_info.value)


def test_validate_join_step_uuids_accepts_two_genuine_non_empty_sides() -> None:
    """Guard against an overly-broad fix: two real, non-empty sides must not raise."""
    link = _link()

    ExecutionPlan._validate_join_step_uuids(link, {uuid4()}, {uuid4()})


class WiringJoinLeftFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"wiring_join_left"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pandas as pd

        return pd.DataFrame({"wiring_join_left": [1, 2, 3], "id": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("id",))]


class WiringJoinRightFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"wiring_join_right"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        return pa.table({"wiring_join_right": [10, 20, 30], "id": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("id",))]


class WiringJoinChild(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("wiring_join_left"), Feature("wiring_join_right")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"wiring_join_child_result"}


def test_run_link_calls_validate_join_step_uuids_with_non_empty_sides() -> None:
    """Proves the guard is wired into run_link, not just independently correct: an ordinary,
    successful join must invoke it with the genuine (non-empty) destination and source uuids."""
    with patch.object(ExecutionPlan, "_validate_join_step_uuids", wraps=ExecutionPlan._validate_join_step_uuids) as spy:
        mloda.prepare(
            features=[Feature("wiring_join_child_result")],
            links={Link.inner_on(WiringJoinLeftFG, WiringJoinRightFG)},
            compute_frameworks={PandasDataFrame, PyArrowTable},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {WiringJoinLeftFG, WiringJoinRightFG, WiringJoinChild}
            ),
        )

    assert spy.call_count >= 1
    _, destination_uuids, source_uuids = spy.call_args[0]
    assert destination_uuids
    assert source_uuids
