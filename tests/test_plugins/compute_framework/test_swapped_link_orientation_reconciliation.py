"""Two child groups pinning the same link to swapped framework pairs.

Resolution reconciles the two orientations at once, so both children see the joined columns.
"""

from typing import Any, Optional

import pyarrow as pa
import pytest

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, Link
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

LEFT_PAYLOAD = "swapped_left_payload"
RIGHT_PAYLOAD = "swapped_right_payload"
JOINED_VALUE = "left|right"


def _column_names(data: Any) -> list[str]:
    if isinstance(data, pa.Table):
        return list(data.column_names)
    return list(data.columns)


def _values(data: Any, column: str) -> list[Any]:
    if isinstance(data, pa.Table):
        return list(data[column].to_pylist())
    return list(data[column])


def _add_joined_column(data: Any, name: str) -> Any:
    """Writes the two parent payloads into one column, so the joined data itself is observable."""
    joined = [f"{left}|{right}" for left, right in zip(_values(data, LEFT_PAYLOAD), _values(data, RIGHT_PAYLOAD))]
    if isinstance(data, pa.Table):
        return data.append_column(name, pa.array(joined))
    data[name] = joined
    return data


class SwappedParentLeft(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"], LEFT_PAYLOAD: ["left"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


class SwappedParentRight(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"], RIGHT_PAYLOAD: ["right"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


class SwappedChildPandasLeft(FeatureGroup):
    """Pins the left parent to pandas and the right parent to pyarrow."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name=SwappedParentLeft.get_class_name(), compute_framework="PandasDataFrame"),
            Feature(name=SwappedParentRight.get_class_name(), compute_framework="PyArrowTable"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _add_joined_column(data, cls.get_class_name())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


class SwappedChildPyArrowLeft(FeatureGroup):
    """Pins the same two parents the other way round, so the link is recorded in both orientations."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name=SwappedParentLeft.get_class_name(), compute_framework="PyArrowTable"),
            Feature(name=SwappedParentRight.get_class_name(), compute_framework="PandasDataFrame"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _add_joined_column(data, cls.get_class_name())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


# Threading is left out: both orientations produce a JoinStep carrying the same link uuid as its completion
# token, so a child can start once either join reported done. That predates this scenario.
@pytest.mark.parametrize("modes", [({ParallelizationMode.SYNC})])
class TestSwappedLinkOrientationReconciliation:
    def test_both_children_receive_the_joined_columns(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        link = Link.inner(
            left=JoinSpec(SwappedParentLeft, Index((SwappedParentLeft.get_class_name(),))),
            right=JoinSpec(SwappedParentRight, Index((SwappedParentRight.get_class_name(),))),
        )

        result = mloda.run_all(
            [
                Feature(name=SwappedChildPandasLeft.get_class_name()),
                Feature(name=SwappedChildPyArrowLeft.get_class_name()),
            ],
            links={link},
            compute_frameworks=["PyArrowTable", "PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    SwappedParentLeft,
                    SwappedParentRight,
                    SwappedChildPandasLeft,
                    SwappedChildPyArrowLeft,
                }
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )

        joined: dict[str, list[Any]] = {}
        for res in result:
            for name in (SwappedChildPandasLeft.get_class_name(), SwappedChildPyArrowLeft.get_class_name()):
                if name in _column_names(res):
                    joined[name] = _values(res, name)

        assert sorted(joined) == [
            SwappedChildPandasLeft.get_class_name(),
            SwappedChildPyArrowLeft.get_class_name(),
        ], f"both children must produce a result, got: {sorted(joined)}"
        for name, values in joined.items():
            assert values == [JOINED_VALUE], f"{name} did not see both parent payloads, got: {values}"
