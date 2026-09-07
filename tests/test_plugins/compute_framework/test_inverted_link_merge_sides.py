"""Inverted link orientation: the left feature group's data must stay the left merge argument.

Asymmetric key sets and differing index column names make a swapped merge observable.
"""

from typing import Any

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


class AsymLeftSource(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"asym_left_key", "asym_left_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"asym_left_key": [4, 3, 2, 1], "asym_left_payload": ["l4", "l3", "l2", "l1"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class AsymRightSource(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"asym_right_key", "asym_right_payload"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"asym_right_key": [3, 4, 5], "asym_right_payload": ["r3", "r4", "r5"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class AsymJoinChild(FeatureGroup):
    """Only the right framework is supported, which inverts the declared link orientation."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name="asym_left_key"),
            Feature(name="asym_left_payload"),
            Feature(name="asym_right_key"),
            Feature(name="asym_right_payload"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Reading the left key column proves the left group's index drove the merge.
        data[cls.get_class_name()] = (
            data["asym_left_key"].astype(str) + ":" + data["asym_left_payload"] + data["asym_right_payload"]
        )
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


@pytest.mark.parametrize("modes", [({ParallelizationMode.SYNC}), ({ParallelizationMode.THREADING})])
def test_inverted_link_keeps_left_group_as_left_merge_side(modes: set[ParallelizationMode], flight_server: Any) -> None:
    link = Link.inner(
        left=JoinSpec(AsymLeftSource, Index(("asym_left_key",))),
        right=JoinSpec(AsymRightSource, Index(("asym_right_key",))),
    )

    result = mloda.run_all(
        [Feature(name=AsymJoinChild.get_class_name())],
        links={link},
        compute_frameworks=["PyArrowTable", "PandasDataFrame"],
        plugin_collector=PluginCollector.enabled_feature_groups({AsymLeftSource, AsymRightSource, AsymJoinChild}),
        flight_server=flight_server,
        parallelization_modes=modes,
    )

    joined = [res for res in result if AsymJoinChild.get_class_name() in list(res.columns)]
    assert len(joined) == 1
    data = joined[0]

    assert len(data) == 2
    # Left key order [4, 3] is kept, so the left group's data was the left merge argument.
    assert list(data[AsymJoinChild.get_class_name()]) == ["4:l4r4", "3:l3r3"]
