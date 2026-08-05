"""Regression tests: SingleFilter.name must follow a set_feature_name rename of its filter feature.

A name snapshotted at construction makes the filter engines target a column that no longer exists.
"""

from typing import Any

from mloda.core.filter.single_filter import SingleFilter
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Features, GlobalFilter, Options, ParallelizationMode, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_core.test_tooling import MlodaTestRunner


class RnfilRenamingFG(FeatureGroup):
    """Root group that renames the raw filter column via set_feature_name."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"rnfil_value", "rnfil_event_time"})

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        if str(feature_name) == "rnfil_event_time":
            return FeatureName("rnfil_event_time_utc")
        return feature_name

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rnfil_value": [1, 2, 3], "rnfil_event_time_utc": [10, 20, 30]}


_ENABLED = PluginCollector.enabled_feature_groups({RnfilRenamingFG})


def test_filter_on_renamed_filter_feature_filters_rows() -> None:
    """A min filter on the pre-rename name must apply to the renamed column."""
    features = Features([Feature("rnfil_value")])

    global_filter = GlobalFilter()
    global_filter.add_filter("rnfil_event_time", "min", {"value": 15})

    result = MlodaTestRunner.run_api(
        features,
        compute_frameworks={PythonDictFramework},
        parallelization_modes={ParallelizationMode.SYNC},
        global_filter=global_filter,
        plugin_collector=_ENABLED,
    )

    assert len(result.results) == 1
    assert result.results[0]["rnfil_value"] == [2, 3]


def test_single_filter_name_tracks_filter_feature_rename() -> None:
    """SingleFilter.name must reflect a later rename of filter_feature.name."""
    single_filter = SingleFilter("rnfil_track_col", "min", {"value": 1})

    single_filter.filter_feature.name = FeatureName("rnfil_track_col_utc")

    assert single_filter.name == "rnfil_track_col_utc"
