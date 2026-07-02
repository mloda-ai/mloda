"""Regression tests: a PythonDict FeatureGroup returning row-wise ``list[dict]`` output
combined with a final/global filter works.

``ComputeFramework.run_calculation`` normalizes the feature output via
``transform``/``validate_native_data`` BEFORE ``run_final_filter``, so the columnar filter
engine and ``_validate_filter_columns`` always see a columnar ``dict[str, list[Any]]``.
"""

from typing import Any, Optional

from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.provider import BaseInputData
from mloda.user import Feature, Features, GlobalFilter, ParallelizationMode, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_core.test_tooling import MlodaTestRunner


class ListDictRootFeatureGroup(FeatureGroup):
    """Root PythonDict FeatureGroup that returns ROW-WISE ``list[dict]`` output.

    ``calculate_feature`` returns ``[{"x": .., "y": ..}, ...]``, a still-accepted transform
    input. A filter on ``x`` works because the output is normalized to columnar first.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"x", "y"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return [
            {"x": 1, "y": 10},
            {"x": 2, "y": 20},
            {"x": 3, "y": 30},
        ]


_ENABLED_LIST_DICT = PluginCollector.enabled_feature_groups({ListDictRootFeatureGroup})


def test_list_dict_output_with_filter_run_all(flight_server: Any) -> None:
    """End-to-end: list[dict] FG output + min filter x >= 2 -> filtered columnar result."""
    features = Features([Feature(name="x"), Feature(name="y")])

    global_filter = GlobalFilter()
    global_filter.add_filter("x", "min", {"value": 2})

    result = MlodaTestRunner.run_api(
        features,
        compute_frameworks={PythonDictFramework},
        parallelization_modes={ParallelizationMode.SYNC},
        flight_server=flight_server,
        global_filter=global_filter,
        plugin_collector=_ENABLED_LIST_DICT,
    )

    assert len(result.results) == 1
    assert result.results[0] == {"x": [2, 3], "y": [20, 30]}


def test_validate_and_apply_filters_on_list_dict_does_not_raise() -> None:
    """Focused: the framework filter path handles a row-wise ``list[dict]`` without raising.

    ``_validate_filter_columns`` followed by ``filter_engine.apply_filters`` on a
    ``list[dict]`` input normalizes to columnar and returns the rows with x >= 2.
    """
    from mloda.core.filter.single_filter import SingleFilter

    fw = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    data: list[dict[str, Any]] = [
        {"x": 1, "y": 10},
        {"x": 2, "y": 20},
        {"x": 3, "y": 30},
    ]

    single_filter = SingleFilter("x", "min", {"value": 2})

    class _FeaturesStub:
        filter_engine = PythonDictFramework.filter_engine()
        filters = {single_filter}

        @staticmethod
        def get_all_names() -> set[str]:
            return {"x", "y"}

    features = _FeaturesStub()

    # The filter path itself also normalizes row-wise input; this must NOT raise.
    fw._validate_filter_columns(data, features, ListDictRootFeatureGroup)

    filtered = features.filter_engine().apply_filters(data, features)

    # After normalization + filtering, the columnar result keeps only x >= 2.
    assert filtered == {"x": [2, 3], "y": [20, 30]}
