"""
Integration tests for the AggregatedFeatureGroup with mloda.
"""

import gc
from collections.abc import Iterator
from typing import Any

import pytest

from mloda.provider import ComputeFramework, FeatureSet
from mloda.user import mloda
from mloda.user import Feature
from mloda.user import Options
from mloda.user import PluginCollector, PluginLoader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregation
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregation  # noqa: F401
from mloda.provider import DefaultOptionKeys

from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import column_values
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


@pytest.fixture(autouse=True, scope="module")
def _collect_local_classes() -> Iterator[None]:
    PluginLoader.all()
    yield
    gc.collect()


class AggregatedParserTestDataCreator(ATestDataCreator):
    """Test data creator for aggregation parser tests."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "Sales": [100, 200, 300, 400, 500],
            "Revenue": [1000, 2000, 3000, 4000, 5000],
        }


class AggregatedNanSourceCreator(ATestDataCreator):
    """Serves a float column holding a NaN on both Pandas and PyArrow, so the aggregation framework is free."""

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return {"aggr_nan_x": [1.0, float("nan"), 3.0]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame, PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls.get_raw_data()


class AggregatedNanPandasOnlySourceCreator(AggregatedNanSourceCreator):
    """Serves a float column holding a NaN on Pandas only, so PyArrow needs a transfer."""

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return {"aggr_nan_p": [1.0, float("nan"), 3.0]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


_NAN_SUM = "aggr_nan_x__sum_aggr"
_PANDAS_ONLY_SUM = "aggr_nan_p__sum_aggr"
_PLUGINS_OF = {
    _NAN_SUM: PluginCollector.enabled_feature_groups({AggregatedNanSourceCreator, AggregatedFeatureGroup}),
    _PANDAS_ONLY_SUM: PluginCollector.enabled_feature_groups(
        {AggregatedNanPandasOnlySourceCreator, AggregatedFeatureGroup}
    ),
}
_NAN_PLUGINS = _PLUGINS_OF[_NAN_SUM]

# No aggregation implementation diverges: null and NaN are both missing, so every path sums to 4.0.
_NAN_SUM_CASES = [
    pytest.param(
        _NAN_SUM,
        [PandasDataFrame, PyArrowTable],
        {},
        "PandasDataFrame",
        "your list order",
        id="unpinned_pandas_listed_first",
    ),
    pytest.param(
        _NAN_SUM,
        [PyArrowTable, PandasDataFrame],
        {},
        "PyArrowTable",
        "your list order",
        id="unpinned_pyarrow_listed_first",
    ),
    pytest.param(_NAN_SUM, [PandasDataFrame], {}, "PandasDataFrame", "only allowed framework", id="pandas_alone"),
    pytest.param(
        _NAN_SUM,
        [PandasDataFrame, PyArrowTable],
        {"compute_framework": "PandasDataFrame"},
        "PandasDataFrame",
        "pinned",
        id="framework_pin",
    ),
    pytest.param(
        _NAN_SUM,
        [PandasDataFrame, PyArrowTable],
        {"compute_framework": "PyArrowTable"},
        "PyArrowTable",
        "pinned",
        id="pyarrow_framework_pin",
    ),
    pytest.param(
        _NAN_SUM,
        [PandasDataFrame, PyArrowTable],
        {"implementation": "PandasAggregation"},
        "PandasDataFrame",
        "pinned",
        id="implementation_pin",
    ),
    pytest.param(
        _NAN_SUM,
        [PandasDataFrame, PyArrowTable],
        {"implementation": "PyArrowAggregation"},
        "PyArrowTable",
        "pinned",
        id="pyarrow_implementation_pin",
    ),
    pytest.param(
        _PANDAS_ONLY_SUM,
        [PandasDataFrame, PyArrowTable],
        {},
        "PandasDataFrame",
        "your list order",
        id="pandas_only_source_unpinned",
    ),
    pytest.param(
        _PANDAS_ONLY_SUM,
        [PandasDataFrame, PyArrowTable],
        {"implementation": "PyArrowAggregation"},
        "PyArrowTable",
        "pinned",
        id="pandas_only_source_pyarrow_pin",
    ),
]


class TestAggregationImplementationChoice:
    """One AggregatedFeatureGroup, one implementation per framework: all agree on a NaN."""

    @pytest.mark.parametrize(("name", "frameworks", "pin", "framework", "reason"), _NAN_SUM_CASES)
    def test_sum_over_a_nan_runs_the_chosen_implementation_with_its_reason(
        self,
        name: str,
        frameworks: list[type[ComputeFramework]],
        pin: dict[str, Any],
        framework: str,
        reason: str,
    ) -> None:
        feature = Feature(name, **pin)
        plugins = _PLUGINS_OF[name]

        steps = mloda.explain([feature], compute_frameworks=frameworks, plugin_collector=plugins)
        step = next(s for s in steps if s.step_kind == "compute" and s.feature_group is AggregatedFeatureGroup)
        assert step.compute_framework_name == framework
        assert step.compute_framework_reason == reason

        results = mloda.run_all([feature], compute_frameworks=frameworks, plugin_collector=plugins)
        assert column_values(results, name) == [4.0] * 3

    def test_the_pinned_implementation_is_the_one_the_plan_reports(self) -> None:
        feature = Feature(_NAN_SUM, implementation="PandasAggregation")

        steps = mloda.explain(
            [feature], compute_frameworks=[PandasDataFrame, PyArrowTable], plugin_collector=_NAN_PLUGINS
        )
        step = next(s for s in steps if s.feature_group is AggregatedFeatureGroup)

        assert step.implementation is PandasAggregation


class TestAggregatedFeatureGroupIntegration:
    def test_integration_with_feature_parser(self) -> None:
        """Test integration with mloda using the parser."""
        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {AggregatedParserTestDataCreator, AggregatedFeatureGroup}
        )

        f1 = Feature(
            "sum_sales",
            Options(
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "sum",
                    DefaultOptionKeys.in_features: "Sales",
                }
            ),
        )

        f2 = Feature(
            "avg_revenue",
            Options(
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "avg",
                    DefaultOptionKeys.in_features: "Revenue",
                }
            ),
        )

        # test with pre parsing the features
        results = mloda.run_all(
            [f1, f2],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1

        # Find the DataFrame with the aggregated feature
        agg_df = None
        for df in results:
            if "avg_revenue" in df.columns and "sum_sales" in df.columns:
                agg_df = df
                break

        assert agg_df is not None, "DataFrame with aggregated feature not found"
        assert agg_df["sum_sales"].iloc[0] == 1500  # Sum of [100, 200, 300, 400, 500]

        # test with mloda parsing the features
        results2 = mloda.run_all(
            [f1, f2],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=plugin_collector,
        )

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))
