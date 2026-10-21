"""
Integration tests for the AggregatedFeatureGroup with mloda.
"""

import gc
import math
from collections.abc import Iterator
from typing import Any

import pytest

from mloda.provider import ComputeFramework, FeatureSet
from mloda.user import mloda
from mloda.user import Feature
from mloda.user import Options
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregation
from mloda.provider import DefaultOptionKeys

from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import column_values
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


@pytest.fixture(autouse=True, scope="module")
def _collect_local_classes() -> Iterator[None]:
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


_NAN_SUM = "aggr_nan_x__sum_aggr"
_NAN_PLUGINS = PluginCollector.enabled_feature_groups({AggregatedNanSourceCreator, AggregatedFeatureGroup})
_DIVERGENT_PANDAS = "runs a version that differs: skips NaN in sum"

_NAN_SUM_CASES = [
    pytest.param(
        [PandasDataFrame, PyArrowTable],
        {},
        "PyArrowTable",
        "prefers a matching version",
        float("nan"),
        id="matching_implementation_wins_over_the_listed_first_divergent_one",
    ),
    pytest.param(
        [PandasDataFrame],
        {},
        "PandasDataFrame",
        f"only allowed framework; {_DIVERGENT_PANDAS}",
        4.0,
        id="pandas_alone",
    ),
    pytest.param(
        [PandasDataFrame, PyArrowTable],
        {"compute_framework": "PandasDataFrame"},
        "PandasDataFrame",
        f"pinned; {_DIVERGENT_PANDAS}",
        4.0,
        id="framework_pin",
    ),
    pytest.param(
        [PandasDataFrame, PyArrowTable],
        {"implementation": "PandasAggregation"},
        "PandasDataFrame",
        f"pinned; {_DIVERGENT_PANDAS}",
        4.0,
        id="implementation_pin",
    ),
]


class TestAggregationImplementationChoice:
    """One AggregatedFeatureGroup, one implementation per framework: the matching one is preferred."""

    @pytest.mark.parametrize(("frameworks", "pin", "framework", "reason", "expected"), _NAN_SUM_CASES)
    def test_sum_over_a_nan_runs_the_chosen_implementation_with_its_reason(
        self,
        frameworks: list[type[ComputeFramework]],
        pin: dict[str, Any],
        framework: str,
        reason: str,
        expected: float,
    ) -> None:
        feature = Feature(_NAN_SUM, **pin)

        steps = mloda.explain([feature], compute_frameworks=frameworks, plugin_collector=_NAN_PLUGINS)
        step = next(s for s in steps if s.step_kind == "compute" and s.feature_group is AggregatedFeatureGroup)
        assert step.compute_framework_name == framework
        assert step.compute_framework_reason == reason

        results = mloda.run_all([feature], compute_frameworks=frameworks, plugin_collector=_NAN_PLUGINS)
        values = column_values(results, _NAN_SUM)
        assert len(values) == 3
        if math.isnan(expected):
            assert all(math.isnan(value) for value in values)
        else:
            assert values == [expected] * 3

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
