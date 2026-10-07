"""
Integration tests for the MissingValueFeatureGroup with mloda.
"""

from typing import Any

import pytest

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.user import mloda
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda.user import Feature
from mloda.user import Options
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueFeatureGroup
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureSet

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class MissingValueParserTestDataCreator(ATestDataCreator):
    """Test data creator for missing value parser tests."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "income": [50000, None, 75000, None, 60000],
            "category": ["A", None, "B", "A", None],
        }


class TestMissingValueFeatureGroupIntegration:
    def test_integration_with_feature_parser(self) -> None:
        """Test integration with mloda using the parser."""
        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {MissingValueParserTestDataCreator, PandasMissingValueFeatureGroup}
        )

        f1 = Feature(
            "x",
            Options(
                context={
                    MissingValueFeatureGroup.IMPUTATION_METHOD: "mean",
                    DefaultOptionKeys.in_features: "income",
                }
            ),
        )

        f2 = Feature(
            "x2",
            Options(
                context={
                    MissingValueFeatureGroup.IMPUTATION_METHOD: "mode",
                    DefaultOptionKeys.in_features: "category",
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

        # Find the DataFrame with the imputed features
        imputed_df = None
        for df in results:
            if "x" in df.columns and "x2" in df.columns:
                imputed_df = df
                break

        assert imputed_df is not None, "DataFrame with imputed features not found"

        # Verify that missing values are imputed
        assert abs(imputed_df["x"].iloc[1] - 61666.67) < 1.0
        assert abs(imputed_df["x"].iloc[3] - 61666.67) < 1.0

        assert imputed_df["x2"].iloc[1] == "A"
        assert imputed_df["x2"].iloc[4] == "A"

        # test with mloda parsing the features
        results2 = mloda.run_all(
            [f1, f2],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=plugin_collector,
        )

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))

    def test_integration_with_constant_imputation(self) -> None:
        """Test integration with mloda using constant imputation."""
        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {MissingValueParserTestDataCreator, PandasMissingValueFeatureGroup}
        )

        # Create a feature with constant imputation
        f1 = Feature(
            "category__constant_imputed",
            Options(
                {
                    MissingValueFeatureGroup.IMPUTATION_METHOD: "constant",
                    DefaultOptionKeys.in_features: "category",
                    "constant_value": "Unknown",
                }
            ),
        )

        # test with pre parsing the features
        results = mloda.run_all([f1], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)

        assert len(results) == 1

        # Find the DataFrame with the imputed features
        imputed_df = None
        for df in results:
            if "category__constant_imputed" in df.columns:
                imputed_df = df
                break

        assert imputed_df is not None, "DataFrame with imputed features not found"

        # Verify that missing values are imputed with the constant value
        assert imputed_df["category__constant_imputed"].iloc[1] == "Unknown"
        assert imputed_df["category__constant_imputed"].iloc[4] == "Unknown"

        # test with mloda parsing the features
        results2 = mloda.run_all([f1], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))


class GroupedMissingValueDataCreator(MissingValueParserTestDataCreator):
    """Root that returns only the requested columns."""

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return {
            "income": [10.0, None, 30.0, 100.0, None],
            "region": ["N", "N", "N", "S", "S"],
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raw = cls.get_raw_data()
        return cls.transform_format_for_testing({n: raw[n] for n in features.get_all_names()})


class TestMissingValueGroupByColumnSelectiveRoot:
    def test_grouped_mean_imputation_behind_column_selective_root(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {GroupedMissingValueDataCreator, PandasMissingValueFeatureGroup}
        )
        f = Feature(
            "income__mean_imputed",
            Options(
                context={
                    MissingValueFeatureGroup.IMPUTATION_METHOD: "mean",
                    DefaultOptionKeys.in_features: "income",
                    "group_by_features": ["region"],
                }
            ),
        )
        results = mloda.run_all([f], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)
        assert len(results) == 1
        assert results[0]["income__mean_imputed"].tolist() == [10.0, 20.0, 30.0, 100.0, 100.0]

    @pytest.mark.parametrize(
        ("source", "root_names", "expected"),
        [
            pytest.param(
                Feature(
                    "income_sum", Options(group={"aggregation_type": "sum", DefaultOptionKeys.in_features: "income"})
                ),
                {"income", "region"},
                [140.0] * 5,
                id="chained_source",
            ),
            pytest.param(
                Feature("income", Options(group={"scope": "a"})),
                {"income", "region"},
                [10.0, 20.0, 30.0, 100.0, 100.0],
                id="scoped_source",
            ),
        ],
    )
    def test_source_shares_one_root_step_with_group_by_column(
        self, source: Feature, root_names: set[str], expected: list[float]
    ) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {GroupedMissingValueDataCreator, PandasMissingValueFeatureGroup, PandasAggregatedFeatureGroup}
        )
        f = Feature(
            "mv",
            Options(
                context={
                    MissingValueFeatureGroup.IMPUTATION_METHOD: "mean",
                    DefaultOptionKeys.in_features: frozenset([source]),
                    "group_by_features": ["region"],
                }
            ),
        )
        session = mloda.prepare([f], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)
        assert session.engine is not None
        root_steps = [
            s
            for s in session.engine.execution_planner
            if isinstance(s, FeatureGroupStep) and s.feature_group is GroupedMissingValueDataCreator
        ]
        assert len(root_steps) == 1
        assert {str(n) for n in root_steps[0].features.get_all_names()} == root_names

        results = session.run()
        assert len(results) == 1
        assert results[0]["mv"].tolist() == expected
