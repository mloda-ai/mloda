"""Tests for the automatic dependency resolution example in README.md.

These tests validate the runnable code example that demonstrates mloda's
automatic dependency resolution: requesting a single high-level feature
and letting mloda resolve all intermediate and raw dependencies.

Dependency tree:
    risk_assessment               (top-level, requested by user)
    |-- debt_to_income            (intermediate, auto-resolved)
    |   |-- debt                  (raw data, auto-resolved)
    |   |-- income                (raw data, auto-resolved)
    |-- age                       (raw data, auto-resolved)
    |-- employment_years          (raw data, auto-resolved)
    |-- credit_score              (raw data, auto-resolved)
"""

from typing import Any, Optional

import pandas as pd

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class CustomerData(FeatureGroup):
    """Raw data source providing 5 columns."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"income", "age", "employment_years", "debt", "credit_score"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame(
            {
                "income": [50000, 100000, 40000],
                "age": [25, 45, 35],
                "employment_years": [2, 20, 8],
                "debt": [10000, 10000, 20000],
                "credit_score": [680, 720, 640],
            }
        )

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class DebtToIncome(FeatureGroup):
    """Intermediate feature: debt / income ratio."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"debt_to_income"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("debt"), Feature("income")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["debt_to_income"] = data["debt"] / data["income"]
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class RiskAssessment(FeatureGroup):
    """Top-level feature combining 4 inputs (one derived, three raw)."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"risk_assessment"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("debt_to_income"), Feature("age"), Feature("employment_years"), Feature("credit_score")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["risk_assessment"] = data["credit_score"] - data["debt_to_income"] * 100 + data["employment_years"] * 2
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


_PLUGIN_COLLECTOR = PluginCollector.enabled_feature_groups({CustomerData, DebtToIncome, RiskAssessment})


class TestDependencyResolutionExample:
    """Validate the automatic dependency resolution example from README.md."""

    def test_full_dependency_tree(self) -> None:
        """Request only risk_assessment; mloda resolves all 5 raw dependencies."""
        result = mloda.run_all(
            features=[Feature(name="risk_assessment")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_PLUGIN_COLLECTOR,
        )

        assert len(result) == 1
        df = result[0]
        assert "risk_assessment" in df.columns
        # Row 0: 680 - (10000/50000)*100 + 2*2 = 680 - 20 + 4 = 664.0
        # Row 1: 720 - (10000/100000)*100 + 20*2 = 720 - 10 + 40 = 750.0
        # Row 2: 640 - (20000/40000)*100 + 8*2 = 640 - 50 + 16 = 606.0
        assert list(df["risk_assessment"]) == [664.0, 750.0, 606.0]

    def test_intermediate_feature_debt_to_income(self) -> None:
        """Request the intermediate debt_to_income; mloda resolves its 2 raw dependencies."""
        result = mloda.run_all(
            features=[Feature(name="debt_to_income")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_PLUGIN_COLLECTOR,
        )

        assert len(result) == 1
        df = result[0]
        assert list(df["debt_to_income"]) == [0.2, 0.1, 0.5]

    def test_raw_feature_directly(self) -> None:
        """Request a raw feature; no dependency resolution needed."""
        result = mloda.run_all(
            features=[Feature(name="income")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_PLUGIN_COLLECTOR,
        )

        assert len(result) == 1
        df = result[0]
        assert list(df["income"]) == [50000, 100000, 40000]
