"""End-to-end scope test through mloda.run_all (issue #508 definition of done).

A derived feature group needs a shared join key ("subject_token") that TWO
enabled sources both declare. Requesting it by bare name is ambiguous. Scoping
the request to one source (Feature("subject_token", feature_group=SourceA))
resolves it uniquely, so the derived feature group can compute its result.
"""

from typing import Any, Optional

import pytest

from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401


class Source508A(FeatureGroup):
    """Source A: provides the shared "subject_token" plus "scoping_value_a"."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"subject_token", "scoping_value_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "subject_token": ["s1", "s2", "s1", "s2"],
            "scoping_value_a": [10, 5, 30, 7],
        }


class Source508B(FeatureGroup):
    """Source B: also provides the shared "subject_token" plus "scoping_value_b"."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"subject_token", "scoping_value_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "subject_token": ["s1", "s2"],
            "scoping_value_b": [1, 2],
        }


class Derived508Scoped(FeatureGroup):
    """Derived FG: reads scoping_value_a and subject_token scoped to Source A."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature("scoping_value_a"),
            Feature("subject_token", feature_group=Source508A),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        grouped = data.groupby("subject_token")["scoping_value_a"].max()
        result = sorted(f"{key}|{value}" for key, value in grouped.items())
        return {cls.get_class_name(): result}


class Derived508Unscoped(FeatureGroup):
    """Derived FG requesting subject_token WITHOUT scope (ambiguous by design)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature("scoping_value_a"),
            Feature("subject_token"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        grouped = data.groupby("subject_token")["scoping_value_a"].max()
        result = sorted(f"{key}|{value}" for key, value in grouped.items())
        return {cls.get_class_name(): result}


def test_scoped_derived_feature_resolves_and_computes(flight_server: Any) -> None:
    """Scoping subject_token to Source A lets the derived FG compute per-token maxima."""
    feature = Feature(name=Derived508Scoped.get_class_name())

    result = mloda.run_all(
        [feature],
        compute_frameworks=["PandasDataFrame"],
        plugin_collector=PluginCollector.enabled_feature_groups({Source508A, Source508B, Derived508Scoped}),
        flight_server=flight_server,
        parallelization_modes={ParallelizationMode.SYNC},
    )

    values: list[str] = []
    for res in result:
        values = list(res[Derived508Scoped.get_class_name()].values)

    assert values == ["s1|30", "s2|7"]


def test_unscoped_derived_feature_is_ambiguous(flight_server: Any) -> None:
    """Characterization: without scope, subject_token still raises 'Multiple feature groups found'."""
    feature = Feature(name=Derived508Unscoped.get_class_name())

    with pytest.raises(ValueError, match="Multiple feature groups found"):
        mloda.run_all(
            [feature],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups({Source508A, Source508B, Derived508Unscoped}),
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
        )
