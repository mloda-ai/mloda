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


# ---------------------------------------------------------------------------
# Behavior B: a scoped join key rides along in the same source read as its
# sibling columns rather than triggering a second computation of the source.
# ---------------------------------------------------------------------------


class Source508CounterA(FeatureGroup):
    """Source A with a class-level call counter, providing subject_token + counter_value_a."""

    calls = 0

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"subject_token", "counter_value_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls += 1
        return {
            "subject_token": ["s1", "s2", "s1", "s2"],
            "counter_value_a": [10, 5, 30, 7],
        }


class Source508CounterB(FeatureGroup):
    """Source B, providing the shared subject_token plus counter_value_b (creates ambiguity)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"subject_token", "counter_value_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "subject_token": ["s1", "s2"],
            "counter_value_b": [1, 2],
        }


class Derived508CounterScoped(FeatureGroup):
    """Derived FG: reads counter_value_a and subject_token scoped to Source A."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature("counter_value_a"),
            Feature("subject_token", feature_group=Source508CounterA),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        grouped = data.groupby("subject_token")["counter_value_a"].max()
        result = sorted(f"{key}|{value}" for key, value in grouped.items())
        return {cls.get_class_name(): result}


def test_scoped_key_shares_read_with_siblings(flight_server: Any) -> None:
    """The scoped join key batches into the SAME source read as its sibling column.

    counter_value_a and the subject_token scoped to Source A both resolve to
    Source A. Because scope is excluded from the similarity hash, they share a
    single read: Source A's calculate_feature runs exactly once, not twice.
    """
    Source508CounterA.calls = 0

    feature = Feature(name=Derived508CounterScoped.get_class_name())

    result = mloda.run_all(
        [feature],
        compute_frameworks=["PandasDataFrame"],
        plugin_collector=PluginCollector.enabled_feature_groups(
            {Source508CounterA, Source508CounterB, Derived508CounterScoped}
        ),
        flight_server=flight_server,
        parallelization_modes={ParallelizationMode.SYNC},
    )

    values: list[str] = []
    for res in result:
        values = list(res[Derived508CounterScoped.get_class_name()].values)

    assert values == ["s1|30", "s2|7"]
    assert Source508CounterA.calls == 1


# ---------------------------------------------------------------------------
# Behavior C: same-name-different-scope keys in DIFFERENT derived feature groups
# resolve independently to their own source without colliding.
# ---------------------------------------------------------------------------


class Source508IndepA(FeatureGroup):
    """Source A: provides subject_token plus value_a."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"subject_token", "value_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "subject_token": ["s1", "s2", "s1", "s2"],
            "value_a": [10, 5, 30, 7],
        }


class Source508IndepB(FeatureGroup):
    """Source B: provides subject_token plus value_b."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"subject_token", "value_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "subject_token": ["s1", "s2", "s1", "s2"],
            "value_b": [100, 200, 300, 400],
        }


class Derived508IndepA(FeatureGroup):
    """Derived FG A: reads value_a and subject_token scoped to Source A."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature("value_a"),
            Feature("subject_token", feature_group=Source508IndepA),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        grouped = data.groupby("subject_token")["value_a"].max()
        result = sorted(f"{key}|{value}" for key, value in grouped.items())
        return {cls.get_class_name(): result}


class Derived508IndepB(FeatureGroup):
    """Derived FG B: reads value_b and subject_token scoped to Source B."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature("value_b"),
            Feature("subject_token", feature_group=Source508IndepB),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        grouped = data.groupby("subject_token")["value_b"].max()
        result = sorted(f"{key}|{value}" for key, value in grouped.items())
        return {cls.get_class_name(): result}


def _column_from_result(result: Any, column: str) -> list[Any]:
    for res in result:
        if column in res:
            return list(res[column].values)
    return []


def test_two_derived_fgs_scope_same_key_to_different_sources(flight_server: Any) -> None:
    """Two derived FGs scope the same-name key to different sources and resolve independently.

    FG_A pulls subject_token from Source A (alongside value_a); FG_B pulls the
    same-named subject_token from Source B (alongside value_b). Requested together
    in one run, each computes correctly from its own source, proving that
    same-name/different-scope features in different FGs do not collide.
    """
    features: list[Feature | str] = [
        Feature(name=Derived508IndepA.get_class_name()),
        Feature(name=Derived508IndepB.get_class_name()),
    ]

    result = mloda.run_all(
        features,
        compute_frameworks=["PandasDataFrame"],
        plugin_collector=PluginCollector.enabled_feature_groups(
            {Source508IndepA, Source508IndepB, Derived508IndepA, Derived508IndepB}
        ),
        flight_server=flight_server,
        parallelization_modes={ParallelizationMode.SYNC},
    )

    values_a = _column_from_result(result, Derived508IndepA.get_class_name())
    values_b = _column_from_result(result, Derived508IndepB.get_class_name())

    assert values_a == ["s1|30", "s2|7"]
    assert values_b == ["s1|300", "s2|400"]


# ---------------------------------------------------------------------------
# Behavior A at the real entry path: two same-name features with different
# scopes in a single top-level request are rejected as duplicates.
# ---------------------------------------------------------------------------


def test_same_name_two_scopes_top_level_request_raises(flight_server: Any) -> None:
    """Two same-name/different-scope features in one top-level request are rejected.

    Scope is excluded from identity, so the two features compare equal and the
    request-level Features validation raises "Duplicate feature setup".
    """
    features: list[Feature | str] = [
        Feature("subject_token", feature_group=Source508A),
        Feature("subject_token", feature_group=Source508B),
    ]

    with pytest.raises(ValueError, match="Duplicate feature setup"):
        mloda.run_all(
            features,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups({Source508A, Source508B}),
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
        )
