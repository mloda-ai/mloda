"""RED-phase tests for per-feature feature-group scoping (GitHub issue #508).

A derived feature group that reads a subset of one source's columns cannot
obtain the shared join key (e.g. ``subject_token``) when two enabled sources
both declare that column: requesting it by name raises
``ValueError: Multiple feature groups found for feature 'subject_token'``.

Target behavior:
  * ``Feature`` gains an optional ``feature_group`` constructor parameter,
    accepting a feature group class name string or the feature group class
    itself, stored as ``feature_group_class_name`` (``str | None``), with an
    ``options.get("feature_group")`` fallback (mirroring ``domain``).
  * ``feature_group_class_name`` participates in ``Feature.__eq__`` and
    ``Feature.__hash__`` (like ``domain``).
  * ``IdentifyFeatureGroupClass._filter_loop`` filters candidate feature
    groups by that scope BEFORE the "Multiple feature groups found"
    validation, so a scoped feature resolves uniquely.
  * A scope matching nothing raises the existing "No feature groups found"
    ValueError, and its message mentions the requested scope name.

PART A exercises ``IdentifyFeatureGroupClass`` directly.
PART B drives the issue's definition-of-done scenario through
``mloda.run_all``.
"""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
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

# Importing the framework registers it as a ComputeFramework subclass, so
# ``compute_frameworks=["PandasDataFrame"]`` resolves even when this module
# runs in isolation.
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import (  # noqa: F401
    PandasDataFrame,
)


class MockComputeFramework(ComputeFramework):
    """Mock compute framework for direct IdentifyFeatureGroupClass tests."""

    pass


# ============================================================================
# PART A - direct resolution scoping via IdentifyFeatureGroupClass
# ============================================================================
class ScopingSubjectTokenSourceA(FeatureGroup):
    """First source feature group declaring the shared 'subject_token' column."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name in {"subject_token", "scoping_value_a"}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"subject_token", "scoping_value_a"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class ScopingSubjectTokenSourceB(FeatureGroup):
    """Second source feature group also declaring the shared 'subject_token' column."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name in {"subject_token", "scoping_value_b"}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"subject_token", "scoping_value_b"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


def _resolve_with_both_sources(feature: Feature) -> IdentifyFeatureGroupClass:
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopingSubjectTokenSourceA: {MockComputeFramework},
        ScopingSubjectTokenSourceB: {MockComputeFramework},
    }
    return IdentifyFeatureGroupClass(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )


class TestFeatureGroupScopeResolution:
    """Resolution scoping via the new Feature ``feature_group`` parameter."""

    def test_unscoped_shared_column_raises_multiple_found(self) -> None:
        """Characterization test: this is the CURRENT behavior and must stay.

        An unscoped feature whose name is declared by two enabled feature
        groups is ambiguous and raises 'Multiple feature groups found'.
        """
        feature = Feature("subject_token")

        with pytest.raises(ValueError, match="Multiple feature groups found"):
            _resolve_with_both_sources(feature)

    def test_scope_by_class_name_string_resolves_uniquely(self) -> None:
        """A feature scoped via a class name string resolves to exactly that class."""
        feature = Feature("subject_token", feature_group="ScopingSubjectTokenSourceA")

        resolved = _resolve_with_both_sources(feature)

        feature_group_class, compute_frameworks = resolved.get()
        assert feature_group_class is ScopingSubjectTokenSourceA
        assert compute_frameworks == {MockComputeFramework}

    def test_scope_by_class_object_resolves_uniquely(self) -> None:
        """A feature scoped via the feature group class itself resolves to that class.

        The class is stored by its resolved name string.
        """
        feature = Feature("subject_token", feature_group=ScopingSubjectTokenSourceA)

        assert feature.feature_group_class_name == "ScopingSubjectTokenSourceA"

        resolved = _resolve_with_both_sources(feature)

        feature_group_class, _ = resolved.get()
        assert feature_group_class is ScopingSubjectTokenSourceA

    def test_scope_via_options_resolves_uniquely(self) -> None:
        """The scope can also be provided via options, mirroring 'domain'."""
        feature = Feature("subject_token", options={"feature_group": "ScopingSubjectTokenSourceB"})

        assert feature.feature_group_class_name == "ScopingSubjectTokenSourceB"

        resolved = _resolve_with_both_sources(feature)

        feature_group_class, _ = resolved.get()
        assert feature_group_class is ScopingSubjectTokenSourceB

    def test_scope_matching_nothing_raises_with_scope_name_in_message(self) -> None:
        """A scope matching no candidate raises 'No feature groups found' naming the scope."""
        feature = Feature("subject_token", feature_group="CompletelyUnknownScopedFeatureGroup")

        with pytest.raises(ValueError, match="No feature groups found") as exc_info:
            _resolve_with_both_sources(feature)

        assert "CompletelyUnknownScopedFeatureGroup" in str(exc_info.value)


class TestFeatureEqualityWithScope:
    """``feature_group_class_name`` participates in Feature equality and hash."""

    def test_different_scopes_are_not_equal_and_hash_differently(self) -> None:
        scoped_to_a = Feature("subject_token", feature_group="ScopingSubjectTokenSourceA")
        scoped_to_b = Feature("subject_token", feature_group="ScopingSubjectTokenSourceB")

        assert scoped_to_a != scoped_to_b
        assert hash(scoped_to_a) != hash(scoped_to_b)

    def test_scoped_and_unscoped_are_not_equal(self) -> None:
        scoped = Feature("subject_token", feature_group="ScopingSubjectTokenSourceA")
        unscoped = Feature("subject_token")

        assert scoped != unscoped

    def test_same_scope_is_equal_and_hashes_equally(self) -> None:
        first = Feature("subject_token", feature_group="ScopingSubjectTokenSourceA")
        second = Feature("subject_token", feature_group="ScopingSubjectTokenSourceA")

        assert first == second
        assert hash(first) == hash(second)


class TestEmptyStringScopeNormalization:
    """An empty-string scope means 'no scope', mirroring ``Feature(domain="")``."""

    def test_empty_string_parameter_normalizes_to_none(self) -> None:
        feature = Feature("x", feature_group="")

        assert feature.feature_group_class_name is None

    def test_empty_string_in_options_normalizes_to_none(self) -> None:
        feature = Feature("x", options={"feature_group": ""})

        assert feature.feature_group_class_name is None


class TestInvalidScopeTypeRaisesTypeError:
    """A scope that is neither a string nor a feature group class raises a clear TypeError."""

    def test_int_scope_raises_type_error_mentioning_feature_group(self) -> None:
        with pytest.raises(TypeError, match="feature_group"):
            Feature("x", feature_group=123)  # type: ignore[arg-type]

    def test_arbitrary_instance_scope_raises_type_error_mentioning_feature_group(self) -> None:
        with pytest.raises(TypeError, match="feature_group"):
            Feature("x", feature_group=object())  # type: ignore[arg-type]


# ============================================================================
# PART B - integration through mloda.run_all (issue #508 definition of done)
# ============================================================================
# Two enabled sources both produce the shared join key 'subject_token'.
#   source A: subject_token = [s1, s2, s1, s2], scoping_value_a = [10, 5, 30, 7]
#   source B: subject_token = [s1, s2],         scoping_value_b = [1, 2]
# The derived feature group reads 'scoping_value_a' AND the shared
# 'subject_token' from source A, computing the max value per subject:
#   s1 -> 30, s2 -> 7   encoded as ["s1|30", "s2|7"]
class ScopingRunSourceA(FeatureGroup):
    """Source A: emits the shared 'subject_token' key and 'scoping_value_a'."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"subject_token", "scoping_value_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "subject_token": ["s1", "s2", "s1", "s2"],
            "scoping_value_a": [10, 5, 30, 7],
        }


class ScopingRunSourceB(FeatureGroup):
    """Source B: also emits the shared 'subject_token' key, plus 'scoping_value_b'."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"subject_token", "scoping_value_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            "subject_token": ["s1", "s2"],
            "scoping_value_b": [1, 2],
        }


class ScopingMaxValuePerSubject(FeatureGroup):
    """Derived FG reading source A columns; the shared key is scoped to source A."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name="scoping_value_a"),
            Feature(name="subject_token", feature_group=ScopingRunSourceA),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        maxima: dict[str, int] = {}
        for token, value in zip(data["subject_token"], data["scoping_value_a"]):
            if token not in maxima or value > maxima[token]:
                maxima[token] = value
        return {cls.get_class_name(): sorted(f"{token}|{value}" for token, value in maxima.items())}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


class ScopingUnscopedMaxValuePerSubject(FeatureGroup):
    """Derived FG requesting the shared key WITHOUT a scope: stays ambiguous."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name="scoping_value_a"),
            Feature(name="subject_token"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): []}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_ENABLED_SCOPED = PluginCollector.enabled_feature_groups(
    {ScopingRunSourceA, ScopingRunSourceB, ScopingMaxValuePerSubject}
)
_ENABLED_UNSCOPED = PluginCollector.enabled_feature_groups(
    {ScopingRunSourceA, ScopingRunSourceB, ScopingUnscopedMaxValuePerSubject}
)


class TestScopedSharedColumnRunAll:
    """Issue #508 definition of done, driven through mloda.run_all."""

    def test_unscoped_shared_column_raises_multiple_found(self, flight_server: Any) -> None:
        """Characterization test: without a scope the shared key stays ambiguous.

        This must keep raising 'Multiple feature groups found' even after the
        fix, because no per-feature scope is given.
        """
        with pytest.raises(ValueError, match="Multiple feature groups found"):
            mloda.run_all(
                [Feature(name=ScopingUnscopedMaxValuePerSubject.get_class_name())],
                compute_frameworks=["PandasDataFrame"],
                plugin_collector=_ENABLED_UNSCOPED,
                flight_server=flight_server,
                parallelization_modes={ParallelizationMode.SYNC},
            )

    def test_scoped_shared_column_computes_per_subject(self, flight_server: Any) -> None:
        """Scoping the shared key to source A resolves it and computes per subject."""
        result = mloda.run_all(
            [Feature(name=ScopingMaxValuePerSubject.get_class_name())],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_ENABLED_SCOPED,
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert len(result) == 1
        values = sorted(str(v) for v in result[0][ScopingMaxValuePerSubject.get_class_name()])
        assert values == ["s1|30", "s2|7"]
