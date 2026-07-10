"""Regression tests for issue #646: an abstract FeatureGroup base must never win resolution.

An abstract (uninstantiable) FeatureGroup base class can currently win feature
matching and then crash the engine with a raw ``TypeError`` at instantiation,
once all of its concrete subclasses are filtered out by compute-framework
support. Resolution must instead reject the abstract base and raise the regular
unresolvable-feature ``ValueError``.

These tests are written to FAIL until the abstract-base guard exists.
"""

from abc import abstractmethod
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda.core.api.request import mlodaAPI
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


ABSTRACT_WINNER_FEATURE = "abstract_winner_test_feature"

# Unique name so no other test double in the parallel suite can collide on it.
ISOLATED_ABSTRACT_AGG_FEATURE = "qty__isolated_abstract_agg_test_646"


class MockComputeFramework(ComputeFramework):
    """Mock compute framework used to keep the abstract candidate accessible."""

    pass


class AbstractWinnerFeatureGroup(FeatureGroup):
    """An abstract FeatureGroup that matches ``ABSTRACT_WINNER_FEATURE``.

    It declares an unimplemented abstract method, so it cannot be instantiated.
    It matches the feature name via ``match_feature_group_criteria`` (never
    instantiating itself during matching), mimicking the real
    ``AggregatedFeatureGroup`` base that survives resolution once its concrete
    subclasses are dropped.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return name == ABSTRACT_WINNER_FEATURE

    @classmethod
    @abstractmethod
    def _perform_the_work(cls, data: Any) -> Any:
        """Abstract hook that makes this class uninstantiable."""
        ...

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class IsolatedAbstractAggBase(FeatureGroup):
    """Abstract base matching a UNIQUE feature name, unrestricted on frameworks.

    Declares an unimplemented abstract method so it cannot be instantiated. It is
    a plausible name-match for ``ISOLATED_ABSTRACT_AGG_FEATURE`` and supports all
    frameworks (including ``PythonDictFramework``), mirroring the real abstract
    ``AggregatedFeatureGroup`` base from issue #646.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return name == ISOLATED_ABSTRACT_AGG_FEATURE

    @classmethod
    @abstractmethod
    def _isolated_hook(cls, data: Any) -> Any:
        """Abstract hook that makes this base uninstantiable."""
        ...

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class IsolatedConcreteAgg(IsolatedAbstractAggBase):
    """Concrete subclass pinned to ``PandasDataFrame``.

    Restricting a run to ``PythonDictFramework`` filters this subclass out,
    leaving only the abstract base as the sole name-match.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _isolated_hook(cls, data: Any) -> Any:
        return data


# --- Issue #646 diagnostic-message regression fixtures -------------------------

# Finding A: scope narrows the abstract hierarchy away; the frameworks are available.
SCOPE_MISMATCH_FEATURE = "scope_mismatch_abstract_test_646"

# Finding B: a concrete subclass matched but rejected the framework via capability hook.
CAPABILITY_SHADOW_FEATURE = "capability_shadowed_abstract_test_646"


class ScopeMismatchUnrelatedFG(FeatureGroup):
    """Unrelated feature group used only as a scope target; never matches the feature."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ScopeMismatchAbstractBase(FeatureGroup):
    """Abstract base matching ``SCOPE_MISMATCH_FEATURE``; uninstantiable via abstract hook."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return name == SCOPE_MISMATCH_FEATURE

    @classmethod
    @abstractmethod
    def _scope_hook(cls, data: Any) -> Any:
        """Abstract hook that makes this base uninstantiable."""
        ...

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ScopeMismatchConcrete(ScopeMismatchAbstractBase):
    """Concrete subclass declaring an AVAILABLE framework (``MockComputeFramework``).

    A scope pinned to ``ScopeMismatchUnrelatedFG`` filters this subclass out, so the
    hierarchy is narrowed away by SCOPE, not by a missing/unavailable framework.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {MockComputeFramework}

    @classmethod
    def _scope_hook(cls, data: Any) -> Any:
        return data


class CapabilityShadowAbstractBase(FeatureGroup):
    """Abstract base matching ``CAPABILITY_SHADOW_FEATURE``; uninstantiable via abstract hook."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return name == CAPABILITY_SHADOW_FEATURE

    @classmethod
    @abstractmethod
    def _capability_hook(cls, data: Any) -> Any:
        """Abstract hook that makes this base uninstantiable."""
        ...

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class CapabilityShadowConcrete(CapabilityShadowAbstractBase):
    """Concrete subclass whose declared framework IS available but is rejected by the
    capability hook: ``supports_compute_framework`` returns False. This is a capability
    rejection, NOT a missing/unavailable framework.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {MockComputeFramework}

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return False

    @classmethod
    def _capability_hook(cls, data: Any) -> Any:
        return data


class TestAbstractOnlyMessageCorrectness:
    """Issue #646 regression: the abstract-only diagnostic must not shadow the real reason.

    The abstract skip is recorded BEFORE the domain/scope filters and BEFORE the
    capability-rejection check, so two adjacent failure modes currently emit the
    factually wrong 'require compute framework(s) [...], none of which are available
    or enabled' message even though the frameworks ARE available.
    """

    def test_scope_mismatch_not_reported_as_missing_framework(self) -> None:
        """Finding A: scoping the request to an unrelated feature group narrows the
        abstract hierarchy away. The concrete subclass declares an AVAILABLE framework,
        so the real reason is the scope mismatch, NOT a missing framework. The error
        must not falsely claim the frameworks are 'none of which are available or enabled'.
        """
        feature = Feature(SCOPE_MISMATCH_FEATURE, feature_group=ScopeMismatchUnrelatedFG)

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            ScopeMismatchAbstractBase: {MockComputeFramework},
            ScopeMismatchConcrete: {MockComputeFramework},
            ScopeMismatchUnrelatedFG: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)

        assert "none of which are available or enabled" not in error_message, (
            f"Scope mismatch must not be misreported as a missing-framework error; the frameworks "
            f"ARE available and the real reason is the scope narrowing, but got: {error_message}"
        )

    def test_capability_rejection_not_shadowed_by_abstract_only(self) -> None:
        """Finding B: a concrete subclass matched criteria/domain/scope but its available
        framework was rejected by ``supports_compute_framework`` (a capability rejection).
        The user must get the capability-rejection message, NOT the abstract-only message
        that wrongly claims the framework is 'none of which are available or enabled'.
        """
        feature = Feature(CAPABILITY_SHADOW_FEATURE)

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            CapabilityShadowAbstractBase: {MockComputeFramework},
            CapabilityShadowConcrete: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)
        lowered = error_message.lower()

        assert "none of which are available or enabled" not in error_message, (
            f"A capability rejection must not be shadowed by the abstract-only message; the "
            f"framework IS available and was rejected by supports_compute_framework, but got: {error_message}"
        )
        assert "supports_compute_framework" in error_message or "unsupported compute framework" in lowered, (
            f"Error should be the capability-rejection message (mentioning 'supports_compute_framework' "
            f"or 'Unsupported compute framework'), but got: {error_message}"
        )


class TestAbstractFeatureGroupNotSelectedByResolution:
    """Unit-level regression: an abstract FeatureGroup must not win resolution."""

    def test_abstract_feature_group_is_not_selected_as_winner(self) -> None:
        """Resolution must reject the abstract-only candidate with a clean ValueError.

        Currently ``IdentifyFeatureGroupClass`` returns the abstract class instead
        of raising, so the crash is deferred to engine instantiation. This test
        pins that resolution itself raises the regular unresolvable-feature
        ``ValueError`` and never leaks the abstract-instantiation ``TypeError``
        message.
        """
        feature = Feature(ABSTRACT_WINNER_FEATURE)

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            AbstractWinnerFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_message = str(exc_info.value)
        assert "Can't instantiate abstract class" not in error_message, (
            f"Resolution must not surface the abstract-instantiation TypeError text, but got: {error_message}"
        )


class TestAbstractAggregatedFeatureGroupIntegration:
    """Integration repro of issue #646 via mlodaAPI.run_all."""

    def test_abstract_aggregated_base_raises_clean_value_error(self) -> None:
        """Restricting frameworks to PythonDictFramework drops the concrete
        ``IsolatedConcreteAgg`` (pinned to ``PandasDataFrame``), leaving only the
        abstract ``IsolatedAbstractAggBase`` as the sole name-match.

        The hierarchy is defined in THIS module under a unique feature name, so no
        other test double in the parallel suite can collide on it. The run must
        raise a regular ``ValueError`` (unresolvable feature), NOT a raw
        ``TypeError`` about instantiating an abstract class, and the message must
        mention the compute framework the concrete subclass requires
        (``PandasDataFrame``).
        """
        PluginLoader.all()

        with pytest.raises(ValueError) as exc_info:
            mlodaAPI.run_all(
                [ISOLATED_ABSTRACT_AGG_FEATURE],
                compute_frameworks=["PythonDictFramework"],
                api_data={"demo": {"qty": [10.0, 20.0, 30.0]}},
            )

        error_message = str(exc_info.value)

        assert "Can't instantiate abstract class" not in error_message, (
            f"Run must not surface the abstract-instantiation TypeError text, but got: {error_message}"
        )
        assert "PandasDataFrame" in error_message, (
            f"Error should mention the compute framework a concrete impl requires "
            f"(e.g. 'PandasDataFrame'), but got: {error_message}"
        )
