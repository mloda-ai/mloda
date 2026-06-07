"""Tests for the MATCH-TIME compute-framework capability hook (issue #482).

These tests pin down the contract for a sanctioned way for a FeatureGroup to
declare an operation unsupported on a given compute framework, evaluated during
feature -> feature-group resolution.

Contract under test (to be implemented by the Green agent):

1. ``FeatureGroup.supports_compute_framework(feature_name, options, compute_framework) -> bool``
   classmethod, default ``True``. Arg order mirrors ``match_feature_group_criteria``.
2. ``IdentifyFeatureGroupClass._filter_loop`` narrows the candidate compute-framework
   set per feature using the hook (route-around when an unsupported framework is not
   the only candidate).
3. When the only candidate is an unsupported framework (e.g. user pin), a DEDICATED,
   distinguishable ``ValueError`` is raised that names the unsupported and supported
   framework(s) and skips the fuzzy "Did you mean" suggestion path.

All tests are written to FAIL until the feature exists.
"""

from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


CAPABILITY_FEATURE = "capability_test_feature"


class CapabilityFwA(ComputeFramework):
    """First compute framework used in capability tests."""

    pass


class CapabilityFwB(ComputeFramework):
    """Second compute framework used in capability tests."""

    pass


class PlainCapabilityFeatureGroup(FeatureGroup):
    """Plain feature group that does NOT override supports_compute_framework.

    Used to assert the default hook behaviour and no-regression resolution.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == CAPABILITY_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class RejectBCapabilityFeatureGroup(FeatureGroup):
    """Feature group that declares the operation unsupported on CapabilityFwB.

    supports_compute_framework returns False for CapabilityFwB and True otherwise.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == CAPABILITY_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return compute_framework is not CapabilityFwB

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class RejectAllCapabilityFeatureGroup(FeatureGroup):
    """Feature group that declares the operation unsupported on every framework."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == CAPABILITY_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class PandasLikeFG(FeatureGroup):
    """Declares only CapabilityFwA and supports the op on it (default hook).

    Mirrors an installed/declared backend whose framework may or may not be
    enabled for a particular run.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CapabilityFwA}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == CAPABILITY_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SqliteLikeFG(FeatureGroup):
    """Declares only CapabilityFwB and rejects the op on it."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CapabilityFwB}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == CAPABILITY_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return compute_framework is not CapabilityFwB

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestComputeFrameworkCapabilityHook:
    """Capability-hook contract for compute-framework support at match time."""

    def test_default_hook_is_true_and_no_regression(self) -> None:
        """Default hook returns True and resolution keeps every declared framework."""
        feature = Feature(CAPABILITY_FEATURE)

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            PlainCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        feature_group_class, compute_frameworks = identified.get()
        assert feature_group_class is PlainCapabilityFeatureGroup
        assert compute_frameworks == {CapabilityFwA, CapabilityFwB}, (
            f"Default behaviour must keep all declared frameworks; got {compute_frameworks}"
        )

        # The hook must exist on the base class and default to True for any framework.
        assert FeatureGroup.supports_compute_framework(CAPABILITY_FEATURE, feature.options, CapabilityFwA) is True
        assert FeatureGroup.supports_compute_framework(CAPABILITY_FEATURE, feature.options, CapabilityFwB) is True

    def test_route_around_unsupported_framework_without_pin(self) -> None:
        """Without a pin, an unsupported framework is dropped, the supported one kept."""
        feature = Feature(CAPABILITY_FEATURE)

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RejectBCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        identified = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        feature_group_class, compute_frameworks = identified.get()
        assert feature_group_class is RejectBCapabilityFeatureGroup
        assert compute_frameworks == {CapabilityFwA}, (
            f"Unsupported CapabilityFwB must be narrowed out, leaving only CapabilityFwA; got {compute_frameworks}"
        )

    def test_distinguishable_error_when_pinned_to_unsupported_framework(self) -> None:
        """Pinning to the unsupported framework yields a dedicated, distinguishable error."""
        feature = Feature(CAPABILITY_FEATURE)
        feature._set_compute_frameworks({CapabilityFwB})

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RejectBCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
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

        assert "unsupported" in lowered or "not supported" in lowered, (
            f"Capability error must signal an unsupported framework, but got: {error_message}"
        )
        assert CapabilityFwB.get_class_name() in error_message, (
            f"Error must name the unsupported framework '{CapabilityFwB.get_class_name()}', but got: {error_message}"
        )
        assert CapabilityFwA.get_class_name() in error_message, (
            f"Error must name the supported framework '{CapabilityFwA.get_class_name()}', but got: {error_message}"
        )
        assert "Did you mean" not in error_message, (
            f"Capability error must skip the fuzzy suggestion path, but got: {error_message}"
        )

    def test_all_frameworks_reject_operation(self) -> None:
        """When every framework rejects the op, a graceful error names the feature."""
        feature = Feature(CAPABILITY_FEATURE)

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RejectAllCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
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

        assert CAPABILITY_FEATURE in error_message, f"Error must mention the feature name, but got: {error_message}"
        assert "unsupported" in lowered or "not supported" in lowered, (
            f"Error must signal that the framework(s) are unsupported, but got: {error_message}"
        )
        assert CapabilityFwA.get_class_name() in error_message, (
            f"Error must name rejected framework '{CapabilityFwA.get_class_name()}', but got: {error_message}"
        )
        assert CapabilityFwB.get_class_name() in error_message, (
            f"Error must name rejected framework '{CapabilityFwB.get_class_name()}', but got: {error_message}"
        )

    def test_capability_error_distinguishable_from_unknown_feature(self) -> None:
        """The capability error is clearly different from the unknown-feature error."""
        # Genuinely unknown feature -> classic resolution error.
        unknown_feature = Feature("totally_unknown_capability_feature_xyz")
        unknown_accessible: FeatureGroupEnvironmentMapping = {
            RejectAllCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        with pytest.raises(ValueError) as unknown_exc:
            IdentifyFeatureGroupClass(
                feature=unknown_feature,
                accessible_plugins=unknown_accessible,
                links=None,
                data_access_collection=None,
            )
        unknown_message = str(unknown_exc.value)
        assert "No feature groups found" in unknown_message, (
            f"Unknown feature must keep the classic error, but got: {unknown_message}"
        )

        # Capability rejection -> distinct, dedicated error.
        pinned_feature = Feature(CAPABILITY_FEATURE)
        pinned_feature._set_compute_frameworks({CapabilityFwB})
        capability_accessible: FeatureGroupEnvironmentMapping = {
            RejectBCapabilityFeatureGroup: {CapabilityFwA, CapabilityFwB},
        }

        with pytest.raises(ValueError) as capability_exc:
            IdentifyFeatureGroupClass(
                feature=pinned_feature,
                accessible_plugins=capability_accessible,
                links=None,
                data_access_collection=None,
            )
        capability_message = str(capability_exc.value)
        lowered = capability_message.lower()

        assert "unsupported" in lowered or "not supported" in lowered, (
            f"Capability error must signal an unsupported framework, but got: {capability_message}"
        )
        assert "No feature groups found for feature name" not in capability_message, (
            f"Capability error must be distinguishable from the unknown-feature error, but got: {capability_message}"
        )

    def test_supported_list_includes_declared_but_not_enabled_framework(self) -> None:
        """A declared+available+supporting framework appears in 'Supported on' even if not enabled this run.

        Change B: the 'Supported on' list must be derived from each criteria-matched
        feature group's ``compute_framework_definition()`` intersected with availability
        and ``supports_compute_framework``, NOT from the run-enabled set. Here CapabilityFwA
        is declared by ``PandasLikeFG`` and available, but is NOT enabled for this run
        (its accessible_plugins value is an empty set). It must still show up as supported.
        """
        feature = Feature(CAPABILITY_FEATURE)
        feature._set_compute_frameworks({CapabilityFwB})

        # FwA declared by PandasLikeFG but NOT enabled (empty set); FwB enabled for SqliteLikeFG.
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            PandasLikeFG: set(),
            SqliteLikeFG: {CapabilityFwB},
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

        assert "unsupported" in lowered or "not supported" in lowered, (
            f"Capability error must signal an unsupported framework, but got: {error_message}"
        )
        assert CapabilityFwB.get_class_name() in error_message, (
            f"Error must name the unsupported framework '{CapabilityFwB.get_class_name()}', but got: {error_message}"
        )
        assert CapabilityFwA.get_class_name() in error_message, (
            "Error 'Supported on' list must include the declared+available+supporting framework "
            f"'{CapabilityFwA.get_class_name()}' even though it was not enabled this run, but got: {error_message}"
        )
        assert "Did you mean" not in error_message, (
            f"Capability error must skip the fuzzy suggestion path, but got: {error_message}"
        )
