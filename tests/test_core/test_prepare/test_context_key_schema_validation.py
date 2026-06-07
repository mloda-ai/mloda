"""Tests for context-key schema validation at the planning chokepoint.

A FeatureGroup may opt in to context-key validation by overriding
``context_key_schema()``. When it does, ``IdentifyFeatureGroupClass`` validates
the resolving feature's context keys against that schema and raises an
actionable ValueError on an unknown key. Feature groups that do not override
``context_key_schema()`` keep today's permissive behavior (no validation).
"""

from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


class MockComputeFramework(ComputeFramework):
    """Mock compute framework for testing."""

    pass


class OptInContextSchemaFeatureGroup(FeatureGroup):
    """Feature group that opts in to context-key validation.

    Declares a single accepted context key ``partition_by`` of type ``str``.
    Matches the feature named ``opt_in_ctx_feature``.
    """

    @classmethod
    def context_key_schema(cls) -> dict[str, Any] | None:
        return {"partition_by": str}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == "opt_in_ctx_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class PermissiveContextFeatureGroup(FeatureGroup):
    """Feature group that does NOT override context_key_schema (default None).

    Matches the feature named ``permissive_ctx_feature``. Keeps permissive
    behavior: any context key is accepted.
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
        return feature_name == "permissive_ctx_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestContextKeySchemaValidation:
    """Validation of context keys during feature group identification."""

    def test_unknown_context_key_raises_with_suggestion(self) -> None:
        """An opt-in FG with a typo'd context key raises a ValueError with a suggestion."""
        feature = Feature(name="opt_in_ctx_feature", options=Options(context={"partiton_by": "col"}))

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            OptInContextSchemaFeatureGroup: {MockComputeFramework},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        error_msg = str(exc_info.value)
        assert "partiton_by" in error_msg
        assert "did you mean" in error_msg
        assert "partition_by" in error_msg

    def test_valid_context_key_constructs(self) -> None:
        """An opt-in FG with a valid context key constructs without error."""
        feature = Feature(name="opt_in_ctx_feature", options=Options(context={"partition_by": "col"}))

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            OptInContextSchemaFeatureGroup: {MockComputeFramework},
        }

        # Act & Assert - should not raise
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

    def test_permissive_feature_group_accepts_any_context_key(self) -> None:
        """A FG that does not override context_key_schema accepts any context key (backward compat)."""
        feature = Feature(name="permissive_ctx_feature", options=Options(context={"anything_goes": 1}))

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            PermissiveContextFeatureGroup: {MockComputeFramework},
        }

        # Act & Assert - should not raise
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

    def test_propagated_context_key_is_exempt(self) -> None:
        """A propagated context key is exempt from validation even if absent from schema."""
        feature = Feature(
            name="opt_in_ctx_feature",
            options=Options(
                context={"partition_by": "c", "propagated_key": "v"},
                propagate_context_keys=frozenset({"propagated_key"}),
            ),
        )

        accessible_plugins: FeatureGroupEnvironmentMapping = {
            OptInContextSchemaFeatureGroup: {MockComputeFramework},
        }

        # Act & Assert - should not raise (propagated_key is exempt)
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
