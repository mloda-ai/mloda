"""Planning-level context-key validation for CONFIG-BASED feature groups (TDD red phase).

A config-based feature group (``FeatureChainParserMixin`` + ``PROPERTY_MAPPING``) opts
into context-key validation by deriving its ``context_key_schema()`` from
``PROPERTY_MAPPING``::

    @classmethod
    def context_key_schema(cls):
        return cls.derive_context_key_schema()

``derive_context_key_schema`` does not exist yet, so the opt-in tests below currently
fail with ``AttributeError`` (raised when ``IdentifyFeatureGroupClass.validate`` calls
``context_key_schema()``). They go green once the Green Agent adds the helper.

Test (d) is a backward-compat / boundary guard: it exercises the *inherited* mixin
matcher with a REQUIRED property and a typo, so the group never matches and validation
never runs. It passes today and must keep passing.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from mloda.provider import ComputeFramework
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureChainParserMixin
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


class MockComputeFramework(ComputeFramework):
    """Mock compute framework for testing (never instantiated on this path)."""

    pass


FIXED_NAME = "config_opt_in_feature"


class ConfigOptInFG(FeatureChainParserMixin, FeatureGroup):
    """Config-based feature group that opts into PROPERTY_MAPPING-derived validation.

    Uses a fixed-name matcher so it resolves uniquely and never accidentally matches
    other plugins (mirrors the isolation style of the error-message tests).
    """

    PROPERTY_MAPPING = {
        "partition_by": {
            "explanation": "partition columns (optional context key)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        "frame_size": {
            "explanation": "rolling frame size (optional context key)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        "group_default_key": {
            "explanation": "declared property without a context flag (group-default)",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "source features",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        name = str(feature_name) if isinstance(feature_name, FeatureName) else feature_name
        return name == FIXED_NAME

    @classmethod
    def context_key_schema(cls) -> Optional[dict[str, Any]]:
        return cls.derive_context_key_schema()

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


REQUIRED_PROP_NAME = "required_prop_feature"


class RequiredPropFG(FeatureChainParserMixin, FeatureGroup):
    """Config-based group with a REQUIRED property, using the INHERITED mixin matcher.

    ``window_type`` has no ``default`` and no ``required_when`` and uses strict
    validation, so it is required for a configuration-based match. A feature whose
    context only carries a typo'd version of the key therefore fails to match.
    """

    PROPERTY_MAPPING = {
        "window_type": {
            "explanation": "required window kind",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            "rolling": "rolling",
            "expanding": "expanding",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "source features",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def context_key_schema(cls) -> Optional[dict[str, Any]]:
        return cls.derive_context_key_schema()

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestConfigBasedContextKeySchema:
    """Validation behavior for config-based opt-in feature groups."""

    def test_typo_on_optional_context_key_raises(self) -> None:
        """(a) A typo on an OPTIONAL context key fires the 'did you mean' error."""
        feature = Feature(
            FIXED_NAME,
            options=Options(context={"in_features": "x", "partiton_by": "col"}),
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {ConfigOptInFG: {MockComputeFramework}}

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
            )

        message = str(exc_info.value)
        assert "partiton_by" in message, message
        assert "did you mean" in message.lower(), message
        assert "partition_by" in message, message

    def test_valid_context_keys_resolve(self) -> None:
        """(b) Valid, declared context keys resolve without error."""
        feature = Feature(
            FIXED_NAME,
            options=Options(context={"in_features": "x", "partition_by": "col", "frame_size": 10}),
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {ConfigOptInFG: {MockComputeFramework}}

        result = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
        )
        assert ConfigOptInFG in result.feature_group_compute_framework_mapping

    def test_declared_group_default_key_in_context_is_accepted(self) -> None:
        """(c) A declared PROPERTY_MAPPING key placed in context is accepted (no false positive)."""
        feature = Feature(
            FIXED_NAME,
            options=Options(context={"in_features": "x", "group_default_key": "v"}),
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {ConfigOptInFG: {MockComputeFramework}}

        result = IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
        )
        assert ConfigOptInFG in result.feature_group_compute_framework_mapping

    def test_typo_on_required_property_does_not_produce_context_key_suggestion(self) -> None:
        """(d) Boundary: a typo on a REQUIRED property fails matching, so context-key
        validation never runs. The error must be the 'no feature groups found' error,
        not a 'context key' suggestion. (Passes today; locks in the documented limit.)
        """
        feature = Feature(
            REQUIRED_PROP_NAME,
            options=Options(context={"in_features": "x", "window_typ": "rolling"}),
        )

        # Direct, robust check: the inherited mixin matcher rejects the feature because
        # the required property is effectively absent (only a typo'd key is present).
        assert RequiredPropFG.match_feature_group_criteria(feature.name, feature.options) is False

        accessible_plugins: FeatureGroupEnvironmentMapping = {RequiredPropFG: {MockComputeFramework}}

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
            )

        message = str(exc_info.value)
        assert "context key" not in message.lower(), message
        assert "no feature groups found" in message.lower(), message
