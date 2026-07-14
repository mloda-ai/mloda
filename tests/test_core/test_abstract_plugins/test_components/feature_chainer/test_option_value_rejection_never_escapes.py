"""A value the mapping rejects, however it is rejected, is a non-match with an actionable reason.

This holds for a feature group that overrides the match hook and calls the parser directly, and for an
``element_validator`` that raises instead of returning falsy. Nothing escapes the filter loop as an exception.

All fixture names carry an "esc732" marker so they cannot collide with other tests in the global plugin registry.
"""

from __future__ import annotations

from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


PIPELINE_KEY = "pipeline_name_esc732"
OPERATIONS_KEY = "operations_esc732"
PREFIX_KEY = "prefix_esc732"

PIPELINE_FEATURE = "income__esc732_pipeline_custom"
OPERATIONS_FEATURE = "text__cleaned_esc732"
PREFIX_FEATURE = "text__prefixed_esc732"

# An unhashable element: `element in {dict}` raises TypeError instead of returning False.
UNHASHABLE_VALUE: dict[str, int] = {"a": 1}

SUPPORTED_OPERATIONS_ESC732: dict[str, str] = {
    "normalize": "Normalize the text (esc732 fixture)",
    "strip": "Strip the text (esc732 fixture)",
}


class MockComputeFrameworkEsc732(ComputeFramework):
    """Mock compute framework, following the existing identify_feature_group test pattern."""


class DirectParserOverrideFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Mirrors SklearnPipelineFeatureGroup: overrides the match hook and calls the parser directly, with no
    ValueError swallow of its own. The strict key is never encoded in the feature name.
    """

    PREFIX_PATTERN = r".*__esc732_pipeline_([\w]+)$"

    PROPERTY_MAPPING = {
        PIPELINE_KEY: PropertySpec(
            "Pipeline to apply (esc732 fixture)",
            allowed_values={"scaling": "Scaling", "imputation": "Imputation"},
            context=True,
            strict_validation=True,
            default=None,
        )
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        name = str(feature_name)
        if options.get(PIPELINE_KEY) is None and "esc732_pipeline_" not in name:
            return False
        return FeatureChainParser.match_configuration_feature_chain_parser(
            name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class NeighborFeatureGroup(FeatureGroup):
    """Claims the same feature name and accepts any options: the feature has a valid owner."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == PIPELINE_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class RaisingTypeErrorValidatorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Mirrors TextCleaningFeatureGroup: element_validator is membership against a dict, so an unhashable
    element makes the validator itself raise TypeError instead of returning False.
    """

    PREFIX_PATTERN = r".*__cleaned_esc732$"

    PROPERTY_MAPPING = {
        OPERATIONS_KEY: PropertySpec(
            "Operations to apply (esc732 fixture)",
            allowed_values=SUPPORTED_OPERATIONS_ESC732,
            context=True,
            strict_validation=True,
            element_validator=lambda op: op in SUPPORTED_OPERATIONS_ESC732,
        ),
        DefaultOptionKeys.in_features: PropertySpec(
            "Source feature (esc732 fixture)",
            context=True,
            strict_validation=False,
        ),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class RaisingAttributeErrorValidatorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """element_validator that raises AttributeError on a value of the wrong type."""

    PREFIX_PATTERN = r".*__prefixed_esc732$"

    PROPERTY_MAPPING = {
        PREFIX_KEY: PropertySpec(
            "Value must start with 'ok' (esc732 fixture)",
            context=True,
            strict_validation=True,
            element_validator=lambda value: value.startswith("ok"),
        )
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _identify(feature: Feature, accessible_plugins: FeatureGroupEnvironmentMapping) -> IdentifyFeatureGroupClass:
    return IdentifyFeatureGroupClass(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )


class TestRaisingElementValidatorIsARejection:
    """A validator that raises rejects the value; it does not blow up the match."""

    def test_name_path_typeerror_validator_is_a_non_match(self) -> None:
        """An unhashable element makes the membership-style validator raise: still a non-match."""
        options = Options(context={OPERATIONS_KEY: [UNHASHABLE_VALUE]})

        result = RaisingTypeErrorValidatorFeatureGroup.match_feature_group_criteria(OPERATIONS_FEATURE, options)

        assert result is False

    def test_config_path_typeerror_validator_is_a_non_match(self) -> None:
        """The config-based path reaches the same verdict for the same value."""
        options = Options(
            context={OPERATIONS_KEY: [UNHASHABLE_VALUE], DefaultOptionKeys.in_features: "text"},
        )

        result = RaisingTypeErrorValidatorFeatureGroup.match_feature_group_criteria("placeholder_esc732", options)

        assert result is False

    def test_name_path_attributeerror_validator_is_a_non_match(self) -> None:
        """A validator raising AttributeError is a rejection too, not an escape."""
        options = Options(context={PREFIX_KEY: 5})

        result = RaisingAttributeErrorValidatorFeatureGroup.match_feature_group_criteria(PREFIX_FEATURE, options)

        assert result is False

    def test_config_path_attributeerror_validator_is_a_non_match(self) -> None:
        """Same verdict on the config-based path."""
        options = Options(context={PREFIX_KEY: 5})

        result = RaisingAttributeErrorValidatorFeatureGroup.match_feature_group_criteria("placeholder_esc732", options)

        assert result is False

    def test_parser_reports_a_raising_validator_as_a_value_rejection(self) -> None:
        """The parser converts the validator's own exception into the ValueError its callers read."""
        with pytest.raises(ValueError) as exc_info:
            FeatureChainParser.match_configuration_feature_chain_parser(
                OPERATIONS_FEATURE,
                Options(context={OPERATIONS_KEY: [UNHASHABLE_VALUE]}),
                property_mapping=RaisingTypeErrorValidatorFeatureGroup.PROPERTY_MAPPING,
                prefix_patterns=[RaisingTypeErrorValidatorFeatureGroup.PREFIX_PATTERN],
            )

        message = str(exc_info.value)
        assert OPERATIONS_KEY in message
        assert str(UNHASHABLE_VALUE) in message


class TestRaisingElementValidatorRejectionReason:
    """The discarded message stays actionable: it names the key and the rejected value."""

    def test_reason_for_typeerror_validator_on_name_path(self) -> None:
        """_strict_validation_rejection_reason is diagnostic-only: it reports, it never raises."""
        options = Options(context={OPERATIONS_KEY: [UNHASHABLE_VALUE]})

        reason = RaisingTypeErrorValidatorFeatureGroup._strict_validation_rejection_reason(OPERATIONS_FEATURE, options)

        assert reason is not None
        assert OPERATIONS_KEY in reason
        assert str(UNHASHABLE_VALUE) in reason

    def test_reason_for_typeerror_validator_on_config_path(self) -> None:
        """The config-based path reports the same rejection."""
        options = Options(
            context={OPERATIONS_KEY: [UNHASHABLE_VALUE], DefaultOptionKeys.in_features: "text"},
        )

        reason = RaisingTypeErrorValidatorFeatureGroup._strict_validation_rejection_reason(
            "placeholder_esc732", options
        )

        assert reason is not None
        assert OPERATIONS_KEY in reason
        assert str(UNHASHABLE_VALUE) in reason

    def test_reason_for_attributeerror_validator(self) -> None:
        """An AttributeError-raising validator produces a reason, not an escape."""
        options = Options(context={PREFIX_KEY: 5})

        reason = RaisingAttributeErrorValidatorFeatureGroup._strict_validation_rejection_reason(PREFIX_FEATURE, options)

        assert reason is not None
        assert PREFIX_KEY in reason
        assert "5" in reason


class TestValidValuesStillMatchThroughRaisingValidators:
    """Guard against over-rejecting: the validators still accept what they always accepted."""

    def test_valid_operation_on_name_path_matches(self) -> None:
        options = Options(context={OPERATIONS_KEY: ["normalize"]})

        assert RaisingTypeErrorValidatorFeatureGroup.match_feature_group_criteria(OPERATIONS_FEATURE, options) is True

    def test_valid_operation_on_config_path_matches(self) -> None:
        options = Options(context={OPERATIONS_KEY: ["normalize"], DefaultOptionKeys.in_features: "text"})

        assert RaisingTypeErrorValidatorFeatureGroup.match_feature_group_criteria("placeholder_esc732", options) is True

    def test_hashable_non_member_is_still_a_plain_rejection(self) -> None:
        """A value the validator can judge (and rejects) keeps its existing verdict."""
        options = Options(context={OPERATIONS_KEY: ["bogus"]})

        assert RaisingTypeErrorValidatorFeatureGroup.match_feature_group_criteria(OPERATIONS_FEATURE, options) is False

    def test_valid_prefix_value_matches(self) -> None:
        options = Options(context={PREFIX_KEY: "ok_value"})

        assert RaisingAttributeErrorValidatorFeatureGroup.match_feature_group_criteria(PREFIX_FEATURE, options) is True

    def test_bare_name_match_without_options_still_matches(self) -> None:
        """Required-PRESENCE stays off on the name path, raising validators or not."""
        assert RaisingTypeErrorValidatorFeatureGroup.match_feature_group_criteria(OPERATIONS_FEATURE, Options()) is True


class TestRejectionNeverEscapesTheEngine:
    """The filter loop must never leak a rejection as an exception, whoever calls the parser."""

    def test_direct_parser_caller_yields_the_standard_no_match_error(self) -> None:
        """A direct parser caller yields the normal "No feature groups found" error carrying the reason."""
        feature = Feature(PIPELINE_FEATURE, Options(context={PIPELINE_KEY: "custom"}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            DirectParserOverrideFeatureGroup: {MockComputeFrameworkEsc732},
        }

        with pytest.raises(ValueError) as exc_info:
            _identify(feature, accessible_plugins)

        message = str(exc_info.value)
        assert "No feature groups found" in message, (
            f"A rejected option value must be a non-match, not a bare parser error out of the engine, "
            f"but got: {message}"
        )
        assert PIPELINE_KEY in message
        assert "custom" in message

    def test_direct_parser_caller_does_not_poison_other_candidates(self) -> None:
        """One rejecting candidate must not deny the feature to the group that legitimately owns it."""
        feature = Feature(PIPELINE_FEATURE, Options(context={PIPELINE_KEY: "custom"}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            DirectParserOverrideFeatureGroup: {MockComputeFrameworkEsc732},
            NeighborFeatureGroup: {MockComputeFrameworkEsc732},
        }

        identified = _identify(feature, accessible_plugins)

        feature_group, _frameworks = identified.get()
        assert feature_group is NeighborFeatureGroup

    def test_raising_element_validator_yields_the_standard_no_match_error(self) -> None:
        """A validator that raises must not escape the engine either."""
        feature = Feature(OPERATIONS_FEATURE, Options(context={OPERATIONS_KEY: [UNHASHABLE_VALUE]}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RaisingTypeErrorValidatorFeatureGroup: {MockComputeFrameworkEsc732},
        }

        with pytest.raises(ValueError) as exc_info:
            _identify(feature, accessible_plugins)

        message = str(exc_info.value)
        assert "No feature groups found" in message, (
            f"A validator that raises must be a non-match, not an exception out of the engine, but got: {message}"
        )
        assert OPERATIONS_KEY in message
        assert str(UNHASHABLE_VALUE) in message

    def test_valid_option_value_still_resolves_through_a_direct_parser_caller(self) -> None:
        """A valid strict value on a string-named feature still identifies its feature group."""
        feature = Feature(
            "income__esc732_pipeline_scaling",
            Options(context={PIPELINE_KEY: "scaling"}),
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            DirectParserOverrideFeatureGroup: {MockComputeFrameworkEsc732},
        }

        identified = _identify(feature, accessible_plugins)

        feature_group, _frameworks = identified.get()
        assert feature_group is DirectParserOverrideFeatureGroup


class TestParserContractUnchangedForOrdinaryRejections:
    """The ordinary membership rejection keeps raising ValueError from the parser."""

    def test_parser_raises_valueerror_for_a_direct_caller(self) -> None:
        """The parser signals a rejected value; the caller is what must not let it escape."""
        with pytest.raises(ValueError) as exc_info:
            FeatureChainParser.match_configuration_feature_chain_parser(
                PIPELINE_FEATURE,
                Options(context={PIPELINE_KEY: "custom"}),
                property_mapping=DirectParserOverrideFeatureGroup.PROPERTY_MAPPING,
                prefix_patterns=[DirectParserOverrideFeatureGroup.PREFIX_PATTERN],
            )

        message = str(exc_info.value)
        assert PIPELINE_KEY in message
        assert "custom" in message
