"""A value the mapping rejects, however it is rejected, is a non-match with an actionable reason.

This holds for a feature group that overrides the match hook and calls the parser directly, and for an
``element_validator`` that raises instead of returning falsy. Nothing escapes the filter loop as an exception.

All fixture names carry an "esc732" marker so they cannot collide with other tests in the global plugin registry.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
    PropertyValueRejection,
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
from tests.test_core.test_prepare.identify_seam import identify_winner


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


# The esc763 fixtures pin issue #763: a user callable that raises a NON-(TypeError/ValueError/AttributeError)
# exception (here KeyError) means "cannot judge this value" = a rejection, never an escape that aborts the
# identification of the OTHER candidate feature groups. Fresh marker so these cannot collide in the registry.
VALIDATOR_KEY_ESC763 = "validator_key_esc763"
GUARD_KEY_ESC763 = "guard_key_esc763"
REQUIRED_WHEN_KEY_ESC763 = "required_when_key_esc763"
DRIVER_KEY_ESC763 = "required_when_driver_esc763"

VALIDATOR_FEATURE_ESC763 = "text__validated_esc763"
GUARD_FEATURE_ESC763 = "text__guarded_esc763"
REQUIRED_WHEN_FEATURE_ESC763 = "text__required_esc763"

# A value the callables cannot judge: looking at it raises KeyError, exactly the class the containment misses today.
POISON_VALUE_ESC763 = "poison_esc763"
# A value the callables CAN judge and accept: proves the fix does not over-reject.
GOOD_VALUE_ESC763 = "good_esc763"
# required_when driver modes: "needs" makes the key required, "optional" leaves it optional.
NEEDS_MODE_ESC763 = "needs_esc763"
OPTIONAL_MODE_ESC763 = "optional_esc763"

# The tier763 markers pin the two-tier logging contract on the user-callable containment sites: an expected
# judgment failure (TypeError, ValueError, AttributeError) stays at DEBUG, while any other exception class means
# the callable itself looks broken and must surface at WARNING. Containment (reject / non-match) is unchanged
# for user callables; a raise from framework-owned build_effective_options surfaces instead (os-005).
TYPE_POISON_VALUE_TIER763 = "type_poison_tier763"
CAUSE_KEY_TIER763 = "cause_key_tier763"

# Both modules create their logger via logging.getLogger(__name__), so the class's module IS the logger name.
PARSER_LOGGER_NAME = FeatureChainParser.__module__
MIXIN_LOGGER_NAME = FeatureChainParserMixin.__module__


def _keyerror_element_validator_esc763(value: Any) -> bool:
    """Judge one element; the poison value raises KeyError (broken-looking), the tier value TypeError (expected)."""
    if value == POISON_VALUE_ESC763:
        raise KeyError("esc763 element_validator cannot judge the poison value")
    if value == TYPE_POISON_VALUE_TIER763:
        raise TypeError("tier763 element_validator expected judgment failure")
    return bool(value == GOOD_VALUE_ESC763)


def _keyerror_match_guard_esc763(value: Any) -> bool:
    """Guard the raw value; the poison value raises KeyError (broken-looking), the tier value TypeError (expected)."""
    if value == POISON_VALUE_ESC763:
        raise KeyError("esc763 match_guard cannot judge the poison value")
    if value == TYPE_POISON_VALUE_TIER763:
        raise TypeError("tier763 match_guard expected judgment failure")
    return bool(value == GOOD_VALUE_ESC763)


def _keyerror_required_when_esc763(options: Options) -> bool:
    """Decide if the guarded key is required; a poison driver raises KeyError, the tier driver TypeError."""
    driver = options.get(DRIVER_KEY_ESC763)
    if driver == POISON_VALUE_ESC763:
        raise KeyError("esc763 required_when cannot judge the poison driver value")
    if driver == TYPE_POISON_VALUE_TIER763:
        raise TypeError("tier763 required_when expected judgment failure")
    return bool(driver == NEEDS_MODE_ESC763)


def _raise_build_effective_options_esc763(cls: Any, *args: Any, **kwargs: Any) -> Options:
    """Force build_effective_options to raise a plain ValueError: a raise the narrowed containment must surface."""
    raise ValueError("esc763 build_effective_options boom")


def _raise_keyerror_build_effective_options_tier763(cls: Any, *args: Any, **kwargs: Any) -> Options:
    """Force build_effective_options to raise KeyError: a raise the narrowed containment must surface, not contain."""
    raise KeyError("tier763 build_effective_options boom")


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
            deferred_binding=True,  # mirrors TextCleaningFeatureGroup: parsed from the name by the group (#769)
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


class RaisingKeyErrorValidatorFeatureGroupEsc763(FeatureChainParserMixin, FeatureGroup):
    """element_validator raises KeyError (not a caught TypeError/ValueError/AttributeError) on a poison value."""

    PREFIX_PATTERN = r".*__validated_esc763$"

    PROPERTY_MAPPING = {
        VALIDATOR_KEY_ESC763: PropertySpec(
            "A value the validator cannot judge raises KeyError (esc763 fixture)",
            context=True,
            strict_validation=True,
            element_validator=_keyerror_element_validator_esc763,
        )
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class RaisingKeyErrorGuardFeatureGroupEsc763(FeatureChainParserMixin, FeatureGroup):
    """match_guard raises KeyError (not a caught class) on a poison value; needs no strict_validation."""

    PREFIX_PATTERN = r".*__guarded_esc763$"

    PROPERTY_MAPPING = {
        GUARD_KEY_ESC763: PropertySpec(
            "A value the guard cannot judge raises KeyError (esc763 fixture)",
            context=True,
            match_guard=_keyerror_match_guard_esc763,
        )
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class RaisingKeyErrorRequiredWhenFeatureGroupEsc763(FeatureChainParserMixin, FeatureGroup):
    """required_when predicate raises KeyError (not a caught class) on a poison driver value.

    Its single required_when key is skippable, so an unscoped matcher would claim any feature whose
    options omit the driver and pollute the global registry. The gated override below scopes it to its
    own feature; the class-definition guard still wraps the override, so the raising predicate runs.
    """

    PREFIX_PATTERN = r".*__required_esc763$"

    PROPERTY_MAPPING = {
        REQUIRED_WHEN_KEY_ESC763: PropertySpec(
            "Guarded by a required_when predicate that can raise KeyError (esc763 fixture)",
            context=True,
            required_when=_keyerror_required_when_esc763,
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
        # Classmethod is mandatory: required_when forbids a staticmethod matcher. Claim only this
        # fixture's own feature so the auto-registered class never owns an unrelated feature.
        if name != REQUIRED_WHEN_FEATURE_ESC763 and options.get(DRIVER_KEY_ESC763) is None:
            return False
        return FeatureChainParser.match_configuration_feature_chain_parser(
            name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class RequiredWhenNeighborFeatureGroupEsc763(FeatureGroup):
    """Claims REQUIRED_WHEN_FEATURE_ESC763 and accepts any options: the valid owner the raising candidate must not deny."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == REQUIRED_WHEN_FEATURE_ESC763

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _identify(
    feature: Feature, accessible_plugins: FeatureGroupEnvironmentMapping
) -> tuple[type[FeatureGroup], set[type[ComputeFramework]]]:
    return identify_winner(feature, accessible_plugins)


def _records_at_or_above(caplog: pytest.LogCaptureFixture, logger_name: str, levelno: int) -> list[logging.LogRecord]:
    """Records the named logger emitted at or above the given level."""
    return [record for record in caplog.records if record.name == logger_name and record.levelno >= levelno]


def _records_at_exactly(caplog: pytest.LogCaptureFixture, logger_name: str, levelno: int) -> list[logging.LogRecord]:
    """Records the named logger emitted at exactly the given level."""
    return [record for record in caplog.records if record.name == logger_name and record.levelno == levelno]


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

        feature_group, _frameworks = identified
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

        feature_group, _frameworks = identified
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


class TestKeyErrorElementValidatorIsARejection:
    """issue #763: a validator raising a non-(TypeError/ValueError/AttributeError) is a rejection, not an escape."""

    def test_keyerror_validator_on_name_path_is_a_non_match(self) -> None:
        """A validator that raises KeyError is a non-match, not an exception out of the matcher."""
        options = Options(context={VALIDATOR_KEY_ESC763: [POISON_VALUE_ESC763]})

        result = RaisingKeyErrorValidatorFeatureGroupEsc763.match_feature_group_criteria(
            VALIDATOR_FEATURE_ESC763, options
        )

        assert result is False

    def test_parser_reports_a_keyerror_validator_as_a_value_rejection(self) -> None:
        """The parser converts the validator's KeyError into the PropertyValueRejection its callers read."""
        with pytest.raises(PropertyValueRejection) as exc_info:
            FeatureChainParser.match_configuration_feature_chain_parser(
                VALIDATOR_FEATURE_ESC763,
                Options(context={VALIDATOR_KEY_ESC763: [POISON_VALUE_ESC763]}),
                property_mapping=RaisingKeyErrorValidatorFeatureGroupEsc763.PROPERTY_MAPPING,
                prefix_patterns=[RaisingKeyErrorValidatorFeatureGroupEsc763.PREFIX_PATTERN],
            )

        message = str(exc_info.value)
        assert VALIDATOR_KEY_ESC763 in message
        assert POISON_VALUE_ESC763 in message

    def test_reason_for_keyerror_validator_is_reported_not_raised(self) -> None:
        """_strict_validation_rejection_reason reports the KeyError rejection; it never raises."""
        options = Options(context={VALIDATOR_KEY_ESC763: [POISON_VALUE_ESC763]})

        reason = RaisingKeyErrorValidatorFeatureGroupEsc763._strict_validation_rejection_reason(
            VALIDATOR_FEATURE_ESC763, options
        )

        assert reason is not None
        assert VALIDATOR_KEY_ESC763 in reason
        assert POISON_VALUE_ESC763 in reason


class TestKeyErrorMatchGuardIsARejection:
    """A match_guard raising a non-caught exception is a non-match, not an escape."""

    def test_keyerror_guard_is_a_non_match(self) -> None:
        """A guard that raises KeyError rejects the value; it does not blow up the match."""
        options = Options(context={GUARD_KEY_ESC763: POISON_VALUE_ESC763})

        result = RaisingKeyErrorGuardFeatureGroupEsc763.match_feature_group_criteria(GUARD_FEATURE_ESC763, options)

        assert result is False


class TestKeyErrorRequiredWhenPredicateIsARejection:
    """A required_when predicate raising a non-caught exception is a non-match, not an escape."""

    def test_keyerror_required_when_predicate_is_a_non_match(self) -> None:
        """A predicate that raises KeyError is a non-match, not an exception out of the matcher."""
        options = Options(context={DRIVER_KEY_ESC763: POISON_VALUE_ESC763})

        result = RaisingKeyErrorRequiredWhenFeatureGroupEsc763.match_feature_group_criteria(
            REQUIRED_WHEN_FEATURE_ESC763, options
        )

        assert result is False


class TestKeyErrorRejectionNeverEscapesTheEngine:
    """The definition-time required_when guard must never leak a KeyError out of the filter loop."""

    def test_keyerror_required_when_yields_the_standard_no_match_error(self) -> None:
        """A required_when predicate that raises must be a non-match, not an exception out of the engine."""
        feature = Feature(REQUIRED_WHEN_FEATURE_ESC763, Options(context={DRIVER_KEY_ESC763: POISON_VALUE_ESC763}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RaisingKeyErrorRequiredWhenFeatureGroupEsc763: {MockComputeFrameworkEsc732},
        }

        with pytest.raises(ValueError) as exc_info:
            _identify(feature, accessible_plugins)

        message = str(exc_info.value)
        assert "No feature groups found" in message, (
            f"A predicate that raises must be a non-match, not an exception out of the engine, but got: {message}"
        )

    def test_keyerror_required_when_does_not_poison_other_candidates(self) -> None:
        """One raising candidate must not deny the feature to the group that legitimately owns it."""
        feature = Feature(REQUIRED_WHEN_FEATURE_ESC763, Options(context={DRIVER_KEY_ESC763: POISON_VALUE_ESC763}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RaisingKeyErrorRequiredWhenFeatureGroupEsc763: {MockComputeFrameworkEsc732},
            RequiredWhenNeighborFeatureGroupEsc763: {MockComputeFrameworkEsc732},
        }

        identified = _identify(feature, accessible_plugins)

        feature_group, _frameworks = identified
        assert feature_group is RequiredWhenNeighborFeatureGroupEsc763


class TestKeyErrorPathsDoNotOverReject:
    """Guard against over-rejecting: a value the callables CAN judge and accept still matches (passes pre-fix)."""

    def test_valid_value_through_keyerror_validator_matches(self) -> None:
        options = Options(context={VALIDATOR_KEY_ESC763: [GOOD_VALUE_ESC763]})

        assert (
            RaisingKeyErrorValidatorFeatureGroupEsc763.match_feature_group_criteria(VALIDATOR_FEATURE_ESC763, options)
            is True
        )

    def test_valid_value_through_keyerror_guard_matches(self) -> None:
        options = Options(context={GUARD_KEY_ESC763: GOOD_VALUE_ESC763})

        assert (
            RaisingKeyErrorGuardFeatureGroupEsc763.match_feature_group_criteria(GUARD_FEATURE_ESC763, options) is True
        )

    def test_untriggered_required_when_predicate_still_matches(self) -> None:
        """When the predicate returns False (option not required), the feature still matches."""
        options = Options(context={DRIVER_KEY_ESC763: OPTIONAL_MODE_ESC763})

        assert (
            RaisingKeyErrorRequiredWhenFeatureGroupEsc763.match_feature_group_criteria(
                REQUIRED_WHEN_FEATURE_ESC763, options
            )
            is True
        )

    def test_triggered_required_when_with_option_present_still_matches(self) -> None:
        """When the predicate returns True but the required option is present, the feature still matches."""
        options = Options(
            context={DRIVER_KEY_ESC763: NEEDS_MODE_ESC763, REQUIRED_WHEN_KEY_ESC763: GOOD_VALUE_ESC763},
        )

        assert (
            RaisingKeyErrorRequiredWhenFeatureGroupEsc763.match_feature_group_criteria(
                REQUIRED_WHEN_FEATURE_ESC763, options
            )
            is True
        )


class TestBuildEffectiveOptionsRaiseSurfaces:
    """os-005 narrows the #763 containment to user-supplied callables (element_validator, match_guard,
    required_when predicates). No user callback runs inside the framework-owned build_effective_options,
    so a raise out of it is a framework defect (or a user configuration error carrying actionable
    guidance) and must propagate out of matching instead of being contained as a non-match.

    The driver is OPTIONAL_MODE_ESC763, so the predicate itself returns False (would match True): the monkeypatched
    build_effective_options, called before the predicate loop, is the sole raise source.
    """

    def test_build_effective_options_raise_propagates_from_the_matcher(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ValueError from build_effective_options propagates out of the matcher instead of being contained."""
        monkeypatch.setattr(
            FeatureChainParser,
            "build_effective_options",
            classmethod(_raise_build_effective_options_esc763),
        )
        options = Options(context={DRIVER_KEY_ESC763: OPTIONAL_MODE_ESC763})

        with pytest.raises(ValueError, match="esc763 build_effective_options boom"):
            RaisingKeyErrorRequiredWhenFeatureGroupEsc763.match_feature_group_criteria(
                REQUIRED_WHEN_FEATURE_ESC763, options
            )

    def test_build_effective_options_raise_propagates_out_of_the_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The engine no longer converts the raise into the standard no-match error: the boom itself surfaces."""
        monkeypatch.setattr(
            FeatureChainParser,
            "build_effective_options",
            classmethod(_raise_build_effective_options_esc763),
        )
        feature = Feature(REQUIRED_WHEN_FEATURE_ESC763, Options(context={DRIVER_KEY_ESC763: OPTIONAL_MODE_ESC763}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            RaisingKeyErrorRequiredWhenFeatureGroupEsc763: {MockComputeFrameworkEsc732},
        }

        with pytest.raises(ValueError, match="esc763 build_effective_options boom") as exc_info:
            _identify(feature, accessible_plugins)

        assert "No feature groups found" not in str(exc_info.value), (
            "a framework-defect raise must not be converted into the standard no-match error"
        )

    def test_keyerror_build_effective_options_raise_propagates_without_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A KeyError from build_effective_options propagates unchanged; the parser logs no WARNING containment."""
        caplog.set_level(logging.DEBUG, logger=PARSER_LOGGER_NAME)
        monkeypatch.setattr(
            FeatureChainParser,
            "build_effective_options",
            classmethod(_raise_keyerror_build_effective_options_tier763),
        )
        options = Options(context={DRIVER_KEY_ESC763: OPTIONAL_MODE_ESC763})

        with pytest.raises(KeyError, match="tier763 build_effective_options boom"):
            RaisingKeyErrorRequiredWhenFeatureGroupEsc763.match_feature_group_criteria(
                REQUIRED_WHEN_FEATURE_ESC763, options
            )

        assert _records_at_or_above(caplog, PARSER_LOGGER_NAME, logging.WARNING) == [], (
            "a propagating build_effective_options raise must not leave a WARNING-tier containment record"
        )

    def test_valueerror_build_effective_options_raise_leaves_no_containment_record(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ValueError from build_effective_options propagates unchanged and leaves no containment record at all."""
        caplog.set_level(logging.DEBUG, logger=PARSER_LOGGER_NAME)
        monkeypatch.setattr(
            FeatureChainParser,
            "build_effective_options",
            classmethod(_raise_build_effective_options_esc763),
        )
        options = Options(context={DRIVER_KEY_ESC763: OPTIONAL_MODE_ESC763})

        with pytest.raises(ValueError, match="esc763 build_effective_options boom"):
            RaisingKeyErrorRequiredWhenFeatureGroupEsc763.match_feature_group_criteria(
                REQUIRED_WHEN_FEATURE_ESC763, options
            )

        assert _records_at_or_above(caplog, PARSER_LOGGER_NAME, logging.WARNING) == []
        debug_records = _records_at_exactly(caplog, PARSER_LOGGER_NAME, logging.DEBUG)
        assert all("non-match" not in record.getMessage() for record in debug_records), (
            "a propagating build_effective_options raise must not leave a DEBUG non-match containment record"
        )


class TestContainedRaiseLogTiers:
    """Two-tier logging for raises contained from user-supplied callables only (element_validator,
    match_guard, required_when predicates): an expected judgment failure (TypeError, ValueError,
    AttributeError) stays at DEBUG, any other exception class looks like a broken callable and must
    surface at WARNING. Containment itself (reject / non-match, never escape) is unchanged for these
    user callables; framework-owned build_effective_options raises propagate instead (os-005, see
    TestBuildEffectiveOptionsRaiseSurfaces).
    """

    def test_keyerror_validator_raise_logs_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A validator raising KeyError still rejects the value, but the containment surfaces at WARNING."""
        caplog.set_level(logging.DEBUG, logger=PARSER_LOGGER_NAME)

        with pytest.raises(PropertyValueRejection):
            FeatureChainParser.match_configuration_feature_chain_parser(
                VALIDATOR_FEATURE_ESC763,
                Options(context={VALIDATOR_KEY_ESC763: [POISON_VALUE_ESC763]}),
                property_mapping=RaisingKeyErrorValidatorFeatureGroupEsc763.PROPERTY_MAPPING,
                prefix_patterns=[RaisingKeyErrorValidatorFeatureGroupEsc763.PREFIX_PATTERN],
            )

        warnings = _records_at_or_above(caplog, PARSER_LOGGER_NAME, logging.WARNING)
        assert warnings, "a KeyError-raising validator looks broken and must log at WARNING"
        assert any(VALIDATOR_KEY_ESC763 in record.getMessage() for record in warnings)
        # The exception text stays for triage; the option value itself is redacted at WARNING (production-visible).
        assert any("element_validator cannot judge the poison value" in record.getMessage() for record in warnings)
        assert all(POISON_VALUE_ESC763 not in record.getMessage() for record in warnings), (
            "option values may carry secrets and must not appear in WARNING-tier containment messages"
        )

    def test_typeerror_validator_raise_logs_at_debug_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """A validator raising TypeError is an expected judgment failure: contained at DEBUG, never WARNING."""
        caplog.set_level(logging.DEBUG, logger=PARSER_LOGGER_NAME)

        with pytest.raises(PropertyValueRejection):
            FeatureChainParser.match_configuration_feature_chain_parser(
                VALIDATOR_FEATURE_ESC763,
                Options(context={VALIDATOR_KEY_ESC763: [TYPE_POISON_VALUE_TIER763]}),
                property_mapping=RaisingKeyErrorValidatorFeatureGroupEsc763.PROPERTY_MAPPING,
                prefix_patterns=[RaisingKeyErrorValidatorFeatureGroupEsc763.PREFIX_PATTERN],
            )

        assert _records_at_or_above(caplog, PARSER_LOGGER_NAME, logging.WARNING) == []
        debug_records = _records_at_exactly(caplog, PARSER_LOGGER_NAME, logging.DEBUG)
        assert debug_records, "the contained TypeError must still log its rejection at DEBUG"
        # DEBUG is the debugging tier: the containment record keeps both the key and the value visible.
        assert any(
            VALIDATOR_KEY_ESC763 in record.getMessage() and TYPE_POISON_VALUE_TIER763 in record.getMessage()
            for record in debug_records
        ), "the DEBUG containment record must name the property key and keep the rejected value visible"

    def test_keyerror_guard_raise_logs_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A guard raising KeyError is still a non-match, but the containment surfaces at WARNING."""
        caplog.set_level(logging.DEBUG, logger=MIXIN_LOGGER_NAME)
        options = Options(context={GUARD_KEY_ESC763: POISON_VALUE_ESC763})

        result = RaisingKeyErrorGuardFeatureGroupEsc763.match_feature_group_criteria(GUARD_FEATURE_ESC763, options)

        assert result is False
        warnings = _records_at_or_above(caplog, MIXIN_LOGGER_NAME, logging.WARNING)
        assert warnings, "a KeyError-raising guard looks broken and must log at WARNING"
        assert any(GUARD_KEY_ESC763 in record.getMessage() for record in warnings)
        # The exception text stays for triage; the option value itself is redacted at WARNING (production-visible).
        assert any("match_guard cannot judge the poison value" in record.getMessage() for record in warnings)
        assert all(POISON_VALUE_ESC763 not in record.getMessage() for record in warnings), (
            "option values may carry secrets and must not appear in WARNING-tier containment messages"
        )

    def test_typeerror_guard_raise_logs_at_debug_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """A guard raising TypeError is an expected judgment failure: contained at DEBUG, never WARNING."""
        caplog.set_level(logging.DEBUG, logger=MIXIN_LOGGER_NAME)
        options = Options(context={GUARD_KEY_ESC763: TYPE_POISON_VALUE_TIER763})

        result = RaisingKeyErrorGuardFeatureGroupEsc763.match_feature_group_criteria(GUARD_FEATURE_ESC763, options)

        assert result is False
        assert _records_at_or_above(caplog, MIXIN_LOGGER_NAME, logging.WARNING) == []
        debug_records = _records_at_exactly(caplog, MIXIN_LOGGER_NAME, logging.DEBUG)
        assert debug_records, "the contained TypeError must still log its rejection at DEBUG"
        # DEBUG is the debugging tier: the containment record keeps both the key and the value visible.
        assert any(
            GUARD_KEY_ESC763 in record.getMessage() and TYPE_POISON_VALUE_TIER763 in record.getMessage()
            for record in debug_records
        ), "the DEBUG containment record must name the option key and keep the rejected value visible"

    def test_keyerror_required_when_raise_logs_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A predicate raising KeyError is still a non-match, but the containment surfaces at WARNING."""
        caplog.set_level(logging.DEBUG, logger=PARSER_LOGGER_NAME)
        options = Options(context={DRIVER_KEY_ESC763: POISON_VALUE_ESC763})

        result = RaisingKeyErrorRequiredWhenFeatureGroupEsc763.match_feature_group_criteria(
            REQUIRED_WHEN_FEATURE_ESC763, options
        )

        assert result is False
        warnings = _records_at_or_above(caplog, PARSER_LOGGER_NAME, logging.WARNING)
        assert warnings, "a KeyError-raising predicate looks broken and must log at WARNING"
        owner = RaisingKeyErrorRequiredWhenFeatureGroupEsc763.__name__
        assert any(
            owner in record.getMessage() or REQUIRED_WHEN_KEY_ESC763 in record.getMessage() for record in warnings
        )

    def test_typeerror_required_when_raise_logs_at_debug_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """A predicate raising TypeError is an expected judgment failure: contained at DEBUG, never WARNING."""
        caplog.set_level(logging.DEBUG, logger=PARSER_LOGGER_NAME)
        options = Options(context={DRIVER_KEY_ESC763: TYPE_POISON_VALUE_TIER763})

        result = RaisingKeyErrorRequiredWhenFeatureGroupEsc763.match_feature_group_criteria(
            REQUIRED_WHEN_FEATURE_ESC763, options
        )

        assert result is False
        assert _records_at_or_above(caplog, PARSER_LOGGER_NAME, logging.WARNING) == []
        debug_records = _records_at_exactly(caplog, PARSER_LOGGER_NAME, logging.DEBUG)
        assert debug_records, "the contained TypeError must still log its rejection at DEBUG"
        owner = RaisingKeyErrorRequiredWhenFeatureGroupEsc763.__name__
        assert any(
            REQUIRED_WHEN_KEY_ESC763 in record.getMessage() or owner in record.getMessage() for record in debug_records
        ), "the DEBUG containment record must name the guarded key or the owning feature group"


class TestValidatorCauseChaining:
    """The PropertyValueRejection a raising validator produces chains the validator's own raise as __cause__.

    The inline mapping records the exact exception instance the validator raised, so the tests assert
    identity, not just class and message. No FeatureGroup class is defined, so nothing enters the registry.
    """

    def test_keyerror_validator_cause_is_the_original_exception(self) -> None:
        """The rejection's __cause__ is the very KeyError instance the validator raised."""
        raised: list[KeyError] = []

        def validator(value: Any) -> bool:
            exc = KeyError("tier763 validator cannot judge this value")
            raised.append(exc)
            raise exc

        mapping = {
            CAUSE_KEY_TIER763: PropertySpec(
                "Raise whose instance must become __cause__ (tier763 fixture)",
                context=True,
                strict_validation=True,
                element_validator=validator,
            )
        }

        with pytest.raises(PropertyValueRejection) as exc_info:
            FeatureChainParser.match_configuration_feature_chain_parser(
                VALIDATOR_FEATURE_ESC763,
                Options(context={CAUSE_KEY_TIER763: [POISON_VALUE_ESC763]}),
                property_mapping=mapping,
                prefix_patterns=[RaisingKeyErrorValidatorFeatureGroupEsc763.PREFIX_PATTERN],
            )

        assert len(raised) == 1
        assert exc_info.value.__cause__ is raised[0]

    def test_typeerror_validator_cause_is_the_original_exception(self) -> None:
        """The rejection's __cause__ is the very TypeError instance the validator raised."""
        raised: list[TypeError] = []

        def validator(value: Any) -> bool:
            exc = TypeError("tier763 validator expected judgment failure")
            raised.append(exc)
            raise exc

        mapping = {
            CAUSE_KEY_TIER763: PropertySpec(
                "Raise whose instance must become __cause__ (tier763 fixture)",
                context=True,
                strict_validation=True,
                element_validator=validator,
            )
        }

        with pytest.raises(PropertyValueRejection) as exc_info:
            FeatureChainParser.match_configuration_feature_chain_parser(
                VALIDATOR_FEATURE_ESC763,
                Options(context={CAUSE_KEY_TIER763: [TYPE_POISON_VALUE_TIER763]}),
                property_mapping=mapping,
                prefix_patterns=[RaisingKeyErrorValidatorFeatureGroupEsc763.PREFIX_PATTERN],
            )

        assert len(raised) == 1
        assert exc_info.value.__cause__ is raised[0]
