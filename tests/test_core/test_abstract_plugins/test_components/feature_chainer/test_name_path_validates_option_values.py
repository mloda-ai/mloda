"""The string-named path validates the option values it carries (issue #732).

``match_configuration_feature_chain_parser`` returned True as soon as the feature name
matched a PREFIX_PATTERN and never looked at ``options``. Every PROPERTY_MAPPING key that
is NOT encoded in the name (a solver, a method, a size) therefore went unvalidated on the
string-named path, including keys declaring ``strict_validation=True``.

What this module pins:

* On a name match, the option values that ARE present are validated exactly as on the
  config-based path: membership against ``allowed_values``, or the ``element_validator``
  when the spec declares one.
* Required-PRESENCE stays OFF on the name path. A key the name carries (the operation) is
  satisfied by the name, so a name-matched feature with no options at all still matches.
  The config-based path keeps enforcing presence.
* The verdict is a non-match, not a hard raise: the parser raises ``ValueError`` and
  ``match_feature_group_criteria`` swallows it into ``False``, the same as on the
  config-based path.
* ``_strict_validation_rejection_reason`` surfaces that discarded message for a name-matched
  feature, naming the key and the rejected value.
* ``match_guard`` semantics are untouched: raw whole value, no ``strict_validation``
  requirement, falsy verdict is a plain non-match with nothing to report.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.options import Options


def _is_positive_int(value: Any) -> bool:
    """Accept a strictly positive int (mirrors the plugins' positive-digit validators)."""
    return isinstance(value, int) and value > 0


def _is_list_of_str(value: Any) -> bool:
    """Accept the raw whole value only when it is a list of strings."""
    return isinstance(value, list) and all(isinstance(element, str) for element in value)


class NamePathFeatureGroup(FeatureChainParserMixin):
    """Name carries ``algorithm``; ``solver``, ``size`` and ``notes`` are options-only."""

    PREFIX_PATTERN = r".*__([\w]+)_(\d+)d$"

    PROPERTY_MAPPING: dict[str, dict[Any, Any]] = {
        "algorithm": {
            DefaultOptionKeys.allowed_values: {"pca": "PCA", "tsne": "t-SNE"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "solver": {
            "explanation": "strict, allowed_values, never encoded in the name",
            DefaultOptionKeys.allowed_values: {"auto": "Auto", "arpack": "ARPACK"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: "auto",
        },
        "size": {
            "explanation": "strict, element_validator, never encoded in the name",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.element_validator: _is_positive_int,
        },
        "notes": {
            "explanation": "non-strict, so no value space is enforced",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: "",
        },
    }


class GuardedNamePathFeatureGroup(FeatureChainParserMixin):
    """A match_guard key on a name-matched feature group."""

    PREFIX_PATTERN = r".*__([\w]+)_guarded$"

    PROPERTY_MAPPING: dict[str, dict[Any, Any]] = {
        "columns": {
            "explanation": "guarded, non-strict: the guard sees the raw whole value",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.match_guard: _is_list_of_str,
        }
    }


class MalformedNameFeatureGroup(FeatureChainParserMixin):
    """PREFIX_PATTERN matches names with no chain separator, so parsing raises."""

    PREFIX_PATTERN = r"^orphan_(\w+)$"

    PROPERTY_MAPPING: dict[str, dict[Any, Any]] = {
        "solver": {
            DefaultOptionKeys.allowed_values: {"auto": "Auto", "arpack": "ARPACK"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: "auto",
        }
    }


class TestNamePathRejectsBadOptionValues:
    """A name match no longer waves the options through unvalidated."""

    def test_membership_rejection_on_name_path(self) -> None:
        """A strict allowed_values key with a value outside the set is a non-match."""
        result = NamePathFeatureGroup.match_feature_group_criteria("f0__pca_2d", Options(context={"solver": "bogus"}))

        assert result is False

    def test_element_validator_rejection_on_name_path(self) -> None:
        """A strict element_validator key rejecting the value is a non-match."""
        result = NamePathFeatureGroup.match_feature_group_criteria("f0__pca_2d", Options(context={"size": -1}))

        assert result is False

    def test_name_path_verdict_equals_config_path_verdict(self) -> None:
        """The same bad value is rejected identically on both paths: False, not a raise."""
        config_options = Options(context={"algorithm": "pca", "size": 2, "solver": "bogus"})
        name_options = Options(context={"solver": "bogus"})

        config_result = NamePathFeatureGroup.match_feature_group_criteria("placeholder", config_options)
        name_result = NamePathFeatureGroup.match_feature_group_criteria("f0__pca_2d", name_options)

        assert config_result is False
        assert name_result is False

    def test_parser_raises_valueerror_on_name_path(self) -> None:
        """The parser signals the rejection with a ValueError; the mixin is what swallows it."""
        with pytest.raises(ValueError, match="bogus"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "f0__pca_2d",
                Options(context={"solver": "bogus"}),
                property_mapping=NamePathFeatureGroup.PROPERTY_MAPPING,
                prefix_patterns=[NamePathFeatureGroup.PREFIX_PATTERN],
            )


class TestNamePathStillAcceptsValidOptions:
    """Guard against over-rejecting: only bad values are rejected, nothing else."""

    def test_valid_options_only_value_still_matches(self) -> None:
        """A member of the strict value space on a name-matched feature still matches."""
        result = NamePathFeatureGroup.match_feature_group_criteria("f0__pca_2d", Options(context={"solver": "arpack"}))

        assert result is True

    def test_valid_element_validator_value_still_matches(self) -> None:
        """A value the element_validator accepts on a name-matched feature still matches."""
        result = NamePathFeatureGroup.match_feature_group_criteria("f0__pca_2d", Options(context={"size": 3}))

        assert result is True

    def test_non_strict_key_accepts_any_value(self) -> None:
        """strict_validation=False declares no value space, so nothing is enforced."""
        result = NamePathFeatureGroup.match_feature_group_criteria(
            "f0__pca_2d", Options(context={"notes": "!!! not a member of anything !!!"})
        )

        assert result is True

    def test_plain_name_match_without_options_still_matches(self) -> None:
        """Required-PRESENCE stays off on the name path: no options at all still matches."""
        result = NamePathFeatureGroup.match_feature_group_criteria("f0__pca_2d", Options())

        assert result is True

    def test_unrelated_name_is_a_plain_non_match(self) -> None:
        """A name that matches no pattern and carries no options is still just a non-match."""
        result = NamePathFeatureGroup.match_feature_group_criteria("unrelated_feature", Options())

        assert result is False


class TestConfigPathPresenceUnchanged:
    """Turning validation ON for the name path must not turn presence OFF for the config path."""

    def test_config_path_still_requires_present_keys(self) -> None:
        """A required key with no default and no name to carry it still rejects the config path."""
        result = NamePathFeatureGroup.match_feature_group_criteria("placeholder", Options(context={"algorithm": "pca"}))

        assert result is False

    def test_config_path_matches_when_required_keys_present(self) -> None:
        """All required keys present with valid values: the config path matches."""
        result = NamePathFeatureGroup.match_feature_group_criteria(
            "placeholder", Options(context={"algorithm": "pca", "size": 2})
        )

        assert result is True


class TestNamePathRejectionReason:
    """The rejection explains itself instead of vanishing into a bare False."""

    def test_reason_names_key_and_value_for_membership_rejection(self) -> None:
        """The discarded ValueError message is surfaced for a name-matched feature."""
        options = Options(context={"solver": "bogus"})

        assert NamePathFeatureGroup.match_feature_group_criteria("f0__pca_2d", options) is False

        reason = NamePathFeatureGroup._strict_validation_rejection_reason("f0__pca_2d", options)

        assert reason is not None
        assert "solver" in reason
        assert "bogus" in reason

    def test_reason_names_key_and_value_for_element_validator_rejection(self) -> None:
        """The element_validator message is surfaced for a name-matched feature too."""
        options = Options(context={"size": -1})

        reason = NamePathFeatureGroup._strict_validation_rejection_reason("f0__pca_2d", options)

        assert reason is not None
        assert "size" in reason
        assert "-1" in reason

    def test_no_reason_when_nothing_was_rejected(self) -> None:
        """A name match with a valid value has nothing to report."""
        options = Options(context={"solver": "arpack"})

        reason = NamePathFeatureGroup._strict_validation_rejection_reason("f0__pca_2d", options)

        assert reason is None

    def test_no_reason_for_bare_name_match(self) -> None:
        """A name match with no options has nothing to report."""
        reason = NamePathFeatureGroup._strict_validation_rejection_reason("f0__pca_2d", Options())

        assert reason is None

    def test_no_reason_for_malformed_name_parse_error(self) -> None:
        """A ValueError from PARSING a malformed name is not an option-value rejection."""
        assert MalformedNameFeatureGroup.match_feature_group_criteria("orphan_thing", Options()) is False

        reason = MalformedNameFeatureGroup._strict_validation_rejection_reason("orphan_thing", Options())

        assert reason is None


class TestMatchGuardSemanticsUnchanged:
    """match_guard keeps its own contract: raw value, no strict requirement, silent non-match."""

    def test_guard_sees_the_raw_whole_value(self) -> None:
        """A list value reaches the guard whole, not unpacked into elements."""
        result = GuardedNamePathFeatureGroup.match_feature_group_criteria(
            "f0__x_guarded", Options(context={"columns": ["a", "b"]})
        )

        assert result is True

    def test_falsy_guard_is_a_plain_non_match(self) -> None:
        """A rejecting guard yields False without raising, even though the key is not strict."""
        result = GuardedNamePathFeatureGroup.match_feature_group_criteria(
            "f0__x_guarded", Options(context={"columns": "not_a_list"})
        )

        assert result is False

    def test_guard_rejection_has_nothing_to_report(self) -> None:
        """A guard rejection is not a strict_validation rejection, so there is no reason."""
        options = Options(context={"columns": "not_a_list"})

        reason = GuardedNamePathFeatureGroup._strict_validation_rejection_reason("f0__x_guarded", options)

        assert reason is None
