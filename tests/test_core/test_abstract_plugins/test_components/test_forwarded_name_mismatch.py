"""
Failing tests pinning the forwarded-name-mismatch check contract.

FeatureChainParserMixin.match_feature_group_criteria, when the feature matches
this group via string parsing (a name-parsed operation value exists and maps to
a PROPERTY_MAPPING key K), must additionally check: if the feature's options
carry K, AND K is in options.inherited_group_keys (it arrived via consumer
forwarding, not set by the author), AND str(option value) differs from the
name-parsed value, then raise ValueError. The message contains the feature
name, the key K, both values, and the remedy string "forward_group_exclude",
plus a mention of the env var MLODA_ALLOW_FORWARDED_NAME_MISMATCH. If that env
var is set to "1" or "true", the check logs ONE logging WARNING (same content)
instead of raising and matching proceeds normally.

No behavior change when: values are equal, K was set by the author (not
inherited), the feature is config-based (no string parse), or K is absent from
the options.

All fixture names carry a "namemis579" marker so they cannot collide with
other tests in the global plugin registry.
"""

from __future__ import annotations

import logging

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.provider import PropertySpec
from mloda.user import Options


OPERATION_KEY = "operation_namemis579"
STRING_FEATURE_NAME = "sales__sum_namemis579"


class _NameMismatchChainedGroup(FeatureChainParserMixin):
    """Minimal chainer subclass following the existing mixin test fixture style.

    The operation key is group-categorized on purpose (context=False): only group
    options flow through consumer forwarding.
    """

    PREFIX_PATTERN = r".*__(sum|max)_namemis579$"
    PROPERTY_MAPPING = {
        OPERATION_KEY: PropertySpec(
            "Operation of the namemis579 fixture",
            allowed_values={
                "sum": "Sum of the in feature (namemis579 fixture)",
                "max": "Maximum of the in feature (namemis579 fixture)",
            },
            context=False,
            strict_validation=True,
        )
    }


def _inherited_child_options(consumer_group: dict[str, str]) -> Options:
    """Build child options exactly like the engine does: inherit_from the consumer."""
    child_options = Options()
    child_options.inherit_from(Options(group=consumer_group))
    return child_options


class TestFixtureSanity:
    def test_string_feature_matches_with_empty_options(self) -> None:
        """Precondition: the fixture group claims the chained name via string parsing."""
        assert _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, Options()) is True


class TestForwardedNameMismatch:
    def test_inherited_differing_value_raises(self) -> None:
        """An inherited option value differing from the name-parsed value must raise."""
        child_options = _inherited_child_options({OPERATION_KEY: "max"})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        message = str(exc_info.value)
        assert STRING_FEATURE_NAME in message
        assert OPERATION_KEY in message
        assert "sum" in message
        assert "max" in message
        assert "forward_group_exclude" in message
        assert "MLODA_ALLOW_FORWARDED_NAME_MISMATCH" in message

    def test_inherited_equal_value_matches(self) -> None:
        """An inherited option value equal to the name-parsed value matches silently."""
        child_options = _inherited_child_options({OPERATION_KEY: "sum"})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True

    def test_author_set_differing_value_does_not_raise(self) -> None:
        """An author-set option value (not inherited) keeps today's silent name precedence."""
        child_options = Options(group={OPERATION_KEY: "max"})
        assert child_options.inherited_group_keys == frozenset()  # precondition: nothing inherited

        result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True

    def test_env_var_downgrades_to_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1 the check warns once and matching proceeds."""
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        child_options = _inherited_child_options({OPERATION_KEY: "max"})

        with caplog.at_level(logging.WARNING):
            result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True
        records = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and OPERATION_KEY in record.getMessage()
        ]
        assert len(records) == 1, (
            f"Expected exactly one WARNING mentioning '{OPERATION_KEY}', got {len(records)}: "
            f"{[record.getMessage() for record in records]}"
        )
        message = records[0].getMessage()
        assert "sum" in message
        assert "max" in message

    def test_config_based_feature_unaffected(self) -> None:
        """A config-based feature (no string parse) keeps today's matching, no raise."""
        child_options = _inherited_child_options({OPERATION_KEY: "max"})

        result = _NameMismatchChainedGroup.match_feature_group_criteria("namemis579_config_feature", child_options)

        assert result is True

    def test_absent_key_unaffected(self) -> None:
        """A string feature whose options lack the key (but carry another inherited key) matches."""
        child_options = _inherited_child_options({"other_key_namemis579": "x"})
        assert child_options.inherited_group_keys == frozenset({"other_key_namemis579"})  # precondition

        result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True
