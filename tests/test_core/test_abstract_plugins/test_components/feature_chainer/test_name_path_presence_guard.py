"""Name-path required-presence must survive a matcher override (issue #769 review).

The rule is enforced inside ``FeatureChainParserMixin.match_parser_criteria``, so a provider that
overrides ``match_feature_group_criteria`` and returns True WITHOUT delegating bypasses it entirely.
Like ``required_when`` (``feature_chain_author_guards.install_required_when_guard``), the enforcement must be
a definition-time wrapper around the matcher: it turns the bypass into a warned non-match, honors
the same exemptions as the inner rule, applies only when the name identifies the group, and never
duplicates the inner path's warning.

Every fixture carries an "r769pg" marker in its class name and keys so it cannot collide with other
feature groups in the global registry.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec, property_spec

FEATURE_CHAIN_PARSER_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser"

# The marker substring the presence warning is contractually required to contain.
MARKER = "required option(s)"


def _presence_warnings(caplog: pytest.LogCaptureFixture, class_name: str) -> list[logging.LogRecord]:
    """One class's required-presence WARNINGs: logger name + WARNING level + marker + class name."""
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == FEATURE_CHAIN_PARSER_LOGGER
        and MARKER in record.getMessage()
        and class_name in record.getMessage()
    ]


class _CallableMatcher:
    """A callable-instance matcher: no descriptor, so a classmethod wrap would pass it the class."""

    def __call__(self, *args: Any, **kwargs: Any) -> bool:
        return True


def _permissive_matcher(*args: Any, **kwargs: Any) -> bool:
    return True


class TestOverrideBypass:
    """A non-delegating True override must not bypass the presence rule."""

    def test_non_delegating_true_override_is_still_a_non_match(self, caplog: pytest.LogCaptureFixture) -> None:
        """The override never reaches the inner rule, so the guard must reject and warn itself."""

        class _BypassR769pg1(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769pg1>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769pg1": PropertySpec("required, carried by the name", context=True),
                "missing_r769pg1": PropertySpec("required, options-only, absent", context=True),
            }

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        # Precondition: the missing key is unconditionally required and name-path relevant.
        missing_spec = _BypassR769pg1.PROPERTY_MAPPING["missing_r769pg1"]
        assert FeatureChainParser._can_skip_required_check(missing_spec) is False

        with caplog.at_level(logging.WARNING):
            result = _BypassR769pg1.match_feature_group_criteria("src__op_r769pg1", Options())

        assert result is False, "a non-delegating True override must not bypass the presence rule"
        warnings = _presence_warnings(caplog, "_BypassR769pg1")
        assert warnings, "the guarded non-match must warn, naming the class"
        assert "missing_r769pg1" in warnings[0].getMessage(), "the warning must name the missing key"

    def test_delegating_override_warns_exactly_once(
        self, caplog: pytest.LogCaptureFixture, presence_checks: list[Options]
    ) -> None:
        """The inner path already warns on the delegated call; the guard must not add a second warning."""

        class _DelegatingR769pg2(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769pg2>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769pg2": PropertySpec("required, carried by the name", context=True),
                "missing_r769pg2": PropertySpec("required, options-only, absent", context=True),
            }

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return super().match_feature_group_criteria(feature_name, options, data_access_collection)

        presence_checks.clear()

        with caplog.at_level(logging.WARNING):
            result = _DelegatingR769pg2.match_feature_group_criteria("src__op_r769pg2", Options())

        assert result is False, "the delegated inner path rejects the missing required key"
        warnings = _presence_warnings(caplog, "_DelegatingR769pg2")
        assert len(warnings) == 1, "one enforcement site per match call: exactly one presence warning"
        assert len(presence_checks) == 1, "one enforcement site per match call: exactly one presence check"

    def test_mixin_group_without_override_checks_presence_once(self, presence_checks: list[Options]) -> None:
        class _PlainMixinR769pg30(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(?P<carried_r769pg30>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769pg30": PropertySpec("required, carried by the name", context=True),
                "missing_r769pg30": property_spec("required, options-only, absent"),
            }

        presence_checks.clear()

        assert _PlainMixinR769pg30.match_feature_group_criteria("src__op_r769pg30", Options()) is False
        assert len(presence_checks) == 1

    def test_mixin_group_evaluates_guard_and_validator_once(self) -> None:
        """The default matcher must not add a second validation pass on top of the mixin matcher."""
        guard_calls: list[Any] = []
        validator_calls: list[Any] = []

        def guard(value: Any) -> bool:
            guard_calls.append(value)
            return True

        def validator(value: Any) -> bool:
            validator_calls.append(value)
            return True

        class _CountingMixinR769pg31(FeatureChainParserMixin, FeatureGroup):
            MIN_IN_FEATURES = 0
            PREFIX_PATTERN = r".*__(\w+)_r769pg31$"
            PROPERTY_MAPPING = {
                "counted_r769pg31": property_spec(
                    "required, judged and guarded", strict=True, element_validator=validator, match_guard=guard
                ),
            }

        options = Options(context={"counted_r769pg31": 3})

        assert _CountingMixinR769pg31.match_feature_group_criteria("x__a_r769pg31", options) is True
        assert guard_calls == [3]
        assert validator_calls == [3]


class TestStaticmethodOverrideBypass:
    """A staticmethod matcher must be guarded too: wrapped without injecting cls, never skipped."""

    def test_staticmethod_true_override_is_still_a_non_match(self, caplog: pytest.LogCaptureFixture) -> None:
        """Skipping staticmethod matchers reopens the bypass; the guard must reject and warn here too."""

        class _StaticBypassR769pg6(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769pg6>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769pg6": PropertySpec("required, carried by the name", context=True),
                "missing_r769pg6": PropertySpec("required, options-only, absent", context=True),
            }

            @staticmethod
            def match_feature_group_criteria(
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        with caplog.at_level(logging.WARNING):
            result = _StaticBypassR769pg6.match_feature_group_criteria("src__op_r769pg6", Options())

        assert result is False, "a staticmethod True override must not bypass the presence rule"
        warnings = _presence_warnings(caplog, "_StaticBypassR769pg6")
        assert warnings, "the guarded non-match must warn, naming the class"
        assert "missing_r769pg6" in warnings[0].getMessage(), "the warning must name the missing key"

    def test_staticmethod_override_with_supplied_key_keeps_the_match(self, caplog: pytest.LogCaptureFixture) -> None:
        """The wrapper must preserve the staticmethod's verdict and calling convention: match, no warning."""

        class _StaticSatisfiedR769pg7(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769pg7>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769pg7": PropertySpec("required, carried by the name", context=True),
                "present_r769pg7": PropertySpec("required, present in options", context=True),
            }

            @staticmethod
            def match_feature_group_criteria(
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        with caplog.at_level(logging.WARNING):
            result = _StaticSatisfiedR769pg7.match_feature_group_criteria(
                "src__op_r769pg7", Options(context={"present_r769pg7": "v_r769pg7"})
            )

        assert result is True, "a satisfied required key must keep the staticmethod's match"
        assert not _presence_warnings(caplog, "_StaticSatisfiedR769pg7")

    def test_staticmethod_override_with_deferred_binding_keeps_the_match(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``deferred_binding=True`` exempts the key for a staticmethod matcher exactly as for a classmethod."""

        class _StaticDeferredR769pg8(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769pg8>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769pg8": PropertySpec("required, carried by the name", context=True),
                "deferred_r769pg8": PropertySpec(
                    "required but bound outside the name", context=True, deferred_binding=True
                ),
            }

            @staticmethod
            def match_feature_group_criteria(
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        with caplog.at_level(logging.WARNING):
            result = _StaticDeferredR769pg8.match_feature_group_criteria("src__op_r769pg8", Options())

        assert result is True, "deferred_binding=True must keep the staticmethod's match"
        assert not _presence_warnings(caplog, "_StaticDeferredR769pg8")

    def test_plain_staticmethod_matcher_is_enforced(self) -> None:
        """A plain group (no mixin) with a required key gets the guard on a staticmethod matcher too."""

        class _StaticPlainR769pg32(FeatureGroup):
            PROPERTY_MAPPING = {"needs_key_r769pg32": property_spec("required, no default")}

            @staticmethod
            def match_feature_group_criteria(
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        assert _StaticPlainR769pg32.match_feature_group_criteria("anything", Options()) is False
        assert (
            _StaticPlainR769pg32.match_feature_group_criteria("anything", Options(context={"needs_key_r769pg32": "v"}))
            is True
        )


class TestBareFunctionMatcherRejected:
    """A bare function matcher (no classmethod/staticmethod descriptor) must be rejected too."""

    def test_bare_function_matcher_is_rejected_at_class_definition(self) -> None:
        """Wrapping it as a classmethod would misread cls as the feature name: reject at class definition."""

        def bare_matcher(
            feature_name: str | FeatureName,
            options: Options,
            data_access_collection: Any = None,
        ) -> bool:
            return True

        with pytest.raises(ValueError) as excinfo:

            class _BareFunctionR769pg20(FeatureChainParserMixin):
                PREFIX_PATTERN = r".*__(?P<carried_r769pg20>\w+)$"
                PROPERTY_MAPPING = {
                    "carried_r769pg20": PropertySpec("required, carried by the name", context=True),
                    "missing_r769pg20": PropertySpec("required, options-only, absent", context=True),
                }
                match_feature_group_criteria = bare_matcher  # type: ignore[assignment]

        message = str(excinfo.value)
        assert "_BareFunctionR769pg20" in message
        assert "classmethod" in message

    @pytest.mark.parametrize(
        "make_matcher",
        [_CallableMatcher, lambda: functools.partial(_permissive_matcher)],
        ids=["callable_instance", "functools_partial"],
    )
    def test_non_function_callable_matcher_is_rejected_at_class_definition(
        self, make_matcher: Callable[[], Callable[..., bool]]
    ) -> None:
        """A callable instance or partial has no descriptor either: reject it like a bare function."""
        with pytest.raises(ValueError) as excinfo:

            class _CallableMatcherR769pg21(FeatureChainParserMixin):
                PREFIX_PATTERN = r".*__(?P<carried_r769pg21>\w+)$"
                PROPERTY_MAPPING = {
                    "carried_r769pg21": PropertySpec("required, carried by the name", context=True),
                    "missing_r769pg21": PropertySpec("required, options-only, absent", context=True),
                }
                match_feature_group_criteria = make_matcher()

        message = str(excinfo.value)
        assert "_CallableMatcherR769pg21" in message
        assert "classmethod" in message

    def test_plain_descriptorless_matcher_is_skipped_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A plain group has no name path to protect: the guard is skipped, and the skip is announced."""

        def bare_matcher(*args: Any, **kwargs: Any) -> bool:
            return True

        with caplog.at_level(logging.WARNING):

            class _BareMatcherPlainR769pg33(FeatureGroup):
                PROPERTY_MAPPING = {"needs_key_r769pg33": property_spec("required, no default")}
                match_feature_group_criteria = bare_matcher

        warned = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and "_BareMatcherPlainR769pg33" in record.getMessage()
        ]
        assert warned, "the skipped guard must be announced with a WARNING naming the class"
        assert _BareMatcherPlainR769pg33.match_feature_group_criteria("anything", Options()) is True


class TestGuardHonorsTheInnerRule:
    """The guard mirrors the inner rule: same exemptions, name-path only, no false rejections."""

    def test_deferred_binding_keeps_the_override_verdict(self, caplog: pytest.LogCaptureFixture) -> None:
        """``deferred_binding=True`` exempts the key inside the guard exactly as inside the inner rule."""

        class _DeferredBypassR769pg3(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769pg3>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769pg3": PropertySpec("required, carried by the name", context=True),
                "deferred_r769pg3": PropertySpec(
                    "required but bound outside the name", context=True, deferred_binding=True
                ),
            }

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        with caplog.at_level(logging.WARNING):
            result = _DeferredBypassR769pg3.match_feature_group_criteria("src__op_r769pg3", Options())

        assert result is True, "deferred_binding=True must keep the override's match"
        assert not _presence_warnings(caplog, "_DeferredBypassR769pg3")

    def test_guard_applies_only_when_the_name_identifies_the_group(self, caplog: pytest.LogCaptureFixture) -> None:
        """A name no pattern matches is not on the name path: the override's True stands."""

        class _NamePathOnlyR769pg4(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769pg4>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769pg4": PropertySpec("required, carried by the name", context=True),
                "missing_r769pg4": PropertySpec("required, options-only, absent", context=True),
            }

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        with caplog.at_level(logging.WARNING):
            result = _NamePathOnlyR769pg4.match_feature_group_criteria("plain_feature_r769pg4", Options())

        assert result is True, "the presence rule is name-path only; the guard must not flip the verdict"
        assert not _presence_warnings(caplog, "_NamePathOnlyR769pg4")

    def test_supplied_required_key_keeps_the_match(self, caplog: pytest.LogCaptureFixture) -> None:
        """A required key present in options satisfies the guard: match, no warning."""

        class _SatisfiedBypassR769pg5(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769pg5>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769pg5": PropertySpec("required, carried by the name", context=True),
                "present_r769pg5": PropertySpec("required, present in options", context=True),
            }

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        with caplog.at_level(logging.WARNING):
            result = _SatisfiedBypassR769pg5.match_feature_group_criteria(
                "src__op_r769pg5", Options(context={"present_r769pg5": "v_r769pg5"})
            )

        assert result is True, "a satisfied required key must keep the override's match"
        assert not _presence_warnings(caplog, "_SatisfiedBypassR769pg5")


class TestMixinWithoutPrefixPattern:
    """A mixin group with no PREFIX_PATTERN has no name path: no guard, the configuration path decides."""

    def test_no_guard_is_installed_and_configuration_presence_decides(self) -> None:
        class _NoPatternR769pg34(FeatureChainParserMixin, FeatureGroup):
            MIN_IN_FEATURES = 0
            PROPERTY_MAPPING = {"needs_key_r769pg34": property_spec("required, no default")}

        assert "match_feature_group_criteria" not in _NoPatternR769pg34.__dict__
        assert _NoPatternR769pg34.match_feature_group_criteria("any_feature_r769pg34", Options()) is False
        assert (
            _NoPatternR769pg34.match_feature_group_criteria(
                "any_feature_r769pg34", Options(context={"needs_key_r769pg34": "v"})
            )
            is True
        )
