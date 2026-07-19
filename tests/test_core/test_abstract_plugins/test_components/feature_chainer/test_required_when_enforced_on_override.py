"""required_when must be enforced no matter who owns match_feature_group_criteria (issue #731).

Today the predicates only run inside ``FeatureChainParserMixin.match_feature_group_criteria``.
Overriding that method (a supported thing) silently drops the conditional-requirement contract.
The enforcement therefore has to be installed on the class at definition time, exactly once,
and it must reach both the mixin matcher and the default FeatureGroup matcher.
"""

from __future__ import annotations

import gc
import re
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    NAME_PATH_PRESENCE_GUARD_FLAG,
    REQUIRED_WHEN_GUARD_FLAG,
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec

OP_TYPE = "op_type"
ORDER_BY = "order_by"
GUARDED_PATTERN = r".*__([\w]+)_guarded$"
CUSTOM_SEPARATOR_PATTERN = r".*::([\w]+)_custom$"
COMPILED_PATTERN = re.compile(r".*__([\w]+)_compiled$")


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Reclaim the throwaway FeatureGroup subclasses defined inside the tests below.

    They sit in reference cycles, so they linger in ``FeatureGroup.__subclasses__()`` until a GC
    cycle runs, and other tests enumerate that registry.
    """
    yield
    gc.collect()
    gc.collect()


class CountingPredicate:
    """required_when predicate that records how often it ran: order_by is required for op_type 'first'."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, options: Options) -> bool:
        self.calls += 1
        return bool(options.get(OP_TYPE) == "first")


def _mapping(predicate: CountingPredicate) -> dict[str, PropertySpec]:
    """PROPERTY_MAPPING with a conditionally required order_by. op_type stays unconditionally required."""
    return {
        OP_TYPE: PropertySpec(
            "Operation to apply",
            allowed_values={"sum": "Sum of values", "first": "First value (requires order_by)"},
            context=True,
            strict_validation=True,
        ),
        ORDER_BY: PropertySpec(
            "Column to order by",
            context=True,
            strict_validation=False,
            required_when=predicate,
        ),
    }


class NameSuppliedPredicate:
    """required_when predicate on the very key the feature name supplies: op_type is required unless order_by is."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, options: Options) -> bool:
        self.calls += 1
        return options.get(ORDER_BY) is None


def _name_supplied_mapping(predicate: NameSuppliedPredicate) -> dict[str, PropertySpec]:
    """PROPERTY_MAPPING whose conditionally required key is the one the feature name parses into."""
    return {
        OP_TYPE: PropertySpec(
            "Operation to apply",
            allowed_values={"sum": "Sum of values", "first": "First value"},
            context=True,
            strict_validation=True,
            required_when=predicate,
        ),
    }


REQUIRES_ORDER_BY = Options(context={OP_TYPE: "first"})
SATISFIED = Options(context={OP_TYPE: "first", ORDER_BY: "ts"})
NOT_REQUIRED = Options(context={OP_TYPE: "sum"})
ONLY_ORDER_BY = Options(context={ORDER_BY: "ts"})


class TestOverriddenMatcher:
    """A feature group that overrides the matcher keeps its required_when contract."""

    def test_non_delegating_override_rejects_when_required_option_absent(self) -> None:
        """The override never calls the mixin, so the enforcement cannot live inside the mixin matcher."""
        predicate = CountingPredicate()

        class NonDelegatingOverride(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return FeatureChainParser.match_configuration_feature_chain_parser(
                    feature_name,
                    options,
                    property_mapping=cls.PROPERTY_MAPPING,
                    prefix_patterns=[cls.PREFIX_PATTERN],
                )

        assert NonDelegatingOverride.match_feature_group_criteria("x__first_guarded", REQUIRES_ORDER_BY) is False
        assert predicate.calls == 1

    def test_non_delegating_override_accepts_when_required_option_present(self) -> None:
        """Enforcement must not turn into blanket rejection."""
        predicate = CountingPredicate()

        class NonDelegatingOverride(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return FeatureChainParser.match_configuration_feature_chain_parser(
                    feature_name,
                    options,
                    property_mapping=cls.PROPERTY_MAPPING,
                    prefix_patterns=[cls.PREFIX_PATTERN],
                )

        assert NonDelegatingOverride.match_feature_group_criteria("x__first_guarded", SATISFIED) is True
        assert NonDelegatingOverride.match_feature_group_criteria("x__sum_guarded", NOT_REQUIRED) is True

    def test_delegating_override_evaluates_predicate_exactly_once(self) -> None:
        """One enforcement site: the guard, not the guard plus an inline call inside the mixin matcher."""
        predicate = CountingPredicate()

        class DelegatingOverride(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return super().match_feature_group_criteria(feature_name, options, data_access_collection)

        assert DelegatingOverride.match_feature_group_criteria("x__first_guarded", REQUIRES_ORDER_BY) is False
        assert predicate.calls == 1

    def test_subclass_of_override_is_enforced_once(self) -> None:
        """Inheriting a guarded matcher must not stack a second guard on top of it."""
        predicate = CountingPredicate()

        class Parent(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return FeatureChainParser.match_configuration_feature_chain_parser(
                    feature_name,
                    options,
                    property_mapping=cls.PROPERTY_MAPPING,
                    prefix_patterns=[cls.PREFIX_PATTERN],
                )

        class Child(Parent):
            """Does not redefine the matcher: it inherits the already guarded one."""

        assert Child.match_feature_group_criteria("x__first_guarded", REQUIRES_ORDER_BY) is False
        assert predicate.calls == 1


class TestMatcherVariants:
    """The guard reaches every matcher a feature group can end up with."""

    def test_default_feature_group_matcher_is_enforced(self) -> None:
        """A plain FeatureGroup (no mixin) matches by class name and must still honor required_when."""
        predicate = CountingPredicate()

        class DefaultMatcherFeatureGroup(FeatureGroup):
            PROPERTY_MAPPING = _mapping(predicate)

        name = DefaultMatcherFeatureGroup.get_class_name()
        assert DefaultMatcherFeatureGroup.match_feature_group_criteria(name, REQUIRES_ORDER_BY) is False
        assert predicate.calls == 1
        assert DefaultMatcherFeatureGroup.match_feature_group_criteria(name, SATISFIED) is True

    def test_standalone_mixin_override_is_enforced(self) -> None:
        """The mixin is usable without FeatureGroup, so it must install the guard itself."""
        predicate = CountingPredicate()

        class StandaloneMixinOverride(FeatureChainParserMixin):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        assert StandaloneMixinOverride.match_feature_group_criteria("x__first_guarded", REQUIRES_ORDER_BY) is False
        assert predicate.calls == 1
        assert StandaloneMixinOverride.match_feature_group_criteria("x__first_guarded", SATISFIED) is True


class TestGuardInstallation:
    """The guard is installed at class definition time, and only where it is declared."""

    def test_required_when_wraps_the_inherited_matcher(self) -> None:
        """A class declaring required_when carries its own guarded matcher, even without overriding one."""
        predicate = CountingPredicate()

        class InheritsMixinMatcher(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

        resolved = InheritsMixinMatcher.match_feature_group_criteria.__func__  # type: ignore[attr-defined]
        assert resolved is not FeatureChainParserMixin.match_feature_group_criteria.__func__  # type: ignore[attr-defined]
        assert InheritsMixinMatcher.match_feature_group_criteria("x__first_guarded", REQUIRES_ORDER_BY) is False
        assert predicate.calls == 1

    def test_no_required_when_means_no_required_when_guard(self) -> None:
        """No conditional requirement declared means no required_when guard on the resolved matcher.

        The unconditionally required op_type key still earns the name-path presence guard (#769),
        so a wrapper IS installed; only the required_when flag must stay absent.
        """

        class NoRequiredWhen(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = {
                OP_TYPE: PropertySpec(
                    "Operation to apply",
                    allowed_values={"sum": "Sum of values"},
                    context=True,
                    strict_validation=True,
                ),
            }

        resolved = NoRequiredWhen.match_feature_group_criteria.__func__  # type: ignore[attr-defined]
        assert getattr(resolved, REQUIRED_WHEN_GUARD_FLAG, False) is False
        # The wrapper that is present is the presence guard, not a mislabeled required_when guard.
        assert getattr(resolved, NAME_PATH_PRESENCE_GUARD_FLAG, False) is True

    def test_no_flaggable_required_key_installs_no_guard_at_all(self) -> None:
        """A defaulted-only mapping (in_features is name-satisfied) gives neither guard a job."""

        class AllDefaultedNoGuard(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = {
                OP_TYPE: PropertySpec(
                    "Operation to apply",
                    allowed_values={"sum": "Sum of values"},
                    context=True,
                    strict_validation=True,
                    default="sum",
                ),
                DefaultOptionKeys.in_features: PropertySpec("source", context=True, strict_validation=False),
            }

        assert "match_feature_group_criteria" not in AllDefaultedNoGuard.__dict__


class TestGuardAnswersInsteadOfRaising:
    """A matcher answers True or False. The guard must never turn a verdict into an exception."""

    def test_custom_separator_matcher_keeps_its_verdict(self) -> None:
        """match_configuration_feature_chain_parser takes a custom pattern; the guard reparses with '__'.

        The name is unparseable under CHAIN_SEPARATOR, which only means there is no name-parsed value
        to merge: the predicates then see the explicit options, and the matcher's verdict stands.
        """
        predicate = CountingPredicate()

        class CustomSeparatorOverride(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = CUSTOM_SEPARATOR_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return FeatureChainParser.match_configuration_feature_chain_parser(
                    feature_name,
                    options,
                    property_mapping=cls.PROPERTY_MAPPING,
                    prefix_patterns=[cls.PREFIX_PATTERN],
                    pattern="::",
                )

        assert CustomSeparatorOverride.match_feature_group_criteria("x::first_custom", ONLY_ORDER_BY) is True
        # The contract still holds on the options the guard can read: op_type 'first' needs order_by.
        assert CustomSeparatorOverride.match_feature_group_criteria("x::first_custom", REQUIRES_ORDER_BY) is False


class TestStaticMethodMatcherRejected:
    """The guard reinstalls the matcher as a classmethod, so a staticmethod matcher must not reach it."""

    def test_staticmethod_matcher_with_required_when_is_rejected_at_class_definition(self) -> None:
        """Wrapping a staticmethod injects cls as the first argument, so the matcher would misread its own
        arguments and return a silently wrong verdict. Reject loudly, at class definition."""
        predicate = CountingPredicate()

        with pytest.raises(ValueError) as excinfo:

            class StaticMatcherFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = GUARDED_PATTERN
                PROPERTY_MAPPING = _mapping(predicate)

                @staticmethod
                def match_feature_group_criteria(
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    return True

        message = str(excinfo.value)
        assert "StaticMatcherFeatureGroup" in message
        assert "classmethod" in message

    def test_staticmethod_matcher_without_required_when_is_left_alone(self) -> None:
        """Nothing to enforce means nothing to install: a staticmethod matcher stays a valid choice."""

        class StaticMatcherNoContract(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = {
                OP_TYPE: PropertySpec(
                    "Operation to apply",
                    allowed_values={"sum": "Sum of values"},
                    context=True,
                    strict_validation=True,
                ),
            }

            @staticmethod
            def match_feature_group_criteria(
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        assert StaticMatcherNoContract.match_feature_group_criteria("x__sum_guarded", NOT_REQUIRED) is True


class TestExactlyOnceAcrossInheritance:
    """One enforcement site per match call, including when the delegation target is itself guarded."""

    def test_delegating_child_of_a_guarded_parent_evaluates_the_predicate_once(self) -> None:
        """The parent declares required_when and keeps the inherited matcher, so the parent carries the guard.
        A child that overrides the matcher and delegates into the parent must not stack a second guard."""
        predicate = CountingPredicate()

        class GuardedParent(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

        class DelegatingChild(GuardedParent):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return super().match_feature_group_criteria(feature_name, options, data_access_collection)

        assert DelegatingChild.match_feature_group_criteria("x__first_guarded", SATISFIED) is True
        assert predicate.calls == 1
        assert DelegatingChild.match_feature_group_criteria("x__first_guarded", REQUIRES_ORDER_BY) is False


class TestPatternDiscovery:
    """The guard must collect the same patterns the matcher matched on."""

    def test_compiled_prefix_pattern_still_supplies_the_name_parsed_value(self) -> None:
        """re.match accepts a compiled pattern, so the mixin matches on it. The guard must see it too, or the
        name-parsed value never reaches the key that requires it and the feature is wrongly rejected."""
        string_predicate = NameSuppliedPredicate()
        compiled_predicate = NameSuppliedPredicate()

        class StringPatternGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = COMPILED_PATTERN.pattern
            PROPERTY_MAPPING = _name_supplied_mapping(string_predicate)

        class CompiledPatternGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = COMPILED_PATTERN
            PROPERTY_MAPPING = _name_supplied_mapping(compiled_predicate)

        assert StringPatternGroup.match_feature_group_criteria("x__first_compiled", Options()) is True
        assert CompiledPatternGroup.match_feature_group_criteria("x__first_compiled", Options()) is True
