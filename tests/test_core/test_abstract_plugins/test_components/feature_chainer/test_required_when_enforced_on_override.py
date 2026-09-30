"""required_when must be enforced no matter who owns match_feature_group_criteria (issue #731).

Today the predicates only run inside ``FeatureChainParserMixin.match_feature_group_criteria``.
Overriding that method (a supported thing) silently drops the conditional-requirement contract.
The enforcement therefore has to be installed on the class at definition time, exactly once,
and it must reach both the mixin matcher and the default FeatureGroup matcher.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer import feature_chain_author_guards
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards import (
    NAME_PATH_PRESENCE_GUARD_FLAG,
    REQUIRED_WHEN_GUARD_FLAG,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.provider import DataCreator, PropertySpec, property_spec
from mloda.user import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

OP_TYPE = "op_type"
ORDER_BY = "order_by"
NEEDS_KEY = "needs_key_pgo"
THRESHOLD_KEY = "threshold_e2e_pgo"
NEEDS_THRESHOLD_FEATURE = "needs_threshold_e2e_pgo"
GUARDED_PATTERN = r".*__([\w]+)_guarded$"
CUSTOM_SEPARATOR_PATTERN = r".*::([\w]+)_custom$"
COMPILED_PATTERN = re.compile(r".*__([\w]+)_compiled$")


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


class _CallableMatcher:
    """A callable-instance matcher: no descriptor, so a classmethod wrap would pass it the class."""

    def __call__(self, *args: Any, **kwargs: Any) -> bool:
        return True


def _permissive_matcher(*args: Any, **kwargs: Any) -> bool:
    return True


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
        assert getattr(resolved, REQUIRED_WHEN_GUARD_FLAG, None) is not resolved
        # The wrapper that is present is the presence guard, not a mislabeled required_when guard.
        assert getattr(resolved, NAME_PATH_PRESENCE_GUARD_FLAG, None) is resolved

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
    """The guard reinstalls the matcher as a classmethod, so a staticmethod or plain-function matcher must not reach it."""

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

    def test_bare_function_matcher_with_required_when_is_rejected_at_class_definition(self) -> None:
        """A bare function would misread cls as its feature_name once wrapped as a classmethod: reject it too."""
        predicate = CountingPredicate()

        def bare_matcher(
            feature_name: str | FeatureName,
            options: Options,
            data_access_collection: Any = None,
        ) -> bool:
            return True

        with pytest.raises(ValueError) as excinfo:

            class BareFunctionMatcherFeatureGroup(FeatureGroup):
                PROPERTY_MAPPING = _mapping(predicate)
                match_feature_group_criteria = bare_matcher  # type: ignore[assignment]

        message = str(excinfo.value)
        assert "BareFunctionMatcherFeatureGroup" in message
        assert "classmethod" in message

    def test_bare_function_matcher_without_guard_is_left_alone(self) -> None:
        """Nothing to enforce means nothing to install: a bare function matcher keeps its own calling convention."""

        def bare_matcher_no_guard(*args: Any, **kwargs: Any) -> bool:
            first = args[0] if args else next(iter(kwargs.values()), None)
            return isinstance(first, str)

        class BareFunctionMatcherNoGuardFeatureGroup(FeatureGroup):
            PROPERTY_MAPPING = {
                OP_TYPE: PropertySpec(
                    "Operation to apply",
                    allowed_values={"sum": "Sum of values"},
                    context=True,
                    strict_validation=True,
                    default="sum",
                ),
            }
            match_feature_group_criteria = bare_matcher_no_guard

        assert BareFunctionMatcherNoGuardFeatureGroup.match_feature_group_criteria("some_name", Options()) is True

    @pytest.mark.parametrize(
        "make_matcher",
        [_CallableMatcher, lambda: functools.partial(_permissive_matcher)],
        ids=["callable_instance", "functools_partial"],
    )
    def test_non_function_callable_matcher_with_required_when_is_rejected_at_class_definition(
        self, make_matcher: Callable[[], Callable[..., bool]]
    ) -> None:
        """A callable instance or partial has no descriptor either: reject it like a bare function."""
        predicate = CountingPredicate()

        with pytest.raises(ValueError) as excinfo:

            class CallableMatcherFeatureGroup(FeatureGroup):
                PROPERTY_MAPPING = _mapping(predicate)
                match_feature_group_criteria = make_matcher()

        message = str(excinfo.value)
        assert "CallableMatcherFeatureGroup" in message
        assert "classmethod" in message

    def test_bound_classmethod_from_another_guarded_group_is_accepted(self) -> None:
        """A bound method keeps its own class binding, so it answers exactly the source group's verdict."""
        predicate = CountingPredicate()

        class SourceGuardedGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

        class BorrowedMatcherGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)
            match_feature_group_criteria = SourceGuardedGroup.match_feature_group_criteria

        for options, expected in ((REQUIRES_ORDER_BY, False), (SATISFIED, True)):
            assert SourceGuardedGroup.match_feature_group_criteria("x__first_guarded", options) is expected
            assert BorrowedMatcherGroup.match_feature_group_criteria("x__first_guarded", options) is expected


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


class TestFunctoolsWrapsOverrideKeepsItsOwnGuard:
    """A genuine override written as ``@classmethod @functools.wraps(<parent's matcher>)``.

    functools.wraps copies the wrapped callable's __dict__ onto the override, so a guard flag the
    parent's matcher carries must not be read as already covering this, independently-bodied, matcher.
    """

    def test_required_when_guard_is_enforced_on_a_wrapped_non_delegating_override(self) -> None:
        predicate = CountingPredicate()

        class GuardedParent(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

        class WrappedOverride(GuardedParent):
            @classmethod
            @functools.wraps(GuardedParent.match_feature_group_criteria)
            def match_feature_group_criteria(  # type: ignore[override]
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                # Genuine, non-delegating body: ignores the predicate entirely.
                return True

        assert WrappedOverride.match_feature_group_criteria("x__first_guarded", REQUIRES_ORDER_BY) is False  # type: ignore[call-arg,arg-type]
        assert WrappedOverride.match_feature_group_criteria("x__first_guarded", SATISFIED) is True  # type: ignore[call-arg,arg-type]

    def test_name_path_presence_guard_is_enforced_on_a_wrapped_non_delegating_override(self) -> None:
        name_path_pattern = r".*__([\w]+)_npguard$"

        class PresenceGuardedParent(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = name_path_pattern
            PROPERTY_MAPPING = {
                OP_TYPE: PropertySpec(
                    "Operation to apply",
                    allowed_values={"sum": "Sum of values"},
                    context=True,
                    strict_validation=True,
                ),
                "missing_npguard": PropertySpec("required, options-only, absent on the name path", context=True),
            }

        class WrappedOverride(PresenceGuardedParent):
            @classmethod
            @functools.wraps(PresenceGuardedParent.match_feature_group_criteria)
            def match_feature_group_criteria(  # type: ignore[override]
                cls,
                feature_name: str | FeatureName,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                # Genuine, non-delegating body: ignores the required key entirely.
                return True

        assert WrappedOverride.match_feature_group_criteria("x__sum_npguard", Options()) is False  # type: ignore[call-arg,arg-type]
        assert (
            WrappedOverride.match_feature_group_criteria(  # type: ignore[call-arg]
                "x__sum_npguard",  # type: ignore[arg-type]
                Options(context={"missing_npguard": "present"}),  # type: ignore[arg-type]
            )
            is True
        )

    def test_plain_subclass_of_a_guarded_matcher_still_stacks_no_extra_wrapper(self) -> None:
        """Regression guard: an ordinary subclass (no override at all) must still get no new wrapper."""
        predicate = CountingPredicate()

        class GuardedParent(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = _mapping(predicate)

        class PlainChild(GuardedParent):
            """No override: inherits the already guarded matcher as-is."""

        assert "match_feature_group_criteria" not in PlainChild.__dict__

        resolved = GuardedParent.match_feature_group_criteria.__func__  # type: ignore[attr-defined]
        assert feature_chain_author_guards._matcher_carries_guard(resolved, REQUIRED_WHEN_GUARD_FLAG)
        assert feature_chain_author_guards._matcher_carries_guard(resolved, NAME_PATH_PRESENCE_GUARD_FLAG)


@pytest.fixture
def presence_checks(monkeypatch: pytest.MonkeyPatch) -> list[Options]:
    """Records the options each missing-required-keys check sees; clear it after defining classes."""
    seen: list[Options] = []
    original = FeatureChainParser._name_path_missing_required_keys

    def spy(cls: type[FeatureChainParser], effective_options: Options, property_mapping: dict[str, Any]) -> list[str]:
        seen.append(effective_options)
        return original(effective_options, property_mapping)

    monkeypatch.setattr(FeatureChainParser, "_name_path_missing_required_keys", classmethod(spy))
    return seen


def _plain_required_mapping() -> dict[str, PropertySpec]:
    return {NEEDS_KEY: property_spec("required, no default")}


class NeedsThresholdFeatureGroup(FeatureGroup):
    """A plain root group whose required key reaches calculate_feature."""

    PROPERTY_MAPPING = {THRESHOLD_KEY: property_spec("required, no default")}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({NEEDS_THRESHOLD_FEATURE})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({NEEDS_THRESHOLD_FEATURE: [repr(features.get_options_key(THRESHOLD_KEY))]})


class TestPlainGroupEnforcement:
    """The presence guard reaches plain groups: default matcher, inherited matcher and override alike."""

    def test_plain_required_key_installs_the_presence_guard_only(self) -> None:
        class PlainWithRequiredKey(FeatureGroup):
            PROPERTY_MAPPING = _plain_required_mapping()

        resolved = PlainWithRequiredKey.match_feature_group_criteria.__func__  # type: ignore[attr-defined]
        assert getattr(resolved, NAME_PATH_PRESENCE_GUARD_FLAG, None) is resolved
        assert getattr(resolved, REQUIRED_WHEN_GUARD_FLAG, None) is not resolved

    def test_plain_group_without_a_flaggable_key_installs_no_guard(self) -> None:
        class PlainAllDefaulted(FeatureGroup):
            PROPERTY_MAPPING = {NEEDS_KEY: property_spec("optional", default=None)}

        class PlainNoMapping(FeatureGroup):
            """No PROPERTY_MAPPING at all."""

        assert "match_feature_group_criteria" not in PlainAllDefaulted.__dict__
        assert "match_feature_group_criteria" not in PlainNoMapping.__dict__

    def test_plain_subclass_inherits_the_guard_without_a_second_wrapper(self) -> None:
        class PlainParent(FeatureGroup):
            PROPERTY_MAPPING = _plain_required_mapping()

        class PlainChild(PlainParent):
            """No override: inherits the guarded matcher."""

        assert "match_feature_group_criteria" not in PlainChild.__dict__
        assert PlainChild.match_feature_group_criteria(PlainChild.get_class_name(), Options()) is False
        assert (
            PlainChild.match_feature_group_criteria(PlainChild.get_class_name(), Options(context={NEEDS_KEY: "v"}))
            is True
        )

    def test_plain_delegating_override_checks_presence_once(self, presence_checks: list[Options]) -> None:
        """The delegating override and its guarded parent evaluate the rule once between them."""

        class PlainParent(FeatureGroup):
            PROPERTY_MAPPING = _plain_required_mapping()

        class DelegatingPlainChild(PlainParent):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return super().match_feature_group_criteria(feature_name, options, data_access_collection)

        name = DelegatingPlainChild.get_class_name()
        presence_checks.clear()

        assert DelegatingPlainChild.match_feature_group_criteria(name, Options()) is False
        assert len(presence_checks) == 1

        presence_checks.clear()
        assert DelegatingPlainChild.match_feature_group_criteria(name, Options(context={NEEDS_KEY: "v"})) is True
        assert len(presence_checks) == 1

    def test_mixin_group_checks_presence_once_when_it_rejects(self, presence_checks: list[Options]) -> None:
        class PatternedMixinGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = {
                OP_TYPE: PropertySpec(
                    "Operation to apply", allowed_values={"sum": "Sum"}, context=True, strict_validation=True
                ),
                NEEDS_KEY: property_spec("required, no default"),
            }

        presence_checks.clear()

        assert PatternedMixinGroup.match_feature_group_criteria("x__sum_guarded", Options()) is False
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

        class CountingMixinGroup(FeatureChainParserMixin, FeatureGroup):
            MIN_IN_FEATURES = 0
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = {
                NEEDS_KEY: property_spec(
                    "required, judged and guarded", strict=True, element_validator=validator, match_guard=guard
                ),
            }

        options = Options(context={NEEDS_KEY: 3})

        assert CountingMixinGroup.match_feature_group_criteria("x__a_guarded", options) is True
        assert guard_calls == [3]
        assert validator_calls == [3]

    def test_plain_staticmethod_matcher_is_enforced(self) -> None:
        class StaticPlainMatcher(FeatureGroup):
            PROPERTY_MAPPING = _plain_required_mapping()

            @staticmethod
            def match_feature_group_criteria(
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: Any = None,
            ) -> bool:
                return True

        assert StaticPlainMatcher.match_feature_group_criteria("anything", Options()) is False
        assert StaticPlainMatcher.match_feature_group_criteria("anything", Options(context={NEEDS_KEY: "v"})) is True

    def test_plain_descriptorless_matcher_is_skipped_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        def bare_matcher(*args: Any, **kwargs: Any) -> bool:
            return True

        with caplog.at_level(logging.WARNING):

            class BareMatcherPlainGroup(FeatureGroup):
                PROPERTY_MAPPING = _plain_required_mapping()
                match_feature_group_criteria = bare_matcher

        warned = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and "BareMatcherPlainGroup" in record.getMessage()
        ]
        assert warned, "the skipped guard must be announced with a WARNING naming the class"
        assert BareMatcherPlainGroup.match_feature_group_criteria("anything", Options()) is True

    def test_descriptorless_matcher_on_a_patterned_group_still_raises(self) -> None:
        def bare_matcher(*args: Any, **kwargs: Any) -> bool:
            return True

        with pytest.raises(ValueError) as excinfo:

            class BareMatcherPatternedGroup(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = GUARDED_PATTERN
                PROPERTY_MAPPING = {
                    OP_TYPE: PropertySpec(
                        "Operation to apply", allowed_values={"sum": "Sum"}, context=True, strict_validation=True
                    ),
                }
                match_feature_group_criteria = bare_matcher

        assert "BareMatcherPatternedGroup" in str(excinfo.value)
        assert "classmethod" in str(excinfo.value)


class TestPlainGroupEndToEnd:
    """A plain root group with a required key is a non-match without it, end to end."""

    def test_missing_required_key_fails_resolution_with_the_reason(self) -> None:
        with pytest.raises(FeatureResolutionError) as exc_info:
            mlodaAPI.run_all(
                [Feature(NEEDS_THRESHOLD_FEATURE)],
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups({NeedsThresholdFeatureGroup}),
            )

        assert (f"  - NeedsThresholdFeatureGroup (option value): required option(s) {THRESHOLD_KEY} are absent") in str(
            exc_info.value
        )

    def test_present_required_key_reaches_calculate_feature(self) -> None:
        results = mlodaAPI.run_all(
            [Feature(NEEDS_THRESHOLD_FEATURE, Options(context={THRESHOLD_KEY: "5"}))],
            compute_frameworks={PyArrowTable},
            plugin_collector=PluginCollector.enabled_feature_groups({NeedsThresholdFeatureGroup}),
        )

        assert results[0].column(NEEDS_THRESHOLD_FEATURE)[0].as_py() == "'5'"
