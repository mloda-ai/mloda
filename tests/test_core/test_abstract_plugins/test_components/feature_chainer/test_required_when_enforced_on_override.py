"""required_when must be enforced no matter who owns match_feature_group_criteria (issue #731).

Today the predicates only run inside ``FeatureChainParserMixin.match_feature_group_criteria``.
Overriding that method (a supported thing) silently drops the conditional-requirement contract.
The enforcement therefore has to be installed on the class at definition time, exactly once,
and it must reach both the mixin matcher and the default FeatureGroup matcher.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DefaultOptionKeys

OP_TYPE = "op_type"
ORDER_BY = "order_by"
GUARDED_PATTERN = r".*__([\w]+)_guarded$"


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


def _mapping(predicate: CountingPredicate) -> dict[str, Any]:
    """PROPERTY_MAPPING with a conditionally required order_by. op_type stays unconditionally required."""
    return {
        OP_TYPE: {
            DefaultOptionKeys.allowed_values: {"sum": "Sum of values", "first": "First value (requires order_by)"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        ORDER_BY: {
            "explanation": "Column to order by",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.required_when: predicate,
        },
    }


REQUIRES_ORDER_BY = Options(context={OP_TYPE: "first"})
SATISFIED = Options(context={OP_TYPE: "first", ORDER_BY: "ts"})
NOT_REQUIRED = Options(context={OP_TYPE: "sum"})


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

    def test_no_required_when_leaves_the_matcher_untouched(self) -> None:
        """No conditional requirement declared means nothing to enforce: no wrapper is installed."""

        class NoRequiredWhen(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = GUARDED_PATTERN
            PROPERTY_MAPPING = {
                OP_TYPE: {
                    DefaultOptionKeys.allowed_values: {"sum": "Sum of values"},
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                },
            }

        assert "match_feature_group_criteria" not in NoRequiredWhen.__dict__
        resolved = NoRequiredWhen.match_feature_group_criteria.__func__  # type: ignore[attr-defined]
        assert resolved is FeatureChainParserMixin.match_feature_group_criteria.__func__  # type: ignore[attr-defined]
