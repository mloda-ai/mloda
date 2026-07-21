"""``PropertySpec`` is the ONLY PROPERTY_MAPPING spec form (issue #694, Phase B).

Phase A introduced ``PropertySpec`` as a typed, frozen spec carrying the ``property_spec``
invariants. This module pins the Phase B hard break:

1. A raw dict spec in PROPERTY_MAPPING raises at class definition, naming the owning class,
   the property key and the ``PropertySpec`` remedy. Same for a bare non-dict spec value.
2. A PROPERTY_MAPPING whose values are ``PropertySpec`` instances defines fine, and the
   matching pipeline behaves exactly as the dict form did: strict accept/reject, context
   categorization of string-parsed values, ``required_when`` and ``match_guard``.
3. ``FeatureGroup.declared_option_values`` reads a ``PropertySpec``'s ``allowed_values``.
4. The unknown-key machinery is deleted: ``PROPERTY_SPEC_KEYS`` and ``REMOVED_PROPERTY_KEYS``
   no longer exist (``PropertySpec``'s constructor is the schema now).
5. ``property_spec(...)`` returns a ``PropertySpec`` (``strict=`` maps to
   ``strict_validation=``) and its authoring rejections still fire.
6. ``PropertySpec`` is exported from ``mloda.provider`` and is the same class.
7. ``FeatureChainParser._can_skip_required_check`` understands ``PropertySpec``: a non-None
   ``default`` or a ``required_when`` predicate makes the key skippable.
"""

from __future__ import annotations

import gc
import importlib
from typing import Any

import pytest

from mloda.core.abstract_plugins.components import default_options_key as default_options_key_module
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec, property_spec
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    Mirrors ``test_property_mapping_spec_schema.py``: the class-definition tests below
    define FeatureGroup subclasses, which linger in ``FeatureGroup.__subclasses__()`` until
    a GC cycle runs and would otherwise be seen by tests that enumerate feature groups.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


def _hardbreak694_needs_order_column(options: Options) -> bool:
    """Predicate: the order column is required when the aggregation is order-dependent."""
    return options.get("hardbreak694_agg") in {"first", "last"}


def _hardbreak694_positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


class TestRawDictSpecIsAHardBreak:
    """Item 1 and 2: PROPERTY_MAPPING accepts PropertySpec instances and NOTHING else."""

    def test_raw_dict_spec_rejected_at_class_definition(self) -> None:
        """A raw dict spec (only known keys, valid yesterday) raises, naming the remedy."""
        with pytest.raises(ValueError) as exc_info:

            class HardBreak694RawDictFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "hardbreak694_operation": {  # type: ignore[dict-item]  # wrong type is the point
                        "explanation": "x",
                        DefaultOptionKeys.context: True,
                    }
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "HardBreak694RawDictFeatureGroup" in message
        assert "hardbreak694_operation" in message
        assert "PropertySpec" in message

    def test_bare_non_dict_spec_rejected_at_class_definition_names_property_spec(self) -> None:
        """A bare tuple of values raises, and the message now names ``PropertySpec``."""
        with pytest.raises(ValueError) as exc_info:

            class HardBreak694BareTupleFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"hardbreak694_operation": ("add", "sub")}  # type: ignore[dict-item]  # wrong type is the point

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "HardBreak694BareTupleFeatureGroup" in message
        assert "hardbreak694_operation" in message
        assert "PropertySpec" in message


class TestPropertySpecMappingBehavesAsBefore:
    """Item 3: PropertySpec entries define fine and match exactly as the dict form did."""

    def test_strict_property_spec_accepts_declared_and_rejects_undeclared_value(self) -> None:
        """A strict PropertySpec matches a declared value and non-matches an undeclared one."""

        class HardBreak694StrictFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "hardbreak694_operation": PropertySpec(
                    "The arithmetic operation to apply",
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                    strict_validation=True,
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            HardBreak694StrictFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"hardbreak694_operation": "sub"})
            )
            is True
        )
        assert (
            HardBreak694StrictFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"hardbreak694_operation": "mul"})
            )
            is False
        )

    def test_context_spec_categorizes_string_parsed_value_into_context(self) -> None:
        """A ``context=True`` PropertySpec sends a name-parsed value to the context category.

        Mirrors the categorization assertions in
        ``tests/test_plugins/integration_plugins/chainer/context/test_parameter_resolution_unit.py``
        at a smaller scale.
        """
        spec = PropertySpec("The aggregation to apply", allowed_values={"first": "First value"}, context=True)

        category = FeatureChainParser._determine_parameter_category("hardbreak694_agg", spec, Options())
        assert category == DefaultOptionKeys.context

        class HardBreak694ContextCategorization(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__([\w]+)_hardbreak694$"
            PROPERTY_MAPPING = {"hardbreak694_agg": spec}

        effective = FeatureChainParser.build_effective_options(
            "source__first_hardbreak694",
            [HardBreak694ContextCategorization.PREFIX_PATTERN],
            HardBreak694ContextCategorization.PROPERTY_MAPPING,
            Options(),
        )
        assert effective.get("hardbreak694_agg") == "first"
        assert "hardbreak694_agg" in effective.context

    def test_required_when_property_spec_still_gates_the_match(self) -> None:
        """``required_when`` on a PropertySpec behaves exactly like the dict form did."""

        class HardBreak694RequiredWhenFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "hardbreak694_agg": PropertySpec(
                    "The aggregation to apply",
                    allowed_values={"sum": "Sum of values", "first": "First value (requires an order column)"},
                    strict_validation=True,
                ),
                "hardbreak694_order_by": PropertySpec(
                    "Column to order by within each partition",
                    required_when=_hardbreak694_needs_order_column,
                ),
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            HardBreak694RequiredWhenFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"hardbreak694_agg": "first"})
            )
            is False
        )
        assert (
            HardBreak694RequiredWhenFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"hardbreak694_agg": "first", "hardbreak694_order_by": "ts"})
            )
            is True
        )
        assert (
            HardBreak694RequiredWhenFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"hardbreak694_agg": "sum"})
            )
            is True
        )

    def test_match_guard_property_spec_still_gates_the_match(self) -> None:
        """``match_guard`` on a PropertySpec behaves exactly like the dict form did."""

        class HardBreak694MatchGuardFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "hardbreak694_window": PropertySpec(
                    "Size of the time window",
                    match_guard=_hardbreak694_positive_int,
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            HardBreak694MatchGuardFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"hardbreak694_window": 7})
            )
            is True
        )
        assert (
            HardBreak694MatchGuardFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"hardbreak694_window": -3})
            )
            is False
        )


class TestDeclaredOptionValuesReadsPropertySpec:
    """Item 4: ``declared_option_values`` returns the stringified PropertySpec value space."""

    def test_declared_option_values_with_tuple_allowed_values(self) -> None:
        """A tuple value space is stringified element-wise."""

        class HardBreak694TupleValuesFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {"hardbreak694_mode": PropertySpec("The mode to run in", allowed_values=("fast", 2))}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert HardBreak694TupleValuesFeatureGroup.declared_option_values("hardbreak694_mode") == frozenset(
            {"fast", "2"}
        )

    def test_declared_option_values_with_mapping_allowed_values(self) -> None:
        """A Mapping value space yields its stringified KEYS."""

        class HardBreak694MappingValuesFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "hardbreak694_operation": PropertySpec(
                    "The arithmetic operation to apply",
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert HardBreak694MappingValuesFeatureGroup.declared_option_values("hardbreak694_operation") == frozenset(
            {"add", "sub"}
        )


class TestUnknownKeyMachineryIsGone:
    """Item 5: the dict-spec schema machinery is deleted along with the dict form."""

    def test_property_spec_keys_is_gone(self) -> None:
        """``PROPERTY_SPEC_KEYS`` no longer exists: the PropertySpec constructor is the schema."""
        assert not hasattr(default_options_key_module, "PROPERTY_SPEC_KEYS")

    def test_removed_property_keys_is_gone(self) -> None:
        """``REMOVED_PROPERTY_KEYS`` no longer exists: there are no spec dicts to rename keys in."""
        assert not hasattr(default_options_key_module, "REMOVED_PROPERTY_KEYS")


class TestPropertySpecBuilderReturnsPropertySpec:
    """Item 6: ``property_spec`` builds a ``PropertySpec``, keeping its authoring surface."""

    def test_property_spec_returns_a_property_spec_instance(self) -> None:
        """``property_spec("x")`` returns a PropertySpec, not a dict."""
        spec = property_spec("x")

        assert isinstance(spec, PropertySpec)
        assert spec.explanation == "x"

    def test_property_spec_maps_strict_keyword_to_strict_validation_field(self) -> None:
        """The builder keyword stays ``strict=``; the field it sets is ``strict_validation``."""
        spec = property_spec("x", strict=True, allowed_values=("a",))

        assert isinstance(spec, PropertySpec)
        assert spec.strict_validation is True
        assert spec.allowed_values == ("a",)

    def test_property_spec_authoring_rejections_still_fire(self) -> None:
        """Spot check: a str ``allowed_values`` is still rejected as a substring trap."""
        substring_trap: Any = "add"  # the forgotten-comma bug: a bare str, not a container

        with pytest.raises(ValueError, match="(?i)substring"):
            property_spec("x", allowed_values=substring_trap)


class TestProviderExportsPropertySpec:
    """Item 7: ``PropertySpec`` is part of the provider surface."""

    def test_property_spec_is_exported_from_provider_and_is_the_same_class(self) -> None:
        """``from mloda.provider import PropertySpec`` resolves to the one core class."""
        provider_module = importlib.import_module("mloda.provider")

        exported = getattr(provider_module, "PropertySpec", None)
        assert exported is not None, "mloda.provider must export PropertySpec"
        assert exported is PropertySpec
        assert "PropertySpec" in getattr(provider_module, "__all__", ())


class TestCanSkipRequiredCheckOnPropertySpec:
    """Item 8: the base parser's skip rule reads PropertySpec fields."""

    def test_spec_with_default_is_skippable(self) -> None:
        """A non-None ``default`` makes the key optional for the base required check."""
        spec = PropertySpec("x", default="fallback")

        assert FeatureChainParser._can_skip_required_check(spec) is True

    def test_spec_with_required_when_is_skippable(self) -> None:
        """A ``required_when`` predicate defers the decision to the mixin layer."""
        spec = PropertySpec("x", required_when=_hardbreak694_needs_order_column)

        assert FeatureChainParser._can_skip_required_check(spec) is True

    def test_spec_without_default_or_required_when_is_not_skippable(self) -> None:
        """No default and no predicate means the key stays required."""
        spec = PropertySpec("x")

        assert FeatureChainParser._can_skip_required_check(spec) is False
