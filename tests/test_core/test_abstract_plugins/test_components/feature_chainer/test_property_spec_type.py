"""Pins the ``PropertySpec`` construction contract (issue #694).

``PropertySpec`` is the frozen dataclass replacing PROPERTY_MAPPING spec dicts. This module
tests the type in isolation: construction defaults, immutability, ``allowed_values``
normalization, the flag and callable shape rules, the strict-needs-a-value-space invariant,
the ``NO_DEFAULT`` optionality sentinel and the declared-default rule (issue #530 semantics).
The one consumer touched here is the type rule itself: the public parser entry point must
reject a spec that is not a ``PropertySpec``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import NO_DEFAULT, PropertySpec
from mloda.core.abstract_plugins.components.options import Options


def _spec(*args: Any, **kwargs: Any) -> PropertySpec:
    """Build a ``PropertySpec`` through an untyped seam.

    The type-invalid calls below (missing argument, unknown keyword, wrong-typed flag) must
    stay runtime tests; routing them through ``Any`` keeps the module mypy --strict clean.
    """
    return PropertySpec(*args, **kwargs)


def _is_int(value: Any) -> bool:
    return isinstance(value, int)


def _explode(value: Any) -> bool:
    raise RuntimeError(f"element_validator exploded on {value!r}")


class TestConstructionAndFields:
    """Minimal construction, required explanation, unknown keywords, immutability, field names."""

    def test_minimal_construction_uses_documented_defaults(self) -> None:
        """``PropertySpec("Some explanation")`` constructs with the documented defaults."""
        spec = PropertySpec("Some explanation")

        assert spec.explanation == "Some explanation"
        assert spec.allowed_values is None
        assert spec.default is NO_DEFAULT
        assert spec.context is True
        assert spec.strict_validation is False
        assert spec.element_validator is None
        assert spec.match_guard is None
        assert spec.required_when is None

    def test_explanation_is_required(self) -> None:
        """``PropertySpec()`` without an explanation raises TypeError."""
        with pytest.raises(TypeError):
            _spec()

    def test_unknown_keyword_raises_type_error(self) -> None:
        """A typo'd keyword is Python's own constructor error, never silently absorbed."""
        with pytest.raises(TypeError):
            _spec("x", strict_validaton=True)

    def test_instances_are_frozen(self) -> None:
        """Assigning any field on an instance raises ``dataclasses.FrozenInstanceError``."""
        spec = PropertySpec("x")

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(spec, "explanation", "changed")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(spec, "strict_validation", True)

    def test_flag_field_is_named_strict_validation(self) -> None:
        """The flag is ``strict_validation`` (bool), NOT ``strict``."""
        spec = PropertySpec("x")

        assert hasattr(spec, "strict_validation")
        assert not hasattr(spec, "strict")
        with pytest.raises(TypeError):
            _spec("x", strict=True)


class TestAllowedValuesNormalization:
    """Non-Mapping iterables are materialized to a tuple; Mappings are kept; bad shapes raise."""

    def test_list_is_materialized_to_a_tuple(self) -> None:
        """A list becomes a tuple, preserving order."""
        spec = PropertySpec("x", allowed_values=["a", "b"])

        assert spec.allowed_values == ("a", "b")

    def test_set_is_materialized_to_a_tuple(self) -> None:
        """A set becomes a tuple carrying exactly the set's members."""
        spec = PropertySpec("x", allowed_values={"a", "b"})

        assert isinstance(spec.allowed_values, tuple)
        assert set(spec.allowed_values) == {"a", "b"}

    def test_generator_is_materialized_and_reiterable(self) -> None:
        """A one-shot generator is consumed once and stored re-iterable."""
        generator = (value for value in ("a", "b"))
        spec = _spec("x", allowed_values=generator)

        values = spec.allowed_values
        assert values is not None
        assert list(values) == ["a", "b"]
        assert list(values) == ["a", "b"], "the stored value space must survive re-iteration"

    def test_mapping_is_kept_as_given(self) -> None:
        """A Mapping is the value-space-with-descriptions form and is kept as given."""
        spec = PropertySpec("x", allowed_values={"a": "A"})

        assert spec.allowed_values == {"a": "A"}

    def test_str_allowed_values_raises_value_error(self) -> None:
        """A bare str would make membership a SUBSTRING test: rejected up front."""
        with pytest.raises(ValueError, match="(?i)substring"):
            _spec("x", allowed_values="add")

    def test_bytes_allowed_values_raises_value_error(self) -> None:
        """Bytes are the same substring trap as str."""
        with pytest.raises(ValueError, match="(?i)substring"):
            _spec("x", allowed_values=b"add")

    def test_non_iterable_scalar_raises_value_error(self) -> None:
        """A scalar is a clear ValueError, not a bare TypeError escaping from tuple()."""
        with pytest.raises(ValueError):
            _spec("x", allowed_values=5)


class TestFlagAndCallableShapeRules:
    """The shape rules moved from class-definition time reject the same inputs here."""

    @pytest.mark.parametrize("bad_flag", ["false", 1])
    def test_strict_validation_must_be_a_real_bool(self, bad_flag: Any) -> None:
        """A truthy non-bool under ``strict_validation`` raises, naming the expected type."""
        with pytest.raises(ValueError, match="(?i)bool"):
            _spec("x", allowed_values=("a",), strict_validation=bad_flag)

    def test_non_callable_element_validator_raises(self) -> None:
        """``element_validator`` must be callable."""
        with pytest.raises(ValueError, match="(?i)callable"):
            _spec("x", allowed_values=("a",), strict_validation=True, element_validator="not callable")

    def test_non_callable_required_when_raises(self) -> None:
        """``required_when`` must be callable."""
        with pytest.raises(ValueError, match="(?i)callable"):
            _spec("x", required_when="not callable")

    def test_non_callable_match_guard_raises(self) -> None:
        """``match_guard`` must be callable."""
        with pytest.raises(ValueError, match="(?i)callable"):
            _spec("x", match_guard="not callable")

    def test_element_validator_without_strict_raises(self) -> None:
        """An ``element_validator`` is never enforced without ``strict_validation=True``."""
        with pytest.raises(ValueError, match="(?i)strict"):
            PropertySpec("x", element_validator=_is_int)


class TestStrictNeedsAValueSpace:
    """``strict_validation=True`` needs something to validate AGAINST."""

    @pytest.mark.parametrize("empty_values", [(), []])
    def test_strict_with_empty_allowed_values_raises(self, empty_values: Any) -> None:
        """An empty declared value space would reject every value."""
        with pytest.raises(ValueError):
            _spec("x", allowed_values=empty_values, strict_validation=True)

    def test_strict_with_neither_value_space_raises(self) -> None:
        """Strict with neither ``allowed_values`` nor ``element_validator`` accepts nothing."""
        with pytest.raises(ValueError):
            PropertySpec("x", strict_validation=True)

    def test_strict_with_element_validator_only_is_valid(self) -> None:
        """A callable ``element_validator`` IS a value space: no ``allowed_values`` needed."""
        spec = PropertySpec("x", strict_validation=True, element_validator=_is_int)

        assert spec.strict_validation is True
        assert spec.element_validator is _is_int
        assert spec.allowed_values is None

    def test_strict_with_element_validator_ignores_an_empty_allowed_values(self) -> None:
        """An ``element_validator`` decides instead of membership, so an empty value space is inert.

        The value-space rules do not fire at all when a validator is present: nothing is rejected
        by an empty ``allowed_values`` the match path never consults.
        """
        spec = PropertySpec("x", strict_validation=True, element_validator=_is_int, allowed_values=[])

        assert spec.allowed_values == ()
        assert spec.element_validator is _is_int


class TestNoDefaultSentinel:
    """``NO_DEFAULT`` (no declared default) vs a DECLARED ``default=None`` (optional, no value)."""

    def test_omitted_default_is_the_sentinel_and_makes_the_key_required(self) -> None:
        """A spec with no declared default is required: the parser may not skip it."""
        spec = PropertySpec("x")

        assert spec.default is NO_DEFAULT
        assert FeatureChainParser._can_skip_required_check(spec) is False

    def test_declared_none_default_makes_the_key_optional(self) -> None:
        """``default=None`` declares an optional key with no value to apply."""
        spec = PropertySpec("x", default=None)

        assert spec.default is None
        assert FeatureChainParser._can_skip_required_check(spec) is True

    def test_declared_none_default_matches_when_the_option_is_absent(self) -> None:
        """The optional key is not required at match time, which is what the plugins rely on."""
        property_mapping = {"weight_column": PropertySpec("Optional weight column", default=None)}

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={}), property_mapping
            )
            is True
        )

    def test_sentinel_repr_names_itself(self) -> None:
        """The sentinel is readable in a spec repr and in an assertion diff."""
        assert repr(NO_DEFAULT) == "NO_DEFAULT"


class TestParserEntryPointRequiresPropertySpec:
    """The public parser entry point enforces the type rule, not just class definition."""

    def test_match_configuration_rejects_a_dict_spec(self) -> None:
        """A raw dict spec handed straight to the matcher is a ValueError, never an AttributeError."""
        property_mapping: Any = {"operation_type": {"add": "Addition"}}

        with pytest.raises(ValueError, match="not a PropertySpec"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"operation_type": "add"}), property_mapping
            )

    def test_validate_property_mapping_defaults_rejects_a_dict_spec(self) -> None:
        """The class-definition check and the entry point share one rule and one message."""
        property_mapping: Any = {"operation_type": {"add": "Addition"}}

        with pytest.raises(ValueError, match="not a PropertySpec"):
            FeatureChainParser.validate_property_mapping_defaults("SomeOwner", property_mapping)


class TestDeclaredDefault:
    """The declared-default rule (issue #530 semantics) applies only under strict."""

    def test_strict_default_outside_allowed_values_raises(self) -> None:
        """A strict default must be within the declared value space."""
        with pytest.raises(ValueError, match="(?i)default"):
            PropertySpec("x", allowed_values=("a", "b"), strict_validation=True, default="c")

    def test_strict_default_inside_allowed_values_is_valid(self) -> None:
        """A strict default inside the declared value space constructs."""
        spec = PropertySpec("x", allowed_values=("a", "b"), strict_validation=True, default="a")

        assert spec.default == "a"

    def test_strict_element_validator_rejecting_default_raises(self) -> None:
        """A strict default rejected by the ``element_validator`` raises."""
        with pytest.raises(ValueError):
            PropertySpec("x", strict_validation=True, element_validator=_is_int, default="a")

    def test_strict_element_validator_raising_on_default_surfaces_as_value_error(self) -> None:
        """An ``element_validator`` that RAISES on the default still surfaces as ValueError."""
        with pytest.raises(ValueError):
            PropertySpec("x", strict_validation=True, element_validator=_explode, default="a")

    def test_non_strict_default_outside_allowed_values_is_valid(self) -> None:
        """The default check only applies under strict: a non-strict spec never runs it."""
        spec = PropertySpec("x", allowed_values=("a", "b"), default="z")

        assert spec.default == "z"

    def test_default_none_means_no_default(self) -> None:
        """``default=None`` declares no default, so no default check fires under strict."""
        spec = PropertySpec("x", allowed_values=("a", "b"), strict_validation=True, default=None)

        assert spec.default is None

    def test_mapping_allowed_values_default_key_is_valid(self) -> None:
        """Under strict, a Mapping value space accepts a default that is one of its KEYS."""
        spec = PropertySpec("x", allowed_values={"a": "A"}, strict_validation=True, default="a")

        assert spec.default == "a"

    def test_mapping_allowed_values_default_not_a_key_raises(self) -> None:
        """Under strict, a default outside the Mapping's KEYS raises."""
        with pytest.raises(ValueError, match="(?i)default"):
            PropertySpec("x", allowed_values={"a": "A"}, strict_validation=True, default="z")

    @pytest.mark.parametrize("unhashable_default", [{"a": 1}, ["a"], (["a"],)])
    def test_unhashable_default_against_a_mapping_raises_value_error(self, unhashable_default: Any) -> None:
        """An unhashable default can never be a Mapping key: a clean ValueError, never a TypeError.

        ``(["a"],)`` is the sharp case: a tuple IS an instance of ``Hashable``, but hashing it
        raises because it contains a list. Only the membership test itself can tell.
        """
        with pytest.raises(ValueError, match="(?i)default"):
            PropertySpec("x", allowed_values={"a": "A"}, strict_validation=True, default=unhashable_default)

    def test_unhashable_default_against_a_tuple_value_space_raises_value_error(self) -> None:
        """The same holds for a non-Mapping value space: membership decides, and it rejects."""
        with pytest.raises(ValueError, match="(?i)default"):
            PropertySpec("x", allowed_values=("a",), strict_validation=True, default={"a": 1})
