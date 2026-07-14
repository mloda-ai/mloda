"""Tests for the ``property_spec`` authoring helper (issue #543).

``property_spec`` builds a conventional PROPERTY_MAPPING entry from explicit
arguments, validating the invariants that previously had to be enforced by hand:

* ``strict=True`` requires a non-empty ``allowed_values`` (a strict enum with no
  value space rejects everything),
* a strict, non-``None`` ``default`` must be within the allowed set,
* a one-shot iterable for ``allowed_values`` is materialized so it survives reuse.

``allowed_values`` WITHOUT ``strict`` is legal: a non-strict value space is not
enforced, but it is consumed, to map a name-parsed value back onto its
PROPERTY_MAPPING key. See ``test_property_mapping_spec_shape.py``.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import property_spec


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    Copied from ``test_property_mapping_default_invariant.py``. The round-trip test
    defines a FeatureGroup subclass; this fixture forces a collection afterwards
    and asserts none of this module's classes linger in the registry.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


class TestPropertySpecImport:
    """The helper is exported from the public provider surface."""

    def test_property_spec_importable_from_provider(self) -> None:
        """``from mloda.provider import property_spec`` works."""
        from mloda.provider import property_spec as imported

        assert callable(imported)


class TestPropertySpecEmission:
    """A valid call emits a conventional spec dict."""

    def test_emits_conventional_spec_dict(self) -> None:
        """All conventional keys are present with the expected values."""
        spec = property_spec("desc", strict=True, allowed_values={"add": "Addition"}, default="add")

        assert spec["explanation"] == "desc"
        assert spec[DefaultOptionKeys.allowed_values] == {"add": "Addition"}
        assert spec[DefaultOptionKeys.strict_validation] is True
        assert spec[DefaultOptionKeys.context] is True
        assert spec[DefaultOptionKeys.default] == "add"


class TestPropertySpecInvariants:
    """Invalid combinations raise ``ValueError``."""

    def test_strict_without_allowed_values_raises(self) -> None:
        """``strict=True`` with no value space rejects everything, so it is illegal."""
        with pytest.raises(ValueError):
            property_spec("d", strict=True)

    def test_strict_default_outside_allowed_set_raises(self) -> None:
        """A strict, non-``None`` default must be within the allowed set."""
        with pytest.raises(ValueError):
            property_spec("d", strict=True, allowed_values={"add": "A"}, default="mul")

    def test_strict_default_none_is_exempt(self) -> None:
        """An omitted/``None`` default is always legal under strict validation."""
        spec = property_spec("d", strict=True, allowed_values={"add": "A"})

        assert spec[DefaultOptionKeys.allowed_values] == {"add": "A"}


class TestPropertySpecIterableAllowedValues:
    """``allowed_values`` may be a plain iterable, and one-shot iterables are materialized."""

    def test_plain_iterable_accepted(self) -> None:
        """A tuple of allowed values is accepted with strict validation."""
        spec = property_spec("d", strict=True, allowed_values=("add", "sub"), default="add")

        emitted = spec[DefaultOptionKeys.allowed_values]
        assert set(emitted) == {"add", "sub"}

    def test_one_shot_generator_is_materialized(self) -> None:
        """A generator must be materialized so the emitted allowed_values is re-iterable."""
        spec = property_spec("d", strict=True, allowed_values=(x for x in ("add", "sub")), default="add")

        emitted = spec[DefaultOptionKeys.allowed_values]
        first = list(emitted)
        second = list(emitted)
        assert first == second
        assert set(first) == {"add", "sub"}


class TestPropertySpecRoundTrip:
    """A spec built by the helper drives membership validation end to end."""

    def test_round_trip_through_property_mapping(self) -> None:
        """The built entry defines without error and accepts/rejects via the parser."""

        class RoundTripFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": property_spec(
                    "op",
                    strict=True,
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                    default="add",
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        property_mapping = RoundTripFeatureGroup.PROPERTY_MAPPING

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"operation_type": "add"}), property_mapping
            )
            is True
        )
        with pytest.raises(ValueError, match="mul"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"operation_type": "mul"}), property_mapping
            )


class TestPropertySpecDefaultOmission:
    """OMITTING the ``default`` argument must NOT emit a spurious ``default`` key.

    The parser's ``_can_skip_required_check`` treats the mere PRESENCE of the
    ``default`` key as "this option is optional", so a spurious key would silently
    make an otherwise-required property optional. The rule is keyed on the
    ARGUMENT, not on its value: omitting ``default`` means the key is REQUIRED,
    while an explicit ``default=None`` means optional with a ``None`` default
    (see ``TestPropertySpecNoneDefault``, issue #733).
    """

    def test_builder_without_default_omits_default_key(self) -> None:
        """No default argument -> the spec carries no ``default`` key at all."""
        spec = property_spec("op", strict=True, allowed_values={"add": "Addition", "sub": "Subtraction"})

        assert DefaultOptionKeys.default not in spec

    def test_builder_without_default_makes_strict_property_required(self) -> None:
        """A defaultless strict property is REQUIRED: absent option -> no match.

        With the spurious ``default`` key the parser wrongly skips the required
        check, so an absent option matches. After the fix the option is required,
        so an absent option yields False while a present, valid option yields True.
        """

        class RequiredOpFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": property_spec(
                    "op",
                    strict=True,
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        property_mapping = RequiredOpFeatureGroup.PROPERTY_MAPPING

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={}), property_mapping
            )
            is False
        )
        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"operation_type": "add"}), property_mapping
            )
            is True
        )

    def test_builder_with_explicit_default_keeps_default_key(self) -> None:
        """An explicit, valid default is preserved (we did not over-correct)."""
        spec = property_spec("op", strict=True, allowed_values={"add": "Addition"}, default="add")

        assert spec[DefaultOptionKeys.default] == "add"


class TestPropertySpecNoneDefault:
    """``default=None`` must be expressible: optional key whose default is ``None`` (issue #733).

    ``None`` is a legitimate default (the shipped hand-written specs use it for
    ``weight_column``, ``constant_value``, ``pipeline_steps``, ...), but the builder
    used ``None`` as its own "argument omitted" marker and dropped the key. That made
    an optional key REQUIRED on migration. A ``NO_DEFAULT`` sentinel separates the two:
    the key is emitted whenever the caller passes ANY default, ``None`` included.
    """

    def test_default_none_emits_default_key(self) -> None:
        """``default=None`` emits the ``default`` key carrying ``None``."""
        spec = property_spec("d", default=None)

        assert DefaultOptionKeys.default in spec
        assert spec[DefaultOptionKeys.default] is None

    def test_default_none_is_optional(self) -> None:
        """A ``default=None`` spec is optional: the parser skips the required check."""
        spec = property_spec("d", default=None)

        assert FeatureChainParser._can_skip_required_check(spec) is True

    def test_omitted_default_stays_required(self) -> None:
        """No ``default`` argument -> no key, and the property stays REQUIRED (issue #562)."""
        spec = property_spec("d")

        assert DefaultOptionKeys.default not in spec
        assert FeatureChainParser._can_skip_required_check(spec) is False

    def test_no_default_sentinel_importable_from_provider(self) -> None:
        """``from mloda.provider import NO_DEFAULT`` works."""
        from mloda.provider import NO_DEFAULT as imported

        assert imported is not None

    def test_explicit_sentinel_behaves_like_omission(self) -> None:
        """Passing ``NO_DEFAULT`` explicitly is exactly the same as omitting the argument."""
        from mloda.provider import NO_DEFAULT

        spec = property_spec("d", default=NO_DEFAULT)

        assert spec == property_spec("d")
        assert DefaultOptionKeys.default not in spec
        assert FeatureChainParser._can_skip_required_check(spec) is False

    def test_none_default_spec_equals_hand_written_dict(self) -> None:
        """A ``default=None`` spec is exactly the dict a plugin author hand-writes today."""
        built = property_spec("Column name for edge weights (optional)", default=None)

        hand_written: dict[str, Any] = {
            "explanation": "Column name for edge weights (optional)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        }
        assert built == hand_written

    def test_strict_spec_with_none_default_builds_and_is_optional(self) -> None:
        """A STRICT spec with ``default=None`` builds and is optional (sklearn PIPELINE_NAME shape).

        Core's ``check_declared_default`` exempts a ``None`` default from the membership
        check, so the strict value space and the ``None`` default coexist.
        """
        spec = property_spec("d", strict=True, allowed_values={"scaling": "Feature scaling"}, default=None)

        assert spec[DefaultOptionKeys.default] is None
        assert spec[DefaultOptionKeys.strict_validation] is True
        assert FeatureChainParser._can_skip_required_check(spec) is True

    def test_absent_optional_option_still_matches_through_core(self) -> None:
        """End to end: a builder-authored ``default=None`` key is not demanded by the matcher."""

        class OptionalWeightFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_centrality$"
            PROPERTY_MAPPING = {
                "centrality_type": property_spec(
                    "Centrality metric",
                    strict=True,
                    allowed_values={"degree": "Degree", "pagerank": "PageRank"},
                ),
                "weight_column": property_spec("Column name for edge weights (optional)", default=None),
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        property_mapping = OptionalWeightFeatureGroup.PROPERTY_MAPPING

        # The optional key is ABSENT: the required key alone satisfies the match.
        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"centrality_type": "degree"}), property_mapping
            )
            is True
        )
        # Present is fine too, and the required key is still required.
        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature",
                Options(context={"centrality_type": "degree", "weight_column": "weight"}),
                property_mapping,
            )
            is True
        )
        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"weight_column": "weight"}), property_mapping
            )
            is False
        )


def _positive_int(value: Any) -> bool:
    """Shared validator for the ``element_validator`` passthrough tests (issue #536)."""
    return isinstance(value, int) and value > 0


def _is_mul(value: Any) -> bool:
    """Validator that only accepts the literal ``"mul"`` (issue #536)."""
    return bool(value == "mul")


def _always_required(options: Any) -> bool:
    """``required_when`` predicate that always demands the option (issue #536)."""
    return True


def _is_list_of_strings(value: Any) -> bool:
    """``match_guard`` accepting only a list of strings (issue #536)."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _boom(value: Any) -> bool:
    """Validator that raises instead of returning a verdict (issue #536)."""
    raise RuntimeError("boom")


class TestPropertySpecElementValidator:
    """``element_validator`` passthrough (issue #536).

    Core's ``_validate_property_value`` applies a spec's ``element_validator``
    INSTEAD of the membership check under strict validation, so a strict spec with
    an ``element_validator`` and no ``allowed_values`` is valid and meaningful
    (see the ``window_size`` example in ``docs/docs/in_depth/property-mapping.md``).
    Like ``allowed_values``, an ``element_validator`` is only enforced under
    strict validation, so passing it without ``strict=True`` is a silent no-op
    and is rejected.
    """

    def test_strict_with_element_validator_needs_no_allowed_values(self) -> None:
        """``strict=True`` plus an ``element_validator`` needs no ``allowed_values``."""
        spec = property_spec("d", strict=True, element_validator=_positive_int)

        assert spec["explanation"] == "d"
        assert spec[DefaultOptionKeys.element_validator] is _positive_int
        assert spec[DefaultOptionKeys.strict_validation] is True
        assert spec[DefaultOptionKeys.context] is True

    def test_element_validator_without_strict_raises(self) -> None:
        """A ``element_validator`` is never enforced without ``strict`` (silent no-op)."""
        with pytest.raises(ValueError):
            property_spec("d", element_validator=_positive_int)

    def test_non_callable_element_validator_raises(self) -> None:
        """A non-callable ``element_validator`` is rejected up front."""
        not_callable: Any = "not callable"
        with pytest.raises(ValueError):
            property_spec("d", strict=True, element_validator=not_callable)


class TestPropertySpecElementValidatorDefault:
    """Strict defaults are checked via the ``element_validator`` when present (issue #536).

    Mirrors ``FeatureChainParser.validate_property_mapping_defaults``: when a spec
    carries an ``element_validator``, the strict default check uses it, taking
    precedence over membership in ``allowed_values``.
    """

    def test_strict_default_accepted_by_element_validator(self) -> None:
        """A default the ``element_validator`` accepts is legal without ``allowed_values``."""
        spec = property_spec("d", strict=True, element_validator=_positive_int, default=5)

        assert spec[DefaultOptionKeys.default] == 5

    def test_strict_default_rejected_by_element_validator_raises(self) -> None:
        """A default the ``element_validator`` rejects is illegal."""
        with pytest.raises(ValueError):
            property_spec("d", strict=True, element_validator=_positive_int, default=-1)

    def test_element_validator_takes_precedence_over_allowed_values(self) -> None:
        """With both present, the default is checked via the ``element_validator``, not membership."""
        spec = property_spec("d", strict=True, allowed_values={"add": "A"}, element_validator=_is_mul, default="mul")

        assert spec[DefaultOptionKeys.default] == "mul"
        assert spec[DefaultOptionKeys.allowed_values] == {"add": "A"}
        assert spec[DefaultOptionKeys.element_validator] is _is_mul


class TestPropertySpecRequiredWhen:
    """``required_when`` passthrough (issue #536).

    ``required_when`` is a conditional-requirement predicate, independent of
    strict validation, so it is legal without ``strict=True``.
    """

    def test_required_when_emitted_without_strict(self) -> None:
        """The predicate is emitted under ``DefaultOptionKeys.required_when`` without ``strict``."""
        spec = property_spec("d", required_when=_always_required)

        assert spec[DefaultOptionKeys.required_when] is _always_required
        assert spec[DefaultOptionKeys.strict_validation] is False

    def test_non_callable_required_when_raises(self) -> None:
        """A non-callable ``required_when`` is rejected up front."""
        not_callable: Any = "not callable"
        with pytest.raises(ValueError):
            property_spec("d", required_when=not_callable)


class TestPropertySpecMatchGuard:
    """``match_guard`` passthrough (issue #536).

    ``match_guard`` checks the raw option value's shape before any list
    unpacking and, unlike ``element_validator``, does not require strict
    validation.
    """

    def test_match_guard_emitted_without_strict(self) -> None:
        """The validator is emitted under ``DefaultOptionKeys.match_guard`` without ``strict``."""
        spec = property_spec("d", match_guard=_is_list_of_strings)

        assert spec[DefaultOptionKeys.match_guard] is _is_list_of_strings
        assert spec[DefaultOptionKeys.strict_validation] is False

    def test_non_callable_match_guard_raises(self) -> None:
        """A non-callable ``match_guard`` is rejected up front."""
        not_callable: Any = "not callable"
        with pytest.raises(ValueError):
            property_spec("d", match_guard=not_callable)


class TestPropertySpecPassthroughOmission:
    """Omitted passthroughs never appear in the emitted dict (issue #536).

    Emitting ``None``-valued passthrough keys would change core behavior (e.g.
    ``_can_skip_required_check`` keys off the mere PRESENCE of
    ``required_when``), so an omitted passthrough must be absent, not ``None``.
    """

    def test_omitted_passthroughs_are_absent(self) -> None:
        """Only explicitly passed passthrough keys are emitted; the rest are absent."""
        plain = property_spec("d")
        assert DefaultOptionKeys.element_validator not in plain
        assert DefaultOptionKeys.required_when not in plain
        assert DefaultOptionKeys.match_guard not in plain

        with_required_when = property_spec("d", required_when=_always_required)
        assert DefaultOptionKeys.element_validator not in with_required_when
        assert DefaultOptionKeys.match_guard not in with_required_when

        with_match_guard = property_spec("d", match_guard=_is_list_of_strings)
        assert DefaultOptionKeys.element_validator not in with_match_guard
        assert DefaultOptionKeys.required_when not in with_match_guard


class TestPropertySpecElementValidatorRoundTrip:
    """A strict ``element_validator`` spec matches core semantics end to end (issue #536)."""

    def test_class_definition_accepts_element_validator_default(self) -> None:
        """The built entry defines without error and equals the hand-written dict.

        Core's class-definition check (``validate_property_mapping_defaults``)
        accepts a strict default via the ``element_validator``; the helper must
        emit exactly the dict an author would hand-write for that spec.
        """

        class WindowFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_window$"
            PROPERTY_MAPPING = {
                "window_size": property_spec(
                    "Size of time window",
                    strict=True,
                    element_validator=_positive_int,
                    default=5,
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        built = WindowFeatureGroup.PROPERTY_MAPPING["window_size"]

        hand_written: dict[str, Any] = {
            "explanation": "Size of time window",
            DefaultOptionKeys.element_validator: _positive_int,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: 5,
        }
        assert built == hand_written

        FeatureChainParser.validate_property_mapping_defaults("Built", {"window_size": built})
        FeatureChainParser.validate_property_mapping_defaults("HandWritten", {"window_size": hand_written})


class TestPropertySpecRaisingElementValidator:
    """A ``element_validator`` that raises on the default is wrapped (issue #536).

    Mirrors core's ``FeatureChainParser.validate_property_mapping_defaults``, which
    distinguishes "the element_validator raised when called with the default"
    from "the element_validator ran and rejected the default": both surface as
    ``ValueError``, with the original exception chained as ``__cause__`` in the
    raising case. The builder's strict-default check must behave the same way
    instead of letting the author's exception propagate raw.
    """

    def test_raising_element_validator_wraps_as_value_error_with_cause(self) -> None:
        """``_boom`` raising ``RuntimeError`` surfaces as ``ValueError`` with the original chained."""
        with pytest.raises(ValueError) as exc_info:
            property_spec("d", strict=True, element_validator=_boom, default=5)

        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestPropertySpecEmptyAllowedValues:
    """An explicitly empty ``allowed_values`` is always an authoring mistake (issue #536).

    Even when an ``element_validator`` is present (so the spec would still be
    enforceable), an explicitly empty allowed set is dead configuration: core's
    ``_extract_property_values`` would surface it as an empty accepted set. The
    builder must reject it up front instead of silently emitting a dead, empty
    ``allowed_values`` key.
    """

    def test_empty_allowed_values_with_element_validator_raises(self) -> None:
        """``strict=True`` with ``allowed_values=[]`` raises even though a validator is present."""
        with pytest.raises(ValueError):
            property_spec("d", strict=True, element_validator=_positive_int, allowed_values=[])


class TestPropertySpecPassthroughRegressionGuards:
    """Regression guards from the second review round of the passthroughs (issue #536)."""

    def test_falsy_non_none_default_is_still_validated(self) -> None:
        """``default=0`` is falsy but not ``None``, so it must still be checked (and rejected)."""
        with pytest.raises(ValueError):
            property_spec("d", strict=True, element_validator=_positive_int, default=0)

    def test_element_validator_precedence_also_rejects_allowed_member(self) -> None:
        """With both present, a default IN ``allowed_values`` is still rejected by the validator."""
        with pytest.raises(ValueError):
            property_spec(
                "d",
                strict=True,
                allowed_values={"add": "A"},
                element_validator=_positive_int,
                default="add",
            )

    def test_required_when_spec_equals_hand_written_dict(self) -> None:
        """A ``required_when`` spec is exactly the dict an author would hand-write."""
        built = property_spec("d", required_when=_always_required)

        hand_written: dict[str, Any] = {
            "explanation": "d",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.required_when: _always_required,
        }
        assert built == hand_written

    def test_match_guard_spec_equals_hand_written_dict(self) -> None:
        """A ``match_guard`` spec is exactly the dict an author would hand-write."""
        built = property_spec("d", match_guard=_is_list_of_strings)

        hand_written: dict[str, Any] = {
            "explanation": "d",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.match_guard: _is_list_of_strings,
        }
        assert built == hand_written
