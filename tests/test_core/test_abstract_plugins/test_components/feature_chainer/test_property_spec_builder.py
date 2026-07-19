"""Tests for the ``property_spec`` authoring helper (issue #543).

``property_spec`` is a thin builder over ``PropertySpec`` (issue #694): it returns a
``PropertySpec`` instance and maps its ``strict=`` keyword onto the ``strict_validation``
field. All spec invariants live in ``PropertySpec.__post_init__`` (pinned in
``test_property_spec_type.py``); this module pins the builder surface:

* the return value IS a ``PropertySpec``,
* ``strict=`` maps to ``strict_validation=``,
* every rejection still fires through the builder,
* built specs equal their directly-constructed counterparts.

``allowed_values`` WITHOUT ``strict`` is legal: a non-strict value space is not
enforced, but it is consumed, to map a name-parsed value back onto its
PROPERTY_MAPPING key. See ``test_property_mapping_spec_shape.py``.
"""

from __future__ import annotations

import copy
import gc
import pickle
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import _NoDefault, is_no_default
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import NO_DEFAULT, PropertySpec, property_spec


def _build(*args: Any, **kwargs: Any) -> PropertySpec:
    """Call ``property_spec`` through an untyped seam.

    The builder's declared ``allowed_values`` type lists the container shapes so a bare str is an
    author-time error. Its runtime is deliberately more lenient (any iterable is materialized), and
    the leniency tests below exercise exactly the shapes the type does not name.
    """
    return property_spec(*args, **kwargs)


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
    """A valid call returns a ``PropertySpec`` with the expected field values."""

    def test_returns_property_spec_with_expected_fields(self) -> None:
        """The builder returns a ``PropertySpec``; ``strict=`` sets ``strict_validation``."""
        spec = property_spec("desc", strict=True, allowed_values={"add": "Addition"}, default="add")

        assert isinstance(spec, PropertySpec)
        assert spec.explanation == "desc"
        assert spec.allowed_values == {"add": "Addition"}
        assert spec.strict_validation is True
        assert spec.context is True
        assert spec.default == "add"


class TestPropertySpecInvariants:
    """Invalid combinations still raise ``ValueError`` through the builder."""

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

        assert spec.allowed_values == {"add": "A"}

    def test_builder_errors_carry_the_property_spec_prefix(self) -> None:
        """Rejections fire in ``PropertySpec.__post_init__``, so the prefix names the type."""
        with pytest.raises(ValueError, match=r"PropertySpec\('d'\)"):
            property_spec("d", strict=True)


class TestPropertySpecIterableAllowedValues:
    """``allowed_values`` may be a plain iterable, and one-shot iterables are materialized."""

    def test_plain_iterable_accepted(self) -> None:
        """A tuple of allowed values is accepted with strict validation."""
        spec = property_spec("d", strict=True, allowed_values=("add", "sub"), default="add")

        assert spec.allowed_values is not None
        assert set(spec.allowed_values) == {"add", "sub"}

    def test_one_shot_generator_is_materialized(self) -> None:
        """A generator must be materialized so the emitted allowed_values is re-iterable."""
        spec = _build("d", strict=True, allowed_values=(x for x in ("add", "sub")), default="add")

        assert spec.allowed_values is not None
        first = list(spec.allowed_values)
        second = list(spec.allowed_values)
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
    """A builder call with no default must leave the ``default`` field at ``NO_DEFAULT``.

    The parser's ``_can_skip_required_check`` treats a DECLARED default as "this option is
    optional". ``NO_DEFAULT`` means no default was declared, so an otherwise-required strict
    property stays required; a declared ``default=None`` is the optional-with-no-value form
    (see ``TestPropertySpecNoneDefault``, issue #733).
    """

    def test_builder_without_default_leaves_default_at_the_sentinel(self) -> None:
        """No default argument -> the spec's ``default`` field is ``NO_DEFAULT``."""
        spec = property_spec("op", strict=True, allowed_values={"add": "Addition", "sub": "Subtraction"})

        assert spec.default is NO_DEFAULT

    def test_builder_with_declared_none_default_makes_the_key_optional(self) -> None:
        """``default=None`` through the builder marks the key optional: an absent option matches."""
        property_mapping = {"weight_column": property_spec("Optional weight column", default=None)}

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={}), property_mapping
            )
            is True
        )

    def test_builder_without_default_makes_strict_property_required(self) -> None:
        """A defaultless strict property is REQUIRED: absent option -> no match.

        An absent option yields False while a present, valid option yields True.
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

    def test_builder_with_explicit_default_keeps_default(self) -> None:
        """An explicit, valid default is preserved (we did not over-correct)."""
        spec = property_spec("op", strict=True, allowed_values={"add": "Addition"}, default="add")

        assert spec.default == "add"


class TestPropertySpecNoneDefault:
    """``default=None`` must be expressible: optional key whose default is ``None`` (issue #733).

    ``None`` is a legitimate default (the shipped specs use it for ``weight_column``,
    ``constant_value``, ``pipeline_steps``, ...), but the builder once used ``None`` as its own
    "argument omitted" marker and dropped the key. That made an optional key REQUIRED on
    migration. The ``NO_DEFAULT`` sentinel separates the two: a declared default is whatever the
    caller passed, ``None`` included, while ``NO_DEFAULT`` means none was declared.
    """

    def test_default_none_is_a_declared_default(self) -> None:
        """``default=None`` lands on the field as a DECLARED ``None``, not as ``NO_DEFAULT``."""
        spec = property_spec("d", default=None)

        assert spec.default is None
        assert not is_no_default(spec.default)

    def test_default_none_is_optional(self) -> None:
        """A ``default=None`` spec is optional: the parser skips the required check."""
        spec = property_spec("d", default=None)

        assert FeatureChainParser._can_skip_required_check(spec) is True

    def test_omitted_default_stays_required(self) -> None:
        """No ``default`` argument -> ``NO_DEFAULT``, and the property stays REQUIRED (issue #562)."""
        spec = property_spec("d")

        assert is_no_default(spec.default)
        assert FeatureChainParser._can_skip_required_check(spec) is False

    def test_no_default_sentinel_importable_from_provider(self) -> None:
        """The provider re-exports THE sentinel, not merely some object of the same name."""
        from mloda.provider import NO_DEFAULT as imported

        assert imported is NO_DEFAULT

    def test_explicit_sentinel_behaves_like_omission(self) -> None:
        """Passing ``NO_DEFAULT`` explicitly is exactly the same as omitting the argument."""
        spec = property_spec("d", default=NO_DEFAULT)

        assert spec == property_spec("d")
        assert is_no_default(spec.default)
        assert FeatureChainParser._can_skip_required_check(spec) is False

    def test_none_default_spec_equals_direct_construction(self) -> None:
        """A ``default=None`` spec is exactly the ``PropertySpec`` a plugin author constructs."""
        built = property_spec("Column name for edge weights (optional)", default=None)

        hand_constructed = PropertySpec(
            "Column name for edge weights (optional)",
            context=True,
            strict_validation=False,
            default=None,
        )
        assert built == hand_constructed

    def test_strict_spec_with_none_default_builds_and_is_optional(self) -> None:
        """A STRICT spec with ``default=None`` builds and is optional (sklearn PIPELINE_NAME shape).

        ``PropertySpec._check_declared_default`` exempts a ``None`` default from the membership
        check, so the strict value space and the ``None`` default coexist.
        """
        spec = property_spec("d", strict=True, allowed_values={"scaling": "Feature scaling"}, default=None)

        assert spec.default is None
        assert spec.strict_validation is True
        assert FeatureChainParser._can_skip_required_check(spec) is True

    def test_absent_optional_option_still_matches_through_core(self) -> None:
        """End to end: a builder-authored ``default=None`` key is not demanded by the matcher."""

        class OptionalWeightFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            # A pattern unique to this module: a test FeatureGroup joins the global
            # get_all_subclasses discovery pool, so reusing the shipped
            # NodeCentralityFeatureGroup shape would make it a competing match.
            PREFIX_PATTERN = r".*__([\w]+)_optweight$"
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


class TestNoDefaultSentinelIdentity:
    """A round-tripped ``NO_DEFAULT`` must still mean "no default" (issue #733).

    A deepcopy, a pickle round trip, or a second import of the module (editable install plus
    site-packages, ``importlib.reload``) all yield a ``_NoDefault`` instance that is a DIFFERENT
    object than the module-level ``NO_DEFAULT``. If the sentinel is recognized by identity, such a
    copy reads as a DECLARED ``default`` VALUE, which silently flips a REQUIRED key to optional:
    exactly the bug class #733 exists to kill. ``is_no_default`` is a type test for that reason.
    """

    def test_deepcopy_returns_the_same_sentinel(self) -> None:
        """``copy.deepcopy`` must not clone the sentinel."""
        assert copy.deepcopy(NO_DEFAULT) is NO_DEFAULT

    def test_pickle_round_trip_returns_the_same_sentinel(self) -> None:
        """A pickle round trip must resolve back to the one sentinel object."""
        assert pickle.loads(pickle.dumps(NO_DEFAULT)) is NO_DEFAULT

    def test_deepcopied_sentinel_behaves_like_omission(self) -> None:
        """A deepcopied sentinel declares no default and leaves the property REQUIRED."""
        spec = property_spec("d", default=copy.deepcopy(NO_DEFAULT))

        assert is_no_default(spec.default)
        assert FeatureChainParser._can_skip_required_check(spec) is False
        assert spec == property_spec("d")

    def test_pickled_sentinel_behaves_like_omission(self) -> None:
        """A pickle round-tripped sentinel declares no default and stays REQUIRED."""
        spec = property_spec("d", default=pickle.loads(pickle.dumps(NO_DEFAULT)))

        assert is_no_default(spec.default)
        assert FeatureChainParser._can_skip_required_check(spec) is False
        assert spec == property_spec("d")

    def test_second_sentinel_instance_behaves_like_omission(self) -> None:
        """A separately constructed ``_NoDefault`` (a second imported copy) means "no default" too.

        It is NOT the module-level object, so only the type test can recognize it.
        """
        spec = property_spec("d", default=_NoDefault())

        assert spec.default is not NO_DEFAULT
        assert is_no_default(spec.default)
        assert FeatureChainParser._can_skip_required_check(spec) is False

    def test_sentinel_repr_is_readable(self) -> None:
        """The sentinel keeps its readable repr for error messages and docs."""
        assert repr(NO_DEFAULT) == "NO_DEFAULT"


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

        assert spec.explanation == "d"
        assert spec.element_validator is _positive_int
        assert spec.strict_validation is True
        assert spec.context is True

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

    When a spec carries an ``element_validator``, the strict default check uses it,
    taking precedence over membership in ``allowed_values``.
    """

    def test_strict_default_accepted_by_element_validator(self) -> None:
        """A default the ``element_validator`` accepts is legal without ``allowed_values``."""
        spec = property_spec("d", strict=True, element_validator=_positive_int, default=5)

        assert spec.default == 5

    def test_strict_default_rejected_by_element_validator_raises(self) -> None:
        """A default the ``element_validator`` rejects is illegal."""
        with pytest.raises(ValueError):
            property_spec("d", strict=True, element_validator=_positive_int, default=-1)

    def test_element_validator_takes_precedence_over_allowed_values(self) -> None:
        """With both present, the default is checked via the ``element_validator``, not membership."""
        spec = property_spec("d", strict=True, allowed_values={"add": "A"}, element_validator=_is_mul, default="mul")

        assert spec.default == "mul"
        assert spec.allowed_values == {"add": "A"}
        assert spec.element_validator is _is_mul


class TestPropertySpecRequiredWhen:
    """``required_when`` passthrough (issue #536).

    ``required_when`` is a conditional-requirement predicate, independent of
    strict validation, so it is legal without ``strict=True``.
    """

    def test_required_when_emitted_without_strict(self) -> None:
        """The predicate lands on the ``required_when`` field without ``strict``."""
        spec = property_spec("d", required_when=_always_required)

        assert spec.required_when is _always_required
        assert spec.strict_validation is False

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
        """The validator lands on the ``match_guard`` field without ``strict``."""
        spec = property_spec("d", match_guard=_is_list_of_strings)

        assert spec.match_guard is _is_list_of_strings
        assert spec.strict_validation is False

    def test_non_callable_match_guard_raises(self) -> None:
        """A non-callable ``match_guard`` is rejected up front."""
        not_callable: Any = "not callable"
        with pytest.raises(ValueError):
            property_spec("d", match_guard=not_callable)


class TestPropertySpecPassthroughOmission:
    """Omitted passthroughs stay at their ``None`` defaults (issue #536).

    ``None`` is the documented "not declared" sentinel: core keys off
    ``required_when is not None`` (and friends), so an omitted passthrough must
    be ``None``, never a truthy placeholder.
    """

    def test_omitted_passthroughs_are_none(self) -> None:
        """Only explicitly passed passthroughs are set; the rest stay ``None``."""
        plain = property_spec("d")
        assert plain.element_validator is None
        assert plain.required_when is None
        assert plain.match_guard is None

        with_required_when = property_spec("d", required_when=_always_required)
        assert with_required_when.element_validator is None
        assert with_required_when.match_guard is None

        with_match_guard = property_spec("d", match_guard=_is_list_of_strings)
        assert with_match_guard.element_validator is None
        assert with_match_guard.required_when is None


class TestPropertySpecElementValidatorRoundTrip:
    """A strict ``element_validator`` spec matches core semantics end to end (issue #536)."""

    def test_class_definition_accepts_element_validator_default(self) -> None:
        """The built entry defines without error and equals the direct construction.

        The builder must produce exactly the ``PropertySpec`` an author would
        construct by hand for that spec, and both pass the class-definition check
        (``validate_property_mapping_defaults``).
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

        hand_constructed = PropertySpec(
            "Size of time window",
            element_validator=_positive_int,
            context=True,
            strict_validation=True,
            default=5,
        )
        assert built == hand_constructed

        FeatureChainParser.validate_property_mapping_defaults("Built", {"window_size": built})
        FeatureChainParser.validate_property_mapping_defaults("HandConstructed", {"window_size": hand_constructed})


class TestPropertySpecRaisingElementValidator:
    """A ``element_validator`` that raises on the default is wrapped (issue #536).

    ``PropertySpec`` distinguishes "the element_validator raised when called with
    the default" from "the element_validator ran and rejected the default": both
    surface as ``ValueError``, with the original exception chained as
    ``__cause__`` in the raising case. The builder must surface the same
    behavior instead of letting the author's exception propagate raw.
    """

    def test_raising_element_validator_wraps_as_value_error_with_cause(self) -> None:
        """``_boom`` raising ``RuntimeError`` surfaces as ``ValueError`` with the original chained."""
        with pytest.raises(ValueError) as exc_info:
            property_spec("d", strict=True, element_validator=_boom, default=5)

        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestPropertySpecEmptyAllowedValues:
    """An empty ``allowed_values`` only rejects when membership is what decides.

    The value-space rules exist because an empty accepted set would reject every value. With an
    ``element_validator`` the validator, not membership, decides: ``allowed_values`` is never
    consulted, so an empty one rejects nothing and the rules do not fire. Without a validator an
    empty (or absent) value space is still the reject-everything spec, and is refused.
    """

    def test_empty_allowed_values_with_element_validator_is_accepted(self) -> None:
        """``strict=True`` with ``allowed_values=[]`` is legal while a validator is present."""
        spec = property_spec("d", strict=True, element_validator=_positive_int, allowed_values=[])

        assert spec.element_validator is _positive_int
        assert spec.allowed_values == ()

    def test_empty_allowed_values_without_element_validator_raises(self) -> None:
        """Without a validator, an empty accepted set would reject every value."""
        with pytest.raises(ValueError, match="(?i)empty allowed_values"):
            property_spec("d", strict=True, allowed_values=[])


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

    def test_required_when_spec_equals_direct_construction(self) -> None:
        """A ``required_when`` spec is exactly the ``PropertySpec`` an author would construct."""
        built = property_spec("d", required_when=_always_required)

        hand_constructed = PropertySpec(
            "d",
            context=True,
            strict_validation=False,
            required_when=_always_required,
        )
        assert built == hand_constructed

    def test_match_guard_spec_equals_direct_construction(self) -> None:
        """A ``match_guard`` spec is exactly the ``PropertySpec`` an author would construct."""
        built = property_spec("d", match_guard=_is_list_of_strings)

        hand_constructed = PropertySpec(
            "d",
            context=True,
            strict_validation=False,
            match_guard=_is_list_of_strings,
        )
        assert built == hand_constructed


class TestPropertySpecAllowExplicitNone:
    """``allow_explicit_none`` passthrough (issue #768).

    The opt-in flag defaults to ``False`` and rides through the builder onto the field; a built
    spec equals the ``PropertySpec`` an author constructs by hand.
    """

    def test_allow_explicit_none_emitted_through_builder(self) -> None:
        """``allow_explicit_none=True`` lands on the field; omitting it leaves the default ``False``."""
        assert property_spec("d", allow_explicit_none=True).allow_explicit_none is True
        assert property_spec("d").allow_explicit_none is False

    def test_allow_explicit_none_spec_equals_direct_construction(self) -> None:
        """An ``allow_explicit_none`` spec is exactly the ``PropertySpec`` an author would construct."""
        built = property_spec("d", allow_explicit_none=True)

        hand_constructed = PropertySpec(
            "d",
            context=True,
            strict_validation=False,
            allow_explicit_none=True,
        )
        assert built == hand_constructed


class TestPropertySpecDeferredBinding:
    """``deferred_binding`` passthrough (issue #769).

    The per-key opt-out defaults to ``False`` and rides through the builder onto the field; a built
    spec equals the ``PropertySpec`` an author constructs by hand, and a non-bool value is rejected at
    construction (mirroring ``allow_explicit_none``). ``deferred_binding=True`` exempts the key from the
    name-path required-presence check only; it does not change config-path requiredness.
    """

    def test_deferred_binding_emitted_through_builder(self) -> None:
        """``deferred_binding=True`` lands on the field; omitting it leaves the default ``False``."""
        assert property_spec("d", deferred_binding=True).deferred_binding is True
        assert property_spec("d").deferred_binding is False

    def test_deferred_binding_spec_equals_direct_construction(self) -> None:
        """A ``deferred_binding`` spec is exactly the ``PropertySpec`` an author would construct."""
        built = property_spec("d", deferred_binding=True)

        hand_constructed = PropertySpec(
            "d",
            context=True,
            strict_validation=False,
            deferred_binding=True,
        )
        assert built == hand_constructed

    def test_non_bool_deferred_binding_raises(self) -> None:
        """A non-bool ``deferred_binding`` is rejected up front, like ``allow_explicit_none``."""
        not_a_bool: Any = "yes"
        with pytest.raises(ValueError, match=r"PropertySpec\('d'\)"):
            property_spec("d", deferred_binding=not_a_bool)
