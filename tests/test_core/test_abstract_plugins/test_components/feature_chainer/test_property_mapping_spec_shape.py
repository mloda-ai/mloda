"""The PROPERTY_MAPPING shape rules survive, relocated to ``PropertySpec`` construction (issue #694).

The dict-spec era enforced these value-shape rules at class-definition time, from
``validate_property_mapping_defaults``. The dict form is gone, but every rule survives: it now
fires while the ``PropertySpec`` literal inside the class body is being CONSTRUCTED. For the
author the moment is the same, module import, and stricter: the FeatureGroup class never comes
into existence. Because there is no class yet to name, the error identifies the offending spec
by its explanation, with the ``PropertySpec('...')`` prefix.

The five rules, one test each, written as an author would hit them (inside a FeatureGroup
class body):

1. a str/bytes ``allowed_values`` (the forgotten-comma bug) would make membership a substring test,
2. a non-collection ``allowed_values`` is not a value space at all,
3. ``strict_validation`` must be a real bool, not merely truthy,
4. ``element_validator``, ``required_when`` and ``match_guard`` must be callable,
5. ``strict_validation=True`` needs a non-empty ``allowed_values`` or an ``element_validator``.

Fine-grained constructor coverage (normalization, every message variant) lives in
``test_property_spec_type.py``; this module pins the relocation.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    Mirrors ``test_property_spec_hard_break.py``: the tests below define FeatureGroup
    subclasses, which linger in ``FeatureGroup.__subclasses__()`` until a GC cycle runs and
    would otherwise be seen by tests that enumerate feature groups.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


def _spec(*args: Any, **kwargs: Any) -> PropertySpec:
    """Build a ``PropertySpec`` through an untyped seam.

    The type-invalid constructions below (scalar value space, non-bool flag, non-callable
    validators) must stay runtime tests; routing them through ``Any`` keeps the module
    mypy --strict clean.
    """
    return PropertySpec(*args, **kwargs)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


class TestShapeRulesFireInsideTheClassBody:
    """Each rule raises while the class body evaluates, so the class never comes into existence."""

    @pytest.mark.parametrize("word", ["add", b"add"], ids=["str", "bytes"])
    def test_str_or_bytes_allowed_values_prevents_the_class(self, word: Any) -> None:
        """Rule 1: the forgotten-comma bug would turn membership into a substring test."""
        with pytest.raises(ValueError, match="(?i)substring"):

            class Shape694SubstringFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "shape694_operation": PropertySpec(
                        "The arithmetic operation to apply",
                        # A forgotten comma: ("add") is the str "add", not the tuple ("add",).
                        allowed_values=word,
                        strict_validation=True,
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        assert all(c.__name__ != "Shape694SubstringFeatureGroup" for c in get_all_subclasses(FeatureGroup)), (
            "the constructor raised inside the class body, so the class must never come into existence"
        )

    def test_non_collection_allowed_values_prevents_the_class(self) -> None:
        """Rule 2: a scalar is not a value space, and never a swallowed TypeError at match time."""
        with pytest.raises(ValueError, match="not a Mapping or an iterable"):

            class Shape694ScalarValueSpaceFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"shape694_window": _spec("Size of the time window", allowed_values=5)}

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

    def test_non_bool_strict_validation_prevents_the_class(self) -> None:
        """Rule 3: a truthy ``"false"`` must never silently enable strict matching."""
        with pytest.raises(ValueError, match="must be a bool"):

            class Shape694TruthyFlagFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "shape694_operation": _spec(
                        "The arithmetic operation to apply",
                        allowed_values={"add": "Addition"},
                        strict_validation="false",
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

    @pytest.mark.parametrize("field", ["element_validator", "required_when", "match_guard"])
    def test_non_callable_validator_prevents_the_class(self, field: str) -> None:
        """Rule 4: a non-callable validator can never escape as a TypeError out of matching."""
        with pytest.raises(ValueError, match=f"{field} must be callable"):

            class Shape694NonCallableValidatorFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"shape694_window": _spec("Size of the time window", **{field: "not callable"})}

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

    def test_strict_without_a_value_space_prevents_the_class(self) -> None:
        """Rule 5: strict with nothing to validate against would reject every value."""
        with pytest.raises(ValueError, match="needs a non-empty allowed_values or an element_validator"):

            class Shape694EmptyStrictFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "shape694_operation": PropertySpec("The arithmetic operation to apply", strict_validation=True)
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

    def test_the_error_names_the_spec_by_its_explanation(self) -> None:
        """With no class to name, the ``PropertySpec('...')`` prefix locates the offending spec."""
        with pytest.raises(ValueError, match=r"PropertySpec\('The arithmetic operation to apply'\)"):

            class Shape694PrefixFeatureGroup(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "shape694_operation": PropertySpec(
                        "The arithmetic operation to apply",
                        allowed_values="add",
                        strict_validation=True,
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data


class TestWellFormedSpecsStillDefine:
    """The relocation rejects only malformed specs: a well-formed mapping defines and matches."""

    def test_well_formed_specs_define_and_drive_matching(self) -> None:
        """Every shape-checked field, correctly shaped, defines and enforces at match time."""

        class Shape694WellFormedFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {
                "shape694_operation": PropertySpec(
                    "The arithmetic operation to apply",
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                    context=True,
                    strict_validation=True,
                    default="add",
                ),
                "shape694_window": PropertySpec(
                    "Size of the time window",
                    context=True,
                    strict_validation=True,
                    element_validator=_positive_int,
                    default=7,
                ),
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert (
            Shape694WellFormedFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"shape694_operation": "sub", "shape694_window": 7})
            )
            is True
        )
        assert (
            Shape694WellFormedFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"shape694_operation": "mul", "shape694_window": 7})
            )
            is False
        )
        assert (
            Shape694WellFormedFeatureGroup.match_feature_group_criteria(
                "any_feature", Options(context={"shape694_operation": "sub", "shape694_window": -3})
            )
            is False
        )
