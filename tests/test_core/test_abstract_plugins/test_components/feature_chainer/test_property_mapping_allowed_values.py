"""Tests for the allowed_values-aware PROPERTY_MAPPING extractor/validator (issue #543).

PROPERTY_MAPPING specs historically conflated allowed-value->docstring entries,
flag keys, and a magic plain-string ``"explanation"`` doc key in one namespace.
``FeatureChainParser._extract_property_values`` recovered the allowed-value set by
subtracting a hardcoded blocklist. This module pins the ``PropertySpec.allowed_values``
field that replaced that:

* authors DECLARE the accepted values under ``allowed_values``,
* ``_extract_property_values`` returns exactly that mapping, so both membership
  validation and the class-definition default invariant follow automatically,
* nothing else in the spec can widen the value space.

The subtraction fallback (and with it the flattened authoring form) is gone; the
spec SCHEMA that replaces it is pinned in ``test_property_mapping_spec_schema.py``.

Reject semantics: ``match_configuration_feature_chain_parser`` raises ``ValueError``
for a strict value outside the accepted set (verified against the existing parser),
so rejection is asserted via ``pytest.raises(ValueError)``.
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
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    Copied from ``test_property_mapping_default_invariant.py``: tests below define
    FeatureGroup subclasses to exercise ``FeatureGroup.__init_subclass__``. Those
    class objects sit in reference cycles, so we force a collection after each test
    and assert that none of this module's classes remain registered.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


class TestStrictMembershipViaAllowedValues:
    """Strict membership flows through the explicit ``allowed_values`` field.

    This is the ONLY way to declare a value space. The flattened form that once carried the
    same values inline is retired, and is now rejected at class definition (see
    ``test_property_mapping_spec_schema.py::TestUnknownSpecKeyFailsAtClassDefinition``).
    """

    def test_allowed_values_field_accepts_and_rejects(self) -> None:
        """A spec using ``allowed_values`` accepts members and rejects non-members."""

        class AllowedValuesFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": PropertySpec(
                    "Arithmetic operation",
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                    context=True,
                    strict_validation=True,
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        property_mapping = AllowedValuesFeatureGroup.PROPERTY_MAPPING

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


class TestClassDefinitionDefaultInvariantHonorsAllowedValues:
    """The default invariant reads accepted values from ``allowed_values``.

    The invariant fires at ``PropertySpec`` construction, which for a class-level
    PROPERTY_MAPPING is during class definition: the class body never finishes.
    """

    def test_rejects_strict_default_outside_allowed_values(self) -> None:
        """A strict default outside the ``allowed_values`` set rejects at class definition."""
        with pytest.raises(ValueError) as exc_info:

            class BadAllowedValuesDefault(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_op$"
                PROPERTY_MAPPING = {
                    "operation_type": PropertySpec(
                        "Arithmetic operation",
                        allowed_values={"add": "Addition", "sub": "Subtraction"},
                        context=True,
                        strict_validation=True,
                        default="mul",
                    )
                }

                @classmethod
                def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                    return data

        message = str(exc_info.value)
        del exc_info
        assert "PropertySpec" in message
        assert "mul" in message
        assert "allowed_values" in message

    def test_accepts_strict_default_in_allowed_values(self) -> None:
        """A strict default inside the ``allowed_values`` set defines without error."""

        class GoodAllowedValuesDefault(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": PropertySpec(
                    "Arithmetic operation",
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                    context=True,
                    strict_validation=True,
                    default="add",
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        assert GoodAllowedValuesDefault.PROPERTY_MAPPING["operation_type"].default == "add"


class TestNoSilentWidening:
    """The spec's documentation text is never promoted to an allowed value."""

    def test_explanation_not_treated_as_allowed_value(self) -> None:
        """When ``allowed_values`` is present, extraction returns exactly that mapping."""
        spec = PropertySpec(
            "operation_type chooses the arithmetic operation",
            allowed_values={"add": "Addition"},
            context=True,
            strict_validation=True,
        )

        extracted = FeatureChainParser._extract_property_values(spec)
        assert extracted == {"add": "Addition"}

    def test_doc_key_name_rejected_by_strict_validation(self) -> None:
        """The doc field name (``explanation``) is not a member, so strict validation rejects it."""
        property_mapping = {
            "operation_type": PropertySpec(
                "operation_type chooses the arithmetic operation",
                allowed_values={"add": "Addition"},
                context=True,
                strict_validation=True,
            )
        }

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"operation_type": "add"}), property_mapping
            )
            is True
        )
        with pytest.raises(ValueError, match="explanation"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"operation_type": "explanation"}), property_mapping
            )
