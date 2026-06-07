"""Unit tests for ``FeatureChainParserMixin.derive_context_key_schema`` (TDD red phase).

These tests pin down the contract for a helper that builds a ``context_key_schema()``
dict from a config-based feature group's ``PROPERTY_MAPPING``. The helper does not
exist yet, so every test currently fails with ``AttributeError``; they go green once
the Green Agent implements ``derive_context_key_schema``.

Contract:
    - With a non-empty ``PROPERTY_MAPPING``: returns ``{str(key): None for key in PROPERTY_MAPPING}``
      (every declared property name as a string, type-check skipped via ``None``).
    - With ``PROPERTY_MAPPING = None`` or no ``PROPERTY_MAPPING``: returns ``None``.
    - With ``PROPERTY_MAPPING = {}``: returns ``None`` (preserves permissive behavior).
"""

from __future__ import annotations

from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureChainParserMixin


class MappingWithKeys(FeatureChainParserMixin):
    """Mixin subclass with a mix of context-marked and plain group-default keys."""

    PROPERTY_MAPPING = {
        "partition_by": {
            "explanation": "partition columns",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        "group_default_key": {
            "explanation": "a plain group-default property (no context flag)",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "source features",
            DefaultOptionKeys.context: True,
        },
    }


class MappingNone(FeatureChainParserMixin):
    """Mixin subclass that explicitly sets PROPERTY_MAPPING to None."""

    PROPERTY_MAPPING = None


class MappingEmpty(FeatureChainParserMixin):
    """Mixin subclass with an empty PROPERTY_MAPPING."""

    PROPERTY_MAPPING: dict[str, object] = {}


def test_derive_returns_all_property_mapping_keys_with_none_values() -> None:
    """Schema keys equal all PROPERTY_MAPPING names (as strings); all values are None."""
    schema = MappingWithKeys.derive_context_key_schema()

    assert schema is not None
    expected_keys = {str(key) for key in MappingWithKeys.PROPERTY_MAPPING}
    assert set(schema.keys()) == expected_keys
    assert all(value is None for value in schema.values())


def test_derive_returns_none_when_property_mapping_is_none() -> None:
    """A None PROPERTY_MAPPING yields a None schema (permissive)."""
    assert MappingNone.derive_context_key_schema() is None


def test_derive_returns_none_when_property_mapping_is_empty() -> None:
    """An empty PROPERTY_MAPPING yields a None schema (permissive)."""
    assert MappingEmpty.derive_context_key_schema() is None
