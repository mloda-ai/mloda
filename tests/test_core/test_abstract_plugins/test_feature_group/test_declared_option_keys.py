"""Tests for FeatureGroup.declared_option_keys() (issue #603).

``declared_option_keys()`` is a classmethod that returns the top-level parameter
names declared in ``PROPERTY_MAPPING`` -- regardless of whether the mapping was
authored as a hand-written spec dict or via the ``property_spec`` builder. It
never returns the inner allowed-value keys (e.g. "add"/"sub").
"""

from typing import Any

from mloda.provider import DefaultOptionKeys, FeatureGroup, property_spec


class DefaultMappingFeatureGroup(FeatureGroup):
    """No PROPERTY_MAPPING override -- exercises the None default."""


class EmptyMappingFeatureGroup(FeatureGroup):
    """PROPERTY_MAPPING explicitly set to an empty dict -- distinct from the None default."""

    PROPERTY_MAPPING = {}


class HandWrittenMappingFeatureGroup(FeatureGroup):
    """Hand-written PROPERTY_MAPPING spec dict with a single declared parameter."""

    PROPERTY_MAPPING = {
        "operation_type": {
            DefaultOptionKeys.allowed_values: {"add": "Addition", "sub": "Subtraction"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
    }


class BuilderMappingFeatureGroup(FeatureGroup):
    """PROPERTY_MAPPING authored via the property_spec builder."""

    PROPERTY_MAPPING = {
        "operation_type": property_spec(
            "the operation to apply",
            strict=True,
            allowed_values={"add": "Addition", "sub": "Subtraction"},
        ),
    }


class MultiKeyMappingFeatureGroup(FeatureGroup):
    """Multiple declared parameters, mixing hand-written and builder-form specs."""

    PROPERTY_MAPPING = {
        "operation_type": {
            DefaultOptionKeys.allowed_values: {"add": "Addition"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "aggregation_type": property_spec(
            "the aggregation to apply",
            strict=True,
            allowed_values={"sum": "Sum", "mean": "Mean"},
        ),
        "window_size": {
            DefaultOptionKeys.allowed_values: {"3": "three"},
            DefaultOptionKeys.context: False,
            DefaultOptionKeys.strict_validation: False,
        },
    }


def test_declared_option_keys_returns_empty_frozenset_when_property_mapping_none() -> None:
    """PROPERTY_MAPPING unset (None, the default) yields an empty frozenset."""
    result = DefaultMappingFeatureGroup.declared_option_keys()

    assert result == frozenset()


def test_declared_option_keys_returns_empty_frozenset_when_property_mapping_empty_dict() -> None:
    """PROPERTY_MAPPING explicitly set to {} also yields an empty frozenset."""
    result = EmptyMappingFeatureGroup.declared_option_keys()

    assert result == frozenset()


def test_declared_option_keys_returns_top_level_keys_for_hand_written_mapping() -> None:
    """Hand-written spec dict: only the top-level parameter name is returned.

    Inner allowed-value keys ("add"/"sub") and metadata flags must not appear.
    """
    result = HandWrittenMappingFeatureGroup.declared_option_keys()

    assert result == frozenset({"operation_type"})
    assert "add" not in result
    assert "sub" not in result


def test_declared_option_keys_returns_top_level_keys_for_property_spec_builder_form() -> None:
    """property_spec builder form: only the top-level parameter name is returned.

    The allowed_values contents ("add"/"sub") must not leak into the result.
    """
    result = BuilderMappingFeatureGroup.declared_option_keys()

    assert result == frozenset({"operation_type"})
    assert "add" not in result
    assert "sub" not in result


def test_declared_option_keys_returns_all_declared_parameter_names() -> None:
    """Multiple declared parameters all come back, cast to str."""
    result = MultiKeyMappingFeatureGroup.declared_option_keys()

    assert result == frozenset({"operation_type", "aggregation_type", "window_size"})
    assert all(isinstance(key, str) for key in result)


def test_declared_option_keys_is_callable_without_instantiation() -> None:
    """The method is a classmethod: callable directly on the class."""
    result = HandWrittenMappingFeatureGroup.declared_option_keys()

    assert isinstance(result, frozenset)


def test_declared_option_keys_returns_frozenset_type() -> None:
    """Return type is frozenset, not list/set/dict_keys."""
    result = MultiKeyMappingFeatureGroup.declared_option_keys()

    assert isinstance(result, frozenset)
    assert not isinstance(result, (list, set))


def test_declared_option_keys_is_immutable() -> None:
    """The returned frozenset has no mutating methods."""
    result = HandWrittenMappingFeatureGroup.declared_option_keys()

    assert not hasattr(result, "add")
    assert not hasattr(result, "update")


def test_declared_option_keys_snapshot_not_tied_to_underlying_dict() -> None:
    """Mutating the original PROPERTY_MAPPING dict after the call does not
    retroactively change an already-returned frozenset (unlike a dict_keys view)."""
    mapping: dict[str, Any] = {
        "operation_type": {
            DefaultOptionKeys.allowed_values: {"add": "Addition"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
    }

    class MutableMappingFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = mapping

    result = MutableMappingFeatureGroup.declared_option_keys()
    mapping["new_param"] = {"explanation": "added after the snapshot"}

    assert result == frozenset({"operation_type"})
    assert "new_param" not in result
