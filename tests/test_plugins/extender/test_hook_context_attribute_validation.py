"""Construction-time validation of HookContext's declared_attributes and result_attributes (issue #2029).

A hand-built HookContext must hold only portable scalars in either attribute field: non-finite
floats, int64-out-of-range ints and non-scalar values are dropped while portable ones are kept,
structural errors (a non-Mapping value or a non-str key) raise TypeError, and degenerate inputs
(None, an empty dict) pass through untouched. The validated fields stay frozen and copy the
source dict on ingest.
"""

import dataclasses
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext


def _make_context(**overrides: Any) -> HookContext:
    required: dict[str, Any] = {
        "hook": ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
        "feature_group_class": "tests.something.FakeFeatureGroup",
        "feature_group_version": "v1",
        "plugin_version": None,
        "feature_names": ("my_feature",),
        "input_features": None,
        "compute_framework_name": "FakeFramework",
    }
    required.update(overrides)
    return HookContext(**required)


def _make_context_with_attribute(field: str, value: Any) -> HookContext:
    """Build a HookContext whose named attribute field carries value."""
    return _make_context(**{field: value})


class TestHookContextAttributeFieldsHoldOnlyPortableScalars:
    """Issue #2029: both attribute fields keep portable scalars and drop everything else at construction."""

    @pytest.mark.parametrize("field", ("declared_attributes", "result_attributes"))
    def test_non_finite_floats_are_dropped_and_the_finite_one_is_kept(self, field: str) -> None:
        attributes: dict[str, Any] = {
            "keep": 1.5,
            "nan": float("nan"),
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
        }

        context = _make_context_with_attribute(field, attributes)

        assert getattr(context, field) == {"keep": 1.5}

    @pytest.mark.parametrize("field", ("declared_attributes", "result_attributes"))
    def test_ints_outside_int64_are_dropped_and_in_range_ints_are_kept(self, field: str) -> None:
        attributes: dict[str, Any] = {
            "max": 2**63 - 1,
            "min": -(2**63),
            "too_big": 2**63,
            "way_big": 2**64,
            "too_small": -(2**64),
        }

        context = _make_context_with_attribute(field, attributes)

        assert getattr(context, field) == {"max": 2**63 - 1, "min": -(2**63)}

    @pytest.mark.parametrize("field", ("declared_attributes", "result_attributes"))
    def test_non_scalar_values_are_dropped_and_scalars_are_kept(self, field: str) -> None:
        attributes: dict[str, Any] = {
            "list": [1],
            "dict": {"a": 1},
            "none": None,
            "tuple": (1, 2),
            "str": "x",
            "bool": True,
            "int": 3,
        }

        context = _make_context_with_attribute(field, attributes)

        assert getattr(context, field) == {"str": "x", "bool": True, "int": 3}


class TestHookContextAttributeStructureErrors:
    """Issue #2029: a non-Mapping value or a non-str key in either attribute field raises TypeError."""

    @pytest.mark.parametrize("value", ([1, 2], [("a", 1)]), ids=["scalar_items", "pair_items"])
    @pytest.mark.parametrize("field", ("declared_attributes", "result_attributes"))
    def test_plain_list_value_raises_type_error(self, field: str, value: Any) -> None:
        with pytest.raises(TypeError):
            _make_context_with_attribute(field, value)

    @pytest.mark.parametrize("field", ("declared_attributes", "result_attributes"))
    def test_non_str_key_raises_type_error(self, field: str) -> None:
        with pytest.raises(TypeError):
            _make_context_with_attribute(field, {1: "x"})


class TestHookContextDegenerateAttributeInputs:
    """Issue #2029: degenerate but well-formed inputs (None, empty dict) pass through untouched."""

    @pytest.mark.parametrize("field", ("declared_attributes", "result_attributes"))
    def test_none_stays_none(self, field: str) -> None:
        context = _make_context_with_attribute(field, None)

        assert getattr(context, field) is None

    @pytest.mark.parametrize("field", ("declared_attributes", "result_attributes"))
    def test_empty_mapping_stays_empty(self, field: str) -> None:
        context = _make_context_with_attribute(field, {})

        assert getattr(context, field) == {}


class TestHookContextValidatedAttributeFieldsStayFrozenAndCopied:
    """Issue #2029: validating the attribute fields must not drop the freeze or the copy-on-ingest."""

    @pytest.mark.parametrize("field", ("declared_attributes", "result_attributes"))
    def test_validated_field_stays_frozen_after_construction(self, field: str) -> None:
        context = _make_context_with_attribute(field, {"unit": "m"})
        before = getattr(context, field)

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(context, field, {"forged": 1})

        assert getattr(context, field) == before

    @pytest.mark.parametrize("field", ("declared_attributes", "result_attributes"))
    def test_mutating_the_source_dict_after_construction_does_not_change_the_field(self, field: str) -> None:
        source: dict[str, Any] = {"unit": "m"}

        context = _make_context_with_attribute(field, source)
        source["late"] = "yes"

        assert getattr(context, field) == {"unit": "m"}
