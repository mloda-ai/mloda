"""The per-key validator behind ``FeatureGroup.PROPERTY_MAPPING``. ``DeclarationSurface`` names the single surface."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.declaration_surface import (
    DeclarationSurface,
    validate_property_spec,
)


def _dsurf_match_guard(value: Any) -> bool:
    return True


class TestDeclarationSurfaceEnum:
    def test_feature_group_attr(self) -> None:
        assert DeclarationSurface.FEATURE_GROUP.attr == "PROPERTY_MAPPING"

    def test_exactly_one_member(self) -> None:
        assert {member.name for member in DeclarationSurface} == {"FEATURE_GROUP"}


class TestValidatePropertySpec:
    """The per-key rules, parametrized by surface; ``via`` appends the reached-through suffix."""

    def test_non_property_spec_value_rejected_on_feature_group_surface(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", 42, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfOwner.PROPERTY_MAPPING['dsurf_key'] is a int, not a PropertySpec." in message

    def test_via_suffix_is_appended(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfMixin", "dsurf_key", 42, DeclarationSurface.FEATURE_GROUP, via="DsurfGroup")

        message = str(exc_info.value)
        del exc_info
        assert message.endswith(" (reached defining DsurfGroup)")

    def test_no_via_suffix_when_via_is_none(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", 42, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "(reached defining" not in message

    def test_feature_group_rejects_framework_set(self) -> None:
        spec = PropertySpec("x", default=None, framework_set=True)

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "framework_set" in message

    def test_feature_group_rejects_scalar_only(self) -> None:
        spec = PropertySpec("x", allowed_values=("a", "b"), strict_validation=True, scalar_only=True, default="a")

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "scalar_only" in message

    def test_feature_group_accepts_match_guard(self) -> None:
        spec = PropertySpec("x", match_guard=_dsurf_match_guard, default=None)

        result = validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        assert result is spec

    def test_feature_group_accepts_context_false(self) -> None:
        spec = PropertySpec("x", context=False, default=None)

        result = validate_property_spec("DsurfOwner", "dsurf_key", spec, DeclarationSurface.FEATURE_GROUP)

        assert result is spec

    def test_feature_group_surface_message_contains_property_mapping_not_reader_options(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", 42, DeclarationSurface.FEATURE_GROUP)

        message = str(exc_info.value)
        del exc_info
        assert "PROPERTY_MAPPING" in message
        assert "READER_OPTIONS" not in message

    def test_no_message_ever_mentions_the_retired_surface_marker_constant(self) -> None:
        messages = []

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", "bad", DeclarationSurface.FEATURE_GROUP)
        messages.append(str(exc_info.value))
        del exc_info

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec(
                "DsurfOwner",
                "dsurf_key",
                PropertySpec("x", framework_set=True, default=None),
                DeclarationSurface.FEATURE_GROUP,
            )
        messages.append(str(exc_info.value))
        del exc_info

        for message in messages:
            assert "PROPERTY_MAPPING_SURFACE" not in message
