"""The per-key validator behind ``FeatureGroup.PROPERTY_MAPPING``. Messages name ``PROPERTY_MAPPING``."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.declaration_surface import (
    validate_property_spec,
)


def _dsurf_match_guard(value: Any) -> bool:
    return True


class TestValidatePropertySpec:
    """The per-key rules."""

    def test_non_property_spec_value_rejected_on_feature_group_surface(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", 42)

        message = str(exc_info.value)
        del exc_info
        assert "DsurfOwner.PROPERTY_MAPPING['dsurf_key'] is a int, not a PropertySpec." in message

    def test_feature_group_accepts_match_guard(self) -> None:
        spec = PropertySpec("x", match_guard=_dsurf_match_guard, default=None)

        result = validate_property_spec("DsurfOwner", "dsurf_key", spec)

        assert result is spec

    def test_feature_group_accepts_context_false(self) -> None:
        spec = PropertySpec("x", context=False, default=None)

        result = validate_property_spec("DsurfOwner", "dsurf_key", spec)

        assert result is spec

    def test_feature_group_surface_message_contains_property_mapping_not_reader_options(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", 42)

        message = str(exc_info.value)
        del exc_info
        assert "PROPERTY_MAPPING" in message
        assert "READER_OPTIONS" not in message

    def test_no_message_ever_mentions_the_retired_surface_marker_constant(self) -> None:
        messages = []

        with pytest.raises(ValueError) as exc_info:
            validate_property_spec("DsurfOwner", "dsurf_key", "bad")
        messages.append(str(exc_info.value))
        del exc_info

        for message in messages:
            assert "PROPERTY_MAPPING_SURFACE" not in message
