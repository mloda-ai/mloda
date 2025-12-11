"""
Tests for DefaultOptionKeys enum.

This module tests the DefaultOptionKeys enum which holds option keys and their
default values for mloda feature groups.
"""

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class TestDefaultOptionKeys:
    """Test suite for DefaultOptionKeys enum."""

    def test_reference_time_value(self) -> None:
        """Verify DefaultOptionKeys.reference_time has the correct value."""
        assert DefaultOptionKeys.reference_time.value == "reference_time"

    def test_time_travel_value(self) -> None:
        """Verify DefaultOptionKeys.time_travel has the correct value."""
        assert DefaultOptionKeys.time_travel.value == "time_travel_filter"

    def test_time_related_keys_are_strings(self) -> None:
        """Verify time-related keys have string values."""
        assert isinstance(DefaultOptionKeys.reference_time.value, str)
        assert isinstance(DefaultOptionKeys.time_travel.value, str)

    def test_enum_membership(self) -> None:
        """Verify time_travel is a member of DefaultOptionKeys enum."""
        assert hasattr(DefaultOptionKeys, "time_travel")
        assert hasattr(DefaultOptionKeys, "reference_time")
