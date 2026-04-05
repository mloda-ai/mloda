"""
Tests for DefaultOptionKeys enum.

This module tests the DefaultOptionKeys enum which holds option keys and their
default values for mloda feature groups.
"""

from mloda.provider import DefaultOptionKeys


class TestDefaultOptionKeys:
    """Test suite for DefaultOptionKeys enum."""

    def test_reference_time_value(self) -> None:
        """Verify DefaultOptionKeys.reference_time has the correct value."""
        assert DefaultOptionKeys.reference_time.value == "reference_time"

    def test_time_travel_value(self) -> None:
        """Verify DefaultOptionKeys.time_travel has the correct value."""
        assert DefaultOptionKeys.time_travel.value == "time_travel"

    def test_time_related_keys_are_strings(self) -> None:
        """Verify time-related keys have string values."""
        assert isinstance(DefaultOptionKeys.reference_time.value, str)
        assert isinstance(DefaultOptionKeys.time_travel.value, str)

    def test_order_by_value(self) -> None:
        """Verify DefaultOptionKeys.order_by has the correct value."""
        assert DefaultOptionKeys.order_by.value == "order_by"

    def test_order_by_usable_as_string_key(self) -> None:
        """Verify order_by works as a dictionary key since DefaultOptionKeys is a str enum."""
        config: dict[str, str] = {DefaultOptionKeys.order_by: "timestamp_col"}
        assert config["order_by"] == "timestamp_col"

    def test_enum_membership(self) -> None:
        """Verify all expected keys are members of DefaultOptionKeys enum."""
        assert hasattr(DefaultOptionKeys, "time_travel")
        assert hasattr(DefaultOptionKeys, "reference_time")
        assert hasattr(DefaultOptionKeys, "order_by")

    def test_all_member_names_match_values(self) -> None:
        """Every enum member's name must equal its value to prevent silent mismatches.

        A str-enum whose name differs from its value causes silent failures
        when users pass the name as a string literal instead of the enum.
        See: https://github.com/mloda-ai/mloda/issues/271
        """
        for member in DefaultOptionKeys:
            assert member.name == member.value, (
                f"DefaultOptionKeys.{member.name} has value {member.value!r}; name and value must be identical"
            )

    def test_string_literal_matches_enum(self) -> None:
        """Using a string literal equal to the enum name must match the enum value.

        This verifies the str-enum contract: DefaultOptionKeys.X == "X" for all X.
        """
        for member in DefaultOptionKeys:
            assert member == member.name
