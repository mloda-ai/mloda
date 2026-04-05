from mloda.core.filter.filter_type_enum import FilterType


class TestFilterTypeEnum:
    def test_members_are_upper_case(self) -> None:
        """FilterType enum members must follow Python UPPER_CASE convention (PEP 8)."""
        for member in FilterType:
            assert member.name == member.name.upper(), (
                f"FilterType.{member.name} should be FilterType.{member.name.upper()}"
            )

    def test_values_are_lowercase_strings(self) -> None:
        """Enum values must remain lowercase strings for backward compatibility."""
        expected = {
            "MIN": "min",
            "MAX": "max",
            "EQUAL": "equal",
            "RANGE": "range",
            "REGEX": "regex",
            "CATEGORICAL_INCLUSION": "categorical_inclusion",
        }
        for name, value in expected.items():
            member = FilterType[name]
            assert member.value == value, f"FilterType.{name}.value should be {value!r}, got {member.value!r}"

    def test_expected_members_exist(self) -> None:
        """All expected filter types are present."""
        expected_names = {"MIN", "MAX", "EQUAL", "RANGE", "REGEX", "CATEGORICAL_INCLUSION"}
        actual_names = {m.name for m in FilterType}
        assert actual_names == expected_names

    def test_member_count(self) -> None:
        """Guard against accidental additions or removals."""
        assert len(FilterType) == 6
