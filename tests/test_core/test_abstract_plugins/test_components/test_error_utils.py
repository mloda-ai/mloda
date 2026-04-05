"""Tests for error_utils helper functions."""

from mloda.core.abstract_plugins.components.error_utils import REPORT_URL, internal_invariant_error


class TestInternalInvariantError:
    """Tests for internal_invariant_error message builder."""

    def test_minimal_message_contains_invariant(self) -> None:
        result = internal_invariant_error("left_uuids must not be empty")
        assert "Internal error: left_uuids must not be empty" in result

    def test_minimal_message_contains_report_url(self) -> None:
        result = internal_invariant_error("some invariant")
        assert REPORT_URL in result

    def test_message_with_actual_values(self) -> None:
        result = internal_invariant_error(
            "Expected exactly 1 UUID.",
            actual_values="uuids={UUID('abc'), UUID('def')}",
        )
        assert "Internal error: Expected exactly 1 UUID." in result
        assert "Actual state: uuids={UUID('abc'), UUID('def')}" in result
        assert REPORT_URL in result

    def test_message_with_hint(self) -> None:
        result = internal_invariant_error(
            "from_cfw is None.",
            hint="Ensure the source framework is provided before execution.",
        )
        assert "Internal error: from_cfw is None." in result
        assert "Ensure the source framework is provided before execution." in result
        assert REPORT_URL in result

    def test_message_with_all_fields(self) -> None:
        result = internal_invariant_error(
            "Discriminator mismatch.",
            actual_values="left=None, right={'key': 'val'}",
            hint="Both discriminators must be set.",
        )
        lines = result.split("\n")
        assert lines[0] == "Internal error: Discriminator mismatch."
        assert lines[1] == "Actual state: left=None, right={'key': 'val'}"
        assert lines[2] == "Both discriminators must be set."
        assert REPORT_URL in lines[3]

    def test_empty_actual_values_omitted(self) -> None:
        result = internal_invariant_error("test invariant", actual_values="")
        assert "Actual state" not in result

    def test_empty_hint_omitted(self) -> None:
        result = internal_invariant_error("test invariant", hint="")
        lines = result.split("\n")
        assert len(lines) == 2
