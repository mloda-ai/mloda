"""Failing tests pinning the safe_field_with_error helper (issue #639 follow-up).

The helper does not exist yet; the import failure is the intended red failure.
"""

from mloda.core.abstract_plugins.components.utils import safe_field_with_error


def _sbfix_boom() -> int:
    raise ValueError("sbfix boom")


class TestSbfixSafeFieldWithError:
    """safe_field_with_error(read, fallback) returns (value, None) or (fallback, str(exc))."""

    def test_success_returns_value_and_none(self) -> None:
        assert safe_field_with_error(lambda: 5, 0) == (5, None)

    def test_failure_returns_fallback_and_message(self) -> None:
        value, error = safe_field_with_error(_sbfix_boom, 0)
        assert value == 0
        assert error is not None
        assert "sbfix boom" in error
