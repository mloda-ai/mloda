"""Tests pinning the safe_field_with_error helper (issue #639 follow-up).

TestSbfixSafeFieldWithErrorSwallowsRaisingStr is a red-phase test: it currently fails because a
caught exception's own str(exc) is read eagerly and escapes the guard.
"""

from mloda.core.abstract_plugins.components.utils import safe_field_with_error


def _sbfix_boom() -> int:
    raise ValueError("sbfix boom")


def _sbfix852_empty_boom() -> int:
    raise RuntimeError()


def _sbfix852_blank_boom() -> int:
    raise RuntimeError("   ")


class TestSbfixSafeFieldWithError:
    """safe_field_with_error(read, fallback) returns (value, None) or (fallback, str(exc))."""

    def test_success_returns_value_and_none(self) -> None:
        assert safe_field_with_error(lambda: 5, 0) == (5, None)

    def test_failure_returns_fallback_and_message(self) -> None:
        value, error = safe_field_with_error(_sbfix_boom, 0)
        assert value == 0
        assert error is not None
        assert "sbfix boom" in error

    def test_empty_message_exception_yields_non_empty_typed_error(self) -> None:
        # An empty-message exception must not collapse to a falsy error (issue #852).
        value, error = safe_field_with_error(_sbfix852_empty_boom, 0)
        assert value == 0
        assert error
        assert "RuntimeError" in error

    def test_blank_message_exception_yields_typed_error(self) -> None:
        # A whitespace-only message must also name the type, not return the blank string (issue #852).
        value, error = safe_field_with_error(_sbfix852_blank_boom, 0)
        assert value == 0
        assert error
        assert "RuntimeError" in error


class TestSbfixSafeFieldWithErrorSwallowsRaisingStr:
    """A caught exception whose own __str__ raises must not escape safe_field_with_error."""

    def test_swallowed_read_with_raising_str_does_not_escape(self) -> None:
        class Bad(Exception):
            def __str__(self) -> str:
                raise RuntimeError("str fails")

        def raises() -> int:
            raise Bad("original message")

        value, error = safe_field_with_error(raises, 0)

        assert value == 0
        # Once fixed, the guarded str() degrades to the type name, same as safe_field's own guard.
        assert error == "Bad"
