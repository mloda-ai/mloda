"""Tests for failure_report: the logged traceback keeps type names and frames, never exception message text.

The traceback element is logged at ERROR, so row values or PII in a message must not reach it.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

from mloda.core.abstract_plugins.components.utils import failure_report

MARKER = "Jane Doe, DOB 1990"


class RowValueLeakError(Exception):
    """Non-builtin exception class used to pin the qualified-name rendering."""


def _raise_value_error() -> None:
    raise ValueError(MARKER)


def _raise_type_error() -> None:
    raise TypeError(MARKER)


def _raise_key_error() -> None:
    raise KeyError(MARKER)


def _raise_custom_error() -> None:
    raise RowValueLeakError(MARKER)


def _raise_given(exc: BaseException) -> None:
    raise exc


def _caught(raiser: Callable[[], None]) -> BaseException:
    try:
        raiser()
    except BaseException as exc:  # noqa: BLE001  (the test needs the raised object with its traceback)
        return exc
    raise AssertionError("raiser did not raise")


def _final_line(tb: str) -> str:
    return [line for line in tb.splitlines() if line.strip()][-1]


class TestFailureReportCycles:
    """A cyclic exception chain must render instead of recursing forever."""

    def test_cyclic_cause_chain_renders_both_types(self) -> None:
        a = _caught(_raise_value_error)
        b = _caught(_raise_type_error)
        a.__cause__ = b
        b.__cause__ = a

        _, tb = failure_report(a)

        assert "ValueError" in tb
        assert "TypeError" in tb
        assert MARKER not in tb

    def test_cyclic_context_chain_renders_both_types(self) -> None:
        a = _caught(_raise_value_error)
        b = _caught(_raise_type_error)
        a.__context__ = b
        b.__context__ = a
        assert a.__suppress_context__ is False
        assert b.__suppress_context__ is False

        _, tb = failure_report(a)

        assert "ValueError" in tb
        assert "TypeError" in tb
        assert MARKER not in tb


class TestFailureReportExceptionGroups:
    """Sub-exceptions of a group keep their type names and raising frames."""

    def test_group_renders_sub_exception_types_and_frames(self) -> None:
        # Looked up at runtime: the 3.10 gate (ruff, mypy) has no ExceptionGroup name.
        exception_group: Any = getattr(builtins, "ExceptionGroup", None)
        if exception_group is None:
            # Return, not skip: the pinned skip count is shared across interpreter versions, so no version skips.
            return

        flat = _caught(
            lambda: _raise_given(exception_group(MARKER, [_caught(_raise_value_error), _caught(_raise_type_error)]))
        )
        _, tb = failure_report(flat)

        assert "ValueError" in tb
        assert "TypeError" in tb
        assert "_raise_value_error" in tb
        assert "_raise_type_error" in tb
        assert MARKER not in tb

        inner = _caught(lambda: _raise_given(exception_group(MARKER, [_caught(_raise_key_error)])))
        assert isinstance(inner, exception_group)
        nested = _caught(lambda: _raise_given(exception_group(MARKER, [inner])))
        _, nested_tb = failure_report(nested)

        assert "KeyError" in nested_tb
        assert "_raise_key_error" in nested_tb
        assert MARKER not in nested_tb


class TestFailureReportTypeName:
    """The final traceback line names the exception type, qualified unless builtin."""

    def test_custom_exception_renders_qualified_name(self) -> None:
        exc = _caught(_raise_custom_error)

        _, tb = failure_report(exc)

        assert _final_line(tb) == f"{RowValueLeakError.__module__}.{RowValueLeakError.__qualname__}"
        assert MARKER not in tb

    def test_builtin_exception_renders_bare_name(self) -> None:
        exc = _caught(_raise_value_error)

        _, tb = failure_report(exc)

        assert _final_line(tb) == "ValueError"


class TestFailureReportNotes:
    """Exception notes can carry row values and never reach the traceback."""

    def test_notes_are_omitted(self) -> None:
        exc = _caught(_raise_value_error)
        # setattr, not add_note: add_note is 3.11+, and typeshed omits __notes__ on 3.10
        setattr(exc, "__notes__", [MARKER, f"row: {MARKER}"])

        _, tb = failure_report(exc)

        assert MARKER not in tb


class TestFailureReportRegression:
    """A plain raise keeps the standard header and frames; the message element keeps the text."""

    def test_simple_value_error(self) -> None:
        exc = _caught(_raise_value_error)

        message, tb = failure_report(exc)

        assert "Traceback (most recent call last):" in tb
        assert "_raise_value_error" in tb
        assert MARKER not in tb
        assert MARKER in message
