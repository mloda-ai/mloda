"""Tests for the SQL window-function primitives in sql_window.py.

These tests exercise dataclass construction and validation only — they have
no relation/backend dependency. Backend-level window tests live in the
SqlRelationWindowTestMixin (to be added in a follow-up commit).
"""

import pytest

from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    CurrentRow,
    Following,
    OrderBy,
    Preceding,
    Unbounded,
    WindowFrame,
)


def test_orderby_bare_column_constructs() -> None:
    """OrderBy('age') must construct cleanly."""
    OrderBy("age")


def test_orderby_descending_true_constructs() -> None:
    """OrderBy('age', descending=True) must construct cleanly."""
    OrderBy("age", descending=True)


def test_orderby_nulls_first_constructs() -> None:
    """OrderBy('age', nulls='first') must construct cleanly."""
    OrderBy("age", nulls="first")


def test_orderby_nulls_last_constructs() -> None:
    """OrderBy('age', nulls='last') must construct cleanly."""
    OrderBy("age", nulls="last")


def test_orderby_invalid_nulls_raises() -> None:
    """OrderBy must reject a nulls value outside {'first','last'} at construction time."""
    with pytest.raises(ValueError, match="nulls"):
        OrderBy("age", nulls="middle")  # type: ignore[arg-type]


def test_orderby_empty_column_raises() -> None:
    """OrderBy must reject an empty column name at construction time."""
    with pytest.raises(ValueError, match="column"):
        OrderBy("")


def test_window_frame_kind_invalid_string_raises() -> None:
    """WindowFrame must reject any kind outside the Literal alphabet at construction time."""
    with pytest.raises(ValueError, match="kind"):
        WindowFrame(kind="invalid", start=Unbounded(), end=Unbounded())  # type: ignore[arg-type]


def test_window_frame_kind_uppercase_rejected() -> None:
    """WindowFrame must reject 'ROWS' even though .upper() would render it harmlessly — the Literal alphabet is lowercase."""
    with pytest.raises(ValueError, match="kind"):
        WindowFrame(kind="ROWS", start=Unbounded(), end=Unbounded())  # type: ignore[arg-type]


def test_window_frame_kind_sql_injection_attempt_raises() -> None:
    """Construction-time validation must block the canonical injection payload before it can reach the SQL renderer."""
    with pytest.raises(ValueError, match="kind"):
        WindowFrame(
            kind="ROWS) AS x; DROP TABLE y; --",  # type: ignore[arg-type]
            start=Unbounded(),
            end=Unbounded(),
        )


def test_window_frame_accepts_each_valid_kind() -> None:
    """The three valid kinds — rows, range, groups — must all construct without error."""
    for kind in ("rows", "range", "groups"):
        WindowFrame(kind=kind, start=Unbounded(), end=Unbounded())


def test_preceding_offset_string_raises() -> None:
    """Preceding must reject a non-int offset; otherwise a string payload would land verbatim in the f-string."""
    with pytest.raises(TypeError, match="offset"):
        Preceding(offset="1; DROP TABLE x; --")  # type: ignore[arg-type]


def test_preceding_offset_float_raises() -> None:
    """Preceding must reject a float offset; the Literal-equivalent contract is int only."""
    with pytest.raises(TypeError, match="offset"):
        Preceding(offset=1.5)  # type: ignore[arg-type]


def test_preceding_offset_bool_raises() -> None:
    """Preceding must reject bool — isinstance(True, int) is True in Python but bool violates the int contract."""
    with pytest.raises(TypeError, match="offset"):
        Preceding(offset=True)


def test_preceding_offset_int_accepted() -> None:
    """Preceding(offset=1) must construct cleanly."""
    Preceding(offset=1)


def test_following_offset_string_raises() -> None:
    """Following must reject a non-int offset (parallel to Preceding)."""
    with pytest.raises(TypeError, match="offset"):
        Following(offset="1; DROP TABLE x; --")  # type: ignore[arg-type]


def test_following_offset_float_raises() -> None:
    """Following must reject a float offset."""
    with pytest.raises(TypeError, match="offset"):
        Following(offset=2.0)  # type: ignore[arg-type]


def test_following_offset_bool_raises() -> None:
    """Following must reject bool — same reason as Preceding."""
    with pytest.raises(TypeError, match="offset"):
        Following(offset=False)


def test_following_offset_int_accepted() -> None:
    """Following(offset=2) must construct cleanly."""
    Following(offset=2)


def test_preceding_offset_negative_raises() -> None:
    """Preceding must reject a negative offset; SQL n PRECEDING requires n >= 0."""
    with pytest.raises(ValueError, match="offset|negative"):
        Preceding(offset=-1)


def test_following_offset_negative_raises() -> None:
    """Following must reject a negative offset; SQL n FOLLOWING requires n >= 0."""
    with pytest.raises(ValueError, match="offset|negative"):
        Following(offset=-1)


def test_preceding_offset_zero_accepted() -> None:
    """Preceding(offset=0) must construct cleanly; 0 PRECEDING is valid SQL equivalent to CURRENT ROW."""
    Preceding(offset=0)


def test_following_offset_zero_accepted() -> None:
    """Following(offset=0) must construct cleanly; 0 FOLLOWING is valid SQL equivalent to CURRENT ROW."""
    Following(offset=0)


def test_window_frame_following_then_preceding_raises() -> None:
    """WindowFrame must reject start=Following(2), end=Preceding(1); start must not be after end on the row axis."""
    with pytest.raises(ValueError, match="start|end|order"):
        WindowFrame(kind="rows", start=Following(2), end=Preceding(1))


def test_window_frame_preceding_descending_raises() -> None:
    """WindowFrame must reject Preceding(1) -> Preceding(3); -1 > -3 on the row axis."""
    with pytest.raises(ValueError, match="start|end|order"):
        WindowFrame(kind="rows", start=Preceding(1), end=Preceding(3))


def test_window_frame_following_descending_raises() -> None:
    """WindowFrame must reject Following(3) -> Following(1); 3 > 1 on the row axis."""
    with pytest.raises(ValueError, match="start|end|order"):
        WindowFrame(kind="rows", start=Following(3), end=Following(1))


def test_window_frame_current_row_to_current_row_accepted() -> None:
    """WindowFrame(CurrentRow, CurrentRow) must construct cleanly; equality on the row axis is allowed."""
    WindowFrame(kind="rows", start=CurrentRow(), end=CurrentRow())


def test_window_frame_unbounded_both_sides_accepted() -> None:
    """WindowFrame(Unbounded, Unbounded) must construct cleanly; -inf < +inf."""
    WindowFrame(kind="rows", start=Unbounded(), end=Unbounded())


def test_window_frame_preceding_to_following_accepted() -> None:
    """WindowFrame(Preceding(2), Following(3)) must construct cleanly; -2 < 3."""
    WindowFrame(kind="rows", start=Preceding(2), end=Following(3))
