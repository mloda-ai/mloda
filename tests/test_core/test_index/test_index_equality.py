from __future__ import annotations

from mloda.user import Index


class TestIndexEquality:
    """Unit tests for Index.__eq__ / __hash__ per Python's data model."""

    def test_eq_none_returns_false(self) -> None:
        """Comparing an Index against None must be False, not raise."""
        assert (Index(("a",)) == None) is False  # noqa: E711

    def test_membership_check_with_mixed_list(self) -> None:
        """`in` must fall through non-Index elements without raising."""
        assert Index(("a",)) in ["x", Index(("a",))]

    def test_dict_lookup_with_colliding_tuple_key_misses(self) -> None:
        """A plain tuple key must never match an Index, even on hash collision."""
        d = {("a",): 1}
        assert d.get(Index(("a",))) is None  # type: ignore[call-overload]

    def test_two_indexes_not_equal(self) -> None:
        """Real Index-to-Index comparison still works after the fix."""
        assert Index(("a",)) != Index(("b",))
