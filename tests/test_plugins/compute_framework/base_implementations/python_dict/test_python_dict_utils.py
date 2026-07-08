"""Red-phase tests for the ``columnar_to_rows`` and ``homogenize_rows`` python_dict_utils helpers (issue 648)."""

from typing import Any

import pytest

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import (
    columnar_to_rows,
    homogenize_rows,
    rows_to_columnar,
)


class TestColumnarToRows:
    def test_list_input_passes_through_by_identity(self) -> None:
        """A row-wise list input is returned unchanged, as the same object."""
        rows: list[dict[str, Any]] = [{"a": 1}, {"a": 2}]
        result = columnar_to_rows(rows)
        assert result is rows

    def test_columnar_dict_pivots_to_rows(self) -> None:
        """A columnar dict pivots into a list of row dicts."""
        data: dict[str, list[Any]] = {"a": [1, 2], "b": ["x", "y"]}
        assert columnar_to_rows(data) == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_pivot_preserves_key_insertion_order(self) -> None:
        """Each row dict keeps the columnar dict's key insertion order."""
        data: dict[str, list[Any]] = {"b": [1], "a": [2], "c": [3]}
        result = columnar_to_rows(data)
        assert [list(row.keys()) for row in result] == [["b", "a", "c"]]

    def test_schemaless_empty_dict_returns_empty_list(self) -> None:
        """The schema-less ``{}`` yields no rows."""
        assert columnar_to_rows({}) == []

    def test_schema_bearing_zero_row_dict_returns_empty_list(self) -> None:
        """A schema-bearing zero-row dict yields no rows."""
        assert columnar_to_rows({"doc_id": []}) == []

    def test_none_cell_values_survive_pivot(self) -> None:
        """``None`` cells are carried through to the row dicts."""
        data: dict[str, list[Any]] = {"a": [1, None], "b": [None, 2]}
        assert columnar_to_rows(data) == [{"a": 1, "b": None}, {"a": None, "b": 2}]

    def test_ragged_column_lengths_raise(self) -> None:
        """Columns of differing length are invalid columnar input."""
        with pytest.raises(ValueError):
            columnar_to_rows({"a": [1, 2], "b": [1]})

    def test_non_list_column_value_raises(self) -> None:
        """A dict whose value is not a list is not columnar and is rejected."""
        with pytest.raises(ValueError):
            columnar_to_rows({"a": 5})

    def test_none_input_raises(self) -> None:
        with pytest.raises(ValueError):
            columnar_to_rows(None)  # type: ignore[arg-type]

    def test_string_input_raises(self) -> None:
        with pytest.raises(ValueError):
            columnar_to_rows("text")  # type: ignore[arg-type]

    def test_int_input_raises(self) -> None:
        with pytest.raises(ValueError):
            columnar_to_rows(5)  # type: ignore[arg-type]

    def test_round_trip_with_rows_to_columnar(self) -> None:
        """``columnar_to_rows(rows_to_columnar(rows))`` is the identity for homogeneous rows."""
        rows: list[dict[str, Any]] = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": None, "b": "z"}]
        assert columnar_to_rows(rows_to_columnar(rows)) == rows


class TestHomogenizeRows:
    def test_missing_keys_backfilled_with_none(self) -> None:
        """Missing keys are filled with ``None``, never with ``[]`` or another placeholder."""
        rows: list[dict[str, Any]] = [{"a": 1}, {"a": 2, "b": 3}]
        result = homogenize_rows(rows)
        assert result == [{"a": 1, "b": None}, {"a": 2, "b": 3}]
        assert result[0]["b"] is None

    def test_key_order_is_first_occurrence_union(self) -> None:
        """Every output row carries the union of keys in first-occurrence order."""
        rows: list[dict[str, Any]] = [{"b": 1}, {"a": 2}, {"c": 3}]
        result = homogenize_rows(rows)
        assert [list(row.keys()) for row in result] == [["b", "a", "c"]] * 3

    def test_union_key_order_with_interleaved_multi_key_rows(self) -> None:
        """Multi-key rows interleave into a first-occurrence union order shared by every row."""
        rows: list[dict[str, Any]] = [{"a": 1, "c": 2}, {"b": 3}]
        result = homogenize_rows(rows)
        assert [list(row.keys()) for row in result] == [["a", "c", "b"]] * 2
        assert result == [{"a": 1, "c": 2, "b": None}, {"a": None, "c": None, "b": 3}]

    def test_uniform_rows_returned_equal_but_as_new_objects(self) -> None:
        """Already-uniform rows come back equal, yet as fresh dict objects."""
        rows: list[dict[str, Any]] = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = homogenize_rows(rows)
        assert result == rows
        assert result[0] is not rows[0]

    def test_empty_list_returns_empty_list(self) -> None:
        assert homogenize_rows([]) == []

    def test_explicit_none_values_are_kept(self) -> None:
        """Values that are explicitly ``None`` are preserved, not treated as missing."""
        rows: list[dict[str, Any]] = [{"a": None, "b": 1}, {"a": 2, "b": None}]
        assert homogenize_rows(rows) == [{"a": None, "b": 1}, {"a": 2, "b": None}]

    def test_non_dict_item_raises(self) -> None:
        with pytest.raises(ValueError):
            homogenize_rows([{"a": 1}, "not a dict"])  # type: ignore[list-item]

    def test_output_is_accepted_by_rows_to_columnar(self) -> None:
        """Homogenized mixed-key rows compose with ``rows_to_columnar`` into a columnar dict."""
        rows: list[dict[str, Any]] = [{"a": 1}, {"b": 2}]
        assert rows_to_columnar(homogenize_rows(rows)) == {"a": [1, None], "b": [None, 2]}
