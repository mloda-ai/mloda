"""Red-phase tests for the ``columnar_to_rows``, ``homogenize_rows`` (issue 648) and ``result_rows``
(issue 717) python_dict_utils helpers."""

from typing import Any

import pytest

from mloda.user import RunResult
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import (
    columnar_to_rows,
    homogenize_rows,
    is_columnar,
    result_rows,
    rows_to_columnar,
    validate_columnar_dict,
)


class TestColumnarToRows:
    def test_list_input_raises(self) -> None:
        """Strict dict-only: a row-wise list is no longer passed through."""
        with pytest.raises(ValueError):
            columnar_to_rows([{"a": 1}])  # type: ignore[arg-type]

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


class TestIsColumnar:
    def test_equal_length_list_columns_is_true(self) -> None:
        assert is_columnar({"a": [1, 2], "b": [3, 4]}) is True

    def test_empty_dict_is_true(self) -> None:
        """The schema-less ``{}`` counts as columnar."""
        assert is_columnar({}) is True

    def test_zero_row_dict_is_true(self) -> None:
        assert is_columnar({"a": []}) is True

    def test_ragged_column_lengths_is_false(self) -> None:
        assert is_columnar({"a": [1, 2], "b": [1]}) is False

    def test_non_list_value_is_false(self) -> None:
        assert is_columnar({"a": 5}) is False

    def test_tuple_value_is_false(self) -> None:
        """A tuple column is not a list and is not columnar."""
        assert is_columnar({"a": (1, 2)}) is False

    @pytest.mark.parametrize("data", [None, "text", 5, [{"a": 1}]])
    def test_non_dict_input_is_false(self, data: Any) -> None:
        assert is_columnar(data) is False


class TestValidateColumnarDict:
    def test_valid_dict_does_not_raise(self) -> None:
        validate_columnar_dict({"a": [1], "b": [2]})

    def test_empty_dict_does_not_raise(self) -> None:
        validate_columnar_dict({})

    def test_ragged_column_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            validate_columnar_dict({"a": [1, 2], "b": [1]})

    def test_non_list_value_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_columnar_dict({"a": 5})


class TestResultRows:
    def test_none_returns_empty_list(self) -> None:
        """``None`` (no result at all) yields no rows."""
        assert result_rows(None) == []

    def test_empty_list_returns_empty_list(self) -> None:
        """An empty partition list yields no rows."""
        assert result_rows([]) == []

    def test_empty_dict_returns_empty_list(self) -> None:
        """The schema-less ``{}`` yields no rows."""
        assert result_rows({}) == []

    def test_bare_columnar_dict_pivots_to_rows(self) -> None:
        """A bare columnar dict pivots into a list of row dicts."""
        data: dict[str, list[Any]] = {"a": [1, 2], "b": ["x", "y"]}
        assert result_rows(data) == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_schema_bearing_zero_row_dict_returns_empty_list(self) -> None:
        """A schema-bearing zero-row columnar dict yields no rows."""
        assert result_rows({"a": []}) == []

    def test_list_of_columnar_partitions_concatenates_in_order(self) -> None:
        """Columnar partitions are pivoted and concatenated in list order."""
        assert result_rows([{"a": [1]}, {"a": [2]}]) == [{"a": 1}, {"a": 2}]

    def test_single_wrapped_columnar_partition_pivots(self) -> None:
        """A dict element satisfying ``is_columnar`` is a partition, never a row; this precedence is intentional."""
        assert result_rows([{"a": [1, 2]}]) == [{"a": 1}, {"a": 2}]

    def test_row_wise_list_passes_through(self) -> None:
        """Non-columnar dict elements are rows and pass through unchanged."""
        rows: list[dict[str, Any]] = [{"a": 1}, {"a": 2}]
        assert result_rows(rows) == [{"a": 1}, {"a": 2}]

    def test_row_wise_list_returns_a_new_list_object(self) -> None:
        """The pass-through returns a NEW list object, not the input object."""
        rows: list[dict[str, Any]] = [{"a": 1}, {"a": 2}]
        result = result_rows(rows)
        assert result == rows
        assert result is not rows

    def test_mixed_partition_list_flattens_in_order(self) -> None:
        """A list element is a partition of row dicts and is extended in order."""
        assert result_rows([{"a": [1]}, [{"a": 2}, {"a": 3}]]) == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_none_elements_inside_a_list_are_skipped(self) -> None:
        """``None`` elements contribute no rows and do not raise."""
        assert result_rows([None, {"a": [1]}, None]) == [{"a": 1}]

    def test_empty_dict_elements_contribute_no_rows(self) -> None:
        """An empty dict element is a schema-less partition with zero rows."""
        assert result_rows([{}, {"a": [1]}]) == [{"a": 1}]

    def test_nested_row_list_with_non_dict_raises(self) -> None:
        """A nested row-list element must contain only dicts."""
        with pytest.raises(ValueError):
            result_rows([[{"a": 1}, 5]])

    def test_int_input_raises(self) -> None:
        with pytest.raises(ValueError):
            result_rows(5)

    def test_string_input_raises(self) -> None:
        with pytest.raises(ValueError):
            result_rows("text")

    def test_top_level_non_columnar_dict_raises(self) -> None:
        """A bare non-columnar dict at TOP LEVEL is ambiguous garbage and raises."""
        with pytest.raises(ValueError):
            result_rows({"a": 5})

    def test_non_columnar_dict_element_is_a_row(self) -> None:
        """INSIDE a list the same non-columnar dict is a row and is appended."""
        assert result_rows([{"a": 5}]) == [{"a": 5}]

    def test_scalar_list_element_raises(self) -> None:
        with pytest.raises(ValueError):
            result_rows([5])

    def test_string_list_element_raises(self) -> None:
        with pytest.raises(ValueError):
            result_rows(["text"])

    def test_ragged_columnar_looking_dict_raises(self) -> None:
        """A ragged columnar-looking dict at top level raises via the strict pivot."""
        with pytest.raises(ValueError):
            result_rows({"a": [1, 2], "b": [1]})

    def test_run_result_is_unwrapped_like_a_list(self) -> None:
        """``RunResult`` subclasses ``list`` and is unwrapped like any partition list."""
        assert result_rows(RunResult([{"a": [1]}], [])) == [{"a": 1}]

    def test_pivot_preserves_key_insertion_order(self) -> None:
        """A pivoted partition keeps the columnar dict's key insertion order."""
        result = result_rows({"b": [1], "a": [2]})
        assert result == [{"b": 1, "a": 2}]
        assert [list(row.keys()) for row in result] == [["b", "a"]]
