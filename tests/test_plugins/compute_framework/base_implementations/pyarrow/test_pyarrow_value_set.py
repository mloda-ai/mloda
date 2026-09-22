"""Unit tests for value_set, the shared helper that builds typed pyarrow value sets for is_in."""

from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_value_set import value_set


class TestPyArrowValueSet:
    def test_non_empty_values_keep_inferred_type(self) -> None:
        result = value_set(pa.array([1, 2, 3], type=pa.int32()), [1, 2])
        assert result.type == pa.int64()
        assert result.to_pylist() == [1, 2]

    def test_empty_values_take_column_type(self) -> None:
        result = value_set(pa.array([1, 2, 3], type=pa.int32()), [])
        assert result.type == pa.int32()
        assert result.to_pylist() == []

    def test_all_none_values_take_column_type(self) -> None:
        result = value_set(pa.array(["a", "b"], type=pa.string()), [None, None])
        assert result.type == pa.string()
        assert result.to_pylist() == [None, None]

    def test_dictionary_column_yields_value_type(self) -> None:
        column = pa.array(["a", "b", "a"]).dictionary_encode()
        result = value_set(column, [])
        assert result.type == pa.string()

    @pytest.mark.parametrize("values", [["a", "b"], [None, None]], ids=["values", "all_none"])
    def test_single_pass_iterable_is_consumed_once(self, values: list[Any]) -> None:
        pulled: list[Any] = []

        def generate() -> Iterator[Any]:
            for v in values:
                pulled.append(v)
                yield v

        result = value_set(pa.array(["x"], type=pa.string()), generate())
        assert pulled == values
        assert result.type == pa.string()
        assert result.to_pylist() == values
