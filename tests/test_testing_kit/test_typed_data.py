"""Tests for TypedTestData construction and accessors."""

import contextlib
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda._testing.typed_data import TypedTestData


def _data() -> TypedTestData:
    return TypedTestData({"a": (DataType.INT64, [1, 2, None]), "b": (DataType.STRING, ("x", "y", "z"))})


def test_accepts_fitting_data_with_types_and_row_count() -> None:
    data = _data()
    assert data.to_python_dict() == {"a": [1, 2, None], "b": ["x", "y", "z"]}
    assert data.types == {"a": DataType.INT64, "b": DataType.STRING}
    assert data.row_count == 3


def test_lists_every_misfit() -> None:
    with pytest.raises(ValueError) as info:
        TypedTestData({"a": (DataType.INT64, ["x"]), "b": (DataType.INT32, [2**31])})
    message = str(info.value)
    for part in ("'a'", "'b'", "INT64", "INT32"):
        assert part in message


def test_refuses_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        TypedTestData({"a": (DataType.INT64, [1, 2]), "b": (DataType.INT64, [1])})


def test_refuses_str_as_values() -> None:
    bad: Any = (DataType.STRING, "abc")
    with pytest.raises(ValueError):
        TypedTestData({"a": bad})


def test_to_python_dict_returns_copy() -> None:
    data = _data()
    first = data.to_python_dict()
    first["a"].append(99)
    first["new"] = []
    assert data.to_python_dict() == {"a": [1, 2, None], "b": ["x", "y", "z"]}
    assert data.row_count == 3


def test_types_is_a_fresh_read_only_view() -> None:
    data = _data()
    with contextlib.suppress(TypeError):
        data.types["a"] = DataType.STRING
    assert data.types["a"] is DataType.INT64


def test_refuses_non_datatype_column_type() -> None:
    bad: Any = ("INT64", [1])
    with pytest.raises(ValueError, match="'a'"):
        TypedTestData({"a": bad})
