"""TypedTestData: validated pure Python columns with declared DataTypes."""

from __future__ import annotations

from typing import Any

import pytest

from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType


def _data() -> TypedTestData:
    return TypedTestData({"a": (DataType.INT8, [1, None]), "b": (DataType.STRING, ("x", "y"))})


def test_accepts_fitting_table_and_round_trips() -> None:
    data = _data()

    assert data.row_count == 2
    assert data.types == {"a": DataType.INT8, "b": DataType.STRING}
    assert data.to_python_dict() == {"a": [1, None], "b": ["x", "y"]}


def test_to_python_dict_and_types_return_fresh_copies() -> None:
    data = _data()

    data.to_python_dict()["a"].append(99)
    data.types["a"] = DataType.STRING

    assert data.to_python_dict()["a"] == [1, None]
    assert data.types["a"] is DataType.INT8


def test_rejects_misfit_naming_column_row_and_type() -> None:
    with pytest.raises(ValueError) as excinfo:
        TypedTestData({"a": (DataType.INT8, [1, 300])})

    message = str(excinfo.value)
    assert "column 'a' row 1" in message and "300" in message and "INT8" in message
    assert "_testing" not in message


def test_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="unequal lengths"):
        TypedTestData({"a": (DataType.INT8, [1, 2]), "b": (DataType.INT8, [1])})


def test_rejects_non_data_type() -> None:
    with pytest.raises(ValueError, match="is not a DataType"):
        TypedTestData({"a": ("INT8", [1])})  # type: ignore[dict-item]


@pytest.mark.parametrize("values", [{1, 2}, "ab"])
def test_rejects_non_sequence_values(values: Any) -> None:
    with pytest.raises(ValueError, match="list or tuple"):
        TypedTestData({"a": (DataType.INT8, values)})
