"""Unit tests of the NaN/null loss records and the Arrow column scan."""

from __future__ import annotations

import dataclasses
import pickle  # nosec B403
from uuid import uuid4

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.nan_null_loss import (
    NanNullLoss,
    NanNullLossError,
    nan_and_null_columns,
)

_NAN = float("nan")


class TestNanAndNullColumns:
    @pytest.mark.parametrize("arrow_type", [pa.float64(), pa.float32()])
    def test_a_float_column_with_both_values_is_listed(self, arrow_type: pa.DataType) -> None:
        table = pa.table({"c": pa.array([1.0, _NAN, None], type=arrow_type)})

        assert nan_and_null_columns(table) == ("c",)

    @pytest.mark.parametrize(
        "column",
        [
            pa.array([1.0, _NAN, 2.0]),
            pa.array([1.0, None, 2.0]),
            pa.array([None, None, None], type=pa.float64()),
            pa.array([1, None, 2], type=pa.int64()),
            pa.array(["a", None, "b"], type=pa.string()),
        ],
        ids=["only_nan", "only_null", "all_null", "int_with_null", "string_with_null"],
    )
    def test_other_columns_are_not_listed(self, column: pa.Array) -> None:
        assert nan_and_null_columns(pa.table({"c": column})) == ()

    def test_a_nested_float_field_is_not_checked(self) -> None:
        nested = pa.array([{"x": _NAN}, {"x": None}], type=pa.struct([("x", pa.float64())]))

        assert nan_and_null_columns(pa.table({"s": nested})) == ()

    def test_a_zero_row_table_lists_nothing(self) -> None:
        table = pa.table({"c": pa.array([], type=pa.float64())})

        assert nan_and_null_columns(table) == ()

    def test_a_chunked_column_is_scanned_across_chunks(self) -> None:
        chunked = pa.chunked_array([pa.array([1.0, _NAN]), pa.array([None], type=pa.float64())])

        assert nan_and_null_columns(pa.table({"c": chunked})) == ("c",)

    def test_several_offenders_come_back_in_column_order(self) -> None:
        both = pa.array([_NAN, None, 1.0])
        table = pa.table({"z": both, "clean": pa.array([1.0, 2.0, 3.0]), "a": both, "m": pa.array([1, 2, 3])})

        assert nan_and_null_columns(table) == ("z", "a")


class TestRecords:
    def test_nan_null_loss_is_a_frozen_dataclass_with_four_fields_in_order(self) -> None:
        entry = NanNullLoss(step_uuid=uuid4(), column="c", source_framework="PyArrowTable", target_framework="P")

        assert [f.name for f in dataclasses.fields(NanNullLoss)] == [
            "step_uuid",
            "column",
            "source_framework",
            "target_framework",
        ]
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.column = "x"  # type: ignore[misc]

    def test_error_is_a_value_error_that_pickles_with_its_message(self) -> None:
        restored = pickle.loads(pickle.dumps(NanNullLossError("step X column c")))  # nosec B301

        assert isinstance(restored, ValueError)
        assert type(restored) is NanNullLossError
        assert str(restored) == "step X column c"
