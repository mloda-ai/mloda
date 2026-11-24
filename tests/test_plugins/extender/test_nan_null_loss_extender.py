"""Unit tests of the NaN/null loss extender on a hand-built HookContext, and of the Arrow column scan."""

from __future__ import annotations

import logging
import pickle  # nosec B403
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.function_extender import ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.function_extender.base_implementations.nan_null_loss import nan_null_loss_extender
from mloda_plugins.function_extender.base_implementations.nan_null_loss.nan_null_loss_extender import (
    NanNullLoss,
    NanNullLossError,
    NanNullLossExtender,
    nan_and_null_columns,
)

_NAN = float("nan")
_BOTH = pa.array([1.0, _NAN, None])
_RESULT = object()


class _Collecting(NanNullLossExtender):
    def __init__(self, refuse: bool = False) -> None:
        super().__init__(refuse=refuse)
        self.records: list[NanNullLoss] = []

    def report(self, loss: NanNullLoss) -> None:
        self.records.append(loss)


class _Func:
    """Stands in for the wrapped hop; counts calls and returns a sentinel."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return _RESULT


def _context(source: Any = PyArrowTable, target: Any = PandasDataFrame, step_uuid: UUID | None = None) -> HookContext:
    return HookContext(
        hook=ExtenderHook.FRAMEWORK_TRANSFORM,
        feature_group_class=None,
        feature_group_version=None,
        compute_framework_name=None if target is None else target.get_class_name(),
        source_compute_framework=source,
        target_compute_framework=target,
        step_uuid=step_uuid,
    )


_NAN_IS_NULL_TARGETS = [PandasDataFrame, SqliteFramework]


def _final_hop(table: pa.Table, target: Any = PandasDataFrame) -> tuple[Any, ...]:
    return (pa.Table, target.expected_data_framework(), table, None)


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


class TestObserving:
    @pytest.mark.parametrize("target", _NAN_IS_NULL_TARGETS)
    def test_one_record_per_offending_column_in_order_and_the_wrapped_result(self, target: Any) -> None:
        step_uuid = uuid4()
        extender = _Collecting()
        func = _Func()
        table = pa.table({"b": _BOTH, "clean": pa.array([1.0, 2.0, 3.0]), "a": _BOTH})

        with _context(target=target, step_uuid=step_uuid).activate():
            result = extender(func, *_final_hop(table, target))

        assert result is _RESULT
        assert func.calls == 1
        assert extender.records == [
            NanNullLoss(step_uuid, "b", "PyArrowTable", target.get_class_name()),
            NanNullLoss(step_uuid, "a", "PyArrowTable", target.get_class_name()),
        ]

    @pytest.mark.parametrize("target", _NAN_IS_NULL_TARGETS)
    def test_the_default_report_logs_one_warning_per_column(
        self, caplog: pytest.LogCaptureFixture, target: Any
    ) -> None:
        step_uuid = uuid4()
        logger_name = nan_null_loss_extender.__name__
        table = pa.table({"b": _BOTH, "a": _BOTH})

        with caplog.at_level(logging.WARNING, logger=logger_name):
            with _context(target=target, step_uuid=step_uuid).activate():
                NanNullLossExtender()(_Func(), *_final_hop(table, target))

        records = [r for r in caplog.records if r.name == logger_name]
        assert [r.levelno for r in records] == [logging.WARNING, logging.WARNING]
        for record, column in zip(records, ("b", "a")):
            for part in (str(step_uuid), column, "PyArrowTable", target.get_class_name()):
                assert part in record.getMessage(), record.getMessage()


class TestRefusing:
    @pytest.mark.parametrize("target", _NAN_IS_NULL_TARGETS)
    def test_raises_naming_step_frameworks_and_every_column_without_calling_the_hop(self, target: Any) -> None:
        step_uuid = uuid4()
        func = _Func()
        table = pa.table({"b": _BOTH, "a": _BOTH})

        with _context(target=target, step_uuid=step_uuid).activate():
            with pytest.raises(NanNullLossError) as excinfo:
                NanNullLossExtender(refuse=True)(func, *_final_hop(table, target))

        message = str(excinfo.value)
        for part in (str(step_uuid), "PyArrowTable", target.get_class_name(), "'b', 'a'", "hold both NaN and null"):
            assert part in message, message
        assert func.calls == 0

    def test_the_error_is_a_value_error(self) -> None:
        assert issubclass(NanNullLossError, ValueError)


class TestPassesThrough:
    @pytest.mark.parametrize("refuse", [False, True])
    @pytest.mark.parametrize(
        "case",
        ["no_context", "no_source", "no_target", "not_lossy", "not_final_hop", "clean_data"],
    )
    def test_the_hop_runs_untouched_and_nothing_is_reported(self, case: str, refuse: bool) -> None:
        extender = _Collecting(refuse=refuse)
        func = _Func()
        args = _final_hop(pa.table({"c": _BOTH}))
        context: HookContext | None = _context()
        if case == "no_context":
            context = None
        elif case == "no_source":
            context = _context(source=None)
        elif case == "no_target":
            context = _context(target=None)
        elif case == "not_lossy":
            context = _context(source=PandasDataFrame, target=PyArrowTable)
        elif case == "not_final_hop":
            args = (pa.Table, list, args[2], None)
        else:
            args = _final_hop(pa.table({"c": pa.array([1.0, _NAN, 2.0])}))

        if context is None:
            result = extender(func, *args)
        else:
            with context.activate():
                result = extender(func, *args)

        assert result is _RESULT
        assert func.calls == 1
        assert extender.records == []


class TestUncheckableFinalHop:
    def test_observing_passes_through(self) -> None:
        extender = _Collecting()
        func = _Func()

        with _context().activate():
            result = extender(func, dict, pd.DataFrame, {"c": [1.0]}, None)

        assert result is _RESULT
        assert func.calls == 1
        assert extender.records == []

    def test_refusing_raises_could_not_be_checked_naming_step_and_frameworks(self) -> None:
        step_uuid = uuid4()
        func = _Func()

        with _context(step_uuid=step_uuid).activate():
            with pytest.raises(NanNullLossError) as excinfo:
                NanNullLossExtender(refuse=True)(func, dict, pd.DataFrame, {"c": [1.0]}, None)

        message = str(excinfo.value)
        for part in ("could not be checked", str(step_uuid), "PyArrowTable", "PandasDataFrame"):
            assert part in message, message
        assert func.calls == 0

    def test_a_pair_that_is_not_lossy_is_not_refused(self) -> None:
        func = _Func()

        with _context(source=PandasDataFrame, target=PythonDictFramework).activate():
            result = NanNullLossExtender(refuse=True)(func, dict, list, {"c": [1.0]}, None)

        assert result is _RESULT


class TestFlags:
    def test_wraps_exactly_the_framework_transform_hook(self) -> None:
        assert NanNullLossExtender().wraps() == {ExtenderHook.FRAMEWORK_TRANSFORM}

    def test_refuse_sets_never_fall_back(self) -> None:
        assert NanNullLossExtender(refuse=True).never_fall_back is True

    def test_observing_sets_raise_on_error_false(self) -> None:
        assert NanNullLossExtender().raise_on_error is False

    @pytest.mark.parametrize("refuse", [False, True])
    def test_the_extender_pickles_keeping_refuse(self, refuse: bool) -> None:
        restored = pickle.loads(pickle.dumps(NanNullLossExtender(refuse=refuse)))  # nosec B301

        assert isinstance(restored, NanNullLossExtender)
        assert restored.refuse is refuse
