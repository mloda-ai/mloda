"""Feature groups report rows that got no value: validation, RunResult.failed_rows and the extender view."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FailedRows, FeatureSet, report_failed_rows
from mloda.user import ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

pytestmark = pytest.mark.timeout(30)


class _ReportingBase(DataCreatorFeatureGroup):
    REPORTS: tuple[tuple[Any, str], ...] = ()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = cls.__name__.lower()
        for rows, reason in cls.REPORTS:
            report_failed_rows(name + "_col", rows, reason)
        return {name + "_col": [10, 20, 30, 40]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.__name__.lower() + "_col"}


class FrOne(_ReportingBase):
    REPORTS = (([1, 3], "timeout"),)


class FrMany(_ReportingBase):
    REPORTS = (([0], "a"), ([2], "b"), ([1, 3], "c"))


class FrOther(_ReportingBase):
    REPORTS = (([2], "quota"),)


class FrNumpy(_ReportingBase):
    REPORTS = ((np.array([1, 3]), "timeout"),)


class FrNone(_ReportingBase):
    REPORTS = ()


def _run(*groups: type[_ReportingBase], modes: set[ParallelizationMode] | None = None, **kwargs: Any) -> Any:
    return mloda.run_all(
        [g.__name__.lower() + "_col" for g in groups],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=PluginCollector.enabled_feature_groups(set(groups)),
        parallelization_modes=modes or {ParallelizationMode.SYNC},
        **kwargs,
    )


class TestReportFailedRowsValidation:
    def test_dataclass_is_frozen_with_the_four_fields(self) -> None:
        entry = FailedRows(feature_group="G", column="c", rows=(1,), reason="r")

        assert [f.name for f in dataclasses.fields(FailedRows)] == ["feature_group", "column", "rows", "reason"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.reason = "x"  # type: ignore[misc]

    def test_outside_a_run_is_a_no_op(self) -> None:
        report_failed_rows("c", [0, 1], "timeout")

    @pytest.mark.parametrize("column", [1, None, b"c"])
    def test_non_str_column_is_a_type_error(self, column: Any) -> None:
        with pytest.raises(TypeError):
            report_failed_rows(column, [0], "r")

    @pytest.mark.parametrize("rows", [["1"], [1.0], [True], [None]])
    def test_non_int_or_bool_row_is_a_type_error(self, rows: Any) -> None:
        with pytest.raises(TypeError):
            report_failed_rows("c", rows, "r")

    @pytest.mark.parametrize("rows", [np.array([0, 2]), [np.int64(0), np.int64(2)], (0, 2), range(0, 4, 2), {0, 2}])
    def test_integral_iterables_are_accepted(self, rows: Any) -> None:
        report_failed_rows("c", rows, "r")

    @pytest.mark.parametrize(
        "rows", [np.array([0, -1]), [np.int64(-1)], np.array([True]), [np.bool_(True)], "12", b"12"]
    )
    def test_numpy_negatives_bools_and_strings_are_refused(self, rows: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            report_failed_rows("c", rows, "r")

    def test_negative_row_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            report_failed_rows("c", [0, -1], "r")

    @pytest.mark.parametrize("reason", ["has space", "row 3 failed", "", "x" * 65, "bad/char", "tab\t"])
    def test_bad_reason_is_a_value_error(self, reason: str) -> None:
        with pytest.raises(ValueError):
            report_failed_rows("c", [0], reason)

    @pytest.mark.parametrize("reason", ["timeout", "a.b-c_d", "x" * 64, "R1"])
    def test_valid_reason_is_accepted(self, reason: str) -> None:
        report_failed_rows("c", [0], reason)


@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
class TestRunResultFailedRows:
    def test_report_reaches_the_result(self, mode: ParallelizationMode) -> None:
        result = _run(FrOne, modes={mode})

        assert result.failed_rows == (
            FailedRows(feature_group="FrOne", column="frone_col", rows=(1, 3), reason="timeout"),
        )
        assert isinstance(result.failed_rows, tuple)

    def test_numpy_rows_are_stored_as_plain_ints(self, mode: ParallelizationMode) -> None:
        (entry,) = _run(FrNumpy, modes={mode}).failed_rows

        assert entry.rows == (1, 3)
        assert all(type(row) is int for row in entry.rows)

    def test_no_report_is_an_empty_tuple(self, mode: ParallelizationMode) -> None:
        assert _run(FrNone, modes={mode}).failed_rows == ()

    def test_several_reports_are_kept_in_order(self, mode: ParallelizationMode) -> None:
        result = _run(FrMany, modes={mode})

        assert [(f.rows, f.reason) for f in result.failed_rows] == [((0,), "a"), ((2,), "b"), ((1, 3), "c")]

    def test_reports_of_two_feature_groups_are_both_present(self, mode: ParallelizationMode) -> None:
        result = _run(FrOne, FrOther, modes={mode})

        assert {(f.feature_group, f.rows, f.reason) for f in result.failed_rows} == {
            ("FrOne", (1, 3), "timeout"),
            ("FrOther", (2,), "quota"),
        }
        assert len(result.failed_rows) == 2


class _FailedRowsExtender(Extender):
    def __init__(self) -> None:
        self.priority = 100
        self.seen: list[Any] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.seen.append((context.feature_group_class, context.failed_rows))
        return result


class TestExtenderSeesFailedRows:
    def test_extender_sees_the_rows_recorded_during_the_call(self) -> None:
        extender = _FailedRowsExtender()

        _run(FrOne, function_extender={extender})

        ((_, rows),) = extender.seen
        assert rows == (FailedRows(feature_group="FrOne", column="frone_col", rows=(1, 3), reason="timeout"),)

    def test_extender_sees_an_empty_tuple_when_nothing_was_reported(self) -> None:
        extender = _FailedRowsExtender()

        _run(FrNone, function_extender={extender})

        ((_, rows),) = extender.seen

        assert rows == ()

    def test_failed_rows_are_reset_per_wrapped_call(self) -> None:
        extender = _FailedRowsExtender()

        _run(FrOne, FrNone, function_extender={extender})

        by_group = {name.rsplit(".", 1)[-1]: rows for name, rows in extender.seen}
        assert by_group["FrOne"] == (FailedRows("FrOne", "frone_col", (1, 3), "timeout"),)
        assert by_group["FrNone"] == ()


class TestMultiprocessingFailedRows:
    @pytest.mark.subprocess
    def test_report_crosses_the_worker_boundary(self, flight_server: Any) -> None:
        result = _run(FrOne, FrOther, modes={ParallelizationMode.MULTIPROCESSING}, flight_server=flight_server)

        assert {(f.feature_group, f.rows, f.reason) for f in result.failed_rows} == {
            ("FrOne", (1, 3), "timeout"),
            ("FrOther", (2,), "quota"),
        }
