"""Failing tests for int-051 Phase 4: wiring ExtenderHook.INPUT_DATA_LOAD into BaseInputData.load().

Covers ComputeFramework.current() (new contextvar, activated only when a calculate or fetch
extender is registered), HookContext population (data_access_identity/format/dataset_version,
run_id/carrier/worker_index/compute_framework_name/feature_group_class inherited from the active
calculate-phase context), the no-extender baseline, deny-before-load / deny-with-fallback, and a
short-circuit guard mirroring Phase 2/3's "activate only when needed" tests.
"""

import logging
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader

_MARKER = "inputload051"
_EXPECTED_FEATURE_GROUP_CLASS = f"{ReadFileFeature.__module__}.{ReadFileFeature.__qualname__}"


class _CalcContextCapturingExtender(Extender):
    """Calls func like a real extender, then reads HookContext.current() afterward."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: HookContext | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.captured = HookContext.current()
        return result


class _InputDataLoadCapturingExtender(Extender):
    """Calls func like a real extender, then reads HookContext.current() afterward."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority
        self.captured: HookContext | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        self.captured = HookContext.current()
        return result


class _InputDataLoadVetoExtender(Extender):
    """raise_on_error selects deny-before-load (True, default) vs deny-with-fallback (False)."""

    def __init__(self, raise_on_error: bool = True) -> None:
        self.priority = 100
        self.raise_on_error = raise_on_error
        self.name = "input_data_load_veto"

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("denied input data load")


class _NoOpValidateInputFeatureExtender(Extender):
    """A harmless VALIDATE_INPUT_FEATURE extender: unrelated to calculate/fetch."""

    def __init__(self, priority: int = 100) -> None:
        self.priority = priority

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_INPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _write_csv(path: Path, column: str, values: list[int]) -> None:
    lines = "\n".join(str(v) for v in values)
    path.write_text(f"{column}\n{lines}\n", encoding="utf-8")


class TestInputDataLoadHookFiresAlongsideCalculateExtender:
    def test_captured_context_matches_the_calculate_hook_context(self, tmp_path: Path) -> None:
        column = f"{_MARKER}_col_a"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [1, 2, 3])

        calc_extender = _CalcContextCapturingExtender()
        fetch_extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            [column],
            compute_frameworks={PythonDictFramework},
            data_access_collection=DataAccessCollection(files={str(path)}),
            function_extender={calc_extender, fetch_extender},
        )

        calc_context = calc_extender.captured
        fetch_context = fetch_extender.captured
        assert calc_context is not None
        assert fetch_context is not None

        assert fetch_context.hook == ExtenderHook.INPUT_DATA_LOAD
        assert fetch_context.data_access_identity
        assert fetch_context.data_access_format
        assert fetch_context.data_access_dataset_version is None

        assert fetch_context.run_id == calc_context.run_id
        assert fetch_context.carrier == calc_context.carrier
        assert fetch_context.worker_index == calc_context.worker_index
        assert fetch_context.compute_framework_name == calc_context.compute_framework_name
        assert fetch_context.feature_group_class == calc_context.feature_group_class


class TestInputDataLoadHookFiresWithOnlyFetchExtenderRegistered:
    """No calculate extender at all: the calculate-phase HookContext must still be built and
    activated (because a fetch extender exists), so INPUT_DATA_LOAD can read from it."""

    def test_fetch_context_still_carries_calculate_phase_identity_fields(self, tmp_path: Path) -> None:
        column = f"{_MARKER}_col_b"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [4, 5, 6])

        fetch_extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            [column],
            compute_frameworks={PythonDictFramework},
            data_access_collection=DataAccessCollection(files={str(path)}),
            function_extender={fetch_extender},
        )

        fetch_context = fetch_extender.captured
        assert fetch_context is not None
        assert fetch_context.hook == ExtenderHook.INPUT_DATA_LOAD
        assert fetch_context.data_access_identity
        assert fetch_context.data_access_format
        assert fetch_context.data_access_dataset_version is None
        assert fetch_context.compute_framework_name == "PythonDictFramework"
        assert fetch_context.worker_index is None
        assert fetch_context.carrier is None
        assert fetch_context.feature_group_class == _EXPECTED_FEATURE_GROUP_CLASS


class TestNoExtenderRegisteredBaselineRegressionGuard:
    """BASELINE (already passes today): a CSV load with no extenders registered is unaffected.

    Guards the "activate only when needed" short-circuit once INPUT_DATA_LOAD is wired in.
    """

    def test_run_all_reads_expected_values(self, tmp_path: Path) -> None:
        column = f"{_MARKER}_col_c"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [7, 8, 9])

        result = mloda.run_all(
            [column],
            compute_frameworks={PythonDictFramework},
            data_access_collection=DataAccessCollection(files={str(path)}),
        )

        assert result[0][column] == [7, 8, 9]


class TestDenyBeforeLoad:
    """A raise_on_error=True (default) INPUT_DATA_LOAD extender that raises instead of delegating
    prevents the load, propagating the error through run_all."""

    def test_veto_raises_and_propagates(self, tmp_path: Path) -> None:
        column = f"{_MARKER}_col_d"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [1, 2])
        extender = _InputDataLoadVetoExtender()

        with pytest.raises(RuntimeError, match="denied input data load"):
            mloda.run_all(
                [column],
                compute_frameworks={PythonDictFramework},
                data_access_collection=DataAccessCollection(files={str(path)}),
                function_extender={extender},
            )


class TestDenyWithFallback:
    """A raise_on_error=False INPUT_DATA_LOAD extender that raises still lets the load succeed
    (falls back), with a warning logged, mirroring _invoke_extender's warning-only semantics."""

    def test_warning_only_veto_logs_and_falls_back(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        column = f"{_MARKER}_col_e"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [1, 2])
        extender = _InputDataLoadVetoExtender(raise_on_error=False)

        with caplog.at_level(logging.WARNING):
            result = mloda.run_all(
                [column],
                compute_frameworks={PythonDictFramework},
                data_access_collection=DataAccessCollection(files={str(path)}),
                function_extender={extender},
            )

        assert result[0][column] == [1, 2]
        assert any(
            record.levelno == logging.WARNING and "denied input data load" in record.message
            for record in caplog.records
        )


class TestComputeFrameworkCurrentShortCircuit:
    """ComputeFramework.current() must stay None unless a calculate or fetch extender is registered:
    an unrelated hook's extender (VALIDATE_INPUT_FEATURE here) must not activate it."""

    def test_current_is_none_when_only_an_unrelated_hook_extender_is_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        column = f"{_MARKER}_col_f"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [1, 2])

        observed: list[Any] = []
        original_load_data = CsvReader.__dict__["load_data"].__func__

        def _probe_load_data(cls: type, data_access: Any, features: Any) -> Any:
            observed.append(ComputeFramework.current())
            return original_load_data(cls, data_access, features)

        monkeypatch.setattr(CsvReader, "load_data", classmethod(_probe_load_data))

        mloda.run_all(
            [column],
            compute_frameworks={PythonDictFramework},
            data_access_collection=DataAccessCollection(files={str(path)}),
            function_extender={_NoOpValidateInputFeatureExtender()},
        )

        assert observed
        assert observed[0] is None
