"""Tests wiring ExtenderHook.INPUT_DATA_LOAD into BaseInputData.load().

Covers ComputeFramework.current(), HookContext population (inherited identity fields plus
data_access_identity/format/dataset_version), the no-extender baseline, deny-before-load /
deny-with-fallback, and the "activate only when needed" short-circuit.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.user import DataAccessCollection, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from tests.test_plugins.feature_group.input_data.test_classes.test_input_classes import DBInputDataTestFeatureGroup

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
    """No calculate extender: the calculate-phase HookContext is still built and activated so INPUT_DATA_LOAD can read from it."""

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
    """Baseline guard: a CSV load with no extenders registered is unaffected."""

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
    """A raise_on_error=True (default) INPUT_DATA_LOAD extender that raises instead of delegating prevents the load."""

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
    """A raise_on_error=False INPUT_DATA_LOAD extender that raises still lets the load succeed, with a warning logged."""

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
    """ComputeFramework.current() stays None unless a calculate or fetch extender is registered."""

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


def _build_calc_context(compute_framework_name: str = "stub") -> HookContext:
    """A minimal calculate-phase HookContext to activate() around a direct _load_data_via_hook call."""
    return HookContext(
        hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
        feature_group_class="test.Fake",
        feature_group_version="1",
        plugin_version=None,
        feature_names=("x",),
        input_features=None,
        compute_framework_name=compute_framework_name,
    )


class _DirectLoadReader(BaseInputData):
    """Minimal reader whose load_data returns a fixed 3-element list, for direct
    _load_data_via_hook calls that bypass matching/init_reader entirely."""

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return [1, 2, 3]


class TestDataAccessIdentityHidesDictCredentialValues:
    """Fix: a dict-shaped data_access (real ReadDB credentials) must expose only key
    names in data_access_identity, never values, since DB credentials pass through
    this exact dict at this exact point (mloda_plugins/feature_group/input_data/read_db.py)."""

    def test_dict_credential_values_are_not_leaked_into_identity(self, tmp_path: Path) -> None:
        db_path = tmp_path / "creds.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE creds_table (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute('INSERT INTO creds_table (name) VALUES ("Alice")')
        conn.commit()
        conn.close()

        fetch_extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            ["name"],
            compute_frameworks={PyArrowTable},
            data_access_collection=DataAccessCollection(
                credentials=[{SQLITEReader.db_path(): str(db_path), "user": "alice", "password": "hunter2"}]  # nosec B105
            ),
            plugin_collector=PluginCollector.enabled_feature_groups({DBInputDataTestFeatureGroup}),
            function_extender={fetch_extender},
        )

        fetch_context = fetch_extender.captured
        assert fetch_context is not None
        identity = fetch_context.data_access_identity
        assert identity is not None
        assert "hunter2" not in identity
        assert "alice" not in identity
        assert "user" in identity
        assert "password" in identity


class TestDataAccessIdentityHidesUriEmbeddedPassword:
    """A postgresql://user:pass@host/db-style data_access string must not leak its password
    segment. No built-in reader accepts a raw credentialed URI as data_access, so this pins
    the contract directly against BaseInputData._load_data_via_hook."""

    def test_uri_password_segment_is_not_in_identity(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = ComputeFramework(function_extender={extender})
        reader = _DirectLoadReader()
        features = FeatureSet()
        data_access = "postgresql://admin:s3cr3t@host:5432/db"

        with cfw.activate(), _build_calc_context().activate():
            BaseInputData._load_data_via_hook(reader, data_access, features)

        assert extender.captured is not None
        identity = extender.captured.data_access_identity
        assert identity is not None
        assert "s3cr3t" not in identity


class TestDataAccessIdentityBaselineForNonCredentialShapedValues:
    """Regression guard: an ordinary (non-credential-shaped) data_access, like a CSV file path,
    must still produce a non-empty, useful identity string."""

    def test_csv_file_path_identity_is_non_empty_and_useful(self, tmp_path: Path) -> None:
        column = f"{_MARKER}_col_g"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [1, 2])

        fetch_extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            [column],
            compute_frameworks={PythonDictFramework},
            data_access_collection=DataAccessCollection(files={str(path)}),
            function_extender={fetch_extender},
        )

        fetch_context = fetch_extender.captured
        assert fetch_context is not None
        identity = fetch_context.data_access_identity
        assert identity
        assert str(path) in identity


_ROW_COUNT_SENTINEL = 424242


class _SentinelRowCountComputeFramework(ComputeFramework):
    """_row_count returns a fixed sentinel, unrelated to len() of any real result; stands in
    for a lazy/SQL-backed framework's deliberately non-materializing row counter."""

    def _row_count(self, data: Any) -> int | None:
        return _ROW_COUNT_SENTINEL


class TestInputDataLoadHookUsesFrameworkRowCountNotDefaultLen:
    """Fix: rows_out on the INPUT_DATA_LOAD hook must reuse cfw._row_count, the same
    framework-aware counter the calculate hook uses, not the generic len()-based default."""

    def test_rows_out_reflects_cfw_row_count_not_len(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = _SentinelRowCountComputeFramework(function_extender={extender})
        reader = _DirectLoadReader()
        features = FeatureSet()

        with cfw.activate(), _build_calc_context().activate():
            result = BaseInputData._load_data_via_hook(reader, {"any": "value"}, features)

        assert result == [1, 2, 3]
        assert extender.captured is not None
        assert extender.captured.rows_out == _ROW_COUNT_SENTINEL
        assert extender.captured.rows_out != len(result)
