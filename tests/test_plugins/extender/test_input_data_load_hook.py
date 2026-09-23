"""Tests wiring ExtenderHook.INPUT_DATA_LOAD into BaseInputData.load().

Covers ComputeFramework.current(), HookContext population (inherited identity fields plus
data_access_identity/format/dataset_version), the no-extender baseline, deny-before-load /
deny-with-fallback, and the "activate only when needed" short-circuit.
"""

import logging
import sqlite3
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.user import DataAccessCollection, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda_plugins.feature_group.input_data.read_files.text_file_reader import TextFileReader
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


class _InputDataLoadTamperingExtender(Extender):
    """Calls func for the real loaded data, then returns DIFFERENT (but shape-valid) data instead of it."""

    def __init__(self, column: str, raise_on_error: bool = True) -> None:
        self.priority = 100
        self.raise_on_error = raise_on_error
        self.name = "input_data_load_tamper"
        self._column = column

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        func(*args, **kwargs)
        return [{self._column: 999}, {self._column: 999}]


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
        assert fetch_context.tenant_id == calc_context.tenant_id
        assert fetch_context.project_id == calc_context.project_id
        assert fetch_context.principal == calc_context.principal
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


class TestExtenderCannotSubstituteTheLoadedData:
    """An extender that calls func for the real load, then returns different data, must not win."""

    def test_tampered_data_is_discarded_in_favor_of_the_real_load(self, tmp_path: Path) -> None:
        column = f"{_MARKER}_col_h"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [1, 2])
        extender = _InputDataLoadTamperingExtender(column)

        result = mloda.run_all(
            [column],
            compute_frameworks={PythonDictFramework},
            data_access_collection=DataAccessCollection(files={str(path)}),
            function_extender={extender},
        )

        assert result[0][column] == [1, 2], "The real loaded data must be used, not the tampered one"


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


def _identity_of(data_access: Any) -> str:
    extender = _InputDataLoadCapturingExtender()
    cfw = ComputeFramework(function_extender={extender})
    reader = _DirectLoadReader()
    features = FeatureSet()

    with cfw.activate(), _build_calc_context().activate():
        BaseInputData._load_data_via_hook(reader, data_access, features)

    assert extender.captured is not None
    identity = extender.captured.data_access_identity
    assert identity is not None
    return identity


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


class TestDataAccessIdentityOfUriStrings:
    """A fullmatching scheme://... drops user info, query and fragment. A jdbc: scheme drops the path entirely;
    any other scheme cuts its path back to the last / before the first percent-escape, and fails closed when a
    path segment has a : followed by =. An azure container userinfo is kept only when it fullmatches an azure
    container name or a well-known $root/$web/$logs alias."""

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            pytest.param(
                "postgresql://admin:s3cr3t@host:5432/db",
                "postgresql://host:5432/db",
                id="userinfo-stripped",
            ),
            pytest.param(
                "abfss://key:s3cr3t@account.dfs.core.windows.net/p",
                "abfss://account.dfs.core.windows.net/p",
                id="abfss-userinfo-with-colon-is-stripped",
            ),
            pytest.param("postgresql://u:p@ss@host/db", "postgresql://host/db", id="at-sign-in-userinfo"),
            pytest.param("postgresql://u:p@host", "postgresql://host", id="no-path"),
            pytest.param(
                "abfss://container@account.dfs.core.windows.net/p?sig=S",
                "abfss://container@account.dfs.core.windows.net/p",
                id="abfss-keeps-container-drops-query",
            ),
            pytest.param(
                "abfs://container@account.dfs.core.windows.net/p",
                "abfs://container@account.dfs.core.windows.net/p",
                id="abfs-keeps-container",
            ),
            pytest.param(
                "wasb://container@account.blob.core.windows.net/p",
                "wasb://container@account.blob.core.windows.net/p",
                id="wasb-keeps-container",
            ),
            pytest.param(
                "wasbs://container@account.blob.core.windows.net/p",
                "wasbs://container@account.blob.core.windows.net/p",
                id="wasbs-keeps-container",
            ),
            pytest.param(
                "ABFSS://container@account.dfs.core.windows.net/p",
                "ABFSS://container@account.dfs.core.windows.net/p",
                id="upper-case-abfss-keeps-container",
            ),
            pytest.param(
                "WASB://container@account.blob.core.windows.net/p",
                "WASB://container@account.blob.core.windows.net/p",
                id="upper-case-wasb-keeps-container",
            ),
            pytest.param(
                "wasbs://key:secret@account.blob.core.windows.net/p",
                "wasbs://account.blob.core.windows.net/p",
                id="wasbs-userinfo-with-colon-is-stripped",
            ),
            pytest.param(
                "wasbs://container@account.blob.core.windows.net/p?sig=S",
                "wasbs://container@account.blob.core.windows.net/p",
                id="wasbs-keeps-container-drops-query",
            ),
            pytest.param("postgresql://token@host/db", "postgresql://host/db", id="non-azure-colon-free-userinfo"),
            pytest.param(
                "postgresql://host/db%3Fpassword=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-question-mark-secret",
            ),
            pytest.param(
                "postgresql://host/db%3fpassword=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-lower-case-question-mark-secret",
            ),
            pytest.param(
                "postgresql://host/db%23password=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-hash-secret",
            ),
            pytest.param(
                "postgresql+psycopg2://u:p@host/db",
                "postgresql+psycopg2://host/db",
                id="scheme-with-plus-suffix",
            ),
            pytest.param(
                "https://host/a%3Fb.csv",
                "https://host/",
                id="path-cut-back-to-segment-before-percent-encoded-question-mark",
            ),
            pytest.param(
                "postgresql://host/db%3Flimit=10",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-question-mark-with-non-secret-key",
            ),
            pytest.param("file:///tmp/a.csv", "file:///tmp/a.csv", id="file-uri-with-empty-host"),
            pytest.param("sqlite:///x.db", "sqlite:///x.db", id="sqlite-uri-with-empty-host"),
            pytest.param(
                "s3://bucket/year=2024/part.parquet",
                "s3://bucket/year=2024/part.parquet",
                id="s3-uri-with-equals-in-path",
            ),
            pytest.param("http://[::1]/x", "http://[::1]/x", id="ipv6-host-without-at-sign"),
            pytest.param(
                "jdbc:postgresql://u:p@host:5432/db?ssl=true",
                "jdbc:postgresql://host:5432",
                id="jdbc-scheme-drops-path-entirely",
            ),
            pytest.param(
                "jdbc:db2://host:50000/DB:password=hunter2",
                "jdbc:db2://host:50000",
                id="jdbc-db2-drops-path-with-secret",
            ),
            pytest.param(
                "jdbc:teradata://host/PASSWORD=hunter2",
                "jdbc:teradata://host",
                id="jdbc-teradata-drops-path-with-secret",
            ),
            pytest.param(
                "jdbc:informix-sqli://h:1533/db:password=hunter2",
                "jdbc:informix-sqli://h:1533",
                id="jdbc-informix-drops-path-with-secret",
            ),
            pytest.param(
                "postgresql://host/db%3Bpassword=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-semicolon",
            ),
            pytest.param(
                "postgresql://host/db%26password=hunter2",
                "postgresql://host/",
                id="path-cut-back-to-segment-before-percent-encoded-ampersand",
            ),
            pytest.param(
                "https://host/data%253Fpassword%253Dhunter2",
                "https://host/",
                id="path-cut-back-to-segment-before-double-percent-encoded-escape",
            ),
            pytest.param(
                "s3://bucket/my%20file.csv",
                "s3://bucket/",
                id="path-cut-back-to-segment-before-percent-encoded-space",
            ),
            pytest.param(
                "s3://bucket/2024%2F01/a.parquet",
                "s3://bucket/",
                id="path-cut-back-to-segment-before-percent-encoded-slash",
            ),
            pytest.param(
                "s3://bucket/dir/my%20file.csv",
                "s3://bucket/dir/",
                id="path-cut-back-to-parent-dir-before-percent-escape",
            ),
            pytest.param(
                "https://host/a/b%20c/d.csv",
                "https://host/a/",
                id="path-cut-back-to-segment-before-mid-path-percent-escape",
            ),
            pytest.param("file:///C:/data/x.csv", "file:///C:/data/x.csv", id="windows-drive-colon-in-file-uri-kept"),
            pytest.param(
                "s3://bucket/2024-01-01T00:00:00/part.parquet",
                "s3://bucket/2024-01-01T00:00:00/part.parquet",
                id="timestamp-colons-in-path-segment-kept",
            ),
            pytest.param(
                "s3://bucket/ts=2024-01-01T00:00:00/part.parquet",
                "s3://bucket/ts=2024-01-01T00:00:00/part.parquet",
                id="equals-before-colon-hive-partition-kept",
            ),
            pytest.param(
                "abfss://$web@account.dfs.core.windows.net/p",
                "abfss://$web@account.dfs.core.windows.net/p",
                id="well-known-dollar-container-alias-is-kept",
            ),
            pytest.param(
                "abfss://Container_X@account.dfs.core.windows.net/p",
                "abfss://account.dfs.core.windows.net/p",
                id="upper-case-and-underscore-container-is-dropped",
            ),
            pytest.param(
                "abfss://ab@account.dfs.core.windows.net/p",
                "abfss://account.dfs.core.windows.net/p",
                id="too-short-container-is-dropped",
            ),
        ],
    )
    def test_identity_keeps_host_and_path_only(self, uri: str, expected: str) -> None:
        assert _identity_of(uri) == expected

    def test_long_uri_without_secrets_returns_quickly(self) -> None:
        value = "s3://b/" + "a" * 40000
        assert _identity_of(value) == value

    def test_long_uri_with_a_trailing_invalid_character_returns_quickly(self) -> None:
        value = "a://" + "a@" * 20000 + "/" + "a" * 20000 + "!"
        assert _identity_of(value) == "str"


class TestDataAccessIdentityDefaultDenyForSchemeLessAndPathValues:
    """None of these values is a Mapping, a fullmatching URI, or an existing local path, so each resolves to
    its type name only."""

    @pytest.mark.parametrize(
        ("value", "secret"),
        [
            pytest.param("https://host/p?email=a@b.com/x&sig=S", None, id="at-sign-in-query-fails-uri-shape"),
            pytest.param(
                "postgresql://host/db?user=u&password=p@ss/word",
                None,
                id="secret-with-at-and-slash-in-query-fails-uri-shape",
            ),
            pytest.param("https://host:8080/a@b/c", None, id="at-sign-in-path-fails-uri-shape"),
            pytest.param("https://u:p@host/a@b/c", None, id="at-sign-in-path-with-userinfo-fails-uri-shape"),
            pytest.param("https://host:/a@b/c", None, id="empty-port-at-sign-in-path-fails-uri-shape"),
            pytest.param("postgresql://u:pa]/ss@host/db", None, id="bracket-in-userinfo-fails-uri-shape"),
            pytest.param("https://host/p#a@b", None, id="at-sign-in-fragment-fails-uri-shape"),
            pytest.param("postgresql://u:pa/ss@host/db", None, id="slash-in-userinfo-fails-uri-shape"),
            pytest.param("postgresql://u:pa?ss@host/db", None, id="question-mark-in-userinfo-fails-uri-shape"),
            pytest.param("postgresql://u:pa#ss@host/db", None, id="hash-in-userinfo-fails-uri-shape"),
            pytest.param("http://[::1]/x@y", None, id="ipv6-host-at-sign-in-path-fails-uri-shape"),
            pytest.param("https://host/p;user=alice/x", None, id="uri-with-plain-connection-key-fails-uri-shape"),
            pytest.param(
                "s3://bucket/a;db=main/part.parquet", None, id="s3-uri-with-plain-connection-key-fails-uri-shape"
            ),
            pytest.param(
                "s3://bucket/key with space password=hunter2", "hunter2", id="space-in-uri-path-fails-uri-shape"
            ),
            pytest.param("postgresql://host/db\npassword=hunter2", "hunter2", id="newline-in-uri-path-fails-uri-shape"),
            pytest.param("https://host/db&password=hunter2", "hunter2", id="ampersand-in-uri-path-fails-uri-shape"),
            pytest.param("s3://bucket/path;password=hunter2", "hunter2", id="semicolon-in-uri-path-fails-uri-shape"),
            pytest.param(
                "jdbc:sqlserver://host:1433;databaseName=db;user=a;password=hunter2",
                "hunter2",
                id="jdbc-sqlserver-semicolons-fail-uri-shape",
            ),
            pytest.param(
                "jdbc:db2://host:50000/DB:user=u;password=hunter2;", "hunter2", id="jdbc-db2-semicolon-terminated"
            ),
            pytest.param(
                "JDBC:db2://host:50000/DB:password=hunter2", "hunter2", id="upper-case-jdbc-prefix-fails-uri-shape"
            ),
            pytest.param(
                "mongodb://host/db:password=hunter2", "hunter2", id="non-jdbc-colon-then-equals-in-path-segment"
            ),
            pytest.param(
                "https://host/a/db:password=hunter2/x.csv",
                "hunter2",
                id="non-jdbc-colon-then-equals-in-mid-path-segment",
            ),
            pytest.param("s3://bucket/a:b=c/part.parquet", None, id="s3-colon-then-equals-in-path-segment"),
            pytest.param("alice:hunter2://host/db", "hunter2", id="colon-before-scheme-separator-is-not-a-scheme"),
            pytest.param("mongodb://u:p@h1:27017,h2:27017/db", None, id="comma-separated-host-list-fails-uri-shape"),
            pytest.param("postgresql://u:12/ss@host/db", "ss", id="colon-digit-slash-in-userinfo-fails-uri-shape"),
            pytest.param(
                "postgresql://u:12?ss@host/db", "ss", id="colon-digit-question-mark-in-userinfo-fails-uri-shape"
            ),
            pytest.param("https://tok/en@host/x", "en", id="slash-in-userinfo-before-at-sign-fails-uri-shape"),
            pytest.param("host=localhost user=alice password=hunter2", "hunter2", id="libpq-keywords"),
            pytest.param("DRIVER={ODBC};UID=alice;PWD=hunter2", "hunter2", id="odbc-keywords"),
            pytest.param("user:hunter2@host/db", "hunter2", id="userinfo-scheme-less"),
            pytest.param("user:pw@host/db?token=hunter2", "hunter2", id="scheme-less-userinfo-query-secret"),
            pytest.param("user:pw@host/db#token=hunter2", "hunter2", id="scheme-less-userinfo-fragment-secret"),
            pytest.param("host=localhost password='hunter 2'", "hunter 2", id="libpq-quoted-value"),
            pytest.param(
                "DRIVER={ODBC Driver 17};UID=alice;PWD={hun;ter2}", "hun;ter2", id="odbc-braces-with-separator"
            ),
            pytest.param("Host=a HOST=b", "=b", id="mixed-case-duplicate-keys"),
            pytest.param("password=hunter2", "hunter2", id="single-known-key"),
            pytest.param(
                "DRIVER={x};Server=https://host;PWD=secret",
                "secret",
                id="keyword-string-containing-scheme-separator",
            ),
            pytest.param("user:p@ss@host/db", "p@ss", id="at-sign-in-userinfo-password"),
            pytest.param("user:p@ss/word@host/db", "ss/word", id="at-sign-and-slash-in-password"),
            pytest.param("dbname=x user=y", "=y", id="dbname-and-user"),
            pytest.param("password = hunter2 host = localhost", "hunter2", id="whitespace-around-equals"),
            pytest.param("password= hunter2 host =localhost", "hunter2", id="uneven-whitespace-around-equals"),
            pytest.param("dbname = mydb password = hunter2 port = 5432", "hunter2", id="spaced-dbname-password-port"),
            pytest.param("PWD={x; secret1 host=y};UID=a", "secret1", id="odbc-brace-value-with-fake-key"),
            pytest.param("user = alice password = hunter2", "hunter2", id="spaced-user-password"),
            pytest.param(
                "host=h password=correct horse=battery", "horse", id="unrecognized-key-after-secret-not-printed"
            ),
            pytest.param("host=h password='my pass=word'", "my pass=word", id="quoted-value-with-key-fragment"),
            pytest.param("pass=hunter2", "hunter2", id="secret-key-pass"),
            pytest.param("sslpassword=hunter2", "hunter2", id="secret-key-sslpassword"),
            pytest.param("key=abc", "abc", id="secret-key-key"),
            pytest.param("access_key=abc secret_key=def", "abc", id="secret-keys-access-secret"),
            pytest.param("aws_secret_access_key=hunter2", "hunter2", id="secret-key-aws-secret-access"),
            pytest.param("client_secret=x client_id=abc", "abc", id="client-id-not-printed"),
            pytest.param("account_key=hunter2", "hunter2", id="secret-key-account-key"),
            pytest.param("auth_token=hunter2", "hunter2", id="secret-key-auth-token"),
            pytest.param("credentials=hunter2", "hunter2", id="secret-key-credentials"),
            pytest.param("sas=hunter2", "hunter2", id="secret-key-sas"),
            pytest.param("service-account-key=hunter2", "hunter2", id="secret-key-with-dashes"),
            pytest.param("host=/var/run/postgresql user=alice password=hunter2", "hunter2", id="host-value-is-a-path"),
            pytest.param("u:p/w@host/db", "p/w", id="slash-in-userinfo-password"),
            pytest.param("u:p w@host/db", "p w", id="space-in-userinfo-password"),
            pytest.param("u:p\\w@host/db", "p\\w", id="backslash-in-userinfo-password"),
            pytest.param(":hunter2@host/db", "hunter2", id="empty-user-userinfo"),
            pytest.param(Path("host=localhost password=hunter2"), "hunter2", id="path-object-keywords"),
            pytest.param(Path("user:hunter2@host/db"), "hunter2", id="path-object-userinfo"),
            pytest.param(PurePosixPath("s3://alice:hunter2@bucket/key"), "hunter2", id="pure-posix-path-uri"),
            pytest.param("user:pw@host/db%3Ftoken=hunter2", "hunter2", id="userinfo-percent-encoded-question-mark"),
            pytest.param(
                "host.com/db%3Fx password=hunter2",
                "hunter2",
                id="percent-encoded-question-mark-then-keyword-scan-guard",
            ),
            pytest.param("notes:2024@work.txt", None, id="userinfo-lookalike"),
            pytest.param("user=alice.csv", None, id="file-name-starting-with-connection-key"),
            pytest.param(
                "host.com/db?config=password:hunter2", "hunter2", id="secret-value-under-unrecognized-key-now-hidden"
            ),
            pytest.param(
                "host.com/db%253Fpassword=hunter2", "hunter2", id="double-percent-encoded-question-mark-now-hidden"
            ),
            pytest.param("password%3Dhunter2", "hunter2", id="percent-encoded-pair-without-anchor-now-hidden"),
            pytest.param(
                "host.com/db%3Bpassword=hunter2", "hunter2", id="percent-encoded-semicolon-without-anchor-now-hidden"
            ),
            pytest.param(
                "host.com/db?user=alice&password=hunter2", "hunter2", id="scheme-less-query-secret-drops-whole-query"
            ),
            pytest.param("localhost:5432/db?password=hunter2", "hunter2", id="scheme-less-query-secret-with-port"),
            pytest.param("host.com/db#password=hunter2", "hunter2", id="scheme-less-fragment-secret"),
            pytest.param(
                "host.com/db?  password=hunter2",
                "hunter2",
                id="scheme-less-query-secret-with-whitespace-after-delimiter",
            ),
            pytest.param("host.com/db%3Fpassword=hunter2", "hunter2", id="percent-encoded-question-mark-secret"),
            pytest.param(
                "host.com/db%3fpassword=hunter2", "hunter2", id="percent-encoded-lower-case-question-mark-secret"
            ),
            pytest.param("host.com/db?password%3Dhunter2", "hunter2", id="percent-encoded-equals-in-query-secret"),
            pytest.param("host.com/db%23password=hunter2", "hunter2", id="percent-encoded-hash-secret"),
            pytest.param(
                "host.com/db?a=1%26password=hunter2", "hunter2", id="percent-encoded-ampersand-in-query-secret"
            ),
            pytest.param("host.com/db?%20password=hunter2", "hunter2", id="percent-encoded-leading-space-secret"),
            pytest.param("host.com/db?pass%77ord=hunter2", "hunter2", id="percent-encoded-key-letter-secret"),
            pytest.param(
                "host.com/db?x password=hunter2", "hunter2", id="scheme-less-query-secret-after-space-separated-param"
            ),
            pytest.param(
                "host.com/db#x password=hunter2",
                "hunter2",
                id="scheme-less-fragment-secret-after-space-separated-param",
            ),
            pytest.param(
                "host.com/db?a=1 token=hunter2", "hunter2", id="scheme-less-query-secret-space-separated-token-key"
            ),
            pytest.param("host.com/db?x\tpassword=hunter2", "hunter2", id="scheme-less-query-secret-tab-separated"),
            pytest.param("host.com/db?x,password=hunter2", "hunter2", id="scheme-less-query-secret-comma-separated"),
            pytest.param("host.com/db?x+password=hunter2", "hunter2", id="scheme-less-query-secret-plus-separated"),
            pytest.param(
                "host.com/db?x%20password=hunter2",
                "hunter2",
                id="scheme-less-query-secret-percent-encoded-space-separated",
            ),
            pytest.param("host.com/db?jwt=hunter2", "hunter2", id="scheme-less-query-secret-key-jwt"),
            pytest.param("host.com/db?p=hunter2", "hunter2", id="scheme-less-query-secret-key-p"),
            pytest.param("host.com/db?sessionid=hunter2", "hunter2", id="scheme-less-query-secret-key-sessionid"),
            pytest.param(
                "host.com/db?redirect=https://x&password=y",
                "password=y",
                id="embedded-scheme-in-query-is-not-misrouted-to-the-uri-branch",
            ),
            pytest.param(
                "host.com/db?a=1;password=2", "password=2", id="semicolon-bounded-secret-after-question-mark-anchor"
            ),
            pytest.param(
                "host.com/db?a=1&b=2;password=3",
                "password=3",
                id="semicolon-bounded-secret-after-ampersand-then-question-mark-anchor",
            ),
            pytest.param("host.com/db;password=hunter2", "hunter2", id="semicolon-bounded-secret-with-no-anchor"),
            pytest.param("some/path/user=alice.csv", None, id="ordinary-path-with-connection-key-lookalike"),
            pytest.param("data/q?a/report=2024.csv", None, id="query-lookalike-path"),
            pytest.param(
                "Data Source=srv;User Id=alice;Password=hunter2", "hunter2", id="semicolon-separated-keyword-string"
            ),
            pytest.param("/srv/a;host=b", None, id="absolute-path-with-semicolon-key"),
            pytest.param("C:\\data;user=1", None, id="windows-path-with-semicolon-key"),
            pytest.param("host.com/db?limit=10", None, id="scheme-less-query-with-non-secret-key"),
            pytest.param("data/plain.csv", None, id="relative-path"),
            pytest.param("user=42/part.parquet", None, id="connection-key-lookalike-path"),
            pytest.param("/data/year=2024/part.parquet", None, id="hive-path"),
            pytest.param("report=2024.csv", None, id="unknown-key-file-name"),
            pytest.param("a=1 b=2", None, id="unknown-keys"),
            pytest.param("year=2024 month=01", None, id="unknown-keys-partition-like"),
            pytest.param("", None, id="empty-string"),
            pytest.param("C:\\dir\\a@b", None, id="windows-backslash-path"),
            pytest.param("C:/a@b", None, id="windows-forward-slash-path"),
            pytest.param("me@work.txt", None, id="email-like"),
            pytest.param("alice@host/db", None, id="username-only-userinfo"),
            pytest.param("/mnt/share/my db=main.csv", None, id="path-with-space-and-key"),
            pytest.param("/var/log/app db=1.log", None, id="path-with-space-and-db-key"),
            pytest.param(Path("data/plain.csv"), None, id="plain-path-object"),
            pytest.param("data/file%3Fname.csv", None, id="percent-encoded-question-mark-in-file-name"),
            pytest.param("host.com/db%3Flimit=10", None, id="percent-encoded-question-mark-with-non-secret-key"),
        ],
    )
    def test_type_name_only_hides_any_secret(self, value: Any, secret: str | None) -> None:
        identity = _identity_of(value)
        assert identity == type(value).__name__
        if secret is not None:
            assert secret not in identity

    def test_long_string_without_equals_returns_quickly(self) -> None:
        value = "a " * 20000
        assert _identity_of(value) == "str"


class _SecretBearingAccess:
    """Object whose repr and str both carry a secret."""

    def __repr__(self) -> str:
        return "SecretBearingAccess(password=hunter2)"

    __str__ = __repr__


class TestDataAccessIdentityOfNonStringValues:
    """Mappings are identified by sorted keys; any other non-str, including every PurePath, by type name only."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                MappingProxyType({"user": "alice", "password": "hunter2"}),  # nosec B105
                "{password, user}",
                id="mapping-proxy-key-names-only",
            ),
            pytest.param(_SecretBearingAccess(), "_SecretBearingAccess", id="arbitrary-object-type-name"),
            pytest.param(b"postgresql://u:hunter2@host/db", "bytes", id="bytes-type-name"),
            pytest.param(["postgresql://u:hunter2@host/db"], "list", id="list-type-name"),
            pytest.param(Path("data/plain.csv"), type(Path("data/plain.csv")).__name__, id="plain-path-type-name"),
        ],
    )
    def test_identity_of_non_string_value(self, value: Any, expected: str) -> None:
        assert _identity_of(value) == expected


_LOCAL_PATH_CASES: tuple[tuple[str, str], ...] = (
    ("plain.csv", "plain-file"),
    ("user=42/part.parquet", "connection-key-lookalike-dir"),
    ("year=2024/part.parquet", "hive-style-dir"),
    ("report=2024.csv", "unknown-key-file-name"),
    ("a@b", "at-sign-in-name"),
    ("me@work.txt", "email-like-name"),
    ("my db=main.csv", "space-and-key-in-name"),
    ("a;host=b", "semicolon-key-in-name"),
    ("q?a/report=2024.csv", "query-lookalike-dir"),
    ("file%3Fname.csv", "percent-encoded-question-mark-in-name"),
    ("notes:2024@work.txt", "userinfo-lookalike-name"),
    ("user=alice.csv", "file-name-starting-with-connection-key"),
)


class TestDataAccessIdentityOfExistingLocalPaths:
    """An existing local file or directory is published as given, for both the str and the Path form;
    the same name under a missing parent falls back to the type name."""

    @pytest.mark.parametrize(
        "relative", [pytest.param(relative, id=case_id) for relative, case_id in _LOCAL_PATH_CASES]
    )
    def test_existing_file_is_published_as_given(self, tmp_path: Path, relative: str) -> None:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

        assert _identity_of(str(path)) == str(path)
        assert _identity_of(path) == str(path)

    def test_existing_directory_is_published_as_given(self, tmp_path: Path) -> None:
        directory = tmp_path / "user=42"
        directory.mkdir()

        assert _identity_of(str(directory)) == str(directory)
        assert _identity_of(directory) == str(directory)

    @pytest.mark.parametrize(
        "relative", [pytest.param(relative, id=case_id) for relative, case_id in _LOCAL_PATH_CASES]
    )
    def test_same_name_under_a_missing_parent_is_the_type_name(self, tmp_path: Path, relative: str) -> None:
        path = tmp_path / "missing" / relative
        assert _identity_of(str(path)) == "str"
        assert _identity_of(path) == type(path).__name__


class TestDataAccessIdentityRegressionGuardForReportedLeak:
    """A credential-shaped value must never come back verbatim from any reader family's data_access_identity."""

    @pytest.mark.parametrize("reader", [CsvReader, TextFileReader, ReadFile, ReadDocument])
    @pytest.mark.parametrize(
        ("value", "secret"),
        [
            pytest.param("u:hunter2@fileserver/share/notes.txt", "hunter2", id="userinfo-scheme-less-file-path"),
            pytest.param("host=h password=hunter2 notes.txt", "hunter2", id="keyword-string-with-file-name"),
            pytest.param(PurePosixPath("s3://alice:hunter2@bucket/key.csv"), "hunter2", id="pure-posix-path-uri"),
        ],
    )
    def test_credential_shaped_value_never_comes_back_verbatim(
        self, reader: type[BaseInputData], value: Any, secret: str
    ) -> None:
        identity = reader.data_access_identity(value)
        assert identity == type(value).__name__
        assert secret not in identity


class TestDataAccessIdentityWiring:
    """The hook reads the identity from the reader's own data_access_identity classmethod."""

    def test_hook_uses_the_readers_data_access_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_DirectLoadReader, "data_access_identity", classmethod(lambda cls, data_access: "sentinel"))
        assert _identity_of("anything") == "sentinel"


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


class TestCarrierIsNotAliasedAcrossTwoInputDataLoadHookContexts:
    """Two INPUT_DATA_LOAD HookContexts built off the SAME ComputeFramework instance's
    run_context.carrier must not share the dict object."""

    def test_two_direct_load_calls_get_distinct_carrier_objects(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        cfw = ComputeFramework(function_extender={extender})
        cfw.run_context = RunContext(carrier=carrier)
        reader = _DirectLoadReader()
        features = FeatureSet()

        with cfw.activate(), _build_calc_context().activate():
            BaseInputData._load_data_via_hook(reader, "access-one", features)
        first_context = extender.captured

        with cfw.activate(), _build_calc_context().activate():
            BaseInputData._load_data_via_hook(reader, "access-two", features)
        second_context = extender.captured

        assert first_context is not None
        assert second_context is not None
        assert first_context.carrier == second_context.carrier == carrier
        assert first_context.carrier is not second_context.carrier

    def test_mutating_one_carrier_does_not_leak_into_the_other_or_run_context(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        cfw = ComputeFramework(function_extender={extender})
        cfw.run_context = RunContext(carrier=carrier)
        reader = _DirectLoadReader()
        features = FeatureSet()

        with cfw.activate(), _build_calc_context().activate():
            BaseInputData._load_data_via_hook(reader, "access-one", features)
        first_context = extender.captured

        with cfw.activate(), _build_calc_context().activate():
            BaseInputData._load_data_via_hook(reader, "access-two", features)
        second_context = extender.captured

        assert first_context is not None
        assert first_context.carrier is not None
        first_context.carrier["mutated"] = "yes"

        assert second_context is not None
        assert second_context.carrier is not None
        assert "mutated" not in second_context.carrier
        assert cfw.run_context.carrier is not None
        assert "mutated" not in cfw.run_context.carrier


class TestInputDataLoadHookCarriesInputFeatureEdges:
    """The INPUT_DATA_LOAD HookContext copies input_feature_edges from the enclosing calculate context."""

    def test_edges_are_copied_from_the_calculate_context(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = ComputeFramework(function_extender={extender})
        cfw.run_context = RunContext()
        reader = _DirectLoadReader()
        calc_context = _build_calc_context()
        calc_context.input_feature_edges = {"a": ("src_a",), "b": ("src_b",)}

        with cfw.activate(), calc_context.activate():
            BaseInputData._load_data_via_hook(reader, "access", FeatureSet())

        assert extender.captured is not None
        assert extender.captured.input_feature_edges == {"a": ("src_a",), "b": ("src_b",)}

    def test_edges_default_to_none_when_the_calculate_context_has_none(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = ComputeFramework(function_extender={extender})
        cfw.run_context = RunContext()
        reader = _DirectLoadReader()

        with cfw.activate(), _build_calc_context().activate():
            BaseInputData._load_data_via_hook(reader, "access", FeatureSet())

        assert extender.captured is not None
        assert extender.captured.input_feature_edges is None


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
