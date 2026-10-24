"""Tests wiring ExtenderHook.INPUT_DATA_LOAD into dispatch_input_data_load().

Covers ComputeFramework.current(), HookContext population (inherited identity fields plus
data_access_identity/format/dataset_version), the no-extender baseline, deny-before-load /
deny-with-fallback, and the "activate only when needed" short-circuit.
"""

import concurrent.futures
import contextvars
import copy
import logging
import os
import pickle  # nosec B403
import sqlite3
import threading
from pathlib import Path, PurePosixPath
from collections.abc import Mapping
from typing import Any, cast

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.input_data.base_input_data import dispatch_input_data_load
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.implementation import loaded_implementations
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy
from mloda.core.abstract_plugins.components.input_data.read_db_fg import DBTable
from mloda.provider import ApiDataFG, GeneratedFeatureGroup, FeatureGroup, FileSource, InputDataDescriptor, SourceMatch
from mloda.steward import GateBypassError
from mloda.user import (
    DataAccessCollection,
    Feature,
    FeatureName,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda_plugins.feature_group.input_data.db_sources.sqlite_fg import SqliteFG
from tests.mixins.reader_feature_groups.format_file_writers import write_sqlite
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    neutral_csv_group,
    toy_dac,
)

_MARKER = "inputload051"
_EXPECTED_FEATURE_GROUP_CLASS = f"{CsvFG.__module__}.{CsvFG.__qualname__}"


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
        self.all_captured: list[HookContext] = []
        self.results: list[Any] = []
        self.func_owner: str | None = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.func_owner = Extender.feature_group_name(func)
        result = func(*args, **kwargs)
        self.results.append(result)
        self.captured = HookContext.current()
        if self.captured is not None:
            self.all_captured.append(self.captured)
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
            compute_frameworks=[PythonDictFramework],
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
        assert fetch_context.structure_hash is not None
        assert fetch_context.structure_hash == calc_context.structure_hash
        assert fetch_context.carrier == calc_context.carrier
        assert fetch_context.worker_index == calc_context.worker_index
        assert fetch_context.tenant_id == calc_context.tenant_id
        assert fetch_context.project_id == calc_context.project_id
        assert fetch_context.principal == calc_context.principal
        assert fetch_context.compute_framework_name == calc_context.compute_framework_name
        assert fetch_context.feature_group_class == calc_context.feature_group_class
        assert calc_context.step_uuid is not None
        assert fetch_context.step_uuid is not None
        assert fetch_context.step_uuid == calc_context.step_uuid

        assert fetch_extender.results
        assert all(isinstance(r, InputDataDescriptor) for r in fetch_extender.results)
        assert any(isinstance(r, FileSource) for r in fetch_extender.results)
        assert fetch_context.status == "success"
        assert fetch_context.rows_out is None


class TestInputDataLoadHookFiresWithOnlyFetchExtenderRegistered:
    """No calculate extender: the calculate-phase HookContext is still built and activated so INPUT_DATA_LOAD can read from it."""

    def test_fetch_context_still_carries_calculate_phase_identity_fields(self, tmp_path: Path) -> None:
        column = f"{_MARKER}_col_b"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [4, 5, 6])

        fetch_extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            [column],
            compute_frameworks=[PythonDictFramework],
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
            compute_frameworks=[PythonDictFramework],
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
                compute_frameworks=[PythonDictFramework],
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
                compute_frameworks=[PythonDictFramework],
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
            compute_frameworks=[PythonDictFramework],
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
        original_load_neutral = CsvFG.load_neutral.__func__  # type: ignore[attr-defined]

        def _probe_load_neutral(cls: type, match: Any, features: Any) -> Any:
            observed.append(ComputeFramework.current())
            return original_load_neutral(cls, match, features)

        monkeypatch.setattr(CsvFG, "load_neutral", classmethod(_probe_load_neutral))

        mloda.run_all(
            [column],
            compute_frameworks=[PythonDictFramework],
            data_access_collection=DataAccessCollection(files={str(path)}),
            function_extender={_NoOpValidateInputFeatureExtender()},
        )

        assert observed
        assert observed[0] is None


def _build_calc_context(compute_framework_name: str = "stub", **overrides: Any) -> HookContext:
    """A minimal calculate-phase HookContext to activate() around a direct dispatch_input_data_load call."""
    fields: dict[str, Any] = {
        "hook": ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
        "feature_group_class": "test.Fake",
        "feature_group_version": "1",
        "plugin_version": None,
        "feature_names": ("x",),
        "input_features": None,
        "compute_framework_name": compute_framework_name,
    }
    fields.update(overrides)
    return HookContext(**fields)


class _DirectLoader:
    """Minimal loader whose load_data returns a fixed 3-element list, for direct dispatch_input_data_load calls."""

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return [1, 2, 3]

    @classmethod
    def data_access_identity(cls, data_access: Any) -> str:
        if isinstance(data_access, Mapping):
            return "{" + ", ".join(sorted(str(key) for key in data_access)) + "}"
        return type(data_access).__name__

    @classmethod
    def data_access_name(cls) -> str:
        return cls.__name__

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {}


def _load(loader: type[_DirectLoader], data_access: Any, features: FeatureSet) -> Any:
    """Run loader.load_data through dispatch_input_data_load with an identity lambda."""
    return dispatch_input_data_load(
        loader,
        loader,
        loader.load_data,
        data_access,
        features,
        identity=lambda: loader.data_access_identity(data_access),
        data_access_format=loader.data_access_name,
    )


class TestDataAccessIdentityHidesDictCredentialValues:
    """Dict credentials publish only the sqlite path in data_access_identity, never other values."""

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
            compute_frameworks=[PyArrowTable],
            data_access_collection=DataAccessCollection(
                credentials=[{"sqlite": str(db_path), "user": "alice", "password": "hunter2"}]  # nosec B105
            ),
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
            function_extender={fetch_extender},
        )

        fetch_context = fetch_extender.captured
        assert fetch_context is not None
        identity = fetch_context.data_access_identity
        assert identity is not None
        assert "hunter2" not in identity
        assert "alice" not in identity
        assert identity == f"{db_path}::creds_table"
        assert fetch_context.data_access_identity_is_fallback is False

    def test_features_from_different_tables_of_one_file_get_distinct_identities(self, tmp_path: Path) -> None:
        db_path = tmp_path / "two_tables.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE table_a (col_a INTEGER)")
        conn.execute("CREATE TABLE table_b (col_b INTEGER)")
        conn.execute("INSERT INTO table_a VALUES (1)")
        conn.execute("INSERT INTO table_b VALUES (2)")
        conn.commit()
        conn.close()

        extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            ["col_a", "col_b"],
            compute_frameworks=[PyArrowTable],
            data_access_collection=DataAccessCollection(credentials=[{"sqlite": str(db_path)}]),
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
            function_extender={extender},
        )

        identities = {c.data_access_identity for c in extender.all_captured}
        assert identities == {f"{db_path}::table_a", f"{db_path}::table_b"}


class TestSqliteFGSources:
    """SqliteFG sources carry the credential-free identity: the absolute path, plus ``::table`` per table."""

    _ROUTE = ClaimRoute("credentials", NamePolicy.CHECKED, True)

    def _sources(self, credential: dict[str, Any]) -> list[SourceMatch]:
        dac = DataAccessCollection(credentials={"identity_handle": credential})
        return SqliteFG.find_sources(self._ROUTE, "identity_col", Options(), dac)

    @pytest.mark.parametrize("kind", ["missing_file", "directory"])
    def test_an_unreadable_database_is_one_source_named_by_its_path_alone(self, tmp_path: Path, kind: str) -> None:
        target = tmp_path / "missing.db" if kind == "missing_file" else tmp_path
        sources = self._sources({"sqlite": str(target)})
        assert [match.source for match in sources] == [os.path.abspath(target)]
        assert isinstance(sources[0].access, DBTable)
        assert sources[0].access.table is None
        assert SqliteFG.data_access_identity(sources[0]) == os.path.abspath(target)

    @pytest.mark.parametrize("kind", ["path_object", "missing_key"])
    def test_an_invalid_credential_yields_no_source(self, tmp_path: Path, kind: str) -> None:
        credential = {"sqlite": tmp_path / "x.db"} if kind == "path_object" else {"user": "alice"}
        assert self._sources(credential) == []

    def test_path_and_table_name_join_with_double_colon(self, tmp_path: Path) -> None:
        db = tmp_path / "x.db"
        write_sqlite(db, {"orders": {"identity_col": [1]}})
        assert [match.source for match in self._sources({"sqlite": str(db)})] == [f"{db}::orders"]

    @pytest.mark.parametrize("table_name", ["", None])
    def test_an_empty_table_name_is_no_preset(self, tmp_path: Path, table_name: Any) -> None:
        db = tmp_path / "x.db"
        write_sqlite(db, {"orders": {"identity_col": [1]}, "items": {"identity_col": [2]}})
        sources = self._sources({"sqlite": str(db), "table_name": table_name})
        assert sorted(match.source for match in sources) == [f"{db}::items", f"{db}::orders"]

    def test_password_is_not_leaked_into_the_source_or_its_repr(self, tmp_path: Path) -> None:
        db = tmp_path / "x.db"
        write_sqlite(db, {"orders": {"identity_col": [1]}})
        credential = {"sqlite": str(db), "user": "alice", "password": "hunter2", "table_name": "orders"}  # nosec B105
        sources = self._sources(credential)
        assert [match.source for match in sources] == [f"{db}::orders"]
        assert "hunter2" not in repr(sources[0])
        assert "alice" not in repr(sources[0])


class TestSourceGroupIdentityNeverEchoesTheAccess:
    """A source group publishes the match's source, never a credential-shaped access value."""

    def test_identity_is_the_source_not_the_access(self) -> None:
        match = SourceMatch(source="/data/key.csv", access=PurePosixPath("s3://alice:hunter2@bucket/key.csv"))

        identity = CsvFG.data_access_identity(match)

        assert identity == "/data/key.csv"
        assert "hunter2" not in identity


class TestDataAccessIdentityWiring:
    """The hook reads the identity from the reader's own data_access_identity classmethod."""

    def test_hook_uses_the_readers_data_access_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = type("_SentinelIdentityLoader", (_DirectLoader,), {})
        monkeypatch.setattr(loader, "data_access_identity", classmethod(lambda cls, data_access: "sentinel"))
        assert _direct_load_context(loader, "anything").data_access_identity == "sentinel"


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
            compute_frameworks=[PythonDictFramework],
            data_access_collection=DataAccessCollection(files={str(path)}),
            function_extender={fetch_extender},
        )

        fetch_context = fetch_extender.captured
        assert fetch_context is not None
        identity = fetch_context.data_access_identity
        assert identity
        assert str(path) in identity


class TestCarrierIsNotAliasedAcrossTwoInputDataLoadHookContexts:
    """Two INPUT_DATA_LOAD HookContexts derived from ONE active calculate context's carrier
    must not share the dict object with each other or with that source."""

    def _two_loads(self) -> tuple[HookContext, HookContext, HookContext]:
        extender = _InputDataLoadCapturingExtender()
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        cfw = ComputeFramework(function_extender={extender})
        cfw.run_context = RunContext()
        features = FeatureSet()
        calc_context = _build_calc_context(carrier=carrier)

        with cfw.activate(), calc_context.activate():
            _load(_DirectLoader, "access-one", features)
            first_context = extender.captured
            _load(_DirectLoader, "access-two", features)
            second_context = extender.captured

        assert first_context is not None
        assert second_context is not None
        return first_context, second_context, calc_context

    def test_two_direct_load_calls_get_distinct_carrier_objects(self) -> None:
        first_context, second_context, calc_context = self._two_loads()

        assert first_context.carrier == second_context.carrier == calc_context.carrier
        assert first_context.carrier is not second_context.carrier
        assert first_context.carrier is not calc_context.carrier
        assert second_context.carrier is not calc_context.carrier

    def test_mutating_one_carrier_does_not_leak_into_the_other_or_source(self) -> None:
        first_context, second_context, calc_context = self._two_loads()

        assert first_context.carrier is not None
        with pytest.raises(TypeError):
            first_context.carrier["mutated"] = "yes"

        assert second_context.carrier is not None
        assert "mutated" not in second_context.carrier
        assert calc_context.carrier is not None
        assert "mutated" not in calc_context.carrier


class TestInputDataLoadHookCarriesInputFeatureEdges:
    """The INPUT_DATA_LOAD HookContext copies input_feature_edges from the enclosing calculate context."""

    def test_edges_are_copied_from_the_calculate_context(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = ComputeFramework(function_extender={extender})
        cfw.run_context = RunContext()
        calc_context = _build_calc_context(input_feature_edges={"a": ("src_a",), "b": ("src_b",)})

        with cfw.activate(), calc_context.activate():
            _load(_DirectLoader, "access", FeatureSet())

        assert extender.captured is not None
        assert extender.captured.input_feature_edges == {"a": ("src_a",), "b": ("src_b",)}

    def test_edges_default_to_none_when_the_calculate_context_has_none(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = ComputeFramework(function_extender={extender})
        cfw.run_context = RunContext()

        with cfw.activate(), _build_calc_context().activate():
            _load(_DirectLoader, "access", FeatureSet())

        assert extender.captured is not None
        assert extender.captured.input_feature_edges is None


class TestInputDataLoadIdentityComesFromTheCalcContext:
    """A forged cfw.run_context cannot change the identity/run fields the INPUT_DATA_LOAD context carries."""

    def test_forged_cfw_run_context_is_ignored(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = ComputeFramework(function_extender={extender})
        calc_context = _build_calc_context(
            run_id="real-run",
            carrier={"traceparent": "real"},
            tenant_id="real-tenant",
            project_id="real-project",
            principal="real-principal",
            worker_index=5,
        )
        cfw.run_context = RunContext(
            run_id="forged-run",
            carrier={"traceparent": "forged"},
            tenant_id="forged-tenant",
            project_id="forged-project",
            principal="forged-principal",
        )
        cfw.worker_index = 9

        with cfw.activate(), calc_context.activate():
            _load(_DirectLoader, "access", FeatureSet())

        captured = extender.captured
        assert captured is not None
        assert captured.run_id == "real-run"
        assert captured.carrier == {"traceparent": "real"}
        assert captured.tenant_id == "real-tenant"
        assert captured.project_id == "real-project"
        assert captured.principal == "real-principal"
        assert captured.worker_index == 5


class _GateInputDataLoadExtender(_InputDataLoadCapturingExtender):
    """A gate INPUT_DATA_LOAD extender that delegates and records its calls."""

    never_fall_back = True


_API_COLUMN = f"{_MARKER}_api_col"


class _CountingReader(_DirectLoader):
    """Reader counting its load_data calls, for direct dispatch_input_data_load calls."""

    loads = 0

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        cls.loads += 1
        return super().load_data(data_access, features)


def _threaded_load_feature_group(mode: str, leaked: list[FeatureSet] | None = None) -> type[FeatureGroup]:
    """A FeatureGroup whose calculate_feature loads the reader from a worker thread.

    mode: bare/copied (FeatureSet built in the hopped thread), received (the calculate FeatureSet),
    built_in_calc (built in the calculation thread), deepcopy (copy of the received one), late (received one
    appended to `leaked` and not loaded in the calculation).
    """

    class _ThreadedLoadFeatureGroup(FeatureGroup):
        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            hopped: FeatureSet | None = None
            if mode == "received":
                hopped = features
            elif mode == "built_in_calc":
                hopped = FeatureSet()
            elif mode == "deepcopy":
                hopped = copy.deepcopy(features)
            elif mode == "late":
                assert leaked is not None
                leaked.append(features)
                return [0]

            def load() -> Any:
                return _load(_CountingReader, "access", FeatureSet() if hopped is None else hopped)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                if mode == "copied":
                    return pool.submit(contextvars.copy_context().run, load).result()
                return pool.submit(load).result()

    return _ThreadedLoadFeatureGroup


def _load_in_thread(features: FeatureSet) -> tuple[Any, BaseException | None]:
    """Load the reader with `features` from a fresh thread; returns (result, raised exception)."""
    outcome: list[Any] = []

    def run() -> None:
        try:
            outcome.append((_load(_CountingReader, "access", features), None))
        except BaseException as exc:  # noqa: BLE001
            outcome.append((None, exc))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    result: tuple[Any, BaseException | None] = outcome[0]
    return result


def _framework_for(kind: str) -> ComputeFramework:
    extenders: dict[str, set[Extender]] = {
        "gate": {_GateInputDataLoadExtender()},
        "non_gate": {_InputDataLoadCapturingExtender()},
        "none": set(),
    }
    return PythonDictFramework(function_extender=extenders[kind])


def _run_calculate(cfw: ComputeFramework, feature_group: type[FeatureGroup]) -> Any:
    return cfw.run_calculate_feature(feature_group, FeatureSet())


class TestThreadHoppedReaderLoadIsDispatchedPerRun:
    """A FeatureSet stamped by its run dispatches a hopped load through the hook; unstamped ones fail closed."""

    def test_received_feature_set_hop_reaches_the_gate_with_the_calculate_identity(self) -> None:
        gate = _GateInputDataLoadExtender()
        calc = _CalcContextCapturingExtender()
        cfw = PythonDictFramework(function_extender={gate, calc})
        cfw.run_context = RunContext(run_id="run-hop", tenant_id="tenant-hop", principal="principal-hop")
        _CountingReader.loads = 0

        result = _run_calculate(cfw, _threaded_load_feature_group("received"))

        assert result == [1, 2, 3]
        assert len(gate.all_captured) == 1
        assert _CountingReader.loads == 1
        assert calc.captured is not None
        load_context = gate.all_captured[0]
        assert load_context.hook == ExtenderHook.INPUT_DATA_LOAD
        assert load_context.run_id == calc.captured.run_id == "run-hop"
        assert load_context.tenant_id == calc.captured.tenant_id == "tenant-hop"
        assert load_context.principal == calc.captured.principal == "principal-hop"
        assert load_context.feature_group_class == calc.captured.feature_group_class

    def test_non_gate_extender_observes_the_hopped_load(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = PythonDictFramework(function_extender={extender})
        _CountingReader.loads = 0

        assert _run_calculate(cfw, _threaded_load_feature_group("received")) == [1, 2, 3]

        assert len(extender.all_captured) == 1
        assert _CountingReader.loads == 1

    def test_a_gate_free_run_hops_while_another_runs_gated_scope_is_open(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        class _BlockingFeatureGroup(FeatureGroup):
            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                entered.set()
                assert release.wait(timeout=5)
                return [0]

        def gated_run() -> None:
            try:
                _run_calculate(_framework_for("gate"), _BlockingFeatureGroup)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=gated_run)
        thread.start()
        try:
            assert entered.wait(timeout=5)
            for kind in ("none", "non_gate"):
                _CountingReader.loads = 0
                result = _run_calculate(_framework_for(kind), _threaded_load_feature_group("received"))
                assert result == [1, 2, 3]
                assert _CountingReader.loads == 1
        finally:
            release.set()
            thread.join(timeout=5)

        assert not thread.is_alive()
        assert errors == []

    def test_feature_set_built_in_the_calculation_thread_and_loaded_in_a_pool_thread_is_gated(self) -> None:
        gate = _GateInputDataLoadExtender()
        cfw = PythonDictFramework(function_extender={gate})
        _CountingReader.loads = 0

        result = _run_calculate(cfw, _threaded_load_feature_group("built_in_calc"))

        assert result == [1, 2, 3]
        assert len(gate.all_captured) == 1
        assert _CountingReader.loads == 1

    def test_deepcopy_made_in_the_calculation_stays_stamped(self) -> None:
        gate = _GateInputDataLoadExtender()
        cfw = PythonDictFramework(function_extender={gate})
        _CountingReader.loads = 0

        result = _run_calculate(cfw, _threaded_load_feature_group("deepcopy"))

        assert result == [1, 2, 3]
        assert len(gate.all_captured) == 1
        assert _CountingReader.loads == 1

    @pytest.mark.parametrize("kind", ["gate", "non_gate", "none"])
    def test_load_after_the_calculation_returned(self, kind: str) -> None:
        leaked: list[FeatureSet] = []
        cfw = _framework_for(kind)
        _run_calculate(cfw, _threaded_load_feature_group("late", leaked))
        _CountingReader.loads = 0

        result, error = _load_in_thread(leaked[0])

        if kind == "gate":
            assert isinstance(error, GateBypassError)
            assert _CountingReader.loads == 0
        else:
            assert error is None
            assert result == [1, 2, 3]
            assert _CountingReader.loads == 1

    def test_inactive_prior_marker_is_not_restored_over_a_gated_calculation_marker(self) -> None:
        leaked: list[FeatureSet] = []
        _run_calculate(_framework_for("non_gate"), _threaded_load_feature_group("late", leaked))
        gated = _framework_for("gate")
        gated.run_calculate_feature(_threaded_load_feature_group("late", []), leaked[0])
        _CountingReader.loads = 0

        result, error = _load_in_thread(leaked[0])

        assert isinstance(error, GateBypassError)
        assert _CountingReader.loads == 0

    @pytest.mark.parametrize("kind", ["gate", "non_gate", "none"])
    def test_pickled_feature_set_loaded_outside_the_scope(self, kind: str) -> None:
        leaked: list[FeatureSet] = []
        cfw = _framework_for(kind)
        _run_calculate(cfw, _threaded_load_feature_group("late", leaked))
        restored = pickle.loads(pickle.dumps(leaked[0]))  # nosec B301
        _CountingReader.loads = 0

        if kind == "gate":
            with pytest.raises(GateBypassError):
                _load(_CountingReader, "access", restored)
            assert _CountingReader.loads == 0
        else:
            assert _load(_CountingReader, "access", restored) == [1, 2, 3]
            assert _CountingReader.loads == 1

    def test_bare_thread_hop_with_a_self_built_feature_set_raises_and_the_reader_never_loads(self) -> None:
        gate = _GateInputDataLoadExtender()
        cfw = PythonDictFramework(function_extender={gate})
        _CountingReader.loads = 0

        with pytest.raises(GateBypassError, match="copy_context"):
            _run_calculate(cfw, _threaded_load_feature_group("bare"))

        assert _CountingReader.loads == 0
        assert len(gate.all_captured) == 0

    def test_non_gate_extender_still_loads_across_a_thread_hop(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = PythonDictFramework(function_extender={extender})
        _CountingReader.loads = 0

        result = _run_calculate(cfw, _threaded_load_feature_group("bare"))

        assert result == [1, 2, 3]
        assert _CountingReader.loads == 1

    def test_no_extender_still_loads_across_a_thread_hop(self) -> None:
        cfw = PythonDictFramework()
        _CountingReader.loads = 0

        assert _run_calculate(cfw, _threaded_load_feature_group("bare")) == [1, 2, 3]
        assert _CountingReader.loads == 1

    def test_copy_context_thread_hop_reaches_the_gate(self) -> None:
        gate = _GateInputDataLoadExtender()
        cfw = PythonDictFramework(function_extender={gate})
        _CountingReader.loads = 0

        result = _run_calculate(cfw, _threaded_load_feature_group("copied"))

        assert result == [1, 2, 3]
        assert len(gate.all_captured) == 1
        assert _CountingReader.loads == 1

    def test_gate_active_cfw_without_calc_context_raises(self) -> None:
        gate = _GateInputDataLoadExtender()
        cfw = ComputeFramework(function_extender={gate})
        _CountingReader.loads = 0

        with cfw.activate(), pytest.raises(GateBypassError):
            _load(_CountingReader, "access", FeatureSet())

        assert _CountingReader.loads == 0
        assert len(gate.all_captured) == 0

    def test_non_gate_extender_without_calc_context_calls_the_reader_directly(self) -> None:
        extender = _InputDataLoadCapturingExtender()
        cfw = ComputeFramework(function_extender={extender})

        with cfw.activate():
            result = _load(_CountingReader, "access", FeatureSet())

        assert result == [1, 2, 3]
        assert extender.captured is None

    def test_gate_scope_is_released_when_the_calculate_raises(self) -> None:
        class _RaisingFeatureGroup(FeatureGroup):
            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                raise RuntimeError("calculate boom")

        cfw = PythonDictFramework(function_extender={_GateInputDataLoadExtender()})
        with pytest.raises(RuntimeError, match="calculate boom"):
            _run_calculate(cfw, _RaisingFeatureGroup)

        _CountingReader.loads = 0
        result = _load(_CountingReader, "access", FeatureSet())

        assert result == [1, 2, 3]
        assert _CountingReader.loads == 1

    def test_gate_wraps_the_api_data_load(self) -> None:
        gate = _GateInputDataLoadExtender()

        result = mloda.run_all(
            [_API_COLUMN],
            compute_frameworks=[PythonDictFramework],
            api_data={"gated": {_API_COLUMN: [1, 2]}},
            function_extender={gate},
        )

        assert len(gate.all_captured) == 1
        assert dict(result[0]) == {_API_COLUMN: [1, 2]}


class _DeclaringReader(_DirectLoader):
    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str | int | float | bool]:
        return {"unit": "mm", "scale": 3, "skipped": [1], "bad": float("nan")}  # type: ignore[dict-item]


class _RaisingDeclarationReader(_DirectLoader):
    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str | int | float | bool]:
        raise RuntimeError("reader declaration boom")


def _direct_load_context(reader_cls: type[_DirectLoader], data_access: Any = "access") -> HookContext:
    extender = _InputDataLoadCapturingExtender()
    cfw = ComputeFramework(function_extender={extender})
    with cfw.activate(), _build_calc_context().activate():
        _load(reader_cls, data_access, FeatureSet())
    assert extender.captured is not None
    return extender.captured


_BASE_INPUT_DATA_LOGGER = "mloda.core.abstract_plugins.components.input_data.base_input_data"


class _NamedIdentityReader(_DirectLoader):
    @classmethod
    def data_access_identity(cls, data_access: Any) -> str:
        return "db://orders"


class _AccessIdentityLoader(_DirectLoader):
    """Publishes the access itself as the identity, like a path or URI naming its source."""

    @classmethod
    def data_access_identity(cls, data_access: Any) -> str:
        return str(data_access)


class TestFallbackDataAccessIdentityIsFlagged:
    """A fallback identity names no source: flagged on the hook context and warned once per reader class."""

    @pytest.mark.parametrize("kind", ["missing-path", "object", "dict", "jdbc-uri"])
    def test_fallback_identity_is_flagged(self, kind: str, tmp_path: Path) -> None:
        data_access: Any = {
            "missing-path": str(tmp_path / "missing.csv"),
            "object": object(),
            "dict": {"sqlite": "/data/a.db"},
            "jdbc-uri": "jdbc:sqlserver://host:1433;databaseName=db;user=a;password=hunter2",
        }[kind]

        context = _direct_load_context(_DirectLoader, data_access)

        assert context.data_access_identity_is_fallback is True

    @pytest.mark.parametrize("kind", ["existing-path", "uri", "override", "override-dict"])
    def test_published_identity_is_not_flagged(
        self, kind: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        existing = tmp_path / "a.csv"
        existing.write_text("x\n1\n")
        data_access: Any = {
            "existing-path": str(existing),
            "uri": "s3://bucket/a.csv",
            "override": "nowhere",
            "override-dict": {"sqlite": "/data/a.db"},
        }[kind]
        base_cls = _NamedIdentityReader if kind.startswith("override") else _AccessIdentityLoader
        reader_cls = type("_FreshPublished", (base_cls,), {})

        with caplog.at_level(logging.WARNING, logger=_BASE_INPUT_DATA_LOGGER):
            context = _direct_load_context(reader_cls, data_access)

        assert context.data_access_identity_is_fallback is False
        assert not [
            r for r in caplog.records if r.name == _BASE_INPUT_DATA_LOGGER and "data_access_identity" in r.getMessage()
        ]

    def test_warns_once_per_reader_class(self, caplog: pytest.LogCaptureFixture) -> None:
        class _FreshA(_DirectLoader):
            pass

        class _FreshB(_DirectLoader):
            pass

        with caplog.at_level(logging.WARNING, logger=_BASE_INPUT_DATA_LOGGER):
            _direct_load_context(_FreshA, "missing-a")
            _direct_load_context(_FreshA, "missing-a")
            records = [r for r in caplog.records if r.name == _BASE_INPUT_DATA_LOGGER]
            assert len(records) == 1
            assert _FreshA.__qualname__ in records[0].getMessage()
            assert "data_access_identity" in records[0].getMessage()

            _direct_load_context(_FreshB, "missing-b")
            records = [r for r in caplog.records if r.name == _BASE_INPUT_DATA_LOGGER]
            assert len(records) == 2
            assert _FreshB.__qualname__ in records[1].getMessage()

    def test_no_warning_without_input_data_load_extender(self, caplog: pytest.LogCaptureFixture) -> None:
        class _FreshNoExtender(_DirectLoader):
            pass

        cfw = ComputeFramework(function_extender=set())
        with caplog.at_level(logging.WARNING, logger=_BASE_INPUT_DATA_LOGGER):
            with cfw.activate(), _build_calc_context().activate():
                _load(_FreshNoExtender, "missing", FeatureSet())

        assert not [r for r in caplog.records if "data_access_identity" in r.getMessage()]


class TestInputDataLoadHookCarriesReaderClassAndDeclaredAttributes:
    """INPUT_DATA_LOAD carries the concrete reader class and the READER's declared attributes."""

    def test_direct_load_carries_reader_class_and_scalar_declarations(self) -> None:
        context = _direct_load_context(_DeclaringReader)

        assert cast(Any, context.reader_class) is _DeclaringReader
        assert context.declared_attributes == {"unit": "mm", "scale": 3}

    def test_reader_without_declarations_surfaces_empty_mapping(self) -> None:
        context = _direct_load_context(_DirectLoader)

        assert cast(Any, context.reader_class) is _DirectLoader
        assert context.declared_attributes == {}

    def test_raising_reader_declaration_degrades_to_none_and_load_succeeds(self) -> None:
        context = _direct_load_context(_RaisingDeclarationReader)

        assert cast(Any, context.reader_class) is _RaisingDeclarationReader
        assert context.declared_attributes is None

    def test_end_to_end_source_group_declarations_reach_both_load_and_calculate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        column = f"{_MARKER}_col_decl"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [1, 2])
        monkeypatch.setattr(CsvFG, "declared_attributes", classmethod(lambda cls, features: {"origin": "format"}))
        calc_extender = _CalcContextCapturingExtender()
        fetch_extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            [column],
            compute_frameworks=[PythonDictFramework],
            data_access_collection=DataAccessCollection(files={str(path)}),
            function_extender={calc_extender, fetch_extender},
        )

        assert fetch_extender.captured is not None
        assert fetch_extender.captured.reader_class is CsvFG
        assert fetch_extender.captured.declared_attributes == {"origin": "format"}
        assert calc_extender.captured is not None
        assert calc_extender.captured.reader_class is None
        assert calc_extender.captured.declared_attributes == {"origin": "format"}


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
        features = FeatureSet()

        with cfw.activate(), _build_calc_context().activate():
            result = _load(_DirectLoader, {"any": "value"}, features)

        assert result == [1, 2, 3]
        assert extender.captured is not None
        assert extender.captured.rows_out == _ROW_COUNT_SENTINEL
        assert extender.captured.rows_out != len(result)


_REPLACED_COLUMN = f"{_MARKER}_replaced_col"


class _ReplacedReadParentFeatureGroup(CsvFG):
    """CsvFG subclass replaced by its own subclass; gated to its own column so other tests are unaffected."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == _REPLACED_COLUMN and super().match_feature_group_criteria(
            feature_name, options, data_access_collection
        )


class _ReplacingReadChildFeatureGroup(_ReplacedReadParentFeatureGroup):
    """Winning subclass."""


class TestInputDataLoadCarriesSpecializedFrom:
    def test_load_context_matches_the_calculate_context(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        _write_csv(path, _REPLACED_COLUMN, [1, 2, 3])

        calc_extender = _CalcContextCapturingExtender()
        fetch_extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            [_REPLACED_COLUMN],
            compute_frameworks=[PythonDictFramework],
            data_access_collection=DataAccessCollection(files={str(path)}),
            plugin_collector=PluginCollector.enabled_feature_groups(
                {_ReplacedReadParentFeatureGroup, _ReplacingReadChildFeatureGroup}
            ),
            function_extender={calc_extender, fetch_extender},
        )

        parent_name = f"{_ReplacedReadParentFeatureGroup.__module__}.{_ReplacedReadParentFeatureGroup.__qualname__}"
        assert calc_extender.captured is not None
        assert fetch_extender.captured is not None
        assert calc_extender.captured.specialized_from == (parent_name,)
        assert fetch_extender.captured.specialized_from == calc_extender.captured.specialized_from


_MP_HOP_COLUMN = f"{_MARKER}_mp_hop_col"


class _FileRecordingGateExtender(Extender):
    """Gate that appends one line per call to a file, so a spawned worker's calls are visible to the parent."""

    never_fall_back = True

    def __init__(self, output_path: Path, record_identity: bool = False) -> None:
        self.priority = 100
        self._output_path = output_path
        self._record_identity = record_identity

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._record_identity:
            with self._output_path.open("a", encoding="utf-8") as handle:
                handle.write("gate\n")
            return func(*args, **kwargs)
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None and context.reader_class is not None
        line = (
            f"{context.data_access_identity}|{context.data_access_identity_is_fallback}|{context.reader_class.__name__}"
        )
        with self._output_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return result


class _MultiprocessingHopFeatureGroup(GeneratedFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_MP_HOP_COLUMN}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            loaded = pool.submit(lambda: _load(_DirectLoader, "access", features))
            return {_MP_HOP_COLUMN: loaded.result()}


@pytest.mark.timeout(30)
class TestThreadHopInASpawnedWorkerReachesTheGate:
    def test_hopped_load_in_a_multiprocessing_worker_reaches_the_gate(self, tmp_path: Path, flight_server: Any) -> None:
        output_path = tmp_path / "gate_calls.txt"

        session = mloda.prepare(
            [Feature(name=_MP_HOP_COLUMN)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups({_MultiprocessingHopFeatureGroup}),
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            function_extender={_FileRecordingGateExtender(output_path)},
        )

        result = session.run(
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result[0][_MP_HOP_COLUMN] == [1, 2, 3] or list(result[0][_MP_HOP_COLUMN]) == [[1, 2, 3]]
        assert output_path.read_text(encoding="utf-8").splitlines() == ["gate"]

    def test_api_data_load_in_a_multiprocessing_worker_is_observed(self, tmp_path: Path, flight_server: Any) -> None:
        output_path = tmp_path / "identity_calls.txt"

        result = mloda.run_all(
            [_API_COLUMN],
            compute_frameworks=["PythonDictFramework"],
            api_data={"mp": {_API_COLUMN: [1, 2]}},
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
            function_extender={_FileRecordingGateExtender(output_path, record_identity=True)},
        )

        assert output_path.read_text(encoding="utf-8").splitlines() == ["api_data:mp|False|ApiDataFG"]
        assert len(result) == 1


class TestInputDataLoadAuditForSourceGroups:
    """A source group fires INPUT_DATA_LOAD naming itself, its source and the loader that ran."""

    def _fire(
        self, group: type, framework: Any, extender: _InputDataLoadCapturingExtender | None = None
    ) -> HookContext:
        extender = extender or _InputDataLoadCapturingExtender()
        mloda.run_all(
            ["toyfmt_audit"],
            compute_frameworks=[framework],
            plugin_collector=PluginCollector.enabled_feature_groups({group}),
            data_access_collection=toy_dac(h1={"toyfmt_audit": [1, 2]}),
            function_extender={extender},
        )
        assert extender.captured is not None
        return extender.captured

    def test_neutral_load_is_audited(self, tmp_path: Path) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")

        context = self._fire(group, PythonDictFramework)

        assert context.data_access_format == group.get_class_name()
        assert context.reader_class is group
        assert context.data_access_identity == "h1:toy"
        assert context.data_access_loader == "neutral"
        assert context.implementation == "neutral"
        assert context.implementation_version is None

    def test_implementation_load_names_the_implementation_class(self, tmp_path: Path, make_implementation: Any) -> None:
        group = neutral_csv_group(tmp_path / "n.csv")
        impl = make_implementation(group, PythonDictFramework, lambda s, m, f: [{"toyfmt_audit": 1}])
        assert impl in loaded_implementations()

        extender = _InputDataLoadCapturingExtender()
        context = self._fire(group, PythonDictFramework, extender)

        assert context.reader_class is group
        assert context.data_access_identity == "h1:toy"
        assert context.data_access_loader == f"{impl.__module__}:{impl.__qualname__}"
        assert context.implementation == context.data_access_loader
        assert extender.func_owner == group.__name__

    def test_readers_leave_the_loader_field_none(self) -> None:
        context = _direct_load_context(_DirectLoader)
        assert context.data_access_loader is None
        assert context.implementation is None
        assert context.implementation_version is None

    def test_stock_csv_group_audits_itself_with_the_neutral_loader(self, tmp_path: Path) -> None:
        column = f"{_MARKER}_col_csv_audit"
        path = tmp_path / "data.csv"
        _write_csv(path, column, [1])
        extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            [column],
            compute_frameworks=[PythonDictFramework],
            data_access_collection=DataAccessCollection(files={str(path)}),
            function_extender={extender},
        )

        context = extender.captured
        assert context is not None
        assert context.reader_class is CsvFG
        assert context.feature_group_class == _EXPECTED_FEATURE_GROUP_CLASS
        assert context.data_access_format == "CsvFG"
        assert context.data_access_identity == os.path.abspath(path)
        assert context.data_access_loader == "neutral"
        assert context.implementation == "neutral"
        assert context.implementation_version is None

    def test_api_data_load_is_audited_with_the_set_key_identity(self) -> None:
        extender = _InputDataLoadCapturingExtender()

        mloda.run_all(
            [_API_COLUMN],
            compute_frameworks=[PythonDictFramework],
            api_data={"audited": {_API_COLUMN: [1, 2]}},
            function_extender={extender},
        )

        context = extender.captured
        assert context is not None
        assert context.data_access_identity == "api_data:audited"
        assert context.data_access_identity_is_fallback is False
        assert context.reader_class is ApiDataFG
