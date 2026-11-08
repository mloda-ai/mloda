"""Contract tests for the shipped ``RunAllSuite`` policy driver.

Pins the public API of ``assert_policy_case`` directly, independently of the empty-result suite:
the success and raises expectations, in SYNC and in a worker-based (THREADING) mode.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any

import pytest

from mloda._testing import framework_adapter as framework_adapter_module
from mloda._testing.compute_framework.run_all_suite import PolicyRaises, PolicySuccess, RunAllSuite, run_all_modes
from mloda._testing.framework_adapter import (
    DuckDBAdapter,
    FrameworkAdapter,
    IcebergAdapter,
    PandasAdapter,
    PolarsAdapter,
    PolarsLazyAdapter,
    PyArrowAdapter,
    PythonDictAdapter,
    SparkAdapter,
    SqliteAdapter,
)
from mloda._testing.source import reclaim
from mloda._testing.suite import SwitchedOffError
from mloda.core.abstract_plugins.compute_framework import EmptyResultError
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
from mloda.provider import DataCreatorFeatureGroup, FeatureSet
from mloda.user import ParallelizationMode, PluginCollector

pytest_plugins = ["pytester"]


class _PythonDictRunAllHost(RunAllSuite):
    """A throwaway host (not collected: no Test prefix); the tests drive its assert_policy_case directly."""

    adapter_class = PythonDictAdapter


def _allowed_collector() -> PluginCollector:
    class _AllowedFeatureGroup(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"empty_result_allowed_col"}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {"empty_result_allowed_col": []}

    return PluginCollector.enabled_feature_groups({reclaim.track(_AllowedFeatureGroup)})


def _schemaless_collector() -> PluginCollector:
    class _SchemalessFeatureGroup(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"empty_result_schemaless_col"}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {}

    return PluginCollector.enabled_feature_groups({reclaim.track(_SchemalessFeatureGroup)})


@pytest.fixture(autouse=True)
def _reclaim_built_classes() -> Iterator[None]:
    yield from reclaim.reclaimed_scope()


@pytest.fixture
def adapter() -> Iterator[FrameworkAdapter]:
    python_dict_adapter = PythonDictAdapter()
    with python_dict_adapter:
        yield python_dict_adapter


def _single_empty(adapter: FrameworkAdapter) -> PolicySuccess:
    def check(result: Any) -> None:
        assert len(result) == 1
        assert adapter.row_count(result[0]) == 0

    return PolicySuccess(assert_result=check)


def test_policy_base_drives_success_case(adapter: FrameworkAdapter, flight_server: Any) -> None:
    """A policy success expectation returns a schema-bearing zero-row result."""
    result = _PythonDictRunAllHost().assert_policy_case(
        adapter,
        feature_name="empty_result_allowed_col",
        plugin_collector=_allowed_collector(),
        expectation=_single_empty(adapter),
        mode=ParallelizationMode.SYNC,
        flight_server=flight_server,
    )

    assert not isinstance(result, BaseException)
    assert len(result) == 1


def test_policy_base_drives_raises_case(adapter: FrameworkAdapter, flight_server: Any) -> None:
    """A policy raises expectation: a zero-column ``{}`` result is schema-less and raises."""
    result = _PythonDictRunAllHost().assert_policy_case(
        adapter,
        feature_name="empty_result_schemaless_col",
        plugin_collector=_schemaless_collector(),
        expectation=PolicyRaises(match_substring="Result carries no schema"),
        mode=ParallelizationMode.SYNC,
        flight_server=flight_server,
    )

    assert isinstance(result, BaseException)


def test_policy_base_drives_worker_mode(adapter: FrameworkAdapter, flight_server: Any) -> None:
    """The same success case under THREADING proves the worker-based path is covered."""
    result = _PythonDictRunAllHost().assert_policy_case(
        adapter,
        feature_name="empty_result_allowed_col",
        plugin_collector=_allowed_collector(),
        expectation=_single_empty(adapter),
        mode=ParallelizationMode.THREADING,
        flight_server=flight_server,
    )

    assert not isinstance(result, BaseException)
    assert len(result) == 1


def _assert_schemaless(adapter: FrameworkAdapter, flight_server: Any, **expectation_fields: Any) -> None:
    _PythonDictRunAllHost().assert_policy_case(
        adapter,
        feature_name="empty_result_schemaless_col",
        plugin_collector=_schemaless_collector(),
        expectation=PolicyRaises(match_substring="Result carries no schema", **expectation_fields),
        mode=ParallelizationMode.SYNC,
        flight_server=flight_server,
    )


def test_policy_raises_checks_error_type_and_also_contains(adapter: FrameworkAdapter, flight_server: Any) -> None:
    _assert_schemaless(adapter, flight_server, error_type=EmptyResultError, also_contains=("schema",))


def test_policy_raises_wrong_error_type_fails(adapter: FrameworkAdapter, flight_server: Any) -> None:
    with pytest.raises(AssertionError):
        _assert_schemaless(adapter, flight_server, error_type=KeyError)


def test_policy_raises_wrong_cause_type_fails(adapter: FrameworkAdapter, flight_server: Any) -> None:
    with pytest.raises(AssertionError):
        _assert_schemaless(adapter, flight_server, cause_type=KeyError)


def test_policy_raises_missing_also_contains_fails(adapter: FrameworkAdapter, flight_server: Any) -> None:
    with pytest.raises(AssertionError):
        _assert_schemaless(adapter, flight_server, also_contains=("no such text",))


CONFTEST_FREE_HOST = """
from mloda._testing.compute_framework.run_all_suite import EmptyResultRunAllSuite
from mloda._testing.framework_adapter import PythonDictAdapter


class TestConftestFreeHost(EmptyResultRunAllSuite):
    adapter_class = PythonDictAdapter
"""


_ADAPTER_CLASSES = [
    PythonDictAdapter,
    PyArrowAdapter,
    PandasAdapter,
    PolarsAdapter,
    PolarsLazyAdapter,
    DuckDBAdapter,
    SqliteAdapter,
    IcebergAdapter,
    SparkAdapter,
]


@pytest.mark.parametrize("adapter_class", _ADAPTER_CLASSES, ids=lambda cls: cls.__name__)
def test_run_all_modes_follow_the_framework(adapter_class: type[FrameworkAdapter]) -> None:
    sync_only = adapter_class in (DuckDBAdapter, SqliteAdapter)
    expected = [ParallelizationMode.SYNC] if sync_only else [ParallelizationMode.SYNC, ParallelizationMode.THREADING]

    assert run_all_modes(adapter_class) == expected


@pytest.mark.parametrize("reason", [None, ""])
def test_run_all_modes_rejects_undeclared_sync_only(reason: str | None) -> None:
    undeclared = type("_UndeclaredDuckDB", (DuckDBAdapter,), {"sync_only_reason": reason})

    with pytest.raises(SwitchedOffError, match="THREADING") as raised:
        run_all_modes(undeclared)
    assert "_UndeclaredDuckDB" in str(raised.value)


def test_run_all_modes_rejects_a_framework_without_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PolarsDataFrame, "supported_parallelization_modes", classmethod(lambda cls: {ParallelizationMode.THREADING})
    )

    with pytest.raises(SwitchedOffError, match="SYNC"):
        run_all_modes(PolarsAdapter)


def test_adapter_classes_list_covers_every_concrete_adapter() -> None:
    concrete = {
        obj
        for obj in vars(framework_adapter_module).values()
        if inspect.isclass(obj)
        and obj.__module__ == framework_adapter_module.__name__
        and issubclass(obj, FrameworkAdapter)
        and not inspect.isabstract(obj)
    }

    assert set(_ADAPTER_CLASSES) == concrete


def test_run_all_modes_rejects_declared_sync_only_that_serves_threading() -> None:
    class _DeclaredPolars(PolarsAdapter):
        sync_only_reason = "x"

    with pytest.raises(SwitchedOffError, match="sync_only_reason") as raised:
        run_all_modes(_DeclaredPolars)
    assert "_DeclaredPolars" in str(raised.value)


MISDECLARED_HOST = """
from mloda._testing.compute_framework.run_all_suite import EmptyResultRunAllSuite
from mloda._testing.framework_adapter import DuckDBAdapter


class _UndeclaredDuckDB(DuckDBAdapter):
    sync_only_reason = None


class TestMisdeclaredHost(EmptyResultRunAllSuite):
    adapter_class = _UndeclaredDuckDB
"""


def test_a_misdeclared_host_fails_at_collection(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_inner=MISDECLARED_HOST)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:randomly")

    outcomes = result.parseoutcomes()
    assert outcomes.get("errors", 0) + outcomes.get("error", 0) >= 1
    assert "passed" not in outcomes
    result.stdout.fnmatch_lines(["*THREADING*"])


def test_a_host_without_a_conftest_uses_the_suites_own_flight_server(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_inner=CONFTEST_FREE_HOST)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:randomly")

    outcomes = result.parseoutcomes()
    assert outcomes.get("passed", 0) > 0
    assert not {"failed", "error", "errors"} & outcomes.keys()
