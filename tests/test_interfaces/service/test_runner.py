"""Runner.run: execute a verified plan id under the execute mode, re-checking plan, limits and policy at run time.

Results carry per-table classification, a bounded untrusted preview of allowed columns only, and no paths.
"""

import json
import logging
import math
import re
import threading
from collections.abc import Iterator, Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.input_data.api_data_fg import ApiDataFG
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.interfaces.service.catalog import Catalog
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.errors import ERROR_CODES, is_error
from mloda.interfaces.service.lock import MLODA_LOCK
from mloda.interfaces.service.planner import Planner
from mloda.interfaces.service.runner import Runner
from mloda.provider import FeatureSet
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.helpers.plugin_stubs import StubFeatureGroup
from tests.test_interfaces.source_groups import (
    FILE_SOURCE_GROUPS,
    PLAN_RAISE_SECRET,
    RAISE,
    RAISE_SECRET,
    RunnerRaiseMcp5,
)

# Running resolves real FeatureGroups and reads source files; a cold cache under xdist load can exceed 10s.
pytestmark = pytest.mark.timeout(30)

CLOCK_START = 1_800_000_000.0
KEY = b"r" * 32
DENIED = "RunnerDeniedMcp5"
PROBE = "income__mcp5_probe"
STALE = "income__mcp5_stale"
TYPED_PREFIX = "income__mcp5_"
TYPED_KINDS = ("decimal", "datetime", "date", "nan")
TYPED = [TYPED_PREFIX + kind for kind in TYPED_KINDS]
FEATURES = ["customer_id", "income", "segment"]
REQUEST: dict[str, Any] = {"features": FEATURES}
CLASSIFICATION = {"customer_id": "personal", "income": "internal", "segment": "public"}
CUSTOMER_IDS = [f"C9X7{index:03d}" for index in range(1, 21)]
RUN_KEYS = {"status", "tables", "audit_id"}
TABLE_KEYS = {"row_count", "columns", "preview"}
AUDIT_ID = re.compile(r"^run_[0-9a-f]{16,}$")
PROBE_CALLS: list[str] = []
LOCK_PROBES: list[bool] = []
DENIED_CALLS: list[str] = []


def _lock_free_elsewhere() -> bool:
    outcome: list[bool] = []

    def attempt() -> None:
        acquired = MLODA_LOCK.acquire(blocking=False)
        if acquired:
            MLODA_LOCK.release()
        outcome.append(acquired)

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join(5)
    return outcome == [True]


class _SuffixFeatureGroup(FeatureGroup):
    """Derives each requested feature from ``income``; subclasses set the suffixes they serve."""

    SUFFIXES: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        name = str(feature_name)
        return bool(cls.SUFFIXES) and any(name == "income" + suffix for suffix in cls.SUFFIXES)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("income")}


class RunnerProbeMcp5(_SuffixFeatureGroup):
    """Doubles income and records that it ran, optionally probing the service lock."""

    SUFFIXES: ClassVar[tuple[str, ...]] = ("__mcp5_probe",)
    ARMED: ClassVar[bool] = False
    PROBE_LOCK: ClassVar[bool] = False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if cls.ARMED:
            PROBE_CALLS.append("calculate_feature")
        if cls.PROBE_LOCK:
            LOCK_PROBES.append(_lock_free_elsewhere())
        return {PROBE: [int(value) * 2 for value in data["income"]]}


def _stale_column(name: str, data: Any) -> dict[str, list[int]]:
    if RunnerProbeMcp5.ARMED:
        PROBE_CALLS.append(name)
    return {STALE: [int(value) for value in data["income"]]}


class RunnerStaleAMcp5(_SuffixFeatureGroup):
    """One of two interchangeable producers of the stale feature."""

    SUFFIXES: ClassVar[tuple[str, ...]] = ("__mcp5_stale",)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _stale_column(cls.__name__, data)


class RunnerStaleBMcp5(_SuffixFeatureGroup):
    """The other producer of the stale feature."""

    SUFFIXES: ClassVar[tuple[str, ...]] = ("__mcp5_stale",)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _stale_column(cls.__name__, data)


class RunnerTypesMcp5(_SuffixFeatureGroup):
    """Produces Decimal, datetime, date and NaN-bearing float columns from income."""

    SUFFIXES: ClassVar[tuple[str, ...]] = tuple("__mcp5_" + kind for kind in TYPED_KINDS)

    @staticmethod
    def _value(kind: str, income: int) -> Any:
        if kind == "decimal":
            return Decimal(income) / Decimal(4)
        if kind == "datetime":
            return datetime(2024, 1, 1, 12, 30) + timedelta(days=income)
        if kind == "date":
            return date(2024, 1, 1) + timedelta(days=income)
        return float("nan") if income >= 30 else income + 0.5

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        incomes = [int(value) for value in data["income"]]
        columns: dict[str, list[Any]] = {}
        for feature_name in features.get_all_names():
            name = str(feature_name)
            kind = name[len(TYPED_PREFIX) :]
            columns[name] = [cls._value(kind, income) for income in incomes]
        return columns


class RunnerDeniedMcp5(StubFeatureGroup):
    """Restricted payroll data."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"mcp5_salary"})
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        DENIED_CALLS.append("calculate_feature")
        return {"mcp5_salary": [1]}


class _Clock:
    def __init__(self, now: float = CLOCK_START) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _base(*extra: type[FeatureGroup]) -> PluginCollector:
    groups: set[type[FeatureGroup]] = {
        *FILE_SOURCE_GROUPS,
        RunnerProbeMcp5,
        RunnerTypesMcp5,
        RunnerDeniedMcp5,
        RunnerRaiseMcp5,
        ApiDataFG,
    }
    return PluginCollector.enabled_feature_groups(groups | set(extra))


def _write_sales(path: Path, rows: int = 5) -> Path:
    lines = ["customer_id,income,segment"]
    lines += [f"{CUSTOMER_IDS[i - 1]},{10 * i},{'a' if i % 2 else 'b'}" for i in range(1, rows + 1)]
    path.write_text("\n".join(lines) + "\n")
    return path


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    path = _write_sales(tmp_path / "sales.csv")
    sources = {"sales": SourceConfig(alias="sales", files=(str(path),))}
    values: dict[str, Any] = {"mode": "execute", "sources": sources, "classification": CLASSIFICATION}
    values.update(kwargs)
    return Settings(**values)


def _planner(
    settings: Settings,
    base: PluginCollector | None = None,
    *,
    catalog_settings: Settings | None = None,
    clock: _Clock | None = None,
    key: bytes = KEY,
) -> Planner:
    collector = base if base is not None else _base()
    catalog = Catalog.build(catalog_settings if catalog_settings is not None else settings, collector)
    return Planner(
        settings, catalog, signing_key=key, principal="alice", plugin_collector=collector, clock=clock or _Clock()
    )


def _plan_id(planner: Planner, request: Mapping[str, Any] = REQUEST) -> str:
    result = planner.explain(dict(request))
    assert not is_error(result), result
    plan_id: str = result["plan_id"]
    return plan_id


def _run(tmp_path: Path, request: Mapping[str, Any] = REQUEST, **kwargs: Any) -> dict[str, Any]:
    planner = _planner(_settings(tmp_path, **kwargs))
    result: dict[str, Any] = Runner(planner).run(_plan_id(planner, request))
    return result


def _error(result: Mapping[str, Any], code: str) -> dict[str, Any]:
    assert is_error(result), result
    error: dict[str, Any] = result["error"]
    assert error["code"] == code, error
    assert error["audit_id"].startswith("aud_")
    json.dumps(result)
    return error


def _success(result: Mapping[str, Any]) -> dict[str, Any]:
    assert not is_error(result), result
    return dict(result)


def _only_table(result: Mapping[str, Any]) -> dict[str, Any]:
    tables = _success(result)["tables"]
    assert isinstance(tables, list)
    assert len(tables) == 1, tables
    table: dict[str, Any] = tables[0]
    assert set(table) == TABLE_KEYS
    return table


def _has_detail(error: Mapping[str, Any], **expected: Any) -> bool:
    return any(all(detail.get(key) == value for key, value in expected.items()) for detail in error["details"])


def _assert_no_paths(result: Mapping[str, Any], tmp_path: Path) -> None:
    dumped = json.dumps(result)
    assert str(tmp_path) not in dumped
    assert tmp_path.name not in dumped
    assert "sales.csv" not in dumped


@pytest.fixture
def probe_calls() -> Iterator[list[str]]:
    PROBE_CALLS.clear()
    RunnerProbeMcp5.ARMED = True
    yield PROBE_CALLS
    RunnerProbeMcp5.ARMED = False
    PROBE_CALLS.clear()


@pytest.fixture
def lock_probes() -> Iterator[list[bool]]:
    LOCK_PROBES.clear()
    RunnerProbeMcp5.PROBE_LOCK = True
    yield LOCK_PROBES
    RunnerProbeMcp5.PROBE_LOCK = False
    LOCK_PROBES.clear()


@pytest.fixture
def denied_calls() -> Iterator[list[str]]:
    DENIED_CALLS.clear()
    yield DENIED_CALLS
    DENIED_CALLS.clear()


class TestMode:
    def test_plan_mode_denies_execution_without_running(self, tmp_path: Path, probe_calls: list[str]) -> None:
        planner = _planner(_settings(tmp_path, mode="plan"))
        plan_id = _plan_id(planner, {"features": ["customer_id", PROBE]})

        result = Runner(planner).run(plan_id)

        error = _error(result, "policy_denied")
        assert "execut" in error["message"].lower()
        assert "disabled" in error["message"].lower()
        assert probe_calls == []
        dumped = json.dumps(result)
        assert all(customer_id not in dumped for customer_id in CUSTOMER_IDS)
        _assert_no_paths(result, tmp_path)

    @pytest.mark.parametrize("plan_id", ["garbage", None, 42, ""])
    def test_mode_is_checked_before_the_plan_id(self, tmp_path: Path, plan_id: Any) -> None:
        planner = _planner(_settings(tmp_path, mode="plan"))
        _error(Runner(planner).run(plan_id), "policy_denied")


class TestPlanId:
    @pytest.mark.parametrize("plan_id", ["garbage", None, 42, "", "mpl1.x.y", ["mpl1"]])
    def test_malformed_plan_id_is_plan_invalid(self, tmp_path: Path, plan_id: Any) -> None:
        planner = _planner(_settings(tmp_path))
        result = Runner(planner).run(plan_id)
        _error(result, "plan_invalid")
        _assert_no_paths(result, tmp_path)

    def test_tampered_plan_id_is_plan_invalid(self, tmp_path: Path, probe_calls: list[str]) -> None:
        planner = _planner(_settings(tmp_path))
        plan_id = _plan_id(planner, {"features": [PROBE]})
        head, body, signature = plan_id.split(".")
        tampered = ".".join([head, body, ("A" if signature[0] != "A" else "B") + signature[1:]])

        _error(Runner(planner).run(tampered), "plan_invalid")
        assert probe_calls == []

    def test_plan_id_from_another_key_is_plan_invalid(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        foreign = _plan_id(_planner(settings, key=b"x" * 32))

        _error(Runner(_planner(settings)).run(foreign), "plan_invalid")

    def test_expired_plan_id_is_plan_invalid(self, tmp_path: Path, probe_calls: list[str]) -> None:
        settings = _settings(tmp_path)
        clock = _Clock()
        planner = _planner(settings, clock=clock)
        plan_id = _plan_id(planner, {"features": [PROBE]})
        clock.now = CLOCK_START + settings.plan_ttl_seconds + 1

        error = _error(Runner(planner).run(plan_id), "plan_invalid")
        assert "expired" in error["message"]
        assert probe_calls == []

    def test_plan_id_runs_on_an_equivalent_planner(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        plan_id = _plan_id(_planner(settings))

        result = Runner(_planner(settings)).run(plan_id)

        assert _only_table(result)["row_count"] == 5


class TestStale:
    def test_changed_resolution_is_plan_stale_without_running(self, tmp_path: Path, probe_calls: list[str]) -> None:
        settings = _settings(tmp_path)
        explained = _planner(settings, _base(RunnerStaleAMcp5))
        plan_id = _plan_id(explained, {"features": [STALE]})

        result = Runner(_planner(settings, _base(RunnerStaleBMcp5))).run(plan_id)

        error = _error(result, "plan_stale")
        assert "explain" in error["message"].lower()
        assert probe_calls == []
        _assert_no_paths(result, tmp_path)

    def test_unchanged_resolution_runs(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        plan_id = _plan_id(_planner(settings, _base(RunnerStaleAMcp5)), {"features": [STALE]})

        result = Runner(_planner(settings, _base(RunnerStaleAMcp5))).run(plan_id)

        table = _only_table(result)
        assert table["row_count"] == 5
        assert [row[STALE] for row in table["preview"]["rows"]] == [10, 20, 30, 40, 50]


class TestRunTimeChecks:
    def test_rows_are_recounted_at_run_time(self, tmp_path: Path, probe_calls: list[str]) -> None:
        settings = _settings(tmp_path, max_rows=10)
        planner = _planner(settings)
        plan_id = _plan_id(planner, {"features": ["customer_id", PROBE]})
        _write_sales(tmp_path / "sales.csv", rows=20)

        result = Runner(planner).run(plan_id)

        error = _error(result, "limit_exceeded")
        assert _has_detail(error, source="sales", rows=20, max_rows=10)
        assert probe_calls == []
        dumped = json.dumps(result)
        assert all(customer_id not in dumped for customer_id in CUSTOMER_IDS)
        _assert_no_paths(result, tmp_path)

    def test_rows_within_max_rows_run(self, tmp_path: Path) -> None:
        assert _only_table(_run(tmp_path, max_rows=10))["row_count"] == 5

    def test_policy_is_rechecked_at_run_time(self, tmp_path: Path, denied_calls: list[str]) -> None:
        settings = _settings(tmp_path)
        plan_id = _plan_id(_planner(settings), {"features": ["mcp5_salary"]})
        denying = _planner(settings, catalog_settings=Settings(deny_feature_groups=(DENIED,)))

        result = Runner(denying).run(plan_id)

        _error(result, "policy_denied")
        assert denied_calls == []


class TestFeatureGroupFailure:
    def test_exception_becomes_an_internal_error_without_its_text(self, tmp_path: Path) -> None:
        result = _run(tmp_path, {"features": ["customer_id", RAISE]})

        error = _error(result, "internal")
        assert "ValueError" in error["message"]
        assert RAISE_SECRET not in json.dumps(result)

    def test_failure_is_logged_once_with_audit_id_and_traceback_without_the_message(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="mloda.interfaces.service"):
            result = _run(tmp_path, {"features": ["customer_id", RAISE]})

        audit_id = _error(result, "internal")["audit_id"]
        records = [r for r in caplog.records if r.name == "mloda.interfaces.service" and r.levelno == logging.ERROR]
        assert len(records) == 1
        message = records[0].getMessage()
        assert audit_id in message
        assert "ValueError" in message
        assert "Traceback" in message
        assert RAISE_SECRET not in message
        assert records[0].exc_info is None

    def test_planning_failure_during_run_is_an_internal_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        planner = _planner(_settings(tmp_path))
        plan_id = _plan_id(planner)

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise ValueError(PLAN_RAISE_SECRET)

        monkeypatch.setattr("mloda.interfaces.service.planner.mloda.diagnose", boom)
        result = Runner(planner).run(plan_id)

        error = _error(result, "internal")
        assert "ValueError" in error["message"]
        assert PLAN_RAISE_SECRET not in json.dumps(result)


class TestSuccessShape:
    def test_result_keys_status_and_audit_id(self, tmp_path: Path) -> None:
        result = _success(_run(tmp_path))

        assert set(result) == RUN_KEYS
        assert result["status"] == "completed"
        assert AUDIT_ID.fullmatch(result["audit_id"]), result["audit_id"]
        assert isinstance(result["tables"], list)

    def test_each_run_has_its_own_audit_id(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path))
        plan_id = _plan_id(planner)
        runner = Runner(planner)

        first = _success(runner.run(plan_id))
        second = _success(runner.run(plan_id))

        assert first["audit_id"] != second["audit_id"]

    def test_one_table_with_row_count(self, tmp_path: Path) -> None:
        assert _only_table(_run(tmp_path))["row_count"] == 5

    def test_columns_carry_classification_and_preview_flag(self, tmp_path: Path) -> None:
        table = _only_table(_run(tmp_path))

        assert table["columns"] == [
            {"name": "customer_id", "classification": "personal", "in_preview": False},
            {"name": "income", "classification": "internal", "in_preview": True},
            {"name": "segment", "classification": "public", "in_preview": True},
        ]

    def test_columns_follow_request_order(self, tmp_path: Path) -> None:
        table = _only_table(_run(tmp_path, {"features": ["segment", "customer_id", "income"]}))
        assert [column["name"] for column in table["columns"]] == ["segment", "customer_id", "income"]

    def test_unclassified_column_is_not_previewed(self, tmp_path: Path) -> None:
        table = _only_table(_run(tmp_path, classification={"customer_id": "personal", "income": "internal"}))

        assert table["columns"][2] == {"name": "segment", "classification": "unclassified", "in_preview": False}
        assert all(set(row) == {"income"} for row in table["preview"]["rows"])

    def test_preview_is_untrusted_and_holds_allowed_columns_only(self, tmp_path: Path) -> None:
        preview = _only_table(_run(tmp_path))["preview"]

        assert set(preview) == {"untrusted", "truncated", "rows"}
        assert preview["untrusted"] is True
        assert preview["rows"] == [
            {"income": 10, "segment": "a"},
            {"income": 20, "segment": "b"},
            {"income": 30, "segment": "a"},
            {"income": 40, "segment": "b"},
            {"income": 50, "segment": "a"},
        ]

    def test_withheld_column_values_never_appear(self, tmp_path: Path) -> None:
        result = _run(tmp_path)

        dumped = json.dumps(_success(result))
        for customer_id in CUSTOMER_IDS[:5]:
            assert customer_id not in dumped

    def test_no_paths_anywhere(self, tmp_path: Path) -> None:
        _assert_no_paths(_success(_run(tmp_path)), tmp_path)

    def test_keep_index_returns_withheld_key_column(self, tmp_path: Path) -> None:
        request = {"features": [{"name": "income", "keep_index": True, "index": ["customer_id"]}]}

        result = _run(tmp_path, request)

        table = _only_table(result)
        assert {"name": "customer_id", "classification": "personal", "in_preview": False} in table["columns"]
        assert {"name": "income", "classification": "internal", "in_preview": True} in table["columns"]
        assert all(set(row) == {"income"} for row in table["preview"]["rows"])
        assert all(customer_id not in json.dumps(result) for customer_id in CUSTOMER_IDS[:5])

    def test_keep_index_key_column_is_previewed_when_allowed(self, tmp_path: Path) -> None:
        request = {"features": [{"name": "income", "keep_index": True, "index": ["customer_id"]}]}

        table = _only_table(_run(tmp_path, request, classification={**CLASSIFICATION, "customer_id": "internal"}))

        assert {"name": "customer_id", "classification": "internal", "in_preview": True} in table["columns"]
        assert all(set(row) == {"customer_id", "income"} for row in table["preview"]["rows"])

    def test_key_column_also_requested_appears_once(self, tmp_path: Path) -> None:
        request = {"features": ["customer_id", {"name": "income", "keep_index": True, "index": ["customer_id"]}]}

        table = _only_table(_run(tmp_path, request))

        assert [column["name"] for column in table["columns"]].count("customer_id") == 1


class TestInlineData:
    REQUEST: ClassVar[dict[str, Any]] = {"features": ["token"], "data": {"Tokens": {"token": ["007", "", None]}}}

    def test_classified_inline_column_round_trips_exact_values(self, tmp_path: Path) -> None:
        classification = {**CLASSIFICATION, "token": "public"}  # nosec B105

        table = _only_table(_run(tmp_path, self.REQUEST, inline_data=True, classification=classification))

        assert table["row_count"] == 3
        assert table["columns"] == [{"name": "token", "classification": "public", "in_preview": True}]
        assert [row["token"] for row in table["preview"]["rows"]] == ["007", "", None]

    def test_unclassified_inline_column_is_withheld(self, tmp_path: Path) -> None:
        result = _run(tmp_path, self.REQUEST, inline_data=True)

        table = _only_table(result)
        assert table["columns"] == [{"name": "token", "classification": "unclassified", "in_preview": False}]
        assert all("token" not in row for row in table["preview"]["rows"])
        assert '"007"' not in json.dumps(result)


class TestPreviewBounds:
    def test_preview_is_truncated_to_max_preview_rows(self, tmp_path: Path) -> None:
        table = _only_table(_run(tmp_path, max_preview_rows=2))

        assert table["row_count"] == 5
        assert table["preview"]["truncated"] is True
        assert table["preview"]["rows"] == [{"income": 10, "segment": "a"}, {"income": 20, "segment": "b"}]

    def test_preview_of_all_rows_is_not_truncated(self, tmp_path: Path) -> None:
        preview = _only_table(_run(tmp_path, max_preview_rows=20))["preview"]

        assert preview["truncated"] is False
        assert len(preview["rows"]) == 5

    def test_preview_exactly_at_the_bound_is_not_truncated(self, tmp_path: Path) -> None:
        preview = _only_table(_run(tmp_path, max_preview_rows=5))["preview"]

        assert preview["truncated"] is False
        assert len(preview["rows"]) == 5

    def test_all_withheld_table_has_no_preview_rows(self, tmp_path: Path) -> None:
        table = _only_table(_run(tmp_path, {"features": ["customer_id"]}))

        assert table["row_count"] == 5
        assert table["columns"] == [{"name": "customer_id", "classification": "personal", "in_preview": False}]
        assert table["preview"]["rows"] == []
        assert table["preview"]["truncated"] is True


class TestOtherFrameworks:
    @pytest.mark.parametrize(
        ("framework", "backend"),
        [("PandasDataFrame", "pandas"), ("PolarsDataFrame", "polars"), ("PyArrowTable", "pyarrow")],
        ids=["PandasDataFrame", "PolarsDataFrame", "PyArrowTable"],
    )
    def test_frames_give_json_native_preview(self, tmp_path: Path, framework: str, backend: str) -> None:
        pytest.importorskip(backend)
        result = _run(tmp_path, compute_frameworks=(framework,))

        table = _only_table(result)
        assert table["row_count"] == 5
        assert [column["name"] for column in table["columns"]] == FEATURES
        rows = table["preview"]["rows"]
        assert rows == [
            {"income": 10, "segment": "a"},
            {"income": 20, "segment": "b"},
            {"income": 30, "segment": "a"},
            {"income": 40, "segment": "b"},
            {"income": 50, "segment": "a"},
        ]
        native = (int, str, float, bool, type(None))
        assert all(type(value) in native for row in rows for value in row.values())
        json.dumps(result, allow_nan=False)
        _assert_no_paths(result, tmp_path)


class TestFilters:
    def test_filter_reduces_row_count_and_preview(self, tmp_path: Path) -> None:
        request = {"features": FEATURES, "filters": [{"feature": "segment", "op": "eq", "value": "a"}]}

        table = _only_table(_run(tmp_path, request))

        assert table["row_count"] == 3
        assert [row["segment"] for row in table["preview"]["rows"]] == ["a", "a", "a"]
        assert [row["income"] for row in table["preview"]["rows"]] == [10, 30, 50]


class TestJsonSafety:
    def test_non_json_values_are_made_json_safe(self, tmp_path: Path) -> None:
        result = _run(tmp_path, {"features": TYPED})

        table = _only_table(result)
        assert [column["name"] for column in table["columns"]] == TYPED
        assert all(column["classification"] == "internal" for column in table["columns"])
        assert all(column["in_preview"] for column in table["columns"])
        rows = table["preview"]["rows"]
        assert len(rows) == 5
        for income, row in zip([10, 20, 30, 40, 50], rows):
            assert set(row) == set(TYPED)
            assert row[TYPED_PREFIX + "decimal"] == str(Decimal(income) / Decimal(4))
            assert row[TYPED_PREFIX + "datetime"] == (datetime(2024, 1, 1, 12, 30) + timedelta(days=income)).isoformat()
            assert row[TYPED_PREFIX + "date"] == (date(2024, 1, 1) + timedelta(days=income)).isoformat()
            expected_nan = None if income >= 30 else income + 0.5
            assert row[TYPED_PREFIX + "nan"] == expected_nan
        json.dumps(result, allow_nan=False)

    def test_no_value_is_a_float_nan(self, tmp_path: Path) -> None:
        rows = _only_table(_run(tmp_path, {"features": TYPED}))["preview"]["rows"]

        values = [value for row in rows for value in row.values()]
        assert not any(isinstance(value, float) and math.isnan(value) for value in values)
        assert all(value is None or isinstance(value, (str, int, float, bool)) for value in values)


class TestFull:
    FULL_KEYS = {"row_count", "columns", "rows"}

    def _full(self, tmp_path: Path, request: Mapping[str, Any] = REQUEST, **kwargs: Any) -> dict[str, Any]:
        planner = _planner(_settings(tmp_path, **kwargs))
        plan_id = _plan_id(planner, request)
        _write_sales(tmp_path / "sales.csv", rows=20)
        result: dict[str, Any] = Runner(planner).run(plan_id, full=True)
        return result

    def _full_table(self, result: Mapping[str, Any]) -> dict[str, Any]:
        tables = _success(result)["tables"]
        assert len(tables) == 1
        table: dict[str, Any] = tables[0]
        assert set(table) == self.FULL_KEYS
        return table

    def test_full_returns_every_row_and_column_including_withheld(self, tmp_path: Path) -> None:
        table = self._full_table(self._full(tmp_path, max_preview_rows=3))

        assert table["row_count"] == 20
        assert len(table["rows"]) == 20
        assert all(set(row) == set(FEATURES) for row in table["rows"])
        assert [row["customer_id"] for row in table["rows"]] == CUSTOMER_IDS
        assert [row["income"] for row in table["rows"]] == [10 * i for i in range(1, 21)]

    def test_full_columns_carry_name_and_classification_only(self, tmp_path: Path) -> None:
        table = self._full_table(self._full(tmp_path))

        assert table["columns"] == [
            {"name": "customer_id", "classification": "personal"},
            {"name": "income", "classification": "internal"},
            {"name": "segment", "classification": "public"},
        ]

    def test_full_includes_unclassified_columns(self, tmp_path: Path) -> None:
        table = self._full_table(self._full(tmp_path, classification={}))

        assert {column["classification"] for column in table["columns"]} == {"unclassified"}
        assert all(set(row) == set(FEATURES) for row in table["rows"])

    def test_full_values_are_json_safe(self, tmp_path: Path) -> None:
        result = self._full(tmp_path, {"features": TYPED}, classification={"income": "internal"})

        rows = self._full_table(result)["rows"]
        assert len(rows) == 20
        assert rows[0][TYPED_PREFIX + "decimal"] == str(Decimal(10) / Decimal(4))
        assert rows[-1][TYPED_PREFIX + "nan"] is None
        json.dumps(result, allow_nan=False)

    def test_max_rows_still_applies_under_full(self, tmp_path: Path) -> None:
        _error(self._full(tmp_path, max_rows=10), "limit_exceeded")

    def test_default_run_is_unchanged(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path, max_preview_rows=2))
        plan_id = _plan_id(planner)

        explicit = Runner(planner).run(plan_id, full=False)

        assert set(_only_table(explicit)) == TABLE_KEYS
        assert len(_only_table(explicit)["preview"]["rows"]) == 2


class TestLock:
    def test_run_holds_the_lock_while_mloda_runs_and_releases_it(self, tmp_path: Path, lock_probes: list[bool]) -> None:
        _success(_run(tmp_path, {"features": [PROBE]}))

        assert lock_probes
        assert not any(lock_probes)
        assert _lock_free_elsewhere()


class TestResultsAreJson:
    @pytest.mark.parametrize(
        "request_value",
        [
            REQUEST,
            {"features": ["customer_id", PROBE]},
            {"features": TYPED},
            {"features": FEATURES, "filters": [{"feature": "segment", "op": "eq", "value": "b"}]},
        ],
    )
    def test_success_results_are_strict_json(self, tmp_path: Path, request_value: dict[str, Any]) -> None:
        result = _success(_run(tmp_path, request_value))

        json.dumps(result, allow_nan=False)
        _assert_no_paths(result, tmp_path)

    @pytest.mark.parametrize("plan_id", ["garbage", None, "mpl1.a.b"])
    def test_error_results_are_well_formed(self, tmp_path: Path, plan_id: Any) -> None:
        result = Runner(_planner(_settings(tmp_path))).run(plan_id)

        assert is_error(result)
        assert result["error"]["code"] in ERROR_CODES
        json.dumps(result, allow_nan=False)
