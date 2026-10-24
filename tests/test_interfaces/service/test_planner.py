"""Planner.explain: request checks, policy, row limits, and a path-free plan with lineage, estimate and egress."""

import dataclasses
import json
import logging
import threading
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.interfaces.service.catalog import Catalog
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.errors import ERROR_CODES, is_error
from mloda.interfaces.service.lock import MLODA_LOCK
from mloda.interfaces.service.planner import (
    MAX_PLAN_ID_LENGTH,
    PLAN_TOKEN_PREFIX,
    PlannedRequest,
    Planner,
    _inline_alias,
)
from mloda.provider import ApiDataFG, DataCreatorFeatureGroup, FeatureSet, ResolutionDiagnosis, SourceMatch
from mloda.user import PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG
from tests.helpers.plugin_stubs import StubFeatureGroup
from tests.test_interfaces.source_groups import FILE_SOURCE_GROUPS, PLAN_RAISE, PLAN_RAISE_SECRET, PlanRaiseMcp5

# Planning resolves real FeatureGroups and reads source files; a cold cache under xdist load can exceed 10s.
pytestmark = pytest.mark.timeout(30)

SECRET = "s3://user:pw@host/x"  # nosec B105
DENIED = "PlannerDeniedMcp4"
DOUBLE_SUFFIX = "__mcp4_double"
DERIVED = "income" + DOUBLE_SUFFIX
IN_SUFFIX = "_mcp4_in"
CLOCK_START = 1_800_000_000.0
KEY = b"k" * 32
STEP_KEYS = {
    "step",
    "kind",
    "features",
    "feature_group",
    "compute_framework",
    "compute_framework_reason",
    "specialized_from",
    "source",
    "inputs",
}
SUCCESS_KEYS = {"plan_id", "expires_at", "normalized_request", "steps", "lineage", "estimate", "egress", "policy"}
DENIED_CALLS: list[str] = []
LOCK_PROBES: list[bool] = []


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


class PlannerDoubleMcp4(FeatureGroup):
    """Doubles a numeric input feature named by the prefix before the suffix."""

    PROBE_LOCK: ClassVar[bool] = False

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        name = str(feature_name)
        return name.endswith(DOUBLE_SUFFIX) and len(name) > len(DOUBLE_SUFFIX)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        if type(self).PROBE_LOCK:
            LOCK_PROBES.append(_lock_free_elsewhere())
        return {Feature(str(feature_name)[: -len(DOUBLE_SUFFIX)])}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = str(features.get_name_of_one_feature())
        source = name[: -len(DOUBLE_SUFFIX)]
        return {name: [value * 2 for value in data[source]]}


class PlannerDeniedMcp4(StubFeatureGroup):
    """Restricted payroll data."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"mcp4_salary"})
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = {PythonDictFramework}
    ARMED: ClassVar[bool] = False

    @classmethod
    def _record(cls, hook: str) -> None:
        if cls.ARMED:
            DENIED_CALLS.append(hook)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        cls._record("match_feature_group_criteria")
        return str(feature_name) in cls.MATCHED_NAMES

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        cls._record("compute_framework_rule")
        return cls.FRAMEWORK_RULE

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        type(self)._record("input_features")
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls._record("calculate_feature")
        return {"mcp4_salary": [1]}


class PlannerInFeaturesMcp4(FeatureGroup):
    """Consumes the in_features declared on a feature whose name ends with the suffix."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name).endswith(IN_SUFFIX)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return set(options.get_in_features())

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {str(features.get_name_of_one_feature()): [1]}


def _base() -> PluginCollector:
    return PluginCollector.enabled_feature_groups(
        {
            *FILE_SOURCE_GROUPS,
            PlannerDoubleMcp4,
            PlannerDeniedMcp4,
            PlannerInFeaturesMcp4,
            PlanRaiseMcp5,
            ApiDataFG,
        }
    )


def _in_features_cases(column: str) -> list[tuple[dict[str, Any], str]]:
    name = "out" + IN_SUFFIX
    mid = "mid" + IN_SUFFIX
    return [
        ({"name": name, "in_features": ["income", column]}, "/features/0/in_features/1"),
        ({"name": name, "options": {"in_features": column}}, "/features/0/options/in_features"),
        ({"name": name, "group_options": {"in_features": [column]}}, "/features/0/group_options/in_features/0"),
        ({"name": name, "context_options": {"in_features": [column]}}, "/features/0/context_options/in_features/0"),
        ({"name": name, "options": {"in_features": {"name": column}}}, "/features/0/options/in_features/name"),
        (
            {"name": name, "options": {"in_features": {"name": mid, "in_features": column}}},
            "/features/0/options/in_features/in_features",
        ),
        (
            {"name": name, "options": {"in_features": {"name": mid, "options": {"in_features": [column]}}}},
            "/features/0/options/in_features/options/in_features/0",
        ),
    ]


def _write_sales(tmp_path: Path, with_denied_column: bool = False) -> Path:
    path = tmp_path / "sales.csv"
    if with_denied_column:
        path.write_text("customer_id,income,mcp4_salary\n1,10,5\n2,20,6\n")
    else:
        path.write_text("customer_id,income,segment\n1,10,a\n2,20,b\n3,30,a\n4,40,b\n5,50,a\n")
    return path


def _write_orders(tmp_path: Path) -> Path:
    path = tmp_path / "orders.csv"
    path.write_text("order_id,amount\n1,3\n2,4\n")
    return path


def _write_events(tmp_path: Path) -> Path:
    path = tmp_path / "events.json"
    path.write_text('{"event_id": 1, "kind": "x"}\n{"event_id": 2, "kind": "y"}\n{"event_id": 3, "kind": "x"}\n')
    return path


def _source(alias: str, path: Path) -> SourceConfig:
    return SourceConfig(alias=alias, files=(str(path),))


def _settings(tmp_path: Path, sources: Mapping[str, SourceConfig] | None = None, **kwargs: Any) -> Settings:
    if sources is None:
        sources = {"sales": _source("sales", _write_sales(tmp_path))}
    return Settings(deny_feature_groups=(DENIED,), sources=sources, **kwargs)


def _planner(settings: Settings) -> Planner:
    base = _base()
    catalog = Catalog.build(settings, base)
    return Planner(
        settings, catalog, signing_key=KEY, principal="alice", plugin_collector=base, clock=lambda: CLOCK_START
    )


def _explain(tmp_path: Path, request: Any, **kwargs: Any) -> dict[str, Any]:
    result: dict[str, Any] = _planner(_settings(tmp_path, **kwargs)).explain(request)
    return result


def _error(result: dict[str, Any], code: str) -> dict[str, Any]:
    assert is_error(result), result
    error: dict[str, Any] = result["error"]
    assert error["code"] == code, error
    return error


def _success(result: dict[str, Any]) -> dict[str, Any]:
    assert not is_error(result), result
    return result


def _has_detail(error: Mapping[str, Any], **expected: Any) -> bool:
    return any(all(detail.get(key) == value for key, value in expected.items()) for detail in error["details"])


def _assert_no_paths(result: Mapping[str, Any], tmp_path: Path, *file_names: str) -> None:
    dumped = json.dumps(result)
    assert str(tmp_path) not in dumped
    assert tmp_path.name not in dumped
    for name in file_names:
        assert name not in dumped


@pytest.fixture
def denied_calls() -> Iterator[list[str]]:
    DENIED_CALLS.clear()
    PlannerDeniedMcp4.ARMED = True
    yield DENIED_CALLS
    PlannerDeniedMcp4.ARMED = False
    DENIED_CALLS.clear()


@pytest.fixture
def lock_probes() -> Iterator[list[bool]]:
    LOCK_PROBES.clear()
    PlannerDoubleMcp4.PROBE_LOCK = True
    yield LOCK_PROBES
    PlannerDoubleMcp4.PROBE_LOCK = False
    LOCK_PROBES.clear()


class TestPlanningFailure:
    def test_planning_exception_is_an_internal_error_without_its_text(self, tmp_path: Path) -> None:
        result = _explain(tmp_path, {"features": [PLAN_RAISE]})

        error = _error(result, "internal")
        assert "ValueError" in error["message"]
        assert "planning" in error["message"].lower()
        assert PLAN_RAISE_SECRET not in json.dumps(result)

    def test_planning_failure_is_logged_once_without_the_message(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="mloda.interfaces.service"):
            result = _explain(tmp_path, {"features": [PLAN_RAISE]})

        audit_id = _error(result, "internal")["audit_id"]
        records = [r for r in caplog.records if r.name == "mloda.interfaces.service" and r.levelno == logging.ERROR]
        assert len(records) == 1
        message = records[0].getMessage()
        assert audit_id in message
        assert PLAN_RAISE_SECRET not in message
        assert records[0].exc_info is None


class TestRequestChecks:
    def test_unknown_top_level_key_is_validation_failed(self, tmp_path: Path) -> None:
        error = _error(_explain(tmp_path, {"features": ["income"], "optons": {}}), "validation_failed")
        assert _has_detail(error, path="/optons")

    @pytest.mark.parametrize("request_value", [None, "income", ["income"], 3])
    def test_non_object_request_is_validation_failed(self, tmp_path: Path, request_value: Any) -> None:
        error = _error(_explain(tmp_path, request_value), "validation_failed")
        assert _has_detail(error, path="")

    def test_unknown_restricted_source_is_validation_failed(self, tmp_path: Path) -> None:
        result = _explain(tmp_path, {"features": ["income"], "restrict_sources": ["sals"]})
        error = _error(result, "validation_failed")
        assert _has_detail(error, path="/restrict_sources/0", problem="unknown_source", suggestion="sales")

    @pytest.mark.parametrize("key", ["CsvFG", "JsonFG", "ParquetFG", RESERVED_READER_OPTION_KEY])
    def test_data_access_option_is_policy_denied(self, tmp_path: Path, key: str) -> None:
        request = {"features": ["customer_id", {"name": "income", "options": {key: SECRET}}]}

        result = _explain(tmp_path, request)

        error = _error(result, "policy_denied")
        assert [detail["path"] for detail in error["details"]] == [f"/features/1/options/{key}"]
        assert SECRET not in json.dumps(result)

    def test_csv_pointer_at_unconfigured_file_is_policy_denied(self, tmp_path: Path) -> None:
        _write_sales(tmp_path)
        other = tmp_path / "other_mcp4.csv"
        other.write_text("income\n987654321\n")
        request = {"features": [{"name": "income", "options": {"CsvFG": str(other)}}]}

        result = _planner(_settings(tmp_path)).explain(request)

        error = _error(result, "policy_denied")
        assert [detail["path"] for detail in error["details"]] == ["/features/0/options/CsvFG"]
        assert "987654321" not in json.dumps(result)
        _assert_no_paths(result, tmp_path, "other_mcp4.csv")

    def test_semantic_feature_config_error_is_validation_failed(self, tmp_path: Path) -> None:
        request = {"features": [{"name": "income", "options": {"x": 1}, "group_options": {"y": 1}}]}

        error = _error(_explain(tmp_path, request), "validation_failed")

        assert _has_detail(error, path="/features/0", problem="invalid_feature_config")


class TestUnresolved:
    def test_unresolved_name_has_path_and_suggestion_from_source_columns(self, tmp_path: Path) -> None:
        result = _explain(tmp_path, {"features": ["customer_id", "incme"]})

        error = _error(result, "validation_failed")
        assert _has_detail(error, path="/features/1", problem="unresolved", suggestion="income")
        _assert_no_paths(result, tmp_path, "sales.csv")

    def test_unresolved_name_without_close_match_has_no_suggestion(self, tmp_path: Path) -> None:
        result = _explain(tmp_path, {"features": ["qqqzzzwww_mcp4"]})

        error = _error(result, "validation_failed")
        details = [detail for detail in error["details"] if detail.get("path") == "/features/0"]
        assert len(details) == 1
        assert details[0]["problem"] == "unresolved"
        assert "suggestion" not in details[0]
        _assert_no_paths(result, tmp_path, "sales.csv")


class TestDiagnosisWithoutFeatureName:
    def test_scope_at_inaccessible_source_group_is_validation_failed(self, tmp_path: Path) -> None:
        request = {"features": [{"name": "income", "feature_group": "SqliteFG"}]}

        result = _explain(tmp_path, request)

        error = _error(result, "validation_failed")
        assert isinstance(error["message"], str)
        assert error["message"]
        assert "not accessible in this run" not in json.dumps(result)
        _assert_no_paths(result, tmp_path, "sales.csv")

    def test_unnamed_diagnosis_is_validation_failed_with_generic_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings(tmp_path)
        leaked = str(tmp_path / "sales.csv")
        core_message = f"options point at CsvFG, but CsvFG is not accessible in this run: {leaked}."
        diagnosis = ResolutionDiagnosis(records=[], complete=False, message=core_message)
        monkeypatch.setattr(mloda, "diagnose", lambda *args, **kwargs: diagnosis)

        result = _planner(settings).explain({"features": ["income"]})

        error = _error(result, "validation_failed")
        assert isinstance(error["message"], str)
        assert error["message"]
        assert core_message not in json.dumps(result)
        _assert_no_paths(result, tmp_path, "sales.csv")


class TestPolicy:
    def test_filter_on_denied_column_is_refused_before_planning(self, tmp_path: Path, denied_calls: list[str]) -> None:
        sources = {"sales": _source("sales", _write_sales(tmp_path, with_denied_column=True))}
        request = {"features": ["income"], "filters": [{"feature": "mcp4_salary", "op": "eq", "value": 5}]}

        result = _explain(
            tmp_path,
            request,
            sources=sources,
            classification={"mcp4_salary": "public", "income": "public"},
            preview_allows=("public",),
        )

        error = _error(result, "policy_denied")
        assert _has_detail(error, path="/filters/0/feature")
        assert denied_calls == []

    @pytest.mark.parametrize(
        ("feature", "pointer"),
        [("mcp4_salary~0", "/features/0"), ("mcp4_salary~0" + DOUBLE_SUFFIX, None)],
    )
    def test_column_index_suffix_does_not_bypass_policy(
        self, tmp_path: Path, denied_calls: list[str], feature: str, pointer: str | None
    ) -> None:
        sources = {"sales": _source("sales", _write_sales(tmp_path, with_denied_column=True))}

        result = _explain(tmp_path, {"features": [feature]}, sources=sources)

        error = _error(result, "policy_denied")
        if pointer is not None:
            assert _has_detail(error, path=pointer)
        assert denied_calls == []

    def test_unresolved_name_never_suggests_denied_column(self, tmp_path: Path) -> None:
        sources = {"sales": _source("sales", _write_sales(tmp_path, with_denied_column=True))}

        result = _explain(tmp_path, {"features": ["mcp4_salarx"]}, sources=sources)

        error = _error(result, "validation_failed")
        assert all(detail.get("suggestion") != "mcp4_salary" for detail in error["details"])
        assert "mcp4_salary" not in json.dumps(result)

    def test_denied_feature_is_policy_denied_without_running_its_code(
        self, tmp_path: Path, denied_calls: list[str]
    ) -> None:
        result = _explain(tmp_path, {"features": ["mcp4_salary"]})

        error = _error(result, "policy_denied")
        assert "alternatives" not in error
        assert "deny_feature_groups" not in json.dumps(result)
        assert denied_calls == []

    def test_denied_feature_carries_alternatives(self, tmp_path: Path, denied_calls: list[str]) -> None:
        result = _explain(tmp_path, {"features": ["mcp4_salary"]}, alternatives={"mcp4_salary": ("income",)})

        error = _error(result, "policy_denied")
        assert error["alternatives"] == ["income"]
        assert denied_calls == []

    @pytest.mark.parametrize(("feature", "pointer"), _in_features_cases("mcp4_salary"))
    def test_denied_in_features_source_is_policy_denied(
        self, tmp_path: Path, denied_calls: list[str], feature: dict[str, Any], pointer: str
    ) -> None:
        sources = {"sales": _source("sales", _write_sales(tmp_path, with_denied_column=True))}

        result = _explain(tmp_path, {"features": [feature]}, sources=sources)

        error = _error(result, "policy_denied")
        assert _has_detail(error, path=pointer)
        assert denied_calls == []

    @pytest.mark.parametrize("feature", [feature for feature, _ in _in_features_cases("income")])
    def test_allowed_in_features_source_plans(self, tmp_path: Path, feature: dict[str, Any]) -> None:
        sources = {"sales": _source("sales", _write_sales(tmp_path, with_denied_column=True))}

        _success(_explain(tmp_path, {"features": [feature]}, sources=sources))

    def test_derived_name_over_denied_column_is_policy_denied(self, tmp_path: Path, denied_calls: list[str]) -> None:
        sources = {"sales": _source("sales", _write_sales(tmp_path, with_denied_column=True))}

        result = _explain(tmp_path, {"features": ["mcp4_salary" + DOUBLE_SUFFIX]}, sources=sources)

        error = _error(result, "policy_denied")
        assert _has_detail(error, path="/features", problem="policy_denied")
        assert denied_calls == []
        _assert_no_paths(result, tmp_path, "sales.csv")

    def test_derived_name_over_denied_column_carries_alternatives(self, tmp_path: Path) -> None:
        sources = {"sales": _source("sales", _write_sales(tmp_path, with_denied_column=True))}

        result = _explain(
            tmp_path,
            {"features": ["mcp4_salary" + DOUBLE_SUFFIX]},
            sources=sources,
            alternatives={"mcp4_salary": ("income",)},
        )

        assert _error(result, "policy_denied")["alternatives"] == ["income"]

    def test_allowed_request_never_calls_into_denied(self, tmp_path: Path, denied_calls: list[str]) -> None:
        _success(_explain(tmp_path, {"features": ["customer_id", DERIVED]}))
        assert denied_calls == []


class TestFilters:
    def test_unmatched_filter_is_validation_failed(self, tmp_path: Path) -> None:
        request = {
            "features": ["customer_id"],
            "filters": [
                {"feature": "segment", "op": "eq", "value": "a"},
                {"feature": "nope_mcp4", "op": "eq", "value": 1},
            ],
        }

        result = _explain(tmp_path, request)

        error = _error(result, "validation_failed")
        assert _has_detail(error, path="/filters/1", problem="unmatched_filter")
        assert not _has_detail(error, path="/filters/0")
        _assert_no_paths(result, tmp_path, "sales.csv")

    def test_matched_filter_on_unrequested_column_is_allowed(self, tmp_path: Path) -> None:
        request = {"features": ["customer_id"], "filters": [{"feature": "segment", "op": "in", "values": ["a"]}]}

        # A filter column must be previewable; an unclassified one is refused (filter_on_withheld_column).
        result = _success(_explain(tmp_path, request, classification={"segment": "public"}))

        assert result["policy"] == {"verdict": "allowed"}
        assert result["normalized_request"] == request


class TestSuccess:
    def test_result_keys(self, tmp_path: Path) -> None:
        result = _success(_explain(tmp_path, {"features": ["customer_id", "income"]}))
        assert set(result) == SUCCESS_KEYS

    def test_plan_id_and_expiry(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)

        result = _success(_planner(settings).explain({"features": ["customer_id", "income"]}))

        assert isinstance(result["plan_id"], str)
        assert result["plan_id"].startswith(PLAN_TOKEN_PREFIX + ".")
        expires_at = result["expires_at"]
        assert isinstance(expires_at, str)
        assert expires_at.endswith("Z")
        parsed = datetime.fromisoformat(expires_at)
        assert parsed.timestamp() == CLOCK_START + settings.plan_ttl_seconds

    def test_normalized_request_and_policy(self, tmp_path: Path) -> None:
        result = _success(_explain(tmp_path, {"features": ["customer_id", "income"]}))

        assert result["normalized_request"] == {"features": ["customer_id", "income"]}
        assert result["policy"] == {"verdict": "allowed"}

    def test_read_step(self, tmp_path: Path) -> None:
        result = _success(_explain(tmp_path, {"features": ["customer_id", "income"]}))

        (step,) = result["steps"]
        assert set(step) == STEP_KEYS
        assert step["step"] == 1
        assert step["kind"] == "compute"
        assert sorted(step["features"]) == ["customer_id", "income"]
        assert step["feature_group"] == "CsvFG"
        assert step["compute_framework"] == "PythonDictFramework"
        assert step["compute_framework_reason"] == "only allowed framework"
        assert step["specialized_from"] == []
        assert step["source"] == "sales"
        assert step["inputs"] == []

    def test_lineage_of_source_columns(self, tmp_path: Path) -> None:
        result = _success(_explain(tmp_path, {"features": ["customer_id", "income"]}))
        assert result["lineage"] == {"customer_id": ["sales.customer_id"], "income": ["sales.income"]}

    def test_estimate_of_countable_source(self, tmp_path: Path) -> None:
        estimate = _success(_explain(tmp_path, {"features": ["customer_id", "income"]}))["estimate"]

        assert set(estimate) == {"sources", "rows_read", "rows_returned", "rows_returned_note"}
        assert estimate["sources"] == [{"alias": "sales", "rows": 5, "rows_exact": True}]
        assert estimate["rows_read"] == 5
        assert estimate["rows_returned"] is None
        assert isinstance(estimate["rows_returned_note"], str)
        assert estimate["rows_returned_note"]

    def test_no_paths_anywhere(self, tmp_path: Path) -> None:
        result = _success(_explain(tmp_path, {"features": ["customer_id", "income"]}))
        _assert_no_paths(result, tmp_path, "sales.csv")

    def test_only_sources_the_plan_reads_are_estimated(self, tmp_path: Path) -> None:
        sources = {
            "sales": _source("sales", _write_sales(tmp_path)),
            "other": _source("other", _write_orders(tmp_path)),
        }

        result = _success(_explain(tmp_path, {"features": ["customer_id"]}, sources=sources))

        assert [entry["alias"] for entry in result["estimate"]["sources"]] == ["sales"]
        _assert_no_paths(result, tmp_path, "sales.csv", "orders.csv")


class TestDerived:
    def test_steps_in_dependency_order(self, tmp_path: Path) -> None:
        result = _success(_explain(tmp_path, {"features": ["customer_id", DERIVED]}))

        steps = result["steps"]
        assert [step["step"] for step in steps] == [1, 2]
        assert all(set(step) == STEP_KEYS for step in steps)
        read, derived = steps
        assert read["feature_group"] == "CsvFG"
        assert read["source"] == "sales"
        assert sorted(read["features"]) == ["customer_id", "income"]
        assert derived["kind"] == "compute"
        assert derived["feature_group"] == "PlannerDoubleMcp4"
        assert derived["features"] == [DERIVED]
        assert derived["compute_framework"] == "PythonDictFramework"
        assert derived["source"] is None
        assert derived["inputs"] == ["income"]

    def test_lineage_traces_derived_feature_to_source_column(self, tmp_path: Path) -> None:
        result = _success(_explain(tmp_path, {"features": ["customer_id", DERIVED]}))
        assert result["lineage"] == {"customer_id": ["sales.customer_id"], DERIVED: ["sales.income"]}

    def test_estimate_counts_source_of_derived_feature(self, tmp_path: Path) -> None:
        result = _success(_explain(tmp_path, {"features": [DERIVED]}))
        assert result["estimate"]["sources"] == [{"alias": "sales", "rows": 5, "rows_exact": True}]


class TestEgress:
    def test_classified_columns(self, tmp_path: Path) -> None:
        classification = {"customer_id": "personal", "income": "internal"}

        result = _success(_explain(tmp_path, {"features": ["customer_id", "income"]}, classification=classification))

        assert result["egress"] == {
            "preview_columns": ["income"],
            "withheld": [{"column": "customer_id", "reason": "classification: personal"}],
        }

    def test_unclassified_column_is_withheld(self, tmp_path: Path) -> None:
        classification = {"customer_id": "personal", "income": "internal"}
        request = {"features": ["customer_id", "income", "segment"]}

        egress = _success(_explain(tmp_path, request, classification=classification))["egress"]

        assert egress["preview_columns"] == ["income"]
        assert egress["withheld"] == [
            {"column": "customer_id", "reason": "classification: personal"},
            {"column": "segment", "reason": "classification: unclassified"},
        ]

    def test_preview_allows_is_honoured(self, tmp_path: Path) -> None:
        classification = {"customer_id": "public", "income": "internal"}

        result = _explain(
            tmp_path, {"features": ["customer_id", "income"]}, classification=classification, preview_allows=("public",)
        )

        assert _success(result)["egress"] == {
            "preview_columns": ["customer_id"],
            "withheld": [{"column": "income", "reason": "classification: internal"}],
        }

    def test_derived_feature_inherits_input_classification(self, tmp_path: Path) -> None:
        classification = {"customer_id": "internal", "income": "personal"}

        result = _explain(tmp_path, {"features": ["customer_id", DERIVED]}, classification=classification)

        assert _success(result)["egress"] == {
            "preview_columns": ["customer_id"],
            "withheld": [{"column": DERIVED, "reason": "classification: personal"}],
        }

    def test_explicit_classification_of_derived_feature_wins(self, tmp_path: Path) -> None:
        classification = {"customer_id": "internal", "income": "personal", DERIVED: "internal"}

        result = _explain(tmp_path, {"features": ["customer_id", DERIVED]}, classification=classification)

        assert _success(result)["egress"] == {"preview_columns": ["customer_id", DERIVED], "withheld": []}

    def test_keep_index_index_column_is_listed_and_withheld_by_classification(self, tmp_path: Path) -> None:
        classification = {"customer_id": "personal", "income": "internal"}
        request = {"features": [{"name": "income", "keep_index": True, "index": ["customer_id"]}]}

        egress = _success(_explain(tmp_path, request, classification=classification))["egress"]

        assert egress["preview_columns"] == ["income"]
        assert egress["withheld"] == [{"column": "customer_id", "reason": "classification: personal"}]

    def test_keep_index_index_column_is_previewed_when_allowed(self, tmp_path: Path) -> None:
        classification = {"customer_id": "internal", "income": "internal"}
        request = {"features": [{"name": "income", "keep_index": True, "index": ["customer_id"]}]}

        egress = _success(_explain(tmp_path, request, classification=classification))["egress"]

        assert sorted(egress["preview_columns"]) == ["customer_id", "income"]
        assert egress["withheld"] == []

    def test_keep_index_lineage_traces_the_kept_column(self, tmp_path: Path) -> None:
        request = {"features": [{"name": "income", "keep_index": True, "index": ["customer_id"]}]}

        lineage = _success(_explain(tmp_path, request))["lineage"]

        assert lineage["customer_id"] == ["sales.customer_id"]

    def test_index_without_keep_index_is_not_in_egress(self, tmp_path: Path) -> None:
        classification = {"customer_id": "internal", "income": "internal"}
        request = {"features": [{"name": "income", "keep_index": False, "index": ["customer_id"]}]}

        egress = _success(_explain(tmp_path, request, classification=classification))["egress"]

        assert egress == {"preview_columns": ["income"], "withheld": []}

    def test_index_without_keep_index_is_still_deny_checked(self, tmp_path: Path) -> None:
        request = {"features": [{"name": "income", "index": ["mcp4_salary"]}]}

        error = _error(_explain(tmp_path, request), "policy_denied")

        assert _has_detail(error, path="/features/0/index/0")

    def test_key_column_also_requested_appears_once_in_egress(self, tmp_path: Path) -> None:
        classification = {"customer_id": "internal", "income": "internal"}
        request = {"features": ["customer_id", {"name": "income", "keep_index": True, "index": ["customer_id"]}]}

        egress = _success(_explain(tmp_path, request, classification=classification))["egress"]

        assert egress == {"preview_columns": ["customer_id", "income"], "withheld": []}


class TestRowLimits:
    def test_source_over_max_rows_is_limit_exceeded(self, tmp_path: Path) -> None:
        result = _explain(tmp_path, {"features": ["customer_id", "income"]}, max_rows=3)

        error = _error(result, "limit_exceeded")
        assert "sales" in error["message"]
        assert _has_detail(error, source="sales", rows=5, max_rows=3)
        _assert_no_paths(result, tmp_path, "sales.csv")

    def test_source_at_max_rows_is_allowed(self, tmp_path: Path) -> None:
        _success(_explain(tmp_path, {"features": ["customer_id", "income"]}, max_rows=5))

    def test_uncountable_source_with_max_rows_is_limit_exceeded(self, tmp_path: Path) -> None:
        sources = {"events": _source("events", _write_events(tmp_path))}

        result = _explain(tmp_path, {"features": ["event_id"]}, sources=sources, max_rows=1000)

        error = _error(result, "limit_exceeded")
        assert _has_detail(error, source="events", reason="uncountable")
        _assert_no_paths(result, tmp_path, "events.json")

    def test_uncountable_source_without_max_rows_is_estimated_as_unknown(self, tmp_path: Path) -> None:
        sources = {"events": _source("events", _write_events(tmp_path))}

        result = _success(_explain(tmp_path, {"features": ["event_id"]}, sources=sources))

        assert result["estimate"]["sources"] == [{"alias": "events", "rows": None, "rows_exact": False}]
        assert result["estimate"]["rows_read"] is None
        assert result["steps"][0]["source"] == "events"
        assert result["steps"][0]["feature_group"] == "JsonFG"
        assert result["lineage"] == {"event_id": ["events.event_id"]}
        _assert_no_paths(result, tmp_path, "events.json")

    def test_json_reader_does_not_count_rows(self, tmp_path: Path) -> None:
        path = str(_write_events(tmp_path))
        assert JsonFG.count_rows(SourceMatch(source=path, access=path), PythonDictFramework) is None

    def test_csv_reader_counts_rows_exactly_on_python_dict(self, tmp_path: Path) -> None:
        path = str(_write_sales(tmp_path))
        assert CsvFG.count_rows(SourceMatch(source=path, access=path), PythonDictFramework) == 5


class PlannerApiSourceMcp4(DataCreatorFeatureGroup):
    """A generated root: its compute step has no source."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mcp4_api_value"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mcp4_api_value": [1, 2]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}


def _prepare_csv(path: Path) -> Any:
    return mloda.prepare(
        [Feature("customer_id")],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({CsvFG}),
        data_access_collection=DataAccessCollection(files={"sales_handle": str(path)}),
    )


def _csv_step(session: Any) -> Any:
    return next(s for s in session.resolved_plan() if s.step_kind == "compute" and s.feature_group is CsvFG)


class TestSessionSourceRoutes:
    def test_session_routes_equal_the_group_calls_for_a_csv_step(self, tmp_path: Path) -> None:
        path = str(_write_sales(tmp_path))
        session = _prepare_csv(Path(path))
        step = _csv_step(session)
        match = SourceMatch(source=path, access=path)

        assert session.count_rows(step) == 5
        assert session.count_rows(step) == CsvFG.count_rows(match, step.compute_framework)
        assert session.describe_columns(step) == CsvFG.describe_columns(match)

    @pytest.mark.parametrize("route", ["describe_columns", "count_rows"])
    def test_a_step_without_a_step_uuid_raises(self, tmp_path: Path, route: str) -> None:
        session = _prepare_csv(_write_sales(tmp_path))
        joinlike = dataclasses.replace(_csv_step(session), step_kind="join", step_uuid=None)

        with pytest.raises(ValueError):
            getattr(session, route)(joinlike)

    @pytest.mark.parametrize("route", ["describe_columns", "count_rows"])
    def test_a_step_from_another_session_raises(self, tmp_path: Path, route: str) -> None:
        path = _write_sales(tmp_path)
        step = _csv_step(_prepare_csv(path))
        other = _prepare_csv(path)

        with pytest.raises(ValueError):
            getattr(other, route)(step)

    @pytest.mark.parametrize("route", ["describe_columns", "count_rows"])
    def test_a_step_with_an_unknown_uuid_raises(self, tmp_path: Path, route: str) -> None:
        session = _prepare_csv(_write_sales(tmp_path))
        stranger = dataclasses.replace(_csv_step(session), step_uuid=uuid4())

        with pytest.raises(ValueError):
            getattr(session, route)(stranger)

    @pytest.mark.parametrize("route", ["describe_columns", "count_rows"])
    def test_a_data_creator_step_raises_naming_the_group(self, route: str) -> None:
        session = mloda.prepare(
            [Feature("mcp4_api_value")],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({PlannerApiSourceMcp4}),
        )
        step = next(s for s in session.resolved_plan() if s.feature_group is PlannerApiSourceMcp4)

        with pytest.raises(ValueError, match="PlannerApiSourceMcp4"):
            getattr(session, route)(step)


class TestRestrictSources:
    def _sources(self, tmp_path: Path) -> dict[str, SourceConfig]:
        return {
            "sales": _source("sales", _write_sales(tmp_path)),
            "other": _source("other", _write_orders(tmp_path)),
        }

    def test_feature_outside_restricted_sources_is_unresolved(self, tmp_path: Path) -> None:
        request = {"features": ["income"], "restrict_sources": ["other"]}

        result = _explain(tmp_path, request, sources=self._sources(tmp_path))

        error = _error(result, "validation_failed")
        assert _has_detail(error, path="/features/0", problem="unresolved")
        _assert_no_paths(result, tmp_path, "sales.csv", "orders.csv")

    def test_feature_inside_restricted_sources_resolves(self, tmp_path: Path) -> None:
        request = {"features": ["order_id"], "restrict_sources": ["other"]}

        result = _success(_explain(tmp_path, request, sources=self._sources(tmp_path)))

        assert result["steps"][0]["source"] == "other"
        assert result["lineage"] == {"order_id": ["other.order_id"]}
        assert result["estimate"]["sources"] == [{"alias": "other", "rows": 2, "rows_exact": True}]


class TestDeterminismAndShape:
    def test_same_request_gives_same_plan(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path, classification={"income": "personal"}))
        request = {"features": ["customer_id", DERIVED]}

        first = _success(planner.explain(request))
        second = _success(planner.explain(request))

        for key in ("steps", "lineage", "estimate", "egress", "normalized_request", "expires_at"):
            assert first[key] == second[key], key

    @pytest.mark.parametrize(
        "request_value",
        [
            {"features": ["customer_id", "income"]},
            {"features": ["customer_id", DERIVED]},
            {"features": ["incme"]},
            {"features": ["mcp4_salary"]},
            {"features": [{"name": "income", "options": {RESERVED_READER_OPTION_KEY: SECRET}}]},
            {"features": ["customer_id"], "filters": [{"feature": "nope_mcp4", "op": "eq", "value": 1}]},
            {"features": []},
            None,
        ],
    )
    def test_every_result_is_json_and_errors_are_well_formed(self, tmp_path: Path, request_value: Any) -> None:
        result = _explain(tmp_path, request_value)

        assert isinstance(result, dict)
        json.dumps(result)
        if is_error(result):
            assert result["error"]["code"] in ERROR_CODES
            assert result["error"]["audit_id"].startswith("aud_")
        _assert_no_paths(result, tmp_path, "sales.csv")


INLINE_REQUEST: dict[str, Any] = {"features": ["token"], "data": {"Tokens": {"token": ["007", "", None]}}}


class TestInlineData:
    def test_data_without_opt_in_is_policy_denied(self, tmp_path: Path) -> None:
        error = _error(_explain(tmp_path, INLINE_REQUEST), "policy_denied")
        assert _has_detail(error, path="/data")
        assert '"007"' not in json.dumps(error)

    def test_inline_request_is_planned(self, tmp_path: Path) -> None:
        result = _success(_explain(tmp_path, INLINE_REQUEST, inline_data=True))

        assert result["lineage"] == {"token": ["Tokens.token"]}
        assert result["estimate"]["sources"] == [{"alias": "Tokens", "rows": 3, "rows_exact": True}]
        assert result["estimate"]["rows_read"] == 3
        assert result["normalized_request"]["data"] == INLINE_REQUEST["data"]

    def test_only_api_data_fg_steps_are_inline_sets(self, tmp_path: Path) -> None:
        planned = _planner(_settings(tmp_path, inline_data=True)).plan(INLINE_REQUEST)
        assert isinstance(planned, PlannedRequest)
        step = next(s for s in planned.steps if s.feature_group is not None and issubclass(s.feature_group, ApiDataFG))
        foreign = dataclasses.replace(step, feature_group=CsvFG)
        assert (foreign.data_access_identity or "").startswith("api_data:")

        assert _inline_alias(step, planned.parsed.inline_data) == "Tokens"
        assert _inline_alias(foreign, planned.parsed.inline_data) is None

    def test_max_rows_applies_to_inline_rows(self, tmp_path: Path) -> None:
        result = _explain(tmp_path, INLINE_REQUEST, inline_data=True, max_rows=2)

        error = _error(result, "limit_exceeded")
        assert _has_detail(error, source="Tokens", reason="over_limit", rows=3, max_rows=2)
        assert '"007"' not in json.dumps(result)

    def test_set_named_like_a_configured_source_is_validation_failed(self, tmp_path: Path) -> None:
        request = {"features": ["token"], "data": {"sales": {"token": ["a"]}}}

        result = _explain(tmp_path, request, inline_data=True)

        error = _error(result, "validation_failed")
        assert _has_detail(error, path="/data/sales")

    def test_inline_column_shadowing_a_source_column_is_refused(self, tmp_path: Path) -> None:
        # An inline column cannot shadow a source column.
        request = {"features": ["income"], "data": {"Inline": {"income": ["a", "b"]}}}

        result = _explain(tmp_path, request, inline_data=True)

        error = _error(result, "validation_failed")
        assert _has_detail(error, problem="unresolved")
        assert '"a"' not in json.dumps(result)

    def test_api_data_fg_option_stays_policy_denied(self, tmp_path: Path) -> None:
        # Already denied before inline planning exists; this pins the guard.
        key = "ApiDataFG"
        request = {"features": [{"name": "token", "options": {key: {"Tokens": {"token": ["007"]}}}}]}

        result = _explain(tmp_path, request, inline_data=True)

        error = _error(result, "policy_denied")
        assert [detail["path"] for detail in error["details"]] == [f"/features/0/options/{key}"]
        assert '"007"' not in json.dumps(result)

    def test_oversized_plan_id_is_limit_exceeded_without_values(self, tmp_path: Path) -> None:
        # Inline JSON under the limit, but the signed base64 plan id is over it.
        marker = "BIGVALUE_mcp_inline"
        value = marker * (int(MAX_PLAN_ID_LENGTH * 0.85) // len(marker))
        data = {"Tokens": {"token": [value]}}
        assert len(json.dumps(data)) <= MAX_PLAN_ID_LENGTH
        request = {"features": ["token"], "data": data}

        result = _explain(tmp_path, request, inline_data=True)

        error = _error(result, "limit_exceeded")
        assert "plan_id" not in result
        assert marker not in json.dumps(result)
        assert error["message"] == "The request is too large for a plan id."

    def test_oversized_inline_json_is_refused_before_planning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def recorder(*args: Any, **kwargs: Any) -> Any:
            calls.append("diagnose")
            raise AssertionError("planning must not run")

        monkeypatch.setattr(mloda, "diagnose", recorder)
        marker = "BIGVALUE_mcp_inline"
        values = [f"{marker}{index:06d}" for index in range(MAX_PLAN_ID_LENGTH // 8)]
        data = {"Tokens": {"token": values}}
        assert len(json.dumps(data)) > MAX_PLAN_ID_LENGTH
        request = {"features": ["token"], "data": data}

        result = _explain(tmp_path, request, inline_data=True)

        _error(result, "limit_exceeded")
        assert calls == []
        assert marker not in json.dumps(result)

    def test_set_name_with_slash_is_reported_at_escaped_pointer(self, tmp_path: Path) -> None:
        sources = {"a/b": _source("a/b", _write_sales(tmp_path))}
        request = {"features": ["token"], "data": {"a/b": {"token": ["x"]}}}

        result = _explain(tmp_path, request, sources=sources, inline_data=True)

        error = _error(result, "validation_failed")
        assert _has_detail(error, path="/data/a~1b")

    def test_denying_api_data_fg_switches_inline_off(self, tmp_path: Path) -> None:
        planner = _planner(
            Settings(
                deny_feature_groups=(DENIED, "ApiDataFG"),
                sources={"sales": _source("sales", _write_sales(tmp_path))},
                inline_data=True,
            )
        )

        request = {"features": ["token"], "data": {"Tokens": {"token": ["SECRETVAL"]}}}

        result = planner.explain(request)

        error = _error(result, "validation_failed")
        assert "plan_id" not in result
        assert "SECRETVAL" not in json.dumps(error)


class TestLock:
    def test_lock_is_reentrant(self) -> None:
        assert isinstance(MLODA_LOCK, type(threading.RLock()))

    def test_planning_holds_the_lock_and_releases_it(self, tmp_path: Path, lock_probes: list[bool]) -> None:
        _success(_explain(tmp_path, {"features": [DERIVED]}))

        assert lock_probes
        assert not any(lock_probes)
        assert _lock_free_elsewhere()


class TestKeepIndex:
    def test_keep_index_without_index_is_validation_failed(self, tmp_path: Path) -> None:
        request = {"features": ["customer_id", {"name": "income", "keep_index": True}]}

        error = _error(_explain(tmp_path, request), "validation_failed")

        assert _has_detail(error, path="/features/1")

    def test_denied_column_in_index_is_policy_denied_at_its_pointer(
        self, tmp_path: Path, denied_calls: list[str]
    ) -> None:
        request = {"features": [{"name": "income", "keep_index": True, "index": ["customer_id", "mcp4_salary"]}]}

        result = _explain(tmp_path, request)

        error = _error(result, "policy_denied")
        assert _has_detail(error, path="/features/0/index/1")
        assert denied_calls == []

    def test_index_column_no_group_provides_is_an_internal_error_without_tables(self, tmp_path: Path) -> None:
        request = {"features": [{"name": "income", "keep_index": True, "index": ["no_such_col_mcp4"]}]}

        result = _explain(tmp_path, request)

        _error(result, "internal")
        assert "tables" not in result
        json.dumps(result)
        _assert_no_paths(result, tmp_path, "sales.csv")
