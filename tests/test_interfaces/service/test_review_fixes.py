"""Review fixes: filters on withheld columns are refused, max_rows covers plugin reads and result tables.

Catalog.describe resolves under the service lock and the catalog's principal.
"""

import getpass
import json
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.verified_context import current_verified_context
from mloda.interfaces.cli.main import build_services
from mloda.interfaces.service.catalog import Catalog
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.errors import is_error
from mloda.interfaces.service.lock import MLODA_LOCK
from mloda.interfaces.service.planner import Planner
from mloda.interfaces.service.runner import Runner
from mloda.provider import BaseInputData, DataCreator, FeatureSet, SourceMatch
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from tests.test_interfaces.source_groups import FILE_SOURCE_GROUPS

# Planning and running resolve real FeatureGroups and read source files; a cold cache can exceed 10s.
pytestmark = pytest.mark.timeout(30)

CLOCK_START = 1_800_000_000.0
KEY = b"m" * 32
CUSTOMER_IDS = [f"C9M9{index:03d}" for index in range(1, 6)]
CLASSIFICATION = {"customer_id": "personal", "income": "internal", "segment": "public"}
DOUBLE_SUFFIX = "__mcp9_double"
MADE = "mcp9_made"
MADE_ROWS = 7
FLAG = "mcp9_flag"
DESCRIBE_PROBE = "mcp9_describe_probe"
CREATOR = "ReviewCreatorMcp9"
FILTER_DETAIL_KEYS = {"path", "problem", "message"}
FLAG_CALLS: list[str] = []
DESCRIBE_RECORDS: list[tuple[bool, str | None]] = []


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


class ReviewDoubleMcp9(FeatureGroup):
    """Doubles the input feature named by the prefix before the suffix."""

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
        return {Feature(str(feature_name)[: -len(DOUBLE_SUFFIX)])}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = str(features.get_name_of_one_feature())
        return {name: [value * 2 for value in data[name[: -len(DOUBLE_SUFFIX)]]]}


class ReviewCreatorMcp9(FeatureGroup):
    """Creates its own rows; no configured source backs it."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({MADE})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {MADE: list(range(1, MADE_ROWS + 1))}


class _FlagFeatureGroup(FeatureGroup):
    """Derives the flag feature from one input column; subclasses pick the column."""

    INPUT: ClassVar[str] = ""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return bool(cls.INPUT) and str(feature_name) == FLAG

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(type(self).INPUT)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        FLAG_CALLS.append(cls.__name__)
        return {FLAG: [f"{value}!" for value in data[cls.INPUT]]}


class ReviewFlagFromSegmentMcp9(_FlagFeatureGroup):
    """Flag derived from the public segment column."""

    INPUT: ClassVar[str] = "segment"


class ReviewFlagFromCustomerMcp9(_FlagFeatureGroup):
    """Flag derived from the personal customer_id column."""

    INPUT: ClassVar[str] = "customer_id"


class ReviewDescribeProbeMcp9(FeatureGroup):
    """Records, when armed, whether describe holds the service lock and which principal is current."""

    ARMED: ClassVar[bool] = False

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) != DESCRIBE_PROBE:
            return False
        if cls.ARMED:
            context = current_verified_context()
            DESCRIBE_RECORDS.append((_lock_free_elsewhere(), None if context is None else context.principal))
        return True

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {DESCRIBE_PROBE: [1]}


def _base(*extra: type[FeatureGroup]) -> PluginCollector:
    groups: set[type[FeatureGroup]] = {*FILE_SOURCE_GROUPS, ReviewDoubleMcp9, ReviewCreatorMcp9}
    return PluginCollector.enabled_feature_groups(groups | set(extra))


def _write_sales(path: Path) -> Path:
    lines = ["customer_id,income,segment"]
    lines += [f"{CUSTOMER_IDS[i - 1]},{10 * i},{'a' if i % 2 else 'b'}" for i in range(1, 6)]
    path.write_text("\n".join(lines) + "\n")
    return path


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    path = _write_sales(tmp_path / "sales.csv")
    values: dict[str, Any] = {
        "mode": "execute",
        "sources": {"sales": SourceConfig(alias="sales", files=(str(path),))},
        "classification": CLASSIFICATION,
    }
    values.update(kwargs)
    return Settings(**values)


def _planner(settings: Settings, base: PluginCollector | None = None) -> Planner:
    collector = base if base is not None else _base()
    return Planner(
        settings,
        Catalog.build(settings, collector),
        signing_key=KEY,
        principal="alice",
        plugin_collector=collector,
        clock=lambda: CLOCK_START,
    )


def _explain(tmp_path: Path, request: Any, **kwargs: Any) -> dict[str, Any]:
    result: dict[str, Any] = _planner(_settings(tmp_path, **kwargs)).explain(request)
    return result


def _error(result: Mapping[str, Any], code: str) -> dict[str, Any]:
    assert is_error(result), result
    error: dict[str, Any] = result["error"]
    assert error["code"] == code, error
    json.dumps(result)
    return error


def _success(result: Mapping[str, Any]) -> dict[str, Any]:
    assert not is_error(result), result
    return dict(result)


def _assert_no_paths(result: Mapping[str, Any], tmp_path: Path) -> None:
    dumped = json.dumps(result)
    assert str(tmp_path) not in dumped
    assert "sales.csv" not in dumped


def _assert_withheld_filters(result: Mapping[str, Any], indexes: list[int]) -> None:
    error = _error(result, "policy_denied")
    details = error["details"]
    assert [detail["path"] for detail in details] == [f"/filters/{index}" for index in indexes], details
    for detail in details:
        assert set(detail) == FILTER_DETAIL_KEYS, detail
        assert detail["problem"] == "filter_on_withheld_column"
        assert isinstance(detail["message"], str)
        assert detail["message"]


def _creator_entry() -> dict[str, Any]:
    return {"alias": None, "feature_group": CREATOR, "rows": None, "rows_exact": False}


def _by_json(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda entry: json.dumps(entry, sort_keys=True))


@pytest.fixture
def flag_calls() -> Iterator[list[str]]:
    FLAG_CALLS.clear()
    yield FLAG_CALLS
    FLAG_CALLS.clear()


@pytest.fixture
def describe_records() -> Iterator[list[tuple[bool, str | None]]]:
    DESCRIBE_RECORDS.clear()
    ReviewDescribeProbeMcp9.ARMED = True
    yield DESCRIBE_RECORDS
    ReviewDescribeProbeMcp9.ARMED = False
    DESCRIBE_RECORDS.clear()


class TestFilterOnWithheldColumn:
    @pytest.mark.parametrize(
        "condition",
        [
            {"op": "eq", "value": CUSTOMER_IDS[2]},
            {"op": "in", "values": [CUSTOMER_IDS[2], CUSTOMER_IDS[3]]},
        ],
        ids=["eq", "in"],
    )
    def test_filter_on_personal_column_is_policy_denied(self, tmp_path: Path, condition: dict[str, Any]) -> None:
        request = {"features": ["segment"], "filters": [{"feature": "customer_id", **condition}]}

        result = _explain(tmp_path, request)

        _assert_withheld_filters(result, [0])
        dumped = json.dumps(result)
        assert all(customer_id not in dumped for customer_id in CUSTOMER_IDS)
        _assert_no_paths(result, tmp_path)

    def test_filter_on_requested_withheld_column_is_policy_denied(self, tmp_path: Path) -> None:
        request = {
            "features": ["customer_id", "segment"],
            "filters": [{"feature": "customer_id", "op": "eq", "value": CUSTOMER_IDS[0]}],
        }

        _assert_withheld_filters(_explain(tmp_path, request), [0])

    def test_range_filter_on_withheld_numeric_column_is_policy_denied(self, tmp_path: Path) -> None:
        classification = {**CLASSIFICATION, "income": "confidential"}
        request = {"features": ["segment"], "filters": [{"feature": "income", "op": "range", "min": 25, "max": 35}]}

        _assert_withheld_filters(_explain(tmp_path, request, classification=classification), [0])

    def test_filter_on_unclassified_column_is_policy_denied(self, tmp_path: Path) -> None:
        classification = {"customer_id": "personal", "income": "internal"}
        request = {"features": ["income"], "filters": [{"feature": "segment", "op": "eq", "value": "a"}]}

        _assert_withheld_filters(_explain(tmp_path, request, classification=classification), [0])

    def test_one_detail_per_offending_filter(self, tmp_path: Path) -> None:
        request = {
            "features": ["segment"],
            "filters": [
                {"feature": "customer_id", "op": "eq", "value": CUSTOMER_IDS[1]},
                {"feature": "segment", "op": "eq", "value": "a"},
                {"feature": "customer_id", "op": "in", "values": [CUSTOMER_IDS[4]]},
            ],
        }

        result = _explain(tmp_path, request)

        _assert_withheld_filters(result, [0, 2])
        dumped = json.dumps(result)
        assert CUSTOMER_IDS[1] not in dumped
        assert CUSTOMER_IDS[4] not in dumped

    def test_filter_on_withheld_derived_feature_is_policy_denied(self, tmp_path: Path) -> None:
        derived = "customer_id" + DOUBLE_SUFFIX
        secret = CUSTOMER_IDS[2] * 2
        request = {"features": ["segment", derived], "filters": [{"feature": derived, "op": "eq", "value": secret}]}

        result = _explain(tmp_path, request)

        _assert_withheld_filters(result, [0])
        assert secret not in json.dumps(result)

    def test_filter_on_derived_feature_with_allowed_explicit_classification_is_allowed(self, tmp_path: Path) -> None:
        derived = "customer_id" + DOUBLE_SUFFIX
        classification = {**CLASSIFICATION, derived: "internal"}
        request = {"features": ["segment", derived], "filters": [{"feature": derived, "op": "eq", "value": "x"}]}

        result = _success(_explain(tmp_path, request, classification=classification))

        assert result["policy"] == {"verdict": "allowed"}

    def test_filter_on_column_preview_allows_is_allowed(self, tmp_path: Path) -> None:
        request = {"features": ["segment"], "filters": [{"feature": "customer_id", "op": "eq", "value": "x"}]}

        _success(_explain(tmp_path, request, preview_allows=("public", "internal", "personal")))

    def test_filter_on_allowed_column_is_allowed(self, tmp_path: Path) -> None:
        request = {"features": ["customer_id"], "filters": [{"feature": "segment", "op": "eq", "value": "a"}]}

        _success(_explain(tmp_path, request))

    def test_unmatched_filter_is_reported_before_the_withheld_check(self, tmp_path: Path) -> None:
        request = {"features": ["segment"], "filters": [{"feature": "nope_mcp9", "op": "eq", "value": 1}]}

        error = _error(_explain(tmp_path, request), "validation_failed")

        assert [(detail["path"], detail["problem"]) for detail in error["details"]] == [
            ("/filters/0", "unmatched_filter")
        ]


class TestRunRechecksWithheldFilters:
    def test_run_refuses_a_filter_whose_column_became_withheld_without_running(
        self, tmp_path: Path, flag_calls: list[str]
    ) -> None:
        # Settings are identical, so the plan id verifies; only the producer of the flag changes,
        # which makes the filtered flag inherit customer_id's personal classification at run time.
        settings = _settings(tmp_path)
        request = {"features": [FLAG], "filters": [{"feature": FLAG, "op": "eq", "value": f"{CUSTOMER_IDS[2]}!"}]}
        explained = _success(_planner(settings, _base(ReviewFlagFromSegmentMcp9)).explain(request))
        running = _planner(settings, _base(ReviewFlagFromCustomerMcp9))

        result = Runner(running).run(explained["plan_id"])

        _assert_withheld_filters(result, [0])
        assert flag_calls == []
        assert "tables" not in result
        assert CUSTOMER_IDS[2] not in json.dumps(result)

    def test_run_of_allowed_filter_on_derived_feature_completes(self, tmp_path: Path, flag_calls: list[str]) -> None:
        settings = _settings(tmp_path)
        planner = _planner(settings, _base(ReviewFlagFromSegmentMcp9))
        request = {"features": [FLAG], "filters": [{"feature": FLAG, "op": "eq", "value": "a!"}]}

        result = _success(Runner(planner).run(_success(planner.explain(request))["plan_id"]))

        (table,) = result["tables"]
        assert table["row_count"] == 3
        assert flag_calls == ["ReviewFlagFromSegmentMcp9"]


class TestPluginReadsInEstimate:
    def test_plugin_root_is_listed_as_uncountable(self, tmp_path: Path) -> None:
        estimate = _success(_explain(tmp_path, {"features": [MADE]}))["estimate"]

        assert estimate["sources"] == [_creator_entry()]
        assert estimate["rows_read"] is None

    def test_plugin_root_beside_a_file_source(self, tmp_path: Path) -> None:
        estimate = _success(_explain(tmp_path, {"features": [MADE, "segment"]}))["estimate"]

        assert _by_json(estimate["sources"]) == _by_json(
            [{"alias": "sales", "rows": 5, "rows_exact": True}, _creator_entry()]
        )
        assert estimate["rows_read"] is None

    def test_root_of_a_derived_feature_is_listed(self, tmp_path: Path) -> None:
        estimate = _success(_explain(tmp_path, {"features": [MADE + DOUBLE_SUFFIX]}))["estimate"]

        assert estimate["sources"] == [_creator_entry()]
        assert estimate["rows_read"] is None

    def test_file_only_estimate_keeps_its_shape(self, tmp_path: Path) -> None:
        estimate = _success(_explain(tmp_path, {"features": ["segment"]}))["estimate"]

        assert estimate["sources"] == [{"alias": "sales", "rows": 5, "rows_exact": True}]
        assert estimate["rows_read"] == 5

    def test_plugin_read_without_max_rows_runs(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path))

        result = _success(Runner(planner).run(_success(planner.explain({"features": [MADE]}))["plan_id"]))

        (table,) = result["tables"]
        assert table["row_count"] == MADE_ROWS


class TestPluginReadsUnderMaxRows:
    def test_plugin_read_with_max_rows_is_limit_exceeded(self, tmp_path: Path) -> None:
        result = _explain(tmp_path, {"features": [MADE]}, max_rows=1000)

        error = _error(result, "limit_exceeded")
        assert error["details"] == [
            {"source": None, "feature_group": CREATOR, "reason": "uncountable", "max_rows": 1000}
        ]
        _assert_no_paths(result, tmp_path)

    def test_plugin_root_of_derived_feature_with_max_rows_is_limit_exceeded(self, tmp_path: Path) -> None:
        error = _error(_explain(tmp_path, {"features": [MADE + DOUBLE_SUFFIX]}, max_rows=1000), "limit_exceeded")

        assert error["details"] == [
            {"source": None, "feature_group": CREATOR, "reason": "uncountable", "max_rows": 1000}
        ]

    def test_file_and_plugin_reads_are_both_reported(self, tmp_path: Path) -> None:
        result = _explain(tmp_path, {"features": [MADE, "segment"]}, max_rows=3)

        details = _error(result, "limit_exceeded")["details"]
        assert {"source": "sales", "reason": "over_limit", "rows": 5, "max_rows": 3} in details
        assert {"source": None, "feature_group": CREATOR, "reason": "uncountable", "max_rows": 3} in details
        assert len(details) == 2
        _assert_no_paths(result, tmp_path)


class TestResultTablesUnderMaxRows:
    def test_result_table_over_max_rows_is_limit_exceeded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A reader whose count under-reports (stale metadata, a count that skips rows a read keeps) passes
        # the planner's source check; the result tables are the last line of defence.
        def under_reporting(cls: type[CsvFG], match: SourceMatch, framework: type[ComputeFramework]) -> int:
            return 2

        monkeypatch.setattr(CsvFG, "count_rows", classmethod(under_reporting))
        planner = _planner(_settings(tmp_path, max_rows=3))
        explained = _success(planner.explain({"features": ["customer_id", "income", "segment"]}))
        assert explained["estimate"]["rows_read"] == 2

        result = Runner(planner).run(explained["plan_id"])

        error = _error(result, "limit_exceeded")
        assert {"reason": "result_over_limit", "rows": 5, "max_rows": 3}.items() <= error["details"][0].items()
        assert "tables" not in result
        dumped = json.dumps(result)
        assert all(customer_id not in dumped for customer_id in CUSTOMER_IDS)
        _assert_no_paths(result, tmp_path)

    def test_result_table_at_max_rows_completes(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path, max_rows=5))

        result = _success(Runner(planner).run(_success(planner.explain({"features": ["segment"]}))["plan_id"]))

        assert result["tables"][0]["row_count"] == 5


class TestDescribeUnderLockAndPrincipal:
    def test_describe_holds_the_lock_under_the_catalog_principal(
        self, tmp_path: Path, describe_records: list[tuple[bool, str | None]]
    ) -> None:
        catalog = Catalog.build(_settings(tmp_path), _base(ReviewDescribeProbeMcp9), principal="carol")

        result = catalog.describe(DESCRIBE_PROBE)

        assert result["feature_group"] == "ReviewDescribeProbeMcp9", result
        assert describe_records
        assert all(not free for free, _ in describe_records), describe_records
        assert all(principal == "carol" for _, principal in describe_records), describe_records
        assert _lock_free_elsewhere()

    def test_describe_without_principal_still_holds_the_lock(
        self, tmp_path: Path, describe_records: list[tuple[bool, str | None]]
    ) -> None:
        catalog = Catalog.build(_settings(tmp_path), _base(ReviewDescribeProbeMcp9))

        catalog.describe(DESCRIBE_PROBE)

        assert describe_records
        assert all(not free for free, _ in describe_records), describe_records
        assert all(principal is None for _, principal in describe_records), describe_records

    def test_build_services_gives_the_catalog_the_planner_principal(
        self, tmp_path: Path, describe_records: list[tuple[bool, str | None]]
    ) -> None:
        catalog, _, _ = build_services(
            _settings(tmp_path), env={}, plugin_collector=_base(ReviewDescribeProbeMcp9), clock=lambda: CLOCK_START
        )

        catalog.describe(DESCRIBE_PROBE)

        assert describe_records
        assert all(principal == getpass.getuser() for _, principal in describe_records), describe_records
        assert all(not free for free, _ in describe_records), describe_records
